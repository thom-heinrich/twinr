# CHANGELOG: 2026-03-28
# BUG-1: Fixed a real race in recent-remote-audio buffering. The old deque was
# mutated by the capture thread while the websocket callback thread could join or
# iterate it, which could surface inconsistent audio windows or raise at exactly
# the moment wake / transcript callbacks needed buffered audio.
# BUG-2: Fixed transport backpressure coupling. Audio capture and websocket sends
# are now decoupled with a bounded queue and dedicated sender thread, preventing
# slow or reconnecting networks from stalling arecord reads and causing wake /
# transcript misses.
# BUG-3: Fixed high and variable capture latency from ALSA defaults. arecord now
# uses explicit period / buffer sizing aligned to chunk_ms instead of inheriting
# the default large buffering behavior. Capture startup is also non-blocking and
# self-healing on busy / transiently missing devices.
# BUG-4: Detect "arecord stays alive but never yields a first frame" capture
# stalls and route them through bounded XVF3800 recovery instead of hanging.
# BUG-5: Detect mid-stream "arecord stays alive but stops yielding bytes" stalls
# too, so the Pi cannot sit forever on one healthy websocket with zero live
# microphone audio after the first few frames.
# SEC-1: BREAKING: ws:// is now rejected for non-loopback orchestrator endpoints
# unless explicitly allowed via config.voice_orchestrator_allow_insecure_ws.
# Raw senior-home microphone audio and shared secret transport must not traverse
# a real network without TLS unless the current transport-only LAN bridge was
# explicitly attested by the operator.
# SEC-2: Sanitized externally sourced values before passing them to emit(...),
# reducing practical log / event-stream injection risk from remote error strings
# or backend labels.
# IMP-1: Added jittered exponential reconnect and capture-restart backoff with
# continuous self-healing instead of a one-shot fatal capture worker.
# IMP-2: Added bounded transport-queue telemetry, drop-oldest latency control,
# queue high-water marks, and richer forensics around reconnect / backpressure.
# IMP-3: Preserved drop-in public API while adding optional tuning hooks via
# config/env for queue depth, reconnect windows, ALSA buffer periods, and
# insecure-development overrides.

"""Own the edge-side streaming voice bridge for the orchestrator path.

This module keeps `realtime_runner.py` orchestration-focused by handling the
bounded `arecord` lifecycle, websocket transport, and decision dispatch for the
Alexa-like server-backed voice path. It does not run turns itself; it only
forwards server-side wake/transcript-commit/follow-up-close/barge-in decisions
back into the realtime loop.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import ipaddress
import math
import os
import queue
import random
import select
import shutil
import subprocess
from threading import Event, Lock, RLock, Thread, current_thread
import time
from typing import Callable, SupportsFloat, SupportsIndex, SupportsInt, cast
from urllib.parse import urlparse
from uuid import uuid4

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.workflows.forensics import WorkflowForensics
from twinr.agent.workflows.respeaker_duplex_keepalive import build_respeaker_duplex_keepalive
from twinr.agent.workflows.voice_frame_speech import EdgeVoiceFrameSpeechAnnotator
from twinr.hardware.audio import resolve_capture_device
from twinr.hardware.audio_env import build_audio_subprocess_env_for_mode
from twinr.hardware.respeaker_capture_recovery import recover_stalled_respeaker_capture
from twinr.ops.streaming_memory_probe import StreamingMemoryProbe
from twinr.orchestrator.voice_client import OrchestratorVoiceWebSocketClient
from twinr.orchestrator.voice_contracts import (
    OrchestratorVoiceAudioFrame,
    OrchestratorVoiceBargeInInterruptEvent,
    OrchestratorVoiceErrorEvent,
    OrchestratorVoiceFollowUpClosedEvent,
    OrchestratorVoiceHelloRequest,
    OrchestratorVoiceIdentityProfilesEvent,
    OrchestratorVoiceReadyEvent,
    OrchestratorVoiceRuntimeStateEvent,
    OrchestratorVoiceTranscriptCommittedEvent,
    OrchestratorVoiceWakeConfirmedEvent,
)
from twinr.orchestrator.voice_activation import VoiceActivationMatch
from twinr.orchestrator.voice_forensics import VoiceFrameTelemetryBucket


@dataclass(slots=True)
class _BufferedRemoteAudioFrame:
    """Keep a short rolling copy of streamed remote-listening audio."""

    pcm_bytes: bytes
    duration_ms: int


@dataclass(slots=True)
class _QueuedAudioFrame:
    """One captured PCM frame waiting for websocket transport."""

    pcm_bytes: bytes
    captured_at_monotonic: float


class _CaptureStreamStalledError(RuntimeError):
    """Signal that arecord stayed alive but stopped producing readable bytes."""

    def __init__(self, *, stalled_ms: int) -> None:
        self.stalled_ms = max(0, int(stalled_ms))
        super().__init__(
            f"Voice orchestrator capture stalled after {self.stalled_ms} ms without bytes"
        )


class EdgeVoiceOrchestrator:
    """Stream bounded PCM frames to the remote voice orchestrator service."""

    _MAX_STDERR_BYTES = 8_192
    _IDENTITY_PROFILE_UNSUPPORTED_ERROR = "Unsupported message type."
    _IDENTITY_PROFILE_ERROR_WINDOW_S = 2.0
    _EMIT_VALUE_MAX_CHARS = 240
    _DEFAULT_CAPTURE_BUFFER_PERIODS = 4
    _DEFAULT_TRANSPORT_QUEUE_MS = 400
    _DEFAULT_WS_RECONNECT_BASE_S = 0.5
    _DEFAULT_WS_RECONNECT_MAX_S = 8.0
    _DEFAULT_CAPTURE_RESTART_BASE_S = 0.5
    _DEFAULT_CAPTURE_RESTART_MAX_S = 6.0
    _QUEUE_POLL_TIMEOUT_S = 0.1

    def __init__(
        self,
        config: TwinrConfig,
        *,
        emit: Callable[[str], None],
        playback_coordinator=None,
        on_voice_activation: Callable[[VoiceActivationMatch], bool],
        on_transcript_committed: Callable[[str, str], bool],
        on_follow_up_closed: Callable[[str], None] | None = None,
        on_barge_in_interrupt: Callable[[], bool],
        on_recent_remote_audio: Callable[[bytes, str], object] | None = None,
        forensics: WorkflowForensics | None = None,
    ) -> None:
        self.config = config
        self.emit = emit
        self._on_voice_activation = on_voice_activation
        self._on_transcript_committed = on_transcript_committed
        self._on_follow_up_closed = on_follow_up_closed
        self._on_barge_in_interrupt = on_barge_in_interrupt
        self._on_recent_remote_audio = on_recent_remote_audio

        resolved_device = resolve_capture_device(
            config.voice_orchestrator_audio_device,
            config.proactive_audio_input_device,
            config.audio_input_device,
        )
        self._device = str(resolved_device or "").strip()
        if not self._device:
            raise ValueError("voice orchestrator capture device could not be resolved")

        self._sample_rate = int(config.audio_sample_rate)
        self._channels = int(config.audio_channels)
        self._chunk_ms = max(20, int(config.audio_chunk_ms))
        self._speech_threshold = max(0, int(config.audio_speech_threshold))
        self._frame_samples = max(160, int(round(self._sample_rate * (self._chunk_ms / 1000.0))))
        self._frame_bytes = max(320, self._frame_samples * self._channels * 2)

        self._select_timeout_s = self._coerce_positive_float(
            self._config_value("voice_orchestrator_select_timeout_s", "TWINR_VOICE_SELECT_TIMEOUT_S"),
            default=min(0.05, max(0.02, self._chunk_ms / 1000.0)),
        )
        self._transport_queue_ms = self._coerce_positive_int(
            self._config_value("voice_orchestrator_transport_queue_ms", "TWINR_VOICE_TRANSPORT_QUEUE_MS"),
            default=max(self._DEFAULT_TRANSPORT_QUEUE_MS, self._chunk_ms * 4),
            minimum=max(self._chunk_ms * 2, 100),
        )
        self._transport_queue_frames = max(4, int(math.ceil(self._transport_queue_ms / self._chunk_ms)))
        self._capture_buffer_periods = self._coerce_positive_int(
            self._config_value("voice_orchestrator_capture_buffer_periods", "TWINR_VOICE_CAPTURE_BUFFER_PERIODS"),
            default=self._DEFAULT_CAPTURE_BUFFER_PERIODS,
            minimum=2,
        )
        self._capture_period_frames = self._frame_samples
        self._capture_buffer_frames = max(
            self._capture_period_frames + 1,
            self._capture_period_frames * self._capture_buffer_periods,
        )
        self._first_frame_timeout_s = self._coerce_positive_float(
            self._config_value("voice_orchestrator_first_frame_timeout_s", "TWINR_VOICE_FIRST_FRAME_TIMEOUT_S"),
            default=max(
                1.0,
                (self._capture_buffer_frames / max(1, self._sample_rate)) * 4.0,
                self._select_timeout_s * 8.0,
            ),
        )
        self._ongoing_frame_timeout_s = self._coerce_positive_float(
            self._config_value(
                "voice_orchestrator_ongoing_frame_timeout_s",
                "TWINR_VOICE_ONGOING_FRAME_TIMEOUT_S",
            ),
            default=self._first_frame_timeout_s,
            minimum=max(self._select_timeout_s * 2.0, self._chunk_ms / 1000.0),
        )
        self._ws_reconnect_base_s = self._coerce_positive_float(
            self._config_value("voice_orchestrator_reconnect_base_s", "TWINR_VOICE_WS_RECONNECT_BASE_S"),
            default=self._DEFAULT_WS_RECONNECT_BASE_S,
        )
        self._ws_reconnect_max_s = self._coerce_positive_float(
            self._config_value("voice_orchestrator_reconnect_max_s", "TWINR_VOICE_WS_RECONNECT_MAX_S"),
            default=self._DEFAULT_WS_RECONNECT_MAX_S,
        )
        self._capture_restart_base_s = self._coerce_positive_float(
            self._config_value("voice_orchestrator_capture_restart_base_s", "TWINR_VOICE_CAPTURE_RESTART_BASE_S"),
            default=self._DEFAULT_CAPTURE_RESTART_BASE_S,
        )
        self._capture_restart_max_s = self._coerce_positive_float(
            self._config_value("voice_orchestrator_capture_restart_max_s", "TWINR_VOICE_CAPTURE_RESTART_MAX_S"),
            default=self._DEFAULT_CAPTURE_RESTART_MAX_S,
        )
        self._allow_insecure_ws = self._coerce_bool(
            getattr(config, "voice_orchestrator_allow_insecure_ws", None),
            default=False,
        )

        ws_url = str(config.voice_orchestrator_ws_url or "").strip()
        if not ws_url:
            raise ValueError("voice_orchestrator_ws_url must not be empty")
        self._ws_url = ws_url
        self._client = OrchestratorVoiceWebSocketClient(
            ws_url,
            shared_secret=config.voice_orchestrator_shared_secret,
            on_event=self._handle_server_event,
            require_tls=self._require_tls_for_url(ws_url),
        )

        self._session_id = f"voice-{uuid4().hex[:12]}"
        self._frame_speech_annotator = EdgeVoiceFrameSpeechAnnotator(
            sample_rate=self._sample_rate,
            channels=self._channels,
        )
        self._stop_event = Event()
        self._paused = Event()
        self._lifecycle_lock = RLock()
        self._state_lock = Lock()
        self._runtime_state_send_lock = RLock()
        self._transport_send_lock = RLock()
        self._thread: Thread | None = None
        self._sender_thread: Thread | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._sender_queue: queue.Queue[_QueuedAudioFrame | None] = queue.Queue(maxsize=self._transport_queue_frames)

        self._sequence = 0
        self._stderr_tail = bytearray()
        self._connected = False
        self._next_reconnect_at = 0.0
        self._ws_reconnect_failures = 0
        self._capture_restart_failures = 0

        self._last_runtime_state: OrchestratorVoiceRuntimeStateEvent | None = None
        self._last_identity_profiles: OrchestratorVoiceIdentityProfilesEvent | None = None
        self._identity_profiles_supported = True
        self._last_identity_profiles_sent_at = 0.0
        self._ready_backend: str | None = None

        self._forensics = forensics if isinstance(forensics, WorkflowForensics) and forensics.enabled else None
        self._trace_id = self._session_id
        self._sent_frame_bucket = VoiceFrameTelemetryBucket(
            chunk_ms=self._chunk_ms,
            speech_threshold=self._speech_threshold,
        )
        self._skipped_frame_count = 0
        self._queue_drop_count = 0
        self._queue_high_watermark = 0
        self._capture_memory_probe = StreamingMemoryProbe.from_config(
            config,
            label="voice_orchestrator.capture_loop",
            owner_label="voice_orchestrator.capture_loop",
            owner_detail="Voice orchestrator capture loop is actively reading audio frames.",
        )
        self._sender_memory_probe = StreamingMemoryProbe.from_config(
            config,
            label="voice_orchestrator.sender_loop",
            owner_label="voice_orchestrator.sender_loop",
            owner_detail="Voice orchestrator sender loop is actively transporting queued audio frames.",
        )

        self._recent_remote_audio_buffer_ms = max(
            4000,
            int(getattr(config, "voice_profile_passive_update_min_duration_ms", 2500) or 2500) + 1500,
        )
        self._recent_remote_frame_limit = max(1, int(math.ceil(self._recent_remote_audio_buffer_ms / self._chunk_ms)))
        self._recent_remote_frames_lock = Lock()
        self._recent_remote_frames: deque[_BufferedRemoteAudioFrame] = deque(maxlen=self._recent_remote_frame_limit)
        self._duplex_keepalive = build_respeaker_duplex_keepalive(
            config=config,
            capture_device=self._device,
            playback_coordinator=playback_coordinator,
            emit=self.emit,
        )
        if self._duplex_keepalive is not None and playback_coordinator is not None:
            playback_coordinator.add_activity_listener(
                self._duplex_keepalive.handle_playback_activity
            )

    @staticmethod
    def _record_memory_probe(
        probe: StreamingMemoryProbe,
        *,
        force: bool = False,
        owner_detail: str | None = None,
    ) -> None:
        try:
            probe.maybe_record(force=force, owner_detail=owner_detail)
        except Exception:
            return

    def _config_value(self, attr_name: str, env_name: str | None = None) -> object | None:
        value = getattr(self.config, attr_name, None)
        if value not in (None, ""):
            return value
        if env_name:
            env_value = os.getenv(env_name)
            if env_value not in (None, ""):
                return env_value
        return None

    @staticmethod
    def _coerce_positive_int(value: object | None, *, default: int, minimum: int = 1) -> int:
        try:
            candidate = value if value not in (None, "") else default
            parsed = int(cast(SupportsInt | SupportsIndex | str | bytes | bytearray, candidate))
        except (TypeError, ValueError):
            parsed = int(default)
        return max(int(minimum), parsed)

    @staticmethod
    def _coerce_positive_float(value: object | None, *, default: float, minimum: float = 0.001) -> float:
        try:
            candidate = value if value not in (None, "") else default
            parsed = float(
                cast(SupportsFloat | SupportsIndex | str | bytes | bytearray, candidate)
            )
        except (TypeError, ValueError):
            parsed = float(default)
        return max(float(minimum), parsed)

    @staticmethod
    def _coerce_bool(value: object | None, *, default: bool) -> bool:
        if value in (None, ""):
            return bool(default)
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return bool(default)

    def _require_tls_for_url(self, ws_url: str) -> bool:
        parsed = urlparse(ws_url)
        scheme = str(parsed.scheme or "").strip().lower()
        host = str(parsed.hostname or "").strip()

        if scheme not in {"ws", "wss"}:
            # BREAKING: explicit websocket schemes are now required to avoid
            # accidentally treating malformed or partial URLs as insecure network
            # endpoints.
            raise ValueError(f"Unsupported voice orchestrator websocket scheme: {scheme or 'missing'}")

        if scheme == "wss":
            return True

        if self._allow_insecure_ws or self._is_loopback_host(host):
            return False

        # BREAKING: non-loopback ws:// is rejected by default because the module
        # ships senior-home microphone audio plus an orchestrator shared secret.
        raise ValueError(
            "Insecure ws:// voice orchestrator endpoint rejected. Use wss:// or explicitly allow insecure ws for "
            "local development only."
        )

    @staticmethod
    def _is_loopback_host(host: str) -> bool:
        normalized = str(host or "").strip().lower().strip("[]")
        if normalized in {"localhost", "localhost.localdomain"}:
            return True
        try:
            return ipaddress.ip_address(normalized).is_loopback
        except ValueError:
            return False

    @classmethod
    def _sanitize_emit_value(cls, value: object) -> str:
        text = str(value if value is not None else "").strip()
        if not text:
            return "unknown"
        cleaned = "".join(ch if ch.isprintable() and ch not in "\r\n\t" else " " for ch in text)
        cleaned = " ".join(cleaned.split())
        if not cleaned:
            return "unknown"
        if len(cleaned) > cls._EMIT_VALUE_MAX_CHARS:
            return cleaned[: cls._EMIT_VALUE_MAX_CHARS - 1].rstrip() + "…"
        return cleaned

    def _emit_status(self, key: str, value: object) -> None:
        self.emit(f"{key}={self._sanitize_emit_value(value)}")

    def _trace_event(
        self,
        msg: str,
        *,
        kind: str,
        details: dict[str, object] | None = None,
        level: str = "INFO",
        kpi: dict[str, object] | None = None,
    ) -> None:
        tracer = self._forensics
        if not isinstance(tracer, WorkflowForensics):
            return
        tracer.event(
            kind=kind,
            msg=msg,
            details={
                "session_id": self._session_id,
                "device": self._device,
                "sample_rate": self._sample_rate,
                "channels": self._channels,
                "chunk_ms": self._chunk_ms,
                "transport_queue_frames": self._transport_queue_frames,
                "transport_queue_ms": self._transport_queue_ms,
                **(details or {}),
            },
            trace_id=self._trace_id,
            level=level,
            kpi=kpi,
            loc_skip=2,
        )

    def _flush_sent_frame_bucket(self) -> None:
        """Persist one bounded edge-side transport window."""

        if not self._sent_frame_bucket.has_data():
            return
        details = self._sent_frame_bucket.flush_details()
        details["queue_drop_count"] = self._queue_drop_count
        details["queue_high_watermark"] = self._queue_high_watermark
        self._trace_event(
            "voice_edge_frame_window_sent",
            kind="io",
            details=details,
        )

    def open(self) -> "EdgeVoiceOrchestrator":
        """Connect the websocket and start the capture / sender workers."""

        with self._lifecycle_lock:
            capture_thread = self._thread
            sender_thread = self._sender_thread
            if (
                capture_thread is not None
                and capture_thread.is_alive()
                and sender_thread is not None
                and sender_thread.is_alive()
            ):
                return self

            self._stop_event.clear()
            self._paused.clear()
            self._sequence = 0
            self._stderr_tail.clear()
            self._connected = False
            self._next_reconnect_at = 0.0
            self._ws_reconnect_failures = 0
            self._capture_restart_failures = 0
            self._ready_backend = None
            self._identity_profiles_supported = True
            self._last_identity_profiles_sent_at = 0.0
            self._sent_frame_bucket = VoiceFrameTelemetryBucket(
                chunk_ms=self._chunk_ms,
                speech_threshold=self._speech_threshold,
            )
            self._skipped_frame_count = 0
            self._queue_drop_count = 0
            self._queue_high_watermark = 0
            self._sender_queue = queue.Queue(maxsize=self._transport_queue_frames)
            self._frame_speech_annotator.reset()
            with self._recent_remote_frames_lock:
                self._recent_remote_frames.clear()
            if self._duplex_keepalive is not None:
                self._duplex_keepalive.open()

            try:
                self._connect_client()
            except Exception as exc:
                self._connected = False
                self._trace_event(
                    "voice_edge_open_connect_failed",
                    kind="warning",
                    level="WARN",
                    details={"error_type": type(exc).__name__},
                )
                self._emit_status("voice_orchestrator_unavailable", type(exc).__name__)

            sender_thread = Thread(target=self._sender_loop, daemon=True, name="twinr-voice-orchestrator-send")
            capture_thread = Thread(target=self._capture_loop, daemon=True, name="twinr-voice-orchestrator")
            self._sender_thread = sender_thread
            self._thread = capture_thread
            sender_thread.start()
            capture_thread.start()
            self._record_memory_probe(
                self._sender_memory_probe,
                force=True,
                owner_detail="Voice orchestrator sender thread started.",
            )
            self._record_memory_probe(
                self._capture_memory_probe,
                force=True,
                owner_detail="Voice orchestrator capture thread started.",
            )
        return self

    def close(self) -> None:
        """Stop capture and close the websocket transport."""

        with self._lifecycle_lock:
            capture_thread = self._thread
            sender_thread = self._sender_thread
            process = self._process
            self._stop_event.set()
            self._enqueue_sender_sentinel()
            duplex_keepalive = self._duplex_keepalive

        if duplex_keepalive is not None:
            duplex_keepalive.close()
        if process is not None:
            self._stop_process(process)

        if capture_thread is not None and capture_thread is not current_thread():
            capture_thread.join(timeout=max(1.0, self._capture_restart_max_s))
        if sender_thread is not None and sender_thread is not current_thread():
            sender_thread.join(timeout=max(1.0, self._ws_reconnect_max_s))

        self._close_client()
        self._frame_speech_annotator.reset()

        self._flush_sent_frame_bucket()
        with self._lifecycle_lock:
            self._thread = None
            self._sender_thread = None
            self._process = None
            self._connected = False
            self._next_reconnect_at = 0.0
            self._ready_backend = None
            with self._recent_remote_frames_lock:
                self._recent_remote_frames.clear()
            self._drain_sender_queue()

    def __enter__(self) -> "EdgeVoiceOrchestrator":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def pause_capture(self, *, reason: str) -> None:
        """Keep compatibility with old callers without pausing server-only capture."""

        self._emit_status("voice_orchestrator_capture_pause_ignored", reason)

    def resume_capture(self, *, reason: str) -> None:
        """Keep compatibility with old callers without mutating capture state."""

        self._emit_status("voice_orchestrator_capture_resume_ignored", reason)

    def notify_runtime_state(
        self,
        *,
        state: str,
        detail: str | None = None,
        follow_up_allowed: bool = False,
        attention_state: str | None = None,
        interaction_intent_state: str | None = None,
        person_visible: bool | None = None,
        presence_active: bool | None = None,
        interaction_ready: bool | None = None,
        targeted_inference_blocked: bool | None = None,
        recommended_channel: str | None = None,
        speaker_associated: bool | None = None,
        speaker_association_confidence: float | None = None,
        background_media_likely: bool | None = None,
        speech_overlap_likely: bool | None = None,
        voice_quiet_until_utc: str | None = None,
    ) -> None:
        """Send the current edge runtime state to the server."""

        event = self._build_runtime_state_event(
            state=state,
            detail=detail,
            follow_up_allowed=follow_up_allowed,
            attention_state=attention_state,
            interaction_intent_state=interaction_intent_state,
            person_visible=person_visible,
            presence_active=presence_active,
            interaction_ready=interaction_ready,
            targeted_inference_blocked=targeted_inference_blocked,
            recommended_channel=recommended_channel,
            speaker_associated=speaker_associated,
            speaker_association_confidence=speaker_association_confidence,
            background_media_likely=background_media_likely,
            speech_overlap_likely=speech_overlap_likely,
            voice_quiet_until_utc=voice_quiet_until_utc,
        )
        with self._state_lock:
            self._last_runtime_state = event
        if not self._ensure_connected():
            return
        try:
            with self._runtime_state_send_lock, self._transport_send_lock:
                self._client.send_runtime_state(event)
        except Exception as exc:
            self._mark_disconnected(
                emit_message=f"voice_orchestrator_state_failed={type(exc).__name__}",
                retry_delay_s=0.0,
            )

    def seed_runtime_state(
        self,
        *,
        state: str,
        detail: str | None = None,
        follow_up_allowed: bool = False,
        attention_state: str | None = None,
        interaction_intent_state: str | None = None,
        person_visible: bool | None = None,
        presence_active: bool | None = None,
        interaction_ready: bool | None = None,
        targeted_inference_blocked: bool | None = None,
        recommended_channel: str | None = None,
        speaker_associated: bool | None = None,
        speaker_association_confidence: float | None = None,
        background_media_likely: bool | None = None,
        speech_overlap_likely: bool | None = None,
        voice_quiet_until_utc: str | None = None,
    ) -> None:
        """Cache one runtime state before the websocket opens."""

        event = self._build_runtime_state_event(
            state=state,
            detail=detail,
            follow_up_allowed=follow_up_allowed,
            attention_state=attention_state,
            interaction_intent_state=interaction_intent_state,
            person_visible=person_visible,
            presence_active=presence_active,
            interaction_ready=interaction_ready,
            targeted_inference_blocked=targeted_inference_blocked,
            recommended_channel=recommended_channel,
            speaker_associated=speaker_associated,
            speaker_association_confidence=speaker_association_confidence,
            background_media_likely=background_media_likely,
            speech_overlap_likely=speech_overlap_likely,
            voice_quiet_until_utc=voice_quiet_until_utc,
        )
        with self._state_lock:
            self._last_runtime_state = event

    @property
    def ready_backend(self) -> str | None:
        """Return the last backend label confirmed by the live gateway."""

        normalized = str(self._ready_backend or "").strip().lower()
        return normalized or None

    def supports_remote_follow_up(self) -> bool:
        """Return whether the live gateway owns follow-up on the same stream."""

        return True

    def notify_identity_profiles(
        self,
        event: OrchestratorVoiceIdentityProfilesEvent,
    ) -> None:
        """Cache and, when connected, send the current household voice profiles."""

        with self._state_lock:
            self._last_identity_profiles = event
        if not self._identity_profiles_supported:
            return
        if not self._ensure_connected():
            return
        try:
            with self._runtime_state_send_lock, self._transport_send_lock:
                self._send_identity_profiles(event)
        except Exception as exc:
            self._mark_disconnected(
                emit_message=f"voice_orchestrator_identity_failed={type(exc).__name__}",
                retry_delay_s=0.0,
            )

    def _close_client(self) -> None:
        try:
            with self._transport_send_lock:
                self._client.close()
        except Exception:
            pass

    def _compute_backoff(self, failures: int, *, base_s: float, max_s: float) -> float:
        exponent = max(0, int(failures) - 1)
        raw = min(float(max_s), float(base_s) * (2 ** exponent))
        jitter = random.uniform(0.0, max(raw * 0.2, 0.05))
        return min(float(max_s), raw + jitter)

    def _connect_client(self) -> None:
        with self._state_lock:
            last_runtime_state = self._last_runtime_state

        with self._transport_send_lock:
            self._client.open()
            self._client.send_hello(self._build_hello_request(last_runtime_state))

            with self._state_lock:
                current_runtime_state = self._last_runtime_state
                current_identity_profiles = self._last_identity_profiles

            if current_runtime_state is not None:
                self._client.send_runtime_state(current_runtime_state)
            if current_identity_profiles is not None and self._identity_profiles_supported:
                self._send_identity_profiles(current_identity_profiles)

        self._connected = True
        self._next_reconnect_at = 0.0
        self._ws_reconnect_failures = 0
        self._trace_event(
            "voice_edge_client_connected",
            kind="io",
            details={
                "state_attested": current_runtime_state is not None,
                "initial_state": current_runtime_state.state if current_runtime_state is not None else "waiting",
                "identity_profiles_supported": self._identity_profiles_supported,
                "identity_profiles_revision": (
                    current_identity_profiles.revision if current_identity_profiles is not None else None
                ),
                "identity_profiles_count": (
                    len(current_identity_profiles.profiles) if current_identity_profiles is not None else 0
                ),
            },
        )

    def _send_identity_profiles(
        self,
        event: OrchestratorVoiceIdentityProfilesEvent,
    ) -> None:
        """Send one optional household voice-profile snapshot when supported."""

        self._client.send_identity_profiles(event)
        self._last_identity_profiles_sent_at = time.monotonic()

    def _should_disable_identity_profiles_for_error(self, error: str) -> bool:
        """Return whether one gateway error proves legacy identity-profile incompatibility."""

        if not self._identity_profiles_supported:
            return False
        if str(error or "").strip() != self._IDENTITY_PROFILE_UNSUPPORTED_ERROR:
            return False
        with self._state_lock:
            has_profiles = self._last_identity_profiles is not None
        if not has_profiles:
            return False
        return (time.monotonic() - self._last_identity_profiles_sent_at) <= self._IDENTITY_PROFILE_ERROR_WINDOW_S

    def _build_runtime_state_event(
        self,
        *,
        state: str,
        detail: str | None = None,
        follow_up_allowed: bool = False,
        attention_state: str | None = None,
        interaction_intent_state: str | None = None,
        person_visible: bool | None = None,
        presence_active: bool | None = None,
        interaction_ready: bool | None = None,
        targeted_inference_blocked: bool | None = None,
        recommended_channel: str | None = None,
        speaker_associated: bool | None = None,
        speaker_association_confidence: float | None = None,
        background_media_likely: bool | None = None,
        speech_overlap_likely: bool | None = None,
        voice_quiet_until_utc: str | None = None,
    ) -> OrchestratorVoiceRuntimeStateEvent:
        """Construct one normalized runtime-state event for caching and send."""

        return OrchestratorVoiceRuntimeStateEvent(
            state=state,
            detail=detail,
            follow_up_allowed=follow_up_allowed,
            attention_state=attention_state,
            interaction_intent_state=interaction_intent_state,
            person_visible=person_visible,
            presence_active=presence_active,
            interaction_ready=interaction_ready,
            targeted_inference_blocked=targeted_inference_blocked,
            recommended_channel=recommended_channel,
            speaker_associated=speaker_associated,
            speaker_association_confidence=speaker_association_confidence,
            background_media_likely=background_media_likely,
            speech_overlap_likely=speech_overlap_likely,
            voice_quiet_until_utc=voice_quiet_until_utc,
        )

    def _build_hello_request(
        self,
        runtime_state: OrchestratorVoiceRuntimeStateEvent | None,
    ) -> OrchestratorVoiceHelloRequest:
        """Attach the last attested runtime state to the opening hello."""

        return OrchestratorVoiceHelloRequest(
            session_id=self._session_id,
            sample_rate=self._sample_rate,
            channels=self._channels,
            chunk_ms=self._chunk_ms,
            trace_id=self._trace_id,
            initial_state=runtime_state.state if runtime_state is not None else "waiting",
            detail=runtime_state.detail if runtime_state is not None else None,
            follow_up_allowed=runtime_state.follow_up_allowed if runtime_state is not None else False,
            attention_state=runtime_state.attention_state if runtime_state is not None else None,
            interaction_intent_state=(
                runtime_state.interaction_intent_state if runtime_state is not None else None
            ),
            person_visible=runtime_state.person_visible if runtime_state is not None else None,
            presence_active=runtime_state.presence_active if runtime_state is not None else None,
            interaction_ready=runtime_state.interaction_ready if runtime_state is not None else None,
            targeted_inference_blocked=(
                runtime_state.targeted_inference_blocked if runtime_state is not None else None
            ),
            recommended_channel=runtime_state.recommended_channel if runtime_state is not None else None,
            speaker_associated=runtime_state.speaker_associated if runtime_state is not None else None,
            speaker_association_confidence=(
                runtime_state.speaker_association_confidence if runtime_state is not None else None
            ),
            background_media_likely=(
                runtime_state.background_media_likely if runtime_state is not None else None
            ),
            speech_overlap_likely=(
                runtime_state.speech_overlap_likely if runtime_state is not None else None
            ),
            voice_quiet_until_utc=runtime_state.voice_quiet_until_utc if runtime_state is not None else None,
            state_attested=runtime_state is not None,
        )

    def _ensure_connected(self) -> bool:
        """Reconnect the websocket after transient closures without restarting Twinr."""

        if self._connected:
            return True
        if self._stop_event.is_set():
            return False

        now = time.monotonic()
        with self._lifecycle_lock:
            if self._connected:
                return True
            if now < self._next_reconnect_at:
                return False

            self._close_client()
            try:
                self._connect_client()
            except Exception as exc:
                self._connected = False
                self._ws_reconnect_failures += 1
                retry_delay_s = self._compute_backoff(
                    self._ws_reconnect_failures,
                    base_s=self._ws_reconnect_base_s,
                    max_s=self._ws_reconnect_max_s,
                )
                self._next_reconnect_at = now + retry_delay_s
                self._trace_event(
                    "voice_edge_reconnect_failed",
                    kind="warning",
                    level="WARN",
                    details={
                        "error_type": type(exc).__name__,
                        "ws_reconnect_failures": self._ws_reconnect_failures,
                        "retry_delay_s": retry_delay_s,
                    },
                )
                self._emit_status("voice_orchestrator_reconnect_failed", type(exc).__name__)
                return False

        self._emit_status("voice_orchestrator_reconnected", "true")
        return True

    def _mark_disconnected(self, *, emit_message: str, retry_delay_s: float) -> None:
        """Drop the current websocket and allow a bounded reconnect attempt later."""

        retry_delay = max(0.0, float(retry_delay_s))
        with self._lifecycle_lock:
            self._connected = False
            self._next_reconnect_at = time.monotonic() + retry_delay
        self._close_client()
        self._trace_event(
            "voice_edge_client_disconnected",
            kind="warning",
            level="WARN",
            details={"emit_message": emit_message, "retry_delay_s": retry_delay},
        )
        self.emit(self._sanitize_emit_value(emit_message))

    def _enqueue_sender_sentinel(self) -> None:
        while True:
            try:
                self._sender_queue.put_nowait(None)
                return
            except queue.Full:
                try:
                    self._sender_queue.get_nowait()
                except queue.Empty:
                    return

    def _drain_sender_queue(self) -> None:
        while True:
            try:
                self._sender_queue.get_nowait()
            except queue.Empty:
                return

    def _sender_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    queued_frame = self._sender_queue.get(timeout=self._QUEUE_POLL_TIMEOUT_S)
                except queue.Empty:
                    continue
                if queued_frame is None:
                    break
                self._send_queued_frame(queued_frame)
                self._record_memory_probe(self._sender_memory_probe)
        finally:
            if self._sender_thread is current_thread():
                with self._lifecycle_lock:
                    self._sender_thread = None

    def _send_queued_frame(self, queued_frame: _QueuedAudioFrame) -> None:
        while not self._stop_event.is_set():
            frame_age_s = time.monotonic() - queued_frame.captured_at_monotonic
            if frame_age_s > (self._transport_queue_ms / 1000.0):
                self._skipped_frame_count += 1
                if self._skipped_frame_count == 1 or self._skipped_frame_count % 10 == 0:
                    self._trace_event(
                        "voice_edge_frame_skipped_stale",
                        kind="warning",
                        level="WARN",
                        details={
                            "skipped_frame_count": self._skipped_frame_count,
                            "frame_age_ms": int(frame_age_s * 1000),
                        },
                    )
                return

            if not self._ensure_connected():
                sleep_s = min(
                    self._QUEUE_POLL_TIMEOUT_S,
                    max(0.01, self._next_reconnect_at - time.monotonic()),
                )
                self._stop_event.wait(sleep_s)
                continue

            if self._send_frame_now(queued_frame.pcm_bytes):
                return

    def _send_frame(self, pcm_bytes: bytes) -> None:
        """Preserve the legacy immediate-send helper used by tests and callers.

        The transport queue is now the normal live path, but direct call sites
        still rely on the old `_send_frame(...)` contract for focused recovery
        and reconnection checks.
        """

        if not pcm_bytes:
            return
        if not self._ensure_connected():
            self._skipped_frame_count += 1
            if self._skipped_frame_count == 1 or self._skipped_frame_count % 10 == 0:
                self._trace_event(
                    "voice_edge_frame_skipped_unconnected",
                    kind="warning",
                    level="WARN",
                    details={"skipped_frame_count": self._skipped_frame_count},
                )
            return
        self._send_frame_now(pcm_bytes)

    def _capture_loop(self) -> None:
        process: subprocess.Popen[bytes] | None = None
        pending_pcm = bytearray()
        started_once = False
        sent_any_frame = False
        capture_recovery_attempted = False
        first_frame_deadline_at: float | None = None
        last_capture_activity_at: float | None = None

        try:
            while not self._stop_event.is_set():
                try:
                    if self._paused.is_set():
                        if process is not None:
                            stale_process = process
                            self._stop_process(stale_process)
                            process = None
                            with self._lifecycle_lock:
                                if self._process is stale_process:
                                    self._process = None
                        self._stop_event.wait(0.05)
                        continue

                    if process is None:
                        process = self._start_process()
                        started_once = True
                        pending_pcm.clear()
                        sent_any_frame = False
                        capture_recovery_attempted = False
                        first_frame_deadline_at = time.monotonic() + self._first_frame_timeout_s
                        last_capture_activity_at = None
                        self._capture_restart_failures = 0
                        self._record_memory_probe(
                            self._capture_memory_probe,
                            force=True,
                            owner_detail="Voice orchestrator capture process started or restarted.",
                        )

                    if process.stdout is None or process.stderr is None:
                        raise RuntimeError("Voice orchestrator capture did not expose stdout/stderr")

                    stdout_fd = process.stdout.fileno()
                    stderr_fd = process.stderr.fileno()
                    ready, _write_ready, _error_ready = select.select(
                        [stdout_fd, stderr_fd],
                        [],
                        [],
                        self._select_timeout_s,
                    )

                    if stderr_fd in ready:
                        self._drain_stderr(process)

                    if stdout_fd not in ready:
                        if process.poll() is not None:
                            raise RuntimeError(self._process_error_message(process))
                        self._raise_for_capture_activity_timeout(
                            sent_any_frame=sent_any_frame,
                            last_capture_activity_at=last_capture_activity_at,
                        )
                        if self._first_frame_timed_out(
                            sent_any_frame=sent_any_frame,
                            first_frame_deadline_at=first_frame_deadline_at,
                        ):
                            raise RuntimeError("Voice orchestrator capture produced no first frame before timeout")
                        continue

                    pcm_chunk = self._read_stdout_chunk(stdout_fd, self._frame_bytes - len(pending_pcm))
                    if not pcm_chunk:
                        if process.poll() is not None:
                            raise RuntimeError(self._process_error_message(process))
                        self._raise_for_capture_activity_timeout(
                            sent_any_frame=sent_any_frame,
                            last_capture_activity_at=last_capture_activity_at,
                        )
                        if self._first_frame_timed_out(
                            sent_any_frame=sent_any_frame,
                            first_frame_deadline_at=first_frame_deadline_at,
                        ):
                            raise RuntimeError("Voice orchestrator capture produced no first frame before timeout")
                        continue

                    last_capture_activity_at = time.monotonic()
                    pending_pcm.extend(pcm_chunk)
                    while len(pending_pcm) >= self._frame_bytes:
                        frame_bytes = bytes(pending_pcm[: self._frame_bytes])
                        del pending_pcm[: self._frame_bytes]
                        self._enqueue_frame(frame_bytes)
                        sent_any_frame = True
                        first_frame_deadline_at = None
                    self._record_memory_probe(self._capture_memory_probe)
                except Exception as exc:
                    if self._stop_event.is_set():
                        break

                    self._trace_event(
                        "voice_edge_capture_iteration_failed",
                        kind="warning",
                        level="WARN",
                        details={
                            "error_type": type(exc).__name__,
                            "error_message": self._sanitize_emit_value(exc),
                            "sent_any_frame": sent_any_frame,
                            "capture_recovery_attempted": capture_recovery_attempted,
                        },
                    )
                    if isinstance(exc, _CaptureStreamStalledError):
                        self._emit_status("voice_orchestrator_capture_stalled_ms", exc.stalled_ms)

                    if process is not None:
                        stale_process = process
                        self._stop_process(stale_process)
                        process = None
                        with self._lifecycle_lock:
                            if self._process is stale_process:
                                self._process = None

                    recovered = False
                    should_attempt_respeaker_recovery = (
                        not capture_recovery_attempted
                        and (
                            not sent_any_frame
                            or isinstance(exc, _CaptureStreamStalledError)
                        )
                    )
                    if should_attempt_respeaker_recovery:
                        capture_recovery_attempted = True
                        recovered = self._recover_transient_respeaker_capture()

                    if recovered:
                        continue

                    self._capture_restart_failures += 1
                    retry_delay_s = self._compute_backoff(
                        self._capture_restart_failures,
                        base_s=self._capture_restart_base_s,
                        max_s=self._capture_restart_max_s,
                    )

                    if self._capture_restart_failures == 1 or self._capture_restart_failures % 5 == 0:
                        self._emit_status("voice_orchestrator_capture_failed", type(exc).__name__)
                    self._emit_status("voice_orchestrator_capture_retrying_in_ms", int(retry_delay_s * 1000))
                    self._record_memory_probe(
                        self._capture_memory_probe,
                        force=True,
                        owner_detail=(
                            "Voice orchestrator capture loop hit an exception and is retrying "
                            f"after {int(retry_delay_s * 1000)} ms."
                        ),
                    )

                    self._stop_event.wait(retry_delay_s)

            if process is not None:
                self._drain_stderr(process)
        finally:
            if process is not None:
                self._stop_process(process)
            self._flush_sent_frame_bucket()
            with self._lifecycle_lock:
                if self._process is process:
                    self._process = None
                if self._thread is current_thread():
                    self._thread = None
            if started_once:
                self._emit_status("voice_orchestrator_capture", "stopped")

    def _enqueue_frame(self, pcm_bytes: bytes) -> None:
        if not pcm_bytes:
            return

        queued_frame = _QueuedAudioFrame(
            pcm_bytes=bytes(pcm_bytes),
            captured_at_monotonic=time.monotonic(),
        )
        try:
            self._sender_queue.put_nowait(queued_frame)
        except queue.Full:
            dropped = False
            try:
                dropped_item = self._sender_queue.get_nowait()
                dropped = dropped_item is not None
            except queue.Empty:
                dropped = False
            try:
                self._sender_queue.put_nowait(queued_frame)
            except queue.Full:
                self._skipped_frame_count += 1
                if self._skipped_frame_count == 1 or self._skipped_frame_count % 10 == 0:
                    self._trace_event(
                        "voice_edge_frame_skipped_queue_full",
                        kind="warning",
                        level="WARN",
                        details={
                            "skipped_frame_count": self._skipped_frame_count,
                            "queue_capacity": self._transport_queue_frames,
                        },
                    )
                return
            if dropped:
                self._queue_drop_count += 1
                if self._queue_drop_count == 1 or self._queue_drop_count % 10 == 0:
                    self._trace_event(
                        "voice_edge_frame_dropped_backpressure",
                        kind="warning",
                        level="WARN",
                        details={
                            "queue_drop_count": self._queue_drop_count,
                            "queue_capacity": self._transport_queue_frames,
                        },
                    )
                    self._emit_status("voice_orchestrator_backpressure_drops", self._queue_drop_count)

        queue_size = self._sender_queue.qsize()
        if queue_size > self._queue_high_watermark:
            self._queue_high_watermark = queue_size

    def _send_frame_now(self, pcm_bytes: bytes) -> bool:
        if not pcm_bytes:
            return True

        try:
            with self._state_lock:
                latest_runtime_state = self._last_runtime_state
            speech_evidence = self._frame_speech_annotator.classify_frame(pcm_bytes)

            with self._transport_send_lock:
                self._client.send_audio_frame(
                    OrchestratorVoiceAudioFrame(
                        sequence=self._sequence,
                        pcm_bytes=pcm_bytes,
                        runtime_state=latest_runtime_state,
                        speech_probability=speech_evidence.speech_probability,
                    )
                )

            self._remember_recent_remote_frame(pcm_bytes)
            self._skipped_frame_count = 0
            self._sent_frame_bucket.add_frame(sequence=self._sequence, pcm_bytes=pcm_bytes)
            if self._sent_frame_bucket.should_flush():
                self._flush_sent_frame_bucket()
            self._sequence += 1
            return True
        except Exception as exc:
            self._trace_event(
                "voice_edge_frame_send_failed",
                kind="warning",
                level="WARN",
                details={"sequence": self._sequence, "error_type": type(exc).__name__},
            )
            self._mark_disconnected(
                emit_message=f"voice_orchestrator_send_failed={type(exc).__name__}",
                retry_delay_s=0.0,
            )
            return False

    def _remember_recent_remote_frame(self, pcm_bytes: bytes) -> None:
        """Retain a short rolling audio buffer for remote commit/passive update use."""

        if not pcm_bytes:
            return
        frame = _BufferedRemoteAudioFrame(
            pcm_bytes=bytes(pcm_bytes),
            duration_ms=self._chunk_ms,
        )
        with self._recent_remote_frames_lock:
            self._recent_remote_frames.append(frame)

    def _recent_remote_audio_bytes(self) -> bytes:
        """Return the buffered remote-user audio window in chronological order."""

        with self._recent_remote_frames_lock:
            frames = tuple(self._recent_remote_frames)
        return b"".join(frame.pcm_bytes for frame in frames)

    def _emit_recent_remote_audio(self, *, source: str) -> None:
        """Forward the latest buffered remote audio to the runtime when needed."""

        callback = self._on_recent_remote_audio
        if callback is None:
            return
        pcm_bytes = self._recent_remote_audio_bytes()
        if not pcm_bytes:
            return
        try:
            callback(pcm_bytes, source)
        except Exception as exc:
            self._trace_event(
                "voice_edge_recent_remote_audio_callback_failed",
                kind="warning",
                level="WARN",
                details={"source": source, "error_type": type(exc).__name__},
            )
            self._emit_status("voice_orchestrator_recent_audio_failed", type(exc).__name__)

    def _handle_server_event(self, event) -> None:
        if isinstance(event, OrchestratorVoiceReadyEvent):
            self._ready_backend = str(event.backend or "").strip().lower() or None
            self._trace_event(
                "voice_edge_server_ready",
                kind="io",
                details={"backend": self._ready_backend},
            )
            self._emit_status("voice_orchestrator_ready", event.backend)
            self._connected = True
            self._next_reconnect_at = 0.0
            self._ws_reconnect_failures = 0
            return

        if isinstance(event, OrchestratorVoiceWakeConfirmedEvent):
            self._emit_recent_remote_audio(source="wake")
            self._trace_event(
                "voice_edge_server_wake_confirmed",
                kind="decision",
                details={
                    "matched_phrase": event.matched_phrase,
                    "remaining_text_chars": len(str(event.remaining_text or "").strip()),
                },
            )
            self._emit_status("voice_orchestrator_wake_confirmed", event.matched_phrase or "unknown")
            self._on_voice_activation(
                VoiceActivationMatch(
                    detected=True,
                    transcript="",
                    matched_phrase=event.matched_phrase,
                    remaining_text=event.remaining_text,
                    normalized_transcript="",
                    backend=event.backend,
                    detector_label=event.detector_label,
                    score=event.score,
                )
            )
            return

        if isinstance(event, OrchestratorVoiceTranscriptCommittedEvent):
            self._emit_recent_remote_audio(source=event.source)
            self._trace_event(
                "voice_edge_server_transcript_committed",
                kind="decision",
                details={"source": event.source, "transcript_chars": len(str(event.transcript or "").strip())},
            )
            self._emit_status("voice_orchestrator_transcript_committed", event.source)
            self._on_transcript_committed(event.transcript, event.source)
            return

        if isinstance(event, OrchestratorVoiceFollowUpClosedEvent):
            self._trace_event(
                "voice_edge_server_follow_up_closed",
                kind="mutation",
                details={"reason": event.reason},
            )
            self._emit_status("voice_orchestrator_follow_up_closed", event.reason)
            if self._on_follow_up_closed is not None:
                self._on_follow_up_closed(event.reason)
            return

        if isinstance(event, OrchestratorVoiceBargeInInterruptEvent):
            self._trace_event(
                "voice_edge_server_barge_in",
                kind="decision",
                details={"transcript_preview_chars": len(str(event.transcript_preview or "").strip())},
            )
            self._emit_status("voice_orchestrator_barge_in_interrupt", "true")
            self._on_barge_in_interrupt()
            return

        if isinstance(event, OrchestratorVoiceErrorEvent):
            if self._should_disable_identity_profiles_for_error(event.error):
                self._identity_profiles_supported = False
                self._last_identity_profiles_sent_at = 0.0
                self._trace_event(
                    "voice_edge_identity_profiles_unsupported",
                    kind="warning",
                    level="WARN",
                    details={"error": event.error},
                )
                self._emit_status("voice_orchestrator_identity_profiles_unsupported", "true")
                return
            self._trace_event(
                "voice_edge_server_error",
                kind="warning",
                level="WARN",
                details={"error": event.error},
            )
            self._mark_disconnected(
                emit_message=f"voice_orchestrator_error={self._sanitize_emit_value(event.error)}",
                retry_delay_s=0.0,
            )
            return

        self._emit_status("voice_orchestrator_event", type(event).__name__)

    def _start_process(self) -> subprocess.Popen[bytes]:
        arecord_path = shutil.which("arecord")
        if arecord_path is None:
            raise RuntimeError("arecord executable not found")

        process = subprocess.Popen(
            [
                arecord_path,
                "-D",
                self._device,
                "-q",
                "-N",
                "-t",
                "raw",
                "-f",
                "S16_LE",
                "-c",
                str(self._channels),
                "-r",
                str(self._sample_rate),
                "--period-size",
                str(self._capture_period_frames),
                "--buffer-size",
                str(self._capture_buffer_frames),
                "-A",
                str(self._chunk_ms * 1000),
            ],
            env=build_audio_subprocess_env_for_mode(
                allow_root_borrowed_session_audio=True,
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            close_fds=True,
        )
        if process.stdout is None or process.stderr is None:
            raise RuntimeError("Voice orchestrator capture process did not expose stdout/stderr")
        os.set_blocking(process.stdout.fileno(), False)
        os.set_blocking(process.stderr.fileno(), False)
        with self._lifecycle_lock:
            self._process = process
        self._trace_event(
            "voice_edge_capture_started",
            kind="io",
            details={
                "frame_bytes": self._frame_bytes,
                "capture_period_frames": self._capture_period_frames,
                "capture_buffer_frames": self._capture_buffer_frames,
                "select_timeout_s": self._select_timeout_s,
                "first_frame_timeout_s": self._first_frame_timeout_s,
                "ongoing_frame_timeout_s": self._ongoing_frame_timeout_s,
            },
        )
        self._emit_status("voice_orchestrator_capture", "started")
        return process

    def _first_frame_timed_out(
        self,
        *,
        sent_any_frame: bool,
        first_frame_deadline_at: float | None,
    ) -> bool:
        if sent_any_frame or first_frame_deadline_at is None:
            return False
        return time.monotonic() >= first_frame_deadline_at

    def _raise_for_capture_activity_timeout(
        self,
        *,
        sent_any_frame: bool,
        last_capture_activity_at: float | None,
    ) -> None:
        """Abort one live capture when bytes stop arriving mid-stream."""

        if not sent_any_frame or last_capture_activity_at is None:
            return
        stalled_s = time.monotonic() - last_capture_activity_at
        if stalled_s < self._ongoing_frame_timeout_s:
            return
        raise _CaptureStreamStalledError(stalled_ms=int(round(stalled_s * 1000.0)))

    def _read_stdout_chunk(self, stdout_fd: int, read_size: int) -> bytes:
        try:
            return os.read(stdout_fd, max(1, read_size))
        except (BlockingIOError, InterruptedError):
            return b""

    def _drain_stderr(self, process: subprocess.Popen[bytes]) -> None:
        stderr = process.stderr
        if stderr is None:
            return
        while True:
            try:
                chunk = os.read(stderr.fileno(), 4096)
            except (BlockingIOError, InterruptedError, OSError):
                return
            if not chunk:
                return
            self._stderr_tail.extend(chunk)
            if len(self._stderr_tail) > self._MAX_STDERR_BYTES:
                del self._stderr_tail[:-self._MAX_STDERR_BYTES]

    def _process_error_message(self, process: subprocess.Popen[bytes]) -> str:
        self._drain_stderr(process)
        stderr = bytes(self._stderr_tail).strip()
        if stderr:
            return stderr.decode("utf-8", errors="ignore")
        return f"Voice orchestrator capture exited with code {process.returncode}"

    def _recover_transient_respeaker_capture(self) -> bool:
        """Recover one stalled XVF3800 capture path before retrying."""

        recovered = recover_stalled_respeaker_capture(
            device=self._device,
            sample_rate=self._sample_rate,
            channels=self._channels,
            chunk_ms=self._chunk_ms,
            max_wait_s=self._capture_restart_max_s,
            should_stop=lambda: self._stop_event.is_set() or self._paused.is_set(),
        )
        if recovered:
            self._emit_status("voice_orchestrator_capture_recovered", "respeaker_recovery")
        return recovered

    def _stop_process(self, process: subprocess.Popen[bytes]) -> None:
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    try:
                        process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        pass
        finally:
            for pipe in (process.stdout, process.stderr):
                if pipe is not None:
                    try:
                        pipe.close()
                    except OSError:
                        pass


__all__ = ["EdgeVoiceOrchestrator"]
