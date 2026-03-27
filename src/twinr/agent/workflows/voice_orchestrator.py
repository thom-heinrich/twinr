"""Own the edge-side streaming voice bridge for the orchestrator path.

This module keeps `realtime_runner.py` orchestration-focused by handling the
bounded `arecord` lifecycle, websocket transport, and decision dispatch for the
Alexa-like server-backed voice path. It does not run turns itself; it only
forwards server-side wake/transcript-commit/follow-up-close/barge-in decisions
back into the realtime loop.
"""

from __future__ import annotations

import os
import select
import shutil
import subprocess
from threading import Event, Lock, Thread, current_thread
import time
from typing import Callable
from uuid import uuid4

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.workflows.forensics import WorkflowForensics
from twinr.hardware.audio import resolve_capture_device
from twinr.hardware.audio_env import build_audio_subprocess_env
from twinr.hardware.respeaker_capture_recovery import wait_for_transient_respeaker_capture_ready
from twinr.orchestrator.voice_client import OrchestratorVoiceWebSocketClient
from twinr.orchestrator.voice_contracts import (
    OrchestratorVoiceAudioFrame,
    OrchestratorVoiceBargeInInterruptEvent,
    OrchestratorVoiceErrorEvent,
    OrchestratorVoiceFollowUpClosedEvent,
    OrchestratorVoiceHelloRequest,
    OrchestratorVoiceReadyEvent,
    OrchestratorVoiceRuntimeStateEvent,
    OrchestratorVoiceTranscriptCommittedEvent,
    OrchestratorVoiceWakeConfirmedEvent,
)
from twinr.orchestrator.voice_activation import VoiceActivationMatch
from twinr.orchestrator.voice_forensics import VoiceFrameTelemetryBucket


class EdgeVoiceOrchestrator:
    """Stream bounded PCM frames to the remote voice orchestrator service."""

    _SELECT_TIMEOUT_S = 0.2
    _STOP_WAIT_TIMEOUT_S = 1.0
    _STOP_JOIN_TIMEOUT_S = 3.0
    _MAX_STDERR_BYTES = 8_192
    _RECONNECT_RETRY_DELAY_S = 1.0

    def __init__(
        self,
        config: TwinrConfig,
        *,
        emit: Callable[[str], None],
        on_voice_activation: Callable[[VoiceActivationMatch], bool],
        on_transcript_committed: Callable[[str, str], bool],
        on_follow_up_closed: Callable[[str], None] | None = None,
        on_barge_in_interrupt: Callable[[], bool],
        forensics: WorkflowForensics | None = None,
    ) -> None:
        self.config = config
        self.emit = emit
        self._on_voice_activation = on_voice_activation
        self._on_transcript_committed = on_transcript_committed
        self._on_follow_up_closed = on_follow_up_closed
        self._on_barge_in_interrupt = on_barge_in_interrupt
        self._device = resolve_capture_device(
            config.voice_orchestrator_audio_device,
            config.proactive_audio_input_device,
            config.audio_input_device,
        )
        self._sample_rate = int(config.audio_sample_rate)
        self._channels = int(config.audio_channels)
        self._chunk_ms = max(20, int(config.audio_chunk_ms))
        self._speech_threshold = max(0, int(config.audio_speech_threshold))
        self._frame_bytes = max(320, int(round(self._sample_rate * (self._chunk_ms / 1000.0))) * self._channels * 2)
        self._client = OrchestratorVoiceWebSocketClient(
            config.voice_orchestrator_ws_url,
            shared_secret=config.voice_orchestrator_shared_secret,
            on_event=self._handle_server_event,
            require_tls=False,
        )
        self._session_id = f"voice-{uuid4().hex[:12]}"
        self._stop_event = Event()
        self._paused = Event()
        self._lifecycle_lock = Lock()
        self._state_lock = Lock()
        self._runtime_state_send_lock = Lock()
        self._thread: Thread | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._sequence = 0
        self._stderr_tail = bytearray()
        self._connected = False
        self._next_reconnect_at = 0.0
        self._last_runtime_state: OrchestratorVoiceRuntimeStateEvent | None = None
        self._ready_backend: str | None = None
        self._forensics = forensics if isinstance(forensics, WorkflowForensics) and forensics.enabled else None
        self._trace_id = self._session_id
        self._sent_frame_bucket = VoiceFrameTelemetryBucket(
            chunk_ms=self._chunk_ms,
            speech_threshold=self._speech_threshold,
        )
        self._skipped_frame_count = 0

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
        self._trace_event(
            "voice_edge_frame_window_sent",
            kind="io",
            details=self._sent_frame_bucket.flush_details(),
        )

    def open(self) -> "EdgeVoiceOrchestrator":
        """Connect the websocket and start the bounded capture worker.

        The edge capture worker must still start when the first websocket dial
        fails. Otherwise Twinr loses the long-lived microphone stream entirely
        and can only reconnect if some unrelated local path later pokes the
        orchestrator. Starting the worker anyway lets frame sends drive the
        bounded reconnect loop as soon as the remote gateway becomes reachable.
        """

        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return self
            self._stop_event.clear()
            self._paused.clear()
            self._sequence = 0
            self._stderr_tail.clear()
            self._next_reconnect_at = 0.0
            self._ready_backend = None
            self._sent_frame_bucket = VoiceFrameTelemetryBucket(
                chunk_ms=self._chunk_ms,
                speech_threshold=self._speech_threshold,
            )
            self._skipped_frame_count = 0
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
                self.emit(f"voice_orchestrator_unavailable={type(exc).__name__}")
            thread = Thread(target=self._capture_loop, daemon=True, name="twinr-voice-orchestrator")
            self._thread = thread
            thread.start()
        return self

    def close(self) -> None:
        """Stop capture and close the websocket transport."""

        with self._lifecycle_lock:
            thread = self._thread
            process = self._process
            self._stop_event.set()
        if process is not None:
            self._stop_process(process)
        if thread is not None and thread is not current_thread():
            thread.join(timeout=self._STOP_JOIN_TIMEOUT_S)
        self._client.close()
        self._flush_sent_frame_bucket()
        with self._lifecycle_lock:
            self._thread = None
            self._process = None
            self._connected = False
            self._next_reconnect_at = 0.0
            self._ready_backend = None

    def __enter__(self) -> "EdgeVoiceOrchestrator":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def pause_capture(self, *, reason: str) -> None:
        """Keep compatibility with old callers without pausing server-only capture."""

        self.emit(f"voice_orchestrator_capture_pause_ignored={reason}")

    def resume_capture(self, *, reason: str) -> None:
        """Keep compatibility with old callers without mutating capture state."""

        self.emit(f"voice_orchestrator_capture_resume_ignored={reason}")

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
            voice_quiet_until_utc=voice_quiet_until_utc,
        )
        with self._state_lock:
            self._last_runtime_state = event
        if not self._ensure_connected():
            return
        try:
            with self._runtime_state_send_lock:
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
        voice_quiet_until_utc: str | None = None,
    ) -> None:
        """Cache one runtime state before the websocket opens.

        The orchestrator hello now carries the last attested edge state so the
        server can fail closed until it knows the current waiting/listening
        context. Startup code uses this setter before entering the websocket
        context so the very first hello already includes the idle person-state
        snapshot instead of opening in permissive waiting/null-intent mode.
        """

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

    def _connect_client(self) -> None:
        with self._state_lock:
            last_runtime_state = self._last_runtime_state
        self._client.open()
        self._client.send_hello(self._build_hello_request(last_runtime_state))
        self._connected = True
        self._next_reconnect_at = 0.0
        with self._runtime_state_send_lock:
            with self._state_lock:
                current_runtime_state = self._last_runtime_state
            if current_runtime_state is not None:
                self._client.send_runtime_state(current_runtime_state)
        self._trace_event(
            "voice_edge_client_connected",
            kind="io",
            details={
                "state_attested": current_runtime_state is not None,
                "initial_state": current_runtime_state.state if current_runtime_state is not None else "waiting",
            },
        )

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
            try:
                self._client.close()
            except Exception:
                pass
            try:
                self._connect_client()
            except Exception as exc:
                self._connected = False
                self._next_reconnect_at = now + self._RECONNECT_RETRY_DELAY_S
                self.emit(f"voice_orchestrator_reconnect_failed={type(exc).__name__}")
                return False
        self.emit("voice_orchestrator_reconnected=true")
        return True

    def _mark_disconnected(self, *, emit_message: str, retry_delay_s: float) -> None:
        """Drop the current websocket and allow a bounded reconnect attempt later."""

        with self._lifecycle_lock:
            self._connected = False
            self._next_reconnect_at = time.monotonic() + max(0.0, float(retry_delay_s))
        try:
            self._client.close()
        except Exception:
            pass
        self._trace_event(
            "voice_edge_client_disconnected",
            kind="warning",
            level="WARN",
            details={"emit_message": emit_message, "retry_delay_s": retry_delay_s},
        )
        self.emit(emit_message)

    def _capture_loop(self) -> None:
        process: subprocess.Popen[bytes] | None = None
        pending_pcm = bytearray()
        started = False
        sent_any_frame = False
        capture_recovery_attempted = False
        try:
            while not self._stop_event.is_set():
                if self._paused.is_set():
                    if process is not None:
                        self._stop_process(process)
                        process = None
                        with self._lifecycle_lock:
                            if self._process is process:
                                self._process = None
                    self._stop_event.wait(0.05)
                    continue
                if process is None:
                    process = self._start_process()
                    started = True
                    pending_pcm.clear()
                    sent_any_frame = False
                    capture_recovery_attempted = False
                if process.stdout is None or process.stderr is None:
                    raise RuntimeError("Voice orchestrator capture did not expose stdout/stderr")
                stdout_fd = process.stdout.fileno()
                stderr_fd = process.stderr.fileno()
                ready, _write_ready, _error_ready = select.select(
                    [stdout_fd, stderr_fd],
                    [],
                    [],
                    self._SELECT_TIMEOUT_S,
                )
                if stderr_fd in ready:
                    self._drain_stderr(process)
                if stdout_fd not in ready:
                    if process.poll() is not None:
                        recovered = False
                        if not sent_any_frame and not capture_recovery_attempted:
                            capture_recovery_attempted = True
                            recovered = self._recover_transient_respeaker_capture()
                        if recovered:
                            self._stop_process(process)
                            process = self._start_process()
                            pending_pcm.clear()
                            continue
                        raise RuntimeError(self._process_error_message(process))
                    continue
                pcm_chunk = self._read_stdout_chunk(stdout_fd, self._frame_bytes - len(pending_pcm))
                if not pcm_chunk:
                    if process.poll() is not None:
                        recovered = False
                        if not sent_any_frame and not capture_recovery_attempted:
                            capture_recovery_attempted = True
                            recovered = self._recover_transient_respeaker_capture()
                        if recovered:
                            self._stop_process(process)
                            process = self._start_process()
                            pending_pcm.clear()
                            continue
                        raise RuntimeError(self._process_error_message(process))
                    continue
                pending_pcm.extend(pcm_chunk)
                while len(pending_pcm) >= self._frame_bytes:
                    frame_bytes = bytes(pending_pcm[: self._frame_bytes])
                    del pending_pcm[: self._frame_bytes]
                    self._send_frame(frame_bytes)
                    sent_any_frame = True
            if process is not None:
                self._drain_stderr(process)
        except Exception as exc:
            if not self._stop_event.is_set():
                self._trace_event(
                    "voice_edge_capture_failed",
                    kind="exception",
                    level="ERROR",
                    details={"error_type": type(exc).__name__},
                )
                self.emit(f"voice_orchestrator_capture_failed={type(exc).__name__}")
        finally:
            if process is not None:
                self._stop_process(process)
            self._flush_sent_frame_bucket()
            with self._lifecycle_lock:
                if self._process is process:
                    self._process = None
                if self._thread is current_thread():
                    self._thread = None
            if started:
                self.emit("voice_orchestrator_capture=stopped")

    def _send_frame(self, pcm_bytes: bytes) -> None:
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
        try:
            with self._state_lock:
                latest_runtime_state = self._last_runtime_state
            self._client.send_audio_frame(
                OrchestratorVoiceAudioFrame(
                    sequence=self._sequence,
                    pcm_bytes=pcm_bytes,
                    runtime_state=latest_runtime_state,
                )
            )
            self._skipped_frame_count = 0
            self._sent_frame_bucket.add_frame(sequence=self._sequence, pcm_bytes=pcm_bytes)
            if self._sent_frame_bucket.should_flush():
                self._flush_sent_frame_bucket()
            self._sequence += 1
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

    def _handle_server_event(self, event) -> None:
        if isinstance(event, OrchestratorVoiceReadyEvent):
            self._ready_backend = str(event.backend or "").strip().lower() or None
            self._trace_event(
                "voice_edge_server_ready",
                kind="io",
                details={"backend": self._ready_backend},
            )
            self.emit(f"voice_orchestrator_ready={event.backend}")
            self._connected = True
            self._next_reconnect_at = 0.0
            return
        if isinstance(event, OrchestratorVoiceWakeConfirmedEvent):
            self._trace_event(
                "voice_edge_server_wake_confirmed",
                kind="decision",
                details={
                    "matched_phrase": event.matched_phrase,
                    "remaining_text_chars": len(str(event.remaining_text or "").strip()),
                },
            )
            self.emit(f"voice_orchestrator_wake_confirmed={event.matched_phrase or 'unknown'}")
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
            self._trace_event(
                "voice_edge_server_transcript_committed",
                kind="decision",
                details={"source": event.source, "transcript_chars": len(str(event.transcript or "").strip())},
            )
            self.emit(f"voice_orchestrator_transcript_committed={event.source}")
            self._on_transcript_committed(event.transcript, event.source)
            return
        if isinstance(event, OrchestratorVoiceFollowUpClosedEvent):
            self._trace_event(
                "voice_edge_server_follow_up_closed",
                kind="mutation",
                details={"reason": event.reason},
            )
            self.emit(f"voice_orchestrator_follow_up_closed={event.reason}")
            if self._on_follow_up_closed is not None:
                self._on_follow_up_closed(event.reason)
            return
        if isinstance(event, OrchestratorVoiceBargeInInterruptEvent):
            self._trace_event(
                "voice_edge_server_barge_in",
                kind="decision",
                details={"transcript_preview_chars": len(str(event.transcript_preview or "").strip())},
            )
            self.emit("voice_orchestrator_barge_in_interrupt=true")
            self._on_barge_in_interrupt()
            return
        if isinstance(event, OrchestratorVoiceErrorEvent):
            self._trace_event(
                "voice_edge_server_error",
                kind="warning",
                level="WARN",
                details={"error": event.error},
            )
            self._mark_disconnected(
                emit_message=f"voice_orchestrator_error={event.error}",
                retry_delay_s=0.0,
            )
            return
        self.emit(f"voice_orchestrator_event={type(event).__name__}")

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
                "-t",
                "raw",
                "-f",
                "S16_LE",
                "-c",
                str(self._channels),
                "-r",
                str(self._sample_rate),
            ],
            env=build_audio_subprocess_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        if process.stdout is None or process.stderr is None:
            raise RuntimeError("Voice orchestrator capture process did not expose stdout/stderr")
        os.set_blocking(process.stdout.fileno(), False)
        os.set_blocking(process.stderr.fileno(), False)
        with self._lifecycle_lock:
            self._process = process
        self._trace_event("voice_edge_capture_started", kind="io", details={"frame_bytes": self._frame_bytes})
        self.emit("voice_orchestrator_capture=started")
        return process

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
        """Wait briefly for a transient XVF3800 re-enumeration before failing."""

        recovered = wait_for_transient_respeaker_capture_ready(
            device=self._device,
            sample_rate=self._sample_rate,
            channels=self._channels,
            chunk_ms=self._chunk_ms,
            should_stop=lambda: self._stop_event.is_set() or self._paused.is_set(),
        )
        if recovered:
            self.emit("voice_orchestrator_capture_recovered=respeaker_reenumeration")
        return recovered

    def _stop_process(self, process: subprocess.Popen[bytes]) -> None:
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=self._STOP_WAIT_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    process.kill()
                    try:
                        process.wait(timeout=self._STOP_WAIT_TIMEOUT_S)
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
