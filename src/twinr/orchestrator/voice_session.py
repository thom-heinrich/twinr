"""Own server-side voice-activation/session decisions for the voice path.

The edge Pi streams bounded PCM frames over the orchestrator websocket. This
module keeps the server-side logic focused on three responsibilities:

- detect a configured activation phrase on the remote ASR stream
- buffer one same-stream utterance until endpoint silence decides it is complete
- route that utterance as wake, transcript commit, or barge-in

It deliberately does not open local devices or mutate the Pi runtime directly.
"""

from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
import math
import time
from typing import Any, Protocol
from uuid import uuid4

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.workflows.forensics import WorkflowForensics
from twinr.hardware.audio import AmbientAudioCaptureWindow, AmbientAudioLevelSample, pcm16_signal_profile
from twinr.orchestrator.remote_asr import RemoteAsrBackendAdapter
from twinr.orchestrator.voice_activation import (
    VoiceActivationMatch,
    VoiceActivationPhraseMatcher,
)
from twinr.orchestrator.voice_audio_debug_store import VoiceAudioDebugArtifactStore
from twinr.orchestrator.voice_forensics import (
    VoiceFrameTelemetryBucket,
    prefixed_signal_profile_details,
)
from twinr.orchestrator.voice_transcript_debug_stream import VoiceTranscriptDebugStream
from twinr.orchestrator.voice_runtime_intent import VoiceRuntimeIntentContext

from .voice_contracts import (
    OrchestratorVoiceAudioFrame,
    OrchestratorVoiceBargeInInterruptEvent,
    OrchestratorVoiceFollowUpClosedEvent,
    OrchestratorVoiceHelloRequest,
    OrchestratorVoiceReadyEvent,
    OrchestratorVoiceRuntimeStateEvent,
    OrchestratorVoiceTranscriptCommittedEvent,
    OrchestratorVoiceWakeConfirmedEvent,
)


class _TranscriptBackend(Protocol):
    """Describe the minimal transcription surface the session needs."""

    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
        language: str | None = None,
        prompt: str | None = None,
    ) -> str:
        """Transcribe one bounded audio payload into plain text."""


@dataclass(slots=True)
class _RecentFrame:
    """Store one recent PCM fragment plus its bounded level metadata."""

    pcm_bytes: bytes
    rms: int
    duration_ms: int


@dataclass(slots=True)
class _PendingTranscriptUtterance:
    """Track one same-stream utterance for transcript-first wake/follow-up routing."""

    origin_state: str
    frames: deque[_RecentFrame] = field(default_factory=deque)
    captured_ms: int = 0
    max_capture_ms: int = 0
    active_ms: int = 0
    trailing_silence_ms: int = 0
    speech_active: bool = False


def _pcm16_rms(samples: bytes) -> int:
    """Return the root-mean-square amplitude for one PCM16 fragment."""

    if not samples:
        return 0
    usable = len(samples) - (len(samples) % 2)
    if usable <= 0:
        return 0
    import math as _math
    import sys as _sys
    from array import array

    pcm_samples = array("h")
    pcm_samples.frombytes(samples[:usable])
    if _sys.byteorder != "little":
        pcm_samples.byteswap()
    mean_square = sum(sample * sample for sample in pcm_samples) / len(pcm_samples)
    return int(_math.sqrt(mean_square))


def _normalize_text_length(text: str) -> int:
    """Return a simple transcript-length signal without regex product logic."""

    return len("".join(str(text or "").split()))


class EdgeOrchestratorVoiceSession:
    """Maintain one server-side streaming voice session."""

    _ACTIVE_STATES = frozenset({"waiting", "listening", "speaking", "follow_up_open"})
    _REMOTE_ASR_UTTERANCE_STATES = frozenset({"waiting", "listening"})
    _SUPPORTED_RUNTIME_STATES = frozenset({"waiting", "listening", "thinking", "speaking", "follow_up_open"})
    _WAITING_VISIBILITY_GRACE_S = 6.0
    _REMOTE_ASR_SPEECH_CONTINUE_RATIO = 0.35

    def __init__(
        self,
        config: TwinrConfig,
        *,
        backend: _TranscriptBackend | None = None,
        wake_phrase_spotter: VoiceActivationPhraseMatcher | None = None,
        monotonic_fn=time.monotonic,
    ) -> None:
        self.config = config
        self.sample_rate = int(config.audio_sample_rate)
        self.channels = int(config.audio_channels)
        self.chunk_ms = max(20, int(getattr(config, "audio_chunk_ms", 100) or 100))
        self.history_ms = max(
            1500,
            int(getattr(config, "voice_orchestrator_history_ms", 4000) or 4000),
        )
        self.wake_candidate_window_ms = max(
            self.chunk_ms,
            int(getattr(config, "voice_orchestrator_wake_candidate_window_ms", 2200) or 2200),
        )
        wake_candidate_min_active_ratio = getattr(
            config,
            "voice_orchestrator_wake_candidate_min_active_ratio",
            None,
        )
        if wake_candidate_min_active_ratio is None:
            wake_candidate_min_active_ratio = 0.0
        self.wake_candidate_min_active_ratio = max(
            0.0,
            float(wake_candidate_min_active_ratio),
        )
        self.wake_candidate_min_transcript_chars = max(
            1,
            int(
                getattr(config, "voice_orchestrator_wake_candidate_min_transcript_chars", 4) or 4
            ),
        )
        self.remote_asr_stage1_window_ms = max(
            self.chunk_ms,
            min(self.history_ms, 1000),
        )
        self.remote_asr_min_wake_duration_ms = max(
            self.chunk_ms,
            min(
                self.remote_asr_stage1_window_ms,
                int(
                    getattr(
                        config,
                        "voice_orchestrator_remote_asr_min_wake_duration_ms",
                        300,
                    )
                    or 0
                ),
            ),
        )
        self.intent_stage1_window_bonus_ms = max(
            0,
            int(
                getattr(config, "voice_orchestrator_intent_stage1_window_bonus_ms", 400)
                or 400
            ),
        )
        self.intent_min_wake_duration_relief_ms = max(
            0,
            int(
                getattr(
                    config,
                    "voice_orchestrator_intent_min_wake_duration_relief_ms",
                    100,
                )
                or 100
            ),
        )
        self.wake_tail_max_ms = max(
            self.chunk_ms,
            int(getattr(config, "voice_orchestrator_wake_tail_max_ms", 2200) or 2200),
        )
        self.wake_tail_endpoint_silence_ms = max(
            self.chunk_ms,
            int(
                getattr(config, "voice_orchestrator_wake_tail_endpoint_silence_ms", 300)
                or 300
            ),
        )
        self.follow_up_timeout_s = max(
            1.0,
            float(getattr(config, "voice_orchestrator_follow_up_timeout_s", 6.0) or 6.0),
        )
        self.intent_follow_up_timeout_bonus_s = max(
            0.0,
            float(
                getattr(config, "voice_orchestrator_intent_follow_up_timeout_bonus_s", 1.5)
                or 1.5
            ),
        )
        self.follow_up_window_ms = max(
            self.chunk_ms,
            int(getattr(config, "voice_orchestrator_follow_up_window_ms", 900) or 900),
        )
        self.follow_up_min_active_ratio = max(
            0.05,
            float(getattr(config, "voice_orchestrator_follow_up_min_active_ratio", 0.22) or 0.22),
        )
        self.follow_up_min_transcript_chars = max(
            1,
            int(getattr(config, "voice_orchestrator_follow_up_min_transcript_chars", 4) or 4),
        )
        self.barge_in_window_ms = max(
            self.chunk_ms,
            int(getattr(config, "voice_orchestrator_barge_in_window_ms", 850) or 850),
        )
        self.barge_in_min_active_ratio = max(
            0.05,
            float(getattr(config, "voice_orchestrator_barge_in_min_active_ratio", 0.28) or 0.28),
        )
        self.barge_in_min_transcript_chars = max(
            1,
            int(getattr(config, "voice_orchestrator_barge_in_min_transcript_chars", 4) or 4),
        )
        self.candidate_cooldown_s = max(
            0.25,
            float(getattr(config, "voice_orchestrator_candidate_cooldown_s", 0.9) or 0.9),
        )
        # Transcript-first wake scans cannot share the heavier generic
        # candidate throttle used for follow-up / barge-in checks, but they
        # also cannot issue multiple synchronous STT decodes per 100 ms frame
        # without backing the websocket stream up into multi-second latency.
        # Keep a dedicated wake cadence and apply it after each completed scan.
        self._wake_candidate_cooldown_s = self.chunk_ms / 1000.0
        self.speech_threshold = int(config.audio_speech_threshold)
        self._monotonic = monotonic_fn
        self._history: deque[_RecentFrame] = deque(maxlen=max(8, math.ceil(self.history_ms / self.chunk_ms)))
        self._next_wake_candidate_check_at: float = 0.0
        self._next_barge_in_candidate_check_at: float = 0.0
        self._barge_in_sent = False
        self._session_id = ""
        self._state = "waiting"
        self._follow_up_allowed = False
        self._runtime_state_attested = False
        self._last_waiting_visible_at: float | None = None
        self._pending_transcript_utterance: _PendingTranscriptUtterance | None = None
        self._follow_up_deadline_at: float | None = None
        self._follow_up_opened_at: float | None = None
        self._intent_context = VoiceRuntimeIntentContext()
        self.backend = backend or _build_transcript_backend(config)
        self._wake_phrase_spotter = wake_phrase_spotter or _build_wake_phrase_spotter(
            config,
            backend=self.backend,
        )
        self._audio_debug_store = VoiceAudioDebugArtifactStore.from_config(config)
        self._transcript_debug_stream = VoiceTranscriptDebugStream.from_config(config)
        self._forensics: WorkflowForensics | None = None
        self._trace_id: str = uuid4().hex
        self._received_frame_bucket = VoiceFrameTelemetryBucket(
            chunk_ms=self.chunk_ms,
            speech_threshold=self.speech_threshold,
        )

    def set_forensics(
        self,
        tracer: WorkflowForensics | None,
        *,
        trace_id: str | None = None,
    ) -> None:
        """Bind one shared forensic tracer for this websocket voice session."""

        if isinstance(tracer, WorkflowForensics) and tracer.enabled:
            self._forensics = tracer
        else:
            self._forensics = None
        configure_backend_forensics = getattr(self.backend, "set_forensics", None)
        if callable(configure_backend_forensics):
            configure_backend_forensics(self._forensics)
        if trace_id:
            self._trace_id = str(trace_id)

    def _trace_details(self, details: dict[str, object] | None = None) -> dict[str, object]:
        payload: dict[str, object] = {
            "session_id": self._session_id or None,
            "state": self._state,
            "backend": self.backend_name,
            "follow_up_allowed": self._follow_up_allowed,
        }
        payload.update(self._intent_context.trace_details())
        if details:
            payload.update(details)
        return payload

    def _trace_event(
        self,
        msg: str,
        *,
        kind: str,
        details: dict[str, object] | None = None,
        reason: dict[str, object] | None = None,
        kpi: dict[str, object] | None = None,
        level: str = "INFO",
    ) -> None:
        tracer = self._forensics
        if not isinstance(tracer, WorkflowForensics):
            return
        tracer.event(
            kind=kind,
            msg=msg,
            details=self._trace_details(details),
            reason=reason,
            kpi=kpi,
            level=level,
            trace_id=self._trace_id,
            loc_skip=2,
        )

    def _trace_decision(
        self,
        msg: str,
        *,
        question: str,
        selected: dict[str, object],
        options: list[dict[str, object]],
        context: dict[str, object] | None = None,
        confidence: object | None = None,
        guardrails: list[str] | None = None,
        kpi_impact_estimate: dict[str, object] | None = None,
    ) -> None:
        tracer = self._forensics
        if not isinstance(tracer, WorkflowForensics):
            return
        tracer.decision(
            msg=msg,
            question=question,
            selected=selected,
            options=options,
            context=self._trace_details(context),
            confidence=confidence,
            guardrails=guardrails,
            kpi_impact_estimate=kpi_impact_estimate,
            trace_id=self._trace_id,
        )

    def _trace_span(self, *, name: str, kind: str, details: dict[str, object] | None = None):
        tracer = self._forensics
        if not isinstance(tracer, WorkflowForensics):
            return nullcontext()
        return tracer.span(
            name=name,
            kind=kind,
            details=self._trace_details(details),
            trace_id=self._trace_id,
        )

    def _capture_sample_details(
        self,
        capture: AmbientAudioCaptureWindow | None,
    ) -> dict[str, object]:
        """Return compact capture metrics for transcript debug entries."""

        if capture is None:
            return {}
        sample = capture.sample
        signal_profile = pcm16_signal_profile(capture.pcm_bytes)
        return {
            "duration_ms": int(sample.duration_ms),
            "chunk_count": int(sample.chunk_count),
            "active_chunk_count": int(sample.active_chunk_count),
            "average_rms": int(sample.average_rms),
            "peak_rms": int(sample.peak_rms),
            "active_ratio": round(float(sample.active_ratio), 6),
            **prefixed_signal_profile_details(signal_profile, prefix="signal"),
        }

    def _record_transcript_debug(
        self,
        *,
        stage: str,
        outcome: str,
        transcript: str | None = None,
        matched_phrase: str | None = None,
        remaining_text: str | None = None,
        detector_label: str | None = None,
        score: float | None = None,
        capture: AmbientAudioCaptureWindow | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        """Persist one raw transcript/debug record for the live gateway."""

        resolved_details = self._intent_context.trace_details()
        if details:
            resolved_details.update(details)
        if capture is not None:
            audio_artifact = self._audio_debug_store.persist_capture(
                capture=capture,
                session_id=self._session_id or None,
                trace_id=self._trace_id,
                stage=stage,
                outcome=outcome,
            )
            if audio_artifact:
                resolved_details.update(audio_artifact)
        self._transcript_debug_stream.append_entry(
            session_id=self._session_id or None,
            trace_id=self._trace_id,
            state=self._state,
            backend=self.backend_name,
            stage=stage,
            outcome=outcome,
            transcript=transcript,
            matched_phrase=matched_phrase,
            remaining_text=remaining_text,
            detector_label=detector_label,
            score=score,
            sample=self._capture_sample_details(capture),
            details=resolved_details,
        )

    def _detect_wake_capture(
        self,
        *,
        capture: AmbientAudioCaptureWindow,
        stage: str,
        details: dict[str, object] | None = None,
    ) -> VoiceActivationMatch | None:
        """Run one activation-detector pass and surface backend errors as buffered evidence."""

        origin_state_value = None if details is None else details.get("origin_state")
        origin_state = str(origin_state_value).strip() or None
        try:
            with self._backend_request_context(
                stage=stage,
                capture=capture,
                origin_state=origin_state,
            ):
                return self._wake_phrase_spotter.detect(capture)
        except Exception as exc:
            error_message = str(exc).strip()
            resolved_details: dict[str, object] = {
                "error_type": type(exc).__name__,
            }
            if error_message:
                resolved_details["error_message"] = error_message[:240]
            if details:
                resolved_details.update(details)
            self._record_transcript_debug(
                stage=stage,
                outcome="backend_error",
                capture=capture,
                details=resolved_details,
            )
            self._trace_event(
                "voice_activation_backend_error",
                kind="warning",
                level="WARN",
                details={
                    "stage": stage,
                    **resolved_details,
                },
            )
            return None

    @property
    def backend_name(self) -> str:
        """Return the selected streaming remote-ASR backend label."""

        return "remote_asr"

    def close(self) -> None:
        """Reset any buffered streaming state."""

        self._flush_received_frame_bucket()
        self._history.clear()
        self._pending_transcript_utterance = None
        self._follow_up_deadline_at = None
        self._follow_up_opened_at = None
        self._last_waiting_visible_at = None

    def handle_hello(self, request: OrchestratorVoiceHelloRequest) -> list[dict[str, Any]]:
        """Accept one new edge voice session and validate stream metadata."""

        self._session_id = request.session_id
        self._trace_id = str(request.trace_id or request.session_id or uuid4().hex).strip() or uuid4().hex
        self._received_frame_bucket = VoiceFrameTelemetryBucket(
            chunk_ms=self.chunk_ms,
            speech_threshold=self.speech_threshold,
        )
        if int(request.sample_rate) != self.sample_rate:
            return [
                {
                    "type": "voice_error",
                    "error": (
                        "Voice session sample rate mismatch. "
                        f"Expected {self.sample_rate} Hz but received {request.sample_rate} Hz."
                    ),
                }
            ]
        if int(request.channels) != self.channels:
            return [
                {
                    "type": "voice_error",
                    "error": (
                        "Voice session channel mismatch. "
                        f"Expected {self.channels} channel(s) but received {request.channels}."
                    ),
                }
            ]
        raw_initial_state = request.initial_state or "waiting"
        self._state = self._normalize_runtime_state(raw_initial_state)
        self._follow_up_allowed = bool(getattr(request, "follow_up_allowed", False))
        self._runtime_state_attested = bool(getattr(request, "state_attested", False))
        self._intent_context = VoiceRuntimeIntentContext.from_runtime_event(request)
        if self._state == "follow_up_open" and self._follow_up_allowed:
            now = self._monotonic()
            self._follow_up_opened_at = now
            self._follow_up_deadline_at = now + self._effective_follow_up_timeout_s()
        else:
            self._follow_up_deadline_at = None
            self._follow_up_opened_at = None
        self._refresh_waiting_visibility_anchor()
        self._trace_event(
            "voice_session_hello_accepted",
            kind="run_start",
            details={
                "sample_rate": request.sample_rate,
                "channels": request.channels,
                "chunk_ms": request.chunk_ms,
                "initial_state": self._state,
                "raw_initial_state": raw_initial_state,
                "detail": getattr(request, "detail", None),
                "follow_up_allowed": self._follow_up_allowed,
                "state_attested": self._runtime_state_attested,
                **self._intent_context.trace_details(),
            },
        )
        return [OrchestratorVoiceReadyEvent(session_id=request.session_id, backend=self.backend_name).to_payload()]

    def handle_runtime_state(self, event: OrchestratorVoiceRuntimeStateEvent) -> list[dict[str, Any]]:
        """Update explicit edge runtime state and drain any timeout-based events."""

        return self._apply_runtime_state(
            event,
            trace_event_name="voice_runtime_state_received",
            trace_kind="mutation",
        )

    def _apply_runtime_state(
        self,
        event: OrchestratorVoiceRuntimeStateEvent,
        *,
        trace_event_name: str,
        trace_kind: str,
    ) -> list[dict[str, Any]]:
        """Apply one runtime-state snapshot and drain any timeout-based events."""

        previous_state = self._state
        raw_state = event.state or "waiting"
        self._state = self._normalize_runtime_state(raw_state)
        self._follow_up_allowed = bool(event.follow_up_allowed)
        self._runtime_state_attested = True
        self._intent_context = VoiceRuntimeIntentContext.from_runtime_event(event)
        if self._state == "follow_up_open" and self._follow_up_allowed:
            now = self._monotonic()
            if previous_state != "follow_up_open" or self._follow_up_opened_at is None:
                self._follow_up_opened_at = now
            self._follow_up_deadline_at = self._follow_up_opened_at + self._effective_follow_up_timeout_s()
        else:
            self._follow_up_deadline_at = None
            self._follow_up_opened_at = None
        self._refresh_waiting_visibility_anchor()
        if previous_state != self._state and self._uses_remote_asr_utterance_path():
            self._reset_remote_asr_utterance_history(
                previous_state=previous_state,
                detail=event.detail,
            )
        if self._state != "speaking":
            self._barge_in_sent = False
        self._cancel_blocked_waiting_activation_buffers(
            previous_state=previous_state,
            detail=event.detail,
        )
        if not self._uses_remote_asr_utterance_path():
            self._pending_transcript_utterance = None
        if self._state not in self._ACTIVE_STATES:
            self._pending_transcript_utterance = None
        self._trace_event(
            trace_event_name,
            kind=trace_kind,
            details={
                "previous_state": previous_state,
                "new_state": self._state,
                "raw_state": raw_state,
                "detail": event.detail,
                "follow_up_allowed": self._follow_up_allowed,
                "state_attested": self._runtime_state_attested,
                **self._intent_context.trace_details(),
            },
        )
        return self._drain_timeouts()

    def _runtime_state_event_matches_current(self, event: OrchestratorVoiceRuntimeStateEvent) -> bool:
        """Return whether one incoming runtime snapshot matches current session state."""

        if not self._runtime_state_attested:
            return False
        if self._normalize_runtime_state(event.state or "waiting") != self._state:
            return False
        if bool(event.follow_up_allowed) != self._follow_up_allowed:
            return False
        return VoiceRuntimeIntentContext.from_runtime_event(event) == self._intent_context

    def _normalize_runtime_state(self, state: str | None) -> str:
        """Map retired runtime-state labels onto the supported remote-only set."""

        normalized = str(state or "").strip() or "waiting"
        if normalized == "wake_armed":
            return "waiting"
        if normalized in self._SUPPORTED_RUNTIME_STATES:
            return normalized
        return normalized

    def _reset_remote_asr_utterance_history(
        self,
        *,
        previous_state: str,
        detail: str | None,
    ) -> None:
        """Drop stale stream history before a fresh remote utterance window starts.

        The remote transcript-first path keeps the Pi microphone stream open
        across wake acknowledgements and answer playback. When Twinr explicitly
        re-enters ``listening`` or ``follow_up_open``, the next utterance should
        start from a clean server-side buffer instead of reusing old wake/beep/
        answer audio that can poison the first ASR commit window.
        """

        preserved_pending = self._preserve_pending_waiting_utterance_on_listening_handoff(
            previous_state=previous_state,
            detail=detail,
        )
        self._history.clear()
        self._pending_transcript_utterance = preserved_pending
        self._trace_event(
            "voice_remote_asr_utterance_history_reset",
            kind="mutation",
            details={
                "previous_state": previous_state,
                "new_state": self._state,
                "detail": detail,
                "preserved_pending_utterance": preserved_pending is not None,
                "preserved_pending_active_ms": (
                    preserved_pending.active_ms if preserved_pending is not None else 0
                ),
                "preserved_pending_captured_ms": (
                    preserved_pending.captured_ms if preserved_pending is not None else 0
                ),
            },
        )

    def _preserve_pending_waiting_utterance_on_listening_handoff(
        self,
        *,
        previous_state: str,
        detail: str | None,
    ) -> _PendingTranscriptUtterance | None:
        """Carry a just-started waiting utterance across the listen-beep handoff.

        The Pi intentionally sends the explicit ``listening`` runtime state only
        after the listen earcon has finished so the fresh server-side window can
        drop that playback audio. A user can still start speaking during that
        short earcon gap, which means the same live stream may already have one
        active waiting-origin utterance in flight by the time the state flips.
        Preserve only that active overlap case; older completed waiting audio
        must still be discarded when the explicit listening window opens.
        """

        pending = self._pending_transcript_utterance
        if pending is None:
            return None
        if previous_state != "waiting" or self._state != "listening":
            return None
        if pending.origin_state != "waiting":
            return None
        if pending.active_ms <= 0:
            return None
        if pending.trailing_silence_ms >= self.wake_tail_endpoint_silence_ms:
            return None
        carried_frames = self._latest_active_speech_burst_frames(tuple(pending.frames))
        if not carried_frames:
            return None
        promoted = self._pending_transcript_utterance_from_frames(
            origin_state="listening",
            frames=carried_frames,
            max_capture_ms=pending.max_capture_ms,
        )
        self._trace_event(
            "voice_remote_asr_listening_handoff_preserved",
            kind="mutation",
            details={
                "previous_state": previous_state,
                "new_state": self._state,
                "detail": detail,
                "carried_frame_count": len(carried_frames),
                "carried_active_ms": promoted.active_ms,
                "carried_captured_ms": promoted.captured_ms,
            },
        )
        return promoted

    def _cancel_blocked_waiting_activation_buffers(
        self,
        *,
        previous_state: str,
        detail: str | None,
    ) -> None:
        """Drop buffered waiting/wake audio once live context explicitly blocks speech.

        A websocket reconnect can briefly leave the gateway on an unknown
        context before the Pi replays a full runtime-state update. If room/TV
        audio opens a pending waiting utterance during that gap, the gateway
        must not keep pushing that same buffered audio into transcript-first STT
        after the current person-state later says speech is not allowed.
        """

        if self._state != "waiting":
            return
        if self._waiting_activation_allowed():
            return
        if self._pending_transcript_utterance is None:
            return
        self._history.clear()
        self._pending_transcript_utterance = None
        details = {
            "previous_state": previous_state,
            "new_state": self._state,
            "detail": detail,
            **self._intent_context.trace_details(),
        }
        self._record_transcript_debug(
            stage="activation_utterance",
            outcome="cancelled_context_blocked",
            details=details,
        )
        self._trace_event(
            "voice_waiting_activation_cancelled_context_blocked",
            kind="mutation",
            details=details,
        )

    def handle_audio_frame(self, frame: OrchestratorVoiceAudioFrame) -> list[dict[str, Any]]:
        """Process one streamed PCM frame and emit any server-side decisions."""

        events = self._drain_timeouts()
        if (
            isinstance(frame.runtime_state, OrchestratorVoiceRuntimeStateEvent)
            and not self._runtime_state_event_matches_current(frame.runtime_state)
        ):
            events.extend(
                self._apply_runtime_state(
                    frame.runtime_state,
                    trace_event_name="voice_runtime_state_received_audio_frame",
                    trace_kind="branch",
                )
            )
        pcm_bytes = bytes(frame.pcm_bytes or b"")
        if not pcm_bytes:
            return events
        self._received_frame_bucket.add_frame(sequence=int(frame.sequence), pcm_bytes=pcm_bytes)
        if self._received_frame_bucket.should_flush():
            self._flush_received_frame_bucket()
        self._remember_frame(pcm_bytes)
        if self._uses_remote_asr_utterance_path():
            utterance_event = self._advance_remote_asr_utterance()
            if utterance_event is not None:
                events.append(utterance_event.to_payload())
            return events
        if self._state == "speaking" and not self._barge_in_sent:
            barge_in_event = self._maybe_detect_barge_in_candidate()
            if barge_in_event is not None:
                self._barge_in_sent = True
                self._state = "thinking"
                events.append(barge_in_event.to_payload())
        return events

    def _uses_remote_asr_utterance_path(self) -> bool:
        """Return whether the same-stream remote ASR utterance scanner owns routing."""

        if self._state in self._REMOTE_ASR_UTTERANCE_STATES:
            return True
        return self._state == "follow_up_open" and self._follow_up_allowed

    def _intent_audio_bias_active(self) -> bool:
        """Return whether compact multimodal context may relax audio-owned gates."""

        return self._intent_context.audio_bias_allowed()

    def _refresh_waiting_visibility_anchor(self) -> None:
        """Remember the last attested visible waiting context.

        Camera/person-state refreshes can briefly drop ``person_visible`` while
        the same person is still standing at the device and speaking the wake
        phrase. Keep one short grace anchor so the server-owned audio wake path
        does not throw away an in-flight wake burst on those transient dips.
        """

        if not self._runtime_state_attested:
            return
        if self._state != "waiting":
            self._last_waiting_visible_at = None
            return
        if self._intent_context.person_visible is True:
            self._last_waiting_visible_at = self._monotonic()

    def _waiting_visibility_grace_active(self) -> bool:
        """Return whether a recent visible waiting context is still fresh."""

        if self._state != "waiting":
            return False
        last_visible_at = self._last_waiting_visible_at
        if last_visible_at is None:
            return False
        return (self._monotonic() - last_visible_at) <= self._WAITING_VISIBILITY_GRACE_S

    def _waiting_activation_allowed(self) -> bool:
        """Return whether idle transcript-first scanning may open a new utterance."""

        if self._state != "waiting":
            return True
        if not self._runtime_state_attested:
            return False
        if self._intent_context.waiting_activation_allowed():
            return True
        return self._waiting_visibility_grace_active()

    def _effective_remote_asr_stage1_window_ms(self) -> int:
        """Return the current transcript-first stage-one scan window."""

        if not self._intent_audio_bias_active():
            return self.remote_asr_stage1_window_ms
        return min(
            self.history_ms,
            self.remote_asr_stage1_window_ms + self.intent_stage1_window_bonus_ms,
        )

    def _effective_remote_asr_min_activation_duration_ms(self) -> int:
        """Return the bounded minimum activation duration for the current context."""

        if not self._intent_audio_bias_active():
            if self._waiting_visibility_grace_active():
                return max(
                    self.chunk_ms,
                    self.remote_asr_min_wake_duration_ms - self.intent_min_wake_duration_relief_ms,
                )
            return self.remote_asr_min_wake_duration_ms
        return max(
            self.chunk_ms,
            self.remote_asr_min_wake_duration_ms - self.intent_min_wake_duration_relief_ms,
        )

    def _effective_follow_up_timeout_s(self) -> float:
        """Return the bounded follow-up timeout for the current multimodal context."""

        if not self._intent_audio_bias_active():
            return self.follow_up_timeout_s
        return self.follow_up_timeout_s + self.intent_follow_up_timeout_bonus_s

    def _remote_asr_speech_continue_threshold(self) -> int:
        """Return the bounded lower threshold for one already-open speech burst."""

        return max(
            1,
            min(
                self.speech_threshold,
                int(round(self.speech_threshold * self._REMOTE_ASR_SPEECH_CONTINUE_RATIO)),
            ),
        )

    def _frame_counts_as_remote_asr_speech(
        self,
        frame: _RecentFrame,
        *,
        continuing: bool = False,
    ) -> bool:
        """Treat speech bursts with bounded hysteresis for wake-duration accounting.

        Quiet onset preservation is handled separately by
        ``_latest_active_speech_burst_frames()``, which keeps a bounded nonzero
        pre-roll ahead of the first threshold-crossing frame. Counting every
        nonzero frame here lets far-field hiss or room bed-noise satisfy the
        minimum wake duration and opens long empty 4.4 s captures that later
        reach the ASR backend as ``no_match`` with empty transcripts.

        Real wake phrases can dip far below their peak frame within one spoken
        burst. Once a burst has already crossed the conservative threshold,
        keep it alive down to one lower continuation threshold so quiet
        follow-through syllables still count without reintroducing the old
        "any nonzero frame is speech" bug.
        """

        if frame.rms >= self.speech_threshold:
            return True
        return continuing and frame.rms >= self._remote_asr_speech_continue_threshold()

    def _remote_asr_speech_flags(
        self,
        frames: tuple[_RecentFrame, ...],
    ) -> tuple[bool, ...]:
        """Return one forward hysteresis speech mask for the provided frames."""

        flags: list[bool] = []
        continuing = False
        for frame in frames:
            continuing = self._frame_counts_as_remote_asr_speech(
                frame,
                continuing=continuing,
            )
            flags.append(continuing)
        return tuple(flags)

    def _pending_utterance_details(
        self,
        pending: _PendingTranscriptUtterance,
    ) -> dict[str, int]:
        """Expose compact buffered-utterance metrics for transcript debug traces."""

        return {
            "pending_captured_ms": int(pending.captured_ms),
            "pending_active_ms": int(pending.active_ms),
            "pending_trailing_silence_ms": int(pending.trailing_silence_ms),
        }

    def _remember_frame(self, pcm_bytes: bytes) -> None:
        duration_ms = max(
            self.chunk_ms,
            int(round((len(pcm_bytes) / max(1, self.channels * 2 * self.sample_rate)) * 1000.0)),
        )
        self._history.append(
            _RecentFrame(
                pcm_bytes=pcm_bytes,
                rms=_pcm16_rms(pcm_bytes),
                duration_ms=duration_ms,
            )
        )

    def _flush_received_frame_bucket(self) -> None:
        """Persist one bounded summary of recently received websocket frames."""

        if not self._received_frame_bucket.has_data():
            return
        self._trace_event(
            "voice_server_frame_window_received",
            kind="io",
            details=self._received_frame_bucket.flush_details(),
        )

    def _backend_request_context(
        self,
        *,
        stage: str,
        capture: AmbientAudioCaptureWindow,
        origin_state: str | None = None,
    ):
        """Expose compact capture metadata to the remote-ASR client adapter."""

        bind_context = getattr(self.backend, "bind_request_context", None)
        if not callable(bind_context):
            return nullcontext()
        signal_profile = pcm16_signal_profile(capture.pcm_bytes)
        return bind_context(
            {
                "session_id": self._session_id,
                "trace_id": self._trace_id,
                "stage": stage,
                "state": self._state,
                "origin_state": origin_state,
                "capture_duration_ms": int(capture.sample.duration_ms),
                "capture_average_rms": int(capture.sample.average_rms),
                "capture_peak_rms": int(capture.sample.peak_rms),
                "capture_active_ratio": round(float(capture.sample.active_ratio), 6),
                **prefixed_signal_profile_details(signal_profile, prefix="capture_signal"),
            }
        )

    def _drain_timeouts(self) -> list[dict[str, Any]]:
        if self._state != "follow_up_open" or self._follow_up_deadline_at is None:
            return []
        if self._monotonic() < self._follow_up_deadline_at:
            return []
        self._follow_up_deadline_at = None
        self._follow_up_opened_at = None
        self._state = "waiting"
        return [OrchestratorVoiceFollowUpClosedEvent(reason="timeout").to_payload()]

    def _advance_remote_asr_utterance(
        self,
    ) -> OrchestratorVoiceWakeConfirmedEvent | OrchestratorVoiceTranscriptCommittedEvent | None:
        """Drive one same-stream utterance until endpointing decides it is complete."""

        latest_frame = self._history[-1] if self._history else None
        if latest_frame is None:
            return None
        if not self._runtime_state_attested:
            return None
        pending = self._pending_transcript_utterance
        if pending is None:
            if not self._waiting_activation_allowed():
                return None
            if not self._frame_counts_as_remote_asr_speech(latest_frame):
                return None
            pending = self._new_pending_transcript_utterance(origin_state=self._state)
            self._pending_transcript_utterance = pending
            if pending.active_ms < self._effective_remote_asr_min_activation_duration_ms():
                self._record_transcript_debug(
                    stage="activation_utterance",
                    outcome="buffering_short_utterance",
                    capture=self._pending_transcript_capture_window(pending),
                    details={
                        "origin_state": pending.origin_state,
                        "active_ms": pending.active_ms,
                        "required_active_ms": self._effective_remote_asr_min_activation_duration_ms(),
                        **self._pending_utterance_details(pending),
                        **self._intent_context.trace_details(),
                    },
                )
            return None
        if pending.origin_state == "waiting" and not self._waiting_activation_allowed():
            self._cancel_blocked_waiting_activation_buffers(
                previous_state=pending.origin_state,
                detail="runtime_context_blocked",
            )
            return None
        self._append_pending_frame(pending, latest_frame)
        pending.speech_active = self._frame_counts_as_remote_asr_speech(
            latest_frame,
            continuing=pending.speech_active,
        )
        if pending.speech_active:
            pending.active_ms += latest_frame.duration_ms
            pending.trailing_silence_ms = 0
        else:
            pending.trailing_silence_ms += latest_frame.duration_ms
        should_finalize = False
        if pending.captured_ms >= pending.max_capture_ms:
            should_finalize = True
        elif pending.active_ms > 0 and pending.trailing_silence_ms >= self.wake_tail_endpoint_silence_ms:
            should_finalize = True
        if not should_finalize:
            return None
        self._pending_transcript_utterance = None
        capture = self._pending_transcript_capture_window(pending)
        if pending.active_ms < self._effective_remote_asr_min_activation_duration_ms():
            self._record_transcript_debug(
                stage="activation_utterance",
                outcome="rejected_short_utterance",
                capture=capture,
                details={
                    "origin_state": pending.origin_state,
                    "active_ms": pending.active_ms,
                    "required_active_ms": self._effective_remote_asr_min_activation_duration_ms(),
                    **self._pending_utterance_details(pending),
                    **self._intent_context.trace_details(),
                },
            )
            return None
        match = self._detect_wake_capture(
            capture=capture,
            stage="activation_utterance",
            details={
                "origin_state": pending.origin_state,
                **self._pending_utterance_details(pending),
            },
        )
        if match is None:
            return None
        outcome = "matched" if match.detected else "no_match"
        self._record_transcript_debug(
            stage="activation_utterance",
            outcome=outcome,
            transcript=match.transcript,
            matched_phrase=match.matched_phrase,
            remaining_text=match.remaining_text,
            detector_label=match.detector_label,
            score=match.score,
            capture=capture,
            details={
                "origin_state": pending.origin_state,
                **self._pending_utterance_details(pending),
            },
        )
        if match.detected:
            self._state = "thinking"
            self._follow_up_deadline_at = None
            self._follow_up_opened_at = None
            self._trace_event(
                "voice_remote_asr_utterance_matched",
                kind="decision",
                details={
                    "origin_state": pending.origin_state,
                    "matched_phrase": match.matched_phrase,
                    "remaining_text_chars": len(str(match.remaining_text or "").strip()),
                    "transcript_chars": len(str(match.transcript or "").strip()),
                },
            )
            return OrchestratorVoiceWakeConfirmedEvent(
                matched_phrase=match.matched_phrase,
                remaining_text=str(match.remaining_text or "").strip(),
                backend=match.backend or self.backend_name,
                detector_label=match.detector_label,
                score=match.score,
            )
        transcript = str(match.transcript or "").strip()
        if pending.origin_state == "follow_up_open" and _normalize_text_length(transcript) >= self.follow_up_min_transcript_chars:
            self._state = "waiting"
            self._follow_up_deadline_at = None
            self._follow_up_opened_at = None
            self._record_transcript_debug(
                stage="activation_utterance",
                outcome="follow_up_committed",
                transcript=transcript,
                capture=capture,
                details={"origin_state": pending.origin_state},
            )
            self._trace_event(
                "voice_remote_asr_follow_up_committed",
                kind="decision",
                details={
                    "transcript_chars": len(transcript),
                    "transcript_preview": transcript[:80],
                },
            )
            return OrchestratorVoiceTranscriptCommittedEvent(
                transcript=transcript,
                source="follow_up",
            )
        if pending.origin_state == "listening" and _normalize_text_length(transcript) >= self.follow_up_min_transcript_chars:
            self._state = "thinking"
            self._record_transcript_debug(
                stage="activation_utterance",
                outcome="listening_committed",
                transcript=transcript,
                capture=capture,
                details={"origin_state": pending.origin_state},
            )
            return OrchestratorVoiceTranscriptCommittedEvent(
                transcript=transcript,
                source="listening",
            )
        return None

    def _new_pending_transcript_utterance(self, *, origin_state: str) -> _PendingTranscriptUtterance:
        """Seed one same-stream utterance from the latest active speech burst."""

        seed_frames = self._latest_active_speech_burst_frames(self._recent_frames_window(self.history_ms))
        if not seed_frames:
            seed_frames = self._recent_frames_window(self.chunk_ms)
        return self._pending_transcript_utterance_from_frames(
            origin_state=origin_state,
            frames=seed_frames,
        )

    def _pending_transcript_utterance_from_frames(
        self,
        *,
        origin_state: str,
        frames: tuple[_RecentFrame, ...],
        max_capture_ms: int | None = None,
    ) -> _PendingTranscriptUtterance:
        """Build one bounded pending utterance from the provided frame slice."""

        pending = _PendingTranscriptUtterance(
            origin_state=origin_state,
            max_capture_ms=max_capture_ms
            if max_capture_ms is not None
            else max(self.history_ms, self.wake_candidate_window_ms + self.wake_tail_max_ms),
        )
        for frame in frames:
            self._append_pending_frame(pending, frame)
            pending.speech_active = self._frame_counts_as_remote_asr_speech(
                frame,
                continuing=pending.speech_active,
            )
            if pending.speech_active:
                pending.active_ms += frame.duration_ms
        pending.trailing_silence_ms = 0 if pending.speech_active else pending.captured_ms
        return pending

    def _recent_frames_window(self, duration_ms: int) -> tuple[_RecentFrame, ...]:
        """Return the newest bounded frame sequence for one duration window."""

        target_ms = max(self.chunk_ms, int(duration_ms))
        frames: list[_RecentFrame] = []
        collected_ms = 0
        for frame in reversed(self._history):
            frames.append(frame)
            collected_ms += frame.duration_ms
            if collected_ms >= target_ms:
                break
        frames.reverse()
        return tuple(frames)

    def _append_pending_frame(
        self,
        pending: _PendingTranscriptUtterance,
        frame: _RecentFrame,
    ) -> None:
        """Append one frame into the wake buffer and trim to the bounded budget."""

        pending.frames.append(frame)
        pending.captured_ms += frame.duration_ms
        while pending.captured_ms > pending.max_capture_ms and pending.frames:
            removed = pending.frames.popleft()
            pending.captured_ms = max(0, pending.captured_ms - removed.duration_ms)

    def _pending_transcript_capture_window(
        self,
        pending: _PendingTranscriptUtterance,
    ) -> AmbientAudioCaptureWindow:
        """Build one capture window from the bounded same-stream utterance buffer."""

        if pending.frames:
            frames = tuple(pending.frames)
        else:
            frames = self._recent_frames_window(self.history_ms)
        return self._capture_window_from_frames(frames)

    def _recent_remote_asr_stage1_capture(self) -> AmbientAudioCaptureWindow:
        """Prefer the start of the latest active speech burst over the newest tail.

        Transcript-first wake scans are intentionally short so the server-side
        STT can catch a compact wake prefix quickly. In real speech, however,
        one STT decode can still take hundreds of milliseconds. If Twinr scans
        only the newest 1000 ms tail, the wake can already have fallen out of
        the window and only the spoken command remains. Anchor the short scan at
        the start of the latest active speech burst inside the bounded history
        so delayed scans still see the wake prefix.
        """

        recent_frames = self._recent_frames_window(self.history_ms)
        burst_frames = self._latest_active_speech_burst_frames(recent_frames)
        if burst_frames:
            return self._capture_window_from_frames(
                self._leading_frames_window(
                    burst_frames,
                    duration_ms=self._effective_remote_asr_stage1_window_ms(),
                )
            )
        return self._capture_window_from_frames(
            self._leading_frames_window(
                recent_frames,
                duration_ms=self._effective_remote_asr_stage1_window_ms(),
            )
        )

    def _latest_active_speech_burst_frames(
        self,
        frames: tuple[_RecentFrame, ...],
    ) -> tuple[_RecentFrame, ...]:
        """Return the latest speech burst plus a tiny pre-roll for quiet onsets.

        The speech threshold is intentionally conservative for far-field room
        noise, but the first consonant of an activation phrase can still land below that
        threshold. If we anchor the transcript-first scan at the first strictly
        active frame, the STT backend often loses the opening phoneme and turns
        ``Twinner`` into ``Winner``. Keep a bounded lead-in from the frames
        immediately before the burst so the short scan still includes that
        quiet onset without falling back to the newest tail.
        """

        resolved_frames = tuple(frames)
        if not resolved_frames:
            return ()
        speech_flags = self._remote_asr_speech_flags(resolved_frames)
        last_active_index: int | None = None
        for index in range(len(resolved_frames) - 1, -1, -1):
            if speech_flags[index]:
                last_active_index = index
                break
        if last_active_index is None:
            return ()
        # Keep one natural "Twinna, ..." utterance together even when the
        # speaker inserts a short comma-length pause between the wake and the
        # spoken request. Re-using the bounded same-stream listening window
        # keeps stage-one segmentation aligned with the later transcript commit
        # path instead of splitting the wake off and scanning only the command
        # tail.
        max_silence_ms = max(
            self.chunk_ms,
            min(self.follow_up_window_ms, self._effective_remote_asr_stage1_window_ms()),
        )
        silence_ms = 0
        start_index = last_active_index
        have_active = False
        for index, frame in enumerate(resolved_frames[: last_active_index + 1]):
            if speech_flags[index]:
                if not have_active or silence_ms >= max_silence_ms:
                    start_index = index
                have_active = True
                silence_ms = 0
                continue
            if have_active:
                silence_ms += max(0, int(frame.duration_ms))
        pre_roll_ms = min(
            self._effective_remote_asr_stage1_window_ms(),
            max(self.chunk_ms, self.chunk_ms * 2),
        )
        pre_roll_start_index = start_index
        collected_pre_roll_ms = 0
        while pre_roll_start_index > 0 and collected_pre_roll_ms < pre_roll_ms:
            previous_frame = resolved_frames[pre_roll_start_index - 1]
            if previous_frame.rms <= 0:
                break
            collected_pre_roll_ms += max(0, int(previous_frame.duration_ms))
            pre_roll_start_index -= 1
        return resolved_frames[pre_roll_start_index : last_active_index + 1]

    def _leading_frames_window(
        self,
        frames: tuple[_RecentFrame, ...],
        *,
        duration_ms: int,
    ) -> tuple[_RecentFrame, ...]:
        """Return the earliest bounded prefix from one recent frame sequence."""

        target_ms = max(self.chunk_ms, int(duration_ms))
        selected: list[_RecentFrame] = []
        collected_ms = 0
        for frame in frames:
            selected.append(frame)
            collected_ms += max(0, int(frame.duration_ms))
            if collected_ms >= target_ms:
                break
        return tuple(selected)

    def _maybe_detect_remote_asr_activation_candidate(self) -> VoiceActivationMatch | None:
        """Transcribe a recent speech window remotely and match activation phrases."""

        now = self._monotonic()
        if now < self._next_wake_candidate_check_at:
            return None
        try:
            with self._trace_span(
                name="voice_remote_asr_stage1_scan",
                kind="llm_call",
                details={
                    "window_ms": self._effective_remote_asr_stage1_window_ms(),
                    **self._intent_context.trace_details(),
                },
            ):
                capture = self._recent_remote_asr_stage1_capture()
            # The colocated STT service already owns its own VAD /
            # active-listening gate. Keep the Twinr-side pre-gate open by
            # default so far-field or quiet activation phrases do not get rejected
            # before transcript-first matching ever runs. Operators can still
            # opt back into an explicit activity ratio if they want a stricter
            # compute budget. Only truly empty windows should be buffered here.
            if capture.sample.peak_rms <= 0 and capture.sample.average_rms <= 0:
                self._record_transcript_debug(
                    stage="activation_stage1",
                    outcome="buffering_no_audio_energy",
                    capture=capture,
                    details={"reason": "nonzero_audio_energy_required"},
                )
                self._trace_decision(
                    "voice_remote_asr_stage1_buffering",
                    question="Should this remote-ASR activation candidate be scanned already?",
                    selected={"id": "buffer", "summary": "Wait for non-empty audio before scanning"},
                    options=[
                        {"id": "buffer", "summary": "Keep buffering until non-empty audio arrives"},
                        {"id": "scan", "summary": "Transcribe the current silent/empty window now"},
                    ],
                    context={
                        "window_ms": capture.sample.duration_ms,
                        "peak_rms": int(capture.sample.peak_rms),
                        "average_rms": int(capture.sample.average_rms),
                    },
                    confidence="high",
                    guardrails=["nonzero_audio_energy_required"],
                )
                return None
            if capture.sample.duration_ms < self._effective_remote_asr_min_activation_duration_ms():
                self._record_transcript_debug(
                    stage="activation_stage1",
                    outcome="buffering_short_wake_burst",
                    capture=capture,
                    details={
                        "reason": "remote_asr_min_activation_duration_ms",
                        "required_window_ms": self._effective_remote_asr_min_activation_duration_ms(),
                        **self._intent_context.trace_details(),
                    },
                )
                self._trace_decision(
                    "voice_remote_asr_stage1_buffering",
                    question="Should this remote-ASR activation candidate be scanned already?",
                    selected={"id": "buffer", "summary": "Wait for a longer wake burst before scanning"},
                    options=[
                        {"id": "buffer", "summary": "Keep buffering wake speech"},
                        {"id": "scan", "summary": "Transcribe the short wake burst now"},
                    ],
                    context={
                        "window_ms": capture.sample.duration_ms,
                        "required_window_ms": self._effective_remote_asr_min_activation_duration_ms(),
                        "active_ratio": round(float(capture.sample.active_ratio), 4),
                    },
                    confidence="high",
                    guardrails=["remote_asr_min_activation_duration_ms"],
                )
                return None
            if (
                self.wake_candidate_min_active_ratio > 0.0
                and capture.sample.active_ratio < self.wake_candidate_min_active_ratio
            ):
                self._record_transcript_debug(
                    stage="activation_stage1",
                    outcome="rejected_low_activity",
                    capture=capture,
                    details={
                        "reason": "wake_candidate_min_active_ratio",
                        "required_active_ratio": self.wake_candidate_min_active_ratio,
                    },
                )
                self._trace_decision(
                    "voice_remote_asr_stage1_rejected",
                    question="Should this remote-ASR activation candidate be scanned?",
                    selected={"id": "reject", "summary": "Active ratio below configured threshold"},
                    options=[
                        {"id": "reject", "summary": "Skip low-activity candidate"},
                        {"id": "scan", "summary": "Run transcript-first wake scan"},
                    ],
                    context={"active_ratio": round(float(capture.sample.active_ratio), 4)},
                    confidence="high",
                    guardrails=["wake_candidate_min_active_ratio"],
                )
                return None
            match = self._detect_wake_capture(
                capture=capture,
                stage="activation_stage1",
            )
            if match is None:
                return None
            if not match.detected:
                self._record_transcript_debug(
                    stage="activation_stage1",
                    outcome="no_match",
                    transcript=match.transcript,
                    matched_phrase=match.matched_phrase,
                    remaining_text=match.remaining_text,
                    detector_label=match.detector_label,
                    score=match.score,
                    capture=capture,
                )
                self._trace_event(
                    "voice_remote_asr_stage1_no_match",
                    kind="branch",
                    details={
                        "active_ratio": round(float(capture.sample.active_ratio), 4),
                        "transcript_chars": len(str(match.transcript or "").strip()),
                    },
                )
                return None
            if _normalize_text_length(match.transcript) < self.wake_candidate_min_transcript_chars:
                self._record_transcript_debug(
                    stage="wake_stage1",
                    outcome="rejected_short_transcript",
                    transcript=match.transcript,
                    matched_phrase=match.matched_phrase,
                    remaining_text=match.remaining_text,
                    detector_label=match.detector_label,
                    score=match.score,
                    capture=capture,
                    details={
                        "reason": "wake_candidate_min_transcript_chars",
                        "required_transcript_chars": self.wake_candidate_min_transcript_chars,
                    },
                )
                self._trace_decision(
                    "voice_remote_asr_stage1_rejected",
                    question="Should this remote-ASR activation candidate be accepted?",
                    selected={"id": "reject", "summary": "Transcript too short after wake scan"},
                    options=[
                        {"id": "reject", "summary": "Reject low-evidence transcript"},
                        {"id": "accept", "summary": "Accept wake candidate"},
                    ],
                    context={
                        "transcript_chars": len(str(match.transcript or "").strip()),
                        "matched_phrase": match.matched_phrase,
                    },
                    confidence="medium",
                    guardrails=["wake_candidate_min_transcript_chars"],
                )
                return None
            self._record_transcript_debug(
                stage="activation_stage1",
                outcome="matched",
                transcript=match.transcript,
                matched_phrase=match.matched_phrase,
                remaining_text=match.remaining_text,
                detector_label=match.detector_label,
                score=match.score,
                capture=capture,
            )
            self._trace_event(
                "voice_remote_asr_stage1_match",
                kind="decision",
                details={
                    "matched_phrase": match.matched_phrase,
                    "transcript_chars": len(str(match.transcript or "").strip()),
                    "remaining_text_chars": len(str(match.remaining_text or "").strip()),
                },
            )
            return match
        finally:
            self._next_wake_candidate_check_at = self._monotonic() + self._wake_candidate_cooldown_s

    def _maybe_detect_barge_in_candidate(self) -> OrchestratorVoiceBargeInInterruptEvent | None:
        """Transcribe one bounded speaking window to detect a user interruption."""

        if self._monotonic() < self._next_barge_in_candidate_check_at:
            return None
        capture = self._recent_capture_window(self.barge_in_window_ms)
        self._next_barge_in_candidate_check_at = self._monotonic() + self.candidate_cooldown_s
        if (
            capture.sample.active_chunk_count <= 0
            or capture.sample.active_ratio < self.barge_in_min_active_ratio
        ):
            self._record_transcript_debug(
                stage="barge_in_candidate",
                outcome="rejected_low_activity",
                capture=capture,
                details={"required_active_ratio": self.barge_in_min_active_ratio},
            )
            self._trace_decision(
                "voice_barge_in_candidate_rejected",
                question="Should this speech candidate trigger a remote action?",
                selected={"id": "reject", "summary": "Insufficient active speech evidence"},
                options=[
                    {"id": "reject", "summary": "Ignore low-activity window"},
                    {"id": "transcribe", "summary": "Transcribe candidate window"},
                ],
                context={
                    "active_chunk_count": capture.sample.active_chunk_count,
                    "active_ratio": round(float(capture.sample.active_ratio), 4),
                },
                confidence="high",
                guardrails=["speech_activity_threshold"],
            )
            return None
        with self._trace_span(
            name="voice_barge_in_candidate_transcribe",
            kind="llm_call",
            details={"window_ms": capture.sample.duration_ms},
        ):
            with self._backend_request_context(stage="barge_in_candidate", capture=capture):
                transcript = self.backend.transcribe(
                    _pcm_capture_to_wav_bytes(capture),
                    filename="voice-window.wav",
                    content_type="audio/wav",
                    language=self.config.openai_realtime_language,
                ).strip()
        self._record_transcript_debug(
            stage="barge_in_candidate",
            outcome="transcribed",
            transcript=transcript,
            capture=capture,
        )
        if _normalize_text_length(transcript) < self.barge_in_min_transcript_chars:
            self._record_transcript_debug(
                stage="barge_in_candidate",
                outcome="rejected_short_transcript",
                transcript=transcript,
                capture=capture,
                details={"required_transcript_chars": self.barge_in_min_transcript_chars},
            )
            self._trace_decision(
                "voice_barge_in_candidate_rejected",
                question="Should this transcribed speech candidate trigger a remote action?",
                selected={"id": "reject", "summary": "Transcript did not contain enough speech"},
                options=[
                    {"id": "reject", "summary": "Ignore low-evidence transcript"},
                    {"id": "accept", "summary": "Trigger remote action"},
                ],
                context={
                    "transcript_chars": len(transcript),
                    "active_ratio": round(float(capture.sample.active_ratio), 4),
                },
                confidence="medium",
                guardrails=["min_transcript_chars"],
            )
            return None
        self._record_transcript_debug(
            stage="barge_in_candidate",
            outcome="interrupt_requested",
            transcript=transcript,
            capture=capture,
        )
        self._trace_event(
            "voice_barge_in_candidate_triggered",
            kind="decision",
            details={"transcript_chars": len(transcript), "transcript_preview": transcript[:80]},
        )
        return OrchestratorVoiceBargeInInterruptEvent(transcript_preview=transcript[:160])

    def _recent_capture_window(self, duration_ms: int) -> AmbientAudioCaptureWindow:
        frames = self._recent_frames_window(duration_ms)
        return self._capture_window_from_frames(frames)

    def _capture_window_from_frames(
        self,
        frames: tuple[_RecentFrame, ...] | list[_RecentFrame],
    ) -> AmbientAudioCaptureWindow:
        """Assemble one ambient capture window from a bounded frame sequence."""

        resolved_frames = tuple(frames)
        if not resolved_frames:
            resolved_frames = (_RecentFrame(pcm_bytes=b"", rms=0, duration_ms=0),)
        pcm_fragments = [frame.pcm_bytes for frame in resolved_frames]
        rms_values = [int(frame.rms) for frame in resolved_frames]
        collected_ms = sum(max(0, int(frame.duration_ms)) for frame in resolved_frames)
        active_chunk_count = sum(1 for active in self._remote_asr_speech_flags(resolved_frames) if active)
        sample = AmbientAudioLevelSample(
            duration_ms=collected_ms,
            chunk_count=len(rms_values),
            active_chunk_count=active_chunk_count,
            average_rms=int(sum(rms_values) / max(1, len(rms_values))),
            peak_rms=max(rms_values),
            active_ratio=active_chunk_count / max(1, len(rms_values)),
        )
        return AmbientAudioCaptureWindow(
            sample=sample,
            pcm_bytes=b"".join(pcm_fragments),
            sample_rate=self.sample_rate,
            channels=self.channels,
        )


def _pcm_capture_to_wav_bytes(capture: AmbientAudioCaptureWindow) -> bytes:
    """Convert a recent PCM capture window into a WAV upload payload."""

    from twinr.hardware.audio import pcm16_to_wav_bytes

    return pcm16_to_wav_bytes(
        capture.pcm_bytes,
        sample_rate=capture.sample_rate,
        channels=capture.channels,
    )


def _build_remote_asr_backend(
    config: TwinrConfig,
    *,
    timeout_s: float | None = None,
) -> RemoteAsrBackendAdapter:
    """Build the thh1986 remote-ASR backend shared by transcript-first paths."""

    remote_asr_url = str(getattr(config, "voice_orchestrator_remote_asr_url", "") or "").strip()
    if not remote_asr_url:
        raise ValueError(
            "TWINR_VOICE_ORCHESTRATOR_REMOTE_ASR_URL is required for the "
            "transcript-first remote voice gateway."
        )
    if timeout_s is None:
        timeout_s = float(getattr(config, "voice_orchestrator_remote_asr_timeout_s", 3.0) or 3.0)
    return RemoteAsrBackendAdapter(
        base_url=remote_asr_url,
        bearer_token=getattr(config, "voice_orchestrator_remote_asr_bearer_token", None),
        language=(
            getattr(config, "voice_orchestrator_remote_asr_language", None)
            or config.openai_realtime_language
        ),
        mode=getattr(config, "voice_orchestrator_remote_asr_mode", "active_listening"),
        timeout_s=float(timeout_s),
        retry_attempts=int(getattr(config, "voice_orchestrator_remote_asr_retry_attempts", 1) or 0),
        retry_backoff_s=float(
            getattr(config, "voice_orchestrator_remote_asr_retry_backoff_s", 0.35) or 0.0
        ),
    )


def _build_transcript_backend(config: TwinrConfig) -> _TranscriptBackend:
    """Build the transcript backend used for follow-up and barge-in scans."""

    return _build_remote_asr_backend(config)


def _build_wake_phrase_spotter(
    config: TwinrConfig,
    *,
    backend: _TranscriptBackend,
) -> VoiceActivationPhraseMatcher:
    """Build the bounded transcript matcher used for wake confirmation."""

    return VoiceActivationPhraseMatcher(
        backend=backend,
        phrases=getattr(config, "voice_activation_phrases", ()),
        language=config.openai_realtime_language,
        suppress_transcription_errors=False,
    )


__all__ = ["EdgeOrchestratorVoiceSession"]
