"""Own server-side wakeword/session decisions for the voice orchestrator path.

The edge Pi streams bounded PCM frames over the orchestrator websocket. This
module keeps the server-side logic focused on three responsibilities:

- detect a configured wakeword with the selected streaming backend
- confirm wakewords with a short post-roll and extract any spoken remainder
- decide when a follow-up or barge-in needs a local bounded capture on the Pi

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
from twinr.hardware.audio import AmbientAudioCaptureWindow, AmbientAudioLevelSample
from twinr.orchestrator.local_stt import LocalSttBackendAdapter
from twinr.proactive.wakeword.kws import WakewordSherpaOnnxFrameSpotter
from twinr.proactive.wakeword.matching import (
    WakewordMatch,
    WakewordPhraseSpotter,
    WakewordTailTranscriptExtractor,
)
from twinr.proactive.wakeword.policy import normalize_wakeword_backend
from twinr.providers.openai import OpenAIBackend

from .voice_contracts import (
    OrchestratorVoiceAudioFrame,
    OrchestratorVoiceBargeInInterruptEvent,
    OrchestratorVoiceFollowUpCaptureRequestedEvent,
    OrchestratorVoiceFollowUpClosedEvent,
    OrchestratorVoiceHelloRequest,
    OrchestratorVoiceReadyEvent,
    OrchestratorVoiceRuntimeStateEvent,
    OrchestratorVoiceWakeConfirmedEvent,
)


class _FrameSpotter(Protocol):
    """Describe the minimal streaming wakeword surface the session needs."""

    def process_pcm_bytes(self, pcm_bytes: bytes, *, channels: int = 1) -> WakewordMatch | None:
        """Feed PCM bytes and maybe return a wakeword match."""

    def reset(self) -> None:
        """Clear any internal streaming state."""


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
class _PendingWakeConfirmation:
    """Track one stage-one wakeword candidate awaiting post-roll confirmation."""

    match: WakewordMatch
    remaining_postroll_ms: int
    stage1_source: str = "backend"
    elapsed_ms: int = 0
    saw_post_wake_activity: bool = False
    trailing_silence_ms: int = 0
    frames: deque[_RecentFrame] = field(default_factory=deque)
    captured_ms: int = 0
    max_capture_ms: int = 0


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

    _ACTIVE_STATES = frozenset({"waiting", "wake_armed", "speaking", "follow_up_open"})

    def __init__(
        self,
        config: TwinrConfig,
        *,
        backend: _TranscriptBackend | None = None,
        frame_spotter: _FrameSpotter | None = None,
        wake_phrase_spotter: WakewordPhraseSpotter | None = None,
        backend_tail_transcript_extractor: WakewordTailTranscriptExtractor | None = None,
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
        self.wake_stage1_mode = str(
            getattr(config, "voice_orchestrator_wake_stage1_mode", "backend") or "backend"
        ).strip().lower() or "backend"
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
        self.local_stt_stage1_window_ms = max(
            self.chunk_ms,
            min(self.history_ms, 1000),
        )
        self.wake_postroll_ms = max(
            self.chunk_ms,
            int(getattr(config, "voice_orchestrator_wake_postroll_ms", 900) or 900),
        )
        self.wake_tail_max_ms = max(
            self.wake_postroll_ms,
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
        self._wake_candidate_cooldown_s = (
            self.chunk_ms / 1000.0 if self.wake_stage1_mode == "local_stt" else self.candidate_cooldown_s
        )
        self.speech_threshold = int(config.audio_speech_threshold)
        self._monotonic = monotonic_fn
        self._history: deque[_RecentFrame] = deque(maxlen=max(8, math.ceil(self.history_ms / self.chunk_ms)))
        self._pending_wake: _PendingWakeConfirmation | None = None
        self._follow_up_deadline_at: float | None = None
        self._next_wake_candidate_check_at: float = 0.0
        self._next_follow_up_candidate_check_at: float = 0.0
        self._next_barge_in_candidate_check_at: float = 0.0
        self._barge_in_sent = False
        self._session_id = ""
        self._state = "waiting"
        self._follow_up_allowed = False
        self.backend = backend or _build_transcript_backend(config)
        if self.wake_stage1_mode == "local_stt":
            self._frame_spotter = frame_spotter or _build_local_stt_stage1_fallback_frame_spotter(
                config
            )
        else:
            self._frame_spotter = frame_spotter or _build_frame_spotter(config)
        self._wake_phrase_spotter = wake_phrase_spotter or _build_wake_phrase_spotter(
            config,
            backend=self.backend,
        )
        self._backend_tail_transcript_extractor = (
            backend_tail_transcript_extractor
            or _build_backend_tail_transcript_extractor(config)
        )
        self._forensics: WorkflowForensics | None = None
        self._trace_id: str = uuid4().hex

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
        if trace_id:
            self._trace_id = str(trace_id)

    def _trace_details(self, details: dict[str, object] | None = None) -> dict[str, object]:
        payload: dict[str, object] = {
            "session_id": self._session_id or None,
            "state": self._state,
            "backend": self.backend_name,
            "follow_up_allowed": self._follow_up_allowed,
        }
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

    @property
    def backend_name(self) -> str:
        """Return the selected streaming wakeword backend label."""

        if self.wake_stage1_mode == "local_stt":
            return "local_stt"
        return normalize_wakeword_backend(
            getattr(self.config, "wakeword_primary_backend", self.config.wakeword_backend),
            default="openwakeword",
        )

    def close(self) -> None:
        """Reset any buffered streaming state."""

        self._history.clear()
        self._pending_wake = None
        self._follow_up_deadline_at = None
        if self._frame_spotter is not None:
            self._frame_spotter.reset()

    def handle_hello(self, request: OrchestratorVoiceHelloRequest) -> list[dict[str, Any]]:
        """Accept one new edge voice session and validate stream metadata."""

        self._session_id = request.session_id
        self._trace_id = uuid4().hex
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
        self._state = request.initial_state or "waiting"
        self._trace_event(
            "voice_session_hello_accepted",
            kind="run_start",
            details={
                "sample_rate": request.sample_rate,
                "channels": request.channels,
                "chunk_ms": request.chunk_ms,
                "initial_state": self._state,
            },
        )
        return [OrchestratorVoiceReadyEvent(session_id=request.session_id, backend=self.backend_name).to_payload()]

    def handle_runtime_state(self, event: OrchestratorVoiceRuntimeStateEvent) -> list[dict[str, Any]]:
        """Update explicit edge runtime state and drain any timeout-based events."""

        previous_state = self._state
        self._state = event.state or "waiting"
        self._follow_up_allowed = bool(event.follow_up_allowed)
        if self._state == "follow_up_open" and self._follow_up_allowed:
            self._follow_up_deadline_at = self._monotonic() + self.follow_up_timeout_s
            self._next_follow_up_candidate_check_at = 0.0
        else:
            self._follow_up_deadline_at = None
        if self._state != "speaking":
            self._barge_in_sent = False
        if self._state not in self._ACTIVE_STATES:
            self._pending_wake = None
            if self._frame_spotter is not None:
                self._frame_spotter.reset()
        self._trace_event(
            "voice_runtime_state_received",
            kind="mutation",
            details={
                "previous_state": previous_state,
                "new_state": self._state,
                "detail": event.detail,
                "follow_up_allowed": self._follow_up_allowed,
            },
        )
        return self._drain_timeouts()

    def handle_audio_frame(self, frame: OrchestratorVoiceAudioFrame) -> list[dict[str, Any]]:
        """Process one streamed PCM frame and emit any server-side decisions."""

        events = self._drain_timeouts()
        pcm_bytes = bytes(frame.pcm_bytes or b"")
        if not pcm_bytes:
            return events
        self._remember_frame(pcm_bytes)
        if self._pending_wake is not None:
            wake_event = self._advance_pending_wake(len(pcm_bytes))
            if wake_event is not None:
                events.append(wake_event.to_payload())
                return events
            return events
        if self._should_run_stage1_wake_detection() and self._maybe_start_stage1_wake(pcm_bytes):
            return events
        if self._state == "speaking" and not self._barge_in_sent:
            barge_in_event = self._maybe_detect_speech_candidate(kind="barge_in")
            if barge_in_event is not None:
                self._barge_in_sent = True
                self._state = "thinking"
                events.append(barge_in_event.to_payload())
                return events
        if self._state == "follow_up_open" and self._follow_up_allowed:
            follow_up_event = self._maybe_detect_speech_candidate(kind="follow_up")
            if follow_up_event is not None:
                if isinstance(follow_up_event, OrchestratorVoiceWakeConfirmedEvent):
                    self._state = "thinking"
                else:
                    self._state = "listening"
                self._follow_up_deadline_at = None
                events.append(follow_up_event.to_payload())
        return events

    def _should_run_stage1_wake_detection(self) -> bool:
        """Keep fresh wake detection active in wake and follow-up windows."""

        if self._state in {"waiting", "wake_armed"}:
            return True
        return self._state == "follow_up_open" and self._follow_up_allowed

    def _maybe_start_stage1_wake(self, pcm_bytes: bytes) -> bool:
        """Start one pending wake confirmation when stage one reports a hit."""

        stage1_source = "backend"
        if self.wake_stage1_mode == "local_stt":
            match = self._maybe_detect_local_stt_wake_candidate()
            if match is not None and match.detected:
                stage1_source = "local_stt"
            elif self._frame_spotter is not None:
                match = self._frame_spotter.process_pcm_bytes(pcm_bytes, channels=self.channels)
                if match is not None and match.detected:
                    stage1_source = "frame_spotter_fallback"
        else:
            match = (
                self._frame_spotter.process_pcm_bytes(pcm_bytes, channels=self.channels)
                if self._frame_spotter is not None
                else None
            )
            if match is not None and match.detected:
                stage1_source = "backend"
        if match is None or not match.detected:
            return False
        self._pending_wake = self._new_pending_wake_confirmation(
            match=match,
            stage1_source=stage1_source,
        )
        self._trace_event(
            "voice_wake_pending_started",
            kind="decision",
            details={
                "matched_phrase": match.matched_phrase,
                "remaining_text_chars": len(str(match.remaining_text or "").strip()),
                "detector_label": match.detector_label,
                "score": match.score,
                "stage1_source": stage1_source,
                "runtime_path": "follow_up_open" if self._state == "follow_up_open" else self._state,
            },
        )
        return True

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

    def _drain_timeouts(self) -> list[dict[str, Any]]:
        if self._state != "follow_up_open" or self._follow_up_deadline_at is None:
            return []
        if self._monotonic() < self._follow_up_deadline_at:
            return []
        self._follow_up_deadline_at = None
        self._state = "waiting"
        return [OrchestratorVoiceFollowUpClosedEvent(reason="timeout").to_payload()]

    def _advance_pending_wake(self, byte_count: int) -> OrchestratorVoiceWakeConfirmedEvent | None:
        pending = self._pending_wake
        if pending is None:
            return None
        latest_frame = self._history[-1] if self._history else None
        if latest_frame is not None:
            self._append_pending_frame(pending, latest_frame)
        frame_duration_ms = max(
            self.chunk_ms,
            int(round((byte_count / max(1, self.channels * 2 * self.sample_rate)) * 1000.0)),
        )
        pending.elapsed_ms += frame_duration_ms
        pending.remaining_postroll_ms -= frame_duration_ms
        latest_rms = int(latest_frame.rms) if latest_frame is not None else 0
        frame_active = latest_rms >= self.speech_threshold
        if frame_active:
            pending.saw_post_wake_activity = True
            pending.trailing_silence_ms = 0
        else:
            pending.trailing_silence_ms += frame_duration_ms
        if pending.remaining_postroll_ms > 0:
            return None
        should_finalize = False
        local_stt_has_remaining_text = (
            self.wake_stage1_mode == "local_stt"
            and _normalize_text_length(pending.match.remaining_text) > 0
        )
        if pending.elapsed_ms >= self.wake_tail_max_ms:
            should_finalize = True
        elif (
            pending.saw_post_wake_activity
            and pending.trailing_silence_ms >= self.wake_tail_endpoint_silence_ms
            and (self.wake_stage1_mode != "local_stt" or local_stt_has_remaining_text)
        ):
            should_finalize = True
        if not should_finalize:
            return None
        self._pending_wake = None
        if pending.match.detected:
            self._state = "thinking"
            remaining_text = self._extract_backend_remaining_text(pending)
            if not remaining_text:
                remaining_text = str(pending.match.remaining_text or "").strip()
            self._trace_event(
                "voice_wake_confirmed",
                kind="decision",
                details={
                    "path": "stage1_direct",
                    "stage1_source": pending.stage1_source,
                    "matched_phrase": pending.match.matched_phrase,
                    "remaining_text_chars": len(remaining_text),
                    "detector_label": pending.match.detector_label,
                    "score": pending.match.score,
                },
            )
            return OrchestratorVoiceWakeConfirmedEvent(
                matched_phrase=pending.match.matched_phrase,
                remaining_text=remaining_text,
                backend=pending.match.backend or self.backend_name,
                detector_label=pending.match.detector_label,
                score=pending.match.score,
            )
        capture = self._pending_capture_window(pending)
        confirmed = self._wake_phrase_spotter.detect(capture)
        if confirmed.detected:
            self._state = "thinking"
            self._trace_event(
                "voice_wake_confirmed",
                kind="decision",
                details={
                    "path": "tail_confirmation",
                    "matched_phrase": confirmed.matched_phrase,
                    "remaining_text_chars": len(str(confirmed.remaining_text or "").strip()),
                    "detector_label": confirmed.detector_label,
                    "score": confirmed.score,
                },
            )
            return OrchestratorVoiceWakeConfirmedEvent(
                matched_phrase=confirmed.matched_phrase,
                remaining_text=confirmed.remaining_text,
                backend=confirmed.backend or self.backend_name,
                detector_label=confirmed.detector_label,
                score=confirmed.score,
            )
        if not confirmed.detected:
            if self._frame_spotter is not None:
                self._frame_spotter.reset()
            return None
        return None

    def _extract_backend_remaining_text(self, pending: _PendingWakeConfirmation) -> str:
        """Recover optional trailing user speech after a backend-led wake hit."""

        extractor = self._backend_tail_transcript_extractor
        if extractor is None:
            return ""
        capture = self._pending_capture_window(pending)
        return extractor.extract(capture)

    def _new_pending_wake_confirmation(
        self,
        *,
        match: WakewordMatch,
        stage1_source: str,
    ) -> _PendingWakeConfirmation:
        """Seed one bounded wake buffer from the recent pre-hit stream context."""

        pending = _PendingWakeConfirmation(
            match=match,
            remaining_postroll_ms=self.wake_postroll_ms,
            max_capture_ms=max(self.history_ms, self.wake_candidate_window_ms + self.wake_tail_max_ms),
            stage1_source=stage1_source,
        )
        for frame in self._recent_frames_window(self.wake_candidate_window_ms):
            self._append_pending_frame(pending, frame)
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
        pending: _PendingWakeConfirmation,
        frame: _RecentFrame,
    ) -> None:
        """Append one frame into the wake buffer and trim to the bounded budget."""

        pending.frames.append(frame)
        pending.captured_ms += frame.duration_ms
        while pending.captured_ms > pending.max_capture_ms and pending.frames:
            removed = pending.frames.popleft()
            pending.captured_ms = max(0, pending.captured_ms - removed.duration_ms)

    def _pending_capture_window(self, pending: _PendingWakeConfirmation) -> AmbientAudioCaptureWindow:
        """Build one capture window from the bounded wake buffer."""

        if pending.frames:
            frames = tuple(pending.frames)
        else:
            frames = self._recent_frames_window(self.history_ms)
        return self._capture_window_from_frames(frames)

    def _recent_local_stt_stage1_capture(self) -> AmbientAudioCaptureWindow:
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
                    duration_ms=self.local_stt_stage1_window_ms,
                )
            )
        return self._capture_window_from_frames(
            self._leading_frames_window(
                recent_frames,
                duration_ms=self.local_stt_stage1_window_ms,
            )
        )

    def _latest_active_speech_burst_frames(
        self,
        frames: tuple[_RecentFrame, ...],
    ) -> tuple[_RecentFrame, ...]:
        """Return the latest speech burst plus a tiny pre-roll for quiet onsets.

        The speech threshold is intentionally conservative for far-field room
        noise, but the first consonant of a wakeword can still land below that
        threshold. If we anchor the transcript-first scan at the first strictly
        active frame, the STT backend often loses the opening phoneme and turns
        ``Twinner`` into ``Winner``. Keep a bounded lead-in from the frames
        immediately before the burst so the short scan still includes that
        quiet onset without falling back to the newest tail.
        """

        resolved_frames = tuple(frames)
        if not resolved_frames:
            return ()
        last_active_index: int | None = None
        for index in range(len(resolved_frames) - 1, -1, -1):
            if resolved_frames[index].rms >= self.speech_threshold:
                last_active_index = index
                break
        if last_active_index is None:
            return ()
        max_silence_ms = max(self.chunk_ms, self.wake_tail_endpoint_silence_ms)
        silence_ms = 0
        start_index = last_active_index
        have_active = False
        for index, frame in enumerate(resolved_frames[: last_active_index + 1]):
            if frame.rms >= self.speech_threshold:
                if not have_active or silence_ms >= max_silence_ms:
                    start_index = index
                have_active = True
                silence_ms = 0
                continue
            if have_active:
                silence_ms += max(0, int(frame.duration_ms))
        pre_roll_ms = min(
            self.local_stt_stage1_window_ms,
            max(self.chunk_ms, self.chunk_ms * 2),
        )
        onset_threshold = max(1, self.speech_threshold // 2)
        pre_roll_start_index = start_index
        collected_pre_roll_ms = 0
        while pre_roll_start_index > 0 and collected_pre_roll_ms < pre_roll_ms:
            previous_frame = resolved_frames[pre_roll_start_index - 1]
            if previous_frame.rms < onset_threshold:
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

    def _maybe_detect_local_stt_wake_candidate(self) -> WakewordMatch | None:
        """Transcribe a recent speech window locally and match wakeword phrases."""

        now = self._monotonic()
        if now < self._next_wake_candidate_check_at:
            return None
        try:
            with self._trace_span(
                name="voice_local_stt_stage1_scan",
                kind="llm_call",
                details={"window_ms": self.local_stt_stage1_window_ms},
            ):
                capture = self._recent_local_stt_stage1_capture()
            # The colocated STT service already owns its own VAD /
            # active-listening gate. Keep the Twinr-side pre-gate open by
            # default so far-field or quiet wakewords do not get rejected
            # before transcript-first matching ever runs. Operators can still
            # opt back into an explicit activity ratio if they want a stricter
            # compute budget.
            if (
                self.wake_candidate_min_active_ratio > 0.0
                and capture.sample.active_ratio < self.wake_candidate_min_active_ratio
            ):
                self._trace_decision(
                    "voice_local_stt_stage1_rejected",
                    question="Should this local STT wake candidate be scanned?",
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
            match = self._wake_phrase_spotter.detect(capture)
            if not match.detected:
                self._trace_event(
                    "voice_local_stt_stage1_no_match",
                    kind="branch",
                    details={
                        "active_ratio": round(float(capture.sample.active_ratio), 4),
                        "transcript_chars": len(str(match.transcript or "").strip()),
                    },
                )
                return None
            if _normalize_text_length(match.transcript) < self.wake_candidate_min_transcript_chars:
                self._trace_decision(
                    "voice_local_stt_stage1_rejected",
                    question="Should this local STT wake candidate be accepted?",
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
            self._trace_event(
                "voice_local_stt_stage1_match",
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

    def _maybe_detect_speech_candidate(
        self,
        *,
        kind: str,
    ) -> (
        OrchestratorVoiceBargeInInterruptEvent
        | OrchestratorVoiceFollowUpCaptureRequestedEvent
        | OrchestratorVoiceWakeConfirmedEvent
        | None
    ):
        if kind == "barge_in":
            if self._monotonic() < self._next_barge_in_candidate_check_at:
                return None
            capture = self._recent_capture_window(self.barge_in_window_ms)
            min_active_ratio = self.barge_in_min_active_ratio
            min_chars = self.barge_in_min_transcript_chars
            self._next_barge_in_candidate_check_at = self._monotonic() + self.candidate_cooldown_s
        else:
            if self._monotonic() < self._next_follow_up_candidate_check_at:
                return None
            capture = self._recent_capture_window(self.follow_up_window_ms)
            min_active_ratio = self.follow_up_min_active_ratio
            min_chars = self.follow_up_min_transcript_chars
            self._next_follow_up_candidate_check_at = self._monotonic() + self.candidate_cooldown_s
            if capture.sample.duration_ms < self.follow_up_window_ms:
                self._trace_decision(
                    "voice_follow_up_candidate_rejected",
                    question="Should this follow-up window be transcribed already?",
                    selected={"id": "reject", "summary": "Wait until the bounded follow-up window has filled"},
                    options=[
                        {"id": "reject", "summary": "Keep buffering follow-up audio before transcribing"},
                        {"id": "transcribe", "summary": "Transcribe the partial follow-up window now"},
                    ],
                    context={
                        "window_ms": capture.sample.duration_ms,
                        "required_window_ms": self.follow_up_window_ms,
                    },
                    confidence="high",
                    guardrails=["follow_up_window_fullness"],
                )
                return None
        if capture.sample.active_chunk_count <= 0 or capture.sample.active_ratio < min_active_ratio:
            self._trace_decision(
                "voice_follow_up_candidate_rejected"
                if kind == "follow_up"
                else "voice_barge_in_candidate_rejected",
                question="Should this speech candidate trigger a remote action?",
                selected={"id": "reject", "summary": "Insufficient active speech evidence"},
                options=[
                    {"id": "reject", "summary": "Ignore low-activity window"},
                    {"id": "transcribe", "summary": "Transcribe candidate window"},
                ],
                context={
                    "kind": kind,
                    "active_chunk_count": capture.sample.active_chunk_count,
                    "active_ratio": round(float(capture.sample.active_ratio), 4),
                },
                confidence="high",
                guardrails=["speech_activity_threshold"],
            )
            return None
        with self._trace_span(
            name=f"voice_{kind}_candidate_transcribe",
            kind="llm_call",
            details={"window_ms": capture.sample.duration_ms},
        ):
            transcript = self.backend.transcribe(
                _pcm_capture_to_wav_bytes(capture),
                filename="voice-window.wav",
                content_type="audio/wav",
                language=self.config.openai_realtime_language,
            ).strip()
        if _normalize_text_length(transcript) < min_chars:
            self._trace_decision(
                "voice_follow_up_candidate_rejected"
                if kind == "follow_up"
                else "voice_barge_in_candidate_rejected",
                question="Should this transcribed speech candidate trigger a remote action?",
                selected={"id": "reject", "summary": "Transcript did not contain enough speech"},
                options=[
                    {"id": "reject", "summary": "Ignore low-evidence transcript"},
                    {"id": "accept", "summary": "Trigger remote action"},
                ],
                context={
                    "kind": kind,
                    "transcript_chars": len(transcript),
                    "active_ratio": round(float(capture.sample.active_ratio), 4),
                },
                confidence="medium",
                guardrails=["min_transcript_chars"],
            )
            return None
        if kind == "follow_up":
            wake_match = self._wake_phrase_spotter.match_transcript(transcript)
            if wake_match.detected:
                self._trace_decision(
                    "voice_follow_up_candidate_routed",
                    question="How should follow-up window speech be routed?",
                    selected={"id": "wake_confirmed", "summary": "Treat repeated wake phrase as a fresh wake turn"},
                    options=[
                        {"id": "wake_confirmed", "summary": "Open a fresh wake turn on the same stream"},
                        {"id": "follow_up_capture_requested", "summary": "Open a new local follow-up capture on the Pi"},
                    ],
                    context={
                        "transcript_chars": len(transcript),
                        "matched_phrase": wake_match.matched_phrase,
                        "remaining_text_chars": len(str(wake_match.remaining_text or "").strip()),
                    },
                    confidence="high",
                    guardrails=["wakeword_prefix_match"],
                )
                return OrchestratorVoiceWakeConfirmedEvent(
                    matched_phrase=wake_match.matched_phrase,
                    remaining_text=str(wake_match.remaining_text or "").strip(),
                    backend=wake_match.backend or "stt",
                    detector_label=wake_match.detector_label,
                    score=wake_match.score,
                )
            self._trace_decision(
                "voice_follow_up_candidate_routed",
                question="How should follow-up window speech be routed?",
                selected={"id": "follow_up_capture_requested", "summary": "Open a new local follow-up capture on the Pi"},
                options=[
                    {"id": "wake_confirmed", "summary": "Open a fresh wake turn on the same stream"},
                    {"id": "follow_up_capture_requested", "summary": "Open a new local follow-up capture on the Pi"},
                ],
                context={
                    "transcript_chars": len(transcript),
                    "transcript_preview": transcript[:80],
                },
                confidence="medium",
                guardrails=["follow_up_window"],
            )
        elif kind == "barge_in":
            self._trace_event(
                "voice_barge_in_candidate_triggered",
                kind="decision",
                details={"transcript_chars": len(transcript), "transcript_preview": transcript[:80]},
            )
        if kind == "barge_in":
            return OrchestratorVoiceBargeInInterruptEvent(transcript_preview=transcript[:160])
        return OrchestratorVoiceFollowUpCaptureRequestedEvent(transcript_preview=transcript[:160])

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
        active_chunk_count = sum(1 for rms in rms_values if rms >= self.speech_threshold)
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


def _build_frame_spotter(config: TwinrConfig) -> _FrameSpotter:
    """Build the configured streaming wakeword detector for server sessions."""

    backend = normalize_wakeword_backend(
        getattr(config, "wakeword_primary_backend", config.wakeword_backend),
        default="openwakeword",
    )
    if backend == "kws":
        return WakewordSherpaOnnxFrameSpotter(
            tokens_path=config.wakeword_kws_tokens_path or "",
            encoder_path=config.wakeword_kws_encoder_path or "",
            decoder_path=config.wakeword_kws_decoder_path or "",
            joiner_path=config.wakeword_kws_joiner_path or "",
            keywords_file_path=config.wakeword_kws_keywords_file_path or "",
            phrases=config.wakeword_phrases,
            project_root=config.project_root,
            sample_rate=config.wakeword_kws_sample_rate,
            feature_dim=config.wakeword_kws_feature_dim,
            max_active_paths=config.wakeword_kws_max_active_paths,
            keywords_score=config.wakeword_kws_keywords_score,
            keywords_threshold=config.wakeword_kws_keywords_threshold,
            num_trailing_blanks=config.wakeword_kws_num_trailing_blanks,
            num_threads=config.wakeword_kws_num_threads,
            provider=config.wakeword_kws_provider,
        )
    if backend == "wekws":
        from twinr.proactive.wakeword.wekws import WakewordWekwsFrameSpotter

        return WakewordWekwsFrameSpotter(
            model_path=config.wakeword_wekws_model_path or "",
            config_path=config.wakeword_wekws_config_path or "",
            words_path=config.wakeword_wekws_words_path or "",
            cmvn_path=config.wakeword_wekws_cmvn_path,
            phrases=config.wakeword_phrases,
            project_root=config.project_root,
            threshold=config.wakeword_wekws_threshold,
            chunk_ms=config.wakeword_wekws_chunk_ms,
            num_threads=config.wakeword_wekws_num_threads,
            provider=config.wakeword_wekws_provider,
        )
    from twinr.proactive.wakeword.spotter import WakewordOpenWakeWordFrameSpotter

    return WakewordOpenWakeWordFrameSpotter(
        wakeword_models=tuple(config.wakeword_openwakeword_models),
        phrases=config.wakeword_phrases,
        custom_verifier_models=dict(config.wakeword_openwakeword_custom_verifier_models),
        custom_verifier_threshold=config.wakeword_openwakeword_custom_verifier_threshold,
        threshold=config.wakeword_openwakeword_threshold,
        vad_threshold=config.wakeword_openwakeword_vad_threshold,
        patience_frames=config.wakeword_openwakeword_patience_frames,
        activation_samples=config.wakeword_openwakeword_activation_samples,
        deactivation_threshold=config.wakeword_openwakeword_deactivation_threshold,
        enable_speex_noise_suppression=config.wakeword_openwakeword_enable_speex,
        inference_framework=config.wakeword_openwakeword_inference_framework,
    )


def _build_local_stt_stage1_fallback_frame_spotter(
    config: TwinrConfig,
) -> _FrameSpotter | None:
    """Keep the configured streaming detector alive behind transcript-first STT.

    Short wake-only turns such as a second `Twinna` right after one answered
    turn do not give transcript-first STT much linguistic context. On real Pi
    captures those short re-wakes can be mistranscribed into unrelated German
    text or disappear entirely even though the stream is still healthy. Keep
    the configured primary streaming detector hot as a bounded fallback so the
    same session can still recover short wake-only turns without giving up the
    richer `local_stt` same-stream tail handling for `Twinna, ...`.
    """
    if not bool(getattr(config, "wakeword_enabled", True)):
        return None
    return _build_frame_spotter(config)


def _build_local_stt_backend(
    config: TwinrConfig,
    *,
    timeout_s: float | None = None,
) -> LocalSttBackendAdapter:
    """Build the colocated local-STT backend shared by transcript-first paths."""

    local_stt_url = str(getattr(config, "voice_orchestrator_local_stt_url", "") or "").strip()
    if not local_stt_url:
        raise ValueError(
            "TWINR_VOICE_ORCHESTRATOR_LOCAL_STT_URL is required when "
            "TWINR_VOICE_ORCHESTRATOR_WAKE_STAGE1_MODE=local_stt."
        )
    if timeout_s is None:
        timeout_s = float(getattr(config, "voice_orchestrator_local_stt_timeout_s", 3.0) or 3.0)
    return LocalSttBackendAdapter(
        base_url=local_stt_url,
        bearer_token=getattr(config, "voice_orchestrator_local_stt_bearer_token", None),
        language=(
            getattr(config, "voice_orchestrator_local_stt_language", None)
            or config.openai_realtime_language
        ),
        mode=getattr(config, "voice_orchestrator_local_stt_mode", "active_listening"),
        timeout_s=float(timeout_s),
        retry_attempts=int(getattr(config, "voice_orchestrator_local_stt_retry_attempts", 1) or 0),
        retry_backoff_s=float(
            getattr(config, "voice_orchestrator_local_stt_retry_backoff_s", 0.35) or 0.0
        ),
    )


def _build_transcript_backend(config: TwinrConfig) -> _TranscriptBackend:
    """Build the transcript backend used for follow-up and barge-in scans."""

    stage1_mode = str(getattr(config, "voice_orchestrator_wake_stage1_mode", "backend") or "backend").strip().lower()
    if stage1_mode == "local_stt":
        return _build_local_stt_backend(config)
    return OpenAIBackend(config=config)


def _build_wake_phrase_spotter(
    config: TwinrConfig,
    *,
    backend: _TranscriptBackend,
) -> WakewordPhraseSpotter:
    """Build the bounded transcript matcher used for wake confirmation."""

    stage1_mode = str(getattr(config, "voice_orchestrator_wake_stage1_mode", "backend") or "backend").strip().lower()
    if stage1_mode == "local_stt":
        return WakewordPhraseSpotter(
            backend=_build_local_stt_backend(config),
            phrases=config.wakeword_phrases,
            language=(
                getattr(config, "voice_orchestrator_local_stt_language", None)
                or config.openai_realtime_language
            ),
        )
    return WakewordPhraseSpotter(
        backend=backend,
        phrases=config.wakeword_phrases,
        language=config.openai_realtime_language,
    )


def _build_backend_tail_transcript_extractor(
    config: TwinrConfig,
) -> WakewordTailTranscriptExtractor | None:
    """Build a low-latency local STT tail extractor for backend-led continuations."""

    local_stt_url = str(getattr(config, "voice_orchestrator_local_stt_url", "") or "").strip()
    if not local_stt_url:
        return None
    backend = _build_local_stt_backend(
        config,
        timeout_s=float(getattr(config, "voice_orchestrator_local_stt_tail_timeout_s", 1.25) or 1.25),
    )
    return WakewordTailTranscriptExtractor(
        backend=backend,  # type: ignore[arg-type]
        phrases=config.wakeword_phrases,
        language=config.openai_realtime_language,
    )


__all__ = ["EdgeOrchestratorVoiceSession"]
