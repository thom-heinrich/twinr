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
from dataclasses import dataclass
import math
import time
from typing import Any, Protocol

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.audio import AmbientAudioCaptureWindow, AmbientAudioLevelSample
from twinr.proactive.wakeword.kws import WakewordSherpaOnnxFrameSpotter
from twinr.proactive.wakeword.matching import WakewordPhraseSpotter, WakewordMatch
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
        backend: OpenAIBackend | None = None,
        frame_spotter: _FrameSpotter | None = None,
        wake_phrase_spotter: WakewordPhraseSpotter | None = None,
        monotonic_fn=time.monotonic,
    ) -> None:
        self.config = config
        self.backend = backend or OpenAIBackend(config=config)
        self.sample_rate = int(config.audio_sample_rate)
        self.channels = int(config.audio_channels)
        self.chunk_ms = max(20, int(getattr(config, "audio_chunk_ms", 100) or 100))
        self.history_ms = max(
            1500,
            int(getattr(config, "voice_orchestrator_history_ms", 4000) or 4000),
        )
        self.wake_postroll_ms = max(
            self.chunk_ms,
            int(getattr(config, "voice_orchestrator_wake_postroll_ms", 900) or 900),
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
        self.speech_threshold = int(config.audio_speech_threshold)
        self._monotonic = monotonic_fn
        self._history: deque[_RecentFrame] = deque(maxlen=max(8, math.ceil(self.history_ms / self.chunk_ms)))
        self._pending_wake: _PendingWakeConfirmation | None = None
        self._follow_up_deadline_at: float | None = None
        self._next_follow_up_candidate_check_at: float = 0.0
        self._next_barge_in_candidate_check_at: float = 0.0
        self._barge_in_sent = False
        self._session_id = ""
        self._state = "waiting"
        self._follow_up_allowed = False
        self._frame_spotter = frame_spotter or _build_frame_spotter(config)
        self._wake_phrase_spotter = wake_phrase_spotter or WakewordPhraseSpotter(
            backend=self.backend,
            phrases=config.wakeword_phrases,
            language=config.openai_realtime_language,
        )

    @property
    def backend_name(self) -> str:
        """Return the selected streaming wakeword backend label."""

        return normalize_wakeword_backend(
            getattr(self.config, "wakeword_primary_backend", self.config.wakeword_backend),
            default="openwakeword",
        )

    def close(self) -> None:
        """Reset any buffered streaming state."""

        self._history.clear()
        self._pending_wake = None
        self._follow_up_deadline_at = None
        self._frame_spotter.reset()

    def handle_hello(self, request: OrchestratorVoiceHelloRequest) -> list[dict[str, Any]]:
        """Accept one new edge voice session and validate stream metadata."""

        self._session_id = request.session_id
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
        return [OrchestratorVoiceReadyEvent(session_id=request.session_id, backend=self.backend_name).to_payload()]

    def handle_runtime_state(self, event: OrchestratorVoiceRuntimeStateEvent) -> list[dict[str, Any]]:
        """Update explicit edge runtime state and drain any timeout-based events."""

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
            self._frame_spotter.reset()
        return self._drain_timeouts()

    def handle_audio_frame(self, frame: OrchestratorVoiceAudioFrame) -> list[dict[str, Any]]:
        """Process one streamed PCM frame and emit any server-side decisions."""

        del frame.sequence
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
        if self._state in {"waiting", "wake_armed"}:
            match = self._frame_spotter.process_pcm_bytes(pcm_bytes, channels=self.channels)
            if match is not None and match.detected:
                self._pending_wake = _PendingWakeConfirmation(
                    match=match,
                    remaining_postroll_ms=self.wake_postroll_ms,
                )
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
                self._state = "listening"
                self._follow_up_deadline_at = None
                events.append(follow_up_event.to_payload())
        return events

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
        frame_duration_ms = max(
            self.chunk_ms,
            int(round((byte_count / max(1, self.channels * 2 * self.sample_rate)) * 1000.0)),
        )
        pending.remaining_postroll_ms -= frame_duration_ms
        if pending.remaining_postroll_ms > 0:
            return None
        self._pending_wake = None
        capture = self._recent_capture_window(self.history_ms)
        confirmed = self._wake_phrase_spotter.detect(capture)
        if not confirmed.detected:
            self._frame_spotter.reset()
            return None
        self._state = "thinking"
        return OrchestratorVoiceWakeConfirmedEvent(
            matched_phrase=confirmed.matched_phrase,
            remaining_text=confirmed.remaining_text,
            backend=confirmed.backend or self.backend_name,
            detector_label=confirmed.detector_label,
            score=confirmed.score,
        )

    def _maybe_detect_speech_candidate(
        self,
        *,
        kind: str,
    ) -> OrchestratorVoiceBargeInInterruptEvent | OrchestratorVoiceFollowUpCaptureRequestedEvent | None:
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
        if capture.sample.active_chunk_count <= 0 or capture.sample.active_ratio < min_active_ratio:
            return None
        transcript = self.backend.transcribe(
            _pcm_capture_to_wav_bytes(capture),
            filename="voice-window.wav",
            content_type="audio/wav",
            language=self.config.openai_realtime_language,
        ).strip()
        if _normalize_text_length(transcript) < min_chars:
            return None
        if kind == "barge_in":
            return OrchestratorVoiceBargeInInterruptEvent(transcript_preview=transcript[:160])
        return OrchestratorVoiceFollowUpCaptureRequestedEvent(transcript_preview=transcript[:160])

    def _recent_capture_window(self, duration_ms: int) -> AmbientAudioCaptureWindow:
        target_ms = max(self.chunk_ms, int(duration_ms))
        pcm_fragments: list[bytes] = []
        rms_values: list[int] = []
        collected_ms = 0
        for frame in reversed(self._history):
            pcm_fragments.append(frame.pcm_bytes)
            rms_values.append(frame.rms)
            collected_ms += frame.duration_ms
            if collected_ms >= target_ms:
                break
        if not pcm_fragments:
            pcm_fragments = [b""]
            rms_values = [0]
        pcm_fragments.reverse()
        rms_values.reverse()
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


__all__ = ["EdgeOrchestratorVoiceSession"]
