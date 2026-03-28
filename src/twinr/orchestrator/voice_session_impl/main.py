"""Compose the refactored orchestrator voice session from focused helpers."""

from __future__ import annotations

from collections import deque
import math
import time
from typing import Any
from uuid import uuid4

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.workflows.forensics import WorkflowForensics
from twinr.hardware.household_voice_identity import HouseholdVoiceProfile
from twinr.orchestrator.voice_activation import (
    VoiceActivationPhraseMatcher,
    contextual_bias_voice_activation_phrases,
)
from twinr.orchestrator.voice_audio_debug_store import VoiceAudioDebugArtifactStore
from twinr.orchestrator.voice_forensics import VoiceFrameTelemetryBucket
from twinr.orchestrator.voice_runtime_intent import VoiceRuntimeIntentContext
from twinr.orchestrator.voice_transcript_debug_stream import VoiceTranscriptDebugStream

from ..voice_contracts import OrchestratorVoiceAudioFrame, OrchestratorVoiceRuntimeStateEvent
from .builders import build_transcript_backend, build_wake_phrase_spotter
from .observability import VoiceSessionObservabilityMixin
from .runtime_state import VoiceSessionRuntimeStateMixin
from .scanner import VoiceSessionScannerMixin
from .transcription import VoiceSessionTranscriptionMixin
from .types import _PendingTranscriptUtterance, _RecentFrame, _TranscriptBackend


class EdgeOrchestratorVoiceSessionImpl(
    VoiceSessionScannerMixin,
    VoiceSessionTranscriptionMixin,
    VoiceSessionRuntimeStateMixin,
    VoiceSessionObservabilityMixin,
):
    """Maintain one server-side streaming voice session."""

    _ACTIVE_STATES = frozenset({"waiting", "listening", "speaking", "follow_up_open"})
    _REMOTE_ASR_UTTERANCE_STATES = frozenset({"waiting", "listening"})
    _SUPPORTED_RUNTIME_STATES = frozenset({"waiting", "listening", "thinking", "speaking", "follow_up_open"})
    _WAITING_VISIBILITY_GRACE_S = 6.0
    _REMOTE_ASR_SPEECH_CONTINUE_RATIO = 0.35
    _STRONG_SPEAKER_STAGE1_WINDOW_BONUS_MS = 250
    _STRONG_SPEAKER_MIN_WAKE_DURATION_RELIEF_MS = 50

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
        self._voice_quiet_until_utc: str | None = None
        self._intent_context = VoiceRuntimeIntentContext()
        self._voice_identity_profiles: tuple[HouseholdVoiceProfile, ...] = ()
        self._voice_identity_profiles_revision: str | None = None
        self.backend = backend or build_transcript_backend(config)
        self._wake_phrase_spotter = wake_phrase_spotter or build_wake_phrase_spotter(
            config,
            backend=self.backend,
        )
        self._strong_bias_wake_phrase_spotter = build_wake_phrase_spotter(
            config,
            backend=self.backend,
            phrases=contextual_bias_voice_activation_phrases(
                getattr(config, "voice_activation_phrases", ())
            ),
        )
        self._audio_debug_store = VoiceAudioDebugArtifactStore.from_config(config)
        self._transcript_debug_stream = VoiceTranscriptDebugStream.from_config(config)
        self._forensics: WorkflowForensics | None = None
        self._trace_id: str = uuid4().hex
        self._received_frame_bucket = VoiceFrameTelemetryBucket(
            chunk_ms=self.chunk_ms,
            speech_threshold=self.speech_threshold,
        )

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
        self._voice_quiet_until_utc = None
        self._last_waiting_visible_at = None
        self._voice_identity_profiles = ()
        self._voice_identity_profiles_revision = None

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


__all__ = ["EdgeOrchestratorVoiceSessionImpl"]
