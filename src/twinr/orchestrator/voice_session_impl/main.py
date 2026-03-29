# CHANGELOG: 2026-03-29
# BUG-1: close() now fully resets transient session state (including barge-in, follow-up, trace, and runtime flags)
#        so reused session objects do not leak stale state into the next interaction.
# BUG-2: backend_name is now derived from the injected backend instead of being hard-coded to "remote_asr",
#        which fixes wrong capability/telemetry labeling for local and newer streaming backends.
# BUG-3: barge-in detection now runs before the streaming-ASR fast path, so TTS can still be interrupted when
#        a backend supports server-side utterance assembly.
# SEC-1: malformed and oversized PCM frames are now aligned, bounded, and chunked before processing to reduce
#        practical denial-of-service risk on Raspberry Pi deployments.
# IMP-1: close() now opportunistically resets backend and wake-word matcher internal buffers for cleaner reuse
#        with 2026 low-latency streaming ASR backends.
# IMP-2: audio ingest is now capability-aware and tolerant of bursty upstream chunking, which better matches
#        current streaming VAD / ASR pipelines.

"""Compose the refactored orchestrator voice session from focused helpers."""

from __future__ import annotations

from collections import deque
import logging
import math
import time
from typing import Any, Iterable
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


_LOGGER = logging.getLogger(__name__)


class EdgeOrchestratorVoiceSessionImpl(
    VoiceSessionScannerMixin,
    VoiceSessionTranscriptionMixin,
    VoiceSessionRuntimeStateMixin,
    VoiceSessionObservabilityMixin,
):
    """Maintain one server-side streaming voice session."""

    _ACTIVE_STATES = frozenset({"waiting", "listening", "speaking", "follow_up_open"})
    _REMOTE_ASR_UTTERANCE_STATES = frozenset({"waiting", "listening"})
    _SUPPORTED_RUNTIME_STATES = frozenset(
        {"waiting", "listening", "thinking", "speaking", "follow_up_open"}
    )
    _WAITING_VISIBILITY_GRACE_S = 6.0
    _REMOTE_ASR_SPEECH_CONTINUE_RATIO = 0.35
    _STRONG_SPEAKER_STAGE1_WINDOW_BONUS_MS = 250
    _STRONG_SPEAKER_MIN_WAKE_DURATION_RELIEF_MS = 50

    _PCM_SAMPLE_WIDTH_BYTES = 2
    _DEFAULT_MAX_FRAME_MS = 500
    _DEFAULT_MAX_INGEST_MS = 2000
    _SUBCHUNK_SEQUENCE_MULTIPLIER = 1000

    def __init__(
        self,
        config: TwinrConfig,
        *,
        backend: _TranscriptBackend | None = None,
        wake_phrase_spotter: VoiceActivationPhraseMatcher | None = None,
        monotonic_fn=time.monotonic,
    ) -> None:
        self.config = config
        self.sample_rate = max(1, int(config.audio_sample_rate))
        self.channels = max(1, int(config.audio_channels))
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
        self.speech_threshold = max(0, int(config.audio_speech_threshold))
        self._monotonic = monotonic_fn
        self._history: deque[_RecentFrame] = deque(
            maxlen=max(8, math.ceil(self.history_ms / self.chunk_ms))
        )

        default_max_frame_ms = max(self.chunk_ms * 4, self._DEFAULT_MAX_FRAME_MS)
        self.max_frame_ms = max(
            self.chunk_ms,
            int(
                getattr(config, "voice_orchestrator_max_frame_ms", default_max_frame_ms)
                or default_max_frame_ms
            ),
        )
        default_max_ingest_ms = max(self.max_frame_ms, self._DEFAULT_MAX_INGEST_MS)
        self.max_ingest_ms = max(
            self.max_frame_ms,
            int(
                getattr(config, "voice_orchestrator_max_ingest_ms", default_max_ingest_ms)
                or default_max_ingest_ms
            ),
        )
        bytes_per_ms = (
            self.sample_rate * self.channels * self._PCM_SAMPLE_WIDTH_BYTES
        ) / 1000.0
        self._frame_alignment_bytes = self.channels * self._PCM_SAMPLE_WIDTH_BYTES
        self._max_frame_bytes = self._align_down_to_frame_boundary(
            max(self._frame_alignment_bytes, int(math.ceil(bytes_per_ms * self.max_frame_ms)))
        )
        self._max_ingest_bytes = self._align_down_to_frame_boundary(
            max(self._max_frame_bytes, int(math.ceil(bytes_per_ms * self.max_ingest_ms)))
        )

        self.backend = backend or build_transcript_backend(config)

        activation_phrases = self._normalize_activation_phrases(
            getattr(config, "voice_activation_phrases", ())
        )
        self._wake_phrase_spotter = wake_phrase_spotter or build_wake_phrase_spotter(
            config,
            backend=self.backend,
            phrases=activation_phrases,
        )
        self._strong_bias_wake_phrase_spotter = build_wake_phrase_spotter(
            config,
            backend=self.backend,
            phrases=contextual_bias_voice_activation_phrases(activation_phrases),
            allow_twi_head_variant_recovery=True,
        )
        self._audio_debug_store = VoiceAudioDebugArtifactStore.from_config(config)
        self._transcript_debug_stream = VoiceTranscriptDebugStream.from_config(config)
        self._forensics: WorkflowForensics | None = None
        self._received_frame_bucket = VoiceFrameTelemetryBucket(
            chunk_ms=self.chunk_ms,
            speech_threshold=self.speech_threshold,
        )

        self._reset_transient_state(new_trace_id=True)

    @property
    def backend_name(self) -> str:
        """Return the selected streaming ASR backend label."""

        # BREAKING: injected backends are no longer reported as "remote_asr" unconditionally.
        return self._resolve_backend_name(self.backend)

    def close(self) -> None:
        """Reset any buffered streaming state."""

        self._flush_received_frame_bucket()
        self._history.clear()
        self._call_optional(self._received_frame_bucket, "reset", "clear")
        self._call_optional(self.backend, "reset", "clear", "flush")
        self._call_optional(self._wake_phrase_spotter, "reset", "clear", "flush")
        self._call_optional(self._strong_bias_wake_phrase_spotter, "reset", "clear", "flush")
        self._call_optional(self._audio_debug_store, "flush")
        self._call_optional(self._transcript_debug_stream, "flush")
        self._reset_transient_state(new_trace_id=True)

    def handle_audio_frame(self, frame: OrchestratorVoiceAudioFrame) -> list[dict[str, Any]]:
        """Process one streamed PCM frame and emit any server-side decisions."""

        events = self._drain_timeouts()

        runtime_state = frame.runtime_state
        if (
            isinstance(runtime_state, OrchestratorVoiceRuntimeStateEvent)
            and self._is_supported_runtime_state_event(runtime_state)
        ):
            events.extend(self.handle_runtime_state(runtime_state))

        self._sync_barge_in_gate()

        pcm_bytes = self._coerce_pcm_bytes(frame.pcm_bytes)
        if not pcm_bytes:
            return events

        sequence = self._coerce_sequence(frame.sequence)
        for part_index, chunk in enumerate(self._iter_bounded_pcm_chunks(pcm_bytes)):
            chunk_events, should_stop = self._process_pcm_chunk(
                sequence=self._sequence_for_subchunk(sequence, part_index),
                pcm_bytes=chunk,
            )
            events.extend(chunk_events)
            if should_stop:
                break

        return events

    def _process_pcm_chunk(
        self,
        *,
        sequence: int,
        pcm_bytes: bytes,
    ) -> tuple[list[dict[str, Any]], bool]:
        events: list[dict[str, Any]] = []

        self._received_frame_bucket.add_frame(sequence=sequence, pcm_bytes=pcm_bytes)
        if self._received_frame_bucket.should_flush():
            self._flush_received_frame_bucket()

        self._remember_frame(pcm_bytes)

        if self._state == "speaking" and not self._barge_in_sent:
            barge_in_event = self._maybe_detect_barge_in_candidate()
            if barge_in_event is not None:
                self._barge_in_sent = True
                self._state = "thinking"
                events.append(barge_in_event.to_payload())
                return events, True

        if self._should_use_streaming_utterance_path():
            utterance_event = self._advance_remote_asr_utterance()
            if utterance_event is not None:
                events.append(utterance_event.to_payload())
                return events, True

        return events, False

    def _should_use_streaming_utterance_path(self) -> bool:
        try:
            state_owns_streaming_utterances = bool(self._uses_remote_asr_utterance_path())
        except AttributeError:
            state_owns_streaming_utterances = self._state in self._REMOTE_ASR_UTTERANCE_STATES
        if not state_owns_streaming_utterances:
            return False

        explicit_capability = self._read_backend_capability(
            "supports_streaming_utterances",
            "streaming_utterances",
            "use_remote_asr_utterances",
        )
        if explicit_capability is not None:
            return explicit_capability

        return state_owns_streaming_utterances

    def _reset_transient_state(self, *, new_trace_id: bool) -> None:
        if new_trace_id or not getattr(self, "_trace_id", ""):
            self._trace_id = uuid4().hex
        self._next_wake_candidate_check_at = 0.0
        self._next_barge_in_candidate_check_at = 0.0
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
        self._forensics = None

    def _sync_barge_in_gate(self) -> None:
        if self._state != "speaking":
            self._barge_in_sent = False

    def _iter_bounded_pcm_chunks(self, pcm_bytes: bytes) -> Iterable[bytes]:
        sanitized = self._sanitize_pcm_bytes(pcm_bytes)
        if not sanitized:
            return

        if len(sanitized) <= self._max_frame_bytes:
            yield sanitized
            return

        for start in range(0, len(sanitized), self._max_frame_bytes):
            yield sanitized[start : start + self._max_frame_bytes]

    def _sanitize_pcm_bytes(self, pcm_bytes: bytes) -> bytes:
        if not pcm_bytes:
            return b""

        if len(pcm_bytes) > self._max_ingest_bytes:
            # BREAKING: oversized upstream frames are truncated after max_ingest_ms to bound resource use.
            _LOGGER.warning(
                "Dropping trailing audio beyond %d ms (%d bytes) for bounded ingest",
                self.max_ingest_ms,
                self._max_ingest_bytes,
            )
            pcm_bytes = pcm_bytes[: self._max_ingest_bytes]

        aligned_size = self._align_down_to_frame_boundary(len(pcm_bytes))
        if aligned_size <= 0:
            return b""

        if aligned_size != len(pcm_bytes):
            pcm_bytes = pcm_bytes[:aligned_size]

        return pcm_bytes

    def _align_down_to_frame_boundary(self, size_bytes: int) -> int:
        if size_bytes <= 0:
            return 0
        remainder = size_bytes % self._frame_alignment_bytes
        return size_bytes - remainder

    def _sequence_for_subchunk(self, base_sequence: int, part_index: int) -> int:
        if part_index <= 0:
            return base_sequence
        return (base_sequence * self._SUBCHUNK_SEQUENCE_MULTIPLIER) + part_index

    def _coerce_sequence(self, value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _coerce_pcm_bytes(self, value: Any) -> bytes:
        if value is None:
            return b""
        if isinstance(value, bytes):
            return value
        if isinstance(value, memoryview):
            return value.tobytes()
        if isinstance(value, bytearray):
            return bytes(value)
        try:
            return bytes(value)
        except (TypeError, ValueError):
            return b""

    def _is_supported_runtime_state_event(
        self,
        runtime_state: OrchestratorVoiceRuntimeStateEvent,
    ) -> bool:
        state_value = getattr(runtime_state, "state", None)
        if state_value is None:
            return True
        if not isinstance(state_value, str):
            return False
        return state_value in self._SUPPORTED_RUNTIME_STATES

    def _normalize_activation_phrases(self, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            normalized = (value.strip(),)
        else:
            try:
                normalized = tuple(
                    phrase.strip()
                    for phrase in value
                    if isinstance(phrase, str) and phrase.strip()
                )
            except TypeError:
                normalized = ()
        return normalized

    def _resolve_backend_name(self, backend: Any) -> str:
        for attr_name in ("backend_name", "name", "label"):
            candidate = getattr(backend, attr_name, None)
            if callable(candidate):
                try:
                    candidate = candidate()
                except TypeError:
                    continue
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()

        backend_type_name = type(backend).__name__.strip()
        normalized_backend_type_name = backend_type_name.lower().replace("-", "_")
        if "remote_asr" in normalized_backend_type_name or "remoteasr" in normalized_backend_type_name:
            return "remote_asr"
        if str(getattr(self.config, "voice_orchestrator_remote_asr_url", "") or "").strip():
            return "remote_asr"
        if backend_type_name:
            return backend_type_name

        return "remote_asr"

    def _read_backend_capability(self, *attr_names: str) -> bool | None:
        for attr_name in attr_names:
            candidate = getattr(self.backend, attr_name, None)
            if callable(candidate):
                try:
                    candidate = candidate()
                except TypeError:
                    continue
            if isinstance(candidate, bool):
                return candidate
        return None

    def _call_optional(self, target: Any, *method_names: str) -> None:
        if target is None:
            return
        for method_name in method_names:
            method = getattr(target, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    _LOGGER.exception(
                        "Optional reset hook %s.%s() failed",
                        type(target).__name__,
                        method_name,
                    )
                return


__all__ = ["EdgeOrchestratorVoiceSessionImpl"]
