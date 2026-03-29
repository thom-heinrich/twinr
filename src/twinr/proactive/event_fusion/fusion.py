# CHANGELOG: 2026-03-29
# BUG-1: Stop silent confidence inflation from duplicate vision/audio support being re-added across ticks.
# BUG-2: Require temporal/causal consistency for multimodal and ordered vision claims to prevent unrelated evidence from fusing.
# BUG-3: Avoid unnecessary keyframe-review planning for non-review claims; this removed repeated per-tick overhead on edge hardware.
# SEC-1: Reject non-finite / stale timestamps and bound duplicate evidence amplification so malformed upstream data cannot poison or DoS the tracker.
# IMP-1: Upgrade to calibration-aware, bucket-compressed evidence aggregation with modality agreement and structural consistency scoring.
# IMP-2: Make fusion deterministic and edge-friendly with single-pass buffer indexing, monotonic-time guarding, and thread-safe observation updates.

"""Fuse short-window audio and vision evidence into conservative event claims."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
from threading import RLock

from twinr.proactive.event_fusion.audio_events import (
    AudioClassifierHints,
    AudioEventConfig,
    AudioEventKind,
    AudioMicroEvent,
    derive_audio_micro_events,
)
from twinr.proactive.event_fusion.buffers import RollingWindowBuffer
from twinr.proactive.event_fusion.claims import (
    EventFusionPolicyContext,
    FusedEventClaim,
    FusionActionLevel,
    build_fused_claim,
)
from twinr.proactive.event_fusion.review import (
    KeyframeReviewConfig,
    build_keyframe_review_plan,
)
from twinr.proactive.event_fusion.vision_sequences import (
    VisionSequenceConfig,
    VisionSequenceEvent,
    VisionSequenceKind,
    derive_vision_sequences,
)
from twinr.proactive.social.engine import SocialAudioObservation, SocialObservation, SocialVisionObservation


_FLOAT_EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class MultimodalEventFusionConfig:
    """Store horizons, alignment constraints, and thresholds for the fusion tracker."""

    audio_horizon_s: float = 8.0
    audio_event_horizon_s: float = 8.0
    vision_horizon_s: float = 8.0
    vision_sequence_horizon_s: float = 8.0
    fusion_horizon_s: float = 8.0
    temporal_decay_half_life_audio_s: float = 2.0
    temporal_decay_half_life_vision_s: float = 3.5
    evidence_bucket_s: float = 1.0
    multi_scale_windows_s: tuple[float, ...] = (2.0, 4.0, 8.0)
    max_cross_modal_gap_s: float = 1.25
    max_causal_gap_s: float = 2.5
    overlap_tolerance_s: float = 0.35
    dedupe_time_quantum_s: float = 0.25
    duplicate_fingerprint_ttl_s: float = 12.0
    out_of_order_tolerance_s: float = 0.25
    max_evidence_entries_per_modality: int = 8
    audio_events: AudioEventConfig = field(default_factory=AudioEventConfig)
    vision_sequences: VisionSequenceConfig = field(default_factory=VisionSequenceConfig)
    keyframe_review: KeyframeReviewConfig = field(default_factory=KeyframeReviewConfig)

    def __post_init__(self) -> None:
        positive_names = (
            "audio_horizon_s",
            "audio_event_horizon_s",
            "vision_horizon_s",
            "vision_sequence_horizon_s",
            "fusion_horizon_s",
            "temporal_decay_half_life_audio_s",
            "temporal_decay_half_life_vision_s",
            "evidence_bucket_s",
            "max_cross_modal_gap_s",
            "max_causal_gap_s",
            "dedupe_time_quantum_s",
            "duplicate_fingerprint_ttl_s",
        )
        non_negative_names = (
            "overlap_tolerance_s",
            "out_of_order_tolerance_s",
        )
        for name in positive_names:
            _require_positive_finite(name, getattr(self, name))
        for name in non_negative_names:
            _require_non_negative_finite(name, getattr(self, name))
        if self.max_evidence_entries_per_modality <= 0:
            raise ValueError("max_evidence_entries_per_modality must be >= 1")
        windows = tuple(
            sorted(
                {
                    float(window_s)
                    for window_s in self.multi_scale_windows_s
                    if isinstance(window_s, (int, float)) and math.isfinite(window_s) and window_s > 0.0
                }
            )
        )
        if not windows:
            raise ValueError("multi_scale_windows_s must contain at least one positive finite window")
        object.__setattr__(self, "multi_scale_windows_s", windows)


@dataclass(frozen=True, slots=True)
class _EvidenceEntry:
    """Compact timestamped confidence entry used for bounded aggregation."""

    timestamp: float
    confidence: float
    modality: str


@dataclass(frozen=True, slots=True)
class _SupportMatch:
    """Store temporally consistent support and a structural consistency score."""

    audio_support: tuple[AudioMicroEvent, ...] = ()
    vision_support: tuple[VisionSequenceEvent, ...] = ()
    structural_score: float = 0.0


class MultimodalEventFusionTracker:
    """Track rolling multimodal history and emit conservative fused claims."""

    def __init__(self, config: MultimodalEventFusionConfig | None = None) -> None:
        """Initialize one compact in-memory event-fusion tracker."""

        self.config = config or MultimodalEventFusionConfig()
        self.audio_observations = RollingWindowBuffer[SocialAudioObservation](horizon_s=self.config.audio_horizon_s)
        self.vision_observations = RollingWindowBuffer[SocialVisionObservation](horizon_s=self.config.vision_horizon_s)
        self.audio_events = RollingWindowBuffer[AudioMicroEvent](horizon_s=self.config.audio_event_horizon_s)
        self.vision_sequences = RollingWindowBuffer[VisionSequenceEvent](horizon_s=self.config.vision_sequence_horizon_s)
        self._lock = RLock()
        self._last_observed_at: float | None = None
        self._seen_audio_event_fingerprints: dict[tuple[object, ...], float] = {}
        self._seen_vision_sequence_fingerprints: dict[tuple[object, ...], float] = {}

    def observe(
        self,
        observation: SocialObservation,
        *,
        audio_hints: AudioClassifierHints | None = None,
        room_busy_or_overlapping: bool = False,
    ) -> tuple[FusedEventClaim, ...]:
        """Ingest one observation tick and return current active fused claims."""

        with self._lock:
            raw_now = observation.observed_at
            if not _is_valid_timestamp(raw_now):
                return ()

            if self._last_observed_at is not None:
                if raw_now + self.config.out_of_order_tolerance_s < self._last_observed_at:
                    return derive_fused_event_claims(
                        now=self._last_observed_at,
                        audio_event_buffer=self.audio_events,
                        vision_observation_buffer=self.vision_observations,
                        vision_sequence_buffer=self.vision_sequences,
                        policy_context=EventFusionPolicyContext.from_observation(
                            observation,
                            room_busy_or_overlapping=room_busy_or_overlapping,
                        ),
                        config=self.config,
                    )
                now = max(raw_now, self._last_observed_at)
            else:
                now = raw_now
            self._last_observed_at = now

            self.audio_observations.append(now, observation.audio)
            audio_events = derive_audio_micro_events(
                observed_at=now,
                observation=observation.audio,
                hints=audio_hints,
                config=self.config.audio_events,
            )
            for event in audio_events:
                if _is_valid_audio_event(event, now=now):
                    self._append_unique_audio_event(event)

            if observation.inspected:
                self.vision_observations.append(now, observation.vision)
                vision_sequences = derive_vision_sequences(
                    observation_buffer=self.vision_observations,
                    now=now,
                    config=self.config.vision_sequences,
                )
                for sequence in vision_sequences:
                    if _is_valid_vision_sequence(sequence, now=now):
                        self._append_unique_vision_sequence(sequence)

            policy = EventFusionPolicyContext.from_observation(
                observation,
                room_busy_or_overlapping=room_busy_or_overlapping,
            )
            return derive_fused_event_claims(
                now=now,
                audio_event_buffer=self.audio_events,
                vision_observation_buffer=self.vision_observations,
                vision_sequence_buffer=self.vision_sequences,
                policy_context=policy,
                config=self.config,
            )

    def _append_unique_audio_event(self, event: AudioMicroEvent) -> None:
        """Append one audio event only if it is not a recent duplicate."""

        timestamp = float(event.observed_at)
        self._prune_fingerprint_cache(self._seen_audio_event_fingerprints, now=timestamp)
        fingerprint = _audio_event_fingerprint(
            event,
            quantum_s=self.config.dedupe_time_quantum_s,
        )
        if fingerprint in self._seen_audio_event_fingerprints:
            return
        self._seen_audio_event_fingerprints[fingerprint] = timestamp
        self.audio_events.append(timestamp, event)

    def _append_unique_vision_sequence(self, sequence: VisionSequenceEvent) -> None:
        """Append one vision sequence only if it is not a recent duplicate."""

        timestamp = float(sequence.window_end_s)
        self._prune_fingerprint_cache(self._seen_vision_sequence_fingerprints, now=timestamp)
        fingerprint = _vision_sequence_fingerprint(
            sequence,
            quantum_s=self.config.dedupe_time_quantum_s,
        )
        if fingerprint in self._seen_vision_sequence_fingerprints:
            return
        self._seen_vision_sequence_fingerprints[fingerprint] = timestamp
        self.vision_sequences.append(timestamp, sequence)

    def _prune_fingerprint_cache(self, cache: dict[tuple[object, ...], float], *, now: float) -> None:
        """Drop stale duplicate fingerprints to keep the cache compact."""

        min_allowed = now - self.config.duplicate_fingerprint_ttl_s
        stale_keys = [fingerprint for fingerprint, ts in cache.items() if ts < min_allowed]
        for fingerprint in stale_keys:
            cache.pop(fingerprint, None)


def derive_fused_event_claims(
    *,
    now: float,
    audio_event_buffer: RollingWindowBuffer[AudioMicroEvent],
    vision_observation_buffer: RollingWindowBuffer[SocialVisionObservation],
    vision_sequence_buffer: RollingWindowBuffer[VisionSequenceEvent],
    policy_context: EventFusionPolicyContext,
    config: MultimodalEventFusionConfig,
) -> tuple[FusedEventClaim, ...]:
    """Return active fused claims from recent multimodal evidence windows."""

    recent_audio = _recent_audio_events_by_kind(
        buffer=audio_event_buffer,
        now=now,
        horizon_s=config.fusion_horizon_s,
        dedupe_time_quantum_s=config.dedupe_time_quantum_s,
    )
    recent_vision = _recent_vision_sequences_by_kind(
        buffer=vision_sequence_buffer,
        now=now,
        horizon_s=config.fusion_horizon_s,
        dedupe_time_quantum_s=config.dedupe_time_quantum_s,
    )

    claims: list[FusedEventClaim] = []

    shout_audio = recent_audio.get(AudioEventKind.SHOUT_LIKE_AUDIO, ())
    if shout_audio:
        claims.append(
            _build_claim_from_support(
                state="shout_like_audio",
                preferred_action_level=FusionActionLevel.PROMPT_ONLY,
                audio_support=shout_audio,
                vision_support=(),
                expected_modalities=1,
                policy_context=policy_context,
                now=now,
                config=config,
                vision_observation_buffer=vision_observation_buffer,
                structural_score=1.0,
            )
        )

    laugh_match = _aligned_audio_vision_support(
        audio_support=recent_audio.get(AudioEventKind.LAUGH_LIKE_AUDIO, ()),
        vision_support=recent_vision.get(VisionSequenceKind.POSITIVE_CONTACT, ()),
        max_gap_s=config.max_cross_modal_gap_s,
        dedupe_time_quantum_s=config.dedupe_time_quantum_s,
    )
    if laugh_match.audio_support and laugh_match.vision_support:
        claims.append(
            _build_claim_from_support(
                state="laugh_like_positive_contact",
                preferred_action_level=FusionActionLevel.PROMPT_ONLY,
                audio_support=laugh_match.audio_support,
                vision_support=laugh_match.vision_support,
                expected_modalities=2,
                policy_context=policy_context,
                now=now,
                config=config,
                vision_observation_buffer=vision_observation_buffer,
                structural_score=laugh_match.structural_score,
            )
        )

    cry_audio = recent_audio.get(AudioEventKind.CRY_LIKE_AUDIO, ())
    slumped = recent_vision.get(VisionSequenceKind.SLUMPED_QUIET, ())
    distress_audio = _dedupe_audio_support(
        shout_audio + cry_audio,
        quantum_s=config.dedupe_time_quantum_s,
    )
    distress_match = _aligned_audio_vision_support(
        audio_support=distress_audio,
        vision_support=slumped,
        max_gap_s=config.max_cross_modal_gap_s,
        dedupe_time_quantum_s=config.dedupe_time_quantum_s,
    )
    if distress_match.audio_support and distress_match.vision_support:
        claims.append(
            _build_claim_from_support(
                state="distress_possible",
                preferred_action_level=FusionActionLevel.REVIEW_ONLY,
                audio_support=distress_match.audio_support,
                vision_support=distress_match.vision_support,
                expected_modalities=2,
                policy_context=policy_context,
                now=now,
                config=config,
                vision_observation_buffer=vision_observation_buffer,
                review_preferred=True,
                structural_score=distress_match.structural_score,
            )
        )

    cry_distress_match = _aligned_audio_vision_support(
        audio_support=cry_audio,
        vision_support=slumped,
        max_gap_s=config.max_cross_modal_gap_s,
        dedupe_time_quantum_s=config.dedupe_time_quantum_s,
    )
    if cry_distress_match.audio_support and cry_distress_match.vision_support:
        claims.append(
            _build_claim_from_support(
                state="cry_like_distress_possible",
                preferred_action_level=FusionActionLevel.REVIEW_ONLY,
                audio_support=cry_distress_match.audio_support,
                vision_support=cry_distress_match.vision_support,
                expected_modalities=2,
                policy_context=policy_context,
                now=now,
                config=config,
                vision_observation_buffer=vision_observation_buffer,
                review_preferred=True,
                structural_score=cry_distress_match.structural_score,
            )
        )

    fall_match = _ordered_vision_chain(
        first_support=recent_vision.get(VisionSequenceKind.DOWNWARD_TRANSITION, ()),
        second_support=recent_vision.get(VisionSequenceKind.FLOOR_POSE_ENTERED, ()),
        max_gap_s=config.max_causal_gap_s,
        overlap_tolerance_s=config.overlap_tolerance_s,
        dedupe_time_quantum_s=config.dedupe_time_quantum_s,
    )
    if fall_match.vision_support:
        claims.append(
            _build_claim_from_support(
                state="possible_fall",
                preferred_action_level=FusionActionLevel.REVIEW_ONLY,
                audio_support=(),
                vision_support=fall_match.vision_support,
                expected_modalities=1,
                policy_context=policy_context,
                now=now,
                config=config,
                vision_observation_buffer=vision_observation_buffer,
                review_preferred=True,
                structural_score=fall_match.structural_score,
            )
        )

    post_drop_match = _post_drop_stillness_support(
        drop_support=_dedupe_vision_support(
            recent_vision.get(VisionSequenceKind.DOWNWARD_TRANSITION, ())
            + recent_vision.get(VisionSequenceKind.FLOOR_POSE_ENTERED, ()),
            quantum_s=config.dedupe_time_quantum_s,
        ),
        stillness_support=recent_vision.get(VisionSequenceKind.FLOOR_STILLNESS, ()),
        max_gap_s=config.max_causal_gap_s,
        overlap_tolerance_s=config.overlap_tolerance_s,
        dedupe_time_quantum_s=config.dedupe_time_quantum_s,
    )
    if post_drop_match.vision_support:
        claims.append(
            _build_claim_from_support(
                state="floor_stillness_after_drop",
                # BREAKING: stillness following a drop now prefers REVIEW_ONLY because the ordered temporal pattern is materially higher risk.
                preferred_action_level=FusionActionLevel.REVIEW_ONLY,
                audio_support=(),
                vision_support=post_drop_match.vision_support,
                expected_modalities=1,
                policy_context=policy_context,
                now=now,
                config=config,
                vision_observation_buffer=vision_observation_buffer,
                review_preferred=True,
                structural_score=post_drop_match.structural_score,
            )
        )

    claims = _suppress_redundant_claims(claims)
    claims.sort(key=_claim_sort_key, reverse=True)
    return tuple(claims)


def _recent_audio_events_by_kind(
    *,
    buffer: RollingWindowBuffer[AudioMicroEvent],
    now: float,
    horizon_s: float,
    dedupe_time_quantum_s: float,
) -> dict[AudioEventKind, tuple[AudioMicroEvent, ...]]:
    """Index recent active audio events by kind with one buffer scan."""

    recent = buffer.between(max(0.0, now - horizon_s), now)
    indexed: dict[AudioEventKind, list[AudioMicroEvent]] = defaultdict(list)
    for sample in recent:
        event = sample.value
        if not _is_valid_audio_event(event, now=now):
            continue
        indexed[event.kind].append(event)
    return {
        kind: _dedupe_audio_support(tuple(events), quantum_s=dedupe_time_quantum_s)
        for kind, events in indexed.items()
    }


def _recent_vision_sequences_by_kind(
    *,
    buffer: RollingWindowBuffer[VisionSequenceEvent],
    now: float,
    horizon_s: float,
    dedupe_time_quantum_s: float,
) -> dict[VisionSequenceKind, tuple[VisionSequenceEvent, ...]]:
    """Index recent active vision sequences by kind with one buffer scan."""

    recent = buffer.between(max(0.0, now - horizon_s), now)
    indexed: dict[VisionSequenceKind, list[VisionSequenceEvent]] = defaultdict(list)
    for sample in recent:
        event = sample.value
        if not _is_valid_vision_sequence(event, now=now):
            continue
        indexed[event.kind].append(event)
    return {
        kind: _dedupe_vision_support(tuple(events), quantum_s=dedupe_time_quantum_s)
        for kind, events in indexed.items()
    }


def _aligned_audio_vision_support(
    *,
    audio_support: tuple[AudioMicroEvent, ...],
    vision_support: tuple[VisionSequenceEvent, ...],
    max_gap_s: float,
    dedupe_time_quantum_s: float,
) -> _SupportMatch:
    """Select temporally aligned audio and vision evidence inside a compact gap."""

    if not audio_support or not vision_support:
        return _SupportMatch()

    audio_support = _dedupe_audio_support(audio_support, quantum_s=dedupe_time_quantum_s)
    vision_support = _dedupe_vision_support(vision_support, quantum_s=dedupe_time_quantum_s)

    matched_audio: list[AudioMicroEvent] = []
    matched_vision: list[VisionSequenceEvent] = []
    pair_scores: list[float] = []

    for vision_event in vision_support:
        best_audio: AudioMicroEvent | None = None
        best_pair_score = -1.0
        for audio_event in audio_support:
            gap_s = _distance_point_to_interval(
                point_s=audio_event.observed_at,
                start_s=vision_event.window_start_s,
                end_s=vision_event.window_end_s,
            )
            if gap_s > max_gap_s:
                continue
            gap_score = 1.0 - min(1.0, gap_s / max(max_gap_s, _FLOAT_EPSILON))
            pair_score = 0.6 * gap_score + 0.4 * ((_safe_confidence(audio_event.confidence) + _safe_confidence(vision_event.confidence)) / 2.0)
            if pair_score > best_pair_score:
                best_pair_score = pair_score
                best_audio = audio_event
        if best_audio is not None:
            matched_audio.append(best_audio)
            matched_vision.append(vision_event)
            pair_scores.append(best_pair_score)

    if not pair_scores:
        return _SupportMatch()

    return _SupportMatch(
        audio_support=_dedupe_audio_support(tuple(matched_audio), quantum_s=dedupe_time_quantum_s),
        vision_support=_dedupe_vision_support(tuple(matched_vision), quantum_s=dedupe_time_quantum_s),
        structural_score=round(sum(pair_scores) / len(pair_scores), 4),
    )


def _ordered_vision_chain(
    *,
    first_support: tuple[VisionSequenceEvent, ...],
    second_support: tuple[VisionSequenceEvent, ...],
    max_gap_s: float,
    overlap_tolerance_s: float,
    dedupe_time_quantum_s: float,
) -> _SupportMatch:
    """Return the best causally ordered pair of vision sequences."""

    if not first_support or not second_support:
        return _SupportMatch()

    first_support = _dedupe_vision_support(first_support, quantum_s=dedupe_time_quantum_s)
    second_support = _dedupe_vision_support(second_support, quantum_s=dedupe_time_quantum_s)

    best_pair: tuple[VisionSequenceEvent, VisionSequenceEvent] | None = None
    best_score = -1.0
    scale = max(max_gap_s, overlap_tolerance_s, 1.0)

    for first_event in first_support:
        for second_event in second_support:
            gap_s = second_event.window_start_s - first_event.window_end_s
            if gap_s < -overlap_tolerance_s or gap_s > max_gap_s:
                continue
            order_score = 1.0 - min(1.0, abs(gap_s) / scale)
            pair_score = 0.65 * order_score + 0.35 * ((_safe_confidence(first_event.confidence) + _safe_confidence(second_event.confidence)) / 2.0)
            if pair_score > best_score:
                best_score = pair_score
                best_pair = (first_event, second_event)

    if best_pair is None:
        return _SupportMatch()

    return _SupportMatch(
        vision_support=best_pair,
        structural_score=round(best_score, 4),
    )


def _post_drop_stillness_support(
    *,
    drop_support: tuple[VisionSequenceEvent, ...],
    stillness_support: tuple[VisionSequenceEvent, ...],
    max_gap_s: float,
    overlap_tolerance_s: float,
    dedupe_time_quantum_s: float,
) -> _SupportMatch:
    """Return the best stillness-after-drop support with causal ordering."""

    if not drop_support or not stillness_support:
        return _SupportMatch()

    drop_support = _dedupe_vision_support(drop_support, quantum_s=dedupe_time_quantum_s)
    stillness_support = _dedupe_vision_support(stillness_support, quantum_s=dedupe_time_quantum_s)

    best_pair: tuple[VisionSequenceEvent, VisionSequenceEvent] | None = None
    best_score = -1.0
    scale = max(max_gap_s, overlap_tolerance_s, 1.0)

    for drop_event in drop_support:
        for stillness_event in stillness_support:
            if stillness_event.window_end_s + overlap_tolerance_s < drop_event.window_end_s:
                continue
            gap_s = stillness_event.window_start_s - drop_event.window_end_s
            if gap_s < -overlap_tolerance_s or gap_s > max_gap_s:
                continue
            order_score = 1.0 - min(1.0, abs(gap_s) / scale)
            stillness_duration_s = max(0.0, stillness_event.window_end_s - stillness_event.window_start_s)
            duration_score = min(1.0, stillness_duration_s / max(1.0, max_gap_s))
            pair_score = (
                0.45 * order_score
                + 0.25 * duration_score
                + 0.30 * ((_safe_confidence(drop_event.confidence) + _safe_confidence(stillness_event.confidence)) / 2.0)
            )
            if pair_score > best_score:
                best_score = pair_score
                best_pair = (drop_event, stillness_event)

    if best_pair is None:
        return _SupportMatch()

    return _SupportMatch(
        vision_support=best_pair,
        structural_score=round(best_score, 4),
    )


def _build_claim_from_support(
    *,
    state: str,
    preferred_action_level: FusionActionLevel,
    audio_support: tuple[AudioMicroEvent, ...],
    vision_support: tuple[VisionSequenceEvent, ...],
    expected_modalities: int,
    policy_context: EventFusionPolicyContext,
    now: float,
    config: MultimodalEventFusionConfig,
    vision_observation_buffer: RollingWindowBuffer[SocialVisionObservation],
    review_preferred: bool = False,
    structural_score: float = 1.0,
) -> FusedEventClaim:
    """Build one claim from all matching evidence inside the current window."""

    audio_support = _dedupe_audio_support(audio_support, quantum_s=config.dedupe_time_quantum_s)
    vision_support = _dedupe_vision_support(vision_support, quantum_s=config.dedupe_time_quantum_s)
    confidence = _aggregate_temporal_confidence(
        now=now,
        audio_support=audio_support,
        vision_support=vision_support,
        expected_modalities=expected_modalities,
        structural_score=structural_score,
        config=config,
    )
    window_start_s = _window_start(audio_support=audio_support, vision_support=vision_support)
    window_end_s = _window_end(audio_support=audio_support, vision_support=vision_support)
    review_plan = None
    if review_preferred and vision_support and window_start_s is not None and window_end_s is not None:
        review_plan = build_keyframe_review_plan(
            claim_state=state,
            observation_buffer=vision_observation_buffer,
            vision_evidence=vision_support,
            audio_evidence=audio_support,
            window_start_s=window_start_s,
            window_end_s=window_end_s,
            config=config.keyframe_review,
        )
    return build_fused_claim(
        state=state,
        confidence=confidence,
        source="temporal_multiscale_evidence_fusion",
        policy_context=policy_context,
        window_start_s=window_start_s,
        window_end_s=window_end_s,
        preferred_action_level=preferred_action_level,
        supporting_audio_events=tuple(sorted({event.kind.value for event in audio_support})),
        supporting_vision_events=tuple(sorted({event.kind.value for event in vision_support})),
        review_recommended=review_preferred and review_plan is not None,
        keyframe_review_plan=review_plan,
    )


def _aggregate_temporal_confidence(
    *,
    now: float,
    audio_support: tuple[AudioMicroEvent, ...],
    vision_support: tuple[VisionSequenceEvent, ...],
    expected_modalities: int,
    structural_score: float,
    config: MultimodalEventFusionConfig,
) -> float:
    """Aggregate recent support using bounded, time-decayed, calibration-aware evidence."""

    decayed_entries = _compress_decayed_entries(
        entries=_decayed_entries(
            now=now,
            audio_support=audio_support,
            vision_support=vision_support,
            config=config,
        ),
        now=now,
        bucket_s=config.evidence_bucket_s,
        max_entries_per_modality=config.max_evidence_entries_per_modality,
    )
    if not decayed_entries:
        return 0.0

    decayed_confidences = tuple(entry.confidence for entry in decayed_entries)
    base_score = sum(decayed_confidences) / len(decayed_confidences)
    multiscale_score = _multi_scale_score(
        entries=decayed_entries,
        now=now,
        scales_s=config.multi_scale_windows_s,
    )
    coverage_score = _coverage_score(
        entries=decayed_entries,
        now=now,
        horizon_s=config.fusion_horizon_s,
        bucket_s=config.evidence_bucket_s,
    )
    support_density = _support_density_score(
        entry_count=len(decayed_entries),
        expected_modalities=expected_modalities,
    )
    modality_balance = _modality_balance(
        audio_support=audio_support,
        vision_support=vision_support,
        expected_modalities=expected_modalities,
    )
    modality_agreement = _modality_agreement(
        entries=decayed_entries,
        expected_modalities=expected_modalities,
    )
    confidence = (
        0.24 * base_score
        + 0.18 * multiscale_score
        + 0.10 * coverage_score
        + 0.12 * modality_balance
        + 0.12 * support_density
        + 0.12 * modality_agreement
        + 0.12 * _clamp01(structural_score)
    )
    if expected_modalities > 1:
        confidence *= 0.85 + 0.15 * min(modality_agreement, _clamp01(structural_score))
    return round(_clamp01(confidence), 4)


def _decayed_entries(
    *,
    now: float,
    audio_support: tuple[AudioMicroEvent, ...],
    vision_support: tuple[VisionSequenceEvent, ...],
    config: MultimodalEventFusionConfig,
) -> tuple[_EvidenceEntry, ...]:
    """Return timestamped decayed confidences for current evidence support."""

    entries: list[_EvidenceEntry] = []
    for event in audio_support:
        if not _is_valid_timestamp(event.observed_at):
            continue
        age_s = max(0.0, now - event.observed_at)
        decayed_confidence = _safe_confidence(event.confidence) * _exp_half_life(
            age_s,
            half_life_s=config.temporal_decay_half_life_audio_s,
        )
        if decayed_confidence > 0.0:
            entries.append(
                _EvidenceEntry(
                    timestamp=event.observed_at,
                    confidence=decayed_confidence,
                    modality="audio",
                )
            )
    for event in vision_support:
        anchor_time = event.window_end_s
        if not _is_valid_timestamp(anchor_time):
            continue
        age_s = max(0.0, now - anchor_time)
        decayed_confidence = _safe_confidence(event.confidence) * _exp_half_life(
            age_s,
            half_life_s=config.temporal_decay_half_life_vision_s,
        )
        if decayed_confidence > 0.0:
            entries.append(
                _EvidenceEntry(
                    timestamp=anchor_time,
                    confidence=decayed_confidence,
                    modality="vision",
                )
            )
    return tuple(entries)


def _compress_decayed_entries(
    *,
    entries: tuple[_EvidenceEntry, ...],
    now: float,
    bucket_s: float,
    max_entries_per_modality: int,
) -> tuple[_EvidenceEntry, ...]:
    """Collapse duplicates into per-bucket maxima and cap retained evidence per modality."""

    if not entries:
        return ()

    by_bucket: dict[tuple[str, int], _EvidenceEntry] = {}
    for entry in entries:
        age_s = max(0.0, now - entry.timestamp)
        bucket_id = int(math.floor(age_s / max(bucket_s, _FLOAT_EPSILON)))
        key = (entry.modality, bucket_id)
        existing = by_bucket.get(key)
        if existing is None or entry.confidence > existing.confidence or (
            math.isclose(entry.confidence, existing.confidence) and entry.timestamp > existing.timestamp
        ):
            by_bucket[key] = entry

    by_modality: dict[str, list[_EvidenceEntry]] = defaultdict(list)
    for entry in by_bucket.values():
        by_modality[entry.modality].append(entry)

    limited: list[_EvidenceEntry] = []
    for modality_entries in by_modality.values():
        modality_entries.sort(key=lambda entry: entry.timestamp, reverse=True)
        limited.extend(modality_entries[:max_entries_per_modality])

    limited.sort(key=lambda entry: entry.timestamp)
    return tuple(limited)


def _multi_scale_score(
    *,
    entries: tuple[_EvidenceEntry, ...],
    now: float,
    scales_s: tuple[float, ...],
) -> float:
    """Return a compact multi-scale recency score from the current evidence."""

    scores: list[float] = []
    for scale_s in scales_s:
        scale_entries = [entry.confidence for entry in entries if now - entry.timestamp <= scale_s]
        if scale_entries:
            scores.append(max(scale_entries))
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def _coverage_score(
    *,
    entries: tuple[_EvidenceEntry, ...],
    now: float,
    horizon_s: float,
    bucket_s: float,
) -> float:
    """Return a coarse coverage score over the fusion horizon."""

    if not entries or horizon_s <= 0.0 or bucket_s <= 0.0:
        return 0.0
    bucket_ids = {
        int(math.floor(max(0.0, now - entry.timestamp) / bucket_s))
        for entry in entries
        if 0.0 <= now - entry.timestamp <= horizon_s + _FLOAT_EPSILON
    }
    max_bucket_count = max(1, int(math.ceil(horizon_s / bucket_s)))
    return min(1.0, len(bucket_ids) / max_bucket_count)


def _modality_balance(
    *,
    audio_support: tuple[AudioMicroEvent, ...],
    vision_support: tuple[VisionSequenceEvent, ...],
    expected_modalities: int,
) -> float:
    """Return how well the current support covers the expected modalities."""

    present = 0
    if audio_support:
        present += 1
    if vision_support:
        present += 1
    return min(1.0, present / max(1, expected_modalities))


def _modality_agreement(
    *,
    entries: tuple[_EvidenceEntry, ...],
    expected_modalities: int,
) -> float:
    """Return whether present modalities agree instead of one dominating the score."""

    if expected_modalities <= 1:
        return 1.0
    by_modality: dict[str, list[float]] = defaultdict(list)
    for entry in entries:
        by_modality[entry.modality].append(entry.confidence)
    if len(by_modality) < expected_modalities:
        return 0.0
    modality_scores = [sum(scores) / len(scores) for scores in by_modality.values() if scores]
    if len(modality_scores) < expected_modalities:
        return 0.0
    return _clamp01(1.0 - (max(modality_scores) - min(modality_scores)))


def _support_density_score(
    *,
    entry_count: int,
    expected_modalities: int,
) -> float:
    """Return a compact density score for repeated temporal evidence."""

    target_count = max(3, expected_modalities * 2)
    return min(1.0, max(0, entry_count) / target_count)


def _window_start(
    *,
    audio_support: tuple[AudioMicroEvent, ...],
    vision_support: tuple[VisionSequenceEvent, ...],
) -> float | None:
    """Return the earliest evidence start across modalities."""

    starts: list[float] = [event.observed_at for event in audio_support if _is_valid_timestamp(event.observed_at)]
    starts.extend(event.window_start_s for event in vision_support if _is_valid_timestamp(event.window_start_s))
    return min(starts) if starts else None


def _window_end(
    *,
    audio_support: tuple[AudioMicroEvent, ...],
    vision_support: tuple[VisionSequenceEvent, ...],
) -> float | None:
    """Return the latest evidence end across modalities."""

    ends: list[float] = [event.observed_at for event in audio_support if _is_valid_timestamp(event.observed_at)]
    ends.extend(event.window_end_s for event in vision_support if _is_valid_timestamp(event.window_end_s))
    return max(ends) if ends else None


def _dedupe_audio_support(
    support: tuple[AudioMicroEvent, ...],
    *,
    quantum_s: float,
) -> tuple[AudioMicroEvent, ...]:
    """Collapse near-identical audio events and keep the strongest instance."""

    if not support:
        return ()
    best_by_key: dict[tuple[object, ...], AudioMicroEvent] = {}
    for event in support:
        if not _is_valid_timestamp(event.observed_at):
            continue
        key = _audio_event_fingerprint(event, quantum_s=quantum_s)
        current = best_by_key.get(key)
        if current is None or _safe_confidence(event.confidence) > _safe_confidence(current.confidence):
            best_by_key[key] = event
    return tuple(sorted(best_by_key.values(), key=lambda event: event.observed_at))


def _dedupe_vision_support(
    support: tuple[VisionSequenceEvent, ...],
    *,
    quantum_s: float,
) -> tuple[VisionSequenceEvent, ...]:
    """Collapse near-identical vision sequences and keep the strongest instance."""

    if not support:
        return ()
    best_by_key: dict[tuple[object, ...], VisionSequenceEvent] = {}
    for event in support:
        if not _is_valid_timestamp(event.window_start_s) or not _is_valid_timestamp(event.window_end_s):
            continue
        key = _vision_sequence_fingerprint(event, quantum_s=quantum_s)
        current = best_by_key.get(key)
        if current is None or _safe_confidence(event.confidence) > _safe_confidence(current.confidence):
            best_by_key[key] = event
    return tuple(sorted(best_by_key.values(), key=lambda event: (event.window_start_s, event.window_end_s)))


def _audio_event_fingerprint(event: AudioMicroEvent, *, quantum_s: float) -> tuple[object, ...]:
    """Build one stable fingerprint for deduping audio micro-events."""

    return (
        event.kind,
        _quantize_time(event.observed_at, quantum_s=quantum_s),
        bool(event.active),
    )


def _vision_sequence_fingerprint(event: VisionSequenceEvent, *, quantum_s: float) -> tuple[object, ...]:
    """Build one stable fingerprint for deduping vision sequences."""

    return (
        event.kind,
        _quantize_time(event.window_start_s, quantum_s=quantum_s),
        _quantize_time(event.window_end_s, quantum_s=quantum_s),
        bool(event.active),
    )


def _quantize_time(timestamp_s: float, *, quantum_s: float) -> int:
    """Quantize one timestamp for duplicate suppression and stable comparisons."""

    return int(round(timestamp_s / max(quantum_s, _FLOAT_EPSILON)))


def _distance_point_to_interval(*, point_s: float, start_s: float, end_s: float) -> float:
    """Return the absolute distance from one timestamp to one closed interval."""

    if point_s < start_s:
        return start_s - point_s
    if point_s > end_s:
        return point_s - end_s
    return 0.0


def _claim_sort_key(claim: FusedEventClaim) -> tuple[float, float, float]:
    """Sort claims deterministically by confidence, recency, and span."""

    window_end_s = getattr(claim, "window_end_s", None)
    window_start_s = getattr(claim, "window_start_s", None)
    confidence = getattr(claim, "confidence", 0.0)
    span_s = 0.0
    if isinstance(window_start_s, (int, float)) and isinstance(window_end_s, (int, float)):
        span_s = max(0.0, window_end_s - window_start_s)
    return (
        float(confidence),
        float(window_end_s) if isinstance(window_end_s, (int, float)) else -1.0,
        span_s,
    )


def _suppress_redundant_claims(claims: list[FusedEventClaim]) -> list[FusedEventClaim]:
    """Suppress obviously redundant generic claims when a stronger specific claim is present."""

    if len(claims) < 2:
        return claims

    best_by_state: dict[str, FusedEventClaim] = {}
    for claim in claims:
        state = getattr(claim, "state", "")
        current = best_by_state.get(state)
        if current is None or _claim_sort_key(claim) > _claim_sort_key(current):
            best_by_state[state] = claim

    specific = best_by_state.get("cry_like_distress_possible")
    generic = best_by_state.get("distress_possible")
    if specific is not None and generic is not None and _claims_overlap(specific, generic):
        best_by_state.pop("distress_possible", None)

    deduped = list(best_by_state.values())
    deduped.sort(key=_claim_sort_key, reverse=True)
    return deduped


def _claims_overlap(left: FusedEventClaim, right: FusedEventClaim) -> bool:
    """Return whether two claims cover materially the same time window."""

    left_start = getattr(left, "window_start_s", None)
    left_end = getattr(left, "window_end_s", None)
    right_start = getattr(right, "window_start_s", None)
    right_end = getattr(right, "window_end_s", None)
    if not all(isinstance(value, (int, float)) for value in (left_start, left_end, right_start, right_end)):
        return False
    overlap_start = max(float(left_start), float(right_start))
    overlap_end = min(float(left_end), float(right_end))
    return overlap_end + _FLOAT_EPSILON >= overlap_start


def _is_valid_audio_event(event: AudioMicroEvent, *, now: float) -> bool:
    """Return whether one audio event is safe to fuse."""

    return (
        bool(event.active)
        and _is_valid_timestamp(event.observed_at)
        and event.observed_at <= now + _FLOAT_EPSILON
    )


def _is_valid_vision_sequence(event: VisionSequenceEvent, *, now: float) -> bool:
    """Return whether one vision sequence is safe to fuse."""

    return (
        bool(event.active)
        and _is_valid_timestamp(event.window_start_s)
        and _is_valid_timestamp(event.window_end_s)
        and event.window_start_s <= event.window_end_s + _FLOAT_EPSILON
        and event.window_end_s <= now + _FLOAT_EPSILON
    )


def _is_valid_timestamp(value: float) -> bool:
    """Return whether one timestamp is finite and non-negative."""

    return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0.0


def _safe_confidence(value: float) -> float:
    """Return one normalized confidence in [0, 1]."""

    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return 0.0
    return _clamp01(float(value))


def _clamp01(value: float) -> float:
    """Clamp one floating point value into [0, 1]."""

    return min(1.0, max(0.0, value))


def _require_positive_finite(name: str, value: float) -> None:
    """Validate one strictly positive finite config value."""

    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite float")


def _require_non_negative_finite(name: str, value: float) -> None:
    """Validate one non-negative finite config value."""

    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a non-negative finite float")


def _exp_half_life(age_s: float, *, half_life_s: float) -> float:
    """Return one exponential decay weight based on evidence age."""

    if half_life_s <= 0.0:
        return 0.0
    return math.exp(-math.log(2.0) * max(0.0, age_s) / half_life_s)


__all__ = [
    "MultimodalEventFusionConfig",
    "MultimodalEventFusionTracker",
    "derive_fused_event_claims",
]