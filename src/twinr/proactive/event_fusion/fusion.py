"""Fuse short-window audio and vision evidence into conservative event claims."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

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


@dataclass(frozen=True, slots=True)
class MultimodalEventFusionConfig:
    """Store horizons and thresholds for the V1 fusion tracker."""

    audio_horizon_s: float = 8.0
    audio_event_horizon_s: float = 8.0
    vision_horizon_s: float = 8.0
    vision_sequence_horizon_s: float = 8.0
    fusion_horizon_s: float = 8.0
    temporal_decay_half_life_audio_s: float = 2.0
    temporal_decay_half_life_vision_s: float = 3.5
    evidence_bucket_s: float = 1.0
    multi_scale_windows_s: tuple[float, ...] = (2.0, 4.0, 8.0)
    audio_events: AudioEventConfig = field(default_factory=AudioEventConfig)
    vision_sequences: VisionSequenceConfig = field(default_factory=VisionSequenceConfig)
    keyframe_review: KeyframeReviewConfig = field(default_factory=KeyframeReviewConfig)


class MultimodalEventFusionTracker:
    """Track rolling multimodal history and emit conservative fused claims."""

    def __init__(self, config: MultimodalEventFusionConfig | None = None) -> None:
        """Initialize one compact in-memory event-fusion tracker."""

        self.config = config or MultimodalEventFusionConfig()
        self.audio_observations = RollingWindowBuffer[SocialAudioObservation](horizon_s=self.config.audio_horizon_s)
        self.vision_observations = RollingWindowBuffer[SocialVisionObservation](horizon_s=self.config.vision_horizon_s)
        self.audio_events = RollingWindowBuffer[AudioMicroEvent](horizon_s=self.config.audio_event_horizon_s)
        self.vision_sequences = RollingWindowBuffer[VisionSequenceEvent](horizon_s=self.config.vision_sequence_horizon_s)

    def observe(
        self,
        observation: SocialObservation,
        *,
        audio_hints: AudioClassifierHints | None = None,
        room_busy_or_overlapping: bool = False,
    ) -> tuple[FusedEventClaim, ...]:
        """Ingest one observation tick and return current active fused claims."""

        now = observation.observed_at
        self.audio_observations.append(now, observation.audio)
        audio_events = derive_audio_micro_events(
            observed_at=now,
            observation=observation.audio,
            hints=audio_hints,
            config=self.config.audio_events,
        )
        for event in audio_events:
            self.audio_events.append(event.observed_at, event)

        if observation.inspected:
            self.vision_observations.append(now, observation.vision)
            vision_sequences = derive_vision_sequences(
                observation_buffer=self.vision_observations,
                now=now,
                config=self.config.vision_sequences,
            )
            for sequence in vision_sequences:
                self.vision_sequences.append(sequence.window_end_s, sequence)

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

    claims: list[FusedEventClaim] = []
    shout_audio = _matching_audio_events(
        audio_event_buffer,
        (AudioEventKind.SHOUT_LIKE_AUDIO,),
        now=now,
        horizon_s=config.fusion_horizon_s,
    )
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
            )
        )

    laugh_audio = _matching_audio_events(
        audio_event_buffer,
        (AudioEventKind.LAUGH_LIKE_AUDIO,),
        now=now,
        horizon_s=config.fusion_horizon_s,
    )
    positive_contact = _matching_vision_sequences(
        vision_sequence_buffer,
        (VisionSequenceKind.POSITIVE_CONTACT,),
        now=now,
        horizon_s=config.fusion_horizon_s,
    )
    if laugh_audio and positive_contact:
        claims.append(
            _build_claim_from_support(
                state="laugh_like_positive_contact",
                preferred_action_level=FusionActionLevel.PROMPT_ONLY,
                audio_support=laugh_audio,
                vision_support=positive_contact,
                expected_modalities=2,
                policy_context=policy_context,
                now=now,
                config=config,
                vision_observation_buffer=vision_observation_buffer,
            )
        )

    cry_audio = _matching_audio_events(
        audio_event_buffer,
        (AudioEventKind.CRY_LIKE_AUDIO,),
        now=now,
        horizon_s=config.fusion_horizon_s,
    )
    slumped = _matching_vision_sequences(
        vision_sequence_buffer,
        (VisionSequenceKind.SLUMPED_QUIET,),
        now=now,
        horizon_s=config.fusion_horizon_s,
    )
    distress_audio = tuple(dict.fromkeys(shout_audio + cry_audio))
    if distress_audio and slumped:
        claims.append(
            _build_claim_from_support(
                state="distress_possible",
                preferred_action_level=FusionActionLevel.REVIEW_ONLY,
                audio_support=distress_audio,
                vision_support=slumped,
                expected_modalities=2,
                policy_context=policy_context,
                now=now,
                config=config,
                vision_observation_buffer=vision_observation_buffer,
                review_preferred=True,
            )
        )
    if cry_audio and slumped:
        claims.append(
            _build_claim_from_support(
                state="cry_like_distress_possible",
                preferred_action_level=FusionActionLevel.REVIEW_ONLY,
                audio_support=cry_audio,
                vision_support=slumped,
                expected_modalities=2,
                policy_context=policy_context,
                now=now,
                config=config,
                vision_observation_buffer=vision_observation_buffer,
                review_preferred=True,
            )
        )

    downward = _matching_vision_sequences(
        vision_sequence_buffer,
        (VisionSequenceKind.DOWNWARD_TRANSITION,),
        now=now,
        horizon_s=config.fusion_horizon_s,
    )
    floor_entered = _matching_vision_sequences(
        vision_sequence_buffer,
        (VisionSequenceKind.FLOOR_POSE_ENTERED,),
        now=now,
        horizon_s=config.fusion_horizon_s,
    )
    if downward and floor_entered:
        claims.append(
            _build_claim_from_support(
                state="possible_fall",
                preferred_action_level=FusionActionLevel.REVIEW_ONLY,
                audio_support=(),
                vision_support=downward + floor_entered,
                expected_modalities=1,
                policy_context=policy_context,
                now=now,
                config=config,
                vision_observation_buffer=vision_observation_buffer,
                review_preferred=True,
            )
        )

    floor_stillness = _matching_vision_sequences(
        vision_sequence_buffer,
        (VisionSequenceKind.FLOOR_STILLNESS,),
        now=now,
        horizon_s=config.fusion_horizon_s,
    )
    if floor_stillness and (downward or floor_entered):
        prior = downward if downward else floor_entered
        claims.append(
            _build_claim_from_support(
                state="floor_stillness_after_drop",
                preferred_action_level=FusionActionLevel.PROMPT_ONLY,
                audio_support=(),
                vision_support=floor_stillness + prior,
                expected_modalities=1,
                policy_context=policy_context,
                now=now,
                config=config,
                vision_observation_buffer=vision_observation_buffer,
                review_preferred=True,
            )
        )

    return tuple(claims)


def _matching_audio_events(
    buffer: RollingWindowBuffer[AudioMicroEvent],
    kinds: tuple[AudioEventKind, ...],
    now: float,
    horizon_s: float,
) -> tuple[AudioMicroEvent, ...]:
    """Return recent active audio events for the requested kinds."""

    recent = buffer.between(max(0.0, now - horizon_s), now)
    return tuple(
        sample.value
        for sample in recent
        if sample.value.kind in kinds and sample.value.active
    )


def _matching_vision_sequences(
    buffer: RollingWindowBuffer[VisionSequenceEvent],
    kinds: tuple[VisionSequenceKind, ...],
    now: float,
    horizon_s: float,
) -> tuple[VisionSequenceEvent, ...]:
    """Return recent active vision sequences for the requested kinds."""

    recent = buffer.between(max(0.0, now - horizon_s), now)
    return tuple(
        sample.value
        for sample in recent
        if sample.value.kind in kinds and sample.value.active
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
) -> FusedEventClaim:
    """Build one claim from all matching evidence inside the current window."""

    confidence = _aggregate_temporal_confidence(
        now=now,
        audio_support=audio_support,
        vision_support=vision_support,
        expected_modalities=expected_modalities,
        config=config,
    )
    window_start_s = _window_start(audio_support=audio_support, vision_support=vision_support)
    window_end_s = _window_end(audio_support=audio_support, vision_support=vision_support)
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
    config: MultimodalEventFusionConfig,
) -> float:
    """Aggregate recent support using multi-scale and time-decayed evidence."""

    decayed_entries = _decayed_entries(
        now=now,
        audio_support=audio_support,
        vision_support=vision_support,
        config=config,
    )
    if not decayed_entries:
        return 0.0
    decayed_confidences = tuple(entry[1] for entry in decayed_entries)
    base_score = sum(decayed_confidences) / len(decayed_confidences)
    multiscale_score = _multi_scale_score(
        entries=decayed_entries,
        now=now,
        scales_s=config.multi_scale_windows_s,
    )
    coverage_score = _coverage_score(
        entries=decayed_entries,
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
    confidence = (
        0.35 * base_score
        + 0.25 * multiscale_score
        + 0.15 * coverage_score
        + 0.10 * modality_balance
        + 0.15 * support_density
    )
    return round(min(1.0, max(0.0, confidence)), 4)


def _decayed_entries(
    *,
    now: float,
    audio_support: tuple[AudioMicroEvent, ...],
    vision_support: tuple[VisionSequenceEvent, ...],
    config: MultimodalEventFusionConfig,
) -> tuple[tuple[float, float, str], ...]:
    """Return timestamped decayed confidences for current evidence support."""

    entries: list[tuple[float, float, str]] = []
    for event in audio_support:
        age_s = max(0.0, now - event.observed_at)
        entries.append(
            (
                event.observed_at,
                event.confidence * _exp_half_life(age_s, half_life_s=config.temporal_decay_half_life_audio_s),
                "audio",
            )
        )
    for event in vision_support:
        anchor_time = event.window_end_s
        age_s = max(0.0, now - anchor_time)
        entries.append(
            (
                anchor_time,
                event.confidence * _exp_half_life(age_s, half_life_s=config.temporal_decay_half_life_vision_s),
                "vision",
            )
        )
    return tuple(entries)


def _multi_scale_score(
    *,
    entries: tuple[tuple[float, float, str], ...],
    now: float,
    scales_s: tuple[float, ...],
) -> float:
    """Return a compact multi-scale recency score from the current evidence."""

    scores: list[float] = []
    for scale_s in scales_s:
        scale_entries = [confidence for timestamp, confidence, _ in entries if now - timestamp <= scale_s]
        if scale_entries:
            scores.append(max(scale_entries))
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def _coverage_score(
    *,
    entries: tuple[tuple[float, float, str], ...],
    horizon_s: float,
    bucket_s: float,
) -> float:
    """Return a coarse coverage score over the fusion horizon."""

    if not entries or horizon_s <= 0.0 or bucket_s <= 0.0:
        return 0.0
    bucket_ids = {
        int(math.floor(timestamp / bucket_s))
        for timestamp, _, _ in entries
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


def _support_density_score(
    *,
    entry_count: int,
    expected_modalities: int,
) -> float:
    """Return a compact density score for repeated temporal evidence."""

    target_count = max(3, expected_modalities + 1)
    return min(1.0, max(0, entry_count) / target_count)


def _window_start(
    *,
    audio_support: tuple[AudioMicroEvent, ...],
    vision_support: tuple[VisionSequenceEvent, ...],
) -> float | None:
    """Return the earliest evidence start across modalities."""

    starts: list[float] = [event.observed_at for event in audio_support]
    starts.extend(event.window_start_s for event in vision_support)
    return min(starts) if starts else None


def _window_end(
    *,
    audio_support: tuple[AudioMicroEvent, ...],
    vision_support: tuple[VisionSequenceEvent, ...],
) -> float | None:
    """Return the latest evidence end across modalities."""

    ends: list[float] = [event.observed_at for event in audio_support]
    ends.extend(event.window_end_s for event in vision_support)
    return max(ends) if ends else None


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
