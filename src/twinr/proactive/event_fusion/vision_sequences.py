"""Derive bounded temporal vision sequences for event fusion.

The social camera layer already normalizes single ticks. This module turns the
recent rolling window of those ticks into short sequence-level facts such as
``downward_transition`` or ``slumped_quiet``.

The 2026 upgrade focuses on deployment realism for edge perception:
- contiguous latest-supporting evidence instead of sparse window matches
- stale / malformed timestamp rejection
- adaptive gap handling for variable frame cadence
- uncertainty-aware deferral when motion evidence is mostly unknown
"""

from __future__ import annotations

# CHANGELOG: 2026-03-29
# BUG-1: Bounded repeated FLOOR_POSE_ENTERED and DOWNWARD_TRANSITION emissions to a short onset window.
# BUG-2: Fixed false hold events caused by sparse non-contiguous samples across the rolling window.
# BUG-3: Fixed false FLOOR_STILLNESS / SLUMPED_QUIET when motion was unknown rather than explicitly still.
# BUG-4: Fixed config bug where hold durations greater than transition_window_s could never be satisfied.
# SEC-1: Hardened against malformed, future-skewed, and stale timestamps that can spoof or suppress alerts.
# IMP-1: Added contiguous-tail evidence extraction with adaptive gap limits for robust streaming behavior on edge devices.
# IMP-2: Added uncertainty-aware deferral and evidence-calibrated confidence scoring for safer downstream fusion.

import math
from dataclasses import dataclass, field
from enum import StrEnum
from statistics import median
from typing import Callable

from twinr.proactive.event_fusion.buffers import RollingWindowBuffer, TimedSample
from twinr.proactive.social.engine import SocialBodyPose, SocialMotionState, SocialVisionObservation


_FLOOR_LIKE_POSES = {SocialBodyPose.FLOOR.value, SocialBodyPose.LYING_LOW.value}
_UPRIGHT_LIKE_POSES = {
    SocialBodyPose.UPRIGHT.value,
    SocialBodyPose.SEATED.value,
    SocialBodyPose.SLUMPED.value,
}
_NON_MOVING_MOTIONS = {SocialMotionState.STILL.value, SocialMotionState.UNKNOWN.value}


class VisionSequenceKind(StrEnum):
    """Describe the V1 sequence vocabulary derived from recent vision history."""

    DOWNWARD_TRANSITION = "downward_transition"
    FLOOR_POSE_ENTERED = "floor_pose_entered"
    FLOOR_STILLNESS = "floor_stillness"
    SLUMPED_QUIET = "slumped_quiet"
    POSITIVE_CONTACT = "positive_contact"
    SHOWING_INTENT = "showing_intent"


@dataclass(frozen=True, slots=True)
class VisionSequenceConfig:
    """Store bounded temporal thresholds for V1 vision sequences."""

    transition_window_s: float = 6.0
    floor_stillness_hold_s: float = 3.0
    slumped_hold_s: float = 3.0
    positive_contact_hold_s: float = 1.5
    showing_intent_hold_s: float = 1.0
    max_staleness_s: float = 2.5
    max_future_skew_s: float = 0.25
    max_sample_gap_s: float = 1.0
    max_adaptive_gap_s: float = 2.0
    adaptive_gap_multiplier: float = 3.0
    edge_emit_window_s: float = 1.25
    min_explicit_still_samples: int = 2
    max_unknown_motion_ratio: float = 0.5


@dataclass(frozen=True, slots=True)
class VisionSequenceEvent:
    """Describe one short-lived temporal vision sequence."""

    kind: VisionSequenceKind
    window_start_s: float
    window_end_s: float
    confidence: float
    source: str
    active: bool = True
    supporting_fields: tuple[str, ...] = field(default_factory=tuple)

    def to_payload(self) -> dict[str, object]:
        """Serialize one vision sequence into plain facts."""

        return {
            "kind": self.kind.value,
            "window_start_s": self.window_start_s,
            "window_end_s": self.window_end_s,
            "confidence": self.confidence,
            "source": self.source,
            "active": self.active,
            "supporting_fields": list(self.supporting_fields),
        }


def derive_vision_sequences(
    *,
    observation_buffer: RollingWindowBuffer[SocialVisionObservation],
    now: float,
    config: VisionSequenceConfig | None = None,
) -> tuple[VisionSequenceEvent, ...]:
    """Return active V1 vision sequences from the current rolling window."""

    cfg = config or VisionSequenceConfig()
    recent_samples = _prepare_recent_samples(observation_buffer, now, cfg)
    if not recent_samples:
        return ()
    gap_limit_s = _gap_limit_s(recent_samples, cfg)
    latest = recent_samples[-1]
    # BREAKING: stale or badly skewed observations now suppress active sequences
    # instead of extrapolating from old data.
    if _is_stale(latest.observed_at, now, cfg, gap_limit_s):
        return ()

    sequences: list[VisionSequenceEvent] = []

    # BREAKING: entry/transition events now emit only near onset
    # (edge_emit_window_s) rather than on every tick for the whole window.
    floor_event = _derive_floor_pose_entered(recent_samples, latest, cfg, gap_limit_s)
    if floor_event is not None:
        sequences.append(floor_event)

    downward_event = _derive_downward_transition(recent_samples, latest, cfg, gap_limit_s)
    if downward_event is not None:
        sequences.append(downward_event)

    # BREAKING: hold events now require contiguous latest-supporting evidence
    # and explicit still/contact support instead of sparse matches anywhere
    # inside the rolling window.
    floor_stillness = _derive_floor_stillness(recent_samples, latest, cfg, gap_limit_s)
    if floor_stillness is not None:
        sequences.append(floor_stillness)

    slumped_quiet = _derive_slumped_quiet(recent_samples, latest, cfg, gap_limit_s)
    if slumped_quiet is not None:
        sequences.append(slumped_quiet)

    positive_contact = _derive_positive_contact(recent_samples, latest, cfg, gap_limit_s)
    if positive_contact is not None:
        sequences.append(positive_contact)

    showing_intent = _derive_showing_intent(recent_samples, latest, cfg, gap_limit_s)
    if showing_intent is not None:
        sequences.append(showing_intent)

    return tuple(sequences)


def _pose_value(pose: SocialBodyPose | str | None) -> str:
    """Return the normalized string value for one body pose enum."""

    if pose is None:
        return SocialBodyPose.UNKNOWN.value
    return pose.value if isinstance(pose, SocialBodyPose) else str(pose)


def _motion_value(motion: SocialMotionState | str | None) -> str:
    """Return the normalized string value for one motion-state enum."""

    if motion is None:
        return SocialMotionState.UNKNOWN.value
    return motion.value if isinstance(motion, SocialMotionState) else str(motion)


def _is_finite_number(value: object) -> bool:
    """Return whether one object is a finite float-like number."""

    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _max_required_window_s(config: VisionSequenceConfig) -> float:
    """Return the largest lookback window required by the active configuration."""

    return max(
        config.transition_window_s,
        config.floor_stillness_hold_s,
        config.slumped_hold_s,
        config.positive_contact_hold_s,
        config.showing_intent_hold_s,
        config.max_staleness_s,
        config.edge_emit_window_s,
    )


def _prepare_recent_samples(
    observation_buffer: RollingWindowBuffer[SocialVisionObservation],
    now: float,
    config: VisionSequenceConfig,
) -> tuple[TimedSample[SocialVisionObservation], ...]:
    """Return sorted, finite, time-valid samples from the required lookback window."""

    if not _is_finite_number(now):
        return ()
    window_s = _max_required_window_s(config)
    raw_samples = observation_buffer.between(
        max(0.0, float(now) - window_s),
        float(now) + config.max_future_skew_s,
    )
    valid_samples = [
        sample
        for sample in raw_samples
        if _is_finite_number(sample.observed_at)
        and float(sample.observed_at) <= float(now) + config.max_future_skew_s
    ]
    if not valid_samples:
        return ()
    valid_samples.sort(key=lambda sample: float(sample.observed_at))
    normalized: list[TimedSample[SocialVisionObservation]] = []
    last_observed_at = -math.inf
    for sample in valid_samples:
        observed_at = float(sample.observed_at)
        if observed_at < last_observed_at:
            continue
        normalized.append(sample)
        last_observed_at = observed_at
    return tuple(normalized)


def _gap_limit_s(
    recent_samples: tuple[TimedSample[SocialVisionObservation], ...],
    config: VisionSequenceConfig,
) -> float:
    """Estimate a robust continuity gap from recent cadence, bounded by config."""

    if len(recent_samples) < 2:
        return config.max_sample_gap_s
    gaps = [
        float(curr.observed_at) - float(prev.observed_at)
        for prev, curr in zip(recent_samples, recent_samples[1:])
        if 0.0 < float(curr.observed_at) - float(prev.observed_at) <= config.transition_window_s
    ]
    if not gaps:
        return config.max_sample_gap_s
    cadence_gap = median(gaps) * config.adaptive_gap_multiplier
    return max(
        config.max_sample_gap_s,
        min(config.max_adaptive_gap_s, float(cadence_gap)),
    )


def _is_stale(observed_at: float, now: float, config: VisionSequenceConfig, gap_limit_s: float) -> bool:
    """Return whether the latest observation is too old to support an active sequence."""

    return float(now) - float(observed_at) > max(config.max_staleness_s, gap_limit_s * 2.0)


def _tail_run(
    recent_samples: tuple[TimedSample[SocialVisionObservation], ...],
    *,
    predicate: Callable[[TimedSample[SocialVisionObservation]], bool],
    gap_limit_s: float,
) -> tuple[TimedSample[SocialVisionObservation], ...]:
    """Return the latest contiguous run that satisfies predicate."""

    if not recent_samples or not predicate(recent_samples[-1]):
        return ()
    run: list[TimedSample[SocialVisionObservation]] = [recent_samples[-1]]
    current = recent_samples[-1]
    for sample in reversed(recent_samples[:-1]):
        if float(current.observed_at) - float(sample.observed_at) > gap_limit_s:
            break
        if not predicate(sample):
            break
        run.append(sample)
        current = sample
    run.reverse()
    return tuple(run)


def _sample_before_run(
    recent_samples: tuple[TimedSample[SocialVisionObservation], ...],
    run: tuple[TimedSample[SocialVisionObservation], ...],
    *,
    gap_limit_s: float,
) -> TimedSample[SocialVisionObservation] | None:
    """Return the sample immediately preceding a contiguous run when it is time-close."""

    if not run or len(run) >= len(recent_samples):
        return None
    idx = len(recent_samples) - len(run) - 1
    candidate = recent_samples[idx]
    if float(run[0].observed_at) - float(candidate.observed_at) > gap_limit_s:
        return None
    return candidate


def _run_duration_s(run: tuple[TimedSample[SocialVisionObservation], ...]) -> float:
    """Return the duration covered by one contiguous run."""

    if len(run) < 2:
        return 0.0
    return max(0.0, float(run[-1].observed_at) - float(run[0].observed_at))


def _clamp(value: float, lower: float, upper: float) -> float:
    """Clamp one float into a closed interval."""

    return max(lower, min(upper, value))


def _score_hold_confidence(
    *,
    run: tuple[TimedSample[SocialVisionObservation], ...],
    required_hold_s: float,
    explicit_support_ratio: float = 1.0,
    unknown_ratio: float = 0.0,
) -> float:
    """Return a bounded confidence score for one sustained sequence."""

    duration_ratio = 1.0 if required_hold_s <= 0.0 else _clamp(_run_duration_s(run) / required_hold_s, 0.0, 1.5)
    sample_ratio = _clamp(len(run) / 4.0, 0.0, 1.0)
    score = 0.50 + 0.18 * min(duration_ratio, 1.0) + 0.12 * sample_ratio + 0.12 * explicit_support_ratio - 0.12 * unknown_ratio
    return _clamp(score, 0.0, 0.98)


def _score_transition_confidence(
    *,
    prior_sample: TimedSample[SocialVisionObservation],
    latest: TimedSample[SocialVisionObservation],
    onset_age_s: float,
    transition_window_s: float,
    gap_limit_s: float,
) -> float:
    """Return a bounded confidence score for one state transition sequence."""

    transition_age_ratio = 1.0 - _clamp(
        (float(latest.observed_at) - float(prior_sample.observed_at)) / max(transition_window_s, 1e-6),
        0.0,
        1.0,
    )
    onset_fresh_ratio = 1.0 - _clamp(onset_age_s / max(gap_limit_s, 1e-6), 0.0, 1.0)
    score = 0.64 + 0.14 * transition_age_ratio + 0.10 * onset_fresh_ratio
    return _clamp(score, 0.0, 0.96)


def _is_floor_like(sample: TimedSample[SocialVisionObservation]) -> bool:
    """Return whether a sample indicates a floor-like pose."""

    return _pose_value(sample.value.body_pose) in _FLOOR_LIKE_POSES


def _is_floor_still_like(sample: TimedSample[SocialVisionObservation]) -> bool:
    """Return whether a sample indicates floor pose without explicit motion."""

    return _is_floor_like(sample) and _motion_value(sample.value.motion_state) in _NON_MOVING_MOTIONS


def _is_slumped_still_like(sample: TimedSample[SocialVisionObservation]) -> bool:
    """Return whether a sample indicates slumped posture without explicit motion."""

    return (
        _pose_value(sample.value.body_pose) == SocialBodyPose.SLUMPED.value
        and _motion_value(sample.value.motion_state) in _NON_MOVING_MOTIONS
    )


def _is_positive_contact_sample(sample: TimedSample[SocialVisionObservation]) -> bool:
    """Return whether one sample supports positive social contact."""

    return sample.value.smiling is True and (
        sample.value.looking_toward_device is True or sample.value.engaged_with_device is True
    )


def _is_showing_intent_sample(sample: TimedSample[SocialVisionObservation]) -> bool:
    """Return whether one sample supports showing intent."""

    return sample.value.showing_intent_likely is True or (
        sample.value.hand_or_object_near_camera is True
        and sample.value.looking_toward_device is True
    )


def _motion_support_stats(
    run: tuple[TimedSample[SocialVisionObservation], ...],
) -> tuple[int, int, int]:
    """Return counts of explicit-still, unknown, and total motion samples."""

    explicit_still = 0
    unknown = 0
    for sample in run:
        motion = _motion_value(sample.value.motion_state)
        if motion == SocialMotionState.STILL.value:
            explicit_still += 1
        elif motion == SocialMotionState.UNKNOWN.value:
            unknown += 1
    return explicit_still, unknown, len(run)


def _has_sufficient_still_support(
    run: tuple[TimedSample[SocialVisionObservation], ...],
    config: VisionSequenceConfig,
) -> tuple[bool, float, float]:
    """Return whether motion evidence is strong enough for a stillness claim."""

    explicit_still, unknown, total = _motion_support_stats(run)
    if total <= 0:
        return False, 0.0, 1.0
    unknown_ratio = unknown / total
    explicit_ratio = explicit_still / total
    if explicit_still < config.min_explicit_still_samples:
        return False, explicit_ratio, unknown_ratio
    if unknown_ratio > config.max_unknown_motion_ratio:
        return False, explicit_ratio, unknown_ratio
    return True, explicit_ratio, unknown_ratio


def _derive_floor_pose_entered(
    recent_samples: tuple[TimedSample[SocialVisionObservation], ...],
    latest: TimedSample[SocialVisionObservation],
    config: VisionSequenceConfig,
    gap_limit_s: float,
) -> VisionSequenceEvent | None:
    """Return a floor-pose-entered event when a new floor-like run just began."""

    floor_run = _tail_run(recent_samples, predicate=_is_floor_like, gap_limit_s=gap_limit_s)
    if not floor_run:
        return None
    onset_age_s = float(latest.observed_at) - float(floor_run[0].observed_at)
    if onset_age_s > config.edge_emit_window_s:
        return None
    prior_sample = _sample_before_run(recent_samples, floor_run, gap_limit_s=gap_limit_s)
    if prior_sample is None:
        return None
    prior_pose = _pose_value(prior_sample.value.body_pose)
    if prior_pose in _FLOOR_LIKE_POSES or prior_pose == SocialBodyPose.UNKNOWN.value:
        return None
    confidence = _score_transition_confidence(
        prior_sample=prior_sample,
        latest=latest,
        onset_age_s=onset_age_s,
        transition_window_s=config.transition_window_s,
        gap_limit_s=gap_limit_s,
    )
    return VisionSequenceEvent(
        kind=VisionSequenceKind.FLOOR_POSE_ENTERED,
        window_start_s=float(floor_run[0].observed_at),
        window_end_s=float(latest.observed_at),
        confidence=confidence,
        source="vision_pose_sequence",
        supporting_fields=("body_pose", "body_state_changed_at"),
    )


def _derive_downward_transition(
    recent_samples: tuple[TimedSample[SocialVisionObservation], ...],
    latest: TimedSample[SocialVisionObservation],
    config: VisionSequenceConfig,
    gap_limit_s: float,
) -> VisionSequenceEvent | None:
    """Return a downward-transition event from recent pose history."""

    floor_run = _tail_run(recent_samples, predicate=_is_floor_like, gap_limit_s=gap_limit_s)
    if not floor_run:
        return None
    onset_age_s = float(latest.observed_at) - float(floor_run[0].observed_at)
    if onset_age_s > config.edge_emit_window_s:
        return None
    prior_sample = _sample_before_run(recent_samples, floor_run, gap_limit_s=gap_limit_s)
    if prior_sample is None:
        return None
    if _pose_value(prior_sample.value.body_pose) not in _UPRIGHT_LIKE_POSES:
        return None
    if float(latest.observed_at) - float(prior_sample.observed_at) > config.transition_window_s:
        return None
    confidence = _score_transition_confidence(
        prior_sample=prior_sample,
        latest=latest,
        onset_age_s=onset_age_s,
        transition_window_s=config.transition_window_s,
        gap_limit_s=gap_limit_s,
    )
    return VisionSequenceEvent(
        kind=VisionSequenceKind.DOWNWARD_TRANSITION,
        window_start_s=float(prior_sample.observed_at),
        window_end_s=float(latest.observed_at),
        confidence=confidence,
        source="vision_pose_transition",
        supporting_fields=("body_pose", "motion_state"),
    )


def _derive_floor_stillness(
    recent_samples: tuple[TimedSample[SocialVisionObservation], ...],
    latest: TimedSample[SocialVisionObservation],
    config: VisionSequenceConfig,
    gap_limit_s: float,
) -> VisionSequenceEvent | None:
    """Return a floor-stillness event when floor pose holds without motion."""

    floor_still_run = _tail_run(recent_samples, predicate=_is_floor_still_like, gap_limit_s=gap_limit_s)
    if not floor_still_run:
        return None
    if _run_duration_s(floor_still_run) < config.floor_stillness_hold_s:
        return None
    enough_support, explicit_ratio, unknown_ratio = _has_sufficient_still_support(floor_still_run, config)
    if not enough_support:
        return None
    return VisionSequenceEvent(
        kind=VisionSequenceKind.FLOOR_STILLNESS,
        window_start_s=float(floor_still_run[0].observed_at),
        window_end_s=float(latest.observed_at),
        confidence=_score_hold_confidence(
            run=floor_still_run,
            required_hold_s=config.floor_stillness_hold_s,
            explicit_support_ratio=explicit_ratio,
            unknown_ratio=unknown_ratio,
        ),
        source="vision_pose_plus_motion_hold",
        supporting_fields=("body_pose", "motion_state"),
    )


def _derive_slumped_quiet(
    recent_samples: tuple[TimedSample[SocialVisionObservation], ...],
    latest: TimedSample[SocialVisionObservation],
    config: VisionSequenceConfig,
    gap_limit_s: float,
) -> VisionSequenceEvent | None:
    """Return a slumped-quiet sequence when slumped posture holds still."""

    slumped_still_run = _tail_run(recent_samples, predicate=_is_slumped_still_like, gap_limit_s=gap_limit_s)
    if not slumped_still_run:
        return None
    if _run_duration_s(slumped_still_run) < config.slumped_hold_s:
        return None
    enough_support, explicit_ratio, unknown_ratio = _has_sufficient_still_support(slumped_still_run, config)
    if not enough_support:
        return None
    return VisionSequenceEvent(
        kind=VisionSequenceKind.SLUMPED_QUIET,
        window_start_s=float(slumped_still_run[0].observed_at),
        window_end_s=float(latest.observed_at),
        confidence=_score_hold_confidence(
            run=slumped_still_run,
            required_hold_s=config.slumped_hold_s,
            explicit_support_ratio=explicit_ratio,
            unknown_ratio=unknown_ratio,
        ),
        source="vision_pose_hold",
        supporting_fields=("body_pose", "motion_state"),
    )


def _derive_positive_contact(
    recent_samples: tuple[TimedSample[SocialVisionObservation], ...],
    latest: TimedSample[SocialVisionObservation],
    config: VisionSequenceConfig,
    gap_limit_s: float,
) -> VisionSequenceEvent | None:
    """Return a positive-contact event from smile plus attention hold."""

    positive_run = _tail_run(recent_samples, predicate=_is_positive_contact_sample, gap_limit_s=gap_limit_s)
    if not positive_run:
        return None
    if _run_duration_s(positive_run) < config.positive_contact_hold_s:
        return None
    engaged_ratio = sum(sample.value.engaged_with_device is True for sample in positive_run) / len(positive_run)
    looking_ratio = sum(sample.value.looking_toward_device is True for sample in positive_run) / len(positive_run)
    explicit_ratio = max(engaged_ratio, looking_ratio)
    return VisionSequenceEvent(
        kind=VisionSequenceKind.POSITIVE_CONTACT,
        window_start_s=float(positive_run[0].observed_at),
        window_end_s=float(latest.observed_at),
        confidence=_score_hold_confidence(
            run=positive_run,
            required_hold_s=config.positive_contact_hold_s,
            explicit_support_ratio=explicit_ratio,
            unknown_ratio=0.0,
        ),
        source="vision_smile_plus_attention_hold",
        supporting_fields=("smiling", "looking_toward_device", "engaged_with_device"),
    )


def _derive_showing_intent(
    recent_samples: tuple[TimedSample[SocialVisionObservation], ...],
    latest: TimedSample[SocialVisionObservation],
    config: VisionSequenceConfig,
    gap_limit_s: float,
) -> VisionSequenceEvent | None:
    """Return a showing-intent event from near-camera or explicit intent holds."""

    intent_run = _tail_run(recent_samples, predicate=_is_showing_intent_sample, gap_limit_s=gap_limit_s)
    if not intent_run:
        return None
    if _run_duration_s(intent_run) < config.showing_intent_hold_s:
        return None
    explicit_intent_ratio = sum(sample.value.showing_intent_likely is True for sample in intent_run) / len(intent_run)
    return VisionSequenceEvent(
        kind=VisionSequenceKind.SHOWING_INTENT,
        window_start_s=float(intent_run[0].observed_at),
        window_end_s=float(latest.observed_at),
        confidence=_score_hold_confidence(
            run=intent_run,
            required_hold_s=config.showing_intent_hold_s,
            explicit_support_ratio=max(0.5, explicit_intent_ratio),
            unknown_ratio=0.0,
        ),
        source="vision_showing_intent_hold",
        supporting_fields=("showing_intent_likely", "hand_or_object_near_camera"),
    )


__all__ = [
    "VisionSequenceConfig",
    "VisionSequenceEvent",
    "VisionSequenceKind",
    "derive_vision_sequences",
]