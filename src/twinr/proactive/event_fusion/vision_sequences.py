"""Derive bounded temporal vision sequences for event fusion.

The social camera layer already normalizes single ticks. This module turns the
recent rolling window of those ticks into short sequence-level facts such as
``downward_transition`` or ``slumped_quiet``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from twinr.proactive.event_fusion.buffers import RollingWindowBuffer, TimedSample
from twinr.proactive.social.engine import SocialBodyPose, SocialMotionState, SocialVisionObservation


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
    latest = observation_buffer.latest()
    if latest is None:
        return ()
    sequences: list[VisionSequenceEvent] = []
    recent_samples = observation_buffer.between(max(0.0, now - cfg.transition_window_s), now)
    floor_event = _derive_floor_pose_entered(recent_samples, latest)
    if floor_event is not None:
        sequences.append(floor_event)
    downward_event = _derive_downward_transition(recent_samples, latest, cfg)
    if downward_event is not None:
        sequences.append(downward_event)
    floor_stillness = _derive_floor_stillness(recent_samples, latest, cfg)
    if floor_stillness is not None:
        sequences.append(floor_stillness)
    slumped_quiet = _derive_slumped_quiet(recent_samples, latest, cfg)
    if slumped_quiet is not None:
        sequences.append(slumped_quiet)
    positive_contact = _derive_positive_contact(recent_samples, latest, cfg)
    if positive_contact is not None:
        sequences.append(positive_contact)
    showing_intent = _derive_showing_intent(recent_samples, latest, cfg)
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


def _derive_floor_pose_entered(
    recent_samples: tuple[TimedSample[SocialVisionObservation], ...],
    latest: TimedSample[SocialVisionObservation],
) -> VisionSequenceEvent | None:
    """Return a floor-pose-entered event when the latest sample is floor-like."""

    latest_pose = _pose_value(latest.value.body_pose)
    if latest_pose not in {SocialBodyPose.FLOOR.value, SocialBodyPose.LYING_LOW.value}:
        return None
    prior_non_floor = next(
        (
            sample
            for sample in reversed(recent_samples[:-1])
            if _pose_value(sample.value.body_pose)
            not in {SocialBodyPose.FLOOR.value, SocialBodyPose.LYING_LOW.value, SocialBodyPose.UNKNOWN.value}
        ),
        None,
    )
    window_start = prior_non_floor.observed_at if prior_non_floor is not None else latest.observed_at
    confidence = 0.8 if prior_non_floor is not None else 0.68
    return VisionSequenceEvent(
        kind=VisionSequenceKind.FLOOR_POSE_ENTERED,
        window_start_s=window_start,
        window_end_s=latest.observed_at,
        confidence=confidence,
        source="vision_pose_sequence",
        supporting_fields=("body_pose", "body_state_changed_at"),
    )


def _derive_downward_transition(
    recent_samples: tuple[TimedSample[SocialVisionObservation], ...],
    latest: TimedSample[SocialVisionObservation],
    config: VisionSequenceConfig,
) -> VisionSequenceEvent | None:
    """Return a downward-transition event from recent pose history."""

    latest_pose = _pose_value(latest.value.body_pose)
    if latest_pose not in {SocialBodyPose.FLOOR.value, SocialBodyPose.LYING_LOW.value}:
        return None
    prior_upright = next(
        (
            sample
            for sample in reversed(recent_samples[:-1])
            if _pose_value(sample.value.body_pose)
            in {SocialBodyPose.UPRIGHT.value, SocialBodyPose.SEATED.value, SocialBodyPose.SLUMPED.value}
        ),
        None,
    )
    if prior_upright is None:
        return None
    if latest.observed_at - prior_upright.observed_at > config.transition_window_s:
        return None
    return VisionSequenceEvent(
        kind=VisionSequenceKind.DOWNWARD_TRANSITION,
        window_start_s=prior_upright.observed_at,
        window_end_s=latest.observed_at,
        confidence=0.82,
        source="vision_pose_transition",
        supporting_fields=("body_pose", "motion_state"),
    )


def _derive_floor_stillness(
    recent_samples: tuple[TimedSample[SocialVisionObservation], ...],
    latest: TimedSample[SocialVisionObservation],
    config: VisionSequenceConfig,
) -> VisionSequenceEvent | None:
    """Return a floor-stillness event when floor pose holds without motion."""

    latest_pose = _pose_value(latest.value.body_pose)
    if latest_pose not in {SocialBodyPose.FLOOR.value, SocialBodyPose.LYING_LOW.value}:
        return None
    floor_samples = [
        sample
        for sample in recent_samples
        if _pose_value(sample.value.body_pose) in {SocialBodyPose.FLOOR.value, SocialBodyPose.LYING_LOW.value}
    ]
    if len(floor_samples) < 2:
        return None
    window_start = floor_samples[0].observed_at
    if latest.observed_at - window_start < config.floor_stillness_hold_s:
        return None
    if not all(_motion_value(sample.value.motion_state) in {SocialMotionState.STILL.value, SocialMotionState.UNKNOWN.value} for sample in floor_samples):
        return None
    return VisionSequenceEvent(
        kind=VisionSequenceKind.FLOOR_STILLNESS,
        window_start_s=window_start,
        window_end_s=latest.observed_at,
        confidence=0.88,
        source="vision_pose_plus_motion_hold",
        supporting_fields=("body_pose", "motion_state"),
    )


def _derive_slumped_quiet(
    recent_samples: tuple[TimedSample[SocialVisionObservation], ...],
    latest: TimedSample[SocialVisionObservation],
    config: VisionSequenceConfig,
) -> VisionSequenceEvent | None:
    """Return a slumped-quiet sequence when slumped posture holds still."""

    if _pose_value(latest.value.body_pose) != SocialBodyPose.SLUMPED.value:
        return None
    slumped_samples = [
        sample for sample in recent_samples if _pose_value(sample.value.body_pose) == SocialBodyPose.SLUMPED.value
    ]
    if len(slumped_samples) < 2:
        return None
    window_start = slumped_samples[0].observed_at
    if latest.observed_at - window_start < config.slumped_hold_s:
        return None
    if not all(_motion_value(sample.value.motion_state) in {SocialMotionState.STILL.value, SocialMotionState.UNKNOWN.value} for sample in slumped_samples):
        return None
    return VisionSequenceEvent(
        kind=VisionSequenceKind.SLUMPED_QUIET,
        window_start_s=window_start,
        window_end_s=latest.observed_at,
        confidence=0.76,
        source="vision_pose_hold",
        supporting_fields=("body_pose", "motion_state"),
    )


def _derive_positive_contact(
    recent_samples: tuple[TimedSample[SocialVisionObservation], ...],
    latest: TimedSample[SocialVisionObservation],
    config: VisionSequenceConfig,
) -> VisionSequenceEvent | None:
    """Return a positive-contact event from smile plus attention hold."""

    if latest.value.smiling is not True:
        return None
    positive_samples = [
        sample
        for sample in recent_samples
        if sample.value.smiling is True
        and (sample.value.looking_toward_device is True or sample.value.engaged_with_device is True)
    ]
    if not positive_samples:
        return None
    window_start = positive_samples[0].observed_at
    if latest.observed_at - window_start < config.positive_contact_hold_s:
        return None
    return VisionSequenceEvent(
        kind=VisionSequenceKind.POSITIVE_CONTACT,
        window_start_s=window_start,
        window_end_s=latest.observed_at,
        confidence=0.72,
        source="vision_smile_plus_attention_hold",
        supporting_fields=("smiling", "looking_toward_device", "engaged_with_device"),
    )


def _derive_showing_intent(
    recent_samples: tuple[TimedSample[SocialVisionObservation], ...],
    latest: TimedSample[SocialVisionObservation],
    config: VisionSequenceConfig,
) -> VisionSequenceEvent | None:
    """Return a showing-intent event from near-camera or explicit intent holds."""

    if latest.value.showing_intent_likely is not True and latest.value.hand_or_object_near_camera is not True:
        return None
    intent_samples = [
        sample
        for sample in recent_samples
        if sample.value.showing_intent_likely is True
        or (
            sample.value.hand_or_object_near_camera is True
            and sample.value.looking_toward_device is True
        )
    ]
    if not intent_samples:
        return None
    window_start = intent_samples[0].observed_at
    if latest.observed_at - window_start < config.showing_intent_hold_s:
        return None
    return VisionSequenceEvent(
        kind=VisionSequenceKind.SHOWING_INTENT,
        window_start_s=window_start,
        window_end_s=latest.observed_at,
        confidence=0.78,
        source="vision_showing_intent_hold",
        supporting_fields=("showing_intent_likely", "hand_or_object_near_camera"),
    )


__all__ = [
    "VisionSequenceConfig",
    "VisionSequenceEvent",
    "VisionSequenceKind",
    "derive_vision_sequences",
]
