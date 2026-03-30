# CHANGELOG: 2026-03-29
# BUG-1: Fixed quiet-state computation so room_quiet/recent_speech_age_s contribute and self-TTS / speech overlap no longer fabricate "quiet" holds.
# BUG-2: Fixed stale/outage handling by rejecting regressing timestamps and treating stale/failed
#        camera frames as non-authoritative instead of generic positive presence.
# BUG-3: Fixed person-visible normalization so redundant evidence (count/box/visible_persons) is used when upstream booleans drift.
# BUG-4: Fixed fall / slumped logic to reject low-confidence pose and motion states instead of trusting weak vision classifications.
# SEC-1: Bounded user-facing names and per-tick object/person lists to reduce prompt-injection surface and resource-exhaustion risk on Pi-class hardware.
# IMP-1: Added sensor-health / freshness-aware multimodal fusion gates aligned with 2025/2026 missing-modality and uncertainty-aware fusion patterns.
# IMP-2: Added configurable freshness / confidence thresholds and cooldown overrides from TwinrConfig for frontier-grade runtime tuning.

"""Evaluate proactive social triggers from normalized sensor observations.

This module defines the social-trigger domain model, bounded threshold config,
and the stateful engine that turns recent vision, audio, and PIR signals into
one candidate proactive prompt.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import TYPE_CHECKING, SupportsFloat, SupportsIndex, TypeAlias, cast

from twinr.agent.base_agent.config import TwinrConfig

from .normalization import (
    coerce_enum_member,
    coerce_non_negative_int_or_default,
    coerce_spatial_box_coordinates,
)
from .scoring import TriggerScoreEvidence, bool_score, hold_progress, recent_progress, weighted_trigger_score

if TYPE_CHECKING:
    from .perception_stream import PerceptionStreamObservation


_FloatLike: TypeAlias = str | bytes | bytearray | SupportsFloat | SupportsIndex

_MAX_DETECTED_OBJECTS = 16
_MAX_VISIBLE_PERSONS = 4
_MAX_DISPLAY_NAME_LENGTH = 48
_MIN_ZONE_LEFT_EDGE = 1.0 / 3.0
_MIN_ZONE_RIGHT_EDGE = 2.0 / 3.0

_DEFAULT_COOLDOWNS: dict[str, float] = {
    "person_returned": 30.0 * 60.0,
    "attention_window": 10.0 * 60.0,
    "slumped_quiet": 20.0 * 60.0,
    "possible_fall": 60.0,
    "floor_stillness": 60.0,
    "showing_intent": 5.0 * 60.0,
    "distress_possible": 15.0 * 60.0,
    "positive_contact": 20.0 * 60.0,
}

_SOCIAL_TRIGGERS = frozenset(
    {
        "person_returned",
        "attention_window",
        "showing_intent",
        "positive_contact",
    }
)
_SAFETY_TRIGGERS = frozenset(
    {
        "slumped_quiet",
        "distress_possible",
        "possible_fall",
        "floor_stillness",
    }
)


def _coerce_bool(value: object, *, default: bool = False) -> bool:
    """Coerce one value to a boolean with fallback."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _coerce_optional_bool(value: object) -> bool | None:
    """Coerce one value to ``bool`` or ``None``."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return None


def _coerce_timestamp(value: object) -> float | None:
    """Coerce one value to a non-negative finite timestamp."""

    try:
        timestamp = float(cast(_FloatLike, value))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(timestamp) or timestamp < 0.0:
        return None
    return timestamp


def _normalize_positive_float(value: object, *, default: float) -> float:
    """Coerce one value to a positive finite float."""

    try:
        number = float(cast(_FloatLike, value))
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number) or number <= 0.0:
        return default
    return number


def _normalize_non_negative_float(value: object, *, default: float = 0.0) -> float:
    """Coerce one value to a non-negative finite float."""

    try:
        number = float(cast(_FloatLike, value))
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number) or number < 0.0:
        return default
    return number


def _coerce_recent_age(value: object) -> float | None:
    """Coerce one age-like value to a non-negative finite float."""

    if value is None:
        return None
    try:
        number = float(cast(_FloatLike, value))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0.0:
        return None
    return number


def _coerce_optional_text(value: object, *, limit: int) -> str | None:
    """Coerce one optional text field into bounded printable text."""

    if value is None:
        return None
    text = " ".join(str(value).split())
    if not text:
        return None
    printable = "".join(character for character in text if character.isprintable())
    printable = printable.strip()
    if not printable:
        return None
    return printable[:limit]


def _coerce_optional_azimuth(value: object) -> int | None:
    """Coerce one azimuth value into ``0..359`` or ``None``."""

    if value is None:
        return None
    try:
        number = int(float(cast(_FloatLike, value)))
    except (TypeError, ValueError):
        return None
    return number % 360


def _normalize_unit_interval(value: object, *, default: float) -> float:
    """Clamp one numeric value into ``[0.0, 1.0]`` with fallback."""

    try:
        number = float(cast(_FloatLike, value))
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number


def _coerce_display_name(value: object) -> str | None:
    """Coerce one display name into a short one-line printable value."""

    text = _coerce_optional_text(value, limit=_MAX_DISPLAY_NAME_LENGTH)
    if text is None:
        return None
    return text.strip(" ,.;:-") or None


class SocialBodyPose(StrEnum):
    """Describe the coarse body-pose classes used by trigger logic."""

    UNKNOWN = "unknown"
    UPRIGHT = "upright"
    SEATED = "seated"
    SLUMPED = "slumped"
    LYING_LOW = "lying_low"
    FLOOR = "floor"


class SocialMotionState(StrEnum):
    """Describe the coarse motion states exposed by the camera path."""

    UNKNOWN = "unknown"
    STILL = "still"
    WALKING = "walking"
    APPROACHING = "approaching"
    LEAVING = "leaving"


class SocialPersonZone(StrEnum):
    """Describe the coarse horizontal zone of the primary visible person."""

    UNKNOWN = "unknown"
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class SocialGestureEvent(StrEnum):
    """Describe the bounded coarse-arm gesture vocabulary exposed to policy consumers."""

    NONE = "none"
    WAVE = "wave"
    STOP = "stop"
    DISMISS = "dismiss"
    CONFIRM = "confirm"
    ARMS_CROSSED = "arms_crossed"
    TWO_HAND_DISMISS = "two_hand_dismiss"
    TIMEOUT_T = "timeout_t"
    UNKNOWN = "unknown"


class SocialFineHandGesture(StrEnum):
    """Describe the bounded fine-hand gesture vocabulary exposed to policy consumers."""

    NONE = "none"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    POINTING = "pointing"
    PEACE_SIGN = "peace_sign"
    OPEN_PALM = "open_palm"
    OK_SIGN = "ok_sign"
    MIDDLE_FINGER = "middle_finger"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class SocialSpatialBox:
    """Describe one normalized bounding box in ``top,left,bottom,right`` order."""

    top: float
    left: float
    bottom: float
    right: float

    def __post_init__(self) -> None:
        """Clamp coordinates into bounds and enforce monotonic edges."""

        top = _normalize_unit_interval(self.top, default=0.0)
        left = _normalize_unit_interval(self.left, default=0.0)
        bottom = _normalize_unit_interval(self.bottom, default=top)
        right = _normalize_unit_interval(self.right, default=left)
        if bottom < top:
            bottom = top
        if right < left:
            right = left
        object.__setattr__(self, "top", top)
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "bottom", bottom)
        object.__setattr__(self, "right", right)

    @property
    def center_x(self) -> float:
        """Return the normalized horizontal center."""

        return (self.left + self.right) / 2.0

    @property
    def center_y(self) -> float:
        """Return the normalized vertical center."""

        return (self.top + self.bottom) / 2.0

    @property
    def area(self) -> float:
        """Return the normalized area."""

        return max(0.0, self.bottom - self.top) * max(0.0, self.right - self.left)


@dataclass(frozen=True, slots=True)
class SocialDetectedObject:
    """Describe one explicit object detection for downstream policy consumers."""

    label: str
    confidence: float
    zone: SocialPersonZone = SocialPersonZone.UNKNOWN
    stable: bool = False
    box: SocialSpatialBox | None = None

    def __post_init__(self) -> None:
        """Normalize object metadata into bounded, inspectable values."""

        label = str(self.label or "").strip().lower().replace(" ", "_")
        if not label:
            label = "unknown"
        object.__setattr__(self, "label", label[:64])
        object.__setattr__(self, "confidence", _normalize_unit_interval(self.confidence, default=0.0))
        object.__setattr__(self, "zone", _coerce_person_zone(self.zone))
        object.__setattr__(self, "stable", _coerce_bool(self.stable))
        object.__setattr__(self, "box", _coerce_spatial_box(self.box))


@dataclass(frozen=True, slots=True)
class SocialVisiblePerson:
    """Describe one visible person anchor for short-lived runtime targeting."""

    box: SocialSpatialBox | None = None
    zone: SocialPersonZone = SocialPersonZone.UNKNOWN
    confidence: float = 0.0

    def __post_init__(self) -> None:
        """Normalize person-anchor metadata into bounded values."""

        object.__setattr__(self, "box", _coerce_spatial_box(self.box))
        zone = _coerce_person_zone(self.zone)
        box = cast(SocialSpatialBox | None, getattr(self, "box"))
        if zone == SocialPersonZone.UNKNOWN and box is not None:
            zone = _zone_from_center_x(box.center_x)
        object.__setattr__(self, "zone", zone)
        object.__setattr__(self, "confidence", _normalize_unit_interval(self.confidence, default=0.0))


def _coerce_body_pose(value: object) -> SocialBodyPose:
    """Coerce one value to a known body pose."""

    return coerce_enum_member(
        value,
        SocialBodyPose,
        unknown=SocialBodyPose.UNKNOWN,
    )


def _coerce_motion_state(value: object) -> SocialMotionState:
    """Coerce one value to a known motion state."""

    return coerce_enum_member(
        value,
        SocialMotionState,
        unknown=SocialMotionState.UNKNOWN,
    )


def _coerce_gesture_event(value: object) -> SocialGestureEvent:
    """Coerce one value to a known gesture event."""

    return coerce_enum_member(
        value,
        SocialGestureEvent,
        unknown=SocialGestureEvent.UNKNOWN,
    )


def _coerce_fine_hand_gesture(value: object) -> SocialFineHandGesture:
    """Coerce one value to a bounded fine-hand gesture enum."""

    return coerce_enum_member(
        value,
        SocialFineHandGesture,
        unknown=SocialFineHandGesture.UNKNOWN,
    )


def _coerce_person_zone(value: object) -> SocialPersonZone:
    """Coerce one value to a known coarse person zone."""

    return coerce_enum_member(
        value,
        SocialPersonZone,
        unknown=SocialPersonZone.UNKNOWN,
    )


def _zone_from_center_x(center_x: float | None) -> SocialPersonZone:
    """Derive one coarse horizontal zone from a normalized horizontal center."""

    if center_x is None:
        return SocialPersonZone.UNKNOWN
    if center_x < _MIN_ZONE_LEFT_EDGE:
        return SocialPersonZone.LEFT
    if center_x > _MIN_ZONE_RIGHT_EDGE:
        return SocialPersonZone.RIGHT
    return SocialPersonZone.CENTER


def _coerce_non_negative_int(value: object, *, default: int) -> int:
    """Coerce one value to a non-negative integer with fallback."""

    return coerce_non_negative_int_or_default(value, default=default)


def _coerce_bounded_text(value: object, *, max_length: int = 160) -> str | None:
    """Coerce one value to bounded operator-safe text."""

    return _coerce_optional_text(value, limit=max_length)


def _coerce_spatial_box(value: object) -> SocialSpatialBox | None:
    """Coerce one value to one normalized spatial box."""

    if isinstance(value, SocialSpatialBox):
        return value
    coordinates = coerce_spatial_box_coordinates(value)
    if coordinates is None:
        return None
    top, left, bottom, right = coordinates
    return SocialSpatialBox(top=top, left=left, bottom=bottom, right=right)


def _coerce_optional_ratio(value: object) -> float | None:
    """Coerce one optional ratio value into ``[0.0, 1.0]``."""

    if value is None:
        return None
    return _normalize_unit_interval(value, default=0.0)


def _coerce_detected_object(value: object) -> SocialDetectedObject | None:
    """Coerce one object-like payload to ``SocialDetectedObject``."""

    if isinstance(value, SocialDetectedObject):
        return value
    if isinstance(value, dict):
        return SocialDetectedObject(
            label=value.get("label", ""),
            confidence=value.get("confidence", 0.0),
            zone=value.get("zone", SocialPersonZone.UNKNOWN),
            stable=value.get("stable", False),
            box=value.get("box"),
        )
    return None


def _coerce_detected_objects(
    value: object,
    *,
    limit: int = _MAX_DETECTED_OBJECTS,
) -> tuple[SocialDetectedObject, ...]:
    """Coerce one iterable payload to a tuple of detected objects."""

    if value is None:
        return ()
    if isinstance(value, tuple) and all(isinstance(item, SocialDetectedObject) for item in value):
        return value[:limit]
    if not isinstance(value, (tuple, list)):
        return ()
    items: list[SocialDetectedObject] = []
    for item in value[:limit]:
        detected = _coerce_detected_object(item)
        if detected is not None:
            items.append(detected)
    return tuple(items)


def _coerce_visible_person(value: object) -> SocialVisiblePerson | None:
    """Coerce one person-like payload to ``SocialVisiblePerson``."""

    if isinstance(value, SocialVisiblePerson):
        return value
    if isinstance(value, dict):
        return SocialVisiblePerson(
            box=value.get("box"),
            zone=value.get("zone", SocialPersonZone.UNKNOWN),
            confidence=value.get("confidence", 0.0),
        )
    return None


def _coerce_visible_persons(
    value: object,
    *,
    limit: int = _MAX_VISIBLE_PERSONS,
) -> tuple[SocialVisiblePerson, ...]:
    """Coerce one iterable payload to a tuple of visible-person anchors."""

    if value is None:
        return ()
    if isinstance(value, tuple) and all(isinstance(item, SocialVisiblePerson) for item in value):
        return value[:limit]
    if not isinstance(value, (tuple, list)):
        return ()
    items: list[SocialVisiblePerson] = []
    for item in value[:limit]:
        person = _coerce_visible_person(item)
        if person is not None:
            items.append(person)
    return tuple(items)


def _is_floor_like_pose(body_pose: SocialBodyPose) -> bool:
    """Return whether one coarse pose should count as floor-like."""

    return body_pose in {SocialBodyPose.LYING_LOW, SocialBodyPose.FLOOR}


def _is_upright_like_pose(body_pose: SocialBodyPose) -> bool:
    """Return whether one pose should count as upright for transitions."""

    return body_pose in {SocialBodyPose.UPRIGHT, SocialBodyPose.SEATED, SocialBodyPose.SLUMPED}


class SocialTriggerPriority(IntEnum):
    """Order social triggers from least to most urgent."""

    POSITIVE_CONTACT = 10
    PERSON_RETURNED = 20
    SHOWING_INTENT = 30
    ATTENTION_WINDOW = 40
    SLUMPED_QUIET = 60
    DISTRESS_POSSIBLE = 70
    POSSIBLE_FALL = 80
    FLOOR_STILLNESS = 90


@dataclass(frozen=True, slots=True)
class SocialVisionObservation:
    """Describe one normalized vision observation tick."""

    person_visible: bool = False
    person_count: int = 0
    person_recently_visible: bool | None = None
    person_appeared_at: float | None = None
    person_disappeared_at: float | None = None
    primary_person_zone: SocialPersonZone = SocialPersonZone.UNKNOWN
    primary_person_box: SocialSpatialBox | None = None
    visible_persons: tuple[SocialVisiblePerson, ...] = ()
    primary_person_center_x: float | None = None
    primary_person_center_y: float | None = None
    looking_toward_device: bool = False
    looking_signal_state: str | None = None
    looking_signal_source: str | None = None
    person_near_device: bool | None = None
    engaged_with_device: bool | None = None
    visual_attention_score: float | None = None
    body_pose: SocialBodyPose = SocialBodyPose.UNKNOWN
    pose_confidence: float | None = None
    body_state_changed_at: float | None = None
    motion_state: SocialMotionState = SocialMotionState.UNKNOWN
    motion_confidence: float | None = None
    motion_state_changed_at: float | None = None
    smiling: bool = False
    hand_or_object_near_camera: bool = False
    showing_intent_likely: bool | None = None
    showing_intent_started_at: float | None = None
    coarse_arm_gesture: SocialGestureEvent = SocialGestureEvent.NONE
    gesture_event: SocialGestureEvent = SocialGestureEvent.NONE
    gesture_confidence: float | None = None
    fine_hand_gesture: SocialFineHandGesture = SocialFineHandGesture.NONE
    fine_hand_gesture_confidence: float | None = None
    objects: tuple[SocialDetectedObject, ...] = ()
    camera_online: bool = False
    camera_ready: bool = False
    camera_ai_ready: bool = False
    camera_error: str | None = None
    last_camera_frame_at: float | None = None
    last_camera_health_change_at: float | None = None
    perception_stream: PerceptionStreamObservation | None = None


@dataclass(frozen=True, slots=True)
class SocialAudioObservation:
    """Describe one normalized audio observation tick."""

    speech_detected: bool | None = None
    distress_detected: bool | None = None
    room_quiet: bool | None = None
    recent_speech_age_s: float | None = None
    assistant_output_active: bool | None = None
    azimuth_deg: int | None = None
    direction_confidence: float | None = None
    device_runtime_mode: str | None = None
    signal_source: str | None = None
    host_control_ready: bool | None = None
    transport_reason: str | None = None
    non_speech_audio_likely: bool | None = None
    background_media_likely: bool | None = None
    speech_overlap_likely: bool | None = None
    barge_in_detected: bool | None = None
    mute_active: bool | None = None


@dataclass(frozen=True, slots=True)
class SocialObservation:
    """Combine normalized sensor observations for one trigger-engine tick."""

    observed_at: float
    inspected: bool = False
    pir_motion_detected: bool = False
    low_motion: bool = False
    vision: SocialVisionObservation = field(default_factory=SocialVisionObservation)
    audio: SocialAudioObservation = field(default_factory=SocialAudioObservation)


@dataclass(frozen=True, slots=True)
class SocialTriggerDecision:
    """Describe one emitted proactive social trigger."""

    trigger_id: str
    prompt: str
    reason: str
    observed_at: float
    priority: SocialTriggerPriority
    score: float = 1.0
    threshold: float = 1.0
    evidence: tuple[TriggerScoreEvidence, ...] = ()


@dataclass(frozen=True, slots=True)
class SocialTriggerEvaluation:
    """Describe one scored trigger candidate before selection."""

    trigger_id: str
    prompt: str
    reason: str
    observed_at: float
    priority: SocialTriggerPriority
    score: float
    threshold: float
    evidence: tuple[TriggerScoreEvidence, ...] = ()
    passed: bool = False
    blocked_reason: str | None = None


@dataclass(frozen=True, slots=True)
class SocialTriggerThresholds:
    """Store hold windows, freshness gates, and score thresholds for social triggers."""

    person_returned_absence_s: float = 20.0 * 60.0
    person_returned_recent_motion_s: float = 30.0
    attention_window_s: float = 6.0
    slumped_quiet_s: float = 20.0
    possible_fall_stillness_s: float = 10.0
    possible_fall_visibility_loss_hold_s: float = 15.0
    possible_fall_visibility_loss_arming_s: float = 6.0
    possible_fall_slumped_visibility_loss_arming_s: float = 4.0
    floor_stillness_s: float = 20.0
    showing_intent_hold_s: float = 1.5
    positive_contact_hold_s: float = 1.5
    distress_hold_s: float = 3.0
    fall_transition_window_s: float = 8.0
    quiet_after_speech_age_s: float = 2.5
    vision_stale_after_s: float = 2.0
    min_pose_confidence: float = 0.45
    min_motion_confidence: float = 0.35
    min_visual_attention_score: float = 0.45
    person_returned_score_threshold: float = 0.9
    attention_window_score_threshold: float = 0.86
    slumped_quiet_score_threshold: float = 0.9
    possible_fall_score_threshold: float = 0.82
    floor_stillness_score_threshold: float = 0.9
    showing_intent_score_threshold: float = 0.84
    positive_contact_score_threshold: float = 0.84
    distress_possible_score_threshold: float = 0.85

    def __post_init__(self) -> None:
        """Normalize duration and score thresholds after construction."""

        duration_defaults = {
            "person_returned_absence_s": 20.0 * 60.0,
            "person_returned_recent_motion_s": 30.0,
            "attention_window_s": 6.0,
            "slumped_quiet_s": 20.0,
            "possible_fall_stillness_s": 10.0,
            "possible_fall_visibility_loss_hold_s": 15.0,
            "possible_fall_visibility_loss_arming_s": 6.0,
            "possible_fall_slumped_visibility_loss_arming_s": 4.0,
            "floor_stillness_s": 20.0,
            "showing_intent_hold_s": 1.5,
            "positive_contact_hold_s": 1.5,
            "distress_hold_s": 3.0,
            "fall_transition_window_s": 8.0,
            "quiet_after_speech_age_s": 2.5,
            "vision_stale_after_s": 2.0,
        }
        ratio_defaults = {
            "min_pose_confidence": 0.45,
            "min_motion_confidence": 0.35,
            "min_visual_attention_score": 0.45,
            "person_returned_score_threshold": 0.9,
            "attention_window_score_threshold": 0.86,
            "slumped_quiet_score_threshold": 0.9,
            "possible_fall_score_threshold": 0.82,
            "floor_stillness_score_threshold": 0.9,
            "showing_intent_score_threshold": 0.84,
            "positive_contact_score_threshold": 0.84,
            "distress_possible_score_threshold": 0.85,
        }
        for name, default in duration_defaults.items():
            object.__setattr__(self, name, _normalize_positive_float(getattr(self, name), default=default))
        for name, default in ratio_defaults.items():
            object.__setattr__(self, name, _normalize_unit_interval(getattr(self, name), default=default))

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "SocialTriggerThresholds":
        """Build thresholds from ``TwinrConfig`` with safe defaults."""

        defaults = cls()
        return cls(
            person_returned_absence_s=getattr(config, "proactive_person_returned_absence_s", defaults.person_returned_absence_s),
            person_returned_recent_motion_s=getattr(
                config,
                "proactive_person_returned_recent_motion_s",
                defaults.person_returned_recent_motion_s,
            ),
            attention_window_s=getattr(config, "proactive_attention_window_s", defaults.attention_window_s),
            slumped_quiet_s=getattr(config, "proactive_slumped_quiet_s", defaults.slumped_quiet_s),
            possible_fall_stillness_s=getattr(
                config,
                "proactive_possible_fall_stillness_s",
                defaults.possible_fall_stillness_s,
            ),
            possible_fall_visibility_loss_hold_s=getattr(
                config,
                "proactive_possible_fall_visibility_loss_hold_s",
                defaults.possible_fall_visibility_loss_hold_s,
            ),
            possible_fall_visibility_loss_arming_s=getattr(
                config,
                "proactive_possible_fall_visibility_loss_arming_s",
                defaults.possible_fall_visibility_loss_arming_s,
            ),
            possible_fall_slumped_visibility_loss_arming_s=getattr(
                config,
                "proactive_possible_fall_slumped_visibility_loss_arming_s",
                defaults.possible_fall_slumped_visibility_loss_arming_s,
            ),
            floor_stillness_s=getattr(config, "proactive_floor_stillness_s", defaults.floor_stillness_s),
            showing_intent_hold_s=getattr(config, "proactive_showing_intent_hold_s", defaults.showing_intent_hold_s),
            positive_contact_hold_s=getattr(config, "proactive_positive_contact_hold_s", defaults.positive_contact_hold_s),
            distress_hold_s=getattr(config, "proactive_distress_hold_s", defaults.distress_hold_s),
            fall_transition_window_s=getattr(
                config,
                "proactive_fall_transition_window_s",
                defaults.fall_transition_window_s,
            ),
            quiet_after_speech_age_s=getattr(
                config,
                "proactive_quiet_after_speech_age_s",
                defaults.quiet_after_speech_age_s,
            ),
            vision_stale_after_s=getattr(
                config,
                "proactive_vision_stale_after_s",
                defaults.vision_stale_after_s,
            ),
            min_pose_confidence=getattr(
                config,
                "proactive_min_pose_confidence",
                defaults.min_pose_confidence,
            ),
            min_motion_confidence=getattr(
                config,
                "proactive_min_motion_confidence",
                defaults.min_motion_confidence,
            ),
            min_visual_attention_score=getattr(
                config,
                "proactive_min_visual_attention_score",
                defaults.min_visual_attention_score,
            ),
            person_returned_score_threshold=getattr(
                config,
                "proactive_person_returned_score_threshold",
                defaults.person_returned_score_threshold,
            ),
            attention_window_score_threshold=getattr(
                config,
                "proactive_attention_window_score_threshold",
                defaults.attention_window_score_threshold,
            ),
            slumped_quiet_score_threshold=getattr(
                config,
                "proactive_slumped_quiet_score_threshold",
                defaults.slumped_quiet_score_threshold,
            ),
            possible_fall_score_threshold=getattr(
                config,
                "proactive_possible_fall_score_threshold",
                defaults.possible_fall_score_threshold,
            ),
            floor_stillness_score_threshold=getattr(
                config,
                "proactive_floor_stillness_score_threshold",
                defaults.floor_stillness_score_threshold,
            ),
            showing_intent_score_threshold=getattr(
                config,
                "proactive_showing_intent_score_threshold",
                defaults.showing_intent_score_threshold,
            ),
            positive_contact_score_threshold=getattr(
                config,
                "proactive_positive_contact_score_threshold",
                defaults.positive_contact_score_threshold,
            ),
            distress_possible_score_threshold=getattr(
                config,
                "proactive_distress_possible_score_threshold",
                defaults.distress_possible_score_threshold,
            ),
        )


class SocialTriggerEngine:
    """Track recent social signals and emit the strongest passing trigger.

    The engine maintains bounded state across observation ticks so it can
    reason about visibility gaps, holds, cooldowns, and fall-like transitions
    without mixing hardware acquisition into the scoring logic.
    """

    def __init__(
        self,
        *,
        user_name: str | None = None,
        thresholds: SocialTriggerThresholds | None = None,
        cooldowns: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize one stateful trigger engine."""

        self.user_name = _coerce_display_name(user_name)
        self.thresholds = thresholds or SocialTriggerThresholds()
        self._cooldowns = self._build_cooldowns(cooldowns)
        self._last_triggered_at: dict[str, float] = {}
        self._last_pir_motion_at: float | None = None
        self._absence_started_at: float | None = None
        self._visible_since: float | None = None
        self._last_non_floor_pose_at: float | None = None
        self._last_slumped_at: float | None = None
        self._possible_fall_candidate_at: float | None = None
        self._possible_fall_loss_candidate_at: float | None = None
        self._possible_fall_loss_pose: SocialBodyPose = SocialBodyPose.UNKNOWN
        self._possible_fall_loss_visible_duration_s: float | None = None
        self._consecutive_visible_inspected_count: int = 0
        self._person_visible: bool = False
        self._current_pose: SocialBodyPose = SocialBodyPose.UNKNOWN
        self._quiet_since: float | None = None
        self._looking_since: float | None = None
        self._slumped_since: float | None = None
        self._floor_since: float | None = None
        self._low_motion_since: float | None = None
        self._showing_since: float | None = None
        self._smile_since: float | None = None
        self._distress_since: float | None = None
        self._last_evaluations: tuple[SocialTriggerEvaluation, ...] = ()
        self._last_observed_at: float | None = None

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "SocialTriggerEngine":
        """Build one trigger engine from the canonical Twinr config."""

        return cls(
            user_name=getattr(config, "user_display_name", None),
            thresholds=SocialTriggerThresholds.from_config(config),
            cooldowns={
                trigger_id: getattr(
                    config,
                    f"proactive_{trigger_id}_cooldown_s",
                    default,
                )
                for trigger_id, default in _DEFAULT_COOLDOWNS.items()
            },
        )

    @property
    def last_evaluations(self) -> tuple[SocialTriggerEvaluation, ...]:
        """Return the scored candidates from the latest observation tick."""

        return self._last_evaluations

    @property
    def best_evaluation(self) -> SocialTriggerEvaluation | None:
        """Return the strongest latest candidate, including near misses."""

        if not self._last_evaluations:
            return None
        passed_candidates = tuple(item for item in self._last_evaluations if item.passed)
        if passed_candidates:
            return max(
                passed_candidates,
                key=lambda item: (int(item.priority), item.score),
            )
        return max(
            self._last_evaluations,
            key=lambda item: (item.score, int(item.priority)),
        )

    def observe(self, observation: SocialObservation) -> SocialTriggerDecision | None:
        """Update state from one observation tick and emit a trigger if ready.

        Args:
            observation: One normalized sensor observation for the current tick.

        Returns:
            The selected proactive trigger decision, or ``None`` when no
            candidate passes validation, scoring, and cooldown checks.
        """

        if not isinstance(observation, SocialObservation):
            self._last_evaluations = ()
            return None

        now = _coerce_timestamp(observation.observed_at)
        if now is None:
            self._last_evaluations = ()
            return None

        if self._last_observed_at is not None and now < self._last_observed_at:
            self._last_evaluations = ()
            return None

        inspected = _coerce_bool(observation.inspected)
        pir_motion_detected = _coerce_bool(observation.pir_motion_detected)
        raw_vision = observation.vision
        vision = self._normalize_vision(observation.vision, inspected=inspected, now=now)
        audio = self._normalize_audio(observation.audio)

        low_motion = (
            _coerce_bool(observation.low_motion)
            or (
                inspected
                and vision.person_visible
                and vision.motion_state == SocialMotionState.STILL
            )
        )

        sensor_absence_signal = self._should_treat_as_sensor_absence_signal(
            now=now,
            inspected=inspected,
            pir_motion_detected=pir_motion_detected,
            vision=raw_vision,
        )

        if pir_motion_detected:
            self._last_pir_motion_at = now

        absence_duration = self._update_presence_state(
            now,
            inspected=inspected,
            person_visible=vision.person_visible,
            sensor_absence_signal=sensor_absence_signal,
        )
        self._update_confirmed_visibility_state(
            inspected=inspected,
            person_visible=vision.person_visible,
        )
        self._update_pose_state(
            now,
            inspected=inspected,
            person_visible=vision.person_visible,
            body_pose=vision.body_pose,
        )

        quiet_active = self._quiet_signal_active(audio)
        looking_active = inspected and vision.person_visible and vision.looking_toward_device
        slumped_active = inspected and vision.person_visible and vision.body_pose == SocialBodyPose.SLUMPED
        floor_active = inspected and vision.person_visible and _is_floor_like_pose(vision.body_pose)
        showing_active = (
            inspected
            and vision.person_visible
            and (
                vision.showing_intent_likely is True
                or (vision.looking_toward_device and vision.hand_or_object_near_camera)
            )
        )
        smiling_active = (
            inspected
            and vision.person_visible
            and vision.looking_toward_device
            and vision.smiling
        )
        distress_active = (
            audio.distress_detected is True
            and audio.background_media_likely is not True
            and audio.non_speech_audio_likely is not True
        )

        self._quiet_since = self._next_since(quiet_active, self._quiet_since, now)
        self._looking_since = self._next_since(looking_active, self._looking_since, now)
        self._slumped_since = self._next_since(slumped_active, self._slumped_since, now)
        self._floor_since = self._next_since(floor_active, self._floor_since, now)
        self._low_motion_since = self._next_since(low_motion, self._low_motion_since, now)
        self._showing_since = self._next_since(showing_active, self._showing_since, now)
        self._smile_since = self._next_since(smiling_active, self._smile_since, now)
        self._distress_since = self._next_since(distress_active, self._distress_since, now)

        evaluations = (
            self._candidate_floor_stillness(now, vision),
            self._candidate_possible_fall(now, vision),
            self._candidate_distress_possible(now, vision, audio),
            self._candidate_slumped_quiet(now, vision),
            self._candidate_attention_window(now, vision, audio),
            self._candidate_showing_intent(now, vision, audio),
            self._candidate_positive_contact(now, vision, audio),
            self._candidate_person_returned(
                now,
                absence_duration=absence_duration,
                person_visible=vision.person_visible,
                audio=audio,
            ),
        )
        self._last_evaluations = evaluations
        self._last_observed_at = now

        ready = [candidate for candidate in evaluations if candidate.passed]
        if not ready:
            return None

        selected = max(
            ready,
            key=lambda item: (int(item.priority), item.score),
        )
        self.note_trigger_dispatched(selected.trigger_id, observed_at=now)
        return SocialTriggerDecision(
            trigger_id=selected.trigger_id,
            prompt=selected.prompt,
            reason=selected.reason,
            observed_at=selected.observed_at,
            priority=selected.priority,
            score=selected.score,
            threshold=selected.threshold,
            evidence=selected.evidence,
        )

    def note_trigger_dispatched(self, trigger_id: str, *, observed_at: float) -> None:
        """Record that one trigger was dispatched outside normal selection flow."""

        trigger_key = _coerce_bounded_text(trigger_id, max_length=64)
        observed_at_value = _coerce_timestamp(observed_at)
        if trigger_key is None or observed_at_value is None:
            return
        self._last_triggered_at[trigger_key] = observed_at_value
        if trigger_key in {"possible_fall", "floor_stillness"}:
            self._possible_fall_candidate_at = None
            self._possible_fall_loss_candidate_at = None
            self._possible_fall_loss_pose = SocialBodyPose.UNKNOWN
            self._possible_fall_loss_visible_duration_s = None

    def _candidate_person_returned(
        self,
        now: float,
        *,
        absence_duration: float | None,
        person_visible: bool,
        audio: SocialAudioObservation,
    ) -> SocialTriggerEvaluation:
        """Score whether a person recently returned after tracked absence."""

        evidence = (
            TriggerScoreEvidence(
                key="absence_hold",
                value=0.0 if absence_duration is None else min(1.0, absence_duration / self.thresholds.person_returned_absence_s),
                weight=0.4,
                detail=(
                    f"absence={self._seconds(absence_duration)} target={self.thresholds.person_returned_absence_s:.1f}s"
                    if absence_duration is not None
                    else "no completed absence window"
                ),
            ),
            TriggerScoreEvidence(
                key="recent_pir_motion",
                value=recent_progress(now, self._last_pir_motion_at, self.thresholds.person_returned_recent_motion_s),
                weight=0.35,
                detail=f"last_motion_age={self._seconds(None if self._last_pir_motion_at is None else now - self._last_pir_motion_at)}",
            ),
            TriggerScoreEvidence(
                key="person_visible",
                value=bool_score(person_visible),
                weight=0.15,
                detail=f"person_visible={person_visible}",
            ),
            TriggerScoreEvidence(
                key="social_channel_clear",
                value=bool_score(self._social_prompt_channel_available(audio)),
                weight=0.1,
                detail=self._social_channel_detail(audio),
            ),
        )

        blocked_reason = None
        if absence_duration is None:
            blocked_reason = "no_completed_absence_window"
        elif self._cooldown_active("person_returned", now):
            blocked_reason = "cooldown_active"
        else:
            blocked_reason = self._social_trigger_block_reason("person_returned", audio)

        return self._evaluate_candidate(
            trigger_id="person_returned",
            observed_at=now,
            priority=SocialTriggerPriority.PERSON_RETURNED,
            threshold=self.thresholds.person_returned_score_threshold,
            reason=(
                f"Person visible again after {int(absence_duration)} seconds without tracked presence."
                if absence_duration is not None
                else "Person has not yet completed a tracked absence window."
            ),
            prompt=self._with_name(
                base="Schön dich zu sehen. Wie geht's dir?",
                with_name="Hey {name}, schön dich zu sehen. Wie geht's dir?",
            ),
            evidence=evidence,
            blocked_reason=blocked_reason,
        )

    def _candidate_attention_window(
        self,
        now: float,
        vision: SocialVisionObservation,
        audio: SocialAudioObservation,
    ) -> SocialTriggerEvaluation:
        """Score whether the device has a short quiet attention window."""

        evidence = (
            TriggerScoreEvidence(
                key="looking_hold",
                value=hold_progress(now, self._looking_since, self.thresholds.attention_window_s),
                weight=0.36,
                detail=self._hold_detail(self._looking_since, now, self.thresholds.attention_window_s),
            ),
            TriggerScoreEvidence(
                key="quiet_hold",
                value=hold_progress(now, self._quiet_since, self.thresholds.attention_window_s),
                weight=0.36,
                detail=self._hold_detail(self._quiet_since, now, self.thresholds.attention_window_s),
            ),
            TriggerScoreEvidence(
                key="co_present_attention",
                value=bool_score(self._conjunction_since(self._looking_since, self._quiet_since) is not None),
                weight=0.1,
                detail="looking and quiet overlap",
            ),
            TriggerScoreEvidence(
                key="visual_attention_confidence",
                value=self._attention_signal_score(vision),
                weight=0.18,
                detail=f"visual_attention={self._ratio_or_na(vision.visual_attention_score)} looking={vision.looking_toward_device}",
            ),
        )

        blocked_reason = None
        if self._cooldown_active("attention_window", now):
            blocked_reason = "cooldown_active"
        else:
            blocked_reason = self._social_trigger_block_reason("attention_window", audio)

        return self._evaluate_candidate(
            trigger_id="attention_window",
            observed_at=now,
            priority=SocialTriggerPriority.ATTENTION_WINDOW,
            threshold=self.thresholds.attention_window_score_threshold,
            reason="Person was visible, looking toward the device, and quiet for a short attention window.",
            prompt="Kann ich dir bei etwas helfen?",
            evidence=evidence,
            blocked_reason=blocked_reason,
        )

    def _candidate_slumped_quiet(
        self,
        now: float,
        vision: SocialVisionObservation,
    ) -> SocialTriggerEvaluation:
        """Score whether a visible person remains slumped, quiet, and still."""

        evidence = (
            TriggerScoreEvidence(
                key="slumped_hold",
                value=hold_progress(now, self._slumped_since, self.thresholds.slumped_quiet_s),
                weight=0.32,
                detail=self._hold_detail(self._slumped_since, now, self.thresholds.slumped_quiet_s),
            ),
            TriggerScoreEvidence(
                key="quiet_hold",
                value=hold_progress(now, self._quiet_since, self.thresholds.slumped_quiet_s),
                weight=0.28,
                detail=self._hold_detail(self._quiet_since, now, self.thresholds.slumped_quiet_s),
            ),
            TriggerScoreEvidence(
                key="low_motion_hold",
                value=hold_progress(now, self._low_motion_since, self.thresholds.slumped_quiet_s),
                weight=0.22,
                detail=self._hold_detail(self._low_motion_since, now, self.thresholds.slumped_quiet_s),
            ),
            TriggerScoreEvidence(
                key="concurrent_signals",
                value=bool_score(
                    self._conjunction_since(
                        self._slumped_since,
                        self._quiet_since,
                        self._low_motion_since,
                    )
                    is not None
                ),
                weight=0.08,
                detail="slumped, quiet, and low-motion overlap",
            ),
            TriggerScoreEvidence(
                key="pose_confidence",
                value=self._pose_reliability_score(vision, expected=SocialBodyPose.SLUMPED),
                weight=0.10,
                detail=f"pose={vision.body_pose.value} pose_confidence={self._ratio_or_na(vision.pose_confidence)}",
            ),
        )

        blocked_reason = "cooldown_active" if self._cooldown_active("slumped_quiet", now) else None
        return self._evaluate_candidate(
            trigger_id="slumped_quiet",
            observed_at=now,
            priority=SocialTriggerPriority.SLUMPED_QUIET,
            threshold=self.thresholds.slumped_quiet_score_threshold,
            reason="Person stayed visibly slumped, quiet, and low-motion.",
            prompt=self._with_name(
                base="Ist alles in Ordnung?",
                with_name="Hey {name}, ist alles in Ordnung?",
            ),
            evidence=evidence,
            blocked_reason=blocked_reason,
        )

    def _candidate_possible_fall(
        self,
        now: float,
        vision: SocialVisionObservation,
    ) -> SocialTriggerEvaluation:
        """Score whether recent state suggests a possible fall."""

        low_motion_since = self._possible_fall_post_transition_since(now, self._low_motion_since)
        quiet_since = self._possible_fall_post_transition_since(now, self._quiet_since)
        evidence = (
            TriggerScoreEvidence(
                key="fall_transition_signal",
                value=self._possible_fall_transition_signal(),
                weight=0.34,
                detail=self._possible_fall_transition_detail(now),
            ),
            TriggerScoreEvidence(
                key="low_or_missing_hold",
                value=self._possible_fall_low_or_missing_hold(now),
                weight=0.22,
                detail=self._possible_fall_low_or_missing_detail(now),
            ),
            TriggerScoreEvidence(
                key="low_motion_hold",
                value=hold_progress(now, low_motion_since, self.thresholds.possible_fall_stillness_s),
                weight=0.18,
                detail=self._hold_detail(low_motion_since, now, self.thresholds.possible_fall_stillness_s),
            ),
            TriggerScoreEvidence(
                key="quiet_hold",
                value=hold_progress(now, quiet_since, self.thresholds.possible_fall_stillness_s),
                weight=0.12,
                detail=self._hold_detail(quiet_since, now, self.thresholds.possible_fall_stillness_s),
            ),
            TriggerScoreEvidence(
                key="sensor_reliability",
                value=self._safety_reliability_score(vision),
                weight=0.14,
                detail=self._safety_reliability_detail(vision),
            ),
        )

        blocked_reason = None
        if self._cooldown_active("possible_fall", now):
            blocked_reason = "cooldown_active"
        elif self._possible_fall_loss_candidate_at is not None and not self._possible_fall_visibility_loss_hold_complete(now):
            blocked_reason = "visibility_loss_hold_incomplete"

        return self._evaluate_candidate(
            trigger_id="possible_fall",
            observed_at=now,
            priority=SocialTriggerPriority.POSSIBLE_FALL,
            threshold=self.thresholds.possible_fall_score_threshold,
            reason="Person dropped sharply lower or disappeared from view after being visible and then stayed still.",
            prompt="Brauchst du Hilfe?",
            evidence=evidence,
            blocked_reason=blocked_reason,
        )

    def _candidate_floor_stillness(
        self,
        now: float,
        vision: SocialVisionObservation,
    ) -> SocialTriggerEvaluation:
        """Score whether a person remains low to the floor and still."""

        evidence = (
            TriggerScoreEvidence(
                key="floor_hold",
                value=hold_progress(now, self._floor_since, self.thresholds.floor_stillness_s),
                weight=0.34,
                detail=self._hold_detail(self._floor_since, now, self.thresholds.floor_stillness_s),
            ),
            TriggerScoreEvidence(
                key="quiet_hold",
                value=hold_progress(now, self._quiet_since, self.thresholds.floor_stillness_s),
                weight=0.28,
                detail=self._hold_detail(self._quiet_since, now, self.thresholds.floor_stillness_s),
            ),
            TriggerScoreEvidence(
                key="low_motion_hold",
                value=hold_progress(now, self._low_motion_since, self.thresholds.floor_stillness_s),
                weight=0.18,
                detail=self._hold_detail(self._low_motion_since, now, self.thresholds.floor_stillness_s),
            ),
            TriggerScoreEvidence(
                key="concurrent_signals",
                value=bool_score(
                    self._conjunction_since(
                        self._floor_since,
                        self._quiet_since,
                        self._low_motion_since,
                    )
                    is not None
                ),
                weight=0.08,
                detail="floor, quiet, and low-motion overlap",
            ),
            TriggerScoreEvidence(
                key="pose_confidence",
                value=self._pose_reliability_score(vision, expected_floor_like=True),
                weight=0.12,
                detail=f"pose={vision.body_pose.value} pose_confidence={self._ratio_or_na(vision.pose_confidence)}",
            ),
        )

        blocked_reason = "cooldown_active" if self._cooldown_active("floor_stillness", now) else None
        return self._evaluate_candidate(
            trigger_id="floor_stillness",
            observed_at=now,
            priority=SocialTriggerPriority.FLOOR_STILLNESS,
            threshold=self.thresholds.floor_stillness_score_threshold,
            reason="Person stayed low to the floor, quiet, and low-motion.",
            prompt=self._with_name(
                base="Antworte mir kurz: Ist alles okay?",
                with_name="Hey {name}, antworte mir kurz: Ist alles okay?",
            ),
            evidence=evidence,
            blocked_reason=blocked_reason,
        )

    def _candidate_showing_intent(
        self,
        now: float,
        vision: SocialVisionObservation,
        audio: SocialAudioObservation,
    ) -> SocialTriggerEvaluation:
        """Score whether the person seems to be showing something to the device."""

        evidence = (
            TriggerScoreEvidence(
                key="showing_hold",
                value=hold_progress(now, self._showing_since, self.thresholds.showing_intent_hold_s),
                weight=0.46,
                detail=self._hold_detail(self._showing_since, now, self.thresholds.showing_intent_hold_s),
            ),
            TriggerScoreEvidence(
                key="hand_or_object_near_camera",
                value=bool_score(vision.hand_or_object_near_camera),
                weight=0.2,
                detail=f"hand_or_object_near_camera={vision.hand_or_object_near_camera}",
            ),
            TriggerScoreEvidence(
                key="looking_toward_device",
                value=bool_score(vision.looking_toward_device),
                weight=0.14,
                detail=f"looking_toward_device={vision.looking_toward_device}",
            ),
            TriggerScoreEvidence(
                key="visual_attention_confidence",
                value=self._attention_signal_score(vision),
                weight=0.1,
                detail=f"visual_attention={self._ratio_or_na(vision.visual_attention_score)}",
            ),
            TriggerScoreEvidence(
                key="social_channel_clear",
                value=bool_score(self._social_prompt_channel_available(audio)),
                weight=0.1,
                detail=self._social_channel_detail(audio),
            ),
        )

        blocked_reason = None
        if self._cooldown_active("showing_intent", now):
            blocked_reason = "cooldown_active"
        else:
            blocked_reason = self._social_trigger_block_reason("showing_intent", audio)

        return self._evaluate_candidate(
            trigger_id="showing_intent",
            observed_at=now,
            priority=SocialTriggerPriority.SHOWING_INTENT,
            threshold=self.thresholds.showing_intent_score_threshold,
            reason="Person looked toward the device while holding a hand or object near the camera.",
            prompt="Möchtest du mir etwas zeigen?",
            evidence=evidence,
            blocked_reason=blocked_reason,
        )

    def _candidate_distress_possible(
        self,
        now: float,
        vision: SocialVisionObservation,
        audio: SocialAudioObservation,
    ) -> SocialTriggerEvaluation:
        """Score whether distress-like audio aligns with concerning posture."""

        evidence = (
            TriggerScoreEvidence(
                key="distress_hold",
                value=hold_progress(now, self._distress_since, self.thresholds.distress_hold_s),
                weight=0.54,
                detail=self._hold_detail(self._distress_since, now, self.thresholds.distress_hold_s),
            ),
            TriggerScoreEvidence(
                key="visible_or_slumped_person",
                value=bool_score(vision.person_visible or vision.body_pose == SocialBodyPose.SLUMPED),
                weight=0.22,
                detail=f"person_visible={vision.person_visible} body_pose={vision.body_pose.value}",
            ),
            TriggerScoreEvidence(
                key="audio_channel_clean",
                value=bool_score(
                    audio.background_media_likely is not True
                    and audio.non_speech_audio_likely is not True
                    and audio.speech_overlap_likely is not True
                ),
                weight=0.14,
                detail=(
                    f"media={audio.background_media_likely} non_speech={audio.non_speech_audio_likely} "
                    f"overlap={audio.speech_overlap_likely}"
                ),
            ),
            TriggerScoreEvidence(
                key="safety_reliability",
                value=self._safety_reliability_score(vision),
                weight=0.10,
                detail=self._safety_reliability_detail(vision),
            ),
        )

        blocked_reason = "cooldown_active" if self._cooldown_active("distress_possible", now) else None
        return self._evaluate_candidate(
            trigger_id="distress_possible",
            observed_at=now,
            priority=SocialTriggerPriority.DISTRESS_POSSIBLE,
            threshold=self.thresholds.distress_possible_score_threshold,
            reason="Distress-like audio coincided with a visible or slumped person.",
            prompt=self._with_name(
                base="Ich wollte nur kurz fragen, ob alles in Ordnung ist.",
                with_name="Hey {name}, ich wollte nur kurz fragen, ob alles in Ordnung ist.",
            ),
            evidence=evidence,
            blocked_reason=blocked_reason,
        )

    def _candidate_positive_contact(
        self,
        now: float,
        vision: SocialVisionObservation,
        audio: SocialAudioObservation,
    ) -> SocialTriggerEvaluation:
        """Score whether the person is positively engaged with the device."""

        evidence = (
            TriggerScoreEvidence(
                key="smile_hold",
                value=hold_progress(now, self._smile_since, self.thresholds.positive_contact_hold_s),
                weight=0.44,
                detail=self._hold_detail(self._smile_since, now, self.thresholds.positive_contact_hold_s),
            ),
            TriggerScoreEvidence(
                key="person_visible",
                value=bool_score(vision.person_visible),
                weight=0.12,
                detail=f"person_visible={vision.person_visible}",
            ),
            TriggerScoreEvidence(
                key="looking_toward_device",
                value=bool_score(vision.looking_toward_device),
                weight=0.12,
                detail=f"looking_toward_device={vision.looking_toward_device}",
            ),
            TriggerScoreEvidence(
                key="smiling_now",
                value=bool_score(vision.smiling),
                weight=0.12,
                detail=f"smiling={vision.smiling}",
            ),
            TriggerScoreEvidence(
                key="visual_attention_confidence",
                value=self._attention_signal_score(vision),
                weight=0.10,
                detail=f"visual_attention={self._ratio_or_na(vision.visual_attention_score)}",
            ),
            TriggerScoreEvidence(
                key="social_channel_clear",
                value=bool_score(self._social_prompt_channel_available(audio)),
                weight=0.10,
                detail=self._social_channel_detail(audio),
            ),
        )

        blocked_reason = None
        if self._cooldown_active("positive_contact", now):
            blocked_reason = "cooldown_active"
        else:
            blocked_reason = self._social_trigger_block_reason("positive_contact", audio)

        return self._evaluate_candidate(
            trigger_id="positive_contact",
            observed_at=now,
            priority=SocialTriggerPriority.POSITIVE_CONTACT,
            threshold=self.thresholds.positive_contact_score_threshold,
            reason="Person smiled while facing the device.",
            prompt=self._with_name(
                base="Schön, dich zu sehen. Was möchtest du machen?",
                with_name="Schön, dich zu sehen, {name}. Was möchtest du machen?",
            ),
            evidence=evidence,
            blocked_reason=blocked_reason,
        )

    def _update_presence_state(
        self,
        now: float,
        *,
        inspected: bool,
        person_visible: bool,
        sensor_absence_signal: bool = False,
    ) -> float | None:
        """Update tracked presence and return any completed absence duration."""

        if not inspected and not sensor_absence_signal:
            return None

        absence_duration: float | None = None
        if person_visible:
            if not self._person_visible and self._absence_started_at is not None:
                absence_duration = now - self._absence_started_at
            self._absence_started_at = None
            self._possible_fall_loss_candidate_at = None
            self._possible_fall_loss_pose = SocialBodyPose.UNKNOWN
            self._possible_fall_loss_visible_duration_s = None
            if self._visible_since is None:
                self._visible_since = now
        else:
            if self._person_visible:
                visible_duration = None if self._visible_since is None else now - self._visible_since
                if self._fell_out_of_view_after_fall_like_presence(now, visible_duration=visible_duration):
                    self._possible_fall_loss_candidate_at = now
                    self._possible_fall_loss_pose = self._current_pose
                    self._possible_fall_loss_visible_duration_s = visible_duration
                self._visible_since = None
            if self._absence_started_at is None:
                self._absence_started_at = now

        self._person_visible = person_visible
        return absence_duration

    def _should_treat_as_sensor_absence_signal(
        self,
        *,
        now: float,
        inspected: bool,
        pir_motion_detected: bool,
        vision: object,
    ) -> bool:
        """Return whether sensor-only input should extend the absence state."""

        # BREAKING: Generic "uninspected + no PIR" no longer counts as absence. We require
        # explicit camera-health metadata so blind or stalled ticks cannot fabricate long
        # absence windows and later fire false "person_returned" prompts.
        if inspected or pir_motion_detected:
            return False
        if not isinstance(vision, SocialVisionObservation):
            return False
        if _coerce_bool(vision.person_visible):
            return False
        last_camera_frame_at = _coerce_timestamp(getattr(vision, "last_camera_frame_at", None))
        last_camera_health_change_at = _coerce_timestamp(getattr(vision, "last_camera_health_change_at", None))
        explicit_health_metadata = (
            vision.camera_error is not None
            or last_camera_frame_at is not None
            or last_camera_health_change_at is not None
        )
        if not explicit_health_metadata:
            return False
        if last_camera_frame_at is not None and (now - last_camera_frame_at) <= self.thresholds.vision_stale_after_s:
            return False
        return self._person_visible or self._absence_started_at is not None

    def _update_pose_state(
        self,
        now: float,
        *,
        inspected: bool,
        person_visible: bool,
        body_pose: SocialBodyPose,
    ) -> None:
        """Update tracked pose and fall-transition state."""

        if not inspected or not person_visible:
            return
        if _is_upright_like_pose(body_pose):
            self._last_non_floor_pose_at = now
        if body_pose == SocialBodyPose.SLUMPED:
            self._last_slumped_at = now
        if body_pose == self._current_pose:
            return
        self._current_pose = body_pose
        if _is_floor_like_pose(body_pose):
            if (
                self._last_non_floor_pose_at is not None
                and (now - self._last_non_floor_pose_at) <= self.thresholds.fall_transition_window_s
            ):
                self._possible_fall_candidate_at = now
        else:
            self._possible_fall_candidate_at = None

    def _cooldown_active(self, trigger_id: str, now: float) -> bool:
        """Return whether one trigger is still on cooldown."""

        last_at = self._last_triggered_at.get(trigger_id)
        if last_at is None:
            return False
        return (now - last_at) < self._cooldowns[trigger_id]

    def _evaluate_candidate(
        self,
        *,
        trigger_id: str,
        observed_at: float,
        priority: SocialTriggerPriority,
        threshold: float,
        reason: str,
        prompt: str,
        evidence: tuple[TriggerScoreEvidence, ...],
        blocked_reason: str | None = None,
    ) -> SocialTriggerEvaluation:
        """Build one scored trigger evaluation from evidence."""

        score_card = weighted_trigger_score(
            threshold=_normalize_unit_interval(threshold, default=1.0),
            evidence=self._normalize_evidence(evidence),
        )
        return SocialTriggerEvaluation(
            trigger_id=trigger_id,
            prompt=prompt,
            reason=reason,
            observed_at=observed_at,
            priority=priority,
            score=score_card.score,
            threshold=score_card.threshold,
            evidence=score_card.evidence,
            passed=(blocked_reason is None and score_card.passed),
            blocked_reason=blocked_reason,
        )

    def _with_name(self, *, base: str, with_name: str) -> str:
        """Format one prompt variant with the configured display name."""

        if self.user_name is None:
            return base
        return with_name.format(name=self.user_name)

    def _conjunction_since(self, *starts: float | None) -> float | None:
        """Return the shared activation start when all inputs are active."""

        active_starts = [start for start in starts if start is not None]
        if len(active_starts) != len(starts):
            return None
        return max(active_starts)

    def _next_since(self, active: bool, current: float | None, now: float) -> float | None:
        """Advance one hold start timestamp when its signal remains active."""

        if not active:
            return None
        if current is None:
            return now
        return current

    def _hold_detail(self, since: float | None, now: float, target_s: float) -> str:
        """Render one human-readable hold-progress detail string."""

        if since is None:
            return f"active_for=0.0s target={target_s:.1f}s"
        return f"active_for={max(0.0, now - since):.1f}s target={target_s:.1f}s"

    def _quiet_signal_active(self, audio: SocialAudioObservation) -> bool:
        """Return whether current audio conditions should count as quiet."""

        if audio.assistant_output_active is True:
            return False
        if audio.speech_overlap_likely is True:
            return False
        if audio.barge_in_detected is True:
            return False
        if audio.speech_detected is True:
            return False
        if audio.room_quiet is True and audio.background_media_likely is not True:
            return True
        if (
            audio.speech_detected is False
            and audio.background_media_likely is not True
            and audio.non_speech_audio_likely is not True
        ):
            return True
        if (
            audio.recent_speech_age_s is not None
            and audio.recent_speech_age_s >= self.thresholds.quiet_after_speech_age_s
            and audio.background_media_likely is not True
            and audio.non_speech_audio_likely is not True
        ):
            return True
        return False

    def _social_prompt_channel_available(self, audio: SocialAudioObservation) -> bool:
        """Return whether non-urgent social prompting should currently be allowed."""

        return (
            audio.mute_active is not True
            and audio.assistant_output_active is not True
            and audio.speech_overlap_likely is not True
            and audio.barge_in_detected is not True
        )

    def _social_channel_detail(self, audio: SocialAudioObservation) -> str:
        """Render current non-urgent social channel conditions."""

        return (
            f"mute={audio.mute_active} assistant_output={audio.assistant_output_active} "
            f"overlap={audio.speech_overlap_likely} barge_in={audio.barge_in_detected}"
        )

    def _social_trigger_block_reason(
        self,
        trigger_id: str,
        audio: SocialAudioObservation,
    ) -> str | None:
        """Return a runtime block reason for non-urgent social triggers."""

        if trigger_id not in _SOCIAL_TRIGGERS:
            return None
        if audio.mute_active is True:
            return "mute_active"
        if audio.assistant_output_active is True:
            return "assistant_output_active"
        if audio.speech_overlap_likely is True:
            return "speech_overlap_likely"
        if audio.barge_in_detected is True:
            return "barge_in_detected"
        return None

    def _attention_signal_score(self, vision: SocialVisionObservation) -> float:
        """Return one bounded score for visual attention reliability."""

        if vision.visual_attention_score is not None:
            return vision.visual_attention_score
        return 1.0 if vision.looking_toward_device else 0.0

    def _pose_reliability_score(
        self,
        vision: SocialVisionObservation,
        *,
        expected: SocialBodyPose | None = None,
        expected_floor_like: bool = False,
    ) -> float:
        """Return one pose-confidence score for the relevant expected state."""

        pose_confidence = vision.pose_confidence
        if expected is not None and vision.body_pose != expected:
            return 0.0
        if expected_floor_like and not _is_floor_like_pose(vision.body_pose):
            return 0.0
        if pose_confidence is None:
            return 1.0 if vision.person_visible else 0.0
        return pose_confidence

    def _safety_reliability_score(self, vision: SocialVisionObservation) -> float:
        """Return one bounded reliability score for safety-relevant vision usage."""

        if vision.person_visible:
            pose_score = 1.0 if vision.pose_confidence is None else vision.pose_confidence
            motion_score = 1.0 if vision.motion_confidence is None else vision.motion_confidence
            return (pose_score + motion_score) / 2.0
        if self._possible_fall_loss_candidate_at is not None:
            return 1.0
        return 0.0

    def _safety_reliability_detail(self, vision: SocialVisionObservation) -> str:
        """Render one reliability detail string for safety-relevant candidates."""

        return (
            f"person_visible={vision.person_visible} pose_confidence={self._ratio_or_na(vision.pose_confidence)} "
            f"motion_confidence={self._ratio_or_na(vision.motion_confidence)}"
        )

    def _ratio_or_na(self, value: float | None) -> str:
        """Format one optional ratio."""

        if value is None:
            return "n/a"
        return f"{value:.2f}"

    def _fell_out_of_view_after_fall_like_presence(self, now: float, *, visible_duration: float | None) -> bool:
        """Return whether recent visibility loss looks fall-like."""

        if self._possible_fall_candidate_at is not None:
            return True
        if self._consecutive_visible_inspected_count < 2:
            return False
        if visible_duration is None or visible_duration < self._fall_visibility_loss_arming_s():
            return False
        if self._current_pose == SocialBodyPose.SLUMPED and self._last_slumped_at is not None:
            return (
                (now - self._last_slumped_at) <= self.thresholds.fall_transition_window_s
                and self._slumped_hold_before_visibility_loss(now) >= self._fall_slumped_visibility_loss_arming_s()
            )
        if self._current_pose == SocialBodyPose.UNKNOWN and self._last_slumped_at is not None:
            return (
                (now - self._last_slumped_at) <= self.thresholds.fall_transition_window_s
                and self._slumped_hold_before_visibility_loss(now) >= self._fall_slumped_visibility_loss_arming_s()
            )
        return False

    def _possible_fall_transition_anchor(self) -> float | None:
        """Return the newest recorded fall-transition anchor."""

        anchors = [
            anchor
            for anchor in (
                self._possible_fall_candidate_at,
                self._possible_fall_loss_candidate_at,
            )
            if anchor is not None
        ]
        if not anchors:
            return None
        return max(anchors)

    def _possible_fall_post_transition_since(
        self,
        now: float,
        signal_since: float | None,
    ) -> float | None:
        """Clamp one hold start to the relevant fall-transition anchor."""

        transition_anchor = self._possible_fall_transition_anchor()
        if transition_anchor is None or signal_since is None:
            return None
        if signal_since > now:
            return now
        return max(signal_since, transition_anchor)

    def _possible_fall_transition_signal(self) -> float:
        """Return the weighted signal strength of the current fall transition."""

        if self._possible_fall_candidate_at is not None:
            return 1.0
        if self._possible_fall_loss_candidate_at is None:
            return 0.0
        if self._possible_fall_loss_pose == SocialBodyPose.SLUMPED:
            return 0.9
        if self._possible_fall_loss_pose in {SocialBodyPose.UPRIGHT, SocialBodyPose.SEATED}:
            return 0.75
        return 0.55

    def _possible_fall_low_or_missing_hold(self, now: float) -> float:
        """Return the stronger of floor hold and visibility-loss hold."""

        floor_hold = hold_progress(now, self._floor_since, self.thresholds.possible_fall_stillness_s)
        missing_hold = hold_progress(
            now,
            self._possible_fall_loss_candidate_at,
            self.thresholds.possible_fall_visibility_loss_hold_s,
        )
        return max(floor_hold, missing_hold)

    def _possible_fall_low_or_missing_detail(self, now: float) -> str:
        """Render floor and visibility-loss hold details."""

        floor_detail = self._hold_detail(self._floor_since, now, self.thresholds.possible_fall_stillness_s)
        missing_detail = self._hold_detail(
            self._possible_fall_loss_candidate_at,
            now,
            self.thresholds.possible_fall_visibility_loss_hold_s,
        )
        return f"floor={floor_detail}; missing={missing_detail}"

    def _possible_fall_visibility_loss_hold_complete(self, now: float) -> bool:
        """Return whether the visibility-loss hold has fully armed."""

        if self._possible_fall_loss_candidate_at is None:
            return True
        return (
            hold_progress(
                now,
                self._possible_fall_loss_candidate_at,
                self.thresholds.possible_fall_visibility_loss_hold_s,
            )
            >= 1.0
        )

    def _possible_fall_transition_detail(self, now: float) -> str:
        """Render one detail string for the active fall-transition signal."""

        if self._possible_fall_candidate_at is not None:
            return f"upright_to_floor_age={self._seconds(now - self._possible_fall_candidate_at)}"
        if self._possible_fall_loss_candidate_at is not None:
            return (
                f"{self._possible_fall_loss_pose.value}_to_visibility_loss_age="
                f"{self._seconds(now - self._possible_fall_loss_candidate_at)} "
                f"visible_for={self._seconds(self._possible_fall_loss_visible_duration_s)}"
            )
        return "no recent fall-like transition"

    def _fall_visibility_loss_arming_s(self) -> float:
        """Return the visibility-loss arming threshold in seconds."""

        return max(0.0, self.thresholds.possible_fall_visibility_loss_arming_s)

    def _fall_slumped_visibility_loss_arming_s(self) -> float:
        """Return the slumped-loss arming threshold in seconds."""

        return max(0.0, self.thresholds.possible_fall_slumped_visibility_loss_arming_s)

    def _slumped_hold_before_visibility_loss(self, now: float) -> float:
        """Return how long the person was slumped before visibility loss."""

        if self._slumped_since is None:
            return 0.0
        return max(0.0, now - self._slumped_since)

    def _update_confirmed_visibility_state(self, *, inspected: bool, person_visible: bool) -> None:
        """Update the bounded consecutive-visible counter."""

        if not inspected:
            return
        if person_visible:
            self._consecutive_visible_inspected_count = min(
                2,
                self._consecutive_visible_inspected_count + 1,
            )
            return
        self._consecutive_visible_inspected_count = 0

    def _seconds(self, value: float | None) -> str:
        """Format one optional duration in seconds."""

        if value is None:
            return "n/a"
        return f"{max(0.0, value):.1f}s"

    def _normalize_audio(self, audio: object) -> SocialAudioObservation:
        """Normalize one audio payload to the strict internal type."""

        if not isinstance(audio, SocialAudioObservation):
            return SocialAudioObservation()
        return SocialAudioObservation(
            speech_detected=_coerce_optional_bool(audio.speech_detected),
            distress_detected=_coerce_optional_bool(audio.distress_detected),
            room_quiet=_coerce_optional_bool(getattr(audio, "room_quiet", None)),
            recent_speech_age_s=_coerce_recent_age(getattr(audio, "recent_speech_age_s", None)),
            assistant_output_active=_coerce_optional_bool(getattr(audio, "assistant_output_active", None)),
            azimuth_deg=_coerce_optional_azimuth(getattr(audio, "azimuth_deg", None)),
            direction_confidence=_coerce_optional_ratio(getattr(audio, "direction_confidence", None)),
            device_runtime_mode=_coerce_optional_text(getattr(audio, "device_runtime_mode", None), limit=64),
            signal_source=_coerce_optional_text(getattr(audio, "signal_source", None), limit=64),
            host_control_ready=_coerce_optional_bool(getattr(audio, "host_control_ready", None)),
            transport_reason=_coerce_optional_text(getattr(audio, "transport_reason", None), limit=120),
            non_speech_audio_likely=_coerce_optional_bool(getattr(audio, "non_speech_audio_likely", None)),
            background_media_likely=_coerce_optional_bool(getattr(audio, "background_media_likely", None)),
            speech_overlap_likely=_coerce_optional_bool(getattr(audio, "speech_overlap_likely", None)),
            barge_in_detected=_coerce_optional_bool(getattr(audio, "barge_in_detected", None)),
            mute_active=_coerce_optional_bool(getattr(audio, "mute_active", None)),
        )

    def _normalize_vision(
        self,
        vision: object,
        *,
        inspected: bool,
        now: float,
    ) -> SocialVisionObservation:
        """Normalize one vision payload to the strict internal type."""

        if not inspected or not isinstance(vision, SocialVisionObservation):
            return SocialVisionObservation()

        camera_online = _coerce_bool(getattr(vision, "camera_online", False))
        camera_ready = _coerce_bool(getattr(vision, "camera_ready", False))
        camera_ai_ready = _coerce_bool(getattr(vision, "camera_ai_ready", False))
        camera_error = _coerce_bounded_text(getattr(vision, "camera_error", None))
        last_camera_frame_at = _coerce_timestamp(getattr(vision, "last_camera_frame_at", None))
        last_camera_health_change_at = _coerce_timestamp(getattr(vision, "last_camera_health_change_at", None))

        vision_fresh = (
            last_camera_frame_at is None
            or 0.0 <= (now - last_camera_frame_at) <= self.thresholds.vision_stale_after_s
        )
        explicit_camera_outage = (
            camera_error is not None
            or (
                last_camera_health_change_at is not None
                and (not camera_online or not camera_ready or not camera_ai_ready)
            )
            or (
                last_camera_frame_at is not None
                and (now - last_camera_frame_at) > self.thresholds.vision_stale_after_s
            )
        )
        camera_usable = vision_fresh and not explicit_camera_outage

        visible_persons = _coerce_visible_persons(getattr(vision, "visible_persons", ()))
        primary_person_box = _coerce_spatial_box(getattr(vision, "primary_person_box", None))
        if primary_person_box is None and visible_persons:
            primary_person_box = visible_persons[0].box

        raw_person_count = _coerce_non_negative_int(getattr(vision, "person_count", 0), default=0)
        redundant_visibility = (
            raw_person_count > 0
            or primary_person_box is not None
            or len(visible_persons) > 0
        )
        person_visible = camera_usable and (
            _coerce_bool(vision.person_visible) or redundant_visibility
        )

        person_count = 0
        if person_visible:
            person_count = max(1, raw_person_count, len(visible_persons), 1)

        visual_attention_score = _coerce_optional_ratio(getattr(vision, "visual_attention_score", None))
        explicit_looking = _coerce_bool(getattr(vision, "looking_toward_device", False))
        looking_toward_device = person_visible and (
            explicit_looking
            or (
                visual_attention_score is not None
                and visual_attention_score >= self.thresholds.min_visual_attention_score
            )
        )

        pose_confidence = _coerce_optional_ratio(getattr(vision, "pose_confidence", None))
        body_pose = _coerce_body_pose(getattr(vision, "body_pose", SocialBodyPose.UNKNOWN))
        if (
            not person_visible
            or (
                body_pose != SocialBodyPose.UNKNOWN
                and pose_confidence is not None
                and pose_confidence < self.thresholds.min_pose_confidence
            )
        ):
            body_pose = SocialBodyPose.UNKNOWN

        motion_confidence = _coerce_optional_ratio(getattr(vision, "motion_confidence", None))
        motion_state = _coerce_motion_state(getattr(vision, "motion_state", SocialMotionState.UNKNOWN))
        if (
            not person_visible
            or (
                motion_state != SocialMotionState.UNKNOWN
                and motion_confidence is not None
                and motion_confidence < self.thresholds.min_motion_confidence
            )
        ):
            motion_state = SocialMotionState.UNKNOWN

        primary_person_center_x = (
            primary_person_box.center_x
            if primary_person_box is not None
            else _coerce_optional_ratio(getattr(vision, "primary_person_center_x", None))
        )
        primary_person_center_y = (
            primary_person_box.center_y
            if primary_person_box is not None
            else _coerce_optional_ratio(getattr(vision, "primary_person_center_y", None))
        )

        primary_person_zone = _coerce_person_zone(getattr(vision, "primary_person_zone", SocialPersonZone.UNKNOWN))
        if primary_person_zone == SocialPersonZone.UNKNOWN:
            if primary_person_box is not None:
                primary_person_zone = _zone_from_center_x(primary_person_box.center_x)
            elif visible_persons:
                primary_person_zone = visible_persons[0].zone

        raw_coarse_arm_gesture = getattr(
            vision,
            "coarse_arm_gesture",
            SocialGestureEvent.NONE,
        )
        raw_gesture_event = getattr(
            vision,
            "gesture_event",
            raw_coarse_arm_gesture,
        )

        if not camera_usable:
            person_visible = False
            person_count = 0
            primary_person_box = None
            primary_person_center_x = None
            primary_person_center_y = None
            primary_person_zone = SocialPersonZone.UNKNOWN
            visible_persons = ()
            looking_toward_device = False
            body_pose = SocialBodyPose.UNKNOWN
            motion_state = SocialMotionState.UNKNOWN

        return SocialVisionObservation(
            person_visible=person_visible,
            person_count=person_count,
            person_recently_visible=_coerce_optional_bool(getattr(vision, "person_recently_visible", None)),
            person_appeared_at=_coerce_timestamp(getattr(vision, "person_appeared_at", None)),
            person_disappeared_at=_coerce_timestamp(getattr(vision, "person_disappeared_at", None)),
            primary_person_zone=primary_person_zone if person_visible else SocialPersonZone.UNKNOWN,
            primary_person_box=primary_person_box,
            visible_persons=visible_persons if person_visible else (),
            primary_person_center_x=primary_person_center_x if person_visible else None,
            primary_person_center_y=primary_person_center_y if person_visible else None,
            looking_toward_device=looking_toward_device,
            looking_signal_state=_coerce_optional_text(getattr(vision, "looking_signal_state", None), limit=32),
            looking_signal_source=_coerce_optional_text(getattr(vision, "looking_signal_source", None), limit=32),
            person_near_device=_coerce_optional_bool(getattr(vision, "person_near_device", None)),
            engaged_with_device=_coerce_optional_bool(getattr(vision, "engaged_with_device", None)),
            visual_attention_score=visual_attention_score,
            body_pose=body_pose,
            pose_confidence=pose_confidence,
            body_state_changed_at=_coerce_timestamp(getattr(vision, "body_state_changed_at", None)),
            motion_state=motion_state,
            motion_confidence=motion_confidence,
            motion_state_changed_at=_coerce_timestamp(getattr(vision, "motion_state_changed_at", None)),
            smiling=person_visible and _coerce_bool(getattr(vision, "smiling", False)),
            hand_or_object_near_camera=person_visible and _coerce_bool(getattr(vision, "hand_or_object_near_camera", False)),
            showing_intent_likely=_coerce_optional_bool(getattr(vision, "showing_intent_likely", None)),
            showing_intent_started_at=_coerce_timestamp(getattr(vision, "showing_intent_started_at", None)),
            coarse_arm_gesture=_coerce_gesture_event(raw_coarse_arm_gesture),
            gesture_event=_coerce_gesture_event(raw_gesture_event),
            gesture_confidence=_coerce_optional_ratio(getattr(vision, "gesture_confidence", None)),
            fine_hand_gesture=_coerce_fine_hand_gesture(
                getattr(vision, "fine_hand_gesture", SocialFineHandGesture.NONE)
            ),
            fine_hand_gesture_confidence=_coerce_optional_ratio(
                getattr(vision, "fine_hand_gesture_confidence", None)
            ),
            objects=_coerce_detected_objects(getattr(vision, "objects", ())),
            camera_online=camera_online,
            camera_ready=camera_ready,
            camera_ai_ready=camera_ai_ready,
            camera_error=camera_error,
            last_camera_frame_at=last_camera_frame_at,
            last_camera_health_change_at=last_camera_health_change_at,
            perception_stream=getattr(vision, "perception_stream", None),
        )

    def _normalize_evidence(
        self,
        evidence: tuple[TriggerScoreEvidence, ...],
    ) -> tuple[TriggerScoreEvidence, ...]:
        """Normalize evidence weights into one stable tuple."""

        sanitized: list[tuple[str, float, float, str]] = []
        total_weight = 0.0
        for item in evidence:
            key = str(getattr(item, "key", ""))
            value = _normalize_unit_interval(getattr(item, "value", 0.0), default=0.0)
            weight = _normalize_non_negative_float(getattr(item, "weight", 0.0), default=0.0)
            detail = _coerce_bounded_text(getattr(item, "detail", ""), max_length=240) or ""
            sanitized.append((key, value, weight, detail))
            total_weight += weight

        if not sanitized:
            return ()

        normalized_weight = 1.0 / len(sanitized) if total_weight <= 0.0 else None
        return tuple(
            TriggerScoreEvidence(
                key=key,
                value=value,
                weight=(normalized_weight if normalized_weight is not None else weight / total_weight),
                detail=detail,
            )
            for key, value, weight, detail in sanitized
        )

    def _build_cooldowns(self, overrides: Mapping[str, object] | None) -> dict[str, float]:
        """Build one normalized cooldown dictionary."""

        cooldowns = dict(_DEFAULT_COOLDOWNS)
        if overrides is None:
            return cooldowns
        for trigger_id, default in cooldowns.items():
            if trigger_id in overrides:
                cooldowns[trigger_id] = _normalize_positive_float(overrides[trigger_id], default=default)
        return cooldowns

__all__ = [
    "SocialAudioObservation",
    "SocialBodyPose",
    "SocialDetectedObject",
    "SocialFineHandGesture",
    "SocialGestureEvent",
    "SocialMotionState",
    "SocialObservation",
    "SocialPersonZone",
    "SocialSpatialBox",
    "SocialTriggerDecision",
    "SocialTriggerEngine",
    "SocialTriggerEvaluation",
    "SocialTriggerPriority",
    "SocialTriggerThresholds",
    "SocialVisionObservation",
    "SocialVisiblePerson",
]
