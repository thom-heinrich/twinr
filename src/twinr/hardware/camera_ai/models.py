"""Define the stable data contracts for Twinr's local AI-camera stack."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
import math
import re
from typing import TypeVar

# AUDIT-FIX(#6): Bound and sanitize contract tokens so malformed model output cannot inject control chars or arbitrary punctuation.
_MAX_LABEL_LENGTH = 64
_MAX_TEXT_LENGTH = 256
_MAX_MODEL_LENGTH = 64
_SAFE_TOKEN_RE = re.compile(r"[^a-z0-9_.+-]+")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]+")
_TRUE_STRINGS = frozenset({"1", "true", "t", "yes", "y", "on"})
_FALSE_STRINGS = frozenset({"0", "false", "f", "no", "n", "off"})
_StrEnumT = TypeVar("_StrEnumT", bound=StrEnum)


class AICameraZone(StrEnum):
    """Describe one coarse horizontal zone."""

    UNKNOWN = "unknown"
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class AICameraBodyPose(StrEnum):
    """Describe one coarse body pose."""

    UNKNOWN = "unknown"
    UPRIGHT = "upright"
    SEATED = "seated"
    SLUMPED = "slumped"
    LYING_LOW = "lying_low"
    FLOOR = "floor"


class AICameraMotionState(StrEnum):
    """Describe one coarse motion class derived from recent person-box deltas."""

    UNKNOWN = "unknown"
    STILL = "still"
    WALKING = "walking"
    APPROACHING = "approaching"
    LEAVING = "leaving"


class AICameraGestureEvent(StrEnum):
    """Describe the bounded coarse-arm gesture vocabulary."""

    NONE = "none"
    WAVE = "wave"
    STOP = "stop"
    DISMISS = "dismiss"
    CONFIRM = "confirm"
    ARMS_CROSSED = "arms_crossed"
    TWO_HAND_DISMISS = "two_hand_dismiss"
    TIMEOUT_T = "timeout_t"
    UNKNOWN = "unknown"


class AICameraFineHandGesture(StrEnum):
    """Describe the bounded fine-hand gesture vocabulary."""

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
class AICameraVisiblePerson:
    """Describe one visible person anchor before higher-level tracking."""

    box: AICameraBox | None
    zone: AICameraZone = AICameraZone.UNKNOWN
    confidence: float = 0.0
    attention_hint_score: float | None = None

    def __post_init__(self) -> None:
        """Normalize person-anchor metadata into stable bounded values."""

        object.__setattr__(self, "box", _coerce_box(self.box))
        object.__setattr__(self, "zone", _coerce_zone(self.zone))
        object.__setattr__(self, "confidence", _clamp_ratio(self.confidence, default=0.0))
        object.__setattr__(
            self,
            "attention_hint_score",
            _coerce_optional_ratio(self.attention_hint_score),
        )


@dataclass(frozen=True, slots=True)
class AICameraBox:
    """Describe one normalized bounding box."""

    top: float
    left: float
    bottom: float
    right: float

    def __post_init__(self) -> None:
        """Clamp and order the normalized edges."""

        top = _clamp_ratio(self.top, default=0.0)
        left = _clamp_ratio(self.left, default=0.0)
        bottom = _clamp_ratio(self.bottom, default=top)
        right = _clamp_ratio(self.right, default=left)
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
    def width(self) -> float:
        """Return the normalized width."""

        return max(0.0, self.right - self.left)

    @property
    def height(self) -> float:
        """Return the normalized height."""

        return max(0.0, self.bottom - self.top)

    @property
    def area(self) -> float:
        """Return the normalized area."""

        return self.width * self.height


@dataclass(frozen=True, slots=True)
class AICameraObjectDetection:
    """Describe one local object detection before surface stabilization."""

    label: str
    confidence: float
    zone: AICameraZone
    box: AICameraBox | None = None

    def __post_init__(self) -> None:
        """Normalize detection metadata into inspectable values."""

        label = _normalize_label(self.label)
        object.__setattr__(self, "label", label if label else "unknown")
        object.__setattr__(self, "confidence", _clamp_ratio(self.confidence, default=0.0))
        object.__setattr__(self, "zone", _coerce_zone(self.zone))
        # AUDIT-FIX(#3): Normalize foreign box payloads at the detection boundary instead of storing raw dict/list objects.
        object.__setattr__(self, "box", _coerce_box(self.box))


@dataclass(frozen=True, slots=True)
class AICameraObservation:
    """Describe one bounded AI-camera observation tick."""

    observed_at: float
    camera_online: bool
    camera_ready: bool
    camera_ai_ready: bool
    camera_error: str | None = None
    last_camera_frame_at: float | None = None
    last_camera_health_change_at: float | None = None
    person_count: int = 0
    primary_person_box: AICameraBox | None = None
    primary_person_zone: AICameraZone = AICameraZone.UNKNOWN
    visible_persons: tuple[AICameraVisiblePerson, ...] = ()
    looking_toward_device: bool | None = None
    looking_signal_state: str | None = None
    looking_signal_source: str | None = None
    person_near_device: bool | None = None
    engaged_with_device: bool | None = None
    visual_attention_score: float | None = None
    body_pose: AICameraBodyPose = AICameraBodyPose.UNKNOWN
    pose_confidence: float | None = None
    motion_state: AICameraMotionState = AICameraMotionState.UNKNOWN
    motion_confidence: float | None = None
    hand_or_object_near_camera: bool = False
    showing_intent_likely: bool | None = None
    gesture_event: AICameraGestureEvent = AICameraGestureEvent.NONE
    gesture_confidence: float | None = None
    fine_hand_gesture: AICameraFineHandGesture = AICameraFineHandGesture.NONE
    fine_hand_gesture_confidence: float | None = None
    objects: tuple[AICameraObjectDetection, ...] = ()
    model: str = "local-imx500"

    def __post_init__(self) -> None:
        """Normalize observation metadata into safe, stable contract values."""

        # AUDIT-FIX(#2): Coerce nested payloads and keep person presence internally consistent when upstream emits raw dicts/lists.
        primary_person_box = _coerce_box(self.primary_person_box)
        objects = _coerce_objects(self.objects)
        visible_persons = _coerce_visible_persons(self.visible_persons)
        person_count = _coerce_non_negative_int(self.person_count, default=0)
        if primary_person_box is not None and person_count < 1:
            person_count = 1
        if person_count < 1 and any(obj.label == "person" and obj.confidence > 0.0 for obj in objects):
            person_count = 1
        if visible_persons and person_count < len(visible_persons):
            person_count = len(visible_persons)
        if primary_person_box is None and visible_persons:
            primary_person_box = visible_persons[0].box
        if primary_person_box is not None and not visible_persons:
            visible_persons = (
                AICameraVisiblePerson(
                    box=primary_person_box,
                    zone=self.primary_person_zone,
                    confidence=1.0,
                ),
            )

        # AUDIT-FIX(#1): Parse readiness booleans strictly and collapse impossible state combinations.
        camera_online = _coerce_bool(self.camera_online, default=False)
        camera_ready = _coerce_bool(self.camera_ready, default=False) and camera_online
        camera_ai_ready = _coerce_bool(self.camera_ai_ready, default=False) and camera_ready

        # AUDIT-FIX(#4): Sanitize timestamps and bounded scores before downstream TTL/ranking logic sees them.
        observed_at = _sanitize_timestamp(self.observed_at, default=0.0)
        if observed_at is None:
            observed_at = 0.0

        object.__setattr__(self, "observed_at", observed_at)
        object.__setattr__(self, "camera_online", camera_online)
        object.__setattr__(self, "camera_ready", camera_ready)
        object.__setattr__(self, "camera_ai_ready", camera_ai_ready)

        # AUDIT-FIX(#5): Bound free-text fields so they remain safe for logs, TTS, and UI surfaces.
        object.__setattr__(
            self,
            "camera_error",
            _normalize_text(self.camera_error, max_length=_MAX_TEXT_LENGTH, default=None),
        )
        object.__setattr__(self, "last_camera_frame_at", _sanitize_timestamp(self.last_camera_frame_at, default=None))
        object.__setattr__(
            self,
            "last_camera_health_change_at",
            _sanitize_timestamp(self.last_camera_health_change_at, default=None),
        )
        object.__setattr__(self, "person_count", person_count)
        object.__setattr__(self, "primary_person_box", primary_person_box)
        object.__setattr__(self, "primary_person_zone", _coerce_zone(self.primary_person_zone))
        object.__setattr__(self, "visible_persons", visible_persons)
        object.__setattr__(self, "looking_toward_device", _coerce_optional_bool(self.looking_toward_device))
        object.__setattr__(
            self,
            "looking_signal_state",
            _normalize_token(self.looking_signal_state, max_length=32) or None,
        )
        object.__setattr__(
            self,
            "looking_signal_source",
            _normalize_token(self.looking_signal_source, max_length=48) or None,
        )
        object.__setattr__(self, "person_near_device", _coerce_optional_bool(self.person_near_device))
        object.__setattr__(self, "engaged_with_device", _coerce_optional_bool(self.engaged_with_device))
        object.__setattr__(self, "visual_attention_score", _coerce_optional_ratio(self.visual_attention_score))
        object.__setattr__(self, "body_pose", _coerce_body_pose(self.body_pose))
        object.__setattr__(self, "pose_confidence", _coerce_optional_ratio(self.pose_confidence))
        object.__setattr__(self, "motion_state", _coerce_motion_state(self.motion_state))
        object.__setattr__(self, "motion_confidence", _coerce_optional_ratio(self.motion_confidence))
        object.__setattr__(self, "hand_or_object_near_camera", _coerce_bool(self.hand_or_object_near_camera, default=False))
        object.__setattr__(self, "showing_intent_likely", _coerce_optional_bool(self.showing_intent_likely))
        object.__setattr__(self, "gesture_event", _coerce_gesture_event(self.gesture_event))
        object.__setattr__(self, "gesture_confidence", _coerce_optional_ratio(self.gesture_confidence))
        object.__setattr__(self, "fine_hand_gesture", _coerce_fine_hand_gesture(self.fine_hand_gesture))
        object.__setattr__(
            self,
            "fine_hand_gesture_confidence",
            _coerce_optional_ratio(self.fine_hand_gesture_confidence),
        )
        object.__setattr__(self, "objects", objects)
        object.__setattr__(self, "model", _normalize_token(self.model, max_length=_MAX_MODEL_LENGTH) or "local-imx500")

    @property
    def primary_person_center_x(self) -> float | None:
        """Return the primary-person horizontal center when available."""

        if self.primary_person_box is None:
            return None
        return self.primary_person_box.center_x

    @property
    def primary_person_center_y(self) -> float | None:
        """Return the primary-person vertical center when available."""

        if self.primary_person_box is None:
            return None
        return self.primary_person_box.center_y


# AUDIT-FIX(#6): Restrict labels to a predictable token alphabet for safe logs, metrics, and serialization surfaces.
def _normalize_token(value: object, *, max_length: int) -> str:
    """Normalize one token-like value into a safe bounded identifier."""

    if value is None or isinstance(value, bool):
        return ""
    text = _CONTROL_CHAR_RE.sub("", str(value))
    text = "_".join(text.strip().lower().split())
    text = _SAFE_TOKEN_RE.sub("_", text)
    text = re.sub(r"_+", "_", text).strip("_.-")
    return text[:max_length]


def _normalize_label(value: object) -> str:
    """Normalize one object label to an inspectable token."""

    return _normalize_token(value, max_length=_MAX_LABEL_LENGTH)


# AUDIT-FIX(#5): Collapse multiline/control-character text before it reaches logs, UI, or speech surfaces.
def _normalize_text(value: object, *, max_length: int, default: str | None = None) -> str | None:
    """Normalize one free-text value into a bounded single-line string."""

    if value is None or isinstance(value, bool):
        return default
    text = _CONTROL_CHAR_RE.sub(" ", str(value))
    text = " ".join(text.split())
    text = text[:max_length].strip()
    return text if text else default


# AUDIT-FIX(#2): Normalize token-like enum payloads so callers cannot inject raw strings outside the declared vocabulary.
def _coerce_str_enum(value: object, *, enum_type: type[_StrEnumT], default: _StrEnumT) -> _StrEnumT:
    """Coerce one token-like payload into the requested ``StrEnum`` value."""

    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(_normalize_token(value, max_length=_MAX_LABEL_LENGTH))
    except ValueError:
        return default


def _coerce_zone(value: object) -> AICameraZone:
    """Coerce one zone-like payload to ``AICameraZone``."""

    return _coerce_str_enum(value, enum_type=AICameraZone, default=AICameraZone.UNKNOWN)


def _coerce_body_pose(value: object) -> AICameraBodyPose:
    """Coerce one pose-like payload to ``AICameraBodyPose``."""

    return _coerce_str_enum(value, enum_type=AICameraBodyPose, default=AICameraBodyPose.UNKNOWN)


def _coerce_motion_state(value: object) -> AICameraMotionState:
    """Coerce one motion-like payload to ``AICameraMotionState``."""

    return _coerce_str_enum(value, enum_type=AICameraMotionState, default=AICameraMotionState.UNKNOWN)


def _coerce_gesture_event(value: object) -> AICameraGestureEvent:
    """Coerce one gesture-like payload to ``AICameraGestureEvent``."""

    return _coerce_str_enum(value, enum_type=AICameraGestureEvent, default=AICameraGestureEvent.UNKNOWN)


def _coerce_fine_hand_gesture(value: object) -> AICameraFineHandGesture:
    """Coerce one fine-hand gesture-like payload to ``AICameraFineHandGesture``."""

    return _coerce_str_enum(value, enum_type=AICameraFineHandGesture, default=AICameraFineHandGesture.UNKNOWN)


# AUDIT-FIX(#4): Reject non-finite and boolean numeric inputs before they enter timestamps or confidence scores.
def _coerce_finite_float(value: object, *, default: float | None) -> float | None:
    """Coerce one object to a finite float, or return ``default``."""

    if value is None or isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _sanitize_timestamp(value: object, *, default: float | None) -> float | None:
    """Coerce one timestamp-like value to a finite non-negative float."""

    number = _coerce_finite_float(value, default=default)
    if number is None:
        return default
    if number < 0.0:
        return default
    return number


def _coerce_optional_ratio(value: object) -> float | None:
    """Coerce one optional ratio-like value into ``[0.0, 1.0]`` or ``None``."""

    number = _coerce_finite_float(value, default=None)
    if number is None:
        return None
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number


# AUDIT-FIX(#1): Parse booleans strictly so strings like "false" no longer evaluate truthy by accident.
def _parse_bool(value: object) -> bool | None:
    """Parse one bool-like value or return ``None`` when the payload is ambiguous."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, float) and math.isfinite(value) and value in (0.0, 1.0):
        return bool(int(value))
    if isinstance(value, str):
        candidate = value.strip().lower()
        if candidate in _TRUE_STRINGS:
            return True
        if candidate in _FALSE_STRINGS:
            return False
    return None


def _coerce_bool(value: object, *, default: bool) -> bool:
    """Coerce one value to ``bool`` using a strict, non-truthy parser."""

    parsed = _parse_bool(value)
    if parsed is None:
        return default
    return parsed


def _coerce_optional_bool(value: object) -> bool | None:
    """Coerce one value to ``bool | None`` using a strict parser."""

    return _parse_bool(value)


# AUDIT-FIX(#2): Clamp malformed count payloads to a safe non-negative integer.
def _coerce_non_negative_int(value: object, *, default: int = 0) -> int:
    """Coerce one count-like value to a non-negative integer."""

    number = _coerce_finite_float(value, default=None)
    if number is None:
        return max(0, default)
    if number < 0.0:
        return 0
    return int(number)


# AUDIT-FIX(#2): Coerce foreign box payloads into a real ``AICameraBox`` so callers never receive raw dict/list values.
def _coerce_box(value: object) -> AICameraBox | None:
    """Coerce one box-like payload to ``AICameraBox``."""

    if value is None:
        return None
    if isinstance(value, AICameraBox):
        return value
    if isinstance(value, Mapping):
        required_keys = ("top", "left", "bottom", "right")
        if not all(key in value for key in required_keys):
            return None
        candidate = AICameraBox(
            top=value["top"],
            left=value["left"],
            bottom=value["bottom"],
            right=value["right"],
        )
        return candidate if candidate.area > 0.0 else None
    if isinstance(value, (list, tuple)) and len(value) == 4:
        top, left, bottom, right = value
        candidate = AICameraBox(top=top, left=left, bottom=bottom, right=right)
        return candidate if candidate.area > 0.0 else None
    if all(hasattr(value, attr) for attr in ("top", "left", "bottom", "right")):
        candidate = AICameraBox(
            top=getattr(value, "top"),
            left=getattr(value, "left"),
            bottom=getattr(value, "bottom"),
            right=getattr(value, "right"),
        )
        return candidate if candidate.area > 0.0 else None
    return None


# AUDIT-FIX(#2): Normalize heterogeneous detection payloads before they become part of the observation contract.
def _coerce_object_detection(value: object) -> AICameraObjectDetection | None:
    """Coerce one detection-like payload to ``AICameraObjectDetection``."""

    if value is None:
        return None
    if isinstance(value, AICameraObjectDetection):
        return value
    if isinstance(value, Mapping):
        if not any(key in value for key in ("label", "confidence", "zone", "box")):
            return None
        return AICameraObjectDetection(
            label=value.get("label"),
            confidence=value.get("confidence"),
            zone=value.get("zone", AICameraZone.UNKNOWN),
            box=value.get("box"),
        )
    if all(hasattr(value, attr) for attr in ("label", "confidence", "zone")):
        return AICameraObjectDetection(
            label=getattr(value, "label"),
            confidence=getattr(value, "confidence"),
            zone=getattr(value, "zone"),
            box=getattr(value, "box", None),
        )
    return None


# AUDIT-FIX(#2): Materialize object collections as an immutable tuple of validated detections.
def _coerce_objects(value: object) -> tuple[AICameraObjectDetection, ...]:
    """Coerce one object-collection payload to a stable tuple of detections."""

    if value is None:
        return ()
    single = _coerce_object_detection(value)
    if single is not None:
        return (single,)
    if isinstance(value, (str, bytes, bytearray)):
        return ()
    if not isinstance(value, (list, tuple)):
        return ()
    detections: list[AICameraObjectDetection] = []
    for item in value:
        detection = _coerce_object_detection(item)
        if detection is not None:
            detections.append(detection)
    return tuple(detections)


def _coerce_visible_person(value: object) -> AICameraVisiblePerson | None:
    """Coerce one person-like payload to ``AICameraVisiblePerson``."""

    if value is None:
        return None
    if isinstance(value, AICameraVisiblePerson):
        return value
    if isinstance(value, Mapping):
        if not any(key in value for key in ("box", "zone", "confidence")):
            return None
        return AICameraVisiblePerson(
            box=value.get("box"),
            zone=value.get("zone", AICameraZone.UNKNOWN),
            confidence=value.get("confidence", 0.0),
            attention_hint_score=value.get("attention_hint_score"),
        )
    if all(hasattr(value, attr) for attr in ("box", "zone", "confidence")):
        return AICameraVisiblePerson(
            box=getattr(value, "box"),
            zone=getattr(value, "zone"),
            confidence=getattr(value, "confidence"),
            attention_hint_score=getattr(value, "attention_hint_score", None),
        )
    box = _coerce_box(value)
    if box is None:
        return None
    return AICameraVisiblePerson(box=box, zone=AICameraZone.UNKNOWN, confidence=1.0)


def _coerce_visible_persons(value: object) -> tuple[AICameraVisiblePerson, ...]:
    """Materialize visible-person anchors as a stable immutable tuple."""

    if value is None:
        return ()
    single = _coerce_visible_person(value)
    if single is not None:
        return (single,)
    if isinstance(value, (str, bytes, bytearray)):
        return ()
    if not isinstance(value, (list, tuple)):
        return ()
    people: list[AICameraVisiblePerson] = []
    for item in value:
        person = _coerce_visible_person(item)
        if person is not None and person.box is not None:
            people.append(person)
    return tuple(people)


# AUDIT-FIX(#7): Reject booleans as numeric ratios so malformed payloads cannot become false 0.0/1.0 confidences or edges.
def _clamp_ratio(value: object, *, default: float) -> float:
    """Clamp one numeric value into ``[0.0, 1.0]``."""

    default_number = _coerce_finite_float(default, default=0.0)
    if default_number is None:
        default_number = 0.0
    if default_number < 0.0:
        default_number = 0.0
    if default_number > 1.0:
        default_number = 1.0
    number = _coerce_finite_float(value, default=default_number)
    if number is None:
        return default_number
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number


__all__ = [
    "AICameraBodyPose",
    "AICameraBox",
    "AICameraFineHandGesture",
    "AICameraGestureEvent",
    "AICameraMotionState",
    "AICameraObjectDetection",
    "AICameraObservation",
    "AICameraVisiblePerson",
    "AICameraZone",
]
