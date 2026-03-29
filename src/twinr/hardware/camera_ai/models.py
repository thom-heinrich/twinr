# CHANGELOG: 2026-03-28
# BUG-1: Fixed inconsistent person state where visible_persons or person detections could imply a primary person box, but primary_person_zone stayed "unknown".
# BUG-2: Fixed impossible timestamp ordering where last_camera_frame_at / last_camera_health_change_at could exceed observed_at and break TTL or freshness logic.
# BUG-3: Fixed missing primary-person derivation from person detections, which previously produced person_count > 0 with no primary_person_box.
# SEC-1: Added hard caps for visible_persons / objects collections and JSON input size to prevent memory/CPU blowups on Raspberry Pi 4 from malformed or hostile payloads.
# IMP-1: Added optional msgspec-powered JSON encode/decode/schema helpers with stdlib fallback, matching 2026 schema-first structured-output practice.
# IMP-2: Added deterministic zone inference from bounding boxes plus broader box-shape support, including normalized xywh and pixel xywh/xyxy mappings with frame dimensions.

"""Define the stable data contracts for Twinr's local AI-camera stack."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields as dataclass_fields
from enum import StrEnum
import json
import math
import re
from typing import TypeVar

try:
    import msgspec
except Exception:  # pragma: no cover - optional dependency
    msgspec = None


AICAMERA_OBSERVATION_CONTRACT_VERSION = "2026-03-28"

# Bounded surface sizes keep malformed or hostile payloads from consuming excessive
# RAM/CPU on a Raspberry Pi 4 while remaining far above realistic in-room counts.
# BREAKING: Oversized visible_persons / objects collections are truncated instead of
# being materialized in full. ``person_count`` still preserves larger upstream counts.
_MAX_VISIBLE_PERSONS = 16
_MAX_OBJECTS = 64
_MAX_JSON_BYTES = 256 * 1024

# Bound and sanitize contract tokens so malformed model output cannot inject control
# chars or arbitrary punctuation.
_MAX_LABEL_LENGTH = 64
_MAX_TEXT_LENGTH = 256
_MAX_MODEL_LENGTH = 64
_SAFE_TOKEN_RE = re.compile(r"[^a-z0-9_.+-]+")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]+")
_MULTI_UNDERSCORE_RE = re.compile(r"_+")
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
class AICameraBox:
    """Describe one normalized bounding box."""

    top: float
    left: float
    bottom: float
    right: float

    @classmethod
    def from_xywh(
        cls,
        x: object,
        y: object,
        width: object,
        height: object,
        *,
        frame_width: object | None = None,
        frame_height: object | None = None,
    ) -> "AICameraBox | None":
        """Build a normalized box from ``x, y, width, height``.

        If ``frame_width`` and ``frame_height`` are omitted, inputs are treated as
        already normalized ratios in ``[0.0, 1.0]``.
        """

        x_number = _coerce_finite_float(x, default=None)
        y_number = _coerce_finite_float(y, default=None)
        width_number = _coerce_finite_float(width, default=None)
        height_number = _coerce_finite_float(height, default=None)
        if None in (x_number, y_number, width_number, height_number):
            return None
        if width_number <= 0.0 or height_number <= 0.0:
            return None

        frame_width_number = _coerce_positive_float(frame_width, default=None)
        frame_height_number = _coerce_positive_float(frame_height, default=None)

        if frame_width_number is None or frame_height_number is None:
            candidate = cls(
                top=y_number,
                left=x_number,
                bottom=y_number + height_number,
                right=x_number + width_number,
            )
        else:
            candidate = cls(
                top=y_number / frame_height_number,
                left=x_number / frame_width_number,
                bottom=(y_number + height_number) / frame_height_number,
                right=(x_number + width_number) / frame_width_number,
            )
        return candidate if candidate.area > 0.0 else None

    @classmethod
    def from_xyxy(
        cls,
        x0: object,
        y0: object,
        x1: object,
        y1: object,
        *,
        frame_width: object | None = None,
        frame_height: object | None = None,
    ) -> "AICameraBox | None":
        """Build a normalized box from ``x0, y0, x1, y1`` corner coordinates.

        If ``frame_width`` and ``frame_height`` are omitted, inputs are treated as
        already normalized ratios in ``[0.0, 1.0]``.
        """

        x0_number = _coerce_finite_float(x0, default=None)
        y0_number = _coerce_finite_float(y0, default=None)
        x1_number = _coerce_finite_float(x1, default=None)
        y1_number = _coerce_finite_float(y1, default=None)
        if None in (x0_number, y0_number, x1_number, y1_number):
            return None

        frame_width_number = _coerce_positive_float(frame_width, default=None)
        frame_height_number = _coerce_positive_float(frame_height, default=None)

        if frame_width_number is None or frame_height_number is None:
            candidate = cls(
                top=y0_number,
                left=x0_number,
                bottom=y1_number,
                right=x1_number,
            )
        else:
            candidate = cls(
                top=y0_number / frame_height_number,
                left=x0_number / frame_width_number,
                bottom=y1_number / frame_height_number,
                right=x1_number / frame_width_number,
            )
        return candidate if candidate.area > 0.0 else None

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

    @property
    def zone(self) -> AICameraZone:
        """Return the coarse horizontal zone derived from ``center_x``."""

        return _infer_zone_from_box(self)

    def as_dict(self) -> dict[str, float]:
        """Return a JSON-safe mapping."""

        return {
            "top": self.top,
            "left": self.left,
            "bottom": self.bottom,
            "right": self.right,
        }


@dataclass(frozen=True, slots=True)
class AICameraVisiblePerson:
    """Describe one visible person anchor before higher-level tracking."""

    box: AICameraBox | None
    zone: AICameraZone = AICameraZone.UNKNOWN
    confidence: float = 0.0
    attention_hint_score: float | None = None

    def __post_init__(self) -> None:
        """Normalize person-anchor metadata into stable bounded values."""

        box = _coerce_box(self.box)
        zone = _coerce_zone(self.zone)
        if zone is AICameraZone.UNKNOWN and box is not None:
            zone = _infer_zone_from_box(box)
        object.__setattr__(self, "box", box)
        object.__setattr__(self, "zone", zone)
        object.__setattr__(self, "confidence", _clamp_ratio(self.confidence, default=0.0))
        object.__setattr__(
            self,
            "attention_hint_score",
            _coerce_optional_ratio(self.attention_hint_score),
        )


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
        box = _coerce_box(self.box)
        zone = _coerce_zone(self.zone)
        if zone is AICameraZone.UNKNOWN and box is not None:
            zone = _infer_zone_from_box(box)
        object.__setattr__(self, "label", label if label else "unknown")
        object.__setattr__(self, "confidence", _clamp_ratio(self.confidence, default=0.0))
        object.__setattr__(self, "zone", zone)
        object.__setattr__(self, "box", box)


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
    gesture_temporal_authoritative: bool = False
    gesture_activation_key: str | None = None
    gesture_activation_token: int | None = None
    gesture_activation_started_at: float | None = None
    gesture_activation_changed_at: float | None = None
    gesture_activation_source: str | None = None
    gesture_activation_rising: bool = False
    objects: tuple[AICameraObjectDetection, ...] = ()
    model: str = "local-imx500"

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
        *,
        reject_unknown_fields: bool = False,
    ) -> "AICameraObservation":
        """Construct one observation from a mapping payload.

        ``reject_unknown_fields`` is optional to stay drop-in friendly. The default
        filters extras silently, while strict call sites can make unknown inputs fatal.
        """

        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        unknown_fields = [str(key) for key in payload.keys() if str(key) not in cls._KNOWN_INPUT_FIELD_SET]
        if reject_unknown_fields and unknown_fields:
            unknown_preview = ", ".join(sorted(unknown_fields)[:8])
            raise ValueError(f"unknown observation field(s): {unknown_preview}")
        kwargs = {field_name: payload[field_name] for field_name in cls._KNOWN_INPUT_FIELDS if field_name in payload}
        return cls(**kwargs)

    @classmethod
    def from_json(
        cls,
        payload: str | bytes | bytearray,
        *,
        reject_unknown_fields: bool = False,
        max_bytes: int = _MAX_JSON_BYTES,
    ) -> "AICameraObservation":
        """Construct one observation from JSON.

        The size guard prevents oversized local IPC payloads from exhausting memory on
        constrained devices.
        """

        buffer = _coerce_json_buffer(payload, max_bytes=max_bytes)
        if msgspec is not None:
            raw_payload = msgspec.json.decode(buffer, type=dict[str, object])
        else:
            try:
                raw_payload = json.loads(buffer.decode("utf-8"))
            except UnicodeDecodeError as exc:
                raise ValueError("observation JSON must be UTF-8") from exc
            except json.JSONDecodeError as exc:
                raise ValueError("observation JSON is invalid") from exc
        if not isinstance(raw_payload, Mapping):
            raise ValueError("observation JSON must decode to an object")
        return cls.from_mapping(raw_payload, reject_unknown_fields=reject_unknown_fields)

    @classmethod
    def json_schema(cls) -> dict[str, object]:
        """Return a machine-readable JSON Schema for the observation contract.

        ``msgspec`` provides the fastest, lowest-overhead schema generation path on
        Python edge stacks. Without it, the rest of the module remains usable.
        """

        if msgspec is None:
            raise RuntimeError("json_schema() requires optional dependency 'msgspec'")
        schema = msgspec.json.schema(cls)
        if isinstance(schema, dict):
            schema.setdefault("$schema", "https://json-schema.org/draft/2020-12/schema")
            schema.setdefault("title", cls.__name__)
            schema.setdefault("description", cls.__doc__ or "")
            schema.setdefault("x-contract-version", AICAMERA_OBSERVATION_CONTRACT_VERSION)
        return schema

    def __post_init__(self) -> None:
        """Normalize observation metadata into safe, stable contract values."""

        primary_person_box = _coerce_box(self.primary_person_box)
        objects = _coerce_objects(self.objects)
        visible_persons = _coerce_visible_persons(self.visible_persons)

        # BREAKING: Missing primary-person fields are now backfilled from the best
        # visible person or highest-confidence person detection instead of remaining
        # ``None``/``unknown`` when the information is already present elsewhere.
        primary_visible_person = _select_primary_visible_person(visible_persons)
        primary_person_detection = _select_primary_person_detection(objects)

        person_count = _coerce_non_negative_int(self.person_count, default=0)
        if visible_persons and person_count < len(visible_persons):
            person_count = len(visible_persons)
        if primary_person_box is not None and person_count < 1:
            person_count = 1
        if person_count < 1 and primary_person_detection is not None:
            person_count = 1

        primary_person_zone = _coerce_zone(self.primary_person_zone)

        if primary_person_box is None and primary_visible_person is not None:
            primary_person_box = primary_visible_person.box
            if primary_person_zone is AICameraZone.UNKNOWN:
                primary_person_zone = primary_visible_person.zone

        if primary_person_box is None and primary_person_detection is not None:
            primary_person_box = primary_person_detection.box
            if primary_person_zone is AICameraZone.UNKNOWN:
                primary_person_zone = primary_person_detection.zone
            if not visible_persons and primary_person_box is not None:
                visible_persons = (
                    AICameraVisiblePerson(
                        box=primary_person_box,
                        zone=primary_person_zone,
                        confidence=primary_person_detection.confidence,
                    ),
                )

        if primary_person_box is not None and primary_person_zone is AICameraZone.UNKNOWN:
            primary_person_zone = _infer_zone_from_box(primary_person_box)

        if primary_person_box is not None and not visible_persons:
            visible_persons = (
                AICameraVisiblePerson(
                    box=primary_person_box,
                    zone=primary_person_zone,
                    confidence=1.0,
                ),
            )

        camera_online = _coerce_bool(self.camera_online, default=False)
        camera_ready = _coerce_bool(self.camera_ready, default=False) and camera_online
        camera_ai_ready = _coerce_bool(self.camera_ai_ready, default=False) and camera_ready

        observed_at = _sanitize_timestamp(self.observed_at, default=0.0)
        if observed_at is None:
            observed_at = 0.0

        object.__setattr__(self, "observed_at", observed_at)
        object.__setattr__(self, "camera_online", camera_online)
        object.__setattr__(self, "camera_ready", camera_ready)
        object.__setattr__(self, "camera_ai_ready", camera_ai_ready)
        object.__setattr__(
            self,
            "camera_error",
            _normalize_text(self.camera_error, max_length=_MAX_TEXT_LENGTH, default=None),
        )
        object.__setattr__(
            self,
            "last_camera_frame_at",
            _sanitize_relative_timestamp(self.last_camera_frame_at, default=None, reference=observed_at),
        )
        object.__setattr__(
            self,
            "last_camera_health_change_at",
            _sanitize_relative_timestamp(self.last_camera_health_change_at, default=None, reference=observed_at),
        )
        object.__setattr__(self, "person_count", person_count)
        object.__setattr__(self, "primary_person_box", primary_person_box)
        object.__setattr__(self, "primary_person_zone", primary_person_zone)
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
        object.__setattr__(
            self,
            "hand_or_object_near_camera",
            _coerce_bool(self.hand_or_object_near_camera, default=False),
        )
        object.__setattr__(self, "showing_intent_likely", _coerce_optional_bool(self.showing_intent_likely))
        object.__setattr__(self, "gesture_event", _coerce_gesture_event(self.gesture_event))
        object.__setattr__(self, "gesture_confidence", _coerce_optional_ratio(self.gesture_confidence))
        object.__setattr__(self, "fine_hand_gesture", _coerce_fine_hand_gesture(self.fine_hand_gesture))
        object.__setattr__(
            self,
            "fine_hand_gesture_confidence",
            _coerce_optional_ratio(self.fine_hand_gesture_confidence),
        )
        object.__setattr__(
            self,
            "gesture_temporal_authoritative",
            _coerce_bool(self.gesture_temporal_authoritative, default=False),
        )
        object.__setattr__(
            self,
            "gesture_activation_key",
            _normalize_text(self.gesture_activation_key, max_length=96, default=None),
        )
        object.__setattr__(
            self,
            "gesture_activation_token",
            _coerce_optional_non_negative_int(self.gesture_activation_token),
        )
        object.__setattr__(
            self,
            "gesture_activation_started_at",
            _sanitize_relative_timestamp(self.gesture_activation_started_at, default=None, reference=observed_at),
        )
        object.__setattr__(
            self,
            "gesture_activation_changed_at",
            _sanitize_relative_timestamp(self.gesture_activation_changed_at, default=None, reference=observed_at),
        )
        object.__setattr__(
            self,
            "gesture_activation_source",
            _normalize_token(self.gesture_activation_source, max_length=64) or None,
        )
        object.__setattr__(
            self,
            "gesture_activation_rising",
            _coerce_bool(self.gesture_activation_rising, default=False),
        )
        object.__setattr__(self, "objects", objects)
        object.__setattr__(
            self,
            "model",
            _normalize_token(self.model, max_length=_MAX_MODEL_LENGTH) or "local-imx500",
        )

    @property
    def contract_version(self) -> str:
        """Return the semantic version of this observation contract."""

        return AICAMERA_OBSERVATION_CONTRACT_VERSION

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

    def to_dict(self, *, include_contract_version: bool = False) -> dict[str, object]:
        """Return a JSON-safe mapping representation."""

        result = _to_builtin(self)
        if not isinstance(result, dict):
            raise TypeError("observation did not serialize to a mapping")
        if include_contract_version:
            result["contract_version"] = self.contract_version
        return result

    def to_json(
        self,
        *,
        include_contract_version: bool = False,
        pretty: bool = False,
        sort_keys: bool = False,
    ) -> str:
        """Return a JSON representation.

        When ``msgspec`` is installed, encoding is performed by the faster edge-friendly
        implementation. Otherwise stdlib JSON is used.
        """

        payload = self.to_dict(include_contract_version=include_contract_version)
        if msgspec is not None:
            encoder = msgspec.json.Encoder(order="sorted" if sort_keys else None)
            encoded = encoder.encode(payload)
            if pretty:
                encoded = msgspec.json.format(encoded, indent=2)
            return encoded.decode("utf-8")
        if pretty:
            return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=sort_keys)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=sort_keys)


AICameraObservation._KNOWN_INPUT_FIELDS = tuple(field.name for field in dataclass_fields(AICameraObservation))
AICameraObservation._KNOWN_INPUT_FIELD_SET = frozenset(AICameraObservation._KNOWN_INPUT_FIELDS)


def _normalize_token(value: object, *, max_length: int) -> str:
    """Normalize one token-like value into a safe bounded identifier."""

    if value is None or isinstance(value, bool):
        return ""
    text = _CONTROL_CHAR_RE.sub("", str(value))
    text = "_".join(text.strip().lower().split())
    text = _SAFE_TOKEN_RE.sub("_", text)
    text = _MULTI_UNDERSCORE_RE.sub("_", text).strip("_.-")
    return text[:max_length]


def _normalize_enum_token(value: object, *, max_length: int) -> str:
    """Normalize one enum-like token while collapsing kebab-case to snake_case."""

    token = _normalize_token(value, max_length=max_length).replace("-", "_")
    token = _MULTI_UNDERSCORE_RE.sub("_", token).strip("_")
    return token[:max_length]


def _normalize_label(value: object) -> str:
    """Normalize one object label to an inspectable token."""

    return _normalize_token(value, max_length=_MAX_LABEL_LENGTH)


def _normalize_text(value: object, *, max_length: int, default: str | None = None) -> str | None:
    """Normalize one free-text value into a bounded single-line string."""

    if value is None or isinstance(value, bool):
        return default
    text = _CONTROL_CHAR_RE.sub(" ", str(value))
    text = " ".join(text.split())
    text = text[:max_length].strip()
    return text if text else default


def _coerce_str_enum(value: object, *, enum_type: type[_StrEnumT], default: _StrEnumT) -> _StrEnumT:
    """Coerce one token-like payload into the requested ``StrEnum`` value."""

    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(_normalize_enum_token(value, max_length=_MAX_LABEL_LENGTH))
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


def _coerce_positive_float(value: object, *, default: float | None) -> float | None:
    """Coerce one object to a finite positive float, or return ``default``."""

    number = _coerce_finite_float(value, default=default)
    if number is None or number <= 0.0:
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


def _sanitize_relative_timestamp(
    value: object,
    *,
    default: float | None,
    reference: float | None,
) -> float | None:
    """Coerce one timestamp and clamp future values to ``reference`` when provided."""

    number = _sanitize_timestamp(value, default=default)
    if number is None:
        return default
    if reference is not None and reference > 0.0 and number > reference:
        return reference
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


def _coerce_non_negative_int(value: object, *, default: int = 0) -> int:
    """Coerce one count-like value to a non-negative integer."""

    number = _coerce_finite_float(value, default=None)
    if number is None:
        return max(0, default)
    if number < 0.0:
        return 0
    return int(number)


def _coerce_optional_non_negative_int(value: object) -> int | None:
    """Coerce one optional count-like value to a non-negative integer."""

    if value is None:
        return None
    number = _coerce_non_negative_int(value, default=-1)
    return None if number < 0 else number


def _infer_zone_from_box(box: AICameraBox | None) -> AICameraZone:
    """Derive a coarse horizontal zone from a normalized box center."""

    if box is None or box.area <= 0.0:
        return AICameraZone.UNKNOWN
    center_x = box.center_x
    if center_x < (1.0 / 3.0):
        return AICameraZone.LEFT
    if center_x > (2.0 / 3.0):
        return AICameraZone.RIGHT
    return AICameraZone.CENTER


def _coerce_box_from_mapping(value: Mapping[str, object]) -> AICameraBox | None:
    """Coerce one mapping-like box payload into ``AICameraBox`` when possible."""

    frame_width = (
        value.get("frame_width")
        or value.get("image_width")
        or value.get("source_width")
        or value.get("width_px")
    )
    frame_height = (
        value.get("frame_height")
        or value.get("image_height")
        or value.get("source_height")
        or value.get("height_px")
    )

    if all(key in value for key in ("top", "left", "bottom", "right")):
        candidate = AICameraBox(
            top=value["top"],
            left=value["left"],
            bottom=value["bottom"],
            right=value["right"],
        )
        return candidate if candidate.area > 0.0 else None

    if all(key in value for key in ("left", "top", "right", "bottom")):
        candidate = AICameraBox(
            top=value["top"],
            left=value["left"],
            bottom=value["bottom"],
            right=value["right"],
        )
        return candidate if candidate.area > 0.0 else None

    if all(key in value for key in ("x", "y", "width", "height")):
        return AICameraBox.from_xywh(
            x=value["x"],
            y=value["y"],
            width=value["width"],
            height=value["height"],
            frame_width=frame_width,
            frame_height=frame_height,
        )

    if all(key in value for key in ("x", "y", "w", "h")):
        return AICameraBox.from_xywh(
            x=value["x"],
            y=value["y"],
            width=value["w"],
            height=value["h"],
            frame_width=frame_width,
            frame_height=frame_height,
        )

    if all(key in value for key in ("x0", "y0", "x1", "y1")):
        return AICameraBox.from_xyxy(
            x0=value["x0"],
            y0=value["y0"],
            x1=value["x1"],
            y1=value["y1"],
            frame_width=frame_width,
            frame_height=frame_height,
        )

    if all(key in value for key in ("xmin", "ymin", "xmax", "ymax")):
        return AICameraBox.from_xyxy(
            x0=value["xmin"],
            y0=value["ymin"],
            x1=value["xmax"],
            y1=value["ymax"],
            frame_width=frame_width,
            frame_height=frame_height,
        )

    return None


def _coerce_box(value: object) -> AICameraBox | None:
    """Coerce one box-like payload to ``AICameraBox``."""

    if value is None:
        return None
    if isinstance(value, AICameraBox):
        return value
    if isinstance(value, Mapping):
        return _coerce_box_from_mapping(value)
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


def _coerce_objects(value: object) -> tuple[AICameraObjectDetection, ...]:
    """Coerce one object-collection payload to a stable tuple of detections."""

    if value is None:
        return ()
    single = _coerce_object_detection(value)
    if single is not None:
        return (single,)
    if isinstance(value, (str, bytes, bytearray)):
        return ()
    if not isinstance(value, Iterable):
        return ()
    detections: list[AICameraObjectDetection] = []
    for item in value:
        detection = _coerce_object_detection(item)
        if detection is not None:
            detections.append(detection)
            if len(detections) >= _MAX_OBJECTS:
                break
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
    if not isinstance(value, Iterable):
        return ()
    people: list[AICameraVisiblePerson] = []
    for item in value:
        person = _coerce_visible_person(item)
        if person is not None and person.box is not None:
            people.append(person)
            if len(people) >= _MAX_VISIBLE_PERSONS:
                break
    return tuple(people)


def _person_priority(person: AICameraVisiblePerson) -> tuple[float, float, float]:
    """Rank visible persons for deterministic primary-person selection."""

    return (
        person.confidence,
        person.attention_hint_score if person.attention_hint_score is not None else -1.0,
        person.box.area if person.box is not None else 0.0,
    )


def _detection_priority(detection: AICameraObjectDetection) -> tuple[float, float]:
    """Rank detections for deterministic fallback selection."""

    return (
        detection.confidence,
        detection.box.area if detection.box is not None else 0.0,
    )


def _select_primary_visible_person(
    visible_persons: tuple[AICameraVisiblePerson, ...],
) -> AICameraVisiblePerson | None:
    """Choose the most useful visible person anchor for primary-person fallback."""

    if not visible_persons:
        return None
    return max(visible_persons, key=_person_priority)


def _select_primary_person_detection(
    objects: tuple[AICameraObjectDetection, ...],
) -> AICameraObjectDetection | None:
    """Choose the best person detection for primary-person fallback."""

    person_detections = [
        detection
        for detection in objects
        if detection.label == "person" and detection.confidence > 0.0 and detection.box is not None
    ]
    if not person_detections:
        return None
    return max(person_detections, key=_detection_priority)


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


def _coerce_json_buffer(payload: str | bytes | bytearray, *, max_bytes: int) -> bytes:
    """Normalize JSON input to bytes and enforce a bounded size."""

    if isinstance(payload, str):
        buffer = payload.encode("utf-8")
    elif isinstance(payload, (bytes, bytearray)):
        buffer = bytes(payload)
    else:
        raise TypeError("payload must be str, bytes, or bytearray")
    if max_bytes > 0 and len(buffer) > max_bytes:
        raise ValueError(f"observation JSON exceeds {max_bytes} bytes")
    return buffer


def _to_builtin(value: object) -> object:
    """Convert contract objects to JSON-safe builtin values."""

    if msgspec is not None:
        try:
            return msgspec.to_builtins(value, str_keys=True)
        except Exception:
            pass

    if isinstance(value, StrEnum):
        return str(value)
    if isinstance(value, AICameraBox):
        return value.as_dict()
    if isinstance(value, AICameraObservation):
        return {
            field_name: _to_builtin(getattr(value, field_name))
            for field_name in value._KNOWN_INPUT_FIELDS
        }
    if isinstance(value, (AICameraVisiblePerson, AICameraObjectDetection)):
        return {
            field.name: _to_builtin(getattr(value, field.name))
            for field in dataclass_fields(type(value))
        }
    if isinstance(value, Mapping):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_builtin(item) for item in value]
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    return value


__all__ = [
    "AICAMERA_OBSERVATION_CONTRACT_VERSION",
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
