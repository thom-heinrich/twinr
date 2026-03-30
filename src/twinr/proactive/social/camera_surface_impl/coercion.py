"""Normalization and serialization helpers for the camera surface package."""

# CHANGELOG: 2026-03-29
# BUG-1: serialize_box/serialize_objects/serialize_visible_persons no longer emit NaN/Inf
#        or unnormalized coordinates, which previously broke strict JSON encoders and could
#        leak invalid geometry/confidence downstream.
# BUG-2: fixed Python truthiness bugs where payloads like "false", "0", "off", or "no"
#        were treated as True for object stability and camera health semantics.
# BUG-3: coerce_timestamp now enforces finite non-negative monotonic timestamps; negative
#        inputs previously produced bogus elapsed durations and stale/fresh decisions.
# BUG-4: SocialSpatialBox / SocialDetectedObject / SocialVisiblePerson instances are now
#        normalized even when already typed, so malformed in-memory objects cannot bypass
#        box/confidence sanitization.
# SEC-1: bound object/person list sizes to prevent practical resource-exhaustion on
#        Raspberry Pi 4 from oversized payloads.
# SEC-2: labels/text are normalized to bounded single-line strings to reduce log/control
#        sequence injection and output amplification from untrusted payloads.
# IMP-1: added camera_semantics_state(..., now=..., max_frame_staleness_s=...) so downstream
#        layers can reason about freshness, authority, and failure reasons instead of a bare bool.
# IMP-2: centralized strict coercion helpers (finite floats, booleans, text) to align the
#        module with 2026 boundary-validation practice without introducing a hard runtime dependency.
# BREAKING: oversize object/person lists are truncated to bounded maxima by default.
# BREAKING: control characters and overlong labels/text are stripped/truncated during coercion/serialization.

from __future__ import annotations

import math
from typing import Final, TypedDict

from ..engine import (
    SocialBodyPose,
    SocialDetectedObject,
    SocialFineHandGesture,
    SocialGestureEvent,
    SocialMotionState,
    SocialPersonZone,
    SocialSpatialBox,
    SocialVisiblePerson,
    SocialVisionObservation,
)
from ..normalization import coerce_enum_member, coerce_spatial_box_coordinates


# BREAKING: payload collections are bounded so malformed or hostile inputs cannot
# allocate unbounded memory/CPU on a Raspberry Pi.
DEFAULT_MAX_DETECTED_OBJECTS: Final[int] = 128
DEFAULT_MAX_VISIBLE_PERSONS: Final[int] = 32
DEFAULT_MAX_LABEL_LENGTH: Final[int] = 64
DEFAULT_MAX_TEXT_LENGTH: Final[int] = 160
DEFAULT_MAX_FRAME_STALENESS_S: Final[float] = 2.0

_TRUE_STRINGS: Final[frozenset[str]] = frozenset({"1", "true", "t", "yes", "y", "on"})
_FALSE_STRINGS: Final[frozenset[str]] = frozenset({"0", "false", "f", "no", "n", "off"})


class SerializedSpatialBox(TypedDict):
    top: float
    left: float
    bottom: float
    right: float


class SerializedDetectedObject(TypedDict):
    label: str
    confidence: float
    zone: str
    stable: bool
    box: SerializedSpatialBox | None


class SerializedVisiblePerson(TypedDict):
    box: SerializedSpatialBox | None
    zone: str
    confidence: float


class CameraSemanticsState(TypedDict):
    health_surface_present: bool
    authoritative: bool
    reason: str
    frame_age_s: float | None
    frame_fresh: bool | None


def _sanitize_text_token(text: str, *, max_length: int) -> str | None:
    cleaned_characters: list[str] = []
    for character in text:
        if character in {"\r", "\n", "\t"}:
            cleaned_characters.append(" ")
            continue
        if character < " " or character == "\x7f":
            continue
        cleaned_characters.append(character)
    normalized = " ".join("".join(cleaned_characters).strip().split())
    if not normalized:
        return None
    return normalized[:max_length]


def _coerce_text_token(value: object, *, max_length: int) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        text = value
    elif isinstance(value, (bytes, bytearray, memoryview)):
        text = bytes(value).decode("utf-8", "ignore")
    elif isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
        text = str(value)
    else:
        return None
    return _sanitize_text_token(text, max_length=max_length)


def _coerce_error_text(value: object) -> str | None:
    if isinstance(value, (str, bytes, bytearray, memoryview)):
        return _coerce_text_token(value, max_length=DEFAULT_MAX_TEXT_LENGTH)
    return None


def coerce_bool(value: object, *, default: bool = False) -> bool:
    """Normalize one raw boolean-like token without Python truthiness foot-guns."""

    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            return default
        return number != 0.0
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _TRUE_STRINGS:
            return True
        if token in _FALSE_STRINGS:
            return False
    return default


def _coerce_finite_float(
    value: object,
    *,
    default: float | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    number = float(value)
    if not math.isfinite(number):
        return default
    if minimum is not None and number < minimum:
        number = minimum
    if maximum is not None and number > maximum:
        number = maximum
    return number


def serialize_box(box: SocialSpatialBox | None) -> SerializedSpatialBox | None:
    """Render one spatial box as a JSON-friendly mapping."""

    if box is None:
        return None
    coordinates = coerce_spatial_box_coordinates((box.top, box.left, box.bottom, box.right))
    if coordinates is None:
        return None
    top, left, bottom, right = coordinates
    return {
        "top": round(top, 4),
        "left": round(left, 4),
        "bottom": round(bottom, 4),
        "right": round(right, 4),
    }


def serialize_objects(
    objects: tuple[SocialDetectedObject, ...],
) -> list[SerializedDetectedObject]:
    """Render stable objects into JSON-friendly dictionaries."""

    rendered: list[SerializedDetectedObject] = []
    for item in objects[:DEFAULT_MAX_DETECTED_OBJECTS]:
        rendered.append(
            {
                # BREAKING: labels are emitted as bounded single-line text only.
                "label": _coerce_text_token(item.label, max_length=DEFAULT_MAX_LABEL_LENGTH) or "",
                "confidence": round(coerce_optional_ratio(item.confidence) or 0.0, 4),
                "zone": coerce_person_zone(item.zone).value,
                "stable": coerce_bool(item.stable, default=False),
                "box": serialize_box(coerce_spatial_box(item.box)),
            }
        )
    return rendered


def serialize_visible_persons(
    people: tuple[SocialVisiblePerson, ...],
) -> list[SerializedVisiblePerson]:
    """Render visible-person anchors into JSON-friendly dictionaries."""

    rendered: list[SerializedVisiblePerson] = []
    for item in people[:DEFAULT_MAX_VISIBLE_PERSONS]:
        rendered.append(
            {
                "box": serialize_box(coerce_spatial_box(item.box)),
                "zone": coerce_person_zone(item.zone).value,
                "confidence": round(coerce_optional_ratio(item.confidence) or 0.0, 4),
            }
        )
    return rendered


def coerce_timestamp(value: object, *, previous: float | None = None) -> float:
    """Normalize one timestamp into a monotonic finite non-negative float."""

    baseline = 0.0 if previous is None else previous
    number = _coerce_finite_float(value, default=baseline)
    if number is None:
        return baseline
    if number < 0.0:
        number = 0.0 if previous is None else previous
    if previous is not None and number < previous:
        return previous
    return number


def coerce_optional_timestamp(value: object) -> float | None:
    """Return one finite non-negative timestamp or ``None``."""

    return _coerce_finite_float(value, default=None, minimum=0.0)


def duration_since(since: float | None, now: float) -> float:
    """Return the elapsed duration since ``since`` with a safe lower bound."""

    current = _coerce_finite_float(now, default=0.0, minimum=0.0)
    if since is None or current is None:
        return 0.0
    start = _coerce_finite_float(since, default=None, minimum=0.0)
    if start is None:
        return 0.0
    return max(0.0, current - start)


def coerce_body_pose(value: object) -> SocialBodyPose:
    """Normalize a raw pose token into the coarse ``SocialBodyPose`` enum."""

    return coerce_enum_member(
        value,
        SocialBodyPose,
        unknown=SocialBodyPose.UNKNOWN,
        allow_stringify=True,
    )


def coerce_motion_state(value: object) -> SocialMotionState:
    """Normalize a raw motion token into the coarse ``SocialMotionState`` enum."""

    return coerce_enum_member(
        value,
        SocialMotionState,
        unknown=SocialMotionState.UNKNOWN,
        allow_stringify=True,
    )


def coerce_person_zone(value: object) -> SocialPersonZone:
    """Normalize a raw zone token into the coarse ``SocialPersonZone`` enum."""

    return coerce_enum_member(
        value,
        SocialPersonZone,
        unknown=SocialPersonZone.UNKNOWN,
        allow_stringify=True,
    )


def coerce_gesture_event(value: object) -> SocialGestureEvent:
    """Normalize a raw gesture token into the ``SocialGestureEvent`` enum."""

    return coerce_enum_member(
        value,
        SocialGestureEvent,
        unknown=SocialGestureEvent.UNKNOWN,
        allow_stringify=True,
    )


def coalesce_coarse_gesture_aliases(observation: object) -> SocialGestureEvent:
    """Resolve coarse-arm gesture aliases without letting default `none` mask real data.

    `SocialVisionObservation` exposes both `coarse_arm_gesture` and the older
    compatibility field `gesture_event`. The dataclass always has both
    attributes, so a direct `getattr(..., "coarse_arm_gesture", fallback)`
    incorrectly prefers the default `none` value and hides a real gesture
    carried only on `gesture_event`. Prefer the first meaningful non-empty
    value while keeping `coarse_arm_gesture` authoritative when both are set.
    """

    primary = coerce_gesture_event(
        getattr(observation, "coarse_arm_gesture", SocialGestureEvent.UNKNOWN)
    )
    if primary not in {SocialGestureEvent.NONE, SocialGestureEvent.UNKNOWN}:
        return primary
    secondary = coerce_gesture_event(
        getattr(observation, "gesture_event", SocialGestureEvent.UNKNOWN)
    )
    if secondary not in {SocialGestureEvent.NONE, SocialGestureEvent.UNKNOWN}:
        return secondary
    if primary is not SocialGestureEvent.UNKNOWN:
        return primary
    return secondary


def coerce_fine_hand_gesture(value: object) -> SocialFineHandGesture:
    """Normalize a raw fine-hand gesture token into the bounded enum."""

    return coerce_enum_member(
        value,
        SocialFineHandGesture,
        unknown=SocialFineHandGesture.UNKNOWN,
        allow_stringify=True,
    )


def coerce_optional_ratio(value: object) -> float | None:
    """Clamp one optional numeric value into ``[0.0, 1.0]``."""

    return _coerce_finite_float(value, default=None, minimum=0.0, maximum=1.0)


def coerce_spatial_box(value: object) -> SocialSpatialBox | None:
    """Coerce one box-like payload into ``SocialSpatialBox``."""

    if isinstance(value, SocialSpatialBox):
        raw_value: object = (value.top, value.left, value.bottom, value.right)
    else:
        raw_value = value
    coordinates = coerce_spatial_box_coordinates(raw_value)
    if coordinates is None:
        return None
    top, left, bottom, right = coordinates
    return SocialSpatialBox(top=top, left=left, bottom=bottom, right=right)


def coerce_detected_objects(value: object) -> tuple[SocialDetectedObject, ...]:
    """Coerce one list-like payload to a tuple of detected objects."""

    if value is None:
        return ()
    if not isinstance(value, (tuple, list)):
        return ()
    parsed: list[SocialDetectedObject] = []
    for item in value[:DEFAULT_MAX_DETECTED_OBJECTS]:
        if isinstance(item, SocialDetectedObject):
            parsed.append(
                SocialDetectedObject(
                    # BREAKING: overlong/control-character labels are normalized here.
                    label=_coerce_text_token(item.label, max_length=DEFAULT_MAX_LABEL_LENGTH) or "",
                    confidence=coerce_optional_ratio(item.confidence) or 0.0,
                    zone=coerce_person_zone(item.zone),
                    stable=coerce_bool(item.stable, default=False),
                    box=coerce_spatial_box(item.box),
                )
            )
            continue
        if not isinstance(item, dict):
            continue
        parsed.append(
            SocialDetectedObject(
                label=_coerce_text_token(item.get("label"), max_length=DEFAULT_MAX_LABEL_LENGTH)
                or "",
                confidence=coerce_optional_ratio(item.get("confidence")) or 0.0,
                zone=coerce_person_zone(item.get("zone")),
                stable=coerce_bool(item.get("stable", False), default=False),
                box=coerce_spatial_box(item.get("box")),
            )
        )
    return tuple(parsed)


def coerce_visible_persons(value: object) -> tuple[SocialVisiblePerson, ...]:
    """Coerce one list-like payload to visible-person anchors."""

    if value is None:
        return ()
    if not isinstance(value, (tuple, list)):
        return ()
    parsed: list[SocialVisiblePerson] = []
    for item in value[:DEFAULT_MAX_VISIBLE_PERSONS]:
        if isinstance(item, SocialVisiblePerson):
            parsed.append(
                SocialVisiblePerson(
                    box=coerce_spatial_box(item.box),
                    zone=coerce_person_zone(item.zone),
                    confidence=coerce_optional_ratio(item.confidence) or 0.0,
                )
            )
            continue
        if not isinstance(item, dict):
            continue
        parsed.append(
            SocialVisiblePerson(
                box=coerce_spatial_box(item.get("box")),
                zone=coerce_person_zone(item.get("zone")),
                confidence=coerce_optional_ratio(item.get("confidence")) or 0.0,
            )
        )
    return tuple(parsed)


def coerce_optional_text(value: object) -> str | None:
    """Return one bounded single-line text value or ``None``."""

    return _coerce_text_token(value, max_length=DEFAULT_MAX_TEXT_LENGTH)


def camera_health_surface_present(observation: SocialVisionObservation) -> bool:
    """Return whether the observation carries meaningful local camera health."""

    camera_online = getattr(observation, "camera_online", None)
    camera_ready = getattr(observation, "camera_ready", None)
    camera_ai_ready = getattr(observation, "camera_ai_ready", None)

    return any(
        (
            coerce_bool(camera_online, default=False),
            coerce_bool(camera_ready, default=False),
            coerce_bool(camera_ai_ready, default=False),
            any(
                value is not None and not isinstance(value, bool)
                for value in (camera_online, camera_ready, camera_ai_ready)
            ),
            _coerce_error_text(getattr(observation, "camera_error", None)) is not None,
            coerce_optional_timestamp(getattr(observation, "last_camera_frame_at", None)) is not None,
            coerce_optional_timestamp(getattr(observation, "last_camera_health_change_at", None))
            is not None,
        )
    )


def camera_semantics_state(
    observation: SocialVisionObservation,
    *,
    now: float | None = None,
    max_frame_staleness_s: float | None = DEFAULT_MAX_FRAME_STALENESS_S,
) -> CameraSemanticsState:
    """Return a structured authority/freshness view for camera-derived semantics."""

    if not camera_health_surface_present(observation):
        return {
            "health_surface_present": False,
            "authoritative": True,
            "reason": "no_health_surface",
            "frame_age_s": None,
            "frame_fresh": None,
        }

    camera_error = _coerce_error_text(getattr(observation, "camera_error", None))
    if not coerce_bool(getattr(observation, "camera_online", False), default=False):
        return {
            "health_surface_present": True,
            "authoritative": False,
            "reason": "camera_offline",
            "frame_age_s": None,
            "frame_fresh": None,
        }
    if not coerce_bool(getattr(observation, "camera_ready", False), default=False):
        return {
            "health_surface_present": True,
            "authoritative": False,
            "reason": "camera_not_ready",
            "frame_age_s": None,
            "frame_fresh": None,
        }
    if not coerce_bool(getattr(observation, "camera_ai_ready", False), default=False):
        return {
            "health_surface_present": True,
            "authoritative": False,
            "reason": "camera_ai_not_ready",
            "frame_age_s": None,
            "frame_fresh": None,
        }
    if camera_error is not None:
        return {
            "health_surface_present": True,
            "authoritative": False,
            "reason": "camera_error",
            "frame_age_s": None,
            "frame_fresh": None,
        }

    frame_timestamp = coerce_optional_timestamp(getattr(observation, "last_camera_frame_at", None))
    current = coerce_optional_timestamp(now)
    if (
        frame_timestamp is not None
        and current is not None
        and max_frame_staleness_s is not None
        and math.isfinite(max_frame_staleness_s)
        and max_frame_staleness_s >= 0.0
    ):
        frame_age_s = max(0.0, current - frame_timestamp)
        frame_fresh = frame_age_s <= max_frame_staleness_s
        if not frame_fresh:
            return {
                "health_surface_present": True,
                "authoritative": False,
                "reason": "stale_frame",
                "frame_age_s": round(frame_age_s, 4),
                "frame_fresh": False,
            }
        return {
            "health_surface_present": True,
            "authoritative": True,
            "reason": "ok",
            "frame_age_s": round(frame_age_s, 4),
            "frame_fresh": True,
        }

    return {
        "health_surface_present": True,
        "authoritative": True,
        "reason": "ok",
        "frame_age_s": None,
        "frame_fresh": None,
    }


def camera_semantics_authoritative(
    observation: SocialVisionObservation,
    *,
    now: float | None = None,
    max_frame_staleness_s: float | None = DEFAULT_MAX_FRAME_STALENESS_S,
) -> bool:
    """Return whether person/pose/gesture fields describe one real camera frame.

    The local AI-camera path can emit explicit health failures. Those failures
    must make camera semantics unknown, not concrete "no person" facts, so the
    gaze/gesture layers can hold the last stable target through brief runtime
    faults instead of clearing immediately.

    When ``now`` and ``max_frame_staleness_s`` are provided, the decision also
    becomes freshness-aware instead of relying only on camera health flags.
    """

    return camera_semantics_state(
        observation,
        now=now,
        max_frame_staleness_s=max_frame_staleness_s,
    )["authoritative"]