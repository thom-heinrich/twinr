"""Normalization and serialization helpers for the camera surface package."""

from __future__ import annotations

import math

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
def serialize_box(box: SocialSpatialBox | None) -> dict[str, float] | None:
    """Render one spatial box as a JSON-friendly mapping."""

    if box is None:
        return None
    return {
        "top": round(box.top, 4),
        "left": round(box.left, 4),
        "bottom": round(box.bottom, 4),
        "right": round(box.right, 4),
    }


def serialize_objects(objects: tuple[SocialDetectedObject, ...]) -> list[dict[str, object]]:
    """Render stable objects into JSON-friendly dictionaries."""

    rendered: list[dict[str, object]] = []
    for item in objects:
        rendered.append(
            {
                "label": item.label,
                "confidence": round(item.confidence, 4),
                "zone": item.zone.value,
                "stable": item.stable,
                "box": serialize_box(item.box),
            }
        )
    return rendered


def serialize_visible_persons(people: tuple[SocialVisiblePerson, ...]) -> list[dict[str, object]]:
    """Render visible-person anchors into JSON-friendly dictionaries."""

    rendered: list[dict[str, object]] = []
    for item in people:
        rendered.append(
            {
                "box": serialize_box(item.box),
                "zone": item.zone.value,
                "confidence": round(item.confidence, 4),
            }
        )
    return rendered


def coerce_timestamp(value: object, *, previous: float | None = None) -> float:
    """Normalize one timestamp into a monotonic finite float."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0 if previous is None else previous
    number = float(value)
    if not math.isfinite(number):
        number = 0.0 if previous is None else previous
    if previous is not None and number < previous:
        return previous
    return number


def coerce_optional_timestamp(value: object) -> float | None:
    """Return one finite non-negative timestamp or ``None``."""

    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        return None
    return number


def duration_since(since: float | None, now: float) -> float:
    """Return the elapsed duration since ``since`` with a safe lower bound."""

    if since is None:
        return 0.0
    return max(0.0, now - since)


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

    primary = coerce_gesture_event(getattr(observation, "coarse_arm_gesture", SocialGestureEvent.UNKNOWN))
    if primary not in {SocialGestureEvent.NONE, SocialGestureEvent.UNKNOWN}:
        return primary
    secondary = coerce_gesture_event(getattr(observation, "gesture_event", SocialGestureEvent.UNKNOWN))
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

    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number


def coerce_spatial_box(value: object) -> SocialSpatialBox | None:
    """Coerce one box-like payload into ``SocialSpatialBox``."""

    if isinstance(value, SocialSpatialBox):
        return value
    coordinates = coerce_spatial_box_coordinates(value)
    if coordinates is None:
        return None
    top, left, bottom, right = coordinates
    return SocialSpatialBox(top=top, left=left, bottom=bottom, right=right)


def coerce_detected_objects(value: object) -> tuple[SocialDetectedObject, ...]:
    """Coerce one list-like payload to a tuple of detected objects."""

    if value is None:
        return ()
    if isinstance(value, tuple) and all(isinstance(item, SocialDetectedObject) for item in value):
        return value
    if not isinstance(value, (tuple, list)):
        return ()
    parsed: list[SocialDetectedObject] = []
    for item in value:
        if isinstance(item, SocialDetectedObject):
            parsed.append(item)
            continue
        if not isinstance(item, dict):
            continue
        parsed.append(
            SocialDetectedObject(
                label=item.get("label", ""),
                confidence=coerce_optional_ratio(item.get("confidence")) or 0.0,
                zone=coerce_person_zone(item.get("zone")),
                stable=bool(item.get("stable", False)),
                box=coerce_spatial_box(item.get("box")),
            )
        )
    return tuple(parsed)


def coerce_visible_persons(value: object) -> tuple[SocialVisiblePerson, ...]:
    """Coerce one list-like payload to visible-person anchors."""

    if value is None:
        return ()
    if isinstance(value, tuple) and all(isinstance(item, SocialVisiblePerson) for item in value):
        return value
    if not isinstance(value, (tuple, list)):
        return ()
    parsed: list[SocialVisiblePerson] = []
    for item in value:
        if isinstance(item, SocialVisiblePerson):
            parsed.append(item)
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
    """Return one bounded text value or ``None``."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[:160]


def camera_health_surface_present(observation: SocialVisionObservation) -> bool:
    """Return whether the observation carries meaningful local camera health."""

    return any(
        (
            bool(getattr(observation, "camera_online", False)),
            bool(getattr(observation, "camera_ready", False)),
            bool(getattr(observation, "camera_ai_ready", False)),
            bool(str(getattr(observation, "camera_error", "") or "").strip()),
            getattr(observation, "last_camera_frame_at", None) is not None,
            getattr(observation, "last_camera_health_change_at", None) is not None,
        )
    )


def camera_semantics_authoritative(observation: SocialVisionObservation) -> bool:
    """Return whether person/pose/gesture fields describe one real camera frame.

    The local AI-camera path can emit explicit health failures. Those failures
    must make camera semantics unknown, not concrete "no person" facts, so the
    gaze/gesture layers can hold the last stable target through brief runtime
    faults instead of clearing immediately.
    """

    if not camera_health_surface_present(observation):
        return True
    if not bool(getattr(observation, "camera_online", False)):
        return False
    if not bool(getattr(observation, "camera_ready", False)):
        return False
    if not bool(getattr(observation, "camera_ai_ready", False)):
        return False
    return not bool(str(getattr(observation, "camera_error", "") or "").strip())
