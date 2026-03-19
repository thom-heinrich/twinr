"""Infer bounded motion states from the primary detected person box over time."""

from __future__ import annotations

import math
import numbers

from .config import _clamp_ratio
from .models import AICameraBox, AICameraMotionState


_MOTION_UNKNOWN_MAX_GAP_S = 12.0
_MOTION_MIN_DELTA_S = 0.15
_RATIO_EPSILON = 1e-6  # AUDIT-FIX(#1): Allow tiny float noise while still rejecting materially invalid normalized box geometry.


def _coerce_finite_float(value: object) -> float | None:
    # AUDIT-FIX(#1): Malformed detector/timestamp values must degrade to UNKNOWN instead of crashing or leaking NaN/inf downstream.
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _coerce_normalized_ratio(value: object) -> float | None:
    # AUDIT-FIX(#1): Motion thresholds assume normalized 0..1 geometry; reject corrupt values after allowing tiny floating-point overshoot.
    numeric = _coerce_finite_float(value)
    if numeric is None:
        return None
    if numeric < -_RATIO_EPSILON or numeric > 1.0 + _RATIO_EPSILON:
        return None
    return min(1.0, max(0.0, numeric))


def _extract_normalized_box(box: AICameraBox) -> tuple[float, float, float, float] | None:
    # AUDIT-FIX(#1): Validate all box fields before arithmetic so broken detector objects cannot trigger exceptions or NaN confidences.
    try:
        center_x = _coerce_normalized_ratio(box.center_x)
        center_y = _coerce_normalized_ratio(box.center_y)
        area = _coerce_normalized_ratio(box.area)
        height = _coerce_normalized_ratio(box.height)
    except AttributeError:
        return None

    if center_x is None or center_y is None or area is None or height is None:
        return None
    return center_x, center_y, area, height


def _is_single_person_count(value: object) -> bool:
    # AUDIT-FIX(#3): Reject booleans while still accepting integer-like runtime types such as numpy integer scalars.
    return isinstance(value, numbers.Integral) and not isinstance(value, bool) and int(value) == 1


def _bounded_confidence(value: float) -> float:
    # AUDIT-FIX(#1): Force every confidence through a finite 0..1 path so downstream state/JSON handling never sees NaN/inf.
    if not math.isfinite(value):
        return 0.0
    bounded = _coerce_finite_float(_clamp_ratio(value, default=0.0))
    return 0.0 if bounded is None else round(bounded, 3)


def infer_motion_state(
    *,
    previous_box: AICameraBox | None,
    current_box: AICameraBox | None,
    previous_observed_at: float | None,
    current_observed_at: float,
    previous_person_count: int,
    current_person_count: int,
) -> tuple[AICameraMotionState, float | None]:
    """Infer one coarse motion state from recent primary-person box deltas."""

    if previous_box is None or current_box is None or previous_observed_at is None:
        return AICameraMotionState.UNKNOWN, None

    if not _is_single_person_count(previous_person_count) or not _is_single_person_count(current_person_count):
        return AICameraMotionState.UNKNOWN, None

    # AUDIT-FIX(#1): Validate runtime sensor payloads at the boundary; type annotations alone do not protect production pipelines.
    previous_box_values = _extract_normalized_box(previous_box)
    current_box_values = _extract_normalized_box(current_box)
    previous_ts = _coerce_finite_float(previous_observed_at)
    current_ts = _coerce_finite_float(current_observed_at)
    if previous_box_values is None or current_box_values is None or previous_ts is None or current_ts is None:
        return AICameraMotionState.UNKNOWN, None

    # AUDIT-FIX(#2): Reject reversed or out-of-order timestamps explicitly instead of manufacturing a synthetic zero-delta sample.
    delta_t = current_ts - previous_ts
    if delta_t < _MOTION_MIN_DELTA_S or delta_t > _MOTION_UNKNOWN_MAX_GAP_S:
        return AICameraMotionState.UNKNOWN, None

    previous_center_x, previous_center_y, previous_area, previous_height = previous_box_values
    current_center_x, current_center_y, current_area, current_height = current_box_values

    delta_x = current_center_x - previous_center_x
    delta_y = current_center_y - previous_center_y
    center_distance = math.hypot(delta_x, delta_y)
    center_speed = center_distance / delta_t
    area_delta = current_area - previous_area
    height_delta = current_height - previous_height
    scale_strength = max(abs(area_delta), abs(height_delta))
    confidence = _bounded_confidence(max(center_speed * 5.0, scale_strength * 3.4))

    if abs(area_delta) >= 0.06 and abs(delta_x) <= 0.10:
        if area_delta > 0.0 and height_delta >= -0.01:
            return AICameraMotionState.APPROACHING, max(confidence, 0.56)
        if area_delta < 0.0 and height_delta <= 0.01:
            return AICameraMotionState.LEAVING, max(confidence, 0.56)

    if center_speed >= 0.08 or (abs(delta_x) >= 0.08 and scale_strength <= 0.08):
        return AICameraMotionState.WALKING, max(confidence, 0.54)

    if center_distance <= 0.035 and scale_strength <= 0.04:
        return AICameraMotionState.STILL, max(0.52, round(0.52 + (0.04 - scale_strength), 3))

    if abs(area_delta) >= 0.04:
        return (
            (AICameraMotionState.APPROACHING if area_delta > 0.0 else AICameraMotionState.LEAVING),
            max(confidence, 0.5),
        )
    return AICameraMotionState.STILL, max(confidence, 0.45)


__all__ = ["infer_motion_state"]