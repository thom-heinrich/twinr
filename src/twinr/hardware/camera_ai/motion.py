"""Infer bounded motion states from the primary detected person box over time."""

from __future__ import annotations

# CHANGELOG: 2026-03-28
# BUG-1: Accepted only >=0.15 s deltas, which suppresses motion inference on common high-FPS edge camera loops
#        (for example 30 fps AI camera streams) and yields excessive UNKNOWN outputs.
# BUG-2: Compared raw area/height deltas without time normalization, so the same physical motion could be labeled
#        differently at different frame rates or after dropped frames.
# BUG-3: Accepted geometrically impossible normalized boxes (for example implied width > 1.0), allowing corrupt
#        detector payloads to produce nonsense motion states instead of degrading safely to UNKNOWN.
# BUG-4: Had no continuity gate, so detector ID switches / scene cuts / large jump discontinuities could be
#        misreported as WALKING or APPROACHING.
# SEC-1: Hardened against practical input-spoofing / malformed upstream payloads by validating box geometry more
#        strictly and rejecting track-mismatched or discontinuous samples before state emission.
# IMP-1: Upgraded from fixed raw-delta heuristics to dt-normalized, scale-aware motion features with adaptive
#        jitter floors that behave consistently across frame rates on Raspberry Pi-class edge devices.
# IMP-2: Added optional consumption of tracker metadata (`track_id`) and detector confidence (`confidence`/`conf`/`score`)
#        so the function can exploit 2026 tracking pipelines without breaking callers that only provide bare boxes.
# BREAKING: None. The public function signature and return type are preserved; classification is intentionally stricter
#           for stale, discontinuous, and malformed samples.

import math
import numbers
from dataclasses import dataclass

from .config import _clamp_ratio
from .models import AICameraBox, AICameraMotionState


# Tuned for online edge inference loops. The old 0.15 s lower bound effectively disabled
# inference on fast camera pipelines, while multi-second gaps are too stale for "recent motion".
_MOTION_UNKNOWN_MAX_GAP_S = 2.5
_MOTION_MIN_DELTA_S = 1.0 / 45.0

_RATIO_EPSILON = 1e-6
_BOX_EPSILON = 1e-4

# Continuity / association thresholds.
_MIN_IOU_CONTINUITY = 1e-3
_MAX_TRACK_JUMP_RATIO = 2.4
_MAX_TRACK_JUMP_RATIO_LONG_GAP = 1.4

# Optional detector confidence floor when a box carries confidence metadata.
_MIN_DETECTION_CONFIDENCE = 0.05

# Motion decision thresholds. These operate on dt-normalized, box-scale-normalized features.
_SCALE_RATE_STRONG = 0.11
_SCALE_RATE_WEAK = 0.05
_WALK_RATE_STRONG = 0.22
_WALK_RATE_WEAK = 0.09
_STILL_SHIFT_RATE_MAX = 0.06
_STILL_SCALE_RATE_MAX = 0.04


def _coerce_finite_float(value: object) -> float | None:
    # AUDIT-FIX: Runtime payloads can still be malformed despite type hints.
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _coerce_normalized_ratio(value: object) -> float | None:
    numeric = _coerce_finite_float(value)
    if numeric is None:
        return None
    if numeric < -_RATIO_EPSILON or numeric > 1.0 + _RATIO_EPSILON:
        return None
    return min(1.0, max(0.0, numeric))


def _is_single_person_count(value: object) -> bool:
    # Reject booleans while still accepting integer-like runtime types (for example numpy scalars).
    return isinstance(value, numbers.Integral) and not isinstance(value, bool) and int(value) == 1


def _bounded_confidence(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    bounded = _coerce_finite_float(_clamp_ratio(value, default=0.0))
    return 0.0 if bounded is None else round(bounded, 3)


def _coerce_track_id(box: AICameraBox) -> int | None:
    try:
        track_id = getattr(box, "track_id")
    except Exception:
        return None
    return int(track_id) if isinstance(track_id, numbers.Integral) and not isinstance(track_id, bool) else None


def _coerce_optional_box_confidence(box: AICameraBox) -> float | None:
    # Frontline trackers / detectors often expose one of these fields.
    for attribute_name in ("confidence", "conf", "score"):
        try:
            raw_value = getattr(box, attribute_name)
        except Exception:
            continue
        bounded = _coerce_normalized_ratio(raw_value)
        if bounded is not None:
            return bounded
    return None


@dataclass(frozen=True, slots=True)
class _NormalizedBox:
    center_x: float
    center_y: float
    area: float
    height: float
    width: float
    left: float
    top: float
    right: float
    bottom: float
    diagonal: float
    track_id: int | None
    detection_confidence: float | None


def _extract_normalized_box(box: AICameraBox) -> _NormalizedBox | None:
    # Validate all box fields before arithmetic so broken detector objects cannot trigger exceptions
    # or propagate NaN / impossible geometry downstream.
    try:
        center_x = _coerce_normalized_ratio(box.center_x)
        center_y = _coerce_normalized_ratio(box.center_y)
        area = _coerce_normalized_ratio(box.area)
        height = _coerce_normalized_ratio(box.height)
    except Exception:
        return None

    if center_x is None or center_y is None or area is None or height is None:
        return None
    if height < _BOX_EPSILON or area < _BOX_EPSILON**2:
        return None

    # area = width * height in normalized coordinates.
    width = area / max(height, _BOX_EPSILON)
    if width < _BOX_EPSILON or width > 1.0 + _RATIO_EPSILON:
        return None
    width = min(1.0, max(0.0, width))

    half_width = 0.5 * width
    half_height = 0.5 * height
    left = center_x - half_width
    right = center_x + half_width
    top = center_y - half_height
    bottom = center_y + half_height

    if (
        left < -_RATIO_EPSILON
        or right > 1.0 + _RATIO_EPSILON
        or top < -_RATIO_EPSILON
        or bottom > 1.0 + _RATIO_EPSILON
    ):
        return None

    left = max(0.0, left)
    right = min(1.0, right)
    top = max(0.0, top)
    bottom = min(1.0, bottom)

    return _NormalizedBox(
        center_x=center_x,
        center_y=center_y,
        area=area,
        height=height,
        width=width,
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        diagonal=math.hypot(width, height),
        track_id=_coerce_track_id(box),
        detection_confidence=_coerce_optional_box_confidence(box),
    )


def _compute_iou(previous_box: _NormalizedBox, current_box: _NormalizedBox) -> float:
    intersection_width = max(0.0, min(previous_box.right, current_box.right) - max(previous_box.left, current_box.left))
    intersection_height = max(0.0, min(previous_box.bottom, current_box.bottom) - max(previous_box.top, current_box.top))
    intersection = intersection_width * intersection_height
    union = max(previous_box.area + current_box.area - intersection, _BOX_EPSILON**2)
    return intersection / union


def _signed_log_ratio_rate(previous_value: float, current_value: float, delta_t: float) -> float:
    return math.log(max(current_value, _BOX_EPSILON) / max(previous_value, _BOX_EPSILON)) / delta_t


def _apply_deadband(value: float, deadband: float) -> float:
    magnitude = max(0.0, abs(value) - deadband)
    return math.copysign(magnitude, value)


@dataclass(frozen=True, slots=True)
class _MotionFeatures:
    delta_t: float
    delta_x: float
    delta_y: float
    center_distance: float
    relative_shift: float
    iou: float
    shift_rate: float
    lateral_shift_rate: float
    foot_shift_rate: float
    height_rate: float
    area_rate: float
    scale_rate: float
    continuity_score: float
    measurement_quality: float


def _extract_motion_features(
    previous_box: _NormalizedBox,
    current_box: _NormalizedBox,
    delta_t: float,
) -> _MotionFeatures:
    mean_height = max((previous_box.height + current_box.height) * 0.5, _BOX_EPSILON)
    mean_diagonal = max((previous_box.diagonal + current_box.diagonal) * 0.5, _BOX_EPSILON)

    delta_x = current_box.center_x - previous_box.center_x
    delta_y = current_box.center_y - previous_box.center_y
    center_distance = math.hypot(delta_x, delta_y)
    relative_shift = center_distance / mean_diagonal
    iou = _compute_iou(previous_box, current_box)

    # Detector jitter is not motion. Use small adaptive deadbands before converting to rates.
    shift_deadband = max(0.0025, min(0.02, 0.02 * mean_diagonal + 0.0015))
    effective_center_distance = max(0.0, center_distance - shift_deadband)
    effective_lateral_shift = max(0.0, abs(delta_x) - shift_deadband * 0.9)
    effective_foot_shift = max(0.0, abs(current_box.bottom - previous_box.bottom) - shift_deadband * 1.1)

    height_rate = _apply_deadband(
        _signed_log_ratio_rate(previous_box.height, current_box.height, delta_t),
        deadband=0.018 / delta_t,
    )
    area_rate = _apply_deadband(
        0.5 * _signed_log_ratio_rate(previous_box.area, current_box.area, delta_t),
        deadband=0.025 / delta_t,
    )
    scale_rate = 0.65 * height_rate + 0.35 * area_rate

    # IoU + relative shift approximates whether both boxes still describe the same physical person.
    continuity_score = max(
        0.0,
        min(
            1.0,
            0.55 * (1.0 if iou >= _MIN_IOU_CONTINUITY else 0.0)
            + 0.45 * (1.0 - min(relative_shift / _MAX_TRACK_JUMP_RATIO, 1.0)),
        ),
    )

    observed_confidences = [
        confidence
        for confidence in (previous_box.detection_confidence, current_box.detection_confidence)
        if confidence is not None
    ]
    measurement_quality = sum(observed_confidences) / len(observed_confidences) if observed_confidences else 1.0

    return _MotionFeatures(
        delta_t=delta_t,
        delta_x=delta_x,
        delta_y=delta_y,
        center_distance=center_distance,
        relative_shift=relative_shift,
        iou=iou,
        shift_rate=effective_center_distance / (mean_height * delta_t),
        lateral_shift_rate=effective_lateral_shift / (mean_height * delta_t),
        foot_shift_rate=effective_foot_shift / (mean_height * delta_t),
        height_rate=height_rate,
        area_rate=area_rate,
        scale_rate=scale_rate,
        continuity_score=continuity_score,
        measurement_quality=measurement_quality,
    )


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

    previous_box_values = _extract_normalized_box(previous_box)
    current_box_values = _extract_normalized_box(current_box)
    previous_ts = _coerce_finite_float(previous_observed_at)
    current_ts = _coerce_finite_float(current_observed_at)
    if previous_box_values is None or current_box_values is None or previous_ts is None or current_ts is None:
        return AICameraMotionState.UNKNOWN, None

    delta_t = current_ts - previous_ts
    if delta_t < _MOTION_MIN_DELTA_S or delta_t > _MOTION_UNKNOWN_MAX_GAP_S:
        return AICameraMotionState.UNKNOWN, None

    # If an upstream tracker provides persistent IDs, trust them as a strong continuity signal.
    if (
        previous_box_values.track_id is not None
        and current_box_values.track_id is not None
        and previous_box_values.track_id != current_box_values.track_id
    ):
        return AICameraMotionState.UNKNOWN, None

    features = _extract_motion_features(previous_box_values, current_box_values, delta_t)

    max_jump_ratio = _MAX_TRACK_JUMP_RATIO if delta_t <= 0.6 else _MAX_TRACK_JUMP_RATIO_LONG_GAP
    if features.iou <= _MIN_IOU_CONTINUITY and features.relative_shift >= max_jump_ratio:
        return AICameraMotionState.UNKNOWN, None

    if features.measurement_quality < _MIN_DETECTION_CONFIDENCE:
        return AICameraMotionState.UNKNOWN, None

    scale_alignment_ok = features.height_rate * features.area_rate >= -0.015
    approach_score = max(features.scale_rate, 0.0)
    leaving_score = max(-features.scale_rate, 0.0)
    walking_score = max(
        features.lateral_shift_rate,
        0.8 * features.shift_rate,
        0.55 * features.foot_shift_rate,
    )

    still_penalty = max(
        features.shift_rate / max(_STILL_SHIFT_RATE_MAX, _BOX_EPSILON),
        abs(features.scale_rate) / max(_STILL_SCALE_RATE_MAX, _BOX_EPSILON),
    )

    identity_bonus = (
        0.08
        if previous_box_values.track_id is not None and current_box_values.track_id is not None
        else 0.0
    )
    base_confidence = min(
        1.0,
        max(0.0, 0.58 * features.continuity_score + 0.42 * features.measurement_quality + identity_bonus),
    )

    if approach_score >= _SCALE_RATE_STRONG and walking_score <= _WALK_RATE_WEAK and scale_alignment_ok:
        confidence = _bounded_confidence(base_confidence * (0.72 + min(approach_score, 0.8) * 0.35))
        return AICameraMotionState.APPROACHING, max(confidence, 0.58)

    if leaving_score >= _SCALE_RATE_STRONG and walking_score <= _WALK_RATE_WEAK and scale_alignment_ok:
        confidence = _bounded_confidence(base_confidence * (0.72 + min(leaving_score, 0.8) * 0.35))
        return AICameraMotionState.LEAVING, max(confidence, 0.58)

    if walking_score >= _WALK_RATE_STRONG and max(approach_score, leaving_score) <= _SCALE_RATE_STRONG * 1.5:
        confidence = _bounded_confidence(base_confidence * (0.70 + min(walking_score, 1.2) * 0.24))
        return AICameraMotionState.WALKING, max(confidence, 0.56)

    if features.shift_rate <= _STILL_SHIFT_RATE_MAX and abs(features.scale_rate) <= _STILL_SCALE_RATE_MAX:
        confidence = _bounded_confidence(base_confidence * (0.84 + 0.16 * max(0.0, 1.0 - still_penalty)))
        return AICameraMotionState.STILL, max(confidence, 0.55)

    if approach_score >= max(_SCALE_RATE_WEAK, walking_score * 0.95) and scale_alignment_ok:
        confidence = _bounded_confidence(base_confidence * (0.64 + min(approach_score, 0.7) * 0.28))
        return AICameraMotionState.APPROACHING, max(confidence, 0.51)

    if leaving_score >= max(_SCALE_RATE_WEAK, walking_score * 0.95) and scale_alignment_ok:
        confidence = _bounded_confidence(base_confidence * (0.64 + min(leaving_score, 0.7) * 0.28))
        return AICameraMotionState.LEAVING, max(confidence, 0.51)

    if walking_score >= _WALK_RATE_WEAK:
        confidence = _bounded_confidence(base_confidence * (0.64 + min(walking_score, 1.1) * 0.22))
        return AICameraMotionState.WALKING, max(confidence, 0.50)

    confidence = _bounded_confidence(base_confidence * (0.70 + 0.10 * max(0.0, 1.0 - still_penalty)))
    return AICameraMotionState.STILL, max(confidence, 0.46)


__all__ = ["infer_motion_state"]