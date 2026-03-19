"""Extract stable pose-derived features from sparse keypoint maps."""

from __future__ import annotations

import math

from .config import _clamp_ratio
from .models import AICameraBox


_KEYPOINT_COUNT = 17
_DEFAULT_MIN_SCORE = 0.20  # AUDIT-FIX(#3): Centralize the default visibility threshold so invalid caller input degrades consistently.


def _coerce_finite_float(value: object) -> float | None:
    """Return one finite float, or ``None`` when the value is not safe to use."""

    if isinstance(value, (bool, str, bytes, bytearray)):  # AUDIT-FIX(#1): Reject silent bool/string coercions from malformed detector payloads.
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):  # AUDIT-FIX(#1): Prevent malformed model values from bubbling out as exceptions.
        return None
    if not math.isfinite(number):  # AUDIT-FIX(#4): Reject NaN/inf before downstream math.
        return None
    return number


def _normalized_threshold(min_score: float) -> float:
    """Return one conservative score threshold in ``[0, 1]``."""

    threshold = _coerce_finite_float(min_score)  # AUDIT-FIX(#3): Sanitize caller-provided thresholds instead of trusting runtime type hints.
    if threshold is None:
        return _DEFAULT_MIN_SCORE
    return _clamp_ratio(threshold, default=_DEFAULT_MIN_SCORE)


def _coerce_joint(joint: object) -> tuple[float, float, float] | None:
    """Return one normalized joint tuple only when all three fields are finite and in range."""

    if isinstance(joint, (str, bytes, bytearray)):  # AUDIT-FIX(#3): Reject string-like payloads that would index into characters.
        return None
    try:
        x_raw = joint[0]  # type: ignore[index]
        y_raw = joint[1]  # type: ignore[index]
        score_raw = joint[2]  # type: ignore[index]
    except (TypeError, IndexError, KeyError):
        return None

    x = _coerce_finite_float(x_raw)
    y = _coerce_finite_float(y_raw)
    score = _coerce_finite_float(score_raw)
    if x is None or y is None or score is None:
        return None
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= score <= 1.0):  # AUDIT-FIX(#3): Reject malformed normalized joints instead of clamping them into plausible positions.
        return None
    return (x, y, score)


def _box_ratio(box: object, attr: str, *, default: float) -> float:
    """Read one fallback-box ratio safely with a conservative default."""

    safe_default = _clamp_ratio(default, default=0.0)
    value = _coerce_finite_float(getattr(box, attr, default))  # AUDIT-FIX(#5): Guard against malformed or partially populated fallback boxes.
    if value is None:
        return safe_default
    return _clamp_ratio(value, default=safe_default)


def parse_keypoints(raw: list[float], *, frame_width: int, frame_height: int) -> dict[int, tuple[float, float, float]]:
    """Convert one flat keypoint list into normalized ``index -> (x, y, score)`` data."""

    if isinstance(raw, (str, bytes, bytearray)):  # AUDIT-FIX(#1): Reject string-like payloads early so character iteration cannot masquerade as numeric keypoints.
        return {}
    try:
        raw_values = list(raw)
    except TypeError:  # AUDIT-FIX(#1): Non-iterable detector payloads now degrade to no keypoints instead of raising.
        return {}

    width = _coerce_finite_float(frame_width)
    height = _coerce_finite_float(frame_height)
    if width is None or height is None or width <= 0.0 or height <= 0.0:  # AUDIT-FIX(#1): Invalid frame geometry would make normalization meaningless, so fail closed.
        return {}

    parsed: dict[int, tuple[float, float, float]] = {}
    expected = min(len(raw_values) // 3, _KEYPOINT_COUNT)
    for index in range(expected):
        x_raw = _coerce_finite_float(raw_values[index * 3 + 0])
        y_raw = _coerce_finite_float(raw_values[index * 3 + 1])
        score_raw = _coerce_finite_float(raw_values[index * 3 + 2])
        if x_raw is None or y_raw is None or score_raw is None:  # AUDIT-FIX(#1): Malformed triplets become invisible joints instead of crashing the pose pipeline.
            parsed[index] = (0.0, 0.0, 0.0)
            continue

        x = _clamp_ratio(x_raw / width, default=0.0)
        y = _clamp_ratio(y_raw / height, default=0.0)
        score = _clamp_ratio(score_raw, default=0.0)
        parsed[index] = (x, y, score)
    return parsed


def attention_score(
    keypoints: dict[int, tuple[float, float, float]],
    *,
    fallback_box: AICameraBox,
) -> float:
    """Return one conservative attention score from pose and coarse centering."""

    nose = visible_joint(keypoints, 0)  # AUDIT-FIX(#3): Reuse sanitized joint selection so malformed keypoints cannot drive attention scoring.
    left_eye = visible_joint(keypoints, 1)  # AUDIT-FIX(#3): Reuse sanitized joint selection so malformed keypoints cannot drive attention scoring.
    right_eye = visible_joint(keypoints, 2)  # AUDIT-FIX(#3): Reuse sanitized joint selection so malformed keypoints cannot drive attention scoring.
    left_shoulder = visible_joint(keypoints, 5)  # AUDIT-FIX(#3): Reuse sanitized joint selection so malformed keypoints cannot drive attention scoring.
    right_shoulder = visible_joint(keypoints, 6)  # AUDIT-FIX(#3): Reuse sanitized joint selection so malformed keypoints cannot drive attention scoring.

    center_x = _box_ratio(fallback_box, "center_x", default=0.0)  # AUDIT-FIX(#5): Invalid fallback boxes now fail low instead of raising or granting perfect centering.
    center_alignment = max(0.0, 1.0 - abs(center_x - 0.5) / 0.5)
    shoulder_alignment = 0.0
    head_alignment = 0.0

    if left_shoulder is not None and right_shoulder is not None:
        shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2.0
        shoulder_span = abs(right_shoulder[0] - left_shoulder[0])
        if nose is not None and shoulder_span > 0.02:
            head_alignment = max(0.0, 1.0 - abs(nose[0] - shoulder_center_x) / shoulder_span)
        shoulder_alignment = max(0.0, 1.0 - abs(shoulder_center_x - 0.5) / 0.5)
    elif nose is not None and left_eye is not None and right_eye is not None:
        eye_center_x = (left_eye[0] + right_eye[0]) / 2.0
        eye_span = abs(right_eye[0] - left_eye[0])
        if eye_span > 0.01:
            head_alignment = max(0.0, 1.0 - abs(nose[0] - eye_center_x) / eye_span)

    score = 0.45 * head_alignment + 0.25 * shoulder_alignment + 0.30 * center_alignment
    return round(_clamp_ratio(score, default=0.0), 3)


def hand_near_camera(
    keypoints: dict[int, tuple[float, float, float]],
    *,
    fallback_box: AICameraBox,
) -> bool:
    """Return whether one wrist is plausibly near the device."""

    top = _box_ratio(fallback_box, "top", default=0.0)  # AUDIT-FIX(#5): Sanitize fallback box position before deriving hand-gating thresholds.
    wrist_y_threshold = min(0.60, max(0.35, top + 0.08))  # AUDIT-FIX(#2): Cap the permissive upper bound so low or corrupt boxes do not turn most centered wrists into camera-near hits.
    for index in (9, 10):
        wrist = visible_joint(keypoints, index)  # AUDIT-FIX(#3): Ignore malformed wrist tuples rather than trusting raw dict values.
        if wrist is None:
            continue
        if wrist[1] <= wrist_y_threshold and abs(wrist[0] - 0.5) <= 0.22:
            return True
    return False


def visible_joint(
    keypoints: dict[int, tuple[float, float, float]],
    index: int,
    *,
    min_score: float = 0.20,
) -> tuple[float, float, float] | None:
    """Return one keypoint only when its score clears the minimum threshold."""

    threshold = _normalized_threshold(min_score)  # AUDIT-FIX(#3): Normalize malformed thresholds so NaN or invalid caller input cannot bypass visibility gating.
    joint = _coerce_joint(keypoints.get(index))  # AUDIT-FIX(#3): Reject malformed, non-finite, or out-of-range joints before visibility checks.
    if joint is None or joint[2] < threshold:
        return None
    return joint


def best_visible_joint(
    keypoints: dict[int, tuple[float, float, float]],
    indices: tuple[int, ...],
    *,
    min_score: float = 0.20,
) -> tuple[float, float, float] | None:
    """Return the highest-confidence visible joint from one candidate set."""

    visible = [
        joint
        for index in indices
        if (joint := visible_joint(keypoints, index, min_score=min_score)) is not None
    ]
    if not visible:
        return None
    return max(visible, key=lambda item: item[2])


def strong_keypoint_count(
    keypoints: dict[int, tuple[float, float, float]],
    *,
    min_score: float = 0.20,
) -> int:
    """Count how many keypoints are strong enough to support coarse inference."""

    threshold = _normalized_threshold(min_score)  # AUDIT-FIX(#3): Apply the same sanitized threshold logic used by ``visible_joint``.
    return sum(
        1
        for joint in keypoints.values()
        if (normalized_joint := _coerce_joint(joint)) is not None and normalized_joint[2] >= threshold
    )


def support_pose_confidence(
    raw_score: float,
    keypoints: dict[int, tuple[float, float, float]],
    *,
    fallback_box: AICameraBox,
) -> float:
    """Normalize pose confidence using keypoint support instead of raw score alone."""

    raw_value = _coerce_finite_float(raw_score)  # AUDIT-FIX(#4): Invalid raw confidence inputs now degrade to zero instead of relying on implicit coercion.
    normalized_raw = _clamp_ratio(raw_value if raw_value is not None else 0.0, default=0.0)
    if normalized_raw <= 0.0:
        return 0.0

    strong_fraction = min(1.0, strong_keypoint_count(keypoints) / float(_KEYPOINT_COUNT))  # AUDIT-FIX(#3): Clamp support to the known 17-keypoint model budget.
    shoulders = sum(1 for index in (5, 6) if visible_joint(keypoints, index) is not None)
    hips = sum(1 for index in (11, 12) if visible_joint(keypoints, index) is not None)
    wrists = sum(1 for index in (9, 10) if visible_joint(keypoints, index) is not None)
    legs = sum(1 for index in (13, 14, 15, 16) if visible_joint(keypoints, index) is not None)
    face = sum(1 for index in (0, 1, 2) if visible_joint(keypoints, index) is not None)

    height = _box_ratio(fallback_box, "height", default=0.0)  # AUDIT-FIX(#5): Invalid box geometry no longer grants structure-support bonuses.
    width = _box_ratio(fallback_box, "width", default=1.0)  # AUDIT-FIX(#5): Invalid box geometry no longer grants structure-support bonuses.
    structure_support = 0.0
    if shoulders > 0 and hips > 0:
        structure_support += 0.35
    elif shoulders > 0 or hips > 0:
        structure_support += 0.20
    if legs >= 2:
        structure_support += 0.20
    if wrists >= 1:
        structure_support += 0.10
    if face >= 1:
        structure_support += 0.10
    if height >= 0.45 and width <= 0.60:
        structure_support += 0.15

    support_score = max(strong_fraction, min(1.0, structure_support))
    return round(_clamp_ratio(min(normalized_raw, support_score), default=0.0), 3)


def landmark_score(landmark: object) -> float:
    """Estimate one landmark confidence from visibility and presence when available."""

    if landmark is None:  # AUDIT-FIX(#6): A missing landmark must fail closed instead of being treated as perfect confidence.
        return 0.0

    has_visibility = hasattr(landmark, "visibility")
    has_presence = hasattr(landmark, "presence")
    scores = []
    for attr in ("visibility", "presence"):
        if not hasattr(landmark, attr):
            continue
        value = _coerce_finite_float(getattr(landmark, attr))  # AUDIT-FIX(#4): Safe coercion avoids TypeError/ValueError on malformed landmark fields.
        if value is None:
            continue
        scores.append(_clamp_ratio(value, default=0.0))

    if scores:
        return round(sum(scores) / float(len(scores)), 3)
    if has_visibility or has_presence:  # AUDIT-FIX(#6): Explicit but unusable confidence fields should be treated as unknown/unsafe, not as perfect confidence.
        return 0.0

    has_coordinates = any(hasattr(landmark, attr) for attr in ("x", "y", "z")) or isinstance(landmark, (tuple, list))  # AUDIT-FIX(#6): Only default to full confidence for landmark-like objects whose backend simply omits confidence fields.
    return 1.0 if has_coordinates else 0.0


__all__ = [
    "attention_score",
    "best_visible_joint",
    "hand_near_camera",
    "landmark_score",
    "parse_keypoints",
    "strong_keypoint_count",
    "support_pose_confidence",
    "visible_joint",
]