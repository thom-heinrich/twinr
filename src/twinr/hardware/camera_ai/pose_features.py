# CHANGELOG: 2026-03-28
# BUG-1: parse_keypoints no longer clamps off-frame coordinates into plausible visible joints; invalid triplets now fail closed.
# BUG-2: landmark_score no longer treats arbitrary/empty tuples/lists as perfect-confidence landmarks.
# SEC-1: parse_keypoints no longer materialize unbounded iterables; parsing is bounded to the active schema budget to avoid Pi-4 memory/CPU denial-of-service.
# IMP-1: Added pose-schema awareness (COCO-17 + BlazePose-33), joint-name resolution, and backend-compatible parsing for flat triplets, nested triplets, and landmark objects.
# IMP-2: Added confidence-aware geometry, better support scoring, and optional low-cost temporal smoothing/hysteresis helpers for edge deployments.

"""Extract stable pose-derived features from sparse keypoint maps."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .config import _clamp_ratio
from .models import AICameraBox


_DEFAULT_MIN_SCORE = 0.20


@dataclass(frozen=True)
class PoseSchema:
    """Skeleton metadata used to resolve joint names and coverage expectations."""

    name: str
    keypoint_count: int
    joints: dict[str, int]
    face: tuple[int, ...]
    shoulders: tuple[int, ...]
    wrists: tuple[int, ...]
    hips: tuple[int, ...]
    legs: tuple[int, ...]


COCO17_SCHEMA = PoseSchema(
    name="coco17",
    keypoint_count=17,
    joints={
        "nose": 0,
        "left_eye": 1,
        "right_eye": 2,
        "left_ear": 3,
        "right_ear": 4,
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_elbow": 7,
        "right_elbow": 8,
        "left_wrist": 9,
        "right_wrist": 10,
        "left_hip": 11,
        "right_hip": 12,
        "left_knee": 13,
        "right_knee": 14,
        "left_ankle": 15,
        "right_ankle": 16,
    },
    face=(0, 1, 2),
    shoulders=(5, 6),
    wrists=(9, 10),
    hips=(11, 12),
    legs=(13, 14, 15, 16),
)

BLAZEPOSE33_SCHEMA = PoseSchema(
    name="blazepose33",
    keypoint_count=33,
    joints={
        "nose": 0,
        "left_eye_inner": 1,
        "left_eye": 2,
        "left_eye_outer": 3,
        "right_eye_inner": 4,
        "right_eye": 5,
        "right_eye_outer": 6,
        "left_ear": 7,
        "right_ear": 8,
        "mouth_left": 9,
        "mouth_right": 10,
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_pinky": 17,
        "right_pinky": 18,
        "left_index": 19,
        "right_index": 20,
        "left_thumb": 21,
        "right_thumb": 22,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28,
        "left_heel": 29,
        "right_heel": 30,
        "left_foot_index": 31,
        "right_foot_index": 32,
    },
    face=(0, 2, 5, 7, 8, 9, 10),
    shoulders=(11, 12),
    wrists=(15, 16),
    hips=(23, 24),
    legs=(25, 26, 27, 28, 29, 30, 31, 32),
)

_SCHEMAS = {
    COCO17_SCHEMA.name: COCO17_SCHEMA,
    "coco": COCO17_SCHEMA,
    "coco_17": COCO17_SCHEMA,
    BLAZEPOSE33_SCHEMA.name: BLAZEPOSE33_SCHEMA,
    "blazepose": BLAZEPOSE33_SCHEMA,
    "mediapipe33": BLAZEPOSE33_SCHEMA,
    "mediapipe_pose": BLAZEPOSE33_SCHEMA,
    "pose_landmarker": BLAZEPOSE33_SCHEMA,
}

_MISSING = object()


@dataclass
class PoseFeatureState:
    """Low-cost temporal state for smoothing jittery per-frame pose features."""

    attention_ema: float = 0.0
    attention_initialized: bool = False
    hand_near_camera_ema: float = 0.0
    hand_near_camera_active: bool = False
    hand_initialized: bool = False


def _coerce_finite_float(value: object) -> float | None:
    """Return one finite float, or ``None`` when the value is not safe to use."""

    if isinstance(value, (bool, str, bytes, bytearray)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _normalized_threshold(min_score: float) -> float:
    """Return one conservative score threshold in ``[0, 1]``."""

    threshold = _coerce_finite_float(min_score)
    if threshold is None:
        return _DEFAULT_MIN_SCORE
    return _clamp_ratio(threshold, default=_DEFAULT_MIN_SCORE)


def _coerce_joint(joint: object) -> tuple[float, float, float] | None:
    """Return one normalized joint tuple only when all three fields are finite and in range."""

    if isinstance(joint, (str, bytes, bytearray)):
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
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= score <= 1.0):
        return None
    return (x, y, score)


def _box_ratio(box: object, attr: str, *, default: float) -> float:
    """Read one fallback-box ratio safely with a conservative default."""

    safe_default = _clamp_ratio(default, default=0.0)
    value = _coerce_finite_float(getattr(box, attr, default))
    if value is None:
        return safe_default
    return _clamp_ratio(value, default=safe_default)


def _coerce_schema(schema: str | PoseSchema | None) -> PoseSchema:
    if isinstance(schema, PoseSchema):
        return schema
    if schema is None:
        return COCO17_SCHEMA
    if isinstance(schema, str):
        normalized = schema.strip().lower()
        if normalized == "auto":
            return COCO17_SCHEMA
        resolved = _SCHEMAS.get(normalized)
        if resolved is not None:
            return resolved
    return COCO17_SCHEMA


def infer_pose_schema(keypoints: Mapping[int, object] | int | None) -> PoseSchema:
    """Infer one likely pose schema from a keypoint map or expected keypoint count."""

    if isinstance(keypoints, int):
        count = max(0, keypoints)
        if count >= BLAZEPOSE33_SCHEMA.keypoint_count:
            return BLAZEPOSE33_SCHEMA
        return COCO17_SCHEMA

    if not keypoints:
        return COCO17_SCHEMA

    integer_indices = [index for index in keypoints.keys() if isinstance(index, int) and index >= 0]
    max_index = max(integer_indices, default=-1)
    if max_index >= 23 or any(index in keypoints for index in (23, 24, 31, 32)):
        return BLAZEPOSE33_SCHEMA
    if max_index >= 16:
        return COCO17_SCHEMA
    return COCO17_SCHEMA


def _resolve_schema(
    schema: str | PoseSchema | None,
    *,
    keypoints: Mapping[int, object] | None = None,
    raw: object | None = None,
    flat_hint: bool | None = None,
) -> PoseSchema:
    explicit = _coerce_schema(schema)
    if isinstance(schema, str) and schema.strip().lower() == "auto":
        if keypoints is not None:
            return infer_pose_schema(keypoints)
        if raw is not None:
            try:
                raw_len = len(raw)  # type: ignore[arg-type]
            except Exception:
                raw_len = None
            if raw_len is not None:
                if flat_hint is True and raw_len >= BLAZEPOSE33_SCHEMA.keypoint_count * 3:
                    return BLAZEPOSE33_SCHEMA
                if flat_hint is False and raw_len >= BLAZEPOSE33_SCHEMA.keypoint_count:
                    return BLAZEPOSE33_SCHEMA
    return explicit


def _resolve_joint_index(
    index: int | str,
    *,
    schema: PoseSchema,
) -> int | None:
    if isinstance(index, bool):
        return None
    if isinstance(index, str):
        return schema.joints.get(index)
    numeric_index = _coerce_finite_float(index)
    if numeric_index is None or not numeric_index.is_integer():
        return None
    resolved = int(numeric_index)
    if resolved < 0 or resolved >= schema.keypoint_count:
        return None
    return resolved


def _landmark_like_coordinates(landmark: object) -> tuple[float, float] | None:
    if landmark is None:
        return None

    x = _coerce_finite_float(getattr(landmark, "x", None))
    y = _coerce_finite_float(getattr(landmark, "y", None))
    if x is not None and y is not None:
        return (x, y)

    if isinstance(landmark, (str, bytes, bytearray)):
        return None
    try:
        x_raw = landmark[0]  # type: ignore[index]
        y_raw = landmark[1]  # type: ignore[index]
    except (TypeError, IndexError, KeyError):
        return None

    x = _coerce_finite_float(x_raw)
    y = _coerce_finite_float(y_raw)
    if x is None or y is None:
        return None
    return (x, y)


def _landmark_confidence(landmark: object, *, default: float = 1.0) -> float:
    scores: list[float] = []
    for attr in ("visibility", "presence", "score", "confidence"):
        if not hasattr(landmark, attr):
            continue
        value = _coerce_finite_float(getattr(landmark, attr))
        if value is None:
            continue
        scores.append(_clamp_ratio(value, default=0.0))

    if not scores:
        try:
            indexed_score = _coerce_finite_float(landmark[2])  # type: ignore[index]
        except (TypeError, IndexError, KeyError):
            indexed_score = None
        if indexed_score is not None:
            scores.append(_clamp_ratio(indexed_score, default=0.0))

    if not scores:
        return _clamp_ratio(default, default=0.0)
    return min(scores)


def _normalize_xy(
    x_raw: float,
    y_raw: float,
    *,
    frame_width: float,
    frame_height: float,
    normalized: bool,
) -> tuple[float, float] | None:
    if normalized:
        if not (0.0 <= x_raw <= 1.0 and 0.0 <= y_raw <= 1.0):
            return None
        return (x_raw, y_raw)

    if not (0.0 <= x_raw <= frame_width and 0.0 <= y_raw <= frame_height):
        return None
    return (x_raw / frame_width, y_raw / frame_height)


def _iter_limited(items: Iterable[object], *, limit: int) -> Iterable[object]:
    count = 0
    for item in items:
        if count >= limit:
            break
        yield item
        count += 1


def _coerce_nested_joint(
    item: object,
    *,
    frame_width: float,
    frame_height: float,
    normalized: bool,
) -> tuple[float, float, float] | None:
    coordinates = _landmark_like_coordinates(item)
    if coordinates is None:
        return None

    x_raw, y_raw = coordinates
    normalized_xy = _normalize_xy(
        x_raw,
        y_raw,
        frame_width=frame_width,
        frame_height=frame_height,
        normalized=normalized,
    )
    if normalized_xy is None:
        return None

    score = _landmark_confidence(item)
    return (normalized_xy[0], normalized_xy[1], score)


def _sanitize_visible_keypoints(
    keypoints: Mapping[int, object],
    *,
    schema: PoseSchema,
    min_score: float,
) -> dict[int, tuple[float, float, float]]:
    threshold = _normalized_threshold(min_score)
    cleaned: dict[int, tuple[float, float, float]] = {}
    for index in range(schema.keypoint_count):
        joint = _coerce_joint(keypoints.get(index))
        if joint is None or joint[2] < threshold:
            continue
        cleaned[index] = joint
    return cleaned


def _joint(
    cleaned: Mapping[int, tuple[float, float, float]],
    index: int | str,
    *,
    schema: PoseSchema,
) -> tuple[float, float, float] | None:
    resolved = _resolve_joint_index(index, schema=schema)
    if resolved is None:
        return None
    return cleaned.get(resolved)


def _geometry_alignment(center_x: float, reference_x: float, tolerance: float) -> float:
    if tolerance <= 0.0:
        return 0.0
    return max(0.0, 1.0 - abs(center_x - reference_x) / tolerance)


def _ema(previous: float, current: float, *, alpha: float) -> float:
    clamped_alpha = _clamp_ratio(alpha, default=0.55)
    return (clamped_alpha * current) + ((1.0 - clamped_alpha) * previous)


def _smooth_attention(score: float, state: PoseFeatureState | None) -> float:
    if state is None:
        return score
    if not state.attention_initialized:
        state.attention_ema = score
        state.attention_initialized = True
        return score
    state.attention_ema = _ema(state.attention_ema, score, alpha=0.55)
    return state.attention_ema


def _smooth_boolean_signal(
    signal: float,
    *,
    state: PoseFeatureState | None,
    previous_value: bool,
) -> bool:
    if state is None:
        return signal >= 0.60

    if not state.hand_initialized:
        state.hand_near_camera_ema = signal
        state.hand_near_camera_active = signal >= 0.60
        state.hand_initialized = True
        return state.hand_near_camera_active

    alpha = 0.70 if signal > state.hand_near_camera_ema else 0.40
    state.hand_near_camera_ema = _ema(state.hand_near_camera_ema, signal, alpha=alpha)

    on_threshold = 0.62
    off_threshold = 0.42
    if previous_value:
        state.hand_near_camera_active = state.hand_near_camera_ema >= off_threshold
    else:
        state.hand_near_camera_active = state.hand_near_camera_ema >= on_threshold
    return state.hand_near_camera_active


def parse_keypoints(
    raw: Iterable[object],
    *,
    frame_width: int,
    frame_height: int,
    schema: str | PoseSchema = "coco17",
    normalized: bool | None = None,
) -> dict[int, tuple[float, float, float]]:
    """Convert raw keypoint payloads into normalized ``index -> (x, y, score)`` data."""

    if isinstance(raw, (str, bytes, bytearray)):
        return {}

    width = _coerce_finite_float(frame_width)
    height = _coerce_finite_float(frame_height)
    if width is None or height is None or width <= 0.0 or height <= 0.0:
        return {}

    try:
        iterator = iter(raw)
    except TypeError:
        return {}

    first_item = next(iterator, _MISSING)
    if first_item is _MISSING:
        return {}

    first_number = _coerce_finite_float(first_item)
    flat_input = first_number is not None
    resolved_schema = _resolve_schema(schema, raw=raw, flat_hint=flat_input)
    limit = resolved_schema.keypoint_count

    parsed: dict[int, tuple[float, float, float]] = {}

    if flat_input:
        numeric_values = [first_number]
        for item in _iter_limited(iterator, limit=(limit * 3) - 1):
            numeric_values.append(_coerce_finite_float(item))
        triplet_count = min(len(numeric_values) // 3, limit)

        normalized_input = False if normalized is None else bool(normalized)
        for index in range(triplet_count):
            x_raw = numeric_values[index * 3 + 0]
            y_raw = numeric_values[index * 3 + 1]
            score_raw = numeric_values[index * 3 + 2]
            if x_raw is None or y_raw is None or score_raw is None:
                parsed[index] = (0.0, 0.0, 0.0)
                continue

            if not (0.0 <= score_raw <= 1.0):
                parsed[index] = (0.0, 0.0, 0.0)
                continue

            normalized_xy = _normalize_xy(
                x_raw,
                y_raw,
                frame_width=width,
                frame_height=height,
                normalized=normalized_input,
            )
            if normalized_xy is None:
                parsed[index] = (0.0, 0.0, 0.0)
                continue

            parsed[index] = (normalized_xy[0], normalized_xy[1], score_raw)
        return parsed

    nested_items = [first_item]
    nested_items.extend(_iter_limited(iterator, limit=limit - 1))

    landmark_like = hasattr(first_item, "x") and hasattr(first_item, "y")
    if normalized is None:
        if landmark_like:
            normalized_input = True
        else:
            first_coordinates = _landmark_like_coordinates(first_item)
            normalized_input = bool(
                first_coordinates is not None
                and 0.0 <= first_coordinates[0] <= 1.0
                and 0.0 <= first_coordinates[1] <= 1.0
            )
    else:
        normalized_input = bool(normalized)

    for index, item in enumerate(nested_items[:limit]):
        joint = _coerce_nested_joint(
            item,
            frame_width=width,
            frame_height=height,
            normalized=normalized_input,
        )
        if joint is None:
            parsed[index] = (0.0, 0.0, 0.0)
            continue
        parsed[index] = joint
    return parsed


def attention_score(
    keypoints: dict[int, tuple[float, float, float]],
    *,
    fallback_box: AICameraBox,
    min_score: float = _DEFAULT_MIN_SCORE,
    schema: str | PoseSchema = "auto",
    state: PoseFeatureState | None = None,
) -> float:
    """Return one conservative attention score from pose and coarse centering."""

    resolved_schema = _resolve_schema(schema, keypoints=keypoints)
    cleaned = _sanitize_visible_keypoints(keypoints, schema=resolved_schema, min_score=min_score)

    nose = _joint(cleaned, "nose", schema=resolved_schema)
    left_eye = _joint(cleaned, "left_eye", schema=resolved_schema)
    right_eye = _joint(cleaned, "right_eye", schema=resolved_schema)
    left_shoulder = _joint(cleaned, "left_shoulder", schema=resolved_schema)
    right_shoulder = _joint(cleaned, "right_shoulder", schema=resolved_schema)

    center_x = _box_ratio(fallback_box, "center_x", default=0.0)
    center_alignment = max(0.0, 1.0 - abs(center_x - 0.5) / 0.5)

    head_alignment = 0.0
    shoulder_alignment = 0.0

    if left_shoulder is not None and right_shoulder is not None:
        shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2.0
        shoulder_span = abs(right_shoulder[0] - left_shoulder[0])
        shoulder_conf = min(left_shoulder[2], right_shoulder[2])
        shoulder_alignment = _geometry_alignment(shoulder_center_x, 0.5, 0.5) * shoulder_conf

        if nose is not None and shoulder_span >= 0.03:
            head_alignment = (
                _geometry_alignment(nose[0], shoulder_center_x, max(shoulder_span * 0.9, 0.06))
                * min(nose[2], shoulder_conf)
            )
    elif nose is not None and left_eye is not None and right_eye is not None:
        eye_center_x = (left_eye[0] + right_eye[0]) / 2.0
        eye_span = abs(right_eye[0] - left_eye[0])
        if eye_span >= 0.015:
            head_alignment = (
                _geometry_alignment(nose[0], eye_center_x, max(eye_span * 0.9, 0.03))
                * min(nose[2], left_eye[2], right_eye[2])
            )
    elif nose is not None:
        head_alignment = center_alignment * nose[2] * 0.60

    score = (0.50 * head_alignment) + (0.20 * shoulder_alignment) + (0.30 * center_alignment)
    smoothed = _smooth_attention(_clamp_ratio(score, default=0.0), state)
    return round(_clamp_ratio(smoothed, default=0.0), 3)


def _hand_near_camera_signal(
    cleaned: Mapping[int, tuple[float, float, float]],
    *,
    schema: PoseSchema,
    fallback_box: AICameraBox,
) -> float:
    top = _box_ratio(fallback_box, "top", default=0.0)
    width = _box_ratio(fallback_box, "width", default=0.0)
    center_x = _box_ratio(fallback_box, "center_x", default=0.5)

    left_shoulder = _joint(cleaned, "left_shoulder", schema=schema)
    right_shoulder = _joint(cleaned, "right_shoulder", schema=schema)
    if left_shoulder is not None and right_shoulder is not None:
        target_center_x = (left_shoulder[0] + right_shoulder[0]) / 2.0
        shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
    else:
        target_center_x = center_x
        shoulder_center_y = None

    wrist_y_threshold = min(0.62, max(0.32, top + 0.10))
    if shoulder_center_y is not None:
        wrist_y_threshold = min(wrist_y_threshold, shoulder_center_y + 0.14)

    horizontal_limit = min(0.30, max(0.18, width * 0.55))

    best_signal = 0.0
    for wrist_name in ("left_wrist", "right_wrist"):
        wrist = _joint(cleaned, wrist_name, schema=schema)
        if wrist is None:
            continue

        vertical_score = 1.0 if wrist[1] <= wrist_y_threshold else max(
            0.0, 1.0 - ((wrist[1] - wrist_y_threshold) / 0.20)
        )
        horizontal_score = _geometry_alignment(wrist[0], target_center_x, horizontal_limit)
        wrist_signal = wrist[2] * min(vertical_score, horizontal_score)
        best_signal = max(best_signal, wrist_signal)
    return _clamp_ratio(best_signal, default=0.0)


def hand_near_camera(
    keypoints: dict[int, tuple[float, float, float]],
    *,
    fallback_box: AICameraBox,
    min_score: float = _DEFAULT_MIN_SCORE,
    schema: str | PoseSchema = "auto",
    state: PoseFeatureState | None = None,
) -> bool:
    """Return whether one wrist is plausibly near the device."""

    resolved_schema = _resolve_schema(schema, keypoints=keypoints)
    cleaned = _sanitize_visible_keypoints(keypoints, schema=resolved_schema, min_score=min_score)
    signal = _hand_near_camera_signal(cleaned, schema=resolved_schema, fallback_box=fallback_box)
    previous_value = state.hand_near_camera_active if state is not None else False
    return _smooth_boolean_signal(signal, state=state, previous_value=previous_value)


def visible_joint(
    keypoints: dict[int, tuple[float, float, float]],
    index: int | str,
    *,
    min_score: float = _DEFAULT_MIN_SCORE,
    schema: str | PoseSchema = "auto",
) -> tuple[float, float, float] | None:
    """Return one keypoint only when its score clears the minimum threshold."""

    resolved_schema = _resolve_schema(schema, keypoints=keypoints)
    resolved_index = _resolve_joint_index(index, schema=resolved_schema)
    if resolved_index is None:
        return None

    threshold = _normalized_threshold(min_score)
    joint = _coerce_joint(keypoints.get(resolved_index))
    if joint is None or joint[2] < threshold:
        return None
    return joint


def best_visible_joint(
    keypoints: dict[int, tuple[float, float, float]],
    indices: tuple[int | str, ...],
    *,
    min_score: float = _DEFAULT_MIN_SCORE,
    schema: str | PoseSchema = "auto",
) -> tuple[float, float, float] | None:
    """Return the highest-confidence visible joint from one candidate set."""

    visible = [
        joint
        for index in indices
        if (joint := visible_joint(keypoints, index, min_score=min_score, schema=schema)) is not None
    ]
    if not visible:
        return None
    return max(visible, key=lambda item: item[2])


def strong_keypoint_count(
    keypoints: dict[int, tuple[float, float, float]],
    *,
    min_score: float = _DEFAULT_MIN_SCORE,
    schema: str | PoseSchema = "auto",
) -> int:
    """Count how many keypoints are strong enough to support coarse inference."""

    resolved_schema = _resolve_schema(schema, keypoints=keypoints)
    threshold = _normalized_threshold(min_score)
    return sum(
        1
        for index in range(resolved_schema.keypoint_count)
        if (joint := _coerce_joint(keypoints.get(index))) is not None and joint[2] >= threshold
    )


def support_pose_confidence(
    raw_score: float,
    keypoints: dict[int, tuple[float, float, float]],
    *,
    fallback_box: AICameraBox,
    min_score: float = _DEFAULT_MIN_SCORE,
    schema: str | PoseSchema = "auto",
) -> float:
    """Normalize pose confidence using keypoint support instead of raw score alone."""

    raw_value = _coerce_finite_float(raw_score)
    normalized_raw = _clamp_ratio(raw_value if raw_value is not None else 0.0, default=0.0)
    if normalized_raw <= 0.0:
        return 0.0

    resolved_schema = _resolve_schema(schema, keypoints=keypoints)
    cleaned = _sanitize_visible_keypoints(keypoints, schema=resolved_schema, min_score=min_score)

    strong_fraction = len(cleaned) / float(resolved_schema.keypoint_count)

    shoulders = sum(1 for index in resolved_schema.shoulders if index in cleaned)
    hips = sum(1 for index in resolved_schema.hips if index in cleaned)
    wrists = sum(1 for index in resolved_schema.wrists if index in cleaned)
    legs = sum(1 for index in resolved_schema.legs if index in cleaned)
    face = sum(1 for index in resolved_schema.face if index in cleaned)

    torso_support = 0.0
    if shoulders == len(resolved_schema.shoulders) and hips == len(resolved_schema.hips):
        torso_support = 1.0
    elif shoulders > 0 and hips > 0:
        torso_support = 0.75
    elif shoulders > 0 or hips > 0:
        torso_support = 0.45

    width = _box_ratio(fallback_box, "width", default=0.0)
    height = _box_ratio(fallback_box, "height", default=0.0)
    box_plausibility = 1.0 if (0.10 <= width <= 0.85 and 0.10 <= height <= 1.0) else 0.0

    support = (
        0.45 * strong_fraction
        + 0.25 * torso_support
        + 0.10 * (wrists / max(1, len(resolved_schema.wrists)))
        + 0.10 * (face / max(1, len(resolved_schema.face)))
        + 0.05 * (legs / max(1, len(resolved_schema.legs)))
        + 0.05 * box_plausibility
    )
    normalized_support = _clamp_ratio(support, default=0.0)
    return round(_clamp_ratio(normalized_raw * normalized_support, default=0.0), 3)


def landmark_score(landmark: object) -> float:
    """Estimate one landmark confidence from visibility and presence when available."""

    if landmark is None:
        return 0.0

    coordinates = _landmark_like_coordinates(landmark)
    if coordinates is None:
        return 0.0

    visibility = _coerce_finite_float(getattr(landmark, "visibility", None))
    presence = _coerce_finite_float(getattr(landmark, "presence", None))
    score = _coerce_finite_float(getattr(landmark, "score", None))
    confidence = _coerce_finite_float(getattr(landmark, "confidence", None))

    explicit_scores = [
        _clamp_ratio(value, default=0.0)
        for value in (visibility, presence, score, confidence)
        if value is not None
    ]
    if explicit_scores:
        return round(min(explicit_scores), 3)

    return 1.0


__all__ = [
    "BLAZEPOSE33_SCHEMA",
    "COCO17_SCHEMA",
    "PoseFeatureState",
    "PoseSchema",
    "attention_score",
    "best_visible_joint",
    "hand_near_camera",
    "infer_pose_schema",
    "landmark_score",
    "parse_keypoints",
    "strong_keypoint_count",
    "support_pose_confidence",
    "visible_joint",
]