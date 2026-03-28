"""Classify coarse arm gestures from a bounded short pose sequence."""

# CHANGELOG: 2026-03-28
# BUG-1: Static gestures no longer latch from stale matches anywhere in the window; activation now requires fresh trailing support.
# BUG-2: Out-of-order frames no longer flush the full sequence state; late frames are ignored and same-timestamp frames replace in place.
# BUG-3: Gesture geometry is normalized in a body-relative coordinate system instead of fixed image-space constants, fixing distance/crop/scale drift.
# BUG-4: Wave detection is now timestamp-aware and jitter-smoothed, reducing flicker-driven false positives.
# SEC-1: Public iterable entry points are scan-bounded and memory-bounded to prevent practical CPU/RAM exhaustion on Raspberry Pi-class deployments.
# IMP-1: Added flexible landmark ingestion for COCO-17 / MoveNet, MediaPipe Pose 33, and named x/y/z(/visibility) landmarks.
# IMP-2: Added visibility-aware landmark gating and causal 1€ smoothing before temporal scoring.
# IMP-3: Streaming classification now includes short dropout hold/hysteresis to survive single-frame misses from live async pipelines.

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from heapq import heappush, heappushpop
import math
from statistics import median
from types import MappingProxyType
from typing import Final

from .config import _clamp_ratio, _coerce_positive_float
from .models import AICameraGestureEvent

Point3D = tuple[float, float, float]
LocalPoint3D = tuple[float, float, float]

_MIN_REQUIRED_FRAMES: Final[int] = 2
_ASSUMED_MAX_CAMERA_FPS: Final[float] = 60.0
_ABSOLUTE_MAX_BUFFER_FRAMES: Final[int] = 512
_ABSOLUTE_MAX_SCANNED_FRAMES: Final[int] = 4096

_COCO17_KEYPOINT_INDEXES: Final[dict[str, int]] = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
}
_MEDIAPIPE33_KEYPOINT_INDEXES: Final[dict[str, int]] = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
}
_JOINT_NAMES: Final[tuple[str, ...]] = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
)

_MIN_LANDMARK_CONFIDENCE: Final[float] = 0.35
_MIN_STATIC_FRAME_SCORE: Final[float] = 0.62
_MIN_STATIC_LATEST_GATE_SCORE: Final[float] = 0.62
_MIN_WAVE_SCORE: Final[float] = 0.60
_MIN_WAVE_FRAMES: Final[int] = 3
_MAX_SUPPORT_GAP_S: Final[float] = 0.25
_RECENT_WAVE_WINDOW_S: Final[float] = 1.25
_STREAM_RELEASE_HOLD_S: Final[float] = 0.12
_ONE_EURO_MIN_CUTOFF: Final[float] = 1.15
_ONE_EURO_BETA: Final[float] = 0.18
_ONE_EURO_D_CUTOFF: Final[float] = 1.0
_TIMESTAMP_EPSILON_S: Final[float] = 1e-6


def _coerce_finite_float(value: object) -> float | None:
    """Return one finite float or ``None`` when the input is unusable."""

    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _coerce_ratio_or_none(value: object) -> float | None:
    """Return one finite ratio in ``[0, 1]`` or ``None`` when unavailable."""

    numeric = _coerce_finite_float(value)
    if numeric is None:
        return None
    return _clamp_ratio(numeric, default=numeric)


def _coerce_min_frames(value: object, *, default: int = _MIN_REQUIRED_FRAMES) -> int:
    """Return one safe frame threshold for temporal classification."""

    numeric = _coerce_finite_float(value)
    if numeric is None:
        return max(_MIN_REQUIRED_FRAMES, int(default))
    if numeric.is_integer():
        return max(_MIN_REQUIRED_FRAMES, int(numeric))
    return max(_MIN_REQUIRED_FRAMES, int(math.ceil(numeric)))


def _mapping_get_case_insensitive(mapping: Mapping[object, object], key: str) -> object | None:
    """Return one mapping value by a case-insensitive string key when present."""

    if key in mapping:
        return mapping.get(key)
    lowered_key = key.casefold()
    for candidate_key, candidate_value in mapping.items():
        if isinstance(candidate_key, str) and candidate_key.casefold() == lowered_key:
            return candidate_value
    return None


def _extract_component(value: object, index: int, *attr_names: str) -> object | None:
    """Read one coordinate-like component from a sequence, mapping, or object."""

    if value is None:
        return None
    if isinstance(value, Mapping):
        for attr_name in attr_names:
            mapped = _mapping_get_case_insensitive(value, attr_name)
            if mapped is not None:
                return mapped
        try:
            return value.get(index)
        except Exception:
            return None
    for attr_name in attr_names:
        attribute = getattr(value, attr_name, None)
        if attribute is not None:
            return attribute
    try:
        return value[index]
    except (TypeError, IndexError, KeyError):
        return None


def _normalize_landmark(value: object) -> tuple[Point3D | None, float | None]:
    """Return one finite landmark and optional confidence from flexible inputs."""

    x = _coerce_finite_float(_extract_component(value, 0, "x"))
    y = _coerce_finite_float(_extract_component(value, 1, "y"))
    z = _coerce_finite_float(_extract_component(value, 2, "z"))
    if x is None or y is None:
        return None, None
    point: Point3D = (x, y, 0.0 if z is None else z)
    confidence = _coerce_ratio_or_none(
        _extract_component(value, 3, "visibility", "presence", "score", "confidence")
    )
    return point, confidence


def _normalize_keypoint(value: object) -> Point3D | None:
    """Return one finite pose keypoint tuple or ``None`` when the sample is malformed."""

    point, _ = _normalize_landmark(value)
    return point


def _freeze_confidence_mapping(values: dict[str, float]) -> Mapping[str, float] | None:
    """Return an immutable confidence mapping or ``None`` when empty."""

    if not values:
        return None
    return MappingProxyType(dict(values))


def _select_keypoint_schema(sparse_keypoints: Mapping[object, object]) -> dict[str, int] | None:
    """Choose the most likely index schema for a sparse keypoint mapping."""

    best_schema: dict[str, int] | None = None
    best_hits = 0
    for schema in (_MEDIAPIPE33_KEYPOINT_INDEXES, _COCO17_KEYPOINT_INDEXES):
        hits = sum(1 for index in schema.values() if index in sparse_keypoints)
        if hits > best_hits:
            best_hits = hits
            best_schema = schema
    return best_schema if best_hits else None


def _extract_pose_joint(
    sparse_keypoints: Mapping[object, object],
    joint_name: str,
    schema: Mapping[str, int] | None,
) -> tuple[Point3D | None, float | None]:
    """Read one pose joint from named or indexed sparse keypoints."""

    named_value = _mapping_get_case_insensitive(sparse_keypoints, joint_name)
    if named_value is not None:
        return _normalize_landmark(named_value)
    if schema is not None:
        indexed_value = sparse_keypoints.get(schema[joint_name])
        return _normalize_landmark(indexed_value)
    return None, None


@dataclass(frozen=True, slots=True)
class TemporalPoseFrame:
    """Store one normalized pose sample for short-window gesture classification."""

    observed_at: float
    left_shoulder: Point3D | None
    right_shoulder: Point3D | None
    left_elbow: Point3D | None
    right_elbow: Point3D | None
    left_wrist: Point3D | None
    right_wrist: Point3D | None
    left_hip: Point3D | None
    right_hip: Point3D | None
    landmark_confidences: Mapping[str, float] | None = None


def _coerce_temporal_pose_frame(frame: object) -> TemporalPoseFrame | None:
    """Normalize one frame-like object into ``TemporalPoseFrame`` when possible."""

    if isinstance(frame, TemporalPoseFrame):
        return frame if math.isfinite(frame.observed_at) else None
    observed_at = _coerce_finite_float(getattr(frame, "observed_at", None))
    if observed_at is None:
        return None
    confidences: dict[str, float] = {}
    points: dict[str, Point3D | None] = {}
    for joint_name in _JOINT_NAMES:
        point, confidence = _normalize_landmark(getattr(frame, joint_name, None))
        points[joint_name] = point
        if confidence is not None:
            confidences[joint_name] = confidence
    existing_confidences = getattr(frame, "landmark_confidences", None)
    if isinstance(existing_confidences, Mapping):
        for joint_name, confidence in existing_confidences.items():
            if not isinstance(joint_name, str):
                continue
            normalized_confidence = _coerce_ratio_or_none(confidence)
            if normalized_confidence is not None:
                confidences[joint_name] = normalized_confidence
    return TemporalPoseFrame(
        observed_at=observed_at,
        left_shoulder=points["left_shoulder"],
        right_shoulder=points["right_shoulder"],
        left_elbow=points["left_elbow"],
        right_elbow=points["right_elbow"],
        left_wrist=points["left_wrist"],
        right_wrist=points["right_wrist"],
        left_hip=points["left_hip"],
        right_hip=points["right_hip"],
        landmark_confidences=_freeze_confidence_mapping(confidences),
    )


def _sorted_valid_frames(frames: object) -> tuple[TemporalPoseFrame, ...]:
    """Return bounded, chronologically ordered frames with finite timestamps only."""

    try:
        iterator = iter(frames)
    except TypeError:
        return ()

    newest_frames_heap: list[tuple[float, int, TemporalPoseFrame]] = []
    insertion_index = 0
    scanned = 0
    for candidate in iterator:
        scanned += 1
        # BREAKING: public iterable entry points stop scanning after a bounded number of items
        # so a buggy or hostile caller cannot pin CPU/RAM indefinitely on a Raspberry Pi-class device.
        if scanned > _ABSOLUTE_MAX_SCANNED_FRAMES:
            break
        normalized_frame = _coerce_temporal_pose_frame(candidate)
        if normalized_frame is None or not math.isfinite(normalized_frame.observed_at):
            continue
        entry = (normalized_frame.observed_at, insertion_index, normalized_frame)
        insertion_index += 1
        if len(newest_frames_heap) < _ABSOLUTE_MAX_BUFFER_FRAMES:
            heappush(newest_frames_heap, entry)
            continue
        if entry[:2] > newest_frames_heap[0][:2]:
            heappushpop(newest_frames_heap, entry)

    ordered_frames = [entry[2] for entry in newest_frames_heap]
    ordered_frames.sort(key=lambda frame: frame.observed_at)

    deduplicated: list[TemporalPoseFrame] = []
    for frame in ordered_frames:
        if deduplicated and abs(frame.observed_at - deduplicated[-1].observed_at) <= _TIMESTAMP_EPSILON_S:
            deduplicated[-1] = frame
            continue
        deduplicated.append(frame)
    return tuple(deduplicated)


def _distance(a: Point3D, b: Point3D) -> float:
    """Return Euclidean distance between two 3D points."""

    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _subtract(a: Point3D, b: Point3D) -> Point3D:
    """Return vector ``a - b``."""

    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Point3D, b: Point3D) -> float:
    """Return 3D dot product."""

    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Point3D, b: Point3D) -> Point3D:
    """Return 3D cross product."""

    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(vector: Point3D) -> float:
    """Return Euclidean norm of a 3D vector."""

    return math.sqrt(_dot(vector, vector))


def _normalize_vector(vector: Point3D) -> Point3D | None:
    """Return a unit vector or ``None`` when the vector is degenerate."""

    magnitude = _norm(vector)
    if magnitude <= 1e-9 or not math.isfinite(magnitude):
        return None
    return (vector[0] / magnitude, vector[1] / magnitude, vector[2] / magnitude)


def _midpoint(a: Point3D, b: Point3D) -> Point3D:
    """Return the midpoint of two 3D points."""

    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, (a[2] + b[2]) / 2.0)


def _median_or_fallback(values: Iterable[float], *, default: float) -> float:
    """Return one finite median or a finite fallback."""

    finite_values = [value for value in values if math.isfinite(value) and value > 1e-6]
    if not finite_values:
        return default
    return float(median(finite_values))


def _landmark_confidence(frame: TemporalPoseFrame, joint_name: str) -> float | None:
    """Return optional normalized confidence for one joint."""

    if frame.landmark_confidences is None:
        return None
    confidence = frame.landmark_confidences.get(joint_name)
    return _coerce_ratio_or_none(confidence)


def _usable_point(frame: TemporalPoseFrame, joint_name: str) -> Point3D | None:
    """Return one point when present and not explicitly low-confidence."""

    point = getattr(frame, joint_name)
    if point is None:
        return None
    confidence = _landmark_confidence(frame, joint_name)
    if confidence is not None and confidence < _MIN_LANDMARK_CONFIDENCE:
        return None
    return point


def _estimate_shoulder_width(frame: TemporalPoseFrame) -> float | None:
    """Return one shoulder-width estimate for a frame."""

    left_shoulder = _usable_point(frame, "left_shoulder")
    right_shoulder = _usable_point(frame, "right_shoulder")
    if left_shoulder is None or right_shoulder is None:
        return None
    return _distance(left_shoulder, right_shoulder)


def _estimate_torso_height(frame: TemporalPoseFrame) -> float | None:
    """Return one shoulder-to-hip center distance estimate for a frame."""

    left_shoulder = _usable_point(frame, "left_shoulder")
    right_shoulder = _usable_point(frame, "right_shoulder")
    left_hip = _usable_point(frame, "left_hip")
    right_hip = _usable_point(frame, "right_hip")
    if left_shoulder is None or right_shoulder is None or left_hip is None or right_hip is None:
        return None
    shoulder_center = _midpoint(left_shoulder, right_shoulder)
    hip_center = _midpoint(left_hip, right_hip)
    return _distance(shoulder_center, hip_center)


@dataclass(frozen=True, slots=True)
class _LocalFrame:
    """Store one pose frame in body-relative coordinates."""

    observed_at: float
    left_shoulder: LocalPoint3D | None
    right_shoulder: LocalPoint3D | None
    left_elbow: LocalPoint3D | None
    right_elbow: LocalPoint3D | None
    left_wrist: LocalPoint3D | None
    right_wrist: LocalPoint3D | None
    left_hip: LocalPoint3D | None
    right_hip: LocalPoint3D | None


def _project_local(
    point: Point3D | None,
    *,
    origin: Point3D,
    lateral_axis: Point3D,
    vertical_axis: Point3D,
    forward_axis: Point3D,
    lateral_scale: float,
    vertical_scale: float,
    depth_scale: float,
) -> LocalPoint3D | None:
    """Project one world/image-space point into body-relative coordinates."""

    if point is None:
        return None
    delta = _subtract(point, origin)
    return (
        _dot(delta, lateral_axis) / lateral_scale,
        _dot(delta, vertical_axis) / vertical_scale,
        _dot(delta, forward_axis) / depth_scale,
    )


def _make_local_frames(ordered_frames: tuple[TemporalPoseFrame, ...]) -> tuple[_LocalFrame, ...]:
    """Convert frames to a body-relative coordinate system using robust sequence scales."""

    shoulder_width = _median_or_fallback(
        (_estimate_shoulder_width(frame) or float("nan") for frame in ordered_frames),
        default=1.0,
    )
    torso_height = _median_or_fallback(
        (_estimate_torso_height(frame) or float("nan") for frame in ordered_frames),
        default=shoulder_width if shoulder_width > 1e-6 else 1.0,
    )
    depth_scale = max(shoulder_width, torso_height, 1e-3)

    local_frames: list[_LocalFrame] = []
    for frame in ordered_frames:
        left_shoulder = _usable_point(frame, "left_shoulder")
        right_shoulder = _usable_point(frame, "right_shoulder")
        left_hip = _usable_point(frame, "left_hip")
        right_hip = _usable_point(frame, "right_hip")
        if left_shoulder is not None and right_shoulder is not None:
            origin = _midpoint(left_shoulder, right_shoulder)
            lateral_axis = _normalize_vector(_subtract(right_shoulder, left_shoulder))
        else:
            origin = (0.0, 0.0, 0.0)
            lateral_axis = None
        if lateral_axis is None:
            lateral_axis = (1.0, 0.0, 0.0)

        if left_hip is not None and right_hip is not None and left_shoulder is not None and right_shoulder is not None:
            shoulder_center = _midpoint(left_shoulder, right_shoulder)
            hip_center = _midpoint(left_hip, right_hip)
            torso_vector = _subtract(hip_center, shoulder_center)
            torso_vertical_component = _subtract(
                torso_vector,
                (
                    lateral_axis[0] * _dot(torso_vector, lateral_axis),
                    lateral_axis[1] * _dot(torso_vector, lateral_axis),
                    lateral_axis[2] * _dot(torso_vector, lateral_axis),
                ),
            )
            vertical_axis = _normalize_vector(torso_vertical_component)
        else:
            vertical_axis = None
        if vertical_axis is None:
            vertical_axis = (0.0, 1.0, 0.0)

        forward_axis = _normalize_vector(_cross(lateral_axis, vertical_axis))
        if forward_axis is None:
            forward_axis = (0.0, 0.0, 1.0)

        local_frames.append(
            _LocalFrame(
                observed_at=frame.observed_at,
                left_shoulder=_project_local(
                    left_shoulder,
                    origin=origin,
                    lateral_axis=lateral_axis,
                    vertical_axis=vertical_axis,
                    forward_axis=forward_axis,
                    lateral_scale=max(shoulder_width, 1e-3),
                    vertical_scale=max(torso_height, 1e-3),
                    depth_scale=depth_scale,
                ),
                right_shoulder=_project_local(
                    right_shoulder,
                    origin=origin,
                    lateral_axis=lateral_axis,
                    vertical_axis=vertical_axis,
                    forward_axis=forward_axis,
                    lateral_scale=max(shoulder_width, 1e-3),
                    vertical_scale=max(torso_height, 1e-3),
                    depth_scale=depth_scale,
                ),
                left_elbow=_project_local(
                    _usable_point(frame, "left_elbow"),
                    origin=origin,
                    lateral_axis=lateral_axis,
                    vertical_axis=vertical_axis,
                    forward_axis=forward_axis,
                    lateral_scale=max(shoulder_width, 1e-3),
                    vertical_scale=max(torso_height, 1e-3),
                    depth_scale=depth_scale,
                ),
                right_elbow=_project_local(
                    _usable_point(frame, "right_elbow"),
                    origin=origin,
                    lateral_axis=lateral_axis,
                    vertical_axis=vertical_axis,
                    forward_axis=forward_axis,
                    lateral_scale=max(shoulder_width, 1e-3),
                    vertical_scale=max(torso_height, 1e-3),
                    depth_scale=depth_scale,
                ),
                left_wrist=_project_local(
                    _usable_point(frame, "left_wrist"),
                    origin=origin,
                    lateral_axis=lateral_axis,
                    vertical_axis=vertical_axis,
                    forward_axis=forward_axis,
                    lateral_scale=max(shoulder_width, 1e-3),
                    vertical_scale=max(torso_height, 1e-3),
                    depth_scale=depth_scale,
                ),
                right_wrist=_project_local(
                    _usable_point(frame, "right_wrist"),
                    origin=origin,
                    lateral_axis=lateral_axis,
                    vertical_axis=vertical_axis,
                    forward_axis=forward_axis,
                    lateral_scale=max(shoulder_width, 1e-3),
                    vertical_scale=max(torso_height, 1e-3),
                    depth_scale=depth_scale,
                ),
                left_hip=_project_local(
                    left_hip,
                    origin=origin,
                    lateral_axis=lateral_axis,
                    vertical_axis=vertical_axis,
                    forward_axis=forward_axis,
                    lateral_scale=max(shoulder_width, 1e-3),
                    vertical_scale=max(torso_height, 1e-3),
                    depth_scale=depth_scale,
                ),
                right_hip=_project_local(
                    right_hip,
                    origin=origin,
                    lateral_axis=lateral_axis,
                    vertical_axis=vertical_axis,
                    forward_axis=forward_axis,
                    lateral_scale=max(shoulder_width, 1e-3),
                    vertical_scale=max(torso_height, 1e-3),
                    depth_scale=depth_scale,
                ),
            )
        )
    return tuple(local_frames)


@dataclass(slots=True)
class _OneEuroFilter:
    """Minimal causal 1€ filter for one scalar signal."""

    min_cutoff: float = _ONE_EURO_MIN_CUTOFF
    beta: float = _ONE_EURO_BETA
    d_cutoff: float = _ONE_EURO_D_CUTOFF
    _last_t: float | None = None
    _last_x: float | None = None
    _last_dx_hat: float = 0.0
    _last_x_hat: float | None = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        cutoff = max(1e-4, cutoff)
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def filter(self, value: float, observed_at: float) -> float:
        if self._last_t is None or self._last_x is None or self._last_x_hat is None:
            self._last_t = observed_at
            self._last_x = value
            self._last_x_hat = value
            self._last_dx_hat = 0.0
            return value

        dt = max(observed_at - self._last_t, 1e-6)
        dx = (value - self._last_x) / dt
        derivative_alpha = self._alpha(self.d_cutoff, dt)
        dx_hat = derivative_alpha * dx + (1.0 - derivative_alpha) * self._last_dx_hat
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        value_alpha = self._alpha(cutoff, dt)
        x_hat = value_alpha * value + (1.0 - value_alpha) * self._last_x_hat

        self._last_t = observed_at
        self._last_x = value
        self._last_x_hat = x_hat
        self._last_dx_hat = dx_hat
        return x_hat


def _smoothed_ordered_frames(ordered_frames: tuple[TemporalPoseFrame, ...]) -> tuple[TemporalPoseFrame, ...]:
    """Return causally smoothed frames to suppress landmark jitter."""

    filters: dict[tuple[str, int], _OneEuroFilter] = {
        (joint_name, axis): _OneEuroFilter() for joint_name in _JOINT_NAMES for axis in range(3)
    }
    smoothed_frames: list[TemporalPoseFrame] = []
    for frame in ordered_frames:
        smoothed_points: dict[str, Point3D | None] = {}
        for joint_name in _JOINT_NAMES:
            point = _usable_point(frame, joint_name)
            if point is None:
                smoothed_points[joint_name] = None
                continue
            smoothed_points[joint_name] = (
                filters[(joint_name, 0)].filter(point[0], frame.observed_at),
                filters[(joint_name, 1)].filter(point[1], frame.observed_at),
                filters[(joint_name, 2)].filter(point[2], frame.observed_at),
            )
        smoothed_frames.append(
            TemporalPoseFrame(
                observed_at=frame.observed_at,
                left_shoulder=smoothed_points["left_shoulder"],
                right_shoulder=smoothed_points["right_shoulder"],
                left_elbow=smoothed_points["left_elbow"],
                right_elbow=smoothed_points["right_elbow"],
                left_wrist=smoothed_points["left_wrist"],
                right_wrist=smoothed_points["right_wrist"],
                left_hip=smoothed_points["left_hip"],
                right_hip=smoothed_points["right_hip"],
                landmark_confidences=frame.landmark_confidences,
            )
        )
    return tuple(smoothed_frames)


def _score_at_least(value: float, threshold: float, soft_margin: float) -> float:
    """Return a soft score that reaches 1 above ``threshold``."""

    if value >= threshold:
        return 1.0
    if soft_margin <= 0.0:
        return 0.0
    return _clamp_ratio((value - (threshold - soft_margin)) / soft_margin, default=0.0)


def _score_at_most(value: float, threshold: float, soft_margin: float) -> float:
    """Return a soft score that reaches 1 below ``threshold``."""

    return _score_at_least(-value, -threshold, soft_margin)


def _score_in_range(value: float, low: float, high: float, soft_margin: float) -> float:
    """Return a soft score for values inside a bounded interval."""

    if low <= value <= high:
        return 1.0
    if value < low:
        return _score_at_least(value, low, soft_margin)
    return _score_at_most(value, high, soft_margin)


def _distance_2d(a: LocalPoint3D, b: LocalPoint3D) -> float:
    """Return 2D distance between two local points."""

    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def _frame_score_arms_crossed_local(frame: _LocalFrame) -> float:
    """Return a soft arms-crossed score for one local frame."""

    if (
        frame.left_shoulder is None
        or frame.right_shoulder is None
        or frame.left_wrist is None
        or frame.right_wrist is None
    ):
        return 0.0

    left_cross = _score_at_least(frame.left_wrist[0], 0.08, 0.22)
    right_cross = _score_at_least(-frame.right_wrist[0], 0.08, 0.22)
    chest_band = min(
        _score_in_range(frame.left_wrist[1], -0.25, 1.05, 0.25),
        _score_in_range(frame.right_wrist[1], -0.25, 1.05, 0.25),
    )
    opposite_shoulder_contact = min(
        _score_at_most(_distance_2d(frame.left_wrist, frame.right_shoulder), 1.10, 0.55),
        _score_at_most(_distance_2d(frame.right_wrist, frame.left_shoulder), 1.10, 0.55),
    )
    wrist_alignment = _score_at_most(abs(frame.left_wrist[1] - frame.right_wrist[1]), 0.55, 0.35)

    if frame.left_elbow is None or frame.right_elbow is None:
        elbow_compaction = 0.75
    else:
        elbow_compaction = min(
            _score_in_range(frame.left_elbow[0], -0.20, 0.50, 0.30),
            _score_in_range(frame.right_elbow[0], -0.50, 0.20, 0.30),
        )

    return _clamp_ratio(
        0.28 * ((left_cross + right_cross) / 2.0)
        + 0.22 * chest_band
        + 0.25 * opposite_shoulder_contact
        + 0.15 * wrist_alignment
        + 0.10 * elbow_compaction,
        default=0.0,
    )


def _frame_score_timeout_t_local(frame: _LocalFrame) -> float:
    """Return a soft timeout-T score for one local frame."""

    variants = (
        (
            frame.left_shoulder,
            frame.left_elbow,
            frame.left_wrist,
            frame.right_shoulder,
            frame.right_elbow,
            frame.right_wrist,
        ),
        (
            frame.right_shoulder,
            frame.right_elbow,
            frame.right_wrist,
            frame.left_shoulder,
            frame.left_elbow,
            frame.left_wrist,
        ),
    )
    best_score = 0.0
    for (
        horizontal_shoulder,
        horizontal_elbow,
        horizontal_wrist,
        vertical_shoulder,
        vertical_elbow,
        vertical_wrist,
    ) in variants:
        if (
            horizontal_shoulder is None
            or horizontal_elbow is None
            or horizontal_wrist is None
            or vertical_shoulder is None
            or vertical_elbow is None
            or vertical_wrist is None
        ):
            continue

        horizontal_flatness = min(
            _score_at_most(abs(horizontal_wrist[1] - horizontal_elbow[1]), 0.14, 0.16),
            _score_at_most(abs(horizontal_elbow[1] - horizontal_shoulder[1]), 0.22, 0.18),
        )
        horizontal_extension = _score_at_least(
            abs(horizontal_wrist[0] - horizontal_shoulder[0]), 0.70, 0.35
        )
        vertical_straightness = min(
            _score_at_most(abs(vertical_wrist[0] - vertical_elbow[0]), 0.12, 0.12),
            _score_at_most(abs(vertical_elbow[0] - vertical_shoulder[0]), 0.20, 0.18),
        )
        vertical_stack = min(
            _score_at_least(vertical_elbow[1] - vertical_wrist[1], 0.10, 0.18),
            _score_in_range(vertical_elbow[1], -0.05, 0.45, 0.22),
            _score_at_most(vertical_wrist[1], 0.05, 0.30),
        )
        intersection = min(
            _score_at_most(abs(horizontal_wrist[0] - vertical_wrist[0]), 0.22, 0.18),
            _score_at_most(abs(horizontal_wrist[1] - vertical_elbow[1]), 0.24, 0.18),
        )

        score = _clamp_ratio(
            0.24 * horizontal_flatness
            + 0.18 * horizontal_extension
            + 0.22 * vertical_straightness
            + 0.18 * vertical_stack
            + 0.18 * intersection,
            default=0.0,
        )
        if score > best_score:
            best_score = score
    return best_score


def _frame_score_two_hand_dismiss_local(frame: _LocalFrame) -> float:
    """Return a soft two-hand-dismiss score for one local frame."""

    if (
        frame.left_shoulder is None
        or frame.right_shoulder is None
        or frame.left_wrist is None
        or frame.right_wrist is None
    ):
        return 0.0

    left_out = _score_at_least(frame.left_shoulder[0] - frame.left_wrist[0], 0.18, 0.25)
    right_out = _score_at_least(frame.right_wrist[0] - frame.right_shoulder[0], 0.18, 0.25)
    shoulder_height = min(
        _score_at_most(abs(frame.left_wrist[1] - frame.left_shoulder[1]), 0.30, 0.22),
        _score_at_most(abs(frame.right_wrist[1] - frame.right_shoulder[1]), 0.30, 0.22),
    )
    symmetry = _score_at_most(abs(frame.left_wrist[1] - frame.right_wrist[1]), 0.28, 0.20)

    if frame.left_elbow is None or frame.right_elbow is None:
        elbow_extension = 0.70
    else:
        elbow_extension = min(
            _score_in_range(frame.left_elbow[0], frame.left_wrist[0], frame.left_shoulder[0], 0.30),
            _score_in_range(frame.right_elbow[0], frame.right_shoulder[0], frame.right_wrist[0], 0.30),
        )

    return _clamp_ratio(
        0.28 * ((left_out + right_out) / 2.0)
        + 0.26 * shoulder_height
        + 0.18 * symmetry
        + 0.28 * elbow_extension,
        default=0.0,
    )


def _trailing_static_candidate(
    local_frames: tuple[_LocalFrame, ...],
    raw_local_frames: tuple[_LocalFrame, ...],
    *,
    min_frames: int,
    event: AICameraGestureEvent,
    score_fn,
) -> tuple[AICameraGestureEvent, float] | None:
    """Return one active static-gesture candidate from fresh trailing support only."""

    if len(local_frames) < min_frames or len(raw_local_frames) < min_frames:
        return None
    per_frame_scores = [(frame.observed_at, float(score_fn(frame))) for frame in local_frames]
    latest_time, latest_score = per_frame_scores[-1]
    raw_latest_score = float(score_fn(raw_local_frames[-1]))
    if latest_score < _MIN_STATIC_FRAME_SCORE or raw_latest_score < _MIN_STATIC_LATEST_GATE_SCORE:
        return None

    support_scores: list[float] = []
    support_start = latest_time
    previous_time = latest_time
    for observed_at, score in reversed(per_frame_scores):
        if previous_time - observed_at > _MAX_SUPPORT_GAP_S and support_scores:
            break
        if score < _MIN_STATIC_FRAME_SCORE:
            if support_scores:
                break
            return None
        support_scores.append(score)
        support_start = observed_at
        previous_time = observed_at

    support_frames = len(support_scores)
    if support_frames < min_frames:
        return None

    support_duration_s = max(0.0, latest_time - support_start)
    mean_support_score = sum(support_scores) / support_frames
    confidence = round(
        _clamp_ratio(
            0.40
            + 0.28 * mean_support_score
            + 0.16 * min(1.0, support_frames / max(float(min_frames), 1.0))
            + 0.16 * min(1.0, support_duration_s / 0.35),
            default=0.40,
        ),
        3,
    )
    return event, confidence


def _wave_candidate(
    local_frames: tuple[_LocalFrame, ...],
    *,
    min_frames: int,
) -> tuple[AICameraGestureEvent, float] | None:
    """Return one wave candidate based on recent raised-hand oscillation."""

    if len(local_frames) < max(min_frames, _MIN_WAVE_FRAMES):
        return None
    newest_time = local_frames[-1].observed_at
    recent_frames = tuple(
        frame for frame in local_frames if newest_time - frame.observed_at <= _RECENT_WAVE_WINDOW_S
    )
    if len(recent_frames) < max(min_frames, _MIN_WAVE_FRAMES):
        return None

    best_score = 0.0
    for wrist_name, shoulder_name in (("left_wrist", "left_shoulder"), ("right_wrist", "right_shoulder")):
        series: list[tuple[float, float, float]] = []
        for frame in recent_frames:
            wrist = getattr(frame, wrist_name)
            shoulder = getattr(frame, shoulder_name)
            if wrist is None or shoulder is None:
                continue
            if wrist[1] >= shoulder[1] - 0.10:
                continue
            series.append((frame.observed_at, wrist[0] - shoulder[0], wrist[1] - shoulder[1]))

        if len(series) < max(min_frames, _MIN_WAVE_FRAMES):
            continue
        if abs(series[-1][0] - newest_time) > _MAX_SUPPORT_GAP_S:
            continue

        xs = [value[1] for value in series]
        amplitude = max(xs) - min(xs)
        if amplitude <= 1e-6:
            continue

        direction_changes = 0
        last_sign = 0
        high_velocity_samples = 0
        for index in range(1, len(series)):
            dt = series[index][0] - series[index - 1][0]
            if dt <= 1e-6:
                continue
            velocity = (series[index][1] - series[index - 1][1]) / dt
            sign = 1 if velocity > 0.35 else -1 if velocity < -0.35 else 0
            if sign:
                high_velocity_samples += 1
            if sign and last_sign and sign != last_sign:
                direction_changes += 1
            if sign:
                last_sign = sign

        raised_depth = min(-value[2] for value in series)
        support_duration_s = max(0.0, series[-1][0] - series[0][0])
        amplitude_score = _score_at_least(amplitude, 0.22, 0.20)
        changes_score = _score_at_least(float(direction_changes), 1.0, 1.0)
        velocity_score = _score_at_least(float(high_velocity_samples), 2.0, 2.0)
        height_score = _score_at_least(-raised_depth, 0.12, 0.18)
        duration_score = _score_at_least(support_duration_s, 0.18, 0.22)

        score = _clamp_ratio(
            0.28 * amplitude_score
            + 0.24 * changes_score
            + 0.18 * velocity_score
            + 0.14 * height_score
            + 0.16 * duration_score,
            default=0.0,
        )
        if score > best_score:
            best_score = score

    if best_score < _MIN_WAVE_SCORE:
        return None
    confidence = round(_clamp_ratio(0.44 + 0.40 * best_score, default=0.44), 3)
    return AICameraGestureEvent.WAVE, confidence


class TemporalPoseGestureClassifier:
    """Classify coarse arm gestures from a short landmark sequence window."""

    def __init__(self, *, window_s: float, min_frames: int) -> None:
        """Initialize one bounded sequence classifier."""

        self.window_s = _coerce_positive_float(window_s, default=1.6)
        if not math.isfinite(self.window_s) or self.window_s <= 0.0:
            self.window_s = 1.6
        self.min_frames = _coerce_min_frames(min_frames, default=_MIN_REQUIRED_FRAMES)
        estimated_window_frames = int(math.ceil(self.window_s * _ASSUMED_MAX_CAMERA_FPS))
        self._max_buffer_frames = min(
            _ABSOLUTE_MAX_BUFFER_FRAMES,
            max(self.min_frames * 3, estimated_window_frames + self.min_frames + 8),
        )
        self._frames: deque[TemporalPoseFrame] = deque()
        self._held_event: AICameraGestureEvent = AICameraGestureEvent.NONE
        self._held_confidence: float | None = None
        self._held_until: float | None = None

    def clear(self) -> None:
        """Discard any buffered sequence state."""

        self._frames.clear()
        self._held_event = AICameraGestureEvent.NONE
        self._held_confidence = None
        self._held_until = None

    def _current_classification(self, *, now: float | None = None) -> tuple[AICameraGestureEvent, float | None]:
        """Classify the currently buffered frames and apply a short dropout hold."""

        event, confidence = classify_temporal_gesture(tuple(self._frames), min_frames=self.min_frames)
        now_value = self._frames[-1].observed_at if self._frames else _coerce_finite_float(now)
        if event != AICameraGestureEvent.NONE:
            self._held_event = event
            self._held_confidence = confidence
            self._held_until = None if now_value is None else now_value + _STREAM_RELEASE_HOLD_S
            return event, confidence
        if (
            self._held_event != AICameraGestureEvent.NONE
            and now_value is not None
            and self._held_until is not None
            and now_value <= self._held_until
        ):
            return self._held_event, self._held_confidence
        self._held_event = AICameraGestureEvent.NONE
        self._held_confidence = None
        self._held_until = None
        return AICameraGestureEvent.NONE, None

    def observe(
        self,
        *,
        observed_at: float,
        sparse_keypoints: Mapping[object, object],
    ) -> tuple[AICameraGestureEvent, float | None]:
        """Append one pose sample and classify the current coarse gesture."""

        observed_at_value = _coerce_finite_float(observed_at)
        if observed_at_value is None:
            return self._current_classification()

        safe_keypoints = sparse_keypoints if isinstance(sparse_keypoints, Mapping) else {}
        schema = _select_keypoint_schema(safe_keypoints)
        points: dict[str, Point3D | None] = {}
        confidences: dict[str, float] = {}
        for joint_name in _JOINT_NAMES:
            point, confidence = _extract_pose_joint(safe_keypoints, joint_name, schema)
            points[joint_name] = point
            if confidence is not None:
                confidences[joint_name] = confidence

        new_frame = TemporalPoseFrame(
            observed_at=observed_at_value,
            left_shoulder=points["left_shoulder"],
            right_shoulder=points["right_shoulder"],
            left_elbow=points["left_elbow"],
            right_elbow=points["right_elbow"],
            left_wrist=points["left_wrist"],
            right_wrist=points["right_wrist"],
            left_hip=points["left_hip"],
            right_hip=points["right_hip"],
            landmark_confidences=_freeze_confidence_mapping(confidences),
        )

        if self._frames:
            last_observed_at = self._frames[-1].observed_at
            if observed_at_value < last_observed_at - _TIMESTAMP_EPSILON_S:
                # BREAKING: a late frame no longer flushes the whole temporal state.
                # On real async camera pipelines that created blind spots after a single delayed callback.
                return self._current_classification(now=last_observed_at)
            if abs(observed_at_value - last_observed_at) <= _TIMESTAMP_EPSILON_S:
                self._frames[-1] = new_frame
                self._prune(now=observed_at_value)
                return self._current_classification(now=observed_at_value)

        self._frames.append(new_frame)
        self._prune(now=observed_at_value)
        while len(self._frames) > self._max_buffer_frames:
            self._frames.popleft()
        return self._current_classification(now=observed_at_value)

    def _prune(self, *, now: float) -> None:
        """Drop samples that are older than the configured temporal window."""

        now_value = _coerce_finite_float(now)
        if now_value is None:
            self.clear()
            return
        cutoff = now_value - self.window_s
        while self._frames and (
            not math.isfinite(self._frames[0].observed_at) or self._frames[0].observed_at < cutoff
        ):
            self._frames.popleft()


def classify_temporal_gesture(
    frames: tuple[TemporalPoseFrame, ...],
    *,
    min_frames: int,
) -> tuple[AICameraGestureEvent, float | None]:
    """Classify one coarse arm gesture from a short sequence of pose frames."""

    effective_min_frames = _coerce_min_frames(min_frames, default=_MIN_REQUIRED_FRAMES)
    ordered_frames = _sorted_valid_frames(frames)
    if len(ordered_frames) < effective_min_frames:
        return AICameraGestureEvent.NONE, None

    smoothed_frames = _smoothed_ordered_frames(ordered_frames)
    local_frames = _make_local_frames(smoothed_frames)
    raw_local_frames = _make_local_frames(ordered_frames)
    if len(local_frames) < effective_min_frames or len(raw_local_frames) < effective_min_frames:
        return AICameraGestureEvent.NONE, None

    candidates: list[tuple[AICameraGestureEvent, float]] = []
    for event, score_fn in (
        (AICameraGestureEvent.ARMS_CROSSED, _frame_score_arms_crossed_local),
        (AICameraGestureEvent.TIMEOUT_T, _frame_score_timeout_t_local),
        (AICameraGestureEvent.TWO_HAND_DISMISS, _frame_score_two_hand_dismiss_local),
    ):
        candidate = _trailing_static_candidate(
            local_frames,
            raw_local_frames,
            min_frames=effective_min_frames,
            event=event,
            score_fn=score_fn,
        )
        if candidate is not None:
            candidates.append(candidate)

    wave_candidate = _wave_candidate(local_frames, min_frames=effective_min_frames)
    if wave_candidate is not None:
        candidates.append(wave_candidate)

    if not candidates:
        return AICameraGestureEvent.NONE, None

    best_event, best_confidence = max(candidates, key=lambda item: item[1])
    return best_event, best_confidence


def frame_is_arms_crossed(frame: TemporalPoseFrame) -> bool:
    """Return whether one pose frame looks like crossed arms at chest height."""

    normalized_frame = _coerce_temporal_pose_frame(frame)
    if normalized_frame is None:
        return False
    local_frames = _make_local_frames((normalized_frame,))
    return bool(local_frames) and _frame_score_arms_crossed_local(local_frames[0]) >= _MIN_STATIC_FRAME_SCORE


def frame_is_timeout_t(frame: TemporalPoseFrame) -> bool:
    """Return whether one pose frame looks like a timeout-T arm configuration."""

    normalized_frame = _coerce_temporal_pose_frame(frame)
    if normalized_frame is None:
        return False
    local_frames = _make_local_frames((normalized_frame,))
    return bool(local_frames) and _frame_score_timeout_t_local(local_frames[0]) >= _MIN_STATIC_FRAME_SCORE


def frame_is_two_hand_dismiss(frame: TemporalPoseFrame) -> bool:
    """Return whether one frame looks like a symmetric two-hand dismiss pose."""

    normalized_frame = _coerce_temporal_pose_frame(frame)
    if normalized_frame is None:
        return False
    local_frames = _make_local_frames((normalized_frame,))
    return (
        bool(local_frames)
        and _frame_score_two_hand_dismiss_local(local_frames[0]) >= _MIN_STATIC_FRAME_SCORE
    )


def frames_form_wave(frames: tuple[TemporalPoseFrame, ...], *, min_frames: int) -> bool:
    """Return whether one sequence contains a raised-hand lateral wave pattern."""

    effective_min_frames = _coerce_min_frames(min_frames, default=_MIN_REQUIRED_FRAMES)
    ordered_frames = _sorted_valid_frames(frames)
    if len(ordered_frames) < max(effective_min_frames, _MIN_WAVE_FRAMES):
        return False
    smoothed_frames = _smoothed_ordered_frames(ordered_frames)
    local_frames = _make_local_frames(smoothed_frames)
    return _wave_candidate(local_frames, min_frames=effective_min_frames) is not None


__all__ = [
    "TemporalPoseFrame",
    "TemporalPoseGestureClassifier",
    "classify_temporal_gesture",
    "frame_is_arms_crossed",
    "frame_is_timeout_t",
    "frame_is_two_hand_dismiss",
    "frames_form_wave",
]