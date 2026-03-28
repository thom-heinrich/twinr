"""Rank decoded pose candidates against the authoritative person detection box."""

from __future__ import annotations

# CHANGELOG: 2026-03-28
# BUG-1: Reject malformed or degenerate keypoint payloads instead of allowing box-only ranking
#        to surface unusable poses that later break downstream consumers.
# BUG-2: Remove ranking-time score quantization; keep full-precision metrics so near-tie crowded
#        frames do not flip identities due to premature rounding.
# SEC-1: Bound candidate and keypoint cardinality to prevent CPU/RAM denial on Raspberry Pi 4
#        from corrupted or hostile upstream decoder output.
# IMP-1: Upgrade ranking from raw-box IoU heuristics to a box-conditioned, keypoint-aware fusion
#        using keypoint-derived boxes, coverage, aspect similarity, and weighted keypoint-in-box evidence.
# IMP-2: Add optional temporal priors (`previous_selected_box`, `previous_selected_keypoints`) for
#        video-stable matching without breaking existing callers.

import itertools
import logging
import math
from dataclasses import dataclass
from typing import TypeAlias

from .config import _clamp_ratio, _coerce_float
from .geometry import box_center_similarity, box_from_pixel_bbox, box_size_similarity, iou
from .models import AICameraBox

LOGGER = logging.getLogger(__name__)

Bounds: TypeAlias = tuple[float, float, float, float]
Point3: TypeAlias = tuple[float, float, float]

_MAX_POSE_CANDIDATES = 128
_MAX_KEYPOINT_VALUES = 1024
_MIN_KEYPOINTS_FOR_VALID_POSE = 2
_MIN_CONFIDENT_KEYPOINT_SCORE = 0.15
_PRIMARY_BOX_EXPANSION = 0.08
_PRIMARY_BOX_EDGE_BONUS = 0.04
_KEYPOINT_BOX_EXPANSION = 0.10


@dataclass(frozen=True, slots=True)
class PoseCandidateMatch:
    """Describe how one HigherHRNet candidate aligns with the primary person."""

    candidate_index: int
    raw_keypoints: list[float]
    raw_score: float
    normalized_score: float
    box: AICameraBox
    overlap: float
    center_similarity: float
    size_similarity: float
    selection_score: float


@dataclass(frozen=True, slots=True)
class _ParsedKeypoints:
    """Internal normalized pose evidence used for ranking."""

    point_count: int
    confident_point_count: int
    confidence_mean: float
    count_score: float
    normalized_points: tuple[Point3, ...]
    normalized_box: Bounds | None
    truncated: bool


def _finite_float(value: object, *, default: float = 0.0) -> float:
    """Coerce one numeric signal to a finite float."""
    coerced = _coerce_float(value, default=default)
    if not math.isfinite(coerced):
        return default
    return coerced


def _finite_ratio(value: object, *, default: float = 0.0) -> float:
    """Coerce one numeric signal to a finite 0..1 ratio."""
    return _clamp_ratio(_finite_float(value, default=default), default=default)


def _strict_finite_float(value: object) -> float | None:
    """Return a finite float or ``None`` if conversion fails."""
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(coerced):
        return None
    return coerced


def _looks_numeric(value: object) -> bool:
    """Return whether one object behaves like a scalar number."""
    return _strict_finite_float(value) is not None


def _bounded_snapshot(values: object, *, limit: int) -> tuple[list[object], bool]:
    """Copy at most ``limit`` items from an arbitrary iterable."""
    iterator = iter(values)
    snapshot = list(itertools.islice(iterator, limit + 1))
    overflow = len(snapshot) > limit
    if overflow:
        snapshot = snapshot[:limit]
    return snapshot, overflow


def _copy_keypoints(raw_keypoints: list[float]) -> list[float]:
    """Return a detached, bounded copy so upstream mutations do not rewrite results."""
    try:
        copied, truncated = _bounded_snapshot(raw_keypoints, limit=_MAX_KEYPOINT_VALUES)
        if truncated:
            LOGGER.warning(
                "Pose candidate keypoints exceeded %d values; truncating stored payload.",
                _MAX_KEYPOINT_VALUES,
            )
        return list(copied)
    except TypeError:
        LOGGER.warning(
            "Invalid raw_keypoints payload of type %s; storing an empty keypoint list.",
            type(raw_keypoints).__name__,
        )
        return []


def _parse_flat_keypoints(values: list[object]) -> list[Point3]:
    """Parse a flat ``[x, y, score]`` or ``[x, y]`` keypoint payload."""
    if len(values) < 2:
        return []

    if len(values) >= 3 and len(values) % 3 == 0:
        step = 3
    elif len(values) % 2 == 0:
        step = 2
    elif len(values) >= 6:
        step = 3
    else:
        return []

    points: list[Point3] = []
    usable_length = len(values) - (len(values) % step)
    for index in range(0, usable_length, step):
        x = _strict_finite_float(values[index])
        y = _strict_finite_float(values[index + 1])
        if x is None or y is None:
            continue
        score = _finite_ratio(values[index + 2], default=1.0) if step == 3 else 1.0
        points.append((x, y, score))
    return points


def _parse_nested_keypoints(values: list[object]) -> list[Point3]:
    """Parse a nested ``[[x, y, score], ...]``-style payload."""
    points: list[Point3] = []
    for item in values:
        try:
            row = list(itertools.islice(iter(item), 4))
        except TypeError:
            continue
        if len(row) < 2:
            continue
        x = _strict_finite_float(row[0])
        y = _strict_finite_float(row[1])
        if x is None or y is None:
            continue
        score = _finite_ratio(row[2], default=1.0) if len(row) >= 3 else 1.0
        points.append((x, y, score))
    return points


def _normalize_point_coords(
    points: list[Point3],
    *,
    frame_width: int | None,
    frame_height: int | None,
) -> tuple[Point3, ...]:
    """Normalize pixel-space keypoints into 0..1-ish coordinates when possible."""
    if not points:
        return ()

    max_abs_x = max(abs(point[0]) for point in points)
    max_abs_y = max(abs(point[1]) for point in points)
    already_normalized = max_abs_x <= 1.5 and max_abs_y <= 1.5
    if already_normalized:
        return tuple(points)

    if not frame_width or not frame_height or frame_width <= 0 or frame_height <= 0:
        return ()

    inv_width = 1.0 / frame_width
    inv_height = 1.0 / frame_height
    return tuple((x * inv_width, y * inv_height, score) for x, y, score in points)


def _expand_bounds(bounds: Bounds, *, fraction: float) -> Bounds:
    """Expand one box by a per-axis ratio."""
    left, top, right, bottom = bounds
    width = right - left
    height = bottom - top
    expand_x = width * fraction
    expand_y = height * fraction
    return (left - expand_x, top - expand_y, right + expand_x, bottom + expand_y)


def _bounds_from_points(points: tuple[Point3, ...], *, expansion: float = 0.0) -> Bounds | None:
    """Return the tight bounds of a normalized keypoint set."""
    if len(points) < _MIN_KEYPOINTS_FOR_VALID_POSE:
        return None

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    left = min(xs)
    right = max(xs)
    top = min(ys)
    bottom = max(ys)

    if right <= left or bottom <= top:
        return None

    return _expand_bounds((left, top, right, bottom), fraction=expansion)


def _parse_keypoints(
    raw_keypoints: object,
    *,
    frame_width: int | None,
    frame_height: int | None,
) -> _ParsedKeypoints:
    """Parse one arbitrary keypoint payload into normalized evidence."""
    try:
        snapshot, truncated = _bounded_snapshot(raw_keypoints, limit=_MAX_KEYPOINT_VALUES)
    except TypeError:
        return _ParsedKeypoints(
            point_count=0,
            confident_point_count=0,
            confidence_mean=0.0,
            count_score=0.0,
            normalized_points=(),
            normalized_box=None,
            truncated=False,
        )

    if not snapshot:
        return _ParsedKeypoints(
            point_count=0,
            confident_point_count=0,
            confidence_mean=0.0,
            count_score=0.0,
            normalized_points=(),
            normalized_box=None,
            truncated=truncated,
        )

    points = _parse_flat_keypoints(snapshot) if _looks_numeric(snapshot[0]) else _parse_nested_keypoints(snapshot)
    if not points:
        return _ParsedKeypoints(
            point_count=0,
            confident_point_count=0,
            confidence_mean=0.0,
            count_score=0.0,
            normalized_points=(),
            normalized_box=None,
            truncated=truncated,
        )

    normalized_points = _normalize_point_coords(
        points,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    confident_points = tuple(
        point for point in normalized_points if point[2] >= _MIN_CONFIDENT_KEYPOINT_SCORE
    ) or normalized_points
    confidence_mean = sum(point[2] for point in points) / len(points)
    count_score = _finite_ratio(len(confident_points) / 6.0, default=0.0)
    normalized_box = _bounds_from_points(confident_points, expansion=_KEYPOINT_BOX_EXPANSION)

    return _ParsedKeypoints(
        point_count=len(points),
        confident_point_count=len(confident_points),
        confidence_mean=_finite_ratio(confidence_mean, default=0.0),
        count_score=count_score,
        normalized_points=normalized_points,
        normalized_box=normalized_box,
        truncated=truncated,
    )


def _normalize_box_bounds(
    left: float,
    top: float,
    right: float,
    bottom: float,
    *,
    frame_width: int | None,
    frame_height: int | None,
) -> Bounds | None:
    """Normalize one box representation into ``(left, top, right, bottom)``."""
    if not all(math.isfinite(value) for value in (left, top, right, bottom)):
        return None

    if right < left:
        left, right = right, left
    if bottom < top:
        top, bottom = bottom, top

    if frame_width and frame_width > 0 and frame_height and frame_height > 0:
        if max(abs(left), abs(right)) > 1.5 or max(abs(top), abs(bottom)) > 1.5:
            left /= frame_width
            right /= frame_width
            top /= frame_height
            bottom /= frame_height

    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def _bounds_from_box(
    box: object,
    *,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> Bounds | None:
    """Best-effort extractor for common bounding-box object layouts."""
    if box is None:
        return None

    attr_groups = (
        ("left", "top", "right", "bottom", "ltrb"),
        ("xmin", "ymin", "xmax", "ymax", "ltrb"),
        ("x_min", "y_min", "x_max", "y_max", "ltrb"),
        ("x1", "y1", "x2", "y2", "ltrb"),
        ("x0", "y0", "x1", "y1", "ltrb"),
        ("left", "top", "width", "height", "xywh"),
        ("xmin", "ymin", "width", "height", "xywh"),
        ("x", "y", "width", "height", "xywh"),
        ("x", "y", "w", "h", "xywh"),
        ("center_x", "center_y", "width", "height", "cxcywh"),
        ("cx", "cy", "w", "h", "cxcywh"),
    )
    for a, b, c, d, mode in attr_groups:
        values = (
            _strict_finite_float(getattr(box, a, None)),
            _strict_finite_float(getattr(box, b, None)),
            _strict_finite_float(getattr(box, c, None)),
            _strict_finite_float(getattr(box, d, None)),
        )
        if any(value is None for value in values):
            continue
        first, second, third, fourth = values  # type: ignore[misc]
        if mode == "ltrb":
            return _normalize_box_bounds(
                first,
                second,
                third,
                fourth,
                frame_width=frame_width,
                frame_height=frame_height,
            )
        if mode == "xywh":
            return _normalize_box_bounds(
                first,
                second,
                first + third,
                second + fourth,
                frame_width=frame_width,
                frame_height=frame_height,
            )
        return _normalize_box_bounds(
            first - (third / 2.0),
            second - (fourth / 2.0),
            first + (third / 2.0),
            second + (fourth / 2.0),
            frame_width=frame_width,
            frame_height=frame_height,
        )

    try:
        values = list(itertools.islice(iter(box), 4))
    except TypeError:
        values = []

    if len(values) == 4 and all(_strict_finite_float(value) is not None for value in values):
        left, top, third, fourth = (_strict_finite_float(value) for value in values)
        assert left is not None and top is not None and third is not None and fourth is not None
        return _normalize_box_bounds(
            left,
            top,
            third,
            fourth,
            frame_width=frame_width,
            frame_height=frame_height,
        )
    return None


def _bounds_area(bounds: Bounds | None) -> float:
    """Return one box area."""
    if bounds is None:
        return 0.0
    left, top, right, bottom = bounds
    return max(0.0, right - left) * max(0.0, bottom - top)


def _intersection_area(a: Bounds | None, b: Bounds | None) -> float:
    """Return the intersection area between two boxes."""
    if a is None or b is None:
        return 0.0
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    return max(0.0, right - left) * max(0.0, bottom - top)


def _bounds_iou(a: Bounds | None, b: Bounds | None) -> float:
    """Return IoU between two generic bounds."""
    if a is None or b is None:
        return 0.0
    inter = _intersection_area(a, b)
    if inter <= 0.0:
        return 0.0
    area_a = _bounds_area(a)
    area_b = _bounds_area(b)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return _finite_ratio(inter / union, default=0.0)


def _bounds_center_similarity(a: Bounds | None, b: Bounds | None) -> float:
    """Return a scale-aware center similarity for two generic bounds."""
    if a is None or b is None:
        return 0.0
    ax = (a[0] + a[2]) * 0.5
    ay = (a[1] + a[3]) * 0.5
    bx = (b[0] + b[2]) * 0.5
    by = (b[1] + b[3]) * 0.5
    dx = ax - bx
    dy = ay - by
    diag = math.hypot(max(a[2] - a[0], b[2] - b[0]), max(a[3] - a[1], b[3] - b[1]))
    if diag <= 0.0:
        return 0.0
    return _finite_ratio(1.0 - (math.hypot(dx, dy) / diag), default=0.0)


def _bounds_size_similarity(a: Bounds | None, b: Bounds | None) -> float:
    """Return a size similarity ratio between two generic bounds."""
    if a is None or b is None:
        return 0.0
    aw = max(0.0, a[2] - a[0])
    ah = max(0.0, a[3] - a[1])
    bw = max(0.0, b[2] - b[0])
    bh = max(0.0, b[3] - b[1])
    if aw <= 0.0 or ah <= 0.0 or bw <= 0.0 or bh <= 0.0:
        return 0.0
    width_similarity = min(aw, bw) / max(aw, bw)
    height_similarity = min(ah, bh) / max(ah, bh)
    return _finite_ratio((width_similarity + height_similarity) * 0.5, default=0.0)


def _bounds_aspect_similarity(a: Bounds | None, b: Bounds | None) -> float:
    """Return one aspect-ratio similarity score."""
    if a is None or b is None:
        return 0.0
    aw = max(0.0, a[2] - a[0])
    ah = max(0.0, a[3] - a[1])
    bw = max(0.0, b[2] - b[0])
    bh = max(0.0, b[3] - b[1])
    if aw <= 0.0 or ah <= 0.0 or bw <= 0.0 or bh <= 0.0:
        return 0.0
    aspect_a = aw / ah
    aspect_b = bw / bh
    return _finite_ratio(min(aspect_a, aspect_b) / max(aspect_a, aspect_b), default=0.0)


def _primary_coverage(candidate_bounds: Bounds | None, primary_bounds: Bounds | None) -> float:
    """Return how much of the primary box is covered by the candidate."""
    primary_area = _bounds_area(primary_bounds)
    if primary_area <= 0.0:
        return 0.0
    return _finite_ratio(_intersection_area(candidate_bounds, primary_bounds) / primary_area, default=0.0)


def _primary_box_expansion(bounds: Bounds) -> float:
    """Return an edge-aware expansion for primary-box membership tests."""
    left, top, right, bottom = bounds
    expansion = _PRIMARY_BOX_EXPANSION
    if left <= 0.02 or top <= 0.02 or right >= 0.98 or bottom >= 0.98:
        expansion += _PRIMARY_BOX_EDGE_BONUS
    return expansion


def _weighted_inside_ratio(points: tuple[Point3, ...], primary_bounds: Bounds | None) -> float:
    """Return the confidence-weighted fraction of keypoints inside the expanded primary box."""
    if not points or primary_bounds is None:
        return 0.0

    expanded = _expand_bounds(primary_bounds, fraction=_primary_box_expansion(primary_bounds))
    total_weight = 0.0
    inside_weight = 0.0
    for x, y, score in points:
        weight = max(score, 0.05)
        total_weight += weight
        if expanded[0] <= x <= expanded[2] and expanded[1] <= y <= expanded[3]:
            inside_weight += weight
    if total_weight <= 0.0:
        return 0.0
    return _finite_ratio(inside_weight / total_weight, default=0.0)


def _safe_geometry_metrics(
    box: AICameraBox,
    primary_person_box: AICameraBox | None,
    *,
    frame_width: int | None,
    frame_height: int | None,
) -> tuple[float, float, float, Bounds | None, Bounds | None]:
    """Compute public box metrics and a generic-bounds fallback representation."""
    candidate_bounds = _bounds_from_box(box, frame_width=frame_width, frame_height=frame_height)
    primary_bounds = _bounds_from_box(primary_person_box, frame_width=frame_width, frame_height=frame_height)

    overlap = 0.0
    center_similarity = 0.0
    size_similarity = 0.0
    if primary_person_box is None:
        return overlap, center_similarity, size_similarity, candidate_bounds, primary_bounds

    try:
        overlap = _finite_ratio(iou(box, primary_person_box), default=0.0)
        center_similarity = _finite_ratio(box_center_similarity(box, primary_person_box), default=0.0)
        size_similarity = _finite_ratio(box_size_similarity(box, primary_person_box), default=0.0)
    except Exception:
        LOGGER.exception(
            "Failed to compare one pose candidate against the primary person box with "
            "geometry helpers; falling back to generic bounds math."
        )
        overlap = _bounds_iou(candidate_bounds, primary_bounds)
        center_similarity = _bounds_center_similarity(candidate_bounds, primary_bounds)
        size_similarity = _bounds_size_similarity(candidate_bounds, primary_bounds)

    return overlap, center_similarity, size_similarity, candidate_bounds, primary_bounds


def _temporal_score(
    *,
    candidate_bounds: Bounds | None,
    candidate_keypoint_box: Bounds | None,
    previous_selected_box: AICameraBox | None,
    previous_selected_keypoints: object | None,
    previous_parsed_keypoints: _ParsedKeypoints | None,
    frame_width: int | None,
    frame_height: int | None,
) -> float:
    """Return an optional temporal prior for video-stable matching."""
    temporal_signals: list[float] = []

    previous_bounds = _bounds_from_box(
        previous_selected_box,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    if candidate_bounds is not None and previous_bounds is not None:
        temporal_signals.append(
            _finite_ratio(
                0.70 * _bounds_iou(candidate_bounds, previous_bounds)
                + 0.30 * _bounds_center_similarity(candidate_bounds, previous_bounds),
                default=0.0,
            )
        )

    if candidate_keypoint_box is not None:
        previous_keypoint_box = previous_parsed_keypoints.normalized_box if previous_parsed_keypoints is not None else None
        if previous_keypoint_box is None and previous_selected_keypoints is not None:
            previous_keypoint_box = _parse_keypoints(
                previous_selected_keypoints,
                frame_width=frame_width,
                frame_height=frame_height,
            ).normalized_box
        if previous_keypoint_box is not None:
            temporal_signals.append(
                _finite_ratio(
                    0.70 * _bounds_iou(candidate_keypoint_box, previous_keypoint_box)
                    + 0.30 * _bounds_center_similarity(candidate_keypoint_box, previous_keypoint_box),
                    default=0.0,
                )
            )

    if not temporal_signals:
        return 0.0
    return _finite_ratio(sum(temporal_signals) / len(temporal_signals), default=0.0)


def _build_pose_candidate_match(
    *,
    candidate_index: int,
    raw_keypoints: list[float],
    raw_score: float,
    box: AICameraBox,
    primary_person_box: AICameraBox | None,
    parsed_keypoints: _ParsedKeypoints,
    frame_width: int | None,
    frame_height: int | None,
    previous_selected_box: AICameraBox | None,
    previous_selected_keypoints: object | None,
    previous_parsed_keypoints: _ParsedKeypoints | None,
) -> PoseCandidateMatch:
    """Score one pose candidate against the detection-space primary person."""
    raw_score = _finite_float(raw_score, default=0.0)
    normalized_score = _finite_ratio(raw_score, default=0.0)

    overlap, center_similarity, size_similarity, candidate_bounds, primary_bounds = _safe_geometry_metrics(
        box,
        primary_person_box,
        frame_width=frame_width,
        frame_height=frame_height,
    )

    keypoint_box_iou = 0.0
    keypoint_box_center_similarity = 0.0
    keypoint_inside_ratio = 0.0
    coverage = 0.0
    aspect_similarity = 0.0

    if primary_bounds is not None:
        coverage = _primary_coverage(candidate_bounds, primary_bounds)
        aspect_similarity = _bounds_aspect_similarity(candidate_bounds, primary_bounds)
        keypoint_box_iou = _bounds_iou(parsed_keypoints.normalized_box, primary_bounds)
        keypoint_box_center_similarity = _bounds_center_similarity(parsed_keypoints.normalized_box, primary_bounds)
        keypoint_inside_ratio = _weighted_inside_ratio(parsed_keypoints.normalized_points, primary_bounds)

    if primary_person_box is None:
        spatial_score = _finite_ratio(
            0.70 * normalized_score
            + 0.15 * parsed_keypoints.confidence_mean
            + 0.15 * parsed_keypoints.count_score,
            default=0.0,
        )
    else:
        spatial_score = _finite_ratio(
            0.26 * overlap
            + 0.14 * coverage
            + 0.08 * center_similarity
            + 0.05 * size_similarity
            + 0.04 * aspect_similarity
            + 0.20 * keypoint_box_iou
            + 0.11 * keypoint_inside_ratio
            + 0.04 * keypoint_box_center_similarity
            + 0.03 * parsed_keypoints.confidence_mean
            + 0.03 * parsed_keypoints.count_score
            + 0.02 * normalized_score,
            default=0.0,
        )

        if overlap < 0.02 and coverage < 0.08 and keypoint_box_iou < 0.02 and keypoint_inside_ratio < 0.20:
            spatial_score *= 0.10
        elif overlap < 0.05 and keypoint_box_iou < 0.05 and center_similarity < 0.25:
            spatial_score *= 0.35

    temporal_score = _temporal_score(
        candidate_bounds=candidate_bounds,
        candidate_keypoint_box=parsed_keypoints.normalized_box,
        previous_selected_box=previous_selected_box,
        previous_selected_keypoints=previous_selected_keypoints,
        previous_parsed_keypoints=previous_parsed_keypoints,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    selection_score = (
        _finite_ratio(0.88 * spatial_score + 0.12 * temporal_score, default=0.0)
        if temporal_score > 0.0
        else spatial_score
    )

    return PoseCandidateMatch(
        candidate_index=candidate_index,
        raw_keypoints=_copy_keypoints(raw_keypoints),
        raw_score=raw_score,
        normalized_score=normalized_score,
        box=box,
        overlap=overlap,
        center_similarity=center_similarity,
        size_similarity=size_similarity,
        selection_score=selection_score,
    )


def score_pose_candidate(
    *,
    candidate_index: int,
    raw_keypoints: list[float],
    raw_score: float,
    box: AICameraBox,
    primary_person_box: AICameraBox | None,
    frame_width: int | None = None,
    frame_height: int | None = None,
    previous_selected_box: AICameraBox | None = None,
    previous_selected_keypoints: object | None = None,
) -> PoseCandidateMatch:
    """Score one pose candidate against the detection-space primary person.

    The function stays drop-in compatible with existing callers. New optional arguments let
    the caller provide frame dimensions and a previous selection for frontier-grade box/pose
    fusion on live video.
    """
    parsed_keypoints = _parse_keypoints(
        raw_keypoints,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    return _build_pose_candidate_match(
        candidate_index=candidate_index,
        raw_keypoints=raw_keypoints,
        raw_score=raw_score,
        box=box,
        primary_person_box=primary_person_box,
        parsed_keypoints=parsed_keypoints,
        frame_width=frame_width,
        frame_height=frame_height,
        previous_selected_box=previous_selected_box,
        previous_selected_keypoints=previous_selected_keypoints,
        previous_parsed_keypoints=None,
    )


def rank_pose_candidates(
    *,
    keypoints: list[list[float]],
    scores: list[float],
    bboxes: list[list[float]],
    primary_person_box: AICameraBox | None,
    frame_width: int,
    frame_height: int,
    previous_selected_box: AICameraBox | None = None,
    previous_selected_keypoints: object | None = None,
) -> list[PoseCandidateMatch]:
    """Return pose candidates ranked by spatial alignment to the primary person.

    Optional temporal hints let callers stabilize identity selection across frames without
    changing existing call sites.
    """
    try:
        keypoints_list, keypoints_truncated = _bounded_snapshot(keypoints, limit=_MAX_POSE_CANDIDATES)
        scores_list, scores_truncated = _bounded_snapshot(scores, limit=_MAX_POSE_CANDIDATES)
        bboxes_list, bboxes_truncated = _bounded_snapshot(bboxes, limit=_MAX_POSE_CANDIDATES)
    except TypeError:
        LOGGER.warning(
            "Invalid pose candidate containers: keypoints=%r scores=%r bboxes=%r; rejecting frame.",
            type(keypoints).__name__,
            type(scores).__name__,
            type(bboxes).__name__,
        )
        return []

    if keypoints_truncated or scores_truncated or bboxes_truncated:
        LOGGER.warning(
            "Pose candidate inputs exceeded %d entries; truncating frame for Pi-safe ranking.",
            _MAX_POSE_CANDIDATES,
        )  # BREAKING: extremely large frames are now bounded instead of being fully materialized.

    if len(keypoints_list) != len(scores_list) or len(scores_list) != len(bboxes_list):
        LOGGER.warning(
            "Pose candidate input length mismatch: keypoints=%d scores=%d bboxes=%d; rejecting frame.",
            len(keypoints_list),
            len(scores_list),
            len(bboxes_list),
        )
        return []

    try:
        safe_frame_width = int(frame_width)
        safe_frame_height = int(frame_height)
    except (TypeError, ValueError):
        LOGGER.warning(
            "Invalid frame dimensions for pose ranking: width=%r height=%r; rejecting frame.",
            frame_width,
            frame_height,
        )
        return []

    if safe_frame_width <= 0 or safe_frame_height <= 0:
        LOGGER.warning(
            "Non-positive frame dimensions for pose ranking: width=%d height=%d; rejecting frame.",
            safe_frame_width,
            safe_frame_height,
        )
        return []

    previous_parsed_keypoints = (
        _parse_keypoints(
            previous_selected_keypoints,
            frame_width=safe_frame_width,
            frame_height=safe_frame_height,
        )
        if previous_selected_keypoints is not None
        else None
    )

    candidates: list[PoseCandidateMatch] = []
    for candidate_index, (raw_keypoints, raw_score, raw_bbox) in enumerate(
        zip(keypoints_list, scores_list, bboxes_list, strict=True)
    ):
        parsed_keypoints = _parse_keypoints(
            raw_keypoints,
            frame_width=safe_frame_width,
            frame_height=safe_frame_height,
        )
        if parsed_keypoints.truncated:
            LOGGER.warning(
                "Pose candidate %d keypoints exceeded %d values; truncating evidence.",
                candidate_index,
                _MAX_KEYPOINT_VALUES,
            )

        if parsed_keypoints.point_count < _MIN_KEYPOINTS_FOR_VALID_POSE:
            LOGGER.warning(
                "Skipping pose candidate %d with only %d valid keypoints.",
                candidate_index,
                parsed_keypoints.point_count,
            )  # BREAKING: malformed or degenerate pose payloads are now dropped instead of surfacing downstream.
            continue

        try:
            box = box_from_pixel_bbox(
                raw_bbox,
                frame_width=safe_frame_width,
                frame_height=safe_frame_height,
            )
            candidates.append(
                _build_pose_candidate_match(
                    candidate_index=candidate_index,
                    raw_keypoints=raw_keypoints,
                    raw_score=raw_score,
                    box=box,
                    primary_person_box=primary_person_box,
                    parsed_keypoints=parsed_keypoints,
                    frame_width=safe_frame_width,
                    frame_height=safe_frame_height,
                    previous_selected_box=previous_selected_box,
                    previous_selected_keypoints=previous_selected_keypoints,
                    previous_parsed_keypoints=previous_parsed_keypoints,
                )
            )
        except Exception:
            LOGGER.exception("Skipping invalid pose candidate %d during ranking.", candidate_index)
            continue

    candidates.sort(
        key=lambda item: (
            -item.selection_score,
            -item.overlap,
            -item.center_similarity,
            -item.size_similarity,
            -item.normalized_score,
            -item.raw_score,
            item.candidate_index,
        )
    )
    return candidates


__all__ = [
    "PoseCandidateMatch",
    "rank_pose_candidates",
    "score_pose_candidate",
]