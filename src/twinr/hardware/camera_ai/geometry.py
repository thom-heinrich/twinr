"""Provide shared geometry helpers for the local AI-camera stack."""
# CHANGELOG: 2026-03-28
# BUG-1: Reject payloads with extra coordinates instead of silently truncating after four values.
# BUG-2: Empty/degenerate boxes no longer score as perfect center matches.
# SEC-1: Harden parsing to fail closed on malformed payloads and string-like inputs; no standalone practical RCE/data-exfil path was found in this module.
# IMP-1: Add 2026-standard overlap metrics and format conversion helpers (IoS, GIoU, DIoU, CIoU; yxyx/xyxy/xywh/cxcywh).
# IMP-2: Cache canonical box geometry and return full-precision similarity scores for association-grade ranking.
# BREAKING: box_center_similarity() and box_size_similarity() now return full-precision floats by default.
# BREAKING: box_from_detection() and box_from_pixel_bbox() accept an optional fmt= keyword; invalid formats raise ValueError.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Literal
import math

from .models import AICameraBox, AICameraZone


_EMPTY_COORDINATE = 0.0
_ZONE_CENTER_FALLBACK = 0.5
_UNIT_MIN = 0.0
_UNIT_MAX = 1.0
_EPSILON = 1e-7
_MAX_CENTER_DISTANCE = math.sqrt(2.0)

BoxFormat = Literal["yxyx", "xyxy", "xywh", "cxcywh"]
OverlapMetric = Literal["iou", "ios", "giou", "diou", "ciou"]


@dataclass(frozen=True, slots=True)
class _CanonicalBox:
    """Internal immutable box with cached geometry."""

    top: float
    left: float
    bottom: float
    right: float
    width: float
    height: float
    area: float
    center_x: float
    center_y: float
    diagonal_squared: float

    @property
    def is_empty(self) -> bool:
        return self.area <= 0.0


def _empty_box() -> AICameraBox:
    """Return one degenerate normalized box that callers can safely ignore."""

    return AICameraBox(
        top=_EMPTY_COORDINATE,
        left=_EMPTY_COORDINATE,
        bottom=_EMPTY_COORDINATE,
        right=_EMPTY_COORDINATE,
    )


def _empty_canonical_box() -> _CanonicalBox:
    """Return one canonical empty box."""

    return _CanonicalBox(
        top=_EMPTY_COORDINATE,
        left=_EMPTY_COORDINATE,
        bottom=_EMPTY_COORDINATE,
        right=_EMPTY_COORDINATE,
        width=0.0,
        height=0.0,
        area=0.0,
        center_x=_EMPTY_COORDINATE,
        center_y=_EMPTY_COORDINATE,
        diagonal_squared=0.0,
    )


def _clamp_unit(value: float, *, default: float) -> float:
    """Clamp one scalar to the normalized frame domain."""

    if not math.isfinite(value):
        return default
    return min(_UNIT_MAX, max(_UNIT_MIN, value))


def _normalize_box_format(fmt: str) -> BoxFormat:
    """Validate one supported bounding-box format string."""

    normalized = fmt.lower()
    if normalized not in {"yxyx", "xyxy", "xywh", "cxcywh"}:
        raise ValueError(f"Unsupported box format: {fmt!r}")
    return normalized  # type: ignore[return-value]


def _iter_scalar_candidates(value: Any) -> Iterator[Any]:
    """Yield scalar candidates from one possibly nested box payload."""

    stack: list[Iterator[Any]] = [iter((value,))]
    while stack:
        try:
            current = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue

        if isinstance(current, (str, bytes, bytearray, dict, set, frozenset)):
            return

        if hasattr(current, "tolist") and not isinstance(current, (list, tuple)):
            try:
                current = current.tolist()
            except Exception:
                pass

        if isinstance(current, (list, tuple)):
            stack.append(iter(current))
            continue

        if isinstance(current, bool):
            return

        try:
            iterator = iter(current)
        except TypeError:
            yield current
            continue

        stack.append(iterator)


def _extract_four_floats(value: Any) -> tuple[float, float, float, float] | None:
    """Extract exactly four numeric coordinates from one payload."""

    items: list[float] = []
    for candidate in _iter_scalar_candidates(value):
        try:
            items.append(float(candidate))
        except (TypeError, ValueError, OverflowError):
            return None

        if len(items) > 4:
            return None

    if len(items) != 4:
        return None

    top, left, bottom, right = items
    return top, left, bottom, right


def _canonical_unit_edges(
    top: float,
    left: float,
    bottom: float,
    right: float,
) -> tuple[float, float, float, float] | None:
    """Return finite, ordered, unit-clamped edges."""

    coordinates = (top, left, bottom, right)
    if not all(math.isfinite(coord) for coord in coordinates):
        return None

    safe_top = _clamp_unit(top, default=_EMPTY_COORDINATE)
    safe_left = _clamp_unit(left, default=_EMPTY_COORDINATE)
    safe_bottom = _clamp_unit(bottom, default=_EMPTY_COORDINATE)
    safe_right = _clamp_unit(right, default=_EMPTY_COORDINATE)

    normalized_top = min(safe_top, safe_bottom)
    normalized_bottom = max(safe_top, safe_bottom)
    normalized_left = min(safe_left, safe_right)
    normalized_right = max(safe_left, safe_right)
    return normalized_top, normalized_left, normalized_bottom, normalized_right


def _width_height_from_edges(
    top: float,
    left: float,
    bottom: float,
    right: float,
) -> tuple[float, float]:
    """Return non-negative width and height for one canonical box."""

    return max(0.0, right - left), max(0.0, bottom - top)


def _canonical_box_from_edges(
    top: float,
    left: float,
    bottom: float,
    right: float,
) -> _CanonicalBox:
    """Build one cached canonical box from already-normalized edges."""

    width, height = _width_height_from_edges(top, left, bottom, right)
    area = width * height if width > 0.0 and height > 0.0 else 0.0
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    diagonal_squared = width * width + height * height
    return _CanonicalBox(
        top=top,
        left=left,
        bottom=bottom,
        right=right,
        width=width,
        height=height,
        area=area,
        center_x=center_x,
        center_y=center_y,
        diagonal_squared=diagonal_squared,
    )


def _canonical_box_from_box(box: AICameraBox) -> _CanonicalBox:
    """Read one box defensively and normalize its edges."""

    try:
        raw_top = float(box.top)
        raw_left = float(box.left)
        raw_bottom = float(box.bottom)
        raw_right = float(box.right)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return _empty_canonical_box()

    edges = _canonical_unit_edges(raw_top, raw_left, raw_bottom, raw_right)
    if edges is None:
        return _empty_canonical_box()

    return _canonical_box_from_edges(*edges)


def _is_non_negative_size(width: float, height: float) -> bool:
    """Return whether one size tuple is finite and non-negative."""

    return math.isfinite(width) and math.isfinite(height) and width >= 0.0 and height >= 0.0


def _coerce_box_edges(value: Any, *, fmt: BoxFormat) -> tuple[float, float, float, float] | None:
    """Convert one supported box payload into normalized ``top,left,bottom,right`` edges."""

    coordinates = _extract_four_floats(value)
    if coordinates is None:
        return None

    a, b, c, d = coordinates
    if fmt == "yxyx":
        top, left, bottom, right = a, b, c, d
    elif fmt == "xyxy":
        left, top, right, bottom = a, b, c, d
    elif fmt == "xywh":
        left, top, width, height = a, b, c, d
        if not _is_non_negative_size(width, height):
            return None
        right = left + width
        bottom = top + height
    else:  # fmt == "cxcywh"
        center_x, center_y, width, height = a, b, c, d
        if not _is_non_negative_size(width, height):
            return None
        half_width = width / 2.0
        half_height = height / 2.0
        left = center_x - half_width
        right = center_x + half_width
        top = center_y - half_height
        bottom = center_y + half_height

    return _canonical_unit_edges(top, left, bottom, right)


def _box_from_edges_or_empty(edges: tuple[float, float, float, float] | None) -> AICameraBox:
    """Build one public box or return one empty box if normalization failed."""

    if edges is None:
        return _empty_box()

    top, left, bottom, right = edges
    return AICameraBox(top=top, left=left, bottom=bottom, right=right)


def _intersection_and_enclosure(
    left: _CanonicalBox,
    right: _CanonicalBox,
) -> tuple[float, float, float, float, float, float]:
    """Return intersection area and enclosure geometry for two canonical boxes."""

    inter_top = max(left.top, right.top)
    inter_left = max(left.left, right.left)
    inter_bottom = min(left.bottom, right.bottom)
    inter_right = min(left.right, right.right)

    if inter_bottom <= inter_top or inter_right <= inter_left:
        intersection = 0.0
    else:
        intersection = (inter_bottom - inter_top) * (inter_right - inter_left)

    enclosure_top = min(left.top, right.top)
    enclosure_left = min(left.left, right.left)
    enclosure_bottom = max(left.bottom, right.bottom)
    enclosure_right = max(left.right, right.right)
    enclosure_area = max(0.0, enclosure_right - enclosure_left) * max(0.0, enclosure_bottom - enclosure_top)
    return intersection, enclosure_top, enclosure_left, enclosure_bottom, enclosure_right, enclosure_area


def _round_if_requested(value: float, *, round_ndigits: int | None) -> float:
    """Optionally round one score for display-only compatibility."""

    if round_ndigits is None:
        return value
    return round(value, round_ndigits)


def _iou_components(left: _CanonicalBox, right: _CanonicalBox) -> tuple[float, float, float]:
    """Return intersection, union and IoU for two canonical boxes."""

    if left.is_empty or right.is_empty:
        return 0.0, 0.0, 0.0

    intersection, _, _, _, _, _ = _intersection_and_enclosure(left, right)
    union = left.area + right.area - intersection
    if union <= 0.0:
        return intersection, 0.0, 0.0
    return intersection, union, intersection / union


def _giou_from_canonical(left: _CanonicalBox, right: _CanonicalBox) -> float:
    """Return Generalized IoU for two canonical boxes."""

    if left.is_empty or right.is_empty:
        return 0.0

    _, union, overlap = _iou_components(left, right)
    _, _, _, _, _, enclosure_area = _intersection_and_enclosure(left, right)
    if enclosure_area <= 0.0 or union <= 0.0:
        return overlap
    return overlap - ((enclosure_area - union) / enclosure_area)


def _diou_from_canonical(left: _CanonicalBox, right: _CanonicalBox) -> float:
    """Return Distance IoU for two canonical boxes."""

    if left.is_empty or right.is_empty:
        return 0.0

    _, _, overlap = _iou_components(left, right)
    _, enclosure_top, enclosure_left, enclosure_bottom, enclosure_right, _ = _intersection_and_enclosure(left, right)
    convex_width = enclosure_right - enclosure_left
    convex_height = enclosure_bottom - enclosure_top
    convex_diagonal_squared = convex_width * convex_width + convex_height * convex_height
    if convex_diagonal_squared <= 0.0:
        return overlap

    center_distance_squared = (
        (left.center_x - right.center_x) * (left.center_x - right.center_x)
        + (left.center_y - right.center_y) * (left.center_y - right.center_y)
    )
    return overlap - (center_distance_squared / (convex_diagonal_squared + _EPSILON))


def _ciou_from_canonical(left: _CanonicalBox, right: _CanonicalBox) -> float:
    """Return Complete IoU for two canonical boxes."""

    if left.is_empty or right.is_empty:
        return 0.0

    diou_value = _diou_from_canonical(left, right)
    if left.height <= 0.0 or right.height <= 0.0:
        return diou_value

    _, _, overlap = _iou_components(left, right)
    v = (4.0 / (math.pi * math.pi)) * (
        math.atan(right.width / (right.height + _EPSILON))
        - math.atan(left.width / (left.height + _EPSILON))
    ) ** 2
    alpha = v / (1.0 - overlap + v + _EPSILON)
    return diou_value - (alpha * v)


def box_is_empty(box: AICameraBox) -> bool:
    """Return whether one box is degenerate after canonicalization."""

    return _canonical_box_from_box(box).is_empty


def box_area(box: AICameraBox) -> float:
    """Return one safe normalized area for one possibly malformed box."""

    return _canonical_box_from_box(box).area


def box_center(box: AICameraBox) -> tuple[float, float] | None:
    """Return one normalized center point or ``None`` for degenerate boxes."""

    canonical = _canonical_box_from_box(box)
    if canonical.is_empty:
        return None
    return canonical.center_x, canonical.center_y


def box_to_xyxy(box: AICameraBox) -> tuple[float, float, float, float]:
    """Convert one box to normalized ``left,top,right,bottom`` coordinates."""

    canonical = _canonical_box_from_box(box)
    return canonical.left, canonical.top, canonical.right, canonical.bottom


def box_to_xywh(box: AICameraBox) -> tuple[float, float, float, float]:
    """Convert one box to normalized ``left,top,width,height`` coordinates."""

    canonical = _canonical_box_from_box(box)
    return canonical.left, canonical.top, canonical.width, canonical.height


def box_to_cxcywh(box: AICameraBox) -> tuple[float, float, float, float]:
    """Convert one box to normalized ``center_x,center_y,width,height`` coordinates."""

    canonical = _canonical_box_from_box(box)
    return canonical.center_x, canonical.center_y, canonical.width, canonical.height


def zone_from_center(center_x: float) -> AICameraZone:
    """Map one normalized ``x`` center to a coarse zone."""

    try:
        numeric_center_x = float(center_x)
    except (TypeError, ValueError, OverflowError):
        numeric_center_x = _ZONE_CENTER_FALLBACK

    safe_center_x = _clamp_unit(numeric_center_x, default=_ZONE_CENTER_FALLBACK)
    if safe_center_x < (1.0 / 3.0):
        return AICameraZone.LEFT
    if safe_center_x > (2.0 / 3.0):
        return AICameraZone.RIGHT
    return AICameraZone.CENTER


def zone_from_box(box: AICameraBox) -> AICameraZone:
    """Map one box to a coarse horizontal zone using its center."""

    center = box_center(box)
    if center is None:
        return AICameraZone.CENTER
    center_x, _ = center
    return zone_from_center(center_x)


def box_from_detection(value: Any, *, fmt: BoxFormat = "yxyx") -> AICameraBox:
    """Build one normalized box from one detector payload.

    Supported formats:
    - ``yxyx``: ``ymin,xmin,ymax,xmax`` (default; SSD / IMX500 style)
    - ``xyxy``: ``xmin,ymin,xmax,ymax``
    - ``xywh``: ``xmin,ymin,width,height``
    - ``cxcywh``: ``center_x,center_y,width,height``
    """

    edges = _coerce_box_edges(value, fmt=_normalize_box_format(fmt))
    return _box_from_edges_or_empty(edges)


def box_from_pixel_bbox(
    value: Any,
    *,
    frame_width: int,
    frame_height: int,
    fmt: BoxFormat = "yxyx",
) -> AICameraBox:
    """Build one normalized box from one pixel-space bounding-box payload."""

    if frame_width <= 0 or frame_height <= 0:
        return _empty_box()

    coordinates = _extract_four_floats(value)
    if coordinates is None:
        return _empty_box()

    a, b, c, d = coordinates
    normalized_fmt = _normalize_box_format(fmt)
    if normalized_fmt == "yxyx":
        pixel_edges = (
            a / float(frame_height),
            b / float(frame_width),
            c / float(frame_height),
            d / float(frame_width),
        )
    elif normalized_fmt == "xyxy":
        pixel_edges = (
            b / float(frame_height),
            a / float(frame_width),
            d / float(frame_height),
            c / float(frame_width),
        )
    elif normalized_fmt == "xywh":
        if not _is_non_negative_size(c, d):
            return _empty_box()
        left = a / float(frame_width)
        top = b / float(frame_height)
        right = (a + c) / float(frame_width)
        bottom = (b + d) / float(frame_height)
        pixel_edges = (top, left, bottom, right)
    else:  # cxcywh
        if not _is_non_negative_size(c, d):
            return _empty_box()
        center_x = a / float(frame_width)
        center_y = b / float(frame_height)
        width = c / float(frame_width)
        height = d / float(frame_height)
        pixel_edges = (
            center_y - height / 2.0,
            center_x - width / 2.0,
            center_y + height / 2.0,
            center_x + width / 2.0,
        )

    return _box_from_edges_or_empty(_canonical_unit_edges(*pixel_edges))


def intersection_area(left: AICameraBox, right: AICameraBox) -> float:
    """Return the normalized intersection area of two boxes."""

    left_box = _canonical_box_from_box(left)
    right_box = _canonical_box_from_box(right)
    if left_box.is_empty or right_box.is_empty:
        return 0.0

    intersection, _, _, _, _, _ = _intersection_and_enclosure(left_box, right_box)
    return intersection


def iou(left: AICameraBox, right: AICameraBox) -> float:
    """Return the normalized IoU of two boxes."""

    _, _, overlap = _iou_components(_canonical_box_from_box(left), _canonical_box_from_box(right))
    return overlap


def ios(left: AICameraBox, right: AICameraBox) -> float:
    """Return Intersection over Smaller for two boxes."""

    left_box = _canonical_box_from_box(left)
    right_box = _canonical_box_from_box(right)
    if left_box.is_empty or right_box.is_empty:
        return 0.0

    intersection, _, _ = _iou_components(left_box, right_box)
    if intersection <= 0.0:
        return 0.0

    denominator = min(left_box.area, right_box.area)
    if denominator <= 0.0:
        return 0.0
    return intersection / denominator


def generalized_iou(left: AICameraBox, right: AICameraBox) -> float:
    """Return Generalized IoU for two boxes. Can be negative for non-overlapping boxes."""

    return _giou_from_canonical(_canonical_box_from_box(left), _canonical_box_from_box(right))


def distance_iou(left: AICameraBox, right: AICameraBox) -> float:
    """Return Distance IoU for two boxes. Can be negative for non-overlapping boxes."""

    return _diou_from_canonical(_canonical_box_from_box(left), _canonical_box_from_box(right))


def complete_iou(left: AICameraBox, right: AICameraBox) -> float:
    """Return Complete IoU for two boxes. Can be negative for non-overlapping boxes."""

    return _ciou_from_canonical(_canonical_box_from_box(left), _canonical_box_from_box(right))


def overlap_score(left: AICameraBox, right: AICameraBox, *, metric: OverlapMetric = "iou") -> float:
    """Return one overlap score selected by metric name."""

    left_box = _canonical_box_from_box(left)
    right_box = _canonical_box_from_box(right)
    normalized_metric = metric.lower()
    if normalized_metric == "iou":
        _, _, overlap = _iou_components(left_box, right_box)
        return overlap
    if normalized_metric == "ios":
        if left_box.is_empty or right_box.is_empty:
            return 0.0
        intersection, _, _ = _iou_components(left_box, right_box)
        if intersection <= 0.0:
            return 0.0
        return intersection / min(left_box.area, right_box.area)
    if normalized_metric == "giou":
        return _giou_from_canonical(left_box, right_box)
    if normalized_metric == "diou":
        return _diou_from_canonical(left_box, right_box)
    if normalized_metric == "ciou":
        return _ciou_from_canonical(left_box, right_box)
    raise ValueError(f"Unsupported overlap metric: {metric!r}")


def box_center_similarity(
    left: AICameraBox,
    right: AICameraBox,
    *,
    round_ndigits: int | None = None,
) -> float:
    """Return how close two non-degenerate box centers are on a normalized frame."""

    left_box = _canonical_box_from_box(left)
    right_box = _canonical_box_from_box(right)
    if left_box.is_empty or right_box.is_empty:
        return 0.0

    distance = math.hypot(left_box.center_x - right_box.center_x, left_box.center_y - right_box.center_y)
    score = max(0.0, 1.0 - min(1.0, distance / _MAX_CENTER_DISTANCE))
    return _round_if_requested(score, round_ndigits=round_ndigits)


def box_size_similarity(
    left: AICameraBox,
    right: AICameraBox,
    *,
    round_ndigits: int | None = None,
) -> float:
    """Return one conservative width-height similarity score for two boxes."""

    left_box = _canonical_box_from_box(left)
    right_box = _canonical_box_from_box(right)
    if left_box.width <= 0.0 or right_box.width <= 0.0 or left_box.height <= 0.0 or right_box.height <= 0.0:
        return 0.0

    width_similarity = min(left_box.width, right_box.width) / max(left_box.width, right_box.width)
    height_similarity = min(left_box.height, right_box.height) / max(left_box.height, right_box.height)
    return _round_if_requested(width_similarity * height_similarity, round_ndigits=round_ndigits)


__all__ = [
    "box_area",
    "box_center",
    "box_center_similarity",
    "box_from_detection",
    "box_from_pixel_bbox",
    "box_is_empty",
    "box_size_similarity",
    "box_to_cxcywh",
    "box_to_xywh",
    "box_to_xyxy",
    "complete_iou",
    "distance_iou",
    "generalized_iou",
    "intersection_area",
    "ios",
    "iou",
    "overlap_score",
    "zone_from_box",
    "zone_from_center",
]