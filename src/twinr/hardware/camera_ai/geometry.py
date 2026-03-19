"""Provide shared geometry helpers for the local AI-camera stack."""
# ##REFACTOR: 2026-03-19##

from __future__ import annotations

from typing import Any
import math
from itertools import islice

from .models import AICameraBox, AICameraZone


_EMPTY_COORDINATE = 0.0
_ZONE_CENTER_FALLBACK = 0.5
_UNIT_MIN = 0.0
_UNIT_MAX = 1.0
_MAX_CENTER_DISTANCE = math.sqrt(2.0)


def _empty_box() -> AICameraBox:
    """Return one degenerate normalized box that callers can safely ignore."""

    return AICameraBox(
        top=_EMPTY_COORDINATE,
        left=_EMPTY_COORDINATE,
        bottom=_EMPTY_COORDINATE,
        right=_EMPTY_COORDINATE,
    )


def _clamp_unit(value: float, *, default: float) -> float:
    """Clamp one scalar to the normalized frame domain."""

    if not math.isfinite(value):
        return default
    return min(_UNIT_MAX, max(_UNIT_MIN, value))


def _extract_four_floats(value: Any) -> tuple[float, float, float, float] | None:
    """Extract exactly four numeric coordinates from one iterable payload."""

    # AUDIT-FIX(#1): Bound parsing to four numeric coordinates and fail closed
    # for malformed detector payloads instead of raising during unpack.
    try:
        iterator = iter(value)
    except TypeError:
        return None

    items: list[float] = []
    for item in islice(iterator, 4):
        try:
            items.append(float(item))
        except (TypeError, ValueError):
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

    # AUDIT-FIX(#2): Reject non-finite coordinates and canonicalize edge order
    # so invalid detector output cannot poison downstream geometry.
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


def _edges_from_box(box: AICameraBox) -> tuple[float, float, float, float]:
    """Read one box defensively and normalize its edges."""

    # AUDIT-FIX(#4): Sanitize externally provided boxes on read so one bad
    # caller cannot propagate NaNs or inverted geometry into similarity scores.
    try:
        raw_top = float(box.top)
        raw_left = float(box.left)
        raw_bottom = float(box.bottom)
        raw_right = float(box.right)
    except (AttributeError, TypeError, ValueError):
        return (
            _EMPTY_COORDINATE,
            _EMPTY_COORDINATE,
            _EMPTY_COORDINATE,
            _EMPTY_COORDINATE,
        )

    edges = _canonical_unit_edges(raw_top, raw_left, raw_bottom, raw_right)
    if edges is None:
        return (
            _EMPTY_COORDINATE,
            _EMPTY_COORDINATE,
            _EMPTY_COORDINATE,
            _EMPTY_COORDINATE,
        )
    return edges


def _width_height_from_edges(
    top: float,
    left: float,
    bottom: float,
    right: float,
) -> tuple[float, float]:
    """Return non-negative width and height for one canonical box."""

    return max(0.0, right - left), max(0.0, bottom - top)


def _area_from_box(box: AICameraBox) -> float:
    """Return one safe area for one possibly malformed box."""

    top, left, bottom, right = _edges_from_box(box)
    width, height = _width_height_from_edges(top, left, bottom, right)
    if width <= 0.0 or height <= 0.0:
        return 0.0
    return width * height


def zone_from_center(center_x: float) -> AICameraZone:
    """Map one normalized ``x`` center to a coarse zone."""

    # AUDIT-FIX(#5): Normalize invalid/out-of-range x-centers before thresholding
    # so zone assignment stays deterministic under bad upstream inputs.
    try:
        numeric_center_x = float(center_x)
    except (TypeError, ValueError):
        numeric_center_x = _ZONE_CENTER_FALLBACK

    safe_center_x = _clamp_unit(numeric_center_x, default=_ZONE_CENTER_FALLBACK)
    if safe_center_x < (1.0 / 3.0):
        return AICameraZone.LEFT
    if safe_center_x > (2.0 / 3.0):
        return AICameraZone.RIGHT
    return AICameraZone.CENTER


def box_from_detection(value: Any) -> AICameraBox:
    """Build one normalized box from the SSD ``ymin,xmin,ymax,xmax`` tensor."""

    coordinates = _extract_four_floats(value)
    if coordinates is None:
        # AUDIT-FIX(#1): Malformed detection payloads degrade to an empty box
        # so the camera pipeline can ignore the sample instead of crashing.
        return _empty_box()

    edges = _canonical_unit_edges(*coordinates)
    if edges is None:
        # AUDIT-FIX(#2): Non-finite normalized detector coordinates are treated
        # as empty detections rather than propagating invalid geometry.
        return _empty_box()

    top, left, bottom, right = edges
    return AICameraBox(top=top, left=left, bottom=bottom, right=right)


def box_from_pixel_bbox(value: list[float], *, frame_width: int, frame_height: int) -> AICameraBox:
    """Build one normalized box from one pixel-space ``ymin,xmin,ymax,xmax`` list."""

    if frame_width <= 0 or frame_height <= 0:
        # AUDIT-FIX(#3): Invalid frame geometry must fail closed; normalizing
        # against 1 hides configuration bugs and fabricates junk boxes.
        return _empty_box()

    coordinates = _extract_four_floats(value)
    if coordinates is None:
        # AUDIT-FIX(#1): Malformed pixel-space payloads must not crash the
        # perception path; return one ignorable empty box instead.
        return _empty_box()

    top, left, bottom, right = coordinates
    edges = _canonical_unit_edges(
        top / float(frame_height),
        left / float(frame_width),
        bottom / float(frame_height),
        right / float(frame_width),
    )
    if edges is None:
        # AUDIT-FIX(#2): Non-finite pixel coordinates normalize to an empty box
        # rather than leaking NaNs into later geometry computations.
        return _empty_box()

    normalized_top, normalized_left, normalized_bottom, normalized_right = edges
    return AICameraBox(
        top=normalized_top,
        left=normalized_left,
        bottom=normalized_bottom,
        right=normalized_right,
    )


def iou(left: AICameraBox, right: AICameraBox) -> float:
    """Return the normalized IoU of two boxes."""

    intersection = intersection_area(left, right)
    if intersection <= 0.0:
        return 0.0

    # AUDIT-FIX(#4): Derive area from sanitized edges instead of trusting
    # possibly malformed model properties on externally supplied boxes.
    left_area = _area_from_box(left)
    right_area = _area_from_box(right)
    union = left_area + right_area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def intersection_area(left: AICameraBox, right: AICameraBox) -> float:
    """Return the normalized intersection area of two boxes."""

    # AUDIT-FIX(#4): Always intersect canonicalized edges so invalid callers
    # degrade to zero overlap instead of producing invalid arithmetic.
    left_top, left_left, left_bottom, left_right = _edges_from_box(left)
    right_top, right_left, right_bottom, right_right = _edges_from_box(right)

    top = max(left_top, right_top)
    left_edge = max(left_left, right_left)
    bottom = min(left_bottom, right_bottom)
    right_edge = min(left_right, right_right)
    if bottom <= top or right_edge <= left_edge:
        return 0.0
    return (bottom - top) * (right_edge - left_edge)


def box_center_similarity(left: AICameraBox, right: AICameraBox) -> float:
    """Return how close two box centers are on a normalized frame."""

    # AUDIT-FIX(#4): Compute centers from sanitized edges so malformed upstream
    # boxes cannot leak NaNs into the similarity score.
    left_top, left_left, left_bottom, left_right = _edges_from_box(left)
    right_top, right_left, right_bottom, right_right = _edges_from_box(right)

    left_center_x = (left_left + left_right) / 2.0
    left_center_y = (left_top + left_bottom) / 2.0
    right_center_x = (right_left + right_right) / 2.0
    right_center_y = (right_top + right_bottom) / 2.0

    distance = math.hypot(left_center_x - right_center_x, left_center_y - right_center_y)
    return round(max(0.0, 1.0 - min(1.0, distance / _MAX_CENTER_DISTANCE)), 3)


def box_size_similarity(left: AICameraBox, right: AICameraBox) -> float:
    """Return one conservative width-height similarity score for two boxes."""

    # AUDIT-FIX(#4): Compute size from sanitized edges so degenerate or invalid
    # boxes deterministically collapse to a zero similarity score.
    left_top, left_left, left_bottom, left_right = _edges_from_box(left)
    right_top, right_left, right_bottom, right_right = _edges_from_box(right)

    left_width, left_height = _width_height_from_edges(left_top, left_left, left_bottom, left_right)
    right_width, right_height = _width_height_from_edges(right_top, right_left, right_bottom, right_right)
    if left_width <= 0.0 or right_width <= 0.0 or left_height <= 0.0 or right_height <= 0.0:
        return 0.0

    width_similarity = min(left_width, right_width) / max(left_width, right_width)
    height_similarity = min(left_height, right_height) / max(left_height, right_height)
    return round(width_similarity * height_similarity, 3)


__all__ = [
    "box_center_similarity",
    "box_from_detection",
    "box_from_pixel_bbox",
    "box_size_similarity",
    "intersection_area",
    "iou",
    "zone_from_center",
]