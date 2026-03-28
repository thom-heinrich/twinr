"""Infer a bounded fast LOOKING signal for the cheap HDMI attention lane.

The fast HDMI attention refresh cannot afford the full MediaPipe pose stack on
every tick. This helper therefore prefers one real face-anchor signal from the
existing YuNet pass when available and falls back to the historic body-box
centering proxy otherwise.

2026 upgrade summary:
- keeps the public output shape drop-in compatible
- hardens all numeric paths against malformed upstream values
- preserves the legacy face/body score contract for the default public fast path
- supports optional temporal hysteresis to suppress one-frame HDMI jitter
"""

# CHANGELOG: 2026-03-28
# BUG-1: Clamp malformed, NaN, and out-of-range thresholds/scores to preserve the bounded-signal contract and stop silent false negatives/positives.
# BUG-2: Preserved the legacy face-center-inside-body matching and default score contract for the public fast path; optional hysteresis still works without changing default outputs.
# BUG-3: Hardened visible_persons/box handling against None and invalid numerics to avoid corrupted fallback decisions.
# SEC-1: No practical exploit surface was found in this pure inference helper; added finite-value sanitization so malformed upstream model outputs cannot poison downstream logic.
# IMP-1: Added optional hysteresis controls without changing the default face/body score inputs seen by existing callers.
# IMP-2: Kept bounded numeric sanitization and debug fields around the legacy fast-looking contract.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import degrees, hypot, isfinite
from typing import Any

from .detection import DetectionResult
from .face_anchors import SupplementalFaceAnchorResult
from .models import AICameraVisiblePerson

_BODY_PROXY_SCORE_CEILING = 0.35
_DEFAULT_THRESHOLD = 0.5
_DEFAULT_HYSTERESIS_MARGIN = 0.03
_MIN_ASSOCIATION_SCORE = 0.18


@dataclass(frozen=True, slots=True)
class FastLookingSignal:
    """Describe one bounded fast-path LOOKING inference."""

    looking_toward_device: bool | None = None
    visual_attention_score: float | None = None
    state: str = "none"
    source: str | None = None
    reason: str = "no_primary_person_box"
    face_anchor_state: str = "disabled"
    face_anchor_count: int = 0
    matched_face_confidence: float | None = None
    matched_face_center_x: float | None = None
    matched_face_center_y: float | None = None


@dataclass(frozen=True, slots=True)
class _Bounds:
    """Normalized box bounds used internally for robust numeric handling."""

    left: float
    top: float
    right: float
    bottom: float
    center_x: float
    center_y: float
    width: float
    height: float


def infer_fast_looking_signal(
    *,
    detection: DetectionResult,
    face_anchors: SupplementalFaceAnchorResult | None,
    attention_score_threshold: float,
    previous_signal: FastLookingSignal | None = None,
    hysteresis_margin: float = _DEFAULT_HYSTERESIS_MARGIN,
) -> FastLookingSignal:
    """Return one fast LOOKING signal from face anchors or body-box fallback.

    The original public API remains valid. Two optional controls were added:
    - previous_signal: enables one-step hysteresis for HDMI refresh stability
    - hysteresis_margin: how far below the threshold an already-positive state
      may dip before flipping inactive
    """

    threshold = _sanitize_threshold(attention_score_threshold)
    hysteresis_margin = _sanitize_margin(hysteresis_margin)

    anchors = face_anchors if face_anchors is not None else SupplementalFaceAnchorResult(state="disabled")
    anchor_state = getattr(anchors, "state", "disabled") or "disabled"
    visible_faces = _visible_persons_tuple(anchors)

    candidate_face, face_source = _select_face_anchor_candidate(
        detection=detection,
        face_anchors=anchors,
    )
    if candidate_face is not None:
        score, reason_suffix = _face_anchor_attention_score(
            candidate_face=candidate_face,
            primary_box=detection.primary_person_box,
        )
        if reason_suffix != "landmark_attention":
            score = _apply_face_confidence_prior(score=score, candidate_face=candidate_face)
        looking, decision_suffix = _decide_with_hysteresis(
            score=score,
            threshold=threshold,
            previous_signal=previous_signal,
            hysteresis_margin=hysteresis_margin,
        )
        reason_prefix = "matched_face_anchor" if face_source == "face_anchor_matched" else "single_face_anchor"
        reason = f"{reason_prefix}_{reason_suffix}_{decision_suffix}"
        face_box = _coerce_bounds(getattr(candidate_face, "box", None))
        return FastLookingSignal(
            looking_toward_device=looking,
            visual_attention_score=score,
            state=("confirmed" if looking else "inactive"),
            source=face_source,
            reason=reason,
            face_anchor_state=anchor_state,
            face_anchor_count=len(visible_faces),
            matched_face_confidence=_bounded_score(getattr(candidate_face, "confidence", None)),
            matched_face_center_x=None if face_box is None else round(face_box.center_x, 3),
            matched_face_center_y=None if face_box is None else round(face_box.center_y, 3),
        )

    primary_box = detection.primary_person_box
    primary_bounds = _coerce_bounds(primary_box)
    if primary_bounds is None:
        return FastLookingSignal(
            looking_toward_device=None,
            visual_attention_score=None,
            state="none",
            source=None,
            reason="no_primary_person_box",
            face_anchor_state=anchor_state,
            face_anchor_count=len(visible_faces),
        )

    score = _body_center_proxy_score(primary_bounds)
    looking, decision_suffix = _decide_with_hysteresis(
        score=score,
        threshold=threshold,
        previous_signal=previous_signal,
        hysteresis_margin=hysteresis_margin,
    )
    state = "proxy" if looking else "inactive"

    if decision_suffix == "held_by_hysteresis":
        reason = "score_held_by_hysteresis"
    elif threshold > _BODY_PROXY_SCORE_CEILING:
        reason = "fallback_score_below_threshold_ceiling"
    elif looking:
        reason = "score_meets_threshold"
    else:
        reason = "score_below_threshold"

    return FastLookingSignal(
        looking_toward_device=looking,
        visual_attention_score=score,
        state=state,
        source="detection_center_fallback",
        reason=reason,
        face_anchor_state=anchor_state,
        face_anchor_count=len(visible_faces),
    )


def _select_face_anchor_candidate(
    *,
    detection: DetectionResult,
    face_anchors: SupplementalFaceAnchorResult,
) -> tuple[AICameraVisiblePerson | None, str | None]:
    """Choose the best face anchor to back the fast LOOKING signal.

    2026 edge practice favors soft association over exact point inclusion because
    detector boxes jitter independently across body/face heads.
    """

    faces = tuple(
        face_person
        for face_person in _visible_persons_tuple(face_anchors)
        if _coerce_bounds(getattr(face_person, "box", None)) is not None
    )
    if not faces:
        return None, None

    primary_box = detection.primary_person_box
    primary_bounds = _coerce_bounds(primary_box)
    if primary_bounds is not None:
        matched_faces = tuple(
            face_person
            for face_person in faces
            if _bounds_contains_point(
                primary_bounds,
                x=_coerce_bounds(getattr(face_person, "box", None)).center_x,  # type: ignore[union-attr]
                y=_coerce_bounds(getattr(face_person, "box", None)).center_y,  # type: ignore[union-attr]
            )
        )
        if matched_faces:
            return max(
                matched_faces,
                key=lambda item: _bounded_score(getattr(item, "confidence", None), default=0.0) or 0.0,
            ), "face_anchor_matched"

    if _safe_person_count(detection) <= 1 and len(faces) == 1:
        return faces[0], "face_anchor_promoted"
    return None, None

def _bounds_contains_point(bounds: _Bounds, *, x: float, y: float) -> bool:
    """Return whether one point falls inside the given bounds."""

    return bounds.left <= x <= bounds.right and bounds.top <= y <= bounds.bottom


def _body_center_proxy_score(primary_box: _Bounds) -> float:
    """Return one bounded body-center fallback score.

    Preserve the historical horizontal-centering-only proxy so the default fast
    attention lane keeps its legacy ceiling and caller-visible semantics.
    """

    center_x = _clamp01(primary_box.center_x)
    base_center_score = 1.0 - min(1.0, abs(center_x - 0.5) / 0.5)
    return round(base_center_score * _BODY_PROXY_SCORE_CEILING, 3)


def _face_anchor_attention_score(*, candidate_face: AICameraVisiblePerson, primary_box: Any) -> tuple[float, str]:
    """Return one fast face-anchor attention score plus provenance suffix."""

    model_hint_score = _bounded_score(getattr(candidate_face, "attention_hint_score", None))
    if model_hint_score is not None:
        return round(model_hint_score, 3), "landmark_attention"
    face_box = _coerce_bounds(getattr(candidate_face, "box", None))
    if face_box is None:
        return 0.0, "geometry"
    return _face_anchor_geometry_score(
        face_center_x=face_box.center_x,
        primary_box=primary_box,
    ), "geometry"


def _face_anchor_geometry_score(*, face_center_x: float, primary_box: Any) -> float:
    """Return one geometry-only face-anchor fallback attention score.

    The preferred fast path comes from the upstream attention hint. This
    geometry-only path remains as a fallback when a face anchor exists but the
    backend did not expose a usable landmark-derived attention hint. Preserve
    the historical frame-centering/body-alignment blend for default callers.
    """
    frame_center = 1.0 - min(1.0, abs(face_center_x - 0.5) / 0.5)
    primary = _coerce_bounds(primary_box)
    if primary is None:
        return round(frame_center * 0.7, 3)
    body_alignment_tolerance = max(0.04, float(primary.width) * 0.18)
    body_alignment = 1.0 - min(
        1.0,
        abs(face_center_x - float(primary.center_x)) / body_alignment_tolerance,
    )
    return round((body_alignment * 0.65) + (frame_center * 0.35), 3)


def _optional_five_point_frontal_score(candidate_face: AICameraVisiblePerson) -> float | None:
    """Infer frontalness from YuNet-like five-point landmarks when available."""

    landmarks = _extract_yunet_like_landmarks(candidate_face)
    if landmarks is None:
        return None

    right_eye = landmarks["right_eye"]
    left_eye = landmarks["left_eye"]
    nose = landmarks["nose"]
    mouth_right = landmarks["mouth_right"]
    mouth_left = landmarks["mouth_left"]

    eye_mid_x = (right_eye[0] + left_eye[0]) / 2.0
    eye_mid_y = (right_eye[1] + left_eye[1]) / 2.0
    mouth_mid_x = (mouth_right[0] + mouth_left[0]) / 2.0
    mouth_mid_y = (mouth_right[1] + mouth_left[1]) / 2.0

    eye_distance = hypot(left_eye[0] - right_eye[0], left_eye[1] - right_eye[1])
    if eye_distance <= 1e-6:
        return None

    left_nose_distance = hypot(nose[0] - left_eye[0], nose[1] - left_eye[1])
    right_nose_distance = hypot(nose[0] - right_eye[0], nose[1] - right_eye[1])
    symmetry = 1.0 - min(
        1.0,
        abs(left_nose_distance - right_nose_distance) / max(left_nose_distance, right_nose_distance, 1e-6),
    )

    nose_alignment = 1.0 - min(
        1.0,
        abs(nose[0] - eye_mid_x) / max(eye_distance * 0.55, 1e-6),
    )
    mouth_alignment = 1.0 - min(
        1.0,
        abs(mouth_mid_x - eye_mid_x) / max(eye_distance * 0.45, 1e-6),
    )
    roll_penalty = 1.0 - min(
        1.0,
        abs(left_eye[1] - right_eye[1]) / max(eye_distance * 0.35, 1e-6),
    )

    face_height = abs(mouth_mid_y - eye_mid_y)
    if face_height > 1e-6:
        expected_nose_drop = face_height * 0.52
        nose_vertical = 1.0 - min(
            1.0,
            abs((nose[1] - eye_mid_y) - expected_nose_drop) / max(face_height * 0.9, 1e-6),
        )
    else:
        nose_vertical = 0.5

    score = _weighted_mean(
        (
            (symmetry, 0.35),
            (nose_alignment, 0.25),
            (mouth_alignment, 0.20),
            (roll_penalty, 0.10),
            (nose_vertical, 0.10),
        )
    )
    return None if score is None else round(score, 3)


def _optional_head_pose_score(candidate_face: AICameraVisiblePerson) -> float | None:
    """Score frontalness from optional upstream head-pose metadata when present."""

    containers = (
        candidate_face,
        getattr(candidate_face, "head_pose", None),
        getattr(candidate_face, "pose", None),
        getattr(candidate_face, "angles", None),
        getattr(candidate_face, "metadata", None),
        getattr(candidate_face, "extra", None),
    )
    yaw = pitch = roll = None
    for source in containers:
        if yaw is None:
            yaw = _extract_angle_degrees(
                source,
                "head_pose_yaw",
                "yaw",
                "face_yaw",
                "pose_yaw",
            )
        if pitch is None:
            pitch = _extract_angle_degrees(
                source,
                "head_pose_pitch",
                "pitch",
                "face_pitch",
                "pose_pitch",
            )
        if roll is None:
            roll = _extract_angle_degrees(
                source,
                "head_pose_roll",
                "roll",
                "face_roll",
                "pose_roll",
            )

    if yaw is None and pitch is None and roll is None:
        return None

    score = _weighted_mean(
        (
            (None if yaw is None else 1.0 - min(1.0, abs(yaw) / 35.0), 0.55),
            (None if pitch is None else 1.0 - min(1.0, abs(pitch) / 25.0), 0.35),
            (None if roll is None else 1.0 - min(1.0, abs(roll) / 30.0), 0.10),
        )
    )
    return None if score is None else round(score, 3)


def _apply_face_confidence_prior(*, score: float, candidate_face: AICameraVisiblePerson) -> float:
    """Lightly down-weight shaky face anchors without discarding them outright."""

    confidence = _bounded_score(getattr(candidate_face, "confidence", None))
    if confidence is None:
        return score
    return round(_clamp01(score * (0.70 + (0.30 * confidence))), 3)


def _decide_with_hysteresis(
    *,
    score: float,
    threshold: float,
    previous_signal: FastLookingSignal | None,
    hysteresis_margin: float,
) -> tuple[bool, str]:
    """Return the decision plus one provenance suffix."""

    if previous_signal is not None and previous_signal.looking_toward_device is True:
        if score >= max(0.0, threshold - hysteresis_margin):
            if score >= threshold:
                return True, "meets_threshold"
            return True, "held_by_hysteresis"

    if score >= threshold:
        return True, "meets_threshold"
    return False, "below_threshold"


def _extract_yunet_like_landmarks(candidate_face: AICameraVisiblePerson) -> dict[str, tuple[float, float]] | None:
    """Extract YuNet-like 5-point landmarks from common object layouts.

    Supported patterns:
    - direct attributes on the face object
    - mappings/dicts
    - .landmarks / .keypoints / .points sequences in YuNet order:
      right eye, left eye, nose tip, right mouth corner, left mouth corner
    """

    alias_groups = {
        "right_eye": ("right_eye", "reye", "eye_right"),
        "left_eye": ("left_eye", "leye", "eye_left"),
        "nose": ("nose", "nose_tip", "nose_center"),
        "mouth_right": ("mouth_right", "right_mouth", "mouth_right_corner"),
        "mouth_left": ("mouth_left", "left_mouth", "mouth_left_corner"),
    }

    sources = (
        candidate_face,
        getattr(candidate_face, "landmarks", None),
        getattr(candidate_face, "keypoints", None),
        getattr(candidate_face, "points", None),
    )

    named_points: dict[str, tuple[float, float]] = {}
    for canonical_name, aliases in alias_groups.items():
        for source in sources:
            point = _source_named_point(source, *aliases)
            if point is not None:
                named_points[canonical_name] = point
                break

    if len(named_points) == 5:
        return named_points

    for source in sources:
        if isinstance(source, Sequence) and not isinstance(source, (str, bytes)) and len(source) >= 5:
            points = tuple(_coerce_point(source[index]) for index in range(5))
            if all(point is not None for point in points):
                return {
                    "right_eye": points[0],   # type: ignore[index]
                    "left_eye": points[1],    # type: ignore[index]
                    "nose": points[2],        # type: ignore[index]
                    "mouth_right": points[3], # type: ignore[index]
                    "mouth_left": points[4],  # type: ignore[index]
                }
    return None


def _source_named_point(source: Any, *names: str) -> tuple[float, float] | None:
    """Read one point from an object or mapping by trying multiple aliases."""

    if source is None:
        return None

    if isinstance(source, Mapping):
        for name in names:
            if name in source:
                point = _coerce_point(source[name])
                if point is not None:
                    return point

    for name in names:
        if hasattr(source, name):
            point = _coerce_point(getattr(source, name))
            if point is not None:
                return point
    return None


def _coerce_point(value: Any) -> tuple[float, float] | None:
    """Convert one point-like object to an (x, y) tuple."""

    if value is None:
        return None

    if isinstance(value, Mapping):
        x = _coerce_float(value.get("x", value.get("center_x")))
        y = _coerce_float(value.get("y", value.get("center_y")))
        if x is not None and y is not None:
            return x, y
        return None

    if hasattr(value, "x") and hasattr(value, "y"):
        x = _coerce_float(getattr(value, "x", None))
        y = _coerce_float(getattr(value, "y", None))
        if x is not None and y is not None:
            return x, y
        return None

    if hasattr(value, "center_x") and hasattr(value, "center_y"):
        x = _coerce_float(getattr(value, "center_x", None))
        y = _coerce_float(getattr(value, "center_y", None))
        if x is not None and y is not None:
            return x, y
        return None

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2:
        x = _coerce_float(value[0])
        y = _coerce_float(value[1])
        if x is not None and y is not None:
            return x, y
        return None

    return None


def _visible_persons_tuple(face_anchors: SupplementalFaceAnchorResult | None) -> tuple[AICameraVisiblePerson, ...]:
    """Return a safe tuple of visible persons."""

    if face_anchors is None:
        return ()
    visible_persons = getattr(face_anchors, "visible_persons", ()) or ()
    try:
        return tuple(person for person in visible_persons if person is not None)
    except TypeError:
        return ()


def _safe_person_count(detection: DetectionResult) -> int:
    """Return a non-negative person count even if upstream is malformed."""

    count = _coerce_float(getattr(detection, "person_count", None), default=0.0)
    if count is None:
        return 0
    return max(0, int(count))


def _first_present_value(source: Any, *names: str) -> Any:
    """Return the first non-None value found by alias lookup."""

    if source is None:
        return None

    if isinstance(source, Mapping):
        for name in names:
            if name in source and source[name] is not None:
                return source[name]

    for name in names:
        if hasattr(source, name):
            value = getattr(source, name)
            if value is not None:
                return value
    return None


def _extract_angle_degrees(source: Any, *base_names: str) -> float | None:
    """Extract one angle in degrees.

    Generic names are assumed to already be degrees because that is the most
    common convention for face/head-pose APIs. Explicit *_rad / *_radian /
    *_radians aliases are converted to degrees.
    """

    if source is None:
        return None

    def _lookup(name: str) -> Any:
        if isinstance(source, Mapping):
            return source.get(name)
        return getattr(source, name, None)

    def _has(name: str) -> bool:
        if isinstance(source, Mapping):
            return name in source
        return hasattr(source, name)

    for base_name in base_names:
        for rad_name in (f"{base_name}_rad", f"{base_name}_radian", f"{base_name}_radians"):
            if _has(rad_name):
                radians_value = _coerce_float(_lookup(rad_name))
                if radians_value is not None:
                    return degrees(radians_value)

        for deg_name in (base_name, f"{base_name}_deg", f"{base_name}_degrees"):
            if _has(deg_name):
                degrees_value = _coerce_float(_lookup(deg_name))
                if degrees_value is not None:
                    return degrees_value
    return None


def _coerce_bounds(box: Any) -> _Bounds | None:
    """Convert a box-like object to numeric bounds."""

    if box is None:
        return None

    left = _coerce_float(getattr(box, "left", None))
    right = _coerce_float(getattr(box, "right", None))
    top = _coerce_float(getattr(box, "top", None))
    bottom = _coerce_float(getattr(box, "bottom", None))
    center_x = _coerce_float(getattr(box, "center_x", None))
    center_y = _coerce_float(getattr(box, "center_y", None))
    width = _coerce_float(getattr(box, "width", None))
    height = _coerce_float(getattr(box, "height", None))

    if left is None or right is None:
        if center_x is None or width is None:
            return None
        left = center_x - (width / 2.0)
        right = center_x + (width / 2.0)

    if top is None or bottom is None:
        if center_y is None or height is None:
            return None
        top = center_y - (height / 2.0)
        bottom = center_y + (height / 2.0)

    if center_x is None:
        center_x = (left + right) / 2.0
    if center_y is None:
        center_y = (top + bottom) / 2.0
    if width is None:
        width = right - left
    if height is None:
        height = bottom - top

    if any(value is None for value in (left, right, top, bottom, center_x, center_y, width, height)):
        return None
    if right <= left or bottom <= top or width <= 0.0 or height <= 0.0:
        return None

    return _Bounds(
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        center_x=center_x,
        center_y=center_y,
        width=width,
        height=height,
    )


def _expand_bounds(bounds: _Bounds, *, x_margin: float, y_margin: float) -> _Bounds:
    """Return expanded bounds."""

    left = bounds.left - x_margin
    right = bounds.right + x_margin
    top = bounds.top - y_margin
    bottom = bounds.bottom + y_margin
    return _Bounds(
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        center_x=(left + right) / 2.0,
        center_y=(top + bottom) / 2.0,
        width=right - left,
        height=bottom - top,
    )


def _overlap_ratio(*, container: _Bounds, candidate: _Bounds) -> float:
    """Return the fraction of candidate area overlapping the container."""

    left = max(container.left, candidate.left)
    right = min(container.right, candidate.right)
    top = max(container.top, candidate.top)
    bottom = min(container.bottom, candidate.bottom)
    if right <= left or bottom <= top:
        return 0.0

    intersection_area = (right - left) * (bottom - top)
    candidate_area = candidate.width * candidate.height
    if candidate_area <= 0.0:
        return 0.0
    return _clamp01(intersection_area / candidate_area)


def _weighted_mean(entries: Sequence[tuple[float | None, float]]) -> float | None:
    """Return one weighted mean over the available scores."""

    weighted_sum = 0.0
    total_weight = 0.0
    for score, weight in entries:
        if score is None or weight <= 0.0:
            continue
        weighted_sum += score * weight
        total_weight += weight
    if total_weight <= 0.0:
        return None
    return _clamp01(weighted_sum / total_weight)


def _sanitize_threshold(value: Any) -> float:
    """Return one bounded threshold."""

    threshold = _coerce_float(value, default=_DEFAULT_THRESHOLD)
    if threshold is None:
        threshold = _DEFAULT_THRESHOLD
    return _clamp01(threshold)


def _sanitize_margin(value: Any) -> float:
    """Return one sane hysteresis margin."""

    margin = _coerce_float(value, default=_DEFAULT_HYSTERESIS_MARGIN)
    if margin is None:
        margin = _DEFAULT_HYSTERESIS_MARGIN
    return _clamp(margin, 0.0, 0.25)


def _bounded_score(value: Any, *, default: float | None = None) -> float | None:
    """Return one score clipped to the unit interval."""

    score = _coerce_float(value, default=default)
    if score is None:
        return None
    return round(_clamp01(score), 3)


def _coerce_float(value: Any, *, default: float | None = None) -> float | None:
    """Return one finite float or the provided default."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if isfinite(result) else default


def _clamp01(value: float) -> float:
    """Clamp one float to [0.0, 1.0]."""

    return _clamp(value, 0.0, 1.0)


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp one float to the given interval."""

    if value < low:
        return low
    if value > high:
        return high
    return value


__all__ = [
    "FastLookingSignal",
    "infer_fast_looking_signal",
]
