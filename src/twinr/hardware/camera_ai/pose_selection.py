"""Rank decoded pose candidates against the authoritative person detection box."""

from __future__ import annotations

import logging  # AUDIT-FIX(#1): Module-level diagnostics for invalid frame/candidate inputs.
import math  # AUDIT-FIX(#3): Guard against NaN/inf values from upstream model outputs.
from dataclasses import dataclass

from .config import _clamp_ratio, _coerce_float
from .geometry import box_center_similarity, box_from_pixel_bbox, box_size_similarity, iou
from .models import AICameraBox

LOGGER = logging.getLogger(__name__)  # AUDIT-FIX(#1): Emit actionable diagnostics instead of failing silently.


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


def _finite_float(value: object, *, default: float = 0.0) -> float:  # AUDIT-FIX(#3): Centralize finite numeric sanitization.
    """Coerce one numeric signal to a finite float."""
    coerced = _coerce_float(value, default=default)
    if not math.isfinite(coerced):
        return default
    return coerced


def _finite_ratio(value: object, *, default: float = 0.0) -> float:  # AUDIT-FIX(#3): Clamp all rank inputs to 0..1.
    """Coerce one numeric signal to a finite 0..1 ratio."""
    return _clamp_ratio(_finite_float(value, default=default), default=default)


def _copy_keypoints(raw_keypoints: list[float]) -> list[float]:  # AUDIT-FIX(#4): Avoid aliasing caller-owned mutable keypoint lists.
    """Return a detached copy so upstream list mutations do not rewrite results."""
    try:
        return list(raw_keypoints)
    except TypeError:
        LOGGER.warning(
            "Invalid raw_keypoints payload of type %s; storing an empty keypoint list.",
            type(raw_keypoints).__name__,
        )
        return []


def score_pose_candidate(
    *,
    candidate_index: int,
    raw_keypoints: list[float],
    raw_score: float,
    box: AICameraBox,
    primary_person_box: AICameraBox | None,
) -> PoseCandidateMatch:
    """Score one pose candidate against the detection-space primary person."""

    raw_score = _finite_float(raw_score, default=0.0)  # AUDIT-FIX(#3): Normalize non-finite public-function score input.
    normalized_score = _finite_ratio(raw_score, default=0.0)  # AUDIT-FIX(#3): Clamp score to a deterministic ratio.
    overlap = 0.0
    center_similarity = 0.0
    size_similarity = 0.0
    selection_score = normalized_score
    if primary_person_box is not None:
        try:
            overlap = round(_finite_ratio(iou(box, primary_person_box), default=0.0), 3)  # AUDIT-FIX(#3): Clamp geometry metrics.
            center_similarity = _finite_ratio(box_center_similarity(box, primary_person_box), default=0.0)
            size_similarity = _finite_ratio(box_size_similarity(box, primary_person_box), default=0.0)
            selection_score = round(
                _finite_ratio(
                    0.70 * overlap + 0.20 * center_similarity + 0.08 * size_similarity + 0.02 * normalized_score,
                    default=0.0,
                ),
                3,
            )
        except Exception:
            LOGGER.exception(
                "Failed to compare pose candidate %d against the primary person box; "
                "falling back to normalized pose score.",
                candidate_index,
            )  # AUDIT-FIX(#1): One bad geometry comparison must not abort the frame.
            overlap = 0.0
            center_similarity = 0.0
            size_similarity = 0.0
            selection_score = normalized_score
    return PoseCandidateMatch(
        candidate_index=candidate_index,
        raw_keypoints=_copy_keypoints(raw_keypoints),  # AUDIT-FIX(#4): Detach result payload from caller-owned mutable lists.
        raw_score=raw_score,
        normalized_score=normalized_score,
        box=box,
        overlap=overlap,
        center_similarity=center_similarity,
        size_similarity=size_similarity,
        selection_score=selection_score,
    )


def rank_pose_candidates(
    *,
    keypoints: list[list[float]],
    scores: list[float],
    bboxes: list[list[float]],
    primary_person_box: AICameraBox | None,
    frame_width: int,
    frame_height: int,
) -> list[PoseCandidateMatch]:
    """Return pose candidates ranked by spatial alignment to the primary person."""

    try:
        keypoints_list = list(keypoints)  # AUDIT-FIX(#1): Snapshot iterable inputs before validation and ranking.
        scores_list = list(scores)
        bboxes_list = list(bboxes)
    except TypeError:
        LOGGER.warning(
            "Invalid pose candidate containers: keypoints=%r scores=%r bboxes=%r; rejecting frame.",
            type(keypoints).__name__,
            type(scores).__name__,
            type(bboxes).__name__,
        )  # AUDIT-FIX(#1): Reject non-iterable decoder output without crashing the caller.
        return []

    if len(keypoints_list) != len(scores_list) or len(scores_list) != len(bboxes_list):
        LOGGER.warning(
            "Pose candidate input length mismatch: keypoints=%d scores=%d bboxes=%d; rejecting frame.",
            len(keypoints_list),
            len(scores_list),
            len(bboxes_list),
        )  # AUDIT-FIX(#2): Fail closed instead of silently truncating inconsistent decoder output.
        return []

    try:
        safe_frame_width = int(frame_width)
        safe_frame_height = int(frame_height)
    except (TypeError, ValueError):
        LOGGER.warning(
            "Invalid frame dimensions for pose ranking: width=%r height=%r; rejecting frame.",
            frame_width,
            frame_height,
        )  # AUDIT-FIX(#1): Invalid dimensions must not reach bbox normalization code.
        return []

    if safe_frame_width <= 0 or safe_frame_height <= 0:
        LOGGER.warning(
            "Non-positive frame dimensions for pose ranking: width=%d height=%d; rejecting frame.",
            safe_frame_width,
            safe_frame_height,
        )  # AUDIT-FIX(#1): Prevent divide-by-zero and degenerate normalization.
        return []

    candidates: list[PoseCandidateMatch] = []
    for candidate_index, (raw_keypoints, raw_score, raw_bbox) in enumerate(
        zip(keypoints_list, scores_list, bboxes_list, strict=True)  # AUDIT-FIX(#2): Keep iteration aligned after length validation.
    ):
        try:
            box = box_from_pixel_bbox(
                raw_bbox,
                frame_width=safe_frame_width,
                frame_height=safe_frame_height,
            )
            candidates.append(
                score_pose_candidate(
                    candidate_index=candidate_index,
                    raw_keypoints=raw_keypoints,
                    raw_score=raw_score,
                    box=box,
                    primary_person_box=primary_person_box,
                )
            )
        except Exception:
            LOGGER.exception(
                "Skipping invalid pose candidate %d during ranking.", candidate_index
            )  # AUDIT-FIX(#1): A single malformed candidate must not crash the full ranking pass.
            continue
    candidates.sort(
        key=lambda item: (
            item.selection_score,
            item.overlap,
            item.center_similarity,
            item.size_similarity,
            item.normalized_score,
            item.raw_score,
        ),
        reverse=True,
    )
    return candidates


__all__ = [
    "PoseCandidateMatch",
    "rank_pose_candidates",
    "score_pose_candidate",
]