"""Infer a bounded fast LOOKING signal for the cheap HDMI attention lane.

The fast HDMI attention refresh cannot afford the full MediaPipe pose stack on
every tick. This helper therefore prefers one real face-anchor signal from the
existing YuNet pass when available and falls back to the historic body-box
centering proxy otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

from .detection import DetectionResult
from .face_anchors import SupplementalFaceAnchorResult
from .models import AICameraVisiblePerson

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


def infer_fast_looking_signal(
    *,
    detection: DetectionResult,
    face_anchors: SupplementalFaceAnchorResult | None,
    attention_score_threshold: float,
) -> FastLookingSignal:
    """Return one fast LOOKING signal from face anchors or body-box fallback."""

    anchors = face_anchors or SupplementalFaceAnchorResult(state="disabled")
    candidate_face, face_source = _select_face_anchor_candidate(
        detection=detection,
        face_anchors=anchors,
    )
    if candidate_face is not None:
        score, reason_suffix = _face_anchor_attention_score(
            candidate_face=candidate_face,
            primary_box=detection.primary_person_box,
        )
        looking = score >= float(attention_score_threshold)
        reason_prefix = "matched_face_anchor" if face_source == "face_anchor_matched" else "single_face_anchor"
        if looking:
            reason = f"{reason_prefix}_{reason_suffix}_meets_threshold"
        else:
            reason = f"{reason_prefix}_{reason_suffix}_below_threshold"
        return FastLookingSignal(
            looking_toward_device=looking,
            visual_attention_score=score,
            state=("confirmed" if looking else "inactive"),
            source=face_source,
            reason=reason,
            face_anchor_state=anchors.state,
            face_anchor_count=len(anchors.visible_persons),
            matched_face_confidence=round(float(candidate_face.confidence), 3),
            matched_face_center_x=None if candidate_face.box is None else round(float(candidate_face.box.center_x), 3),
            matched_face_center_y=None if candidate_face.box is None else round(float(candidate_face.box.center_y), 3),
        )

    primary_box = detection.primary_person_box
    if primary_box is None:
        return FastLookingSignal(
            looking_toward_device=None,
            visual_attention_score=None,
            state="none",
            source=None,
            reason="no_primary_person_box",
            face_anchor_state=anchors.state,
            face_anchor_count=len(anchors.visible_persons),
        )

    score = _body_center_proxy_score(primary_box.center_x)
    looking = score >= float(attention_score_threshold)
    if looking:
        reason = "score_meets_threshold"
    elif float(attention_score_threshold) > 0.35:
        reason = "fallback_score_below_threshold_ceiling"
    else:
        reason = "score_below_threshold"
    return FastLookingSignal(
        looking_toward_device=looking,
        visual_attention_score=score,
        state=("proxy" if looking else "inactive"),
        source="detection_center_fallback",
        reason=reason,
        face_anchor_state=anchors.state,
        face_anchor_count=len(anchors.visible_persons),
    )


def _select_face_anchor_candidate(
    *,
    detection: DetectionResult,
    face_anchors: SupplementalFaceAnchorResult,
) -> tuple[AICameraVisiblePerson | None, str | None]:
    """Choose the best face anchor to back the fast LOOKING signal."""

    faces = tuple(
        face_person
        for face_person in face_anchors.visible_persons
        if getattr(face_person, "box", None) is not None
    )
    if not faces:
        return None, None

    primary_box = detection.primary_person_box
    if primary_box is not None:
        matched_faces = tuple(
            face_person
            for face_person in faces
            if _box_contains_point(
                primary_box,
                x=face_person.box.center_x,  # type: ignore[union-attr]
                y=face_person.box.center_y,  # type: ignore[union-attr]
            )
        )
        if matched_faces:
            return max(matched_faces, key=lambda item: item.confidence), "face_anchor_matched"

    if detection.person_count <= 1 and len(faces) == 1:
        return faces[0], "face_anchor_promoted"
    return None, None


def _box_contains_point(box, *, x: float, y: float) -> bool:
    """Return whether one normalized point falls inside the given box."""

    return box.left <= x <= box.right and box.top <= y <= box.bottom


def _body_center_proxy_score(center_x: float) -> float:
    """Return one bounded body-center fallback score."""

    base_center_score = 1.0 - min(1.0, abs(center_x - 0.5) / 0.5)
    return round(base_center_score * 0.35, 3)


def _face_anchor_attention_score(*, candidate_face: AICameraVisiblePerson, primary_box) -> tuple[float, str]:
    """Return one fast face-anchor attention score plus provenance suffix."""

    if candidate_face.attention_hint_score is not None:
        return round(float(candidate_face.attention_hint_score), 3), "landmark_attention"
    return (
        _face_anchor_geometry_score(
            face_center_x=candidate_face.box.center_x,
            primary_box=primary_box,
        ),
        "geometry",
    )


def _face_anchor_geometry_score(*, face_center_x: float, primary_box) -> float:
    """Return one geometry-only face-anchor fallback attention score.

    The preferred fast path comes from YuNet landmarks. This geometry-only path
    remains as a fallback when a face anchor exists but the backend did not
    expose a usable landmark-derived attention hint. We score two bounded cues:
    - frame centering of the face itself
    - horizontal alignment of the face with the current body/person anchor

    This keeps clear face-to-camera alignment strong enough to confirm looking
    while suppressing side glances where the face box shifts toward one edge of
    the body box.
    """

    frame_center = 1.0 - min(1.0, abs(face_center_x - 0.5) / 0.5)
    if primary_box is None:
        return round(frame_center * 0.7, 3)
    body_alignment_tolerance = max(0.04, float(primary_box.width) * 0.18)
    body_alignment = 1.0 - min(1.0, abs(face_center_x - float(primary_box.center_x)) / body_alignment_tolerance)
    return round((body_alignment * 0.65) + (frame_center * 0.35), 3)


__all__ = [
    "FastLookingSignal",
    "infer_fast_looking_signal",
]
