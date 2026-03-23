"""Supplement IMX500 person anchors with bounded local face detections.

This module exists for one narrow purpose: when the SSD detector only sees one
or zero people in conversational table scenes, a lightweight local face
detector can still recover additional visible-person anchors for downstream
attention targeting. It does not identify people and it does not replace the
primary IMX500 body/object path; it only adds missing visible anchors when the
local preview frame clearly contains more faces than the SSD path reported.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import logging
import math

from twinr.config import TwinrConfig

from .detection import DetectionResult
from .models import AICameraBox, AICameraVisiblePerson, AICameraZone


LOGGER = logging.getLogger(__name__)

_DEFAULT_DETECTOR_MODEL = "state/opencv/models/face_detection_yunet_2023mar.onnx"
_DEFAULT_SCORE_THRESHOLD = 0.82
_DEFAULT_NMS_THRESHOLD = 0.3
_DEFAULT_TOP_K = 20
_DEFAULT_MIN_FACE_HEIGHT = 0.08
_MAX_FACE_COUNT = 4


@dataclass(frozen=True, slots=True)
class SupplementalFaceAnchorConfig:
    """Store one bounded local face-anchor detector configuration."""

    detector_model_path: Path
    score_threshold: float = _DEFAULT_SCORE_THRESHOLD
    nms_threshold: float = _DEFAULT_NMS_THRESHOLD
    top_k: int = _DEFAULT_TOP_K
    min_face_height: float = _DEFAULT_MIN_FACE_HEIGHT

    @classmethod
    def from_runtime_config(cls, config: TwinrConfig) -> "SupplementalFaceAnchorConfig":
        """Build one detector config from the global Twinr config."""

        project_root = Path(config.project_root)
        raw_path = getattr(config, "portrait_match_detector_model_path", _DEFAULT_DETECTOR_MODEL) or _DEFAULT_DETECTOR_MODEL
        detector_model_path = Path(raw_path)
        if not detector_model_path.is_absolute():
            detector_model_path = (project_root / detector_model_path).resolve(strict=False)
        return cls(detector_model_path=detector_model_path)


@dataclass(frozen=True, slots=True)
class SupplementalFaceAnchorResult:
    """Describe one bounded face-anchor detection attempt."""

    state: str = "disabled"
    visible_persons: tuple[AICameraVisiblePerson, ...] = ()
    face_count: int = 0
    detail: str | None = None


class OpenCVFaceAnchorDetector:
    """Use OpenCV YuNet to recover additional visible-person anchors."""

    def __init__(self, *, config: SupplementalFaceAnchorConfig) -> None:
        self.config = config
        self._cv2 = None
        self._detector = None

    @classmethod
    def from_runtime_config(cls, config: TwinrConfig) -> "OpenCVFaceAnchorDetector":
        """Build one detector from the global Twinr config."""

        return cls(config=SupplementalFaceAnchorConfig.from_runtime_config(config))

    def detect(self, frame_rgb: object) -> SupplementalFaceAnchorResult:
        """Detect bounded face anchors from one RGB frame."""

        try:
            cv2 = self._ensure_dependencies()
            detector = self._ensure_detector(cv2)
        except ModuleNotFoundError:
            return SupplementalFaceAnchorResult(
                state="backend_unavailable",
                detail="opencv_python_unavailable",
            )
        except FileNotFoundError as exc:
            return SupplementalFaceAnchorResult(
                state="model_unavailable",
                detail=str(exc),
            )

        try:
            height, width = frame_rgb.shape[:2]
        except Exception:
            return SupplementalFaceAnchorResult(state="invalid_frame")
        if width < 16 or height < 16:
            return SupplementalFaceAnchorResult(state="frame_too_small")

        try:
            detector.setInputSize((int(width), int(height)))
            _retval, faces = detector.detect(frame_rgb)
        except Exception as exc:
            return SupplementalFaceAnchorResult(
                state="backend_unavailable",
                detail=f"detect_failed:{type(exc).__name__}",
            )

        if faces is None:
            return SupplementalFaceAnchorResult(state="no_face_detected")
        try:
            face_rows = list(faces[:_MAX_FACE_COUNT])
        except Exception:
            try:
                face_rows = list(faces)
            except Exception:
                return SupplementalFaceAnchorResult(state="invalid_faces")

        visible_persons: list[AICameraVisiblePerson] = []
        for row in face_rows[:_MAX_FACE_COUNT]:
            person = _visible_person_from_face_row(
                row,
                frame_width=width,
                frame_height=height,
                min_face_height=self.config.min_face_height,
            )
            if person is None:
                continue
            visible_persons.append(person)

        visible_persons.sort(
            key=lambda item: (
                item.confidence,
                0.0 if item.box is None else item.box.area,
            ),
            reverse=True,
        )
        return SupplementalFaceAnchorResult(
            state=("ok" if visible_persons else "no_face_detected"),
            visible_persons=tuple(visible_persons[:_MAX_FACE_COUNT]),
            face_count=len(visible_persons),
        )

    def _ensure_dependencies(self):
        if self._cv2 is None:
            import cv2  # type: ignore[import-not-found]

            self._cv2 = cv2
        return self._cv2

    def _ensure_detector(self, cv2):
        if self._detector is None:
            if not self.config.detector_model_path.is_file():
                raise FileNotFoundError(f"detector model missing: {self.config.detector_model_path}")
            self._detector = cv2.FaceDetectorYN.create(
                str(self.config.detector_model_path),
                "",
                (320, 320),
                self.config.score_threshold,
                self.config.nms_threshold,
                self.config.top_k,
            )
        return self._detector


def merge_detection_with_face_anchors(
    *,
    detection: DetectionResult,
    face_anchors: SupplementalFaceAnchorResult,
) -> DetectionResult:
    """Merge supplemental face anchors into one existing detection result.

    The returned ``visible_persons`` sequence is consumed as a short-lived
    attention-target surface downstream. When a face clearly sits inside an
    already detected person box, prefer the face box as that person's visible
    anchor so HDMI gaze-follow looks toward the person's head instead of the
    body centroid.
    """

    if not face_anchors.visible_persons:
        return detection

    base_people = list(detection.visible_persons)
    if not base_people and detection.primary_person_box is not None:
        base_people.append(
            AICameraVisiblePerson(
                box=detection.primary_person_box,
                zone=detection.primary_person_zone,
                confidence=1.0,
            )
        )

    merged_people = list(base_people)
    changed = False
    for face_person in face_anchors.visible_persons:
        match_index = _matching_existing_person_index(face_person, merged_people)
        if match_index is not None:
            merged_people[match_index] = _attention_enriched_person(
                existing=merged_people[match_index],
                face_person=face_person,
            )
            changed = True
            continue
        merged_people.append(face_person)
        changed = True

    if not changed:
        return detection

    primary_person_box = detection.primary_person_box
    primary_person_zone = detection.primary_person_zone
    if primary_person_box is None and merged_people:
        primary_person_box = merged_people[0].box
        primary_person_zone = merged_people[0].zone

    return DetectionResult(
        person_count=max(int(detection.person_count), len(merged_people)),
        primary_person_box=primary_person_box,
        primary_person_zone=primary_person_zone,
        visible_persons=tuple(merged_people[:_MAX_FACE_COUNT]),
        person_near_device=detection.person_near_device,
        hand_or_object_near_camera=detection.hand_or_object_near_camera,
        objects=detection.objects,
    )


def _visible_person_from_face_row(
    row: Sequence[object],
    *,
    frame_width: int,
    frame_height: int,
    min_face_height: float,
) -> AICameraVisiblePerson | None:
    """Convert one YuNet face row into one visible-person anchor."""

    if len(row) < 5:
        return None
    try:
        x = float(row[0])
        y = float(row[1])
        width = float(row[2])
        height = float(row[3])
        score = float(row[-1])
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x, y, width, height, score)):
        return None
    if width <= 0.0 or height <= 0.0 or frame_width <= 0 or frame_height <= 0:
        return None

    top = max(0.0, min(1.0, y / frame_height))
    left = max(0.0, min(1.0, x / frame_width))
    bottom = max(0.0, min(1.0, (y + height) / frame_height))
    right = max(0.0, min(1.0, (x + width) / frame_width))
    box = AICameraBox(top=top, left=left, bottom=bottom, right=right)
    if box.height < min_face_height:
        return None
    return AICameraVisiblePerson(
        box=box,
        zone=_zone_from_center(box.center_x),
        confidence=max(0.0, min(1.0, score)),
        attention_hint_score=_attention_hint_score_from_face_row(
            row,
            frame_width=frame_width,
        ),
    )


def _matches_existing_person(
    face_person: AICameraVisiblePerson,
    existing_people: Sequence[AICameraVisiblePerson],
) -> bool:
    """Return whether a supplemental face already belongs to an existing person."""

    return _matching_existing_person_index(face_person, existing_people) is not None


def _matching_existing_person_index(
    face_person: AICameraVisiblePerson,
    existing_people: Sequence[AICameraVisiblePerson],
) -> int | None:
    """Return the index of the existing person that contains this face anchor."""

    face_box = face_person.box
    if face_box is None:
        return None
    for index, existing in enumerate(existing_people):
        existing_box = existing.box
        if existing_box is None:
            continue
        if (
            existing_box.left <= face_box.center_x <= existing_box.right
            and existing_box.top <= face_box.center_y <= existing_box.bottom
        ):
            return index
    return None


def _attention_enriched_person(
    *,
    existing: AICameraVisiblePerson,
    face_person: AICameraVisiblePerson,
) -> AICameraVisiblePerson:
    """Return one visible-person anchor enriched with a matched face box."""

    if face_person.box is None:
        return existing
    return AICameraVisiblePerson(
        box=face_person.box,
        zone=face_person.zone,
        confidence=max(existing.confidence, face_person.confidence),
        attention_hint_score=face_person.attention_hint_score,
    )


def _attention_hint_score_from_face_row(
    row: Sequence[object],
    *,
    frame_width: int,
) -> float | None:
    """Derive one fast head-orientation hint from YuNet face landmarks.

    YuNet returns eye, nose, and mouth landmarks together with the face box.
    Those landmarks are cheap enough to use on every HDMI refresh and give a
    much better proxy for "looking toward Twinr" than raw face-detection
    confidence or body-box centering alone. The score stays frontal-friendly
    near the optical axis but also rewards off-axis faces whose yaw points back
    toward the device, which is the normal geometry when somebody sits to the
    left or right of the camera and still looks directly at Twinr.
    """

    if frame_width <= 0 or len(row) < 15:
        return None
    try:
        box_x = float(row[0]) / frame_width
        box_width = float(row[2]) / frame_width
        right_eye_x = float(row[4]) / frame_width
        left_eye_x = float(row[6]) / frame_width
        nose_x = float(row[8]) / frame_width
        mouth_right_x = float(row[10]) / frame_width
        mouth_left_x = float(row[12]) / frame_width
    except (TypeError, ValueError):
        return None
    values = (box_x, box_width, right_eye_x, left_eye_x, nose_x, mouth_right_x, mouth_left_x)
    if not all(math.isfinite(value) for value in values):
        return None
    face_center_x = box_x + (box_width / 2.0)
    eye_center_x = (left_eye_x + right_eye_x) / 2.0
    mouth_center_x = (mouth_left_x + mouth_right_x) / 2.0
    eye_span = abs(right_eye_x - left_eye_x)
    mouth_span = abs(mouth_left_x - mouth_right_x)
    if eye_span < 0.01 or mouth_span < 0.01:
        return None
    nose_eye_alignment = max(0.0, 1.0 - abs(nose_x - eye_center_x) / eye_span)
    nose_mouth_alignment = max(0.0, 1.0 - abs(nose_x - mouth_center_x) / eye_span)
    eye_left_distance = abs(nose_x - right_eye_x)
    eye_right_distance = abs(left_eye_x - nose_x)
    mouth_left_distance = abs(nose_x - mouth_right_x)
    mouth_right_distance = abs(mouth_left_x - nose_x)
    eye_balance = max(0.0, 1.0 - abs(eye_left_distance - eye_right_distance) / eye_span)
    mouth_balance = max(0.0, 1.0 - abs(mouth_left_distance - mouth_right_distance) / mouth_span)
    center_alignment = (nose_eye_alignment * 0.7) + (nose_mouth_alignment * 0.3)
    symmetry_alignment = (eye_balance * 0.7) + (mouth_balance * 0.3)
    frontal_alignment = (center_alignment * 0.35) + (symmetry_alignment * 0.65)

    frame_offset = max(-1.0, min(1.0, (face_center_x - 0.5) / 0.5))
    off_axis_weight = min(0.85, abs(frame_offset))
    expected_centering_shift = -frame_offset * 0.45
    eye_shift = max(-1.0, min(1.0, (nose_x - eye_center_x) / eye_span))
    mouth_shift = max(-1.0, min(1.0, (nose_x - mouth_center_x) / eye_span))
    eye_centering = max(0.0, 1.0 - abs(eye_shift - expected_centering_shift) / 0.55)
    mouth_centering = max(0.0, 1.0 - abs(mouth_shift - expected_centering_shift) / 0.55)
    center_seeking_alignment = (eye_centering * 0.7) + (mouth_centering * 0.3)
    return round(
        (frontal_alignment * (1.0 - off_axis_weight))
        + (center_seeking_alignment * off_axis_weight),
        3,
    )


def _zone_from_center(center_x: float) -> AICameraZone:
    """Map one normalized x coordinate into a coarse horizontal zone."""

    if center_x <= 0.36:
        return AICameraZone.LEFT
    if center_x >= 0.64:
        return AICameraZone.RIGHT
    return AICameraZone.CENTER


__all__ = [
    "OpenCVFaceAnchorDetector",
    "SupplementalFaceAnchorConfig",
    "SupplementalFaceAnchorResult",
    "merge_detection_with_face_anchors",
]
