"""Resolve bounded hand-landmark ROIs for the Pi-side MediaPipe camera path.

This module owns the local hand ROI selection and MediaPipe Hand Landmarker
execution that complement Twinr's IMX500 person gate. It keeps the ROI logic
bounded, maps crop-local landmarks back into full-frame normalized coordinates,
and returns typed detections that higher layers can reuse for gesture
recognition without re-implementing crop math.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any
import math

from twinr.hardware.camera_ai.mediapipe_runtime import normalize_image_data
from twinr.hardware.camera_ai.models import AICameraBox


DEFAULT_MEDIAPIPE_HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

_WRIST_SCORE_THRESHOLD = 0.35
_ELBOW_SCORE_THRESHOLD = 0.30
_SHOULDER_SCORE_THRESHOLD = 0.30


class HandRoiSource(StrEnum):
    """Describe how one hand ROI candidate was derived."""

    PRIMARY_PERSON_UPPER_BODY = "primary_person_upper_body"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"


@dataclass(frozen=True, slots=True)
class HandLandmarkWorkerConfig:
    """Store bounded runtime settings for the local hand-landmark worker."""

    model_path: str
    num_hands: int = 2
    max_roi_candidates: int = 4
    primary_person_roi_padding: float = 0.18
    primary_person_upper_body_ratio: float = 0.78
    wrist_roi_scale: float = 0.34
    min_hand_detection_confidence: float = 0.35
    min_hand_presence_confidence: float = 0.35
    min_hand_tracking_confidence: float = 0.35

    @classmethod
    def from_config(cls, config: object) -> "HandLandmarkWorkerConfig":
        """Build one worker config from a camera-config-like object."""

        return cls(
            model_path=str(getattr(config, "hand_landmarker_model_path", "") or "").strip(),
            num_hands=max(1, int(getattr(config, "num_hands", 2) or 2)),
            max_roi_candidates=max(1, int(getattr(config, "max_roi_candidates", 4) or 4)),
            primary_person_roi_padding=_clamp_ratio(
                getattr(config, "primary_person_roi_padding", 0.18),
                default=0.18,
            ),
            primary_person_upper_body_ratio=_clamp_ratio(
                getattr(config, "primary_person_upper_body_ratio", 0.78),
                default=0.78,
            ),
            wrist_roi_scale=_clamp_ratio(
                getattr(config, "wrist_roi_scale", 0.34),
                default=0.34,
            ),
            min_hand_detection_confidence=_clamp_ratio(
                getattr(config, "min_hand_detection_confidence", 0.35),
                default=0.35,
            ),
            min_hand_presence_confidence=_clamp_ratio(
                getattr(config, "min_hand_presence_confidence", 0.35),
                default=0.35,
            ),
            min_hand_tracking_confidence=_clamp_ratio(
                getattr(config, "min_hand_tracking_confidence", 0.35),
                default=0.35,
            ),
        )


@dataclass(frozen=True, slots=True)
class HandLandmarkPoint:
    """Describe one full-frame normalized hand landmark."""

    x: float
    y: float
    z: float
    score: float | None = None


@dataclass(frozen=True, slots=True)
class HandLandmarkDetection:
    """Describe one hand-landmark detection mapped back to full-frame space."""

    roi: AICameraBox
    roi_source: HandRoiSource
    handedness: str | None
    handedness_score: float | None
    landmarks: tuple[HandLandmarkPoint, ...]
    roi_frame_rgb: Any

    @property
    def confidence(self) -> float:
        """Return the best available detection confidence."""

        if self.handedness_score is not None:
            return self.handedness_score
        if not self.landmarks:
            return 0.0
        scores = [point.score for point in self.landmarks if point.score is not None]
        if not scores:
            return 0.0
        return round(sum(scores) / float(len(scores)), 3)


@dataclass(frozen=True, slots=True)
class HandLandmarkResult:
    """Describe the bounded set of hand ROI detections for one frame."""

    detections: tuple[HandLandmarkDetection, ...] = ()
    final_timestamp_ms: int | None = None

    @property
    def primary_detection(self) -> HandLandmarkDetection | None:
        """Return the highest-confidence detection when available."""

        if not self.detections:
            return None
        return self.detections[0]


@dataclass(frozen=True, slots=True)
class _HandRoiCandidate:
    """Describe one bounded hand ROI candidate before inference."""

    box: AICameraBox
    source: HandRoiSource
    priority: int


class MediaPipeHandLandmarkWorker:
    """Run MediaPipe Hand Landmarker on bounded candidate ROIs."""

    def __init__(self, *, config: HandLandmarkWorkerConfig) -> None:
        """Initialize one lazy hand-landmark worker."""

        self.config = config
        self._hand_landmarker: Any | None = None

    def close(self) -> None:
        """Close the active MediaPipe task when supported."""

        if self._hand_landmarker is None:
            return
        close_fn = getattr(self._hand_landmarker, "close", None)
        if callable(close_fn):
            close_fn()
        self._hand_landmarker = None

    def analyze(
        self,
        *,
        runtime: dict[str, Any],
        frame_rgb: Any,
        timestamp_ms: int,
        primary_person_box: AICameraBox,
        sparse_keypoints: dict[int, tuple[float, float, float]] | None = None,
    ) -> HandLandmarkResult:
        """Run hand-landmark inference on bounded ROIs for one frame."""

        candidates = _build_hand_roi_candidates(
            primary_person_box=primary_person_box,
            sparse_keypoints=sparse_keypoints or {},
            config=self.config,
        )
        if not candidates:
            return HandLandmarkResult()

        hand_landmarker = self._ensure_hand_landmarker(runtime)
        detections: list[HandLandmarkDetection] = []
        final_timestamp_ms = None
        for index, candidate in enumerate(candidates):
            crop = _crop_frame_rgb(frame_rgb, candidate.box)
            if crop is None:
                continue
            image_data = normalize_image_data(crop)
            image = runtime["mp"].Image(image_format=runtime["mp"].ImageFormat.SRGB, data=image_data)
            candidate_timestamp_ms = timestamp_ms + index
            result = hand_landmarker.detect_for_video(image, candidate_timestamp_ms)
            final_timestamp_ms = candidate_timestamp_ms
            detections.extend(
                _parse_hand_landmark_result(
                    result=result,
                    roi=candidate.box,
                    roi_source=candidate.source,
                    roi_frame_rgb=image_data,
                )
            )
        detections.sort(
            key=lambda item: (item.confidence, item.roi.area),
            reverse=True,
        )
        return HandLandmarkResult(
            detections=tuple(detections[: self.config.num_hands]),
            final_timestamp_ms=final_timestamp_ms,
        )

    def _ensure_hand_landmarker(self, runtime: dict[str, Any]) -> Any:
        """Reuse or create the configured hand landmarker."""

        if self._hand_landmarker is not None:
            return self._hand_landmarker
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"mediapipe_hand_landmarker_model_missing:{model_path}")
        vision = runtime["vision"]
        options = vision.HandLandmarkerOptions(
            base_options=runtime["BaseOptions"](model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.config.num_hands,
            min_hand_detection_confidence=self.config.min_hand_detection_confidence,
            min_hand_presence_confidence=self.config.min_hand_presence_confidence,
            min_tracking_confidence=self.config.min_hand_tracking_confidence,
        )
        self._hand_landmarker = vision.HandLandmarker.create_from_options(options)
        return self._hand_landmarker


def _build_hand_roi_candidates(
    *,
    primary_person_box: AICameraBox,
    sparse_keypoints: dict[int, tuple[float, float, float]],
    config: HandLandmarkWorkerConfig,
) -> tuple[_HandRoiCandidate, ...]:
    """Return bounded ROI candidates for one primary person."""

    candidates: list[_HandRoiCandidate] = [
        _HandRoiCandidate(
            box=_build_primary_person_upper_body_roi(
                primary_person_box=primary_person_box,
                config=config,
            ),
            source=HandRoiSource.PRIMARY_PERSON_UPPER_BODY,
            priority=2,
        )
    ]
    left_wrist_candidate = _build_wrist_roi_candidate(
        wrist_key=9,
        elbow_key=7,
        shoulder_key=5,
        primary_person_box=primary_person_box,
        sparse_keypoints=sparse_keypoints,
        config=config,
        source=HandRoiSource.LEFT_WRIST,
    )
    if left_wrist_candidate is not None:
        candidates.append(left_wrist_candidate)
    right_wrist_candidate = _build_wrist_roi_candidate(
        wrist_key=10,
        elbow_key=8,
        shoulder_key=6,
        primary_person_box=primary_person_box,
        sparse_keypoints=sparse_keypoints,
        config=config,
        source=HandRoiSource.RIGHT_WRIST,
    )
    if right_wrist_candidate is not None:
        candidates.append(right_wrist_candidate)
    candidates.sort(key=lambda item: (item.priority, -item.box.area))
    deduped: list[_HandRoiCandidate] = []
    for candidate in candidates:
        if any(_box_iou(candidate.box, existing.box) >= 0.78 for existing in deduped):
            continue
        deduped.append(candidate)
        if len(deduped) >= config.max_roi_candidates:
            break
    return tuple(deduped)


def _build_primary_person_upper_body_roi(
    *,
    primary_person_box: AICameraBox,
    config: HandLandmarkWorkerConfig,
) -> AICameraBox:
    """Expand the primary person box into one upper-body hand-search ROI."""

    horizontal_padding = primary_person_box.width * config.primary_person_roi_padding
    vertical_padding = primary_person_box.height * config.primary_person_roi_padding
    return AICameraBox(
        top=primary_person_box.top - vertical_padding,
        left=primary_person_box.left - horizontal_padding,
        bottom=(
            primary_person_box.top
            + (primary_person_box.height * config.primary_person_upper_body_ratio)
            + vertical_padding
        ),
        right=primary_person_box.right + horizontal_padding,
    )


def _build_wrist_roi_candidate(
    *,
    wrist_key: int,
    elbow_key: int,
    shoulder_key: int,
    primary_person_box: AICameraBox,
    sparse_keypoints: dict[int, tuple[float, float, float]],
    config: HandLandmarkWorkerConfig,
    source: HandRoiSource,
) -> _HandRoiCandidate | None:
    """Build one wrist-centered ROI candidate when the pose is usable."""

    wrist = sparse_keypoints.get(wrist_key)
    if wrist is None or wrist[2] < _WRIST_SCORE_THRESHOLD:
        return None
    elbow = sparse_keypoints.get(elbow_key)
    shoulder = sparse_keypoints.get(shoulder_key)
    size = max(
        primary_person_box.width * config.wrist_roi_scale,
        primary_person_box.height * config.wrist_roi_scale * 0.7,
        0.14,
    )
    if elbow is not None and elbow[2] >= _ELBOW_SCORE_THRESHOLD:
        size = max(size, _distance_2d(wrist, elbow) * 2.4)
    if shoulder is not None and shoulder[2] >= _SHOULDER_SCORE_THRESHOLD:
        size = max(size, _distance_2d(wrist, shoulder) * 1.15)
    size = min(0.48, size)
    half = size / 2.0
    return _HandRoiCandidate(
        box=AICameraBox(
            top=wrist[1] - half,
            left=wrist[0] - half,
            bottom=wrist[1] + half,
            right=wrist[0] + half,
        ),
        source=source,
        priority=0,
    )


def _parse_hand_landmark_result(
    *,
    result: Any,
    roi: AICameraBox,
    roi_source: HandRoiSource,
    roi_frame_rgb: Any,
) -> tuple[HandLandmarkDetection, ...]:
    """Map one MediaPipe hand-landmarker result into typed detections."""

    hand_landmarks = getattr(result, "hand_landmarks", None) or ()
    handedness = getattr(result, "handedness", None) or ()
    detections: list[HandLandmarkDetection] = []
    for index, local_landmarks in enumerate(hand_landmarks):
        handedness_label, handedness_score = _resolve_handedness(handedness, index=index)
        detections.append(
            HandLandmarkDetection(
                roi=roi,
                roi_source=roi_source,
                handedness=handedness_label,
                handedness_score=handedness_score,
                landmarks=tuple(
                    _project_landmark_to_full_frame(
                        landmark=landmark,
                        roi=roi,
                    )
                    for landmark in local_landmarks or ()
                ),
                roi_frame_rgb=roi_frame_rgb,
            )
        )
    return tuple(detections)


def _project_landmark_to_full_frame(*, landmark: Any, roi: AICameraBox) -> HandLandmarkPoint:
    """Map one ROI-local hand landmark back into full-frame normalized space."""

    x = roi.left + (_clamp_ratio(getattr(landmark, "x", 0.0), default=0.0) * roi.width)
    y = roi.top + (_clamp_ratio(getattr(landmark, "y", 0.0), default=0.0) * roi.height)
    score = _coerce_optional_ratio(
        getattr(landmark, "visibility", None),
        default=_coerce_optional_ratio(getattr(landmark, "presence", None), default=None),
    )
    return HandLandmarkPoint(
        x=_clamp_ratio(x, default=0.0),
        y=_clamp_ratio(y, default=0.0),
        z=_coerce_float(getattr(landmark, "z", 0.0), default=0.0),
        score=score,
    )


def _resolve_handedness(handedness: Any, *, index: int) -> tuple[str | None, float | None]:
    """Resolve the top handedness label and score for one detected hand."""

    if index >= len(handedness):
        return None, None
    categories = handedness[index] or ()
    best_label = None
    best_score = 0.0
    for category in categories:
        score = _clamp_ratio(getattr(category, "score", 0.0), default=0.0)
        if score <= best_score:
            continue
        best_score = score
        best_label = _normalize_label(getattr(category, "category_name", None))
    if best_label is None:
        return None, None
    return best_label, round(best_score, 3)


def _crop_frame_rgb(frame_rgb: Any, box: AICameraBox) -> Any | None:
    """Crop one RGB frame-like array to the requested normalized ROI box."""

    shape = getattr(frame_rgb, "shape", None)
    if not shape or len(shape) < 2:
        return None
    frame_height = int(shape[0] or 0)
    frame_width = int(shape[1] or 0)
    if frame_height <= 1 or frame_width <= 1:
        return None
    top = max(0, min(frame_height - 1, int(math.floor(box.top * frame_height))))
    left = max(0, min(frame_width - 1, int(math.floor(box.left * frame_width))))
    bottom = max(top + 1, min(frame_height, int(math.ceil(box.bottom * frame_height))))
    right = max(left + 1, min(frame_width, int(math.ceil(box.right * frame_width))))
    if bottom <= top or right <= left:
        return None
    return frame_rgb[top:bottom, left:right]


def _box_iou(first: AICameraBox, second: AICameraBox) -> float:
    """Return the intersection-over-union for two normalized boxes."""

    overlap_top = max(first.top, second.top)
    overlap_left = max(first.left, second.left)
    overlap_bottom = min(first.bottom, second.bottom)
    overlap_right = min(first.right, second.right)
    if overlap_bottom <= overlap_top or overlap_right <= overlap_left:
        return 0.0
    overlap_area = (overlap_bottom - overlap_top) * (overlap_right - overlap_left)
    union_area = first.area + second.area - overlap_area
    if union_area <= 0.0:
        return 0.0
    return overlap_area / union_area


def _normalize_label(value: object) -> str:
    """Normalize one classifier label into a stable lowercase token."""

    return "_".join(str(value or "").strip().lower().split())


def _distance_2d(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    """Return the normalized 2D distance between two keypoints."""

    return math.hypot(first[0] - second[0], first[1] - second[1])


def _clamp_ratio(value: object, *, default: float) -> float:
    """Clamp one numeric value into the unit interval."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number


def _coerce_float(value: object, *, default: float) -> float:
    """Coerce one value into a finite float."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _coerce_optional_ratio(value: object, *, default: float | None) -> float | None:
    """Coerce one optional ratio into the unit interval."""

    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return round(number, 3)


__all__ = [
    "DEFAULT_MEDIAPIPE_HAND_LANDMARKER_MODEL_URL",
    "HandLandmarkDetection",
    "HandLandmarkPoint",
    "HandLandmarkResult",
    "HandLandmarkWorkerConfig",
    "HandRoiSource",
    "MediaPipeHandLandmarkWorker",
    "_build_hand_roi_candidates",
    "_project_landmark_to_full_frame",
]
