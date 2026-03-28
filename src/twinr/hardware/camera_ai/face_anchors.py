# CHANGELOG: 2026-03-28
# BUG-1: Fixed RGB/BGR mismatch for the YuNet/OpenCV path; OpenCV now receives BGR while MediaPipe receives SRGB.
# BUG-2: Fixed unbounded per-frame inference cost by bounding detector input size and rescaling detections back to the source frame.
# BUG-3: Fixed thread-unsafe reuse of a mutable detector instance around setInputSize()/detect() with a re-entrant lock.
# BUG-4: Preserved the legacy primary-person contract; matched face anchors still enrich visible_persons for attention, but they no longer retarget the exported primary_person_box or reorder existing detection people.
# SEC-1: Hardened the module against practical Raspberry-Pi-side DoS and path leakage by validating model files, bounding work, and sanitizing error detail.
# IMP-1: Added an optional 2026-frontier MediaPipe Tasks backend with VIDEO-mode tracking and automatic fallback to YuNet.
# IMP-2: Added backend-aware preprocessing, face-anchor deduplication, stable ordering, and dedicated supplemental_face_anchor_* runtime config keys.

"""Supplement IMX500 person anchors with bounded local face detections.

This module exists for one narrow purpose: when the SSD detector only sees one
or zero people in conversational table scenes, a lightweight local face
detector can still recover additional visible-person anchors for downstream
attention targeting. It does not identify people and it does not replace the
primary IMX500 body/object path; it only adds missing visible anchors when the
local preview frame clearly contains more faces than the SSD path reported.

The module keeps the public drop-in API from the original YuNet-only version,
but it now supports two local backends:

1. MediaPipe Tasks Face Detector (preferred when available/configured)
2. OpenCV FaceDetectorYN / YuNet (fallback and compatibility path)

Inputs:
    * ``frame_rgb``: a numpy-compatible HxWx3 RGB preview frame.
    * ``TwinrConfig``: runtime config with ``project_root`` and optional
      ``supplemental_face_anchor_*`` overrides.

Outputs:
    * ``SupplementalFaceAnchorResult`` for local face-anchor recovery
    * ``DetectionResult`` from ``merge_detection_with_face_anchors``

Hard requirements:
    * For OpenCV/YuNet: ``opencv-python`` and a local ONNX face detector model
    * For MediaPipe: ``mediapipe`` and a local ``.task`` face detector model
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import logging
import math
import threading
import time

from twinr.agent.base_agent import TwinrConfig

from .detection import DetectionResult
from .models import AICameraBox, AICameraVisiblePerson, AICameraZone


LOGGER = logging.getLogger(__name__)

_DEFAULT_DETECTOR_MODEL = "state/opencv/models/face_detection_yunet_2023mar.onnx"
_DEFAULT_MEDIAPIPE_MODEL_CANDIDATES = (
    "state/mediapipe/models/face_detector_short_range.task",
    "state/mediapipe/face_detector_short_range.task",
    "state/mediapipe/models/blaze_face_short_range.task",
    "state/mediapipe/blaze_face_short_range.task",
)
_DEFAULT_BACKEND = "auto"
_DEFAULT_SCORE_THRESHOLD = 0.82
_DEFAULT_NMS_THRESHOLD = 0.3
_DEFAULT_TOP_K = 20
_DEFAULT_MIN_FACE_HEIGHT = 0.08
_DEFAULT_MAX_INPUT_WIDTH = 320
_DEFAULT_MAX_INPUT_HEIGHT = 240
_DEFAULT_MAX_INPUT_PIXELS = _DEFAULT_MAX_INPUT_WIDTH * _DEFAULT_MAX_INPUT_HEIGHT
_DEFAULT_MAX_MODEL_BYTES = 32 * 1024 * 1024
_MAX_FACE_COUNT = 4


@dataclass(frozen=True, slots=True)
class SupplementalFaceAnchorConfig:
    """Store one bounded local face-anchor detector configuration."""

    detector_model_path: Path
    backend: str = _DEFAULT_BACKEND
    mediapipe_detector_model_path: Path | None = None
    score_threshold: float = _DEFAULT_SCORE_THRESHOLD
    nms_threshold: float = _DEFAULT_NMS_THRESHOLD
    top_k: int = _DEFAULT_TOP_K
    min_face_height: float = _DEFAULT_MIN_FACE_HEIGHT
    max_input_width: int = _DEFAULT_MAX_INPUT_WIDTH
    max_input_height: int = _DEFAULT_MAX_INPUT_HEIGHT
    max_input_pixels: int = _DEFAULT_MAX_INPUT_PIXELS
    max_model_bytes: int = _DEFAULT_MAX_MODEL_BYTES

    @classmethod
    def from_runtime_config(cls, config: TwinrConfig) -> "SupplementalFaceAnchorConfig":
        """Build one detector config from the global Twinr config."""

        project_root = Path(config.project_root)

        raw_backend = _read_config_value(
            config,
            "supplemental_face_anchor_backend",
            default=_DEFAULT_BACKEND,
        )
        backend = _normalized_backend(raw_backend)

        raw_opencv_path = _read_config_value(
            config,
            "supplemental_face_anchor_detector_model_path",
            "portrait_match_detector_model_path",
            default=_DEFAULT_DETECTOR_MODEL,
        )
        detector_model_path = _resolve_runtime_path(project_root, raw_opencv_path) or (
            project_root / _DEFAULT_DETECTOR_MODEL
        ).resolve(strict=False)

        raw_mediapipe_path = _read_config_value(
            config,
            "supplemental_face_anchor_mediapipe_model_path",
            default=None,
        )
        mediapipe_detector_model_path = _resolve_runtime_path(project_root, raw_mediapipe_path)
        if mediapipe_detector_model_path is None:
            mediapipe_detector_model_path = _first_existing_path(
                project_root,
                _DEFAULT_MEDIAPIPE_MODEL_CANDIDATES,
            )

        return cls(
            detector_model_path=detector_model_path,
            backend=backend,
            mediapipe_detector_model_path=mediapipe_detector_model_path,
            score_threshold=_clamp_float(
                _read_config_value(
                    config,
                    "supplemental_face_anchor_score_threshold",
                    default=_DEFAULT_SCORE_THRESHOLD,
                ),
                default=_DEFAULT_SCORE_THRESHOLD,
                minimum=0.0,
                maximum=1.0,
            ),
            nms_threshold=_clamp_float(
                _read_config_value(
                    config,
                    "supplemental_face_anchor_nms_threshold",
                    default=_DEFAULT_NMS_THRESHOLD,
                ),
                default=_DEFAULT_NMS_THRESHOLD,
                minimum=0.0,
                maximum=1.0,
            ),
            top_k=max(
                1,
                _safe_int(
                    _read_config_value(
                        config,
                        "supplemental_face_anchor_top_k",
                        default=_DEFAULT_TOP_K,
                    ),
                    default=_DEFAULT_TOP_K,
                ),
            ),
            min_face_height=_clamp_float(
                _read_config_value(
                    config,
                    "supplemental_face_anchor_min_face_height",
                    default=_DEFAULT_MIN_FACE_HEIGHT,
                ),
                default=_DEFAULT_MIN_FACE_HEIGHT,
                minimum=0.01,
                maximum=0.95,
            ),
            max_input_width=max(
                32,
                _safe_int(
                    _read_config_value(
                        config,
                        "supplemental_face_anchor_max_input_width",
                        default=_DEFAULT_MAX_INPUT_WIDTH,
                    ),
                    default=_DEFAULT_MAX_INPUT_WIDTH,
                ),
            ),
            max_input_height=max(
                32,
                _safe_int(
                    _read_config_value(
                        config,
                        "supplemental_face_anchor_max_input_height",
                        default=_DEFAULT_MAX_INPUT_HEIGHT,
                    ),
                    default=_DEFAULT_MAX_INPUT_HEIGHT,
                ),
            ),
            max_input_pixels=max(
                1024,
                _safe_int(
                    _read_config_value(
                        config,
                        "supplemental_face_anchor_max_input_pixels",
                        default=_DEFAULT_MAX_INPUT_PIXELS,
                    ),
                    default=_DEFAULT_MAX_INPUT_PIXELS,
                ),
            ),
            max_model_bytes=max(
                1 << 20,
                _safe_int(
                    _read_config_value(
                        config,
                        "supplemental_face_anchor_max_model_bytes",
                        default=_DEFAULT_MAX_MODEL_BYTES,
                    ),
                    default=_DEFAULT_MAX_MODEL_BYTES,
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class SupplementalFaceAnchorResult:
    """Describe one bounded face-anchor detection attempt."""

    state: str = "disabled"
    visible_persons: tuple[AICameraVisiblePerson, ...] = ()
    face_count: int = 0
    detail: str | None = None


class OpenCVFaceAnchorDetector:
    """Recover additional visible-person anchors with MediaPipe or OpenCV.

    The class keeps its legacy public name for drop-in compatibility.

    # BREAKING: when ``backend="auto"`` and a MediaPipe ``.task`` model is
    # configured and importable, this detector prefers MediaPipe Tasks over
    # YuNet because the Tasks VIDEO mode adds low-latency tracking semantics.
    """

    def __init__(self, *, config: SupplementalFaceAnchorConfig) -> None:
        self.config = config
        self._lock = threading.RLock()

        self._cv2 = None
        self._opencv_detector = None
        self._opencv_last_input_size: tuple[int, int] | None = None

        self._mediapipe = None
        self._mediapipe_face_detector = None
        self._mediapipe_last_timestamp_ms = 0

    @classmethod
    def from_runtime_config(cls, config: TwinrConfig) -> "OpenCVFaceAnchorDetector":
        """Build one detector from the global Twinr config."""

        return cls(config=SupplementalFaceAnchorConfig.from_runtime_config(config))

    def detect(self, frame_rgb: object) -> SupplementalFaceAnchorResult:
        """Detect bounded face anchors from one RGB frame."""

        np = _ensure_numpy()
        if np is None:
            return SupplementalFaceAnchorResult(
                state="backend_unavailable",
                detail="numpy_unavailable",
            )

        frame = _coerce_rgb_frame(frame_rgb, np)
        if frame is None:
            return SupplementalFaceAnchorResult(state="invalid_frame")

        height, width = frame.shape[:2]
        if width < 16 or height < 16:
            return SupplementalFaceAnchorResult(state="frame_too_small")

        resized_frame, scale_x, scale_y = _resize_frame_bounded(
            frame,
            max_width=self.config.max_input_width,
            max_height=self.config.max_input_height,
            max_pixels=self.config.max_input_pixels,
            np=np,
        )
        if resized_frame is None:
            return SupplementalFaceAnchorResult(state="invalid_frame")

        backend_order = _backend_order(self.config.backend)
        unavailable_results: list[SupplementalFaceAnchorResult] = []

        for backend_name in backend_order:
            if backend_name == "mediapipe":
                result = self._detect_with_mediapipe(
                    frame_rgb_bounded=resized_frame,
                    original_width=width,
                    original_height=height,
                    scale_x=scale_x,
                    scale_y=scale_y,
                )
            elif backend_name == "opencv":
                result = self._detect_with_opencv(
                    frame_rgb_bounded=resized_frame,
                    original_width=width,
                    original_height=height,
                    scale_x=scale_x,
                    scale_y=scale_y,
                )
            else:
                result = SupplementalFaceAnchorResult(
                    state="backend_unavailable",
                    detail=f"unsupported_backend:{backend_name}",
                )

            if result.state not in {"backend_unavailable", "model_unavailable"}:
                return result
            unavailable_results.append(result)

            if self.config.backend != "auto":
                return result

        if unavailable_results:
            return unavailable_results[-1]
        return SupplementalFaceAnchorResult(state="disabled")

    def _detect_with_mediapipe(
        self,
        *,
        frame_rgb_bounded: Any,
        original_width: int,
        original_height: int,
        scale_x: float,
        scale_y: float,
    ) -> SupplementalFaceAnchorResult:
        model_path = self.config.mediapipe_detector_model_path
        if model_path is None:
            return SupplementalFaceAnchorResult(
                state="model_unavailable",
                detail="mediapipe_model_missing",
            )
        try:
            mp, detector = self._ensure_mediapipe_face_detector()
        except (ImportError, ModuleNotFoundError) as exc:
            return SupplementalFaceAnchorResult(
                state="backend_unavailable",
                detail=f"mediapipe_import_failed:{type(exc).__name__}",
            )
        except (FileNotFoundError, ValueError) as exc:
            return SupplementalFaceAnchorResult(
                state="model_unavailable",
                detail=str(exc),
            )
        except Exception as exc:  # pragma: no cover - safety fallback
            LOGGER.exception("Failed to initialize MediaPipe face detector")
            return SupplementalFaceAnchorResult(
                state="backend_unavailable",
                detail=f"mediapipe_init_failed:{type(exc).__name__}",
            )

        try:
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame_rgb_bounded,
            )
            with self._lock:
                timestamp_ms = self._next_mediapipe_timestamp_ms()
                result = detector.detect_for_video(mp_image, timestamp_ms)
        except Exception as exc:
            LOGGER.debug("MediaPipe face detection failed", exc_info=True)
            return SupplementalFaceAnchorResult(
                state="backend_unavailable",
                detail=f"mediapipe_detect_failed:{type(exc).__name__}",
            )

        detections = tuple(getattr(result, "detections", ()) or ())
        if not detections:
            return SupplementalFaceAnchorResult(state="no_face_detected")

        visible_people: list[AICameraVisiblePerson] = []
        for detection in detections:
            person = _visible_person_from_mediapipe_detection(
                detection,
                original_width=original_width,
                original_height=original_height,
                scale_x=scale_x,
                scale_y=scale_y,
                min_face_height=self.config.min_face_height,
            )
            if person is None:
                continue
            visible_people.append(person)

        deduped_people = _dedupe_visible_people(visible_people)
        deduped_people.sort(key=_person_sort_key, reverse=True)

        return SupplementalFaceAnchorResult(
            state=("ok" if deduped_people else "no_face_detected"),
            visible_persons=tuple(deduped_people[:_MAX_FACE_COUNT]),
            face_count=len(deduped_people),
            detail="backend:mediapipe",
        )

    def _detect_with_opencv(
        self,
        *,
        frame_rgb_bounded: Any,
        original_width: int,
        original_height: int,
        scale_x: float,
        scale_y: float,
    ) -> SupplementalFaceAnchorResult:
        try:
            cv2 = self._ensure_cv2()
            detector = self._ensure_opencv_detector(cv2)
        except (ImportError, ModuleNotFoundError) as exc:
            return SupplementalFaceAnchorResult(
                state="backend_unavailable",
                detail=f"opencv_import_failed:{type(exc).__name__}",
            )
        except (FileNotFoundError, ValueError) as exc:
            return SupplementalFaceAnchorResult(
                state="model_unavailable",
                detail=str(exc),
            )
        except Exception as exc:  # pragma: no cover - safety fallback
            LOGGER.exception("Failed to initialize OpenCV face detector")
            return SupplementalFaceAnchorResult(
                state="backend_unavailable",
                detail=f"opencv_init_failed:{type(exc).__name__}",
            )

        bounded_height, bounded_width = frame_rgb_bounded.shape[:2]

        try:
            # BUG-1 fix: FaceDetectorYN is exercised in OpenCV's own samples with
            # BGR frames from cv.imread()/VideoCapture.read(), not RGB frames.
            frame_bgr_bounded = cv2.cvtColor(frame_rgb_bounded, cv2.COLOR_RGB2BGR)
            with self._lock:
                input_size = (int(bounded_width), int(bounded_height))
                if self._opencv_last_input_size != input_size:
                    detector.setInputSize(input_size)
                    self._opencv_last_input_size = input_size
                _retval, faces = detector.detect(frame_bgr_bounded)
        except Exception as exc:
            LOGGER.debug("OpenCV YuNet face detection failed", exc_info=True)
            return SupplementalFaceAnchorResult(
                state="backend_unavailable",
                detail=f"opencv_detect_failed:{type(exc).__name__}",
            )

        if faces is None:
            return SupplementalFaceAnchorResult(state="no_face_detected")

        try:
            face_rows = list(faces)
        except Exception:
            return SupplementalFaceAnchorResult(state="invalid_faces")

        visible_people: list[AICameraVisiblePerson] = []
        for row in face_rows:
            rescaled_row = _rescale_yunet_face_row(
                row,
                scale_x=scale_x,
                scale_y=scale_y,
            )
            person = _visible_person_from_face_row(
                rescaled_row,
                frame_width=original_width,
                frame_height=original_height,
                min_face_height=self.config.min_face_height,
            )
            if person is None:
                continue
            visible_people.append(person)

        deduped_people = _dedupe_visible_people(visible_people)
        deduped_people.sort(key=_person_sort_key, reverse=True)

        return SupplementalFaceAnchorResult(
            state=("ok" if deduped_people else "no_face_detected"),
            visible_persons=tuple(deduped_people[:_MAX_FACE_COUNT]),
            face_count=len(deduped_people),
            detail="backend:opencv",
        )

    def _ensure_cv2(self):
        if self._cv2 is None:
            import cv2  # type: ignore[import-not-found]

            self._cv2 = cv2
        return self._cv2

    def _ensure_opencv_detector(self, cv2):
        if self._opencv_detector is None:
            path = _validated_model_path(
                self.config.detector_model_path,
                allowed_suffixes={".onnx"},
                max_model_bytes=self.config.max_model_bytes,
                detail_prefix="opencv_model",
            )
            with self._lock:
                if self._opencv_detector is None:
                    self._opencv_detector = cv2.FaceDetectorYN.create(
                        str(path),
                        "",
                        (self.config.max_input_width, self.config.max_input_height),
                        self.config.score_threshold,
                        self.config.nms_threshold,
                        self.config.top_k,
                    )
                    self._opencv_last_input_size = None
        return self._opencv_detector

    def _ensure_mediapipe_face_detector(self):
        if self._mediapipe_face_detector is None:
            path = _validated_model_path(
                self.config.mediapipe_detector_model_path,
                allowed_suffixes={".task"},
                max_model_bytes=self.config.max_model_bytes,
                detail_prefix="mediapipe_model",
            )
            import mediapipe as mp  # type: ignore[import-not-found]

            BaseOptions = mp.tasks.BaseOptions
            FaceDetector = mp.tasks.vision.FaceDetector
            FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
            RunningMode = mp.tasks.vision.RunningMode

            with self._lock:
                if self._mediapipe_face_detector is None:
                    self._mediapipe = mp
                    self._mediapipe_face_detector = FaceDetector.create_from_options(
                        FaceDetectorOptions(
                            base_options=BaseOptions(model_asset_path=str(path)),
                            running_mode=RunningMode.VIDEO,
                            min_detection_confidence=self.config.score_threshold,
                            min_suppression_threshold=self.config.nms_threshold,
                        )
                    )
        return self._mediapipe, self._mediapipe_face_detector

    def _next_mediapipe_timestamp_ms(self) -> int:
        now_ms = time.monotonic_ns() // 1_000_000
        if now_ms <= self._mediapipe_last_timestamp_ms:
            now_ms = self._mediapipe_last_timestamp_ms + 1
        self._mediapipe_last_timestamp_ms = now_ms
        return now_ms


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
    body centroid, while preserving the legacy primary-person/body-box export.
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
            existing_person = merged_people[match_index]
            merged_people[match_index] = _attention_enriched_person(
                existing=existing_person,
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

    box = _normalized_box_from_pixels(
        x=x,
        y=y,
        width=width,
        height=height,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    if box is None or box.height < min_face_height:
        return None

    try:
        right_eye_x = float(row[4]) / frame_width
        left_eye_x = float(row[6]) / frame_width
        nose_x = float(row[8]) / frame_width
        mouth_right_x = float(row[10]) / frame_width
        mouth_left_x = float(row[12]) / frame_width
    except (TypeError, ValueError, IndexError):
        attention_hint_score = None
    else:
        attention_hint_score = _attention_hint_score_from_landmarks(
            face_center_x=box.center_x,
            left_eye_x=left_eye_x,
            right_eye_x=right_eye_x,
            nose_x=nose_x,
            mouth_center_x=(mouth_left_x + mouth_right_x) / 2.0,
            mouth_span=abs(mouth_left_x - mouth_right_x),
        )

    return AICameraVisiblePerson(
        box=box,
        zone=_zone_from_center(box.center_x),
        confidence=max(0.0, min(1.0, score)),
        attention_hint_score=attention_hint_score,
    )


def _visible_person_from_mediapipe_detection(
    detection: object,
    *,
    original_width: int,
    original_height: int,
    scale_x: float,
    scale_y: float,
    min_face_height: float,
) -> AICameraVisiblePerson | None:
    """Convert one MediaPipe detection into one visible-person anchor."""

    bounding_box = getattr(detection, "bounding_box", None)
    if bounding_box is None:
        return None

    try:
        x = float(getattr(bounding_box, "origin_x"))
        y = float(getattr(bounding_box, "origin_y"))
        width = float(getattr(bounding_box, "width"))
        height = float(getattr(bounding_box, "height"))
    except (TypeError, ValueError, AttributeError):
        return None

    box = _normalized_box_from_pixels(
        x=x * scale_x,
        y=y * scale_y,
        width=width * scale_x,
        height=height * scale_y,
        frame_width=original_width,
        frame_height=original_height,
    )
    if box is None or box.height < min_face_height:
        return None

    score = _mediapipe_detection_score(detection)
    attention_hint_score = _attention_hint_score_from_mediapipe_detection(
        detection,
        face_center_x=box.center_x,
    )

    return AICameraVisiblePerson(
        box=box,
        zone=_zone_from_center(box.center_x),
        confidence=max(0.0, min(1.0, score)),
        attention_hint_score=attention_hint_score,
    )


def _mediapipe_detection_score(detection: object) -> float:
    categories = getattr(detection, "categories", None) or ()
    for category in categories:
        try:
            score = float(getattr(category, "score"))
        except (TypeError, ValueError, AttributeError):
            continue
        if math.isfinite(score):
            return score
    return 0.0


def _attention_hint_score_from_mediapipe_detection(
    detection: object,
    *,
    face_center_x: float,
) -> float | None:
    keypoints = getattr(detection, "keypoints", None)
    if not keypoints:
        keypoints = getattr(detection, "normalized_keypoints", None)
    keypoints = tuple(keypoints or ())
    if len(keypoints) < 4:
        return None

    try:
        left_eye_x = float(getattr(keypoints[0], "x"))
        right_eye_x = float(getattr(keypoints[1], "x"))
        nose_x = float(getattr(keypoints[2], "x"))
        mouth_x = float(getattr(keypoints[3], "x"))
    except (TypeError, ValueError, AttributeError):
        return None

    return _attention_hint_score_from_landmarks(
        face_center_x=face_center_x,
        left_eye_x=left_eye_x,
        right_eye_x=right_eye_x,
        nose_x=nose_x,
        mouth_center_x=mouth_x,
        mouth_span=None,
    )


def _attention_hint_score_from_landmarks(
    *,
    face_center_x: float,
    left_eye_x: float,
    right_eye_x: float,
    nose_x: float,
    mouth_center_x: float,
    mouth_span: float | None,
) -> float | None:
    values = (face_center_x, left_eye_x, right_eye_x, nose_x, mouth_center_x)
    if not all(math.isfinite(value) for value in values):
        return None

    eye_span = abs(right_eye_x - left_eye_x)
    if eye_span < 0.01:
        return None

    if mouth_span is None or not math.isfinite(mouth_span) or mouth_span < 0.01:
        mouth_span = eye_span

    eye_center_x = (left_eye_x + right_eye_x) / 2.0
    nose_eye_alignment = max(0.0, 1.0 - abs(nose_x - eye_center_x) / eye_span)
    nose_mouth_alignment = max(0.0, 1.0 - abs(nose_x - mouth_center_x) / eye_span)

    eye_left_distance = abs(nose_x - left_eye_x)
    eye_right_distance = abs(right_eye_x - nose_x)
    eye_balance = max(0.0, 1.0 - abs(eye_left_distance - eye_right_distance) / eye_span)

    center_alignment = (nose_eye_alignment * 0.7) + (nose_mouth_alignment * 0.3)
    frontal_alignment = (center_alignment * 0.45) + (eye_balance * 0.55)

    frame_offset = max(-1.0, min(1.0, (face_center_x - 0.5) / 0.5))
    off_axis_weight = min(0.85, abs(frame_offset))
    expected_centering_shift = -frame_offset * 0.45

    eye_shift = max(-1.0, min(1.0, (nose_x - eye_center_x) / eye_span))
    mouth_shift = max(-1.0, min(1.0, (nose_x - mouth_center_x) / max(eye_span, mouth_span)))
    eye_centering = max(0.0, 1.0 - abs(eye_shift - expected_centering_shift) / 0.55)
    mouth_centering = max(0.0, 1.0 - abs(mouth_shift - expected_centering_shift) / 0.65)
    center_seeking_alignment = (eye_centering * 0.7) + (mouth_centering * 0.3)

    return round(
        (frontal_alignment * (1.0 - off_axis_weight))
        + (center_seeking_alignment * off_axis_weight),
        3,
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
        if _box_contains_box_center(existing_box, face_box):
            return index
        if _box_iou(existing_box, face_box) >= 0.15:
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
        attention_hint_score=_best_attention_hint(
            face_person.attention_hint_score,
            existing.attention_hint_score,
        ),
    )


def _best_attention_hint(*values: float | None) -> float | None:
    valid = [value for value in values if value is not None and math.isfinite(value)]
    if not valid:
        return None
    return max(valid)


def _zone_from_center(center_x: float) -> AICameraZone:
    """Map one normalized x coordinate into a coarse horizontal zone."""

    if center_x <= 0.36:
        return AICameraZone.LEFT
    if center_x >= 0.64:
        return AICameraZone.RIGHT
    return AICameraZone.CENTER


def _normalized_box_from_pixels(
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    frame_width: int,
    frame_height: int,
) -> AICameraBox | None:
    if frame_width <= 0 or frame_height <= 0:
        return None
    if not all(math.isfinite(value) for value in (x, y, width, height)):
        return None
    if width <= 0.0 or height <= 0.0:
        return None

    top = max(0.0, min(1.0, y / frame_height))
    left = max(0.0, min(1.0, x / frame_width))
    bottom = max(0.0, min(1.0, (y + height) / frame_height))
    right = max(0.0, min(1.0, (x + width) / frame_width))
    if bottom <= top or right <= left:
        return None
    return AICameraBox(top=top, left=left, bottom=bottom, right=right)


def _person_sort_key(person: AICameraVisiblePerson) -> tuple[float, float, float]:
    box = person.box
    area = 0.0 if box is None else _box_area(box)
    attention = -1.0 if person.attention_hint_score is None else person.attention_hint_score
    return (person.confidence, attention, area)


def _dedupe_visible_people(
    people: Sequence[AICameraVisiblePerson],
    *,
    iou_threshold: float = 0.6,
    center_distance_threshold: float = 0.08,
) -> list[AICameraVisiblePerson]:
    kept: list[AICameraVisiblePerson] = []
    ordered = sorted(people, key=_person_sort_key, reverse=True)
    for candidate in ordered:
        candidate_box = candidate.box
        if candidate_box is None:
            kept.append(candidate)
            continue

        duplicate = False
        for existing in kept:
            existing_box = existing.box
            if existing_box is None:
                continue
            if _box_iou(candidate_box, existing_box) >= iou_threshold:
                duplicate = True
                break
            center_dx = abs(candidate_box.center_x - existing_box.center_x)
            center_dy = abs(candidate_box.center_y - existing_box.center_y)
            if center_dx <= center_distance_threshold and center_dy <= center_distance_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def _box_contains_box_center(container: AICameraBox, contained: AICameraBox) -> bool:
    return (
        container.left <= contained.center_x <= container.right
        and container.top <= contained.center_y <= container.bottom
    )


def _box_contains_person_center(container: AICameraBox, person: AICameraVisiblePerson) -> bool:
    box = person.box
    if box is None:
        return False
    return _box_contains_box_center(container, box)


def _box_area(box: AICameraBox) -> float:
    return max(0.0, box.right - box.left) * max(0.0, box.bottom - box.top)


def _box_iou(first: AICameraBox, second: AICameraBox) -> float:
    inter_left = max(first.left, second.left)
    inter_top = max(first.top, second.top)
    inter_right = min(first.right, second.right)
    inter_bottom = min(first.bottom, second.bottom)
    if inter_right <= inter_left or inter_bottom <= inter_top:
        return 0.0
    intersection = (inter_right - inter_left) * (inter_bottom - inter_top)
    union = _box_area(first) + _box_area(second) - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _read_config_value(config: TwinrConfig, *names: str, default: Any) -> Any:
    for name in names:
        value = getattr(config, name, None)
        if value is not None and value != "":
            return value
    return default


def _normalized_backend(value: object) -> str:
    text = str(value or _DEFAULT_BACKEND).strip().lower()
    if text in {"auto", "mediapipe", "opencv"}:
        return text
    return _DEFAULT_BACKEND


def _backend_order(backend: str) -> tuple[str, ...]:
    if backend == "mediapipe":
        return ("mediapipe",)
    if backend == "opencv":
        return ("opencv",)
    return ("mediapipe", "opencv")


def _clamp_float(
    value: object,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return max(minimum, min(maximum, number))


def _safe_int(value: object, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_runtime_path(project_root: Path, raw_path: object) -> Path | None:
    if raw_path is None:
        return None
    text = str(raw_path).strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = (project_root / path).resolve(strict=False)
    return path


def _first_existing_path(project_root: Path, raw_paths: Sequence[str]) -> Path | None:
    for raw_path in raw_paths:
        path = _resolve_runtime_path(project_root, raw_path)
        if path is not None and path.is_file():
            return path
    return None


def _validated_model_path(
    path: Path | None,
    *,
    allowed_suffixes: set[str],
    max_model_bytes: int,
    detail_prefix: str,
) -> Path:
    if path is None:
        raise FileNotFoundError(f"{detail_prefix}_missing")
    suffix = path.suffix.lower()
    if suffix not in allowed_suffixes:
        raise ValueError(f"{detail_prefix}_invalid_suffix:{suffix or 'none'}")
    if not path.is_file():
        raise FileNotFoundError(f"{detail_prefix}_missing:{path.name}")
    try:
        size_bytes = path.stat().st_size
    except OSError:
        raise FileNotFoundError(f"{detail_prefix}_unreadable:{path.name}") from None
    if size_bytes <= 0:
        raise ValueError(f"{detail_prefix}_empty:{path.name}")
    if size_bytes > max_model_bytes:
        raise ValueError(f"{detail_prefix}_too_large:{path.name}")
    return path


def _ensure_numpy():
    try:
        import numpy as np  # type: ignore[import-not-found]
    except (ImportError, ModuleNotFoundError):
        return None
    return np


def _coerce_rgb_frame(frame_rgb: object, np) -> Any | None:
    try:
        frame = np.asarray(frame_rgb)
    except Exception:
        return None

    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=2)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = frame[..., :3]

    if frame.ndim != 3 or frame.shape[2] != 3:
        return None

    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating):
            frame = np.clip(frame, 0.0, 255.0).astype(np.uint8, copy=False)
        elif np.issubdtype(frame.dtype, np.integer):
            frame = np.clip(frame, 0, 255).astype(np.uint8, copy=False)
        else:
            return None

    if not frame.flags.c_contiguous:
        frame = np.ascontiguousarray(frame)
    return frame


def _resize_frame_bounded(
    frame_rgb: Any,
    *,
    max_width: int,
    max_height: int,
    max_pixels: int,
    np,
) -> tuple[Any | None, float, float]:
    frame_height, frame_width = frame_rgb.shape[:2]
    if frame_width <= 0 or frame_height <= 0:
        return None, 1.0, 1.0

    scale = 1.0
    if frame_width > max_width:
        scale = min(scale, max_width / float(frame_width))
    if frame_height > max_height:
        scale = min(scale, max_height / float(frame_height))
    if (frame_width * frame_height) > max_pixels:
        scale = min(scale, math.sqrt(max_pixels / float(frame_width * frame_height)))

    if scale >= 0.999:
        return frame_rgb, 1.0, 1.0

    new_width = max(16, int(round(frame_width * scale)))
    new_height = max(16, int(round(frame_height * scale)))
    if new_width == frame_width and new_height == frame_height:
        return frame_rgb, 1.0, 1.0

    x_idx = np.linspace(0, frame_width - 1, new_width).astype(np.intp)
    y_idx = np.linspace(0, frame_height - 1, new_height).astype(np.intp)
    resized = frame_rgb[y_idx][:, x_idx]
    if not resized.flags.c_contiguous:
        resized = np.ascontiguousarray(resized)
    return resized, (frame_width / float(new_width)), (frame_height / float(new_height))


def _rescale_yunet_face_row(
    row: Sequence[object],
    *,
    scale_x: float,
    scale_y: float,
) -> tuple[object, ...]:
    scaled = list(row)
    for index in (0, 2, 4, 6, 8, 10, 12):
        if index < len(scaled):
            try:
                scaled[index] = float(scaled[index]) * scale_x
            except (TypeError, ValueError):
                return tuple(row)
    for index in (1, 3, 5, 7, 9, 11, 13):
        if index < len(scaled):
            try:
                scaled[index] = float(scaled[index]) * scale_y
            except (TypeError, ValueError):
                return tuple(row)
    return tuple(scaled)


__all__ = [
    "OpenCVFaceAnchorDetector",
    "SupplementalFaceAnchorConfig",
    "SupplementalFaceAnchorResult",
    "merge_detection_with_face_anchors",
]
