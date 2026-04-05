"""Run MediaPipe pose and hand-gesture inference on Pi-side RGB frames."""

from __future__ import annotations

# CHANGELOG: 2026-03-28
# BUG-1: Fixed false-positive showing_intent_likely decisions by treating UNKNOWN
#        and NONE as non-concrete fine-gesture states instead of any non-NONE value.
# BUG-2: Fixed wrong-person pose selection in multi-person frames by selecting the
#        pose that best matches primary_person_box instead of always taking pose 0.
# BUG-3: Fixed incorrect hand_near_camera=False outputs on pose-miss frames even when
#        the hand worker actually returned detections.
# BUG-4: Fixed potential native task churn / lifecycle leaks by caching ROI gesture
#        recognizers at the pipeline level just like the pose and full-frame recognizers.
# SEC-1: Added bounded ROI fan-out plus bounded frame/ROI downscaling to reduce practical
#        CPU/RAM denial-of-service risk from malformed or unexpectedly large inputs on Pi 4.
# IMP-1: Exposed sparse pose world coordinates and selected_pose_index to surface the 3D
#        information current MediaPipe tasks already provide for more robust downstream use.
# IMP-2: Added cadence control and short-lived caching for the expensive full-frame gesture
#        fallback so the Pi-side hot path avoids redundant detector work between nearby frames.
# IMP-3: Added per-stage latency reporting for on-device profiling and deployment tuning.

import logging
import math
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from twinr.hardware.hand_landmarks import (
    HandLandmarkResult,
    HandLandmarkWorkerConfig,
    MediaPipeHandLandmarkWorker,
)

from .config import MediaPipeVisionConfig
from .fine_hand_gestures import (
    BUILTIN_FINE_GESTURE_MAP,
    CUSTOM_FINE_GESTURE_MAP,
    combine_builtin_and_custom_gesture_choice,
    prefer_gesture_choice,
    resolve_fine_hand_gesture,
)
from .mediapipe_runtime import MediaPipeTaskRuntime
from .models import AICameraBodyPose, AICameraBox, AICameraFineHandGesture, AICameraGestureEvent
from .pose_classification import classify_body_pose
from .pose_features import attention_score, hand_near_camera, landmark_score
from .temporal_gestures import TemporalPoseGestureClassifier


logger = logging.getLogger(__name__)


_COCO_KEYPOINT_INDEX = {
    0: 0,
    2: 1,
    5: 2,
    11: 5,
    12: 6,
    13: 7,
    14: 8,
    15: 9,
    16: 10,
    23: 11,
    24: 12,
    25: 13,
    26: 14,
    27: 15,
    28: 16,
}


def _coerce_finite_float(value: Any) -> float | None:
    """Return one finite float or ``None`` for malformed numeric values."""

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric_value):
        return None
    return numeric_value


def _coerce_confidence(value: Any) -> float | None:
    """Return one normalized confidence score in ``[0.0, 1.0]`` when valid."""

    numeric_value = _coerce_finite_float(value)
    if numeric_value is None:
        return None
    return max(0.0, min(1.0, numeric_value))


def _coerce_nonnegative_int(value: Any, *, default: int) -> int:
    """Return one bounded non-negative integer configuration value."""

    try:
        numeric_value = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, numeric_value)


def _coerce_positive_int(
    value: Any,
    *,
    default: int,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    """Return one bounded positive integer configuration value."""

    try:
        numeric_value = int(value)
    except (TypeError, ValueError):
        numeric_value = default
    numeric_value = max(minimum, numeric_value)
    if maximum is not None:
        numeric_value = min(maximum, numeric_value)
    return numeric_value


def _is_concrete_fine_gesture(gesture: AICameraFineHandGesture) -> bool:
    """Return whether one fine gesture is a real user-facing symbol."""

    return gesture not in {
        AICameraFineHandGesture.NONE,
        AICameraFineHandGesture.UNKNOWN,
    }


def _sanitize_gesture_choice(
    choice: tuple[Any, Any],
) -> tuple[AICameraFineHandGesture, float | None]:
    """Normalize one gesture-choice tuple from gesture-resolution helpers."""

    gesture, confidence = choice
    if not isinstance(gesture, AICameraFineHandGesture):
        gesture = AICameraFineHandGesture.NONE
    return gesture, _coerce_confidence(confidence)


def _is_concrete_fine_gesture_choice(
    choice: tuple[AICameraFineHandGesture, float | None],
) -> bool:
    """Return whether one gesture choice carries a real user-facing symbol."""

    return _is_concrete_fine_gesture(choice[0])


def _iterable_length_hint(values: Any) -> int | None:
    """Return the length of one iterable when cheaply available."""

    try:
        return len(values)  # type: ignore[arg-type]
    except Exception:
        return None


def _get_box_attr(box: Any, name: str) -> Any:
    """Read one attribute or dictionary key from a box-like object."""

    if box is None:
        return None
    if isinstance(box, dict):
        return box.get(name)
    return getattr(box, name, None)


def _normalize_box_xyxy(box: Any) -> tuple[float, float, float, float] | None:
    """Extract one normalized ``(x0, y0, x1, y1)`` box when possible.

    The helper accepts the common box layouts used in edge CV stacks:
    - x_min/y_min/x_max/y_max, xmin/ymin/xmax/ymax, left/top/right/bottom
    - x/y/width/height, x/y/w/h, origin_x/origin_y/width/height
    - list/tuple of four values in x0, y0, x1, y1 order
    """

    if box is None:
        return None

    if isinstance(box, (list, tuple)) and len(box) == 4:
        raw_values = [_coerce_finite_float(value) for value in box]
        if all(value is not None for value in raw_values):
            x0, y0, x1, y1 = raw_values  # type: ignore[misc]
        else:
            return None
    else:
        xyxy_name_sets = (
            ("x_min", "y_min", "x_max", "y_max"),
            ("xmin", "ymin", "xmax", "ymax"),
            ("left", "top", "right", "bottom"),
            ("x0", "y0", "x1", "y1"),
            ("min_x", "min_y", "max_x", "max_y"),
        )
        x0 = y0 = x1 = y1 = None
        for names in xyxy_name_sets:
            raw_values = [_coerce_finite_float(_get_box_attr(box, name)) for name in names]
            if all(value is not None for value in raw_values):
                x0, y0, x1, y1 = raw_values  # type: ignore[misc]
                break
        if x0 is None:
            xywh_name_sets = (
                ("x", "y", "width", "height"),
                ("x", "y", "w", "h"),
                ("left", "top", "width", "height"),
                ("origin_x", "origin_y", "width", "height"),
            )
            for names in xywh_name_sets:
                raw_values = [_coerce_finite_float(_get_box_attr(box, name)) for name in names]
                if all(value is not None for value in raw_values):
                    origin_x, origin_y, width, height = raw_values  # type: ignore[misc]
                    x0 = origin_x
                    y0 = origin_y
                    x1 = origin_x + max(0.0, width)
                    y1 = origin_y + max(0.0, height)
                    break
        if x0 is None or y0 is None or x1 is None or y1 is None:
            return None

    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0

    max_magnitude = max(abs(x0), abs(y0), abs(x1), abs(y1))
    if 2.0 < max_magnitude <= 100.0 and min(x0, y0, x1, y1) >= -10.0:
        x0 /= 100.0
        y0 /= 100.0
        x1 /= 100.0
        y1 /= 100.0
    elif max_magnitude > 2.0:
        # The pose landmarks are normalized image coordinates. If the box is clearly
        # expressed in pixels instead of normalized coordinates, we cannot safely align
        # it here because this module does not know the original frame shape.
        return None

    return x0, y0, x1, y1


def _pose_bbox_xyxy(pose_landmarks: Any) -> tuple[float, float, float, float] | None:
    """Return one normalized landmark bounding box for a pose candidate."""

    x_values: list[float] = []
    y_values: list[float] = []
    for landmark in pose_landmarks or ():
        x = _coerce_finite_float(getattr(landmark, "x", None))
        y = _coerce_finite_float(getattr(landmark, "y", None))
        if x is None or y is None:
            continue
        x_values.append(x)
        y_values.append(y)
    if not x_values or not y_values:
        return None
    return min(x_values), min(y_values), max(x_values), max(y_values)


def _bbox_center_xy(box_xyxy: tuple[float, float, float, float] | None) -> tuple[float, float] | None:
    """Return the center of one box."""

    if box_xyxy is None:
        return None
    x0, y0, x1, y1 = box_xyxy
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _box_iou(
    box_a: tuple[float, float, float, float] | None,
    box_b: tuple[float, float, float, float] | None,
) -> float:
    """Return IoU between two boxes, or ``0.0`` when invalid."""

    if box_a is None or box_b is None:
        return 0.0
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def _count_valid_landmarks(pose_landmarks: Any) -> int:
    """Return the number of landmarks with finite normalized coordinates."""

    valid_count = 0
    for landmark in pose_landmarks or ():
        x = _coerce_finite_float(getattr(landmark, "x", None))
        y = _coerce_finite_float(getattr(landmark, "y", None))
        if x is None or y is None:
            continue
        valid_count += 1
    return valid_count


def _select_pose_index(
    pose_landmarks: Any,
    *,
    preferred_box: AICameraBox | None = None,
) -> int:
    """Select the pose candidate best aligned with the upstream primary person.

    MediaPipe can return multiple poses. This pipeline already receives the
    upstream primary_person_box from the IMX500 gate, so selecting pose 0
    unconditionally silently binds the wrong skeleton whenever multiple people
    overlap the frame. Prefer the candidate with the highest overlap against
    the upstream box, and use center distance as the tiebreaker.
    """

    if not pose_landmarks:
        return 0
    if len(pose_landmarks) <= 1:
        return 0

    preferred_box_xyxy = _normalize_box_xyxy(preferred_box)
    if preferred_box_xyxy is None:
        return 0
    preferred_center = _bbox_center_xy(preferred_box_xyxy)

    best_index = 0
    best_score: tuple[float, float, int] | None = None
    for pose_index, pose in enumerate(pose_landmarks):
        pose_box_xyxy = _pose_bbox_xyxy(pose)
        iou = _box_iou(preferred_box_xyxy, pose_box_xyxy)
        pose_center = _bbox_center_xy(pose_box_xyxy)
        if preferred_center is None or pose_center is None:
            negative_distance = float("-inf")
        else:
            negative_distance = -math.dist(preferred_center, pose_center)
        score = (iou, negative_distance, _count_valid_landmarks(pose))
        if best_score is None or score > best_score:
            best_index = pose_index
            best_score = score
    return best_index


def _frame_hw(frame_rgb: Any) -> tuple[int, int] | None:
    """Return one ``(height, width)`` pair for numpy-like frame objects."""

    shape = getattr(frame_rgb, "shape", None)
    if shape is None:
        return None
    try:
        if len(shape) < 2:
            return None
        height = int(shape[0])
        width = int(shape[1])
    except Exception:
        return None
    if height <= 0 or width <= 0:
        return None
    return height, width


def _downscale_frame_stride(
    frame_rgb: Any,
    *,
    max_pixels: int | None,
) -> Any:
    """Cheaply downscale one numpy-like RGB frame by integer stride.

    MediaPipe pose/hand/gesture tasks ultimately resize into compact model
    inputs (for example, pose 224/256 and hand/gesture 192/224 in the current
    task bundles), so feeding oversized camera frames mainly increases memory
    bandwidth and CPU preprocessing cost on the Pi without adding equivalent
    signal value for these landmark tasks.
    """

    if max_pixels is None or max_pixels <= 0:
        return frame_rgb
    hw = _frame_hw(frame_rgb)
    if hw is None:
        return frame_rgb
    height, width = hw
    pixel_count = height * width
    if pixel_count <= max_pixels:
        return frame_rgb

    stride = max(2, int(math.ceil(math.sqrt(pixel_count / float(max_pixels)))))
    try:
        reduced = frame_rgb[::stride, ::stride]
        copy_method = getattr(reduced, "copy", None)
        if callable(copy_method):
            reduced = copy_method()
        return reduced
    except Exception:
        logger.exception(
            "Failed to downscale frame of shape=%r with stride=%s; using original frame",
            getattr(frame_rgb, "shape", None),
            stride,
        )
        return frame_rgb


def _extract_sparse_keypoints_from_pose(
    pose_landmarks: Any,
) -> tuple[dict[int, tuple[float, float, float]], float | None]:
    """Extract a COCO-like sparse keypoint map from one pose candidate."""

    if not pose_landmarks:
        return {}, None

    sparse: dict[int, tuple[float, float, float]] = {}
    confidence_values: list[float] = []
    for mediapipe_index, coco_index in _COCO_KEYPOINT_INDEX.items():
        if mediapipe_index >= len(pose_landmarks):
            continue
        landmark = pose_landmarks[mediapipe_index]
        if landmark is None:
            continue

        x = _coerce_finite_float(getattr(landmark, "x", None))
        y = _coerce_finite_float(getattr(landmark, "y", None))
        try:
            score = _coerce_confidence(landmark_score(landmark))
        except Exception:
            continue

        if x is None or y is None or score is None:
            continue

        sparse[coco_index] = (x, y, score)
        confidence_values.append(score)

    if not confidence_values:
        return {}, None
    return sparse, round(sum(confidence_values) / float(len(confidence_values)), 3)


def _extract_sparse_world_keypoints_from_pose(
    world_pose_landmarks: Any,
) -> dict[int, tuple[float, float, float]]:
    """Extract sparse world-space keypoints from one pose candidate."""

    if not world_pose_landmarks:
        return {}

    sparse_world: dict[int, tuple[float, float, float]] = {}
    for mediapipe_index, coco_index in _COCO_KEYPOINT_INDEX.items():
        if mediapipe_index >= len(world_pose_landmarks):
            continue
        landmark = world_pose_landmarks[mediapipe_index]
        if landmark is None:
            continue
        x = _coerce_finite_float(getattr(landmark, "x", None))
        y = _coerce_finite_float(getattr(landmark, "y", None))
        z = _coerce_finite_float(getattr(landmark, "z", None))
        if x is None or y is None or z is None:
            continue
        sparse_world[coco_index] = (x, y, z)
    return sparse_world


def _extract_sparse_keypoints_and_world(
    result: Any,
    *,
    preferred_box: AICameraBox | None = None,
) -> tuple[
    dict[int, tuple[float, float, float]],
    dict[int, tuple[float, float, float]],
    float | None,
    int | None,
]:
    """Extract sparse 2D/3D pose keypoints for the best-aligned pose candidate."""

    pose_landmarks = getattr(result, "pose_landmarks", None) or ()
    if not pose_landmarks:
        return {}, {}, None, None

    pose_index = _select_pose_index(pose_landmarks, preferred_box=preferred_box)
    if pose_index >= len(pose_landmarks):
        pose_index = 0

    selected_pose = pose_landmarks[pose_index]
    if not selected_pose:
        return {}, {}, None, None

    sparse_keypoints, pose_confidence = _extract_sparse_keypoints_from_pose(selected_pose)

    world_pose_landmarks_all = getattr(result, "pose_world_landmarks", None) or ()
    selected_world_pose = (
        world_pose_landmarks_all[pose_index]
        if pose_index < len(world_pose_landmarks_all)
        else None
    )
    sparse_world_keypoints = _extract_sparse_world_keypoints_from_pose(selected_world_pose)
    return sparse_keypoints, sparse_world_keypoints, pose_confidence, pose_index


@dataclass(frozen=True, slots=True)
class MediaPipeVisionResult:
    """Describe one bounded CPU-side MediaPipe inference result."""

    body_pose: AICameraBodyPose = AICameraBodyPose.UNKNOWN
    pose_confidence: float | None = None
    looking_toward_device: bool | None = None
    visual_attention_score: float | None = None
    hand_near_camera: bool = False
    showing_intent_likely: bool | None = None
    gesture_event: AICameraGestureEvent = AICameraGestureEvent.NONE
    gesture_confidence: float | None = None
    fine_hand_gesture: AICameraFineHandGesture = AICameraFineHandGesture.NONE
    fine_hand_gesture_confidence: float | None = None
    sparse_keypoints: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    sparse_world_keypoints: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    selected_pose_index: int | None = None
    stage_latency_ms: dict[str, float] = field(default_factory=dict)


class MediaPipeVisionPipeline:
    """Run MediaPipe pose and gesture tasks on RGB frames gated by IMX500."""

    def __init__(self, *, config: MediaPipeVisionConfig) -> None:
        """Initialize one lazy MediaPipe pipeline."""

        self.config = config
        self._runtime = MediaPipeTaskRuntime(config=config)
        self._last_timestamp_ms = 0
        self._pose_landmarker: Any | None = None
        self._hand_landmark_worker: MediaPipeHandLandmarkWorker | None = None
        self._gesture_recognizer: Any | None = None
        self._custom_gesture_recognizer: Any | None = None
        self._roi_gesture_recognizer: Any | None = None
        self._custom_roi_gesture_recognizer: Any | None = None
        self._sequence = TemporalPoseGestureClassifier(
            window_s=config.sequence_window_s,
            min_frames=config.sequence_min_frames,
        )
        self._last_fine_gesture_choice: tuple[AICameraFineHandGesture, float | None] = (
            AICameraFineHandGesture.NONE,
            None,
        )
        self._last_fine_gesture_timestamp_ms = 0
        self._last_full_frame_gesture_timestamp_ms = 0
        self._logged_pose_frame_downscale = False
        self._logged_roi_frame_downscale = False
        self._logged_roi_detection_truncation = False
        self._lock = RLock()

    def close(self) -> None:
        """Close active MediaPipe task instances when supported."""

        with self._lock:
            try:
                if self._hand_landmark_worker is not None:
                    try:
                        self._hand_landmark_worker.close()
                    except Exception:
                        logger.exception("Failed to close hand-landmark worker cleanly")

                for task_name, task in (
                    ("pose landmarker", self._pose_landmarker),
                    ("gesture recognizer", self._gesture_recognizer),
                    ("custom gesture recognizer", self._custom_gesture_recognizer),
                    ("ROI gesture recognizer", self._roi_gesture_recognizer),
                    ("custom ROI gesture recognizer", self._custom_roi_gesture_recognizer),
                ):
                    close_method = getattr(task, "close", None)
                    if not callable(close_method):
                        continue
                    try:
                        close_method()  # pylint: disable=not-callable
                    except Exception:
                        logger.exception("Failed to close %s cleanly", task_name)

                try:
                    self._runtime.close()
                except Exception:
                    logger.exception("Failed to close MediaPipe runtime cleanly")
            finally:
                self._last_timestamp_ms = 0
                self._pose_landmarker = None
                self._hand_landmark_worker = None
                self._gesture_recognizer = None
                self._custom_gesture_recognizer = None
                self._roi_gesture_recognizer = None
                self._custom_roi_gesture_recognizer = None
                self._clear_transient_state_locked()

    def reset_temporal_state(self) -> None:
        """Discard buffered temporal state and transient gesture caches."""

        with self._lock:
            self._clear_transient_state_locked()

    def analyze(
        self,
        *,
        frame_rgb: Any,
        observed_at: float,
        primary_person_box: AICameraBox,
    ) -> MediaPipeVisionResult:
        """Run the pose and gesture tasks against one RGB frame."""

        stage_latency_ms: dict[str, float] = {}
        with self._lock:
            prepare_started = time.perf_counter()
            try:
                runtime = self._load_runtime()
                prepared_frame_rgb = self._prepare_pose_and_gesture_frame(frame_rgb)
                image = self._build_image(runtime, frame_rgb=prepared_frame_rgb)
                timestamp_ms = self._timestamp_ms(observed_at)
                sequence_observed_at = _coerce_finite_float(observed_at)
                if sequence_observed_at is None:
                    sequence_observed_at = timestamp_ms / 1000.0
            except Exception:
                logger.exception("Failed to prepare MediaPipe inference inputs")
                self._clear_transient_state_locked()
                return MediaPipeVisionResult(stage_latency_ms=dict(stage_latency_ms))
            stage_latency_ms["prepare"] = round((time.perf_counter() - prepare_started) * 1000.0, 3)

            sparse_keypoints: dict[int, tuple[float, float, float]] = {}
            sparse_world_keypoints: dict[int, tuple[float, float, float]] = {}
            pose_confidence: float | None = None
            selected_pose_index: int | None = None

            pose_started = time.perf_counter()
            try:
                (
                    sparse_keypoints,
                    sparse_world_keypoints,
                    pose_confidence,
                    selected_pose_index,
                ) = self._extract_pose_hints_locked(
                    runtime=runtime,
                    image=image,
                    timestamp_ms=timestamp_ms,
                    preferred_box=primary_person_box,
                )
            except Exception:
                logger.exception("Pose inference failed")
                self._sequence.clear()
            stage_latency_ms["pose"] = round((time.perf_counter() - pose_started) * 1000.0, 3)

            hand_landmark_result: HandLandmarkResult | None = None
            hand_started = time.perf_counter()
            try:
                hand_landmark_result = self._analyze_hand_landmarks(
                    runtime=runtime,
                    frame_rgb=frame_rgb,
                    timestamp_ms=timestamp_ms,
                    primary_person_box=primary_person_box,
                    sparse_keypoints=sparse_keypoints,
                )
                final_hand_timestamp_ms = getattr(hand_landmark_result, "final_timestamp_ms", None)
                if final_hand_timestamp_ms is not None:
                    self._reserve_timestamp_window(
                        start_timestamp_ms=int(final_hand_timestamp_ms),
                        slots=1,
                    )
            except Exception:
                logger.exception("Hand-landmark inference failed")
            stage_latency_ms["hand_landmarks"] = round(
                (time.perf_counter() - hand_started) * 1000.0,
                3,
            )

            fine_gesture_started = time.perf_counter()
            fine_gesture = AICameraFineHandGesture.NONE
            fine_confidence: float | None = None
            try:
                fine_gesture, fine_confidence = self._recognize_fine_gesture(
                    runtime=runtime,
                    image=image,
                    timestamp_ms=timestamp_ms,
                    hand_landmark_result=hand_landmark_result,
                )
            except Exception:
                logger.exception("Fine-hand gesture recognition failed")
                fine_gesture, fine_confidence = AICameraFineHandGesture.NONE, None
            stage_latency_ms["fine_gesture"] = round(
                (time.perf_counter() - fine_gesture_started) * 1000.0,
                3,
            )

            hand_detections_present = bool(getattr(hand_landmark_result, "detections", None))
            hand_near_without_pose = hand_detections_present or _is_concrete_fine_gesture(fine_gesture)
            if not sparse_keypoints:
                self._sequence.clear()
                return MediaPipeVisionResult(
                    pose_confidence=_coerce_confidence(pose_confidence),
                    hand_near_camera=hand_near_without_pose,
                    showing_intent_likely=(
                        True if _is_concrete_fine_gesture(fine_gesture) else None
                    ),
                    fine_hand_gesture=fine_gesture,
                    fine_hand_gesture_confidence=_coerce_confidence(fine_confidence),
                    sparse_keypoints=dict(sparse_keypoints),
                    sparse_world_keypoints=dict(sparse_world_keypoints),
                    selected_pose_index=selected_pose_index,
                    stage_latency_ms=dict(stage_latency_ms),
                )

            body_pose_started = time.perf_counter()
            body_pose = AICameraBodyPose.UNKNOWN
            try:
                body_pose = classify_body_pose(sparse_keypoints, fallback_box=primary_person_box)
            except Exception:
                logger.exception("Body-pose classification failed")
            stage_latency_ms["body_pose"] = round(
                (time.perf_counter() - body_pose_started) * 1000.0,
                3,
            )

            attention_started = time.perf_counter()
            visual_attention: float | None = None
            looking_toward_device: bool | None = None
            try:
                visual_attention = _coerce_finite_float(
                    attention_score(sparse_keypoints, fallback_box=primary_person_box)
                )
                if visual_attention is not None:
                    looking_toward_device = (
                        visual_attention >= self.config.attention_score_threshold
                    )
            except Exception:
                logger.exception("Visual-attention scoring failed")
            stage_latency_ms["attention"] = round(
                (time.perf_counter() - attention_started) * 1000.0,
                3,
            )

            hand_near_started = time.perf_counter()
            hand_near = False
            try:
                hand_near = bool(hand_near_camera(sparse_keypoints, fallback_box=primary_person_box))
            except Exception:
                logger.exception("Hand-near-camera scoring failed")
            stage_latency_ms["hand_near_camera"] = round(
                (time.perf_counter() - hand_near_started) * 1000.0,
                3,
            )

            coarse_started = time.perf_counter()
            coarse_gesture = AICameraGestureEvent.NONE
            coarse_confidence: float | None = None
            try:
                coarse_gesture, coarse_confidence = self._sequence.observe(
                    observed_at=sequence_observed_at,
                    sparse_keypoints=sparse_keypoints,
                )
            except Exception:
                logger.exception("Temporal gesture classification failed")
                self._sequence.clear()
            stage_latency_ms["coarse_gesture"] = round(
                (time.perf_counter() - coarse_started) * 1000.0,
                3,
            )

            showing_intent_likely = hand_near and (
                bool(looking_toward_device) or _is_concrete_fine_gesture(fine_gesture)
            )
            return MediaPipeVisionResult(
                body_pose=body_pose,
                pose_confidence=_coerce_confidence(pose_confidence),
                looking_toward_device=looking_toward_device,
                visual_attention_score=visual_attention,
                hand_near_camera=hand_near,
                showing_intent_likely=showing_intent_likely,
                gesture_event=coarse_gesture,
                gesture_confidence=_coerce_confidence(coarse_confidence),
                fine_hand_gesture=fine_gesture,
                fine_hand_gesture_confidence=_coerce_confidence(fine_confidence),
                sparse_keypoints=dict(sparse_keypoints),
                sparse_world_keypoints=dict(sparse_world_keypoints),
                selected_pose_index=selected_pose_index,
                stage_latency_ms=dict(stage_latency_ms),
            )

    def analyze_pose_hints(
        self,
        *,
        frame_rgb: Any,
        observed_at: float,
    ) -> MediaPipeVisionResult:
        """Return only bounded pose hints for ROI seeding without gesture work."""

        stage_latency_ms: dict[str, float] = {}
        with self._lock:
            prepare_started = time.perf_counter()
            try:
                runtime = self._load_runtime()
                prepared_frame_rgb = self._prepare_pose_and_gesture_frame(frame_rgb)
                image = self._build_image(runtime, frame_rgb=prepared_frame_rgb)
                timestamp_ms = self._timestamp_ms(observed_at)
            except Exception:
                logger.exception("Failed to prepare MediaPipe pose-hint inputs")
                return MediaPipeVisionResult(stage_latency_ms=dict(stage_latency_ms))
            stage_latency_ms["prepare"] = round((time.perf_counter() - prepare_started) * 1000.0, 3)

            pose_started = time.perf_counter()
            try:
                (
                    sparse_keypoints,
                    sparse_world_keypoints,
                    pose_confidence,
                    selected_pose_index,
                ) = self._extract_pose_hints_locked(
                    runtime=runtime,
                    image=image,
                    timestamp_ms=timestamp_ms,
                    preferred_box=None,
                )
            except Exception:
                logger.exception("Pose-hint inference failed")
                return MediaPipeVisionResult(stage_latency_ms=dict(stage_latency_ms))
            stage_latency_ms["pose"] = round((time.perf_counter() - pose_started) * 1000.0, 3)

            return MediaPipeVisionResult(
                pose_confidence=_coerce_confidence(pose_confidence),
                sparse_keypoints=dict(sparse_keypoints),
                sparse_world_keypoints=dict(sparse_world_keypoints),
                selected_pose_index=selected_pose_index,
                stage_latency_ms=dict(stage_latency_ms),
            )

    def _recognize_fine_gesture(
        self,
        *,
        runtime: dict[str, Any],
        image: Any,
        timestamp_ms: int,
        hand_landmark_result: HandLandmarkResult | None,
    ) -> tuple[AICameraFineHandGesture, float | None]:
        """Recognize one fine hand gesture with ROI-first plus full-frame fallback."""

        best_builtin = (AICameraFineHandGesture.NONE, None)
        best_custom = (AICameraFineHandGesture.NONE, None)

        if hand_landmark_result is not None and getattr(hand_landmark_result, "detections", None):
            try:
                roi_builtin, roi_custom = self._recognize_fine_gesture_from_hand_rois(
                    runtime=runtime,
                    hand_landmark_result=hand_landmark_result,
                )
                roi_choice = combine_builtin_and_custom_gesture_choice(roi_builtin, roi_custom)
                if _is_concrete_fine_gesture_choice(roi_choice):
                    self._remember_fine_gesture_choice(timestamp_ms=timestamp_ms, choice=roi_choice)
                    return roi_choice
                best_builtin = prefer_gesture_choice(best_builtin, roi_builtin)
                best_custom = prefer_gesture_choice(best_custom, roi_custom)
            except Exception:
                logger.exception("ROI fine gesture fallback failed")

        if not self._should_run_full_frame_gesture(timestamp_ms=timestamp_ms):
            cached_choice = self._get_cached_fine_gesture_choice(timestamp_ms=timestamp_ms)
            if _is_concrete_fine_gesture_choice(cached_choice):
                return cached_choice
            if _is_concrete_fine_gesture_choice(best_builtin) or _is_concrete_fine_gesture_choice(best_custom):
                return combine_builtin_and_custom_gesture_choice(best_builtin, best_custom)
            return cached_choice

        self._reserve_timestamp_window(
            start_timestamp_ms=timestamp_ms,
            slots=1,
        )
        self._last_full_frame_gesture_timestamp_ms = timestamp_ms

        gesture_recognizer: Any | None = None
        try:
            gesture_recognizer = self._ensure_gesture_recognizer(runtime)
        except FileNotFoundError:
            raise
        except Exception:
            logger.exception("Failed to initialize built-in gesture recognizer")

        custom_gesture_recognizer: Any | None = None
        if self.config.custom_gesture_model_path:
            try:
                custom_gesture_recognizer = self._ensure_custom_gesture_recognizer(runtime)
            except FileNotFoundError:
                raise
            except Exception:
                logger.exception("Failed to initialize custom gesture recognizer")

        if gesture_recognizer is not None:
            try:
                builtin_choice = resolve_fine_hand_gesture(
                    result=self._runtime.gesture_recognize_for_video(
                        gesture_recognizer,
                        image=image,
                        timestamp_ms=timestamp_ms,
                    ),
                    category_map=BUILTIN_FINE_GESTURE_MAP,
                    min_score=self.config.builtin_gesture_min_score,
                )
                best_builtin = prefer_gesture_choice(
                    best_builtin,
                    _sanitize_gesture_choice(builtin_choice),
                )
            except Exception:
                logger.exception("Built-in fine gesture inference failed for full-frame candidate")

        if custom_gesture_recognizer is not None:
            try:
                custom_choice = resolve_fine_hand_gesture(
                    result=self._runtime.gesture_recognize_for_video(
                        custom_gesture_recognizer,
                        image=image,
                        timestamp_ms=timestamp_ms,
                    ),
                    category_map=CUSTOM_FINE_GESTURE_MAP,
                    min_score=self.config.custom_gesture_min_score,
                )
                best_custom = prefer_gesture_choice(
                    best_custom,
                    _sanitize_gesture_choice(custom_choice),
                )
            except Exception:
                logger.exception("Custom fine gesture inference failed for full-frame candidate")

        combined_choice = combine_builtin_and_custom_gesture_choice(best_builtin, best_custom)
        self._remember_fine_gesture_choice(timestamp_ms=timestamp_ms, choice=combined_choice)
        return combined_choice

    def _recognize_fine_gesture_from_hand_rois(
        self,
        *,
        runtime: dict[str, Any],
        hand_landmark_result: HandLandmarkResult,
    ) -> tuple[
        tuple[AICameraFineHandGesture, float | None],
        tuple[AICameraFineHandGesture, float | None],
    ]:
        """Run bounded image-mode gesture recognition on ROI crops from hand landmarks."""

        if not getattr(hand_landmark_result, "detections", None):
            return (AICameraFineHandGesture.NONE, None), (AICameraFineHandGesture.NONE, None)

        builtin_choice = (AICameraFineHandGesture.NONE, None)
        custom_choice = (AICameraFineHandGesture.NONE, None)
        builtin_recognizer = self._ensure_roi_gesture_recognizer(runtime)
        custom_recognizer: Any | None = None
        if self.config.custom_gesture_model_path:
            custom_recognizer = self._ensure_custom_roi_gesture_recognizer(runtime)

        detection_limit = self._max_roi_gesture_detections()
        for detection_index, detection in enumerate(hand_landmark_result.detections):
            if detection_index >= detection_limit:
                if not self._logged_roi_detection_truncation:
                    logger.info(
                        "Truncating ROI gesture fan-out to %s hand crops per frame for Pi-side stability",
                        detection_limit,
                    )
                    self._logged_roi_detection_truncation = True
                break

            roi_frame = getattr(detection, "roi_frame_rgb", None)
            if roi_frame is None:
                continue
            prepared_roi_frame = self._prepare_roi_gesture_frame(roi_frame)
            roi_image = self._build_image(runtime, frame_rgb=prepared_roi_frame)

            roi_builtin = resolve_fine_hand_gesture(
                result=self._runtime.gesture_recognize_image(
                    builtin_recognizer,
                    image=roi_image,
                ),
                category_map=BUILTIN_FINE_GESTURE_MAP,
                min_score=self.config.builtin_gesture_min_score,
            )
            builtin_choice = prefer_gesture_choice(
                builtin_choice,
                _sanitize_gesture_choice(roi_builtin),
            )
            if custom_recognizer is None:
                continue
            roi_custom = resolve_fine_hand_gesture(
                result=self._runtime.gesture_recognize_image(
                    custom_recognizer,
                    image=roi_image,
                ),
                category_map=CUSTOM_FINE_GESTURE_MAP,
                min_score=self.config.custom_gesture_min_score,
            )
            custom_choice = prefer_gesture_choice(
                custom_choice,
                _sanitize_gesture_choice(roi_custom),
            )
        return builtin_choice, custom_choice

    def _analyze_hand_landmarks(
        self,
        *,
        runtime: dict[str, Any],
        frame_rgb: Any,
        timestamp_ms: int,
        primary_person_box: AICameraBox,
        sparse_keypoints: dict[int, tuple[float, float, float]],
    ) -> HandLandmarkResult:
        """Run the bounded hand-landmark worker for the current person ROI."""

        worker = self._ensure_hand_landmark_worker()
        return worker.analyze(
            runtime=runtime,
            frame_rgb=frame_rgb,
            timestamp_ms=timestamp_ms,
            primary_person_box=primary_person_box,
            sparse_keypoints=sparse_keypoints,
        )

    def _ensure_hand_landmark_worker(self) -> MediaPipeHandLandmarkWorker:
        """Reuse or create the configured hand-landmark worker."""

        if self._hand_landmark_worker is not None:
            return self._hand_landmark_worker
        self._hand_landmark_worker = MediaPipeHandLandmarkWorker(
            config=HandLandmarkWorkerConfig.from_config(self.config),
        )
        return self._hand_landmark_worker

    def _extract_pose_hints_locked(
        self,
        *,
        runtime: dict[str, Any],
        image: Any,
        timestamp_ms: int,
        preferred_box: AICameraBox | None,
    ) -> tuple[
        dict[int, tuple[float, float, float]],
        dict[int, tuple[float, float, float]],
        float | None,
        int | None,
    ]:
        """Run pose inference and return sparse keypoints for downstream ROI seeding."""

        pose_landmarker = self._ensure_pose_landmarker(runtime)
        pose_result = pose_landmarker.detect_for_video(image, timestamp_ms)
        return _extract_sparse_keypoints_and_world(
            pose_result,
            preferred_box=preferred_box,
        )

    def _load_runtime(self) -> dict[str, Any]:
        """Preserve the historic runtime-loader override point for tests."""

        return self._runtime.load_runtime()

    def _build_image(self, runtime: dict[str, Any], *, frame_rgb: Any) -> Any:
        """Preserve the historic image-builder override point for tests."""

        return self._runtime.build_image(runtime, frame_rgb=frame_rgb)

    def _timestamp_ms(self, observed_at: float) -> int:
        """Preserve the historic timestamp override point for tests."""

        raw_observed_at = _coerce_finite_float(observed_at)
        fallback_timestamp_ms = self._last_timestamp_ms + 1 if self._last_timestamp_ms else 1
        if raw_observed_at is None:
            logger.warning(
                "Received non-finite observed_at=%r; falling back to local timestamp sequencing",
                observed_at,
            )
            raw_timestamp_ms = fallback_timestamp_ms
        else:
            try:
                raw_timestamp_ms = int(self._runtime.timestamp_ms(raw_observed_at))
            except Exception:
                logger.exception(
                    "Failed to convert observed_at=%r to MediaPipe timestamp; falling back to local sequencing",
                    observed_at,
                )
                raw_timestamp_ms = fallback_timestamp_ms

        if raw_timestamp_ms <= self._last_timestamp_ms:
            raw_timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = raw_timestamp_ms
        return raw_timestamp_ms

    def _reserve_timestamp_window(self, *, start_timestamp_ms: int, slots: int) -> None:
        """Reserve one monotonically increasing timestamp window for follow-up recognizer calls."""

        if slots <= 0:
            return
        final_timestamp_ms = start_timestamp_ms + max(0, slots - 1)
        self._last_timestamp_ms = max(self._last_timestamp_ms, final_timestamp_ms)
        try:
            self._runtime.reserve_timestamp(final_timestamp_ms)
        except Exception:
            logger.exception(
                "Failed to reserve MediaPipe timestamp window ending at %s",
                final_timestamp_ms,
            )

    def _ensure_pose_landmarker(self, runtime: dict[str, Any]) -> Any:
        """Preserve the historic pose-landmarker override point for tests."""

        if self._pose_landmarker is None:
            self._pose_landmarker = self._runtime.ensure_pose_landmarker(runtime)
        return self._pose_landmarker

    def _ensure_gesture_recognizer(self, runtime: dict[str, Any]) -> Any:
        """Preserve the historic gesture-recognizer override point for tests."""

        if self._gesture_recognizer is None:
            self._gesture_recognizer = self._runtime.ensure_gesture_recognizer(runtime)
        return self._gesture_recognizer

    def _ensure_custom_gesture_recognizer(self, runtime: dict[str, Any]) -> Any:
        """Preserve the historic custom-gesture override point for tests."""

        if self._custom_gesture_recognizer is None:
            self._custom_gesture_recognizer = self._runtime.ensure_custom_gesture_recognizer(
                runtime,
            )
        return self._custom_gesture_recognizer

    def _ensure_roi_gesture_recognizer(self, runtime: dict[str, Any]) -> Any:
        """Preserve the historic ROI gesture-recognizer override point for tests."""

        if self._roi_gesture_recognizer is None:
            self._roi_gesture_recognizer = self._runtime.ensure_roi_gesture_recognizer(runtime)
        return self._roi_gesture_recognizer

    def _ensure_custom_roi_gesture_recognizer(self, runtime: dict[str, Any]) -> Any:
        """Preserve the historic ROI custom-gesture override point for tests."""

        if self._custom_roi_gesture_recognizer is None:
            self._custom_roi_gesture_recognizer = self._runtime.ensure_custom_roi_gesture_recognizer(
                runtime,
            )
        return self._custom_roi_gesture_recognizer

    def _clear_transient_state_locked(self) -> None:
        """Reset in-memory temporal and short-lived gesture state."""

        self._sequence.clear()
        self._last_fine_gesture_choice = (AICameraFineHandGesture.NONE, None)
        self._last_fine_gesture_timestamp_ms = 0
        self._last_full_frame_gesture_timestamp_ms = 0

    def _prepare_pose_and_gesture_frame(self, frame_rgb: Any) -> Any:
        """Bound whole-frame preprocessing cost on Pi-side inference."""

        prepared_frame = _downscale_frame_stride(
            frame_rgb,
            max_pixels=self._max_pose_frame_pixels(),
        )
        if prepared_frame is not frame_rgb and not self._logged_pose_frame_downscale:
            logger.info(
                "Downscaling oversized pose/gesture frames before MediaPipe task execution",
            )
            self._logged_pose_frame_downscale = True
        return prepared_frame

    def _prepare_roi_gesture_frame(self, frame_rgb: Any) -> Any:
        """Bound ROI gesture crop preprocessing cost on Pi-side inference."""

        prepared_frame = _downscale_frame_stride(
            frame_rgb,
            max_pixels=self._max_roi_frame_pixels(),
        )
        if prepared_frame is not frame_rgb and not self._logged_roi_frame_downscale:
            logger.info(
                "Downscaling oversized ROI gesture crops before MediaPipe task execution",
            )
            self._logged_roi_frame_downscale = True
        return prepared_frame

    def _max_pose_frame_pixels(self) -> int:
        """Return the maximum whole-frame resolution budget for pose/gesture tasks."""

        return _coerce_positive_int(
            getattr(self.config, "max_pose_frame_pixels", 1280 * 720),
            default=1280 * 720,
        )

    def _max_roi_frame_pixels(self) -> int:
        """Return the maximum ROI crop resolution budget for ROI gesture tasks."""

        return _coerce_positive_int(
            getattr(self.config, "max_roi_frame_pixels", 256 * 256),
            default=256 * 256,
        )

    def _max_roi_gesture_detections(self) -> int:
        """Return the per-frame ROI gesture fan-out limit."""

        config_default = getattr(self.config, "max_roi_gesture_detections", None)
        if config_default is None:
            config_default = getattr(self.config, "num_hands", 2)
        return _coerce_positive_int(
            config_default,
            default=2,
            minimum=1,
            maximum=4,
        )

    def _full_frame_gesture_min_interval_ms(self) -> int:
        """Return the minimum spacing between expensive full-frame gesture passes."""

        return _coerce_nonnegative_int(
            getattr(self.config, "full_frame_gesture_min_interval_ms", 100),
            default=100,
        )

    def _fine_gesture_cache_max_age_ms(self) -> int:
        """Return the maximum age for reusing the previous fine gesture."""

        return _coerce_nonnegative_int(
            getattr(self.config, "fine_gesture_cache_max_age_ms", 160),
            default=160,
        )

    def _should_run_full_frame_gesture(self, *, timestamp_ms: int) -> bool:
        """Return whether the expensive full-frame fallback should run now."""

        min_interval_ms = self._full_frame_gesture_min_interval_ms()
        if min_interval_ms <= 0:
            return True
        if self._last_full_frame_gesture_timestamp_ms <= 0:
            return True
        return (timestamp_ms - self._last_full_frame_gesture_timestamp_ms) >= min_interval_ms

    def _remember_fine_gesture_choice(
        self,
        *,
        timestamp_ms: int,
        choice: tuple[AICameraFineHandGesture, float | None],
    ) -> None:
        """Store the latest fine-gesture decision for short-lived reuse."""

        self._last_fine_gesture_choice = _sanitize_gesture_choice(choice)
        self._last_fine_gesture_timestamp_ms = timestamp_ms

    def _get_cached_fine_gesture_choice(
        self,
        *,
        timestamp_ms: int,
    ) -> tuple[AICameraFineHandGesture, float | None]:
        """Return the latest reusable fine gesture, if still fresh."""

        max_age_ms = self._fine_gesture_cache_max_age_ms()
        if max_age_ms <= 0:
            return (AICameraFineHandGesture.NONE, None)
        if self._last_fine_gesture_timestamp_ms <= 0:
            return (AICameraFineHandGesture.NONE, None)
        if (timestamp_ms - self._last_fine_gesture_timestamp_ms) > max_age_ms:
            return (AICameraFineHandGesture.NONE, None)

        cached_choice = self._last_fine_gesture_choice
        if cached_choice[0] == AICameraFineHandGesture.UNKNOWN:
            return (AICameraFineHandGesture.NONE, None)
        return cached_choice


def extract_sparse_keypoints(
    result: Any,
    preferred_box: AICameraBox | None = None,
) -> tuple[dict[int, tuple[float, float, float]], float | None]:
    """Extract a COCO-like sparse keypoint map from one pose-landmarker result."""

    sparse_keypoints, _sparse_world_keypoints, pose_confidence, _pose_index = (
        _extract_sparse_keypoints_and_world(result, preferred_box=preferred_box)
    )
    return sparse_keypoints, pose_confidence


__all__ = [
    "MediaPipeVisionPipeline",
    "MediaPipeVisionResult",
    "extract_sparse_keypoints",
]
