"""Run MediaPipe pose and hand-gesture inference on Pi-side RGB camera frames.

This module owns the CPU-side MediaPipe stack that complements the IMX500
always-on gate. It keeps MediaPipe imports lazy, exposes a small typed result
surface to the IMX500 adapter, and classifies coarse arm gestures from short
pose sequences instead of single-frame arm heuristics.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import importlib
import logging
import math

from twinr.hardware.ai_camera import (
    AICameraBodyPose,
    AICameraBox,
    AICameraFineHandGesture,
    AICameraGestureEvent,
    _attention_score,
    _classify_body_pose,
    _hand_near_camera,
)


logger = logging.getLogger(__name__)

DEFAULT_MEDIAPIPE_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
)
DEFAULT_MEDIAPIPE_GESTURE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
)

_COCO_KEYPOINT_INDEX = {
    0: 0,   # nose
    2: 1,   # left eye
    5: 2,   # right eye
    11: 5,  # left shoulder
    12: 6,  # right shoulder
    13: 7,  # left elbow
    14: 8,  # right elbow
    15: 9,  # left wrist
    16: 10,  # right wrist
    23: 11,  # left hip
    24: 12,  # right hip
    25: 13,  # left knee
    26: 14,  # right knee
    27: 15,  # left ankle
    28: 16,  # right ankle
}

_BUILTIN_FINE_GESTURE_MAP = {
    "thumb_up": AICameraFineHandGesture.THUMBS_UP,
    "thumbs_up": AICameraFineHandGesture.THUMBS_UP,
    "thumb_down": AICameraFineHandGesture.THUMBS_DOWN,
    "thumbs_down": AICameraFineHandGesture.THUMBS_DOWN,
    "pointing_up": AICameraFineHandGesture.POINTING,
    "pointing": AICameraFineHandGesture.POINTING,
    "open_palm": AICameraFineHandGesture.OPEN_PALM,
}

_CUSTOM_FINE_GESTURE_MAP = {
    "ok": AICameraFineHandGesture.OK_SIGN,
    "ok_sign": AICameraFineHandGesture.OK_SIGN,
    "okay": AICameraFineHandGesture.OK_SIGN,
    "middle_finger": AICameraFineHandGesture.MIDDLE_FINGER,
    "flip_off": AICameraFineHandGesture.MIDDLE_FINGER,
}


@dataclass(frozen=True, slots=True)
class MediaPipeVisionConfig:
    """Store bounded MediaPipe runtime settings for Pi-side inference."""

    pose_model_path: str
    gesture_model_path: str
    custom_gesture_model_path: str | None = None
    num_hands: int = 2
    attention_score_threshold: float = 0.62
    sequence_window_s: float = 1.6
    sequence_min_frames: int = 4
    builtin_gesture_min_score: float = 0.50
    custom_gesture_min_score: float = 0.55
    min_pose_detection_confidence: float = 0.50
    min_pose_presence_confidence: float = 0.50
    min_pose_tracking_confidence: float = 0.50
    min_hand_detection_confidence: float = 0.50
    min_hand_presence_confidence: float = 0.50
    min_hand_tracking_confidence: float = 0.50

    @classmethod
    def from_ai_camera_config(cls, config: object) -> "MediaPipeVisionConfig":
        """Build one MediaPipe config from ``AICameraAdapterConfig``-like input."""

        return cls(
            pose_model_path=str(getattr(config, "mediapipe_pose_model_path", "") or "").strip(),
            gesture_model_path=str(getattr(config, "mediapipe_gesture_model_path", "") or "").strip(),
            custom_gesture_model_path=(
                str(getattr(config, "mediapipe_custom_gesture_model_path", "") or "").strip() or None
            ),
            num_hands=max(1, int(getattr(config, "mediapipe_num_hands", 2) or 2)),
            attention_score_threshold=_clamp_ratio(
                getattr(config, "attention_score_threshold", 0.62),
                default=0.62,
            ),
            sequence_window_s=_coerce_positive_float(
                getattr(config, "sequence_window_s", 1.6),
                default=1.6,
            ),
            sequence_min_frames=max(2, int(getattr(config, "sequence_min_frames", 4) or 4)),
        )


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


@dataclass(frozen=True, slots=True)
class _TemporalPoseFrame:
    """Store one normalized pose sample for short-window gesture classification."""

    observed_at: float
    left_shoulder: tuple[float, float, float] | None
    right_shoulder: tuple[float, float, float] | None
    left_elbow: tuple[float, float, float] | None
    right_elbow: tuple[float, float, float] | None
    left_wrist: tuple[float, float, float] | None
    right_wrist: tuple[float, float, float] | None
    left_hip: tuple[float, float, float] | None
    right_hip: tuple[float, float, float] | None


class TemporalPoseGestureClassifier:
    """Classify coarse arm gestures from a short landmark sequence window."""

    def __init__(self, *, window_s: float, min_frames: int) -> None:
        """Initialize one bounded sequence classifier."""

        self.window_s = _coerce_positive_float(window_s, default=1.6)
        self.min_frames = max(2, int(min_frames))
        self._frames: deque[_TemporalPoseFrame] = deque()

    def clear(self) -> None:
        """Discard any buffered sequence state."""

        self._frames.clear()

    def observe(
        self,
        *,
        observed_at: float,
        sparse_keypoints: dict[int, tuple[float, float, float]],
    ) -> tuple[AICameraGestureEvent, float | None]:
        """Append one pose sample and classify the current coarse gesture."""

        self._frames.append(
            _TemporalPoseFrame(
                observed_at=observed_at,
                left_shoulder=sparse_keypoints.get(5),
                right_shoulder=sparse_keypoints.get(6),
                left_elbow=sparse_keypoints.get(7),
                right_elbow=sparse_keypoints.get(8),
                left_wrist=sparse_keypoints.get(9),
                right_wrist=sparse_keypoints.get(10),
                left_hip=sparse_keypoints.get(11),
                right_hip=sparse_keypoints.get(12),
            )
        )
        self._prune(now=observed_at)
        return _classify_temporal_gesture(tuple(self._frames), min_frames=self.min_frames)

    def _prune(self, *, now: float) -> None:
        """Drop samples that are older than the configured temporal window."""

        cutoff = now - self.window_s
        while self._frames and self._frames[0].observed_at < cutoff:
            self._frames.popleft()


class MediaPipeVisionPipeline:
    """Run MediaPipe pose and gesture tasks on RGB frames gated by IMX500."""

    def __init__(self, *, config: MediaPipeVisionConfig) -> None:
        """Initialize one lazy MediaPipe pipeline."""

        self.config = config
        self._pose_landmarker: Any | None = None
        self._gesture_recognizer: Any | None = None
        self._custom_gesture_recognizer: Any | None = None
        self._last_timestamp_ms = 0
        self._sequence = TemporalPoseGestureClassifier(
            window_s=config.sequence_window_s,
            min_frames=config.sequence_min_frames,
        )

    def close(self) -> None:
        """Close active MediaPipe task instances when supported."""

        for instance in (self._pose_landmarker, self._gesture_recognizer, self._custom_gesture_recognizer):
            if instance is None:
                continue
            close_fn = getattr(instance, "close", None)
            if callable(close_fn):
                close_fn()
        self._pose_landmarker = None
        self._gesture_recognizer = None
        self._custom_gesture_recognizer = None
        self._sequence.clear()

    def reset_temporal_state(self) -> None:
        """Discard only the buffered sequence state between distinct presence sessions."""

        self._sequence.clear()

    def analyze(
        self,
        *,
        frame_rgb: Any,
        observed_at: float,
        primary_person_box: AICameraBox,
    ) -> MediaPipeVisionResult:
        """Run the pose and gesture tasks against one RGB frame."""

        runtime = self._load_runtime()
        image = self._build_image(runtime, frame_rgb=frame_rgb)
        timestamp_ms = self._timestamp_ms(observed_at)

        pose_landmarker = self._ensure_pose_landmarker(runtime)
        pose_result = pose_landmarker.detect_for_video(image, timestamp_ms)
        sparse_keypoints, pose_confidence = _extract_sparse_keypoints(pose_result)
        if not sparse_keypoints:
            return MediaPipeVisionResult()

        body_pose = _classify_body_pose(sparse_keypoints, fallback_box=primary_person_box)
        visual_attention_score = _attention_score(sparse_keypoints, fallback_box=primary_person_box)
        looking_toward_device = visual_attention_score >= self.config.attention_score_threshold
        hand_near_camera = _hand_near_camera(sparse_keypoints, fallback_box=primary_person_box)
        coarse_gesture, coarse_confidence = self._sequence.observe(
            observed_at=observed_at,
            sparse_keypoints=sparse_keypoints,
        )
        fine_gesture, fine_confidence = self._recognize_fine_gesture(
            runtime=runtime,
            image=image,
            timestamp_ms=timestamp_ms,
        )
        showing_intent_likely = hand_near_camera and (looking_toward_device or fine_gesture != AICameraFineHandGesture.NONE)
        return MediaPipeVisionResult(
            body_pose=body_pose,
            pose_confidence=pose_confidence,
            looking_toward_device=looking_toward_device,
            visual_attention_score=visual_attention_score,
            hand_near_camera=hand_near_camera,
            showing_intent_likely=showing_intent_likely,
            gesture_event=coarse_gesture,
            gesture_confidence=coarse_confidence,
            fine_hand_gesture=fine_gesture,
            fine_hand_gesture_confidence=fine_confidence,
        )

    def _recognize_fine_gesture(
        self,
        *,
        runtime: dict[str, Any],
        image: Any,
        timestamp_ms: int,
    ) -> tuple[AICameraFineHandGesture, float | None]:
        """Recognize one fine hand gesture from built-in and optional custom models.

        A configured custom gesture model path is treated as authoritative. If it
        is configured but the asset is missing, the caller receives an explicit
        file error instead of a silent downgrade to the built-in recognizer.
        """

        builtin_gesture, builtin_score = _resolve_fine_hand_gesture(
            result=self._ensure_gesture_recognizer(runtime).recognize_for_video(image, timestamp_ms),
            category_map=_BUILTIN_FINE_GESTURE_MAP,
            min_score=self.config.builtin_gesture_min_score,
        )
        custom_gesture = AICameraFineHandGesture.NONE
        custom_score = None
        if self.config.custom_gesture_model_path:
            custom_gesture, custom_score = _resolve_fine_hand_gesture(
                result=self._ensure_custom_gesture_recognizer(runtime).recognize_for_video(image, timestamp_ms),
                category_map=_CUSTOM_FINE_GESTURE_MAP,
                min_score=self.config.custom_gesture_min_score,
            )
        if custom_gesture != AICameraFineHandGesture.NONE:
            return custom_gesture, custom_score
        return builtin_gesture, builtin_score

    def _load_runtime(self) -> dict[str, Any]:
        """Import the minimum MediaPipe runtime objects lazily."""

        try:
            mp = importlib.import_module("mediapipe")
            tasks_python = importlib.import_module("mediapipe.tasks.python")
            vision = importlib.import_module("mediapipe.tasks.python.vision")
        except Exception as exc:  # pragma: no cover - depends on local environment.
            raise RuntimeError("mediapipe_unavailable") from exc
        return {
            "mp": mp,
            "BaseOptions": getattr(tasks_python, "BaseOptions"),
            "vision": vision,
        }

    def _build_image(self, runtime: dict[str, Any], *, frame_rgb: Any) -> Any:
        """Wrap one RGB array-like frame in a MediaPipe image container."""

        mp = runtime["mp"]
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    def _timestamp_ms(self, observed_at: float) -> int:
        """Return a monotonically increasing task timestamp in milliseconds."""

        timestamp_ms = max(1, int(round(float(observed_at) * 1000.0)))
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms
        return timestamp_ms

    def _ensure_pose_landmarker(self, runtime: dict[str, Any]) -> Any:
        """Reuse or create the configured pose landmarker."""

        if self._pose_landmarker is not None:
            return self._pose_landmarker
        model_path = Path(self.config.pose_model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"mediapipe_pose_model_missing:{model_path}")
        vision = runtime["vision"]
        options = vision.PoseLandmarkerOptions(
            base_options=runtime["BaseOptions"](model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=self.config.min_pose_detection_confidence,
            min_pose_presence_confidence=self.config.min_pose_presence_confidence,
            min_tracking_confidence=self.config.min_pose_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._pose_landmarker = vision.PoseLandmarker.create_from_options(options)
        return self._pose_landmarker

    def _ensure_gesture_recognizer(self, runtime: dict[str, Any]) -> Any:
        """Reuse or create the configured built-in gesture recognizer."""

        if self._gesture_recognizer is not None:
            return self._gesture_recognizer
        model_path = Path(self.config.gesture_model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"mediapipe_gesture_model_missing:{model_path}")
        vision = runtime["vision"]
        options = vision.GestureRecognizerOptions(
            base_options=runtime["BaseOptions"](model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.config.num_hands,
            min_hand_detection_confidence=self.config.min_hand_detection_confidence,
            min_hand_presence_confidence=self.config.min_hand_presence_confidence,
            min_tracking_confidence=self.config.min_hand_tracking_confidence,
        )
        self._gesture_recognizer = vision.GestureRecognizer.create_from_options(options)
        return self._gesture_recognizer

    def _ensure_custom_gesture_recognizer(self, runtime: dict[str, Any]) -> Any:
        """Reuse or create the configured custom gesture recognizer."""

        if self._custom_gesture_recognizer is not None:
            return self._custom_gesture_recognizer
        model_path = Path(self.config.custom_gesture_model_path or "")
        if not model_path.exists():
            raise FileNotFoundError(f"mediapipe_custom_gesture_model_missing:{model_path}")
        vision = runtime["vision"]
        options = vision.GestureRecognizerOptions(
            base_options=runtime["BaseOptions"](model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.config.num_hands,
            min_hand_detection_confidence=self.config.min_hand_detection_confidence,
            min_hand_presence_confidence=self.config.min_hand_presence_confidence,
            min_tracking_confidence=self.config.min_hand_tracking_confidence,
        )
        self._custom_gesture_recognizer = vision.GestureRecognizer.create_from_options(options)
        return self._custom_gesture_recognizer


def _extract_sparse_keypoints(result: Any) -> tuple[dict[int, tuple[float, float, float]], float | None]:
    """Extract a COCO-like sparse keypoint map from one pose-landmarker result."""

    pose_landmarks = getattr(result, "pose_landmarks", None) or ()
    if not pose_landmarks:
        return {}, None
    first_pose = pose_landmarks[0]
    sparse: dict[int, tuple[float, float, float]] = {}
    confidence_values: list[float] = []
    for mediapipe_index, coco_index in _COCO_KEYPOINT_INDEX.items():
        if mediapipe_index >= len(first_pose):
            continue
        landmark = first_pose[mediapipe_index]
        score = _landmark_score(landmark)
        sparse[coco_index] = (
            _clamp_ratio(getattr(landmark, "x", 0.0), default=0.0),
            _clamp_ratio(getattr(landmark, "y", 0.0), default=0.0),
            score,
        )
        confidence_values.append(score)
    if not confidence_values:
        return {}, None
    return sparse, round(sum(confidence_values) / float(len(confidence_values)), 3)


def _resolve_fine_hand_gesture(
    *,
    result: Any,
    category_map: dict[str, AICameraFineHandGesture],
    min_score: float,
) -> tuple[AICameraFineHandGesture, float | None]:
    """Map one gesture-recognizer result into Twinr's bounded fine-gesture enum."""

    gestures = getattr(result, "gestures", None) or ()
    best_gesture = AICameraFineHandGesture.NONE
    best_score = 0.0
    for gesture_set in gestures:
        for category in gesture_set or ():
            label = _normalize_category_name(getattr(category, "category_name", None))
            if not label:
                continue
            mapped = category_map.get(label)
            if mapped is None:
                continue
            score = _clamp_ratio(getattr(category, "score", 0.0), default=0.0)
            if score < min_score or score <= best_score:
                continue
            best_gesture = mapped
            best_score = score
    if best_gesture == AICameraFineHandGesture.NONE:
        return best_gesture, None
    return best_gesture, round(best_score, 3)


def _classify_temporal_gesture(
    frames: tuple[_TemporalPoseFrame, ...],
    *,
    min_frames: int,
) -> tuple[AICameraGestureEvent, float | None]:
    """Classify one coarse arm gesture from a short sequence of pose frames."""

    if len(frames) < max(2, min_frames):
        return AICameraGestureEvent.NONE, None

    arms_crossed_matches = sum(1 for frame in frames if _frame_is_arms_crossed(frame))
    timeout_matches = sum(1 for frame in frames if _frame_is_timeout_t(frame))
    two_hand_dismiss_matches = sum(1 for frame in frames if _frame_is_two_hand_dismiss(frame))
    if arms_crossed_matches >= min_frames:
        return AICameraGestureEvent.ARMS_CROSSED, round(_clamp_ratio(0.58 + 0.08 * (arms_crossed_matches / len(frames)), default=0.58), 3)
    if timeout_matches >= min_frames:
        return AICameraGestureEvent.TIMEOUT_T, round(_clamp_ratio(0.58 + 0.08 * (timeout_matches / len(frames)), default=0.58), 3)
    if two_hand_dismiss_matches >= min_frames:
        return AICameraGestureEvent.TWO_HAND_DISMISS, round(
            _clamp_ratio(0.60 + 0.08 * (two_hand_dismiss_matches / len(frames)), default=0.60),
            3,
        )
    if _frames_form_wave(frames, min_frames=min_frames):
        return AICameraGestureEvent.WAVE, 0.66
    return AICameraGestureEvent.NONE, None


def _frame_is_arms_crossed(frame: _TemporalPoseFrame) -> bool:
    """Return whether one pose frame looks like crossed arms at chest height."""

    if not all((frame.left_shoulder, frame.right_shoulder, frame.left_wrist, frame.right_wrist)):
        return False
    left_shoulder = frame.left_shoulder
    right_shoulder = frame.right_shoulder
    left_wrist = frame.left_wrist
    right_wrist = frame.right_wrist
    assert left_shoulder is not None and right_shoulder is not None
    assert left_wrist is not None and right_wrist is not None
    shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2.0
    shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
    chest_bottom = shoulder_y + 0.24
    return (
        shoulder_y - 0.06 <= left_wrist[1] <= chest_bottom
        and shoulder_y - 0.06 <= right_wrist[1] <= chest_bottom
        and left_wrist[0] >= shoulder_center_x + 0.02
        and right_wrist[0] <= shoulder_center_x - 0.02
    )


def _frame_is_timeout_t(frame: _TemporalPoseFrame) -> bool:
    """Return whether one pose frame looks like a timeout-T arm configuration."""

    variants = (
        (
            frame.left_shoulder,
            frame.left_elbow,
            frame.left_wrist,
            frame.right_shoulder,
            frame.right_elbow,
            frame.right_wrist,
        ),
        (
            frame.right_shoulder,
            frame.right_elbow,
            frame.right_wrist,
            frame.left_shoulder,
            frame.left_elbow,
            frame.left_wrist,
        ),
    )
    for horizontal_shoulder, horizontal_elbow, horizontal_wrist, vertical_shoulder, vertical_elbow, vertical_wrist in variants:
        if not all((horizontal_shoulder, horizontal_elbow, horizontal_wrist, vertical_shoulder, vertical_elbow, vertical_wrist)):
            continue
        assert horizontal_shoulder is not None and horizontal_elbow is not None and horizontal_wrist is not None
        assert vertical_shoulder is not None and vertical_elbow is not None and vertical_wrist is not None
        horizontal_ok = (
            abs(horizontal_wrist[1] - horizontal_elbow[1]) <= 0.08
            and abs(horizontal_elbow[1] - horizontal_shoulder[1]) <= 0.16
        )
        vertical_ok = (
            abs(vertical_wrist[0] - vertical_elbow[0]) <= 0.06
            and abs(vertical_elbow[0] - vertical_shoulder[0]) <= 0.06
            and vertical_wrist[1] < vertical_elbow[1] < (vertical_shoulder[1] + 0.20)
        )
        wrists_close = (
            abs(horizontal_wrist[0] - vertical_wrist[0]) <= 0.12
            and abs(horizontal_wrist[1] - vertical_elbow[1]) <= 0.12
        )
        if horizontal_ok and vertical_ok and wrists_close:
            return True
    return False


def _frame_is_two_hand_dismiss(frame: _TemporalPoseFrame) -> bool:
    """Return whether one frame looks like a symmetric two-hand dismiss pose."""

    if not all((frame.left_shoulder, frame.right_shoulder, frame.left_wrist, frame.right_wrist)):
        return False
    left_shoulder = frame.left_shoulder
    right_shoulder = frame.right_shoulder
    left_wrist = frame.left_wrist
    right_wrist = frame.right_wrist
    assert left_shoulder is not None and right_shoulder is not None
    assert left_wrist is not None and right_wrist is not None
    return (
        left_wrist[0] <= left_shoulder[0] - 0.12
        and right_wrist[0] >= right_shoulder[0] + 0.12
        and abs(left_wrist[1] - left_shoulder[1]) <= 0.18
        and abs(right_wrist[1] - right_shoulder[1]) <= 0.18
    )


def _frames_form_wave(frames: tuple[_TemporalPoseFrame, ...], *, min_frames: int) -> bool:
    """Return whether one sequence contains a raised-hand lateral wave pattern."""

    for wrist_name, shoulder_name in (("left_wrist", "left_shoulder"), ("right_wrist", "right_shoulder")):
        series: list[float] = []
        for frame in frames:
            wrist = getattr(frame, wrist_name)
            shoulder = getattr(frame, shoulder_name)
            if wrist is None or shoulder is None:
                continue
            if wrist[1] >= shoulder[1] - 0.05:
                continue
            series.append(wrist[0] - shoulder[0])
        if len(series) < min_frames:
            continue
        amplitude = max(series) - min(series)
        if amplitude < 0.10:
            continue
        direction_changes = 0
        last_sign = 0
        for index in range(1, len(series)):
            delta = series[index] - series[index - 1]
            sign = 1 if delta > 0.02 else -1 if delta < -0.02 else 0
            if sign and last_sign and sign != last_sign:
                direction_changes += 1
            if sign:
                last_sign = sign
        if direction_changes >= 1:
            return True
    return False


def _landmark_score(landmark: Any) -> float:
    """Estimate one landmark confidence from visibility/presence when available."""

    visibility = getattr(landmark, "visibility", None)
    presence = getattr(landmark, "presence", None)
    scores = [
        _clamp_ratio(value, default=0.0)
        for value in (visibility, presence)
        if value is not None and math.isfinite(float(value))
    ]
    if not scores:
        return 1.0
    return round(sum(scores) / float(len(scores)), 3)


def _normalize_category_name(value: object) -> str:
    """Normalize one classifier category label to a stable token."""

    return "_".join(str(value or "").strip().lower().split())


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


def _coerce_positive_float(value: object, *, default: float) -> float:
    """Coerce one value to a positive finite float."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number) or number <= 0.0:
        return default
    return number


__all__ = [
    "DEFAULT_MEDIAPIPE_GESTURE_MODEL_URL",
    "DEFAULT_MEDIAPIPE_POSE_MODEL_URL",
    "MediaPipeVisionConfig",
    "MediaPipeVisionPipeline",
    "MediaPipeVisionResult",
    "TemporalPoseGestureClassifier",
    "_classify_temporal_gesture",
    "_extract_sparse_keypoints",
    "_resolve_fine_hand_gesture",
]
