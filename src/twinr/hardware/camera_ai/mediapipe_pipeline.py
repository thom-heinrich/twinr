"""Run MediaPipe pose and hand-gesture inference on Pi-side RGB frames."""

from __future__ import annotations

import logging
import math
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

    # AUDIT-FIX(#6): Reject NaN/inf/missing numeric fields instead of fabricating coordinates.
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric_value):
        return None
    return numeric_value


def _coerce_confidence(value: Any) -> float | None:
    """Return one normalized confidence score in ``[0.0, 1.0]`` when valid."""

    # AUDIT-FIX(#6): Clamp malformed confidence scores before they reach classifiers/results.
    numeric_value = _coerce_finite_float(value)
    if numeric_value is None:
        return None
    return max(0.0, min(1.0, numeric_value))


def _sanitize_gesture_choice(
    choice: tuple[Any, Any],
) -> tuple[AICameraFineHandGesture, float | None]:
    """Normalize one gesture-choice tuple from gesture-resolution helpers."""

    gesture, confidence = choice
    if not isinstance(gesture, AICameraFineHandGesture):
        gesture = AICameraFineHandGesture.NONE  # AUDIT-FIX(#7): Guard against malformed resolver output.
    return gesture, _coerce_confidence(confidence)


def _is_concrete_fine_gesture_choice(
    choice: tuple[AICameraFineHandGesture, float | None],
) -> bool:
    """Return whether one gesture choice carries a real user-facing symbol."""

    return choice[0] not in {AICameraFineHandGesture.NONE, AICameraFineHandGesture.UNKNOWN}


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
        self._sequence = TemporalPoseGestureClassifier(
            window_s=config.sequence_window_s,
            min_frames=config.sequence_min_frames,
        )
        self._lock = RLock()  # AUDIT-FIX(#5): Serialize mutable state and native task access across analyze/close/reset.

    def close(self) -> None:
        """Close active MediaPipe task instances when supported."""

        with self._lock:  # AUDIT-FIX(#5): Prevent shutdown racing active inference.
            try:
                if self._hand_landmark_worker is not None:
                    try:
                        self._hand_landmark_worker.close()
                    except Exception:
                        logger.exception("Failed to close hand-landmark worker cleanly")
                try:
                    self._runtime.close()
                except Exception:
                    logger.exception("Failed to close MediaPipe runtime cleanly")  # AUDIT-FIX(#8): Best-effort cleanup.
            finally:
                self._last_timestamp_ms = 0
                self._pose_landmarker = None
                self._hand_landmark_worker = None
                self._gesture_recognizer = None
                self._custom_gesture_recognizer = None
                self._sequence.clear()

    def reset_temporal_state(self) -> None:
        """Discard only the buffered sequence state between presence sessions."""

        with self._lock:  # AUDIT-FIX(#5): Keep temporal-state resets atomic with analyze().
            self._sequence.clear()

    def analyze(
        self,
        *,
        frame_rgb: Any,
        observed_at: float,
        primary_person_box: AICameraBox,
    ) -> MediaPipeVisionResult:
        """Run the pose and gesture tasks against one RGB frame."""

        with self._lock:  # AUDIT-FIX(#5): Protect shared timestamps, sequence buffers, and MediaPipe handles.
            try:
                runtime = self._load_runtime()
                image = self._build_image(runtime, frame_rgb=frame_rgb)
                timestamp_ms = self._timestamp_ms(observed_at)
                sequence_observed_at = _coerce_finite_float(observed_at)
                if sequence_observed_at is None:
                    sequence_observed_at = timestamp_ms / 1000.0  # AUDIT-FIX(#1): Keep temporal classifier alive when upstream timestamps are malformed.
            except Exception:
                logger.exception("Failed to prepare MediaPipe inference inputs")
                self._sequence.clear()  # AUDIT-FIX(#1): Fail closed to UNKNOWN instead of crashing the pipeline.
                return MediaPipeVisionResult()

            sparse_keypoints: dict[int, tuple[float, float, float]] = {}
            pose_confidence: float | None = None
            try:
                sparse_keypoints, pose_confidence = self._extract_pose_hints_locked(
                    runtime=runtime,
                    image=image,
                    timestamp_ms=timestamp_ms,
                )
            except Exception:
                logger.exception("Pose inference failed")
                self._sequence.clear()  # AUDIT-FIX(#3): Drop stale temporal history when pose disappears or fails.

            hand_landmark_result: HandLandmarkResult | None = None
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
                logger.exception("Hand-landmark inference failed")  # AUDIT-FIX(#1): Preserve partial pose output on hand stage failure.

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
                logger.exception("Fine-hand gesture recognition failed")  # AUDIT-FIX(#1): Degrade to NONE instead of aborting.
                fine_gesture, fine_confidence = AICameraFineHandGesture.NONE, None

            if not sparse_keypoints:
                self._sequence.clear()  # AUDIT-FIX(#3): Prevent stale gesture sequences from surviving missing-pose frames.
                return MediaPipeVisionResult(
                    pose_confidence=pose_confidence,
                    showing_intent_likely=(True if fine_gesture != AICameraFineHandGesture.NONE else None),
                    fine_hand_gesture=fine_gesture,
                    fine_hand_gesture_confidence=_coerce_confidence(fine_confidence),
                    sparse_keypoints=dict(sparse_keypoints),
                )

            body_pose = AICameraBodyPose.UNKNOWN
            try:
                body_pose = classify_body_pose(sparse_keypoints, fallback_box=primary_person_box)
            except Exception:
                logger.exception("Body-pose classification failed")  # AUDIT-FIX(#1): Preserve remaining signals on classifier errors.

            visual_attention: float | None = None
            looking_toward_device: bool | None = None
            try:
                visual_attention = _coerce_finite_float(
                    attention_score(sparse_keypoints, fallback_box=primary_person_box)
                )
                if visual_attention is not None:
                    looking_toward_device = visual_attention >= self.config.attention_score_threshold
            except Exception:
                logger.exception("Visual-attention scoring failed")  # AUDIT-FIX(#1): Return partial result instead of raising.

            hand_near = False
            try:
                hand_near = bool(hand_near_camera(sparse_keypoints, fallback_box=primary_person_box))
            except Exception:
                logger.exception("Hand-near-camera scoring failed")  # AUDIT-FIX(#1): Preserve pose/fine-gesture signals.

            coarse_gesture = AICameraGestureEvent.NONE
            coarse_confidence: float | None = None
            try:
                coarse_gesture, coarse_confidence = self._sequence.observe(
                    observed_at=sequence_observed_at,
                    sparse_keypoints=sparse_keypoints,
                )
            except Exception:
                logger.exception("Temporal gesture classification failed")
                self._sequence.clear()  # AUDIT-FIX(#3): Reset sequence after temporal classifier failure.

            showing_intent_likely = hand_near and (
                bool(looking_toward_device) or fine_gesture != AICameraFineHandGesture.NONE
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
            )

    def analyze_pose_hints(
        self,
        *,
        frame_rgb: Any,
        observed_at: float,
    ) -> MediaPipeVisionResult:
        """Return only bounded pose hints for ROI seeding without gesture work."""

        with self._lock:
            try:
                runtime = self._load_runtime()
                image = self._build_image(runtime, frame_rgb=frame_rgb)
                timestamp_ms = self._timestamp_ms(observed_at)
            except Exception:
                logger.exception("Failed to prepare MediaPipe pose-hint inputs")
                return MediaPipeVisionResult()

            try:
                sparse_keypoints, pose_confidence = self._extract_pose_hints_locked(
                    runtime=runtime,
                    image=image,
                    timestamp_ms=timestamp_ms,
                )
            except Exception:
                logger.exception("Pose-hint inference failed")
                return MediaPipeVisionResult()
            return MediaPipeVisionResult(
                pose_confidence=_coerce_confidence(pose_confidence),
                sparse_keypoints=dict(sparse_keypoints),
            )

    def _recognize_fine_gesture(
        self,
        *,
        runtime: dict[str, Any],
        image: Any,
        timestamp_ms: int,
        hand_landmark_result: HandLandmarkResult | None,
    ) -> tuple[AICameraFineHandGesture, float | None]:
        """Recognize one fine hand gesture with ROI-first plus full-frame fallback.

        On the Pi, hand landmarks already localize bounded wrist crops for the
        common interaction case where a user presents a hand near the device.
        Running full-frame gesture recognition before those ROI crops forces the
        hot path to pay for both detectors on every frame, which visibly raises
        live gesture latency. Prefer ROI recognizers first when concrete hand
        detections exist, then fall back to the stable full-frame tracker only
        when the ROI path did not yield a real symbol.
        """

        best_builtin = (AICameraFineHandGesture.NONE, None)
        best_custom = (AICameraFineHandGesture.NONE, None)
        self._reserve_timestamp_window(
            start_timestamp_ms=timestamp_ms,
            slots=1,
        )  # AUDIT-FIX(#2): Reserve exactly one timestamp slot for the stable full-frame gesture pass.

        if hand_landmark_result is not None and hand_landmark_result.detections:
            try:
                roi_builtin, roi_custom = self._recognize_fine_gesture_from_hand_rois(
                    runtime=runtime,
                    hand_landmark_result=hand_landmark_result,
                )
                roi_choice = combine_builtin_and_custom_gesture_choice(roi_builtin, roi_custom)
                if _is_concrete_fine_gesture_choice(roi_choice):
                    return roi_choice
                best_builtin = prefer_gesture_choice(best_builtin, roi_builtin)
                best_custom = prefer_gesture_choice(best_custom, roi_custom)
            except Exception:
                logger.exception("ROI fine gesture fallback failed")

        gesture_recognizer: Any | None = None
        try:
            gesture_recognizer = self._ensure_gesture_recognizer(runtime)
        except FileNotFoundError:
            raise
        except Exception:
            logger.exception("Failed to initialize built-in gesture recognizer")  # AUDIT-FIX(#1): Continue without built-in model.

        custom_gesture_recognizer: Any | None = None
        if self.config.custom_gesture_model_path:
            try:
                custom_gesture_recognizer = self._ensure_custom_gesture_recognizer(runtime)
            except FileNotFoundError:
                raise
            except Exception:
                logger.exception("Failed to initialize custom gesture recognizer")  # AUDIT-FIX(#1): Continue without custom model.

        if gesture_recognizer is not None:
            try:
                builtin_choice = resolve_fine_hand_gesture(
                    result=gesture_recognizer.recognize_for_video(
                        image,
                        timestamp_ms,
                    ),
                    category_map=BUILTIN_FINE_GESTURE_MAP,
                    min_score=self.config.builtin_gesture_min_score,
                )
                best_builtin = prefer_gesture_choice(
                    best_builtin,
                    _sanitize_gesture_choice(builtin_choice),
                )  # AUDIT-FIX(#7): Normalize malformed resolver outputs before comparison.
            except Exception:
                logger.exception("Built-in fine gesture inference failed for full-frame candidate")

        if custom_gesture_recognizer is not None:
            try:
                custom_choice = resolve_fine_hand_gesture(
                    result=custom_gesture_recognizer.recognize_for_video(
                        image,
                        timestamp_ms,
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

        return combine_builtin_and_custom_gesture_choice(best_builtin, best_custom)

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

        if not hand_landmark_result.detections:
            return (AICameraFineHandGesture.NONE, None), (AICameraFineHandGesture.NONE, None)

        builtin_choice = (AICameraFineHandGesture.NONE, None)
        custom_choice = (AICameraFineHandGesture.NONE, None)
        builtin_recognizer = self._ensure_roi_gesture_recognizer(runtime)
        custom_recognizer: Any | None = None
        if self.config.custom_gesture_model_path:
            custom_recognizer = self._ensure_custom_roi_gesture_recognizer(runtime)

        for detection in hand_landmark_result.detections:
            roi_frame = getattr(detection, "roi_frame_rgb", None)
            if roi_frame is None:
                continue
            roi_image = self._build_image(runtime, frame_rgb=roi_frame)
            roi_builtin = resolve_fine_hand_gesture(
                result=builtin_recognizer.recognize(roi_image),
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
                result=custom_recognizer.recognize(roi_image),
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
    ) -> tuple[dict[int, tuple[float, float, float]], float | None]:
        """Run pose inference and return sparse keypoints for downstream ROI seeding."""

        pose_landmarker = self._ensure_pose_landmarker(runtime)
        pose_result = pose_landmarker.detect_for_video(image, timestamp_ms)
        sparse_keypoints, pose_confidence = extract_sparse_keypoints(pose_result)
        return dict(sparse_keypoints), pose_confidence

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
            raw_timestamp_ms = self._last_timestamp_ms + 1  # AUDIT-FIX(#2): MediaPipe video tasks require strictly increasing timestamps.
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
            )  # AUDIT-FIX(#2): Keep local monotonic state even if runtime bookkeeping fails.

    def _ensure_pose_landmarker(self, runtime: dict[str, Any]) -> Any:
        """Preserve the historic pose-landmarker override point for tests."""

        if self._pose_landmarker is None:
            self._pose_landmarker = self._runtime.ensure_pose_landmarker(
                runtime,
            )  # AUDIT-FIX(#4): Cache the created task instance for stable lifecycle ownership.
        return self._pose_landmarker

    def _ensure_gesture_recognizer(self, runtime: dict[str, Any]) -> Any:
        """Preserve the historic gesture-recognizer override point for tests."""

        if self._gesture_recognizer is None:
            self._gesture_recognizer = self._runtime.ensure_gesture_recognizer(
                runtime,
            )  # AUDIT-FIX(#4): Cache the created task instance for stable lifecycle ownership.
        return self._gesture_recognizer

    def _ensure_custom_gesture_recognizer(self, runtime: dict[str, Any]) -> Any:
        """Preserve the historic custom-gesture override point for tests."""

        if self._custom_gesture_recognizer is None:
            self._custom_gesture_recognizer = self._runtime.ensure_custom_gesture_recognizer(
                runtime,
            )  # AUDIT-FIX(#4): Cache the created task instance for stable lifecycle ownership.
        return self._custom_gesture_recognizer

    def _ensure_roi_gesture_recognizer(self, runtime: dict[str, Any]) -> Any:
        """Preserve the historic ROI gesture-recognizer override point for tests."""

        return self._runtime.ensure_roi_gesture_recognizer(runtime)

    def _ensure_custom_roi_gesture_recognizer(self, runtime: dict[str, Any]) -> Any:
        """Preserve the historic ROI custom-gesture override point for tests."""

        return self._runtime.ensure_custom_roi_gesture_recognizer(runtime)


def extract_sparse_keypoints(result: Any) -> tuple[dict[int, tuple[float, float, float]], float | None]:
    """Extract a COCO-like sparse keypoint map from one pose-landmarker result."""

    pose_landmarks = getattr(result, "pose_landmarks", None) or ()
    if not pose_landmarks:
        return {}, None
    first_pose = pose_landmarks[0]
    if not first_pose:
        return {}, None

    sparse: dict[int, tuple[float, float, float]] = {}
    confidence_values: list[float] = []
    for mediapipe_index, coco_index in _COCO_KEYPOINT_INDEX.items():
        if mediapipe_index >= len(first_pose):
            continue
        landmark = first_pose[mediapipe_index]
        if landmark is None:
            continue

        x = _coerce_finite_float(getattr(landmark, "x", None))
        y = _coerce_finite_float(getattr(landmark, "y", None))
        try:
            score = _coerce_confidence(landmark_score(landmark))
        except Exception:
            continue  # AUDIT-FIX(#6): Skip malformed landmarks instead of poisoning the entire frame with one bad point.

        if x is None or y is None or score is None:
            continue  # AUDIT-FIX(#6): Avoid fabricating (0.0, 0.0) landmarks from missing/non-finite values.

        sparse[coco_index] = (x, y, score)
        confidence_values.append(score)

    if not confidence_values:
        return {}, None
    return sparse, round(sum(confidence_values) / float(len(confidence_values)), 3)


__all__ = [
    "MediaPipeVisionPipeline",
    "MediaPipeVisionResult",
    "extract_sparse_keypoints",
]
