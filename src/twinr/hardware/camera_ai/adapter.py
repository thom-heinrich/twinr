"""Capture bounded local-first IMX500 camera observations for Twinr."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any
import logging
import math
import time

from twinr.agent.base_agent.config import TwinrConfig

from .config import AICameraAdapterConfig, MediaPipeVisionConfig
from .detection import DetectionResult, capture_detection
from .face_anchors import OpenCVFaceAnchorDetector, merge_detection_with_face_anchors
from .imx500_runtime import IMX500RuntimeSessionManager
from .live_gesture_pipeline import LiveGesturePipeline
from .mediapipe_pipeline import MediaPipeVisionPipeline
from .models import (
    AICameraBodyPose,
    AICameraFineHandGesture,
    AICameraGestureEvent,
    AICameraMotionState,
    AICameraObservation,
    AICameraZone,  # AUDIT-FIX(#10): retained for compatibility per audit constraint; unused in this module.
)
from .motion import infer_motion_state
from .pose_classification import classify_body_pose, classify_gesture
from .pose_features import attention_score, hand_near_camera, parse_keypoints, support_pose_confidence
from .pose_selection import rank_pose_candidates


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PoseResult:
    """Describe one parsed pose frame."""

    body_pose: AICameraBodyPose
    pose_confidence: float | None
    looking_toward_device: bool | None
    visual_attention_score: float | None
    hand_near_camera: bool
    showing_intent_likely: bool | None
    gesture_event: AICameraGestureEvent
    gesture_confidence: float | None
    fine_hand_gesture: AICameraFineHandGesture = AICameraFineHandGesture.NONE
    fine_hand_gesture_confidence: float | None = None


class LocalAICameraAdapter:
    """Provide one bounded local-first IMX500 observation surface."""

    def __init__(
        self,
        *,
        config: AICameraAdapterConfig | None = None,
        face_anchor_detector: object | None = None,
        clock: Any = time.time,
        sleep_fn: Any = time.sleep,
        monotonic_clock: Any = time.monotonic,
    ) -> None:
        """Initialize one bounded adapter with lazy runtime dependencies."""

        self.config = config or AICameraAdapterConfig()
        self._clock = clock
        self._sleep = sleep_fn
        self._monotonic_clock = monotonic_clock  # AUDIT-FIX(#4): Use monotonic time for internal duration math.
        self._lock = Lock()
        self._health_lock = Lock()  # AUDIT-FIX(#1): Health/frame metadata can be updated by timeout callers outside the main capture lock.
        self._runtime_manager = IMX500RuntimeSessionManager(config=self.config, sleep_fn=self._sleep)
        self._last_frame_at: float | None = None
        self._last_health_change_at: float | None = None
        self._last_health_signature: tuple[bool, bool, bool, str | None] | None = None
        self._last_pose_at: float | None = None
        self._last_pose_monotonic: float | None = None  # AUDIT-FIX(#4): Cache freshness must be independent of wall-clock jumps.
        self._last_pose_result: PoseResult | None = None
        self._last_pose_box_metrics: dict[str, float] | None = None  # AUDIT-FIX(#3): Reuse cached pose only for the same tracked person.
        self._mediapipe_pipeline: MediaPipeVisionPipeline | None = None
        self._live_gesture_pipeline: LiveGesturePipeline | None = None
        self._last_motion_box = None
        self._last_motion_person_count = 0
        self._last_motion_at: float | None = None
        self._last_motion_monotonic: float | None = None  # AUDIT-FIX(#4): Motion deltas must use monotonic time.
        self._last_motion_state = AICameraMotionState.UNKNOWN
        self._last_motion_confidence: float | None = None
        self._face_anchor_detector = face_anchor_detector

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "LocalAICameraAdapter":
        """Build one adapter directly from ``TwinrConfig``."""

        return cls(
            config=AICameraAdapterConfig.from_config(config),
            face_anchor_detector=OpenCVFaceAnchorDetector.from_runtime_config(config),
        )

    def close(self) -> None:
        """Close any active Picamera2 and MediaPipe sessions."""

        timeout_s = self._lock_timeout_s()  # AUDIT-FIX(#4): Clamp invalid timeout config to a bounded safe value.
        if not self._lock.acquire(timeout=timeout_s):
            logger.warning("Timed out waiting to acquire AI camera adapter lock during close.")
            return
        try:
            self._reset_runtime_state_locked(close_pipeline=True, clear_pose=True, clear_motion=True)  # AUDIT-FIX(#1): Serialize teardown with observe to avoid mid-flight closure.
        finally:
            self._lock.release()

    def observe(self) -> AICameraObservation:
        """Capture one local IMX500 observation or one explicit health failure."""

        lock_timeout_s = self._lock_timeout_s()  # AUDIT-FIX(#4): Normalize misconfigured durations before lock acquisition.
        if not self._lock.acquire(timeout=lock_timeout_s):
            return self._health_only_observation(
                observed_at=self._now(),  # AUDIT-FIX(#8): Timestamp lock timeouts at the actual failure point.
                online=True,
                ready=False,
                ai_ready=False,
                error="camera_lock_timeout",
            )
        observed_at = self._now()  # AUDIT-FIX(#8): Capture observation time after the wait on the adapter lock.
        observed_monotonic = self._monotonic_now()  # AUDIT-FIX(#4): Internal freshness/motion logic must use monotonic time.
        try:
            try:
                runtime = self._load_detection_runtime()
            except Exception as exc:  # pragma: no cover - depends on local environment.
                code = self._classify_error(exc)
                logger.warning("Local AI camera runtime load failed with %s.", code)
                logger.debug("Local AI camera runtime load exception details.", exc_info=True)
                self._reset_runtime_state_locked(close_pipeline=True, clear_pose=True, clear_motion=True)  # AUDIT-FIX(#2): Best-effort cleanup must not mask the primary failure.
                return self._health_only_observation(
                    observed_at=observed_at,
                    online=False,
                    ready=False,
                    ai_ready=False,
                    error=code,
                )
            online_error = self._probe_online(runtime)
            if online_error is not None:
                logger.warning("Local AI camera online probe failed with %s.", online_error)
                self._reset_runtime_state_locked(close_pipeline=True, clear_pose=True, clear_motion=True)  # AUDIT-FIX(#2): Reset broken runtime state before the next observation attempt.
                return self._health_only_observation(
                    observed_at=observed_at,
                    online=False,
                    ready=False,
                    ai_ready=False,
                    error=online_error,
                )

            detection = self._capture_detection(runtime, observed_at=observed_at)
            detection = self._coerce_detection_result(detection)
            frame_rgb = None
            frame_error = None
            if self._needs_rgb_frame_for_observation(detection=detection):
                frame_rgb, frame_error = self._capture_optional_rgb_frame(
                    runtime,
                    observed_at=observed_at,
                )
            pose_result, pose_error = self._resolve_pose(
                runtime,
                observed_at=observed_at,
                observed_monotonic=observed_monotonic,
                detection=detection,
                frame_rgb=frame_rgb,
                frame_error=frame_error,
            )
            detection = self._supplement_visible_persons(
                detection=detection,
                frame_rgb=frame_rgb,
            )
            observation = self._compose_observation(
                observed_at=observed_at,
                observed_monotonic=observed_monotonic,
                detection=detection,
                pose=pose_result,
                pose_error=pose_error,
            )
            return self._with_health(
                observation,
                online=True,
                ready=True,
                ai_ready=(pose_error is None),
                error=pose_error,
                frame_at=observed_at,
            )
        except Exception as exc:  # pragma: no cover - hardware and library behavior are environment-dependent.
            code = self._classify_error(exc)
            logger.warning("Local AI camera observation failed with %s.", code)
            logger.debug("Local AI camera observation exception details.", exc_info=True)
            self._reset_runtime_state_locked(close_pipeline=True, clear_pose=True, clear_motion=True)  # AUDIT-FIX(#2): Never let cleanup failure replace the original camera failure path.
            return self._health_only_observation(
                observed_at=observed_at,
                online=True,
                ready=False,
                ai_ready=False,
                error=code,
            )
        finally:
            self._lock.release()

    def observe_attention(self) -> AICameraObservation:
        """Capture one cheap person/anchor-only observation for HDMI eye-follow.

        The local HDMI attention loop must not pay for the full MediaPipe
        gesture stack on every refresh. This path keeps the same IMX500 person
        detection and optional face-anchor supplementation, but deliberately
        skips expensive pose/gesture inference so eye-follow stays reactive even
        while explicit hand-symbol tuning changes elsewhere.
        """

        lock_timeout_s = self._lock_timeout_s()
        if not self._lock.acquire(timeout=lock_timeout_s):
            return self._health_only_observation(
                observed_at=self._now(),
                online=True,
                ready=False,
                ai_ready=False,
                error="camera_lock_timeout",
            )
        observed_at = self._now()
        observed_monotonic = self._monotonic_now()
        try:
            try:
                runtime = self._load_detection_runtime()
            except Exception as exc:  # pragma: no cover - depends on local environment.
                code = self._classify_error(exc)
                logger.warning("Local AI camera attention runtime load failed with %s.", code)
                logger.debug("Local AI camera attention runtime load exception details.", exc_info=True)
                self._reset_runtime_state_locked(close_pipeline=False, clear_pose=False, clear_motion=False)
                return self._health_only_observation(
                    observed_at=observed_at,
                    online=False,
                    ready=False,
                    ai_ready=False,
                    error=code,
                )
            online_error = self._probe_online(runtime)
            if online_error is not None:
                logger.warning("Local AI camera attention online probe failed with %s.", online_error)
                return self._health_only_observation(
                    observed_at=observed_at,
                    online=False,
                    ready=False,
                    ai_ready=False,
                    error=online_error,
                )

            detection = self._coerce_detection_result(
                self._capture_detection(runtime, observed_at=observed_at)
            )
            frame_rgb = None
            if self._needs_rgb_frame_for_attention(detection=detection):
                frame_rgb, _frame_error = self._capture_optional_rgb_frame(
                    runtime,
                    observed_at=observed_at,
                )
                detection = self._supplement_visible_persons(
                    detection=detection,
                    frame_rgb=frame_rgb,
                )
            observation = self._compose_observation(
                observed_at=observed_at,
                observed_monotonic=observed_monotonic,
                detection=detection,
                pose=None,
                pose_error=None,
            )
            return self._with_health(
                observation,
                online=True,
                ready=True,
                ai_ready=True,
                error=None,
                frame_at=observed_at,
            )
        except Exception as exc:  # pragma: no cover - hardware and library behavior are environment-dependent.
            code = self._classify_error(exc)
            logger.warning("Local AI camera attention observation failed with %s.", code)
            logger.debug("Local AI camera attention observation exception details.", exc_info=True)
            return self._health_only_observation(
                observed_at=observed_at,
                online=True,
                ready=False,
                ai_ready=False,
                error=code,
            )
        finally:
            self._lock.release()

    def observe_gesture(self) -> AICameraObservation:
        """Capture one dedicated live-stream gesture observation for HDMI emoji ack.

        This path intentionally bypasses the general pose/social observation
        pipeline. It reuses the existing RGB preview session, but only feeds the
        frame into the thin live-stream gesture recognizers so user-facing
        symbol acknowledgement stays responsive and cannot regress eye-follow.
        """

        lock_timeout_s = self._lock_timeout_s()
        if not self._lock.acquire(timeout=lock_timeout_s):
            return self._health_only_observation(
                observed_at=self._now(),
                online=True,
                ready=False,
                ai_ready=False,
                error="camera_lock_timeout",
            )
        observed_at = self._now()
        try:
            try:
                runtime = self._load_detection_runtime()
            except Exception as exc:  # pragma: no cover - depends on local environment.
                code = self._classify_error(exc)
                logger.warning("Local AI camera gesture runtime load failed with %s.", code)
                logger.debug("Local AI camera gesture runtime load exception details.", exc_info=True)
                self._safe_close_live_gesture_pipeline_locked()
                self._safe_close_runtime_locked()
                return self._health_only_observation(
                    observed_at=observed_at,
                    online=False,
                    ready=False,
                    ai_ready=False,
                    error=code,
                )
            online_error = self._probe_online(runtime)
            if online_error is not None:
                logger.warning("Local AI camera gesture online probe failed with %s.", online_error)
                return self._health_only_observation(
                    observed_at=observed_at,
                    online=False,
                    ready=False,
                    ai_ready=False,
                    error=online_error,
                )
            try:
                frame_rgb = self._capture_rgb_frame(runtime, observed_at=observed_at)
                gesture_observation = self._ensure_live_gesture_pipeline().observe(
                    frame_rgb=frame_rgb,
                    observed_at=observed_at,
                )
            except Exception as exc:  # pragma: no cover - hardware-dependent path.
                code = self._classify_error(exc)
                logger.warning("Local AI camera live gesture observation failed with %s.", code)
                logger.debug("Local AI camera live gesture exception details.", exc_info=True)
                self._safe_close_live_gesture_pipeline_locked()
                self._safe_close_runtime_locked()
                return self._health_only_observation(
                    observed_at=observed_at,
                    online=True,
                    ready=False,
                    ai_ready=False,
                    error=code,
                )

            observation = AICameraObservation(
                observed_at=observed_at,
                camera_online=True,
                camera_ready=True,
                camera_ai_ready=True,
                hand_or_object_near_camera=gesture_observation.hand_count > 0,
                showing_intent_likely=(
                    True
                    if gesture_observation.hand_count > 0
                    or gesture_observation.fine_hand_gesture != AICameraFineHandGesture.NONE
                    or gesture_observation.gesture_event != AICameraGestureEvent.NONE
                    else None
                ),
                gesture_event=gesture_observation.gesture_event,
                gesture_confidence=gesture_observation.gesture_confidence,
                fine_hand_gesture=gesture_observation.fine_hand_gesture,
                fine_hand_gesture_confidence=gesture_observation.fine_hand_gesture_confidence,
                model="local-imx500+mediapipe-live-gesture",
            )
            return self._with_health(
                observation,
                online=True,
                ready=True,
                ai_ready=True,
                error=None,
                frame_at=observed_at,
            )
        finally:
            self._lock.release()

    def _resolve_pose(
        self,
        runtime: dict[str, Any],
        *,
        observed_at: float,
        observed_monotonic: float,
        detection: DetectionResult,
        frame_rgb: Any | None,
        frame_error: str | None,
    ) -> tuple[PoseResult | None, str | None]:
        """Return one fresh or cached pose result for the current detection frame."""

        if detection.person_count <= 0:
            self._safe_reset_mediapipe_temporal_state_locked()  # AUDIT-FIX(#9): No-person frames must not fail if MediaPipe temporal state reset breaks.
            self._clear_pose_cache()  # AUDIT-FIX(#3): Any previous pose is stale once no person is detected.
            return None, None

        if self.config.pose_backend == "mediapipe":
            return self._resolve_mediapipe_pose(
                runtime,
                observed_at=observed_at,
                observed_monotonic=observed_monotonic,
                detection=detection,
                frame_rgb=frame_rgb,
                frame_error=frame_error,
            )
        if not self.config.pose_network_path:
            self._clear_pose_cache()  # AUDIT-FIX(#3): Disable stale pose reuse when pose inference is unavailable.
            return None, None

        if self._should_reuse_pose_cache(
            observed_monotonic=observed_monotonic,
            primary_person_box=detection.primary_person_box,
        ):
            return self._last_pose_result, None

        try:
            pose_postprocess = self._runtime_manager.load_pose_postprocess()
        except Exception as exc:  # AUDIT-FIX(#5): Missing/broken pose postprocess assets should degrade to detection-only instead of failing the whole observation.
            code = self._classify_error(exc)
            logger.warning("Local AI camera pose postprocess load failed with %s.", code)
            logger.debug("Local AI camera pose postprocess load exception details.", exc_info=True)
            self._clear_pose_cache()  # AUDIT-FIX(#3): A failed pose setup invalidates any cached pose result.
            return None, code

        try:
            pose = self._capture_pose(
                runtime,
                pose_postprocess=pose_postprocess,
                observed_at=observed_at,
                primary_person_box=detection.primary_person_box,
            )
        except Exception as exc:  # pragma: no cover - hardware-dependent path.
            code = self._classify_error(exc)
            logger.warning("Local AI camera pose decode failed with %s.", code)
            logger.debug("Local AI camera pose decode exception details.", exc_info=True)
            self._clear_pose_cache()  # AUDIT-FIX(#3): Failed pose decode must not leave a stale cache behind.
            return None, code

        self._store_pose_result(
            pose,
            observed_at=observed_at,
            observed_monotonic=observed_monotonic,
            primary_person_box=detection.primary_person_box,
        )  # AUDIT-FIX(#3): Refresh the pose cache only after a successful decode for the current tracked person.
        return pose, None

    def _resolve_mediapipe_pose(
        self,
        runtime: dict[str, Any],
        *,
        observed_at: float,
        observed_monotonic: float,
        detection: DetectionResult,
        frame_rgb: Any | None,
        frame_error: str | None,
    ) -> tuple[PoseResult | None, str | None]:
        """Run the Pi-side MediaPipe pose and gesture path on the RGB preview frame."""

        if detection.primary_person_box is None:
            self._safe_reset_mediapipe_temporal_state_locked()  # AUDIT-FIX(#9): Missing primary boxes should reset temporal state without crashing.
            self._clear_pose_cache()  # AUDIT-FIX(#3): Prevent stale pose reuse when the tracked box disappears.
            return None, None
        if frame_error is not None:
            self._safe_close_mediapipe_pipeline_locked()
            self._clear_pose_cache()
            return None, frame_error
        try:
            if frame_rgb is None:
                frame_rgb = self._capture_rgb_frame(runtime, observed_at=observed_at)
            pipeline = self._ensure_mediapipe_pipeline()
            result = pipeline.analyze(
                frame_rgb=frame_rgb,
                observed_at=observed_at,
                primary_person_box=detection.primary_person_box,
            )
        except Exception as exc:  # pragma: no cover - depends on Pi runtime and model assets.
            code = self._classify_error(exc)
            logger.warning("Local AI camera MediaPipe inference failed with %s.", code)
            logger.debug("Local AI camera MediaPipe inference exception details.", exc_info=True)
            self._safe_close_mediapipe_pipeline_locked()  # AUDIT-FIX(#2): Recreate a bad MediaPipe pipeline on the next attempt.
            self._clear_pose_cache()  # AUDIT-FIX(#3): Failed inference invalidates any cached pose result.
            return None, code
        pose = PoseResult(
            body_pose=result.body_pose,
            pose_confidence=result.pose_confidence,
            looking_toward_device=result.looking_toward_device,
            visual_attention_score=result.visual_attention_score,
            hand_near_camera=result.hand_near_camera,
            showing_intent_likely=result.showing_intent_likely,
            gesture_event=result.gesture_event,
            gesture_confidence=result.gesture_confidence,
            fine_hand_gesture=result.fine_hand_gesture,
            fine_hand_gesture_confidence=result.fine_hand_gesture_confidence,
        )
        self._store_pose_result(
            pose,
            observed_at=observed_at,
            observed_monotonic=observed_monotonic,
            primary_person_box=detection.primary_person_box,
        )  # AUDIT-FIX(#3): Keep cached pose state aligned with the current person box.
        return pose, None

    def _capture_pose(
        self,
        runtime: dict[str, Any],
        *,
        pose_postprocess: Any,
        observed_at: float,
        primary_person_box: Any,
    ) -> PoseResult:
        """Capture one pose frame and decode one coarse pose sample."""

        session = self._runtime_manager.ensure_session(
            runtime,
            network_path=self.config.pose_network_path,
            task_name="pose",
        )
        metadata = self._runtime_manager.capture_metadata(session, observed_at=observed_at)
        outputs = session.imx500.get_outputs(metadata, add_batch=True)
        if outputs is None:
            raise RuntimeError("pose_outputs_missing")
        normalized_outputs = list(outputs)  # AUDIT-FIX(#6): Normalize runtime outputs before checking length; avoid ambiguous truthiness.
        if len(normalized_outputs) < 3:
            raise RuntimeError("pose_outputs_missing")
        if (
            len(normalized_outputs) >= 2
            and normalized_outputs[0].shape[-1] == 17
            and normalized_outputs[1].shape[-1] == 34
        ):
            normalized_outputs[0], normalized_outputs[1] = normalized_outputs[1], normalized_outputs[0]

        input_width, input_height = session.input_size
        keypoints, scores, bboxes = pose_postprocess(
            normalized_outputs,
            (self.config.main_size[1], self.config.main_size[0]),
            (0, 0),
            (0, 0),
            False,
            input_image_size=(input_height, input_width),
            output_shape=(normalized_outputs[0].shape[1], normalized_outputs[0].shape[2]),
        )
        if self._is_empty_result(keypoints) or self._is_empty_result(scores) or self._is_empty_result(bboxes):  # AUDIT-FIX(#6): NumPy-like results need length-based emptiness checks.
            raise RuntimeError("pose_people_missing")

        selected_keypoints, selected_score, selected_box = self._select_primary_pose(
            keypoints=keypoints,
            scores=scores,
            bboxes=bboxes,
            primary_person_box=primary_person_box,
        )
        parsed_keypoints = parse_keypoints(
            selected_keypoints,
            frame_width=self.config.main_size[0],
            frame_height=self.config.main_size[1],
        )
        pose_confidence = support_pose_confidence(
            selected_score,
            parsed_keypoints,
            fallback_box=selected_box,
        )
        if pose_confidence is None or pose_confidence < self.config.pose_confidence_threshold:
            raise RuntimeError("pose_confidence_low")
        visual_attention = attention_score(parsed_keypoints, fallback_box=selected_box)
        looking_toward_device = visual_attention >= self.config.attention_score_threshold
        hand_near = hand_near_camera(parsed_keypoints, fallback_box=selected_box)
        gesture_event, gesture_confidence = classify_gesture(
            parsed_keypoints,
            attention_score=visual_attention,
            fallback_box=selected_box,
        )
        return PoseResult(
            body_pose=classify_body_pose(parsed_keypoints, fallback_box=selected_box),
            pose_confidence=pose_confidence,
            looking_toward_device=looking_toward_device,
            visual_attention_score=visual_attention,
            hand_near_camera=hand_near,
            showing_intent_likely=hand_near and looking_toward_device,
            gesture_event=gesture_event,
            gesture_confidence=gesture_confidence,
        )

    def _ensure_mediapipe_pipeline(self) -> MediaPipeVisionPipeline:
        """Reuse or create the Pi-side MediaPipe inference pipeline lazily."""

        if self._mediapipe_pipeline is not None:
            return self._mediapipe_pipeline
        self._mediapipe_pipeline = MediaPipeVisionPipeline(
            config=MediaPipeVisionConfig.from_ai_camera_config(self.config),
        )
        return self._mediapipe_pipeline

    def _ensure_live_gesture_pipeline(self) -> LiveGesturePipeline:
        """Reuse or create the dedicated live-stream gesture pipeline lazily."""

        if self._live_gesture_pipeline is not None:
            return self._live_gesture_pipeline
        self._live_gesture_pipeline = LiveGesturePipeline(
            config=MediaPipeVisionConfig.from_ai_camera_config(self.config),
        )
        return self._live_gesture_pipeline

    def _load_detection_runtime(self) -> dict[str, Any]:
        """Preserve the historic detection-runtime override point for tests."""

        return self._runtime_manager.load_detection_runtime()

    def _probe_online(self, runtime: dict[str, Any]) -> str | None:
        """Preserve the historic online-probe override point for tests."""

        return self._runtime_manager.probe_online(runtime)

    def _capture_detection(self, runtime: dict[str, Any], *, observed_at: float) -> DetectionResult:
        """Preserve the historic detection-capture override point for tests."""

        return capture_detection(
            runtime_manager=self._runtime_manager,
            runtime=runtime,
            config=self.config,
            observed_at=observed_at,
        )

    def _capture_rgb_frame(self, runtime: dict[str, Any], *, observed_at: float) -> Any:
        """Preserve the historic RGB-capture override point for tests."""

        return self._runtime_manager.capture_rgb_frame(runtime, observed_at=observed_at)

    def _select_primary_pose(
        self,
        *,
        keypoints: list[list[float]],
        scores: list[float],
        bboxes: list[list[float]],
        primary_person_box: Any,
    ) -> tuple[list[float], float, Any]:
        """Return the pose sample that best matches the primary person."""

        candidates = rank_pose_candidates(
            keypoints=keypoints,
            scores=scores,
            bboxes=bboxes,
            primary_person_box=primary_person_box,
            frame_width=self.config.main_size[0],
            frame_height=self.config.main_size[1],
        )
        if not candidates:
            raise RuntimeError("pose_people_missing")
        selected = candidates[0]
        return selected.raw_keypoints, selected.raw_score, selected.box

    def _compose_observation(
        self,
        *,
        observed_at: float,
        observed_monotonic: float,
        detection: DetectionResult,
        pose: PoseResult | None,
        pose_error: str | None,
    ) -> AICameraObservation:
        """Merge detection and pose signals into one bounded observation."""

        primary_person_box = detection.primary_person_box
        primary_person_zone = detection.primary_person_zone
        person_near_device = detection.person_near_device

        visual_attention = None
        looking_toward_device = None
        engaged_with_device = None
        body_pose = AICameraBodyPose.UNKNOWN
        pose_confidence = None
        motion_state = AICameraMotionState.UNKNOWN
        motion_confidence = None
        gesture_event = AICameraGestureEvent.UNKNOWN if pose_error is not None else AICameraGestureEvent.NONE
        gesture_confidence = None
        fine_hand_gesture = AICameraFineHandGesture.UNKNOWN if pose_error is not None else AICameraFineHandGesture.NONE
        fine_hand_gesture_confidence = None
        showing_intent_likely = None
        hand_or_object_near_camera = detection.hand_or_object_near_camera

        if primary_person_box is not None:
            center_x = self._box_center_x(primary_person_box)
            if center_x is not None:
                base_center_score = 1.0 - min(1.0, abs(center_x - 0.5) / 0.5)
                visual_attention = round(base_center_score * 0.35, 3)
                looking_toward_device = visual_attention >= self.config.attention_score_threshold
                engaged_with_device = person_near_device is True and visual_attention >= self.config.engaged_score_threshold
                showing_intent_likely = hand_or_object_near_camera and (looking_toward_device or person_near_device is True)

        if pose is not None:
            if pose.visual_attention_score is not None:
                visual_attention = pose.visual_attention_score  # AUDIT-FIX(#7): Preserve detection-derived fallback signals when pose output is partial.
            if pose.looking_toward_device is not None:
                looking_toward_device = pose.looking_toward_device  # AUDIT-FIX(#7): Do not overwrite usable fallback gaze state with None.
            engaged_with_device = (
                detection.person_count > 0
                and (person_near_device is True or looking_toward_device is True)
                and (visual_attention or 0.0) >= self.config.engaged_score_threshold
            )
            body_pose = pose.body_pose
            pose_confidence = pose.pose_confidence
            gesture_event = pose.gesture_event
            gesture_confidence = pose.gesture_confidence
            fine_hand_gesture = pose.fine_hand_gesture
            fine_hand_gesture_confidence = pose.fine_hand_gesture_confidence
            hand_or_object_near_camera = hand_or_object_near_camera or pose.hand_near_camera
            if pose.showing_intent_likely is not None:
                showing_intent_likely = pose.showing_intent_likely  # AUDIT-FIX(#7): Keep fallback intent heuristics when pose-specific intent is unavailable.

        motion_state, motion_confidence = self._resolve_motion(
            observed_at=observed_at,
            observed_monotonic=observed_monotonic,
            person_count=detection.person_count,
            primary_person_box=primary_person_box,
        )

        return AICameraObservation(
            observed_at=observed_at,
            camera_online=True,
            camera_ready=True,
            camera_ai_ready=True,
            camera_error=pose_error,
            person_count=detection.person_count,
            primary_person_box=primary_person_box,
            primary_person_zone=primary_person_zone,
            visible_persons=detection.visible_persons,
            looking_toward_device=looking_toward_device,
            person_near_device=person_near_device,
            engaged_with_device=engaged_with_device,
            visual_attention_score=visual_attention,
            body_pose=body_pose,
            pose_confidence=pose_confidence,
            motion_state=motion_state,
            motion_confidence=motion_confidence,
            hand_or_object_near_camera=hand_or_object_near_camera,
            showing_intent_likely=showing_intent_likely,
            gesture_event=gesture_event,
            gesture_confidence=gesture_confidence,
            fine_hand_gesture=fine_hand_gesture,
            fine_hand_gesture_confidence=fine_hand_gesture_confidence,
            objects=detection.objects,
            model=("local-imx500+mediapipe" if self.config.pose_backend == "mediapipe" else "local-imx500"),
        )

    def _coerce_detection_result(self, detection: Any) -> DetectionResult:
        """Normalize foreign detection stubs into the stable detection contract."""

        if isinstance(detection, DetectionResult):
            return detection
        return DetectionResult(
            person_count=getattr(detection, "person_count", 0),
            primary_person_box=getattr(detection, "primary_person_box", None),
            primary_person_zone=getattr(detection, "primary_person_zone", AICameraZone.UNKNOWN),
            visible_persons=tuple(getattr(detection, "visible_persons", ()) or ()),
            person_near_device=getattr(detection, "person_near_device", None),
            hand_or_object_near_camera=bool(getattr(detection, "hand_or_object_near_camera", False)),
            objects=tuple(getattr(detection, "objects", ()) or ()),
        )

    def _needs_rgb_frame_for_observation(self, *, detection: DetectionResult) -> bool:
        """Return whether this observation needs a preview RGB frame."""

        if self.config.pose_backend == "mediapipe" and detection.person_count > 0:
            return True
        return self._face_anchor_detector is not None and detection.person_count < 2

    def _needs_rgb_frame_for_attention(self, *, detection: DetectionResult) -> bool:
        """Return whether the cheap attention path needs one preview frame.

        Attention-only refresh uses RGB solely for optional face-anchor
        supplementation when SSD does not already see two people. It never
        promotes RGB capture to mandatory pose/gesture inference.
        """

        return self._face_anchor_detector is not None and detection.person_count < 2

    def _capture_optional_rgb_frame(
        self,
        runtime: dict[str, Any],
        *,
        observed_at: float,
    ) -> tuple[Any | None, str | None]:
        """Capture one RGB frame without promoting preview failures to full camera failure."""

        try:
            return self._capture_rgb_frame(runtime, observed_at=observed_at), None
        except Exception as exc:
            code = self._classify_error(exc)
            logger.warning("Local AI camera RGB preview capture failed with %s.", code)
            logger.debug("Local AI camera RGB preview exception details.", exc_info=True)
            return None, code

    def _supplement_visible_persons(
        self,
        *,
        detection: DetectionResult,
        frame_rgb: Any | None,
    ) -> DetectionResult:
        """Add bounded supplemental face anchors when SSD saw fewer than two people."""

        if self._face_anchor_detector is None or frame_rgb is None or detection.person_count >= 2:
            return detection
        try:
            face_result = self._face_anchor_detector.detect(frame_rgb)
            return merge_detection_with_face_anchors(
                detection=detection,
                face_anchors=face_result,
            )
        except Exception:
            logger.warning("Local AI camera supplemental face-anchor detection failed.")
            logger.debug("Local AI camera face-anchor exception details.", exc_info=True)
            return detection

    def _health_only_observation(
        self,
        *,
        observed_at: float,
        online: bool,
        ready: bool,
        ai_ready: bool,
        error: str | None,
    ) -> AICameraObservation:
        """Return one explicit health-only observation."""

        observation = AICameraObservation(
            observed_at=observed_at,
            camera_online=online,
            camera_ready=ready,
            camera_ai_ready=ai_ready,
            camera_error=error,
        )
        return self._with_health(observation, online=online, ready=ready, ai_ready=ai_ready, error=error, frame_at=None)

    def _with_health(
        self,
        observation: AICameraObservation,
        *,
        online: bool,
        ready: bool,
        ai_ready: bool,
        error: str | None,
        frame_at: float | None,
    ) -> AICameraObservation:
        """Attach the latest health timestamps to one observation."""

        with self._health_lock:  # AUDIT-FIX(#1): Serialize health/frame metadata updates across success and timeout paths.
            if frame_at is not None:
                self._last_frame_at = frame_at
            health_signature = (online, ready, ai_ready, error)
            if health_signature != self._last_health_signature:
                self._last_health_signature = health_signature
                self._last_health_change_at = observation.observed_at
            last_camera_frame_at = self._last_frame_at
            last_camera_health_change_at = self._last_health_change_at
        return AICameraObservation(
            observed_at=observation.observed_at,
            camera_online=online,
            camera_ready=ready,
            camera_ai_ready=ai_ready,
            camera_error=error,
            last_camera_frame_at=last_camera_frame_at,
            last_camera_health_change_at=last_camera_health_change_at,
            person_count=observation.person_count,
            primary_person_box=observation.primary_person_box,
            primary_person_zone=observation.primary_person_zone,
            visible_persons=observation.visible_persons,
            looking_toward_device=observation.looking_toward_device,
            person_near_device=observation.person_near_device,
            engaged_with_device=observation.engaged_with_device,
            visual_attention_score=observation.visual_attention_score,
            body_pose=observation.body_pose,
            pose_confidence=observation.pose_confidence,
            motion_state=observation.motion_state,
            motion_confidence=observation.motion_confidence,
            hand_or_object_near_camera=observation.hand_or_object_near_camera,
            showing_intent_likely=observation.showing_intent_likely,
            gesture_event=observation.gesture_event,
            gesture_confidence=observation.gesture_confidence,
            fine_hand_gesture=observation.fine_hand_gesture,
            fine_hand_gesture_confidence=observation.fine_hand_gesture_confidence,
            objects=observation.objects,
            model=observation.model,
        )

    def _resolve_motion(
        self,
        *,
        observed_at: float,
        observed_monotonic: float,
        person_count: int,
        primary_person_box: Any,
    ) -> tuple[AICameraMotionState, float | None]:
        """Derive one coarse motion state from recent primary-person box deltas."""

        motion_state, motion_confidence = infer_motion_state(
            previous_box=self._last_motion_box,
            current_box=primary_person_box,
            previous_observed_at=self._last_motion_monotonic,
            current_observed_at=observed_monotonic,
            previous_person_count=self._last_motion_person_count,
            current_person_count=person_count,
        )
        self._last_motion_box = primary_person_box
        self._last_motion_person_count = max(0, int(person_count))
        self._last_motion_at = observed_at
        self._last_motion_monotonic = observed_monotonic
        self._last_motion_state = motion_state
        self._last_motion_confidence = motion_confidence
        return motion_state, motion_confidence

    def _reset_runtime_state_locked(
        self,
        *,
        close_pipeline: bool,
        clear_pose: bool,
        clear_motion: bool,
    ) -> None:
        """Best-effort cleanup for runtime failures while the adapter lock is held."""

        self._safe_close_runtime_locked()  # AUDIT-FIX(#2): Cleanup must never raise over the primary capture failure.
        if close_pipeline:
            self._safe_close_mediapipe_pipeline_locked()  # AUDIT-FIX(#2): Drop potentially corrupted MediaPipe state before the next attempt.
            self._safe_close_live_gesture_pipeline_locked()
        if clear_pose:
            self._clear_pose_cache()  # AUDIT-FIX(#3): Reset stale pose cache across runtime failures and close().
        if clear_motion:
            self._clear_motion_state()  # AUDIT-FIX(#3): Reset stale motion history across runtime failures and close().

    def _safe_close_runtime_locked(self) -> None:
        """Close the IMX500 runtime manager without raising."""

        try:
            self._runtime_manager.close()
        except Exception:  # pragma: no cover - depends on runtime state.
            logger.debug("Ignoring runtime close failure during AI camera cleanup.", exc_info=True)

    def _safe_close_mediapipe_pipeline_locked(self) -> None:
        """Close the MediaPipe pipeline without raising."""

        pipeline = self._mediapipe_pipeline
        self._mediapipe_pipeline = None
        if pipeline is None:
            return
        try:
            pipeline.close()
        except Exception:  # pragma: no cover - depends on MediaPipe runtime state.
            logger.debug("Ignoring MediaPipe close failure during AI camera cleanup.", exc_info=True)

    def _safe_close_live_gesture_pipeline_locked(self) -> None:
        """Close the live-stream gesture pipeline without raising."""

        pipeline = self._live_gesture_pipeline
        self._live_gesture_pipeline = None
        if pipeline is None:
            return
        try:
            pipeline.close()
        except Exception:  # pragma: no cover - depends on MediaPipe runtime state.
            logger.debug("Ignoring live gesture pipeline close failure during AI camera cleanup.", exc_info=True)

    def _safe_reset_mediapipe_temporal_state_locked(self) -> None:
        """Reset MediaPipe temporal state without failing observation capture."""

        pipeline = self._mediapipe_pipeline
        if pipeline is None:
            return
        try:
            pipeline.reset_temporal_state()
        except Exception:  # pragma: no cover - depends on MediaPipe runtime state.
            logger.warning("Local AI camera MediaPipe temporal reset failed; recreating pipeline.")  # AUDIT-FIX(#9): A reset failure should degrade gracefully, not abort the frame.
            logger.debug("Local AI camera MediaPipe temporal reset exception details.", exc_info=True)
            self._safe_close_mediapipe_pipeline_locked()

    def _clear_pose_cache(self) -> None:
        """Clear pose cache state."""

        self._last_pose_result = None  # AUDIT-FIX(#3): Failed or absent frames must not leave stale cached pose data behind.
        self._last_pose_at = None
        self._last_pose_monotonic = None
        self._last_pose_box_metrics = None

    def _store_pose_result(
        self,
        pose: PoseResult,
        *,
        observed_at: float,
        observed_monotonic: float,
        primary_person_box: Any,
    ) -> None:
        """Store one pose result for bounded cache reuse."""

        self._last_pose_result = pose
        self._last_pose_at = observed_at
        self._last_pose_monotonic = observed_monotonic
        self._last_pose_box_metrics = self._extract_box_metrics(primary_person_box)

    def _clear_motion_state(self) -> None:
        """Clear motion history."""

        self._last_motion_box = None  # AUDIT-FIX(#3): Motion inference must not bridge across resets or hard failures.
        self._last_motion_person_count = 0
        self._last_motion_at = None
        self._last_motion_monotonic = None
        self._last_motion_state = AICameraMotionState.UNKNOWN
        self._last_motion_confidence = None

    def _should_reuse_pose_cache(
        self,
        *,
        observed_monotonic: float,
        primary_person_box: Any,
    ) -> bool:
        """Return whether the cached pose is still valid for the current tracked person."""

        refresh_s = self._pose_refresh_s()  # AUDIT-FIX(#4): Use sanitized monotonic freshness windows for pose cache reuse.
        if refresh_s <= 0.0:
            return False
        if self._last_pose_result is None or self._last_pose_monotonic is None or self._last_pose_box_metrics is None:
            return False
        age_s = observed_monotonic - self._last_pose_monotonic
        if age_s < 0.0 or age_s > refresh_s:
            return False
        current_box_metrics = self._extract_box_metrics(primary_person_box)
        if current_box_metrics is None:
            return False
        return self._box_metrics_similar(self._last_pose_box_metrics, current_box_metrics)

    def _box_center_x(self, box: Any) -> float | None:
        """Return one finite center-x value for a box when available."""

        center_x = self._attribute_float(box, "center_x")
        if center_x is not None:
            return center_x
        metrics = self._extract_box_metrics(box)
        if metrics is None:
            return None
        return metrics["center_x"]

    def _attribute_float(self, obj: Any, attr: str) -> float | None:
        """Return one finite float attribute when present."""

        if obj is None or not hasattr(obj, attr):
            return None
        return self._coerce_finite_float(getattr(obj, attr))

    def _extract_box_metrics(self, box: Any) -> dict[str, float] | None:
        """Extract normalized box metrics for cache matching and heuristics."""

        if box is None:
            return None

        center_x = self._attribute_float(box, "center_x")
        center_y = self._attribute_float(box, "center_y")
        width = self._attribute_float(box, "width")
        height = self._attribute_float(box, "height")
        if center_x is not None and center_y is not None and width is not None and height is not None:
            return {
                "center_x": center_x,
                "center_y": center_y,
                "width": width,
                "height": height,
            }

        left = self._attribute_float(box, "left")
        top = self._attribute_float(box, "top")
        right = self._attribute_float(box, "right")
        bottom = self._attribute_float(box, "bottom")
        if left is not None and top is not None and right is not None and bottom is not None:
            return {
                "center_x": (left + right) / 2.0,
                "center_y": (top + bottom) / 2.0,
                "width": abs(right - left),
                "height": abs(bottom - top),
            }

        x = self._attribute_float(box, "x")
        y = self._attribute_float(box, "y")
        w = self._attribute_float(box, "w")
        h = self._attribute_float(box, "h")
        if x is not None and y is not None and w is not None and h is not None:
            return {
                "center_x": x + (w / 2.0),
                "center_y": y + (h / 2.0),
                "width": abs(w),
                "height": abs(h),
            }

        return None

    def _box_metrics_similar(
        self,
        previous_metrics: dict[str, float],
        current_metrics: dict[str, float],
    ) -> bool:
        """Return whether two normalized box measurements likely describe the same person."""

        center_delta = max(
            abs(previous_metrics["center_x"] - current_metrics["center_x"]),
            abs(previous_metrics["center_y"] - current_metrics["center_y"]),
        )
        size_delta = max(
            abs(previous_metrics["width"] - current_metrics["width"]),
            abs(previous_metrics["height"] - current_metrics["height"]),
        )
        return center_delta <= 0.08 and size_delta <= 0.20

    def _is_empty_result(self, value: Any) -> bool:
        """Return whether one result container is empty without relying on ambiguous truthiness."""

        if value is None:
            return True
        try:
            return len(value) == 0
        except TypeError:
            return False

    def _coerce_finite_float(self, value: Any) -> float | None:
        """Return one finite float or ``None``."""

        try:
            normalized = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(normalized):
            return None
        return normalized

    def _normalized_nonnegative_float(self, value: Any, *, default: float) -> float:
        """Return one finite non-negative float or a bounded default."""

        normalized = self._coerce_finite_float(value)
        if normalized is None or normalized < 0.0:
            return default
        return normalized

    def _lock_timeout_s(self) -> float:
        """Return one bounded lock timeout."""

        return self._normalized_nonnegative_float(self.config.lock_timeout_s, default=1.0)

    def _pose_refresh_s(self) -> float:
        """Return one bounded pose-cache refresh interval."""

        return self._normalized_nonnegative_float(self.config.pose_refresh_s, default=0.0)

    def _classify_error(self, exc: Exception) -> str:
        """Return one stable operator-facing error code for the exception."""

        message = " ".join(self._iter_error_messages(exc))
        if "camera __init__ sequence did not complete" in message or "device or resource busy" in message:
            return "camera_busy"
        if "session_start_failed" in message:
            return "camera_session_start_failed"
        if "requested camera dev-node not found" in message:
            return "imx500_not_enumerated"
        if "picamera2_unavailable" in message:
            return "picamera2_unavailable"
        if "detection_capture_failed" in message:
            return "detection_capture_failed"
        if "detection_outputs_missing" in message:
            return "detection_outputs_missing"
        if "detection_outputs_invalid_container" in message:
            return "detection_outputs_invalid_container"
        if "detection_outputs_incomplete" in message:
            return "detection_outputs_incomplete"
        if "detection_parse_failed" in message:
            return "detection_parse_failed"
        if "metadata_timeout" in message:
            return "metadata_timeout"
        if "mediapipe_custom_gesture_model_missing" in message:
            return "mediapipe_custom_gesture_model_missing"
        if "mediapipe_pose_model_missing" in message:
            return "mediapipe_pose_model_missing"
        if "mediapipe_hand_landmarker_model_missing" in message:
            return "mediapipe_hand_landmarker_model_missing"
        if "mediapipe_gesture_model_missing" in message:
            return "mediapipe_gesture_model_missing"
        if "mediapipe_unavailable" in message:
            return "mediapipe_unavailable"
        if "model_missing" in message:
            return "model_missing"
        if "pose_dependency_missing" in message:
            return "pose_dependency_missing"
        if "rgb_frame_missing" in message:
            return "rgb_frame_missing"
        if "pose_outputs_missing" in message or "pose_people_missing" in message or "operands could not be broadcast together" in message:
            return "pose_decode_failed"
        if "pose_confidence_low" in message:
            return "pose_confidence_low"
        if isinstance(exc, FileNotFoundError):
            return "model_missing"
        return exc.__class__.__name__.lower()

    def _iter_error_messages(self, exc: BaseException) -> tuple[str, ...]:
        """Flatten the exception/cause chain into compact lowercase messages."""

        messages: list[str] = []
        seen: set[int] = set()
        current: BaseException | None = exc
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            text = " ".join(str(current).strip().lower().split())
            if text:
                messages.append(text)
            next_error = current.__cause__ or current.__context__
            current = next_error if isinstance(next_error, BaseException) else None
        return tuple(messages)

    def _now(self) -> float:
        """Return one bounded wall-clock timestamp."""

        try:
            value = float(self._clock())
        except Exception:
            return time.time()
        if not math.isfinite(value) or value < 0.0:
            return time.time()
        return value

    def _monotonic_now(self) -> float:
        """Return one bounded monotonic timestamp for internal interval math."""

        try:
            value = float(self._monotonic_clock())
        except Exception:
            return time.monotonic()
        if not math.isfinite(value) or value < 0.0:
            return time.monotonic()
        return value


__all__ = ["LocalAICameraAdapter", "PoseResult"]
