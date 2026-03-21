from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.ai_camera import (
    AICameraBodyPose,
    AICameraBox,
    AICameraFineHandGesture,
    AICameraGestureEvent,
    AICameraMotionState,
    AICameraVisiblePerson,
    AICameraZone,
    AICameraAdapterConfig,
    LocalAICameraAdapter,
)
from twinr.hardware.camera_ai.detection import DetectionResult
from twinr.hardware.camera_ai.face_anchors import (
    SupplementalFaceAnchorResult,
    merge_detection_with_face_anchors,
)
from twinr.hardware.camera_ai.config import MediaPipeVisionConfig
from twinr.hardware.camera_ai.mediapipe_pipeline import MediaPipeVisionResult
from twinr.hardware.camera_ai.motion import infer_motion_state as _infer_motion_state
from twinr.hardware.camera_ai.pose_classification import (
    classify_body_pose as _classify_body_pose,
    classify_gesture as _classify_gesture,
)
from twinr.hardware.camera_ai.pose_features import support_pose_confidence as _support_pose_confidence
from twinr.hardware.camera_ai.pose_selection import rank_pose_candidates as _rank_pose_candidates
from twinr.hardware.ai_camera_diagnostics import capture_pose_probe
from twinr.hardware.hand_landmarks import HandLandmarkWorkerConfig
from twinr.config import TwinrConfig


class AICameraTests(unittest.TestCase):
    def test_box_clamps_and_orders_normalized_edges(self) -> None:
        box = AICameraBox(top=0.8, left=-0.2, bottom=0.1, right=1.4)

        self.assertEqual(box.top, 0.8)
        self.assertEqual(box.left, 0.0)
        self.assertEqual(box.bottom, 0.8)
        self.assertEqual(box.right, 1.0)
        self.assertAlmostEqual(box.center_x, 0.5, places=3)
        self.assertAlmostEqual(box.area, 0.0, places=3)

    def test_adapter_returns_explicit_health_failure_when_runtime_is_unavailable(self) -> None:
        adapter = LocalAICameraAdapter()
        adapter._load_detection_runtime = lambda: (_ for _ in ()).throw(RuntimeError("picamera2_unavailable"))

        observation = adapter.observe()

        self.assertFalse(observation.camera_online)
        self.assertFalse(observation.camera_ready)
        self.assertFalse(observation.camera_ai_ready)
        self.assertEqual(observation.camera_error, "picamera2_unavailable")

    def test_adapter_uses_mediapipe_pose_backend_when_configured(self) -> None:
        adapter = LocalAICameraAdapter(
            config=AICameraAdapterConfig(
                pose_backend="mediapipe",
                mediapipe_pose_model_path="state/mediapipe/models/pose_landmarker_full.task",
                mediapipe_gesture_model_path="state/mediapipe/models/gesture_recognizer.task",
            )
        )
        adapter._load_detection_runtime = lambda: {}
        adapter._probe_online = lambda runtime: None
        adapter._capture_detection = lambda runtime, observed_at: SimpleNamespace(
            person_count=1,
            primary_person_box=AICameraBox(top=0.12, left=0.24, bottom=0.92, right=0.68),
            primary_person_zone="center",
            person_near_device=True,
            hand_or_object_near_camera=False,
            objects=(),
        )
        adapter._capture_rgb_frame = lambda runtime, observed_at: object()
        adapter._ensure_mediapipe_pipeline = lambda: SimpleNamespace(
            analyze=lambda **_: MediaPipeVisionResult(
                body_pose=AICameraBodyPose.UPRIGHT,
                pose_confidence=0.74,
                looking_toward_device=True,
                visual_attention_score=0.82,
                hand_near_camera=True,
                showing_intent_likely=True,
                gesture_event=AICameraGestureEvent.WAVE,
                gesture_confidence=0.69,
                fine_hand_gesture=AICameraFineHandGesture.THUMBS_UP,
                fine_hand_gesture_confidence=0.88,
            )
        )
        adapter._resolve_motion = lambda **_: (AICameraMotionState.STILL, 0.57)

        observation = adapter.observe()

        self.assertEqual(observation.model, "local-imx500+mediapipe")
        self.assertEqual(observation.body_pose, AICameraBodyPose.UPRIGHT)
        self.assertEqual(observation.gesture_event, AICameraGestureEvent.WAVE)
        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.THUMBS_UP)
        self.assertAlmostEqual(observation.fine_hand_gesture_confidence or 0.0, 0.88, places=3)
        self.assertTrue(observation.camera_ready)

    def test_adapter_reports_missing_custom_mediapipe_gesture_model_explicitly(self) -> None:
        adapter = LocalAICameraAdapter(
            config=AICameraAdapterConfig(
                pose_backend="mediapipe",
                mediapipe_pose_model_path="state/mediapipe/models/pose_landmarker_full.task",
                mediapipe_hand_landmarker_model_path="state/mediapipe/models/hand_landmarker.task",
                mediapipe_gesture_model_path="state/mediapipe/models/gesture_recognizer.task",
                mediapipe_custom_gesture_model_path="state/mediapipe/models/custom_gesture.task",
            )
        )
        adapter._load_detection_runtime = lambda: {}
        adapter._probe_online = lambda runtime: None
        adapter._capture_detection = lambda runtime, observed_at: SimpleNamespace(
            person_count=1,
            primary_person_box=AICameraBox(top=0.12, left=0.24, bottom=0.92, right=0.68),
            primary_person_zone="center",
            person_near_device=True,
            hand_or_object_near_camera=False,
            objects=(),
        )
        adapter._capture_rgb_frame = lambda runtime, observed_at: object()
        adapter._ensure_mediapipe_pipeline = lambda: SimpleNamespace(
            analyze=lambda **_: (_ for _ in ()).throw(
                FileNotFoundError("mediapipe_custom_gesture_model_missing:state/mediapipe/models/custom_gesture.task")
            )
        )

        observation = adapter.observe()

        self.assertTrue(observation.camera_online)
        self.assertTrue(observation.camera_ready)
        self.assertFalse(observation.camera_ai_ready)
        self.assertEqual(observation.camera_error, "mediapipe_custom_gesture_model_missing")
        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.UNKNOWN)

    def test_adapter_reports_missing_hand_landmarker_model_explicitly(self) -> None:
        adapter = LocalAICameraAdapter(
            config=AICameraAdapterConfig(
                pose_backend="mediapipe",
                mediapipe_pose_model_path="state/mediapipe/models/pose_landmarker_full.task",
                mediapipe_hand_landmarker_model_path="state/mediapipe/models/hand_landmarker.task",
                mediapipe_gesture_model_path="state/mediapipe/models/gesture_recognizer.task",
            )
        )
        adapter._load_detection_runtime = lambda: {}
        adapter._probe_online = lambda runtime: None
        adapter._capture_detection = lambda runtime, observed_at: SimpleNamespace(
            person_count=1,
            primary_person_box=AICameraBox(top=0.12, left=0.24, bottom=0.92, right=0.68),
            primary_person_zone="center",
            person_near_device=True,
            hand_or_object_near_camera=False,
            objects=(),
        )
        adapter._capture_rgb_frame = lambda runtime, observed_at: object()
        adapter._ensure_mediapipe_pipeline = lambda: SimpleNamespace(
            analyze=lambda **_: (_ for _ in ()).throw(
                FileNotFoundError("mediapipe_hand_landmarker_model_missing:state/mediapipe/models/hand_landmarker.task")
            )
        )

        observation = adapter.observe()

        self.assertTrue(observation.camera_online)
        self.assertTrue(observation.camera_ready)
        self.assertFalse(observation.camera_ai_ready)
        self.assertEqual(observation.camera_error, "mediapipe_hand_landmarker_model_missing")
        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.UNKNOWN)

    def test_adapter_reports_detection_capture_failure_as_camera_unready(self) -> None:
        adapter = LocalAICameraAdapter()
        adapter._load_detection_runtime = lambda: {}
        adapter._probe_online = lambda runtime: None

        def _raise_detection_failure(runtime, observed_at):
            try:
                try:
                    raise OSError("Device or resource busy")
                except OSError as exc:
                    raise RuntimeError("session_start_failed") from exc
            except RuntimeError as exc:
                raise RuntimeError("detection_capture_failed") from exc

        adapter._capture_detection = _raise_detection_failure

        observation = adapter.observe()

        self.assertTrue(observation.camera_online)
        self.assertFalse(observation.camera_ready)
        self.assertFalse(observation.camera_ai_ready)
        self.assertEqual(observation.camera_error, "camera_busy")
        self.assertEqual(observation.person_count, 0)

    def test_adapter_supplements_second_visible_person_from_face_anchors(self) -> None:
        adapter = LocalAICameraAdapter(
            config=AICameraAdapterConfig(pose_backend="imx500"),
            face_anchor_detector=SimpleNamespace(
                detect=lambda frame: SupplementalFaceAnchorResult(
                    state="ok",
                    visible_persons=(
                        AICameraVisiblePerson(
                            box=AICameraBox(top=0.18, left=0.08, bottom=0.42, right=0.28),
                            zone=AICameraZone.LEFT,
                            confidence=0.91,
                        ),
                    ),
                    face_count=1,
                )
            ),
        )
        adapter._load_detection_runtime = lambda: {}
        adapter._probe_online = lambda runtime: None
        adapter._capture_detection = lambda runtime, observed_at: DetectionResult(
            person_count=1,
            primary_person_box=AICameraBox(top=0.12, left=0.62, bottom=0.94, right=0.94),
            primary_person_zone=AICameraZone.RIGHT,
            visible_persons=(
                AICameraVisiblePerson(
                    box=AICameraBox(top=0.12, left=0.62, bottom=0.94, right=0.94),
                    zone=AICameraZone.RIGHT,
                    confidence=0.78,
                ),
            ),
            person_near_device=False,
            hand_or_object_near_camera=False,
            objects=(),
        )
        adapter._capture_rgb_frame = lambda runtime, observed_at: SimpleNamespace(shape=(480, 640, 3))
        adapter._resolve_motion = lambda **_: (AICameraMotionState.STILL, 0.57)

        observation = adapter.observe()

        self.assertEqual(observation.person_count, 2)
        self.assertEqual(len(observation.visible_persons), 2)
        self.assertEqual(observation.visible_persons[0].zone, AICameraZone.RIGHT)
        self.assertEqual(observation.visible_persons[1].zone, AICameraZone.LEFT)

    def test_merge_detection_with_face_anchors_retargets_matching_person_to_face_box(self) -> None:
        detection = DetectionResult(
            person_count=1,
            primary_person_box=AICameraBox(top=0.10, left=0.30, bottom=0.95, right=0.82),
            primary_person_zone=AICameraZone.CENTER,
            visible_persons=(
                AICameraVisiblePerson(
                    box=AICameraBox(top=0.10, left=0.30, bottom=0.95, right=0.82),
                    zone=AICameraZone.CENTER,
                    confidence=0.73,
                ),
            ),
            person_near_device=False,
            hand_or_object_near_camera=False,
            objects=(),
        )

        merged = merge_detection_with_face_anchors(
            detection=detection,
            face_anchors=SupplementalFaceAnchorResult(
                state="ok",
                visible_persons=(
                    AICameraVisiblePerson(
                        box=AICameraBox(top=0.18, left=0.44, bottom=0.36, right=0.58),
                        zone=AICameraZone.CENTER,
                        confidence=0.93,
                    ),
                ),
                face_count=1,
            ),
        )

        self.assertEqual(merged.person_count, 1)
        self.assertEqual(len(merged.visible_persons), 1)
        self.assertEqual(merged.primary_person_box, detection.primary_person_box)
        self.assertEqual(
            merged.visible_persons[0].box,
            AICameraBox(top=0.18, left=0.44, bottom=0.36, right=0.58),
        )
        self.assertEqual(merged.visible_persons[0].zone, AICameraZone.CENTER)

    def test_adapter_config_exposes_live_hand_gesture_tuning_through_to_worker_config(self) -> None:
        twinr_config = TwinrConfig(
            proactive_local_camera_frame_rate=18,
            proactive_local_camera_builtin_gesture_min_score=0.31,
            proactive_local_camera_custom_gesture_min_score=0.44,
            proactive_local_camera_min_hand_detection_confidence=0.27,
            proactive_local_camera_min_hand_presence_confidence=0.28,
            proactive_local_camera_min_hand_tracking_confidence=0.29,
            proactive_local_camera_max_roi_candidates=5,
            proactive_local_camera_primary_person_roi_padding=0.22,
            proactive_local_camera_primary_person_upper_body_ratio=0.81,
            proactive_local_camera_wrist_roi_scale=0.39,
        )

        adapter_config = AICameraAdapterConfig.from_config(twinr_config)
        vision_config = MediaPipeVisionConfig.from_ai_camera_config(adapter_config)
        hand_config = HandLandmarkWorkerConfig.from_config(vision_config)

        self.assertEqual(adapter_config.frame_rate, 18)
        self.assertAlmostEqual(vision_config.builtin_gesture_min_score, 0.31, places=3)
        self.assertAlmostEqual(vision_config.custom_gesture_min_score, 0.44, places=3)
        self.assertAlmostEqual(hand_config.min_hand_detection_confidence, 0.27, places=3)
        self.assertAlmostEqual(hand_config.min_hand_presence_confidence, 0.28, places=3)
        self.assertAlmostEqual(hand_config.min_hand_tracking_confidence, 0.29, places=3)
        self.assertEqual(hand_config.max_roi_candidates, 5)
        self.assertAlmostEqual(hand_config.primary_person_roi_padding, 0.22, places=3)
        self.assertAlmostEqual(hand_config.primary_person_upper_body_ratio, 0.81, places=3)
        self.assertAlmostEqual(hand_config.wrist_roi_scale, 0.39, places=3)

    def test_adapter_config_auto_enables_staged_custom_gesture_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_model = Path(temp_dir) / "state" / "mediapipe" / "models" / "custom_gesture.task"
            custom_model.parent.mkdir(parents=True, exist_ok=True)
            custom_model.write_bytes(b"custom")

            twinr_config = TwinrConfig(project_root=temp_dir)

            adapter_config = AICameraAdapterConfig.from_config(twinr_config)
            vision_config = MediaPipeVisionConfig.from_ai_camera_config(adapter_config)

        self.assertEqual(adapter_config.mediapipe_custom_gesture_model_path, "state/mediapipe/models/custom_gesture.task")
        self.assertEqual(vision_config.custom_gesture_model_path, "state/mediapipe/models/custom_gesture.task")

    def test_support_pose_confidence_uses_keypoint_coverage_not_raw_score_alone(self) -> None:
        box = AICameraBox(top=0.10, left=0.35, bottom=0.92, right=0.62)
        keypoints = {
            0: (0.50, 0.18, 1.0),
            6: (0.47, 0.36, 1.0),
            10: (0.56, 0.55, 1.0),
            12: (0.50, 0.58, 1.0),
            16: (0.51, 0.90, 1.0),
        }

        confidence = _support_pose_confidence(2.7, keypoints, fallback_box=box)

        self.assertLess(confidence, 1.0)
        self.assertGreaterEqual(confidence, 0.30)

    def test_body_pose_falls_back_to_single_side_chain(self) -> None:
        box = AICameraBox(top=0.10, left=0.34, bottom=0.94, right=0.64)
        keypoints = {
            6: (0.49, 0.28, 0.95),
            12: (0.51, 0.56, 0.91),
            16: (0.52, 0.90, 0.93),
        }

        pose = _classify_body_pose(keypoints, fallback_box=box)

        self.assertEqual(pose, AICameraBodyPose.UPRIGHT)

    def test_body_pose_falls_back_to_tall_box_when_pose_is_sparse(self) -> None:
        box = AICameraBox(top=0.05, left=0.33, bottom=0.95, right=0.61)
        keypoints = {
            2: (0.52, 0.16, 0.88),
            10: (0.58, 0.56, 0.91),
            15: (0.42, 0.91, 0.94),
        }

        pose = _classify_body_pose(keypoints, fallback_box=box)

        self.assertEqual(pose, AICameraBodyPose.UPRIGHT)

    def test_body_pose_does_not_mark_near_square_tall_box_as_lying_low(self) -> None:
        box = AICameraBox(top=0.04, left=0.06, bottom=0.94, right=0.94)
        keypoints = {
            5: (0.38, 0.26, 0.91),
            11: (0.40, 0.57, 0.89),
            15: (0.41, 0.90, 0.92),
        }

        pose = _classify_body_pose(keypoints, fallback_box=box)

        self.assertNotEqual(pose, AICameraBodyPose.LYING_LOW)

    def test_gesture_can_fall_back_to_bbox_stop_when_wrist_is_raised(self) -> None:
        box = AICameraBox(top=0.22, left=0.40, bottom=0.76, right=0.60)
        keypoints = {
            10: (0.51, 0.26, 0.94),
        }

        gesture, confidence = _classify_gesture(keypoints, attention_score=0.35, fallback_box=box)

        self.assertEqual(gesture, AICameraGestureEvent.STOP)
        self.assertIsNotNone(confidence)

    def test_gesture_detects_left_hand_stop_from_shoulder_and_wrist(self) -> None:
        box = AICameraBox(top=0.20, left=0.34, bottom=0.88, right=0.66)
        keypoints = {
            5: (0.42, 0.47, 0.96),
            9: (0.44, 0.28, 0.95),
        }

        gesture, confidence = _classify_gesture(keypoints, attention_score=0.42, fallback_box=box)

        self.assertEqual(gesture, AICameraGestureEvent.STOP)
        self.assertGreater(confidence or 0.0, 0.8)

    def test_gesture_detects_right_hand_dismiss_from_shoulder_and_wrist(self) -> None:
        box = AICameraBox(top=0.18, left=0.30, bottom=0.86, right=0.72)
        keypoints = {
            6: (0.58, 0.46, 0.93),
            10: (0.82, 0.48, 0.94),
        }

        gesture, confidence = _classify_gesture(keypoints, attention_score=0.37, fallback_box=box)

        self.assertEqual(gesture, AICameraGestureEvent.DISMISS)
        self.assertGreater(confidence or 0.0, 0.6)

    def test_gesture_detects_right_hand_confirm_from_elbow_chain(self) -> None:
        box = AICameraBox(top=0.16, left=0.33, bottom=0.87, right=0.70)
        keypoints = {
            6: (0.58, 0.47, 0.94),
            8: (0.61, 0.61, 0.91),
            10: (0.59, 0.41, 0.95),
        }

        gesture, confidence = _classify_gesture(keypoints, attention_score=0.33, fallback_box=box)

        self.assertEqual(gesture, AICameraGestureEvent.CONFIRM)
        self.assertGreater(confidence or 0.0, 0.6)

    def test_gesture_detects_wave_from_raised_lateral_arm(self) -> None:
        box = AICameraBox(top=0.14, left=0.28, bottom=0.88, right=0.74)
        keypoints = {
            6: (0.58, 0.45, 0.94),
            8: (0.67, 0.34, 0.92),
            10: (0.79, 0.23, 0.95),
        }

        gesture, confidence = _classify_gesture(keypoints, attention_score=0.41, fallback_box=box)

        self.assertEqual(gesture, AICameraGestureEvent.WAVE)
        self.assertGreater(confidence or 0.0, 0.6)

    def test_gesture_detects_arms_crossed_from_crossing_wrists(self) -> None:
        box = AICameraBox(top=0.22, left=0.28, bottom=0.90, right=0.72)
        keypoints = {
            5: (0.42, 0.43, 0.93),
            6: (0.58, 0.43, 0.94),
            9: (0.57, 0.55, 0.95),
            10: (0.43, 0.56, 0.95),
        }

        gesture, confidence = _classify_gesture(keypoints, attention_score=0.28, fallback_box=box)

        self.assertEqual(gesture, AICameraGestureEvent.ARMS_CROSSED)
        self.assertGreater(confidence or 0.0, 0.6)

    def test_gesture_detects_two_hand_dismiss_from_symmetric_outreach(self) -> None:
        box = AICameraBox(top=0.20, left=0.24, bottom=0.90, right=0.76)
        keypoints = {
            5: (0.40, 0.45, 0.91),
            6: (0.60, 0.45, 0.92),
            9: (0.16, 0.47, 0.94),
            10: (0.84, 0.46, 0.95),
        }

        gesture, confidence = _classify_gesture(keypoints, attention_score=0.31, fallback_box=box)

        self.assertEqual(gesture, AICameraGestureEvent.TWO_HAND_DISMISS)
        self.assertGreater(confidence or 0.0, 0.65)

    def test_gesture_detects_timeout_t_from_horizontal_and_vertical_arms(self) -> None:
        box = AICameraBox(top=0.16, left=0.30, bottom=0.88, right=0.72)
        keypoints = {
            5: (0.42, 0.44, 0.92),
            6: (0.58, 0.44, 0.93),
            7: (0.48, 0.46, 0.90),
            8: (0.58, 0.36, 0.91),
            9: (0.54, 0.46, 0.94),
            10: (0.58, 0.24, 0.95),
        }

        gesture, confidence = _classify_gesture(keypoints, attention_score=0.34, fallback_box=box)

        self.assertEqual(gesture, AICameraGestureEvent.TIMEOUT_T)
        self.assertGreater(confidence or 0.0, 0.6)

    def test_gesture_bbox_fallback_detects_wide_dismiss_without_shoulder_joint(self) -> None:
        box = AICameraBox(top=0.20, left=0.34, bottom=0.84, right=0.66)
        keypoints = {
            10: (0.83, 0.44, 0.93),
        }

        gesture, confidence = _classify_gesture(keypoints, attention_score=0.29, fallback_box=box)

        self.assertEqual(gesture, AICameraGestureEvent.DISMISS)
        self.assertGreater(confidence or 0.0, 0.5)

    def test_gesture_does_not_trigger_for_relaxed_low_wrist(self) -> None:
        box = AICameraBox(top=0.18, left=0.34, bottom=0.88, right=0.66)
        keypoints = {
            5: (0.43, 0.45, 0.93),
            9: (0.47, 0.62, 0.94),
            6: (0.57, 0.45, 0.92),
            10: (0.60, 0.64, 0.91),
        }

        gesture, confidence = _classify_gesture(keypoints, attention_score=0.35, fallback_box=box)

        self.assertEqual(gesture, AICameraGestureEvent.NONE)
        self.assertIsNone(confidence)

    def test_motion_state_detects_approaching_from_box_growth(self) -> None:
        state, confidence = _infer_motion_state(
            previous_box=AICameraBox(top=0.26, left=0.34, bottom=0.70, right=0.62),
            current_box=AICameraBox(top=0.18, left=0.28, bottom=0.84, right=0.68),
            previous_observed_at=10.0,
            current_observed_at=11.0,
            previous_person_count=1,
            current_person_count=1,
        )

        self.assertEqual(state, AICameraMotionState.APPROACHING)
        self.assertGreater(confidence or 0.0, 0.5)

    def test_motion_state_detects_walking_from_lateral_shift(self) -> None:
        state, confidence = _infer_motion_state(
            previous_box=AICameraBox(top=0.14, left=0.18, bottom=0.86, right=0.42),
            current_box=AICameraBox(top=0.15, left=0.43, bottom=0.87, right=0.67),
            previous_observed_at=20.0,
            current_observed_at=21.2,
            previous_person_count=1,
            current_person_count=1,
        )

        self.assertEqual(state, AICameraMotionState.WALKING)
        self.assertGreater(confidence or 0.0, 0.5)

    def test_motion_state_detects_still_from_small_box_delta(self) -> None:
        state, confidence = _infer_motion_state(
            previous_box=AICameraBox(top=0.16, left=0.32, bottom=0.88, right=0.61),
            current_box=AICameraBox(top=0.17, left=0.33, bottom=0.87, right=0.60),
            previous_observed_at=30.0,
            current_observed_at=31.0,
            previous_person_count=1,
            current_person_count=1,
        )

        self.assertEqual(state, AICameraMotionState.STILL)
        self.assertGreater(confidence or 0.0, 0.5)

    def test_motion_state_stays_unknown_when_person_count_is_not_singular(self) -> None:
        state, confidence = _infer_motion_state(
            previous_box=AICameraBox(top=0.16, left=0.32, bottom=0.88, right=0.61),
            current_box=AICameraBox(top=0.17, left=0.33, bottom=0.87, right=0.60),
            previous_observed_at=30.0,
            current_observed_at=31.0,
            previous_person_count=2,
            current_person_count=1,
        )

        self.assertEqual(state, AICameraMotionState.UNKNOWN)
        self.assertIsNone(confidence)

    def test_pose_candidate_ranking_prefers_spatial_match_over_raw_score_spike(self) -> None:
        primary_box = AICameraBox(top=0.12, left=0.34, bottom=0.90, right=0.65)

        candidates = _rank_pose_candidates(
            keypoints=[
                [0.0] * 51,
                [0.0] * 51,
            ],
            scores=[0.45, 2.30],
            bboxes=[
                [70.0, 215.0, 430.0, 430.0],
                [20.0, 10.0, 470.0, 630.0],
            ],
            primary_person_box=primary_box,
            frame_width=640,
            frame_height=480,
        )

        self.assertEqual(candidates[0].candidate_index, 0)
        self.assertGreater(candidates[0].selection_score, candidates[1].selection_score)
        self.assertGreater(candidates[0].overlap, candidates[1].overlap)

    def test_capture_pose_probe_returns_bounded_error_when_runtime_is_unavailable(self) -> None:
        adapter = LocalAICameraAdapter()
        adapter._load_detection_runtime = lambda: (_ for _ in ()).throw(RuntimeError("picamera2_unavailable"))

        diagnostic = capture_pose_probe(adapter)

        self.assertEqual(diagnostic.camera_error, "picamera2_unavailable")
        self.assertEqual(diagnostic.person_count, 0)
        self.assertEqual(diagnostic.pose_people_count, 0)
        self.assertEqual(diagnostic.body_pose, AICameraBodyPose.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
