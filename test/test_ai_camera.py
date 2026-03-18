from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.ai_camera import (
    AICameraBodyPose,
    AICameraBox,
    AICameraGestureEvent,
    LocalAICameraAdapter,
    _classify_body_pose,
    _classify_gesture,
    _rank_pose_candidates,
    _support_pose_confidence,
)
from twinr.hardware.ai_camera_diagnostics import capture_pose_probe


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
