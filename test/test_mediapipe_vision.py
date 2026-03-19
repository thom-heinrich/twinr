from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.ai_camera import AICameraFineHandGesture, AICameraGestureEvent
from twinr.hardware.mediapipe_vision import (
    MediaPipeVisionConfig,
    MediaPipeVisionPipeline,
    TemporalPoseGestureClassifier,
    _classify_temporal_gesture,
    _resolve_fine_hand_gesture,
)


class _Category:
    def __init__(self, category_name: str, score: float) -> None:
        self.category_name = category_name
        self.score = score


class _GestureResult:
    def __init__(self, gestures):
        self.gestures = gestures


class _Frame:
    def __init__(
        self,
        *,
        observed_at: float,
        left_shoulder=None,
        right_shoulder=None,
        left_elbow=None,
        right_elbow=None,
        left_wrist=None,
        right_wrist=None,
    ) -> None:
        self.observed_at = observed_at
        self.left_shoulder = left_shoulder
        self.right_shoulder = right_shoulder
        self.left_elbow = left_elbow
        self.right_elbow = right_elbow
        self.left_wrist = left_wrist
        self.right_wrist = right_wrist
        self.left_hip = None
        self.right_hip = None


class MediaPipeVisionTests(unittest.TestCase):
    def test_fine_gesture_mapping_picks_builtin_pointing(self) -> None:
        gesture, confidence = _resolve_fine_hand_gesture(
            result=_GestureResult([[ _Category("Pointing_Up", 0.87) ]]),
            category_map={"pointing_up": AICameraFineHandGesture.POINTING},
            min_score=0.50,
        )

        self.assertEqual(gesture, AICameraFineHandGesture.POINTING)
        self.assertAlmostEqual(confidence or 0.0, 0.87, places=3)

    def test_fine_gesture_mapping_picks_custom_ok_sign(self) -> None:
        gesture, confidence = _resolve_fine_hand_gesture(
            result=_GestureResult([[_Category("ok_sign", 0.91)]]),
            category_map={"ok_sign": AICameraFineHandGesture.OK_SIGN},
            min_score=0.55,
        )

        self.assertEqual(gesture, AICameraFineHandGesture.OK_SIGN)
        self.assertAlmostEqual(confidence or 0.0, 0.91, places=3)

    def test_temporal_classifier_detects_wave_from_lateral_raised_hand_motion(self) -> None:
        frames = (
            _Frame(
                observed_at=0.0,
                right_shoulder=(0.60, 0.48, 1.0),
                right_wrist=(0.72, 0.26, 1.0),
            ),
            _Frame(
                observed_at=0.4,
                right_shoulder=(0.60, 0.48, 1.0),
                right_wrist=(0.84, 0.24, 1.0),
            ),
            _Frame(
                observed_at=0.8,
                right_shoulder=(0.60, 0.48, 1.0),
                right_wrist=(0.70, 0.25, 1.0),
            ),
            _Frame(
                observed_at=1.2,
                right_shoulder=(0.60, 0.48, 1.0),
                right_wrist=(0.86, 0.24, 1.0),
            ),
        )

        gesture, confidence = _classify_temporal_gesture(frames, min_frames=4)

        self.assertEqual(gesture, AICameraGestureEvent.WAVE)
        self.assertGreater(confidence or 0.0, 0.6)

    def test_temporal_classifier_detects_arms_crossed_from_sequence(self) -> None:
        frames = tuple(
            _Frame(
                observed_at=float(index),
                left_shoulder=(0.42, 0.42, 1.0),
                right_shoulder=(0.58, 0.42, 1.0),
                left_wrist=(0.60, 0.55, 1.0),
                right_wrist=(0.40, 0.55, 1.0),
            )
            for index in range(4)
        )

        gesture, confidence = _classify_temporal_gesture(frames, min_frames=4)

        self.assertEqual(gesture, AICameraGestureEvent.ARMS_CROSSED)
        self.assertGreater(confidence or 0.0, 0.5)

    def test_temporal_classifier_detects_timeout_t_from_sequence(self) -> None:
        frames = tuple(
            _Frame(
                observed_at=float(index),
                left_shoulder=(0.42, 0.44, 1.0),
                left_elbow=(0.49, 0.45, 1.0),
                left_wrist=(0.58, 0.45, 1.0),
                right_shoulder=(0.60, 0.44, 1.0),
                right_elbow=(0.60, 0.36, 1.0),
                right_wrist=(0.60, 0.24, 1.0),
            )
            for index in range(4)
        )

        gesture, confidence = _classify_temporal_gesture(frames, min_frames=4)

        self.assertEqual(gesture, AICameraGestureEvent.TIMEOUT_T)
        self.assertGreater(confidence or 0.0, 0.5)

    def test_configured_custom_gesture_model_missing_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            builtin_model = Path(temp_dir) / "gesture.task"
            builtin_model.write_bytes(b"builtin")
            pipeline = MediaPipeVisionPipeline(
                config=MediaPipeVisionConfig(
                    pose_model_path="pose.task",
                    gesture_model_path=str(builtin_model),
                    custom_gesture_model_path=str(Path(temp_dir) / "missing-custom.task"),
                )
            )
            pipeline._gesture_recognizer = type(
                "_StubGestureRecognizer",
                (),
                {"recognize_for_video": staticmethod(lambda image, timestamp_ms: _GestureResult([]))},
            )()

            with self.assertRaises(FileNotFoundError) as context:
                pipeline._recognize_fine_gesture(runtime={}, image=object(), timestamp_ms=1)

        self.assertIn("mediapipe_custom_gesture_model_missing", str(context.exception))

    def test_temporal_pose_classifier_prunes_old_frames(self) -> None:
        classifier = TemporalPoseGestureClassifier(window_s=1.0, min_frames=2)
        classifier.observe(
            observed_at=0.0,
            sparse_keypoints={
                6: (0.60, 0.48, 1.0),
                10: (0.72, 0.26, 1.0),
            },
        )
        classifier.observe(
            observed_at=2.0,
            sparse_keypoints={
                6: (0.60, 0.48, 1.0),
                10: (0.84, 0.24, 1.0),
            },
        )

        self.assertEqual(len(classifier._frames), 1)


if __name__ == "__main__":
    unittest.main()
