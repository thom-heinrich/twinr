from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.camera_ai.config import MediaPipeVisionConfig
from twinr.hardware.camera_ai.live_gesture_pipeline import LiveGesturePipeline
from twinr.hardware.camera_ai.models import AICameraFineHandGesture, AICameraGestureEvent


class _Category:
    def __init__(self, category_name: str, score: float) -> None:
        self.category_name = category_name
        self.score = score


class _Landmark:
    def __init__(self, x: float, y: float = 0.5, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z


class _GestureResult:
    def __init__(self, *, gestures, hand_landmarks) -> None:
        self.gestures = gestures
        self.hand_landmarks = hand_landmarks


class _StubRecognizer:
    def __init__(self, callback, results) -> None:
        self._callback = callback
        self._results = list(results)

    def recognize_async(self, image, timestamp_ms: int) -> None:
        result = self._results.pop(0) if len(self._results) > 1 else self._results[0]
        self._callback(result, image, timestamp_ms)


class _StubRuntime:
    def __init__(self, *, builtin_results, custom_results=None) -> None:
        self._builtin_results = list(builtin_results)
        self._custom_results = list(custom_results or [])
        self._builtin = None
        self._custom = None

    def load_runtime(self):
        return {}

    def build_image(self, runtime, *, frame_rgb):
        return frame_rgb

    def timestamp_ms(self, observed_at: float) -> int:
        return int(round(observed_at * 1000.0))

    def ensure_live_gesture_recognizer(self, runtime, *, result_callback):
        if self._builtin is None:
            self._builtin = _StubRecognizer(result_callback, self._builtin_results)
        return self._builtin

    def ensure_live_custom_gesture_recognizer(self, runtime, *, result_callback):
        if self._custom is None:
            self._custom = _StubRecognizer(result_callback, self._custom_results)
        return self._custom

    def close(self) -> None:
        return None


class LiveGesturePipelineTests(unittest.TestCase):
    def test_observe_uses_live_stream_builtin_result_for_fast_symbol(self) -> None:
        pipeline = LiveGesturePipeline(
            config=MediaPipeVisionConfig(
                pose_model_path="pose.task",
                hand_landmarker_model_path="hand.task",
                gesture_model_path="gesture.task",
            )
        )
        pipeline._runtime = _StubRuntime(
            builtin_results=[
                _GestureResult(
                    gestures=[[ _Category("Thumb_Up", 0.92) ]],
                    hand_landmarks=[[ _Landmark(0.42), _Landmark(0.44) ]],
                )
            ]
        )

        observation = pipeline.observe(frame_rgb="frame", observed_at=10.0)

        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.THUMBS_UP)
        self.assertAlmostEqual(observation.fine_hand_gesture_confidence or 0.0, 0.92, places=3)
        self.assertEqual(observation.gesture_event, AICameraGestureEvent.NONE)
        self.assertEqual(observation.hand_count, 1)

    def test_observe_allows_custom_ok_sign_to_win_over_generic_open_palm(self) -> None:
        pipeline = LiveGesturePipeline(
            config=MediaPipeVisionConfig(
                pose_model_path="pose.task",
                hand_landmarker_model_path="hand.task",
                gesture_model_path="gesture.task",
                custom_gesture_model_path="custom.task",
            )
        )
        pipeline._runtime = _StubRuntime(
            builtin_results=[
                _GestureResult(
                    gestures=[[ _Category("Open_Palm", 0.64) ]],
                    hand_landmarks=[[ _Landmark(0.55), _Landmark(0.59) ]],
                )
            ],
            custom_results=[
                _GestureResult(
                    gestures=[[ _Category("ok_sign", 0.91) ]],
                    hand_landmarks=[[ _Landmark(0.55), _Landmark(0.59) ]],
                )
            ],
        )

        observation = pipeline.observe(frame_rgb="frame", observed_at=4.0)

        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.OK_SIGN)
        self.assertAlmostEqual(observation.fine_hand_gesture_confidence or 0.0, 0.91, places=3)

    def test_observe_detects_wave_from_open_palm_motion_sequence(self) -> None:
        pipeline = LiveGesturePipeline(
            config=MediaPipeVisionConfig(
                pose_model_path="pose.task",
                hand_landmarker_model_path="hand.task",
                gesture_model_path="gesture.task",
            )
        )
        pipeline._runtime = _StubRuntime(
            builtin_results=[
                _GestureResult(
                    gestures=[[ _Category("Open_Palm", 0.90) ]],
                    hand_landmarks=[[ _Landmark(0.26), _Landmark(0.28) ]],
                ),
                _GestureResult(
                    gestures=[[ _Category("Open_Palm", 0.92) ]],
                    hand_landmarks=[[ _Landmark(0.48), _Landmark(0.50) ]],
                ),
                _GestureResult(
                    gestures=[[ _Category("Open_Palm", 0.91) ]],
                    hand_landmarks=[[ _Landmark(0.31), _Landmark(0.33) ]],
                ),
            ]
        )

        pipeline.observe(frame_rgb="frame", observed_at=1.0)
        pipeline.observe(frame_rgb="frame", observed_at=1.2)
        observation = pipeline.observe(frame_rgb="frame", observed_at=1.4)

        self.assertEqual(observation.gesture_event, AICameraGestureEvent.WAVE)
        self.assertGreater(observation.gesture_confidence or 0.0, 0.6)


if __name__ == "__main__":
    unittest.main()
