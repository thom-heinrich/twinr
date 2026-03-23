from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.camera_ai.config import MediaPipeVisionConfig
from twinr.hardware.camera_ai.live_gesture_pipeline import LiveGesturePipeline, _crop_hand_box
from twinr.hardware.camera_ai.models import AICameraBox, AICameraFineHandGesture, AICameraGestureEvent


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


class _FakeFrame:
    def __init__(self, height: int, width: int) -> None:
        self.shape = (height, width, 3)

    def __getitem__(self, key):
        row_slice, col_slice = key
        start_row = int(row_slice.start or 0)
        stop_row = int(row_slice.stop or self.shape[0])
        start_col = int(col_slice.start or 0)
        stop_col = int(col_slice.stop or self.shape[1])
        return _FakeFrame(
            max(1, stop_row - start_row),
            max(1, stop_col - start_col),
        )


class _StubRecognizer:
    def __init__(self, callback, results) -> None:
        self._callback = callback
        self._results = list(results)

    def _next(self):
        return self._results.pop(0) if len(self._results) > 1 else self._results[0]

    def recognize_async(self, image, timestamp_ms: int) -> None:
        result = self._next()
        self._callback(result, image, timestamp_ms)

    def recognize(self, image):
        return self._next()


class _StubRuntime:
    def __init__(self, *, builtin_results, custom_results=None, roi_builtin_results=None, roi_custom_results=None) -> None:
        self._builtin_results = list(builtin_results)
        self._custom_results = list(custom_results or [_GestureResult(gestures=[], hand_landmarks=[])])
        self._roi_builtin_results = list(roi_builtin_results or [])
        self._roi_custom_results = list(roi_custom_results or [])
        self._builtin = None
        self._custom = None
        self._roi_builtin = None
        self._roi_custom = None
        self.live_num_hands_override = None

    def load_runtime(self):
        return {}

    def build_image(self, runtime, *, frame_rgb):
        return frame_rgb

    def timestamp_ms(self, observed_at: float) -> int:
        return int(round(observed_at * 1000.0))

    def ensure_live_gesture_recognizer(self, runtime, *, result_callback, num_hands_override=None):
        self.live_num_hands_override = num_hands_override
        if self._builtin is None:
            self._builtin = _StubRecognizer(result_callback, self._builtin_results)
        return self._builtin

    def ensure_live_custom_gesture_recognizer(self, runtime, *, result_callback):
        if self._custom is None:
            self._custom = _StubRecognizer(result_callback, self._custom_results)
        return self._custom

    def ensure_roi_gesture_recognizer(self, runtime):
        if self._roi_builtin is None:
            self._roi_builtin = _StubRecognizer(None, self._roi_builtin_results)
        return self._roi_builtin

    def ensure_custom_roi_gesture_recognizer(self, runtime):
        if self._roi_custom is None:
            self._roi_custom = _StubRecognizer(None, self._roi_custom_results)
        return self._roi_custom

    def close(self) -> None:
        return None


class _StubHandLandmarkWorker:
    def __init__(self, result) -> None:
        if isinstance(result, (list, tuple)):
            self._results = list(result)
        else:
            self._results = [result]
        self._full_frame_results = list(self._results)

    def analyze(self, **kwargs):
        return self._results.pop(0) if len(self._results) > 1 else self._results[0]

    def analyze_full_frame(self, **kwargs):
        return (
            self._full_frame_results.pop(0)
            if len(self._full_frame_results) > 1
            else self._full_frame_results[0]
        )

    def close(self) -> None:
        return None


class _RecordingHandLandmarkWorker(_StubHandLandmarkWorker):
    def __init__(self, result) -> None:
        super().__init__(result)
        self.calls = []
        self.full_frame_calls = []

    def analyze(self, **kwargs):
        self.calls.append(dict(kwargs))
        return super().analyze(**kwargs)

    def analyze_full_frame(self, **kwargs):
        self.full_frame_calls.append(dict(kwargs))
        return super().analyze_full_frame(**kwargs)


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

    def test_observe_uses_live_stream_custom_result_for_custom_symbol(self) -> None:
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
                    gestures=[[ _Category("none", 0.95) ]],
                    hand_landmarks=[[ _Landmark(0.42), _Landmark(0.44) ]],
                )
            ],
            custom_results=[
                _GestureResult(
                    gestures=[[ _Category("ok_sign", 0.91) ]],
                    hand_landmarks=[[ _Landmark(0.43), _Landmark(0.46) ]],
                )
            ],
        )

        observation = pipeline.observe(frame_rgb="frame", observed_at=10.0)

        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.OK_SIGN)
        self.assertAlmostEqual(observation.fine_hand_gesture_confidence or 0.0, 0.91, places=3)
        self.assertEqual(observation.hand_count, 1)
        debug_snapshot = pipeline.debug_snapshot()
        self.assertTrue(debug_snapshot["live_custom_enabled"])
        self.assertEqual(debug_snapshot["resolved_source"], "live_stream")

    def test_observe_recovers_custom_ok_sign_from_person_roi_when_live_path_stays_generic(self) -> None:
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
                    gestures=[[ _Category("none", 0.94) ]],
                    hand_landmarks=[[ _Landmark(0.55), _Landmark(0.59) ]],
                )
            ],
            roi_builtin_results=[
                _GestureResult(
                    gestures=[[ _Category("none", 0.25) ]],
                    hand_landmarks=[],
                )
            ],
            roi_custom_results=[
                _GestureResult(
                    gestures=[[ _Category("ok_sign", 0.91) ]],
                    hand_landmarks=[],
                )
            ],
        )
        pipeline._hand_landmark_worker = _StubHandLandmarkWorker(
            SimpleNamespace(
                detections=(SimpleNamespace(roi_frame_rgb="roi-frame"),),
                final_timestamp_ms=4000,
            )
        )

        observation = pipeline.observe(
            frame_rgb="frame",
            observed_at=4.0,
            primary_person_box=AICameraBox(top=0.2, left=0.2, bottom=0.8, right=0.8),
            person_count=1,
        )

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

    def test_observe_recovers_symbol_from_tight_hand_roi_when_live_label_is_none(self) -> None:
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
                    gestures=[[ _Category("none", 0.91) ]],
                    hand_landmarks=[[ _Landmark(0.60, 0.40), _Landmark(0.66, 0.48), _Landmark(0.58, 0.52) ]],
                )
            ],
            roi_builtin_results=[
                _GestureResult(
                    gestures=[[ _Category("Victory", 0.88) ]],
                    hand_landmarks=[],
                )
            ],
        )

        observation = pipeline.observe(frame_rgb="frame", observed_at=7.0)

        self.assertEqual(observation.hand_count, 1)
        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.PEACE_SIGN)
        self.assertAlmostEqual(observation.fine_hand_gesture_confidence or 0.0, 0.88, places=3)

    def test_observe_recovers_symbol_from_primary_person_roi_when_live_path_stays_none(self) -> None:
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
                    gestures=[[ _Category("none", 0.94) ]],
                    hand_landmarks=[],
                )
            ],
            roi_builtin_results=[
                _GestureResult(
                    gestures=[[ _Category("Victory", 0.91) ]],
                    hand_landmarks=[],
                )
            ],
        )
        pipeline._hand_landmark_worker = _StubHandLandmarkWorker(
            SimpleNamespace(
                detections=(SimpleNamespace(roi_frame_rgb="roi-frame"),),
                final_timestamp_ms=7000,
            )
        )

        observation = pipeline.observe(
            frame_rgb="frame",
            observed_at=7.0,
            primary_person_box=AICameraBox(top=0.2, left=0.2, bottom=0.8, right=0.8),
            person_count=1,
        )

        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.PEACE_SIGN)
        debug_snapshot = pipeline.debug_snapshot()
        self.assertEqual(debug_snapshot["resolved_source"], "person_roi")
        self.assertEqual(debug_snapshot["person_roi_detection_count"], 1)

    def test_observe_recovers_symbol_from_secondary_visible_person_roi_when_primary_box_stays_empty(self) -> None:
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
                    gestures=[[_Category("none", 0.94)]],
                    hand_landmarks=[],
                )
            ],
            roi_builtin_results=[
                _GestureResult(
                    gestures=[[_Category("Pointing_Up", 0.89)]],
                    hand_landmarks=[],
                )
            ],
        )
        pipeline._hand_landmark_worker = _StubHandLandmarkWorker(
            [
                SimpleNamespace(
                    detections=(),
                    final_timestamp_ms=8000,
                ),
                SimpleNamespace(
                    detections=(SimpleNamespace(roi_frame_rgb="roi-frame-secondary"),),
                    final_timestamp_ms=8001,
                ),
            ]
        )

        observation = pipeline.observe(
            frame_rgb="frame",
            observed_at=8.0,
            primary_person_box=AICameraBox(top=0.2, left=0.1, bottom=0.9, right=0.45),
            visible_person_boxes=(
                AICameraBox(top=0.2, left=0.1, bottom=0.9, right=0.45),
                AICameraBox(top=0.18, left=0.55, bottom=0.88, right=0.92),
            ),
            person_count=2,
        )

        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.POINTING)
        debug_snapshot = pipeline.debug_snapshot()
        self.assertEqual(debug_snapshot["resolved_source"], "visible_person_roi")
        self.assertEqual(debug_snapshot["person_roi_candidate_count"], 2)
        self.assertEqual(debug_snapshot["person_roi_match_index"], 1)
        self.assertEqual(debug_snapshot["person_roi_detection_count"], 1)

    def test_observe_forwards_pose_hints_only_to_matching_primary_person_roi(self) -> None:
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
                    gestures=[[_Category("none", 0.94)]],
                    hand_landmarks=[],
                )
            ],
            roi_builtin_results=[
                _GestureResult(
                    gestures=[[_Category("Pointing_Up", 0.89)]],
                    hand_landmarks=[],
                )
            ],
        )
        worker = _RecordingHandLandmarkWorker(
            [
                SimpleNamespace(
                    detections=(SimpleNamespace(roi_frame_rgb="roi-frame-primary"),),
                    final_timestamp_ms=9000,
                ),
                SimpleNamespace(
                    detections=(),
                    final_timestamp_ms=9001,
                ),
            ]
        )
        pipeline._hand_landmark_worker = worker
        primary_box = AICameraBox(top=0.2, left=0.1, bottom=0.9, right=0.45)
        secondary_box = AICameraBox(top=0.18, left=0.55, bottom=0.88, right=0.92)
        sparse_keypoints = {
            9: (0.18, 0.36, 0.93),
            10: (0.36, 0.35, 0.94),
        }

        observation = pipeline.observe(
            frame_rgb="frame",
            observed_at=9.0,
            primary_person_box=primary_box,
            visible_person_boxes=(primary_box, secondary_box),
            person_count=2,
            sparse_keypoints=sparse_keypoints,
        )

        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.POINTING)
        self.assertEqual(worker.calls[0]["sparse_keypoints"], sparse_keypoints)
        self.assertEqual(worker.calls[1]["sparse_keypoints"], {})
        self.assertEqual(pipeline.debug_snapshot()["person_roi_pose_hint_match_index"], 0)

    def test_observe_reuses_recent_primary_person_box_when_detection_flickers_off(self) -> None:
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
                    gestures=[[ _Category("none", 0.94) ]],
                    hand_landmarks=[],
                ),
                _GestureResult(
                    gestures=[[ _Category("none", 0.94) ]],
                    hand_landmarks=[],
                ),
            ],
            roi_builtin_results=[
                _GestureResult(
                    gestures=[[ _Category("Victory", 0.90) ]],
                    hand_landmarks=[],
                )
            ],
        )
        pipeline._hand_landmark_worker = _StubHandLandmarkWorker(
            SimpleNamespace(
                detections=(SimpleNamespace(roi_frame_rgb="roi-frame"),),
                final_timestamp_ms=1000,
            )
        )

        pipeline.observe(
            frame_rgb="frame",
            observed_at=1.0,
            primary_person_box=AICameraBox(top=0.2, left=0.2, bottom=0.8, right=0.8),
            person_count=1,
        )
        observation = pipeline.observe(
            frame_rgb="frame",
            observed_at=1.2,
            primary_person_box=None,
            person_count=0,
        )

        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.PEACE_SIGN)
        self.assertEqual(pipeline.debug_snapshot()["resolved_source"], "recent_person_roi")

    def test_observe_reuses_recent_visible_person_boxes_when_live_visible_boxes_flicker_off(self) -> None:
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
                    gestures=[[_Category("none", 0.94)]],
                    hand_landmarks=[],
                ),
                _GestureResult(
                    gestures=[[_Category("none", 0.94)]],
                    hand_landmarks=[],
                ),
            ],
            roi_builtin_results=[
                _GestureResult(
                    gestures=[[_Category("Victory", 0.90)]],
                    hand_landmarks=[],
                )
            ],
        )
        pipeline._hand_landmark_worker = _StubHandLandmarkWorker(
            [
                SimpleNamespace(
                    detections=(),
                    final_timestamp_ms=1000,
                ),
                SimpleNamespace(
                    detections=(SimpleNamespace(roi_frame_rgb="roi-frame-secondary"),),
                    final_timestamp_ms=1001,
                ),
                SimpleNamespace(
                    detections=(),
                    final_timestamp_ms=1200,
                ),
                SimpleNamespace(
                    detections=(SimpleNamespace(roi_frame_rgb="roi-frame-secondary"),),
                    final_timestamp_ms=1201,
                ),
            ]
        )

        pipeline.observe(
            frame_rgb="frame",
            observed_at=1.0,
            primary_person_box=AICameraBox(top=0.2, left=0.1, bottom=0.9, right=0.45),
            visible_person_boxes=(
                AICameraBox(top=0.2, left=0.1, bottom=0.9, right=0.45),
                AICameraBox(top=0.18, left=0.55, bottom=0.88, right=0.92),
            ),
            person_count=2,
        )
        observation = pipeline.observe(
            frame_rgb="frame",
            observed_at=1.2,
            primary_person_box=None,
            visible_person_boxes=(),
            person_count=0,
        )

        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.PEACE_SIGN)
        self.assertEqual(pipeline.debug_snapshot()["resolved_source"], "recent_visible_person_roi")

    def test_observe_reuses_recent_live_hand_box_when_current_live_result_loses_boxes(self) -> None:
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
                    gestures=[[ _Category("none", 0.91) ]],
                    hand_landmarks=[[ _Landmark(0.60, 0.40), _Landmark(0.66, 0.48), _Landmark(0.58, 0.52) ]],
                ),
                _GestureResult(
                    gestures=[[ _Category("none", 0.91) ]],
                    hand_landmarks=[],
                ),
            ],
            roi_builtin_results=[
                _GestureResult(
                    gestures=[[ _Category("Pointing_Up", 0.87) ]],
                    hand_landmarks=[],
                )
            ],
        )

        pipeline.observe(frame_rgb="frame", observed_at=2.0)
        observation = pipeline.observe(frame_rgb="frame", observed_at=2.2)

        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.POINTING)
        self.assertEqual(pipeline.debug_snapshot()["resolved_source"], "recent_live_hand_roi")

    def test_observe_recovers_symbol_from_full_frame_hand_landmarks_when_no_person_or_live_hand_box_exists(self) -> None:
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
                    gestures=[[_Category("none", 0.96)]],
                    hand_landmarks=[],
                )
            ],
            roi_builtin_results=[
                _GestureResult(
                    gestures=[[_Category("none", 0.20)]],
                    hand_landmarks=[],
                )
            ],
            roi_custom_results=[
                _GestureResult(
                    gestures=[[_Category("middle_finger", 0.93)]],
                    hand_landmarks=[],
                )
            ],
        )
        worker = _RecordingHandLandmarkWorker(
            SimpleNamespace(
                detections=(SimpleNamespace(roi_frame_rgb="full-frame-hand"),),
                final_timestamp_ms=11000,
            )
        )
        pipeline._hand_landmark_worker = worker

        observation = pipeline.observe(
            frame_rgb="frame",
            observed_at=11.0,
            primary_person_box=None,
            visible_person_boxes=(),
            person_count=0,
        )

        self.assertEqual(observation.hand_count, 1)
        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.MIDDLE_FINGER)
        self.assertEqual(len(worker.calls), 0)
        self.assertEqual(len(worker.full_frame_calls), 1)
        debug_snapshot = pipeline.debug_snapshot()
        self.assertEqual(debug_snapshot["resolved_source"], "full_frame_hand_roi")
        self.assertEqual(debug_snapshot["full_frame_hand_detection_count"], 1)
        self.assertEqual(debug_snapshot["full_frame_hand_attempt_reason"], "no_person_roi_or_live_hand_box")

    def test_observe_recovers_symbol_from_full_frame_hand_landmarks_when_visible_person_roi_finds_no_hand(self) -> None:
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
                    gestures=[[_Category("none", 0.96)]],
                    hand_landmarks=[],
                )
            ],
            roi_builtin_results=[
                _GestureResult(
                    gestures=[[_Category("none", 0.20)]],
                    hand_landmarks=[],
                ),
            ],
            roi_custom_results=[
                _GestureResult(
                    gestures=[[_Category("ok_sign", 0.92)]],
                    hand_landmarks=[],
                ),
            ],
        )
        worker = _RecordingHandLandmarkWorker(
            [
                SimpleNamespace(
                    detections=(),
                    final_timestamp_ms=12000,
                ),
            ]
        )
        worker._full_frame_results = [
            SimpleNamespace(
                detections=(SimpleNamespace(roi_frame_rgb="full-frame-hand"),),
                final_timestamp_ms=12001,
            )
        ]
        pipeline._hand_landmark_worker = worker

        observation = pipeline.observe(
            frame_rgb="frame",
            observed_at=12.0,
            primary_person_box=AICameraBox(top=0.2, left=0.2, bottom=0.8, right=0.8),
            visible_person_boxes=(AICameraBox(top=0.2, left=0.2, bottom=0.8, right=0.8),),
            person_count=1,
        )

        self.assertEqual(observation.hand_count, 1)
        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.OK_SIGN)
        self.assertEqual(len(worker.calls), 1)
        self.assertEqual(len(worker.full_frame_calls), 1)
        debug_snapshot = pipeline.debug_snapshot()
        self.assertEqual(debug_snapshot["person_roi_detection_count"], 0)
        self.assertEqual(debug_snapshot["resolved_source"], "full_frame_hand_roi")
        self.assertEqual(
            debug_snapshot["full_frame_hand_attempt_reason"],
            "visible_person_roi_without_hand_detection",
        )

    def test_observe_recovers_symbol_from_full_frame_hand_landmarks_when_live_hand_roi_finds_no_symbol(self) -> None:
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
                    gestures=[[_Category("none", 0.94)]],
                    hand_landmarks=[[ _Landmark(0.86, 0.28), _Landmark(0.91, 0.36), _Landmark(0.83, 0.42) ]],
                )
            ],
            roi_builtin_results=[
                _GestureResult(
                    gestures=[[_Category("none", 0.30)]],
                    hand_landmarks=[],
                ),
                _GestureResult(
                    gestures=[[_Category("Thumb_Up", 0.89)]],
                    hand_landmarks=[],
                ),
            ],
        )
        worker = _RecordingHandLandmarkWorker(
            SimpleNamespace(
                detections=(SimpleNamespace(roi_frame_rgb="full-frame-hand"),),
                final_timestamp_ms=13001,
            )
        )
        pipeline._hand_landmark_worker = worker

        observation = pipeline.observe(
            frame_rgb="frame",
            observed_at=13.0,
            primary_person_box=None,
            visible_person_boxes=(),
            person_count=0,
        )

        self.assertEqual(observation.hand_count, 1)
        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.THUMBS_UP)
        self.assertEqual(len(worker.calls), 0)
        self.assertEqual(len(worker.full_frame_calls), 1)
        debug_snapshot = pipeline.debug_snapshot()
        self.assertEqual(debug_snapshot["live_roi_hand_box_count"], 1)
        self.assertEqual(debug_snapshot["live_roi_combined_gesture"], "none")
        self.assertEqual(debug_snapshot["resolved_source"], "full_frame_hand_roi")
        self.assertEqual(
            debug_snapshot["full_frame_hand_attempt_reason"],
            "live_hand_roi_without_symbol",
        )

    def test_crop_hand_box_expands_to_context_rich_square_window(self) -> None:
        crop = _crop_hand_box(
            _FakeFrame(100, 120),
            (0.45, 0.48, 0.50, 0.52),
        )

        self.assertEqual(crop.shape, (34, 40, 3))


if __name__ == "__main__":
    unittest.main()
