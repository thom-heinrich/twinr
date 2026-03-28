import json
from pathlib import Path
import threading
from types import SimpleNamespace
import sys
from tempfile import TemporaryDirectory
import time
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.workflows.forensics import WorkflowForensics, bind_workflow_forensics
from twinr.hardware.camera_ai.config import MediaPipeVisionConfig
from twinr.hardware.camera_ai.live_gesture_pipeline import (
    LiveGestureObservePolicy,
    LiveGesturePipeline,
    _crop_hand_box,
)
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
    def __init__(self, callback, results, *, async_delay_s: float = 0.0) -> None:
        self._callback = callback
        self._results = list(results)
        self._async_delay_s = max(0.0, float(async_delay_s))
        self._threads: list[threading.Thread] = []

    def _next(self):
        return self._results.pop(0) if len(self._results) > 1 else self._results[0]

    def recognize_async(self, image, timestamp_ms: int) -> None:
        result = self._next()
        if self._callback is None:
            return
        if self._async_delay_s <= 0.0:
            self._callback(result, image, timestamp_ms)
            return

        def _emit() -> None:
            time.sleep(self._async_delay_s)
            self._callback(result, image, timestamp_ms)

        thread = threading.Thread(target=_emit, daemon=True)
        self._threads.append(thread)
        thread.start()

    def recognize(self, image):
        return self._next()

    def wait(self) -> None:
        for thread in list(self._threads):
            thread.join(timeout=1.0)
        self._threads = []


class _StubRuntime:
    def __init__(
        self,
        *,
        builtin_results,
        custom_results=None,
        roi_builtin_results=None,
        roi_custom_results=None,
        builtin_async_delay_s: float = 0.0,
        custom_async_delay_s: float = 0.0,
    ) -> None:
        self._builtin_results = list(builtin_results)
        self._custom_results = list(custom_results or [_GestureResult(gestures=[], hand_landmarks=[])])
        self._roi_builtin_results = list(roi_builtin_results or [])
        self._roi_custom_results = list(roi_custom_results or [])
        self._builtin_async_delay_s = builtin_async_delay_s
        self._custom_async_delay_s = custom_async_delay_s
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
            self._builtin = _StubRecognizer(
                result_callback,
                self._builtin_results,
                async_delay_s=self._builtin_async_delay_s,
            )
        return self._builtin

    def ensure_live_custom_gesture_recognizer(self, runtime, *, result_callback):
        if self._custom is None:
            self._custom = _StubRecognizer(
                result_callback,
                self._custom_results,
                async_delay_s=self._custom_async_delay_s,
            )
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
        for recognizer in (self._builtin, self._custom):
            if recognizer is not None:
                recognizer.wait()
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
        self.calls: list[dict[str, object]] = []
        self.full_frame_calls: list[dict[str, object]] = []

    def analyze(self, **kwargs):
        self.calls.append(dict(kwargs))
        return super().analyze(**kwargs)

    def analyze_full_frame(self, **kwargs):
        self.full_frame_calls.append(dict(kwargs))
        return super().analyze_full_frame(**kwargs)


class _ImageSelectiveRoiRuntime(_StubRuntime):
    def __init__(
        self,
        *,
        builtin_results,
        image_to_result,
        custom_results=None,
        roi_custom_results=None,
    ) -> None:
        super().__init__(
            builtin_results=builtin_results,
            custom_results=custom_results,
            roi_custom_results=roi_custom_results,
        )
        self.image_to_result = dict(image_to_result)
        self.roi_inputs: list[object] = []

    def ensure_roi_gesture_recognizer(self, runtime):
        outer = self

        class _Recognizer:
            def recognize(self, image):
                outer.roi_inputs.append(image)
                return outer.image_to_result.get(
                    image,
                    _GestureResult(gestures=[], hand_landmarks=[]),
                )

        if self._roi_builtin is None:
            self._roi_builtin = _Recognizer()
        return self._roi_builtin


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

    def test_observe_ignores_live_stream_custom_symbol_outside_three_gesture_path(self) -> None:
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
                    hand_landmarks=[],
                )
            ],
            custom_results=[
                _GestureResult(
                    gestures=[[ _Category("ok_sign", 0.91) ]],
                    hand_landmarks=[[ _Landmark(0.43), _Landmark(0.46) ]],
                )
            ],
            roi_builtin_results=[
                _GestureResult(
                    gestures=[[_Category("none", 0.95)]],
                    hand_landmarks=[],
                )
            ],
            roi_custom_results=[
                _GestureResult(
                    gestures=[[_Category("ok_sign", 0.91)]],
                    hand_landmarks=[],
                )
            ],
        )
        pipeline._hand_landmark_worker = _StubHandLandmarkWorker(
            SimpleNamespace(
                detections=(),
                final_timestamp_ms=10000,
            )
        )

        observation = pipeline.observe(frame_rgb="frame", observed_at=10.0)

        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.NONE)
        debug_snapshot = pipeline.debug_snapshot()
        self.assertTrue(debug_snapshot["live_custom_enabled"])
        self.assertEqual(debug_snapshot["resolved_source"], "none")
        self.assertIsNotNone(pipeline._runtime._custom)

    def test_observe_ignores_custom_person_roi_symbol_outside_three_gesture_path(self) -> None:
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

        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.NONE)
        debug_snapshot = pipeline.debug_snapshot()
        self.assertTrue(debug_snapshot["live_custom_enabled"])
        self.assertEqual(debug_snapshot["resolved_source"], "none")
        self.assertIsNotNone(pipeline._runtime._roi_custom)

    def test_observe_prefers_custom_three_gesture_live_result_over_conflicting_builtin(self) -> None:
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
                    gestures=[[_Category("thumbs_down", 0.61)]],
                    hand_landmarks=[[_Landmark(0.45), _Landmark(0.49)]],
                )
            ],
            custom_results=[
                _GestureResult(
                    gestures=[[_Category("peace_sign", 0.88)]],
                    hand_landmarks=[[_Landmark(0.45), _Landmark(0.49)]],
                )
            ],
        )

        observation = pipeline.observe(frame_rgb="frame", observed_at=10.0)

        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.PEACE_SIGN)
        self.assertAlmostEqual(observation.fine_hand_gesture_confidence or 0.0, 0.88, places=3)
        debug_snapshot = pipeline.debug_snapshot()
        self.assertTrue(debug_snapshot["live_custom_enabled"])

    def test_observe_waits_briefly_for_delayed_same_frame_custom_live_result(self) -> None:
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
                    gestures=[[_Category("none", 0.97)]],
                    hand_landmarks=[],
                )
            ],
            custom_results=[
                _GestureResult(
                    gestures=[[_Category("peace_sign", 0.91)]],
                    hand_landmarks=[[_Landmark(0.41), _Landmark(0.45)]],
                )
            ],
            custom_async_delay_s=0.02,
        )

        try:
            observation = pipeline.observe(frame_rgb="frame", observed_at=10.0)
            debug_snapshot = pipeline.debug_snapshot()
        finally:
            pipeline.close()

        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.PEACE_SIGN)
        self.assertAlmostEqual(observation.fine_hand_gesture_confidence or 0.0, 0.91, places=3)
        self.assertEqual(debug_snapshot["resolved_source"], "live_stream")
        self.assertTrue(debug_snapshot["current_live_builtin_ready"])
        self.assertTrue(debug_snapshot["current_live_custom_ready"])
        self.assertGreater(debug_snapshot["current_live_result_wait_s"], 0.0)

    def test_observe_rejects_low_confidence_live_stream_custom_symbol(self) -> None:
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
                    gestures=[[_Category("none", 0.97)]],
                    hand_landmarks=[[_Landmark(0.41), _Landmark(0.45)]],
                )
            ],
            custom_results=[
                _GestureResult(
                    gestures=[[_Category("thumbs_up", 0.55)]],
                    hand_landmarks=[[_Landmark(0.41), _Landmark(0.45)]],
                )
            ],
            roi_builtin_results=[
                _GestureResult(
                    gestures=[[_Category("none", 0.97)]],
                    hand_landmarks=[],
                )
            ],
            roi_custom_results=[
                _GestureResult(
                    gestures=[[_Category("none", 0.97)]],
                    hand_landmarks=[],
                )
            ],
        )
        pipeline._hand_landmark_worker = _StubHandLandmarkWorker(
            SimpleNamespace(
                detections=(),
                final_timestamp_ms=10000,
            )
        )

        observation = pipeline.observe(frame_rgb="frame", observed_at=10.0)

        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.NONE)
        debug_snapshot = pipeline.debug_snapshot()
        self.assertEqual(debug_snapshot["resolved_source"], "none")
        self.assertEqual(debug_snapshot["live_custom_gesture"], "none")

    def test_observe_writes_person_roi_decisions_into_bound_forensics_trace(self) -> None:
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
                    gestures=[[_Category("none", 0.94)]],
                    hand_landmarks=[[_Landmark(0.55), _Landmark(0.59)]],
                )
            ],
            roi_builtin_results=[
                _GestureResult(
                    gestures=[[_Category("none", 0.25)]],
                    hand_landmarks=[],
                )
            ],
            roi_custom_results=[
                _GestureResult(
                    gestures=[[_Category("ok_sign", 0.91)]],
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

        with TemporaryDirectory() as temp_dir:
            trace_dir = Path(temp_dir) / "state" / "forensics" / "workflow"
            tracer = WorkflowForensics(
                project_root=Path(temp_dir),
                service="live-gesture-test",
                enabled=True,
                mode="forensic",
                base_dir=trace_dir,
            )
            with bind_workflow_forensics(tracer, trace_id="gesture-trace-test"):
                observation = pipeline.observe(
                    frame_rgb="frame",
                    observed_at=4.0,
                    primary_person_box=AICameraBox(top=0.2, left=0.2, bottom=0.8, right=0.8),
                    person_count=1,
                )
            tracer.close()

            self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.NONE)
            run_id = (trace_dir / "LATEST").read_text(encoding="utf-8").strip()
            records = [
                json.loads(line)
                for line in (trace_dir / run_id / "run.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            msgs = {record["msg"] for record in records}
            self.assertIn("live_gesture_pipeline_person_roi_selection", msgs)
            self.assertIn("live_gesture_pipeline_resolution", msgs)

    def test_observe_does_not_emit_wave_from_open_palm_motion_in_three_gesture_path(self) -> None:
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
            ],
            roi_builtin_results=[
                _GestureResult(
                    gestures=[[ _Category("none", 0.95) ]],
                    hand_landmarks=[],
                )
            ],
        )
        pipeline._hand_landmark_worker = _StubHandLandmarkWorker(
            SimpleNamespace(
                detections=(),
                final_timestamp_ms=2000,
            )
        )

        pipeline.observe(frame_rgb="frame", observed_at=1.0)
        pipeline.observe(frame_rgb="frame", observed_at=1.2)
        observation = pipeline.observe(frame_rgb="frame", observed_at=1.4)

        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.NONE)
        self.assertEqual(observation.gesture_event, AICameraGestureEvent.NONE)

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

    def test_observe_user_facing_fast_policy_uses_current_live_hand_roi_only(self) -> None:
        pipeline = LiveGesturePipeline(
            config=MediaPipeVisionConfig(
                pose_model_path="pose.task",
                hand_landmarker_model_path="hand.task",
                gesture_model_path="gesture.task",
            )
        )
        pipeline._runtime = _ImageSelectiveRoiRuntime(
            builtin_results=[
                _GestureResult(
                    gestures=[[_Category("none", 0.95)]],
                    hand_landmarks=[[_Landmark(0.42, 0.42), _Landmark(0.47, 0.57)]],
                )
            ],
            image_to_result={
                "frame": _GestureResult(
                    gestures=[[_Category("Thumb_Up", 0.83)]],
                    hand_landmarks=[],
                )
            },
        )
        worker = _RecordingHandLandmarkWorker(
            SimpleNamespace(
                detections=(SimpleNamespace(roi_frame_rgb="roi-frame"),),
                final_timestamp_ms=7000,
            )
        )
        worker._full_frame_results = [
            SimpleNamespace(
                detections=(SimpleNamespace(roi_frame_rgb="full-frame-hand"),),
                final_timestamp_ms=7001,
            )
        ]
        pipeline._hand_landmark_worker = worker

        observation = pipeline.observe(
            frame_rgb="frame",
            observed_at=7.0,
            observe_policy=LiveGestureObservePolicy.user_facing_fast(),
            primary_person_box=AICameraBox(top=0.2, left=0.2, bottom=0.8, right=0.8),
            visible_person_boxes=(AICameraBox(top=0.2, left=0.2, bottom=0.8, right=0.8),),
            person_count=1,
        )

        debug_snapshot = pipeline.debug_snapshot()
        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.THUMBS_UP)
        self.assertEqual(debug_snapshot["resolved_source"], "live_hand_roi")
        self.assertEqual(debug_snapshot["observe_policy"], "user_facing_fast")
        self.assertEqual(
            debug_snapshot["person_roi_block_reason"],
            "observe_policy_disallows_person_roi_recovery",
        )
        self.assertIsNone(debug_snapshot["full_frame_hand_attempt_reason"])
        self.assertEqual(len(worker.calls), 0)
        self.assertEqual(len(worker.full_frame_calls), 0)
        self.assertIsNone(pipeline._runtime._custom)

    def test_observe_user_facing_fast_policy_keeps_live_custom_peace_sign(self) -> None:
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
                    gestures=[[_Category("none", 0.97)]],
                    hand_landmarks=[[_Landmark(0.46, 0.42), _Landmark(0.53, 0.58)]],
                )
            ],
            custom_results=[
                _GestureResult(
                    gestures=[[_Category("peace_sign", 0.91)]],
                    hand_landmarks=[[_Landmark(0.46, 0.42), _Landmark(0.53, 0.58)]],
                )
            ],
        )

        observation = pipeline.observe(
            frame_rgb="frame",
            observed_at=7.0,
            observe_policy=LiveGestureObservePolicy.user_facing_fast(),
        )

        debug_snapshot = pipeline.debug_snapshot()
        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.PEACE_SIGN)
        self.assertAlmostEqual(observation.fine_hand_gesture_confidence or 0.0, 0.91, places=3)
        self.assertEqual(observation.hand_count, 1)
        self.assertEqual(debug_snapshot["resolved_source"], "live_stream")
        self.assertTrue(debug_snapshot["live_custom_enabled"])
        self.assertTrue(debug_snapshot["observe_policy_enable_custom_live"])

    def test_observe_user_facing_fast_policy_ignores_recent_hand_box_reuse(self) -> None:
        pipeline = LiveGesturePipeline(
            config=MediaPipeVisionConfig(
                pose_model_path="pose.task",
                hand_landmarker_model_path="hand.task",
                gesture_model_path="gesture.task",
            )
        )
        pipeline._runtime = _ImageSelectiveRoiRuntime(
            builtin_results=[
                _GestureResult(
                    gestures=[[_Category("none", 0.95)]],
                    hand_landmarks=[[_Landmark(0.41, 0.41), _Landmark(0.49, 0.59)]],
                ),
                _GestureResult(
                    gestures=[[_Category("none", 0.95)]],
                    hand_landmarks=[],
                ),
            ],
            image_to_result={
                "frame": _GestureResult(
                    gestures=[[_Category("thumbs_down", 0.88)]],
                    hand_landmarks=[],
                )
            },
        )

        pipeline.observe(frame_rgb="frame", observed_at=7.0)
        observation = pipeline.observe(
            frame_rgb="frame",
            observed_at=7.2,
            observe_policy=LiveGestureObservePolicy.user_facing_fast(),
            primary_person_box=AICameraBox(top=0.2, left=0.2, bottom=0.8, right=0.8),
            visible_person_boxes=(AICameraBox(top=0.2, left=0.2, bottom=0.8, right=0.8),),
            person_count=1,
        )

        debug_snapshot = pipeline.debug_snapshot()
        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.NONE)
        self.assertEqual(debug_snapshot["resolved_source"], "none")
        self.assertEqual(debug_snapshot["effective_live_hand_box_count"], 0)
        self.assertEqual(debug_snapshot["live_hand_box_source"], "none")
        self.assertEqual(
            debug_snapshot["person_roi_block_reason"],
            "observe_policy_disallows_person_roi_recovery",
        )

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

    def test_observe_blocks_person_roi_when_fresh_live_results_confirm_no_hand(self) -> None:
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
                    gestures=[[_Category("none", 0.98)]],
                    hand_landmarks=[],
                )
            ],
            custom_results=[
                _GestureResult(
                    gestures=[[_Category("none", 0.98)]],
                    hand_landmarks=[],
                )
            ],
            roi_builtin_results=[
                _GestureResult(
                    gestures=[[_Category("none", 0.98)]],
                    hand_landmarks=[],
                )
            ],
            roi_custom_results=[
                _GestureResult(
                    gestures=[[_Category("thumbs_down", 0.91)]],
                    hand_landmarks=[],
                )
            ],
        )
        pipeline._hand_landmark_worker = _StubHandLandmarkWorker(
            SimpleNamespace(
                detections=(SimpleNamespace(roi_frame_rgb="roi-frame"),),
                final_timestamp_ms=12000,
            )
        )

        observation = pipeline.observe(
            frame_rgb="frame",
            observed_at=12.0,
            primary_person_box=AICameraBox(top=0.2, left=0.2, bottom=0.8, right=0.8),
            visible_person_boxes=(AICameraBox(top=0.2, left=0.2, bottom=0.8, right=0.8),),
            person_count=1,
        )

        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.NONE)
        self.assertIsNone(observation.fine_hand_gesture_confidence)
        debug_snapshot = pipeline.debug_snapshot()
        self.assertTrue(debug_snapshot["fresh_live_results_confirm_no_hand"])
        self.assertEqual(debug_snapshot["resolved_source"], "none")
        self.assertEqual(debug_snapshot["person_roi_block_reason"], "fresh_live_results_confirm_no_hand")

    def test_observe_blocks_person_roi_when_builtin_no_hand_is_current_but_custom_is_still_pending(self) -> None:
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
                    gestures=[[_Category("none", 0.98)]],
                    hand_landmarks=[],
                )
            ],
            custom_results=[
                _GestureResult(
                    gestures=[[_Category("none", 0.98)]],
                    hand_landmarks=[],
                )
            ],
            roi_builtin_results=[
                _GestureResult(
                    gestures=[[_Category("none", 0.98)]],
                    hand_landmarks=[],
                )
            ],
            roi_custom_results=[
                _GestureResult(
                    gestures=[[_Category("thumbs_down", 0.91)]],
                    hand_landmarks=[],
                )
            ],
            custom_async_delay_s=0.35,
        )
        pipeline._hand_landmark_worker = _StubHandLandmarkWorker(
            SimpleNamespace(
                detections=(SimpleNamespace(roi_frame_rgb="roi-frame"),),
                final_timestamp_ms=12000,
            )
        )

        try:
            observation = pipeline.observe(
                frame_rgb="frame",
                observed_at=12.0,
                primary_person_box=AICameraBox(top=0.2, left=0.2, bottom=0.8, right=0.8),
                visible_person_boxes=(AICameraBox(top=0.2, left=0.2, bottom=0.8, right=0.8),),
                person_count=1,
            )
            debug_snapshot = pipeline.debug_snapshot()
        finally:
            pipeline.close()

        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.NONE)
        self.assertIsNone(observation.fine_hand_gesture_confidence)
        self.assertTrue(debug_snapshot["current_live_builtin_ready"])
        self.assertFalse(debug_snapshot["current_live_custom_ready"])
        self.assertTrue(debug_snapshot["fresh_live_results_confirm_no_hand"])
        self.assertEqual(debug_snapshot["resolved_source"], "none")
        self.assertEqual(debug_snapshot["person_roi_block_reason"], "fresh_live_results_confirm_no_hand")

    def test_observe_blocks_full_frame_hand_roi_when_fresh_live_results_confirm_no_hand(self) -> None:
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
                    gestures=[[_Category("none", 0.98)]],
                    hand_landmarks=[],
                )
            ],
            custom_results=[
                _GestureResult(
                    gestures=[[_Category("none", 0.98)]],
                    hand_landmarks=[],
                )
            ],
            roi_builtin_results=[
                _GestureResult(
                    gestures=[[_Category("Victory", 0.92)]],
                    hand_landmarks=[],
                ),
            ],
            roi_custom_results=[
                _GestureResult(
                    gestures=[[_Category("none", 0.98)]],
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

        self.assertTrue(pipeline.debug_snapshot()["fresh_live_results_confirm_no_hand"])
        self.assertEqual(observation.hand_count, 0)
        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.NONE)
        self.assertEqual(len(worker.calls), 1)
        self.assertEqual(len(worker.full_frame_calls), 0)
        debug_snapshot = pipeline.debug_snapshot()
        self.assertEqual(debug_snapshot["person_roi_detection_count"], 0)
        self.assertEqual(debug_snapshot["resolved_source"], "none")
        self.assertEqual(
            debug_snapshot["full_frame_hand_attempt_reason"],
            "fresh_live_results_confirm_no_hand",
        )

    def test_observe_allows_person_roi_with_recent_live_hand_box_after_current_no_hand(self) -> None:
        pipeline = LiveGesturePipeline(
            config=MediaPipeVisionConfig(
                pose_model_path="pose.task",
                hand_landmarker_model_path="hand.task",
                gesture_model_path="gesture.task",
                custom_gesture_model_path="custom.task",
            )
        )
        pipeline._runtime = _ImageSelectiveRoiRuntime(
            builtin_results=[
                _GestureResult(
                    gestures=[[_Category("none", 0.95)]],
                    hand_landmarks=[[ _Landmark(0.60, 0.40), _Landmark(0.66, 0.48), _Landmark(0.58, 0.52) ]],
                ),
                _GestureResult(
                    gestures=[[_Category("none", 0.95)]],
                    hand_landmarks=[],
                ),
            ],
            custom_results=[
                _GestureResult(
                    gestures=[[_Category("none", 0.95)]],
                    hand_landmarks=[[ _Landmark(0.60, 0.40), _Landmark(0.66, 0.48), _Landmark(0.58, 0.52) ]],
                ),
                _GestureResult(
                    gestures=[[_Category("none", 0.95)]],
                    hand_landmarks=[],
                ),
            ],
            roi_custom_results=[
                _GestureResult(
                    gestures=[[_Category("none", 0.95)]],
                    hand_landmarks=[],
                ),
            ],
            image_to_result={
                "roi-frame": _GestureResult(
                    gestures=[[_Category("Thumb_Down", 0.88)]],
                    hand_landmarks=[],
                ),
            },
        )
        pipeline._hand_landmark_worker = _StubHandLandmarkWorker(
            SimpleNamespace(
                detections=(SimpleNamespace(roi_frame_rgb="roi-frame"),),
                final_timestamp_ms=2200,
            )
        )

        pipeline.observe(frame_rgb=_FakeFrame(240, 320), observed_at=2.0)
        observation = pipeline.observe(
            frame_rgb=_FakeFrame(240, 320),
            observed_at=2.2,
            primary_person_box=AICameraBox(top=0.2, left=0.2, bottom=0.85, right=0.8),
            visible_person_boxes=(AICameraBox(top=0.2, left=0.2, bottom=0.85, right=0.8),),
            person_count=1,
        )

        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.THUMBS_DOWN)
        self.assertEqual(observation.fine_hand_gesture_confidence, 0.88)
        debug_snapshot = pipeline.debug_snapshot()
        self.assertTrue(debug_snapshot["fresh_live_results_confirm_no_hand"])
        self.assertEqual(debug_snapshot["live_hand_box_source"], "recent")
        self.assertEqual(debug_snapshot["resolved_source"], "person_roi")
        self.assertIsNone(debug_snapshot["person_roi_block_reason"])

    def test_observe_allows_peace_sign_person_roi_even_when_fresh_live_results_confirm_no_hand(self) -> None:
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
                    gestures=[[_Category("none", 0.98)]],
                    hand_landmarks=[],
                )
            ],
            custom_results=[
                _GestureResult(
                    gestures=[[_Category("none", 0.98)]],
                    hand_landmarks=[],
                )
            ],
            roi_builtin_results=[
                _GestureResult(
                    gestures=[[_Category("Victory", 0.87)]],
                    hand_landmarks=[],
                )
            ],
            roi_custom_results=[
                _GestureResult(
                    gestures=[[_Category("none", 0.98)]],
                    hand_landmarks=[],
                )
            ],
        )
        pipeline._hand_landmark_worker = _StubHandLandmarkWorker(
            SimpleNamespace(
                detections=(
                    SimpleNamespace(
                        roi_frame_rgb="roi-frame",
                        roi_source=SimpleNamespace(value="right_wrist"),
                        handedness="right",
                        handedness_score=0.99,
                    ),
                ),
                final_timestamp_ms=12500,
            )
        )

        observation = pipeline.observe(
            frame_rgb="frame",
            observed_at=12.5,
            primary_person_box=AICameraBox(top=0.2, left=0.2, bottom=0.8, right=0.8),
            visible_person_boxes=(AICameraBox(top=0.2, left=0.2, bottom=0.8, right=0.8),),
            person_count=1,
        )

        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.PEACE_SIGN)
        self.assertEqual(observation.fine_hand_gesture_confidence, 0.87)
        debug_snapshot = pipeline.debug_snapshot()
        self.assertTrue(debug_snapshot["fresh_live_results_confirm_no_hand"])
        self.assertEqual(debug_snapshot["resolved_source"], "person_roi")
        self.assertIsNone(debug_snapshot["person_roi_block_reason"])
        self.assertIsNone(debug_snapshot["full_frame_hand_attempt_reason"])

    def test_observe_person_roi_debug_lists_per_hand_crop_classification(self) -> None:
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
                    gestures=[[ _Category("None", 0.82) ]],
                    hand_landmarks=[],
                ),
                _GestureResult(
                    gestures=[[ _Category("Thumb_Down", 0.41) ]],
                    hand_landmarks=[],
                ),
            ],
        )
        pipeline._hand_landmark_worker = _StubHandLandmarkWorker(
            SimpleNamespace(
                detections=(
                    SimpleNamespace(
                        roi_frame_rgb="roi-frame-0",
                        roi_source=SimpleNamespace(value="primary_person_upper_body"),
                        handedness="right",
                        handedness_score=0.98,
                    ),
                    SimpleNamespace(
                        roi_frame_rgb="roi-frame-1",
                        roi_source=SimpleNamespace(value="primary_person_full_body"),
                        handedness="right",
                        handedness_score=0.97,
                    ),
                ),
                final_timestamp_ms=7100,
            )
        )

        observation = pipeline.observe(
            frame_rgb="frame",
            observed_at=7.1,
            primary_person_box=AICameraBox(top=0.2, left=0.2, bottom=0.8, right=0.8),
            person_count=1,
        )

        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.NONE)
        debug_snapshot = pipeline.debug_snapshot()
        self.assertIsNone(debug_snapshot["person_roi_builtin_detection_index"])
        self.assertEqual(len(debug_snapshot["person_roi_detection_debug"]), 2)
        self.assertEqual(
            debug_snapshot["person_roi_detection_debug"][0]["builtin_categories"][0]["normalized_label"],
            "none",
        )
        self.assertEqual(
            debug_snapshot["person_roi_detection_debug"][1]["builtin_categories"][0]["normalized_label"],
            "thumb_down",
        )
        self.assertEqual(
            debug_snapshot["person_roi_detection_debug"][1]["builtin_raw_gesture"],
            AICameraFineHandGesture.THUMBS_DOWN.value,
        )
        self.assertFalse(debug_snapshot["person_roi_detection_debug"][1]["builtin_source_accepted"])
        self.assertEqual(
            debug_snapshot["person_roi_detection_debug"][1]["builtin_source_min_confidence"],
            0.82,
        )

    def test_observe_prefers_tighter_wrist_person_roi_over_broader_body_crop_when_scores_are_close(self) -> None:
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
                    gestures=[[ _Category("Thumb_Down", 0.79) ]],
                    hand_landmarks=[],
                ),
                _GestureResult(
                    gestures=[[ _Category("Victory", 0.71) ]],
                    hand_landmarks=[],
                ),
            ],
        )
        pipeline._hand_landmark_worker = _StubHandLandmarkWorker(
            SimpleNamespace(
                detections=(
                    SimpleNamespace(
                        roi_frame_rgb="roi-frame-0",
                        roi_source=SimpleNamespace(value="primary_person_upper_body"),
                        handedness="right",
                        handedness_score=0.98,
                    ),
                    SimpleNamespace(
                        roi_frame_rgb="roi-frame-1",
                        roi_source=SimpleNamespace(value="right_wrist"),
                        handedness="right",
                        handedness_score=0.98,
                    ),
                ),
                final_timestamp_ms=7150,
            )
        )

        observation = pipeline.observe(
            frame_rgb="frame",
            observed_at=7.15,
            primary_person_box=AICameraBox(top=0.2, left=0.2, bottom=0.8, right=0.8),
            person_count=1,
        )

        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.PEACE_SIGN)
        debug_snapshot = pipeline.debug_snapshot()
        self.assertEqual(debug_snapshot["person_roi_builtin_detection_index"], 1)
        self.assertEqual(debug_snapshot["person_roi_combined_source"], "right_wrist")

    def test_observe_person_roi_retries_with_looser_hand_context_when_tight_crop_stays_none(self) -> None:
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
                    gestures=[[ _Category("none", 0.95) ]],
                    hand_landmarks=[],
                ),
                _GestureResult(
                    gestures=[[ _Category("Victory", 0.84) ]],
                    hand_landmarks=[],
                ),
            ],
        )
        pipeline._hand_landmark_worker = _StubHandLandmarkWorker(
            SimpleNamespace(
                detections=(
                    SimpleNamespace(
                        roi_frame_rgb="roi-tight",
                        gesture_context_frame_rgb="roi-context",
                        roi_source=SimpleNamespace(value="right_wrist"),
                        handedness="right",
                        handedness_score=0.98,
                    ),
                ),
                final_timestamp_ms=7160,
            )
        )

        observation = pipeline.observe(
            frame_rgb="frame",
            observed_at=7.16,
            primary_person_box=AICameraBox(top=0.2, left=0.2, bottom=0.8, right=0.8),
            person_count=1,
        )

        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.PEACE_SIGN)
        debug_snapshot = pipeline.debug_snapshot()
        self.assertEqual(debug_snapshot["person_roi_combined_source"], "right_wrist")
        self.assertTrue(debug_snapshot["person_roi_detection_debug"][0]["builtin_context_retry_used"])
        self.assertEqual(
            debug_snapshot["person_roi_detection_debug"][0]["builtin_context_gesture"],
            AICameraFineHandGesture.PEACE_SIGN.value,
        )

    def test_observe_prefers_full_frame_landmark_crop_for_person_roi_symbol_read(self) -> None:
        pipeline = LiveGesturePipeline(
            config=MediaPipeVisionConfig(
                pose_model_path="pose.task",
                hand_landmarker_model_path="hand.task",
                gesture_model_path="gesture.task",
            )
        )
        runtime = _ImageSelectiveRoiRuntime(
            builtin_results=[
                _GestureResult(
                    gestures=[[ _Category("none", 0.94) ]],
                    hand_landmarks=[],
                )
            ],
            image_to_result={
                "gesture-frame": _GestureResult(
                    gestures=[[ _Category("Victory", 0.87) ]],
                    hand_landmarks=[],
                ),
                "roi-frame": _GestureResult(
                    gestures=[[ _Category("none", 0.99) ]],
                    hand_landmarks=[],
                ),
            },
        )
        pipeline._runtime = runtime
        pipeline._hand_landmark_worker = _StubHandLandmarkWorker(
            SimpleNamespace(
                detections=(
                    SimpleNamespace(
                        roi_frame_rgb="roi-frame",
                        gesture_frame_rgb="gesture-frame",
                        gesture_context_frame_rgb="gesture-context",
                        roi_source=SimpleNamespace(value="right_wrist"),
                        handedness="right",
                        handedness_score=0.98,
                    ),
                ),
                final_timestamp_ms=7165,
            )
        )

        observation = pipeline.observe(
            frame_rgb="frame",
            observed_at=7.165,
            primary_person_box=AICameraBox(top=0.2, left=0.2, bottom=0.8, right=0.8),
            person_count=1,
        )

        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.PEACE_SIGN)
        self.assertEqual(runtime.roi_inputs, ["gesture-frame"])
        detection_debug = pipeline.debug_snapshot()["person_roi_detection_debug"][0]
        self.assertEqual(detection_debug["gesture_frame_source"], "full_frame_landmark_crop")

    def test_observe_accepts_hand_localized_full_frame_thumb_from_person_roi_below_body_crop_floor(self) -> None:
        pipeline = LiveGesturePipeline(
            config=MediaPipeVisionConfig(
                pose_model_path="pose.task",
                hand_landmarker_model_path="hand.task",
                gesture_model_path="gesture.task",
            )
        )
        runtime = _ImageSelectiveRoiRuntime(
            builtin_results=[
                _GestureResult(
                    gestures=[[_Category("none", 0.94)]],
                    hand_landmarks=[],
                )
            ],
            image_to_result={
                "gesture-frame": _GestureResult(
                    gestures=[[_Category("Thumb_Down", 0.393)]],
                    hand_landmarks=[],
                ),
                "roi-frame": _GestureResult(
                    gestures=[[_Category("none", 0.99)]],
                    hand_landmarks=[],
                ),
            },
        )
        pipeline._runtime = runtime
        pipeline._hand_landmark_worker = _StubHandLandmarkWorker(
            SimpleNamespace(
                detections=(
                    SimpleNamespace(
                        roi_frame_rgb="roi-frame",
                        gesture_frame_rgb="gesture-frame",
                        roi_source=SimpleNamespace(value="primary_person_upper_body"),
                        handedness="right",
                        handedness_score=0.98,
                    ),
                ),
                final_timestamp_ms=7165,
            )
        )

        observation = pipeline.observe(
            frame_rgb="frame",
            observed_at=7.165,
            primary_person_box=AICameraBox(top=0.2, left=0.2, bottom=0.8, right=0.8),
            person_count=1,
        )

        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.THUMBS_DOWN)
        self.assertAlmostEqual(observation.fine_hand_gesture_confidence or 0.0, 0.393, places=3)
        detection_debug = pipeline.debug_snapshot()["person_roi_detection_debug"][0]
        self.assertEqual(detection_debug["gesture_frame_source"], "full_frame_landmark_crop")
        self.assertIsNone(detection_debug["builtin_source_min_confidence"])
        self.assertTrue(detection_debug["builtin_source_accepted"])
        self.assertEqual(runtime.roi_inputs, ["gesture-frame"])

    def test_observe_keeps_rejecting_body_crop_thumb_when_no_full_frame_hand_crop_exists(self) -> None:
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
                    gestures=[[_Category("Thumb_Down", 0.393)]],
                    hand_landmarks=[],
                )
            ],
        )
        pipeline._hand_landmark_worker = _StubHandLandmarkWorker(
            SimpleNamespace(
                detections=(
                    SimpleNamespace(
                        roi_frame_rgb="roi-frame",
                        roi_source=SimpleNamespace(value="primary_person_upper_body"),
                        handedness="right",
                        handedness_score=0.98,
                    ),
                ),
                final_timestamp_ms=7166,
            )
        )

        observation = pipeline.observe(
            frame_rgb="frame",
            observed_at=7.166,
            primary_person_box=AICameraBox(top=0.2, left=0.2, bottom=0.8, right=0.8),
            person_count=1,
        )

        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.NONE)
        detection_debug = pipeline.debug_snapshot()["person_roi_detection_debug"][0]
        self.assertEqual(detection_debug["gesture_frame_source"], "roi_local_crop")
        self.assertEqual(detection_debug["builtin_source_min_confidence"], 0.76)
        self.assertFalse(detection_debug["builtin_source_accepted"])

    def test_observe_handles_numpy_full_frame_landmark_crop_in_debug_summary(self) -> None:
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
                    gestures=[[_Category("Victory", 0.87)]],
                    hand_landmarks=[],
                )
            ],
        )
        pipeline._hand_landmark_worker = _StubHandLandmarkWorker(
            SimpleNamespace(
                detections=(
                    SimpleNamespace(
                        roi_frame_rgb=np.zeros((12, 10, 3), dtype=np.uint8),
                        gesture_frame_rgb=np.zeros((20, 18, 3), dtype=np.uint8),
                        gesture_context_frame_rgb=np.zeros((24, 22, 3), dtype=np.uint8),
                        roi_source=SimpleNamespace(value="right_wrist"),
                        handedness="right",
                        handedness_score=0.98,
                        confidence=0.93,
                    ),
                ),
                final_timestamp_ms=7165,
            )
        )

        observation = pipeline.observe(
            frame_rgb=np.zeros((32, 32, 3), dtype=np.uint8),
            observed_at=7.165,
            primary_person_box=AICameraBox(top=0.2, left=0.2, bottom=0.8, right=0.8),
            person_count=1,
        )

        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.PEACE_SIGN)
        detection_debug = pipeline.debug_snapshot()["person_roi_detection_debug"][0]
        self.assertEqual(detection_debug["gesture_frame_source"], "full_frame_landmark_crop")
        self.assertEqual(detection_debug["roi_frame_shape"], (12, 10, 3))
        self.assertEqual(detection_debug["gesture_frame_shape"], (20, 18, 3))
        self.assertEqual(detection_debug["gesture_context_frame_shape"], (24, 22, 3))

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
                    gestures=[[_Category("Victory", 0.51)]],
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

        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.PEACE_SIGN)
        debug_snapshot = pipeline.debug_snapshot()
        self.assertEqual(debug_snapshot["resolved_source"], "visible_person_roi")
        self.assertEqual(debug_snapshot["person_roi_candidate_count"], 2)
        self.assertEqual(debug_snapshot["person_roi_match_index"], 1)
        self.assertEqual(debug_snapshot["person_roi_detection_count"], 1)

    def test_observe_short_circuits_secondary_visible_person_scan_after_confident_pose_hint_roi(self) -> None:
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
                    gestures=[[_Category("none", 0.94)]],
                    hand_landmarks=[[ _Landmark(0.32, 0.46), _Landmark(0.38, 0.54) ]],
                )
            ],
            custom_results=[
                _GestureResult(
                    gestures=[[_Category("none", 0.94)]],
                    hand_landmarks=[[ _Landmark(0.32, 0.46), _Landmark(0.38, 0.54) ]],
                )
            ],
            roi_builtin_results=[
                _GestureResult(
                    gestures=[[_Category("none", 0.94)]],
                    hand_landmarks=[],
                )
            ],
            roi_custom_results=[
                _GestureResult(
                    gestures=[[_Category("thumbs_up", 0.91)]],
                    hand_landmarks=[],
                )
            ],
        )
        worker = _RecordingHandLandmarkWorker(
            [
                SimpleNamespace(
                    detections=(SimpleNamespace(roi_frame_rgb="roi-frame-primary"),),
                    final_timestamp_ms=8400,
                ),
                SimpleNamespace(
                    detections=(SimpleNamespace(roi_frame_rgb="roi-frame-secondary"),),
                    final_timestamp_ms=8401,
                ),
            ]
        )
        pipeline._hand_landmark_worker = worker
        primary_person_box = AICameraBox(top=0.2, left=0.1, bottom=0.9, right=0.45)

        observation = pipeline.observe(
            frame_rgb="frame",
            observed_at=8.4,
            primary_person_box=primary_person_box,
            visible_person_boxes=(
                primary_person_box,
                AICameraBox(top=0.18, left=0.55, bottom=0.88, right=0.92),
            ),
            person_count=2,
            sparse_keypoints={0: (0.3, 0.4, 0.9)},
        )

        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.THUMBS_UP)
        self.assertEqual(len(worker.calls), 1)
        debug_snapshot = pipeline.debug_snapshot()
        self.assertEqual(debug_snapshot["resolved_source"], "visible_person_roi")
        self.assertEqual(debug_snapshot["person_roi_match_index"], 0)
        self.assertTrue(debug_snapshot["person_roi_short_circuit_used"])
        self.assertEqual(debug_snapshot["person_roi_short_circuit_index"], 0)
        self.assertEqual(debug_snapshot["person_roi_short_circuit_reason"], "primary_pose_hint_confident_gesture")

    def test_observe_keeps_scanning_secondary_visible_person_when_pose_hint_roi_stays_below_short_circuit_floor(self) -> None:
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
                    gestures=[[_Category("Thumb_Up", 0.51)]],
                    hand_landmarks=[],
                ),
                _GestureResult(
                    gestures=[[_Category("Thumb_Down", 0.89)]],
                    hand_landmarks=[],
                ),
            ],
        )
        worker = _RecordingHandLandmarkWorker(
            [
                SimpleNamespace(
                    detections=(SimpleNamespace(roi_frame_rgb="roi-frame-primary"),),
                    final_timestamp_ms=8450,
                ),
                SimpleNamespace(
                    detections=(SimpleNamespace(roi_frame_rgb="roi-frame-secondary"),),
                    final_timestamp_ms=8451,
                ),
            ]
        )
        pipeline._hand_landmark_worker = worker
        primary_person_box = AICameraBox(top=0.2, left=0.1, bottom=0.9, right=0.45)

        observation = pipeline.observe(
            frame_rgb="frame",
            observed_at=8.45,
            primary_person_box=primary_person_box,
            visible_person_boxes=(
                primary_person_box,
                AICameraBox(top=0.18, left=0.55, bottom=0.88, right=0.92),
            ),
            person_count=2,
            sparse_keypoints={0: (0.3, 0.4, 0.9)},
        )

        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.THUMBS_DOWN)
        self.assertEqual(len(worker.calls), 2)
        debug_snapshot = pipeline.debug_snapshot()
        self.assertEqual(debug_snapshot["resolved_source"], "visible_person_roi")
        self.assertEqual(debug_snapshot["person_roi_match_index"], 1)
        self.assertFalse(debug_snapshot["person_roi_short_circuit_used"])
        self.assertIsNone(debug_snapshot["person_roi_short_circuit_reason"])

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
                    gestures=[[_Category("Victory", 0.51)]],
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

        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.PEACE_SIGN)
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
                    gestures=[[ _Category("Victory", 0.87) ]],
                    hand_landmarks=[],
                )
            ],
        )

        pipeline.observe(frame_rgb="frame", observed_at=2.0)
        observation = pipeline.observe(frame_rgb="frame", observed_at=2.2)

        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.PEACE_SIGN)
        self.assertEqual(pipeline.debug_snapshot()["resolved_source"], "recent_live_hand_roi")

    def test_observe_recovers_symbol_from_full_frame_hand_landmarks_when_no_person_or_live_hand_box_exists(self) -> None:
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
                    gestures=[[_Category("none", 0.96)]],
                    hand_landmarks=[],
                )
            ],
            roi_builtin_results=[
                _GestureResult(
                    gestures=[[_Category("Thumb_Down", 0.93)]],
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
        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.THUMBS_DOWN)
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
                    gestures=[[_Category("Victory", 0.92)]],
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
        self.assertEqual(observation.fine_hand_gesture, AICameraFineHandGesture.PEACE_SIGN)
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
