import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.camera_ai.models import AICameraBox, AICameraFineHandGesture, AICameraGestureEvent
from twinr.hardware.hand_landmarks import HandLandmarkDetection, HandLandmarkResult, HandLandmarkPoint, HandRoiSource
from twinr.hardware.camera_ai.config import MediaPipeVisionConfig
from twinr.hardware.camera_ai.mediapipe_runtime import (
    MEDIAPIPE_LIVE_CALLBACK_CACHE_KEY_ATTR,
    MediaPipeTaskRuntime,
)
from twinr.hardware.camera_ai.fine_hand_gestures import (
    combine_builtin_and_custom_gesture_choice as _combine_builtin_and_custom_gesture_choice,
    combine_task_specific_custom_gesture_choice as _combine_task_specific_custom_gesture_choice,
    prefer_gesture_choice as _prefer_gesture_choice,
    resolve_fine_hand_gesture as _resolve_fine_hand_gesture,
)
from twinr.hardware.camera_ai.mediapipe_pipeline import MediaPipeVisionPipeline
from twinr.hardware.camera_ai.temporal_gestures import (
    TemporalPoseGestureClassifier,
    classify_temporal_gesture as _classify_temporal_gesture,
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


class _FakeImageArray:
    def __init__(self, *, contiguous: bool, dtype: str = "uint8") -> None:
        self.shape = (32, 24, 3)
        self.dtype = dtype
        self.flags = SimpleNamespace(c_contiguous=contiguous)

    def copy(self, order: str | None = None):
        return _FakeImageArray(contiguous=True, dtype=self.dtype)


class MediaPipeVisionTests(unittest.TestCase):
    def test_runtime_reuses_live_recognizer_when_callbacks_share_cache_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "gesture.task"
            model_path.write_bytes(b"gesture")
            runtime = MediaPipeTaskRuntime(
                config=MediaPipeVisionConfig(
                    pose_model_path="pose.task",
                    hand_landmarker_model_path="hand.task",
                    gesture_model_path=str(model_path),
                )
            )
            created_callbacks: list[object] = []

            def _create_from_options(options):
                created_callbacks.append(options.result_callback)
                return SimpleNamespace(close=lambda: None)

            vision_runtime = {
                "vision": SimpleNamespace(
                    RunningMode=SimpleNamespace(LIVE_STREAM="live"),
                    GestureRecognizerOptions=lambda **kwargs: SimpleNamespace(**kwargs),
                    GestureRecognizer=SimpleNamespace(create_from_options=_create_from_options),
                ),
                "BaseOptions": lambda **kwargs: SimpleNamespace(**kwargs),
                "ClassifierOptions": lambda **kwargs: SimpleNamespace(**kwargs),
            }

            first_calls: list[int] = []
            second_calls: list[int] = []

            def _first_callback(result, output_image, timestamp_ms):
                del result, output_image
                first_calls.append(timestamp_ms)

            def _second_callback(result, output_image, timestamp_ms):
                del result, output_image
                second_calls.append(timestamp_ms)

            setattr(
                _first_callback,
                MEDIAPIPE_LIVE_CALLBACK_CACHE_KEY_ATTR,
                ("test_runtime_reuses_live_recognizer", "builtin"),
            )
            setattr(
                _second_callback,
                MEDIAPIPE_LIVE_CALLBACK_CACHE_KEY_ATTR,
                ("test_runtime_reuses_live_recognizer", "builtin"),
            )

            recognizer_one = runtime.ensure_live_gesture_recognizer(
                vision_runtime,
                result_callback=_first_callback,
                num_hands_override=1,
            )
            recognizer_two = runtime.ensure_live_gesture_recognizer(
                vision_runtime,
                result_callback=_second_callback,
                num_hands_override=1,
            )

            self.assertIs(recognizer_one, recognizer_two)
            self.assertEqual(len(created_callbacks), 1)
            runtime.close()

    def test_runtime_direct_live_callback_proxy_swaps_target_without_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "gesture.task"
            model_path.write_bytes(b"gesture")
            runtime = MediaPipeTaskRuntime(
                config=MediaPipeVisionConfig(
                    pose_model_path="pose.task",
                    hand_landmarker_model_path="hand.task",
                    gesture_model_path=str(model_path),
                )
            )
            created_callbacks: list[object] = []

            def _create_from_options(options):
                created_callbacks.append(options.result_callback)
                return SimpleNamespace(close=lambda: None)

            vision_runtime = {
                "vision": SimpleNamespace(
                    RunningMode=SimpleNamespace(LIVE_STREAM="live"),
                    GestureRecognizerOptions=lambda **kwargs: SimpleNamespace(**kwargs),
                    GestureRecognizer=SimpleNamespace(create_from_options=_create_from_options),
                ),
                "BaseOptions": lambda **kwargs: SimpleNamespace(**kwargs),
                "ClassifierOptions": lambda **kwargs: SimpleNamespace(**kwargs),
            }

            calls: list[tuple[str, int]] = []

            def _first_callback(result, output_image, timestamp_ms):
                del result, output_image
                calls.append(("first", timestamp_ms))

            def _second_callback(result, output_image, timestamp_ms):
                del result, output_image
                calls.append(("second", timestamp_ms))

            setattr(
                _first_callback,
                MEDIAPIPE_LIVE_CALLBACK_CACHE_KEY_ATTR,
                ("test_runtime_direct_live_callback_proxy", "builtin"),
            )
            setattr(
                _second_callback,
                MEDIAPIPE_LIVE_CALLBACK_CACHE_KEY_ATTR,
                ("test_runtime_direct_live_callback_proxy", "builtin"),
            )

            with mock.patch.dict(os.environ, {"TWINR_MEDIAPIPE_LIVE_CALLBACK_ASYNC": "0"}, clear=False):
                recognizer_one = runtime.ensure_live_gesture_recognizer(
                    vision_runtime,
                    result_callback=_first_callback,
                    num_hands_override=1,
                )
                created_callbacks[0]("result", "image", 101)
                recognizer_two = runtime.ensure_live_gesture_recognizer(
                    vision_runtime,
                    result_callback=_second_callback,
                    num_hands_override=1,
                )
                created_callbacks[0]("result", "image", 202)

            self.assertIs(recognizer_one, recognizer_two)
            self.assertEqual(len(created_callbacks), 1)
            self.assertEqual(calls, [("first", 101), ("second", 202)])
            runtime.close()

    def test_runtime_reset_live_gesture_recognizers_closes_only_live_slots(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            gesture_model_path = Path(temp_dir) / "gesture.task"
            custom_model_path = Path(temp_dir) / "custom.task"
            gesture_model_path.write_bytes(b"gesture")
            custom_model_path.write_bytes(b"custom")
            runtime = MediaPipeTaskRuntime(
                config=MediaPipeVisionConfig(
                    pose_model_path="pose.task",
                    hand_landmarker_model_path="hand.task",
                    gesture_model_path=str(gesture_model_path),
                    custom_gesture_model_path=str(custom_model_path),
                )
            )
            closed: list[str] = []

            def _create_builtin(options):
                del options
                return SimpleNamespace(close=lambda: closed.append("builtin"))

            def _create_custom(options):
                del options
                return SimpleNamespace(close=lambda: closed.append("custom"))

            vision_runtime = {
                "vision": SimpleNamespace(
                    RunningMode=SimpleNamespace(LIVE_STREAM="live"),
                    GestureRecognizerOptions=lambda **kwargs: SimpleNamespace(**kwargs),
                    GestureRecognizer=SimpleNamespace(
                        create_from_options=lambda options: (
                            _create_custom(options)
                            if getattr(options.base_options, "model_asset_buffer", None) == b"custom"
                            else _create_builtin(options)
                        )
                    ),
                ),
                "BaseOptions": lambda **kwargs: SimpleNamespace(**kwargs),
                "ClassifierOptions": lambda **kwargs: SimpleNamespace(**kwargs),
            }

            runtime.ensure_live_gesture_recognizer(
                vision_runtime,
                result_callback=lambda *_args: None,
                num_hands_override=1,
            )
            runtime.ensure_live_custom_gesture_recognizer(
                vision_runtime,
                result_callback=lambda *_args: None,
            )

            runtime.reset_live_gesture_recognizers()

            self.assertCountEqual(closed, ["builtin", "custom"])
            self.assertIsNone(runtime._live_gesture_recognizer)
            self.assertIsNone(runtime._live_custom_gesture_recognizer)
            runtime.close()

    def test_runtime_build_image_normalizes_non_contiguous_uint8_view(self) -> None:
        runtime = MediaPipeTaskRuntime(
            config=MediaPipeVisionConfig(
                pose_model_path="pose.task",
                hand_landmarker_model_path="hand.task",
                gesture_model_path="gesture.task",
            )
        )
        observed = {}
        source = _FakeImageArray(contiguous=False)
        image = runtime.build_image(
            {
                "mp": SimpleNamespace(
                    ImageFormat=SimpleNamespace(SRGB="srgb"),
                    Image=lambda image_format, data: (
                        observed.setdefault("data", data),
                        SimpleNamespace(image_format=image_format, data=data),
                    )[1],
                )
            },
            frame_rgb=source,
        )

        self.assertFalse(source.flags.c_contiguous)
        self.assertTrue(observed["data"].flags.c_contiguous)
        self.assertEqual(observed["data"].dtype, "uint8")
        self.assertIs(image.data, observed["data"])

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

    def test_fine_gesture_mapping_picks_builtin_victory_as_peace_sign(self) -> None:
        gesture, confidence = _resolve_fine_hand_gesture(
            result=_GestureResult([[_Category("Victory", 0.89)]]),
            category_map={"victory": AICameraFineHandGesture.PEACE_SIGN},
            min_score=0.50,
        )

        self.assertEqual(gesture, AICameraFineHandGesture.PEACE_SIGN)
        self.assertAlmostEqual(confidence or 0.0, 0.89, places=3)

    def test_fine_gesture_mapping_suppresses_custom_positive_when_none_wins(self) -> None:
        gesture, confidence = _resolve_fine_hand_gesture(
            result=_GestureResult([[_Category("none", 0.82), _Category("ok_sign", 0.78)]]),
            category_map={"ok_sign": AICameraFineHandGesture.OK_SIGN},
            min_score=0.55,
        )

        self.assertEqual(gesture, AICameraFineHandGesture.NONE)
        self.assertIsNone(confidence)

    def test_task_specific_custom_choice_prefers_custom_three_gesture_over_conflicting_builtin(self) -> None:
        choice = _combine_task_specific_custom_gesture_choice(
            (AICameraFineHandGesture.THUMBS_DOWN, 0.61),
            (AICameraFineHandGesture.PEACE_SIGN, 0.88),
            preferred_custom_gestures={
                AICameraFineHandGesture.THUMBS_UP,
                AICameraFineHandGesture.THUMBS_DOWN,
                AICameraFineHandGesture.PEACE_SIGN,
            },
        )

        self.assertEqual(choice, (AICameraFineHandGesture.PEACE_SIGN, 0.88))

    def test_task_specific_custom_choice_keeps_builtin_when_custom_label_is_not_preferred(self) -> None:
        choice = _combine_task_specific_custom_gesture_choice(
            (AICameraFineHandGesture.THUMBS_UP, 0.84),
            (AICameraFineHandGesture.OK_SIGN, 0.93),
            preferred_custom_gestures={
                AICameraFineHandGesture.THUMBS_UP,
                AICameraFineHandGesture.THUMBS_DOWN,
                AICameraFineHandGesture.PEACE_SIGN,
            },
        )

        self.assertEqual(choice, (AICameraFineHandGesture.THUMBS_UP, 0.84))

    def test_runtime_configures_official_gesture_classifier_filters_for_none_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "gesture.task"
            model_path.write_bytes(b"gesture")
            runtime = MediaPipeTaskRuntime(
                config=MediaPipeVisionConfig(
                    pose_model_path="pose.task",
                    hand_landmarker_model_path="hand.task",
                    gesture_model_path=str(model_path),
                    builtin_gesture_min_score=0.41,
                    custom_gesture_min_score=0.52,
                )
            )
            captured: dict[str, object] = {}

            def _create_from_options(options):
                captured["options"] = options
                return SimpleNamespace()

            recognizer = runtime.ensure_roi_gesture_recognizer(
                {
                    "vision": SimpleNamespace(
                        RunningMode=SimpleNamespace(IMAGE="image"),
                        GestureRecognizerOptions=lambda **kwargs: SimpleNamespace(**kwargs),
                        GestureRecognizer=SimpleNamespace(create_from_options=_create_from_options),
                    ),
                    "BaseOptions": lambda **kwargs: SimpleNamespace(**kwargs),
                    "ClassifierOptions": lambda **kwargs: SimpleNamespace(**kwargs),
                }
            )

        self.assertIsNotNone(recognizer)
        options = captured["options"]
        self.assertEqual(options.canned_gesture_classifier_options.category_denylist, ["None"])
        self.assertAlmostEqual(options.canned_gesture_classifier_options.score_threshold, 0.41, places=3)
        self.assertIsNone(getattr(options, "custom_gesture_classifier_options", None))

    def test_runtime_skips_classifier_options_for_custom_gesture_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_model_path = Path(temp_dir) / "custom.task"
            custom_model_path.write_bytes(b"custom")
            runtime = MediaPipeTaskRuntime(
                config=MediaPipeVisionConfig(
                    pose_model_path="pose.task",
                    hand_landmarker_model_path="hand.task",
                    gesture_model_path="gesture.task",
                    custom_gesture_model_path=str(custom_model_path),
                    custom_gesture_min_score=0.52,
                )
            )
            captured: dict[str, object] = {}

            def _create_from_options(options):
                captured["options"] = options
                return SimpleNamespace()

            recognizer = runtime.ensure_custom_roi_gesture_recognizer(
                {
                    "vision": SimpleNamespace(
                        RunningMode=SimpleNamespace(IMAGE="image"),
                        GestureRecognizerOptions=lambda **kwargs: SimpleNamespace(**kwargs),
                        GestureRecognizer=SimpleNamespace(create_from_options=_create_from_options),
                    ),
                    "BaseOptions": lambda **kwargs: SimpleNamespace(**kwargs),
                    "ClassifierOptions": lambda **kwargs: SimpleNamespace(**kwargs),
                }
            )

        self.assertIsNotNone(recognizer)
        options = captured["options"]
        self.assertIsNone(getattr(options, "canned_gesture_classifier_options", None))
        self.assertIsNone(getattr(options, "custom_gesture_classifier_options", None))

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
                    hand_landmarker_model_path=str(Path(temp_dir) / "hand.task"),
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
                pipeline._recognize_fine_gesture(
                    runtime={},
                    image=object(),
                    timestamp_ms=1,
                    hand_landmark_result=HandLandmarkResult(),
                )

        self.assertIn("mediapipe_custom_gesture_model_missing", str(context.exception))

    def test_pipeline_prefers_roi_gesture_when_hand_rois_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = MediaPipeVisionPipeline(
                config=MediaPipeVisionConfig(
                    pose_model_path="pose.task",
                    hand_landmarker_model_path=str(Path(temp_dir) / "hand.task"),
                    gesture_model_path=str(Path(temp_dir) / "gesture.task"),
                )
            )
            pipeline._build_image = lambda runtime, frame_rgb: frame_rgb
            pipeline._gesture_recognizer = type(
                "_StubGestureRecognizer",
                (),
                {
                    "recognize_for_video": staticmethod(
                        lambda image, timestamp_ms: (_ for _ in ()).throw(
                            AssertionError("full-frame recognizer should not run when ROI already produced a symbol")
                        )
                    )
                },
            )()
            pipeline._ensure_roi_gesture_recognizer = lambda runtime: type(
                "_StubRoiGestureRecognizer",
                (),
                {"recognize": staticmethod(lambda image: _GestureResult([[_Category("Pointing_Up", 0.91)]]))},
            )()

            gesture, confidence = pipeline._recognize_fine_gesture(
                runtime={},
                image="full-frame",
                timestamp_ms=1,
                hand_landmark_result=HandLandmarkResult(
                    detections=(
                        HandLandmarkDetection(
                            roi_source=HandRoiSource.RIGHT_WRIST,
                            roi_frame_rgb="roi-crop",
                            roi=AICameraBox(top=0.20, left=0.40, bottom=0.52, right=0.66),
                            handedness="right",
                            handedness_score=0.91,
                            landmarks=(HandLandmarkPoint(x=0.5, y=0.5, z=0.0),),
                        ),
                    )
                ),
            )

        self.assertEqual(gesture, AICameraFineHandGesture.POINTING)
        self.assertAlmostEqual(confidence or 0.0, 0.91, places=3)

    def test_pipeline_falls_back_to_full_frame_when_roi_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = MediaPipeVisionPipeline(
                config=MediaPipeVisionConfig(
                    pose_model_path="pose.task",
                    hand_landmarker_model_path=str(Path(temp_dir) / "hand.task"),
                    gesture_model_path=str(Path(temp_dir) / "gesture.task"),
                )
            )
            pipeline._build_image = lambda runtime, frame_rgb: frame_rgb
            pipeline._gesture_recognizer = type(
                "_StubGestureRecognizer",
                (),
                {
                    "recognize_for_video": staticmethod(
                        lambda image, timestamp_ms: _GestureResult([[_Category("Pointing_Up", 0.89)]])
                    )
                },
            )()
            pipeline._ensure_roi_gesture_recognizer = lambda runtime: type(
                "_StubRoiGestureRecognizer",
                (),
                {"recognize": staticmethod(lambda image: _GestureResult([[_Category("none", 0.95)]]))},
            )()

            gesture, confidence = pipeline._recognize_fine_gesture(
                runtime={},
                image="full-frame",
                timestamp_ms=1,
                hand_landmark_result=HandLandmarkResult(
                    detections=(
                        HandLandmarkDetection(
                            roi_source=HandRoiSource.RIGHT_WRIST,
                            roi_frame_rgb="roi-crop",
                            roi=AICameraBox(top=0.20, left=0.40, bottom=0.52, right=0.66),
                            handedness="right",
                            handedness_score=0.91,
                            landmarks=(HandLandmarkPoint(x=0.5, y=0.5, z=0.0),),
                        ),
                    )
                ),
            )

        self.assertEqual(gesture, AICameraFineHandGesture.POINTING)
        self.assertAlmostEqual(confidence or 0.0, 0.89, places=3)

    def test_pipeline_falls_back_to_roi_gesture_when_full_frame_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = MediaPipeVisionPipeline(
                config=MediaPipeVisionConfig(
                    pose_model_path="pose.task",
                    hand_landmarker_model_path=str(Path(temp_dir) / "hand.task"),
                    gesture_model_path=str(Path(temp_dir) / "gesture.task"),
                )
            )
            pipeline._build_image = lambda runtime, frame_rgb: frame_rgb
            pipeline._gesture_recognizer = type(
                "_StubGestureRecognizer",
                (),
                {
                    "recognize_for_video": staticmethod(
                        lambda image, timestamp_ms: _GestureResult([[_Category("none", 0.91)]])
                    )
                },
            )()
            pipeline._ensure_roi_gesture_recognizer = lambda runtime: type(
                "_StubRoiGestureRecognizer",
                (),
                {"recognize": staticmethod(lambda image: _GestureResult([[_Category("Pointing_Up", 0.87)]]))},
            )()

            gesture, confidence = pipeline._recognize_fine_gesture(
                runtime={},
                image="full-frame",
                timestamp_ms=1,
                hand_landmark_result=HandLandmarkResult(
                    detections=(
                        HandLandmarkDetection(
                            roi_source=HandRoiSource.RIGHT_WRIST,
                            roi_frame_rgb="roi-crop",
                            roi=AICameraBox(top=0.20, left=0.40, bottom=0.52, right=0.66),
                            handedness="right",
                            handedness_score=0.91,
                            landmarks=(HandLandmarkPoint(x=0.5, y=0.5, z=0.0),),
                        ),
                    )
                ),
            )

        self.assertEqual(gesture, AICameraFineHandGesture.POINTING)
        self.assertAlmostEqual(confidence or 0.0, 0.87, places=3)

    def test_pipeline_falls_back_to_custom_roi_gesture_when_full_frame_is_generic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_model = Path(temp_dir) / "custom.task"
            custom_model.write_bytes(b"custom")
            pipeline = MediaPipeVisionPipeline(
                config=MediaPipeVisionConfig(
                    pose_model_path="pose.task",
                    hand_landmarker_model_path=str(Path(temp_dir) / "hand.task"),
                    gesture_model_path=str(Path(temp_dir) / "gesture.task"),
                    custom_gesture_model_path=str(custom_model),
                )
            )
            pipeline._build_image = lambda runtime, frame_rgb: frame_rgb
            pipeline._gesture_recognizer = type(
                "_StubGestureRecognizer",
                (),
                {
                    "recognize_for_video": staticmethod(
                        lambda image, timestamp_ms: _GestureResult([[_Category("Open_Palm", 0.78)]])
                    )
                },
            )()
            pipeline._custom_gesture_recognizer = type(
                "_StubCustomGestureRecognizer",
                (),
                {
                    "recognize_for_video": staticmethod(
                        lambda image, timestamp_ms: _GestureResult([[_Category("none", 0.88)]])
                    )
                },
            )()
            pipeline._ensure_roi_gesture_recognizer = lambda runtime: type(
                "_StubRoiGestureRecognizer",
                (),
                {"recognize": staticmethod(lambda image: _GestureResult([[_Category("Open_Palm", 0.76)]]))},
            )()
            pipeline._ensure_custom_roi_gesture_recognizer = lambda runtime: type(
                "_StubCustomRoiGestureRecognizer",
                (),
                {"recognize": staticmethod(lambda image: _GestureResult([[_Category("ok_sign", 0.91)]]))},
            )()

            gesture, confidence = pipeline._recognize_fine_gesture(
                runtime={},
                image="full-frame",
                timestamp_ms=1,
                hand_landmark_result=HandLandmarkResult(
                    detections=(
                        HandLandmarkDetection(
                            roi_source=HandRoiSource.LEFT_WRIST,
                            roi_frame_rgb="roi-crop",
                            roi=AICameraBox(top=0.18, left=0.32, bottom=0.54, right=0.60),
                            handedness="left",
                            handedness_score=0.89,
                            landmarks=(HandLandmarkPoint(x=0.5, y=0.5, z=0.0),),
                        ),
                    )
                ),
            )

        self.assertEqual(gesture, AICameraFineHandGesture.OK_SIGN)
        self.assertAlmostEqual(confidence or 0.0, 0.91, places=3)

    def test_prefer_gesture_choice_keeps_higher_confidence(self) -> None:
        choice = _prefer_gesture_choice(
            (AICameraFineHandGesture.THUMBS_UP, 0.62),
            (AICameraFineHandGesture.POINTING, 0.87),
        )

        self.assertEqual(choice, (AICameraFineHandGesture.POINTING, 0.87))

    def test_combine_builtin_and_custom_keeps_builtin_priority_gesture(self) -> None:
        choice = _combine_builtin_and_custom_gesture_choice(
            (AICameraFineHandGesture.PEACE_SIGN, 0.88),
            (AICameraFineHandGesture.OK_SIGN, 0.93),
        )

        self.assertEqual(choice, (AICameraFineHandGesture.PEACE_SIGN, 0.88))

    def test_combine_builtin_and_custom_allows_custom_over_generic_open_palm(self) -> None:
        choice = _combine_builtin_and_custom_gesture_choice(
            (AICameraFineHandGesture.OPEN_PALM, 0.71),
            (AICameraFineHandGesture.OK_SIGN, 0.91),
        )

        self.assertEqual(choice, (AICameraFineHandGesture.OK_SIGN, 0.91))

    def test_pipeline_keeps_builtin_peace_sign_when_custom_ok_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_model = Path(temp_dir) / "custom.task"
            custom_model.write_bytes(b"custom")
            pipeline = MediaPipeVisionPipeline(
                config=MediaPipeVisionConfig(
                    pose_model_path="pose.task",
                    hand_landmarker_model_path=str(Path(temp_dir) / "hand.task"),
                    gesture_model_path=str(Path(temp_dir) / "gesture.task"),
                    custom_gesture_model_path=str(custom_model),
                )
            )
            pipeline._build_image = lambda runtime, frame_rgb: frame_rgb
            pipeline._gesture_recognizer = type(
                "_StubGestureRecognizer",
                (),
                {
                    "recognize_for_video": staticmethod(
                        lambda image, timestamp_ms: _GestureResult([[_Category("Victory", 0.88)]])
                    )
                },
            )()
            pipeline._custom_gesture_recognizer = type(
                "_StubCustomGestureRecognizer",
                (),
                {
                    "recognize_for_video": staticmethod(
                        lambda image, timestamp_ms: _GestureResult([[_Category("ok_sign", 0.94)]])
                    )
                },
            )()

            gesture, confidence = pipeline._recognize_fine_gesture(
                runtime={},
                image="full-frame",
                timestamp_ms=11,
                hand_landmark_result=HandLandmarkResult(),
            )

        self.assertEqual(gesture, AICameraFineHandGesture.PEACE_SIGN)
        self.assertAlmostEqual(confidence or 0.0, 0.88, places=3)

    def test_pipeline_uses_single_full_frame_timestamp_for_stable_gesture_tracking(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = MediaPipeVisionPipeline(
                config=MediaPipeVisionConfig(
                    pose_model_path="pose.task",
                    hand_landmarker_model_path=str(Path(temp_dir) / "hand.task"),
                    gesture_model_path=str(Path(temp_dir) / "gesture.task"),
                )
            )
            pipeline._build_image = lambda runtime, frame_rgb: frame_rgb
            timestamps: list[int] = []
            pipeline._gesture_recognizer = type(
                "_StubGestureRecognizer",
                (),
                {
                    "recognize_for_video": staticmethod(
                        lambda image, timestamp_ms: (
                            timestamps.append(timestamp_ms),
                            _GestureResult([]),
                        )[1]
                    )
                },
            )()

            pipeline._recognize_fine_gesture(
                runtime={},
                image="full-frame",
                timestamp_ms=7,
                hand_landmark_result=HandLandmarkResult(
                    detections=(
                        HandLandmarkDetection(
                            roi_source=HandRoiSource.LEFT_WRIST,
                            roi_frame_rgb="roi-crop",
                            roi=AICameraBox(top=0.20, left=0.34, bottom=0.52, right=0.58),
                            handedness="left",
                            handedness_score=0.78,
                            landmarks=(HandLandmarkPoint(x=0.5, y=0.5, z=0.0),),
                        ),
                    )
                ),
            )

        self.assertEqual(timestamps, [7])
        self.assertEqual(pipeline._last_timestamp_ms, 7)

    def test_pipeline_reserves_hand_landmark_final_timestamp_even_without_detections(self) -> None:
        pipeline = MediaPipeVisionPipeline(
            config=MediaPipeVisionConfig(
                pose_model_path="pose.task",
                hand_landmarker_model_path="hand.task",
                gesture_model_path="gesture.task",
            )
        )
        reserved: list[int] = []
        pipeline._runtime = SimpleNamespace(
            load_runtime=lambda: {},
            build_image=lambda runtime, frame_rgb: frame_rgb,
            timestamp_ms=lambda observed_at: 7,
            reserve_timestamp=lambda timestamp_ms: reserved.append(timestamp_ms),
        )
        pipeline._pose_landmarker = SimpleNamespace(
            detect_for_video=lambda image, timestamp_ms: SimpleNamespace(pose_landmarks=[])
        )
        pipeline._analyze_hand_landmarks = lambda **kwargs: HandLandmarkResult(final_timestamp_ms=9)
        pipeline._recognize_fine_gesture = lambda **kwargs: (AICameraFineHandGesture.NONE, None)

        pipeline.analyze(
            frame_rgb="frame",
            observed_at=1.0,
            primary_person_box=AICameraBox(top=0.2, left=0.3, bottom=0.8, right=0.7),
        )

        self.assertEqual(reserved, [9])
        self.assertEqual(pipeline._last_timestamp_ms, 9)

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
