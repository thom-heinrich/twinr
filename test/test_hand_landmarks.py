from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.camera_ai.models import AICameraBox
from twinr.hardware.hand_landmarks import (
    HandLandmarkResult,
    HandLandmarkWorkerConfig,
    HandRoiSource,
    MediaPipeHandLandmarkWorker,
    _build_hand_roi_candidates,
    _project_landmark_to_full_frame,
)


class _FakeFrame:
    def __init__(self, height: int, width: int, *, contiguous: bool = True, dtype: str = "uint8") -> None:
        self.shape = (height, width, 3)
        self.dtype = dtype
        self.flags = SimpleNamespace(c_contiguous=contiguous)

    def __getitem__(self, key):
        row_slice, col_slice = key
        start_row = int(row_slice.start or 0)
        stop_row = int(row_slice.stop or self.shape[0])
        start_col = int(col_slice.start or 0)
        stop_col = int(col_slice.stop or self.shape[1])
        return _FakeFrame(
            max(1, stop_row - start_row),
            max(1, stop_col - start_col),
            contiguous=False,
            dtype=self.dtype,
        )

    def copy(self, order: str | None = None):
        return _FakeFrame(self.shape[0], self.shape[1], contiguous=True, dtype=self.dtype)


class HandLandmarkWorkerTests(unittest.TestCase):
    def test_build_hand_roi_candidates_adds_wrist_windows_before_upper_body(self) -> None:
        candidates = _build_hand_roi_candidates(
            primary_person_box=AICameraBox(top=0.14, left=0.20, bottom=0.96, right=0.78),
            sparse_keypoints={
                5: (0.34, 0.42, 0.90),
                7: (0.28, 0.50, 0.88),
                9: (0.18, 0.54, 0.93),
                6: (0.62, 0.42, 0.91),
                8: (0.71, 0.50, 0.87),
                10: (0.83, 0.52, 0.95),
            },
            config=HandLandmarkWorkerConfig(model_path="state/mediapipe/models/hand_landmarker.task"),
        )

        self.assertEqual({candidates[0].source, candidates[1].source}, {HandRoiSource.LEFT_WRIST, HandRoiSource.RIGHT_WRIST})
        self.assertEqual(candidates[2].source, HandRoiSource.PRIMARY_PERSON_UPPER_BODY)
        wrist_centers = sorted(round(candidate.box.center_x, 2) for candidate in candidates[:2])
        self.assertEqual(wrist_centers, [0.18, 0.83])

    def test_project_landmark_to_full_frame_maps_roi_local_coords(self) -> None:
        point = _project_landmark_to_full_frame(
            landmark=SimpleNamespace(x=0.50, y=0.25, z=-0.07),
            roi=AICameraBox(top=0.20, left=0.30, bottom=0.60, right=0.70),
        )

        self.assertAlmostEqual(point.x, 0.50, places=3)
        self.assertAlmostEqual(point.y, 0.30, places=3)
        self.assertAlmostEqual(point.z, -0.07, places=3)

    def test_analyze_returns_highest_confidence_detection_first(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "hand_landmarker.task"
            model_path.write_bytes(b"hand")
            worker = MediaPipeHandLandmarkWorker(
                config=HandLandmarkWorkerConfig(
                    model_path=str(model_path),
                    num_hands=2,
                )
            )
            worker._hand_landmarker = SimpleNamespace(
                detect_for_video=lambda image, timestamp_ms: SimpleNamespace(
                    hand_landmarks=[
                        [SimpleNamespace(x=0.2, y=0.3, z=0.0)],
                        [SimpleNamespace(x=0.7, y=0.4, z=0.0)],
                    ],
                    handedness=[
                        [SimpleNamespace(category_name="Right", score=0.62)],
                        [SimpleNamespace(category_name="Left", score=0.91)],
                    ],
                )
            )
            frame = _FakeFrame(120, 160)
            runtime = {
                "mp": SimpleNamespace(
                    ImageFormat=SimpleNamespace(SRGB="srgb"),
                    Image=lambda image_format, data: SimpleNamespace(image_format=image_format, data=data),
                ),
                "vision": SimpleNamespace(),
                "BaseOptions": object,
            }

            result = worker.analyze(
                runtime=runtime,
                frame_rgb=frame,
                timestamp_ms=1,
                primary_person_box=AICameraBox(top=0.10, left=0.20, bottom=0.92, right=0.78),
                sparse_keypoints={},
            )

        self.assertEqual(len(result.detections), 2)
        self.assertEqual(result.detections[0].handedness, "left")
        self.assertGreater(result.detections[0].confidence, result.detections[1].confidence)

    def test_missing_model_path_fails_closed(self) -> None:
        worker = MediaPipeHandLandmarkWorker(
            config=HandLandmarkWorkerConfig(model_path="state/mediapipe/models/missing-hand.task")
        )

        with self.assertRaises(FileNotFoundError) as context:
            worker._ensure_hand_landmarker(
                {
                    "vision": SimpleNamespace(HandLandmarkerOptions=object, HandLandmarker=object),
                    "BaseOptions": object,
                }
            )

        self.assertIn("mediapipe_hand_landmarker_model_missing", str(context.exception))

    def test_analyze_normalizes_non_contiguous_roi_before_mediapipe_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "hand_landmarker.task"
            model_path.write_bytes(b"hand")
            worker = MediaPipeHandLandmarkWorker(
                config=HandLandmarkWorkerConfig(
                    model_path=str(model_path),
                    num_hands=1,
                    max_roi_candidates=1,
                )
            )
            worker._hand_landmarker = SimpleNamespace(
                detect_for_video=lambda image, timestamp_ms: SimpleNamespace(
                    hand_landmarks=[],
                    handedness=[],
                )
            )
            observed: list[tuple[bool | None, object]] = []
            runtime = {
                "mp": SimpleNamespace(
                    ImageFormat=SimpleNamespace(SRGB="srgb"),
                    Image=lambda image_format, data: (
                        observed.append((getattr(getattr(data, "flags", None), "c_contiguous", None), getattr(data, "dtype", None))),
                        SimpleNamespace(image_format=image_format, data=data),
                    )[1],
                ),
                "vision": SimpleNamespace(),
                "BaseOptions": object,
            }
            full_frame = _FakeFrame(120, 160)

            worker.analyze(
                runtime=runtime,
                frame_rgb=full_frame,
                timestamp_ms=1,
                primary_person_box=AICameraBox(top=0.10, left=0.20, bottom=0.92, right=0.78),
                sparse_keypoints={},
            )

        self.assertEqual(observed, [(True, "uint8")])

    def test_analyze_uses_monotonic_timestamps_across_roi_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "hand_landmarker.task"
            model_path.write_bytes(b"hand")
            worker = MediaPipeHandLandmarkWorker(
                config=HandLandmarkWorkerConfig(
                    model_path=str(model_path),
                    num_hands=1,
                    max_roi_candidates=3,
                )
            )
            timestamps: list[int] = []
            worker._hand_landmarker = SimpleNamespace(
                detect_for_video=lambda image, timestamp_ms: (
                    timestamps.append(timestamp_ms),
                    SimpleNamespace(hand_landmarks=[], handedness=[]),
                )[1]
            )
            runtime = {
                "mp": SimpleNamespace(
                    ImageFormat=SimpleNamespace(SRGB="srgb"),
                    Image=lambda image_format, data: SimpleNamespace(image_format=image_format, data=data),
                ),
                "vision": SimpleNamespace(),
                "BaseOptions": object,
            }

            result = worker.analyze(
                runtime=runtime,
                frame_rgb=_FakeFrame(120, 160),
                timestamp_ms=7,
                primary_person_box=AICameraBox(top=0.12, left=0.20, bottom=0.94, right=0.78),
                sparse_keypoints={
                    5: (0.34, 0.42, 0.90),
                    7: (0.28, 0.50, 0.88),
                    9: (0.18, 0.54, 0.93),
                    6: (0.62, 0.42, 0.91),
                    8: (0.71, 0.50, 0.87),
                    10: (0.83, 0.52, 0.95),
                },
            )

        self.assertEqual(timestamps, [7, 8, 9])
        self.assertEqual(result.final_timestamp_ms, 9)


if __name__ == "__main__":
    unittest.main()
