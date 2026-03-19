from pathlib import Path
import importlib.util
import sys
import tempfile
import unittest


PIAICAM_DIR = Path(__file__).resolve().parents[1] / "hardware" / "piaicam"
if str(PIAICAM_DIR) not in sys.path:
    sys.path.insert(0, str(PIAICAM_DIR))


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, PIAICAM_DIR / filename)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


workflow = _load_module("twinr_custom_gesture_workflow", "custom_gesture_workflow.py")
capture_script = _load_module("twinr_capture_custom_gesture_dataset", "capture_custom_gesture_dataset.py")
train_script = _load_module("twinr_train_custom_gesture_model", "train_custom_gesture_model.py")


class _FakeCamera:
    def __init__(self) -> None:
        self.configuration = None
        self.started = False
        self.stopped = False
        self.closed = False
        self.files: list[str] = []

    def create_still_configuration(self, *, main):
        self.configuration = {"main": main}
        return self.configuration

    def configure(self, configuration):
        self.configuration = configuration

    def start(self) -> None:
        self.started = True

    def capture_file(self, path: str) -> None:
        self.files.append(path)
        Path(path).write_bytes(b"jpeg")

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


class _StubSplit:
    def __init__(self, size: int) -> None:
        self.size = size


class _StubDataset:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.label_names = ["middle_finger", "none", "ok_sign"]
        self.num_classes = 3
        self.fraction = None

    def split(self, fraction: float):
        self.fraction = fraction
        return _StubSplit(8), _StubSplit(2)


class _StubRecognizer:
    def __init__(self, export_dir: Path) -> None:
        self.export_dir = export_dir

    def evaluate(self, dataset) -> dict[str, float]:
        return {"loss": 0.12, "accuracy": 0.91}

    def export_model(self, model_name: str = "gesture_recognizer.task") -> None:
        (self.export_dir / model_name).write_bytes(b"task")

    def export_labels(self, export_dir: str, label_filename: str = "labels.txt") -> None:
        Path(export_dir).mkdir(parents=True, exist_ok=True)
        (Path(export_dir) / label_filename).write_text("none\nok_sign\nmiddle_finger\n", encoding="utf-8")


class _StubGestureRecognizerAPI:
    last_dataset = None

    class HParams:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class GestureRecognizerOptions:
        def __init__(self, *, hparams=None) -> None:
            self.hparams = hparams

    class Dataset:
        @staticmethod
        def from_folder(dirname: str):
            dataset = _StubDataset(Path(dirname))
            _StubGestureRecognizerAPI.last_dataset = dataset
            return dataset

    class GestureRecognizer:
        @staticmethod
        def create(*, train_data, validation_data, options):
            export_dir = Path(options.hparams.kwargs["export_dir"])
            export_dir.mkdir(parents=True, exist_ok=True)
            return _StubRecognizer(export_dir)


class _StubGestureRecognizerAPIRejectingHands:
    class Dataset:
        @staticmethod
        def from_folder(dirname: str):
            raise ValueError("No valid hand is detected.")


class PIAICustomGestureWorkflowTests(unittest.TestCase):
    def test_collect_dataset_manifest_requires_none_label(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir)
            (dataset_root / "ok_sign").mkdir()
            (dataset_root / "ok_sign" / "sample.jpg").write_bytes(b"jpeg")
            (dataset_root / "middle_finger").mkdir()
            (dataset_root / "middle_finger" / "sample.jpg").write_bytes(b"jpeg")

            with self.assertRaises(ValueError) as context:
                workflow.collect_dataset_manifest(dataset_root)

        self.assertIn("custom_gesture_dataset_missing_labels:none", str(context.exception))

    def test_plan_capture_targets_normalizes_label_name(self) -> None:
        normalized_label, label_dir, targets = workflow.plan_capture_targets(
            Path("state/mediapipe/custom_gesture_dataset"),
            label="OK Sign",
            count=2,
            timestamp_slug="20260319T100000Z",
        )

        self.assertEqual(normalized_label, "ok_sign")
        self.assertEqual(label_dir.as_posix(), "state/mediapipe/custom_gesture_dataset/ok_sign")
        self.assertEqual(
            [path.as_posix() for path in targets],
            [
                "state/mediapipe/custom_gesture_dataset/ok_sign/ok_sign-20260319T100000Z-0001.jpg",
                "state/mediapipe/custom_gesture_dataset/ok_sign/ok_sign-20260319T100000Z-0002.jpg",
            ],
        )

    def test_capture_dataset_writes_requested_files(self) -> None:
        fake_camera = _FakeCamera()
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = capture_script.capture_dataset(
                dataset_root=Path(temp_dir),
                label="middle finger",
                count=2,
                interval_s=0.0,
                warmup_s=0.0,
                width=640,
                height=480,
                camera_factory=lambda: fake_camera,
                sleep_fn=lambda _: None,
            )

            self.assertEqual(summary["label"], "middle_finger")
            self.assertEqual(summary["count"], 2)
            self.assertEqual(len(summary["files"]), 2)
            self.assertTrue(fake_camera.started)
            self.assertTrue(fake_camera.stopped)
            self.assertTrue(fake_camera.closed)
            for path in summary["files"]:
                self.assertTrue(Path(path).exists())

    def test_train_custom_gesture_model_dry_run_validates_dataset_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir) / "dataset"
            for label in ("none", "ok_sign", "middle_finger"):
                label_dir = dataset_root / label
                label_dir.mkdir(parents=True)
                (label_dir / "sample.jpg").write_bytes(b"jpeg")
            output_dir = Path(temp_dir) / "runs" / "20260319T100000Z"
            runtime_dir = Path(temp_dir) / "runtime"

            summary = train_script.train_custom_gesture_model(
                train_script.TrainingConfig(
                    dataset_root=dataset_root,
                    output_dir=output_dir,
                    runtime_model_dir=runtime_dir,
                    model_name="custom_gesture.task",
                    runtime_model_name="custom_gesture.task",
                    validation_split=0.2,
                    epochs=4,
                    batch_size=2,
                    learning_rate=0.001,
                    min_images_per_label=1,
                    required_labels=("none", "ok_sign", "middle_finger"),
                    dry_run=True,
                ),
                api_loader=lambda: (_ for _ in ()).throw(AssertionError("api_loader should not run for dry_run")),
            )

        self.assertEqual(summary["status"], "dry_run")
        self.assertEqual(summary["dataset"]["total_images"], 3)
        self.assertEqual(
            summary["runtime_env_hint"],
            "TWINR_PROACTIVE_LOCAL_CAMERA_MEDIAPIPE_CUSTOM_GESTURE_MODEL_PATH="
            f"{runtime_dir.as_posix()}/custom_gesture.task",
        )

    def test_train_custom_gesture_model_exports_and_copies_task(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir) / "dataset"
            for label in ("none", "ok_sign", "middle_finger"):
                label_dir = dataset_root / label
                label_dir.mkdir(parents=True)
                (label_dir / "sample.jpg").write_bytes(b"jpeg")
            output_dir = Path(temp_dir) / "runs" / "20260319T100000Z"
            runtime_dir = Path(temp_dir) / "runtime"

            summary = train_script.train_custom_gesture_model(
                train_script.TrainingConfig(
                    dataset_root=dataset_root,
                    output_dir=output_dir,
                    runtime_model_dir=runtime_dir,
                    model_name="custom_gesture.task",
                    runtime_model_name="custom_gesture.task",
                    validation_split=0.2,
                    epochs=4,
                    batch_size=2,
                    learning_rate=0.001,
                    min_images_per_label=1,
                    required_labels=("none", "ok_sign", "middle_finger"),
                ),
                api_loader=lambda: _StubGestureRecognizerAPI,
            )

            self.assertEqual(summary["status"], "trained")
            self.assertEqual(summary["dataset"]["training_size"], 8)
            self.assertEqual(summary["dataset"]["validation_size"], 2)
            self.assertEqual(summary["evaluation"]["accuracy"], 0.91)
            self.assertAlmostEqual(_StubGestureRecognizerAPI.last_dataset.fraction or 0.0, 0.8, places=3)
            self.assertTrue((output_dir / "custom_gesture.task").exists())
            self.assertTrue((runtime_dir / "custom_gesture.task").exists())
            self.assertTrue((output_dir / "labels.txt").exists())

    def test_train_custom_gesture_model_maps_missing_hand_dataset_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir) / "dataset"
            for label in ("none", "ok_sign", "middle_finger"):
                label_dir = dataset_root / label
                label_dir.mkdir(parents=True)
                (label_dir / "sample.jpg").write_bytes(b"jpeg")
            output_dir = Path(temp_dir) / "runs" / "20260319T100000Z"
            runtime_dir = Path(temp_dir) / "runtime"

            with self.assertRaises(RuntimeError) as context:
                train_script.train_custom_gesture_model(
                    train_script.TrainingConfig(
                        dataset_root=dataset_root,
                        output_dir=output_dir,
                        runtime_model_dir=runtime_dir,
                        model_name="custom_gesture.task",
                        runtime_model_name="custom_gesture.task",
                        validation_split=0.2,
                        epochs=4,
                        batch_size=2,
                        learning_rate=0.001,
                        min_images_per_label=1,
                        required_labels=("none", "ok_sign", "middle_finger"),
                    ),
                    api_loader=lambda: _StubGestureRecognizerAPIRejectingHands,
                )

        self.assertIn("custom_gesture_dataset_no_detectable_hands", str(context.exception))

    def test_load_gesture_recognizer_api_maps_pkg_resources_failure(self) -> None:
        original_import = __import__

        def failing_import(name, *args, **kwargs):
            if name == "pkg_resources":
                raise ModuleNotFoundError("No module named 'pkg_resources'")
            return original_import(name, *args, **kwargs)

        import builtins

        builtins_import = builtins.__import__
        builtins.__import__ = failing_import
        try:
            with self.assertRaises(RuntimeError) as context:
                train_script._load_gesture_recognizer_api()
        finally:
            builtins.__import__ = builtins_import

        self.assertIn("mediapipe_model_maker_requires_setuptools_lt_81", str(context.exception))


if __name__ == "__main__":
    unittest.main()
