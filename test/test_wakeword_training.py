from pathlib import Path
from struct import pack
import json
import tempfile
import unittest
from unittest import mock
import wave

import numpy as np

import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.proactive.wakeword.evaluation import WakewordEvalMetrics
from twinr.proactive.wakeword import training as wakeword_training
from twinr.proactive.wakeword.training import train_wakeword_base_model_from_dataset_root


def _write_wav(path: Path, *, amplitude: int, sample_count: int = 16000) -> None:
    frames = b"".join(pack("<h", amplitude) for _ in range(sample_count))
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(frames)


class WakewordTrainingTests(unittest.TestCase):
    def test_expanded_sample_weights_follow_dataset_provenance(self) -> None:
        weights = wakeword_training._expanded_sample_weights(
            audio_paths=[
                Path("synthetic.wav"),
                Path("extra_pos_room.wav"),
            ],
            rounds=2,
            positive=True,
        )
        negative_weights = wakeword_training._expanded_sample_weights(
            audio_paths=[
                Path("synthetic_neg.wav"),
                Path("extra_neg_room.wav"),
                Path("mined_neg_confusion.wav"),
            ],
            rounds=1,
            positive=False,
        )

        self.assertEqual(weights.tolist(), [1.0, 2.0, 1.0, 2.0])
        self.assertEqual(negative_weights.tolist(), [1.5, 3.0, 6.0])

    def test_export_openwakeword_model_to_onnx_forces_single_file_export(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_model = root / "twinr_v2.onnx"
            sidecar_path = output_model.with_suffix(output_model.suffix + ".data")
            calls: dict[str, object] = {}

            class _FakeModel:
                def to(self, _device: str):
                    return self

            class _FakeInitializer:
                def __init__(self, data_location: int) -> None:
                    self.data_location = data_location

            external_model = types.SimpleNamespace(
                graph=types.SimpleNamespace(initializer=[_FakeInitializer(1)])
            )
            embedded_model = types.SimpleNamespace(
                graph=types.SimpleNamespace(initializer=[_FakeInitializer(0)])
            )

            def _fake_export(_model, _sample, output_path: str, **kwargs) -> None:
                calls["export_kwargs"] = kwargs
                Path(output_path).write_bytes(b"onnx-external")
                sidecar_path.write_bytes(b"weights")

            fake_torch = types.SimpleNamespace(
                rand=lambda shape: np.zeros(shape, dtype=np.float32),
                onnx=types.SimpleNamespace(export=_fake_export),
            )

            def _fake_save_model(_model, path, *, save_as_external_data=False) -> None:
                calls.setdefault("save_model", []).append(
                    {
                        "path": str(path),
                        "save_as_external_data": bool(save_as_external_data),
                    }
                )
                Path(path).write_bytes(b"onnx-embedded")

            fake_onnx = types.SimpleNamespace(
                TensorProto=types.SimpleNamespace(EXTERNAL=1),
                load=lambda _path, *, load_external_data=False: (
                    embedded_model if load_external_data else external_model
                ),
                save_model=_fake_save_model,
            )

            with mock.patch.dict(sys.modules, {"torch": fake_torch, "onnx": fake_onnx}):
                wakeword_training._export_openwakeword_model_to_onnx(
                    model=_FakeModel(),
                    input_shape=(16, 96),
                    output_model_path=output_model,
                    model_name="twinr_v2",
                )

            self.assertTrue(output_model.exists())
            self.assertFalse(sidecar_path.exists())
            self.assertEqual(
                calls["export_kwargs"],
                {
                    "output_names": ["twinr_v2"],
                    "opset_version": 13,
                    "dynamo": False,
                    "external_data": False,
                },
            )
            self.assertEqual(
                calls["save_model"],
                [
                    {
                        "path": str(output_model),
                        "save_as_external_data": False,
                    }
                ],
            )

    def test_train_wakeword_base_model_from_dataset_root_writes_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "dataset"
            for split_name, amplitude in (
                ("positive_train", 1200),
                ("negative_train", 300),
                ("positive_test", 1000),
                ("negative_test", 200),
            ):
                split_root = dataset_root / split_name
                split_root.mkdir(parents=True, exist_ok=True)
                for index in range(2):
                    _write_wav(split_root / f"{split_name}_{index}.wav", amplitude=amplitude)

            acceptance_positive = root / "acceptance_positive.wav"
            acceptance_negative = root / "acceptance_negative.wav"
            _write_wav(acceptance_positive, amplitude=1300)
            _write_wav(acceptance_negative, amplitude=150)
            acceptance_manifest = root / "captured_manifest.json"
            acceptance_manifest.write_text(
                json.dumps(
                    [
                        {"captured_audio_path": str(acceptance_positive), "label": "correct"},
                        {"captured_audio_path": str(acceptance_negative), "label": "false_positive"},
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            output_model = root / "twinr_v2.onnx"
            output_metadata = root / "twinr_v2.metadata.json"
            calls: dict[str, object] = {}

            def _fake_train_backend(**kwargs) -> None:
                calls.update(kwargs)
                Path(kwargs["output_model_path"]).write_bytes(b"fake-onnx-model")

            def _fake_acceptance_evaluator(**kwargs):
                self.assertEqual(Path(kwargs["manifest_path"]), acceptance_manifest)
                return (
                    0.12,
                    WakewordEvalMetrics(
                        total=2,
                        true_positive=1,
                        false_positive=0,
                        true_negative=1,
                        false_negative=0,
                        precision=1.0,
                        recall=1.0,
                        false_positive_rate=0.0,
                        false_negative_rate=0.0,
                    ),
                )

            report = train_wakeword_base_model_from_dataset_root(
                dataset_root=dataset_root,
                output_model_path=output_model,
                metadata_path=output_metadata,
                acceptance_manifest=acceptance_manifest,
                training_rounds=3,
                evaluation_config=TwinrConfig(
                    wakeword_primary_backend="openwakeword",
                    wakeword_fallback_backend="stt",
                    wakeword_verifier_mode="disabled",
                ),
                train_backend=_fake_train_backend,
                acceptance_evaluator=_fake_acceptance_evaluator,
            )

            metadata = json.loads(output_metadata.read_text(encoding="utf-8"))
            positive_train_features = np.load(Path(calls["positive_train_features_path"]))
            negative_train_features = np.load(Path(calls["negative_train_features_path"]))
            output_exists = output_model.exists()

        self.assertTrue(output_exists)
        self.assertEqual(report.selected_threshold, 0.12)
        self.assertEqual(metadata["selected_threshold"], 0.12)
        self.assertEqual(metadata["model_type"], "mlp")
        self.assertEqual(metadata["acceptance_eval_mode"], "runtime_stream_replay")
        self.assertEqual(metadata["train_positive_clips"], 2)
        self.assertEqual(metadata["train_negative_clips"], 2)
        self.assertEqual(positive_train_features.shape[0], 6)
        self.assertEqual(negative_train_features.shape[0], 6)
        self.assertIn("acceptance_metrics", metadata)

    def test_train_sklearn_mlp_model_exports_twinr_style_onnx(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rng = np.random.default_rng(1234)
            positive_train = root / "positive_train.npy"
            negative_train = root / "negative_train.npy"
            positive_validation = root / "positive_validation.npy"
            negative_validation = root / "negative_validation.npy"
            output_model = root / "twinr_mlp.onnx"
            np.save(positive_train, rng.normal(loc=0.8, scale=0.2, size=(24, 16, 96)).astype(np.float32))
            np.save(negative_train, rng.normal(loc=-0.8, scale=0.2, size=(24, 16, 96)).astype(np.float32))
            np.save(positive_validation, rng.normal(loc=0.8, scale=0.2, size=(8, 16, 96)).astype(np.float32))
            np.save(negative_validation, rng.normal(loc=-0.8, scale=0.2, size=(8, 16, 96)).astype(np.float32))

            wakeword_training._train_sklearn_mlp_model(
                positive_train_features_path=positive_train,
                negative_train_features_path=negative_train,
                positive_validation_features_path=positive_validation,
                negative_validation_features_path=negative_validation,
                output_model_path=output_model,
                model_name="twinr_mlp",
                layer_dim=64,
                steps=500,
                seed=1234,
            )

            import onnx

            model = onnx.load(str(output_model))
            output_exists = output_model.exists()

        self.assertTrue(output_exists)
        self.assertEqual(model.graph.output[0].name, "twinr_mlp")
        self.assertEqual(model.graph.node[0].op_type, "Reshape")
        self.assertIn("Gemm", {node.op_type for node in model.graph.node})
        self.assertEqual(len(model.graph.initializer), 9)
