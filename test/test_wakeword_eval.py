from pathlib import Path
from struct import pack
from types import ModuleType
import json
import sys
import tempfile
import unittest
import wave
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.proactive.wakeword.calibration import WakewordCalibrationProfile, apply_wakeword_calibration
from twinr.proactive.wakeword.evaluation import (
    append_wakeword_capture_label,
    autotune_wakeword_profile,
    load_labeled_ops_captures,
    load_eval_manifest,
    run_wakeword_eval,
    train_wakeword_custom_verifier_from_manifest,
)


def _write_wav(path: Path, *, amplitude: int, sample_count: int = 1600) -> None:
    frames = b"".join(pack("<h", amplitude) for _ in range(sample_count))
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(frames)


class FakeAdaptiveModel:
    def __init__(self, *, wakeword_models) -> None:
        self.models = {str(name): object() for name in wakeword_models}

    def reset(self) -> None:
        return None

    def predict_clip(self, samples, *, padding=1, chunk_size=1280):
        del padding, chunk_size
        peak = max(abs(int(sample)) for sample in samples) if len(samples) else 0
        score = 0.12 if peak >= 900 else 0.03
        return [{"twinr": score}]


def _model_factory(**kwargs):
    return FakeAdaptiveModel(wakeword_models=kwargs.get("wakeword_models", ("twinr",)))


class WakewordEvalTests(unittest.TestCase):
    def test_load_eval_manifest_accepts_json_array_and_prefers_captured_audio_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            generated = root / "generated.wav"
            captured = root / "captured.wav"
            manifest = root / "captured_manifest.json"
            _write_wav(generated, amplitude=200)
            _write_wav(captured, amplitude=1200)
            manifest.write_text(
                json.dumps(
                    [
                        {
                            "audio_path": generated.name,
                            "captured_audio_path": captured.name,
                            "label": "correct",
                            "notes": "room replay",
                        }
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            entries = load_eval_manifest(manifest)

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].audio_path, captured)
        self.assertEqual(entries[0].notes, "room replay")

    def test_apply_wakeword_calibration_can_override_stt_phrases_only(self) -> None:
        base_config = TwinrConfig(
            wakeword_enabled=True,
            wakeword_phrases=("hallo twinr", "twinr"),
            wakeword_stt_phrases=("hallo twinr", "twinr"),
        )

        calibrated = apply_wakeword_calibration(
            base_config,
            WakewordCalibrationProfile(
                stt_phrases=("hallo twinr", "hallo twin", "twinr", "twin"),
            ),
        )

        self.assertEqual(calibrated.wakeword_phrases, ("hallo twinr", "twinr"))
        self.assertEqual(
            calibrated.wakeword_stt_phrases,
            ("hallo twinr", "hallo twin", "twinr", "twin"),
        )

    def test_append_wakeword_capture_label_roundtrips_through_ops_log(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            capture_path = Path(temp_dir) / "capture.wav"
            _write_wav(capture_path, amplitude=1200)

            append_wakeword_capture_label(
                config,
                capture_path=capture_path,
                label="false_positive",
                notes="tv in background",
            )
            entries = load_labeled_ops_captures(config)

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].audio_path, capture_path)
        self.assertEqual(entries[0].label, "false_positive")
        self.assertEqual(entries[0].notes, "tv in background")

    def test_train_wakeword_custom_verifier_uses_captured_room_clips(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "captured_manifest.json"
            generated_positive = root / "generated_positive.wav"
            generated_negative = root / "generated_negative.wav"
            _write_wav(generated_positive, amplitude=100)
            _write_wav(generated_negative, amplitude=100, sample_count=1600)
            positive_paths = []
            negative_paths = []
            manifest_items: list[dict[str, str]] = []
            for index in range(3):
                captured_positive = root / f"captured_positive_{index}.wav"
                _write_wav(captured_positive, amplitude=1200, sample_count=16000)
                positive_paths.append(captured_positive)
                manifest_items.append(
                    {
                        "audio_path": generated_positive.name,
                        "captured_audio_path": captured_positive.name,
                        "label": "correct",
                    }
                )
            for index in range(2):
                captured_negative = root / f"captured_negative_{index}.wav"
                _write_wav(captured_negative, amplitude=400, sample_count=16000 * 6)
                negative_paths.append(captured_negative)
                manifest_items.append(
                    {
                        "audio_path": generated_negative.name,
                        "captured_audio_path": captured_negative.name,
                        "label": "false_positive",
                    }
                )
            manifest.write_text(json.dumps(manifest_items) + "\n", encoding="utf-8")
            output_path = root / "trained.verifier.pkl"
            calls: list[dict[str, object]] = []
            fake_openwakeword = ModuleType("openwakeword")

            def _train_custom_verifier(**kwargs):
                calls.append(kwargs)

            fake_openwakeword.train_custom_verifier = _train_custom_verifier

            with patch.dict(sys.modules, {"openwakeword": fake_openwakeword}):
                report = train_wakeword_custom_verifier_from_manifest(
                    manifest_path=manifest,
                    output_path=output_path,
                    model_name="twinr_v1",
                    inference_framework="onnx",
                )

        self.assertEqual(len(calls), 1)
        self.assertEqual(
            calls[0]["positive_reference_clips"],
            [str(path) for path in positive_paths],
        )
        self.assertEqual(
            calls[0]["negative_reference_clips"],
            [str(path) for path in negative_paths],
        )
        self.assertEqual(calls[0]["output_path"], str(output_path))
        self.assertEqual(calls[0]["model_name"], "twinr_v1")
        self.assertEqual(calls[0]["inference_framework"], "onnx")
        self.assertEqual(report.positive_clips, 3)
        self.assertEqual(report.negative_clips, 2)
        self.assertGreaterEqual(report.negative_seconds, 12.0)

    def test_train_wakeword_custom_verifier_requires_minimum_reference_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "captured_manifest.json"
            positive = root / "captured_positive.wav"
            negative = root / "captured_negative.wav"
            _write_wav(positive, amplitude=1200, sample_count=16000)
            _write_wav(negative, amplitude=300, sample_count=16000 * 4)
            manifest.write_text(
                json.dumps(
                    [
                        {
                            "captured_audio_path": positive.name,
                            "label": "correct",
                        },
                        {
                            "captured_audio_path": negative.name,
                            "label": "false_positive",
                        },
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "at least 3 positive clips"):
                train_wakeword_custom_verifier_from_manifest(
                    manifest_path=manifest,
                    output_path=root / "trained.verifier.pkl",
                    model_name="twinr_v1",
                )

    def test_run_wakeword_eval_writes_report_for_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            positive = root / "wakeword_positive.wav"
            negative = root / "wakeword_negative.wav"
            manifest = root / "manifest.jsonl"
            _write_wav(positive, amplitude=1200)
            _write_wav(negative, amplitude=100)
            manifest.write_text(
                "\n".join(
                    (
                        json.dumps({"audio_path": positive.name, "label": "correct"}),
                        json.dumps({"audio_path": negative.name, "label": "false_positive"}),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            config = TwinrConfig(
                project_root=temp_dir,
                wakeword_enabled=True,
                wakeword_primary_backend="openwakeword",
                wakeword_fallback_backend="stt",
                wakeword_verifier_mode="disabled",
                wakeword_phrases=("twinr",),
                wakeword_openwakeword_models=("twinr",),
                wakeword_openwakeword_threshold=0.08,
                wakeword_openwakeword_patience_frames=1,
                wakeword_openwakeword_activation_samples=1,
            )

            report = run_wakeword_eval(
                config=config,
                manifest_path=manifest,
                backend=None,
                model_factory=_model_factory,
            )

            self.assertIsNotNone(report.report_path)
            self.assertTrue(report.report_path.exists())

        self.assertEqual(report.evaluated_entries, 2)
        self.assertEqual(report.metrics.true_positive, 1)
        self.assertEqual(report.metrics.true_negative, 1)
        self.assertEqual(report.metrics.false_positive, 0)
        self.assertEqual(report.metrics.false_negative, 0)

    def test_autotune_writes_recommended_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            positive = root / "wakeword_positive.wav"
            negative = root / "wakeword_negative.wav"
            manifest = root / "manifest.jsonl"
            _write_wav(positive, amplitude=1200)
            _write_wav(negative, amplitude=100)
            manifest.write_text(
                "\n".join(
                    (
                        json.dumps({"audio_path": positive.name, "label": "correct"}),
                        json.dumps({"audio_path": negative.name, "label": "false_positive"}),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            config = TwinrConfig(
                project_root=temp_dir,
                wakeword_enabled=True,
                wakeword_primary_backend="openwakeword",
                wakeword_fallback_backend="stt",
                wakeword_verifier_mode="disabled",
                wakeword_phrases=("twinr",),
                wakeword_openwakeword_models=("twinr",),
                wakeword_openwakeword_threshold=0.01,
                wakeword_openwakeword_patience_frames=1,
                wakeword_openwakeword_activation_samples=1,
            )

            recommendation = autotune_wakeword_profile(
                config=config,
                manifest_path=manifest,
                backend=None,
                model_factory=_model_factory,
            )

            self.assertIsNotNone(recommendation.profile_path)
            self.assertTrue(recommendation.profile_path.exists())

        self.assertGreaterEqual(recommendation.score, 0.0)
        self.assertGreaterEqual(recommendation.profile.threshold, 0.03)


if __name__ == "__main__":
    unittest.main()
