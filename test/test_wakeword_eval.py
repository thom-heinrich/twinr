from pathlib import Path
from struct import pack
import json
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.proactive.wakeword.evaluation import (
    append_wakeword_capture_label,
    autotune_wakeword_profile,
    load_labeled_ops_captures,
    run_wakeword_eval,
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
