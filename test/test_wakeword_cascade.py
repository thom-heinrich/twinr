from pathlib import Path
from struct import pack
from types import ModuleType
import json
import sys
import tempfile
import unittest
import wave
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.audio import AmbientAudioCaptureWindow, AmbientAudioLevelSample
from twinr.proactive.wakeword import WakewordMatch
from twinr.proactive.wakeword.cascade import (
    WakewordSequenceCaptureVerifier,
    _SequenceExample,
    _load_manifest_entries,
    _training_sample_weight,
    train_wakeword_sequence_verifier_from_manifest,
)


def _write_wav(path: Path, *, amplitude: int, sample_count: int = 16000) -> None:
    frames = b"".join(pack("<h", amplitude) for _ in range(sample_count))
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(frames)


def _capture(amplitude: int, *, sample_count: int = 16000) -> AmbientAudioCaptureWindow:
    pcm_bytes = b"".join(pack("<h", amplitude) for _ in range(sample_count))
    return AmbientAudioCaptureWindow(
        sample=AmbientAudioLevelSample(
            duration_ms=int((sample_count / 16000) * 1000),
            chunk_count=8,
            active_chunk_count=4 if amplitude >= 1000 else 1,
            average_rms=abs(amplitude),
            peak_rms=abs(amplitude),
            active_ratio=0.5 if amplitude >= 1000 else 0.1,
        ),
        pcm_bytes=pcm_bytes,
        sample_rate=16000,
        channels=1,
    )


class FakeAudioFeatures:
    def __init__(self, *, inference_framework: str, device: str, ncpu: int) -> None:
        del inference_framework, device, ncpu

    def embed_clips(self, clips, *, batch_size: int, ncpu: int):
        del batch_size, ncpu
        rows = []
        for clip in clips:
            peak = int(np.max(np.abs(clip))) if len(clip) else 0
            base = 2.0 if peak >= 1000 else 0.1
            rows.append(
                np.asarray(
                    [
                        [base, base + 0.1, base + 0.2],
                        [base + 0.2, base + 0.3, base + 0.4],
                        [base + 0.4, base + 0.5, base + 0.6],
                        [base + 0.6, base + 0.7, base + 0.8],
                    ],
                    dtype=np.float32,
                )
            )
        return np.stack(rows, axis=0)


class FakeOpenWakeWordModel:
    def __init__(self, *, wakeword_models, inference_framework: str) -> None:
        del inference_framework
        raw_name = str(wakeword_models[0])
        self.label = Path(raw_name).stem if raw_name.endswith((".onnx", ".tflite")) else raw_name

    def reset(self) -> None:
        return None

    def predict_clip(self, samples, *, padding: int = 1, chunk_size: int = 1280):
        del padding, chunk_size
        peak = int(np.max(np.abs(samples))) if len(samples) else 0
        high = 0.92 if peak >= 1000 else 0.08
        return [
            {self.label: high},
            {self.label: high},
            {self.label: high},
            {self.label: high},
        ]


class WakewordCascadeTests(unittest.TestCase):
    def test_load_manifest_entries_extracts_canonical_wakeword_family_from_phrase_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "captured_manifest.json"
            manifest.write_text(
                json.dumps(
                    [
                        {
                            "captured_audio_path": str(root / "positive.wav"),
                            "label": "positive",
                            "text": "Twinna wie ist das Wetter heute",
                        },
                        {
                            "captured_audio_path": str(root / "negative.wav"),
                            "label": "negative",
                            "text": "Wie ist das Wetter heute",
                        },
                        {
                            "captured_audio_path": str(root / "winner.wav"),
                            "label": "negative",
                            "text": "Hallo Winner",
                        },
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            entries = _load_manifest_entries(manifest)

        self.assertEqual([entry.family_key for entry in entries], ["twinna", "wie_ist_das_wetter_heute", "winner"])

    def test_training_sample_weight_prioritizes_focus_families(self) -> None:
        background_positive = _SequenceExample(
            index=0,
            expected_detected=True,
            family_key="twinr",
            sequence_matrix=np.ones((4, 3), dtype=np.float32),
            score_summary=np.asarray([0.8], dtype=np.float32),
        )
        focused_positive = _SequenceExample(
            index=1,
            expected_detected=True,
            family_key="twinna",
            sequence_matrix=np.ones((4, 3), dtype=np.float32),
            score_summary=np.asarray([0.8], dtype=np.float32),
        )
        background_negative = _SequenceExample(
            index=2,
            expected_detected=False,
            family_key="wie_ist_das_wetter_heute",
            sequence_matrix=np.ones((4, 3), dtype=np.float32),
            score_summary=np.asarray([0.8], dtype=np.float32),
        )
        focused_negative = _SequenceExample(
            index=3,
            expected_detected=False,
            family_key="winner",
            sequence_matrix=np.ones((4, 3), dtype=np.float32),
            score_summary=np.asarray([0.8], dtype=np.float32),
        )

        focus_negative_families = ("winner", "twin", "winter", "tina", "timer", "twitter")

        self.assertGreater(
            _training_sample_weight(
                focused_positive,
                focus_negative_families=focus_negative_families,
            ),
            _training_sample_weight(
                background_positive,
                focus_negative_families=focus_negative_families,
            ),
        )
        self.assertGreater(
            _training_sample_weight(
                focused_negative,
                focus_negative_families=focus_negative_families,
            ),
            _training_sample_weight(
                background_negative,
                focus_negative_families=focus_negative_families,
            ),
        )

    def test_train_sequence_verifier_and_verify_positive_vs_negative_captures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "captured_manifest.json"
            sequence_asset = root / "twinr_v2.sequence_verifier.pkl"
            primary_model = root / "twinr_v2.onnx"
            primary_model.write_bytes(b"fake-model")
            manifest_items = []
            for index in range(3):
                positive = root / f"positive_{index}.wav"
                _write_wav(positive, amplitude=1600, sample_count=16000)
                manifest_items.append(
                    {
                        "captured_audio_path": str(positive),
                        "label": "correct",
                    }
                )
            for index in range(2):
                negative = root / f"negative_{index}.wav"
                _write_wav(negative, amplitude=120, sample_count=16000 * 6)
                manifest_items.append(
                    {
                        "captured_audio_path": str(negative),
                        "label": "false_positive",
                    }
                )
            manifest.write_text(json.dumps(manifest_items) + "\n", encoding="utf-8")

            fake_openwakeword_model = ModuleType("openwakeword.model")
            fake_openwakeword_utils = ModuleType("openwakeword.utils")
            fake_openwakeword_model.Model = FakeOpenWakeWordModel
            fake_openwakeword_utils.AudioFeatures = FakeAudioFeatures

            with patch.dict(
                sys.modules,
                {
                    "openwakeword.model": fake_openwakeword_model,
                    "openwakeword.utils": fake_openwakeword_utils,
                },
            ):
                report = train_wakeword_sequence_verifier_from_manifest(
                    manifest_path=manifest,
                    output_path=sequence_asset,
                    model_name=str(primary_model),
                    inference_framework="onnx",
                )
                verifier = WakewordSequenceCaptureVerifier(
                    verifier_models={"twinr_v2": str(sequence_asset)},
                    threshold=0.5,
                )
                positive_verification = verifier.verify(
                    _capture(1800),
                    detector_match=WakewordMatch(
                        detected=True,
                        transcript="",
                        backend="openwakeword",
                        detector_label="twinr_v2",
                        matched_phrase="twinr",
                        score=0.91,
                    ),
                )
                negative_verification = verifier.verify(
                    _capture(90),
                    detector_match=WakewordMatch(
                        detected=True,
                        transcript="",
                        backend="openwakeword",
                        detector_label="twinr_v2",
                        matched_phrase="twinr",
                        score=0.91,
                    ),
                )
                output_exists = sequence_asset.exists()

        self.assertEqual(report.positive_clips, 3)
        self.assertEqual(report.negative_clips, 2)
        self.assertGreaterEqual(report.negative_seconds, 12.0)
        self.assertTrue(output_exists)
        self.assertEqual(positive_verification.status, "accepted")
        self.assertEqual(negative_verification.status, "rejected")

    def test_sequence_capture_verifier_skips_unconfigured_detector_label(self) -> None:
        verifier = WakewordSequenceCaptureVerifier(verifier_models={}, threshold=0.5)

        result = verifier.verify(
            _capture(1800),
            detector_match=WakewordMatch(
                detected=True,
                transcript="",
                backend="openwakeword",
                detector_label="other_model",
                matched_phrase="twinr",
                score=0.8,
            ),
        )

        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.reason, "detector_label_unconfigured")


if __name__ == "__main__":
    unittest.main()
