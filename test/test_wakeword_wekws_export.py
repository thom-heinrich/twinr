from pathlib import Path
import json
import sys
import tempfile
import wave
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.proactive.wakeword.wekws_export import export_wakeword_manifests_to_wekws


def _write_test_wav(path: Path, *, frame_count: int = 1600, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x01\x00" * frame_count)


class WekwsExportTests(unittest.TestCase):
    def test_export_wakeword_manifests_to_wekws_writes_expected_split_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_positive = root / "audio" / "train_positive.wav"
            train_negative = root / "audio" / "train_negative.wav"
            dev_positive = root / "audio" / "dev_positive.wav"
            ignored_clip = root / "audio" / "ignored.wav"
            for clip in (train_positive, train_negative, dev_positive, ignored_clip):
                _write_test_wav(clip)
            train_manifest = root / "train.json"
            dev_manifest = root / "dev.json"
            train_manifest.write_text(
                json.dumps(
                    [
                        {"captured_audio_path": str(train_positive), "label": "positive"},
                        {"captured_audio_path": str(train_negative), "label": "negative"},
                        {"captured_audio_path": str(ignored_clip), "label": "unclear"},
                    ]
                ),
                encoding="utf-8",
            )
            dev_manifest.write_text(
                json.dumps(
                    [
                        {"captured_audio_path": str(dev_positive), "label": "correct"},
                    ]
                ),
                encoding="utf-8",
            )

            report = export_wakeword_manifests_to_wekws(
                output_dir=root / "wekws",
                train_manifest=train_manifest,
                dev_manifest=dev_manifest,
                positive_token="twinr_family",
                filler_token="filler",
            )

            self.assertEqual(report.positive_token, "<TWINR_FAMILY>")
            self.assertEqual(report.filler_token, "<FILLER>")
            self.assertEqual(report.dict_path.read_text(encoding="utf-8"), "<FILLER> -1\n<TWINR_FAMILY> 0\n")
            self.assertEqual(report.words_path.read_text(encoding="utf-8"), "<FILLER>\n<TWINR_FAMILY>\n")
            train_wav_scp = (report.output_dir / "train" / "wav.scp").read_text(encoding="utf-8").splitlines()
            train_text = (report.output_dir / "train" / "text").read_text(encoding="utf-8").splitlines()
            train_utt2spk = (report.output_dir / "train" / "utt2spk").read_text(encoding="utf-8").splitlines()
            train_wav_dur = (report.output_dir / "train" / "wav.dur").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(train_wav_scp), 2)
            self.assertEqual(len(train_text), 2)
            self.assertEqual(len(train_utt2spk), 2)
            self.assertEqual(len(train_wav_dur), 2)
            self.assertTrue(train_text[0].endswith("<TWINR_FAMILY>"))
            self.assertTrue(train_text[1].endswith("<FILLER>"))
            self.assertEqual(report.split_reports[0].ignored_count, 0)
            metadata = json.loads(report.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["schema"], "twinr_wekws_export_v1")
            self.assertEqual(metadata["splits"][0]["positive_count"], 1)
            self.assertEqual(metadata["splits"][0]["negative_count"], 1)

    def test_export_requires_at_least_one_written_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clip = root / "ignored.wav"
            _write_test_wav(clip)
            manifest = root / "ignored.json"
            manifest.write_text(
                json.dumps([{"captured_audio_path": str(clip), "label": "unclear"}]),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "did not produce any WeKws-exportable entries"):
                export_wakeword_manifests_to_wekws(
                    output_dir=root / "wekws",
                    train_manifest=manifest,
                )
