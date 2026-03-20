from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "generate_multivoice_dataset.py"
    spec = spec_from_file_location("twinr_generate_multivoice_dataset", script_path)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class GenerateMultivoiceDatasetTests(unittest.TestCase):
    def test_load_generators_supports_piper_main_fallback(self) -> None:
        module = _load_script_module()
        fake_generator_module = SimpleNamespace(
            generate_samples=object(),
            generate_samples_onnx=object(),
        )

        with patch.object(
            module.importlib,
            "import_module",
            side_effect=[ModuleNotFoundError("legacy path missing"), fake_generator_module],
        ):
            generate_samples, generate_samples_onnx = module._load_generators(Path("/tmp/piper"))

        self.assertIs(generate_samples, fake_generator_module.generate_samples)
        self.assertIs(generate_samples_onnx, fake_generator_module.generate_samples_onnx)

    def test_positive_phrases_cover_current_twinr_family(self) -> None:
        module = _load_script_module()
        phrases = {item.lower() for item in module.PHRASE_PROFILES["family"]["positive"]}

        self.assertIn("hallo twinr", phrases)
        self.assertIn("twinr", phrases)
        self.assertIn("hallo twinna", phrases)
        self.assertIn("hallo twina", phrases)
        self.assertIn("hallo twinner", phrases)

    def test_strict_twinr_profile_excludes_alias_spellings_from_positives(self) -> None:
        module = _load_script_module()
        positive_phrases, negative_phrases = module._resolve_phrase_profile("strict_twinr")

        self.assertIn("hallo twinr", positive_phrases)
        self.assertNotIn("hallo twinna", positive_phrases)
        self.assertIn("hallo twinna", negative_phrases)
        self.assertIn("twinner", negative_phrases)

    def test_negative_phrases_cover_known_hard_confusions(self) -> None:
        module = _load_script_module()
        phrases = {item.lower() for item in module.PHRASE_PROFILES["family"]["negative"]}

        for item in ("hallo twin", "winner", "twitter", "timer", "tina", "winter"):
            self.assertIn(item, phrases)

    def test_copy_extra_samples_splits_positive_room_captures(self) -> None:
        module = _load_script_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            source_dir = temp_root / "source"
            train_dir = temp_root / "train"
            test_dir = temp_root / "test"
            source_dir.mkdir()
            train_dir.mkdir()
            test_dir.mkdir()
            for index in range(5):
                (source_dir / f"clip_{index}.wav").write_bytes(b"RIFFfake")

            module._copy_extra_samples(
                source_dirs=[source_dir],
                train_dir=train_dir,
                test_dir=test_dir,
                prefix="extra_pos",
            )

            self.assertEqual(4, len(list(train_dir.glob("*.wav"))))
            self.assertEqual(1, len(list(test_dir.glob("*.wav"))))
            self.assertTrue(all(path.name.startswith("extra_pos_") for path in train_dir.glob("*.wav")))
            self.assertTrue(all(path.name.startswith("extra_pos_") for path in test_dir.glob("*.wav")))

    def test_copy_extra_samples_respects_excluded_paths(self) -> None:
        module = _load_script_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            source_dir = temp_root / "source"
            train_dir = temp_root / "train"
            test_dir = temp_root / "test"
            source_dir.mkdir()
            train_dir.mkdir()
            test_dir.mkdir()
            kept = source_dir / "keep.wav"
            skipped = source_dir / "skip.wav"
            kept.write_bytes(b"RIFFkeep")
            skipped.write_bytes(b"RIFFskip")

            module._copy_extra_samples(
                source_dirs=[source_dir],
                train_dir=train_dir,
                test_dir=test_dir,
                prefix="extra_pos",
                exclude_paths={skipped.resolve(strict=False)},
            )

            copied_names = {path.name for path in train_dir.glob("*.wav")} | {
                path.name for path in test_dir.glob("*.wav")
            }
            self.assertEqual({"extra_pos_keep.wav"}, copied_names)

    def test_mine_hard_negative_records_filters_by_threshold_and_text_cap(self) -> None:
        module = _load_script_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            clip_a = temp_root / "winner_a.wav"
            clip_b = temp_root / "winner_b.wav"
            clip_c = temp_root / "timer.wav"
            clip_pos = temp_root / "positive.wav"
            clip_skip = temp_root / "skip.wav"
            model_path = temp_root / "fake_model.onnx"
            for path in (clip_a, clip_b, clip_c, clip_pos, clip_skip):
                path.write_bytes(b"RIFFfake")
            model_path.write_bytes(b"fake-onnx")
            manifest_path = temp_root / "captured_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    [
                        {"captured_audio_path": str(clip_a), "label": "negative", "text": "Winner"},
                        {"captured_audio_path": str(clip_b), "label": "negative", "text": "Winner"},
                        {"captured_audio_path": str(clip_c), "label": "negative", "text": "Timer"},
                        {"captured_audio_path": str(clip_skip), "label": "negative", "text": "Skip"},
                        {"captured_audio_path": str(clip_pos), "label": "positive", "text": "Twinr"},
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            score_by_path = {
                str(clip_a.resolve(strict=False)): 0.91,
                str(clip_b.resolve(strict=False)): 0.76,
                str(clip_c.resolve(strict=False)): 0.49,
                str(clip_skip.resolve(strict=False)): 0.99,
            }

            class _FakeModel:
                def reset(self) -> None:
                    return None

                def predict_clip(self, clip: str, padding: int = 1, chunk_size: int = 1280):
                    del padding, chunk_size
                    return [{"fake_model": score_by_path[str(Path(clip).resolve(strict=False))]}]

            rows = module._mine_hard_negative_records(
                manifest_path=manifest_path,
                model_path=model_path,
                threshold=0.5,
                exclude_paths={clip_skip.resolve(strict=False)},
                max_per_text=1,
                model_factory=lambda **_kwargs: _FakeModel(),
            )

            self.assertEqual(
                [
                    {
                        "audio_path": str(clip_a.resolve(strict=False)),
                        "label": "negative",
                        "text": "Winner",
                        "peak_score": 0.91,
                    }
                ],
                rows,
            )

    def test_generate_split_normalizes_phrase_iterables_to_lists(self) -> None:
        module = _load_script_module()
        calls: list[object] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "positive_train"

            def _fake_generate_pt(**kwargs) -> None:
                calls.append(kwargs["text"])

            def _fake_generate_samples_onnx(**kwargs) -> None:
                calls.append(kwargs["text"])

            module.generate_split(
                output_dir,
                phrases=("hallo twinr", "twinr"),
                generator_count=1,
                onnx_count=1,
                onnx_models=[Path("/tmp/voice.onnx")],
                generator_model=Path("/tmp/generator.pt"),
                generate_pt=_fake_generate_pt,
                generate_samples_onnx=_fake_generate_samples_onnx,
            )

        self.assertEqual([["hallo twinr", "twinr"], ["hallo twinr", "twinr"]], calls)
