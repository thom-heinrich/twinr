from pathlib import Path
import json
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.proactive.wakeword.kws_assets import (
    derive_kws_keyword_names,
    provision_builtin_kws_bundle,
)


def _build_fake_bundle_archive(root: Path) -> Path:
    bundle_root = root / "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
    bundle_root.mkdir(parents=True, exist_ok=True)
    (bundle_root / "tokens.txt").write_text("TOK\n", encoding="utf-8")
    (bundle_root / "bpe.model").write_text("fake-bpe\n", encoding="utf-8")
    (bundle_root / "README.md").write_text("# upstream\n", encoding="utf-8")
    (bundle_root / "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx").write_bytes(b"encoder")
    (bundle_root / "decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx").write_bytes(b"decoder")
    (bundle_root / "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx").write_bytes(b"joiner")
    archive_path = root / "bundle.tar.bz2"
    with tarfile.open(archive_path, "w:bz2") as archive:
        archive.add(bundle_root, arcname=bundle_root.name)
    return archive_path


class WakewordKwsAssetsTests(unittest.TestCase):
    def test_derive_kws_keyword_names_deduplicates_greeting_variants(self) -> None:
        names = derive_kws_keyword_names(
            phrases=("Hey Twinna", "Hallo Twina", "Twinner hey", "Twinr", "Hey Twinna"),
        )
        self.assertEqual(names, ("twinna", "twina", "twinner", "twinr"))

    def test_provision_builtin_kws_bundle_writes_ready_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive_path = _build_fake_bundle_archive(root)
            output_dir = root / "kws"
            text2token_calls: list[tuple[list[str], str, str, str]] = []

            def _fake_text2token(texts, *, tokens, tokens_type, bpe_model):
                text2token_calls.append((list(texts), tokens, tokens_type, bpe_model))
                return [[f"TOK_{index}", line.split("@", 1)[1].strip()] for index, line in enumerate(texts)]

            bundle = provision_builtin_kws_bundle(
                output_dir=output_dir,
                explicit_keywords=("Twinna", "Twinr"),
                archive_path=archive_path,
                text2token_fn=_fake_text2token,
            )

            self.assertEqual(bundle.keyword_names, ("twinna", "twinr"))
            self.assertEqual(
                (output_dir / "keywords_raw.txt").read_text(encoding="utf-8").splitlines(),
                ["TWINNA @twinna", "TWINR @twinr"],
            )
            self.assertEqual(
                (output_dir / "keywords.txt").read_text(encoding="utf-8").splitlines(),
                ["TOK_0 twinna", "TOK_1 twinr"],
            )
            metadata = json.loads((output_dir / "bundle_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["bundle_id"], "gigaspeech_3_3m_bpe_int8")
            self.assertEqual(metadata["keyword_names"], ["twinna", "twinr"])
            self.assertTrue((output_dir / "tokens.txt").is_file())
            self.assertTrue((output_dir / "encoder.onnx").is_file())
            self.assertTrue((output_dir / "decoder.onnx").is_file())
            self.assertTrue((output_dir / "joiner.onnx").is_file())
            self.assertTrue((output_dir / "bpe.model").is_file())
            self.assertTrue((output_dir / "upstream.README.md").is_file())
            self.assertEqual(len(text2token_calls), 1)
            self.assertEqual(text2token_calls[0][0], ["TWINNA @twinna", "TWINR @twinr"])


if __name__ == "__main__":
    unittest.main()
