"""Tests for the Twinr-to-WeKws experiment workspace preparer."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from twinr.proactive.wakeword.wekws_experiment import prepare_wekws_experiment


def _write_split(root: Path, split_name: str, rows: list[tuple[str, str, float, str]]) -> None:
    split_dir = root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "wav.scp").write_text(
        "".join(f"{utt_id} {wav_path}\n" for utt_id, wav_path, _, _ in rows),
        encoding="utf-8",
    )
    (split_dir / "text").write_text(
        "".join(f"{utt_id} {token}\n" for utt_id, _, _, token in rows),
        encoding="utf-8",
    )
    (split_dir / "wav.dur").write_text(
        "".join(f"{utt_id} {duration:.3f}\n" for utt_id, _, duration, _ in rows),
        encoding="utf-8",
    )
    (split_dir / "utt2spk").write_text(
        "".join(f"{utt_id} {split_name}\n" for utt_id, _, _, _ in rows),
        encoding="utf-8",
    )


class PrepareWekwsExperimentTests(unittest.TestCase):
    def test_prepare_wekws_experiment_writes_workspace_and_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "dataset"
            (dataset_dir / "dict").mkdir(parents=True, exist_ok=True)
            (dataset_dir / "dict" / "dict.txt").write_text(
                "<FILLER> -1\n<TWINR_FAMILY> 0\n",
                encoding="utf-8",
            )
            (dataset_dir / "dict" / "words.txt").write_text(
                "<FILLER>\n<TWINR_FAMILY>\n",
                encoding="utf-8",
            )
            _write_split(
                dataset_dir,
                "train",
                [
                    ("train_0001", str(root / "train_0001.wav"), 1.25, "<TWINR_FAMILY>"),
                    ("train_0002", str(root / "train_0002.wav"), 0.9, "<FILLER>"),
                ],
            )
            _write_split(
                dataset_dir,
                "dev",
                [("dev_0001", str(root / "dev_0001.wav"), 1.1, "<TWINR_FAMILY>")],
            )
            _write_split(
                dataset_dir,
                "test",
                [("test_0001", str(root / "test_0001.wav"), 1.4, "<FILLER>")],
            )

            report = prepare_wekws_experiment(
                output_dir=root / "experiment",
                exported_dataset_dir=dataset_dir,
                recipe_id="ds_tcn_fbank",
                base_checkpoint=root / "base.pt",
            )

            self.assertEqual(report.recipe.recipe_id, "ds_tcn_fbank")
            train_data_list = json.loads(
                (report.output_dir / "data" / "train" / "data.list")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual(train_data_list["key"], "train_0001")
            self.assertEqual(train_data_list["txt"], "<TWINR_FAMILY>")
            self.assertEqual(train_data_list["wav"], str(root / "train_0001.wav"))

            script_text = report.script_path.read_text(encoding="utf-8")
            cmvn_text = (report.output_dir / "compute_cmvn.py").read_text(encoding="utf-8")
            export_text = (report.output_dir / "export_onnx.py").read_text(encoding="utf-8")
            sitecustomize_text = (report.output_dir / "sitecustomize.py").read_text(
                encoding="utf-8"
            )
            self.assertIn('WEKWS_ROOT="${WEKWS_ROOT:?set WEKWS_ROOT to a wekws checkout}"', script_text)
            self.assertIn('"$WEKWS_ROOT/wekws/bin/train.py"', script_text)
            self.assertIn('--config "$EXP_DIR/conf/ds_tcn_fbank.yaml"', script_text)
            self.assertIn('--dict "$EXP_DIR/dict"', script_text)
            self.assertIn(f'--checkpoint "{(root / "base.pt").resolve(strict=False)}"', script_text)
            self.assertIn('WEKWS_PYTHON="${WEKWS_PYTHON:-python}"', script_text)
            self.assertIn(
                'export PYTHONPATH="$EXP_DIR:$WEKWS_ROOT${PYTHONPATH:+:$PYTHONPATH}"',
                script_text,
            )
            self.assertIn('"$WEKWS_PYTHON" "$EXP_DIR/compute_cmvn.py"', script_text)
            self.assertIn('"$WEKWS_PYTHON" "$EXP_DIR/export_onnx.py"', script_text)
            self.assertIn('"$WEKWS_PYTHON" -m torch.distributed.run', script_text)
            self.assertIn("Compute JSON CMVN stats for one prepared WeKws workspace", cmvn_text)
            self.assertIn("backbone.num_layers", export_text)
            self.assertIn("has_embedded_cmvn", export_text)
            self.assertIn("meta.key, meta.value = 'cmvn_mode'", export_text)
            self.assertIn("providers=['CPUExecutionProvider']", export_text)
            self.assertIn("utils.sox_utils = SimpleNamespace", sitecustomize_text)
            self.assertIn("torchaudio.info = _compat_info", sitecustomize_text)
            self.assertIn("torchaudio.load = _compat_load", sitecustomize_text)

            metadata = json.loads(report.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["schema"], "twinr_wekws_experiment_v1")
            self.assertEqual(metadata["recipe_id"], "ds_tcn_fbank")
            self.assertEqual(
                metadata["sitecustomize_path"],
                str(report.output_dir / "sitecustomize.py"),
            )
            self.assertEqual(metadata["cmvn_script_path"], str(report.output_dir / "compute_cmvn.py"))
            self.assertEqual(metadata["export_script_path"], str(report.output_dir / "export_onnx.py"))
            self.assertEqual(len(metadata["splits"]), 3)

    def test_prepare_wekws_experiment_requires_dev_split(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "dataset"
            (dataset_dir / "dict").mkdir(parents=True, exist_ok=True)
            (dataset_dir / "dict" / "dict.txt").write_text(
                "<FILLER> -1\n<TWINR_FAMILY> 0\n",
                encoding="utf-8",
            )
            (dataset_dir / "dict" / "words.txt").write_text(
                "<FILLER>\n<TWINR_FAMILY>\n",
                encoding="utf-8",
            )
            _write_split(
                dataset_dir,
                "train",
                [("train_0001", str(root / "train_0001.wav"), 1.25, "<TWINR_FAMILY>")],
            )

            with self.assertRaises(FileNotFoundError):
                prepare_wekws_experiment(
                    output_dir=root / "experiment",
                    exported_dataset_dir=dataset_dir,
                )

    def test_mdtc_stream_recipe_uses_max_pooling_wakeup_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "dataset"
            (dataset_dir / "dict").mkdir(parents=True, exist_ok=True)
            (dataset_dir / "dict" / "dict.txt").write_text(
                "<FILLER> -1\n<TWINR_FAMILY> 0\n",
                encoding="utf-8",
            )
            (dataset_dir / "dict" / "words.txt").write_text(
                "<FILLER>\n<TWINR_FAMILY>\n",
                encoding="utf-8",
            )
            _write_split(
                dataset_dir,
                "train",
                [("train_0001", str(root / "train_0001.wav"), 1.25, "<TWINR_FAMILY>")],
            )
            _write_split(
                dataset_dir,
                "dev",
                [("dev_0001", str(root / "dev_0001.wav"), 1.1, "<FILLER>")],
            )

            report = prepare_wekws_experiment(
                output_dir=root / "experiment",
                exported_dataset_dir=dataset_dir,
                recipe_id="mdtc_fbank_stream",
            )

            config_text = report.config_path.read_text(encoding="utf-8")
            self.assertIn("criterion: max_pooling", config_text)
            self.assertNotIn("classifier:", config_text)


if __name__ == "__main__":
    unittest.main()
