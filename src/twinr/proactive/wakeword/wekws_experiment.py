"""Prepare reproducible WeKws experiment workspaces from Twinr exports.

Twinr's professional wakeword path should not stop at exporting Kaldi-style
splits. Conventional KWS stacks such as WeKws expect a full experiment
workspace with ``dict/``, ``data/<split>/data.list``, a training config, and a
reproducible command path for training, scoring, and ONNX export. This module
materializes that workspace from an already exported Twinr-to-WeKws dataset so
custom stage-1 keyword detectors can be trained outside the generic demo-bundle
path.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any

_DEFAULT_RECIPE_ID = "mdtc_fbank_stream"


@dataclass(frozen=True, slots=True)
class WekwsRecipeSpec:
    """Describe one built-in WeKws experiment recipe."""

    recipe_id: str
    description: str
    config: dict[str, Any]
    requires_cmvn: bool
    average_model_count: int
    window_shift_ms: int


@dataclass(frozen=True, slots=True)
class PreparedWekwsSplit:
    """Describe one prepared WeKws experiment split."""

    split_name: str
    source_dir: Path
    output_dir: Path
    utterance_count: int
    data_list_path: Path


@dataclass(frozen=True, slots=True)
class PreparedWekwsExperiment:
    """Describe one prepared WeKws experiment workspace."""

    recipe: WekwsRecipeSpec
    output_dir: Path
    dataset_dir: Path
    config_path: Path
    dict_dir: Path
    script_path: Path
    export_script_path: Path
    metadata_path: Path
    model_dir: Path
    split_reports: tuple[PreparedWekwsSplit, ...]


def available_wekws_recipe_specs() -> dict[str, WekwsRecipeSpec]:
    """Return the built-in WeKws recipes Twinr can materialize."""

    return {
        "ds_tcn_fbank": WekwsRecipeSpec(
            recipe_id="ds_tcn_fbank",
            description=(
                "Depthwise-separable TCN fbank recipe aligned to the official "
                "Hey Snips example. Good first conventional KWS baseline."
            ),
            config={
                "dataset_conf": {
                    "filter_conf": {
                        "max_length": 2048,
                        "min_length": 0,
                        "token_max_length": 200,
                        "token_min_length": 1,
                        "max_output_input_ratio": 1,
                        "min_output_input_ratio": 0.0005,
                    },
                    "resample_conf": {"resample_rate": 16000},
                    "speed_perturb": False,
                    "reverb_prob": 0.2,
                    "noise_prob": 0.3,
                    "feats_type": "fbank",
                    "fbank_conf": {
                        "num_mel_bins": 40,
                        "frame_shift": 10,
                        "frame_length": 25,
                        "dither": 1.0,
                    },
                    "spec_aug": False,
                    "spec_aug_conf": {
                        "num_t_mask": 1,
                        "num_f_mask": 1,
                        "max_t": 20,
                        "max_f": 10,
                    },
                    "shuffle": True,
                    "shuffle_conf": {"shuffle_size": 1500},
                    "sort": False,
                    "batch_conf": {"batch_size": 256},
                },
                "model": {
                    "hidden_dim": 64,
                    "preprocessing": {"type": "linear"},
                    "backbone": {
                        "type": "tcn",
                        "ds": True,
                        "num_layers": 4,
                        "kernel_size": 8,
                        "dropout": 0.1,
                    },
                },
                "optim": "adam",
                "optim_conf": {
                    "lr": 0.001,
                    "weight_decay": 0.0001,
                },
                "training_config": {
                    "grad_clip": 5,
                    "max_epoch": 80,
                    "log_interval": 10,
                },
            },
            requires_cmvn=True,
            average_model_count=30,
            window_shift_ms=50,
        ),
        "mdtc_fbank_stream": WekwsRecipeSpec(
            recipe_id="mdtc_fbank_stream",
            description=(
                "Causal MDTC streaming recipe with fbank features for a "
                "stronger Twinr stage-1 baseline."
            ),
            config={
                "dataset_conf": {
                    "filter_conf": {
                        "max_length": 2048,
                        "min_length": 0,
                        "token_max_length": 200,
                        "token_min_length": 1,
                        "max_output_input_ratio": 1,
                        "min_output_input_ratio": 0.0005,
                    },
                    "resample_conf": {"resample_rate": 16000},
                    "speed_perturb": False,
                    "reverb_prob": 0.2,
                    "noise_prob": 0.3,
                    "feats_type": "fbank",
                    "fbank_conf": {
                        "num_mel_bins": 80,
                        "frame_shift": 10,
                        "frame_length": 25,
                        "dither": 1.0,
                    },
                    "spec_aug": True,
                    "spec_aug_conf": {
                        "num_t_mask": 2,
                        "num_f_mask": 2,
                        "max_t": 20,
                        "max_f": 20,
                    },
                    "shuffle": True,
                    "shuffle_conf": {"shuffle_size": 1500},
                    "sort": False,
                    "batch_conf": {"batch_size": 128},
                },
                "model": {
                    "hidden_dim": 64,
                    "preprocessing": {"type": "linear"},
                    "backbone": {
                        "type": "mdtc",
                        "num_stack": 4,
                        "stack_size": 4,
                        "kernel_size": 5,
                        "hidden_dim": 64,
                        "causal": True,
                    },
                },
                "optim": "adam",
                "optim_conf": {
                    "lr": 0.001,
                    "weight_decay": 0.00005,
                },
                "training_config": {
                    "grad_clip": 5,
                    "max_epoch": 100,
                    "log_interval": 10,
                    "criterion": "max_pooling",
                },
            },
            requires_cmvn=True,
            average_model_count=30,
            window_shift_ms=50,
        ),
    }


def prepare_wekws_experiment(
    *,
    output_dir: str | Path,
    exported_dataset_dir: str | Path,
    recipe_id: str = _DEFAULT_RECIPE_ID,
    num_keywords: int = 1,
    seed: int = 666,
    gpus: str = "0",
    num_workers: int = 8,
    cmvn_num_workers: int = 16,
    min_duration_frames: int = 50,
    base_checkpoint: str | Path | None = None,
) -> PreparedWekwsExperiment:
    """Materialize one WeKws experiment workspace from an exported dataset."""

    recipes = available_wekws_recipe_specs()
    try:
        recipe = recipes[str(recipe_id).strip()]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported WeKws recipe {recipe_id!r}. Available: {', '.join(sorted(recipes))}"
        ) from exc
    resolved_dataset_dir = Path(exported_dataset_dir).expanduser().resolve(strict=True)
    resolved_output_dir = Path(output_dir).expanduser().resolve(strict=False)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    dict_dir = resolved_output_dir / "dict"
    data_root = resolved_output_dir / "data"
    conf_dir = resolved_output_dir / "conf"
    model_dir = resolved_output_dir / "exp" / recipe.recipe_id
    for directory in (dict_dir, data_root, conf_dir, model_dir):
        directory.mkdir(parents=True, exist_ok=True)
    _copy_required_file(
        resolved_dataset_dir / "dict" / "dict.txt",
        dict_dir / "dict.txt",
    )
    _copy_required_file(
        resolved_dataset_dir / "dict" / "words.txt",
        dict_dir / "words.txt",
    )
    split_reports: list[PreparedWekwsSplit] = []
    for split_name in ("train", "dev", "test"):
        source_split_dir = resolved_dataset_dir / split_name
        if not source_split_dir.is_dir():
            if split_name in {"train", "dev"}:
                raise FileNotFoundError(source_split_dir)
            continue
        output_split_dir = data_root / split_name
        output_split_dir.mkdir(parents=True, exist_ok=True)
        _copy_required_file(source_split_dir / "wav.scp", output_split_dir / "wav.scp")
        _copy_required_file(source_split_dir / "text", output_split_dir / "text")
        _copy_required_file(source_split_dir / "wav.dur", output_split_dir / "wav.dur")
        if (source_split_dir / "utt2spk").is_file():
            _copy_required_file(source_split_dir / "utt2spk", output_split_dir / "utt2spk")
        utterance_count = _write_data_list(
            wav_scp_path=output_split_dir / "wav.scp",
            text_path=output_split_dir / "text",
            duration_path=output_split_dir / "wav.dur",
            output_path=output_split_dir / "data.list",
        )
        split_reports.append(
            PreparedWekwsSplit(
                split_name=split_name,
                source_dir=source_split_dir,
                output_dir=output_split_dir,
                utterance_count=utterance_count,
                data_list_path=output_split_dir / "data.list",
            )
        )
    config_path = conf_dir / f"{recipe.recipe_id}.yaml"
    config_path.write_text(_dump_yaml(recipe.config), encoding="utf-8")
    sitecustomize_path = resolved_output_dir / "sitecustomize.py"
    sitecustomize_path.write_text(
        _render_wekws_sitecustomize_script(),
        encoding="utf-8",
    )
    sitecustomize_path.chmod(0o755)
    cmvn_script_path = resolved_output_dir / "compute_cmvn.py"
    cmvn_script_path.write_text(
        _render_cmvn_helper_script(),
        encoding="utf-8",
    )
    cmvn_script_path.chmod(0o755)
    export_script_path = resolved_output_dir / "export_onnx.py"
    export_script_path.write_text(
        _render_onnx_export_helper_script(),
        encoding="utf-8",
    )
    export_script_path.chmod(0o755)
    script_path = resolved_output_dir / "run_wekws.sh"
    script_path.write_text(
        _render_run_script(
            recipe=recipe,
            workspace_root=resolved_output_dir,
            config_path=config_path,
            cmvn_script_path=cmvn_script_path,
            export_script_path=export_script_path,
            dict_dir=dict_dir,
            data_root=data_root,
            model_dir=model_dir,
            num_keywords=num_keywords,
            seed=seed,
            gpus=gpus,
            num_workers=num_workers,
            cmvn_num_workers=cmvn_num_workers,
            min_duration_frames=min_duration_frames,
            base_checkpoint=(
                None
                if base_checkpoint is None
                else Path(base_checkpoint).expanduser().resolve(strict=False)
            ),
        ),
        encoding="utf-8",
    )
    script_path.chmod(0o755)
    metadata_path = resolved_output_dir / "wekws_experiment_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema": "twinr_wekws_experiment_v1",
                "dataset_dir": str(resolved_dataset_dir),
                "output_dir": str(resolved_output_dir),
                "recipe_id": recipe.recipe_id,
                "description": recipe.description,
                "config_path": str(config_path),
                "sitecustomize_path": str(sitecustomize_path),
                "cmvn_script_path": str(cmvn_script_path),
                "export_script_path": str(export_script_path),
                "dict_dir": str(dict_dir),
                "model_dir": str(model_dir),
                "script_path": str(script_path),
                "num_keywords": int(num_keywords),
                "seed": int(seed),
                "gpus": str(gpus),
                "num_workers": int(num_workers),
                "cmvn_num_workers": int(cmvn_num_workers),
                "min_duration_frames": int(min_duration_frames),
                "base_checkpoint": None if base_checkpoint is None else str(base_checkpoint),
                "splits": [
                    {
                        "name": report.split_name,
                        "source_dir": str(report.source_dir),
                        "output_dir": str(report.output_dir),
                        "utterance_count": report.utterance_count,
                        "data_list_path": str(report.data_list_path),
                    }
                    for report in split_reports
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return PreparedWekwsExperiment(
        recipe=recipe,
        output_dir=resolved_output_dir,
        dataset_dir=resolved_dataset_dir,
        config_path=config_path,
        dict_dir=dict_dir,
        script_path=script_path,
        export_script_path=export_script_path,
        metadata_path=metadata_path,
        model_dir=model_dir,
        split_reports=tuple(split_reports),
    )


def _copy_required_file(source: Path, destination: Path) -> None:
    """Copy one required file into the prepared experiment workspace."""

    if not source.is_file():
        raise FileNotFoundError(source)
    shutil.copy2(source, destination)


def _read_two_column_mapping(path: Path) -> dict[str, str]:
    """Read a simple Kaldi-style mapping file with ``key value`` rows."""

    mapping: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        key, value = line.split(maxsplit=1)
        mapping[key] = value
    if not mapping:
        raise ValueError(f"{path} did not contain any entries.")
    return mapping


def _write_data_list(
    *,
    wav_scp_path: Path,
    text_path: Path,
    duration_path: Path,
    output_path: Path,
) -> int:
    """Write one WeKws ``data.list`` JSONL file from Kaldi-style split files."""

    wav_table = _read_two_column_mapping(wav_scp_path)
    duration_table = _read_two_column_mapping(duration_path)
    entries: list[str] = []
    for raw_line in text_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        utt_id = parts[0]
        if utt_id not in wav_table:
            raise KeyError(f"{utt_id} from {text_path} is missing in {wav_scp_path}")
        if utt_id not in duration_table:
            raise KeyError(f"{utt_id} from {text_path} is missing in {duration_path}")
        token_text = parts[1] if len(parts) > 1 else "<SILENCE>"
        try:
            duration_seconds = float(duration_table[utt_id])
        except ValueError as exc:
            raise ValueError(f"Invalid duration for {utt_id} in {duration_path}") from exc
        entries.append(
            json.dumps(
                {
                    "key": utt_id,
                    "txt": token_text,
                    "duration": duration_seconds,
                    "wav": wav_table[utt_id],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    if not entries:
        raise ValueError(f"{text_path} did not produce any data.list rows.")
    output_path.write_text("\n".join(entries) + "\n", encoding="utf-8")
    return len(entries)


def _dump_yaml(value: dict[str, Any]) -> str:
    """Render one configuration mapping as YAML without introducing a hard dependency."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - Twinr env ships PyYAML today.
        raise RuntimeError("PyYAML is required to render a WeKws experiment config.") from exc
    return yaml.safe_dump(value, sort_keys=False)


def _render_run_script(
    *,
    recipe: WekwsRecipeSpec,
    workspace_root: Path,
    config_path: Path,
    cmvn_script_path: Path,
    export_script_path: Path,
    dict_dir: Path,
    data_root: Path,
    model_dir: Path,
    num_keywords: int,
    seed: int,
    gpus: str,
    num_workers: int,
    cmvn_num_workers: int,
    min_duration_frames: int,
    base_checkpoint: Path | None,
) -> str:
    """Render the reproducible shell runner for one prepared WeKws experiment."""

    config_relpath = config_path.relative_to(workspace_root)
    cmvn_script_relpath = cmvn_script_path.relative_to(workspace_root)
    export_script_relpath = export_script_path.relative_to(workspace_root)
    dict_relpath = dict_dir.relative_to(workspace_root)
    model_relpath = model_dir.relative_to(workspace_root)
    cmvn_flag = (
        f'  python "$EXP_DIR/{cmvn_script_relpath}" --num_workers {int(cmvn_num_workers)} \\\n'
        f'    --train_config "$EXP_DIR/{config_relpath}" \\\n'
        f'    --in_scp "$EXP_DIR/data/train/wav.scp" \\\n'
        f'    --out_cmvn "$EXP_DIR/data/train/global_cmvn"\n'
        if recipe.requires_cmvn
        else "  :\n"
    )
    checkpoint_line = ""
    if base_checkpoint is not None:
        checkpoint_line = f'    --checkpoint "{base_checkpoint}" \\\n'
    cmvn_command = cmvn_flag.replace("python ", '"$WEKWS_PYTHON" ')
    model_mkdir_line = f'  mkdir -p "$EXP_DIR/exp/{recipe.recipe_id}"\n'
    result_dir_line = (
        f'  RESULT_DIR="$EXP_DIR/exp/{recipe.recipe_id}/test_$(basename "$SCORE_CHECKPOINT")"\n'
    )
    onnx_model_line = (
        f'  ONNX_MODEL="${{ONNX_MODEL:-$EXP_DIR/exp/{recipe.recipe_id}/$(basename "$SCORE_CHECKPOINT" .pt).onnx}}"\n'
    )
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        'WEKWS_ROOT="${WEKWS_ROOT:?set WEKWS_ROOT to a wekws checkout}"\n'
        'WEKWS_PYTHON="${WEKWS_PYTHON:-python}"\n'
        f'GPUS="${{GPUS:-{gpus}}}"\n'
        'NUM_GPUS=$(awk -F \',\' \'{print NF}\' <<< "$GPUS")\n'
        'EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'export PYTHONPATH="$EXP_DIR:$WEKWS_ROOT${PYTHONPATH:+:$PYTHONPATH}"\n\n'
        f'SCORE_CHECKPOINT="${{SCORE_CHECKPOINT:-$EXP_DIR/exp/{recipe.recipe_id}/avg_{recipe.average_model_count}.pt}}"\n'
        'stage="${1:-0}"\n'
        'stop_stage="${2:-4}"\n\n'
        'if [ "$stage" -le 0 ] && [ "$stop_stage" -ge 0 ]; then\n'
        + cmvn_command
        + "fi\n\n"
        + 'if [ "$stage" -le 1 ] && [ "$stop_stage" -ge 1 ]; then\n'
        + model_mkdir_line
        + '  "$WEKWS_PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \\\n'
        '    "$WEKWS_ROOT/wekws/bin/train.py" --gpus "$GPUS" \\\n'
        f'    --config "$EXP_DIR/{config_relpath}" \\\n'
        '    --train_data "$EXP_DIR/data/train/data.list" \\\n'
        '    --cv_data "$EXP_DIR/data/dev/data.list" \\\n'
        f'    --model_dir "$EXP_DIR/{model_relpath}" \\\n'
        f"    --num_workers {int(num_workers)} \\\n"
        f"    --num_keywords {int(num_keywords)} \\\n"
        f"    --min_duration {int(min_duration_frames)} \\\n"
        f"    --seed {int(seed)} \\\n"
        f'    --dict "$EXP_DIR/{dict_relpath}" \\\n'
        + (
            '    --cmvn_file "$EXP_DIR/data/train/global_cmvn" \\\n'
            '    --norm_var \\\n'
            if recipe.requires_cmvn
            else ""
        )
        + checkpoint_line
        + "    ${TRAIN_EXTRA_ARGS:-}\n"
        "fi\n\n"
        'if [ "$stage" -le 2 ] && [ "$stop_stage" -ge 2 ]; then\n'
        '  "$WEKWS_PYTHON" "$WEKWS_ROOT/wekws/bin/average_model.py" \\\n'
        '    --dst_model "$SCORE_CHECKPOINT" \\\n'
        f'    --src_path "$EXP_DIR/{model_relpath}" \\\n'
        f"    --num {int(recipe.average_model_count)} \\\n"
        "    --val_best\n"
        "fi\n\n"
        + 'if [ "$stage" -le 3 ] && [ "$stop_stage" -ge 3 ] && [ -f "$EXP_DIR/data/test/data.list" ]; then\n'
        + result_dir_line
        + '  mkdir -p "$RESULT_DIR"\n'
        '  "$WEKWS_PYTHON" "$WEKWS_ROOT/wekws/bin/score.py" \\\n'
        f'    --config "$EXP_DIR/{model_relpath}/config.yaml" \\\n'
        '    --test_data "$EXP_DIR/data/test/data.list" \\\n'
        '    --gpu -1 \\\n'
        '    --batch_size 256 \\\n'
        '    --checkpoint "$SCORE_CHECKPOINT" \\\n'
        '    --score_file "$RESULT_DIR/score.txt" \\\n'
        f'    --dict "$EXP_DIR/{dict_relpath}" \\\n'
        f"    --num_workers {int(num_workers)} \\\n"
        "    ${SCORE_EXTRA_ARGS:-}\n"
        '  while read -r keyword; do\n'
        '    if [ "$keyword" = "<FILLER>" ]; then\n'
        '      continue\n'
        '    fi\n'
        '    "$WEKWS_PYTHON" "$WEKWS_ROOT/wekws/bin/compute_det.py" \\\n'
        '      --keyword "$keyword" \\\n'
        '      --test_data "$EXP_DIR/data/test/data.list" \\\n'
        f"      --window_shift {int(recipe.window_shift_ms)} \\\n"
        '      --score_file "$RESULT_DIR/score.txt" \\\n'
        '      --stats_file "$RESULT_DIR/stats.${keyword}.txt"\n'
        '  done < "$EXP_DIR/dict/words.txt"\n'
        '  "$WEKWS_PYTHON" "$WEKWS_ROOT/wekws/bin/plot_det_curve.py" \\\n'
        '    --keywords_dict "$EXP_DIR/dict/dict.txt" \\\n'
        '    --stats_dir "$RESULT_DIR" \\\n'
        '    --figure_file "$RESULT_DIR/det.png" \\\n'
        '    --xlim 10 --x_step 2 --ylim 10 --y_step 2\n'
        "fi\n\n"
        + 'if [ "$stage" -le 4 ] && [ "$stop_stage" -ge 4 ]; then\n'
        + onnx_model_line
        + f'  "$WEKWS_PYTHON" "$EXP_DIR/{export_script_relpath}" \\\n'
        f'    --config "$EXP_DIR/{model_relpath}/config.yaml" \\\n'
        '    --checkpoint "$SCORE_CHECKPOINT" \\\n'
        '    --onnx_model "$ONNX_MODEL"\n'
        "fi\n"
    )


def _render_wekws_sitecustomize_script() -> str:
    """Render one Python startup shim for known WeKws dependency mismatches."""

    return (
        "#!/usr/bin/env python3\n"
        "\"\"\"Patch known training-environment mismatches for prepared WeKws runs.\n\n"
        "This helper is loaded automatically through Python's ``sitecustomize``\n"
        "mechanism because the prepared experiment directory is injected into\n"
        "``PYTHONPATH`` ahead of the upstream WeKws checkout. Keep the shim\n"
        "minimal and restricted to well-understood compatibility gaps so Twinr's\n"
        "training workflow stays reproducible without mutating the shared GPU\n"
        "machine outside the experiment workspace.\n"
        "\"\"\"\n\n"
        "from __future__ import annotations\n\n"
        "import audioop\n"
        "import io\n"
        "from pathlib import Path\n"
        "from types import SimpleNamespace\n\n"
        "import numpy as np\n"
        "import torch\n"
        "import wave\n\n"
        "try:\n"
        "    import torchaudio\n"
        "except Exception:  # pragma: no cover - depends on remote training env.\n"
        "    torchaudio = None\n\n"
        "def _coerce_wave_source(source):\n"
        "    if isinstance(source, (bytes, bytearray)):\n"
        "        return io.BytesIO(bytes(source))\n"
        "    if hasattr(source, 'read'):\n"
        "        data = source.read()\n"
        "        if hasattr(source, 'seek'):\n"
        "            try:\n"
        "                source.seek(0)\n"
        "            except Exception:\n"
        "                pass\n"
        "        return io.BytesIO(data)\n"
        "    return str(Path(source).expanduser())\n\n"
        "def _wave_info(source):\n"
        "    with wave.open(_coerce_wave_source(source), 'rb') as wav_file:\n"
        "        sample_rate = int(wav_file.getframerate())\n"
        "        num_frames = int(wav_file.getnframes())\n"
        "        num_channels = int(wav_file.getnchannels())\n"
        "    return SimpleNamespace(\n"
        "        sample_rate=sample_rate,\n"
        "        num_frames=num_frames,\n"
        "        num_channels=num_channels,\n"
        "        bits_per_sample=16,\n"
        "    )\n\n"
        "def _wave_load(source, *, frame_offset: int = 0, num_frames: int = -1):\n"
        "    with wave.open(_coerce_wave_source(source), 'rb') as wav_file:\n"
        "        sample_rate = int(wav_file.getframerate())\n"
        "        sample_width = int(wav_file.getsampwidth())\n"
        "        channels = max(1, int(wav_file.getnchannels()))\n"
        "        total_frames = int(wav_file.getnframes())\n"
        "        start_frame = max(0, min(total_frames, int(frame_offset)))\n"
        "        wav_file.setpos(start_frame)\n"
        "        if int(num_frames) < 0:\n"
        "            frames_to_read = total_frames - start_frame\n"
        "        else:\n"
        "            frames_to_read = max(0, min(total_frames - start_frame, int(num_frames)))\n"
        "        raw = wav_file.readframes(frames_to_read)\n"
        "    if raw and sample_width != 2:\n"
        "        raw = audioop.lin2lin(raw, sample_width, 2)\n"
        "    if not raw:\n"
        "        return torch.zeros((channels, 0), dtype=torch.float32), sample_rate\n"
        "    samples = np.frombuffer(raw, dtype='<i2').astype(np.float32, copy=True)\n"
        "    usable = len(samples) - (len(samples) % channels)\n"
        "    samples = samples[:usable]\n"
        "    if usable <= 0:\n"
        "        return torch.zeros((channels, 0), dtype=torch.float32), sample_rate\n"
        "    waveform = samples.reshape(-1, channels).T / float(1 << 15)\n"
        "    return torch.from_numpy(waveform.copy()), sample_rate\n\n"
        "if torchaudio is not None:\n"
        "    utils = getattr(torchaudio, 'utils', None)\n"
        "    if utils is not None and not hasattr(utils, 'sox_utils'):\n"
        "        utils.sox_utils = SimpleNamespace(set_buffer_size=lambda _value: None)\n"
        "    _original_info = getattr(torchaudio, 'info', None)\n"
        "    _original_load = getattr(torchaudio, 'load', None)\n\n"
        "    def _compat_info(source, *args, **kwargs):\n"
        "        try:\n"
        "            return _wave_info(source)\n"
        "        except Exception:\n"
        "            if _original_info is None:\n"
        "                raise\n"
        "            return _original_info(source, *args, **kwargs)\n\n"
        "    def _compat_load(source, *args, **kwargs):\n"
        "        frame_offset = int(kwargs.pop('frame_offset', 0))\n"
        "        num_frames = int(kwargs.pop('num_frames', -1))\n"
        "        if kwargs:\n"
        "            return _original_load(\n"
        "                source,\n"
        "                *args,\n"
        "                frame_offset=frame_offset,\n"
        "                num_frames=num_frames,\n"
        "                **kwargs,\n"
        "            )\n"
        "        try:\n"
        "            return _wave_load(source, frame_offset=frame_offset, num_frames=num_frames)\n"
        "        except Exception:\n"
        "            if _original_load is None:\n"
        "                raise\n"
        "            return _original_load(\n"
        "                source,\n"
        "                *args,\n"
        "                frame_offset=frame_offset,\n"
        "                num_frames=num_frames,\n"
        "            )\n\n"
        "    torchaudio.info = _compat_info\n"
        "    torchaudio.load = _compat_load\n"
    )


def _render_cmvn_helper_script() -> str:
    """Render a standalone CMVN helper compatible with modern torchaudio."""

    return (
        "#!/usr/bin/env python3\n"
        "\"\"\"Compute JSON CMVN stats for one prepared WeKws workspace.\n\n"
        "Purpose\n"
        "-------\n"
        "Work around the current WeKws upstream helper relying on\n"
        "``torchaudio.info(..., backend='sox')`` which is not stable across\n"
        "modern torchaudio builds. The generated helper keeps the WeKws\n"
        "workspace self-contained and computes the same JSON CMVN schema that\n"
        "``wekws.model.kws_model`` expects at training and inference time.\n"
        "\"\"\"\n\n"
        "from __future__ import annotations\n\n"
        "import audioop\n"
        "import argparse\n"
        "import json\n"
        "from pathlib import Path\n\n"
        "import numpy as np\n"
        "import torch\n"
        "import torchaudio.compliance.kaldi as kaldi\n"
        "import wave\n"
        "import yaml\n\n"
        "_FRAME_LENGTH_MS = 25.0\n"
        "_FRAME_SHIFT_MS = 10.0\n\n"
        "def _load_rows(path: Path) -> list[tuple[str, str]]:\n"
        "    rows: list[tuple[str, str]] = []\n"
        "    for raw_line in path.read_text(encoding='utf-8').splitlines():\n"
        "        line = raw_line.strip()\n"
        "        if not line:\n"
        "            continue\n"
        "        key, value = line.split(maxsplit=1)\n"
        "        rows.append((key, value))\n"
        "    if not rows:\n"
        "        raise ValueError(f'{path} did not contain any wav.scp rows.')\n"
        "    return rows\n\n"
        "def _segment_spec(value: str) -> tuple[Path, float | None, float | None]:\n"
        "    parts = [part.strip() for part in value.split(',') if part.strip()]\n"
        "    if len(parts) not in {1, 3}:\n"
        "        raise ValueError(f'Unsupported wav.scp entry: {value!r}')\n"
        "    path = Path(parts[0]).expanduser().resolve(strict=True)\n"
        "    if len(parts) == 1:\n"
        "        return path, None, None\n"
        "    return path, float(parts[1]), float(parts[2])\n\n"
        "def _load_pcm16_mono(value: str) -> tuple[bytes, int]:\n"
        "    path, start_seconds, end_seconds = _segment_spec(value)\n"
        "    with wave.open(str(path), 'rb') as wav_file:\n"
        "        sample_rate = int(wav_file.getframerate())\n"
        "        sample_width = int(wav_file.getsampwidth())\n"
        "        channels = int(wav_file.getnchannels())\n"
        "        total_frames = int(wav_file.getnframes())\n"
        "        start_frame = 0 if start_seconds is None else max(0, int(round(start_seconds * sample_rate)))\n"
        "        end_frame = total_frames\n"
        "        if end_seconds is not None:\n"
        "            end_frame = min(total_frames, max(start_frame, int(round(end_seconds * sample_rate))))\n"
        "        wav_file.setpos(min(start_frame, total_frames))\n"
        "        raw = wav_file.readframes(max(0, end_frame - start_frame))\n"
        "    if not raw:\n"
        "        return b'', sample_rate\n"
        "    if sample_width != 2:\n"
        "        raw = audioop.lin2lin(raw, sample_width, 2)\n"
        "    if channels > 1:\n"
        "        raw = audioop.tomono(raw, 2, 0.5, 0.5)\n"
        "    return raw, sample_rate\n\n"
        "def _pcm16_bytes_to_waveform(pcm16_bytes: bytes) -> torch.Tensor:\n"
        "    if not pcm16_bytes:\n"
        "        return torch.zeros((1, 0), dtype=torch.float32)\n"
        "    samples = np.frombuffer(pcm16_bytes, dtype='<i2').astype(np.float32, copy=False)\n"
        "    if samples.size == 0:\n"
        "        return torch.zeros((1, 0), dtype=torch.float32)\n"
        "    return torch.from_numpy(samples.copy()).unsqueeze(0)\n\n"
        "def _compute_fbank(\n"
        "    pcm16_bytes: bytes,\n"
        "    *,\n"
        "    sample_rate: int,\n"
        "    num_mel_bins: int,\n"
        "    target_sample_rate: int,\n"
        ") -> torch.Tensor:\n"
        "    if not pcm16_bytes:\n"
        "        return torch.zeros((0, num_mel_bins), dtype=torch.float32)\n"
        "    if sample_rate != target_sample_rate:\n"
        "        pcm16_bytes, _state = audioop.ratecv(\n"
        "            pcm16_bytes,\n"
        "            2,\n"
        "            1,\n"
        "            sample_rate,\n"
        "            target_sample_rate,\n"
        "            None,\n"
        "        )\n"
        "        sample_rate = target_sample_rate\n"
        "    waveform = _pcm16_bytes_to_waveform(pcm16_bytes)\n"
        "    minimum_samples = max(1, int(round(sample_rate * (_FRAME_LENGTH_MS / 1000.0))))\n"
        "    if waveform.size(1) < minimum_samples:\n"
        "        return torch.zeros((0, num_mel_bins), dtype=torch.float32)\n"
        "    return kaldi.fbank(\n"
        "        waveform,\n"
        "        num_mel_bins=num_mel_bins,\n"
        "        frame_length=_FRAME_LENGTH_MS,\n"
        "        frame_shift=_FRAME_SHIFT_MS,\n"
        "        dither=0.0,\n"
        "        energy_floor=0.0,\n"
        "        sample_frequency=sample_rate,\n"
        "    )\n\n"
        "def main() -> None:\n"
        "    parser = argparse.ArgumentParser(description='Compute JSON CMVN stats for a prepared WeKws experiment.')\n"
        "    parser.add_argument('--num_workers', type=int, default=0)\n"
        "    parser.add_argument('--train_config', required=True)\n"
        "    parser.add_argument('--in_scp', required=True)\n"
        "    parser.add_argument('--out_cmvn', required=True)\n"
        "    args = parser.parse_args()\n\n"
        "    with Path(args.train_config).open('r', encoding='utf-8') as fin:\n"
        "        configs = yaml.load(fin, Loader=yaml.FullLoader)\n"
        "    dataset_conf = configs['dataset_conf']\n"
        "    feats_type = dataset_conf.get('feats_type', 'fbank')\n"
        "    if feats_type != 'fbank':\n"
        "        raise ValueError(f'Unsupported feats_type: {feats_type!r}')\n"
        "    num_mel_bins = int(dataset_conf['fbank_conf']['num_mel_bins'])\n"
        "    target_sample_rate = int(dataset_conf.get('resample_conf', {}).get('resample_rate', 16000))\n\n"
        "    mean_stat = torch.zeros(num_mel_bins, dtype=torch.float64)\n"
        "    var_stat = torch.zeros(num_mel_bins, dtype=torch.float64)\n"
        "    frame_num = 0\n"
        "    for _key, value in _load_rows(Path(args.in_scp)):\n"
        "        pcm16_bytes, sample_rate = _load_pcm16_mono(value)\n"
        "        feats = _compute_fbank(\n"
        "            pcm16_bytes,\n"
        "            sample_rate=sample_rate,\n"
        "            num_mel_bins=num_mel_bins,\n"
        "            target_sample_rate=target_sample_rate,\n"
        "        )\n"
        "        if feats.numel() == 0:\n"
        "            continue\n"
        "        feats64 = feats.to(dtype=torch.float64)\n"
        "        mean_stat += feats64.sum(dim=0)\n"
        "        var_stat += feats64.square().sum(dim=0)\n"
        "        frame_num += int(feats64.size(0))\n"
        "    if frame_num <= 0:\n"
        "        raise ValueError('No usable CMVN frames were produced from the supplied wav.scp.')\n"
        "    payload = {\n"
        "        'mean_stat': mean_stat.tolist(),\n"
        "        'var_stat': var_stat.tolist(),\n"
        "        'frame_num': int(frame_num),\n"
        "    }\n"
        "    Path(args.out_cmvn).write_text(json.dumps(payload), encoding='utf-8')\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )


def _render_onnx_export_helper_script() -> str:
    """Render a standalone ONNX exporter compatible with MDTC-style backbones."""

    return (
        "#!/usr/bin/env python3\n"
        "\"\"\"Export a prepared WeKws checkpoint to ONNX for Twinr runtime use.\n\n"
        "Purpose\n"
        "-------\n"
        "The current upstream WeKws exporter assumes ``backbone.num_layers``\n"
        "exists even for non-FSMN models. That breaks MDTC-based wakeword\n"
        "recipes although training and checkpoint averaging already succeed. This\n"
        "local exporter keeps the prepared experiment self-contained and exports\n"
        "the same cache-aware ONNX contract Twinr's runtime backend expects.\n"
        "\"\"\"\n\n"
        "from __future__ import annotations\n\n"
        "import argparse\n\n"
        "import onnx\n"
        "import onnxruntime as ort\n"
        "import torch\n"
        "import yaml\n\n"
        "from wekws.model.kws_model import init_model\n"
        "from wekws.utils.checkpoint import load_checkpoint\n\n"
        "def _get_args() -> argparse.Namespace:\n"
        "    parser = argparse.ArgumentParser(description='export one WeKws checkpoint to ONNX')\n"
        "    parser.add_argument('--config', required=True, help='WeKws config file')\n"
        "    parser.add_argument('--checkpoint', required=True, help='Checkpoint to export')\n"
        "    parser.add_argument('--onnx_model', required=True, help='Output ONNX model path')\n"
        "    return parser.parse_args()\n\n"
        "def main() -> None:\n"
        "    args = _get_args()\n"
        "    with open(args.config, 'r', encoding='utf-8') as handle:\n"
        "        configs = yaml.load(handle, Loader=yaml.FullLoader)\n"
        "    feature_dim = int(configs['model']['input_dim'])\n"
        "    backbone_config = dict(configs['model'].get('backbone', {}))\n"
        "    backbone_type = str(backbone_config.get('type', '')).strip().lower()\n"
        "    is_fsmn = backbone_type == 'fsmn'\n"
        "    num_layers = int(backbone_config.get('num_layers', 1))\n"
        "    model = init_model(configs['model'])\n"
        "    if configs.get('training_config', {}).get('criterion', 'max_pooling') == 'ctc':\n"
        "        model.forward = model.forward_softmax\n"
        "    print(model)\n"
        "    load_checkpoint(model, args.checkpoint)\n"
        "    model.eval()\n"
        "    has_embedded_cmvn = getattr(model, 'global_cmvn', None) is not None\n"
        "    padding = int(getattr(model.backbone, 'padding', 0))\n"
        "    dummy_input = torch.randn(1, 100, feature_dim, dtype=torch.float)\n"
        "    cache = torch.zeros(1, model.hdim, padding, dtype=torch.float)\n"
        "    if is_fsmn and padding > 0:\n"
        "        cache = cache.unsqueeze(-1).expand(-1, -1, -1, num_layers)\n"
        "    dynamic_axes = {'input': {1: 'T'}, 'output': {1: 'T'}}\n"
        "    torch.onnx.export(\n"
        "        model,\n"
        "        (dummy_input, cache),\n"
        "        args.onnx_model,\n"
        "        input_names=['input', 'cache'],\n"
        "        output_names=['output', 'r_cache'],\n"
        "        dynamic_axes=dynamic_axes,\n"
        "        opset_version=13,\n"
        "        verbose=False,\n"
        "        do_constant_folding=True,\n"
        "    )\n"
        "    onnx_model = onnx.load(args.onnx_model)\n"
        "    meta = onnx_model.metadata_props.add()\n"
        "    meta.key, meta.value = 'cache_dim', str(model.hdim)\n"
        "    meta = onnx_model.metadata_props.add()\n"
        "    meta.key, meta.value = 'cache_len', str(padding)\n"
        "    meta = onnx_model.metadata_props.add()\n"
        "    meta.key, meta.value = 'cmvn_mode', ('embedded' if has_embedded_cmvn else 'none')\n"
        "    onnx.save(onnx_model, args.onnx_model)\n"
        "    torch_output = model(dummy_input, cache)\n"
        "    ort_session = ort.InferenceSession(\n"
        "        args.onnx_model,\n"
        "        providers=['CPUExecutionProvider'],\n"
        "    )\n"
        "    onnx_output = ort_session.run(\n"
        "        None,\n"
        "        {'input': dummy_input.numpy(), 'cache': cache.numpy()},\n"
        "    )\n"
        "    if torch.allclose(torch_output[0], torch.tensor(onnx_output[0]), atol=1e-6) and \\\n"
        "       torch.allclose(torch_output[1], torch.tensor(onnx_output[1]), atol=1e-6):\n"
        "        print('Export to onnx succeed!')\n"
        "    else:\n"
        "        print('Export to onnx succeed, but pytorch/onnx diverged for the same input.')\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )


__all__ = [
    "PreparedWekwsExperiment",
    "PreparedWekwsSplit",
    "WekwsRecipeSpec",
    "available_wekws_recipe_specs",
    "prepare_wekws_experiment",
]
