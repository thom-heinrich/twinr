#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "mediapipe-model-maker>=0.2.1.4",
#   "setuptools<81",
#   "tensorflow>=2.15",
# ]
# ///
"""Train and package Twinr's custom MediaPipe hand-gesture recognizer.

This helper runs on the leading repository checkout, not on the Pi runtime.
It trains a MediaPipe Gesture Recognizer task bundle from a folder-per-label
dataset, exports the `.task` artifact, and copies the final model into the
runtime model directory that Twinr expects.

Purpose
-------
Turn a labeled gesture dataset such as `none`, `thumbs_up`, `thumbs_down`, and
`peace_sign` into a bounded `.task` asset and a JSON summary that Twinr can
later deploy to `/twinr/state/mediapipe/models/`.

Usage
-----
Command-line invocation examples::

    state/mediapipe/model_maker_venv/bin/python hardware/piaicam/train_custom_gesture_model.py --dataset-root state/mediapipe/custom_gesture_dataset
    state/mediapipe/model_maker_venv/bin/python hardware/piaicam/train_custom_gesture_model.py --dataset-root state/mediapipe/custom_gesture_dataset --dry-run
    state/mediapipe/model_maker_venv/bin/python hardware/piaicam/train_custom_gesture_model.py --dataset-root state/mediapipe/custom_gesture_dataset --epochs 12 --batch-size 4

Inputs
------
- A folder-per-label dataset root that includes at minimum `none`,
  `thumbs_up`, `thumbs_down`, and `peace_sign`.
- A Python environment that can import `mediapipe_model_maker`. If the import
  fails because `pkg_resources` is missing, pin `setuptools<81`.

Notes
-----
Extra label directories may coexist in the dataset root for experiments or old
models. Before calling Model Maker, Twinr stages only the explicitly required
labels into a fresh training directory so the classifier head stays locked to
the intended product contract.

Outputs
-------
- Writes one timestamped training run under `state/mediapipe/custom_gesture_training/`.
- Exports a `.task` bundle plus `labels.txt`.
- Copies the final `.task` into `state/mediapipe/models/custom_gesture.task`
  by default and prints the Twinr env var needed on the Pi runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
import argparse
import json
import sys

from custom_gesture_workflow import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_REQUIRED_LABELS,
    DEFAULT_RUNTIME_MODEL_DIR,
    DEFAULT_RUNTIME_MODEL_NAME,
    DEFAULT_TRAINING_RUNS_ROOT,
    collect_dataset_manifest,
    copy_model_to_runtime,
    current_timestamp_slug,
    ensure_minimum_examples,
    runtime_env_hint,
    stage_required_label_dataset,
    write_json_summary,
)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Store one bounded custom-gesture training request."""

    dataset_root: Path
    output_dir: Path
    runtime_model_dir: Path
    model_name: str
    runtime_model_name: str
    validation_split: float
    epochs: int
    batch_size: int
    learning_rate: float
    min_images_per_label: int
    required_labels: tuple[str, ...]
    dry_run: bool = False


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the custom-gesture training helper."""

    parser = argparse.ArgumentParser(
        description="Train and package a bounded MediaPipe custom gesture model for Twinr.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root laid out as one directory per label.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Training run directory. Defaults to state/mediapipe/custom_gesture_training/<timestamp>.",
    )
    parser.add_argument(
        "--runtime-model-dir",
        type=Path,
        default=DEFAULT_RUNTIME_MODEL_DIR,
        help="Directory that should receive the exported runtime .task bundle.",
    )
    parser.add_argument(
        "--model-name",
        default="custom_gesture.task",
        help="Filename used for the exported training artifact inside the run directory.",
    )
    parser.add_argument(
        "--runtime-model-name",
        default=DEFAULT_RUNTIME_MODEL_NAME,
        help="Filename used for the copied runtime .task artifact.",
    )
    parser.add_argument(
        "--required-label",
        action="append",
        default=None,
        help="Required dataset label. May be passed multiple times. Defaults to none, thumbs_up, thumbs_down, peace_sign.",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.20,
        help="Fraction of the dataset reserved for validation.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs for the MediaPipe Model Maker job.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for model training.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for model training.",
    )
    parser.add_argument(
        "--min-images-per-label",
        type=int,
        default=1,
        help="Fail if any required label has fewer images than this number.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the dataset and print the planned output paths without training.",
    )
    return parser


def train_custom_gesture_model(
    config: TrainingConfig,
    *,
    api_loader=None,
) -> dict[str, object]:
    """Train one custom gesture model and return a JSON-friendly summary."""

    loader = api_loader or _load_gesture_recognizer_api
    manifest = collect_dataset_manifest(
        config.dataset_root,
        required_labels=config.required_labels,
        include_only_required=True,
    )
    ensure_minimum_examples(manifest, min_images_per_label=config.min_images_per_label)
    staged_dataset_root = config.output_dir / "training_dataset"
    exported_model_path = config.output_dir / config.model_name
    runtime_model_path = config.runtime_model_dir / config.runtime_model_name
    summary: dict[str, object] = {
        "status": "dry_run" if config.dry_run else "trained",
        "required_labels": list(config.required_labels),
        "dataset": manifest.to_summary(),
        "training": {
            "output_dir": str(config.output_dir),
            "model_name": config.model_name,
            "runtime_model_dir": str(config.runtime_model_dir),
            "runtime_model_name": config.runtime_model_name,
            "validation_split": config.validation_split,
            "training_split": 1.0 - config.validation_split,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "min_images_per_label": config.min_images_per_label,
            "training_dataset_root": str(staged_dataset_root),
        },
        "exported_model_path": str(exported_model_path),
        "runtime_model_path": str(runtime_model_path),
        "runtime_env_hint": runtime_env_hint(runtime_model_path),
    }
    if config.dry_run:
        return summary

    gesture_recognizer = loader()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    staged_manifest = stage_required_label_dataset(
        source_manifest=manifest,
        output_root=staged_dataset_root,
    )
    dataset = _load_training_dataset(
        gesture_recognizer=gesture_recognizer,
        dataset_root=staged_manifest.root,
    )
    training_data, validation_data = dataset.split(1.0 - config.validation_split)
    options = gesture_recognizer.GestureRecognizerOptions(
        hparams=gesture_recognizer.HParams(
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            epochs=config.epochs,
            export_dir=str(config.output_dir),
            shuffle=True,
        )
    )
    recognizer = gesture_recognizer.GestureRecognizer.create(
        train_data=training_data,
        validation_data=validation_data,
        options=options,
    )
    evaluation = recognizer.evaluate(validation_data)
    recognizer.export_model(model_name=config.model_name)
    recognizer.export_labels(str(config.output_dir))
    copied_runtime_model = copy_model_to_runtime(
        source_model_path=exported_model_path,
        runtime_model_dir=config.runtime_model_dir,
        runtime_model_name=config.runtime_model_name,
    )
    summary["runtime_model_path"] = str(copied_runtime_model)
    summary["runtime_env_hint"] = runtime_env_hint(copied_runtime_model)
    dataset_summary = cast(dict[str, object], summary["dataset"])
    dataset_summary["label_names"] = list(getattr(dataset, "label_names", ()))
    dataset_summary["num_classes"] = int(
        getattr(dataset, "num_classes", len(manifest.labels)) or len(manifest.labels)
    )
    dataset_summary["training_size"] = _dataset_size(training_data)
    dataset_summary["validation_size"] = _dataset_size(validation_data)
    summary["evaluation"] = _normalize_evaluation(evaluation)
    return summary


def main(argv: list[str] | None = None) -> int:
    """Run the bounded custom-gesture training helper."""

    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir = args.output_dir or (DEFAULT_TRAINING_RUNS_ROOT / current_timestamp_slug())
    config = TrainingConfig(
        dataset_root=args.dataset_root,
        output_dir=output_dir,
        runtime_model_dir=args.runtime_model_dir,
        model_name=str(args.model_name or "custom_gesture.task").strip() or "custom_gesture.task",
        runtime_model_name=str(args.runtime_model_name or DEFAULT_RUNTIME_MODEL_NAME).strip()
        or DEFAULT_RUNTIME_MODEL_NAME,
        validation_split=_coerce_validation_split(args.validation_split),
        epochs=max(1, int(args.epochs)),
        batch_size=max(1, int(args.batch_size)),
        learning_rate=max(1e-6, float(args.learning_rate)),
        min_images_per_label=max(1, int(args.min_images_per_label)),
        required_labels=tuple(args.required_label or DEFAULT_REQUIRED_LABELS),
        dry_run=bool(args.dry_run),
    )
    summary = train_custom_gesture_model(config)
    write_json_summary(config.output_dir / "summary.json", summary)
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def _load_gesture_recognizer_api():
    """Import the MediaPipe Model Maker API with a clear compatibility error."""

    try:
        import pkg_resources  # noqa: F401
        from mediapipe_model_maker import gesture_recognizer
    except ModuleNotFoundError as exc:
        missing_name = getattr(exc, "name", None) or ""
        if missing_name == "pkg_resources" or "pkg_resources" in str(exc):
            raise RuntimeError("mediapipe_model_maker_requires_setuptools_lt_81") from exc
        raise RuntimeError(f"mediapipe_model_maker_unavailable:{missing_name or 'unknown'}") from exc
    except Exception as exc:  # pragma: no cover - depends on local training env.
        raise RuntimeError("mediapipe_model_maker_unavailable") from exc
    return gesture_recognizer


def _load_training_dataset(*, gesture_recognizer, dataset_root: Path):
    """Load one Model Maker dataset and map opaque hand-detection failures.

    MediaPipe Model Maker currently raises a generic ``ValueError`` with the
    message ``No valid hand is detected.`` when the capture set does not
    actually show a usable hand. Surface that as a bounded, operator-readable
    runtime error so the workflow tells the user to recapture instead of
    looking like an internal training crash.
    """

    try:
        return gesture_recognizer.Dataset.from_folder(str(dataset_root))
    except ValueError as exc:
        if "No valid hand is detected." in str(exc):
            raise RuntimeError(
                "custom_gesture_dataset_no_detectable_hands:"
                f"{Path(dataset_root)}:"
                "recapture_with_hand_fully_visible"
            ) from exc
        raise


def _coerce_validation_split(value: object) -> float:
    """Clamp the validation split into a safe open interval."""

    split = float(cast(Any, value))
    if split <= 0.0 or split >= 0.5:
        raise ValueError("custom_gesture_validation_split_invalid")
    return split


def _dataset_size(dataset: object) -> int | None:
    """Return the size for one Model Maker dataset split when available."""

    size = getattr(dataset, "size", None)
    if size is None:
        return None
    return int(size() if callable(size) else size)


def _normalize_evaluation(value: Any) -> dict[str, object]:
    """Convert MediaPipe evaluation output into a JSON-friendly summary."""

    if isinstance(value, dict):
        return {str(key): _normalize_scalar(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return {"metrics": [_normalize_scalar(item) for item in value]}
    return {"metrics": _normalize_scalar(value)}


def _normalize_scalar(value: Any) -> object:
    """Normalize one scalar or nested evaluation value for JSON output."""

    if isinstance(value, dict):
        return {str(key): _normalize_scalar(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_scalar(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
