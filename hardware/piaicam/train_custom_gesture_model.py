#!/usr/bin/env python3
# CHANGELOG: 2026-03-27
# BUG-1: Replaced validation-only evaluation with deterministic train/validation/test splits and held-out test metrics.
# BUG-2: Fixed silent dataset attrition by auditing image readability, duplicate files, and hand detectability before training.
# SEC-1: Blocked path traversal / arbitrary overwrite via model artifact names and switched runtime model publication to an atomic copy.
# IMP-1: Added 2026-compatible environment guards and dependency pins for the stale MediaPipe Model Maker stack.
# IMP-2: Exposed current MediaPipe customization knobs (preprocessing confidence, dropout, layer widths, lr decay, gamma, steps) plus reproducible seeds and rich run provenance.
# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = [
#   "mediapipe-model-maker==0.2.1.4",
#   "tensorflow>=2.15,<2.16",
#   "keras>=2.15,<2.16",
#   "setuptools<81",
#   "Pillow>=10,<13",
# ]
# ///
"""Train and package Twinr's custom MediaPipe hand-gesture recognizer.

This helper runs on the leading repository checkout, not on the Pi runtime.
It trains a MediaPipe Gesture Recognizer task bundle from a folder-per-label
custom dataset, exports the `.task` artifact, and atomically publishes the
final model into the runtime model directory that Twinr expects.

Purpose
-------
Turn a labeled gesture dataset such as `none`, `thumbs_up`, `thumbs_down`, and
`peace_sign` into a bounded `.task` asset plus JSON summaries that Twinr can
later deploy to `/twinr/state/mediapipe/models/`.

What changed versus the legacy helper
-------------------------------------
- Performs deterministic stratified train/validation/test splits instead of
  evaluating on the validation split only.
- Audits image readability, exact duplicates, and hand detectability before
  training so MediaPipe's internal "drop invalid sample" behavior cannot
  silently skew class balance.
- Publishes runtime artifacts atomically and records SHA-256 hashes for run
  provenance.
- Exposes MediaPipe's current customization knobs for preprocessing and model
  head tuning.

Usage
-----
Command-line invocation examples::

    state/mediapipe/model_maker_venv/bin/python hardware/piaicam/train_custom_gesture_model.py --dataset-root state/mediapipe/custom_gesture_dataset
    state/mediapipe/model_maker_venv/bin/python hardware/piaicam/train_custom_gesture_model.py --dataset-root state/mediapipe/custom_gesture_dataset --dry-run
    state/mediapipe/model_maker_venv/bin/python hardware/piaicam/train_custom_gesture_model.py --dataset-root state/mediapipe/custom_gesture_dataset --epochs 12 --batch-size 4 --dropout-rate 0.1 --layer-width 128

Inputs
------
- A folder-per-label dataset root that includes at minimum `none`,
  `thumbs_up`, `thumbs_down`, and `peace_sign`.
- A Python 3.10 environment that can import `mediapipe_model_maker`.
- Optionally, explicit preprocessing / model-head hyperparameters.

Outputs
-------
- Writes one timestamped training run under
  `state/mediapipe/custom_gesture_training/`.
- Exports a `.task` bundle plus `labels.txt`.
- Writes `summary.json` and `dataset_audit.json` with run provenance.
- Atomically copies the final `.task` into
  `state/mediapipe/models/custom_gesture.task` by default and prints the Twinr
  env var hint for the Pi runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Sequence, cast
import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import tempfile

from PIL import Image, ImageFile, UnidentifiedImageError

try:
    from custom_gesture_workflow import (
        DEFAULT_DATASET_ROOT,
        DEFAULT_REQUIRED_LABELS,
        DEFAULT_RUNTIME_MODEL_DIR,
        DEFAULT_RUNTIME_MODEL_NAME,
        DEFAULT_TRAINING_RUNS_ROOT,
        current_timestamp_slug,
        runtime_env_hint,
    )
except Exception:  # pragma: no cover - fallback keeps this script self-contained.
    DEFAULT_DATASET_ROOT = Path("state/mediapipe/custom_gesture_dataset")
    DEFAULT_REQUIRED_LABELS = ("none", "thumbs_up", "thumbs_down", "peace_sign")
    DEFAULT_RUNTIME_MODEL_DIR = Path("state/mediapipe/models")
    DEFAULT_RUNTIME_MODEL_NAME = "custom_gesture.task"
    DEFAULT_TRAINING_RUNS_ROOT = Path("state/mediapipe/custom_gesture_training")

    def current_timestamp_slug() -> str:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    def runtime_env_hint(model_path: Path) -> str:
        return f"TWINR_CUSTOM_GESTURE_TASK={model_path}"


Image.MAX_IMAGE_PIXELS = 100_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = False

SUPPORTED_PYTHON = (3, 10)
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".gif",
    ".tif",
    ".tiff",
}


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Store one bounded custom-gesture training request."""

    dataset_root: Path
    output_dir: Path
    runtime_model_dir: Path
    model_name: str
    runtime_model_name: str
    validation_split: float = 0.10
    # Backward-compatible default for programmatic callers that previously
    # only knew about the validation split.
    test_split: float = 0.0
    epochs: int = 10
    batch_size: int = 2
    learning_rate: float = 0.001
    lr_decay: float = 0.99
    gamma: float = 2.0
    steps_per_epoch: int | None = None
    dropout_rate: float = 0.05
    layer_widths: tuple[int, ...] = ()
    min_images_per_label: int = 3
    required_labels: tuple[str, ...] = DEFAULT_REQUIRED_LABELS
    preprocessing_min_detection_confidence: float = 0.5
    preprocessing_shuffle: bool = True
    seed: int = 1337
    dedupe_exact: bool = False
    hand_audit: bool = False
    max_image_pixels: int = 20_000_000
    overwrite_output: bool = False
    dry_run: bool = False


@dataclass(frozen=True, slots=True)
class ImageRecord:
    label: str
    path: Path
    file_size_bytes: int


@dataclass(frozen=True, slots=True)
class AuditedImage:
    label: str
    path: Path
    file_size_bytes: int
    width: int
    height: int
    sha256: str


@dataclass(frozen=True, slots=True)
class SplitPaths:
    train: Path
    validation: Path
    test: Path | None


@dataclass(slots=True)
class DatasetAudit:
    root: str
    required_labels: list[str]
    preprocessing_min_detection_confidence: float
    dedupe_exact: bool
    hand_audit: bool
    raw_counts: dict[str, int] = field(default_factory=dict)
    accepted_counts: dict[str, int] = field(default_factory=dict)
    split_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    skipped_unreadable: dict[str, int] = field(default_factory=dict)
    skipped_symlink: dict[str, int] = field(default_factory=dict)
    skipped_duplicate: dict[str, int] = field(default_factory=dict)
    skipped_no_hand: dict[str, int] = field(default_factory=dict)
    accepted_examples: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_summary(self) -> dict[str, object]:
        return {
            "root": self.root,
            "required_labels": self.required_labels,
            "preprocessing_min_detection_confidence": self.preprocessing_min_detection_confidence,
            "dedupe_exact": self.dedupe_exact,
            "hand_audit": self.hand_audit,
            "raw_counts": self.raw_counts,
            "accepted_counts": self.accepted_counts,
            "split_counts": self.split_counts,
            "skipped_unreadable": self.skipped_unreadable,
            "skipped_symlink": self.skipped_symlink,
            "skipped_duplicate": self.skipped_duplicate,
            "skipped_no_hand": self.skipped_no_hand,
            "accepted_examples": self.accepted_examples,
            "warnings": list(self.warnings),
        }


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
        default=0.10,
        help=(
            "# BREAKING: Fraction of the audited dataset reserved for validation. "
            "Default is now 0.10 because the script also keeps a held-out test split."
        ),
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.10,
        help="Fraction of the audited dataset reserved for held-out testing.",
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
        "--lr-decay",
        type=float,
        default=0.99,
        help="Learning-rate decay for MediaPipe Model Maker HParams.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=2.0,
        help="Gamma for MediaPipe Model Maker focal loss.",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="Optional MediaPipe Model Maker steps_per_epoch override.",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.05,
        help="Classifier-head dropout rate.",
    )
    parser.add_argument(
        "--layer-width",
        action="append",
        type=int,
        default=None,
        help="Classifier-head hidden width. May be passed multiple times.",
    )
    parser.add_argument(
        "--min-images-per-label",
        type=int,
        default=3,
        help=(
            "# BREAKING: Fail if any required label has fewer usable images than this number after audit. "
            "The legacy default of 1 allowed empty or degenerate splits."
        ),
    )
    parser.add_argument(
        "--preprocessing-min-detection-confidence",
        type=float,
        default=0.5,
        help="MediaPipe hand-detection confidence threshold used during audit and dataset loading.",
    )
    parser.add_argument(
        "--preprocessing-shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether MediaPipe shuffles samples while preprocessing landmarks.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Seed used for reproducible stratified splits and TensorFlow initialization.",
    )
    parser.add_argument(
        "--dedupe-exact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable SHA-256 exact duplicate removal before splitting.",
    )
    parser.add_argument(
        "--hand-audit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify that every retained sample contains at least one detectable hand before training.",
    )
    parser.add_argument(
        "--max-image-pixels",
        type=int,
        default=20_000_000,
        help="Reject images above this pixel count during audit to avoid pathological resource usage.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Allow reusing an existing output directory by deleting prior staged datasets and artifacts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and audit the dataset, print planned paths, but do not train.",
    )
    return parser


def train_custom_gesture_model(
    config: TrainingConfig,
    *,
    api_loader=None,
) -> dict[str, object]:
    """Train one custom gesture model and return a JSON-friendly summary."""

    _ensure_clean_output_dir(config.output_dir, overwrite_output=config.overwrite_output)
    raw_records = _collect_raw_records(config.dataset_root, config.required_labels)
    dataset_audit, audited_records = _audit_dataset(raw_records, config)
    split_plan = _build_stratified_split_plan(
        audited_records,
        validation_split=config.validation_split,
        test_split=config.test_split,
        seed=config.seed,
    )
    dataset_audit.split_counts = {
        label: {name: len(paths) for name, paths in label_plan.items()}
        for label, label_plan in split_plan.items()
    }
    _ensure_minimum_examples(dataset_audit, config)

    staged_root = config.output_dir / "training_dataset"
    split_paths = SplitPaths(
        train=staged_root / "train",
        validation=staged_root / "validation",
        test=staged_root / "test" if config.test_split > 0 else None,
    )
    exported_model_path = config.output_dir / config.model_name
    runtime_model_path = config.runtime_model_dir / config.runtime_model_name
    audit_path = config.output_dir / "dataset_audit.json"

    summary: dict[str, object] = {
        "status": "dry_run" if config.dry_run else "trained",
        "required_labels": list(config.required_labels),
        "dataset": dataset_audit.to_summary(),
        "training": {
            "output_dir": str(config.output_dir),
            "model_name": config.model_name,
            "runtime_model_dir": str(config.runtime_model_dir),
            "runtime_model_name": config.runtime_model_name,
            "validation_split": config.validation_split,
            "test_split": config.test_split,
            "training_split": 1.0 - config.validation_split - config.test_split,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "lr_decay": config.lr_decay,
            "gamma": config.gamma,
            "steps_per_epoch": config.steps_per_epoch,
            "dropout_rate": config.dropout_rate,
            "layer_widths": list(config.layer_widths),
            "min_images_per_label": config.min_images_per_label,
            "preprocessing_min_detection_confidence": config.preprocessing_min_detection_confidence,
            "preprocessing_shuffle": config.preprocessing_shuffle,
            "seed": config.seed,
            "dedupe_exact": config.dedupe_exact,
            "hand_audit": config.hand_audit,
            "training_dataset_root": str(staged_root),
            "train_split_dir": str(split_paths.train),
            "validation_split_dir": str(split_paths.validation),
            "test_split_dir": str(split_paths.test) if split_paths.test else None,
        },
        "exported_model_path": str(exported_model_path),
        "runtime_model_path": str(runtime_model_path),
        "runtime_env_hint": runtime_env_hint(runtime_model_path),
        "dataset_audit_path": str(audit_path),
        "environment": _collect_environment_summary(),
        "warnings": list(dataset_audit.warnings),
    }
    _write_json_summary(audit_path, dataset_audit.to_summary())
    if config.dry_run:
        return summary

    if api_loader is None:
        _ensure_supported_python()
    loader = api_loader or _load_gesture_recognizer_api
    gesture_recognizer = loader()
    _configure_reproducibility(seed=config.seed)
    _stage_split_dataset(split_plan, split_paths, overwrite_output=config.overwrite_output)

    train_data = _load_training_dataset(
        gesture_recognizer=gesture_recognizer,
        dataset_root=split_paths.train,
        min_detection_confidence=config.preprocessing_min_detection_confidence,
        preprocessing_shuffle=config.preprocessing_shuffle,
        expected_size=sum(len(split_plan[label]["train"]) for label in config.required_labels),
        split_name="train",
    )
    validation_data = _load_training_dataset(
        gesture_recognizer=gesture_recognizer,
        dataset_root=split_paths.validation,
        min_detection_confidence=config.preprocessing_min_detection_confidence,
        preprocessing_shuffle=config.preprocessing_shuffle,
        expected_size=sum(len(split_plan[label]["validation"]) for label in config.required_labels),
        split_name="validation",
    )
    test_data = None
    if split_paths.test is not None:
        test_data = _load_training_dataset(
            gesture_recognizer=gesture_recognizer,
            dataset_root=split_paths.test,
            min_detection_confidence=config.preprocessing_min_detection_confidence,
            preprocessing_shuffle=config.preprocessing_shuffle,
            expected_size=sum(len(split_plan[label]["test"]) for label in config.required_labels),
            split_name="test",
        )

    hparams_kwargs: dict[str, Any] = {
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "lr_decay": config.lr_decay,
        "gamma": config.gamma,
        "export_dir": str(config.output_dir),
        "shuffle": True,
    }
    if config.steps_per_epoch is not None:
        hparams_kwargs["steps_per_epoch"] = config.steps_per_epoch
    hparams = gesture_recognizer.HParams(**hparams_kwargs)
    model_options_factory = getattr(gesture_recognizer, "ModelOptions", None)
    model_options = None
    if callable(model_options_factory):
        model_options = model_options_factory(
            dropout_rate=config.dropout_rate,
            layer_widths=list(config.layer_widths),
        )
    try:
        if model_options is not None:
            options = gesture_recognizer.GestureRecognizerOptions(
                model_options=model_options,
                hparams=hparams,
            )
        else:
            options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
    except TypeError:
        options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
    recognizer = gesture_recognizer.GestureRecognizer.create(
        train_data=train_data,
        validation_data=validation_data,
        options=options,
    )

    validation_evaluation = recognizer.evaluate(validation_data)
    test_evaluation = recognizer.evaluate(test_data) if test_data is not None else None
    recognizer.export_model(model_name=config.model_name)
    recognizer.export_labels(str(config.output_dir))
    copied_runtime_model = _atomic_copy_model_to_runtime(
        source_model_path=exported_model_path,
        runtime_model_dir=config.runtime_model_dir,
        runtime_model_name=config.runtime_model_name,
    )
    summary["runtime_model_path"] = str(copied_runtime_model)
    summary["runtime_env_hint"] = runtime_env_hint(copied_runtime_model)

    dataset_summary = cast(dict[str, object], summary["dataset"])
    dataset_summary["label_names"] = list(getattr(train_data, "label_names", ()))
    dataset_summary["num_classes"] = int(
        getattr(train_data, "num_classes", len(config.required_labels)) or len(config.required_labels)
    )
    dataset_summary["training_size"] = _dataset_size(train_data)
    dataset_summary["validation_size"] = _dataset_size(validation_data)
    dataset_summary["test_size"] = _dataset_size(test_data)
    summary["evaluation"] = {
        "validation": _normalize_evaluation(validation_evaluation),
        "test": _normalize_evaluation(test_evaluation) if test_evaluation is not None else None,
    }
    summary["artifacts"] = {
        "exported_model": _artifact_summary(exported_model_path),
        "runtime_model": _artifact_summary(copied_runtime_model),
        "labels": _artifact_summary(config.output_dir / "labels.txt"),
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    """Run the bounded custom gesture training helper."""

    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir = args.output_dir or (DEFAULT_TRAINING_RUNS_ROOT / current_timestamp_slug())
    config = TrainingConfig(
        dataset_root=args.dataset_root,
        output_dir=output_dir,
        runtime_model_dir=args.runtime_model_dir,
        model_name=_sanitize_artifact_name(args.model_name, default="custom_gesture.task", required_suffix=".task"),
        runtime_model_name=_sanitize_artifact_name(
            args.runtime_model_name,
            default=DEFAULT_RUNTIME_MODEL_NAME,
            required_suffix=".task",
        ),
        validation_split=_coerce_holdout_split(args.validation_split, name="validation"),
        test_split=_coerce_holdout_split(args.test_split, name="test"),
        epochs=max(1, int(args.epochs)),
        batch_size=max(1, int(args.batch_size)),
        learning_rate=max(1e-6, float(args.learning_rate)),
        lr_decay=_coerce_fraction(args.lr_decay, name="lr_decay", allow_zero=False),
        gamma=max(0.0, float(args.gamma)),
        steps_per_epoch=max(1, int(args.steps_per_epoch)) if args.steps_per_epoch is not None else None,
        dropout_rate=_coerce_fraction(args.dropout_rate, name="dropout_rate"),
        layer_widths=tuple(max(1, int(item)) for item in (args.layer_width or ())),
        min_images_per_label=max(1, int(args.min_images_per_label)),
        required_labels=_normalize_required_labels(args.required_label or DEFAULT_REQUIRED_LABELS),
        preprocessing_min_detection_confidence=_coerce_fraction(
            args.preprocessing_min_detection_confidence,
            name="preprocessing_min_detection_confidence",
        ),
        preprocessing_shuffle=bool(args.preprocessing_shuffle),
        seed=int(args.seed),
        dedupe_exact=bool(args.dedupe_exact),
        hand_audit=bool(args.hand_audit),
        max_image_pixels=max(1, int(args.max_image_pixels)),
        overwrite_output=bool(args.overwrite_output),
        dry_run=bool(args.dry_run),
    )
    _validate_split_budget(config.validation_split, config.test_split)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary = train_custom_gesture_model(config)
    _write_json_summary(config.output_dir / "summary.json", summary)
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def _ensure_supported_python() -> None:
    """Fail early with the interpreter version Model Maker still supports in practice."""

    if sys.version_info[:2] != SUPPORTED_PYTHON:
        raise RuntimeError(
            "mediapipe_model_maker_requires_python_3_10:"
            f"detected_{platform.python_version()}"
        )


def _normalize_required_labels(values: Sequence[str]) -> tuple[str, ...]:
    """Deduplicate and validate required label names while preserving order."""

    normalized: list[str] = []
    seen: set[str] = set()
    for raw in values:
        label = str(raw).strip()
        if not label:
            continue
        if any(ch in label for ch in ("/", "\\", "\x00")):
            raise ValueError(f"custom_gesture_label_invalid:{label!r}")
        if label not in seen:
            normalized.append(label)
            seen.add(label)
    if not normalized:
        raise ValueError("custom_gesture_required_labels_empty")
    if "none" not in seen:
        raise ValueError("custom_gesture_required_labels_must_include_none")
    return tuple(normalized)


def _sanitize_artifact_name(value: object, *, default: str, required_suffix: str | None = None) -> str:
    """Reject path traversal and normalize one artifact filename."""

    candidate = str(value or default).strip() or default
    path = Path(candidate)
    if path.name != candidate or candidate in {".", ".."}:
        raise ValueError(f"custom_gesture_artifact_name_invalid:{candidate!r}")
    if any(part in {"..", ""} for part in path.parts):
        raise ValueError(f"custom_gesture_artifact_name_invalid:{candidate!r}")
    if any(ch in candidate for ch in ("/", "\\", "\x00")):
        raise ValueError(f"custom_gesture_artifact_name_invalid:{candidate!r}")
    if required_suffix and not candidate.endswith(required_suffix):
        raise ValueError(f"custom_gesture_artifact_name_must_end_with_{required_suffix}:{candidate!r}")
    return candidate


def _coerce_holdout_split(value: object, *, name: str) -> float:
    """Clamp one holdout split into a safe [0, 0.49] interval."""

    split = float(cast(Any, value))
    if split < 0.0 or split >= 0.5:
        raise ValueError(f"custom_gesture_{name}_split_invalid")
    return split


def _coerce_fraction(value: object, *, name: str, allow_zero: bool = True) -> float:
    """Validate a generic [0, 1] or (0, 1] float parameter."""

    numeric = float(cast(Any, value))
    lower_ok = numeric >= 0.0 if allow_zero else numeric > 0.0
    if not lower_ok or numeric > 1.0:
        raise ValueError(f"custom_gesture_{name}_invalid")
    return numeric


def _validate_split_budget(validation_split: float, test_split: float) -> None:
    """Ensure the requested split fractions leave room for training."""

    if validation_split + test_split >= 0.5:
        raise ValueError("custom_gesture_holdout_split_budget_invalid")


def _ensure_clean_output_dir(output_dir: Path, *, overwrite_output: bool) -> None:
    """Prevent stale artifacts from contaminating a new run."""

    if output_dir.exists() and not output_dir.is_dir():
        raise RuntimeError(f"custom_gesture_output_dir_not_directory:{output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite_output:
        # BREAKING: legacy behavior silently reused existing run directories.
        raise RuntimeError(
            "custom_gesture_output_dir_not_empty_use_overwrite_output:"
            f"{output_dir}"
        )
    if overwrite_output and output_dir.exists():
        for child in list(output_dir.iterdir()):
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink(missing_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)


def _collect_raw_records(dataset_root: Path, required_labels: Sequence[str]) -> dict[str, list[ImageRecord]]:
    """Scan one folder-per-label dataset and collect candidate image files."""

    root = dataset_root.expanduser().resolve(strict=False)
    if not root.exists() or not root.is_dir():
        raise RuntimeError(f"custom_gesture_dataset_root_missing:{root}")

    records: dict[str, list[ImageRecord]] = {}
    missing: list[str] = []
    for label in required_labels:
        label_dir = root / label
        if not label_dir.exists() or not label_dir.is_dir():
            missing.append(label)
            continue
        label_records: list[ImageRecord] = []
        for path in sorted(label_dir.iterdir()):
            if path.name.startswith("."):
                continue
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                size = path.lstat().st_size
            except OSError:
                size = 0
            label_records.append(ImageRecord(label=label, path=path, file_size_bytes=size))
        records[label] = label_records
    if missing:
        raise RuntimeError("custom_gesture_required_label_missing:" + ",".join(missing))
    return records


def _audit_dataset(
    raw_records: dict[str, list[ImageRecord]],
    config: TrainingConfig,
) -> tuple[DatasetAudit, dict[str, list[AuditedImage]]]:
    """Audit readability, duplicates, and hand detectability before training."""

    audit = DatasetAudit(
        root=str(config.dataset_root),
        required_labels=list(config.required_labels),
        preprocessing_min_detection_confidence=config.preprocessing_min_detection_confidence,
        dedupe_exact=config.dedupe_exact,
        hand_audit=config.hand_audit,
    )
    global_hashes: dict[str, tuple[str, Path]] = {}
    audited: dict[str, list[AuditedImage]] = {label: [] for label in config.required_labels}
    hand_detector = _create_hand_detector(config.preprocessing_min_detection_confidence) if config.hand_audit else None
    try:
        for label in config.required_labels:
            records = raw_records.get(label, [])
            audit.raw_counts[label] = len(records)
            audit.skipped_unreadable[label] = 0
            audit.skipped_symlink[label] = 0
            audit.skipped_duplicate[label] = 0
            audit.skipped_no_hand[label] = 0
            audit.accepted_examples[label] = []
            for record in records:
                if record.path.is_symlink():
                    audit.skipped_symlink[label] += 1
                    continue
                audited_image = _audit_single_image(
                    record,
                    max_image_pixels=config.max_image_pixels,
                    hand_detector=hand_detector,
                )
                if audited_image is None:
                    audit.skipped_unreadable[label] += 1
                    continue
                if config.dedupe_exact:
                    duplicate_source = global_hashes.get(audited_image.sha256)
                    if duplicate_source is not None:
                        audit.skipped_duplicate[label] += 1
                        source_label, source_path = duplicate_source
                        if source_label != label:
                            audit.warnings.append(
                                "cross_label_exact_duplicate:"
                                f"{source_label}:{source_path.name}:{label}:{audited_image.path.name}"
                            )
                        continue
                    global_hashes[audited_image.sha256] = (label, audited_image.path)
                if config.hand_audit and not _hand_detector_accepts(hand_detector, audited_image.path):
                    audit.skipped_no_hand[label] += 1
                    continue
                audited[label].append(audited_image)
                if len(audit.accepted_examples[label]) < 5:
                    audit.accepted_examples[label].append(str(audited_image.path))
            audit.accepted_counts[label] = len(audited[label])
            if audit.accepted_counts[label] < 100:
                audit.warnings.append(
                    f"label_{label}_has_only_{audit.accepted_counts[label]}_usable_images_after_audit_recommended_at_least_100"
                )
    finally:
        if hand_detector is not None:
            hand_detector.close()
    return audit, audited


def _create_hand_detector(min_detection_confidence: float):
    """Create a MediaPipe Hands detector for offline image auditing."""

    import mediapipe as mp  # pylint: disable=import-error

    return mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_detection_confidence,
    )


def _audit_single_image(
    record: ImageRecord,
    *,
    max_image_pixels: int,
    hand_detector,
) -> AuditedImage | None:
    """Verify one image is readable and bounded."""

    try:
        with Image.open(record.path) as image:
            image.verify()
        with Image.open(record.path) as image:
            image = image.convert("RGB")
            width, height = image.size
            if width <= 0 or height <= 0:
                return None
            if width * height > max_image_pixels:
                return None
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return None
    return AuditedImage(
        label=record.label,
        path=record.path,
        file_size_bytes=record.file_size_bytes,
        width=width,
        height=height,
        sha256=_sha256_file(record.path),
    )


def _hand_detector_accepts(hand_detector, image_path: Path) -> bool:
    """Return whether MediaPipe can detect at least one hand in the image."""

    if hand_detector is None:
        return True
    try:
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            result = hand_detector.process(_pil_to_numpy(rgb))
    except Exception:
        return False
    return bool(getattr(result, "multi_hand_landmarks", None))


def _pil_to_numpy(image: Image.Image):
    """Convert a Pillow image into an RGB ndarray lazily."""

    import numpy as np

    return np.asarray(image)


def _build_stratified_split_plan(
    audited_records: dict[str, list[AuditedImage]],
    *,
    validation_split: float,
    test_split: float,
    seed: int,
) -> dict[str, dict[str, list[Path]]]:
    """Create deterministic per-label train/validation/test splits."""

    rng = random.Random(seed)
    plan: dict[str, dict[str, list[Path]]] = {}
    train_fraction = 1.0 - validation_split - test_split
    for label, records in audited_records.items():
        shuffled = list(records)
        rng.shuffle(shuffled)
        counts = _allocate_split_counts(
            total=len(shuffled),
            fractions={
                "train": train_fraction,
                "validation": validation_split,
                "test": test_split,
            },
        )
        cursor = 0
        label_plan: dict[str, list[Path]] = {}
        for split_name in ("train", "validation", "test"):
            split_count = counts[split_name]
            label_plan[split_name] = [item.path for item in shuffled[cursor : cursor + split_count]]
            cursor += split_count
        plan[label] = label_plan
    return plan


def _allocate_split_counts(total: int, fractions: dict[str, float]) -> dict[str, int]:
    """Allocate per-class counts with at least one sample in every non-zero split."""

    positive = {name: frac for name, frac in fractions.items() if frac > 0.0}
    if total < len(positive):
        raise RuntimeError(
            "custom_gesture_too_few_examples_for_requested_splits:"
            f"total={total}:splits={len(positive)}"
        )
    counts = {name: 0 for name in fractions}
    for name in positive:
        counts[name] = 1
    remaining = total - len(positive)
    if remaining <= 0:
        return counts
    base_allocations = {name: remaining * frac for name, frac in positive.items()}
    for name, value in base_allocations.items():
        whole = int(math.floor(value))
        counts[name] += whole
        remaining -= whole
    remainders = sorted(
        ((value - math.floor(value), name) for name, value in base_allocations.items()),
        reverse=True,
    )
    for _, name in remainders[:remaining]:
        counts[name] += 1
    return counts


def _ensure_minimum_examples(audit: DatasetAudit, config: TrainingConfig) -> None:
    """Reject audited datasets that cannot support the requested splits."""

    required_non_zero_splits = 1 + int(config.validation_split > 0.0) + int(config.test_split > 0.0)
    effective_min = max(config.min_images_per_label, required_non_zero_splits)
    too_small = {
        label: count
        for label, count in audit.accepted_counts.items()
        if count < effective_min
    }
    if too_small:
        details = ",".join(f"{label}:{count}" for label, count in sorted(too_small.items()))
        raise RuntimeError(
            "custom_gesture_minimum_examples_not_met_after_audit:"
            f"required={effective_min}:{details}"
        )


def _stage_split_dataset(
    split_plan: dict[str, dict[str, list[Path]]],
    split_paths: SplitPaths,
    *,
    overwrite_output: bool,
) -> None:
    """Stage deterministic split directories using ordinary file copies."""

    all_targets = [split_paths.train, split_paths.validation]
    if split_paths.test is not None:
        all_targets.append(split_paths.test)
    for target in all_targets:
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
    for label, label_plan in split_plan.items():
        for split_name, paths in label_plan.items():
            if split_name == "test" and split_paths.test is None:
                continue
            target_root = getattr(split_paths, split_name)
            if target_root is None:
                continue
            label_dir = target_root / label
            label_dir.mkdir(parents=True, exist_ok=True)
            for index, source in enumerate(paths, start=1):
                target_name = f"{index:06d}_{source.name}"
                shutil.copy2(source, label_dir / target_name)


def _load_gesture_recognizer_api():
    """Import the MediaPipe Model Maker API with a compatibility-focused error."""

    try:
        import pkg_resources  # noqa: F401
        from mediapipe_model_maker import gesture_recognizer
    except ModuleNotFoundError as exc:
        missing_name = str(getattr(exc, "name", "") or "")
        if missing_name == "pkg_resources" or "pkg_resources" in str(exc):
            raise RuntimeError("mediapipe_model_maker_requires_setuptools_lt_81") from exc
        if missing_name.startswith("keras") or "keras.src.engine" in str(exc):
            raise RuntimeError("mediapipe_model_maker_requires_keras_2_and_tensorflow_2_15") from exc
        raise RuntimeError(f"mediapipe_model_maker_unavailable:{missing_name or 'unknown'}") from exc
    except Exception as exc:  # pragma: no cover - depends on local training env.
        raise RuntimeError("mediapipe_model_maker_unavailable") from exc
    return gesture_recognizer


def _load_training_dataset(
    *,
    gesture_recognizer,
    dataset_root: Path,
    min_detection_confidence: float,
    preprocessing_shuffle: bool,
    expected_size: int,
    split_name: str,
):
    """Load one Model Maker dataset and map opaque hand-detection failures."""

    dataset_kwargs: dict[str, Any] = {
        "dirname": str(dataset_root),
    }
    preprocessing_factory = getattr(gesture_recognizer, "HandDataPreprocessingParams", None)
    if callable(preprocessing_factory):
        dataset_kwargs["hparams"] = preprocessing_factory(
            shuffle=preprocessing_shuffle,
            min_detection_confidence=min_detection_confidence,
        )
    try:
        try:
            dataset = gesture_recognizer.Dataset.from_folder(**dataset_kwargs)
        except TypeError:
            dataset = gesture_recognizer.Dataset.from_folder(str(dataset_root))
    except ValueError as exc:
        if "No valid hand is detected." in str(exc):
            raise RuntimeError(
                "custom_gesture_dataset_no_detectable_hands:"
                f"{Path(dataset_root)}:"
                "recapture_with_hand_fully_visible"
            ) from exc
        raise
    actual_size = _dataset_size(dataset)
    if actual_size is not None and actual_size != expected_size:
        raise RuntimeError(
            "custom_gesture_dataset_attrition_after_model_maker_preprocessing:"
            f"split={split_name}:expected={expected_size}:actual={actual_size}"
        )
    return dataset


def _configure_reproducibility(*, seed: int) -> None:
    """Seed Python, NumPy, and TensorFlow for repeatable runs."""

    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import tensorflow as tf  # pylint: disable=import-error

        tf.random.set_seed(seed)
    except Exception:
        pass


def _atomic_copy_model_to_runtime(
    *,
    source_model_path: Path,
    runtime_model_dir: Path,
    runtime_model_name: str,
) -> Path:
    """Atomically publish the runtime task bundle to avoid partial reads."""

    if not source_model_path.exists():
        raise RuntimeError(f"custom_gesture_export_missing:{source_model_path}")
    runtime_model_dir = runtime_model_dir.expanduser().resolve(strict=False)
    runtime_model_dir.mkdir(parents=True, exist_ok=True)
    destination = runtime_model_dir / runtime_model_name
    with tempfile.NamedTemporaryFile(
        dir=str(runtime_model_dir),
        prefix=f".{runtime_model_name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
    try:
        shutil.copy2(source_model_path, temp_path)
        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
    return destination


def _artifact_summary(path: Path) -> dict[str, object] | None:
    """Return size and digest metadata for one artifact when present."""

    if not path.exists():
        return None
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "sha256": _sha256_file(path),
    }


def _sha256_file(path: Path) -> str:
    """Stream a file into a SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_summary(path: Path, payload: dict[str, object]) -> None:
    """Write JSON deterministically via atomic replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as handle:
        handle.write(rendered)
        temp_path = Path(handle.name)
    try:
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _collect_environment_summary() -> dict[str, object]:
    """Capture the package versions that matter for reproducing the run."""

    packages: dict[str, str | None] = {}
    for name in ("mediapipe-model-maker", "mediapipe", "tensorflow", "keras", "setuptools", "Pillow"):
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def _dataset_size(dataset: object) -> int | None:
    """Return the size for one Model Maker dataset split when available."""

    if dataset is None:
        return None
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
