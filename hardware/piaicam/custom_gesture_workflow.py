"""Share bounded dataset and packaging helpers for custom gesture workflows.

This module keeps filesystem and summary logic separate from the Pi-side
capture script and the local MediaPipe Model Maker training script. It owns
dataset-label normalization, dataset validation, capture target planning, and
runtime-model copy helpers so the workflow stays predictable and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil


DEFAULT_DATASET_ROOT = Path("state/mediapipe/custom_gesture_dataset")
DEFAULT_TRAINING_RUNS_ROOT = Path("state/mediapipe/custom_gesture_training")
DEFAULT_RUNTIME_MODEL_DIR = Path("state/mediapipe/models")
DEFAULT_RUNTIME_MODEL_NAME = "custom_gesture.task"
DEFAULT_REQUIRED_LABELS = ("none", "ok_sign", "middle_finger")
SUPPORTED_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


@dataclass(frozen=True, slots=True)
class DatasetLabelSummary:
    """Describe one normalized label directory in a custom gesture dataset."""

    label: str
    directory: Path
    count: int


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Describe one validated dataset root for custom gesture training."""

    root: Path
    labels: tuple[DatasetLabelSummary, ...]

    @property
    def label_names(self) -> tuple[str, ...]:
        """Return the normalized label names in deterministic order."""

        return tuple(label.label for label in self.labels)

    @property
    def total_images(self) -> int:
        """Return the total number of supported image files in the dataset."""

        return sum(label.count for label in self.labels)

    def counts_by_label(self) -> dict[str, int]:
        """Return a JSON-friendly label-to-count mapping."""

        return {label.label: label.count for label in self.labels}

    def to_summary(self) -> dict[str, object]:
        """Convert the manifest into a JSON-friendly summary structure."""

        return {
            "dataset_root": str(self.root),
            "labels": [
                {
                    "label": label.label,
                    "directory": str(label.directory),
                    "count": label.count,
                }
                for label in self.labels
            ],
            "total_images": self.total_images,
        }


def normalize_label_name(value: str) -> str:
    """Normalize one user-provided label into a stable dataset directory name."""

    compact = " ".join(str(value or "").strip().lower().replace("-", " ").split())
    if not compact:
        raise ValueError("custom_gesture_label_empty")
    return compact.replace(" ", "_")


def current_timestamp_slug() -> str:
    """Return one stable UTC timestamp slug for filenames and run folders."""

    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def collect_dataset_manifest(
    dataset_root: Path,
    *,
    required_labels: tuple[str, ...] = DEFAULT_REQUIRED_LABELS,
) -> DatasetManifest:
    """Validate the dataset layout and summarize supported image files."""

    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"custom_gesture_dataset_missing:{root}")
    if not root.is_dir():
        raise ValueError(f"custom_gesture_dataset_not_directory:{root}")

    normalized_required = _normalize_required_labels(required_labels)
    counts: dict[str, DatasetLabelSummary] = {}
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        label = normalize_label_name(directory.name)
        image_count = len(_list_image_files(directory))
        counts[label] = DatasetLabelSummary(label=label, directory=directory, count=image_count)

    missing = [label for label in normalized_required if label not in counts]
    if missing:
        raise ValueError(f"custom_gesture_dataset_missing_labels:{','.join(missing)}")

    empty = [label for label in normalized_required if counts[label].count <= 0]
    if empty:
        raise ValueError(f"custom_gesture_dataset_empty_labels:{','.join(empty)}")

    return DatasetManifest(
        root=root,
        labels=tuple(counts[label] for label in sorted(counts)),
    )


def ensure_minimum_examples(manifest: DatasetManifest, *, min_images_per_label: int) -> None:
    """Require a minimum number of examples per label before training."""

    minimum = max(1, int(min_images_per_label))
    too_small = [label.label for label in manifest.labels if label.count < minimum]
    if too_small:
        raise ValueError(
            f"custom_gesture_dataset_too_small:{minimum}:{','.join(sorted(too_small))}"
        )


def plan_capture_targets(
    dataset_root: Path,
    *,
    label: str,
    count: int,
    timestamp_slug: str | None = None,
    prefix: str | None = None,
) -> tuple[str, Path, tuple[Path, ...]]:
    """Plan deterministic JPEG output paths for one capture session."""

    normalized_label = normalize_label_name(label)
    capture_count = max(0, int(count))
    if capture_count <= 0:
        raise ValueError("custom_gesture_capture_count_invalid")
    slug = timestamp_slug or current_timestamp_slug()
    stem_prefix = normalize_label_name(prefix) if prefix else normalized_label
    label_dir = Path(dataset_root) / normalized_label
    targets = tuple(
        label_dir / f"{stem_prefix}-{slug}-{index:04d}.jpg"
        for index in range(1, capture_count + 1)
    )
    return normalized_label, label_dir, targets


def copy_model_to_runtime(
    *,
    source_model_path: Path,
    runtime_model_dir: Path,
    runtime_model_name: str = DEFAULT_RUNTIME_MODEL_NAME,
) -> Path:
    """Copy the exported task bundle into the runtime model directory."""

    source = Path(source_model_path)
    if not source.exists():
        raise FileNotFoundError(f"custom_gesture_model_missing:{source}")
    runtime_dir = Path(runtime_model_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = runtime_dir / runtime_model_name
    shutil.copy2(source, runtime_path)
    return runtime_path


def runtime_env_hint(model_path: Path) -> str:
    """Return the env assignment Twinr expects for a custom gesture model."""

    return (
        "TWINR_PROACTIVE_LOCAL_CAMERA_MEDIAPIPE_CUSTOM_GESTURE_MODEL_PATH="
        f"{Path(model_path).as_posix()}"
    )


def write_json_summary(path: Path, payload: dict[str, object]) -> None:
    """Write one indented JSON summary to disk."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _list_image_files(directory: Path) -> list[Path]:
    """List supported still-image files in deterministic order."""

    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )


def _normalize_required_labels(required_labels: tuple[str, ...]) -> tuple[str, ...]:
    """Normalize and deduplicate the required-label list."""

    normalized: list[str] = []
    for label in required_labels:
        normalized_label = normalize_label_name(label)
        if normalized_label not in normalized:
            normalized.append(normalized_label)
    if "none" not in normalized:
        normalized.insert(0, "none")
    return tuple(normalized)
