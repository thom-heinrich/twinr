# CHANGELOG: 2026-03-27
# BUG-1: stage_required_label_dataset now stages only the validated required labels tracked in the manifest; the old default flow could silently stage exploratory labels and change the classifier head.
# BUG-2: ensure_minimum_examples now checks the validated required labels by default instead of every discovered label, so legacy exploratory labels no longer block training.
# BUG-3: collect_dataset_manifest now rejects normalized-label collisions instead of silently overwriting one directory summary with another.
# BUG-4: plan_capture_targets now sanitizes label/prefix/slug inputs and avoids filename collisions instead of planning overwrites.
# SEC-1: normalize_label_name and runtime-model filename validation now block absolute paths, path traversal, and separator injection.
# SEC-2: dataset scanning and staging now reject symlinked label directories and symlinked image files to prevent path escape and unintended file copies.
# SEC-3: model deployment and JSON summary writes are now atomic, so power loss or crashes do not leave partially written runtime artifacts.
# IMP-1: staging is now atomic through a sibling temp directory and uses hard-link-first materialization with copy fallback for lower write amplification on edge storage.
# IMP-2: WebP inputs remain accepted for dataset import, but training staging now auto-transcodes them to PNG because MediaPipe/Keras folder loaders are stricter than the previous helper implied.
# IMP-3: dataset validation is now image-aware: it can reject corrupt, animated, or oversized still images before training fails deep inside MediaPipe.
# IMP-4: manifest summaries now carry the validated required-label set and richer counts so downstream training and reporting stay reproducible.

"""Share bounded dataset and packaging helpers for custom gesture workflows.

This module keeps filesystem and summary logic separate from the Pi-side
capture script, public seed import script, and the local MediaPipe Model Maker
training script. It owns dataset-label normalization, dataset validation,
capture target planning, filtered training-dataset staging, and runtime-model
copy helpers so the workflow stays predictable and testable.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePath
import json
import os
import re
import shutil
import tempfile
import unicodedata
import warnings

DEFAULT_DATASET_ROOT = Path("state/mediapipe/custom_gesture_dataset")
DEFAULT_TRAINING_RUNS_ROOT = Path("state/mediapipe/custom_gesture_training")
DEFAULT_RUNTIME_MODEL_DIR = Path("state/mediapipe/models")
DEFAULT_RUNTIME_MODEL_NAME = "custom_gesture.task"
DEFAULT_REQUIRED_LABELS = ("none", "thumbs_up", "thumbs_down", "peace_sign")
SUPPORTED_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
TRAINING_COMPATIBLE_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
DEFAULT_STAGE_LINK_MODE = "auto"
DEFAULT_MAX_IMAGE_PIXELS = 32_000_000


@dataclass(frozen=True, slots=True)
class DatasetLabelSummary:
    """Describe one normalized label directory in a custom gesture dataset."""

    label: str
    directory: Path
    count: int
    training_compatible_count: int = 0
    transcode_required_count: int = 0
    total_bytes: int = 0

    def to_summary(self) -> dict[str, object]:
        """Return a JSON-friendly summary for one label directory."""

        return {
            "label": self.label,
            "directory": str(self.directory),
            "count": self.count,
            "training_compatible_count": self.training_compatible_count,
            "transcode_required_count": self.transcode_required_count,
            "total_bytes": self.total_bytes,
        }


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Describe one validated dataset root for custom gesture training."""

    root: Path
    labels: tuple[DatasetLabelSummary, ...]
    required_label_names: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()

    @property
    def label_names(self) -> tuple[str, ...]:
        """Return the normalized label names in deterministic order."""

        return tuple(label.label for label in self.labels)

    @property
    def selected_label_names(self) -> tuple[str, ...]:
        """Return the validated training labels tracked by this manifest."""

        if self.required_label_names:
            known = set(self.label_names)
            return tuple(label for label in self.required_label_names if label in known)
        return self.label_names

    @property
    def total_images(self) -> int:
        """Return the total number of supported image files in the dataset."""

        return sum(label.count for label in self.labels)

    @property
    def selected_total_images(self) -> int:
        """Return the total images for the tracked training labels only."""

        selected = set(self.selected_label_names)
        return sum(label.count for label in self.labels if label.label in selected)

    def counts_by_label(self, *, only_required: bool = False) -> dict[str, int]:
        """Return a JSON-friendly label-to-count mapping."""

        selected = set(self.required_label_names) if only_required and self.required_label_names else None
        return {
            label.label: label.count
            for label in self.labels
            if selected is None or label.label in selected
        }

    def to_summary(self) -> dict[str, object]:
        """Convert the manifest into a JSON-friendly summary structure."""

        return {
            "dataset_root": str(self.root),
            "required_labels": list(self.required_label_names),
            "selected_label_names": list(self.selected_label_names),
            "labels": [label.to_summary() for label in self.labels],
            "total_images": self.total_images,
            "selected_total_images": self.selected_total_images,
            "issues": list(self.issues),
        }


@dataclass(frozen=True, slots=True)
class _CollectedImageFile:
    """Describe one validated source image file."""

    path: Path
    suffix: str
    size_bytes: int
    needs_training_transcode: bool


def normalize_label_name(value: str) -> str:
    """Normalize one user-provided label into a stable dataset directory name."""

    # BREAKING: normalization is now security-hardened; unsafe path and punctuation
    # characters are collapsed so labels can no longer escape the dataset root.
    text = unicodedata.normalize("NFKD", str(value or "")).casefold().strip()
    ascii_text = text.encode("ascii", "ignore").decode("ascii")
    compact = re.sub(r"[\s\-]+", "_", ascii_text)
    compact = re.sub(r"[^a-z0-9_]+", "_", compact)
    compact = re.sub(r"_+", "_", compact).strip("._")
    if not compact:
        raise ValueError("custom_gesture_label_empty")
    if compact in {".", ".."}:
        raise ValueError("custom_gesture_label_invalid")
    return compact


def current_timestamp_slug() -> str:
    """Return one stable UTC timestamp slug for filenames and run folders."""

    # BREAKING: the slug now includes microseconds to avoid same-second collisions.
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def collect_dataset_manifest(
    dataset_root: Path,
    *,
    required_labels: tuple[str, ...] = DEFAULT_REQUIRED_LABELS,
    include_only_required: bool = False,
    validate_images: bool = True,
    max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
) -> DatasetManifest:
    """Validate the dataset layout and summarize supported image files."""

    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"custom_gesture_dataset_missing:{root}")
    if not root.is_dir():
        raise ValueError(f"custom_gesture_dataset_not_directory:{root}")

    normalized_required = _normalize_required_labels(required_labels)
    label_directories = _discover_label_directories(root)
    available_labels = {label for label, _directory in label_directories}

    missing = [label for label in normalized_required if label not in available_labels]
    if missing:
        raise ValueError(f"custom_gesture_dataset_missing_labels:{','.join(missing)}")

    counts: dict[str, DatasetLabelSummary] = {}
    for label, directory in label_directories:
        if include_only_required and label not in normalized_required:
            continue
        image_files = _collect_image_files(
            directory,
            validate_images=validate_images,
            max_image_pixels=max_image_pixels,
        )
        counts[label] = DatasetLabelSummary(
            label=label,
            directory=directory,
            count=len(image_files),
            training_compatible_count=sum(
                1 for image_file in image_files if not image_file.needs_training_transcode
            ),
            transcode_required_count=sum(
                1 for image_file in image_files if image_file.needs_training_transcode
            ),
            total_bytes=sum(image_file.size_bytes for image_file in image_files),
        )

    empty = [label for label in normalized_required if counts[label].count <= 0]
    if empty:
        raise ValueError(f"custom_gesture_dataset_empty_labels:{','.join(empty)}")

    issues = _build_manifest_issues(counts, normalized_required)
    return DatasetManifest(
        root=root,
        labels=tuple(counts[label] for label in sorted(counts)),
        required_label_names=normalized_required,
        issues=issues,
    )


def stage_required_label_dataset(
    *,
    source_manifest: DatasetManifest,
    output_root: Path,
    required_labels: tuple[str, ...] | None = None,
    link_mode: str = DEFAULT_STAGE_LINK_MODE,
    validate_images: bool = True,
    max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
) -> DatasetManifest:
    """Copy the selected manifest labels into one clean training dataset root.

    MediaPipe Model Maker loads every label directory present under the dataset
    root. Twinr's gesture workflow therefore stages just the required labels
    into a fresh training directory before calling ``Dataset.from_folder`` so
    legacy or exploratory labels do not silently change the classifier head.

    WebP inputs are automatically transcoded to PNG inside the staged training
    root because the upstream Keras directory loader accepts JPEG, PNG, BMP and
    GIF, but not WebP.
    """

    target_root = Path(output_root)
    if target_root.exists() or target_root.is_symlink():
        raise ValueError(f"custom_gesture_training_dataset_target_exists:{target_root}")

    # BREAKING: by default this now stages source_manifest.required_label_names
    # instead of every discovered label. Pass required_labels=source_manifest.label_names
    # to preserve the legacy "stage everything in the manifest" behavior.
    selected_label_names = _normalize_required_labels(required_labels) if required_labels else (
        source_manifest.required_label_names or source_manifest.label_names
    )
    label_index = {label.label: label for label in source_manifest.labels}
    missing = [label for label in selected_label_names if label not in label_index]
    if missing:
        raise ValueError(f"custom_gesture_dataset_missing_labels:{','.join(missing)}")

    target_root.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(
        tempfile.mkdtemp(
            prefix=f".{target_root.name}.tmp-",
            dir=str(target_root.parent),
        )
    )
    staged_labels: list[DatasetLabelSummary] = []
    try:
        for label_name in selected_label_names:
            label_summary = label_index[label_name]
            target_dir = temp_root / label_summary.label
            target_dir.mkdir(parents=True, exist_ok=False)
            image_files = _collect_image_files(
                label_summary.directory,
                validate_images=validate_images,
                max_image_pixels=max_image_pixels,
            )
            staged_count = 0
            staged_training_compatible_count = 0
            staged_transcode_count = 0
            staged_total_bytes = 0
            for image_file in image_files:
                target_path = _plan_staged_image_path(target_dir, image_file)
                if image_file.needs_training_transcode:
                    _transcode_image_for_training(image_file.path, target_path)
                    staged_transcode_count += 1
                else:
                    _materialize_file(image_file.path, target_path, link_mode=link_mode)
                    staged_training_compatible_count += 1
                staged_count += 1
                staged_total_bytes += target_path.stat().st_size
            staged_labels.append(
                DatasetLabelSummary(
                    label=label_summary.label,
                    directory=target_dir,
                    count=staged_count,
                    training_compatible_count=staged_training_compatible_count + staged_transcode_count,
                    transcode_required_count=staged_transcode_count,
                    total_bytes=staged_total_bytes,
                )
            )
        os.replace(temp_root, target_root)
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise

    finalized_labels = tuple(
        DatasetLabelSummary(
            label=summary.label,
            directory=target_root / summary.label,
            count=summary.count,
            training_compatible_count=summary.training_compatible_count,
            transcode_required_count=summary.transcode_required_count,
            total_bytes=summary.total_bytes,
        )
        for summary in staged_labels
    )
    return DatasetManifest(
        root=target_root,
        labels=finalized_labels,
        required_label_names=selected_label_names,
        issues=(),
    )


def ensure_minimum_examples(
    manifest: DatasetManifest,
    *,
    min_images_per_label: int,
    required_labels: tuple[str, ...] | None = None,
) -> None:
    """Require a minimum number of examples per label before training."""

    minimum = max(1, int(min_images_per_label))
    # BREAKING: the default scope is now the manifest's validated required labels.
    selected_labels = set(
        _normalize_required_labels(required_labels)
        if required_labels
        else (manifest.required_label_names or manifest.label_names)
    )
    too_small = [
        label.label
        for label in manifest.labels
        if label.label in selected_labels and label.count < minimum
    ]
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
    """Plan collision-resistant JPEG output paths for one capture session."""

    normalized_label = normalize_label_name(label)
    capture_count = max(0, int(count))
    if capture_count <= 0:
        raise ValueError("custom_gesture_capture_count_invalid")

    slug = _normalize_filename_component(timestamp_slug) if timestamp_slug else current_timestamp_slug()
    stem_prefix = normalize_label_name(prefix) if prefix else normalized_label
    label_dir = Path(dataset_root) / normalized_label

    if label_dir.exists():
        if label_dir.is_symlink():
            raise ValueError(f"custom_gesture_capture_label_dir_symlink:{label_dir}")
        if not label_dir.is_dir():
            raise ValueError(f"custom_gesture_capture_label_dir_invalid:{label_dir}")

    existing_names = set()
    if label_dir.exists():
        existing_names = {
            entry.name
            for entry in label_dir.iterdir()
            if entry.is_file()
        }

    targets: list[Path] = []
    index = 1
    while len(targets) < capture_count:
        candidate = label_dir / f"{stem_prefix}-{slug}-{index:04d}.jpg"
        if candidate.name not in existing_names:
            targets.append(candidate)
            existing_names.add(candidate.name)
        index += 1
    return normalized_label, label_dir, tuple(targets)


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
    if not source.is_file():
        raise ValueError(f"custom_gesture_model_not_file:{source}")

    # BREAKING: runtime_model_name must now resolve to a single .task filename.
    runtime_name = _normalize_runtime_model_name(runtime_model_name)
    runtime_dir = Path(runtime_model_dir)
    if runtime_dir.exists() and not runtime_dir.is_dir():
        raise ValueError(f"custom_gesture_runtime_model_dir_invalid:{runtime_dir}")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = runtime_dir / runtime_name
    _atomic_copy_file(source, runtime_path)
    return runtime_path


def runtime_env_hint(model_path: Path) -> str:
    """Return the env assignment Twinr expects for a custom gesture model."""

    return (
        "TWINR_PROACTIVE_LOCAL_CAMERA_MEDIAPIPE_CUSTOM_GESTURE_MODEL_PATH="
        f"{Path(model_path).as_posix()}"
    )


def write_json_summary(path: Path, payload: dict[str, object]) -> None:
    """Write one indented JSON summary to disk atomically."""

    target = Path(path)
    serialized = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    _atomic_write_text(target, serialized)


def _build_manifest_issues(
    counts: dict[str, DatasetLabelSummary],
    required_label_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Build non-fatal manifest issues for downstream summaries."""

    issues: list[str] = []
    selected = set(required_label_names)
    transcode_labels = sorted(
        label_name
        for label_name, summary in counts.items()
        if label_name in selected and summary.transcode_required_count > 0
    )
    if transcode_labels:
        issues.append(
            f"custom_gesture_training_transcode_required:{','.join(transcode_labels)}"
        )
    return tuple(issues)


def _discover_label_directories(root: Path) -> list[tuple[str, Path]]:
    """Return normalized label directories while rejecting unsafe collisions."""

    collisions: dict[str, list[Path]] = {}
    normalized_directories: list[tuple[str, Path]] = []

    for path in sorted(root.iterdir(), key=lambda entry: entry.name):
        if path.is_symlink():
            if path.exists() and path.is_dir():
                raise ValueError(f"custom_gesture_dataset_symlinked_label_dir:{path}")
            continue
        if not path.is_dir():
            continue
        label = normalize_label_name(path.name)
        collisions.setdefault(label, []).append(path)
        normalized_directories.append((label, path))

    duplicate_labels = {
        label: directories
        for label, directories in collisions.items()
        if len(directories) > 1
    }
    if duplicate_labels:
        details = ";".join(
            f"{label}={','.join(str(directory) for directory in directories)}"
            for label, directories in sorted(duplicate_labels.items())
        )
        raise ValueError(f"custom_gesture_dataset_duplicate_normalized_labels:{details}")

    return sorted(normalized_directories, key=lambda item: item[0])


def _collect_image_files(
    directory: Path,
    *,
    validate_images: bool,
    max_image_pixels: int,
) -> list[_CollectedImageFile]:
    """List supported still-image files in deterministic order."""

    image_files: list[_CollectedImageFile] = []
    for path in sorted(directory.iterdir(), key=lambda entry: entry.name):
        if path.is_symlink():
            if path.exists() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
                raise ValueError(f"custom_gesture_dataset_symlinked_image:{path}")
            continue
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        if validate_images:
            _validate_image_file(path, suffix=suffix, max_image_pixels=max_image_pixels)
        image_files.append(
            _CollectedImageFile(
                path=path,
                suffix=suffix,
                size_bytes=path.stat().st_size,
                needs_training_transcode=suffix not in TRAINING_COMPATIBLE_IMAGE_SUFFIXES,
            )
        )
    return image_files


def _validate_image_file(path: Path, *, suffix: str, max_image_pixels: int) -> None:
    """Validate one still image eagerly enough to fail before training."""

    if path.stat().st_size <= 0:
        raise ValueError(f"custom_gesture_image_empty:{path}")

    pillow = _optional_pillow()
    if pillow is None:
        _validate_image_file_basic(path, suffix=suffix)
        return

    image_module, unidentified_image_error = pillow
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", image_module.DecompressionBombWarning)
            with image_module.open(path) as image:
                width, height = image.size
                if width <= 0 or height <= 0:
                    raise ValueError(f"custom_gesture_image_invalid_size:{path}")
                if width * height > max_image_pixels:
                    raise ValueError(f"custom_gesture_image_too_large:{path}")
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError(f"custom_gesture_image_animated:{path}")
                image.verify()
    except ValueError:
        raise
    except getattr(image_module, "DecompressionBombError", Exception):
        raise ValueError(f"custom_gesture_image_decompression_bomb:{path}") from None
    except getattr(image_module, "DecompressionBombWarning", Exception):
        raise ValueError(f"custom_gesture_image_decompression_bomb:{path}") from None
    except unidentified_image_error:
        raise ValueError(f"custom_gesture_image_corrupt:{path}") from None
    except OSError:
        raise ValueError(f"custom_gesture_image_corrupt:{path}") from None


def _validate_image_file_basic(path: Path, *, suffix: str) -> None:
    """Fallback header validation when Pillow is unavailable."""

    with path.open("rb") as handle:
        header = handle.read(16)

    signatures = {
        ".jpg": (b"\xff\xd8\xff",),
        ".jpeg": (b"\xff\xd8\xff",),
        ".png": (b"\x89PNG\r\n\x1a\n",),
        ".bmp": (b"BM",),
        ".webp": (b"RIFF",),
    }
    valid_prefixes = signatures.get(suffix, ())
    if not any(header.startswith(prefix) for prefix in valid_prefixes):
        raise ValueError(f"custom_gesture_image_corrupt:{path}")
    if suffix == ".webp" and header[8:12] != b"WEBP":
        raise ValueError(f"custom_gesture_image_corrupt:{path}")


def _optional_pillow():
    """Import Pillow lazily so capture-only environments stay lightweight."""

    try:
        from PIL import Image, UnidentifiedImageError
    except Exception:
        return None
    return Image, UnidentifiedImageError


def _plan_staged_image_path(target_dir: Path, image_file: _CollectedImageFile) -> Path:
    """Return one collision-safe staged training path."""

    base_name = _normalize_filename_component(image_file.path.stem)
    suffix = ".png" if image_file.needs_training_transcode else image_file.suffix
    candidate = target_dir / f"{base_name}{suffix}"
    if not candidate.exists():
        return candidate

    index = 2
    while True:
        candidate = target_dir / f"{base_name}-{index:02d}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def _transcode_image_for_training(source_path: Path, target_path: Path) -> None:
    """Transcode one source image into a training-compatible PNG target."""

    pillow = _optional_pillow()
    if pillow is None:
        raise RuntimeError(f"custom_gesture_pillow_required_for_transcode:{source_path}")

    image_module, _ = pillow
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", image_module.DecompressionBombWarning)
            with image_module.open(source_path) as image:
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError(f"custom_gesture_image_animated:{source_path}")
                converted = image.convert("RGB")
                converted.save(target_path, format="PNG")
                converted.close()
    except ValueError:
        raise
    except getattr(image_module, "DecompressionBombError", Exception):
        raise ValueError(f"custom_gesture_image_decompression_bomb:{source_path}") from None
    except getattr(image_module, "DecompressionBombWarning", Exception):
        raise ValueError(f"custom_gesture_image_decompression_bomb:{source_path}") from None
    except OSError:
        raise ValueError(f"custom_gesture_image_corrupt:{source_path}") from None


def _materialize_file(source_path: Path, target_path: Path, *, link_mode: str) -> None:
    """Materialize one staged file with hard-link-first semantics when possible."""

    mode = str(link_mode or DEFAULT_STAGE_LINK_MODE).strip().lower()
    if mode not in {"auto", "copy", "hardlink"}:
        raise ValueError(f"custom_gesture_stage_link_mode_invalid:{link_mode}")

    if mode in {"auto", "hardlink"}:
        try:
            os.link(source_path, target_path)
            return
        except OSError:
            if mode == "hardlink":
                raise
    shutil.copy2(source_path, target_path)


def _normalize_required_labels(required_labels: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Normalize and deduplicate the required-label list."""

    normalized: list[str] = []
    for label in required_labels:
        normalized_label = normalize_label_name(label)
        if normalized_label not in normalized:
            normalized.append(normalized_label)
    if "none" not in normalized:
        normalized.insert(0, "none")
    return tuple(normalized)


def _normalize_filename_component(value: str | None) -> str:
    """Normalize one filename-safe component without introducing separators."""

    text = unicodedata.normalize("NFKD", str(value or "")).strip()
    ascii_text = text.encode("ascii", "ignore").decode("ascii")
    compact = re.sub(r"[^A-Za-z0-9._-]+", "_", ascii_text)
    compact = re.sub(r"_+", "_", compact).strip("._-")
    if not compact:
        raise ValueError("custom_gesture_filename_component_invalid")
    if compact in {".", ".."}:
        raise ValueError("custom_gesture_filename_component_invalid")
    if "/" in compact or "\\" in compact:
        raise ValueError("custom_gesture_filename_component_invalid")
    return compact


def _normalize_runtime_model_name(runtime_model_name: str) -> str:
    """Validate the runtime model filename and keep it within the target dir."""

    raw_name = str(runtime_model_name or "").strip()
    if not raw_name:
        raise ValueError("custom_gesture_runtime_model_name_invalid")

    path_name = PurePath(raw_name)
    if raw_name != path_name.name or path_name.is_absolute():
        raise ValueError(f"custom_gesture_runtime_model_name_invalid:{runtime_model_name}")

    normalized = _normalize_filename_component(path_name.name)
    if "." not in normalized:
        normalized = f"{normalized}.task"
    suffix = Path(normalized).suffix.lower()
    if suffix != ".task":
        raise ValueError(f"custom_gesture_runtime_model_name_invalid_suffix:{runtime_model_name}")
    return normalized


def _atomic_copy_file(source_path: Path, target_path: Path) -> None:
    """Copy one file atomically by writing a sibling temp file first."""

    parent = target_path.parent
    parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists() and target_path.is_dir():
        raise ValueError(f"custom_gesture_atomic_target_is_directory:{target_path}")

    fd, temp_name = tempfile.mkstemp(prefix=f".{target_path.name}.tmp-", dir=str(parent))
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        shutil.copy2(source_path, temp_path)
        with temp_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temp_path, target_path)
        _fsync_directory(parent)
    finally:
        with suppress(FileNotFoundError):
            temp_path.unlink()


def _atomic_write_text(target_path: Path, text: str) -> None:
    """Write UTF-8 text atomically to disk."""

    parent = target_path.parent
    parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists() and target_path.is_dir():
        raise ValueError(f"custom_gesture_atomic_target_is_directory:{target_path}")

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target_path.name}.tmp-",
        dir=str(parent),
        text=True,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target_path)
        _fsync_directory(parent)
    finally:
        with suppress(FileNotFoundError):
            temp_path.unlink()


def _fsync_directory(directory: Path) -> None:
    """Best-effort fsync for one directory after atomic replacement."""

    if not hasattr(os, "O_DIRECTORY"):
        return
    dir_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
