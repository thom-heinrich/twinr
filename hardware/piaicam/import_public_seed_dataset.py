#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""Import a bounded public gesture seed dataset into Twinr's training layout.

This helper downloads one known public archive, extracts a deterministic sample
for Twinr's three product-critical gestures plus `none`, and writes those
images into the same folder-per-label dataset layout that the Pi capture and
training helpers already use.

Purpose
-------
Bootstrap Twinr's custom MediaPipe gesture recognizer with public examples for
`thumbs_up`, `thumbs_down`, `peace_sign`, and `none` before Pi-specific
captures are added.

Usage
-----
Command-line invocation examples::

    python3 hardware/piaicam/import_public_seed_dataset.py --dry-run
    python3 hardware/piaicam/import_public_seed_dataset.py --count-per-label 128
    python3 hardware/piaicam/import_public_seed_dataset.py --archive-path /tmp/hagrid-4ges.zip

Inputs
------
- One supported public archive URL or pre-downloaded ZIP file.
- One target dataset root laid out as ``<dataset_root>/<label>/*.jpg``.
- One deterministic sample count per label.

Outputs
-------
- Downloads the public ZIP archive into a local cache path when needed.
- Extracts a bounded sample into the target dataset root using Twinr label names.
- Prints one JSON summary with source, available counts, and imported files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import random
import shutil
import sys
import urllib.request
import zipfile

from custom_gesture_workflow import DEFAULT_DATASET_ROOT, current_timestamp_slug, normalize_label_name


DEFAULT_PUBLIC_SEED_ARCHIVE_URL = (
    "https://huggingface.co/datasets/"
    "tanli12/hagrid-classification-512p-no-gesture-4-ges-zip/resolve/main/"
    "hagrid-classification-512p-no-gesture-150k.zip?download=true"
)
DEFAULT_PUBLIC_SEED_ARCHIVE_PATH = Path("state/mediapipe/public_seed_cache/hagrid-classification-4ges.zip")
DEFAULT_IMPORT_PREFIX = "public_seed_hagrid"
DEFAULT_SOURCE_LABEL_MAP = {
    "like": "thumbs_up",
    "dislike": "thumbs_down",
    "peace": "peace_sign",
    "no_gesture": "none",
}
SUPPORTED_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the bounded public-seed importer."""

    parser = argparse.ArgumentParser(
        description="Import a bounded public gesture seed dataset into Twinr's folder-per-label training layout.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Target Twinr dataset root that will receive thumbs_up/thumbs_down/peace_sign/none subdirectories.",
    )
    parser.add_argument(
        "--archive-path",
        type=Path,
        default=DEFAULT_PUBLIC_SEED_ARCHIVE_PATH,
        help="Local cache path for the downloaded public seed archive.",
    )
    parser.add_argument(
        "--download-url",
        default=DEFAULT_PUBLIC_SEED_ARCHIVE_URL,
        help="Public ZIP archive URL used when --archive-path does not already exist.",
    )
    parser.add_argument(
        "--count-per-label",
        type=int,
        default=128,
        help="Number of images to import for each mapped Twinr label.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed used when sampling from the public archive.",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_IMPORT_PREFIX,
        help="Filename prefix used for imported seed files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect the public archive and print the planned files without extracting images.",
    )
    return parser


def import_public_seed_dataset(
    *,
    dataset_root: Path,
    archive_path: Path,
    download_url: str,
    count_per_label: int,
    seed: int,
    prefix: str,
    dry_run: bool = False,
) -> dict[str, object]:
    """Import one deterministic public seed sample into Twinr's dataset root."""

    normalized_prefix = normalize_label_name(prefix)
    normalized_count = max(1, int(count_per_label))
    cache_path = ensure_public_seed_archive(archive_path=archive_path, download_url=download_url)
    available_members = index_public_seed_archive(cache_path)
    selected_members = select_public_seed_members(
        members_by_label=available_members,
        count_per_label=normalized_count,
        seed=max(0, int(seed)),
    )
    import_plan = plan_import_targets(
        dataset_root=dataset_root,
        selected_members=selected_members,
        prefix=normalized_prefix,
    )
    summary: dict[str, object] = {
        "status": "dry_run" if dry_run else "imported",
        "dataset_root": str(Path(dataset_root)),
        "archive_path": str(cache_path),
        "download_url": download_url,
        "count_per_label": normalized_count,
        "seed": max(0, int(seed)),
        "source_label_map": dict(DEFAULT_SOURCE_LABEL_MAP),
        "available_counts": {label: len(paths) for label, paths in sorted(available_members.items())},
        "imported_counts": {label: len(paths) for label, paths in sorted(selected_members.items())},
        "files": [str(path) for targets in import_plan.values() for path in targets],
    }
    if dry_run:
        return summary
    extract_public_seed_members(archive_path=cache_path, selected_members=selected_members, import_plan=import_plan)
    return summary


def ensure_public_seed_archive(*, archive_path: Path, download_url: str) -> Path:
    """Return one local public-seed archive path, downloading it if absent."""

    target = Path(archive_path)
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(download_url, timeout=60) as response, target.open("wb") as output:
        shutil.copyfileobj(response, output)
    return target


def index_public_seed_archive(archive_path: Path) -> dict[str, tuple[str, ...]]:
    """Index supported public-seed archive members by Twinr target label."""

    indexed: dict[str, list[str]] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for member_name in archive.namelist():
            parts = [part for part in member_name.split("/") if part]
            if len(parts) < 3:
                continue
            source_label = normalize_label_name(parts[1])
            target_label = DEFAULT_SOURCE_LABEL_MAP.get(source_label)
            if target_label is None:
                continue
            if Path(member_name).suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            indexed.setdefault(target_label, []).append(member_name)
    missing = [label for label in DEFAULT_SOURCE_LABEL_MAP.values() if label not in indexed]
    if missing:
        raise ValueError(f"public_seed_archive_missing_labels:{','.join(sorted(missing))}")
    return {label: tuple(sorted(paths)) for label, paths in sorted(indexed.items())}


def select_public_seed_members(
    *,
    members_by_label: dict[str, tuple[str, ...]],
    count_per_label: int,
    seed: int,
) -> dict[str, tuple[str, ...]]:
    """Pick one deterministic bounded subset of members per Twinr label."""

    rng = random.Random(seed)
    selected: dict[str, tuple[str, ...]] = {}
    for label, members in sorted(members_by_label.items()):
        if len(members) < count_per_label:
            raise ValueError(
                f"public_seed_archive_too_small:{label}:{count_per_label}:{len(members)}"
            )
        shuffled = list(members)
        rng.shuffle(shuffled)
        selected[label] = tuple(sorted(shuffled[:count_per_label]))
    return selected


def plan_import_targets(
    *,
    dataset_root: Path,
    selected_members: dict[str, tuple[str, ...]],
    prefix: str,
) -> dict[str, tuple[Path, ...]]:
    """Plan deterministic output paths for the selected public seed members."""

    slug = current_timestamp_slug()
    plans: dict[str, tuple[Path, ...]] = {}
    for label, members in sorted(selected_members.items()):
        label_dir = Path(dataset_root) / label
        targets: list[Path] = []
        for index, member_name in enumerate(members, start=1):
            suffix = Path(member_name).suffix.lower() or ".jpg"
            targets.append(label_dir / f"{prefix}-{label}-{slug}-{index:04d}{suffix}")
        plans[label] = tuple(targets)
    return plans


def extract_public_seed_members(
    *,
    archive_path: Path,
    selected_members: dict[str, tuple[str, ...]],
    import_plan: dict[str, tuple[Path, ...]],
) -> None:
    """Extract the selected public-seed members into the planned target paths."""

    with zipfile.ZipFile(archive_path) as archive:
        for label, members in selected_members.items():
            targets = import_plan[label]
            for target in targets:
                target.parent.mkdir(parents=True, exist_ok=True)
            for member_name, target_path in zip(members, targets, strict=True):
                with archive.open(member_name) as source, Path(target_path).open("wb") as output:
                    shutil.copyfileobj(source, output)


def main(argv: list[str] | None = None) -> int:
    """Run the bounded public-seed import helper."""

    parser = build_parser()
    args = parser.parse_args(argv)
    summary = import_public_seed_dataset(
        dataset_root=args.dataset_root,
        archive_path=args.archive_path,
        download_url=str(args.download_url),
        count_per_label=args.count_per_label,
        seed=args.seed,
        prefix=str(args.prefix),
        dry_run=bool(args.dry_run),
    )
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
