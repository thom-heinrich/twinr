#!/usr/bin/env python3
# CHANGELOG: 2026-03-27
# BUG-1: Stop silently coercing invalid --count-per-label values; reject non-positive counts instead of importing the wrong dataset size.
# BUG-2: Replace direct-to-cache downloads with validated atomic downloads so interrupted network transfers cannot leave a corrupt ZIP that future runs trust.
# BUG-3: Stage imports and commit atomically so a bad image/member cannot leave the dataset root partially mutated.
# SEC-1: Resolve Hugging Face sources to a concrete commit before download, and support SHA-256 pinning for arbitrary URLs to reduce mutable-upstream / supply-chain risk.
# SEC-2: Add ZIP/image size guards, compression-ratio guards, CRC/header checks, and Pillow decompression-bomb protection to reduce disk/memory exhaustion risk on Raspberry Pi deployments.
# IMP-1: Normalize every imported image to EXIF-corrected RGB JPEG for consistent MediaPipe-compatible training inputs.
# IMP-2: Make repeated imports idempotent with content-addressed filenames, duplicate skipping, and richer provenance in the JSON summary.
# IMP-3: Generalize archive indexing to find mapped labels by directory name anywhere in the member path, not only at one hard-coded depth.
# BREAKING: This script now declares Pillow and huggingface_hub as runtime dependencies via PEP 723 metadata.
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface_hub>=0.30.0",
#   "Pillow>=12.0.0",
# ]
# ///
"""Import a bounded public gesture seed dataset into Twinr's training layout.

This helper downloads one known public archive, extracts a deterministic sample
for Twinr's three product-critical gestures plus ``none``, normalizes them into
EXIF-corrected RGB JPEGs, and writes them into Twinr's folder-per-label dataset
layout.

Purpose
-------
Bootstrap Twinr's custom MediaPipe gesture recognizer with public examples for
``thumbs_up``, ``thumbs_down``, ``peace_sign``, and ``none`` before Pi-specific
captures are added.

Usage
-----
Command-line invocation examples::

    python3 hardware/piaicam/import_public_seed_dataset.py --dry-run
    python3 hardware/piaicam/import_public_seed_dataset.py --count-per-label 128
    python3 hardware/piaicam/import_public_seed_dataset.py --archive-path /tmp/hagrid-4ges.zip

Inputs
------
- One supported public archive URL, Hugging Face dataset file, or pre-downloaded ZIP file.
- One target dataset root laid out as ``<dataset_root>/<label>/*.jpg``.
- One deterministic sample count per label.

Outputs
-------
- Downloads the public ZIP archive into a local cache path when needed.
- Extracts and normalizes a bounded sample into the target dataset root using Twinr label names.
- Prints one JSON summary with source provenance, counts, skipped duplicates, and imported files.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import shutil
import sys
import tempfile
import urllib.parse
import urllib.request
import warnings
import zipfile
from dataclasses import dataclass
from typing import Any, Iterable, cast

from custom_gesture_workflow import DEFAULT_DATASET_ROOT, normalize_label_name


DEFAULT_PUBLIC_SEED_ARCHIVE_URL = (
    "https://huggingface.co/datasets/"
    "tanli12/hagrid-classification-512p-no-gesture-4-ges-zip/resolve/main/"
    "hagrid-classification-512p-no-gesture-150k.zip?download=true"
)
DEFAULT_PUBLIC_SEED_HF_REPO_ID = "tanli12/hagrid-classification-512p-no-gesture-4-ges-zip"
DEFAULT_PUBLIC_SEED_HF_FILENAME = "hagrid-classification-512p-no-gesture-150k.zip"
DEFAULT_PUBLIC_SEED_HF_REVISION = "main"
DEFAULT_PUBLIC_SEED_ARCHIVE_PATH = Path("state/mediapipe/public_seed_cache/hagrid-classification-4ges.zip")
DEFAULT_IMPORT_PREFIX = "public_seed_hagrid"
DEFAULT_SOURCE_LABEL_MAP = {
    "like": "thumbs_up",
    "dislike": "thumbs_down",
    "peace": "peace_sign",
    "no_gesture": "none",
}
SUPPORTED_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")
DEFAULT_JPEG_QUALITY = 95
DEFAULT_MAX_ARCHIVE_BYTES = 20 * 1024 * 1024 * 1024
DEFAULT_MAX_MEMBER_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_IMAGE_PIXELS = 25_000_000
DEFAULT_MAX_COMPRESSION_RATIO = 200.0
HF_DATASET_RESOLVE_RE = re.compile(
    r"^https://huggingface\.co/datasets/(?P<repo_id>[^/]+/[^/]+)/resolve/(?P<revision>[^/]+)/(?P<filename>[^?]+)"
)


@dataclass(frozen=True)
class ArchiveMember:
    """A validated archive member used by the import pipeline."""

    name: str
    file_size: int
    compress_size: int
    crc: int


@dataclass(frozen=True)
class SourceResolution:
    """Resolved archive source provenance."""

    archive_path: Path
    source_kind: str
    download_url: str
    hf_repo_id: str | None = None
    hf_filename: str | None = None
    requested_revision: str | None = None
    resolved_revision: str | None = None
    archive_bytes: int | None = None
    sha256: str | None = None


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
        help="Legacy public ZIP archive URL used when --archive-path does not already exist.",
    )
    parser.add_argument(
        "--hf-repo-id",
        default=DEFAULT_PUBLIC_SEED_HF_REPO_ID,
        help="Hugging Face dataset repo used for secure downloads of the public ZIP archive.",
    )
    parser.add_argument(
        "--hf-filename",
        default=DEFAULT_PUBLIC_SEED_HF_FILENAME,
        help="Filename inside the Hugging Face dataset repo for the public ZIP archive.",
    )
    parser.add_argument(
        "--hf-revision",
        default=DEFAULT_PUBLIC_SEED_HF_REVISION,
        help="Requested Hugging Face revision. Branch names are resolved to a concrete commit before download.",
    )
    parser.add_argument(
        "--expected-sha256",
        default=None,
        help="Optional SHA-256 checksum used to pin local or URL-fetched archives.",
    )
    # BREAKING: Direct non-Hugging Face URL downloads now require --expected-sha256 unless --allow-unsafe-url is explicitly set.
    parser.add_argument(
        "--allow-unsafe-url",
        action="store_true",
        help="Allow direct non-Hugging Face URL downloads without --expected-sha256.",
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
        help="Human-readable filename prefix used for imported seed files.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=DEFAULT_JPEG_QUALITY,
        help="JPEG quality used for normalized output images.",
    )
    parser.add_argument(
        "--max-archive-bytes",
        type=int,
        default=DEFAULT_MAX_ARCHIVE_BYTES,
        help="Hard limit for the archive file size in bytes.",
    )
    parser.add_argument(
        "--max-member-bytes",
        type=int,
        default=DEFAULT_MAX_MEMBER_BYTES,
        help="Hard limit for one selected archive member's uncompressed size in bytes.",
    )
    parser.add_argument(
        "--max-image-pixels",
        type=int,
        default=DEFAULT_MAX_IMAGE_PIXELS,
        help="Hard limit for decoded image pixels used during validation.",
    )
    parser.add_argument(
        "--max-compression-ratio",
        type=float,
        default=DEFAULT_MAX_COMPRESSION_RATIO,
        help="Reject selected members whose uncompressed/compressed size ratio exceeds this value.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect the public archive, validate the selected images, and print the planned files without writing them.",
    )
    return parser


def import_public_seed_dataset(
    *,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    archive_path: Path = DEFAULT_PUBLIC_SEED_ARCHIVE_PATH,
    download_url: str = DEFAULT_PUBLIC_SEED_ARCHIVE_URL,
    hf_repo_id: str = DEFAULT_PUBLIC_SEED_HF_REPO_ID,
    hf_filename: str = DEFAULT_PUBLIC_SEED_HF_FILENAME,
    hf_revision: str = DEFAULT_PUBLIC_SEED_HF_REVISION,
    expected_sha256: str | None = None,
    allow_unsafe_url: bool = False,
    count_per_label: int = 128,
    seed: int = 42,
    prefix: str = DEFAULT_IMPORT_PREFIX,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    max_archive_bytes: int = DEFAULT_MAX_ARCHIVE_BYTES,
    max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES,
    max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
    max_compression_ratio: float = DEFAULT_MAX_COMPRESSION_RATIO,
    dry_run: bool = False,
) -> dict[str, object]:
    """Import one deterministic public seed sample into Twinr's dataset root."""

    normalized_prefix = require_nonempty_slug(prefix, field_name="prefix")
    normalized_count = require_positive_int(count_per_label, field_name="count_per_label")
    normalized_seed = max(0, int(seed))
    normalized_quality = min(100, max(70, int(jpeg_quality)))
    normalized_expected_sha256 = normalize_optional_sha256(expected_sha256)
    normalized_archive_path = Path(archive_path)
    normalized_dataset_root = Path(dataset_root)

    source = ensure_public_seed_archive(
        archive_path=normalized_archive_path,
        download_url=str(download_url),
        hf_repo_id=str(hf_repo_id),
        hf_filename=str(hf_filename),
        hf_revision=str(hf_revision),
        expected_sha256=normalized_expected_sha256,
        allow_unsafe_url=bool(allow_unsafe_url),
        max_archive_bytes=require_positive_int(max_archive_bytes, field_name="max_archive_bytes"),
    )
    available_members = index_public_seed_archive(source.archive_path)
    selected_members = select_public_seed_members(
        members_by_label=available_members,
        count_per_label=normalized_count,
        seed=normalized_seed,
    )
    existing_hashes = index_existing_dataset_hashes(normalized_dataset_root, labels=selected_members.keys())
    materialized = materialize_selected_members(
        archive_path=source.archive_path,
        dataset_root=normalized_dataset_root,
        selected_members=selected_members,
        existing_hashes=existing_hashes,
        prefix=normalized_prefix,
        jpeg_quality=normalized_quality,
        max_member_bytes=require_positive_int(max_member_bytes, field_name="max_member_bytes"),
        max_image_pixels=require_positive_int(max_image_pixels, field_name="max_image_pixels"),
        max_compression_ratio=max(1.0, float(max_compression_ratio)),
        dry_run=bool(dry_run),
    )
    imported_counts = cast(dict[str, int], materialized["imported_counts"])
    skipped_duplicate_counts = cast(dict[str, int], materialized["skipped_duplicate_counts"])
    files = cast(list[str], materialized["files"])
    skipped_files = cast(list[str], materialized["skipped_files"])
    summary: dict[str, object] = {
        "status": "dry_run" if dry_run else "imported",
        "dataset_root": str(normalized_dataset_root),
        "archive_path": str(source.archive_path),
        "download_url": source.download_url,
        "source_kind": source.source_kind,
        "hf_repo_id": source.hf_repo_id,
        "hf_filename": source.hf_filename,
        "requested_revision": source.requested_revision,
        "resolved_revision": source.resolved_revision,
        "archive_bytes": source.archive_bytes,
        "archive_sha256": source.sha256,
        "count_per_label": normalized_count,
        "seed": normalized_seed,
        "source_label_map": dict(DEFAULT_SOURCE_LABEL_MAP),
        "available_counts": {label: len(paths) for label, paths in sorted(available_members.items())},
        "selected_counts": {label: len(paths) for label, paths in sorted(selected_members.items())},
        "imported_counts": dict(imported_counts),
        "skipped_duplicate_counts": dict(skipped_duplicate_counts),
        "files": list(files),
        "skipped_files": list(skipped_files),
    }
    return summary


def require_positive_int(value: int, *, field_name: str) -> int:
    """Return a validated positive integer."""

    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"invalid_{field_name}:{normalized}")
    return normalized


def require_nonempty_slug(value: str, *, field_name: str) -> str:
    """Normalize a CLI label-like value and ensure it remains non-empty."""

    normalized = normalize_label_name(str(value))
    if not normalized:
        raise ValueError(f"invalid_{field_name}:empty")
    return normalized


def normalize_optional_sha256(value: str | None) -> str | None:
    """Return one normalized lowercase SHA-256 hex digest or None."""

    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError("invalid_expected_sha256")
    return normalized


def ensure_public_seed_archive(
    *,
    archive_path: Path,
    download_url: str,
    hf_repo_id: str,
    hf_filename: str,
    hf_revision: str,
    expected_sha256: str | None,
    allow_unsafe_url: bool,
    max_archive_bytes: int,
) -> SourceResolution:
    """Return one validated local public-seed archive path, downloading it if absent."""

    target = Path(archive_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    def _can_use_default_hf_source() -> bool:
        return str(download_url).strip() == DEFAULT_PUBLIC_SEED_ARCHIVE_URL

    def _resolve_remote_hf_source() -> tuple[str, str, str] | None:
        parsed_hf = parse_huggingface_dataset_resolve_url(download_url)
        if parsed_hf is not None:
            return parsed_hf["repo_id"], parsed_hf["filename"], parsed_hf["revision"]
        if _can_use_default_hf_source():
            return hf_repo_id, hf_filename, hf_revision
        return None

    if target.exists():
        try:
            archive_bytes = target.stat().st_size
            if archive_bytes > max_archive_bytes:
                raise ValueError(f"public_seed_archive_too_large:{archive_bytes}:{max_archive_bytes}")
            validate_zip_file(target)
            actual_sha256 = sha256_file(target) if expected_sha256 is not None else None
            if expected_sha256 is not None and actual_sha256 != expected_sha256:
                raise ValueError("public_seed_archive_sha256_mismatch")
            return SourceResolution(
                archive_path=target,
                source_kind="local",
                download_url=download_url,
                hf_repo_id=hf_repo_id,
                hf_filename=hf_filename,
                requested_revision=hf_revision,
                archive_bytes=archive_bytes,
                sha256=actual_sha256,
            )
        except Exception:
            with contextlib.suppress(FileNotFoundError):
                target.unlink()

    remote_hf_source = _resolve_remote_hf_source()
    if remote_hf_source is not None:
        effective_repo_id, effective_filename, effective_revision = remote_hf_source
        return download_archive_from_huggingface(
            archive_path=target,
            repo_id=effective_repo_id,
            filename=effective_filename,
            revision=effective_revision or DEFAULT_PUBLIC_SEED_HF_REVISION,
            download_url=download_url,
            expected_sha256=expected_sha256,
            max_archive_bytes=max_archive_bytes,
        )

    if expected_sha256 is None and not allow_unsafe_url:
        raise ValueError("unsafe_download_url_requires_expected_sha256")
    return download_archive_from_url(
        archive_path=target,
        download_url=download_url,
        expected_sha256=expected_sha256,
        max_archive_bytes=max_archive_bytes,
    )


def parse_huggingface_dataset_resolve_url(download_url: str) -> dict[str, str] | None:
    """Parse one Hugging Face dataset resolve URL into repo/file metadata."""

    match = HF_DATASET_RESOLVE_RE.match(download_url)
    if match is None:
        return None
    filename = urllib.parse.unquote(match.group("filename"))
    return {
        "repo_id": match.group("repo_id"),
        "revision": match.group("revision"),
        "filename": filename,
    }


def download_archive_from_huggingface(
    *,
    archive_path: Path,
    repo_id: str,
    filename: str,
    revision: str,
    download_url: str,
    expected_sha256: str | None,
    max_archive_bytes: int,
) -> SourceResolution:
    """Download and pin one Hugging Face dataset file to a concrete commit."""

    hf_hub_download = _load_hf_hub_download()
    dry_run_info = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        revision=revision,
        dry_run=True,
    )
    archive_bytes = int(dry_run_info.file_size)
    resolved_revision = str(dry_run_info.commit_hash)
    if archive_bytes > max_archive_bytes:
        raise ValueError(f"public_seed_archive_too_large:{archive_bytes}:{max_archive_bytes}")
    cached_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            revision=resolved_revision,
        )
    )
    validate_zip_file(cached_path)
    actual_sha256 = sha256_file(cached_path) if expected_sha256 is not None else None
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError("public_seed_archive_sha256_mismatch")
    copy_file_atomically(cached_path, archive_path)
    validate_zip_file(archive_path)
    return SourceResolution(
        archive_path=archive_path,
        source_kind="huggingface",
        download_url=download_url,
        hf_repo_id=repo_id,
        hf_filename=filename,
        requested_revision=revision,
        resolved_revision=resolved_revision,
        archive_bytes=archive_bytes,
        sha256=actual_sha256,
    )


def download_archive_from_url(
    *,
    archive_path: Path,
    download_url: str,
    expected_sha256: str | None,
    max_archive_bytes: int,
) -> SourceResolution:
    """Download one archive from a direct URL using an atomic temporary file."""

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = archive_path.with_name(f".{archive_path.name}.partial")
    hash_state = hashlib.sha256()
    bytes_written = 0
    try:
        with contextlib.ExitStack() as stack:
            response = stack.enter_context(urllib.request.urlopen(download_url, timeout=60))
            output = stack.enter_context(temp_path.open("wb"))
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > max_archive_bytes:
                    raise ValueError(f"public_seed_archive_too_large:{bytes_written}:{max_archive_bytes}")
                output.write(chunk)
                hash_state.update(chunk)
            output.flush()
            os.fsync(output.fileno())
        actual_sha256 = hash_state.hexdigest()
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise ValueError("public_seed_archive_sha256_mismatch")
        temp_path.replace(archive_path)
        validate_zip_file(archive_path)
        return SourceResolution(
            archive_path=archive_path,
            source_kind="url",
            download_url=download_url,
            archive_bytes=bytes_written,
            sha256=actual_sha256,
        )
    finally:
        with contextlib.suppress(FileNotFoundError):
            temp_path.unlink()


def validate_zip_file(archive_path: Path) -> None:
    """Reject archives that are not readable ZIP files with a valid central directory."""

    try:
        with zipfile.ZipFile(archive_path) as archive:
            if not archive.infolist():
                raise ValueError(f"public_seed_archive_empty:{archive_path}")
    except zipfile.BadZipFile as exc:
        raise ValueError(f"public_seed_archive_bad_zip:{archive_path}") from exc


def copy_file_atomically(source_path: Path, target_path: Path) -> None:
    """Copy one file to target_path atomically on the destination filesystem."""

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        delete=False,
        dir=target_path.parent,
        prefix=f".{target_path.name}.",
        suffix=".tmp",
    ) as temp_file:
        with source_path.open("rb") as source_file:
            shutil.copyfileobj(source_file, temp_file, length=1024 * 1024)
        temp_file.flush()
        os.fsync(temp_file.fileno())
        temp_path = Path(temp_file.name)
    temp_path.replace(target_path)


def index_public_seed_archive(archive_path: Path) -> dict[str, tuple[ArchiveMember, ...]]:
    """Index supported public-seed archive members by Twinr target label."""

    indexed: dict[str, list[ArchiveMember]] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            source_label = infer_source_label_from_member_path(info.filename)
            if source_label is None:
                continue
            target_label = DEFAULT_SOURCE_LABEL_MAP.get(source_label)
            if target_label is None:
                continue
            if Path(info.filename).suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            indexed.setdefault(target_label, []).append(
                ArchiveMember(
                    name=info.filename,
                    file_size=int(info.file_size),
                    compress_size=int(info.compress_size),
                    crc=int(info.CRC),
                )
            )
    missing = [label for label in DEFAULT_SOURCE_LABEL_MAP.values() if label not in indexed]
    if missing:
        raise ValueError(f"public_seed_archive_missing_labels:{','.join(sorted(missing))}")
    return {
        label: tuple(sorted(members, key=lambda member: member.name))
        for label, members in sorted(indexed.items())
    }


def infer_source_label_from_member_path(member_name: str) -> str | None:
    """Infer the source gesture label from any directory component in one archive member path."""

    parts = [normalize_label_name(part) for part in Path(member_name).parts[:-1] if part not in {"/", ""}]
    for part in reversed(parts):
        if part in DEFAULT_SOURCE_LABEL_MAP:
            return part
    return None


def select_public_seed_members(
    *,
    members_by_label: dict[str, tuple[ArchiveMember, ...]],
    count_per_label: int,
    seed: int,
) -> dict[str, tuple[ArchiveMember, ...]]:
    """Pick one deterministic bounded subset of members per Twinr label."""

    rng = random.Random(seed)
    selected: dict[str, tuple[ArchiveMember, ...]] = {}
    for label, members in sorted(members_by_label.items()):
        if len(members) < count_per_label:
            raise ValueError(
                f"public_seed_archive_too_small:{label}:{count_per_label}:{len(members)}"
            )
        shuffled = list(members)
        rng.shuffle(shuffled)
        selected[label] = tuple(sorted(shuffled[:count_per_label], key=lambda member: member.name))
    return selected


def index_existing_dataset_hashes(
    dataset_root: Path,
    *,
    labels: Iterable[str],
) -> dict[str, set[str]]:
    """Index exact existing file hashes per label for duplicate skipping."""

    indexed: dict[str, set[str]] = {str(label): set() for label in labels}
    for label in indexed:
        label_dir = Path(dataset_root) / label
        if not label_dir.exists():
            continue
        for path in sorted(label_dir.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED_SUFFIXES and path.suffix.lower() != ".jpg":
                continue
            indexed[label].add(sha256_file(path))
    return indexed


def materialize_selected_members(
    *,
    archive_path: Path,
    dataset_root: Path,
    selected_members: dict[str, tuple[ArchiveMember, ...]],
    existing_hashes: dict[str, set[str]],
    prefix: str,
    jpeg_quality: int,
    max_member_bytes: int,
    max_image_pixels: int,
    max_compression_ratio: float,
    dry_run: bool,
) -> dict[str, object]:
    """Validate, normalize, and optionally commit the selected archive members."""

    stage_parent = dataset_root.parent if dataset_root.parent != Path("") else Path(".")
    stage_parent.mkdir(parents=True, exist_ok=True)
    imported_counts = {label: 0 for label in selected_members}
    skipped_duplicate_counts = {label: 0 for label in selected_members}
    files: list[str] = []
    skipped_files: list[str] = []
    pending_commits: list[tuple[Path, Path]] = []

    with tempfile.TemporaryDirectory(prefix=".public_seed_stage-", dir=str(stage_parent)) as stage_dir_name:
        stage_dir = Path(stage_dir_name)
        with zipfile.ZipFile(archive_path) as archive:
            info_by_name = {info.filename: info for info in archive.infolist()}
            for label, members in sorted(selected_members.items()):
                for ordinal, member in enumerate(members, start=1):
                    info = info_by_name.get(member.name)
                    if info is None:
                        raise ValueError(f"public_seed_archive_member_missing:{member.name}")
                    validate_member_safety(
                        member,
                        max_member_bytes=max_member_bytes,
                        max_compression_ratio=max_compression_ratio,
                    )
                    raw_bytes = read_zip_member(archive, member.name, max_bytes=max_member_bytes)
                    normalized_bytes = normalize_image_bytes(
                        raw_bytes,
                        max_image_pixels=max_image_pixels,
                        jpeg_quality=jpeg_quality,
                    )
                    digest = hashlib.sha256(normalized_bytes).hexdigest()
                    target_name = build_target_filename(
                        prefix=prefix,
                        label=label,
                        ordinal=ordinal,
                        digest=digest,
                    )
                    target_path = Path(dataset_root) / label / target_name
                    if digest in existing_hashes.setdefault(label, set()):
                        skipped_duplicate_counts[label] += 1
                        skipped_files.append(str(target_path))
                        continue
                    files.append(str(target_path))
                    imported_counts[label] += 1
                    existing_hashes[label].add(digest)
                    if dry_run:
                        continue
                    stage_path = stage_dir / label / target_name
                    write_bytes_atomically(stage_path, normalized_bytes)
                    pending_commits.append((stage_path, target_path))
        if dry_run:
            return {
                "imported_counts": imported_counts,
                "skipped_duplicate_counts": skipped_duplicate_counts,
                "files": files,
                "skipped_files": skipped_files,
            }
        for stage_path, target_path in pending_commits:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if target_path.exists():
                skipped_files.append(str(target_path))
                continue
            stage_path.replace(target_path)
    return {
        "imported_counts": imported_counts,
        "skipped_duplicate_counts": skipped_duplicate_counts,
        "files": files,
        "skipped_files": skipped_files,
    }


def validate_member_safety(
    member: ArchiveMember,
    *,
    max_member_bytes: int,
    max_compression_ratio: float,
) -> None:
    """Reject suspicious archive members before reading them."""

    if member.file_size <= 0:
        raise ValueError(f"public_seed_member_empty:{member.name}")
    if member.file_size > max_member_bytes:
        raise ValueError(f"public_seed_member_too_large:{member.name}:{member.file_size}:{max_member_bytes}")
    if member.compress_size < 0:
        raise ValueError(f"public_seed_member_invalid_compress_size:{member.name}")
    if member.compress_size == 0 and member.file_size > 0:
        raise ValueError(f"public_seed_member_zero_compressed_size:{member.name}")
    ratio = member.file_size / max(1, member.compress_size)
    if ratio > max_compression_ratio:
        raise ValueError(
            f"public_seed_member_suspicious_compression_ratio:{member.name}:{ratio:.2f}:{max_compression_ratio:.2f}"
        )


def read_zip_member(archive: zipfile.ZipFile, member_name: str, *, max_bytes: int) -> bytes:
    """Read one ZIP member with a hard byte limit and CRC enforcement."""

    chunks: list[bytes] = []
    total = 0
    try:
        with archive.open(member_name) as source:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"public_seed_member_too_large:{member_name}:{total}:{max_bytes}")
                chunks.append(chunk)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"public_seed_archive_crc_error:{member_name}") from exc
    return b"".join(chunks)


def normalize_image_bytes(raw_bytes: bytes, *, max_image_pixels: int, jpeg_quality: int) -> bytes:
    """Validate and normalize one image into EXIF-corrected RGB JPEG bytes."""

    # BREAKING: Imported images are now normalized to EXIF-corrected RGB JPEG for consistent downstream training inputs.

    image_module, image_ops, unidentified_image_error = _load_pillow()
    previous_max_image_pixels = image_module.MAX_IMAGE_PIXELS
    image_module.MAX_IMAGE_PIXELS = max_image_pixels
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", image_module.DecompressionBombWarning)
            try:
                with image_module.open(io.BytesIO(raw_bytes)) as probe:
                    probe.verify()
                with image_module.open(io.BytesIO(raw_bytes)) as image:
                    image = image_ops.exif_transpose(image)
                    image.load()
                    if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
                        rgba_image = image.convert("RGBA")
                        background = image_module.new("RGBA", rgba_image.size, (255, 255, 255, 255))
                        image = image_module.alpha_composite(background, rgba_image).convert("RGB")
                    elif image.mode != "RGB":
                        image = image.convert("RGB")
                    output = io.BytesIO()
                    image.save(output, format="JPEG", quality=jpeg_quality, optimize=True)
                    return output.getvalue()
            except (unidentified_image_error, image_module.DecompressionBombError, OSError) as exc:
                raise ValueError("public_seed_member_invalid_image") from exc
    finally:
        image_module.MAX_IMAGE_PIXELS = previous_max_image_pixels


def build_target_filename(*, prefix: str, label: str, ordinal: int, digest: str) -> str:
    """Build one deterministic content-addressed output filename."""

    # BREAKING: Output filenames are now stable content-addressed JPEG names instead of timestamp-slugged source-suffix names.
    return f"{prefix}-{label}-{ordinal:04d}-{digest[:16]}.jpg"


def write_bytes_atomically(path: Path, payload: bytes) -> None:
    """Write bytes to a path using a same-directory temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        delete=False,
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as temp_file:
        temp_file.write(payload)
        temp_file.flush()
        os.fsync(temp_file.fileno())
        temp_path = Path(temp_file.name)
    temp_path.replace(path)


def sha256_file(path: Path) -> str:
    """Compute one SHA-256 digest for a file."""

    hash_state = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            hash_state.update(chunk)
    return hash_state.hexdigest()


def _load_hf_hub_download():
    """Import huggingface_hub lazily so local-archive/help paths stay usable."""

    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise RuntimeError("huggingface_hub_required_for_hf_download") from exc
    return hf_hub_download


def _load_pillow() -> tuple[Any, Any, type[Exception]]:
    """Import Pillow lazily so --help and archive-only flows fail only when needed."""

    try:
        from PIL import Image, ImageOps, UnidentifiedImageError
    except ModuleNotFoundError as exc:
        raise RuntimeError("pillow_required_for_public_seed_import") from exc
    return Image, ImageOps, UnidentifiedImageError


def main(argv: list[str] | None = None) -> int:
    """Run the bounded public-seed import helper."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = import_public_seed_dataset(
            dataset_root=args.dataset_root,
            archive_path=args.archive_path,
            download_url=str(args.download_url),
            hf_repo_id=str(args.hf_repo_id),
            hf_filename=str(args.hf_filename),
            hf_revision=str(args.hf_revision),
            expected_sha256=args.expected_sha256,
            allow_unsafe_url=bool(args.allow_unsafe_url),
            count_per_label=args.count_per_label,
            seed=args.seed,
            prefix=str(args.prefix),
            jpeg_quality=args.jpeg_quality,
            max_archive_bytes=args.max_archive_bytes,
            max_member_bytes=args.max_member_bytes,
            max_image_pixels=args.max_image_pixels,
            max_compression_ratio=args.max_compression_ratio,
            dry_run=bool(args.dry_run),
        )
    except Exception as exc:
        json.dump(
            {
                "status": "error",
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
        return 1
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
