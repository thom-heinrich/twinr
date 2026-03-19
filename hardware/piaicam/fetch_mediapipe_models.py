#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///
"""Download the official MediaPipe task models needed by Twinr's Pi camera path.

This helper is intentionally separate from the runtime. It fetches the official
pose, hand-landmarker, and built-in gesture task bundles into a bounded local
directory so the runtime does not perform hidden downloads during normal camera
inference.

Purpose
-------
Stage the official MediaPipe `.task` assets required by Twinr's local-first
Pi camera pipeline before runtime acceptance or deployment.

Usage
-----
Command-line invocation examples::

    python3 hardware/piaicam/fetch_mediapipe_models.py
    python3 hardware/piaicam/fetch_mediapipe_models.py --output-dir /twinr/state/mediapipe/models
    python3 hardware/piaicam/fetch_mediapipe_models.py --force

Outputs
-------
- Writes bounded `.task` model files under the selected output directory.
- Prints a JSON summary describing file paths, sizes, and sha256 digests.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
import sys
import urllib.request


DEFAULT_OUTPUT_DIR = Path("state/mediapipe/models")
DEFAULT_TIMEOUT_S = 60.0

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
)
HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
GESTURE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
)


@dataclass(frozen=True, slots=True)
class DownloadSpec:
    """Describe one official model asset to fetch."""

    name: str
    url: str
    filename: str


MODEL_SPECS = (
    DownloadSpec(
        name="pose_landmarker_full",
        url=POSE_MODEL_URL,
        filename="pose_landmarker_full.task",
    ),
    DownloadSpec(
        name="hand_landmarker",
        url=HAND_LANDMARKER_MODEL_URL,
        filename="hand_landmarker.task",
    ),
    DownloadSpec(
        name="gesture_recognizer",
        url=GESTURE_MODEL_URL,
        filename="gesture_recognizer.task",
    ),
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the bounded model fetch helper."""

    parser = argparse.ArgumentParser(
        description="Download the official MediaPipe task models used by Twinr's Pi-side camera pipeline.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the task bundles will be written.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even when they already exist.",
    )
    return parser


def download_models(*, output_dir: Path, timeout_s: float, force: bool) -> dict[str, object]:
    """Download the official task bundles and return a small JSON-friendly summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    for spec in MODEL_SPECS:
        target = output_dir / spec.filename
        if target.exists() and not force:
            results.append(
                {
                    "name": spec.name,
                    "path": str(target),
                    "status": "present",
                    "size_bytes": target.stat().st_size,
                    "sha256": _sha256(target),
                }
            )
            continue
        with urllib.request.urlopen(spec.url, timeout=timeout_s) as response:
            payload = response.read()
        target.write_bytes(payload)
        results.append(
            {
                "name": spec.name,
                "path": str(target),
                "status": "downloaded",
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return {
        "output_dir": str(output_dir),
        "models": results,
    }


def _sha256(path: Path) -> str:
    """Return the sha256 digest for one file on disk."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    """Run the bounded official-model fetch helper."""

    parser = build_parser()
    args = parser.parse_args(argv)
    summary = download_models(
        output_dir=args.output_dir,
        timeout_s=max(1.0, float(args.timeout_s)),
        force=bool(args.force),
    )
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write(os.linesep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
