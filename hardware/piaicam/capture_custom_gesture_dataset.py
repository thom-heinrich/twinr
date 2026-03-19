#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "picamera2>=0.3",
# ]
# ///
"""Capture labeled stills for Twinr's custom hand-gesture dataset.

This helper runs only on hardware that has a working Pi camera path. It keeps
data collection bounded and explicit: one label at a time, a fixed number of
JPEGs, and a deterministic output directory structure that the training helper
can consume directly.

Purpose
-------
Capture `none`, `ok_sign`, `middle_finger`, or later custom labels into the
folder layout expected by Twinr's MediaPipe gesture-model training workflow.

Usage
-----
Command-line invocation examples::

    python3 hardware/piaicam/capture_custom_gesture_dataset.py --label none --count 24
    python3 hardware/piaicam/capture_custom_gesture_dataset.py --label ok sign --count 40 --interval-s 0.35
    python3 hardware/piaicam/capture_custom_gesture_dataset.py --label middle_finger --dataset-root /twinr/state/mediapipe/custom_gesture_dataset

Outputs
-------
- Writes bounded JPEG captures under ``<dataset_root>/<label>/``.
- Prints a JSON summary listing the written files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys
import time

from custom_gesture_workflow import DEFAULT_DATASET_ROOT, plan_capture_targets


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the bounded capture helper."""

    parser = argparse.ArgumentParser(
        description="Capture a bounded labeled still-image set for Twinr's custom MediaPipe gesture workflow.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root directory that will receive one label subdirectory per gesture.",
    )
    parser.add_argument(
        "--label",
        required=True,
        help="Label name for this capture run, for example none, ok_sign, or middle_finger.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=24,
        help="Number of stills to capture for the selected label.",
    )
    parser.add_argument(
        "--interval-s",
        type=float,
        default=0.40,
        help="Delay between still captures in seconds.",
    )
    parser.add_argument(
        "--warmup-s",
        type=float,
        default=1.2,
        help="Initial camera warm-up delay before the first capture.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Still-capture width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Still-capture height in pixels.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Optional filename stem prefix. Defaults to the normalized label name.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned output files without touching the camera.",
    )
    return parser


def capture_dataset(
    *,
    dataset_root: Path,
    label: str,
    count: int,
    interval_s: float,
    warmup_s: float,
    width: int,
    height: int,
    prefix: str | None = None,
    camera_factory=None,
    sleep_fn=time.sleep,
) -> dict[str, object]:
    """Capture one bounded labeled image sequence from the Pi camera."""

    normalized_label, label_dir, targets = plan_capture_targets(
        dataset_root=dataset_root,
        label=label,
        count=count,
        prefix=prefix,
    )
    label_dir.mkdir(parents=True, exist_ok=True)
    camera = _build_camera(camera_factory)
    try:
        configuration = camera.create_still_configuration(main={"size": (max(1, int(width)), max(1, int(height)))})
        camera.configure(configuration)
        camera.start()
        if warmup_s > 0.0:
            sleep_fn(float(warmup_s))
        for index, target in enumerate(targets):
            camera.capture_file(str(target))
            if index + 1 < len(targets) and interval_s > 0.0:
                sleep_fn(float(interval_s))
    finally:
        _close_camera(camera)
    return {
        "dataset_root": str(Path(dataset_root)),
        "label": normalized_label,
        "count": len(targets),
        "files": [str(path) for path in targets],
    }


def main(argv: list[str] | None = None) -> int:
    """Run the bounded dataset-capture helper."""

    parser = build_parser()
    args = parser.parse_args(argv)
    normalized_label, label_dir, targets = plan_capture_targets(
        dataset_root=args.dataset_root,
        label=args.label,
        count=max(1, int(args.count)),
        prefix=args.prefix,
    )
    if args.dry_run:
        summary = {
            "dataset_root": str(args.dataset_root),
            "label": normalized_label,
            "count": len(targets),
            "label_dir": str(label_dir),
            "files": [str(path) for path in targets],
            "status": "dry_run",
        }
    else:
        summary = capture_dataset(
            dataset_root=args.dataset_root,
            label=args.label,
            count=max(1, int(args.count)),
            interval_s=max(0.0, float(args.interval_s)),
            warmup_s=max(0.0, float(args.warmup_s)),
            width=max(1, int(args.width)),
            height=max(1, int(args.height)),
            prefix=args.prefix,
        )
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def _build_camera(camera_factory):
    """Build one Picamera2-compatible camera object lazily."""

    if camera_factory is not None:
        return camera_factory()
    try:
        from picamera2 import Picamera2
    except Exception as exc:  # pragma: no cover - depends on the host environment.
        raise RuntimeError("picamera2_unavailable") from exc
    return Picamera2()


def _close_camera(camera) -> None:
    """Stop and close one Picamera2-compatible camera object if supported."""

    for method_name in ("stop", "close"):
        method = getattr(camera, method_name, None)
        if callable(method):
            method()


if __name__ == "__main__":
    raise SystemExit(main())
