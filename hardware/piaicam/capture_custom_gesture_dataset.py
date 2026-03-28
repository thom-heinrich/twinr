#!/usr/bin/env python3
# CHANGELOG: 2026-03-27
# BUG-1: Replaced blocking capture_file loop with request-based capture + atomic rename so JPEG writes do not stall the camera event loop.
# BUG-2: Warm-up now waits for real camera frames/metadata and camera shutdown now still closes after stop() failures.
# SEC-1: Enforced dataset-root path containment plus sane count/pixel guardrails to reduce arbitrary writes and Pi 4 disk/RAM exhaustion.
# IMP-1: Upgraded to Picamera2 >=0.3.27 and modern still-capture defaults: YUV420 JPEG, queue=False, flush=True, extra buffers, structured JSON errors.
# IMP-2: Added autofocus/auto-control settling and per-run manifests with image metadata for reproducible MediaPipe dataset collection.
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   # BREAKING: relies on Picamera2 features added/fixed after 0.3.26
#   "picamera2>=0.3.27",
# ]
# ///
"""Capture labeled stills for Twinr's custom hand-gesture dataset.

This helper runs only on hardware that has a working Pi camera path. It keeps
data collection bounded and explicit: one label at a time, a fixed number of
JPEGs, and a deterministic output directory structure that the training helper
can consume directly.

Compared with the earlier helper, this version is stricter and more
reproducible:
- it validates that every planned output stays under the requested dataset root;
- it captures fresh requests and saves them atomically;
- it records per-image camera metadata in a run manifest stored outside the
  dataset tree, so MediaPipe label folders remain image-only.

Purpose
-------
Capture `none`, `thumbs_up`, `thumbs_down`, `peace_sign`, or later custom
labels into the folder layout expected by Twinr's MediaPipe gesture-model
training workflow.

Usage
-----
Command-line invocation examples::

    python3 hardware/piaicam/capture_custom_gesture_dataset.py --label none --count 24
    python3 hardware/piaicam/capture_custom_gesture_dataset.py --label peace sign --count 40 --interval-s 0.35
    python3 hardware/piaicam/capture_custom_gesture_dataset.py --label thumbs_down --dataset-root /twinr/state/mediapipe/custom_gesture_dataset

Outputs
-------
- Writes bounded JPEG captures under ``<dataset_root>/<label>/``.
- Writes one per-run JSON manifest beside the dataset tree.
- Prints a JSON summary listing the written files and manifest path.
"""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import uuid
from typing import Any, Iterator, Sequence

from custom_gesture_workflow import DEFAULT_DATASET_ROOT, plan_capture_targets

MAX_CAPTURE_COUNT = 2048
MAX_TOTAL_PIXELS = 16_000_000
DEFAULT_SETTLE_FRAMES = 8
DEFAULT_BUFFER_COUNT = 3
JPEG_SUFFIXES = {".jpg", ".jpeg"}


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
        help="Label name for this capture run, for example none, thumbs_up, thumbs_down, or peace_sign.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=24,
        help=f"Number of stills to capture for the selected label. Default 24; hard-limited to {MAX_CAPTURE_COUNT} unless --allow-large-runs is used.",
    )
    parser.add_argument(
        "--interval-s",
        type=float,
        default=0.40,
        help="Minimum delay between capture starts in seconds.",
    )
    parser.add_argument(
        "--warmup-s",
        type=float,
        default=1.2,
        help="Initial camera warm-up delay before settling/auto-focus.",
    )
    parser.add_argument(
        "--settle-frames",
        type=int,
        default=DEFAULT_SETTLE_FRAMES,
        help="Additional metadata frames to wait for after start and after control updates.",
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
    # BREAKING: AF-capable modules now default to a one-shot autofocus pass for sharper first-party dataset captures.
    parser.add_argument(
        "--focus-mode",
        choices=("auto_once", "continuous", "manual", "off", "inherit"),
        default="auto_once",
        help="Autofocus behaviour for AF-capable camera modules. 'inherit' keeps the camera default.",
    )
    parser.add_argument(
        "--lens-position",
        type=float,
        default=None,
        help="Manual lens position in dioptres when --focus-mode manual is used. 0.0 means infinity on AF-capable modules.",
    )
    # BREAKING: AE/AWB locking is enabled by default to reduce label-correlated camera drift inside one capture run.
    parser.add_argument(
        "--lock-auto-controls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Lock the current AE/AWB result after warm-up to reduce label-correlated drift across a capture run.",
    )
    parser.add_argument(
        "--write-manifest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write a per-run JSON manifest beside the dataset tree.",
    )
    parser.add_argument(
        "--allow-large-runs",
        action="store_true",
        help="Override the built-in Pi 4 safety guardrails for very large counts or unusually large resolutions.",
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
    focus_mode: str = "auto_once",
    lens_position: float | None = None,
    settle_frames: int = DEFAULT_SETTLE_FRAMES,
    lock_auto_controls: bool = True,
    write_manifest: bool = True,
    allow_large_runs: bool = False,
    camera_factory=None,
    sleep_fn=time.sleep,
) -> dict[str, object]:
    """Capture one bounded labeled image sequence from the Pi camera."""

    dataset_root = Path(dataset_root).expanduser()
    count = int(count)
    width = int(width)
    height = int(height)
    interval_s = float(interval_s)
    warmup_s = float(warmup_s)
    settle_frames = int(settle_frames)

    _validate_limits(
        count=count,
        width=width,
        height=height,
        interval_s=interval_s,
        warmup_s=warmup_s,
        settle_frames=settle_frames,
        allow_large_runs=allow_large_runs,
    )

    normalized_label, label_dir, targets = plan_capture_targets(
        dataset_root=dataset_root,
        label=label,
        count=count,
        prefix=prefix,
    )
    dataset_root_resolved, label_dir_resolved, targets = _validate_target_paths(
        dataset_root=dataset_root,
        label_dir=label_dir,
        targets=targets,
    )

    label_dir_resolved.mkdir(parents=True, exist_ok=True)

    started_at = _utcnow()
    run_id = _make_run_id(normalized_label)
    manifest_path = _manifest_path_for(dataset_root_resolved, normalized_label, run_id)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "status": "ok",
        "dataset_root": str(dataset_root_resolved),
        "label": normalized_label,
        "label_dir": str(label_dir_resolved),
        "planned_count": len(targets),
        "count": 0,
        "files": [],
        "run_id": run_id,
        "started_at_utc": started_at,
    }

    camera = _build_camera(camera_factory)
    camera_model = None
    camera_properties: dict[str, Any] = {}
    focus_info: dict[str, Any] = {"requested_mode": focus_mode, "status": "not_started"}
    lock_info: dict[str, Any] = {"requested": bool(lock_auto_controls), "applied": False}
    captures: list[dict[str, Any]] = []

    try:
        configuration = _create_still_configuration(camera, width=width, height=height)
        camera.configure(configuration)

        camera_properties = _json_safe(getattr(camera, "camera_properties", {}))
        camera_model = _camera_model_from_properties(camera_properties)

        _apply_pre_start_focus_controls(camera, focus_mode=focus_mode, lens_position=lens_position)
        camera.start()

        _initial_settle(camera=camera, warmup_s=warmup_s, settle_frames=settle_frames, sleep_fn=sleep_fn)
        focus_info = _stabilize_focus(
            camera=camera,
            focus_mode=focus_mode,
            lens_position=lens_position,
            settle_frames=settle_frames,
        )
        if lock_auto_controls:
            lock_info = _lock_current_auto_controls(camera=camera, settle_frames=settle_frames)

        next_capture_monotonic = time.monotonic()
        for index, target in enumerate(targets, start=1):
            now = time.monotonic()
            if now < next_capture_monotonic:
                sleep_fn(next_capture_monotonic - now)

            record = _capture_to_target(
                camera=camera,
                target=target,
                index=index,
            )
            captures.append(record)
            summary["count"] = len(captures)
            summary["files"] = [item["path"] for item in captures]
            next_capture_monotonic = max(next_capture_monotonic, time.monotonic()) + interval_s

    except KeyboardInterrupt:
        summary["status"] = "interrupted"
    finally:
        _close_camera(camera)

    manifest_data = {
        "run_id": run_id,
        "status": summary["status"],
        "started_at_utc": started_at,
        "finished_at_utc": _utcnow(),
        "dataset_root": str(dataset_root_resolved),
        "label": normalized_label,
        "label_dir": str(label_dir_resolved),
        "planned_count": len(targets),
        "captured_count": len(captures),
        "options": {
            "interval_s": interval_s,
            "warmup_s": warmup_s,
            "settle_frames": settle_frames,
            "width": width,
            "height": height,
            "prefix": prefix,
            "focus_mode": focus_mode,
            "lens_position": lens_position,
            "lock_auto_controls": lock_auto_controls,
            "write_manifest": write_manifest,
            "allow_large_runs": allow_large_runs,
            "buffer_count": DEFAULT_BUFFER_COUNT,
            "fresh_request_flush": True,
            "queue": False,
            "pixel_format": "YUV420",
        },
        "camera": {
            "model": camera_model,
            "properties": camera_properties,
            "picamera2_version": _picamera2_version(),
        },
        "focus": focus_info,
        "locked_controls": lock_info,
        "files": [item["path"] for item in captures],
        "captures": captures,
    }

    if write_manifest:
        _write_json_atomic(manifest_path, manifest_data)
        summary["manifest_path"] = str(manifest_path)

    summary["camera_model"] = camera_model
    summary["focus"] = focus_info
    summary["lock_auto_controls"] = lock_info
    summary["finished_at_utc"] = manifest_data["finished_at_utc"]
    return summary


def main(argv: list[str] | None = None) -> int:
    """Run the bounded dataset-capture helper."""

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        requested_count = int(args.count)
        requested_width = int(args.width)
        requested_height = int(args.height)
        requested_interval = float(args.interval_s)
        requested_warmup = float(args.warmup_s)
        requested_settle_frames = int(args.settle_frames)

        _validate_limits(
            count=requested_count,
            width=requested_width,
            height=requested_height,
            interval_s=requested_interval,
            warmup_s=requested_warmup,
            settle_frames=requested_settle_frames,
            allow_large_runs=bool(args.allow_large_runs),
        )

        normalized_label, label_dir, targets = plan_capture_targets(
            dataset_root=Path(args.dataset_root).expanduser(),
            label=args.label,
            count=requested_count,
            prefix=args.prefix,
        )
        dataset_root_resolved, label_dir_resolved, targets = _validate_target_paths(
            dataset_root=Path(args.dataset_root).expanduser(),
            label_dir=label_dir,
            targets=targets,
        )

        if args.dry_run:
            manifest_path = _manifest_path_for(dataset_root_resolved, normalized_label, _make_run_id(normalized_label))
            summary = {
                "status": "dry_run",
                "dataset_root": str(dataset_root_resolved),
                "label": normalized_label,
                "label_dir": str(label_dir_resolved),
                "planned_count": len(targets),
                "count": len(targets),
                "files": [str(path) for path in targets],
                "manifest_path": str(manifest_path) if args.write_manifest else None,
            }
        else:
            summary = capture_dataset(
                dataset_root=dataset_root_resolved,
                label=args.label,
                count=requested_count,
                interval_s=requested_interval,
                warmup_s=requested_warmup,
                width=requested_width,
                height=requested_height,
                prefix=args.prefix,
                focus_mode=args.focus_mode,
                lens_position=args.lens_position,
                settle_frames=requested_settle_frames,
                lock_auto_controls=bool(args.lock_auto_controls),
                write_manifest=bool(args.write_manifest),
                allow_large_runs=bool(args.allow_large_runs),
            )
    except KeyboardInterrupt:
        summary = {
            "status": "interrupted",
            "error": {
                "type": "KeyboardInterrupt",
                "message": "capture_interrupted",
            },
        }
        exit_code = 130
    except Exception as exc:
        summary = {
            "status": "error",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
        exit_code = 1
    else:
        exit_code = 0

    json.dump(_json_safe(summary), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return exit_code


def _validate_limits(
    *,
    count: int,
    width: int,
    height: int,
    interval_s: float,
    warmup_s: float,
    settle_frames: int,
    allow_large_runs: bool,
) -> None:
    """Validate resource and timing inputs for a Pi 4-friendly capture run."""

    if count < 1:
        raise ValueError("count_must_be_positive")
    if width < 1 or height < 1:
        raise ValueError("width_and_height_must_be_positive")
    if interval_s < 0.0:
        raise ValueError("interval_s_must_be_non_negative")
    if warmup_s < 0.0:
        raise ValueError("warmup_s_must_be_non_negative")
    if settle_frames < 0:
        raise ValueError("settle_frames_must_be_non_negative")

    total_pixels = width * height
    if not allow_large_runs:
        # BREAKING: very large runs now require an explicit opt-in instead of silently filling the SD card.
        if count > MAX_CAPTURE_COUNT:
            raise ValueError(f"count_exceeds_safe_limit:{MAX_CAPTURE_COUNT}")
        # BREAKING: unusually large still resolutions now require an explicit opt-in to avoid Pi 4 memory pressure.
        if total_pixels > MAX_TOTAL_PIXELS:
            raise ValueError(f"resolution_exceeds_safe_limit:{MAX_TOTAL_PIXELS}")


def _validate_target_paths(
    *,
    dataset_root: Path,
    label_dir: Path,
    targets: Sequence[Path],
) -> tuple[Path, Path, tuple[Path, ...]]:
    """Ensure the planner cannot escape the requested dataset root."""

    dataset_root_resolved = Path(dataset_root).expanduser().resolve(strict=False)
    label_dir_resolved = Path(label_dir).expanduser().resolve(strict=False)

    if not label_dir_resolved.is_relative_to(dataset_root_resolved):
        raise ValueError("label_dir_escapes_dataset_root")

    validated_targets: list[Path] = []
    for target in targets:
        resolved_target = Path(target).expanduser().resolve(strict=False)
        if not resolved_target.is_relative_to(dataset_root_resolved):
            raise ValueError("target_escapes_dataset_root")
        if resolved_target.suffix.lower() not in JPEG_SUFFIXES:
            raise ValueError("non_jpeg_target_not_allowed")
        validated_targets.append(resolved_target)

    return dataset_root_resolved, label_dir_resolved, tuple(validated_targets)


def _capture_to_target(*, camera, target: Path, index: int) -> dict[str, Any]:
    """Capture one fresh request and atomically publish it to its final path."""

    target.parent.mkdir(parents=True, exist_ok=True)
    temp_target = _temporary_target_for(target)
    metadata: dict[str, Any] = {}
    capture_started_at = _utcnow()
    capture_file = getattr(camera, "capture_file", None)

    try:
        if _has_request_capture_api(camera):
            with _captured_request(camera, flush=True) as request:
                request.save("main", str(temp_target))
                getter = getattr(request, "get_metadata", None)
                if callable(getter):
                    metadata = getter() or {}
        elif callable(capture_file):
            capture_file(str(temp_target))
        else:
            raise RuntimeError("capture_request_unavailable")
        os.replace(temp_target, target)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            temp_target.unlink()
        raise

    return {
        "index": index,
        "path": str(target),
        "captured_at_utc": capture_started_at,
        "size_bytes": target.stat().st_size,
        "metadata": _json_safe(metadata),
    }


@contextlib.contextmanager
def _captured_request(camera, *, flush: bool) -> Iterator[Any]:
    """Yield one fresh completed request and always release it correctly."""

    captured_request = getattr(camera, "captured_request", None)
    if callable(captured_request):
        try:
            with captured_request(flush=flush) as request:
                yield request
                return
        except TypeError:
            with captured_request() as request:
                yield request
                return

    capture_request = getattr(camera, "capture_request", None)
    if not callable(capture_request):
        raise RuntimeError("capture_request_unavailable")

    try:
        request = capture_request(flush=flush)
    except TypeError:
        request = capture_request()

    try:
        yield request
    finally:
        releaser = getattr(request, "release", None)
        if callable(releaser):
            releaser()


def _has_request_capture_api(camera) -> bool:
    """Return whether the camera supports fresh-request capture APIs."""

    return callable(getattr(camera, "captured_request", None)) or callable(
        getattr(camera, "capture_request", None)
    )


def _initial_settle(*, camera, warmup_s: float, settle_frames: int, sleep_fn) -> None:
    """Let the camera deliver real frames before the first capture."""

    if warmup_s > 0.0:
        sleep_fn(warmup_s)
    _drain_metadata_frames(camera, settle_frames)


def _stabilize_focus(
    *,
    camera,
    focus_mode: str,
    lens_position: float | None,
    settle_frames: int,
) -> dict[str, Any]:
    """Apply AF policy when the module supports it."""

    info: dict[str, Any] = {
        "requested_mode": focus_mode,
        "supported": False,
        "status": "skipped",
    }
    if focus_mode == "inherit":
        info["status"] = "inherited"
        return info

    if not _camera_supports_control(camera, "AfMode") and not callable(getattr(camera, "autofocus_cycle", None)):
        info["status"] = "unsupported"
        return info

    info["supported"] = True
    try:
        from libcamera import controls  # pylint: disable=import-error
    except Exception:
        info["status"] = "libcamera_controls_unavailable"
        return info

    try:
        if focus_mode == "continuous":
            _set_controls(camera, {"AfMode": controls.AfModeEnum.Continuous})
            _drain_metadata_frames(camera, max(1, settle_frames))
            info["status"] = "continuous_active"
            return info

        if focus_mode == "manual":
            requested_lens = lens_position
            if requested_lens is None:
                requested_lens = _default_lens_position(camera)
            control_values: dict[str, Any] = {"AfMode": controls.AfModeEnum.Manual}
            if requested_lens is not None and _camera_supports_control(camera, "LensPosition"):
                control_values["LensPosition"] = float(requested_lens)
            _set_controls(camera, control_values)
            _drain_metadata_frames(camera, max(1, settle_frames))
            info["status"] = "manual_active"
            info["lens_position"] = requested_lens
            return info

        if focus_mode == "off":
            _set_controls(camera, {"AfMode": controls.AfModeEnum.Manual})
            _drain_metadata_frames(camera, max(1, settle_frames))
            info["status"] = "autofocus_disabled"
            return info

        autofocus_cycle = getattr(camera, "autofocus_cycle", None)
        if callable(autofocus_cycle):
            if _camera_supports_control(camera, "AfMode"):
                with contextlib.suppress(Exception):
                    _set_controls(camera, {"AfMode": controls.AfModeEnum.Auto})
            success = autofocus_cycle()
            _drain_metadata_frames(camera, max(1, settle_frames))
            info["status"] = "auto_once_complete"
            info["success"] = bool(success)
            return info

        info["status"] = "autofocus_helper_unavailable"
        return info
    except Exception as exc:
        info["status"] = "focus_error"
        info["error"] = f"{type(exc).__name__}:{exc}"
        return info


def _apply_pre_start_focus_controls(camera, *, focus_mode: str, lens_position: float | None) -> None:
    """Apply controls that should already affect the first live frames."""

    if focus_mode not in {"continuous", "manual", "off"}:
        return
    if not _camera_supports_control(camera, "AfMode"):
        return

    try:
        from libcamera import controls  # pylint: disable=import-error
    except Exception:
        return

    control_values: dict[str, Any] = {}
    if focus_mode == "continuous":
        control_values["AfMode"] = controls.AfModeEnum.Continuous
    elif focus_mode in {"manual", "off"}:
        control_values["AfMode"] = controls.AfModeEnum.Manual
        if focus_mode == "manual" and lens_position is not None and _camera_supports_control(camera, "LensPosition"):
            control_values["LensPosition"] = float(lens_position)
    if control_values:
        with contextlib.suppress(Exception):
            _set_controls(camera, control_values)


def _lock_current_auto_controls(*, camera, settle_frames: int) -> dict[str, Any]:
    """Freeze the current AE/AWB result to keep one capture run consistent."""

    current = _capture_metadata(camera)
    if not current:
        return {"requested": True, "applied": False, "status": "metadata_unavailable"}

    controls_to_lock: dict[str, Any] = {}

    if _camera_supports_control(camera, "ExposureTime") and "ExposureTime" in current:
        controls_to_lock["ExposureTime"] = int(current["ExposureTime"])
    if _camera_supports_control(camera, "AnalogueGain") and "AnalogueGain" in current:
        controls_to_lock["AnalogueGain"] = float(current["AnalogueGain"])
    if controls_to_lock and _camera_supports_control(camera, "AeEnable"):
        controls_to_lock["AeEnable"] = False

    colour_gains = current.get("ColourGains")
    if colour_gains is not None:
        try:
            colour_gains = tuple(float(component) for component in colour_gains)
        except TypeError:
            colour_gains = None
    if colour_gains is not None and _camera_supports_control(camera, "ColourGains"):
        controls_to_lock["ColourGains"] = colour_gains
    if (
        ("ColourGains" in controls_to_lock or "ColourTemperature" in current or "ColourCorrectionMatrix" in current)
        and _camera_supports_control(camera, "AwbEnable")
    ):
        controls_to_lock["AwbEnable"] = False

    if not controls_to_lock:
        return {"requested": True, "applied": False, "status": "no_lockable_controls"}

    try:
        _set_controls(camera, controls_to_lock)
        _drain_metadata_frames(camera, max(1, settle_frames))
        confirmed = _capture_metadata(camera)
        return {
            "requested": True,
            "applied": True,
            "status": "locked",
            "controls": _json_safe(controls_to_lock),
            "confirmed_metadata": _json_safe(_interesting_metadata_subset(confirmed)),
        }
    except Exception as exc:
        return {
            "requested": True,
            "applied": False,
            "status": "lock_error",
            "error": f"{type(exc).__name__}:{exc}",
            "controls": _json_safe(controls_to_lock),
        }


def _drain_metadata_frames(camera, frame_count: int) -> None:
    """Wait for a number of real frames, if metadata capture is available."""

    if frame_count <= 0:
        return
    capture_metadata = getattr(camera, "capture_metadata", None)
    if not callable(capture_metadata):
        return
    for _ in range(frame_count):
        with contextlib.suppress(Exception):
            capture_metadata()


def _capture_metadata(camera) -> dict[str, Any]:
    """Return the latest capture metadata, or an empty dict if unavailable."""

    capture_metadata = getattr(camera, "capture_metadata", None)
    if not callable(capture_metadata):
        return {}
    try:
        metadata = capture_metadata()
    except Exception:
        return {}
    return dict(metadata or {})


def _interesting_metadata_subset(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return the most useful fields for later dataset triage."""

    keys = (
        "SensorTimestamp",
        "ExposureTime",
        "AnalogueGain",
        "ColourGains",
        "ColourTemperature",
        "LensPosition",
        "AfState",
        "FrameDuration",
        "Lux",
    )
    return {key: metadata[key] for key in keys if key in metadata}


def _camera_supports_control(camera, name: str) -> bool:
    """Check whether the active camera advertises a named control."""

    controls_map = getattr(camera, "camera_controls", None)
    if isinstance(controls_map, dict):
        return name in controls_map
    return False


def _default_lens_position(camera) -> float | None:
    """Return the camera's advertised default lens position if available."""

    controls_map = getattr(camera, "camera_controls", None)
    if not isinstance(controls_map, dict):
        return None
    lens_info = controls_map.get("LensPosition")
    if isinstance(lens_info, (list, tuple)) and len(lens_info) >= 3:
        try:
            return float(lens_info[2])
        except (TypeError, ValueError):
            return None
    return None


def _set_controls(camera, control_values: dict[str, Any]) -> None:
    """Set camera controls in one call."""

    if not control_values:
        return
    setter = getattr(camera, "set_controls", None)
    if not callable(setter):
        raise RuntimeError("set_controls_unavailable")
    setter(control_values)


def _build_camera(camera_factory):
    """Build one Picamera2-compatible camera object lazily."""

    if camera_factory is not None:
        return camera_factory()
    try:
        from picamera2 import Picamera2  # pylint: disable=import-error
    except Exception as exc:  # pragma: no cover - depends on the host environment.
        raise RuntimeError("picamera2_unavailable") from exc
    return Picamera2()


def _create_still_configuration(camera, *, width: int, height: int) -> Any:
    """Create a still configuration while tolerating older/fake test doubles."""

    create_still_configuration = getattr(camera, "create_still_configuration", None)
    if not callable(create_still_configuration):
        raise RuntimeError("create_still_configuration_unavailable")

    option_sets = (
        {
            "main": {"size": (width, height), "format": "YUV420"},
            "buffer_count": DEFAULT_BUFFER_COUNT,
            "queue": False,
        },
        {
            "main": {"size": (width, height), "format": "YUV420"},
            "queue": False,
        },
        {
            "main": {"size": (width, height), "format": "YUV420"},
        },
        {
            "main": {"size": (width, height)},
        },
    )
    last_type_error: TypeError | None = None
    for kwargs in option_sets:
        try:
            return create_still_configuration(**kwargs)
        except TypeError as exc:
            last_type_error = exc
    if last_type_error is not None:
        raise last_type_error
    raise RuntimeError("create_still_configuration_failed")


def _close_camera(camera) -> None:
    """Stop and close one Picamera2-compatible camera object if supported."""

    for method_name in ("stop", "close"):
        method = getattr(camera, method_name, None)
        if callable(method):
            with contextlib.suppress(Exception):
                method()


def _temporary_target_for(target: Path) -> Path:
    """Create a same-directory temporary filename that keeps the final suffix."""

    unique = uuid.uuid4().hex[:8]
    return target.with_name(f".{target.stem}.tmp.{unique}{target.suffix}")


def _manifest_path_for(dataset_root: Path, normalized_label: str, run_id: str) -> Path:
    """Place manifests beside the dataset tree so label folders stay image-only."""

    dataset_root = dataset_root.resolve(strict=False)
    manifest_root = dataset_root.parent / f".{dataset_root.name}_capture_manifests"
    return manifest_root / normalized_label / f"{run_id}.json"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically to avoid torn manifests on sudden interruption."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.stem}.tmp.{uuid.uuid4().hex[:8]}{path.suffix}")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temp_path.unlink()


def _camera_model_from_properties(camera_properties: dict[str, Any]) -> str | None:
    """Extract a stable camera model string from camera properties."""

    for key in ("Model", "SensorModel", "CameraModel"):
        value = camera_properties.get(key)
        if value:
            return str(value)
    return None


def _make_run_id(normalized_label: str) -> str:
    """Build a collision-resistant but human-readable run id."""

    safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in normalized_label).strip("_") or "label"
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{safe_label}_{uuid.uuid4().hex[:8]}"


def _picamera2_version() -> str | None:
    """Return the installed Picamera2 version when discoverable."""

    try:
        return importlib.metadata.version("picamera2")
    except importlib.metadata.PackageNotFoundError:
        return None


def _utcnow() -> str:
    """Return a UTC timestamp in ISO 8601 format."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_safe(value: Any) -> Any:
    """Convert libcamera/Picamera2 objects into JSON-safe Python values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_tuple") and callable(value.to_tuple):
        return _json_safe(value.to_tuple())
    if hasattr(value, "tolist") and callable(value.tolist):
        return _json_safe(value.tolist())
    if hasattr(value, "name") and isinstance(getattr(value, "name"), str):
        return value.name
    return repr(value)


if __name__ == "__main__":
    raise SystemExit(main())
