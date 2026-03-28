#!/usr/bin/env python3
# CHANGELOG: 2026-03-27
# BUG-1: Prevent stale artifacts in reused --output-dir directories from causing false-positive passes.
# BUG-2: Make auto-generated run directories collision-resistant under concurrent / same-second starts.
# BUG-3: Remove the unnecessary hard dependency on imx500-package; packaging is not required for runtime smoke tests.
# BUG-4: Accept rpicam metadata files that serialize one list of per-frame JSON objects instead of one top-level dict.
# BUG-5: Use `rpicam-vid` for legacy AI config smokes because this Pi build emits JSON metadata there, while `rpicam-hello --metadata` does not.
# SEC-1: Create private 0700/0600 run artifacts by default and reject symlinked --output-dir paths.
# IMP-1: Prefer 2026-frontier IMX500 smoke tests via Picamera2 + model-zoo RPKs when available, with automatic fallback to legacy rpicam JSON assets.
# IMP-2: Preflight-check firmware/models, capture environment/version details, and verify AI inference from IMX500 metadata/KPI when possible.
# /// script
# requires-python = ">=3.10"
# ///

"""Run bounded smoke tests for the Raspberry Pi AI Camera.

Purpose
-------
Verify that the Raspberry Pi AI Camera is detectable, can capture stills,
can record a short video, and can initialize one or more IMX500 AI paths
without starting the full Twinr runtime.

Usage
-----
Command-line invocation examples::

    python hardware/piaicam/smoke_piaicam.py
    python hardware/piaicam/smoke_piaicam.py --profile quick
    python hardware/piaicam/smoke_piaicam.py --ai-config /usr/share/rpi-camera-assets/imx500_posenet.json
    python hardware/piaicam/smoke_piaicam.py --ai-model /usr/share/imx500-models/imx500_network_yolo11n_pp.rpk

Inputs
------
- ``--camera`` camera index to test (default: 0)
- ``--profile`` whether to run the quick or full test plan
- ``--ai-backend`` auto / Picamera2(model-zoo RPK) / rpicam(JSON assets)
- ``--ai-model`` optional repeated override for IMX500 ``.rpk`` model firmware
- ``--ai-config`` optional repeated override for legacy rpicam post-process JSON
- ``--output-dir`` optional explicit run directory for artifacts and logs

Outputs
-------
- Creates one run directory under ``state/piaicam/runs/`` by default
- Writes per-phase stdout/stderr logs plus still/video artifacts
- Writes ``summary.json`` with the full result set
- Exit code 0 when every enabled phase succeeds, 1 otherwise

Notes
-----
The default ``auto`` AI backend now prefers Picamera2 with current
``/usr/share/imx500-models`` RPKs when available. The script falls back
to the legacy ``/usr/share/rpi-camera-assets`` JSON assets when Picamera2
or frontier RPK models are unavailable.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Iterable, Literal

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_ROOT = REPO_ROOT / "state" / "piaicam" / "runs"
DEFAULT_ASSET_DIR = Path("/usr/share/rpi-camera-assets")
DEFAULT_MODEL_DIR = Path("/usr/share/imx500-models")
DEFAULT_FIRMWARE_FILES = (
    Path("/lib/firmware/imx500_loader.fpk"),
    Path("/lib/firmware/imx500_firmware.fpk"),
)
LEGACY_FULL_AI_CONFIGS = (
    "imx500_mobilenet_ssd.json",
    "imx500_posenet.json",
)
LEGACY_QUICK_AI_CONFIGS = LEGACY_FULL_AI_CONFIGS[:1]
FRONTIER_FULL_AI_MODELS = (
    "imx500_network_yolo11n_pp.rpk",
    "imx500_network_efficientnetv2_b0.rpk",
    "imx500_network_deeplabv3plus.rpk",
    "imx500_network_higherhrnet_coco.rpk",
)
FRONTIER_QUICK_AI_MODELS = FRONTIER_FULL_AI_MODELS[:1]
BASE_REQUIRED_TOOLS = (
    "rpicam-hello",
    "rpicam-still",
    "rpicam-vid",
)
OPTIONAL_TOOLS = ("imx500-package",)
AI_METADATA_KEYS = (
    "CnnOutputTensor",
    "CnnInputTensor",
    "CnnOutputTensorInfo",
    "CnnInputTensorInfo",
    "CnnKpiInfo",
)
DIR_MODE = 0o700
FILE_MODE = 0o600


@dataclass
class AISpec:
    """Describe one AI smoke-test phase."""

    name: str
    path: str
    backend: str
    kind: str
    task: str
    source: str


@dataclass
class StepResult:
    """Record the outcome of one smoke-test phase."""

    name: str
    ok: bool
    summary: str
    command: list[str] = field(default_factory=list)
    returncode: int | None = None
    duration_s: float = 0.0
    stdout_log: str | None = None
    stderr_log: str | None = None
    artifacts: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


class WallClockTimeout(RuntimeError):
    """Raised when a bounded local phase exceeds its wall-clock budget."""


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the Pi AI camera smoke test."""

    parser = argparse.ArgumentParser(description="Run bounded Pi AI Camera smoke tests")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to test")
    parser.add_argument(
        "--profile",
        choices=("quick", "full"),
        default="full",
        help="Quick runs one AI target; full runs the broader plan",
    )
    # BREAKING: auto now prefers Picamera2 + model-zoo RPKs when they are present.
    parser.add_argument(
        "--ai-backend",
        choices=("auto", "picamera2", "rpicam"),
        default="auto",
        help="auto prefers frontier Picamera2/RPK tests, otherwise falls back to legacy rpicam JSON assets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Explicit run directory for logs and artifacts",
    )
    parser.add_argument(
        "--still-timeout",
        type=float,
        default=2.0,
        help="Seconds to allow the still capture phase inside rpicam-still",
    )
    parser.add_argument(
        "--video-timeout",
        type=float,
        default=3.0,
        help="Seconds to record in the short video phase",
    )
    parser.add_argument(
        "--video-width",
        type=int,
        default=1280,
        help="Requested width for the short video phase",
    )
    parser.add_argument(
        "--video-height",
        type=int,
        default=720,
        help="Requested height for the short video phase",
    )
    parser.add_argument(
        "--ai-timeout",
        type=float,
        default=2.0,
        help="Seconds to observe inference metadata after the AI pipeline starts",
    )
    parser.add_argument(
        "--ai-max-seconds",
        type=float,
        default=300.0,
        help="Wall-clock timeout per AI phase, including IMX500 firmware upload",
    )
    parser.add_argument(
        "--ai-model",
        action="append",
        default=[],
        help="Optional repeated IMX500 .rpk override (frontier Picamera2 path)",
    )
    parser.add_argument(
        "--ai-config",
        action="append",
        default=[],
        help="Optional repeated rpicam post-process JSON override (legacy path)",
    )
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=DEFAULT_ASSET_DIR,
        help="Directory holding built-in AI post-process JSON files",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory holding built-in IMX500 model-zoo .rpk files",
    )
    parser.add_argument(
        "--ai-warmup-frames",
        type=int,
        default=4,
        help="Frames to ignore before requiring AI metadata in Picamera2 model phases",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop after the first non-fatal phase failure",
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject invalid runtime arguments up-front."""

    if args.camera < 0:
        parser.error("--camera must be >= 0")
    if args.still_timeout <= 0:
        parser.error("--still-timeout must be > 0")
    if args.video_timeout <= 0:
        parser.error("--video-timeout must be > 0")
    if args.video_width <= 0:
        parser.error("--video-width must be > 0")
    if args.video_height <= 0:
        parser.error("--video-height must be > 0")
    if args.ai_timeout <= 0:
        parser.error("--ai-timeout must be > 0")
    if args.ai_max_seconds <= 0:
        parser.error("--ai-max-seconds must be > 0")
    if args.ai_warmup_frames < 0:
        parser.error("--ai-warmup-frames must be >= 0")
    if args.ai_model and args.ai_config:
        parser.error("--ai-model and --ai-config are mutually exclusive")


def _timestamp_slug() -> str:
    """Return a UTC timestamp suitable for collision-resistant run directory names."""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _milliseconds(seconds: float) -> str:
    """Convert CLI second values into rpicam millisecond arguments."""

    return f"{max(1, round(seconds * 1000))}ms"


def _safe_name(raw: str) -> str:
    """Map arbitrary labels to filesystem-safe log file stems."""

    cleaned: list[str] = []
    for char in raw.lower():
        if char.isalnum():
            cleaned.append(char)
        else:
            cleaned.append("_")
    value = "".join(cleaned)
    while "__" in value:
        value = value.replace("__", "_")
    return value.strip("_") or "phase"


def _chmod_private(path: Path) -> None:
    """Best-effort chmod for generated artifacts."""

    try:
        mode = DIR_MODE if path.is_dir() and not path.is_symlink() else FILE_MODE
        path.chmod(mode)
    except OSError:
        pass


def _mkdir_private(path: Path, *, exist_ok: bool) -> None:
    """Create a private directory and normalize permissions."""

    path.mkdir(parents=True, exist_ok=exist_ok, mode=DIR_MODE)
    _chmod_private(path)


def _atomic_write_text(path: Path, content: str) -> None:
    """Persist a UTF-8 text file with an atomic replace."""

    _mkdir_private(path.parent, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        os.fchmod(fd, FILE_MODE)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        _chmod_private(path)
    finally:
        try:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        except OSError:
            pass


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Persist a JSON file atomically."""

    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _delete_file_if_exists(path: Path) -> None:
    """Delete an existing output file/symlink so a later existence check is fresh."""

    try:
        if path.is_symlink() or path.exists():
            if path.is_dir() and not path.is_symlink():
                raise RuntimeError(f"refusing to remove directory where a file artifact is expected: {path}")
            path.unlink()
    except FileNotFoundError:
        return


def _ensure_no_existing_symlink_in_path(path: Path) -> None:
    """Reject output paths that traverse already-existing symlinks."""

    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    for node in [candidate, *candidate.parents]:
        if node.exists() and node.is_symlink():
            raise RuntimeError(f"refusing symlinked output path component: {node}")


def _create_run_dir(requested: Path | None) -> tuple[Path, bool]:
    """Create the run directory and report whether it was auto-generated."""

    _mkdir_private(DEFAULT_RUNS_ROOT, exist_ok=True)

    if requested is not None:
        # BREAKING: symlinked output directories are rejected to avoid writing camera
        # artifacts outside the intended directory tree.
        _ensure_no_existing_symlink_in_path(requested)
        requested = requested.expanduser()
        if not requested.is_absolute():
            requested = Path.cwd() / requested
        if requested.exists() and not requested.is_dir():
            raise RuntimeError(f"--output-dir exists but is not a directory: {requested}")
        _mkdir_private(requested, exist_ok=True)
        return requested.resolve(), False

    pid = os.getpid()
    for attempt in range(100):
        suffix = "" if attempt == 0 else f"-{attempt}"
        run_dir = DEFAULT_RUNS_ROOT / f"{_timestamp_slug()}-{pid}{suffix}"
        try:
            _mkdir_private(run_dir, exist_ok=False)
        except FileExistsError:
            time.sleep(0.001)
            continue
        latest_link = DEFAULT_RUNS_ROOT / "latest"
        try:
            if latest_link.is_symlink() or latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(run_dir.name)
        except OSError:
            pass
        return run_dir, True
    raise RuntimeError("unable to allocate a unique run directory after repeated attempts")


def _coerce_text(value: str | bytes | None) -> str:
    """Normalize subprocess output to text."""

    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _validate_expected_files(paths: Iterable[Path]) -> tuple[bool, list[str], dict[str, int]]:
    """Ensure expected artifacts exist, are regular files, and are non-empty."""

    problems: list[str] = []
    sizes: dict[str, int] = {}
    for path in paths:
        if not path.exists():
            problems.append(f"{path.name}: missing")
            continue
        if not path.is_file():
            problems.append(f"{path.name}: not a regular file")
            continue
        size = path.stat().st_size
        sizes[path.name] = size
        if size <= 0:
            problems.append(f"{path.name}: empty")
        _chmod_private(path)
    return not problems, problems, sizes


def _run_command(
    *,
    name: str,
    command: list[str],
    run_dir: Path,
    timeout_s: float,
    expected_files: list[Path] | None = None,
) -> StepResult:
    """Execute one command, log stdout/stderr, and collect artifact paths."""

    expected_files = expected_files or []
    start = time.perf_counter()
    safe = _safe_name(name)
    stdout_path = run_dir / f"{safe}.stdout.log"
    stderr_path = run_dir / f"{safe}.stderr.log"

    for path in [stdout_path, stderr_path, *expected_files]:
        _delete_file_if_exists(path)

    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=max(1.0, timeout_s),
            check=False,
            errors="replace",
        )
        stdout = _coerce_text(completed.stdout)
        stderr = _coerce_text(completed.stderr)
        returncode = completed.returncode
        ok = returncode == 0
        summary = "command completed successfully" if ok else f"command exited with return code {returncode}"
    except subprocess.TimeoutExpired as exc:
        stdout = _coerce_text(exc.stdout)
        stderr = _coerce_text(exc.stderr)
        returncode = None
        ok = False
        summary = f"command timed out after {timeout_s:.1f}s"

    _atomic_write_text(stdout_path, stdout)
    _atomic_write_text(stderr_path, stderr)

    files_ok, file_problems, file_sizes = _validate_expected_files(expected_files)
    if ok and not files_ok:
        ok = False
        summary = "expected artifacts invalid: " + ", ".join(file_problems)

    artifacts = [str(path.resolve()) for path in expected_files if path.exists()]
    duration_s = time.perf_counter() - start

    return StepResult(
        name=name,
        ok=ok,
        summary=summary,
        command=command,
        returncode=returncode,
        duration_s=duration_s,
        stdout_log=str(stdout_path.resolve()),
        stderr_log=str(stderr_path.resolve()),
        artifacts=artifacts,
        details={"expected_file_sizes": file_sizes},
    )


def _distribution_version(name: str) -> str | None:
    """Return an installed package version if importlib metadata can resolve it."""

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _picamera2_module_available() -> bool:
    """Return whether the Picamera2 module can be imported."""

    return importlib.util.find_spec("picamera2") is not None


def _infer_model_task(path: Path) -> str:
    """Infer the task family from a model filename."""

    name = path.name.lower()
    if "deeplab" in name or "segment" in name:
        return "segmentation"
    if "higherhrnet" in name or "posenet" in name or "pose" in name:
        return "pose_estimation"
    if "yolo" in name or "efficientdet" in name or "nanodet" in name or "ssd_" in name:
        return "object_detection"
    return "classification"


def _infer_config_task(path: Path) -> str:
    """Infer the task family from a legacy config filename."""

    name = path.name.lower()
    if "pose" in name:
        return "pose_estimation"
    if "segment" in name:
        return "segmentation"
    return "object_detection"


def _make_model_spec(path: Path, *, source: str) -> AISpec:
    """Build one model-backed AI phase description."""

    resolved = path.expanduser().resolve()
    return AISpec(
        name=resolved.stem,
        path=str(resolved),
        backend="picamera2",
        kind="model",
        task=_infer_model_task(resolved),
        source=source,
    )


def _make_config_spec(path: Path, *, source: str) -> AISpec:
    """Build one legacy config-backed AI phase description."""

    resolved = path.expanduser().resolve()
    return AISpec(
        name=resolved.stem,
        path=str(resolved),
        backend="rpicam",
        kind="config",
        task=_infer_config_task(resolved),
        source=source,
    )


def _resolve_ai_specs(args: argparse.Namespace) -> tuple[list[AISpec], str]:
    """Resolve the requested AI targets for the selected backend/profile."""

    if args.ai_model:
        return [_make_model_spec(Path(raw), source="user_model") for raw in args.ai_model], "user_model"

    if args.ai_config:
        return [_make_config_spec(Path(raw), source="user_config") for raw in args.ai_config], "user_config"

    frontier_names = FRONTIER_QUICK_AI_MODELS if args.profile == "quick" else FRONTIER_FULL_AI_MODELS
    legacy_names = LEGACY_QUICK_AI_CONFIGS if args.profile == "quick" else LEGACY_FULL_AI_CONFIGS
    frontier_paths = [(args.model_dir / name).expanduser().resolve() for name in frontier_names]
    legacy_paths = [(args.asset_dir / name).expanduser().resolve() for name in legacy_names]

    if args.ai_backend == "picamera2":
        return [_make_model_spec(path, source="default_frontier") for path in frontier_paths], "default_frontier"

    if args.ai_backend == "rpicam":
        return [_make_config_spec(path, source="default_legacy") for path in legacy_paths], "default_legacy"

    available_frontier = [path for path in frontier_paths if path.exists()]
    if _picamera2_module_available() and available_frontier:
        selected = available_frontier[:1] if args.profile == "quick" else available_frontier
        return [_make_model_spec(path, source="auto_frontier") for path in selected], "auto_frontier"

    return [_make_config_spec(path, source="auto_legacy") for path in legacy_paths], "auto_legacy"


def _extract_network_files_from_config(node: Any) -> list[str]:
    """Recursively collect network_file references from a post-process JSON tree."""

    found: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "network_file" and isinstance(child, str):
                    found.append(child)
                walk(child)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(node)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in found:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def _sanitize_for_json(value: Any, *, depth: int = 0) -> Any:
    """Convert arbitrary runtime values into JSON-safe summaries."""

    if depth > 4:
        return str(type(value).__name__)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _sanitize_for_json(child, depth=depth + 1) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        if len(items) > 32:
            return {
                "type": type(value).__name__,
                "len": len(items),
                "head": [_sanitize_for_json(item, depth=depth + 1) for item in items[:8]],
            }
        return [_sanitize_for_json(item, depth=depth + 1) for item in items]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"type": type(value).__name__, "len": len(value)}
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _sanitize_for_json(tolist(), depth=depth + 1)
        except Exception:
            return {"type": type(value).__name__}
    shape = getattr(value, "shape", None)
    if shape is not None:
        return {"type": type(value).__name__, "shape": _sanitize_for_json(shape, depth=depth + 1)}
    return str(value)


def _metadata_entries(metadata: Any) -> tuple[dict[str, Any], ...]:
    """Normalize one rpicam/Picamera2 metadata payload into JSON-object entries."""

    if isinstance(metadata, dict):
        return (metadata,)
    if isinstance(metadata, list):
        return tuple(entry for entry in metadata if isinstance(entry, dict))
    return ()


def _metadata_keys(metadata: Any) -> list[str]:
    """Return the stable union of top-level keys from one metadata payload."""

    keys: set[str] = set()
    for entry in _metadata_entries(metadata):
        keys.update(str(key) for key in entry.keys())
    return sorted(keys)


def _select_ai_metadata_entry(metadata: Any) -> tuple[dict[str, Any] | None, int | None]:
    """Pick one representative metadata entry, preferring frames with IMX500 data."""

    entries = _metadata_entries(metadata)
    if not entries:
        return None, None
    for index in range(len(entries) - 1, -1, -1):
        entry = entries[index]
        if any(key in entry for key in AI_METADATA_KEYS):
            return entry, index
    return entries[-1], len(entries) - 1


def _summarize_ai_metadata(metadata: Any) -> dict[str, Any]:
    """Return a compact IMX500-metadata summary without dumping raw tensors."""

    entries = _metadata_entries(metadata)
    selected_metadata, selected_index = _select_ai_metadata_entry(metadata)
    summary: dict[str, Any] = {
        "metadata_keys": _metadata_keys(metadata),
        "metadata_entry_count": len(entries),
        "selected_metadata_index": selected_index,
        "has_any_cnn_metadata": False,
    }
    if selected_metadata is None:
        return summary

    for key in AI_METADATA_KEYS:
        if key in selected_metadata:
            value = selected_metadata[key]
            entry: dict[str, Any] = {"present": True}
            try:
                entry["len"] = len(value)  # type: ignore[arg-type]
            except Exception:
                pass
            if key.endswith("Info"):
                entry["value"] = _sanitize_for_json(value)
            summary[key] = entry

    kpi = selected_metadata.get("CnnKpiInfo")
    if isinstance(kpi, (list, tuple)) and len(kpi) >= 2:
        try:
            summary["kpi_ms"] = {
                "dnn_runtime_ms": round(float(kpi[0]) / 1000.0, 3),
                "dsp_runtime_ms": round(float(kpi[1]) / 1000.0, 3),
            }
        except (TypeError, ValueError):
            pass

    summary["has_any_cnn_metadata"] = any(
        key in entry
        for entry in entries
        for key in AI_METADATA_KEYS
    )
    return summary


def _discover_cameras(list_output: str) -> list[dict[str, object]]:
    """Parse rpicam camera-list output without regex heuristics."""

    cameras: list[dict[str, object]] = []
    for raw_line in list_output.splitlines():
        line = raw_line.strip()
        if " : " not in line:
            continue
        index_text, description = line.split(" : ", 1)
        if not index_text.isdigit():
            continue
        name = description.split(" [", 1)[0].strip()
        cameras.append(
            {
                "index": int(index_text),
                "name": name,
                "description": description.strip(),
            }
        )
    return cameras


def _looks_like_imx500_camera(camera: dict[str, object]) -> bool:
    """Best-effort check that the selected camera is the AI Camera."""

    description = str(camera.get("description", "")).lower()
    name = str(camera.get("name", "")).lower()
    return "imx500" in description or "imx500" in name


def _collect_environment_payload(ai_specs: list[AISpec]) -> dict[str, Any]:
    """Collect versions and dependency information for support/debugging."""

    payload: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "hostname": socket.gethostname(),
        "required_tools": {},
        "optional_tools": {},
        "picamera2_module_available": _picamera2_module_available(),
        "picamera2_version": _distribution_version("picamera2"),
        "ai_specs": [asdict(spec) for spec in ai_specs],
    }

    for tool in BASE_REQUIRED_TOOLS:
        payload["required_tools"][tool] = shutil.which(tool)
    for tool in OPTIONAL_TOOLS:
        payload["optional_tools"][tool] = shutil.which(tool)

    version_output: dict[str, str] = {}
    rpicam_hello = shutil.which("rpicam-hello")
    if rpicam_hello:
        try:
            completed = subprocess.run(
                [rpicam_hello, "--version"],
                capture_output=True,
                text=True,
                timeout=10.0,
                check=False,
                errors="replace",
            )
            version_output["stdout"] = completed.stdout
            version_output["stderr"] = completed.stderr
            version_output["returncode"] = str(completed.returncode)
        except Exception as exc:  # pragma: no cover - best effort probe
            version_output["error"] = str(exc)
    payload["rpicam_version_probe"] = version_output
    return payload


def _tool_check(run_dir: Path, ai_specs: list[AISpec]) -> StepResult:
    """Verify that the required binaries, firmware, and AI assets exist."""

    missing: list[str] = []
    invalid: list[str] = []
    referenced_network_files: dict[str, list[str]] = {}

    for tool in BASE_REQUIRED_TOOLS:
        if shutil.which(tool) is None:
            missing.append(tool)

    if ai_specs:
        for firmware in DEFAULT_FIRMWARE_FILES:
            if not firmware.exists():
                missing.append(str(firmware))

    for spec in ai_specs:
        spec_path = Path(spec.path)
        if not spec_path.exists():
            missing.append(str(spec_path))
            continue
        if spec.kind == "config":
            try:
                payload = json.loads(spec_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                invalid.append(f"{spec_path}: invalid JSON ({exc})")
                continue
            network_files = _extract_network_files_from_config(payload)
            referenced_network_files[str(spec_path)] = network_files
            for raw_network in network_files:
                network_path = Path(raw_network).expanduser()
                if not network_path.is_absolute():
                    network_path = (spec_path.parent / network_path).resolve()
                if not network_path.exists():
                    missing.append(str(network_path))

    if any(spec.backend == "picamera2" for spec in ai_specs) and not _picamera2_module_available():
        missing.append("picamera2 Python module")

    env_payload = _collect_environment_payload(ai_specs)
    env_payload["referenced_network_files"] = referenced_network_files
    env_payload["required_firmware_files"] = [str(path) for path in DEFAULT_FIRMWARE_FILES]
    payload_path = run_dir / "tool_check.json"
    _atomic_write_json(payload_path, env_payload)

    problems = [*missing, *invalid]
    if problems:
        summary = "preflight failed: " + ", ".join(problems)
        ok = False
    else:
        summary = "required tools, firmware, and AI assets are present"
        ok = True

    return StepResult(
        name="tool_check",
        ok=ok,
        summary=summary,
        artifacts=[str(payload_path.resolve())],
        details={"referenced_network_files": referenced_network_files},
    )


def _print_step(result: StepResult) -> None:
    """Emit a one-line human-readable summary for a phase result."""

    status = "ok" if result.ok else "fail"
    print(f"[{status}] {result.name}: {result.summary} ({result.duration_s:.2f}s)")
    for artifact in result.artifacts:
        print(f"  artifact={artifact}")


class _WallClockTimeoutContext:
    """POSIX wall-clock timeout guard for in-process Picamera2 phases."""

    def __init__(self, seconds: float) -> None:
        self.seconds = max(0.0, seconds)
        self._enabled = bool(self.seconds) and os.name == "posix" and hasattr(signal, "setitimer")
        self._old_handler: Any = None

    def _handler(self, signum: int, frame: Any) -> None:
        raise WallClockTimeout(f"wall-clock timeout after {self.seconds:.1f}s")

    def __enter__(self) -> "_WallClockTimeoutContext":
        if self._enabled:
            self._old_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, self._handler)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Literal[False]:
        if self._enabled:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, self._old_handler)
        return False


def _run_still_phase(args: argparse.Namespace, run_dir: Path) -> StepResult:
    """Run the still capture smoke test."""

    still_path = run_dir / "still.jpg"
    still_meta_path = run_dir / "still_metadata.json"
    result = _run_command(
        name="still_capture",
        command=[
            "rpicam-still",
            "--camera",
            str(args.camera),
            "-n",
            "-t",
            _milliseconds(args.still_timeout),
            "-o",
            str(still_path),
            "--metadata",
            str(still_meta_path),
            "--metadata-format",
            "json",
        ],
        run_dir=run_dir,
        timeout_s=max(30.0, args.still_timeout + 15.0),
        expected_files=[still_path, still_meta_path],
    )
    if result.ok:
        try:
            metadata = json.loads(still_meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            result.ok = False
            result.summary = f"still metadata JSON invalid: {exc}"
        else:
            result.details["metadata_keys"] = _metadata_keys(metadata)
            result.details["metadata_entry_count"] = len(_metadata_entries(metadata))
    return result


def _run_video_phase(args: argparse.Namespace, run_dir: Path) -> StepResult:
    """Run the short video capture smoke test."""

    video_path = run_dir / "video.h264"
    video_pts_path = run_dir / "video.pts"
    video_meta_path = run_dir / "video_metadata.json"
    result = _run_command(
        name="video_capture",
        command=[
            "rpicam-vid",
            "--camera",
            str(args.camera),
            "-n",
            "-t",
            _milliseconds(args.video_timeout),
            "--width",
            str(args.video_width),
            "--height",
            str(args.video_height),
            "--codec",
            "h264",
            "--inline",
            "-o",
            str(video_path),
            "--save-pts",
            str(video_pts_path),
            "--metadata",
            str(video_meta_path),
            "--metadata-format",
            "json",
        ],
        run_dir=run_dir,
        timeout_s=max(45.0, args.video_timeout + 20.0),
        expected_files=[video_path, video_pts_path, video_meta_path],
    )
    if result.ok:
        try:
            metadata = json.loads(video_meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            result.ok = False
            result.summary = f"video metadata JSON invalid: {exc}"
        else:
            result.details["metadata_keys"] = _metadata_keys(metadata)
            result.details["metadata_entry_count"] = len(_metadata_entries(metadata))
    return result


def _run_ai_config_phase(spec: AISpec, args: argparse.Namespace, run_dir: Path) -> StepResult:
    """Run a legacy rpicam JSON post-process AI smoke phase."""

    config_path = Path(spec.path)
    phase_name = f"ai_{spec.name}"
    safe = _safe_name(phase_name)
    video_path = run_dir / f"{safe}.h264"
    pts_path = run_dir / f"{safe}.pts"
    metadata_path = run_dir / f"{safe}_metadata.json"
    legacy_runtime_s = min(args.ai_max_seconds, max(7.0, args.ai_timeout + 5.0))
    result = _run_command(
        name=phase_name,
        command=[
            "rpicam-vid",
            "--camera",
            str(args.camera),
            "-n",
            "-t",
            _milliseconds(legacy_runtime_s),
            "--width",
            str(args.video_width),
            "--height",
            str(args.video_height),
            "--codec",
            "h264",
            "--inline",
            "--post-process-file",
            str(config_path),
            "-o",
            str(video_path),
            "--save-pts",
            str(pts_path),
            "--metadata",
            str(metadata_path),
            "--metadata-format",
            "json",
        ],
        run_dir=run_dir,
        timeout_s=min(args.ai_max_seconds + 15.0, max(30.0, legacy_runtime_s + 10.0)),
        expected_files=[video_path, pts_path, metadata_path],
    )
    if not result.ok:
        return result

    try:
        parsed_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result.ok = False
        result.summary = f"legacy AI metadata JSON invalid: {exc}"
        return result

    result.details["ai_spec"] = asdict(spec)
    result.details["ai_metadata"] = _summarize_ai_metadata(parsed_metadata)
    if result.details["ai_metadata"].get("has_any_cnn_metadata"):
        result.summary = "legacy rpicam AI phase completed and exposed IMX500 metadata"
    else:
        result.summary = "legacy rpicam AI phase completed; metadata captured but no explicit Cnn* fields were exported"
    return result


def _run_ai_model_phase(spec: AISpec, args: argparse.Namespace, run_dir: Path) -> StepResult:
    """Run a frontier Picamera2 + IMX500 model smoke phase."""

    phase_name = f"ai_{spec.name}"
    safe = _safe_name(phase_name)
    stdout_path = run_dir / f"{safe}.stdout.log"
    stderr_path = run_dir / f"{safe}.stderr.log"
    metadata_path = run_dir / f"{safe}_metadata.json"
    for path in [stdout_path, stderr_path, metadata_path]:
        _delete_file_if_exists(path)

    start = time.perf_counter()
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    result = StepResult(name=phase_name, ok=False, summary="AI phase did not run")
    model_path = Path(spec.path)

    def out(message: str) -> None:
        stdout_lines.append(message)

    def err(message: str) -> None:
        stderr_lines.append(message)

    picam2 = None
    try:
        with _WallClockTimeoutContext(args.ai_max_seconds):
            from picamera2 import Picamera2  # pylint: disable=import-error
            from picamera2.devices import IMX500  # pylint: disable=import-error

            out(f"model={model_path}")
            out(f"task={spec.task}")
            out(f"picamera2_version={_distribution_version('picamera2') or 'unknown'}")

            # This must happen before Picamera2(...) according to the official examples.
            imx500 = IMX500(str(model_path))
            helper_camera_num = getattr(imx500, "camera_num", None)
            out(f"helper_camera_num={helper_camera_num}")

            if helper_camera_num != args.camera:
                raise RuntimeError(
                    f"IMX500 helper bound to camera {helper_camera_num}, but --camera requested {args.camera}"
                )

            intrinsics = getattr(imx500, "network_intrinsics", None)
            if intrinsics is not None:
                out(f"network_intrinsics={intrinsics}")

            controls: dict[str, Any] = {}
            inference_rate = getattr(intrinsics, "inference_rate", None) if intrinsics is not None else None
            if inference_rate:
                controls["FrameRate"] = inference_rate

            picam2 = Picamera2(args.camera)
            config = picam2.create_preview_configuration(
                main={"size": (args.video_width, args.video_height)},
                controls=controls,
                buffer_count=12,
            )

            imx500.show_network_fw_progress_bar()
            picam2.start(config, show_preview=False)

            observe_deadline = time.monotonic() + max(0.1, args.ai_timeout)
            frames_seen = 0
            first_metadata_keys: list[str] | None = None
            selected_metadata: dict[str, Any] | None = None
            output_shapes: Any = None
            output_tensor_count: int | None = None
            kpi_info: Any = None

            while time.monotonic() < observe_deadline:
                metadata = picam2.capture_metadata()
                frames_seen += 1
                if first_metadata_keys is None:
                    first_metadata_keys = sorted(str(key) for key in metadata.keys())

                if frames_seen <= args.ai_warmup_frames:
                    continue

                has_any = any(key in metadata for key in AI_METADATA_KEYS)
                if not has_any:
                    continue

                selected_metadata = metadata
                try:
                    output_shapes = IMX500.get_output_shapes(metadata)
                except Exception as exc:
                    err(f"get_output_shapes failed: {exc}")
                try:
                    outputs = imx500.get_outputs(metadata, add_batch=True)
                    if outputs is not None:
                        output_tensor_count = len(outputs)
                except Exception as exc:
                    err(f"get_outputs failed: {exc}")
                try:
                    kpi_info = IMX500.get_kpi_info(metadata)
                except Exception as exc:
                    err(f"get_kpi_info failed: {exc}")
                break

            if selected_metadata is None:
                raise RuntimeError(
                    f"no IMX500 metadata observed within {args.ai_timeout:.1f}s after startup "
                    f"(frames_seen={frames_seen})"
                )

            metadata_summary = _summarize_ai_metadata(selected_metadata)
            if output_shapes is not None:
                metadata_summary["output_shapes"] = _sanitize_for_json(output_shapes)
            if output_tensor_count is not None:
                metadata_summary["output_tensor_count"] = output_tensor_count
            if isinstance(kpi_info, tuple) and len(kpi_info) >= 2:
                try:
                    metadata_summary["kpi_ms_picamera2"] = {
                        "dnn_runtime_ms": round(float(kpi_info[0]), 3),
                        "dsp_runtime_ms": round(float(kpi_info[1]), 3),
                    }
                except (TypeError, ValueError):
                    pass

            metadata_payload = {
                "backend": "picamera2",
                "camera_index": args.camera,
                "helper_camera_num": helper_camera_num,
                "model": str(model_path),
                "task": spec.task,
                "frames_seen": frames_seen,
                "first_metadata_keys": first_metadata_keys,
                "network_intrinsics": str(intrinsics) if intrinsics is not None else None,
                "ai_metadata": metadata_summary,
            }
            _atomic_write_json(metadata_path, metadata_payload)

            result.ok = True
            result.summary = "Picamera2 IMX500 model phase completed and exposed inference metadata"
            result.artifacts = [str(metadata_path.resolve())]
            result.details = {
                "ai_spec": asdict(spec),
                "frames_seen": frames_seen,
                "first_metadata_keys": first_metadata_keys or [],
                "ai_metadata": metadata_summary,
            }
    except WallClockTimeout as exc:
        err(str(exc))
        result.summary = str(exc)
    except Exception as exc:
        err(f"{type(exc).__name__}: {exc}")
        err(traceback.format_exc())
        result.summary = f"{type(exc).__name__}: {exc}"
    finally:
        if picam2 is not None:
            try:
                picam2.stop()
            except Exception as exc:
                err(f"stop failed: {exc}")
            try:
                picam2.close()
            except Exception as exc:
                err(f"close failed: {exc}")
        _atomic_write_text(stdout_path, "\n".join(stdout_lines) + ("\n" if stdout_lines else ""))
        _atomic_write_text(stderr_path, "\n".join(stderr_lines) + ("\n" if stderr_lines else ""))
        result.stdout_log = str(stdout_path.resolve())
        result.stderr_log = str(stderr_path.resolve())
        result.duration_s = time.perf_counter() - start
        if metadata_path.exists() and str(metadata_path.resolve()) not in result.artifacts:
            _chmod_private(metadata_path)
            result.artifacts.append(str(metadata_path.resolve()))
    return result


def _finish(
    args: argparse.Namespace,
    run_dir: Path,
    auto_created: bool,
    ai_specs: list[AISpec],
    ai_resolution: str,
    results: list[StepResult],
    *,
    cameras: list[dict[str, object]] | None = None,
    selected_camera: dict[str, object] | None = None,
) -> int:
    """Write the final summary file and return the overall exit code."""

    overall_ok = all(result.ok for result in results)
    # BREAKING: summary.json now carries ai_specs/ai_models in addition to the
    # older ai_configs field because the default AI path may be model-backed.
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "camera_index": args.camera,
        "profile": args.profile,
        "ai_backend_requested": args.ai_backend,
        "ai_resolution": ai_resolution,
        "run_dir": str(run_dir.resolve()),
        "auto_created_run_dir": auto_created,
        "ai_specs": [asdict(spec) for spec in ai_specs],
        "ai_configs": [spec.path for spec in ai_specs if spec.kind == "config"],
        "ai_models": [spec.path for spec in ai_specs if spec.kind == "model"],
        "cameras": cameras or [],
        "selected_camera": selected_camera,
        "overall_ok": overall_ok,
        "results": [asdict(result) for result in results],
    }
    summary_path = run_dir / "summary.json"
    _atomic_write_json(summary_path, summary)
    print(f"summary={summary_path}")
    print(f"overall_ok={str(overall_ok).lower()}")
    return 0 if overall_ok else 1


def main() -> int:
    """Run the configured Pi AI camera smoke-test phases."""

    parser = build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)

    # SEC-1: ensure child processes create 0600/0700 artifacts from the outset.
    os.umask(0o077)

    try:
        run_dir, auto_created = _create_run_dir(args.output_dir)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    ai_specs, ai_resolution = _resolve_ai_specs(args)
    results: list[StepResult] = []

    print(f"run_dir={run_dir}")
    print(f"profile={args.profile} ai_backend={args.ai_backend} ai_resolution={ai_resolution} camera={args.camera} host={socket.gethostname()}")

    tool_result = _tool_check(run_dir, ai_specs)
    results.append(tool_result)
    _print_step(tool_result)
    if not tool_result.ok:
        return _finish(args, run_dir, auto_created, ai_specs, ai_resolution, results)

    list_result = _run_command(
        name="enumerate_camera",
        command=["rpicam-hello", "--list-cameras"],
        run_dir=run_dir,
        timeout_s=30.0,
    )
    results.append(list_result)
    _print_step(list_result)
    if not list_result.ok:
        return _finish(args, run_dir, auto_created, ai_specs, ai_resolution, results)

    if list_result.stdout_log is None:
        select_result = StepResult(
            name="select_camera",
            ok=False,
            summary="camera enumeration produced no stdout log path",
        )
        results.append(select_result)
        _print_step(select_result)
        return _finish(args, run_dir, auto_created, ai_specs, ai_resolution, results)

    camera_list_log = Path(list_result.stdout_log)
    camera_listing = camera_list_log.read_text(encoding="utf-8")
    cameras = _discover_cameras(camera_listing)
    selected = next((camera for camera in cameras if camera["index"] == args.camera), None)
    if selected is None:
        select_result = StepResult(
            name="select_camera",
            ok=False,
            summary=f"camera index {args.camera} not present in camera list",
            artifacts=[str(camera_list_log.resolve())],
            details={"cameras": cameras},
        )
        results.append(select_result)
        _print_step(select_result)
        return _finish(args, run_dir, auto_created, ai_specs, ai_resolution, results, cameras=cameras)

    select_result = StepResult(
        name="select_camera",
        ok=True,
        summary=f"selected camera {args.camera}: {selected['description']}",
        artifacts=[str(camera_list_log.resolve())],
        details={"selected_camera": selected},
    )
    results.append(select_result)
    _print_step(select_result)

    if ai_specs and not _looks_like_imx500_camera(selected):
        ai_camera_result = StepResult(
            name="validate_ai_camera",
            ok=False,
            summary=f"selected camera {args.camera} is not an IMX500 AI Camera",
            details={"selected_camera": selected, "ai_specs": [asdict(spec) for spec in ai_specs]},
        )
        results.append(ai_camera_result)
        _print_step(ai_camera_result)
        return _finish(
            args,
            run_dir,
            auto_created,
            ai_specs,
            ai_resolution,
            results,
            cameras=cameras,
            selected_camera=selected,
        )

    still_result = _run_still_phase(args, run_dir)
    results.append(still_result)
    _print_step(still_result)
    if args.stop_on_failure and not still_result.ok:
        return _finish(
            args,
            run_dir,
            auto_created,
            ai_specs,
            ai_resolution,
            results,
            cameras=cameras,
            selected_camera=selected,
        )

    video_result = _run_video_phase(args, run_dir)
    results.append(video_result)
    _print_step(video_result)
    if args.stop_on_failure and not video_result.ok:
        return _finish(
            args,
            run_dir,
            auto_created,
            ai_specs,
            ai_resolution,
            results,
            cameras=cameras,
            selected_camera=selected,
        )

    for spec in ai_specs:
        if spec.backend == "picamera2":
            ai_result = _run_ai_model_phase(spec, args, run_dir)
        else:
            ai_result = _run_ai_config_phase(spec, args, run_dir)
        results.append(ai_result)
        _print_step(ai_result)
        if args.stop_on_failure and not ai_result.ok:
            break

    return _finish(
        args,
        run_dir,
        auto_created,
        ai_specs,
        ai_resolution,
        results,
        cameras=cameras,
        selected_camera=selected,
    )


if __name__ == "__main__":
    raise SystemExit(main())
