#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///

"""Run bounded smoke tests for the Raspberry Pi AI Camera.

Purpose
-------
Verify that the Raspberry Pi AI Camera is detectable, can capture stills,
can record a short video, and can initialize one or more IMX500 AI
post-processing pipelines without starting the full Twinr runtime.

Usage
-----
Command-line invocation examples::

    python hardware/piaicam/smoke_piaicam.py
    python hardware/piaicam/smoke_piaicam.py --profile quick
    python hardware/piaicam/smoke_piaicam.py --ai-config /usr/share/rpi-camera-assets/imx500_posenet.json

Inputs
------
- ``--camera`` camera index to test (default: 0)
- ``--profile`` whether to run the quick or full test plan
- ``--ai-config`` optional repeated override for AI post-process configs
- ``--output-dir`` optional explicit run directory for artifacts and logs

Outputs
-------
- Creates one run directory under ``state/piaicam/runs/`` by default
- Writes per-phase stdout/stderr logs plus still/video artifacts
- Writes ``summary.json`` with the full result set
- Exit code 0 when every enabled phase succeeds, 1 otherwise

Notes
-----
The default ``full`` profile is bounded but can still take longer than the
pure still/video phases because IMX500 network firmware uploads may take time
on first use.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import socket
import subprocess
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_ROOT = REPO_ROOT / "state" / "piaicam" / "runs"
DEFAULT_ASSET_DIR = Path("/usr/share/rpi-camera-assets")
DEFAULT_FULL_AI_CONFIGS = (
    "imx500_mobilenet_ssd.json",
    "imx500_posenet.json",
)
DEFAULT_QUICK_AI_CONFIGS = DEFAULT_FULL_AI_CONFIGS[:1]
REQUIRED_TOOLS = (
    "rpicam-hello",
    "rpicam-still",
    "rpicam-vid",
    "imx500-package",
)


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


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the Pi AI camera smoke test."""

    parser = argparse.ArgumentParser(description="Run bounded Pi AI Camera smoke tests")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to test")
    parser.add_argument(
        "--profile",
        choices=("quick", "full"),
        default="full",
        help="Quick runs one AI config; full runs all bundled AI configs",
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
        help="Seconds to run each AI preview phase after startup",
    )
    parser.add_argument(
        "--ai-max-seconds",
        type=float,
        default=300.0,
        help="Wall-clock timeout per AI phase, including IMX500 firmware upload",
    )
    parser.add_argument(
        "--ai-config",
        action="append",
        default=[],
        help="Optional repeated AI post-process JSON override",
    )
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=DEFAULT_ASSET_DIR,
        help="Directory holding built-in AI post-process JSON files",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop after the first non-fatal phase failure",
    )
    return parser


def _timestamp_slug() -> str:
    """Return a UTC timestamp suitable for run directory names."""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _milliseconds(seconds: float) -> str:
    """Convert CLI second values into rpicam millisecond arguments."""

    return f"{max(1, round(seconds * 1000))}ms"


def _safe_name(raw: str) -> str:
    """Map arbitrary labels to filesystem-safe log file stems."""

    cleaned = []
    for char in raw.lower():
        if char.isalnum():
            cleaned.append(char)
        else:
            cleaned.append("_")
    while "__" in "".join(cleaned):
        cleaned = list("".join(cleaned).replace("__", "_"))
    return "".join(cleaned).strip("_") or "phase"


def _create_run_dir(requested: Path | None) -> tuple[Path, bool]:
    """Create the run directory and report whether it was auto-generated."""

    if requested is not None:
        requested.mkdir(parents=True, exist_ok=True)
        return requested.resolve(), False

    run_dir = DEFAULT_RUNS_ROOT / _timestamp_slug()
    run_dir.mkdir(parents=True, exist_ok=False)
    latest_link = DEFAULT_RUNS_ROOT / "latest"
    try:
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(run_dir.name)
    except OSError:
        pass
    return run_dir, True


def _write_text(path: Path, content: str) -> None:
    """Persist a UTF-8 text file for phase logging."""

    path.write_text(content, encoding="utf-8")


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
    safe_name = _safe_name(name)
    stdout_path = run_dir / f"{safe_name}.stdout.log"
    stderr_path = run_dir / f"{safe_name}.stderr.log"

    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        returncode = completed.returncode
        ok = returncode == 0
        summary = "command completed successfully" if ok else f"command exited with return code {returncode}"
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        returncode = None
        ok = False
        summary = f"command timed out after {timeout_s:.1f}s"

    _write_text(stdout_path, stdout)
    _write_text(stderr_path, stderr)

    missing_files = [path for path in expected_files if not path.exists()]
    if ok and missing_files:
        ok = False
        summary = "expected artifacts missing: " + ", ".join(str(path.name) for path in missing_files)

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
    )


def _resolve_ai_configs(args: argparse.Namespace) -> list[Path]:
    """Resolve the requested AI post-process configs for the selected profile."""

    if args.ai_config:
        return [Path(raw).expanduser().resolve() for raw in args.ai_config]

    config_names = DEFAULT_QUICK_AI_CONFIGS if args.profile == "quick" else DEFAULT_FULL_AI_CONFIGS
    return [(args.asset_dir / name).resolve() for name in config_names]


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


def _tool_check(run_dir: Path, ai_configs: list[Path]) -> StepResult:
    """Verify that the required binaries and built-in AI configs exist."""

    missing: list[str] = []
    found_tools: dict[str, str] = {}
    for tool in REQUIRED_TOOLS:
        path = shutil.which(tool)
        if path is None:
            missing.append(tool)
        else:
            found_tools[tool] = path

    missing_configs = [str(path) for path in ai_configs if not path.exists()]
    if missing_configs:
        missing.extend(missing_configs)

    payload = {
        "required_tools": found_tools,
        "ai_configs": [str(path) for path in ai_configs],
    }
    payload_path = run_dir / "tool_check.json"
    payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    if missing:
        summary = "missing requirements: " + ", ".join(missing)
        ok = False
    else:
        summary = "required Pi AI camera tools and AI configs are present"
        ok = True

    return StepResult(
        name="tool_check",
        ok=ok,
        summary=summary,
        artifacts=[str(payload_path.resolve())],
    )


def _print_step(result: StepResult) -> None:
    """Emit a one-line human-readable summary for a phase result."""

    status = "ok" if result.ok else "fail"
    print(f"[{status}] {result.name}: {result.summary} ({result.duration_s:.2f}s)")
    for artifact in result.artifacts:
        print(f"  artifact={artifact}")


def main() -> int:
    """Run the configured Pi AI camera smoke-test phases."""

    args = build_parser().parse_args()
    run_dir, auto_created = _create_run_dir(args.output_dir)
    ai_configs = _resolve_ai_configs(args)
    results: list[StepResult] = []

    print(f"run_dir={run_dir}")
    print(f"profile={args.profile} camera={args.camera} host={socket.gethostname()}")

    tool_result = _tool_check(run_dir, ai_configs)
    results.append(tool_result)
    _print_step(tool_result)
    if not tool_result.ok:
        return _finish(args, run_dir, auto_created, ai_configs, results)

    list_result = _run_command(
        name="enumerate_camera",
        command=["rpicam-hello", "--list-cameras"],
        run_dir=run_dir,
        timeout_s=30.0,
    )
    results.append(list_result)
    _print_step(list_result)
    if not list_result.ok:
        return _finish(args, run_dir, auto_created, ai_configs, results)

    camera_listing = Path(list_result.stdout_log).read_text(encoding="utf-8")
    cameras = _discover_cameras(camera_listing)
    selected = next((camera for camera in cameras if camera["index"] == args.camera), None)
    if selected is None:
        select_result = StepResult(
            name="select_camera",
            ok=False,
            summary=f"camera index {args.camera} not present in camera list",
            artifacts=[str(Path(list_result.stdout_log).resolve())],
        )
        results.append(select_result)
        _print_step(select_result)
        return _finish(args, run_dir, auto_created, ai_configs, results, cameras=cameras)

    select_result = StepResult(
        name="select_camera",
        ok=True,
        summary=f"selected camera {args.camera}: {selected['description']}",
        artifacts=[str(Path(list_result.stdout_log).resolve())],
    )
    results.append(select_result)
    _print_step(select_result)

    still_path = run_dir / "still.jpg"
    still_meta_path = run_dir / "still_metadata.json"
    still_result = _run_command(
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
        expected_files=[still_path],
    )
    if still_meta_path.exists():
        still_result.artifacts.append(str(still_meta_path.resolve()))
    results.append(still_result)
    _print_step(still_result)
    if args.stop_on_failure and not still_result.ok:
        return _finish(args, run_dir, auto_created, ai_configs, results, cameras=cameras)

    video_path = run_dir / "video.h264"
    video_pts_path = run_dir / "video.pts"
    video_meta_path = run_dir / "video_metadata.json"
    video_result = _run_command(
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
        expected_files=[video_path, video_pts_path],
    )
    if video_meta_path.exists():
        video_result.artifacts.append(str(video_meta_path.resolve()))
    results.append(video_result)
    _print_step(video_result)
    if args.stop_on_failure and not video_result.ok:
        return _finish(args, run_dir, auto_created, ai_configs, results, cameras=cameras)

    for config_path in ai_configs:
        phase_name = f"ai_{config_path.stem}"
        ai_meta_path = run_dir / f"{_safe_name(phase_name)}_metadata.json"
        ai_result = _run_command(
            name=phase_name,
            command=[
                "rpicam-hello",
                "--camera",
                str(args.camera),
                "-n",
                "-t",
                _milliseconds(args.ai_timeout),
                "--post-process-file",
                str(config_path),
                "--metadata",
                str(ai_meta_path),
                "--metadata-format",
                "json",
            ],
            run_dir=run_dir,
            timeout_s=args.ai_max_seconds,
        )
        if ai_meta_path.exists():
            ai_result.artifacts.append(str(ai_meta_path.resolve()))
        results.append(ai_result)
        _print_step(ai_result)
        if args.stop_on_failure and not ai_result.ok:
            break

    return _finish(args, run_dir, auto_created, ai_configs, results, cameras=cameras)


def _finish(
    args: argparse.Namespace,
    run_dir: Path,
    auto_created: bool,
    ai_configs: list[Path],
    results: list[StepResult],
    *,
    cameras: list[dict[str, object]] | None = None,
) -> int:
    """Write the final summary file and return the overall exit code."""

    overall_ok = all(result.ok for result in results)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "camera_index": args.camera,
        "profile": args.profile,
        "run_dir": str(run_dir.resolve()),
        "auto_created_run_dir": auto_created,
        "ai_configs": [str(path) for path in ai_configs],
        "cameras": cameras or [],
        "overall_ok": overall_ok,
        "results": [asdict(result) for result in results],
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"summary={summary_path}")
    print(f"overall_ok={str(overall_ok).lower()}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
