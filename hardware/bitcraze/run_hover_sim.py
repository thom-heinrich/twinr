#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""Run the bounded Twinr hover worker against a local CrazySim instance.

Purpose
-------
Launch one validated CrazySim single-agent session and drive the existing
``run_hover_test.py`` worker against the simulator over CFLib UDP transport.
This keeps one hover implementation while giving Twinr a bounded, reproducible
SITL path for takeoff/stabilize/abort work.

Usage
-----
::

    ./.venv/bin/python hardware/bitcraze/run_hover_sim.py \
      --crazysim-root /tmp/crazysim \
      --backend mujoco \
      -- --height-m 0.25 --hover-duration-s 1.0 --json

Notes
-----
The simulator runner owns the worker URI, workspace, preflight battery gates,
deck requirements, and on-device failsafe mode. Pass-through hover arguments
must not override those fields.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Literal, Sequence, cast
from urllib.parse import urlparse

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from twinr.hardware.crazysim_adapter import (  # noqa: E402
    CrazySimSingleAgentConfig,
    CrazySimSingleAgentLauncher,
    resolve_crazysim_checkout,
)


_FORBIDDEN_HOVER_ARGS = frozenset(
    {
        "--uri",
        "--workspace",
        "--min-vbat-v",
        "--min-battery-level",
        "--min-clearance-m",
        "--on-device-failsafe-mode",
        "--require-deck",
        "--json",
        "--trace-file",
        "--runtime-mode",
    }
)


@dataclass(frozen=True, slots=True)
class HoverWorkerRunResult:
    """Capture one bounded hover-worker subprocess result."""

    payload: dict[str, object]
    returncode: int


def _default_python_bin() -> Path:
    """Return the canonical Twinr repo interpreter for hover SITL runs."""

    return _REPO_ROOT / ".venv" / "bin" / "python"


def _normalize_hover_args(raw_args: Sequence[str]) -> tuple[str, ...]:
    """Normalize pass-through hover args and block conflicting overrides."""

    normalized = tuple(str(item) for item in raw_args if str(item).strip())
    if normalized and normalized[0] == "--":
        normalized = normalized[1:]
    conflicts = tuple(argument for argument in normalized if argument in _FORBIDDEN_HOVER_ARGS)
    if conflicts:
        conflict_text = ", ".join(sorted(dict.fromkeys(conflicts)))
        raise ValueError(
            "run_hover_sim.py owns the following hover-worker flags and they must not be "
            f"passed through: {conflict_text}"
        )
    return normalized


def _resolve_trace_file(path: Path | None) -> Path:
    """Return one concrete trace path for the bounded hover worker."""

    if path is not None:
        return path.expanduser().resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("/tmp") / f"crazysim_hover_trace_{timestamp}.jsonl"


def _resolve_disturbance_runtime_file(trace_file: Path) -> Path:
    """Return the deterministic disturbance-runtime log path for one worker trace."""

    return trace_file.with_name(f"{trace_file.stem}_disturbance_runtime.jsonl")


def _build_hover_worker_command(
    *,
    python_bin: Path,
    uri: str,
    workspace: Path,
    trace_file: Path,
    hover_args: Sequence[str],
) -> tuple[str, ...]:
    """Build the exact bounded hover-worker command for CrazySim SITL."""

    return (
        str(python_bin),
        str(_SCRIPT_DIR / "run_hover_test.py"),
        "--uri",
        str(uri),
        "--workspace",
        str(workspace),
        "--min-vbat-v",
        "0",
        "--min-battery-level",
        "0",
        "--min-clearance-m",
        "0",
        "--on-device-failsafe-mode",
        "off",
        "--runtime-mode",
        "sitl",
        "--trace-file",
        str(trace_file),
        "--json",
        *tuple(hover_args),
    )


def _validate_python_bin(path: Path) -> Path:
    """Fail closed if the chosen Python interpreter is unavailable or lacks CFLib."""

    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).absolute()
    if not candidate.exists():
        raise FileNotFoundError(f"python interpreter does not exist: {candidate}")
    if not candidate.is_file():
        raise FileNotFoundError(f"python interpreter is not a file: {candidate}")
    probe = subprocess.run(
        (str(candidate), "-c", "import cflib.crtp"),
        cwd=str(_REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        stderr_text = probe.stderr.strip() or probe.stdout.strip()
        raise RuntimeError(
            f"python interpreter `{candidate}` cannot import cflib.crtp: {stderr_text or 'unknown import failure'}"
        )
    sitl_probe = subprocess.run(
        (
            str(candidate),
            "-c",
            (
                "import inspect; "
                "from cflib.crtp.udpdriver import UdpDriver; "
                "source = inspect.getsource(UdpDriver.scan_interface); "
                "raise SystemExit(0 if 'return []' not in source else 2)"
            ),
        ),
        cwd=str(_REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    if sitl_probe.returncode != 0:
        raise RuntimeError(
            "python interpreter "
            f"`{candidate}` exposes a non-SITL UDP driver in cflib. "
            "CrazySim requires the official crazyflie-lib-python source build "
            "with the rewritten UDP driver described in the CrazySim README "
            "(commit 99ad0e3 or newer)."
        )
    return candidate


def _parse_udp_uri(uri: str) -> tuple[str, int]:
    """Parse one CrazySim UDP URI fail-closed."""

    parsed = urlparse(str(uri))
    if parsed.scheme != "udp":
        raise ValueError(f"CrazySim URI must use the udp:// scheme; got {uri!r}")
    if not parsed.hostname or parsed.port is None:
        raise ValueError(f"CrazySim URI must include host and port; got {uri!r}")
    return (str(parsed.hostname), int(parsed.port))


def _probe_udp_uri_ready(uri: str, *, probe_timeout_s: float) -> bool:
    """Probe one CrazySim UDP URI using the official cflib scan contract.

    The upstream CrazySim/cflib SITL lane relies on UdpDriver.scan_interface(),
    which sends a null CRTP probe byte (0xFF) and expects any reply on the
    CFLib UDP port. Reusing that exact wire contract gives Twinr a deterministic
    readiness signal instead of guessing with fixed sleeps.
    """

    host, port = _parse_udp_uri(uri)
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.settimeout(max(0.01, float(probe_timeout_s)))
    try:
        probe.connect((host, port))
        probe.send(b"\xFF")
        response = probe.recv(1024)
    except (OSError, socket.timeout):
        return False
    finally:
        probe.close()
    return bool(response)


def _wait_for_udp_uri_ready(
    uri: str,
    *,
    readiness_timeout_s: float,
    probe_timeout_s: float,
    probe_period_s: float,
) -> tuple[int, float]:
    """Wait until the CrazySim UDP endpoint answers the official readiness probe."""

    deadline_s = time.monotonic() + max(0.0, float(readiness_timeout_s))
    attempts = 0
    started_s = time.monotonic()
    while True:
        attempts += 1
        if _probe_udp_uri_ready(uri, probe_timeout_s=probe_timeout_s):
            return (attempts, time.monotonic() - started_s)
        if time.monotonic() >= deadline_s:
            raise RuntimeError(
                "CrazySim URI "
                f"`{uri}` did not answer the official UDP readiness probe within "
                f"{float(readiness_timeout_s):.1f} s after launch"
            )
        time.sleep(max(0.01, float(probe_period_s)))


def _parse_cflib_ready_payload(*, stdout_text: str, stderr_text: str) -> dict[str, object]:
    """Parse one bounded CFLib readiness-probe payload fail-closed."""

    if not stdout_text:
        raise RuntimeError(
            "CrazySim CFLib readiness probe produced no JSON payload"
            + ("" if not stderr_text else f": stderr={stderr_text!r}")
        )
    try:
        payload = json.loads(stdout_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "CrazySim CFLib readiness probe did not emit valid JSON: "
            f"stdout={stdout_text!r} stderr={stderr_text!r}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError("CrazySim CFLib readiness probe JSON payload must be an object")
    return payload


def _build_cflib_ready_probe_command(
    *,
    python_bin: Path,
    uri: str,
    workspace: Path,
) -> tuple[str, ...]:
    """Build one bounded subprocess command that proves CFLib can really connect."""

    probe_program = """
from __future__ import annotations
import json
from pathlib import Path
import sys

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

uri = sys.argv[1]
workspace = Path(sys.argv[2]).expanduser().resolve()
workspace.mkdir(parents=True, exist_ok=True)
cache_dir = workspace / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)

cflib.crtp.init_drivers()
cf = Crazyflie(rw_cache=str(cache_dir))
with SyncCrazyflie(uri, cf=cf):
    pass

print(
    json.dumps(
        {
            "uri": uri,
            "workspace": str(workspace),
            "cache_dir": str(cache_dir),
            "connected": True,
        },
        sort_keys=True,
    )
)
""".strip()
    return (
        str(python_bin),
        "-c",
        probe_program,
        str(uri),
        str(workspace),
    )


def _wait_for_cflib_uri_ready(
    *,
    python_bin: Path,
    uri: str,
    workspace: Path,
    readiness_timeout_s: float,
    probe_timeout_s: float,
    probe_period_s: float,
) -> tuple[int, float, dict[str, object]]:
    """Wait until a real bounded CFLib SyncCrazyflie connect succeeds."""

    deadline_s = time.monotonic() + max(0.0, float(readiness_timeout_s))
    attempts = 0
    started_s = time.monotonic()
    last_error_text = "unknown CFLib readiness failure"
    while True:
        attempts += 1
        command = _build_cflib_ready_probe_command(
            python_bin=python_bin,
            uri=uri,
            workspace=workspace,
        )
        try:
            completed = subprocess.run(
                command,
                cwd=str(_REPO_ROOT),
                check=False,
                capture_output=True,
                text=True,
                timeout=max(0.1, float(probe_timeout_s)),
            )
        except subprocess.TimeoutExpired:
            last_error_text = (
                "CrazySim CFLib readiness probe exceeded the bounded timeout of "
                f"{float(probe_timeout_s):.1f} s"
            )
        else:
            stdout_text = completed.stdout.strip()
            stderr_text = completed.stderr.strip()
            if completed.returncode == 0:
                payload = _parse_cflib_ready_payload(
                    stdout_text=stdout_text,
                    stderr_text=stderr_text,
                )
                return (attempts, time.monotonic() - started_s, payload)
            error_text = stderr_text or stdout_text or f"returncode={completed.returncode}"
            last_error_text = (
                "CrazySim CFLib readiness probe failed with returncode "
                f"{completed.returncode}: {error_text}"
            )
        if time.monotonic() >= deadline_s:
            raise RuntimeError(
                "CrazySim URI "
                f"`{uri}` never completed a bounded CFLib SyncCrazyflie connect within "
                f"{float(readiness_timeout_s):.1f} s after launch; last_error={last_error_text}"
            )
        time.sleep(max(0.01, float(probe_period_s)))


def _launcher_process_output(process: subprocess.Popen[str] | None) -> tuple[str, str]:
    """Drain launcher stdout/stderr after a bounded stop."""

    if process is None:
        return ("", "")
    try:
        stdout_text, stderr_text = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        return ("", "")
    return (stdout_text.strip(), stderr_text.strip())


def _parse_hover_worker_payload(*, stdout_text: str, stderr_text: str) -> dict[str, object]:
    """Parse the bounded hover-worker JSON payload fail-closed."""

    if not stdout_text:
        raise RuntimeError("run_hover_test.py produced no JSON report on stdout")
    try:
        payload = json.loads(stdout_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"run_hover_test.py did not emit valid JSON: stdout={stdout_text!r} stderr={stderr_text!r}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError("run_hover_test.py JSON payload must be an object")
    return payload


def _run_hover_worker(command: Sequence[str], *, timeout_s: float) -> HoverWorkerRunResult:
    """Run the real hover worker as a bounded subprocess and retain structured failure output."""

    try:
        completed = subprocess.run(
            tuple(command),
            cwd=str(_REPO_ROOT),
            check=False,
            capture_output=True,
            text=True,
            timeout=float(timeout_s),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"run_hover_test.py exceeded the bounded timeout of {float(timeout_s):.1f} s"
        ) from exc
    stdout_text = completed.stdout.strip()
    stderr_text = completed.stderr.strip()
    payload = _parse_hover_worker_payload(
        stdout_text=stdout_text,
        stderr_text=stderr_text,
    )
    return HoverWorkerRunResult(
        payload=payload,
        returncode=int(completed.returncode),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crazysim-root", type=Path, required=True, help="Path to an existing CrazySim checkout.")
    parser.add_argument("--backend", choices=("gazebo", "mujoco"), default="mujoco", help="CrazySim backend to launch.")
    parser.add_argument("--model", default="cf2x_T350", help="CrazySim model name for the selected backend.")
    parser.add_argument("--x-m", type=float, default=0.0, help="Initial X spawn position in meters.")
    parser.add_argument("--y-m", type=float, default=0.0, help="Initial Y spawn position in meters.")
    parser.add_argument("--startup-settle-s", type=float, default=3.0, help="Seconds to wait for CrazySim startup before launching the hover worker.")
    parser.add_argument("--uri-ready-timeout-s", type=float, default=10.0, help="Maximum seconds to wait for the official CrazySim UDP readiness probe after launcher startup (default: 10.0).")
    parser.add_argument("--uri-ready-probe-timeout-s", type=float, default=0.2, help="Per-probe UDP response timeout in seconds for CrazySim readiness checks (default: 0.2).")
    parser.add_argument("--uri-ready-probe-period-s", type=float, default=0.1, help="Delay between CrazySim UDP readiness probes in seconds (default: 0.1).")
    parser.add_argument("--cflib-ready-timeout-s", type=float, default=20.0, help="Maximum seconds to wait for a real bounded CFLib SyncCrazyflie connect after CrazySim launch (default: 20.0).")
    parser.add_argument("--cflib-ready-probe-timeout-s", type=float, default=5.0, help="Per-attempt timeout in seconds for the bounded CFLib readiness probe (default: 5.0).")
    parser.add_argument("--cflib-ready-probe-period-s", type=float, default=0.2, help="Delay between bounded CFLib readiness probes in seconds (default: 0.2).")
    parser.add_argument("--hover-timeout-s", type=float, default=30.0, help="Hard timeout for the bounded hover worker subprocess.")
    parser.add_argument("--workspace", type=Path, default=Path("/tmp/twinr-crazysim-workspace"), help="Workspace root for cflib cache files during SITL.")
    parser.add_argument("--trace-file", type=Path, default=None, help="Optional worker trace JSONL path. Defaults to /tmp/crazysim_hover_trace_<timestamp>.jsonl.")
    parser.add_argument("--python-bin", type=Path, default=_default_python_bin(), help="Python interpreter used for run_hover_test.py. Must import cflib.")
    parser.add_argument(
        "--display",
        default=":0",
        help="DISPLAY value exported to CrazySim when --visualize is enabled.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Launch the MuJoCo passive viewer. Acceptance stays headless by default.",
    )
    parser.add_argument("--disturbance-spec-json", type=Path, default=None, help="Optional physical disturbance schedule JSON injected into the MuJoCo plant.")
    parser.add_argument("--wind-speed-mps", type=float, default=0.0, help="Optional CrazySim constant wind speed in m/s.")
    parser.add_argument("--wind-direction-deg", type=float, default=0.0, help="Wind direction in degrees for CrazySim's built-in wind model.")
    parser.add_argument("--gust-intensity-mps", type=float, default=0.0, help="Optional CrazySim gust peak deviation in m/s.")
    parser.add_argument("--turbulence", choices=("none", "light", "moderate", "severe"), default="none", help="CrazySim Dryden turbulence level.")
    parser.add_argument("--mass-kg", type=float, default=None, help="Optional CrazySim mass override in kg.")
    parser.add_argument("--sensor-noise", action="store_true", help="Enable CrazySim sensor noise.")
    parser.add_argument("--ground-effect", action="store_true", help="Enable CrazySim ground effect.")
    parser.add_argument("--flowdeck", action="store_true", help="Enable CrazySim flowdeck simulation.")
    parser.add_argument("--downwash", action="store_true", help="Enable CrazySim downwash model.")
    parser.add_argument("--json", action="store_true", help="Emit the wrapper result as JSON.")
    parser.add_argument("hover_args", nargs=argparse.REMAINDER, help="Additional run_hover_test.py arguments, passed after `--`.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    hover_args = _normalize_hover_args(args.hover_args)
    python_bin = _validate_python_bin(Path(args.python_bin))
    trace_file = _resolve_trace_file(args.trace_file)
    disturbance_runtime_jsonl = (
        None
        if args.disturbance_spec_json is None
        else _resolve_disturbance_runtime_file(trace_file)
    )
    checkout = resolve_crazysim_checkout(Path(args.crazysim_root))
    turbulence = cast(
        Literal["none", "light", "moderate", "severe"],
        args.turbulence,
    )
    launcher = CrazySimSingleAgentLauncher(
        CrazySimSingleAgentConfig(
            checkout=checkout,
            backend=args.backend,
            model=str(args.model),
            x_m=float(args.x_m),
            y_m=float(args.y_m),
            startup_settle_s=float(args.startup_settle_s),
            extra_env={} if not bool(args.visualize) else {"DISPLAY": str(args.display)},
            visualize=bool(args.visualize),
            disturbance_spec_json=None if args.disturbance_spec_json is None else Path(args.disturbance_spec_json),
            phase_trace_jsonl=trace_file if args.disturbance_spec_json is not None else None,
            disturbance_runtime_jsonl=disturbance_runtime_jsonl,
            wind_speed_mps=float(args.wind_speed_mps),
            wind_direction_deg=float(args.wind_direction_deg),
            gust_intensity_mps=float(args.gust_intensity_mps),
            turbulence=turbulence,
            mass_kg=None if args.mass_kg is None else float(args.mass_kg),
            sensor_noise=bool(args.sensor_noise),
            ground_effect=bool(args.ground_effect),
            flowdeck=bool(args.flowdeck),
            downwash=bool(args.downwash),
        )
    )
    launcher.start()
    readiness_attempts = 0
    readiness_elapsed_s = 0.0
    cflib_readiness_attempts = 0
    cflib_readiness_elapsed_s = 0.0
    cflib_ready_payload: dict[str, object] | None = None
    workspace = Path(args.workspace).expanduser().resolve()
    try:
        try:
            readiness_attempts, readiness_elapsed_s = _wait_for_udp_uri_ready(
                launcher.uri,
                readiness_timeout_s=float(args.uri_ready_timeout_s),
                probe_timeout_s=float(args.uri_ready_probe_timeout_s),
                probe_period_s=float(args.uri_ready_probe_period_s),
            )
        except RuntimeError as exc:
            launcher_process = launcher.process
            launcher.stop()
            stdout_text, stderr_text = _launcher_process_output(launcher_process)
            raise RuntimeError(
                str(exc)
                + f"; launcher_stdout={stdout_text!r} launcher_stderr={stderr_text!r}"
            ) from exc
        try:
            (
                cflib_readiness_attempts,
                cflib_readiness_elapsed_s,
                cflib_ready_payload,
            ) = _wait_for_cflib_uri_ready(
                python_bin=python_bin,
                uri=launcher.uri,
                workspace=workspace,
                readiness_timeout_s=float(args.cflib_ready_timeout_s),
                probe_timeout_s=float(args.cflib_ready_probe_timeout_s),
                probe_period_s=float(args.cflib_ready_probe_period_s),
            )
        except RuntimeError as exc:
            launcher_process = launcher.process
            launcher.stop()
            stdout_text, stderr_text = _launcher_process_output(launcher_process)
            raise RuntimeError(
                str(exc)
                + f"; launcher_stdout={stdout_text!r} launcher_stderr={stderr_text!r}"
            ) from exc
        hover_command = _build_hover_worker_command(
            python_bin=python_bin,
            uri=launcher.uri,
            workspace=workspace,
            trace_file=trace_file,
            hover_args=hover_args,
        )
        hover_result = _run_hover_worker(hover_command, timeout_s=float(args.hover_timeout_s))
    finally:
        launcher.stop()

    payload = {
        "sim": {
            "checkout_root": str(checkout.root),
            "backend": args.backend,
            "model": str(args.model),
            "uri": launcher.uri,
            "workspace": str(Path(args.workspace).expanduser().resolve()),
            "trace_file": str(trace_file),
            "uri_ready_timeout_s": float(args.uri_ready_timeout_s),
            "uri_ready_probe_timeout_s": float(args.uri_ready_probe_timeout_s),
            "uri_ready_probe_period_s": float(args.uri_ready_probe_period_s),
            "uri_ready_attempts": int(readiness_attempts),
            "uri_ready_elapsed_s": float(readiness_elapsed_s),
            "cflib_ready_timeout_s": float(args.cflib_ready_timeout_s),
            "cflib_ready_probe_timeout_s": float(args.cflib_ready_probe_timeout_s),
            "cflib_ready_probe_period_s": float(args.cflib_ready_probe_period_s),
            "cflib_ready_attempts": int(cflib_readiness_attempts),
            "cflib_ready_elapsed_s": float(cflib_readiness_elapsed_s),
            "cflib_ready_payload": cflib_ready_payload,
            "hover_timeout_s": float(args.hover_timeout_s),
            "disturbance_spec_json": (
                None
                if args.disturbance_spec_json is None
                else str(Path(args.disturbance_spec_json).expanduser().resolve())
            ),
            "disturbance_runtime_jsonl": (
                None
                if disturbance_runtime_jsonl is None
                else str(Path(disturbance_runtime_jsonl).expanduser().resolve())
            ),
            "wind_speed_mps": float(args.wind_speed_mps),
            "wind_direction_deg": float(args.wind_direction_deg),
            "gust_intensity_mps": float(args.gust_intensity_mps),
            "turbulence": str(args.turbulence),
            "mass_kg": None if args.mass_kg is None else float(args.mass_kg),
        },
        "hover_worker_returncode": int(hover_result.returncode),
        "hover_worker_command": tuple(hover_command),
        "hover_report": hover_result.payload,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        report_status = hover_result.payload.get("status", "unknown")
        report_outcome = hover_result.payload.get("outcome_class", "unknown")
        print(f"sim.backend={args.backend}")
        print(f"sim.model={args.model}")
        print(f"sim.uri={launcher.uri}")
        print(f"sim.trace_file={trace_file}")
        print(f"hover.returncode={hover_result.returncode}")
        print(f"hover.status={report_status}")
        print(f"hover.outcome_class={report_outcome}")
    return int(hover_result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
