#!/usr/bin/env python3
"""Run deterministic CrazySim hover scenarios through Twinr's replay lane.

This script either:
1. launches one real bounded CrazySim hover run and captures its baseline
   report, or
2. reuses an existing stored hover report.

It then mutates that baseline into deterministic scenarios such as drift bias,
flow dropout, z-range outliers, and close obstacle proximity, and replays each
scenario through the same bounded hover primitive used elsewhere in Twinr.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Sequence

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from hover_sim_scenarios import (  # noqa: E402
    apply_hover_sim_scenario,
    hover_sim_scenario_names,
    run_hover_sim_scenario_suite,
    serialize_hover_sim_scenario_result,
    write_hover_replay_artifact_json,
)
from twinr.hardware.crazyflie_hover_replay import load_hover_replay_artifact  # noqa: E402


def _default_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("/tmp") / f"twinr-crazysim-scenarios-{timestamp}"


def _normalize_nonresolving_path(path: Path) -> Path:
    """Return an absolute path without resolving a venv symlink target."""

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (Path.cwd() / candidate).absolute()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--crazysim-root", type=Path, default=None, help="Path to a CrazySim checkout used to generate one real baseline hover report.")
    source_group.add_argument("--baseline-report-json", type=Path, default=None, help="Path to an existing hover report JSON used as the baseline artifact.")
    parser.add_argument("--baseline-trace-file", type=Path, default=None, help="Optional phase trace for an existing baseline report.")
    parser.add_argument("--backend", choices=("gazebo", "mujoco"), default="mujoco", help="CrazySim backend used when generating a new baseline report.")
    parser.add_argument("--model", default="cf2x_T350", help="CrazySim model name used when generating a new baseline report.")
    parser.add_argument("--x-m", type=float, default=0.0, help="Initial X spawn position in meters when generating a new baseline report.")
    parser.add_argument("--y-m", type=float, default=0.0, help="Initial Y spawn position in meters when generating a new baseline report.")
    parser.add_argument("--startup-settle-s", type=float, default=3.0, help="Seconds to wait for CrazySim startup before launching the hover worker.")
    parser.add_argument("--hover-timeout-s", type=float, default=30.0, help="Hard timeout for the bounded hover worker subprocess.")
    parser.add_argument("--workspace", type=Path, default=Path("/tmp/twinr-crazysim-scenarios-workspace"), help="Workspace root for cflib cache files during SITL baseline generation.")
    parser.add_argument("--trace-file", type=Path, default=None, help="Optional trace path used when generating a new baseline report.")
    parser.add_argument("--python-bin", type=Path, default=None, help="Optional Python interpreter passed through to run_hover_sim.py.")
    parser.add_argument("--display", default=":0", help="DISPLAY value exported to CrazySim when generating a new baseline report.")
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir(), help="Directory where the baseline and mutated scenario reports are written.")
    parser.add_argument("--scenario", action="append", choices=hover_sim_scenario_names(), help="One or more scenario names to run. Defaults to all supported scenarios.")
    parser.add_argument("--setpoint-period-s", type=float, default=0.1, help="Replay setpoint cadence in seconds.")
    parser.add_argument("--json", action="store_true", help="Emit the scenario-suite result as JSON.")
    parser.add_argument("hover_args", nargs=argparse.REMAINDER, help="Additional run_hover_sim.py / run_hover_test.py arguments, passed after `--`.")
    return parser


def _write_baseline_hover_report_json(
    hover_report: dict[str, object],
    *,
    output_dir: Path,
) -> Path:
    raw_report = hover_report.get("report", hover_report)
    if not isinstance(raw_report, dict):
        raise RuntimeError("hover baseline payload must contain a report object")
    path = output_dir / "baseline_hover_report.json"
    path.write_text(json.dumps({"report": raw_report}, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _normalize_hover_args(raw_args: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(item) for item in raw_args if str(item).strip())
    if normalized and normalized[0] == "--":
        normalized = normalized[1:]
    return normalized


def _generate_baseline_from_crazysim(args: argparse.Namespace) -> tuple[Path, Path | None, dict[str, object]]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    hover_args = _normalize_hover_args(args.hover_args)
    command: list[str] = [
        str(sys.executable),
        str(_SCRIPT_DIR / "run_hover_sim.py"),
        "--crazysim-root",
        str(Path(args.crazysim_root).expanduser().resolve()),
        "--backend",
        str(args.backend),
        "--model",
        str(args.model),
        "--x-m",
        str(float(args.x_m)),
        "--y-m",
        str(float(args.y_m)),
        "--startup-settle-s",
        str(float(args.startup_settle_s)),
        "--hover-timeout-s",
        str(float(args.hover_timeout_s)),
        "--workspace",
        str(Path(args.workspace).expanduser().resolve()),
        "--display",
        str(args.display),
        "--json",
    ]
    if args.trace_file is not None:
        command.extend(("--trace-file", str(Path(args.trace_file).expanduser().resolve())))
    if args.python_bin is not None:
        command.extend(("--python-bin", str(_normalize_nonresolving_path(Path(args.python_bin)))))
    if hover_args:
        command.append("--")
        command.extend(hover_args)
    completed = subprocess.run(
        tuple(command),
        cwd=str(_REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    stdout_text = completed.stdout.strip()
    stderr_text = completed.stderr.strip()
    if completed.returncode != 0:
        raise RuntimeError(
            "run_hover_sim.py failed while generating the scenario baseline: "
            f"returncode={completed.returncode} stdout={stdout_text!r} stderr={stderr_text!r}"
        )
    if not stdout_text:
        raise RuntimeError("run_hover_sim.py produced no JSON output while generating the scenario baseline")
    payload = json.loads(stdout_text)
    if not isinstance(payload, dict):
        raise RuntimeError("run_hover_sim.py JSON payload must be an object")
    hover_report = payload.get("hover_report")
    if not isinstance(hover_report, dict):
        raise RuntimeError("run_hover_sim.py JSON payload is missing `hover_report`")
    baseline_report_path = _write_baseline_hover_report_json(hover_report, output_dir=output_dir)
    sim_payload = payload.get("sim")
    trace_copy_path: Path | None = None
    if isinstance(sim_payload, dict):
        trace_path_raw = sim_payload.get("trace_file")
        if trace_path_raw is not None:
            trace_source_path = Path(str(trace_path_raw)).expanduser().resolve()
            if trace_source_path.is_file():
                trace_copy_path = output_dir / "baseline_hover_trace.jsonl"
                shutil.copyfile(trace_source_path, trace_copy_path)
    return (baseline_report_path, trace_copy_path, payload)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_payload: dict[str, object] | None = None
    if args.baseline_report_json is not None:
        baseline_report_path = Path(args.baseline_report_json).expanduser().resolve()
        baseline_trace_path = (
            None
            if args.baseline_trace_file is None
            else Path(args.baseline_trace_file).expanduser().resolve()
        )
    else:
        baseline_report_path, baseline_trace_path, baseline_payload = _generate_baseline_from_crazysim(args)

    artifact = load_hover_replay_artifact(
        baseline_report_path,
        trace_path=baseline_trace_path,
    )
    scenario_names = tuple(args.scenario or hover_sim_scenario_names())
    for scenario_name in scenario_names:
        scenario_artifact = apply_hover_sim_scenario(artifact, scenario_name=scenario_name)
        write_hover_replay_artifact_json(
            scenario_artifact,
            output_dir / f"scenario_{scenario_name}.json",
        )
    scenario_results = run_hover_sim_scenario_suite(
        artifact,
        scenario_names=scenario_names,
        setpoint_period_s=float(args.setpoint_period_s),
    )
    payload = {
        "baseline": {
            "report_json": str(baseline_report_path),
            "trace_file": None if baseline_trace_path is None else str(baseline_trace_path),
            "source": "report" if args.baseline_report_json is not None else "crazysim",
        },
        "generated_from_crazysim": baseline_payload,
        "output_dir": str(output_dir),
        "scenario_names": scenario_names,
        "results": tuple(serialize_hover_sim_scenario_result(result) for result in scenario_results),
        "all_matched_expectations": all(result.matches_expectation for result in scenario_results),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
