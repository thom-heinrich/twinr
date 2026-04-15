#!/usr/bin/env python3
"""Run real CrazySim disturbance scenarios through Twinr's hover lane."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys
from typing import Sequence

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from hover_physical_disturbance_scenarios import (  # noqa: E402
    _default_output_dir,
    hover_physical_disturbance_scenario_names,
    run_hover_physical_disturbance_suite,
    serialize_hover_physical_disturbance_result,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crazysim-root", type=Path, required=True)
    parser.add_argument("--backend", choices=("gazebo", "mujoco"), default="mujoco")
    parser.add_argument("--model", default="cf2x_T350")
    parser.add_argument("--x-m", type=float, default=0.0)
    parser.add_argument("--y-m", type=float, default=0.0)
    parser.add_argument("--startup-settle-s", type=float, default=3.0)
    parser.add_argument("--hover-timeout-s", type=float, default=30.0)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help=(
            "Workspace root for cflib cache files during physical SITL runs. "
            "Defaults to <output-dir>/_workspaces to keep suites isolated."
        ),
    )
    parser.add_argument("--python-bin", type=Path, default=None)
    parser.add_argument("--display", default=":0")
    parser.add_argument("--setpoint-period-s", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument(
        "--scenario",
        action="append",
        choices=hover_physical_disturbance_scenario_names(),
        help="One or more physical disturbance scenario names. Defaults to all supported scenarios.",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "hover_args",
        nargs=argparse.REMAINDER,
        help="Additional run_hover_sim.py / run_hover_test.py arguments, passed after `--`.",
    )
    return parser


def _normalize_hover_args(raw_args: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(item) for item in raw_args if str(item).strip())
    if normalized and normalized[0] == "--":
        normalized = normalized[1:]
    return normalized


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace_root = (
        output_dir / "_workspaces"
        if args.workspace is None
        else Path(args.workspace).expanduser().resolve()
    )
    scenario_names = tuple(args.scenario or hover_physical_disturbance_scenario_names())
    results = run_hover_physical_disturbance_suite(
        scenario_names=scenario_names,
        crazysim_root=Path(args.crazysim_root).expanduser().resolve(),
        output_dir=output_dir,
        backend=str(args.backend),
        model=str(args.model),
        x_m=float(args.x_m),
        y_m=float(args.y_m),
        startup_settle_s=float(args.startup_settle_s),
        hover_timeout_s=float(args.hover_timeout_s),
        workspace=workspace_root,
        python_bin=None if args.python_bin is None else Path(args.python_bin),
        display=str(args.display),
        setpoint_period_s=float(args.setpoint_period_s),
        hover_args=_normalize_hover_args(args.hover_args),
    )
    payload = {
        "output_dir": str(output_dir),
        "scenario_names": scenario_names,
        "results": tuple(serialize_hover_physical_disturbance_result(result) for result in results),
        "all_matched_expectations": all(result.matches_expectation for result in results),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
