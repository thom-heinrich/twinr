#!/usr/bin/env python3
"""Run the fixed hover acceptance gate across replay, SITL, and physical SIM.

This script is an orchestration lane only. It does not implement a second
hover stack. It combines:

1. one or more stored JSONL/report replay cases through `replay_hover_trace.py`
2. one fresh or stored CrazySim scenario suite through `run_hover_sim_scenarios.py`
3. one fresh or stored CrazySim physical disturbance suite through
   `run_hover_sim_disturbances.py`

The gate passes only when every replay reproduces the expected outcome class,
the adversarial scenario suite reports `all_matched_expectations=true`, and
the physical disturbance suite reports `all_matched_expectations=true`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
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

from hover_physical_disturbance_scenarios import hover_physical_disturbance_scenario_names  # noqa: E402
from hover_sim_scenarios import hover_sim_scenario_names  # noqa: E402
from replay_hover_trace import run_hover_replay  # noqa: E402
from run_hover_test import HOVER_RUNTIME_MODE_HARDWARE, HOVER_RUNTIME_MODE_SITL  # noqa: E402
from twinr.hardware.crazyflie_hover_replay import load_hover_replay_artifact  # noqa: E402


_REPLAY_CASE_SEPARATOR = "|"
_SELECTED_BASELINE_REPLAY_TOKEN = "@selected_baseline"


@dataclass(frozen=True, slots=True)
class HoverReplayGateCase:
    """Describe one replay artifact that must pass the acceptance gate."""

    report_json: Path | None
    trace_file: Path | None = None
    runtime_mode: str = HOVER_RUNTIME_MODE_HARDWARE
    expected_outcome_class: str | None = None
    use_selected_baseline: bool = False


def _default_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("/tmp") / f"twinr-hover-acceptance-gate-{timestamp}"


def _normalize_nonresolving_path(path: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (Path.cwd() / candidate).absolute()


def _parse_replay_case(raw: str) -> HoverReplayGateCase:
    parts = [part.strip() for part in str(raw).split(_REPLAY_CASE_SEPARATOR)]
    if not parts or not parts[0]:
        raise ValueError(
            "replay case must start with a report path in the form "
            "`REPORT_JSON[|TRACE_FILE][|RUNTIME_MODE][|EXPECTED_OUTCOME_CLASS]`"
        )
    while len(parts) < 4:
        parts.append("")
    if len(parts) > 4:
        raise ValueError(
            "replay case accepts at most 4 fields: "
            "`REPORT_JSON[|TRACE_FILE][|RUNTIME_MODE][|EXPECTED_OUTCOME_CLASS]`"
        )
    runtime_mode = parts[2] or HOVER_RUNTIME_MODE_HARDWARE
    if runtime_mode not in {HOVER_RUNTIME_MODE_HARDWARE, HOVER_RUNTIME_MODE_SITL}:
        raise ValueError(
            f"unsupported replay runtime mode `{runtime_mode}`; choose "
            f"`{HOVER_RUNTIME_MODE_HARDWARE}` or `{HOVER_RUNTIME_MODE_SITL}`"
        )
    use_selected_baseline = parts[0] == _SELECTED_BASELINE_REPLAY_TOKEN
    return HoverReplayGateCase(
        report_json=(
            None
            if use_selected_baseline
            else Path(parts[0]).expanduser().resolve()
        ),
        trace_file=None if not parts[1] else Path(parts[1]).expanduser().resolve(),
        runtime_mode=runtime_mode,
        expected_outcome_class=parts[3] or None,
        use_selected_baseline=use_selected_baseline,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay-case",
        action="append",
        required=True,
        help=(
            "Replay case token of the form "
            "`REPORT_JSON[|TRACE_FILE][|RUNTIME_MODE][|EXPECTED_OUTCOME_CLASS]`. "
            f"Use `{_SELECTED_BASELINE_REPLAY_TOKEN}` as REPORT_JSON to replay the "
            "currently selected baseline (fresh or stored) instead of a fixed path. "
            "When EXPECTED_OUTCOME_CLASS is omitted, the stored report outcome is used."
        ),
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--crazysim-root",
        type=Path,
        default=None,
        help="Path to a CrazySim checkout used to generate one fresh baseline report.",
    )
    source_group.add_argument(
        "--baseline-report-json",
        type=Path,
        default=None,
        help="Path to an existing hover report JSON used as the scenario-suite baseline.",
    )
    parser.add_argument(
        "--baseline-trace-file",
        type=Path,
        default=None,
        help="Optional phase trace for an existing scenario-suite baseline report.",
    )
    parser.add_argument(
        "--physical-suite-json",
        type=Path,
        default=None,
        help=(
            "Path to a stored physical-disturbance suite JSON. Required when the "
            "gate runs from --baseline-report-json because live-flight acceptance "
            "always requires replay + scenario + physical disturbance proof."
        ),
    )
    parser.add_argument("--backend", choices=("gazebo", "mujoco"), default="mujoco")
    parser.add_argument("--model", default="cf2x_T350")
    parser.add_argument("--x-m", type=float, default=0.0)
    parser.add_argument("--y-m", type=float, default=0.0)
    parser.add_argument("--startup-settle-s", type=float, default=3.0)
    parser.add_argument(
        "--fresh-baseline-runs",
        type=int,
        default=3,
        help=(
            "Number of fresh nominal CrazySim baselines required for live-flight "
            "eligibility when --crazysim-root is used (default: 3)."
        ),
    )
    parser.add_argument("--hover-timeout-s", type=float, default=30.0)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("/tmp/twinr-crazysim-scenarios-workspace"),
    )
    parser.add_argument("--trace-file", type=Path, default=None)
    parser.add_argument("--python-bin", type=Path, default=None)
    parser.add_argument("--display", default=":0")
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument(
        "--scenario",
        action="append",
        choices=hover_sim_scenario_names(),
        help="One or more scenario names to run. Defaults to all supported scenarios.",
    )
    parser.add_argument(
        "--physical-scenario",
        action="append",
        choices=hover_physical_disturbance_scenario_names(),
        help="One or more physical disturbance scenario names to run. Defaults to all supported scenarios.",
    )
    parser.add_argument("--setpoint-period-s", type=float, default=0.1)
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


def _suite_workspace(args: argparse.Namespace, suite_name: str) -> Path:
    """Return one gate-local CrazySim workspace for the selected suite."""

    return (
        Path(args.output_dir).expanduser().resolve()
        / "_gate_workspaces"
        / str(suite_name)
    )


def _run_replay_case(case: HoverReplayGateCase, *, setpoint_period_s: float) -> dict[str, object]:
    if case.report_json is None:
        raise RuntimeError("replay case is missing a resolved report path")
    artifact = load_hover_replay_artifact(case.report_json, trace_path=case.trace_file)
    expected_outcome_class = case.expected_outcome_class or str(artifact.report_payload["outcome_class"])
    replay_payload = run_hover_replay(
        artifact,
        runtime_mode=case.runtime_mode,
        setpoint_period_s=float(setpoint_period_s),
    )
    replay = replay_payload["replay"]
    actual_outcome_class = str(replay["outcome_class"])
    return {
        "case": {
            "report_json": None if case.report_json is None else str(case.report_json),
            "trace_file": None if case.trace_file is None else str(case.trace_file),
            "runtime_mode": case.runtime_mode,
            "expected_outcome_class": case.expected_outcome_class,
            "use_selected_baseline": case.use_selected_baseline,
        },
        "expected_outcome_class": expected_outcome_class,
        "actual_outcome_class": actual_outcome_class,
        "matches_expected_outcome": actual_outcome_class == expected_outcome_class,
        "artifact": replay_payload["artifact"],
        "replay": replay,
    }


def _resolve_replay_case(
    case: HoverReplayGateCase,
    *,
    selected_baseline_report_json: Path | None,
    selected_baseline_trace_file: Path | None,
) -> HoverReplayGateCase:
    """Resolve one replay case against the selected gate baseline when requested."""

    if not case.use_selected_baseline:
        return case
    return HoverReplayGateCase(
        report_json=selected_baseline_report_json,
        trace_file=(
            case.trace_file
            if case.trace_file is not None
            else selected_baseline_trace_file
        ),
        runtime_mode=case.runtime_mode,
        expected_outcome_class=case.expected_outcome_class,
        use_selected_baseline=True,
    )


def _run_resolved_replay_case(
    case: HoverReplayGateCase,
    *,
    selected_baseline_report_json: Path | None,
    selected_baseline_trace_file: Path | None,
    setpoint_period_s: float,
) -> dict[str, object]:
    """Run one replay case or fail closed with structured output if unresolved."""

    resolved_case = _resolve_replay_case(
        case,
        selected_baseline_report_json=selected_baseline_report_json,
        selected_baseline_trace_file=selected_baseline_trace_file,
    )
    if resolved_case.report_json is None:
        return {
            "case": {
                "report_json": None,
                "trace_file": None if resolved_case.trace_file is None else str(resolved_case.trace_file),
                "runtime_mode": resolved_case.runtime_mode,
                "expected_outcome_class": resolved_case.expected_outcome_class,
                "use_selected_baseline": resolved_case.use_selected_baseline,
            },
            "expected_outcome_class": resolved_case.expected_outcome_class,
            "actual_outcome_class": None,
            "matches_expected_outcome": False,
            "failure": "selected baseline was unavailable for the requested replay case",
        }
    return _run_replay_case(
        resolved_case,
        setpoint_period_s=setpoint_period_s,
    )


def _write_hover_report_json(path: Path, hover_report: dict[str, object]) -> Path:
    """Persist one structured hover report in the stored replay-artifact shape."""

    raw_report = _unwrap_hover_report(hover_report)
    if raw_report is None:
        raise RuntimeError("hover baseline payload must contain a report object")
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps({"report": raw_report}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return resolved


def _unwrap_hover_report(hover_report: dict[str, object]) -> dict[str, object] | None:
    """Return the real hover report body from either stored or wrapper payloads."""

    raw_report = hover_report.get("report", hover_report)
    if not isinstance(raw_report, dict):
        return None
    return raw_report


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeError(f"{label} JSON `{resolved}` does not exist")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label} JSON `{resolved}` is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} JSON `{resolved}` must contain an object payload")
    payload["loaded_from_json"] = str(resolved)
    return payload


def _fresh_baseline_contract_summary(report: dict[str, object]) -> tuple[bool, bool, tuple[str, ...]]:
    """Return nominal/touchdown pass bits for one fresh SITL baseline report."""

    unwrapped_report = _unwrap_hover_report(report)
    if unwrapped_report is None:
        return (False, False, ("hover report is missing report data",))
    primitive_outcome = unwrapped_report.get("primitive_outcome")
    if not isinstance(primitive_outcome, dict):
        return (False, False, ("hover report is missing primitive_outcome",))

    failures: list[str] = []
    if bool(unwrapped_report.get("completed")) is not True:
        failures.append("report.completed was not true")
    if str(unwrapped_report.get("status")) != "completed":
        failures.append(
            f"report.status was `{unwrapped_report.get('status')}` instead of `completed`"
        )
    if str(unwrapped_report.get("outcome_class")) != "bounded_hover_ok":
        failures.append(
            "report.outcome_class was "
            f"`{unwrapped_report.get('outcome_class')}` instead of `bounded_hover_ok`"
        )
    if bool(unwrapped_report.get("landed")) is not True:
        failures.append("report.landed was not true")
    if primitive_outcome.get("took_off") is not True:
        failures.append("primitive_outcome.took_off was not true")
    if primitive_outcome.get("stable_hover_established") is not True:
        failures.append("primitive_outcome.stable_hover_established was not true")
    if primitive_outcome.get("landed") is not True:
        failures.append("primitive_outcome.landed was not true")
    if primitive_outcome.get("forced_motor_cutoff") is not False:
        failures.append("primitive_outcome.forced_motor_cutoff was not false")

    touchdown_source = primitive_outcome.get("touchdown_confirmation_source")
    touchdown_failures: list[str] = []
    if touchdown_source in {None, "", "timeout_forced_cutoff"}:
        touchdown_failures.append(
            "primitive_outcome.touchdown_confirmation_source did not report a deterministic nominal touchdown"
        )

    return (
        not failures,
        not failures and not touchdown_failures,
        tuple((*failures, *touchdown_failures)),
    )


def _run_fresh_baseline_repeatability(args: argparse.Namespace) -> dict[str, object]:
    """Run repeated fresh nominal MuJoCo baselines and persist the selected artifact."""

    run_count = max(1, int(args.fresh_baseline_runs))
    output_root = Path(args.output_dir).expanduser().resolve() / "fresh_baseline_repeatability"
    output_root.mkdir(parents=True, exist_ok=True)
    hover_args = _normalize_hover_args(args.hover_args)
    runs: list[dict[str, object]] = []

    for run_index in range(1, run_count + 1):
        run_output_dir = output_root / f"run_{run_index:02d}"
        run_output_dir.mkdir(parents=True, exist_ok=True)
        trace_file = run_output_dir / "baseline_hover_trace.jsonl"
        workspace = _suite_workspace(args, f"fresh_baseline_run_{run_index:02d}")
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
            str(workspace),
            "--trace-file",
            str(trace_file),
            "--display",
            str(args.display),
            "--json",
        ]
        if args.python_bin is not None:
            command.extend(
                ("--python-bin", str(_normalize_nonresolving_path(Path(args.python_bin))))
            )
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
        payload: dict[str, object] | None = None
        hover_report: dict[str, object] | None = None
        failure_reasons: list[str] = []
        if stdout_text:
            try:
                loaded_payload = json.loads(stdout_text)
            except json.JSONDecodeError as exc:
                failure_reasons.append(
                    "run_hover_sim.py did not emit valid JSON for a fresh baseline run: "
                    f"{exc}"
                )
            else:
                if not isinstance(loaded_payload, dict):
                    failure_reasons.append(
                        "run_hover_sim.py fresh baseline payload was not a JSON object"
                    )
                else:
                    payload = loaded_payload
                    report_candidate = payload.get("hover_report")
                    if not isinstance(report_candidate, dict):
                        failure_reasons.append(
                            "run_hover_sim.py fresh baseline payload was missing `hover_report`"
                        )
                    else:
                        hover_report = report_candidate
        else:
            failure_reasons.append("run_hover_sim.py emitted no stdout for a fresh baseline run")

        report_json_path: Path | None = None
        if hover_report is not None:
            report_json_path = _write_hover_report_json(
                run_output_dir / "baseline_hover_report.json",
                hover_report,
            )
        unwrapped_hover_report = (
            _unwrap_hover_report(hover_report) if isinstance(hover_report, dict) else None
        )
        primitive_outcome = (
            unwrapped_hover_report.get("primitive_outcome")
            if isinstance(unwrapped_hover_report, dict)
            else None
        )
        nominal_passed = False
        touchdown_passed = False
        contract_failures: tuple[str, ...] = ()
        if hover_report is not None:
            nominal_passed, touchdown_passed, contract_failures = _fresh_baseline_contract_summary(
                hover_report
            )
        else:
            contract_failures = tuple(failure_reasons)

        runs.append(
            {
                "index": run_index,
                "command": tuple(command),
                "returncode": int(completed.returncode),
                "hover_worker_returncode": (
                    None
                    if payload is None
                    else payload.get("hover_worker_returncode")
                ),
                "report_json": None if report_json_path is None else str(report_json_path),
                "trace_file": str(trace_file) if trace_file.is_file() else None,
                "status": None if unwrapped_hover_report is None else unwrapped_hover_report.get("status"),
                "outcome_class": (
                    None if unwrapped_hover_report is None else unwrapped_hover_report.get("outcome_class")
                ),
                "completed": None if unwrapped_hover_report is None else unwrapped_hover_report.get("completed"),
                "landed": None if unwrapped_hover_report is None else unwrapped_hover_report.get("landed"),
                "primitive_outcome": primitive_outcome,
                "matches_nominal_contract": nominal_passed,
                "touchdown_contract_ok": touchdown_passed,
                "contract_failures": contract_failures,
                "stdout": stdout_text,
                "stderr": stderr_text,
            }
        )

    selected_baseline = next(
        (
            run
            for run in runs
            if bool(run["matches_nominal_contract"])
            and run.get("report_json") is not None
        ),
        None,
    )
    repeatability_passed = all(bool(run["matches_nominal_contract"]) for run in runs)
    touchdown_passed = all(bool(run["touchdown_contract_ok"]) for run in runs)
    return {
        "run_count": run_count,
        "output_dir": str(output_root),
        "runs": tuple(runs),
        "repeatability_passed": repeatability_passed,
        "touchdown_passed": touchdown_passed,
        "selected_baseline_report_json": (
            None if selected_baseline is None else selected_baseline["report_json"]
        ),
        "selected_baseline_trace_file": (
            None if selected_baseline is None else selected_baseline["trace_file"]
        ),
    }


def _run_scenario_suite(
    args: argparse.Namespace,
    *,
    baseline_report_json_override: Path | None = None,
    baseline_trace_file_override: Path | None = None,
) -> dict[str, object]:
    scenario_output_dir = Path(args.output_dir).expanduser().resolve() / "scenario_suite"
    command: list[str] = [
        str(sys.executable),
        str(_SCRIPT_DIR / "run_hover_sim_scenarios.py"),
        "--output-dir",
        str(scenario_output_dir),
        "--setpoint-period-s",
        str(float(args.setpoint_period_s)),
        "--json",
    ]
    baseline_report_json = (
        baseline_report_json_override
        if baseline_report_json_override is not None
        else args.baseline_report_json
    )
    baseline_trace_file = (
        baseline_trace_file_override
        if baseline_trace_file_override is not None
        else args.baseline_trace_file
    )
    if baseline_report_json is not None:
        command.extend(
            (
                "--baseline-report-json",
                str(Path(baseline_report_json).expanduser().resolve()),
            )
        )
        if baseline_trace_file is not None:
            command.extend(
                (
                    "--baseline-trace-file",
                    str(Path(baseline_trace_file).expanduser().resolve()),
                )
            )
    else:
        command.extend(
            (
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
                str(_suite_workspace(args, "scenario_suite")),
                "--display",
                str(args.display),
            )
        )
        if args.trace_file is not None:
            command.extend(("--trace-file", str(Path(args.trace_file).expanduser().resolve())))
        if args.python_bin is not None:
            command.extend(("--python-bin", str(_normalize_nonresolving_path(Path(args.python_bin)))))
    for scenario_name in tuple(args.scenario or ()):
        command.extend(("--scenario", str(scenario_name)))
    hover_args = _normalize_hover_args(args.hover_args)
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
            "run_hover_sim_scenarios.py failed while running the acceptance gate: "
            f"returncode={completed.returncode} stdout={stdout_text!r} stderr={stderr_text!r}"
        )
    if not stdout_text:
        raise RuntimeError("run_hover_sim_scenarios.py produced no JSON output for the acceptance gate")
    payload = json.loads(stdout_text)
    if not isinstance(payload, dict):
        raise RuntimeError("run_hover_sim_scenarios.py JSON payload must be an object")
    payload["command"] = tuple(command)
    return payload


def _run_physical_suite(args: argparse.Namespace) -> dict[str, object]:
    if args.physical_suite_json is not None:
        return _load_json_object(Path(args.physical_suite_json), label="physical disturbance suite")
    if args.crazysim_root is None:
        raise RuntimeError(
            "run_hover_acceptance_gate.py requires either --crazysim-root for a fresh "
            "physical disturbance proof or --physical-suite-json for a stored one"
        )
    physical_output_dir = Path(args.output_dir).expanduser().resolve() / "physical_suite"
    command: list[str] = [
        str(sys.executable),
        str(_SCRIPT_DIR / "run_hover_sim_disturbances.py"),
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
        str(_suite_workspace(args, "physical_suite")),
        "--display",
        str(args.display),
        "--output-dir",
        str(physical_output_dir),
        "--setpoint-period-s",
        str(float(args.setpoint_period_s)),
        "--json",
    ]
    if args.python_bin is not None:
        command.extend(("--python-bin", str(_normalize_nonresolving_path(Path(args.python_bin)))))
    for scenario_name in tuple(args.physical_scenario or ()):
        command.extend(("--scenario", str(scenario_name)))
    hover_args = _normalize_hover_args(args.hover_args)
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
            "run_hover_sim_disturbances.py failed while running the acceptance gate: "
            f"returncode={completed.returncode} stdout={stdout_text!r} stderr={stderr_text!r}"
        )
    if not stdout_text:
        raise RuntimeError(
            "run_hover_sim_disturbances.py produced no JSON output for the acceptance gate"
        )
    payload = json.loads(stdout_text)
    if not isinstance(payload, dict):
        raise RuntimeError("run_hover_sim_disturbances.py JSON payload must be an object")
    payload["command"] = tuple(command)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.crazysim_root is None and args.physical_suite_json is None:
        raise RuntimeError(
            "run_hover_acceptance_gate.py requires --physical-suite-json when "
            "running from --baseline-report-json because the acceptance gate "
            "must include the physical disturbance suite"
        )
    replay_cases = tuple(_parse_replay_case(raw_case) for raw_case in tuple(args.replay_case))
    fresh_baseline_repeatability_payload: dict[str, object] | None = None
    selected_baseline_report_json: Path | None = (
        None
        if args.baseline_report_json is None
        else Path(args.baseline_report_json).expanduser().resolve()
    )
    selected_baseline_trace_file: Path | None = (
        None
        if args.baseline_trace_file is None
        else Path(args.baseline_trace_file).expanduser().resolve()
    )
    if args.crazysim_root is not None:
        fresh_baseline_repeatability_payload = _run_fresh_baseline_repeatability(args)
        selected_report_raw = fresh_baseline_repeatability_payload["selected_baseline_report_json"]
        selected_trace_raw = fresh_baseline_repeatability_payload["selected_baseline_trace_file"]
        selected_baseline_report_json = (
            None if selected_report_raw is None else Path(str(selected_report_raw))
        )
        selected_baseline_trace_file = (
            None if selected_trace_raw is None else Path(str(selected_trace_raw))
        )
    replay_results = tuple(
        _run_resolved_replay_case(
            case,
            selected_baseline_report_json=selected_baseline_report_json,
            selected_baseline_trace_file=selected_baseline_trace_file,
            setpoint_period_s=float(args.setpoint_period_s),
        )
        for case in replay_cases
    )
    if args.crazysim_root is not None and selected_baseline_report_json is None:
        scenario_suite_payload = {
            "skipped": True,
            "reason": "fresh_baseline_repeatability_failed",
            "all_matched_expectations": False,
        }
        physical_suite_payload = {
            "skipped": True,
            "reason": "fresh_baseline_repeatability_failed",
            "all_matched_expectations": False,
        }
    else:
        scenario_suite_payload = _run_scenario_suite(
            args,
            baseline_report_json_override=selected_baseline_report_json,
            baseline_trace_file_override=selected_baseline_trace_file,
        )
        physical_suite_payload = _run_physical_suite(args)
    all_replays_matched = all(bool(result["matches_expected_outcome"]) for result in replay_results)
    all_scenarios_matched = bool(scenario_suite_payload["all_matched_expectations"])
    all_physical_matched = bool(physical_suite_payload["all_matched_expectations"])
    fresh_baseline_repeatability_passed = (
        None
        if fresh_baseline_repeatability_payload is None
        else bool(fresh_baseline_repeatability_payload["repeatability_passed"])
    )
    fresh_baseline_touchdown_passed = (
        None
        if fresh_baseline_repeatability_payload is None
        else bool(fresh_baseline_repeatability_payload["touchdown_passed"])
    )
    if args.crazysim_root is not None:
        gate_passed = (
            all_replays_matched
            and all_scenarios_matched
            and all_physical_matched
            and fresh_baseline_repeatability_passed is True
            and fresh_baseline_touchdown_passed is True
        )
    else:
        gate_passed = all_replays_matched and all_scenarios_matched and all_physical_matched
    live_flight_eligible = (
        args.crazysim_root is not None
        and gate_passed
        and fresh_baseline_repeatability_passed is True
        and fresh_baseline_touchdown_passed is True
    )
    payload = {
        "gate_passed": gate_passed,
        "live_flight_eligible": live_flight_eligible,
        "all_replays_matched": all_replays_matched,
        "all_scenarios_matched": all_scenarios_matched,
        "all_physical_matched": all_physical_matched,
        "fresh_baseline_repeatability_passed": fresh_baseline_repeatability_passed,
        "fresh_baseline_touchdown_passed": fresh_baseline_touchdown_passed,
        "fresh_baseline_repeatability": fresh_baseline_repeatability_payload,
        "replay_results": replay_results,
        "scenario_suite": scenario_suite_payload,
        "physical_suite": physical_suite_payload,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, sort_keys=True))
    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
