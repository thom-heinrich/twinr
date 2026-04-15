"""Run real physical CrazySim disturbance cases through Twinr's hover lane.

This module launches fresh MuJoCo SITL runs with explicit force/torque
disturbances injected into the live plant. It then replays the captured hover
report through Twinr's bounded hover evaluator so recover-vs-abort contracts
stay identical across replay and physical-sim proofs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Mapping, Sequence

from hover_sim_scenarios import (
    HoverScenarioContractEvaluation,
    HoverSimScenarioExpectation,
    HoverGuardExpectation,
    HoverRecoveryExpectation,
    HoverRecoveryWindow,
    evaluate_hover_replay_against_expectation,
)
from replay_hover_trace import run_hover_replay
from run_hover_test import HOVER_RUNTIME_MODE_SITL
from twinr.hardware.crazyflie_hover_replay import load_hover_replay_artifact
from twinr.hardware.crazyflie_sim_disturbance import (
    CrazySimDisturbancePlan,
    CrazySimDisturbancePulse,
    CrazySimDisturbanceRuntimeEvent,
    load_crazysim_disturbance_runtime_events,
    write_crazysim_disturbance_plan,
)


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_RUN_HOVER_SIM_SCRIPT = _SCRIPT_DIR / "run_hover_sim.py"
_HOLD_PHASE = "hover_primitive_hold"
_HOLD_BEGIN = "begin"
_IN_HOVER_DISTURBANCE_START_S = 0.20


@dataclass(frozen=True, slots=True)
class HoverPhysicalDisturbanceSpec:
    """Describe one real MuJoCo disturbance experiment."""

    name: str
    description: str
    expectation: HoverSimScenarioExpectation
    disturbance_plan: CrazySimDisturbancePlan | None = None
    wind_speed_mps: float = 0.0
    wind_direction_deg: float = 0.0
    gust_intensity_mps: float = 0.0
    turbulence: str = "none"
    mass_kg: float | None = None


@dataclass(frozen=True, slots=True)
class HoverPhysicalDisturbanceResult:
    """Capture one physical disturbance run and its replayed contract proof."""

    scenario: HoverPhysicalDisturbanceSpec
    sim_returncode: int
    sim_payload: dict[str, object]
    replay_payload: dict[str, object]
    evaluation: HoverScenarioContractEvaluation
    report_json: Path
    trace_file: Path | None
    disturbance_spec_json: Path | None
    disturbance_runtime_jsonl: Path | None
    recovery_window: HoverRecoveryWindow | None

    @property
    def actual_outcome_class(self) -> str:
        replay = self.replay_payload.get("replay")
        if not isinstance(replay, Mapping):
            raise ValueError("physical disturbance replay payload is missing `replay`")
        return str(replay.get("outcome_class"))

    @property
    def matches_expectation(self) -> bool:
        return (
            self.actual_outcome_class == self.scenario.expectation.outcome_class
            and not self.evaluation.missing_failure_substrings
            and not self.evaluation.contract_failures
            and not self.evaluation.guard_failures
            and not self.evaluation.recovery_failures
        )


def _default_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("/tmp") / f"twinr-crazysim-physical-disturbances-{timestamp}"


def _extract_target_height_m(report_payload: Mapping[str, object]) -> float:
    """Return the required hover height from one serialized hover report."""

    height_raw = report_payload.get("height_m")
    if not isinstance(height_raw, (int, float)):
        raise ValueError("hover report is missing numeric `height_m`")
    return float(height_raw)


def _normalize_nonresolving_path(path: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (Path.cwd() / candidate).absolute()


def _transient_plan(
    *,
    name: str,
    world_force_n: tuple[float, float, float] = (0.0, 0.0, 0.0),
    body_torque_nm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    start_s: float = 0.25,
    duration_s: float = 0.25,
    activation_height_m: float = 0.08,
) -> CrazySimDisturbancePlan:
    return CrazySimDisturbancePlan(
        name=name,
        description=f"Transient physical disturbance `{name}` applied after liftoff.",
        activation_mode="after_airborne",
        activation_height_m=float(activation_height_m),
        pulses=(
            CrazySimDisturbancePulse(
                name=name,
                start_s=start_s,
                duration_s=duration_s,
                target_agent=0,
                world_force_n=world_force_n,
                body_torque_nm=body_torque_nm,
            ),
        ),
    )


def _phase_anchored_plan(
    *,
    name: str,
    world_force_n: tuple[float, float, float] = (0.0, 0.0, 0.0),
    body_torque_nm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    start_s: float = _IN_HOVER_DISTURBANCE_START_S,
    duration_s: float = 0.25,
) -> CrazySimDisturbancePlan:
    return CrazySimDisturbancePlan(
        name=name,
        description=f"Physical disturbance `{name}` anchored on host hover hold begin.",
        activation_mode="after_host_phase",
        activation_height_m=0.0,
        activation_phase=_HOLD_PHASE,
        activation_status=_HOLD_BEGIN,
        pulses=(
            CrazySimDisturbancePulse(
                name=name,
                start_s=start_s,
                duration_s=duration_s,
                target_agent=0,
                world_force_n=world_force_n,
                body_torque_nm=body_torque_nm,
            ),
        ),
    )


def _coerce_sim_mapping(sim_payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = sim_payload.get(key)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"physical disturbance payload is missing mapping `{key}`")
    return value


def _disturbance_runtime_file(sim_payload: Mapping[str, object]) -> Path | None:
    sim_mapping = _coerce_sim_mapping(sim_payload, "sim")
    runtime_path = sim_mapping.get("disturbance_runtime_jsonl")
    if runtime_path is None:
        return None
    candidate = Path(str(runtime_path)).expanduser().resolve()
    return candidate if candidate.is_file() else None


def _coerce_replay_phase_events(
    replay_payload: Mapping[str, object],
) -> Sequence[Mapping[str, object]]:
    replay_mapping = replay_payload.get("replay")
    if not isinstance(replay_mapping, Mapping):
        raise RuntimeError("physical disturbance replay payload is missing `replay`")
    phase_events = replay_mapping.get("phase_events")
    if not isinstance(phase_events, Sequence) or isinstance(phase_events, (str, bytes, bytearray)):
        raise RuntimeError(
            "physical disturbance replay payload is missing replay `phase_events`"
        )
    return tuple(
        event
        for event in phase_events
        if isinstance(event, Mapping)
    )


def _phase_elapsed_s(
    phase_events: Sequence[Mapping[str, object]],
    *,
    phase: str,
    status: str,
) -> float:
    for event in phase_events:
        if str(event.get("phase")) != phase or str(event.get("status")) != status:
            continue
        elapsed_s = event.get("elapsed_s")
        if not isinstance(elapsed_s, (int, float)):
            raise RuntimeError(
                f"phase event `{phase}` `{status}` is missing numeric `elapsed_s`"
            )
        return float(elapsed_s)
    raise RuntimeError(
        f"physical disturbance replay payload never reached phase `{phase}` `{status}`"
    )


def _disturbance_recovery_window(
    *,
    scenario: HoverPhysicalDisturbanceSpec,
    disturbance_runtime_events: Sequence[CrazySimDisturbanceRuntimeEvent],
    replay_phase_events: Sequence[Mapping[str, object]],
) -> HoverRecoveryWindow:
    if scenario.disturbance_plan is None:
        raise RuntimeError(
            f"physical disturbance scenario `{scenario.name}` is missing its disturbance plan"
        )
    plan_anchor: CrazySimDisturbanceRuntimeEvent | None = None
    pulse_active_events: list[CrazySimDisturbanceRuntimeEvent] = []
    for event in disturbance_runtime_events:
        if event.plan_name != scenario.disturbance_plan.name:
            continue
        if event.kind == "plan_anchor":
            plan_anchor = event
            continue
        if event.kind == "pulse_active":
            pulse_active_events.append(event)
    if plan_anchor is None:
        raise RuntimeError(
            f"physical disturbance scenario `{scenario.name}` never emitted a plan_anchor runtime event"
        )
    if plan_anchor.host_phase != scenario.disturbance_plan.activation_phase:
        raise RuntimeError(
            f"physical disturbance scenario `{scenario.name}` anchored on unexpected host phase "
            f"`{plan_anchor.host_phase}`"
        )
    if plan_anchor.host_status != scenario.disturbance_plan.activation_status:
        raise RuntimeError(
            f"physical disturbance scenario `{scenario.name}` anchored on unexpected host status "
            f"`{plan_anchor.host_status}`"
        )
    if not pulse_active_events:
        raise RuntimeError(
            f"physical disturbance scenario `{scenario.name}` never emitted an active disturbance pulse"
        )
    replay_anchor_elapsed_s = _phase_elapsed_s(
        replay_phase_events,
        phase=str(scenario.disturbance_plan.activation_phase),
        status=str(scenario.disturbance_plan.activation_status),
    )
    disturbance_start_elapsed_s = replay_anchor_elapsed_s + min(
        float(pulse.start_s) for pulse in scenario.disturbance_plan.pulses
    )
    disturbance_end_elapsed_s = replay_anchor_elapsed_s + max(
        float(pulse.end_s) for pulse in scenario.disturbance_plan.pulses
    )
    return HoverRecoveryWindow(
        disturbance_start_elapsed_s=disturbance_start_elapsed_s,
        disturbance_end_elapsed_s=disturbance_end_elapsed_s,
        source="physical_runtime",
    )


def available_hover_physical_disturbance_scenarios() -> tuple[HoverPhysicalDisturbanceSpec, ...]:
    """Return the supported real CrazySim disturbance cases."""

    return (
        HoverPhysicalDisturbanceSpec(
            name="physical_forward_impulse_recovery",
            description="Apply a short forward world-force impulse and require a bounded backward recovery command.",
            disturbance_plan=_phase_anchored_plan(
                name="forward_impulse",
                world_force_n=(0.040, 0.0, 0.0),
                start_s=_IN_HOVER_DISTURBANCE_START_S,
                duration_s=0.35,
            ),
            expectation=HoverSimScenarioExpectation(
                outcome_class="bounded_hover_ok",
                runtime_mode=HOVER_RUNTIME_MODE_SITL,
                primitive_aborted=False,
                stable_hover_established=True,
                touchdown_confirmation_source="range_only_sitl",
                guard=HoverGuardExpectation(must_block=False),
                recovery=HoverRecoveryExpectation(
                    disturbance_start_progress=0.15,
                    disturbance_end_progress=0.45,
                    max_recovery_delay_s=0.50,
                    settle_height_error_abs_max_m=0.03,
                    settle_required_commands=1,
                    forward_direction="negative",
                    min_forward_abs_mps=0.008,
                ),
            ),
        ),
        HoverPhysicalDisturbanceSpec(
            name="physical_left_impulse_recovery",
            description="Apply a short leftward world-force impulse and require a bounded opposing lateral recovery command.",
            disturbance_plan=_phase_anchored_plan(
                name="left_impulse",
                world_force_n=(0.0, -0.020, 0.0),
                start_s=_IN_HOVER_DISTURBANCE_START_S,
                duration_s=0.35,
            ),
            expectation=HoverSimScenarioExpectation(
                outcome_class="bounded_hover_ok",
                runtime_mode=HOVER_RUNTIME_MODE_SITL,
                primitive_aborted=False,
                stable_hover_established=True,
                touchdown_confirmation_source="range_only_sitl",
                guard=HoverGuardExpectation(must_block=False),
                recovery=HoverRecoveryExpectation(
                    disturbance_start_progress=0.15,
                    disturbance_end_progress=0.45,
                    max_recovery_delay_s=0.50,
                    settle_height_error_abs_max_m=0.03,
                    settle_required_commands=1,
                    left_direction="positive",
                    min_left_abs_mps=0.020,
                ),
            ),
        ),
        HoverPhysicalDisturbanceSpec(
            name="physical_height_drop_recovery",
            description=(
                "Apply a short downward force impulse and require the combined "
                "flight stack to resettle into bounded hover without tripping "
                "the guard."
            ),
            disturbance_plan=_phase_anchored_plan(
                name="height_drop",
                world_force_n=(0.0, 0.0, -0.030),
                start_s=_IN_HOVER_DISTURBANCE_START_S,
                duration_s=0.25,
            ),
            expectation=HoverSimScenarioExpectation(
                outcome_class="bounded_hover_ok",
                runtime_mode=HOVER_RUNTIME_MODE_SITL,
                primitive_aborted=False,
                stable_hover_established=True,
                touchdown_confirmation_source="range_only_sitl",
                guard=HoverGuardExpectation(must_block=False),
                recovery=HoverRecoveryExpectation(
                    disturbance_start_progress=0.20,
                    disturbance_end_progress=0.50,
                    max_recovery_delay_s=0.50,
                    settle_height_error_abs_max_m=0.03,
                    settle_required_commands=1,
                ),
            ),
        ),
        HoverPhysicalDisturbanceSpec(
            name="physical_persistent_forward_abort",
            description="Apply a persistent forward force bias so the hover guard must abort rather than pretending to recover.",
            disturbance_plan=_phase_anchored_plan(
                name="persistent_forward_force",
                world_force_n=(0.080, 0.0, 0.0),
                start_s=_IN_HOVER_DISTURBANCE_START_S,
                duration_s=2.00,
            ),
            expectation=HoverSimScenarioExpectation(
                outcome_class="unstable_hover_aborted",
                failure_substrings=(
                    "hover stability guard tripped",
                    "xy drift reached",
                ),
                runtime_mode=HOVER_RUNTIME_MODE_SITL,
                primitive_aborted=True,
                stable_hover_established=True,
                touchdown_confirmation_source="range_only_sitl",
                guard=HoverGuardExpectation(
                    must_block=True,
                    required_blocked_codes=("xy_drift",),
                ),
            ),
        ),
        HoverPhysicalDisturbanceSpec(
            name="physical_roll_torque_abort",
            description="Apply a violent transient roll-torque disturbance so takeoff must fail closed before the hover lane pretends to stabilize.",
            disturbance_plan=_transient_plan(
                name="roll_torque",
                body_torque_nm=(0.00400, 0.0, 0.0),
                duration_s=0.20,
            ),
            expectation=HoverSimScenarioExpectation(
                outcome_class="takeoff_failed",
                failure_substrings=(
                    "supervisor reported unsafe flags",
                    "roll reached",
                ),
                runtime_mode=HOVER_RUNTIME_MODE_SITL,
                primitive_aborted=True,
                stable_hover_established=False,
                touchdown_confirmation_source="range_only_sitl",
            ),
        ),
    )


def hover_physical_disturbance_scenario_names() -> tuple[str, ...]:
    """Return the supported physical disturbance scenario names."""

    return tuple(spec.name for spec in available_hover_physical_disturbance_scenarios())


def hover_physical_disturbance_spec(name: str) -> HoverPhysicalDisturbanceSpec:
    """Return one validated physical disturbance scenario by name."""

    normalized = str(name).strip()
    for spec in available_hover_physical_disturbance_scenarios():
        if spec.name == normalized:
            return spec
    allowed = ", ".join(hover_physical_disturbance_scenario_names())
    raise ValueError(f"unsupported physical hover disturbance scenario `{name}`; choose one of: {allowed}")


def _write_hover_report_json(
    hover_report: dict[str, object],
    *,
    output_dir: Path,
    scenario_name: str,
) -> Path:
    raw_report = hover_report.get("report", hover_report)
    if not isinstance(raw_report, Mapping):
        raise RuntimeError("hover report payload must contain a report object")
    path = output_dir / f"{scenario_name}_hover_report.json"
    path.write_text(json.dumps({"report": raw_report}, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _build_run_hover_sim_command(
    *,
    crazysim_root: Path,
    backend: str,
    model: str,
    x_m: float,
    y_m: float,
    startup_settle_s: float,
    hover_timeout_s: float,
    workspace: Path,
    python_bin: Path | None,
    display: str,
    trace_file: Path,
    disturbance_spec_json: Path | None,
    scenario: HoverPhysicalDisturbanceSpec,
    hover_args: Sequence[str],
) -> tuple[str, ...]:
    command: list[str] = [
        str(sys.executable),
        str(_RUN_HOVER_SIM_SCRIPT),
        "--crazysim-root",
        str(crazysim_root),
        "--backend",
        str(backend),
        "--model",
        str(model),
        "--x-m",
        str(float(x_m)),
        "--y-m",
        str(float(y_m)),
        "--startup-settle-s",
        str(float(startup_settle_s)),
        "--hover-timeout-s",
        str(float(hover_timeout_s)),
        "--workspace",
        str(workspace),
        "--trace-file",
        str(trace_file),
        "--display",
        str(display),
        "--wind-speed-mps",
        str(float(scenario.wind_speed_mps)),
        "--wind-direction-deg",
        str(float(scenario.wind_direction_deg)),
        "--gust-intensity-mps",
        str(float(scenario.gust_intensity_mps)),
        "--turbulence",
        str(scenario.turbulence),
        "--json",
    ]
    if disturbance_spec_json is not None:
        command.extend(("--disturbance-spec-json", str(disturbance_spec_json)))
    if python_bin is not None:
        command.extend(("--python-bin", str(_normalize_nonresolving_path(Path(python_bin)))))
    if scenario.mass_kg is not None:
        command.extend(("--mass-kg", str(float(scenario.mass_kg))))
    if hover_args:
        command.append("--")
        command.extend(tuple(hover_args))
    return tuple(command)


def _normalize_hover_args(raw_args: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(item) for item in raw_args if str(item).strip())
    if normalized and normalized[0] == "--":
        normalized = normalized[1:]
    return normalized


def run_hover_physical_disturbance_scenario(
    *,
    scenario_name: str,
    crazysim_root: Path,
    output_dir: Path,
    backend: str = "mujoco",
    model: str = "cf2x_T350",
    x_m: float = 0.0,
    y_m: float = 0.0,
    startup_settle_s: float = 3.0,
    hover_timeout_s: float = 30.0,
    workspace: Path = Path("/tmp/twinr-crazysim-physical-workspace"),
    python_bin: Path | None = None,
    display: str = ":0",
    setpoint_period_s: float = 0.1,
    hover_args: Sequence[str] = (),
) -> HoverPhysicalDisturbanceResult:
    """Run one real MuJoCo disturbance scenario and replay its captured outcome."""

    spec = hover_physical_disturbance_spec(scenario_name)
    scenario_dir = output_dir / spec.name
    scenario_dir.mkdir(parents=True, exist_ok=True)
    disturbance_spec_json = None
    if spec.disturbance_plan is not None:
        disturbance_spec_json = write_crazysim_disturbance_plan(
            scenario_dir / "disturbance_plan.json",
            spec.disturbance_plan,
        )
    trace_file = scenario_dir / "hover_trace.jsonl"
    scenario_workspace = Path(workspace).expanduser().resolve() / spec.name
    command = _build_run_hover_sim_command(
        crazysim_root=Path(crazysim_root).expanduser().resolve(),
        backend=str(backend),
        model=str(model),
        x_m=float(x_m),
        y_m=float(y_m),
        startup_settle_s=float(startup_settle_s),
        hover_timeout_s=float(hover_timeout_s),
        workspace=scenario_workspace,
        python_bin=python_bin,
        display=str(display),
        trace_file=trace_file,
        disturbance_spec_json=disturbance_spec_json,
        scenario=spec,
        hover_args=_normalize_hover_args(hover_args),
    )
    completed = subprocess.run(
        command,
        cwd=str(_REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    stdout_text = completed.stdout.strip()
    stderr_text = completed.stderr.strip()
    if not stdout_text:
        raise RuntimeError(
            "run_hover_sim.py produced no JSON output for physical disturbance scenario "
            f"`{spec.name}`: returncode={completed.returncode} stderr={stderr_text!r}"
        )
    try:
        sim_payload = json.loads(stdout_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "run_hover_sim.py did not emit valid JSON for physical disturbance scenario "
            f"`{spec.name}`: returncode={completed.returncode} stdout={stdout_text!r} stderr={stderr_text!r}"
        ) from exc
    if not isinstance(sim_payload, dict):
        raise RuntimeError("run_hover_sim.py physical disturbance payload must be an object")
    hover_report = sim_payload.get("hover_report")
    if not isinstance(hover_report, dict):
        raise RuntimeError("run_hover_sim.py physical disturbance payload is missing `hover_report`")
    report_json = _write_hover_report_json(
        hover_report,
        output_dir=scenario_dir,
        scenario_name=spec.name,
    )
    artifact = load_hover_replay_artifact(
        report_json,
        trace_path=trace_file if trace_file.is_file() else None,
    )
    replay_payload = run_hover_replay(
        artifact,
        runtime_mode=spec.expectation.runtime_mode,
        setpoint_period_s=float(setpoint_period_s),
        min_clearance_m=spec.expectation.min_clearance_m,
        include_phase_events=(
            spec.expectation.guard is not None
            or spec.expectation.recovery is not None
        ),
        include_command_log=spec.expectation.recovery is not None,
    )
    disturbance_runtime_jsonl = _disturbance_runtime_file(sim_payload)
    recovery_window: HoverRecoveryWindow | None = None
    if spec.disturbance_plan is not None:
        if disturbance_runtime_jsonl is None:
            raise RuntimeError(
                f"run_hover_sim.py physical disturbance payload is missing runtime evidence for `{spec.name}`"
            )
        if spec.expectation.recovery is not None:
            disturbance_runtime_events = load_crazysim_disturbance_runtime_events(
                disturbance_runtime_jsonl
            )
            recovery_window = _disturbance_recovery_window(
                scenario=spec,
                disturbance_runtime_events=disturbance_runtime_events,
                replay_phase_events=_coerce_replay_phase_events(replay_payload),
            )
    try:
        evaluation = evaluate_hover_replay_against_expectation(
            replay_payload=replay_payload,
            expectation=spec.expectation,
            target_height_m=_extract_target_height_m(artifact.report_payload),
            recovery_window=recovery_window,
        )
    except Exception as exc:
        raise RuntimeError(
            f"physical disturbance scenario `{spec.name}` could not be evaluated against its replay contract"
        ) from exc
    return HoverPhysicalDisturbanceResult(
        scenario=spec,
        sim_returncode=int(completed.returncode),
        sim_payload=sim_payload,
        replay_payload=replay_payload,
        evaluation=evaluation,
        report_json=report_json,
        trace_file=trace_file if trace_file.is_file() else None,
        disturbance_spec_json=disturbance_spec_json,
        disturbance_runtime_jsonl=disturbance_runtime_jsonl,
        recovery_window=recovery_window,
    )


def run_hover_physical_disturbance_suite(
    *,
    scenario_names: Iterable[str],
    crazysim_root: Path,
    output_dir: Path,
    backend: str = "mujoco",
    model: str = "cf2x_T350",
    x_m: float = 0.0,
    y_m: float = 0.0,
    startup_settle_s: float = 3.0,
    hover_timeout_s: float = 30.0,
    workspace: Path = Path("/tmp/twinr-crazysim-physical-workspace"),
    python_bin: Path | None = None,
    display: str = ":0",
    setpoint_period_s: float = 0.1,
    hover_args: Sequence[str] = (),
) -> tuple[HoverPhysicalDisturbanceResult, ...]:
    """Run a bounded suite of real MuJoCo disturbance cases."""

    return tuple(
        run_hover_physical_disturbance_scenario(
            scenario_name=name,
            crazysim_root=crazysim_root,
            output_dir=output_dir,
            backend=backend,
            model=model,
            x_m=x_m,
            y_m=y_m,
            startup_settle_s=startup_settle_s,
            hover_timeout_s=hover_timeout_s,
            workspace=workspace,
            python_bin=python_bin,
            display=display,
            setpoint_period_s=setpoint_period_s,
            hover_args=hover_args,
        )
        for name in tuple(dict.fromkeys(str(item) for item in scenario_names))
    )


def serialize_hover_physical_disturbance_result(
    result: HoverPhysicalDisturbanceResult,
) -> dict[str, object]:
    """Return one JSON-compatible physical disturbance result."""

    return {
        "scenario": {
            "name": result.scenario.name,
            "description": result.scenario.description,
        },
        "sim_returncode": int(result.sim_returncode),
        "report_json": str(result.report_json),
        "trace_file": None if result.trace_file is None else str(result.trace_file),
        "disturbance_spec_json": (
            None
            if result.disturbance_spec_json is None
            else str(result.disturbance_spec_json)
        ),
        "disturbance_runtime_jsonl": (
            None
            if result.disturbance_runtime_jsonl is None
            else str(result.disturbance_runtime_jsonl)
        ),
        "recovery_window": (
            None
            if result.recovery_window is None
            else {
                "disturbance_start_elapsed_s": result.recovery_window.disturbance_start_elapsed_s,
                "disturbance_end_elapsed_s": result.recovery_window.disturbance_end_elapsed_s,
                "source": result.recovery_window.source,
            }
        ),
        "actual_outcome_class": result.actual_outcome_class,
        "expected_outcome_class": result.scenario.expectation.outcome_class,
        "matches_expectation": result.matches_expectation,
        "failure_tuple": result.evaluation.failure_tuple,
        "matched_failure_substrings": result.evaluation.matched_failure_substrings,
        "missing_failure_substrings": result.evaluation.missing_failure_substrings,
        "contract_failures": result.evaluation.contract_failures,
        "guard_failures": result.evaluation.guard_failures,
        "recovery_failures": result.evaluation.recovery_failures,
        "guard_metrics": None
        if result.evaluation.guard_metrics is None
        else {
            "blocked_count": result.evaluation.guard_metrics.blocked_count,
            "blocked_codes": result.evaluation.guard_metrics.blocked_codes,
            "degraded_count": result.evaluation.guard_metrics.degraded_count,
            "degraded_codes": result.evaluation.guard_metrics.degraded_codes,
            "failures": result.evaluation.guard_metrics.failures,
        },
        "recovery_metrics": None
        if result.evaluation.recovery_metrics is None
        else {
            "max_forward_mps": result.evaluation.recovery_metrics.max_forward_mps,
            "min_forward_mps": result.evaluation.recovery_metrics.min_forward_mps,
            "max_left_mps": result.evaluation.recovery_metrics.max_left_mps,
            "min_left_mps": result.evaluation.recovery_metrics.min_left_mps,
            "max_height_delta_m": result.evaluation.recovery_metrics.max_height_delta_m,
            "min_height_delta_m": result.evaluation.recovery_metrics.min_height_delta_m,
            "recovered_within_window": result.evaluation.recovery_metrics.recovered_within_window,
            "window_source": result.evaluation.recovery_metrics.window_source,
            "failures": result.evaluation.recovery_metrics.failures,
        },
        "sim_payload": result.sim_payload,
    }
