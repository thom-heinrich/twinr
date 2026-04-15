"""Build deterministic CrazySim hover scenarios on top of the replay lane.

This module does not create a second hover implementation. It takes one real
CrazySim or stored hover artifact, applies deterministic telemetry mutations,
and replays those scenarios through the same bounded hover primitive and
stability evaluation used elsewhere in Twinr.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from replay_hover_trace import run_hover_replay
from run_hover_test import HOVER_RUNTIME_MODE_HARDWARE, HOVER_RUNTIME_MODE_SITL
from twinr.hardware.crazyflie_hover_recovery import (
    HoverGuardExpectation,
    HoverGuardMetrics,
    HoverRecoveryExpectation,
    HoverRecoveryMetrics,
    HoverRecoveryWindow,
    evaluate_hover_guard_contract,
    evaluate_hover_recovery,
)
from twinr.hardware.crazyflie_hover_replay import (
    HoverReplayArtifact,
    CrazyflieTelemetrySample,
)


_CLEARANCE_KEYS = (
    "range.front",
    "range.back",
    "range.left",
    "range.right",
    "range.up",
)
_SUPERVISOR_IS_FLYING_MASK = 1 << 4


@dataclass(frozen=True, slots=True)
class HoverSimScenarioExpectation:
    """Describe the replay outcome expected from one scenario."""

    outcome_class: str
    failure_substrings: tuple[str, ...] = ()
    runtime_mode: str = HOVER_RUNTIME_MODE_SITL
    min_clearance_m: float | None = None
    primitive_aborted: bool | None = None
    stable_hover_established: bool | None = None
    touchdown_confirmation_source: str | None = None
    guard: HoverGuardExpectation | None = None
    recovery: HoverRecoveryExpectation | None = None


@dataclass(frozen=True, slots=True)
class HoverSimScenarioSpec:
    """Describe one deterministic CrazySim hover scenario."""

    name: str
    description: str
    expectation: HoverSimScenarioExpectation


@dataclass(frozen=True, slots=True)
class HoverSimScenarioResult:
    """Persist one replayed scenario result plus expectation checks."""

    scenario: HoverSimScenarioSpec
    replay_payload: dict[str, object]
    failure_tuple: tuple[str, ...]
    matched_failure_substrings: tuple[str, ...]
    missing_failure_substrings: tuple[str, ...]
    contract_failures: tuple[str, ...]
    guard_failures: tuple[str, ...]
    recovery_failures: tuple[str, ...]
    guard_metrics: HoverGuardMetrics | None = None
    recovery_metrics: HoverRecoveryMetrics | None = None

    @property
    def actual_outcome_class(self) -> str:
        replay = self.replay_payload.get("replay")
        if not isinstance(replay, Mapping):
            raise ValueError("scenario replay payload is missing `replay`")
        value = replay.get("outcome_class")
        return str(value)

    @property
    def matches_expectation(self) -> bool:
        expectation = self.scenario.expectation
        return (
            self.actual_outcome_class == expectation.outcome_class
            and not self.missing_failure_substrings
            and not self.contract_failures
            and not self.guard_failures
            and not self.recovery_failures
        )


@dataclass(frozen=True, slots=True)
class HoverScenarioContractEvaluation:
    """Capture one replay-vs-expectation contract evaluation."""

    failure_tuple: tuple[str, ...]
    matched_failure_substrings: tuple[str, ...]
    missing_failure_substrings: tuple[str, ...]
    contract_failures: tuple[str, ...]
    guard_failures: tuple[str, ...]
    recovery_failures: tuple[str, ...]
    guard_metrics: HoverGuardMetrics | None = None
    recovery_metrics: HoverRecoveryMetrics | None = None


def available_hover_sim_scenarios() -> tuple[HoverSimScenarioSpec, ...]:
    """Return the supported deterministic CrazySim hover scenarios."""

    return (
        HoverSimScenarioSpec(
            name="baseline_nominal",
            description="Replay the unmodified CrazySim hover artifact under the SITL hover contract.",
            expectation=HoverSimScenarioExpectation(
                outcome_class="bounded_hover_ok",
                runtime_mode=HOVER_RUNTIME_MODE_SITL,
                primitive_aborted=False,
                stable_hover_established=True,
                touchdown_confirmation_source="range_only_sitl",
            ),
        ),
        HoverSimScenarioSpec(
            name="transient_forward_drift_recovery",
            description="Inject one bounded forward drift impulse that should cause the outer loop to command a bounded backward correction and then settle again.",
            expectation=HoverSimScenarioExpectation(
                outcome_class="bounded_hover_ok",
                runtime_mode=HOVER_RUNTIME_MODE_SITL,
                primitive_aborted=False,
                stable_hover_established=True,
                touchdown_confirmation_source="range_only_sitl",
                guard=HoverGuardExpectation(must_block=False),
                recovery=HoverRecoveryExpectation(
                    disturbance_start_progress=0.25,
                    disturbance_end_progress=0.40,
                    max_recovery_delay_s=0.80,
                    settle_height_error_abs_max_m=0.02,
                    settle_velocity_abs_max_mps=0.08,
                    settle_required_commands=1,
                    forward_direction="negative",
                    min_forward_abs_mps=0.05,
                ),
            ),
        ),
        HoverSimScenarioSpec(
            name="transient_left_drift_recovery",
            description="Inject one bounded left/right drift impulse that should cause the outer loop to command an opposing lateral correction and then settle again.",
            expectation=HoverSimScenarioExpectation(
                outcome_class="bounded_hover_ok",
                runtime_mode=HOVER_RUNTIME_MODE_SITL,
                primitive_aborted=False,
                stable_hover_established=True,
                touchdown_confirmation_source="range_only_sitl",
                guard=HoverGuardExpectation(must_block=False),
                recovery=HoverRecoveryExpectation(
                    disturbance_start_progress=0.25,
                    disturbance_end_progress=0.40,
                    max_recovery_delay_s=0.80,
                    settle_height_error_abs_max_m=0.02,
                    settle_velocity_abs_max_mps=0.08,
                    settle_required_commands=1,
                    left_direction="positive",
                    min_left_abs_mps=0.05,
                ),
            ),
        ),
        HoverSimScenarioSpec(
            name="transient_height_drop_recovery",
            description="Inject one bounded height underestimate that should cause the outer loop to raise the commanded hover height briefly and then settle again.",
            expectation=HoverSimScenarioExpectation(
                outcome_class="bounded_hover_ok",
                runtime_mode=HOVER_RUNTIME_MODE_SITL,
                primitive_aborted=False,
                stable_hover_established=True,
                touchdown_confirmation_source="range_only_sitl",
                guard=HoverGuardExpectation(must_block=False),
                recovery=HoverRecoveryExpectation(
                    disturbance_start_progress=0.25,
                    disturbance_end_progress=0.40,
                    max_recovery_delay_s=0.80,
                    settle_required_commands=1,
                    height_direction="positive",
                    min_height_delta_m=0.01,
                ),
            ),
        ),
        HoverSimScenarioSpec(
            name="persistent_forward_drift_abort",
            description="Inject a sustained forward drift bias so the same axis that can recover transiently must now trip the hover guard and abort.",
            expectation=HoverSimScenarioExpectation(
                outcome_class="unstable_hover_aborted",
                failure_substrings=(
                    "horizontal speed",
                    "xy drift",
                ),
                runtime_mode=HOVER_RUNTIME_MODE_SITL,
                primitive_aborted=True,
                touchdown_confirmation_source="range_only_sitl",
                guard=HoverGuardExpectation(
                    must_block=True,
                    required_blocked_codes=("speed",),
                ),
            ),
        ),
        HoverSimScenarioSpec(
            name="drift_bias",
            description="Inject a growing lateral pose/velocity bias after takeoff so the hover guard must abort on bounded lateral motion.",
            expectation=HoverSimScenarioExpectation(
                outcome_class="takeoff_failed",
                failure_substrings=("horizontal speed",),
                runtime_mode=HOVER_RUNTIME_MODE_SITL,
                primitive_aborted=True,
                stable_hover_established=False,
                touchdown_confirmation_source="range_only_sitl",
                guard=HoverGuardExpectation(
                    must_block=True,
                    required_blocked_codes=("speed",),
                ),
            ),
        ),
        HoverSimScenarioSpec(
            name="flow_dropout",
            description="Inject a hardware-style optical-flow collapse so the hover guard must reject the lateral state.",
            expectation=HoverSimScenarioExpectation(
                outcome_class="unstable_hover_aborted",
                failure_substrings=("optical-flow quality",),
                runtime_mode=HOVER_RUNTIME_MODE_HARDWARE,
                primitive_aborted=True,
                stable_hover_established=True,
                touchdown_confirmation_source="range+supervisor",
                guard=HoverGuardExpectation(
                    must_block=True,
                    required_blocked_codes=("flow_untrusted",),
                ),
            ),
        ),
        HoverSimScenarioSpec(
            name="zrange_outlier",
            description="Inject a downward-range surface switch so trusted-height disagreement must abort the hover lane.",
            expectation=HoverSimScenarioExpectation(
                outcome_class="takeoff_failed",
                failure_substrings=("height telemetry became untrusted",),
                runtime_mode=HOVER_RUNTIME_MODE_SITL,
                primitive_aborted=True,
                stable_hover_established=False,
                touchdown_confirmation_source="range_only_sitl",
            ),
        ),
        HoverSimScenarioSpec(
            name="attitude_spike",
            description="Inject an abrupt roll/pitch spike after takeoff so the hover guard must abort on excessive attitude.",
            expectation=HoverSimScenarioExpectation(
                outcome_class="takeoff_failed",
                failure_substrings=("roll reached", "pitch reached"),
                runtime_mode=HOVER_RUNTIME_MODE_SITL,
                primitive_aborted=True,
                stable_hover_established=False,
                touchdown_confirmation_source="range_only_sitl",
            ),
        ),
        HoverSimScenarioSpec(
            name="wall_proximity",
            description="Inject close frontal and lateral obstacle readings so stability evaluation must classify the hover as unsafe.",
            expectation=HoverSimScenarioExpectation(
                outcome_class="unstable_hover_aborted",
                failure_substrings=(
                    "front clearance reached",
                    "back clearance reached",
                    "left clearance reached",
                    "right clearance reached",
                ),
                runtime_mode=HOVER_RUNTIME_MODE_SITL,
                min_clearance_m=0.35,
                primitive_aborted=False,
                stable_hover_established=True,
                touchdown_confirmation_source="range_only_sitl",
            ),
        ),
    )


def hover_sim_scenario_names() -> tuple[str, ...]:
    """Return the supported scenario names."""

    return tuple(spec.name for spec in available_hover_sim_scenarios())


def hover_sim_scenario_spec(name: str) -> HoverSimScenarioSpec:
    """Return one validated scenario spec by name."""

    normalized = str(name).strip()
    for spec in available_hover_sim_scenarios():
        if spec.name == normalized:
            return spec
    allowed = ", ".join(hover_sim_scenario_names())
    raise ValueError(f"unsupported hover sim scenario `{name}`; choose one of: {allowed}")


def _supervisor_is_flying(value: float | int | None) -> bool:
    if value is None:
        return False
    return (int(value) & (1 << 4)) != 0


def _first_airborne_timestamp_ms(samples: Sequence[CrazyflieTelemetrySample]) -> int:
    for sample in samples:
        if _supervisor_is_flying(sample.values.get("supervisor.info")):
            return int(sample.timestamp_ms)
        z_estimate = sample.values.get("stateEstimate.z")
        if z_estimate is not None and float(z_estimate) >= 0.08:
            return int(sample.timestamp_ms)
        zrange = sample.values.get("range.zrange")
        if zrange is not None and float(zrange) >= 80.0:
            return int(sample.timestamp_ms)
    raise ValueError("hover sim scenario needs one airborne sample but the artifact never reaches takeoff-active height")


def _last_timestamp_ms(samples: Sequence[CrazyflieTelemetrySample]) -> int:
    if not samples:
        raise ValueError("hover sim scenario artifact contains no telemetry samples")
    return max(int(sample.timestamp_ms) for sample in samples)


def _last_active_flight_timestamp_ms(samples: Sequence[CrazyflieTelemetrySample]) -> int:
    last_active_timestamp_ms: int | None = None
    for sample in samples:
        if _supervisor_is_flying(sample.values.get("supervisor.info")):
            last_active_timestamp_ms = int(sample.timestamp_ms)
            continue
        z_estimate = sample.values.get("stateEstimate.z")
        if z_estimate is not None and float(z_estimate) >= 0.05:
            last_active_timestamp_ms = int(sample.timestamp_ms)
            continue
        zrange = sample.values.get("range.zrange")
        if zrange is not None and float(zrange) >= 50.0:
            last_active_timestamp_ms = int(sample.timestamp_ms)
    if last_active_timestamp_ms is None:
        raise ValueError("hover sim scenario needs one active-flight sample but the artifact never leaves the ground")
    return last_active_timestamp_ms


def _sample_progress(sample: CrazyflieTelemetrySample, *, start_ms: int, end_ms: int) -> float:
    if int(sample.timestamp_ms) <= start_ms:
        return 0.0
    if end_ms <= start_ms:
        return 1.0
    return max(0.0, min(1.0, (int(sample.timestamp_ms) - start_ms) / float(end_ms - start_ms)))


def _replace_sample_values(
    sample: CrazyflieTelemetrySample,
    updates: Mapping[str, float | int | None],
) -> CrazyflieTelemetrySample:
    merged = dict(sample.values)
    merged.update({str(key): value for key, value in updates.items()})
    return replace(sample, values=merged)


def _coerce_report_float(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if value is None:
        raise ValueError(f"hover sim scenario payload is missing `{key}`")
    if not isinstance(value, (int, float, str)):
        raise ValueError(f"hover sim scenario payload `{key}` must be numeric, got {type(value).__name__}")
    return float(value)


def evaluate_hover_replay_against_expectation(
    *,
    replay_payload: Mapping[str, object],
    expectation: HoverSimScenarioExpectation,
    target_height_m: float,
    recovery_window: HoverRecoveryWindow | None = None,
) -> HoverScenarioContractEvaluation:
    """Evaluate one replay payload against a shared hover scenario expectation."""

    replay = replay_payload.get("replay")
    if not isinstance(replay, Mapping):
        raise ValueError("hover replay payload is missing a `replay` object")
    failures_raw = replay.get("failures")
    if not isinstance(failures_raw, Sequence) or isinstance(failures_raw, (str, bytes, bytearray)):
        raise ValueError("hover replay payload is missing a failure sequence")
    failure_tuple = tuple(str(item) for item in failures_raw)
    matched = tuple(
        substring
        for substring in expectation.failure_substrings
        if any(substring in failure for failure in failure_tuple)
    )
    missing = tuple(
        substring
        for substring in expectation.failure_substrings
        if substring not in matched
    )
    primitive_outcome_raw = replay.get("primitive_outcome")
    if not isinstance(primitive_outcome_raw, Mapping):
        raise ValueError("hover replay payload is missing primitive outcome details")
    contract_failures: list[str] = []
    primitive_aborted = bool(primitive_outcome_raw.get("aborted"))
    stable_hover_established = bool(primitive_outcome_raw.get("stable_hover_established"))
    touchdown_confirmation_source = primitive_outcome_raw.get("touchdown_confirmation_source")
    if (
        expectation.primitive_aborted is not None
        and primitive_aborted != expectation.primitive_aborted
    ):
        contract_failures.append(
            "primitive aborted contract mismatch: "
            f"expected {expectation.primitive_aborted} but observed {primitive_aborted}"
        )
    if (
        expectation.stable_hover_established is not None
        and stable_hover_established != expectation.stable_hover_established
    ):
        contract_failures.append(
            "stable_hover_established contract mismatch: "
            f"expected {expectation.stable_hover_established} but observed {stable_hover_established}"
        )
    if (
        expectation.touchdown_confirmation_source is not None
        and str(touchdown_confirmation_source) != expectation.touchdown_confirmation_source
    ):
        contract_failures.append(
            "touchdown_confirmation_source contract mismatch: "
            f"expected {expectation.touchdown_confirmation_source!r} "
            f"but observed {touchdown_confirmation_source!r}"
        )
    phase_events: tuple[Mapping[str, object], ...] | None = None
    guard_metrics: HoverGuardMetrics | None = None
    guard_failures: tuple[str, ...] = ()
    if expectation.guard is not None:
        phase_events_raw = replay.get("phase_events")
        if (
            not isinstance(phase_events_raw, Sequence)
            or isinstance(phase_events_raw, (str, bytes, bytearray))
        ):
            raise ValueError("hover replay payload is missing guard phase events")
        phase_events = tuple(item for item in phase_events_raw if isinstance(item, Mapping))
        guard_metrics = evaluate_hover_guard_contract(
            phase_events=phase_events,
            expectation=expectation.guard,
        )
        guard_failures = guard_metrics.failures
    recovery_metrics: HoverRecoveryMetrics | None = None
    recovery_failures: tuple[str, ...] = ()
    if expectation.recovery is not None:
        if phase_events is None:
            phase_events_raw = replay.get("phase_events")
            if (
                not isinstance(phase_events_raw, Sequence)
                or isinstance(phase_events_raw, (str, bytes, bytearray))
            ):
                raise ValueError("hover replay payload is missing recovery phase events")
            phase_events = tuple(item for item in phase_events_raw if isinstance(item, Mapping))
        command_log_raw = replay.get("command_log")
        if (
            not isinstance(command_log_raw, Sequence)
            or isinstance(command_log_raw, (str, bytes, bytearray))
        ):
            raise ValueError("hover replay payload is missing recovery command log")
        try:
            recovery_metrics = evaluate_hover_recovery(
                command_log=tuple(item for item in command_log_raw if isinstance(item, Mapping)),
                phase_events=phase_events,
                target_height_m=float(target_height_m),
                expectation=expectation.recovery,
                disturbance_window=recovery_window,
            )
        except ValueError as exc:
            recovery_failures = (str(exc),)
        else:
            recovery_failures = recovery_metrics.failures
    return HoverScenarioContractEvaluation(
        failure_tuple=failure_tuple,
        matched_failure_substrings=matched,
        missing_failure_substrings=missing,
        contract_failures=tuple(contract_failures),
        guard_failures=guard_failures,
        recovery_failures=recovery_failures,
        guard_metrics=guard_metrics,
        recovery_metrics=recovery_metrics,
    )


def _mutate_samples(
    samples: Sequence[CrazyflieTelemetrySample],
    *,
    mutation: Callable[[CrazyflieTelemetrySample, float], Mapping[str, float | int | None] | None],
    end_ms: int | None = None,
) -> tuple[CrazyflieTelemetrySample, ...]:
    airborne_start_ms = _first_airborne_timestamp_ms(samples)
    bounded_end_ms = _last_timestamp_ms(samples) if end_ms is None else int(end_ms)
    if bounded_end_ms <= airborne_start_ms:
        raise ValueError(
            "hover sim scenario needs the active-flight window to extend past the airborne start timestamp"
        )
    mutated: list[CrazyflieTelemetrySample] = []
    for sample in samples:
        progress = _sample_progress(sample, start_ms=airborne_start_ms, end_ms=bounded_end_ms)
        updates = mutation(sample, progress)
        if updates:
            mutated.append(_replace_sample_values(sample, updates))
        else:
            mutated.append(sample)
    return tuple(mutated)


def _scenario_baseline_nominal(samples: Sequence[CrazyflieTelemetrySample]) -> tuple[CrazyflieTelemetrySample, ...]:
    return tuple(samples)


def _scenario_transient_forward_drift_recovery(
    samples: Sequence[CrazyflieTelemetrySample],
) -> tuple[CrazyflieTelemetrySample, ...]:
    def _mutation(sample: CrazyflieTelemetrySample, progress: float) -> Mapping[str, float | int | None] | None:
        if progress < 0.25 or progress > 0.40:
            return None
        updates: dict[str, float | int | None] = {}
        if sample.block_name == "hover-attitude":
            x_raw = sample.values.get("stateEstimate.x")
            if x_raw is not None:
                updates["stateEstimate.x"] = float(x_raw) + 0.12
        if sample.block_name == "hover-velocity":
            updates["stateEstimate.vx"] = 0.08
        return updates or None

    return _mutate_samples(
        samples,
        mutation=_mutation,
        end_ms=_last_active_flight_timestamp_ms(samples),
    )


def _scenario_transient_left_drift_recovery(
    samples: Sequence[CrazyflieTelemetrySample],
) -> tuple[CrazyflieTelemetrySample, ...]:
    def _mutation(sample: CrazyflieTelemetrySample, progress: float) -> Mapping[str, float | int | None] | None:
        if progress < 0.25 or progress > 0.40:
            return None
        updates: dict[str, float | int | None] = {}
        if sample.block_name == "hover-attitude":
            y_raw = sample.values.get("stateEstimate.y")
            if y_raw is not None:
                updates["stateEstimate.y"] = float(y_raw) - 0.12
        if sample.block_name == "hover-velocity":
            updates["stateEstimate.vy"] = -0.08
        return updates or None

    return _mutate_samples(
        samples,
        mutation=_mutation,
        end_ms=_last_active_flight_timestamp_ms(samples),
    )


def _scenario_transient_height_drop_recovery(
    samples: Sequence[CrazyflieTelemetrySample],
) -> tuple[CrazyflieTelemetrySample, ...]:
    def _mutation(sample: CrazyflieTelemetrySample, progress: float) -> Mapping[str, float | int | None] | None:
        if progress < 0.35 or progress > 0.50:
            return None
        updates: dict[str, float | int | None] = {}
        if sample.block_name == "hover-attitude":
            z_raw = sample.values.get("stateEstimate.z")
            if z_raw is not None:
                updates["stateEstimate.z"] = max(0.02, float(z_raw) - 0.03)
        if sample.block_name == "hover-sensors":
            zrange_raw = sample.values.get("range.zrange")
            if isinstance(zrange_raw, (int, float)) and int(zrange_raw) != 0:
                updates["range.zrange"] = max(20, int(zrange_raw) - 30)
        return updates or None

    return _mutate_samples(
        samples,
        mutation=_mutation,
        end_ms=_last_active_flight_timestamp_ms(samples),
    )


def _scenario_persistent_forward_drift_abort(
    samples: Sequence[CrazyflieTelemetrySample],
) -> tuple[CrazyflieTelemetrySample, ...]:
    def _mutation(sample: CrazyflieTelemetrySample, progress: float) -> Mapping[str, float | int | None] | None:
        if progress < 0.25:
            return None
        updates: dict[str, float | int | None] = {}
        if sample.block_name == "hover-attitude":
            x_raw = sample.values.get("stateEstimate.x")
            if x_raw is not None:
                updates["stateEstimate.x"] = float(x_raw) + 0.30 + (0.18 * progress)
        if sample.block_name == "hover-velocity":
            updates["stateEstimate.vx"] = 0.52
        return updates or None

    return _mutate_samples(
        samples,
        mutation=_mutation,
        end_ms=_last_active_flight_timestamp_ms(samples),
    )


def _scenario_drift_bias(samples: Sequence[CrazyflieTelemetrySample]) -> tuple[CrazyflieTelemetrySample, ...]:
    def _mutation(sample: CrazyflieTelemetrySample, progress: float) -> Mapping[str, float | int | None] | None:
        if progress <= 0.05:
            return None
        scaled_progress = min(1.0, (progress - 0.05) / 0.95)
        updates: dict[str, float | int | None] = {}
        if sample.block_name == "hover-attitude":
            x_raw = sample.values.get("stateEstimate.x")
            y_raw = sample.values.get("stateEstimate.y")
            if x_raw is not None:
                updates["stateEstimate.x"] = float(x_raw) + (0.40 * scaled_progress)
            if y_raw is not None:
                updates["stateEstimate.y"] = float(y_raw) + (0.10 * scaled_progress)
        if sample.block_name == "hover-velocity":
            updates["stateEstimate.vx"] = 0.52
            updates["stateEstimate.vy"] = 0.12
        return updates or None

    return _mutate_samples(samples, mutation=_mutation)


def _scenario_flow_dropout(samples: Sequence[CrazyflieTelemetrySample]) -> tuple[CrazyflieTelemetrySample, ...]:
    z_estimate_by_timestamp_ms = {
        int(sample.timestamp_ms): float(z_estimate)
        for sample in samples
        if sample.block_name == "hover-attitude"
        for z_estimate in (sample.values.get("stateEstimate.z"),)
        if z_estimate is not None
    }

    def _mutation(sample: CrazyflieTelemetrySample, progress: float) -> Mapping[str, float | int | None] | None:
        if sample.block_name != "hover-sensors":
            return None
        updates: dict[str, float | int | None] = {
            "motion.squal": 80 if progress <= 0.45 else 0,
        }
        zrange_value = sample.values.get("range.zrange")
        if zrange_value in {None, 0}:
            z_estimate = z_estimate_by_timestamp_ms.get(int(sample.timestamp_ms))
            if z_estimate is not None:
                updates["range.zrange"] = int(max(0.0, z_estimate) * 1000.0)
        effective_zrange = updates.get("range.zrange", zrange_value)
        supervisor_info = sample.values.get("supervisor.info")
        if supervisor_info is not None and effective_zrange is not None and int(effective_zrange) <= 30:
            updates["supervisor.info"] = int(supervisor_info) & ~_SUPERVISOR_IS_FLYING_MASK
        return updates

    return _mutate_samples(samples, mutation=_mutation)


def _scenario_zrange_outlier(samples: Sequence[CrazyflieTelemetrySample]) -> tuple[CrazyflieTelemetrySample, ...]:
    def _mutation(sample: CrazyflieTelemetrySample, progress: float) -> Mapping[str, float | int | None] | None:
        if sample.block_name != "hover-sensors" or progress <= 0.05:
            return None
        z_estimate = sample.values.get("stateEstimate.z")
        if z_estimate is None:
            return {"range.zrange": 980}
        return {"range.zrange": int((float(z_estimate) + 0.75) * 1000.0)}

    return _mutate_samples(samples, mutation=_mutation)


def _scenario_attitude_spike(samples: Sequence[CrazyflieTelemetrySample]) -> tuple[CrazyflieTelemetrySample, ...]:
    def _mutation(sample: CrazyflieTelemetrySample, progress: float) -> Mapping[str, float | int | None] | None:
        if sample.block_name != "hover-attitude" or progress <= 0.05:
            return None
        return {
            "stabilizer.roll": 12.5,
            "stabilizer.pitch": -13.0,
        }

    return _mutate_samples(samples, mutation=_mutation)


def _scenario_wall_proximity(samples: Sequence[CrazyflieTelemetrySample]) -> tuple[CrazyflieTelemetrySample, ...]:
    def _mutation(sample: CrazyflieTelemetrySample, progress: float) -> Mapping[str, float | int | None] | None:
        if sample.block_name != "hover-clearance" or progress <= 0.15:
            return None
        return {
            "range.front": 120,
            "range.left": 180,
            "range.right": 220,
            "range.back": 260,
            "range.up": 450,
        }

    return _mutate_samples(samples, mutation=_mutation)


_SCENARIO_MUTATORS: dict[str, Callable[[Sequence[CrazyflieTelemetrySample]], tuple[CrazyflieTelemetrySample, ...]]] = {
    "baseline_nominal": _scenario_baseline_nominal,
    "transient_forward_drift_recovery": _scenario_transient_forward_drift_recovery,
    "transient_left_drift_recovery": _scenario_transient_left_drift_recovery,
    "transient_height_drop_recovery": _scenario_transient_height_drop_recovery,
    "persistent_forward_drift_abort": _scenario_persistent_forward_drift_abort,
    "drift_bias": _scenario_drift_bias,
    "flow_dropout": _scenario_flow_dropout,
    "zrange_outlier": _scenario_zrange_outlier,
    "attitude_spike": _scenario_attitude_spike,
    "wall_proximity": _scenario_wall_proximity,
}


def apply_hover_sim_scenario(
    artifact: HoverReplayArtifact,
    *,
    scenario_name: str,
) -> HoverReplayArtifact:
    """Return one mutated hover artifact for the selected scenario."""

    spec = hover_sim_scenario_spec(scenario_name)
    mutator = _SCENARIO_MUTATORS.get(spec.name)
    if mutator is None:
        raise ValueError(f"hover sim scenario `{spec.name}` has no mutator")
    mutated_samples = mutator(artifact.telemetry_samples)
    if len(mutated_samples) != len(artifact.telemetry_samples):
        raise ValueError(
            f"hover sim scenario `{spec.name}` changed the telemetry sample count "
            f"from {len(artifact.telemetry_samples)} to {len(mutated_samples)}"
        )
    report_payload = dict(artifact.report_payload)
    report_payload["telemetry"] = tuple(
        {
            "timestamp_ms": int(sample.timestamp_ms),
            "block_name": str(sample.block_name),
            "values": dict(sample.values),
            "received_monotonic_s": sample.received_monotonic_s,
        }
        for sample in mutated_samples
    )
    return HoverReplayArtifact(
        report_path=artifact.report_path,
        report_payload=report_payload,
        telemetry_samples=mutated_samples,
        available_blocks=artifact.available_blocks,
        skipped_blocks=artifact.skipped_blocks,
        trace_path=artifact.trace_path,
        trace_events=artifact.trace_events,
    )


def run_hover_sim_scenario(
    artifact: HoverReplayArtifact,
    *,
    scenario_name: str,
    setpoint_period_s: float = 0.1,
) -> HoverSimScenarioResult:
    """Replay one deterministic hover scenario through the shared primitive."""

    spec = hover_sim_scenario_spec(scenario_name)
    scenario_artifact = apply_hover_sim_scenario(artifact, scenario_name=spec.name)
    replay_payload = run_hover_replay(
        scenario_artifact,
        runtime_mode=spec.expectation.runtime_mode,
        setpoint_period_s=float(setpoint_period_s),
        min_clearance_m=spec.expectation.min_clearance_m,
        include_phase_events=(
            spec.expectation.recovery is not None
            or spec.expectation.guard is not None
        ),
        include_command_log=spec.expectation.recovery is not None,
    )
    evaluation = evaluate_hover_replay_against_expectation(
        replay_payload=replay_payload,
        expectation=spec.expectation,
        target_height_m=_coerce_report_float(scenario_artifact.report_payload, "height_m"),
    )
    return HoverSimScenarioResult(
        scenario=spec,
        replay_payload=replay_payload,
        failure_tuple=evaluation.failure_tuple,
        matched_failure_substrings=evaluation.matched_failure_substrings,
        missing_failure_substrings=evaluation.missing_failure_substrings,
        contract_failures=evaluation.contract_failures,
        guard_failures=evaluation.guard_failures,
        guard_metrics=evaluation.guard_metrics,
        recovery_failures=evaluation.recovery_failures,
        recovery_metrics=evaluation.recovery_metrics,
    )


def run_hover_sim_scenario_suite(
    artifact: HoverReplayArtifact,
    *,
    scenario_names: Iterable[str],
    setpoint_period_s: float = 0.1,
) -> tuple[HoverSimScenarioResult, ...]:
    """Replay a bounded suite of deterministic hover scenarios."""

    return tuple(
        run_hover_sim_scenario(
            artifact,
            scenario_name=name,
            setpoint_period_s=setpoint_period_s,
        )
        for name in tuple(dict.fromkeys(str(item) for item in scenario_names))
    )


def write_hover_replay_artifact_json(artifact: HoverReplayArtifact, path: Path) -> Path:
    """Persist one replay artifact as a standalone hover report JSON file."""

    payload = {"report": dict(artifact.report_payload)}
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return resolved


def serialize_hover_sim_scenario_result(result: HoverSimScenarioResult) -> dict[str, object]:
    """Return one JSON-serializable scenario result payload."""

    return {
        "scenario": {
            "name": result.scenario.name,
            "description": result.scenario.description,
            "expectation": asdict(result.scenario.expectation),
        },
        "actual_outcome_class": result.actual_outcome_class,
        "matches_expectation": result.matches_expectation,
        "failures": result.failure_tuple,
        "matched_failure_substrings": result.matched_failure_substrings,
        "missing_failure_substrings": result.missing_failure_substrings,
        "contract_failures": result.contract_failures,
        "guard_failures": result.guard_failures,
        "guard_metrics": None if result.guard_metrics is None else asdict(result.guard_metrics),
        "recovery_failures": result.recovery_failures,
        "recovery_metrics": None if result.recovery_metrics is None else asdict(result.recovery_metrics),
        "replay": result.replay_payload["replay"],
    }
