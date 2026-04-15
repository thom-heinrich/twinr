"""Evaluate bounded hover-recovery behavior from replayed command and phase logs.

This module does not simulate a second flight stack. It inspects the existing
replay artifacts emitted by Twinr's bounded Crazyflie hover lane and answers a
specific question: when a deterministic transient disturbance is injected into
the telemetry, did the host-side outer loop issue corrective commands in the
expected direction and return to a settled hover command within a bounded time?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence


RecoveryDirection = Literal["positive", "negative"]
_RECOVERY_DELAY_EPSILON_S = 1e-9


@dataclass(frozen=True, slots=True)
class HoverRecoveryExpectation:
    """Describe the expected recovery behavior for one transient disturbance."""

    disturbance_start_progress: float
    disturbance_end_progress: float
    max_recovery_delay_s: float
    settle_velocity_abs_max_mps: float = 0.03
    settle_height_error_abs_max_m: float = 0.01
    settle_required_commands: int = 3
    forward_direction: RecoveryDirection | None = None
    min_forward_abs_mps: float | None = None
    left_direction: RecoveryDirection | None = None
    min_left_abs_mps: float | None = None
    height_direction: RecoveryDirection | None = None
    min_height_delta_m: float | None = None


@dataclass(frozen=True, slots=True)
class HoverRecoveryMetrics:
    """Capture the observed recovery metrics for one replayed disturbance."""

    disturbance_start_elapsed_s: float
    disturbance_end_elapsed_s: float
    response_end_elapsed_s: float
    max_forward_mps: float
    min_forward_mps: float
    max_left_mps: float
    min_left_mps: float
    max_height_delta_m: float
    min_height_delta_m: float
    settle_elapsed_s: float | None
    recovery_delay_s: float | None
    recovered_within_window: bool
    window_source: str
    failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HoverRecoveryWindow:
    """Describe one exact disturbance window inside the replayed hover command log."""

    disturbance_start_elapsed_s: float
    disturbance_end_elapsed_s: float
    source: str = "explicit"


@dataclass(frozen=True, slots=True)
class HoverGuardExpectation:
    """Describe the expected guard behavior for one replayed disturbance."""

    must_block: bool | None = None
    required_blocked_codes: tuple[str, ...] = ()
    forbidden_blocked_codes: tuple[str, ...] = ()
    required_degraded_codes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class HoverGuardMetrics:
    """Capture the guard-state contract observed during one replayed disturbance."""

    degraded_count: int
    blocked_count: int
    degraded_codes: tuple[str, ...]
    blocked_codes: tuple[str, ...]
    degraded_phases: tuple[str, ...]
    blocked_phases: tuple[str, ...]
    first_degraded_elapsed_s: float | None
    first_blocked_elapsed_s: float | None
    failures: tuple[str, ...]


def _coerce_float(mapping: Mapping[str, object], key: str) -> float:
    value = mapping.get(key)
    if value is None:
        raise ValueError(f"recovery input is missing `{key}`")
    if not isinstance(value, (int, float, str)):
        raise ValueError(f"recovery input `{key}` must be numeric, got {type(value).__name__}")
    return float(value)


def _coerce_string_tuple(mapping: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = mapping.get(key)
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"recovery input `{key}` must be a sequence of strings")
    return tuple(str(item) for item in value)


def _takeoff_done_elapsed_s(phase_events: Sequence[Mapping[str, object]]) -> float:
    for event in phase_events:
        if str(event.get("phase")) == "hover_primitive_takeoff" and str(event.get("status")) == "done":
            return _coerce_float(event, "elapsed_s")
    raise ValueError("hover recovery evaluation requires `hover_primitive_takeoff` `done` in the phase log")


def _phase_elapsed_s(
    phase_events: Sequence[Mapping[str, object]],
    *,
    phase: str,
    status: str,
) -> float | None:
    for event in phase_events:
        if str(event.get("phase")) == phase and str(event.get("status")) == status:
            return _coerce_float(event, "elapsed_s")
    return None


def _hover_control_end_elapsed_s(phase_events: Sequence[Mapping[str, object]]) -> float:
    land_begin_elapsed_s = _phase_elapsed_s(
        phase_events,
        phase="hover_primitive_land",
        status="begin",
    )
    if land_begin_elapsed_s is not None:
        return land_begin_elapsed_s
    hold_done_elapsed_s = _phase_elapsed_s(
        phase_events,
        phase="hover_primitive_hold",
        status="done",
    )
    if hold_done_elapsed_s is not None:
        return hold_done_elapsed_s
    abort_begin_elapsed_s = _phase_elapsed_s(
        phase_events,
        phase="hover_primitive_abort",
        status="begin",
    )
    if abort_begin_elapsed_s is not None:
        return abort_begin_elapsed_s
    raise ValueError(
        "hover recovery evaluation requires either `hover_primitive_hold` `done` "
        "`hover_primitive_land` `begin`, or `hover_primitive_abort` `begin` in the phase log"
    )


def _bounded_progress(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _expected_signed_peak(
    *,
    direction: RecoveryDirection,
    positive_peak: float,
    negative_peak: float,
) -> float:
    if direction == "positive":
        return positive_peak
    return abs(negative_peak)


def _matches_expected_axis_command(
    *,
    value: float,
    direction: RecoveryDirection | None,
    minimum_magnitude: float | None,
) -> bool:
    if direction is None or minimum_magnitude is None:
        return False
    if direction == "positive":
        return value >= float(minimum_magnitude)
    return value <= -float(minimum_magnitude)


def _command_matches_expected_recovery(
    *,
    command: Mapping[str, object],
    target_height_m: float,
    expectation: HoverRecoveryExpectation,
) -> bool:
    forward_mps = _coerce_float(command, "vx_mps")
    left_mps = _coerce_float(command, "vy_mps")
    height_delta_m = _coerce_float(command, "height_m") - float(target_height_m)
    return any(
        (
            _matches_expected_axis_command(
                value=forward_mps,
                direction=expectation.forward_direction,
                minimum_magnitude=expectation.min_forward_abs_mps,
            ),
            _matches_expected_axis_command(
                value=left_mps,
                direction=expectation.left_direction,
                minimum_magnitude=expectation.min_left_abs_mps,
            ),
            _matches_expected_axis_command(
                value=height_delta_m,
                direction=expectation.height_direction,
                minimum_magnitude=expectation.min_height_delta_m,
            ),
        )
    )


def evaluate_hover_guard_contract(
    *,
    phase_events: Sequence[Mapping[str, object]],
    expectation: HoverGuardExpectation,
) -> HoverGuardMetrics:
    """Evaluate whether guard events match the expected recover-vs-abort contract."""

    degraded_codes: list[str] = []
    blocked_codes: list[str] = []
    degraded_phases: list[str] = []
    blocked_phases: list[str] = []
    first_degraded_elapsed_s: float | None = None
    first_blocked_elapsed_s: float | None = None

    for event in phase_events:
        if str(event.get("phase")) not in {
            "hover_primitive_stability_guard",
            "hover_primitive_trim_guard",
        }:
            continue
        status = str(event.get("status"))
        if status not in {"degraded", "blocked"}:
            continue
        data = event.get("data")
        if not isinstance(data, Mapping):
            raise ValueError("hover guard evaluation requires `data` on every guard phase event")
        failure_codes = _coerce_string_tuple(data, "failure_codes")
        if not failure_codes:
            raise ValueError("hover guard evaluation requires non-empty `failure_codes` on guard phase events")
        guard_phase = str(data.get("phase") or "")
        elapsed_s = _coerce_float(event, "elapsed_s")
        if status == "degraded":
            degraded_codes.extend(failure_codes)
            if guard_phase:
                degraded_phases.append(guard_phase)
            if first_degraded_elapsed_s is None:
                first_degraded_elapsed_s = elapsed_s
            continue
        blocked_codes.extend(failure_codes)
        if guard_phase:
            blocked_phases.append(guard_phase)
        if first_blocked_elapsed_s is None:
            first_blocked_elapsed_s = elapsed_s

    failures: list[str] = []
    unique_degraded_codes = tuple(dict.fromkeys(degraded_codes))
    unique_blocked_codes = tuple(dict.fromkeys(blocked_codes))
    unique_degraded_phases = tuple(dict.fromkeys(degraded_phases))
    unique_blocked_phases = tuple(dict.fromkeys(blocked_phases))

    if expectation.must_block is True and not unique_blocked_codes:
        failures.append("guard never entered the blocked state for this abort scenario")
    if expectation.must_block is False and unique_blocked_codes:
        failures.append(
            "guard entered the blocked state during a bounded recovery scenario: "
            + ", ".join(unique_blocked_codes)
        )
    for required_code in expectation.required_blocked_codes:
        if required_code not in unique_blocked_codes:
            failures.append(f"guard never blocked on required code `{required_code}`")
    for forbidden_code in expectation.forbidden_blocked_codes:
        if forbidden_code in unique_blocked_codes:
            failures.append(f"guard blocked on forbidden code `{forbidden_code}`")
    for required_code in expectation.required_degraded_codes:
        if required_code not in unique_degraded_codes:
            failures.append(f"guard never entered degraded state for required code `{required_code}`")

    return HoverGuardMetrics(
        degraded_count=len(degraded_codes),
        blocked_count=len(blocked_codes),
        degraded_codes=unique_degraded_codes,
        blocked_codes=unique_blocked_codes,
        degraded_phases=unique_degraded_phases,
        blocked_phases=unique_blocked_phases,
        first_degraded_elapsed_s=first_degraded_elapsed_s,
        first_blocked_elapsed_s=first_blocked_elapsed_s,
        failures=tuple(failures),
    )


def evaluate_hover_recovery(
    *,
    command_log: Sequence[Mapping[str, object]],
    phase_events: Sequence[Mapping[str, object]],
    target_height_m: float,
    expectation: HoverRecoveryExpectation,
    disturbance_window: HoverRecoveryWindow | None = None,
) -> HoverRecoveryMetrics:
    """Evaluate whether one replayed disturbance recovered within the contract."""

    hover_commands = tuple(
        command
        for command in command_log
        if str(command.get("kind")) == "hover"
    )
    if not hover_commands:
        raise ValueError("hover recovery evaluation requires at least one hover command")

    takeoff_done_elapsed_s = _takeoff_done_elapsed_s(phase_events)
    hover_control_end_elapsed_s = _hover_control_end_elapsed_s(phase_events)
    if hover_control_end_elapsed_s <= takeoff_done_elapsed_s:
        raise ValueError(
            "hover recovery evaluation requires the active hover window to extend past takeoff confirmation"
        )
    active_hover_commands = tuple(
        command
        for command in hover_commands
        if takeoff_done_elapsed_s <= _coerce_float(command, "elapsed_s") <= hover_control_end_elapsed_s
    )
    if not active_hover_commands:
        raise ValueError("hover recovery evaluation found no hover commands inside the active hover window")
    active_duration_s = hover_control_end_elapsed_s - takeoff_done_elapsed_s
    window_source = "progress"
    if disturbance_window is None:
        start_progress = _bounded_progress(expectation.disturbance_start_progress)
        end_progress = _bounded_progress(expectation.disturbance_end_progress)
        if end_progress <= start_progress:
            raise ValueError(
                "hover recovery expectation requires disturbance_end_progress > disturbance_start_progress"
            )
        disturbance_start_elapsed_s = takeoff_done_elapsed_s + (start_progress * active_duration_s)
        disturbance_end_elapsed_s = takeoff_done_elapsed_s + (end_progress * active_duration_s)
    else:
        disturbance_start_elapsed_s = float(disturbance_window.disturbance_start_elapsed_s)
        disturbance_end_elapsed_s = float(disturbance_window.disturbance_end_elapsed_s)
        window_source = str(disturbance_window.source)
    if disturbance_end_elapsed_s <= disturbance_start_elapsed_s:
        raise ValueError("hover recovery requires disturbance_end_elapsed_s > disturbance_start_elapsed_s")
    if disturbance_start_elapsed_s < takeoff_done_elapsed_s:
        raise ValueError("hover recovery disturbance window begins before takeoff confirmation")
    if disturbance_end_elapsed_s > hover_control_end_elapsed_s:
        raise ValueError("hover recovery disturbance window extends past the active hover window")

    disturbance_commands = tuple(
        command
        for command in active_hover_commands
        if disturbance_start_elapsed_s <= _coerce_float(command, "elapsed_s") <= disturbance_end_elapsed_s
    )
    if not disturbance_commands:
        raise ValueError("hover recovery evaluation found no commands inside the disturbance window")

    response_window_end_elapsed_s = min(
        hover_control_end_elapsed_s,
        disturbance_end_elapsed_s + float(expectation.max_recovery_delay_s),
    )
    response_window_commands = tuple(
        command
        for command in active_hover_commands
        if disturbance_start_elapsed_s <= _coerce_float(command, "elapsed_s") <= response_window_end_elapsed_s
    )
    if not response_window_commands:
        raise ValueError("hover recovery found no commands inside the response window")

    forward_values = tuple(_coerce_float(command, "vx_mps") for command in response_window_commands)
    left_values = tuple(_coerce_float(command, "vy_mps") for command in response_window_commands)
    height_deltas = tuple(
        _coerce_float(command, "height_m") - float(target_height_m) for command in response_window_commands
    )

    failures: list[str] = []
    max_forward_mps = max(forward_values)
    min_forward_mps = min(forward_values)
    max_left_mps = max(left_values)
    min_left_mps = min(left_values)
    max_height_delta_m = max(height_deltas)
    min_height_delta_m = min(height_deltas)

    if expectation.forward_direction is not None and expectation.min_forward_abs_mps is not None:
        signed_peak = _expected_signed_peak(
            direction=expectation.forward_direction,
            positive_peak=max_forward_mps,
            negative_peak=min_forward_mps,
        )
        if signed_peak < float(expectation.min_forward_abs_mps):
            failures.append(
                "forward recovery command never reached the expected "
                f"{expectation.forward_direction} correction of {float(expectation.min_forward_abs_mps):.2f} m/s"
            )

    if expectation.left_direction is not None and expectation.min_left_abs_mps is not None:
        signed_peak = _expected_signed_peak(
            direction=expectation.left_direction,
            positive_peak=max_left_mps,
            negative_peak=min_left_mps,
        )
        if signed_peak < float(expectation.min_left_abs_mps):
            failures.append(
                "left-axis recovery command never reached the expected "
                f"{expectation.left_direction} correction of {float(expectation.min_left_abs_mps):.2f} m/s"
            )

    if expectation.height_direction is not None and expectation.min_height_delta_m is not None:
        signed_peak = _expected_signed_peak(
            direction=expectation.height_direction,
            positive_peak=max_height_delta_m,
            negative_peak=min_height_delta_m,
        )
        if signed_peak < float(expectation.min_height_delta_m):
            failures.append(
                "height recovery command never reached the expected "
                f"{expectation.height_direction} correction of {float(expectation.min_height_delta_m):.2f} m"
            )

    height_only_recovery = (
        expectation.height_direction is not None
        and expectation.forward_direction is None
        and expectation.left_direction is None
    )
    response_end_elapsed_s = disturbance_end_elapsed_s
    active_recovery_commands = tuple(
        command
        for command in active_hover_commands
        if _coerce_float(command, "elapsed_s") >= disturbance_start_elapsed_s
        and _command_matches_expected_recovery(
            command=command,
            target_height_m=target_height_m,
            expectation=expectation,
        )
    )
    response_end_height_m: float | None = None
    if active_recovery_commands:
        if height_only_recovery:
            step_tolerance_m = max(
                1e-9,
                float(expectation.settle_height_error_abs_max_m),
            )
            previous_recovery_height_m: float | None = None
            first_recovery_elapsed_s: float | None = None
            last_growth_elapsed_s: float | None = None
            for command in active_recovery_commands:
                elapsed_s = _coerce_float(command, "elapsed_s")
                height_m = _coerce_float(command, "height_m")
                if first_recovery_elapsed_s is None:
                    first_recovery_elapsed_s = elapsed_s
                if previous_recovery_height_m is not None:
                    height_delta_m = height_m - previous_recovery_height_m
                    if expectation.height_direction == "positive":
                        if height_delta_m >= step_tolerance_m:
                            last_growth_elapsed_s = elapsed_s
                    elif height_delta_m <= -step_tolerance_m:
                        last_growth_elapsed_s = elapsed_s
                previous_recovery_height_m = height_m
            assert first_recovery_elapsed_s is not None
            response_end_elapsed_s = (
                first_recovery_elapsed_s
                if last_growth_elapsed_s is None
                else last_growth_elapsed_s
            )
            response_end_height_m = _coerce_float(active_recovery_commands[-1], "height_m")
        else:
            response_end_elapsed_s = max(
                _coerce_float(command, "elapsed_s") for command in active_recovery_commands
            )

    post_disturbance_commands = tuple(
        command
        for command in active_hover_commands
        if response_end_elapsed_s < _coerce_float(command, "elapsed_s") <= hover_control_end_elapsed_s
    )
    if not post_disturbance_commands:
        raise ValueError("hover recovery evaluation found no active hover commands after the recovery response")

    settle_elapsed_s: float | None = None
    recovery_delay_s: float | None = None
    consecutive_settled_commands = 0
    settle_required_commands = max(1, int(expectation.settle_required_commands))
    previous_settle_height_m: float | None = response_end_height_m
    for command in post_disturbance_commands:
        forward_mps = abs(_coerce_float(command, "vx_mps"))
        left_mps = abs(_coerce_float(command, "vy_mps"))
        height_m = _coerce_float(command, "height_m")
        height_error_m = abs(height_m - float(target_height_m))
        if height_only_recovery:
            height_settled = (
                previous_settle_height_m is not None
                and abs(height_m - previous_settle_height_m)
                <= float(expectation.settle_height_error_abs_max_m)
            )
        else:
            height_settled = (
                height_error_m <= float(expectation.settle_height_error_abs_max_m)
            )
        if (
            forward_mps <= float(expectation.settle_velocity_abs_max_mps)
            and left_mps <= float(expectation.settle_velocity_abs_max_mps)
            and height_settled
        ):
            consecutive_settled_commands += 1
            if consecutive_settled_commands >= settle_required_commands:
                settle_elapsed_s = _coerce_float(command, "elapsed_s")
                recovery_delay_s = max(0.0, settle_elapsed_s - response_end_elapsed_s)
                break
        else:
            consecutive_settled_commands = 0
        previous_settle_height_m = height_m

    recovered_within_window = (
        recovery_delay_s is not None
        and recovery_delay_s
        <= (float(expectation.max_recovery_delay_s) + _RECOVERY_DELAY_EPSILON_S)
    )
    if recovery_delay_s is None:
        failures.append("recovery command never settled back into bounded hover output")
    elif not recovered_within_window:
        failures.append(
            "recovery command settled too slowly: "
            f"{recovery_delay_s:.2f} s exceeds the {float(expectation.max_recovery_delay_s):.2f} s limit"
        )

    return HoverRecoveryMetrics(
        disturbance_start_elapsed_s=disturbance_start_elapsed_s,
        disturbance_end_elapsed_s=disturbance_end_elapsed_s,
        response_end_elapsed_s=response_end_elapsed_s,
        max_forward_mps=max_forward_mps,
        min_forward_mps=min_forward_mps,
        max_left_mps=max_left_mps,
        min_left_mps=min_left_mps,
        max_height_delta_m=max_height_delta_m,
        min_height_delta_m=min_height_delta_m,
        settle_elapsed_s=settle_elapsed_s,
        recovery_delay_s=recovery_delay_s,
        recovered_within_window=recovered_within_window,
        window_source=window_source,
        failures=tuple(failures),
    )
