"""Compute bounded vertical bootstrap thrust for Crazyflie takeoff.

This module owns the host-side control math for the first centimeters of a real
Crazyflie takeoff. It does not replace the Crazyflie's internal attitude
controller. Instead, it computes one bounded raw-thrust command that should
lift the aircraft into a small trusted range/flow window before hover-mode
setpoints are allowed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VerticalBootstrapConfig:
    """Describe one bounded closed-loop vertical bootstrap contract."""

    target_height_m: float
    min_thrust_percentage: float
    feedforward_thrust_percentage: float
    max_thrust_percentage: float
    reference_duration_s: float
    progress_to_ceiling_s: float
    max_duration_s: float
    height_gain_per_m: float
    vertical_speed_gain_per_mps: float
    min_range_height_m: float
    max_range_height_m: float
    min_range_rise_m: float
    max_observation_age_s: float
    max_ceiling_without_progress_s: float
    required_liveness_samples: int = 1
    require_motion_squal_liveness: bool = True
    min_motion_squal: int = 1
    max_attitude_abs_deg: float = 10.0


@dataclass(frozen=True, slots=True)
class VerticalBootstrapObservation:
    """Represent one current sensor snapshot for the bootstrap controller."""

    elapsed_s: float
    distance_m: float | None
    distance_age_s: float | None
    vertical_speed_mps: float | None = None
    vertical_speed_age_s: float | None = None
    motion_squal: int | None = None
    motion_squal_age_s: float | None = None
    roll_deg: float | None = None
    pitch_deg: float | None = None
    attitude_age_s: float | None = None


@dataclass(frozen=True, slots=True)
class VerticalBootstrapState:
    """Carry deterministic controller state across bootstrap control ticks."""

    baseline_distance_m: float | None
    last_elapsed_s: float = 0.0
    consecutive_liveness_samples: int = 0
    ceiling_without_progress_s: float = 0.0
    progress_boost_percentage: float = 0.0


@dataclass(frozen=True, slots=True)
class VerticalBootstrapDecision:
    """Describe one bounded bootstrap control decision."""

    commanded_thrust_percentage: float
    raw_commanded_thrust_percentage: float
    feedforward_thrust_percentage: float
    progress_boost_percentage: float
    height_term_percentage: float
    vertical_speed_term_percentage: float
    thrust_headroom_percentage: float
    reference_height_m: float
    reference_progress: float
    reference_vertical_speed_mps: float
    baseline_distance_m: float
    current_height_m: float
    target_liftoff_height_m: float
    height_error_m: float
    vertical_speed_error_mps: float
    range_rise_m: float | None
    range_live: bool
    range_height_ready: bool
    range_rise_ready: bool
    flow_live: bool
    motion_squal_fresh: bool
    handoff_ready: bool
    failure_codes: tuple[str, ...]
    abort_reason: str | None = None
    distance_fresh: bool = False
    vertical_speed_fresh: bool = False
    ceiling_without_progress_s: float = 0.0
    at_thrust_ceiling: bool = False
    progress_missing: bool = False


def initialize_vertical_bootstrap_state(
    *,
    baseline_distance_m: float | None,
) -> VerticalBootstrapState:
    """Return one fresh controller state from the observed ground baseline."""

    bounded_baseline_distance_m: float | None = None
    if baseline_distance_m is not None:
        bounded_baseline_distance_m = max(0.0, float(baseline_distance_m))
    return VerticalBootstrapState(
        baseline_distance_m=bounded_baseline_distance_m,
    )


def step_vertical_bootstrap_controller(
    *,
    config: VerticalBootstrapConfig,
    state: VerticalBootstrapState,
    observation: VerticalBootstrapObservation,
) -> tuple[VerticalBootstrapState, VerticalBootstrapDecision]:
    """Compute one bounded bootstrap-thrust decision.

    The controller follows a tiny reference climb and only reports
    ``handoff_ready`` once range rise, bounded micro-liftoff height, and flow
    liveness are all proven.
    """

    bounded_elapsed_s = max(0.0, float(observation.elapsed_s))
    bounded_max_observation_age_s = max(0.0, float(config.max_observation_age_s))
    bounded_reference_duration_s = max(1e-6, float(config.reference_duration_s))
    bounded_progress_to_ceiling_s = max(1e-6, float(config.progress_to_ceiling_s))
    bounded_target_height_m = max(0.0, float(config.target_height_m))
    bounded_min_range_height_m = max(0.0, float(config.min_range_height_m))
    bounded_max_range_height_m = max(
        bounded_min_range_height_m,
        float(config.max_range_height_m),
    )
    bounded_min_range_rise_m = max(0.0, float(config.min_range_rise_m))
    bounded_min_thrust_percentage = max(0.0, float(config.min_thrust_percentage))
    bounded_feedforward_thrust_percentage = max(
        bounded_min_thrust_percentage,
        float(config.feedforward_thrust_percentage),
    )
    bounded_max_thrust_percentage = max(
        bounded_feedforward_thrust_percentage,
        float(config.max_thrust_percentage),
    )
    bounded_required_liveness_samples = max(1, int(config.required_liveness_samples))
    bounded_max_ceiling_without_progress_s = max(
        0.0,
        float(config.max_ceiling_without_progress_s),
    )

    distance_fresh = (
        observation.distance_m is not None
        and observation.distance_age_s is not None
        and float(observation.distance_age_s) <= bounded_max_observation_age_s
    )
    baseline_distance_m = state.baseline_distance_m
    if baseline_distance_m is None and distance_fresh:
        assert observation.distance_m is not None
        baseline_distance_m = max(0.0, float(observation.distance_m))
    if baseline_distance_m is None:
        baseline_distance_m = 0.0

    target_liftoff_height_m = min(
        bounded_max_range_height_m,
        max(bounded_target_height_m, bounded_min_range_height_m),
    )
    reference_progress = min(1.0, bounded_elapsed_s / bounded_reference_duration_s)
    reference_height_m = baseline_distance_m + (
        max(0.0, target_liftoff_height_m - baseline_distance_m) * reference_progress
    )
    reference_vertical_speed_mps = 0.0
    if reference_progress < 1.0:
        reference_vertical_speed_mps = max(
            0.0,
            (target_liftoff_height_m - baseline_distance_m) / bounded_reference_duration_s,
        )

    current_height_m = baseline_distance_m
    range_rise_m: float | None = None
    if distance_fresh:
        assert observation.distance_m is not None
        current_height_m = max(0.0, float(observation.distance_m))
        range_rise_m = current_height_m - baseline_distance_m

    vertical_speed_fresh = (
        observation.vertical_speed_mps is not None
        and observation.vertical_speed_age_s is not None
        and float(observation.vertical_speed_age_s) <= bounded_max_observation_age_s
    )
    current_vertical_speed_mps = 0.0
    if vertical_speed_fresh:
        assert observation.vertical_speed_mps is not None
        current_vertical_speed_mps = float(observation.vertical_speed_mps)

    height_error_m = reference_height_m - current_height_m
    vertical_speed_error_mps = reference_vertical_speed_mps - current_vertical_speed_mps
    height_term_percentage = float(config.height_gain_per_m) * height_error_m
    vertical_speed_term_percentage = (
        float(config.vertical_speed_gain_per_mps) * vertical_speed_error_mps
    )

    range_height_ready = distance_fresh and current_height_m >= bounded_min_range_height_m
    range_rise_ready = (
        distance_fresh
        and range_rise_m is not None
        and range_rise_m >= bounded_min_range_rise_m
    )
    range_live = range_height_ready and range_rise_ready
    motion_squal_fresh = (
        observation.motion_squal is not None
        and observation.motion_squal_age_s is not None
        and float(observation.motion_squal_age_s) <= bounded_max_observation_age_s
    )
    flow_live = True
    if config.require_motion_squal_liveness:
        flow_live = (
            motion_squal_fresh
            and observation.motion_squal is not None
            and int(observation.motion_squal) >= int(config.min_motion_squal)
        )

    failure_codes: list[str] = []
    if not range_live:
        failure_codes.append("range_liveness_missing")
    if config.require_motion_squal_liveness and not flow_live:
        failure_codes.append("flow_liveness_missing")

    attitude_unsafe = (
        observation.roll_deg is not None
        and observation.pitch_deg is not None
        and observation.attitude_age_s is not None
        and float(observation.attitude_age_s) <= bounded_max_observation_age_s
        and (
            abs(float(observation.roll_deg)) > float(config.max_attitude_abs_deg)
            or abs(float(observation.pitch_deg)) > float(config.max_attitude_abs_deg)
        )
    )
    if attitude_unsafe:
        failure_codes.append("attitude_unsafe")

    overshoot = distance_fresh and current_height_m > bounded_max_range_height_m
    if overshoot:
        failure_codes.append("bootstrap_range_overshoot")

    bounded_dt_s = max(0.0, bounded_elapsed_s - max(0.0, float(state.last_elapsed_s)))
    progress_missing = not range_live
    max_progress_boost_percentage = max(
        0.0,
        bounded_max_thrust_percentage - bounded_feedforward_thrust_percentage,
    )
    progress_boost_rate_percentage_per_s = 0.0
    if max_progress_boost_percentage > 0.0:
        progress_boost_rate_percentage_per_s = (
            max_progress_boost_percentage / bounded_progress_to_ceiling_s
        )
    progress_boost_percentage = max(0.0, float(state.progress_boost_percentage))
    if progress_missing:
        progress_boost_percentage = min(
            max_progress_boost_percentage,
            progress_boost_percentage
            + (progress_boost_rate_percentage_per_s * bounded_dt_s),
        )
    else:
        progress_boost_percentage = 0.0
    raw_commanded_thrust_percentage = (
        bounded_feedforward_thrust_percentage
        + progress_boost_percentage
        + height_term_percentage
        + vertical_speed_term_percentage
    )
    commanded_thrust_percentage = raw_commanded_thrust_percentage
    commanded_thrust_percentage = min(
        bounded_max_thrust_percentage,
        max(bounded_min_thrust_percentage, commanded_thrust_percentage),
    )
    thrust_headroom_percentage = max(
        0.0,
        bounded_max_thrust_percentage - commanded_thrust_percentage,
    )
    at_thrust_ceiling = commanded_thrust_percentage >= (bounded_max_thrust_percentage - 1e-6)
    ceiling_without_progress_s = 0.0
    if at_thrust_ceiling and progress_missing:
        ceiling_without_progress_s = float(state.ceiling_without_progress_s) + bounded_dt_s

    abort_reason: str | None = None
    if overshoot:
        abort_reason = (
            "vertical bootstrap exceeded the bounded pre-hover height "
            f"({current_height_m:.2f} m > {bounded_max_range_height_m:.2f} m)"
        )
    elif attitude_unsafe:
        assert observation.roll_deg is not None
        assert observation.pitch_deg is not None
        abort_reason = (
            "vertical bootstrap observed unsafe attitude before hover handoff "
            f"(roll={float(observation.roll_deg):.2f} deg, pitch={float(observation.pitch_deg):.2f} deg)"
        )
    elif (
        bounded_max_ceiling_without_progress_s > 0.0
        and ceiling_without_progress_s > bounded_max_ceiling_without_progress_s
    ):
        abort_reason = (
            "vertical bootstrap saturated at bounded thrust without proving lift "
            f"for {ceiling_without_progress_s:.2f} s"
        )
        failure_codes.append("bootstrap_ceiling_without_progress")
    elif bounded_elapsed_s > max(0.0, float(config.max_duration_s)):
        abort_reason = (
            "vertical bootstrap did not establish fresh z-range/flow liveness "
            f"within {float(config.max_duration_s):.2f} s"
        )
        failure_codes.append("bootstrap_timeout")

    consecutive_liveness_samples = (
        int(state.consecutive_liveness_samples) + 1
        if abort_reason is None and range_live and flow_live
        else 0
    )
    handoff_ready = consecutive_liveness_samples >= bounded_required_liveness_samples

    next_state = VerticalBootstrapState(
        baseline_distance_m=baseline_distance_m,
        last_elapsed_s=bounded_elapsed_s,
        consecutive_liveness_samples=consecutive_liveness_samples,
        ceiling_without_progress_s=ceiling_without_progress_s,
        progress_boost_percentage=progress_boost_percentage,
    )
    decision = VerticalBootstrapDecision(
        commanded_thrust_percentage=commanded_thrust_percentage,
        raw_commanded_thrust_percentage=raw_commanded_thrust_percentage,
        feedforward_thrust_percentage=bounded_feedforward_thrust_percentage,
        progress_boost_percentage=progress_boost_percentage,
        height_term_percentage=height_term_percentage,
        vertical_speed_term_percentage=vertical_speed_term_percentage,
        thrust_headroom_percentage=thrust_headroom_percentage,
        reference_height_m=reference_height_m,
        reference_progress=reference_progress,
        reference_vertical_speed_mps=reference_vertical_speed_mps,
        baseline_distance_m=baseline_distance_m,
        current_height_m=current_height_m,
        target_liftoff_height_m=target_liftoff_height_m,
        height_error_m=height_error_m,
        vertical_speed_error_mps=vertical_speed_error_mps,
        range_rise_m=range_rise_m,
        range_live=range_live,
        range_height_ready=range_height_ready,
        range_rise_ready=range_rise_ready,
        flow_live=flow_live,
        motion_squal_fresh=motion_squal_fresh,
        handoff_ready=handoff_ready and abort_reason is None,
        failure_codes=tuple(failure_codes),
        abort_reason=abort_reason,
        distance_fresh=distance_fresh,
        vertical_speed_fresh=vertical_speed_fresh,
        ceiling_without_progress_s=ceiling_without_progress_s,
        at_thrust_ceiling=at_thrust_ceiling,
        progress_missing=progress_missing,
    )
    return (next_state, decision)
