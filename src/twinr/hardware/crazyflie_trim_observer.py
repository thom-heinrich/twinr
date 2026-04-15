"""Estimate bounded hover trim commands for persistent Crazyflie disturbances."""

from __future__ import annotations

from dataclasses import dataclass

from twinr.hardware.crazyflie_trusted_state import TrustedHoverState


def _clamp(value: float, *, lower: float, upper: float) -> float:
    return max(float(lower), min(float(value), float(upper)))


@dataclass(frozen=True, slots=True)
class TrimObserverConfig:
    """Describe the bounded online trim observer used during hover."""

    max_observation_age_s: float = 0.35
    velocity_integral_gain: float = 0.35
    height_integral_gain: float = 0.25
    max_trim_velocity_mps: float = 0.12
    max_height_trim_m: float = 0.04
    max_attitude_abs_deg: float = 8.0
    settle_velocity_abs_max_mps: float = 0.03
    settle_height_error_abs_max_m: float = 0.03
    required_converged_samples: int = 4


@dataclass(frozen=True, slots=True)
class TrimObserverState:
    """Persist the current bounded trim estimate across hover phases."""

    forward_trim_mps: float = 0.0
    left_trim_mps: float = 0.0
    height_trim_m: float = 0.0
    consecutive_converged_samples: int = 0
    converged: bool = False
    failures: tuple[str, ...] = ()


def update_trim_observer(
    state: TrimObserverState,
    *,
    trusted_state: TrustedHoverState,
    target_height_m: float,
    vx_mps: float | None,
    vy_mps: float | None,
    velocity_age_s: float | None,
    roll_deg: float | None,
    pitch_deg: float | None,
    attitude_age_s: float | None,
    config: TrimObserverConfig,
) -> TrimObserverState:
    """Return the next bounded trim estimate from the current hover observation."""

    failures = list(trusted_state.failures)
    max_age_s = float(config.max_observation_age_s)

    if roll_deg is None or pitch_deg is None or attitude_age_s is None:
        failures.append("attitude telemetry is unavailable for trim identification")
    elif float(attitude_age_s) > max_age_s:
        failures.append(
            f"attitude telemetry age {float(attitude_age_s):.3f} s exceeds {max_age_s:.3f} s"
        )
    else:
        max_attitude_abs_deg = float(config.max_attitude_abs_deg)
        if abs(float(roll_deg)) > max_attitude_abs_deg:
            failures.append(
                f"roll {float(roll_deg):.2f} deg exceeds the {max_attitude_abs_deg:.2f} deg trim-identify guard"
            )
        if abs(float(pitch_deg)) > max_attitude_abs_deg:
            failures.append(
                f"pitch {float(pitch_deg):.2f} deg exceeds the {max_attitude_abs_deg:.2f} deg trim-identify guard"
            )

    if vx_mps is None or vy_mps is None or velocity_age_s is None:
        failures.append("horizontal velocity telemetry is unavailable for trim identification")
    elif float(velocity_age_s) > max_age_s:
        failures.append(
            f"horizontal velocity telemetry age {float(velocity_age_s):.3f} s exceeds {max_age_s:.3f} s"
        )

    next_forward_trim_mps = float(state.forward_trim_mps)
    next_left_trim_mps = float(state.left_trim_mps)
    next_height_trim_m = float(state.height_trim_m)
    consecutive_converged_samples = 0

    if not failures:
        assert vx_mps is not None
        assert vy_mps is not None
        assert trusted_state.trusted_height_m is not None
        next_forward_trim_mps = _clamp(
            float(state.forward_trim_mps) - (float(config.velocity_integral_gain) * float(vx_mps)),
            lower=-float(config.max_trim_velocity_mps),
            upper=float(config.max_trim_velocity_mps),
        )
        next_left_trim_mps = _clamp(
            float(state.left_trim_mps) - (float(config.velocity_integral_gain) * float(vy_mps)),
            lower=-float(config.max_trim_velocity_mps),
            upper=float(config.max_trim_velocity_mps),
        )
        height_error_m = float(target_height_m) - float(trusted_state.trusted_height_m)
        next_height_trim_m = _clamp(
            float(state.height_trim_m) + (float(config.height_integral_gain) * height_error_m),
            lower=-float(config.max_height_trim_m),
            upper=float(config.max_height_trim_m),
        )
        settled = (
            abs(float(vx_mps)) <= float(config.settle_velocity_abs_max_mps)
            and abs(float(vy_mps)) <= float(config.settle_velocity_abs_max_mps)
            and abs(height_error_m) <= float(config.settle_height_error_abs_max_m)
        )
        consecutive_converged_samples = (
            int(state.consecutive_converged_samples) + 1 if settled else 0
        )

    converged = consecutive_converged_samples >= int(config.required_converged_samples)
    return TrimObserverState(
        forward_trim_mps=next_forward_trim_mps,
        left_trim_mps=next_left_trim_mps,
        height_trim_m=next_height_trim_m,
        consecutive_converged_samples=consecutive_converged_samples,
        converged=converged,
        failures=tuple(failures),
    )
