"""Compute one first-class trusted hover state from raw Crazyflie telemetry."""

from __future__ import annotations

from dataclasses import dataclass

from twinr.hardware.crazyflie_flow_anchor import HeightTrustConfig, compute_trusted_height


def _is_fresh(value: float | None, age_s: float | None, *, max_age_s: float) -> bool:
    return value is not None and age_s is not None and float(age_s) <= float(max_age_s)


@dataclass(frozen=True, slots=True)
class LateralTrustConfig:
    """Describe when the live lateral hover state is trustworthy enough."""

    max_observation_age_s: float = 0.35
    min_motion_squal: int = 30
    require_motion_squal: bool = True
    require_pose: bool = True
    require_velocity: bool = True
    require_supervisor_flying: bool = True


@dataclass(frozen=True, slots=True)
class TrustedHoverState:
    """Summarize the current trusted hover state used by higher-level control."""

    trusted_height_m: float | None
    trusted_height_source: str
    sensor_disagreement_m: float | None
    flow_confident: bool
    pose_available: bool
    velocity_available: bool
    supervisor_flying: bool
    failures: tuple[str, ...]


def compute_trusted_hover_state(
    *,
    raw_height_m: float | None,
    raw_height_age_s: float | None,
    estimate_z_m: float | None,
    estimate_z_age_s: float | None,
    x_m: float | None,
    y_m: float | None,
    pose_age_s: float | None,
    vx_mps: float | None,
    vy_mps: float | None,
    velocity_age_s: float | None,
    motion_squal: int | None,
    motion_squal_age_s: float | None,
    is_flying: bool | None,
    supervisor_age_s: float | None,
    height_config: HeightTrustConfig,
    lateral_config: LateralTrustConfig,
) -> TrustedHoverState:
    """Return the current trusted hover state across height and lateral signals."""

    trusted_height = compute_trusted_height(
        raw_height_m=raw_height_m,
        raw_height_age_s=raw_height_age_s,
        estimate_z_m=estimate_z_m,
        estimate_z_age_s=estimate_z_age_s,
        config=height_config,
    )
    failures = list(trusted_height.failures)
    max_age_s = float(lateral_config.max_observation_age_s)

    flow_confident = True
    if lateral_config.require_motion_squal:
        if motion_squal is None or motion_squal_age_s is None:
            failures.append("optical-flow quality is unavailable")
            flow_confident = False
        elif float(motion_squal_age_s) > max_age_s:
            failures.append(
                f"optical-flow quality age {float(motion_squal_age_s):.3f} s exceeds {max_age_s:.3f} s"
            )
            flow_confident = False
        elif int(motion_squal) < int(lateral_config.min_motion_squal):
            failures.append(
                "optical-flow quality "
                f"{int(motion_squal)} is below the {int(lateral_config.min_motion_squal)} stability floor"
            )
            flow_confident = False

    pose_available = _is_fresh(x_m, pose_age_s, max_age_s=max_age_s) and _is_fresh(
        y_m, pose_age_s, max_age_s=max_age_s
    )
    if lateral_config.require_pose and not pose_available:
        failures.append("xy pose telemetry is unavailable")

    velocity_available = _is_fresh(vx_mps, velocity_age_s, max_age_s=max_age_s) and _is_fresh(
        vy_mps, velocity_age_s, max_age_s=max_age_s
    )
    if lateral_config.require_velocity and not velocity_available:
        failures.append("horizontal velocity telemetry is unavailable")

    supervisor_flying = (
        is_flying is True
        and supervisor_age_s is not None
        and float(supervisor_age_s) <= max_age_s
    )
    if lateral_config.require_supervisor_flying and not supervisor_flying:
        failures.append("supervisor does not currently report the craft as flying")

    return TrustedHoverState(
        trusted_height_m=trusted_height.height_m,
        trusted_height_source=trusted_height.source,
        sensor_disagreement_m=trusted_height.sensor_disagreement_m,
        flow_confident=flow_confident,
        pose_available=pose_available,
        velocity_available=velocity_available,
        supervisor_flying=supervisor_flying,
        failures=tuple(failures),
    )
