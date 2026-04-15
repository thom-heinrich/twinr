"""Compute bounded Flow-based anchor-hold control for Crazyflie hover.

This module owns the host-side outer-loop math for Twinr's bounded
``flow_relative_hover`` lane. It does not replace the Crazyflie's internal
attitude controller. Instead, it consumes the already available onboard
state estimate, optical-flow-derived motion, and downward range signal to
produce small corrective body-frame velocity targets plus a bounded height
correction around the commanded hover target.

The same module also defines the height-trust contract. A raw downward-range
sample must not be treated as absolute truth when it disagrees strongly with
the flight estimate or when it arrives stale. This keeps the hover lane from
blindly accepting surface switches or ToF outliers as the aircraft's true
altitude.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class HeightTrustConfig:
    """Describe when a hover-height observation is trustworthy."""

    max_observation_age_s: float = 0.35
    max_sensor_disagreement_m: float = 0.25
    allow_estimate_fallback: bool = True


@dataclass(frozen=True, slots=True)
class TrustedHeightResult:
    """Return the current best-trusted height and any trust failures."""

    height_m: float | None
    source: str
    sensor_disagreement_m: float | None
    failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FlowAnchorObservation:
    """Represent one current hover-control observation."""

    x_m: float | None
    y_m: float | None
    pose_age_s: float | None
    vx_mps: float | None
    vy_mps: float | None
    velocity_age_s: float | None
    yaw_deg: float | None
    yaw_age_s: float | None
    raw_height_m: float | None
    raw_height_age_s: float | None
    estimate_z_m: float | None
    estimate_z_age_s: float | None


@dataclass(frozen=True, slots=True)
class FlowAnchorControlConfig:
    """Describe the bounded outer-loop anchor-hold controller."""

    enabled: bool = True
    max_observation_age_s: float = 0.35
    position_gain_p: float = 0.9
    velocity_gain_d: float = 0.7
    height_gain_p: float = 0.9
    max_correction_velocity_mps: float = 0.20
    max_height_correction_m: float = 0.06
    max_commanded_height_m: float | None = None


@dataclass(frozen=True, slots=True)
class FlowAnchorControlCommand:
    """Describe one bounded hover control command from the anchor loop."""

    forward_mps: float
    left_mps: float
    height_m: float | None
    trusted_height_m: float | None
    trusted_height_source: str
    sensor_disagreement_m: float | None
    failures: tuple[str, ...]


def _is_fresh(value: float | None, age_s: float | None, *, max_age_s: float) -> bool:
    return value is not None and age_s is not None and float(age_s) <= float(max_age_s)


def compute_trusted_height(
    *,
    raw_height_m: float | None,
    raw_height_age_s: float | None,
    estimate_z_m: float | None,
    estimate_z_age_s: float | None,
    config: HeightTrustConfig,
) -> TrustedHeightResult:
    """Return the current best-trusted height sample.

    Preference order:
    1. fresh downward range, if it agrees with a fresh state estimate or the
       estimate is unavailable
    2. fresh state estimate, when fallback is explicitly allowed
    3. no trusted height
    """

    failures: list[str] = []
    raw_fresh = _is_fresh(raw_height_m, raw_height_age_s, max_age_s=config.max_observation_age_s)
    estimate_fresh = _is_fresh(estimate_z_m, estimate_z_age_s, max_age_s=config.max_observation_age_s)
    sensor_disagreement_m: float | None = None

    if raw_fresh and estimate_fresh:
        assert raw_height_m is not None
        assert estimate_z_m is not None
        sensor_disagreement_m = abs(float(raw_height_m) - float(estimate_z_m))
        if sensor_disagreement_m > float(config.max_sensor_disagreement_m):
            failures.append(
                "trusted height is unavailable because downward range and state estimate disagree by "
                f"{sensor_disagreement_m:.2f} m"
            )
            if config.allow_estimate_fallback:
                return TrustedHeightResult(
                    height_m=float(estimate_z_m),
                    source="state_estimate",
                    sensor_disagreement_m=sensor_disagreement_m,
                    failures=tuple(failures),
                )
            return TrustedHeightResult(
                height_m=None,
                source="none",
                sensor_disagreement_m=sensor_disagreement_m,
                failures=tuple(failures),
            )

    if raw_fresh:
        assert raw_height_m is not None
        return TrustedHeightResult(
            height_m=float(raw_height_m),
            source="range",
            sensor_disagreement_m=sensor_disagreement_m,
            failures=tuple(failures),
        )
    if estimate_fresh and config.allow_estimate_fallback:
        assert estimate_z_m is not None
        return TrustedHeightResult(
            height_m=float(estimate_z_m),
            source="state_estimate",
            sensor_disagreement_m=sensor_disagreement_m,
            failures=tuple(failures),
        )

    failures.append("trusted height is unavailable because both downward range and state estimate are stale or missing")
    return TrustedHeightResult(
        height_m=None,
        source="none",
        sensor_disagreement_m=sensor_disagreement_m,
        failures=tuple(failures),
    )


def compute_flow_anchor_command(
    *,
    observation: FlowAnchorObservation,
    anchor_xy: tuple[float, float] | None,
    target_height_m: float,
    control_config: FlowAnchorControlConfig,
    height_trust_config: HeightTrustConfig,
) -> FlowAnchorControlCommand:
    """Return one bounded outer-loop correction for `flow_relative_hover`."""

    trusted_height = compute_trusted_height(
        raw_height_m=observation.raw_height_m,
        raw_height_age_s=observation.raw_height_age_s,
        estimate_z_m=observation.estimate_z_m,
        estimate_z_age_s=observation.estimate_z_age_s,
        config=height_trust_config,
    )
    failures = list(trusted_height.failures)
    bounded_target_height_m = max(0.0, float(target_height_m))

    if not control_config.enabled:
        return FlowAnchorControlCommand(
            forward_mps=0.0,
            left_mps=0.0,
            height_m=trusted_height.height_m if trusted_height.height_m is not None else bounded_target_height_m,
            trusted_height_m=trusted_height.height_m,
            trusted_height_source=trusted_height.source,
            sensor_disagreement_m=trusted_height.sensor_disagreement_m,
            failures=tuple(failures),
        )

    pose_fresh = (
        observation.x_m is not None
        and observation.y_m is not None
        and observation.pose_age_s is not None
        and float(observation.pose_age_s) <= float(control_config.max_observation_age_s)
    )
    velocity_fresh = (
        observation.vx_mps is not None
        and observation.vy_mps is not None
        and observation.velocity_age_s is not None
        and float(observation.velocity_age_s) <= float(control_config.max_observation_age_s)
    )
    yaw_fresh = (
        observation.yaw_deg is not None
        and observation.yaw_age_s is not None
        and float(observation.yaw_age_s) <= float(control_config.max_observation_age_s)
    )
    if anchor_xy is None or not pose_fresh:
        failures.append("flow anchor control is unavailable because no fresh anchor pose is available")
    if not velocity_fresh:
        failures.append("flow anchor control is unavailable because velocity telemetry is stale or missing")
    if not yaw_fresh:
        failures.append("flow anchor control is unavailable because yaw telemetry is stale or missing")

    corrected_height_m = bounded_target_height_m
    if trusted_height.height_m is not None:
        height_error_m = bounded_target_height_m - float(trusted_height.height_m)
        corrected_height_m = bounded_target_height_m + (
            float(control_config.height_gain_p) * height_error_m
        )
        corrected_height_m = max(
            bounded_target_height_m - float(control_config.max_height_correction_m),
            min(
                corrected_height_m,
                bounded_target_height_m + float(control_config.max_height_correction_m),
            ),
        )
    elif control_config.max_commanded_height_m is not None:
        corrected_height_m = min(bounded_target_height_m, float(control_config.max_commanded_height_m))

    if failures:
        return FlowAnchorControlCommand(
            forward_mps=0.0,
            left_mps=0.0,
            height_m=corrected_height_m,
            trusted_height_m=trusted_height.height_m,
            trusted_height_source=trusted_height.source,
            sensor_disagreement_m=trusted_height.sensor_disagreement_m,
            failures=tuple(failures),
        )

    assert anchor_xy is not None
    assert observation.x_m is not None and observation.y_m is not None
    assert observation.vx_mps is not None and observation.vy_mps is not None
    assert observation.yaw_deg is not None

    error_world_x = float(anchor_xy[0]) - float(observation.x_m)
    error_world_y = float(anchor_xy[1]) - float(observation.y_m)
    target_world_vx = (
        float(control_config.position_gain_p) * error_world_x
        - float(control_config.velocity_gain_d) * float(observation.vx_mps)
    )
    target_world_vy = (
        float(control_config.position_gain_p) * error_world_y
        - float(control_config.velocity_gain_d) * float(observation.vy_mps)
    )
    target_world_speed = math.hypot(target_world_vx, target_world_vy)
    if target_world_speed > float(control_config.max_correction_velocity_mps) and target_world_speed > 0.0:
        scale = float(control_config.max_correction_velocity_mps) / target_world_speed
        target_world_vx *= scale
        target_world_vy *= scale

    yaw_rad = math.radians(float(observation.yaw_deg))
    forward_mps = (math.cos(yaw_rad) * target_world_vx) + (math.sin(yaw_rad) * target_world_vy)
    left_mps = (-math.sin(yaw_rad) * target_world_vx) + (math.cos(yaw_rad) * target_world_vy)

    return FlowAnchorControlCommand(
        forward_mps=forward_mps,
        left_mps=left_mps,
        height_m=corrected_height_m,
        trusted_height_m=trusted_height.height_m,
        trusted_height_source=trusted_height.source,
        sensor_disagreement_m=trusted_height.sensor_disagreement_m,
        failures=tuple(failures),
    )
