# CHANGELOG: 2026-03-27
# BUG-1: Reset per-run primitive state so one StatefulHoverPrimitive instance can execute multiple runs safely.
# BUG-2: Bounded touchdown recovery now prevents indefinite hangs on missing/stale touchdown telemetry.
# BUG-3: Trace-writer and telemetry-provider failures are now non-fatal to the flight path; observability errors no longer skip landing.
# BUG-4: Stop/notify-stop now drains the outbound queue before returning, matching Bitcraze's documented safe handover pattern.
# BUG-5: Estimator-settle logger setup failures now return a blocked report instead of crashing the hover test.
# SEC-1: Added runtime link-health gating so severe radio/USB degradation can trigger an early abort-to-land before watchdog shutdown.
# IMP-1: Pre-arm snapshots now capture previous controller params and expose restore_hover_pre_arm(...) for cleanup.
# IMP-2: Added supervisor-bit helpers, built-in cflib link-statistics integration, deadline-based scheduling, and richer landing telemetry.
# BUG-6: Takeoff confirmation used only a minimal liftoff threshold and could enter hover while the craft was still far below the commanded hover height. Confirmation now requires reaching the target-height band before hover becomes active.
# BUG-7: Host takeoff gating, hover acceptance, and the on-device failsafe used drift-prone duplicated liftoff thresholds. The shared takeoff-active-height helper now defines one threshold for all bounded hover consumers.
# BUG-8: Landing used range-only touchdown success and still emitted post-touchdown hover setpoints, which could produce a short re-hop after ground contact. Touchdown now requires both low z-range and supervisor grounded before the final quiet stop path.
# BUG-9: Hover stability was only graded after the flight artifact was built, so visibly unstable hover could continue for seconds before landing. The primitive now runs an explicit stabilize phase and a live in-flight stability guard.
# BUG-10: Flow-only hover used zero-velocity commands without a host-side outer-loop, so bounded hover could drift badly even while the inner Crazyflie controller stayed active. The primitive now adds a bounded flow-anchor outer-loop plus trusted-height gating on top of the existing onboard controller.
# BUG-11: Hardware takeoff still entered hover-mode before z-range/flow liveness was proven, so dead `range.zrange` could turn the first lift into an uncontrolled climb. Hardware micro-liftoff now uses bounded raw thrust and only hands over to hover-mode after fresh z-range and flow liveness are both proven.
# BUG-12: The first raw-thrust micro-liftoff used one fixed/ramped thrust guess, so real hardware could still fail opposite ways: no lift on one pack/surface, or excessive lift on another. The hardware bootstrap is now one bounded closed-loop vertical controller that compares expected range rise against live z-range/vz truth every tick before hover handoff.

"""Provide deterministic Crazyflie hover-primitive helpers.

This module owns the bounded low-level pieces behind Twinr's hover test:
deterministic controller/estimator pre-arm setup, Kalman settling checks, and
an explicit hover-setpoint primitive with a built-in abort/landing path. It
exists to keep ``run_hover_test.py`` focused on orchestration and reporting
instead of growing into a mixed hardware-control file.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import time
from threading import Lock
from typing import Any, Callable, Iterable, Mapping, Sequence, cast

from twinr.hardware.crazyflie_flow_anchor import (
    FlowAnchorControlCommand,
    FlowAnchorControlConfig,
    FlowAnchorObservation,
    HeightTrustConfig,
    compute_flow_anchor_command,
)
from twinr.hardware.crazyflie_trim_observer import (
    TrimObserverConfig,
    TrimObserverState,
    update_trim_observer,
)
from twinr.hardware.crazyflie_trusted_state import (
    LateralTrustConfig,
    compute_trusted_hover_state,
)
from twinr.hardware.crazyflie_vertical_bootstrap import (
    VerticalBootstrapConfig as HoverVerticalBootstrapConfig,
    VerticalBootstrapObservation,
    initialize_vertical_bootstrap_state,
    step_vertical_bootstrap_controller,
)


HOVER_PARAM_VERIFY_ATTEMPTS = 5
HOVER_PARAM_VERIFY_SETTLE_S = 0.05
HOVER_ESTIMATOR_RESET_SETTLE_S = 0.1
HOVER_ESTIMATOR_SETTLE_PERIOD_MS = 100
HOVER_ESTIMATOR_SETTLE_TIMEOUT_S = 5.0
HOVER_ESTIMATOR_SETTLE_WINDOW_SAMPLES = 10
HOVER_ESTIMATOR_VARIANCE_THRESHOLD = 0.001
HOVER_ESTIMATOR_ATTITUDE_ABS_MAX_DEG = 5.0
HOVER_ESTIMATOR_MOTION_SQUAL_MIN = 30
HOVER_ESTIMATOR_MOTION_SQUAL_REQUIRED_RATIO = 0.8
HOVER_ESTIMATOR_FLOW_GATE_MIN_HEIGHT_M = 0.10
HOVER_ESTIMATOR_RANGE_INVALID_MM = 32000.0
HOVER_SETPOINT_PERIOD_S = 0.1
HOVER_STOP_SETPOINT_REPEAT = 5
HOVER_NOTIFY_STOP_FLUSH_S = 0.10
HOVER_LANDING_FLOOR_HEIGHT_M = 0.08
HOVER_LANDING_FLOOR_SETTLE_S = 0.25
HOVER_TOUCHDOWN_HEIGHT_M = 0.03
HOVER_TOUCHDOWN_SETTLE_S = 0.20
HOVER_TOUCHDOWN_VELOCITY_MPS = 0.08
HOVER_TOUCHDOWN_CONFIRM_HEIGHT_M = 0.05
HOVER_TOUCHDOWN_CONFIRM_SAMPLES = 3
HOVER_TOUCHDOWN_CONFIRM_TIMEOUT_S = 2.0
HOVER_TOUCHDOWN_CONFIRM_MAX_AGE_S = 0.35
HOVER_TOUCHDOWN_RECOVERY_TIMEOUT_S = 1.5
HOVER_ZERO_HEIGHT_SETTLE_S = 0.10
HOVER_TAKEOFF_CONFIRM_MIN_HEIGHT_M = 0.08
HOVER_TAKEOFF_CONFIRM_SAMPLES = 2
HOVER_TAKEOFF_CONFIRM_TIMEOUT_S = 1.5
HOVER_TAKEOFF_CONFIRM_MAX_AGE_S = 0.35
HOVER_TAKEOFF_TARGET_HEIGHT_TOLERANCE_M = 0.05
HOVER_MICRO_LIFTOFF_HEIGHT_M = HOVER_TAKEOFF_CONFIRM_MIN_HEIGHT_M
HOVER_STABILIZE_TIMEOUT_S = 1.0
HOVER_STABILITY_REQUIRED_SAMPLES = 3
HOVER_STABILITY_ABORT_SAMPLES = 2
HOVER_STABILITY_MAX_OBSERVATION_AGE_S = 0.35
HOVER_STABILITY_MAX_ATTITUDE_ABS_DEG = 10.0
HOVER_STABILITY_MAX_HORIZONTAL_SPEED_MPS = 0.40
HOVER_STABILITY_MAX_XY_DRIFT_M = 0.20
HOVER_STABILITY_MAX_HEIGHT_ERROR_M = 0.05

HOVER_LINK_HEALTH_MAX_LATENCY_MS = 150.0
HOVER_LINK_HEALTH_MIN_LINK_QUALITY = 30.0
HOVER_LINK_HEALTH_GRACE_S = 0.35
HOVER_LINK_HEALTH_MAX_OBSERVATION_AGE_S = 0.35

SUPERVISOR_INFO_CAN_ARM_BIT = 0
SUPERVISOR_INFO_IS_ARMED_BIT = 1
SUPERVISOR_INFO_AUTO_ARM_BIT = 2
SUPERVISOR_INFO_CAN_FLY_BIT = 3
SUPERVISOR_INFO_IS_FLYING_BIT = 4
SUPERVISOR_INFO_IS_TUMBLED_BIT = 5
SUPERVISOR_INFO_IS_LOCKED_BIT = 6
SUPERVISOR_INFO_IS_CRASHED_BIT = 7
SUPERVISOR_INFO_HIGH_LEVEL_ACTIVE_BIT = 8
SUPERVISOR_INFO_HIGH_LEVEL_FINISHED_BIT = 9
SUPERVISOR_INFO_HIGH_LEVEL_DISABLED_BIT = 10


def compute_takeoff_active_height_m(
    target_height_m: float,
    *,
    min_height_m: float = HOVER_TAKEOFF_CONFIRM_MIN_HEIGHT_M,
    target_height_tolerance_m: float = HOVER_TAKEOFF_TARGET_HEIGHT_TOLERANCE_M,
) -> float:
    """Return the shared height threshold that marks a bounded takeoff as active.

    This helper is the single contract used by the host takeoff confirmation
    gate, hover acceptance, and the on-device failsafe arming policy. It keeps
    the bounded hover lane from drifting into multiple incompatible definitions
    of "airborne enough to trust lateral clearance and hover acceptance".
    """

    bounded_target_height_m = max(0.0, float(target_height_m))
    bounded_min_height_m = max(0.0, float(min_height_m))
    bounded_tolerance_m = max(0.0, float(target_height_tolerance_m))
    return min(
        bounded_target_height_m,
        max(bounded_min_height_m, bounded_target_height_m - bounded_tolerance_m),
    )


@dataclass(frozen=True, slots=True)
class HoverPreArmConfig:
    """Describe the deterministic estimator/controller setup before takeoff."""

    estimator: int = 2
    controller: int = 1
    motion_disable: int = 0
    require_motion_disable_param: bool = True
    reset_wait_s: float = HOVER_ESTIMATOR_RESET_SETTLE_S
    verify_attempts: int = HOVER_PARAM_VERIFY_ATTEMPTS
    verify_settle_s: float = HOVER_PARAM_VERIFY_SETTLE_S


@dataclass(frozen=True, slots=True)
class HoverPreArmSnapshot:
    """Record the final pre-arm controller values persisted in hover artifacts."""

    estimator_requested: int
    estimator: int | None
    controller_requested: int
    controller: int | None
    motion_disable_requested: int
    motion_disable: int | None
    kalman_reset_after: int | None
    kalman_reset_performed: bool
    verified: bool
    failures: tuple[str, ...]
    estimator_before: int | None = None
    controller_before: int | None = None
    motion_disable_before: int | None = None


@dataclass(frozen=True, slots=True)
class HoverPreArmRestoreReport:
    """Summarize restoration of pre-arm params captured before the hover test."""

    estimator_target: int | None
    estimator: int | None
    controller_target: int | None
    controller: int | None
    motion_disable_target: int | None
    motion_disable: int | None
    kalman_reset_after: int | None
    verified: bool
    failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HoverEstimatorSettlingConfig:
    """Describe the bounded Kalman-settling gate before hover takeoff."""

    timeout_s: float = HOVER_ESTIMATOR_SETTLE_TIMEOUT_S
    period_in_ms: int = HOVER_ESTIMATOR_SETTLE_PERIOD_MS
    window_size: int = HOVER_ESTIMATOR_SETTLE_WINDOW_SAMPLES
    variance_threshold: float = HOVER_ESTIMATOR_VARIANCE_THRESHOLD
    attitude_abs_max_deg: float = HOVER_ESTIMATOR_ATTITUDE_ABS_MAX_DEG
    motion_squal_min: int = HOVER_ESTIMATOR_MOTION_SQUAL_MIN
    motion_squal_required_ratio: float = HOVER_ESTIMATOR_MOTION_SQUAL_REQUIRED_RATIO
    flow_gate_min_height_m: float = HOVER_ESTIMATOR_FLOW_GATE_MIN_HEIGHT_M
    require_motion_squal: bool = True
    require_ground_range: bool = True


@dataclass(frozen=True, slots=True)
class HoverEstimatorSettlingReport:
    """Summarize whether the estimator settled cleanly before takeoff."""

    stable: bool
    sample_count: int
    duration_s: float | None
    var_px_span: float | None
    var_py_span: float | None
    var_pz_span: float | None
    roll_abs_max_deg: float | None
    pitch_abs_max_deg: float | None
    motion_squal_min: int | None
    motion_squal_mean: float | None
    motion_squal_nonzero_ratio: float | None
    zrange_min_m: float | None
    zrange_observed: bool
    failures: tuple[str, ...]
    zrange_max_m: float | None = None
    flow_gate_deferred: bool = False


@dataclass(frozen=True, slots=True)
class HoverLinkHealthConfig:
    """Describe an optional runtime link-health gate during takeoff/hover."""

    enabled: bool = True
    max_consecutive_unhealthy_s: float = HOVER_LINK_HEALTH_GRACE_S
    max_latency_ms: float | None = HOVER_LINK_HEALTH_MAX_LATENCY_MS
    min_link_quality: float | None = HOVER_LINK_HEALTH_MIN_LINK_QUALITY
    min_uplink_rssi: float | None = None
    min_uplink_rate_hz: float | None = None
    min_downlink_rate_hz: float | None = None
    max_uplink_congestion: float | None = None
    max_downlink_congestion: float | None = None
    max_observation_age_s: float = HOVER_LINK_HEALTH_MAX_OBSERVATION_AGE_S


@dataclass(frozen=True, slots=True)
class HoverStabilityConfig:
    """Describe the bounded hover-stability contract after takeoff."""

    settle_timeout_s: float = HOVER_STABILIZE_TIMEOUT_S
    required_stable_samples: int = HOVER_STABILITY_REQUIRED_SAMPLES
    abort_after_unstable_samples: int = HOVER_STABILITY_ABORT_SAMPLES
    max_observation_age_s: float = HOVER_STABILITY_MAX_OBSERVATION_AGE_S
    max_height_error_m: float = HOVER_STABILITY_MAX_HEIGHT_ERROR_M
    max_attitude_abs_deg: float = HOVER_STABILITY_MAX_ATTITUDE_ABS_DEG
    max_horizontal_speed_mps: float = HOVER_STABILITY_MAX_HORIZONTAL_SPEED_MPS
    max_xy_drift_m: float = HOVER_STABILITY_MAX_XY_DRIFT_M
    min_motion_squal: int = HOVER_ESTIMATOR_MOTION_SQUAL_MIN
    require_motion_squal: bool = True
    anchor_control: FlowAnchorControlConfig = field(default_factory=FlowAnchorControlConfig)
    height_trust: HeightTrustConfig = field(default_factory=HeightTrustConfig)
    lateral_trust: LateralTrustConfig = field(default_factory=LateralTrustConfig)
    trim_observer: TrimObserverConfig = field(default_factory=TrimObserverConfig)


@dataclass(frozen=True, slots=True)
class HoverPrimitiveConfig:
    """Describe one bounded takeoff-hold-land primitive."""

    target_height_m: float
    hover_duration_s: float
    takeoff_velocity_mps: float
    land_velocity_mps: float
    takeoff_confirm_min_height_m: float = HOVER_TAKEOFF_CONFIRM_MIN_HEIGHT_M
    takeoff_confirm_samples: int = HOVER_TAKEOFF_CONFIRM_SAMPLES
    takeoff_confirm_timeout_s: float = HOVER_TAKEOFF_CONFIRM_TIMEOUT_S
    takeoff_confirm_max_age_s: float = HOVER_TAKEOFF_CONFIRM_MAX_AGE_S
    takeoff_confirm_target_height_tolerance_m: float = HOVER_TAKEOFF_TARGET_HEIGHT_TOLERANCE_M
    micro_liftoff_height_m: float = HOVER_MICRO_LIFTOFF_HEIGHT_M
    setpoint_period_s: float = HOVER_SETPOINT_PERIOD_S
    landing_floor_height_m: float = HOVER_LANDING_FLOOR_HEIGHT_M
    landing_floor_settle_s: float = HOVER_LANDING_FLOOR_SETTLE_S
    touchdown_height_m: float = HOVER_TOUCHDOWN_HEIGHT_M
    touchdown_settle_s: float = HOVER_TOUCHDOWN_SETTLE_S
    touchdown_velocity_mps: float = HOVER_TOUCHDOWN_VELOCITY_MPS
    touchdown_confirm_height_m: float = HOVER_TOUCHDOWN_CONFIRM_HEIGHT_M
    touchdown_confirm_samples: int = HOVER_TOUCHDOWN_CONFIRM_SAMPLES
    touchdown_confirm_timeout_s: float = HOVER_TOUCHDOWN_CONFIRM_TIMEOUT_S
    touchdown_confirm_max_age_s: float = HOVER_TOUCHDOWN_CONFIRM_MAX_AGE_S
    touchdown_recovery_timeout_s: float = HOVER_TOUCHDOWN_RECOVERY_TIMEOUT_S
    touchdown_require_supervisor_grounded: bool = True
    touchdown_range_only_confirmation_source: str = "range_only"
    force_motor_cutoff_after_touchdown_recovery: bool = True
    zero_height_settle_s: float = HOVER_ZERO_HEIGHT_SETTLE_S
    notify_stop_flush_s: float = HOVER_NOTIFY_STOP_FLUSH_S
    link_health: HoverLinkHealthConfig | None = field(default_factory=HoverLinkHealthConfig)
    stability: HoverStabilityConfig | None = None
    vertical_bootstrap: HoverVerticalBootstrapConfig | None = None


@dataclass(frozen=True, slots=True)
class HoverTranslationConfig:
    """Describe one bounded horizontal translation at a maintained hover height."""

    forward_m: float = 0.0
    left_m: float = 0.0
    velocity_mps: float = 0.2
    target_height_m: float | None = None
    settle_duration_s: float = 0.0
    link_health: HoverLinkHealthConfig | None = field(default_factory=HoverLinkHealthConfig)
    stability: HoverStabilityConfig | None = None


@dataclass(frozen=True, slots=True)
class HoverPrimitiveOutcome:
    """Summarize one bounded hover primitive execution."""

    final_phase: str
    took_off: bool
    landed: bool
    aborted: bool
    abort_reason: str | None
    commanded_max_height_m: float
    setpoint_count: int
    forced_motor_cutoff: bool = False
    touchdown_confirmation_source: str | None = None
    touchdown_distance_m: float | None = None
    touchdown_supervisor_grounded: bool = False
    stable_hover_established: bool = False
    trim_identified: bool = False
    qualified_hover_reached: bool = False
    landing_trim_identified: bool = False
    abort_phase: str | None = None


@dataclass(frozen=True, slots=True)
class HoverGroundDistanceObservation:
    """Represent one fresh downward-range observation used during landing."""

    distance_m: float | None
    age_s: float | None
    is_flying: bool | None = None
    supervisor_age_s: float | None = None
    supervisor_info: int | None = None


@dataclass(frozen=True, slots=True)
class HoverStabilityObservation:
    """Represent one live hover-stability snapshot from the telemetry lane."""

    height_m: float | None
    height_age_s: float | None
    z_estimate_m: float | None = None
    z_estimate_age_s: float | None = None
    x_m: float | None = None
    y_m: float | None = None
    pose_age_s: float | None = None
    vx_mps: float | None = None
    vy_mps: float | None = None
    velocity_age_s: float | None = None
    vz_mps: float | None = None
    vz_age_s: float | None = None
    roll_deg: float | None = None
    pitch_deg: float | None = None
    yaw_deg: float | None = None
    yaw_age_s: float | None = None
    attitude_age_s: float | None = None
    motion_squal: int | None = None
    motion_squal_age_s: float | None = None
    is_flying: bool | None = None
    supervisor_age_s: float | None = None
    supervisor_info: int | None = None


@dataclass(frozen=True, slots=True)
class HoverLinkHealthObservation:
    """Represent one latest link-health sample from cflib or an external source."""

    age_s: float | None
    latency_ms: float | None = None
    link_quality: float | None = None
    uplink_rssi: float | None = None
    uplink_rate_hz: float | None = None
    downlink_rate_hz: float | None = None
    uplink_congestion: float | None = None
    downlink_congestion: float | None = None


class HoverPrimitiveAbort(RuntimeError):
    """Signal that the bounded hover primitive was asked to abort."""


class HoverPrimitiveSafetyError(RuntimeError):
    """Signal that a bounded hover safety gate prevented a safe landing cut."""


def _emit_trace(
    trace_writer: Any | None,
    phase: str,
    *,
    status: str,
    message: str | None = None,
    data: Mapping[str, object] | None = None,
) -> None:
    """Emit one optional trace event without depending on a concrete writer type.

    Trace/telemetry failures must never become flight-critical.
    """

    if trace_writer is None:
        return
    emit = getattr(trace_writer, "emit", None)
    if emit is None:
        return
    try:
        emit(
            phase,
            status=status,
            message=message,
            data=data,
        )
    except Exception:
        return


def _param_handle(sync_cf: Any) -> Any:
    """Return the Crazyflie param handle from a Crazyflie or SyncCrazyflie object."""

    return sync_cf.cf.param if hasattr(sync_cf, "cf") else sync_cf.param


def _normalize_float(raw: object) -> float | None:
    """Normalize one payload value into a finite float when possible."""

    if raw is None:
        return None
    try:
        value = float(cast(Any, raw))
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _normalize_int(raw: object) -> int | None:
    """Normalize one payload value into an integer when possible."""

    value = _normalize_float(raw)
    if value is None:
        return None
    return int(value)


def _normalize_bool(raw: object) -> bool | None:
    """Normalize a truthy payload into a boolean when possible."""

    if isinstance(raw, bool):
        return raw
    value = _normalize_int(raw)
    if value is None:
        return None
    return bool(value)


def _read_param_int(sync_cf: Any, name: str) -> int | None:
    """Read one Crazyflie param and normalize it to an integer when possible."""

    param = _param_handle(sync_cf)
    try:
        raw = param.get_value(name)
    except Exception:
        return None
    return _normalize_int(raw)


def _set_and_verify_param(
    sync_cf: Any,
    name: str,
    value: int,
    *,
    config: HoverPreArmConfig,
    sleep: Callable[[float], None] = time.sleep,
) -> int | None:
    """Set one Crazyflie param and poll until the firmware reports the new value."""

    param = _param_handle(sync_cf)
    param.set_value(name, str(int(value)))
    for _attempt in range(max(1, int(config.verify_attempts))):
        if config.verify_settle_s > 0:
            sleep(config.verify_settle_s)
        observed = _read_param_int(sync_cf, name)
        if observed == int(value):
            return observed
    return _read_param_int(sync_cf, name)


def _wait_for_param_value(
    sync_cf: Any,
    name: str,
    expected_value: int,
    *,
    config: HoverPreArmConfig,
    sleep: Callable[[float], None] = time.sleep,
) -> int | None:
    """Poll one Crazyflie param until it reports the expected integer value."""

    for _attempt in range(max(1, int(config.verify_attempts))):
        if config.verify_settle_s > 0:
            sleep(config.verify_settle_s)
        observed = _read_param_int(sync_cf, name)
        if observed == int(expected_value):
            return observed
    return _read_param_int(sync_cf, name)


def _supervisor_has_flag(supervisor_info: int | None, bit: int) -> bool | None:
    """Return whether one supervisor-info bit is asserted."""

    if supervisor_info is None:
        return None
    if bit < 0:
        return None
    return bool(int(supervisor_info) & (1 << int(bit)))


def _supervisor_is_flying(supervisor_info: int | None) -> bool | None:
    """Decode the current 'is flying' state from supervisor.info."""

    return _supervisor_has_flag(supervisor_info, SUPERVISOR_INFO_IS_FLYING_BIT)


def apply_hover_pre_arm(
    sync_cf: Any,
    *,
    config: HoverPreArmConfig = HoverPreArmConfig(),
    trace_writer: Any | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> HoverPreArmSnapshot:
    """Apply and verify the deterministic hover-time controller setup.

    Args:
        sync_cf: ``Crazyflie`` or ``SyncCrazyflie`` instance with a live link.
        config: Desired estimator/controller settings for bounded hover tests.
        trace_writer: Optional phase-trace sink with an ``emit()`` method.
        sleep: Sleep function used for reset timing; injectable for tests.

    Returns:
        A snapshot of the final observed controller params and whether they
        matched the requested values.
    """

    estimator_before = _read_param_int(sync_cf, "stabilizer.estimator")
    controller_before = _read_param_int(sync_cf, "stabilizer.controller")
    motion_disable_before = (
        _read_param_int(sync_cf, "motion.disable") if config.require_motion_disable_param else None
    )

    _emit_trace(
        trace_writer,
        "pre_arm_params",
        status="begin",
        data={
            "estimator": config.estimator,
            "controller": config.controller,
            "motion_disable": config.motion_disable,
            "estimator_before": estimator_before,
            "controller_before": controller_before,
            "motion_disable_before": motion_disable_before,
        },
    )

    failures: list[str] = []
    estimator_value = _set_and_verify_param(
        sync_cf,
        "stabilizer.estimator",
        config.estimator,
        config=config,
        sleep=sleep,
    )
    controller_value = _set_and_verify_param(
        sync_cf,
        "stabilizer.controller",
        config.controller,
        config=config,
        sleep=sleep,
    )
    motion_disable_value: int | None = None
    if config.require_motion_disable_param:
        motion_disable_value = _set_and_verify_param(
            sync_cf,
            "motion.disable",
            config.motion_disable,
            config=config,
            sleep=sleep,
        )

    kalman_reset_after: int | None = None
    kalman_reset_performed = False
    try:
        param = _param_handle(sync_cf)
        param.set_value("kalman.resetEstimation", "1")
        kalman_reset_performed = True
        if config.reset_wait_s > 0:
            sleep(config.reset_wait_s)
        param.set_value("kalman.resetEstimation", "0")
        kalman_reset_after = _wait_for_param_value(
            sync_cf,
            "kalman.resetEstimation",
            0,
            config=config,
            sleep=sleep,
        )
    except Exception as exc:
        failures.append(f"kalman.resetEstimation failed during pre-arm setup: {exc}")

    if estimator_value != int(config.estimator):
        failures.append(
            "stabilizer.estimator did not verify to "
            f"{config.estimator}; observed {estimator_value!r}"
        )
    if controller_value != int(config.controller):
        failures.append(
            "stabilizer.controller did not verify to "
            f"{config.controller}; observed {controller_value!r}"
        )
    if config.require_motion_disable_param and motion_disable_value != int(config.motion_disable):
        failures.append(
            "motion.disable did not verify to "
            f"{config.motion_disable}; observed {motion_disable_value!r}"
        )
    if kalman_reset_performed and kalman_reset_after != 0:
        failures.append(
            "kalman.resetEstimation did not return to 0 after reset; "
            f"observed {kalman_reset_after!r}"
        )

    snapshot = HoverPreArmSnapshot(
        estimator_requested=int(config.estimator),
        estimator=estimator_value,
        controller_requested=int(config.controller),
        controller=controller_value,
        motion_disable_requested=int(config.motion_disable),
        motion_disable=motion_disable_value,
        kalman_reset_after=kalman_reset_after,
        kalman_reset_performed=kalman_reset_performed,
        verified=not failures,
        failures=tuple(failures),
        estimator_before=estimator_before,
        controller_before=controller_before,
        motion_disable_before=motion_disable_before,
    )
    _emit_trace(
        trace_writer,
        "pre_arm_params",
        status="done" if snapshot.verified else "blocked",
        data={
            "verified": snapshot.verified,
            "failures": snapshot.failures,
            "estimator": snapshot.estimator,
            "controller": snapshot.controller,
            "motion_disable": snapshot.motion_disable,
            "kalman_reset_after": snapshot.kalman_reset_after,
        },
    )
    return snapshot


def restore_hover_pre_arm(
    sync_cf: Any,
    snapshot: HoverPreArmSnapshot,
    *,
    config: HoverPreArmConfig = HoverPreArmConfig(),
    trace_writer: Any | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> HoverPreArmRestoreReport:
    """Restore pre-arm controller params captured before the hover test."""

    failures: list[str] = []
    _emit_trace(
        trace_writer,
        "pre_arm_restore",
        status="begin",
        data={
            "estimator_target": snapshot.estimator_before,
            "controller_target": snapshot.controller_before,
            "motion_disable_target": snapshot.motion_disable_before,
        },
    )

    estimator_value: int | None = None
    controller_value: int | None = None
    motion_disable_value: int | None = None

    if snapshot.estimator_before is None:
        failures.append("original stabilizer.estimator is unknown; cannot restore")
    else:
        estimator_value = _set_and_verify_param(
            sync_cf,
            "stabilizer.estimator",
            snapshot.estimator_before,
            config=config,
            sleep=sleep,
        )
        if estimator_value != snapshot.estimator_before:
            failures.append(
                "failed to restore stabilizer.estimator to "
                f"{snapshot.estimator_before}; observed {estimator_value!r}"
            )

    if snapshot.controller_before is None:
        failures.append("original stabilizer.controller is unknown; cannot restore")
    else:
        controller_value = _set_and_verify_param(
            sync_cf,
            "stabilizer.controller",
            snapshot.controller_before,
            config=config,
            sleep=sleep,
        )
        if controller_value != snapshot.controller_before:
            failures.append(
                "failed to restore stabilizer.controller to "
                f"{snapshot.controller_before}; observed {controller_value!r}"
            )

    if not config.require_motion_disable_param:
        motion_disable_value = None
    elif snapshot.motion_disable_before is None:
        failures.append("original motion.disable is unknown; cannot restore")
    else:
        motion_disable_value = _set_and_verify_param(
            sync_cf,
            "motion.disable",
            snapshot.motion_disable_before,
            config=config,
            sleep=sleep,
        )
        if motion_disable_value != snapshot.motion_disable_before:
            failures.append(
                "failed to restore motion.disable to "
                f"{snapshot.motion_disable_before}; observed {motion_disable_value!r}"
            )

    kalman_reset_after: int | None = None
    try:
        param = _param_handle(sync_cf)
        param.set_value("kalman.resetEstimation", "1")
        if config.reset_wait_s > 0:
            sleep(config.reset_wait_s)
        param.set_value("kalman.resetEstimation", "0")
        kalman_reset_after = _wait_for_param_value(
            sync_cf,
            "kalman.resetEstimation",
            0,
            config=config,
            sleep=sleep,
        )
        if kalman_reset_after != 0:
            failures.append(
                "kalman.resetEstimation did not return to 0 after restore reset; "
                f"observed {kalman_reset_after!r}"
            )
    except Exception as exc:
        failures.append(f"kalman.resetEstimation failed during pre-arm restore: {exc}")

    report = HoverPreArmRestoreReport(
        estimator_target=snapshot.estimator_before,
        estimator=estimator_value,
        controller_target=snapshot.controller_before,
        controller=controller_value,
        motion_disable_target=snapshot.motion_disable_before,
        motion_disable=motion_disable_value,
        kalman_reset_after=kalman_reset_after,
        verified=not failures,
        failures=tuple(failures),
    )
    _emit_trace(
        trace_writer,
        "pre_arm_restore",
        status="done" if report.verified else "blocked",
        data={
            "verified": report.verified,
            "failures": report.failures,
            "estimator": report.estimator,
            "controller": report.controller,
            "motion_disable": report.motion_disable,
            "kalman_reset_after": report.kalman_reset_after,
        },
    )
    return report


def _normalized_zrange_m(raw: object) -> float | None:
    """Convert one ``range.zrange`` sample from mm to meters when valid."""

    value = _normalize_float(raw)
    if value is None or value <= 0.0 or value >= HOVER_ESTIMATOR_RANGE_INVALID_MM:
        return None
    return value / 1000.0


def _series_span(values: Iterable[float | None]) -> float | None:
    """Return the numeric span for a series when at least one value exists."""

    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return max(filtered) - min(filtered)


def _series_abs_max(values: Iterable[float | None]) -> float | None:
    """Return the largest absolute finite value in a series."""

    filtered = [abs(value) for value in values if value is not None]
    if not filtered:
        return None
    return max(filtered)


def _settling_failures(
    samples: Sequence[Mapping[str, float | None]],
    *,
    config: HoverEstimatorSettlingConfig,
) -> list[str]:
    """Return the concrete reasons a recent estimator window is not yet stable."""

    failures: list[str] = []
    if len(samples) < max(1, int(config.window_size)):
        failures.append(
            f"only {len(samples)} estimator samples available; need {int(config.window_size)} for settle gate"
        )
        return failures

    zrange_values = [
        float(zrange_m)
        for sample in samples
        for zrange_m in (sample.get("range.zrange_m"),)
        if zrange_m is not None
    ]
    zrange_max_m = max(zrange_values) if zrange_values else None
    flow_gate_deferred = zrange_max_m is not None and zrange_max_m < float(config.flow_gate_min_height_m)

    px_span = _series_span(sample.get("kalman.varPX") for sample in samples)
    py_span = _series_span(sample.get("kalman.varPY") for sample in samples)
    pz_span = _series_span(sample.get("kalman.varPZ") for sample in samples)
    variance_checks = [("kalman.varPZ", pz_span)]
    if not flow_gate_deferred:
        variance_checks.insert(0, ("kalman.varPY", py_span))
        variance_checks.insert(0, ("kalman.varPX", px_span))
    for label, span in variance_checks:
        if span is None or span > float(config.variance_threshold):
            failures.append(
                f"{label} span {span if span is not None else 'unknown'} exceeds the "
                f"{config.variance_threshold:.4f} estimator settle gate"
            )

    roll_abs_max = _series_abs_max(sample.get("stabilizer.roll") for sample in samples)
    pitch_abs_max = _series_abs_max(sample.get("stabilizer.pitch") for sample in samples)
    if roll_abs_max is None or roll_abs_max > float(config.attitude_abs_max_deg):
        failures.append(
            "roll attitude is not yet quiet enough for takeoff "
            f"(max {roll_abs_max if roll_abs_max is not None else 'unknown'} deg)"
        )
    if pitch_abs_max is None or pitch_abs_max > float(config.attitude_abs_max_deg):
        failures.append(
            "pitch attitude is not yet quiet enough for takeoff "
            f"(max {pitch_abs_max if pitch_abs_max is not None else 'unknown'} deg)"
        )

    if config.require_motion_squal and not flow_gate_deferred:
        squal_values: list[int] = []
        for sample in samples:
            sample_value = sample.get("motion.squal")
            if sample_value is not None:
                squal_values.append(int(sample_value))
        if not squal_values:
            failures.append("motion.squal never produced a usable estimator-settle sample")
        else:
            squal_mean = sum(squal_values) / len(squal_values)
            squal_good_ratio = sum(
                1 for value in squal_values if value >= int(config.motion_squal_min)
            ) / len(squal_values)
            if squal_mean < float(config.motion_squal_min):
                failures.append(
                    f"motion.squal mean {squal_mean:.1f} is below the {config.motion_squal_min} settle gate"
                )
            if squal_good_ratio < float(config.motion_squal_required_ratio):
                failures.append(
                    "motion.squal stayed below the settle gate too often "
                    f"({squal_good_ratio:.2f} < {config.motion_squal_required_ratio:.2f})"
                )

    if config.require_ground_range and not zrange_values:
        failures.append("range.zrange never produced a valid pre-arm ground reading")
    return failures


def wait_for_estimator_settle(
    sync_cf: Any,
    log_config_cls: Any,
    sync_logger_cls: Any,
    *,
    config: HoverEstimatorSettlingConfig = HoverEstimatorSettlingConfig(),
    trace_writer: Any | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> HoverEstimatorSettlingReport:
    """Wait for the Kalman estimator to settle before bounded takeoff.

    The gate combines the official Bitcraze Kalman-variance check with extra
    Twinr safety signals: quiet roll/pitch on the bench, usable optical-flow
    quality, and a valid downward range reading before takeoff.

    Bitcraze documents the PMW3901 optical-flow sensor as only tracking motion
    from about 80 mm and has suggested ignoring flow measurements below roughly
    100 mm during takeoff. When the Crazyflie is still on the floor below that
    band, this gate defers the XY/flow-specific checks and only enforces the
    vertical/attitude bench checks until the craft can actually lift into the
    flow sensor's usable range.
    """

    _emit_trace(
        trace_writer,
        "estimator_settle",
        status="begin",
        data={
            "timeout_s": config.timeout_s,
            "window_size": config.window_size,
            "variance_threshold": config.variance_threshold,
            "motion_squal_min": config.motion_squal_min,
            "flow_gate_min_height_m": config.flow_gate_min_height_m,
            "require_motion_squal": config.require_motion_squal,
            "require_ground_range": config.require_ground_range,
        },
    )

    recent: deque[dict[str, float | None]] = deque(maxlen=max(1, int(config.window_size)))
    samples: list[dict[str, float | None]] = []
    stable = False
    start = monotonic()

    try:
        log_config = log_config_cls(name="hover-estimator-settle", period_in_ms=max(10, int(config.period_in_ms)))
        log_config.add_variable("kalman.varPX", "float")
        log_config.add_variable("kalman.varPY", "float")
        log_config.add_variable("kalman.varPZ", "float")
        if config.require_motion_squal:
            log_config.add_variable("motion.squal", "uint16_t")
        log_config.add_variable("stabilizer.roll", "float")
        log_config.add_variable("stabilizer.pitch", "float")
        if config.require_ground_range:
            log_config.add_variable("range.zrange", "uint16_t")
    except Exception as exc:
        config_failures = (f"estimator settling logger could not be configured: {exc}",)
        report = HoverEstimatorSettlingReport(
            stable=False,
            sample_count=0,
            duration_s=None,
            var_px_span=None,
            var_py_span=None,
            var_pz_span=None,
            roll_abs_max_deg=None,
            pitch_abs_max_deg=None,
            motion_squal_min=None,
            motion_squal_mean=None,
            motion_squal_nonzero_ratio=None,
            zrange_min_m=None,
            zrange_max_m=None,
            zrange_observed=False,
            flow_gate_deferred=False,
            failures=config_failures,
        )
        _emit_trace(
            trace_writer,
            "estimator_settle",
            status="blocked",
            data={"stable": False, "sample_count": 0, "failures": report.failures},
        )
        return report

    try:
        with sync_logger_cls(sync_cf, log_config) as logger:
            for _timestamp, data, _log_block in logger:
                sample = {
                    "kalman.varPX": _normalize_float(data.get("kalman.varPX")),
                    "kalman.varPY": _normalize_float(data.get("kalman.varPY")),
                    "kalman.varPZ": _normalize_float(data.get("kalman.varPZ")),
                    "stabilizer.roll": _normalize_float(data.get("stabilizer.roll")),
                    "stabilizer.pitch": _normalize_float(data.get("stabilizer.pitch")),
                }
                if config.require_ground_range:
                    sample["range.zrange_m"] = _normalized_zrange_m(data.get("range.zrange"))
                if config.require_motion_squal:
                    sample["motion.squal"] = _normalize_float(data.get("motion.squal"))
                samples.append(sample)
                recent.append(sample)
                if not _settling_failures(tuple(recent), config=config):
                    stable = True
                    break
                if monotonic() - start >= float(config.timeout_s):
                    break
    except Exception as exc:
        recent_window = tuple(recent)
        read_failures = [f"estimator settling logger failed while reading samples: {exc}"]
        read_failures.extend(_settling_failures(recent_window, config=config))
        report = HoverEstimatorSettlingReport(
            stable=False,
            sample_count=len(samples),
            duration_s=max(0.0, monotonic() - start) if samples else None,
            var_px_span=_series_span(sample.get("kalman.varPX") for sample in recent_window),
            var_py_span=_series_span(sample.get("kalman.varPY") for sample in recent_window),
            var_pz_span=_series_span(sample.get("kalman.varPZ") for sample in recent_window),
            roll_abs_max_deg=_series_abs_max(sample.get("stabilizer.roll") for sample in recent_window),
            pitch_abs_max_deg=_series_abs_max(sample.get("stabilizer.pitch") for sample in recent_window),
            motion_squal_min=None,
            motion_squal_mean=None,
            motion_squal_nonzero_ratio=None,
            zrange_min_m=None,
            zrange_max_m=None,
            zrange_observed=any(sample.get("range.zrange_m") is not None for sample in recent_window),
            flow_gate_deferred=False,
            failures=tuple(read_failures),
        )
        _emit_trace(
            trace_writer,
            "estimator_settle",
            status="blocked",
            data={
                "stable": False,
                "sample_count": report.sample_count,
                "failures": report.failures,
            },
        )
        return report

    recent_window = tuple(recent)
    settle_failures: list[str] = [] if stable else _settling_failures(recent_window, config=config)
    if not stable:
        settle_failures.insert(
            0,
            f"estimator settling timed out after {float(config.timeout_s):.1f} s before takeoff",
        )

    squal_values: list[int] = []
    zrange_values: list[float] = []
    for sample in recent_window:
        motion_squal = sample.get("motion.squal")
        if motion_squal is not None:
            squal_values.append(int(motion_squal))
        zrange_m = sample.get("range.zrange_m")
        if zrange_m is not None:
            zrange_values.append(zrange_m)
    zrange_max_m = max(zrange_values) if zrange_values else None
    flow_gate_deferred = zrange_max_m is not None and zrange_max_m < float(config.flow_gate_min_height_m)

    report = HoverEstimatorSettlingReport(
        stable=stable,
        sample_count=len(samples),
        duration_s=max(0.0, monotonic() - start) if samples else None,
        var_px_span=_series_span(sample.get("kalman.varPX") for sample in recent_window),
        var_py_span=_series_span(sample.get("kalman.varPY") for sample in recent_window),
        var_pz_span=_series_span(sample.get("kalman.varPZ") for sample in recent_window),
        roll_abs_max_deg=_series_abs_max(sample.get("stabilizer.roll") for sample in recent_window),
        pitch_abs_max_deg=_series_abs_max(sample.get("stabilizer.pitch") for sample in recent_window),
        motion_squal_min=min(squal_values) if squal_values else None,
        motion_squal_mean=(sum(squal_values) / len(squal_values)) if squal_values else None,
        motion_squal_nonzero_ratio=(
            sum(1 for value in squal_values if value > 0) / len(squal_values) if squal_values else None
        ),
        zrange_min_m=min(zrange_values) if zrange_values else None,
        zrange_max_m=zrange_max_m,
        zrange_observed=bool(zrange_values),
        flow_gate_deferred=flow_gate_deferred,
        failures=tuple(settle_failures),
    )
    _emit_trace(
        trace_writer,
        "estimator_settle",
        status="done" if report.stable else "blocked",
        data={
            "stable": report.stable,
            "sample_count": report.sample_count,
            "failures": report.failures,
            "motion_squal_mean": report.motion_squal_mean,
            "zrange_min_m": report.zrange_min_m,
            "zrange_max_m": report.zrange_max_m,
            "flow_gate_deferred": report.flow_gate_deferred,
        },
    )
    return report


def _mapping_or_attr_get(raw: object, name: str) -> object:
    """Fetch a field from a mapping-like or attribute-like object."""

    if isinstance(raw, Mapping):
        return raw.get(name)
    return getattr(raw, name, None)


def _coerce_ground_distance_observation(raw: object) -> HoverGroundDistanceObservation:
    """Normalize external touchdown telemetry into HoverGroundDistanceObservation."""

    if isinstance(raw, HoverGroundDistanceObservation):
        supervisor_info = raw.supervisor_info
        is_flying = raw.is_flying
        if is_flying is None:
            is_flying = _supervisor_is_flying(supervisor_info)
        return HoverGroundDistanceObservation(
            distance_m=_normalize_float(raw.distance_m),
            age_s=_normalize_float(raw.age_s),
            is_flying=is_flying,
            supervisor_age_s=_normalize_float(raw.supervisor_age_s),
            supervisor_info=_normalize_int(supervisor_info),
        )

    supervisor_info = _normalize_int(_mapping_or_attr_get(raw, "supervisor_info"))
    is_flying = _normalize_bool(_mapping_or_attr_get(raw, "is_flying"))
    if is_flying is None:
        is_flying = _supervisor_is_flying(supervisor_info)

    return HoverGroundDistanceObservation(
        distance_m=_normalize_float(_mapping_or_attr_get(raw, "distance_m")),
        age_s=_normalize_float(_mapping_or_attr_get(raw, "age_s")),
        is_flying=is_flying,
        supervisor_age_s=_normalize_float(_mapping_or_attr_get(raw, "supervisor_age_s")),
        supervisor_info=supervisor_info,
    )


def _coerce_link_health_observation(raw: object) -> HoverLinkHealthObservation:
    """Normalize external cflib link statistics into one observation object."""

    if isinstance(raw, HoverLinkHealthObservation):
        return HoverLinkHealthObservation(
            age_s=_normalize_float(raw.age_s),
            latency_ms=_normalize_float(raw.latency_ms),
            link_quality=_normalize_float(raw.link_quality),
            uplink_rssi=_normalize_float(raw.uplink_rssi),
            uplink_rate_hz=_normalize_float(raw.uplink_rate_hz),
            downlink_rate_hz=_normalize_float(raw.downlink_rate_hz),
            uplink_congestion=_normalize_float(raw.uplink_congestion),
            downlink_congestion=_normalize_float(raw.downlink_congestion),
        )

    return HoverLinkHealthObservation(
        age_s=_normalize_float(_mapping_or_attr_get(raw, "age_s")),
        latency_ms=_normalize_float(_mapping_or_attr_get(raw, "latency_ms")),
        link_quality=_normalize_float(_mapping_or_attr_get(raw, "link_quality")),
        uplink_rssi=_normalize_float(_mapping_or_attr_get(raw, "uplink_rssi")),
        uplink_rate_hz=_normalize_float(_mapping_or_attr_get(raw, "uplink_rate_hz")),
        downlink_rate_hz=_normalize_float(_mapping_or_attr_get(raw, "downlink_rate_hz")),
        uplink_congestion=_normalize_float(_mapping_or_attr_get(raw, "uplink_congestion")),
        downlink_congestion=_normalize_float(_mapping_or_attr_get(raw, "downlink_congestion")),
    )


def _coerce_hover_stability_observation(raw: object) -> HoverStabilityObservation:
    """Normalize external hover-stability telemetry into one observation object."""

    if isinstance(raw, HoverStabilityObservation):
        supervisor_info = _normalize_int(raw.supervisor_info)
        is_flying = raw.is_flying
        if is_flying is None:
            is_flying = _supervisor_is_flying(supervisor_info)
        return HoverStabilityObservation(
            height_m=_normalize_float(raw.height_m),
            height_age_s=_normalize_float(raw.height_age_s),
            z_estimate_m=_normalize_float(raw.z_estimate_m),
            z_estimate_age_s=_normalize_float(raw.z_estimate_age_s),
            x_m=_normalize_float(raw.x_m),
            y_m=_normalize_float(raw.y_m),
            pose_age_s=_normalize_float(raw.pose_age_s),
            vx_mps=_normalize_float(raw.vx_mps),
            vy_mps=_normalize_float(raw.vy_mps),
            velocity_age_s=_normalize_float(raw.velocity_age_s),
            vz_mps=_normalize_float(raw.vz_mps),
            vz_age_s=_normalize_float(raw.vz_age_s),
            roll_deg=_normalize_float(raw.roll_deg),
            pitch_deg=_normalize_float(raw.pitch_deg),
            yaw_deg=_normalize_float(raw.yaw_deg),
            yaw_age_s=_normalize_float(raw.yaw_age_s),
            attitude_age_s=_normalize_float(raw.attitude_age_s),
            motion_squal=_normalize_int(raw.motion_squal),
            motion_squal_age_s=_normalize_float(raw.motion_squal_age_s),
            is_flying=is_flying,
            supervisor_age_s=_normalize_float(raw.supervisor_age_s),
            supervisor_info=supervisor_info,
        )

    supervisor_info = _normalize_int(_mapping_or_attr_get(raw, "supervisor_info"))
    is_flying = _normalize_bool(_mapping_or_attr_get(raw, "is_flying"))
    if is_flying is None:
        is_flying = _supervisor_is_flying(supervisor_info)
    return HoverStabilityObservation(
        height_m=_normalize_float(_mapping_or_attr_get(raw, "height_m")),
        height_age_s=_normalize_float(_mapping_or_attr_get(raw, "height_age_s")),
        z_estimate_m=_normalize_float(_mapping_or_attr_get(raw, "z_estimate_m")),
        z_estimate_age_s=_normalize_float(_mapping_or_attr_get(raw, "z_estimate_age_s")),
        x_m=_normalize_float(_mapping_or_attr_get(raw, "x_m")),
        y_m=_normalize_float(_mapping_or_attr_get(raw, "y_m")),
        pose_age_s=_normalize_float(_mapping_or_attr_get(raw, "pose_age_s")),
        vx_mps=_normalize_float(_mapping_or_attr_get(raw, "vx_mps")),
        vy_mps=_normalize_float(_mapping_or_attr_get(raw, "vy_mps")),
        velocity_age_s=_normalize_float(_mapping_or_attr_get(raw, "velocity_age_s")),
        vz_mps=_normalize_float(_mapping_or_attr_get(raw, "vz_mps")),
        vz_age_s=_normalize_float(_mapping_or_attr_get(raw, "vz_age_s")),
        roll_deg=_normalize_float(_mapping_or_attr_get(raw, "roll_deg")),
        pitch_deg=_normalize_float(_mapping_or_attr_get(raw, "pitch_deg")),
        yaw_deg=_normalize_float(_mapping_or_attr_get(raw, "yaw_deg")),
        yaw_age_s=_normalize_float(_mapping_or_attr_get(raw, "yaw_age_s")),
        attitude_age_s=_normalize_float(_mapping_or_attr_get(raw, "attitude_age_s")),
        motion_squal=_normalize_int(_mapping_or_attr_get(raw, "motion_squal")),
        motion_squal_age_s=_normalize_float(_mapping_or_attr_get(raw, "motion_squal_age_s")),
        is_flying=is_flying,
        supervisor_age_s=_normalize_float(_mapping_or_attr_get(raw, "supervisor_age_s")),
        supervisor_info=supervisor_info,
    )


def _compute_hover_control_command(
    observation: HoverStabilityObservation,
    *,
    target_height_m: float,
    anchor_xy: tuple[float, float] | None,
    config: HoverStabilityConfig,
) -> FlowAnchorControlCommand:
    """Project one stability observation into a bounded outer-loop hover command."""

    return compute_flow_anchor_command(
        observation=FlowAnchorObservation(
            x_m=observation.x_m,
            y_m=observation.y_m,
            pose_age_s=observation.pose_age_s,
            vx_mps=observation.vx_mps,
            vy_mps=observation.vy_mps,
            velocity_age_s=observation.velocity_age_s,
            yaw_deg=observation.yaw_deg,
            yaw_age_s=observation.yaw_age_s,
            raw_height_m=observation.height_m,
            raw_height_age_s=observation.height_age_s,
            estimate_z_m=observation.z_estimate_m,
            estimate_z_age_s=observation.z_estimate_age_s,
        ),
        anchor_xy=anchor_xy,
        target_height_m=target_height_m,
        control_config=config.anchor_control,
        height_trust_config=config.height_trust,
    )


def _hover_stability_violations(
    observation: HoverStabilityObservation,
    *,
    target_height_m: float,
    anchor_xy: tuple[float, float] | None,
    config: HoverStabilityConfig,
    control_command: FlowAnchorControlCommand,
) -> list[tuple[str, str]]:
    """Return current hover-stability violations for the guarded bounded hover lane."""

    violations: list[tuple[str, str]] = []
    max_age_s = float(config.max_observation_age_s)
    trusted_height_m = control_command.trusted_height_m

    for failure in control_command.failures:
        if failure.startswith("trusted height is unavailable"):
            violations.append(("height_untrusted", failure))
        elif failure.startswith("flow anchor control is unavailable"):
            violations.append(("anchor_control", failure))

    if trusted_height_m is None:
        violations.append(("height_not_held", "trusted hover height is unavailable"))
    elif abs(trusted_height_m - float(target_height_m)) > float(config.max_height_error_m):
        violations.append(
            (
                "height_not_held",
                (
                    f"trusted hover height {trusted_height_m:.2f} m deviates from the "
                    f"{float(target_height_m):.2f} m target by more than "
                    f"{float(config.max_height_error_m):.2f} m"
                ),
            )
        )

    if config.require_motion_squal:
        if observation.motion_squal is None or observation.motion_squal_age_s is None:
            violations.append(("flow_untrusted", "optical-flow quality is unavailable"))
        elif observation.motion_squal_age_s > max_age_s:
            violations.append(
                (
                    "flow_untrusted",
                    f"optical-flow quality age {observation.motion_squal_age_s:.3f} s exceeds {max_age_s:.3f} s",
                )
            )
        elif int(observation.motion_squal) < int(config.min_motion_squal):
            violations.append(
                (
                    "flow_untrusted",
                    (
                        f"optical-flow quality {int(observation.motion_squal)} is below the "
                        f"{int(config.min_motion_squal)} stability floor"
                    ),
                )
            )

    if observation.is_flying is not True:
        violations.append(("height_not_held", "supervisor does not currently report the craft as flying"))
    elif observation.supervisor_age_s is not None and observation.supervisor_age_s > max_age_s:
        violations.append(
            (
                "height_not_held",
                f"supervisor flight-state age {observation.supervisor_age_s:.3f} s exceeds {max_age_s:.3f} s",
            )
        )

    if (
        observation.roll_deg is None
        or observation.pitch_deg is None
        or observation.attitude_age_s is None
    ):
        violations.append(("roll_pitch", "attitude telemetry is unavailable"))
    else:
        if observation.attitude_age_s > max_age_s:
            violations.append(
                (
                    "roll_pitch",
                    f"attitude telemetry age {observation.attitude_age_s:.3f} s exceeds {max_age_s:.3f} s",
                )
            )
        else:
            if abs(observation.roll_deg) > float(config.max_attitude_abs_deg):
                violations.append(
                    (
                        "roll_pitch",
                        (
                            f"roll {observation.roll_deg:.2f} deg exceeds the "
                            f"{float(config.max_attitude_abs_deg):.2f} deg hover guard"
                        ),
                    )
                )
            if abs(observation.pitch_deg) > float(config.max_attitude_abs_deg):
                violations.append(
                    (
                        "roll_pitch",
                        (
                            f"pitch {observation.pitch_deg:.2f} deg exceeds the "
                            f"{float(config.max_attitude_abs_deg):.2f} deg hover guard"
                        ),
                    )
                )

    if observation.vx_mps is None or observation.vy_mps is None or observation.velocity_age_s is None:
        violations.append(("speed", "horizontal velocity telemetry is unavailable"))
    else:
        if observation.velocity_age_s > max_age_s:
            violations.append(
                (
                    "speed",
                    f"horizontal velocity telemetry age {observation.velocity_age_s:.3f} s exceeds {max_age_s:.3f} s",
                )
            )
        else:
            horizontal_speed_mps = math.hypot(observation.vx_mps, observation.vy_mps)
            if horizontal_speed_mps > float(config.max_horizontal_speed_mps):
                violations.append(
                    (
                        "speed",
                        (
                            f"horizontal speed {horizontal_speed_mps:.2f} m/s exceeds the "
                            f"{float(config.max_horizontal_speed_mps):.2f} m/s hover guard"
                        ),
                    )
                )

    if anchor_xy is None or observation.x_m is None or observation.y_m is None or observation.pose_age_s is None:
        violations.append(("xy_drift", "xy pose telemetry is unavailable"))
    else:
        if observation.pose_age_s > max_age_s:
            violations.append(
                (
                    "xy_drift",
                    f"xy pose telemetry age {observation.pose_age_s:.3f} s exceeds {max_age_s:.3f} s",
                )
            )
        else:
            drift_m = math.hypot(observation.x_m - anchor_xy[0], observation.y_m - anchor_xy[1])
            if drift_m > float(config.max_xy_drift_m):
                violations.append(
                    (
                        "xy_drift",
                        (
                            f"xy drift {drift_m:.2f} m exceeds the "
                            f"{float(config.max_xy_drift_m):.2f} m hover guard"
                        ),
                    )
                )

    return violations


def _link_health_failures(
    observation: HoverLinkHealthObservation,
    *,
    config: HoverLinkHealthConfig,
) -> list[str]:
    """Return current link-health violations for the configured runtime gate."""

    failures: list[str] = []
    age_s = observation.age_s
    if age_s is None:
        failures.append("link health is unavailable")
        return failures
    if age_s > float(config.max_observation_age_s):
        failures.append(
            f"link health observation age {age_s:.3f} s exceeds {float(config.max_observation_age_s):.3f} s"
        )

    if config.max_latency_ms is not None and observation.latency_ms is not None:
        if observation.latency_ms > float(config.max_latency_ms):
            failures.append(
                f"link latency {observation.latency_ms:.1f} ms exceeds {float(config.max_latency_ms):.1f} ms"
            )

    if config.min_link_quality is not None and observation.link_quality is not None:
        if observation.link_quality < float(config.min_link_quality):
            failures.append(
                "link quality "
                f"{observation.link_quality:.1f} is below {float(config.min_link_quality):.1f}"
            )

    if config.min_uplink_rssi is not None and observation.uplink_rssi is not None:
        if observation.uplink_rssi < float(config.min_uplink_rssi):
            failures.append(
                f"uplink RSSI {observation.uplink_rssi:.1f} is below {float(config.min_uplink_rssi):.1f}"
            )

    if config.min_uplink_rate_hz is not None and observation.uplink_rate_hz is not None:
        if observation.uplink_rate_hz < float(config.min_uplink_rate_hz):
            failures.append(
                f"uplink rate {observation.uplink_rate_hz:.1f} Hz is below {float(config.min_uplink_rate_hz):.1f} Hz"
            )

    if config.min_downlink_rate_hz is not None and observation.downlink_rate_hz is not None:
        if observation.downlink_rate_hz < float(config.min_downlink_rate_hz):
            failures.append(
                "downlink rate "
                f"{observation.downlink_rate_hz:.1f} Hz is below {float(config.min_downlink_rate_hz):.1f} Hz"
            )

    if config.max_uplink_congestion is not None and observation.uplink_congestion is not None:
        if observation.uplink_congestion > float(config.max_uplink_congestion):
            failures.append(
                "uplink congestion "
                f"{observation.uplink_congestion:.2f} exceeds {float(config.max_uplink_congestion):.2f}"
            )

    if config.max_downlink_congestion is not None and observation.downlink_congestion is not None:
        if observation.downlink_congestion > float(config.max_downlink_congestion):
            failures.append(
                "downlink congestion "
                f"{observation.downlink_congestion:.2f} exceeds {float(config.max_downlink_congestion):.2f}"
            )

    return failures


class _CflibLinkHealthMonitor:
    """Adapt cflib's link_statistics callbacks into pull-based observations."""

    def __init__(self, cf: Any, *, monotonic: Callable[[], float] = time.monotonic) -> None:
        self._cf = cf
        self._monotonic = monotonic
        self._lock = Lock()
        self._available = False
        self._closed = False
        self._last_update_s: float | None = None
        self._values: dict[str, float | None] = {
            "latency_ms": None,
            "link_quality": None,
            "uplink_rssi": None,
            "uplink_rate_hz": None,
            "downlink_rate_hz": None,
            "uplink_congestion": None,
            "downlink_congestion": None,
        }
        self._subscriptions: list[tuple[Any, Callable[..., None]]] = []

        link_statistics = getattr(cf, "link_statistics", None)
        if link_statistics is None:
            return

        self._available = True
        start = getattr(link_statistics, "start", None)
        if callable(start):
            try:
                start()
            except Exception:
                pass

        self._subscribe(link_statistics, "latency_updated", "latency_ms")
        self._subscribe(link_statistics, "link_quality_updated", "link_quality")
        self._subscribe(link_statistics, "uplink_rssi_updated", "uplink_rssi")
        self._subscribe(link_statistics, "uplink_rate_updated", "uplink_rate_hz")
        self._subscribe(link_statistics, "downlink_rate_updated", "downlink_rate_hz")
        self._subscribe(link_statistics, "uplink_congestion_updated", "uplink_congestion")
        self._subscribe(link_statistics, "downlink_congestion_updated", "downlink_congestion")

    @property
    def available(self) -> bool:
        return self._available and not self._closed

    def _subscribe(self, link_statistics: Any, caller_name: str, field_name: str) -> None:
        caller = getattr(link_statistics, caller_name, None)
        add_callback = getattr(caller, "add_callback", None)
        if not callable(add_callback):
            return

        def _callback(value: object) -> None:
            normalized = _normalize_float(value)
            with self._lock:
                self._values[field_name] = normalized
                self._last_update_s = self._monotonic()

        try:
            add_callback(_callback)
        except Exception:
            return
        self._subscriptions.append((caller, _callback))

    def observation(self) -> HoverLinkHealthObservation:
        with self._lock:
            snapshot = dict(self._values)
            last_update_s = self._last_update_s
        age_s = None if last_update_s is None else max(0.0, self._monotonic() - last_update_s)
        return HoverLinkHealthObservation(
            age_s=age_s,
            latency_ms=snapshot["latency_ms"],
            link_quality=snapshot["link_quality"],
            uplink_rssi=snapshot["uplink_rssi"],
            uplink_rate_hz=snapshot["uplink_rate_hz"],
            downlink_rate_hz=snapshot["downlink_rate_hz"],
            uplink_congestion=snapshot["uplink_congestion"],
            downlink_congestion=snapshot["downlink_congestion"],
        )

    def close(self) -> None:
        if self._closed:
            return
        for caller, callback in self._subscriptions:
            remove_callback = getattr(caller, "remove_callback", None)
            if callable(remove_callback):
                try:
                    remove_callback(callback)
                except Exception:
                    pass
        self._subscriptions.clear()
        self._closed = True


class StatefulHoverPrimitive:
    """Run one explicit hover-setpoint primitive with bounded abort handling."""

    def __init__(
        self,
        sync_cf: Any,
        *,
        ground_distance_provider: Callable[[], HoverGroundDistanceObservation] | None = None,
        stability_provider: Callable[[], HoverStabilityObservation] | None = None,
        link_health_provider: Callable[[], HoverLinkHealthObservation] | None = None,
        trace_writer: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cf = sync_cf.cf if hasattr(sync_cf, "cf") else sync_cf
        self._ground_distance_provider = ground_distance_provider
        self._stability_provider = stability_provider
        self._external_link_health_provider = link_health_provider
        self._trace_writer = trace_writer
        self._sleep = sleep
        self._monotonic = monotonic
        self._setpoint_period_s = HOVER_SETPOINT_PERIOD_S
        self._current_height_m = 0.0
        self._commanded_max_height_m = 0.0
        self._setpoint_count = 0
        self._took_off = False
        self._landed = False
        self._final_phase = "idle"
        self._abort_requested_reason: str | None = None
        self._forced_motor_cutoff = False
        self._touchdown_confirmation_source: str | None = None
        self._touchdown_distance_m: float | None = None
        self._touchdown_supervisor_grounded = False
        self._stable_hover_established = False
        self._trim_identified = False
        self._qualified_hover_reached = False
        self._landing_trim_identified = False
        self._hover_mode_commands_sent = False
        self._trim_state = TrimObserverState()
        self._link_unhealthy_since_s: float | None = None
        self._last_link_failures: tuple[str, ...] | None = None

        self._owned_link_health_monitor: _CflibLinkHealthMonitor | None = None
        if self._external_link_health_provider is None:
            monitor = _CflibLinkHealthMonitor(self._cf, monotonic=self._monotonic)
            if monitor.available:
                self._owned_link_health_monitor = monitor

    @property
    def landed(self) -> bool:
        """Return whether the primitive reached a stop-setpoint landing path."""

        return self._landed

    @property
    def took_off(self) -> bool:
        """Return whether the primitive ever commanded a nonzero height."""

        return self._took_off

    def request_abort(self, reason: str) -> None:
        """Ask the currently running primitive to abort into its landing path."""

        self._abort_requested_reason = str(reason).strip() or "abort requested"

    def close(self) -> None:
        """Release any optional cflib callback subscriptions created by this helper."""

        if self._owned_link_health_monitor is not None:
            self._owned_link_health_monitor.close()
            self._owned_link_health_monitor = None

    def _reset_run_state(self) -> None:
        """Reset mutable execution state while preserving externally requested aborts."""

        self._current_height_m = 0.0
        self._commanded_max_height_m = 0.0
        self._setpoint_count = 0
        self._took_off = False
        self._landed = False
        self._final_phase = "idle"
        self._forced_motor_cutoff = False
        self._touchdown_confirmation_source = None
        self._touchdown_distance_m = None
        self._touchdown_supervisor_grounded = False
        self._stable_hover_established = False
        self._trim_identified = False
        self._qualified_hover_reached = False
        self._landing_trim_identified = False
        self._hover_mode_commands_sent = False
        self._trim_state = TrimObserverState()
        self._link_unhealthy_since_s = None
        self._last_link_failures = None

    def run(
        self,
        config: HoverPrimitiveConfig,
        *,
        after_takeoff: Callable[[], None] | None = None,
    ) -> HoverPrimitiveOutcome:
        """Execute the bounded takeoff-hold-land primitive.

        Args:
            config: Desired takeoff, hold, and landing limits.
            after_takeoff: Optional callback invoked after takeoff confirmation
                and before the hover hold starts. Use this to synchronize
                external safety/runtime state with the now-confirmed lift-off.

        Returns:
            The final primitive outcome, including whether an explicit abort was
            requested.

        Raises:
            KeyboardInterrupt: Re-raised after the primitive lands.
            Exception: Re-raised after the primitive lands if command sending
                fails mid-flight.
        """

        pending_abort_reason = self._abort_requested_reason
        self._reset_run_state()
        self._abort_requested_reason = pending_abort_reason
        self._setpoint_period_s = max(0.02, float(config.setpoint_period_s))
        link_health_config = self._enabled_link_health_config(config.link_health)
        stability_config = config.stability

        try:
            self.takeoff(
                target_height_m=config.target_height_m,
                velocity_mps=config.takeoff_velocity_mps,
                takeoff_confirm_min_height_m=config.takeoff_confirm_min_height_m,
                takeoff_confirm_samples=config.takeoff_confirm_samples,
                takeoff_confirm_timeout_s=config.takeoff_confirm_timeout_s,
                takeoff_confirm_max_age_s=config.takeoff_confirm_max_age_s,
                takeoff_confirm_target_height_tolerance_m=config.takeoff_confirm_target_height_tolerance_m,
                micro_liftoff_height_m=config.micro_liftoff_height_m,
                vertical_bootstrap_config=config.vertical_bootstrap,
                stability_config=stability_config,
                link_health_config=link_health_config,
            )
            if after_takeoff is not None:
                after_takeoff()
            self.stabilize_hover(
                target_height_m=config.target_height_m,
                stability_config=stability_config,
                link_health_config=link_health_config,
            )
            self.hold_for(
                config.hover_duration_s,
                link_health_config=link_health_config,
                stability_config=stability_config,
            )
        except HoverPrimitiveAbort:
            self._final_phase = "abort_landing"
            abort_reason = self._abort_requested_reason
            _emit_trace(
                self._trace_writer,
                "hover_primitive_abort",
                status="begin",
                message=abort_reason,
            )
            if not self._hover_mode_commands_sent and not self._took_off:
                _emit_trace(
                    self._trace_writer,
                    "hover_primitive_abort_manual_liftoff",
                    status="begin",
                    message=abort_reason,
                )
                self._cut_motors(
                    self._setpoint_period_s,
                    notify_stop_flush_s=config.notify_stop_flush_s,
                )
                _emit_trace(
                    self._trace_writer,
                    "hover_primitive_abort_manual_liftoff",
                    status="done",
                    message=abort_reason,
                )
            else:
                self._land(config)
            _emit_trace(
                self._trace_writer,
                "hover_primitive_abort",
                status="done",
                message=abort_reason,
            )
            outcome = self.current_outcome(
                final_phase="abort_landing",
                aborted=True,
                abort_reason=abort_reason,
            )
            self._abort_requested_reason = None
            return outcome
        except KeyboardInterrupt:
            self._final_phase = "interrupt_landing"
            self._land(config)
            self._abort_requested_reason = None
            raise
        except Exception:
            self._final_phase = "error_landing"
            self._land(config)
            self._abort_requested_reason = None
            raise

        self.land(config)
        outcome = self.current_outcome(
            final_phase="landed",
            aborted=False,
            abort_reason=None,
        )
        self._abort_requested_reason = None
        return outcome

    def current_outcome(
        self,
        *,
        final_phase: str | None = None,
        aborted: bool,
        abort_reason: str | None,
    ) -> HoverPrimitiveOutcome:
        """Return one normalized outcome snapshot for the current flight session."""

        return HoverPrimitiveOutcome(
            final_phase=final_phase or self._final_phase,
            took_off=self._took_off,
            landed=self._landed,
            aborted=aborted,
            abort_reason=abort_reason,
            commanded_max_height_m=self._commanded_max_height_m,
            setpoint_count=self._setpoint_count,
            forced_motor_cutoff=self._forced_motor_cutoff,
            touchdown_confirmation_source=self._touchdown_confirmation_source,
            touchdown_distance_m=self._touchdown_distance_m,
            touchdown_supervisor_grounded=self._touchdown_supervisor_grounded,
            stable_hover_established=self._stable_hover_established,
            trim_identified=self._trim_identified,
            qualified_hover_reached=self._qualified_hover_reached,
            landing_trim_identified=self._landing_trim_identified,
            abort_phase=self._final_phase if aborted else None,
        )

    @staticmethod
    def _enabled_link_health_config(
        config: HoverLinkHealthConfig | None,
    ) -> HoverLinkHealthConfig | None:
        """Return one runtime link-health gate only when it is explicitly enabled."""

        if config is None or not config.enabled:
            return None
        return config

    def takeoff(
        self,
        *,
        target_height_m: float,
        velocity_mps: float,
        takeoff_confirm_min_height_m: float = HOVER_TAKEOFF_CONFIRM_MIN_HEIGHT_M,
        takeoff_confirm_samples: int = HOVER_TAKEOFF_CONFIRM_SAMPLES,
        takeoff_confirm_timeout_s: float = HOVER_TAKEOFF_CONFIRM_TIMEOUT_S,
        takeoff_confirm_max_age_s: float = HOVER_TAKEOFF_CONFIRM_MAX_AGE_S,
        takeoff_confirm_target_height_tolerance_m: float = HOVER_TAKEOFF_TARGET_HEIGHT_TOLERANCE_M,
        micro_liftoff_height_m: float = HOVER_MICRO_LIFTOFF_HEIGHT_M,
        vertical_bootstrap_config: HoverVerticalBootstrapConfig | None = None,
        stability_config: HoverStabilityConfig | None = None,
        link_health_config: HoverLinkHealthConfig | None = None,
    ) -> None:
        """Climb to one bounded hover height while honoring abort and link gates."""

        if stability_config is None:
            raise HoverPrimitiveSafetyError(
                "hover takeoff requires a stability config for micro-liftoff trim identification"
            )
        if self._stability_provider is None:
            raise HoverPrimitiveSafetyError(
                "hover takeoff requires a stability provider for micro-liftoff trim identification"
            )

        self._final_phase = "takeoff"
        _emit_trace(
            self._trace_writer,
            "hover_primitive_takeoff",
            status="begin",
            data={
                "target_height_m": target_height_m,
                "velocity_mps": velocity_mps,
            },
        )
        micro_liftoff_height_m = min(
            max(0.0, float(micro_liftoff_height_m)),
            max(0.0, float(target_height_m)),
        )
        trim_identify_height_m = min(
            max(micro_liftoff_height_m, HOVER_ESTIMATOR_FLOW_GATE_MIN_HEIGHT_M),
            max(0.0, float(target_height_m)),
        )
        enabled_link_health = self._enabled_link_health_config(link_health_config)
        if vertical_bootstrap_config is None:
            self._micro_liftoff(
                target_height_m=micro_liftoff_height_m,
                velocity_mps=velocity_mps,
                confirm_timeout_s=takeoff_confirm_timeout_s,
                max_age_s=takeoff_confirm_max_age_s,
                link_health_config=enabled_link_health,
            )
        else:
            self._vertical_bootstrap(
                target_height_m=micro_liftoff_height_m,
                config=vertical_bootstrap_config,
                link_health_config=enabled_link_health,
            )
        self._identify_trim(
            target_height_m=trim_identify_height_m,
            stability_config=stability_config,
            link_health_config=enabled_link_health,
        )
        self._ramp_height(
            target_height_m,
            velocity_mps=velocity_mps,
            check_abort=True,
            link_health_config=enabled_link_health,
        )
        self._wait_for_takeoff_confirmation(
            target_height_m=target_height_m,
            min_height_m=takeoff_confirm_min_height_m,
            required_samples=takeoff_confirm_samples,
            timeout_s=takeoff_confirm_timeout_s,
            max_age_s=takeoff_confirm_max_age_s,
            target_height_tolerance_m=takeoff_confirm_target_height_tolerance_m,
            mark_took_off=True,
            link_health_config=enabled_link_health,
        )
        _emit_trace(self._trace_writer, "hover_primitive_takeoff", status="done")

    def stabilize_hover(
        self,
        *,
        target_height_m: float,
        stability_config: HoverStabilityConfig | None = None,
        link_health_config: HoverLinkHealthConfig | None = None,
    ) -> None:
        """Require one bounded stable hover window before the hold phase starts."""

        if stability_config is None:
            return
        self._final_phase = "stabilize"
        _emit_trace(
            self._trace_writer,
            "hover_primitive_stabilize",
            status="begin",
            data={
                "target_height_m": target_height_m,
                "settle_timeout_s": stability_config.settle_timeout_s,
                "required_stable_samples": stability_config.required_stable_samples,
            },
        )
        self._await_stable_hover(
            target_height_m=target_height_m,
            stability_config=stability_config,
            link_health_config=self._enabled_link_health_config(link_health_config),
        )
        self._current_height_m = max(0.0, float(target_height_m))
        self._stable_hover_established = True
        self._qualified_hover_reached = True
        _emit_trace(self._trace_writer, "hover_primitive_stabilize", status="done")

    def _micro_liftoff(
        self,
        *,
        target_height_m: float,
        velocity_mps: float,
        confirm_timeout_s: float,
        max_age_s: float,
        link_health_config: HoverLinkHealthConfig | None = None,
    ) -> None:
        """Lift a small bounded distance before trim identification begins."""

        _emit_trace(
            self._trace_writer,
            "hover_primitive_micro_liftoff",
            status="begin",
            data={"target_height_m": target_height_m, "velocity_mps": velocity_mps},
        )
        self._ramp_height(
            target_height_m,
            velocity_mps=velocity_mps,
            check_abort=True,
            link_health_config=link_health_config,
        )
        self._wait_for_takeoff_confirmation(
            target_height_m=target_height_m,
            min_height_m=target_height_m,
            required_samples=1,
            timeout_s=max(self._setpoint_period_s * 4.0, float(confirm_timeout_s)),
            max_age_s=max_age_s,
            target_height_tolerance_m=0.0,
            mark_took_off=False,
            trace_phase="hover_primitive_micro_liftoff_confirm",
            link_health_config=link_health_config,
        )
        _emit_trace(self._trace_writer, "hover_primitive_micro_liftoff", status="done")

    def _vertical_bootstrap(
        self,
        *,
        target_height_m: float,
        config: HoverVerticalBootstrapConfig,
        link_health_config: HoverLinkHealthConfig | None = None,
    ) -> None:
        """Use one bounded closed-loop vertical bootstrap before hover handoff."""

        bounded_target_height_m = max(0.0, float(target_height_m))
        baseline_observation = self._observe_ground_distance()
        baseline_distance_m: float | None = None
        if (
            baseline_observation.distance_m is not None
            and baseline_observation.age_s is not None
            and baseline_observation.age_s <= float(config.max_observation_age_s)
        ):
            baseline_distance_m = float(baseline_observation.distance_m)
        controller_state = initialize_vertical_bootstrap_state(
            baseline_distance_m=baseline_distance_m,
        )

        _emit_trace(
            self._trace_writer,
            "hover_primitive_vertical_bootstrap",
            status="begin",
            data={
                "target_height_m": bounded_target_height_m,
                "min_thrust_percentage": float(config.min_thrust_percentage),
                "feedforward_thrust_percentage": float(config.feedforward_thrust_percentage),
                "max_thrust_percentage": float(config.max_thrust_percentage),
                "reference_duration_s": float(config.reference_duration_s),
                "progress_to_ceiling_s": float(config.progress_to_ceiling_s),
                "max_duration_s": float(config.max_duration_s),
                "min_range_height_m": float(config.min_range_height_m),
                "max_range_height_m": float(config.max_range_height_m),
                "min_range_rise_m": float(config.min_range_rise_m),
                "required_liveness_samples": int(config.required_liveness_samples),
                "baseline_distance_m": baseline_distance_m,
                "require_motion_squal_liveness": config.require_motion_squal_liveness,
                "min_motion_squal": int(config.min_motion_squal),
            },
        )

        deadline_s = self._monotonic() + max(0.0, float(config.max_duration_s))
        next_deadline_s = self._monotonic()
        start_monotonic_s = next_deadline_s
        last_failure_codes: tuple[str, ...] | None = None
        tick_count = 0
        max_commanded_thrust_percentage = 0.0
        max_raw_commanded_thrust_percentage = 0.0
        max_distance_m = baseline_distance_m
        max_range_rise_m: float | None = None
        max_motion_squal: int | None = None
        max_progress_boost_percentage = 0.0
        first_range_live_elapsed_s: float | None = None
        first_flow_live_elapsed_s: float | None = None

        def _bootstrap_summary_data() -> dict[str, object]:
            return {
                "tick_count": tick_count,
                "max_commanded_thrust_percentage": max_commanded_thrust_percentage,
                "max_raw_commanded_thrust_percentage": max_raw_commanded_thrust_percentage,
                "max_distance_m": max_distance_m,
                "max_range_rise_m": max_range_rise_m,
                "max_motion_squal": max_motion_squal,
                "max_progress_boost_percentage": max_progress_boost_percentage,
                "first_range_live_elapsed_s": first_range_live_elapsed_s,
                "first_flow_live_elapsed_s": first_flow_live_elapsed_s,
            }

        while True:
            self._ensure_not_aborted()
            self._ensure_link_health(link_health_config)
            observation = self._observe_ground_distance()
            stability = self._observe_hover_stability()
            elapsed_s = max(0.0, self._monotonic() - start_monotonic_s)
            controller_state, controller_decision = step_vertical_bootstrap_controller(
                config=config,
                state=controller_state,
                observation=VerticalBootstrapObservation(
                    elapsed_s=elapsed_s + self._setpoint_period_s,
                    distance_m=observation.distance_m,
                    distance_age_s=observation.age_s,
                    vertical_speed_mps=stability.vz_mps,
                    vertical_speed_age_s=stability.vz_age_s,
                    motion_squal=stability.motion_squal,
                    motion_squal_age_s=stability.motion_squal_age_s,
                    roll_deg=stability.roll_deg,
                    pitch_deg=stability.pitch_deg,
                    attitude_age_s=stability.attitude_age_s,
                ),
            )
            tick_count += 1
            max_commanded_thrust_percentage = max(
                max_commanded_thrust_percentage,
                float(controller_decision.commanded_thrust_percentage),
            )
            max_raw_commanded_thrust_percentage = max(
                max_raw_commanded_thrust_percentage,
                float(controller_decision.raw_commanded_thrust_percentage),
            )
            if observation.distance_m is not None:
                max_distance_m = max(
                    float(max_distance_m)
                    if max_distance_m is not None
                    else float(observation.distance_m),
                    float(observation.distance_m),
                )
            if controller_decision.range_rise_m is not None:
                max_range_rise_m = max(
                    float(max_range_rise_m)
                    if max_range_rise_m is not None
                    else float(controller_decision.range_rise_m),
                    float(controller_decision.range_rise_m),
                )
            if stability.motion_squal is not None:
                max_motion_squal = max(
                    int(max_motion_squal) if max_motion_squal is not None else int(stability.motion_squal),
                    int(stability.motion_squal),
                )
            max_progress_boost_percentage = max(
                max_progress_boost_percentage,
                float(controller_decision.progress_boost_percentage),
            )
            if controller_decision.range_live and first_range_live_elapsed_s is None:
                first_range_live_elapsed_s = elapsed_s
            if controller_decision.flow_live and first_flow_live_elapsed_s is None:
                first_flow_live_elapsed_s = elapsed_s
            failure_codes_tuple = tuple(controller_decision.failure_codes)
            self._send_manual_liftoff_setpoint(
                thrust_percentage=controller_decision.commanded_thrust_percentage
            )
            _emit_trace(
                self._trace_writer,
                "hover_primitive_vertical_bootstrap_tick",
                status="sample",
                data={
                    "tick_index": tick_count,
                    "elapsed_s": elapsed_s,
                    "distance_m": observation.distance_m,
                    "distance_age_s": observation.age_s,
                    "baseline_distance_m": controller_decision.baseline_distance_m,
                    "current_height_m": controller_decision.current_height_m,
                    "target_liftoff_height_m": controller_decision.target_liftoff_height_m,
                    "range_rise_m": controller_decision.range_rise_m,
                    "range_live": controller_decision.range_live,
                    "range_height_ready": controller_decision.range_height_ready,
                    "range_rise_ready": controller_decision.range_rise_ready,
                    "flow_live": controller_decision.flow_live,
                    "motion_squal_fresh": controller_decision.motion_squal_fresh,
                    "motion_squal": stability.motion_squal,
                    "motion_squal_age_s": stability.motion_squal_age_s,
                    "vz_mps": stability.vz_mps,
                    "vz_age_s": stability.vz_age_s,
                    "roll_deg": stability.roll_deg,
                    "pitch_deg": stability.pitch_deg,
                    "reference_progress": controller_decision.reference_progress,
                    "reference_height_m": controller_decision.reference_height_m,
                    "reference_vertical_speed_mps": controller_decision.reference_vertical_speed_mps,
                    "height_error_m": controller_decision.height_error_m,
                    "vertical_speed_error_mps": controller_decision.vertical_speed_error_mps,
                    "feedforward_thrust_percentage": controller_decision.feedforward_thrust_percentage,
                    "progress_boost_percentage": controller_decision.progress_boost_percentage,
                    "height_term_percentage": controller_decision.height_term_percentage,
                    "vertical_speed_term_percentage": controller_decision.vertical_speed_term_percentage,
                    "raw_commanded_thrust_percentage": controller_decision.raw_commanded_thrust_percentage,
                    "commanded_thrust_percentage": controller_decision.commanded_thrust_percentage,
                    "thrust_headroom_percentage": controller_decision.thrust_headroom_percentage,
                    "at_thrust_ceiling": controller_decision.at_thrust_ceiling,
                    "progress_missing": controller_decision.progress_missing,
                    "ceiling_without_progress_s": controller_decision.ceiling_without_progress_s,
                    "handoff_ready": controller_decision.handoff_ready,
                    "failure_codes": failure_codes_tuple,
                },
            )
            next_deadline_s += self._setpoint_period_s
            self._sleep_until(next_deadline_s)

            if failure_codes_tuple and failure_codes_tuple != last_failure_codes:
                last_failure_codes = failure_codes_tuple
                _emit_trace(
                    self._trace_writer,
                    "hover_primitive_vertical_bootstrap",
                    status="degraded",
                    data={
                        "failure_codes": failure_codes_tuple,
                        "commanded_thrust_percentage": controller_decision.commanded_thrust_percentage,
                        "reference_height_m": controller_decision.reference_height_m,
                        "reference_vertical_speed_mps": controller_decision.reference_vertical_speed_mps,
                        "height_error_m": controller_decision.height_error_m,
                        "vertical_speed_error_mps": controller_decision.vertical_speed_error_mps,
                        "progress_boost_percentage": controller_decision.progress_boost_percentage,
                        "distance_m": observation.distance_m,
                        "age_s": observation.age_s,
                        "range_rise_m": controller_decision.range_rise_m,
                        "motion_squal": stability.motion_squal,
                        "motion_squal_age_s": stability.motion_squal_age_s,
                        "vz_mps": stability.vz_mps,
                        "vz_age_s": stability.vz_age_s,
                        "roll_deg": stability.roll_deg,
                        "pitch_deg": stability.pitch_deg,
                        "ceiling_without_progress_s": controller_decision.ceiling_without_progress_s,
                        "raw_commanded_thrust_percentage": controller_decision.raw_commanded_thrust_percentage,
                        "thrust_headroom_percentage": controller_decision.thrust_headroom_percentage,
                        "at_thrust_ceiling": controller_decision.at_thrust_ceiling,
                        "progress_missing": controller_decision.progress_missing,
                        **_bootstrap_summary_data(),
                    },
                )

            if controller_decision.abort_reason is not None:
                _emit_trace(
                    self._trace_writer,
                    "hover_primitive_vertical_bootstrap",
                    status="blocked",
                    message=controller_decision.abort_reason,
                    data={
                        "failure_codes": failure_codes_tuple or ("vertical_bootstrap_abort",),
                        "commanded_thrust_percentage": controller_decision.commanded_thrust_percentage,
                        "reference_height_m": controller_decision.reference_height_m,
                        "distance_m": observation.distance_m,
                        "age_s": observation.age_s,
                        "range_rise_m": controller_decision.range_rise_m,
                        "progress_boost_percentage": controller_decision.progress_boost_percentage,
                        "motion_squal": stability.motion_squal,
                        "motion_squal_age_s": stability.motion_squal_age_s,
                        "vz_mps": stability.vz_mps,
                        "vz_age_s": stability.vz_age_s,
                        "ceiling_without_progress_s": controller_decision.ceiling_without_progress_s,
                        "raw_commanded_thrust_percentage": controller_decision.raw_commanded_thrust_percentage,
                        "thrust_headroom_percentage": controller_decision.thrust_headroom_percentage,
                        "at_thrust_ceiling": controller_decision.at_thrust_ceiling,
                        "progress_missing": controller_decision.progress_missing,
                        **_bootstrap_summary_data(),
                    },
                )
                self.request_abort(controller_decision.abort_reason)
                raise HoverPrimitiveAbort(controller_decision.abort_reason)

            if controller_decision.handoff_ready:
                observed_height_m = max(
                    bounded_target_height_m,
                    float(observation.distance_m)
                    if observation.distance_m is not None
                    else bounded_target_height_m,
                )
                self._current_height_m = observed_height_m
                self._commanded_max_height_m = max(
                    self._commanded_max_height_m,
                    observed_height_m,
                )
                _emit_trace(
                    self._trace_writer,
                    "hover_primitive_vertical_bootstrap",
                    status="done",
                    data={
                        "commanded_thrust_percentage": controller_decision.commanded_thrust_percentage,
                        "reference_height_m": controller_decision.reference_height_m,
                        "distance_m": observation.distance_m,
                        "range_rise_m": controller_decision.range_rise_m,
                        "progress_boost_percentage": controller_decision.progress_boost_percentage,
                        "motion_squal": stability.motion_squal,
                        "vz_mps": stability.vz_mps,
                        "observed_height_m": observed_height_m,
                        "raw_commanded_thrust_percentage": controller_decision.raw_commanded_thrust_percentage,
                        "thrust_headroom_percentage": controller_decision.thrust_headroom_percentage,
                        **_bootstrap_summary_data(),
                    },
                )
                return

            if self._monotonic() > deadline_s:
                reason = (
                    "vertical bootstrap exceeded its bounded control window without "
                    "proving hover handoff"
                )
                _emit_trace(
                    self._trace_writer,
                    "hover_primitive_vertical_bootstrap",
                    status="blocked",
                    message=reason,
                    data={
                        "failure_codes": failure_codes_tuple or ("vertical_bootstrap_deadline",),
                        "commanded_thrust_percentage": controller_decision.commanded_thrust_percentage,
                        "distance_m": observation.distance_m,
                        "age_s": observation.age_s,
                        "range_rise_m": controller_decision.range_rise_m,
                        "progress_boost_percentage": controller_decision.progress_boost_percentage,
                        "motion_squal": stability.motion_squal,
                        "raw_commanded_thrust_percentage": controller_decision.raw_commanded_thrust_percentage,
                        "thrust_headroom_percentage": controller_decision.thrust_headroom_percentage,
                        "at_thrust_ceiling": controller_decision.at_thrust_ceiling,
                        "progress_missing": controller_decision.progress_missing,
                        **_bootstrap_summary_data(),
                    },
                )
                self.request_abort(reason)
                raise HoverPrimitiveAbort(reason)

    def _identify_trim(
        self,
        *,
        target_height_m: float,
        stability_config: HoverStabilityConfig,
        link_health_config: HoverLinkHealthConfig | None = None,
    ) -> None:
        """Identify the neutral hover trim before climbing into nominal hover."""

        _emit_trace(
            self._trace_writer,
            "hover_primitive_trim_identify",
            status="begin",
            data={
                "target_height_m": target_height_m,
                "settle_timeout_s": stability_config.settle_timeout_s,
                "required_converged_samples": stability_config.trim_observer.required_converged_samples,
            },
        )
        if self._current_height_m < float(target_height_m):
            self._ramp_height(
                target_height_m,
                velocity_mps=max(0.05, min(0.10, float(target_height_m) / max(self._setpoint_period_s, 0.05))),
                check_abort=True,
                link_health_config=link_health_config,
            )
        self._await_trim_convergence(
            target_height_m=target_height_m,
            stability_config=stability_config,
            link_health_config=link_health_config,
        )
        self._current_height_m = max(0.0, float(target_height_m))
        self._trim_identified = True
        _emit_trace(self._trace_writer, "hover_primitive_trim_identify", status="done")

    def hold_for(
        self,
        duration_s: float,
        *,
        link_health_config: HoverLinkHealthConfig | None = None,
        stability_config: HoverStabilityConfig | None = None,
    ) -> None:
        """Keep the current hover anchor stable for a bounded duration."""

        self._final_phase = "hold"
        _emit_trace(
            self._trace_writer,
            "hover_primitive_hold",
            status="begin",
            data={"hover_duration_s": duration_s, "height_m": self._current_height_m},
        )
        self._hold(
            duration_s,
            check_abort=True,
            link_health_config=self._enabled_link_health_config(link_health_config),
            stability_config=stability_config,
            target_height_m=self._current_height_m,
        )
        _emit_trace(self._trace_writer, "hover_primitive_hold", status="done")

    def translate_body(self, config: HoverTranslationConfig) -> None:
        """Translate in body coordinates while maintaining one bounded hover height.

        Positive ``forward_m`` moves the aircraft forward. Positive ``left_m``
        moves it left in the Crazyflie body frame, matching the hover-setpoint
        velocity contract exposed by cflib.
        """

        target_height_m = (
            self._current_height_m if config.target_height_m is None else max(0.0, float(config.target_height_m))
        )
        forward_m = float(config.forward_m)
        left_m = float(config.left_m)
        horizontal_distance_m = math.hypot(forward_m, left_m)
        enabled_link_health = self._enabled_link_health_config(config.link_health)

        self._final_phase = "translate"
        _emit_trace(
            self._trace_writer,
            "hover_primitive_translate",
            status="begin",
            data={
                "forward_m": forward_m,
                "left_m": left_m,
                "target_height_m": target_height_m,
                "velocity_mps": config.velocity_mps,
            },
        )

        if horizontal_distance_m > 0.0:
            velocity_mps = max(0.01, float(config.velocity_mps))
            duration_s = horizontal_distance_m / velocity_mps
            commanded_forward_mps = forward_m / duration_s
            commanded_left_mps = left_m / duration_s
            steps = max(1, int(math.ceil(duration_s / self._setpoint_period_s)))
            next_deadline_s = self._monotonic()
            for _step_index in range(steps):
                self._ensure_not_aborted()
                self._ensure_link_health(enabled_link_health)
                self._send_motion_setpoint(
                    forward_mps=commanded_forward_mps,
                    left_mps=commanded_left_mps,
                    yawrate_dps=0.0,
                    height_m=target_height_m,
                )
                next_deadline_s += self._setpoint_period_s
                self._sleep_until(next_deadline_s)

        self._send_hover_setpoint(target_height_m)
        _emit_trace(self._trace_writer, "hover_primitive_translate", status="done")

        if config.settle_duration_s > 0.0:
            self.hold_for(
                float(config.settle_duration_s),
                link_health_config=enabled_link_health,
                stability_config=config.stability,
            )

    def land(self, config: HoverPrimitiveConfig) -> None:
        """Run the staged landing path once for the current flight session."""

        self._final_phase = "landing"
        _emit_trace(
            self._trace_writer,
            "hover_primitive_land",
            status="begin",
            data={
                "velocity_mps": config.land_velocity_mps,
                "floor_height_m": config.landing_floor_height_m,
                "touchdown_height_m": config.touchdown_height_m,
            },
        )
        self._land(config)
        _emit_trace(self._trace_writer, "hover_primitive_land", status="done")

    def _ensure_not_aborted(self) -> None:
        if self._abort_requested_reason is not None:
            raise HoverPrimitiveAbort(self._abort_requested_reason)

    def _send_motion_setpoint(
        self,
        *,
        forward_mps: float,
        left_mps: float,
        yawrate_dps: float,
        height_m: float,
    ) -> None:
        """Send one explicit hover-setpoint command and update local session state."""

        bounded_height = max(0.0, float(height_m))
        self._cf.commander.send_hover_setpoint(
            float(forward_mps),
            float(left_mps),
            float(yawrate_dps),
            bounded_height,
        )
        self._hover_mode_commands_sent = True
        self._current_height_m = bounded_height
        self._commanded_max_height_m = max(self._commanded_max_height_m, bounded_height)
        self._setpoint_count += 1

    def _send_manual_liftoff_setpoint(self, *, thrust_percentage: float) -> None:
        """Send one bounded raw-thrust micro-liftoff command on the manual channel."""

        send_setpoint_manual = getattr(self._cf.commander, "send_setpoint_manual", None)
        if not callable(send_setpoint_manual):
            raise HoverPrimitiveSafetyError(
                "manual micro-liftoff requires cflib commander.send_setpoint_manual"
            )
        send_setpoint_manual(
            0.0,
            0.0,
            0.0,
            float(thrust_percentage),
            False,
        )
        self._setpoint_count += 1

    def _send_hover_setpoint(self, height_m: float) -> None:
        self._send_motion_setpoint(
            forward_mps=0.0,
            left_mps=0.0,
            yawrate_dps=0.0,
            height_m=height_m,
        )

    def _send_anchor_hold_setpoint(
        self,
        command: FlowAnchorControlCommand,
        *,
        target_height_m: float,
    ) -> None:
        """Send one bounded hover-setpoint derived from the flow-anchor outer-loop."""

        commanded_height_m = (
            float(target_height_m) if command.height_m is None else float(command.height_m)
        )
        self._send_motion_setpoint(
            forward_mps=float(command.forward_mps),
            left_mps=float(command.left_mps),
            yawrate_dps=0.0,
            height_m=commanded_height_m,
        )

    def _sleep_until(self, deadline_s: float) -> None:
        remaining_s = float(deadline_s) - self._monotonic()
        if remaining_s > 0.0:
            self._sleep(remaining_s)

    def _observe_ground_distance(self) -> HoverGroundDistanceObservation:
        """Return the latest downward-range observation for touchdown gating."""

        if self._ground_distance_provider is None:
            return HoverGroundDistanceObservation(distance_m=None, age_s=None)
        try:
            observation = self._ground_distance_provider()
        except Exception as exc:
            _emit_trace(
                self._trace_writer,
                "hover_primitive_touchdown_provider",
                status="degraded",
                message=f"ground-distance provider raised {exc!r}; continuing with no reading",
            )
            return HoverGroundDistanceObservation(distance_m=None, age_s=None)
        return _coerce_ground_distance_observation(observation)

    def _observe_hover_stability(self) -> HoverStabilityObservation:
        """Return the latest hover-stability observation from the runtime telemetry lane."""

        if self._stability_provider is None:
            return HoverStabilityObservation(height_m=None, height_age_s=None)
        return _coerce_hover_stability_observation(self._stability_provider())

    def _observe_link_health(self) -> HoverLinkHealthObservation:
        """Return one latest runtime communication-health observation."""

        if self._external_link_health_provider is not None:
            try:
                return _coerce_link_health_observation(self._external_link_health_provider())
            except Exception as exc:
                _emit_trace(
                    self._trace_writer,
                    "hover_primitive_link_health",
                    status="degraded",
                    message=f"link-health provider raised {exc!r}; continuing with no reading",
                )
                return HoverLinkHealthObservation(age_s=None)

        if self._owned_link_health_monitor is not None:
            try:
                return self._owned_link_health_monitor.observation()
            except Exception as exc:
                _emit_trace(
                    self._trace_writer,
                    "hover_primitive_link_health",
                    status="degraded",
                    message=f"built-in cflib link-health monitor failed: {exc}",
                )
                return HoverLinkHealthObservation(age_s=None)

        return HoverLinkHealthObservation(age_s=None)

    def _ensure_link_health(self, link_health_config: HoverLinkHealthConfig | None) -> None:
        """Abort early when the link stays unhealthy for longer than the grace period."""

        if link_health_config is None or not link_health_config.enabled:
            return
        if self._external_link_health_provider is None and self._owned_link_health_monitor is None:
            return

        observation = self._observe_link_health()
        failures = tuple(_link_health_failures(observation, config=link_health_config))
        if not failures:
            if self._link_unhealthy_since_s is not None:
                _emit_trace(
                    self._trace_writer,
                    "hover_primitive_link_health",
                    status="done",
                    data={"message": "link health recovered"},
                )
            self._link_unhealthy_since_s = None
            self._last_link_failures = None
            return

        now_s = self._monotonic()
        if self._link_unhealthy_since_s is None:
            self._link_unhealthy_since_s = now_s
            self._last_link_failures = failures
            _emit_trace(
                self._trace_writer,
                "hover_primitive_link_health",
                status="degraded",
                data={"failures": failures},
            )
            return

        if failures != self._last_link_failures:
            self._last_link_failures = failures
            _emit_trace(
                self._trace_writer,
                "hover_primitive_link_health",
                status="degraded",
                data={"failures": failures},
            )

        if (now_s - self._link_unhealthy_since_s) >= float(link_health_config.max_consecutive_unhealthy_s):
            reason = "runtime link health degraded: " + "; ".join(failures)
            self.request_abort(reason)
            raise HoverPrimitiveAbort(reason)

    def _ramp_height(
        self,
        target_height_m: float,
        *,
        velocity_mps: float,
        check_abort: bool = True,
        link_health_config: HoverLinkHealthConfig | None = None,
    ) -> None:
        start_height_m = float(self._current_height_m)
        target_height_m = max(0.0, float(target_height_m))
        velocity_mps = max(0.01, float(velocity_mps))
        period_s = self._setpoint_period_s
        duration_s = abs(target_height_m - start_height_m) / velocity_mps
        steps = max(1, int(math.ceil(duration_s / period_s)))
        next_deadline_s = self._monotonic()
        for step_index in range(steps):
            if check_abort:
                self._ensure_not_aborted()
                self._ensure_link_health(link_health_config)
            progress = (step_index + 1) / steps
            next_height = start_height_m + ((target_height_m - start_height_m) * progress)
            self._send_hover_setpoint(next_height)
            next_deadline_s += period_s
            self._sleep_until(next_deadline_s)

    def _wait_for_takeoff_confirmation(
        self,
        *,
        target_height_m: float,
        min_height_m: float,
        required_samples: int,
        timeout_s: float,
        max_age_s: float,
        target_height_tolerance_m: float,
        mark_took_off: bool,
        trace_phase: str = "hover_primitive_takeoff_confirm",
        link_health_config: HoverLinkHealthConfig | None = None,
    ) -> None:
        """Require fresh runtime telemetry before reporting a successful liftoff."""

        bounded_target_height_m = max(0.0, float(target_height_m))
        bounded_tolerance_m = max(0.0, float(target_height_tolerance_m))
        threshold_m = compute_takeoff_active_height_m(
            bounded_target_height_m,
            min_height_m=min_height_m,
            target_height_tolerance_m=bounded_tolerance_m,
        )
        required = max(1, int(required_samples))
        deadline_s = self._monotonic() + max(0.0, float(timeout_s))
        consecutive_confirmed_samples = 0

        _emit_trace(
            self._trace_writer,
            trace_phase,
            status="begin",
            data={
                "target_height_m": target_height_m,
                "threshold_m": threshold_m,
                "target_height_tolerance_m": bounded_tolerance_m,
                "required_samples": required,
                "timeout_s": timeout_s,
                "max_age_s": max_age_s,
            },
        )

        while True:
            self._ensure_not_aborted()
            self._ensure_link_health(link_health_config)
            observation = self._observe_ground_distance()
            range_confirmed = (
                observation.distance_m is not None
                and observation.age_s is not None
                and observation.age_s <= max_age_s
                and observation.distance_m >= threshold_m
            )
            supervisor_blocks_confirmation = (
                observation.is_flying is False
                and observation.supervisor_age_s is not None
                and observation.supervisor_age_s <= max_age_s
            )
            if range_confirmed and not supervisor_blocks_confirmation:
                consecutive_confirmed_samples += 1
            else:
                consecutive_confirmed_samples = 0

            self._send_hover_setpoint(target_height_m)

            if consecutive_confirmed_samples >= required:
                if mark_took_off:
                    self._took_off = True
                _emit_trace(
                    self._trace_writer,
                    trace_phase,
                    status="done",
                    data={
                        "distance_m": observation.distance_m,
                        "age_s": observation.age_s,
                        "confirmed_samples": consecutive_confirmed_samples,
                        "is_flying": observation.is_flying,
                        "supervisor_age_s": observation.supervisor_age_s,
                    },
                )
                return

            if self._monotonic() > deadline_s:
                reason = (
                    "takeoff confirmation failed: downward range never reached the "
                    f"required hover band ({threshold_m:.2f} m for target {bounded_target_height_m:.2f} m) "
                    f"within {float(timeout_s):.2f} s"
                )
                _emit_trace(
                    self._trace_writer,
                    trace_phase,
                    status="blocked",
                    data={
                        "distance_m": observation.distance_m,
                        "age_s": observation.age_s,
                        "confirmed_samples": consecutive_confirmed_samples,
                        "is_flying": observation.is_flying,
                        "supervisor_age_s": observation.supervisor_age_s,
                    },
                    message=reason,
                )
                self.request_abort(reason)
                raise HoverPrimitiveAbort(reason)

            self._sleep(self._setpoint_period_s)

    @staticmethod
    def _trim_observer_state_from_observation(
        observation: HoverStabilityObservation,
        *,
        target_height_m: float,
        config: HoverStabilityConfig,
        current_state: TrimObserverState,
    ) -> TrimObserverState:
        """Project one hover observation into the bounded trim observer."""

        trusted_state = compute_trusted_hover_state(
            raw_height_m=observation.height_m,
            raw_height_age_s=observation.height_age_s,
            estimate_z_m=observation.z_estimate_m,
            estimate_z_age_s=observation.z_estimate_age_s,
            x_m=observation.x_m,
            y_m=observation.y_m,
            pose_age_s=observation.pose_age_s,
            vx_mps=observation.vx_mps,
            vy_mps=observation.vy_mps,
            velocity_age_s=observation.velocity_age_s,
            motion_squal=observation.motion_squal,
            motion_squal_age_s=observation.motion_squal_age_s,
            is_flying=observation.is_flying,
            supervisor_age_s=observation.supervisor_age_s,
            height_config=config.height_trust,
            lateral_config=config.lateral_trust,
        )
        return update_trim_observer(
            current_state,
            trusted_state=trusted_state,
            target_height_m=float(target_height_m),
            vx_mps=observation.vx_mps,
            vy_mps=observation.vy_mps,
            velocity_age_s=observation.velocity_age_s,
            roll_deg=observation.roll_deg,
            pitch_deg=observation.pitch_deg,
            attitude_age_s=observation.attitude_age_s,
            config=config.trim_observer,
        )

    @staticmethod
    def _apply_trim_to_control_command(
        command: FlowAnchorControlCommand,
        *,
        trim_state: TrimObserverState,
        target_height_m: float,
        control_config: FlowAnchorControlConfig,
    ) -> FlowAnchorControlCommand:
        """Blend the bounded online trim estimate into the hover command."""

        forward_mps = float(command.forward_mps) + float(trim_state.forward_trim_mps)
        left_mps = float(command.left_mps) + float(trim_state.left_trim_mps)
        horizontal_speed = math.hypot(forward_mps, left_mps)
        max_speed = float(control_config.max_correction_velocity_mps)
        if horizontal_speed > max_speed and horizontal_speed > 0.0:
            scale = max_speed / horizontal_speed
            forward_mps *= scale
            left_mps *= scale

        commanded_height_m = (
            float(target_height_m) if command.height_m is None else float(command.height_m)
        ) + float(trim_state.height_trim_m)
        if control_config.max_commanded_height_m is not None:
            commanded_height_m = min(commanded_height_m, float(control_config.max_commanded_height_m))
        commanded_height_m = max(0.0, commanded_height_m)
        failures = tuple(dict.fromkeys((*command.failures, *trim_state.failures)))
        return FlowAnchorControlCommand(
            forward_mps=forward_mps,
            left_mps=left_mps,
            height_m=commanded_height_m,
            trusted_height_m=command.trusted_height_m,
            trusted_height_source=command.trusted_height_source,
            sensor_disagreement_m=command.sensor_disagreement_m,
            failures=failures,
        )

    def _await_trim_convergence(
        self,
        *,
        target_height_m: float,
        stability_config: HoverStabilityConfig,
        link_health_config: HoverLinkHealthConfig | None = None,
    ) -> None:
        """Wait for the online trim observer to converge before nominal climb."""

        deadline_s = self._monotonic() + max(0.0, float(stability_config.settle_timeout_s))
        drift_anchor_xy: tuple[float, float] | None = None
        last_failure_codes: tuple[str, ...] | None = None
        last_failure_messages: tuple[str, ...] | None = None

        while True:
            self._ensure_not_aborted()
            self._ensure_link_health(link_health_config)
            observation = self._observe_hover_stability()
            if (
                drift_anchor_xy is None
                and observation.x_m is not None
                and observation.y_m is not None
                and observation.pose_age_s is not None
                and observation.pose_age_s <= float(stability_config.max_observation_age_s)
            ):
                drift_anchor_xy = (observation.x_m, observation.y_m)
            control_command = _compute_hover_control_command(
                observation,
                target_height_m=float(target_height_m),
                anchor_xy=drift_anchor_xy,
                config=stability_config,
            )
            trim_state = self._trim_observer_state_from_observation(
                observation,
                target_height_m=float(target_height_m),
                config=stability_config,
                current_state=self._trim_state,
            )
            self._trim_state = trim_state
            adjusted_command = self._apply_trim_to_control_command(
                control_command,
                trim_state=trim_state,
                target_height_m=float(target_height_m),
                control_config=stability_config.anchor_control,
            )
            if trim_state.converged:
                self._send_anchor_hold_setpoint(
                    adjusted_command,
                    target_height_m=float(target_height_m),
                )
                return

            failure_messages = tuple(
                dict.fromkeys(
                    (*adjusted_command.failures, *trim_state.failures, "trim observer has not yet converged")
                )
            )
            failure_codes = tuple(
                dict.fromkeys(
                    code
                    for code, _message in _hover_stability_violations(
                        observation,
                        target_height_m=float(target_height_m),
                        anchor_xy=drift_anchor_xy,
                        config=stability_config,
                        control_command=adjusted_command,
                    )
                )
            )
            if "trim_not_converged" not in failure_codes:
                failure_codes = (*failure_codes, "trim_not_converged")
            if (
                failure_codes != last_failure_codes
                or failure_messages != last_failure_messages
            ):
                last_failure_codes = failure_codes
                last_failure_messages = failure_messages
                _emit_trace(
                    self._trace_writer,
                    "hover_primitive_trim_guard",
                    status="degraded",
                    data={
                        "phase": "trim_identify",
                        "failure_codes": failure_codes,
                        "failures": failure_messages,
                        "target_height_m": target_height_m,
                        "observed_height_m": observation.height_m,
                        "trusted_height_m": adjusted_command.trusted_height_m,
                        "trusted_height_source": adjusted_command.trusted_height_source,
                        "sensor_disagreement_m": adjusted_command.sensor_disagreement_m,
                        "x_m": observation.x_m,
                        "y_m": observation.y_m,
                        "vx_mps": observation.vx_mps,
                        "vy_mps": observation.vy_mps,
                        "roll_deg": observation.roll_deg,
                        "pitch_deg": observation.pitch_deg,
                        "motion_squal": observation.motion_squal,
                        "forward_trim_mps": trim_state.forward_trim_mps,
                        "left_trim_mps": trim_state.left_trim_mps,
                        "height_trim_m": trim_state.height_trim_m,
                    },
                )

            self._send_anchor_hold_setpoint(
                adjusted_command,
                target_height_m=float(target_height_m),
            )
            if self._monotonic() > deadline_s:
                reason = "hover trim identification failed: " + "; ".join(
                    last_failure_messages or ("trim observer did not converge before timeout",)
                )
                _emit_trace(
                    self._trace_writer,
                    "hover_primitive_trim_guard",
                    status="blocked",
                    message=reason,
                    data={
                        "phase": "trim_identify",
                        "failure_codes": last_failure_codes or ("trim_not_converged",),
                        "failures": last_failure_messages or ("trim observer did not converge before timeout",),
                        "target_height_m": target_height_m,
                    },
                )
                self.request_abort(reason)
                raise HoverPrimitiveAbort(reason)

            self._sleep(self._setpoint_period_s)

    def _hold(
        self,
        duration_s: float,
        *,
        check_abort: bool = True,
        link_health_config: HoverLinkHealthConfig | None = None,
        stability_config: HoverStabilityConfig | None = None,
        target_height_m: float | None = None,
    ) -> None:
        if duration_s <= 0:
            self._send_hover_setpoint(self._current_height_m)
            return
        deadline_s = self._monotonic() + float(duration_s)
        period_s = self._setpoint_period_s
        next_deadline_s = self._monotonic()
        drift_anchor_xy: tuple[float, float] | None = None
        consecutive_unstable_samples = 0
        while self._monotonic() < deadline_s:
            target_height_for_guard = (
                float(self._current_height_m) if target_height_m is None else float(target_height_m)
            )
            control_command = FlowAnchorControlCommand(
                forward_mps=0.0,
                left_mps=0.0,
                height_m=target_height_for_guard,
                trusted_height_m=None,
                trusted_height_source="none",
                sensor_disagreement_m=None,
                failures=(),
            )
            if check_abort:
                self._ensure_not_aborted()
                self._ensure_link_health(link_health_config)
                if stability_config is not None:
                    observation = self._observe_hover_stability()
                    if (
                        drift_anchor_xy is None
                        and observation.x_m is not None
                        and observation.y_m is not None
                        and observation.pose_age_s is not None
                        and observation.pose_age_s <= float(stability_config.max_observation_age_s)
                    ):
                        drift_anchor_xy = (observation.x_m, observation.y_m)
                    control_command = _compute_hover_control_command(
                        observation,
                        target_height_m=target_height_for_guard,
                        anchor_xy=drift_anchor_xy,
                        config=stability_config,
                    )
                    self._trim_state = self._trim_observer_state_from_observation(
                        observation,
                        target_height_m=target_height_for_guard,
                        config=stability_config,
                        current_state=self._trim_state,
                    )
                    control_command = self._apply_trim_to_control_command(
                        control_command,
                        trim_state=self._trim_state,
                        target_height_m=target_height_for_guard,
                        control_config=stability_config.anchor_control,
                    )
                    violations = _hover_stability_violations(
                        observation,
                        target_height_m=target_height_for_guard,
                        anchor_xy=drift_anchor_xy,
                        config=stability_config,
                        control_command=control_command,
                    )
                    if violations:
                        consecutive_unstable_samples += 1
                        if consecutive_unstable_samples >= max(
                            1,
                            int(stability_config.abort_after_unstable_samples),
                        ):
                            failure_codes = tuple(code for code, _message in violations)
                            failure_messages = tuple(message for _code, message in violations)
                            reason = "hover stability guard tripped: " + "; ".join(failure_messages)
                            _emit_trace(
                                self._trace_writer,
                                "hover_primitive_stability_guard",
                                status="blocked",
                                message=reason,
                                data={
                                    "phase": "hold",
                                    "failure_codes": failure_codes,
                                    "failures": failure_messages,
                                    "target_height_m": target_height_for_guard,
                                    "observed_height_m": observation.height_m,
                                    "trusted_height_m": control_command.trusted_height_m,
                                    "trusted_height_source": control_command.trusted_height_source,
                                    "sensor_disagreement_m": control_command.sensor_disagreement_m,
                                    "x_m": observation.x_m,
                                    "y_m": observation.y_m,
                                    "vx_mps": observation.vx_mps,
                                    "vy_mps": observation.vy_mps,
                                    "roll_deg": observation.roll_deg,
                                    "pitch_deg": observation.pitch_deg,
                                    "motion_squal": observation.motion_squal,
                                },
                            )
                            self.request_abort(reason)
                            raise HoverPrimitiveAbort(reason)
                    else:
                        consecutive_unstable_samples = 0
            self._send_anchor_hold_setpoint(
                control_command,
                target_height_m=target_height_for_guard,
            )
            next_deadline_s += period_s
            self._sleep_until(min(next_deadline_s, deadline_s))
        if target_height_m is not None:
            self._current_height_m = max(0.0, float(target_height_m))

    def _hold_height(
        self,
        height_m: float,
        duration_s: float,
        *,
        check_abort: bool = True,
        link_health_config: HoverLinkHealthConfig | None = None,
        stability_config: HoverStabilityConfig | None = None,
    ) -> None:
        """Keep one explicit hover height stable for a bounded duration."""

        self._current_height_m = max(0.0, float(height_m))
        self._hold(
            duration_s,
            check_abort=check_abort,
            link_health_config=link_health_config,
            stability_config=stability_config,
            target_height_m=self._current_height_m,
        )

    def _await_stable_hover(
        self,
        *,
        target_height_m: float,
        stability_config: HoverStabilityConfig,
        link_health_config: HoverLinkHealthConfig | None = None,
    ) -> None:
        """Wait for one bounded stable hover window before entering the hold phase."""

        if self._stability_provider is None:
            raise HoverPrimitiveSafetyError("hover stability provider is unavailable for the bounded hover guard")

        deadline_s = self._monotonic() + max(0.0, float(stability_config.settle_timeout_s))
        required_stable_samples = max(1, int(stability_config.required_stable_samples))
        consecutive_stable_samples = 0
        drift_anchor_xy: tuple[float, float] | None = None
        last_failure_codes: tuple[str, ...] | None = None
        last_failure_messages: tuple[str, ...] | None = None

        while True:
            self._ensure_not_aborted()
            self._ensure_link_health(link_health_config)
            observation = self._observe_hover_stability()
            if (
                drift_anchor_xy is None
                and observation.x_m is not None
                and observation.y_m is not None
                and observation.pose_age_s is not None
                and observation.pose_age_s <= float(stability_config.max_observation_age_s)
            ):
                drift_anchor_xy = (observation.x_m, observation.y_m)
            control_command = _compute_hover_control_command(
                observation,
                target_height_m=float(target_height_m),
                anchor_xy=drift_anchor_xy,
                config=stability_config,
            )
            self._trim_state = self._trim_observer_state_from_observation(
                observation,
                target_height_m=float(target_height_m),
                config=stability_config,
                current_state=self._trim_state,
            )
            control_command = self._apply_trim_to_control_command(
                control_command,
                trim_state=self._trim_state,
                target_height_m=float(target_height_m),
                control_config=stability_config.anchor_control,
            )
            violations = _hover_stability_violations(
                observation,
                target_height_m=float(target_height_m),
                anchor_xy=drift_anchor_xy,
                config=stability_config,
                control_command=control_command,
            )
            if violations:
                consecutive_stable_samples = 0
                failure_codes = tuple(code for code, _message in violations)
                failure_messages = tuple(message for _code, message in violations)
                if (
                    failure_codes != last_failure_codes
                    or failure_messages != last_failure_messages
                ):
                    last_failure_codes = failure_codes
                    last_failure_messages = failure_messages
                    _emit_trace(
                        self._trace_writer,
                        "hover_primitive_stability_guard",
                        status="degraded",
                        data={
                            "phase": "stabilize",
                            "failure_codes": failure_codes,
                            "failures": failure_messages,
                            "target_height_m": target_height_m,
                            "observed_height_m": observation.height_m,
                            "trusted_height_m": control_command.trusted_height_m,
                            "trusted_height_source": control_command.trusted_height_source,
                            "sensor_disagreement_m": control_command.sensor_disagreement_m,
                            "x_m": observation.x_m,
                            "y_m": observation.y_m,
                            "vx_mps": observation.vx_mps,
                            "vy_mps": observation.vy_mps,
                            "roll_deg": observation.roll_deg,
                            "pitch_deg": observation.pitch_deg,
                            "motion_squal": observation.motion_squal,
                        },
                    )
            else:
                consecutive_stable_samples += 1
                if consecutive_stable_samples >= required_stable_samples:
                    return

            self._send_anchor_hold_setpoint(
                control_command,
                target_height_m=float(target_height_m),
            )

            if self._monotonic() > deadline_s:
                failure_messages = last_failure_messages or ("hover stabilization timed out without fresh stable telemetry",)
                failure_codes = last_failure_codes or ("height_not_held",)
                reason = "hover stabilization failed: " + "; ".join(failure_messages)
                _emit_trace(
                    self._trace_writer,
                    "hover_primitive_stability_guard",
                    status="blocked",
                    message=reason,
                    data={
                        "phase": "stabilize",
                        "failure_codes": failure_codes,
                        "failures": failure_messages,
                        "target_height_m": target_height_m,
                    },
                )
                self.request_abort(reason)
                raise HoverPrimitiveAbort(reason)

            self._sleep(self._setpoint_period_s)

    def _wait_for_touchdown_confirmation(self, config: HoverPrimitiveConfig) -> None:
        """Wait until the landing path has a deterministic completion signal.

        Once airborne, this method must not abort the active landing sequence.
        It first waits a bounded time for explicit touchdown evidence. If that
        takes too long, it enters a deterministic recovery loop that keeps
        commanding zero-height hover setpoints until both the downward range and
        the firmware supervisor confirm touchdown. Recovery itself is also
        bounded; if combined confirmation never arrives, the helper can either
        force the final motor cutoff or raise a safety error.
        """

        threshold_m = max(0.0, float(config.touchdown_confirm_height_m))
        required_samples = max(1, int(config.touchdown_confirm_samples))
        primary_timeout_s = max(0.0, float(config.touchdown_confirm_timeout_s))
        recovery_timeout_s = max(0.0, float(config.touchdown_recovery_timeout_s))
        max_age_s = max(0.0, float(config.touchdown_confirm_max_age_s))
        require_supervisor_grounded = bool(config.touchdown_require_supervisor_grounded)
        range_only_confirmation_source = (
            str(config.touchdown_range_only_confirmation_source).strip() or "range_only"
        )

        primary_deadline_s = self._monotonic() + primary_timeout_s
        recovery_deadline_s = primary_deadline_s + recovery_timeout_s

        consecutive_grounded_samples = 0
        last_distance_m: float | None = None
        last_age_s: float | None = None
        last_is_flying: bool | None = None
        last_supervisor_age_s: float | None = None
        entered_recovery = False

        _emit_trace(
            self._trace_writer,
            "hover_primitive_touchdown_confirm",
            status="begin",
            data={
                "threshold_m": threshold_m,
                "required_samples": required_samples,
                "timeout_s": primary_timeout_s,
                "recovery_timeout_s": recovery_timeout_s,
                "max_age_s": max_age_s,
                "require_supervisor_grounded": require_supervisor_grounded,
                "range_only_confirmation_source": range_only_confirmation_source,
            },
        )

        while True:
            observation = self._observe_ground_distance()
            last_distance_m = observation.distance_m
            last_age_s = observation.age_s
            last_is_flying = observation.is_flying
            last_supervisor_age_s = observation.supervisor_age_s

            range_confirmed = (
                observation.distance_m is not None
                and observation.age_s is not None
                and observation.age_s <= max_age_s
                and observation.distance_m <= threshold_m
            )
            supervisor_confirmed = (
                observation.is_flying is False
                and observation.supervisor_age_s is not None
                and observation.supervisor_age_s <= max_age_s
            )
            touchdown_confirmed = range_confirmed and (
                supervisor_confirmed or not require_supervisor_grounded
            )
            if touchdown_confirmed:
                consecutive_grounded_samples += 1
            else:
                consecutive_grounded_samples = 0

            self._send_hover_setpoint(0.0)

            if consecutive_grounded_samples >= required_samples:
                confirmation_source = (
                    "range+supervisor"
                    if require_supervisor_grounded
                    else range_only_confirmation_source
                )
                self._touchdown_confirmation_source = confirmation_source
                self._touchdown_distance_m = observation.distance_m
                self._touchdown_supervisor_grounded = supervisor_confirmed
                _emit_trace(
                    self._trace_writer,
                    "hover_primitive_touchdown_confirm",
                    status="done",
                    data={
                        "confirmation_source": confirmation_source,
                        "distance_m": observation.distance_m,
                        "age_s": observation.age_s,
                        "confirmed_samples": consecutive_grounded_samples,
                        "is_flying": observation.is_flying,
                        "supervisor_age_s": observation.supervisor_age_s,
                    },
                )
                return

            now_s = self._monotonic()
            if not entered_recovery and now_s > primary_deadline_s:
                entered_recovery = True
                _emit_trace(
                    self._trace_writer,
                    "hover_primitive_touchdown_confirm",
                    status="degraded",
                    data={
                        "message": "bounded touchdown confirmation timed out; entering deterministic landing recovery",
                        "last_distance_m": last_distance_m,
                        "last_age_s": last_age_s,
                        "last_is_flying": last_is_flying,
                        "last_supervisor_age_s": last_supervisor_age_s,
                    },
                )

            if entered_recovery and now_s > recovery_deadline_s:
                if bool(config.force_motor_cutoff_after_touchdown_recovery):
                    self._forced_motor_cutoff = True
                    self._touchdown_confirmation_source = "timeout_forced_cutoff"
                    self._touchdown_distance_m = last_distance_m
                    _emit_trace(
                        self._trace_writer,
                        "hover_primitive_touchdown_confirm",
                        status="degraded",
                        data={
                            "confirmation_source": "timeout_forced_cutoff",
                            "message": "touchdown confirmation remained unavailable; forcing motor cutoff after bounded recovery",
                            "last_distance_m": last_distance_m,
                            "last_age_s": last_age_s,
                            "last_is_flying": last_is_flying,
                            "last_supervisor_age_s": last_supervisor_age_s,
                        },
                    )
                    return
                raise HoverPrimitiveSafetyError(
                    "touchdown confirmation remained unavailable after bounded landing recovery"
                )

            self._sleep(self._setpoint_period_s)

    def _land(self, config: HoverPrimitiveConfig) -> None:
        """Run a staged clean landing before the final motor cutoff."""

        period_s = self._setpoint_period_s
        current_height = max(0.0, float(self._current_height_m))
        if current_height <= 0.0:
            self._cut_motors(period_s, notify_stop_flush_s=config.notify_stop_flush_s)
            return

        floor_height = min(current_height, max(0.0, float(config.landing_floor_height_m)))
        touchdown_height = min(floor_height, max(0.0, float(config.touchdown_height_m)))
        approach_velocity_mps = max(0.01, float(config.land_velocity_mps))
        touchdown_velocity_mps = max(
            0.01,
            min(float(config.land_velocity_mps), float(config.touchdown_velocity_mps)),
        )

        if current_height > floor_height:
            _emit_trace(
                self._trace_writer,
                "hover_primitive_land_floor",
                status="begin",
                data={"floor_height_m": floor_height, "velocity_mps": approach_velocity_mps},
            )
            self._ramp_height(floor_height, velocity_mps=approach_velocity_mps, check_abort=False)
            _emit_trace(self._trace_writer, "hover_primitive_land_floor", status="done")
        else:
            self._current_height_m = floor_height

        if config.landing_floor_settle_s > 0:
            settle_phase = (
                "hover_primitive_landing_identify"
                if config.stability is not None
                else "hover_primitive_land_floor_settle"
            )
            _emit_trace(
                self._trace_writer,
                settle_phase,
                status="begin",
                data={"duration_s": config.landing_floor_settle_s, "height_m": floor_height},
            )
            self._hold_height(
                floor_height,
                float(config.landing_floor_settle_s),
                check_abort=False,
                stability_config=config.stability,
            )
            self._landing_trim_identified = self._landing_trim_identified or (config.stability is not None)
            _emit_trace(self._trace_writer, settle_phase, status="done")

        if self._current_height_m > touchdown_height:
            _emit_trace(
                self._trace_writer,
                "hover_primitive_touchdown",
                status="begin",
                data={"touchdown_height_m": touchdown_height, "velocity_mps": touchdown_velocity_mps},
            )
            self._ramp_height(touchdown_height, velocity_mps=touchdown_velocity_mps, check_abort=False)
            _emit_trace(self._trace_writer, "hover_primitive_touchdown", status="done")
        else:
            self._current_height_m = touchdown_height

        if config.touchdown_settle_s > 0:
            _emit_trace(
                self._trace_writer,
                "hover_primitive_touchdown_settle",
                status="begin",
                data={"duration_s": config.touchdown_settle_s, "height_m": touchdown_height},
            )
            self._hold_height(
                touchdown_height,
                float(config.touchdown_settle_s),
                check_abort=False,
                stability_config=config.stability,
            )
            _emit_trace(self._trace_writer, "hover_primitive_touchdown_settle", status="done")

        if self._current_height_m > 0.0:
            _emit_trace(
                self._trace_writer,
                "hover_primitive_zero_height",
                status="begin",
                data={"velocity_mps": touchdown_velocity_mps},
            )
            self._ramp_height(0.0, velocity_mps=touchdown_velocity_mps, check_abort=False)
            _emit_trace(self._trace_writer, "hover_primitive_zero_height", status="done")

        if config.zero_height_settle_s > 0:
            _emit_trace(
                self._trace_writer,
                "hover_primitive_zero_height_settle",
                status="begin",
                data={"duration_s": config.zero_height_settle_s},
            )
            self._hold_height(0.0, float(config.zero_height_settle_s), check_abort=False)
            _emit_trace(self._trace_writer, "hover_primitive_zero_height_settle", status="done")

        self._wait_for_touchdown_confirmation(config)
        self._cut_motors(period_s, notify_stop_flush_s=config.notify_stop_flush_s)

    def _cut_motors(self, period_s: float, *, notify_stop_flush_s: float = HOVER_NOTIFY_STOP_FLUSH_S) -> None:
        """Send the final stop/notified-stop sequence after touchdown settle."""

        send_stop_setpoint = getattr(self._cf.commander, "send_stop_setpoint", None)
        send_notify_setpoint_stop = getattr(self._cf.commander, "send_notify_setpoint_stop", None)
        for _ in range(HOVER_STOP_SETPOINT_REPEAT):
            if callable(send_stop_setpoint):
                send_stop_setpoint()
            self._sleep(period_s)
        if callable(send_notify_setpoint_stop):
            send_notify_setpoint_stop()
        if notify_stop_flush_s > 0:
            self._sleep(float(notify_stop_flush_s))
        self._current_height_m = 0.0
        self._landed = True
