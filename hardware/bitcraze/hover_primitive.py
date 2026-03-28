# CHANGELOG: 2026-03-27
# BUG-1: Reset per-run primitive state so one StatefulHoverPrimitive instance can execute multiple runs safely.
# BUG-2: Bounded touchdown recovery now prevents indefinite hangs on missing/stale touchdown telemetry.
# BUG-3: Trace-writer and telemetry-provider failures are now non-fatal to the flight path; observability errors no longer skip landing.
# BUG-4: Stop/notify-stop now drains the outbound queue before returning, matching Bitcraze's documented safe handover pattern.
# BUG-5: Estimator-settle logger setup failures now return a blocked report instead of crashing the hover test.
# SEC-1: Added runtime link-health gating so severe radio/USB degradation can trigger an early abort-to-land before watchdog shutdown.
# IMP-1: Pre-arm snapshots now capture previous controller params and expose restore_hover_pre_arm(...) for cleanup.
# IMP-2: Added supervisor-bit helpers, built-in cflib link-statistics integration, deadline-based scheduling, and richer landing telemetry.

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


@dataclass(frozen=True, slots=True)
class HoverPreArmConfig:
    """Describe the deterministic estimator/controller setup before takeoff."""

    estimator: int = 2
    controller: int = 1
    motion_disable: int = 0
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
class HoverPrimitiveConfig:
    """Describe one bounded takeoff-hold-land primitive."""

    target_height_m: float
    hover_duration_s: float
    takeoff_velocity_mps: float
    land_velocity_mps: float
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
    force_motor_cutoff_after_touchdown_recovery: bool = True
    zero_height_settle_s: float = HOVER_ZERO_HEIGHT_SETTLE_S
    notify_stop_flush_s: float = HOVER_NOTIFY_STOP_FLUSH_S
    link_health: HoverLinkHealthConfig | None = field(default_factory=HoverLinkHealthConfig)


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


@dataclass(frozen=True, slots=True)
class HoverGroundDistanceObservation:
    """Represent one fresh downward-range observation used during landing."""

    distance_m: float | None
    age_s: float | None
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
    motion_disable_before = _read_param_int(sync_cf, "motion.disable")

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
    if motion_disable_value != int(config.motion_disable):
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

    if snapshot.motion_disable_before is None:
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

    px_span = _series_span(sample.get("kalman.varPX") for sample in samples)
    py_span = _series_span(sample.get("kalman.varPY") for sample in samples)
    pz_span = _series_span(sample.get("kalman.varPZ") for sample in samples)
    for label, span in (
        ("kalman.varPX", px_span),
        ("kalman.varPY", py_span),
        ("kalman.varPZ", pz_span),
    ):
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

    squal_values: list[int] = []
    for sample in samples:
        sample_value = sample.get("motion.squal")
        if sample_value is not None:
            squal_values.append(int(sample_value))
    if not squal_values:
        failures.append("motion.squal never produced a usable estimator-settle sample")
    else:
        squal_mean = sum(squal_values) / len(squal_values)
        squal_good_ratio = sum(1 for value in squal_values if value >= int(config.motion_squal_min)) / len(
            squal_values
        )
        if squal_mean < float(config.motion_squal_min):
            failures.append(
                f"motion.squal mean {squal_mean:.1f} is below the {config.motion_squal_min} settle gate"
            )
        if squal_good_ratio < float(config.motion_squal_required_ratio):
            failures.append(
                "motion.squal stayed below the settle gate too often "
                f"({squal_good_ratio:.2f} < {config.motion_squal_required_ratio:.2f})"
            )

    if not any(sample.get("range.zrange_m") is not None for sample in samples):
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
        log_config.add_variable("motion.squal", "uint16_t")
        log_config.add_variable("stabilizer.roll", "float")
        log_config.add_variable("stabilizer.pitch", "float")
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
            zrange_observed=False,
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
                    "motion.squal": _normalize_float(data.get("motion.squal")),
                    "stabilizer.roll": _normalize_float(data.get("stabilizer.roll")),
                    "stabilizer.pitch": _normalize_float(data.get("stabilizer.pitch")),
                    "range.zrange_m": _normalized_zrange_m(data.get("range.zrange")),
                }
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
            zrange_observed=any(sample.get("range.zrange_m") is not None for sample in recent_window),
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
        zrange_observed=bool(zrange_values),
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
        link_health_provider: Callable[[], HoverLinkHealthObservation] | None = None,
        trace_writer: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cf = sync_cf.cf if hasattr(sync_cf, "cf") else sync_cf
        self._ground_distance_provider = ground_distance_provider
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
        self._link_unhealthy_since_s = None
        self._last_link_failures = None

    def run(self, config: HoverPrimitiveConfig) -> HoverPrimitiveOutcome:
        """Execute the bounded takeoff-hold-land primitive.

        Args:
            config: Desired takeoff, hold, and landing limits.

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
        link_health_config = config.link_health if config.link_health is not None and config.link_health.enabled else None

        try:
            self._final_phase = "takeoff"
            _emit_trace(
                self._trace_writer,
                "hover_primitive_takeoff",
                status="begin",
                data={
                    "target_height_m": config.target_height_m,
                    "velocity_mps": config.takeoff_velocity_mps,
                },
            )
            self._ramp_height(
                config.target_height_m,
                velocity_mps=config.takeoff_velocity_mps,
                check_abort=True,
                link_health_config=link_health_config,
            )
            _emit_trace(self._trace_writer, "hover_primitive_takeoff", status="done")

            self._final_phase = "hold"
            _emit_trace(
                self._trace_writer,
                "hover_primitive_hold",
                status="begin",
                data={"hover_duration_s": config.hover_duration_s},
            )
            self._hold(config.hover_duration_s, check_abort=True, link_health_config=link_health_config)
            _emit_trace(self._trace_writer, "hover_primitive_hold", status="done")
        except HoverPrimitiveAbort:
            self._final_phase = "abort_landing"
            abort_reason = self._abort_requested_reason
            _emit_trace(
                self._trace_writer,
                "hover_primitive_abort",
                status="begin",
                message=abort_reason,
            )
            self._land(config)
            _emit_trace(
                self._trace_writer,
                "hover_primitive_abort",
                status="done",
                message=abort_reason,
            )
            outcome = HoverPrimitiveOutcome(
                final_phase="abort_landing",
                took_off=self._took_off,
                landed=self._landed,
                aborted=True,
                abort_reason=abort_reason,
                commanded_max_height_m=self._commanded_max_height_m,
                setpoint_count=self._setpoint_count,
                forced_motor_cutoff=self._forced_motor_cutoff,
                touchdown_confirmation_source=self._touchdown_confirmation_source,
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
        outcome = HoverPrimitiveOutcome(
            final_phase="landed",
            took_off=self._took_off,
            landed=self._landed,
            aborted=False,
            abort_reason=None,
            commanded_max_height_m=self._commanded_max_height_m,
            setpoint_count=self._setpoint_count,
            forced_motor_cutoff=self._forced_motor_cutoff,
            touchdown_confirmation_source=self._touchdown_confirmation_source,
        )
        self._abort_requested_reason = None
        return outcome

    def _ensure_not_aborted(self) -> None:
        if self._abort_requested_reason is not None:
            raise HoverPrimitiveAbort(self._abort_requested_reason)

    def _send_hover_setpoint(self, height_m: float) -> None:
        bounded_height = max(0.0, float(height_m))
        self._cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, bounded_height)
        self._current_height_m = bounded_height
        self._commanded_max_height_m = max(self._commanded_max_height_m, bounded_height)
        self._setpoint_count += 1
        if bounded_height > 0.0:
            self._took_off = True

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

    def _hold(
        self,
        duration_s: float,
        *,
        check_abort: bool = True,
        link_health_config: HoverLinkHealthConfig | None = None,
    ) -> None:
        if duration_s <= 0:
            self._send_hover_setpoint(self._current_height_m)
            return
        deadline_s = self._monotonic() + float(duration_s)
        period_s = self._setpoint_period_s
        next_deadline_s = self._monotonic()
        while self._monotonic() < deadline_s:
            if check_abort:
                self._ensure_not_aborted()
                self._ensure_link_health(link_health_config)
            self._send_hover_setpoint(self._current_height_m)
            next_deadline_s += period_s
            self._sleep_until(min(next_deadline_s, deadline_s))

    def _hold_height(
        self,
        height_m: float,
        duration_s: float,
        *,
        check_abort: bool = True,
        link_health_config: HoverLinkHealthConfig | None = None,
    ) -> None:
        """Keep one explicit hover height stable for a bounded duration."""

        self._current_height_m = max(0.0, float(height_m))
        self._hold(duration_s, check_abort=check_abort, link_health_config=link_health_config)

    def _wait_for_touchdown_confirmation(self, config: HoverPrimitiveConfig) -> None:
        """Wait until the landing path has a deterministic completion signal.

        Once airborne, this method must not abort the active landing sequence.
        It first waits a bounded time for explicit touchdown evidence. If that
        takes too long, it enters a deterministic recovery loop that keeps
        commanding zero-height hover setpoints until either the downward range
        or the firmware supervisor confirms touchdown. Recovery itself is also
        bounded; if confirmation never arrives, the helper can either force the
        final motor cutoff or raise a safety error.
        """

        threshold_m = max(0.0, float(config.touchdown_confirm_height_m))
        required_samples = max(1, int(config.touchdown_confirm_samples))
        primary_timeout_s = max(0.0, float(config.touchdown_confirm_timeout_s))
        recovery_timeout_s = max(0.0, float(config.touchdown_recovery_timeout_s))
        max_age_s = max(0.0, float(config.touchdown_confirm_max_age_s))

        primary_deadline_s = self._monotonic() + primary_timeout_s
        recovery_deadline_s = primary_deadline_s + recovery_timeout_s

        consecutive_range_samples = 0
        consecutive_supervisor_samples = 0
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
            if range_confirmed:
                consecutive_range_samples += 1
            else:
                consecutive_range_samples = 0
            if supervisor_confirmed:
                consecutive_supervisor_samples += 1
            else:
                consecutive_supervisor_samples = 0

            self._send_hover_setpoint(0.0)

            if consecutive_range_samples >= required_samples:
                self._touchdown_confirmation_source = "range"
                _emit_trace(
                    self._trace_writer,
                    "hover_primitive_touchdown_confirm",
                    status="done",
                    data={
                        "confirmation_source": "range",
                        "distance_m": observation.distance_m,
                        "age_s": observation.age_s,
                        "confirmed_samples": consecutive_range_samples,
                        "is_flying": observation.is_flying,
                    },
                )
                return

            if consecutive_supervisor_samples >= required_samples:
                self._touchdown_confirmation_source = "supervisor"
                _emit_trace(
                    self._trace_writer,
                    "hover_primitive_touchdown_confirm",
                    status="done",
                    data={
                        "confirmation_source": "supervisor",
                        "distance_m": observation.distance_m,
                        "age_s": observation.age_s,
                        "confirmed_samples": consecutive_supervisor_samples,
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
                    _emit_trace(
                        self._trace_writer,
                        "hover_primitive_touchdown_confirm",
                        status="degraded",
                        data={
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
            _emit_trace(
                self._trace_writer,
                "hover_primitive_land_floor_settle",
                status="begin",
                data={"duration_s": config.landing_floor_settle_s, "height_m": floor_height},
            )
            self._hold_height(floor_height, float(config.landing_floor_settle_s), check_abort=False)
            _emit_trace(self._trace_writer, "hover_primitive_land_floor_settle", status="done")

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
            self._hold_height(touchdown_height, float(config.touchdown_settle_s), check_abort=False)
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

        self._wait_for_touchdown_confirmation(config)

        if config.zero_height_settle_s > 0:
            _emit_trace(
                self._trace_writer,
                "hover_primitive_zero_height_settle",
                status="begin",
                data={"duration_s": config.zero_height_settle_s},
            )
            self._hold_height(0.0, float(config.zero_height_settle_s), check_abort=False)
            _emit_trace(self._trace_writer, "hover_primitive_zero_height_settle", status="done")

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
