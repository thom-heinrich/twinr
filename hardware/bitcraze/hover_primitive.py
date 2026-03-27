"""Provide deterministic Crazyflie hover-primitive helpers.

This module owns the bounded low-level pieces behind Twinr's hover test:
deterministic controller/estimator pre-arm setup, Kalman settling checks, and
an explicit hover-setpoint primitive with a built-in abort/landing path. It
exists to keep ``run_hover_test.py`` focused on orchestration and reporting
instead of growing into a mixed hardware-control file.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import time
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
HOVER_LANDING_FLOOR_HEIGHT_M = 0.08
HOVER_LANDING_FLOOR_SETTLE_S = 0.25
HOVER_TOUCHDOWN_HEIGHT_M = 0.03
HOVER_TOUCHDOWN_SETTLE_S = 0.20
HOVER_TOUCHDOWN_VELOCITY_MPS = 0.08
HOVER_TOUCHDOWN_CONFIRM_HEIGHT_M = 0.05
HOVER_TOUCHDOWN_CONFIRM_SAMPLES = 3
HOVER_TOUCHDOWN_CONFIRM_TIMEOUT_S = 2.0
HOVER_TOUCHDOWN_CONFIRM_MAX_AGE_S = 0.35
HOVER_ZERO_HEIGHT_SETTLE_S = 0.10


@dataclass(frozen=True)
class HoverPreArmConfig:
    """Describe the deterministic estimator/controller setup before takeoff."""

    estimator: int = 2
    controller: int = 1
    motion_disable: int = 0
    reset_wait_s: float = HOVER_ESTIMATOR_RESET_SETTLE_S
    verify_attempts: int = HOVER_PARAM_VERIFY_ATTEMPTS
    verify_settle_s: float = HOVER_PARAM_VERIFY_SETTLE_S


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class HoverEstimatorSettlingConfig:
    """Describe the bounded Kalman-settling gate before hover takeoff."""

    timeout_s: float = HOVER_ESTIMATOR_SETTLE_TIMEOUT_S
    period_in_ms: int = HOVER_ESTIMATOR_SETTLE_PERIOD_MS
    window_size: int = HOVER_ESTIMATOR_SETTLE_WINDOW_SAMPLES
    variance_threshold: float = HOVER_ESTIMATOR_VARIANCE_THRESHOLD
    attitude_abs_max_deg: float = HOVER_ESTIMATOR_ATTITUDE_ABS_MAX_DEG
    motion_squal_min: int = HOVER_ESTIMATOR_MOTION_SQUAL_MIN
    motion_squal_required_ratio: float = HOVER_ESTIMATOR_MOTION_SQUAL_REQUIRED_RATIO


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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
    zero_height_settle_s: float = HOVER_ZERO_HEIGHT_SETTLE_S


@dataclass(frozen=True)
class HoverPrimitiveOutcome:
    """Summarize one bounded hover primitive execution."""

    final_phase: str
    took_off: bool
    landed: bool
    aborted: bool
    abort_reason: str | None
    commanded_max_height_m: float
    setpoint_count: int


@dataclass(frozen=True)
class HoverGroundDistanceObservation:
    """Represent one fresh downward-range observation used during landing."""

    distance_m: float | None
    age_s: float | None
    is_flying: bool | None = None
    supervisor_age_s: float | None = None


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
    """Emit one optional trace event without depending on a concrete writer type."""

    if trace_writer is None:
        return
    emit = getattr(trace_writer, "emit", None)
    if emit is None:
        return
    emit(
        phase,
        status=status,
        message=message,
        data=data,
    )


def _param_handle(sync_cf: Any) -> Any:
    """Return the Crazyflie param handle from a Crazyflie or SyncCrazyflie object."""

    return sync_cf.cf.param if hasattr(sync_cf, "cf") else sync_cf.param


def _read_param_int(sync_cf: Any, name: str) -> int | None:
    """Read one Crazyflie param and normalize it to an integer when possible."""

    param = _param_handle(sync_cf)
    try:
        raw = param.get_value(name)
    except Exception:
        return None
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return None


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

    _emit_trace(
        trace_writer,
        "pre_arm_params",
        status="begin",
        data={
            "estimator": config.estimator,
            "controller": config.controller,
            "motion_disable": config.motion_disable,
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
    if kalman_reset_after != 0:
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
        kalman_reset_performed=True,
        verified=not failures,
        failures=tuple(failures),
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


def _normalize_float(raw: object) -> float | None:
    """Normalize one logger payload value into a finite float when possible."""

    if raw is None:
        return None
    try:
        value = float(cast(Any, raw))
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


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
    for label, span in (("kalman.varPX", px_span), ("kalman.varPY", py_span), ("kalman.varPZ", pz_span)):
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

    log_config = log_config_cls(name="hover-estimator-settle", period_in_ms=max(10, int(config.period_in_ms)))
    log_config.add_variable("kalman.varPX", "float")
    log_config.add_variable("kalman.varPY", "float")
    log_config.add_variable("kalman.varPZ", "float")
    log_config.add_variable("motion.squal", "uint16_t")
    log_config.add_variable("stabilizer.roll", "float")
    log_config.add_variable("stabilizer.pitch", "float")
    log_config.add_variable("range.zrange", "uint16_t")

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
    start = monotonic()
    samples: list[dict[str, float | None]] = []
    recent: deque[dict[str, float | None]] = deque(maxlen=max(1, int(config.window_size)))
    stable = False

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

    recent_window = tuple(recent)
    failures = [] if stable else _settling_failures(recent_window, config=config)
    if not stable:
        failures.insert(
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
        failures=tuple(failures),
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


class StatefulHoverPrimitive:
    """Run one explicit hover-setpoint primitive with bounded abort handling."""

    def __init__(
        self,
        sync_cf: Any,
        *,
        ground_distance_provider: Callable[[], HoverGroundDistanceObservation] | None = None,
        trace_writer: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cf = sync_cf.cf if hasattr(sync_cf, "cf") else sync_cf
        self._ground_distance_provider = ground_distance_provider
        self._trace_writer = trace_writer
        self._sleep = sleep
        self._monotonic = monotonic
        self._setpoint_period_s = HOVER_SETPOINT_PERIOD_S
        self._abort_reason: str | None = None
        self._current_height_m = 0.0
        self._commanded_max_height_m = 0.0
        self._setpoint_count = 0
        self._took_off = False
        self._landed = False
        self._final_phase = "idle"

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

        self._abort_reason = str(reason).strip() or "abort requested"

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

        self._setpoint_period_s = max(0.02, float(config.setpoint_period_s))
        self._final_phase = "takeoff"
        try:
            _emit_trace(
                self._trace_writer,
                "hover_primitive_takeoff",
                status="begin",
                data={
                    "target_height_m": config.target_height_m,
                    "velocity_mps": config.takeoff_velocity_mps,
                },
            )
            self._ramp_height(config.target_height_m, velocity_mps=config.takeoff_velocity_mps)
            _emit_trace(self._trace_writer, "hover_primitive_takeoff", status="done")

            self._final_phase = "hold"
            _emit_trace(
                self._trace_writer,
                "hover_primitive_hold",
                status="begin",
                data={"hover_duration_s": config.hover_duration_s},
            )
            self._hold(config.hover_duration_s)
            _emit_trace(self._trace_writer, "hover_primitive_hold", status="done")
        except HoverPrimitiveAbort:
            self._final_phase = "abort_landing"
            _emit_trace(
                self._trace_writer,
                "hover_primitive_abort",
                status="begin",
                message=self._abort_reason,
            )
            self._land(config)
            _emit_trace(
                self._trace_writer,
                "hover_primitive_abort",
                status="done",
                message=self._abort_reason,
            )
            return HoverPrimitiveOutcome(
                final_phase="abort_landing",
                took_off=self._took_off,
                landed=self._landed,
                aborted=True,
                abort_reason=self._abort_reason,
                commanded_max_height_m=self._commanded_max_height_m,
                setpoint_count=self._setpoint_count,
            )
        except KeyboardInterrupt:
            self._final_phase = "interrupt_landing"
            self._land(config)
            raise
        except Exception:
            self._final_phase = "error_landing"
            self._land(config)
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
        return HoverPrimitiveOutcome(
            final_phase="landed",
            took_off=self._took_off,
            landed=self._landed,
            aborted=False,
            abort_reason=None,
            commanded_max_height_m=self._commanded_max_height_m,
            setpoint_count=self._setpoint_count,
        )

    def _ensure_not_aborted(self) -> None:
        if self._abort_reason is not None:
            raise HoverPrimitiveAbort(self._abort_reason)

    def _send_hover_setpoint(self, height_m: float) -> None:
        bounded_height = max(0.0, float(height_m))
        self._cf.commander.send_hover_setpoint(0.0, 0.0, 0.0, bounded_height)
        self._current_height_m = bounded_height
        self._commanded_max_height_m = max(self._commanded_max_height_m, bounded_height)
        self._setpoint_count += 1
        if bounded_height > 0.0:
            self._took_off = True

    def _ramp_height(self, target_height_m: float, *, velocity_mps: float, check_abort: bool = True) -> None:
        start_height_m = float(self._current_height_m)
        target_height_m = max(0.0, float(target_height_m))
        velocity_mps = max(0.01, float(velocity_mps))
        period_s = self._setpoint_period_s
        duration_s = abs(target_height_m - start_height_m) / velocity_mps
        steps = max(1, int(math.ceil(duration_s / period_s)))
        for step_index in range(steps):
            if check_abort:
                self._ensure_not_aborted()
            progress = (step_index + 1) / steps
            next_height = start_height_m + ((target_height_m - start_height_m) * progress)
            self._send_hover_setpoint(next_height)
            self._sleep(period_s)

    def _hold(self, duration_s: float, *, check_abort: bool = True) -> None:
        if duration_s <= 0:
            self._send_hover_setpoint(self._current_height_m)
            return
        deadline = self._monotonic() + float(duration_s)
        period_s = self._setpoint_period_s
        while self._monotonic() < deadline:
            if check_abort:
                self._ensure_not_aborted()
            self._send_hover_setpoint(self._current_height_m)
            self._sleep(period_s)

    def _hold_height(self, height_m: float, duration_s: float, *, check_abort: bool = True) -> None:
        """Keep one explicit hover height stable for a bounded duration."""

        self._current_height_m = max(0.0, float(height_m))
        self._hold(duration_s, check_abort=check_abort)

    def _observe_ground_distance(self) -> HoverGroundDistanceObservation:
        """Return the latest downward-range observation for touchdown gating."""

        if self._ground_distance_provider is None:
            return HoverGroundDistanceObservation(distance_m=None, age_s=None)
        observation = self._ground_distance_provider()
        if isinstance(observation, HoverGroundDistanceObservation):
            return observation
        return HoverGroundDistanceObservation(distance_m=None, age_s=None)

    def _wait_for_touchdown_confirmation(self, config: HoverPrimitiveConfig) -> None:
        """Wait until the landing path has a deterministic completion signal.

        Once airborne, this method must not abort the active landing sequence.
        It first waits a bounded time for explicit touchdown evidence. If that
        takes too long, it enters a deterministic recovery loop that keeps
        commanding zero-height hover setpoints until either the downward range
        or the firmware supervisor confirms touchdown.
        """

        threshold_m = max(0.0, float(config.touchdown_confirm_height_m))
        required_samples = max(1, int(config.touchdown_confirm_samples))
        timeout_s = max(0.0, float(config.touchdown_confirm_timeout_s))
        max_age_s = max(0.0, float(config.touchdown_confirm_max_age_s))
        deadline = self._monotonic() + timeout_s
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
                "timeout_s": timeout_s,
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
            if not entered_recovery and self._monotonic() > deadline:
                entered_recovery = True
                _emit_trace(
                    self._trace_writer,
                    "hover_primitive_touchdown_confirm",
                    status="degraded",
                    data={
                        "message": "bounded touchdown confirmation timed out; continuing deterministic landing recovery",
                        "last_distance_m": last_distance_m,
                        "last_age_s": last_age_s,
                        "last_is_flying": last_is_flying,
                        "last_supervisor_age_s": last_supervisor_age_s,
                    },
                )
            self._sleep(self._setpoint_period_s)

    def _land(self, config: HoverPrimitiveConfig) -> None:
        """Run a staged clean landing before the final motor cutoff."""

        period_s = self._setpoint_period_s
        current_height = max(0.0, float(self._current_height_m))
        if current_height <= 0.0:
            self._cut_motors(period_s)
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

        self._cut_motors(period_s)

    def _cut_motors(self, period_s: float) -> None:
        """Send the final stop/notified-stop sequence after touchdown settle."""

        send_stop_setpoint = getattr(self._cf.commander, "send_stop_setpoint", None)
        send_notify_setpoint_stop = getattr(self._cf.commander, "send_notify_setpoint_stop", None)
        for _ in range(HOVER_STOP_SETPOINT_REPEAT):
            if callable(send_stop_setpoint):
                send_stop_setpoint()
            self._sleep(period_s)
        if callable(send_notify_setpoint_stop):
            send_notify_setpoint_stop()
        self._current_height_m = 0.0
        self._landed = True
