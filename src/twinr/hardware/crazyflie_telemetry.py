"""Own first-class Crazyflie runtime telemetry for Twinr.

This module centralizes the live Crazyflie telemetry lane that used to live
inside individual Bitcraze workers. It owns log-block planning, param
snapshots, bounded sample buffering, cflib link-statistics capture, trace-fed
command/failsafe state, and typed runtime snapshots that higher layers can
consume without reimplementing the transport details. It also carries the
shared post-flight summaries for the bounded hover lane, including the split
between raw-flow truth and airborne-window truth plus the bounded takeoff
lateral-command forensics used to debug early drift.

The design is intentionally profile-driven. Twinr must not try to stream the
entire Crazyflie log TOC at runtime. Instead, the runtime exposes the full TOC
shape for discoverability and then starts one explicit bounded profile per
consumer, such as operator status, hover acceptance, or local inspect.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from enum import Enum
import math
from numbers import Integral, Real
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence, cast

from twinr.hardware.crazyflie_flow_anchor import HeightTrustConfig, compute_trusted_height

HOVER_RANGE_INVALID_MM = 32000.0
_SUPERVISOR_CAN_FLY_MASK = 1 << 3
_SUPERVISOR_IS_FLYING_MASK = 1 << 4
_SUPERVISOR_TUMBLED_MASK = 1 << 5
_SUPERVISOR_LOCKED_MASK = 1 << 6
_SUPERVISOR_CRASHED_MASK = 1 << 7
_SUPERVISOR_FLAG_BITS = (
    ("tumbled", _SUPERVISOR_TUMBLED_MASK),
    ("locked", _SUPERVISOR_LOCKED_MASK),
    ("crashed", _SUPERVISOR_CRASHED_MASK),
)
_POWER_STATE_NAMES = {
    0: "battery",
    1: "charging",
    2: "charged",
    3: "low_power",
    4: "shutdown",
}

_TELEMETRY_TAKEOFF_MIN_CONFIRM_HEIGHT_M = 0.05
_TELEMETRY_TAKEOFF_FAILURE_TIMEOUT_S = 1.5
_TWINR_FS_STATE_MISSION_TAKEOFF = 6
_TWINR_FS_LATERAL_SOURCE_NONE = 0
_TWINR_FS_LATERAL_SOURCE_MISSION_TAKEOFF = 1


class TelemetryProfile(str, Enum):
    """Enumerate the supported runtime telemetry profiles."""

    OPERATOR = "operator"
    HOVER_ACCEPTANCE = "hover_acceptance"
    HOVER_ACCEPTANCE_SITL = "hover_acceptance_sitl"
    INSPECT_LOCAL_ZONE = "inspect_local_zone"
    FORENSICS = "forensics"


@dataclass(frozen=True, slots=True)
class TelemetryLogBlockSpec:
    """Describe one Crazyflie log block used by a runtime telemetry profile."""

    name: str
    variables: tuple[str, ...]
    required: bool = True
    period_in_ms: int = 100


@dataclass(frozen=True, slots=True)
class TelemetryParamGroupSpec:
    """Describe one Crazyflie parameter group captured into the runtime state."""

    group: str
    required: bool = False


@dataclass(frozen=True, slots=True)
class CrazyflieTelemetrySample:
    """Persist one normalized log sample from one Crazyflie log block."""

    timestamp_ms: int
    block_name: str
    values: dict[str, float | int | None]
    received_monotonic_s: float | None = None


@dataclass(frozen=True, slots=True)
class CrazyflieTelemetrySummary:
    """Summarize one bounded set of in-flight telemetry samples."""

    sample_count: int
    available_blocks: tuple[str, ...]
    skipped_blocks: tuple[str, ...]
    duration_s: float | None
    roll_abs_max_deg: float | None
    pitch_abs_max_deg: float | None
    xy_drift_m: float | None
    z_drift_m: float | None
    z_span_m: float | None
    vx_abs_max_mps: float | None
    vy_abs_max_mps: float | None
    vz_abs_max_mps: float | None
    horizontal_speed_max_mps: float | None
    flow_squal_min: int | None
    flow_squal_mean: float | None
    flow_nonzero_samples: int
    flow_observed: bool
    motion_delta_x_abs_max: float | None
    motion_delta_y_abs_max: float | None
    zrange_min_m: float | None
    zrange_max_m: float | None
    zrange_sample_count: int
    zrange_observed: bool
    front_min_m: float | None
    back_min_m: float | None
    left_min_m: float | None
    right_min_m: float | None
    up_min_m: float | None
    clearance_observed: bool
    thrust_mean: float | None
    thrust_max: float | None
    gyro_abs_max_dps: float | None
    battery_min_v: float | None
    battery_drop_v: float | None
    radio_rssi_latest_dbm: float | None
    radio_rssi_min_dbm: float | None
    radio_connected_latest: bool | None
    radio_disconnect_seen: bool
    latest_supervisor_info: int | None
    supervisor_flags_seen: tuple[str, ...]
    stable_supervisor: bool
    trusted_height_min_m: float | None = None
    trusted_height_max_m: float | None = None
    height_sensor_disagreement_max_m: float | None = None
    height_sensor_untrusted_samples: int = 0
    raw_sample_count: int = 0
    airborne_window_detected: bool = False
    analysis_window_status: str = "ok"
    raw_flow_squal_min: int | None = None
    raw_flow_squal_mean: float | None = None
    raw_flow_nonzero_samples: int = 0
    raw_flow_observed: bool = False
    takeoff_state_sample_count: int = 0
    takeoff_lateral_window_detected: bool = False
    takeoff_command_source_codes_seen: tuple[int, ...] = ()
    takeoff_commanded_vx_abs_max_mps: float | None = None
    takeoff_commanded_vy_abs_max_mps: float | None = None
    takeoff_estimated_vx_abs_max_mps: float | None = None
    takeoff_estimated_vy_abs_max_mps: float | None = None
    takeoff_estimated_horizontal_speed_max_mps: float | None = None
    takeoff_disturbance_estimate_vx_abs_max_mps: float | None = None
    takeoff_disturbance_estimate_vy_abs_max_mps: float | None = None
    takeoff_lateral_classification: str = "takeoff_state_not_observed"


@dataclass(frozen=True, slots=True)
class TelemetryCatalogSummary:
    """Describe the discovered Crazyflie log/param catalog sizes."""

    log_group_count: int
    log_variable_count: int
    param_group_count: int
    param_count: int


@dataclass(frozen=True, slots=True)
class GroundDistanceObservation:
    """Expose the latest downward-range observation for hover primitives."""

    distance_m: float | None
    age_s: float | None
    is_flying: bool | None = None
    supervisor_age_s: float | None = None
    supervisor_info: int | None = None


@dataclass(frozen=True, slots=True)
class LinkHealthObservation:
    """Expose the latest cflib link-statistics observation."""

    age_s: float | None
    latency_ms: float | None = None
    link_quality: float | None = None
    uplink_rssi: float | None = None
    uplink_rate_hz: float | None = None
    downlink_rate_hz: float | None = None
    uplink_congestion: float | None = None
    downlink_congestion: float | None = None


@dataclass(frozen=True, slots=True)
class DeckTelemetryState:
    """Represent Crazyflie deck flags from the runtime param surface."""

    flags: dict[str, int | None]
    refreshed_age_s: float | None


@dataclass(frozen=True, slots=True)
class PowerTelemetryState:
    """Represent current Crazyflie power telemetry."""

    vbat_v: float | None
    battery_level: int | None
    state: int | None
    state_name: str


@dataclass(frozen=True, slots=True)
class RangeTelemetryState:
    """Represent current downward and directional ranging telemetry."""

    zrange_m: float | None
    front_m: float | None
    back_m: float | None
    left_m: float | None
    right_m: float | None
    up_m: float | None
    downward_observed: bool


@dataclass(frozen=True, slots=True)
class FlightTelemetryState:
    """Represent current flight-state telemetry relevant to bounded autonomy."""

    roll_deg: float | None
    pitch_deg: float | None
    yaw_deg: float | None
    x_m: float | None
    y_m: float | None
    z_m: float | None
    vx_mps: float | None
    vy_mps: float | None
    vz_mps: float | None
    thrust: float | None
    motion_squal: int | None
    supervisor_info: int | None
    can_fly: bool | None
    is_flying: bool | None
    unsafe_supervisor_flags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LinkTelemetryState:
    """Represent current radio and link-statistics telemetry."""

    radio_connected: bool | None
    rssi_dbm: float | None
    observation_age_s: float | None
    latency_ms: float | None
    link_quality: float | None
    uplink_rssi: float | None
    uplink_rate_hz: float | None
    downlink_rate_hz: float | None
    uplink_congestion: float | None
    downlink_congestion: float | None
    monitor_available: bool
    monitor_failure: str | None


@dataclass(frozen=True, slots=True)
class FailsafeTelemetryState:
    """Represent the current `twinrFs` runtime surface."""

    loaded: bool
    protocol_version: int | None
    enabled: bool | None
    state_code: int | None
    state_name: str | None
    reason_code: int | None
    reason_name: str | None
    heartbeat_age_ms: int | None
    last_status_received_age_s: float | None
    session_id: int | None
    rejected_packets: int | None
    last_reject_code: int | None
    takeoff_debug_flags: int | None
    takeoff_range_fresh_count: int | None
    takeoff_range_rise_count: int | None
    takeoff_flow_live_count: int | None
    takeoff_attitude_quiet_count: int | None
    takeoff_truth_stale_count: int | None
    takeoff_truth_flap_count: int | None
    takeoff_progress_class: int | None
    filtered_battery_mv: float | None
    hover_thrust_estimate: float | None
    commanded_vx_mps: float | None
    commanded_vy_mps: float | None
    lateral_command_source_code: int | None
    disturbance_estimate_vx: float | None
    disturbance_estimate_vy: float | None
    disturbance_severity_permille: int | None
    disturbance_recoverable: bool | None


@dataclass(frozen=True, slots=True)
class CommandTelemetryState:
    """Represent the latest commanded primitive/mission state."""

    mission_name: str | None
    phase: str
    phase_status: str | None
    age_s: float | None
    target_height_m: float | None
    hover_duration_s: float | None
    forward_m: float | None
    left_m: float | None
    translation_velocity_mps: float | None
    takeoff_confirmed: bool
    aborted: bool
    abort_reason: str | None
    last_message: str | None


@dataclass(frozen=True, slots=True)
class TelemetryDivergenceEvent:
    """Describe one runtime command-vs-observed divergence."""

    code: str
    severity: str
    message: str


@dataclass(frozen=True, slots=True)
class TelemetrySnapshot:
    """Represent one bounded realtime telemetry snapshot."""

    profile: str
    collected_at: str
    healthy: bool
    failures: tuple[str, ...]
    freshness_by_signal: dict[str, float | None]
    catalog: TelemetryCatalogSummary
    deck: DeckTelemetryState
    power: PowerTelemetryState
    range: RangeTelemetryState
    flight: FlightTelemetryState
    link: LinkTelemetryState
    failsafe: FailsafeTelemetryState
    command: CommandTelemetryState
    divergences: tuple[TelemetryDivergenceEvent, ...]
    available_blocks: tuple[str, ...]
    skipped_blocks: tuple[str, ...]


class CompositeTraceWriter:
    """Fan one trace event out to multiple `emit()` sinks."""

    def __init__(self, *writers: Any) -> None:
        self._writers = tuple((writer for writer in writers if writer is not None))

    def emit(
        self,
        phase: str,
        *,
        status: str,
        message: str | None = None,
        data: Mapping[str, object] | None = None,
    ) -> None:
        """Forward one trace event to every configured sink."""

        for writer in self._writers:
            emit = getattr(writer, "emit", None)
            if not callable(emit):
                raise AttributeError("trace sink must expose an emit() method")
            emit(phase, status=status, message=message, data=data)


def _normalize_numeric_value(raw: object) -> float | int | None:
    """Normalize one Crazyflie log/param value into a bounded numeric type."""

    if raw is None:
        return None
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, Integral):
        return int(raw)
    if isinstance(raw, Real):
        value = float(raw)
        if math.isnan(value) or math.isinf(value):
            return None
        return int(value) if value.is_integer() else value
    if isinstance(raw, (str, bytes, bytearray)):
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
    else:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return int(value) if value.is_integer() else value


def _normalize_range_mm_to_m(raw: object) -> float | None:
    """Normalize one downward/directional range reading into meters."""

    numeric = _normalize_numeric_value(raw)
    if numeric is None:
        return None
    value_mm = float(numeric)
    if value_mm <= 0.0 or value_mm >= HOVER_RANGE_INVALID_MM:
        return None
    return value_mm / 1000.0


def _supervisor_flag_names(value: int | None) -> tuple[str, ...]:
    """Return the currently asserted supervisor warning flags."""

    if value is None:
        return ()
    return tuple((name for name, mask in _SUPERVISOR_FLAG_BITS if int(value) & mask))


def _power_state_name(state: int | None) -> str:
    """Render one Crazyflie PM state into a stable operator token."""

    if state is None:
        return "unknown"
    return _POWER_STATE_NAMES.get(int(state), f"unknown_{int(state)}")


def _series_for_key(
    samples: Iterable[CrazyflieTelemetrySample],
    key: str,
    *,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> list[tuple[int, float]]:
    """Collect one telemetry variable as a timestamped float series."""

    values: list[tuple[int, float]] = []
    for sample in samples:
        if start_ms is not None and sample.timestamp_ms < start_ms:
            continue
        if end_ms is not None and sample.timestamp_ms > end_ms:
            continue
        value = sample.values.get(key)
        if value is None:
            continue
        values.append((int(sample.timestamp_ms), float(value)))
    return values


def _values_for_key(
    samples: Iterable[CrazyflieTelemetrySample],
    key: str,
    *,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> list[float]:
    """Collect one telemetry variable without timestamps."""

    return [value for _, value in _series_for_key(samples, key, start_ms=start_ms, end_ms=end_ms)]


def _drift(values: Sequence[float]) -> float | None:
    """Return end-minus-start drift for one telemetry series."""

    if len(values) < 2:
        return None
    return float(values[-1] - values[0])


def _span(values: Sequence[float]) -> float | None:
    """Return min/max span for one telemetry series."""

    if not values:
        return None
    return float(max(values) - min(values))


def _valid_range_values_m(values_mm: Iterable[float]) -> list[float]:
    """Filter invalid millimeter range sentinels and convert to meters."""

    values_m: list[float] = []
    for value in values_mm:
        numeric = float(value)
        if numeric <= 0.0 or numeric >= HOVER_RANGE_INVALID_MM:
            continue
        values_m.append(numeric / 1000.0)
    return values_m


def _airborne_window_bounds_ms(
    samples: tuple[CrazyflieTelemetrySample, ...],
) -> tuple[int | None, int | None]:
    """Infer the in-air telemetry window from altitude and downward range."""

    candidate_timestamps = [
        timestamp_ms
        for timestamp_ms, value in _series_for_key(samples, "stateEstimate.z")
        if value >= 0.02
    ]
    candidate_timestamps.extend(
        (
            timestamp_ms
            for timestamp_ms, value in _series_for_key(samples, "range.zrange")
            if 0.0 < value < HOVER_RANGE_INVALID_MM and value / 1000.0 >= 0.02
        )
    )
    if not candidate_timestamps:
        return (None, None)
    return (min(candidate_timestamps), max(candidate_timestamps))


def _trusted_height_window_stats(
    samples: tuple[CrazyflieTelemetrySample, ...],
) -> tuple[float | None, float | None, float | None, int]:
    """Summarize trusted-height behavior across one telemetry window."""

    trust_config = HeightTrustConfig()
    last_raw_height_m: float | None = None
    last_raw_timestamp_ms: int | None = None
    last_estimate_z_m: float | None = None
    last_estimate_timestamp_ms: int | None = None
    trusted_values_m: list[float] = []
    sensor_disagreements_m: list[float] = []
    untrusted_samples = 0

    for sample in samples:
        raw_value = sample.values.get("range.zrange")
        if raw_value is not None:
            raw_numeric = float(raw_value)
            if 0.0 < raw_numeric < HOVER_RANGE_INVALID_MM:
                last_raw_height_m = raw_numeric / 1000.0
                last_raw_timestamp_ms = sample.timestamp_ms
        estimate_value = sample.values.get("stateEstimate.z")
        if estimate_value is not None:
            last_estimate_z_m = float(estimate_value)
            last_estimate_timestamp_ms = sample.timestamp_ms

        raw_age_s = None
        if last_raw_timestamp_ms is not None:
            raw_age_s = max(0.0, (sample.timestamp_ms - last_raw_timestamp_ms) / 1000.0)
        estimate_age_s = None
        if last_estimate_timestamp_ms is not None:
            estimate_age_s = max(0.0, (sample.timestamp_ms - last_estimate_timestamp_ms) / 1000.0)

        trusted_height = compute_trusted_height(
            raw_height_m=last_raw_height_m,
            raw_height_age_s=raw_age_s,
            estimate_z_m=last_estimate_z_m,
            estimate_z_age_s=estimate_age_s,
            config=trust_config,
        )
        if trusted_height.height_m is not None:
            trusted_values_m.append(float(trusted_height.height_m))
        if trusted_height.sensor_disagreement_m is not None:
            sensor_disagreements_m.append(float(trusted_height.sensor_disagreement_m))
        if trusted_height.failures:
            untrusted_samples += 1

    return (
        min(trusted_values_m) if trusted_values_m else None,
        max(trusted_values_m) if trusted_values_m else None,
        max(sensor_disagreements_m) if sensor_disagreements_m else None,
        untrusted_samples,
    )


def _state_window_samples(
    samples: tuple[CrazyflieTelemetrySample, ...],
    *,
    state_key: str,
    state_code: int,
    padding_ms: int,
) -> tuple[CrazyflieTelemetrySample, ...]:
    """Return one bounded raw-sample window for a specific on-device state."""

    state_timestamps_ms = [
        sample.timestamp_ms
        for sample in samples
        if any(int(value) == int(state_code) for value in _values_for_key((sample,), state_key))
    ]
    if not state_timestamps_ms:
        return ()
    window_start_ms = max(0, min(state_timestamps_ms) - max(0, int(padding_ms)))
    window_end_ms = max(state_timestamps_ms) + max(0, int(padding_ms))
    return tuple(sample for sample in samples if window_start_ms <= sample.timestamp_ms <= window_end_ms)


def summarize_crazyflie_telemetry(
    samples: tuple[CrazyflieTelemetrySample, ...],
    *,
    available_blocks: tuple[str, ...] = (),
    skipped_blocks: tuple[str, ...] = (),
) -> CrazyflieTelemetrySummary:
    """Summarize one bounded Crazyflie telemetry sample window."""

    window_start_ms, window_end_ms = _airborne_window_bounds_ms(samples)
    filter_start_ms = None if window_start_ms is None else max(0, window_start_ms - 100)
    filter_end_ms = None if window_end_ms is None else window_end_ms + 100
    window_samples = tuple(
        sample
        for sample in samples
        if filter_start_ms is not None
        and filter_end_ms is not None
        and filter_start_ms <= sample.timestamp_ms <= filter_end_ms
    )
    if filter_start_ms is None or filter_end_ms is None:
        window_samples = ()
    analysis_window_status = "ok"
    if not samples:
        analysis_window_status = "no_samples"
    elif filter_start_ms is None or filter_end_ms is None:
        analysis_window_status = "raw_samples_missing_airborne_window"

    raw_squal_values = [int(value) for value in _values_for_key(samples, "motion.squal")]
    roll_values = _values_for_key(window_samples, "stabilizer.roll")
    pitch_values = _values_for_key(window_samples, "stabilizer.pitch")
    x_values = _values_for_key(window_samples, "stateEstimate.x")
    y_values = _values_for_key(window_samples, "stateEstimate.y")
    z_values = _values_for_key(window_samples, "stateEstimate.z")
    vx_values = _values_for_key(window_samples, "stateEstimate.vx")
    vy_values = _values_for_key(window_samples, "stateEstimate.vy")
    vz_values = _values_for_key(window_samples, "stateEstimate.vz")
    squal_values = [int(value) for value in _values_for_key(window_samples, "motion.squal")]
    delta_x_values = _values_for_key(window_samples, "motion.deltaX")
    delta_y_values = _values_for_key(window_samples, "motion.deltaY")
    zrange_values_m = _valid_range_values_m(_values_for_key(window_samples, "range.zrange"))
    front_values_m = _valid_range_values_m(_values_for_key(window_samples, "range.front"))
    back_values_m = _valid_range_values_m(_values_for_key(window_samples, "range.back"))
    left_values_m = _valid_range_values_m(_values_for_key(window_samples, "range.left"))
    right_values_m = _valid_range_values_m(_values_for_key(window_samples, "range.right"))
    up_values_m = _valid_range_values_m(_values_for_key(window_samples, "range.up"))
    thrust_values = _values_for_key(window_samples, "stabilizer.thrust")
    gyro_values = [
        *(_values_for_key(window_samples, "gyro.x")),
        *(_values_for_key(window_samples, "gyro.y")),
        *(_values_for_key(window_samples, "gyro.z")),
    ]
    battery_values = _values_for_key(window_samples, "pm.vbat")
    radio_rssi_values = _values_for_key(window_samples, "radio.rssi")
    radio_connected_values = [
        int(value) for value in _values_for_key(window_samples, "radio.isConnected")
    ]
    supervisor_values = [int(value) for value in _values_for_key(window_samples, "supervisor.info")]
    trusted_height_min_m, trusted_height_max_m, height_sensor_disagreement_max_m, height_sensor_untrusted_samples = _trusted_height_window_stats(
        window_samples
    )
    takeoff_samples = _state_window_samples(
        samples,
        state_key="twinrFs.state",
        state_code=_TWINR_FS_STATE_MISSION_TAKEOFF,
        padding_ms=100,
    )
    takeoff_cmd_vx_values = _values_for_key(takeoff_samples, "twinrFs.cmdVx")
    takeoff_cmd_vy_values = _values_for_key(takeoff_samples, "twinrFs.cmdVy")
    takeoff_cmd_source_values = [
        int(value) for value in _values_for_key(takeoff_samples, "twinrFs.cmdSrc")
    ]
    takeoff_estimated_vx_values = _values_for_key(takeoff_samples, "stateEstimate.vx")
    takeoff_estimated_vy_values = _values_for_key(takeoff_samples, "stateEstimate.vy")
    takeoff_disturbance_vx_values = _values_for_key(takeoff_samples, "twinrFs.distVx")
    takeoff_disturbance_vy_values = _values_for_key(takeoff_samples, "twinrFs.distVy")
    takeoff_lateral_classification = "takeoff_state_not_observed"
    if takeoff_samples:
        takeoff_lateral_classification = "takeoff_lateral_telemetry_missing"
        if takeoff_cmd_source_values or takeoff_cmd_vx_values or takeoff_cmd_vy_values:
            unexpected_source_seen = any(
                value not in {_TWINR_FS_LATERAL_SOURCE_NONE, _TWINR_FS_LATERAL_SOURCE_MISSION_TAKEOFF}
                for value in takeoff_cmd_source_values
            )
            nonzero_command_seen = any((value != 0.0 for value in takeoff_cmd_vx_values)) or any(
                (value != 0.0 for value in takeoff_cmd_vy_values)
            )
            nonzero_disturbance_seen = any((value != 0.0 for value in takeoff_disturbance_vx_values)) or any(
                (value != 0.0 for value in takeoff_disturbance_vy_values)
            )
            if unexpected_source_seen:
                takeoff_lateral_classification = "unexpected_lateral_source_during_takeoff"
            elif nonzero_command_seen:
                takeoff_lateral_classification = "nonzero_on_device_lateral_command_during_takeoff"
            elif nonzero_disturbance_seen:
                takeoff_lateral_classification = "estimator_bias_present_without_on_device_lateral_command"
            else:
                takeoff_lateral_classification = "no_on_device_lateral_command_or_estimator_bias_during_takeoff"
    xy_drift_m = None
    if len(x_values) >= 2 and len(y_values) >= 2:
        xy_drift_m = math.sqrt((x_values[-1] - x_values[0]) ** 2 + (y_values[-1] - y_values[0]) ** 2)
    horizontal_speed_max_mps = None
    if vx_values and vy_values:
        horizontal_speed_max_mps = max(
            (
                math.sqrt(vx_value * vx_value + vy_value * vy_value)
                for vx_value, vy_value in zip(vx_values, vy_values)
            ),
            default=None,
        )
    flags_seen = tuple(
        name
        for name, mask in _SUPERVISOR_FLAG_BITS
        if any((int(value) & mask) != 0 for value in supervisor_values)
    )
    return CrazyflieTelemetrySummary(
        sample_count=len(window_samples),
        raw_sample_count=len(samples),
        airborne_window_detected=filter_start_ms is not None and filter_end_ms is not None,
        available_blocks=available_blocks,
        skipped_blocks=skipped_blocks,
        duration_s=max(0.0, (window_end_ms - window_start_ms) / 1000.0)
        if window_start_ms is not None and window_end_ms is not None
        else None,
        roll_abs_max_deg=max((abs(value) for value in roll_values), default=None),
        pitch_abs_max_deg=max((abs(value) for value in pitch_values), default=None),
        xy_drift_m=xy_drift_m,
        z_drift_m=_drift(z_values),
        z_span_m=_span(z_values),
        vx_abs_max_mps=max((abs(value) for value in vx_values), default=None),
        vy_abs_max_mps=max((abs(value) for value in vy_values), default=None),
        vz_abs_max_mps=max((abs(value) for value in vz_values), default=None),
        horizontal_speed_max_mps=horizontal_speed_max_mps,
        flow_squal_min=min(squal_values) if squal_values else None,
        flow_squal_mean=sum(squal_values) / len(squal_values) if squal_values else None,
        flow_nonzero_samples=sum((1 for value in squal_values if value > 0)),
        flow_observed=any((value > 0 for value in squal_values)),
        motion_delta_x_abs_max=max((abs(value) for value in delta_x_values), default=None),
        motion_delta_y_abs_max=max((abs(value) for value in delta_y_values), default=None),
        zrange_min_m=min(zrange_values_m) if zrange_values_m else None,
        zrange_max_m=max(zrange_values_m) if zrange_values_m else None,
        zrange_sample_count=len(zrange_values_m),
        zrange_observed=any((value > 0 for value in zrange_values_m)),
        front_min_m=min(front_values_m) if front_values_m else None,
        back_min_m=min(back_values_m) if back_values_m else None,
        left_min_m=min(left_values_m) if left_values_m else None,
        right_min_m=min(right_values_m) if right_values_m else None,
        up_min_m=min(up_values_m) if up_values_m else None,
        clearance_observed=any((front_values_m, back_values_m, left_values_m, right_values_m, up_values_m)),
        thrust_mean=sum(thrust_values) / len(thrust_values) if thrust_values else None,
        thrust_max=max(thrust_values) if thrust_values else None,
        gyro_abs_max_dps=max((abs(value) for value in gyro_values), default=None),
        battery_min_v=min(battery_values) if battery_values else None,
        battery_drop_v=(battery_values[0] - battery_values[-1]) if len(battery_values) >= 2 else None,
        radio_rssi_latest_dbm=radio_rssi_values[-1] if radio_rssi_values else None,
        radio_rssi_min_dbm=min(radio_rssi_values) if radio_rssi_values else None,
        radio_connected_latest=bool(radio_connected_values[-1]) if radio_connected_values else None,
        radio_disconnect_seen=any((value == 0 for value in radio_connected_values)),
        latest_supervisor_info=supervisor_values[-1] if supervisor_values else None,
        supervisor_flags_seen=flags_seen,
        stable_supervisor=not flags_seen,
        trusted_height_min_m=trusted_height_min_m,
        trusted_height_max_m=trusted_height_max_m,
        height_sensor_disagreement_max_m=height_sensor_disagreement_max_m,
        height_sensor_untrusted_samples=height_sensor_untrusted_samples,
        analysis_window_status=analysis_window_status,
        raw_flow_squal_min=min(raw_squal_values) if raw_squal_values else None,
        raw_flow_squal_mean=sum(raw_squal_values) / len(raw_squal_values) if raw_squal_values else None,
        raw_flow_nonzero_samples=sum((1 for value in raw_squal_values if value > 0)),
        raw_flow_observed=any((value > 0 for value in raw_squal_values)),
        takeoff_state_sample_count=len(takeoff_samples),
        takeoff_lateral_window_detected=bool(takeoff_samples),
        takeoff_command_source_codes_seen=tuple(sorted(set(takeoff_cmd_source_values))),
        takeoff_commanded_vx_abs_max_mps=max((abs(value) for value in takeoff_cmd_vx_values), default=None),
        takeoff_commanded_vy_abs_max_mps=max((abs(value) for value in takeoff_cmd_vy_values), default=None),
        takeoff_estimated_vx_abs_max_mps=max((abs(value) for value in takeoff_estimated_vx_values), default=None),
        takeoff_estimated_vy_abs_max_mps=max((abs(value) for value in takeoff_estimated_vy_values), default=None),
        takeoff_estimated_horizontal_speed_max_mps=max(
            (
                math.sqrt(vx_value * vx_value + vy_value * vy_value)
                for vx_value, vy_value in zip(takeoff_estimated_vx_values, takeoff_estimated_vy_values)
            ),
            default=None,
        ),
        takeoff_disturbance_estimate_vx_abs_max_mps=max(
            (abs(value) for value in takeoff_disturbance_vx_values),
            default=None,
        ),
        takeoff_disturbance_estimate_vy_abs_max_mps=max(
            (abs(value) for value in takeoff_disturbance_vy_values),
            default=None,
        ),
        takeoff_lateral_classification=takeoff_lateral_classification,
    )


OPERATOR_LOG_BLOCKS = (
    TelemetryLogBlockSpec(
        name="operator-health",
        variables=(
            "pm.vbat",
            "pm.batteryLevel",
            "pm.state",
            "radio.isConnected",
            "radio.rssi",
            "supervisor.info",
            "range.zrange",
            "motion.squal",
        ),
    ),
    TelemetryLogBlockSpec(
        name="operator-clearance",
        variables=("range.front", "range.back", "range.left", "range.right", "range.up"),
        required=False,
    ),
    TelemetryLogBlockSpec(
        name="operator-failsafe",
        variables=("twinrFs.state", "twinrFs.reason", "twinrFs.heartbeatAgeMs"),
        required=False,
    ),
)

HOVER_ACCEPTANCE_LOG_BLOCKS = (
    TelemetryLogBlockSpec(
        name="hover-attitude",
        variables=(
            "stabilizer.roll",
            "stabilizer.pitch",
            "stabilizer.yaw",
            "stateEstimate.x",
            "stateEstimate.y",
            "stateEstimate.z",
        ),
    ),
    TelemetryLogBlockSpec(
        name="hover-sensors",
        variables=(
            "motion.squal",
            "motion.deltaX",
            "motion.deltaY",
            "range.zrange",
            "pm.vbat",
            "supervisor.info",
            "radio.rssi",
            "radio.isConnected",
        ),
    ),
    TelemetryLogBlockSpec(
        name="hover-velocity",
        variables=("stateEstimate.vx", "stateEstimate.vy", "stateEstimate.vz", "stabilizer.thrust"),
    ),
    TelemetryLogBlockSpec(
        name="hover-gyro",
        variables=("gyro.x", "gyro.y", "gyro.z"),
    ),
    TelemetryLogBlockSpec(
        name="hover-clearance",
        variables=("range.front", "range.back", "range.left", "range.right", "range.up"),
        required=False,
    ),
    TelemetryLogBlockSpec(
        name="hover-failsafe",
        variables=(
            "twinrFs.state",
            "twinrFs.reason",
            "twinrFs.heartbeatAgeMs",
            "twinrFs.rejectedPkts",
            "twinrFs.downRangeMm",
            "twinrFs.tkDbg",
            "twinrFs.tkRfCnt",
            "twinrFs.tkRrCnt",
            "twinrFs.tkFlCnt",
            "twinrFs.tkAtCnt",
            "twinrFs.tkStCnt",
            "twinrFs.tkFpCnt",
            "twinrFs.tkPgCls",
            "twinrFs.tkBatMv",
            "twinrFs.thrEst",
            "twinrFs.cmdVx",
            "twinrFs.cmdVy",
            "twinrFs.cmdSrc",
            "twinrFs.distVx",
            "twinrFs.distVy",
            "twinrFs.distSev",
            "twinrFs.distRec",
        ),
        required=False,
    ),
)

HOVER_ACCEPTANCE_SITL_LOG_BLOCKS = (
    TelemetryLogBlockSpec(
        name="hover-attitude",
        variables=(
            "stabilizer.roll",
            "stabilizer.pitch",
            "stabilizer.yaw",
            "stateEstimate.x",
            "stateEstimate.y",
            "stateEstimate.z",
        ),
    ),
    TelemetryLogBlockSpec(
        name="hover-sensors",
        variables=(
            "range.zrange",
            "pm.vbat",
            "pm.state",
            "supervisor.info",
            "radio.rssi",
            "radio.isConnected",
        ),
    ),
    TelemetryLogBlockSpec(
        name="hover-velocity",
        variables=("stateEstimate.vx", "stateEstimate.vy", "stateEstimate.vz", "stabilizer.thrust"),
    ),
    TelemetryLogBlockSpec(
        name="hover-gyro",
        variables=("gyro.x", "gyro.y", "gyro.z"),
    ),
    TelemetryLogBlockSpec(
        name="hover-clearance",
        variables=("range.front", "range.back", "range.left", "range.right", "range.up"),
        required=False,
    ),
    TelemetryLogBlockSpec(
        name="hover-failsafe",
        variables=(
            "twinrFs.state",
            "twinrFs.reason",
            "twinrFs.heartbeatAgeMs",
            "twinrFs.rejectedPkts",
            "twinrFs.downRangeMm",
            "twinrFs.tkDbg",
            "twinrFs.tkRfCnt",
            "twinrFs.tkRrCnt",
            "twinrFs.tkFlCnt",
            "twinrFs.tkAtCnt",
            "twinrFs.tkStCnt",
            "twinrFs.tkFpCnt",
            "twinrFs.tkPgCls",
            "twinrFs.tkBatMv",
            "twinrFs.thrEst",
            "twinrFs.cmdVx",
            "twinrFs.cmdVy",
            "twinrFs.cmdSrc",
            "twinrFs.distVx",
            "twinrFs.distVy",
            "twinrFs.distSev",
            "twinrFs.distRec",
        ),
        required=False,
    ),
)

FORENSICS_EXTRA_LOG_BLOCKS = (
    TelemetryLogBlockSpec(
        name="hover-kalman",
        variables=(
            "kalman.varPX",
            "kalman.varPY",
            "kalman.varPZ",
            "kalman.statePX",
            "kalman.statePY",
            "kalman.statePZ",
        ),
        required=False,
    ),
)

PROFILE_LOG_BLOCKS: dict[TelemetryProfile, tuple[TelemetryLogBlockSpec, ...]] = {
    TelemetryProfile.OPERATOR: OPERATOR_LOG_BLOCKS,
    TelemetryProfile.HOVER_ACCEPTANCE: HOVER_ACCEPTANCE_LOG_BLOCKS,
    TelemetryProfile.HOVER_ACCEPTANCE_SITL: HOVER_ACCEPTANCE_SITL_LOG_BLOCKS,
    TelemetryProfile.INSPECT_LOCAL_ZONE: HOVER_ACCEPTANCE_LOG_BLOCKS,
    TelemetryProfile.FORENSICS: HOVER_ACCEPTANCE_LOG_BLOCKS + FORENSICS_EXTRA_LOG_BLOCKS,
}

PROFILE_PARAM_GROUPS: dict[TelemetryProfile, tuple[TelemetryParamGroupSpec, ...]] = {
    TelemetryProfile.OPERATOR: (
        TelemetryParamGroupSpec("deck", required=True),
        TelemetryParamGroupSpec("pm"),
        TelemetryParamGroupSpec("supervisor"),
        TelemetryParamGroupSpec("twinrFs"),
    ),
    TelemetryProfile.HOVER_ACCEPTANCE: (
        TelemetryParamGroupSpec("deck", required=True),
        TelemetryParamGroupSpec("motion"),
        TelemetryParamGroupSpec("stabilizer"),
        TelemetryParamGroupSpec("supervisor"),
        TelemetryParamGroupSpec("twinrFs"),
    ),
    TelemetryProfile.HOVER_ACCEPTANCE_SITL: (
        TelemetryParamGroupSpec("deck", required=True),
        TelemetryParamGroupSpec("motion"),
        TelemetryParamGroupSpec("stabilizer"),
        TelemetryParamGroupSpec("supervisor"),
        TelemetryParamGroupSpec("twinrFs"),
    ),
    TelemetryProfile.INSPECT_LOCAL_ZONE: (
        TelemetryParamGroupSpec("deck", required=True),
        TelemetryParamGroupSpec("motion"),
        TelemetryParamGroupSpec("stabilizer"),
        TelemetryParamGroupSpec("supervisor"),
        TelemetryParamGroupSpec("twinrFs"),
    ),
    TelemetryProfile.FORENSICS: (
        TelemetryParamGroupSpec("deck", required=True),
        TelemetryParamGroupSpec("motion"),
        TelemetryParamGroupSpec("stabilizer"),
        TelemetryParamGroupSpec("supervisor"),
        TelemetryParamGroupSpec("twinrFs"),
    ),
}

PROFILE_REQUIRED_SIGNALS: dict[TelemetryProfile, dict[str, float]] = {
    TelemetryProfile.OPERATOR: {
        "pm.vbat": 2.0,
        "radio.isConnected": 2.0,
        "supervisor.info": 2.0,
    },
    TelemetryProfile.HOVER_ACCEPTANCE: {
        "pm.vbat": 1.5,
        "radio.isConnected": 1.5,
        "supervisor.info": 1.5,
        "range.zrange": 1.5,
        "motion.squal": 1.5,
    },
    TelemetryProfile.HOVER_ACCEPTANCE_SITL: {
        "pm.vbat": 1.5,
        "radio.isConnected": 1.5,
        "supervisor.info": 1.5,
        "range.zrange": 1.5,
    },
    TelemetryProfile.INSPECT_LOCAL_ZONE: {
        "pm.vbat": 1.5,
        "radio.isConnected": 1.5,
        "supervisor.info": 1.5,
        "range.zrange": 1.5,
        "motion.squal": 1.5,
    },
    TelemetryProfile.FORENSICS: {
        "pm.vbat": 1.5,
        "radio.isConnected": 1.5,
        "supervisor.info": 1.5,
        "range.zrange": 1.5,
        "motion.squal": 1.5,
    },
}


def profile_log_blocks(profile: TelemetryProfile | str) -> tuple[TelemetryLogBlockSpec, ...]:
    """Return the log-block plan for one telemetry profile."""

    resolved = profile if isinstance(profile, TelemetryProfile) else TelemetryProfile(str(profile))
    return PROFILE_LOG_BLOCKS[resolved]


def profile_param_groups(profile: TelemetryProfile | str) -> tuple[TelemetryParamGroupSpec, ...]:
    """Return the param-group plan for one telemetry profile."""

    resolved = profile if isinstance(profile, TelemetryProfile) else TelemetryProfile(str(profile))
    return PROFILE_PARAM_GROUPS[resolved]


class _LinkStatisticsMonitor:
    """Capture cflib link-statistics into a pull-based bounded snapshot."""

    def __init__(self, cf: Any, *, monotonic: Callable[[], float]) -> None:
        self._cf = cf
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._subscriptions: list[tuple[Any, Callable[..., None]]] = []
        self._started_by_runtime = False
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
        self.available = False
        self.failure: str | None = None
        self._link_statistics = getattr(self._cf, "link_statistics", None)
        self._configure()

    def _configure(self) -> None:
        required = (
            "start",
            "stop",
            "latency_updated",
            "link_quality_updated",
            "uplink_rssi_updated",
            "uplink_rate_updated",
            "downlink_rate_updated",
            "uplink_congestion_updated",
            "downlink_congestion_updated",
        )
        if self._link_statistics is None:
            self.failure = "link_statistics_missing"
            return
        if not all(hasattr(self._link_statistics, name) for name in required):
            self.failure = "link_statistics_incomplete"
            return
        self.available = True

    def start(self) -> None:
        """Register bounded cflib callbacks and start collection."""

        if not self.available or self._link_statistics is None:
            return
        self._subscribe("latency_updated", "latency_ms")
        self._subscribe("link_quality_updated", "link_quality")
        self._subscribe("uplink_rssi_updated", "uplink_rssi")
        self._subscribe("uplink_rate_updated", "uplink_rate_hz")
        self._subscribe("downlink_rate_updated", "downlink_rate_hz")
        self._subscribe("uplink_congestion_updated", "uplink_congestion")
        self._subscribe("downlink_congestion_updated", "downlink_congestion")
        if not bool(getattr(self._link_statistics, "_is_active", False)):
            getattr(self._link_statistics, "start")()
            self._started_by_runtime = True

    def _subscribe(self, caller_name: str, field_name: str) -> None:
        caller = getattr(self._link_statistics, caller_name)
        add_callback = getattr(caller, "add_callback", None)
        if not callable(add_callback):
            self.failure = f"link_statistics_add_callback_missing:{caller_name}"
            self.available = False
            raise RuntimeError(self.failure)

        def _callback(value: object) -> None:
            normalized = _normalize_numeric_value(value)
            with self._lock:
                self._values[field_name] = None if normalized is None else float(normalized)
                self._last_update_s = self._monotonic()

        add_callback(_callback)
        self._subscriptions.append((caller, _callback))

    def observation(self) -> LinkHealthObservation:
        """Return the latest bounded link-health observation."""

        with self._lock:
            values = dict(self._values)
            last_update_s = self._last_update_s
        age_s = None if last_update_s is None else max(0.0, self._monotonic() - last_update_s)
        return LinkHealthObservation(
            age_s=age_s,
            latency_ms=values["latency_ms"],
            link_quality=values["link_quality"],
            uplink_rssi=values["uplink_rssi"],
            uplink_rate_hz=values["uplink_rate_hz"],
            downlink_rate_hz=values["downlink_rate_hz"],
            uplink_congestion=values["uplink_congestion"],
            downlink_congestion=values["downlink_congestion"],
        )

    def stop(self) -> tuple[str, ...]:
        """Detach callbacks and stop collection if this runtime started it."""

        failures: list[str] = []
        for caller, callback in reversed(self._subscriptions):
            remove_callback = getattr(caller, "remove_callback", None)
            if callable(remove_callback):
                try:
                    remove_callback(callback)
                except Exception as exc:  # pragma: no cover - live cflib cleanup
                    failures.append(f"link_callback_remove:{exc.__class__.__name__}:{exc}")
        self._subscriptions.clear()
        if self._started_by_runtime and self._link_statistics is not None:
            try:
                getattr(self._link_statistics, "stop")()
            except Exception as exc:  # pragma: no cover - live cflib cleanup
                failures.append(f"link_statistics_stop:{exc.__class__.__name__}:{exc}")
        self._started_by_runtime = False
        return tuple(failures)


class CrazyflieTelemetryRuntime:
    """Own one bounded Crazyflie runtime telemetry session."""

    def __init__(
        self,
        sync_cf: Any,
        log_config_cls: Any,
        *,
        profile: TelemetryProfile | str,
        max_samples: int = 512,
        period_in_ms: int | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cf = sync_cf.cf if hasattr(sync_cf, "cf") else sync_cf
        self._log_config_cls = log_config_cls
        self.profile = profile if isinstance(profile, TelemetryProfile) else TelemetryProfile(str(profile))
        self._log_blocks = profile_log_blocks(self.profile)
        self._param_groups = profile_param_groups(self.profile)
        self._required_signals = PROFILE_REQUIRED_SIGNALS[self.profile]
        self._max_samples = max(1, int(max_samples))
        self._period_in_ms_override = None if period_in_ms is None else max(10, int(period_in_ms))
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._configs: list[Any] = []
        self._samples: deque[CrazyflieTelemetrySample] = deque(maxlen=self._max_samples)
        self._available_blocks: list[str] = []
        self._skipped_blocks: dict[str, str] = {}
        self._latest_by_key: dict[str, tuple[float | int | None, float]] = {}
        self._param_snapshot: dict[str, object] = {}
        self._param_snapshot_monotonic_s: float | None = None
        self._catalog = TelemetryCatalogSummary(0, 0, 0, 0)
        self._failures: list[str] = []
        self._started = False
        self._command_state: dict[str, object] = {
            "mission_name": None,
            "phase": "idle",
            "phase_status": None,
            "updated_monotonic_s": None,
            "target_height_m": None,
            "hover_duration_s": None,
            "forward_m": None,
            "left_m": None,
            "translation_velocity_mps": None,
            "takeoff_confirmed": False,
            "aborted": False,
            "abort_reason": None,
            "last_message": None,
            "stability_guard_codes": (),
            "stability_guard_failures": (),
            "stability_guard_phase": None,
            "stability_guard_status": None,
            "touchdown_confirmation_source": None,
            "touchdown_distance_m": None,
            "touchdown_supervisor_grounded": False,
        }
        self._failsafe_trace_state: dict[str, object] = {
            "state_code": None,
            "state_name": None,
            "reason_code": None,
            "reason_name": None,
            "session_id": None,
            "last_status_received_monotonic_s": None,
            "down_range_mm": None,
            "pre_liftoff_trigger_count": 0,
        }
        self._link_monitor = _LinkStatisticsMonitor(self._cf, monotonic=self._monotonic)

    @property
    def available_blocks(self) -> tuple[str, ...]:
        """Return the successfully started log blocks."""

        return tuple(self._available_blocks)

    @property
    def skipped_blocks(self) -> tuple[str, ...]:
        """Return the optional log blocks that failed to start."""

        return tuple(self._skipped_blocks)

    @property
    def skipped_block_reasons(self) -> dict[str, str]:
        """Return the failure reasons for optional skipped blocks."""

        return dict(self._skipped_blocks)

    def start(self) -> None:
        """Start the configured telemetry profile."""

        if self._started:
            return
        self._catalog = self._snapshot_catalog()
        self.refresh_params()
        self._link_monitor.start()
        for spec in self._log_blocks:
            config = self._log_config_cls(
                name=spec.name,
                period_in_ms=self._period_in_ms_override or spec.period_in_ms,
            )
            for variable_name in spec.variables:
                config.add_variable(variable_name)
            try:
                self._cf.log.add_config(config)
                config.data_received_cb.add_callback(self._record_sample)
                config.start()
            except Exception as exc:
                cleanup_failures = self._cleanup_config(config)
                if spec.required:
                    self.stop()
                    failure = f"required_telemetry_block_failed:{spec.name}:{exc.__class__.__name__}:{exc}"
                    self._failures.append(failure)
                    self._failures.extend(cleanup_failures)
                    raise RuntimeError(failure) from exc
                self._skipped_blocks[spec.name] = f"{exc.__class__.__name__}:{exc}"
                self._failures.extend(cleanup_failures)
                continue
            self._configs.append(config)
            self._available_blocks.append(spec.name)
        self._started = True

    def stop(self) -> None:
        """Stop active log blocks and link-statistics collection."""

        cleanup_failures: list[str] = []
        for config in reversed(self._configs):
            cleanup_failures.extend(self._cleanup_config(config))
        self._configs.clear()
        cleanup_failures.extend(self._link_monitor.stop())
        self._failures.extend(cleanup_failures)
        self._started = False

    def refresh_params(self) -> None:
        """Refresh the configured param groups into the current runtime state."""

        param_snapshot: dict[str, object] = {}
        param_groups = self._param_toc_mapping()
        for group_spec in self._param_groups:
            group_mapping = param_groups.get(group_spec.group)
            if not isinstance(group_mapping, Mapping):
                if group_spec.required:
                    self._failures.append(f"required_param_group_missing:{group_spec.group}")
                continue
            for param_name in sorted((str(name) for name in group_mapping.keys())):
                full_name = f"{group_spec.group}.{param_name}"
                try:
                    value = self._cf.param.get_value(full_name)
                except Exception as exc:
                    failure = f"param_read_failed:{full_name}:{exc.__class__.__name__}:{exc}"
                    if group_spec.required:
                        self._failures.append(failure)
                    continue
                param_snapshot[full_name] = _normalize_numeric_value(value) if group_spec.group == "deck" else value
        self._param_snapshot = param_snapshot
        self._param_snapshot_monotonic_s = self._monotonic()

    def snapshot(self) -> tuple[CrazyflieTelemetrySample, ...]:
        """Return the current bounded sample ring."""

        with self._lock:
            return tuple(self._samples)

    def recent_samples(self, *, window_s: float | None = None) -> tuple[CrazyflieTelemetrySample, ...]:
        """Return recent samples, optionally bounded by receive time."""

        with self._lock:
            samples = tuple(self._samples)
        if window_s is None:
            return samples
        threshold_s = self._monotonic() - max(0.0, float(window_s))
        return tuple(
            sample
            for sample in samples
            if sample.received_monotonic_s is not None and sample.received_monotonic_s >= threshold_s
        )

    def latest_value(self, key: str) -> tuple[float | int | None, float | None]:
        """Return the latest numeric value for one log variable and its age."""

        with self._lock:
            entry = self._latest_by_key.get(key)
        if entry is None:
            return (None, None)
        value, received_monotonic_s = entry
        return (value, max(0.0, self._monotonic() - received_monotonic_s))

    def ground_distance_observation(self) -> GroundDistanceObservation:
        """Build the current downward-range observation for the hover primitive."""

        latest_value, age_s = self.latest_value("range.zrange")
        supervisor_value, supervisor_age_s = self.latest_value("supervisor.info")
        supervisor_info = int(supervisor_value) if supervisor_value is not None else None
        is_flying = None if supervisor_info is None else bool(supervisor_info & _SUPERVISOR_IS_FLYING_MASK)
        if latest_value is None:
            return GroundDistanceObservation(
                distance_m=None,
                age_s=None,
                is_flying=is_flying,
                supervisor_age_s=supervisor_age_s,
                supervisor_info=supervisor_info,
            )
        numeric = float(latest_value)
        if numeric <= 0.0 or numeric >= HOVER_RANGE_INVALID_MM:
            return GroundDistanceObservation(
                distance_m=None,
                age_s=age_s,
                is_flying=is_flying,
                supervisor_age_s=supervisor_age_s,
                supervisor_info=supervisor_info,
            )
        return GroundDistanceObservation(
            distance_m=numeric / 1000.0,
            age_s=age_s,
            is_flying=is_flying,
            supervisor_age_s=supervisor_age_s,
            supervisor_info=supervisor_info,
        )

    def link_health_observation(self) -> LinkHealthObservation:
        """Build the current cflib link-statistics observation."""

        return self._link_monitor.observation()

    def emit(
        self,
        phase: str,
        *,
        status: str,
        message: str | None = None,
        data: Mapping[str, object] | None = None,
    ) -> None:
        """Consume worker trace events into command/failsafe runtime state."""

        phase_text = str(phase)
        payload = dict(data or {})
        now_s = self._monotonic()
        if phase_text in {"run_hover_test", "run_local_inspect_mission"} and status == "begin":
            self._command_state.update(
                {
                    "mission_name": phase_text,
                    "phase": "idle",
                    "phase_status": None,
                    "updated_monotonic_s": now_s,
                    "target_height_m": None,
                    "hover_duration_s": None,
                    "forward_m": None,
                    "left_m": None,
                    "translation_velocity_mps": None,
                    "takeoff_confirmed": False,
                    "aborted": False,
                    "abort_reason": None,
                    "last_message": message,
                    "stability_guard_codes": (),
                    "stability_guard_failures": (),
                    "stability_guard_phase": None,
                    "stability_guard_status": None,
                    "touchdown_confirmation_source": None,
                    "touchdown_distance_m": None,
                    "touchdown_supervisor_grounded": False,
                }
            )
            self._failsafe_trace_state["pre_liftoff_trigger_count"] = 0
        elif phase_text == "hover_primitive_takeoff":
            self._command_state.update(
                {
                    "phase": "takeoff",
                    "phase_status": str(status),
                    "updated_monotonic_s": now_s,
                    "target_height_m": _normalize_numeric_value(payload.get("target_height_m")),
                    "takeoff_confirmed": False,
                    "last_message": message,
                    "aborted": False,
                    "abort_reason": None,
                    "stability_guard_codes": (),
                    "stability_guard_failures": (),
                    "stability_guard_phase": None,
                    "stability_guard_status": None,
                    "touchdown_confirmation_source": None,
                    "touchdown_distance_m": None,
                    "touchdown_supervisor_grounded": False,
                }
            )
            self._failsafe_trace_state["pre_liftoff_trigger_count"] = 0
        elif phase_text == "hover_primitive_takeoff_confirm" and status == "done":
            self._command_state.update(
                {
                    "phase": "takeoff",
                    "phase_status": "confirmed",
                    "updated_monotonic_s": now_s,
                    "takeoff_confirmed": True,
                    "last_message": message,
                }
            )
        elif phase_text == "hover_primitive_stabilize":
            self._command_state.update(
                {
                    "phase": "stabilize",
                    "phase_status": str(status),
                    "updated_monotonic_s": now_s,
                    "last_message": message,
                }
            )
        elif phase_text == "hover_primitive_hold":
            self._command_state.update(
                {
                    "phase": "hold",
                    "phase_status": str(status),
                    "updated_monotonic_s": now_s,
                    "hover_duration_s": _normalize_numeric_value(payload.get("hover_duration_s")),
                    "last_message": message,
                }
            )
        elif phase_text == "hover_primitive_stability_guard":
            self._command_state.update(
                {
                    "phase": cast(str | None, payload.get("phase")) or cast(str | None, self._command_state.get("phase")) or "hold",
                    "phase_status": str(status),
                    "updated_monotonic_s": now_s,
                    "last_message": message,
                    "stability_guard_codes": _as_string_tuple(payload.get("failure_codes")),
                    "stability_guard_failures": _as_string_tuple(payload.get("failures")),
                    "stability_guard_phase": cast(str | None, payload.get("phase")),
                    "stability_guard_status": str(status),
                }
            )
        elif phase_text == "hover_primitive_translate":
            self._command_state.update(
                {
                    "phase": "translate",
                    "phase_status": str(status),
                    "updated_monotonic_s": now_s,
                    "forward_m": _normalize_numeric_value(payload.get("forward_m")),
                    "left_m": _normalize_numeric_value(payload.get("left_m")),
                    "translation_velocity_mps": _normalize_numeric_value(payload.get("velocity_mps")),
                    "target_height_m": _normalize_numeric_value(payload.get("target_height_m")),
                    "last_message": message,
                }
            )
        elif phase_text == "hover_primitive_touchdown_confirm":
            confirmation_source = cast(str | None, payload.get("confirmation_source"))
            is_flying = payload.get("is_flying")
            self._command_state.update(
                {
                    "phase": "land",
                    "phase_status": str(status),
                    "updated_monotonic_s": now_s,
                    "last_message": message,
                    "touchdown_confirmation_source": confirmation_source,
                    "touchdown_distance_m": _normalize_numeric_value(
                        payload.get("distance_m", payload.get("last_distance_m"))
                    ),
                    "touchdown_supervisor_grounded": is_flying is False,
                }
            )
        elif phase_text == "hover_primitive_land":
            self._command_state.update(
                {
                    "phase": "land",
                    "phase_status": str(status),
                    "updated_monotonic_s": now_s,
                    "last_message": message,
                }
            )
        elif phase_text == "hover_primitive_abort":
            self._command_state.update(
                {
                    "phase": "abort_landing",
                    "phase_status": str(status),
                    "updated_monotonic_s": now_s,
                    "aborted": True,
                    "abort_reason": message,
                    "last_message": message,
                }
            )
        elif phase_text == "on_device_failsafe_status" and status == "done":
            state_code = _normalize_numeric_value(payload.get("state"))
            reason_code = _normalize_numeric_value(payload.get("reason"))
            pre_liftoff_trigger_count = _as_int(self._failsafe_trace_state.get("pre_liftoff_trigger_count")) or 0
            if (
                state_code == 2
                and reason_code in {4, 5}
                and not bool(self._command_state.get("takeoff_confirmed"))
            ):
                pre_liftoff_trigger_count += 1
            self._failsafe_trace_state.update(
                {
                    "state_code": state_code,
                    "state_name": payload.get("state_name"),
                    "reason_code": reason_code,
                    "reason_name": payload.get("reason_name"),
                    "session_id": _normalize_numeric_value(payload.get("session_id")),
                    "last_status_received_monotonic_s": now_s,
                    "down_range_mm": _normalize_numeric_value(payload.get("down_range_mm")),
                    "pre_liftoff_trigger_count": pre_liftoff_trigger_count,
                }
            )

    def latest_snapshot(self) -> TelemetrySnapshot:
        """Return the latest bounded telemetry snapshot."""

        deck = self._build_deck_state()
        power = self._build_power_state()
        range_state = self._build_range_state()
        flight = self._build_flight_state()
        link = self._build_link_state()
        failsafe = self._build_failsafe_state()
        command = self._build_command_state()
        freshness = {signal: self.latest_value(signal)[1] for signal in self._required_signals}
        failures = list(self._failures)
        for signal_name, max_age_s in self._required_signals.items():
            value, age_s = self.latest_value(signal_name)
            if value is None:
                failures.append(f"required_signal_missing:{signal_name}")
                continue
            if age_s is None:
                failures.append(f"required_signal_age_missing:{signal_name}")
                continue
            if age_s > float(max_age_s):
                failures.append(f"required_signal_stale:{signal_name}:{age_s:.3f}s")
        if self._link_monitor.failure is not None:
            failures.append(self._link_monitor.failure)
        divergences = self._build_divergences(
            power=power,
            range_state=range_state,
            flight=flight,
            link=link,
            command=command,
        )
        healthy = not failures and not any((event.severity == "error" for event in divergences))
        return TelemetrySnapshot(
            profile=self.profile.value,
            collected_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            healthy=healthy,
            failures=tuple(dict.fromkeys(failures)),
            freshness_by_signal=freshness,
            catalog=self._catalog,
            deck=deck,
            power=power,
            range=range_state,
            flight=flight,
            link=link,
            failsafe=failsafe,
            command=command,
            divergences=tuple(divergences),
            available_blocks=self.available_blocks,
            skipped_blocks=self.skipped_blocks,
        )

    def _snapshot_catalog(self) -> TelemetryCatalogSummary:
        log_groups = self._toc_mapping(getattr(getattr(self._cf, "log", None), "toc", None))
        param_groups = self._toc_mapping(getattr(getattr(self._cf, "param", None), "toc", None))
        return TelemetryCatalogSummary(
            log_group_count=len(log_groups),
            log_variable_count=sum((len(group) for group in log_groups.values() if isinstance(group, Mapping))),
            param_group_count=len(param_groups),
            param_count=sum((len(group) for group in param_groups.values() if isinstance(group, Mapping))),
        )

    def _param_toc_mapping(self) -> Mapping[str, object]:
        return self._toc_mapping(getattr(getattr(self._cf, "param", None), "toc", None))

    @staticmethod
    def _toc_mapping(raw_toc: object) -> Mapping[str, object]:
        toc = getattr(raw_toc, "toc", None)
        return toc if isinstance(toc, Mapping) else {}

    def _record_sample(self, timestamp_ms: int, data: Mapping[str, object], log_block: Any) -> None:
        normalized_values = {
            str(key): _normalize_numeric_value(value) for key, value in dict(data).items()
        }
        received_monotonic_s = self._monotonic()
        sample = CrazyflieTelemetrySample(
            timestamp_ms=int(timestamp_ms),
            block_name=str(getattr(log_block, "name", "telemetry")),
            values=normalized_values,
            received_monotonic_s=received_monotonic_s,
        )
        with self._lock:
            self._samples.append(sample)
            for key, value in normalized_values.items():
                self._latest_by_key[key] = (value, received_monotonic_s)

    def _cleanup_config(self, config: Any) -> list[str]:
        failures: list[str] = []
        try:
            config.stop()
        except Exception as exc:  # pragma: no cover - live cflib cleanup
            failures.append(f"log_config_stop:{getattr(config, 'name', 'unknown')}:{exc.__class__.__name__}:{exc}")
        try:
            config.delete()
        except Exception as exc:  # pragma: no cover - live cflib cleanup
            failures.append(
                f"log_config_delete:{getattr(config, 'name', 'unknown')}:{exc.__class__.__name__}:{exc}"
            )
        try:
            config.data_received_cb.remove_callback(self._record_sample)
        except Exception as exc:  # pragma: no cover - live cflib cleanup
            failures.append(
                f"log_callback_remove:{getattr(config, 'name', 'unknown')}:{exc.__class__.__name__}:{exc}"
            )
        return failures

    def _param_value(self, key: str) -> object | None:
        return self._param_snapshot.get(key)

    def _build_deck_state(self) -> DeckTelemetryState:
        flags = {
            key.split(".", 1)[1]: cast(int | None, _normalize_numeric_value(value))
            for key, value in self._param_snapshot.items()
            if key.startswith("deck.")
        }
        age_s = (
            None
            if self._param_snapshot_monotonic_s is None
            else max(0.0, self._monotonic() - self._param_snapshot_monotonic_s)
        )
        return DeckTelemetryState(flags=flags, refreshed_age_s=age_s)

    def _build_power_state(self) -> PowerTelemetryState:
        vbat_raw, _age = self.latest_value("pm.vbat")
        battery_level_raw, _age = self.latest_value("pm.batteryLevel")
        state_raw, _age = self.latest_value("pm.state")
        state = int(state_raw) if state_raw is not None else None
        return PowerTelemetryState(
            vbat_v=float(vbat_raw) if vbat_raw is not None else None,
            battery_level=int(battery_level_raw) if battery_level_raw is not None else None,
            state=state,
            state_name=_power_state_name(state),
        )

    def _build_range_state(self) -> RangeTelemetryState:
        return RangeTelemetryState(
            zrange_m=_normalize_range_mm_to_m(self.latest_value("range.zrange")[0]),
            front_m=_normalize_range_mm_to_m(self.latest_value("range.front")[0]),
            back_m=_normalize_range_mm_to_m(self.latest_value("range.back")[0]),
            left_m=_normalize_range_mm_to_m(self.latest_value("range.left")[0]),
            right_m=_normalize_range_mm_to_m(self.latest_value("range.right")[0]),
            up_m=_normalize_range_mm_to_m(self.latest_value("range.up")[0]),
            downward_observed=_normalize_range_mm_to_m(self.latest_value("range.zrange")[0]) is not None,
        )

    def _build_flight_state(self) -> FlightTelemetryState:
        supervisor_raw, _age = self.latest_value("supervisor.info")
        supervisor_info = int(supervisor_raw) if supervisor_raw is not None else None
        return FlightTelemetryState(
            roll_deg=_as_float(self.latest_value("stabilizer.roll")[0]),
            pitch_deg=_as_float(self.latest_value("stabilizer.pitch")[0]),
            yaw_deg=_as_float(self.latest_value("stabilizer.yaw")[0]),
            x_m=_as_float(self.latest_value("stateEstimate.x")[0]),
            y_m=_as_float(self.latest_value("stateEstimate.y")[0]),
            z_m=_as_float(self.latest_value("stateEstimate.z")[0]),
            vx_mps=_as_float(self.latest_value("stateEstimate.vx")[0]),
            vy_mps=_as_float(self.latest_value("stateEstimate.vy")[0]),
            vz_mps=_as_float(self.latest_value("stateEstimate.vz")[0]),
            thrust=_as_float(self.latest_value("stabilizer.thrust")[0]),
            motion_squal=_as_int(self.latest_value("motion.squal")[0]),
            supervisor_info=supervisor_info,
            can_fly=None if supervisor_info is None else bool(supervisor_info & _SUPERVISOR_CAN_FLY_MASK),
            is_flying=None if supervisor_info is None else bool(supervisor_info & _SUPERVISOR_IS_FLYING_MASK),
            unsafe_supervisor_flags=_supervisor_flag_names(supervisor_info),
        )

    def _build_link_state(self) -> LinkTelemetryState:
        radio_connected_raw, age_s = self.latest_value("radio.isConnected")
        radio_connected = None if radio_connected_raw is None else bool(int(radio_connected_raw))
        rssi = _as_float(self.latest_value("radio.rssi")[0])
        observation = self._link_monitor.observation()
        return LinkTelemetryState(
            radio_connected=radio_connected,
            rssi_dbm=rssi,
            observation_age_s=age_s,
            latency_ms=observation.latency_ms,
            link_quality=observation.link_quality,
            uplink_rssi=observation.uplink_rssi,
            uplink_rate_hz=observation.uplink_rate_hz,
            downlink_rate_hz=observation.downlink_rate_hz,
            uplink_congestion=observation.uplink_congestion,
            downlink_congestion=observation.downlink_congestion,
            monitor_available=self._link_monitor.available,
            monitor_failure=self._link_monitor.failure,
        )

    def _build_failsafe_state(self) -> FailsafeTelemetryState:
        protocol_version = _as_int(_normalize_numeric_value(self._param_value("twinrFs.protocolVersion")))
        enabled_raw = _normalize_numeric_value(self._param_value("twinrFs.enable"))
        enabled = None if enabled_raw is None else bool(int(enabled_raw))
        state_code = _as_int(self.latest_value("twinrFs.state")[0])
        reason_code = _as_int(self.latest_value("twinrFs.reason")[0])
        if state_code is None:
            state_code = _as_int(self._failsafe_trace_state.get("state_code"))
        if reason_code is None:
            reason_code = _as_int(self._failsafe_trace_state.get("reason_code"))
        trace_received_at = cast(float | None, self._failsafe_trace_state.get("last_status_received_monotonic_s"))
        last_status_received_age_s = None if trace_received_at is None else max(0.0, self._monotonic() - trace_received_at)
        return FailsafeTelemetryState(
            loaded=protocol_version is not None or state_code is not None,
            protocol_version=protocol_version,
            enabled=enabled,
            state_code=state_code,
            state_name=cast(str | None, self._failsafe_trace_state.get("state_name")),
            reason_code=reason_code,
            reason_name=cast(str | None, self._failsafe_trace_state.get("reason_name")),
            heartbeat_age_ms=_as_int(self.latest_value("twinrFs.heartbeatAgeMs")[0]),
            last_status_received_age_s=last_status_received_age_s,
            session_id=_as_int(self._failsafe_trace_state.get("session_id")),
            rejected_packets=_as_int(self.latest_value("twinrFs.rejectedPkts")[0]),
            last_reject_code=_as_int(self.latest_value("twinrFs.lastRejectCode")[0]),
            takeoff_debug_flags=_as_int(self.latest_value("twinrFs.tkDbg")[0]),
            takeoff_range_fresh_count=_as_int(self.latest_value("twinrFs.tkRfCnt")[0]),
            takeoff_range_rise_count=_as_int(self.latest_value("twinrFs.tkRrCnt")[0]),
            takeoff_flow_live_count=_as_int(self.latest_value("twinrFs.tkFlCnt")[0]),
            takeoff_attitude_quiet_count=_as_int(self.latest_value("twinrFs.tkAtCnt")[0]),
            takeoff_truth_stale_count=_as_int(self.latest_value("twinrFs.tkStCnt")[0]),
            takeoff_truth_flap_count=_as_int(self.latest_value("twinrFs.tkFpCnt")[0]),
            takeoff_progress_class=_as_int(self.latest_value("twinrFs.tkPgCls")[0]),
            filtered_battery_mv=_as_float(self.latest_value("twinrFs.tkBatMv")[0]),
            hover_thrust_estimate=_as_float(self.latest_value("twinrFs.thrEst")[0]),
            commanded_vx_mps=_as_float(self.latest_value("twinrFs.cmdVx")[0]),
            commanded_vy_mps=_as_float(self.latest_value("twinrFs.cmdVy")[0]),
            lateral_command_source_code=_as_int(self.latest_value("twinrFs.cmdSrc")[0]),
            disturbance_estimate_vx=_as_float(self.latest_value("twinrFs.distVx")[0]),
            disturbance_estimate_vy=_as_float(self.latest_value("twinrFs.distVy")[0]),
            disturbance_severity_permille=_as_int(self.latest_value("twinrFs.distSev")[0]),
            disturbance_recoverable=_as_optional_bool(self.latest_value("twinrFs.distRec")[0]),
        )

    def _build_command_state(self) -> CommandTelemetryState:
        updated_monotonic_s = cast(float | None, self._command_state.get("updated_monotonic_s"))
        age_s = None if updated_monotonic_s is None else max(0.0, self._monotonic() - updated_monotonic_s)
        return CommandTelemetryState(
            mission_name=cast(str | None, self._command_state.get("mission_name")),
            phase=str(self._command_state.get("phase") or "idle"),
            phase_status=cast(str | None, self._command_state.get("phase_status")),
            age_s=age_s,
            target_height_m=_as_float(self._command_state.get("target_height_m")),
            hover_duration_s=_as_float(self._command_state.get("hover_duration_s")),
            forward_m=_as_float(self._command_state.get("forward_m")),
            left_m=_as_float(self._command_state.get("left_m")),
            translation_velocity_mps=_as_float(self._command_state.get("translation_velocity_mps")),
            takeoff_confirmed=bool(self._command_state.get("takeoff_confirmed")),
            aborted=bool(self._command_state.get("aborted")),
            abort_reason=cast(str | None, self._command_state.get("abort_reason")),
            last_message=cast(str | None, self._command_state.get("last_message")),
        )

    def _build_divergences(
        self,
        *,
        power: PowerTelemetryState,
        range_state: RangeTelemetryState,
        flight: FlightTelemetryState,
        link: LinkTelemetryState,
        command: CommandTelemetryState,
    ) -> list[TelemetryDivergenceEvent]:
        divergences: list[TelemetryDivergenceEvent] = []
        if command.phase in {"takeoff", "hold", "translate"} and command.target_height_m is not None:
            if (
                not command.takeoff_confirmed
                and command.age_s is not None
                and command.age_s >= _TELEMETRY_TAKEOFF_FAILURE_TIMEOUT_S
            ):
                observed_height_m = range_state.zrange_m if range_state.zrange_m is not None else flight.z_m
                required_height_m = max(
                    _TELEMETRY_TAKEOFF_MIN_CONFIRM_HEIGHT_M,
                    min(float(command.target_height_m) * 0.5, float(command.target_height_m)),
                )
                if observed_height_m is None or observed_height_m < required_height_m:
                    divergences.append(
                        TelemetryDivergenceEvent(
                            code="takeoff_not_achieved",
                            severity="error",
                            message=(
                                f"commanded takeoff to {float(command.target_height_m):.2f} m but observed height "
                                f"{observed_height_m if observed_height_m is not None else 'none'} stayed below "
                                f"{required_height_m:.2f} m"
                            ),
                        )
                    )
        pre_liftoff_trigger_count = _as_int(self._failsafe_trace_state.get("pre_liftoff_trigger_count")) or 0
        if pre_liftoff_trigger_count > 0:
            down_range_mm = _as_int(self._failsafe_trace_state.get("down_range_mm"))
            failsafe_state_name = cast(str | None, self._failsafe_trace_state.get("state_name"))
            failsafe_reason_name = cast(str | None, self._failsafe_trace_state.get("reason_name"))
            divergences.append(
                TelemetryDivergenceEvent(
                    code="failsafe_triggered_before_liftoff",
                    severity="error",
                    message=(
                        "on-device failsafe entered "
                        f"{failsafe_state_name or 'failsafe'} for {failsafe_reason_name or 'unknown'} before takeoff "
                        f"was confirmed; last down-range was {down_range_mm if down_range_mm is not None else 'unknown'} mm"
                    ),
                )
            )
        if pre_liftoff_trigger_count > 1:
            divergences.append(
                TelemetryDivergenceEvent(
                    code="failsafe_retrigger_loop",
                    severity="error",
                    message=(
                        "on-device failsafe re-entered before liftoff more than once in the same session, "
                        "which indicates a repeated ground abort loop"
                    ),
                )
            )
        if pre_liftoff_trigger_count > 0 and command.takeoff_confirmed:
            divergences.append(
                TelemetryDivergenceEvent(
                    code="takeoff_conflict_with_failsafe",
                    severity="error",
                    message=(
                        "host takeoff was confirmed after the on-device failsafe had already triggered before liftoff, "
                        "which indicates conflicting ascent control"
                    ),
                )
            )
        stability_guard_codes = _as_string_tuple(self._command_state.get("stability_guard_codes"))
        stability_guard_failures = _as_string_tuple(self._command_state.get("stability_guard_failures"))
        stability_guard_status = cast(str | None, self._command_state.get("stability_guard_status"))
        guard_severity = "error" if stability_guard_status == "blocked" else "warning"
        if "roll_pitch" in stability_guard_codes:
            divergences.append(
                TelemetryDivergenceEvent(
                    code="hover_guard_roll_pitch_exceeded",
                    severity=guard_severity,
                    message="hover stability guard observed excessive roll/pitch: " + "; ".join(stability_guard_failures),
                )
            )
        if "speed" in stability_guard_codes:
            divergences.append(
                TelemetryDivergenceEvent(
                    code="hover_guard_speed_exceeded",
                    severity=guard_severity,
                    message="hover stability guard observed excessive horizontal speed: " + "; ".join(stability_guard_failures),
                )
            )
        if "height_not_held" in stability_guard_codes:
            divergences.append(
                TelemetryDivergenceEvent(
                    code="hover_guard_height_not_held",
                    severity=guard_severity,
                    message="hover stability guard observed height/supervisor divergence: " + "; ".join(stability_guard_failures),
                )
            )
        if "height_untrusted" in stability_guard_codes:
            divergences.append(
                TelemetryDivergenceEvent(
                    code="hover_guard_height_untrusted",
                    severity=guard_severity,
                    message="hover stability guard observed untrusted height telemetry: " + "; ".join(stability_guard_failures),
                )
            )
        if "flow_untrusted" in stability_guard_codes:
            divergences.append(
                TelemetryDivergenceEvent(
                    code="hover_guard_flow_untrusted",
                    severity=guard_severity,
                    message="hover stability guard observed untrusted optical-flow telemetry: " + "; ".join(stability_guard_failures),
                )
            )
        if "anchor_control" in stability_guard_codes:
            divergences.append(
                TelemetryDivergenceEvent(
                    code="hover_guard_anchor_control_unavailable",
                    severity=guard_severity,
                    message="hover stability guard could not compute a bounded anchor-hold command: " + "; ".join(stability_guard_failures),
                )
            )
        if "xy_drift" in stability_guard_codes:
            divergences.append(
                TelemetryDivergenceEvent(
                    code="hover_guard_xy_drift_exceeded",
                    severity=guard_severity,
                    message="hover stability guard observed excessive xy drift: " + "; ".join(stability_guard_failures),
                )
            )
        touchdown_confirmation_source = cast(str | None, self._command_state.get("touchdown_confirmation_source"))
        if touchdown_confirmation_source == "timeout_forced_cutoff":
            divergences.append(
                TelemetryDivergenceEvent(
                    code="touchdown_not_confirmed",
                    severity="error",
                    message="landing had to force final motor cutoff because touchdown never reached a joint range+supervisor confirmation",
                )
            )
        if (
            touchdown_confirmation_source == "range_only"
            or (
                touchdown_confirmation_source is not None
                and touchdown_confirmation_source not in {"range+supervisor", "range_only_sitl"}
                and not bool(self._command_state.get("touchdown_supervisor_grounded"))
            )
        ):
            divergences.append(
                TelemetryDivergenceEvent(
                    code="touchdown_range_only_without_supervisor_ground",
                    severity="error",
                    message=(
                        "landing completion relied on touchdown evidence without a fresh supervisor-grounded confirmation"
                    ),
                )
            )
        if command.phase in {"takeoff", "hold", "translate", "land"} and link.radio_connected is False:
            divergences.append(
                TelemetryDivergenceEvent(
                    code="radio_drop_during_flight",
                    severity="error",
                    message="firmware reports the radio link disconnected during an active flight phase",
                )
            )
        if range_state.zrange_m is None:
            divergences.append(
                TelemetryDivergenceEvent(
                    code="range_stream_missing",
                    severity="warning",
                    message="downward z-range is currently unavailable",
                )
            )
        if power.vbat_v is None:
            divergences.append(
                TelemetryDivergenceEvent(
                    code="battery_stream_missing",
                    severity="warning",
                    message="battery voltage telemetry is currently unavailable",
                )
            )
        return divergences


def _as_float(raw: object) -> float | None:
    numeric = _normalize_numeric_value(raw)
    return None if numeric is None else float(numeric)


def _as_int(raw: object) -> int | None:
    numeric = _normalize_numeric_value(raw)
    return None if numeric is None else int(numeric)


def _as_optional_bool(raw: object) -> bool | None:
    numeric = _normalize_numeric_value(raw)
    if numeric is None:
        return None
    return bool(int(numeric))


def _as_string_tuple(raw: object) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, (list, tuple, set, frozenset)):
        return tuple((str(item) for item in raw))
    return ()


def snapshot_to_payload(snapshot: TelemetrySnapshot) -> dict[str, object]:
    """Convert one telemetry snapshot into a JSON-safe payload."""

    return asdict(snapshot)
