#!/usr/bin/env python3
# CHANGELOG: 2026-03-27
# BUG-1: The script claimed to be a bounded indoor hover primitive but silently clamped user input and had no upper bounds on height/duration/velocity; a typo could therefore trigger a materially different flight than requested.
# BUG-2: The success path unconditionally reported landed=True and completed=True when stability checks passed, even if the hover primitive outcome did not actually confirm takeoff/landing.
# BUG-3: Invalid --require-deck values escaped before the main exception handler, causing an uncaught traceback instead of a normal operator-facing failure.
# SEC-1: --trace-file accepted arbitrary paths and followed the final path component blindly; on a Raspberry Pi service account this can be abused to append to attacker-chosen files via symlink/special-file tricks.
# IMP-1: Replace blocking one-shot SyncLogger reads with bounded callback-based log sampling and add preflight gating on supervisor.info, radio.isConnected, and pm.state.
# IMP-2: Add graceful SIGTERM/SIGHUP aborts, adaptive telemetry sample budgeting, richer radio/supervisor evidence, and stricter argument validation for a truly bounded acceptance primitive.
# BREAKING: Numeric CLI inputs are now validated and rejected instead of being silently clamped.
# BREAKING: The bounded hover primitive now enforces maximum indoor-safety envelopes for height, hover duration, and vertical velocities.
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "cflib>=0.1.31",
# ]
# ///
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import signal
import stat
import sys
import threading
import time
from types import FrameType
from typing import Any, Iterable, Mapping, TypedDict, cast
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from hover_primitive import HoverEstimatorSettlingConfig, HoverEstimatorSettlingReport, HoverGroundDistanceObservation, HoverPreArmConfig, HoverPreArmSnapshot, HoverPrimitiveConfig, HoverPrimitiveOutcome, StatefulHoverPrimitive, apply_hover_pre_arm, wait_for_estimator_settle  # noqa: E402
from on_device_failsafe import ON_DEVICE_FAILSAFE_CRITICAL_BATTERY_V, ON_DEVICE_FAILSAFE_HEARTBEAT_TIMEOUT_S, ON_DEVICE_FAILSAFE_LOW_BATTERY_V, ON_DEVICE_FAILSAFE_MIN_UP_CLEARANCE_M, OnDeviceFailsafeConfig, OnDeviceFailsafeHeartbeatSession, OnDeviceFailsafeSessionReport, probe_on_device_failsafe  # noqa: E402
DEFAULT_URI = 'radio://0/80/2M'
DEFAULT_REQUIRED_DECKS = ('bcFlow2', 'bcZRanger2')
DECK_NAME_ALIASES = {'flow2': 'bcFlow2', 'flow': 'bcFlow2', 'bcflow2': 'bcFlow2', 'zranger2': 'bcZRanger2', 'zranger': 'bcZRanger2', 'bczranger2': 'bcZRanger2', 'multiranger': 'bcMultiranger', 'multi-ranger': 'bcMultiranger', 'bcmultiranger': 'bcMultiranger', 'aideck': 'bcAI', 'ai': 'bcAI', 'bcai': 'bcAI'}
DECK_PARAM_NAMES = ('bcMultiranger', 'bcFlow2', 'bcZRanger2', 'bcAI')
HOVER_TELEMETRY_PERIOD_MS = 100
HOVER_TELEMETRY_STARTUP_SETTLE_S = 0.2
HOVER_TELEMETRY_MAX_SAMPLES = 400
HOVER_TELEMETRY_MAX_SAMPLES_CAP = 5000
HOVER_TELEMETRY_AIRBORNE_MIN_ALTITUDE_M = 0.02
HOVER_TELEMETRY_WINDOW_PADDING_MS = HOVER_TELEMETRY_PERIOD_MS
HOVER_RANGE_INVALID_MM = 32000.0
HOVER_PREFLIGHT_CLEARANCE_SAMPLE_S = 0.4
HOVER_PREFLIGHT_CLEARANCE_PERIOD_S = 0.1
HOVER_PREFLIGHT_LOG_TIMEOUT_S = 2.0
HOVER_DEFAULT_MIN_CLEARANCE_M = 0.35
HOVER_STABILITY_ALTITUDE_MARGIN_M = 0.25
HOVER_STABILITY_MIN_UNDER_LOAD_VBAT_V = 3.5
HOVER_DEFAULT_ESTIMATOR = 2
HOVER_DEFAULT_CONTROLLER = 1
HOVER_DEFAULT_MOTION_DISABLE = 0
HOVER_DEFAULT_ON_DEVICE_FAILSAFE_MODE = 'required'
HOVER_MIN_HEIGHT_M = 0.1
HOVER_MAX_HEIGHT_M = 0.5
HOVER_MIN_DURATION_S = 1.0
HOVER_MAX_DURATION_S = 10.0
HOVER_MIN_VERTICAL_VELOCITY_MPS = 0.05
HOVER_MAX_VERTICAL_VELOCITY_MPS = 0.5
HOVER_MAX_CONNECT_SETTLE_S = 5.0
HOVER_MAX_ESTIMATOR_SETTLE_TIMEOUT_S = 20.0
HOVER_MAX_CLEARANCE_M = 4.0
HOVER_MAX_MIN_BATTERY_LEVEL = 100
HOVER_MAX_BATTERY_V = 5.0
HOVER_MAX_FAILSAFE_HEARTBEAT_TIMEOUT_S = 5.0
HOVER_PRE_FLIGHT_SLACK_S = 2.0
POWER_STATE_NAMES = {0: 'battery', 1: 'charging', 2: 'charged', 3: 'low_power', 4: 'shutdown'}
_SUPERVISOR_CAN_ARM_MASK = 1 << 0
_SUPERVISOR_IS_ARMED_MASK = 1 << 1
_SUPERVISOR_AUTO_ARM_MASK = 1 << 2
_SUPERVISOR_CAN_FLY_MASK = 1 << 3
_SUPERVISOR_IS_FLYING_MASK = 1 << 4
_SUPERVISOR_TUMBLED_MASK = 1 << 5
_SUPERVISOR_LOCKED_MASK = 1 << 6
_SUPERVISOR_CRASHED_MASK = 1 << 7
_SUPERVISOR_HL_FLYING_MASK = 1 << 8
_SUPERVISOR_HL_TRAJECTORY_FINISHED_MASK = 1 << 9
_SUPERVISOR_HL_DISABLED_MASK = 1 << 10
_SUPERVISOR_FLAG_BITS = (('tumbled', _SUPERVISOR_TUMBLED_MASK), ('locked', _SUPERVISOR_LOCKED_MASK), ('crashed', _SUPERVISOR_CRASHED_MASK))

@dataclass(frozen=True)
class HoverPowerSnapshot:
    vbat_v: float | None
    battery_level: int | None
    state: int | None

@dataclass(frozen=True)
class HoverStatusSnapshot:
    supervisor_info: int | None
    can_arm: bool | None
    is_armed: bool | None
    auto_arm: bool | None
    can_fly: bool | None
    is_flying: bool | None
    tumbled: bool | None
    locked: bool | None
    crashed: bool | None
    hl_flying: bool | None
    hl_trajectory_finished: bool | None
    hl_disabled: bool | None
    radio_connected: bool | None
    zrange_m: float | None
    motion_squal: int | None

@dataclass(frozen=True)
class HoverClearanceSnapshot:
    front_m: float | None
    back_m: float | None
    left_m: float | None
    right_m: float | None
    up_m: float | None
    down_m: float | None

@dataclass(frozen=True)
class HoverWorkerTraceEvent:
    index: int
    pid: int
    ts_utc: str
    elapsed_s: float
    phase: str
    status: str
    message: str | None = None
    data: dict[str, object] | None = None


class HoverRuntimeConfig(TypedDict):
    uri: str
    workspace: Path
    height_m: float
    hover_duration_s: float
    takeoff_velocity_mps: float
    land_velocity_mps: float
    connect_settle_s: float
    min_vbat_v: float
    min_battery_level: int
    min_clearance_m: float
    stabilizer_estimator: int
    stabilizer_controller: int
    motion_disable: int
    estimator_settle_timeout_s: float
    on_device_failsafe_mode: str
    on_device_failsafe_heartbeat_timeout_s: float
    on_device_failsafe_low_battery_v: float
    on_device_failsafe_critical_battery_v: float
    on_device_failsafe_min_up_clearance_m: float
    required_decks: tuple[str, ...]


class HoverWorkerTraceWriter:

    def __init__(self, path: Path | None) -> None:
        self.path = self._normalize_path(path)
        self._lock = threading.Lock()
        self._start_monotonic = time.monotonic()
        self._next_index = 0
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._probe_path()

    @staticmethod
    def _normalize_path(path: Path | None) -> Path | None:
        if path is None:
            return None
        candidate = path.expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        return candidate

    def _probe_path(self) -> None:
        fd = self._open_fd()
        os.close(fd)

    def _open_fd(self) -> int:
        if self.path is None:
            raise RuntimeError('trace file path is not configured')
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        nofollow_flag = getattr(os, 'O_NOFOLLOW', 0)
        fd = os.open(str(self.path), flags | nofollow_flag, 384)
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            os.close(fd)
            raise RuntimeError(f'trace path `{self.path}` must be a regular file')
        return fd

    def emit(self, phase: str, *, status: str, message: str | None=None, data: Mapping[str, object] | None=None) -> None:
        if self.path is None:
            return
        with self._lock:
            event = HoverWorkerTraceEvent(index=self._next_index, pid=os.getpid(), ts_utc=datetime.now(timezone.utc).isoformat(), elapsed_s=max(0.0, time.monotonic() - self._start_monotonic), phase=str(phase), status=str(status), message=str(message) if message is not None else None, data=dict(data) if data is not None else None)
            self._next_index += 1
            encoded = json.dumps(asdict(event), sort_keys=True, default=str)
            fd = self._open_fd()
            try:
                with os.fdopen(fd, 'a', encoding='utf-8') as handle:
                    handle.write(encoded)
                    handle.write('\n')
                    handle.flush()
                    try:
                        os.fsync(handle.fileno())
                    except OSError:
                        pass
            except Exception:
                try:
                    os.close(fd)
                except OSError:
                    pass
                raise

class GracefulSignalAbortContext:

    def __init__(self, trace_writer: HoverWorkerTraceWriter | None=None) -> None:
        self._trace_writer = trace_writer
        self._previous_handlers: dict[int, Any] = {}
        self.last_signal_name: str | None = None
        self._signals = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, 'SIGHUP'):
            self._signals.append(signal.SIGHUP)

    def __enter__(self) -> GracefulSignalAbortContext:
        for signum in self._signals:
            self._previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle_signal)
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        for signum, previous in self._previous_handlers.items():
            signal.signal(signum, previous)
        self._previous_handlers.clear()

    def _handle_signal(self, signum: int, frame: FrameType | None) -> None:
        self.last_signal_name = signal.Signals(signum).name
        if self._trace_writer is not None:
            self._trace_writer.emit('signal', status='error', message=self.last_signal_name, data={'signum': signum})
        raise KeyboardInterrupt(self.last_signal_name)

@dataclass(frozen=True)
class HoverTelemetryBlockSpec:
    name: str
    variables: tuple[str, ...]
    required: bool = True
HOVER_TELEMETRY_BLOCKS = (HoverTelemetryBlockSpec(name='hover-attitude', variables=('stabilizer.roll', 'stabilizer.pitch', 'stabilizer.yaw', 'stateEstimate.x', 'stateEstimate.y', 'stateEstimate.z')), HoverTelemetryBlockSpec(name='hover-sensors', variables=('motion.squal', 'motion.deltaX', 'motion.deltaY', 'range.zrange', 'pm.vbat', 'supervisor.info', 'radio.rssi', 'radio.isConnected')), HoverTelemetryBlockSpec(name='hover-velocity', variables=('stateEstimate.vx', 'stateEstimate.vy', 'stateEstimate.vz', 'stabilizer.thrust')), HoverTelemetryBlockSpec(name='hover-gyro', variables=('gyro.x', 'gyro.y', 'gyro.z')), HoverTelemetryBlockSpec(name='hover-clearance', variables=('range.front', 'range.back', 'range.left', 'range.right', 'range.up'), required=False))

@dataclass(frozen=True)
class HoverTelemetrySample:
    timestamp_ms: int
    block_name: str
    values: dict[str, float | int | None]

@dataclass(frozen=True)
class HoverTelemetrySummary:
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

@dataclass(frozen=True)
class HoverTestReport:
    uri: str
    workspace: str
    height_m: float
    hover_duration_s: float
    takeoff_velocity_mps: float
    land_velocity_mps: float
    connect_settle_s: float
    min_vbat_v: float
    min_battery_level: int
    min_clearance_m: float
    deck_flags: dict[str, int | None]
    required_decks: tuple[str, ...]
    clearance_snapshot: HoverClearanceSnapshot | None
    status_snapshot: HoverStatusSnapshot | None
    pre_arm_snapshot: HoverPreArmSnapshot | None
    estimator_settle: HoverEstimatorSettlingReport | None
    power: HoverPowerSnapshot
    status: str
    completed: bool
    landed: bool
    interrupted: bool
    primitive_outcome: HoverPrimitiveOutcome | None
    telemetry: tuple[HoverTelemetrySample, ...]
    telemetry_summary: HoverTelemetrySummary | None
    failures: tuple[str, ...]
    recommendations: tuple[str, ...]
    on_device_failsafe: OnDeviceFailsafeSessionReport | None = None

def normalize_required_deck_name(value: str) -> str:
    normalized = str(value or '').strip().lower()
    if not normalized:
        raise ValueError('required deck names must not be empty')
    try:
        return DECK_NAME_ALIASES[normalized]
    except KeyError as exc:
        allowed = ', '.join(sorted(dict.fromkeys(DECK_NAME_ALIASES)))
        raise ValueError(f'unsupported deck name `{value}`; choose one of: {allowed}') from exc

def _power_state_name(state: int | None) -> str:
    if state is None:
        return 'unknown'
    return POWER_STATE_NAMES.get(int(state), f'unknown_{state}')

def _bool_from_bitfield(value: int | None, mask: int) -> bool | None:
    if value is None:
        return None
    return int(value) & int(mask) != 0

def _normalize_range_mm_to_m(raw: object) -> float | None:
    numeric = _normalize_numeric_value(raw)
    if numeric is None:
        return None
    value_mm = float(numeric)
    if value_mm <= 0.0 or value_mm >= HOVER_RANGE_INVALID_MM:
        return None
    return value_mm / 1000.0

def _decode_status_snapshot(sample: Mapping[str, object]) -> HoverStatusSnapshot:
    supervisor_info_raw = _normalize_numeric_value(sample.get('supervisor.info'))
    supervisor_info = int(supervisor_info_raw) if supervisor_info_raw is not None else None
    radio_connected_raw = _normalize_numeric_value(sample.get('radio.isConnected'))
    radio_connected = None if radio_connected_raw is None else bool(int(radio_connected_raw))
    motion_squal_raw = _normalize_numeric_value(sample.get('motion.squal'))
    motion_squal = int(motion_squal_raw) if motion_squal_raw is not None else None
    return HoverStatusSnapshot(supervisor_info=supervisor_info, can_arm=_bool_from_bitfield(supervisor_info, _SUPERVISOR_CAN_ARM_MASK), is_armed=_bool_from_bitfield(supervisor_info, _SUPERVISOR_IS_ARMED_MASK), auto_arm=_bool_from_bitfield(supervisor_info, _SUPERVISOR_AUTO_ARM_MASK), can_fly=_bool_from_bitfield(supervisor_info, _SUPERVISOR_CAN_FLY_MASK), is_flying=_bool_from_bitfield(supervisor_info, _SUPERVISOR_IS_FLYING_MASK), tumbled=_bool_from_bitfield(supervisor_info, _SUPERVISOR_TUMBLED_MASK), locked=_bool_from_bitfield(supervisor_info, _SUPERVISOR_LOCKED_MASK), crashed=_bool_from_bitfield(supervisor_info, _SUPERVISOR_CRASHED_MASK), hl_flying=_bool_from_bitfield(supervisor_info, _SUPERVISOR_HL_FLYING_MASK), hl_trajectory_finished=_bool_from_bitfield(supervisor_info, _SUPERVISOR_HL_TRAJECTORY_FINISHED_MASK), hl_disabled=_bool_from_bitfield(supervisor_info, _SUPERVISOR_HL_DISABLED_MASK), radio_connected=radio_connected, zrange_m=_normalize_range_mm_to_m(sample.get('range.zrange')), motion_squal=motion_squal)

def evaluate_hover_preflight(*, deck_flags: dict[str, int | None], required_decks: tuple[str, ...], power: HoverPowerSnapshot, status_snapshot: HoverStatusSnapshot | None, clearance_snapshot: HoverClearanceSnapshot | None, min_vbat_v: float, min_battery_level: int, min_clearance_m: float) -> list[str]:
    failures: list[str] = []
    for deck_name in required_decks:
        if deck_flags.get(deck_name) != 1:
            failures.append(f'required deck {deck_name} is not detected')
    if power.vbat_v is None:
        failures.append('battery voltage is unavailable')
    elif power.vbat_v < min_vbat_v:
        failures.append(f'battery voltage {power.vbat_v:.2f} V is below the {min_vbat_v:.2f} V hover gate')
    if power.battery_level is None:
        failures.append('battery level is unavailable')
    elif power.battery_level < min_battery_level:
        failures.append(f'battery level {power.battery_level}% is below the {min_battery_level}% hover gate')
    if power.state is None:
        failures.append('battery power state is unavailable')
    elif power.state != 0:
        failures.append(f'battery power state is `{_power_state_name(power.state)}` ({power.state}) instead of `battery` (0)')
    if status_snapshot is None:
        failures.append('supervisor status is unavailable')
    else:
        if status_snapshot.supervisor_info is None:
            failures.append('supervisor.info is unavailable')
        if status_snapshot.radio_connected is False:
            failures.append('radio link is not marked connected by the firmware')
        elif status_snapshot.radio_connected is None:
            failures.append('radio connection state is unavailable')
        if status_snapshot.can_fly is False:
            failures.append('supervisor reports the Crazyflie is not ready to fly')
        if status_snapshot.tumbled:
            failures.append('supervisor reports the Crazyflie is tumbled')
        if status_snapshot.locked:
            failures.append('supervisor reports the Crazyflie is locked and must be restarted')
        if status_snapshot.crashed:
            failures.append('supervisor reports the Crazyflie is in a crashed state')
    if clearance_snapshot is not None:
        for direction_name, value in (('front', clearance_snapshot.front_m), ('back', clearance_snapshot.back_m), ('left', clearance_snapshot.left_m), ('right', clearance_snapshot.right_m), ('up', clearance_snapshot.up_m)):
            if value is not None and value < min_clearance_m:
                failures.append(f'{direction_name} clearance {value:.2f} m is below the {min_clearance_m:.2f} m hover gate')
    return failures

def recommendations_for_report(report: HoverTestReport) -> tuple[str, ...]:
    recommendations: list[str] = []
    if report.failures:
        recommendations.extend(report.failures)
    if report.status == 'blocked':
        recommendations.append('Charge the battery, confirm Flow/Z-ranger deck presence, and clear supervisor faults before retrying the hover test.')
    elif report.status == 'interrupted':
        recommendations.append('Confirm the Crazyflie is on the ground before retrying the hover test.')
    elif report.status == 'unstable':
        recommendations.append('Hover completed but acceptance gates failed; inspect the telemetry and primitive outcome before retrying.')
    elif report.status == 'completed':
        recommendations.append('Hover test completed and landed. The bounded flight primitive looks ready.')
    if report.status_snapshot is not None and report.status_snapshot.locked:
        recommendations.append('Restart the Crazyflie or perform crash recovery before the next flight attempt.')
    if report.on_device_failsafe is not None:
        if report.on_device_failsafe.mode == 'required' and (not report.on_device_failsafe.availability.loaded):
            recommendations.append('Flash or load the Twinr on-device failsafe app before the next hover attempt.')
        recommendations.extend(report.on_device_failsafe.failures)
    return tuple(dict.fromkeys(recommendations))

def _normalize_numeric_value(raw: object) -> float | int | None:
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
        if value.is_integer():
            return int(value)
        return value
    try:
        value = float(cast(Any, raw))
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    if value.is_integer():
        return int(value)
    return value

class HoverTelemetryCollector:

    def __init__(self, sync_cf, log_config_cls, *, period_in_ms: int=HOVER_TELEMETRY_PERIOD_MS, max_samples: int=HOVER_TELEMETRY_MAX_SAMPLES) -> None:
        self._cf = sync_cf.cf if hasattr(sync_cf, 'cf') else sync_cf
        self._log_config_cls = log_config_cls
        self._period_in_ms = max(10, int(period_in_ms))
        self._max_samples = max(1, int(max_samples))
        self._configs: list[Any] = []
        self._samples: list[HoverTelemetrySample] = []
        self._available_blocks: list[str] = []
        self._skipped_blocks: dict[str, str] = {}
        self._lock = threading.Lock()
        self._latest_by_key: dict[str, tuple[float | int | None, float]] = {}
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        for spec in HOVER_TELEMETRY_BLOCKS:
            config = self._log_config_cls(name=spec.name, period_in_ms=self._period_in_ms)
            for variable_name in spec.variables:
                config.add_variable(variable_name)
            try:
                self._cf.log.add_config(config)
                config.data_received_cb.add_callback(self._record_sample)
                config.start()
            except Exception as exc:
                self._cleanup_config(config)
                if spec.required:
                    self.stop()
                    raise RuntimeError(f'required_hover_telemetry_block_failed:{spec.name}:{exc.__class__.__name__}:{exc}') from exc
                self._skipped_blocks[spec.name] = f'{exc.__class__.__name__}:{exc}'
                continue
            self._configs.append(config)
            self._available_blocks.append(spec.name)
        self._started = True

    def stop(self) -> None:
        for config in reversed(self._configs):
            self._cleanup_config(config)
        self._configs.clear()
        self._started = False

    def snapshot(self) -> tuple[HoverTelemetrySample, ...]:
        with self._lock:
            return tuple(self._samples)

    def latest_value(self, key: str) -> tuple[float | int | None, float | None]:
        with self._lock:
            if key not in self._latest_by_key:
                return (None, None)
            value, received_monotonic_s = self._latest_by_key[key]
            return (value, max(0.0, time.monotonic() - received_monotonic_s))

    @property
    def available_blocks(self) -> tuple[str, ...]:
        return tuple(self._available_blocks)

    @property
    def skipped_blocks(self) -> tuple[str, ...]:
        return tuple(self._skipped_blocks)

    def _record_sample(self, timestamp_ms: int, data: Mapping[str, object], log_block: Any) -> None:
        normalized_values = {str(key): _normalize_numeric_value(value) for key, value in dict(data).items()}
        sample = HoverTelemetrySample(timestamp_ms=int(timestamp_ms), block_name=str(getattr(log_block, 'name', 'hover-log')), values=normalized_values)
        received_monotonic_s = time.monotonic()
        with self._lock:
            if len(self._samples) < self._max_samples:
                self._samples.append(sample)
            for key, value in normalized_values.items():
                self._latest_by_key[key] = (value, received_monotonic_s)

    def _cleanup_config(self, config: Any) -> None:
        try:
            config.stop()
        except Exception:
            pass
        try:
            config.delete()
        except Exception:
            pass
        try:
            config.data_received_cb.remove_callback(self._record_sample)
        except Exception:
            pass

def _series_for_key(samples: Iterable[HoverTelemetrySample], key: str, *, start_ms: int | None=None, end_ms: int | None=None) -> list[tuple[int, float]]:
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

def _values_for_key(samples: Iterable[HoverTelemetrySample], key: str, *, start_ms: int | None=None, end_ms: int | None=None) -> list[float]:
    return [value for _, value in _series_for_key(samples, key, start_ms=start_ms, end_ms=end_ms)]

def _drift(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return values[-1] - values[0]

def _span(values: list[float]) -> float | None:
    if not values:
        return None
    return max(values) - min(values)

def _valid_range_values_m(values_mm: Iterable[float]) -> list[float]:
    values_m: list[float] = []
    for value in values_mm:
        numeric = float(value)
        if numeric <= 0.0 or numeric >= HOVER_RANGE_INVALID_MM:
            continue
        values_m.append(numeric / 1000.0)
    return values_m

def _series_min(values: list[float]) -> float | None:
    if not values:
        return None
    return min(values)

def _series_max(values: list[float]) -> float | None:
    if not values:
        return None
    return max(values)

def _airborne_window_bounds_ms(samples: tuple[HoverTelemetrySample, ...]) -> tuple[int | None, int | None]:
    candidate_timestamps = [timestamp_ms for timestamp_ms, value in _series_for_key(samples, 'stateEstimate.z') if value >= HOVER_TELEMETRY_AIRBORNE_MIN_ALTITUDE_M]
    candidate_timestamps.extend((timestamp_ms for timestamp_ms, value in _series_for_key(samples, 'range.zrange') if 0.0 < value < HOVER_RANGE_INVALID_MM and value / 1000.0 >= HOVER_TELEMETRY_AIRBORNE_MIN_ALTITUDE_M))
    if not candidate_timestamps:
        return (None, None)
    return (min(candidate_timestamps), max(candidate_timestamps))

def summarize_hover_telemetry(samples: tuple[HoverTelemetrySample, ...], *, available_blocks: tuple[str, ...]=(), skipped_blocks: tuple[str, ...]=()) -> HoverTelemetrySummary:
    window_start_ms, window_end_ms = _airborne_window_bounds_ms(samples)
    filter_start_ms = None if window_start_ms is None else max(0, window_start_ms - HOVER_TELEMETRY_WINDOW_PADDING_MS)
    filter_end_ms = None if window_end_ms is None else window_end_ms + HOVER_TELEMETRY_WINDOW_PADDING_MS
    window_samples = tuple((sample for sample in samples if filter_start_ms is not None and filter_end_ms is not None and (filter_start_ms <= sample.timestamp_ms <= filter_end_ms)))
    if filter_start_ms is None or filter_end_ms is None:
        window_samples = ()
    roll_values = _values_for_key(window_samples, 'stabilizer.roll')
    pitch_values = _values_for_key(window_samples, 'stabilizer.pitch')
    x_values = _values_for_key(window_samples, 'stateEstimate.x')
    y_values = _values_for_key(window_samples, 'stateEstimate.y')
    z_values = _values_for_key(window_samples, 'stateEstimate.z')
    vx_values = _values_for_key(window_samples, 'stateEstimate.vx')
    vy_values = _values_for_key(window_samples, 'stateEstimate.vy')
    vz_values = _values_for_key(window_samples, 'stateEstimate.vz')
    squal_values = [int(value) for value in _values_for_key(window_samples, 'motion.squal')]
    delta_x_values = _values_for_key(window_samples, 'motion.deltaX')
    delta_y_values = _values_for_key(window_samples, 'motion.deltaY')
    zrange_values_m = _valid_range_values_m(_values_for_key(window_samples, 'range.zrange'))
    front_values_m = _valid_range_values_m(_values_for_key(window_samples, 'range.front'))
    back_values_m = _valid_range_values_m(_values_for_key(window_samples, 'range.back'))
    left_values_m = _valid_range_values_m(_values_for_key(window_samples, 'range.left'))
    right_values_m = _valid_range_values_m(_values_for_key(window_samples, 'range.right'))
    up_values_m = _valid_range_values_m(_values_for_key(window_samples, 'range.up'))
    thrust_values = _values_for_key(window_samples, 'stabilizer.thrust')
    gyro_x_values = _values_for_key(window_samples, 'gyro.x')
    gyro_y_values = _values_for_key(window_samples, 'gyro.y')
    gyro_z_values = _values_for_key(window_samples, 'gyro.z')
    battery_values = _values_for_key(window_samples, 'pm.vbat')
    radio_rssi_values = _values_for_key(window_samples, 'radio.rssi')
    radio_connected_values = [int(value) for value in _values_for_key(window_samples, 'radio.isConnected')]
    supervisor_values = [int(value) for value in _values_for_key(window_samples, 'supervisor.info')]
    flags_seen = tuple((flag_name for flag_name, mask in _SUPERVISOR_FLAG_BITS if any((value & mask != 0 for value in supervisor_values))))
    xy_drift_m: float | None = None
    if len(x_values) >= 2 and len(y_values) >= 2:
        xy_drift_m = math.sqrt((x_values[-1] - x_values[0]) ** 2 + (y_values[-1] - y_values[0]) ** 2)
    horizontal_speed_max_mps = None
    if vx_values and vy_values:
        horizontal_speed_max_mps = max((math.sqrt(vx_value * vx_value + vy_value * vy_value) for vx_value, vy_value in zip(vx_values, vy_values)))
    gyro_abs_max_dps = max((abs(value) for value in (*gyro_x_values, *gyro_y_values, *gyro_z_values)), default=None)
    clearance_observed = any((values for values in (front_values_m, back_values_m, left_values_m, right_values_m, up_values_m)))
    return HoverTelemetrySummary(sample_count=len(window_samples), available_blocks=available_blocks, skipped_blocks=skipped_blocks, duration_s=max(0.0, (window_end_ms - window_start_ms) / 1000.0) if window_start_ms is not None and window_end_ms is not None else None, roll_abs_max_deg=max((abs(value) for value in roll_values), default=None), pitch_abs_max_deg=max((abs(value) for value in pitch_values), default=None), xy_drift_m=xy_drift_m, z_drift_m=_drift(z_values), z_span_m=_span(z_values), vx_abs_max_mps=max((abs(value) for value in vx_values), default=None), vy_abs_max_mps=max((abs(value) for value in vy_values), default=None), vz_abs_max_mps=max((abs(value) for value in vz_values), default=None), horizontal_speed_max_mps=horizontal_speed_max_mps, flow_squal_min=min(squal_values) if squal_values else None, flow_squal_mean=sum(squal_values) / len(squal_values) if squal_values else None, flow_nonzero_samples=sum((1 for value in squal_values if value > 0)), flow_observed=any((value > 0 for value in squal_values)), motion_delta_x_abs_max=max((abs(value) for value in delta_x_values), default=None), motion_delta_y_abs_max=max((abs(value) for value in delta_y_values), default=None), zrange_min_m=min(zrange_values_m) if zrange_values_m else None, zrange_max_m=max(zrange_values_m) if zrange_values_m else None, zrange_sample_count=len(zrange_values_m), zrange_observed=any((value > 0 for value in zrange_values_m)), front_min_m=_series_min(front_values_m), back_min_m=_series_min(back_values_m), left_min_m=_series_min(left_values_m), right_min_m=_series_min(right_values_m), up_min_m=_series_min(up_values_m), clearance_observed=clearance_observed, thrust_mean=sum(thrust_values) / len(thrust_values) if thrust_values else None, thrust_max=_series_max(thrust_values), gyro_abs_max_dps=gyro_abs_max_dps, battery_min_v=min(battery_values) if battery_values else None, battery_drop_v=battery_values[0] - battery_values[-1] if len(battery_values) >= 2 else None, radio_rssi_latest_dbm=radio_rssi_values[-1] if radio_rssi_values else None, radio_rssi_min_dbm=_series_min(radio_rssi_values), radio_connected_latest=bool(radio_connected_values[-1]) if radio_connected_values else None, radio_disconnect_seen=any((value == 0 for value in radio_connected_values)), latest_supervisor_info=supervisor_values[-1] if supervisor_values else None, supervisor_flags_seen=flags_seen, stable_supervisor=not flags_seen)

def evaluate_hover_stability(summary: HoverTelemetrySummary, *, target_height_m: float) -> list[str]:
    failures: list[str] = []
    if summary.sample_count <= 0:
        failures.append('no in-flight telemetry samples were captured during the hover test')
    if not summary.flow_observed:
        failures.append('optical-flow quality never became nonzero during the hover test')
    if not summary.zrange_observed:
        failures.append('downward z-range never produced a nonzero reading during the hover test')
    if summary.supervisor_flags_seen:
        failures.append(f"supervisor reported unsafe flags during the hover test: {', '.join(summary.supervisor_flags_seen)}")
    if summary.radio_disconnect_seen:
        failures.append('firmware reported radio disconnection during the hover test')
    max_safe_altitude_m = max(HOVER_TELEMETRY_AIRBORNE_MIN_ALTITUDE_M, float(target_height_m) + HOVER_STABILITY_ALTITUDE_MARGIN_M)
    if summary.zrange_max_m is not None and summary.zrange_max_m > max_safe_altitude_m:
        failures.append(f'hover altitude reached {summary.zrange_max_m:.2f} m which exceeds the {max_safe_altitude_m:.2f} m safety ceiling for a {target_height_m:.2f} m hover')
    if summary.battery_min_v is not None and summary.battery_min_v < HOVER_STABILITY_MIN_UNDER_LOAD_VBAT_V:
        failures.append(f'battery sagged to {summary.battery_min_v:.2f} V under load which is below the {HOVER_STABILITY_MIN_UNDER_LOAD_VBAT_V:.2f} V hover safety floor')
    return failures

def _evaluate_primitive_outcome(primitive_outcome: HoverPrimitiveOutcome | None) -> list[str]:
    if primitive_outcome is None:
        return ['hover primitive did not return an outcome']
    failures: list[str] = []
    if not primitive_outcome.took_off:
        failures.append('hover primitive did not report a successful takeoff')
    if not primitive_outcome.landed:
        failures.append('hover primitive did not report a completed landing')
    return failures

def _estimate_telemetry_max_samples(*, height_m: float, hover_duration_s: float, takeoff_velocity_mps: float, land_velocity_mps: float, period_in_ms: int, block_count: int) -> int:
    takeoff_s = max(0.0, float(height_m) / max(float(takeoff_velocity_mps), 1e-06))
    landing_s = max(0.0, float(height_m) / max(float(land_velocity_mps), 1e-06))
    total_s = takeoff_s + float(hover_duration_s) + landing_s + HOVER_TELEMETRY_STARTUP_SETTLE_S + HOVER_PRE_FLIGHT_SLACK_S
    events_per_block = max(1, math.ceil(total_s * 1000.0 / max(int(period_in_ms), 10)))
    estimated = int(events_per_block * max(1, int(block_count)) * 1.2) + 32
    return max(HOVER_TELEMETRY_MAX_SAMPLES, min(estimated, HOVER_TELEMETRY_MAX_SAMPLES_CAP))

def _import_cflib():
    # pylint: disable=import-error
    import cflib.crtp
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.log import LogConfig
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
    from cflib.crazyflie.syncLogger import SyncLogger
    from cflib.utils.multiranger import Multiranger
    return (cflib.crtp, Crazyflie, LogConfig, Multiranger, SyncCrazyflie, SyncLogger)

def _read_deck_flags(sync_cf, deck_names: Iterable[str]) -> dict[str, int | None]:
    flags: dict[str, int | None] = {}
    for deck_name in deck_names:
        value: int | None
        try:
            raw = sync_cf.cf.param.get_value(f'deck.{deck_name}')
        except Exception:
            value = None
        else:
            try:
                value = int(str(raw).strip())
            except (TypeError, ValueError):
                value = None
        flags[deck_name] = value
    return flags

def _read_single_log_sample(sync_cf, log_config_cls, *, name: str, variables: tuple[str, ...], period_in_ms: int=100, timeout_s: float=HOVER_PREFLIGHT_LOG_TIMEOUT_S) -> dict[str, object]:
    config = log_config_cls(name=name, period_in_ms=max(10, int(period_in_ms)))
    for variable_name in variables:
        config.add_variable(variable_name)
    received = threading.Event()
    payload: dict[str, object] = {}
    errors: list[str] = []

    def _on_data(timestamp: int, data: Mapping[str, object], log_block: Any) -> None:
        if received.is_set():
            return
        payload.update(dict(data))
        received.set()

    def _on_error(log_conf: Any, message: str) -> None:
        errors.append(str(message))
        received.set()
    sync_cf.cf.log.add_config(config)
    config.data_received_cb.add_callback(_on_data)
    error_cb = getattr(config, 'error_cb', None)
    if error_cb is not None:
        error_cb.add_callback(_on_error)
    try:
        config.start()
        if not received.wait(max(0.1, float(timeout_s))):
            raise TimeoutError(f'{name} did not yield a log sample within {timeout_s:.2f}s')
        if errors and (not payload):
            raise RuntimeError(f'{name} failed before producing data: {errors[0]}')
        if not payload:
            raise RuntimeError(f'{name} yielded no data')
        return payload
    finally:
        try:
            config.stop()
        except Exception:
            pass
        try:
            config.delete()
        except Exception:
            pass
        try:
            config.data_received_cb.remove_callback(_on_data)
        except Exception:
            pass
        if error_cb is not None:
            try:
                error_cb.remove_callback(_on_error)
            except Exception:
                pass

def _read_preflight_snapshots(sync_cf, log_config_cls) -> tuple[HoverPowerSnapshot, HoverStatusSnapshot]:
    sample = _read_single_log_sample(sync_cf, log_config_cls, name='hover-preflight', variables=('pm.vbat', 'pm.batteryLevel', 'pm.state', 'supervisor.info', 'radio.isConnected', 'range.zrange', 'motion.squal'), period_in_ms=100, timeout_s=HOVER_PREFLIGHT_LOG_TIMEOUT_S)
    vbat_raw = _normalize_numeric_value(sample.get('pm.vbat'))
    battery_level_raw = _normalize_numeric_value(sample.get('pm.batteryLevel'))
    state_raw = _normalize_numeric_value(sample.get('pm.state'))
    power = HoverPowerSnapshot(
        vbat_v=float(vbat_raw) if vbat_raw is not None else None,
        battery_level=int(battery_level_raw) if battery_level_raw is not None else None,
        state=int(state_raw) if state_raw is not None else None,
    )
    return (power, _decode_status_snapshot(sample))

def _normalize_clearance_value(raw: object) -> float | None:
    numeric = _normalize_numeric_value(raw)
    if numeric is None:
        return None
    value = float(numeric)
    if value <= 0.0:
        return None
    return value

def _read_clearance_snapshot(sync_cf, multiranger_cls) -> HoverClearanceSnapshot:
    readings: dict[str, list[float]] = {'front': [], 'back': [], 'left': [], 'right': [], 'up': [], 'down': []}
    start = time.monotonic()
    with multiranger_cls(sync_cf, zranger=True) as multiranger:
        while time.monotonic() - start < HOVER_PREFLIGHT_CLEARANCE_SAMPLE_S:
            for direction_name in readings:
                value = _normalize_clearance_value(getattr(multiranger, direction_name))
                if value is not None:
                    readings[direction_name].append(value)
            time.sleep(HOVER_PREFLIGHT_CLEARANCE_PERIOD_S)
    return HoverClearanceSnapshot(front_m=min(readings['front']) if readings['front'] else None, back_m=min(readings['back']) if readings['back'] else None, left_m=min(readings['left']) if readings['left'] else None, right_m=min(readings['right']) if readings['right'] else None, up_m=min(readings['up']) if readings['up'] else None, down_m=min(readings['down']) if readings['down'] else None)

def _latest_ground_distance_from_telemetry(telemetry: HoverTelemetryCollector | None) -> HoverGroundDistanceObservation:
    if telemetry is None:
        return HoverGroundDistanceObservation(distance_m=None, age_s=None)
    latest_value, age_s = telemetry.latest_value('range.zrange')
    supervisor_value, supervisor_age_s = telemetry.latest_value('supervisor.info')
    is_flying: bool | None = None
    if supervisor_value is not None:
        is_flying = int(supervisor_value) & _SUPERVISOR_IS_FLYING_MASK != 0
    if latest_value is None:
        return HoverGroundDistanceObservation(distance_m=None, age_s=None, is_flying=is_flying, supervisor_age_s=supervisor_age_s)
    numeric = float(latest_value)
    if numeric <= 0.0 or numeric >= HOVER_RANGE_INVALID_MM:
        return HoverGroundDistanceObservation(distance_m=None, age_s=age_s, is_flying=is_flying, supervisor_age_s=supervisor_age_s)
    return HoverGroundDistanceObservation(distance_m=numeric / 1000.0, age_s=age_s, is_flying=is_flying, supervisor_age_s=supervisor_age_s)

def run_hover_test(*, uri: str, workspace: Path, height_m: float, hover_duration_s: float, takeoff_velocity_mps: float, land_velocity_mps: float, connect_settle_s: float, min_vbat_v: float, min_battery_level: int, min_clearance_m: float, stabilizer_estimator: int, stabilizer_controller: int, motion_disable: int, estimator_settle_timeout_s: float, on_device_failsafe_mode: str, on_device_failsafe_heartbeat_timeout_s: float, on_device_failsafe_low_battery_v: float, on_device_failsafe_critical_battery_v: float, on_device_failsafe_min_up_clearance_m: float, required_decks: tuple[str, ...], trace_writer: HoverWorkerTraceWriter | None=None) -> HoverTestReport:
    trace = trace_writer or HoverWorkerTraceWriter(None)
    trace.emit('run_hover_test', status='begin', data={'uri': uri, 'workspace': str(workspace), 'height_m': height_m, 'hover_duration_s': hover_duration_s})
    trace.emit('cflib_import', status='begin')
    crtp, crazyflie_cls, log_config_cls, multiranger_cls, sync_crazyflie_cls, sync_logger_cls = _import_cflib()
    trace.emit('cflib_import', status='done')
    workspace = workspace.expanduser()
    workspace.mkdir(parents=True, exist_ok=True)
    workspace = workspace.resolve()
    cache_dir = workspace / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    trace.emit('workspace_ready', status='done', data={'workspace': str(workspace), 'cache_dir': str(cache_dir)})
    trace.emit('crtp_init', status='begin')
    crtp.init_drivers()
    trace.emit('crtp_init', status='done')
    crazyflie = crazyflie_cls(rw_cache=str(cache_dir))
    sync_context = sync_crazyflie_cls(uri, cf=crazyflie)
    sync_cf = None
    entered_sync_context = False
    deck_flags: dict[str, int | None] = {}
    power = HoverPowerSnapshot(vbat_v=None, battery_level=None, state=None)
    status_snapshot: HoverStatusSnapshot | None = None
    clearance_snapshot: HoverClearanceSnapshot | None = None
    pre_arm_snapshot: HoverPreArmSnapshot | None = None
    estimator_settle: HoverEstimatorSettlingReport | None = None
    primitive_outcome: HoverPrimitiveOutcome | None = None
    telemetry_samples: tuple[HoverTelemetrySample, ...] = ()
    telemetry_summary: HoverTelemetrySummary | None = None
    on_device_failsafe_report: OnDeviceFailsafeSessionReport | None = None
    took_off = False
    landed = False
    primitive: StatefulHoverPrimitive | None = None
    telemetry = None
    on_device_failsafe_session: OnDeviceFailsafeHeartbeatSession | None = None

    def _stop_telemetry() -> None:
        if telemetry is None:
            return
        trace.emit('telemetry_stop', status='begin')
        telemetry.stop()
        trace.emit('telemetry_stop', status='done')

    def _snapshot_telemetry() -> tuple[HoverTelemetrySample, ...]:
        trace.emit('telemetry_snapshot', status='begin')
        samples = telemetry.snapshot() if telemetry is not None else ()
        trace.emit('telemetry_snapshot', status='done', data={'sample_count': len(samples)})
        return samples

    def _build_report(*, status: str, completed: bool, landed_flag: bool, interrupted: bool, failures: Iterable[str], telemetry_payload: tuple[HoverTelemetrySample, ...]=(), telemetry_summary_payload: HoverTelemetrySummary | None=None) -> HoverTestReport:
        report = HoverTestReport(uri=uri, workspace=str(workspace), height_m=height_m, hover_duration_s=hover_duration_s, takeoff_velocity_mps=takeoff_velocity_mps, land_velocity_mps=land_velocity_mps, connect_settle_s=connect_settle_s, min_vbat_v=min_vbat_v, min_battery_level=min_battery_level, min_clearance_m=min_clearance_m, deck_flags=deck_flags, required_decks=required_decks, clearance_snapshot=clearance_snapshot, status_snapshot=status_snapshot, pre_arm_snapshot=pre_arm_snapshot, estimator_settle=estimator_settle, power=power, status=status, completed=completed, landed=landed_flag, interrupted=interrupted, primitive_outcome=primitive_outcome, telemetry=telemetry_payload, telemetry_summary=telemetry_summary_payload, failures=tuple(failures), recommendations=(), on_device_failsafe=on_device_failsafe_report)
        trace.emit('report_build', status='begin', data={'status': status})
        final_report = replace(report, recommendations=recommendations_for_report(report))
        trace.emit('report_build', status='done', data={'status': final_report.status})
        return final_report
    try:
        trace.emit('sync_connect', status='begin', data={'uri': uri})
        sync_cf = sync_context.__enter__()
        entered_sync_context = True
        trace.emit('sync_connect', status='done')
        if connect_settle_s > 0:
            trace.emit('connect_settle', status='begin', data={'sleep_s': connect_settle_s})
            time.sleep(connect_settle_s)
            trace.emit('connect_settle', status='done')
        trace.emit('deck_flags', status='begin')
        deck_flags = _read_deck_flags(sync_cf, DECK_PARAM_NAMES)
        trace.emit('deck_flags', status='done', data={'deck_flags': deck_flags})
        if deck_flags.get('bcMultiranger') == 1:
            trace.emit('clearance_snapshot', status='begin')
            clearance_snapshot = _read_clearance_snapshot(sync_cf, multiranger_cls)
            trace.emit('clearance_snapshot', status='done', data=asdict(clearance_snapshot))
        else:
            trace.emit('clearance_snapshot', status='skipped', data={'reason': 'multiranger_not_detected'})
        trace.emit('preflight_snapshot', status='begin')
        power, status_snapshot = _read_preflight_snapshots(sync_cf, log_config_cls)
        trace.emit('preflight_snapshot', status='done', data={'vbat_v': power.vbat_v, 'battery_level': power.battery_level, 'power_state': power.state, 'power_state_name': _power_state_name(power.state), 'supervisor_info': None if status_snapshot is None else status_snapshot.supervisor_info, 'can_fly': None if status_snapshot is None else status_snapshot.can_fly, 'radio_connected': None if status_snapshot is None else status_snapshot.radio_connected, 'zrange_m': None if status_snapshot is None else status_snapshot.zrange_m})
        failures = evaluate_hover_preflight(deck_flags=deck_flags, required_decks=required_decks, power=power, status_snapshot=status_snapshot, clearance_snapshot=clearance_snapshot, min_vbat_v=min_vbat_v, min_battery_level=min_battery_level, min_clearance_m=min_clearance_m)
        if failures:
            trace.emit('preflight', status='blocked', data={'failures': failures})
            return _build_report(status='blocked', completed=False, landed_flag=False, interrupted=False, failures=failures)
        trace.emit('preflight', status='done')
        pre_arm_snapshot = apply_hover_pre_arm(sync_cf, config=HoverPreArmConfig(estimator=stabilizer_estimator, controller=stabilizer_controller, motion_disable=motion_disable), trace_writer=trace)
        if pre_arm_snapshot.failures:
            return _build_report(status='blocked', completed=False, landed_flag=False, interrupted=False, failures=pre_arm_snapshot.failures)
        estimator_settle = wait_for_estimator_settle(sync_cf, log_config_cls, sync_logger_cls, config=HoverEstimatorSettlingConfig(timeout_s=estimator_settle_timeout_s), trace_writer=trace)
        if not estimator_settle.stable:
            return _build_report(status='blocked', completed=False, landed_flag=False, interrupted=False, failures=estimator_settle.failures)
        if on_device_failsafe_mode != 'off':
            trace.emit('on_device_failsafe_probe', status='begin', data={'mode': on_device_failsafe_mode})
            availability = probe_on_device_failsafe(sync_cf)
            trace.emit('on_device_failsafe_probe', status='done', data={'loaded': availability.loaded, 'protocol_version': availability.protocol_version, 'state': availability.state_name, 'reason': availability.reason_name})
            on_device_failsafe_config = OnDeviceFailsafeConfig(heartbeat_timeout_s=on_device_failsafe_heartbeat_timeout_s, low_battery_v=on_device_failsafe_low_battery_v, critical_battery_v=on_device_failsafe_critical_battery_v, min_clearance_m=min_clearance_m, min_up_clearance_m=on_device_failsafe_min_up_clearance_m)
            on_device_failsafe_report = OnDeviceFailsafeSessionReport(
                mode=on_device_failsafe_mode,
                config=on_device_failsafe_config,
                availability=availability,
                session_id=None,
                started=False,
                closed=False,
                disabled_cleanly=False,
                packets_sent=0,
                status_packets_received=0,
                heartbeat_deadline_misses=0,
                started_monotonic_s=None,
                closed_monotonic_s=None,
                last_heartbeat_sent_monotonic_s=None,
                last_status_received_monotonic_s=None,
                last_status=None,
                link_metrics=None,
                failures=availability.failures,
            )
            if not availability.loaded and on_device_failsafe_mode == 'required':
                return _build_report(status='blocked', completed=False, landed_flag=False, interrupted=False, failures=('required on-device failsafe app `twinrFs` is not loaded on the Crazyflie firmware',))
            if availability.loaded:
                on_device_failsafe_session = OnDeviceFailsafeHeartbeatSession(sync_cf, mode=on_device_failsafe_mode, config=on_device_failsafe_config, availability=availability, trace_writer=trace)
                on_device_failsafe_session.start()
                on_device_failsafe_report = on_device_failsafe_session.report()
        telemetry_max_samples = _estimate_telemetry_max_samples(height_m=height_m, hover_duration_s=hover_duration_s, takeoff_velocity_mps=takeoff_velocity_mps, land_velocity_mps=land_velocity_mps, period_in_ms=HOVER_TELEMETRY_PERIOD_MS, block_count=len(HOVER_TELEMETRY_BLOCKS))
        telemetry = HoverTelemetryCollector(sync_cf, log_config_cls, max_samples=telemetry_max_samples)
        trace.emit('telemetry_start', status='begin')
        telemetry.start()
        trace.emit('telemetry_start', status='done', data={'available_blocks': telemetry.available_blocks, 'skipped_blocks': telemetry.skipped_blocks, 'max_samples': telemetry_max_samples})
        if HOVER_TELEMETRY_STARTUP_SETTLE_S > 0:
            trace.emit('telemetry_settle', status='begin', data={'sleep_s': HOVER_TELEMETRY_STARTUP_SETTLE_S})
            time.sleep(HOVER_TELEMETRY_STARTUP_SETTLE_S)
            trace.emit('telemetry_settle', status='done')
        trace.emit('hover_primitive_create', status='begin')
        primitive = StatefulHoverPrimitive(sync_cf, ground_distance_provider=lambda: _latest_ground_distance_from_telemetry(telemetry), trace_writer=trace)
        trace.emit('hover_primitive_create', status='done')
        primitive_outcome = primitive.run(HoverPrimitiveConfig(target_height_m=height_m, hover_duration_s=hover_duration_s, takeoff_velocity_mps=takeoff_velocity_mps, land_velocity_mps=land_velocity_mps))
        took_off = primitive_outcome.took_off
        landed = primitive_outcome.landed
        _stop_telemetry()
    except KeyboardInterrupt:
        trace.emit('worker_interrupt', status='error', message='KeyboardInterrupt')
        try:
            _stop_telemetry()
        except Exception as exc:
            trace.emit('telemetry_stop', status='error', message=f'{exc.__class__.__name__}:{exc}')
        telemetry_samples = _snapshot_telemetry()
        trace.emit('telemetry_summary', status='begin')
        telemetry_summary = summarize_hover_telemetry(telemetry_samples, available_blocks=telemetry.available_blocks if telemetry is not None else (), skipped_blocks=telemetry.skipped_blocks if telemetry is not None else ())
        trace.emit('telemetry_summary', status='done', data={'sample_count': telemetry_summary.sample_count})
        if primitive is not None:
            took_off = took_off or primitive.took_off
            landed = landed or primitive.landed
        if on_device_failsafe_session is not None:
            on_device_failsafe_report = on_device_failsafe_session.report()
        return _build_report(status='interrupted', completed=False, landed_flag=landed, interrupted=True, failures=('hover test interrupted; landing requested',), telemetry_payload=telemetry_samples, telemetry_summary_payload=telemetry_summary)
    except Exception as exc:
        trace.emit('run_hover_test_exception', status='error', message=f'{exc.__class__.__name__}:{exc}')
        try:
            _stop_telemetry()
        except Exception as stop_exc:
            trace.emit('telemetry_stop', status='error', message=f'{stop_exc.__class__.__name__}:{stop_exc}')
        if primitive is not None:
            took_off = took_off or primitive.took_off
            landed = landed or primitive.landed
        raise
    finally:
        if on_device_failsafe_session is not None:
            disable_on_close = bool(landed or not took_off)
            trace.emit('on_device_failsafe_release', status='begin', data={'disable_on_close': disable_on_close, 'took_off': took_off, 'landed': landed})
            on_device_failsafe_session.close(disable=disable_on_close)
            on_device_failsafe_report = on_device_failsafe_session.report()
            trace.emit('on_device_failsafe_release', status='done', data={'disabled_cleanly': on_device_failsafe_report.disabled_cleanly, 'status_packets_received': on_device_failsafe_report.status_packets_received})
        if entered_sync_context:
            trace.emit('sync_disconnect', status='begin')
            try:
                suppressed = bool(sync_context.__exit__(*sys.exc_info()))
            except BaseException as exc:
                trace.emit('sync_disconnect', status='error', message=f'{exc.__class__.__name__}:{exc}')
                raise
            else:
                trace.emit('sync_disconnect', status='done', data={'suppressed': suppressed})
    telemetry_samples = _snapshot_telemetry()
    trace.emit('telemetry_summary', status='begin')
    telemetry_summary = summarize_hover_telemetry(telemetry_samples, available_blocks=telemetry.available_blocks if telemetry is not None else (), skipped_blocks=telemetry.skipped_blocks if telemetry is not None else ())
    trace.emit('telemetry_summary', status='done', data={'sample_count': telemetry_summary.sample_count, 'flow_observed': telemetry_summary.flow_observed, 'zrange_observed': telemetry_summary.zrange_observed, 'stable_supervisor': telemetry_summary.stable_supervisor, 'radio_disconnect_seen': telemetry_summary.radio_disconnect_seen})
    stability_failures = evaluate_hover_stability(telemetry_summary, target_height_m=height_m)
    primitive_failures = list(_evaluate_primitive_outcome(primitive_outcome))
    all_failures = tuple(primitive_failures + stability_failures)
    trace.emit('stability_eval', status='done', data={'failures': all_failures})
    return _build_report(status='completed' if not all_failures else 'unstable', completed=not all_failures, landed_flag=bool(primitive_outcome.landed) if primitive_outcome is not None else landed, interrupted=False, failures=all_failures, telemetry_payload=telemetry_samples, telemetry_summary_payload=telemetry_summary)

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run a bounded Crazyflie hover test.')
    parser.add_argument('--uri', default=DEFAULT_URI, help='Crazyflie radio URI (default: radio://0/80/2M)')
    parser.add_argument('--workspace', default='/twinr/bitcraze', help='Bitcraze workspace root for cflib cache files')
    parser.add_argument('--height-m', type=float, default=0.25, help='Hover height in meters (default: 0.25)')
    parser.add_argument('--hover-duration-s', type=float, default=3.0, help='Hover hold time in seconds (default: 3.0)')
    parser.add_argument('--takeoff-velocity-mps', type=float, default=0.2, help='Takeoff velocity in meters per second (default: 0.2)')
    parser.add_argument('--land-velocity-mps', type=float, default=0.2, help='Landing velocity in meters per second (default: 0.2)')
    parser.add_argument('--connect-settle-s', type=float, default=1.0, help='Initial wait after connect before preflight and takeoff (default: 1.0)')
    parser.add_argument('--min-vbat-v', type=float, default=3.8, help='Minimum battery voltage for hover acceptance (default: 3.8)')
    parser.add_argument('--min-battery-level', type=int, default=20, help='Minimum battery level percent for hover acceptance (default: 20)')
    parser.add_argument('--min-clearance-m', type=float, default=HOVER_DEFAULT_MIN_CLEARANCE_M, help='Minimum observed Multi-ranger clearance in meters when that deck is present (default: 0.35)')
    parser.add_argument('--stabilizer-estimator', type=int, default=HOVER_DEFAULT_ESTIMATOR, help='Deterministic stabilizer.estimator value for hover pre-arm (default: 2)')
    parser.add_argument('--stabilizer-controller', type=int, default=HOVER_DEFAULT_CONTROLLER, help='Deterministic stabilizer.controller value for hover pre-arm (default: 1)')
    parser.add_argument('--motion-disable', type=int, default=HOVER_DEFAULT_MOTION_DISABLE, help='Deterministic motion.disable value for hover pre-arm (default: 0)')
    parser.add_argument('--estimator-settle-timeout-s', type=float, default=HoverEstimatorSettlingConfig().timeout_s, help='Maximum seconds to wait for Kalman settle before takeoff (default: 5.0)')
    parser.add_argument('--on-device-failsafe-mode', choices=('required', 'preferred', 'off'), default=HOVER_DEFAULT_ON_DEVICE_FAILSAFE_MODE, help='Whether the hover worker requires the Twinr firmware failsafe app (default: required).')
    parser.add_argument('--on-device-failsafe-heartbeat-timeout-s', type=float, default=ON_DEVICE_FAILSAFE_HEARTBEAT_TIMEOUT_S, help='Firmware heartbeat timeout for the on-device failsafe in seconds (default: 0.35).')
    parser.add_argument('--on-device-failsafe-low-battery-v', type=float, default=ON_DEVICE_FAILSAFE_LOW_BATTERY_V, help='Firmware low-battery safe-land threshold in volts (default: 3.55).')
    parser.add_argument('--on-device-failsafe-critical-battery-v', type=float, default=ON_DEVICE_FAILSAFE_CRITICAL_BATTERY_V, help='Firmware critical-battery safe-land threshold in volts (default: 3.35).')
    parser.add_argument('--on-device-failsafe-min-up-clearance-m', type=float, default=ON_DEVICE_FAILSAFE_MIN_UP_CLEARANCE_M, help='Firmware upward-clearance trigger in meters (default: 0.25).')
    parser.add_argument('--require-deck', action='append', default=[], help='Require one deck flag (examples: flow2, zranger2, multiranger, aideck); repeat as needed')
    parser.add_argument('--trace-file', default='', help='Optional JSONL trace path for worker-phase diagnostics and timeout forensics.')
    parser.add_argument('--json', action='store_true', help='Emit the full report as JSON')
    return parser

def _validate_bounded_number(name: str, raw_value: float, *, minimum: float | None=None, maximum: float | None=None, unit: str='') -> float:
    value = float(raw_value)
    unit_suffix = f' {unit}' if unit else ''
    if minimum is not None and value < minimum:
        raise ValueError(f'{name} must be >= {minimum:.2f}{unit_suffix}; got {value:.2f}{unit_suffix}')
    if maximum is not None and value > maximum:
        raise ValueError(f'{name} must be <= {maximum:.2f}{unit_suffix}; got {value:.2f}{unit_suffix}')
    return value

def _validate_runtime_config(args: argparse.Namespace) -> HoverRuntimeConfig:
    normalized_decks: list[str] = []
    for raw_name in args.require_deck or []:
        if not str(raw_name or '').strip():
            continue
        deck_name = normalize_required_deck_name(raw_name)
        if deck_name not in normalized_decks:
            normalized_decks.append(deck_name)
    required_decks = tuple(normalized_decks) or DEFAULT_REQUIRED_DECKS
    height_m = _validate_bounded_number('height_m', args.height_m, minimum=HOVER_MIN_HEIGHT_M, maximum=HOVER_MAX_HEIGHT_M, unit='m')
    hover_duration_s = _validate_bounded_number('hover_duration_s', args.hover_duration_s, minimum=HOVER_MIN_DURATION_S, maximum=HOVER_MAX_DURATION_S, unit='s')
    takeoff_velocity_mps = _validate_bounded_number('takeoff_velocity_mps', args.takeoff_velocity_mps, minimum=HOVER_MIN_VERTICAL_VELOCITY_MPS, maximum=HOVER_MAX_VERTICAL_VELOCITY_MPS, unit='m/s')
    land_velocity_mps = _validate_bounded_number('land_velocity_mps', args.land_velocity_mps, minimum=HOVER_MIN_VERTICAL_VELOCITY_MPS, maximum=HOVER_MAX_VERTICAL_VELOCITY_MPS, unit='m/s')
    connect_settle_s = _validate_bounded_number('connect_settle_s', args.connect_settle_s, minimum=0.0, maximum=HOVER_MAX_CONNECT_SETTLE_S, unit='s')
    min_vbat_v = _validate_bounded_number('min_vbat_v', args.min_vbat_v, minimum=0.0, maximum=HOVER_MAX_BATTERY_V, unit='V')
    min_battery_level = int(args.min_battery_level)
    if not 0 <= min_battery_level <= HOVER_MAX_MIN_BATTERY_LEVEL:
        raise ValueError(f'min_battery_level must be between 0 and {HOVER_MAX_MIN_BATTERY_LEVEL}; got {min_battery_level}')
    min_clearance_m = _validate_bounded_number('min_clearance_m', args.min_clearance_m, minimum=0.0, maximum=HOVER_MAX_CLEARANCE_M, unit='m')
    estimator_settle_timeout_s = _validate_bounded_number('estimator_settle_timeout_s', args.estimator_settle_timeout_s, minimum=0.5, maximum=HOVER_MAX_ESTIMATOR_SETTLE_TIMEOUT_S, unit='s')
    heartbeat_timeout_s = _validate_bounded_number('on_device_failsafe_heartbeat_timeout_s', args.on_device_failsafe_heartbeat_timeout_s, minimum=0.05, maximum=HOVER_MAX_FAILSAFE_HEARTBEAT_TIMEOUT_S, unit='s')
    failsafe_low_battery_v = _validate_bounded_number('on_device_failsafe_low_battery_v', args.on_device_failsafe_low_battery_v, minimum=0.0, maximum=HOVER_MAX_BATTERY_V, unit='V')
    failsafe_critical_battery_v = _validate_bounded_number('on_device_failsafe_critical_battery_v', args.on_device_failsafe_critical_battery_v, minimum=0.0, maximum=HOVER_MAX_BATTERY_V, unit='V')
    if failsafe_low_battery_v < failsafe_critical_battery_v:
        raise ValueError('on_device_failsafe_low_battery_v must be >= on_device_failsafe_critical_battery_v')
    failsafe_min_up_clearance_m = _validate_bounded_number('on_device_failsafe_min_up_clearance_m', args.on_device_failsafe_min_up_clearance_m, minimum=0.0, maximum=HOVER_MAX_CLEARANCE_M, unit='m')
    return {'uri': str(args.uri).strip() or DEFAULT_URI, 'workspace': Path(str(args.workspace).strip() or '/twinr/bitcraze'), 'height_m': height_m, 'hover_duration_s': hover_duration_s, 'takeoff_velocity_mps': takeoff_velocity_mps, 'land_velocity_mps': land_velocity_mps, 'connect_settle_s': connect_settle_s, 'min_vbat_v': min_vbat_v, 'min_battery_level': min_battery_level, 'min_clearance_m': min_clearance_m, 'stabilizer_estimator': int(args.stabilizer_estimator), 'stabilizer_controller': int(args.stabilizer_controller), 'motion_disable': int(args.motion_disable), 'estimator_settle_timeout_s': estimator_settle_timeout_s, 'on_device_failsafe_mode': str(args.on_device_failsafe_mode).strip() or HOVER_DEFAULT_ON_DEVICE_FAILSAFE_MODE, 'on_device_failsafe_heartbeat_timeout_s': heartbeat_timeout_s, 'on_device_failsafe_low_battery_v': failsafe_low_battery_v, 'on_device_failsafe_critical_battery_v': failsafe_critical_battery_v, 'on_device_failsafe_min_up_clearance_m': failsafe_min_up_clearance_m, 'required_decks': required_decks}

def _print_human_report(report: HoverTestReport) -> None:
    print(f'status={report.status}')
    print(f'uri={report.uri}')
    print(f'workspace={report.workspace}')
    print(f'height_m={report.height_m}')
    print(f'hover_duration_s={report.hover_duration_s}')
    print(f'power.vbat_v={report.power.vbat_v}')
    print(f'power.battery_level={report.power.battery_level}')
    print(f'power.state={report.power.state}')
    print(f'power.state_name={_power_state_name(report.power.state)}')
    print(f'min_clearance_m={report.min_clearance_m}')
    if report.status_snapshot is not None:
        print(f'status.supervisor_info={report.status_snapshot.supervisor_info}')
        print(f'status.can_arm={str(report.status_snapshot.can_arm).lower()}')
        print(f'status.can_fly={str(report.status_snapshot.can_fly).lower()}')
        print(f'status.is_flying={str(report.status_snapshot.is_flying).lower()}')
        print(f'status.tumbled={str(report.status_snapshot.tumbled).lower()}')
        print(f'status.locked={str(report.status_snapshot.locked).lower()}')
        print(f'status.crashed={str(report.status_snapshot.crashed).lower()}')
        print(f'status.radio_connected={str(report.status_snapshot.radio_connected).lower()}')
        print(f'status.zrange_m={report.status_snapshot.zrange_m}')
        print(f'status.motion_squal={report.status_snapshot.motion_squal}')
    if report.clearance_snapshot is not None:
        print(f'clearance.front_m={report.clearance_snapshot.front_m}')
        print(f'clearance.back_m={report.clearance_snapshot.back_m}')
        print(f'clearance.left_m={report.clearance_snapshot.left_m}')
        print(f'clearance.right_m={report.clearance_snapshot.right_m}')
        print(f'clearance.up_m={report.clearance_snapshot.up_m}')
        print(f'clearance.down_m={report.clearance_snapshot.down_m}')
    if report.pre_arm_snapshot is not None:
        print(f'pre_arm.estimator={report.pre_arm_snapshot.estimator}')
        print(f'pre_arm.controller={report.pre_arm_snapshot.controller}')
        print(f'pre_arm.motion_disable={report.pre_arm_snapshot.motion_disable}')
        print(f'pre_arm.kalman_reset_after={report.pre_arm_snapshot.kalman_reset_after}')
        print(f'pre_arm.verified={str(report.pre_arm_snapshot.verified).lower()}')
    if report.estimator_settle is not None:
        print(f'settle.stable={str(report.estimator_settle.stable).lower()}')
        print(f'settle.sample_count={report.estimator_settle.sample_count}')
        print(f'settle.var_px_span={report.estimator_settle.var_px_span}')
        print(f'settle.var_py_span={report.estimator_settle.var_py_span}')
        print(f'settle.var_pz_span={report.estimator_settle.var_pz_span}')
        print(f'settle.motion_squal_mean={report.estimator_settle.motion_squal_mean}')
        print(f'settle.motion_squal_nonzero_ratio={report.estimator_settle.motion_squal_nonzero_ratio}')
        print(f'settle.zrange_min_m={report.estimator_settle.zrange_min_m}')
    if report.primitive_outcome is not None:
        print(f'primitive.final_phase={report.primitive_outcome.final_phase}')
        print(f'primitive.took_off={str(report.primitive_outcome.took_off).lower()}')
        print(f'primitive.landed={str(report.primitive_outcome.landed).lower()}')
        print(f'primitive.commanded_max_height_m={report.primitive_outcome.commanded_max_height_m}')
        print(f'primitive.setpoint_count={report.primitive_outcome.setpoint_count}')
    if report.on_device_failsafe is not None:
        print(f'on_device_failsafe.mode={report.on_device_failsafe.mode}')
        print(f'on_device_failsafe.loaded={str(report.on_device_failsafe.availability.loaded).lower()}')
        print(f'on_device_failsafe.started={str(report.on_device_failsafe.started).lower()}')
        print(f'on_device_failsafe.disabled_cleanly={str(report.on_device_failsafe.disabled_cleanly).lower()}')
        print(f'on_device_failsafe.status_packets_received={report.on_device_failsafe.status_packets_received}')
        if report.on_device_failsafe.last_status is not None:
            print(f'on_device_failsafe.last_state={report.on_device_failsafe.last_status.state_name}')
            print(f'on_device_failsafe.last_reason={report.on_device_failsafe.last_status.reason_name}')
            print(f'on_device_failsafe.last_vbat_mv={report.on_device_failsafe.last_status.vbat_mv}')
            print(f'on_device_failsafe.last_min_clearance_mm={report.on_device_failsafe.last_status.min_clearance_mm}')
    if report.telemetry_summary is not None:
        print(f'telemetry.sample_count={report.telemetry_summary.sample_count}')
        print(f"telemetry.available_blocks={','.join(report.telemetry_summary.available_blocks)}")
        if report.telemetry_summary.skipped_blocks:
            print(f"telemetry.skipped_blocks={','.join(report.telemetry_summary.skipped_blocks)}")
        print(f'telemetry.flow_observed={str(report.telemetry_summary.flow_observed).lower()}')
        print(f'telemetry.zrange_observed={str(report.telemetry_summary.zrange_observed).lower()}')
        print(f'telemetry.xy_drift_m={report.telemetry_summary.xy_drift_m}')
        print(f'telemetry.z_drift_m={report.telemetry_summary.z_drift_m}')
        print(f'telemetry.z_span_m={report.telemetry_summary.z_span_m}')
        print(f'telemetry.horizontal_speed_max_mps={report.telemetry_summary.horizontal_speed_max_mps}')
        print(f'telemetry.flow_squal_mean={report.telemetry_summary.flow_squal_mean}')
        print(f'telemetry.motion_delta_x_abs_max={report.telemetry_summary.motion_delta_x_abs_max}')
        print(f'telemetry.motion_delta_y_abs_max={report.telemetry_summary.motion_delta_y_abs_max}')
        print(f'telemetry.front_min_m={report.telemetry_summary.front_min_m}')
        print(f'telemetry.back_min_m={report.telemetry_summary.back_min_m}')
        print(f'telemetry.left_min_m={report.telemetry_summary.left_min_m}')
        print(f'telemetry.right_min_m={report.telemetry_summary.right_min_m}')
        print(f'telemetry.up_min_m={report.telemetry_summary.up_min_m}')
        print(f'telemetry.thrust_max={report.telemetry_summary.thrust_max}')
        print(f'telemetry.gyro_abs_max_dps={report.telemetry_summary.gyro_abs_max_dps}')
        print(f'telemetry.radio_rssi_latest_dbm={report.telemetry_summary.radio_rssi_latest_dbm}')
        print(f'telemetry.radio_disconnect_seen={str(report.telemetry_summary.radio_disconnect_seen).lower()}')
        print(f'telemetry.battery_drop_v={report.telemetry_summary.battery_drop_v}')
        if report.telemetry_summary.supervisor_flags_seen:
            print(f"telemetry.supervisor_flags={','.join(report.telemetry_summary.supervisor_flags_seen)}")
    for deck_name, flag in sorted(report.deck_flags.items()):
        print(f"deck.{deck_name}={(flag if flag is not None else 'unknown')}")
    for failure in report.failures:
        print(f'failure={failure}')
    for recommendation in report.recommendations:
        print(f'recommendation={recommendation}')

def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        trace_writer = HoverWorkerTraceWriter(Path(str(args.trace_file).strip()) if str(args.trace_file).strip() else None)
    except Exception as exc:
        failure_message = f'invalid_trace_file:{exc.__class__.__name__}:{exc}'
        if args.json:
            print(json.dumps({'report': None, 'failures': [failure_message]}, indent=2, sort_keys=True))
        else:
            print(f'failure={failure_message}')
        return 1
    trace_writer.emit('main', status='begin', data={'json': bool(args.json)})
    try:
        config = _validate_runtime_config(args)
    except Exception as exc:
        failure_message = f'invalid_hover_test_arguments:{exc.__class__.__name__}:{exc}'
        trace_writer.emit('main', status='error', message=failure_message)
        if args.json:
            print(json.dumps({'report': None, 'failures': [failure_message]}, indent=2, sort_keys=True))
        else:
            print(f'failure={failure_message}')
        trace_writer.emit('main', status='done', data={'exit_code': 1})
        return 1
    signal_context = GracefulSignalAbortContext(trace_writer)
    try:
        with signal_context:
            report = run_hover_test(trace_writer=trace_writer, **config)
    except KeyboardInterrupt:
        signal_name = signal_context.last_signal_name or 'KeyboardInterrupt'
        failure_message = f'hover_test_interrupted:{signal_name}'
        trace_writer.emit('main', status='error', message=failure_message)
        if args.json:
            print(json.dumps({'report': None, 'failures': [failure_message]}, indent=2, sort_keys=True))
        else:
            print(f'failure={failure_message}')
        trace_writer.emit('main', status='done', data={'exit_code': 130})
        return 130
    except Exception as exc:
        failure_message = f'hover_test_exception:{exc.__class__.__name__}:{exc}'
        failure_payload = {'report': None, 'failures': [failure_message]}
        trace_writer.emit('main', status='error', message=failure_message)
        if args.json:
            trace_writer.emit('json_emit', status='begin', data={'kind': 'failure'})
            print(json.dumps(failure_payload, indent=2, sort_keys=True))
            trace_writer.emit('json_emit', status='done', data={'kind': 'failure'})
        else:
            print(f'failure={failure_message}')
        trace_writer.emit('main', status='done', data={'exit_code': 1})
        return 1
    if args.json:
        trace_writer.emit('json_emit', status='begin', data={'kind': 'success', 'status': report.status})
        print(json.dumps({'report': asdict(report), 'failures': list(report.failures)}, indent=2, sort_keys=True))
        trace_writer.emit('json_emit', status='done', data={'kind': 'success', 'status': report.status})
    else:
        _print_human_report(report)
    exit_code = 1
    if report.status == 'completed':
        exit_code = 0
    elif report.status == 'interrupted':
        exit_code = 130
    trace_writer.emit('main', status='done', data={'exit_code': exit_code, 'report_status': report.status})
    return exit_code
if __name__ == '__main__':
    raise SystemExit(main())
