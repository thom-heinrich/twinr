#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "cflib>=0.1.31",
# ]
# ///
"""Run one bounded Crazyflie hover test.

Purpose
-------
Execute the smallest useful indoor flight primitive for Twinr's drone stack:
connect, verify the required decks and battery state, take off to a low hover,
hold briefly, and land again. The command is operator-facing and intentionally
bounded so it can serve as the first live flight acceptance step once the
mission daemon is ready to allow hover tests. Before takeoff, the worker now
also applies deterministic estimator/controller params, resets the Kalman
filter, and blocks on an estimator-settling gate so hover artifacts include
pre-arm evidence instead of only in-air telemetry.

Usage
-----
Command-line examples::

    /twinr/bitcraze/.venv/bin/python hardware/bitcraze/run_hover_test.py --json
    /twinr/bitcraze/.venv/bin/python hardware/bitcraze/run_hover_test.py --height-m 0.25 --hover-duration-s 3.0
    /twinr/bitcraze/.venv/bin/python hardware/bitcraze/run_hover_test.py --json --trace-file /tmp/hover-trace.jsonl

Outputs
-------
- Human-readable status lines by default
- JSON report with ``--json`` including pre-arm, estimator-settle, primitive,
  and in-air telemetry evidence
- Optional JSONL phase trace with ``--trace-file`` for teardown/timeout forensics
- Exit code ``0`` when the hover test completed and landed
- Exit code ``1`` when preflight failed or the hover test raised an error
- Exit code ``130`` when the run was interrupted and a landing was requested
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Iterable, Mapping, cast

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from hover_primitive import (  # noqa: E402
    HoverEstimatorSettlingConfig,
    HoverEstimatorSettlingReport,
    HoverGroundDistanceObservation,
    HoverPreArmConfig,
    HoverPreArmSnapshot,
    HoverPrimitiveConfig,
    HoverPrimitiveOutcome,
    StatefulHoverPrimitive,
    apply_hover_pre_arm,
    wait_for_estimator_settle,
)
from on_device_failsafe import (  # noqa: E402
    ON_DEVICE_FAILSAFE_CRITICAL_BATTERY_V,
    ON_DEVICE_FAILSAFE_HEARTBEAT_TIMEOUT_S,
    ON_DEVICE_FAILSAFE_LOW_BATTERY_V,
    ON_DEVICE_FAILSAFE_MIN_UP_CLEARANCE_M,
    OnDeviceFailsafeConfig,
    OnDeviceFailsafeHeartbeatSession,
    OnDeviceFailsafeSessionReport,
    probe_on_device_failsafe,
)


DEFAULT_URI = "radio://0/80/2M"
DEFAULT_REQUIRED_DECKS = ("bcFlow2", "bcZRanger2")
DECK_NAME_ALIASES = {
    "flow2": "bcFlow2",
    "flow": "bcFlow2",
    "bcflow2": "bcFlow2",
    "zranger2": "bcZRanger2",
    "zranger": "bcZRanger2",
    "bczranger2": "bcZRanger2",
    "multiranger": "bcMultiranger",
    "multi-ranger": "bcMultiranger",
    "bcmultiranger": "bcMultiranger",
    "aideck": "bcAI",
    "ai": "bcAI",
    "bcai": "bcAI",
}
DECK_PARAM_NAMES = ("bcMultiranger", "bcFlow2", "bcZRanger2", "bcAI")
HOVER_TELEMETRY_PERIOD_MS = 100
HOVER_TELEMETRY_STARTUP_SETTLE_S = 0.2
HOVER_TELEMETRY_MAX_SAMPLES = 400
HOVER_TELEMETRY_AIRBORNE_MIN_ALTITUDE_M = 0.02
HOVER_TELEMETRY_WINDOW_PADDING_MS = HOVER_TELEMETRY_PERIOD_MS
HOVER_RANGE_INVALID_MM = 32000.0
HOVER_PREFLIGHT_CLEARANCE_SAMPLE_S = 0.4
HOVER_PREFLIGHT_CLEARANCE_PERIOD_S = 0.1
HOVER_DEFAULT_MIN_CLEARANCE_M = 0.35
HOVER_STABILITY_ALTITUDE_MARGIN_M = 0.25
HOVER_STABILITY_MIN_UNDER_LOAD_VBAT_V = 3.5
HOVER_DEFAULT_ESTIMATOR = 2
HOVER_DEFAULT_CONTROLLER = 1
HOVER_DEFAULT_MOTION_DISABLE = 0
HOVER_DEFAULT_ON_DEVICE_FAILSAFE_MODE = "required"
_SUPERVISOR_FLAG_BITS = (
    ("tumbled", 1 << 5),
    ("locked", 1 << 6),
    ("crashed", 1 << 7),
)
_SUPERVISOR_IS_FLYING_MASK = 1 << 4


@dataclass(frozen=True)
class HoverPowerSnapshot:
    """Summarize the bounded power facts used by hover-test preflight."""

    vbat_v: float | None
    battery_level: int | None
    state: int | None


@dataclass(frozen=True)
class HoverClearanceSnapshot:
    """Summarize one bounded Multi-ranger clearance snapshot in meters."""

    front_m: float | None
    back_m: float | None
    left_m: float | None
    right_m: float | None
    up_m: float | None
    down_m: float | None


@dataclass(frozen=True)
class HoverWorkerTraceEvent:
    """Record one bounded worker-phase event for post-flight diagnostics."""

    ts_utc: str
    elapsed_s: float
    phase: str
    status: str
    message: str | None = None
    data: dict[str, object] | None = None


class HoverWorkerTraceWriter:
    """Append durable worker-phase breadcrumbs to a JSONL trace file."""

    def __init__(self, path: Path | None) -> None:
        self.path = path.expanduser().resolve(strict=False) if path is not None else None
        self._lock = threading.Lock()
        self._start_monotonic = time.monotonic()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(
        self,
        phase: str,
        *,
        status: str,
        message: str | None = None,
        data: Mapping[str, object] | None = None,
    ) -> None:
        """Persist one worker-phase event immediately so timeouts keep evidence."""

        if self.path is None:
            return
        event = HoverWorkerTraceEvent(
            ts_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_s=max(0.0, time.monotonic() - self._start_monotonic),
            phase=str(phase),
            status=str(status),
            message=str(message) if message is not None else None,
            data=dict(data) if data is not None else None,
        )
        encoded = json.dumps(asdict(event), sort_keys=True, default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.write("\n")
                handle.flush()
                try:
                    os.fsync(handle.fileno())
                except OSError:
                    pass


@dataclass(frozen=True)
class HoverTelemetryBlockSpec:
    """Describe one bounded telemetry log block used during hover tests."""

    name: str
    variables: tuple[str, ...]
    required: bool = True


HOVER_TELEMETRY_BLOCKS = (
    HoverTelemetryBlockSpec(
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
    HoverTelemetryBlockSpec(
        name="hover-sensors",
        variables=(
            "motion.squal",
            "motion.deltaX",
            "motion.deltaY",
            "range.zrange",
            "pm.vbat",
            "supervisor.info",
            "radio.rssi",
        ),
    ),
    HoverTelemetryBlockSpec(
        name="hover-velocity",
        variables=(
            "stateEstimate.vx",
            "stateEstimate.vy",
            "stateEstimate.vz",
            "stabilizer.thrust",
        ),
    ),
    HoverTelemetryBlockSpec(
        name="hover-gyro",
        variables=(
            "gyro.x",
            "gyro.y",
            "gyro.z",
        ),
    ),
    HoverTelemetryBlockSpec(
        name="hover-clearance",
        variables=(
            "range.front",
            "range.back",
            "range.left",
            "range.right",
            "range.up",
        ),
        required=False,
    ),
)


@dataclass(frozen=True)
class HoverTelemetrySample:
    """Represent one bounded telemetry event captured during the hover run."""

    timestamp_ms: int
    block_name: str
    values: dict[str, float | int | None]


@dataclass(frozen=True)
class HoverTelemetrySummary:
    """Summarize the stability-relevant telemetry captured during the hover run."""

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
    latest_supervisor_info: int | None
    supervisor_flags_seen: tuple[str, ...]
    stable_supervisor: bool


@dataclass(frozen=True)
class HoverTestReport:
    """Represent one bounded hover-test run and its preflight outcome."""

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
    """Normalize one deck selector into the firmware deck-flag name."""

    normalized = str(value or "").strip().lower()
    if not normalized:
        raise ValueError("required deck names must not be empty")
    try:
        return DECK_NAME_ALIASES[normalized]
    except KeyError as exc:
        allowed = ", ".join(sorted(dict.fromkeys(DECK_NAME_ALIASES)))
        raise ValueError(f"unsupported deck name `{value}`; choose one of: {allowed}") from exc


def evaluate_hover_preflight(
    *,
    deck_flags: dict[str, int | None],
    required_decks: tuple[str, ...],
    power: HoverPowerSnapshot,
    clearance_snapshot: HoverClearanceSnapshot | None,
    min_vbat_v: float,
    min_battery_level: int,
    min_clearance_m: float,
) -> list[str]:
    """Return the concrete preflight failures for one bounded hover test."""

    failures: list[str] = []
    for deck_name in required_decks:
        if deck_flags.get(deck_name) != 1:
            failures.append(f"required deck {deck_name} is not detected")
    if power.vbat_v is None:
        failures.append("battery voltage is unavailable")
    elif power.vbat_v < min_vbat_v:
        failures.append(f"battery voltage {power.vbat_v:.2f} V is below the {min_vbat_v:.2f} V hover gate")
    if power.battery_level is None:
        failures.append("battery level is unavailable")
    elif power.battery_level < min_battery_level:
        failures.append(
            f"battery level {power.battery_level}% is below the {min_battery_level}% hover gate"
        )
    if clearance_snapshot is not None:
        for direction_name, value in (
            ("front", clearance_snapshot.front_m),
            ("back", clearance_snapshot.back_m),
            ("left", clearance_snapshot.left_m),
            ("right", clearance_snapshot.right_m),
            ("up", clearance_snapshot.up_m),
        ):
            if value is not None and value < min_clearance_m:
                failures.append(
                    f"{direction_name} clearance {value:.2f} m is below the {min_clearance_m:.2f} m hover gate"
                )
    return failures


def recommendations_for_report(report: HoverTestReport) -> tuple[str, ...]:
    """Return the next operator actions for the current hover-test result."""

    recommendations: list[str] = []
    if report.failures:
        recommendations.extend(report.failures)
    if report.status == "blocked":
        recommendations.append(
            "Charge the battery and confirm Flow/Z-ranger deck presence before retrying the hover test."
        )
    elif report.status == "interrupted":
        recommendations.append("Confirm the Crazyflie is on the ground before retrying the hover test.")
    elif report.status == "unstable":
        recommendations.append("Hover completed but stability gates failed; inspect the telemetry summary before retrying.")
    elif report.status == "completed":
        recommendations.append("Hover test completed and landed. The bounded flight primitive looks ready.")
    if report.on_device_failsafe is not None:
        if report.on_device_failsafe.mode == "required" and not report.on_device_failsafe.availability.loaded:
            recommendations.append(
                "Flash or load the Twinr on-device failsafe app before the next hover attempt."
            )
        recommendations.extend(report.on_device_failsafe.failures)
    return tuple(dict.fromkeys(recommendations))


def _normalize_numeric_value(raw: object) -> float | int | None:
    """Normalize one Crazyflie log payload value into a JSON-safe numeric type."""

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
    """Capture one bounded set of in-flight telemetry logs for hover analysis."""

    def __init__(
        self,
        sync_cf,
        log_config_cls,
        *,
        period_in_ms: int = HOVER_TELEMETRY_PERIOD_MS,
        max_samples: int = HOVER_TELEMETRY_MAX_SAMPLES,
    ) -> None:
        self._cf = sync_cf.cf if hasattr(sync_cf, "cf") else sync_cf
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
        """Create, register, and start the bounded telemetry log blocks."""

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
                    raise RuntimeError(
                        f"required_hover_telemetry_block_failed:{spec.name}:{exc.__class__.__name__}:{exc}"
                    ) from exc
                self._skipped_blocks[spec.name] = f"{exc.__class__.__name__}:{exc}"
                continue
            self._configs.append(config)
            self._available_blocks.append(spec.name)
        self._started = True

    def stop(self) -> None:
        """Stop and delete all started log blocks best-effort."""

        for config in reversed(self._configs):
            self._cleanup_config(config)
        self._configs.clear()
        self._started = False

    def snapshot(self) -> tuple[HoverTelemetrySample, ...]:
        """Return the captured telemetry samples in timestamp order."""

        with self._lock:
            return tuple(self._samples)

    def latest_value(self, key: str) -> tuple[float | int | None, float | None]:
        """Return the freshest recorded value for a telemetry key and its host age."""

        with self._lock:
            if key not in self._latest_by_key:
                return None, None
            value, received_monotonic_s = self._latest_by_key[key]
            return value, max(0.0, time.monotonic() - received_monotonic_s)
        return None, None

    @property
    def available_blocks(self) -> tuple[str, ...]:
        """Return the names of the telemetry blocks that started successfully."""

        return tuple(self._available_blocks)

    @property
    def skipped_blocks(self) -> tuple[str, ...]:
        """Return the names of optional telemetry blocks that were skipped."""

        return tuple(self._skipped_blocks)

    def _record_sample(self, timestamp_ms: int, data: Mapping[str, object], log_block: Any) -> None:
        """Record one bounded telemetry callback payload."""

        normalized_values = {str(key): _normalize_numeric_value(value) for key, value in dict(data).items()}
        sample = HoverTelemetrySample(
            timestamp_ms=int(timestamp_ms),
            block_name=str(getattr(log_block, "name", "hover-log")),
            values=normalized_values,
        )
        received_monotonic_s = time.monotonic()
        with self._lock:
            if len(self._samples) < self._max_samples:
                self._samples.append(sample)
            for key, value in normalized_values.items():
                self._latest_by_key[key] = (value, received_monotonic_s)

    def _cleanup_config(self, config: Any) -> None:
        """Best-effort stop and delete helper for one telemetry block."""

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


def _series_for_key(
    samples: Iterable[HoverTelemetrySample],
    key: str,
    *,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> list[tuple[int, float]]:
    """Return one timestamped numeric telemetry series for a key."""

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
    samples: Iterable[HoverTelemetrySample],
    key: str,
    *,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> list[float]:
    """Return one numeric series for a telemetry key."""

    return [value for _, value in _series_for_key(samples, key, start_ms=start_ms, end_ms=end_ms)]


def _duration_s_from_samples(samples: tuple[HoverTelemetrySample, ...]) -> float | None:
    """Return the bounded telemetry duration in seconds."""

    if len(samples) < 2:
        return None
    start_ms = min(sample.timestamp_ms for sample in samples)
    end_ms = max(sample.timestamp_ms for sample in samples)
    return max(0.0, (end_ms - start_ms) / 1000.0)


def _drift(values: list[float]) -> float | None:
    """Return the last-minus-first drift for one telemetry series."""

    if len(values) < 2:
        return None
    return values[-1] - values[0]


def _span(values: list[float]) -> float | None:
    """Return the max-minus-min span for one telemetry series."""

    if not values:
        return None
    return max(values) - min(values)


def _valid_range_values_m(values_mm: Iterable[float]) -> list[float]:
    """Convert one Crazyflie range series to meters and drop no-return sentinels."""

    values_m: list[float] = []
    for value in values_mm:
        numeric = float(value)
        if numeric <= 0.0 or numeric >= HOVER_RANGE_INVALID_MM:
            continue
        values_m.append(numeric / 1000.0)
    return values_m


def _millimeters_to_meters(values: Iterable[float]) -> list[float]:
    """Convert the Crazyflie range.zrange series from millimeters to meters."""

    return [value / 1000.0 for value in values]


def _series_min(values: list[float]) -> float | None:
    """Return the minimum of one numeric telemetry series when available."""

    if not values:
        return None
    return min(values)


def _series_max(values: list[float]) -> float | None:
    """Return the maximum of one numeric telemetry series when available."""

    if not values:
        return None
    return max(values)


def _airborne_window_bounds_ms(samples: tuple[HoverTelemetrySample, ...]) -> tuple[int | None, int | None]:
    """Infer the airborne telemetry window from altitude-related samples."""

    candidate_timestamps = [
        timestamp_ms
        for timestamp_ms, value in _series_for_key(samples, "stateEstimate.z")
        if value >= HOVER_TELEMETRY_AIRBORNE_MIN_ALTITUDE_M
    ]
    candidate_timestamps.extend(
        timestamp_ms
        for timestamp_ms, value in _series_for_key(samples, "range.zrange")
        if 0.0 < value < HOVER_RANGE_INVALID_MM
        and (value / 1000.0) >= HOVER_TELEMETRY_AIRBORNE_MIN_ALTITUDE_M
    )
    if not candidate_timestamps:
        return None, None
    return min(candidate_timestamps), max(candidate_timestamps)


def summarize_hover_telemetry(
    samples: tuple[HoverTelemetrySample, ...],
    *,
    available_blocks: tuple[str, ...] = (),
    skipped_blocks: tuple[str, ...] = (),
) -> HoverTelemetrySummary:
    """Reduce bounded hover telemetry into the stability summary stored in artifacts."""

    window_start_ms, window_end_ms = _airborne_window_bounds_ms(samples)
    filter_start_ms = None if window_start_ms is None else max(0, window_start_ms - HOVER_TELEMETRY_WINDOW_PADDING_MS)
    filter_end_ms = None if window_end_ms is None else window_end_ms + HOVER_TELEMETRY_WINDOW_PADDING_MS
    window_samples = tuple(
        sample
        for sample in samples
        if filter_start_ms is not None
        and filter_end_ms is not None
        and filter_start_ms <= sample.timestamp_ms <= filter_end_ms
    )
    if filter_start_ms is None or filter_end_ms is None:
        window_samples = ()

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
    gyro_x_values = _values_for_key(window_samples, "gyro.x")
    gyro_y_values = _values_for_key(window_samples, "gyro.y")
    gyro_z_values = _values_for_key(window_samples, "gyro.z")
    battery_values = _values_for_key(window_samples, "pm.vbat")
    radio_rssi_values = _values_for_key(window_samples, "radio.rssi")
    supervisor_values = [int(value) for value in _values_for_key(window_samples, "supervisor.info")]

    flags_seen = tuple(
        flag_name
        for flag_name, mask in _SUPERVISOR_FLAG_BITS
        if any((value & mask) != 0 for value in supervisor_values)
    )
    xy_drift_m: float | None = None
    if len(x_values) >= 2 and len(y_values) >= 2:
        xy_drift_m = math.sqrt((x_values[-1] - x_values[0]) ** 2 + (y_values[-1] - y_values[0]) ** 2)
    horizontal_speed_max_mps = None
    if vx_values and vy_values:
        horizontal_speed_max_mps = max(
            math.sqrt(vx_value * vx_value + vy_value * vy_value)
            for vx_value, vy_value in zip(vx_values, vy_values)
        )
    gyro_abs_max_dps = max(
        (abs(value) for value in (*gyro_x_values, *gyro_y_values, *gyro_z_values)),
        default=None,
    )
    clearance_observed = any(
        values for values in (front_values_m, back_values_m, left_values_m, right_values_m, up_values_m)
    )

    return HoverTelemetrySummary(
        sample_count=len(window_samples),
        available_blocks=available_blocks,
        skipped_blocks=skipped_blocks,
        duration_s=(
            max(0.0, (window_end_ms - window_start_ms) / 1000.0)
            if window_start_ms is not None and window_end_ms is not None
            else None
        ),
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
        flow_squal_mean=(sum(squal_values) / len(squal_values)) if squal_values else None,
        flow_nonzero_samples=sum(1 for value in squal_values if value > 0),
        flow_observed=any(value > 0 for value in squal_values),
        motion_delta_x_abs_max=max((abs(value) for value in delta_x_values), default=None),
        motion_delta_y_abs_max=max((abs(value) for value in delta_y_values), default=None),
        zrange_min_m=min(zrange_values_m) if zrange_values_m else None,
        zrange_max_m=max(zrange_values_m) if zrange_values_m else None,
        zrange_sample_count=len(zrange_values_m),
        zrange_observed=any(value > 0 for value in zrange_values_m),
        front_min_m=_series_min(front_values_m),
        back_min_m=_series_min(back_values_m),
        left_min_m=_series_min(left_values_m),
        right_min_m=_series_min(right_values_m),
        up_min_m=_series_min(up_values_m),
        clearance_observed=clearance_observed,
        thrust_mean=(sum(thrust_values) / len(thrust_values)) if thrust_values else None,
        thrust_max=_series_max(thrust_values),
        gyro_abs_max_dps=gyro_abs_max_dps,
        battery_min_v=min(battery_values) if battery_values else None,
        battery_drop_v=(battery_values[0] - battery_values[-1]) if len(battery_values) >= 2 else None,
        radio_rssi_latest_dbm=radio_rssi_values[-1] if radio_rssi_values else None,
        radio_rssi_min_dbm=_series_min(radio_rssi_values),
        latest_supervisor_info=supervisor_values[-1] if supervisor_values else None,
        supervisor_flags_seen=flags_seen,
        stable_supervisor=not flags_seen,
    )


def evaluate_hover_stability(summary: HoverTelemetrySummary, *, target_height_m: float) -> list[str]:
    """Return bounded stability failures derived from the captured telemetry."""

    failures: list[str] = []
    if summary.sample_count <= 0:
        failures.append("no in-flight telemetry samples were captured during the hover test")
    if not summary.flow_observed:
        failures.append("optical-flow quality never became nonzero during the hover test")
    if not summary.zrange_observed:
        failures.append("downward z-range never produced a nonzero reading during the hover test")
    if summary.supervisor_flags_seen:
        failures.append(
            f"supervisor reported unsafe flags during the hover test: {', '.join(summary.supervisor_flags_seen)}"
        )
    max_safe_altitude_m = max(
        HOVER_TELEMETRY_AIRBORNE_MIN_ALTITUDE_M,
        float(target_height_m) + HOVER_STABILITY_ALTITUDE_MARGIN_M,
    )
    if summary.zrange_max_m is not None and summary.zrange_max_m > max_safe_altitude_m:
        failures.append(
            "hover altitude reached "
            f"{summary.zrange_max_m:.2f} m which exceeds the {max_safe_altitude_m:.2f} m safety ceiling "
            f"for a {target_height_m:.2f} m hover"
        )
    if summary.battery_min_v is not None and summary.battery_min_v < HOVER_STABILITY_MIN_UNDER_LOAD_VBAT_V:
        failures.append(
            "battery sagged to "
            f"{summary.battery_min_v:.2f} V under load which is below the "
            f"{HOVER_STABILITY_MIN_UNDER_LOAD_VBAT_V:.2f} V hover safety floor"
        )
    return failures


def _import_cflib():
    """Import cflib lazily so repo-local tests can import this script."""

    import cflib.crtp  # type: ignore[import-not-found]  # pylint: disable=import-error
    from cflib.crazyflie import Crazyflie  # type: ignore[import-not-found]  # pylint: disable=import-error
    from cflib.crazyflie.log import LogConfig  # type: ignore[import-not-found]  # pylint: disable=import-error
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie  # type: ignore[import-not-found]  # pylint: disable=import-error
    from cflib.crazyflie.syncLogger import SyncLogger  # type: ignore[import-not-found]  # pylint: disable=import-error
    from cflib.utils.multiranger import Multiranger  # type: ignore[import-not-found]  # pylint: disable=import-error

    return cflib.crtp, Crazyflie, LogConfig, Multiranger, SyncCrazyflie, SyncLogger


def _read_deck_flags(sync_cf, deck_names: Iterable[str]) -> dict[str, int | None]:
    """Read the current deck presence flags from the Crazyflie firmware."""

    flags: dict[str, int | None] = {}
    for deck_name in deck_names:
        value: int | None
        try:
            raw = sync_cf.cf.param.get_value(f"deck.{deck_name}")
        except Exception:
            value = None
        else:
            try:
                value = int(str(raw).strip())
            except (TypeError, ValueError):
                value = None
        flags[deck_name] = value
    return flags


def _read_power_snapshot(sync_cf, log_config_cls, sync_logger_cls) -> HoverPowerSnapshot:
    """Read one bounded battery snapshot from the Crazyflie power logs."""

    config = log_config_cls(name="hover-power", period_in_ms=100)
    config.add_variable("pm.vbat", "float")
    config.add_variable("pm.batteryLevel", "uint8_t")
    config.add_variable("pm.state", "uint8_t")
    with sync_logger_cls(sync_cf, config) as logger:
        for _, data, _ in logger:
            return HoverPowerSnapshot(
                vbat_v=float(data.get("pm.vbat")) if data.get("pm.vbat") is not None else None,
                battery_level=int(data.get("pm.batteryLevel")) if data.get("pm.batteryLevel") is not None else None,
                state=int(data.get("pm.state")) if data.get("pm.state") is not None else None,
            )
    raise RuntimeError("hover-power logger yielded no samples")


def _normalize_clearance_value(raw: object) -> float | None:
    """Normalize one Multi-ranger reading into meters or ``None`` when invalid."""

    numeric = _normalize_numeric_value(raw)
    if numeric is None:
        return None
    value = float(numeric)
    if value <= 0.0:
        return None
    return value


def _read_clearance_snapshot(sync_cf, multiranger_cls) -> HoverClearanceSnapshot:
    """Read one short Multi-ranger snapshot for preflight clearance gating."""

    readings: dict[str, list[float]] = {
        "front": [],
        "back": [],
        "left": [],
        "right": [],
        "up": [],
        "down": [],
    }
    start = time.monotonic()
    with multiranger_cls(sync_cf) as multiranger:
        while time.monotonic() - start < HOVER_PREFLIGHT_CLEARANCE_SAMPLE_S:
            for direction_name in readings:
                value = _normalize_clearance_value(getattr(multiranger, direction_name))
                if value is not None:
                    readings[direction_name].append(value)
            time.sleep(HOVER_PREFLIGHT_CLEARANCE_PERIOD_S)
    return HoverClearanceSnapshot(
        front_m=min(readings["front"]) if readings["front"] else None,
        back_m=min(readings["back"]) if readings["back"] else None,
        left_m=min(readings["left"]) if readings["left"] else None,
        right_m=min(readings["right"]) if readings["right"] else None,
        up_m=min(readings["up"]) if readings["up"] else None,
        down_m=min(readings["down"]) if readings["down"] else None,
    )


def _latest_ground_distance_from_telemetry(telemetry: HoverTelemetryCollector | None) -> HoverGroundDistanceObservation:
    """Translate the latest telemetry ``range.zrange`` sample into a landing observation."""

    if telemetry is None:
        return HoverGroundDistanceObservation(distance_m=None, age_s=None)
    latest_value, age_s = telemetry.latest_value("range.zrange")
    supervisor_value, supervisor_age_s = telemetry.latest_value("supervisor.info")
    is_flying: bool | None = None
    if supervisor_value is not None:
        is_flying = (int(supervisor_value) & _SUPERVISOR_IS_FLYING_MASK) != 0
    if latest_value is None:
        return HoverGroundDistanceObservation(
            distance_m=None,
            age_s=None,
            is_flying=is_flying,
            supervisor_age_s=supervisor_age_s,
        )
    numeric = float(latest_value)
    if numeric <= 0.0 or numeric >= HOVER_RANGE_INVALID_MM:
        return HoverGroundDistanceObservation(
            distance_m=None,
            age_s=age_s,
            is_flying=is_flying,
            supervisor_age_s=supervisor_age_s,
        )
    return HoverGroundDistanceObservation(
        distance_m=(numeric / 1000.0),
        age_s=age_s,
        is_flying=is_flying,
        supervisor_age_s=supervisor_age_s,
    )


def run_hover_test(
    *,
    uri: str,
    workspace: Path,
    height_m: float,
    hover_duration_s: float,
    takeoff_velocity_mps: float,
    land_velocity_mps: float,
    connect_settle_s: float,
    min_vbat_v: float,
    min_battery_level: int,
    min_clearance_m: float,
    stabilizer_estimator: int,
    stabilizer_controller: int,
    motion_disable: int,
    estimator_settle_timeout_s: float,
    on_device_failsafe_mode: str,
    on_device_failsafe_heartbeat_timeout_s: float,
    on_device_failsafe_low_battery_v: float,
    on_device_failsafe_critical_battery_v: float,
    on_device_failsafe_min_up_clearance_m: float,
    required_decks: tuple[str, ...],
    trace_writer: HoverWorkerTraceWriter | None = None,
) -> HoverTestReport:
    """Connect, preflight, and run one bounded hover test."""

    trace = trace_writer or HoverWorkerTraceWriter(None)
    trace.emit(
        "run_hover_test",
        status="begin",
        data={
            "uri": uri,
            "workspace": str(workspace),
            "height_m": height_m,
            "hover_duration_s": hover_duration_s,
        },
    )
    trace.emit("cflib_import", status="begin")
    (
        crtp,
        crazyflie_cls,
        log_config_cls,
        multiranger_cls,
        sync_crazyflie_cls,
        sync_logger_cls,
    ) = _import_cflib()
    trace.emit("cflib_import", status="done")
    workspace = workspace.expanduser().resolve()
    cache_dir = workspace / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    trace.emit("workspace_ready", status="done", data={"workspace": str(workspace), "cache_dir": str(cache_dir)})

    trace.emit("crtp_init", status="begin")
    crtp.init_drivers()
    trace.emit("crtp_init", status="done")
    crazyflie = crazyflie_cls(rw_cache=str(cache_dir))
    sync_context = sync_crazyflie_cls(uri, cf=crazyflie)
    sync_cf = None
    entered_sync_context = False
    deck_flags: dict[str, int | None] = {}
    power = HoverPowerSnapshot(vbat_v=None, battery_level=None, state=None)
    clearance_snapshot: HoverClearanceSnapshot | None = None
    pre_arm_snapshot: HoverPreArmSnapshot | None = None
    estimator_settle: HoverEstimatorSettlingReport | None = None
    primitive_outcome: HoverPrimitiveOutcome | None = None
    telemetry_samples: tuple[HoverTelemetrySample, ...] = ()
    telemetry_summary: HoverTelemetrySummary | None = None
    stability_failures: list[str] = []
    on_device_failsafe_report: OnDeviceFailsafeSessionReport | None = None
    took_off = False
    landed = False
    primitive: StatefulHoverPrimitive | None = None
    telemetry = None
    on_device_failsafe_session: OnDeviceFailsafeHeartbeatSession | None = None

    def _stop_telemetry() -> None:
        if telemetry is None:
            return
        trace.emit("telemetry_stop", status="begin")
        telemetry.stop()
        trace.emit("telemetry_stop", status="done")

    def _snapshot_telemetry() -> tuple[HoverTelemetrySample, ...]:
        trace.emit("telemetry_snapshot", status="begin")
        samples = telemetry.snapshot() if telemetry is not None else ()
        trace.emit("telemetry_snapshot", status="done", data={"sample_count": len(samples)})
        return samples

    try:
        trace.emit("sync_connect", status="begin", data={"uri": uri})
        sync_cf = sync_context.__enter__()
        entered_sync_context = True
        trace.emit("sync_connect", status="done")
        if connect_settle_s > 0:
            trace.emit("connect_settle", status="begin", data={"sleep_s": connect_settle_s})
            time.sleep(connect_settle_s)
            trace.emit("connect_settle", status="done")
        trace.emit("deck_flags", status="begin")
        deck_flags = _read_deck_flags(sync_cf, DECK_PARAM_NAMES)
        trace.emit("deck_flags", status="done", data={"deck_flags": deck_flags})
        if deck_flags.get("bcMultiranger") == 1:
            trace.emit("clearance_snapshot", status="begin")
            clearance_snapshot = _read_clearance_snapshot(sync_cf, multiranger_cls)
            trace.emit(
                "clearance_snapshot",
                status="done",
                data=asdict(clearance_snapshot),
            )
        else:
            trace.emit("clearance_snapshot", status="skipped", data={"reason": "multiranger_not_detected"})
        trace.emit("power_snapshot", status="begin")
        power = _read_power_snapshot(sync_cf, log_config_cls, sync_logger_cls)
        trace.emit(
            "power_snapshot",
            status="done",
            data={
                "vbat_v": power.vbat_v,
                "battery_level": power.battery_level,
                "state": power.state,
            },
        )
        failures = evaluate_hover_preflight(
            deck_flags=deck_flags,
            required_decks=required_decks,
            power=power,
            clearance_snapshot=clearance_snapshot,
            min_vbat_v=min_vbat_v,
            min_battery_level=min_battery_level,
            min_clearance_m=min_clearance_m,
        )
        if failures:
            trace.emit("preflight", status="blocked", data={"failures": failures})
            report = HoverTestReport(
                uri=uri,
                workspace=str(workspace),
                height_m=height_m,
                hover_duration_s=hover_duration_s,
                takeoff_velocity_mps=takeoff_velocity_mps,
                land_velocity_mps=land_velocity_mps,
                connect_settle_s=connect_settle_s,
                min_vbat_v=min_vbat_v,
                min_battery_level=min_battery_level,
                min_clearance_m=min_clearance_m,
                deck_flags=deck_flags,
                required_decks=required_decks,
                clearance_snapshot=clearance_snapshot,
                pre_arm_snapshot=pre_arm_snapshot,
                estimator_settle=estimator_settle,
                power=power,
                status="blocked",
                completed=False,
                landed=False,
                interrupted=False,
                primitive_outcome=primitive_outcome,
                telemetry=(),
                telemetry_summary=None,
                failures=tuple(failures),
                recommendations=(),
                on_device_failsafe=on_device_failsafe_report,
            )
            trace.emit("report_build", status="begin", data={"status": "blocked"})
            final_report = replace(report, recommendations=recommendations_for_report(report))
            trace.emit("report_build", status="done", data={"status": final_report.status})
            return final_report

        trace.emit("preflight", status="done")
        pre_arm_snapshot = apply_hover_pre_arm(
            sync_cf,
            config=HoverPreArmConfig(
                estimator=stabilizer_estimator,
                controller=stabilizer_controller,
                motion_disable=motion_disable,
            ),
            trace_writer=trace,
        )
        if pre_arm_snapshot.failures:
            report = HoverTestReport(
                uri=uri,
                workspace=str(workspace),
                height_m=height_m,
                hover_duration_s=hover_duration_s,
                takeoff_velocity_mps=takeoff_velocity_mps,
                land_velocity_mps=land_velocity_mps,
                connect_settle_s=connect_settle_s,
                min_vbat_v=min_vbat_v,
                min_battery_level=min_battery_level,
                min_clearance_m=min_clearance_m,
                deck_flags=deck_flags,
                required_decks=required_decks,
                clearance_snapshot=clearance_snapshot,
                pre_arm_snapshot=pre_arm_snapshot,
                estimator_settle=estimator_settle,
                power=power,
                status="blocked",
                completed=False,
                landed=False,
                interrupted=False,
                primitive_outcome=primitive_outcome,
                telemetry=(),
                telemetry_summary=None,
                failures=pre_arm_snapshot.failures,
                recommendations=(),
                on_device_failsafe=on_device_failsafe_report,
            )
            trace.emit("report_build", status="begin", data={"status": "blocked"})
            final_report = replace(report, recommendations=recommendations_for_report(report))
            trace.emit("report_build", status="done", data={"status": final_report.status})
            return final_report

        estimator_settle = wait_for_estimator_settle(
            sync_cf,
            log_config_cls,
            sync_logger_cls,
            config=HoverEstimatorSettlingConfig(timeout_s=estimator_settle_timeout_s),
            trace_writer=trace,
        )
        if not estimator_settle.stable:
            report = HoverTestReport(
                uri=uri,
                workspace=str(workspace),
                height_m=height_m,
                hover_duration_s=hover_duration_s,
                takeoff_velocity_mps=takeoff_velocity_mps,
                land_velocity_mps=land_velocity_mps,
                connect_settle_s=connect_settle_s,
                min_vbat_v=min_vbat_v,
                min_battery_level=min_battery_level,
                min_clearance_m=min_clearance_m,
                deck_flags=deck_flags,
                required_decks=required_decks,
                clearance_snapshot=clearance_snapshot,
                pre_arm_snapshot=pre_arm_snapshot,
                estimator_settle=estimator_settle,
                power=power,
                status="blocked",
                completed=False,
                landed=False,
                interrupted=False,
                primitive_outcome=primitive_outcome,
                telemetry=(),
                telemetry_summary=None,
                failures=estimator_settle.failures,
                recommendations=(),
                on_device_failsafe=on_device_failsafe_report,
            )
            trace.emit("report_build", status="begin", data={"status": "blocked"})
            final_report = replace(report, recommendations=recommendations_for_report(report))
            trace.emit("report_build", status="done", data={"status": final_report.status})
            return final_report

        if on_device_failsafe_mode != "off":
            trace.emit(
                "on_device_failsafe_probe",
                status="begin",
                data={"mode": on_device_failsafe_mode},
            )
            availability = probe_on_device_failsafe(sync_cf)
            trace.emit(
                "on_device_failsafe_probe",
                status="done",
                data={
                    "loaded": availability.loaded,
                    "protocol_version": availability.protocol_version,
                    "state": availability.state_name,
                    "reason": availability.reason_name,
                },
            )
            on_device_failsafe_config = OnDeviceFailsafeConfig(
                heartbeat_timeout_s=on_device_failsafe_heartbeat_timeout_s,
                low_battery_v=on_device_failsafe_low_battery_v,
                critical_battery_v=on_device_failsafe_critical_battery_v,
                min_clearance_m=min_clearance_m,
                min_up_clearance_m=on_device_failsafe_min_up_clearance_m,
            )
            on_device_failsafe_report = OnDeviceFailsafeSessionReport(
                mode=on_device_failsafe_mode,
                config=on_device_failsafe_config,
                availability=availability,
                session_id=None,
                started=False,
                disabled_cleanly=False,
                packets_sent=0,
                status_packets_received=0,
                last_status=None,
                failures=availability.failures,
            )
            if not availability.loaded and on_device_failsafe_mode == "required":
                report = HoverTestReport(
                    uri=uri,
                    workspace=str(workspace),
                    height_m=height_m,
                    hover_duration_s=hover_duration_s,
                    takeoff_velocity_mps=takeoff_velocity_mps,
                    land_velocity_mps=land_velocity_mps,
                    connect_settle_s=connect_settle_s,
                    min_vbat_v=min_vbat_v,
                    min_battery_level=min_battery_level,
                    min_clearance_m=min_clearance_m,
                    deck_flags=deck_flags,
                    required_decks=required_decks,
                    clearance_snapshot=clearance_snapshot,
                    pre_arm_snapshot=pre_arm_snapshot,
                    estimator_settle=estimator_settle,
                    power=power,
                    status="blocked",
                    completed=False,
                    landed=False,
                    interrupted=False,
                    primitive_outcome=primitive_outcome,
                    telemetry=(),
                    telemetry_summary=None,
                    failures=(
                        "required on-device failsafe app `twinrFs` is not loaded on the Crazyflie firmware",
                    ),
                    recommendations=(),
                    on_device_failsafe=on_device_failsafe_report,
                )
                trace.emit("report_build", status="begin", data={"status": "blocked"})
                final_report = replace(report, recommendations=recommendations_for_report(report))
                trace.emit("report_build", status="done", data={"status": final_report.status})
                return final_report
            if availability.loaded:
                on_device_failsafe_session = OnDeviceFailsafeHeartbeatSession(
                    sync_cf,
                    mode=on_device_failsafe_mode,
                    config=on_device_failsafe_config,
                    availability=availability,
                    trace_writer=trace,
                )
                on_device_failsafe_session.start()
                on_device_failsafe_report = on_device_failsafe_session.report()

        telemetry = HoverTelemetryCollector(sync_cf, log_config_cls)
        trace.emit("telemetry_start", status="begin")
        telemetry.start()
        trace.emit(
            "telemetry_start",
            status="done",
            data={
                "available_blocks": telemetry.available_blocks,
                "skipped_blocks": telemetry.skipped_blocks,
            },
        )
        if HOVER_TELEMETRY_STARTUP_SETTLE_S > 0:
            trace.emit("telemetry_settle", status="begin", data={"sleep_s": HOVER_TELEMETRY_STARTUP_SETTLE_S})
            time.sleep(HOVER_TELEMETRY_STARTUP_SETTLE_S)
            trace.emit("telemetry_settle", status="done")
        trace.emit("hover_primitive_create", status="begin")
        primitive = StatefulHoverPrimitive(
            sync_cf,
            ground_distance_provider=lambda: _latest_ground_distance_from_telemetry(telemetry),
            trace_writer=trace,
        )
        trace.emit("hover_primitive_create", status="done")
        primitive_outcome = primitive.run(
            HoverPrimitiveConfig(
                target_height_m=height_m,
                hover_duration_s=hover_duration_s,
                takeoff_velocity_mps=takeoff_velocity_mps,
                land_velocity_mps=land_velocity_mps,
            )
        )
        took_off = primitive_outcome.took_off
        landed = primitive_outcome.landed
        _stop_telemetry()
    except KeyboardInterrupt:
        trace.emit("worker_interrupt", status="error", message="KeyboardInterrupt")
        try:
            _stop_telemetry()
        except Exception as exc:
            trace.emit(
                "telemetry_stop",
                status="error",
                message=f"{exc.__class__.__name__}:{exc}",
            )
        telemetry_samples = _snapshot_telemetry()
        trace.emit("telemetry_summary", status="begin")
        telemetry_summary = summarize_hover_telemetry(
            telemetry_samples,
            available_blocks=telemetry.available_blocks if telemetry is not None else (),
            skipped_blocks=telemetry.skipped_blocks if telemetry is not None else (),
        )
        trace.emit(
            "telemetry_summary",
            status="done",
            data={"sample_count": telemetry_summary.sample_count},
        )
        if primitive is not None:
            took_off = took_off or primitive.took_off
            landed = landed or primitive.landed
        if on_device_failsafe_session is not None:
            on_device_failsafe_report = on_device_failsafe_session.report()
        report = HoverTestReport(
            uri=uri,
            workspace=str(workspace),
            height_m=height_m,
            hover_duration_s=hover_duration_s,
            takeoff_velocity_mps=takeoff_velocity_mps,
            land_velocity_mps=land_velocity_mps,
            connect_settle_s=connect_settle_s,
            min_vbat_v=min_vbat_v,
            min_battery_level=min_battery_level,
            min_clearance_m=min_clearance_m,
            deck_flags=deck_flags,
            required_decks=required_decks,
            clearance_snapshot=clearance_snapshot,
            pre_arm_snapshot=pre_arm_snapshot,
            estimator_settle=estimator_settle,
            power=power,
            status="interrupted",
            completed=False,
            landed=landed,
            interrupted=True,
            primitive_outcome=primitive_outcome,
            telemetry=telemetry_samples,
            telemetry_summary=telemetry_summary,
            failures=("hover test interrupted; landing requested",),
            recommendations=(),
            on_device_failsafe=on_device_failsafe_report,
        )
        trace.emit("report_build", status="begin", data={"status": "interrupted"})
        final_report = replace(report, recommendations=recommendations_for_report(report))
        trace.emit("report_build", status="done", data={"status": final_report.status})
        return final_report
    except Exception as exc:
        trace.emit(
            "run_hover_test_exception",
            status="error",
            message=f"{exc.__class__.__name__}:{exc}",
        )
        try:
            _stop_telemetry()
        except Exception as stop_exc:
            trace.emit(
                "telemetry_stop",
                status="error",
                message=f"{stop_exc.__class__.__name__}:{stop_exc}",
            )
        if primitive is not None:
            took_off = took_off or primitive.took_off
            landed = landed or primitive.landed
        raise
    finally:
        if on_device_failsafe_session is not None:
            disable_on_close = bool(landed or not took_off)
            trace.emit(
                "on_device_failsafe_release",
                status="begin",
                data={"disable_on_close": disable_on_close, "took_off": took_off, "landed": landed},
            )
            on_device_failsafe_session.close(disable=disable_on_close)
            on_device_failsafe_report = on_device_failsafe_session.report()
            trace.emit(
                "on_device_failsafe_release",
                status="done",
                data={
                    "disabled_cleanly": on_device_failsafe_report.disabled_cleanly,
                    "status_packets_received": on_device_failsafe_report.status_packets_received,
                },
            )
        if entered_sync_context:
            trace.emit("sync_disconnect", status="begin")
            try:
                suppressed = bool(sync_context.__exit__(*sys.exc_info()))
            except BaseException as exc:  # pragma: no cover - live teardown path
                trace.emit(
                    "sync_disconnect",
                    status="error",
                    message=f"{exc.__class__.__name__}:{exc}",
                )
                raise
            else:
                trace.emit("sync_disconnect", status="done", data={"suppressed": suppressed})

    telemetry_samples = _snapshot_telemetry()
    trace.emit("telemetry_summary", status="begin")
    telemetry_summary = summarize_hover_telemetry(
        telemetry_samples,
        available_blocks=telemetry.available_blocks if telemetry is not None else (),
        skipped_blocks=telemetry.skipped_blocks if telemetry is not None else (),
    )
    trace.emit(
        "telemetry_summary",
        status="done",
        data={
            "sample_count": telemetry_summary.sample_count,
            "flow_observed": telemetry_summary.flow_observed,
            "zrange_observed": telemetry_summary.zrange_observed,
            "stable_supervisor": telemetry_summary.stable_supervisor,
        },
    )
    stability_failures = evaluate_hover_stability(telemetry_summary, target_height_m=height_m)
    trace.emit("stability_eval", status="done", data={"failures": stability_failures})

    report = HoverTestReport(
        uri=uri,
        workspace=str(workspace),
        height_m=height_m,
        hover_duration_s=hover_duration_s,
        takeoff_velocity_mps=takeoff_velocity_mps,
        land_velocity_mps=land_velocity_mps,
        connect_settle_s=connect_settle_s,
        min_vbat_v=min_vbat_v,
        min_battery_level=min_battery_level,
        min_clearance_m=min_clearance_m,
        deck_flags=deck_flags,
        required_decks=required_decks,
        clearance_snapshot=clearance_snapshot,
        pre_arm_snapshot=pre_arm_snapshot,
        estimator_settle=estimator_settle,
        power=power,
        status="completed" if not stability_failures else "unstable",
        completed=not stability_failures,
        landed=True,
        interrupted=False,
        primitive_outcome=primitive_outcome,
        telemetry=telemetry_samples,
        telemetry_summary=telemetry_summary,
        failures=tuple(stability_failures),
        recommendations=(),
        on_device_failsafe=on_device_failsafe_report,
    )
    trace.emit("report_build", status="begin", data={"status": report.status})
    final_report = replace(report, recommendations=recommendations_for_report(report))
    trace.emit("report_build", status="done", data={"status": final_report.status})
    return final_report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a bounded Crazyflie hover test.")
    parser.add_argument("--uri", default=DEFAULT_URI, help="Crazyflie radio URI (default: radio://0/80/2M)")
    parser.add_argument("--workspace", default="/twinr/bitcraze", help="Bitcraze workspace root for cflib cache files")
    parser.add_argument("--height-m", type=float, default=0.25, help="Hover height in meters (default: 0.25)")
    parser.add_argument("--hover-duration-s", type=float, default=3.0, help="Hover hold time in seconds (default: 3.0)")
    parser.add_argument(
        "--takeoff-velocity-mps",
        type=float,
        default=0.2,
        help="Takeoff velocity in meters per second (default: 0.2)",
    )
    parser.add_argument(
        "--land-velocity-mps",
        type=float,
        default=0.2,
        help="Landing velocity in meters per second (default: 0.2)",
    )
    parser.add_argument(
        "--connect-settle-s",
        type=float,
        default=1.0,
        help="Initial wait after connect before preflight and takeoff (default: 1.0)",
    )
    parser.add_argument(
        "--min-vbat-v",
        type=float,
        default=3.8,
        help="Minimum battery voltage for hover acceptance (default: 3.8)",
    )
    parser.add_argument(
        "--min-battery-level",
        type=int,
        default=20,
        help="Minimum battery level percent for hover acceptance (default: 20)",
    )
    parser.add_argument(
        "--min-clearance-m",
        type=float,
        default=HOVER_DEFAULT_MIN_CLEARANCE_M,
        help="Minimum observed Multi-ranger clearance in meters when that deck is present (default: 0.35)",
    )
    parser.add_argument(
        "--stabilizer-estimator",
        type=int,
        default=HOVER_DEFAULT_ESTIMATOR,
        help="Deterministic stabilizer.estimator value for hover pre-arm (default: 2)",
    )
    parser.add_argument(
        "--stabilizer-controller",
        type=int,
        default=HOVER_DEFAULT_CONTROLLER,
        help="Deterministic stabilizer.controller value for hover pre-arm (default: 1)",
    )
    parser.add_argument(
        "--motion-disable",
        type=int,
        default=HOVER_DEFAULT_MOTION_DISABLE,
        help="Deterministic motion.disable value for hover pre-arm (default: 0)",
    )
    parser.add_argument(
        "--estimator-settle-timeout-s",
        type=float,
        default=HoverEstimatorSettlingConfig().timeout_s,
        help="Maximum seconds to wait for Kalman settle before takeoff (default: 5.0)",
    )
    parser.add_argument(
        "--on-device-failsafe-mode",
        choices=("required", "preferred", "off"),
        default=HOVER_DEFAULT_ON_DEVICE_FAILSAFE_MODE,
        help="Whether the hover worker requires the Twinr firmware failsafe app (default: required).",
    )
    parser.add_argument(
        "--on-device-failsafe-heartbeat-timeout-s",
        type=float,
        default=ON_DEVICE_FAILSAFE_HEARTBEAT_TIMEOUT_S,
        help="Firmware heartbeat timeout for the on-device failsafe in seconds (default: 0.35).",
    )
    parser.add_argument(
        "--on-device-failsafe-low-battery-v",
        type=float,
        default=ON_DEVICE_FAILSAFE_LOW_BATTERY_V,
        help="Firmware low-battery safe-land threshold in volts (default: 3.55).",
    )
    parser.add_argument(
        "--on-device-failsafe-critical-battery-v",
        type=float,
        default=ON_DEVICE_FAILSAFE_CRITICAL_BATTERY_V,
        help="Firmware critical-battery safe-land threshold in volts (default: 3.35).",
    )
    parser.add_argument(
        "--on-device-failsafe-min-up-clearance-m",
        type=float,
        default=ON_DEVICE_FAILSAFE_MIN_UP_CLEARANCE_M,
        help="Firmware upward-clearance trigger in meters (default: 0.25).",
    )
    parser.add_argument(
        "--require-deck",
        action="append",
        default=[],
        help="Require one deck flag (examples: flow2, zranger2, multiranger, aideck); repeat as needed",
    )
    parser.add_argument(
        "--trace-file",
        default="",
        help="Optional JSONL trace path for worker-phase diagnostics and timeout forensics.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the full report as JSON")
    return parser


def _print_human_report(report: HoverTestReport) -> None:
    """Print one compact human-readable hover-test report."""

    print(f"status={report.status}")
    print(f"uri={report.uri}")
    print(f"workspace={report.workspace}")
    print(f"height_m={report.height_m}")
    print(f"hover_duration_s={report.hover_duration_s}")
    print(f"power.vbat_v={report.power.vbat_v}")
    print(f"power.battery_level={report.power.battery_level}")
    print(f"power.state={report.power.state}")
    print(f"min_clearance_m={report.min_clearance_m}")
    if report.clearance_snapshot is not None:
        print(f"clearance.front_m={report.clearance_snapshot.front_m}")
        print(f"clearance.back_m={report.clearance_snapshot.back_m}")
        print(f"clearance.left_m={report.clearance_snapshot.left_m}")
        print(f"clearance.right_m={report.clearance_snapshot.right_m}")
        print(f"clearance.up_m={report.clearance_snapshot.up_m}")
        print(f"clearance.down_m={report.clearance_snapshot.down_m}")
    if report.pre_arm_snapshot is not None:
        print(f"pre_arm.estimator={report.pre_arm_snapshot.estimator}")
        print(f"pre_arm.controller={report.pre_arm_snapshot.controller}")
        print(f"pre_arm.motion_disable={report.pre_arm_snapshot.motion_disable}")
        print(f"pre_arm.kalman_reset_after={report.pre_arm_snapshot.kalman_reset_after}")
        print(f"pre_arm.verified={str(report.pre_arm_snapshot.verified).lower()}")
    if report.estimator_settle is not None:
        print(f"settle.stable={str(report.estimator_settle.stable).lower()}")
        print(f"settle.sample_count={report.estimator_settle.sample_count}")
        print(f"settle.var_px_span={report.estimator_settle.var_px_span}")
        print(f"settle.var_py_span={report.estimator_settle.var_py_span}")
        print(f"settle.var_pz_span={report.estimator_settle.var_pz_span}")
        print(f"settle.motion_squal_mean={report.estimator_settle.motion_squal_mean}")
        print(f"settle.motion_squal_nonzero_ratio={report.estimator_settle.motion_squal_nonzero_ratio}")
        print(f"settle.zrange_min_m={report.estimator_settle.zrange_min_m}")
    if report.primitive_outcome is not None:
        print(f"primitive.final_phase={report.primitive_outcome.final_phase}")
        print(f"primitive.took_off={str(report.primitive_outcome.took_off).lower()}")
        print(f"primitive.landed={str(report.primitive_outcome.landed).lower()}")
        print(f"primitive.commanded_max_height_m={report.primitive_outcome.commanded_max_height_m}")
        print(f"primitive.setpoint_count={report.primitive_outcome.setpoint_count}")
    if report.on_device_failsafe is not None:
        print(f"on_device_failsafe.mode={report.on_device_failsafe.mode}")
        print(f"on_device_failsafe.loaded={str(report.on_device_failsafe.availability.loaded).lower()}")
        print(f"on_device_failsafe.started={str(report.on_device_failsafe.started).lower()}")
        print(f"on_device_failsafe.disabled_cleanly={str(report.on_device_failsafe.disabled_cleanly).lower()}")
        print(f"on_device_failsafe.status_packets_received={report.on_device_failsafe.status_packets_received}")
        if report.on_device_failsafe.last_status is not None:
            print(f"on_device_failsafe.last_state={report.on_device_failsafe.last_status.state_name}")
            print(f"on_device_failsafe.last_reason={report.on_device_failsafe.last_status.reason_name}")
            print(f"on_device_failsafe.last_vbat_mv={report.on_device_failsafe.last_status.vbat_mv}")
            print(
                f"on_device_failsafe.last_min_clearance_mm={report.on_device_failsafe.last_status.min_clearance_mm}"
            )
    if report.telemetry_summary is not None:
        print(f"telemetry.sample_count={report.telemetry_summary.sample_count}")
        print(f"telemetry.available_blocks={','.join(report.telemetry_summary.available_blocks)}")
        if report.telemetry_summary.skipped_blocks:
            print(f"telemetry.skipped_blocks={','.join(report.telemetry_summary.skipped_blocks)}")
        print(f"telemetry.flow_observed={str(report.telemetry_summary.flow_observed).lower()}")
        print(f"telemetry.zrange_observed={str(report.telemetry_summary.zrange_observed).lower()}")
        print(f"telemetry.xy_drift_m={report.telemetry_summary.xy_drift_m}")
        print(f"telemetry.z_drift_m={report.telemetry_summary.z_drift_m}")
        print(f"telemetry.z_span_m={report.telemetry_summary.z_span_m}")
        print(f"telemetry.horizontal_speed_max_mps={report.telemetry_summary.horizontal_speed_max_mps}")
        print(f"telemetry.flow_squal_mean={report.telemetry_summary.flow_squal_mean}")
        print(f"telemetry.motion_delta_x_abs_max={report.telemetry_summary.motion_delta_x_abs_max}")
        print(f"telemetry.motion_delta_y_abs_max={report.telemetry_summary.motion_delta_y_abs_max}")
        print(f"telemetry.front_min_m={report.telemetry_summary.front_min_m}")
        print(f"telemetry.back_min_m={report.telemetry_summary.back_min_m}")
        print(f"telemetry.left_min_m={report.telemetry_summary.left_min_m}")
        print(f"telemetry.right_min_m={report.telemetry_summary.right_min_m}")
        print(f"telemetry.up_min_m={report.telemetry_summary.up_min_m}")
        print(f"telemetry.thrust_max={report.telemetry_summary.thrust_max}")
        print(f"telemetry.gyro_abs_max_dps={report.telemetry_summary.gyro_abs_max_dps}")
        print(f"telemetry.radio_rssi_latest_dbm={report.telemetry_summary.radio_rssi_latest_dbm}")
        print(f"telemetry.battery_drop_v={report.telemetry_summary.battery_drop_v}")
        if report.telemetry_summary.supervisor_flags_seen:
            print(f"telemetry.supervisor_flags={','.join(report.telemetry_summary.supervisor_flags_seen)}")
    for deck_name, flag in sorted(report.deck_flags.items()):
        print(f"deck.{deck_name}={flag if flag is not None else 'unknown'}")
    for failure in report.failures:
        print(f"failure={failure}")
    for recommendation in report.recommendations:
        print(f"recommendation={recommendation}")


def main() -> int:
    """Parse args, run one hover test, and emit the bounded result."""

    args = _build_parser().parse_args()
    trace_writer = HoverWorkerTraceWriter(
        Path(str(args.trace_file).strip()) if str(args.trace_file).strip() else None
    )
    trace_writer.emit("main", status="begin", data={"json": bool(args.json)})
    required_decks = tuple(
        normalize_required_deck_name(name) for name in (args.require_deck or []) if str(name or "").strip()
    ) or DEFAULT_REQUIRED_DECKS
    try:
        report = run_hover_test(
            uri=str(args.uri).strip() or DEFAULT_URI,
            workspace=Path(args.workspace),
            height_m=max(0.1, float(args.height_m)),
            hover_duration_s=max(1.0, float(args.hover_duration_s)),
            takeoff_velocity_mps=max(0.05, float(args.takeoff_velocity_mps)),
            land_velocity_mps=max(0.05, float(args.land_velocity_mps)),
            connect_settle_s=max(0.0, float(args.connect_settle_s)),
            min_vbat_v=max(0.0, float(args.min_vbat_v)),
            min_battery_level=max(0, int(args.min_battery_level)),
            min_clearance_m=max(0.0, float(args.min_clearance_m)),
            stabilizer_estimator=int(args.stabilizer_estimator),
            stabilizer_controller=int(args.stabilizer_controller),
            motion_disable=int(args.motion_disable),
            estimator_settle_timeout_s=max(0.5, float(args.estimator_settle_timeout_s)),
            on_device_failsafe_mode=str(args.on_device_failsafe_mode).strip() or HOVER_DEFAULT_ON_DEVICE_FAILSAFE_MODE,
            on_device_failsafe_heartbeat_timeout_s=max(
                0.05,
                float(args.on_device_failsafe_heartbeat_timeout_s),
            ),
            on_device_failsafe_low_battery_v=max(0.0, float(args.on_device_failsafe_low_battery_v)),
            on_device_failsafe_critical_battery_v=max(0.0, float(args.on_device_failsafe_critical_battery_v)),
            on_device_failsafe_min_up_clearance_m=max(
                0.0,
                float(args.on_device_failsafe_min_up_clearance_m),
            ),
            required_decks=required_decks,
            trace_writer=trace_writer,
        )
    except Exception as exc:
        failure_message = f"hover_test_exception:{exc.__class__.__name__}:{exc}"
        failure_payload = {
            "report": None,
            "failures": [failure_message],
        }
        trace_writer.emit("main", status="error", message=failure_message)
        if args.json:
            trace_writer.emit("json_emit", status="begin", data={"kind": "failure"})
            print(json.dumps(failure_payload, indent=2, sort_keys=True))
            trace_writer.emit("json_emit", status="done", data={"kind": "failure"})
        else:
            print(f"failure={failure_message}")
        trace_writer.emit("main", status="done", data={"exit_code": 1})
        return 1

    if args.json:
        trace_writer.emit("json_emit", status="begin", data={"kind": "success", "status": report.status})
        print(json.dumps({"report": asdict(report), "failures": list(report.failures)}, indent=2, sort_keys=True))
        trace_writer.emit("json_emit", status="done", data={"kind": "success", "status": report.status})
    else:
        _print_human_report(report)
    exit_code = 1
    if report.status == "completed":
        exit_code = 0
    elif report.status == "interrupted":
        exit_code = 130
    trace_writer.emit("main", status="done", data={"exit_code": exit_code, "report_status": report.status})
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
