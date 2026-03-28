#!/usr/bin/env python3
# CHANGELOG: 2026-03-27
# BUG-1: start() is now transactional and removes the Appchannel callback again if the first heartbeat or worker startup fails.
# BUG-2: optional trace_writer failures can no longer abort startup, heartbeat transmission, or shutdown; tracing is fail-open.
# BUG-3: invalid threshold combinations (for example timeout <= period or low_battery <= critical_battery) now fail fast instead of silently producing bad firmware config.
# SEC-1: session ids are now generated with secrets.randbelow() instead of predictable time-based low bits.
# SEC-2: # BREAKING: start() now refuses to arm the host heartbeat session when probe_on_device_failsafe() did not verify the firmware app.
# IMP-1: the heartbeat worker now uses monotonic deadline scheduling, drift compensation, and deadline-miss accounting instead of simple periodic waits.
# IMP-2: optional cflib link_statistics integration records latency/link quality/RSSI/rates/congestion on current stable cflib when available.
# IMP-3: exact packet-size parsing, context-manager support, wait_for_status(), richer reports, and stricter Appchannel MTU guards improve deployment diagnostics.

"""Drive Twinr's on-device Crazyflie failsafe over the Appchannel.

Purpose
-------
Keep the host-side part of the on-device failsafe intentionally small. This
module only proves that the firmware app is present, sends bounded heartbeat
packets with the current failsafe thresholds, and records status packets sent
back by the app. The actual safety-critical land/avoid controller lives on the
Crazyflie in C; Python only configures and observes it.

Frontier notes
--------------
This host module intentionally stays on the stable cflib API surface. It adds
fail-closed startup, validated configuration, deadline-based heartbeat
scheduling, and optional link-performance observability, while leaving the
actual safety controller on-device where communication dropouts cannot remove
the safety path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import secrets
import struct
import threading
import time
from typing import Any, Callable, Mapping


ON_DEVICE_FAILSAFE_PROTOCOL_VERSION = 1
ON_DEVICE_FAILSAFE_PACKET_KIND_HEARTBEAT = 1
ON_DEVICE_FAILSAFE_PACKET_KIND_STATUS = 2
ON_DEVICE_FAILSAFE_FLAG_ENABLE = 1 << 0
ON_DEVICE_FAILSAFE_FLAG_REQUIRE_CLEARANCE = 1 << 1
ON_DEVICE_FAILSAFE_HEARTBEAT_PERIOD_S = 0.1
ON_DEVICE_FAILSAFE_HEARTBEAT_TIMEOUT_S = 0.35
ON_DEVICE_FAILSAFE_LOW_BATTERY_V = 3.55
ON_DEVICE_FAILSAFE_CRITICAL_BATTERY_V = 3.35
ON_DEVICE_FAILSAFE_MIN_CLEARANCE_M = 0.35
ON_DEVICE_FAILSAFE_MIN_UP_CLEARANCE_M = 0.25
ON_DEVICE_FAILSAFE_DESCENT_RATE_MPS = 0.12
ON_DEVICE_FAILSAFE_MAX_REPEL_VELOCITY_MPS = 0.15
ON_DEVICE_FAILSAFE_BRAKE_HOLD_S = 0.20
ON_DEVICE_FAILSAFE_APPCHANNEL_MTU = 30

_UINT16_MAX = 0xFFFF
_HEARTBEAT_STRUCT = struct.Struct("<BBBBHHHHHHHHH")
_STATUS_STRUCT = struct.Struct("<BBBBHHHHH")
_AVAILABLE_PARAM_NAMES = (
    "twinrFs.protocolVersion",
    "twinrFs.enable",
    "twinrFs.state",
    "twinrFs.reason",
)
_STATE_NAMES = {
    0: "disabled",
    1: "monitoring",
    2: "failsafe_brake",
    3: "failsafe_descend",
    4: "touchdown_confirm",
    5: "landed",
}
_REASON_NAMES = {
    0: "none",
    1: "heartbeat_loss",
    2: "low_battery",
    3: "critical_battery",
    4: "clearance",
    5: "up_clearance",
    6: "manual_disable",
}


@dataclass(frozen=True, slots=True)
class OnDeviceFailsafeConfig:
    """Describe the firmware-side failsafe thresholds mirrored over Appchannel."""

    enabled: bool = True
    require_clearance: bool = True
    heartbeat_period_s: float = ON_DEVICE_FAILSAFE_HEARTBEAT_PERIOD_S
    heartbeat_timeout_s: float = ON_DEVICE_FAILSAFE_HEARTBEAT_TIMEOUT_S
    low_battery_v: float = ON_DEVICE_FAILSAFE_LOW_BATTERY_V
    critical_battery_v: float = ON_DEVICE_FAILSAFE_CRITICAL_BATTERY_V
    min_clearance_m: float = ON_DEVICE_FAILSAFE_MIN_CLEARANCE_M
    min_up_clearance_m: float = ON_DEVICE_FAILSAFE_MIN_UP_CLEARANCE_M
    descent_rate_mps: float = ON_DEVICE_FAILSAFE_DESCENT_RATE_MPS
    max_repel_velocity_mps: float = ON_DEVICE_FAILSAFE_MAX_REPEL_VELOCITY_MPS
    brake_hold_s: float = ON_DEVICE_FAILSAFE_BRAKE_HOLD_S


@dataclass(frozen=True, slots=True)
class OnDeviceFailsafeAvailability:
    """Summarize whether the Twinr firmware app is visible in the TOC."""

    loaded: bool
    protocol_version: int | None
    enabled: int | None
    state_code: int | None
    state_name: str | None
    reason_code: int | None
    reason_name: str | None
    failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OnDeviceFailsafeStatus:
    """Represent one status packet returned by the firmware app."""

    session_id: int
    state_code: int
    state_name: str
    reason_code: int
    reason_name: str
    heartbeat_age_ms: int
    vbat_mv: int
    min_clearance_mm: int
    down_range_mm: int


@dataclass(frozen=True, slots=True)
class OnDeviceFailsafeLinkMetrics:
    """Summarize optional link-health metrics exposed by current cflib builds."""

    latency_p95_ms: float | None
    link_quality: float | None
    uplink_rssi: float | None
    uplink_rate: float | None
    downlink_rate: float | None
    uplink_congestion: float | None
    downlink_congestion: float | None


# BREAKING: Session reports now include lifecycle timing, deadline-miss accounting,
# and optional link metrics in addition to the legacy fields.
@dataclass(frozen=True, slots=True)
class OnDeviceFailsafeSessionReport:
    """Persist the host-observed evidence for one firmware failsafe session."""

    mode: str
    config: OnDeviceFailsafeConfig
    availability: OnDeviceFailsafeAvailability
    session_id: int | None
    started: bool
    closed: bool
    disabled_cleanly: bool
    packets_sent: int
    status_packets_received: int
    heartbeat_deadline_misses: int
    started_monotonic_s: float | None
    closed_monotonic_s: float | None
    last_heartbeat_sent_monotonic_s: float | None
    last_status_received_monotonic_s: float | None
    last_status: OnDeviceFailsafeStatus | None
    link_metrics: OnDeviceFailsafeLinkMetrics | None
    failures: tuple[str, ...]


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
    try:
        emit(phase, status=status, message=message, data=dict(data) if data is not None else None)
    except Exception:
        return


def _normalize_uint16(value: float, *, minimum: int = 0) -> int:
    """Clamp one numeric config value into the firmware packet range."""

    numeric = int(round(float(value)))
    if numeric < minimum:
        return int(minimum)
    if numeric > _UINT16_MAX:
        return _UINT16_MAX
    return numeric


def _validate_positive_scaled_field(
    name: str,
    value: float,
    *,
    scale: float,
    minimum: int = 1,
) -> None:
    """Fail fast when one config field can not be represented safely."""

    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc
    if numeric <= 0.0:
        raise ValueError(f"{name} must be > 0, got {numeric}")
    scaled = int(round(numeric * scale))
    if scaled < minimum:
        raise ValueError(f"{name} is too small after scaling, got {numeric}")
    if scaled > _UINT16_MAX:
        raise ValueError(f"{name} is too large for uint16 transport, got {numeric}")


def validate_on_device_failsafe_config(config: OnDeviceFailsafeConfig) -> OnDeviceFailsafeConfig:
    """Validate one config before it is pushed into the firmware app."""

    if not isinstance(config, OnDeviceFailsafeConfig):
        raise TypeError(f"config must be OnDeviceFailsafeConfig, got {type(config).__name__}")
    _validate_positive_scaled_field("heartbeat_period_s", config.heartbeat_period_s, scale=1000.0, minimum=1)
    _validate_positive_scaled_field("heartbeat_timeout_s", config.heartbeat_timeout_s, scale=1000.0, minimum=1)
    if float(config.heartbeat_timeout_s) <= float(config.heartbeat_period_s):
        raise ValueError(
            "heartbeat_timeout_s must be strictly larger than heartbeat_period_s to avoid self-triggered failsafe"
        )
    _validate_positive_scaled_field("low_battery_v", config.low_battery_v, scale=1000.0, minimum=1)
    _validate_positive_scaled_field("critical_battery_v", config.critical_battery_v, scale=1000.0, minimum=1)
    if float(config.low_battery_v) <= float(config.critical_battery_v):
        raise ValueError("low_battery_v must be strictly larger than critical_battery_v")
    _validate_positive_scaled_field("min_clearance_m", config.min_clearance_m, scale=1000.0, minimum=1)
    _validate_positive_scaled_field("min_up_clearance_m", config.min_up_clearance_m, scale=1000.0, minimum=1)
    _validate_positive_scaled_field("descent_rate_mps", config.descent_rate_mps, scale=1000.0, minimum=1)
    _validate_positive_scaled_field(
        "max_repel_velocity_mps",
        config.max_repel_velocity_mps,
        scale=1000.0,
        minimum=1,
    )
    _validate_positive_scaled_field("brake_hold_s", config.brake_hold_s, scale=1000.0, minimum=1)
    return config


def _validate_session_id(session_id: int) -> int:
    """Validate that one session id fits the firmware transport contract."""

    session_id_int = int(session_id)
    if not 1 <= session_id_int <= _UINT16_MAX:
        raise ValueError(f"session_id must be in [1, {_UINT16_MAX}], got {session_id_int}")
    return session_id_int


def _unwrap_cf(sync_cf: Any) -> Any:
    """Resolve SyncCrazyflie-like wrappers into the underlying Crazyflie object."""

    return sync_cf.cf if hasattr(sync_cf, "cf") else sync_cf


def _require_param_surface(cf: Any) -> Any:
    """Fail fast unless the Crazyflie handle exposes the parameter surface."""

    if not hasattr(cf, "param"):
        raise TypeError("sync_cf/cf must expose param")
    return cf


def _require_appchannel_surface(cf: Any) -> Any:
    """Fail fast unless the Crazyflie handle exposes the Appchannel surface."""

    appchannel = getattr(cf, "appchannel", None)
    if appchannel is None or not hasattr(appchannel, "send_packet") or not hasattr(appchannel, "packet_received"):
        raise TypeError("sync_cf/cf must expose appchannel.send_packet() and appchannel.packet_received")
    return cf


def _state_name(code: int | None) -> str | None:
    """Translate one firmware state code into the stable host-side name."""

    if code is None:
        return None
    return _STATE_NAMES.get(int(code), f"unknown_{int(code)}")


def _reason_name(code: int | None) -> str | None:
    """Translate one firmware reason code into the stable host-side name."""

    if code is None:
        return None
    return _REASON_NAMES.get(int(code), f"unknown_{int(code)}")


def build_on_device_failsafe_heartbeat_packet(
    config: OnDeviceFailsafeConfig,
    *,
    session_id: int,
    enable: bool | None = None,
) -> bytes:
    """Encode one bounded heartbeat/config packet for the firmware app."""

    validate_on_device_failsafe_config(config)
    session_id_int = _validate_session_id(session_id)
    flags = 0
    if enable if enable is not None else config.enabled:
        flags |= ON_DEVICE_FAILSAFE_FLAG_ENABLE
    if config.require_clearance:
        flags |= ON_DEVICE_FAILSAFE_FLAG_REQUIRE_CLEARANCE
    payload = _HEARTBEAT_STRUCT.pack(
        ON_DEVICE_FAILSAFE_PROTOCOL_VERSION,
        ON_DEVICE_FAILSAFE_PACKET_KIND_HEARTBEAT,
        flags,
        0,
        session_id_int,
        _normalize_uint16(config.heartbeat_timeout_s * 1000.0, minimum=1),
        _normalize_uint16(config.low_battery_v * 1000.0, minimum=1),
        _normalize_uint16(config.critical_battery_v * 1000.0, minimum=1),
        _normalize_uint16(config.min_clearance_m * 1000.0, minimum=1),
        _normalize_uint16(config.min_up_clearance_m * 1000.0, minimum=1),
        _normalize_uint16(config.descent_rate_mps * 1000.0, minimum=1),
        _normalize_uint16(config.max_repel_velocity_mps * 1000.0, minimum=1),
        _normalize_uint16(config.brake_hold_s * 1000.0, minimum=1),
    )
    if len(payload) > ON_DEVICE_FAILSAFE_APPCHANNEL_MTU:
        raise ValueError(
            f"heartbeat packet is {len(payload)} bytes, exceeds Appchannel MTU {ON_DEVICE_FAILSAFE_APPCHANNEL_MTU}"
        )
    return payload


def parse_on_device_failsafe_status_packet(
    data: bytes | bytearray | memoryview,
) -> OnDeviceFailsafeStatus | None:
    """Decode one firmware status packet when it matches the Twinr protocol."""

    payload = memoryview(data)
    if len(payload) != _STATUS_STRUCT.size:
        return None
    unpacked = _STATUS_STRUCT.unpack(payload)
    (
        version,
        packet_kind,
        state_code,
        reason_code,
        session_id,
        heartbeat_age_ms,
        vbat_mv,
        min_clearance_mm,
        down_range_mm,
    ) = unpacked
    if version != ON_DEVICE_FAILSAFE_PROTOCOL_VERSION:
        return None
    if packet_kind != ON_DEVICE_FAILSAFE_PACKET_KIND_STATUS:
        return None
    if int(session_id) <= 0:
        return None
    return OnDeviceFailsafeStatus(
        session_id=int(session_id),
        state_code=int(state_code),
        state_name=_state_name(int(state_code)) or f"unknown_{int(state_code)}",
        reason_code=int(reason_code),
        reason_name=_reason_name(int(reason_code)) or f"unknown_{int(reason_code)}",
        heartbeat_age_ms=int(heartbeat_age_ms),
        vbat_mv=int(vbat_mv),
        min_clearance_mm=int(min_clearance_mm),
        down_range_mm=int(down_range_mm),
    )


def probe_on_device_failsafe(sync_cf: Any) -> OnDeviceFailsafeAvailability:
    """Read the app-layer param surface that proves the firmware app is loaded.

    Call this only after the Crazyflie is fully connected and parameters are
    available in cflib.
    """

    cf = _require_param_surface(_unwrap_cf(sync_cf))
    param = cf.param
    observed: dict[str, int | None] = {}
    failures: list[str] = []
    for param_name in _AVAILABLE_PARAM_NAMES:
        try:
            raw_value = param.get_value(param_name)
        except Exception as exc:
            failures.append(f"{param_name}:{exc.__class__.__name__}:{exc}")
            continue
        try:
            observed[param_name] = int(str(raw_value).strip())
        except (TypeError, ValueError):
            observed[param_name] = None
            failures.append(f"{param_name}:non_integer_value")

    # BREAKING: loaded is now only True when the full Twinr param surface is readable
    # and the protocol version matches. Previously a single readable protocolVersion
    # param could still mark the app as loaded.
    surface_complete = all(name in observed and observed.get(name) is not None for name in _AVAILABLE_PARAM_NAMES)
    protocol_version = observed.get("twinrFs.protocolVersion")
    loaded = surface_complete and protocol_version == ON_DEVICE_FAILSAFE_PROTOCOL_VERSION

    if surface_complete and protocol_version != ON_DEVICE_FAILSAFE_PROTOCOL_VERSION:
        failures.append(
            f"twinrFs.protocolVersion reported {protocol_version}, expected {ON_DEVICE_FAILSAFE_PROTOCOL_VERSION}"
        )
    if not surface_complete:
        missing = tuple(name for name in _AVAILABLE_PARAM_NAMES if observed.get(name) is None)
        if missing:
            failures.append(f"incomplete_param_surface:{','.join(missing)}")

    return OnDeviceFailsafeAvailability(
        loaded=loaded,
        protocol_version=protocol_version,
        enabled=observed.get("twinrFs.enable"),
        state_code=observed.get("twinrFs.state"),
        state_name=_state_name(observed.get("twinrFs.state")),
        reason_code=observed.get("twinrFs.reason"),
        reason_name=_reason_name(observed.get("twinrFs.reason")),
        failures=tuple(failures),
    )


class OnDeviceFailsafeHeartbeatSession:
    """Keep one bounded Appchannel heartbeat alive while the host controls flight."""

    def __init__(
        self,
        sync_cf: Any,
        *,
        mode: str,
        config: OnDeviceFailsafeConfig,
        availability: OnDeviceFailsafeAvailability,
        trace_writer: Any | None = None,
        session_id: int | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        allow_unverified_start: bool = False,
        manage_link_statistics: bool = True,
    ) -> None:
        self._cf = _require_appchannel_surface(_unwrap_cf(sync_cf))
        mode_text = str(mode).strip()
        if not mode_text:
            raise ValueError("mode must be a non-empty string")
        self.mode = mode_text
        self.config = validate_on_device_failsafe_config(config)
        self.availability = availability
        self.trace_writer = trace_writer
        self._monotonic = monotonic
        self._sleep = sleep
        self._allow_unverified_start = bool(allow_unverified_start)
        self._manage_link_statistics = bool(manage_link_statistics)
        self._session_id = (
            _validate_session_id(session_id)
            if session_id is not None
            else secrets.randbelow(_UINT16_MAX) + 1
        )
        self._packets_sent = 0
        self._status_packets_received = 0
        self._heartbeat_deadline_misses = 0
        self._last_status: OnDeviceFailsafeStatus | None = None
        self._failures: list[str] = []
        self._stats_lock = threading.Lock()
        self._lifecycle_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._status_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._disabled_cleanly = False
        self._closed = False
        self._started_monotonic_s: float | None = None
        self._closed_monotonic_s: float | None = None
        self._last_heartbeat_sent_monotonic_s: float | None = None
        self._last_status_received_monotonic_s: float | None = None
        self._next_heartbeat_deadline_s: float | None = None

        self._link_statistics = getattr(self._cf, "link_statistics", None) if self._manage_link_statistics else None
        self._link_statistics_started_by_session = False
        self._link_latency_p95_ms: float | None = None
        self._link_quality: float | None = None
        self._uplink_rssi: float | None = None
        self._uplink_rate: float | None = None
        self._downlink_rate: float | None = None
        self._uplink_congestion: float | None = None
        self._downlink_congestion: float | None = None

    @property
    def session_id(self) -> int:
        """Return the bounded firmware session id used by this heartbeat thread."""

        return self._session_id

    @property
    def closed(self) -> bool:
        """Tell whether the session has already been closed."""

        return self._closed

    def __enter__(self) -> OnDeviceFailsafeHeartbeatSession:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(disable=True)
        return None

    def start(self) -> None:
        """Register callbacks, send the first heartbeat, and start looping."""

        with self._lifecycle_lock:
            if self._started:
                return
            if self._closed:
                raise RuntimeError("cannot start an OnDeviceFailsafeHeartbeatSession after it has been closed")

            # BREAKING: default startup is now fail-closed. The host refuses to start
            # unless probe_on_device_failsafe() already verified the firmware app.
            if not self._allow_unverified_start and not self.availability.loaded:
                raise RuntimeError(
                    "on-device failsafe firmware app is not verified; call probe_on_device_failsafe() after full "
                    "connection and pass the successful availability object, or set allow_unverified_start=True"
                )

            self._started_monotonic_s = self._monotonic()
            _emit_trace(
                self.trace_writer,
                "on_device_failsafe_start",
                status="begin",
                data={"mode": self.mode, "session_id": self._session_id},
            )

            callback_registered = False
            try:
                self._cf.appchannel.packet_received.add_callback(self._handle_status_packet)
                callback_registered = True
                self._start_link_statistics_if_available()
                self._send_heartbeat(enable=True)
                self._next_heartbeat_deadline_s = self._monotonic() + float(self.config.heartbeat_period_s)
                thread = threading.Thread(
                    target=self._heartbeat_loop,
                    name=f"twinr-on-device-failsafe-{self._session_id}",
                    daemon=True,
                )
                self._thread = thread
                self._started = True
                thread.start()
            except Exception:
                self._stop_event.set()
                if callback_registered:
                    self._safe_remove_appchannel_callback()
                self._stop_link_statistics_if_owned()
                self._thread = None
                self._started = False
                self._started_monotonic_s = None
                self._next_heartbeat_deadline_s = None
                self._stop_event.clear()
                raise

            _emit_trace(
                self.trace_writer,
                "on_device_failsafe_start",
                status="done",
                data={"mode": self.mode, "session_id": self._session_id},
            )

    def close(self, *, disable: bool) -> None:
        """Stop heartbeats and optionally disable the firmware app cleanly."""

        with self._lifecycle_lock:
            if self._closed:
                return

            self._stop_event.set()
            thread = self._thread
            if thread is not None:
                thread.join(timeout=max(float(self.config.heartbeat_period_s) * 3.0, 0.1))

            if disable and self._started:
                try:
                    self._send_heartbeat(enable=False)
                    self._disabled_cleanly = True
                except Exception as exc:  # pragma: no cover - live link error path
                    self._record_failure(f"disable_packet:{exc.__class__.__name__}:{exc}")
                    _emit_trace(
                        self.trace_writer,
                        "on_device_failsafe_stop",
                        status="error",
                        message=f"{exc.__class__.__name__}:{exc}",
                    )

            if self._started:
                self._safe_remove_appchannel_callback()
            self._stop_link_statistics_if_owned()

            self._closed = True
            self._closed_monotonic_s = self._monotonic()
            _emit_trace(
                self.trace_writer,
                "on_device_failsafe_stop",
                status="done",
                data={"disabled_cleanly": self._disabled_cleanly, "disable": disable},
            )

    def wait_for_status(self, timeout_s: float | None = None) -> OnDeviceFailsafeStatus | None:
        """Return the current status immediately or wait for the first one to arrive."""

        with self._stats_lock:
            if self._last_status is not None:
                return self._last_status
        waited = self._status_event.wait(timeout=timeout_s)
        if not waited:
            return None
        with self._stats_lock:
            return self._last_status

    def report(self) -> OnDeviceFailsafeSessionReport:
        """Return the durable host-side evidence for this heartbeat session."""

        with self._stats_lock:
            last_status = self._last_status
            packets_sent = self._packets_sent
            status_packets_received = self._status_packets_received
            heartbeat_deadline_misses = self._heartbeat_deadline_misses
            last_heartbeat_sent_monotonic_s = self._last_heartbeat_sent_monotonic_s
            last_status_received_monotonic_s = self._last_status_received_monotonic_s
            failures = tuple(self._failures)
            link_metrics = self._snapshot_link_metrics_unlocked()
        return OnDeviceFailsafeSessionReport(
            mode=self.mode,
            config=self.config,
            availability=self.availability,
            session_id=self._session_id if self._started else None,
            started=self._started,
            closed=self._closed,
            disabled_cleanly=self._disabled_cleanly,
            packets_sent=packets_sent,
            status_packets_received=status_packets_received,
            heartbeat_deadline_misses=heartbeat_deadline_misses,
            started_monotonic_s=self._started_monotonic_s,
            closed_monotonic_s=self._closed_monotonic_s,
            last_heartbeat_sent_monotonic_s=last_heartbeat_sent_monotonic_s,
            last_status_received_monotonic_s=last_status_received_monotonic_s,
            last_status=last_status,
            link_metrics=link_metrics,
            failures=failures,
        )

    def _heartbeat_loop(self) -> None:
        """Transmit bounded heartbeat packets until the worker stops the session."""

        period_s = float(self.config.heartbeat_period_s)
        next_deadline_s = self._next_heartbeat_deadline_s or (self._monotonic() + period_s)

        while not self._stop_event.is_set():
            remaining_s = next_deadline_s - self._monotonic()
            if remaining_s > 0.0 and self._stop_event.wait(timeout=remaining_s):
                return

            now_s = self._monotonic()
            lateness_s = now_s - next_deadline_s
            if lateness_s > max(0.005, period_s * 0.25):
                with self._stats_lock:
                    self._heartbeat_deadline_misses += 1
                    misses = self._heartbeat_deadline_misses
                _emit_trace(
                    self.trace_writer,
                    "on_device_failsafe_heartbeat_deadline",
                    status="late",
                    data={"lateness_ms": round(lateness_s * 1000.0, 3), "deadline_misses": misses},
                )

            try:
                self._send_heartbeat(enable=True)
            except Exception as exc:  # pragma: no cover - live radio path
                self._record_failure(f"heartbeat:{exc.__class__.__name__}:{exc}")
                _emit_trace(
                    self.trace_writer,
                    "on_device_failsafe_heartbeat",
                    status="error",
                    message=f"{exc.__class__.__name__}:{exc}",
                )
                self._stop_event.set()
                return

            next_deadline_s += period_s
            now_after_send_s = self._monotonic()
            if now_after_send_s > next_deadline_s:
                skipped_periods = int((now_after_send_s - next_deadline_s) // period_s) + 1
                next_deadline_s += skipped_periods * period_s

    def _send_heartbeat(self, *, enable: bool) -> None:
        """Send one heartbeat/config packet to the firmware app."""

        payload = build_on_device_failsafe_heartbeat_packet(
            self.config,
            session_id=self._session_id,
            enable=enable,
        )
        self._cf.appchannel.send_packet(payload)
        sent_at_s = self._monotonic()
        with self._stats_lock:
            self._packets_sent += 1
            self._last_heartbeat_sent_monotonic_s = sent_at_s
            packets_sent = self._packets_sent
        _emit_trace(
            self.trace_writer,
            "on_device_failsafe_heartbeat",
            status="done",
            data={"enable": enable, "packets_sent": packets_sent},
        )

    def _handle_status_packet(self, payload: bytes) -> None:
        """Parse one returned status packet and keep only Twinr's protocol."""

        try:
            status = parse_on_device_failsafe_status_packet(payload)
            if status is None:
                return
            if status.session_id != self._session_id:
                return
            received_at_s = self._monotonic()
            with self._stats_lock:
                self._last_status = status
                self._status_packets_received += 1
                self._last_status_received_monotonic_s = received_at_s
            self._status_event.set()
            _emit_trace(
                self.trace_writer,
                "on_device_failsafe_status",
                status="done",
                data=asdict(status),
            )
        except Exception as exc:
            self._record_failure(f"status_callback:{exc.__class__.__name__}:{exc}")
            _emit_trace(
                self.trace_writer,
                "on_device_failsafe_status",
                status="error",
                message=f"{exc.__class__.__name__}:{exc}",
            )

    def _record_failure(self, failure: str) -> None:
        """Record one host-side failure string in a thread-safe way."""

        with self._stats_lock:
            self._failures.append(str(failure))

    def _snapshot_link_metrics_unlocked(self) -> OnDeviceFailsafeLinkMetrics | None:
        """Create one immutable snapshot of the current link metrics."""

        if self._link_statistics is None:
            return None
        return OnDeviceFailsafeLinkMetrics(
            latency_p95_ms=self._link_latency_p95_ms,
            link_quality=self._link_quality,
            uplink_rssi=self._uplink_rssi,
            uplink_rate=self._uplink_rate,
            downlink_rate=self._downlink_rate,
            uplink_congestion=self._uplink_congestion,
            downlink_congestion=self._downlink_congestion,
        )

    def _start_link_statistics_if_available(self) -> None:
        """Hook into cflib link statistics when the installed cflib exposes them."""

        ls = self._link_statistics
        if ls is None:
            return
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
        if not all(hasattr(ls, name) for name in required):
            self._link_statistics = None
            return

        ls.latency_updated.add_callback(self._handle_link_latency_updated)
        ls.link_quality_updated.add_callback(self._handle_link_quality_updated)
        ls.uplink_rssi_updated.add_callback(self._handle_uplink_rssi_updated)
        ls.uplink_rate_updated.add_callback(self._handle_uplink_rate_updated)
        ls.downlink_rate_updated.add_callback(self._handle_downlink_rate_updated)
        ls.uplink_congestion_updated.add_callback(self._handle_uplink_congestion_updated)
        ls.downlink_congestion_updated.add_callback(self._handle_downlink_congestion_updated)

        if not bool(getattr(ls, "_is_active", False)):
            ls.start()
            self._link_statistics_started_by_session = True

    def _stop_link_statistics_if_owned(self) -> None:
        """Detach link-statistics callbacks and stop collection only if this session started it."""

        ls = self._link_statistics
        if ls is None:
            return

        self._safe_remove_callback(getattr(ls, "latency_updated", None), self._handle_link_latency_updated)
        self._safe_remove_callback(getattr(ls, "link_quality_updated", None), self._handle_link_quality_updated)
        self._safe_remove_callback(getattr(ls, "uplink_rssi_updated", None), self._handle_uplink_rssi_updated)
        self._safe_remove_callback(getattr(ls, "uplink_rate_updated", None), self._handle_uplink_rate_updated)
        self._safe_remove_callback(getattr(ls, "downlink_rate_updated", None), self._handle_downlink_rate_updated)
        self._safe_remove_callback(
            getattr(ls, "uplink_congestion_updated", None),
            self._handle_uplink_congestion_updated,
        )
        self._safe_remove_callback(
            getattr(ls, "downlink_congestion_updated", None),
            self._handle_downlink_congestion_updated,
        )

        if self._link_statistics_started_by_session:
            try:
                ls.stop()
            finally:
                self._link_statistics_started_by_session = False

    def _safe_remove_appchannel_callback(self) -> None:
        """Best-effort removal of the Appchannel callback."""

        self._safe_remove_callback(self._cf.appchannel.packet_received, self._handle_status_packet)

    @staticmethod
    def _safe_remove_callback(caller: Any | None, callback: Callable[..., Any]) -> None:
        """Best-effort callback removal for cflib Caller-like objects."""

        if caller is None:
            return
        remove_callback = getattr(caller, "remove_callback", None)
        if remove_callback is None:
            return
        try:
            remove_callback(callback)
        except (ValueError, AttributeError):
            return

    def _set_link_metric(self, attribute_name: str, value: Any) -> None:
        """Store one numeric link metric without letting callback exceptions escape."""

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return
        with self._stats_lock:
            setattr(self, attribute_name, numeric_value)

    def _handle_link_latency_updated(self, value: Any) -> None:
        self._set_link_metric("_link_latency_p95_ms", value)

    def _handle_link_quality_updated(self, value: Any) -> None:
        self._set_link_metric("_link_quality", value)

    def _handle_uplink_rssi_updated(self, value: Any) -> None:
        self._set_link_metric("_uplink_rssi", value)

    def _handle_uplink_rate_updated(self, value: Any) -> None:
        self._set_link_metric("_uplink_rate", value)

    def _handle_downlink_rate_updated(self, value: Any) -> None:
        self._set_link_metric("_downlink_rate", value)

    def _handle_uplink_congestion_updated(self, value: Any) -> None:
        self._set_link_metric("_uplink_congestion", value)

    def _handle_downlink_congestion_updated(self, value: Any) -> None:
        self._set_link_metric("_downlink_congestion", value)
