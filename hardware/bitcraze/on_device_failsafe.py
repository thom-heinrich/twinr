#!/usr/bin/env python3
"""Drive Twinr's on-device Crazyflie failsafe over the Appchannel.

Purpose
-------
Keep the host-side part of the on-device failsafe intentionally small. This
module only proves that the firmware app is present, sends bounded heartbeat
packets with the current failsafe thresholds, and records status packets sent
back by the app. The actual safety-critical land/avoid controller lives on the
Crazyflie in C; Python only configures and observes it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class OnDeviceFailsafeSessionReport:
    """Persist the host-observed evidence for one firmware failsafe session."""

    mode: str
    config: OnDeviceFailsafeConfig
    availability: OnDeviceFailsafeAvailability
    session_id: int | None
    started: bool
    disabled_cleanly: bool
    packets_sent: int
    status_packets_received: int
    last_status: OnDeviceFailsafeStatus | None
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
    emit(phase, status=status, message=message, data=dict(data) if data is not None else None)


def _normalize_uint16(value: float, *, minimum: int = 0) -> int:
    """Clamp one numeric config value into the firmware packet range."""

    numeric = int(round(float(value)))
    if numeric < minimum:
        return int(minimum)
    if numeric > _UINT16_MAX:
        return _UINT16_MAX
    return numeric


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

    flags = 0
    if enable if enable is not None else config.enabled:
        flags |= ON_DEVICE_FAILSAFE_FLAG_ENABLE
    if config.require_clearance:
        flags |= ON_DEVICE_FAILSAFE_FLAG_REQUIRE_CLEARANCE
    return _HEARTBEAT_STRUCT.pack(
        ON_DEVICE_FAILSAFE_PROTOCOL_VERSION,
        ON_DEVICE_FAILSAFE_PACKET_KIND_HEARTBEAT,
        flags,
        0,
        _normalize_uint16(session_id, minimum=1),
        _normalize_uint16(config.heartbeat_timeout_s * 1000.0, minimum=1),
        _normalize_uint16(config.low_battery_v * 1000.0, minimum=1),
        _normalize_uint16(config.critical_battery_v * 1000.0, minimum=1),
        _normalize_uint16(config.min_clearance_m * 1000.0, minimum=1),
        _normalize_uint16(config.min_up_clearance_m * 1000.0, minimum=1),
        _normalize_uint16(config.descent_rate_mps * 1000.0, minimum=1),
        _normalize_uint16(config.max_repel_velocity_mps * 1000.0, minimum=1),
        _normalize_uint16(config.brake_hold_s * 1000.0, minimum=1),
    )


def parse_on_device_failsafe_status_packet(data: bytes | bytearray | memoryview) -> OnDeviceFailsafeStatus | None:
    """Decode one firmware status packet when it matches the Twinr protocol."""

    payload = bytes(data)
    if len(payload) < _STATUS_STRUCT.size:
        return None
    unpacked = _STATUS_STRUCT.unpack(payload[: _STATUS_STRUCT.size])
    version, packet_kind, state_code, reason_code, session_id, heartbeat_age_ms, vbat_mv, min_clearance_mm, down_range_mm = unpacked
    if version != ON_DEVICE_FAILSAFE_PROTOCOL_VERSION:
        return None
    if packet_kind != ON_DEVICE_FAILSAFE_PACKET_KIND_STATUS:
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
    """Read the app-layer param surface that proves the firmware app is loaded."""

    param = sync_cf.cf.param if hasattr(sync_cf, "cf") else sync_cf.param
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
    loaded = "twinrFs.protocolVersion" in observed and observed.get("twinrFs.protocolVersion") is not None
    protocol_version = observed.get("twinrFs.protocolVersion")
    if loaded and protocol_version != ON_DEVICE_FAILSAFE_PROTOCOL_VERSION:
        failures.append(
            f"twinrFs.protocolVersion reported {protocol_version}, expected {ON_DEVICE_FAILSAFE_PROTOCOL_VERSION}"
        )
        loaded = False
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
    ) -> None:
        self._cf = sync_cf.cf if hasattr(sync_cf, "cf") else sync_cf
        self.mode = str(mode)
        self.config = config
        self.availability = availability
        self.trace_writer = trace_writer
        self._monotonic = monotonic
        self._sleep = sleep
        self._session_id = int(session_id) if session_id is not None else ((time.time_ns() & _UINT16_MAX) or 1)
        self._packets_sent = 0
        self._status_packets_received = 0
        self._last_status: OnDeviceFailsafeStatus | None = None
        self._failures: list[str] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._disabled_cleanly = False
        self._closed = False

    @property
    def session_id(self) -> int:
        """Return the bounded firmware session id used by this heartbeat thread."""

        return self._session_id

    def start(self) -> None:
        """Register the status callback, send the first heartbeat, and start looping."""

        if self._started:
            return
        _emit_trace(
            self.trace_writer,
            "on_device_failsafe_start",
            status="begin",
            data={"mode": self.mode, "session_id": self._session_id},
        )
        self._cf.appchannel.packet_received.add_callback(self._handle_status_packet)
        self._send_heartbeat(enable=True)
        thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"twinr-on-device-failsafe-{self._session_id}",
            daemon=True,
        )
        self._thread = thread
        self._started = True
        thread.start()
        _emit_trace(
            self.trace_writer,
            "on_device_failsafe_start",
            status="done",
            data={"mode": self.mode, "session_id": self._session_id},
        )

    def close(self, *, disable: bool) -> None:
        """Stop heartbeats and optionally disable the firmware app cleanly."""

        if self._closed:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.2, self.config.heartbeat_period_s * 3.0))
        if disable and self._started:
            try:
                self._send_heartbeat(enable=False)
                self._disabled_cleanly = True
            except Exception as exc:  # pragma: no cover - live link error path
                self._failures.append(f"disable_packet:{exc.__class__.__name__}:{exc}")
                _emit_trace(
                    self.trace_writer,
                    "on_device_failsafe_stop",
                    status="error",
                    message=f"{exc.__class__.__name__}:{exc}",
                )
        if self._started:
            try:
                self._cf.appchannel.packet_received.remove_callback(self._handle_status_packet)
            except ValueError:
                pass
        self._closed = True
        _emit_trace(
            self.trace_writer,
            "on_device_failsafe_stop",
            status="done",
            data={"disabled_cleanly": self._disabled_cleanly, "disable": disable},
        )

    def report(self) -> OnDeviceFailsafeSessionReport:
        """Return the durable host-side evidence for this heartbeat session."""

        with self._lock:
            last_status = self._last_status
            packets_sent = self._packets_sent
            status_packets_received = self._status_packets_received
        return OnDeviceFailsafeSessionReport(
            mode=self.mode,
            config=self.config,
            availability=self.availability,
            session_id=self._session_id if self._started else None,
            started=self._started,
            disabled_cleanly=self._disabled_cleanly,
            packets_sent=packets_sent,
            status_packets_received=status_packets_received,
            last_status=last_status,
            failures=tuple(self._failures),
        )

    def _heartbeat_loop(self) -> None:
        """Transmit bounded heartbeat packets until the worker stops the session."""

        while not self._stop_event.wait(timeout=max(0.02, self.config.heartbeat_period_s)):
            try:
                self._send_heartbeat(enable=True)
            except Exception as exc:  # pragma: no cover - live radio path
                self._failures.append(f"heartbeat:{exc.__class__.__name__}:{exc}")
                _emit_trace(
                    self.trace_writer,
                    "on_device_failsafe_heartbeat",
                    status="error",
                    message=f"{exc.__class__.__name__}:{exc}",
                )
                break

    def _send_heartbeat(self, *, enable: bool) -> None:
        """Send one heartbeat/config packet to the firmware app."""

        payload = build_on_device_failsafe_heartbeat_packet(
            self.config,
            session_id=self._session_id,
            enable=enable,
        )
        self._cf.appchannel.send_packet(payload)
        with self._lock:
            self._packets_sent += 1
        _emit_trace(
            self.trace_writer,
            "on_device_failsafe_heartbeat",
            status="done",
            data={"enable": enable, "packets_sent": self._packets_sent},
        )

    def _handle_status_packet(self, payload: bytes) -> None:
        """Parse one returned status packet and keep only Twinr's protocol."""

        status = parse_on_device_failsafe_status_packet(payload)
        if status is None:
            return
        if status.session_id != self._session_id:
            return
        with self._lock:
            self._last_status = status
            self._status_packets_received += 1
        _emit_trace(
            self.trace_writer,
            "on_device_failsafe_status",
            status="done",
            data=asdict(status),
        )
