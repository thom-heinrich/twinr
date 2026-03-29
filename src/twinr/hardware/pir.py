"""Adapt PIR motion sensing onto Twinr's button-monitor GPIO backend.

This module keeps PIR handling thin by reusing ``twinr.hardware.buttons`` for
GPIO access, edge normalization, and backend selection while exposing PIR-
specific motion semantics to callers.
"""

# CHANGELOG: 2026-03-28
# BUG-1: Detect and reject wrong-line backend events instead of silently relabeling them as the configured PIR line.
# BUG-2: Add optional PIR warm-up gating so startup calibration pulses do not become motion alerts in real deployments.
# BUG-3: Preserve explicit event-clock semantics for synthetic level-derived events and expose richer event metadata.
# SEC-1: Restrict chip_name to gpiochip device names by default so config cannot point the monitor at arbitrary filesystem paths.
# IMP-1: Add async poll/iteration hooks backed by pollable GPIO file descriptors when available, with a safe thread fallback.
# IMP-2: Add wait_for_no_motion(), occupancy_detected(), readiness/diagnostics APIs, and optional presence_hold_s for 2026 ambient-sensing usage.

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import math
import re
import time
from typing import Any, AsyncIterator, Iterator

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.buttons import ButtonAction, ButtonBinding, GpioButtonMonitor


class PirMonitorError(RuntimeError):
    """Raised when the PIR monitor cannot be opened or read safely."""


_GPIO_CHIP_RE = re.compile(r"^(?:/dev/)?gpiochip\d+$")
_VALID_BIASES = {
    "pull-down": "pull-down",
    "pulldown": "pull-down",
    "down": "pull-down",
    "pull-up": "pull-up",
    "pullup": "pull-up",
    "up": "pull-up",
    "disabled": "disabled",
    "disable": "disabled",
    "off": "disabled",
    "none": "disabled",
}
_VALID_EVENT_CLOCKS = {"auto", "monotonic", "realtime", "hte", "unknown"}


def _validate_non_empty_str(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _validate_non_negative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _validate_non_negative_timeout(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative finite number or None")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative finite number or None") from exc
    if not math.isfinite(numeric_value) or numeric_value < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number or None")
    return numeric_value


def _validate_chip_name(chip_name: str, *, allow_nonstandard_chip_path: bool) -> str:
    chip_name = _validate_non_empty_str("chip_name", chip_name)
    if allow_nonstandard_chip_path:
        return chip_name
    if not _GPIO_CHIP_RE.fullmatch(chip_name):
        # BREAKING: non-gpiochip paths are rejected by default instead of being passed through to the backend.
        raise ValueError(
            "chip_name must be a gpiochip device name like 'gpiochip0' or '/dev/gpiochip0'"
        )
    return chip_name


def _normalize_bias(name: str, value: str) -> str:
    normalized = _validate_non_empty_str(name, value).lower().replace("_", "-")
    try:
        return _VALID_BIASES[normalized]
    except KeyError as exc:
        # BREAKING: unknown bias strings now fail fast instead of crashing later inside the GPIO backend.
        allowed = ", ".join(sorted(set(_VALID_BIASES.values())))
        raise ValueError(f"{name} must be one of: {allowed}") from exc


def _normalize_event_clock(name: str, value: str) -> str:
    normalized = _validate_non_empty_str(name, value).lower()
    if normalized not in _VALID_EVENT_CLOCKS:
        raise ValueError(f"{name} must be one of: {', '.join(sorted(_VALID_EVENT_CLOCKS))}")
    return normalized


def _deadline_from_timeout(timeout_s: float | None) -> int | None:
    if timeout_s is None:
        return None
    return time.monotonic_ns() + int(timeout_s * 1_000_000_000)


def _remaining_seconds(deadline_ns: int | None) -> float | None:
    if deadline_ns is None:
        return None
    remaining_ns = deadline_ns - time.monotonic_ns()
    if remaining_ns <= 0:
        return 0.0
    return remaining_ns / 1_000_000_000


def _safe_attr_chain(obj: Any, *names: str) -> Any | None:
    current = obj
    for name in names:
        if current is None or not hasattr(current, name):
            return None
        current = getattr(current, name)
    return current


def _call_maybe(callable_or_value: Any, *args: Any, **kwargs: Any) -> Any:
    if callable(callable_or_value):
        return callable_or_value(*args, **kwargs)
    return callable_or_value


def _coerce_binary_value(line_offset: int, raw_value: Any) -> int:
    if isinstance(raw_value, bool):
        return int(raw_value)
    if isinstance(raw_value, int) and raw_value in (0, 1):
        return raw_value

    enum_name = getattr(raw_value, "name", None)
    if isinstance(enum_name, str):
        normalized = enum_name.strip().upper()
        if normalized in {"ACTIVE", "HIGH", "ON", "PRESSED"}:
            return 1
        if normalized in {"INACTIVE", "LOW", "OFF", "RELEASED"}:
            return 0

    try:
        numeric_value = int(raw_value)
    except (TypeError, ValueError):
        numeric_value = None
    if numeric_value in (0, 1):
        return numeric_value

    raise PirMonitorError(
        f"Unexpected snapshot value for PIR line {line_offset}: {raw_value!r}"
    )


@dataclass(frozen=True, slots=True)
class PirBinding:
    """Bind the logical PIR sensor name to one GPIO line offset."""

    name: str
    line_offset: int


@dataclass(frozen=True, slots=True)
class PirMotionEvent:
    """Represent one normalized PIR motion event."""

    name: str
    line_offset: int
    motion_detected: bool
    raw_edge: str
    timestamp_ns: int
    event_kind: str = "unknown"
    source: str = "edge"
    event_clock: str | None = None
    sequence_no: int | None = None
    line_sequence_no: int | None = None


@dataclass(frozen=True, slots=True)
class PirMonitorDiagnostics:
    """Summarize runtime PIR monitor state and any discoverable backend metadata."""

    chip_name: str
    line_offset: int
    binding_name: str
    is_open: bool
    ready: bool
    ready_in_s: float
    active_high: bool
    bias: str
    debounce_ms: int
    consumer: str
    warmup_s: float
    presence_hold_s: float
    event_clock: str
    backend_fd: int | None = None
    backend_consumer: str | None = None
    backend_used: bool | None = None
    backend_edge_detection: str | None = None
    backend_bias: str | None = None
    backend_debounced: bool | None = None
    backend_debounce_ms: float | None = None


def build_pir_binding(config: TwinrConfig) -> PirBinding:
    """Build the configured PIR binding from ``TwinrConfig``."""

    if config.pir_motion_gpio is None:
        raise ValueError("Set TWINR_PIR_MOTION_GPIO before watching PIR motion")
    line_offset = _validate_non_negative_int(
        "TWINR_PIR_MOTION_GPIO",
        config.pir_motion_gpio,
    )
    return PirBinding(name="pir", line_offset=line_offset)


class GpioPirMonitor:
    """Monitor PIR motion through the shared GPIO button backend."""

    def __init__(
        self,
        chip_name: str,
        binding: PirBinding,
        *,
        active_high: bool = True,
        bias: str = "pull-down",
        debounce_ms: int = 120,
        consumer: str = "twinr-pir",
        warmup_s: float = 0.0,
        presence_hold_s: float = 0.0,
        event_clock: str = "auto",
        allow_nonstandard_chip_path: bool = False,
        monitor: GpioButtonMonitor | None = None,
        close_injected_monitor: bool = False,
    ) -> None:
        self.chip_name = _validate_chip_name(
            chip_name,
            allow_nonstandard_chip_path=allow_nonstandard_chip_path,
        )
        if not isinstance(binding, PirBinding):
            raise TypeError("binding must be a PirBinding")
        _validate_non_empty_str("binding.name", binding.name)
        _validate_non_negative_int("binding.line_offset", binding.line_offset)
        if not isinstance(active_high, bool):
            raise ValueError("active_high must be a bool")
        if not isinstance(close_injected_monitor, bool):
            raise ValueError("close_injected_monitor must be a bool")
        if not isinstance(allow_nonstandard_chip_path, bool):
            raise ValueError("allow_nonstandard_chip_path must be a bool")

        self.binding = binding
        self.active_high = active_high
        self.bias = _normalize_bias("bias", bias)
        self.debounce_ms = _validate_non_negative_int("debounce_ms", debounce_ms)
        self.consumer = _validate_non_empty_str("consumer", consumer)
        self.warmup_s = _validate_non_negative_timeout("warmup_s", warmup_s) or 0.0
        self.presence_hold_s = (
            _validate_non_negative_timeout("presence_hold_s", presence_hold_s) or 0.0
        )
        self._requested_event_clock = _normalize_event_clock("event_clock", event_clock)
        self._resolved_event_clock = (
            "monotonic" if self._requested_event_clock == "auto" else self._requested_event_clock
        )
        self._manage_monitor_lifecycle = monitor is None or close_injected_monitor
        self._is_open = monitor is not None and not self._manage_monitor_lifecycle
        self._opened_monotonic_ns: int | None = time.monotonic_ns() if self._is_open else None
        self._last_motion_monotonic_ns: int | None = None
        self._last_event: PirMotionEvent | None = None
        self._monitor = (
            monitor
            if monitor is not None
            else GpioButtonMonitor(
                chip_name=self.chip_name,
                bindings=(ButtonBinding(name=binding.name, line_offset=binding.line_offset),),
                active_low=not active_high,
                bias=self.bias,
                debounce_ms=self.debounce_ms,
                consumer=self.consumer,
            )
        )
        if self._is_open:
            self._refresh_backend_metadata()

    def _ensure_open(self) -> None:
        if not self._is_open:
            raise PirMonitorError("PIR monitor is not open; call open() before reading motion events")

    def _backend_fd(self) -> int | None:
        fileno = getattr(self._monitor, "fileno", None)
        if callable(fileno):
            try:
                fd = int(fileno())
            except Exception:
                fd = None
            if fd is not None and fd >= 0:
                return fd

        for attr_chain in (
            ("fd",),
            ("request", "fd"),
            ("_request", "fd"),
            ("_line_request", "fd"),
        ):
            fd = _safe_attr_chain(self._monitor, *attr_chain)
            if isinstance(fd, int) and fd >= 0:
                return fd
        return None

    def fileno(self) -> int:
        """Return a pollable backend file descriptor when the backend exposes one."""

        self._ensure_open()
        fd = self._backend_fd()
        if fd is None:
            raise PirMonitorError(
                "The underlying GPIO backend does not expose a pollable file descriptor"
            )
        return fd

    def _backend_line_info(self) -> Any | None:
        candidates = (
            (self._monitor, "line_info"),
            (self._monitor, "get_line_info"),
            (_safe_attr_chain(self._monitor, "chip"), "line_info"),
            (_safe_attr_chain(self._monitor, "chip"), "get_line_info"),
            (_safe_attr_chain(self._monitor, "_chip"), "line_info"),
            (_safe_attr_chain(self._monitor, "_chip"), "get_line_info"),
        )
        for owner, attr_name in candidates:
            if owner is None or not hasattr(owner, attr_name):
                continue
            try:
                line_info = _call_maybe(getattr(owner, attr_name), self.binding.line_offset)
            except TypeError:
                try:
                    line_info = _call_maybe(getattr(owner, attr_name))
                except Exception:
                    continue
            except Exception:
                continue
            if line_info is not None:
                return line_info
        return None

    def _refresh_backend_metadata(self) -> None:
        line_info = self._backend_line_info()
        if line_info is None:
            return

        backend_event_clock = getattr(line_info, "event_clock", None)
        backend_event_clock_name = getattr(backend_event_clock, "name", backend_event_clock)
        if isinstance(backend_event_clock_name, str):
            normalized = backend_event_clock_name.strip().lower().replace("_", "-")
            if normalized in (_VALID_EVENT_CLOCKS - {"auto"}):
                self._resolved_event_clock = normalized

    def _synthetic_timestamp_ns(self) -> int:
        if self._resolved_event_clock == "realtime":
            return time.time_ns()
        return time.monotonic_ns()

    def _normalize_raw_edge(self, raw_edge: Any, *, motion_detected: bool) -> str:
        if isinstance(raw_edge, str) and raw_edge.strip():
            normalized = raw_edge.strip().lower().replace("_", "-")
            aliases = {
                "pressed": "rising" if self.active_high else "falling",
                "released": "falling" if self.active_high else "rising",
                "active": "rising" if self.active_high else "falling",
                "inactive": "falling" if self.active_high else "rising",
                "both": "both",
                "level": "level",
            }
            return aliases.get(normalized, normalized)
        return "rising" if motion_detected else "falling"

    def _event_is_motion_detected(self, event: Any) -> bool:
        action = getattr(event, "action", None)
        if action == ButtonAction.PRESSED:
            return True
        if action != ButtonAction.PRESSED:
            released = getattr(ButtonAction, "RELEASED", object())
            if action == released:
                return False

        action_name = getattr(action, "name", None)
        if isinstance(action_name, str):
            normalized = action_name.strip().upper()
            if normalized == "PRESSED":
                return True
            if normalized == "RELEASED":
                return False

        if isinstance(action, str):
            normalized = action.strip().lower()
            if normalized in {"pressed", "active", "motion", "motion-detected"}:
                return True
            if normalized in {"released", "inactive", "no-motion", "motion-ended"}:
                return False

        raise PirMonitorError(f"Unexpected PIR backend action payload: {action!r}")

    def _make_level_event(self, motion_detected: bool, *, source: str = "level") -> PirMotionEvent:
        return PirMotionEvent(
            name=self.binding.name,
            line_offset=self.binding.line_offset,
            motion_detected=motion_detected,
            raw_edge="level",
            timestamp_ns=self._synthetic_timestamp_ns(),
            event_kind="motion-present" if motion_detected else "motion-absent",
            source=source,
            event_clock=self._resolved_event_clock,
        )

    def _to_motion_event(self, event: Any) -> PirMotionEvent:
        event_line_offset = getattr(event, "line_offset", self.binding.line_offset)
        if event_line_offset != self.binding.line_offset:
            raise PirMonitorError(
                f"Received PIR event for unexpected line {event_line_offset}; expected {self.binding.line_offset}"
            )

        motion_detected = self._event_is_motion_detected(event)
        raw_edge = self._normalize_raw_edge(
            getattr(event, "raw_edge", None),
            motion_detected=motion_detected,
        )
        sequence_no = getattr(event, "sequence_no", None)
        if sequence_no is None:
            sequence_no = getattr(event, "global_seqno", None)
        line_sequence_no = getattr(event, "line_sequence_no", None)
        if line_sequence_no is None:
            line_sequence_no = getattr(event, "line_seqno", None)

        return PirMotionEvent(
            name=self.binding.name,
            line_offset=self.binding.line_offset,
            motion_detected=motion_detected,
            raw_edge=raw_edge,
            timestamp_ns=int(getattr(event, "timestamp_ns")),
            event_kind="motion-start" if motion_detected else "motion-end",
            source="edge",
            event_clock=self._resolved_event_clock,
            sequence_no=sequence_no if isinstance(sequence_no, int) else None,
            line_sequence_no=line_sequence_no if isinstance(line_sequence_no, int) else None,
        )

    def _remember_event(self, event: PirMotionEvent) -> None:
        self._last_event = event
        if event.motion_detected:
            self._last_motion_monotonic_ns = time.monotonic_ns()

    def ready(self) -> bool:
        """Return whether the optional PIR warm-up window has elapsed."""

        self._ensure_open()
        if self._opened_monotonic_ns is None or self.warmup_s <= 0.0:
            return True
        warmup_deadline_ns = self._opened_monotonic_ns + int(self.warmup_s * 1_000_000_000)
        return time.monotonic_ns() >= warmup_deadline_ns

    def ready_in_s(self) -> float:
        """Return the remaining warm-up time in seconds."""

        self._ensure_open()
        if self._opened_monotonic_ns is None or self.warmup_s <= 0.0:
            return 0.0
        warmup_deadline_ns = self._opened_monotonic_ns + int(self.warmup_s * 1_000_000_000)
        remaining_ns = warmup_deadline_ns - time.monotonic_ns()
        return max(0.0, remaining_ns / 1_000_000_000)

    def open(self) -> "GpioPirMonitor":
        """Open the PIR monitor and prime its GPIO backend."""

        if self._is_open:
            self._refresh_backend_metadata()
            if self._opened_monotonic_ns is None:
                self._opened_monotonic_ns = time.monotonic_ns()
            return self
        if not self._manage_monitor_lifecycle:
            self._is_open = True
            self._opened_monotonic_ns = time.monotonic_ns()
            self._refresh_backend_metadata()
            return self
        try:
            self._monitor.open()
        except Exception as exc:
            raise PirMonitorError(f"Failed to open PIR monitor on {self.chip_name!r}") from exc
        self._is_open = True
        self._opened_monotonic_ns = time.monotonic_ns()
        self._last_motion_monotonic_ns = None
        self._last_event = None
        self._refresh_backend_metadata()
        return self

    def close(self) -> None:
        """Close the PIR monitor and release owned GPIO resources."""

        if not self._is_open:
            return
        try:
            if self._manage_monitor_lifecycle:
                self._monitor.close()
        except Exception as exc:
            self._is_open = False
            self._opened_monotonic_ns = None
            raise PirMonitorError(f"Failed to close PIR monitor on {self.chip_name!r}") from exc
        self._is_open = False
        self._opened_monotonic_ns = None

    def __enter__(self) -> "GpioPirMonitor":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
            return
        try:
            self.close()
        except Exception as close_exc:
            if exc is not None and hasattr(exc, "add_note"):
                exc.add_note(f"Failed to close PIR monitor cleanly: {close_exc!r}")

    def _maybe_accept_event(self, event: PirMotionEvent) -> PirMotionEvent | None:
        if not self.ready():
            return None
        self._remember_event(event)
        return event

    def poll(self, timeout: float | None = None) -> PirMotionEvent | None:
        """Return the next PIR event or ``None`` after the timeout."""

        self._ensure_open()
        timeout = _validate_non_negative_timeout("timeout", timeout)
        deadline_ns = _deadline_from_timeout(timeout)

        while True:
            current_timeout = _remaining_seconds(deadline_ns)
            try:
                raw_event = self._monitor.poll(timeout=current_timeout)
            except Exception as exc:
                raise PirMonitorError(f"Failed to poll PIR monitor on {self.chip_name!r}") from exc
            if raw_event is None:
                return None

            event = self._to_motion_event(raw_event)
            accepted_event = self._maybe_accept_event(event)
            if accepted_event is not None:
                return accepted_event

            if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
                return None

    async def poll_async(self, timeout: float | None = None) -> PirMotionEvent | None:
        """Asynchronously wait for the next PIR event."""

        self._ensure_open()
        timeout = _validate_non_negative_timeout("timeout", timeout)
        fd = self._backend_fd()
        if fd is None:
            return await asyncio.to_thread(self.poll, timeout)

        loop = asyncio.get_running_loop()
        waiter = loop.create_future()

        def _on_ready() -> None:
            if not waiter.done():
                waiter.set_result(None)

        loop.add_reader(fd, _on_ready)
        try:
            if timeout is None:
                await waiter
            else:
                try:
                    await asyncio.wait_for(waiter, timeout=timeout)
                except asyncio.TimeoutError:
                    return None
        finally:
            loop.remove_reader(fd)

        return self.poll(timeout=0.0)

    def iter_events(
        self,
        *,
        duration_s: float | None = None,
        poll_timeout: float = 0.5,
    ) -> Iterator[PirMotionEvent]:
        """Yield PIR motion events until the optional duration expires."""

        self._ensure_open()
        duration_s = _validate_non_negative_timeout("duration_s", duration_s)
        poll_timeout = _validate_non_negative_timeout("poll_timeout", poll_timeout) or 0.0
        deadline_ns = _deadline_from_timeout(duration_s)

        while True:
            remaining_s = _remaining_seconds(deadline_ns)
            if remaining_s is not None and remaining_s <= 0.0:
                return

            timeout = poll_timeout if remaining_s is None else min(poll_timeout, remaining_s)
            event = self.poll(timeout=timeout)
            if event is None:
                if remaining_s is not None and _remaining_seconds(deadline_ns) <= 0.0:
                    return
                continue
            yield event

    async def iter_events_async(
        self,
        *,
        duration_s: float | None = None,
        poll_timeout: float = 0.5,
    ) -> AsyncIterator[PirMotionEvent]:
        """Asynchronously yield PIR motion events until the optional duration expires."""

        self._ensure_open()
        duration_s = _validate_non_negative_timeout("duration_s", duration_s)
        poll_timeout = _validate_non_negative_timeout("poll_timeout", poll_timeout) or 0.0
        deadline_ns = _deadline_from_timeout(duration_s)

        while True:
            remaining_s = _remaining_seconds(deadline_ns)
            if remaining_s is not None and remaining_s <= 0.0:
                return

            timeout = poll_timeout if remaining_s is None else min(poll_timeout, remaining_s)
            event = await self.poll_async(timeout=timeout)
            if event is None:
                if remaining_s is not None and _remaining_seconds(deadline_ns) <= 0.0:
                    return
                continue
            yield event

    def snapshot_value(self) -> int:
        """Return the current raw PIR GPIO level as ``0`` or ``1``."""

        self._ensure_open()
        try:
            values = self._monitor.snapshot_values()
        except Exception as exc:
            raise PirMonitorError(f"Failed to snapshot PIR value on {self.chip_name!r}") from exc

        if self.binding.line_offset not in values:
            raise PirMonitorError(
                f"Configured PIR line {self.binding.line_offset} missing from GPIO snapshot"
            )

        return _coerce_binary_value(
            self.binding.line_offset,
            values[self.binding.line_offset],
        )

    def motion_detected(self) -> bool:
        """Return whether the current raw PIR level indicates active motion."""

        value = self.snapshot_value()
        return bool(value) if self.active_high else not bool(value)

    def occupancy_detected(self) -> bool:
        """Return filtered presence using the current level plus optional hold time."""

        self._ensure_open()
        if not self.ready():
            return False
        if self.motion_detected():
            return True
        if self.presence_hold_s <= 0.0 or self._last_motion_monotonic_ns is None:
            return False
        hold_ns = int(self.presence_hold_s * 1_000_000_000)
        return (time.monotonic_ns() - self._last_motion_monotonic_ns) <= hold_ns

    def diagnostics(self) -> PirMonitorDiagnostics:
        """Return runtime diagnostics plus any discoverable backend metadata."""

        line_info = self._backend_line_info()

        backend_consumer = None
        backend_used = None
        backend_edge_detection = None
        backend_bias = None
        backend_debounced = None
        backend_debounce_ms = None
        backend_event_clock = self._resolved_event_clock

        if line_info is not None:
            backend_consumer = getattr(line_info, "consumer", None)
            backend_used = getattr(line_info, "used", None)

            edge_detection = getattr(line_info, "edge_detection", None)
            edge_name = getattr(edge_detection, "name", edge_detection)
            if edge_name is not None:
                backend_edge_detection = str(edge_name).lower().replace("_", "-")

            bias = getattr(line_info, "bias", None)
            bias_name = getattr(bias, "name", bias)
            if bias_name is not None:
                backend_bias = str(bias_name).lower().replace("_", "-")

            backend_debounced = getattr(line_info, "debounced", None)

            debounce_period = getattr(line_info, "debounce_period", None)
            total_seconds = getattr(debounce_period, "total_seconds", None)
            if callable(total_seconds):
                backend_debounce_ms = total_seconds() * 1000.0

            event_clock = getattr(line_info, "event_clock", None)
            event_clock_name = getattr(event_clock, "name", event_clock)
            if event_clock_name is not None:
                backend_event_clock = str(event_clock_name).lower().replace("_", "-")

        return PirMonitorDiagnostics(
            chip_name=self.chip_name,
            line_offset=self.binding.line_offset,
            binding_name=self.binding.name,
            is_open=self._is_open,
            ready=self.ready() if self._is_open else False,
            ready_in_s=self.ready_in_s() if self._is_open else 0.0,
            active_high=self.active_high,
            bias=self.bias,
            debounce_ms=self.debounce_ms,
            consumer=self.consumer,
            warmup_s=self.warmup_s,
            presence_hold_s=self.presence_hold_s,
            event_clock=backend_event_clock,
            backend_fd=self._backend_fd() if self._is_open else None,
            backend_consumer=backend_consumer,
            backend_used=backend_used if isinstance(backend_used, bool) else None,
            backend_edge_detection=backend_edge_detection,
            backend_bias=backend_bias,
            backend_debounced=backend_debounced if isinstance(backend_debounced, bool) else None,
            backend_debounce_ms=backend_debounce_ms,
        )

    def wait_for_motion(
        self,
        *,
        duration_s: float = 12.0,
        poll_timeout: float = 0.25,
    ) -> PirMotionEvent | None:
        """Wait for the next motion event, including already-active motion."""

        self._ensure_open()
        duration_s = _validate_non_negative_timeout("duration_s", duration_s)
        poll_timeout = _validate_non_negative_timeout("poll_timeout", poll_timeout) or 0.0
        deadline_ns = _deadline_from_timeout(duration_s)

        while True:
            if self.ready() and self.motion_detected():
                event = self._make_level_event(True)
                self._remember_event(event)
                return event

            remaining_s = _remaining_seconds(deadline_ns)
            if remaining_s is not None and remaining_s <= 0.0:
                return None

            timeout = poll_timeout if remaining_s is None else min(poll_timeout, remaining_s)
            event = self.poll(timeout=timeout)
            if event is not None and event.motion_detected:
                return event

    async def wait_for_motion_async(
        self,
        *,
        duration_s: float = 12.0,
        poll_timeout: float = 0.25,
    ) -> PirMotionEvent | None:
        """Asynchronously wait for the next motion event, including already-active motion."""

        self._ensure_open()
        duration_s = _validate_non_negative_timeout("duration_s", duration_s)
        poll_timeout = _validate_non_negative_timeout("poll_timeout", poll_timeout) or 0.0
        deadline_ns = _deadline_from_timeout(duration_s)

        while True:
            if self.ready() and self.motion_detected():
                event = self._make_level_event(True)
                self._remember_event(event)
                return event

            remaining_s = _remaining_seconds(deadline_ns)
            if remaining_s is not None and remaining_s <= 0.0:
                return None

            timeout = poll_timeout if remaining_s is None else min(poll_timeout, remaining_s)
            event = await self.poll_async(timeout=timeout)
            if event is not None and event.motion_detected:
                return event

    def wait_for_no_motion(
        self,
        *,
        duration_s: float = 12.0,
        poll_timeout: float = 0.25,
    ) -> PirMotionEvent | None:
        """Wait for the line to become inactive, including already-inactive state."""

        self._ensure_open()
        duration_s = _validate_non_negative_timeout("duration_s", duration_s)
        poll_timeout = _validate_non_negative_timeout("poll_timeout", poll_timeout) or 0.0
        deadline_ns = _deadline_from_timeout(duration_s)

        while True:
            if not self.motion_detected():
                event = self._make_level_event(False)
                self._remember_event(event)
                return event

            remaining_s = _remaining_seconds(deadline_ns)
            if remaining_s is not None and remaining_s <= 0.0:
                return None

            timeout = poll_timeout if remaining_s is None else min(poll_timeout, remaining_s)
            event = self.poll(timeout=timeout)
            if event is not None and not event.motion_detected:
                return event

    async def wait_for_no_motion_async(
        self,
        *,
        duration_s: float = 12.0,
        poll_timeout: float = 0.25,
    ) -> PirMotionEvent | None:
        """Asynchronously wait for the line to become inactive."""

        self._ensure_open()
        duration_s = _validate_non_negative_timeout("duration_s", duration_s)
        poll_timeout = _validate_non_negative_timeout("poll_timeout", poll_timeout) or 0.0
        deadline_ns = _deadline_from_timeout(duration_s)

        while True:
            if not self.motion_detected():
                event = self._make_level_event(False)
                self._remember_event(event)
                return event

            remaining_s = _remaining_seconds(deadline_ns)
            if remaining_s is not None and remaining_s <= 0.0:
                return None

            timeout = poll_timeout if remaining_s is None else min(poll_timeout, remaining_s)
            event = await self.poll_async(timeout=timeout)
            if event is not None and not event.motion_detected:
                return event


def configured_pir_monitor(
    config: TwinrConfig,
    *,
    monitor: GpioButtonMonitor | None = None,
    close_injected_monitor: bool = False,
) -> GpioPirMonitor:
    """Build a PIR monitor from ``TwinrConfig`` and an optional injected monitor."""

    binding = build_pir_binding(config)
    return GpioPirMonitor(
        chip_name=config.gpio_chip,
        binding=binding,
        active_high=config.pir_active_high,
        bias=config.pir_bias,
        debounce_ms=config.pir_debounce_ms,
        consumer=getattr(config, "pir_consumer", "twinr-pir"),
        warmup_s=getattr(config, "pir_warmup_s", 0.0),
        presence_hold_s=getattr(config, "pir_presence_hold_s", 0.0),
        event_clock=getattr(config, "pir_event_clock", "auto"),
        allow_nonstandard_chip_path=getattr(config, "pir_allow_nonstandard_chip_path", False),
        monitor=monitor,
        close_injected_monitor=close_injected_monitor,
    )