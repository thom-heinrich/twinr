"""Adapt PIR motion sensing onto Twinr's button-monitor GPIO backend.

This module keeps PIR handling thin by reusing ``twinr.hardware.buttons`` for
GPIO access, edge normalization, and backend selection while exposing PIR-
specific motion semantics to callers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math  # AUDIT-FIX(#1): Validate numeric inputs before they reach the GPIO layer.
import time
from typing import Iterator  # AUDIT-FIX(#6): Expose a precise iterator type for static analysis.

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.buttons import ButtonAction, ButtonBinding, GpioButtonMonitor


class PirMonitorError(RuntimeError):
    """Raised when the PIR monitor cannot be opened or read safely."""


def _validate_non_empty_str(name: str, value: str) -> str:
    # AUDIT-FIX(#1): Reject empty identifiers up front instead of failing deep in the hardware backend.
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _validate_non_negative_int(name: str, value: int) -> int:
    # AUDIT-FIX(#1): Reject negative and bool inputs before they silently turn into bad GPIO settings.
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _validate_non_negative_timeout(name: str, value: float | None) -> float | None:
    # AUDIT-FIX(#1): Guard against negative, NaN, and infinite timeouts that can hang or explode at runtime.
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


def build_pir_binding(config: TwinrConfig) -> PirBinding:
    """Build the configured PIR binding from ``TwinrConfig``."""

    if config.pir_motion_gpio is None:
        raise ValueError("Set TWINR_PIR_MOTION_GPIO before watching PIR motion")
    line_offset = _validate_non_negative_int(  # AUDIT-FIX(#1): Fail early on invalid GPIO line offsets.
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
        monitor: GpioButtonMonitor | None = None,
        close_injected_monitor: bool = False,
    ) -> None:
        self.chip_name = _validate_non_empty_str(  # AUDIT-FIX(#1): Validate identifiers before opening the device.
            "chip_name",
            chip_name,
        )
        if not isinstance(binding, PirBinding):
            raise TypeError("binding must be a PirBinding")
        _validate_non_empty_str("binding.name", binding.name)  # AUDIT-FIX(#1): Reject empty binding names early.
        _validate_non_negative_int(  # AUDIT-FIX(#1): Reject invalid binding offsets before constructing ButtonBinding.
            "binding.line_offset",
            binding.line_offset,
        )
        if not isinstance(active_high, bool):
            raise ValueError("active_high must be a bool")  # AUDIT-FIX(#1): Prevent silent truthiness coercion.
        if not isinstance(close_injected_monitor, bool):
            raise ValueError("close_injected_monitor must be a bool")  # AUDIT-FIX(#1): Prevent silent truthiness coercion.
        self.binding = binding
        self.active_high = active_high
        self.bias = _validate_non_empty_str("bias", bias)  # AUDIT-FIX(#1): Reject empty bias values early.
        self.debounce_ms = _validate_non_negative_int(  # AUDIT-FIX(#1): Reject invalid debounce values before hardware setup.
            "debounce_ms",
            debounce_ms,
        )
        self.consumer = _validate_non_empty_str("consumer", consumer)  # AUDIT-FIX(#1): Reject empty consumer labels early.
        self._manage_monitor_lifecycle = (  # AUDIT-FIX(#2): Make injected-monitor ownership explicit.
            monitor is None or close_injected_monitor
        )
        self._is_open = monitor is not None and not self._manage_monitor_lifecycle  # AUDIT-FIX(#4): Treat caller-owned injected monitors as caller-managed state.
        self._monitor = (
            monitor
            if monitor is not None
            else GpioButtonMonitor(  # AUDIT-FIX(#2): Use an explicit None-check; falsey monitor objects must not be discarded.
                chip_name=self.chip_name,
                bindings=(ButtonBinding(name=binding.name, line_offset=binding.line_offset),),
                active_low=not active_high,
                bias=self.bias,
                debounce_ms=self.debounce_ms,
                consumer=self.consumer,
            )
        )

    def _ensure_open(self) -> None:
        # AUDIT-FIX(#4): Raise a consistent domain error instead of leaking backend-specific state failures.
        if not self._is_open:
            raise PirMonitorError("PIR monitor is not open; call open() before reading motion events")

    def _to_motion_event(self, event) -> PirMotionEvent:
        # AUDIT-FIX(#6): Centralize event translation so poll() and iter_events() cannot drift.
        return PirMotionEvent(
            name=self.binding.name,
            line_offset=self.binding.line_offset,
            motion_detected=(event.action == ButtonAction.PRESSED),
            raw_edge=event.raw_edge,
            timestamp_ns=event.timestamp_ns,
        )

    def open(self) -> "GpioPirMonitor":
        """Open the PIR monitor and prime its GPIO backend."""

        if self._is_open:
            return self
        if not self._manage_monitor_lifecycle:
            self._is_open = True  # AUDIT-FIX(#2): Re-open the wrapper state without touching caller-owned GPIO resources.
            return self
        try:
            self._monitor.open()  # AUDIT-FIX(#4): Make open() idempotent and wrap hardware failures consistently.
        except Exception as exc:
            raise PirMonitorError(f"Failed to open PIR monitor on {self.chip_name!r}") from exc
        self._is_open = True
        return self

    def close(self) -> None:
        """Close the PIR monitor and release owned GPIO resources."""

        if not self._is_open:
            return
        try:
            if self._manage_monitor_lifecycle:
                self._monitor.close()  # AUDIT-FIX(#2): Do not close injected monitors unless ownership was explicit.
        except Exception as exc:
            self._is_open = False
            raise PirMonitorError(f"Failed to close PIR monitor on {self.chip_name!r}") from exc
        self._is_open = False

    def __enter__(self) -> "GpioPirMonitor":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
            return
        try:
            self.close()  # AUDIT-FIX(#5): Preserve the original exception when cleanup also fails.
        except Exception as close_exc:
            if exc is not None:
                exc.add_note(f"Failed to close PIR monitor cleanly: {close_exc!r}")

    def poll(self, timeout: float | None = None) -> PirMotionEvent | None:
        """Return the next PIR event or ``None`` after the timeout."""

        self._ensure_open()  # AUDIT-FIX(#4): Enforce a clear lifecycle contract for read calls.
        timeout = _validate_non_negative_timeout("timeout", timeout)  # AUDIT-FIX(#1): Reject invalid poll timeouts early.
        try:
            event = self._monitor.poll(timeout=timeout)
        except Exception as exc:
            raise PirMonitorError(f"Failed to poll PIR monitor on {self.chip_name!r}") from exc
        if event is None:
            return None
        return self._to_motion_event(event)

    def iter_events(
        self,
        *,
        duration_s: float | None = None,
        poll_timeout: float = 0.5,
    ) -> Iterator[PirMotionEvent]:
        """Yield PIR motion events until the optional duration expires."""

        self._ensure_open()  # AUDIT-FIX(#4): Enforce a clear lifecycle contract for streaming reads.
        duration_s = _validate_non_negative_timeout("duration_s", duration_s)  # AUDIT-FIX(#1): Reject invalid durations early.
        poll_timeout = _validate_non_negative_timeout("poll_timeout", poll_timeout)  # AUDIT-FIX(#1): Reject invalid poll timeouts early.
        try:
            for event in self._monitor.iter_events(duration_s=duration_s, poll_timeout=poll_timeout):
                yield self._to_motion_event(event)
        except Exception as exc:
            raise PirMonitorError(f"Failed while iterating PIR events on {self.chip_name!r}") from exc

    def snapshot_value(self) -> int:
        """Return the current raw PIR GPIO level as ``0`` or ``1``."""

        self._ensure_open()  # AUDIT-FIX(#4): Enforce a clear lifecycle contract for snapshot reads.
        try:
            values = self._monitor.snapshot_values()
        except Exception as exc:
            raise PirMonitorError(f"Failed to snapshot PIR value on {self.chip_name!r}") from exc
        if self.binding.line_offset not in values:
            raise PirMonitorError(  # AUDIT-FIX(#3): Missing GPIO state is a hard fault, not a silent 'no motion'.
                f"Configured PIR line {self.binding.line_offset} missing from GPIO snapshot"
            )
        raw_value = values[self.binding.line_offset]
        if isinstance(raw_value, bool):
            return int(raw_value)
        if isinstance(raw_value, int) and raw_value in (0, 1):
            return raw_value
        raise PirMonitorError(  # AUDIT-FIX(#3): Reject malformed snapshot payloads instead of silently coercing them.
            f"Unexpected snapshot value for PIR line {self.binding.line_offset}: {raw_value!r}"
        )

    def motion_detected(self) -> bool:
        """Return whether the current PIR level indicates active motion."""

        value = self.snapshot_value()
        return bool(value) if self.active_high else not bool(value)

    def wait_for_motion(
        self,
        *,
        duration_s: float = 12.0,
        poll_timeout: float = 0.25,
    ) -> PirMotionEvent | None:
        """Wait for the next motion event, including already-active motion."""

        duration_s = _validate_non_negative_timeout("duration_s", duration_s)  # AUDIT-FIX(#1): Reject invalid wait durations early.
        poll_timeout = _validate_non_negative_timeout("poll_timeout", poll_timeout)  # AUDIT-FIX(#1): Reject invalid wait poll timeouts early.
        if self.motion_detected():
            return PirMotionEvent(
                name=self.binding.name,
                line_offset=self.binding.line_offset,
                motion_detected=True,
                raw_edge="level",
                timestamp_ns=time.monotonic_ns(),
            )
        for event in self.iter_events(duration_s=duration_s, poll_timeout=poll_timeout):
            if event.motion_detected:
                return event
        return None


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
        monitor=monitor,
        close_injected_monitor=close_injected_monitor,  # AUDIT-FIX(#2): Preserve explicit caller control over injected monitor lifetime.
    )
