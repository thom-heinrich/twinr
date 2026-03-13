from __future__ import annotations

from dataclasses import dataclass
import time

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.buttons import ButtonAction, ButtonBinding, GpioButtonMonitor


@dataclass(frozen=True, slots=True)
class PirBinding:
    name: str
    line_offset: int


@dataclass(frozen=True, slots=True)
class PirMotionEvent:
    name: str
    line_offset: int
    motion_detected: bool
    raw_edge: str
    timestamp_ns: int


def build_pir_binding(config: TwinrConfig) -> PirBinding:
    if config.pir_motion_gpio is None:
        raise ValueError("Set TWINR_PIR_MOTION_GPIO before watching PIR motion")
    return PirBinding(name="pir", line_offset=config.pir_motion_gpio)


class GpioPirMonitor:
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
    ) -> None:
        self.chip_name = chip_name
        self.binding = binding
        self.active_high = active_high
        self.bias = bias
        self.debounce_ms = debounce_ms
        self.consumer = consumer
        self._monitor = monitor or GpioButtonMonitor(
            chip_name=chip_name,
            bindings=(ButtonBinding(name=binding.name, line_offset=binding.line_offset),),
            active_low=not active_high,
            bias=bias,
            debounce_ms=debounce_ms,
            consumer=consumer,
        )

    def open(self) -> "GpioPirMonitor":
        self._monitor.open()
        return self

    def close(self) -> None:
        self._monitor.close()

    def __enter__(self) -> "GpioPirMonitor":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def poll(self, timeout: float | None = None) -> PirMotionEvent | None:
        event = self._monitor.poll(timeout=timeout)
        if event is None:
            return None
        return PirMotionEvent(
            name=self.binding.name,
            line_offset=self.binding.line_offset,
            motion_detected=(event.action == ButtonAction.PRESSED),
            raw_edge=event.raw_edge,
            timestamp_ns=event.timestamp_ns,
        )

    def iter_events(
        self,
        *,
        duration_s: float | None = None,
        poll_timeout: float = 0.5,
    ):
        for event in self._monitor.iter_events(duration_s=duration_s, poll_timeout=poll_timeout):
            yield PirMotionEvent(
                name=self.binding.name,
                line_offset=self.binding.line_offset,
                motion_detected=(event.action == ButtonAction.PRESSED),
                raw_edge=event.raw_edge,
                timestamp_ns=event.timestamp_ns,
            )

    def snapshot_value(self) -> int:
        values = self._monitor.snapshot_values()
        return int(values.get(self.binding.line_offset, 0))

    def motion_detected(self) -> bool:
        value = self.snapshot_value()
        return bool(value) if self.active_high else not bool(value)

    def wait_for_motion(
        self,
        *,
        duration_s: float = 12.0,
        poll_timeout: float = 0.25,
    ) -> PirMotionEvent | None:
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
) -> GpioPirMonitor:
    binding = build_pir_binding(config)
    return GpioPirMonitor(
        chip_name=config.gpio_chip,
        binding=binding,
        active_high=config.pir_active_high,
        bias=config.pir_bias,
        debounce_ms=config.pir_debounce_ms,
        monitor=monitor,
    )
