from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from shutil import which
from typing import Iterator
import select
import subprocess
import sys
import time

from twinr.agent.base_agent.config import TwinrConfig

try:
    import gpiod
except ImportError:  # pragma: no cover - handled at runtime on non-Pi systems
    system_dist_packages = Path("/usr/lib/python3/dist-packages")
    if system_dist_packages.exists():
        sys.path.append(str(system_dist_packages))
        try:
            import gpiod  # type: ignore[no-redef]
        except ImportError:
            gpiod = None
    else:
        gpiod = None


class ButtonAction(StrEnum):
    PRESSED = "pressed"
    RELEASED = "released"


@dataclass(frozen=True, slots=True)
class ButtonBinding:
    name: str
    line_offset: int


@dataclass(frozen=True, slots=True)
class ButtonEvent:
    name: str
    line_offset: int
    action: ButtonAction
    raw_edge: str
    timestamp_ns: int


def _require_gpiod():
    if gpiod is None:
        raise RuntimeError("python3-gpiod is required for GPIO button monitoring")
    return gpiod


def _normalize_edge_type(event_type: int) -> int:
    if gpiod is not None:
        if event_type == gpiod.LineEvent.RISING_EDGE:
            return 1
        if event_type == gpiod.LineEvent.FALLING_EDGE:
            return 0
    return int(event_type)


def _bias_flags(bias: str) -> int:
    module = _require_gpiod()
    normalized = bias.strip().lower()
    mapping = {
        "as-is": 0,
        "none": 0,
        "disable": module.LINE_REQ_FLAG_BIAS_DISABLE,
        "disabled": module.LINE_REQ_FLAG_BIAS_DISABLE,
        "pull-up": module.LINE_REQ_FLAG_BIAS_PULL_UP,
        "pull-down": module.LINE_REQ_FLAG_BIAS_PULL_DOWN,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported button bias: {bias}")
    return mapping[normalized]


def edge_name(event_type: int) -> str:
    normalized = _normalize_edge_type(event_type)
    if normalized == 1:
        return "rising"
    if normalized == 0:
        return "falling"
    return "unknown"


def edge_to_action(event_type: int, active_low: bool) -> ButtonAction:
    is_rising = _normalize_edge_type(event_type) == 1
    if active_low:
        return ButtonAction.RELEASED if is_rising else ButtonAction.PRESSED
    return ButtonAction.PRESSED if is_rising else ButtonAction.RELEASED


def build_button_bindings(config: TwinrConfig) -> tuple[ButtonBinding, ...]:
    bindings: list[ButtonBinding] = []
    if config.green_button_gpio is not None:
        bindings.append(ButtonBinding(name="green", line_offset=config.green_button_gpio))
    if config.yellow_button_gpio is not None:
        bindings.append(ButtonBinding(name="yellow", line_offset=config.yellow_button_gpio))

    line_offsets = [binding.line_offset for binding in bindings]
    if len(line_offsets) != len(set(line_offsets)):
        raise ValueError("Green and yellow buttons must use distinct GPIO lines")
    return tuple(bindings)


def build_probe_bindings(lines: tuple[int, ...] | list[int]) -> tuple[ButtonBinding, ...]:
    unique_lines = sorted(set(lines))
    return tuple(ButtonBinding(name=f"gpio{line_offset}", line_offset=line_offset) for line_offset in unique_lines)


class GpioButtonMonitor:
    def __init__(
        self,
        chip_name: str,
        bindings: tuple[ButtonBinding, ...],
        *,
        active_low: bool = True,
        bias: str = "pull-up",
        debounce_ms: int = 80,
        consumer: str = "twinr-buttons",
    ) -> None:
        if not bindings:
            raise ValueError("At least one button binding is required")
        self.chip_name = chip_name
        self.bindings = bindings
        self.active_low = active_low
        self.bias = bias
        self.debounce_ms = debounce_ms
        self.consumer = consumer
        self.debounce_ns = debounce_ms * 1_000_000
        self._chip = None
        self._line_by_offset: dict[int, object] = {}
        self._last_event_ns: dict[int, int] = {}
        self._last_values: dict[int, int] = {}
        self._binding_by_offset = {binding.line_offset: binding for binding in bindings}

    def open(self) -> "GpioButtonMonitor":
        if self._chip is not None or self._last_values:
            return self

        if gpiod is None:
            if which("gpioget") is None:
                raise RuntimeError("Install `python3-gpiod` or `gpioget` to monitor GPIO buttons")
            self._last_values = self._read_cli_values()
            return self

        module = _require_gpiod()
        self._chip = module.Chip(self.chip_name)
        flags = _bias_flags(self.bias)

        try:
            for binding in self.bindings:
                line = self._chip.get_line(binding.line_offset)
                line.request(self.consumer, module.LINE_REQ_DIR_IN, flags)
                self._line_by_offset[binding.line_offset] = line
                self._last_values[binding.line_offset] = int(line.get_value())
        except Exception:
            self.close()
            raise

        return self

    def close(self) -> None:
        for line in self._line_by_offset.values():
            line.release()
        self._line_by_offset.clear()
        self._last_event_ns.clear()
        self._last_values.clear()
        if self._chip is not None:
            self._chip.close()
            self._chip = None

    def __enter__(self) -> "GpioButtonMonitor":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def poll(self, timeout: float | None = None) -> ButtonEvent | None:
        if self._chip is None and not self._last_values:
            self.open()

        if not self._last_values:
            return None

        deadline = None if timeout is None else time.monotonic() + timeout

        while True:
            timestamp_ns = time.monotonic_ns()
            current_values = self._read_current_values()
            for binding in self.bindings:
                previous_value = self._last_values.get(binding.line_offset)
                current_value = current_values.get(binding.line_offset)
                if previous_value is None or current_value is None or current_value == previous_value:
                    continue
                last_timestamp = self._last_event_ns.get(binding.line_offset)
                if last_timestamp is not None and timestamp_ns - last_timestamp < self.debounce_ns:
                    continue
                self._last_event_ns[binding.line_offset] = timestamp_ns
                self._last_values[binding.line_offset] = current_value
                event_type = 1 if current_value > previous_value else 0
                return ButtonEvent(
                    name=binding.name,
                    line_offset=binding.line_offset,
                    action=edge_to_action(event_type, self.active_low),
                    raw_edge=edge_name(event_type),
                    timestamp_ns=timestamp_ns,
                )

            if deadline is not None and time.monotonic() >= deadline:
                return None
            time.sleep(0.01 if deadline is None else min(0.01, max(0.0, deadline - time.monotonic())))

    def _read_current_values(self) -> dict[int, int]:
        if self._line_by_offset:
            return {
                line_offset: int(line.get_value())
                for line_offset, line in self._line_by_offset.items()
            }
        return self._read_cli_values()

    def _read_cli_values(self) -> dict[int, int]:
        command = [
            "gpioget",
            f"--bias={self.bias}",
            self.chip_name,
            *(str(binding.line_offset) for binding in self.bindings),
        ]
        output = subprocess.check_output(command, text=True).strip()
        values = [int(value) for value in output.split()]
        return {
            binding.line_offset: values[index]
            for index, binding in enumerate(self.bindings)
        }

    def iter_events(
        self,
        *,
        duration_s: float | None = None,
        poll_timeout: float = 0.5,
    ) -> Iterator[ButtonEvent]:
        deadline = None if duration_s is None else time.monotonic() + duration_s
        while True:
            timeout = poll_timeout
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                timeout = min(timeout, remaining)
            event = self.poll(timeout=timeout)
            if event is not None:
                yield event


def configured_button_monitor(config: TwinrConfig) -> GpioButtonMonitor:
    bindings = build_button_bindings(config)
    if not bindings:
        raise ValueError("Set TWINR_GREEN_BUTTON_GPIO and TWINR_YELLOW_BUTTON_GPIO before watching buttons")
    return GpioButtonMonitor(
        chip_name=config.gpio_chip,
        bindings=bindings,
        active_low=config.button_active_low,
        bias=config.button_bias,
        debounce_ms=config.button_debounce_ms,
    )
