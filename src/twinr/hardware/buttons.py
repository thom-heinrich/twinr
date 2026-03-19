"""Monitor Twinr GPIO buttons across libgpiod and CLI fallback backends.

This module owns low-level button edge normalization and raw state snapshots.
PIR motion wraps this backend instead of duplicating GPIO access and debounce
logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from pathlib import Path
from queue import Empty, Queue
from shutil import which
from threading import Event as ThreadEvent, Lock, Thread
from typing import Iterator
import logging
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


_CLI_READ_TIMEOUT_S = 2.0
_SAMPLE_INTERVAL_S = 0.01

_LOGGER = logging.getLogger(__name__)
_GPIOGET_CLI_SPECS: dict[str, "_GpiogetCliSpec"] = {}


class ButtonAction(StrEnum):
    """Describe the normalized logical action for a button transition."""

    PRESSED = "pressed"
    RELEASED = "released"


@dataclass(frozen=True, slots=True)
class ButtonBinding:
    """Bind a logical button name to one GPIO line offset."""

    name: str
    line_offset: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Button name must not be empty")
        _validate_line_offset(self.line_offset, field_name=f"{self.name} button GPIO")


@dataclass(frozen=True, slots=True)
class ButtonEvent:
    """Represent one normalized button event."""

    name: str
    line_offset: int
    action: ButtonAction
    raw_edge: str
    timestamp_ns: int


@dataclass(frozen=True, slots=True)
class _GpiogetCliSpec:
    """Describe which named flags one installed ``gpioget`` binary supports."""

    supports_chip_option: bool = True
    supports_numeric_option: bool = True


def _require_gpiod():
    if gpiod is None:
        raise RuntimeError("python3-gpiod is required for GPIO button monitoring")
    return gpiod


def _has_modern_gpiod_api(module: object | None) -> bool:
    # AUDIT-FIX(#1): Detect the supported libgpiod request/edge API instead of hard-coding the legacy object model.
    if module is None:
        return False
    chip_type = getattr(module, "Chip", None)
    has_request_api = bool(
        callable(getattr(module, "request_lines", None))
        or (chip_type is not None and hasattr(chip_type, "request_lines"))
    )
    return bool(getattr(module, "LineSettings", None) and has_request_api)


def _has_legacy_gpiod_api(module: object | None) -> bool:
    if module is None:
        return False
    chip_type = getattr(module, "Chip", None)
    return bool(chip_type is not None and hasattr(chip_type, "get_line"))


def _validate_line_offset(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer GPIO line offset")
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return value


def _normalize_chip_name(chip_name: str) -> str:
    # AUDIT-FIX(#3): Validate chip identifiers up front so bad .env values fail deterministically during startup.
    if not isinstance(chip_name, str):
        raise TypeError("GPIO chip name must be a string")
    normalized = chip_name.strip()
    if not normalized:
        raise ValueError("GPIO chip name must not be empty")
    return normalized


def _chip_path_for_modern_gpiod(chip_name: str) -> str:
    normalized = _normalize_chip_name(chip_name)
    if normalized.startswith("/"):
        return normalized
    if normalized.isdigit():
        return f"/dev/gpiochip{normalized}"
    if normalized.startswith("gpiochip"):
        return f"/dev/{normalized}"
    return normalized


def _normalize_timeout(timeout: float | None) -> float | None:
    if timeout is None:
        return None
    return max(0.0, float(timeout))


def _normalize_bias_name(bias: str) -> str:
    normalized = bias.strip().lower()
    mapping = {
        "": "as-is",
        "as-is": "as-is",
        "none": "as-is",
        "disable": "disabled",
        "disabled": "disabled",
        "pull-up": "pull-up",
        "pull-down": "pull-down",
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported button bias: {bias}")
    return mapping[normalized]


def _normalize_edge_type(event_type: object) -> int:
    # AUDIT-FIX(#8): Normalize legacy and modern libgpiod edge enums plus CLI-style numeric/string variants into one stable rising/falling representation.
    module = gpiod
    if module is not None:
        edge_event_cls = getattr(module, "EdgeEvent", None)
        edge_event_type = getattr(edge_event_cls, "Type", None) if edge_event_cls is not None else None
        if edge_event_type is not None:
            if event_type == edge_event_type.RISING_EDGE:
                return 1
            if event_type == edge_event_type.FALLING_EDGE:
                return 0

        line_event_cls = getattr(module, "LineEvent", None)
        if line_event_cls is not None:
            if event_type == line_event_cls.RISING_EDGE:
                return 1
            if event_type == line_event_cls.FALLING_EDGE:
                return 0

    if isinstance(event_type, str):
        normalized = event_type.strip().lower()
        if normalized in {"1", "rising", "rise"}:
            return 1
        if normalized in {"0", "2", "falling", "fall"}:
            return 0
        return -1

    raw_value = getattr(event_type, "value", event_type)
    try:
        numeric = int(raw_value)
    except (TypeError, ValueError):
        return -1

    if numeric == 1:
        return 1
    if numeric in {0, 2}:
        return 0
    return numeric


def _coerce_line_value(value: object) -> int:
    raw_value = getattr(value, "value", value)
    if isinstance(raw_value, bool):
        return int(raw_value)
    if isinstance(raw_value, int):
        return 1 if raw_value else 0

    name = getattr(value, "name", None)
    if isinstance(name, str):
        normalized_name = name.strip().upper()
        if normalized_name == "ACTIVE":
            return 1
        if normalized_name == "INACTIVE":
            return 0

    normalized = str(value).strip().lower()
    if normalized in {"1", "active", "value.active", "line.value.active"}:
        return 1
    if normalized in {"0", "inactive", "value.inactive", "line.value.inactive"}:
        return 0
    raise ValueError(f"Unsupported GPIO line value: {value!r}")


def _bias_flags(bias: str) -> int:
    module = _require_gpiod()
    normalized = _normalize_bias_name(bias)
    mapping = {
        "as-is": 0,
        "disabled": module.LINE_REQ_FLAG_BIAS_DISABLE,
        "pull-up": module.LINE_REQ_FLAG_BIAS_PULL_UP,
        "pull-down": module.LINE_REQ_FLAG_BIAS_PULL_DOWN,
    }
    return mapping[normalized]


def _line_settings_bias(module: object, bias: str) -> object:
    line_module = getattr(module, "line", module)
    bias_enum = getattr(line_module, "Bias", None)
    if bias_enum is None:
        raise RuntimeError("Installed gpiod does not expose Bias settings")
    normalized = _normalize_bias_name(bias)
    mapping = {
        "as-is": bias_enum.AS_IS,
        "disabled": bias_enum.DISABLED,
        "pull-up": bias_enum.PULL_UP,
        "pull-down": bias_enum.PULL_DOWN,
    }
    return mapping[normalized]


def _cli_bias_arg(bias: str) -> str | None:
    normalized = _normalize_bias_name(bias)
    if normalized == "as-is":
        return None
    return normalized


def _detect_gpioget_cli_spec(gpioget_path: str) -> _GpiogetCliSpec:
    """Probe and cache the supported ``gpioget`` CLI shape for one binary path."""

    cached = _GPIOGET_CLI_SPECS.get(gpioget_path)
    if cached is not None:
        return cached

    spec = _GpiogetCliSpec()
    try:
        completed = subprocess.run(
            [gpioget_path, "--help"],
            capture_output=True,
            text=True,
            timeout=_CLI_READ_TIMEOUT_S,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        # If help probing is unavailable, keep the newer CLI as the initial guess
        # and let the execution path fall back when the binary rejects named flags.
        pass
    else:
        help_text = " ".join(
            part
            for part in (completed.stdout, completed.stderr)
            if isinstance(part, str) and part.strip()
        )
        spec = _GpiogetCliSpec(
            supports_chip_option="--chip" in help_text,
            supports_numeric_option="--numeric" in help_text,
        )

    _GPIOGET_CLI_SPECS[gpioget_path] = spec
    return spec


def _build_gpioget_command(
    gpioget_path: str,
    *,
    chip_name: str,
    bindings: tuple[ButtonBinding, ...],
    bias: str,
) -> list[str]:
    """Build the correct ``gpioget`` invocation for the detected CLI generation."""

    cli_spec = _detect_gpioget_cli_spec(gpioget_path)
    command = [gpioget_path]
    cli_bias = _cli_bias_arg(bias)
    if cli_bias is not None:
        command.append(f"--bias={cli_bias}")

    if cli_spec.supports_chip_option:
        command.extend(["--chip", chip_name])
    else:
        command.append(chip_name)

    if cli_spec.supports_numeric_option:
        command.append("--numeric")

    command.extend(str(binding.line_offset) for binding in bindings)
    return command


def _gpioget_rejected_named_options(command: list[str], exc: subprocess.CalledProcessError) -> bool:
    """Return whether ``gpioget`` explicitly rejected the newer named-option CLI."""

    if "--chip" not in command and "--numeric" not in command:
        return False
    stderr = exc.stderr.strip().lower() if isinstance(exc.stderr, str) else ""
    return "unrecognized option" in stderr and ("--chip" in stderr or "--numeric" in stderr)


def edge_name(event_type: object) -> str:
    """Return a stable rising/falling label for a raw GPIO edge value."""

    normalized = _normalize_edge_type(event_type)
    if normalized == 1:
        return "rising"
    if normalized == 0:
        return "falling"
    return "unknown"


def edge_to_action(event_type: object, active_low: bool) -> ButtonAction:
    """Convert a raw GPIO edge into a normalized button action."""

    is_rising = _normalize_edge_type(event_type) == 1
    if active_low:
        return ButtonAction.RELEASED if is_rising else ButtonAction.PRESSED
    return ButtonAction.PRESSED if is_rising else ButtonAction.RELEASED


def build_button_bindings(config: TwinrConfig) -> tuple[ButtonBinding, ...]:
    """Build configured Twinr button bindings from ``TwinrConfig``."""

    # AUDIT-FIX(#3): Reject invalid/negative GPIO offsets before the monitor touches hardware.
    bindings: list[ButtonBinding] = []
    if config.green_button_gpio is not None:
        bindings.append(
            ButtonBinding(
                name="green",
                line_offset=_validate_line_offset(
                    config.green_button_gpio,
                    field_name="Green button GPIO",
                ),
            )
        )
    if config.yellow_button_gpio is not None:
        bindings.append(
            ButtonBinding(
                name="yellow",
                line_offset=_validate_line_offset(
                    config.yellow_button_gpio,
                    field_name="Yellow button GPIO",
                ),
            )
        )

    line_offsets = [binding.line_offset for binding in bindings]
    if len(line_offsets) != len(set(line_offsets)):
        raise ValueError("Green and yellow buttons must use distinct GPIO lines")
    return tuple(bindings)


def build_probe_bindings(lines: tuple[int, ...] | list[int]) -> tuple[ButtonBinding, ...]:
    """Build synthetic button bindings for GPIO probe utilities."""

    # AUDIT-FIX(#3): Probe bindings use the same offset validation as production bindings.
    validated_lines = sorted(
        {_validate_line_offset(line, field_name="Probe GPIO") for line in lines}
    )
    return tuple(ButtonBinding(name=f"gpio{line_offset}", line_offset=line_offset) for line_offset in validated_lines)


class GpioButtonMonitor:
    """Monitor Twinr button GPIO lines across supported Linux backends."""

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
        if isinstance(debounce_ms, bool) or not isinstance(debounce_ms, int):
            raise TypeError("debounce_ms must be an integer")
        if debounce_ms < 0:
            raise ValueError("debounce_ms must be >= 0")

        # AUDIT-FIX(#3): Normalize and validate configuration early so invalid GPIO values fail fast.
        self.chip_name = _normalize_chip_name(chip_name)
        self.bindings = bindings
        self.active_low = active_low
        self.bias = _normalize_bias_name(bias)
        self.debounce_ms = debounce_ms
        self.consumer = consumer
        self.debounce_ns = debounce_ms * 1_000_000

        self._chip = None
        self._request = None
        self._request_poller = None
        self._line_by_offset: dict[int, object] = {}
        self._last_event_ns: dict[int, int] = {}
        self._last_values: dict[int, int] = {}
        self._binding_by_offset = {binding.line_offset: binding for binding in bindings}
        self._pending_events: Queue[ButtonEvent] = Queue()
        self._pending_events_lock = Lock()
        self._sampler_state_lock = Lock()
        self._background_poll_stop: ThreadEvent | None = None
        self._background_poll_thread: Thread | None = None
        self._background_poll_error: Exception | None = None
        self._cli_tools: dict[str, str] = {}

    def _is_open(self) -> bool:
        return bool(
            self._request is not None
            or self._chip is not None
            or self._line_by_offset
            or self._last_values
        )

    def open(self) -> "GpioButtonMonitor":
        """Open the configured GPIO backend and prime button state."""

        if self._is_open():
            return self

        module = gpiod
        if _has_modern_gpiod_api(module):
            # AUDIT-FIX(#1): Support the current event-driven libgpiod Python API instead of assuming legacy objects only.
            self._open_modern_gpiod()
            return self

        if _has_legacy_gpiod_api(module):
            # AUDIT-FIX(#5): Legacy gpiod path now validates bias before opening the chip to avoid leaking descriptors on config errors.
            self._open_legacy_gpiod()
            self._start_background_sampler()
            return self

        # AUDIT-FIX(#4): Resolve and cache absolute CLI tool paths once, then reuse them instead of executing PATH-dependent bare names.
        self._resolve_cli_tool("gpioget")
        # AUDIT-FIX(#2): Prime the CLI fallback using the corrected gpioget invocation so failures happen during open(), not first poll().
        self._last_values = self._read_cli_values()
        self._start_background_sampler()
        return self

    def _open_modern_gpiod(self) -> None:
        # AUDIT-FIX(#1): Use libgpiod's request_lines/read_edge_events path so modern installs work and button events are edge-driven.
        module = _require_gpiod()
        line_module = getattr(module, "line", module)

        direction_enum = getattr(line_module, "Direction", None)
        edge_enum = getattr(line_module, "Edge", None)
        if direction_enum is None or edge_enum is None:
            raise RuntimeError("Installed gpiod does not expose Direction/Edge settings")

        settings = module.LineSettings(
            direction=direction_enum.INPUT,
            edge_detection=edge_enum.BOTH,
            bias=_line_settings_bias(module, self.bias),
            debounce_period=timedelta(milliseconds=self.debounce_ms),
        )
        offsets = tuple(binding.line_offset for binding in self.bindings)
        request_config = {offsets: settings}
        chip_path = _chip_path_for_modern_gpiod(self.chip_name)

        request_lines = getattr(module, "request_lines", None)
        if callable(request_lines):
            request = request_lines(
                chip_path,
                config=request_config,
                consumer=self.consumer,
            )
            self._request = request
            self._chip = None
        else:
            chip = module.Chip(chip_path)
            try:
                self._request = chip.request_lines(
                    config=request_config,
                    consumer=self.consumer,
                )
            except Exception:
                chip.close()
                raise
            self._chip = chip

        request_fd = getattr(self._request, "fd", None)
        if isinstance(request_fd, int):
            poller = select.poll()
            poller.register(request_fd, select.POLLIN | select.POLLPRI | select.POLLERR | select.POLLHUP)
            self._request_poller = poller

        self._last_values = self._read_request_values()

    def _open_legacy_gpiod(self) -> None:
        # AUDIT-FIX(#5): Legacy acquisition is staged through locals so partial open failures cannot leak live chip/line handles.
        module = _require_gpiod()
        flags = _bias_flags(self.bias)

        chip = None
        line_by_offset: dict[int, object] = {}
        last_values: dict[int, int] = {}
        try:
            chip = module.Chip(self.chip_name)
            for binding in self.bindings:
                line = chip.get_line(binding.line_offset)
                line.request(self.consumer, module.LINE_REQ_DIR_IN, flags)
                line_by_offset[binding.line_offset] = line
                last_values[binding.line_offset] = _coerce_line_value(line.get_value())
        except Exception:
            for line in line_by_offset.values():
                try:
                    line.release()
                except Exception:
                    _LOGGER.warning(
                        "Failed to release GPIO line during legacy button-monitor rollback.",
                        exc_info=True,
                    )
            if chip is not None:
                try:
                    chip.close()
                except Exception:
                    _LOGGER.warning(
                        "Failed to close GPIO chip during legacy button-monitor rollback.",
                        exc_info=True,
                    )
            raise

        self._chip = chip
        self._line_by_offset = line_by_offset
        self._last_values = last_values

    def close(self) -> None:
        """Release any open GPIO handles and stop background sampling."""

        # AUDIT-FIX(#6): Cleanup is best-effort and clears internal state even if one release call fails, so exception paths do not leak hardware handles.
        request = self._request
        line_by_offset = self._line_by_offset
        chip = self._chip
        background_poll_stop = self._background_poll_stop
        background_poll_thread = self._background_poll_thread

        self._request = None
        self._request_poller = None
        self._line_by_offset = {}
        self._chip = None
        self._background_poll_stop = None
        self._background_poll_thread = None
        self._background_poll_error = None
        self._last_event_ns.clear()
        self._last_values.clear()
        if background_poll_stop is not None:
            background_poll_stop.set()
        if background_poll_thread is not None and background_poll_thread.is_alive():
            background_poll_thread.join(timeout=max(0.2, self.debounce_ms / 1000.0 + (_SAMPLE_INTERVAL_S * 4)))
        while True:
            try:
                self._pending_events.get_nowait()
            except Empty:
                break

        if request is not None:
            try:
                request.release()
            except Exception:
                _LOGGER.warning("Failed to release GPIO request during button-monitor shutdown.", exc_info=True)

        for line in line_by_offset.values():
            try:
                line.release()
            except Exception:
                _LOGGER.warning("Failed to release GPIO line during button-monitor shutdown.", exc_info=True)

        if chip is not None:
            try:
                chip.close()
            except Exception:
                _LOGGER.warning("Failed to close GPIO chip during button-monitor shutdown.", exc_info=True)

    def __enter__(self) -> "GpioButtonMonitor":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _resolve_cli_tool(self, tool_name: str) -> str:
        # AUDIT-FIX(#4): Cache the absolute CLI tool path once to avoid PATH-based tool swapping after the presence check.
        tool_path = self._cli_tools.get(tool_name)
        if tool_path is not None:
            return tool_path

        resolved = which(tool_name)
        if resolved is None:
            raise RuntimeError(f"Install `{tool_name}` to monitor GPIO buttons without python3-gpiod")
        self._cli_tools[tool_name] = resolved
        return resolved

    def _read_request_values(self) -> dict[int, int]:
        if self._request is None:
            return {}

        offsets = list(getattr(self._request, "offsets", []))
        values = list(self._request.get_values())
        if not offsets:
            offsets = [binding.line_offset for binding in self.bindings]
        if len(values) != len(offsets):
            raise RuntimeError(
                f"gpiod returned {len(values)} values for {len(offsets)} offsets"
            )
        return {
            int(offset): _coerce_line_value(value)
            for offset, value in zip(offsets, values, strict=False)
        }

    def _wait_for_request_events(self, timeout: float | None) -> bool:
        if self._request is None:
            return False

        normalized_timeout = _normalize_timeout(timeout)

        if self._request_poller is not None:
            timeout_ms = None if normalized_timeout is None else int(normalized_timeout * 1000)
            events = self._request_poller.poll(timeout_ms)
            if not events:
                return False
            for _, mask in events:
                if mask & (select.POLLERR | select.POLLHUP):
                    raise RuntimeError("GPIO edge-event request reported POLLERR/POLLHUP")
            return True

        wait_edge_events = getattr(self._request, "wait_edge_events", None)
        if callable(wait_edge_events):
            return bool(wait_edge_events(normalized_timeout))

        return True

    def _queue_request_events(self) -> None:
        if self._request is None:
            return

        raw_events = list(self._request.read_edge_events())
        for raw_event in raw_events:
            line_offset = int(raw_event.line_offset)
            binding = self._binding_by_offset.get(line_offset)
            if binding is None:
                continue

            normalized_edge = _normalize_edge_type(raw_event.event_type)
            if normalized_edge not in {0, 1}:
                continue

            timestamp_ns = int(raw_event.timestamp_ns)
            last_timestamp = self._last_event_ns.get(line_offset)
            current_value = 1 if normalized_edge == 1 else 0
            if last_timestamp is not None and timestamp_ns - last_timestamp < self.debounce_ns:
                self._last_values[line_offset] = current_value
                continue

            self._last_event_ns[line_offset] = timestamp_ns
            self._last_values[line_offset] = current_value
            self._enqueue_pending_event(
                ButtonEvent(
                    name=binding.name,
                    line_offset=line_offset,
                    action=edge_to_action(raw_event.event_type, self.active_low),
                    raw_edge=edge_name(raw_event.event_type),
                    timestamp_ns=timestamp_ns,
                )
            )

    def _enqueue_pending_event(self, event: ButtonEvent) -> None:
        with self._pending_events_lock:
            self._pending_events.put_nowait(event)

    def _dequeue_pending_event(self) -> ButtonEvent | None:
        with self._pending_events_lock:
            try:
                return self._pending_events.get_nowait()
            except Empty:
                return None

    def _collect_level_change_events(
        self,
        *,
        timestamp_ns: int,
        current_values: dict[int, int],
    ) -> list[ButtonEvent]:
        events: list[ButtonEvent] = []
        with self._sampler_state_lock:
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
                events.append(
                    ButtonEvent(
                        name=binding.name,
                        line_offset=binding.line_offset,
                        action=edge_to_action(event_type, self.active_low),
                        raw_edge=edge_name(event_type),
                        timestamp_ns=timestamp_ns,
                    )
                )
        return events

    def _background_sampling_enabled(self) -> bool:
        return self._request is None and (bool(self._line_by_offset) or bool(self._cli_tools))

    def _start_background_sampler(self) -> None:
        if not self._background_sampling_enabled():
            return
        if self._background_poll_thread is not None:
            return
        stop_event = ThreadEvent()
        self._background_poll_stop = stop_event
        self._background_poll_error = None
        self._background_poll_thread = Thread(
            target=self._background_poll_worker,
            name=f"{self.consumer}-sampler",
            daemon=True,
        )
        self._background_poll_thread.start()

    def _background_poll_worker(self) -> None:
        stop_event = self._background_poll_stop
        if stop_event is None:
            return
        try:
            while not stop_event.wait(_SAMPLE_INTERVAL_S):
                timestamp_ns = time.monotonic_ns()
                current_values = self._read_current_values()
                for event in self._collect_level_change_events(
                    timestamp_ns=timestamp_ns,
                    current_values=current_values,
                ):
                    self._enqueue_pending_event(event)
        except Exception as exc:
            self._background_poll_error = exc

    def poll(self, timeout: float | None = None) -> ButtonEvent | None:
        """Return the next button event or ``None`` after the timeout."""

        if not self._is_open():
            self.open()

        if not self._last_values and self._request is None and not self._line_by_offset:
            return None

        normalized_timeout = _normalize_timeout(timeout)

        queued_event = self._dequeue_pending_event()
        if queued_event is not None:
            return queued_event

        if self._request is not None:
            deadline = None if normalized_timeout is None else time.monotonic() + normalized_timeout
            while True:
                queued_event = self._dequeue_pending_event()
                if queued_event is not None:
                    return queued_event

                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                if deadline is not None and remaining <= 0:
                    return None

                if not self._wait_for_request_events(remaining):
                    return None
                self._queue_request_events()

        if self._background_poll_thread is not None:
            if self._background_poll_error is not None:
                raise RuntimeError("Legacy GPIO button sampler failed") from self._background_poll_error
            if normalized_timeout == 0.0:
                return self._dequeue_pending_event()
            try:
                return self._pending_events.get(timeout=normalized_timeout)
            except Empty:
                if self._background_poll_error is not None:
                    raise RuntimeError("Legacy GPIO button sampler failed") from self._background_poll_error
                return None

        deadline = None if normalized_timeout is None else time.monotonic() + normalized_timeout

        while True:
            timestamp_ns = time.monotonic_ns()
            current_values = self._read_current_values()
            events = self._collect_level_change_events(
                timestamp_ns=timestamp_ns,
                current_values=current_values,
            )
            if events:
                return events[0]

            if deadline is not None and time.monotonic() >= deadline:
                return None
            sleep_for = _SAMPLE_INTERVAL_S if deadline is None else min(_SAMPLE_INTERVAL_S, max(0.0, deadline - time.monotonic()))
            if sleep_for > 0:
                time.sleep(sleep_for)

    def _read_current_values(self) -> dict[int, int]:
        if self._request is not None:
            return self._read_request_values()
        if self._line_by_offset:
            return {
                line_offset: _coerce_line_value(line.get_value())
                for line_offset, line in self._line_by_offset.items()
            }
        return self._read_cli_values()

    def _read_cli_values(self) -> dict[int, int]:
        gpioget_path = self._resolve_cli_tool("gpioget")
        command = _build_gpioget_command(
            gpioget_path,
            chip_name=self.chip_name,
            bindings=self.bindings,
            bias=self.bias,
        )

        # AUDIT-FIX(#2): Probe and cache the installed gpioget CLI shape so both
        # legacy libgpiod v1 positional syntax and newer named-option syntax stay supported.
        # AUDIT-FIX(#7): Bound CLI reads with a timeout and capture stderr so transient GPIO/tool issues do not hang the monitor forever.
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=_CLI_READ_TIMEOUT_S,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            if _gpioget_rejected_named_options(command, exc):
                _GPIOGET_CLI_SPECS[gpioget_path] = _GpiogetCliSpec(
                    supports_chip_option=False,
                    supports_numeric_option=False,
                )
                command = _build_gpioget_command(
                    gpioget_path,
                    chip_name=self.chip_name,
                    bindings=self.bindings,
                    bias=self.bias,
                )
                try:
                    completed = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        timeout=_CLI_READ_TIMEOUT_S,
                        check=True,
                    )
                except subprocess.CalledProcessError as retry_exc:
                    stderr = retry_exc.stderr.strip() if isinstance(retry_exc.stderr, str) else ""
                    detail = f": {stderr}" if stderr else ""
                    raise RuntimeError(
                        f"gpioget failed with exit code {retry_exc.returncode}{detail}"
                    ) from retry_exc
            else:
                stderr = exc.stderr.strip() if isinstance(exc.stderr, str) else ""
                detail = f": {stderr}" if stderr else ""
                raise RuntimeError(f"gpioget failed with exit code {exc.returncode}{detail}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Timed out while reading GPIO values via gpioget") from exc
        except OSError as exc:
            raise RuntimeError("Unable to execute gpioget for GPIO button monitoring") from exc
        output = completed.stdout.strip()
        if not output:
            raise RuntimeError("gpioget returned no GPIO values")

        tokens = output.split()
        # AUDIT-FIX(#9): Validate the CLI output shape explicitly so malformed tool output becomes a clear error instead of IndexError/ValueError noise.
        if len(tokens) != len(self.bindings):
            raise RuntimeError(
                f"gpioget returned {len(tokens)} values for {len(self.bindings)} requested GPIO lines: {output!r}"
            )

        values: list[int] = []
        for token in tokens:
            if token not in {"0", "1"}:
                raise RuntimeError(f"gpioget returned non-numeric GPIO value: {token!r}")
            values.append(int(token))

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
        """Yield button events until the optional duration expires."""

        normalized_duration = None if duration_s is None else max(0.0, float(duration_s))
        # AUDIT-FIX(#10): Clamp non-positive poll timeouts so callers cannot accidentally create a hot spin-loop.
        normalized_poll_timeout = max(0.01, float(poll_timeout))

        deadline = None if normalized_duration is None else time.monotonic() + normalized_duration
        while True:
            timeout = normalized_poll_timeout
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                timeout = min(timeout, remaining)
            event = self.poll(timeout=timeout)
            if event is not None:
                yield event

    def snapshot_values(self) -> dict[int, int]:
        """Return the latest raw GPIO values keyed by line offset."""

        if not self._is_open():
            self.open()
        values = self._read_current_values()
        self._last_values.update(values)
        return dict(values)


def configured_button_monitor(config: TwinrConfig) -> GpioButtonMonitor:
    """Build a button monitor from ``TwinrConfig``."""

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
