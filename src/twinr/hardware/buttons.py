# CHANGELOG: 2026-03-28
# BUG-1: snapshot_values() no longer mutates the event-detection baseline, which previously allowed concurrent snapshots to drop real button presses/releases.
# BUG-2: close() now stops background workers before releasing GPIO handles/processes, fixing shutdown races and post-close reads against released resources.
# BUG-3: sampler-based debounce is now stability-based instead of first-edge throttling, preventing false presses/releases from short glitches and mechanical bounce.
# BUG-4: CLI compatibility now handles libgpiod v1/v2 bias spelling differences ('disable' vs 'disabled') instead of failing on mixed deployments.
# SEC-1: CLI tool resolution now prefers trusted system paths before PATH lookup, reducing practical PATH-hijack risk for services that spawn gpioget/gpiomon.
# SEC-2: pending button events are now bounded with drop-oldest backpressure, preventing memory exhaustion if a line chatters or is physically abused.
# IMP-1: CLI fallback is now event-driven via gpiomon when available, replacing high-rate gpioget subprocess polling on Raspberry Pi-class hardware.
# IMP-2: modern libgpiod requests now configure kernel event buffers and detect sequence gaps so event loss becomes visible and the state cache is re-synced.
# IMP-3: CLI fallback supports both libgpiod v1.6-style positional chip syntax and v2-style named options for gpioget/gpiomon.

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
from queue import Empty, Full, Queue
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
_GPIOMON_STARTUP_TIMEOUT_S = 0.15
_LEVEL_SAMPLE_INTERVAL_S = 0.01
_CLI_SAMPLE_INTERVAL_S = 0.02
_THREAD_JOIN_TIMEOUT_S = _CLI_READ_TIMEOUT_S + 0.5
_DEFAULT_EVENT_BUFFER_SIZE = 64
_DEFAULT_MAX_PENDING_EVENTS = 256
_BACKGROUND_THREAD_WAKEUP_S = 0.5

_LOGGER = logging.getLogger(__name__)
_GPIOGET_CLI_SPECS: dict[str, "_GpiogetCliSpec"] = {}
_GPIOMON_CLI_SPECS: dict[str, "_GpiomonCliSpec"] = {}
_CLI_SPEC_LOCK = Lock()
_TRUSTED_CLI_TOOL_DIRS = (
    Path("/usr/bin"),
    Path("/usr/sbin"),
    Path("/bin"),
    Path("/sbin"),
    Path("/usr/local/bin"),
    Path("/usr/local/sbin"),
)


def _is_trusted_cli_path(path: str) -> bool:
    try:
        resolved_parent = Path(path).resolve().parent
    except OSError:
        return False
    return resolved_parent in _TRUSTED_CLI_TOOL_DIRS


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
    """Describe which gpioget CLI flavor one installed binary supports."""

    supports_chip_option: bool = True
    supports_numeric_option: bool = True
    disabled_bias_keyword: str = "disabled"


@dataclass(frozen=True, slots=True)
class _GpiomonCliSpec:
    """Describe which gpiomon CLI flavor one installed binary supports."""

    supports_chip_option: bool = True
    supports_consumer_option: bool = True
    supports_edges_option: bool = True
    supports_event_clock_option: bool = True
    supports_debounce_option: bool = True
    supports_line_buffered_option: bool = False
    disabled_bias_keyword: str = "disabled"


def _require_gpiod():
    if gpiod is None:
        raise RuntimeError("python3-gpiod is required for GPIO button monitoring")
    return gpiod


def _has_modern_gpiod_api(module: object | None) -> bool:
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
    if not isinstance(bias, str):
        raise TypeError("GPIO button bias must be a string")
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


def _line_settings_clock(module: object) -> object | None:
    line_module = getattr(module, "line", module)
    clock_enum = getattr(line_module, "Clock", None)
    if clock_enum is None:
        return None
    return getattr(clock_enum, "MONOTONIC", None)


def _cli_bias_arg(bias: str, *, disabled_keyword: str) -> str | None:
    normalized = _normalize_bias_name(bias)
    if normalized == "as-is":
        return None
    if normalized == "disabled":
        return disabled_keyword
    return normalized


def _detect_gpioget_cli_spec(gpioget_path: str) -> _GpiogetCliSpec:
    with _CLI_SPEC_LOCK:
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
        pass
    else:
        help_text = " ".join(
            part
            for part in (completed.stdout, completed.stderr)
            if isinstance(part, str) and part.strip()
        ).lower()
        spec = _GpiogetCliSpec(
            supports_chip_option="--chip" in help_text,
            supports_numeric_option="--numeric" in help_text,
            disabled_bias_keyword="disable" if "disable" in help_text and "disabled" not in help_text else "disabled",
        )

    with _CLI_SPEC_LOCK:
        _GPIOGET_CLI_SPECS[gpioget_path] = spec
    return spec


def _detect_gpiomon_cli_spec(gpiomon_path: str) -> _GpiomonCliSpec:
    with _CLI_SPEC_LOCK:
        cached = _GPIOMON_CLI_SPECS.get(gpiomon_path)
    if cached is not None:
        return cached

    spec = _GpiomonCliSpec()
    try:
        completed = subprocess.run(
            [gpiomon_path, "--help"],
            capture_output=True,
            text=True,
            timeout=_CLI_READ_TIMEOUT_S,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass
    else:
        help_text = " ".join(
            part
            for part in (completed.stdout, completed.stderr)
            if isinstance(part, str) and part.strip()
        ).lower()
        spec = _GpiomonCliSpec(
            supports_chip_option="--chip" in help_text,
            supports_consumer_option="--consumer" in help_text,
            supports_edges_option="--edges" in help_text,
            supports_event_clock_option="--event-clock" in help_text,
            supports_debounce_option="--debounce-period" in help_text,
            supports_line_buffered_option="line-buffered" in help_text,
            disabled_bias_keyword="disable" if "disable" in help_text and "disabled" not in help_text else "disabled",
        )

    with _CLI_SPEC_LOCK:
        _GPIOMON_CLI_SPECS[gpiomon_path] = spec
    return spec


def _store_gpioget_cli_spec(gpioget_path: str, spec: _GpiogetCliSpec) -> None:
    with _CLI_SPEC_LOCK:
        _GPIOGET_CLI_SPECS[gpioget_path] = spec


def _store_gpiomon_cli_spec(gpiomon_path: str, spec: _GpiomonCliSpec) -> None:
    with _CLI_SPEC_LOCK:
        _GPIOMON_CLI_SPECS[gpiomon_path] = spec


def _legacy_gpioget_cli_spec() -> _GpiogetCliSpec:
    return _GpiogetCliSpec(
        supports_chip_option=False,
        supports_numeric_option=False,
        disabled_bias_keyword="disable",
    )


def _legacy_gpiomon_cli_spec() -> _GpiomonCliSpec:
    return _GpiomonCliSpec(
        supports_chip_option=False,
        supports_consumer_option=False,
        supports_edges_option=False,
        supports_event_clock_option=False,
        supports_debounce_option=False,
        supports_line_buffered_option=True,
        disabled_bias_keyword="disable",
    )


def _build_gpioget_command(
    gpioget_path: str,
    *,
    chip_name: str,
    bindings: tuple[ButtonBinding, ...],
    bias: str,
) -> list[str]:
    cli_spec = _detect_gpioget_cli_spec(gpioget_path)
    command = [gpioget_path]

    cli_bias = _cli_bias_arg(bias, disabled_keyword=cli_spec.disabled_bias_keyword)
    if cli_bias is not None:
        command.extend(["--bias", cli_bias])

    if cli_spec.supports_chip_option:
        command.extend(["--chip", chip_name])
    else:
        command.append(chip_name)

    if cli_spec.supports_numeric_option:
        command.append("--numeric")

    command.extend(str(binding.line_offset) for binding in bindings)
    return command


def _build_gpiomon_command(
    gpiomon_path: str,
    *,
    chip_name: str,
    bindings: tuple[ButtonBinding, ...],
    bias: str,
    consumer: str,
    debounce_ms: int,
) -> list[str]:
    cli_spec = _detect_gpiomon_cli_spec(gpiomon_path)
    command = [gpiomon_path]
    positional_args: list[str] = []

    cli_bias = _cli_bias_arg(bias, disabled_keyword=cli_spec.disabled_bias_keyword)
    if cli_bias is not None:
        command.extend(["--bias", cli_bias])

    if cli_spec.supports_chip_option:
        command.extend(["--chip", chip_name])
    else:
        positional_args.append(chip_name)

    if cli_spec.supports_consumer_option:
        command.extend(["--consumer", consumer])

    if cli_spec.supports_edges_option:
        command.extend(["--edges", "both"])

    if cli_spec.supports_event_clock_option:
        command.extend(["--event-clock", "monotonic"])

    if cli_spec.supports_debounce_option and debounce_ms > 0:
        command.extend(["--debounce-period", f"{debounce_ms}ms"])

    if cli_spec.supports_line_buffered_option:
        command.append("--line-buffered")

    command.extend(["--format", "%o\t%e"])
    positional_args.extend(str(binding.line_offset) for binding in bindings)
    command.extend(positional_args)
    return command


def _command_mentions_named_cli(command: list[str], names: tuple[str, ...]) -> bool:
    return any(arg in names for arg in command)


def _command_uses_disabled_bias(command: list[str]) -> bool:
    for index, part in enumerate(command):
        if part == "--bias" and index + 1 < len(command):
            return command[index + 1] == "disabled"
        if part.startswith("--bias="):
            return part.partition("=")[2] == "disabled"
    return False


def _cli_rejected_named_options(command: list[str], stderr: str) -> bool:
    normalized_stderr = stderr.lower()
    if "unrecognized option" not in normalized_stderr:
        return False
    return _command_mentions_named_cli(
        command,
        names=("--chip", "--numeric", "--consumer", "--edges", "--event-clock", "--debounce-period"),
    )


def _cli_rejected_disabled_bias(command: list[str], stderr: str) -> bool:
    normalized_stderr = stderr.lower()
    if not _command_uses_disabled_bias(command):
        return False
    return "bias" in normalized_stderr and any(
        marker in normalized_stderr
        for marker in ("invalid", "unknown", "unrecognized", "bad")
    )


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

    validated_lines = sorted(
        {_validate_line_offset(line, field_name="Probe GPIO") for line in lines}
    )
    return tuple(
        ButtonBinding(name=f"gpio{line_offset}", line_offset=line_offset)
        for line_offset in validated_lines
    )


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
        event_buffer_size: int = _DEFAULT_EVENT_BUFFER_SIZE,
        max_pending_events: int = _DEFAULT_MAX_PENDING_EVENTS,
    ) -> None:
        if not bindings:
            raise ValueError("At least one button binding is required")
        if isinstance(debounce_ms, bool) or not isinstance(debounce_ms, int):
            raise TypeError("debounce_ms must be an integer")
        if debounce_ms < 0:
            raise ValueError("debounce_ms must be >= 0")
        if isinstance(event_buffer_size, bool) or not isinstance(event_buffer_size, int):
            raise TypeError("event_buffer_size must be an integer")
        if event_buffer_size < 0:
            raise ValueError("event_buffer_size must be >= 0")
        if isinstance(max_pending_events, bool) or not isinstance(max_pending_events, int):
            raise TypeError("max_pending_events must be an integer")
        if max_pending_events <= 0:
            raise ValueError("max_pending_events must be > 0")

        self.chip_name = _normalize_chip_name(chip_name)
        self.bindings = bindings
        self.active_low = active_low
        self.bias = _normalize_bias_name(bias)
        self.debounce_ms = debounce_ms
        self.consumer = consumer
        self.event_buffer_size = event_buffer_size
        self.max_pending_events = max_pending_events
        self.debounce_ns = debounce_ms * 1_000_000

        self._chip = None
        self._request = None
        self._request_poller = None
        self._line_by_offset: dict[int, object] = {}
        self._binding_by_offset = {binding.line_offset: binding for binding in bindings}
        self._pending_events: Queue[ButtonEvent] = Queue(maxsize=max_pending_events)
        self._pending_events_lock = Lock()
        self._sampler_state_lock = Lock()
        self._background_poll_stop: ThreadEvent | None = None
        self._background_poll_thread: Thread | None = None
        self._background_poll_error: Exception | None = None
        self._background_poll_interval_s = _LEVEL_SAMPLE_INTERVAL_S
        self._cli_tools: dict[str, str] = {}
        self._last_values: dict[int, int] = {}
        self._debounce_candidates: dict[int, tuple[int, int]] = {}
        self._last_global_seqno: int | None = None
        self._last_line_seqno: dict[int, int] = {}
        self._gpiomon_process: subprocess.Popen[str] | None = None
        self._gpiomon_uses_kernel_debounce = False
        self._backend_name = "unopened"
        self._dropped_pending_events = 0

    @property
    def backend(self) -> str:
        """Return the active backend label."""

        return self._backend_name

    @property
    def dropped_events(self) -> int:
        """Return how many pending events were dropped under backpressure."""

        return self._dropped_pending_events

    def _is_open(self) -> bool:
        return bool(
            self._request is not None
            or self._chip is not None
            or self._line_by_offset
            or self._gpiomon_process is not None
            or self._background_poll_thread is not None
            or self._last_values
        )

    def _prime_state(self, values: dict[int, int]) -> None:
        with self._sampler_state_lock:
            self._last_values = dict(values)
            self._debounce_candidates.clear()
            self._last_line_seqno.clear()
            self._last_global_seqno = None

    def open(self) -> "GpioButtonMonitor":
        """Open the configured GPIO backend and prime button state."""

        if self._is_open():
            return self

        module = gpiod
        if _has_modern_gpiod_api(module):
            self._open_modern_gpiod()
            self._backend_name = "gpiod-modern"
            return self

        if self._open_gpiomon_cli_if_available():
            return self

        if _has_legacy_gpiod_api(module):
            self._open_legacy_gpiod()
            self._backend_name = "gpiod-legacy-sampler"
            self._background_poll_interval_s = _LEVEL_SAMPLE_INTERVAL_S
            self._start_background_sampler()
            return self

        self._resolve_cli_tool("gpioget")
        self._prime_state(self._read_cli_values())
        self._backend_name = "gpioget-cli-sampler"
        self._background_poll_interval_s = _CLI_SAMPLE_INTERVAL_S
        self._start_background_sampler()
        return self

    def _open_modern_gpiod(self) -> None:
        module = _require_gpiod()
        line_module = getattr(module, "line", module)

        direction_enum = getattr(line_module, "Direction", None)
        edge_enum = getattr(line_module, "Edge", None)
        if direction_enum is None or edge_enum is None:
            raise RuntimeError("Installed gpiod does not expose Direction/Edge settings")

        settings_kwargs: dict[str, object] = {
            "direction": direction_enum.INPUT,
            "edge_detection": edge_enum.BOTH,
            "bias": _line_settings_bias(module, self.bias),
            "debounce_period": timedelta(milliseconds=self.debounce_ms),
        }
        event_clock = _line_settings_clock(module)
        if event_clock is not None:
            settings_kwargs["event_clock"] = event_clock

        settings = module.LineSettings(**settings_kwargs)
        offsets = tuple(binding.line_offset for binding in self.bindings)
        request_config = {offsets: settings}
        chip_path = _chip_path_for_modern_gpiod(self.chip_name)

        is_gpiochip_device = getattr(module, "is_gpiochip_device", None)
        if chip_path.startswith("/") and callable(is_gpiochip_device) and not is_gpiochip_device(chip_path):
            raise ValueError(f"GPIO chip path is not a gpiochip device: {chip_path}")

        request_lines = getattr(module, "request_lines", None)
        if callable(request_lines):
            request = self._request_modern_lines(
                request_lines,
                chip_path,
                config=request_config,
                consumer=self.consumer,
            )
            self._request = request
            self._chip = None
        else:
            chip = module.Chip(chip_path)
            try:
                self._request = self._request_modern_lines(
                    chip.request_lines,
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

        self._prime_state(self._read_request_values())

    def _request_modern_lines(self, request_callable, *args, **kwargs):
        if self.event_buffer_size > 0:
            try:
                return request_callable(*args, **kwargs, event_buffer_size=self.event_buffer_size)
            except TypeError:
                pass
        return request_callable(*args, **kwargs)

    def _open_legacy_gpiod(self) -> None:
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
        self._prime_state(last_values)

    def _open_gpiomon_cli_if_available(self) -> bool:
        try:
            self._resolve_cli_tool("gpioget")
            gpiomon_path = self._resolve_cli_tool("gpiomon")
        except RuntimeError:
            return False

        self._prime_state(self._read_cli_values())
        self._start_gpiomon_process(gpiomon_path)
        self._backend_name = "gpiomon-cli"
        return True

    def close(self) -> None:
        """Release any open GPIO handles and stop background work."""

        background_poll_stop = self._background_poll_stop
        background_poll_thread = self._background_poll_thread
        gpiomon_process = self._gpiomon_process
        request = self._request
        line_by_offset = self._line_by_offset
        chip = self._chip

        self._request = None
        self._request_poller = None
        self._line_by_offset = {}
        self._chip = None
        self._background_poll_stop = None
        self._background_poll_thread = None
        self._gpiomon_process = None

        if background_poll_stop is not None:
            background_poll_stop.set()

        if gpiomon_process is not None:
            self._terminate_process(gpiomon_process)

        if background_poll_thread is not None and background_poll_thread.is_alive():
            background_poll_thread.join(timeout=_THREAD_JOIN_TIMEOUT_S)
            if background_poll_thread.is_alive() and gpiomon_process is not None:
                self._kill_process(gpiomon_process)
                background_poll_thread.join(timeout=_THREAD_JOIN_TIMEOUT_S)

        self._background_poll_error = None
        self._backend_name = "closed"
        self._gpiomon_uses_kernel_debounce = False
        self._debounce_candidates.clear()
        self._last_line_seqno.clear()
        self._last_global_seqno = None
        self._last_values.clear()

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
        tool_path = self._cli_tools.get(tool_name)
        if tool_path is not None:
            return tool_path

        for base_dir in _TRUSTED_CLI_TOOL_DIRS:
            candidate = base_dir / tool_name
            if candidate.is_file() and candidate.stat().st_mode & 0o111:
                resolved = str(candidate)
                self._cli_tools[tool_name] = resolved
                return resolved

        resolved = which(tool_name)
        if resolved is None:
            raise RuntimeError(f"Install `{tool_name}` to monitor GPIO buttons without python3-gpiod")

        resolved_path = str(Path(resolved).resolve())
        self._cli_tools[tool_name] = resolved_path
        if not _is_trusted_cli_path(resolved_path):
            _LOGGER.warning(
                "Resolved GPIO helper %s outside trusted system directories: %s",
                tool_name,
                resolved_path,
            )
        return resolved_path

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
            return bool(wait_edge_events(normalized_timeout))  # pylint: disable=not-callable

        return True

    def _queue_request_events(self) -> None:
        if self._request is None:
            return

        raw_events = list(self._request.read_edge_events())
        queued_events: list[ButtonEvent] = []
        loss_detected = False

        with self._sampler_state_lock:
            for raw_event in raw_events:
                line_offset = int(raw_event.line_offset)
                binding = self._binding_by_offset.get(line_offset)
                if binding is None:
                    continue

                normalized_edge = _normalize_edge_type(raw_event.event_type)
                if normalized_edge not in {0, 1}:
                    continue

                global_seqno = getattr(raw_event, "global_seqno", None)
                if isinstance(global_seqno, int):
                    if self._last_global_seqno is not None and global_seqno != self._last_global_seqno + 1:
                        loss_detected = True
                    self._last_global_seqno = global_seqno

                line_seqno = getattr(raw_event, "line_seqno", None)
                if isinstance(line_seqno, int):
                    previous_line_seqno = self._last_line_seqno.get(line_offset)
                    if previous_line_seqno is not None and line_seqno != previous_line_seqno + 1:
                        loss_detected = True
                    self._last_line_seqno[line_offset] = line_seqno

                current_value = 1 if normalized_edge == 1 else 0
                self._debounce_candidates.pop(line_offset, None)
                self._last_values[line_offset] = current_value
                queued_events.append(
                    ButtonEvent(
                        name=binding.name,
                        line_offset=line_offset,
                        action=edge_to_action(raw_event.event_type, self.active_low),
                        raw_edge=edge_name(raw_event.event_type),
                        timestamp_ns=int(raw_event.timestamp_ns),
                    )
                )

        if loss_detected:
            _LOGGER.warning(
                "Detected a GPIO event sequence gap on %s; re-synchronizing cached line values.",
                self.chip_name,
            )
            refreshed_values = self._read_request_values()
            with self._sampler_state_lock:
                self._last_values.update(refreshed_values)
                for line_offset in refreshed_values:
                    self._debounce_candidates.pop(line_offset, None)

        for event in queued_events:
            self._enqueue_pending_event(event)

    def _enqueue_pending_event(self, event: ButtonEvent) -> None:
        with self._pending_events_lock:
            while True:
                try:
                    self._pending_events.put_nowait(event)
                    return
                except Full:
                    try:
                        self._pending_events.get_nowait()
                    except Empty:
                        continue
                    self._dropped_pending_events += 1
                    if self._dropped_pending_events in {1, 10, 100} or self._dropped_pending_events % 1000 == 0:
                        _LOGGER.warning(
                            "Dropping oldest pending GPIO event because the queue is full (%s drops so far).",
                            self._dropped_pending_events,
                        )

    def _dequeue_pending_event(self) -> ButtonEvent | None:
        with self._pending_events_lock:
            try:
                return self._pending_events.get_nowait()
            except Empty:
                return None

    def _debounced_event_timestamp_ns(self, candidate_since_ns: int, observed_ns: int) -> int:
        if self.debounce_ns <= 0:
            return observed_ns
        return min(observed_ns, candidate_since_ns + self.debounce_ns)

    def _stage_candidate_locked(
        self,
        binding: ButtonBinding,
        *,
        current_value: int,
        timestamp_ns: int,
    ) -> list[ButtonEvent]:
        line_offset = binding.line_offset
        stable_value = self._last_values.get(line_offset)

        if stable_value is None:
            self._last_values[line_offset] = current_value
            self._debounce_candidates.pop(line_offset, None)
            return []

        if current_value == stable_value:
            self._debounce_candidates.pop(line_offset, None)
            return []

        candidate = self._debounce_candidates.get(line_offset)
        if candidate is None or candidate[0] != current_value:
            self._debounce_candidates[line_offset] = (current_value, timestamp_ns)
            candidate = (current_value, timestamp_ns)

        candidate_since_ns = candidate[1]
        if self.debounce_ns > 0 and timestamp_ns - candidate_since_ns < self.debounce_ns:
            return []

        self._last_values[line_offset] = current_value
        self._debounce_candidates.pop(line_offset, None)
        event_type = 1 if current_value > stable_value else 0
        event_timestamp_ns = self._debounced_event_timestamp_ns(candidate_since_ns, timestamp_ns)
        return [
            ButtonEvent(
                name=binding.name,
                line_offset=line_offset,
                action=edge_to_action(event_type, self.active_low),
                raw_edge=edge_name(event_type),
                timestamp_ns=event_timestamp_ns,
            )
        ]

    def _flush_due_debounce_candidates(self, *, now_ns: int) -> list[ButtonEvent]:
        events: list[ButtonEvent] = []
        with self._sampler_state_lock:
            for binding in self.bindings:
                candidate = self._debounce_candidates.get(binding.line_offset)
                if candidate is None:
                    continue
                current_value, candidate_since_ns = candidate
                if self.debounce_ns > 0 and now_ns - candidate_since_ns < self.debounce_ns:
                    continue
                events.extend(
                    self._stage_candidate_locked(
                        binding,
                        current_value=current_value,
                        timestamp_ns=now_ns,
                    )
                )
        return events

    def _collect_level_change_events(
        self,
        *,
        timestamp_ns: int,
        current_values: dict[int, int],
    ) -> list[ButtonEvent]:
        events: list[ButtonEvent] = []
        with self._sampler_state_lock:
            for binding in self.bindings:
                current_value = current_values.get(binding.line_offset)
                if current_value is None:
                    continue
                events.extend(
                    self._stage_candidate_locked(
                        binding,
                        current_value=current_value,
                        timestamp_ns=timestamp_ns,
                    )
                )
        return events

    def _start_background_sampler(self) -> None:
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
            while not stop_event.wait(self._background_poll_interval_s):
                timestamp_ns = time.monotonic_ns()
                current_values = self._read_current_values()
                for event in self._collect_level_change_events(
                    timestamp_ns=timestamp_ns,
                    current_values=current_values,
                ):
                    self._enqueue_pending_event(event)
        except Exception as exc:
            self._background_poll_error = exc

    def _start_gpiomon_process(self, gpiomon_path: str) -> None:
        if self._background_poll_thread is not None:
            return

        process = self._spawn_gpiomon_process(gpiomon_path)
        self._gpiomon_process = process

        cli_spec = _detect_gpiomon_cli_spec(gpiomon_path)
        self._gpiomon_uses_kernel_debounce = bool(cli_spec.supports_debounce_option and self.debounce_ms > 0)

        stop_event = ThreadEvent()
        self._background_poll_stop = stop_event
        self._background_poll_error = None
        self._background_poll_thread = Thread(
            target=self._background_gpiomon_worker,
            name=f"{self.consumer}-gpiomon",
            daemon=True,
        )
        self._background_poll_thread.start()

    def _spawn_gpiomon_process(self, gpiomon_path: str) -> subprocess.Popen[str]:
        command = _build_gpiomon_command(
            gpiomon_path,
            chip_name=self.chip_name,
            bindings=self.bindings,
            bias=self.bias,
            consumer=self.consumer,
            debounce_ms=self.debounce_ms,
        )
        process = self._launch_gpiomon_process(command)

        if process.poll() is not None:
            stderr_text = self._read_process_stderr(process)
            if _cli_rejected_named_options(command, stderr_text) or _cli_rejected_disabled_bias(command, stderr_text):
                _store_gpiomon_cli_spec(gpiomon_path, _legacy_gpiomon_cli_spec())
                command = _build_gpiomon_command(
                    gpiomon_path,
                    chip_name=self.chip_name,
                    bindings=self.bindings,
                    bias=self.bias,
                    consumer=self.consumer,
                    debounce_ms=self.debounce_ms,
                )
                process = self._launch_gpiomon_process(command)
                if process.poll() is not None:
                    stderr_text = self._read_process_stderr(process)
                    raise RuntimeError(self._format_gpiomon_startup_error(process, stderr_text))
            else:
                raise RuntimeError(self._format_gpiomon_startup_error(process, stderr_text))
        return process

    def _launch_gpiomon_process(self, command: list[str]) -> subprocess.Popen[str]:
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise RuntimeError("Unable to execute gpiomon for GPIO button monitoring") from exc

        time.sleep(_GPIOMON_STARTUP_TIMEOUT_S)
        return process

    def _format_gpiomon_startup_error(self, process: subprocess.Popen[str], stderr_text: str) -> str:
        detail = f": {stderr_text}" if stderr_text else ""
        return f"gpiomon failed to start (exit code {process.returncode}){detail}"

    def _read_process_stderr(self, process: subprocess.Popen[str]) -> str:
        if process.stderr is None:
            return ""
        try:
            return process.stderr.read().strip()
        except Exception:
            return ""

    def _terminate_process(self, process: subprocess.Popen[str]) -> None:
        try:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=_CLI_READ_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            self._kill_process(process)
        except Exception:
            _LOGGER.warning("Failed to terminate GPIO helper process cleanly.", exc_info=True)

    def _kill_process(self, process: subprocess.Popen[str]) -> None:
        try:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=_CLI_READ_TIMEOUT_S)
        except Exception:
            _LOGGER.warning("Failed to kill GPIO helper process.", exc_info=True)

    def _next_debounce_flush_timeout(self) -> float:
        if self.debounce_ns <= 0 or self._gpiomon_uses_kernel_debounce:
            return _BACKGROUND_THREAD_WAKEUP_S

        with self._sampler_state_lock:
            if not self._debounce_candidates:
                return _BACKGROUND_THREAD_WAKEUP_S
            now_ns = time.monotonic_ns()
            remaining_ns = min(
                max(0, candidate_since_ns + self.debounce_ns - now_ns)
                for _, candidate_since_ns in self._debounce_candidates.values()
            )
        return min(_BACKGROUND_THREAD_WAKEUP_S, remaining_ns / 1_000_000_000)

    def _background_gpiomon_worker(self) -> None:
        stop_event = self._background_poll_stop
        process = self._gpiomon_process
        if stop_event is None or process is None or process.stdout is None:
            return

        stdout_fd = process.stdout.fileno()

        try:
            while not stop_event.is_set():
                timeout_s = self._next_debounce_flush_timeout()
                ready, _, _ = select.select([stdout_fd], [], [], timeout_s)

                now_ns = time.monotonic_ns()
                for event in self._flush_due_debounce_candidates(now_ns=now_ns):
                    self._enqueue_pending_event(event)

                if stop_event.is_set():
                    return

                if not ready:
                    continue

                raw_line = process.stdout.readline()
                if raw_line == "":
                    if stop_event.is_set() or process.poll() == 0:
                        return
                    stderr_text = self._read_process_stderr(process)
                    raise RuntimeError(
                        f"gpiomon exited unexpectedly with code {process.poll()}: {stderr_text or 'no stderr output'}"
                    )

                parsed_events = self._parse_gpiomon_output_line(raw_line.strip(), timestamp_ns=time.monotonic_ns())
                for event in parsed_events:
                    self._enqueue_pending_event(event)
        except Exception as exc:
            self._background_poll_error = exc

    def _parse_gpiomon_output_line(self, line: str, *, timestamp_ns: int) -> list[ButtonEvent]:
        if not line:
            return []

        tokens = line.split()
        if len(tokens) != 2:
            _LOGGER.debug("Ignoring unexpected gpiomon output line: %r", line)
            return []

        try:
            line_offset = int(tokens[0])
        except ValueError:
            _LOGGER.debug("Ignoring gpiomon line with non-numeric offset: %r", line)
            return []

        binding = self._binding_by_offset.get(line_offset)
        if binding is None:
            return []

        normalized_edge = _normalize_edge_type(tokens[1])
        if normalized_edge not in {0, 1}:
            _LOGGER.debug("Ignoring gpiomon line with unsupported edge value: %r", line)
            return []

        current_value = 1 if normalized_edge == 1 else 0
        if self._gpiomon_uses_kernel_debounce:
            with self._sampler_state_lock:
                self._debounce_candidates.pop(line_offset, None)
                self._last_values[line_offset] = current_value
            return [
                ButtonEvent(
                    name=binding.name,
                    line_offset=line_offset,
                    action=edge_to_action(normalized_edge, self.active_low),
                    raw_edge=edge_name(normalized_edge),
                    timestamp_ns=timestamp_ns,
                )
            ]

        with self._sampler_state_lock:
            return self._stage_candidate_locked(
                binding,
                current_value=current_value,
                timestamp_ns=timestamp_ns,
            )

    def poll(self, timeout: float | None = None) -> ButtonEvent | None:
        """Return the next button event or ``None`` after the timeout."""

        if not self._is_open():
            self.open()

        if not self._last_values and self._request is None and not self._line_by_offset and self._gpiomon_process is None:
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
            if normalized_timeout == 0.0:
                if self._background_poll_error is not None:
                    raise RuntimeError("GPIO button background worker failed") from self._background_poll_error
                return self._dequeue_pending_event()

            deadline = None if normalized_timeout is None else time.monotonic() + normalized_timeout
            while True:
                if self._background_poll_error is not None:
                    raise RuntimeError("GPIO button background worker failed") from self._background_poll_error

                queued_event = self._dequeue_pending_event()
                if queued_event is not None:
                    return queued_event

                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    wait_timeout = min(_BACKGROUND_THREAD_WAKEUP_S, remaining)
                else:
                    wait_timeout = _BACKGROUND_THREAD_WAKEUP_S

                try:
                    return self._pending_events.get(timeout=wait_timeout)
                except Empty:
                    continue

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
            sleep_for = (
                _LEVEL_SAMPLE_INTERVAL_S
                if deadline is None
                else min(_LEVEL_SAMPLE_INTERVAL_S, max(0.0, deadline - time.monotonic()))
            )
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
        if self._gpiomon_process is not None:
            with self._sampler_state_lock:
                return dict(self._last_values)
        return self._read_cli_values()

    def _read_cli_values(self) -> dict[int, int]:
        gpioget_path = self._resolve_cli_tool("gpioget")
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
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if isinstance(exc.stderr, str) else ""
            if _cli_rejected_named_options(command, stderr) or _cli_rejected_disabled_bias(command, stderr):
                _store_gpioget_cli_spec(gpioget_path, _legacy_gpioget_cli_spec())
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
        return dict(self._read_current_values())


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
