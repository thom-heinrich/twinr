"""Bounded servo-follow adapter for Twinr body orientation on the Pi.

This module owns the low-level mapping from one already-derived normalized
attention target to servo pulse widths. It does not decide who Twinr should
look at; higher runtime layers provide a conservative `target_center_x` and
confidence value. The adapter keeps movement bounded, applies configurable
soft limits inside the calibrated pulse range, holds the last stable target
briefly across short sensor dropouts, can optionally stay physically still
while a person remains visible and only follow that person's exit trajectory,
and can release output both near neutral and after a stable off-center move
so the servo does not keep buzzing against a loaded mount.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math
from pathlib import Path
import subprocess
import time
from typing import Callable, Protocol, cast

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.servo_continuous import (
    ContinuousRotationServoConfig,
    ContinuousRotationServoPlanner,
)
from twinr.hardware.servo_maestro import PololuMaestroServoPulseWriter
from twinr.hardware.servo_peer import PeerPololuMaestroServoPulseWriter
from twinr.hardware.servo_state import AttentionServoRuntimeState, AttentionServoStateStore


_DEFAULT_SERVO_FREQUENCY_HZ = 50
_DEFAULT_SERVO_DRIVER = "auto"
_DEFAULT_TARGET_HOLD_S = 1.1
_DEFAULT_LOSS_EXTRAPOLATION_S = 0.8
_DEFAULT_LOSS_EXTRAPOLATION_GAIN = 0.65
_DEFAULT_MIN_CONFIDENCE = 0.58
_DEFAULT_DEADBAND = 0.045
_DEFAULT_MIN_PULSE_WIDTH_US = 1050
_DEFAULT_CENTER_PULSE_WIDTH_US = 1500
_DEFAULT_MAX_PULSE_WIDTH_US = 1950
_DEFAULT_MAX_STEP_US = 45
_DEFAULT_TARGET_SMOOTHING_S = 0.9
_DEFAULT_MAX_VELOCITY_US_PER_S = 80.0
_DEFAULT_MAX_ACCELERATION_US_PER_S2 = 220.0
_DEFAULT_MAX_JERK_US_PER_S3 = 900.0
_DEFAULT_REST_MAX_VELOCITY_US_PER_S = 35.0
_DEFAULT_REST_MAX_ACCELERATION_US_PER_S2 = 120.0
_DEFAULT_REST_MAX_JERK_US_PER_S3 = 450.0
_DEFAULT_MIN_COMMAND_DELTA_US = 8
_DEFAULT_VISIBLE_RETARGET_TOLERANCE_US = 40
_DEFAULT_REFERENCE_INTERVAL_S = 0.2
_DEFAULT_SOFT_LIMIT_MARGIN_US = 70
_DEFAULT_IDLE_RELEASE_S = 1.0
_DEFAULT_SETTLED_RELEASE_S = 0.0
_DEFAULT_FOLLOW_EXIT_ONLY = False
_DEFAULT_VISIBLE_RECENTER_INTERVAL_S = 30.0
_DEFAULT_VISIBLE_RECENTER_CENTER_TOLERANCE = 0.12
_DEFAULT_MECHANICAL_RANGE_DEGREES = 270.0
_DEFAULT_EXIT_FOLLOW_MAX_DEGREES = 60.0
_DEFAULT_EXIT_ACTIVATION_DELAY_S = 0.75
_DEFAULT_EXIT_SETTLE_HOLD_S = 0.6
_DEFAULT_EXIT_REACQUIRE_CENTER_TOLERANCE = 0.08
_DEFAULT_EXIT_VISIBLE_EDGE_THRESHOLD = 0.62
_DEFAULT_EXIT_VISIBLE_BOX_EDGE_THRESHOLD = 0.92
_DEFAULT_EXIT_COOLDOWN_S = 30.0
_DEFAULT_CONTROL_MODE = "position"
_DEFAULT_STATE_PATH = "state/attention_servo_state.json"
_DEFAULT_CONTINUOUS_MAX_SPEED_DEGREES_PER_S = 120.0
_DEFAULT_CONTINUOUS_SLOW_ZONE_DEGREES = 45.0
_DEFAULT_CONTINUOUS_STOP_TOLERANCE_DEGREES = 4.0
_DEFAULT_CONTINUOUS_MIN_SPEED_PULSE_DELTA_US = 70
_DEFAULT_CONTINUOUS_MAX_SPEED_PULSE_DELTA_US = 160
_MIN_RELEASE_TOLERANCE_US = 12
_MAX_RELEASE_TOLERANCE_US = 48
_SERVO_PRECHECK_TIMEOUT_S = 2.0
_RECOVERABLE_SERVO_FAULT_RETRY_S = 1.0
_CONFLICTING_GPIO_OVERLAY_NAMES = frozenset({"pwm-pio"})


def _clamp_ratio(value: float) -> float:
    if value != value:
        return 0.5
    return max(0.0, min(1.0, value))


def _bounded_float(value: object, *, default: float, minimum: float, maximum: float) -> float:
    raw_value = default if value is None else value
    if not isinstance(raw_value, (int, float, str)):
        normalized = default
    else:
        try:
            normalized = float(raw_value)
        except (TypeError, ValueError):
            normalized = default
    if not isinstance(normalized, float) or not math.isfinite(normalized):
        normalized = default
    return max(minimum, min(maximum, normalized))


def _bounded_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
    raw_value = default if value is None else value
    if not isinstance(raw_value, (int, float, str)):
        normalized = default
    else:
        try:
            normalized = int(raw_value)
        except (TypeError, ValueError):
            normalized = default
    return max(minimum, min(maximum, normalized))


def _clamp(value: float, *, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _run_best_effort_text_command(args: list[str], *, timeout_s: float = _SERVO_PRECHECK_TIMEOUT_S) -> str | None:
    """Return stdout for one short best-effort probe command."""

    try:
        result = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=float(timeout_s),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _detect_conflicting_servo_gpio_environment(
    *,
    gpio: int,
    run_text_command: Callable[[list[str]], str | None] = _run_best_effort_text_command,
) -> list[str]:
    """Describe live GPIO conflict evidence that makes servo startup unsafe."""

    checked_gpio = int(gpio)
    gpio_token = f"gpio={checked_gpio}"
    pin_token = f"pin={checked_gpio}"
    conflicts: list[str] = []
    overlay_output = run_text_command(["dtoverlay", "-l"])
    if overlay_output:
        for raw_line in overlay_output.splitlines():
            line = raw_line.strip()
            if ":" not in line:
                continue
            descriptor = line.split(":", 1)[1].strip()
            if not descriptor or (gpio_token not in descriptor and pin_token not in descriptor):
                continue
            overlay_name = descriptor.split()[0].strip().lower()
            if overlay_name in _CONFLICTING_GPIO_OVERLAY_NAMES:
                conflicts.append(f"overlay {descriptor}")
    process_output = run_text_command(["ps", "-eo", "pid=,stat=,args="])
    if process_output:
        for raw_line in process_output.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(None, 2)
            if len(parts) < 3:
                continue
            pid, stat, args = parts
            lowered_args = args.lower()
            if "dtoverlay" not in lowered_args:
                continue
            if "pwm-pio" in lowered_args:
                if gpio_token in lowered_args or pin_token in lowered_args:
                    conflicts.append(f"process pid={pid} stat={stat} args={args}")
                continue
            if "dtoverlay -r" in lowered_args and conflicts:
                conflicts.append(f"process pid={pid} stat={stat} args={args}")
    return conflicts


def _assert_servo_gpio_environment_ready(config: "AttentionServoConfig") -> None:
    """Fail closed when a proven conflicting GPIO overlay/process state is present."""

    driver = str(config.driver or _DEFAULT_SERVO_DRIVER).strip().lower() or _DEFAULT_SERVO_DRIVER
    if driver in {"pololu_maestro", "peer_pololu_maestro"}:
        return
    if not config.enabled or config.gpio is None:
        return
    conflicts = _detect_conflicting_servo_gpio_environment(gpio=int(config.gpio))
    if not conflicts:
        return
    rendered_conflicts = "; ".join(conflicts[:4])
    if len(conflicts) > 4:
        rendered_conflicts += f"; +{len(conflicts) - 4} more"
    raise RuntimeError(
        f"Refusing attention servo startup on GPIO{int(config.gpio)} due to conflicting overlay/process state: "
        f"{rendered_conflicts}"
    )


def _normalize_chip_index(chip_name: str) -> int:
    normalized = str(chip_name or "gpiochip0").strip()
    if not normalized:
        raise ValueError("GPIO chip name must not be empty")
    if normalized.startswith("/dev/"):
        normalized = normalized.rsplit("/", 1)[-1]
    if normalized.startswith("gpiochip"):
        suffix = normalized[len("gpiochip") :]
        if suffix and suffix.isdigit():
            return int(suffix)
        raise ValueError(f"Unsupported GPIO chip name for servo output: {chip_name}")
    if normalized.isdigit():
        return int(normalized)
    raise ValueError(f"Unsupported GPIO chip name for servo output: {chip_name}")


class ServoPulseWriter(Protocol):
    """Write servo pulse widths to one GPIO line."""

    def probe(self, gpio: int) -> None:
        """Validate one output line/channel before runtime startup when supported."""

    def write(self, *, gpio_chip: str, gpio: int, pulse_width_us: int) -> None:
        """Start or update the pulse train for one GPIO line."""

    def disable(self, *, gpio_chip: str, gpio: int) -> None:
        """Stop driving one GPIO line with servo pulses."""

    def close(self) -> None:
        """Release any underlying resources."""

    def current_pulse_width_us(self, *, gpio_chip: str, gpio: int) -> int | None:
        """Return the most recent known pulse width for one GPIO line when available."""


class _NoopServoPulseWriter:
    """Allow fail-closed servo startup without touching GPIO state."""

    def probe(self, gpio: int) -> None:
        del gpio

    def write(self, *, gpio_chip: str, gpio: int, pulse_width_us: int) -> None:
        del gpio_chip, gpio, pulse_width_us

    def disable(self, *, gpio_chip: str, gpio: int) -> None:
        del gpio_chip, gpio

    def close(self) -> None:
        return

    def current_pulse_width_us(self, *, gpio_chip: str, gpio: int) -> int | None:
        del gpio_chip, gpio
        return None


class LGPIOServoPulseWriter:
    """Drive one Pi servo line through `lgpio.tx_servo`."""

    def __init__(self) -> None:
        self._lgpio: object | None = None
        self._handles: dict[int, int] = {}
        self._claimed_outputs: dict[int, set[int]] = {}

    def _module(self):
        if self._lgpio is None:
            try:
                import lgpio  # pylint: disable=import-error
            except Exception as exc:  # pragma: no cover - import path depends on host image
                raise RuntimeError("python3-lgpio is required for attention servo output") from exc
            self._lgpio = lgpio
        return self._lgpio

    def _handle_for_chip(self, gpio_chip: str) -> tuple[object, int]:
        module = self._module()
        chip_index = _normalize_chip_index(gpio_chip)
        handle = self._handles.get(chip_index)
        if handle is None:
            handle = module.gpiochip_open(chip_index)
            self._handles[chip_index] = handle
        return module, handle

    def _claim_output_if_needed(self, *, module: object, handle: int, gpio_chip: str, gpio: int) -> None:
        chip_index = _normalize_chip_index(gpio_chip)
        claimed_outputs = self._claimed_outputs.setdefault(chip_index, set())
        if gpio in claimed_outputs:
            return
        claim_output = getattr(module, "gpio_claim_output", None)
        if callable(claim_output):
            claim_output(handle, int(gpio), 0)
        claimed_outputs.add(int(gpio))

    def probe(self, gpio: int) -> None:
        del gpio
        self._module()

    def write(self, *, gpio_chip: str, gpio: int, pulse_width_us: int) -> None:
        module, handle = self._handle_for_chip(gpio_chip)
        self._claim_output_if_needed(
            module=module,
            handle=handle,
            gpio_chip=gpio_chip,
            gpio=int(gpio),
        )
        tx_servo = getattr(module, "tx_servo", None)
        if not callable(tx_servo):
            raise RuntimeError("python3-lgpio with tx_servo support is required for attention servo output")
        tx_servo(  # pylint: disable=not-callable
            handle,
            int(gpio),
            int(pulse_width_us),
            servo_frequency=_DEFAULT_SERVO_FREQUENCY_HZ,
        )

    def disable(self, *, gpio_chip: str, gpio: int) -> None:
        module, handle = self._handle_for_chip(gpio_chip)
        chip_index = _normalize_chip_index(gpio_chip)
        tx_servo = getattr(module, "tx_servo", None)
        if not callable(tx_servo):
            raise RuntimeError("python3-lgpio with tx_servo support is required for attention servo output")
        tx_servo(  # pylint: disable=not-callable
            handle,
            int(gpio),
            0,
            servo_frequency=_DEFAULT_SERVO_FREQUENCY_HZ,
        )
        claim_input = cast(Callable[[int, int], None] | None, getattr(module, "gpio_claim_input", None))
        if claim_input is not None:
            claim_input(handle, int(gpio))
        claimed_outputs = self._claimed_outputs.get(chip_index)
        if claimed_outputs is not None:
            claimed_outputs.discard(int(gpio))
            if not claimed_outputs:
                self._claimed_outputs.pop(chip_index, None)

    def close(self) -> None:
        module = self._module() if self._lgpio is not None else None
        if module is None:
            return
        for handle in self._handles.values():
            try:
                module.gpiochip_close(handle)
            except Exception:
                continue
        self._handles.clear()
        self._claimed_outputs.clear()

    def current_pulse_width_us(self, *, gpio_chip: str, gpio: int) -> int | None:
        del gpio_chip, gpio
        return None


class LGPIOPWMServoPulseWriter(LGPIOServoPulseWriter):
    """Drive one Pi servo line through `lgpio.tx_pwm` at servo frequency.

    `lgpio.tx_servo` documents its own timing jitter and resulting servo fidget.
    This writer keeps the same GPIO claiming/release semantics but programs
    a 50 Hz PWM duty cycle directly from the requested pulse width so Pi hosts
    without working hardware PWM can still prefer the calmer software-PWM path.
    """

    def _tx_pwm(self, module: object, *, handle: int, gpio: int, pwm_frequency: float, duty_cycle: float) -> None:
        tx_pwm = getattr(module, "tx_pwm", None)
        if not callable(tx_pwm):
            raise RuntimeError("python3-lgpio with tx_pwm support is required for attention servo PWM output")
        status = tx_pwm(handle, int(gpio), float(pwm_frequency), float(duty_cycle))
        if isinstance(status, int) and status < 0:
            raise RuntimeError(f"lgpio tx_pwm failed with status {status}")

    def _duty_cycle_percent_for_pulse_width_us(self, pulse_width_us: int) -> float:
        checked_pulse_width_us = max(0, min(20_000, int(pulse_width_us)))
        return round((checked_pulse_width_us / 20_000.0) * 100.0, 6)

    def write(self, *, gpio_chip: str, gpio: int, pulse_width_us: int) -> None:
        module, handle = self._handle_for_chip(gpio_chip)
        checked_gpio = int(gpio)
        self._claim_output_if_needed(
            module=module,
            handle=handle,
            gpio_chip=gpio_chip,
            gpio=checked_gpio,
        )
        self._tx_pwm(
            module,
            handle=handle,
            gpio=checked_gpio,
            pwm_frequency=_DEFAULT_SERVO_FREQUENCY_HZ,
            duty_cycle=self._duty_cycle_percent_for_pulse_width_us(int(pulse_width_us)),
        )

    def disable(self, *, gpio_chip: str, gpio: int) -> None:
        module, handle = self._handle_for_chip(gpio_chip)
        chip_index = _normalize_chip_index(gpio_chip)
        checked_gpio = int(gpio)
        self._tx_pwm(
            module,
            handle=handle,
            gpio=checked_gpio,
            pwm_frequency=0.0,
            duty_cycle=0.0,
        )
        claim_input = cast(Callable[[int, int], None] | None, getattr(module, "gpio_claim_input", None))
        if claim_input is not None:
            claim_input(handle, checked_gpio)
        claimed_outputs = self._claimed_outputs.get(chip_index)
        if claimed_outputs is not None:
            claimed_outputs.discard(checked_gpio)
            if not claimed_outputs:
                self._claimed_outputs.pop(chip_index, None)


class PigpioServoPulseWriter:
    """Drive one Pi servo line through pigpio's hardware-timed servo pulses."""

    def __init__(self) -> None:
        self._pigpio: object | None = None
        self._pi = None
        self._claimed_outputs: set[int] = set()

    def _module(self):
        if self._pigpio is None:
            try:
                import pigpio  # pylint: disable=import-error
            except Exception as exc:  # pragma: no cover - depends on host image
                raise RuntimeError("python3-pigpio is required for hardware-timed attention servo output") from exc
            self._pigpio = pigpio
        return self._pigpio

    def _assert_supported_chip(self, gpio_chip: str) -> None:
        if _normalize_chip_index(gpio_chip) != 0:
            raise RuntimeError(f"pigpio only supports gpiochip0-style Broadcom GPIO numbering, got {gpio_chip!r}")

    def _connection(self):
        module = self._module()
        if self._pi is None:
            pi = module.pi()
            if not bool(getattr(pi, "connected", False)):
                try:
                    pi.stop()
                except Exception:
                    pass
                raise RuntimeError("pigpiod is required for hardware-timed attention servo output")
            self._pi = pi
        return module, self._pi

    def probe(self, gpio: int) -> None:
        del gpio
        self._connection()

    def write(self, *, gpio_chip: str, gpio: int, pulse_width_us: int) -> None:
        self._assert_supported_chip(gpio_chip)
        module, pi = self._connection()
        checked_gpio = int(gpio)
        if checked_gpio not in self._claimed_outputs:
            pi.set_mode(checked_gpio, module.OUTPUT)
            self._claimed_outputs.add(checked_gpio)
        status = pi.set_servo_pulsewidth(checked_gpio, int(pulse_width_us))
        if isinstance(status, int) and status < 0:
            raise RuntimeError(f"pigpio set_servo_pulsewidth failed with status {status}")

    def disable(self, *, gpio_chip: str, gpio: int) -> None:
        self._assert_supported_chip(gpio_chip)
        if self._pi is None:
            return
        module, pi = self._connection()
        checked_gpio = int(gpio)
        status = pi.set_servo_pulsewidth(checked_gpio, 0)
        if isinstance(status, int) and status < 0:
            raise RuntimeError(f"pigpio set_servo_pulsewidth failed with status {status}")
        pi.set_mode(checked_gpio, module.INPUT)
        self._claimed_outputs.discard(checked_gpio)

    def close(self) -> None:
        if self._pi is not None:
            try:
                self._pi.stop()
            finally:
                self._pi = None
        self._claimed_outputs.clear()

    def current_pulse_width_us(self, *, gpio_chip: str, gpio: int) -> int | None:
        del gpio_chip, gpio
        return None


@dataclass(frozen=True, slots=True)
class _SysfsPWMDescriptor:
    pwm_chip_index: int
    pwm_channel_index: int
    pinctrl_alt_mode: str


class SysfsPWMServoPulseWriter:
    """Drive one Pi servo line through the kernel PWM sysfs interface."""

    _PERIOD_NS = 20_000_000
    _EXPORT_WAIT_S = 0.25
    _EXPORT_POLL_INTERVAL_S = 0.01

    def __init__(
        self,
        *,
        sysfs_root: Path | str = "/sys/class/pwm",
        pinctrl_bin: str = "pinctrl",
    ) -> None:
        self._sysfs_root = Path(sysfs_root)
        self._pinctrl_bin = str(pinctrl_bin or "pinctrl")
        self._descriptors: dict[int, _SysfsPWMDescriptor] = {}
        self._enabled_gpios: set[int] = set()

    def _run_pinctrl(self, *args: str) -> str:
        result = subprocess.run(
            [self._pinctrl_bin, *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def _descriptor_for_gpio(self, gpio: int) -> _SysfsPWMDescriptor:
        checked_gpio = int(gpio)
        cached = self._descriptors.get(checked_gpio)
        if cached is not None:
            return cached
        try:
            funcs_output = self._run_pinctrl("funcs", str(checked_gpio))
        except Exception as exc:
            raise RuntimeError("pinctrl is required for sysfs PWM attention servo output") from exc
        tokens = [token.strip() for token in funcs_output.split(",")]
        pwm_token: str | None = None
        alt_index: int | None = None
        for index, token in enumerate(tokens[2:], start=0):
            normalized = token.strip().upper()
            if normalized.startswith("PWM") and "_CHAN" in normalized:
                pwm_token = normalized
                alt_index = index
                break
        if pwm_token is None or alt_index is None:
            raise RuntimeError(f"GPIO{checked_gpio} does not expose a hardware PWM function via pinctrl")
        controller_text, _, channel_text = pwm_token.partition("_CHAN")
        if not controller_text.startswith("PWM") or not channel_text.isdigit():
            raise RuntimeError(f"Unsupported PWM function token for GPIO{checked_gpio}: {pwm_token}")
        controller_index_text = controller_text[len("PWM") :]
        if not controller_index_text.isdigit():
            raise RuntimeError(f"Unsupported PWM controller token for GPIO{checked_gpio}: {pwm_token}")
        descriptor = _SysfsPWMDescriptor(
            pwm_chip_index=int(controller_index_text),
            pwm_channel_index=int(channel_text),
            pinctrl_alt_mode=f"a{alt_index}",
        )
        self._descriptors[checked_gpio] = descriptor
        return descriptor

    def probe(self, gpio: int) -> None:
        descriptor = self._descriptor_for_gpio(int(gpio))
        pwm_chip_dir = self._sysfs_root / f"pwmchip{descriptor.pwm_chip_index}"
        if not pwm_chip_dir.exists():
            raise RuntimeError(f"PWM chip path not available for attention servo output: {pwm_chip_dir}")

    def _pwm_dir_for_descriptor(self, descriptor: _SysfsPWMDescriptor) -> Path:
        return self._sysfs_root / f"pwmchip{descriptor.pwm_chip_index}" / f"pwm{descriptor.pwm_channel_index}"

    def _ensure_exported(self, descriptor: _SysfsPWMDescriptor) -> Path:
        pwm_dir = self._pwm_dir_for_descriptor(descriptor)
        if pwm_dir.exists():
            return pwm_dir
        pwm_chip_dir = self._sysfs_root / f"pwmchip{descriptor.pwm_chip_index}"
        export_path = pwm_chip_dir / "export"
        if not export_path.exists():
            raise RuntimeError(f"PWM chip path not available for attention servo output: {pwm_chip_dir}")
        export_path.write_text(str(descriptor.pwm_channel_index), encoding="utf-8")
        deadline = time.monotonic() + self._EXPORT_WAIT_S
        while time.monotonic() < deadline:
            if pwm_dir.exists():
                return pwm_dir
            time.sleep(self._EXPORT_POLL_INTERVAL_S)
        raise RuntimeError(f"PWM channel did not appear after export: {pwm_dir}")

    def _configure_gpio(self, gpio: int, descriptor: _SysfsPWMDescriptor) -> Path:
        pwm_dir = self._ensure_exported(descriptor)
        enable_path = pwm_dir / "enable"
        if enable_path.exists():
            enable_path.write_text("0", encoding="utf-8")
        (pwm_dir / "period").write_text(str(self._PERIOD_NS), encoding="utf-8")
        polarity_path = pwm_dir / "polarity"
        if polarity_path.exists():
            polarity_path.write_text("normal", encoding="utf-8")
        self._run_pinctrl("set", str(int(gpio)), descriptor.pinctrl_alt_mode, "pn")
        return pwm_dir

    def _duty_cycle_ns_for_pulse_width_us(self, pulse_width_us: int) -> int:
        checked_pulse_width_us = max(0, min(self._PERIOD_NS // 1000, int(pulse_width_us)))
        return checked_pulse_width_us * 1000

    def write(self, *, gpio_chip: str, gpio: int, pulse_width_us: int) -> None:
        if _normalize_chip_index(gpio_chip) != 0:
            raise RuntimeError(f"sysfs PWM servo output only supports gpiochip0, got {gpio_chip!r}")
        checked_gpio = int(gpio)
        descriptor = self._descriptor_for_gpio(checked_gpio)
        pwm_dir = self._configure_gpio(checked_gpio, descriptor)
        (pwm_dir / "duty_cycle").write_text(
            str(self._duty_cycle_ns_for_pulse_width_us(pulse_width_us)),
            encoding="utf-8",
        )
        (pwm_dir / "enable").write_text("1", encoding="utf-8")
        self._enabled_gpios.add(checked_gpio)

    def disable(self, *, gpio_chip: str, gpio: int) -> None:
        if _normalize_chip_index(gpio_chip) != 0:
            raise RuntimeError(f"sysfs PWM servo output only supports gpiochip0, got {gpio_chip!r}")
        checked_gpio = int(gpio)
        descriptor = self._descriptor_for_gpio(checked_gpio)
        pwm_dir = self._pwm_dir_for_descriptor(descriptor)
        if pwm_dir.exists():
            enable_path = pwm_dir / "enable"
            if enable_path.exists():
                enable_path.write_text("0", encoding="utf-8")
        self._run_pinctrl("set", str(checked_gpio), "ip", "pd")
        self._enabled_gpios.discard(checked_gpio)

    def close(self) -> None:
        for gpio in tuple(sorted(self._enabled_gpios)):
            try:
                self.disable(gpio_chip="gpiochip0", gpio=gpio)
            except Exception:
                continue
        self._enabled_gpios.clear()

    def current_pulse_width_us(self, *, gpio_chip: str, gpio: int) -> int | None:
        del gpio_chip, gpio
        return None


class TwinrKernelServoPulseWriter:
    """Drive one Pi servo line through Twinr's custom kernel servo module."""

    _DEFAULT_SYSFS_ROOT = Path("/sys/class/twinr_servo/servo0")
    _DEFAULT_PERIOD_US = 20_000

    def __init__(
        self,
        *,
        sysfs_root: Path | str = _DEFAULT_SYSFS_ROOT,
        period_us: int = _DEFAULT_PERIOD_US,
    ) -> None:
        self._sysfs_root = Path(sysfs_root)
        self._period_us = int(period_us)

    def _path(self, name: str) -> Path:
        return self._sysfs_root / name

    def probe(self, gpio: int) -> None:
        checked_gpio = int(gpio)
        if checked_gpio < 0:
            raise RuntimeError("Twinr kernel servo output requires a non-negative GPIO line")
        required = (
            self._path("gpio"),
            self._path("period_us"),
            self._path("pulse_width_us"),
            self._path("enabled"),
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(
                "Twinr kernel servo module is required for attention servo output: "
                + ", ".join(missing)
            )

    def _read_int(self, path: Path, *, default: int) -> int:
        try:
            return int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return default

    def _write_text(self, path: Path, value: str) -> None:
        path.write_text(f"{value}\n", encoding="utf-8")

    def current_pulse_width_us(self, *, gpio_chip: str, gpio: int) -> int | None:
        del gpio_chip
        checked_gpio = int(gpio)
        try:
            self.probe(checked_gpio)
        except RuntimeError:
            return None
        current_gpio = self._read_int(self._path("gpio"), default=-1)
        if current_gpio not in {-1, checked_gpio}:
            return None
        pulse_width_us = self._read_int(self._path("pulse_width_us"), default=-1)
        if pulse_width_us < 500 or pulse_width_us > 2500:
            return None
        return pulse_width_us

    def _ensure_gpio_claimed(self, gpio: int) -> None:
        self.probe(gpio)
        enabled_path = self._path("enabled")
        gpio_path = self._path("gpio")
        current_gpio = self._read_int(gpio_path, default=-1)
        if current_gpio == int(gpio):
            return
        if self._read_int(enabled_path, default=0) != 0:
            self._write_text(enabled_path, "0")
        self._write_text(gpio_path, str(int(gpio)))

    def _ensure_period(self) -> None:
        enabled_path = self._path("enabled")
        period_path = self._path("period_us")
        current_period = self._read_int(period_path, default=self._period_us)
        if current_period == self._period_us:
            return
        if self._read_int(enabled_path, default=0) != 0:
            self._write_text(enabled_path, "0")
        self._write_text(period_path, str(self._period_us))

    def write(self, *, gpio_chip: str, gpio: int, pulse_width_us: int) -> None:
        del gpio_chip
        checked_gpio = int(gpio)
        self._ensure_gpio_claimed(checked_gpio)
        self._ensure_period()
        self._write_text(self._path("pulse_width_us"), str(int(pulse_width_us)))
        if self._read_int(self._path("enabled"), default=0) != 1:
            self._write_text(self._path("enabled"), "1")

    def disable(self, *, gpio_chip: str, gpio: int) -> None:
        del gpio_chip
        checked_gpio = int(gpio)
        try:
            self.probe(checked_gpio)
        except RuntimeError:
            return
        enabled_path = self._path("enabled")
        gpio_path = self._path("gpio")
        if self._read_int(enabled_path, default=0) != 0:
            self._write_text(enabled_path, "0")
        if self._read_int(gpio_path, default=-1) >= 0:
            self._write_text(gpio_path, "-1")

    def close(self) -> None:
        gpio_path = self._path("gpio")
        current_gpio = self._read_int(gpio_path, default=-1)
        if current_gpio >= 0:
            self.disable(gpio_chip="gpiochip0", gpio=current_gpio)


def _default_pulse_writer_for_config(config: "AttentionServoConfig") -> ServoPulseWriter:
    _assert_servo_gpio_environment_ready(config)
    driver = str(config.driver or _DEFAULT_SERVO_DRIVER).strip().lower() or _DEFAULT_SERVO_DRIVER
    if driver == "pololu_maestro":
        # Keep Maestro startup lazy so transient USB re-enumeration or a stale
        # by-id path does not permanently disable the servo until a full restart.
        return PololuMaestroServoPulseWriter(device_path=config.maestro_device)
    if driver == "peer_pololu_maestro":
        return PeerPololuMaestroServoPulseWriter(
            base_url=config.peer_base_url or "",
            timeout_s=config.peer_timeout_s,
        )
    if driver == "twinr_kernel":
        writer = TwinrKernelServoPulseWriter()
        if config.gpio is not None:
            writer.probe(config.gpio)
        return writer
    if driver == "sysfs_pwm":
        sysfs_writer: ServoPulseWriter = SysfsPWMServoPulseWriter()
        if config.gpio is not None:
            sysfs_writer.probe(config.gpio)
        return sysfs_writer
    if driver == "pigpio":
        return PigpioServoPulseWriter()
    if driver == "lgpio_pwm":
        return LGPIOPWMServoPulseWriter()
    if driver == "lgpio":
        return LGPIOServoPulseWriter()
    if driver != "auto":
        raise RuntimeError(f"Unsupported attention servo driver {driver!r}")
    try:
        writer = TwinrKernelServoPulseWriter()
        if config.gpio is not None:
            writer.probe(config.gpio)
        return writer
    except RuntimeError:
        pass
    try:
        auto_sysfs_writer: ServoPulseWriter = SysfsPWMServoPulseWriter()
        if config.gpio is not None:
            auto_sysfs_writer.probe(config.gpio)
        return auto_sysfs_writer
    except RuntimeError:
        pass
    try:
        pigpio_writer = PigpioServoPulseWriter()
        pigpio_writer._connection()
        return pigpio_writer
    except RuntimeError:
        pass
    return LGPIOPWMServoPulseWriter()


@dataclass(frozen=True, slots=True)
class AttentionServoConfig:
    """Store bounded attention-servo tuning from the global Twinr config."""

    enabled: bool = False
    driver: str = _DEFAULT_SERVO_DRIVER
    control_mode: str = _DEFAULT_CONTROL_MODE
    maestro_device: str | None = None
    peer_base_url: str | None = None
    peer_timeout_s: float = 1.5
    state_path: str = _DEFAULT_STATE_PATH
    gpio_chip: str = "gpiochip0"
    gpio: int | None = None
    invert_direction: bool = False
    target_hold_s: float = _DEFAULT_TARGET_HOLD_S
    loss_extrapolation_s: float = _DEFAULT_LOSS_EXTRAPOLATION_S
    loss_extrapolation_gain: float = _DEFAULT_LOSS_EXTRAPOLATION_GAIN
    min_confidence: float = _DEFAULT_MIN_CONFIDENCE
    deadband: float = _DEFAULT_DEADBAND
    min_pulse_width_us: int = _DEFAULT_MIN_PULSE_WIDTH_US
    center_pulse_width_us: int = _DEFAULT_CENTER_PULSE_WIDTH_US
    max_pulse_width_us: int = _DEFAULT_MAX_PULSE_WIDTH_US
    max_step_us: int = _DEFAULT_MAX_STEP_US
    target_smoothing_s: float = _DEFAULT_TARGET_SMOOTHING_S
    max_velocity_us_per_s: float = _DEFAULT_MAX_VELOCITY_US_PER_S
    max_acceleration_us_per_s2: float = _DEFAULT_MAX_ACCELERATION_US_PER_S2
    max_jerk_us_per_s3: float = _DEFAULT_MAX_JERK_US_PER_S3
    rest_max_velocity_us_per_s: float = _DEFAULT_REST_MAX_VELOCITY_US_PER_S
    rest_max_acceleration_us_per_s2: float = _DEFAULT_REST_MAX_ACCELERATION_US_PER_S2
    rest_max_jerk_us_per_s3: float = _DEFAULT_REST_MAX_JERK_US_PER_S3
    min_command_delta_us: int = _DEFAULT_MIN_COMMAND_DELTA_US
    visible_retarget_tolerance_us: int = _DEFAULT_VISIBLE_RETARGET_TOLERANCE_US
    reference_interval_s: float = _DEFAULT_REFERENCE_INTERVAL_S
    soft_limit_margin_us: int = _DEFAULT_SOFT_LIMIT_MARGIN_US
    idle_release_s: float = _DEFAULT_IDLE_RELEASE_S
    settled_release_s: float = _DEFAULT_SETTLED_RELEASE_S
    follow_exit_only: bool = _DEFAULT_FOLLOW_EXIT_ONLY
    visible_recenter_interval_s: float = _DEFAULT_VISIBLE_RECENTER_INTERVAL_S
    visible_recenter_center_tolerance: float = _DEFAULT_VISIBLE_RECENTER_CENTER_TOLERANCE
    mechanical_range_degrees: float = _DEFAULT_MECHANICAL_RANGE_DEGREES
    exit_follow_max_degrees: float = _DEFAULT_EXIT_FOLLOW_MAX_DEGREES
    exit_activation_delay_s: float = _DEFAULT_EXIT_ACTIVATION_DELAY_S
    exit_settle_hold_s: float = _DEFAULT_EXIT_SETTLE_HOLD_S
    exit_reacquire_center_tolerance: float = _DEFAULT_EXIT_REACQUIRE_CENTER_TOLERANCE
    exit_visible_edge_threshold: float = _DEFAULT_EXIT_VISIBLE_EDGE_THRESHOLD
    exit_visible_box_edge_threshold: float = _DEFAULT_EXIT_VISIBLE_BOX_EDGE_THRESHOLD
    exit_cooldown_s: float = _DEFAULT_EXIT_COOLDOWN_S
    continuous_max_speed_degrees_per_s: float = _DEFAULT_CONTINUOUS_MAX_SPEED_DEGREES_PER_S
    continuous_slow_zone_degrees: float = _DEFAULT_CONTINUOUS_SLOW_ZONE_DEGREES
    continuous_stop_tolerance_degrees: float = _DEFAULT_CONTINUOUS_STOP_TOLERANCE_DEGREES
    continuous_min_speed_pulse_delta_us: int = _DEFAULT_CONTINUOUS_MIN_SPEED_PULSE_DELTA_US
    continuous_max_speed_pulse_delta_us: int = _DEFAULT_CONTINUOUS_MAX_SPEED_PULSE_DELTA_US

    @property
    def uses_continuous_rotation(self) -> bool:
        return str(self.control_mode or _DEFAULT_CONTROL_MODE).strip().lower() == "continuous_rotation"

    @property
    def safe_min_pulse_width_us(self) -> int:
        """Return the calibrated lower pulse bound after the soft margin."""

        return min(
            self.center_pulse_width_us,
            self.min_pulse_width_us + self._bounded_left_margin_us(),
        )

    @property
    def safe_max_pulse_width_us(self) -> int:
        """Return the calibrated upper pulse bound after the soft margin."""

        return max(
            self.center_pulse_width_us,
            self.max_pulse_width_us - self._bounded_right_margin_us(),
        )

    def _bounded_left_margin_us(self) -> int:
        return min(self.soft_limit_margin_us, max(0, self.center_pulse_width_us - self.min_pulse_width_us))

    def _bounded_right_margin_us(self) -> int:
        return min(self.soft_limit_margin_us, max(0, self.max_pulse_width_us - self.center_pulse_width_us))

    @property
    def exit_follow_offset_limit(self) -> float:
        """Return the maximum normalized left/right offset for exit-follow."""

        half_range_degrees = max(1.0, self.mechanical_range_degrees * 0.5)
        return max(0.0, min(1.0, self.exit_follow_max_degrees / half_range_degrees))

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "AttentionServoConfig":
        """Build one bounded servo config from the global Twinr config."""

        driver = str(getattr(config, "attention_servo_driver", _DEFAULT_SERVO_DRIVER) or _DEFAULT_SERVO_DRIVER)
        control_mode = str(getattr(config, "attention_servo_control_mode", _DEFAULT_CONTROL_MODE) or _DEFAULT_CONTROL_MODE)
        maestro_device = getattr(config, "attention_servo_maestro_device", None)
        peer_base_url = getattr(config, "attention_servo_peer_base_url", None)
        peer_timeout_s = getattr(config, "attention_servo_peer_timeout_s", 1.5)
        raw_state_path = getattr(config, "attention_servo_state_path", _DEFAULT_STATE_PATH)
        resolved_state_path = Path(str(raw_state_path or _DEFAULT_STATE_PATH)).expanduser()
        if not resolved_state_path.is_absolute():
            project_root = Path(str(getattr(config, "project_root", ".") or ".")).expanduser().resolve(strict=False)
            resolved_state_path = (project_root / resolved_state_path).resolve(strict=False)
        configured_gpio = getattr(config, "attention_servo_gpio", None)
        if str(driver).strip().lower() in {"pololu_maestro", "peer_pololu_maestro"}:
            configured_gpio = getattr(config, "attention_servo_maestro_channel", None)
        min_pulse = _bounded_int(
            getattr(config, "attention_servo_min_pulse_width_us", _DEFAULT_MIN_PULSE_WIDTH_US),
            default=_DEFAULT_MIN_PULSE_WIDTH_US,
            minimum=500,
            maximum=2500,
        )
        max_pulse = _bounded_int(
            getattr(config, "attention_servo_max_pulse_width_us", _DEFAULT_MAX_PULSE_WIDTH_US),
            default=_DEFAULT_MAX_PULSE_WIDTH_US,
            minimum=500,
            maximum=2500,
        )
        if max_pulse < min_pulse:
            min_pulse, max_pulse = max_pulse, min_pulse
        center_pulse = _bounded_int(
            getattr(config, "attention_servo_center_pulse_width_us", _DEFAULT_CENTER_PULSE_WIDTH_US),
            default=_DEFAULT_CENTER_PULSE_WIDTH_US,
            minimum=min_pulse,
            maximum=max_pulse,
        )
        mechanical_range_degrees = _bounded_float(
            getattr(
                config,
                "attention_servo_mechanical_range_degrees",
                _DEFAULT_MECHANICAL_RANGE_DEGREES,
            ),
            default=_DEFAULT_MECHANICAL_RANGE_DEGREES,
            minimum=30.0,
            maximum=360.0,
        )
        target_hold_s = _bounded_float(
            getattr(config, "attention_servo_target_hold_s", _DEFAULT_TARGET_HOLD_S),
            default=_DEFAULT_TARGET_HOLD_S,
            minimum=0.0,
            maximum=5.0,
        )
        return cls(
            enabled=bool(getattr(config, "attention_servo_enabled", False)),
            driver=driver,
            control_mode=control_mode,
            maestro_device=None if maestro_device is None else (str(maestro_device).strip() or None),
            peer_base_url=None if peer_base_url is None else (str(peer_base_url).strip().rstrip("/") or None),
            peer_timeout_s=_bounded_float(
                peer_timeout_s,
                default=1.5,
                minimum=0.1,
                maximum=30.0,
            ),
            state_path=str(resolved_state_path),
            gpio_chip=str(getattr(config, "gpio_chip", "gpiochip0") or "gpiochip0"),
            gpio=configured_gpio,
            invert_direction=bool(getattr(config, "attention_servo_invert_direction", False)),
            target_hold_s=target_hold_s,
            loss_extrapolation_s=_bounded_float(
                getattr(config, "attention_servo_loss_extrapolation_s", _DEFAULT_LOSS_EXTRAPOLATION_S),
                default=_DEFAULT_LOSS_EXTRAPOLATION_S,
                minimum=0.0,
                maximum=5.0,
            ),
            loss_extrapolation_gain=_bounded_float(
                getattr(config, "attention_servo_loss_extrapolation_gain", _DEFAULT_LOSS_EXTRAPOLATION_GAIN),
                default=_DEFAULT_LOSS_EXTRAPOLATION_GAIN,
                minimum=0.0,
                maximum=4.0,
            ),
            min_confidence=_bounded_float(
                getattr(config, "attention_servo_min_confidence", _DEFAULT_MIN_CONFIDENCE),
                default=_DEFAULT_MIN_CONFIDENCE,
                minimum=0.0,
                maximum=1.0,
            ),
            deadband=_bounded_float(
                getattr(config, "attention_servo_deadband", _DEFAULT_DEADBAND),
                default=_DEFAULT_DEADBAND,
                minimum=0.0,
                maximum=0.3,
            ),
            min_pulse_width_us=min_pulse,
            center_pulse_width_us=center_pulse,
            max_pulse_width_us=max_pulse,
            max_step_us=_bounded_int(
                getattr(config, "attention_servo_max_step_us", _DEFAULT_MAX_STEP_US),
                default=_DEFAULT_MAX_STEP_US,
                minimum=1,
                maximum=max(1, max_pulse - min_pulse),
            ),
            target_smoothing_s=_bounded_float(
                getattr(config, "attention_servo_target_smoothing_s", _DEFAULT_TARGET_SMOOTHING_S),
                default=_DEFAULT_TARGET_SMOOTHING_S,
                minimum=0.0,
                maximum=5.0,
            ),
            max_velocity_us_per_s=_bounded_float(
                getattr(config, "attention_servo_max_velocity_us_per_s", _DEFAULT_MAX_VELOCITY_US_PER_S),
                default=_DEFAULT_MAX_VELOCITY_US_PER_S,
                minimum=1.0,
                maximum=1000.0,
            ),
            max_acceleration_us_per_s2=_bounded_float(
                getattr(
                    config,
                    "attention_servo_max_acceleration_us_per_s2",
                    _DEFAULT_MAX_ACCELERATION_US_PER_S2,
                ),
                default=_DEFAULT_MAX_ACCELERATION_US_PER_S2,
                minimum=1.0,
                maximum=10000.0,
            ),
            max_jerk_us_per_s3=_bounded_float(
                getattr(config, "attention_servo_max_jerk_us_per_s3", _DEFAULT_MAX_JERK_US_PER_S3),
                default=_DEFAULT_MAX_JERK_US_PER_S3,
                minimum=1.0,
                maximum=100000.0,
            ),
            rest_max_velocity_us_per_s=_bounded_float(
                getattr(
                    config,
                    "attention_servo_rest_max_velocity_us_per_s",
                    _DEFAULT_REST_MAX_VELOCITY_US_PER_S,
                ),
                default=_DEFAULT_REST_MAX_VELOCITY_US_PER_S,
                minimum=1.0,
                maximum=1000.0,
            ),
            rest_max_acceleration_us_per_s2=_bounded_float(
                getattr(
                    config,
                    "attention_servo_rest_max_acceleration_us_per_s2",
                    _DEFAULT_REST_MAX_ACCELERATION_US_PER_S2,
                ),
                default=_DEFAULT_REST_MAX_ACCELERATION_US_PER_S2,
                minimum=1.0,
                maximum=10000.0,
            ),
            rest_max_jerk_us_per_s3=_bounded_float(
                getattr(
                    config,
                    "attention_servo_rest_max_jerk_us_per_s3",
                    _DEFAULT_REST_MAX_JERK_US_PER_S3,
                ),
                default=_DEFAULT_REST_MAX_JERK_US_PER_S3,
                minimum=1.0,
                maximum=100000.0,
            ),
            min_command_delta_us=_bounded_int(
                getattr(config, "attention_servo_min_command_delta_us", _DEFAULT_MIN_COMMAND_DELTA_US),
                default=_DEFAULT_MIN_COMMAND_DELTA_US,
                minimum=1,
                maximum=max(1, max_pulse - min_pulse),
            ),
            visible_retarget_tolerance_us=_bounded_int(
                getattr(
                    config,
                    "attention_servo_visible_retarget_tolerance_us",
                    _DEFAULT_VISIBLE_RETARGET_TOLERANCE_US,
                ),
                default=_DEFAULT_VISIBLE_RETARGET_TOLERANCE_US,
                minimum=0,
                maximum=max(1, max_pulse - min_pulse),
            ),
            reference_interval_s=_bounded_float(
                getattr(config, "display_attention_refresh_interval_s", _DEFAULT_REFERENCE_INTERVAL_S),
                default=_DEFAULT_REFERENCE_INTERVAL_S,
                minimum=0.05,
                maximum=2.0,
            ),
            soft_limit_margin_us=_bounded_int(
                getattr(config, "attention_servo_soft_limit_margin_us", _DEFAULT_SOFT_LIMIT_MARGIN_US),
                default=_DEFAULT_SOFT_LIMIT_MARGIN_US,
                minimum=0,
                maximum=max(0, max_pulse - min_pulse),
            ),
            idle_release_s=_bounded_float(
                getattr(config, "attention_servo_idle_release_s", _DEFAULT_IDLE_RELEASE_S),
                default=_DEFAULT_IDLE_RELEASE_S,
                minimum=0.0,
                maximum=10.0,
            ),
            settled_release_s=_bounded_float(
                getattr(config, "attention_servo_settled_release_s", _DEFAULT_SETTLED_RELEASE_S),
                default=_DEFAULT_SETTLED_RELEASE_S,
                minimum=0.0,
                maximum=10.0,
            ),
            follow_exit_only=bool(
                getattr(config, "attention_servo_follow_exit_only", _DEFAULT_FOLLOW_EXIT_ONLY)
            ),
            visible_recenter_interval_s=_bounded_float(
                getattr(
                    config,
                    "attention_servo_visible_recenter_interval_s",
                    _DEFAULT_VISIBLE_RECENTER_INTERVAL_S,
                ),
                default=_DEFAULT_VISIBLE_RECENTER_INTERVAL_S,
                minimum=0.0,
                maximum=600.0,
            ),
            visible_recenter_center_tolerance=_bounded_float(
                getattr(
                    config,
                    "attention_servo_visible_recenter_center_tolerance",
                    _DEFAULT_VISIBLE_RECENTER_CENTER_TOLERANCE,
                ),
                default=_DEFAULT_VISIBLE_RECENTER_CENTER_TOLERANCE,
                minimum=0.0,
                maximum=0.3,
            ),
            mechanical_range_degrees=mechanical_range_degrees,
            exit_follow_max_degrees=_bounded_float(
                getattr(
                    config,
                    "attention_servo_exit_follow_max_degrees",
                    _DEFAULT_EXIT_FOLLOW_MAX_DEGREES,
                ),
                default=_DEFAULT_EXIT_FOLLOW_MAX_DEGREES,
                minimum=0.0,
                maximum=max(1.0, mechanical_range_degrees * 0.5),
            ),
            exit_activation_delay_s=_bounded_float(
                getattr(
                    config,
                    "attention_servo_exit_activation_delay_s",
                    _DEFAULT_EXIT_ACTIVATION_DELAY_S,
                ),
                default=_DEFAULT_EXIT_ACTIVATION_DELAY_S,
                minimum=0.0,
                maximum=max(0.0, target_hold_s),
            ),
            exit_settle_hold_s=_bounded_float(
                getattr(
                    config,
                    "attention_servo_exit_settle_hold_s",
                    _DEFAULT_EXIT_SETTLE_HOLD_S,
                ),
                default=_DEFAULT_EXIT_SETTLE_HOLD_S,
                minimum=0.0,
                maximum=10.0,
            ),
            exit_reacquire_center_tolerance=_bounded_float(
                getattr(
                    config,
                    "attention_servo_exit_reacquire_center_tolerance",
                    _DEFAULT_EXIT_REACQUIRE_CENTER_TOLERANCE,
                ),
                default=_DEFAULT_EXIT_REACQUIRE_CENTER_TOLERANCE,
                minimum=0.0,
                maximum=0.3,
            ),
            exit_visible_edge_threshold=_bounded_float(
                getattr(
                    config,
                    "attention_servo_exit_visible_edge_threshold",
                    _DEFAULT_EXIT_VISIBLE_EDGE_THRESHOLD,
                ),
                default=_DEFAULT_EXIT_VISIBLE_EDGE_THRESHOLD,
                minimum=0.55,
                maximum=0.95,
            ),
            exit_visible_box_edge_threshold=_bounded_float(
                getattr(
                    config,
                    "attention_servo_exit_visible_box_edge_threshold",
                    _DEFAULT_EXIT_VISIBLE_BOX_EDGE_THRESHOLD,
                ),
                default=_DEFAULT_EXIT_VISIBLE_BOX_EDGE_THRESHOLD,
                minimum=0.75,
                maximum=0.99,
            ),
            exit_cooldown_s=_bounded_float(
                getattr(
                    config,
                    "attention_servo_exit_cooldown_s",
                    _DEFAULT_EXIT_COOLDOWN_S,
                ),
                default=_DEFAULT_EXIT_COOLDOWN_S,
                minimum=0.0,
                maximum=300.0,
            ),
            continuous_max_speed_degrees_per_s=_bounded_float(
                getattr(
                    config,
                    "attention_servo_continuous_max_speed_degrees_per_s",
                    _DEFAULT_CONTINUOUS_MAX_SPEED_DEGREES_PER_S,
                ),
                default=_DEFAULT_CONTINUOUS_MAX_SPEED_DEGREES_PER_S,
                minimum=1.0,
                maximum=720.0,
            ),
            continuous_slow_zone_degrees=_bounded_float(
                getattr(
                    config,
                    "attention_servo_continuous_slow_zone_degrees",
                    _DEFAULT_CONTINUOUS_SLOW_ZONE_DEGREES,
                ),
                default=_DEFAULT_CONTINUOUS_SLOW_ZONE_DEGREES,
                minimum=0.5,
                maximum=max(1.0, mechanical_range_degrees * 0.5),
            ),
            continuous_stop_tolerance_degrees=_bounded_float(
                getattr(
                    config,
                    "attention_servo_continuous_stop_tolerance_degrees",
                    _DEFAULT_CONTINUOUS_STOP_TOLERANCE_DEGREES,
                ),
                default=_DEFAULT_CONTINUOUS_STOP_TOLERANCE_DEGREES,
                minimum=0.0,
                maximum=max(1.0, mechanical_range_degrees * 0.5),
            ),
            continuous_min_speed_pulse_delta_us=_bounded_int(
                getattr(
                    config,
                    "attention_servo_continuous_min_speed_pulse_delta_us",
                    _DEFAULT_CONTINUOUS_MIN_SPEED_PULSE_DELTA_US,
                ),
                default=_DEFAULT_CONTINUOUS_MIN_SPEED_PULSE_DELTA_US,
                minimum=0,
                maximum=max(1, min(center_pulse - min_pulse, max_pulse - center_pulse)),
            ),
            continuous_max_speed_pulse_delta_us=_bounded_int(
                getattr(
                    config,
                    "attention_servo_continuous_max_speed_pulse_delta_us",
                    _DEFAULT_CONTINUOUS_MAX_SPEED_PULSE_DELTA_US,
                ),
                default=_DEFAULT_CONTINUOUS_MAX_SPEED_PULSE_DELTA_US,
                minimum=0,
                maximum=max(1, min(center_pulse - min_pulse, max_pulse - center_pulse)),
            ),
        )


@dataclass(frozen=True, slots=True)
class AttentionServoDecision:
    """Describe one servo-follow update for tests and bounded telemetry."""

    observed_at: float | None = None
    active: bool = False
    reason: str = "disabled"
    confidence: float = 0.0
    target_center_x: float | None = None
    applied_center_x: float | None = None
    target_pulse_width_us: int | None = None
    commanded_pulse_width_us: int | None = None


class AttentionServoController:
    """Keep a calm servo orientation aligned with the current attention target."""

    def __init__(
        self,
        *,
        config: AttentionServoConfig,
        pulse_writer: ServoPulseWriter | None = None,
    ) -> None:
        self.config = config
        self._pulse_writer = pulse_writer or _default_pulse_writer_for_config(config)
        self._continuous_planner = self._build_continuous_planner()
        self._state_store = self._build_state_store()
        startup_state = self._load_runtime_state()
        self._continuous_startup_heading_degrees = (
            0.0
            if startup_state is None or not startup_state.zero_reference_confirmed
            else startup_state.heading_degrees
        )
        self._startup_hold_until_armed = bool(startup_state and startup_state.hold_until_armed)
        self._zero_reference_confirmed = bool(startup_state and startup_state.zero_reference_confirmed)
        self._last_saved_runtime_state = startup_state
        self._last_runtime_state_mtime_ns = (
            None if self._state_store is None else self._state_store.mtime_ns()
        )
        self._last_target_center_x: float | None = None
        self._last_target_at: float | None = None
        self._last_target_velocity_x_per_s = 0.0
        self._last_commanded_pulse_width_us: int | None = None
        self._last_physical_pulse_width_us: int | None = None
        self._planned_pulse_width_us: float | None = None
        self._planned_velocity_us_per_s = 0.0
        self._planned_acceleration_us_per_s2 = 0.0
        self._smoothed_center_x: float | None = None
        self._last_update_at: float | None = None
        self._centered_since: float | None = None
        self._settled_since: float | None = None
        self._released_pulse_width_us: int | None = None
        self._visible_target_pulse_width_us: int | None = None
        self._last_exit_hold_center_x: float | None = None
        self._recent_visible_targets: deque[tuple[float, float]] = deque()
        self._recent_visible_target_boxes: deque[tuple[float, float, float]] = deque()
        self._exit_pursuit_target_pulse_width_us: int | None = None
        self._exit_pursuit_center_x: float | None = None
        self._exit_pursuit_settled_at: float | None = None
        self._exit_cooldown_until_at: float | None = None
        self._visible_edge_departure_since_at: float | None = None
        self._visible_recenter_since_at: float | None = None
        self._visible_recenter_target_pulse_width_us: int | None = None
        self._visible_recenter_center_x: float | None = None
        self._visible_recenter_settled_at: float | None = None
        self._startup_rest_alignment_pending = False
        self._fault_reason: str | None = None
        self._disabled_due_to_fault = False
        self._fault_retry_after_at: float | None = None
        self._prime_last_physical_pulse_width_from_writer()
        self._release_stale_output_if_disabled()

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "AttentionServoController":
        """Build one servo controller from the global Twinr config."""

        servo_config = AttentionServoConfig.from_config(config)
        try:
            return cls(config=servo_config)
        except Exception as exc:
            disabled_config = replace(servo_config, enabled=False)
            controller = cls(config=disabled_config, pulse_writer=_NoopServoPulseWriter())
            controller._fault_reason = f"startup_preflight_failed: {exc.__class__.__name__}: {exc}"
            controller._disabled_due_to_fault = True
            return controller

    def _build_continuous_planner(self) -> ContinuousRotationServoPlanner | None:
        if not self.config.uses_continuous_rotation:
            return None
        return ContinuousRotationServoPlanner(
            ContinuousRotationServoConfig(
                center_pulse_width_us=self.config.center_pulse_width_us,
                min_pulse_width_us=self.config.safe_min_pulse_width_us,
                max_pulse_width_us=self.config.safe_max_pulse_width_us,
                max_heading_degrees=self._maximum_target_heading_degrees(),
                max_speed_degrees_per_s=self.config.continuous_max_speed_degrees_per_s,
                slow_zone_degrees=self.config.continuous_slow_zone_degrees,
                stop_tolerance_degrees=self.config.continuous_stop_tolerance_degrees,
                min_speed_pulse_delta_us=self.config.continuous_min_speed_pulse_delta_us,
                max_speed_pulse_delta_us=self.config.continuous_max_speed_pulse_delta_us,
            )
        )

    def _build_state_store(self) -> AttentionServoStateStore | None:
        if not self.config.uses_continuous_rotation:
            return None
        return AttentionServoStateStore(self.config.state_path)

    def _load_runtime_state(self) -> AttentionServoRuntimeState | None:
        if self._state_store is None:
            return None
        return self._state_store.load()

    def _current_runtime_state(self, *, observed_at: float | None) -> AttentionServoRuntimeState | None:
        if self._continuous_planner is None:
            return None
        return AttentionServoRuntimeState(
            heading_degrees=round(self._continuous_planner.estimated_heading_degrees, 3),
            hold_until_armed=self._startup_hold_until_armed,
            zero_reference_confirmed=self._zero_reference_confirmed,
            updated_at=observed_at,
        )

    def _persist_runtime_state(self, *, observed_at: float | None, force: bool = False) -> None:
        current_state = self._current_runtime_state(observed_at=observed_at)
        if current_state is None or self._state_store is None:
            return
        previous_state = self._last_saved_runtime_state
        state_changed = (
            previous_state is None
            or previous_state.heading_degrees != current_state.heading_degrees
            or previous_state.hold_until_armed != current_state.hold_until_armed
            or previous_state.zero_reference_confirmed != current_state.zero_reference_confirmed
        )
        if not force and not state_changed:
            return
        self._state_store.save(current_state)
        self._last_saved_runtime_state = current_state
        self._last_runtime_state_mtime_ns = self._state_store.mtime_ns()

    def _clear_tracking_state_for_external_override(self) -> None:
        """Drop transient follow latches before a live operator state override."""

        self._last_target_center_x = None
        self._last_target_at = None
        self._last_target_velocity_x_per_s = 0.0
        self._smoothed_center_x = None
        self._centered_since = None
        self._settled_since = None
        self._visible_target_pulse_width_us = None
        self._exit_cooldown_until_at = None
        self._clear_exit_pursuit(clear_recent_visible_targets=True)

    def _apply_loaded_runtime_state(
        self,
        state: AttentionServoRuntimeState,
        *,
        observed_at: float | None,
    ) -> None:
        """Adopt one externally updated state snapshot into the live controller."""

        previous_hold = self._startup_hold_until_armed
        previous_zero_reference = self._zero_reference_confirmed
        previous_heading_degrees = self._continuous_startup_heading_degrees
        refreshed_heading_degrees = (
            0.0 if not state.zero_reference_confirmed else state.heading_degrees
        )
        self._startup_hold_until_armed = bool(state.hold_until_armed)
        self._zero_reference_confirmed = bool(state.zero_reference_confirmed)
        self._continuous_startup_heading_degrees = refreshed_heading_degrees
        self._last_saved_runtime_state = state
        entering_manual_hold = not previous_hold and self._startup_hold_until_armed
        heading_changed = not math.isclose(
            previous_heading_degrees,
            refreshed_heading_degrees,
            abs_tol=0.001,
        )
        zero_reference_changed = previous_zero_reference != self._zero_reference_confirmed
        if entering_manual_hold or heading_changed or zero_reference_changed:
            self._clear_tracking_state_for_external_override()
        if self._continuous_planner is None:
            return
        if not heading_changed and not zero_reference_changed:
            return
        self._continuous_planner.reset(
            heading_degrees=refreshed_heading_degrees,
            observed_at=observed_at,
        )
        self._last_physical_pulse_width_us = self.config.center_pulse_width_us
        self._released_pulse_width_us = (
            self.config.center_pulse_width_us
            if self._last_commanded_pulse_width_us is None
            else None
        )
        self._reset_motion_state(self.config.center_pulse_width_us)

    def _refresh_runtime_state_from_store(self, *, observed_at: float | None) -> None:
        """Reload the persisted runtime state when an operator changes it live."""

        if self._state_store is None:
            return
        refreshed_state = self._state_store.load()
        self._last_runtime_state_mtime_ns = self._state_store.mtime_ns()
        if refreshed_state is None or refreshed_state == self._last_saved_runtime_state:
            return
        self._apply_loaded_runtime_state(refreshed_state, observed_at=observed_at)

    def _command_manual_hold_center(self, *, observed_at: float | None) -> int:
        """Drive the neutral stop pulse continuously while startup hold is active."""

        target_pulse_width_us = self.config.center_pulse_width_us
        try:
            if self._last_commanded_pulse_width_us != target_pulse_width_us:
                self._pulse_writer.write(
                    gpio_chip=self.config.gpio_chip,
                    gpio=self.config.gpio if self.config.gpio is not None else 0,
                    pulse_width_us=target_pulse_width_us,
                )
            if self._continuous_planner is not None:
                self._continuous_planner.note_commanded_pulse_width(
                    target_pulse_width_us,
                    observed_at=observed_at,
                )
        except Exception as exc:
            self._fault_reason = f"{exc.__class__.__name__}: {exc}"
            raise
        self._last_commanded_pulse_width_us = target_pulse_width_us
        self._last_physical_pulse_width_us = target_pulse_width_us
        self._released_pulse_width_us = None
        self._reset_motion_state(target_pulse_width_us)
        self._persist_runtime_state(observed_at=observed_at, force=True)
        return target_pulse_width_us

    def update(
        self,
        *,
        observed_at: float | None,
        active: bool,
        target_center_x: float | None,
        confidence: float | None,
        visible_target_present: bool | None = None,
        visible_target_box_left: float | None = None,
        visible_target_box_right: float | None = None,
    ) -> AttentionServoDecision:
        """Apply one bounded servo update from a pre-derived attention target."""

        checked_at = None if observed_at is None else float(observed_at)
        checked_confidence = _clamp_ratio(
            _bounded_float(confidence, default=0.0, minimum=0.0, maximum=1.0)
        )
        checked_visible_target_present = (
            self._visible_target_present(
                active=active,
                target_center_x=target_center_x,
            )
            if visible_target_present is None
            else (
                bool(visible_target_present)
                and target_center_x is not None
            )
        )
        effective_active = (
            active
            if visible_target_present is None
            else (active and bool(visible_target_present))
        )
        if not self.config.enabled and not self._disabled_due_to_fault:
            return AttentionServoDecision(
                observed_at=observed_at,
                active=False,
                reason="disabled",
                confidence=checked_confidence,
                target_center_x=target_center_x,
            )
        if self.config.gpio is None:
            return AttentionServoDecision(
                observed_at=observed_at,
                active=False,
                reason="unconfigured_gpio",
                confidence=checked_confidence,
                target_center_x=target_center_x,
            )
        if self._fault_reason is not None and not self._maybe_recover_from_fault(observed_at=checked_at):
            return AttentionServoDecision(
                observed_at=observed_at,
                active=False,
                reason="faulted",
                confidence=checked_confidence,
                target_center_x=target_center_x,
            )
        self._refresh_runtime_state_from_store(observed_at=checked_at)
        if self._startup_hold_until_armed and self._continuous_planner is not None:
            commanded_pulse_width_us = self._command_manual_hold_center(observed_at=checked_at)
            self._last_update_at = checked_at
            return AttentionServoDecision(
                observed_at=observed_at,
                active=False,
                reason="manual_hold",
                confidence=checked_confidence,
                target_center_x=target_center_x,
                applied_center_x=0.5,
                target_pulse_width_us=self.config.center_pulse_width_us,
                commanded_pulse_width_us=commanded_pulse_width_us,
            )
        if self.config.follow_exit_only:
            return self._update_follow_exit_only(
                observed_at=checked_at,
                target_center_x=target_center_x,
                confidence=checked_confidence,
                visible_target_present=checked_visible_target_present,
                visible_target_box_left=visible_target_box_left,
                visible_target_box_right=visible_target_box_right,
            )
        if self.config.follow_exit_only and checked_visible_target_present:
            return self._update_visible_target_for_exit_only(
                observed_at=checked_at,
                target_center_x=float(target_center_x),
                confidence=checked_confidence,
            )

        applied_center_x, effective_active, reason = self._resolve_target(
            observed_at=checked_at,
            active=effective_active,
            target_center_x=target_center_x,
            confidence=checked_confidence,
        )
        if reason == "awaiting_exit_confirmation":
            return self._await_exit_confirmation(
                observed_at=checked_at,
                confidence=checked_confidence,
                target_center_x=target_center_x,
                applied_center_x=applied_center_x,
            )
        applied_center_x = self._clamped_follow_center_x(applied_center_x)
        applied_center_x = self._smoothed_target_center_x(
            observed_at=checked_at,
            target_center_x=applied_center_x,
        )
        target_pulse_width_us = self._target_pulse_width_for_center_x(
            center_x=applied_center_x,
            observed_at=checked_at,
        )
        target_pulse_width_us = self._stabilize_visible_target_pulse_width(
            target_pulse_width_us=target_pulse_width_us,
            reason=reason,
        )
        if self._maybe_release_idle(
            observed_at=checked_at,
            active=effective_active,
            target_pulse_width_us=target_pulse_width_us,
        ):
            self._last_update_at = checked_at
            return AttentionServoDecision(
                observed_at=checked_at,
                active=False,
                reason="idle_released",
                confidence=checked_confidence,
                target_center_x=target_center_x,
                applied_center_x=applied_center_x,
                target_pulse_width_us=target_pulse_width_us,
                commanded_pulse_width_us=None,
            )
        released_decision = self._maybe_hold_released_target(
            observed_at=checked_at,
            active=effective_active,
            reason=reason,
            confidence=checked_confidence,
            target_center_x=target_center_x,
            applied_center_x=applied_center_x,
            target_pulse_width_us=target_pulse_width_us,
        )
        if released_decision is not None:
            return released_decision
        planned_pulse_width_us = self._advance_planned_pulse_width(
            target_pulse_width_us,
            observed_at=checked_at,
            motion_profile="tracking",
        )
        commanded_pulse_width_us = self._command_pulse_width_for_plan(
            planned_pulse_width_us,
            target_pulse_width_us=target_pulse_width_us,
            motion_profile="tracking",
        )
        released_exit_decision = self._maybe_release_exit_hold(
            observed_at=checked_at,
            reason=reason,
            confidence=checked_confidence,
            target_center_x=target_center_x,
            applied_center_x=applied_center_x,
            target_pulse_width_us=target_pulse_width_us,
            commanded_pulse_width_us=commanded_pulse_width_us,
        )
        if released_exit_decision is not None:
            self._last_update_at = checked_at
            return released_exit_decision
        if self._maybe_release_settled_target(
            observed_at=checked_at,
            target_pulse_width_us=target_pulse_width_us,
            commanded_pulse_width_us=commanded_pulse_width_us,
        ):
            self._last_update_at = checked_at
            return AttentionServoDecision(
                observed_at=observed_at,
                active=effective_active,
                reason="settled_released",
                confidence=checked_confidence,
                target_center_x=target_center_x,
                applied_center_x=applied_center_x,
                target_pulse_width_us=target_pulse_width_us,
                commanded_pulse_width_us=None,
            )

        try:
            if (
                self._last_commanded_pulse_width_us is None
                or commanded_pulse_width_us != self._last_commanded_pulse_width_us
            ):
                self._pulse_writer.write(
                    gpio_chip=self.config.gpio_chip,
                    gpio=self.config.gpio,
                    pulse_width_us=commanded_pulse_width_us,
                )
                if self._continuous_planner is not None:
                    self._continuous_planner.note_commanded_pulse_width(
                        commanded_pulse_width_us,
                        observed_at=checked_at,
                    )
                self._last_commanded_pulse_width_us = commanded_pulse_width_us
                self._last_physical_pulse_width_us = commanded_pulse_width_us
                self._persist_runtime_state(observed_at=checked_at)
            self._last_update_at = checked_at
        except Exception as exc:
            self._fault_reason = f"{exc.__class__.__name__}: {exc}"
            raise

        return AttentionServoDecision(
            observed_at=observed_at,
            active=effective_active,
            reason=reason,
            confidence=checked_confidence,
            target_center_x=target_center_x,
            applied_center_x=applied_center_x,
            target_pulse_width_us=target_pulse_width_us,
            commanded_pulse_width_us=commanded_pulse_width_us,
        )

    def _update_follow_exit_only(
        self,
        *,
        observed_at: float | None,
        target_center_x: float | None,
        confidence: float,
        visible_target_present: bool,
        visible_target_box_left: float | None,
        visible_target_box_right: float | None,
    ) -> AttentionServoDecision:
        normalized_target_center_x = None if target_center_x is None else _clamp_ratio(float(target_center_x))
        if (
            self._exit_cooldown_until_at is not None
            and observed_at is not None
            and observed_at >= self._exit_cooldown_until_at
        ):
            self._exit_cooldown_until_at = None
        if self._exit_cooldown_until_at is not None:
            return self._hold_exit_cooldown(
                observed_at=observed_at,
                confidence=confidence,
                target_center_x=target_center_x,
                normalized_target_center_x=normalized_target_center_x,
                visible_target_present=visible_target_present,
            )
        if self._startup_rest_alignment_pending and self._exit_pursuit_target_pulse_width_us is None:
            if visible_target_present and normalized_target_center_x is not None:
                return self._handle_visible_target_during_startup_alignment(
                    observed_at=observed_at,
                    confidence=confidence,
                    target_center_x=target_center_x,
                    normalized_target_center_x=normalized_target_center_x,
                )
            rest_decision = self._drive_rest_position(
                observed_at=observed_at,
                confidence=confidence,
                target_center_x=target_center_x,
            )
            if rest_decision.reason == "idle_released":
                self._startup_rest_alignment_pending = False
            return rest_decision
        if visible_target_present and normalized_target_center_x is not None:
            return self._handle_visible_target_for_exit_only(
                observed_at=observed_at,
                confidence=confidence,
                target_center_x=target_center_x,
                normalized_target_center_x=normalized_target_center_x,
                visible_target_box_left=visible_target_box_left,
                visible_target_box_right=visible_target_box_right,
            )
        self._clear_visible_recenter(reset_timer=True)
        if self._exit_pursuit_target_pulse_width_us is not None:
            if self._should_return_to_rest_from_exit_pursuit(
                observed_at=observed_at,
                visible_target_present=visible_target_present,
            ):
                self._clear_exit_pursuit(clear_recent_visible_targets=False)
                self._last_target_center_x = None
                self._last_target_at = None
                self._last_target_velocity_x_per_s = 0.0
                self._smoothed_center_x = None
                self._visible_target_pulse_width_us = None
                return self._drive_rest_position(
                    observed_at=observed_at,
                    confidence=confidence,
                    target_center_x=target_center_x,
                )
            return self._drive_exit_pursuit_target(
                observed_at=observed_at,
                confidence=confidence,
                target_center_x=target_center_x,
                reason="pursuing_exit_direction",
            )
        if (
            self._last_target_center_x is not None
            and self._last_target_at is not None
            and observed_at is not None
            and (observed_at - self._last_target_at) <= self.config.target_hold_s
        ):
            loss_elapsed_s = max(0.0, observed_at - self._last_target_at)
            if loss_elapsed_s < self.config.exit_activation_delay_s:
                self._last_exit_hold_center_x = None
                self._release_active_output(observed_at=observed_at)
                self._last_update_at = observed_at
                return AttentionServoDecision(
                    observed_at=observed_at,
                    active=False,
                    reason="awaiting_exit_confirmation",
                    confidence=confidence,
                    target_center_x=target_center_x,
                    applied_center_x=self._last_target_center_x,
                    target_pulse_width_us=None,
                    commanded_pulse_width_us=None,
                )
            if self._begin_exit_pursuit(observed_at=observed_at) is not None:
                return self._drive_exit_pursuit_target(
                    observed_at=observed_at,
                    confidence=confidence,
                    target_center_x=target_center_x,
                    reason="pursuing_exit_direction",
                )
        self._clear_exit_pursuit(clear_recent_visible_targets=False)
        return self._drive_rest_position(
            observed_at=observed_at,
            confidence=confidence,
            target_center_x=target_center_x,
        )

    def _handle_visible_target_for_exit_only(
        self,
        *,
        observed_at: float | None,
        confidence: float,
        target_center_x: float | None,
        normalized_target_center_x: float,
        visible_target_box_left: float | None,
        visible_target_box_right: float | None,
    ) -> AttentionServoDecision:
        self._remember_visible_target_box(
            observed_at=observed_at,
            box_left=visible_target_box_left,
            box_right=visible_target_box_right,
        )
        self._remember_visible_target(
            observed_at=observed_at,
            target_center_x=normalized_target_center_x,
        )
        self._update_target_velocity(
            observed_at=observed_at,
            target_center_x=normalized_target_center_x,
        )
        self._last_target_center_x = normalized_target_center_x
        self._last_target_at = observed_at
        self._last_exit_hold_center_x = None
        self._smoothed_center_x = normalized_target_center_x
        edge_departure_confirmed = self._visible_edge_departure_confirmed(
            observed_at=observed_at,
            target_center_x=normalized_target_center_x,
        )
        if self._exit_pursuit_target_pulse_width_us is None:
            if edge_departure_confirmed and self._begin_exit_pursuit(observed_at=observed_at) is not None:
                return self._drive_exit_pursuit_target(
                    observed_at=observed_at,
                    confidence=confidence,
                    target_center_x=target_center_x,
                    reason="pursuing_edge_departure",
                )
            if self._visible_recenter_target_pulse_width_us is not None:
                return self._drive_visible_recenter_target(
                    observed_at=observed_at,
                    confidence=confidence,
                    target_center_x=target_center_x,
                    normalized_target_center_x=normalized_target_center_x,
                )
            if self._target_is_visible_recenter_centered(normalized_target_center_x):
                self._clear_visible_recenter(reset_timer=True)
                self._exit_pursuit_settled_at = None
                self._release_active_output(observed_at=observed_at)
                self._last_update_at = observed_at
                return AttentionServoDecision(
                    observed_at=observed_at,
                    active=False,
                    reason="waiting_for_exit",
                    confidence=confidence,
                    target_center_x=target_center_x,
                    applied_center_x=normalized_target_center_x,
                    target_pulse_width_us=None,
                    commanded_pulse_width_us=None,
                )
            if self._should_begin_visible_recenter(observed_at=observed_at):
                if self._begin_visible_recenter(normalized_target_center_x=normalized_target_center_x):
                    return self._drive_visible_recenter_target(
                        observed_at=observed_at,
                        confidence=confidence,
                        target_center_x=target_center_x,
                        normalized_target_center_x=normalized_target_center_x,
                    )
                self._clear_visible_recenter(reset_timer=True)
            self._exit_pursuit_settled_at = None
            self._release_active_output(observed_at=observed_at)
            self._last_update_at = observed_at
            return AttentionServoDecision(
                observed_at=observed_at,
                active=False,
                reason="waiting_for_exit",
                confidence=confidence,
                target_center_x=target_center_x,
                applied_center_x=normalized_target_center_x,
                target_pulse_width_us=None,
                commanded_pulse_width_us=None,
            )
        if not self._target_is_exit_reacquired_centered(normalized_target_center_x):
            return self._drive_exit_pursuit_target(
                observed_at=observed_at,
                confidence=confidence,
                target_center_x=target_center_x,
                reason="pursuing_edge_departure",
            )
        target_pulse_width_us = self._exit_pursuit_target_pulse_width_us
        self._start_exit_cooldown(observed_at=observed_at)
        self._release_active_output(observed_at=observed_at)
        self._last_update_at = observed_at
        return AttentionServoDecision(
            observed_at=observed_at,
            active=False,
            reason="reacquired_visible_cooldown",
            confidence=confidence,
            target_center_x=target_center_x,
            applied_center_x=normalized_target_center_x,
            target_pulse_width_us=target_pulse_width_us,
            commanded_pulse_width_us=None,
        )

    def _hold_exit_cooldown(
        self,
        *,
        observed_at: float | None,
        confidence: float,
        target_center_x: float | None,
        normalized_target_center_x: float | None,
        visible_target_present: bool,
    ) -> AttentionServoDecision:
        if visible_target_present and normalized_target_center_x is not None:
            self._remember_visible_target(
                observed_at=observed_at,
                target_center_x=normalized_target_center_x,
            )
            self._update_target_velocity(
                observed_at=observed_at,
                target_center_x=normalized_target_center_x,
            )
            self._last_target_center_x = normalized_target_center_x
            self._last_target_at = observed_at
            self._smoothed_center_x = normalized_target_center_x
        self._release_active_output(observed_at=observed_at)
        self._last_update_at = observed_at
        return AttentionServoDecision(
            observed_at=observed_at,
            active=False,
            reason="exit_cooldown",
            confidence=confidence,
            target_center_x=target_center_x,
            applied_center_x=normalized_target_center_x,
            target_pulse_width_us=None,
            commanded_pulse_width_us=None,
        )

    def _drive_exit_pursuit_target(
        self,
        *,
        observed_at: float | None,
        confidence: float,
        target_center_x: float | None,
        reason: str,
    ) -> AttentionServoDecision:
        applied_center_x = self._exit_pursuit_center_x
        if applied_center_x is None:
            return self._drive_rest_position(
                observed_at=observed_at,
                confidence=confidence,
                target_center_x=target_center_x,
            )
        target_pulse_width_us = self._target_pulse_width_for_center_x(
            center_x=applied_center_x,
            observed_at=observed_at,
        )
        self._exit_pursuit_target_pulse_width_us = target_pulse_width_us
        if (
            self._exit_pursuit_settled_at is not None
            and self._last_commanded_pulse_width_us is None
        ):
            self._last_update_at = observed_at
            return AttentionServoDecision(
                observed_at=observed_at,
                active=False,
                reason="holding_exit_limit",
                confidence=confidence,
                target_center_x=target_center_x,
                applied_center_x=applied_center_x,
                target_pulse_width_us=target_pulse_width_us,
                commanded_pulse_width_us=None,
            )
        commanded_pulse_width_us = self._write_target_pulse_width(
            observed_at=observed_at,
            target_pulse_width_us=target_pulse_width_us,
            motion_profile="exit",
        )
        tolerance_us = self._release_tolerance_us()
        if abs(target_pulse_width_us - commanded_pulse_width_us) <= tolerance_us:
            if (
                observed_at is not None
                and self._exit_pursuit_settled_at is not None
                and observed_at >= self._exit_pursuit_settled_at
                and (observed_at - self._exit_pursuit_settled_at) >= self.config.exit_settle_hold_s
            ):
                self._release_active_output(observed_at=observed_at)
                self._last_update_at = observed_at
                return AttentionServoDecision(
                    observed_at=observed_at,
                    active=False,
                    reason="holding_exit_limit",
                    confidence=confidence,
                    target_center_x=target_center_x,
                    applied_center_x=applied_center_x,
                    target_pulse_width_us=target_pulse_width_us,
                    commanded_pulse_width_us=None,
                )
            if observed_at is not None and (
                self._exit_pursuit_settled_at is None or observed_at < self._exit_pursuit_settled_at
            ):
                self._exit_pursuit_settled_at = observed_at
        else:
            self._exit_pursuit_settled_at = None
        return AttentionServoDecision(
            observed_at=observed_at,
            active=True,
            reason=reason,
            confidence=confidence,
            target_center_x=target_center_x,
            applied_center_x=applied_center_x,
            target_pulse_width_us=target_pulse_width_us,
            commanded_pulse_width_us=commanded_pulse_width_us,
        )

    def _drive_visible_recenter_target(
        self,
        *,
        observed_at: float | None,
        confidence: float,
        target_center_x: float | None,
        normalized_target_center_x: float,
    ) -> AttentionServoDecision:
        applied_center_x = self._visible_recenter_center_x
        if applied_center_x is None:
            self._clear_visible_recenter(reset_timer=True)
            self._release_active_output(observed_at=observed_at)
            self._last_update_at = observed_at
            return AttentionServoDecision(
                observed_at=observed_at,
                active=False,
                reason="waiting_for_exit",
                confidence=confidence,
                target_center_x=target_center_x,
                applied_center_x=normalized_target_center_x,
                target_pulse_width_us=None,
                commanded_pulse_width_us=None,
            )
        target_pulse_width_us = self._target_pulse_width_for_center_x(
            center_x=applied_center_x,
            observed_at=observed_at,
        )
        self._visible_recenter_target_pulse_width_us = target_pulse_width_us
        if self._target_is_visible_recenter_centered(normalized_target_center_x):
            self._clear_visible_recenter(reset_timer=True)
            self._release_active_output(observed_at=observed_at)
            self._last_update_at = observed_at
            return AttentionServoDecision(
                observed_at=observed_at,
                active=False,
                reason="waiting_for_exit",
                confidence=confidence,
                target_center_x=target_center_x,
                applied_center_x=normalized_target_center_x,
                target_pulse_width_us=target_pulse_width_us,
                commanded_pulse_width_us=None,
            )
        target_side = self._exit_side_for_center_x(normalized_target_center_x)
        latched_side = self._exit_side_for_center_x(applied_center_x)
        if target_side != 0 and latched_side != 0 and target_side != latched_side:
            self._clear_visible_recenter(reset_timer=True)
            self._release_active_output(observed_at=observed_at)
            self._last_update_at = observed_at
            return AttentionServoDecision(
                observed_at=observed_at,
                active=False,
                reason="waiting_for_exit",
                confidence=confidence,
                target_center_x=target_center_x,
                applied_center_x=normalized_target_center_x,
                target_pulse_width_us=None,
                commanded_pulse_width_us=None,
            )
        commanded_pulse_width_us = self._write_target_pulse_width(
            observed_at=observed_at,
            target_pulse_width_us=target_pulse_width_us,
            motion_profile="tracking",
        )
        tolerance_us = self._release_tolerance_us()
        if abs(target_pulse_width_us - commanded_pulse_width_us) <= tolerance_us:
            if (
                observed_at is not None
                and self._visible_recenter_settled_at is not None
                and observed_at >= self._visible_recenter_settled_at
                and (observed_at - self._visible_recenter_settled_at) >= self.config.settled_release_s
            ):
                self._clear_visible_recenter(reset_timer=True)
                self._release_active_output(observed_at=observed_at)
                self._last_update_at = observed_at
                return AttentionServoDecision(
                    observed_at=observed_at,
                    active=False,
                    reason="waiting_for_exit",
                    confidence=confidence,
                    target_center_x=target_center_x,
                    applied_center_x=normalized_target_center_x,
                    target_pulse_width_us=target_pulse_width_us,
                    commanded_pulse_width_us=None,
                )
            if observed_at is not None and (
                self._visible_recenter_settled_at is None
                or observed_at < self._visible_recenter_settled_at
            ):
                self._visible_recenter_settled_at = observed_at
        else:
            self._visible_recenter_settled_at = None
        return AttentionServoDecision(
            observed_at=observed_at,
            active=True,
            reason="pursuing_visible_recenter",
            confidence=confidence,
            target_center_x=target_center_x,
            applied_center_x=applied_center_x,
            target_pulse_width_us=target_pulse_width_us,
            commanded_pulse_width_us=commanded_pulse_width_us,
        )

    def _drive_rest_position(
        self,
        *,
        observed_at: float | None,
        confidence: float,
        target_center_x: float | None,
    ) -> AttentionServoDecision:
        target_pulse_width_us = self.config.center_pulse_width_us
        if self._maybe_release_idle(
            observed_at=observed_at,
            active=False,
            target_pulse_width_us=target_pulse_width_us,
        ):
            self._last_update_at = observed_at
            return AttentionServoDecision(
                observed_at=observed_at,
                active=False,
                reason="idle_released",
                confidence=confidence,
                target_center_x=target_center_x,
                applied_center_x=0.5,
                target_pulse_width_us=target_pulse_width_us,
                commanded_pulse_width_us=None,
            )
        commanded_pulse_width_us = self._write_target_pulse_width(
            observed_at=observed_at,
            target_pulse_width_us=target_pulse_width_us,
            motion_profile="rest",
        )
        return AttentionServoDecision(
            observed_at=observed_at,
            active=False,
            reason="recentering",
            confidence=confidence,
            target_center_x=target_center_x,
            applied_center_x=0.5,
            target_pulse_width_us=target_pulse_width_us,
            commanded_pulse_width_us=commanded_pulse_width_us,
        )

    def _write_target_pulse_width(
        self,
        *,
        observed_at: float | None,
        target_pulse_width_us: int,
        motion_profile: str,
    ) -> int:
        planned_pulse_width_us = self._advance_planned_pulse_width(
            target_pulse_width_us,
            observed_at=observed_at,
            motion_profile=motion_profile,
        )
        commanded_pulse_width_us = self._command_pulse_width_for_plan(
            planned_pulse_width_us,
            target_pulse_width_us=target_pulse_width_us,
            motion_profile=motion_profile,
        )
        try:
            if self.config.gpio is None:
                raise RuntimeError("Attention servo output is enabled without a configured output channel")
            if (
                self._last_commanded_pulse_width_us is None
                or commanded_pulse_width_us != self._last_commanded_pulse_width_us
            ):
                self._pulse_writer.write(
                    gpio_chip=self.config.gpio_chip,
                    gpio=self.config.gpio,
                    pulse_width_us=commanded_pulse_width_us,
                )
                if self._continuous_planner is not None:
                    self._continuous_planner.note_commanded_pulse_width(
                        commanded_pulse_width_us,
                        observed_at=observed_at,
                    )
                self._last_commanded_pulse_width_us = commanded_pulse_width_us
                self._last_physical_pulse_width_us = commanded_pulse_width_us
                self._persist_runtime_state(observed_at=observed_at)
            if (
                str(motion_profile or "").strip().lower() == "rest"
                and commanded_pulse_width_us == self.config.center_pulse_width_us
            ):
                self._reset_motion_state(commanded_pulse_width_us)
            self._last_update_at = observed_at
        except Exception as exc:
            self._fault_reason = f"{exc.__class__.__name__}: {exc}"
            raise
        return commanded_pulse_width_us

    def _begin_exit_pursuit(self, *, observed_at: float | None) -> float | None:
        anchor_center_x = self._recent_exit_anchor_center_x(observed_at=observed_at)
        if anchor_center_x is None:
            return None
        anchor_offset = anchor_center_x - 0.5
        if abs(anchor_offset) <= self.config.deadband:
            return None
        exit_sign = 1.0 if anchor_offset > 0.0 else -1.0
        exit_center_x = _clamp_ratio(0.5 + (0.5 * self.config.exit_follow_offset_limit * exit_sign))
        self._clear_visible_recenter(reset_timer=True)
        self._exit_pursuit_center_x = exit_center_x
        self._exit_pursuit_target_pulse_width_us = self._target_pulse_width_for_center_x(
            center_x=exit_center_x,
            observed_at=observed_at,
        )
        self._exit_pursuit_settled_at = None
        self._last_exit_hold_center_x = exit_center_x
        return exit_center_x

    def _start_exit_cooldown(self, *, observed_at: float | None) -> None:
        cooldown_anchor_at = time.monotonic() if observed_at is None else observed_at
        self._exit_cooldown_until_at = cooldown_anchor_at + self.config.exit_cooldown_s
        self._clear_exit_pursuit(clear_recent_visible_targets=True)
        self._last_target_center_x = None
        self._last_target_at = None
        self._last_target_velocity_x_per_s = 0.0
        self._smoothed_center_x = None
        self._visible_target_pulse_width_us = None

    def _clear_exit_pursuit(self, *, clear_recent_visible_targets: bool) -> None:
        self._exit_pursuit_target_pulse_width_us = None
        self._exit_pursuit_center_x = None
        self._exit_pursuit_settled_at = None
        self._last_exit_hold_center_x = None
        self._visible_edge_departure_since_at = None
        self._clear_visible_recenter(reset_timer=True)
        if clear_recent_visible_targets:
            self._recent_visible_targets.clear()
            self._recent_visible_target_boxes.clear()

    def _should_return_to_rest_from_exit_pursuit(
        self,
        *,
        observed_at: float | None,
        visible_target_present: bool,
    ) -> bool:
        if visible_target_present or self._exit_pursuit_settled_at is None:
            return False
        if observed_at is None:
            return False
        return_anchor_at = self._last_target_at
        if return_anchor_at is None:
            return_anchor_at = self._exit_pursuit_settled_at
        return (observed_at - return_anchor_at) >= self.config.exit_cooldown_s

    def close(self) -> None:
        """Stop the current pulse train and release any underlying resources."""

        gpio = self.config.gpio
        try:
            if self.config.enabled and gpio is not None and self._last_commanded_pulse_width_us is not None:
                self._pulse_writer.disable(
                    gpio_chip=self.config.gpio_chip,
                    gpio=gpio,
                )
        finally:
            self._last_commanded_pulse_width_us = None
            self._reset_motion_state()
            if self._continuous_planner is not None:
                self._continuous_planner.reset()
            self._last_target_velocity_x_per_s = 0.0
            self._recent_visible_targets.clear()
            self._recent_visible_target_boxes.clear()
            self._exit_pursuit_target_pulse_width_us = None
            self._exit_pursuit_center_x = None
            self._exit_pursuit_settled_at = None
            self._exit_cooldown_until_at = None
            self._visible_edge_departure_since_at = None
            self._clear_visible_recenter(reset_timer=True)
            self._smoothed_center_x = None
            self._last_update_at = None
            self._centered_since = None
            self._settled_since = None
            self._released_pulse_width_us = None
            self._visible_target_pulse_width_us = None
            self._last_physical_pulse_width_us = None
            self._last_exit_hold_center_x = None
            self._pulse_writer.close()

    def debug_snapshot(self, *, observed_at: float | None) -> dict[str, object]:
        """Return one bounded debug snapshot of the controller's internal state."""

        checked_at = None if observed_at is None else float(observed_at)
        recent_targets: list[dict[str, object]] = []
        for sample_at, sample_center_x in list(self._recent_visible_targets)[-6:]:
            age_s = None if checked_at is None else round(max(0.0, checked_at - sample_at), 3)
            recent_targets.append(
                {
                    "age_s": age_s,
                    "center_x": round(sample_center_x, 4),
                }
            )
        last_target_age_s = None
        if checked_at is not None and self._last_target_at is not None:
            last_target_age_s = round(max(0.0, checked_at - self._last_target_at), 3)
        visible_departure_age_s = None
        if checked_at is not None and self._visible_edge_departure_since_at is not None:
            visible_departure_age_s = round(max(0.0, checked_at - self._visible_edge_departure_since_at), 3)
        visible_recenter_age_s = None
        if checked_at is not None and self._visible_recenter_since_at is not None:
            visible_recenter_age_s = round(max(0.0, checked_at - self._visible_recenter_since_at), 3)
        visible_recenter_settled_age_s = None
        if checked_at is not None and self._visible_recenter_settled_at is not None:
            visible_recenter_settled_age_s = round(max(0.0, checked_at - self._visible_recenter_settled_at), 3)
        exit_cooldown_remaining_s = None
        if checked_at is not None and self._exit_cooldown_until_at is not None:
            exit_cooldown_remaining_s = round(max(0.0, self._exit_cooldown_until_at - checked_at), 3)
        recent_exit_anchor_center_x = (
            None if checked_at is None else self._recent_exit_anchor_center_x(observed_at=checked_at)
        )
        recent_exit_box_edge = (
            None if checked_at is None else self._recent_exit_anchor_box_edge(observed_at=checked_at)
        )
        return {
            "last_target_center_x": None if self._last_target_center_x is None else round(self._last_target_center_x, 4),
            "last_target_age_s": last_target_age_s,
            "last_target_velocity_x_per_s": round(self._last_target_velocity_x_per_s, 4),
            "recent_visible_targets": recent_targets,
            "recent_exit_anchor_center_x": (
                None if recent_exit_anchor_center_x is None else round(recent_exit_anchor_center_x, 4)
            ),
            "recent_exit_box_edge": None if recent_exit_box_edge is None else round(recent_exit_box_edge, 4),
            "visible_edge_departure_age_s": visible_departure_age_s,
            "visible_recenter_age_s": visible_recenter_age_s,
            "visible_recenter_center_x": (
                None if self._visible_recenter_center_x is None else round(self._visible_recenter_center_x, 4)
            ),
            "visible_recenter_target_pulse_width_us": self._visible_recenter_target_pulse_width_us,
            "visible_recenter_settled_age_s": visible_recenter_settled_age_s,
            "exit_pursuit_center_x": None if self._exit_pursuit_center_x is None else round(self._exit_pursuit_center_x, 4),
            "exit_pursuit_target_pulse_width_us": self._exit_pursuit_target_pulse_width_us,
            "exit_pursuit_settled_at": self._exit_pursuit_settled_at,
            "exit_cooldown_remaining_s": exit_cooldown_remaining_s,
            "last_commanded_pulse_width_us": self._last_commanded_pulse_width_us,
            "last_physical_pulse_width_us": self._last_physical_pulse_width_us,
            "released_pulse_width_us": self._released_pulse_width_us,
            "visible_target_pulse_width_us": self._visible_target_pulse_width_us,
            "startup_rest_alignment_pending": self._startup_rest_alignment_pending,
            "startup_hold_until_armed": self._startup_hold_until_armed,
            "zero_reference_confirmed": self._zero_reference_confirmed,
            "continuous_estimated_heading_degrees": (
                None
                if self._continuous_planner is None
                else round(self._continuous_planner.estimated_heading_degrees, 3)
            ),
            "continuous_desired_heading_degrees": (
                None
                if self._continuous_planner is None
                else round(self._continuous_planner.desired_heading_degrees, 3)
            ),
            "fault_reason": self._fault_reason,
            "disabled_due_to_fault": self._disabled_due_to_fault,
            "config": {
                "control_mode": self.config.control_mode,
                "follow_exit_only": self.config.follow_exit_only,
                "visible_recenter_interval_s": round(self.config.visible_recenter_interval_s, 3),
                "visible_recenter_center_tolerance": round(
                    self.config.visible_recenter_center_tolerance,
                    4,
                ),
                "exit_visible_edge_threshold": round(self.config.exit_visible_edge_threshold, 4),
                "exit_visible_box_edge_threshold": round(self.config.exit_visible_box_edge_threshold, 4),
                "exit_activation_delay_s": round(self.config.exit_activation_delay_s, 3),
                "exit_reacquire_center_tolerance": round(self.config.exit_reacquire_center_tolerance, 4),
                "exit_cooldown_s": round(self.config.exit_cooldown_s, 3),
                "exit_follow_offset_limit": round(self.config.exit_follow_offset_limit, 4),
                "deadband": round(self.config.deadband, 4),
                "state_path": self.config.state_path,
            },
        }

    def _prime_last_physical_pulse_width_from_writer(self) -> None:
        """Seed the motion planner from the kernel's remembered pulse width on controller startup.

        The Twinr kernel writer persists the last pulse width even after the GPIO
        line is released. Reusing that value avoids a fresh controller assuming
        the servo is already centered and yanking loaded hardware back to rest.
        """

        if self._continuous_planner is not None:
            self._continuous_planner.reset(heading_degrees=self._continuous_startup_heading_degrees)
            self._last_physical_pulse_width_us = self.config.center_pulse_width_us
            self._released_pulse_width_us = self.config.center_pulse_width_us
            self._reset_motion_state(self.config.center_pulse_width_us)
            return
        if self.config.gpio is None:
            return
        current_pulse_reader = getattr(self._pulse_writer, "current_pulse_width_us", None)
        if not callable(current_pulse_reader):
            return
        try:
            pulse_width_us = current_pulse_reader(
                gpio_chip=self.config.gpio_chip,
                gpio=self.config.gpio,
            )
        except Exception:
            return
        if pulse_width_us is None:
            return
        checked_pulse_width_us = max(
            self.config.safe_min_pulse_width_us,
            min(self.config.safe_max_pulse_width_us, int(pulse_width_us)),
        )
        self._last_physical_pulse_width_us = checked_pulse_width_us
        self._released_pulse_width_us = checked_pulse_width_us
        self._reset_motion_state(checked_pulse_width_us)
        if checked_pulse_width_us != self.config.center_pulse_width_us:
            self._startup_rest_alignment_pending = True

    def _target_available(
        self,
        *,
        active: bool,
        target_center_x: float | None,
        confidence: float,
    ) -> bool:
        return (
            active
            and target_center_x is not None
            and confidence >= self.config.min_confidence
        )

    def _visible_target_present(
        self,
        *,
        active: bool,
        target_center_x: float | None,
    ) -> bool:
        return active and target_center_x is not None

    def _release_active_output(self, *, observed_at: float | None = None) -> None:
        if self._last_commanded_pulse_width_us is None:
            if self._continuous_planner is not None:
                self._continuous_planner.note_stopped(observed_at=observed_at)
                self._persist_runtime_state(observed_at=observed_at, force=True)
            return
        try:
            self._pulse_writer.disable(
                gpio_chip=self.config.gpio_chip,
                gpio=self.config.gpio if self.config.gpio is not None else 0,
            )
        except Exception as exc:
            self._fault_reason = f"{exc.__class__.__name__}: {exc}"
            raise
        if self._continuous_planner is not None:
            self._continuous_planner.note_stopped(observed_at=observed_at)
        self._released_pulse_width_us = self._last_commanded_pulse_width_us
        self._last_physical_pulse_width_us = self._last_commanded_pulse_width_us
        self._last_commanded_pulse_width_us = None
        self._reset_motion_state(self._released_pulse_width_us)
        self._persist_runtime_state(observed_at=observed_at, force=True)

    def _release_stale_output_if_disabled(self) -> None:
        """Best-effort release for outputs left active by an earlier process when this controller starts disabled."""

        if self.config.enabled or self.config.gpio is None:
            return
        try:
            self._pulse_writer.disable(
                gpio_chip=self.config.gpio_chip,
                gpio=self.config.gpio,
            )
        except Exception:
            return

    def _driver_supports_fault_recovery(self) -> bool:
        return str(self.config.driver or _DEFAULT_SERVO_DRIVER).strip().lower() in {
            "pololu_maestro",
            "peer_pololu_maestro",
        }

    def _maybe_recover_from_fault(self, *, observed_at: float | None) -> bool:
        if self._fault_reason is None:
            return True
        if not self._driver_supports_fault_recovery():
            return False
        retry_anchor_at = time.monotonic() if observed_at is None else observed_at
        if self._fault_retry_after_at is not None and retry_anchor_at < self._fault_retry_after_at:
            return False
        close_writer = getattr(self._pulse_writer, "close", None)
        if callable(close_writer):
            try:
                close_writer()
            except Exception:
                pass
        try:
            if self.config.gpio is not None:
                self._pulse_writer.probe(self.config.gpio)
            self._fault_reason = None
            self._disabled_due_to_fault = False
            self._fault_retry_after_at = None
            self._last_commanded_pulse_width_us = None
            self._last_target_center_x = None
            self._last_target_at = None
            self._last_target_velocity_x_per_s = 0.0
            self._smoothed_center_x = None
            if self._continuous_planner is not None:
                self._continuous_planner.reset(
                    heading_degrees=self._continuous_planner.estimated_heading_degrees,
                    observed_at=observed_at,
                )
                self._last_physical_pulse_width_us = self.config.center_pulse_width_us
                self._released_pulse_width_us = self.config.center_pulse_width_us
                self._reset_motion_state(self.config.center_pulse_width_us)
            else:
                self._prime_last_physical_pulse_width_from_writer()
            return True
        except Exception as exc:
            self._fault_reason = f"{exc.__class__.__name__}: {exc}"
            self._fault_retry_after_at = retry_anchor_at + _RECOVERABLE_SERVO_FAULT_RETRY_S
            return False

    def _update_visible_target_for_exit_only(
        self,
        *,
        observed_at: float | None,
        target_center_x: float,
        confidence: float,
    ) -> AttentionServoDecision:
        """Remember the visible target trajectory without physically following it."""

        normalized_center_x = _clamp_ratio(target_center_x)
        self._remember_visible_target(
            observed_at=observed_at,
            target_center_x=normalized_center_x,
        )
        self._update_target_velocity(
            observed_at=observed_at,
            target_center_x=normalized_center_x,
        )
        self._last_target_center_x = normalized_center_x
        self._last_target_at = observed_at
        self._last_exit_hold_center_x = None
        self._smoothed_center_x = normalized_center_x
        self._visible_target_pulse_width_us = None
        self._centered_since = None
        self._settled_since = None
        self._release_active_output(observed_at=observed_at)
        self._last_update_at = observed_at
        return AttentionServoDecision(
            observed_at=observed_at,
            active=False,
            reason="waiting_for_exit",
            confidence=confidence,
            target_center_x=target_center_x,
            applied_center_x=normalized_center_x,
            target_pulse_width_us=None,
            commanded_pulse_width_us=None,
        )

    def _handle_visible_target_during_startup_alignment(
        self,
        *,
        observed_at: float | None,
        confidence: float,
        target_center_x: float | None,
        normalized_target_center_x: float,
    ) -> AttentionServoDecision:
        """Return stale startup alignment to neutral before enabling exit-only waiting semantics.

        The kernel writer remembers the last pulse width after release. If the
        runtime restarts while the servo is still off-center from an older exit
        move, entering `waiting_for_exit` immediately would treat that stale
        physical pose as neutral and later produce only a tiny nudge on loss.
        During this bounded one-time alignment phase we still remember the
        visible target history, but physically return to center first.
        """

        self._remember_visible_target(
            observed_at=observed_at,
            target_center_x=normalized_target_center_x,
        )
        self._update_target_velocity(
            observed_at=observed_at,
            target_center_x=normalized_target_center_x,
        )
        self._last_target_center_x = normalized_target_center_x
        self._last_target_at = observed_at
        self._smoothed_center_x = normalized_target_center_x
        rest_decision = self._drive_rest_position(
            observed_at=observed_at,
            confidence=confidence,
            target_center_x=target_center_x,
        )
        if rest_decision.reason == "idle_released":
            self._startup_rest_alignment_pending = False
        if rest_decision.reason == "recentering":
            return AttentionServoDecision(
                observed_at=rest_decision.observed_at,
                active=False,
                reason="startup_recentering",
                confidence=rest_decision.confidence,
                target_center_x=rest_decision.target_center_x,
                applied_center_x=rest_decision.applied_center_x,
                target_pulse_width_us=rest_decision.target_pulse_width_us,
                commanded_pulse_width_us=rest_decision.commanded_pulse_width_us,
            )
        return rest_decision

    def _remember_visible_target(
        self,
        *,
        observed_at: float | None,
        target_center_x: float,
    ) -> None:
        """Keep a short visible history so exit-follow prefers the outward edge over brief inward tracker wobble."""

        if observed_at is None:
            return
        checked_at = float(observed_at)
        history_window_s = max(
            self.config.target_hold_s,
            self.config.loss_extrapolation_s,
            self.config.exit_activation_delay_s,
        )
        cutoff_at = checked_at - max(0.0, history_window_s)
        while self._recent_visible_targets and self._recent_visible_targets[0][0] < cutoff_at:
            self._recent_visible_targets.popleft()
        self._recent_visible_targets.append((checked_at, _clamp_ratio(target_center_x)))

    def _remember_visible_target_box(
        self,
        *,
        observed_at: float | None,
        box_left: float | None,
        box_right: float | None,
    ) -> None:
        """Keep a short visible-box history so edge pursuit keys off real frame boundaries."""

        if observed_at is None or box_left is None or box_right is None:
            return
        checked_at = float(observed_at)
        normalized_left = _clamp_ratio(float(box_left))
        normalized_right = _clamp_ratio(float(box_right))
        if normalized_right < normalized_left:
            normalized_left, normalized_right = normalized_right, normalized_left
        history_window_s = max(
            self.config.target_hold_s,
            self.config.loss_extrapolation_s,
            self.config.exit_activation_delay_s,
        )
        cutoff_at = checked_at - max(0.0, history_window_s)
        while self._recent_visible_target_boxes and self._recent_visible_target_boxes[0][0] < cutoff_at:
            self._recent_visible_target_boxes.popleft()
        self._recent_visible_target_boxes.append((checked_at, normalized_left, normalized_right))

    def _visible_edge_departure_confirmed(
        self,
        *,
        observed_at: float | None,
        target_center_x: float,
    ) -> bool:
        """Return whether one visible side-departure should trigger monotone pursuit.

        The HDMI fast path can keep a user weakly visible as a small edge anchor
        while they are already leaving the frame, and the attention target can
        briefly snap inward again before the exit delay elapses. In exit-only
        mode that should still unlock the same one-direction pursuit after a
        short confirmation delay instead of blocking forever on a broad
        `person_visible=true` or a single centered wobble.
        """

        if observed_at is None:
            self._visible_edge_departure_since_at = None
            return False
        checked_at = float(observed_at)
        candidate_center_x = self._recent_exit_anchor_center_x(observed_at=checked_at)
        if candidate_center_x is None:
            candidate_center_x = float(target_center_x)
        exit_side = self._exit_side_for_center_x(candidate_center_x)
        edge_threshold = self.config.exit_visible_edge_threshold
        if edge_threshold <= 0.5:
            self._visible_edge_departure_since_at = None
            return False
        distance_from_center = abs(candidate_center_x - 0.5)
        required_distance = max(0.0, edge_threshold - 0.5)
        if distance_from_center < required_distance:
            self._visible_edge_departure_since_at = None
            return False
        edge_position = self._recent_exit_anchor_box_edge(observed_at=checked_at)
        if edge_position is not None and exit_side != 0:
            box_edge_threshold = self.config.exit_visible_box_edge_threshold
            if exit_side > 0 and edge_position < box_edge_threshold:
                self._visible_edge_departure_since_at = None
                return False
            if exit_side < 0 and edge_position > (1.0 - box_edge_threshold):
                self._visible_edge_departure_since_at = None
                return False
        if self._visible_edge_departure_since_at is None:
            self._visible_edge_departure_since_at = checked_at
            return False
        return (checked_at - self._visible_edge_departure_since_at) >= self.config.exit_activation_delay_s

    def _target_is_exit_reacquired_centered(self, target_center_x: float) -> bool:
        """Return whether one visible target is centered enough to end exit pursuit."""

        return abs(float(target_center_x) - 0.5) <= self.config.exit_reacquire_center_tolerance

    def _target_is_visible_recenter_centered(self, target_center_x: float) -> bool:
        """Return whether one visible target is close enough to center to skip periodic recentering."""

        return abs(float(target_center_x) - 0.5) <= self.config.visible_recenter_center_tolerance

    def _clear_visible_recenter(self, *, reset_timer: bool) -> None:
        self._visible_recenter_target_pulse_width_us = None
        self._visible_recenter_center_x = None
        self._visible_recenter_settled_at = None
        if reset_timer:
            self._visible_recenter_since_at = None

    def _should_begin_visible_recenter(self, *, observed_at: float | None) -> bool:
        if self._visible_recenter_target_pulse_width_us is not None:
            return True
        checked_at = time.monotonic() if observed_at is None else float(observed_at)
        interval_s = max(0.0, self.config.visible_recenter_interval_s)
        if interval_s <= 0.0:
            self._visible_recenter_since_at = checked_at
            return True
        since_at = self._visible_recenter_since_at
        if since_at is None or checked_at < since_at:
            self._visible_recenter_since_at = checked_at
            return False
        return (checked_at - since_at) >= interval_s

    def _begin_visible_recenter(self, *, normalized_target_center_x: float) -> bool:
        if self._target_is_visible_recenter_centered(normalized_target_center_x):
            return False
        self._visible_recenter_center_x = normalized_target_center_x
        self._visible_recenter_target_pulse_width_us = self._target_pulse_width_for_center_x(
            center_x=normalized_target_center_x,
            observed_at=self._last_update_at,
        )
        self._visible_recenter_settled_at = None
        return True

    def _recent_exit_anchor_center_x(self, *, observed_at: float | None) -> float | None:
        """Return the furthest recent visible point on the current exit side."""

        last_target_center_x = self._last_target_center_x
        if last_target_center_x is None:
            return None
        checked_at = None if observed_at is None else float(observed_at)
        history_window_s = max(
            self.config.target_hold_s,
            self.config.loss_extrapolation_s,
            self.config.exit_activation_delay_s,
        )
        if checked_at is not None:
            cutoff_at = checked_at - max(0.0, history_window_s)
            while self._recent_visible_targets and self._recent_visible_targets[0][0] < cutoff_at:
                self._recent_visible_targets.popleft()
        exit_side = self._current_exit_side(last_target_center_x)
        if exit_side == 0:
            return last_target_center_x
        anchor_center_x = last_target_center_x
        anchor_offset = abs(anchor_center_x - 0.5)
        for _, sample_center_x in self._recent_visible_targets:
            sample_offset = sample_center_x - 0.5
            if exit_side > 0 and sample_offset <= 0.0:
                continue
            if exit_side < 0 and sample_offset >= 0.0:
                continue
            sample_distance = abs(sample_offset)
            if sample_distance > anchor_offset:
                anchor_center_x = sample_center_x
                anchor_offset = sample_distance
        return anchor_center_x

    def _recent_exit_anchor_box_edge(self, *, observed_at: float | None) -> float | None:
        """Return the furthest recent visible box edge on the current exit side."""

        last_target_center_x = self._last_target_center_x
        if last_target_center_x is None:
            return None
        checked_at = None if observed_at is None else float(observed_at)
        history_window_s = max(
            self.config.target_hold_s,
            self.config.loss_extrapolation_s,
            self.config.exit_activation_delay_s,
        )
        if checked_at is not None:
            cutoff_at = checked_at - max(0.0, history_window_s)
            while self._recent_visible_target_boxes and self._recent_visible_target_boxes[0][0] < cutoff_at:
                self._recent_visible_target_boxes.popleft()
        exit_side = self._current_exit_side(last_target_center_x)
        if exit_side == 0:
            return None
        anchor_edge = None
        for _, sample_left, sample_right in self._recent_visible_target_boxes:
            sample_center_x = (sample_left + sample_right) * 0.5
            if exit_side > 0:
                if sample_center_x <= 0.5:
                    continue
                candidate_edge = sample_right
                if anchor_edge is None or candidate_edge > anchor_edge:
                    anchor_edge = candidate_edge
            else:
                if sample_center_x >= 0.5:
                    continue
                candidate_edge = sample_left
                if anchor_edge is None or candidate_edge < anchor_edge:
                    anchor_edge = candidate_edge
        return anchor_edge

    def _current_exit_side(self, last_target_center_x: float) -> int:
        """Return the most plausible exit side from recent target position and velocity."""

        last_side = self._exit_side_for_center_x(last_target_center_x)
        velocity_side = 0
        if self._last_target_velocity_x_per_s > 1e-4:
            velocity_side = 1
        elif self._last_target_velocity_x_per_s < -1e-4:
            velocity_side = -1
        exit_side = velocity_side or last_side
        if last_side != 0 and velocity_side != 0 and velocity_side != last_side:
            exit_side = last_side
        return exit_side

    def _exit_side_for_center_x(self, center_x: float | None) -> int:
        """Return the horizontal side token for one normalized target center."""

        if center_x is None:
            return 0
        offset = float(center_x) - 0.5
        if offset > 0.0:
            return 1
        if offset < 0.0:
            return -1
        return 0

    def _await_exit_confirmation(
        self,
        *,
        observed_at: float | None,
        confidence: float,
        target_center_x: float | None,
        applied_center_x: float,
    ) -> AttentionServoDecision:
        """Keep the servo still until visibility loss persists long enough."""

        self._visible_target_pulse_width_us = None
        self._centered_since = None
        self._settled_since = None
        self._release_active_output(observed_at=observed_at)
        self._last_update_at = observed_at
        return AttentionServoDecision(
            observed_at=observed_at,
            active=False,
            reason="awaiting_exit_confirmation",
            confidence=confidence,
            target_center_x=target_center_x,
            applied_center_x=applied_center_x,
            target_pulse_width_us=None,
            commanded_pulse_width_us=None,
        )

    def _resolve_target(
        self,
        *,
        observed_at: float | None,
        active: bool,
        target_center_x: float | None,
        confidence: float,
    ) -> tuple[float, bool, str]:
        checked_at = None if observed_at is None else float(observed_at)
        target_available = self._target_available(
            active=active,
            target_center_x=target_center_x,
            confidence=confidence,
        )
        if target_available:
            if target_center_x is None:
                raise RuntimeError("Target availability requires a concrete normalized target center")
            normalized_center_x = _clamp_ratio(float(target_center_x))
            self._update_target_velocity(
                observed_at=checked_at,
                target_center_x=normalized_center_x,
            )
            self._last_target_center_x = normalized_center_x
            self._last_target_at = checked_at
            self._last_exit_hold_center_x = None
            return normalized_center_x, True, "following_target"

        if (
            self._last_target_center_x is not None
            and self._last_target_at is not None
            and checked_at is not None
            and (checked_at - self._last_target_at) <= self.config.target_hold_s
        ):
            if self.config.follow_exit_only:
                loss_elapsed_s = max(0.0, checked_at - self._last_target_at)
                if loss_elapsed_s < self.config.exit_activation_delay_s:
                    self._last_exit_hold_center_x = None
                    return self._last_target_center_x, False, "awaiting_exit_confirmation"
            projected_center_x, projected_reason = self._projected_target_center_x(
                observed_at=checked_at,
                loss_activation_delay_s=(
                    self.config.exit_activation_delay_s
                    if self.config.follow_exit_only
                    else 0.0
                ),
            )
            if self.config.follow_exit_only:
                self._last_exit_hold_center_x = projected_center_x
            return projected_center_x, True, projected_reason

        if self.config.follow_exit_only and self._last_exit_hold_center_x is not None:
            return self._last_exit_hold_center_x, False, "holding_exit_position"

        self._last_target_center_x = None
        self._last_target_at = None
        self._last_target_velocity_x_per_s = 0.0
        self._last_exit_hold_center_x = None
        return 0.5, False, "recentering"

    def _update_target_velocity(self, *, observed_at: float | None, target_center_x: float) -> None:
        previous_center_x = self._last_target_center_x
        previous_at = self._last_target_at
        if (
            observed_at is None
            or previous_center_x is None
            or previous_at is None
            or observed_at <= previous_at
        ):
            self._last_target_velocity_x_per_s = 0.0
            return
        dt = max(0.001, observed_at - previous_at)
        observed_velocity_x_per_s = (target_center_x - previous_center_x) / dt
        observed_velocity_x_per_s = _clamp(
            observed_velocity_x_per_s,
            minimum=-2.0,
            maximum=2.0,
        )
        self._last_target_velocity_x_per_s = (
            (self._last_target_velocity_x_per_s * 0.6)
            + (observed_velocity_x_per_s * 0.4)
        )

    def _projected_target_center_x(
        self,
        *,
        observed_at: float,
        loss_activation_delay_s: float = 0.0,
    ) -> tuple[float, str]:
        seed_center_x = self._recent_exit_anchor_center_x(observed_at=observed_at)
        if seed_center_x is None or self._last_target_at is None:
            return 0.5, "recentering"
        raw_elapsed_s = max(0.0, observed_at - self._last_target_at)
        elapsed_s = max(0.0, raw_elapsed_s - max(0.0, loss_activation_delay_s))
        projection_window_s = min(
            max(0.0, self.config.target_hold_s - max(0.0, loss_activation_delay_s)),
            self.config.loss_extrapolation_s,
        )
        if projection_window_s <= 0.0:
            return seed_center_x, "holding_recent_target"
        effective_elapsed_s = min(elapsed_s, projection_window_s)
        decay_progress = effective_elapsed_s / projection_window_s
        seed_offset = seed_center_x - 0.5
        seed_side = 0
        if seed_offset > 0.0:
            seed_side = 1
        elif seed_offset < 0.0:
            seed_side = -1
        velocity_x_per_s = self._last_target_velocity_x_per_s
        if seed_side > 0 and velocity_x_per_s < 0.0:
            velocity_x_per_s = 0.0
        elif seed_side < 0 and velocity_x_per_s > 0.0:
            velocity_x_per_s = 0.0
        projected_offset = (
            velocity_x_per_s
            * self.config.loss_extrapolation_gain
            * effective_elapsed_s
            * max(0.0, 1.0 - (0.5 * decay_progress))
        )
        projected_center_x = _clamp_ratio(seed_center_x + projected_offset)
        if elapsed_s <= projection_window_s:
            return projected_center_x, "projecting_recent_trajectory"
        return projected_center_x, "holding_projected_trajectory"

    def _clamped_follow_center_x(self, center_x: float) -> float:
        """Clamp exit-only follow targets to the configured off-center angle."""

        normalized_center_x = _clamp_ratio(center_x)
        if not self.config.follow_exit_only:
            return normalized_center_x
        normalized_offset = (normalized_center_x - 0.5) * 2.0
        limited_offset = _clamp(
            normalized_offset,
            minimum=-self.config.exit_follow_offset_limit,
            maximum=self.config.exit_follow_offset_limit,
        )
        return _clamp_ratio(0.5 + (limited_offset * 0.5))

    def _maximum_target_heading_degrees(self) -> float:
        if self.config.follow_exit_only:
            return max(1.0, float(self.config.exit_follow_max_degrees))
        return max(1.0, float(self.config.mechanical_range_degrees) * 0.5)

    def _desired_heading_degrees_for_center_x(self, center_x: float) -> float:
        normalized_center_x = self._clamped_follow_center_x(center_x)
        normalized_offset = (normalized_center_x - 0.5) * 2.0
        if self.config.invert_direction:
            normalized_offset *= -1.0
        if abs(normalized_offset) <= self.config.deadband:
            normalized_offset = 0.0
        return normalized_offset * self._maximum_target_heading_degrees()

    def _target_pulse_width_for_center_x(self, *, center_x: float, observed_at: float | None) -> int:
        if self._continuous_planner is None:
            return self._pulse_width_for_center_x(center_x)
        return self._continuous_planner.target_pulse_width_for_heading(
            self._desired_heading_degrees_for_center_x(center_x),
            observed_at=observed_at,
        )

    def _pulse_width_for_center_x(self, center_x: float) -> int:
        normalized_center_x = self._clamped_follow_center_x(center_x)
        normalized_offset = (normalized_center_x - 0.5) * 2.0
        if self.config.invert_direction:
            normalized_offset *= -1.0
        if abs(normalized_offset) <= self.config.deadband:
            normalized_offset = 0.0
        if normalized_offset >= 0.0:
            span = self.config.safe_max_pulse_width_us - self.config.center_pulse_width_us
        else:
            span = self.config.center_pulse_width_us - self.config.safe_min_pulse_width_us
        pulse_width = int(round(self.config.center_pulse_width_us + (normalized_offset * span)))
        return max(self.config.safe_min_pulse_width_us, min(self.config.safe_max_pulse_width_us, pulse_width))

    def _effective_dt_s(self, *, observed_at: float | None) -> float:
        previous = self._last_update_at
        if observed_at is None or previous is None or observed_at <= previous:
            return self.config.reference_interval_s
        return max(0.001, min(observed_at - previous, self.config.reference_interval_s))

    def _stabilize_visible_target_pulse_width(self, *, target_pulse_width_us: int, reason: str) -> int:
        """Latch visible targets so servo follow ignores millimeter re-justification."""

        checked_target_us = max(
            self.config.safe_min_pulse_width_us,
            min(self.config.safe_max_pulse_width_us, int(target_pulse_width_us)),
        )
        if reason != "following_target":
            self._visible_target_pulse_width_us = None
            return checked_target_us
        tolerance_us = max(0, self.config.visible_retarget_tolerance_us)
        latched_target_us = self._visible_target_pulse_width_us
        if tolerance_us <= 0 or latched_target_us is None:
            self._visible_target_pulse_width_us = checked_target_us
            return checked_target_us
        if abs(checked_target_us - latched_target_us) <= tolerance_us:
            return latched_target_us
        self._visible_target_pulse_width_us = checked_target_us
        return checked_target_us

    def _smoothed_target_center_x(
        self,
        *,
        observed_at: float | None,
        target_center_x: float,
    ) -> float:
        normalized_target = _clamp_ratio(target_center_x)
        previous = 0.5 if self._smoothed_center_x is None else self._smoothed_center_x
        if self.config.target_smoothing_s <= 0.0:
            smoothed = normalized_target
        else:
            dt = self._effective_dt_s(observed_at=observed_at)
            alpha = max(0.0, min(1.0, dt / max(dt, self.config.target_smoothing_s)))
            smoothed = previous + ((normalized_target - previous) * alpha)
        bounded_smoothed = _clamp_ratio(smoothed)
        self._smoothed_center_x = bounded_smoothed
        return bounded_smoothed

    def _reset_motion_state(self, pulse_width_us: int | None = None) -> None:
        seeded_pulse_width = (
            None
            if pulse_width_us is None
            else float(
                max(
                    self.config.safe_min_pulse_width_us,
                    min(self.config.safe_max_pulse_width_us, int(pulse_width_us)),
                )
            )
        )
        self._planned_pulse_width_us = seeded_pulse_width
        self._planned_velocity_us_per_s = 0.0
        self._planned_acceleration_us_per_s2 = 0.0

    def _release_tolerance_us(self) -> int:
        return max(
            _MIN_RELEASE_TOLERANCE_US,
            min(_MAX_RELEASE_TOLERANCE_US, self.config.max_step_us * 2),
        )

    def _seeded_planned_pulse_width_us(self) -> float:
        if self._planned_pulse_width_us is not None:
            return self._planned_pulse_width_us
        if self._last_commanded_pulse_width_us is not None:
            return float(self._last_commanded_pulse_width_us)
        if self._last_physical_pulse_width_us is not None:
            return float(self._last_physical_pulse_width_us)
        if self._released_pulse_width_us is not None:
            return float(self._released_pulse_width_us)
        return float(self.config.center_pulse_width_us)

    def _advance_planned_pulse_width(
        self,
        target_pulse_width_us: int,
        *,
        observed_at: float | None,
        motion_profile: str,
    ) -> float:
        current_pulse_width_us = self._seeded_planned_pulse_width_us()
        bounded_target_us = float(
            max(
                self.config.safe_min_pulse_width_us,
                min(self.config.safe_max_pulse_width_us, int(target_pulse_width_us)),
            )
        )
        error_us = bounded_target_us - current_pulse_width_us
        if abs(error_us) <= 0.5:
            self._planned_pulse_width_us = bounded_target_us
            self._planned_velocity_us_per_s = 0.0
            self._planned_acceleration_us_per_s2 = 0.0
            return bounded_target_us

        dt = self._effective_dt_s(observed_at=observed_at)
        (
            max_velocity_us_per_s,
            max_acceleration_us_per_s2,
            max_jerk_us_per_s3,
        ) = self._motion_limits_for_profile(motion_profile)
        stopping_velocity_us_per_s = math.sqrt(max(0.0, 2.0 * max_acceleration_us_per_s2 * abs(error_us)))
        desired_velocity_us_per_s = math.copysign(
            min(max_velocity_us_per_s, stopping_velocity_us_per_s),
            error_us,
        )
        desired_acceleration_us_per_s2 = (desired_velocity_us_per_s - self._planned_velocity_us_per_s) / dt
        desired_acceleration_us_per_s2 = _clamp(
            desired_acceleration_us_per_s2,
            minimum=-max_acceleration_us_per_s2,
            maximum=max_acceleration_us_per_s2,
        )
        max_acceleration_delta_us_per_s2 = max_jerk_us_per_s3 * dt
        next_acceleration_us_per_s2 = self._planned_acceleration_us_per_s2 + _clamp(
            desired_acceleration_us_per_s2 - self._planned_acceleration_us_per_s2,
            minimum=-max_acceleration_delta_us_per_s2,
            maximum=max_acceleration_delta_us_per_s2,
        )
        next_acceleration_us_per_s2 = _clamp(
            next_acceleration_us_per_s2,
            minimum=-max_acceleration_us_per_s2,
            maximum=max_acceleration_us_per_s2,
        )
        next_velocity_us_per_s = self._planned_velocity_us_per_s + (next_acceleration_us_per_s2 * dt)
        next_velocity_us_per_s = _clamp(
            next_velocity_us_per_s,
            minimum=-max_velocity_us_per_s,
            maximum=max_velocity_us_per_s,
        )
        next_pulse_width_us = current_pulse_width_us + (
            (self._planned_velocity_us_per_s + next_velocity_us_per_s) * 0.5 * dt
        )
        if (error_us > 0.0 and next_pulse_width_us >= bounded_target_us) or (
            error_us < 0.0 and next_pulse_width_us <= bounded_target_us
        ):
            next_pulse_width_us = bounded_target_us
            next_velocity_us_per_s = 0.0
            next_acceleration_us_per_s2 = 0.0
        next_pulse_width_us = _clamp(
            next_pulse_width_us,
            minimum=float(self.config.safe_min_pulse_width_us),
            maximum=float(self.config.safe_max_pulse_width_us),
        )
        self._planned_pulse_width_us = next_pulse_width_us
        self._planned_velocity_us_per_s = next_velocity_us_per_s
        self._planned_acceleration_us_per_s2 = next_acceleration_us_per_s2
        return next_pulse_width_us

    def _motion_limits_for_profile(self, motion_profile: str) -> tuple[float, float, float]:
        normalized_profile = str(motion_profile or "tracking").strip().lower() or "tracking"
        if normalized_profile == "rest":
            return (
                max(1.0, self.config.rest_max_velocity_us_per_s),
                max(1.0, self.config.rest_max_acceleration_us_per_s2),
                max(1.0, self.config.rest_max_jerk_us_per_s3),
            )
        return (
            max(1.0, self.config.max_velocity_us_per_s),
            max(1.0, self.config.max_acceleration_us_per_s2),
            max(1.0, self.config.max_jerk_us_per_s3),
        )

    def _command_pulse_width_for_plan(
        self,
        planned_pulse_width_us: float,
        *,
        target_pulse_width_us: int,
        motion_profile: str,
    ) -> int:
        checked_target_us = max(
            self.config.safe_min_pulse_width_us,
            min(self.config.safe_max_pulse_width_us, int(target_pulse_width_us)),
        )
        candidate_pulse_width_us = int(round(planned_pulse_width_us))
        previous_commanded_pulse_width_us = (
            self._last_commanded_pulse_width_us
            if self._last_commanded_pulse_width_us is not None
            else (
                self._last_physical_pulse_width_us
                if self._last_physical_pulse_width_us is not None
                else self.config.center_pulse_width_us
            )
        )
        if (
            str(motion_profile or "").strip().lower() == "rest"
            and checked_target_us == self.config.center_pulse_width_us
            and abs(checked_target_us - previous_commanded_pulse_width_us) <= self._release_tolerance_us()
        ):
            return checked_target_us

        command_delta_us = candidate_pulse_width_us - previous_commanded_pulse_width_us
        if (
            self._last_commanded_pulse_width_us is not None
            and abs(command_delta_us) < self.config.min_command_delta_us
            and abs(checked_target_us - previous_commanded_pulse_width_us) >= self.config.min_command_delta_us
        ):
            return previous_commanded_pulse_width_us
        if abs(command_delta_us) > self.config.max_step_us:
            candidate_pulse_width_us = previous_commanded_pulse_width_us + (
                self.config.max_step_us if command_delta_us > 0 else -self.config.max_step_us
            )
        if (
            abs(command_delta_us) < self.config.min_command_delta_us
            and self._last_commanded_pulse_width_us is not None
        ):
            return previous_commanded_pulse_width_us
        return max(
            self.config.safe_min_pulse_width_us,
            min(self.config.safe_max_pulse_width_us, candidate_pulse_width_us),
        )

    def _maybe_release_idle(
        self,
        *,
        observed_at: float | None,
        active: bool,
        target_pulse_width_us: int,
    ) -> bool:
        """Release the output after calm recentering so idle servos do not buzz."""

        if active or self.config.idle_release_s <= 0.0:
            self._centered_since = None
            return False
        tolerance_us = self._release_tolerance_us()
        if abs(target_pulse_width_us - self.config.center_pulse_width_us) > tolerance_us:
            self._centered_since = None
            return False
        current_pulse_width_us = int(round(self._seeded_planned_pulse_width_us()))
        if current_pulse_width_us != self.config.center_pulse_width_us:
            self._centered_since = None
            return False
        if observed_at is None:
            return False
        if self._centered_since is None or observed_at < self._centered_since:
            self._centered_since = observed_at
            return False
        if (observed_at - self._centered_since) < self.config.idle_release_s:
            return False
        if self._last_commanded_pulse_width_us is None:
            return True
        try:
            self._pulse_writer.disable(
                gpio_chip=self.config.gpio_chip,
                gpio=self.config.gpio if self.config.gpio is not None else 0,
            )
        except Exception as exc:
            self._fault_reason = f"{exc.__class__.__name__}: {exc}"
            raise
        if self._continuous_planner is not None:
            self._continuous_planner.note_stopped(observed_at=observed_at)
        self._last_commanded_pulse_width_us = None
        self._last_physical_pulse_width_us = self.config.center_pulse_width_us
        self._reset_motion_state(self.config.center_pulse_width_us)
        self._persist_runtime_state(observed_at=observed_at, force=True)
        return True

    def _maybe_hold_released_target(
        self,
        *,
        observed_at: float | None,
        active: bool,
        reason: str,
        confidence: float,
        target_center_x: float | None,
        applied_center_x: float,
        target_pulse_width_us: int,
    ) -> AttentionServoDecision | None:
        released_pulse_width_us = self._released_pulse_width_us
        if released_pulse_width_us is None:
            return None
        if target_pulse_width_us == self.config.center_pulse_width_us:
            if released_pulse_width_us != self.config.center_pulse_width_us:
                self._reset_motion_state(released_pulse_width_us)
                self._released_pulse_width_us = None
                self._settled_since = None
                return None
        tolerance_us = self._release_tolerance_us()
        if abs(target_pulse_width_us - released_pulse_width_us) > tolerance_us:
            self._reset_motion_state(released_pulse_width_us)
            self._released_pulse_width_us = None
            self._settled_since = None
            return None
        self._settled_since = observed_at
        self._last_update_at = observed_at
        return AttentionServoDecision(
            observed_at=observed_at,
            active=active,
            reason="settled_released",
            confidence=confidence,
            target_center_x=target_center_x,
            applied_center_x=applied_center_x,
            target_pulse_width_us=target_pulse_width_us,
            commanded_pulse_width_us=None,
        )

    def _maybe_release_exit_hold(
        self,
        *,
        observed_at: float | None,
        reason: str,
        confidence: float,
        target_center_x: float | None,
        applied_center_x: float,
        target_pulse_width_us: int,
        commanded_pulse_width_us: int,
    ) -> AttentionServoDecision | None:
        """Release the servo after it reaches the projected exit endpoint."""

        if (
            self._continuous_planner is not None
            or not self.config.follow_exit_only
            or reason not in {"holding_projected_trajectory", "holding_exit_position"}
            or self._last_commanded_pulse_width_us is None
        ):
            return None
        tolerance_us = self._release_tolerance_us()
        if abs(target_pulse_width_us - commanded_pulse_width_us) > tolerance_us:
            return None
        try:
            self._pulse_writer.disable(
                gpio_chip=self.config.gpio_chip,
                gpio=self.config.gpio if self.config.gpio is not None else 0,
            )
        except Exception as exc:
            self._fault_reason = f"{exc.__class__.__name__}: {exc}"
            raise
        if self._continuous_planner is not None:
            self._continuous_planner.note_stopped(observed_at=observed_at)
        self._released_pulse_width_us = commanded_pulse_width_us
        self._last_commanded_pulse_width_us = None
        self._last_physical_pulse_width_us = commanded_pulse_width_us
        self._reset_motion_state(commanded_pulse_width_us)
        self._settled_since = None
        self._persist_runtime_state(observed_at=observed_at, force=True)
        return AttentionServoDecision(
            observed_at=observed_at,
            active=False,
            reason="exit_hold_released",
            confidence=confidence,
            target_center_x=target_center_x,
            applied_center_x=applied_center_x,
            target_pulse_width_us=target_pulse_width_us,
            commanded_pulse_width_us=None,
        )

    def _maybe_release_settled_target(
        self,
        *,
        observed_at: float | None,
        target_pulse_width_us: int,
        commanded_pulse_width_us: int,
    ) -> bool:
        """Release a stable off-center target so loaded servos can relax quietly."""

        if (
            self._continuous_planner is not None
            or observed_at is None
            or self.config.settled_release_s <= 0.0
            or self._last_commanded_pulse_width_us is None
        ):
            self._settled_since = None
            return False
        tolerance_us = self._release_tolerance_us()
        if abs(target_pulse_width_us - self.config.center_pulse_width_us) <= tolerance_us:
            self._settled_since = None
            return False
        if (
            abs(target_pulse_width_us - commanded_pulse_width_us) > tolerance_us
            or commanded_pulse_width_us != self._last_commanded_pulse_width_us
        ):
            self._settled_since = None
            return False
        if self._settled_since is None or observed_at < self._settled_since:
            self._settled_since = observed_at
            return False
        if (observed_at - self._settled_since) < self.config.settled_release_s:
            return False
        try:
            self._pulse_writer.disable(
                gpio_chip=self.config.gpio_chip,
                gpio=self.config.gpio if self.config.gpio is not None else 0,
            )
        except Exception as exc:
            self._fault_reason = f"{exc.__class__.__name__}: {exc}"
            raise
        if self._continuous_planner is not None:
            self._continuous_planner.note_stopped(observed_at=observed_at)
        self._released_pulse_width_us = commanded_pulse_width_us
        self._last_commanded_pulse_width_us = None
        self._last_physical_pulse_width_us = commanded_pulse_width_us
        self._reset_motion_state(commanded_pulse_width_us)
        self._settled_since = None
        self._persist_runtime_state(observed_at=observed_at, force=True)
        return True


__all__ = [
    "AttentionServoConfig",
    "AttentionServoController",
    "AttentionServoDecision",
    "LGPIOPWMServoPulseWriter",
    "LGPIOServoPulseWriter",
    "PigpioServoPulseWriter",
    "SysfsPWMServoPulseWriter",
    "TwinrKernelServoPulseWriter",
]
