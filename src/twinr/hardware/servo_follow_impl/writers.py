"""Implement servo pulse writers and driver selection for Twinr."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Callable, Protocol, cast

from twinr.runtime_paths import prime_raspberry_pi_system_site_packages
from twinr.hardware.servo_maestro import PololuMaestroServoPulseWriter
from twinr.hardware.servo_peer import PeerPololuMaestroServoPulseWriter

from .constants import (
    _DEFAULT_SERVO_DRIVER,
    _DEFAULT_SERVO_FREQUENCY_HZ,
)

from .common import _normalize_chip_index
from .preflight import _assert_servo_gpio_environment_ready

if TYPE_CHECKING:
    from .config import AttentionServoConfig


def _public_servo_follow_symbol(name: str, default: object) -> object:
    """Resolve wrapper-level patch points before falling back to internal symbols."""

    public_module = sys.modules.get("twinr.hardware.servo_follow")
    if public_module is None:
        return default
    return getattr(public_module, name, default)

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
            prime_raspberry_pi_system_site_packages()
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
            prime_raspberry_pi_system_site_packages()
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
        writer_class = cast(
            Callable[..., ServoPulseWriter],
            _public_servo_follow_symbol(
                "PololuMaestroServoPulseWriter",
                PololuMaestroServoPulseWriter,
            ),
        )
        return writer_class(device_path=config.maestro_device)
    if driver == "peer_pololu_maestro":
        writer_class = cast(
            Callable[..., ServoPulseWriter],
            _public_servo_follow_symbol(
                "PeerPololuMaestroServoPulseWriter",
                PeerPololuMaestroServoPulseWriter,
            ),
        )
        return writer_class(
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
