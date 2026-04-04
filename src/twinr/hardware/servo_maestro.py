"""Pololu Maestro command-port adapter for Twinr attention-servo output.

Twinr's higher servo controller works in pulse widths already, so this adapter
only translates those microsecond targets into the Maestro compact serial
protocol on the USB command port. It intentionally stays small and focused:
device discovery, bounded command I/O, and current-position reads for startup
alignment.

Important operational invariant: the Maestro itself must be configured for
``USB_DUAL_PORT`` or ``USB_CHAINED``. In the factory-default UART modes the
virtual USB command port does not talk to the Maestro command processor, so
serial writes appear to succeed while the servo never receives a target.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
import select
import termios


_DEFAULT_READ_TIMEOUT_S = 0.25
_MAESTRO_AUTODETECT_BAUD_BYTE = 0xAA
_MAESTRO_COMMAND_SET_TARGET = 0x84
_MAESTRO_COMMAND_SET_SPEED = 0x87
_MAESTRO_COMMAND_SET_ACCELERATION = 0x89
_MAESTRO_COMMAND_GET_POSITION = 0x90
_MAESTRO_MAX_CHANNEL = 23
_MAESTRO_TARGET_SCALE = 4
_MAESTRO_MAX_TARGET_QUARTER_US = 0x3FFF
_MAESTRO_MAX_SPEED = 3968
_MAESTRO_MAX_ACCELERATION = 255
_MAESTRO_SPEED_UNIT_US_PER_S = 25.0
_MAESTRO_ACCELERATION_UNIT_US_PER_S2 = 312.5
_POLOLU_VENDOR_ID = "1ffb"
_MAESTRO_PRODUCT_IDS = frozenset(
    {
        "0089",  # Micro Maestro 6
        "008a",  # Mini Maestro 12
        "008b",  # Mini Maestro 18
        "008c",  # Mini Maestro 24
    }
)
_AUTODETECT_SERIAL_BY_ID_PATTERNS = (
    "/dev/serial/by-id/*Pololu*Maestro*if00*",
    "/dev/serial/by-id/*Pololu*Mini_Maestro*if00*",
)


def _validate_channel(channel: int) -> int:
    checked_channel = int(channel)
    if checked_channel < 0 or checked_channel > _MAESTRO_MAX_CHANNEL:
        raise RuntimeError(
            f"Pololu Maestro attention-servo output requires channel 0-{_MAESTRO_MAX_CHANNEL}, got {checked_channel}"
        )
    return checked_channel


def _read_optional_sysfs_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip().lower() or None
    except OSError:
        return None


def _find_parent_sysfs_value(start_path: Path, filename: str) -> str | None:
    for candidate in (start_path, *start_path.parents):
        value = _read_optional_sysfs_text(candidate / filename)
        if value is not None:
            return value
    return None


def _tty_acm_sort_key(device_path: str) -> tuple[int, str]:
    name = Path(device_path).name
    suffix = name[6:] if name.startswith("ttyACM") else ""
    try:
        return int(suffix), name
    except ValueError:
        return 1_000_000, name


def _autodetect_pololu_maestro_tty_command_ports() -> list[str]:
    """Discover Maestro ttyACM command ports when udev by-id symlinks are absent."""

    preferred: list[str] = []
    fallback: list[str] = []
    sys_class_tty = Path("/sys/class/tty")
    for tty_entry in sorted((str(path) for path in sys_class_tty.glob("ttyACM*")), key=_tty_acm_sort_key):
        tty_name = Path(tty_entry).name
        device_path = Path("/dev") / tty_name
        if not device_path.exists():
            continue
        try:
            sysfs_device = (sys_class_tty / tty_name / "device").resolve(strict=True)
        except OSError:
            continue
        vendor_id = _find_parent_sysfs_value(sysfs_device, "idVendor")
        product_id = _find_parent_sysfs_value(sysfs_device, "idProduct")
        if vendor_id != _POLOLU_VENDOR_ID or product_id not in _MAESTRO_PRODUCT_IDS:
            continue
        interface_number = _find_parent_sysfs_value(sysfs_device, "bInterfaceNumber")
        resolved_path = str(device_path)
        # Prefer the Maestro command port interface when sysfs exposes it.
        if interface_number == "00":
            preferred.append(resolved_path)
        else:
            fallback.append(resolved_path)
    return preferred + fallback


def _autodetect_pololu_maestro_command_ports() -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for pattern in _AUTODETECT_SERIAL_BY_ID_PATTERNS:
        for matched_path in sorted(glob.glob(pattern)):
            if matched_path not in seen and Path(matched_path).exists():
                candidates.append(matched_path)
                seen.add(matched_path)
    for matched_path in _autodetect_pololu_maestro_tty_command_ports():
        if matched_path not in seen:
            candidates.append(matched_path)
            seen.add(matched_path)
    return candidates


def resolve_pololu_maestro_command_port(device_path: str | Path | None) -> str:
    """Return one concrete Maestro command-port path or raise a clear error."""

    configured = None if device_path is None else str(device_path).strip()
    if configured:
        candidate = Path(configured)
        if candidate.exists():
            return str(candidate)
        autodetected_candidates = _autodetect_pololu_maestro_command_ports()
        if autodetected_candidates:
            return autodetected_candidates[0]
        raise RuntimeError(
            "Configured Pololu Maestro command port does not exist: "
            f"{candidate}. Autodetection also found no Pololu Maestro command port."
        )
    autodetected_candidates = _autodetect_pololu_maestro_command_ports()
    if autodetected_candidates:
        return autodetected_candidates[0]
    raise RuntimeError(
        "Pololu Maestro command port not found. "
        "Set TWINR_ATTENTION_SERVO_MAESTRO_DEVICE to a valid /dev/serial/by-id/...-if00 path or expose the Maestro as /dev/ttyACM*."
    )


class PololuMaestroServoPulseWriter:
    """Drive one servo channel through the Maestro USB command port."""

    def __init__(
        self,
        *,
        device_path: str | Path | None = None,
        read_timeout_s: float = _DEFAULT_READ_TIMEOUT_S,
        speed_limit_us_per_s: float | None = None,
        acceleration_limit_us_per_s2: float | None = None,
    ) -> None:
        self._configured_device_path = None if device_path is None else str(device_path).strip() or None
        self._read_timeout_s = max(0.01, float(read_timeout_s))
        self._resolved_device_path: str | None = None
        self._fd: int | None = None
        self._speed_limit = self._encode_speed_limit(speed_limit_us_per_s)
        self._acceleration_limit = self._encode_acceleration_limit(acceleration_limit_us_per_s2)
        self._configured_motion_profile_by_channel: dict[int, tuple[int, int]] = {}

    @property
    def hardware_motion_profile_enabled(self) -> bool:
        """Return whether the writer owns motion smoothing inside the Maestro."""

        return self._speed_limit > 0 or self._acceleration_limit > 0

    @property
    def reports_live_position(self) -> bool:
        """Return whether `current_pulse_width_us` reflects the live emitted pulse."""

        return True

    @property
    def resolved_device_path(self) -> str:
        if self._resolved_device_path is None or not Path(self._resolved_device_path).exists():
            self._resolved_device_path = resolve_pololu_maestro_command_port(self._configured_device_path)
        return self._resolved_device_path

    def probe(self, channel: int) -> None:
        checked_channel = _validate_channel(channel)
        self._ensure_fd()
        self.current_pulse_width_us(gpio_chip="gpiochip0", gpio=checked_channel)

    def _invalidate_connection(self, fd: int | None = None) -> None:
        invalidated_fd = fd
        if invalidated_fd is None:
            invalidated_fd = self._fd
        if invalidated_fd == self._fd:
            self._fd = None
        self._resolved_device_path = None
        if invalidated_fd is not None:
            self._close_device(invalidated_fd)

    def _ensure_fd(self) -> int:
        if self._fd is not None:
            return self._fd
        fd = self._open_device(self.resolved_device_path)
        try:
            self._configure_device(fd)
            self._initialize_device(fd)
        except Exception:
            self._invalidate_connection(fd)
            raise
        self._fd = fd
        return fd

    def _open_device(self, device_path: str) -> int:
        flags = os.O_RDWR | getattr(os, "O_NOCTTY", 0) | getattr(os, "O_CLOEXEC", 0)
        try:
            return os.open(device_path, flags)
        except OSError as exc:
            raise RuntimeError(f"Failed to open Pololu Maestro command port {device_path}: {exc}") from exc

    def _configure_device(self, fd: int) -> None:
        try:
            options = termios.tcgetattr(fd)
        except termios.error as exc:
            raise RuntimeError("Failed to read Pololu Maestro command-port termios settings") from exc
        for name in ("INLCR", "IGNCR", "ICRNL", "IXON", "IXOFF"):
            if hasattr(termios, name):
                options[0] &= ~getattr(termios, name)
        for name in ("ONLCR", "OCRNL"):
            if hasattr(termios, name):
                options[1] &= ~getattr(termios, name)
        for name in ("ECHO", "ECHONL", "ICANON", "ISIG", "IEXTEN"):
            if hasattr(termios, name):
                options[3] &= ~getattr(termios, name)
        options[6][termios.VMIN] = 0
        options[6][termios.VTIME] = 0
        try:
            termios.tcsetattr(fd, termios.TCSANOW, options)
        except termios.error as exc:
            raise RuntimeError("Failed to configure Pololu Maestro command-port raw mode") from exc

    def _flush_input(self, fd: int) -> None:
        try:
            termios.tcflush(fd, termios.TCIFLUSH)
        except termios.error:
            return

    def _initialize_device(self, fd: int) -> None:
        """Kick Maestro auto-baud detection before compact-protocol traffic."""

        self._write_bytes(fd, bytes((_MAESTRO_AUTODETECT_BAUD_BYTE,)))
        self._flush_input(fd)
        self._configured_motion_profile_by_channel.clear()

    def _write_bytes(self, fd: int, payload: bytes) -> None:
        view = memoryview(payload)
        while view:
            try:
                written = os.write(fd, view)
            except OSError as exc:
                self._invalidate_connection(fd)
                raise RuntimeError(f"Failed to write Pololu Maestro command bytes: {exc}") from exc
            if written <= 0:
                self._invalidate_connection(fd)
                raise RuntimeError("Failed to write Pololu Maestro command bytes: short write")
            view = view[written:]

    def _read_bytes(self, fd: int, size: int) -> bytes:
        remaining = int(size)
        chunks: list[bytes] = []
        while remaining > 0:
            readable, _, _ = select.select([fd], [], [], self._read_timeout_s)
            if not readable:
                self._invalidate_connection(fd)
                raise RuntimeError(
                    "Timed out waiting for Pololu Maestro command-port response. "
                    "Twinr's USB serial Maestro path requires SerialMode=USB_DUAL_PORT or USB_CHAINED."
                )
            try:
                chunk = os.read(fd, remaining)
            except OSError as exc:
                self._invalidate_connection(fd)
                raise RuntimeError(f"Failed to read Pololu Maestro command-port response: {exc}") from exc
            if not chunk:
                self._invalidate_connection(fd)
                raise RuntimeError("Pololu Maestro command port closed while reading a response")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _close_device(self, fd: int) -> None:
        try:
            os.close(fd)
        except OSError:
            return

    def _encode_target_command(self, *, channel: int, target_quarter_us: int) -> bytes:
        checked_channel = _validate_channel(channel)
        checked_target = max(0, min(_MAESTRO_MAX_TARGET_QUARTER_US, int(target_quarter_us)))
        return bytes(
            (
                _MAESTRO_COMMAND_SET_TARGET,
                checked_channel,
                checked_target & 0x7F,
                (checked_target >> 7) & 0x7F,
            )
        )

    def _encode_speed_command(self, *, channel: int, speed_limit: int) -> bytes:
        checked_channel = _validate_channel(channel)
        checked_speed = max(0, min(_MAESTRO_MAX_SPEED, int(speed_limit)))
        return bytes(
            (
                _MAESTRO_COMMAND_SET_SPEED,
                checked_channel,
                checked_speed & 0x7F,
                (checked_speed >> 7) & 0x7F,
            )
        )

    def _encode_acceleration_command(self, *, channel: int, acceleration_limit: int) -> bytes:
        checked_channel = _validate_channel(channel)
        checked_acceleration = max(0, min(_MAESTRO_MAX_ACCELERATION, int(acceleration_limit)))
        return bytes(
            (
                _MAESTRO_COMMAND_SET_ACCELERATION,
                checked_channel,
                checked_acceleration & 0x7F,
                (checked_acceleration >> 7) & 0x7F,
            )
        )

    def _encode_speed_limit(self, speed_limit_us_per_s: float | None) -> int:
        if speed_limit_us_per_s is None:
            return 0
        checked_speed = max(0.0, float(speed_limit_us_per_s))
        if checked_speed <= 0.0:
            return 0
        return max(
            1,
            min(_MAESTRO_MAX_SPEED, int(round(checked_speed / _MAESTRO_SPEED_UNIT_US_PER_S))),
        )

    def _encode_acceleration_limit(self, acceleration_limit_us_per_s2: float | None) -> int:
        if acceleration_limit_us_per_s2 is None:
            return 0
        checked_acceleration = max(0.0, float(acceleration_limit_us_per_s2))
        if checked_acceleration <= 0.0:
            return 0
        return max(
            1,
            min(
                _MAESTRO_MAX_ACCELERATION,
                int(round(checked_acceleration / _MAESTRO_ACCELERATION_UNIT_US_PER_S2)),
            ),
        )

    def _ensure_motion_profile(self, *, fd: int, channel: int) -> None:
        checked_channel = _validate_channel(channel)
        desired_profile = (self._speed_limit, self._acceleration_limit)
        if self._configured_motion_profile_by_channel.get(checked_channel) == desired_profile:
            return
        self._write_bytes(
            fd,
            self._encode_speed_command(channel=checked_channel, speed_limit=self._speed_limit),
        )
        self._write_bytes(
            fd,
            self._encode_acceleration_command(
                channel=checked_channel,
                acceleration_limit=self._acceleration_limit,
            ),
        )
        self._configured_motion_profile_by_channel[checked_channel] = desired_profile

    def write(self, *, gpio_chip: str, gpio: int, pulse_width_us: int) -> None:
        del gpio_chip
        fd = self._ensure_fd()
        checked_channel = _validate_channel(int(gpio))
        self._ensure_motion_profile(fd=fd, channel=checked_channel)
        self._write_bytes(
            fd,
            self._encode_target_command(
                channel=checked_channel,
                target_quarter_us=int(pulse_width_us) * _MAESTRO_TARGET_SCALE,
            ),
        )

    def disable(self, *, gpio_chip: str, gpio: int) -> None:
        del gpio_chip
        fd = self._ensure_fd()
        checked_channel = _validate_channel(int(gpio))
        self._ensure_motion_profile(fd=fd, channel=checked_channel)
        self._write_bytes(
            fd,
            self._encode_target_command(channel=checked_channel, target_quarter_us=0),
        )

    def current_pulse_width_us(self, *, gpio_chip: str, gpio: int) -> int | None:
        del gpio_chip
        fd = self._ensure_fd()
        checked_channel = _validate_channel(int(gpio))
        self._flush_input(fd)
        self._write_bytes(fd, bytes((_MAESTRO_COMMAND_GET_POSITION, checked_channel)))
        response = self._read_bytes(fd, 2)
        position_quarter_us = int(response[0]) + (int(response[1]) << 8)
        if position_quarter_us <= 0:
            return None
        return int(round(position_quarter_us / float(_MAESTRO_TARGET_SCALE)))

    def close(self) -> None:
        self._invalidate_connection()
