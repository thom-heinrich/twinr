from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.servo_maestro import (
    PololuMaestroServoPulseWriter,
    _autodetect_pololu_maestro_tty_command_ports,
    resolve_pololu_maestro_command_port,
)


class FakePololuMaestroServoPulseWriter(PololuMaestroServoPulseWriter):
    def __init__(
        self,
        *,
        device_path: str,
        speed_limit_us_per_s: float | None = None,
        acceleration_limit_us_per_s2: float | None = None,
    ) -> None:
        super().__init__(
            device_path=device_path,
            speed_limit_us_per_s=speed_limit_us_per_s,
            acceleration_limit_us_per_s2=acceleration_limit_us_per_s2,
        )
        self.opened_paths: list[str] = []
        self.written_payloads: list[bytes] = []
        self.read_responses: list[bytes] = []
        self.closed_fds: list[int] = []
        self.flush_calls: list[int] = []

    def _open_device(self, device_path: str) -> int:
        self.opened_paths.append(device_path)
        return 77

    def _configure_device(self, fd: int) -> None:
        del fd

    def _flush_input(self, fd: int) -> None:
        self.flush_calls.append(fd)

    def _write_bytes(self, fd: int, payload: bytes) -> None:
        del fd
        self.written_payloads.append(payload)

    def _read_bytes(self, fd: int, size: int) -> bytes:
        del fd
        response = self.read_responses.pop(0)
        if len(response) != size:
            raise AssertionError(f"Expected {size} response bytes, got {len(response)}")
        return response

    def _close_device(self, fd: int) -> None:
        self.closed_fds.append(fd)


class ResolvePololuMaestroCommandPortTests(unittest.TestCase):
    def test_explicit_existing_path_wins(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            device_path = Path(temp_dir) / "ttyACM0"
            device_path.write_text("", encoding="utf-8")

            resolved = resolve_pololu_maestro_command_port(device_path)

        self.assertEqual(resolved, str(device_path))

    def test_autodetect_prefers_if00_symlink(self) -> None:
        with mock.patch(
            "twinr.hardware.servo_maestro.glob.glob",
            return_value=["/dev/serial/by-id/usb-Pololu_Maestro-if00"],
        ):
            with mock.patch("twinr.hardware.servo_maestro.Path.exists", return_value=True):
                resolved = resolve_pololu_maestro_command_port(None)

        self.assertEqual(resolved, "/dev/serial/by-id/usb-Pololu_Maestro-if00")

    def test_missing_configured_path_falls_back_to_autodetected_port(self) -> None:
        with mock.patch(
            "twinr.hardware.servo_maestro._autodetect_pololu_maestro_command_ports",
            return_value=["/dev/ttyACM0"],
        ):
            resolved = resolve_pololu_maestro_command_port("/dev/serial/by-id/missing-maestro-if00")

        self.assertEqual(resolved, "/dev/ttyACM0")

    def test_tty_acm_autodetect_prefers_command_interface_zero_zero(self) -> None:
        fake_ttys = [
            Path("/sys/class/tty/ttyACM1"),
            Path("/sys/class/tty/ttyACM0"),
        ]

        def fake_resolve(self: Path, *, strict: bool = False) -> Path:
            del strict
            return Path(f"/sys/devices/fake/{self.parent.name}")

        def fake_device_exists(self: Path) -> bool:
            return str(self) in {"/dev/ttyACM0", "/dev/ttyACM1"}

        with (
            mock.patch("twinr.hardware.servo_maestro.Path.glob", return_value=fake_ttys),
            mock.patch("pathlib.Path.resolve", new=fake_resolve),
            mock.patch(
                "twinr.hardware.servo_maestro._find_parent_sysfs_value",
                side_effect=lambda start_path, filename: (
                    "1ffb"
                    if filename == "idVendor"
                    else "008a"
                    if filename == "idProduct"
                    else "00"
                    if filename == "bInterfaceNumber" and str(start_path).endswith("ttyACM0")
                    else "02"
                    if filename == "bInterfaceNumber"
                    else None
                ),
            ),
            mock.patch("pathlib.Path.exists", fake_device_exists),
        ):
            detected = _autodetect_pololu_maestro_tty_command_ports()

        self.assertEqual(detected, ["/dev/ttyACM0", "/dev/ttyACM1"])


class PololuMaestroServoPulseWriterTests(unittest.TestCase):
    def _build_writer(self) -> FakePololuMaestroServoPulseWriter:
        with tempfile.TemporaryDirectory() as temp_dir:
            device_path = Path(temp_dir) / "ttyACM0"
            device_path.write_text("", encoding="utf-8")
            writer = FakePololuMaestroServoPulseWriter(device_path=str(device_path))
            writer._resolved_device_path = str(device_path)
            return writer

    def test_write_encodes_set_target_in_quarter_microseconds(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            device_path = Path(temp_dir) / "ttyACM0"
            device_path.write_text("", encoding="utf-8")
            writer = FakePololuMaestroServoPulseWriter(device_path=str(device_path))
            writer._resolved_device_path = str(device_path)

            writer.write(gpio_chip="ignored", gpio=0, pulse_width_us=1500)

        self.assertEqual(writer.opened_paths, [str(device_path)])
        self.assertEqual(
            writer.written_payloads,
            [
                bytes((0xAA,)),
                bytes((0x87, 0x00, 0x00, 0x00)),
                bytes((0x89, 0x00, 0x00, 0x00)),
                bytes((0x84, 0x00, 0x70, 0x2E)),
            ],
        )

    def test_write_applies_configured_speed_and_acceleration_before_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            device_path = Path(temp_dir) / "ttyACM0"
            device_path.write_text("", encoding="utf-8")
            writer = FakePololuMaestroServoPulseWriter(
                device_path=str(device_path),
                speed_limit_us_per_s=100.0,
                acceleration_limit_us_per_s2=625.0,
            )
            writer._resolved_device_path = str(device_path)

            writer.write(gpio_chip="ignored", gpio=1, pulse_width_us=1600)

        self.assertEqual(
            writer.written_payloads,
            [
                bytes((0xAA,)),
                bytes((0x87, 0x01, 0x04, 0x00)),
                bytes((0x89, 0x01, 0x02, 0x00)),
                bytes((0x84, 0x01, 0x00, 0x32)),
            ],
        )

    def test_probe_requires_command_port_response(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            device_path = Path(temp_dir) / "ttyACM0"
            device_path.write_text("", encoding="utf-8")
            writer = FakePololuMaestroServoPulseWriter(device_path=str(device_path))
            writer._resolved_device_path = str(device_path)
            writer.read_responses.append(bytes((0x00, 0x00)))

            writer.probe(0)

        self.assertEqual(writer.opened_paths, [str(device_path)])
        self.assertEqual(writer.flush_calls, [77, 77])
        self.assertEqual(
            writer.written_payloads,
            [bytes((0xAA,)), bytes((0x90, 0x00))],
        )

    def test_disable_sends_zero_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            device_path = Path(temp_dir) / "ttyACM0"
            device_path.write_text("", encoding="utf-8")
            writer = FakePololuMaestroServoPulseWriter(device_path=str(device_path))
            writer._resolved_device_path = str(device_path)

            writer.disable(gpio_chip="ignored", gpio=0)

        self.assertEqual(
            writer.written_payloads,
            [
                bytes((0xAA,)),
                bytes((0x87, 0x00, 0x00, 0x00)),
                bytes((0x89, 0x00, 0x00, 0x00)),
                bytes((0x84, 0x00, 0x00, 0x00)),
            ],
        )

    def test_current_pulse_width_reads_channel_position(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            device_path = Path(temp_dir) / "ttyACM0"
            device_path.write_text("", encoding="utf-8")
            writer = FakePololuMaestroServoPulseWriter(device_path=str(device_path))
            writer._resolved_device_path = str(device_path)
            writer.read_responses.append(bytes((0x70, 0x17)))

            pulse_width_us = writer.current_pulse_width_us(gpio_chip="ignored", gpio=0)

        self.assertEqual(pulse_width_us, 1500)
        self.assertEqual(writer.flush_calls, [77, 77])
        self.assertEqual(
            writer.written_payloads,
            [bytes((0xAA,)), bytes((0x90, 0x00))],
        )

    def test_close_releases_open_fd(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            device_path = Path(temp_dir) / "ttyACM0"
            device_path.write_text("", encoding="utf-8")
            writer = FakePololuMaestroServoPulseWriter(device_path=str(device_path))
            writer._resolved_device_path = str(device_path)
            writer.write(gpio_chip="ignored", gpio=0, pulse_width_us=1500)
            writer.close()

        self.assertEqual(writer.closed_fds, [77])

    def test_resolved_device_path_reresolves_when_cached_path_disappears(self) -> None:
        writer = FakePololuMaestroServoPulseWriter(device_path="/dev/serial/by-id/missing-maestro-if00")
        writer._resolved_device_path = "/dev/ttyACM2"

        def fake_exists(self: Path) -> bool:
            return str(self) == "/dev/ttyACM0"

        with (
            mock.patch("pathlib.Path.exists", fake_exists),
            mock.patch(
                "twinr.hardware.servo_maestro.resolve_pololu_maestro_command_port",
                return_value="/dev/ttyACM0",
            ) as resolver,
        ):
            resolved = writer.resolved_device_path

        self.assertEqual(resolved, "/dev/ttyACM0")
        resolver.assert_called_once_with("/dev/serial/by-id/missing-maestro-if00")


if __name__ == "__main__":
    unittest.main()
