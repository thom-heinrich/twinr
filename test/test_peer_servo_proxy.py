"""Regression coverage for the peer Pi Pololu Maestro servo proxy."""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "hardware" / "ops" / "peer_servo_proxy.py"
_SPEC = importlib.util.spec_from_file_location("peer_servo_proxy", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class _FakeWriter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int | None]] = []

    @property
    def resolved_device_path(self) -> str:
        return "/dev/ttyACM0"

    def probe(self, channel: int) -> None:
        self.calls.append(("probe", channel, None))

    def write(self, *, gpio_chip: str, gpio: int, pulse_width_us: int) -> None:
        del gpio_chip
        self.calls.append(("write", gpio, pulse_width_us))

    def disable(self, *, gpio_chip: str, gpio: int) -> None:
        del gpio_chip
        self.calls.append(("disable", gpio, None))

    def current_pulse_width_us(self, *, gpio_chip: str, gpio: int) -> int | None:
        del gpio_chip
        self.calls.append(("position", gpio, None))
        return 1500

    def close(self) -> None:
        return


class PeerServoProxyTests(unittest.TestCase):
    def test_service_health_payload_exposes_resolved_device_path(self) -> None:
        service = _MODULE.PeerServoProxyService(writer=_FakeWriter())

        payload = service.health_payload()

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["resolved_device_path"], "/dev/ttyACM0")

    def test_service_write_validates_and_forwards_channel_command(self) -> None:
        writer = _FakeWriter()
        service = _MODULE.PeerServoProxyService(writer=writer)

        payload = service.write(channel=1, pulse_width_us=1500)

        self.assertEqual(payload, {"ok": True, "channel": 1, "pulse_width_us": 1500})
        self.assertEqual(writer.calls, [("write", 1, 1500)])

    def test_handler_returns_position_payload(self) -> None:
        handler_class = _MODULE.build_handler(_MODULE.PeerServoProxyService(writer=_FakeWriter()))
        written = bytearray()
        headers: list[tuple[str, str]] = []

        handler = handler_class.__new__(handler_class)
        handler.path = "/servo/position?channel=1"
        handler.wfile = SimpleNamespace(write=written.extend)
        handler.client_address = ("10.42.0.1", 4242)
        handler.server = SimpleNamespace()
        handler.command = "GET"
        handler.request_version = "HTTP/1.1"
        handler.requestline = f"GET {handler.path} HTTP/1.1"
        handler.send_response = lambda status: headers.append(("status", str(status)))
        handler.send_header = lambda name, value: headers.append((name, value))
        handler.end_headers = lambda: None
        handler.log_message = lambda fmt, *args: None

        handler.do_GET()

        self.assertIn(("status", "200"), headers)
        self.assertIn(("Content-Type", "application/json"), headers)
        self.assertIn(b'"pulse_width_us": 1500', bytes(written))

    def test_handler_rejects_invalid_write_payload(self) -> None:
        handler_class = _MODULE.build_handler(_MODULE.PeerServoProxyService(writer=_FakeWriter()))
        written = bytearray()
        headers: list[tuple[str, str]] = []
        payload_bytes = b'{"channel": 1, "pulse_width_us": 3000}'

        handler = handler_class.__new__(handler_class)
        handler.path = "/servo/write"
        handler.rfile = io.BytesIO(payload_bytes)
        handler.headers = {"Content-Length": str(len(payload_bytes))}
        handler.wfile = SimpleNamespace(write=written.extend)
        handler.client_address = ("10.42.0.1", 4242)
        handler.server = SimpleNamespace()
        handler.command = "POST"
        handler.request_version = "HTTP/1.1"
        handler.requestline = f"POST {handler.path} HTTP/1.1"
        handler.send_response = lambda status: headers.append(("status", str(status)))
        handler.send_header = lambda name, value: headers.append((name, value))
        handler.end_headers = lambda: None
        handler.log_message = lambda fmt, *args: None

        handler.do_POST()

        self.assertIn(("status", "400"), headers)
        self.assertIn(b"pulse_width_us must be between", bytes(written))
