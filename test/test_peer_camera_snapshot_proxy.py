"""Regression coverage for the peer Pi camera snapshot proxy."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


_PNG_BYTES = b"\x89PNG\r\n\x1a\npeerpng"
_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "hardware" / "ops" / "peer_camera_snapshot_proxy.py"
_SPEC = importlib.util.spec_from_file_location("peer_camera_snapshot_proxy", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class PeerCameraSnapshotProxyTests(unittest.TestCase):
    def test_parse_snapshot_request_applies_defaults_and_bounds(self) -> None:
        with patch.object(_MODULE.shutil, "which", return_value="/usr/bin/rpicam-still"):
            service = _MODULE.SnapshotProxyService(default_width=800, default_height=600, default_timeout_ms=3200)

        request = service.parse_snapshot_request("")

        self.assertEqual(request, _MODULE.SnapshotRequest(width=800, height=600, timeout_ms=3200))

    def test_capture_png_reuses_matching_recent_cache_entry(self) -> None:
        with patch.object(_MODULE.shutil, "which", return_value="/usr/bin/rpicam-still"):
            service = _MODULE.SnapshotProxyService(cache_ttl_ms=500)
        request = _MODULE.SnapshotRequest(width=640, height=480, timeout_ms=4000)

        with patch.object(service, "_capture_png_uncached", side_effect=[_PNG_BYTES, b"new"]) as capture_mock:
            first = service.capture_png(request)
            second = service.capture_png(request)

        self.assertEqual(first, _PNG_BYTES)
        self.assertEqual(second, _PNG_BYTES)
        self.assertEqual(capture_mock.call_count, 1)

    def test_handler_returns_png_snapshot_payload(self) -> None:
        with patch.object(_MODULE.shutil, "which", return_value="/usr/bin/rpicam-still"):
            service = _MODULE.SnapshotProxyService()
        handler_class = _MODULE.build_handler(service)
        written = bytearray()
        headers: list[tuple[str, str]] = []

        handler = handler_class.__new__(handler_class)
        handler.path = "/snapshot.png?width=320&height=240&timeout_ms=1500"
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

        with patch.object(service, "capture_png", return_value=_PNG_BYTES) as capture_mock:
            handler.do_GET()

        capture_request = capture_mock.call_args.args[0]
        self.assertEqual(capture_request, _MODULE.SnapshotRequest(width=320, height=240, timeout_ms=1500))
        self.assertEqual(bytes(written), _PNG_BYTES)
        self.assertIn(("status", "200"), headers)
        self.assertIn(("Content-Type", "image/png"), headers)

    def test_handler_returns_bad_request_for_invalid_query(self) -> None:
        with patch.object(_MODULE.shutil, "which", return_value="/usr/bin/rpicam-still"):
            service = _MODULE.SnapshotProxyService()
        handler_class = _MODULE.build_handler(service)
        written = bytearray()
        headers: list[tuple[str, str]] = []

        handler = handler_class.__new__(handler_class)
        handler.path = "/snapshot.png?width=abc"
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

        self.assertIn(("status", "400"), headers)
        self.assertIn("invalid literal for int()", bytes(written).decode("utf-8"))
