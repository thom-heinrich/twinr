#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///
"""Serve bounded Pololu Maestro channel commands from a peer Pi over HTTP.

Purpose
-------
Expose the small Pi's locally attached Pololu Maestro to the main Twinr Pi
without pretending the USB device exists on the main host. The main Pi keeps
the high-level follow policy; this helper service only forwards validated
channel/pulse commands to the local Maestro command port.

Usage
-----
Run in the foreground for manual checks::

    python3 hardware/ops/peer_servo_proxy.py --bind 10.42.0.2 --port 8768
    python3 hardware/ops/peer_servo_proxy.py --bind 10.42.0.2 --port 8768 --maestro-device /dev/ttyACM0

The intended productive path is the companion systemd unit
``hardware/ops/twinr-peer-servo-proxy.service`` on the helper Pi.

Inputs
------
- ``GET /healthz`` to return a short service status payload
- ``GET /servo/position?channel=1`` to read the current Maestro target/position
- ``POST /servo/probe`` with ``{\"channel\": 1}`` to validate one channel path
- ``POST /servo/write`` with ``{\"channel\": 1, \"pulse_width_us\": 1500}``
- ``POST /servo/disable`` with ``{\"channel\": 1}``

Outputs
-------
- HTTP 200 with JSON payloads for successful requests
- HTTP 4xx for invalid channel/pulse inputs
- HTTP 5xx for local Maestro/USB failures

Notes
-----
Bind this service only to the direct-link address on ``eth0``. It is
transport-only and must not absorb Twinr follow policy.
"""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
from typing import Protocol
from urllib.parse import parse_qs, urlsplit


_DEFAULT_BIND_HOST = "10.42.0.2"
_DEFAULT_PORT = 8768
_MIN_CHANNEL = 0
_MAX_CHANNEL = 23
_MIN_PULSE_WIDTH_US = 500
_MAX_PULSE_WIDTH_US = 2500
_MAX_BODY_BYTES = 4096


class PeerServoProxyService:
    """Own one local Maestro writer and serialize bounded HTTP access to it."""

    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        maestro_device: str | None = None,
        writer: MaestroServoWriter | None = None,
    ) -> None:
        self._lock = threading.RLock()
        if writer is not None:
            self._writer: MaestroServoWriter = writer
            self.repo_root = repo_root
            self.maestro_device = None if maestro_device is None else str(maestro_device).strip() or None
            return

        resolved_repo_root = (repo_root or Path(__file__).resolve().parents[2]).resolve(strict=False)
        src_root = resolved_repo_root / "src"
        if str(src_root) not in sys.path:
            sys.path.insert(0, str(src_root))
        from twinr.hardware.servo_maestro import PololuMaestroServoPulseWriter

        self.repo_root = resolved_repo_root
        self.maestro_device = None if maestro_device is None else str(maestro_device).strip() or None
        self._writer = PololuMaestroServoPulseWriter(device_path=self.maestro_device)

    def close(self) -> None:
        closer = getattr(self._writer, "close", None)
        if callable(closer):
            closer()

    def health_payload(self) -> dict[str, object]:
        resolved_device_path: str | None = None
        resolve_error: str | None = None
        try:
            resolved_candidate = getattr(self._writer, "resolved_device_path", None)
            if isinstance(resolved_candidate, str):
                resolved_device_path = resolved_candidate
        except Exception as exc:
            resolve_error = str(exc)
        return {
            "ok": True,
            "service": "peer_servo_proxy",
            "configured_device_path": self.maestro_device,
            "resolved_device_path": resolved_device_path,
            "resolve_error": resolve_error,
        }

    def probe(self, *, channel: int) -> dict[str, object]:
        checked_channel = _validate_channel(channel)
        with self._lock:
            self._writer.probe(checked_channel)
        return {"ok": True, "channel": checked_channel}

    def write(self, *, channel: int, pulse_width_us: int) -> dict[str, object]:
        checked_channel = _validate_channel(channel)
        checked_pulse_width_us = _validate_pulse_width_us(pulse_width_us)
        with self._lock:
            self._writer.write(
                gpio_chip="gpiochip0",
                gpio=checked_channel,
                pulse_width_us=checked_pulse_width_us,
            )
        return {
            "ok": True,
            "channel": checked_channel,
            "pulse_width_us": checked_pulse_width_us,
        }

    def disable(self, *, channel: int) -> dict[str, object]:
        checked_channel = _validate_channel(channel)
        with self._lock:
            self._writer.disable(gpio_chip="gpiochip0", gpio=checked_channel)
        return {"ok": True, "channel": checked_channel}

    def position(self, *, channel: int) -> dict[str, object]:
        checked_channel = _validate_channel(channel)
        with self._lock:
            pulse_width_us = self._writer.current_pulse_width_us(gpio_chip="gpiochip0", gpio=checked_channel)
        return {
            "ok": True,
            "channel": checked_channel,
            "pulse_width_us": None if pulse_width_us is None else int(pulse_width_us),
        }


def build_handler(service: PeerServoProxyService) -> type[BaseHTTPRequestHandler]:
    """Return one request handler class bound to the shared proxy service."""

    class PeerServoProxyHandler(BaseHTTPRequestHandler):
        server_version = "TwinrPeerServoProxy/1.0"
        sys_version = ""

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler name
            parsed = urlsplit(self.path)
            if parsed.path == "/healthz":
                self._send_json(HTTPStatus.OK, service.health_payload())
                return
            if parsed.path == "/servo/position":
                try:
                    channel = _query_int(parsed.query, "channel")
                    payload = service.position(channel=channel)
                except ValueError as exc:
                    self._send_error_text(HTTPStatus.BAD_REQUEST, str(exc))
                    return
                except RuntimeError as exc:
                    self._send_error_text(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
                    return
                self._send_json(HTTPStatus.OK, payload)
                return
            self._send_error_text(HTTPStatus.NOT_FOUND, "not found")

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler name
            try:
                payload = _read_json_payload(self)
            except ValueError as exc:
                self._send_error_text(HTTPStatus.BAD_REQUEST, str(exc))
                return
            try:
                if self.path == "/servo/probe":
                    response = service.probe(channel=_payload_int(payload, "channel"))
                elif self.path == "/servo/write":
                    response = service.write(
                        channel=_payload_int(payload, "channel"),
                        pulse_width_us=_payload_int(payload, "pulse_width_us"),
                    )
                elif self.path == "/servo/disable":
                    response = service.disable(channel=_payload_int(payload, "channel"))
                else:
                    self._send_error_text(HTTPStatus.NOT_FOUND, "not found")
                    return
            except ValueError as exc:
                self._send_error_text(HTTPStatus.BAD_REQUEST, str(exc))
                return
            except RuntimeError as exc:
                self._send_error_text(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
                return
            self._send_json(HTTPStatus.OK, response)

        def log_message(self, fmt: str, *args: object) -> None:
            message = fmt % args
            print(f"peer_servo_proxy {self.address_string()} {message}", flush=True)

        def _send_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
            body = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_error_text(self, status: HTTPStatus, message: str) -> None:
            body = message.encode("utf-8", errors="replace")
            self.send_response(int(status))
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return PeerServoProxyHandler


def _validate_channel(raw_channel: object) -> int:
    checked_channel = _coerce_int(raw_channel, field_name="channel")
    if checked_channel < _MIN_CHANNEL or checked_channel > _MAX_CHANNEL:
        raise ValueError(f"channel must be between {_MIN_CHANNEL} and {_MAX_CHANNEL}")
    return checked_channel


def _validate_pulse_width_us(raw_pulse_width_us: object) -> int:
    checked_pulse_width_us = _coerce_int(raw_pulse_width_us, field_name="pulse_width_us")
    if checked_pulse_width_us < _MIN_PULSE_WIDTH_US or checked_pulse_width_us > _MAX_PULSE_WIDTH_US:
        raise ValueError(
            f"pulse_width_us must be between {_MIN_PULSE_WIDTH_US} and {_MAX_PULSE_WIDTH_US}"
        )
    return checked_pulse_width_us


def _query_int(raw_query: str, key: str) -> int:
    parsed = parse_qs(raw_query, keep_blank_values=False)
    raw_value = parsed.get(key, [None])[0]
    if raw_value is None:
        raise ValueError(f"missing query parameter: {key}")
    return int(raw_value)


def _payload_int(payload: dict[str, object], key: str) -> int:
    raw_value = payload.get(key)
    if raw_value is None:
        raise ValueError(f"missing JSON field: {key}")
    return _coerce_int(raw_value, field_name=key)


def _read_json_payload(handler: BaseHTTPRequestHandler) -> dict[str, object]:
    content_length = int(handler.headers.get("Content-Length", "0") or "0")
    if content_length <= 0:
        raise ValueError("missing request body")
    if content_length > _MAX_BODY_BYTES:
        raise ValueError("request body too large")
    payload = json.loads(handler.rfile.read(content_length).decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default=_DEFAULT_BIND_HOST, help="Bind address for the direct-link HTTP listener.")
    parser.add_argument("--port", type=int, default=_DEFAULT_PORT, help="TCP port for the peer-servo proxy.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Authoritative Twinr repo root containing src/twinr.",
    )
    parser.add_argument(
        "--maestro-device",
        default=None,
        help="Optional local Maestro command-port path. Defaults to local autodetection on the helper Pi.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    service = PeerServoProxyService(
        repo_root=args.repo_root,
        maestro_device=args.maestro_device,
    )
    handler_class = build_handler(service)
    server = ThreadingHTTPServer((args.bind, int(args.port)), handler_class)
    print(f"peer_servo_proxy listening on http://{args.bind}:{int(args.port)}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        service.close()
        server.server_close()
    return 0


class MaestroServoWriter(Protocol):
    """Describe the bounded writer contract exposed through the proxy."""

    @property
    def resolved_device_path(self) -> str:
        """Return the locally resolved Maestro command-port path when known."""

    def probe(self, channel: int) -> None:
        """Validate one Maestro channel."""

    def write(self, *, gpio_chip: str, gpio: int, pulse_width_us: int) -> None:
        """Write one pulse width to one Maestro channel."""

    def disable(self, *, gpio_chip: str, gpio: int) -> None:
        """Release one Maestro channel."""

    def current_pulse_width_us(self, *, gpio_chip: str, gpio: int) -> int | None:
        """Read one Maestro channel pulse width or position."""

    def close(self) -> None:
        """Release any writer-side resources."""


def _coerce_int(value: object, *, field_name: str) -> int:
    if not isinstance(value, (int, float, str)):
        raise ValueError(f"{field_name} must be an integer-like value")
    return int(value)


if __name__ == "__main__":
    raise SystemExit(main())
