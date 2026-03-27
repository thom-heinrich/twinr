#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///
"""Serve bounded local servo commands from a peer Pi over HTTP.

Purpose
-------
Expose one helper Pi's local servo output path to the main Twinr Pi without
pretending the device exists on the main host. The main Pi keeps the high-level
follow policy; this helper service only forwards validated channel/pulse
commands to one local writer such as a Pololu Maestro command port or a direct
GPIO18 servo output path.

Usage
-----
Run in the foreground for manual checks::

    python3 hardware/ops/peer_servo_proxy.py --bind 10.42.0.2 --port 8768
    python3 hardware/ops/peer_servo_proxy.py --bind 10.42.0.2 --port 8768 --maestro-device /dev/ttyACM0
    python3 hardware/ops/peer_servo_proxy.py --bind 10.42.0.2 --port 8768 --driver lgpio_pwm --gpio 18 --logical-channel 1

The intended productive path is the companion systemd unit
``hardware/ops/twinr-peer-servo-proxy.service`` on the helper Pi.

Inputs
------
- ``GET /healthz`` to return a short service status payload
- ``GET /servo/position?channel=1`` to read the current target/position
- ``POST /servo/probe`` with ``{\"channel\": 1}`` to validate one local path
- ``POST /servo/write`` with ``{\"channel\": 1, \"pulse_width_us\": 1500}``
- ``POST /servo/disable`` with ``{\"channel\": 1}``

Outputs
-------
- HTTP 200 with JSON payloads for successful requests
- HTTP 4xx for invalid channel/pulse inputs
- HTTP 5xx for local writer failures

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
from typing import Callable, Protocol, cast
from urllib.parse import parse_qs, urlsplit


_DEFAULT_BIND_HOST = "10.42.0.2"
_DEFAULT_PORT = 8768
_DEFAULT_DRIVER = "pololu_maestro"
_DEFAULT_GPIO_CHIP = "gpiochip0"
_DEFAULT_LOGICAL_CHANNEL = 1
_MIN_CHANNEL = 0
_MAX_CHANNEL = 23
_MIN_PULSE_WIDTH_US = 500
_MAX_PULSE_WIDTH_US = 2500
_MAX_BODY_BYTES = 4096
_DEFAULT_SERVO_FREQUENCY_HZ = 50.0


class PeerServoProxyService:
    """Own one local servo writer and serialize bounded HTTP access to it."""

    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        driver: str = _DEFAULT_DRIVER,
        gpio_chip: str = _DEFAULT_GPIO_CHIP,
        gpio: int | None = None,
        logical_channel: int = _DEFAULT_LOGICAL_CHANNEL,
        maestro_device: str | None = None,
        writer: ServoWriter | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self.driver = _normalize_driver(driver)
        self.gpio_chip = _normalize_gpio_chip(gpio_chip)
        self.local_gpio = None if gpio is None else int(gpio)
        self.logical_channel = int(logical_channel)
        self.maestro_device = None if maestro_device is None else str(maestro_device).strip() or None
        if writer is not None:
            self._writer: ServoWriter = writer
            self.repo_root = repo_root
            return

        resolved_repo_root = (repo_root or Path(__file__).resolve().parents[2]).resolve(strict=False)
        src_root = resolved_repo_root / "src"
        if str(src_root) not in sys.path:
            sys.path.insert(0, str(src_root))

        self.repo_root = resolved_repo_root
        self._writer = _build_writer(
            repo_root=self.repo_root,
            driver=self.driver,
            gpio_chip=self.gpio_chip,
            gpio=self.local_gpio,
            maestro_device=self.maestro_device,
        )

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
            "driver": self.driver,
            "gpio_chip": self.gpio_chip,
            "local_gpio": self.local_gpio,
            "logical_channel": self.logical_channel,
            "configured_device_path": self.maestro_device,
            "resolved_device_path": resolved_device_path,
            "resolve_error": resolve_error,
        }

    def probe(self, *, channel: int) -> dict[str, object]:
        checked_channel = _validate_channel(channel)
        local_gpio = self._resolve_local_gpio(checked_channel)
        with self._lock:
            self._writer.probe(local_gpio)
        return {"ok": True, "channel": checked_channel}

    def write(self, *, channel: int, pulse_width_us: int) -> dict[str, object]:
        checked_channel = _validate_channel(channel)
        checked_pulse_width_us = _validate_pulse_width_us(pulse_width_us)
        local_gpio = self._resolve_local_gpio(checked_channel)
        with self._lock:
            self._writer.write(
                gpio_chip=self.gpio_chip,
                gpio=local_gpio,
                pulse_width_us=checked_pulse_width_us,
            )
        return {
            "ok": True,
            "channel": checked_channel,
            "pulse_width_us": checked_pulse_width_us,
        }

    def disable(self, *, channel: int) -> dict[str, object]:
        checked_channel = _validate_channel(channel)
        local_gpio = self._resolve_local_gpio(checked_channel)
        with self._lock:
            self._writer.disable(gpio_chip=self.gpio_chip, gpio=local_gpio)
        return {"ok": True, "channel": checked_channel}

    def position(self, *, channel: int) -> dict[str, object]:
        checked_channel = _validate_channel(channel)
        local_gpio = self._resolve_local_gpio(checked_channel)
        with self._lock:
            pulse_width_us = self._writer.current_pulse_width_us(gpio_chip=self.gpio_chip, gpio=local_gpio)
        return {
            "ok": True,
            "channel": checked_channel,
            "pulse_width_us": None if pulse_width_us is None else int(pulse_width_us),
        }

    def _resolve_local_gpio(self, channel: int) -> int:
        if self.driver == "pololu_maestro":
            return int(channel)
        if self.local_gpio is None:
            raise RuntimeError("peer_servo_proxy local GPIO output requires --gpio")
        if int(channel) != self.logical_channel:
            raise ValueError(
                f"channel must match configured logical channel {self.logical_channel} for driver {self.driver}"
            )
        return int(self.local_gpio)


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
        "--driver",
        default=_DEFAULT_DRIVER,
        choices=("pololu_maestro", "lgpio_pwm", "auto"),
        help="Local helper-Pi servo writer. 'auto' prefers direct GPIO when --gpio is set, otherwise Maestro.",
    )
    parser.add_argument(
        "--gpio-chip",
        default=_DEFAULT_GPIO_CHIP,
        help="GPIO chip name for direct helper-Pi output, e.g. gpiochip0.",
    )
    parser.add_argument(
        "--gpio",
        type=int,
        default=None,
        help="Local helper-Pi GPIO number for direct servo output, e.g. 18.",
    )
    parser.add_argument(
        "--logical-channel",
        type=int,
        default=_DEFAULT_LOGICAL_CHANNEL,
        help="Logical channel number exposed over HTTP when using direct GPIO output.",
    )
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
        driver=args.driver,
        gpio_chip=args.gpio_chip,
        gpio=args.gpio,
        logical_channel=args.logical_channel,
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


class ServoWriter(Protocol):
    """Describe the bounded writer contract exposed through the proxy."""

    @property
    def resolved_device_path(self) -> str | None:
        """Return one resolved local device path when the writer has one."""

    def probe(self, channel: int) -> None:
        """Validate one locally reachable output."""

    def write(self, *, gpio_chip: str, gpio: int, pulse_width_us: int) -> None:
        """Write one pulse width to one local output."""

    def disable(self, *, gpio_chip: str, gpio: int) -> None:
        """Release one local output."""

    def current_pulse_width_us(self, *, gpio_chip: str, gpio: int) -> int | None:
        """Read one local output pulse width or position when known."""

    def close(self) -> None:
        """Release any writer-side resources."""


class _LGPIOPWMServoPulseWriter:
    """Drive one helper-Pi servo line through `lgpio.tx_pwm` at 50 Hz."""

    def __init__(self) -> None:
        self._lgpio: object | None = None
        self._handles: dict[int, int] = {}
        self._claimed_outputs: dict[int, set[int]] = {}
        self._last_pulse_widths: dict[tuple[int, int], int] = {}

    @property
    def resolved_device_path(self) -> None:
        return None

    def _module(self) -> object:
        if self._lgpio is None:
            try:
                import lgpio  # pylint: disable=import-error
            except Exception as exc:  # pragma: no cover - depends on proxy image
                raise RuntimeError("python3-lgpio is required for peer GPIO servo output") from exc
            self._lgpio = lgpio
        return self._lgpio

    def _handle_for_chip(self, gpio_chip: str) -> tuple[object, int]:
        module = self._module()
        chip_index = _normalize_chip_index(gpio_chip)
        handle = self._handles.get(chip_index)
        if handle is None:
            gpiochip_open = cast(Callable[[int], int], getattr(module, "gpiochip_open"))
            handle = gpiochip_open(chip_index)
            self._handles[chip_index] = handle
        return module, handle

    def _claim_output_if_needed(self, *, module: object, handle: int, gpio_chip: str, gpio: int) -> None:
        chip_index = _normalize_chip_index(gpio_chip)
        claimed_outputs = self._claimed_outputs.setdefault(chip_index, set())
        checked_gpio = int(gpio)
        if checked_gpio in claimed_outputs:
            return
        claim_output = cast(Callable[[int, int, int], object] | None, getattr(module, "gpio_claim_output", None))
        if callable(claim_output):
            claim_output(handle, checked_gpio, 0)
        claimed_outputs.add(checked_gpio)

    def _apply_servo_pulse(self, module: object, *, handle: int, gpio: int, pulse_width_us: int) -> None:
        tx_servo = getattr(module, "tx_servo", None)
        status: object | None = None
        if callable(tx_servo):
            status = tx_servo(handle, int(gpio), int(pulse_width_us), int(_DEFAULT_SERVO_FREQUENCY_HZ), 0, 0)
        else:
            tx_pwm = getattr(module, "tx_pwm", None)
            if not callable(tx_pwm):
                raise RuntimeError("python3-lgpio with tx_servo or tx_pwm support is required for peer GPIO servo output")
            status = tx_pwm(
                handle,
                int(gpio),
                float(_DEFAULT_SERVO_FREQUENCY_HZ),
                float(self._duty_cycle_percent_for_pulse_width_us(int(pulse_width_us))),
            )
        if isinstance(status, int) and status < 0:
            raise RuntimeError(f"lgpio servo pulse write failed with status {status}")

    def _duty_cycle_percent_for_pulse_width_us(self, pulse_width_us: int) -> float:
        checked_pulse_width_us = max(0, min(20_000, int(pulse_width_us)))
        return round((checked_pulse_width_us / 20_000.0) * 100.0, 6)

    def probe(self, channel: int) -> None:
        del channel
        self._module()

    def write(self, *, gpio_chip: str, gpio: int, pulse_width_us: int) -> None:
        module, handle = self._handle_for_chip(gpio_chip)
        checked_gpio = int(gpio)
        self._claim_output_if_needed(
            module=module,
            handle=handle,
            gpio_chip=gpio_chip,
            gpio=checked_gpio,
        )
        self._apply_servo_pulse(
            module,
            handle=handle,
            gpio=checked_gpio,
            pulse_width_us=int(pulse_width_us),
        )
        self._last_pulse_widths[(_normalize_chip_index(gpio_chip), checked_gpio)] = int(pulse_width_us)

    def disable(self, *, gpio_chip: str, gpio: int) -> None:
        module, handle = self._handle_for_chip(gpio_chip)
        chip_index = _normalize_chip_index(gpio_chip)
        checked_gpio = int(gpio)
        self._apply_servo_pulse(
            module,
            handle=handle,
            gpio=checked_gpio,
            pulse_width_us=0,
        )
        claim_input_candidate = getattr(module, "gpio_claim_input", None)
        if claim_input_candidate is not None:
            claim_input_fn = cast(Callable[[int, int], object], claim_input_candidate)
            claim_input_fn(handle, checked_gpio)
        claimed_outputs = self._claimed_outputs.get(chip_index)
        if claimed_outputs is not None:
            claimed_outputs.discard(checked_gpio)
            if not claimed_outputs:
                self._claimed_outputs.pop(chip_index, None)
        self._last_pulse_widths.pop((chip_index, checked_gpio), None)

    def current_pulse_width_us(self, *, gpio_chip: str, gpio: int) -> int | None:
        return self._last_pulse_widths.get((_normalize_chip_index(gpio_chip), int(gpio)))

    def close(self) -> None:
        module = self._module() if self._lgpio is not None else None
        if module is None:
            return
        for handle in self._handles.values():
            try:
                gpiochip_close = cast(Callable[[int], object], getattr(module, "gpiochip_close"))
                gpiochip_close(handle)
            except Exception:
                continue
        self._handles.clear()
        self._claimed_outputs.clear()
        self._last_pulse_widths.clear()


def _normalize_chip_index(chip_name: str) -> int:
    normalized = _normalize_gpio_chip(chip_name)
    if normalized.startswith("gpiochip"):
        suffix = normalized[len("gpiochip") :]
        if suffix and suffix.isdigit():
            return int(suffix)
    if normalized.isdigit():
        return int(normalized)
    raise ValueError(f"Unsupported GPIO chip name for peer servo output: {chip_name}")


def _normalize_driver(raw_driver: str | None) -> str:
    normalized = str(raw_driver or _DEFAULT_DRIVER).strip().lower() or _DEFAULT_DRIVER
    if normalized in {"maestro", "pololu"}:
        normalized = "pololu_maestro"
    if normalized not in {"pololu_maestro", "lgpio_pwm", "auto"}:
        raise ValueError(f"Unsupported peer servo proxy driver: {raw_driver}")
    return normalized


def _normalize_gpio_chip(raw_gpio_chip: str | None) -> str:
    normalized = str(raw_gpio_chip or _DEFAULT_GPIO_CHIP).strip()
    if not normalized:
        raise ValueError("GPIO chip name must not be empty")
    return normalized


def _build_writer(
    *,
    repo_root: Path,
    driver: str,
    gpio_chip: str,
    gpio: int | None,
    maestro_device: str | None,
) -> ServoWriter:
    normalized_driver = _normalize_driver(driver)
    if normalized_driver == "auto":
        if gpio is not None:
            writer = _LGPIOPWMServoPulseWriter()
            writer.probe(int(gpio))
            return writer
        normalized_driver = "pololu_maestro"
    if normalized_driver == "lgpio_pwm":
        if gpio is None:
            raise RuntimeError("peer_servo_proxy --driver lgpio_pwm requires --gpio")
        writer = _LGPIOPWMServoPulseWriter()
        writer.probe(int(gpio))
        return writer
    resolved_repo_root = repo_root.resolve(strict=False)
    src_root = resolved_repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from twinr.hardware.servo_maestro import PololuMaestroServoPulseWriter

    del gpio_chip, gpio
    return PololuMaestroServoPulseWriter(device_path=maestro_device)


def _coerce_int(value: object, *, field_name: str) -> int:
    if not isinstance(value, (int, float, str)):
        raise ValueError(f"{field_name} must be an integer-like value")
    return int(value)


if __name__ == "__main__":
    raise SystemExit(main())
