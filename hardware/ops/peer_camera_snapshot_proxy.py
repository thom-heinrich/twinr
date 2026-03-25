#!/usr/bin/env python3
"""Serve bounded still snapshots from a peer Pi camera over HTTP.

Purpose
-------
Expose one narrow snapshot endpoint on the dedicated camera proxy Pi so the
main Twinr Pi can fetch fresh still images over the direct Ethernet link
without using ad-hoc SSH commands or a full Twinr checkout on the proxy host.

Usage
-----
Run in the foreground for manual checks::

    python3 hardware/ops/peer_camera_snapshot_proxy.py --bind 10.42.0.2 --port 8766

The intended productive path is the companion systemd unit in
``hardware/ops/twinr-peer-camera-proxy.service``.

Inputs
------
- ``GET /snapshot.png?width=640&height=480&timeout_ms=4000`` to trigger a
  bounded ``rpicam-still`` capture and return PNG bytes.
- ``GET /healthz`` to return a small JSON status payload.

Outputs
-------
- HTTP 200 with ``image/png`` for successful snapshots.
- HTTP 200 with ``application/json`` for health checks.
- HTTP 4xx/5xx with short plain-text errors for invalid requests or capture
  failures.

Notes
-----
Bind this service only to the peer-link address on ``eth0``. It is
intentionally transport-only and must not grow Twinr runtime policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any
from urllib.parse import parse_qs, urlsplit

_DEFAULT_BIND_HOST = "10.42.0.2"
_DEFAULT_PORT = 8766
_DEFAULT_WIDTH = 640
_DEFAULT_HEIGHT = 480
_DEFAULT_TIMEOUT_MS = 4000
_DEFAULT_CACHE_TTL_MS = 500
_MIN_WIDTH = 64
_MAX_WIDTH = 4056
_MIN_HEIGHT = 64
_MAX_HEIGHT = 3040
_MIN_TIMEOUT_MS = 250
_MAX_TIMEOUT_MS = 10000
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class SnapshotProxyError(RuntimeError):
    """Represent one operator-safe snapshot proxy failure."""


@dataclass(frozen=True, slots=True)
class SnapshotRequest:
    """Describe one validated still-capture request."""

    width: int
    height: int
    timeout_ms: int


@dataclass(frozen=True, slots=True)
class SnapshotCacheEntry:
    """Store one captured PNG payload for short-term request coalescing."""

    request: SnapshotRequest
    data: bytes
    captured_at_monotonic: float


class SnapshotProxyService:
    """Own bounded ``rpicam-still`` snapshots plus a tiny in-memory cache."""

    def __init__(
        self,
        *,
        camera_binary: str = "rpicam-still",
        default_width: int = _DEFAULT_WIDTH,
        default_height: int = _DEFAULT_HEIGHT,
        default_timeout_ms: int = _DEFAULT_TIMEOUT_MS,
        cache_ttl_ms: int = _DEFAULT_CACHE_TTL_MS,
    ) -> None:
        self.camera_binary = self._resolve_camera_binary(camera_binary)
        self.default_width = _validate_int("default_width", default_width, minimum=_MIN_WIDTH, maximum=_MAX_WIDTH)
        self.default_height = _validate_int("default_height", default_height, minimum=_MIN_HEIGHT, maximum=_MAX_HEIGHT)
        self.default_timeout_ms = _validate_int(
            "default_timeout_ms",
            default_timeout_ms,
            minimum=_MIN_TIMEOUT_MS,
            maximum=_MAX_TIMEOUT_MS,
        )
        self.cache_ttl_ms = _validate_int("cache_ttl_ms", cache_ttl_ms, minimum=0, maximum=5000)
        self._lock = threading.Lock()
        self._cache_entry: SnapshotCacheEntry | None = None
        self._capture_count = 0

    def parse_snapshot_request(self, raw_query: str) -> SnapshotRequest:
        """Validate one HTTP snapshot query string into a bounded request."""

        query = parse_qs(raw_query, keep_blank_values=False)
        return SnapshotRequest(
            width=_query_int(query, "width", default=self.default_width, minimum=_MIN_WIDTH, maximum=_MAX_WIDTH),
            height=_query_int(query, "height", default=self.default_height, minimum=_MIN_HEIGHT, maximum=_MAX_HEIGHT),
            timeout_ms=_query_int(
                query,
                "timeout_ms",
                default=self.default_timeout_ms,
                minimum=_MIN_TIMEOUT_MS,
                maximum=_MAX_TIMEOUT_MS,
            ),
        )

    def health_payload(self) -> dict[str, Any]:
        """Return one small health payload for operators and systemd checks."""

        cache_age_ms: int | None = None
        if self._cache_entry is not None:
            cache_age_ms = int(max(0.0, (time.monotonic() - self._cache_entry.captured_at_monotonic) * 1000.0))
        return {
            "ok": True,
            "camera_binary": self.camera_binary,
            "capture_count": self._capture_count,
            "cache_ttl_ms": self.cache_ttl_ms,
            "cache_age_ms": cache_age_ms,
        }

    def capture_png(self, request: SnapshotRequest) -> bytes:
        """Return one PNG snapshot for the validated request."""

        now = time.monotonic()
        with self._lock:
            if self._cache_entry is not None and self._cache_entry.request == request:
                age_ms = (now - self._cache_entry.captured_at_monotonic) * 1000.0
                if age_ms <= self.cache_ttl_ms:
                    return self._cache_entry.data
            payload = self._capture_png_uncached(request)
            self._cache_entry = SnapshotCacheEntry(
                request=request,
                data=payload,
                captured_at_monotonic=time.monotonic(),
            )
            self._capture_count += 1
            return payload

    def _capture_png_uncached(self, request: SnapshotRequest) -> bytes:
        """Run one uncached bounded ``rpicam-still`` capture and return PNG bytes."""

        temp_path: str | None = None
        try:
            fd, temp_path = tempfile.mkstemp(prefix=".peer-camera-proxy-", suffix=".png")
            os.close(fd)
            result = subprocess.run(
                [
                    self.camera_binary,
                    "-v",
                    "0",
                    "--nopreview",
                    "--immediate",
                    "--timeout",
                    f"{request.timeout_ms}ms",
                    "--width",
                    str(request.width),
                    "--height",
                    str(request.height),
                    "--encoding",
                    "png",
                    "--output",
                    temp_path,
                ],
                check=False,
                capture_output=True,
                stdin=subprocess.DEVNULL,
                timeout=max(3.0, request.timeout_ms / 1000.0 + 2.0),
            )
            if result.returncode != 0:
                raise SnapshotProxyError(_summarize_bytes(result.stderr))
            payload = Path(temp_path).read_bytes()
            if not payload.startswith(_PNG_SIGNATURE):
                raise SnapshotProxyError("rpicam-still returned non-PNG bytes")
            return payload
        except subprocess.TimeoutExpired as exc:
            raise SnapshotProxyError(f"snapshot timed out after {request.timeout_ms}ms") from exc
        except OSError as exc:
            raise SnapshotProxyError(str(exc)) from exc
        finally:
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except FileNotFoundError:
                    pass

    @staticmethod
    def _resolve_camera_binary(camera_binary: str) -> str:
        """Resolve the still-capture executable on PATH or fail clearly."""

        binary = shutil.which(camera_binary)
        if binary is None:
            raise SnapshotProxyError(f"camera binary not found on PATH: {camera_binary}")
        return binary


def build_handler(service: SnapshotProxyService) -> type[BaseHTTPRequestHandler]:
    """Return one request handler class bound to the shared proxy service."""

    class SnapshotProxyHandler(BaseHTTPRequestHandler):
        server_version = "TwinrPeerCameraProxy/1.0"
        sys_version = ""

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler name
            parsed = urlsplit(self.path)
            if parsed.path == "/healthz":
                payload = json.dumps(service.health_payload(), sort_keys=True).encode("utf-8")
                self._send_bytes(HTTPStatus.OK, payload, content_type="application/json")
                return
            if parsed.path not in {"/snapshot", "/snapshot.png"}:
                self._send_bytes(HTTPStatus.NOT_FOUND, b"not found", content_type="text/plain; charset=utf-8")
                return
            try:
                request = service.parse_snapshot_request(parsed.query)
                payload = service.capture_png(request)
            except ValueError as exc:
                self._send_bytes(
                    HTTPStatus.BAD_REQUEST,
                    str(exc).encode("utf-8", errors="replace"),
                    content_type="text/plain; charset=utf-8",
                )
                return
            except SnapshotProxyError as exc:
                self._send_bytes(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    str(exc).encode("utf-8", errors="replace"),
                    content_type="text/plain; charset=utf-8",
                )
                return
            self._send_bytes(HTTPStatus.OK, payload, content_type="image/png")

        def log_message(self, fmt: str, *args: object) -> None:
            message = fmt % args
            print(f"peer_camera_snapshot_proxy {self.address_string()} {message}", flush=True)

        def _send_bytes(self, status: HTTPStatus, payload: bytes, *, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

    return SnapshotProxyHandler


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the peer snapshot proxy."""

    parser = argparse.ArgumentParser(description="Serve bounded still snapshots from the peer Pi camera")
    parser.add_argument("--bind", default=_DEFAULT_BIND_HOST, help="Bind address, ideally the direct-peer eth0 IP")
    parser.add_argument("--port", type=int, default=_DEFAULT_PORT, help="TCP port to listen on")
    parser.add_argument("--camera-binary", default="rpicam-still", help="Still-capture executable to run")
    parser.add_argument("--default-width", type=int, default=_DEFAULT_WIDTH, help="Fallback width when the client omits width")
    parser.add_argument("--default-height", type=int, default=_DEFAULT_HEIGHT, help="Fallback height when the client omits height")
    parser.add_argument(
        "--default-timeout-ms",
        type=int,
        default=_DEFAULT_TIMEOUT_MS,
        help="Fallback rpicam-still timeout in milliseconds when the client omits timeout_ms",
    )
    parser.add_argument(
        "--cache-ttl-ms",
        type=int,
        default=_DEFAULT_CACHE_TTL_MS,
        help="Reuse a recent matching snapshot for this many milliseconds",
    )
    return parser


def main() -> int:
    """Parse CLI args, start the server, and block until interrupted."""

    args = build_parser().parse_args()
    service = SnapshotProxyService(
        camera_binary=args.camera_binary,
        default_width=args.default_width,
        default_height=args.default_height,
        default_timeout_ms=args.default_timeout_ms,
        cache_ttl_ms=args.cache_ttl_ms,
    )
    handler_class = build_handler(service)
    with ThreadingHTTPServer((args.bind, args.port), handler_class) as server:
        print(
            (
                "peer_camera_snapshot_proxy_listening "
                f"bind={args.bind}:{args.port} default_width={args.default_width} "
                f"default_height={args.default_height} default_timeout_ms={args.default_timeout_ms} "
                f"cache_ttl_ms={args.cache_ttl_ms}"
            ),
            flush=True,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            return 0
    return 0


def _query_int(
    query: dict[str, list[str]],
    name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    """Parse one optional integer query parameter through the shared bounds."""

    value = query.get(name, [str(default)])[0]
    return _validate_int(name, int(value), minimum=minimum, maximum=maximum)


def _validate_int(name: str, value: int, *, minimum: int, maximum: int) -> int:
    """Validate one integer against an inclusive min/max range."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must stay between {minimum} and {maximum}")
    return value


def _summarize_bytes(payload: bytes | None) -> str:
    """Collapse stderr bytes into one short printable operator-facing reason."""

    if not payload:
        return "empty error output"
    text = payload.decode("utf-8", errors="replace")
    return " ".join(text.split())[:512] or "empty error output"


if __name__ == "__main__":
    raise SystemExit(main())
