#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///
"""Serve live AI-camera observations from the dedicated proxy Pi.

Purpose
-------
Expose the proxy Pi's local IMX500 observation surface over the direct
peer-LAN link so the main Twinr Pi can keep its existing HDMI attention and
gesture flows even when its own camera connector is broken.

Usage
-----
Command-line invocation examples::

    python3 hardware/ops/peer_ai_camera_observation_proxy.py
    python3 hardware/ops/peer_ai_camera_observation_proxy.py --host 10.42.0.2 --port 8767
    python3 hardware/ops/peer_ai_camera_observation_proxy.py --repo-root /opt/twinr-peer-ai-camera/repo --env-file /opt/twinr-peer-ai-camera/repo/.proxy.env

Inputs
------
- ``--repo-root`` path containing the authoritative ``src/twinr`` package copy
- ``--env-file`` optional config file for local camera tuning on the proxy Pi
- ``--host`` / ``--port`` bind address for the transport-only HTTP surface

Outputs
-------
- Serves JSON endpoints ``/healthz``, ``/observe``, ``/observe_attention``,
  ``/observe_gesture``, and ``/observe_frame_bundle``
- Returns serialized ``AICameraObservation`` payloads plus optional debug facts
- Can also return one bounded detection-plus-frame bundle so the main Pi can
  run the hot attention/gesture lifting locally from one coherent helper round
"""

from __future__ import annotations

import argparse
import base64
from copy import deepcopy
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import logging
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable, cast
from urllib.parse import parse_qs, urlsplit


_DEFAULT_SNAPSHOT_WIDTH = 640
_DEFAULT_SNAPSHOT_HEIGHT = 480
_MIN_SNAPSHOT_WIDTH = 64
_MAX_SNAPSHOT_WIDTH = 4056
_MIN_SNAPSHOT_HEIGHT = 64
_MAX_SNAPSHOT_HEIGHT = 3040
_MIN_TIMEOUT_MS = 250
_MAX_TIMEOUT_MS = 10000


LOGGER = logging.getLogger(__name__)


class AICameraObservationProxyService:
    """Own one local AI-camera adapter and serialize its bounded observations."""

    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        env_file: Path | None = None,
        adapter: object | None = None,
        default_snapshot_width: int = _DEFAULT_SNAPSHOT_WIDTH,
        default_snapshot_height: int = _DEFAULT_SNAPSHOT_HEIGHT,
    ) -> None:
        """Initialize one proxy service from the repo root or a test adapter."""

        self._request_lock = threading.RLock()
        self.default_snapshot_width = _bounded_int(
            "default_snapshot_width",
            default_snapshot_width,
            minimum=_MIN_SNAPSHOT_WIDTH,
            maximum=_MAX_SNAPSHOT_WIDTH,
        )
        self.default_snapshot_height = _bounded_int(
            "default_snapshot_height",
            default_snapshot_height,
            minimum=_MIN_SNAPSHOT_HEIGHT,
            maximum=_MAX_SNAPSHOT_HEIGHT,
        )
        self._last_observe_payload: dict[str, object] | None = None
        self._last_attention_payload: dict[str, object] | None = None
        self._last_gesture_payload: dict[str, object] | None = None
        self._last_frame_bundle_payloads: dict[tuple[int, int], dict[str, object]] = {}
        if adapter is not None:
            self._adapter: Any = adapter
            self.repo_root = None
            self.env_file = env_file
            return

        resolved_repo_root = (repo_root or Path(__file__).resolve().parents[2]).resolve(strict=False)
        src_root = resolved_repo_root / "src"
        if str(src_root) not in sys.path:
            sys.path.insert(0, str(src_root))
        from twinr.agent.base_agent.config import TwinrConfig
        from twinr.hardware.ai_camera import LocalAICameraAdapter

        self.repo_root = resolved_repo_root
        self.env_file = env_file
        if env_file is None:
            config = TwinrConfig()
        else:
            config = TwinrConfig.from_env(env_file)
        self._adapter = LocalAICameraAdapter.from_config(config)

    def close(self) -> None:
        """Close the underlying adapter when it exposes a close hook."""

        closer = getattr(self._adapter, "close", None)
        if callable(closer):
            closer()

    def health_payload(self) -> dict[str, object]:
        """Return one small static health payload."""

        return {
            "status": "ok",
            "service": "peer_ai_camera_observation_proxy",
            "snapshot_width": self.default_snapshot_width,
            "snapshot_height": self.default_snapshot_height,
        }

    def observe_payload(self) -> dict[str, object]:
        """Return one serialized general observation payload."""

        return self._observation_payload_with_busy_cache(
            cache_name="_last_observe_payload",
            producer=lambda: (self._adapter.observe(), None),
        )

    def observe_attention_payload(self) -> dict[str, object]:
        """Return one serialized attention-fast-path payload."""

        return self._observation_payload_with_busy_cache(
            cache_name="_last_attention_payload",
            producer=self._observe_attention_uncached,
        )

    def observe_gesture_payload(self) -> dict[str, object]:
        """Return one serialized gesture-fast-path payload."""

        return self._observation_payload_with_busy_cache(
            cache_name="_last_gesture_payload",
            producer=self._observe_gesture_uncached,
        )

    def observe_frame_bundle_payload(self, *, width: int, height: int) -> dict[str, object]:
        """Return one detection-plus-frame bundle from one coherent helper round."""

        bounded_width = _bounded_int(
            "width",
            width,
            minimum=_MIN_SNAPSHOT_WIDTH,
            maximum=_MAX_SNAPSHOT_WIDTH,
        )
        bounded_height = _bounded_int(
            "height",
            height,
            minimum=_MIN_SNAPSHOT_HEIGHT,
            maximum=_MAX_SNAPSHOT_HEIGHT,
        )
        cache_key = (bounded_width, bounded_height)
        if self._request_lock.acquire(blocking=False):
            try:
                payload = self._build_frame_bundle_payload_locked(
                    width=bounded_width,
                    height=bounded_height,
                )
                self._last_frame_bundle_payloads[cache_key] = payload
                return deepcopy(payload)
            finally:
                self._request_lock.release()
        cached_payload = self._last_frame_bundle_payloads.get(cache_key)
        if isinstance(cached_payload, dict):
            payload = deepcopy(cached_payload)
            payload["cache_state"] = "busy_reused"
            return payload
        with self._request_lock:
            payload = self._build_frame_bundle_payload_locked(
                width=bounded_width,
                height=bounded_height,
            )
            self._last_frame_bundle_payloads[cache_key] = payload
            return deepcopy(payload)

    def snapshot_png(self, *, width: int, height: int) -> bytes:
        """Return one bounded PNG snapshot from the shared AI-camera session."""

        bounded_width = _bounded_int(
            "width",
            width,
            minimum=_MIN_SNAPSHOT_WIDTH,
            maximum=_MAX_SNAPSHOT_WIDTH,
        )
        bounded_height = _bounded_int(
            "height",
            height,
            minimum=_MIN_SNAPSHOT_HEIGHT,
            maximum=_MAX_SNAPSHOT_HEIGHT,
        )
        with self._request_lock:
            capture_snapshot_png = getattr(self._adapter, "capture_snapshot_png", None)
            if callable(capture_snapshot_png):
                payload = capture_snapshot_png(width=bounded_width, height=bounded_height)
                return _coerce_png_bytes(payload)
            runtime_loader = getattr(self._adapter, "_load_detection_runtime", None)
            rgb_capturer = getattr(self._adapter, "_capture_rgb_frame", None)
            if not callable(runtime_loader) or not callable(rgb_capturer):
                raise RuntimeError("adapter_snapshot_unavailable")
            runtime = runtime_loader()
            frame_rgb = rgb_capturer(runtime, observed_at=time.time())
            return _frame_to_png_bytes(
                frame_rgb,
                width=bounded_width,
                height=bounded_height,
            )

    def _observation_payload_with_busy_cache(
        self,
        *,
        cache_name: str,
        producer: Callable[[], tuple[Any, dict[str, object] | None]],
    ) -> dict[str, object]:
        """Return one fresh payload, or the latest cached payload when the camera is busy.

        The proxy uses ``ThreadingHTTPServer`` while the underlying AI-camera adapter
        can only serve one request at a time. Returning the last good payload when the
        adapter is already busy prevents unbounded request queues on the helper Pi,
        which would otherwise make the main Pi time out and lose live HDMI behavior.
        """

        if self._request_lock.acquire(blocking=False):
            try:
                observation, debug_details = producer()
                payload = self._serialize_observation_payload(
                    observation,
                    debug_details=debug_details,
                )
                setattr(self, cache_name, payload)
                return deepcopy(payload)
            finally:
                self._request_lock.release()
        cached_payload = getattr(self, cache_name, None)
        if isinstance(cached_payload, dict):
            payload = deepcopy(cached_payload)
            payload["cache_state"] = "busy_reused"
            return payload
        with self._request_lock:
            observation, debug_details = producer()
            payload = self._serialize_observation_payload(
                observation,
                debug_details=debug_details,
            )
            setattr(self, cache_name, payload)
            return deepcopy(payload)

    def _observe_attention_uncached(self) -> tuple[Any, dict[str, object] | None]:
        """Read one fresh attention observation from the local adapter."""

        observation = self._adapter.observe_attention()
        debug_getter = getattr(self._adapter, "get_last_attention_debug_details", None)
        debug_details = debug_getter() if callable(debug_getter) else None
        return observation, debug_details

    def _observe_gesture_uncached(self) -> tuple[Any, dict[str, object] | None]:
        """Read one fresh gesture observation from the local adapter."""

        observation = self._adapter.observe_gesture()
        debug_getter = getattr(self._adapter, "get_last_gesture_debug_details", None)
        debug_details = debug_getter() if callable(debug_getter) else None
        return observation, debug_details

    def _build_frame_bundle_payload_locked(self, *, width: int, height: int) -> dict[str, object]:
        """Capture one bounded detection-plus-frame bundle while the adapter lock is held."""

        observed_at = time.time()
        runtime_loader = getattr(self._adapter, "_load_detection_runtime", None)
        online_probe = getattr(self._adapter, "_probe_online", None)
        detection_capturer = getattr(self._adapter, "_capture_detection", None)
        rgb_capturer = getattr(self._adapter, "_capture_rgb_frame", None)
        if not callable(runtime_loader) or not callable(online_probe):
            raise RuntimeError("frame_bundle_unavailable")
        if not callable(detection_capturer) or not callable(rgb_capturer):
            raise RuntimeError("frame_bundle_unavailable")
        try:
            runtime = runtime_loader()
            online_error = online_probe(runtime)
            if online_error is not None:
                observation = _health_only_observation(
                    observed_at=observed_at,
                    online=False,
                    ready=False,
                    ai_ready=False,
                    error=online_error,
                )
                return self._serialize_observation_payload(
                    observation,
                    debug_details={
                        "bundle_mode": "detection_frame",
                        "pipeline_error": online_error,
                    },
                )
            detection = detection_capturer(runtime, observed_at=observed_at)
            frame_rgb = rgb_capturer(runtime, observed_at=observed_at)
            frame_png_bytes = _frame_to_png_bytes(
                frame_rgb,
                width=width,
                height=height,
            )
            observation = _detection_observation(
                observed_at=observed_at,
                detection=detection,
            )
            payload = self._serialize_observation_payload(
                observation,
                debug_details={
                    "bundle_mode": "detection_frame",
                    "frame_width": width,
                    "frame_height": height,
                },
            )
            payload["frame_png_base64"] = base64.b64encode(frame_png_bytes).decode("ascii")
            payload["frame_content_type"] = "image/png"
            payload["frame_width"] = width
            payload["frame_height"] = height
            return payload
        except Exception as exc:  # pragma: no cover - depends on Pi runtime behavior.
            error_code = _classify_adapter_error(self._adapter, exc)
            observation = _health_only_observation(
                observed_at=observed_at,
                online=False,
                ready=False,
                ai_ready=False,
                error=error_code,
            )
            return self._serialize_observation_payload(
                observation,
                debug_details={
                    "bundle_mode": "detection_frame",
                    "pipeline_error": error_code,
                },
            )

    def _serialize_observation_payload(
        self,
        observation: Any,
        *,
        debug_details: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Convert one observation dataclass into the JSON payload contract."""

        payload = {
            "observation": asdict(observation),
            "captured_at": getattr(observation, "last_camera_frame_at", None) or getattr(observation, "observed_at", None),
            "model": getattr(observation, "model", None),
        }
        if isinstance(debug_details, dict) and debug_details:
            payload["debug_details"] = dict(debug_details)
        return payload


def build_handler(service: AICameraObservationProxyService):
    """Build one request handler bound to the configured proxy service."""

    class Handler(BaseHTTPRequestHandler):
        """Serve bounded JSON camera observations over the peer LAN."""

        server_version = "TwinrPeerAICameraProxy/1.0"

        def do_GET(self) -> None:  # pylint: disable=invalid-name
            """Route one GET request to the bounded observation endpoints."""

            parsed = urlsplit(self.path)
            try:
                if parsed.path == "/healthz":
                    self._write_json(HTTPStatus.OK, service.health_payload())
                    return
                if parsed.path == "/observe":
                    self._write_json(HTTPStatus.OK, service.observe_payload())
                    return
                if parsed.path == "/observe_attention":
                    self._write_json(HTTPStatus.OK, service.observe_attention_payload())
                    return
                if parsed.path == "/observe_gesture":
                    self._write_json(HTTPStatus.OK, service.observe_gesture_payload())
                    return
                if parsed.path == "/observe_frame_bundle":
                    width = _query_int(
                        parsed.query,
                        "width",
                        default=service.default_snapshot_width,
                        minimum=_MIN_SNAPSHOT_WIDTH,
                        maximum=_MAX_SNAPSHOT_WIDTH,
                    )
                    height = _query_int(
                        parsed.query,
                        "height",
                        default=service.default_snapshot_height,
                        minimum=_MIN_SNAPSHOT_HEIGHT,
                        maximum=_MAX_SNAPSHOT_HEIGHT,
                    )
                    _query_int(
                        parsed.query,
                        "timeout_ms",
                        default=4000,
                        minimum=_MIN_TIMEOUT_MS,
                        maximum=_MAX_TIMEOUT_MS,
                    )
                    self._write_json(
                        HTTPStatus.OK,
                        service.observe_frame_bundle_payload(width=width, height=height),
                    )
                    return
                if parsed.path in {"/snapshot", "/snapshot.png"}:
                    width = _query_int(
                        parsed.query,
                        "width",
                        default=service.default_snapshot_width,
                        minimum=_MIN_SNAPSHOT_WIDTH,
                        maximum=_MAX_SNAPSHOT_WIDTH,
                    )
                    height = _query_int(
                        parsed.query,
                        "height",
                        default=service.default_snapshot_height,
                        minimum=_MIN_SNAPSHOT_HEIGHT,
                        maximum=_MAX_SNAPSHOT_HEIGHT,
                    )
                    _query_int(
                        parsed.query,
                        "timeout_ms",
                        default=4000,
                        minimum=_MIN_TIMEOUT_MS,
                        maximum=_MAX_TIMEOUT_MS,
                    )
                    self._write_bytes(
                        HTTPStatus.OK,
                        service.snapshot_png(width=width, height=height),
                        content_type="image/png",
                    )
                    return
                self._write_text(HTTPStatus.NOT_FOUND, "not found")
            except ValueError as exc:
                self._write_text(HTTPStatus.BAD_REQUEST, str(exc))
            except Exception as exc:  # pragma: no cover - defensive runtime guard.
                LOGGER.exception("Peer AI camera proxy request failed.")
                self._write_text(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

        def log_message(self, fmt: str, *args: object) -> None:
            """Log one request line through the standard logger."""

            LOGGER.info("%s - %s", self.address_string(), fmt % args)

        def _write_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
            """Write one JSON response with stable UTF-8 headers."""

            body = json.dumps(payload, ensure_ascii=True, allow_nan=False).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _write_text(self, status: HTTPStatus, text: str) -> None:
            """Write one short plain-text error response."""

            body = text.encode("utf-8", errors="replace")
            self._write_bytes(status, body, content_type="text/plain; charset=utf-8")

        def _write_bytes(self, status: HTTPStatus, body: bytes, *, content_type: str) -> None:
            """Write one raw byte response with explicit content type."""

            self.send_response(int(status))
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def _bounded_int(name: str, value: object, *, minimum: int, maximum: int) -> int:
    """Return one validated integer constrained to the requested bounds."""

    try:
        number = int(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if number < minimum or number > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return number


def _query_int(raw_query: str, name: str, *, default: int, minimum: int, maximum: int) -> int:
    """Read one bounded integer query parameter from the URL query string."""

    values = parse_qs(raw_query, keep_blank_values=False).get(name)
    if not values:
        return default
    return _bounded_int(name, values[-1], minimum=minimum, maximum=maximum)


def _coerce_png_bytes(payload: object) -> bytes:
    """Return one PNG byte payload or raise a stable error."""

    if isinstance(payload, bytes) and payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return payload
    raise RuntimeError("snapshot_png_invalid")


def _health_only_observation(
    *,
    observed_at: float,
    online: bool,
    ready: bool,
    ai_ready: bool,
    error: str | None,
) -> Any:
    """Build one bounded health-only observation without importing runtime policy."""

    from twinr.hardware.ai_camera import AICameraObservation

    return AICameraObservation(
        observed_at=observed_at,
        camera_online=online,
        camera_ready=ready,
        camera_ai_ready=ai_ready,
        camera_error=error,
        last_camera_frame_at=observed_at if ready else None,
    )


def _detection_observation(*, observed_at: float, detection: Any) -> Any:
    """Project one detection result onto the shared observation contract."""

    from twinr.hardware.ai_camera import AICameraObservation, AICameraZone

    return AICameraObservation(
        observed_at=observed_at,
        camera_online=True,
        camera_ready=True,
        camera_ai_ready=True,
        last_camera_frame_at=observed_at,
        person_count=getattr(detection, "person_count", 0),
        primary_person_box=getattr(detection, "primary_person_box", None),
        primary_person_zone=getattr(detection, "primary_person_zone", AICameraZone.UNKNOWN),
        visible_persons=tuple(getattr(detection, "visible_persons", ()) or ()),
        person_near_device=getattr(detection, "person_near_device", None),
        hand_or_object_near_camera=bool(getattr(detection, "hand_or_object_near_camera", False)),
        objects=tuple(getattr(detection, "objects", ()) or ()),
        model="local-imx500-detection-bundle",
    )


def _classify_adapter_error(adapter: object, exc: Exception) -> str:
    """Map one helper-side exception onto a stable operator-visible error code."""

    classifier = getattr(adapter, "_classify_error", None)
    if callable(classifier):
        try:
            return str(classifier(exc) or "frame_bundle_failed")
        except Exception:  # pragma: no cover - defensive guard around adapter helpers.
            LOGGER.debug("Helper adapter error classifier failed.", exc_info=True)
    return "frame_bundle_failed"


def _frame_to_png_bytes(frame_rgb: object, *, width: int, height: int) -> bytes:
    """Encode one RGB frame into bounded PNG bytes."""

    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - depends on Pi runtime packages.
        raise RuntimeError("pillow_unavailable") from exc

    try:
        image = Image.fromarray(cast(Any, frame_rgb), mode="RGB")
    except Exception as exc:
        raise RuntimeError("snapshot_frame_invalid") from exc

    if image.size != (width, height):
        image = image.resize((width, height))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the peer AI-camera proxy."""

    parser = argparse.ArgumentParser(
        description="Serve local AI-camera observations from the dedicated proxy Pi over HTTP.",
    )
    parser.add_argument("--host", default="10.42.0.2", help="Bind address for the direct-link HTTP listener.")
    parser.add_argument("--port", type=int, default=8767, help="Bind port for the direct-link HTTP listener.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repo root containing src/twinr for the proxy runtime.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Optional env file used to override local camera tuning on the proxy Pi.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Standard-library logging level.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the peer AI-camera observation proxy until interrupted."""

    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).strip().upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    service = AICameraObservationProxyService(
        repo_root=args.repo_root,
        env_file=args.env_file,
    )
    server = ThreadingHTTPServer((args.host, args.port), build_handler(service))
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        LOGGER.info("Stopping peer AI-camera proxy.")
    finally:
        server.server_close()
        service.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
