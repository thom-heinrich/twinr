"""Regression coverage for the peer Pi AI-camera observation proxy."""

from __future__ import annotations

import base64
import importlib.util
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.ai_camera import AICameraFineHandGesture, AICameraObservation


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "hardware" / "ops" / "peer_ai_camera_observation_proxy.py"
_SPEC = importlib.util.spec_from_file_location("peer_ai_camera_observation_proxy", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jK7sAAAAASUVORK5CYII="
)


class _FakeAdapter:
    def __init__(self) -> None:
        self._last_attention_debug = {"attention_source": "face_anchor"}
        self._last_gesture_debug = {"resolved_source": "builtin"}
        self.attention_calls = 0
        self.gesture_calls = 0
        self.bundle_calls = 0

    def observe(self):
        return AICameraObservation(
            observed_at=10.0,
            camera_online=True,
            camera_ready=True,
            camera_ai_ready=True,
            person_count=1,
        )

    def observe_attention(self):
        self.attention_calls += 1
        return AICameraObservation(
            observed_at=11.0,
            camera_online=True,
            camera_ready=True,
            camera_ai_ready=True,
            person_count=1,
            primary_person_zone="center",
        )

    def observe_gesture(self):
        self.gesture_calls += 1
        return AICameraObservation(
            observed_at=12.0,
            camera_online=True,
            camera_ready=True,
            camera_ai_ready=True,
            fine_hand_gesture=AICameraFineHandGesture.PEACE_SIGN,
            fine_hand_gesture_confidence=0.81,
        )

    def get_last_attention_debug_details(self):
        return dict(self._last_attention_debug)

    def get_last_gesture_debug_details(self):
        return dict(self._last_gesture_debug)

    def capture_snapshot_png(self, *, width: int, height: int):
        self.snapshot_request = {"width": width, "height": height}
        return _PNG_BYTES

    def _load_detection_runtime(self):
        return {}

    def _probe_online(self, runtime):
        return None

    def _capture_detection(self, runtime, *, observed_at: float):
        del runtime, observed_at
        self.bundle_calls += 1
        return SimpleNamespace(
            person_count=1,
            primary_person_box={
                "top": 0.1,
                "left": 0.2,
                "bottom": 0.9,
                "right": 0.8,
            },
            primary_person_zone="center",
            visible_persons=(),
            person_near_device=True,
            hand_or_object_near_camera=False,
            objects=(),
        )

    def _capture_rgb_frame(self, runtime, *, observed_at: float):
        del runtime, observed_at
        import numpy as np

        return np.full((2, 2, 3), 255, dtype=np.uint8)

    def _classify_error(self, exc):
        return f"classified_{type(exc).__name__.lower()}"


class PeerAICameraObservationProxyTests(unittest.TestCase):
    def test_observation_payload_serializes_camera_dataclass(self) -> None:
        service = _MODULE.AICameraObservationProxyService(adapter=_FakeAdapter())

        payload = service.observe_payload()

        self.assertEqual(payload["observation"]["person_count"], 1)
        self.assertEqual(payload["captured_at"], 10.0)

    def test_attention_handler_returns_json_payload(self) -> None:
        service = _MODULE.AICameraObservationProxyService(adapter=_FakeAdapter())
        handler_class = _MODULE.build_handler(service)
        written = bytearray()
        headers: list[tuple[str, str]] = []

        handler = handler_class.__new__(handler_class)
        handler.path = "/observe_attention"
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
        self.assertIn(("Content-Type", "application/json; charset=utf-8"), headers)
        body = written.decode("utf-8")
        self.assertIn('"person_count": 1', body)
        self.assertIn('"attention_source": "face_anchor"', body)

    def test_handler_returns_not_found_for_unknown_route(self) -> None:
        service = _MODULE.AICameraObservationProxyService(adapter=_FakeAdapter())
        handler_class = _MODULE.build_handler(service)
        written = bytearray()
        headers: list[tuple[str, str]] = []

        handler = handler_class.__new__(handler_class)
        handler.path = "/nope"
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

        self.assertIn(("status", "404"), headers)
        self.assertEqual(written.decode("utf-8"), "not found")

    def test_snapshot_handler_returns_png_bytes(self) -> None:
        adapter = _FakeAdapter()
        service = _MODULE.AICameraObservationProxyService(adapter=adapter)
        handler_class = _MODULE.build_handler(service)
        written = bytearray()
        headers: list[tuple[str, str]] = []

        handler = handler_class.__new__(handler_class)
        handler.path = "/snapshot.png?width=320&height=240&timeout_ms=4000"
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
        self.assertIn(("Content-Type", "image/png"), headers)
        self.assertEqual(bytes(written), _PNG_BYTES)
        self.assertEqual(adapter.snapshot_request, {"width": 320, "height": 240})

    def test_frame_bundle_handler_returns_detection_and_png_payload(self) -> None:
        adapter = _FakeAdapter()
        service = _MODULE.AICameraObservationProxyService(adapter=adapter)
        handler_class = _MODULE.build_handler(service)
        written = bytearray()
        headers: list[tuple[str, str]] = []

        handler = handler_class.__new__(handler_class)
        handler.path = "/observe_frame_bundle?width=320&height=240&timeout_ms=4000"
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
        self.assertIn(("Content-Type", "application/json; charset=utf-8"), headers)
        body = written.decode("utf-8")
        self.assertIn('"person_count": 1', body)
        self.assertIn('"frame_png_base64"', body)
        self.assertEqual(adapter.bundle_calls, 1)

    def test_attention_payload_reuses_cached_observation_while_busy(self) -> None:
        adapter = _FakeAdapter()
        service = _MODULE.AICameraObservationProxyService(adapter=adapter)

        first_payload = service.observe_attention_payload()
        self.assertEqual(first_payload["observation"]["person_count"], 1)
        self.assertEqual(adapter.attention_calls, 1)

        entered = threading.Event()
        release = threading.Event()

        def hold_lock() -> None:
            with service._request_lock:  # pylint: disable=protected-access
                entered.set()
                release.wait(timeout=2.0)

        holder = threading.Thread(target=hold_lock)
        holder.start()
        self.assertTrue(entered.wait(timeout=1.0))
        try:
            cached_payload = service.observe_attention_payload()
        finally:
            release.set()
            holder.join(timeout=1.0)

        self.assertEqual(adapter.attention_calls, 1)
        self.assertEqual(cached_payload["observation"]["person_count"], 1)
        self.assertEqual(cached_payload["cache_state"], "busy_reused")

    def test_gesture_payload_reuses_cached_observation_while_busy(self) -> None:
        adapter = _FakeAdapter()
        service = _MODULE.AICameraObservationProxyService(adapter=adapter)

        first_payload = service.observe_gesture_payload()
        self.assertEqual(
            first_payload["observation"]["fine_hand_gesture"],
            AICameraFineHandGesture.PEACE_SIGN.value,
        )
        self.assertEqual(adapter.gesture_calls, 1)

        entered = threading.Event()
        release = threading.Event()

        def hold_lock() -> None:
            with service._request_lock:  # pylint: disable=protected-access
                entered.set()
                release.wait(timeout=2.0)

        holder = threading.Thread(target=hold_lock)
        holder.start()
        self.assertTrue(entered.wait(timeout=1.0))
        try:
            cached_payload = service.observe_gesture_payload()
        finally:
            release.set()
            holder.join(timeout=1.0)

        self.assertEqual(adapter.gesture_calls, 1)
        self.assertEqual(cached_payload["observation"]["fine_hand_gesture"], "peace_sign")
        self.assertEqual(cached_payload["cache_state"], "busy_reused")

    def test_frame_bundle_reuses_cached_payload_while_busy(self) -> None:
        adapter = _FakeAdapter()
        service = _MODULE.AICameraObservationProxyService(adapter=adapter)

        first_payload = service.observe_frame_bundle_payload(width=320, height=240)
        self.assertEqual(first_payload["observation"]["person_count"], 1)
        self.assertEqual(adapter.bundle_calls, 1)

        entered = threading.Event()
        release = threading.Event()

        def hold_lock() -> None:
            with service._request_lock:  # pylint: disable=protected-access
                entered.set()
                release.wait(timeout=2.0)

        holder = threading.Thread(target=hold_lock)
        holder.start()
        self.assertTrue(entered.wait(timeout=1.0))
        try:
            cached_payload = service.observe_frame_bundle_payload(width=320, height=240)
        finally:
            release.set()
            holder.join(timeout=1.0)

        self.assertEqual(adapter.bundle_calls, 1)
        self.assertEqual(cached_payload["observation"]["person_count"], 1)
        self.assertEqual(cached_payload["cache_state"], "busy_reused")

    def test_frame_bundle_returns_busy_timeout_after_stale_cache_expires(self) -> None:
        adapter = _FakeAdapter()
        service = _MODULE.AICameraObservationProxyService(
            adapter=adapter,
            busy_cache_max_age_s=0.05,
        )

        first_payload = service.observe_frame_bundle_payload(width=320, height=240)
        self.assertEqual(first_payload["observation"]["person_count"], 1)
        self.assertEqual(adapter.bundle_calls, 1)

        entered = threading.Event()
        release = threading.Event()

        def hold_lock() -> None:
            with service._request_lock:  # pylint: disable=protected-access
                entered.set()
                release.wait(timeout=2.0)

        holder = threading.Thread(target=hold_lock)
        holder.start()
        self.assertTrue(entered.wait(timeout=1.0))
        try:
            import time

            time.sleep(0.08)
            timed_out_payload = service.observe_frame_bundle_payload(
                width=320,
                height=240,
                timeout_ms=250,
            )
        finally:
            release.set()
            holder.join(timeout=1.0)

        self.assertEqual(adapter.bundle_calls, 1)
        self.assertFalse(timed_out_payload["observation"]["camera_ready"])
        self.assertEqual(
            timed_out_payload["observation"]["camera_error"],
            "camera_proxy_busy_timeout",
        )
        self.assertEqual(
            timed_out_payload["debug_details"]["pipeline_error"],
            "camera_proxy_busy_timeout",
        )

    def test_snapshot_handler_returns_gateway_timeout_when_helper_stays_busy(self) -> None:
        adapter = _FakeAdapter()
        service = _MODULE.AICameraObservationProxyService(adapter=adapter)
        handler_class = _MODULE.build_handler(service)
        written = bytearray()
        headers: list[tuple[str, str]] = []

        entered = threading.Event()
        release = threading.Event()

        def hold_lock() -> None:
            with service._request_lock:  # pylint: disable=protected-access
                entered.set()
                release.wait(timeout=2.0)

        holder = threading.Thread(target=hold_lock)
        holder.start()
        self.assertTrue(entered.wait(timeout=1.0))
        try:
            handler = handler_class.__new__(handler_class)
            handler.path = "/snapshot.png?width=320&height=240&timeout_ms=250"
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
        finally:
            release.set()
            holder.join(timeout=1.0)

        self.assertIn(("status", "504"), headers)
        self.assertEqual(written.decode("utf-8"), "camera_proxy_busy_timeout")


if __name__ == "__main__":
    unittest.main()
