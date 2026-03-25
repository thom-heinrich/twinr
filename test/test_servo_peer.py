from pathlib import Path
import json
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.servo_peer import PeerPololuMaestroServoPulseWriter


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = json.dumps(payload).encode("utf-8")
        self.headers = self

    def get_content_charset(self, default: str = "utf-8") -> str:
        return default

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        return None


class PeerPololuMaestroServoPulseWriterTests(unittest.TestCase):
    def test_write_posts_channel_and_pulse_width(self) -> None:
        requests: list[tuple[str, bytes | None]] = []

        def fake_urlopen(request, timeout: float):
            del timeout
            requests.append((request.full_url, request.data))
            return _FakeHTTPResponse({"ok": True})

        writer = PeerPololuMaestroServoPulseWriter(base_url="http://10.42.0.2:8768/")

        with mock.patch("twinr.hardware.servo_peer.urlopen", side_effect=fake_urlopen):
            writer.write(gpio_chip="gpiochip0", gpio=1, pulse_width_us=1500)

        self.assertEqual(requests, [("http://10.42.0.2:8768/servo/write", b'{"channel": 1, "pulse_width_us": 1500}')])

    def test_current_pulse_width_reads_remote_json_payload(self) -> None:
        def fake_urlopen(request, timeout: float):
            del timeout
            self.assertEqual(request.full_url, "http://10.42.0.2:8768/servo/position?channel=1")
            return _FakeHTTPResponse({"ok": True, "pulse_width_us": 1490})

        writer = PeerPololuMaestroServoPulseWriter(base_url="http://10.42.0.2:8768")

        with mock.patch("twinr.hardware.servo_peer.urlopen", side_effect=fake_urlopen):
            pulse_width_us = writer.current_pulse_width_us(gpio_chip="gpiochip0", gpio=1)

        self.assertEqual(pulse_width_us, 1490)

    def test_probe_raises_clear_error_for_remote_failure_payload(self) -> None:
        def fake_urlopen(request, timeout: float):
            del request, timeout
            return _FakeHTTPResponse({"ok": False, "error": "maestro_missing"})

        writer = PeerPololuMaestroServoPulseWriter(base_url="http://10.42.0.2:8768")

        with (
            mock.patch("twinr.hardware.servo_peer.urlopen", side_effect=fake_urlopen),
            self.assertRaisesRegex(RuntimeError, "maestro_missing"),
        ):
            writer.probe(1)

    def test_constructor_requires_base_url(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "TWINR_ATTENTION_SERVO_PEER_BASE_URL"):
            PeerPololuMaestroServoPulseWriter(base_url="  ")
