"""Regression coverage for Twinr's on-device Crazyflie failsafe adapter."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any
import unittest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "hardware" / "bitcraze" / "on_device_failsafe.py"
_SPEC = importlib.util.spec_from_file_location("bitcraze_on_device_failsafe_module", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE: Any = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class OnDeviceFailsafeTests(unittest.TestCase):
    def test_build_heartbeat_packet_encodes_expected_thresholds(self) -> None:
        packet = _MODULE.build_on_device_failsafe_heartbeat_packet(
            _MODULE.OnDeviceFailsafeConfig(
                heartbeat_timeout_s=0.42,
                low_battery_v=3.61,
                critical_battery_v=3.33,
                min_clearance_m=0.37,
                min_up_clearance_m=0.21,
                descent_rate_mps=0.11,
                max_repel_velocity_mps=0.14,
                brake_hold_s=0.18,
            ),
            session_id=123,
        )

        unpacked = _MODULE._HEARTBEAT_STRUCT.unpack(packet)
        self.assertEqual(unpacked[0], _MODULE.ON_DEVICE_FAILSAFE_PROTOCOL_VERSION)
        self.assertEqual(unpacked[1], _MODULE.ON_DEVICE_FAILSAFE_PACKET_KIND_HEARTBEAT)
        self.assertEqual(unpacked[4], 123)
        self.assertEqual(unpacked[5], 420)
        self.assertEqual(unpacked[6], 3610)
        self.assertEqual(unpacked[7], 3330)
        self.assertEqual(unpacked[8], 370)
        self.assertEqual(unpacked[9], 210)
        self.assertEqual(unpacked[10], 110)
        self.assertEqual(unpacked[11], 140)
        self.assertEqual(unpacked[12], 180)

    def test_parse_status_packet_returns_named_status(self) -> None:
        payload = _MODULE._STATUS_STRUCT.pack(
            _MODULE.ON_DEVICE_FAILSAFE_PROTOCOL_VERSION,
            _MODULE.ON_DEVICE_FAILSAFE_PACKET_KIND_STATUS,
            3,
            1,
            55,
            320,
            3470,
            180,
            42,
        )

        status = _MODULE.parse_on_device_failsafe_status_packet(payload)

        self.assertIsNotNone(status)
        self.assertEqual(status.session_id, 55)
        self.assertEqual(status.state_name, "failsafe_descend")
        self.assertEqual(status.reason_name, "heartbeat_loss")
        self.assertEqual(status.vbat_mv, 3470)

    def test_probe_on_device_failsafe_reports_loaded_state(self) -> None:
        class _FakeParam:
            def get_value(self, name: str) -> str:
                values = {
                    "twinrFs.protocolVersion": "1",
                    "twinrFs.enable": "1",
                    "twinrFs.state": "2",
                    "twinrFs.reason": "4",
                }
                return values[name]

        class _FakeCF:
            def __init__(self) -> None:
                self.param = _FakeParam()

        availability = _MODULE.probe_on_device_failsafe(_FakeCF())

        self.assertTrue(availability.loaded)
        self.assertEqual(availability.protocol_version, 1)
        self.assertEqual(availability.state_name, "failsafe_brake")
        self.assertEqual(availability.reason_name, "clearance")

    def test_heartbeat_session_sends_packets_and_records_status(self) -> None:
        packets: list[bytes] = []

        class _FakeCaller:
            def __init__(self) -> None:
                self.callbacks: list[Any] = []

            def add_callback(self, callback: Any) -> None:
                self.callbacks.append(callback)

            def remove_callback(self, callback: Any) -> None:
                self.callbacks.remove(callback)

            def emit(self, payload: bytes) -> None:
                for callback in list(self.callbacks):
                    callback(payload)

        class _FakeAppchannel:
            def __init__(self) -> None:
                self.packet_received = _FakeCaller()

            def send_packet(self, payload: bytes) -> None:
                packets.append(payload)

        class _FakeCF:
            def __init__(self) -> None:
                self.appchannel = _FakeAppchannel()

        availability = _MODULE.OnDeviceFailsafeAvailability(
            loaded=True,
            protocol_version=1,
            enabled=1,
            state_code=1,
            state_name="monitoring",
            reason_code=0,
            reason_name="none",
            failures=(),
        )
        session = _MODULE.OnDeviceFailsafeHeartbeatSession(
            _FakeCF(),
            mode="required",
            config=_MODULE.OnDeviceFailsafeConfig(heartbeat_period_s=0.05),
            availability=availability,
            session_id=77,
        )
        session.start()
        session._cf.appchannel.packet_received.emit(
            _MODULE._STATUS_STRUCT.pack(
                _MODULE.ON_DEVICE_FAILSAFE_PROTOCOL_VERSION,
                _MODULE.ON_DEVICE_FAILSAFE_PACKET_KIND_STATUS,
                2,
                1,
                77,
                150,
                3520,
                210,
                35,
            )
        )
        session.close(disable=True)
        report = session.report()

        self.assertGreaterEqual(len(packets), 2)
        self.assertTrue(report.started)
        self.assertTrue(report.disabled_cleanly)
        self.assertEqual(report.status_packets_received, 1)
        self.assertIsNotNone(report.last_status)
        self.assertEqual(report.last_status.reason_name, "heartbeat_loss")


if __name__ == "__main__":
    unittest.main()
