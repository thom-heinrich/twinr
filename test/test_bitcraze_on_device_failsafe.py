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
    def test_emit_trace_surfaces_writer_errors(self) -> None:
        class _BrokenTraceWriter:
            def emit(self, *_args: Any, **_kwargs: Any) -> None:
                raise RuntimeError("trace boom")

        with self.assertRaisesRegex(RuntimeError, "trace boom"):
            _MODULE._emit_trace(_BrokenTraceWriter(), "phase", status="ok")

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
        self.assertEqual(
            unpacked[2],
            _MODULE.ON_DEVICE_FAILSAFE_FLAG_ENABLE
            | _MODULE.ON_DEVICE_FAILSAFE_FLAG_REQUIRE_CLEARANCE
            | _MODULE.ON_DEVICE_FAILSAFE_FLAG_ARM_LATERAL_CLEARANCE,
        )
        self.assertEqual(unpacked[4], 123)
        self.assertEqual(unpacked[5], 420)
        self.assertEqual(unpacked[6], 3610)
        self.assertEqual(unpacked[7], 3330)
        self.assertEqual(unpacked[8], 370)
        self.assertEqual(unpacked[9], 210)
        self.assertEqual(unpacked[10], 110)
        self.assertEqual(unpacked[11], 140)
        self.assertEqual(unpacked[12], 180)

    def test_build_heartbeat_packet_can_leave_lateral_clearance_disarmed(self) -> None:
        packet = _MODULE.build_on_device_failsafe_heartbeat_packet(
            _MODULE.OnDeviceFailsafeConfig(arm_lateral_clearance=False),
            session_id=9,
        )

        unpacked = _MODULE._HEARTBEAT_STRUCT.unpack(packet)
        self.assertEqual(
            unpacked[2],
            _MODULE.ON_DEVICE_FAILSAFE_FLAG_ENABLE | _MODULE.ON_DEVICE_FAILSAFE_FLAG_REQUIRE_CLEARANCE,
        )

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

    def test_parse_status_packet_accepts_newer_firmware_surface_version(self) -> None:
        payload = _MODULE._STATUS_STRUCT.pack(
            2,
            _MODULE.ON_DEVICE_FAILSAFE_PACKET_KIND_STATUS,
            1,
            0,
            77,
            25,
            3900,
            180,
            30,
        )

        status = _MODULE.parse_on_device_failsafe_status_packet(payload)

        self.assertIsNotNone(status)
        assert status is not None
        self.assertEqual(status.session_id, 77)
        self.assertEqual(status.state_name, "monitoring")

    def test_parse_status_packet_v3_includes_mission_fields(self) -> None:
        payload = _MODULE._STATUS_STRUCT_V3.pack(
            3,
            _MODULE.ON_DEVICE_FAILSAFE_PACKET_KIND_STATUS,
            7,
            0,
            88,
            40,
            4010,
            350,
            95,
            _MODULE.ON_DEVICE_HOVER_STATE_FLAG_MISSION_ACTIVE
            | _MODULE.ON_DEVICE_HOVER_STATE_FLAG_TAKEOFF_PROVEN,
            0,
            250,
            180,
            205,
        )

        status = _MODULE.parse_on_device_failsafe_status_packet(payload)

        self.assertIsNotNone(status)
        assert status is not None
        self.assertEqual(status.state_name, "mission_hover")
        self.assertEqual(status.mission_flags, 24)
        self.assertEqual(status.target_height_mm, 250)
        self.assertEqual(status.commanded_height_mm, 180)
        self.assertEqual(status.state_estimate_z_mm, 205)
        self.assertIsNone(status.debug_flags)

    def test_parse_status_packet_v4_includes_debug_fields(self) -> None:
        payload = _MODULE._STATUS_STRUCT_V4.pack(
            4,
            _MODULE.ON_DEVICE_FAILSAFE_PACKET_KIND_STATUS,
            7,
            0,
            88,
            40,
            4010,
            350,
            95,
            _MODULE.ON_DEVICE_HOVER_STATE_FLAG_MISSION_ACTIVE
            | _MODULE.ON_DEVICE_HOVER_STATE_FLAG_TAKEOFF_PROVEN,
            0b10101,
            250,
            180,
            205,
            2660,
            73,
            3,
            0,
            511,
        )

        status = _MODULE.parse_on_device_failsafe_status_packet(payload)

        self.assertIsNotNone(status)
        assert status is not None
        self.assertEqual(status.state_name, "mission_hover")
        self.assertEqual(status.mission_flags, 24)
        self.assertEqual(status.debug_flags, 0b10101)
        self.assertEqual(status.up_range_mm, 2660)
        self.assertEqual(status.motion_squal, 73)
        self.assertEqual(status.touchdown_confirm_count, 3)
        self.assertEqual(status.hover_thrust_permille, 511)

    def test_reason_name_maps_takeoff_attitude_quiet(self) -> None:
        self.assertEqual(_MODULE._reason_name(10), "takeoff_attitude_quiet")

    def test_reason_name_maps_truth_stale(self) -> None:
        self.assertEqual(_MODULE._reason_name(11), "truth_stale")

    def test_reason_name_maps_state_flapping(self) -> None:
        self.assertEqual(_MODULE._reason_name(12), "state_flapping")

    def test_reason_name_maps_ceiling_without_progress(self) -> None:
        self.assertEqual(_MODULE._reason_name(13), "ceiling_without_progress")

    def test_reason_name_maps_disturbance_nonrecoverable(self) -> None:
        self.assertEqual(_MODULE._reason_name(14), "disturbance_nonrecoverable")

    def test_reason_name_maps_takeoff_overshoot(self) -> None:
        self.assertEqual(_MODULE._reason_name(15), "takeoff_overshoot")

    def test_build_hover_command_packet_encodes_bounded_hover_intent(self) -> None:
        payload = _MODULE.build_on_device_hover_command_packet(
            _MODULE.OnDeviceHoverIntent(
                target_height_m=0.25,
                hover_duration_s=1.5,
                takeoff_ramp_s=0.8,
                micro_liftoff_height_m=0.08,
                target_tolerance_m=0.05,
            ),
            session_id=41,
            command_kind=_MODULE.ON_DEVICE_HOVER_COMMAND_KIND_START,
        )

        unpacked = _MODULE._COMMAND_STRUCT.unpack(payload)
        self.assertEqual(unpacked[0], _MODULE.ON_DEVICE_FAILSAFE_COMMAND_PROTOCOL_VERSION)
        self.assertEqual(unpacked[1], _MODULE.ON_DEVICE_FAILSAFE_PACKET_KIND_COMMAND)
        self.assertEqual(unpacked[2], _MODULE.ON_DEVICE_HOVER_COMMAND_KIND_START)
        self.assertEqual(unpacked[4], 41)
        self.assertEqual(unpacked[5], 250)
        self.assertEqual(unpacked[6], 1500)
        self.assertEqual(unpacked[7], 800)
        self.assertEqual(unpacked[8], 80)
        self.assertEqual(unpacked[9], 50)

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

    def test_probe_on_device_failsafe_accepts_protocol_v2_surface(self) -> None:
        class _FakeParam:
            def get_value(self, name: str) -> str:
                values = {
                    "twinrFs.protocolVersion": "2",
                    "twinrFs.enable": "1",
                    "twinrFs.state": "1",
                    "twinrFs.reason": "0",
                }
                return values[name]

        class _FakeCF:
            def __init__(self) -> None:
                self.param = _FakeParam()

        availability = _MODULE.probe_on_device_failsafe(_FakeCF())

        self.assertTrue(availability.loaded)
        self.assertEqual(availability.protocol_version, 2)
        self.assertEqual(availability.state_name, "monitoring")
        self.assertEqual(availability.reason_name, "none")

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

    def test_set_lateral_clearance_armed_updates_config_and_sends_immediate_packet(self) -> None:
        packets: list[bytes] = []

        class _FakeCaller:
            def __init__(self) -> None:
                self.callbacks: list[Any] = []

            def add_callback(self, callback: Any) -> None:
                self.callbacks.append(callback)

            def remove_callback(self, callback: Any) -> None:
                self.callbacks.remove(callback)

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
            config=_MODULE.OnDeviceFailsafeConfig(heartbeat_period_s=0.05, arm_lateral_clearance=False),
            availability=availability,
            session_id=91,
        )

        session.start()
        initial_packet_count = len(packets)
        session.set_lateral_clearance_armed(True)

        self.assertTrue(session.config.arm_lateral_clearance)
        self.assertGreater(len(packets), initial_packet_count)
        unpacked = _MODULE._HEARTBEAT_STRUCT.unpack(packets[-1])
        self.assertTrue(unpacked[2] & _MODULE.ON_DEVICE_FAILSAFE_FLAG_ARM_LATERAL_CLEARANCE)
        session.close(disable=True)

    def test_start_bounded_hover_requires_protocol_version_four(self) -> None:
        class _FakeCaller:
            def __init__(self) -> None:
                self.callbacks: list[Any] = []

            def add_callback(self, callback: Any) -> None:
                self.callbacks.append(callback)

            def remove_callback(self, callback: Any) -> None:
                self.callbacks.remove(callback)

        class _FakeAppchannel:
            def __init__(self) -> None:
                self.packet_received = _FakeCaller()

            def send_packet(self, payload: bytes) -> None:
                del payload

        class _FakeCF:
            def __init__(self) -> None:
                self.appchannel = _FakeAppchannel()

        session = _MODULE.OnDeviceFailsafeHeartbeatSession(
            _FakeCF(),
            mode="required",
            config=_MODULE.OnDeviceFailsafeConfig(heartbeat_period_s=0.05),
            availability=_MODULE.OnDeviceFailsafeAvailability(
                loaded=True,
                protocol_version=3,
                enabled=1,
                state_code=1,
                state_name="monitoring",
                reason_code=0,
                reason_name="none",
                failures=(),
            ),
            session_id=19,
        )
        session.start()
        with self.assertRaisesRegex(RuntimeError, "protocol version >= 4"):
            session.start_bounded_hover(
                _MODULE.OnDeviceHoverIntent(target_height_m=0.10, hover_duration_s=1.0)
            )
        session.close(disable=True)

    def test_wait_for_bounded_hover_result_tracks_mission_lifecycle(self) -> None:
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

        session = _MODULE.OnDeviceFailsafeHeartbeatSession(
            _FakeCF(),
            mode="required",
            config=_MODULE.OnDeviceFailsafeConfig(heartbeat_period_s=0.05),
            availability=_MODULE.OnDeviceFailsafeAvailability(
                loaded=True,
                protocol_version=4,
                enabled=1,
                state_code=1,
                state_name="monitoring",
                reason_code=0,
                reason_name="none",
                failures=(),
            ),
            session_id=77,
        )
        session.start()
        intent = _MODULE.OnDeviceHoverIntent(target_height_m=0.10, hover_duration_s=1.0)
        session.start_bounded_hover(intent)
        self.assertEqual(
            _MODULE._COMMAND_STRUCT.unpack(packets[-1])[2],
            _MODULE.ON_DEVICE_HOVER_COMMAND_KIND_START,
        )
        for state_code in (6, 7, 8, 9):
            session._cf.appchannel.packet_received.emit(
                _MODULE._STATUS_STRUCT_V4.pack(
                    4,
                    _MODULE.ON_DEVICE_FAILSAFE_PACKET_KIND_STATUS,
                    state_code,
                    0,
                    77,
                    20,
                    3980,
                    320,
                    30,
                    _MODULE.ON_DEVICE_HOVER_STATE_FLAG_MISSION_ACTIVE,
                    0,
                    100,
                    100,
                    100,
                    2600,
                    90,
                    0,
                    0,
                    500,
                )
            )
        result = session.wait_for_bounded_hover_result(timeout_s=0.2)
        session.close(disable=True)

        self.assertTrue(result.took_off)
        self.assertTrue(result.qualified_hover_reached)
        self.assertTrue(result.landing_reached)
        self.assertTrue(result.landed)
        self.assertEqual(result.final_status.state_name, "mission_complete")
        self.assertEqual(
            result.observed_state_names,
            ("mission_takeoff", "mission_hover", "mission_landing", "mission_complete"),
        )


if __name__ == "__main__":
    unittest.main()
