"""Regression coverage for the shared Crazyflie runtime telemetry lane."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.crazyflie_telemetry import (
    CrazyflieTelemetryRuntime,
    TelemetryProfile,
    profile_log_blocks,
)


class _FakeCaller:
    def __init__(self) -> None:
        self.callbacks: list[object] = []

    def add_callback(self, callback: object) -> None:
        self.callbacks.append(callback)

    def remove_callback(self, callback: object) -> None:
        self.callbacks.remove(callback)


class _FakeLinkStatistics:
    def __init__(self) -> None:
        self._is_active = False
        self.latency_updated = _FakeCaller()
        self.link_quality_updated = _FakeCaller()
        self.uplink_rssi_updated = _FakeCaller()
        self.uplink_rate_updated = _FakeCaller()
        self.downlink_rate_updated = _FakeCaller()
        self.uplink_congestion_updated = _FakeCaller()
        self.downlink_congestion_updated = _FakeCaller()

    def start(self) -> None:
        self._is_active = True

    def stop(self) -> None:
        self._is_active = False


class _FakeLogConfig:
    def __init__(self, *, name: str, period_in_ms: int) -> None:
        self.name = name
        self.period_in_ms = period_in_ms
        self.variables: list[str] = []
        self.data_received_cb = _FakeCaller()
        self.started = False

    def add_variable(self, variable_name: str) -> None:
        self.variables.append(variable_name)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def delete(self) -> None:
        return None


class _FakeLog:
    def __init__(self) -> None:
        self.toc = SimpleNamespace(
            toc={
                "pm": {"vbat": {}, "batteryLevel": {}, "state": {}},
                "radio": {"isConnected": {}, "rssi": {}},
                "supervisor": {"info": {}},
                "range": {"zrange": {}, "front": {}, "back": {}, "left": {}, "right": {}, "up": {}},
                "motion": {"squal": {}, "deltaX": {}, "deltaY": {}},
                "stateEstimate": {"x": {}, "y": {}, "z": {}, "vx": {}, "vy": {}, "vz": {}},
                "stabilizer": {"roll": {}, "pitch": {}, "yaw": {}, "thrust": {}},
                "gyro": {"x": {}, "y": {}, "z": {}},
                "twinrFs": {
                    "state": {},
                    "reason": {},
                    "heartbeatAgeMs": {},
                    "rejectedPkts": {},
                    "downRangeMm": {},
                    "tkDbg": {},
                    "tkRfCnt": {},
                    "tkRrCnt": {},
                    "tkFlCnt": {},
                    "tkAtCnt": {},
                    "tkStCnt": {},
                    "tkFpCnt": {},
                    "tkPgCls": {},
                    "tkBatMv": {},
                    "thrEst": {},
                    "cmdVx": {},
                    "cmdVy": {},
                    "cmdSrc": {},
                    "distVx": {},
                    "distVy": {},
                    "distSev": {},
                    "distRec": {},
                    "lastRejectCode": {},
                },
                "kalman": {"varPX": {}, "varPY": {}, "varPZ": {}, "statePX": {}, "statePY": {}, "statePZ": {}},
            }
        )
        self.configs: list[_FakeLogConfig] = []

    def add_config(self, config: _FakeLogConfig) -> None:
        self.configs.append(config)


class _FakeParam:
    def __init__(self, values: dict[str, object]) -> None:
        self._values = values
        self.toc = SimpleNamespace(
            toc={
                "deck": {"bcFlow2": {}, "bcZRanger2": {}, "bcMultiranger": {}, "bcAI": {}},
                "twinrFs": {"protocolVersion": {}, "enable": {}},
                "motion": {"disable": {}},
                "stabilizer": {"controller": {}, "estimator": {}},
                "supervisor": {"tmblChckEn": {}},
            }
        )

    def get_value(self, key: str) -> object:
        return self._values[key]


class _FakeCF:
    def __init__(self) -> None:
        self.log = _FakeLog()
        self.param = _FakeParam(
            {
                "deck.bcFlow2": 1,
                "deck.bcZRanger2": 1,
                "deck.bcMultiranger": 1,
                "deck.bcAI": 1,
                "twinrFs.protocolVersion": 7,
                "twinrFs.enable": 1,
                "motion.disable": 0,
                "stabilizer.controller": 1,
                "stabilizer.estimator": 2,
                "supervisor.tmblChckEn": 1,
            }
        )
        self.link_statistics = _FakeLinkStatistics()


class _FakeSyncCrazyflie:
    def __init__(self) -> None:
        self.cf = _FakeCF()


class CrazyflieTelemetryRuntimeTests(unittest.TestCase):
    def test_hover_acceptance_profile_keeps_the_shared_failsafe_block(self) -> None:
        block_names = tuple((block.name for block in profile_log_blocks(TelemetryProfile.HOVER_ACCEPTANCE)))
        self.assertIn("hover-failsafe", block_names)

    def test_runtime_snapshot_exposes_typed_live_state(self) -> None:
        clock = [0.0]
        runtime = CrazyflieTelemetryRuntime(
            _FakeSyncCrazyflie(),
            _FakeLogConfig,
            profile=TelemetryProfile.HOVER_ACCEPTANCE,
            max_samples=32,
            monotonic=lambda: clock[0],
        )
        runtime.start()
        try:
            clock[0] = 0.2
            runtime._record_sample(200, {  # type: ignore[attr-defined]
                "pm.vbat": 4.05,
                "pm.batteryLevel": 92,
                "pm.state": 0,
                "radio.isConnected": 1,
                "radio.rssi": -41.0,
                "supervisor.info": 8,
                "range.zrange": 260.0,
                "range.front": 900.0,
                "motion.squal": 85,
                "stateEstimate.z": 0.24,
                "stabilizer.roll": 0.2,
                "stabilizer.pitch": -0.1,
                "stabilizer.yaw": 5.0,
                "stateEstimate.x": 0.01,
                "stateEstimate.y": -0.02,
                "stateEstimate.vx": 0.0,
                "stateEstimate.vy": 0.0,
                "stateEstimate.vz": 0.0,
                "stabilizer.thrust": 32000,
                "twinrFs.state": 2,
                "twinrFs.reason": 0,
                "twinrFs.heartbeatAgeMs": 40,
                "twinrFs.rejectedPkts": 0,
                "twinrFs.downRangeMm": 260,
                "twinrFs.tkDbg": 0x83,
                "twinrFs.tkRfCnt": 2,
                "twinrFs.tkRrCnt": 2,
                "twinrFs.tkFlCnt": 2,
                "twinrFs.tkAtCnt": 3,
                "twinrFs.tkStCnt": 1,
                "twinrFs.tkFpCnt": 4,
                "twinrFs.tkPgCls": 2,
                "twinrFs.tkBatMv": 4012.0,
                "twinrFs.thrEst": 0.48,
                "twinrFs.cmdVx": 0.0,
                "twinrFs.cmdVy": 0.0,
                "twinrFs.cmdSrc": 1,
                "twinrFs.distVx": 0.03,
                "twinrFs.distVy": -0.02,
                "twinrFs.distSev": 420,
                "twinrFs.distRec": 1,
            }, SimpleNamespace(name="hover-sensors"))
            runtime.emit(
                "hover_primitive_takeoff",
                status="begin",
                data={"target_height_m": 0.25, "velocity_mps": 0.2},
            )
            runtime.emit(
                "hover_primitive_takeoff_confirm",
                status="done",
                data={"observed_height_m": 0.24},
            )
            runtime.emit(
                "on_device_failsafe_status",
                status="done",
                data={"state": 2, "state_name": "armed", "reason": 0, "reason_name": "ok", "session_id": 7},
            )

            snapshot = runtime.latest_snapshot()
        finally:
            runtime.stop()

        self.assertTrue(snapshot.healthy)
        self.assertEqual(snapshot.deck.flags["bcAI"], 1)
        self.assertEqual(snapshot.power.battery_level, 92)
        self.assertEqual(snapshot.range.zrange_m, 0.26)
        self.assertEqual(snapshot.link.radio_connected, True)
        self.assertEqual(snapshot.failsafe.protocol_version, 7)
        self.assertEqual(snapshot.failsafe.state_name, "armed")
        self.assertEqual(snapshot.failsafe.takeoff_debug_flags, 0x83)
        self.assertEqual(snapshot.failsafe.takeoff_range_fresh_count, 2)
        self.assertEqual(snapshot.failsafe.takeoff_range_rise_count, 2)
        self.assertEqual(snapshot.failsafe.takeoff_flow_live_count, 2)
        self.assertEqual(snapshot.failsafe.takeoff_attitude_quiet_count, 3)
        self.assertEqual(snapshot.failsafe.takeoff_truth_stale_count, 1)
        self.assertEqual(snapshot.failsafe.takeoff_truth_flap_count, 4)
        self.assertEqual(snapshot.failsafe.takeoff_progress_class, 2)
        self.assertEqual(snapshot.failsafe.filtered_battery_mv, 4012.0)
        self.assertEqual(snapshot.failsafe.hover_thrust_estimate, 0.48)
        self.assertEqual(snapshot.failsafe.commanded_vx_mps, 0.0)
        self.assertEqual(snapshot.failsafe.commanded_vy_mps, 0.0)
        self.assertEqual(snapshot.failsafe.lateral_command_source_code, 1)
        self.assertEqual(snapshot.failsafe.disturbance_estimate_vx, 0.03)
        self.assertEqual(snapshot.failsafe.disturbance_estimate_vy, -0.02)
        self.assertEqual(snapshot.failsafe.disturbance_severity_permille, 420)
        self.assertEqual(snapshot.failsafe.disturbance_recoverable, True)
        self.assertEqual(snapshot.command.phase, "takeoff")
        self.assertTrue(snapshot.command.takeoff_confirmed)

    def test_runtime_flags_takeoff_not_achieved_from_live_observation(self) -> None:
        clock = [0.0]
        runtime = CrazyflieTelemetryRuntime(
            _FakeSyncCrazyflie(),
            _FakeLogConfig,
            profile=TelemetryProfile.HOVER_ACCEPTANCE,
            max_samples=16,
            monotonic=lambda: clock[0],
        )
        runtime.start()
        try:
            runtime.emit(
                "hover_primitive_takeoff",
                status="begin",
                data={"target_height_m": 0.25, "velocity_mps": 0.2},
            )
            clock[0] = 0.6
            runtime._record_sample(600, {  # type: ignore[attr-defined]
                "pm.vbat": 4.02,
                "radio.isConnected": 1,
                "supervisor.info": 8,
                "range.zrange": 8.0,
                "motion.squal": 80,
            }, SimpleNamespace(name="hover-sensors"))
            clock[0] = 1.7
            snapshot = runtime.latest_snapshot()
        finally:
            runtime.stop()

        divergence_codes = {item.code for item in snapshot.divergences}
        self.assertIn("takeoff_not_achieved", divergence_codes)

    def test_runtime_flags_failsafe_trigger_before_liftoff(self) -> None:
        runtime = CrazyflieTelemetryRuntime(
            _FakeSyncCrazyflie(),
            _FakeLogConfig,
            profile=TelemetryProfile.HOVER_ACCEPTANCE,
            max_samples=16,
        )
        runtime.start()
        try:
            runtime.emit(
                "hover_primitive_takeoff",
                status="begin",
                data={"target_height_m": 0.25, "velocity_mps": 0.2},
            )
            runtime.emit(
                "on_device_failsafe_status",
                status="done",
                data={
                    "state": 2,
                    "state_name": "failsafe_brake",
                    "reason": 4,
                    "reason_name": "clearance",
                    "session_id": 7,
                    "down_range_mm": 21,
                },
            )
            snapshot = runtime.latest_snapshot()
        finally:
            runtime.stop()

        divergence_codes = {item.code for item in snapshot.divergences}
        self.assertIn("failsafe_triggered_before_liftoff", divergence_codes)
        self.assertNotIn("failsafe_retrigger_loop", divergence_codes)

    def test_runtime_flags_retrigger_loop_before_liftoff(self) -> None:
        runtime = CrazyflieTelemetryRuntime(
            _FakeSyncCrazyflie(),
            _FakeLogConfig,
            profile=TelemetryProfile.HOVER_ACCEPTANCE,
            max_samples=16,
        )
        runtime.start()
        try:
            runtime.emit(
                "hover_primitive_takeoff",
                status="begin",
                data={"target_height_m": 0.25, "velocity_mps": 0.2},
            )
            for down_range_mm in (21, 23):
                runtime.emit(
                    "on_device_failsafe_status",
                    status="done",
                    data={
                        "state": 2,
                        "state_name": "failsafe_brake",
                        "reason": 4,
                        "reason_name": "clearance",
                        "session_id": 7,
                        "down_range_mm": down_range_mm,
                    },
                )
            snapshot = runtime.latest_snapshot()
        finally:
            runtime.stop()

        divergence_codes = {item.code for item in snapshot.divergences}
        self.assertIn("failsafe_triggered_before_liftoff", divergence_codes)
        self.assertIn("failsafe_retrigger_loop", divergence_codes)

    def test_runtime_flags_takeoff_conflict_when_host_confirms_after_pre_liftoff_failsafe(self) -> None:
        runtime = CrazyflieTelemetryRuntime(
            _FakeSyncCrazyflie(),
            _FakeLogConfig,
            profile=TelemetryProfile.HOVER_ACCEPTANCE,
            max_samples=16,
        )
        runtime.start()
        try:
            runtime.emit(
                "hover_primitive_takeoff",
                status="begin",
                data={"target_height_m": 0.25, "velocity_mps": 0.2},
            )
            runtime.emit(
                "on_device_failsafe_status",
                status="done",
                data={
                    "state": 2,
                    "state_name": "failsafe_brake",
                    "reason": 4,
                    "reason_name": "clearance",
                    "session_id": 7,
                    "down_range_mm": 21,
                },
            )
            runtime.emit(
                "hover_primitive_takeoff_confirm",
                status="done",
                data={"observed_height_m": 0.095},
            )
            snapshot = runtime.latest_snapshot()
        finally:
            runtime.stop()

        divergence_codes = {item.code for item in snapshot.divergences}
        self.assertIn("takeoff_conflict_with_failsafe", divergence_codes)

    def test_runtime_flags_hover_guard_divergences(self) -> None:
        runtime = CrazyflieTelemetryRuntime(
            _FakeSyncCrazyflie(),
            _FakeLogConfig,
            profile=TelemetryProfile.HOVER_ACCEPTANCE,
            max_samples=16,
        )
        runtime.start()
        try:
            runtime.emit(
                "hover_primitive_stability_guard",
                status="blocked",
                data={
                    "phase": "hold",
                    "failure_codes": ("roll_pitch", "speed", "xy_drift", "height_not_held"),
                    "failures": ("roll too high", "speed too high", "drift too high", "height missing"),
                },
            )
            snapshot = runtime.latest_snapshot()
        finally:
            runtime.stop()

        divergence_codes = {item.code for item in snapshot.divergences}
        self.assertIn("hover_guard_roll_pitch_exceeded", divergence_codes)
        self.assertIn("hover_guard_speed_exceeded", divergence_codes)
        self.assertIn("hover_guard_xy_drift_exceeded", divergence_codes)
        self.assertIn("hover_guard_height_not_held", divergence_codes)

    def test_runtime_flags_untrusted_height_and_flow_guard_divergences(self) -> None:
        runtime = CrazyflieTelemetryRuntime(
            _FakeSyncCrazyflie(),
            _FakeLogConfig,
            profile=TelemetryProfile.HOVER_ACCEPTANCE,
            max_samples=16,
        )
        runtime.start()
        try:
            runtime.emit(
                "hover_primitive_stability_guard",
                status="blocked",
                data={
                    "phase": "stabilize",
                    "failure_codes": ("height_untrusted", "flow_untrusted", "anchor_control"),
                    "failures": (
                        "trusted height is unavailable because downward range and state estimate disagree by 0.76 m",
                        "optical-flow quality 3 is below the 30 stability floor",
                        "flow anchor control is unavailable because no fresh anchor pose is available",
                    ),
                },
            )
            snapshot = runtime.latest_snapshot()
        finally:
            runtime.stop()

        divergence_codes = {item.code for item in snapshot.divergences}
        self.assertIn("hover_guard_height_untrusted", divergence_codes)
        self.assertIn("hover_guard_flow_untrusted", divergence_codes)
        self.assertIn("hover_guard_anchor_control_unavailable", divergence_codes)

    def test_runtime_flags_forced_cutoff_touchdown_divergence(self) -> None:
        runtime = CrazyflieTelemetryRuntime(
            _FakeSyncCrazyflie(),
            _FakeLogConfig,
            profile=TelemetryProfile.HOVER_ACCEPTANCE,
            max_samples=16,
        )
        runtime.start()
        try:
            runtime.emit(
                "hover_primitive_touchdown_confirm",
                status="degraded",
                data={
                    "confirmation_source": "timeout_forced_cutoff",
                    "last_distance_m": 0.09,
                    "last_is_flying": True,
                },
            )
            snapshot = runtime.latest_snapshot()
        finally:
            runtime.stop()

        divergence_codes = {item.code for item in snapshot.divergences}
        self.assertIn("touchdown_not_confirmed", divergence_codes)
        self.assertIn("touchdown_range_only_without_supervisor_ground", divergence_codes)

    def test_runtime_does_not_flag_explicit_sitl_range_only_touchdown_as_error(self) -> None:
        runtime = CrazyflieTelemetryRuntime(
            _FakeSyncCrazyflie(),
            _FakeLogConfig,
            profile=TelemetryProfile.HOVER_ACCEPTANCE_SITL,
            max_samples=16,
        )
        runtime.start()
        try:
            runtime.emit(
                "hover_primitive_touchdown_confirm",
                status="done",
                data={
                    "confirmation_source": "range_only_sitl",
                    "distance_m": 0.02,
                    "is_flying": True,
                },
            )
            snapshot = runtime.latest_snapshot()
        finally:
            runtime.stop()

        divergence_codes = {item.code for item in snapshot.divergences}
        self.assertNotIn("touchdown_not_confirmed", divergence_codes)
        self.assertNotIn("touchdown_range_only_without_supervisor_ground", divergence_codes)


if __name__ == "__main__":
    unittest.main()
