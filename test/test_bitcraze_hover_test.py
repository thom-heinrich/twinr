"""Regression coverage for the bounded Bitcraze hover-test worker."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from typing import Any
import unittest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "hardware" / "bitcraze" / "run_hover_test.py"
_SPEC = importlib.util.spec_from_file_location("bitcraze_hover_test_script", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE: Any = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class HoverTestWorkerTests(unittest.TestCase):
    def test_normalize_required_deck_name_accepts_known_alias(self) -> None:
        self.assertEqual(_MODULE.normalize_required_deck_name("flow2"), "bcFlow2")
        self.assertEqual(_MODULE.normalize_required_deck_name("zranger2"), "bcZRanger2")

    def test_evaluate_hover_preflight_reports_missing_deck_and_low_power(self) -> None:
        failures = _MODULE.evaluate_hover_preflight(
            deck_flags={"bcFlow2": 0, "bcZRanger2": 1},
            required_decks=("bcFlow2", "bcZRanger2"),
            power=_MODULE.HoverPowerSnapshot(vbat_v=3.55, battery_level=12, state=1),
            clearance_snapshot=None,
            min_vbat_v=3.8,
            min_battery_level=20,
            min_clearance_m=0.35,
        )

        self.assertIn("required deck bcFlow2 is not detected", failures)
        self.assertIn("battery voltage 3.55 V is below the 3.80 V hover gate", failures)
        self.assertIn("battery level 12% is below the 20% hover gate", failures)

    def test_evaluate_hover_preflight_reports_close_obstacles_when_multiranger_snapshot_exists(self) -> None:
        failures = _MODULE.evaluate_hover_preflight(
            deck_flags={"bcFlow2": 1, "bcZRanger2": 1, "bcMultiranger": 1},
            required_decks=("bcFlow2", "bcZRanger2"),
            power=_MODULE.HoverPowerSnapshot(vbat_v=4.0, battery_level=70, state=0),
            clearance_snapshot=_MODULE.HoverClearanceSnapshot(
                front_m=0.12,
                back_m=None,
                left_m=0.55,
                right_m=0.18,
                up_m=1.20,
                down_m=0.0,
            ),
            min_vbat_v=3.8,
            min_battery_level=20,
            min_clearance_m=0.35,
        )

        self.assertIn("front clearance 0.12 m is below the 0.35 m hover gate", failures)
        self.assertIn("right clearance 0.18 m is below the 0.35 m hover gate", failures)

    def test_run_hover_test_blocks_when_required_on_device_failsafe_is_missing(self) -> None:
        class _FakeCRTP:
            @staticmethod
            def init_drivers() -> None:
                return None

        class _FakeCrazyflie:
            def __init__(self, *, rw_cache: str) -> None:
                self.rw_cache = rw_cache

        class _FakeSyncCrazyflie:
            def __init__(self, _uri: str, *, cf: object) -> None:
                self.cf = type("_FakeCFHandle", (), {"log": type("_FakeLog", (), {"add_config": lambda *_args, **_kwargs: None})()})()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

        class _UnusedLogConfig:
            def __init__(self, *args, **kwargs) -> None:
                return None

        class _UnusedSyncLogger:
            def __init__(self, *args, **kwargs) -> None:
                return None

        class _UnusedMultiranger:
            def __init__(self, *_args, **_kwargs) -> None:
                return None

        original_import = _MODULE._import_cflib
        original_read_deck_flags = _MODULE._read_deck_flags
        original_read_power_snapshot = _MODULE._read_power_snapshot
        original_apply_pre_arm = _MODULE.apply_hover_pre_arm
        original_wait_for_estimator_settle = _MODULE.wait_for_estimator_settle
        original_probe_failsafe = _MODULE.probe_on_device_failsafe
        availability_cls = original_probe_failsafe.__globals__["OnDeviceFailsafeAvailability"]
        try:
            _MODULE._import_cflib = lambda: (
                _FakeCRTP,
                _FakeCrazyflie,
                _UnusedLogConfig,
                _UnusedMultiranger,
                _FakeSyncCrazyflie,
                _UnusedSyncLogger,
            )
            _MODULE._read_deck_flags = lambda *_args, **_kwargs: {"bcFlow2": 1, "bcZRanger2": 1, "bcAI": 1}
            _MODULE._read_power_snapshot = (
                lambda *_args, **_kwargs: _MODULE.HoverPowerSnapshot(vbat_v=4.0, battery_level=90, state=0)
            )
            _MODULE.apply_hover_pre_arm = lambda *_args, **_kwargs: _MODULE.HoverPreArmSnapshot(
                estimator_requested=2,
                estimator=2,
                controller_requested=1,
                controller=1,
                motion_disable_requested=0,
                motion_disable=0,
                kalman_reset_after=0,
                kalman_reset_performed=True,
                verified=True,
                failures=(),
            )
            _MODULE.wait_for_estimator_settle = lambda *_args, **_kwargs: _MODULE.HoverEstimatorSettlingReport(
                stable=True,
                sample_count=10,
                duration_s=1.0,
                var_px_span=0.0003,
                var_py_span=0.0004,
                var_pz_span=0.0005,
                roll_abs_max_deg=0.6,
                pitch_abs_max_deg=0.7,
                motion_squal_min=55,
                motion_squal_mean=88.0,
                motion_squal_nonzero_ratio=1.0,
                zrange_min_m=0.018,
                zrange_observed=True,
                failures=(),
            )
            _MODULE.probe_on_device_failsafe = lambda *_args, **_kwargs: availability_cls(
                loaded=False,
                protocol_version=None,
                enabled=None,
                state_code=None,
                state_name=None,
                reason_code=None,
                reason_name=None,
                failures=("twinrFs.protocolVersion:missing",),
            )

            with tempfile.TemporaryDirectory() as temp_dir:
                report = _MODULE.run_hover_test(
                    uri="radio://0/80/2M",
                    workspace=Path(temp_dir),
                    height_m=0.25,
                    hover_duration_s=3.0,
                    takeoff_velocity_mps=0.2,
                    land_velocity_mps=0.2,
                    connect_settle_s=0.0,
                    min_vbat_v=3.8,
                    min_battery_level=20,
                    min_clearance_m=0.35,
                    stabilizer_estimator=2,
                    stabilizer_controller=1,
                    motion_disable=0,
                    estimator_settle_timeout_s=5.0,
                    on_device_failsafe_mode="required",
                    on_device_failsafe_heartbeat_timeout_s=0.35,
                    on_device_failsafe_low_battery_v=3.55,
                    on_device_failsafe_critical_battery_v=3.35,
                    on_device_failsafe_min_up_clearance_m=0.25,
                    required_decks=("bcFlow2", "bcZRanger2"),
                )
        finally:
            _MODULE._import_cflib = original_import
            _MODULE._read_deck_flags = original_read_deck_flags
            _MODULE._read_power_snapshot = original_read_power_snapshot
            _MODULE.apply_hover_pre_arm = original_apply_pre_arm
            _MODULE.wait_for_estimator_settle = original_wait_for_estimator_settle
            _MODULE.probe_on_device_failsafe = original_probe_failsafe

        self.assertEqual(report.status, "blocked")
        self.assertIsNotNone(report.on_device_failsafe)
        self.assertFalse(report.on_device_failsafe.availability.loaded)
        self.assertIn("required on-device failsafe app", report.failures[0])

    def test_recommendations_cover_completed_hover_test(self) -> None:
        report = _MODULE.HoverTestReport(
            uri="radio://0/80/2M",
            workspace="/twinr/bitcraze",
            height_m=0.25,
            hover_duration_s=3.0,
            takeoff_velocity_mps=0.2,
            land_velocity_mps=0.2,
            connect_settle_s=1.0,
            min_vbat_v=3.8,
            min_battery_level=20,
            min_clearance_m=0.35,
            deck_flags={"bcFlow2": 1, "bcZRanger2": 1},
            required_decks=("bcFlow2", "bcZRanger2"),
            clearance_snapshot=None,
            pre_arm_snapshot=None,
            estimator_settle=None,
            power=_MODULE.HoverPowerSnapshot(vbat_v=4.0, battery_level=70, state=0),
            status="completed",
            completed=True,
            landed=True,
            interrupted=False,
            primitive_outcome=None,
            telemetry=(),
            telemetry_summary=None,
            failures=(),
            recommendations=(),
        )

        recommendations = _MODULE.recommendations_for_report(report)

        self.assertIn("Hover test completed and landed. The bounded flight primitive looks ready.", recommendations)

    def test_latest_ground_distance_from_telemetry_uses_valid_range_sample_and_age(self) -> None:
        class _FakeTelemetryCollector:
            def latest_value(self, key: str) -> tuple[float | int | None, float | None]:
                self.seen_key = getattr(self, "seen_key", []) + [key]
                if key == "range.zrange":
                    return 47, 0.12
                if key == "supervisor.info":
                    return 0, 0.04
                raise AssertionError(f"unexpected key {key}")

        telemetry = _FakeTelemetryCollector()
        observation = _MODULE._latest_ground_distance_from_telemetry(telemetry)

        self.assertEqual(getattr(telemetry, "seen_key", None), ["range.zrange", "supervisor.info"])
        self.assertAlmostEqual(observation.distance_m or 0.0, 0.047, places=3)
        self.assertAlmostEqual(observation.age_s or 0.0, 0.12, places=3)
        self.assertFalse(observation.is_flying)
        self.assertAlmostEqual(observation.supervisor_age_s or 0.0, 0.04, places=3)

    def test_latest_ground_distance_from_telemetry_drops_invalid_sentinel(self) -> None:
        class _FakeTelemetryCollector:
            def latest_value(self, key: str) -> tuple[float | int | None, float | None]:
                if key == "range.zrange":
                    return 32766, 0.08
                if key == "supervisor.info":
                    return 16, 0.03
                raise AssertionError(f"unexpected key {key}")

        observation = _MODULE._latest_ground_distance_from_telemetry(_FakeTelemetryCollector())

        self.assertIsNone(observation.distance_m)
        self.assertAlmostEqual(observation.age_s or 0.0, 0.08, places=3)
        self.assertTrue(observation.is_flying)
        self.assertAlmostEqual(observation.supervisor_age_s or 0.0, 0.03, places=3)

    def test_summarize_hover_telemetry_reports_flow_zrange_and_safe_supervisor(self) -> None:
        samples = (
            _MODULE.HoverTelemetrySample(
                timestamp_ms=1000,
                block_name="hover-attitude",
                values={
                    "stabilizer.roll": 1.5,
                    "stabilizer.pitch": -2.0,
                    "stateEstimate.x": 0.00,
                    "stateEstimate.y": 0.00,
                    "stateEstimate.z": 0.14,
                    "stateEstimate.vx": 0.01,
                    "stateEstimate.vy": 0.00,
                    "stateEstimate.vz": 0.02,
                    "stabilizer.thrust": 24500.0,
                },
            ),
            _MODULE.HoverTelemetrySample(
                timestamp_ms=1100,
                block_name="hover-sensors",
                values={
                    "motion.squal": 45,
                    "motion.deltaX": 0.4,
                    "motion.deltaY": 0.2,
                    "range.zrange": 152,
                    "range.front": 410,
                    "range.back": 380,
                    "range.left": 420,
                    "range.right": 415,
                    "range.up": 700,
                    "pm.vbat": 4.08,
                    "radio.rssi": -57,
                    "supervisor.info": 0,
                },
            ),
            _MODULE.HoverTelemetrySample(
                timestamp_ms=1200,
                block_name="hover-attitude",
                values={
                    "stabilizer.roll": -1.0,
                    "stabilizer.pitch": 1.5,
                    "stateEstimate.x": 0.02,
                    "stateEstimate.y": 0.01,
                    "stateEstimate.z": 0.15,
                    "stateEstimate.vx": 0.03,
                    "stateEstimate.vy": 0.01,
                    "stateEstimate.vz": -0.01,
                    "stabilizer.thrust": 25100.0,
                    "gyro.x": 1.0,
                    "gyro.y": -2.0,
                    "gyro.z": 3.0,
                },
            ),
            _MODULE.HoverTelemetrySample(
                timestamp_ms=1300,
                block_name="hover-sensors",
                values={
                    "motion.squal": 50,
                    "motion.deltaX": -0.3,
                    "motion.deltaY": 0.1,
                    "range.zrange": 149,
                    "range.front": 405,
                    "range.back": 360,
                    "range.left": 415,
                    "range.right": 400,
                    "range.up": 680,
                    "pm.vbat": 4.04,
                    "radio.rssi": -55,
                    "supervisor.info": 0,
                },
            ),
        )

        summary = _MODULE.summarize_hover_telemetry(
            samples,
            available_blocks=("hover-attitude", "hover-sensors", "hover-velocity", "hover-clearance", "hover-gyro"),
            skipped_blocks=(),
        )

        self.assertEqual(summary.sample_count, 4)
        self.assertIn("hover-velocity", summary.available_blocks)
        self.assertTrue(summary.flow_observed)
        self.assertTrue(summary.zrange_observed)
        self.assertTrue(summary.clearance_observed)
        self.assertTrue(summary.stable_supervisor)
        self.assertAlmostEqual(summary.xy_drift_m or 0.0, 0.02236067977, places=6)
        self.assertAlmostEqual(summary.zrange_min_m or 0.0, 0.149, places=3)
        self.assertAlmostEqual(summary.front_min_m or 0.0, 0.405, places=3)
        self.assertAlmostEqual(summary.horizontal_speed_max_mps or 0.0, 0.0316227766, places=6)
        self.assertAlmostEqual(summary.thrust_max or 0.0, 25100.0, places=3)
        self.assertAlmostEqual(summary.radio_rssi_latest_dbm or 0.0, -55.0, places=3)
        self.assertAlmostEqual(summary.battery_drop_v or 0.0, 0.04, places=2)
        self.assertEqual(_MODULE.evaluate_hover_stability(summary, target_height_m=0.25), [])

    def test_summarize_hover_telemetry_uses_airborne_window_and_filters_invalid_ranges(self) -> None:
        samples = (
            _MODULE.HoverTelemetrySample(
                timestamp_ms=1000,
                block_name="hover-attitude",
                values={
                    "stabilizer.roll": 0.1,
                    "stabilizer.pitch": 0.2,
                    "stateEstimate.x": -0.84,
                    "stateEstimate.y": 0.25,
                    "stateEstimate.z": 0.0,
                },
            ),
            _MODULE.HoverTelemetrySample(
                timestamp_ms=1100,
                block_name="hover-attitude",
                values={
                    "stabilizer.roll": 0.1,
                    "stabilizer.pitch": 0.1,
                    "stateEstimate.x": 0.0,
                    "stateEstimate.y": 0.0,
                    "stateEstimate.z": 0.0,
                },
            ),
            _MODULE.HoverTelemetrySample(
                timestamp_ms=2000,
                block_name="hover-attitude",
                values={
                    "stabilizer.roll": 0.5,
                    "stabilizer.pitch": 0.6,
                    "stateEstimate.x": -0.07,
                    "stateEstimate.y": 0.02,
                    "stateEstimate.z": 0.03,
                },
            ),
            _MODULE.HoverTelemetrySample(
                timestamp_ms=2050,
                block_name="hover-sensors",
                values={
                    "motion.squal": 110,
                    "motion.deltaX": 3.0,
                    "motion.deltaY": 1.0,
                    "range.zrange": 35,
                    "pm.vbat": 3.90,
                    "radio.rssi": -55,
                    "supervisor.info": 0,
                },
            ),
            _MODULE.HoverTelemetrySample(
                timestamp_ms=2100,
                block_name="hover-clearance",
                values={
                    "range.front": 32766,
                    "range.back": 900,
                    "range.left": 2000,
                    "range.right": 250,
                    "range.up": 1800,
                },
            ),
            _MODULE.HoverTelemetrySample(
                timestamp_ms=2200,
                block_name="hover-velocity",
                values={
                    "stateEstimate.vx": 0.05,
                    "stateEstimate.vy": -0.02,
                    "stateEstimate.vz": 0.01,
                    "stabilizer.thrust": 24000.0,
                    "gyro.x": 1.0,
                    "gyro.y": 2.0,
                    "gyro.z": 3.0,
                },
            ),
            _MODULE.HoverTelemetrySample(
                timestamp_ms=3000,
                block_name="hover-attitude",
                values={
                    "stabilizer.roll": -0.2,
                    "stabilizer.pitch": -0.3,
                    "stateEstimate.x": -0.04,
                    "stateEstimate.y": -0.06,
                    "stateEstimate.z": 0.10,
                },
            ),
            _MODULE.HoverTelemetrySample(
                timestamp_ms=3050,
                block_name="hover-sensors",
                values={
                    "motion.squal": 130,
                    "motion.deltaX": 4.0,
                    "motion.deltaY": 2.0,
                    "range.zrange": 102,
                    "pm.vbat": 3.83,
                    "radio.rssi": -56,
                    "supervisor.info": 0,
                },
            ),
            _MODULE.HoverTelemetrySample(
                timestamp_ms=3100,
                block_name="hover-clearance",
                values={
                    "range.front": 32766,
                    "range.back": 1000,
                    "range.left": 2200,
                    "range.right": 462,
                    "range.up": 1700,
                },
            ),
            _MODULE.HoverTelemetrySample(
                timestamp_ms=3200,
                block_name="hover-velocity",
                values={
                    "stateEstimate.vx": -0.04,
                    "stateEstimate.vy": -0.01,
                    "stateEstimate.vz": -0.04,
                    "stabilizer.thrust": 44000.0,
                    "gyro.x": 2.0,
                    "gyro.y": 3.0,
                    "gyro.z": 4.0,
                },
            ),
        )

        summary = _MODULE.summarize_hover_telemetry(
            samples,
            available_blocks=("hover-attitude", "hover-sensors", "hover-velocity", "hover-clearance"),
            skipped_blocks=(),
        )

        self.assertEqual(summary.sample_count, 7)
        self.assertAlmostEqual(summary.duration_s or 0.0, 1.05, places=3)
        self.assertAlmostEqual(summary.xy_drift_m or 0.0, 0.08544003745, places=6)
        self.assertAlmostEqual(summary.zrange_min_m or 0.0, 0.035, places=3)
        self.assertAlmostEqual(summary.right_min_m or 0.0, 0.25, places=3)
        self.assertAlmostEqual(summary.left_min_m or 0.0, 2.0, places=3)
        self.assertIsNone(summary.front_min_m)
        self.assertAlmostEqual(summary.battery_drop_v or 0.0, 0.07, places=2)
        self.assertAlmostEqual(summary.thrust_max or 0.0, 24000.0, places=3)

    def test_evaluate_hover_stability_reports_missing_flow_and_unsafe_supervisor(self) -> None:
        summary = _MODULE.HoverTelemetrySummary(
            sample_count=3,
            available_blocks=("hover-attitude", "hover-sensors"),
            skipped_blocks=("hover-clearance",),
            duration_s=0.2,
            roll_abs_max_deg=3.0,
            pitch_abs_max_deg=2.0,
            xy_drift_m=0.01,
            z_drift_m=0.0,
            z_span_m=0.01,
            vx_abs_max_mps=0.01,
            vy_abs_max_mps=0.01,
            vz_abs_max_mps=0.02,
            horizontal_speed_max_mps=0.014,
            flow_squal_min=0,
            flow_squal_mean=0.0,
            flow_nonzero_samples=0,
            flow_observed=False,
            motion_delta_x_abs_max=0.2,
            motion_delta_y_abs_max=0.2,
            zrange_min_m=None,
            zrange_max_m=None,
            zrange_sample_count=0,
            zrange_observed=False,
            front_min_m=None,
            back_min_m=None,
            left_min_m=None,
            right_min_m=None,
            up_min_m=None,
            clearance_observed=False,
            thrust_mean=24000.0,
            thrust_max=25000.0,
            gyro_abs_max_dps=3.0,
            battery_min_v=4.0,
            battery_drop_v=0.02,
            radio_rssi_latest_dbm=-60.0,
            radio_rssi_min_dbm=-65.0,
            latest_supervisor_info=1 << 7,
            supervisor_flags_seen=("crashed",),
            stable_supervisor=False,
        )

        failures = _MODULE.evaluate_hover_stability(summary, target_height_m=0.25)

        self.assertIn("optical-flow quality never became nonzero during the hover test", failures)
        self.assertIn("downward z-range never produced a nonzero reading during the hover test", failures)
        self.assertIn("supervisor reported unsafe flags during the hover test: crashed", failures)

    def test_evaluate_hover_stability_reports_altitude_overshoot_and_battery_sag(self) -> None:
        summary = _MODULE.HoverTelemetrySummary(
            sample_count=40,
            available_blocks=("hover-attitude", "hover-sensors", "hover-clearance"),
            skipped_blocks=(),
            duration_s=4.2,
            roll_abs_max_deg=3.2,
            pitch_abs_max_deg=4.1,
            xy_drift_m=0.08,
            z_drift_m=0.04,
            z_span_m=0.62,
            vx_abs_max_mps=0.12,
            vy_abs_max_mps=0.09,
            vz_abs_max_mps=1.8,
            horizontal_speed_max_mps=0.14,
            flow_squal_min=80,
            flow_squal_mean=120.0,
            flow_nonzero_samples=40,
            flow_observed=True,
            motion_delta_x_abs_max=12.0,
            motion_delta_y_abs_max=10.0,
            zrange_min_m=0.18,
            zrange_max_m=0.88,
            zrange_sample_count=40,
            zrange_observed=True,
            front_min_m=0.40,
            back_min_m=0.90,
            left_min_m=0.80,
            right_min_m=0.65,
            up_min_m=1.2,
            clearance_observed=True,
            thrust_mean=42000.0,
            thrust_max=65535.0,
            gyro_abs_max_dps=30.0,
            battery_min_v=3.34,
            battery_drop_v=0.55,
            radio_rssi_latest_dbm=35.0,
            radio_rssi_min_dbm=35.0,
            latest_supervisor_info=0,
            supervisor_flags_seen=(),
            stable_supervisor=True,
        )

        failures = _MODULE.evaluate_hover_stability(summary, target_height_m=0.25)

        self.assertIn("hover altitude reached 0.88 m which exceeds the 0.50 m safety ceiling for a 0.25 m hover", failures)
        self.assertIn("battery sagged to 3.34 V under load which is below the 3.50 V hover safety floor", failures)

    def test_run_hover_test_persists_pre_arm_and_estimator_settle_evidence(self) -> None:
        events: list[str] = []

        class _FakeCRTP:
            @staticmethod
            def init_drivers() -> None:
                events.append("init_drivers")

        class _FakeCrazyflie:
            def __init__(self, *, rw_cache: str) -> None:
                self.rw_cache = rw_cache

        class _FakeSyncCrazyflie:
            def __init__(self, _uri: str, *, cf: object) -> None:
                self.cf = type("_FakeCFHandle", (), {"log": type("_FakeLog", (), {"add_config": lambda *_args, **_kwargs: None})()})()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

        class _UnusedLogConfig:
            def __init__(self, *args, **kwargs) -> None:
                return None

        class _UnusedSyncLogger:
            def __init__(self, *args, **kwargs) -> None:
                return None

        class _UnusedMultiranger:
            def __init__(self, *_args, **_kwargs) -> None:
                return None

        original_import = _MODULE._import_cflib
        original_read_deck_flags = _MODULE._read_deck_flags
        original_read_power_snapshot = _MODULE._read_power_snapshot
        original_telemetry_collector = _MODULE.HoverTelemetryCollector
        original_apply_pre_arm = _MODULE.apply_hover_pre_arm
        original_wait_for_estimator_settle = _MODULE.wait_for_estimator_settle
        original_stateful_hover_primitive = _MODULE.StatefulHoverPrimitive
        original_sleep = _MODULE.time.sleep
        try:
            _MODULE._import_cflib = lambda: (
                _FakeCRTP,
                _FakeCrazyflie,
                _UnusedLogConfig,
                _UnusedMultiranger,
                _FakeSyncCrazyflie,
                _UnusedSyncLogger,
            )
            _MODULE._read_deck_flags = lambda *_args, **_kwargs: {"bcFlow2": 1, "bcZRanger2": 1, "bcAI": 1}
            _MODULE._read_power_snapshot = (
                lambda *_args, **_kwargs: _MODULE.HoverPowerSnapshot(vbat_v=4.0, battery_level=90, state=0)
            )
            _MODULE.apply_hover_pre_arm = lambda *_args, **_kwargs: _MODULE.HoverPreArmSnapshot(
                estimator_requested=2,
                estimator=2,
                controller_requested=1,
                controller=1,
                motion_disable_requested=0,
                motion_disable=0,
                kalman_reset_after=0,
                kalman_reset_performed=True,
                verified=True,
                failures=(),
            )
            _MODULE.wait_for_estimator_settle = lambda *_args, **_kwargs: _MODULE.HoverEstimatorSettlingReport(
                stable=True,
                sample_count=10,
                duration_s=1.0,
                var_px_span=0.0003,
                var_py_span=0.0004,
                var_pz_span=0.0005,
                roll_abs_max_deg=0.6,
                pitch_abs_max_deg=0.7,
                motion_squal_min=55,
                motion_squal_mean=88.0,
                motion_squal_nonzero_ratio=1.0,
                zrange_min_m=0.018,
                zrange_observed=True,
                failures=(),
            )

            class _FakeTelemetryCollector:
                def __init__(self, *_args, **_kwargs) -> None:
                    events.append("telemetry_init")
                    self.available_blocks = ("hover-attitude", "hover-sensors", "hover-velocity")
                    self.skipped_blocks = ("hover-clearance",)

                def start(self) -> None:
                    events.append("telemetry_start")

                def stop(self) -> None:
                    events.append("telemetry_stop")

                def snapshot(self):
                    return (
                        _MODULE.HoverTelemetrySample(
                            timestamp_ms=1000,
                            block_name="hover-attitude",
                            values={
                                "stabilizer.roll": 0.5,
                                "stabilizer.pitch": -0.4,
                                "stateEstimate.x": 0.0,
                                "stateEstimate.y": 0.0,
                                "stateEstimate.z": 0.15,
                                "stateEstimate.vx": 0.02,
                                "stateEstimate.vy": 0.01,
                                "stateEstimate.vz": 0.0,
                                "stabilizer.thrust": 24000.0,
                            },
                        ),
                        _MODULE.HoverTelemetrySample(
                            timestamp_ms=1100,
                            block_name="hover-sensors",
                            values={
                                "motion.squal": 40,
                                "motion.deltaX": 0.4,
                                "motion.deltaY": 0.2,
                                "range.zrange": 150,
                                "pm.vbat": 4.0,
                                "radio.rssi": -58,
                                "supervisor.info": 0,
                            },
                        ),
                    )

            class _FakeStatefulHoverPrimitive:
                def __init__(self, *_args, **_kwargs) -> None:
                    events.append("primitive_init")
                    self.landed = True
                    self.took_off = True

                def run(self, config: Any) -> object:
                    events.append(
                        "primitive_run:"
                        f"{config.target_height_m:.2f}:{config.hover_duration_s:.2f}:{config.takeoff_velocity_mps:.2f}"
                    )
                    return _MODULE.HoverPrimitiveOutcome(
                        final_phase="landed",
                        took_off=True,
                        landed=True,
                        aborted=False,
                        abort_reason=None,
                        commanded_max_height_m=config.target_height_m,
                        setpoint_count=12,
                    )

            _MODULE.HoverTelemetryCollector = _FakeTelemetryCollector
            _MODULE.StatefulHoverPrimitive = _FakeStatefulHoverPrimitive
            _MODULE.time.sleep = lambda _seconds: events.append("sleep")
            with tempfile.TemporaryDirectory() as temp_dir:
                trace_path = Path(temp_dir) / "hover-trace.jsonl"
                report = _MODULE.run_hover_test(
                    uri="radio://0/80/2M",
                    workspace=Path(temp_dir),
                    height_m=0.25,
                    hover_duration_s=3.0,
                    takeoff_velocity_mps=0.2,
                    land_velocity_mps=0.2,
                    connect_settle_s=0.0,
                    min_vbat_v=3.8,
                    min_battery_level=20,
                    min_clearance_m=0.35,
                    stabilizer_estimator=2,
                    stabilizer_controller=1,
                    motion_disable=0,
                    estimator_settle_timeout_s=5.0,
                    on_device_failsafe_mode="off",
                    on_device_failsafe_heartbeat_timeout_s=0.35,
                    on_device_failsafe_low_battery_v=3.55,
                    on_device_failsafe_critical_battery_v=3.35,
                    on_device_failsafe_min_up_clearance_m=0.25,
                    required_decks=("bcFlow2", "bcZRanger2"),
                    trace_writer=_MODULE.HoverWorkerTraceWriter(trace_path),
                )
                trace_events = [
                    json.loads(line)
                    for line in trace_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
        finally:
            _MODULE._import_cflib = original_import
            _MODULE._read_deck_flags = original_read_deck_flags
            _MODULE._read_power_snapshot = original_read_power_snapshot
            _MODULE.HoverTelemetryCollector = original_telemetry_collector
            _MODULE.apply_hover_pre_arm = original_apply_pre_arm
            _MODULE.wait_for_estimator_settle = original_wait_for_estimator_settle
            _MODULE.StatefulHoverPrimitive = original_stateful_hover_primitive
            _MODULE.time.sleep = original_sleep

        self.assertTrue(report.completed)
        self.assertEqual(
            [event for event in events if event.startswith("primitive_run:")],
            ["primitive_run:0.25:3.00:0.20"],
        )
        self.assertIsNotNone(report.pre_arm_snapshot)
        self.assertTrue(report.pre_arm_snapshot.verified)
        self.assertIsNotNone(report.estimator_settle)
        self.assertTrue(report.estimator_settle.stable)
        self.assertIsNotNone(report.primitive_outcome)
        self.assertEqual(report.primitive_outcome.final_phase, "landed")
        self.assertIsNotNone(report.telemetry_summary)
        self.assertTrue(report.telemetry_summary.flow_observed)
        self.assertIsNone(report.on_device_failsafe)
        self.assertIn(("sync_connect", "done"), {(item["phase"], item["status"]) for item in trace_events})
        self.assertIn(("preflight", "done"), {(item["phase"], item["status"]) for item in trace_events})
        self.assertIn(("telemetry_stop", "done"), {(item["phase"], item["status"]) for item in trace_events})
        self.assertIn(("sync_disconnect", "done"), {(item["phase"], item["status"]) for item in trace_events})
        self.assertEqual(trace_events[-1]["phase"], "report_build")
        self.assertEqual(trace_events[-1]["status"], "done")


if __name__ == "__main__":
    unittest.main()
