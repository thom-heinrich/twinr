"""Regression coverage for the bounded Bitcraze hover-test worker."""

from __future__ import annotations

from dataclasses import replace
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


def _safe_status_snapshot() -> Any:
    return _MODULE.HoverStatusSnapshot(
        supervisor_info=0,
        can_arm=True,
        is_armed=False,
        auto_arm=False,
        can_fly=True,
        is_flying=False,
        tumbled=False,
        locked=False,
        crashed=False,
        hl_flying=False,
        hl_trajectory_finished=False,
        hl_disabled=False,
        radio_connected=True,
        zrange_m=0.04,
        motion_squal=80,
    )


class HoverTestWorkerTests(unittest.TestCase):
    def test_stability_config_for_sitl_disables_nested_flow_requirement(self) -> None:
        config = _MODULE._stability_config_for_runtime_mode(_MODULE.HOVER_RUNTIME_MODE_SITL)

        self.assertFalse(config.require_motion_squal)
        self.assertFalse(config.lateral_trust.require_motion_squal)

    def test_vertical_bootstrap_config_is_hardware_only(self) -> None:
        hardware_config = _MODULE._vertical_bootstrap_config_for_runtime_mode(
            _MODULE.HOVER_RUNTIME_MODE_HARDWARE,
            micro_liftoff_height_m=0.08,
            takeoff_confirm_target_height_tolerance_m=0.05,
        )
        sitl_config = _MODULE._vertical_bootstrap_config_for_runtime_mode(
            _MODULE.HOVER_RUNTIME_MODE_SITL,
            micro_liftoff_height_m=0.08,
            takeoff_confirm_target_height_tolerance_m=0.05,
        )

        self.assertIsNotNone(hardware_config)
        assert hardware_config is not None
        self.assertEqual(hardware_config.min_thrust_percentage, 36.0)
        self.assertEqual(hardware_config.feedforward_thrust_percentage, 40.0)
        self.assertEqual(hardware_config.max_thrust_percentage, 52.0)
        self.assertEqual(hardware_config.reference_duration_s, 0.75)
        self.assertEqual(hardware_config.progress_to_ceiling_s, 0.35)
        self.assertEqual(hardware_config.min_range_height_m, 0.08)
        self.assertEqual(hardware_config.max_range_height_m, 0.13)
        self.assertEqual(hardware_config.min_motion_squal, 1)
        self.assertIsNone(sitl_config)

    def test_normalize_required_deck_name_accepts_known_alias(self) -> None:
        self.assertEqual(_MODULE.normalize_required_deck_name("flow2"), "bcFlow2")
        self.assertEqual(_MODULE.normalize_required_deck_name("zranger2"), "bcZRanger2")

    def test_evaluate_hover_preflight_reports_missing_deck_and_low_power(self) -> None:
        failures = _MODULE.evaluate_hover_preflight(
            runtime_mode=_MODULE.HOVER_RUNTIME_MODE_HARDWARE,
            deck_flags={"bcFlow2": 0, "bcZRanger2": 1},
            required_decks=("bcFlow2", "bcZRanger2"),
            power=_MODULE.HoverPowerSnapshot(vbat_v=3.55, battery_level=12, state=1),
            status_snapshot=_safe_status_snapshot(),
            clearance_snapshot=None,
            min_vbat_v=3.8,
            min_battery_level=20,
            min_clearance_m=0.35,
            lateral_clearance_arm_height_m=0.20,
        )

        self.assertIn("required deck bcFlow2 is not detected", failures)
        self.assertIn("battery voltage 3.55 V is below the 3.80 V hover gate", failures)
        self.assertIn("battery level 12% is below the 20% hover gate", failures)

    def test_evaluate_hover_preflight_reports_power_state_flapping(self) -> None:
        failures = _MODULE.evaluate_hover_preflight(
            runtime_mode=_MODULE.HOVER_RUNTIME_MODE_HARDWARE,
            deck_flags={"bcFlow2": 1, "bcZRanger2": 1},
            required_decks=("bcFlow2", "bcZRanger2"),
            power=_MODULE.HoverPowerSnapshot(
                vbat_v=4.05,
                battery_level=90,
                state=0,
                observed_states=(2, 0, 2, 0),
                sample_count=4,
            ),
            status_snapshot=_safe_status_snapshot(),
            clearance_snapshot=None,
            min_vbat_v=3.8,
            min_battery_level=20,
            min_clearance_m=0.35,
            lateral_clearance_arm_height_m=0.20,
        )

        self.assertIn(
            "battery power state flapped during preflight window: charged (2) -> battery (0)",
            failures,
        )

    def test_evaluate_hover_preflight_reports_supervisor_state_flapping(self) -> None:
        stable_status = _safe_status_snapshot()
        flapping_status = _MODULE.HoverStatusSnapshot(
            supervisor_info=stable_status.supervisor_info,
            can_arm=stable_status.can_arm,
            is_armed=stable_status.is_armed,
            auto_arm=stable_status.auto_arm,
            can_fly=stable_status.can_fly,
            is_flying=stable_status.is_flying,
            tumbled=stable_status.tumbled,
            locked=stable_status.locked,
            crashed=stable_status.crashed,
            hl_flying=stable_status.hl_flying,
            hl_trajectory_finished=stable_status.hl_trajectory_finished,
            hl_disabled=stable_status.hl_disabled,
            radio_connected=stable_status.radio_connected,
            zrange_m=stable_status.zrange_m,
            motion_squal=stable_status.motion_squal,
            observed_can_arm=(True, False, True),
            observed_can_fly=(True, False, True),
            observed_is_armed=(False, True, False),
            observed_supervisor_info=(0, 8, 0),
            sample_count=3,
        )
        failures = _MODULE.evaluate_hover_preflight(
            runtime_mode=_MODULE.HOVER_RUNTIME_MODE_HARDWARE,
            deck_flags={"bcFlow2": 1, "bcZRanger2": 1},
            required_decks=("bcFlow2", "bcZRanger2"),
            power=_MODULE.HoverPowerSnapshot(vbat_v=4.05, battery_level=90, state=0),
            status_snapshot=flapping_status,
            clearance_snapshot=None,
            min_vbat_v=3.8,
            min_battery_level=20,
            min_clearance_m=0.35,
            lateral_clearance_arm_height_m=0.20,
        )

        self.assertIn(
            "supervisor can_fly flapped during preflight window: true -> false",
            failures,
        )
        self.assertIn(
            "supervisor is_armed flapped during preflight window: false -> true",
            failures,
        )

    def test_evaluate_hover_preflight_allows_missing_battery_level_in_sitl(self) -> None:
        failures = _MODULE.evaluate_hover_preflight(
            runtime_mode=_MODULE.HOVER_RUNTIME_MODE_SITL,
            deck_flags={"bcMultiranger": 1},
            required_decks=(),
            power=_MODULE.HoverPowerSnapshot(vbat_v=4.05, battery_level=None, state=0),
            status_snapshot=_safe_status_snapshot(),
            clearance_snapshot=None,
            min_vbat_v=0.0,
            min_battery_level=0,
            min_clearance_m=0.0,
            lateral_clearance_arm_height_m=0.20,
        )

        self.assertEqual(failures, [])

    def test_evaluate_hover_preflight_ignores_lateral_clearance_below_start_gate(self) -> None:
        failures = _MODULE.evaluate_hover_preflight(
            runtime_mode=_MODULE.HOVER_RUNTIME_MODE_HARDWARE,
            deck_flags={"bcFlow2": 1, "bcZRanger2": 1, "bcMultiranger": 1},
            required_decks=("bcFlow2", "bcZRanger2"),
            power=_MODULE.HoverPowerSnapshot(vbat_v=4.0, battery_level=70, state=0),
            status_snapshot=_safe_status_snapshot(),
            clearance_snapshot=_MODULE.HoverClearanceSnapshot(
                front_m=0.12,
                back_m=None,
                left_m=0.55,
                right_m=0.18,
                up_m=1.20,
                down_m=0.02,
            ),
            min_vbat_v=3.8,
            min_battery_level=20,
            min_clearance_m=0.35,
            lateral_clearance_arm_height_m=0.20,
        )

        self.assertEqual(failures, [])

    def test_evaluate_hover_preflight_still_blocks_up_clearance_below_start_gate(self) -> None:
        failures = _MODULE.evaluate_hover_preflight(
            runtime_mode=_MODULE.HOVER_RUNTIME_MODE_HARDWARE,
            deck_flags={"bcFlow2": 1, "bcZRanger2": 1, "bcMultiranger": 1},
            required_decks=("bcFlow2", "bcZRanger2"),
            power=_MODULE.HoverPowerSnapshot(vbat_v=4.0, battery_level=70, state=0),
            status_snapshot=_safe_status_snapshot(),
            clearance_snapshot=_MODULE.HoverClearanceSnapshot(
                front_m=0.12,
                back_m=None,
                left_m=0.55,
                right_m=0.18,
                up_m=0.10,
                down_m=0.02,
            ),
            min_vbat_v=3.8,
            min_battery_level=20,
            min_clearance_m=0.35,
            lateral_clearance_arm_height_m=0.20,
        )

        self.assertEqual(failures, ["up clearance 0.10 m is below the 0.35 m hover gate"])

    def test_evaluate_hover_preflight_reports_lateral_clearance_once_active_height_is_reached(self) -> None:
        status_snapshot = _safe_status_snapshot()
        airborne_status = _MODULE.HoverStatusSnapshot(
            supervisor_info=status_snapshot.supervisor_info,
            can_arm=status_snapshot.can_arm,
            is_armed=status_snapshot.is_armed,
            auto_arm=status_snapshot.auto_arm,
            can_fly=status_snapshot.can_fly,
            is_flying=status_snapshot.is_flying,
            tumbled=status_snapshot.tumbled,
            locked=status_snapshot.locked,
            crashed=status_snapshot.crashed,
            hl_flying=status_snapshot.hl_flying,
            hl_trajectory_finished=status_snapshot.hl_trajectory_finished,
            hl_disabled=status_snapshot.hl_disabled,
            radio_connected=status_snapshot.radio_connected,
            zrange_m=0.26,
            motion_squal=status_snapshot.motion_squal,
        )
        failures = _MODULE.evaluate_hover_preflight(
            runtime_mode=_MODULE.HOVER_RUNTIME_MODE_HARDWARE,
            deck_flags={"bcFlow2": 1, "bcZRanger2": 1, "bcMultiranger": 1},
            required_decks=("bcFlow2", "bcZRanger2"),
            power=_MODULE.HoverPowerSnapshot(vbat_v=4.0, battery_level=70, state=0),
            status_snapshot=airborne_status,
            clearance_snapshot=_MODULE.HoverClearanceSnapshot(
                front_m=0.12,
                back_m=None,
                left_m=0.55,
                right_m=0.18,
                up_m=1.20,
                down_m=0.26,
            ),
            min_vbat_v=3.8,
            min_battery_level=20,
            min_clearance_m=0.35,
            lateral_clearance_arm_height_m=0.20,
        )

        self.assertIn("front clearance 0.12 m is below the 0.35 m hover gate", failures)
        self.assertIn("right clearance 0.18 m is below the 0.35 m hover gate", failures)

    def test_evaluate_hover_preflight_reports_lateral_clearance_when_supervisor_marks_flying(self) -> None:
        airborne_status = _MODULE.HoverStatusSnapshot(
            supervisor_info=0,
            can_arm=True,
            is_armed=True,
            auto_arm=True,
            can_fly=True,
            is_flying=True,
            tumbled=False,
            locked=False,
            crashed=False,
            hl_flying=True,
            hl_trajectory_finished=False,
            hl_disabled=False,
            radio_connected=True,
            zrange_m=0.04,
            motion_squal=80,
        )
        failures = _MODULE.evaluate_hover_preflight(
            runtime_mode=_MODULE.HOVER_RUNTIME_MODE_HARDWARE,
            deck_flags={"bcFlow2": 1, "bcZRanger2": 1, "bcMultiranger": 1},
            required_decks=("bcFlow2", "bcZRanger2"),
            power=_MODULE.HoverPowerSnapshot(vbat_v=4.0, battery_level=70, state=0),
            status_snapshot=airborne_status,
            clearance_snapshot=_MODULE.HoverClearanceSnapshot(
                front_m=0.12,
                back_m=None,
                left_m=0.55,
                right_m=0.18,
                up_m=1.20,
                down_m=0.04,
            ),
            min_vbat_v=3.8,
            min_battery_level=20,
            min_clearance_m=0.35,
            lateral_clearance_arm_height_m=0.20,
        )

        self.assertIn("front clearance 0.12 m is below the 0.35 m hover gate", failures)

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
                self.front = None
                self.back = None
                self.left = None
                self.right = None
                self.up = None
                self.down = None

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

        original_import = _MODULE._import_cflib
        original_read_deck_flags = _MODULE._read_deck_flags
        original_read_preflight_snapshots = _MODULE._read_preflight_snapshots
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
            _MODULE._read_preflight_snapshots = (
                lambda *_args, **_kwargs: (
                    _MODULE.HoverPowerSnapshot(vbat_v=4.0, battery_level=90, state=0),
                    _safe_status_snapshot(),
                )
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
                    runtime_mode=_MODULE.HOVER_RUNTIME_MODE_HARDWARE,
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
            _MODULE._read_preflight_snapshots = original_read_preflight_snapshots
            _MODULE.apply_hover_pre_arm = original_apply_pre_arm
            _MODULE.wait_for_estimator_settle = original_wait_for_estimator_settle
            _MODULE.probe_on_device_failsafe = original_probe_failsafe

        self.assertEqual(report.status, "blocked")
        self.assertIsNotNone(report.on_device_failsafe)
        self.assertFalse(report.on_device_failsafe.availability.loaded)
        self.assertIn("required on-device hover app", report.failures[0])

    def test_run_hover_test_blocks_when_hardware_protocol_lacks_hover_command_surface(self) -> None:
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
                self.front = None
                self.back = None
                self.left = None
                self.right = None
                self.up = None
                self.down = None

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

        original_import = _MODULE._import_cflib
        original_read_deck_flags = _MODULE._read_deck_flags
        original_read_preflight_snapshots = _MODULE._read_preflight_snapshots
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
            _MODULE._read_preflight_snapshots = (
                lambda *_args, **_kwargs: (
                    _MODULE.HoverPowerSnapshot(vbat_v=4.0, battery_level=90, state=0),
                    _safe_status_snapshot(),
                )
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
                loaded=True,
                protocol_version=2,
                enabled=1,
                state_code=1,
                state_name="monitoring",
                reason_code=0,
                reason_name="none",
                failures=(),
            )

            with tempfile.TemporaryDirectory() as temp_dir:
                report = _MODULE.run_hover_test(
                    runtime_mode=_MODULE.HOVER_RUNTIME_MODE_HARDWARE,
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
            _MODULE._read_preflight_snapshots = original_read_preflight_snapshots
            _MODULE.apply_hover_pre_arm = original_apply_pre_arm
            _MODULE.wait_for_estimator_settle = original_wait_for_estimator_settle
            _MODULE.probe_on_device_failsafe = original_probe_failsafe

        self.assertEqual(report.status, "blocked")
        self.assertIn(
            f"requires twinrFs protocol >= {_MODULE.ON_DEVICE_FAILSAFE_COMMAND_PROTOCOL_VERSION}",
            report.failures[0],
        )

    def test_hardware_hover_result_maps_from_on_device_mission(self) -> None:
        status_cls = _MODULE.probe_on_device_failsafe.__globals__["OnDeviceFailsafeStatus"]
        result = _MODULE.OnDeviceHoverResult(
            final_status=status_cls(
                session_id=17,
                state_code=9,
                state_name="mission_complete",
                reason_code=0,
                reason_name="none",
                heartbeat_age_ms=20,
                vbat_mv=3980,
                min_clearance_mm=350,
                down_range_mm=24,
                mission_flags=0,
                target_height_mm=100,
                commanded_height_mm=100,
                state_estimate_z_mm=95,
            ),
            observed_state_names=("mission_takeoff", "mission_hover", "mission_landing", "mission_complete"),
            observed_reason_names=("none",),
            took_off=True,
            landed=True,
            qualified_hover_reached=True,
            landing_reached=True,
            failures=(),
        )

        primitive_outcome = _MODULE._primitive_outcome_from_on_device_hover(
            result,
            target_height_m=0.10,
        )

        self.assertEqual(primitive_outcome.final_phase, "mission_complete")
        self.assertTrue(primitive_outcome.took_off)
        self.assertTrue(primitive_outcome.landed)
        self.assertTrue(primitive_outcome.stable_hover_established)
        self.assertTrue(primitive_outcome.trim_identified)
        self.assertTrue(primitive_outcome.qualified_hover_reached)
        self.assertTrue(primitive_outcome.landing_trim_identified)
        self.assertEqual(primitive_outcome.touchdown_confirmation_source, "on_device")
        self.assertAlmostEqual(primitive_outcome.touchdown_distance_m or 0.0, 0.024, places=3)

    def test_hardware_hover_result_surfaces_on_device_debug_context(self) -> None:
        status_cls = _MODULE.probe_on_device_failsafe.__globals__["OnDeviceFailsafeStatus"]
        result = _MODULE.OnDeviceHoverResult(
            final_status=status_cls(
                session_id=17,
                state_code=2,
                state_name="failsafe_brake",
                reason_code=10,
                reason_name="takeoff_attitude_quiet",
                heartbeat_age_ms=20,
                vbat_mv=3980,
                min_clearance_mm=350,
                down_range_mm=24,
                mission_flags=0,
                debug_flags=(
                    _MODULE.ON_DEVICE_FAILSAFE_DEBUG_FLAG_RANGE_READY
                    | _MODULE.ON_DEVICE_FAILSAFE_DEBUG_FLAG_FLOW_READY
                    | _MODULE.ON_DEVICE_FAILSAFE_DEBUG_FLAG_ATTITUDE_READY
                ),
                target_height_mm=100,
                commanded_height_mm=80,
                state_estimate_z_mm=65,
                up_range_mm=2400,
                motion_squal=44,
                touchdown_confirm_count=0,
                hover_thrust_permille=530,
            ),
            observed_state_names=("mission_takeoff", "failsafe_brake"),
            observed_reason_names=("none", "takeoff_attitude_quiet"),
            took_off=False,
            landed=False,
            qualified_hover_reached=False,
            landing_reached=False,
            failures=(),
        )

        primitive_outcome = _MODULE._primitive_outcome_from_on_device_hover(
            result,
            target_height_m=0.10,
        )

        assert primitive_outcome.abort_reason is not None
        self.assertIn("failsafe_brake:takeoff_attitude_quiet", primitive_outcome.abort_reason)
        self.assertIn("debug_flags=0x83(range_ready,flow_ready,attitude_ready)", primitive_outcome.abort_reason)
        self.assertIn("motion_squal=44", primitive_outcome.abort_reason)
        self.assertIn("hover_thrust_permille=530", primitive_outcome.abort_reason)

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
            status_snapshot=None,
            pre_arm_snapshot=None,
            estimator_settle=None,
            power=_MODULE.HoverPowerSnapshot(vbat_v=4.0, battery_level=70, state=0),
            status="completed",
            outcome_class="bounded_hover_ok",
            completed=True,
            landed=True,
            interrupted=False,
            primitive_outcome=None,
            telemetry=(),
            telemetry_summary=None,
            replay_start_timestamp_ms=None,
            failures=(),
            recommendations=(),
        )

        recommendations = _MODULE.recommendations_for_report(report)

        self.assertIn("Hover test completed and landed. The bounded flight primitive looks ready.", recommendations)

    def test_evaluate_primitive_outcome_surfaces_explicit_abort_reason(self) -> None:
        failures = _MODULE._evaluate_primitive_outcome(
            _MODULE.HoverPrimitiveOutcome(
                final_phase="abort_landing",
                took_off=False,
                landed=True,
                aborted=True,
                abort_reason="takeoff confirmation failed: downward range never exceeded 0.08 m within 1.50 s",
                commanded_max_height_m=0.25,
                setpoint_count=14,
            )
        )

        self.assertIn("hover primitive did not report a successful takeoff", failures)
        self.assertIn(
            "takeoff confirmation failed: downward range never exceeded 0.08 m within 1.50 s",
            failures,
        )

    def test_evaluate_primitive_outcome_keeps_abort_reason_after_takeoff(self) -> None:
        failures = _MODULE._evaluate_primitive_outcome(
            _MODULE.HoverPrimitiveOutcome(
                final_phase="abort_landing",
                took_off=True,
                landed=True,
                aborted=True,
                abort_reason="hover stability guard tripped: optical-flow quality 0 is below the 30 stability floor",
                commanded_max_height_m=0.10,
                setpoint_count=21,
                stable_hover_established=False,
            )
        )

        self.assertIn("hover primitive never established one bounded stable-hover window before hold", failures)
        self.assertIn(
            "hover stability guard tripped: optical-flow quality 0 is below the 30 stability floor",
            failures,
        )

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

    def test_latest_ground_distance_from_sitl_telemetry_uses_state_estimate_height(self) -> None:
        class _FakeTelemetryCollector:
            def latest_value(self, key: str) -> tuple[float | int | None, float | None]:
                if key == "stateEstimate.z":
                    return 0.23, 0.06
                if key == "supervisor.info":
                    return 16, 0.02
                raise AssertionError(f"unexpected key {key}")

        observation = _MODULE._latest_ground_distance_from_sitl_telemetry(_FakeTelemetryCollector())

        self.assertAlmostEqual(observation.distance_m or 0.0, 0.23, places=3)
        self.assertAlmostEqual(observation.age_s or 0.0, 0.06, places=3)
        self.assertTrue(observation.is_flying)
        self.assertAlmostEqual(observation.supervisor_age_s or 0.0, 0.02, places=3)

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
                    "stateEstimate.z": 0.21,
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
                    "range.zrange": 212,
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
                    "stateEstimate.z": 0.22,
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
                    "range.zrange": 208,
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
        self.assertEqual(summary.raw_sample_count, 4)
        self.assertTrue(summary.airborne_window_detected)
        self.assertEqual(summary.analysis_window_status, "ok")
        self.assertIn("hover-velocity", summary.available_blocks)
        self.assertTrue(summary.flow_observed)
        self.assertTrue(summary.raw_flow_observed)
        self.assertTrue(summary.zrange_observed)
        self.assertTrue(summary.clearance_observed)
        self.assertTrue(summary.stable_supervisor)
        self.assertAlmostEqual(summary.xy_drift_m or 0.0, 0.02236067977, places=6)
        self.assertAlmostEqual(summary.zrange_min_m or 0.0, 0.208, places=3)
        self.assertAlmostEqual(summary.trusted_height_min_m or 0.0, 0.208, places=3)
        self.assertAlmostEqual(summary.trusted_height_max_m or 0.0, 0.212, places=3)
        self.assertAlmostEqual(summary.front_min_m or 0.0, 0.405, places=3)
        self.assertAlmostEqual(summary.horizontal_speed_max_mps or 0.0, 0.0316227766, places=6)
        self.assertAlmostEqual(summary.thrust_max or 0.0, 25100.0, places=3)
        self.assertAlmostEqual(summary.radio_rssi_latest_dbm or 0.0, -55.0, places=3)
        self.assertAlmostEqual(summary.battery_drop_v or 0.0, 0.04, places=2)
        self.assertEqual(
            _MODULE.evaluate_hover_stability(
                summary,
                target_height_m=0.25,
                runtime_mode=_MODULE.HOVER_RUNTIME_MODE_HARDWARE,
            ),
            [],
        )

    def test_evaluate_hover_stability_allows_missing_flow_signal_in_sitl(self) -> None:
        samples = (
            _MODULE.HoverTelemetrySample(
                timestamp_ms=1000,
                block_name="hover-attitude",
                values={
                    "stabilizer.roll": 0.4,
                    "stabilizer.pitch": -0.5,
                    "stateEstimate.x": 0.00,
                    "stateEstimate.y": 0.00,
                    "stateEstimate.z": 0.21,
                },
            ),
            _MODULE.HoverTelemetrySample(
                timestamp_ms=1100,
                block_name="hover-sensors",
                values={
                    "range.zrange": 212,
                    "pm.vbat": 4.08,
                    "supervisor.info": 0,
                    "radio.isConnected": 1,
                },
            ),
            _MODULE.HoverTelemetrySample(
                timestamp_ms=1200,
                block_name="hover-velocity",
                values={
                    "stateEstimate.vx": 0.01,
                    "stateEstimate.vy": 0.00,
                    "stateEstimate.vz": 0.02,
                    "stabilizer.thrust": 24500.0,
                },
            ),
        )

        summary = _MODULE.summarize_hover_telemetry(
            samples,
            available_blocks=("hover-attitude", "hover-sensors", "hover-velocity", "hover-gyro"),
            skipped_blocks=(),
        )
        summary = replace(
            summary,
            flow_observed=False,
            flow_squal_min=None,
            flow_squal_mean=None,
            zrange_observed=False,
            zrange_min_m=None,
            zrange_max_m=None,
            zrange_sample_count=0,
        )

        failures = _MODULE.evaluate_hover_stability(
            summary,
            target_height_m=0.25,
            runtime_mode=_MODULE.HOVER_RUNTIME_MODE_SITL,
        )

        self.assertEqual(failures, [])

    def test_validate_runtime_config_defaults_to_no_required_decks_in_sitl(self) -> None:
        parser = _MODULE._build_parser()
        args = parser.parse_args(["--runtime-mode", "sitl"])

        config = _MODULE._validate_runtime_config(args)

        self.assertEqual(config["runtime_mode"], "sitl")
        self.assertEqual(config["required_decks"], ())

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

    def test_summarize_hover_telemetry_tracks_trusted_height_separately_from_range_outlier(self) -> None:
        samples = (
            _MODULE.HoverTelemetrySample(
                timestamp_ms=1000,
                block_name="hover-attitude",
                values={
                    "stateEstimate.x": 0.0,
                    "stateEstimate.y": 0.0,
                    "stateEstimate.z": 0.22,
                    "stabilizer.roll": 0.2,
                    "stabilizer.pitch": 0.1,
                },
            ),
            _MODULE.HoverTelemetrySample(
                timestamp_ms=1050,
                block_name="hover-sensors",
                values={
                    "range.zrange": 220,
                    "motion.squal": 110,
                    "supervisor.info": 0,
                },
            ),
            _MODULE.HoverTelemetrySample(
                timestamp_ms=1100,
                block_name="hover-velocity",
                values={
                    "stateEstimate.vx": 0.01,
                    "stateEstimate.vy": 0.00,
                    "stateEstimate.vz": 0.00,
                    "stabilizer.thrust": 30000.0,
                },
            ),
            _MODULE.HoverTelemetrySample(
                timestamp_ms=1200,
                block_name="hover-attitude",
                values={
                    "stateEstimate.x": 0.01,
                    "stateEstimate.y": 0.01,
                    "stateEstimate.z": 0.24,
                    "stabilizer.roll": 0.4,
                    "stabilizer.pitch": 0.3,
                },
            ),
            _MODULE.HoverTelemetrySample(
                timestamp_ms=1250,
                block_name="hover-sensors",
                values={
                    "range.zrange": 1008,
                    "motion.squal": 105,
                    "supervisor.info": 0,
                },
            ),
        )

        summary = _MODULE.summarize_hover_telemetry(
            samples,
            available_blocks=("hover-attitude", "hover-sensors", "hover-velocity"),
            skipped_blocks=(),
        )

        self.assertAlmostEqual(summary.zrange_max_m or 0.0, 1.008, places=3)
        self.assertAlmostEqual(summary.trusted_height_max_m or 0.0, 0.24, places=3)
        self.assertGreater(summary.height_sensor_untrusted_samples, 0)
        self.assertGreater(summary.height_sensor_disagreement_max_m or 0.0, 0.70)

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
            radio_connected_latest=True,
            radio_disconnect_seen=False,
            latest_supervisor_info=1 << 7,
            supervisor_flags_seen=("crashed",),
            stable_supervisor=False,
            trusted_height_min_m=None,
            trusted_height_max_m=None,
            height_sensor_disagreement_max_m=None,
            height_sensor_untrusted_samples=0,
        )

        failures = _MODULE.evaluate_hover_stability(
            summary,
            target_height_m=0.25,
            runtime_mode=_MODULE.HOVER_RUNTIME_MODE_HARDWARE,
        )

        self.assertIn("optical-flow quality never became nonzero during the hover test", failures)
        self.assertIn("downward z-range never produced a nonzero reading during the hover test", failures)
        self.assertIn("supervisor reported unsafe flags during the hover test: crashed", failures)
        self.assertIn("trusted hover height never became available during the hover test", failures)

    def test_evaluate_hover_stability_reports_missing_airborne_window_without_claiming_no_samples(self) -> None:
        summary = _MODULE.HoverTelemetrySummary(
            sample_count=0,
            available_blocks=("hover-attitude", "hover-sensors", "hover-velocity"),
            skipped_blocks=(),
            duration_s=None,
            roll_abs_max_deg=None,
            pitch_abs_max_deg=None,
            xy_drift_m=None,
            z_drift_m=None,
            z_span_m=None,
            vx_abs_max_mps=None,
            vy_abs_max_mps=None,
            vz_abs_max_mps=None,
            horizontal_speed_max_mps=None,
            flow_squal_min=None,
            flow_squal_mean=None,
            flow_nonzero_samples=0,
            flow_observed=False,
            motion_delta_x_abs_max=None,
            motion_delta_y_abs_max=None,
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
            thrust_mean=None,
            thrust_max=None,
            gyro_abs_max_dps=None,
            battery_min_v=None,
            battery_drop_v=None,
            radio_rssi_latest_dbm=None,
            radio_rssi_min_dbm=None,
            radio_connected_latest=True,
            radio_disconnect_seen=False,
            latest_supervisor_info=0,
            supervisor_flags_seen=(),
            stable_supervisor=True,
            trusted_height_min_m=None,
            trusted_height_max_m=None,
            height_sensor_disagreement_max_m=None,
            height_sensor_untrusted_samples=0,
            raw_sample_count=17,
            airborne_window_detected=False,
            analysis_window_status="raw_samples_missing_airborne_window",
            raw_flow_squal_min=95,
            raw_flow_squal_mean=101.0,
            raw_flow_nonzero_samples=17,
            raw_flow_observed=True,
        )

        failures = _MODULE.evaluate_hover_stability(
            summary,
            target_height_m=0.10,
            runtime_mode=_MODULE.HOVER_RUNTIME_MODE_HARDWARE,
        )

        self.assertIn(
            "telemetry was captured during the hover test, but neither downward range nor state-estimate height entered the airborne telemetry window",
            failures,
        )
        self.assertIn(
            "raw optical-flow quality was present, but the hover test never established an airborne telemetry window",
            failures,
        )
        self.assertNotIn("no in-flight telemetry samples were captured during the hover test", failures)
        self.assertNotIn(
            "optical-flow quality never became nonzero during the hover test",
            failures,
        )

    def test_summarize_hover_telemetry_classifies_takeoff_lateral_source_without_airborne_window(self) -> None:
        samples = (
            _MODULE.HoverTelemetrySample(
                timestamp_ms=1000,
                block_name="hover-failsafe",
                values={
                    "twinrFs.state": 6,
                    "twinrFs.cmdVx": 0.0,
                    "twinrFs.cmdVy": 0.0,
                    "twinrFs.cmdSrc": 1,
                    "twinrFs.distVx": 0.0,
                    "twinrFs.distVy": 0.0,
                },
            ),
            _MODULE.HoverTelemetrySample(
                timestamp_ms=1050,
                block_name="hover-velocity",
                values={
                    "stateEstimate.vx": 0.42,
                    "stateEstimate.vy": -0.18,
                    "stateEstimate.vz": 0.03,
                },
            ),
            _MODULE.HoverTelemetrySample(
                timestamp_ms=1100,
                block_name="hover-sensors",
                values={
                    "motion.squal": 97,
                    "range.zrange": 0,
                    "supervisor.info": 0,
                },
            ),
            _MODULE.HoverTelemetrySample(
                timestamp_ms=1200,
                block_name="hover-failsafe",
                values={
                    "twinrFs.state": 6,
                    "twinrFs.cmdVx": 0.0,
                    "twinrFs.cmdVy": 0.0,
                    "twinrFs.cmdSrc": 1,
                    "twinrFs.distVx": 0.0,
                    "twinrFs.distVy": 0.0,
                },
            ),
        )

        summary = _MODULE.summarize_hover_telemetry(
            samples,
            available_blocks=("hover-failsafe", "hover-velocity", "hover-sensors"),
            skipped_blocks=(),
        )

        self.assertEqual(summary.analysis_window_status, "raw_samples_missing_airborne_window")
        self.assertTrue(summary.raw_flow_observed)
        self.assertTrue(summary.takeoff_lateral_window_detected)
        self.assertEqual(summary.takeoff_command_source_codes_seen, (1,))
        self.assertEqual(summary.takeoff_commanded_vx_abs_max_mps, 0.0)
        self.assertEqual(summary.takeoff_commanded_vy_abs_max_mps, 0.0)
        self.assertAlmostEqual(summary.takeoff_estimated_vx_abs_max_mps or 0.0, 0.42, places=3)
        self.assertAlmostEqual(summary.takeoff_estimated_vy_abs_max_mps or 0.0, 0.18, places=3)
        self.assertEqual(
            summary.takeoff_lateral_classification,
            "no_on_device_lateral_command_or_estimator_bias_during_takeoff",
        )

    def test_evaluate_hover_stability_reports_target_height_shortfall(self) -> None:
        summary = _MODULE.HoverTelemetrySummary(
            sample_count=12,
            available_blocks=("hover-attitude", "hover-sensors"),
            skipped_blocks=(),
            duration_s=1.2,
            roll_abs_max_deg=2.0,
            pitch_abs_max_deg=1.8,
            xy_drift_m=0.03,
            z_drift_m=0.02,
            z_span_m=0.05,
            vx_abs_max_mps=0.05,
            vy_abs_max_mps=0.04,
            vz_abs_max_mps=0.08,
            horizontal_speed_max_mps=0.06,
            flow_squal_min=70,
            flow_squal_mean=90.0,
            flow_nonzero_samples=12,
            flow_observed=True,
            motion_delta_x_abs_max=3.0,
            motion_delta_y_abs_max=2.0,
            zrange_min_m=0.04,
            zrange_max_m=0.11,
            zrange_sample_count=12,
            zrange_observed=True,
            front_min_m=0.60,
            back_min_m=0.70,
            left_min_m=0.65,
            right_min_m=0.75,
            up_min_m=1.2,
            clearance_observed=True,
            thrust_mean=28000.0,
            thrust_max=32000.0,
            gyro_abs_max_dps=10.0,
            battery_min_v=3.90,
            battery_drop_v=0.08,
            radio_rssi_latest_dbm=-50.0,
            radio_rssi_min_dbm=-53.0,
            radio_connected_latest=True,
            radio_disconnect_seen=False,
            latest_supervisor_info=0,
            supervisor_flags_seen=(),
            stable_supervisor=True,
            trusted_height_min_m=0.04,
            trusted_height_max_m=0.11,
            height_sensor_disagreement_max_m=0.01,
            height_sensor_untrusted_samples=0,
        )

        failures = _MODULE.evaluate_hover_stability(
            summary,
            target_height_m=0.25,
            runtime_mode=_MODULE.HOVER_RUNTIME_MODE_HARDWARE,
        )

        self.assertIn(
            "hover reached only 0.11 m which is below the 0.20 m target-height gate for a 0.25 m hover",
            failures,
        )

    def test_evaluate_hover_stability_reports_excessive_drift_speed_and_attitude(self) -> None:
        summary = _MODULE.HoverTelemetrySummary(
            sample_count=20,
            available_blocks=("hover-attitude", "hover-sensors", "hover-clearance"),
            skipped_blocks=(),
            duration_s=2.1,
            roll_abs_max_deg=12.45,
            pitch_abs_max_deg=17.61,
            xy_drift_m=0.44,
            z_drift_m=0.06,
            z_span_m=0.12,
            vx_abs_max_mps=0.95,
            vy_abs_max_mps=0.57,
            vz_abs_max_mps=0.14,
            horizontal_speed_max_mps=1.11,
            flow_squal_min=55,
            flow_squal_mean=85.0,
            flow_nonzero_samples=20,
            flow_observed=True,
            motion_delta_x_abs_max=9.0,
            motion_delta_y_abs_max=7.0,
            zrange_min_m=0.09,
            zrange_max_m=0.25,
            zrange_sample_count=20,
            zrange_observed=True,
            front_min_m=0.80,
            back_min_m=1.10,
            left_min_m=0.90,
            right_min_m=0.95,
            up_min_m=1.6,
            clearance_observed=True,
            thrust_mean=33000.0,
            thrust_max=41000.0,
            gyro_abs_max_dps=26.0,
            battery_min_v=3.88,
            battery_drop_v=0.12,
            radio_rssi_latest_dbm=-48.0,
            radio_rssi_min_dbm=-52.0,
            radio_connected_latest=True,
            radio_disconnect_seen=False,
            latest_supervisor_info=0,
            supervisor_flags_seen=(),
            stable_supervisor=True,
            trusted_height_min_m=0.09,
            trusted_height_max_m=0.25,
            height_sensor_disagreement_max_m=0.02,
            height_sensor_untrusted_samples=0,
        )

        failures = _MODULE.evaluate_hover_stability(
            summary,
            target_height_m=0.25,
            runtime_mode=_MODULE.HOVER_RUNTIME_MODE_HARDWARE,
        )

        self.assertIn("roll reached 12.45 deg which exceeds the 10.00 deg hover stability limit", failures)
        self.assertIn("pitch reached 17.61 deg which exceeds the 10.00 deg hover stability limit", failures)
        self.assertIn("xy drift reached 0.44 m which exceeds the 0.20 m hover stability limit", failures)
        self.assertIn(
            "horizontal speed reached 1.11 m/s which exceeds the 0.40 m/s hover stability limit",
            failures,
        )

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
            radio_connected_latest=True,
            radio_disconnect_seen=False,
            latest_supervisor_info=0,
            supervisor_flags_seen=(),
            stable_supervisor=True,
            trusted_height_min_m=0.18,
            trusted_height_max_m=0.88,
            height_sensor_disagreement_max_m=0.01,
            height_sensor_untrusted_samples=0,
        )

        failures = _MODULE.evaluate_hover_stability(
            summary,
            target_height_m=0.25,
            runtime_mode=_MODULE.HOVER_RUNTIME_MODE_HARDWARE,
        )

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
                self.front = None
                self.back = None
                self.left = None
                self.right = None
                self.up = None
                self.down = None

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

        original_import = _MODULE._import_cflib
        original_read_deck_flags = _MODULE._read_deck_flags
        original_read_preflight_snapshots = _MODULE._read_preflight_snapshots
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
            _MODULE._read_preflight_snapshots = (
                lambda *_args, **_kwargs: (
                    _MODULE.HoverPowerSnapshot(vbat_v=4.0, battery_level=90, state=0),
                    _safe_status_snapshot(),
                )
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
                                "stateEstimate.z": 0.21,
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
                                "range.zrange": 210,
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

                def run(self, config: Any, *, after_takeoff: Any | None = None) -> object:
                    events.append(
                        "primitive_run:"
                        f"{config.target_height_m:.2f}:{config.hover_duration_s:.2f}:{config.takeoff_velocity_mps:.2f}"
                    )
                    if after_takeoff is not None:
                        after_takeoff()
                    return _MODULE.HoverPrimitiveOutcome(
                        final_phase="landed",
                        took_off=True,
                        landed=True,
                        aborted=False,
                        abort_reason=None,
                        commanded_max_height_m=config.target_height_m,
                        setpoint_count=12,
                        stable_hover_established=True,
                        touchdown_confirmation_source="range+supervisor",
                        touchdown_supervisor_grounded=True,
                    )

            _MODULE.HoverTelemetryCollector = _FakeTelemetryCollector
            _MODULE.StatefulHoverPrimitive = _FakeStatefulHoverPrimitive
            _MODULE.time.sleep = lambda _seconds: events.append("sleep")
            with tempfile.TemporaryDirectory() as temp_dir:
                trace_path = Path(temp_dir) / "hover-trace.jsonl"
                report = _MODULE.run_hover_test(
                    runtime_mode=_MODULE.HOVER_RUNTIME_MODE_SITL,
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
                    required_decks=(),
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
            _MODULE._read_preflight_snapshots = original_read_preflight_snapshots
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

    def test_run_hover_test_disables_link_health_gate_in_sitl(self) -> None:
        captured_link_health: list[object] = []

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
                self.front = None
                self.back = None
                self.left = None
                self.right = None
                self.up = None
                self.down = None

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

        original_import = _MODULE._import_cflib
        original_read_deck_flags = _MODULE._read_deck_flags
        original_read_preflight_snapshots = _MODULE._read_preflight_snapshots
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
            _MODULE._read_deck_flags = lambda *_args, **_kwargs: {"bcFlow2": 1, "bcMultiranger": 1}
            _MODULE._read_preflight_snapshots = (
                lambda *_args, **_kwargs: (
                    _MODULE.HoverPowerSnapshot(vbat_v=4.2, battery_level=None, state=0),
                    _MODULE.HoverStatusSnapshot(
                        supervisor_info=14,
                        can_arm=False,
                        is_armed=True,
                        auto_arm=True,
                        can_fly=True,
                        is_flying=False,
                        tumbled=False,
                        locked=False,
                        crashed=False,
                        hl_flying=False,
                        hl_trajectory_finished=False,
                        hl_disabled=False,
                        radio_connected=True,
                        zrange_m=None,
                        motion_squal=None,
                    ),
                )
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
                var_py_span=0.0003,
                var_pz_span=0.0003,
                roll_abs_max_deg=0.0,
                pitch_abs_max_deg=0.0,
                motion_squal_min=None,
                motion_squal_mean=None,
                motion_squal_nonzero_ratio=None,
                zrange_min_m=None,
                zrange_observed=False,
                failures=(),
            )

            class _FakeTelemetryCollector:
                def __init__(self, *_args, **_kwargs) -> None:
                    self.available_blocks = ("hover-attitude", "hover-sensors", "hover-velocity")
                    self.skipped_blocks = ("hover-failsafe",)

                def start(self) -> None:
                    return None

                def stop(self) -> None:
                    return None

                def latest_value(self, key: str) -> tuple[float | int | None, float | None]:
                    values = {
                        "stateEstimate.z": (0.25, 0.01),
                        "supervisor.info": (30, 0.01),
                        "radio.isConnected": (1, 0.01),
                        "pm.vbat": (4.2, 0.01),
                    }
                    return values.get(key, (None, None))

                def snapshot(self):
                    return ()

                def link_health_observation(self):
                    return type("_LinkObservation", (), {"age_s": 0.01, "latency_ms": 999.0})()

            class _FakeStatefulHoverPrimitive:
                def __init__(self, *_args, **_kwargs) -> None:
                    return None

                def run(self, config: Any, *, after_takeoff: Any | None = None) -> object:
                    captured_link_health.append(config.link_health)
                    if after_takeoff is not None:
                        after_takeoff()
                    return _MODULE.HoverPrimitiveOutcome(
                        final_phase="landed",
                        took_off=True,
                        landed=True,
                        aborted=False,
                        abort_reason=None,
                        commanded_max_height_m=config.target_height_m,
                        setpoint_count=8,
                        stable_hover_established=True,
                        touchdown_confirmation_source="range_only_sitl",
                        touchdown_supervisor_grounded=False,
                    )

            _MODULE.HoverTelemetryCollector = _FakeTelemetryCollector
            _MODULE.StatefulHoverPrimitive = _FakeStatefulHoverPrimitive
            _MODULE.time.sleep = lambda _seconds: None
            with tempfile.TemporaryDirectory() as temp_dir:
                report = _MODULE.run_hover_test(
                    runtime_mode=_MODULE.HOVER_RUNTIME_MODE_SITL,
                    uri="udp://127.0.0.1:19850",
                    workspace=Path(temp_dir),
                    height_m=0.25,
                    hover_duration_s=1.0,
                    takeoff_velocity_mps=0.2,
                    land_velocity_mps=0.2,
                    connect_settle_s=0.0,
                    min_vbat_v=0.0,
                    min_battery_level=0,
                    min_clearance_m=0.0,
                    stabilizer_estimator=2,
                    stabilizer_controller=1,
                    motion_disable=0,
                    estimator_settle_timeout_s=5.0,
                    on_device_failsafe_mode="off",
                    on_device_failsafe_heartbeat_timeout_s=0.35,
                    on_device_failsafe_low_battery_v=3.55,
                    on_device_failsafe_critical_battery_v=3.35,
                    on_device_failsafe_min_up_clearance_m=0.25,
                    required_decks=(),
                )
        finally:
            _MODULE._import_cflib = original_import
            _MODULE._read_deck_flags = original_read_deck_flags
            _MODULE._read_preflight_snapshots = original_read_preflight_snapshots
            _MODULE.HoverTelemetryCollector = original_telemetry_collector
            _MODULE.apply_hover_pre_arm = original_apply_pre_arm
            _MODULE.wait_for_estimator_settle = original_wait_for_estimator_settle
            _MODULE.StatefulHoverPrimitive = original_stateful_hover_primitive
            _MODULE.time.sleep = original_sleep

        self.assertIsNotNone(report.primitive_outcome)
        self.assertEqual(captured_link_health, [None])


if __name__ == "__main__":
    unittest.main()
