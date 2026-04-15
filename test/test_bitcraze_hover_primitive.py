"""Regression coverage for the bounded Crazyflie hover primitive helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any
import unittest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "hardware" / "bitcraze" / "hover_primitive.py"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
_SPEC = importlib.util.spec_from_file_location("bitcraze_hover_primitive_module", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE: Any = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class _FakeClock:
    def __init__(self) -> None:
        self.current = 0.0

    def monotonic(self) -> float:
        return self.current

    def sleep(self, seconds: float) -> None:
        self.current += max(0.0, float(seconds))


class _TraceRecorder:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(
        self,
        phase: str,
        *,
        status: str,
        message: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.events.append(
            {
                "phase": phase,
                "status": status,
                "message": message,
                "data": data,
            }
        )


def _stable_hover_observation(
    *,
    height_m: float = 0.20,
    x_m: float = 0.0,
    y_m: float = 0.0,
    vx_mps: float = 0.0,
    vy_mps: float = 0.0,
    vz_mps: float = 0.0,
    roll_deg: float = 0.4,
    pitch_deg: float = -0.4,
    yaw_deg: float = 0.0,
    motion_squal: int = 95,
    is_flying: bool = True,
) -> Any:
    return _MODULE.HoverStabilityObservation(
        height_m=height_m,
        height_age_s=0.02,
        z_estimate_m=height_m,
        z_estimate_age_s=0.02,
        x_m=x_m,
        y_m=y_m,
        pose_age_s=0.02,
        vx_mps=vx_mps,
        vy_mps=vy_mps,
        velocity_age_s=0.02,
        vz_mps=vz_mps,
        vz_age_s=0.02,
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
        yaw_deg=yaw_deg,
        yaw_age_s=0.02,
        attitude_age_s=0.02,
        motion_squal=motion_squal,
        motion_squal_age_s=0.02,
        is_flying=is_flying,
        supervisor_age_s=0.02,
    )


def _dynamic_stability_provider(
    holder: dict[str, Any],
    *,
    min_height_m: float = 0.10,
    x_m: float = 0.0,
    y_m: float = 0.0,
    vx_mps: float = 0.0,
    vy_mps: float = 0.0,
    roll_deg: float = 0.4,
    pitch_deg: float = -0.4,
    motion_squal: int = 95,
) -> Any:
    def _observe() -> Any:
        primitive = holder.get("primitive")
        current_height_m = min_height_m
        if primitive is not None:
            current_height_m = max(min_height_m, float(getattr(primitive, "_current_height_m", min_height_m)))
        return _stable_hover_observation(
            height_m=current_height_m,
            x_m=x_m,
            y_m=y_m,
            vx_mps=vx_mps,
            vy_mps=vy_mps,
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            motion_squal=motion_squal,
        )

    return _observe


class HoverPrimitiveHelperTests(unittest.TestCase):
    def test_apply_hover_pre_arm_sets_and_verifies_params(self) -> None:
        writes: list[tuple[str, str]] = []

        class _FakeParam:
            def __init__(self) -> None:
                self.values = {
                    "stabilizer.estimator": "0",
                    "stabilizer.controller": "0",
                    "motion.disable": "1",
                    "kalman.resetEstimation": "0",
                }
                self._reset_reads_after_zero = 0

            def set_value(self, name: str, value: str) -> None:
                writes.append((name, value))
                self.values[name] = value
                if name == "kalman.resetEstimation" and value == "0":
                    self._reset_reads_after_zero = 0

            def get_value(self, name: str) -> str | None:
                if name == "kalman.resetEstimation" and self.values[name] == "0" and self._reset_reads_after_zero < 1:
                    self._reset_reads_after_zero += 1
                    return "1"
                return self.values.get(name)

        class _FakeCF:
            def __init__(self) -> None:
                self.param = _FakeParam()

        snapshot = _MODULE.apply_hover_pre_arm(
            _FakeCF(),
            config=_MODULE.HoverPreArmConfig(estimator=2, controller=1, motion_disable=0),
            sleep=lambda _seconds: None,
        )

        self.assertTrue(snapshot.verified)
        self.assertEqual(snapshot.estimator, 2)
        self.assertEqual(snapshot.controller, 1)
        self.assertEqual(snapshot.motion_disable, 0)
        self.assertEqual(snapshot.kalman_reset_after, 0)
        self.assertIn(("kalman.resetEstimation", "1"), writes)
        self.assertIn(("kalman.resetEstimation", "0"), writes)

    def test_apply_hover_pre_arm_allows_missing_motion_disable_param_when_disabled(self) -> None:
        writes: list[tuple[str, str]] = []

        class _FakeParam:
            def __init__(self) -> None:
                self.values = {
                    "stabilizer.estimator": "0",
                    "stabilizer.controller": "0",
                    "kalman.resetEstimation": "0",
                }
                self._reset_reads_after_zero = 0

            def set_value(self, name: str, value: str) -> None:
                writes.append((name, value))
                self.values[name] = value
                if name == "kalman.resetEstimation" and value == "0":
                    self._reset_reads_after_zero = 0

            def get_value(self, name: str) -> str | None:
                if name == "kalman.resetEstimation" and self.values[name] == "0" and self._reset_reads_after_zero < 1:
                    self._reset_reads_after_zero += 1
                    return "1"
                if name == "motion.disable":
                    raise KeyError(name)
                return self.values.get(name)

        class _FakeCF:
            def __init__(self) -> None:
                self.param = _FakeParam()

        snapshot = _MODULE.apply_hover_pre_arm(
            _FakeCF(),
            config=_MODULE.HoverPreArmConfig(
                estimator=2,
                controller=1,
                motion_disable=0,
                require_motion_disable_param=False,
            ),
            sleep=lambda _seconds: None,
        )

        self.assertTrue(snapshot.verified)
        self.assertEqual(snapshot.motion_disable, None)
        self.assertNotIn(("motion.disable", "0"), writes)

    def test_wait_for_estimator_settle_reports_stable_window(self) -> None:
        class _FakeLogConfig:
            def __init__(self, *, name: str, period_in_ms: int) -> None:
                self.name = name
                self.period_in_ms = period_in_ms
                self.variables: list[tuple[str, str | None]] = []

            def add_variable(self, name: str, fetch_as: str | None = None) -> None:
                self.variables.append((name, fetch_as))

        rows = [
            (
                index * 100,
                {
                    "kalman.varPX": 0.00010 + (index * 0.00001),
                    "kalman.varPY": 0.00011 + (index * 0.00001),
                    "kalman.varPZ": 0.00012 + (index * 0.00001),
                    "motion.squal": 90,
                    "stabilizer.roll": 0.4,
                    "stabilizer.pitch": -0.5,
                    "range.zrange": 120,
                },
                object(),
            )
            for index in range(10)
        ]

        class _FakeSyncLogger:
            def __init__(self, _sync_cf: object, _config: object) -> None:
                self._rows = rows

            def __enter__(self):
                return iter(self._rows)

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

        clock = _FakeClock()
        report = _MODULE.wait_for_estimator_settle(
            object(),
            _FakeLogConfig,
            _FakeSyncLogger,
            config=_MODULE.HoverEstimatorSettlingConfig(timeout_s=2.0, window_size=10),
            monotonic=clock.monotonic,
        )

        self.assertTrue(report.stable)
        self.assertEqual(report.sample_count, 10)
        self.assertAlmostEqual(report.motion_squal_mean or 0.0, 90.0, places=3)
        self.assertAlmostEqual(report.zrange_min_m or 0.0, 0.120, places=3)
        self.assertAlmostEqual(report.zrange_max_m or 0.0, 0.120, places=3)
        self.assertFalse(report.flow_gate_deferred)
        self.assertEqual(report.failures, ())

    def test_wait_for_estimator_settle_defers_flow_gate_while_grounded(self) -> None:
        class _FakeLogConfig:
            def __init__(self, *, name: str, period_in_ms: int) -> None:
                self.name = name
                self.period_in_ms = period_in_ms
                self.variables: list[tuple[str, str | None]] = []

            def add_variable(self, name: str, fetch_as: str | None = None) -> None:
                self.variables.append((name, fetch_as))

        rows = [
            (
                index * 100,
                {
                    "kalman.varPX": 0.020 + (index * 0.002),
                    "kalman.varPY": 0.021 + (index * 0.002),
                    "kalman.varPZ": 0.00012 + (index * 0.00001),
                    "motion.squal": 5,
                    "stabilizer.roll": 0.4,
                    "stabilizer.pitch": -0.5,
                    "range.zrange": 15,
                },
                object(),
            )
            for index in range(10)
        ]

        class _FakeSyncLogger:
            def __init__(self, _sync_cf: object, _config: object) -> None:
                self._rows = rows

            def __enter__(self):
                return iter(self._rows)

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

        clock = _FakeClock()
        report = _MODULE.wait_for_estimator_settle(
            object(),
            _FakeLogConfig,
            _FakeSyncLogger,
            config=_MODULE.HoverEstimatorSettlingConfig(timeout_s=2.0, window_size=10),
            monotonic=clock.monotonic,
        )

        self.assertTrue(report.stable)
        self.assertTrue(report.flow_gate_deferred)
        self.assertAlmostEqual(report.zrange_min_m or 0.0, 0.015, places=3)
        self.assertAlmostEqual(report.zrange_max_m or 0.0, 0.015, places=3)
        self.assertEqual(report.failures, ())

    def test_wait_for_estimator_settle_blocks_timeout_and_bad_flow_quality(self) -> None:
        class _FakeLogConfig:
            def __init__(self, *, name: str, period_in_ms: int) -> None:
                self.name = name
                self.period_in_ms = period_in_ms

            def add_variable(self, _name: str, _fetch_as: str | None = None) -> None:
                return None

        rows = [
            (
                index * 100,
                {
                    "kalman.varPX": 0.005,
                    "kalman.varPY": 0.004,
                    "kalman.varPZ": 0.006,
                    "motion.squal": 5,
                    "stabilizer.roll": 0.1,
                    "stabilizer.pitch": 0.2,
                    "range.zrange": 0,
                },
                object(),
            )
            for index in range(8)
        ]

        class _FakeSyncLogger:
            def __init__(self, _sync_cf: object, _config: object) -> None:
                self._rows = rows

            def __enter__(self):
                return iter(self._rows)

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

        clock = _FakeClock()

        def _monotonic() -> float:
            clock.current += 0.25
            return clock.current

        report = _MODULE.wait_for_estimator_settle(
            object(),
            _FakeLogConfig,
            _FakeSyncLogger,
            config=_MODULE.HoverEstimatorSettlingConfig(timeout_s=1.0, window_size=4, motion_squal_min=30),
            monotonic=_monotonic,
        )

        self.assertFalse(report.stable)
        self.assertTrue(any("timed out" in failure for failure in report.failures))
        self.assertTrue(any("motion.squal" in failure for failure in report.failures))

    def test_wait_for_estimator_settle_allows_sitl_without_motion_squal(self) -> None:
        class _FakeLogConfig:
            def __init__(self, *, name: str, period_in_ms: int) -> None:
                self.name = name
                self.period_in_ms = period_in_ms
                self.variables: list[tuple[str, str | None]] = []

            def add_variable(self, name: str, fetch_as: str | None = None) -> None:
                self.variables.append((name, fetch_as))

        rows = [
            (
                index * 100,
                {
                    "kalman.varPX": 0.00010 + (index * 0.00001),
                    "kalman.varPY": 0.00011 + (index * 0.00001),
                    "kalman.varPZ": 0.00012 + (index * 0.00001),
                    "stabilizer.roll": 0.4,
                    "stabilizer.pitch": -0.5,
                },
                object(),
            )
            for index in range(10)
        ]

        class _FakeSyncLogger:
            def __init__(self, _sync_cf: object, _config: object) -> None:
                self._rows = rows

            def __enter__(self):
                return iter(self._rows)

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

        clock = _FakeClock()
        report = _MODULE.wait_for_estimator_settle(
            object(),
            _FakeLogConfig,
            _FakeSyncLogger,
            config=_MODULE.HoverEstimatorSettlingConfig(
                timeout_s=2.0,
                window_size=10,
                require_motion_squal=False,
                require_ground_range=False,
            ),
            monotonic=clock.monotonic,
        )

        self.assertTrue(report.stable)
        self.assertIsNone(report.motion_squal_mean)
        self.assertEqual(report.failures, ())

    def test_stateful_hover_primitive_runs_takeoff_hold_and_land(self) -> None:
        hover_calls: list[float] = []
        stop_calls: list[str] = []
        touchdown_observations = iter(
            (
                _MODULE.HoverGroundDistanceObservation(distance_m=0.16, age_s=0.02, is_flying=True, supervisor_age_s=0.02),
                _MODULE.HoverGroundDistanceObservation(distance_m=0.17, age_s=0.02, is_flying=True, supervisor_age_s=0.02),
                _MODULE.HoverGroundDistanceObservation(distance_m=0.18, age_s=0.02, is_flying=True, supervisor_age_s=0.02),
                _MODULE.HoverGroundDistanceObservation(distance_m=0.12, age_s=0.02),
                _MODULE.HoverGroundDistanceObservation(distance_m=0.09, age_s=0.02),
                _MODULE.HoverGroundDistanceObservation(distance_m=0.049, age_s=0.02, is_flying=False, supervisor_age_s=0.02),
                _MODULE.HoverGroundDistanceObservation(distance_m=0.045, age_s=0.02, is_flying=False, supervisor_age_s=0.02),
                _MODULE.HoverGroundDistanceObservation(distance_m=0.041, age_s=0.02, is_flying=False, supervisor_age_s=0.02),
            )
        )

        class _FakeCommander:
            def send_hover_setpoint(self, _vx: float, _vy: float, _yawrate: float, zdistance: float) -> None:
                hover_calls.append(zdistance)

            def send_stop_setpoint(self) -> None:
                stop_calls.append("stop")

            def send_notify_setpoint_stop(self) -> None:
                stop_calls.append("notify")

        class _FakeCF:
            def __init__(self) -> None:
                self.commander = _FakeCommander()

        clock = _FakeClock()
        primitive_holder: dict[str, Any] = {"primitive": None}
        primitive = _MODULE.StatefulHoverPrimitive(
            _FakeCF(),
            ground_distance_provider=lambda: next(touchdown_observations),
            stability_provider=_dynamic_stability_provider(primitive_holder, min_height_m=0.10),
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )
        primitive_holder["primitive"] = primitive
        outcome = primitive.run(
            _MODULE.HoverPrimitiveConfig(
                target_height_m=0.2,
                hover_duration_s=0.3,
                takeoff_velocity_mps=0.2,
                land_velocity_mps=0.2,
                setpoint_period_s=0.1,
                stability=_MODULE.HoverStabilityConfig(required_stable_samples=1),
            )
        )

        self.assertTrue(outcome.took_off)
        self.assertTrue(outcome.landed)
        self.assertFalse(outcome.aborted)
        self.assertEqual(outcome.final_phase, "landed")
        self.assertGreater(outcome.setpoint_count, 0)
        self.assertAlmostEqual(max(hover_calls), 0.2, places=3)
        self.assertGreaterEqual(sum(1 for value in hover_calls if abs(value - 0.08) < 1e-6), 2)
        self.assertGreaterEqual(sum(1 for value in hover_calls if abs(value - 0.03) < 1e-6), 2)
        self.assertGreaterEqual(sum(1 for value in hover_calls if abs(value - 0.0) < 1e-6), 1)
        self.assertEqual(stop_calls.count("stop"), _MODULE.HOVER_STOP_SETPOINT_REPEAT)
        self.assertIn("notify", stop_calls)

    def test_stateful_hover_primitive_translate_body_maintains_hover_anchor(self) -> None:
        hover_calls: list[tuple[float, float, float, float]] = []

        class _FakeCommander:
            def send_hover_setpoint(self, vx: float, vy: float, yawrate: float, zdistance: float) -> None:
                hover_calls.append((vx, vy, yawrate, zdistance))

            def send_stop_setpoint(self) -> None:
                return None

            def send_notify_setpoint_stop(self) -> None:
                return None

        class _FakeCF:
            def __init__(self) -> None:
                self.commander = _FakeCommander()

        clock = _FakeClock()
        primitive_holder: dict[str, Any] = {"primitive": None}
        primitive = _MODULE.StatefulHoverPrimitive(
            _FakeCF(),
            ground_distance_provider=lambda: _MODULE.HoverGroundDistanceObservation(distance_m=0.18, age_s=0.02, is_flying=True, supervisor_age_s=0.02),
            stability_provider=_dynamic_stability_provider(primitive_holder, min_height_m=0.10),
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )
        primitive_holder["primitive"] = primitive

        primitive.takeoff(
            target_height_m=0.20,
            velocity_mps=0.20,
            stability_config=_MODULE.HoverStabilityConfig(required_stable_samples=1),
        )
        primitive.translate_body(
            _MODULE.HoverTranslationConfig(
                forward_m=0.12,
                left_m=0.06,
                velocity_mps=0.12,
                target_height_m=0.24,
                settle_duration_s=0.20,
            )
        )
        outcome = primitive.current_outcome(final_phase=None, aborted=False, abort_reason=None)

        translating_calls = [
            call for call in hover_calls
            if abs(call[0]) > 1e-6 or abs(call[1]) > 1e-6
        ]
        self.assertTrue(translating_calls)
        self.assertTrue(all(abs(call[2]) < 1e-6 for call in translating_calls))
        self.assertTrue(all(abs(call[3] - 0.24) < 1e-6 for call in translating_calls))
        self.assertTrue(any(call[0] > 0.0 for call in translating_calls))
        self.assertTrue(any(call[1] > 0.0 for call in translating_calls))
        self.assertTrue(outcome.took_off)
        self.assertFalse(outcome.landed)
        self.assertGreater(outcome.commanded_max_height_m, 0.23)

    def test_stateful_hover_primitive_micro_liftoff_uses_configured_confirm_timeout(self) -> None:
        hover_calls: list[float] = []

        class _FakeCommander:
            def send_hover_setpoint(self, _vx: float, _vy: float, _yawrate: float, zdistance: float) -> None:
                hover_calls.append(zdistance)

            def send_stop_setpoint(self) -> None:
                return None

            def send_notify_setpoint_stop(self) -> None:
                return None

        class _FakeCF:
            def __init__(self) -> None:
                self.commander = _FakeCommander()

        clock = _FakeClock()
        primitive_holder: dict[str, Any] = {"primitive": None}

        def _observe_ground_distance() -> Any:
            if clock.monotonic() < 2.05:
                distance_m = 0.04
            else:
                distance_m = 0.10
            return _MODULE.HoverGroundDistanceObservation(
                distance_m=distance_m,
                age_s=0.02,
                is_flying=True,
                supervisor_age_s=0.02,
            )

        primitive = _MODULE.StatefulHoverPrimitive(
            _FakeCF(),
            ground_distance_provider=_observe_ground_distance,
            stability_provider=_dynamic_stability_provider(primitive_holder, min_height_m=0.10),
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )
        primitive_holder["primitive"] = primitive

        primitive.takeoff(
            target_height_m=0.10,
            velocity_mps=0.05,
            takeoff_confirm_timeout_s=0.8,
            stability_config=_MODULE.HoverStabilityConfig(required_stable_samples=1),
        )
        outcome = primitive.current_outcome(final_phase="takeoff_only", aborted=False, abort_reason=None)

        self.assertTrue(outcome.took_off)
        self.assertTrue(outcome.trim_identified)
        self.assertFalse(outcome.aborted)
        self.assertGreaterEqual(max(hover_calls), 0.10)

    def test_stateful_hover_primitive_vertical_bootstrap_waits_for_sensor_liveness(self) -> None:
        command_log: list[tuple[str, float]] = []
        trace_writer = _TraceRecorder()

        class _FakeCommander:
            def send_setpoint_manual(
                self,
                _roll: float,
                _pitch: float,
                _yawrate: float,
                thrust_percentage: float,
                _rate: bool,
            ) -> None:
                command_log.append(("manual", thrust_percentage))

            def send_hover_setpoint(self, _vx: float, _vy: float, _yawrate: float, zdistance: float) -> None:
                command_log.append(("hover", zdistance))

            def send_stop_setpoint(self) -> None:
                return None

            def send_notify_setpoint_stop(self) -> None:
                return None

        class _FakeCF:
            def __init__(self) -> None:
                self.commander = _FakeCommander()

        clock = _FakeClock()
        primitive_holder: dict[str, Any] = {"primitive": None}

        def _observe_ground_distance() -> Any:
            distance_m = 0.02
            if clock.monotonic() >= 0.30:
                distance_m = 0.09
            return _MODULE.HoverGroundDistanceObservation(
                distance_m=distance_m,
                age_s=0.02,
                is_flying=clock.monotonic() >= 0.30,
                supervisor_age_s=0.02,
            )

        def _observe_stability() -> Any:
            primitive = primitive_holder.get("primitive")
            current_height_m = 0.10
            if primitive is not None:
                current_height_m = max(0.10, float(getattr(primitive, "_current_height_m", 0.10)))
            motion_squal = 0 if clock.monotonic() < 0.30 else 85
            return _stable_hover_observation(
                height_m=current_height_m,
                motion_squal=motion_squal,
            )

        primitive = _MODULE.StatefulHoverPrimitive(
            _FakeCF(),
            ground_distance_provider=_observe_ground_distance,
            stability_provider=_observe_stability,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
            trace_writer=trace_writer,
        )
        primitive_holder["primitive"] = primitive

        primitive.takeoff(
            target_height_m=0.10,
            velocity_mps=0.05,
            vertical_bootstrap_config=_MODULE.HoverVerticalBootstrapConfig(
                target_height_m=0.08,
                min_thrust_percentage=36.0,
                feedforward_thrust_percentage=40.0,
                max_thrust_percentage=44.0,
                max_duration_s=0.8,
                reference_duration_s=0.6,
                progress_to_ceiling_s=0.35,
                height_gain_per_m=120.0,
                vertical_speed_gain_per_mps=30.0,
                min_range_height_m=0.08,
                max_range_height_m=0.13,
                min_range_rise_m=0.02,
                max_observation_age_s=0.35,
                max_ceiling_without_progress_s=0.45,
            ),
            stability_config=_MODULE.HoverStabilityConfig(required_stable_samples=1),
        )
        outcome = primitive.current_outcome(final_phase="takeoff_only", aborted=False, abort_reason=None)

        self.assertTrue(outcome.took_off)
        self.assertTrue(outcome.trim_identified)
        manual_values = [
            value for command_kind, value in command_log if command_kind == "manual"
        ]
        self.assertGreaterEqual(len(manual_values), 1)
        self.assertGreater(max(manual_values), min(manual_values))
        self.assertLessEqual(max(manual_values), 44.0)
        first_hover_index = next(
            index for index, (command_kind, _value) in enumerate(command_log)
            if command_kind == "hover"
        )
        self.assertTrue(
            all(command_kind == "manual" for command_kind, _value in command_log[:first_hover_index])
        )
        bootstrap_tick_events = [
            event
            for event in trace_writer.events
            if event["phase"] == "hover_primitive_vertical_bootstrap_tick"
        ]
        self.assertGreaterEqual(len(bootstrap_tick_events), 1)
        last_tick = bootstrap_tick_events[-1]["data"]
        assert last_tick is not None
        self.assertIn("raw_commanded_thrust_percentage", last_tick)
        self.assertIn("thrust_headroom_percentage", last_tick)
        self.assertIn("progress_boost_percentage", last_tick)
        self.assertIn("range_live", last_tick)
        self.assertIn("flow_live", last_tick)
        bootstrap_done_events = [
            event
            for event in trace_writer.events
            if event["phase"] == "hover_primitive_vertical_bootstrap"
            and event["status"] == "done"
        ]
        self.assertEqual(len(bootstrap_done_events), 1)
        done_data = bootstrap_done_events[0]["data"]
        assert done_data is not None
        self.assertIn("max_commanded_thrust_percentage", done_data)
        self.assertIn("max_progress_boost_percentage", done_data)
        self.assertIn("tick_count", done_data)

    def test_stateful_hover_primitive_vertical_bootstrap_aborts_without_hover_handoff(self) -> None:
        manual_calls: list[float] = []
        hover_calls: list[float] = []
        stop_calls: list[str] = []

        class _FakeCommander:
            def send_setpoint_manual(
                self,
                _roll: float,
                _pitch: float,
                _yawrate: float,
                thrust_percentage: float,
                _rate: bool,
            ) -> None:
                manual_calls.append(thrust_percentage)

            def send_hover_setpoint(self, _vx: float, _vy: float, _yawrate: float, zdistance: float) -> None:
                hover_calls.append(zdistance)

            def send_stop_setpoint(self) -> None:
                stop_calls.append("stop")

            def send_notify_setpoint_stop(self) -> None:
                stop_calls.append("notify")

        class _FakeCF:
            def __init__(self) -> None:
                self.commander = _FakeCommander()

        clock = _FakeClock()
        primitive_holder: dict[str, Any] = {"primitive": None}

        primitive = _MODULE.StatefulHoverPrimitive(
            _FakeCF(),
            ground_distance_provider=lambda: _MODULE.HoverGroundDistanceObservation(
                distance_m=0.006,
                age_s=0.02,
                is_flying=False,
                supervisor_age_s=0.02,
            ),
            stability_provider=lambda: _stable_hover_observation(
                height_m=0.006,
                motion_squal=0,
                is_flying=False,
            ),
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )
        primitive_holder["primitive"] = primitive

        outcome = primitive.run(
            _MODULE.HoverPrimitiveConfig(
                target_height_m=0.10,
                hover_duration_s=0.2,
                takeoff_velocity_mps=0.05,
                land_velocity_mps=0.05,
                setpoint_period_s=0.1,
                stability=_MODULE.HoverStabilityConfig(required_stable_samples=1),
                vertical_bootstrap=_MODULE.HoverVerticalBootstrapConfig(
                    target_height_m=0.08,
                    min_thrust_percentage=36.0,
                    feedforward_thrust_percentage=40.0,
                    max_thrust_percentage=44.0,
                    max_duration_s=0.3,
                    reference_duration_s=0.25,
                    progress_to_ceiling_s=0.20,
                    height_gain_per_m=120.0,
                    vertical_speed_gain_per_mps=30.0,
                    min_range_height_m=0.08,
                    max_range_height_m=0.13,
                    min_range_rise_m=0.02,
                    max_observation_age_s=0.35,
                    max_ceiling_without_progress_s=0.20,
                ),
            )
        )

        self.assertTrue(outcome.aborted)
        self.assertTrue(outcome.landed)
        self.assertFalse(outcome.took_off)
        self.assertTrue(
            (
                "vertical bootstrap did not establish fresh z-range/flow liveness"
                in (outcome.abort_reason or "")
            )
            or (
                "vertical bootstrap saturated at bounded thrust without proving lift"
                in (outcome.abort_reason or "")
            )
        )
        self.assertGreaterEqual(len(manual_calls), 1)
        self.assertEqual(hover_calls, [])
        self.assertEqual(stop_calls.count("stop"), _MODULE.HOVER_STOP_SETPOINT_REPEAT)
        self.assertIn("notify", stop_calls)

    def test_stateful_hover_primitive_aborts_when_takeoff_never_leaves_ground(self) -> None:
        hover_calls: list[float] = []

        class _FakeCommander:
            def send_hover_setpoint(self, _vx: float, _vy: float, _yawrate: float, zdistance: float) -> None:
                hover_calls.append(zdistance)

            def send_stop_setpoint(self) -> None:
                return None

            def send_notify_setpoint_stop(self) -> None:
                return None

        class _FakeCF:
            def __init__(self) -> None:
                self.commander = _FakeCommander()

        clock = _FakeClock()
        primitive_holder: dict[str, Any] = {"primitive": None}
        primitive = _MODULE.StatefulHoverPrimitive(
            _FakeCF(),
            ground_distance_provider=lambda: _MODULE.HoverGroundDistanceObservation(distance_m=0.03, age_s=0.02, is_flying=False, supervisor_age_s=0.02),
            stability_provider=_dynamic_stability_provider(primitive_holder, min_height_m=0.10),
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )
        primitive_holder["primitive"] = primitive

        outcome = primitive.run(
            _MODULE.HoverPrimitiveConfig(
                target_height_m=0.20,
                hover_duration_s=0.2,
                takeoff_velocity_mps=0.2,
                land_velocity_mps=0.2,
                takeoff_confirm_timeout_s=0.3,
                setpoint_period_s=0.1,
                stability=_MODULE.HoverStabilityConfig(required_stable_samples=1),
            )
        )

        self.assertTrue(outcome.aborted)
        self.assertFalse(outcome.took_off)
        self.assertTrue(outcome.landed)
        self.assertEqual(outcome.final_phase, "abort_landing")
        self.assertIn("takeoff confirmation failed", outcome.abort_reason or "")
        self.assertTrue(any(value > 0.0 for value in hover_calls))

    def test_stateful_hover_primitive_aborts_when_liftoff_never_reaches_target_band(self) -> None:
        hover_calls: list[float] = []

        class _FakeCommander:
            def send_hover_setpoint(self, _vx: float, _vy: float, _yawrate: float, zdistance: float) -> None:
                hover_calls.append(zdistance)

            def send_stop_setpoint(self) -> None:
                return None

            def send_notify_setpoint_stop(self) -> None:
                return None

        class _FakeCF:
            def __init__(self) -> None:
                self.commander = _FakeCommander()

        clock = _FakeClock()
        primitive_holder: dict[str, Any] = {"primitive": None}
        primitive = _MODULE.StatefulHoverPrimitive(
            _FakeCF(),
            ground_distance_provider=lambda: _MODULE.HoverGroundDistanceObservation(
                distance_m=0.09,
                age_s=0.02,
                is_flying=True,
                supervisor_age_s=0.02,
            ),
            stability_provider=_dynamic_stability_provider(primitive_holder, min_height_m=0.10),
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )
        primitive_holder["primitive"] = primitive

        outcome = primitive.run(
            _MODULE.HoverPrimitiveConfig(
                target_height_m=0.20,
                hover_duration_s=0.2,
                takeoff_velocity_mps=0.2,
                land_velocity_mps=0.2,
                takeoff_confirm_timeout_s=0.3,
                takeoff_confirm_target_height_tolerance_m=0.05,
                setpoint_period_s=0.1,
                stability=_MODULE.HoverStabilityConfig(required_stable_samples=1),
            )
        )

        self.assertTrue(outcome.aborted)
        self.assertFalse(outcome.took_off)
        self.assertTrue(outcome.landed)
        self.assertIn("required hover band", outcome.abort_reason or "")
        self.assertTrue(any(abs(value - 0.2) < 1e-6 for value in hover_calls))

    def test_stateful_hover_primitive_recovers_until_touchdown_has_range_and_supervisor(self) -> None:
        hover_calls: list[float] = []
        stop_calls: list[str] = []
        grounded_samples_seen = 0
        test_case = self
        touchdown_observations = iter(
            (
                _MODULE.HoverGroundDistanceObservation(distance_m=0.18, age_s=0.02, is_flying=True, supervisor_age_s=0.02),
                _MODULE.HoverGroundDistanceObservation(distance_m=0.17, age_s=0.02, is_flying=True, supervisor_age_s=0.02),
                _MODULE.HoverGroundDistanceObservation(distance_m=0.16, age_s=0.02, is_flying=True, supervisor_age_s=0.02),
                _MODULE.HoverGroundDistanceObservation(distance_m=0.045, age_s=0.02, is_flying=False, supervisor_age_s=0.02),
                _MODULE.HoverGroundDistanceObservation(distance_m=0.041, age_s=0.02, is_flying=False, supervisor_age_s=0.02),
                _MODULE.HoverGroundDistanceObservation(distance_m=0.039, age_s=0.02, is_flying=False, supervisor_age_s=0.02),
            )
        )

        def _observe_touchdown() -> _MODULE.HoverGroundDistanceObservation:
            nonlocal grounded_samples_seen
            observation = next(touchdown_observations)
            if observation.is_flying is False and observation.distance_m is not None and observation.distance_m <= 0.05:
                grounded_samples_seen += 1
            return observation

        class _FakeCommander:
            def send_hover_setpoint(self, _vx: float, _vy: float, _yawrate: float, zdistance: float) -> None:
                hover_calls.append(zdistance)

            def send_stop_setpoint(self) -> None:
                test_case.assertGreaterEqual(
                    grounded_samples_seen,
                    3,
                    "motor cut must not happen before combined touchdown confirmation is observed",
                )
                stop_calls.append("stop")

            def send_notify_setpoint_stop(self) -> None:
                stop_calls.append("notify")

        class _FakeCF:
            def __init__(self) -> None:
                self.commander = _FakeCommander()

        clock = _FakeClock()
        primitive_holder: dict[str, Any] = {"primitive": None}
        primitive = _MODULE.StatefulHoverPrimitive(
            _FakeCF(),
            ground_distance_provider=lambda: _observe_touchdown(),
            stability_provider=_dynamic_stability_provider(primitive_holder, min_height_m=0.10),
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )
        primitive_holder["primitive"] = primitive

        outcome = primitive.run(
            _MODULE.HoverPrimitiveConfig(
                target_height_m=0.2,
                hover_duration_s=0.2,
                takeoff_velocity_mps=0.2,
                land_velocity_mps=0.2,
                touchdown_confirm_timeout_s=0.2,
                setpoint_period_s=0.1,
                stability=_MODULE.HoverStabilityConfig(required_stable_samples=1),
            )
        )

        self.assertTrue(outcome.landed)
        self.assertTrue(primitive.landed)
        self.assertGreaterEqual(sum(1 for value in hover_calls if abs(value - 0.0) < 1e-6), 3)
        self.assertEqual(stop_calls.count("stop"), _MODULE.HOVER_STOP_SETPOINT_REPEAT)
        self.assertEqual(outcome.touchdown_confirmation_source, "range+supervisor")
        self.assertTrue(outcome.touchdown_supervisor_grounded)

    def test_stateful_hover_primitive_allows_explicit_sitl_range_only_touchdown(self) -> None:
        hover_calls: list[float] = []
        stop_calls: list[str] = []
        touchdown_observations = iter(
            (
                _MODULE.HoverGroundDistanceObservation(distance_m=0.12, age_s=0.02, is_flying=True, supervisor_age_s=0.02),
                _MODULE.HoverGroundDistanceObservation(distance_m=0.18, age_s=0.02, is_flying=True, supervisor_age_s=0.02),
                _MODULE.HoverGroundDistanceObservation(distance_m=0.21, age_s=0.02, is_flying=True, supervisor_age_s=0.02),
                _MODULE.HoverGroundDistanceObservation(distance_m=0.045, age_s=0.02, is_flying=True, supervisor_age_s=0.02),
                _MODULE.HoverGroundDistanceObservation(distance_m=0.041, age_s=0.02, is_flying=True, supervisor_age_s=0.02),
                _MODULE.HoverGroundDistanceObservation(distance_m=0.039, age_s=0.02, is_flying=True, supervisor_age_s=0.02),
            )
        )

        class _FakeCommander:
            def send_hover_setpoint(self, _vx: float, _vy: float, _yawrate: float, zdistance: float) -> None:
                hover_calls.append(zdistance)

            def send_stop_setpoint(self) -> None:
                stop_calls.append("stop")

            def send_notify_setpoint_stop(self) -> None:
                stop_calls.append("notify")

        class _FakeCF:
            def __init__(self) -> None:
                self.commander = _FakeCommander()

        clock = _FakeClock()
        primitive_holder: dict[str, Any] = {"primitive": None}
        primitive = _MODULE.StatefulHoverPrimitive(
            _FakeCF(),
            ground_distance_provider=lambda: next(touchdown_observations),
            stability_provider=_dynamic_stability_provider(primitive_holder, min_height_m=0.10),
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )
        primitive_holder["primitive"] = primitive

        outcome = primitive.run(
            _MODULE.HoverPrimitiveConfig(
                target_height_m=0.2,
                hover_duration_s=0.2,
                takeoff_velocity_mps=0.2,
                land_velocity_mps=0.2,
                touchdown_confirm_timeout_s=0.2,
                touchdown_require_supervisor_grounded=False,
                touchdown_range_only_confirmation_source="range_only_sitl",
                setpoint_period_s=0.1,
                stability=_MODULE.HoverStabilityConfig(required_stable_samples=1),
            )
        )

        self.assertTrue(outcome.landed)
        self.assertFalse(outcome.forced_motor_cutoff)
        self.assertEqual(outcome.touchdown_confirmation_source, "range_only_sitl")
        self.assertFalse(outcome.touchdown_supervisor_grounded)
        self.assertEqual(stop_calls.count("stop"), _MODULE.HOVER_STOP_SETPOINT_REPEAT)
        self.assertEqual(stop_calls.count("notify"), 1)
        self.assertGreaterEqual(sum(1 for value in hover_calls if abs(value - 0.0) < 1e-6), 3)

    def test_stateful_hover_primitive_honors_abort_request(self) -> None:
        class _FakeCommander:
            def send_hover_setpoint(self, _vx: float, _vy: float, _yawrate: float, _zdistance: float) -> None:
                return None

            def send_stop_setpoint(self) -> None:
                return None

            def send_notify_setpoint_stop(self) -> None:
                return None

        class _FakeCF:
            def __init__(self) -> None:
                self.commander = _FakeCommander()

        clock = _FakeClock()
        primitive_holder: dict[str, Any] = {"primitive": None}
        primitive: Any

        def _sleep(seconds: float) -> None:
            if primitive.took_off:
                primitive.request_abort("operator abort")
            clock.sleep(seconds)

        primitive = _MODULE.StatefulHoverPrimitive(
            _FakeCF(),
            ground_distance_provider=iter(
                (
                    _MODULE.HoverGroundDistanceObservation(distance_m=0.16, age_s=0.02, is_flying=True, supervisor_age_s=0.02),
                    _MODULE.HoverGroundDistanceObservation(distance_m=0.17, age_s=0.02, is_flying=True, supervisor_age_s=0.02),
                    _MODULE.HoverGroundDistanceObservation(distance_m=0.18, age_s=0.02, is_flying=True, supervisor_age_s=0.02),
                    _MODULE.HoverGroundDistanceObservation(distance_m=0.04, age_s=0.02, is_flying=False, supervisor_age_s=0.02),
                    _MODULE.HoverGroundDistanceObservation(distance_m=0.04, age_s=0.02, is_flying=False, supervisor_age_s=0.02),
                    _MODULE.HoverGroundDistanceObservation(distance_m=0.04, age_s=0.02, is_flying=False, supervisor_age_s=0.02),
                )
            ).__next__,
            stability_provider=_dynamic_stability_provider(primitive_holder, min_height_m=0.10),
            sleep=_sleep,
            monotonic=clock.monotonic,
        )
        primitive_holder["primitive"] = primitive
        outcome = primitive.run(
            _MODULE.HoverPrimitiveConfig(
                target_height_m=0.2,
                hover_duration_s=0.4,
                takeoff_velocity_mps=0.2,
                land_velocity_mps=0.2,
                setpoint_period_s=0.1,
                stability=_MODULE.HoverStabilityConfig(required_stable_samples=1),
            )
        )

        self.assertTrue(outcome.aborted)
        self.assertTrue(outcome.landed)
        self.assertEqual(outcome.abort_reason, "operator abort")
        self.assertEqual(outcome.final_phase, "abort_landing")

    def test_stateful_hover_primitive_aborts_when_live_stability_guard_trips(self) -> None:
        hover_calls: list[float] = []
        stop_calls: list[str] = []
        stability_observations = iter(
            (
                _MODULE.HoverStabilityObservation(
                    height_m=0.10,
                    height_age_s=0.02,
                    z_estimate_m=0.10,
                    z_estimate_age_s=0.02,
                    x_m=0.0,
                    y_m=0.0,
                    pose_age_s=0.02,
                    vx_mps=0.0,
                    vy_mps=0.0,
                    velocity_age_s=0.02,
                    roll_deg=0.5,
                    pitch_deg=0.3,
                    yaw_deg=0.0,
                    yaw_age_s=0.02,
                    attitude_age_s=0.02,
                    motion_squal=95,
                    motion_squal_age_s=0.02,
                    is_flying=True,
                    supervisor_age_s=0.02,
                ),
                _MODULE.HoverStabilityObservation(
                    height_m=0.20,
                    height_age_s=0.02,
                    z_estimate_m=0.20,
                    z_estimate_age_s=0.02,
                    x_m=0.0,
                    y_m=0.0,
                    pose_age_s=0.02,
                    vx_mps=0.0,
                    vy_mps=0.0,
                    velocity_age_s=0.02,
                    roll_deg=0.5,
                    pitch_deg=0.3,
                    yaw_deg=0.0,
                    yaw_age_s=0.02,
                    attitude_age_s=0.02,
                    motion_squal=95,
                    motion_squal_age_s=0.02,
                    is_flying=True,
                    supervisor_age_s=0.02,
                ),
                _MODULE.HoverStabilityObservation(
                    height_m=0.20,
                    height_age_s=0.02,
                    z_estimate_m=0.20,
                    z_estimate_age_s=0.02,
                    x_m=0.35,
                    y_m=0.0,
                    pose_age_s=0.02,
                    vx_mps=0.55,
                    vy_mps=0.0,
                    velocity_age_s=0.02,
                    roll_deg=12.0,
                    pitch_deg=0.3,
                    yaw_deg=0.0,
                    yaw_age_s=0.02,
                    attitude_age_s=0.02,
                    motion_squal=90,
                    motion_squal_age_s=0.02,
                    is_flying=True,
                    supervisor_age_s=0.02,
                ),
                _MODULE.HoverStabilityObservation(
                    height_m=0.20,
                    height_age_s=0.02,
                    z_estimate_m=0.20,
                    z_estimate_age_s=0.02,
                    x_m=0.37,
                    y_m=0.0,
                    pose_age_s=0.02,
                    vx_mps=0.57,
                    vy_mps=0.0,
                    velocity_age_s=0.02,
                    roll_deg=12.5,
                    pitch_deg=0.3,
                    yaw_deg=0.0,
                    yaw_age_s=0.02,
                    attitude_age_s=0.02,
                    motion_squal=88,
                    motion_squal_age_s=0.02,
                    is_flying=True,
                    supervisor_age_s=0.02,
                ),
            )
        )
        touchdown_observations = iter(
            (
                _MODULE.HoverGroundDistanceObservation(distance_m=0.16, age_s=0.02, is_flying=True, supervisor_age_s=0.02),
                _MODULE.HoverGroundDistanceObservation(distance_m=0.17, age_s=0.02, is_flying=True, supervisor_age_s=0.02),
                _MODULE.HoverGroundDistanceObservation(distance_m=0.18, age_s=0.02, is_flying=True, supervisor_age_s=0.02),
                _MODULE.HoverGroundDistanceObservation(distance_m=0.045, age_s=0.02, is_flying=False, supervisor_age_s=0.02),
                _MODULE.HoverGroundDistanceObservation(distance_m=0.041, age_s=0.02, is_flying=False, supervisor_age_s=0.02),
                _MODULE.HoverGroundDistanceObservation(distance_m=0.039, age_s=0.02, is_flying=False, supervisor_age_s=0.02),
            )
        )

        class _FakeCommander:
            def send_hover_setpoint(self, _vx: float, _vy: float, _yawrate: float, zdistance: float) -> None:
                hover_calls.append(zdistance)

            def send_stop_setpoint(self) -> None:
                stop_calls.append("stop")

            def send_notify_setpoint_stop(self) -> None:
                stop_calls.append("notify")

        class _FakeCF:
            def __init__(self) -> None:
                self.commander = _FakeCommander()

        clock = _FakeClock()
        primitive = _MODULE.StatefulHoverPrimitive(
            _FakeCF(),
            ground_distance_provider=lambda: next(touchdown_observations),
            stability_provider=lambda: next(stability_observations),
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )

        outcome = primitive.run(
            _MODULE.HoverPrimitiveConfig(
                target_height_m=0.20,
                hover_duration_s=0.4,
                takeoff_velocity_mps=0.2,
                land_velocity_mps=0.2,
                setpoint_period_s=0.1,
                stability=_MODULE.HoverStabilityConfig(
                    settle_timeout_s=0.3,
                    required_stable_samples=1,
                    abort_after_unstable_samples=2,
                    trim_observer=_MODULE.TrimObserverConfig(required_converged_samples=1),
                ),
            )
        )

        self.assertTrue(outcome.aborted)
        self.assertTrue(outcome.took_off)
        self.assertTrue(outcome.landed)
        self.assertIn("hover stability guard tripped", outcome.abort_reason or "")
        self.assertGreaterEqual(stop_calls.count("stop"), _MODULE.HOVER_STOP_SETPOINT_REPEAT)
        self.assertTrue(any(abs(value - 0.2) < 1e-6 for value in hover_calls))

    def test_hover_stability_guard_allows_missing_motion_squal_when_disabled(self) -> None:
        observation = _MODULE.HoverStabilityObservation(
            height_m=0.20,
            height_age_s=0.02,
            z_estimate_m=0.20,
            z_estimate_age_s=0.02,
            x_m=0.0,
            y_m=0.0,
            pose_age_s=0.02,
            vx_mps=0.0,
            vy_mps=0.0,
            velocity_age_s=0.02,
            roll_deg=0.5,
            pitch_deg=0.3,
            yaw_deg=0.0,
            yaw_age_s=0.02,
            attitude_age_s=0.02,
            motion_squal=None,
            motion_squal_age_s=None,
            is_flying=True,
            supervisor_age_s=0.02,
        )
        config = _MODULE.HoverStabilityConfig(require_motion_squal=False)
        command = _MODULE.FlowAnchorControlCommand(
            forward_mps=0.0,
            left_mps=0.0,
            height_m=0.20,
            trusted_height_m=0.20,
            trusted_height_source="zrange",
            sensor_disagreement_m=0.0,
            failures=(),
        )

        violations = _MODULE._hover_stability_violations(
            observation,
            target_height_m=0.20,
            anchor_xy=(0.0, 0.0),
            config=config,
            control_command=command,
        )

        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
