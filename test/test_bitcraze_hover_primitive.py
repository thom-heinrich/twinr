"""Regression coverage for the bounded Crazyflie hover primitive helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any
import unittest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "hardware" / "bitcraze" / "hover_primitive.py"
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
                    "range.zrange": 18,
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
        self.assertAlmostEqual(report.zrange_min_m or 0.0, 0.018, places=3)
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

    def test_stateful_hover_primitive_runs_takeoff_hold_and_land(self) -> None:
        hover_calls: list[float] = []
        stop_calls: list[str] = []
        touchdown_observations = iter(
            (
                _MODULE.HoverGroundDistanceObservation(distance_m=0.12, age_s=0.02),
                _MODULE.HoverGroundDistanceObservation(distance_m=0.09, age_s=0.02),
                _MODULE.HoverGroundDistanceObservation(distance_m=0.049, age_s=0.02),
                _MODULE.HoverGroundDistanceObservation(distance_m=0.045, age_s=0.02),
                _MODULE.HoverGroundDistanceObservation(distance_m=0.041, age_s=0.02),
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
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )
        outcome = primitive.run(
            _MODULE.HoverPrimitiveConfig(
                target_height_m=0.2,
                hover_duration_s=0.3,
                takeoff_velocity_mps=0.2,
                land_velocity_mps=0.2,
                setpoint_period_s=0.1,
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

    def test_stateful_hover_primitive_recovers_until_supervisor_reports_not_flying(self) -> None:
        hover_calls: list[float] = []
        stop_calls: list[str] = []
        supervisor_false_samples_seen = 0
        test_case = self
        touchdown_observations = iter(
            (
                _MODULE.HoverGroundDistanceObservation(distance_m=0.18, age_s=0.02, is_flying=True, supervisor_age_s=0.02),
                _MODULE.HoverGroundDistanceObservation(distance_m=0.17, age_s=0.02, is_flying=True, supervisor_age_s=0.02),
                _MODULE.HoverGroundDistanceObservation(distance_m=0.16, age_s=0.02, is_flying=True, supervisor_age_s=0.02),
                _MODULE.HoverGroundDistanceObservation(distance_m=None, age_s=0.02, is_flying=False, supervisor_age_s=0.02),
                _MODULE.HoverGroundDistanceObservation(distance_m=None, age_s=0.02, is_flying=False, supervisor_age_s=0.02),
                _MODULE.HoverGroundDistanceObservation(distance_m=None, age_s=0.02, is_flying=False, supervisor_age_s=0.02),
            )
        )

        def _observe_touchdown() -> _MODULE.HoverGroundDistanceObservation:
            nonlocal supervisor_false_samples_seen
            observation = next(touchdown_observations)
            if observation.is_flying is False:
                supervisor_false_samples_seen += 1
            return observation

        class _FakeCommander:
            def send_hover_setpoint(self, _vx: float, _vy: float, _yawrate: float, zdistance: float) -> None:
                hover_calls.append(zdistance)

            def send_stop_setpoint(self) -> None:
                test_case.assertGreaterEqual(
                    supervisor_false_samples_seen,
                    3,
                    "motor cut must not happen before supervisor confirms not-flying",
                )
                stop_calls.append("stop")

            def send_notify_setpoint_stop(self) -> None:
                stop_calls.append("notify")

        class _FakeCF:
            def __init__(self) -> None:
                self.commander = _FakeCommander()

        clock = _FakeClock()
        primitive = _MODULE.StatefulHoverPrimitive(
            _FakeCF(),
            ground_distance_provider=lambda: _observe_touchdown(),
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )

        outcome = primitive.run(
            _MODULE.HoverPrimitiveConfig(
                target_height_m=0.2,
                hover_duration_s=0.2,
                takeoff_velocity_mps=0.2,
                land_velocity_mps=0.2,
                touchdown_confirm_timeout_s=0.2,
                setpoint_period_s=0.1,
            )
        )

        self.assertTrue(outcome.landed)
        self.assertTrue(primitive.landed)
        self.assertGreaterEqual(sum(1 for value in hover_calls if abs(value - 0.0) < 1e-6), 3)
        self.assertEqual(stop_calls.count("stop"), _MODULE.HOVER_STOP_SETPOINT_REPEAT)

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
        primitive = _MODULE.StatefulHoverPrimitive(
            _FakeCF(),
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )

        def _sleep(seconds: float) -> None:
            if primitive.took_off:
                primitive.request_abort("operator abort")
            clock.sleep(seconds)

        primitive = _MODULE.StatefulHoverPrimitive(
            _FakeCF(),
            ground_distance_provider=lambda: _MODULE.HoverGroundDistanceObservation(distance_m=0.04, age_s=0.02),
            sleep=_sleep,
            monotonic=clock.monotonic,
        )
        outcome = primitive.run(
            _MODULE.HoverPrimitiveConfig(
                target_height_m=0.2,
                hover_duration_s=0.4,
                takeoff_velocity_mps=0.2,
                land_velocity_mps=0.2,
                setpoint_period_s=0.1,
            )
        )

        self.assertTrue(outcome.aborted)
        self.assertTrue(outcome.landed)
        self.assertEqual(outcome.abort_reason, "operator abort")
        self.assertEqual(outcome.final_phase, "abort_landing")


if __name__ == "__main__":
    unittest.main()
