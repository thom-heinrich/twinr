"""Regression coverage for the bounded Crazyflie vertical bootstrap controller."""

from __future__ import annotations

from dataclasses import replace
from typing import Any
import unittest

from twinr.hardware.crazyflie_vertical_bootstrap import (
    VerticalBootstrapConfig,
    VerticalBootstrapObservation,
    initialize_vertical_bootstrap_state,
    step_vertical_bootstrap_controller,
)


def _config(**overrides: Any) -> VerticalBootstrapConfig:
    base = VerticalBootstrapConfig(
        target_height_m=0.08,
        min_thrust_percentage=36.0,
        feedforward_thrust_percentage=40.0,
        max_thrust_percentage=52.0,
        reference_duration_s=0.75,
        progress_to_ceiling_s=0.35,
        max_duration_s=1.0,
        height_gain_per_m=120.0,
        vertical_speed_gain_per_mps=45.0,
        min_range_height_m=0.08,
        max_range_height_m=0.13,
        min_range_rise_m=0.02,
        max_observation_age_s=0.35,
        max_ceiling_without_progress_s=0.35,
        required_liveness_samples=1,
        require_motion_squal_liveness=True,
        min_motion_squal=1,
        max_attitude_abs_deg=10.0,
    )
    return replace(base, **overrides)


class VerticalBootstrapControllerTests(unittest.TestCase):
    def test_controller_increases_thrust_when_lift_progress_lags_reference(self) -> None:
        state = initialize_vertical_bootstrap_state(baseline_distance_m=0.006)
        next_state, decision = step_vertical_bootstrap_controller(
            config=_config(),
            state=state,
            observation=VerticalBootstrapObservation(
                elapsed_s=0.10,
                distance_m=0.006,
                distance_age_s=0.02,
                vertical_speed_mps=0.0,
                vertical_speed_age_s=0.02,
                motion_squal=0,
                motion_squal_age_s=0.02,
                roll_deg=0.0,
                pitch_deg=0.0,
                attitude_age_s=0.02,
            ),
        )

        self.assertGreater(decision.commanded_thrust_percentage, 40.0)
        self.assertGreater(decision.raw_commanded_thrust_percentage, 40.0)
        self.assertGreater(decision.progress_boost_percentage, 0.0)
        self.assertGreater(decision.height_term_percentage, 0.0)
        self.assertGreater(decision.vertical_speed_term_percentage, 0.0)
        self.assertGreater(decision.thrust_headroom_percentage, 0.0)
        self.assertFalse(decision.at_thrust_ceiling)
        self.assertTrue(decision.progress_missing)
        self.assertFalse(decision.range_live)
        self.assertFalse(decision.range_height_ready)
        self.assertFalse(decision.range_rise_ready)
        self.assertFalse(decision.flow_live)
        self.assertEqual(
            decision.failure_codes,
            ("range_liveness_missing", "flow_liveness_missing"),
        )
        self.assertEqual(next_state.consecutive_liveness_samples, 0)
        self.assertGreater(next_state.progress_boost_percentage, 0.0)

    def test_controller_marks_handoff_ready_on_live_range_and_flow(self) -> None:
        state = initialize_vertical_bootstrap_state(baseline_distance_m=0.006)
        _state_after_first_tick, _first_decision = step_vertical_bootstrap_controller(
            config=_config(),
            state=state,
            observation=VerticalBootstrapObservation(
                elapsed_s=0.10,
                distance_m=0.030,
                distance_age_s=0.02,
                vertical_speed_mps=0.05,
                vertical_speed_age_s=0.02,
                motion_squal=50,
                motion_squal_age_s=0.02,
                roll_deg=0.2,
                pitch_deg=-0.1,
                attitude_age_s=0.02,
            ),
        )
        state = _state_after_first_tick
        _state_after_second_tick, decision = step_vertical_bootstrap_controller(
            config=_config(),
            state=state,
            observation=VerticalBootstrapObservation(
                elapsed_s=0.40,
                distance_m=0.090,
                distance_age_s=0.02,
                vertical_speed_mps=0.10,
                vertical_speed_age_s=0.02,
                motion_squal=80,
                motion_squal_age_s=0.02,
                roll_deg=0.2,
                pitch_deg=-0.1,
                attitude_age_s=0.02,
            ),
        )

        self.assertTrue(decision.range_live)
        self.assertTrue(decision.flow_live)
        self.assertTrue(decision.handoff_ready)
        self.assertIsNone(decision.abort_reason)
        self.assertEqual(decision.progress_boost_percentage, 0.0)

    def test_controller_aborts_when_ceiling_thrust_proves_no_lift(self) -> None:
        state = initialize_vertical_bootstrap_state(baseline_distance_m=0.006)
        state, _ = step_vertical_bootstrap_controller(
            config=_config(
                max_thrust_percentage=44.0,
                max_ceiling_without_progress_s=0.20,
                height_gain_per_m=160.0,
                vertical_speed_gain_per_mps=60.0,
            ),
            state=state,
            observation=VerticalBootstrapObservation(
                elapsed_s=0.10,
                distance_m=0.006,
                distance_age_s=0.02,
                vertical_speed_mps=0.0,
                vertical_speed_age_s=0.02,
                motion_squal=0,
                motion_squal_age_s=0.02,
                roll_deg=0.0,
                pitch_deg=0.0,
                attitude_age_s=0.02,
            ),
        )
        _next_state, decision = step_vertical_bootstrap_controller(
            config=_config(
                max_thrust_percentage=44.0,
                max_ceiling_without_progress_s=0.20,
                height_gain_per_m=160.0,
                vertical_speed_gain_per_mps=60.0,
            ),
            state=state,
            observation=VerticalBootstrapObservation(
                elapsed_s=0.40,
                distance_m=0.006,
                distance_age_s=0.02,
                vertical_speed_mps=0.0,
                vertical_speed_age_s=0.02,
                motion_squal=0,
                motion_squal_age_s=0.02,
                roll_deg=0.0,
                pitch_deg=0.0,
                attitude_age_s=0.02,
            ),
        )

        self.assertIn("bootstrap_ceiling_without_progress", decision.failure_codes)
        self.assertTrue(decision.at_thrust_ceiling)
        self.assertAlmostEqual(decision.thrust_headroom_percentage, 0.0)
        self.assertIn("saturated at bounded thrust without proving lift", decision.abort_reason or "")

    def test_controller_progress_drive_reaches_ceiling_before_timeout(self) -> None:
        state = initialize_vertical_bootstrap_state(baseline_distance_m=0.029)
        config = _config(progress_to_ceiling_s=0.35)
        observed_thrusts: list[float] = []

        for elapsed_s in (0.10, 0.20, 0.30, 0.40):
            state, decision = step_vertical_bootstrap_controller(
                config=config,
                state=state,
                observation=VerticalBootstrapObservation(
                    elapsed_s=elapsed_s,
                    distance_m=0.029,
                    distance_age_s=0.02,
                    vertical_speed_mps=0.0,
                    vertical_speed_age_s=0.02,
                    motion_squal=0,
                    motion_squal_age_s=0.02,
                    roll_deg=0.0,
                    pitch_deg=0.0,
                    attitude_age_s=0.02,
                ),
            )
            observed_thrusts.append(decision.commanded_thrust_percentage)

        self.assertGreaterEqual(observed_thrusts[-1], config.max_thrust_percentage)
        self.assertTrue(decision.at_thrust_ceiling)
        self.assertGreater(decision.progress_boost_percentage, 0.0)
        self.assertTrue(decision.progress_missing)

    def test_controller_aborts_on_overshoot(self) -> None:
        state = initialize_vertical_bootstrap_state(baseline_distance_m=0.006)
        _next_state, decision = step_vertical_bootstrap_controller(
            config=_config(),
            state=state,
            observation=VerticalBootstrapObservation(
                elapsed_s=0.30,
                distance_m=0.140,
                distance_age_s=0.02,
                vertical_speed_mps=0.15,
                vertical_speed_age_s=0.02,
                motion_squal=80,
                motion_squal_age_s=0.02,
                roll_deg=0.0,
                pitch_deg=0.0,
                attitude_age_s=0.02,
            ),
        )

        self.assertIn("bootstrap_range_overshoot", decision.failure_codes)
        self.assertIn("exceeded the bounded pre-hover height", decision.abort_reason or "")


if __name__ == "__main__":
    unittest.main()
