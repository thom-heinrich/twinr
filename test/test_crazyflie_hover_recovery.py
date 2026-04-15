"""Regression coverage for replayed Crazyflie hover-recovery analysis."""

from __future__ import annotations

import unittest

from twinr.hardware.crazyflie_hover_recovery import (
    HoverGuardExpectation,
    HoverRecoveryExpectation,
    HoverRecoveryWindow,
    evaluate_hover_guard_contract,
    evaluate_hover_recovery,
)


class HoverRecoveryAnalysisTests(unittest.TestCase):
    def test_evaluate_hover_guard_contract_distinguishes_recover_from_abort(self) -> None:
        phase_events = (
            {
                "phase": "hover_primitive_stability_guard",
                "status": "degraded",
                "elapsed_s": 0.42,
                "data": {"phase": "stabilize", "failure_codes": ("xy_drift",)},
            },
            {
                "phase": "hover_primitive_stability_guard",
                "status": "blocked",
                "elapsed_s": 0.65,
                "data": {"phase": "stabilize", "failure_codes": ("speed", "xy_drift")},
            },
        )

        recover_metrics = evaluate_hover_guard_contract(
            phase_events=phase_events[:1],
            expectation=HoverGuardExpectation(must_block=False),
        )
        abort_metrics = evaluate_hover_guard_contract(
            phase_events=phase_events,
            expectation=HoverGuardExpectation(
                must_block=True,
                required_blocked_codes=("speed", "xy_drift"),
            ),
        )

        self.assertEqual(recover_metrics.failures, ())
        self.assertEqual(recover_metrics.blocked_count, 0)
        self.assertEqual(abort_metrics.failures, ())
        self.assertEqual(abort_metrics.blocked_codes, ("speed", "xy_drift"))
        self.assertEqual(abort_metrics.blocked_phases, ("stabilize",))

    def test_evaluate_hover_recovery_accepts_bounded_negative_forward_correction(self) -> None:
        command_log = (
            {"kind": "hover", "elapsed_s": 0.0, "vx_mps": 0.0, "vy_mps": 0.0, "height_m": 0.25},
            {"kind": "hover", "elapsed_s": 0.2, "vx_mps": 0.0, "vy_mps": 0.0, "height_m": 0.25},
            {"kind": "hover", "elapsed_s": 0.4, "vx_mps": -0.16, "vy_mps": 0.0, "height_m": 0.25},
            {"kind": "hover", "elapsed_s": 0.5, "vx_mps": -0.11, "vy_mps": 0.0, "height_m": 0.25},
            {"kind": "hover", "elapsed_s": 0.7, "vx_mps": 0.01, "vy_mps": 0.0, "height_m": 0.25},
            {"kind": "hover", "elapsed_s": 0.8, "vx_mps": 0.0, "vy_mps": 0.0, "height_m": 0.25},
            {"kind": "hover", "elapsed_s": 0.9, "vx_mps": 0.0, "vy_mps": 0.0, "height_m": 0.25},
        )
        phase_events = (
            {"phase": "hover_primitive_takeoff", "status": "done", "elapsed_s": 0.2},
            {"phase": "hover_primitive_hold", "status": "done", "elapsed_s": 0.9},
        )

        metrics = evaluate_hover_recovery(
            command_log=command_log,
            phase_events=phase_events,
            target_height_m=0.25,
            expectation=HoverRecoveryExpectation(
                disturbance_start_progress=0.25,
                disturbance_end_progress=0.50,
                max_recovery_delay_s=0.50,
                forward_direction="negative",
                min_forward_abs_mps=0.10,
            ),
        )

        self.assertEqual(metrics.failures, ())
        self.assertTrue(metrics.recovered_within_window)
        self.assertIsNotNone(metrics.recovery_delay_s)
        assert metrics.recovery_delay_s is not None
        self.assertLessEqual(metrics.recovery_delay_s, 0.50)

    def test_evaluate_hover_recovery_accepts_explicit_runtime_window_and_late_peak(self) -> None:
        command_log = (
            {"kind": "hover", "elapsed_s": 2.5, "vx_mps": 0.0, "vy_mps": 0.0, "height_m": 0.25},
            {"kind": "hover", "elapsed_s": 2.8, "vx_mps": 0.0, "vy_mps": 0.0, "height_m": 0.25},
            {"kind": "hover", "elapsed_s": 2.9, "vx_mps": 0.0, "vy_mps": 0.0, "height_m": 0.25},
            {"kind": "hover", "elapsed_s": 3.0, "vx_mps": 0.0, "vy_mps": 0.01, "height_m": 0.25},
            {"kind": "hover", "elapsed_s": 3.1, "vx_mps": 0.0, "vy_mps": 0.08, "height_m": 0.25},
            {"kind": "hover", "elapsed_s": 3.2, "vx_mps": 0.0, "vy_mps": 0.10, "height_m": 0.25},
            {"kind": "hover", "elapsed_s": 3.3, "vx_mps": 0.0, "vy_mps": 0.01, "height_m": 0.25},
            {"kind": "hover", "elapsed_s": 3.4, "vx_mps": 0.0, "vy_mps": 0.0, "height_m": 0.25},
        )
        phase_events = (
            {"phase": "hover_primitive_takeoff", "status": "done", "elapsed_s": 2.4},
            {"phase": "hover_primitive_hold", "status": "done", "elapsed_s": 3.6},
        )

        metrics = evaluate_hover_recovery(
            command_log=command_log,
            phase_events=phase_events,
            target_height_m=0.25,
            expectation=HoverRecoveryExpectation(
                disturbance_start_progress=0.15,
                disturbance_end_progress=0.45,
                max_recovery_delay_s=0.50,
                left_direction="positive",
                min_left_abs_mps=0.02,
                settle_required_commands=1,
            ),
            disturbance_window=HoverRecoveryWindow(
                disturbance_start_elapsed_s=2.8,
                disturbance_end_elapsed_s=3.15,
                source="physical_runtime",
            ),
        )

        self.assertEqual(metrics.failures, ())
        self.assertEqual(metrics.window_source, "physical_runtime")
        self.assertAlmostEqual(metrics.response_end_elapsed_s, 3.2, places=6)
        self.assertAlmostEqual(metrics.max_left_mps, 0.10, places=6)

    def test_evaluate_hover_recovery_rejects_explicit_window_before_takeoff(self) -> None:
        command_log = (
            {"kind": "hover", "elapsed_s": 0.1, "vx_mps": 0.0, "vy_mps": 0.0, "height_m": 0.25},
            {"kind": "hover", "elapsed_s": 0.2, "vx_mps": 0.0, "vy_mps": 0.0, "height_m": 0.25},
            {"kind": "hover", "elapsed_s": 0.3, "vx_mps": 0.0, "vy_mps": 0.0, "height_m": 0.25},
        )
        phase_events = (
            {"phase": "hover_primitive_takeoff", "status": "done", "elapsed_s": 0.2},
            {"phase": "hover_primitive_hold", "status": "done", "elapsed_s": 0.4},
        )

        with self.assertRaises(ValueError) as exc_info:
            evaluate_hover_recovery(
                command_log=command_log,
                phase_events=phase_events,
                target_height_m=0.25,
                expectation=HoverRecoveryExpectation(
                    disturbance_start_progress=0.25,
                    disturbance_end_progress=0.50,
                    max_recovery_delay_s=0.50,
                ),
                disturbance_window=HoverRecoveryWindow(
                    disturbance_start_elapsed_s=0.1,
                    disturbance_end_elapsed_s=0.25,
                    source="physical_runtime",
                ),
            )

        self.assertIn("begins before takeoff confirmation", str(exc_info.exception))

    def test_evaluate_hover_recovery_fails_when_settle_window_never_returns(self) -> None:
        command_log = (
            {"kind": "hover", "elapsed_s": 0.0, "vx_mps": 0.0, "vy_mps": 0.0, "height_m": 0.25},
            {"kind": "hover", "elapsed_s": 0.2, "vx_mps": 0.0, "vy_mps": 0.0, "height_m": 0.25},
            {"kind": "hover", "elapsed_s": 0.4, "vx_mps": -0.16, "vy_mps": 0.0, "height_m": 0.25},
            {"kind": "hover", "elapsed_s": 0.5, "vx_mps": -0.12, "vy_mps": 0.0, "height_m": 0.25},
            {"kind": "hover", "elapsed_s": 0.7, "vx_mps": -0.08, "vy_mps": 0.0, "height_m": 0.25},
            {"kind": "hover", "elapsed_s": 0.8, "vx_mps": -0.07, "vy_mps": 0.0, "height_m": 0.25},
            {"kind": "hover", "elapsed_s": 0.9, "vx_mps": -0.06, "vy_mps": 0.0, "height_m": 0.25},
        )
        phase_events = (
            {"phase": "hover_primitive_takeoff", "status": "done", "elapsed_s": 0.2},
            {"phase": "hover_primitive_hold", "status": "done", "elapsed_s": 0.9},
        )

        metrics = evaluate_hover_recovery(
            command_log=command_log,
            phase_events=phase_events,
            target_height_m=0.25,
            expectation=HoverRecoveryExpectation(
                disturbance_start_progress=0.25,
                disturbance_end_progress=0.50,
                max_recovery_delay_s=0.30,
                forward_direction="negative",
                min_forward_abs_mps=0.10,
            ),
        )

        self.assertFalse(metrics.recovered_within_window)
        self.assertIn(
            "recovery command never settled back into bounded hover output",
            metrics.failures,
        )

    def test_evaluate_hover_recovery_accepts_height_recovery_with_bounded_residual_trim(self) -> None:
        command_log = (
            {"kind": "hover", "elapsed_s": 0.0, "vx_mps": 0.0, "vy_mps": 0.0, "height_m": 0.25},
            {"kind": "hover", "elapsed_s": 0.2, "vx_mps": 0.0, "vy_mps": 0.0, "height_m": 0.25},
            {"kind": "hover", "elapsed_s": 0.4, "vx_mps": 0.0, "vy_mps": 0.0, "height_m": 0.31},
            {"kind": "hover", "elapsed_s": 0.5, "vx_mps": 0.0, "vy_mps": 0.0, "height_m": 0.314},
            {"kind": "hover", "elapsed_s": 0.6, "vx_mps": 0.0, "vy_mps": 0.0, "height_m": 0.315},
            {"kind": "hover", "elapsed_s": 0.7, "vx_mps": 0.0, "vy_mps": 0.0, "height_m": 0.315},
            {"kind": "hover", "elapsed_s": 0.8, "vx_mps": 0.0, "vy_mps": 0.0, "height_m": 0.315},
            {"kind": "hover", "elapsed_s": 0.9, "vx_mps": 0.0, "vy_mps": 0.0, "height_m": 0.315},
        )
        phase_events = (
            {"phase": "hover_primitive_takeoff", "status": "done", "elapsed_s": 0.2},
            {"phase": "hover_primitive_hold", "status": "done", "elapsed_s": 0.9},
        )

        metrics = evaluate_hover_recovery(
            command_log=command_log,
            phase_events=phase_events,
            target_height_m=0.25,
            expectation=HoverRecoveryExpectation(
                disturbance_start_progress=0.25,
                disturbance_end_progress=0.50,
                max_recovery_delay_s=0.30,
                settle_required_commands=1,
                height_direction="positive",
                min_height_delta_m=0.03,
            ),
        )

        self.assertEqual(metrics.failures, ())
        self.assertTrue(metrics.recovered_within_window)

    def test_evaluate_hover_recovery_accepts_height_recovery_when_first_plateau_sample_is_last_hover_sample(
        self,
    ) -> None:
        command_log = (
            {"kind": "hover", "elapsed_s": 2.3, "vx_mps": 0.0, "vy_mps": 0.0, "height_m": 0.25},
            {"kind": "hover", "elapsed_s": 2.5, "vx_mps": 0.0, "vy_mps": 0.0, "height_m": 0.25},
            {"kind": "hover", "elapsed_s": 2.6, "vx_mps": 0.0, "vy_mps": 0.0, "height_m": 0.2968},
            {"kind": "hover", "elapsed_s": 2.7, "vx_mps": 0.0, "vy_mps": 0.0, "height_m": 0.3017},
            {"kind": "hover", "elapsed_s": 2.8, "vx_mps": 0.0, "vy_mps": 0.0, "height_m": 0.3000},
            {"kind": "hover", "elapsed_s": 2.9, "vx_mps": 0.0, "vy_mps": 0.0, "height_m": 0.3012},
            {"kind": "hover", "elapsed_s": 3.0, "vx_mps": 0.0, "vy_mps": 0.0, "height_m": 0.3032},
            {"kind": "hover", "elapsed_s": 3.1, "vx_mps": 0.0, "vy_mps": 0.0, "height_m": 0.3070},
            {"kind": "hover", "elapsed_s": 3.2, "vx_mps": 0.0, "vy_mps": 0.0, "height_m": 0.3125},
            {"kind": "hover", "elapsed_s": 3.3, "vx_mps": 0.0, "vy_mps": 0.0, "height_m": 0.3228},
            {"kind": "hover", "elapsed_s": 3.4, "vx_mps": 0.0, "vy_mps": 0.0, "height_m": 0.3258},
            {"kind": "hover", "elapsed_s": 3.5, "vx_mps": 0.0, "vy_mps": 0.0, "height_m": 0.2311},
        )
        phase_events = (
            {"phase": "hover_primitive_takeoff", "status": "done", "elapsed_s": 2.3},
            {"phase": "hover_primitive_hold", "status": "done", "elapsed_s": 3.5},
            {"phase": "hover_primitive_land", "status": "begin", "elapsed_s": 3.5},
        )

        metrics = evaluate_hover_recovery(
            command_log=command_log,
            phase_events=phase_events,
            target_height_m=0.25,
            expectation=HoverRecoveryExpectation(
                disturbance_start_progress=0.25,
                disturbance_end_progress=0.40,
                max_recovery_delay_s=0.80,
                settle_required_commands=1,
                height_direction="positive",
                min_height_delta_m=0.01,
            ),
        )

        self.assertEqual(metrics.failures, ())
        self.assertTrue(metrics.recovered_within_window)
        self.assertAlmostEqual(metrics.response_end_elapsed_s, 3.3, places=6)
        self.assertAlmostEqual(metrics.settle_elapsed_s or 0.0, 3.4, places=6)


if __name__ == "__main__":
    unittest.main()
