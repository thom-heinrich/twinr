"""Regression coverage for the Crazyflie hover trusted-state and trim observer."""

from __future__ import annotations

import unittest

from twinr.hardware.crazyflie_flow_anchor import HeightTrustConfig
from twinr.hardware.crazyflie_trim_observer import (
    TrimObserverConfig,
    TrimObserverState,
    update_trim_observer,
)
from twinr.hardware.crazyflie_trusted_state import (
    LateralTrustConfig,
    compute_trusted_hover_state,
)


class CrazyflieTrimObserverTests(unittest.TestCase):
    def test_trusted_hover_state_reports_low_flow_and_missing_pose(self) -> None:
        state = compute_trusted_hover_state(
            raw_height_m=0.10,
            raw_height_age_s=0.02,
            estimate_z_m=0.10,
            estimate_z_age_s=0.02,
            x_m=None,
            y_m=None,
            pose_age_s=None,
            vx_mps=0.0,
            vy_mps=0.0,
            velocity_age_s=0.02,
            motion_squal=5,
            motion_squal_age_s=0.02,
            is_flying=True,
            supervisor_age_s=0.02,
            height_config=HeightTrustConfig(),
            lateral_config=LateralTrustConfig(),
        )

        self.assertFalse(state.flow_confident)
        self.assertFalse(state.pose_available)
        self.assertIn("optical-flow quality 5 is below the 30 stability floor", state.failures)
        self.assertIn("xy pose telemetry is unavailable", state.failures)

    def test_trim_observer_converges_and_builds_counter_trim(self) -> None:
        config = TrimObserverConfig(required_converged_samples=3)
        state = TrimObserverState()
        trusted_state = compute_trusted_hover_state(
            raw_height_m=0.10,
            raw_height_age_s=0.02,
            estimate_z_m=0.10,
            estimate_z_age_s=0.02,
            x_m=0.0,
            y_m=0.0,
            pose_age_s=0.02,
            vx_mps=0.02,
            vy_mps=-0.01,
            velocity_age_s=0.02,
            motion_squal=90,
            motion_squal_age_s=0.02,
            is_flying=True,
            supervisor_age_s=0.02,
            height_config=HeightTrustConfig(),
            lateral_config=LateralTrustConfig(),
        )

        for _ in range(3):
            state = update_trim_observer(
                state,
                trusted_state=trusted_state,
                target_height_m=0.10,
                vx_mps=0.02,
                vy_mps=-0.01,
                velocity_age_s=0.02,
                roll_deg=0.5,
                pitch_deg=-0.4,
                attitude_age_s=0.02,
                config=config,
            )

        self.assertTrue(state.converged)
        self.assertLess(state.forward_trim_mps, 0.0)
        self.assertGreater(state.left_trim_mps, 0.0)
        self.assertEqual(state.failures, ())

    def test_trim_observer_refuses_to_converge_on_untrusted_state(self) -> None:
        state = update_trim_observer(
            TrimObserverState(),
            trusted_state=compute_trusted_hover_state(
                raw_height_m=None,
                raw_height_age_s=None,
                estimate_z_m=None,
                estimate_z_age_s=None,
                x_m=None,
                y_m=None,
                pose_age_s=None,
                vx_mps=None,
                vy_mps=None,
                velocity_age_s=None,
                motion_squal=None,
                motion_squal_age_s=None,
                is_flying=False,
                supervisor_age_s=None,
                height_config=HeightTrustConfig(),
                lateral_config=LateralTrustConfig(),
            ),
            target_height_m=0.10,
            vx_mps=None,
            vy_mps=None,
            velocity_age_s=None,
            roll_deg=None,
            pitch_deg=None,
            attitude_age_s=None,
            config=TrimObserverConfig(),
        )

        self.assertFalse(state.converged)
        self.assertGreater(len(state.failures), 0)
