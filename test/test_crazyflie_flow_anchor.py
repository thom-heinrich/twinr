"""Regression coverage for bounded host-side Flow anchor hover control."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.crazyflie_flow_anchor import (
    FlowAnchorControlConfig,
    FlowAnchorObservation,
    HeightTrustConfig,
    compute_flow_anchor_command,
    compute_trusted_height,
)


class CrazyflieFlowAnchorTests(unittest.TestCase):
    def test_compute_trusted_height_rejects_large_sensor_disagreement(self) -> None:
        trusted_height = compute_trusted_height(
            raw_height_m=1.00,
            raw_height_age_s=0.02,
            estimate_z_m=0.24,
            estimate_z_age_s=0.02,
            config=HeightTrustConfig(max_sensor_disagreement_m=0.25, allow_estimate_fallback=False),
        )

        self.assertIsNone(trusted_height.height_m)
        self.assertEqual(trusted_height.source, "none")
        self.assertGreater(trusted_height.sensor_disagreement_m or 0.0, 0.70)
        self.assertTrue(trusted_height.failures)

    def test_compute_flow_anchor_command_generates_bounded_body_frame_correction(self) -> None:
        command = compute_flow_anchor_command(
            observation=FlowAnchorObservation(
                x_m=0.30,
                y_m=-0.10,
                pose_age_s=0.02,
                vx_mps=0.05,
                vy_mps=-0.02,
                velocity_age_s=0.02,
                yaw_deg=0.0,
                yaw_age_s=0.02,
                raw_height_m=0.23,
                raw_height_age_s=0.02,
                estimate_z_m=0.24,
                estimate_z_age_s=0.02,
            ),
            anchor_xy=(0.0, 0.0),
            target_height_m=0.25,
            control_config=FlowAnchorControlConfig(
                position_gain_p=1.0,
                velocity_gain_d=0.5,
                max_correction_velocity_mps=0.20,
                height_gain_p=1.0,
                max_height_correction_m=0.06,
            ),
            height_trust_config=HeightTrustConfig(),
        )

        self.assertLess(command.forward_mps, 0.0)
        self.assertGreater(command.left_mps, 0.0)
        self.assertAlmostEqual(command.trusted_height_m or 0.0, 0.23, places=3)
        self.assertGreater(command.height_m or 0.0, 0.23)
        self.assertLessEqual(abs(command.forward_mps), 0.20)
        self.assertLessEqual(abs(command.left_mps), 0.20)
        self.assertEqual(command.failures, ())


if __name__ == "__main__":
    unittest.main()
