from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.servo_continuous import (
    ContinuousRotationServoConfig,
    ContinuousRotationServoPlanner,
)


class ContinuousRotationServoPlannerTests(unittest.TestCase):
    def test_target_pulse_width_stays_center_inside_stop_tolerance(self) -> None:
        planner = ContinuousRotationServoPlanner(ContinuousRotationServoConfig(stop_tolerance_degrees=4.0))

        pulse_width_us = planner.target_pulse_width_for_heading(3.0, observed_at=10.0)

        self.assertEqual(pulse_width_us, 1500)

    def test_target_pulse_width_grows_gently_just_outside_stop_tolerance(self) -> None:
        planner = ContinuousRotationServoPlanner(
            ContinuousRotationServoConfig(
                stop_tolerance_degrees=4.0,
                min_speed_pulse_delta_us=70,
                max_speed_pulse_delta_us=160,
                slow_zone_degrees=45.0,
            )
        )

        pulse_width_us = planner.target_pulse_width_for_heading(10.0, observed_at=10.0)

        self.assertGreater(pulse_width_us, 1570)
        self.assertLess(pulse_width_us, 1600)

    def test_note_commanded_pulse_width_advances_virtual_heading(self) -> None:
        planner = ContinuousRotationServoPlanner(
            ContinuousRotationServoConfig(
                max_heading_degrees=90.0,
                max_speed_degrees_per_s=180.0,
                min_speed_pulse_delta_us=70,
                max_speed_pulse_delta_us=160,
            )
        )

        planner.note_commanded_pulse_width(1660, observed_at=10.0)
        planner.target_pulse_width_for_heading(90.0, observed_at=10.5)

        self.assertGreater(planner.estimated_heading_degrees, 30.0)
        self.assertLessEqual(planner.estimated_heading_degrees, 90.0)


if __name__ == "__main__":
    unittest.main()
