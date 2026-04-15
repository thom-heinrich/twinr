"""Regression coverage for the Crazyflie live start-envelope contract."""

from __future__ import annotations

import unittest

from twinr.hardware.crazyflie_start_contract import (
    StartEnvelopeConfig,
    evaluate_start_clearance_envelope,
)


class CrazyflieStartContractTests(unittest.TestCase):
    def test_start_clearance_blocks_tight_lateral_envelope(self) -> None:
        failures = evaluate_start_clearance_envelope(
            front_m=0.21,
            back_m=0.50,
            left_m=0.20,
            right_m=0.34,
            up_m=1.2,
            config=StartEnvelopeConfig(min_clearance_m=0.35),
        )

        self.assertEqual(
            failures,
            (
                "front clearance 0.21 m is below the 0.35 m hover gate",
                "left clearance 0.20 m is below the 0.35 m hover gate",
                "right clearance 0.34 m is below the 0.35 m hover gate",
            ),
        )

    def test_start_clearance_can_skip_laterals_in_sitl(self) -> None:
        failures = evaluate_start_clearance_envelope(
            front_m=0.10,
            back_m=0.10,
            left_m=0.10,
            right_m=0.10,
            up_m=1.2,
            config=StartEnvelopeConfig(
                min_clearance_m=0.35,
                require_lateral_clearance=False,
            ),
        )

        self.assertEqual(failures, ())
