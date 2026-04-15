"""Regression coverage for Twinr's bounded local inspect navigation policy."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any
import unittest


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "hardware"
    / "bitcraze"
    / "local_navigation.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "bitcraze_local_navigation_module",
    _MODULE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE: Any = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class LocalNavigationPolicyTests(unittest.TestCase):
    def test_plan_prefers_largest_bounded_travel_budget(self) -> None:
        policy = _MODULE.LocalInspectNavigationPolicy(
            nominal_translation_m=0.25,
            min_translation_m=0.10,
            max_translation_m=0.30,
            required_post_move_clearance_m=0.20,
        )

        plan = _MODULE.plan_local_inspect_navigation(
            clearance_snapshot={
                "front": 0.44,
                "left": 0.62,
                "right": 0.39,
                "back": 0.31,
            },
            policy=policy,
        )

        self.assertEqual(plan.decision, "translate_then_capture")
        self.assertIsNotNone(plan.selected_translation)
        self.assertEqual(plan.selected_translation.direction, "left")
        self.assertAlmostEqual(plan.selected_translation.allowed_distance_m, 0.25, places=3)
        self.assertEqual(tuple(candidate.direction for candidate in plan.candidates), ("forward", "left", "right", "back"))

    def test_plan_falls_back_to_hover_anchor_when_no_safe_lane_exists(self) -> None:
        policy = _MODULE.LocalInspectNavigationPolicy(
            nominal_translation_m=0.20,
            min_translation_m=0.10,
            max_translation_m=0.30,
            required_post_move_clearance_m=0.16,
        )

        plan = _MODULE.plan_local_inspect_navigation(
            clearance_snapshot={
                "front": 0.20,
                "left": 0.23,
                "right": 0.19,
                "back": None,
                "up": "0.40",
            },
            policy=policy,
        )

        self.assertEqual(plan.decision, "hover_anchor_only")
        self.assertIsNone(plan.selected_translation)
        self.assertEqual(plan.candidates, ())
        self.assertIn("No lateral translation", plan.reason)
        self.assertAlmostEqual(plan.clearance_snapshot["up"] or 0.0, 0.40, places=3)

    def test_policy_normalization_rejects_invalid_translation_bounds(self) -> None:
        with self.assertRaisesRegex(ValueError, "nominal_translation_m must be >= min_translation_m"):
            _MODULE.LocalInspectNavigationPolicy(
                nominal_translation_m=0.08,
                min_translation_m=0.10,
            ).normalized()


if __name__ == "__main__":
    unittest.main()
