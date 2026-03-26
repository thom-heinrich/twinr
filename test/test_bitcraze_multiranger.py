"""Regression coverage for the Bitcraze Multi-ranger probe helper."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "hardware" / "bitcraze" / "probe_multiranger.py"
_SPEC = importlib.util.spec_from_file_location("bitcraze_multiranger_probe_script", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class BitcrazeMultirangerProbeTests(unittest.TestCase):
    def test_normalize_required_deck_name_accepts_common_aliases(self) -> None:
        self.assertEqual(_MODULE.normalize_required_deck_name("multiranger"), "bcMultiranger")
        self.assertEqual(_MODULE.normalize_required_deck_name("flow2"), "bcFlow2")
        self.assertEqual(_MODULE.normalize_required_deck_name("zranger"), "bcZRanger2")
        self.assertEqual(_MODULE.normalize_required_deck_name("aideck"), "bcAI")

    def test_normalize_required_deck_name_rejects_unknown_names(self) -> None:
        with self.assertRaises(ValueError):
            _MODULE.normalize_required_deck_name("unknown-deck")

    def test_summarize_samples_tracks_valid_and_missing_values(self) -> None:
        summary = _MODULE.summarize_samples([None, 0.42, 0.18, None])

        self.assertIsNone(summary.latest_m)
        self.assertEqual(summary.minimum_m, 0.18)
        self.assertEqual(summary.maximum_m, 0.42)
        self.assertEqual(summary.valid_samples, 2)
        self.assertEqual(summary.missing_samples, 2)

    def test_validate_report_fails_when_required_deck_is_missing(self) -> None:
        report = _MODULE.MultirangerProbeReport(
            uri="radio://0/80/2M",
            workspace="/twinr/bitcraze",
            duration_s=2.0,
            sample_period_s=0.1,
            sample_count=10,
            deck_flags={"bcMultiranger": 0, "bcFlow2": 1, "bcZRanger2": 1, "bcAI": 1},
            ranges_m={
                direction: _MODULE.RangeSummary(None, None, None, 0, 10)
                for direction in _MODULE.RANGE_DIRECTIONS
            },
            recommendations=(),
        )

        failures = _MODULE.validate_report(
            report,
            required_decks=("bcMultiranger",),
            require_readable_ranges=False,
        )

        self.assertEqual(failures, ["required deck bcMultiranger is not detected"])

    def test_recommendations_flag_missing_flow_support(self) -> None:
        report = _MODULE.MultirangerProbeReport(
            uri="radio://0/80/2M",
            workspace="/twinr/bitcraze",
            duration_s=2.0,
            sample_period_s=0.1,
            sample_count=10,
            deck_flags={"bcMultiranger": 1, "bcFlow2": 0, "bcZRanger2": 0, "bcAI": 1},
            ranges_m={
                "front": _MODULE.RangeSummary(0.4, 0.2, 0.6, 10, 0),
                "back": _MODULE.RangeSummary(0.5, 0.3, 0.5, 10, 0),
                "left": _MODULE.RangeSummary(0.6, 0.4, 0.6, 10, 0),
                "right": _MODULE.RangeSummary(0.7, 0.7, 0.7, 10, 0),
                "up": _MODULE.RangeSummary(1.0, 0.9, 1.1, 10, 0),
                "down": _MODULE.RangeSummary(None, None, None, 0, 10),
            },
            recommendations=(),
        )

        recommendations = _MODULE.recommendations_for_report(report)

        self.assertIn("Pair the Multi-ranger deck with the Flow deck for stable ground/down sensing.", recommendations)
        self.assertIn(
            "Downward z-range is unavailable; expect `down` to stay empty without a Z-ranger/Flow deck.",
            recommendations,
        )
