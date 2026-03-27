"""Regression coverage for tracked browser runtime typed-result helpers."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.browser_automation.runtime import (  # pylint: disable=wrong-import-position
    build_auth_state_summary,
    build_typed_result,
    normalize_typed_result,
    results_schema_expects_null,
)


class BrowserRuntimeTypedResultTests(unittest.TestCase):
    def test_results_schema_expects_null_detects_null_type(self) -> None:
        self.assertTrue(results_schema_expects_null({"type": "null"}))
        self.assertFalse(results_schema_expects_null({"type": "array"}))
        self.assertFalse(results_schema_expects_null(None))

    def test_build_typed_result_emits_json_safe_payload(self) -> None:
        payload = build_typed_result(
            status="success",
            results=[{"status": "Canceled", "arrival_date": None}],
            answer_text='{"results":[{"status":"Canceled","arrival_date":null}]}',
            reason="supported",
            key_points=("Latest order is canceled.",),
            used_capabilities=("auth_navigation", "dense_reader"),
        )
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["results"], [{"status": "Canceled", "arrival_date": None}])
        self.assertEqual(payload["used_capabilities"], ["auth_navigation", "dense_reader"])

    def test_normalize_typed_result_rejects_invalid_status(self) -> None:
        self.assertIsNone(
            normalize_typed_result(
                {
                    "status": "maybe",
                    "results": [],
                }
            )
        )

    def test_build_auth_state_summary_counts_navigation_surface(self) -> None:
        payload = build_auth_state_summary(
            current_url="http://localhost:7770/sales/order/history/",
            used_authenticated_context=True,
            visible_link_count=8,
            content_block_count=4,
            candidate_href_count=2,
            visited_url_count=3,
        )
        self.assertTrue(payload["used_authenticated_context"])
        self.assertEqual(payload["candidate_href_count"], 2)
        self.assertEqual(payload["visited_url_count"], 3)


if __name__ == "__main__":
    unittest.main()
