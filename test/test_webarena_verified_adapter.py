# ruff: noqa: E402
"""Regression coverage for the tracked WebArena Verified adapter."""

from pathlib import Path
from types import SimpleNamespace
import json
import sys
import unittest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from twinr.browser_automation import BrowserAutomationResult

from test.browser_benchmarks.webarena_verified_adapter import (
    WebArenaVerifiedAdapterError,
    build_agent_response_payload,
    build_minimal_trace_entries,
    build_twinr_goal,
    build_webarena_task_run,
    build_webarena_verified_config,
    parse_json_answer,
)


class WebArenaVerifiedAdapterTests(unittest.TestCase):
    def test_build_twinr_goal_includes_results_contract(self) -> None:
        goal = build_twinr_goal(
            intent="What is the rating of Ugreen lightning to 3.5mm cable?",
            results_schema={"type": "array", "items": {"type": "number"}},
        )
        self.assertIn('{"results": [...]}', goal)
        self.assertIn('"type": "number"', goal)
        self.assertIn("Do not add prose", goal)

    def test_parse_json_answer_tolerates_sources_suffix(self) -> None:
        payload = parse_json_answer('{"results": [65]}\n\n## Sources\n- http://localhost:7770/')
        self.assertEqual(payload, {"results": [65]})

    def test_parse_json_answer_rejects_missing_json(self) -> None:
        with self.assertRaises(WebArenaVerifiedAdapterError):
            parse_json_answer("No structured answer here.")

    def test_build_agent_response_payload_uses_results_array(self) -> None:
        task = SimpleNamespace(expected_action="retrieve")
        browser_result = BrowserAutomationResult(
            ok=True,
            status="completed",
            summary="ok",
            data={"answer_markdown": '{"results": [65]}'},
        )
        payload = build_agent_response_payload(task=task, browser_result=browser_result)
        self.assertEqual(
            payload,
            {
                "task_type": "RETRIEVE",
                "status": "SUCCESS",
                "retrieved_data": [65],
                "error_details": None,
            },
        )

    def test_build_agent_response_payload_wraps_failure(self) -> None:
        task = SimpleNamespace(expected_action="retrieve")
        browser_result = BrowserAutomationResult(
            ok=False,
            status="failed",
            summary="insufficient_evidence",
        )
        payload = build_agent_response_payload(task=task, browser_result=browser_result)
        self.assertEqual(payload["status"], "UNKNOWN_ERROR")
        self.assertEqual(payload["error_details"], "insufficient_evidence")

    def test_build_webarena_task_run_renders_start_url(self) -> None:
        config = build_webarena_verified_config(shopping_url="http://localhost:7770")
        task = SimpleNamespace(
            task_id=386,
            sites=(SimpleNamespace(value="shopping", url_name_template="__SHOPPING__"),),
            start_urls=("__SHOPPING__",),
            intent="What is the rating of Ugreen lightning to 3.5mm cable?",
            eval=(SimpleNamespace(results_schema={"type": "array", "items": {"type": "number"}}),),
        )
        task_run = build_webarena_task_run(task=task, config=config)
        self.assertEqual(task_run.start_url, "http://localhost:7770")
        self.assertIn('"type": "number"', json.dumps(task_run.results_schema))

    def test_build_minimal_trace_entries_uses_visited_urls(self) -> None:
        browser_result = BrowserAutomationResult(
            ok=True,
            status="completed",
            summary="ok",
            final_url="http://localhost:7770/product",
            data={"visited_urls": ["http://localhost:7770/", "http://localhost:7770/product"]},
        )
        entries = build_minimal_trace_entries(browser_result=browser_result)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["request"]["url"], "http://localhost:7770/")
        self.assertEqual(entries[-1]["response"]["status"], 200)


if __name__ == "__main__":
    unittest.main()
