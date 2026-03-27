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
    WebArenaVerifiedTaskRun,
    build_agent_response_payload,
    build_minimal_trace_entries,
    build_twinr_request,
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

    def test_build_twinr_goal_handles_null_result_schema(self) -> None:
        goal = build_twinr_goal(
            intent="Get the order number of my most recent under delivery order",
            results_schema={"type": "null"},
        )
        self.assertIn('{"results": null}', goal)
        self.assertIn('{"results": [...]}', goal)

    def test_parse_json_answer_tolerates_sources_suffix(self) -> None:
        payload = parse_json_answer('{"results": [65]}\n\n## Sources\n- http://localhost:7770/')
        self.assertEqual(payload, {"results": [65]})

    def test_parse_json_answer_accepts_plain_json_scalar(self) -> None:
        payload = parse_json_answer("1")
        self.assertEqual(payload, 1)

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

    def test_build_agent_response_payload_prefers_typed_result(self) -> None:
        task = SimpleNamespace(expected_action="retrieve")
        browser_result = BrowserAutomationResult(
            ok=True,
            status="completed",
            summary="ok",
            data={
                "typed_result": {
                    "status": "success",
                    "results": [{"status": "Canceled", "arrival_date": None}],
                    "answer_text": None,
                    "reason": "supported",
                    "key_points": [],
                    "used_capabilities": [],
                },
                "answer_markdown": "not valid json",
            },
        )
        payload = build_agent_response_payload(task=task, browser_result=browser_result)
        self.assertEqual(payload["status"], "SUCCESS")
        self.assertEqual(payload["retrieved_data"], [{"status": "Canceled", "arrival_date": None}])

    def test_build_agent_response_payload_maps_typed_not_found(self) -> None:
        task = SimpleNamespace(expected_action="retrieve")
        browser_result = BrowserAutomationResult(
            ok=True,
            status="completed",
            summary="ok",
            data={
                "typed_result": {
                    "status": "not_found",
                    "results": None,
                    "answer_text": None,
                    "reason": "no supported match",
                    "key_points": [],
                    "used_capabilities": [],
                },
                "answer_markdown": "000000189",
            },
        )
        payload = build_agent_response_payload(task=task, browser_result=browser_result)
        self.assertEqual(payload["status"], "NOT_FOUND_ERROR")
        self.assertIsNone(payload["retrieved_data"])

    def test_build_agent_response_payload_uses_null_schema_success_shape(self) -> None:
        task = SimpleNamespace(
            expected_action="retrieve",
            eval=(SimpleNamespace(results_schema={"type": "null"}),),
        )
        browser_result = BrowserAutomationResult(
            ok=True,
            status="completed",
            summary="ok",
            data={"answer_markdown": '{"results": ["000000170"]}'},
        )
        payload = build_agent_response_payload(task=task, browser_result=browser_result)
        self.assertEqual(payload["status"], "SUCCESS")
        self.assertIsNone(payload["retrieved_data"])

    def test_build_agent_response_payload_maps_null_results_to_not_found(self) -> None:
        task = SimpleNamespace(expected_action="retrieve")
        browser_result = BrowserAutomationResult(
            ok=True,
            status="completed",
            summary="ok",
            data={"answer_markdown": '{"results": [null]}'},
        )
        payload = build_agent_response_payload(task=task, browser_result=browser_result)
        self.assertEqual(
            payload,
            {
                "task_type": "RETRIEVE",
                "status": "NOT_FOUND_ERROR",
                "retrieved_data": None,
                "error_details": None,
            },
        )

    def test_build_agent_response_payload_maps_explicit_null_results_to_not_found(self) -> None:
        task = SimpleNamespace(expected_action="retrieve")
        browser_result = BrowserAutomationResult(
            ok=True,
            status="completed",
            summary="ok",
            data={"answer_markdown": '{"results": null}'},
        )
        payload = build_agent_response_payload(task=task, browser_result=browser_result)
        self.assertEqual(payload["status"], "NOT_FOUND_ERROR")
        self.assertIsNone(payload["retrieved_data"])

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

    def test_build_agent_response_payload_maps_not_found_failure(self) -> None:
        task = SimpleNamespace(expected_action="retrieve")
        browser_result = BrowserAutomationResult(
            ok=False,
            status="failed",
            summary="not_found: no matching reviews were visible after inspecting the page",
            error_code="not_found",
        )
        payload = build_agent_response_payload(task=task, browser_result=browser_result)
        self.assertEqual(payload["status"], "NOT_FOUND_ERROR")
        self.assertIsNone(payload["retrieved_data"])
        self.assertIn("no matching reviews", payload["error_details"])

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

    def test_build_webarena_verified_config_preserves_credentials_and_header_login(self) -> None:
        config = build_webarena_verified_config(
            shopping_admin_url="http://localhost:7780",
            shopping_admin_credentials={"username": "admin"},
            shopping_admin_use_header_login=True,
        )
        env = next(iter(config.environments.values()))
        self.assertEqual(env.credentials, {"username": "admin"})
        self.assertTrue(env.use_header_login)

    def test_build_twinr_request_keeps_browser_context_metadata(self) -> None:
        request = build_twinr_request(
            task_run=WebArenaVerifiedTaskRun(
                task_id=11,
                site_name="shopping_admin",
                intent="Find the customer email.",
                start_url="http://localhost:7780/admin",
                goal="Find the customer email and return JSON only.",
                results_schema={"type": "array"},
                browser_context_storage_state_path="/tmp/shopping_admin_state.json",
                browser_context_extra_http_headers={"X-M2-Admin-Auto-Login": "admin"},
            ),
            max_steps=8,
            max_runtime_s=90.0,
        )
        self.assertEqual(
            request.metadata["browser_context_storage_state_path"],
            "/tmp/shopping_admin_state.json",
        )
        self.assertEqual(
            request.metadata["browser_context_extra_http_headers"],
            {"X-M2-Admin-Auto-Login": "admin"},
        )
        self.assertEqual(request.metadata["task_kind"], "auth_read")
        self.assertEqual(request.metadata["source_intent"], "Find the customer email.")
        self.assertEqual(request.metadata["results_schema"], {"type": "array"})

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
