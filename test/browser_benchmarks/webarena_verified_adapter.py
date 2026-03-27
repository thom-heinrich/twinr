"""Bridge one WebArena Verified task to Twinr browser automation.

This adapter intentionally stays thin. It loads the official task definition,
renders the benchmark start URL from a supplied environment config, augments
the task goal with a small JSON output contract, runs Twinr's current browser
driver, and converts the driver result into the deterministic
``webarena-verified`` agent-response shape.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from collections.abc import Mapping, Sequence
import json

from twinr.agent.base_agent.config import TwinrConfig
from twinr.browser_automation import BrowserAutomationRequest, BrowserAutomationResult, load_browser_automation_driver
from twinr.browser_automation.runtime import normalize_typed_result, results_schema_expects_null

from webarena_verified.api import WebArenaVerified
from webarena_verified.types.config import EnvironmentConfig, WebArenaVerifiedConfig
from webarena_verified.types.task import WebArenaVerifiedTask


class WebArenaVerifiedAdapterError(RuntimeError):
    """Raise when a benchmark task cannot be mapped onto Twinr cleanly."""


@dataclass(frozen=True, slots=True)
class WebArenaVerifiedTaskRun:
    """Capture one benchmark task rendered into Twinr's browser contract."""

    task_id: int
    site_name: str
    intent: str
    start_url: str
    goal: str
    results_schema: dict[str, Any]
    browser_context_storage_state_path: str | None = None
    browser_context_extra_http_headers: dict[str, str] | None = None


def build_webarena_verified_config(
    *,
    shopping_url: str | None = None,
    shopping_admin_url: str | None = None,
    reddit_url: str | None = None,
    gitlab_url: str | None = None,
    wikipedia_url: str | None = None,
    map_url: str | None = None,
    shopping_credentials: Mapping[str, str] | None = None,
    shopping_admin_credentials: Mapping[str, str] | None = None,
    reddit_credentials: Mapping[str, str] | None = None,
    gitlab_credentials: Mapping[str, str] | None = None,
    shopping_use_header_login: bool = False,
    shopping_admin_use_header_login: bool = False,
    reddit_use_header_login: bool = False,
    gitlab_use_header_login: bool = False,
) -> WebArenaVerifiedConfig:
    """Build a small WebArena Verified config from explicit site URLs."""

    environment_map: dict[str, EnvironmentConfig] = {}
    for placeholder, url, credentials, use_header_login in (
        ("__SHOPPING__", shopping_url, shopping_credentials, shopping_use_header_login),
        ("__SHOPPING_ADMIN__", shopping_admin_url, shopping_admin_credentials, shopping_admin_use_header_login),
        ("__REDDIT__", reddit_url, reddit_credentials, reddit_use_header_login),
        ("__GITLAB__", gitlab_url, gitlab_credentials, gitlab_use_header_login),
        ("__WIKIPEDIA__", wikipedia_url, None, False),
        ("__MAP__", map_url, None, False),
    ):
        if url:
            environment_map[placeholder] = EnvironmentConfig(
                urls=[url],
                credentials=dict(credentials or {}) or None,
                use_header_login=bool(use_header_login),
            )
    return WebArenaVerifiedConfig(environments=environment_map or None)


def build_webarena_task_run(*, task: WebArenaVerifiedTask, config: WebArenaVerifiedConfig) -> WebArenaVerifiedTaskRun:
    """Render one official task into a concrete Twinr browser request payload."""

    if len(task.sites) != 1:
        raise WebArenaVerifiedAdapterError("Only single-site WebArena Verified tasks are supported in this spike.")
    if len(task.start_urls) != 1:
        raise WebArenaVerifiedAdapterError("Only single-start-url WebArena Verified tasks are supported in this spike.")
    site = task.sites[0]
    if config.environments is None:
        raise WebArenaVerifiedAdapterError("WebArena Verified config has no environment mappings.")
    environment = _lookup_environment(config=config, site=site)
    if environment is None:
        raise WebArenaVerifiedAdapterError(f"Missing environment URL for {site.url_name_template}.")
    start_url = environment.render_url(task.start_urls[0], site)
    first_eval = task.eval[0]
    results_schema = _to_jsonable(getattr(first_eval, "results_schema", {}) or {})
    goal = build_twinr_goal(intent=task.intent, results_schema=results_schema)
    return WebArenaVerifiedTaskRun(
        task_id=int(task.task_id),
        site_name=str(site.value),
        intent=str(task.intent),
        start_url=start_url,
        goal=goal,
        results_schema=results_schema,
    )


def build_twinr_goal(*, intent: str, results_schema: dict[str, Any]) -> str:
    """Append one explicit JSON contract without changing the task objective."""

    schema_text = json.dumps(results_schema or {"type": "array"}, ensure_ascii=False, sort_keys=True)
    if results_schema_expects_null(results_schema):
        return (
            f"{intent}\n\n"
            "Return your final answer as valid JSON only.\n"
            'If no matching result is supported by the visible evidence, return exactly {"results": null}.\n'
            'If one or more matching results are supported, return exactly {"results": [...]} with only the supported matches.\n'
            "Do not add prose, explanations, or markdown fences."
        )
    return (
        f"{intent}\n\n"
        "Return your final answer as valid JSON only, with this exact shape: "
        '{"results": [...]}.\n'
        'The items inside "results" must follow this JSON Schema: '
        f"{schema_text}\n"
        "Do not add prose, explanations, or markdown fences."
    )


def load_webarena_verified_task(
    *,
    task_id: int,
    config: WebArenaVerifiedConfig,
) -> tuple[WebArenaVerified, WebArenaVerifiedTask]:
    """Load one official task together with the evaluator facade."""

    benchmark = WebArenaVerified(config=config)
    return benchmark, benchmark.get_task(int(task_id))


def build_twinr_request(
    *,
    task_run: WebArenaVerifiedTaskRun,
    max_steps: int,
    max_runtime_s: float,
) -> BrowserAutomationRequest:
    """Convert one rendered benchmark task into a bounded Twinr request."""

    allowed_host = str(task_run.start_url.split("://", 1)[-1].split("/", 1)[0]).split(":", 1)[0]
    allowed_domains = tuple(domain for domain in (allowed_host, "127.0.0.1", "localhost") if domain)
    task_kind = "auth_read" if (
        task_run.browser_context_storage_state_path or task_run.browser_context_extra_http_headers
    ) else "read"
    return BrowserAutomationRequest(
        task_id=f"webarena_{task_run.task_id}",
        goal=task_run.goal,
        start_url=task_run.start_url,
        allowed_domains=allowed_domains,
        max_steps=int(max_steps),
        max_runtime_s=float(max_runtime_s),
        capture_screenshot=True,
        capture_html=False,
        metadata={
            "task_kind": task_kind,
            "eval_case_id": f"webarena_verified_{task_run.task_id}",
            "eval_suite": "webarena_verified_smoke",
            "webarena_task_id": int(task_run.task_id),
            "webarena_site": task_run.site_name,
            "source_intent": task_run.intent,
            "results_schema": dict(task_run.results_schema or {}),
            "browser_context_storage_state_path": task_run.browser_context_storage_state_path,
            "browser_context_extra_http_headers": dict(task_run.browser_context_extra_http_headers or {}),
        },
    )


def run_twinr_task(
    *,
    env_file: Path,
    task_run: WebArenaVerifiedTaskRun,
    project_root: Path,
    max_steps: int,
    max_runtime_s: float,
) -> BrowserAutomationResult:
    """Execute one rendered WebArena Verified task through Twinr's browser driver."""

    config = replace(TwinrConfig.from_env(env_file), browser_automation_enabled=True)
    driver = load_browser_automation_driver(config=config, project_root=project_root)
    request = build_twinr_request(
        task_run=task_run,
        max_steps=max_steps,
        max_runtime_s=max_runtime_s,
    )
    return driver.execute(request)


def parse_json_answer(answer_markdown: str) -> Any:
    """Parse the leading JSON payload from a browser answer string."""

    text = str(answer_markdown or "").strip()
    if not text:
        raise WebArenaVerifiedAdapterError("Twinr returned an empty answer payload.")
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip().startswith("```"):
            text = "\n".join(lines[1:-1]).strip()
    decoder = json.JSONDecoder()
    try:
        payload, end = decoder.raw_decode(text)
    except json.JSONDecodeError:
        payload = None
        end = -1
    else:
        if not text[end:].strip():
            return payload
    for index, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            payload, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return payload
    raise WebArenaVerifiedAdapterError("Twinr answer did not contain a parseable JSON payload.")


def build_agent_response_payload(
    *,
    task: WebArenaVerifiedTask,
    browser_result: BrowserAutomationResult,
) -> dict[str, Any]:
    """Translate a Twinr result into WebArena Verified's final agent-response shape."""

    task_type = str(task.expected_action).upper()
    task_results_schema = None
    if getattr(task, "eval", None):
        task_results_schema = getattr(task.eval[0], "results_schema", None)
    expects_null_results = results_schema_expects_null(task_results_schema)
    if not browser_result.ok:
        error_status = "UNKNOWN_ERROR"
        if str(browser_result.error_code or "").strip().lower() == "not_found":
            error_status = "NOT_FOUND_ERROR"
        return {
            "task_type": task_type,
            "status": error_status,
            "retrieved_data": None,
            "error_details": browser_result.summary,
        }

    typed_result = normalize_typed_result(browser_result.data.get("typed_result"))
    if typed_result is not None:
        if str(typed_result.get("status") or "").strip().lower() == "not_found":
            return {
                "task_type": task_type,
                "status": "NOT_FOUND_ERROR",
                "retrieved_data": None,
                "error_details": None,
            }
        typed_results = typed_result.get("results")
        if typed_results is None:
            answer_text = str(typed_result.get("answer_text") or "").strip()
            if answer_text:
                typed_results = [answer_text]
        elif not isinstance(typed_results, list):
            typed_results = [typed_results]
        if isinstance(typed_results, list):
            typed_results = list(typed_results)
            if not typed_results or all(item is None for item in typed_results):
                return {
                    "task_type": task_type,
                    "status": "NOT_FOUND_ERROR",
                    "retrieved_data": None,
                    "error_details": None,
                }
            if expects_null_results:
                return {
                    "task_type": task_type,
                    "status": "SUCCESS",
                    "retrieved_data": None,
                    "error_details": None,
                }
            return {
                "task_type": task_type,
                "status": "SUCCESS",
                "retrieved_data": typed_results,
                "error_details": None,
            }

    try:
        payload = parse_json_answer(str(browser_result.data.get("answer_markdown") or ""))
    except WebArenaVerifiedAdapterError as exc:
        return {
            "task_type": task_type,
            "status": "UNKNOWN_ERROR",
            "retrieved_data": None,
            "error_details": str(exc),
        }

    if isinstance(payload, dict) and "results" in payload:
        raw_retrieved_data = payload["results"]
    elif isinstance(payload, list):
        raw_retrieved_data = payload
    else:
        raw_retrieved_data = [payload]
    if raw_retrieved_data is None:
        return {
            "task_type": task_type,
            "status": "NOT_FOUND_ERROR",
            "retrieved_data": None,
            "error_details": None,
        }
    if expects_null_results:
        return {
            "task_type": task_type,
            "status": "SUCCESS",
            "retrieved_data": None,
            "error_details": None,
        }
    if not isinstance(raw_retrieved_data, list):
        raise WebArenaVerifiedAdapterError("Benchmark response contract must resolve to a results list.")
    retrieved_data = list(raw_retrieved_data)
    if not retrieved_data or all(item is None for item in retrieved_data):
        return {
            "task_type": task_type,
            "status": "NOT_FOUND_ERROR",
            "retrieved_data": None,
            "error_details": None,
        }
    return {
        "task_type": task_type,
        "status": "SUCCESS",
        "retrieved_data": retrieved_data,
        "error_details": None,
    }


def evaluate_browser_result(
    *,
    benchmark: WebArenaVerified,
    task: WebArenaVerifiedTask,
    browser_result: BrowserAutomationResult,
) -> dict[str, Any]:
    """Evaluate one Twinr browser result with the official WebArena Verified evaluator."""

    agent_response = build_agent_response_payload(task=task, browser_result=browser_result)
    evaluation = benchmark.evaluate_task(
        task_id=int(task.task_id),
        agent_response=agent_response,
        network_trace=build_minimal_trace_entries(browser_result=browser_result),
    )
    return {
        "agent_response": agent_response,
        "eval_result": evaluation.model_dump(mode="json"),
        "pass": bool(float(evaluation.score) > 0.0 and str(evaluation.status).upper().endswith("SUCCESS")),
    }


def _lookup_environment(*, config: WebArenaVerifiedConfig, site: Any) -> EnvironmentConfig | None:
    """Resolve a site mapping from either enum-backed tasks or lightweight test stubs."""

    if config.environments is None:
        return None
    for key in (site, getattr(site, "value", None), getattr(site, "url_name_template", None)):
        if key is None:
            continue
        try:
            environment = config.environments.get(key)
        except TypeError:
            environment = None
        if environment is not None:
            return environment
    return None


def _to_jsonable(value: Any) -> Any:
    """Convert mappingproxy-like schema objects into plain JSON-friendly values."""

    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_jsonable(item) for item in value]
    return value


def build_minimal_trace_entries(*, browser_result: BrowserAutomationResult) -> list[dict[str, Any]]:
    """Build a smallest-valid HAR-like trace from Twinr's visited URLs.

    WebArena Verified currently requires a non-empty trace even for pure
    ``AgentResponseEvaluator`` tasks. Twinr's browser workspace does not yet
    emit HAR, so the smoke adapter synthesizes one minimal navigation trace from
    the URLs the driver actually visited.
    """

    visited_urls = [str(url) for url in (browser_result.data.get("visited_urls") or []) if str(url)]
    final_url = str(browser_result.final_url or "")
    if final_url and final_url not in visited_urls:
        visited_urls.append(final_url)
    if not visited_urls:
        visited_urls.append("about:blank")

    entries: list[dict[str, Any]] = []
    for index, url in enumerate(visited_urls):
        headers = [
            {"name": "accept", "value": "text/html"},
            {"name": "sec-fetch-dest", "value": "document"},
            {"name": "sec-fetch-mode", "value": "navigate"},
            {"name": "sec-fetch-user", "value": "?1"},
        ]
        if index > 0:
            headers.append({"name": "referer", "value": visited_urls[index - 1]})
        entries.append(
            {
                "request": {
                    "method": "GET",
                    "url": url,
                    "headers": headers,
                    "cookies": [],
                    "queryString": [],
                    "headersSize": -1,
                    "bodySize": 0,
                },
                "response": {
                    "status": 200,
                    "headers": [{"name": "content-type", "value": "text/html; charset=utf-8"}],
                    "cookies": [],
                    "content": {"mimeType": "text/html", "size": 0, "text": ""},
                    "redirectURL": "",
                    "headersSize": -1,
                    "bodySize": 0,
                },
            }
        )
    return entries
