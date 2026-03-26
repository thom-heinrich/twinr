"""Handle optional live browser-automation tool calls for Twinr.

This module keeps the realtime-tool boundary small: it validates the tool
payload, loads the optional local browser driver through Twinr's versioned
browser-automation API, and shapes the result back into a JSON-safe dict for
the model. Concrete browser stacks stay outside git in the repo-root
``browser_automation/`` workspace.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from twinr.browser_automation import (
    BrowserAutomationLoadError,
    BrowserAutomationRequest,
    BrowserAutomationResult,
    load_browser_automation_driver,
)
from twinr.agent.tools.runtime.browser_follow_up import repair_browser_request_from_pending_hint

from .handler_telemetry import emit_best_effort, record_event_best_effort, record_usage_best_effort
from .support import ArgumentValidationError, optional_bool, optional_float

_DEFAULT_MAX_STEPS = 6
_DEFAULT_MAX_RUNTIME_S = 45.0
_PENDING_SEARCH_FOLLOW_UP_SCOPE_SUFFIX = (
    "Prüfe nur, was auf der offiziell geprüften Website aktuell sichtbar ist. "
    "Wenn dort kein aktueller Nachweis sichtbar ist, sage das klar und bleibe bei dieser sichtbaren Website-Evidence."
)


def _ensure_arguments_mapping(arguments: dict[str, object]) -> dict[str, object]:
    """Return a defensive copy of the tool payload or fail clearly."""

    if not isinstance(arguments, Mapping):
        raise ArgumentValidationError("arguments must be a JSON object")
    return dict(arguments)


def _optional_text(arguments: Mapping[str, object], key: str) -> str | None:
    """Normalize one optional text field from the browser payload."""

    value = arguments.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_text(arguments: Mapping[str, object], key: str) -> str:
    """Return one required text field or raise a validation error."""

    text = _optional_text(arguments, key)
    if not text:
        raise ArgumentValidationError(f"{key} is required")
    return text


def _optional_int(arguments: Mapping[str, object], key: str, *, default: int) -> int:
    """Parse one optional positive integer from the browser payload."""

    if key not in arguments:
        return default
    value = arguments.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        raise ArgumentValidationError(f"{key} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        try:
            return int(stripped)
        except ValueError as exc:
            raise ArgumentValidationError(f"{key} must be an integer") from exc
    raise ArgumentValidationError(f"{key} must be an integer")


def _parse_allowed_domains(arguments: Mapping[str, object]) -> tuple[str, ...]:
    """Return the required domain allowlist for one browser run."""

    raw_value = arguments.get("allowed_domains")
    if raw_value is None:
        raise ArgumentValidationError("allowed_domains is required")
    if not isinstance(raw_value, (list, tuple)):
        raise ArgumentValidationError("allowed_domains must be a list of host names")

    normalized: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_value):
        domain = str(item or "").strip().lower()
        if not domain:
            raise ArgumentValidationError(f"allowed_domains[{index}] must not be empty")
        if any(char.isspace() for char in domain) or "/" in domain:
            raise ArgumentValidationError(f"allowed_domains[{index}] must be a host name")
        if domain in seen:
            continue
        normalized.append(domain)
        seen.add(domain)

    if not normalized:
        raise ArgumentValidationError("allowed_domains must contain at least one host name")
    return tuple(normalized)


def _project_root(owner: Any) -> Path:
    """Resolve the configured Twinr project root from the handler owner."""

    config = getattr(owner, "config", None)
    if config is None:
        raise RuntimeError("browser_automation requires owner.config")
    return Path(str(getattr(config, "project_root", ".") or ".")).expanduser().resolve()


def _repair_arguments_from_pending_hint(
    owner: Any,
    arguments: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object] | None]:
    """Apply a pending search follow-up hint to one browser payload when needed."""

    raw_allowed_domains = arguments.get("allowed_domains")
    allowed_domains: tuple[str, ...]
    if isinstance(raw_allowed_domains, (list, tuple)):
        allowed_domains = tuple(str(item or "").strip() for item in raw_allowed_domains)
    else:
        allowed_domains = ()
    repair = repair_browser_request_from_pending_hint(
        getattr(owner, "runtime", None),
        start_url=_optional_text(arguments, "start_url"),
        allowed_domains=allowed_domains,
    )
    if repair is None:
        return dict(arguments), None
    repaired_arguments = dict(arguments)
    goal = _optional_text(arguments, "goal")
    normalized_goal = str(goal) if goal is not None else ""
    if normalized_goal and _PENDING_SEARCH_FOLLOW_UP_SCOPE_SUFFIX not in normalized_goal:
        repaired_arguments["goal"] = f"{normalized_goal} {_PENDING_SEARCH_FOLLOW_UP_SCOPE_SUFFIX}"
    if repair.effective_start_url:
        repaired_arguments["start_url"] = repair.effective_start_url
    if repair.effective_allowed_domains:
        repaired_arguments["allowed_domains"] = list(repair.effective_allowed_domains)
    return repaired_arguments, asdict(repair)


def _build_request(arguments: dict[str, object]) -> BrowserAutomationRequest:
    """Validate the tool payload and convert it into the stable browser request."""

    goal = _required_text(arguments, "goal")
    start_url = _optional_text(arguments, "start_url")
    allowed_domains = _parse_allowed_domains(arguments)
    max_steps = _optional_int(arguments, "max_steps", default=_DEFAULT_MAX_STEPS)
    max_runtime_s = optional_float(arguments, "max_runtime_s", default=_DEFAULT_MAX_RUNTIME_S)
    capture_screenshot = optional_bool(arguments, "capture_screenshot", default=True)
    capture_html = optional_bool(arguments, "capture_html", default=False)
    try:
        return BrowserAutomationRequest(
            task_id=f"browser-{uuid4().hex[:12]}",
            goal=goal,
            start_url=start_url,
            allowed_domains=allowed_domains,
            max_steps=max_steps,
            max_runtime_s=max_runtime_s,
            capture_screenshot=bool(capture_screenshot),
            capture_html=bool(capture_html),
        )
    except (TypeError, ValueError) as exc:
        raise ArgumentValidationError(str(exc)) from exc


def _result_payload(result: BrowserAutomationResult) -> dict[str, object]:
    """Serialize the stable browser result dataclass into a tool payload."""

    payload: dict[str, object] = {
        "status": result.status,
        "ok": result.ok,
        "summary": result.summary,
        "used_web_search": True,
        "artifacts": [asdict(artifact) for artifact in result.artifacts],
        "data": dict(result.data),
    }
    if result.final_url:
        payload["final_url"] = result.final_url
    if result.error_code:
        payload["error_code"] = result.error_code
        if result.status in {"failed", "cancelled"}:
            payload["error"] = result.error_code
    return payload


def _generic_failure_payload(*, error_code: str, summary: str) -> dict[str, object]:
    """Return one fail-closed browser tool result without raising out of the turn."""

    return {
        "status": "failed",
        "ok": False,
        "summary": summary,
        "error": error_code,
        "error_code": error_code,
        "used_web_search": True,
        "artifacts": [],
        "data": {},
    }


def handle_browser_automation(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """Run one bounded browser-automation task through the optional local driver.

    Args:
        owner: Tool executor owner exposing ``config`` and optional telemetry
            callbacks.
        arguments: Tool payload with required ``goal`` and ``allowed_domains``
            plus optional browser budget fields.

    Returns:
        JSON-safe browser automation result payload.
    """

    arguments = _ensure_arguments_mapping(arguments)
    arguments, request_repair = _repair_arguments_from_pending_hint(owner, arguments)
    request = _build_request(arguments)
    project_root = _project_root(owner)
    config = owner.config

    emit_best_effort(owner, "browser_automation_status=starting")
    if request_repair is not None:
        emit_best_effort(owner, "browser_automation_request_repaired=true")
        record_event_best_effort(
            owner,
            "browser_automation_request_repaired",
            "Browser automation reused the pending official-site follow-up hint from live search.",
            request_repair,
        )
    record_event_best_effort(
        owner,
        "browser_automation_started",
        "Browser automation tool was invoked.",
        {
            "goal": request.goal,
            "allowed_domains": list(request.allowed_domains),
            "start_url": request.start_url,
            "max_steps": request.max_steps,
            "max_runtime_s": request.max_runtime_s,
        },
    )

    try:
        driver = load_browser_automation_driver(
            config=config,
            project_root=project_root,
        )
    except BrowserAutomationLoadError:
        emit_best_effort(owner, "browser_automation_status=unavailable")
        record_event_best_effort(
            owner,
            "browser_automation_unavailable",
            "Browser automation was requested but no usable local driver was available.",
            {
                "goal": request.goal,
                "allowed_domains": list(request.allowed_domains),
                "start_url": request.start_url,
                "request_repair": request_repair,
            },
        )
        return _generic_failure_payload(
            error_code="unavailable",
            summary="Browser automation is currently unavailable.",
        )

    try:
        result = driver.execute(request)
    except Exception:
        emit_best_effort(owner, "browser_automation_status=failed")
        record_event_best_effort(
            owner,
            "browser_automation_failed",
            "Browser automation raised an execution error.",
            {
                "goal": request.goal,
                "allowed_domains": list(request.allowed_domains),
                "start_url": request.start_url,
                "request_repair": request_repair,
            },
        )
        return _generic_failure_payload(
            error_code="execution_error",
            summary="Browser automation failed before it could finish the task.",
        )

    payload = _result_payload(result)
    if request_repair is not None:
        payload_data = payload.get("data")
        if isinstance(payload_data, dict):
            payload_data["request_repair"] = request_repair
    emit_best_effort(owner, f"browser_automation_status={result.status}")
    record_event_best_effort(
        owner,
        "browser_automation_completed",
        "Browser automation finished.",
        {
            "status": result.status,
            "goal": request.goal,
            "final_url": result.final_url,
            "artifact_count": len(result.artifacts),
            "error_code": result.error_code,
            "request_repair": request_repair,
        },
    )
    record_usage_best_effort(
        owner,
        {
            "request_kind": "browser_automation",
            "source": "realtime_tool",
            "request_source": "tool",
            "used_web_search": True,
        },
    )
    return payload
