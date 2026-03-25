"""Persist generic realtime tool-call observability for operator surfaces.

This module records one normalized success or failure event per realtime tool
call. The executor owns the timing and generic metadata extraction so concrete
handlers can stay focused on business logic, payload normalization, and
durable side effects.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import time
from typing import Any

from ..handlers.support import SensitiveActionConfirmationRequired


_FAILED_STATUSES = {"error", "failed", "failure"}


def _normalize_optional_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _coerce_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return None


def _status_from_result(result: Mapping[str, object]) -> str:
    raw_status = _normalize_optional_text(result.get("status"))
    return raw_status or "ok"


def _result_failed(result: Mapping[str, object], status: str) -> bool:
    normalized = status.casefold()
    if normalized in _FAILED_STATUSES:
        return True
    return bool(_normalize_optional_text(result.get("error")))


def _record_event_safe(owner: Any, event: str, message: str, *, level: str = "info", **data: object) -> None:
    record_event = getattr(owner, "_record_event", None)
    if not callable(record_event):
        return
    try:
        record_event(event, message, level=level, **data)
    except Exception:
        return


@dataclass(frozen=True, slots=True)
class ToolCallObservation:
    """Capture one normalized runtime view of a tool call."""

    tool_name: str
    status: str
    latency_ms: int
    requested_model: str | None = None
    actual_model: str | None = None
    fallback_reason: str | None = None
    used_web_search: bool | None = None
    response_id: str | None = None
    request_id: str | None = None
    error_code: str | None = None
    error_type: str | None = None
    failure_origin: str | None = None

    @property
    def failed(self) -> bool:
        return self.failure_origin is not None

    def event_name(self) -> str:
        return "tool_call_failed" if self.failed else "tool_call_finished"

    def event_message(self) -> str:
        if self.failed:
            return "Realtime tool call failed."
        return "Realtime tool call finished."

    def event_level(self) -> str:
        return "error" if self.failed else "info"

    def to_event_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "tool_name": self.tool_name,
            "status": self.status,
            "latency_ms": self.latency_ms,
        }
        if self.requested_model:
            payload["requested_model"] = self.requested_model
        if self.actual_model:
            payload["actual_model"] = self.actual_model
        if self.fallback_reason:
            payload["fallback_reason"] = self.fallback_reason
        if self.used_web_search is not None:
            payload["used_web_search"] = self.used_web_search
        if self.response_id:
            payload["response_id"] = self.response_id
        if self.request_id:
            payload["request_id"] = self.request_id
        if self.error_code:
            payload["error_code"] = self.error_code
        if self.error_type:
            payload["error_type"] = self.error_type
        if self.failure_origin:
            payload["failure_origin"] = self.failure_origin
        return payload


def _observation_from_result(
    *,
    tool_name: str,
    result: object,
    latency_ms: int,
) -> ToolCallObservation:
    if not isinstance(result, Mapping):
        return ToolCallObservation(tool_name=tool_name, status="ok", latency_ms=latency_ms)

    status = _status_from_result(result)
    error_code = _normalize_optional_text(result.get("error"))
    failure_origin = "result_status" if _result_failed(result, status) else None
    return ToolCallObservation(
        tool_name=tool_name,
        status=status,
        latency_ms=latency_ms,
        requested_model=_normalize_optional_text(result.get("requested_model")),
        actual_model=_normalize_optional_text(result.get("actual_model") or result.get("model")),
        fallback_reason=_normalize_optional_text(result.get("fallback_reason")),
        used_web_search=_coerce_bool(result.get("used_web_search")),
        response_id=_normalize_optional_text(result.get("response_id")),
        request_id=_normalize_optional_text(result.get("request_id")),
        error_code=error_code,
        failure_origin=failure_origin,
    )


def observe_realtime_tool_call(
    owner: Any,
    *,
    tool_name: str,
    handler: Callable[[Any, dict[str, object]], dict[str, object]],
    arguments: dict[str, object],
) -> dict[str, object]:
    """Execute one tool handler and persist one normalized observability event."""

    started = time.monotonic()
    try:
        result = handler(owner, arguments)
    except SensitiveActionConfirmationRequired as exc:
        latency_ms = max(0, int(round((time.monotonic() - started) * 1000.0)))
        result = {
            "status": "confirmation_required",
            "detail": str(exc),
            "requires_confirmation": True,
        }
        observation = _observation_from_result(
            tool_name=tool_name,
            result=result,
            latency_ms=latency_ms,
        )
        _record_event_safe(
            owner,
            observation.event_name(),
            observation.event_message(),
            level=observation.event_level(),
            **observation.to_event_payload(),
        )
        return result
    except Exception as exc:
        latency_ms = max(0, int(round((time.monotonic() - started) * 1000.0)))
        observation = ToolCallObservation(
            tool_name=tool_name,
            status="exception",
            latency_ms=latency_ms,
            error_type=type(exc).__name__,
            failure_origin="exception",
        )
        _record_event_safe(
            owner,
            observation.event_name(),
            observation.event_message(),
            level=observation.event_level(),
            **observation.to_event_payload(),
        )
        raise

    latency_ms = max(0, int(round((time.monotonic() - started) * 1000.0)))
    observation = _observation_from_result(
        tool_name=tool_name,
        result=result,
        latency_ms=latency_ms,
    )
    _record_event_safe(
        owner,
        observation.event_name(),
        observation.event_message(),
        level=observation.event_level(),
        **observation.to_event_payload(),
    )
    return result
