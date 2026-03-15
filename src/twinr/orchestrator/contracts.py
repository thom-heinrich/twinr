from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from twinr.agent.base_agent.contracts import AgentToolCall, AgentToolResult


@dataclass(frozen=True, slots=True)
class OrchestratorTurnRequest:
    prompt: str
    conversation: tuple[tuple[str, str], ...] = ()
    supervisor_conversation: tuple[tuple[str, str], ...] = ()

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": "run_turn",
            "prompt": self.prompt,
            "conversation": [list(item) for item in self.conversation],
            "supervisor_conversation": [list(item) for item in self.supervisor_conversation],
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> OrchestratorTurnRequest:
        return cls(
            prompt=str(payload.get("prompt", "") or "").strip(),
            conversation=_coerce_conversation(payload.get("conversation")),
            supervisor_conversation=_coerce_conversation(payload.get("supervisor_conversation")),
        )


@dataclass(frozen=True, slots=True)
class OrchestratorToolRequest:
    call_id: str
    name: str
    arguments: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": "tool_request",
            "call_id": self.call_id,
            "name": self.name,
            "arguments": self.arguments,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> OrchestratorToolRequest:
        return cls(
            call_id=str(payload.get("call_id", "") or "").strip(),
            name=str(payload.get("name", "") or "").strip(),
            arguments=_coerce_dict(payload.get("arguments")),
        )


@dataclass(frozen=True, slots=True)
class OrchestratorToolResponse:
    call_id: str
    ok: bool
    output: dict[str, Any] | None = None
    error: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "type": "tool_result",
            "call_id": self.call_id,
            "ok": self.ok,
        }
        if self.output is not None:
            payload["output"] = _json_safe(self.output)
        if self.error is not None:
            payload["error"] = self.error
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> OrchestratorToolResponse:
        return cls(
            call_id=str(payload.get("call_id", "") or "").strip(),
            ok=bool(payload.get("ok")),
            output=_coerce_optional_dict(payload.get("output")),
            error=_coerce_optional_text(payload.get("error")),
        )


@dataclass(frozen=True, slots=True)
class OrchestratorAckEvent:
    ack_id: str
    text: str

    def to_payload(self) -> dict[str, Any]:
        return {"type": "ack", "ack_id": self.ack_id, "text": self.text}


@dataclass(frozen=True, slots=True)
class OrchestratorTextDeltaEvent:
    delta: str

    def to_payload(self) -> dict[str, Any]:
        return {"type": "text_delta", "delta": self.delta}


@dataclass(frozen=True, slots=True)
class OrchestratorErrorEvent:
    error: str

    def to_payload(self) -> dict[str, Any]:
        return {"type": "turn_error", "error": self.error}


@dataclass(frozen=True, slots=True)
class OrchestratorTurnCompleteEvent:
    text: str
    rounds: int
    used_web_search: bool
    response_id: str | None = None
    request_id: str | None = None
    model: str | None = None
    token_usage: dict[str, Any] | None = None
    tool_calls: tuple[AgentToolCall, ...] = ()
    tool_results: tuple[AgentToolResult, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": "turn_complete",
            "text": self.text,
            "rounds": self.rounds,
            "used_web_search": self.used_web_search,
            "response_id": self.response_id,
            "request_id": self.request_id,
            "model": self.model,
            "token_usage": _json_safe(self.token_usage),
            "tool_calls": [_json_safe(asdict(item)) for item in self.tool_calls],
            "tool_results": [_json_safe(asdict(item)) for item in self.tool_results],
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> OrchestratorTurnCompleteEvent:
        return cls(
            text=str(payload.get("text", "") or ""),
            rounds=int(payload.get("rounds", 0) or 0),
            used_web_search=bool(payload.get("used_web_search")),
            response_id=_coerce_optional_text(payload.get("response_id")),
            request_id=_coerce_optional_text(payload.get("request_id")),
            model=_coerce_optional_text(payload.get("model")),
            token_usage=_coerce_optional_dict(payload.get("token_usage")),
            tool_calls=tuple(_coerce_tool_calls(payload.get("tool_calls"))),
            tool_results=tuple(_coerce_tool_results(payload.get("tool_results"))),
        )


@dataclass(slots=True)
class OrchestratorClientTurnResult:
    text: str
    rounds: int
    used_web_search: bool
    response_id: str | None = None
    request_id: str | None = None
    model: str | None = None
    token_usage: dict[str, Any] | None = None
    tool_calls: tuple[AgentToolCall, ...] = ()
    tool_results: tuple[AgentToolResult, ...] = ()
    ack_events: list[OrchestratorAckEvent] = field(default_factory=list)


def _coerce_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _coerce_optional_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    return None


def _coerce_conversation(value: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list):
        return ()
    items: list[tuple[str, str]] = []
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            role = str(item[0]).strip()
            content = str(item[1]).strip()
            if role and content:
                items.append((role, content))
    return tuple(items)


def _coerce_tool_calls(value: Any) -> list[AgentToolCall]:
    if not isinstance(value, list):
        return []
    items: list[AgentToolCall] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        items.append(
            AgentToolCall(
                name=str(item.get("name", "") or "").strip(),
                call_id=str(item.get("call_id", "") or "").strip(),
                arguments=_coerce_dict(item.get("arguments")),
                raw_arguments=str(item.get("raw_arguments", "") or ""),
            )
        )
    return items


def _coerce_tool_results(value: Any) -> list[AgentToolResult]:
    if not isinstance(value, list):
        return []
    items: list[AgentToolResult] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        items.append(
            AgentToolResult(
                call_id=str(item.get("call_id", "") or "").strip(),
                name=str(item.get("name", "") or "").strip(),
                output=item.get("output"),
                serialized_output=str(item.get("serialized_output", "") or ""),
            )
        )
    return items


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if is_dataclass(value):
        return _json_safe(asdict(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_safe(model_dump())
    dict_method = getattr(value, "dict", None)
    if callable(dict_method):
        return _json_safe(dict_method())
    value_dict = getattr(value, "__dict__", None)
    if isinstance(value_dict, dict) and value_dict:
        return _json_safe(value_dict)
    return str(value)
