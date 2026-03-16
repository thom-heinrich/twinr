"""Define websocket transport contracts for the Twinr orchestrator.

This module owns the dataclasses that cross the edge-orchestrator websocket
boundary plus the defensive coercion helpers that keep malformed payloads from
crashing turn handling.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from twinr.agent.base_agent.contracts import AgentToolCall, AgentToolResult


_MAX_JSON_DEPTH = 32  # AUDIT-FIX(#1): Bound serializer recursion depth so malformed/cyclic payloads cannot crash turn finalization.
_RECURSION_SENTINEL = "<recursive-ref>"
_MAX_DEPTH_SENTINEL = "<max-depth-exceeded>"
_TRUE_STRINGS = frozenset({"1", "true", "t", "yes", "y", "on"})
_FALSE_STRINGS = frozenset({"0", "false", "f", "no", "n", "off", ""})


@dataclass(frozen=True, slots=True)
class OrchestratorTurnRequest:
    """Represent one remote request to run an orchestrated turn."""

    prompt: str
    conversation: tuple[tuple[str, str], ...] = ()
    supervisor_conversation: tuple[tuple[str, str], ...] = ()

    def to_payload(self) -> dict[str, Any]:
        """Serialize the turn request into a websocket payload."""

        return {
            "type": "run_turn",
            "prompt": self.prompt,
            "conversation": [list(item) for item in self.conversation],
            "supervisor_conversation": [list(item) for item in self.supervisor_conversation],
        }

    @classmethod
    def from_payload(cls, payload: Any) -> OrchestratorTurnRequest:
        """Build a turn request from an incoming websocket payload."""

        payload_dict = _coerce_dict(payload)  # AUDIT-FIX(#3): Tolerate malformed transport frames instead of assuming a dict and crashing.
        return cls(
            prompt=str(payload_dict.get("prompt", "") or "").strip(),
            conversation=_coerce_conversation(payload_dict.get("conversation")),
            supervisor_conversation=_coerce_conversation(payload_dict.get("supervisor_conversation")),
        )


@dataclass(frozen=True, slots=True)
class OrchestratorToolRequest:
    """Represent one remote tool execution request."""

    call_id: str
    name: str
    arguments: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", _coerce_dict(self.arguments))  # AUDIT-FIX(#8): Freeze a defensive copy so callers cannot mutate request state after construction.

    def to_payload(self) -> dict[str, Any]:
        """Serialize the tool request into a websocket payload."""

        return {
            "type": "tool_request",
            "call_id": self.call_id,
            "name": self.name,
            "arguments": _json_safe(self.arguments),  # AUDIT-FIX(#8): Return a defensive JSON-safe copy instead of exposing the live mutable dict reference.
        }

    @classmethod
    def from_payload(cls, payload: Any) -> OrchestratorToolRequest:
        """Build a tool request from an incoming websocket payload."""

        payload_dict = _coerce_dict(payload)  # AUDIT-FIX(#3): Tolerate malformed transport frames instead of assuming a dict and crashing.
        return cls(
            call_id=str(payload_dict.get("call_id", "") or "").strip(),
            name=str(payload_dict.get("name", "") or "").strip(),
            arguments=_coerce_dict(payload_dict.get("arguments")),
        )


@dataclass(frozen=True, slots=True)
class OrchestratorToolResponse:
    """Represent the result of a remotely executed tool call."""

    call_id: str
    ok: bool
    output: Any | None = None  # AUDIT-FIX(#4): Preserve non-dict tool outputs such as lists, strings, and numbers instead of silently dropping them.
    error: str | None = None

    def to_payload(self) -> dict[str, Any]:
        """Serialize the tool response into a websocket payload."""

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
    def from_payload(cls, payload: Any) -> OrchestratorToolResponse:
        """Build a tool response from an incoming websocket payload."""

        payload_dict = _coerce_dict(payload)  # AUDIT-FIX(#3): Tolerate malformed transport frames instead of assuming a dict and crashing.
        return cls(
            call_id=str(payload_dict.get("call_id", "") or "").strip(),
            ok=_coerce_bool(payload_dict.get("ok")),  # AUDIT-FIX(#5): Parse transport booleans explicitly so "false"/"0" do not become True.
            output=_coerce_optional_json_value(payload_dict.get("output")),  # AUDIT-FIX(#4): Preserve arbitrary JSON-like tool outputs on decode.
            error=_coerce_optional_text(payload_dict.get("error")),
        )


@dataclass(frozen=True, slots=True)
class OrchestratorAckEvent:
    """Represent a normalized fast-ack event for the client."""

    ack_id: str
    text: str

    def to_payload(self) -> dict[str, Any]:
        """Serialize the ack event into a websocket payload."""

        return {"type": "ack", "ack_id": self.ack_id, "text": self.text}


@dataclass(frozen=True, slots=True)
class OrchestratorTextDeltaEvent:
    """Represent one streamed text delta from the orchestrator."""

    delta: str

    def to_payload(self) -> dict[str, Any]:
        """Serialize the text delta into a websocket payload."""

        return {"type": "text_delta", "delta": self.delta}


@dataclass(frozen=True, slots=True)
class OrchestratorErrorEvent:
    """Represent a sanitized turn-level transport failure."""

    error: str

    def to_payload(self) -> dict[str, Any]:
        """Serialize the error event into a websocket payload."""

        return {"type": "turn_error", "error": self.error}


@dataclass(frozen=True, slots=True)
class OrchestratorTurnCompleteEvent:
    """Represent the final result of one orchestrated turn."""

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
        """Serialize the completed turn into a websocket payload."""

        return {
            "type": "turn_complete",
            "text": self.text,
            "rounds": self.rounds,
            "used_web_search": self.used_web_search,
            "response_id": self.response_id,
            "request_id": self.request_id,
            "model": self.model,
            "token_usage": _json_safe(self.token_usage),
            "tool_calls": [_json_safe_record(item) for item in self.tool_calls],  # AUDIT-FIX(#1): Serialize contract objects without assuming dataclass-only implementations.
            "tool_results": [_json_safe_record(item) for item in self.tool_results],  # AUDIT-FIX(#1): Serialize contract objects without assuming dataclass-only implementations.
        }

    @classmethod
    def from_payload(cls, payload: Any) -> OrchestratorTurnCompleteEvent:
        """Build a completed-turn event from an incoming websocket payload."""

        payload_dict = _coerce_dict(payload)  # AUDIT-FIX(#3): Tolerate malformed transport frames instead of assuming a dict and crashing.
        return cls(
            text=str(payload_dict.get("text", "") or ""),
            rounds=_coerce_non_negative_int(payload_dict.get("rounds")),  # AUDIT-FIX(#6): Prevent ValueError crashes and clamp invalid negative round counts.
            used_web_search=_coerce_bool(payload_dict.get("used_web_search")),  # AUDIT-FIX(#5): Parse transport booleans explicitly so "false"/"0" do not become True.
            response_id=_coerce_optional_text(payload_dict.get("response_id")),
            request_id=_coerce_optional_text(payload_dict.get("request_id")),
            model=_coerce_optional_text(payload_dict.get("model")),
            token_usage=_coerce_optional_dict(payload_dict.get("token_usage")),
            tool_calls=tuple(_coerce_tool_calls(payload_dict.get("tool_calls"))),
            tool_results=tuple(_coerce_tool_results(payload_dict.get("tool_results"))),
        )


@dataclass(slots=True)
class OrchestratorClientTurnResult:
    """Collect the decoded result of one client-managed orchestrator turn."""

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
    """Convert a scalar payload field into stripped text or ``None``."""

    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_optional_json_value(value: Any) -> Any | None:
    """Normalize an optional JSON-like payload value."""

    if value is None:
        return None
    return _json_safe(value)  # AUDIT-FIX(#4): Keep arbitrary JSON-like outputs instead of narrowing everything to dict-only payloads.


def _coerce_dict(value: Any) -> dict[str, Any]:
    """Convert mapping-like payload values into plain dictionaries."""

    if isinstance(value, Mapping):  # AUDIT-FIX(#7): Accept mapping-like payloads instead of silently rejecting non-literal dict implementations.
        try:
            return dict(value)
        except Exception:
            return {}
    return {}


def _coerce_optional_dict(value: Any) -> dict[str, Any] | None:
    """Convert an optional mapping-like payload value into a plain dictionary."""

    if value is None:
        return None
    if isinstance(value, Mapping):  # AUDIT-FIX(#7): Accept mapping-like payloads instead of silently rejecting non-literal dict implementations.
        try:
            return dict(value)
        except Exception:
            return None
    return None


def _is_non_string_sequence(value: Any) -> bool:
    """Return whether ``value`` is a sequence that is not string-like."""

    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _coerce_bool(value: Any) -> bool:
    """Parse transport booleans while failing closed on unknown tokens."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            return False
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
    return False  # AUDIT-FIX(#5): Unknown tokens fail closed instead of truthifying every non-empty object/string.


def _coerce_non_negative_int(value: Any) -> int:
    """Parse a non-negative integer field from transport data."""

    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return 0
        return max(int(value), 0)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0
        try:
            parsed = int(text, 10)
        except ValueError:
            return 0
        return max(parsed, 0)
    return 0  # AUDIT-FIX(#6): Malformed values now degrade to a safe default instead of raising during frame parsing.


def _coerce_conversation(value: Any) -> tuple[tuple[str, str], ...]:
    """Convert serialized conversation pairs into normalized tuples."""

    if not _is_non_string_sequence(value):  # AUDIT-FIX(#7): Accept tuples from in-process callers while rejecting scalar strings/bytes.
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
    """Decode serialized tool-call records into ``AgentToolCall`` objects."""

    if not _is_non_string_sequence(value):  # AUDIT-FIX(#7): Accept tuples from in-process callers while rejecting scalar strings/bytes.
        return []
    items: list[AgentToolCall] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        try:
            items.append(  # AUDIT-FIX(#7): Skip malformed entries instead of letting a single bad tool call crash the whole turn decode.
                AgentToolCall(
                    name=str(item.get("name", "") or "").strip(),
                    call_id=str(item.get("call_id", "") or "").strip(),
                    arguments=_coerce_dict(item.get("arguments")),
                    raw_arguments=str(item.get("raw_arguments", "") or ""),
                )
            )
        except Exception:
            continue
    return items


def _coerce_tool_results(value: Any) -> list[AgentToolResult]:
    """Decode serialized tool-result records into ``AgentToolResult`` objects."""

    if not _is_non_string_sequence(value):  # AUDIT-FIX(#7): Accept tuples from in-process callers while rejecting scalar strings/bytes.
        return []
    items: list[AgentToolResult] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        try:
            items.append(  # AUDIT-FIX(#7): Skip malformed entries instead of letting a single bad tool result crash the whole turn decode.
                AgentToolResult(
                    call_id=str(item.get("call_id", "") or "").strip(),
                    name=str(item.get("name", "") or "").strip(),
                    output=_coerce_optional_json_value(item.get("output")),  # AUDIT-FIX(#4): Preserve arbitrary JSON-like tool outputs on nested tool-result decode.
                    serialized_output=str(item.get("serialized_output", "") or ""),
                )
            )
        except Exception:
            continue
    return items


def _safe_stringify(value: Any) -> str:
    """Stringify arbitrary values without letting ``str()`` raise."""

    try:
        return str(value)
    except Exception:
        return f"<unserializable:{type(value).__name__}>"


def _json_safe_record(value: Any) -> dict[str, Any]:
    """Convert an arbitrary record-like object into a JSON-safe dictionary."""

    if is_dataclass(value) and not isinstance(value, type):
        try:
            safe_value = _json_safe(asdict(value))  # AUDIT-FIX(#1): Fast-path simple dataclass contract objects while retaining safe fallback below.
            if isinstance(safe_value, dict):
                return safe_value
        except (TypeError, RecursionError, ValueError):
            pass
    safe_value = _json_safe(value)
    if isinstance(safe_value, dict):
        return safe_value
    return {"value": safe_value}


def _json_safe(value: Any, *, _seen: set[int] | None = None, _depth: int = 0) -> Any:
    """Convert nested transport data into JSON-safe values.

    Normalizes dates, paths, dataclasses, and model-like objects while bounding
    recursion and breaking cycles.
    """

    if _depth >= _MAX_JSON_DEPTH:
        return _MAX_DEPTH_SENTINEL  # AUDIT-FIX(#1): Hard-stop pathological nesting instead of crashing with RecursionError.
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _safe_stringify(value)  # AUDIT-FIX(#1): Normalize NaN/Inf into JSON-safe strings.
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)

    if _seen is None:
        _seen = set()

    if isinstance(value, Mapping):
        object_id = id(value)
        if object_id in _seen:
            return _RECURSION_SENTINEL  # AUDIT-FIX(#1): Break container reference cycles instead of infinite recursion.
        _seen.add(object_id)
        try:
            return {str(key): _json_safe(item, _seen=_seen, _depth=_depth + 1) for key, item in value.items()}
        finally:
            _seen.discard(object_id)

    if isinstance(value, (list, tuple, set)):
        object_id = id(value)
        if object_id in _seen:
            return _RECURSION_SENTINEL  # AUDIT-FIX(#1): Break container reference cycles instead of infinite recursion.
        _seen.add(object_id)
        try:
            return [_json_safe(item, _seen=_seen, _depth=_depth + 1) for item in value]
        finally:
            _seen.discard(object_id)

    if is_dataclass(value) and not isinstance(value, type):
        object_id = id(value)
        if object_id in _seen:
            return _RECURSION_SENTINEL  # AUDIT-FIX(#1): Break dataclass reference cycles instead of infinite recursion.
        _seen.add(object_id)
        try:
            return {
                item.name: _json_safe(getattr(value, item.name), _seen=_seen, _depth=_depth + 1)
                for item in fields(value)
            }
        finally:
            _seen.discard(object_id)

    object_id = id(value)
    if object_id in _seen:
        return _RECURSION_SENTINEL  # AUDIT-FIX(#1): Break object serializer cycles instead of infinite recursion.
    _seen.add(object_id)
    try:
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            try:
                return _json_safe(model_dump(), _seen=_seen, _depth=_depth + 1)
            except Exception:
                return _safe_stringify(value)
        dict_method = getattr(value, "dict", None)
        if callable(dict_method):
            try:
                return _json_safe(dict_method(), _seen=_seen, _depth=_depth + 1)
            except Exception:
                return _safe_stringify(value)
        return _safe_stringify(value)  # AUDIT-FIX(#2): Do not walk arbitrary __dict__ state, which can leak secrets/internal object internals.
    finally:
        _seen.discard(object_id)
