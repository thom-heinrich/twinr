"""Define websocket transport contracts for the Twinr orchestrator.

This module owns the dataclasses that cross the edge-orchestrator websocket
boundary plus the defensive coercion helpers that keep malformed payloads from
crashing turn handling.
"""

from __future__ import annotations

# CHANGELOG: 2026-03-29
# BUG-1: Stop stripping prompt/history content and preserve empty conversation messages so whitespace-sensitive prompts/code survive transport unchanged.
# BUG-2: Deep-normalize nested payload dictionaries at construction time to eliminate post-construction mutation leaks/races across async turn handling.
# BUG-3: Remove the dataclass asdict() deep-copy fast-path that doubled memory for large tool outputs and caused avoidable Pi 4 latency spikes.
# BUG-4: Encode bytes-like payloads as tagged base64 transport objects instead of lossy Python repr strings.
# SEC-1: Enforce hard ceilings for text length, container breadth, binary size, recursion depth, and total JSON nodes to reduce practical memory/CPU DoS risk on Raspberry Pi 4.
# IMP-1: Add schema_version tagging, symmetric JSON bytes encode/decode helpers, and from_payload() support for all streamed event types.
# IMP-2: Add an optional msgspec fast path for JSON transport on Python 3.9+ with tiny aarch64 wheels and schema-first validation compatibility.
# BREAKING: bytes/bytearray/memoryview values now serialize as {"__twinr_bytes__": true, ...} tagged base64 envelopes instead of Python repr strings.
# BREAKING: oversize text/container/binary payloads are now truncated or capped instead of being forwarded unbounded.

import base64
import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID

from twinr.agent.base_agent.contracts import AgentToolCall, AgentToolResult

try:
    import msgspec
except Exception:  # pragma: no cover - optional dependency
    msgspec = None


_TRANSPORT_SCHEMA_VERSION = 2

_MAX_JSON_DEPTH = 32
_MAX_JSON_NODES = 10_000
_MAX_CONTAINER_ITEMS = 512
_MAX_CONVERSATION_ITEMS = 256
_MAX_TOOL_RECORDS = 256
_MAX_TEXT_CHARS = 64_000
_MAX_ID_CHARS = 256
_MAX_ROLE_CHARS = 64
_MAX_KEY_CHARS = 256
_MAX_BINARY_BYTES = 256 * 1024

_RECURSION_SENTINEL = "<recursive-ref>"
_MAX_DEPTH_SENTINEL = "<max-depth-exceeded>"
_MAX_NODES_SENTINEL = "<max-nodes-exceeded>"
_TRUNCATED_SENTINEL = "<truncated>"
_BYTES_TAG = "__twinr_bytes__"

_TRUE_STRINGS = frozenset({"1", "true", "t", "yes", "y", "on"})
_FALSE_STRINGS = frozenset({"0", "false", "f", "no", "n", "off", ""})

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _JsonBudget:
    remaining_nodes: int = _MAX_JSON_NODES


def _warn_unexpected_type(payload_dict: Mapping[str, Any], expected_type: str) -> None:
    """Log, but do not reject, frames with an unexpected payload type."""

    payload_type = payload_dict.get("type")
    if payload_type in (None, "", expected_type):
        return
    _LOGGER.warning(
        "Unexpected websocket payload type %r while decoding %s.",
        payload_type,
        expected_type,
    )


def _safe_stringify(value: Any) -> str:
    """Stringify arbitrary values without letting ``str()`` raise."""

    try:
        return str(value)
    except Exception:
        return f"<unserializable:{type(value).__name__}>"


def _truncate_text(text: str, max_chars: int) -> str:
    """Bound text length while preserving leading/trailing whitespace."""

    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= len(_TRUNCATED_SENTINEL):
        return _TRUNCATED_SENTINEL[:max_chars]
    _LOGGER.warning(
        "Truncating websocket text field from %s to %s characters.",
        len(text),
        max_chars,
    )
    return text[: max_chars - len(_TRUNCATED_SENTINEL)] + _TRUNCATED_SENTINEL


def _coerce_text(value: Any, *, strip: bool = False, max_chars: int = _MAX_TEXT_CHARS) -> str:
    """Convert a scalar payload field into bounded text."""

    if value is None:
        return ""
    text = value if isinstance(value, str) else _safe_stringify(value)
    text = text.replace("\x00", "")
    if strip:
        text = text.strip()
    return _truncate_text(text, max_chars)


def _coerce_optional_text(value: Any, *, strip: bool = False, max_chars: int = _MAX_TEXT_CHARS) -> str | None:
    """Convert a scalar payload field into bounded text or ``None``."""

    if value is None:
        return None
    text = _coerce_text(value, strip=strip, max_chars=max_chars)
    return text or None


def _coerce_identifier(value: Any, *, max_chars: int = _MAX_ID_CHARS) -> str:
    """Convert identifiers into bounded stripped text."""

    return _coerce_text(value, strip=True, max_chars=max_chars)


def _is_non_string_sequence(value: Any) -> bool:
    """Return whether ``value`` is a sequence that is not string-like."""

    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview))


def _decode_json_payload(value: Any) -> Any:
    """Decode raw JSON bytes or strings into builtin Python objects."""

    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
    elif isinstance(value, str):
        raw = value.encode("utf-8")
    else:
        return value

    if msgspec is not None:
        try:
            return msgspec.json.decode(raw)
        except Exception:
            pass

    try:
        return json.loads(raw)
    except Exception:
        _LOGGER.debug("Failed to decode websocket JSON payload.", exc_info=True)
        return {}


def _coerce_payload_dict(value: Any) -> dict[str, Any]:
    """Convert mappings or raw JSON objects into bounded plain dictionaries."""

    value = _decode_json_payload(value)
    if isinstance(value, Mapping):
        safe_value = _json_safe(value)
        if isinstance(safe_value, dict):
            return safe_value
    return {}


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
    return False


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
            return max(int(text, 10), 0)
        except ValueError:
            return 0
    return 0


def _encode_bytes_like(value: bytes | bytearray | memoryview) -> dict[str, Any]:
    """Encode bytes-like data into a JSON-safe tagged base64 envelope."""

    raw = bytes(value)
    original_size = len(raw)
    truncated = original_size > _MAX_BINARY_BYTES
    if truncated:
        _LOGGER.warning(
            "Truncating bytes-like websocket field from %s to %s bytes.",
            original_size,
            _MAX_BINARY_BYTES,
        )
        raw = raw[:_MAX_BINARY_BYTES]
    return {
        _BYTES_TAG: True,
        "encoding": "base64",
        "size": original_size,
        "truncated": truncated,
        "data": base64.b64encode(raw).decode("ascii"),
    }


def _is_pydantic_model_instance(value: Any) -> bool:
    """Best-effort detection for pydantic model instances without importing pydantic."""

    cls = type(value)
    return hasattr(cls, "model_fields") or hasattr(cls, "__fields__")


def _json_safe(
    value: Any,
    *,
    _seen: set[int] | None = None,
    _depth: int = 0,
    _budget: _JsonBudget | None = None,
) -> Any:
    """Convert nested transport data into bounded JSON-safe values."""

    if _depth >= _MAX_JSON_DEPTH:
        return _MAX_DEPTH_SENTINEL
    if _budget is None:
        _budget = _JsonBudget()
    _budget.remaining_nodes -= 1
    if _budget.remaining_nodes < 0:
        return _MAX_NODES_SENTINEL

    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        return _truncate_text(value.replace("\x00", ""), _MAX_TEXT_CHARS)
    if isinstance(value, float):
        return value if math.isfinite(value) else _safe_stringify(value)
    if isinstance(value, Decimal):
        if not value.is_finite():
            return _safe_stringify(value)
        return format(value, "f")
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (Path, UUID)):
        return str(value)
    if isinstance(value, Enum):
        return _json_safe(value.value, _seen=_seen, _depth=_depth + 1, _budget=_budget)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _encode_bytes_like(value)

    if _seen is None:
        _seen = set()

    if isinstance(value, Mapping):
        object_id = id(value)
        if object_id in _seen:
            return _RECURSION_SENTINEL
        _seen.add(object_id)
        try:
            result: dict[str, Any] = {}
            for index, (key, item) in enumerate(value.items()):
                if index >= _MAX_CONTAINER_ITEMS:
                    _LOGGER.warning(
                        "Truncating websocket mapping with more than %s items.",
                        _MAX_CONTAINER_ITEMS,
                    )
                    break
                result[_truncate_text(_safe_stringify(key), _MAX_KEY_CHARS)] = _json_safe(
                    item,
                    _seen=_seen,
                    _depth=_depth + 1,
                    _budget=_budget,
                )
            return result
        finally:
            _seen.discard(object_id)

    if isinstance(value, (list, tuple, set, frozenset)):
        object_id = id(value)
        if object_id in _seen:
            return _RECURSION_SENTINEL
        _seen.add(object_id)
        try:
            iterable = value
            if isinstance(value, (set, frozenset)):
                iterable = sorted(value, key=_safe_stringify)
            result: list[Any] = []
            for index, item in enumerate(iterable):
                if index >= _MAX_CONTAINER_ITEMS:
                    _LOGGER.warning(
                        "Truncating websocket sequence with more than %s items.",
                        _MAX_CONTAINER_ITEMS,
                    )
                    break
                result.append(_json_safe(item, _seen=_seen, _depth=_depth + 1, _budget=_budget))
            return result
        finally:
            _seen.discard(object_id)

    if is_dataclass(value) and not isinstance(value, type):
        object_id = id(value)
        if object_id in _seen:
            return _RECURSION_SENTINEL
        _seen.add(object_id)
        try:
            return {
                item.name: _json_safe(
                    getattr(value, item.name),
                    _seen=_seen,
                    _depth=_depth + 1,
                    _budget=_budget,
                )
                for item in fields(value)
            }
        finally:
            _seen.discard(object_id)

    object_id = id(value)
    if object_id in _seen:
        return _RECURSION_SENTINEL
    _seen.add(object_id)
    try:
        if _is_pydantic_model_instance(value):
            model_dump = getattr(value, "model_dump", None)
            if callable(model_dump):
                try:
                    return _json_safe(model_dump(mode="python"), _seen=_seen, _depth=_depth + 1, _budget=_budget)
                except Exception:
                    return _safe_stringify(value)
            dict_method = getattr(value, "dict", None)
            if callable(dict_method):
                try:
                    return _json_safe(dict_method(), _seen=_seen, _depth=_depth + 1, _budget=_budget)
                except Exception:
                    return _safe_stringify(value)
        if msgspec is not None:
            to_builtins = getattr(msgspec, "to_builtins", None)
            if callable(to_builtins):
                try:
                    builtins_value = to_builtins(value)
                except Exception:
                    builtins_value = value
                if builtins_value is not value:
                    return _json_safe(builtins_value, _seen=_seen, _depth=_depth + 1, _budget=_budget)
        return _truncate_text(_safe_stringify(value), _MAX_TEXT_CHARS)
    finally:
        _seen.discard(object_id)


def _json_safe_record(value: Any) -> dict[str, Any]:
    """Convert an arbitrary record-like object into a JSON-safe dictionary."""

    safe_value = _json_safe(value)
    if isinstance(safe_value, dict):
        return safe_value
    return {"value": safe_value}


def _coerce_optional_json_value(value: Any) -> Any | None:
    """Normalize an optional JSON-like payload value."""

    if value is None:
        return None
    return _json_safe(value)


def _coerce_dict(value: Any) -> dict[str, Any]:
    """Convert mapping-like payload values into bounded plain dictionaries."""

    safe_value = _json_safe(value)
    return safe_value if isinstance(safe_value, dict) else {}


def _coerce_optional_dict(value: Any) -> dict[str, Any] | None:
    """Convert an optional mapping-like payload value into a bounded plain dictionary."""

    if value is None:
        return None
    safe_value = _json_safe(value)
    return safe_value if isinstance(safe_value, dict) else None


def _coerce_conversation(value: Any) -> tuple[tuple[str, str], ...]:
    """Convert serialized conversation pairs into normalized tuples."""

    value = _decode_json_payload(value)
    if not _is_non_string_sequence(value):
        return ()
    items: list[tuple[str, str]] = []
    for index, item in enumerate(value):
        if index >= _MAX_CONVERSATION_ITEMS:
            _LOGGER.warning(
                "Truncating websocket conversation with more than %s turns.",
                _MAX_CONVERSATION_ITEMS,
            )
            break
        role = ""
        content = ""
        if isinstance(item, Mapping):
            role = _coerce_text(item.get("role"), strip=True, max_chars=_MAX_ROLE_CHARS)
            content = _coerce_text(item.get("content"), strip=False, max_chars=_MAX_TEXT_CHARS)
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            role = _coerce_text(item[0], strip=True, max_chars=_MAX_ROLE_CHARS)
            content = _coerce_text(item[1], strip=False, max_chars=_MAX_TEXT_CHARS)
        if role:
            items.append((role, content))
    return tuple(items)


def _coerce_tool_calls(value: Any) -> list[AgentToolCall]:
    """Decode serialized tool-call records into ``AgentToolCall`` objects."""

    value = _decode_json_payload(value)
    if not _is_non_string_sequence(value):
        return []
    items: list[AgentToolCall] = []
    for index, item in enumerate(value):
        if index >= _MAX_TOOL_RECORDS:
            _LOGGER.warning(
                "Truncating websocket tool-call list with more than %s items.",
                _MAX_TOOL_RECORDS,
            )
            break
        try:
            if isinstance(item, AgentToolCall):
                items.append(
                    AgentToolCall(
                        name=_coerce_identifier(getattr(item, "name", "")),
                        call_id=_coerce_identifier(getattr(item, "call_id", "")),
                        arguments=_coerce_dict(getattr(item, "arguments", {})),
                        raw_arguments=_coerce_text(getattr(item, "raw_arguments", ""), strip=False),
                    )
                )
                continue
            if not isinstance(item, Mapping):
                continue
            items.append(
                AgentToolCall(
                    name=_coerce_identifier(item.get("name")),
                    call_id=_coerce_identifier(item.get("call_id")),
                    arguments=_coerce_dict(item.get("arguments")),
                    raw_arguments=_coerce_text(item.get("raw_arguments", ""), strip=False),
                )
            )
        except Exception:
            _LOGGER.warning("Skipping malformed orchestrator tool-call record during decode.", exc_info=True)
            continue
    return items


def _coerce_tool_results(value: Any) -> list[AgentToolResult]:
    """Decode serialized tool-result records into ``AgentToolResult`` objects."""

    value = _decode_json_payload(value)
    if not _is_non_string_sequence(value):
        return []
    items: list[AgentToolResult] = []
    for index, item in enumerate(value):
        if index >= _MAX_TOOL_RECORDS:
            _LOGGER.warning(
                "Truncating websocket tool-result list with more than %s items.",
                _MAX_TOOL_RECORDS,
            )
            break
        try:
            if isinstance(item, AgentToolResult):
                items.append(
                    AgentToolResult(
                        call_id=_coerce_identifier(getattr(item, "call_id", "")),
                        name=_coerce_identifier(getattr(item, "name", "")),
                        output=_coerce_optional_json_value(getattr(item, "output", None)),
                        serialized_output=_coerce_text(getattr(item, "serialized_output", ""), strip=False),
                    )
                )
                continue
            if not isinstance(item, Mapping):
                continue
            items.append(
                AgentToolResult(
                    call_id=_coerce_identifier(item.get("call_id")),
                    name=_coerce_identifier(item.get("name")),
                    output=_coerce_optional_json_value(item.get("output")),
                    serialized_output=_coerce_text(item.get("serialized_output", ""), strip=False),
                )
            )
        except Exception:
            _LOGGER.warning("Skipping malformed orchestrator tool-result record during decode.", exc_info=True)
            continue
    return items


def _coerce_ack_events(value: Any) -> list["OrchestratorAckEvent"]:
    """Decode ack records into normalized ack event objects."""

    value = _decode_json_payload(value)
    if not _is_non_string_sequence(value):
        return []
    items: list[OrchestratorAckEvent] = []
    for index, item in enumerate(value):
        if index >= _MAX_TOOL_RECORDS:
            _LOGGER.warning(
                "Truncating websocket ack-event list with more than %s items.",
                _MAX_TOOL_RECORDS,
            )
            break
        if isinstance(item, OrchestratorAckEvent):
            items.append(item)
            continue
        if isinstance(item, Mapping):
            items.append(OrchestratorAckEvent.from_payload(item))
    return items


def encode_transport_json(value: Any) -> bytes:
    """Encode a transport payload into UTF-8 JSON bytes."""

    safe_value = _json_safe(value)
    if msgspec is not None:
        try:
            return msgspec.json.encode(safe_value)
        except Exception:
            _LOGGER.debug("msgspec JSON encode failed; falling back to stdlib json.", exc_info=True)
    return json.dumps(safe_value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def transport_contract_schema() -> dict[str, Any]:
    """Return a compact JSON-Schema-like description of websocket payload contracts."""

    json_value_schema: dict[str, Any] = {
        "description": "Any JSON-safe value plus Twinr tagged bytes envelopes.",
        "oneOf": [
            {"type": "null"},
            {"type": "boolean"},
            {"type": "integer"},
            {"type": "number"},
            {"type": "string", "maxLength": _MAX_TEXT_CHARS},
            {"type": "array", "maxItems": _MAX_CONTAINER_ITEMS},
            {"type": "object"},
        ],
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://twinr.local/schemas/orchestrator-websocket-contracts.json",
        "title": "Twinr Orchestrator WebSocket Contracts",
        "description": "Schema summary for the payloads crossing the edge-orchestrator websocket boundary.",
        "x-schema-version": _TRANSPORT_SCHEMA_VERSION,
        "$defs": {
            "conversation_entry": {
                "type": "array",
                "prefixItems": [
                    {"type": "string", "minLength": 1, "maxLength": _MAX_ROLE_CHARS},
                    {"type": "string", "maxLength": _MAX_TEXT_CHARS},
                ],
                "items": False,
            },
            "tool_call": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "maxLength": _MAX_ID_CHARS},
                    "call_id": {"type": "string", "maxLength": _MAX_ID_CHARS},
                    "arguments": {"type": "object"},
                    "raw_arguments": {"type": "string", "maxLength": _MAX_TEXT_CHARS},
                },
            },
            "tool_result": {
                "type": "object",
                "properties": {
                    "call_id": {"type": "string", "maxLength": _MAX_ID_CHARS},
                    "name": {"type": "string", "maxLength": _MAX_ID_CHARS},
                    "output": json_value_schema,
                    "serialized_output": {"type": "string", "maxLength": _MAX_TEXT_CHARS},
                },
            },
        },
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "schema_version": {"const": _TRANSPORT_SCHEMA_VERSION},
                    "type": {"const": "run_turn"},
                    "prompt": {"type": "string", "maxLength": _MAX_TEXT_CHARS},
                    "conversation": {"type": "array", "maxItems": _MAX_CONVERSATION_ITEMS},
                    "supervisor_conversation": {"type": "array", "maxItems": _MAX_CONVERSATION_ITEMS},
                },
                "required": ["type", "prompt"],
            },
            {
                "type": "object",
                "properties": {
                    "schema_version": {"const": _TRANSPORT_SCHEMA_VERSION},
                    "type": {"const": "tool_request"},
                    "call_id": {"type": "string", "maxLength": _MAX_ID_CHARS},
                    "name": {"type": "string", "maxLength": _MAX_ID_CHARS},
                    "arguments": {"type": "object"},
                },
                "required": ["type", "call_id", "name", "arguments"],
            },
            {
                "type": "object",
                "properties": {
                    "schema_version": {"const": _TRANSPORT_SCHEMA_VERSION},
                    "type": {"const": "tool_result"},
                    "call_id": {"type": "string", "maxLength": _MAX_ID_CHARS},
                    "ok": {"type": "boolean"},
                    "output": json_value_schema,
                    "error": {"type": "string", "maxLength": _MAX_TEXT_CHARS},
                },
                "required": ["type", "call_id", "ok"],
            },
            {
                "type": "object",
                "properties": {
                    "schema_version": {"const": _TRANSPORT_SCHEMA_VERSION},
                    "type": {"const": "ack"},
                    "ack_id": {"type": "string", "maxLength": _MAX_ID_CHARS},
                    "text": {"type": "string", "maxLength": _MAX_TEXT_CHARS},
                },
                "required": ["type", "ack_id", "text"],
            },
            {
                "type": "object",
                "properties": {
                    "schema_version": {"const": _TRANSPORT_SCHEMA_VERSION},
                    "type": {"const": "text_delta"},
                    "delta": {"type": "string", "maxLength": _MAX_TEXT_CHARS},
                },
                "required": ["type", "delta"],
            },
            {
                "type": "object",
                "properties": {
                    "schema_version": {"const": _TRANSPORT_SCHEMA_VERSION},
                    "type": {"const": "turn_error"},
                    "error": {"type": "string", "maxLength": _MAX_TEXT_CHARS},
                },
                "required": ["type", "error"],
            },
            {
                "type": "object",
                "properties": {
                    "schema_version": {"const": _TRANSPORT_SCHEMA_VERSION},
                    "type": {"const": "turn_complete"},
                    "text": {"type": "string", "maxLength": _MAX_TEXT_CHARS},
                    "rounds": {"type": "integer", "minimum": 0},
                    "used_web_search": {"type": "boolean"},
                    "response_id": {"type": ["string", "null"], "maxLength": _MAX_ID_CHARS},
                    "request_id": {"type": ["string", "null"], "maxLength": _MAX_ID_CHARS},
                    "model": {"type": ["string", "null"], "maxLength": _MAX_ID_CHARS},
                    "token_usage": {"type": ["object", "null"]},
                    "tool_calls": {"type": "array", "maxItems": _MAX_TOOL_RECORDS},
                    "tool_results": {"type": "array", "maxItems": _MAX_TOOL_RECORDS},
                },
                "required": ["type", "text", "rounds", "used_web_search"],
            },
        ],
    }


@dataclass(frozen=True, slots=True)
class OrchestratorTurnRequest:
    """Represent one remote request to run an orchestrated turn."""

    prompt: str
    conversation: tuple[tuple[str, str], ...] = ()
    supervisor_conversation: tuple[tuple[str, str], ...] = ()
    prefetched_supervisor_decision: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt", _coerce_text(self.prompt, strip=False, max_chars=_MAX_TEXT_CHARS))
        object.__setattr__(self, "conversation", _coerce_conversation(self.conversation))
        object.__setattr__(self, "supervisor_conversation", _coerce_conversation(self.supervisor_conversation))
        prefetched = self.prefetched_supervisor_decision
        object.__setattr__(
            self,
            "prefetched_supervisor_decision",
            _coerce_dict(prefetched) if isinstance(prefetched, Mapping) else None,
        )

    def to_payload(self) -> dict[str, Any]:
        """Serialize the turn request into a websocket payload."""

        return {
            "schema_version": _TRANSPORT_SCHEMA_VERSION,
            "type": "run_turn",
            "prompt": self.prompt,
            "conversation": [list(item) for item in self.conversation],
            "supervisor_conversation": [list(item) for item in self.supervisor_conversation],
            "prefetched_supervisor_decision": dict(self.prefetched_supervisor_decision or {}),
        }

    def to_json_bytes(self) -> bytes:
        """Serialize the turn request into UTF-8 JSON bytes."""

        return encode_transport_json(self.to_payload())

    @classmethod
    def from_payload(cls, payload: Any) -> "OrchestratorTurnRequest":
        """Build a turn request from an incoming websocket payload."""

        payload_dict = _coerce_payload_dict(payload)
        _warn_unexpected_type(payload_dict, "run_turn")
        return cls(
            prompt=_coerce_text(payload_dict.get("prompt"), strip=False, max_chars=_MAX_TEXT_CHARS),
            conversation=_coerce_conversation(payload_dict.get("conversation")),
            supervisor_conversation=_coerce_conversation(payload_dict.get("supervisor_conversation")),
            prefetched_supervisor_decision=(
                _coerce_dict(payload_dict.get("prefetched_supervisor_decision"))
                if isinstance(payload_dict.get("prefetched_supervisor_decision"), Mapping)
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class OrchestratorToolRequest:
    """Represent one remote tool execution request."""

    call_id: str
    name: str
    arguments: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "call_id", _coerce_identifier(self.call_id))
        object.__setattr__(self, "name", _coerce_identifier(self.name))
        object.__setattr__(self, "arguments", _coerce_dict(self.arguments))

    def to_payload(self) -> dict[str, Any]:
        """Serialize the tool request into a websocket payload."""

        return {
            "schema_version": _TRANSPORT_SCHEMA_VERSION,
            "type": "tool_request",
            "call_id": self.call_id,
            "name": self.name,
            "arguments": _json_safe(self.arguments),
        }

    def to_json_bytes(self) -> bytes:
        """Serialize the tool request into UTF-8 JSON bytes."""

        return encode_transport_json(self.to_payload())

    @classmethod
    def from_payload(cls, payload: Any) -> "OrchestratorToolRequest":
        """Build a tool request from an incoming websocket payload."""

        payload_dict = _coerce_payload_dict(payload)
        _warn_unexpected_type(payload_dict, "tool_request")
        return cls(
            call_id=_coerce_identifier(payload_dict.get("call_id")),
            name=_coerce_identifier(payload_dict.get("name")),
            arguments=_coerce_dict(payload_dict.get("arguments")),
        )


@dataclass(frozen=True, slots=True)
class OrchestratorToolResponse:
    """Represent the result of a remotely executed tool call."""

    call_id: str
    ok: bool
    output: Any | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "call_id", _coerce_identifier(self.call_id))
        object.__setattr__(self, "ok", _coerce_bool(self.ok))
        object.__setattr__(self, "output", _coerce_optional_json_value(self.output))
        object.__setattr__(self, "error", _coerce_optional_text(self.error, strip=False, max_chars=_MAX_TEXT_CHARS))

    def to_payload(self) -> dict[str, Any]:
        """Serialize the tool response into a websocket payload."""

        payload = {
            "schema_version": _TRANSPORT_SCHEMA_VERSION,
            "type": "tool_result",
            "call_id": self.call_id,
            "ok": self.ok,
        }
        if self.output is not None:
            payload["output"] = _json_safe(self.output)
        if self.error is not None:
            payload["error"] = self.error
        return payload

    def to_json_bytes(self) -> bytes:
        """Serialize the tool response into UTF-8 JSON bytes."""

        return encode_transport_json(self.to_payload())

    @classmethod
    def from_payload(cls, payload: Any) -> "OrchestratorToolResponse":
        """Build a tool response from an incoming websocket payload."""

        payload_dict = _coerce_payload_dict(payload)
        _warn_unexpected_type(payload_dict, "tool_result")
        return cls(
            call_id=_coerce_identifier(payload_dict.get("call_id")),
            ok=_coerce_bool(payload_dict.get("ok")),
            output=_coerce_optional_json_value(payload_dict.get("output")),
            error=_coerce_optional_text(payload_dict.get("error"), strip=False, max_chars=_MAX_TEXT_CHARS),
        )


@dataclass(frozen=True, slots=True)
class OrchestratorAckEvent:
    """Represent a normalized fast-ack event for the client."""

    ack_id: str
    text: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "ack_id", _coerce_identifier(self.ack_id))
        object.__setattr__(self, "text", _coerce_text(self.text, strip=False, max_chars=_MAX_TEXT_CHARS))

    def to_payload(self) -> dict[str, Any]:
        """Serialize the ack event into a websocket payload."""

        return {
            "schema_version": _TRANSPORT_SCHEMA_VERSION,
            "type": "ack",
            "ack_id": self.ack_id,
            "text": self.text,
        }

    def to_json_bytes(self) -> bytes:
        """Serialize the ack event into UTF-8 JSON bytes."""

        return encode_transport_json(self.to_payload())

    @classmethod
    def from_payload(cls, payload: Any) -> "OrchestratorAckEvent":
        """Build an ack event from an incoming websocket payload."""

        payload_dict = _coerce_payload_dict(payload)
        _warn_unexpected_type(payload_dict, "ack")
        return cls(
            ack_id=_coerce_identifier(payload_dict.get("ack_id")),
            text=_coerce_text(payload_dict.get("text"), strip=False, max_chars=_MAX_TEXT_CHARS),
        )


@dataclass(frozen=True, slots=True)
class OrchestratorTextDeltaEvent:
    """Represent one streamed text delta from the orchestrator."""

    delta: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "delta", _coerce_text(self.delta, strip=False, max_chars=_MAX_TEXT_CHARS))

    def to_payload(self) -> dict[str, Any]:
        """Serialize the text delta into a websocket payload."""

        return {
            "schema_version": _TRANSPORT_SCHEMA_VERSION,
            "type": "text_delta",
            "delta": self.delta,
        }

    def to_json_bytes(self) -> bytes:
        """Serialize the text delta into UTF-8 JSON bytes."""

        return encode_transport_json(self.to_payload())

    @classmethod
    def from_payload(cls, payload: Any) -> "OrchestratorTextDeltaEvent":
        """Build a text-delta event from an incoming websocket payload."""

        payload_dict = _coerce_payload_dict(payload)
        _warn_unexpected_type(payload_dict, "text_delta")
        return cls(delta=_coerce_text(payload_dict.get("delta"), strip=False, max_chars=_MAX_TEXT_CHARS))


@dataclass(frozen=True, slots=True)
class OrchestratorErrorEvent:
    """Represent a sanitized turn-level transport failure."""

    error: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "error", _coerce_text(self.error, strip=False, max_chars=_MAX_TEXT_CHARS))

    def to_payload(self) -> dict[str, Any]:
        """Serialize the error event into a websocket payload."""

        return {
            "schema_version": _TRANSPORT_SCHEMA_VERSION,
            "type": "turn_error",
            "error": self.error,
        }

    def to_json_bytes(self) -> bytes:
        """Serialize the error event into UTF-8 JSON bytes."""

        return encode_transport_json(self.to_payload())

    @classmethod
    def from_payload(cls, payload: Any) -> "OrchestratorErrorEvent":
        """Build an error event from an incoming websocket payload."""

        payload_dict = _coerce_payload_dict(payload)
        _warn_unexpected_type(payload_dict, "turn_error")
        return cls(error=_coerce_text(payload_dict.get("error"), strip=False, max_chars=_MAX_TEXT_CHARS))


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _coerce_text(self.text, strip=False, max_chars=_MAX_TEXT_CHARS))
        object.__setattr__(self, "rounds", _coerce_non_negative_int(self.rounds))
        object.__setattr__(self, "used_web_search", _coerce_bool(self.used_web_search))
        object.__setattr__(self, "response_id", _coerce_optional_text(self.response_id, strip=True, max_chars=_MAX_ID_CHARS))
        object.__setattr__(self, "request_id", _coerce_optional_text(self.request_id, strip=True, max_chars=_MAX_ID_CHARS))
        object.__setattr__(self, "model", _coerce_optional_text(self.model, strip=True, max_chars=_MAX_ID_CHARS))
        object.__setattr__(self, "token_usage", _coerce_optional_dict(self.token_usage))
        object.__setattr__(self, "tool_calls", tuple(_coerce_tool_calls(self.tool_calls)))
        object.__setattr__(self, "tool_results", tuple(_coerce_tool_results(self.tool_results)))

    def to_payload(self) -> dict[str, Any]:
        """Serialize the completed turn into a websocket payload."""

        return {
            "schema_version": _TRANSPORT_SCHEMA_VERSION,
            "type": "turn_complete",
            "text": self.text,
            "rounds": self.rounds,
            "used_web_search": self.used_web_search,
            "response_id": self.response_id,
            "request_id": self.request_id,
            "model": self.model,
            "token_usage": _json_safe(self.token_usage),
            "tool_calls": [_json_safe_record(item) for item in self.tool_calls],
            "tool_results": [_json_safe_record(item) for item in self.tool_results],
        }

    def to_json_bytes(self) -> bytes:
        """Serialize the completed turn into UTF-8 JSON bytes."""

        return encode_transport_json(self.to_payload())

    @classmethod
    def from_payload(cls, payload: Any) -> "OrchestratorTurnCompleteEvent":
        """Build a completed-turn event from an incoming websocket payload."""

        payload_dict = _coerce_payload_dict(payload)
        _warn_unexpected_type(payload_dict, "turn_complete")
        return cls(
            text=_coerce_text(payload_dict.get("text"), strip=False, max_chars=_MAX_TEXT_CHARS),
            rounds=_coerce_non_negative_int(payload_dict.get("rounds")),
            used_web_search=_coerce_bool(payload_dict.get("used_web_search")),
            response_id=_coerce_optional_text(payload_dict.get("response_id"), strip=True, max_chars=_MAX_ID_CHARS),
            request_id=_coerce_optional_text(payload_dict.get("request_id"), strip=True, max_chars=_MAX_ID_CHARS),
            model=_coerce_optional_text(payload_dict.get("model"), strip=True, max_chars=_MAX_ID_CHARS),
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

    def __post_init__(self) -> None:
        self.text = _coerce_text(self.text, strip=False, max_chars=_MAX_TEXT_CHARS)
        self.rounds = _coerce_non_negative_int(self.rounds)
        self.used_web_search = _coerce_bool(self.used_web_search)
        self.response_id = _coerce_optional_text(self.response_id, strip=True, max_chars=_MAX_ID_CHARS)
        self.request_id = _coerce_optional_text(self.request_id, strip=True, max_chars=_MAX_ID_CHARS)
        self.model = _coerce_optional_text(self.model, strip=True, max_chars=_MAX_ID_CHARS)
        self.token_usage = _coerce_optional_dict(self.token_usage)
        self.tool_calls = tuple(_coerce_tool_calls(self.tool_calls))
        self.tool_results = tuple(_coerce_tool_results(self.tool_results))
        self.ack_events = _coerce_ack_events(self.ack_events)
