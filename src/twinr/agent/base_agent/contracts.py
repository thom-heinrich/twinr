# CHANGELOG: 2026-03-27
# BUG-1: ToolCallingTurnResponse now fails fast when a tool-calling turn omits a continuation token,
#        because continue_turn_streaming cannot resume that turn correctly without one.
# BUG-2: SupervisorDecision runtime_tool_arguments now accepts generic Mapping payloads,
#        deep-copies nested data, and stops silently discarding malformed values.
# BUG-3: ConversationClosureProviderDecision and streaming result/event value objects now validate
#        confidence ranges and required text fields to prevent silent misrouting and invalid state.
# BUG-4: Audio MIME inference now normalizes to audio-safe container types instead of forwarding
#        arbitrary guessed non-audio MIME types to STT providers.
# SEC-1: transcribe_path now rejects symlinks, non-regular files, and oversized payloads, using
#        O_NOFOLLOW where available to prevent practical data exfiltration, device-file hangs, and Pi OOMs.
# SEC-2: Audio files are read in bounded chunks instead of unbounded Path.read_bytes().
# IMP-1: Added AgentToolSpec plus provider-neutral / OpenAI / Anthropic / MCP export helpers for
#        strict JSON-Schema tool contracts.
# IMP-2: Added structured tool-result metadata (is_error, mime_type, structured_output, metadata)
#        and tool-schema normalization helpers aligned with 2026 structured-agent patterns.
# IMP-3: CompositeSpeechAgentProvider now forwards live streaming STT sessions when the injected
#        STT provider supports them.
# IMP-4: Added stronger invariants, safer config rollback snapshots, an explicit public export list,
#        and JSON-schema subset validation helpers for edge-friendly contract enforcement.

"""Define provider contracts and composition helpers for the base agent.

This module is the structural contract layer between workflow/runtime code and
provider implementations. It keeps protocol surfaces, tool-call value objects,
and composite provider glue in one import-stable place.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
import errno
import io
import json
import logging
import math
import mimetypes
import os
from pathlib import Path
import re
import stat
import threading
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias, runtime_checkable

if TYPE_CHECKING:
    from twinr.agent.base_agent.config import TwinrConfig
else:
    TwinrConfig = Any  # AUDIT-FIX(#6): Stellt einen Runtime-Fallback für Annotation-Resolution via get_type_hints bereit.

_LOGGER = logging.getLogger(__name__)

ConversationLike = Sequence[object]
ImageInputLike = object
JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
ToolSchemaDialect: TypeAlias = Literal["neutral", "openai", "anthropic", "mcp"]


def _get_env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        _LOGGER.warning("Ignoring invalid integer environment override for %s=%r", name, raw_value)
        return default


DEFAULT_MAX_AUDIO_FILE_BYTES = _get_env_int("TWINR_MAX_AUDIO_FILE_BYTES", 64 * 1024 * 1024)
_SAFE_AUDIO_CONTENT_TYPES = frozenset(
    {
        "audio/aac",
        "audio/flac",
        "audio/m4a",
        "audio/mp3",
        "audio/mp4",
        "audio/mpeg",
        "audio/ogg",
        "audio/opus",
        "audio/wav",
        "audio/webm",
        "audio/x-flac",
        "audio/x-m4a",
        "audio/x-wav",
        "application/ogg",
        "video/mp4",
        "video/ogg",
        "video/quicktime",
        "video/webm",
    }
)
_SUPERVISOR_CONTEXT_SCOPES = frozenset({"tiny_recent", "full_context"})
_TOOL_TURN_STATUSES = frozenset({"completed", "needs_tool_results", "incomplete", "error"})

__all__ = [
    "AgentTextProvider",
    "AgentToolCall",
    "AgentToolResult",
    "AgentToolSpec",
    "CombinedSpeechAgentProvider",
    "CompositeSpeechAgentProvider",
    "ConfigurableProvider",
    "ConversationClosureProvider",
    "ConversationClosureProviderDecision",
    "ConversationLike",
    "FirstWordProvider",
    "FirstWordReply",
    "FoundationModelProvider",
    "ImageInputLike",
    "PathSpeechToTextProvider",
    "PrintFormatter",
    "ProviderBundle",
    "SearchResponse",
    "SpeechToTextProvider",
    "StreamingSpeechEndpointEvent",
    "StreamingSpeechToTextProvider",
    "StreamingSpeechToTextSession",
    "StreamingTranscriptionResult",
    "SupervisorDecision",
    "SupervisorDecisionProvider",
    "TextResponse",
    "TextToSpeechProvider",
    "ToolCallingAgentProvider",
    "ToolCallingTurnResponse",
    "normalize_supervisor_decision_context_scope",
    "normalize_supervisor_decision_runtime_tool_arguments",
    "normalize_supervisor_decision_runtime_tool_name",
    "normalize_tool_schema_sequence",
    "supervisor_decision_requires_full_context",
    "validate_json_schema_subset",
]


def _normalize_required_text(name: str, value: object) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _normalize_optional_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _validate_probability(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    probability = float(value)
    if not math.isfinite(probability):
        raise ValueError(f"{name} must be finite")
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")
    return probability


def _safe_deepcopy(value: Any) -> Any:
    try:
        return deepcopy(value)
    except Exception:
        return value


def _json_dumps(value: JSONValue) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


_JSON_TYPE_ALIASES: dict[str, tuple[type[Any], ...]] = {
    "null": (type(None),),
    "boolean": (bool,),
    "integer": (int,),
    "number": (int, float),
    "string": (str,),
    "array": (list, tuple),
    "object": (dict, Mapping),
}


def _matches_json_type(value: object, expected_type: str) -> bool:
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return ((isinstance(value, int) and not isinstance(value, bool)) or isinstance(value, float)) and math.isfinite(float(value))
    return isinstance(value, _JSON_TYPE_ALIASES.get(expected_type, ()))


# NOTE: This validator intentionally implements a common production subset of JSON Schema
# that maps cleanly to the strict tool/input schemas used by major model APIs.
def validate_json_schema_subset(data: object, schema: Mapping[str, Any], *, path: str = "$", schema_name: str = "schema") -> None:
    """Validate data against a pragmatic JSON-Schema subset.

    Supported keywords: ``type``, ``properties``, ``required``, ``additionalProperties``,
    ``items``, ``enum``, ``minItems``, ``maxItems``, ``minLength``, ``maxLength``,
    ``pattern``, ``minimum``, ``maximum``.
    """

    schema_type = schema.get("type")
    if schema_type is not None:
        allowed_types = (schema_type,) if isinstance(schema_type, str) else tuple(schema_type)
        if not allowed_types:
            raise ValueError(f"{schema_name} at {path} declares an empty type set")
        if not any(_matches_json_type(data, expected_type) for expected_type in allowed_types):
            rendered = "/".join(str(item) for item in allowed_types)
            raise TypeError(f"{path} must match JSON type {rendered}")

    if "enum" in schema and data not in tuple(schema["enum"]):
        raise ValueError(f"{path} must be one of the schema enum values")

    if isinstance(data, str):
        min_length = schema.get("minLength")
        if min_length is not None and len(data) < int(min_length):
            raise ValueError(f"{path} must contain at least {int(min_length)} characters")
        max_length = schema.get("maxLength")
        if max_length is not None and len(data) > int(max_length):
            raise ValueError(f"{path} must contain at most {int(max_length)} characters")
        pattern = schema.get("pattern")
        if pattern is not None and re.search(str(pattern), data) is None:
            raise ValueError(f"{path} must match pattern {pattern!r}")
        return

    if ((isinstance(data, int) and not isinstance(data, bool)) or isinstance(data, float)) and math.isfinite(float(data)):
        minimum = schema.get("minimum")
        if minimum is not None and float(data) < float(minimum):
            raise ValueError(f"{path} must be >= {minimum}")
        maximum = schema.get("maximum")
        if maximum is not None and float(data) > float(maximum):
            raise ValueError(f"{path} must be <= {maximum}")
        return

    if isinstance(data, (list, tuple)):
        min_items = schema.get("minItems")
        if min_items is not None and len(data) < int(min_items):
            raise ValueError(f"{path} must contain at least {int(min_items)} items")
        max_items = schema.get("maxItems")
        if max_items is not None and len(data) > int(max_items):
            raise ValueError(f"{path} must contain at most {int(max_items)} items")
        items_schema = schema.get("items")
        if isinstance(items_schema, Mapping):
            for index, item in enumerate(data):
                validate_json_schema_subset(item, items_schema, path=f"{path}[{index}]", schema_name=schema_name)
        return

    if isinstance(data, Mapping):
        properties = schema.get("properties")
        if properties is not None and not isinstance(properties, Mapping):
            raise TypeError(f"{schema_name} at {path} has non-mapping properties")
        required = tuple(str(item) for item in schema.get("required", ()))
        for key in required:
            if key not in data:
                raise ValueError(f"{path}.{key} is required")
        if isinstance(properties, Mapping):
            for key, subschema in properties.items():
                if key in data:
                    if not isinstance(subschema, Mapping):
                        raise TypeError(f"{schema_name} at {path}.{key} has a non-mapping property schema")
                    validate_json_schema_subset(data[key], subschema, path=f"{path}.{key}", schema_name=schema_name)
        additional_properties = schema.get("additionalProperties", True)
        if additional_properties is False and isinstance(properties, Mapping):
            unknown_keys = set(data).difference(properties)
            if unknown_keys:
                formatted = ", ".join(sorted(str(item) for item in unknown_keys))
                raise ValueError(f"{path} contains unsupported properties: {formatted}")
        elif isinstance(additional_properties, Mapping):
            for key, value in data.items():
                if not isinstance(properties, Mapping) or key not in properties:
                    validate_json_schema_subset(
                        value,
                        additional_properties,
                        path=f"{path}.{key}",
                        schema_name=schema_name,
                    )
        return


def _ensure_json_schema_object(name: str, schema: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(schema, Mapping):
        raise TypeError(f"{name} must be a mapping")
    normalized = _safe_deepcopy(dict(schema))
    schema_type = normalized.get("type")
    if schema_type not in {None, "object"}:
        raise ValueError(f"{name} must describe a JSON object schema")
    return normalized


@dataclass(frozen=True, slots=True)
class AgentToolSpec:
    """Describe one tool using a strict, provider-neutral JSON-Schema contract."""

    name: str
    description: str
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any] | None = None
    strict: bool = True
    requires_confirmation: bool = False
    timeout_seconds: float | None = None
    idempotent: bool = True
    ui_resource_uri: str | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalize_required_text("AgentToolSpec.name", self.name))
        object.__setattr__(self, "description", _normalize_required_text("AgentToolSpec.description", self.description))
        object.__setattr__(self, "input_schema", _ensure_json_schema_object("AgentToolSpec.input_schema", self.input_schema))
        if self.output_schema is not None:
            object.__setattr__(self, "output_schema", _ensure_json_schema_object("AgentToolSpec.output_schema", self.output_schema))
        if self.timeout_seconds is not None:
            timeout_seconds = float(self.timeout_seconds)
            if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
                raise ValueError("AgentToolSpec.timeout_seconds must be a positive finite float")
            object.__setattr__(self, "timeout_seconds", timeout_seconds)
        object.__setattr__(
            self,
            "ui_resource_uri",
            _normalize_optional_text(self.ui_resource_uri),
        )
        object.__setattr__(
            self,
            "tags",
            tuple(tag for tag in (_normalize_optional_text(tag) for tag in self.tags) if tag),
        )

    def validate_arguments(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Validate arguments against the tool's input schema and return a deep-copied dict."""

        if not isinstance(arguments, Mapping):
            raise TypeError("Tool arguments must be a mapping")
        normalized = _safe_deepcopy(dict(arguments))
        validate_json_schema_subset(normalized, self.input_schema, path=f"$.tools.{self.name}", schema_name=self.name)
        return normalized

    def to_neutral_tool_schema(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "input_schema": _safe_deepcopy(dict(self.input_schema)),
            "strict": self.strict,
            "requires_confirmation": self.requires_confirmation,
            "idempotent": self.idempotent,
        }
        if self.output_schema is not None:
            payload["output_schema"] = _safe_deepcopy(dict(self.output_schema))
        if self.timeout_seconds is not None:
            payload["timeout_seconds"] = self.timeout_seconds
        if self.tags:
            payload["tags"] = list(self.tags)
        if self.ui_resource_uri is not None:
            payload["ui_resource_uri"] = self.ui_resource_uri
        return payload

    def to_openai_function_definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": _safe_deepcopy(dict(self.input_schema)),
            "strict": self.strict,
        }

    def to_anthropic_tool_definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": _safe_deepcopy(dict(self.input_schema)),
            "strict": self.strict,
        }

    def to_mcp_tool_definition(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "inputSchema": _safe_deepcopy(dict(self.input_schema)),
        }
        if self.output_schema is not None:
            payload["outputSchema"] = _safe_deepcopy(dict(self.output_schema))
        meta: dict[str, Any] = {}
        if self.ui_resource_uri is not None:
            meta.setdefault("ui", {})["resourceUri"] = self.ui_resource_uri
        if self.tags:
            meta["tags"] = list(self.tags)
        if self.requires_confirmation:
            meta["requiresConfirmation"] = True
        if self.timeout_seconds is not None:
            meta["timeoutSeconds"] = self.timeout_seconds
        if self.idempotent is False:
            meta["idempotent"] = False
        if meta:
            payload["_meta"] = meta
        return payload

    def export(self, dialect: ToolSchemaDialect = "neutral") -> dict[str, Any]:
        if dialect == "neutral":
            return self.to_neutral_tool_schema()
        if dialect == "openai":
            return self.to_openai_function_definition()
        if dialect == "anthropic":
            return self.to_anthropic_tool_definition()
        if dialect == "mcp":
            return self.to_mcp_tool_definition()
        raise ValueError(f"Unsupported tool-schema dialect: {dialect}")


ToolSchemaLike: TypeAlias = Mapping[str, Any] | AgentToolSpec


def normalize_tool_schema_sequence(
    tool_schemas: Sequence[ToolSchemaLike],
    *,
    dialect: ToolSchemaDialect = "neutral",
) -> tuple[dict[str, Any], ...]:
    """Normalize tool schemas into deep-copied dict payloads.

    This keeps existing raw dict call sites working while allowing richer
    ``AgentToolSpec`` objects to flow through the same API surface.
    """

    normalized: list[dict[str, Any]] = []
    for tool_schema in tool_schemas:
        if isinstance(tool_schema, AgentToolSpec):
            normalized.append(tool_schema.export(dialect=dialect))
            continue
        if not isinstance(tool_schema, Mapping):
            raise TypeError("tool_schemas entries must be mappings or AgentToolSpec instances")
        normalized.append(_safe_deepcopy(dict(tool_schema)))
    return tuple(normalized)


@runtime_checkable
class ConfigurableProvider(Protocol):
    """Protocol for provider objects that expose a mutable ``config`` field."""

    config: TwinrConfig


@runtime_checkable
class TextResponse(Protocol):
    """Protocol for text-generation responses with shared metadata fields."""

    text: str
    response_id: str | None
    request_id: str | None
    model: str | None
    token_usage: object | None
    used_web_search: bool


@runtime_checkable
class SearchResponse(Protocol):
    """Protocol for search responses that include an answer and sources."""

    answer: str
    sources: tuple[str, ...]
    response_id: str | None
    request_id: str | None
    model: str | None
    token_usage: object | None
    used_web_search: bool


@dataclass(frozen=True, slots=True)
class AgentToolCall:
    """Represent one requested tool invocation from a model turn."""

    name: str
    call_id: str
    arguments: dict[str, Any]
    raw_arguments: str = "{}"

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalize_required_text("AgentToolCall.name", self.name))
        object.__setattr__(self, "call_id", _normalize_required_text("AgentToolCall.call_id", self.call_id))
        if not isinstance(self.arguments, Mapping):
            raise TypeError("AgentToolCall.arguments must be a mapping")
        normalized_arguments = _safe_deepcopy(dict(self.arguments))
        object.__setattr__(
            self,
            "arguments",
            normalized_arguments,  # AUDIT-FIX(#4): Kapselt Tool-Argumente defensiv, damit externe Mutationen keinen shared mutable state erzeugen.
        )
        raw_arguments = _normalize_optional_text(self.raw_arguments)
        if raw_arguments is None:
            raw_arguments = _json_dumps(normalized_arguments)
        object.__setattr__(self, "raw_arguments", raw_arguments)


@dataclass(frozen=True, slots=True)
class AgentToolResult:
    """Capture the serialized output returned to a tool-calling model."""

    call_id: str
    name: str
    output: Any
    serialized_output: str = ""
    is_error: bool = False
    mime_type: str = "text/plain"
    structured_output: JSONValue | None = None
    metadata: dict[str, JSONValue] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "call_id", _normalize_required_text("AgentToolResult.call_id", self.call_id))
        object.__setattr__(self, "name", _normalize_required_text("AgentToolResult.name", self.name))
        object.__setattr__(self, "output", _safe_deepcopy(self.output))
        object.__setattr__(self, "mime_type", _normalize_required_text("AgentToolResult.mime_type", self.mime_type).lower())
        if self.metadata is not None:
            if not isinstance(self.metadata, Mapping):
                raise TypeError("AgentToolResult.metadata must be a mapping when provided")
            object.__setattr__(self, "metadata", _safe_deepcopy(dict(self.metadata)))
        if self.structured_output is not None:
            _json_dumps(self.structured_output)
        serialized_output = _normalize_optional_text(self.serialized_output)
        if serialized_output is None and self.structured_output is not None:
            serialized_output = _json_dumps(self.structured_output)
        if serialized_output is None:
            serialized_output = str(self.output)
        object.__setattr__(self, "serialized_output", serialized_output)


@dataclass(frozen=True, slots=True)
class ToolCallingTurnResponse:
    """Store one tool-capable model turn and its continuation metadata."""

    text: str
    tool_calls: tuple[AgentToolCall, ...] = ()
    response_id: str | None = None
    request_id: str | None = None
    model: str | None = None
    token_usage: object | None = None
    used_web_search: bool = False
    continuation_token: str | None = None
    turn_status: str | None = None
    provider_turn_state: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        normalized_tool_calls = tuple(self.tool_calls)
        if any(not isinstance(tool_call, AgentToolCall) for tool_call in normalized_tool_calls):
            raise TypeError("ToolCallingTurnResponse.tool_calls must contain AgentToolCall instances")
        normalized_text = str(self.text or "")
        continuation_token = _normalize_optional_text(self.continuation_token)
        if normalized_tool_calls and continuation_token is None:
            # BREAKING: Tool-calling turns without continuation state are now rejected early because
            # continue_turn_streaming cannot resume them correctly.
            raise ValueError(
                "ToolCallingTurnResponse with tool_calls must include continuation_token"
            )
        normalized_status = _normalize_optional_text(self.turn_status)
        if normalized_status is None:
            normalized_status = "needs_tool_results" if normalized_tool_calls else "completed"
        normalized_status = normalized_status.lower()
        if normalized_status not in _TOOL_TURN_STATUSES:
            raise ValueError(f"Unsupported tool turn status: {self.turn_status}")
        if self.provider_turn_state is not None:
            if not isinstance(self.provider_turn_state, Mapping):
                raise TypeError("ToolCallingTurnResponse.provider_turn_state must be a mapping")
            provider_turn_state = _safe_deepcopy(dict(self.provider_turn_state))
        else:
            provider_turn_state = None
        object.__setattr__(self, "text", normalized_text)
        object.__setattr__(
            self,
            "tool_calls",
            normalized_tool_calls,  # AUDIT-FIX(#5): Normalisiert auf Tuple und verhindert Alias-Bugs über mutierbare Sequenzen.
        )
        object.__setattr__(self, "continuation_token", continuation_token)
        object.__setattr__(self, "turn_status", normalized_status)
        object.__setattr__(self, "provider_turn_state", provider_turn_state)


@dataclass(frozen=True, slots=True)
class SupervisorDecision:
    """Represent the supervisor model's decision for streaming orchestration."""

    action: str
    spoken_ack: str | None = None
    spoken_reply: str | None = None
    kind: str | None = None
    goal: str | None = None
    prompt: str | None = None
    allow_web_search: bool | None = None
    location_hint: str | None = None
    date_context: str | None = None
    context_scope: str | None = None
    runtime_tool_name: str | None = None
    runtime_tool_arguments: dict[str, object] | None = None
    response_id: str | None = None
    request_id: str | None = None
    model: str | None = None
    token_usage: object | None = None

    def __post_init__(self) -> None:
        normalized = _normalize_required_text("SupervisorDecision.action", self.action).lower()
        if normalized not in {"direct", "handoff", "end_conversation"}:
            raise ValueError(f"Unsupported supervisor action: {self.action}")
        runtime_tool_name = normalize_supervisor_decision_runtime_tool_name(
            self.runtime_tool_name,
            strict=self.runtime_tool_arguments is not None,
        )
        runtime_tool_arguments = normalize_supervisor_decision_runtime_tool_arguments(
            self.runtime_tool_arguments,
            strict=self.runtime_tool_arguments is not None,
        )
        if runtime_tool_arguments is not None and runtime_tool_name is None:
            raise ValueError("SupervisorDecision.runtime_tool_arguments requires runtime_tool_name")
        object.__setattr__(self, "action", normalized)
        object.__setattr__(self, "spoken_ack", _normalize_optional_text(self.spoken_ack))
        object.__setattr__(self, "spoken_reply", _normalize_optional_text(self.spoken_reply))
        object.__setattr__(self, "kind", _normalize_optional_text(self.kind))
        object.__setattr__(self, "goal", _normalize_optional_text(self.goal))
        object.__setattr__(self, "prompt", _normalize_optional_text(self.prompt))
        object.__setattr__(self, "location_hint", _normalize_optional_text(self.location_hint))
        object.__setattr__(self, "date_context", _normalize_optional_text(self.date_context))
        object.__setattr__(
            self,
            "context_scope",
            normalize_supervisor_decision_context_scope(self.context_scope, strict=self.context_scope is not None),
        )
        object.__setattr__(self, "runtime_tool_name", runtime_tool_name)
        object.__setattr__(self, "runtime_tool_arguments", runtime_tool_arguments)


def normalize_supervisor_decision_context_scope(
    value: object,
    *,
    strict: bool = False,
) -> str | None:
    """Return the validated supervisor decision context scope, if present."""

    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in _SUPERVISOR_CONTEXT_SCOPES:
        return normalized
    if strict:
        raise ValueError(f"Unsupported supervisor context scope: {value!r}")
    return None


def normalize_supervisor_decision_runtime_tool_name(
    value: object,
    *,
    strict: bool = False,
) -> str | None:
    """Return one stripped runtime-local tool name, if present."""

    if value is None:
        return None
    normalized = str(value).strip()
    if normalized:
        return normalized
    if strict:
        raise ValueError("SupervisorDecision.runtime_tool_name must not be empty")
    return None


def normalize_supervisor_decision_runtime_tool_arguments(
    value: object,
    *,
    strict: bool = False,
) -> dict[str, object] | None:
    """Return JSON-like runtime-local tool arguments when the payload is mapping-like."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        if strict:
            raise TypeError("SupervisorDecision.runtime_tool_arguments must be a mapping")
        return None
    normalized: dict[str, object] = {}
    for key, item in value.items():
        key_text = str(key).strip()
        if not key_text:
            if strict:
                raise ValueError("SupervisorDecision.runtime_tool_arguments contains an empty key")
            continue
        normalized[key_text] = _safe_deepcopy(item)
    return normalized


def supervisor_decision_requires_full_context(decision: object | None) -> bool:
    """Return whether a supervisor decision says the fast lane lacks context."""

    if decision is None:
        return False
    return normalize_supervisor_decision_context_scope(
        getattr(decision, "context_scope", None)
    ) == "full_context"


@dataclass(frozen=True, slots=True)
class FirstWordReply:
    """Store the first-word lane reply chosen for low-latency speech output."""

    mode: str
    spoken_text: str
    response_id: str | None = None
    request_id: str | None = None
    model: str | None = None
    token_usage: object | None = None

    def __post_init__(self) -> None:
        normalized_mode = _normalize_required_text("FirstWordReply.mode", self.mode).lower()
        if normalized_mode not in {"direct", "filler"}:
            raise ValueError(f"Unsupported first-word mode: {self.mode}")
        text = _normalize_required_text("FirstWordReply.spoken_text", self.spoken_text)
        object.__setattr__(self, "mode", normalized_mode)
        object.__setattr__(self, "spoken_text", text)


@dataclass(frozen=True, slots=True)
class ConversationClosureProviderDecision:
    """Store one structured closure-decision response from a provider.

    Attributes:
        close_now: Whether Twinr should stop automatic follow-up listening
            after the just-finished exchange.
        confidence: Normalized confidence score in the range ``0.0`` to
            ``1.0``.
        reason: Short provider-supplied reason string.
        matched_topics: Up to two matched steering-topic titles echoed from the
            current turn context.
        response_id: Provider response identifier when available.
        request_id: Transport request identifier when available.
        model: Provider model identifier when available.
        token_usage: Provider token-usage metadata when available.
    """

    close_now: bool
    confidence: float
    reason: str
    matched_topics: tuple[str, ...] = ()
    response_id: str | None = None
    request_id: str | None = None
    model: str | None = None
    token_usage: object | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "confidence", _validate_probability("ConversationClosureProviderDecision.confidence", self.confidence))
        object.__setattr__(self, "reason", _normalize_required_text("ConversationClosureProviderDecision.reason", self.reason))
        normalized_topics = tuple(
            topic
            for topic in (_normalize_optional_text(topic) for topic in self.matched_topics)
            if topic
        )
        if len(normalized_topics) > 2:
            raise ValueError("ConversationClosureProviderDecision.matched_topics may contain at most two entries")
        object.__setattr__(self, "matched_topics", normalized_topics)


@dataclass(frozen=True, slots=True)
class StreamingTranscriptionResult:
    """Store the current streaming transcription snapshot for a session."""

    transcript: str
    request_id: str | None = None
    saw_interim: bool = False
    saw_speech_final: bool = False
    saw_utterance_end: bool = False
    confidence: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "transcript", str(self.transcript or ""))
        object.__setattr__(self, "confidence", _validate_probability("StreamingTranscriptionResult.confidence", self.confidence))


@dataclass(frozen=True, slots=True)
class StreamingSpeechEndpointEvent:
    """Represent a streaming endpoint signal emitted by an STT backend."""

    transcript: str
    event_type: str
    request_id: str | None = None
    is_final: bool = False
    speech_final: bool = False
    from_finalize: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "transcript", str(self.transcript or ""))
        object.__setattr__(self, "event_type", _normalize_required_text("StreamingSpeechEndpointEvent.event_type", self.event_type))


@runtime_checkable
class SpeechToTextProvider(ConfigurableProvider, Protocol):
    """Protocol for providers that transcribe complete audio payloads."""

    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
        language: str | None = None,
        prompt: str | None = None,
    ) -> str:
        ...


@runtime_checkable
class PathSpeechToTextProvider(SpeechToTextProvider, Protocol):
    """Protocol for STT providers that can transcribe directly from a path."""

    def transcribe_path(
        self,
        path: str | Path,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> str:
        ...


@runtime_checkable
class StreamingSpeechToTextSession(Protocol):
    """Protocol for a live streaming transcription session instance."""

    def send_pcm(self, pcm_bytes: bytes) -> None:
        ...

    def snapshot(self) -> StreamingTranscriptionResult:
        ...

    def finalize(self) -> StreamingTranscriptionResult:
        ...

    def close(self) -> None:
        ...


@runtime_checkable
class StreamingSpeechToTextProvider(PathSpeechToTextProvider, Protocol):
    """Protocol for providers that open live streaming STT sessions."""

    def start_streaming_session(
        self,
        *,
        sample_rate: int,
        channels: int,
        language: str | None = None,
        prompt: str | None = None,
        on_interim: Callable[[str], None] | None = None,
        on_endpoint: Callable[[StreamingSpeechEndpointEvent], None] | None = None,
    ) -> StreamingSpeechToTextSession:
        ...


@runtime_checkable
class AgentTextProvider(ConfigurableProvider, Protocol):
    """Protocol for text-generation providers used by the base agent."""

    def respond_streaming(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> TextResponse:
        ...

    def respond_with_metadata(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
    ) -> TextResponse:
        ...

    def respond_to_images_with_metadata(
        self,
        prompt: str,
        *,
        images: Sequence[ImageInputLike],
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
    ) -> TextResponse:
        ...

    def search_live_info_with_metadata(
        self,
        question: str,
        *,
        conversation: ConversationLike | None = None,
        location_hint: str | None = None,
        date_context: str | None = None,
    ) -> SearchResponse:
        ...

    def compose_print_job_with_metadata(
        self,
        *,
        conversation: ConversationLike | None = None,
        focus_hint: str | None = None,
        direct_text: str | None = None,
        request_source: str = "button",
    ) -> TextResponse:
        ...

    def phrase_due_reminder_with_metadata(
        self,
        reminder: object,
        *,
        now: datetime | None = None,
    ) -> TextResponse:
        ...

    def phrase_proactive_prompt_with_metadata(
        self,
        *,
        trigger_id: str,
        reason: str,
        default_prompt: str,
        priority: int,
        conversation: ConversationLike | None = None,
        recent_prompts: tuple[str, ...] = (),
        observation_facts: tuple[str, ...] = (),
    ) -> TextResponse:
        ...

    def fulfill_automation_prompt_with_metadata(
        self,
        prompt: str,
        *,
        allow_web_search: bool,
        delivery: str = "spoken",
    ) -> TextResponse:
        ...


@runtime_checkable
class ToolCallingAgentProvider(ConfigurableProvider, Protocol):
    """Protocol for agents that can emit tool calls and continue turns."""

    def start_turn_streaming(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        tool_schemas: Sequence[ToolSchemaLike] = (),
        allow_web_search: bool | None = None,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> ToolCallingTurnResponse:
        ...

    def continue_turn_streaming(
        self,
        *,
        continuation_token: str,
        tool_results: Sequence[AgentToolResult],
        instructions: str | None = None,
        tool_schemas: Sequence[ToolSchemaLike] = (),
        allow_web_search: bool | None = None,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> ToolCallingTurnResponse:
        ...


@runtime_checkable
class SupervisorDecisionProvider(ConfigurableProvider, Protocol):
    """Protocol for providers that return a ``SupervisorDecision``."""

    def decide(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
    ) -> SupervisorDecision:
        ...


@runtime_checkable
class FirstWordProvider(ConfigurableProvider, Protocol):
    """Protocol for providers that produce a bounded first-word reply."""

    def reply(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
    ) -> FirstWordReply:
        ...


@runtime_checkable
class ConversationClosureProvider(ConfigurableProvider, Protocol):
    """Protocol for providers that return one closure-decision object."""

    def decide(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        timeout_seconds: float | None = None,
    ) -> ConversationClosureProviderDecision:
        ...


@runtime_checkable
class TextToSpeechProvider(ConfigurableProvider, Protocol):
    """Protocol for providers that synthesize spoken output."""

    def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        response_format: str | None = None,
        instructions: str | None = None,
    ) -> bytes:
        ...

    def synthesize_stream(
        self,
        text: str,
        *,
        voice: str | None = None,
        response_format: str | None = None,
        instructions: str | None = None,
        chunk_size: int = 4096,
    ) -> Iterator[bytes]:
        ...


@runtime_checkable
class CombinedSpeechAgentProvider(
    SpeechToTextProvider,
    AgentTextProvider,
    TextToSpeechProvider,
    Protocol,
):
    """Protocol for a single provider implementing STT, text, and TTS."""

    pass


def _ensure_runtime_protocol(name: str, provider: object, protocol: type[Any]) -> None:
    """Raise early when a configured provider misses a required protocol."""

    # AUDIT-FIX(#2): Erzwingt Fail-Fast-Validierung der injizierten Provider gegen die Runtime-Protokolle.
    if not isinstance(provider, protocol):
        raise TypeError(
            f"{name} must implement {protocol.__name__}; got {type(provider).__name__}"
        )


def _unique_configurable_providers(
    providers: Sequence[ConfigurableProvider],
) -> tuple[ConfigurableProvider, ...]:
    """Deduplicate provider objects while preserving their original order."""

    unique_providers: list[ConfigurableProvider] = []
    seen: set[int] = set()
    for provider in providers:
        provider_id = id(provider)
        if provider_id in seen:
            continue
        seen.add(provider_id)
        unique_providers.append(provider)
    return tuple(unique_providers)


def _snapshot_config(value: TwinrConfig) -> TwinrConfig:
    return _safe_deepcopy(value)


def _apply_config_atomically(
    providers: Sequence[ConfigurableProvider],
    value: TwinrConfig,
) -> None:
    """Apply one config object across providers with best-effort rollback."""

    # AUDIT-FIX(#3): Wendet Konfigurationsänderungen rollback-fähig an, damit der Composite nicht halb umkonfiguriert zurückbleibt.
    previous_configs: list[tuple[ConfigurableProvider, TwinrConfig]] = []
    try:
        for provider in _unique_configurable_providers(providers):
            previous_configs.append((provider, _snapshot_config(provider.config)))
            provider.config = value
    except Exception as exc:
        for provider, previous_value in reversed(previous_configs):
            try:
                provider.config = previous_value
            except Exception:
                _LOGGER.warning(
                    "Failed to roll back provider config for %s after composite apply_config failure.",
                    type(provider).__name__,
                    exc_info=True,
                )
        raise RuntimeError("Failed to apply config consistently across providers") from exc


def _guess_audio_content_type(path: str | Path) -> str:
    """Infer an audio content type from a file path with a WAV fallback."""

    guessed_type, _ = mimetypes.guess_type(str(path), strict=False)
    if guessed_type is None:
        return "audio/wav"
    normalized = guessed_type.lower()
    if normalized.startswith("audio/") or normalized in _SAFE_AUDIO_CONTENT_TYPES:
        return normalized
    return "audio/wav"


def _open_audio_file_safely(path: str | Path) -> io.BufferedReader:
    """Open an audio file defensively and reject unsafe path targets.

    # BREAKING: Symlinks and non-regular files are now rejected to prevent
    # accidental secret-file reads and device-file hangs in production deployments.
    """

    audio_path = Path(path)
    open_flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        open_flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        open_flags |= os.O_NOFOLLOW

    try:
        fd = os.open(os.fspath(audio_path), open_flags)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Audio file not found: {audio_path}") from exc
    except IsADirectoryError as exc:
        raise IsADirectoryError(f"Audio path is not a file: {audio_path}") from exc
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise PermissionError(f"Refusing to follow symlink for audio path: {audio_path}") from exc
        raise

    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError(f"Audio path is not a regular file: {audio_path}")
        if file_stat.st_size > DEFAULT_MAX_AUDIO_FILE_BYTES:
            raise ValueError(
                f"Audio file exceeds the configured limit of {DEFAULT_MAX_AUDIO_FILE_BYTES} bytes: {audio_path}"
            )
        return io.open(fd, "rb", closefd=True)
    except Exception:
        os.close(fd)
        raise


def _validate_audio_path_for_read(path: str | Path) -> Path:
    """Validate an audio path and return it as a ``Path`` instance."""

    audio_path = Path(path)
    with _open_audio_file_safely(audio_path):
        pass
    return audio_path


def _read_audio_bytes_from_path(path: str | Path) -> bytes:
    """Read audio bytes from a path with bounded, regular-file-only semantics."""

    # AUDIT-FIX(#1): Liest den Pfad in einem Schritt ein und vermeidet damit blindes Attributrouting plus unnötige Check-then-use-Rennen.
    audio_path = Path(path)
    try:
        with _open_audio_file_safely(audio_path) as file_obj:
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = file_obj.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > DEFAULT_MAX_AUDIO_FILE_BYTES:
                    raise ValueError(
                        f"Audio file exceeds the configured limit of {DEFAULT_MAX_AUDIO_FILE_BYTES} bytes: {audio_path}"
                    )
                chunks.append(chunk)
            return b"".join(chunks)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Audio file not found: {audio_path}") from exc
    except IsADirectoryError as exc:
        raise IsADirectoryError(f"Audio path is not a file: {audio_path}") from exc


@dataclass
class ProviderBundle:
    """Bundle the base agent's primary providers into one validated object."""

    stt: SpeechToTextProvider
    agent: AgentTextProvider
    tts: TextToSpeechProvider
    tool_agent: ToolCallingAgentProvider | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        _ensure_runtime_protocol("stt", self.stt, SpeechToTextProvider)  # AUDIT-FIX(#2): Validiert Provider bereits beim Bundling statt erst beim Live-Aufruf.
        _ensure_runtime_protocol("agent", self.agent, AgentTextProvider)  # AUDIT-FIX(#2): Verhindert späte Laufzeitcrashs durch fehlerhafte DI.
        _ensure_runtime_protocol("tts", self.tts, TextToSpeechProvider)  # AUDIT-FIX(#2): Verhindert späte Laufzeitcrashs durch fehlerhafte DI.
        if self.tool_agent is not None:
            _ensure_runtime_protocol("tool_agent", self.tool_agent, ToolCallingAgentProvider)  # AUDIT-FIX(#2): Optionaler Tool-Agent wird identisch fail-fast validiert.


class CompositeSpeechAgentProvider:
    """Delegate STT, text, and TTS calls to separate coordinated providers.

    The composite exposes one provider-like surface while keeping the three
    injected providers' configs synchronized and offering a path-based STT
    fallback when the selected STT provider only accepts raw bytes.
    """

    def __init__(
        self,
        *,
        stt: SpeechToTextProvider,
        agent: AgentTextProvider,
        tts: TextToSpeechProvider,
    ) -> None:
        _ensure_runtime_protocol("stt", stt, SpeechToTextProvider)  # AUDIT-FIX(#2): Stoppt Fehlverkabelung beim Systemstart statt beim ersten Senior-Request.
        _ensure_runtime_protocol("agent", agent, AgentTextProvider)  # AUDIT-FIX(#2): Stoppt Fehlverkabelung beim Systemstart statt beim ersten Senior-Request.
        _ensure_runtime_protocol("tts", tts, TextToSpeechProvider)  # AUDIT-FIX(#2): Stoppt Fehlverkabelung beim Systemstart statt beim ersten Senior-Request.
        self._stt = stt
        self._agent = agent
        self._tts = tts
        self._providers = _unique_configurable_providers((stt, agent, tts))
        self._config_lock = threading.RLock()
        self.config = agent.config  # AUDIT-FIX(#3): Synchronisiert die Initialkonfiguration über alle Provider, damit der Composite konsistent startet.

    @property
    def config(self) -> TwinrConfig:
        with self._config_lock:
            return self._agent.config

    @config.setter
    def config(self, value: TwinrConfig) -> None:
        with self._config_lock:
            _apply_config_atomically(self._providers, value)  # AUDIT-FIX(#3): Verhindert partielle Provider-Updates bei Setter-Fehlern.

    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
        language: str | None = None,
        prompt: str | None = None,
    ) -> str:
        return self._stt.transcribe(
            audio_bytes,
            filename=filename,
            content_type=content_type,
            language=language,
            prompt=prompt,
        )

    def transcribe_path(
        self,
        path: str | Path,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> str:
        audio_path = Path(path)
        if isinstance(self._stt, PathSpeechToTextProvider):
            validated_path = _validate_audio_path_for_read(audio_path)
            return self._stt.transcribe_path(validated_path, language=language, prompt=prompt)

        audio_bytes = _read_audio_bytes_from_path(audio_path)
        return self._stt.transcribe(
            audio_bytes,
            filename=audio_path.name,
            content_type=_guess_audio_content_type(audio_path),  # AUDIT-FIX(#1): Fallback auf byte-basierte STT-Transkription statt blindem Attributzugriff.
            language=language,
            prompt=prompt,
        )

    def start_streaming_session(
        self,
        *,
        sample_rate: int,
        channels: int,
        language: str | None = None,
        prompt: str | None = None,
        on_interim: Callable[[str], None] | None = None,
        on_endpoint: Callable[[StreamingSpeechEndpointEvent], None] | None = None,
    ) -> StreamingSpeechToTextSession:
        if not isinstance(self._stt, StreamingSpeechToTextProvider):
            raise NotImplementedError(
                f"{type(self._stt).__name__} does not support streaming STT sessions"
            )
        return self._stt.start_streaming_session(
            sample_rate=sample_rate,
            channels=channels,
            language=language,
            prompt=prompt,
            on_interim=on_interim,
            on_endpoint=on_endpoint,
        )

    def respond_streaming(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> TextResponse:
        return self._agent.respond_streaming(
            prompt,
            conversation=conversation,
            instructions=instructions,
            allow_web_search=allow_web_search,
            on_text_delta=on_text_delta,
        )

    def respond_with_metadata(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
    ) -> TextResponse:
        return self._agent.respond_with_metadata(
            prompt,
            conversation=conversation,
            instructions=instructions,
            allow_web_search=allow_web_search,
        )

    def respond_to_images_with_metadata(
        self,
        prompt: str,
        *,
        images: Sequence[ImageInputLike],
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
    ) -> TextResponse:
        return self._agent.respond_to_images_with_metadata(
            prompt,
            images=images,
            conversation=conversation,
            instructions=instructions,
            allow_web_search=allow_web_search,
        )

    def search_live_info_with_metadata(
        self,
        question: str,
        *,
        conversation: ConversationLike | None = None,
        location_hint: str | None = None,
        date_context: str | None = None,
    ) -> SearchResponse:
        return self._agent.search_live_info_with_metadata(
            question,
            conversation=conversation,
            location_hint=location_hint,
            date_context=date_context,
        )

    def compose_print_job_with_metadata(
        self,
        *,
        conversation: ConversationLike | None = None,
        focus_hint: str | None = None,
        direct_text: str | None = None,
        request_source: str = "button",
    ) -> TextResponse:
        return self._agent.compose_print_job_with_metadata(
            conversation=conversation,
            focus_hint=focus_hint,
            direct_text=direct_text,
            request_source=request_source,
        )

    def phrase_due_reminder_with_metadata(
        self,
        reminder: object,
        *,
        now: datetime | None = None,
    ) -> TextResponse:
        return self._agent.phrase_due_reminder_with_metadata(reminder, now=now)

    def phrase_proactive_prompt_with_metadata(
        self,
        *,
        trigger_id: str,
        reason: str,
        default_prompt: str,
        priority: int,
        conversation: ConversationLike | None = None,
        recent_prompts: tuple[str, ...] = (),
        observation_facts: tuple[str, ...] = (),
    ) -> TextResponse:
        return self._agent.phrase_proactive_prompt_with_metadata(
            trigger_id=trigger_id,
            reason=reason,
            default_prompt=default_prompt,
            priority=priority,
            conversation=conversation,
            recent_prompts=recent_prompts,
            observation_facts=observation_facts,
        )

    def fulfill_automation_prompt_with_metadata(
        self,
        prompt: str,
        *,
        allow_web_search: bool,
        delivery: str = "spoken",
    ) -> TextResponse:
        return self._agent.fulfill_automation_prompt_with_metadata(
            prompt,
            allow_web_search=allow_web_search,
            delivery=delivery,
        )

    def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        response_format: str | None = None,
        instructions: str | None = None,
    ) -> bytes:
        return self._tts.synthesize(
            text,
            voice=voice,
            response_format=response_format,
            instructions=instructions,
        )

    def synthesize_stream(
        self,
        text: str,
        *,
        voice: str | None = None,
        response_format: str | None = None,
        instructions: str | None = None,
        chunk_size: int = 4096,
    ) -> Iterator[bytes]:
        return self._tts.synthesize_stream(
            text,
            voice=voice,
            response_format=response_format,
            instructions=instructions,
            chunk_size=chunk_size,
        )


class FoundationModelProvider(Protocol):
    """Protocol for simple prompt-in, text-out model adapters."""

    def respond(self, prompt: str) -> str:
        ...


class PrintFormatter(Protocol):
    """Protocol for formatters that adapt text to the printer surface."""

    def format_for_print(self, text: str) -> str:
        ...