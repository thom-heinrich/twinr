# CHANGELOG: 2026-03-30
# BUG-1: Replaced the old top-level schema keyword stripping with recursive strict-schema normalization that preserves supported 2026 OpenAI JSON Schema features (for example enum and nested anyOf) and removes only documented unsupported keywords.
# BUG-2: Conversation-closure output is now fully validated, including follow_up_action enum and confidence range, so malformed structured output cannot silently flip listening behavior.
# BUG-3: Supervisor runtime_tool_arguments_json is no longer silently discarded on invalid JSON; malformed runtime tool payloads now fail fast instead of producing incorrect default actions.
# SEC-1: Tool-loop requests no longer hardcode store=True; they honor config/env policy, add encrypted reasoning includes for stateless mode, and keep the privacy decision in deployment control.
# SEC-2: Function-call argument payloads are capped before JSON parsing to avoid memory spikes and denial-of-service behavior on Pi-class devices.
# IMP-1: Function-tool schemas are normalized for strict mode recursively, namespace tools are supported, and tool_search is auto-enabled for deferred tools on GPT-5.4+ models.
# IMP-2: Structured micro-decisions now use low verbosity by default and the adapter can opt into server-side compaction via context_management for long-running chains.
# IMP-3: # BREAKING: This adapter defaults parallel tool calls to false for application tools unless config/env explicitly enables them, favoring safer single-action turns for a physical assistant runtime.

"""Expose Twinr-facing OpenAI adapters and provider bundle assembly.

This module wraps the shared ``OpenAIBackend`` in contract-specific adapters
for speech-to-text, text responses, text-to-speech, tool-calling,
supervisor-routing, and first-word fallback flows. Import these adapters from
``twinr.providers.openai`` or this package root instead of reaching into the
lower-level ``core`` or ``capabilities`` packages.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import copy
import json
import logging
import os
import re

from twinr.agent.base_agent.contracts import (
    AgentToolCall,
    AgentToolResult,
    CompositeSpeechAgentProvider,
    ConversationClosureProviderDecision,
    ConversationLike,
    FirstWordReply,
    ProviderBundle,
    SearchResponse,
    SupervisorDecision,
    TextResponse,
    ToolCallingTurnResponse,
)
from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.conversation.language import user_response_language_instruction
from twinr.agent.base_agent.prompting.personality import merge_instructions
from twinr.ops.usage import extract_model_name, extract_token_usage
from twinr.text_utils import extract_json_object

from .backend import OpenAIBackend
from ..core.types import OpenAIImageInput


logger = logging.getLogger(__name__)
_O_SERIES_MODEL_PATTERN = re.compile(r"^o\d+(?:[-_.].*)?$")
_GPT_VERSION_PATTERN = re.compile(r"^gpt-(\d+)(?:\.(\d+))?(?:[-_.].*)?$")
_DEFAULT_TOOL_ARGUMENTS_MAX_BYTES = 262_144
_SUPPORTED_TEXT_VERBOSITIES = {"low", "medium", "high"}
_STRICT_SCHEMA_UNSUPPORTED_KEYWORDS = {
    "allOf",
    "not",
    "dependentRequired",
    "dependentSchemas",
    "if",
    "then",
    "else",
}
_UNSUPPORTED_OUTPUT_ITEM_TYPES = {
    "custom_tool_call": "Twinr's AgentToolCall contract only supports JSON-object function calls; custom tools return free-form text inputs and need a different executor contract.",
    "mcp_approval_request": "MCP approval requests need a dedicated approval-handling path and are not compatible with AgentToolCall.",
}

_CONVERSATION_CLOSURE_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "close_now": {
            "type": "boolean",
            "description": "Whether Twinr should suppress automatic follow-up listening after the just-finished exchange.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Confidence score for the closure decision.",
        },
        "reason": {
            "type": "string",
            "description": "Short canonical English reason code or phrase.",
        },
        "follow_up_action": {
            "type": "string",
            "enum": ["continue", "end"],
            "description": "continue when the assistant reply still expects immediate user input right now; end when Twinr should return to waiting after the answer.",
        },
        "matched_topics": {
            "type": "array",
            "description": "Up to two matched topic titles echoed from the provided steering context.",
            "items": {"type": "string"},
        },
    },
    "required": ["close_now", "confidence", "reason", "follow_up_action", "matched_topics"],
    "additionalProperties": False,
}

_SUPERVISOR_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["direct", "handoff", "end_conversation"],
            "description": (
                "direct for a short immediate answer, handoff for specialist work, end_conversation to stop for now. "
                "Runtime-local state mutations or checks such as bounded voice-quiet set/status/clear must use handoff, not direct."
            ),
        },
        "spoken_ack": {
            "type": ["string", "null"],
            "description": "Optional short immediate acknowledgement for handoff. Use null when no bridge line should be spoken. Must stay null for direct replies.",
        },
        "spoken_reply": {
            "type": ["string", "null"],
            "description": "Full short user-facing answer for direct or end_conversation. Must stay null for handoff.",
        },
        "kind": {
            "type": ["string", "null"],
            "enum": ["general", "search", "memory", "automation", None],
            "description": "Short handoff category. Null unless action is handoff.",
        },
        "goal": {
            "type": ["string", "null"],
            "description": "Short specialist goal. Null unless action is handoff.",
        },
        "prompt": {
            "type": ["string", "null"],
            "description": (
                "Optional clean rewritten task or search query for the specialist. "
                "Use null to reuse the original user wording."
            ),
        },
        "allow_web_search": {
            "type": ["boolean", "null"],
            "description": "True only when the specialist may use live web search.",
        },
        "location_hint": {
            "type": ["string", "null"],
            "description": (
                "Explicit place already named by the user for this turn. "
                "Use null only when the turn did not name a concrete place."
            ),
        },
        "date_context": {
            "type": ["string", "null"],
            "description": (
                "Explicit absolute or resolved local date context for the turn. "
                "Use null when no date anchor is needed."
            ),
        },
        "context_scope": {
            "type": ["string", "null"],
            "enum": ["tiny_recent", "full_context", None],
            "description": (
                "tiny_recent when the downstream runtime-local specialist can finish safely from the tiny recent context plus live runtime state alone, including bounded voice-quiet or current listening-state work. "
                "full_context when the answer depends on broader memory or richer provider context."
            ),
        },
        "runtime_tool_name": {
            "type": ["string", "null"],
            "description": (
                "Optional exact Twinr runtime tool name for a one-shot tiny_recent handoff that can be executed directly without a second specialist model hop, "
                "for example manage_voice_quiet_mode for bounded quiet/status/resume control."
            ),
        },
        "runtime_tool_arguments_json": {
            "type": ["string", "null"],
            "description": (
                "Optional compact JSON object string with the concrete arguments for runtime_tool_name. "
                "Use null unless the handoff can be satisfied by one direct runtime-local tool call."
            ),
        },
    },
    "required": [
        "action",
        "spoken_ack",
        "spoken_reply",
        "kind",
        "goal",
        "prompt",
        "allow_web_search",
        "location_hint",
        "date_context",
        "context_scope",
        "runtime_tool_name",
        "runtime_tool_arguments_json",
    ],
    "additionalProperties": False,
}

_FIRST_WORD_REPLY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "mode": {
            "type": "string",
            "enum": ["direct", "filler"],
            "description": "direct for a tiny safe answer, filler for a tiny provisional progress line.",
        },
        "spoken_text": {
            "type": "string",
            "description": "One short user-facing spoken line.",
        },
    },
    "required": ["mode", "spoken_text"],
    "additionalProperties": False,
}

_FIRST_WORD_MODEL_FALLBACKS: tuple[str, ...] = ()


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _coerce_mapping(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dict(dumped)
    return None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _coerce_topic_titles(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    normalized_topics: list[str] = []
    seen: set[str] = set()
    for raw_topic in value:
        topic = _optional_text(raw_topic)
        if not topic:
            continue
        key = topic.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized_topics.append(topic)
        if len(normalized_topics) >= 2:
            break
    return tuple(normalized_topics)


def _validated_choice(value: Any, *, allowed: Sequence[str], default: str, context: str) -> str:
    candidate = str(value or default).strip() or default
    if candidate in allowed:
        return candidate
    raise RuntimeError(f"{context} must be one of {tuple(allowed)!r}, got {candidate!r}")


def _validated_probability(value: Any, *, context: str) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{context} must be a float in [0, 1]") from exc
    if 0.0 <= probability <= 1.0:
        return probability
    raise RuntimeError(f"{context} must be in [0, 1], got {probability!r}")


def _validated_optional_choice(value: Any, *, allowed: Sequence[str], context: str) -> str | None:
    candidate = _optional_text(value)
    if candidate is None:
        return None
    if candidate in allowed:
        return candidate
    raise RuntimeError(f"{context} must be one of {tuple(allowed)!r}, got {candidate!r}")


def _validated_bool(value: Any, *, context: str) -> bool:
    parsed = _optional_bool(value)
    if parsed is not None:
        return parsed
    raise RuntimeError(f"{context} must be boolean, got {value!r}")


def _optional_json_object(value: Any) -> dict[str, object] | None:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items() if str(key).strip()}
    payload = _optional_text(value)
    if payload is None:
        return None
    try:
        parsed = json.loads(payload)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, dict):
        return None
    return {str(key): item for key, item in parsed.items() if str(key).strip()}


def _validated_optional_json_object(value: Any, *, context: str) -> dict[str, object] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items() if str(key).strip()}
    payload = _optional_text(value)
    if payload is None:
        return None
    try:
        parsed = json.loads(payload)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{context} must be a compact JSON object string") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{context} must decode to a JSON object")
    return {str(key): item for key, item in parsed.items() if str(key).strip()}


def _load_json_object(text: str, *, context: str) -> dict[str, Any]:
    payload_text = text.strip()
    if not payload_text:
        raise RuntimeError(f"{context} returned empty structured output")
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{context} returned invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{context} must decode to a JSON object")
    return payload


def _extract_detail_message(detail: Any) -> str | None:
    if detail is None:
        return None
    if isinstance(detail, dict):
        code = _optional_text(detail.get("code"))
        message = _optional_text(detail.get("message"))
        reason = _optional_text(detail.get("reason"))
        parts = [part for part in (code, reason, message) if part]
        return ": ".join(parts) if parts else _optional_text(detail)
    code = _optional_text(getattr(detail, "code", None))
    message = _optional_text(getattr(detail, "message", None))
    reason = _optional_text(getattr(detail, "reason", None))
    parts = [part for part in (code, reason, message) if part]
    if parts:
        return ": ".join(parts)
    return _optional_text(detail)


def _validate_response_status(response: Any, *, context: str) -> None:
    status = str(getattr(response, "status", "") or "").strip().lower()
    if not status or status == "completed":
        return
    detail_parts = [f"{context} finished with status={status!r}"]
    error_detail = _extract_detail_message(getattr(response, "error", None))
    if error_detail:
        detail_parts.append(f"error={error_detail}")
    incomplete_detail = _extract_detail_message(getattr(response, "incomplete_details", None))
    if incomplete_detail:
        detail_parts.append(f"incomplete={incomplete_detail}")
    raise RuntimeError("; ".join(detail_parts))


def _should_retry_incomplete_max_output_tokens(response: Any, *, request: dict[str, Any]) -> bool:
    if "max_output_tokens" not in request:
        return False
    status = str(getattr(response, "status", "") or "").strip().lower()
    if status != "incomplete":
        return False
    incomplete_detail = _extract_detail_message(getattr(response, "incomplete_details", None)) or ""
    return "max_output_tokens" in incomplete_detail.lower()


def _expanded_max_output_tokens(value: Any) -> int:
    try:
        current = max(16, int(value))
    except (TypeError, ValueError):
        current = 64
    return min(512, max(current + 64, current * 2))


def _iter_retry_max_output_tokens(
    response: Any,
    *,
    request: dict[str, Any],
    retry_max_output_tokens: Sequence[int] | None,
) -> tuple[int, ...]:
    if not _should_retry_incomplete_max_output_tokens(response, request=request):
        return ()
    if retry_max_output_tokens is None:
        return (_expanded_max_output_tokens(request.get("max_output_tokens")),)
    try:
        current_budget = max(16, int(request.get("max_output_tokens")))
    except (TypeError, ValueError):
        current_budget = 16
    normalized: list[int] = []
    seen: set[int] = set()
    for raw_budget in retry_max_output_tokens:
        try:
            budget = max(16, int(raw_budget))
        except (TypeError, ValueError):
            continue
        if budget <= current_budget or budget in seen:
            continue
        seen.add(budget)
        normalized.append(budget)
    return tuple(normalized)


def _emit_text_delta(on_text_delta: Callable[[str], None] | None, text: str, *, context: str) -> None:
    if on_text_delta is None or not text:
        return
    try:
        on_text_delta(text)
    except Exception:
        logger.warning(
            "%s callback failed; continuing without streaming callback delivery",
            context,
            exc_info=True,
        )


def _is_reasoning_unsupported_error(exc: Exception) -> bool:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", {})
        message = str(error.get("message", "")).lower()
        param = str(error.get("param", "")).lower()
        code = str(error.get("code", "")).lower()
        if "reasoning" in param:
            return True
        if "reasoning" in message and ("not supported" in message or "unsupported" in message):
            return True
        if code == "unsupported_parameter" and "reasoning" in message:
            return True
    message = str(exc).lower()
    return "reasoning" in message and ("not supported" in message or "unsupported" in message)


def _parse_bool_setting(value: Any) -> bool | None:
    return _optional_bool(value)


def _config_value(config: TwinrConfig | None, *names: str) -> Any:
    if config is None:
        return None
    for name in names:
        if hasattr(config, name):
            value = getattr(config, name)
            if value is not None:
                return value
    return None


def _env_value(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and str(value).strip() != "":
            return value
    return None


def _resolved_store_responses(config: TwinrConfig | None) -> bool:
    config_value = _config_value(
        config,
        "openai_store_responses",
        "openai_response_store",
        "store_responses",
    )
    parsed = _parse_bool_setting(config_value)
    if parsed is not None:
        return parsed
    env_value = _env_value("TWINR_OPENAI_STORE_RESPONSES", "OPENAI_STORE_RESPONSES")
    parsed = _parse_bool_setting(env_value)
    if parsed is not None:
        return parsed
    return True


def _resolved_parallel_tool_calls(config: TwinrConfig | None) -> bool:
    config_value = _config_value(
        config,
        "openai_parallel_tool_calls",
        "parallel_tool_calls",
    )
    parsed = _parse_bool_setting(config_value)
    if parsed is not None:
        return parsed
    env_value = _env_value("TWINR_OPENAI_PARALLEL_TOOL_CALLS", "OPENAI_PARALLEL_TOOL_CALLS")
    parsed = _parse_bool_setting(env_value)
    if parsed is not None:
        return parsed
    return False


def _resolved_tool_argument_max_bytes(config: TwinrConfig | None) -> int:
    config_value = _config_value(
        config,
        "openai_tool_argument_max_bytes",
        "tool_argument_max_bytes",
    )
    env_value = _env_value("TWINR_OPENAI_TOOL_ARGUMENT_MAX_BYTES", "OPENAI_TOOL_ARGUMENT_MAX_BYTES")
    raw_value = config_value if config_value is not None else env_value
    try:
        return max(1024, int(raw_value))
    except (TypeError, ValueError):
        return _DEFAULT_TOOL_ARGUMENTS_MAX_BYTES


def _resolved_structured_verbosity(config: TwinrConfig | None) -> str:
    config_value = _config_value(
        config,
        "openai_structured_text_verbosity",
        "structured_text_verbosity",
        "openai_text_verbosity",
        "text_verbosity",
    )
    env_value = _env_value("TWINR_OPENAI_STRUCTURED_VERBOSITY", "OPENAI_STRUCTURED_VERBOSITY")
    candidate = str(config_value if config_value is not None else env_value or "low").strip().lower()
    return candidate if candidate in _SUPPORTED_TEXT_VERBOSITIES else "low"


def _resolved_compact_threshold(config: TwinrConfig | None) -> int | None:
    config_value = _config_value(
        config,
        "openai_context_compact_threshold",
        "context_compact_threshold",
    )
    env_value = _env_value("TWINR_OPENAI_COMPACT_THRESHOLD", "OPENAI_COMPACT_THRESHOLD")
    raw_value = config_value if config_value is not None else env_value
    if raw_value in (None, "", 0, "0"):
        return None
    try:
        threshold = int(raw_value)
    except (TypeError, ValueError):
        return None
    return threshold if threshold > 0 else None


def _ensure_text_options(request: dict[str, Any]) -> dict[str, Any]:
    text_options = request.get("text")
    if isinstance(text_options, dict):
        return text_options
    text_options = {}
    request["text"] = text_options
    return text_options


def _add_include_value(request: dict[str, Any], value: str) -> None:
    includes = request.get("include")
    if isinstance(includes, list):
        if value not in includes:
            includes.append(value)
        return
    request["include"] = [value]


def _maybe_enable_context_management(request: dict[str, Any], *, config: TwinrConfig | None) -> None:
    threshold = _resolved_compact_threshold(config)
    if threshold is None or "context_management" in request:
        return
    request["context_management"] = {"compact_threshold": threshold}


def _model_supports_reasoning_effort(model: str) -> bool:
    normalized = (model or "").strip().lower()
    if not normalized:
        return False
    return normalized.startswith("gpt-5") or bool(_O_SERIES_MODEL_PATTERN.match(normalized))


def _apply_reasoning_effort_request(
    backend: Any,
    request: dict[str, Any],
    *,
    model: str,
    reasoning_effort: str | None,
) -> None:
    helper = getattr(backend, "_apply_reasoning_effort", None)
    if callable(helper):
        helper(request, model=model, reasoning_effort=reasoning_effort)
        return
    normalized_effort = (reasoning_effort or "").strip().lower()
    if not normalized_effort:
        request.pop("reasoning", None)
        return
    if not _model_supports_reasoning_effort(model):
        request.pop("reasoning", None)
        return
    request["reasoning"] = {"effort": normalized_effort}


def _parse_gpt_version(model: str) -> tuple[int, int] | None:
    match = _GPT_VERSION_PATTERN.match((model or "").strip().lower())
    if not match:
        return None
    major = int(match.group(1))
    minor = int(match.group(2) or 0)
    return major, minor


def _model_supports_tool_search(model: str) -> bool:
    version = _parse_gpt_version(model)
    if version is None:
        return False
    major, minor = version
    return major > 5 or (major == 5 and minor >= 4)


def _normalize_strict_json_schema(schema: Any, *, root: bool = False) -> Any:
    """Normalize one JSON Schema to OpenAI strict-mode requirements.

    The 2026 Structured Outputs subset supports nested ``anyOf`` and ``enum``
    but still rejects specific composition keywords such as ``allOf`` and
    conditional branches. We preserve supported constraints, remove only the
    documented unsupported ones, and make object schemas strict by default.
    """
    if isinstance(schema, list):
        return [_normalize_strict_json_schema(item) for item in schema]
    if not isinstance(schema, Mapping):
        return copy.deepcopy(schema)

    normalized: dict[str, Any] = {}
    for key, value in schema.items():
        if key in _STRICT_SCHEMA_UNSUPPORTED_KEYWORDS:
            continue
        if key == "properties" and isinstance(value, Mapping):
            properties: dict[str, Any] = {}
            for prop_name, prop_schema in value.items():
                properties[str(prop_name)] = _normalize_strict_json_schema(prop_schema)
            normalized[key] = properties
            continue
        if key in {"items", "contains", "additionalProperties", "propertyNames", "unevaluatedItems", "unevaluatedProperties"}:
            normalized[key] = _normalize_strict_json_schema(value)
            continue
        if key in {"$defs", "definitions", "patternProperties"} and isinstance(value, Mapping):
            normalized[key] = {
                str(name): _normalize_strict_json_schema(item)
                for name, item in value.items()
            }
            continue
        if root and key in {"anyOf", "oneOf", "prefixItems"}:
            continue
        if key in {"anyOf", "oneOf", "prefixItems"} and isinstance(value, list):
            normalized[key] = [_normalize_strict_json_schema(item) for item in value]
            continue
        normalized[key] = copy.deepcopy(value)

    if root and normalized.get("type") != "object":
        raise RuntimeError("Function parameters must use a root JSON object schema for strict mode")

    schema_type = normalized.get("type")
    if schema_type == "object" or (isinstance(schema_type, list) and "object" in schema_type):
        properties = normalized.get("properties")
        if isinstance(properties, Mapping):
            property_names = [str(name) for name in properties]
            normalized["required"] = property_names
        normalized["additionalProperties"] = False
    return normalized


def _normalize_openai_tool_schema(schema: dict[str, Any]) -> dict[str, Any]:
    copied = copy.deepcopy(schema)
    tool_type = str(copied.get("type", "")).strip()

    if tool_type == "function":
        parameters = copied.get("parameters")
        if parameters is None:
            return copied
        if isinstance(parameters, Mapping):
            copied["parameters"] = _normalize_strict_json_schema(dict(parameters), root=True)
            copied.setdefault("strict", True)
        return copied

    if tool_type == "namespace":
        tools = copied.get("tools")
        if isinstance(tools, Sequence) and not isinstance(tools, (str, bytes, bytearray)):
            copied["tools"] = [
                _normalize_openai_tool_schema(tool) if isinstance(tool, dict) else copy.deepcopy(tool)
                for tool in tools
            ]
        return copied

    return copied


def _iter_tool_definitions(tools: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    flattened: list[dict[str, Any]] = []

    def _walk(tool: dict[str, Any]) -> None:
        flattened.append(tool)
        if str(tool.get("type", "")).strip() != "namespace":
            return
        nested = tool.get("tools")
        if not isinstance(nested, Sequence) or isinstance(nested, (str, bytes, bytearray)):
            return
        for item in nested:
            if isinstance(item, dict):
                _walk(item)

    for tool in tools:
        if isinstance(tool, dict):
            _walk(tool)
    return tuple(flattened)


def _has_deferred_loading_tools(tools: Sequence[dict[str, Any]]) -> bool:
    for tool in _iter_tool_definitions(tools):
        if str(tool.get("type", "")).strip() == "function" and _optional_bool(tool.get("defer_loading")):
            return True
    return False


def _request_has_app_tools(tools: Sequence[dict[str, Any]]) -> bool:
    for tool in _iter_tool_definitions(tools):
        if str(tool.get("type", "")).strip() in {"function", "custom", "namespace"}:
            return True
    return False


def _ensure_tool_search_tool(tools: list[dict[str, Any]], *, model: str) -> None:
    if not _has_deferred_loading_tools(tools):
        return
    if not _model_supports_tool_search(model):
        return
    for tool in tools:
        if isinstance(tool, dict) and str(tool.get("type", "")).strip() == "tool_search":
            return
    tools.insert(0, {"type": "tool_search"})


def _provider_bundle_field_names(bundle_cls: type[Any]) -> set[str]:
    if is_dataclass(bundle_cls):
        return {field.name for field in fields(bundle_cls)}
    annotations = getattr(bundle_cls, "__annotations__", None)
    if isinstance(annotations, dict):
        return set(annotations)
    return set()


def _extract_structured_response_object(
    backend: OpenAIBackend,
    response: Any,
    *,
    context: str,
) -> dict[str, Any]:
    parsed_payload = _coerce_mapping(getattr(response, "output_parsed", None))
    if parsed_payload is not None:
        return parsed_payload

    for item in getattr(response, "output", None) or ():
        for content in getattr(item, "content", None) or ():
            parsed_content = _coerce_mapping(getattr(content, "parsed", None))
            if parsed_content is not None:
                return parsed_content

    payload_text = _coerce_text(backend._extract_output_text(response))
    if not payload_text.strip():
        raise RuntimeError(f"{context} returned empty structured output")
    try:
        return extract_json_object(payload_text)
    except ValueError as exc:
        raise RuntimeError(f"{context} returned invalid JSON: {exc}") from exc


def _create_response_with_reasoning_fallback(
    backend: Any,
    request: dict[str, Any],
    *,
    context: str,
    retry_max_output_tokens: Sequence[int] | None = None,
) -> Any:
    def _create_once(request_payload: dict[str, Any]) -> Any:
        try:
            return backend._client.responses.create(**request_payload)
        except Exception as exc:
            if not _is_reasoning_unsupported_error(exc) or "reasoning" not in request_payload:
                raise
            retry_request = dict(request_payload)
            retry_request.pop("reasoning", None)
            return backend._client.responses.create(**retry_request)

    response = _create_once(request)
    for retry_budget in _iter_retry_max_output_tokens(
        response,
        request=request,
        retry_max_output_tokens=retry_max_output_tokens,
    ):
        retry_request = dict(request)
        retry_request["max_output_tokens"] = retry_budget
        response = _create_once(retry_request)

    _validate_response_status(response, context=context)
    return response


@dataclass
class OpenAISpeechToTextProvider:
    backend: OpenAIBackend

    @property
    def config(self) -> TwinrConfig:
        return self.backend.config

    @config.setter
    def config(self, value: TwinrConfig) -> None:
        self.backend.config = value

    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
        language: str | None = None,
        prompt: str | None = None,
    ) -> str:
        if not audio_bytes:
            return ""
        return self.backend.transcribe(
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
        if isinstance(path, str) and not path.strip():
            return ""
        return self.backend.transcribe_path(path, language=language, prompt=prompt)


@dataclass
class OpenAIAgentTextProvider:
    backend: OpenAIBackend

    @property
    def config(self) -> TwinrConfig:
        return self.backend.config

    @config.setter
    def config(self, value: TwinrConfig) -> None:
        self.backend.config = value

    def respond_streaming(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> TextResponse:
        return self.backend.respond_streaming(
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
        return self.backend.respond_with_metadata(
            prompt,
            conversation=conversation,
            instructions=instructions,
            allow_web_search=allow_web_search,
        )

    def respond_to_images_with_metadata(
        self,
        prompt: str,
        *,
        images: Sequence[OpenAIImageInput],
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
    ) -> TextResponse:
        return self.backend.respond_to_images_with_metadata(
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
        return self.backend.search_live_info_with_metadata(
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
        return self.backend.compose_print_job_with_metadata(
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
        return self.backend.phrase_due_reminder_with_metadata(reminder, now=now)

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
        return self.backend.phrase_proactive_prompt_with_metadata(
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
        return self.backend.fulfill_automation_prompt_with_metadata(
            prompt,
            allow_web_search=allow_web_search,
            delivery=delivery,
        )


@dataclass
class OpenAITextToSpeechProvider:
    backend: OpenAIBackend

    @property
    def config(self) -> TwinrConfig:
        return self.backend.config

    @config.setter
    def config(self, value: TwinrConfig) -> None:
        self.backend.config = value

    def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        response_format: str | None = None,
        instructions: str | None = None,
    ) -> bytes:
        if not text.strip():
            return b""
        return self.backend.synthesize(
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
    ):
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if not text.strip():
            return iter(())
        return self.backend.synthesize_stream(
            text,
            voice=voice,
            response_format=response_format,
            instructions=instructions,
            chunk_size=chunk_size,
        )


@dataclass
class OpenAIToolCallingAgentProvider:
    backend: OpenAIBackend
    model_override: str | None = None
    reasoning_effort_override: str | None = None
    base_instructions_override: str | None = None
    replace_base_instructions: bool = False

    @property
    def config(self) -> TwinrConfig:
        return self.backend.config

    @config.setter
    def config(self, value: TwinrConfig) -> None:
        self.backend.config = value

    def _resolved_model(self) -> str:
        override = (self.model_override or "").strip()
        if override:
            return override
        return self.config.default_model

    def _resolved_reasoning_effort(self) -> str:
        override = (self.reasoning_effort_override or "").strip()
        if override:
            return override
        return self.config.openai_reasoning_effort

    def _merged_base_instructions(self, instructions: str | None) -> str | None:
        if self.replace_base_instructions:
            return merge_instructions(
                self.base_instructions_override or self.backend._resolve_tool_loop_base_instructions(),
                instructions,
            )
        return merge_instructions(
            self.backend._resolve_tool_loop_base_instructions(),
            self.base_instructions_override,
            instructions,
        )

    def _apply_tool_loop_runtime_policy(self, request: dict[str, Any], *, model: str) -> None:
        store = _resolved_store_responses(self.config)
        request["store"] = store
        if not store and (_model_supports_reasoning_effort(model) or "reasoning" in request):
            _add_include_value(request, "reasoning.encrypted_content")
        _maybe_enable_context_management(request, config=self.config)

        tools = request.get("tools")
        if isinstance(tools, list) and _request_has_app_tools(tools):
            request["parallel_tool_calls"] = _resolved_parallel_tool_calls(self.config)
        elif "parallel_tool_calls" in request:
            request.pop("parallel_tool_calls", None)

        if isinstance(tools, list):
            _ensure_tool_search_tool(tools, model=model)

    def start_turn_streaming(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        tool_schemas: Sequence[dict[str, Any]] = (),
        allow_web_search: bool | None = None,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> ToolCallingTurnResponse:
        model = self._resolved_model()
        reasoning_effort = self._resolved_reasoning_effort()
        request = self.backend._build_response_request(
            prompt,
            conversation=conversation,
            instructions=self._merged_base_instructions(instructions),
            allow_web_search=allow_web_search,
            model=model,
            reasoning_effort=reasoning_effort,
            prompt_cache_scope="tool_loop_start",
        )
        _apply_reasoning_effort_request(
            self.backend,
            request,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        self._merge_tool_schemas(request, tool_schemas, model=model)
        self._apply_tool_loop_runtime_policy(request, model=model)
        return self._run_streaming_request(request, on_text_delta=on_text_delta)

    def continue_turn_streaming(
        self,
        *,
        continuation_token: str,
        tool_results: Sequence[AgentToolResult],
        instructions: str | None = None,
        tool_schemas: Sequence[dict[str, Any]] = (),
        allow_web_search: bool | None = None,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> ToolCallingTurnResponse:
        if not continuation_token.strip():
            raise RuntimeError("continue_turn_streaming requires a continuation_token")
        model = self._resolved_model()
        request: dict[str, Any] = {
            "model": model,
            "previous_response_id": continuation_token,
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": result.call_id,
                    "output": result.serialized_output,
                }
                for result in tool_results
            ],
        }
        _apply_reasoning_effort_request(
            self.backend,
            request,
            model=model,
            reasoning_effort=self._resolved_reasoning_effort(),
        )
        merged_instructions = merge_instructions(
            self._merged_base_instructions(instructions),
            user_response_language_instruction(self.config.openai_realtime_language),
        )
        if merged_instructions:
            request["instructions"] = merged_instructions

        use_web_search = self.config.openai_enable_web_search if allow_web_search is None else allow_web_search
        web_search_tools = self.backend._build_tools(use_web_search, model=model)
        if web_search_tools:
            request["tools"] = list(web_search_tools)

        self._merge_tool_schemas(request, tool_schemas, model=model)
        self.backend._apply_prompt_cache(request, scope="tool_loop_continue", model=model)
        self._apply_tool_loop_runtime_policy(request, model=model)
        return self._run_streaming_request(request, on_text_delta=on_text_delta)

    def _merge_tool_schemas(
        self,
        request: dict[str, Any],
        tool_schemas: Sequence[dict[str, Any]],
        *,
        model: str,
    ) -> None:
        tools = list(request.get("tools") or [])
        normalized_tool_schemas = [_normalize_openai_tool_schema(schema) for schema in tool_schemas]
        tools.extend(normalized_tool_schemas)
        if tools:
            _ensure_tool_search_tool(tools, model=model)
            request["tools"] = tools
            request["tool_choice"] = "auto"
        elif request.get("tools"):
            request["tool_choice"] = "auto"

    def _run_streaming_request(
        self,
        request: dict[str, Any],
        *,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> ToolCallingTurnResponse:
        try:
            streamed_text, response = self._consume_stream(request, on_text_delta=on_text_delta)
        except Exception as exc:
            if not _is_reasoning_unsupported_error(exc) or "reasoning" not in request:
                raise
            retry_request = dict(request)
            retry_request.pop("reasoning", None)
            streamed_text, response = self._consume_stream(retry_request, on_text_delta=on_text_delta)

        fallback_text = _coerce_text(self.backend._extract_output_text(response)).strip()
        text = streamed_text.strip() or fallback_text
        if text and not streamed_text.strip():
            _emit_text_delta(on_text_delta, text, context="tool-loop fallback text")
        response_id = getattr(response, "id", None)
        return ToolCallingTurnResponse(
            text=text,
            tool_calls=self._extract_tool_calls(response),
            response_id=response_id,
            request_id=getattr(response, "_request_id", None),
            model=extract_model_name(response, request["model"]),
            token_usage=extract_token_usage(response),
            used_web_search=self.backend._used_web_search(response),
            continuation_token=response_id,
        )

    def _consume_stream(
        self,
        request: dict[str, Any],
        *,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> tuple[str, Any]:
        streamed_chunks: list[str] = []
        with self.backend._client.responses.stream(**request) as stream:
            for event in stream:
                if getattr(event, "type", None) != "response.output_text.delta":
                    continue
                delta = _coerce_text(getattr(event, "delta", ""))
                if not delta:
                    continue
                streamed_chunks.append(delta)
                _emit_text_delta(on_text_delta, delta, context="tool-loop stream delta")
            response = stream.get_final_response()
        _validate_response_status(response, context="tool-loop response")
        return "".join(streamed_chunks), response

    def _extract_tool_calls(self, response: Any) -> tuple[AgentToolCall, ...]:
        output_items = getattr(response, "output", None) or []
        function_calls: list[AgentToolCall] = []
        max_argument_bytes = _resolved_tool_argument_max_bytes(self.config)

        for item in output_items:
            item_type = str(getattr(item, "type", "")).strip()
            if item_type in _UNSUPPORTED_OUTPUT_ITEM_TYPES:
                raise RuntimeError(
                    f"Unsupported response output item type {item_type!r}: "
                    f"{_UNSUPPORTED_OUTPUT_ITEM_TYPES[item_type]}"
                )
            if item_type != "function_call":
                continue

            name = str(getattr(item, "name", "")).strip()
            call_id = str(getattr(item, "call_id", "")).strip()
            if not name or not call_id:
                continue

            parsed_arguments = _coerce_mapping(getattr(item, "parsed_arguments", None))
            if parsed_arguments is not None:
                arguments = parsed_arguments
                raw_arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
            else:
                raw_arguments = str(getattr(item, "arguments", "") or "{}").strip() or "{}"
                if len(raw_arguments.encode("utf-8", "replace")) > max_argument_bytes:
                    raise RuntimeError(
                        f"Tool arguments for {name!r} exceed the configured size limit "
                        f"of {max_argument_bytes} bytes"
                    )
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Tool arguments are not valid JSON: {exc.msg}") from exc
                if not isinstance(arguments, dict):
                    raise RuntimeError("Tool arguments must decode to a JSON object")

            function_calls.append(
                AgentToolCall(
                    name=name,
                    call_id=call_id,
                    arguments=arguments,
                    raw_arguments=raw_arguments,
                )
            )

        return tuple(function_calls)


@dataclass
class OpenAIConversationClosureDecisionProvider:
    backend: OpenAIBackend
    model_override: str | None = None
    reasoning_effort_override: str | None = None
    base_instructions_override: str | None = None
    replace_base_instructions: bool = False

    @property
    def config(self) -> TwinrConfig:
        return self.backend.config

    @config.setter
    def config(self, value: TwinrConfig) -> None:
        self.backend.config = value

    def _resolved_model(self) -> str:
        override = (self.model_override or "").strip()
        if override:
            return override
        resolved = (self.config.conversation_closure_model or "").strip()
        return resolved or self.config.default_model

    def _resolved_reasoning_effort(self) -> str | None:
        override = (self.reasoning_effort_override or "").strip()
        if override:
            return override
        resolved = (self.config.conversation_closure_reasoning_effort or "").strip()
        return resolved or None

    def _merged_base_instructions(self, instructions: str | None) -> str | None:
        return merge_instructions(self.base_instructions_override, instructions)

    def decide(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        timeout_seconds: float | None = None,
    ) -> ConversationClosureProviderDecision:
        model = self._resolved_model()
        reasoning_effort = self._resolved_reasoning_effort()
        request = self.backend._build_response_request(
            prompt,
            conversation=conversation,
            instructions=self._merged_base_instructions(instructions),
            allow_web_search=False,
            model=model,
            reasoning_effort=reasoning_effort or "",
            max_output_tokens=max(16, int(self.config.conversation_closure_max_output_tokens)),
            prompt_cache_scope="conversation_closure",
        )
        _apply_reasoning_effort_request(
            self.backend,
            request,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        if timeout_seconds is not None:
            request["timeout"] = timeout_seconds

        text_options = _ensure_text_options(request)
        text_options["verbosity"] = _resolved_structured_verbosity(self.config)
        text_options["format"] = {
            "type": "json_schema",
            "name": "twinr_conversation_closure_decision",
            "schema": _CONVERSATION_CLOSURE_DECISION_SCHEMA,
            "strict": True,
        }

        response = _create_response_with_reasoning_fallback(
            self.backend,
            request,
            context="conversation closure decision",
        )
        payload = _extract_structured_response_object(
            self.backend,
            response,
            context="conversation closure decision",
        )
        return ConversationClosureProviderDecision(
            close_now=_validated_bool(payload.get("close_now"), context="conversation closure close_now"),
            confidence=_validated_probability(payload.get("confidence"), context="conversation closure confidence"),
            reason=_coerce_text(payload.get("reason")),
            matched_topics=_coerce_topic_titles(payload.get("matched_topics")),
            follow_up_action=_validated_choice(
                payload.get("follow_up_action"),
                allowed=("continue", "end"),
                default="end",
                context="conversation closure follow_up_action",
            ),
            response_id=getattr(response, "id", None),
            request_id=getattr(response, "_request_id", None),
            model=extract_model_name(response, model),
            token_usage=extract_token_usage(response),
        )


@dataclass
class OpenAISupervisorDecisionProvider:
    backend: OpenAIBackend
    model_override: str | None = None
    reasoning_effort_override: str | None = None
    base_instructions_override: str | None = None
    replace_base_instructions: bool = False

    @property
    def config(self) -> TwinrConfig:
        return self.backend.config

    @config.setter
    def config(self, value: TwinrConfig) -> None:
        self.backend.config = value

    def _resolved_model(self) -> str:
        override = (self.model_override or "").strip()
        if override:
            return override
        return self.config.default_model

    def _resolved_reasoning_effort(self) -> str:
        override = (self.reasoning_effort_override or "").strip()
        if override:
            return override
        return self.config.openai_reasoning_effort

    def _resolved_max_output_tokens(self, *, model: str) -> int:
        configured = max(32, int(self.config.streaming_supervisor_max_output_tokens))
        normalized_model = str(model or "").strip().lower()
        if normalized_model.startswith(("gpt-5", "o")):
            return max(configured, 512)
        return configured

    def _retry_max_output_tokens(self, *, model: str, initial: int) -> tuple[int, ...] | None:
        normalized_model = str(model or "").strip().lower()
        if not normalized_model.startswith(("gpt-5", "o")):
            return None
        ladder = (512, 768, 1024)
        return tuple(budget for budget in ladder if budget > max(32, int(initial)))

    def _merged_base_instructions(self, instructions: str | None) -> str | None:
        if self.replace_base_instructions:
            return merge_instructions(
                self.base_instructions_override or self.backend._resolve_tool_loop_base_instructions(),
                instructions,
            )
        return merge_instructions(
            self.backend._resolve_tool_loop_base_instructions(),
            self.base_instructions_override,
            instructions,
        )

    def decide(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
    ) -> SupervisorDecision:
        model = self._resolved_model()
        reasoning_effort = self._resolved_reasoning_effort()
        request = self.backend._build_response_request(
            prompt,
            conversation=conversation,
            instructions=self._merged_base_instructions(instructions),
            allow_web_search=False,
            model=model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=self._resolved_max_output_tokens(model=model),
            prompt_cache_scope="supervisor_decision",
        )
        _apply_reasoning_effort_request(
            self.backend,
            request,
            model=model,
            reasoning_effort=reasoning_effort,
        )

        text_options = _ensure_text_options(request)
        text_options["verbosity"] = _resolved_structured_verbosity(self.config)
        text_options["format"] = {
            "type": "json_schema",
            "name": "twinr_supervisor_decision",
            "schema": _SUPERVISOR_DECISION_SCHEMA,
            "strict": True,
        }

        response = _create_response_with_reasoning_fallback(
            self.backend,
            request,
            context="supervisor decision",
            retry_max_output_tokens=self._retry_max_output_tokens(
                model=model,
                initial=int(request["max_output_tokens"]),
            ),
        )
        payload = _extract_structured_response_object(
            self.backend,
            response,
            context="supervisor decision",
        )

        runtime_tool_name = _optional_text(payload.get("runtime_tool_name"))
        runtime_tool_arguments = _validated_optional_json_object(
            payload.get("runtime_tool_arguments_json"),
            context="supervisor runtime_tool_arguments_json",
        )
        if runtime_tool_arguments is not None and not runtime_tool_name:
            raise RuntimeError("supervisor decision returned runtime_tool_arguments_json without runtime_tool_name")

        return SupervisorDecision(
            action=_validated_choice(
                payload.get("action"),
                allowed=("direct", "handoff", "end_conversation"),
                default="handoff",
                context="supervisor decision action",
            ),
            spoken_ack=_optional_text(payload.get("spoken_ack")),
            spoken_reply=_optional_text(payload.get("spoken_reply")),
            kind=_validated_optional_choice(
                payload.get("kind"),
                allowed=("general", "search", "memory", "automation"),
                context="supervisor decision kind",
            ),
            goal=_optional_text(payload.get("goal")),
            prompt=_optional_text(payload.get("prompt")),
            allow_web_search=_optional_bool(payload.get("allow_web_search")),
            location_hint=_optional_text(payload.get("location_hint")),
            date_context=_optional_text(payload.get("date_context")),
            context_scope=_validated_optional_choice(
                payload.get("context_scope"),
                allowed=("tiny_recent", "full_context"),
                context="supervisor decision context_scope",
            ),
            runtime_tool_name=runtime_tool_name,
            runtime_tool_arguments=runtime_tool_arguments,
            response_id=getattr(response, "id", None),
            request_id=getattr(response, "_request_id", None),
            model=extract_model_name(response, model),
            token_usage=extract_token_usage(response),
        )


@dataclass
class OpenAIFirstWordProvider:
    backend: OpenAIBackend
    model_override: str | None = None
    reasoning_effort_override: str | None = None
    base_instructions_override: str | None = None
    replace_base_instructions: bool = False

    @property
    def config(self) -> TwinrConfig:
        return self.backend.config

    @config.setter
    def config(self, value: TwinrConfig) -> None:
        self.backend.config = value

    def _resolved_model(self) -> str:
        override = (self.model_override or "").strip()
        if override:
            return override
        resolved = (self.config.streaming_first_word_model or "").strip()
        return resolved or self.config.default_model

    def _resolved_reasoning_effort(self) -> str | None:
        override = (self.reasoning_effort_override or "").strip()
        if override:
            return override
        resolved = (self.config.streaming_first_word_reasoning_effort or "").strip()
        return resolved or None

    def _merged_base_instructions(self, instructions: str | None) -> str | None:
        return merge_instructions(self.base_instructions_override, instructions)

    def reply(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
    ) -> FirstWordReply:
        preferred_model = self._resolved_model()
        reasoning_effort = self._resolved_reasoning_effort()

        def _call(model: str):
            request = self.backend._build_response_request(
                prompt,
                conversation=conversation,
                instructions=self._merged_base_instructions(instructions),
                allow_web_search=False,
                model=model,
                reasoning_effort=reasoning_effort or "",
                max_output_tokens=max(16, int(self.config.streaming_first_word_max_output_tokens)),
                prompt_cache_scope="first_word",
            )
            _apply_reasoning_effort_request(
                self.backend,
                request,
                model=model,
                reasoning_effort=reasoning_effort,
            )

            text_options = _ensure_text_options(request)
            text_options["verbosity"] = _resolved_structured_verbosity(self.config)
            text_options["format"] = {
                "type": "json_schema",
                "name": "twinr_first_word_reply",
                "schema": _FIRST_WORD_REPLY_SCHEMA,
                "strict": True,
            }
            return _create_response_with_reasoning_fallback(
                self.backend,
                request,
                context="first-word reply",
            )

        response, model_used = self.backend._call_with_model_fallback(
            preferred_model,
            _FIRST_WORD_MODEL_FALLBACKS,
            _call,
        )
        payload = _extract_structured_response_object(
            self.backend,
            response,
            context="first-word reply",
        )
        return FirstWordReply(
            mode=_validated_choice(
                payload.get("mode"),
                allowed=("direct", "filler"),
                default="filler",
                context="first-word mode",
            ),
            spoken_text=_coerce_text(payload.get("spoken_text")).strip(),
            response_id=getattr(response, "id", None),
            request_id=getattr(response, "_request_id", None),
            model=extract_model_name(response, model_used),
            token_usage=extract_token_usage(response),
        )


@dataclass
class OpenAIProviderBundle(ProviderBundle):
    backend: OpenAIBackend
    combined: CompositeSpeechAgentProvider

    @classmethod
    def from_backend(cls, backend: OpenAIBackend) -> "OpenAIProviderBundle":
        stt = OpenAISpeechToTextProvider(backend)
        agent = OpenAIAgentTextProvider(backend)
        tts = OpenAITextToSpeechProvider(backend)
        tool_agent = OpenAIToolCallingAgentProvider(backend)
        supervisor = OpenAISupervisorDecisionProvider(backend)
        first_word = OpenAIFirstWordProvider(backend)
        combined = CompositeSpeechAgentProvider(stt=stt, agent=agent, tts=tts)

        bundle_kwargs: dict[str, Any] = {
            "stt": stt,
            "agent": agent,
            "tts": tts,
            "tool_agent": tool_agent,
            "backend": backend,
            "combined": combined,
        }

        field_names = _provider_bundle_field_names(cls)
        for field_name in (
            "supervisor",
            "supervisor_provider",
            "supervisor_decision",
            "supervisor_decision_provider",
        ):
            if field_name in field_names:
                bundle_kwargs[field_name] = supervisor
        for field_name in ("first_word", "first_word_provider"):
            if field_name in field_names:
                bundle_kwargs[field_name] = first_word

        return cls(**bundle_kwargs)

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "OpenAIProviderBundle":
        return cls.from_backend(OpenAIBackend(config=config))
