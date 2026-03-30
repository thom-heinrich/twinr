# CHANGELOG: 2026-03-30
# BUG-1: Persist request tool definitions across continuation turns so Groq does not reject valid assistant tool_call history when callers omit tool_schemas on continue_turn_streaming().
# BUG-2: Stop stacking manual retries on top of groq-python's built-in retries; use per-request SDK options instead so transient failures do not multiply latency and block the Pi for minutes.
# SEC-1: Bound serialized tool-result payload size before storing it in memory and sending it back to the model, preventing practical memory/token-cost DoS from oversized local tool outputs.
# SEC-2: Use Groq-native web search and vision first, and make cross-provider fallback for live-search / vision turns opt-in so sensitive senior-user prompts are not silently routed to another vendor.
# IMP-1: Add 2026 Groq built-in tool support (browser_search on GPT-OSS; web_search on compound) and expose used_web_search from executed_tools instead of ignoring allow_web_search.
# IMP-2: Add configurable service tier, reasoning controls, tool-call self-repair retry, OpenAI-style nested schema support, and native Groq vision handling for lower-latency Pi deployments.

"""Provide Groq-backed text, vision, and tool-calling adapters for Twinr.

This module translates Twinr's agent-provider contracts into Groq chat
completion requests. It also owns the continuation state required to resume
tool-calling turns after Twinr executes the requested tools locally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Lock
from typing import Any, Callable, Iterable, Mapping, Sequence, cast
from uuid import uuid4
import base64
import copy
import json
import logging
import time
import threading

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.contracts import (
    AgentToolCall,
    AgentToolResult,
    AgentTextProvider,
    ConversationLike,
    ImageInputLike,
    SearchResponse,
    TextResponse,
    ToolCallingTurnResponse,
)
from twinr.agent.base_agent.conversation.language import user_response_language_instruction
from twinr.agent.base_agent.prompting.personality import load_personality_instructions, merge_instructions
from twinr.ops.usage import TokenUsage
from twinr.providers.groq.client import default_groq_client
from twinr.providers.groq.types import GroqTextResponse

logger = logging.getLogger(__name__)

_DEFAULT_GROQ_REQUEST_TIMEOUT_SECONDS = 45.0
_DEFAULT_GROQ_MAX_RETRIES = 1
_DEFAULT_CONTINUATION_TTL_SECONDS = 300.0
_DEFAULT_MAX_CONTINUATIONS = 128
_DEFAULT_MAX_TOOL_RESULT_CHARS = 24_000
_DEFAULT_GROQ_SERVICE_TIER = ""
_DEFAULT_GROQ_TEXT_SEARCH_MODEL = "groq/compound-mini"
_DEFAULT_GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
# BREAKING: Live-search turns no longer cross-route to the support provider by default; set config.groq_allow_search_fallback=True to restore the old behavior.
_DEFAULT_ALLOW_SEARCH_FALLBACK = False
# BREAKING: Vision turns no longer cross-route to the support provider by default; set config.groq_allow_vision_fallback=True to restore the old behavior.
_DEFAULT_ALLOW_VISION_FALLBACK = False
_LOCAL_TOOL_REPAIR_INSTRUCTION = (
    "Your previous attempt to call a local tool failed validation. "
    "You must respond using only the provided tools. "
    "If you call a tool, emit strictly valid OpenAI-style function calls with a single JSON object "
    "for arguments that exactly matches the tool schema. "
    "Do not invent tool names. "
    "Do not place tool syntax inside free-form text."
)


@dataclass(slots=True)
class _ContinuationState:
    """Track the pending tool-calling state for one Groq continuation token."""

    messages: list[dict[str, Any]]
    expected_tool_call_ids: tuple[str, ...]
    request_tools: tuple[dict[str, Any], ...] = ()
    created_monotonic: float = 0.0
    last_access_monotonic: float = 0.0
    in_flight: bool = False


@dataclass(frozen=True, slots=True)
class _ToolCompletionPayload:
    """Store one normalized Groq tool-completion result before continuation exists."""

    text: str
    tool_calls: tuple[AgentToolCall, ...]
    response_id: str | None = None
    request_id: str | None = None
    model: str | None = None
    token_usage: object | None = None
    used_web_search: bool = False


@dataclass(frozen=True, slots=True)
class _ChatRunResult:
    """Normalize one assistant message plus provider metadata."""

    text: str
    response_id: str | None
    request_id: str | None
    model: str | None
    token_usage: TokenUsage | None
    used_web_search: bool = False
    executed_tools: tuple[dict[str, Any], ...] = ()


def _safe_int(value: object) -> int | None:
    """Best-effort convert a value to ``int`` without raising."""
    if value is None:
        return None
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError):
        return None


def _non_negative_int(value: object, default: int) -> int:
    """Return ``default`` when a value is missing or negative."""
    coerced = _safe_int(value)
    if coerced is None or coerced < 0:
        return default
    return coerced


def _positive_float(value: object, default: float | None) -> float | None:
    """Return ``default`` when a value is missing or not strictly positive."""
    if value is None:
        return default
    try:
        coerced = float(cast(Any, value))
    except (TypeError, ValueError):
        return default
    if coerced <= 0:
        return default
    return coerced


def _as_text_response(response: GroqTextResponse) -> TextResponse:
    """Bridge the immutable Groq response dataclass to Twinr's text-response protocol."""
    return cast(TextResponse, response)


def _groq_request_timeout_seconds(config: TwinrConfig) -> float | None:
    """Read the per-request Groq timeout override from config."""
    return _positive_float(
        getattr(config, "groq_request_timeout_seconds", _DEFAULT_GROQ_REQUEST_TIMEOUT_SECONDS),
        _DEFAULT_GROQ_REQUEST_TIMEOUT_SECONDS,
    )


def _groq_sdk_max_retries(config: TwinrConfig) -> int:
    """Return the retry count to pass directly into the groq-python SDK."""
    return _non_negative_int(getattr(config, "groq_max_retries", _DEFAULT_GROQ_MAX_RETRIES), _DEFAULT_GROQ_MAX_RETRIES)


def _continuation_ttl_seconds(config: TwinrConfig) -> float:
    """Read the maximum idle age for stored continuation state."""
    return _positive_float(
        getattr(config, "groq_tool_continuation_ttl_seconds", _DEFAULT_CONTINUATION_TTL_SECONDS),
        _DEFAULT_CONTINUATION_TTL_SECONDS,
    ) or _DEFAULT_CONTINUATION_TTL_SECONDS


def _max_continuations(config: TwinrConfig) -> int:
    """Read the cap for simultaneously stored continuation states."""
    return max(
        1,
        _non_negative_int(
            getattr(config, "groq_tool_max_continuations", _DEFAULT_MAX_CONTINUATIONS),
            _DEFAULT_MAX_CONTINUATIONS,
        ),
    )


def _max_tool_result_chars(config: TwinrConfig) -> int:
    """Read the maximum number of characters allowed for one serialized tool result."""
    return max(
        512,
        _non_negative_int(
            getattr(config, "groq_max_tool_result_chars", _DEFAULT_MAX_TOOL_RESULT_CHARS),
            _DEFAULT_MAX_TOOL_RESULT_CHARS,
        ),
    )


def _groq_service_tier(config: TwinrConfig) -> str | None:
    """Return the service tier to request, or ``None`` to omit it."""
    value = str(getattr(config, "groq_service_tier", _DEFAULT_GROQ_SERVICE_TIER) or "").strip().lower()
    if value in {"auto", "on_demand", "flex", "performance"}:
        return value
    return None


def _groq_reasoning_format(config: TwinrConfig, *, tool_or_json_mode: bool) -> str | None:
    """Return the requested reasoning format, defaulting to ``hidden`` for tool paths."""
    configured = str(getattr(config, "groq_reasoning_format", "") or "").strip().lower()
    if configured in {"hidden", "raw", "parsed"}:
        return configured
    if tool_or_json_mode:
        return "hidden"
    return None


def _groq_reasoning_effort(config: TwinrConfig) -> str | None:
    """Return the configured Groq reasoning effort if valid."""
    configured = str(getattr(config, "groq_reasoning_effort", "") or "").strip().lower()
    if configured in {"none", "default", "low", "medium", "high"}:
        return configured
    return None


def _model_supports_gpt_oss_builtin_tools(model: str) -> bool:
    """Return whether a model supports GPT-OSS built-in server-side tools."""
    normalized = model.strip().lower()
    return normalized in {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}


def _model_is_compound(model: str) -> bool:
    """Return whether a model is one of Groq's compound systems."""
    normalized = model.strip().lower()
    return normalized in {"groq/compound", "groq/compound-mini"}


def _text_search_model(config: TwinrConfig) -> str:
    """Return the model used for native Groq live-search turns."""
    configured = getattr(config, "groq_text_search_model", None)
    if configured is not None:
        text = str(configured or "").strip()
        return text or _DEFAULT_GROQ_TEXT_SEARCH_MODEL
    base_model = str(getattr(config, "groq_model", "") or "").strip()
    if _model_supports_gpt_oss_builtin_tools(base_model) or _model_is_compound(base_model):
        return base_model
    return _DEFAULT_GROQ_TEXT_SEARCH_MODEL


def _vision_model(config: TwinrConfig) -> str:
    """Return the vision-capable Groq model used for image turns."""
    text = str(getattr(config, "groq_vision_model", _DEFAULT_GROQ_VISION_MODEL) or "").strip()
    return text or _DEFAULT_GROQ_VISION_MODEL


def _allow_search_fallback(config: TwinrConfig) -> bool:
    """Whether search turns may be routed to the support provider after Groq failure."""
    return bool(getattr(config, "groq_allow_search_fallback", _DEFAULT_ALLOW_SEARCH_FALLBACK))


def _allow_vision_fallback(config: TwinrConfig) -> bool:
    """Whether image turns may be routed to the support provider after Groq failure."""
    return bool(getattr(config, "groq_allow_vision_fallback", _DEFAULT_ALLOW_VISION_FALLBACK))


def _text_provider_error_text(config: TwinrConfig) -> str:
    """Return the user-facing fallback text for Groq text failures."""
    text = str(
        getattr(
            config,
            "groq_text_provider_error_text",
            "I am having trouble right now. Please try again.",
        )
        or ""
    ).strip()
    return text or "I am having trouble right now. Please try again."


def _tool_provider_error_text(config: TwinrConfig) -> str:
    """Return the user-facing fallback text for Groq tool-call failures."""
    text = str(
        getattr(
            config,
            "groq_tool_provider_error_text",
            "I could not finish that step. Please try again.",
        )
        or ""
    ).strip()
    return text or "I could not finish that step. Please try again."


def _tool_continuation_expired_text(config: TwinrConfig) -> str:
    """Return the text used when a continuation token is no longer valid."""
    text = str(
        getattr(
            config,
            "groq_tool_continuation_expired_text",
            "I lost the previous step. Please try again.",
        )
        or ""
    ).strip()
    return text or "I lost the previous step. Please try again."


def _tool_continuation_busy_text(config: TwinrConfig) -> str:
    """Return the text used when a continuation token is already in flight."""
    text = str(
        getattr(
            config,
            "groq_tool_continuation_busy_text",
            "That step is already being finished. Please try again in a moment.",
        )
        or ""
    ).strip()
    return text or "That step is already being finished. Please try again in a moment."


def _tool_result_error_text(config: TwinrConfig) -> str:
    """Return the text used when tool results do not match expected calls."""
    text = str(
        getattr(
            config,
            "groq_tool_result_error_text",
            "I could not verify the tool results. Please try again.",
        )
        or ""
    ).strip()
    return text or "I could not verify the tool results. Please try again."


def _safe_emit_text(on_text_delta: Callable[[str], None] | None, text: str) -> None:
    """Emit streaming text to a callback without letting the callback raise."""
    if on_text_delta is None or not text:
        return
    try:
        on_text_delta(text)
    except Exception:
        logger.exception("Groq text delta callback failed")


def _coerce_request_client(client: Any, config: TwinrConfig) -> Any:
    """Apply per-request Groq SDK options without stacking manual retries."""
    timeout_seconds = _groq_request_timeout_seconds(config)
    max_retries = _groq_sdk_max_retries(config)
    with_options = getattr(client, "with_options", None)
    if callable(with_options):
        option_payload: dict[str, Any] = {"max_retries": max_retries}
        if timeout_seconds is not None:
            option_payload["timeout"] = timeout_seconds
        try:
            return with_options(**option_payload)
        except TypeError:
            logger.debug("Groq client.with_options() rejected timeout/max_retries; using base client")
    return client


def _invoke_chat_completion(
    client: Any,
    request: dict[str, Any],
    *,
    config: TwinrConfig,
) -> Any:
    """Call Groq chat completions using per-request SDK options."""
    request_client = _coerce_request_client(client, config)
    payload = dict(request)
    if request_client is client:
        timeout_seconds = _groq_request_timeout_seconds(config)
        if timeout_seconds is not None:
            payload["timeout"] = timeout_seconds
    return request_client.chat.completions.create(**payload)


def _run_callable_with_wall_clock_timeout(
    func: Callable[[], _ChatRunResult | tuple[Any, Any]],
    *,
    timeout_seconds: float | None,
    action_label: str,
) -> _ChatRunResult | tuple[Any, Any]:
    """Run one blocking provider call with an absolute wall-clock budget."""
    if timeout_seconds is None:
        return func()

    result_queue: Queue[tuple[bool, object]] = Queue(maxsize=1)

    def runner() -> None:
        try:
            result_queue.put((True, func()))
        except BaseException as exc:  # pragma: no cover - exercised via caller behavior.
            result_queue.put((False, exc))

    worker = threading.Thread(
        target=runner,
        name=f"groq-{action_label.replace(' ', '-')}",
        daemon=True,
    )
    worker.start()
    try:
        succeeded, payload = result_queue.get(timeout=max(0.0, timeout_seconds))
    except Empty as exc:
        raise TimeoutError(
            f"{action_label} exceeded the configured wall-clock timeout of {timeout_seconds:.1f}s"
        ) from exc
    if succeeded:
        return cast(_ChatRunResult | tuple[Any, Any], payload)
    raise cast(BaseException, payload)


def _chat_usage(source: object) -> TokenUsage | None:
    """Convert provider usage metadata into Twinr's ``TokenUsage`` contract."""
    usage = getattr(source, "usage", None)
    if usage is None:
        return None
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    if prompt_tokens is None and hasattr(usage, "input_tokens"):
        prompt_tokens = getattr(usage, "input_tokens", None)
    if completion_tokens is None and hasattr(usage, "output_tokens"):
        completion_tokens = getattr(usage, "output_tokens", None)
    token_usage = TokenUsage(
        input_tokens=_safe_int(prompt_tokens),
        output_tokens=_safe_int(completion_tokens),
        total_tokens=_safe_int(total_tokens),
    )
    return token_usage if token_usage.has_values else None


def _extract_text_fragment(value: object) -> str:
    """Normalize nested SDK content blocks into plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        fragments = [_extract_text_fragment(item) for item in value]
        return "".join(fragment for fragment in fragments if fragment)
    if isinstance(value, dict):
        item_type = str(value.get("type", "")).strip().lower()
        if item_type in {"text", "output_text", "input_text"}:
            text_value = value.get("text", "")
            if isinstance(text_value, dict) and "value" in text_value:
                return _extract_text_fragment(text_value.get("value"))
            return _extract_text_fragment(text_value)
        if "content" in value:
            return _extract_text_fragment(value.get("content"))
        if "text" in value:
            return _extract_text_fragment(value.get("text"))
        if "value" in value:
            return _extract_text_fragment(value.get("value"))
        return ""
    item_type = str(getattr(value, "type", "")).strip().lower()
    if item_type in {"text", "output_text", "input_text"}:
        text_value = getattr(value, "text", "")
        if hasattr(text_value, "value"):
            return _extract_text_fragment(getattr(text_value, "value"))
        return _extract_text_fragment(text_value)
    if hasattr(value, "content"):
        return _extract_text_fragment(getattr(value, "content"))
    if hasattr(value, "text"):
        return _extract_text_fragment(getattr(value, "text"))
    if hasattr(value, "value"):
        return _extract_text_fragment(getattr(value, "value"))
    return str(value)


def _message_text(message: object) -> str:
    """Extract plain text from a Groq/OpenAI-style assistant message object."""
    content = getattr(message, "content", "")
    return _extract_text_fragment(content)


def _message_attr(item: object, key: str) -> object:
    """Read an attribute or mapping key from mixed SDK/message objects."""
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def _normalize_role(role: object) -> str:
    """Map message roles into the subset supported by Twinr and Groq."""
    normalized = str(role or "").strip().lower()
    if normalized in {"system", "user", "assistant", "tool"}:
        return normalized
    return "user"


def _coerce_message(item: object) -> tuple[str, str]:
    """Normalize one conversation entry into ``(role, content)`` form."""
    if isinstance(item, tuple) and len(item) == 2:
        role, content = item
        return _normalize_role(role), _extract_text_fragment(content)
    role = _normalize_role(_message_attr(item, "role"))
    content = _extract_text_fragment(_message_attr(item, "content"))
    return role, content


def _sanitize_assistant_tool_calls(raw_tool_calls: object) -> list[dict[str, Any]]:
    """Convert prior assistant tool calls into protocol-safe dicts."""
    if not isinstance(raw_tool_calls, Sequence) or isinstance(raw_tool_calls, (str, bytes, bytearray)):
        return []
    sanitized: list[dict[str, Any]] = []
    for tool_call in raw_tool_calls:
        call_id = str(_message_attr(tool_call, "id") or "").strip()
        function = _message_attr(tool_call, "function")
        function_name = str(_message_attr(function, "name") or "").strip()
        raw_arguments = _message_attr(function, "arguments")
        if isinstance(raw_arguments, str):
            arguments_text = raw_arguments.strip() or "{}"
        else:
            try:
                arguments_text = json.dumps(raw_arguments if raw_arguments is not None else {}, ensure_ascii=False)
            except TypeError:
                arguments_text = "{}"
        if not call_id or not function_name:
            continue
        sanitized.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": arguments_text,
                },
            }
        )
    return sanitized


def _merge_system_message(messages: list[dict[str, Any]], instructions: str | None) -> list[dict[str, Any]]:
    """Merge extra system instructions into a copied message list."""
    merged = merge_instructions(instructions)
    if not merged:
        return copy.deepcopy(messages)
    updated = copy.deepcopy(messages)
    if updated and updated[0].get("role") == "system":
        updated[0]["content"] = merge_instructions(str(updated[0].get("content", "")).strip(), merged)
    else:
        updated.insert(0, {"role": "system", "content": merged})
    return updated


def _build_messages(
    config: TwinrConfig,
    prompt: str,
    *,
    conversation: ConversationLike | None = None,
    instructions: str | None = None,
) -> list[dict[str, Any]]:
    """Build a Groq-compatible message list from Twinr conversation state."""
    system_parts: list[str] = []
    base_instructions = merge_instructions(
        load_personality_instructions(config),
        instructions,
        user_response_language_instruction(config.openai_realtime_language),
    )
    if base_instructions:
        system_parts.append(base_instructions)
    messages: list[dict[str, Any]] = []
    if conversation:
        for item in conversation:
            role, content = _coerce_message(item)
            if role == "system":
                if content:
                    system_parts.append(content)
                continue
            message: dict[str, Any] = {"role": role, "content": content}
            if role == "assistant":
                assistant_tool_calls = _sanitize_assistant_tool_calls(_message_attr(item, "tool_calls"))
                if assistant_tool_calls:
                    message["tool_calls"] = assistant_tool_calls
                if not content and not assistant_tool_calls:
                    continue
            elif role == "tool":
                tool_call_id = str(_message_attr(item, "tool_call_id") or "").strip()
                if not tool_call_id:
                    logger.warning("Skipping tool conversation message without tool_call_id")
                    continue
                if not content:
                    continue
                message["tool_call_id"] = tool_call_id
            elif not content:
                continue
            messages.append(message)
    merged_instructions = merge_instructions(*system_parts)
    if merged_instructions:
        messages.insert(0, {"role": "system", "content": merged_instructions})
    prompt_text = prompt.strip()
    if prompt_text:
        messages.append({"role": "user", "content": prompt_text})
    return messages


def _normalize_parameters_schema(parameters: object) -> dict[str, Any]:
    """Return a safe function-parameters schema for Groq/OpenAI tool definitions."""
    if isinstance(parameters, dict):
        return parameters
    return {"type": "object", "properties": {}}


def _convert_tool_schemas(tool_schemas: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate Twinr tool schema definitions into Groq's request format."""
    converted: list[dict[str, Any]] = []
    for schema in tool_schemas:
        if not isinstance(schema, dict):
            continue
        schema_type = str(schema.get("type") or "").strip()
        if schema_type and schema_type != "function":
            if schema_type in {"browser_search", "code_interpreter"}:
                converted.append({"type": schema_type})
            continue
        nested_function = schema.get("function")
        if isinstance(nested_function, dict):
            name = str(nested_function.get("name") or "").strip()
            description = str(nested_function.get("description") or "").strip()
            parameters = _normalize_parameters_schema(nested_function.get("parameters"))
        else:
            name = str(schema.get("name") or "").strip()
            description = str(schema.get("description") or "").strip()
            parameters = _normalize_parameters_schema(schema.get("parameters"))
        if not name:
            logger.warning("Skipping tool schema without a valid function name")
            continue
        function_schema: dict[str, Any] = {
            "name": name,
            "parameters": parameters,
        }
        if description:
            function_schema["description"] = description
        converted.append(
            {
                "type": "function",
                "function": function_schema,
            }
        )
    return converted


def _tool_definition_key(tool_definition: Mapping[str, Any]) -> tuple[str, str]:
    """Return a stable deduplication key for one Groq request tool definition."""
    tool_type = str(tool_definition.get("type") or "").strip()
    if tool_type == "function":
        function = tool_definition.get("function")
        if isinstance(function, Mapping):
            return tool_type, str(function.get("name") or "").strip()
    return tool_type, ""


def _merge_request_tools(*tool_groups: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge multiple Groq request-tool lists while preserving order."""
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for group in tool_groups:
        for tool in group:
            if not isinstance(tool, dict):
                continue
            key = _tool_definition_key(tool)
            if key in seen:
                continue
            seen.add(key)
            merged.append(copy.deepcopy(tool))
    return merged


def _gpt_oss_builtin_tools_for_request(*, allow_web_search: bool) -> list[dict[str, Any]]:
    """Return GPT-OSS built-in tools to enable for a request."""
    tools: list[dict[str, Any]] = []
    if allow_web_search:
        tools.append({"type": "browser_search"})
    return tools


def _compound_web_search_options(config: TwinrConfig) -> tuple[str, dict[str, Any]]:
    """Return the model and request extras for a native Groq web-search text turn."""
    model = _text_search_model(config)
    request_extras: dict[str, Any] = {}
    if _model_is_compound(model):
        request_extras["compound_custom"] = {
            "tools": {
                "enabled_tools": ["web_search"],
            }
        }
    elif _model_supports_gpt_oss_builtin_tools(model):
        request_extras["tools"] = [{"type": "browser_search"}]
    return model, request_extras


def _parse_tool_arguments(raw_arguments: str) -> dict[str, Any] | None:
    """Parse model-emitted tool arguments and require a JSON object payload."""
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _truncate_text(text: str, *, max_chars: int, marker: str) -> str:
    """Truncate text without splitting the marker budget."""
    if len(text) <= max_chars:
        return text
    budget = max(0, max_chars - len(marker))
    return f"{text[:budget]}{marker}"


def _serialize_tool_output(serialized_output: object, *, config: TwinrConfig) -> str:
    """Serialize tool output into a protocol-safe bounded string for continuation turns."""
    if isinstance(serialized_output, str):
        text = serialized_output
    else:
        try:
            text = json.dumps(serialized_output, ensure_ascii=False)
        except TypeError:
            text = str(serialized_output)
    max_chars = _max_tool_result_chars(config)
    if len(text) <= max_chars:
        return text
    truncated_chars = len(text) - max_chars
    marker = (
        f"\n\n[tool output truncated by Twinr Groq adapter; {truncated_chars} trailing characters omitted "
        f"to stay within memory and token limits]"
    )
    return _truncate_text(text, max_chars=max_chars, marker=marker)


def _request_kwargs(
    config: TwinrConfig,
    *,
    model: str,
    messages: list[dict[str, Any]],
    stream: bool = False,
    tools: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Build shared Groq chat-completion request kwargs."""
    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if stream:
        request["stream"] = True
    service_tier = _groq_service_tier(config)
    if service_tier is not None:
        request["service_tier"] = service_tier
    reasoning_format = _groq_reasoning_format(config, tool_or_json_mode=bool(tools))
    if reasoning_format is not None:
        request["reasoning_format"] = reasoning_format
    reasoning_effort = _groq_reasoning_effort(config)
    if reasoning_effort is not None:
        request["reasoning_effort"] = reasoning_effort
    if tools:
        request["tools"] = list(tools)
        request["tool_choice"] = "auto"
    return request


def _exception_response_json(exc: Exception) -> dict[str, Any] | None:
    """Best-effort decode Groq error payloads from SDK exceptions."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    json_method = getattr(response, "json", None)
    if callable(json_method):
        try:
            parsed = json_method()
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    text = getattr(response, "text", None)
    if isinstance(text, str):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def _is_tool_use_failed_error(exc: Exception) -> bool:
    """Return whether an exception is Groq's known ``tool_use_failed`` 400 error."""
    if _safe_int(getattr(exc, "status_code", None)) != 400:
        return False
    payload = _exception_response_json(exc)
    if not isinstance(payload, dict):
        return False
    error = payload.get("error")
    if not isinstance(error, dict):
        return False
    return str(error.get("code") or "").strip().lower() == "tool_use_failed"


def _extract_executed_tools(message: object) -> tuple[dict[str, Any], ...]:
    """Normalize Groq built-in tool execution details from an assistant message."""
    raw = getattr(message, "executed_tools", None)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return ()
    normalized: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            normalized.append(copy.deepcopy(item))
            continue
        tool_dict: dict[str, Any] = {}
        for key in ("type", "arguments", "output", "index", "search_results"):
            value = getattr(item, key, None)
            if value is not None:
                tool_dict[key] = value
        if tool_dict:
            normalized.append(tool_dict)
    return tuple(normalized)


def _message_used_web_search(message: object) -> bool:
    """Return whether Groq reports server-side search execution for a message."""
    for item in _extract_executed_tools(message):
        tool_type = str(item.get("type") or "").strip().lower()
        if tool_type in {"search", "web_search", "browser_search"}:
            return True
    return False


def _run_chat_completion(
    client: Any,
    request: dict[str, Any],
    *,
    config: TwinrConfig,
) -> tuple[Any, object]:
    """Execute one non-streaming Groq chat completion and return ``(completion, message)``."""
    completion = _invoke_chat_completion(client, request, config=config)
    choices = getattr(completion, "choices", None) or []
    message = getattr(choices[0], "message", None) if choices else None
    if message is None:
        raise RuntimeError("Groq completion returned no assistant message")
    return completion, message


def _chat_result_from_completion(completion: Any, message: object) -> _ChatRunResult:
    """Convert a Groq completion + assistant message to a normalized chat result."""
    return _ChatRunResult(
        text=_message_text(message),
        response_id=str(getattr(completion, "id", "")).strip() or None,
        request_id=str(getattr(completion, "_request_id", "")).strip() or None,
        model=str(getattr(completion, "model", "")).strip() or None,
        token_usage=_chat_usage(completion),
        used_web_search=_message_used_web_search(message),
        executed_tools=_extract_executed_tools(message),
    )


def _coerce_image_url_from_like(image: ImageInputLike) -> str | None:
    """Best-effort normalize Twinr image inputs into Groq/OpenAI image URLs."""
    candidates: list[str | None] = []
    if isinstance(image, str):
        candidates.append(image)
    elif isinstance(image, Mapping):
        image_mapping = cast(Mapping[str, Any], image)
        candidates.extend(
            [
                cast(str | None, image_mapping.get("url")),
                cast(str | None, image_mapping.get("image_url")),
                cast(str | None, image_mapping.get("data_url")),
                cast(str | None, image_mapping.get("data_uri")),
            ]
        )
        base64_data = image_mapping.get("base64") or image_mapping.get("b64") or image_mapping.get("base64_data")
        mime_type = str(image_mapping.get("mime_type") or image_mapping.get("media_type") or "image/jpeg").strip()
        if isinstance(base64_data, str) and base64_data.strip():
            candidates.append(f"data:{mime_type};base64,{base64_data.strip()}")
        binary = image_mapping.get("bytes") or image_mapping.get("data")
        if isinstance(binary, (bytes, bytearray)):
            candidates.append(f"data:{mime_type};base64,{base64.b64encode(bytes(binary)).decode('ascii')}")
    else:
        candidates.extend(
            [
                cast(str | None, getattr(image, "url", None)),
                cast(str | None, getattr(image, "image_url", None)),
                cast(str | None, getattr(image, "data_url", None)),
                cast(str | None, getattr(image, "data_uri", None)),
            ]
        )
        base64_data = getattr(image, "base64", None) or getattr(image, "b64", None) or getattr(image, "base64_data", None)
        mime_type = str(getattr(image, "mime_type", None) or getattr(image, "media_type", None) or "image/jpeg").strip()
        if isinstance(base64_data, str) and base64_data.strip():
            candidates.append(f"data:{mime_type};base64,{base64_data.strip()}")
        binary = getattr(image, "bytes", None) or getattr(image, "data", None)
        if isinstance(binary, (bytes, bytearray)):
            candidates.append(f"data:{mime_type};base64,{base64.b64encode(bytes(binary)).decode('ascii')}")
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _build_image_messages(
    config: TwinrConfig,
    prompt: str,
    *,
    images: Sequence[ImageInputLike],
    conversation: ConversationLike | None = None,
    instructions: str | None = None,
) -> list[dict[str, Any]] | None:
    """Build a Groq-compatible multimodal message list, or ``None`` if inputs are unsupported."""
    messages = _build_messages(
        config,
        "",
        conversation=conversation,
        instructions=instructions,
    )
    content: list[dict[str, Any]] = []
    prompt_text = prompt.strip()
    if prompt_text:
        content.append({"type": "text", "text": prompt_text})
    for image in images:
        url = _coerce_image_url_from_like(image)
        if url is None:
            return None
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": url,
                },
            }
        )
    if not content:
        return None
    messages.append({"role": "user", "content": content})
    return messages


@dataclass
class GroqAgentTextProvider:
    """Serve text turns through Groq with bounded fallbacks and native search/vision support."""

    config: TwinrConfig
    support_provider: AgentTextProvider
    client: Any | None = None
    _client: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Create a validated Groq client when tests did not inject one."""
        self._client = self.client or default_groq_client(self.config)

    def _fallback_streaming(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> TextResponse:
        """Delegate a streaming turn to the support provider after Groq failure."""
        try:
            return self.support_provider.respond_streaming(
                prompt,
                conversation=conversation,
                instructions=instructions,
                allow_web_search=allow_web_search,
                on_text_delta=on_text_delta,
            )
        except Exception:
            logger.exception("Support provider streaming fallback failed")
            fallback_text = _text_provider_error_text(self.config)
            _safe_emit_text(on_text_delta, fallback_text)
            return _as_text_response(
                GroqTextResponse(
                    text=fallback_text,
                    model=self.config.groq_model,
                )
            )

    def _fallback_with_metadata(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
    ) -> TextResponse:
        """Delegate a non-streaming text turn to the support provider."""
        try:
            return self.support_provider.respond_with_metadata(
                prompt,
                conversation=conversation,
                instructions=instructions,
                allow_web_search=allow_web_search,
            )
        except Exception:
            logger.exception("Support provider metadata fallback failed")
            return _as_text_response(
                GroqTextResponse(
                    text=_text_provider_error_text(self.config),
                    model=self.config.groq_model,
                )
            )

    def _groq_search_response(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
    ) -> TextResponse:
        """Run a native Groq live-search turn using built-in server-side tools."""
        search_model, extra_request = _compound_web_search_options(self.config)
        request_timeout_seconds = _groq_request_timeout_seconds(self.config)
        owned_client_ref: dict[str, Any] = {}

        def _run_search_request() -> tuple[Any, Any]:
            request = _request_kwargs(
                self.config,
                model=search_model,
                messages=_build_messages(
                    self.config,
                    prompt,
                    conversation=conversation,
                    instructions=instructions,
                ),
            )
            request.update(extra_request)
            request_client = self._client
            if self.client is None:
                request_client = default_groq_client(self.config)
                owned_client_ref["client"] = request_client
            try:
                return _run_chat_completion(request_client, request, config=self.config)
            finally:
                if self.client is None:
                    close = getattr(request_client, "close", None)
                    if callable(close):
                        try:
                            close()
                        except Exception:
                            logger.debug("Failed to close one-off Groq search client", exc_info=True)
        try:
            completion, message = cast(
                tuple[Any, Any],
                _run_callable_with_wall_clock_timeout(
                    _run_search_request,
                    timeout_seconds=request_timeout_seconds,
                    action_label="native Groq search",
                ),
            )
            result = _chat_result_from_completion(completion, message)
            return _as_text_response(
                GroqTextResponse(
                    text=result.text,
                    response_id=result.response_id,
                    request_id=result.request_id,
                    model=result.model or search_model,
                    token_usage=result.token_usage,
                    used_web_search=True,
                )
            )
        except Exception:
            close = getattr(owned_client_ref.get("client"), "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.debug("Failed to force-close timed-out Groq search client", exc_info=True)
            logger.warning("Native Groq search request failed", exc_info=True)
            if _allow_search_fallback(self.config):
                return self._fallback_with_metadata(
                    prompt,
                    conversation=conversation,
                    instructions=instructions,
                    allow_web_search=True,
                )
            return _as_text_response(
                GroqTextResponse(
                    text=_text_provider_error_text(self.config),
                    model=search_model,
                )
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
        """Stream a text response from Groq or fall back to the support provider."""
        if allow_web_search:
            response = self._groq_search_response(
                prompt,
                conversation=conversation,
                instructions=instructions,
            )
            _safe_emit_text(on_text_delta, response.text)
            return response
        request = _request_kwargs(
            self.config,
            model=self.config.groq_model,
            messages=_build_messages(
                self.config,
                prompt,
                conversation=conversation,
                instructions=instructions,
            ),
            stream=True,
        )
        text_fragments: list[str] = []
        response_id: str | None = None
        request_id: str | None = None
        model = self.config.groq_model
        try:
            stream = _invoke_chat_completion(self._client, request, config=self.config)
        except Exception:
            logger.warning("Groq streaming request failed before any data; using support provider fallback", exc_info=True)
            return self._fallback_streaming(
                prompt,
                conversation=conversation,
                instructions=instructions,
                allow_web_search=allow_web_search,
                on_text_delta=on_text_delta,
            )
        try:
            for chunk in stream:
                if response_id is None:
                    response_id = str(getattr(chunk, "id", "")).strip() or None
                    request_id = str(getattr(chunk, "_request_id", "")).strip() or None
                    model = str(getattr(chunk, "model", "")).strip() or model
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                content = getattr(delta, "content", None)
                if not content:
                    continue
                delta_text = _extract_text_fragment(content)
                if not delta_text:
                    continue
                text_fragments.append(delta_text)
                _safe_emit_text(on_text_delta, delta_text)
        except Exception:
            if text_fragments:
                logger.warning("Groq streaming response interrupted after partial output", exc_info=True)
            else:
                logger.warning("Groq streaming response failed before any text; using support provider fallback", exc_info=True)
                return self._fallback_streaming(
                    prompt,
                    conversation=conversation,
                    instructions=instructions,
                    allow_web_search=allow_web_search,
                    on_text_delta=on_text_delta,
                )
        return _as_text_response(
            GroqTextResponse(
                text="".join(text_fragments),
                response_id=response_id,
                request_id=request_id,
                model=model,
            )
        )

    def respond_with_metadata(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
    ) -> TextResponse:
        """Run a non-streaming Groq text turn and preserve response metadata."""
        if allow_web_search:
            return self._groq_search_response(
                prompt,
                conversation=conversation,
                instructions=instructions,
            )
        try:
            completion, message = _run_chat_completion(
                self._client,
                _request_kwargs(
                    self.config,
                    model=self.config.groq_model,
                    messages=_build_messages(
                        self.config,
                        prompt,
                        conversation=conversation,
                        instructions=instructions,
                    ),
                ),
                config=self.config,
            )
        except Exception:
            logger.warning("Groq metadata request failed; using support provider fallback", exc_info=True)
            return self._fallback_with_metadata(
                prompt,
                conversation=conversation,
                instructions=instructions,
                allow_web_search=allow_web_search,
            )
        result = _chat_result_from_completion(completion, message)
        return _as_text_response(
            GroqTextResponse(
                text=result.text,
                response_id=result.response_id,
                request_id=result.request_id,
                model=result.model or self.config.groq_model,
                token_usage=result.token_usage,
                used_web_search=result.used_web_search,
            )
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
        """Run image turns on Groq vision first, then fall back only if explicitly allowed."""
        messages = _build_image_messages(
            self.config,
            prompt,
            images=images,
            conversation=conversation,
            instructions=instructions,
        )
        if messages is None:
            if _allow_vision_fallback(self.config):
                return self.support_provider.respond_to_images_with_metadata(
                    prompt,
                    images=images,
                    conversation=conversation,
                    instructions=instructions,
                    allow_web_search=allow_web_search,
                )
            return _as_text_response(
                GroqTextResponse(
                    text=_text_provider_error_text(self.config),
                    model=_vision_model(self.config),
                )
            )
        request = _request_kwargs(
            self.config,
            model=_vision_model(self.config),
            messages=messages,
        )
        try:
            completion, message = _run_chat_completion(self._client, request, config=self.config)
            result = _chat_result_from_completion(completion, message)
            return _as_text_response(
                GroqTextResponse(
                    text=result.text,
                    response_id=result.response_id,
                    request_id=result.request_id,
                    model=result.model or _vision_model(self.config),
                    token_usage=result.token_usage,
                    used_web_search=result.used_web_search,
                )
            )
        except Exception:
            logger.warning("Groq vision request failed", exc_info=True)
            if _allow_vision_fallback(self.config):
                return self.support_provider.respond_to_images_with_metadata(
                    prompt,
                    images=images,
                    conversation=conversation,
                    instructions=instructions,
                    allow_web_search=allow_web_search,
                )
            return _as_text_response(
                GroqTextResponse(
                    text=_text_provider_error_text(self.config),
                    model=_vision_model(self.config),
                )
            )

    def search_live_info_with_metadata(
        self,
        question: str,
        *,
        conversation: ConversationLike | None = None,
        location_hint: str | None = None,
        date_context: str | None = None,
    ) -> SearchResponse:
        """Delegate live-search turns to the support provider."""
        return self.support_provider.search_live_info_with_metadata(
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
        """Delegate print-job phrasing to the support provider."""
        return self.support_provider.compose_print_job_with_metadata(
            conversation=conversation,
            focus_hint=focus_hint,
            direct_text=direct_text,
            request_source=request_source,
        )

    def phrase_due_reminder_with_metadata(self, reminder: object, *, now=None) -> TextResponse:
        """Delegate due-reminder phrasing to the support provider."""
        return self.support_provider.phrase_due_reminder_with_metadata(reminder, now=now)

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
        """Delegate proactive prompt phrasing to the support provider."""
        return self.support_provider.phrase_proactive_prompt_with_metadata(
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
        """Delegate automation fulfillment phrasing to the support provider."""
        return self.support_provider.fulfill_automation_prompt_with_metadata(
            prompt,
            allow_web_search=allow_web_search,
            delivery=delivery,
        )


@dataclass
class GroqToolCallingAgentProvider:
    """Run Groq tool-calling turns with bounded continuation state."""

    config: TwinrConfig
    client: Any | None = None
    _client: Any = field(init=False, repr=False)
    _continuations: dict[str, _ContinuationState] = field(init=False, repr=False)
    _lock: Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Create a validated Groq client and initialize continuation storage."""
        self._client = self.client or default_groq_client(self.config)
        self._continuations = {}
        self._lock = Lock()

    def _error_response(
        self,
        text: str,
        *,
        continuation_token: str | None = None,
        used_web_search: bool = False,
    ) -> ToolCallingTurnResponse:
        """Build a normalized tool-calling error response."""
        return ToolCallingTurnResponse(
            text=text,
            tool_calls=tuple(),
            response_id=None,
            request_id=None,
            model=self.config.groq_model,
            token_usage=None,
            used_web_search=used_web_search,
            continuation_token=continuation_token,
        )

    def _purge_continuations_locked(self, now: float | None = None) -> None:
        """Drop expired or excess continuation entries while holding ``_lock``."""
        if now is None:
            now = time.monotonic()
        ttl_seconds = _continuation_ttl_seconds(self.config)
        max_items = _max_continuations(self.config)
        expired_tokens = [
            token
            for token, state in self._continuations.items()
            if (now - state.last_access_monotonic) > ttl_seconds
        ]
        for token in expired_tokens:
            self._continuations.pop(token, None)
        overflow = len(self._continuations) - max_items
        if overflow > 0:
            oldest_tokens = sorted(
                self._continuations.items(),
                key=lambda item: (item[1].last_access_monotonic, item[1].created_monotonic),
            )[:overflow]
            for token, _state in oldest_tokens:
                self._continuations.pop(token, None)

    def _reserve_continuation(self, continuation_token: str) -> _ContinuationState | None:
        """Reserve a continuation token for exclusive reuse by one caller."""
        with self._lock:
            self._purge_continuations_locked()
            state = self._continuations.get(continuation_token)
            if state is None:
                return None
            if state.in_flight:
                return _ContinuationState(messages=[], expected_tool_call_ids=(), in_flight=True)
            state.in_flight = True
            state.last_access_monotonic = time.monotonic()
            return copy.deepcopy(state)

    def _release_continuation(
        self,
        continuation_token: str,
        *,
        keep_state: bool,
        state: _ContinuationState | None = None,
    ) -> None:
        """Release or replace a continuation entry after one continuation attempt."""
        with self._lock:
            current = self._continuations.get(continuation_token)
            if current is None:
                return
            if keep_state:
                current.in_flight = False
                current.last_access_monotonic = time.monotonic()
                if state is not None:
                    current.messages = state.messages
                    current.expected_tool_call_ids = state.expected_tool_call_ids
                    current.request_tools = state.request_tools
                    current.created_monotonic = state.created_monotonic
            else:
                self._continuations.pop(continuation_token, None)

    def _validate_and_order_tool_results(
        self,
        tool_results: Sequence[AgentToolResult],
        expected_tool_call_ids: tuple[str, ...],
    ) -> tuple[list[AgentToolResult] | None, str | None]:
        """Verify that returned tool results exactly match the expected call IDs."""
        if not expected_tool_call_ids:
            return None, "no expected tool calls"
        by_id: dict[str, AgentToolResult] = {}
        for result in tool_results:
            call_id = str(getattr(result, "call_id", "") or "").strip()
            if not call_id:
                return None, "missing call_id"
            if call_id in by_id:
                return None, "duplicate call_id"
            by_id[call_id] = result
        if set(by_id) != set(expected_tool_call_ids):
            return None, "tool call ids do not match expected continuation state"
        ordered_results = [by_id[call_id] for call_id in expected_tool_call_ids]
        return ordered_results, None

    def _effective_request_tools(
        self,
        *,
        tool_schemas: Sequence[dict[str, Any]],
        allow_web_search: bool | None,
    ) -> list[dict[str, Any]]:
        """Build the request tools for one tool-calling completion."""
        converted_local_tools = _convert_tool_schemas(tool_schemas)
        builtin_tools: list[dict[str, Any]] = []
        if allow_web_search and _model_supports_gpt_oss_builtin_tools(self.config.groq_model):
            builtin_tools = _gpt_oss_builtin_tools_for_request(allow_web_search=True)
        return _merge_request_tools(converted_local_tools, builtin_tools)

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
        """Start a Groq tool-calling turn and persist continuation state if needed."""
        messages = _build_messages(
            self.config,
            prompt,
            conversation=conversation,
            instructions=instructions,
        )
        request_tools = self._effective_request_tools(
            tool_schemas=tool_schemas,
            allow_web_search=allow_web_search,
        )
        try:
            response, assistant_message, request_tools = self._run_tool_completion(
                messages,
                request_tools=request_tools,
            )
        except Exception:
            logger.warning("Groq tool start request failed", exc_info=True)
            fallback = self._error_response(_tool_provider_error_text(self.config))
            _safe_emit_text(on_text_delta, fallback.text)
            return fallback
        if response.text and on_text_delta is not None and not response.tool_calls:
            _safe_emit_text(on_text_delta, response.text)
        if response.tool_calls:
            token = uuid4().hex
            now = time.monotonic()
            with self._lock:
                self._purge_continuations_locked(now)
                self._continuations[token] = _ContinuationState(
                    messages=copy.deepcopy([*messages, assistant_message]),
                    expected_tool_call_ids=tuple(call.call_id for call in response.tool_calls),
                    request_tools=tuple(copy.deepcopy(request_tools)),
                    created_monotonic=now,
                    last_access_monotonic=now,
                    in_flight=False,
                )
            return ToolCallingTurnResponse(
                text=response.text,
                tool_calls=response.tool_calls,
                response_id=response.response_id,
                request_id=response.request_id,
                model=response.model,
                token_usage=response.token_usage,
                used_web_search=response.used_web_search,
                continuation_token=token,
            )
        return ToolCallingTurnResponse(
            text=response.text,
            tool_calls=(),
            response_id=response.response_id,
            request_id=response.request_id,
            model=response.model,
            token_usage=response.token_usage,
            used_web_search=response.used_web_search,
        )

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
        """Resume a Groq tool-calling turn after Twinr executes the tool results."""
        reserved_state = self._reserve_continuation(continuation_token)
        if reserved_state is None:
            return self._error_response(_tool_continuation_expired_text(self.config))
        if reserved_state.in_flight and not reserved_state.messages:
            return self._error_response(
                _tool_continuation_busy_text(self.config),
                continuation_token=continuation_token,
            )
        messages = reserved_state.messages
        if instructions:
            messages = _merge_system_message(messages, instructions)
        continuation_tools = list(reserved_state.request_tools)
        if tool_schemas or (allow_web_search and _model_supports_gpt_oss_builtin_tools(self.config.groq_model)):
            continuation_tools = _merge_request_tools(
                continuation_tools,
                self._effective_request_tools(
                    tool_schemas=tool_schemas,
                    allow_web_search=allow_web_search,
                ),
            )
        ordered_tool_results, validation_error = self._validate_and_order_tool_results(
            tool_results,
            reserved_state.expected_tool_call_ids,
        )
        if validation_error is not None or ordered_tool_results is None:
            logger.warning("Rejected mismatched Groq tool continuation results: %s", validation_error)
            self._release_continuation(continuation_token, keep_state=True)
            return self._error_response(
                _tool_result_error_text(self.config),
                continuation_token=continuation_token,
            )
        for result in ordered_tool_results:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result.call_id,
                    "content": _serialize_tool_output(result.serialized_output, config=self.config),
                }
            )
        try:
            response, assistant_message, continuation_tools = self._run_tool_completion(
                messages,
                request_tools=continuation_tools,
            )
        except Exception:
            logger.warning("Groq tool continuation request failed", exc_info=True)
            self._release_continuation(continuation_token, keep_state=True)
            fallback = self._error_response(
                _tool_provider_error_text(self.config),
                continuation_token=continuation_token,
            )
            _safe_emit_text(on_text_delta, fallback.text)
            return fallback
        if response.text and on_text_delta is not None and not response.tool_calls:
            _safe_emit_text(on_text_delta, response.text)
        if response.tool_calls:
            updated_state = _ContinuationState(
                messages=copy.deepcopy([*messages, assistant_message]),
                expected_tool_call_ids=tuple(call.call_id for call in response.tool_calls),
                request_tools=tuple(copy.deepcopy(continuation_tools)),
                created_monotonic=reserved_state.created_monotonic,
                last_access_monotonic=time.monotonic(),
                in_flight=False,
            )
            self._release_continuation(continuation_token, keep_state=True, state=updated_state)
        else:
            self._release_continuation(continuation_token, keep_state=False)
        return ToolCallingTurnResponse(
            text=response.text,
            tool_calls=response.tool_calls,
            response_id=response.response_id,
            request_id=response.request_id,
            model=response.model,
            token_usage=response.token_usage,
            used_web_search=response.used_web_search,
            continuation_token=continuation_token if response.tool_calls else None,
        )

    def _run_tool_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        request_tools: Sequence[dict[str, Any]],
        _repair_attempted: bool = False,
    ) -> tuple[_ToolCompletionPayload, dict[str, Any], list[dict[str, Any]]]:
        """Execute one Groq tool-completion request and normalize its result."""
        request = _request_kwargs(
            self.config,
            model=self.config.groq_model,
            messages=messages,
            tools=request_tools,
        )
        request_tools_list = list(request_tools)
        try:
            completion, message = _run_chat_completion(self._client, request, config=self.config)
        except Exception as exc:
            if (
                request_tools_list
                and not _repair_attempted
                and _is_tool_use_failed_error(exc)
            ):
                repaired_messages = _merge_system_message(messages, _LOCAL_TOOL_REPAIR_INSTRUCTION)
                return self._run_tool_completion(
                    repaired_messages,
                    request_tools=request_tools_list,
                    _repair_attempted=True,
                )
            raise
        response_id = str(getattr(completion, "id", "")).strip() or None
        request_id = str(getattr(completion, "_request_id", "")).strip() or None
        model = str(getattr(completion, "model", "")).strip() or self.config.groq_model
        tool_calls: list[AgentToolCall] = []
        assistant_tool_calls: list[dict[str, Any]] = []
        for tool_call in getattr(message, "tool_calls", None) or []:
            call_id = str(getattr(tool_call, "id", "")).strip()
            function = getattr(tool_call, "function", None)
            function_name = str(getattr(function, "name", "")).strip()
            raw_arguments = str(getattr(function, "arguments", "") or "{}").strip() or "{}"
            arguments = _parse_tool_arguments(raw_arguments)
            if arguments is None:
                raise RuntimeError("Groq tool arguments are not valid JSON objects")
            if not call_id or not function_name:
                continue
            tool_calls.append(
                AgentToolCall(
                    name=function_name,
                    call_id=call_id,
                    arguments=arguments,
                    raw_arguments=raw_arguments,
                )
            )
            assistant_tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": raw_arguments,
                    },
                }
            )
        text = _message_text(message)
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": text,
        }
        if assistant_tool_calls:
            assistant_message["tool_calls"] = assistant_tool_calls
        return (
            _ToolCompletionPayload(
                text=text,
                tool_calls=tuple(tool_calls),
                response_id=response_id,
                request_id=request_id,
                model=model,
                token_usage=_chat_usage(completion),
                used_web_search=_message_used_web_search(message),
            ),
            assistant_message,
            request_tools_list,
        )
