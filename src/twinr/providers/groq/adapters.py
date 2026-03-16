"""Provide Groq-backed text and tool-calling adapters for Twinr.

This module translates Twinr's agent-provider contracts into Groq chat
completion requests. It also owns the continuation state required to resume
tool-calling turns after Twinr executes the requested tools locally.
"""

from __future__ import annotations
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, Sequence
from uuid import uuid4
import copy
import json
import logging
import time
from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.contracts import (
    AgentToolCall,
    AgentToolResult,
    AgentTextProvider,
    ConversationLike,
    ImageInputLike,
    SearchResponse,
    TextResponse,
    ToolCallingAgentProvider,
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
_DEFAULT_GROQ_RETRY_BACKOFF_SECONDS = 0.35
_DEFAULT_CONTINUATION_TTL_SECONDS = 300.0
_DEFAULT_MAX_CONTINUATIONS = 128
@dataclass(slots=True)
class _ContinuationState:
    """Track the pending tool-calling state for one Groq continuation token."""

    messages: list[dict[str, Any]]
    expected_tool_call_ids: tuple[str, ...]
    created_monotonic: float
    last_access_monotonic: float
    in_flight: bool = False
def _safe_int(value: object) -> int | None:
    """Best-effort convert a value to ``int`` without raising."""
    if value is None:
        return None
    try:
        return int(value)
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
        coerced = float(value)
    except (TypeError, ValueError):
        return default
    if coerced <= 0:
        return default
    return coerced
def _groq_request_timeout_seconds(config: TwinrConfig) -> float | None:
    """Read the per-request Groq timeout override from config."""
    # AUDIT-FIX(#1): Use a bounded request timeout with a safe default, while remaining backward compatible with configs that omit the key.
    return _positive_float(
        getattr(config, "groq_request_timeout_seconds", _DEFAULT_GROQ_REQUEST_TIMEOUT_SECONDS),
        _DEFAULT_GROQ_REQUEST_TIMEOUT_SECONDS,
    )
def _groq_total_attempts(config: TwinrConfig) -> int:
    """Return the total number of Groq request attempts including retries."""
    # AUDIT-FIX(#1): Bound retries so intermittent Wi-Fi does not turn into an unbounded blocking loop.
    retries = _non_negative_int(getattr(config, "groq_max_retries", _DEFAULT_GROQ_MAX_RETRIES), _DEFAULT_GROQ_MAX_RETRIES)
    return retries + 1
def _groq_retry_backoff_seconds(config: TwinrConfig) -> float:
    """Read the retry backoff used between Groq request attempts."""
    # AUDIT-FIX(#1): Back off briefly between retries to avoid hammering a flaky network or provider edge.
    return _positive_float(
        getattr(config, "groq_retry_backoff_seconds", _DEFAULT_GROQ_RETRY_BACKOFF_SECONDS),
        _DEFAULT_GROQ_RETRY_BACKOFF_SECONDS,
    ) or _DEFAULT_GROQ_RETRY_BACKOFF_SECONDS
def _continuation_ttl_seconds(config: TwinrConfig) -> float:
    """Read the maximum idle age for stored continuation state."""
    # AUDIT-FIX(#3): Expire abandoned continuations so the long-running process does not leak memory indefinitely.
    return _positive_float(
        getattr(config, "groq_tool_continuation_ttl_seconds", _DEFAULT_CONTINUATION_TTL_SECONDS),
        _DEFAULT_CONTINUATION_TTL_SECONDS,
    ) or _DEFAULT_CONTINUATION_TTL_SECONDS
def _max_continuations(config: TwinrConfig) -> int:
    """Read the cap for simultaneously stored continuation states."""
    # AUDIT-FIX(#3): Cap in-memory continuations to stay within RPi 4 memory constraints during long uptimes.
    return max(1, _non_negative_int(getattr(config, "groq_tool_max_continuations", _DEFAULT_MAX_CONTINUATIONS), _DEFAULT_MAX_CONTINUATIONS))
def _text_provider_error_text(config: TwinrConfig) -> str:
    """Return the user-facing fallback text for Groq text failures."""
    # AUDIT-FIX(#4): Provide a simple, configurable fallback utterance instead of surfacing raw provider failures to the user.
    text = str(getattr(config, "groq_text_provider_error_text", "I am having trouble right now. Please try again.")).strip()
    return text or "I am having trouble right now. Please try again."
def _tool_provider_error_text(config: TwinrConfig) -> str:
    """Return the user-facing fallback text for Groq tool-call failures."""
    # AUDIT-FIX(#4): Keep tool-provider failures recoverable and senior-friendly instead of crashing the request path.
    text = str(getattr(config, "groq_tool_provider_error_text", "I could not finish that step. Please try again.")).strip()
    return text or "I could not finish that step. Please try again."
def _tool_continuation_expired_text(config: TwinrConfig) -> str:
    """Return the text used when a continuation token is no longer valid."""
    # AUDIT-FIX(#2): Return a controlled message when the continuation state is gone, rather than throwing a raw runtime exception.
    text = str(getattr(config, "groq_tool_continuation_expired_text", "I lost the previous step. Please try again.")).strip()
    return text or "I lost the previous step. Please try again."
def _tool_continuation_busy_text(config: TwinrConfig) -> str:
    """Return the text used when a continuation token is already in flight."""
    # AUDIT-FIX(#2): Reject concurrent reuse of the same continuation token deterministically.
    text = str(getattr(config, "groq_tool_continuation_busy_text", "That step is already being finished. Please try again in a moment.")).strip()
    return text or "That step is already being finished. Please try again in a moment."
def _tool_result_error_text(config: TwinrConfig) -> str:
    """Return the text used when tool results do not match expected calls."""
    # AUDIT-FIX(#2): Keep malformed or mismatched tool-result handoffs recoverable without corrupting conversation state.
    text = str(getattr(config, "groq_tool_result_error_text", "I could not verify the tool results. Please try again.")).strip()
    return text or "I could not verify the tool results. Please try again."
def _safe_emit_text(on_text_delta: Callable[[str], None] | None, text: str) -> None:
    """Emit streaming text to a callback without letting the callback raise."""
    if on_text_delta is None or not text:
        return
    try:
        on_text_delta(text)
    except Exception:
        # AUDIT-FIX(#4): Do not let downstream audio/render callbacks crash an otherwise valid model response.
        logger.exception("Groq text delta callback failed")
def _invoke_chat_completion(
    client: Any,
    request: dict[str, Any],
    *,
    config: TwinrConfig,
) -> Any:
    """Call Groq chat completions with bounded retries and optional timeout."""
    use_timeout = _groq_request_timeout_seconds(config) is not None
    timeout_seconds = _groq_request_timeout_seconds(config)
    total_attempts = _groq_total_attempts(config)
    backoff_seconds = _groq_retry_backoff_seconds(config)
    last_error: Exception | None = None
    base_request = dict(request)
    for attempt in range(1, total_attempts + 1):
        payload = dict(base_request)
        if use_timeout and timeout_seconds is not None:
            # AUDIT-FIX(#1): Apply a per-request timeout when the SDK supports it.
            payload["timeout"] = timeout_seconds
        try:
            return client.chat.completions.create(**payload)
        except TypeError as exc:
            if use_timeout and "timeout" in str(exc).lower():
                logger.debug("Groq SDK rejected per-request timeout; retrying without timeout keyword")
                use_timeout = False
                last_error = exc
            else:
                last_error = exc
        except Exception as exc:
            last_error = exc
        if attempt < total_attempts:
            time.sleep(backoff_seconds * attempt)
    raise RuntimeError("Groq chat completion request failed") from last_error
def _chat_usage(source: object) -> TokenUsage | None:
    """Convert provider usage metadata into Twinr's ``TokenUsage`` contract."""
    usage = getattr(source, "usage", None)
    if usage is None:
        return None
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    token_usage = TokenUsage(
        # AUDIT-FIX(#7): Tolerate malformed provider usage metadata instead of crashing after an otherwise successful completion.
        input_tokens=_safe_int(prompt_tokens),
        # AUDIT-FIX(#7): Tolerate malformed provider usage metadata instead of crashing after an otherwise successful completion.
        output_tokens=_safe_int(completion_tokens),
        # AUDIT-FIX(#7): Tolerate malformed provider usage metadata instead of crashing after an otherwise successful completion.
        total_tokens=_safe_int(total_tokens),
    )
    return token_usage if token_usage.has_values else None
def _extract_text_fragment(value: object) -> str:
    """Normalize nested SDK content blocks into plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        fragments = [_extract_text_fragment(item) for item in value]
        return "\n".join(fragment for fragment in fragments if fragment).strip()
    if isinstance(value, dict):
        item_type = str(value.get("type", "")).strip().lower()
        if item_type == "text":
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
    if item_type == "text":
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
    return str(value).strip()
def _message_text(message: object) -> str:
    """Extract plain text from a Groq/OpenAI-style assistant message object."""
    # AUDIT-FIX(#7): Extract text recursively from typed SDK content blocks instead of dropping or repr()-stringifying them.
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
                # AUDIT-FIX(#5): Preserve assistant tool_calls in history so tool-using turns remain protocol-correct on subsequent requests.
                assistant_tool_calls = _sanitize_assistant_tool_calls(_message_attr(item, "tool_calls"))
                if assistant_tool_calls:
                    message["tool_calls"] = assistant_tool_calls
                if not content and not assistant_tool_calls:
                    continue
            elif role == "tool":
                # AUDIT-FIX(#5): Preserve tool-role history instead of silently downgrading it to user text.
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
def _convert_tool_schemas(tool_schemas: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate Twinr tool schema definitions into Groq's request format."""
    converted: list[dict[str, Any]] = []
    for schema in tool_schemas:
        if not isinstance(schema, dict) or schema.get("type") != "function":
            continue
        name = str(schema.get("name") or "").strip()
        if not name:
            logger.warning("Skipping tool schema without a valid function name")
            continue
        parameters = schema.get("parameters")
        if not isinstance(parameters, dict):
            # AUDIT-FIX(#6): Normalize invalid parameter schemas so one malformed tool definition does not break the entire request.
            parameters = {"type": "object", "properties": {}}
        function_schema: dict[str, Any] = {
            "name": name,
            "parameters": parameters,
        }
        description = str(schema.get("description") or "").strip()
        if description:
            function_schema["description"] = description
        converted.append(
            {
                "type": "function",
                "function": function_schema,
            }
        )
    return converted
def _parse_tool_arguments(raw_arguments: str) -> dict[str, Any] | None:
    """Parse model-emitted tool arguments and require a JSON object payload."""
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed
def _serialize_tool_output(serialized_output: object) -> str:
    """Serialize tool output into a protocol-safe string for continuation turns."""
    if isinstance(serialized_output, str):
        return serialized_output
    try:
        return json.dumps(serialized_output, ensure_ascii=False)
    except TypeError:
        return str(serialized_output)
@dataclass
class GroqAgentTextProvider:
    """Serve text turns through Groq with OpenAI-based fallback paths."""

    config: TwinrConfig
    support_provider: AgentTextProvider
    client: Any | None = None
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
            # AUDIT-FIX(#1): Fall back to the support provider when Groq is unavailable so the senior does not hit a dead end on transient outages.
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
            return GroqTextResponse(
                text=fallback_text,
                model=self.config.groq_model,
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
            # AUDIT-FIX(#1): Fall back to the support provider when Groq fails on non-streaming turns as well.
            return self.support_provider.respond_with_metadata(
                prompt,
                conversation=conversation,
                instructions=instructions,
                allow_web_search=allow_web_search,
            )
        except Exception:
            logger.exception("Support provider metadata fallback failed")
            return GroqTextResponse(
                text=_text_provider_error_text(self.config),
                model=self.config.groq_model,
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
        """Stream a text response from Groq or fall back to the support provider.

        Args:
            prompt: User prompt for the current turn.
            conversation: Optional prior conversation items for context.
            instructions: Optional system-level instruction overrides.
            allow_web_search: Whether the turn requires live search support.
            on_text_delta: Optional callback for streamed text fragments.

        Returns:
            A normalized text response compatible with Twinr's provider contract.
        """
        if allow_web_search:
            return self._fallback_streaming(
                prompt,
                conversation=conversation,
                instructions=instructions,
                allow_web_search=allow_web_search,
                on_text_delta=on_text_delta,
            )
        request = {
            "model": self.config.groq_model,
            "messages": _build_messages(
                self.config,
                prompt,
                conversation=conversation,
                instructions=instructions,
            ),
            "stream": True,
        }
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
                # AUDIT-FIX(#7): Normalize structured streaming deltas instead of leaking Python repr strings into speech/output.
                delta_text = _extract_text_fragment(content)
                if not delta_text:
                    continue
                text_fragments.append(delta_text)
                _safe_emit_text(on_text_delta, delta_text)
        except Exception:
            if text_fragments:
                # AUDIT-FIX(#4): Preserve already-streamed content if the provider drops mid-response instead of converting the whole turn into an exception.
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
        return GroqTextResponse(
            text="".join(text_fragments).strip(),
            response_id=response_id,
            request_id=request_id,
            model=model,
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
            return self._fallback_with_metadata(
                prompt,
                conversation=conversation,
                instructions=instructions,
                allow_web_search=allow_web_search,
            )
        try:
            completion = _invoke_chat_completion(
                self._client,
                {
                    "model": self.config.groq_model,
                    "messages": _build_messages(
                        self.config,
                        prompt,
                        conversation=conversation,
                        instructions=instructions,
                    ),
                },
                config=self.config,
            )
            choice = (getattr(completion, "choices", None) or [None])[0]
            message = getattr(choice, "message", None)
            if message is None:
                raise RuntimeError("Groq completion returned no assistant message")
        except Exception:
            logger.warning("Groq metadata request failed; using support provider fallback", exc_info=True)
            return self._fallback_with_metadata(
                prompt,
                conversation=conversation,
                instructions=instructions,
                allow_web_search=allow_web_search,
            )
        return GroqTextResponse(
            text=_message_text(message),
            response_id=str(getattr(completion, "id", "")).strip() or None,
            request_id=str(getattr(completion, "_request_id", "")).strip() or None,
            model=str(getattr(completion, "model", "")).strip() or self.config.groq_model,
            token_usage=_chat_usage(completion),
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
        """Delegate image turns to the support provider."""
        return self.support_provider.respond_to_images_with_metadata(
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
    def __post_init__(self) -> None:
        """Create a validated Groq client and initialize continuation storage."""
        self._client = self.client or default_groq_client(self.config)
        self._continuations: dict[str, _ContinuationState] = {}
        self._lock = Lock()
    def _error_response(
        self,
        text: str,
        *,
        continuation_token: str | None = None,
    ) -> ToolCallingTurnResponse:
        """Build a normalized tool-calling error response."""
        return ToolCallingTurnResponse(
            text=text,
            tool_calls=tuple(),
            response_id=None,
            request_id=None,
            model=self.config.groq_model,
            token_usage=None,
            used_web_search=False,
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
            # AUDIT-FIX(#3): Reap expired continuations so dropped or interrupted tool turns do not accumulate forever.
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
                return _ContinuationState(
                    messages=[],
                    expected_tool_call_ids=(),
                    created_monotonic=0.0,
                    last_access_monotonic=0.0,
                    in_flight=True,
                )
            # AUDIT-FIX(#2): Mark the continuation in flight so duplicate network deliveries cannot race and corrupt shared state.
            state.in_flight = True
            state.last_access_monotonic = time.monotonic()
            return copy.deepcopy(state)
    def _release_continuation(self, continuation_token: str, *, keep_state: bool, state: _ContinuationState | None = None) -> None:
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
            # AUDIT-FIX(#2): Require an exact call-id match so forged, stale, or partial tool results cannot be injected into the model turn.
            return None, "tool call ids do not match expected continuation state"
        ordered_results = [by_id[call_id] for call_id in expected_tool_call_ids]
        return ordered_results, None
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
        """Start a Groq tool-calling turn and persist continuation state if needed.

        Args:
            prompt: User prompt for the current turn.
            conversation: Optional prior conversation items for context.
            instructions: Optional system-level instruction overrides.
            tool_schemas: Available Twinr tool schemas for this turn.
            allow_web_search: Whether the caller requested live search support.
            on_text_delta: Optional callback for immediate spoken/UI output.

        Returns:
            A tool-calling response that may include tool calls and a continuation token.
        """
        if allow_web_search:
            logger.debug("Groq tool provider received allow_web_search=True; native live search is not implemented in this provider")
        messages = _build_messages(
            self.config,
            prompt,
            conversation=conversation,
            instructions=instructions,
        )
        try:
            response, assistant_message = self._run_tool_completion(
                messages,
                tool_schemas=tool_schemas,
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
                    # AUDIT-FIX(#2): Persist the exact tool-call IDs the model emitted so the continuation can verify the next handoff.
                    messages=copy.deepcopy([*messages, assistant_message]),
                    expected_tool_call_ids=tuple(call.call_id for call in response.tool_calls),
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
                used_web_search=False,
                continuation_token=token,
            )
        return response
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
        if allow_web_search:
            logger.debug("Groq tool provider received allow_web_search=True during continuation; native live search is not implemented in this provider")
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
            # AUDIT-FIX(#8): Honor continuation-time instruction updates instead of silently discarding them.
            messages = _merge_system_message(messages, instructions)
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
                    # AUDIT-FIX(#2): Normalize tool outputs to protocol-safe strings before re-submitting them to the model.
                    "content": _serialize_tool_output(result.serialized_output),
                }
            )
        try:
            response, assistant_message = self._run_tool_completion(
                messages,
                tool_schemas=tool_schemas,
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
            used_web_search=False,
            continuation_token=continuation_token if response.tool_calls else None,
        )
    def _run_tool_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        tool_schemas: Sequence[dict[str, Any]],
    ) -> tuple[ToolCallingTurnResponse, dict[str, Any]]:
        """Execute one Groq tool-completion request and normalize its result."""
        request: dict[str, Any] = {
            "model": self.config.groq_model,
            "messages": messages,
        }
        converted_tools = _convert_tool_schemas(tool_schemas)
        if converted_tools:
            request["tools"] = converted_tools
            request["tool_choice"] = "auto"
        completion = _invoke_chat_completion(self._client, request, config=self.config)
        response_id = str(getattr(completion, "id", "")).strip() or None
        request_id = str(getattr(completion, "_request_id", "")).strip() or None
        model = str(getattr(completion, "model", "")).strip() or self.config.groq_model
        choices = getattr(completion, "choices", None) or []
        message = getattr(choices[0], "message", None) if choices else None
        if message is None:
            raise RuntimeError("Groq tool completion returned no assistant message")
        tool_calls: list[AgentToolCall] = []
        assistant_tool_calls: list[dict[str, Any]] = []
        for tool_call in getattr(message, "tool_calls", None) or []:
            call_id = str(getattr(tool_call, "id", "")).strip()
            function = getattr(tool_call, "function", None)
            function_name = str(getattr(function, "name", "")).strip()
            raw_arguments = str(getattr(function, "arguments", "") or "{}").strip() or "{}"
            arguments = _parse_tool_arguments(raw_arguments)
            if arguments is None:
                # AUDIT-FIX(#4): Treat malformed model-emitted tool JSON as a recoverable provider failure instead of propagating a raw JSONDecodeError.
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
            ToolCallingTurnResponse(
                text=text,
                tool_calls=tuple(tool_calls),
                response_id=response_id,
                request_id=request_id,
                model=model,
                token_usage=_chat_usage(completion),
                used_web_search=False,
            ),
            assistant_message,
        )
