from __future__ import annotations

import copy
import json
import logging
import math
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Callable, Sequence

from twinr.agent.base_agent.contracts import (
    AgentToolCall,
    AgentToolResult,
    ConversationLike,
    ToolCallingAgentProvider,
    ToolCallingTurnResponse,
)

# AUDIT-FIX(#7): Widen the handler type so static typing matches both supported invocation modes.
ToolHandler = Callable[[dict[str, Any] | AgentToolCall], Any]

# AUDIT-FIX(#1): Add internal logging so detailed failures stay server-side instead of leaking into tool outputs.
LOGGER = logging.getLogger(__name__)
_PROVIDER_FAILURE_MESSAGE = "The assistant could not complete this request."
_UNSUPPORTED_TOOL_MESSAGE = "This action is not available."
_TOOL_EXECUTION_ERROR_MESSAGE = "The action could not be completed."
_UNSERIALIZABLE_TOOL_OUTPUT_MESSAGE = "The action returned data that could not be processed."


@dataclass(frozen=True, slots=True)
class StreamingToolLoopResult:
    text: str
    rounds: int
    tool_calls: tuple[AgentToolCall, ...]
    tool_results: tuple[AgentToolResult, ...]
    response_id: str | None = None
    request_id: str | None = None
    model: str | None = None
    token_usage: object | None = None
    used_web_search: bool = False


class ToolCallingStreamingLoop:
    def __init__(
        self,
        provider: ToolCallingAgentProvider,
        *,
        tool_handlers: dict[str, ToolHandler],
        tool_schemas: Sequence[dict[str, Any]],
        max_rounds: int = 6,
        stream_final_only: bool = False,
    ) -> None:
        # AUDIT-FIX(#2): Fail fast on invalid loop configuration instead of discovering it mid-conversation.
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")

        validated_tool_handlers = dict(tool_handlers)
        for tool_name, handler in validated_tool_handlers.items():
            if not isinstance(tool_name, str) or not tool_name:
                raise ValueError("tool handler names must be non-empty strings")
            if not callable(handler):
                raise TypeError(f"tool handler for '{tool_name}' must be callable")

        # AUDIT-FIX(#2): Deep-copy schemas so external mutation cannot change runtime behavior after initialization.
        frozen_tool_schemas = tuple(copy.deepcopy(schema) for schema in tool_schemas)
        schema_names = _collect_tool_schema_names(frozen_tool_schemas)
        missing_handlers = sorted(name for name in schema_names if name not in validated_tool_handlers)
        if missing_handlers:
            missing_handlers_csv = ", ".join(missing_handlers)
            raise ValueError(f"tool_schemas reference missing handlers: {missing_handlers_csv}")

        self.provider = provider
        self.tool_handlers = validated_tool_handlers
        self.tool_schemas = frozen_tool_schemas
        self.max_rounds = max_rounds
        self.stream_final_only = stream_final_only

    def run(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> StreamingToolLoopResult:
        aggregate_text = ""
        all_tool_calls: list[AgentToolCall] = []
        all_tool_results: list[AgentToolResult] = []
        continuation_token: str | None = None
        next_tool_results: tuple[AgentToolResult, ...] = ()
        used_web_search = False
        # AUDIT-FIX(#3): Disable a broken streaming callback after the first failure so the turn can still complete.
        safe_on_text_delta = _build_safe_text_delta_callback(on_text_delta)

        for round_index in range(1, self.max_rounds + 1):
            round_on_text_delta = None if self.stream_final_only else safe_on_text_delta
            response = self._request_turn(
                prompt,
                conversation=conversation,
                instructions=instructions,
                allow_web_search=allow_web_search,
                on_text_delta=round_on_text_delta,
                continuation_token=continuation_token,
                tool_results=next_tool_results,
                round_index=round_index,
            )

            try:
                # AUDIT-FIX(#3): Guard against malformed provider responses before they can crash the loop.
                response_text = _normalize_response_text(getattr(response, "text", ""))
                tool_calls = _normalize_tool_calls(getattr(response, "tool_calls", ()))
            except Exception:
                LOGGER.exception("Agent provider returned malformed response on round %s", round_index)
                raise RuntimeError(_PROVIDER_FAILURE_MESSAGE) from None
            aggregate_text = _append_round_text(aggregate_text, response_text)
            used_web_search = used_web_search or bool(getattr(response, "used_web_search", False))
            all_tool_calls.extend(tool_calls)
            if not tool_calls:
                aggregate_result_text = aggregate_text.strip()
                final_round_text = response_text.strip()
                # AUDIT-FIX(#6): Fall back to accumulated text when the final provider chunk is empty.
                result_text = (
                    final_round_text or aggregate_result_text
                    if self.stream_final_only
                    else aggregate_result_text
                )
                if self.stream_final_only and result_text and safe_on_text_delta is not None:
                    safe_on_text_delta(result_text)
                return StreamingToolLoopResult(
                    text=result_text,
                    rounds=round_index,
                    tool_calls=tuple(all_tool_calls),
                    tool_results=tuple(all_tool_results),
                    response_id=_optional_str(getattr(response, "response_id", None)),
                    request_id=_optional_str(getattr(response, "request_id", None)),
                    model=_optional_str(getattr(response, "model", None)),
                    token_usage=getattr(response, "token_usage", None),
                    used_web_search=used_web_search,
                )

            continuation_token = _extract_continuation_token(response)
            if not continuation_token:
                LOGGER.error(
                    "Agent provider returned tool calls without a continuation token or response_id"
                )
                raise RuntimeError(_PROVIDER_FAILURE_MESSAGE)
            next_tool_results = tuple(self._execute_tool_call(call) for call in tool_calls)
            all_tool_results.extend(next_tool_results)

        LOGGER.error("Agent tool loop exceeded max_rounds=%s", self.max_rounds)
        raise RuntimeError(_PROVIDER_FAILURE_MESSAGE)

    def _request_turn(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None,
        instructions: str | None,
        allow_web_search: bool | None,
        on_text_delta: Callable[[str], None] | None,
        continuation_token: str | None,
        tool_results: tuple[AgentToolResult, ...],
        round_index: int,
    ) -> ToolCallingTurnResponse:
        try:
            if continuation_token is None:
                response = self.provider.start_turn_streaming(
                    prompt,
                    conversation=conversation,
                    instructions=instructions,
                    tool_schemas=self.tool_schemas,
                    allow_web_search=allow_web_search,
                    on_text_delta=on_text_delta,
                )
            else:
                response = self.provider.continue_turn_streaming(
                    continuation_token=continuation_token,
                    tool_results=tool_results,
                    instructions=instructions,
                    tool_schemas=self.tool_schemas,
                    allow_web_search=allow_web_search,
                    on_text_delta=on_text_delta,
                )
        except Exception:
            # AUDIT-FIX(#3): Keep provider exceptions out of user/model-facing strings while preserving diagnostics in logs.
            LOGGER.exception("Agent provider streaming failed on round %s", round_index)
            raise RuntimeError(_PROVIDER_FAILURE_MESSAGE) from None

        if response is None:
            LOGGER.error("Agent provider returned no response on round %s", round_index)
            raise RuntimeError(_PROVIDER_FAILURE_MESSAGE)
        return response

    def _execute_tool_call(self, tool_call: AgentToolCall) -> AgentToolResult:
        handler = self.tool_handlers.get(tool_call.name)
        if handler is None:
            LOGGER.warning("Unsupported tool requested: %s", tool_call.name)
            # AUDIT-FIX(#1): Return a sanitized, user-safe error envelope instead of exposing internal details.
            output: Any = {
                "status": "error",
                "message": _UNSUPPORTED_TOOL_MESSAGE,
                "tool": tool_call.name,
            }
        else:
            try:
                if getattr(handler, "_twinr_accepts_tool_call", False):
                    # AUDIT-FIX(#4): Isolate tool-call objects from handler-side mutation before execution.
                    result = handler(copy.deepcopy(tool_call))
                else:
                    arguments = tool_call.arguments
                    if not isinstance(arguments, dict):
                        raise TypeError("tool arguments must be a JSON object")
                    # AUDIT-FIX(#4): Validate and copy arguments so malformed or mutated inputs cannot corrupt loop state.
                    result = handler(copy.deepcopy(arguments))
            except Exception:
                LOGGER.exception("Tool handler failed for tool '%s'", tool_call.name)
                # AUDIT-FIX(#1): Keep raw exception text out of downstream tool outputs.
                output = {
                    "status": "error",
                    "message": _TOOL_EXECUTION_ERROR_MESSAGE,
                    "tool": tool_call.name,
                }
            else:
                output = result if result is not None else {"status": "ok"}
        return AgentToolResult(
            call_id=tool_call.call_id,
            name=tool_call.name,
            output=output,
            serialized_output=_serialize_tool_output(output),
        )


def _append_round_text(existing: str, addition: str) -> str:
    # AUDIT-FIX(#6): Preserve round-local formatting instead of stripping indentation/newlines that may be meaningful.
    if not addition or not addition.strip():
        return existing
    if not existing:
        return addition
    if existing[-1].isspace() or addition[0].isspace() or addition[0] in ".,!?;:":
        return f"{existing}{addition}"
    if existing[-1] in ".!?:":
        return f"{existing}\n{addition}"
    return f"{existing} {addition}"


def _serialize_tool_output(output: Any) -> str:
    if isinstance(output, str):
        return output
    try:
        # AUDIT-FIX(#5): Convert tool outputs into strict JSON so non-serializable objects and NaN/Infinity cannot crash the loop.
        return json.dumps(_make_json_safe(output), ensure_ascii=False, allow_nan=False)
    except Exception:
        LOGGER.exception("Failed to serialize tool output of type %s", type(output).__name__)
        fallback_output = {
            "status": "error",
            "message": _UNSERIALIZABLE_TOOL_OUTPUT_MESSAGE,
            "output_type": type(output).__name__,
        }
        return json.dumps(fallback_output, ensure_ascii=False, allow_nan=False)


def _build_safe_text_delta_callback(
    callback: Callable[[str], None] | None,
) -> Callable[[str], None] | None:
    if callback is None:
        return None

    disabled = False

    def safe_callback(text: str) -> None:
        nonlocal disabled
        if disabled or not text:
            return
        try:
            callback(text)
        except Exception:
            LOGGER.exception(
                "on_text_delta callback failed; disabling further streaming callbacks"
            )
            disabled = True

    return safe_callback


def _normalize_response_text(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, str):
        return text
    return str(text)


def _normalize_tool_calls(tool_calls: Any) -> tuple[AgentToolCall, ...]:
    if tool_calls is None:
        return ()
    if isinstance(tool_calls, (str, bytes, bytearray)):
        raise TypeError("agent provider returned invalid tool_calls")
    try:
        normalized_tool_calls = tuple(tool_calls)
    except TypeError as exc:
        raise TypeError("agent provider returned invalid tool_calls") from exc
    for tool_call in normalized_tool_calls:
        if not hasattr(tool_call, "call_id") or not hasattr(tool_call, "name"):
            raise TypeError("agent provider returned malformed tool_calls")
    return normalized_tool_calls


def _extract_continuation_token(response: ToolCallingTurnResponse) -> str | None:
    continuation_token = getattr(response, "continuation_token", None)
    if isinstance(continuation_token, str) and continuation_token:
        return continuation_token
    response_id = getattr(response, "response_id", None)
    if isinstance(response_id, str) and response_id:
        return response_id
    return None


def _collect_tool_schema_names(tool_schemas: Sequence[dict[str, Any]]) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    duplicate_names: set[str] = set()

    for schema in tool_schemas:
        if not isinstance(schema, dict):
            raise TypeError("each tool schema must be a dict")
        tool_name = _extract_tool_schema_name(schema)
        if tool_name is None:
            continue
        if tool_name in seen:
            duplicate_names.add(tool_name)
            continue
        seen.add(tool_name)
        names.append(tool_name)

    if duplicate_names:
        duplicate_names_csv = ", ".join(sorted(duplicate_names))
        raise ValueError(f"duplicate tool schema names: {duplicate_names_csv}")

    return tuple(names)


def _extract_tool_schema_name(schema: dict[str, Any]) -> str | None:
    tool_name = schema.get("name")
    if isinstance(tool_name, str) and tool_name:
        return tool_name
    function_schema = schema.get("function")
    if isinstance(function_schema, dict):
        function_name = function_schema.get("name")
        if isinstance(function_name, str) and function_name:
            return function_name
    return None


def _make_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if is_dataclass(value):
        return _make_json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_make_json_safe(item) for item in value]

    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return isoformat()
        except Exception:
            LOGGER.debug("isoformat() failed during JSON sanitization", exc_info=True)

    return str(value)


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None