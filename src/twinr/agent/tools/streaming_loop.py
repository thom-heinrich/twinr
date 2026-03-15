from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from twinr.agent.base_agent.contracts import (
    AgentToolCall,
    AgentToolResult,
    ConversationLike,
    ToolCallingAgentProvider,
    ToolCallingTurnResponse,
)

ToolHandler = Callable[[dict[str, Any]], Any]


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
        self.provider = provider
        self.tool_handlers = dict(tool_handlers)
        self.tool_schemas = tuple(tool_schemas)
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
        last_response: ToolCallingTurnResponse | None = None
        continuation_token: str | None = None
        next_tool_results: tuple[AgentToolResult, ...] = ()
        used_web_search = False

        for round_index in range(1, self.max_rounds + 1):
            round_on_text_delta = None if self.stream_final_only else on_text_delta
            if continuation_token is None:
                response = self.provider.start_turn_streaming(
                    prompt,
                    conversation=conversation,
                    instructions=instructions,
                    tool_schemas=self.tool_schemas,
                    allow_web_search=allow_web_search,
                    on_text_delta=round_on_text_delta,
                )
            else:
                response = self.provider.continue_turn_streaming(
                    continuation_token=continuation_token,
                    tool_results=next_tool_results,
                    instructions=instructions,
                    tool_schemas=self.tool_schemas,
                    allow_web_search=allow_web_search,
                    on_text_delta=round_on_text_delta,
                )

            last_response = response
            aggregate_text = _append_round_text(aggregate_text, response.text)
            used_web_search = used_web_search or response.used_web_search
            all_tool_calls.extend(response.tool_calls)
            if not response.tool_calls:
                result_text = response.text.strip() if self.stream_final_only else aggregate_text.strip()
                if self.stream_final_only and result_text and on_text_delta is not None:
                    on_text_delta(result_text)
                return StreamingToolLoopResult(
                    text=result_text,
                    rounds=round_index,
                    tool_calls=tuple(all_tool_calls),
                    tool_results=tuple(all_tool_results),
                    response_id=response.response_id,
                    request_id=response.request_id,
                    model=response.model,
                    token_usage=response.token_usage,
                    used_web_search=used_web_search,
                )

            continuation_token = response.continuation_token or response.response_id
            if not continuation_token:
                raise RuntimeError("Agent provider returned tool calls without a continuation token")
            next_tool_results = tuple(self._execute_tool_call(call) for call in response.tool_calls)
            all_tool_results.extend(next_tool_results)

        raise RuntimeError(f"Agent tool loop exceeded max_rounds={self.max_rounds}")

    def _execute_tool_call(self, tool_call: AgentToolCall) -> AgentToolResult:
        handler = self.tool_handlers.get(tool_call.name)
        if handler is None:
            output: Any = {"status": "error", "message": f"Unsupported tool: {tool_call.name}"}
        else:
            try:
                if getattr(handler, "_twinr_accepts_tool_call", False):
                    result = handler(tool_call)
                else:
                    result = handler(tool_call.arguments)
            except Exception as exc:
                output = {"status": "error", "message": str(exc)}
            else:
                output = result if result is not None else {"status": "ok"}
        return AgentToolResult(
            call_id=tool_call.call_id,
            name=tool_call.name,
            output=output,
            serialized_output=_serialize_tool_output(output),
        )


def _append_round_text(existing: str, addition: str) -> str:
    new_text = addition.strip()
    if not new_text:
        return existing
    if not existing:
        return new_text
    if existing.endswith(("\n", " ", "\t")) or new_text.startswith((".", ",", "!", "?", ";", ":")):
        return f"{existing}{new_text}"
    if existing.endswith((".", "!", "?", ":")):
        return f"{existing}\n{new_text}"
    return f"{existing} {new_text}"


def _serialize_tool_output(output: Any) -> str:
    if isinstance(output, str):
        return output
    return json.dumps(output, ensure_ascii=False)
