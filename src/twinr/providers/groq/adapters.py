from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, Sequence
from uuid import uuid4
import json

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
from twinr.agent.base_agent.language import user_response_language_instruction
from twinr.agent.base_agent.personality import load_personality_instructions, merge_instructions
from twinr.ops.usage import TokenUsage
from twinr.providers.groq.client import default_groq_client
from twinr.providers.groq.types import GroqTextResponse


def _chat_usage(source: object) -> TokenUsage | None:
    usage = getattr(source, "usage", None)
    if usage is None:
        return None
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    token_usage = TokenUsage(
        input_tokens=int(prompt_tokens) if prompt_tokens is not None else None,
        output_tokens=int(completion_tokens) if completion_tokens is not None else None,
        total_tokens=int(total_tokens) if total_tokens is not None else None,
    )
    return token_usage if token_usage.has_values else None


def _message_text(message: object) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        fragments: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                fragments.append(str(item["text"]).strip())
        return "\n".join(fragment for fragment in fragments if fragment).strip()
    return str(content or "").strip()


def _coerce_message(item: object) -> tuple[str, str]:
    if isinstance(item, tuple) and len(item) == 2:
        role, content = item
        return str(role).strip().lower(), str(content).strip()
    role = str(getattr(item, "role", "")).strip().lower()
    content = str(getattr(item, "content", "")).strip()
    return role, content


def _build_messages(
    config: TwinrConfig,
    prompt: str,
    *,
    conversation: ConversationLike | None = None,
    instructions: str | None = None,
) -> list[dict[str, Any]]:
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
            if not content:
                continue
            if role not in {"system", "user", "assistant"}:
                role = "user"
            if role == "system":
                system_parts.append(content)
                continue
            messages.append({"role": role, "content": content})
    merged_instructions = merge_instructions(*system_parts)
    if merged_instructions:
        messages.insert(0, {"role": "system", "content": merged_instructions})
    prompt_text = prompt.strip()
    if prompt_text:
        messages.append({"role": "user", "content": prompt_text})
    return messages


def _convert_tool_schemas(tool_schemas: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for schema in tool_schemas:
        if schema.get("type") != "function":
            continue
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": schema.get("name"),
                    "description": schema.get("description"),
                    "parameters": schema.get("parameters") or {"type": "object", "properties": {}},
                },
            }
        )
    return converted


@dataclass
class GroqAgentTextProvider:
    config: TwinrConfig
    support_provider: AgentTextProvider
    client: Any | None = None

    def __post_init__(self) -> None:
        self._client = self.client or default_groq_client(self.config)

    def respond_streaming(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> TextResponse:
        if allow_web_search:
            return self.support_provider.respond_streaming(
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
        for chunk in self._client.chat.completions.create(**request):
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
            text_fragments.append(str(content))
            if on_text_delta is not None:
                on_text_delta(str(content))
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
        if allow_web_search:
            return self.support_provider.respond_with_metadata(
                prompt,
                conversation=conversation,
                instructions=instructions,
                allow_web_search=allow_web_search,
            )
        completion = self._client.chat.completions.create(
            model=self.config.groq_model,
            messages=_build_messages(
                self.config,
                prompt,
                conversation=conversation,
                instructions=instructions,
            ),
        )
        choice = (getattr(completion, "choices", None) or [None])[0]
        message = getattr(choice, "message", None)
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
        return self.support_provider.compose_print_job_with_metadata(
            conversation=conversation,
            focus_hint=focus_hint,
            direct_text=direct_text,
            request_source=request_source,
        )

    def phrase_due_reminder_with_metadata(self, reminder: object, *, now=None) -> TextResponse:
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
        return self.support_provider.fulfill_automation_prompt_with_metadata(
            prompt,
            allow_web_search=allow_web_search,
            delivery=delivery,
        )


@dataclass
class GroqToolCallingAgentProvider:
    config: TwinrConfig
    client: Any | None = None

    def __post_init__(self) -> None:
        self._client = self.client or default_groq_client(self.config)
        self._continuations: dict[str, list[dict[str, Any]]] = {}
        self._lock = Lock()

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
        del allow_web_search
        messages = _build_messages(
            self.config,
            prompt,
            conversation=conversation,
            instructions=instructions,
        )
        response, assistant_message = self._run_tool_completion(
            messages,
            tool_schemas=tool_schemas,
        )
        if response.text and on_text_delta is not None and not response.tool_calls:
            on_text_delta(response.text)
        if response.tool_calls:
            token = uuid4().hex
            with self._lock:
                self._continuations[token] = [*messages, assistant_message]
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
        del instructions, allow_web_search
        with self._lock:
            messages = list(self._continuations.get(continuation_token) or [])
        if not messages:
            raise RuntimeError("Groq tool continuation token is unknown or expired")
        for result in tool_results:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result.call_id,
                    "content": result.serialized_output,
                }
            )
        response, assistant_message = self._run_tool_completion(
            messages,
            tool_schemas=tool_schemas,
        )
        if response.text and on_text_delta is not None and not response.tool_calls:
            on_text_delta(response.text)
        with self._lock:
            if response.tool_calls:
                self._continuations[continuation_token] = [*messages, assistant_message]
            else:
                self._continuations.pop(continuation_token, None)
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
        request: dict[str, Any] = {
            "model": self.config.groq_model,
            "messages": messages,
        }
        converted_tools = _convert_tool_schemas(tool_schemas)
        if converted_tools:
            request["tools"] = converted_tools
            request["tool_choice"] = "auto"
        completion = self._client.chat.completions.create(**request)
        response_id = str(getattr(completion, "id", "")).strip() or None
        request_id = str(getattr(completion, "_request_id", "")).strip() or None
        model = str(getattr(completion, "model", "")).strip() or self.config.groq_model
        choices = getattr(completion, "choices", None) or []
        message = getattr(choices[0], "message", None) if choices else None
        tool_calls: list[AgentToolCall] = []
        assistant_tool_calls: list[dict[str, Any]] = []
        for tool_call in getattr(message, "tool_calls", None) or []:
            call_id = str(getattr(tool_call, "id", "")).strip()
            function = getattr(tool_call, "function", None)
            function_name = str(getattr(function, "name", "")).strip()
            raw_arguments = str(getattr(function, "arguments", "") or "{}").strip() or "{}"
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Groq tool arguments are not valid JSON: {exc.msg}") from exc
            if not isinstance(arguments, dict):
                raise RuntimeError("Groq tool arguments must decode to a JSON object")
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
