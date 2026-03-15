from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence
import json

from twinr.agent.base_agent.contracts import (
    AgentToolCall,
    AgentToolResult,
    AgentTextProvider,
    CompositeSpeechAgentProvider,
    ConversationLike,
    FirstWordProvider,
    FirstWordReply,
    ProviderBundle,
    SearchResponse,
    SpeechToTextProvider,
    SupervisorDecision,
    SupervisorDecisionProvider,
    TextResponse,
    TextToSpeechProvider,
    ToolCallingAgentProvider,
    ToolCallingTurnResponse,
)
from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.language import user_response_language_instruction
from twinr.agent.base_agent.personality import merge_instructions
from twinr.ops.usage import extract_model_name, extract_token_usage

from .backend import OpenAIBackend
from .types import OpenAIImageInput


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
        request = self.backend._build_response_request(
            prompt,
            conversation=conversation,
            instructions=self._merged_base_instructions(instructions),
            allow_web_search=allow_web_search,
            model=model,
            reasoning_effort=self._resolved_reasoning_effort(),
            prompt_cache_scope="tool_loop_start",
        )
        _apply_reasoning_effort_request(
            self.backend,
            request,
            model=model,
            reasoning_effort=self._resolved_reasoning_effort(),
        )
        request["store"] = True
        self._merge_tool_schemas(request, tool_schemas)
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
            "store": True,
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
        web_search_tools = self.backend._build_tools(use_web_search)
        if web_search_tools:
            request["tools"] = list(web_search_tools)
        self._merge_tool_schemas(request, tool_schemas)
        self.backend._apply_prompt_cache(
            request,
            scope="tool_loop_continue",
            model=model,
        )
        return self._run_streaming_request(request, on_text_delta=on_text_delta)

    def _merge_tool_schemas(
        self,
        request: dict[str, Any],
        tool_schemas: Sequence[dict[str, Any]],
    ) -> None:
        if not tool_schemas:
            if request.get("tools"):
                request["tool_choice"] = "auto"
            return
        tools = list(request.get("tools") or [])
        tools.extend(_normalize_openai_function_schema(schema) for schema in tool_schemas)
        request["tools"] = tools
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

        text = streamed_text.strip() or self.backend._extract_output_text(response)
        if text and not streamed_text.strip() and on_text_delta is not None:
            on_text_delta(text)
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
        streamed_text = ""
        with self.backend._client.responses.stream(**request) as stream:
            for event in stream:
                if getattr(event, "type", None) != "response.output_text.delta":
                    continue
                delta = str(getattr(event, "delta", ""))
                if not delta:
                    continue
                streamed_text += delta
                if on_text_delta is not None:
                    on_text_delta(delta)
            response = stream.get_final_response()
        return streamed_text, response

    def _extract_tool_calls(self, response: Any) -> tuple[AgentToolCall, ...]:
        output_items = getattr(response, "output", None) or []
        function_calls: list[AgentToolCall] = []
        for item in output_items:
            if str(getattr(item, "type", "")).strip() != "function_call":
                continue
            name = str(getattr(item, "name", "")).strip()
            call_id = str(getattr(item, "call_id", "")).strip()
            raw_arguments = str(getattr(item, "arguments", "") or "{}").strip() or "{}"
            if not name or not call_id:
                continue
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


_SUPERVISOR_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["direct", "handoff", "end_conversation"],
            "description": "direct for a short immediate answer, handoff for specialist work, end_conversation to stop for now.",
        },
        "spoken_ack": {
            "type": ["string", "null"],
            "description": "Short immediate acknowledgement only for handoff. Must stay null for direct replies.",
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
        "allow_web_search": {
            "type": ["boolean", "null"],
            "description": "True only when the specialist may use live web search.",
        },
    },
    "required": [
        "action",
        "spoken_ack",
        "spoken_reply",
        "kind",
        "goal",
        "allow_web_search",
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

_FIRST_WORD_MODEL_FALLBACKS: tuple[str, ...] = ("gpt-4o-mini",)


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
        request = self.backend._build_response_request(
            prompt,
            conversation=conversation,
            instructions=self._merged_base_instructions(instructions),
            allow_web_search=False,
            model=model,
            reasoning_effort=self._resolved_reasoning_effort(),
            max_output_tokens=max(32, int(self.config.streaming_supervisor_max_output_tokens)),
            prompt_cache_scope="supervisor_decision",
        )
        request["text"] = {
            "format": {
                "type": "json_schema",
                "name": "twinr_supervisor_decision",
                "schema": _SUPERVISOR_DECISION_SCHEMA,
                "strict": True,
            }
        }
        response = self.backend._client.responses.create(**request)
        payload = json.loads(self.backend._extract_output_text(response) or "{}")
        return SupervisorDecision(
            action=str(payload.get("action", "handoff") or "handoff"),
            spoken_ack=_optional_text(payload.get("spoken_ack")),
            spoken_reply=_optional_text(payload.get("spoken_reply")),
            kind=_optional_text(payload.get("kind")),
            goal=_optional_text(payload.get("goal")),
            allow_web_search=_optional_bool(payload.get("allow_web_search")),
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
        return self.config.streaming_first_word_model

    def _resolved_reasoning_effort(self) -> str | None:
        override = (self.reasoning_effort_override or "").strip()
        if override:
            return override
        resolved = (self.config.streaming_first_word_reasoning_effort or "").strip()
        return resolved or None

    def _merged_base_instructions(self, instructions: str | None) -> str | None:
        if self.replace_base_instructions:
            return merge_instructions(self.base_instructions_override, instructions)
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
            request["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "twinr_first_word_reply",
                    "schema": _FIRST_WORD_REPLY_SCHEMA,
                    "strict": True,
                }
            }
            return self.backend._client.responses.create(**request)

        response, model_used = self.backend._call_with_model_fallback(
            preferred_model,
            _FIRST_WORD_MODEL_FALLBACKS,
            _call,
        )
        payload = json.loads(self.backend._extract_output_text(response) or "{}")
        return FirstWordReply(
            mode=str(payload.get("mode", "filler") or "filler"),
            spoken_text=str(payload.get("spoken_text", "") or "").strip(),
            response_id=getattr(response, "id", None),
            request_id=getattr(response, "_request_id", None),
            model=extract_model_name(response, model_used),
            token_usage=extract_token_usage(response),
        )


def _is_reasoning_unsupported_error(exc: Exception) -> bool:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        message = str(body.get("error", {}).get("message", "")).lower()
        param = str(body.get("error", {}).get("param", "")).lower()
        if "reasoning.effort" in param or "reasoning.effort" in message:
            return True
    message = str(exc).lower()
    return "reasoning.effort" in message and "not supported" in message


def _apply_reasoning_effort_request(
    backend: Any,
    request: dict[str, Any],
    *,
    model: str,
    reasoning_effort: str | None,
) -> None:
    helper = getattr(backend, "_apply_reasoning_effort", None)
    if callable(helper):
        helper(
            request,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        return
    normalized_effort = (reasoning_effort or "").strip().lower()
    if not normalized_effort:
        request.pop("reasoning", None)
        return
    if not _model_supports_reasoning_effort(model):
        request.pop("reasoning", None)
        return
    request["reasoning"] = {"effort": normalized_effort}


def _model_supports_reasoning_effort(model: str) -> bool:
    normalized = (model or "").strip().lower()
    if not normalized:
        return False
    return normalized.startswith(("gpt-5", "o1", "o3", "o4"))


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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


def _normalize_openai_function_schema(schema: dict[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(schema))
    if copied.get("type") != "function":
        return copied
    parameters = copied.get("parameters")
    if not isinstance(parameters, dict):
        return copied
    for key in ("anyOf", "oneOf", "allOf", "not", "enum"):
        parameters.pop(key, None)
    return copied


@dataclass
class OpenAIProviderBundle(ProviderBundle):
    backend: OpenAIBackend
    combined: CompositeSpeechAgentProvider

    @classmethod
    def from_backend(cls, backend: OpenAIBackend) -> OpenAIProviderBundle:
        stt = OpenAISpeechToTextProvider(backend)
        agent = OpenAIAgentTextProvider(backend)
        tts = OpenAITextToSpeechProvider(backend)
        tool_agent = OpenAIToolCallingAgentProvider(backend)
        return cls(
            stt=stt,
            agent=agent,
            tts=tts,
            tool_agent=tool_agent,
            backend=backend,
            combined=CompositeSpeechAgentProvider(stt=stt, agent=agent, tts=tts),
        )

    @classmethod
    def from_config(cls, config: TwinrConfig) -> OpenAIProviderBundle:
        return cls.from_backend(OpenAIBackend(config=config))
