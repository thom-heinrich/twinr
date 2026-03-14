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
    ProviderBundle,
    SearchResponse,
    SpeechToTextProvider,
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

    @property
    def config(self) -> TwinrConfig:
        return self.backend.config

    @config.setter
    def config(self, value: TwinrConfig) -> None:
        self.backend.config = value

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
        request = self.backend._build_response_request(
            prompt,
            conversation=conversation,
            instructions=merge_instructions(self.backend._resolve_base_instructions(), instructions),
            allow_web_search=allow_web_search,
            model=self.config.default_model,
            reasoning_effort=self.config.openai_reasoning_effort,
        )
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
        request: dict[str, Any] = {
            "model": self.config.default_model,
            "previous_response_id": continuation_token,
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": result.call_id,
                    "output": result.serialized_output,
                }
                for result in tool_results
            ],
            "reasoning": {"effort": self.config.openai_reasoning_effort},
            "store": False,
        }
        merged_instructions = merge_instructions(
            merge_instructions(self.backend._resolve_base_instructions(), instructions),
            user_response_language_instruction(self.config.openai_realtime_language),
        )
        if merged_instructions:
            request["instructions"] = merged_instructions
        use_web_search = self.config.openai_enable_web_search if allow_web_search is None else allow_web_search
        web_search_tools = self.backend._build_tools(use_web_search)
        if web_search_tools:
            request["tools"] = list(web_search_tools)
        self._merge_tool_schemas(request, tool_schemas)
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
        tools.extend(tool_schemas)
        request["tools"] = tools
        request["tool_choice"] = "auto"

    def _run_streaming_request(
        self,
        request: dict[str, Any],
        *,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> ToolCallingTurnResponse:
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
