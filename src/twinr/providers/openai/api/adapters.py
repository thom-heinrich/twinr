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
from typing import Any, Callable, Sequence
import json
import logging
import re

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
from twinr.agent.base_agent.conversation.language import user_response_language_instruction
from twinr.agent.base_agent.prompting.personality import merge_instructions
from twinr.ops.usage import extract_model_name, extract_token_usage

from .backend import OpenAIBackend
from ..core.types import OpenAIImageInput


logger = logging.getLogger(__name__)  # AUDIT-FIX(#3): Callback isolation needs local logging instead of hard-failing completed turns.
_O_SERIES_MODEL_PATTERN = re.compile(r"^o\d+(?:[-_.].*)?$")  # AUDIT-FIX(#5): Reasoning support must follow generic o-series model IDs.


@dataclass
class OpenAISpeechToTextProvider:
    """Bridge Twinr speech-to-text contract calls onto the shared backend.

    Attributes:
        backend: Shared backend instance that performs transcription requests.
    """

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
        """Transcribe in-memory audio bytes through the shared backend.

        Returns an empty string when ``audio_bytes`` is empty so accidental
        blank captures do not trigger a doomed upstream request.
        """

        if not audio_bytes:
            return ""  # AUDIT-FIX(#4): Short-circuit accidental empty captures instead of sending a doomed STT request upstream.
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
        """Transcribe audio from a filesystem path through the backend.

        Returns an empty string when ``path`` is blank so callers can treat a
        missing audio path the same way as a missing capture.
        """

        if isinstance(path, str) and not path.strip():
            return ""  # AUDIT-FIX(#4): Blank path input is equivalent to “no audio” and should not trigger a backend failure.
        return self.backend.transcribe_path(path, language=language, prompt=prompt)


@dataclass
class OpenAIAgentTextProvider:
    """Bridge Twinr text and multimodal agent calls onto the backend.

    Attributes:
        backend: Shared backend instance used for text, image, search, print,
            reminder, and proactive phrasing requests.
    """

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
        """Stream a text response through the shared backend."""

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
        """Return a non-streaming text response with request metadata."""

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
        """Return a multimodal response for a text prompt plus images."""

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
        """Run a live-search answer request through the backend."""

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
        """Compose a printer-friendly response through the backend."""

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
        """Phrase a due reminder through the backend reminder flow."""

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
        """Phrase a proactive spoken prompt through the backend."""

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
        """Fulfill an automation prompt through the backend."""

        return self.backend.fulfill_automation_prompt_with_metadata(
            prompt,
            allow_web_search=allow_web_search,
            delivery=delivery,
        )


@dataclass
class OpenAITextToSpeechProvider:
    """Bridge Twinr text-to-speech calls onto the shared backend.

    Attributes:
        backend: Shared backend instance that performs TTS synthesis requests.
    """

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
        """Synthesize text into audio bytes through the backend.

        Returns empty bytes when ``text`` is blank so callers can treat an
        empty reply as silence instead of an upstream request.
        """

        if not text.strip():
            return b""  # AUDIT-FIX(#4): Blank spoken replies should degrade to silence, not an upstream TTS request.
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
        """Stream synthesized audio chunks through the backend.

        Raises:
            ValueError: If ``chunk_size`` is less than 1.
        """

        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")  # AUDIT-FIX(#4): Reject invalid chunk sizing before the backend can fail unpredictably.
        if not text.strip():
            return iter(())  # AUDIT-FIX(#4): Blank spoken replies should produce an empty stream.
        return self.backend.synthesize_stream(
            text,
            voice=voice,
            response_format=response_format,
            instructions=instructions,
            chunk_size=chunk_size,
        )


@dataclass
class OpenAIToolCallingAgentProvider:
    """Run Twinr tool-calling turns against the shared OpenAI backend.

    Attributes:
        backend: Shared backend instance that owns the OpenAI client.
        model_override: Optional model name that overrides config defaults.
        reasoning_effort_override: Optional reasoning-effort override for this
            provider instance.
        base_instructions_override: Optional instructions merged into tool-loop
            prompts before they are sent upstream.
        replace_base_instructions: When true, replace the backend base
            instructions instead of appending to them.
    """

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
        """Resolve the model name for tool-calling requests."""

        override = (self.model_override or "").strip()
        if override:
            return override
        return self.config.default_model

    def _resolved_reasoning_effort(self) -> str:
        """Resolve the reasoning-effort value for tool-calling requests."""

        override = (self.reasoning_effort_override or "").strip()
        if override:
            return override
        return self.config.openai_reasoning_effort

    def _merged_base_instructions(self, instructions: str | None) -> str | None:
        """Merge backend tool-loop instructions with local overrides."""

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
        """Start a streamed tool-calling turn.

        Args:
            prompt: User or system prompt to send to the model.
            conversation: Prior conversation state to include in the request.
            instructions: Optional turn-specific instruction suffix.
            tool_schemas: Tool definitions exposed to the model.
            allow_web_search: Override for live web-search availability.
            on_text_delta: Optional callback invoked for streamed text chunks.

        Returns:
            A tool-calling turn response containing streamed text, tool calls,
            metadata, and the continuation token for the next step.
        """

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
            reasoning_effort=reasoning_effort,  # AUDIT-FIX(#3,#5): Normalize reasoning config before the request leaves this provider.
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
        """Continue a streamed tool-calling turn with tool results.

        Args:
            continuation_token: Response identifier from the previous turn.
            tool_results: Serialized tool outputs to feed back to the model.
            instructions: Optional instruction suffix for the continuation.
            tool_schemas: Tool definitions exposed for follow-up calls.
            allow_web_search: Override for live web-search availability.
            on_text_delta: Optional callback invoked for streamed text chunks.

        Returns:
            A tool-calling turn response for the continuation request.

        Raises:
            RuntimeError: If ``continuation_token`` is blank.
        """

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
            reasoning_effort=self._resolved_reasoning_effort(),  # AUDIT-FIX(#3,#5): Apply the same reasoning gating on tool-loop continuation requests.
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
        """Merge normalized function schemas into an outbound request."""

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
        """Run a streamed tool request with unsupported-reasoning fallback."""

        try:
            streamed_text, response = self._consume_stream(request, on_text_delta=on_text_delta)
        except Exception as exc:
            if not _is_reasoning_unsupported_error(exc) or "reasoning" not in request:
                raise
            retry_request = dict(request)
            retry_request.pop("reasoning", None)
            streamed_text, response = self._consume_stream(retry_request, on_text_delta=on_text_delta)

        fallback_text = _coerce_text(self.backend._extract_output_text(response)).strip()
        text = streamed_text.strip() or fallback_text  # AUDIT-FIX(#1,#2): Never propagate None for tool-only turns or failed text extraction.
        if text and not streamed_text.strip():
            _emit_text_delta(on_text_delta, text, context="tool-loop fallback text")  # AUDIT-FIX(#2): Callback failures must not crash a completed model turn.
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
        """Consume streamed output-text deltas from an OpenAI response stream."""

        streamed_chunks: list[str] = []
        with self.backend._client.responses.stream(**request) as stream:
            for event in stream:
                if getattr(event, "type", None) != "response.output_text.delta":
                    continue
                delta = _coerce_text(getattr(event, "delta", ""))
                if not delta:
                    continue
                streamed_chunks.append(delta)
                _emit_text_delta(on_text_delta, delta, context="tool-loop stream delta")  # AUDIT-FIX(#2): Protect the upstream response from callback-layer faults.
            response = stream.get_final_response()
        _validate_response_status(response, context="tool-loop response")  # AUDIT-FIX(#1): Failed/incomplete responses must not be treated as successful turns.
        return "".join(streamed_chunks), response

    def _extract_tool_calls(self, response: Any) -> tuple[AgentToolCall, ...]:
        """Extract validated tool-call payloads from a final response object.

        Raises:
            RuntimeError: If a function-call argument payload is not valid JSON
                or does not decode to a JSON object.
        """

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
                "tiny_recent only when the fast lane can answer safely from the tiny recent context alone. "
                "full_context when the answer depends on broader memory or richer provider context."
            ),
        },
    },
    "required": [
        "action",
        "spoken_ack",
        "spoken_reply",
        "kind",
        "goal",
        "allow_web_search",
        "location_hint",
        "date_context",
        "context_scope",
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
    """Produce structured supervisor-routing decisions through the backend.

    Attributes:
        backend: Shared backend instance that owns the OpenAI client.
        model_override: Optional supervisor model override.
        reasoning_effort_override: Optional supervisor reasoning override.
        base_instructions_override: Optional instruction override merged into
            the backend base instructions.
        replace_base_instructions: When true, replace backend base
            instructions instead of appending to them.
    """

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
        """Resolve the model name for supervisor decisions."""

        override = (self.model_override or "").strip()
        if override:
            return override
        return self.config.default_model

    def _resolved_reasoning_effort(self) -> str:
        """Resolve the reasoning-effort value for supervisor decisions."""

        override = (self.reasoning_effort_override or "").strip()
        if override:
            return override
        return self.config.openai_reasoning_effort

    def _merged_base_instructions(self, instructions: str | None) -> str | None:
        """Merge backend tool-loop instructions with supervisor overrides."""

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
        """Return a validated structured supervisor decision."""

        model = self._resolved_model()
        reasoning_effort = self._resolved_reasoning_effort()
        request = self.backend._build_response_request(
            prompt,
            conversation=conversation,
            instructions=self._merged_base_instructions(instructions),
            allow_web_search=False,
            model=model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max(32, int(self.config.streaming_supervisor_max_output_tokens)),
            prompt_cache_scope="supervisor_decision",
        )
        _apply_reasoning_effort_request(
            self.backend,
            request,
            model=model,
            reasoning_effort=reasoning_effort,  # AUDIT-FIX(#3,#5): Gate reasoning support before supervisor requests hit the API.
        )
        request["text"] = {
            "format": {
                "type": "json_schema",
                "name": "twinr_supervisor_decision",
                "schema": _SUPERVISOR_DECISION_SCHEMA,
                "strict": True,
            }
        }
        response = _create_response_with_reasoning_fallback(
            self.backend,
            request,
            context="supervisor decision",  # AUDIT-FIX(#1,#3): Validate response status and retry once without unsupported reasoning.
        )
        payload = _load_json_object(
            _coerce_text(self.backend._extract_output_text(response)),
            context="supervisor decision",
        )
        return SupervisorDecision(
            action=_validated_choice(
                payload.get("action"),
                allowed=("direct", "handoff", "end_conversation"),
                default="handoff",
                context="supervisor decision action",
            ),  # AUDIT-FIX(#1): Reject malformed structured output instead of silently routing the turn incorrectly.
            spoken_ack=_optional_text(payload.get("spoken_ack")),
            spoken_reply=_optional_text(payload.get("spoken_reply")),
            kind=_optional_text(payload.get("kind")),
            goal=_optional_text(payload.get("goal")),
            allow_web_search=_optional_bool(payload.get("allow_web_search")),
            location_hint=_optional_text(payload.get("location_hint")),
            date_context=_optional_text(payload.get("date_context")),
            context_scope=_optional_text(payload.get("context_scope")),
            response_id=getattr(response, "id", None),
            request_id=getattr(response, "_request_id", None),
            model=extract_model_name(response, model),
            token_usage=extract_token_usage(response),
        )


@dataclass
class OpenAIFirstWordProvider:
    """Produce short first-word acknowledgements through the backend.

    Attributes:
        backend: Shared backend instance that owns the OpenAI client.
        model_override: Optional first-word model override.
        reasoning_effort_override: Optional first-word reasoning override.
        base_instructions_override: Optional instruction override for the
            first-word call.
        replace_base_instructions: Included for interface parity with other
            providers; first-word instructions currently merge identically.
    """

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
        """Resolve the model name for first-word responses."""

        override = (self.model_override or "").strip()
        if override:
            return override
        resolved = (self.config.streaming_first_word_model or "").strip()
        return resolved or self.config.default_model  # AUDIT-FIX(#3): Prevent blank first-word model config from turning into an invalid API request.

    def _resolved_reasoning_effort(self) -> str | None:
        """Resolve the reasoning-effort value for first-word responses."""

        override = (self.reasoning_effort_override or "").strip()
        if override:
            return override
        resolved = (self.config.streaming_first_word_reasoning_effort or "").strip()
        return resolved or None

    def _merged_base_instructions(self, instructions: str | None) -> str | None:
        """Merge first-word base instructions with any per-call additions."""

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
        """Return a validated first-word acknowledgement or filler reply."""

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
                reasoning_effort=reasoning_effort,  # AUDIT-FIX(#3,#5): Fallback models like gpt-4o-mini must not inherit unsupported reasoning config.
            )
            request["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "twinr_first_word_reply",
                    "schema": _FIRST_WORD_REPLY_SCHEMA,
                    "strict": True,
                }
            }
            return _create_response_with_reasoning_fallback(
                self.backend,
                request,
                context="first-word reply",  # AUDIT-FIX(#1,#3): Validate final status and retry without unsupported reasoning when needed.
            )

        response, model_used = self.backend._call_with_model_fallback(
            preferred_model,
            _FIRST_WORD_MODEL_FALLBACKS,
            _call,
        )
        payload = _load_json_object(
            _coerce_text(self.backend._extract_output_text(response)),
            context="first-word reply",
        )
        return FirstWordReply(
            mode=_validated_choice(
                payload.get("mode"),
                allowed=("direct", "filler"),
                default="filler",
                context="first-word mode",
            ),  # AUDIT-FIX(#1): Enforce the contract instead of forwarding malformed mode values downstream.
            spoken_text=_coerce_text(payload.get("spoken_text")).strip(),
            response_id=getattr(response, "id", None),
            request_id=getattr(response, "_request_id", None),
            model=extract_model_name(response, model_used),
            token_usage=extract_token_usage(response),
        )


def _is_reasoning_unsupported_error(exc: Exception) -> bool:
    """Detect whether an exception represents unsupported reasoning params."""

    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", {})
        message = str(error.get("message", "")).lower()
        param = str(error.get("param", "")).lower()
        code = str(error.get("code", "")).lower()
        if "reasoning" in param:
            return True  # AUDIT-FIX(#3): Retry both `reasoning` and `reasoning.effort` parameter rejections.
        if "reasoning" in message and ("not supported" in message or "unsupported" in message):
            return True
        if code == "unsupported_parameter" and "reasoning" in message:
            return True
    message = str(exc).lower()
    return "reasoning" in message and ("not supported" in message or "unsupported" in message)


def _apply_reasoning_effort_request(
    backend: Any,
    request: dict[str, Any],
    *,
    model: str,
    reasoning_effort: str | None,
) -> None:
    """Attach normalized reasoning settings when the target model supports it."""

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
    """Return whether a model identifier supports reasoning-effort control."""

    normalized = (model or "").strip().lower()
    if not normalized:
        return False
    return normalized.startswith("gpt-5") or bool(_O_SERIES_MODEL_PATTERN.match(normalized))  # AUDIT-FIX(#5): Reasoning support is defined for gpt-5 and generic o-series models, not a frozen allowlist.


def _optional_text(value: Any) -> str | None:
    """Normalize a value into trimmed optional text."""

    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_bool(value: Any) -> bool | None:
    """Normalize a value into an optional boolean."""

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
    """Copy a function schema and remove unsupported composition keywords."""

    copied = json.loads(json.dumps(schema))
    if copied.get("type") != "function":
        return copied
    parameters = copied.get("parameters")
    if not isinstance(parameters, dict):
        return copied
    for key in ("anyOf", "oneOf", "allOf", "not", "enum"):
        parameters.pop(key, None)
    return copied


def _coerce_text(value: Any) -> str:
    """Convert arbitrary SDK output into a string without propagating None."""

    # AUDIT-FIX(#2): Tool-loop callers expect string output even when the SDK returns None for tool-only turns.
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _validated_choice(
    value: Any,
    *,
    allowed: Sequence[str],
    default: str,
    context: str,
) -> str:
    """Validate that a choice value stays inside an allowed enum domain.

    Raises:
        RuntimeError: If the normalized candidate is not in ``allowed``.
    """

    # AUDIT-FIX(#1): Structured-output fields must stay inside their declared enum domain before they hit downstream routing.
    candidate = str(value or default).strip() or default
    if candidate in allowed:
        return candidate
    raise RuntimeError(f"{context} must be one of {tuple(allowed)!r}, got {candidate!r}")


def _load_json_object(text: str, *, context: str) -> dict[str, Any]:
    """Decode a non-empty JSON object payload.

    Raises:
        RuntimeError: If the payload is empty, malformed, or not a JSON
            object.
    """

    # AUDIT-FIX(#1): Treat empty/malformed structured output as a protocol failure, not a silent default branch.
    payload_text = text.strip()
    if not payload_text:
        raise RuntimeError(f"{context} returned empty structured output")  # AUDIT-FIX(#1): Empty structured-output bodies are protocol violations, not silent defaults.
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{context} returned invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{context} must decode to a JSON object")
    return payload


def _emit_text_delta(
    on_text_delta: Callable[[str], None] | None,
    text: str,
    *,
    context: str,
) -> None:
    """Deliver a text delta callback without letting observer failures escape."""

    # AUDIT-FIX(#2): Streaming callbacks are observer-side effects and must not be allowed to kill the provider response.
    if on_text_delta is None or not text:
        return
    try:
        on_text_delta(text)
    except Exception:
        logger.warning("%s callback failed; continuing without streaming callback delivery", context, exc_info=True)


def _create_response_with_reasoning_fallback(
    backend: Any,
    request: dict[str, Any],
    *,
    context: str,
) -> Any:
    """Create a response and retry once without reasoning if unsupported."""

    # AUDIT-FIX(#1,#3): Centralize create-path status validation and unsupported-reasoning retry logic.
    try:
        response = backend._client.responses.create(**request)
    except Exception as exc:
        if not _is_reasoning_unsupported_error(exc) or "reasoning" not in request:
            raise
        retry_request = dict(request)
        retry_request.pop("reasoning", None)
        response = backend._client.responses.create(**retry_request)
    _validate_response_status(response, context=context)
    return response


def _validate_response_status(response: Any, *, context: str) -> None:
    """Raise when the OpenAI response status is not a completed success.

    Raises:
        RuntimeError: If the response status is present and not ``completed``.
    """

    # AUDIT-FIX(#1): Failed/incomplete SDK responses must surface as errors before blank text is mistaken for success.
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


def _extract_detail_message(detail: Any) -> str | None:
    """Extract a readable detail message from SDK error payloads."""

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


def _provider_bundle_field_names(bundle_cls: type[Any]) -> set[str]:
    """Return supported field names for a provider bundle class."""

    # AUDIT-FIX(#6): Support newer ProviderBundle shapes without breaking older dataclass layouts.
    if is_dataclass(bundle_cls):
        return {field.name for field in fields(bundle_cls)}
    annotations = getattr(bundle_cls, "__annotations__", None)
    if isinstance(annotations, dict):
        return set(annotations)
    return set()


@dataclass
class OpenAIProviderBundle(ProviderBundle):
    """Assemble the canonical OpenAI adapter bundle for Twinr runtime code.

    Attributes:
        backend: Shared backend instance used by the bundle's adapters.
        combined: Composite provider exposing the speech agent triplet.
    """

    backend: OpenAIBackend
    combined: CompositeSpeechAgentProvider

    @classmethod
    def from_backend(cls, backend: OpenAIBackend) -> OpenAIProviderBundle:
        """Build a provider bundle around an existing backend instance."""

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
        for field_name in ("supervisor", "supervisor_provider", "supervisor_decision", "supervisor_decision_provider"):
            if field_name in field_names:
                bundle_kwargs[field_name] = supervisor  # AUDIT-FIX(#6): Wire optional supervisor fields when the ProviderBundle shape supports them.
        for field_name in ("first_word", "first_word_provider"):
            if field_name in field_names:
                bundle_kwargs[field_name] = first_word  # AUDIT-FIX(#6): Wire optional first-word fields without breaking older bundle shapes.
        return cls(**bundle_kwargs)

    @classmethod
    def from_config(cls, config: TwinrConfig) -> OpenAIProviderBundle:
        """Build a provider bundle from configuration by creating a backend."""

        return cls.from_backend(OpenAIBackend(config=config))
