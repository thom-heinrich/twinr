from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections.abc import Iterator
from typing import Any, Callable, Sequence
import mimetypes
import re
from zoneinfo import ZoneInfo

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.personality import load_personality_instructions, merge_instructions
from twinr.ops.usage import TokenUsage, extract_model_name, extract_token_usage

PRINT_FORMATTER_INSTRUCTIONS = (
    "You rewrite assistant answers for a narrow thermal receipt printer. "
    "Keep the output short, concrete, and easy for a senior user to scan. "
    "Use plain text only. Prefer 2 to 4 short lines. Avoid markdown, emojis, and filler."
)
PRINT_COMPOSER_INSTRUCTIONS = (
    "You prepare short thermal printer notes for Twinr. "
    "Use only the provided recent conversation context and any explicit print hint/text. "
    "Infer the most relevant recent information the user likely wants on paper. "
    "Return plain text only, with concise receipt-friendly wording. "
    "Preserve the key concrete facts from the latest relevant exchange, especially dates, times, places, names, numbers, and actionable details. "
    "Do not collapse a multi-fact answer into a vague one-liner if more detail is available. "
    "Aim for 3 to 6 short lines when there is enough concrete content. "
    "Do not invent facts, do not add explanations about formatting, and do not output markdown."
)
SEARCH_AGENT_INSTRUCTIONS = (
    "You are Twinr's live-information search agent. "
    "Use web search to answer any freshness-sensitive or externally verifiable question, not just predefined categories. "
    "Respond in clear standard German for a senior user. "
    "Use plain text only, with no markdown, tables, or bullet lists. "
    "Prefer concrete facts, names, phone numbers, times, weather values, and exact dates when available. "
    "Resolve relative dates like today, tomorrow, heute, morgen, this afternoon, and next Monday against the provided local date/time context. "
    "Answer in at most two short sentences whenever possible. "
    "Keep the spoken answer concise, practical, and trustworthy. "
    "If important uncertainty remains, say so briefly."
)
STT_MODEL_FALLBACKS = ("whisper-1",)
TTS_MODEL_FALLBACKS = ("tts-1", "tts-1-hd")
SEARCH_MODEL_FALLBACKS = ("gpt-5.2-chat-latest",)
_LEGACY_TTS_VOICES = {"nova", "shimmer", "echo", "onyx", "fable", "alloy", "ash", "sage", "coral"}
_LEGACY_TTS_FALLBACK_VOICE = "sage"


@dataclass(frozen=True, slots=True)
class OpenAITextResponse:
    text: str
    response_id: str | None = None
    request_id: str | None = None
    model: str | None = None
    token_usage: TokenUsage | None = None
    used_web_search: bool = False


@dataclass(frozen=True, slots=True)
class OpenAISearchResult:
    answer: str
    sources: tuple[str, ...] = ()
    response_id: str | None = None
    request_id: str | None = None
    model: str | None = None
    token_usage: TokenUsage | None = None
    used_web_search: bool = False


@dataclass(frozen=True, slots=True)
class OpenAIImageInput:
    data: bytes
    content_type: str
    filename: str = "image"
    detail: str | None = None
    label: str | None = None

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        detail: str | None = None,
        label: str | None = None,
    ) -> "OpenAIImageInput":
        file_path = Path(path)
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        return cls(
            data=file_path.read_bytes(),
            content_type=content_type,
            filename=file_path.name,
            detail=detail,
            label=label,
        )


ConversationLike = Sequence[tuple[str, str]] | Sequence[object]


def _default_client_factory(config: TwinrConfig) -> Any:
    if not config.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required to use the OpenAI backend")

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - exercised when dependency is missing at runtime
        raise RuntimeError(
            "The OpenAI SDK is not installed. Run `pip install -e .` in /twinr first."
        ) from exc

    kwargs: dict[str, Any] = {"api_key": config.openai_api_key}
    if _should_send_project_header(config):
        kwargs["project"] = config.openai_project_id
    return OpenAI(**kwargs)


def _should_send_project_header(config: TwinrConfig) -> bool:
    if not config.openai_project_id:
        return False
    if config.openai_send_project_header is not None:
        return config.openai_send_project_header
    api_key = config.openai_api_key or ""
    return not api_key.startswith("sk-proj-")


class OpenAIBackend:
    def __init__(
        self,
        config: TwinrConfig,
        *,
        client: Any | None = None,
        client_factory: Callable[[TwinrConfig], Any] | None = None,
        base_instructions: str | None = None,
    ) -> None:
        self.config = config
        factory = client_factory or _default_client_factory
        self._client = client or factory(config)
        self._base_instructions_override = base_instructions

    def _resolve_base_instructions(self) -> str | None:
        if self._base_instructions_override is not None:
            return self._base_instructions_override
        return load_personality_instructions(self.config)

    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
        language: str | None = None,
        prompt: str | None = None,
    ) -> str:
        request: dict[str, Any] = {
            "model": self.config.openai_stt_model,
            "file": (filename, audio_bytes, content_type),
            "response_format": "text",
        }
        if language:
            request["language"] = language
        if prompt:
            request["prompt"] = prompt
        response, _model_used = self._call_with_model_fallback(
            self.config.openai_stt_model,
            STT_MODEL_FALLBACKS,
            lambda model: self._client.audio.transcriptions.create(**{**request, "model": model}),
        )
        if isinstance(response, str):
            return response.strip()
        return str(getattr(response, "text", "")).strip()

    def transcribe_path(
        self,
        path: str | Path,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> str:
        file_path = Path(path)
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        return self.transcribe(
            file_path.read_bytes(),
            filename=file_path.name,
            content_type=content_type,
            language=language,
            prompt=prompt,
        )

    def respond(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
    ) -> str:
        return self.respond_with_metadata(
            prompt,
            conversation=conversation,
            instructions=instructions,
            allow_web_search=allow_web_search,
        ).text

    def respond_with_metadata(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
    ) -> OpenAITextResponse:
        request = self._build_response_request(
            prompt,
            conversation=conversation,
            instructions=merge_instructions(self._resolve_base_instructions(), instructions),
            allow_web_search=allow_web_search,
            model=self.config.default_model,
            reasoning_effort=self.config.openai_reasoning_effort,
        )
        response = self._client.responses.create(**request)
        return OpenAITextResponse(
            text=self._extract_output_text(response),
            response_id=getattr(response, "id", None),
            request_id=getattr(response, "_request_id", None),
            model=extract_model_name(response, request["model"]),
            token_usage=extract_token_usage(response),
            used_web_search=self._used_web_search(response),
        )

    def respond_to_images(
        self,
        prompt: str,
        *,
        images: Sequence[OpenAIImageInput],
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
    ) -> str:
        return self.respond_to_images_with_metadata(
            prompt,
            images=images,
            conversation=conversation,
            instructions=instructions,
            allow_web_search=allow_web_search,
        ).text

    def respond_to_images_with_metadata(
        self,
        prompt: str,
        *,
        images: Sequence[OpenAIImageInput],
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
    ) -> OpenAITextResponse:
        if not images:
            raise ValueError("At least one image is required for a vision request")

        request = self._build_response_request(
            prompt,
            conversation=conversation,
            instructions=merge_instructions(self._resolve_base_instructions(), instructions),
            allow_web_search=allow_web_search,
            model=self.config.default_model,
            reasoning_effort=self.config.openai_reasoning_effort,
            extra_user_content=self._build_image_content(images),
        )
        response = self._client.responses.create(**request)
        return OpenAITextResponse(
            text=self._extract_output_text(response),
            response_id=getattr(response, "id", None),
            request_id=getattr(response, "_request_id", None),
            model=extract_model_name(response, request["model"]),
            token_usage=extract_token_usage(response),
            used_web_search=self._used_web_search(response),
        )

    def respond_streaming(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> OpenAITextResponse:
        request = self._build_response_request(
            prompt,
            conversation=conversation,
            instructions=merge_instructions(self._resolve_base_instructions(), instructions),
            allow_web_search=allow_web_search,
            model=self.config.default_model,
            reasoning_effort=self.config.openai_reasoning_effort,
        )
        with self._client.responses.stream(**request) as stream:
            for event in stream:
                if getattr(event, "type", None) == "response.output_text.delta" and on_text_delta is not None:
                    delta = str(getattr(event, "delta", ""))
                    if delta:
                        on_text_delta(delta)
            response = stream.get_final_response()
        return OpenAITextResponse(
            text=self._extract_output_text(response),
            response_id=getattr(response, "id", None),
            request_id=getattr(response, "_request_id", None),
            model=extract_model_name(response, request["model"]),
            token_usage=extract_token_usage(response),
            used_web_search=self._used_web_search(response),
        )

    def search_live_info_with_metadata(
        self,
        question: str,
        *,
        conversation: ConversationLike | None = None,
        location_hint: str | None = None,
        date_context: str | None = None,
    ) -> OpenAISearchResult:
        normalized_question = question.strip()
        if not normalized_question:
            raise RuntimeError("search_live_info requires a non-empty question")
        prompt = self._build_search_prompt(
            normalized_question,
            location_hint=location_hint,
            date_context=date_context,
        )
        instructions = merge_instructions(self._resolve_base_instructions(), SEARCH_AGENT_INSTRUCTIONS)
        best_result: OpenAISearchResult | None = None

        for model in self._candidate_search_models():
            for max_output_tokens in (320, 480):
                request = self._build_response_request(
                    prompt,
                    conversation=conversation,
                    instructions=instructions,
                    allow_web_search=True,
                    model=model,
                    reasoning_effort="medium",
                    max_output_tokens=max_output_tokens,
                )
                request["include"] = ["web_search_call.action.sources"]
                response = self._client.responses.create(**request)
                candidate = OpenAISearchResult(
                    answer=self._sanitize_search_answer(self._extract_output_text(response)),
                    sources=self._extract_web_search_sources(response),
                    response_id=getattr(response, "id", None),
                    request_id=getattr(response, "_request_id", None),
                    model=extract_model_name(response, model),
                    token_usage=extract_token_usage(response),
                    used_web_search=self._used_web_search(response),
                )
                if candidate.answer and not self._response_has_incomplete_message(response):
                    return candidate
                if candidate.answer and (best_result is None or len(candidate.answer) > len(best_result.answer)):
                    best_result = candidate

        if best_result is not None:
            return best_result
        raise RuntimeError("OpenAI web search returned no usable answer text")

    def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        response_format: str | None = None,
        instructions: str | None = None,
    ) -> bytes:
        tts_instructions = instructions or self.config.openai_tts_instructions
        response, _model_used = self._call_with_model_fallback(
            self.config.openai_tts_model,
            TTS_MODEL_FALLBACKS,
            lambda model: self._client.audio.speech.create(
                **self._build_tts_request(
                    text,
                    model=model,
                    voice=voice,
                    response_format=response_format,
                    instructions=tts_instructions,
                )
            ),
        )
        if hasattr(response, "read"):
            return bytes(response.read())
        if hasattr(response, "content"):
            return bytes(response.content)
        if isinstance(response, (bytes, bytearray)):
            return bytes(response)
        raise RuntimeError("Unexpected speech synthesis response type")

    def synthesize_stream(
        self,
        text: str,
        *,
        voice: str | None = None,
        response_format: str | None = None,
        instructions: str | None = None,
        chunk_size: int = 4096,
    ) -> Iterator[bytes]:
        tts_instructions = instructions or self.config.openai_tts_instructions

        def iterator() -> Iterator[bytes]:
            attempted_models: list[str] = []
            last_error: Exception | None = None
            for model in (self.config.openai_tts_model, *TTS_MODEL_FALLBACKS):
                if not model or model in attempted_models:
                    continue
                attempted_models.append(model)
                try:
                    with self._client.audio.speech.with_streaming_response.create(
                        **self._build_tts_request(
                            text,
                            model=model,
                            voice=voice,
                            response_format=response_format,
                            instructions=tts_instructions,
                        )
                    ) as response:
                        for chunk in response.iter_bytes(chunk_size):
                            if chunk:
                                yield bytes(chunk)
                    return
                except Exception as exc:
                    if not self._is_model_access_error(exc):
                        raise
                    last_error = exc
            if last_error is not None:
                candidate_list = ", ".join(attempted_models)
                raise RuntimeError(
                    f"OpenAI project does not have access to any candidate models for this request: {candidate_list}"
                ) from last_error
            raise RuntimeError("No model candidates were available for the OpenAI request")

        return iterator()

    def _build_tts_request(
        self,
        text: str,
        *,
        model: str,
        voice: str | None,
        response_format: str | None,
        instructions: str | None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": model,
            "voice": self._resolve_tts_voice(model, voice or self.config.openai_tts_voice),
            "input": text.strip(),
            "response_format": response_format or self.config.openai_tts_format,
        }
        if instructions:
            request["instructions"] = instructions
        return request

    def _resolve_tts_voice(self, model: str, requested_voice: str) -> str:
        normalized_model = model.strip().lower()
        if normalized_model in {"tts-1", "tts-1-hd"} and requested_voice not in _LEGACY_TTS_VOICES:
            return _LEGACY_TTS_FALLBACK_VOICE
        return requested_voice

    def format_for_print(self, text: str) -> str:
        return self.format_for_print_with_metadata(text).text

    def format_for_print_with_metadata(self, text: str) -> OpenAITextResponse:
        request = self._build_response_request(
            text,
            instructions=PRINT_FORMATTER_INSTRUCTIONS,
            allow_web_search=False,
            model=self.config.default_model,
            reasoning_effort="low",
            max_output_tokens=140,
        )
        response = self._client.responses.create(**request)
        return OpenAITextResponse(
            text=self._extract_output_text(response),
            response_id=getattr(response, "id", None),
            request_id=getattr(response, "_request_id", None),
            model=extract_model_name(response, request["model"]),
            token_usage=extract_token_usage(response),
            used_web_search=False,
        )

    def compose_print_job(
        self,
        *,
        conversation: ConversationLike | None = None,
        focus_hint: str | None = None,
        direct_text: str | None = None,
        request_source: str = "button",
    ) -> str:
        return self.compose_print_job_with_metadata(
            conversation=conversation,
            focus_hint=focus_hint,
            direct_text=direct_text,
            request_source=request_source,
        ).text

    def compose_print_job_with_metadata(
        self,
        *,
        conversation: ConversationLike | None = None,
        focus_hint: str | None = None,
        direct_text: str | None = None,
        request_source: str = "button",
    ) -> OpenAITextResponse:
        prompt = self._build_print_composer_prompt(
            conversation=conversation,
            focus_hint=focus_hint,
            direct_text=direct_text,
            request_source=request_source,
        )
        request = self._build_response_request(
            prompt,
            conversation=self._limit_print_conversation(conversation),
            instructions=self._print_composer_instructions(),
            allow_web_search=False,
            model=self.config.default_model,
            reasoning_effort="medium",
            max_output_tokens=180,
        )
        response = self._client.responses.create(**request)
        candidate_text = self._sanitize_print_text(self._extract_output_text(response))
        fallback_source = self._fallback_print_source(conversation, direct_text)
        final_text = candidate_text
        if self._should_use_print_fallback(candidate_text, fallback_source):
            fallback = self.format_for_print_with_metadata(fallback_source)
            final_text = self._sanitize_print_text(fallback.text)
        return OpenAITextResponse(
            text=final_text,
            response_id=getattr(response, "id", None),
            request_id=getattr(response, "_request_id", None),
            model=extract_model_name(response, request["model"]),
            token_usage=extract_token_usage(response),
            used_web_search=False,
        )

    def _call_with_model_fallback(
        self,
        preferred_model: str,
        fallback_models: Sequence[str],
        call: Callable[[str], Any],
    ) -> tuple[Any, str]:
        attempted_models: list[str] = []
        last_error: Exception | None = None
        for model in (preferred_model, *fallback_models):
            if not model or model in attempted_models:
                continue
            attempted_models.append(model)
            try:
                return call(model), model
            except Exception as exc:
                if not self._is_model_access_error(exc):
                    raise
                last_error = exc
        if last_error is not None:
            candidate_list = ", ".join(attempted_models)
            raise RuntimeError(
                f"OpenAI project does not have access to any candidate models for this request: {candidate_list}"
            ) from last_error
        raise RuntimeError("No model candidates were available for the OpenAI request")

    def _is_model_access_error(self, exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        body = getattr(exc, "body", None)
        error_code = None
        if isinstance(body, dict):
            error_code = body.get("error", {}).get("code")
        message = str(exc).lower()
        return (
            error_code == "model_not_found"
            or "does not have access to model" in message
            or "model_not_found" in message
            or status_code in {403, 404}
        )

    def _build_response_request(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
        model: str,
        reasoning_effort: str,
        max_output_tokens: int | None = None,
        extra_user_content: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": model,
            "input": self._build_input(prompt, conversation, extra_user_content=extra_user_content),
            "reasoning": {"effort": reasoning_effort},
            "store": False,
        }
        if instructions:
            request["instructions"] = instructions
        if max_output_tokens is not None:
            request["max_output_tokens"] = max_output_tokens

        use_web_search = self.config.openai_enable_web_search if allow_web_search is None else allow_web_search
        tools = self._build_tools(use_web_search)
        if tools:
            request["tools"] = tools
            request["tool_choice"] = "auto"
        return request

    def _build_input(
        self,
        prompt: str,
        conversation: ConversationLike | None = None,
        *,
        extra_user_content: Sequence[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if conversation:
            for item in conversation:
                role, content = self._coerce_message(item)
                if not content:
                    continue
                messages.append(
                    {
                        "role": role,
                        "content": [{"type": self._content_type_for_role(role), "text": content}],
                    }
                )
        user_content: list[dict[str, Any]] = []
        prompt_text = prompt.strip()
        if prompt_text:
            user_content.append({"type": "input_text", "text": prompt_text})
        if extra_user_content:
            user_content.extend(extra_user_content)
        messages.append({"role": "user", "content": user_content})
        return messages

    def _build_image_content(self, images: Sequence[OpenAIImageInput]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for image in images:
            if image.label:
                content.append({"type": "input_text", "text": image.label})
            image_item: dict[str, Any] = {
                "type": "input_image",
                "image_url": self._image_data_url(image),
            }
            detail = (image.detail or self.config.openai_vision_detail or "").strip()
            if detail:
                image_item["detail"] = detail
            content.append(image_item)
        return content

    def _image_data_url(self, image: OpenAIImageInput) -> str:
        if not image.content_type.startswith("image/"):
            raise ValueError(f"Unsupported image content type: {image.content_type}")
        encoded = base64.b64encode(image.data).decode("ascii")
        return f"data:{image.content_type};base64,{encoded}"

    def _coerce_message(self, item: object) -> tuple[str, str]:
        if isinstance(item, tuple) and len(item) == 2:
            role, content = item
            return str(role), str(content).strip()
        role = str(getattr(item, "role"))
        content = str(getattr(item, "content")).strip()
        return role, content

    def _content_type_for_role(self, role: str) -> str:
        if role.strip().lower() == "assistant":
            return "output_text"
        return "input_text"

    def _build_tools(self, use_web_search: bool) -> list[dict[str, Any]]:
        if not use_web_search:
            return []

        tool: dict[str, Any] = {
            "type": "web_search",
            "search_context_size": self.config.openai_web_search_context_size,
        }
        user_location = self._build_user_location()
        if user_location:
            tool["user_location"] = user_location
        return [tool]

    def _build_user_location(self) -> dict[str, str] | None:
        fields = {
            "type": "approximate",
            "country": self.config.openai_web_search_country,
            "region": self.config.openai_web_search_region,
            "city": self.config.openai_web_search_city,
            "timezone": self.config.openai_web_search_timezone,
        }
        if not any(value for key, value in fields.items() if key != "type"):
            return None
        return {key: value for key, value in fields.items() if value}

    def _build_search_prompt(
        self,
        question: str,
        *,
        location_hint: str | None,
        date_context: str | None,
    ) -> str:
        parts = [f"User question: {question}"]
        resolved_location = (location_hint or self.config.openai_web_search_city or "").strip()
        if resolved_location:
            parts.append(f"Location hint: {resolved_location}")
        resolved_date_context = (date_context or self._relative_date_context()).strip()
        if resolved_date_context:
            parts.append(f"Local date/time context: {resolved_date_context}")
        parts.append("Answer now with the best live information you can verify from web search.")
        return "\n".join(parts)

    def _extract_output_text(self, response: Any) -> str:
        text = str(getattr(response, "output_text", "")).strip()
        if text:
            return text

        output_items = getattr(response, "output", None) or []
        fragments: list[str] = []
        for item in output_items:
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", None) in {"output_text", "text"} and getattr(content, "text", None):
                    fragments.append(str(content.text).strip())
        return "\n".join(fragment for fragment in fragments if fragment).strip()

    def _print_composer_instructions(self) -> str:
        return (
            f"{PRINT_COMPOSER_INSTRUCTIONS} "
            f"Never exceed {self.config.print_max_lines} lines or {self.config.print_max_chars} characters. "
            "If there is no clear printable content, return exactly: NO_PRINT_CONTENT"
        )

    def _build_print_composer_prompt(
        self,
        *,
        conversation: ConversationLike | None,
        focus_hint: str | None,
        direct_text: str | None,
        request_source: str,
    ) -> str:
        safe_focus_hint = (focus_hint or "").strip()[:240]
        safe_direct_text = (direct_text or "").strip()[: max(self.config.print_max_chars * 2, 240)]
        latest_user, latest_assistant = self._latest_print_exchange(conversation)
        parts = [
            f"Request source: {request_source.strip() or 'button'}",
            "Interpretation rule: if the request came from the print button, assume the user wants a compact print of the latest exchange.",
            f"Focus hint: {safe_focus_hint or '[none]'}",
            f"Latest user message: {latest_user or '[none]'}",
            f"Latest assistant message: {latest_assistant or '[none]'}",
            f"Direct print text: {safe_direct_text or '[none]'}",
            "Create the final thermal receipt text now.",
        ]
        return "\n".join(parts)

    def _limit_print_conversation(self, conversation: ConversationLike | None) -> ConversationLike | None:
        if not conversation:
            return conversation
        turns = list(conversation)
        if self.config.print_context_turns <= 0 or len(turns) <= self.config.print_context_turns:
            return turns
        return turns[-self.config.print_context_turns :]

    def _sanitize_print_text(self, text: str) -> str:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if normalized == "NO_PRINT_CONTENT":
            raise RuntimeError("No clear printable content is available")

        cleaned_lines: list[str] = []
        for raw_line in normalized.split("\n"):
            compact = " ".join(raw_line.strip().split())
            if compact:
                cleaned_lines.append(compact)

        if not cleaned_lines:
            raise RuntimeError("Print composer returned empty output")

        truncated = len(cleaned_lines) > self.config.print_max_lines
        cleaned_lines = cleaned_lines[: self.config.print_max_lines]
        result = "\n".join(cleaned_lines)
        if len(result) > self.config.print_max_chars:
            result = result[: self.config.print_max_chars].rstrip()
            truncated = True
        if truncated and result and not result.endswith("…"):
            if len(result) >= self.config.print_max_chars:
                result = result[:-1].rstrip()
            result = result.rstrip(" .") + "…"
        return result.strip()

    def _latest_print_exchange(self, conversation: ConversationLike | None) -> tuple[str | None, str | None]:
        latest_user: str | None = None
        latest_assistant: str | None = None
        if not conversation:
            return None, None
        for item in conversation:
            role, content = self._coerce_message(item)
            normalized_role = role.strip().lower()
            if normalized_role == "user" and content:
                latest_user = content
            elif normalized_role == "assistant" and content:
                latest_assistant = content
        return latest_user, latest_assistant

    def _fallback_print_source(
        self,
        conversation: ConversationLike | None,
        direct_text: str | None,
    ) -> str:
        if direct_text and direct_text.strip():
            return direct_text.strip()
        _latest_user, latest_assistant = self._latest_print_exchange(conversation)
        return (latest_assistant or "").strip()

    def _should_use_print_fallback(self, candidate_text: str, fallback_source: str) -> bool:
        if not fallback_source:
            return False
        line_count = len([line for line in candidate_text.splitlines() if line.strip()])
        candidate_length = len(candidate_text.strip())
        source_length = len(fallback_source.strip())
        return source_length >= 90 and (line_count < 2 or candidate_length < 48)

    def _used_web_search(self, response: Any) -> bool:
        for item in getattr(response, "output", None) or []:
            if getattr(item, "type", None) in {"web_search_call", "web_search_preview_call"}:
                return True
        return False

    def _extract_web_search_sources(self, response: Any) -> tuple[str, ...]:
        urls: list[str] = []
        for item in getattr(response, "output", None) or []:
            if getattr(item, "type", None) not in {"web_search_call", "web_search_preview_call"}:
                continue
            action = getattr(item, "action", None)
            sources = getattr(action, "sources", None)
            if sources is None and isinstance(action, dict):
                sources = action.get("sources")
            for source in sources or []:
                url = getattr(source, "url", None)
                if url is None and isinstance(source, dict):
                    url = source.get("url")
                normalized = str(url or "").strip()
                if normalized and normalized not in urls:
                    urls.append(normalized)
        return tuple(urls)

    def _relative_date_context(self) -> str:
        timezone_name = self.config.openai_web_search_timezone or "Europe/Berlin"
        try:
            zone = ZoneInfo(timezone_name)
        except Exception:
            zone = ZoneInfo("UTC")
            timezone_name = "UTC"
        now = datetime.now(zone)
        return f"{now.strftime('%A, %Y-%m-%d %H:%M')} ({timezone_name})"

    def _candidate_search_models(self) -> tuple[str, ...]:
        candidates: list[str] = []
        for model in (
            self.config.openai_search_model,
            *SEARCH_MODEL_FALLBACKS,
            self.config.default_model,
        ):
            normalized = str(model or "").strip()
            if normalized and normalized not in candidates:
                candidates.append(normalized)
        return tuple(candidates)

    def _response_has_incomplete_message(self, response: Any) -> bool:
        for item in getattr(response, "output", None) or []:
            if getattr(item, "type", None) != "message":
                continue
            if getattr(item, "status", None) == "incomplete":
                return True
        return False

    def _sanitize_search_answer(self, text: str) -> str:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("**", "").strip()
        if not normalized:
            return normalized
        cleaned_lines: list[str] = []
        for raw_line in normalized.split("\n"):
            line = re.sub(r"^[-*•]\s*", "", raw_line.strip())
            line = " ".join(line.split())
            if line:
                cleaned_lines.append(line)
        cleaned = " ".join(cleaned_lines).strip() or normalized
        while cleaned and cleaned[-1] in "([{-–,:;":
            cleaned = cleaned[:-1].rstrip()
        return cleaned
