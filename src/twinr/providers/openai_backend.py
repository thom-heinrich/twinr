from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator
from typing import Any, Callable, Sequence
import mimetypes

from twinr.config import TwinrConfig
from twinr.personality import load_personality_instructions, merge_instructions

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
STT_MODEL_FALLBACKS = ("whisper-1",)
TTS_MODEL_FALLBACKS = ("tts-1", "tts-1-hd")
_LEGACY_TTS_VOICES = {"nova", "shimmer", "echo", "onyx", "fable", "alloy", "ash", "sage", "coral"}
_LEGACY_TTS_FALLBACK_VOICE = "sage"


@dataclass(frozen=True, slots=True)
class OpenAITextResponse:
    text: str
    response_id: str | None = None
    request_id: str | None = None
    used_web_search: bool = False


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
        self._base_instructions = base_instructions if base_instructions is not None else load_personality_instructions(config)

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
            instructions=merge_instructions(self._base_instructions, instructions),
            allow_web_search=allow_web_search,
            model=self.config.default_model,
            reasoning_effort=self.config.openai_reasoning_effort,
        )
        response = self._client.responses.create(**request)
        return OpenAITextResponse(
            text=self._extract_output_text(response),
            response_id=getattr(response, "id", None),
            request_id=getattr(response, "_request_id", None),
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
            instructions=merge_instructions(self._base_instructions, instructions),
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
            used_web_search=self._used_web_search(response),
        )

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
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": model,
            "input": self._build_input(prompt, conversation),
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
        messages.append(
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt.strip()}],
            }
        )
        return messages

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
