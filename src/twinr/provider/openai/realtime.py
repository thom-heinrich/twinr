from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass
from typing import Any, Callable

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.personality import load_personality_instructions, merge_instructions
from twinr.provider.openai.backend import _should_send_project_header

_DEFAULT_REALTIME_INSTRUCTIONS = (
    "Speak in clear, warm, natural standard German. "
    "Keep responses concise, practical, and easy for a senior user to understand. "
    "Do not use an English accent. "
    "If the user explicitly asks for a printout, use the print_receipt tool with a short focus hint and optional exact text."
)


@dataclass(frozen=True, slots=True)
class OpenAIRealtimeTurn:
    transcript: str
    response_text: str
    response_id: str | None = None


def _default_async_client_factory(config: TwinrConfig) -> Any:
    if not config.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required to use the OpenAI realtime backend")

    try:
        from openai import AsyncOpenAI
    except ImportError as exc:  # pragma: no cover - exercised when dependency is missing at runtime
        raise RuntimeError(
            "The OpenAI SDK is not installed. Run `pip install -e .` in /twinr first."
        ) from exc

    kwargs: dict[str, Any] = {"api_key": config.openai_api_key}
    if _should_send_project_header(config):
        kwargs["project"] = config.openai_project_id
    return AsyncOpenAI(**kwargs)


class OpenAIRealtimeSession:
    def __init__(
        self,
        config: TwinrConfig,
        *,
        client: Any | None = None,
        client_factory: Callable[[TwinrConfig], Any] | None = None,
        base_instructions: str | None = None,
        tool_handlers: dict[str, Callable[[dict[str, Any]], Any]] | None = None,
    ) -> None:
        self.config = config
        factory = client_factory or _default_async_client_factory
        self._client = client or factory(config)
        self._base_instructions = base_instructions if base_instructions is not None else load_personality_instructions(config)
        self._tool_handlers = dict(tool_handlers or {})
        self._runner: asyncio.Runner | None = None
        self._manager = None
        self._connection = None

    def open(self) -> "OpenAIRealtimeSession":
        if self._connection is not None:
            return self

        self._runner = asyncio.Runner()
        self._manager = self._client.realtime.connect(model=self.config.openai_realtime_model)
        try:
            self._connection = self._runner.run(self._manager.__aenter__())
            self._runner.run(self._configure_session())
        except Exception:
            self.close()
            raise
        return self

    def close(self) -> None:
        if self._runner is not None and self._manager is not None:
            try:
                self._runner.run(self._manager.__aexit__(None, None, None))
            except RuntimeError:
                pass
        self._connection = None
        self._manager = None
        if self._runner is not None:
            self._runner.close()
            self._runner = None

    def __enter__(self) -> "OpenAIRealtimeSession":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def run_audio_turn(
        self,
        audio_pcm: bytes,
        *,
        conversation: tuple[tuple[str, str], ...] | None = None,
        on_audio_chunk: Callable[[bytes], None] | None = None,
        on_output_text_delta: Callable[[str], None] | None = None,
    ) -> OpenAIRealtimeTurn:
        if not audio_pcm:
            raise RuntimeError("Realtime turn requires non-empty PCM audio input")
        if self._runner is None or self._connection is None:
            self.open()
        return self._runner.run(
            self._run_audio_turn(
                audio_pcm,
                conversation=conversation,
                on_audio_chunk=on_audio_chunk,
                on_output_text_delta=on_output_text_delta,
            )
        )

    def run_text_turn(
        self,
        prompt: str,
        *,
        conversation: tuple[tuple[str, str], ...] | None = None,
        on_audio_chunk: Callable[[bytes], None] | None = None,
        on_output_text_delta: Callable[[str], None] | None = None,
    ) -> OpenAIRealtimeTurn:
        if not prompt.strip():
            raise RuntimeError("Realtime turn requires a non-empty prompt")
        if self._runner is None or self._connection is None:
            self.open()
        return self._runner.run(
            self._run_text_turn(
                prompt.strip(),
                conversation=conversation,
                on_audio_chunk=on_audio_chunk,
                on_output_text_delta=on_output_text_delta,
            )
        )

    async def _configure_session(self) -> None:
        session: dict[str, Any] = {
            "type": "realtime",
            "output_modalities": ["audio"],
            "instructions": self._session_instructions(),
            "audio": {
                "input": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": self.config.openai_realtime_input_sample_rate,
                    },
                    "noise_reduction": {
                        "type": "far_field",
                    },
                    "transcription": {
                        "model": self.config.openai_realtime_transcription_model,
                        "language": self.config.openai_realtime_language,
                    },
                    "turn_detection": None,
                },
                "output": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": self.config.openai_realtime_input_sample_rate,
                    },
                    "voice": self.config.openai_realtime_voice,
                    "speed": 1.0,
                },
            },
        }
        tools = self._session_tools()
        if tools:
            session["tools"] = tools
            session["tool_choice"] = "auto"
        await self._connection.session.update(session=session)

    async def _run_audio_turn(
        self,
        audio_pcm: bytes,
        *,
        conversation: tuple[tuple[str, str], ...] | None,
        on_audio_chunk: Callable[[bytes], None] | None,
        on_output_text_delta: Callable[[str], None] | None,
    ) -> OpenAIRealtimeTurn:
        await self._seed_conversation(conversation)
        for chunk in self._iter_audio_chunks(audio_pcm):
            await self._connection.input_audio_buffer.append(audio=chunk)
        await self._connection.input_audio_buffer.commit()
        await self._connection.response.create(response={})
        return await self._consume_turn_events(
            on_audio_chunk=on_audio_chunk,
            on_output_text_delta=on_output_text_delta,
        )

    async def _run_text_turn(
        self,
        prompt: str,
        *,
        conversation: tuple[tuple[str, str], ...] | None,
        on_audio_chunk: Callable[[bytes], None] | None,
        on_output_text_delta: Callable[[str], None] | None,
    ) -> OpenAIRealtimeTurn:
        await self._seed_conversation(conversation)
        await self._connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        )
        await self._connection.response.create(response={})
        return await self._consume_turn_events(
            transcript_hint=prompt,
            on_audio_chunk=on_audio_chunk,
            on_output_text_delta=on_output_text_delta,
        )

    async def _consume_turn_events(
        self,
        *,
        transcript_hint: str = "",
        on_audio_chunk: Callable[[bytes], None] | None,
        on_output_text_delta: Callable[[str], None] | None,
    ) -> OpenAIRealtimeTurn:
        input_transcript_fragments: list[str] = []
        response_text_fragments: list[str] = []
        response_id: str | None = None

        while True:
            event = await self._connection.recv()
            event_type = getattr(event, "type", "")

            if event_type == "error":
                raise RuntimeError(self._format_error(event))

            if event_type == "conversation.item.input_audio_transcription.delta":
                delta = str(getattr(event, "delta", ""))
                if delta:
                    input_transcript_fragments.append(delta)
                continue

            if event_type == "conversation.item.input_audio_transcription.completed":
                transcript_hint = str(getattr(event, "transcript", "")).strip() or transcript_hint
                continue

            if event_type in {
                "response.audio_transcript.delta",
                "response.output_audio_transcript.delta",
                "response.text.delta",
                "response.output_text.delta",
            }:
                delta = str(getattr(event, "delta", ""))
                if delta:
                    response_text_fragments.append(delta)
                    if on_output_text_delta is not None:
                        on_output_text_delta(delta)
                continue

            if event_type in {
                "response.audio_transcript.done",
                "response.output_audio_transcript.done",
                "response.text.done",
                "response.output_text.done",
            }:
                final_text = str(getattr(event, "transcript", "") or getattr(event, "text", "")).strip()
                if final_text and not "".join(response_text_fragments).strip():
                    response_text_fragments.append(final_text)
                    if on_output_text_delta is not None:
                        on_output_text_delta(final_text)
                continue

            if event_type in {"response.audio.delta", "response.output_audio.delta"}:
                if on_audio_chunk is not None:
                    on_audio_chunk(base64.b64decode(str(getattr(event, "delta", ""))))
                continue

            if event_type == "response.done":
                response = getattr(event, "response", None)
                response_id = getattr(response, "id", None)
                function_calls = self._extract_function_calls(response)
                if function_calls:
                    await self._handle_function_calls(function_calls)
                    await self._connection.response.create(response={})
                    continue
                if not "".join(response_text_fragments).strip():
                    extracted = self._extract_response_text(response)
                    if extracted:
                        response_text_fragments.append(extracted)
                break

        transcript = transcript_hint or "".join(input_transcript_fragments).strip() or "[voice input]"
        response_text = "".join(response_text_fragments).strip()
        if not response_text:
            raise RuntimeError("Realtime response completed without text transcript")
        return OpenAIRealtimeTurn(
            transcript=transcript,
            response_text=response_text,
            response_id=response_id,
        )

    async def _seed_conversation(self, conversation: tuple[tuple[str, str], ...] | None) -> None:
        if not conversation:
            return
        for role, content in conversation:
            normalized_role = role.strip().lower()
            normalized_content = content.strip()
            if normalized_role not in {"system", "user", "assistant"} or not normalized_content:
                continue
            await self._connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": normalized_role,
                    "content": [
                        {
                            "type": self._content_type_for_role(normalized_role),
                            "text": normalized_content,
                        }
                    ],
                }
            )

    def _iter_audio_chunks(self, audio_pcm: bytes, *, chunk_size: int = 32_768) -> tuple[str, ...]:
        chunks: list[str] = []
        for index in range(0, len(audio_pcm), chunk_size):
            chunk = audio_pcm[index : index + chunk_size]
            if not chunk:
                continue
            chunks.append(base64.b64encode(chunk).decode("ascii"))
        return tuple(chunks)

    def _session_instructions(self) -> str:
        return merge_instructions(
            self._base_instructions,
            _DEFAULT_REALTIME_INSTRUCTIONS,
            self.config.openai_realtime_instructions,
        ) or _DEFAULT_REALTIME_INSTRUCTIONS

    def _session_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        if "print_receipt" in self._tool_handlers:
            tools.append(
                {
                    "type": "function",
                    "name": "print_receipt",
                    "description": (
                        "Print short, user-facing content on the thermal receipt printer "
                        "when the user explicitly asks for a printout. "
                        "Use focus_hint to describe what from the recent context should be printed. "
                        "Optionally pass text when exact printable wording is already known."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "focus_hint": {
                                "type": "string",
                                "description": "Short hint describing what from the recent conversation should be printed.",
                            },
                            "text": {
                                "type": "string",
                                "description": "Optional exact text if the printable wording is already known.",
                            }
                        },
                        "required": [],
                        "additionalProperties": False,
                    },
                }
            )
        return tools

    def _format_error(self, event: Any) -> str:
        error = getattr(event, "error", None)
        if error is None:
            return "Realtime API returned an unknown error"
        message = getattr(error, "message", None)
        code = getattr(error, "code", None)
        if code and message:
            return f"{code}: {message}"
        if message:
            return str(message)
        return str(error)

    def _extract_response_text(self, response: Any) -> str:
        output_items = getattr(response, "output", None) or []
        fragments: list[str] = []
        for item in output_items:
            for content in getattr(item, "content", None) or []:
                transcript = getattr(content, "transcript", None)
                text = getattr(content, "text", None)
                if transcript:
                    fragments.append(str(transcript).strip())
                elif text:
                    fragments.append(str(text).strip())
        return " ".join(fragment for fragment in fragments if fragment).strip()

    def _extract_function_calls(self, response: Any) -> tuple[tuple[str, str, str], ...]:
        output_items = getattr(response, "output", None) or []
        function_calls: list[tuple[str, str, str]] = []
        for item in output_items:
            if str(getattr(item, "type", "")).strip() != "function_call":
                continue
            name = str(getattr(item, "name", "")).strip()
            call_id = str(getattr(item, "call_id", "")).strip()
            arguments = str(getattr(item, "arguments", "") or "{}").strip() or "{}"
            if not name or not call_id:
                continue
            function_calls.append((name, call_id, arguments))
        return tuple(function_calls)

    async def _handle_function_calls(self, function_calls: tuple[tuple[str, str, str], ...]) -> None:
        for name, call_id, arguments_json in function_calls:
            handler = self._tool_handlers.get(name)
            if handler is None:
                output = {"status": "error", "message": f"Unsupported tool: {name}"}
            else:
                arguments = self._parse_tool_arguments(arguments_json)
                try:
                    result = handler(arguments)
                    output = result if result is not None else {"status": "ok"}
                except Exception as exc:
                    output = {"status": "error", "message": str(exc)}
            await self._connection.conversation.item.create(
                item={
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": self._serialize_tool_output(output),
                }
            )

    def _parse_tool_arguments(self, arguments_json: str) -> dict[str, Any]:
        try:
            parsed = json.loads(arguments_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Tool arguments are not valid JSON: {exc.msg}") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("Tool arguments must decode to a JSON object")
        return parsed

    def _serialize_tool_output(self, output: Any) -> str:
        if isinstance(output, str):
            return output
        return json.dumps(output, ensure_ascii=False)

    def _content_type_for_role(self, role: str) -> str:
        if role == "assistant":
            return "output_text"
        return "input_text"
