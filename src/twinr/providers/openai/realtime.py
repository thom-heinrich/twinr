from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable
from zoneinfo import ZoneInfo

from twinr.agent.tools import build_realtime_tool_schemas
from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.language import memory_and_response_contract
from twinr.agent.base_agent.simple_settings import (
    adjustable_settings_context,
)
from twinr.agent.base_agent.personality import load_personality_instructions, merge_instructions
from twinr.ops.usage import TokenUsage, extract_model_name, extract_token_usage
from twinr.providers.openai.backend import _should_send_project_header

_DEFAULT_REALTIME_INSTRUCTIONS = (
    "Keep user-facing replies clear, warm, natural, concise, practical, and easy for a senior user to understand. "
    "If the user explicitly asks for a printout, use the print_receipt tool. "
    "If the user gave exact wording, quoted text, or said exactly this text, you must pass that literal wording in the tool field text. "
    "Use focus_hint only as a short hint about the target content. "
    "If the user asks for any current, external, or otherwise freshness-sensitive information that benefits from web research, first say one short sentence in the configured user-facing language that you are checking the web and that this may take a moment, then call the search_live_info tool. "
    "If the user asks to be reminded later, asks you to set a timer, or says things like erinnere mich, remind me, timer, wecker, or alarm, use the schedule_reminder tool. "
    "For schedule_reminder you must resolve relative times like heute, morgen, uebermorgen, this evening, in ten minutes, and next Monday against the local date/time context and pass due_at as an absolute ISO 8601 datetime with timezone offset. "
    "If the user asks for a recurring scheduled action such as every day, every morning, every week, weekdays, daily news, daily weather, or daily printed headlines, use the time automation tools instead of schedule_reminder. "
    "Use create_time_automation to create a new recurring or one-off automation, list_automations to inspect existing automations, update_time_automation to change one, and delete_automation to remove one. "
    "If the user asks for automations based on PIR motion, the background microphone, quiet periods, or camera presence/object readings, use create_sensor_automation or update_sensor_automation. "
    "For live recurring content like news, weather, or headlines, use content_mode llm_prompt with allow_web_search true. "
    "For printed scheduled output, use delivery printed. For spoken scheduled output, use delivery spoken. "
    "Do not guess a vague time like morning; if the schedule is not concrete enough to run safely, ask a short follow-up question instead of creating the automation. "
    "For sensor automations, only use the supported trigger kinds and require a concrete hold_seconds value for quiet or no-motion requests. "
    "If the user explicitly asks you to remember or update a contact with a phone number, email, relation, or role, use the remember_contact tool. "
    "If the user asks for the phone number, email, or contact details of a remembered person, use the lookup_contact tool. "
    "If the user asks what saved detail is ambiguous, what Twinr is unsure about, or which conflicting memory options exist, use the get_memory_conflicts tool. "
    "If the user clearly identifies which stored option is correct for an open memory conflict, use the resolve_memory_conflict tool with the matching slot_key and selected_memory_id. "
    "If the user explicitly asks you to remember a stable personal preference such as a liked brand, favored shop, disliked food, or similar preference, use the remember_preference tool. "
    "If the user explicitly asks you to remember a future intention or short plan such as wanting to go for a walk today, use the remember_plan tool. "
    "If the user explicitly asks you to remember an important fact for future turns, use the remember_memory tool. "
    "If the user explicitly asks you to change your future speaking style or behavior, use the update_personality tool. "
    "If the user explicitly asks you to remember a stable user-profile fact or preference, use the update_user_profile tool. "
    "For remember_memory, remember_contact, remember_preference, remember_plan, update_user_profile, and update_personality, all semantic text fields must be canonical English. "
    "Keep names, phone numbers, email addresses, IDs, codes, and direct quotes verbatim. "
    "If the user explicitly asks you to change a supported simple device setting such as remembering more or less recent conversation, use the update_simple_setting tool. "
    "Treat direct complaints like you are too forgetful or please remember more as an explicit request to adjust memory_capacity. "
    "Map remember more, less forgetful, keep more context, or remember less to memory_capacity. "
    "If the user asks which voices are available, answer from the supported Twinr voice catalog in the system context instead of saying you do not know. "
    "Use spoken_voice when the user explicitly asks you to change how your voice sounds, for example calmer, warmer, deeper, brighter, or a different named voice. "
    "Resolve descriptive voice requests to the best supported Twinr voice from the system voice catalog and pass that supported voice name to update_simple_setting. "
    "Use speech_speed when the user explicitly asks you to speak slower or faster. "
    "For these bounded simple settings, do not ask an extra confirmation question unless a system message says the current speaker signal is uncertain or unknown. "
    "If the request is ambiguous about the direction or exact value, ask one short follow-up question instead of guessing. "
    "If the user explicitly asks you to create or refresh the local voice profile from the current spoken turn, use the enroll_voice_profile tool. "
    "If the user asks whether a local voice profile exists or wants its current status, use the get_voice_profile_status tool. "
    "If the user explicitly asks you to delete the local voice profile, use the reset_voice_profile tool. "
    "When a system message says the current speaker signal is uncertain or unknown, ask for explicit confirmation before persistent or security-sensitive tool actions and set confirmed=true only after the user clearly confirms. "
    "If the user asks you to look at them, an object, a document, or something they are showing to the camera, call the inspect_camera tool. "
    "If the user clearly wants to stop or pause the conversation for now, call the end_conversation tool and include a short goodbye in spoken_reply so the turn can finish immediately."
)


@dataclass(frozen=True, slots=True)
class OpenAIRealtimeTurn:
    transcript: str
    response_text: str
    response_id: str | None = None
    model: str | None = None
    token_usage: TokenUsage | None = None
    end_conversation: bool = False


@dataclass(frozen=True, slots=True)
class _HandledRealtimeTools:
    names: tuple[str, ...]
    continue_response: bool = True
    immediate_response_text: str | None = None


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
        self._base_instructions_override = base_instructions
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
                    "speed": float(self.config.openai_realtime_speed),
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
        model: str | None = None
        token_usage: TokenUsage | None = None
        end_conversation = False

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
                model = extract_model_name(response)
                token_usage = extract_token_usage(response)
                function_calls = self._extract_function_calls(response)
                if function_calls:
                    handled_tools = await self._handle_function_calls(function_calls)
                    if "end_conversation" in handled_tools.names:
                        end_conversation = True
                    if not handled_tools.continue_response:
                        immediate_text = str(handled_tools.immediate_response_text or "").strip()
                        if immediate_text:
                            response_text_fragments.append(immediate_text)
                            if on_output_text_delta is not None:
                                on_output_text_delta(immediate_text)
                        break
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
            model=model,
            token_usage=token_usage,
            end_conversation=end_conversation,
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
            self._resolve_base_instructions(),
            memory_and_response_contract(self.config.openai_realtime_language),
            _DEFAULT_REALTIME_INSTRUCTIONS,
            self._reminder_time_context(),
            adjustable_settings_context(self.config),
            self.config.openai_realtime_instructions,
        ) or _DEFAULT_REALTIME_INSTRUCTIONS

    def _resolve_base_instructions(self) -> str | None:
        if self._base_instructions_override is not None:
            return self._base_instructions_override
        return load_personality_instructions(self.config)

    def _session_tools(self) -> list[dict[str, Any]]:
        return build_realtime_tool_schemas(self._tool_handlers.keys())

    def _reminder_time_context(self) -> str:
        try:
            zone = ZoneInfo(self.config.local_timezone_name)
            timezone_name = self.config.local_timezone_name
        except Exception:
            zone = ZoneInfo("UTC")
            timezone_name = "UTC"
        now = datetime.now(zone)
        return (
            "Local date/time context for resolving reminders, timers, and scheduled automations: "
            f"{now.strftime('%A, %Y-%m-%d %H:%M:%S %z')} ({timezone_name})."
        )

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

    async def _handle_function_calls(self, function_calls: tuple[tuple[str, str, str], ...]) -> _HandledRealtimeTools:
        handled_names: list[str] = []
        immediate_response_text: str | None = None
        continue_response = True
        for name, call_id, arguments_json in function_calls:
            handler = self._tool_handlers.get(name)
            arguments = self._parse_tool_arguments(arguments_json)
            if handler is None:
                output = {"status": "error", "message": f"Unsupported tool: {name}"}
            else:
                handled_names.append(name)
                try:
                    result = handler(arguments)
                    output = result if result is not None else {"status": "ok"}
                except Exception as exc:
                    output = {"status": "error", "message": str(exc)}
            if (
                name == "end_conversation"
                and len(function_calls) == 1
            ):
                spoken_reply = str(arguments.get("spoken_reply", "") or "").strip()
                if spoken_reply:
                    immediate_response_text = spoken_reply
                    continue_response = False
            await self._connection.conversation.item.create(
                item={
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": self._serialize_tool_output(output),
                }
            )
        return _HandledRealtimeTools(
            names=tuple(handled_names),
            continue_response=continue_response,
            immediate_response_text=immediate_response_text,
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
