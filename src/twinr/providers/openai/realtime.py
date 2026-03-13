from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable
from zoneinfo import ZoneInfo

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.personality import load_personality_instructions, merge_instructions
from twinr.automations import supported_sensor_trigger_kinds
from twinr.ops.usage import TokenUsage, extract_model_name, extract_token_usage
from twinr.providers.openai.backend import _should_send_project_header

_DEFAULT_REALTIME_INSTRUCTIONS = (
    "Speak in clear, warm, natural standard German. "
    "Keep responses concise, practical, and easy for a senior user to understand. "
    "Do not use an English accent. "
    "If the user explicitly asks for a printout, use the print_receipt tool with a short focus hint and optional exact text. "
    "If the user asks for any current, external, or otherwise freshness-sensitive information that benefits from web research, first say one short German sentence that you are checking the web and that this may take a moment, then call the search_live_info tool. "
    "If the user asks to be reminded later, asks you to set a timer, or says things like erinnere mich, remind me, timer, wecker, or alarm, use the schedule_reminder tool. "
    "For schedule_reminder you must resolve relative times like heute, morgen, uebermorgen, this evening, in ten minutes, and next Monday against the local date/time context and pass due_at as an absolute ISO 8601 datetime with timezone offset. "
    "If the user asks for a recurring scheduled action such as every day, every morning, every week, weekdays, daily news, daily weather, or daily printed headlines, use the time automation tools instead of schedule_reminder. "
    "Use create_time_automation to create a new recurring or one-off automation, list_automations to inspect existing automations, update_time_automation to change one, and delete_automation to remove one. "
    "If the user asks for automations based on PIR motion, the background microphone, quiet periods, or camera presence/object readings, use create_sensor_automation or update_sensor_automation. "
    "For live recurring content like news, weather, or headlines, use content_mode llm_prompt with allow_web_search true. "
    "For printed scheduled output, use delivery printed. For spoken scheduled output, use delivery spoken. "
    "Do not guess a vague time like morning; if the schedule is not concrete enough to run safely, ask a short follow-up question instead of creating the automation. "
    "For sensor automations, only use the supported trigger kinds and require a concrete hold_seconds value for quiet or no-motion requests. "
    "If the user explicitly asks you to remember an important fact for future turns, use the remember_memory tool. "
    "If the user explicitly asks you to change your future speaking style or behavior, use the update_personality tool. "
    "If the user explicitly asks you to remember a stable user-profile fact or preference, use the update_user_profile tool. "
    "If the user explicitly asks you to create or refresh the local voice profile from the current spoken turn, use the enroll_voice_profile tool. "
    "If the user asks whether a local voice profile exists or wants its current status, use the get_voice_profile_status tool. "
    "If the user explicitly asks you to delete the local voice profile, use the reset_voice_profile tool. "
    "When a system message says the current speaker signal is uncertain or unknown, ask for explicit confirmation before persistent or security-sensitive tool actions and set confirmed=true only after the user clearly confirms. "
    "If the user asks you to look at them, an object, a document, or something they are showing to the camera, call the inspect_camera tool. "
    "If the user clearly wants to stop or pause the conversation for now, call the end_conversation tool and then say a short goodbye."
)


@dataclass(frozen=True, slots=True)
class OpenAIRealtimeTurn:
    transcript: str
    response_text: str
    response_id: str | None = None
    model: str | None = None
    token_usage: TokenUsage | None = None
    end_conversation: bool = False


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
                    if "end_conversation" in handled_tools:
                        end_conversation = True
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
            _DEFAULT_REALTIME_INSTRUCTIONS,
            self._reminder_time_context(),
            self.config.openai_realtime_instructions,
        ) or _DEFAULT_REALTIME_INSTRUCTIONS

    def _resolve_base_instructions(self) -> str | None:
        if self._base_instructions_override is not None:
            return self._base_instructions_override
        return load_personality_instructions(self.config)

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
        if "search_live_info" in self._tool_handlers:
            tools.append(
                {
                    "type": "function",
                    "name": "search_live_info",
                    "description": (
                        "Look up fresh or externally verifiable web information for the user. "
                        "Use this for broad web research, not only a fixed list of example domains."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The exact question to research on the web.",
                            },
                            "location_hint": {
                                "type": "string",
                                "description": "Optional location such as a city or district relevant to the search.",
                            },
                            "date_context": {
                                "type": "string",
                                "description": "Optional absolute date or time context if the user referred to relative dates.",
                            },
                        },
                        "required": ["question"],
                        "additionalProperties": False,
                    },
                }
            )
        if "schedule_reminder" in self._tool_handlers:
            tools.append(
                {
                    "type": "function",
                    "name": "schedule_reminder",
                    "description": (
                        "Schedule a future reminder or timer when the user asks to be reminded later or to set a timer. "
                        "Always send due_at as an absolute ISO 8601 local datetime with timezone offset."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "due_at": {
                                "type": "string",
                                "description": "Absolute local due time in ISO 8601 format, for example 2026-03-14T12:00:00+01:00.",
                            },
                            "summary": {
                                "type": "string",
                                "description": "Short summary of what Twinr should remind the user about.",
                            },
                            "details": {
                                "type": "string",
                                "description": "Optional extra detail to include when the reminder is spoken.",
                            },
                            "kind": {
                                "type": "string",
                                "description": "Short type such as reminder, timer, appointment, medication, task, or alarm.",
                            },
                            "original_request": {
                                "type": "string",
                                "description": "Optional short quote or paraphrase of the user's original reminder request.",
                            },
                        },
                        "required": ["due_at", "summary"],
                        "additionalProperties": False,
                    },
                }
            )
        if "list_automations" in self._tool_handlers:
            tools.append(
                {
                    "type": "function",
                    "name": "list_automations",
                    "description": (
                        "List the currently configured time-based and sensor-triggered automations so you can answer questions about them "
                        "or choose one to update or delete."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "include_disabled": {
                                "type": "boolean",
                                "description": "Set true if disabled automations should also be included.",
                            }
                        },
                        "required": [],
                        "additionalProperties": False,
                    },
                }
            )
        if "create_time_automation" in self._tool_handlers:
            tools.append(
                {
                    "type": "function",
                    "name": "create_time_automation",
                    "description": (
                        "Create a time-based automation for one-off or recurring actions such as daily weather, "
                        "daily news, or printed headlines."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Short operator-friendly name for the automation.",
                            },
                            "description": {
                                "type": "string",
                                "description": "Optional short description of what the automation does.",
                            },
                            "schedule": {
                                "type": "string",
                                "enum": ["once", "daily", "weekly"],
                                "description": "Time schedule type.",
                            },
                            "due_at": {
                                "type": "string",
                                "description": "Absolute ISO 8601 local datetime with timezone offset for once schedules.",
                            },
                            "time_of_day": {
                                "type": "string",
                                "description": "Local time in HH:MM for daily or weekly schedules.",
                            },
                            "weekdays": {
                                "type": "array",
                                "description": "Weekday numbers for weekly schedules, where Monday is 0 and Sunday is 6.",
                                "items": {"type": "integer"},
                            },
                            "delivery": {
                                "type": "string",
                                "enum": ["spoken", "printed"],
                                "description": "Whether the automation should speak or print when it runs.",
                            },
                            "content_mode": {
                                "type": "string",
                                "enum": ["llm_prompt", "static_text"],
                                "description": "Use llm_prompt for generated content or static_text for fixed wording.",
                            },
                            "content": {
                                "type": "string",
                                "description": "The prompt or static text the automation should use.",
                            },
                            "allow_web_search": {
                                "type": "boolean",
                                "description": "Set true when the automation needs fresh live information from the web.",
                            },
                            "enabled": {
                                "type": "boolean",
                                "description": "Whether the automation should be active immediately.",
                            },
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional short tags for operator organization.",
                            },
                            "timezone_name": {
                                "type": "string",
                                "description": "Optional timezone name. Use the local Twinr timezone unless there is a clear reason not to.",
                            },
                            "confirmed": {
                                "type": "boolean",
                                "description": "Set true only after the user clearly confirmed this persistent automation change when extra confirmation is needed.",
                            },
                        },
                        "required": ["name", "schedule", "delivery", "content_mode", "content"],
                        "additionalProperties": False,
                    },
                }
            )
        if "create_sensor_automation" in self._tool_handlers:
            tools.append(
                {
                    "type": "function",
                    "name": "create_sensor_automation",
                    "description": (
                        "Create an automation triggered by PIR motion, camera visibility/object readings, "
                        "or background microphone/VAD state."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Short operator-friendly name for the automation.",
                            },
                            "description": {
                                "type": "string",
                                "description": "Optional short description of what the automation does.",
                            },
                            "trigger_kind": {
                                "type": "string",
                                "enum": list(supported_sensor_trigger_kinds()),
                                "description": "Supported sensor trigger type.",
                            },
                            "hold_seconds": {
                                "type": "number",
                                "description": "Optional required hold duration before firing. Required for quiet/no-motion triggers.",
                            },
                            "cooldown_seconds": {
                                "type": "number",
                                "description": "Optional cooldown after the automation fired.",
                            },
                            "delivery": {
                                "type": "string",
                                "enum": ["spoken", "printed"],
                                "description": "Whether the automation should speak or print when it runs.",
                            },
                            "content_mode": {
                                "type": "string",
                                "enum": ["llm_prompt", "static_text"],
                                "description": "Use llm_prompt for generated content or static_text for fixed wording.",
                            },
                            "content": {
                                "type": "string",
                                "description": "The prompt or static text the automation should use.",
                            },
                            "allow_web_search": {
                                "type": "boolean",
                                "description": "Set true when the automation needs fresh live information from the web.",
                            },
                            "enabled": {
                                "type": "boolean",
                                "description": "Whether the automation should be active immediately.",
                            },
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional short tags for operator organization.",
                            },
                            "confirmed": {
                                "type": "boolean",
                                "description": "Set true only after the user clearly confirmed this persistent automation change when extra confirmation is needed.",
                            },
                        },
                        "required": ["name", "trigger_kind", "delivery", "content_mode", "content"],
                        "additionalProperties": False,
                    },
                }
            )
        if "update_time_automation" in self._tool_handlers:
            tools.append(
                {
                    "type": "function",
                    "name": "update_time_automation",
                    "description": (
                        "Update an existing time-based automation. Use list_automations first if you need to identify it."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "automation_ref": {
                                "type": "string",
                                "description": "Automation id or a clear automation name.",
                            },
                            "name": {
                                "type": "string",
                                "description": "Optional new automation name.",
                            },
                            "description": {
                                "type": "string",
                                "description": "Optional new description.",
                            },
                            "schedule": {
                                "type": "string",
                                "enum": ["once", "daily", "weekly"],
                                "description": "Optional new time schedule type.",
                            },
                            "due_at": {
                                "type": "string",
                                "description": "Absolute ISO 8601 local datetime with timezone offset for once schedules.",
                            },
                            "time_of_day": {
                                "type": "string",
                                "description": "Local time in HH:MM for daily or weekly schedules.",
                            },
                            "weekdays": {
                                "type": "array",
                                "description": "Weekday numbers for weekly schedules, where Monday is 0 and Sunday is 6.",
                                "items": {"type": "integer"},
                            },
                            "delivery": {
                                "type": "string",
                                "enum": ["spoken", "printed"],
                                "description": "Optional new delivery mode.",
                            },
                            "content_mode": {
                                "type": "string",
                                "enum": ["llm_prompt", "static_text"],
                                "description": "Optional new content mode.",
                            },
                            "content": {
                                "type": "string",
                                "description": "Optional new prompt or static text.",
                            },
                            "allow_web_search": {
                                "type": "boolean",
                                "description": "Optional new live-search flag for llm_prompt content.",
                            },
                            "enabled": {
                                "type": "boolean",
                                "description": "Optional enabled toggle.",
                            },
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional full replacement tag list.",
                            },
                            "timezone_name": {
                                "type": "string",
                                "description": "Optional new timezone name.",
                            },
                            "confirmed": {
                                "type": "boolean",
                                "description": "Set true only after the user clearly confirmed this persistent automation change when extra confirmation is needed.",
                            },
                        },
                        "required": ["automation_ref"],
                        "additionalProperties": False,
                    },
                }
            )
        if "update_sensor_automation" in self._tool_handlers:
            tools.append(
                {
                    "type": "function",
                    "name": "update_sensor_automation",
                    "description": (
                        "Update an existing supported sensor-triggered automation. Use list_automations first if you need to identify it."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "automation_ref": {
                                "type": "string",
                                "description": "Automation id or a clear automation name.",
                            },
                            "name": {
                                "type": "string",
                                "description": "Optional new automation name.",
                            },
                            "description": {
                                "type": "string",
                                "description": "Optional new description.",
                            },
                            "trigger_kind": {
                                "type": "string",
                                "enum": list(supported_sensor_trigger_kinds()),
                                "description": "Optional new supported sensor trigger type.",
                            },
                            "hold_seconds": {
                                "type": "number",
                                "description": "Optional hold duration before firing.",
                            },
                            "cooldown_seconds": {
                                "type": "number",
                                "description": "Optional cooldown after the automation fired.",
                            },
                            "delivery": {
                                "type": "string",
                                "enum": ["spoken", "printed"],
                                "description": "Optional new delivery mode.",
                            },
                            "content_mode": {
                                "type": "string",
                                "enum": ["llm_prompt", "static_text"],
                                "description": "Optional new content mode.",
                            },
                            "content": {
                                "type": "string",
                                "description": "Optional new prompt or static text.",
                            },
                            "allow_web_search": {
                                "type": "boolean",
                                "description": "Optional new live-search flag for llm_prompt content.",
                            },
                            "enabled": {
                                "type": "boolean",
                                "description": "Optional enabled toggle.",
                            },
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional full replacement tag list.",
                            },
                            "confirmed": {
                                "type": "boolean",
                                "description": "Set true only after the user clearly confirmed this persistent automation change when extra confirmation is needed.",
                            },
                        },
                        "required": ["automation_ref"],
                        "additionalProperties": False,
                    },
                }
            )
        if "delete_automation" in self._tool_handlers:
            tools.append(
                {
                    "type": "function",
                    "name": "delete_automation",
                    "description": "Delete an existing scheduled automation.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "automation_ref": {
                                "type": "string",
                                "description": "Automation id or a clear automation name.",
                            },
                            "confirmed": {
                                "type": "boolean",
                                "description": "Set true only after the user clearly confirmed the deletion when extra confirmation is needed.",
                            },
                        },
                        "required": ["automation_ref"],
                        "additionalProperties": False,
                    },
                }
            )
        if "remember_memory" in self._tool_handlers:
            tools.append(
                {
                    "type": "function",
                    "name": "remember_memory",
                    "description": (
                        "Store an important memory for future turns when the user explicitly asks you to remember something. "
                        "Use only for clear remember/save requests, not for ordinary conversation."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "kind": {
                                "type": "string",
                                "description": "Short type such as appointment, contact, reminder, preference, fact, or task.",
                            },
                            "summary": {
                                "type": "string",
                                "description": "Short factual summary of what should be remembered.",
                            },
                            "details": {
                                "type": "string",
                                "description": "Optional extra detail that helps later recall.",
                            },
                            "confirmed": {
                                "type": "boolean",
                                "description": "Set true only after the user clearly confirmed the persistent save when extra confirmation is needed.",
                            },
                        },
                        "required": ["summary"],
                        "additionalProperties": False,
                    },
                }
            )
        if "update_user_profile" in self._tool_handlers:
            tools.append(
                {
                    "type": "function",
                    "name": "update_user_profile",
                    "description": (
                        "Update stable user profile or preference context for future turns when the user explicitly asks you to remember it."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "description": "Short category such as preferred_name, location, preference, contact, or routine.",
                            },
                            "instruction": {
                                "type": "string",
                                "description": "Short, durable instruction or fact to store in the user profile.",
                            },
                            "confirmed": {
                                "type": "boolean",
                                "description": "Set true only after the user clearly confirmed this persistent profile change when extra confirmation is needed.",
                            },
                        },
                        "required": ["category", "instruction"],
                        "additionalProperties": False,
                    },
                }
            )
        if "update_personality" in self._tool_handlers:
            tools.append(
                {
                    "type": "function",
                    "name": "update_personality",
                    "description": (
                        "Update how Twinr should speak or behave in future turns when the user explicitly asks for a behavior change."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "description": "Short category such as response_style, humor, language, verbosity, or greeting_style.",
                            },
                            "instruction": {
                                "type": "string",
                                "description": "Short future-behavior instruction to store in Twinr personality context.",
                            },
                            "confirmed": {
                                "type": "boolean",
                                "description": "Set true only after the user clearly confirmed this persistent behavior change when extra confirmation is needed.",
                            },
                        },
                        "required": ["category", "instruction"],
                        "additionalProperties": False,
                    },
                }
            )
        if "enroll_voice_profile" in self._tool_handlers:
            tools.append(
                {
                    "type": "function",
                    "name": "enroll_voice_profile",
                    "description": (
                        "Create or refresh the local Twinr voice profile from the current spoken turn. "
                        "Use only when the user explicitly asks Twinr to learn or update their voice profile."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "confirmed": {
                                "type": "boolean",
                                "description": "Set true only after the user clearly confirmed a replacement when extra confirmation is needed.",
                            }
                        },
                        "required": [],
                        "additionalProperties": False,
                    },
                }
            )
        if "get_voice_profile_status" in self._tool_handlers:
            tools.append(
                {
                    "type": "function",
                    "name": "get_voice_profile_status",
                    "description": "Read the local voice-profile status and current live speaker signal.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                }
            )
        if "reset_voice_profile" in self._tool_handlers:
            tools.append(
                {
                    "type": "function",
                    "name": "reset_voice_profile",
                    "description": "Delete the local Twinr voice profile when the user explicitly asks to remove it.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "confirmed": {
                                "type": "boolean",
                                "description": "Set true only after the user clearly confirmed the reset when extra confirmation is needed.",
                            }
                        },
                        "required": [],
                        "additionalProperties": False,
                    },
                }
            )
        if "end_conversation" in self._tool_handlers:
            tools.append(
                {
                    "type": "function",
                    "name": "end_conversation",
                    "description": (
                        "End the current follow-up listening loop when the user clearly indicates they are done for now, "
                        "for example by saying thanks, stop, pause, bye, or tschuss."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "Optional short note describing why the conversation should end.",
                            }
                        },
                        "required": [],
                        "additionalProperties": False,
                    },
                }
            )
        if "inspect_camera" in self._tool_handlers:
            tools.append(
                {
                    "type": "function",
                    "name": "inspect_camera",
                    "description": (
                        "Inspect the current live camera view when the user asks you to look at them, "
                        "an object, a document, or something they are showing."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The exact user request about what should be inspected in the camera view.",
                            }
                        },
                        "required": ["question"],
                        "additionalProperties": False,
                    },
                }
            )
        return tools

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

    async def _handle_function_calls(self, function_calls: tuple[tuple[str, str, str], ...]) -> tuple[str, ...]:
        handled_names: list[str] = []
        for name, call_id, arguments_json in function_calls:
            handler = self._tool_handlers.get(name)
            if handler is None:
                output = {"status": "error", "message": f"Unsupported tool: {name}"}
            else:
                handled_names.append(name)
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
        return tuple(handled_names)

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
