"""Drive OpenAI realtime session lifecycle and turn orchestration.

Exports the synchronous session wrapper that keeps a dedicated background
event loop, configures Twinr-specific realtime instructions and tools, and
turns streamed provider events into bounded ``OpenAIRealtimeTurn`` results.
Import the public surface from ``twinr.providers.openai.realtime`` rather
than this implementation module.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
import threading
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any, Callable

from twinr.agent.tools import build_realtime_tool_schemas
from twinr.agent.tools.prompting import build_tool_agent_instructions
from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.conversation.language import memory_and_response_contract
from twinr.agent.base_agent.prompting.personality import load_personality_instructions, merge_instructions
from twinr.ops.usage import TokenUsage, extract_model_name, extract_token_usage
from twinr.providers.openai.core.client import _should_send_project_header

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OpenAIRealtimeTurn:
    """Represent one completed realtime turn.

    Attributes:
        transcript: Final user transcript for the turn. Falls back to a
            transcript hint or ``"[voice input]"`` when the provider does
            not return a completed transcription event.
        response_text: Final assistant text transcript for the turn.
        response_id: Provider response identifier when available.
        model: Provider model identifier extracted from the response payload.
        token_usage: Token accounting extracted from the response payload.
        end_conversation: Whether the handled tools ended the conversation.
    """

    transcript: str
    response_text: str
    response_id: str | None = None
    model: str | None = None
    token_usage: TokenUsage | None = None
    end_conversation: bool = False


@dataclass(frozen=True, slots=True)
class _HandledRealtimeTools:
    """Capture the tool calls handled during a response cycle.

    Attributes:
        names: Tool names that executed successfully.
        continue_response: Whether the model should continue after tool output
            has been returned.
        immediate_response_text: Text to emit immediately when a tool ends the
            turn without asking the model for a follow-up response.
    """

    names: tuple[str, ...]
    continue_response: bool = True
    immediate_response_text: str | None = None


def _default_async_client_factory(config: TwinrConfig) -> Any:
    """Build the default async OpenAI client for realtime sessions.

    Args:
        config: Twinr configuration with OpenAI credentials and optional
            realtime client timeout settings.

    Returns:
        An ``AsyncOpenAI`` client configured for realtime session use.

    Raises:
        RuntimeError: If the API key is missing or the OpenAI SDK is not
            installed in the runtime environment.
    """

    if not config.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required to use the OpenAI realtime backend")

    try:
        from openai import AsyncOpenAI
    except ImportError as exc:  # pragma: no cover - exercised when dependency is missing at runtime
        raise RuntimeError(
            "The OpenAI SDK is not installed. Run `pip install -e .` in /twinr first."
        ) from exc

    kwargs: dict[str, Any] = {"api_key": config.openai_api_key}
    client_timeout_seconds = _coerce_optional_float(
        getattr(config, "openai_realtime_client_timeout_seconds", None),
        default=30.0,
        minimum=0.1,
    )
    if client_timeout_seconds is not None:
        kwargs["timeout"] = client_timeout_seconds
    client_max_retries = _coerce_optional_int(
        getattr(config, "openai_realtime_client_max_retries", None),
        default=2,
        minimum=0,
    )
    if client_max_retries is not None:
        kwargs["max_retries"] = client_max_retries
    if _should_send_project_header(config):
        kwargs["project"] = config.openai_project_id
    return AsyncOpenAI(**kwargs)


def _coerce_optional_float(value: Any, *, default: float | None, minimum: float) -> float | None:
    """Normalize an optional float config value within safe bounds.

    Args:
        value: Raw config value to interpret.
        default: Value to return when coercion fails or the input is absent.
        minimum: Lowest accepted numeric value.

    Returns:
        The coerced float when it is valid and above ``minimum``; otherwise
        ``default``.
    """

    if value is None:
        return default
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return default
    if coerced < minimum:
        return default
    return coerced


def _coerce_optional_int(value: Any, *, default: int | None, minimum: int) -> int | None:
    """Normalize an optional integer config value within safe bounds.

    Args:
        value: Raw config value to interpret.
        default: Value to return when coercion fails or the input is absent.
        minimum: Lowest accepted integer value.

    Returns:
        The coerced integer when it is valid and above ``minimum``; otherwise
        ``default``.
    """

    if value is None:
        return default
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return default
    if coerced < minimum:
        return default
    return coerced


class OpenAIRealtimeSession:
    """Manage a live OpenAI realtime websocket session for Twinr.

    This synchronous wrapper owns a dedicated background event loop so Twinr's
    runtime can drive OpenAI's async realtime API without exposing asyncio to
    higher layers. It keeps provider waits bounded, refreshes dynamic session
    instructions before each turn, and normalizes streamed text, audio, and
    tool events into ``OpenAIRealtimeTurn`` results.
    """

    def __init__(
        self,
        config: TwinrConfig,
        *,
        client: Any | None = None,
        client_factory: Callable[[TwinrConfig], Any] | None = None,
        base_instructions: str | None = None,
        tool_handlers: dict[str, Callable[[dict[str, Any]], Any]] | None = None,
    ) -> None:
        """Initialize the realtime session wrapper.

        Args:
            config: Twinr configuration with realtime transport and prompt
                settings.
            client: Prebuilt OpenAI client for tests or custom wiring. When
                omitted, ``client_factory`` builds the default client.
            client_factory: Factory for constructing the async OpenAI client
                when ``client`` is absent.
            base_instructions: Optional instruction override that replaces the
                personality-derived base instructions.
            tool_handlers: Mapping from realtime tool names to handler
                callables. Handlers may be sync, async, or return awaitables.
        """

        self.config = config
        factory = client_factory or _default_async_client_factory
        self._client = client or factory(config)
        self._base_instructions_override = base_instructions
        self._tool_handlers = dict(tool_handlers or {})
        self._state_lock = threading.RLock()  # AUDIT-FIX(#1): serialize lifecycle/turn operations so the sync API is safe from concurrent callers.
        self._loop: asyncio.AbstractEventLoop | None = None  # AUDIT-FIX(#1): use a dedicated background event loop instead of asyncio.Runner in the caller thread.
        self._loop_thread: threading.Thread | None = None  # AUDIT-FIX(#1): keep realtime I/O off the uvicorn event loop thread.
        self._loop_started = threading.Event()  # AUDIT-FIX(#1): wait until the background loop is ready before submitting work.
        self._manager = None
        self._connection = None
        self._conversation_seeded = False  # AUDIT-FIX(#5): seed external conversation history at most once per live provider session.
        self._turns_completed = 0  # AUDIT-FIX(#5): if external history is supplied after live turns, reconnect before reseeding to avoid duplicate history.

    def open(self) -> "OpenAIRealtimeSession":
        """Open the realtime provider session and return this instance."""
        with self._state_lock:
            self._open_locked()
            return self

    def close(self) -> None:
        """Close the realtime provider session and background loop."""
        with self._state_lock:
            self._close_locked()

    def __enter__(self) -> "OpenAIRealtimeSession":
        """Enter the context manager with an open provider session."""
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        """Exit the context manager and close the provider session."""
        self.close()

    def __del__(self) -> None:
        """Attempt best-effort cleanup when the session is garbage-collected."""
        # AUDIT-FIX(#6): best-effort cleanup for forgotten sessions; never let GC-time cleanup raise.
        try:
            self.close()
        except Exception:
            logger.warning("OpenAI realtime session cleanup failed during garbage collection.", exc_info=True)

    def run_audio_turn(
        self,
        audio_pcm: bytes,
        *,
        conversation: tuple[tuple[str, str], ...] | None = None,
        on_audio_chunk: Callable[[bytes], None] | None = None,
        on_output_text_delta: Callable[[str], None] | None = None,
    ) -> OpenAIRealtimeTurn:
        """Run one audio turn against the live realtime session.

        Args:
            audio_pcm: Raw PCM audio bytes at the configured sample rate.
            conversation: Optional persisted conversation history to seed into
                a fresh provider session.
            on_audio_chunk: Optional callback for streamed assistant audio
                chunks.
            on_output_text_delta: Optional callback for streamed assistant text
                fragments.

        Returns:
            The completed realtime turn with transcript, response text, and
            provider metadata.

        Raises:
            RuntimeError: If the audio is empty or the provider/session work
                fails.
        """

        if not audio_pcm:
            raise RuntimeError("Realtime turn requires non-empty PCM audio input")
        normalized_conversation = self._normalize_conversation(conversation)  # AUDIT-FIX(#12): tolerate malformed persisted conversation entries instead of crashing on .strip().
        with self._state_lock:
            try:
                self._prepare_for_turn_locked(normalized_conversation)  # AUDIT-FIX(#4): refresh session configuration every turn so dynamic time context stays current.
                turn = self._run_on_session_loop_locked(
                    self._run_audio_turn(
                        audio_pcm,
                        conversation=normalized_conversation,
                        on_audio_chunk=on_audio_chunk,
                        on_output_text_delta=on_output_text_delta,
                    ),
                    stage="running realtime audio turn",
                    timeout=self._turn_timeout_seconds(),
                )
            except Exception:
                self._close_locked()  # AUDIT-FIX(#6): any turn failure forces a clean reconnect path for the next turn.
                raise
            self._mark_turn_complete_locked(normalized_conversation)
            return turn

    def run_text_turn(
        self,
        prompt: str,
        *,
        conversation: tuple[tuple[str, str], ...] | None = None,
        on_audio_chunk: Callable[[bytes], None] | None = None,
        on_output_text_delta: Callable[[str], None] | None = None,
    ) -> OpenAIRealtimeTurn:
        """Run one text turn against the live realtime session.

        Args:
            prompt: User text to append as the current turn input.
            conversation: Optional persisted conversation history to seed into
                a fresh provider session.
            on_audio_chunk: Optional callback for streamed assistant audio
                chunks.
            on_output_text_delta: Optional callback for streamed assistant text
                fragments.

        Returns:
            The completed realtime turn with transcript, response text, and
            provider metadata.

        Raises:
            RuntimeError: If the prompt is empty or the provider/session work
                fails.
        """

        normalized_prompt = prompt.strip()
        if not normalized_prompt:
            raise RuntimeError("Realtime turn requires a non-empty prompt")
        normalized_conversation = self._normalize_conversation(conversation)  # AUDIT-FIX(#12): tolerate malformed persisted conversation entries instead of crashing on .strip().
        with self._state_lock:
            try:
                self._prepare_for_turn_locked(normalized_conversation)  # AUDIT-FIX(#4): refresh session configuration every turn so dynamic time context stays current.
                turn = self._run_on_session_loop_locked(
                    self._run_text_turn(
                        normalized_prompt,
                        conversation=normalized_conversation,
                        on_audio_chunk=on_audio_chunk,
                        on_output_text_delta=on_output_text_delta,
                    ),
                    stage="running realtime text turn",
                    timeout=self._turn_timeout_seconds(),
                )
            except Exception:
                self._close_locked()  # AUDIT-FIX(#6): any turn failure forces a clean reconnect path for the next turn.
                raise
            self._mark_turn_complete_locked(normalized_conversation)
            return turn

    async def _configure_session(self) -> None:
        """Push the current Twinr session configuration to the provider."""
        session: dict[str, Any] = {
            "type": "realtime",
            "output_modalities": ["audio", "text"],  # AUDIT-FIX(#8): request text explicitly because this class requires response_text on every successful turn.
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
        await self._await_provider(
            self._connection.session.update(session=session),
            stage="configuring realtime session",
            timeout=self._connect_timeout_seconds(),
        )

    async def _run_audio_turn(
        self,
        audio_pcm: bytes,
        *,
        conversation: tuple[tuple[str, str], ...] | None,
        on_audio_chunk: Callable[[bytes], None] | None,
        on_output_text_delta: Callable[[str], None] | None,
    ) -> OpenAIRealtimeTurn:
        """Stream one audio turn through the provider connection."""
        await self._seed_conversation(conversation)
        for chunk in self._iter_audio_chunks(audio_pcm):
            await self._await_provider(
                self._connection.input_audio_buffer.append(audio=chunk),
                stage="streaming audio to realtime input buffer",
            )  # AUDIT-FIX(#2): every network write is bounded so intermittent Wi-Fi cannot hang the device forever.
        await self._await_provider(
            self._connection.input_audio_buffer.commit(),
            stage="committing realtime audio input buffer",
        )
        await self._await_provider(
            self._connection.response.create(response={}),
            stage="creating realtime response",
        )
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
        """Submit one text turn through the provider connection."""
        await self._seed_conversation(conversation)
        await self._await_provider(
            self._connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            ),
            stage="creating realtime user message",
        )
        await self._await_provider(
            self._connection.response.create(response={}),
            stage="creating realtime response",
        )
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
        """Consume provider events until a complete realtime turn finishes.

        Args:
            transcript_hint: Fallback transcript to use when the provider does
                not emit a completed user transcription.
            on_audio_chunk: Optional callback for streamed assistant audio.
            on_output_text_delta: Optional callback for streamed assistant
                text deltas.

        Returns:
            The finalized turn assembled from streamed provider events.

        Raises:
            RuntimeError: If the provider reports an error, ends without text,
                or a callback fails.
        """

        input_transcript_fragments: list[str] = []
        response_text_fragments: list[str] = []
        response_id: str | None = None
        model: str | None = None
        token_usage: TokenUsage | None = None
        end_conversation = False

        while True:
            event = await self._await_provider(
                self._connection.recv(),
                stage="waiting for realtime events",
            )
            event_type = str(getattr(event, "type", "")).strip()

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
                    self._emit_output_text_delta(delta, on_output_text_delta)  # AUDIT-FIX(#7): callback failures are wrapped with context and trigger a clean session reset.
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
                    self._emit_output_text_delta(final_text, on_output_text_delta)  # AUDIT-FIX(#7): callback failures are wrapped with context and trigger a clean session reset.
                continue

            if event_type in {"response.audio.delta", "response.output_audio.delta"}:
                audio_delta = str(getattr(event, "delta", ""))
                if audio_delta:
                    self._emit_audio_chunk(audio_delta, on_audio_chunk)  # AUDIT-FIX(#7): invalid base64/callback errors are surfaced cleanly and the session is recycled.
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
                            self._emit_output_text_delta(immediate_text, on_output_text_delta)
                        break
                    await self._await_provider(
                        self._connection.response.create(response={}),
                        stage="creating follow-up realtime response after tool execution",
                    )
                    continue
                if not "".join(response_text_fragments).strip():
                    extracted = self._extract_response_text(response)
                    if extracted:
                        response_text_fragments.append(extracted)
                if not "".join(response_text_fragments).strip():
                    incomplete_reason = self._response_incomplete_reason(response)
                    if incomplete_reason:
                        raise RuntimeError(incomplete_reason)
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
        """Replay persisted conversation history into a fresh provider session."""
        if not conversation or self._conversation_seeded:
            return
        for role, content in conversation:
            await self._await_provider(
                self._connection.conversation.item.create(
                    item={
                        "type": "message",
                        "role": role,
                        "content": [
                            {
                                "type": self._content_type_for_role(role),
                                "text": content,
                            }
                        ],
                    }
                ),
                stage="seeding realtime conversation history",
            )
        self._conversation_seeded = True  # AUDIT-FIX(#5): once a provider session has seeded external history, do not replay it again into the same live conversation.

    def _iter_audio_chunks(self, audio_pcm: bytes, *, chunk_size: int = 32_768):
        """Yield base64-encoded audio chunks for provider upload.

        Args:
            audio_pcm: Raw PCM audio bytes for the current turn.
            chunk_size: Maximum size of each raw PCM chunk before encoding.

        Yields:
            Base64-encoded PCM fragments ready for the realtime input buffer.

        Raises:
            RuntimeError: If ``chunk_size`` is not positive.
        """

        if chunk_size <= 0:
            raise RuntimeError("Audio chunk size must be a positive integer")
        for index in range(0, len(audio_pcm), chunk_size):
            chunk = audio_pcm[index : index + chunk_size]
            if not chunk:
                continue
            yield base64.b64encode(chunk).decode("ascii")  # AUDIT-FIX(#11): stream chunks lazily to avoid a full in-memory base64 tuple on RPi 4.

    def _session_instructions(self) -> str:
        """Assemble the full instruction block for the current session."""
        tool_instructions = build_tool_agent_instructions(
            self.config,
            extra_instructions=self.config.openai_realtime_instructions,
        )
        return merge_instructions(
            self._resolve_base_instructions(),
            memory_and_response_contract(self.config.openai_realtime_language),
            tool_instructions,
        ) or tool_instructions

    def _resolve_base_instructions(self) -> str | None:
        """Resolve the base instructions before Twinr realtime additions."""
        if self._base_instructions_override is not None:
            return self._base_instructions_override
        return load_personality_instructions(self.config)

    def _session_tools(self) -> list[dict[str, Any]]:
        """Build the realtime tool schema list for the active handlers."""
        return build_realtime_tool_schemas(self._tool_handlers.keys())

    def _format_error(self, event: Any) -> str:
        """Extract a readable error message from a realtime error event."""
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
        """Collect transcript and text fragments from a provider response."""
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
        """Extract function-call tuples from a provider response payload."""
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
        """Execute realtime tool calls and return their follow-up policy."""
        handled_names: list[str] = []
        immediate_response_text: str | None = None
        continue_response = True
        for name, call_id, arguments_json in function_calls:
            handler = self._tool_handlers.get(name)
            tool_succeeded = False
            try:
                arguments = self._parse_tool_arguments(arguments_json)
            except RuntimeError as exc:
                arguments = {}
                output = {
                    "status": "error",
                    "message": "Tool arguments were invalid.",
                    "error_type": type(exc).__name__,
                }  # AUDIT-FIX(#9): malformed model-generated tool JSON should not abort the whole turn.
            else:
                if handler is None:
                    output = {"status": "error", "message": f"Unsupported tool: {name}"}
                else:
                    try:
                        result = await self._execute_tool_handler(handler, arguments)  # AUDIT-FIX(#3): support async handlers, awaitables, and bounded execution.
                        output = result if result is not None else {"status": "ok"}
                        tool_succeeded = True
                        handled_names.append(name)
                    except Exception as exc:
                        output = {
                            "status": "error",
                            "message": "Tool execution failed.",
                            "error_type": type(exc).__name__,
                        }  # AUDIT-FIX(#10): do not leak raw exception text back into the model/user channel.
            if name == "end_conversation" and len(function_calls) == 1 and tool_succeeded:
                spoken_reply = str(arguments.get("spoken_reply", "") or "").strip()
                if spoken_reply:
                    immediate_response_text = spoken_reply
                    continue_response = False
            await self._await_provider(
                self._connection.conversation.item.create(
                    item={
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": self._serialize_tool_output(output),
                    }
                ),
                stage=f"returning tool output for {name}",
            )
        return _HandledRealtimeTools(
            names=tuple(handled_names),
            continue_response=continue_response,
            immediate_response_text=immediate_response_text,
        )

    def _parse_tool_arguments(self, arguments_json: str) -> dict[str, Any]:
        """Decode model-generated tool arguments into a JSON object.

        Raises:
            RuntimeError: If the payload is not valid JSON or does not decode
                to an object.
        """

        try:
            parsed = json.loads(arguments_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Tool arguments are not valid JSON: {exc.msg}") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("Tool arguments must decode to a JSON object")
        return parsed

    def _serialize_tool_output(self, output: Any) -> str:
        """Serialize a tool result into the provider's output payload format."""
        if isinstance(output, str):
            return output
        try:
            return json.dumps(output, ensure_ascii=False, default=self._json_default)
        except (TypeError, ValueError):
            fallback = {
                "status": "error",
                "message": "Tool output could not be serialized safely.",
            }  # AUDIT-FIX(#3): non-JSON-safe tool results must degrade to a valid tool payload instead of crashing the turn.
            return json.dumps(fallback, ensure_ascii=False)

    def _content_type_for_role(self, role: str) -> str:
        """Map a conversation role to the provider text content type."""
        if role == "assistant":
            return "output_text"
        return "input_text"

    async def _await_provider(self, awaitable: Any, *, stage: str, timeout: float | None = None) -> Any:
        """Await provider work with a bounded timeout and contextual errors."""
        wait_timeout = self._event_timeout_seconds() if timeout is None else timeout
        try:
            return await asyncio.wait_for(awaitable, timeout=wait_timeout)
        except asyncio.TimeoutError as exc:
            raise RuntimeError(f"Timed out while {stage}") from exc  # AUDIT-FIX(#2): convert unbounded provider waits into bounded failures with actionable context.

    async def _execute_tool_handler(self, handler: Callable[[dict[str, Any]], Any], arguments: dict[str, Any]) -> Any:
        """Run one tool handler with timeout enforcement and sync offloading."""
        tool_timeout = self._tool_timeout_seconds()
        if self._is_async_handler(handler):
            result = await asyncio.wait_for(handler(arguments), timeout=tool_timeout)
        elif self._offload_sync_tool_handlers():
            result = await asyncio.wait_for(
                asyncio.to_thread(handler, arguments),
                timeout=tool_timeout,
            )
        else:
            result = handler(arguments)
        if inspect.isawaitable(result):
            return await asyncio.wait_for(result, timeout=tool_timeout)
        return result

    def _is_async_handler(self, handler: Callable[[dict[str, Any]], Any]) -> bool:
        """Return whether a tool handler is implemented as coroutine code."""
        return inspect.iscoroutinefunction(handler) or inspect.iscoroutinefunction(getattr(handler, "__call__", None))

    def _offload_sync_tool_handlers(self) -> bool:
        """Decide whether synchronous tool handlers should run in a worker thread."""
        raw_value = getattr(self.config, "openai_realtime_offload_sync_tool_handlers", True)  # AUDIT-FIX(#3): keep default behavior resilient, but allow opt-out if a legacy handler is thread-affine.
        if isinstance(raw_value, str):
            return raw_value.strip().lower() not in {"0", "false", "no", "off"}
        return bool(raw_value)

    def _json_default(self, value: Any) -> Any:
        """Provide a bounded JSON fallback for unsupported tool output values."""
        if isinstance(value, bytes):
            return {"type": "bytes", "length": len(value)}
        return {"type": type(value).__name__}

    def _emit_output_text_delta(
        self,
        delta: str,
        callback: Callable[[str], None] | None,
    ) -> None:
        """Forward one streamed text fragment to the caller callback."""
        if callback is None:
            return
        try:
            callback(delta)
        except Exception as exc:
            raise RuntimeError("Realtime text callback failed") from exc

    def _emit_audio_chunk(
        self,
        audio_delta_base64: str,
        callback: Callable[[bytes], None] | None,
    ) -> None:
        """Decode and forward one streamed audio chunk to the caller callback."""
        if callback is None:
            return
        try:
            decoded = base64.b64decode(audio_delta_base64, validate=True)
        except Exception as exc:
            raise RuntimeError("Realtime audio delta was not valid base64") from exc
        try:
            callback(decoded)
        except Exception as exc:
            raise RuntimeError("Realtime audio callback failed") from exc

    def _response_incomplete_reason(self, response: Any) -> str | None:
        """Return a provider-side incomplete status message when present."""
        status = str(getattr(response, "status", "") or "").strip().lower()
        if not status or status == "completed":
            return None
        details = getattr(response, "status_details", None)
        reason = str(getattr(details, "reason", "") or "").strip()
        if reason:
            return f"Realtime response ended with status {status}: {reason}"  # AUDIT-FIX(#8): surface provider-side incomplete/failure states instead of falling through to a misleading generic no-text error.
        return f"Realtime response ended with status {status}"  # AUDIT-FIX(#8): surface provider-side incomplete/failure states instead of falling through to a misleading generic no-text error.

    def _normalize_conversation(
        self,
        conversation: tuple[tuple[str, str], ...] | None,
    ) -> tuple[tuple[str, str], ...] | None:
        """Normalize persisted conversation history into provider-safe tuples."""
        if not conversation:
            return None
        normalized: list[tuple[str, str]] = []  # AUDIT-FIX(#12): normalize persisted history defensively because file-backed state can contain malformed entries.
        for item in conversation:
            try:
                role_raw, content_raw = item
            except (TypeError, ValueError):
                continue
            role = str(role_raw).strip().lower()
            content = str(content_raw).strip()
            if role not in {"system", "user", "assistant"} or not content:
                continue
            normalized.append((role, content))
        return tuple(normalized) or None

    def _prepare_for_turn_locked(self, conversation: tuple[tuple[str, str], ...] | None) -> None:
        """Ensure the provider session is open and ready for the next turn."""
        self._open_locked()
        if conversation is not None and self._turns_completed > 0:
            self._close_locked()
            self._open_locked()  # AUDIT-FIX(#5): a live provider session already contains prior turns, so reconnect before replaying external history to avoid duplicate context.
        self._run_on_session_loop_locked(
            self._configure_session(),
            stage="refreshing realtime session configuration",
            timeout=self._connect_timeout_seconds(),
        )

    def _mark_turn_complete_locked(self, conversation: tuple[tuple[str, str], ...] | None) -> None:
        """Record local state after a turn completes successfully."""
        self._turns_completed += 1
        if conversation is not None:
            self._conversation_seeded = True

    def _open_locked(self) -> None:
        """Open the provider connection and configure it when not already live."""
        if self._connection is not None and self._loop is not None and self._loop.is_running():
            return
        self._start_loop_thread_locked()
        try:
            manager, connection = self._run_on_session_loop_locked(
                self._open_connection(),
                stage="opening realtime session",
                timeout=self._connect_timeout_seconds(),
            )
            self._manager = manager
            self._connection = connection
            self._conversation_seeded = False
            self._turns_completed = 0
            self._run_on_session_loop_locked(
                self._configure_session(),
                stage="configuring realtime session",
                timeout=self._connect_timeout_seconds(),
            )
        except Exception:
            self._close_locked()
            raise

    def _close_locked(self) -> None:
        """Tear down the provider connection and background event loop."""
        manager = self._manager
        loop = self._loop
        loop_thread = self._loop_thread

        self._manager = None
        self._connection = None
        self._conversation_seeded = False
        self._turns_completed = 0

        if manager is not None and loop is not None and loop.is_running():
            try:
                self._run_on_session_loop_locked(
                    self._close_connection(manager),
                    stage="closing realtime session",
                    timeout=self._close_timeout_seconds(),
                )
            except Exception:
                logger.warning(
                    "OpenAI realtime session cleanup failed while closing the provider manager.",
                    exc_info=True,
                )  # AUDIT-FIX(#6): cleanup must stay best-effort and must not mask the original failure.

        if loop is not None and loop.is_running():
            try:
                loop.call_soon_threadsafe(loop.stop)
            except RuntimeError:
                pass

        if loop_thread is not None:
            loop_thread.join(timeout=self._loop_thread_join_timeout_seconds())

        self._loop = None
        self._loop_thread = None
        self._loop_started.clear()

    def _start_loop_thread_locked(self) -> None:
        """Start the dedicated background asyncio loop for provider work."""
        if self._loop_thread is not None and self._loop_thread.is_alive() and self._loop is not None and self._loop.is_running():
            return

        self._loop_started.clear()
        loop_holder: dict[str, asyncio.AbstractEventLoop] = {}

        def _thread_main() -> None:
            loop = asyncio.new_event_loop()
            loop_holder["loop"] = loop
            asyncio.set_event_loop(loop)
            self._loop = loop
            self._loop_started.set()
            try:
                loop.run_forever()
            finally:
                pending = asyncio.all_tasks(loop=loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()

        self._loop_thread = threading.Thread(
            target=_thread_main,
            name="OpenAIRealtimeSessionLoop",
            daemon=True,
        )
        self._loop_thread.start()
        if not self._loop_started.wait(timeout=self._loop_thread_start_timeout_seconds()):
            self._loop = None
            self._loop_thread = None
            raise RuntimeError("Timed out while starting the realtime event loop")  # AUDIT-FIX(#1): fail fast instead of deadlocking on a half-started loop.
        self._loop = loop_holder.get("loop")

    def _run_on_session_loop_locked(
        self,
        coro: Any,
        *,
        stage: str,
        timeout: float | None,
    ) -> Any:
        """Run a coroutine on the session loop and wait synchronously for it."""
        loop = self._loop
        if loop is None or not loop.is_running():
            if inspect.iscoroutine(coro):
                coro.close()
            raise RuntimeError("Realtime session event loop is not running")
        try:
            future = asyncio.run_coroutine_threadsafe(coro, loop)
        except RuntimeError as exc:
            if inspect.iscoroutine(coro):
                coro.close()
            raise RuntimeError("Realtime session event loop is not available") from exc
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError as exc:
            future.cancel()
            raise RuntimeError(f"Timed out while {stage}") from exc  # AUDIT-FIX(#2): protect sync callers from indefinite waits on the background loop.
        except Exception:
            if not future.done():
                future.cancel()
            raise

    async def _open_connection(self) -> tuple[Any, Any]:
        """Enter the provider's realtime connection manager."""
        manager = self._client.realtime.connect(model=self.config.openai_realtime_model)
        connection = await manager.__aenter__()
        return manager, connection

    async def _close_connection(self, manager: Any) -> None:
        """Exit the provider's realtime connection manager."""
        await manager.__aexit__(None, None, None)

    def _connect_timeout_seconds(self) -> float:
        """Return the bounded timeout for opening or configuring a session."""
        return _coerce_optional_float(
            getattr(self.config, "openai_realtime_connect_timeout_seconds", None),
            default=20.0,
            minimum=0.1,
        ) or 20.0

    def _event_timeout_seconds(self) -> float:
        """Return the bounded timeout for individual provider events."""
        return _coerce_optional_float(
            getattr(self.config, "openai_realtime_event_timeout_seconds", None),
            default=45.0,
            minimum=0.1,
        ) or 45.0

    def _turn_timeout_seconds(self) -> float:
        """Return the bounded timeout for one full realtime turn."""
        return _coerce_optional_float(
            getattr(self.config, "openai_realtime_turn_timeout_seconds", None),
            default=180.0,
            minimum=0.1,
        ) or 180.0

    def _tool_timeout_seconds(self) -> float:
        """Return the bounded timeout for one tool handler execution."""
        return _coerce_optional_float(
            getattr(self.config, "openai_realtime_tool_timeout_seconds", None),
            default=30.0,
            minimum=0.1,
        ) or 30.0

    def _close_timeout_seconds(self) -> float:
        """Return the bounded timeout for closing the provider session."""
        return _coerce_optional_float(
            getattr(self.config, "openai_realtime_close_timeout_seconds", None),
            default=5.0,
            minimum=0.1,
        ) or 5.0

    def _loop_thread_start_timeout_seconds(self) -> float:
        """Return the timeout for bringing up the background loop thread."""
        return _coerce_optional_float(
            getattr(self.config, "openai_realtime_loop_thread_start_timeout_seconds", None),
            default=5.0,
            minimum=0.1,
        ) or 5.0

    def _loop_thread_join_timeout_seconds(self) -> float:
        """Return the timeout for joining the background loop thread."""
        return _coerce_optional_float(
            getattr(self.config, "openai_realtime_loop_thread_join_timeout_seconds", None),
            default=5.0,
            minimum=0.1,
        ) or 5.0
