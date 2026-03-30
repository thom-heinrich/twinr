# CHANGELOG: 2026-03-30
# BUG-1: Removed duplicate session.update on fresh opens/reconnects; turns now configure the provider exactly once per turn.
# BUG-2: Fixed text turns wasting bandwidth/CPU by forcing audio output; responses are now text-only unless the caller actually requests streamed audio.
# BUG-3: Fixed provider-audio incompatibility by enforcing OpenAI Realtime PCM16 @ 24 kHz and auto-resampling inbound mono PCM16 when Twinr is configured for another input rate.
# BUG-4: Fixed event-loop stalls from inline caller callbacks; streamed text/audio callbacks now run on bounded worker threads so slow playback/UI code does not block provider I/O.
# BUG-5: Fixed unbounded tool-follow-up loops that could burn latency/cost until turn timeout by capping tool round-trips per turn.
# SEC-1: Added bounded tool argument/output sizes to prevent prompt/tool-driven memory blowups and oversized provider payloads on Raspberry Pi deployments.
# SEC-2: Reconnect now occurs when immutable realtime session properties change (for example model/voice/tracing), preventing invalid live-session mutations and state drift.
# IMP-1: Added 2026 Realtime hooks for hosted prompts, tracing, truncation/retention-ratio, VAD selection, max output tokens, include-fields, and transport overrides without breaking the public API.
# IMP-2: Added transport-aware client construction (organization/base_url/websocket_base_url/default_headers), safer callback backpressure handling, and best-effort response cancellation on mid-stream failures.

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
import copy
import inspect
import json
import logging
import sys
import threading
from array import array
from collections.abc import Mapping
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from queue import Full, Queue
from typing import Any, Callable, cast

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.conversation.language import memory_and_response_contract
from twinr.agent.base_agent.prompting.personality import load_personality_instructions, merge_instructions
from twinr.agent.tools import build_realtime_tool_schemas
from twinr.agent.tools.prompting import build_tool_agent_instructions
from twinr.ops.usage import TokenUsage, extract_model_name, extract_token_usage
from twinr.providers.openai.core.client import _should_send_project_header

logger = logging.getLogger(__name__)

_PROVIDER_PCM_SAMPLE_RATE = 24_000
_DEFAULT_AUDIO_CHUNK_SIZE = 32_768
_DEFAULT_TOOL_ARGUMENT_MAX_CHARS = 32_768
_DEFAULT_TOOL_OUTPUT_MAX_CHARS = 65_536
_DEFAULT_MAX_TOOL_ROUNDTRIPS_PER_TURN = 8
_DEFAULT_CALLBACK_QUEUE_MAXSIZE = 256
_DEFAULT_CALLBACK_JOIN_TIMEOUT_SECONDS = 2.0


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


class _StreamingCallbackDispatcher:
    """Run streamed callbacks off the provider event loop with backpressure."""

    def __init__(
        self,
        name: str,
        callback: Callable[[Any], None],
        *,
        max_queue_size: int,
        join_timeout_seconds: float,
    ) -> None:
        self._name = name
        self._callback = callback
        self._queue: Queue[Any] = Queue(maxsize=max(1, max_queue_size))
        self._join_timeout_seconds = max(0.1, float(join_timeout_seconds))
        self._sentinel = object()
        self._lock = threading.Lock()
        self._error: Exception | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name=f"OpenAIRealtime{name.title()}Callback",
            daemon=True,
        )
        self._thread.start()

    def submit(self, payload: Any) -> None:
        self.raise_if_failed()
        try:
            self._queue.put_nowait(payload)
        except Full as exc:
            raise RuntimeError(f"Realtime {self._name} callback fell behind") from exc
        self.raise_if_failed()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                try:
                    self._queue.put(self._sentinel, timeout=self._join_timeout_seconds)
                except Exception:
                    pass
        self._thread.join(timeout=self._join_timeout_seconds)
        self.raise_if_failed()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"Realtime {self._name} callback failed") from self._error

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is self._sentinel:
                return
            try:
                self._callback(item)
            except Exception as exc:  # pragma: no cover - depends on user callback behavior
                with self._lock:
                    if self._error is None:
                        self._error = exc
                return


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

    organization = str(getattr(config, "openai_organization_id", "") or "").strip()
    if organization:
        kwargs["organization"] = organization

    base_url = getattr(config, "openai_base_url", None)
    if base_url:
        kwargs["base_url"] = base_url

    websocket_base_url = getattr(config, "openai_realtime_websocket_base_url", None)
    if websocket_base_url:
        kwargs["websocket_base_url"] = websocket_base_url

    default_headers = getattr(config, "openai_default_headers", None)
    if isinstance(default_headers, Mapping) and default_headers:
        kwargs["default_headers"] = dict(default_headers)

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


def _deep_merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep-copied merge where nested mappings override recursively."""

    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(merged.get(key), Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


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
        session_defaults: Mapping[str, Any] | None = None,
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
            session_defaults: Optional provider session fields that should be
                merged into Twinr's baseline realtime session update before the
                connection is configured.
            tool_handlers: Mapping from realtime tool names to handler
                callables. Handlers may be sync, async, or return awaitables.
        """

        self.config = config
        self._client_factory = client_factory or _default_async_client_factory
        self._client_is_custom = client is not None
        self._client = client or self._client_factory(config)
        self._client_transport_fingerprint = self._compute_client_transport_fingerprint()
        self._base_instructions_override = base_instructions
        self._session_defaults = (
            copy.deepcopy(dict(session_defaults))
            if isinstance(session_defaults, Mapping)
            else None
        )
        self._tool_handlers = dict(tool_handlers or {})
        self._state_lock = threading.RLock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._loop_started = threading.Event()
        self._manager = None
        self._connection = None
        self._conversation_seeded = False
        self._turns_completed = 0
        self._session_configured = False
        self._session_fingerprint: str | None = None
        self._pending_output_modalities: tuple[str, ...] = ("text",)

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
            audio_pcm: Raw mono little-endian PCM16 bytes at the configured
                Twinr capture rate. The wrapper will resample to the provider's
                required 24 kHz PCM16 stream when necessary.
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
        normalized_conversation = self._normalize_conversation(conversation)
        provider_audio_pcm = self._prepare_input_audio(audio_pcm)
        with self._state_lock:
            self._pending_output_modalities = self._desired_output_modalities(on_audio_chunk)
            try:
                self._prepare_for_turn_locked(normalized_conversation)
                turn = self._run_on_session_loop_locked(
                    self._run_audio_turn(
                        provider_audio_pcm,
                        conversation=normalized_conversation,
                        on_audio_chunk=on_audio_chunk,
                        on_output_text_delta=on_output_text_delta,
                    ),
                    stage="running realtime audio turn",
                    timeout=self._turn_timeout_seconds(),
                )
            except Exception:
                self._close_locked()
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
        normalized_conversation = self._normalize_conversation(conversation)
        with self._state_lock:
            self._pending_output_modalities = self._desired_output_modalities(on_audio_chunk)
            try:
                self._prepare_for_turn_locked(normalized_conversation)
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
                self._close_locked()
                raise
            self._mark_turn_complete_locked(normalized_conversation)
            return turn

    async def _configure_session(self) -> None:
        """Push the current Twinr session configuration to the provider."""
        session: dict[str, Any] = {
            "type": "realtime",
            "output_modalities": list(self._pending_output_modalities),
            "instructions": self._session_instructions(),
            "audio": {
                "input": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": _PROVIDER_PCM_SAMPLE_RATE,  # BREAKING: provider-facing PCM16 is fixed to 24 kHz; inbound audio is auto-resampled when Twinr captures at another rate.
                    },
                    "noise_reduction": self._audio_input_noise_reduction_config(),
                    "transcription": self._audio_input_transcription_config(),
                    "turn_detection": self._turn_detection_config(),
                },
                "output": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": _PROVIDER_PCM_SAMPLE_RATE,  # BREAKING: streamed assistant PCM16 is always 24 kHz because that is the provider's fixed output rate for PCM.
                    },
                    "voice": self._resolved_voice(),
                    "speed": self._speech_speed(),
                },
            },
        }

        include = self._include_fields_config()
        if include:
            session["include"] = include

        prompt = self._prompt_config()
        if prompt:
            session["prompt"] = prompt

        tracing = self._tracing_config()
        if tracing is not None:
            session["tracing"] = tracing

        truncation = self._truncation_config()
        if truncation is not None:
            session["truncation"] = truncation

        max_response_output_tokens = self._max_response_output_tokens()
        if max_response_output_tokens is not None:
            session["max_response_output_tokens"] = max_response_output_tokens

        temperature = self._temperature()
        if temperature is not None:
            session["temperature"] = temperature

        if self._session_defaults:
            session = _deep_merge_dicts(session, self._session_defaults)

        tools = self._session_tools()
        if tools:
            session["tools"] = tools
            session["tool_choice"] = self._tool_choice()
        await self._await_provider(
            self._connection.session.update(session=session),
            stage="configuring realtime session",
            timeout=self._connect_timeout_seconds(),
        )
        self._session_configured = True
        self._session_fingerprint = self._immutable_session_fingerprint()

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
            )
        await self._await_provider(
            self._connection.input_audio_buffer.commit(),
            stage="committing realtime audio input buffer",
        )
        await self._await_provider(
            self._connection.response.create(response=self._response_create_payload()),
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
            self._connection.response.create(response=self._response_create_payload()),
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
        tool_roundtrips = 0

        audio_dispatcher = self._make_callback_dispatcher("audio", on_audio_chunk)
        text_dispatcher = self._make_callback_dispatcher("text", on_output_text_delta)

        try:
            while True:
                self._raise_callback_dispatcher_failure(audio_dispatcher)
                self._raise_callback_dispatcher_failure(text_dispatcher)

                event = await self._await_provider(
                    self._connection.recv(),
                    stage="waiting for realtime events",
                )
                event_type = str(getattr(event, "type", "")).strip()

                if event_type == "error":
                    raise RuntimeError(self._format_error(event))

                if event_type == "response.created":
                    response = getattr(event, "response", None)
                    response_id = getattr(response, "id", None) or response_id
                    continue

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
                        self._emit_output_text_delta(delta, text_dispatcher)
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
                        self._emit_output_text_delta(final_text, text_dispatcher)
                    continue

                if event_type == "response.content_part.done":
                    part = getattr(event, "part", None)
                    final_text = str(
                        getattr(part, "transcript", "") or getattr(part, "text", "")
                    ).strip()
                    if final_text and not "".join(response_text_fragments).strip():
                        response_text_fragments.append(final_text)
                        self._emit_output_text_delta(final_text, text_dispatcher)
                    continue

                if event_type in {"response.audio.delta", "response.output_audio.delta"}:
                    audio_delta = str(getattr(event, "delta", ""))
                    if audio_delta:
                        self._emit_audio_chunk(audio_delta, audio_dispatcher)
                    continue

                if event_type == "response.done":
                    response = getattr(event, "response", None)
                    response_id = getattr(response, "id", None) or response_id
                    model = extract_model_name(response)
                    token_usage = extract_token_usage(response)
                    function_calls = self._extract_function_calls(response)
                    if function_calls:
                        tool_roundtrips += 1
                        if tool_roundtrips > self._max_tool_roundtrips_per_turn():
                            raise RuntimeError("Realtime tool loop exceeded the per-turn safety limit")
                        handled_tools = await self._handle_function_calls(function_calls)
                        if "end_conversation" in handled_tools.names:
                            end_conversation = True
                        if not handled_tools.continue_response:
                            immediate_text = str(handled_tools.immediate_response_text or "").strip()
                            if immediate_text:
                                response_text_fragments.append(immediate_text)
                                self._emit_output_text_delta(immediate_text, text_dispatcher)
                            break
                        await self._await_provider(
                            self._connection.response.create(response=self._response_create_payload()),
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
        except Exception:
            await self._cancel_active_response()
            raise
        finally:
            dispatcher_close_error: Exception | None = None
            for dispatcher in (text_dispatcher, audio_dispatcher):
                try:
                    self._close_callback_dispatcher(dispatcher)
                except Exception as exc:
                    if dispatcher_close_error is None:
                        dispatcher_close_error = exc
                    else:
                        logger.warning("Realtime callback dispatcher cleanup failed.", exc_info=True)
            if dispatcher_close_error is not None and sys.exc_info()[0] is None:
                raise dispatcher_close_error

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
        self._conversation_seeded = True

    def _iter_audio_chunks(self, audio_pcm: bytes, *, chunk_size: int = _DEFAULT_AUDIO_CHUNK_SIZE):
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
            yield base64.b64encode(chunk).decode("ascii")

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
                }
            else:
                if handler is None:
                    output = {"status": "error", "message": f"Unsupported tool: {name}"}
                else:
                    try:
                        result = await self._execute_tool_handler(handler, arguments)
                        output = result if result is not None else {"status": "ok"}
                        tool_succeeded = True
                        handled_names.append(name)
                    except Exception as exc:
                        logger.warning("Realtime tool %s failed.", name, exc_info=True)
                        output = {
                            "status": "error",
                            "message": "Tool execution failed.",
                            "error_type": type(exc).__name__,
                        }
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

        max_chars = self._tool_argument_max_chars()
        if len(arguments_json) > max_chars:
            raise RuntimeError("Tool arguments exceeded the maximum supported size")
        try:
            parsed = json.loads(arguments_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Tool arguments are not valid JSON: {exc.msg}") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("Tool arguments must decode to a JSON object")
        return parsed

    def _serialize_tool_output(self, output: Any) -> str:
        """Serialize a tool result into the provider's output payload format."""
        max_chars = self._tool_output_max_chars()
        if isinstance(output, str):
            return self._truncate_text(output, max_chars=max_chars)
        try:
            serialized = json.dumps(output, ensure_ascii=False, default=self._json_default)
        except (TypeError, ValueError):
            fallback = {
                "status": "error",
                "message": "Tool output could not be serialized safely.",
            }
            return json.dumps(fallback, ensure_ascii=False)
        if len(serialized) <= max_chars:
            return serialized
        truncated = {
            "status": "error",
            "message": "Tool output exceeded the maximum supported size.",
            "preview": self._truncate_text(serialized, max_chars=max(128, max_chars // 4)),
        }
        return json.dumps(truncated, ensure_ascii=False)

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
            raise RuntimeError(f"Timed out while {stage}") from exc

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
        raw_value = getattr(self.config, "openai_realtime_offload_sync_tool_handlers", True)
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
        dispatcher: _StreamingCallbackDispatcher | None,
    ) -> None:
        """Forward one streamed text fragment to the caller callback."""
        if dispatcher is None:
            return
        dispatcher.submit(delta)

    def _emit_audio_chunk(
        self,
        audio_delta_base64: str,
        dispatcher: _StreamingCallbackDispatcher | None,
    ) -> None:
        """Decode and forward one streamed audio chunk to the caller callback."""
        if dispatcher is None:
            return
        try:
            decoded = base64.b64decode(audio_delta_base64, validate=True)
        except Exception as exc:
            raise RuntimeError("Realtime audio delta was not valid base64") from exc
        dispatcher.submit(decoded)

    def _response_incomplete_reason(self, response: Any) -> str | None:
        """Return a provider-side incomplete status message when present."""
        status = str(getattr(response, "status", "") or "").strip().lower()
        if not status or status == "completed":
            return None
        details = getattr(response, "status_details", None)
        reason = str(getattr(details, "reason", "") or "").strip()
        if reason:
            return f"Realtime response ended with status {status}: {reason}"
        return f"Realtime response ended with status {status}"

    def _normalize_conversation(
        self,
        conversation: tuple[tuple[str, str], ...] | None,
    ) -> tuple[tuple[str, str], ...] | None:
        """Normalize persisted conversation history into provider-safe tuples."""
        if not conversation:
            return None
        normalized: list[tuple[str, str]] = []
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
        self._refresh_client_if_needed_locked()
        self._open_locked()
        if conversation is not None and self._turns_completed > 0:
            self._close_locked()
            self._open_locked()
        desired_fingerprint = self._immutable_session_fingerprint()
        if self._session_fingerprint is not None and self._session_fingerprint != desired_fingerprint:
            self._close_locked()
            self._open_locked()
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
        """Open the provider connection when not already live."""
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
            self._session_configured = False
            self._session_fingerprint = None
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
        self._session_configured = False
        self._session_fingerprint = None

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
                )

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
        if (
            self._loop_thread is not None
            and self._loop_thread.is_alive()
            and self._loop is not None
            and self._loop.is_running()
        ):
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
            raise RuntimeError("Timed out while starting the realtime event loop")
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
            raise RuntimeError(f"Timed out while {stage}") from exc
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

    async def _cancel_active_response(self) -> None:
        """Best-effort cancel of any in-flight response after local failures."""
        if self._connection is None:
            return
        response_api = getattr(self._connection, "response", None)
        cancel_method = cast(Callable[[], Any] | None, getattr(response_api, "cancel", None))
        if cancel_method is None:
            return
        if not callable(cancel_method):
            return
        try:
            maybe_awaitable = cancel_method()  # pylint: disable=not-callable
            if inspect.isawaitable(maybe_awaitable):
                await self._await_provider(
                    maybe_awaitable,
                    stage="cancelling realtime response",
                    timeout=min(2.0, self._event_timeout_seconds()),
                )
        except Exception:
            logger.debug("Best-effort realtime response cancellation failed.", exc_info=True)

    def _desired_output_modalities(
        self,
        on_audio_chunk: Callable[[bytes], None] | None,
    ) -> tuple[str, ...]:
        """Choose text-only or audio output for the upcoming turn."""
        if on_audio_chunk is None:
            return ("text",)
        return ("audio",)

    def _response_create_payload(self) -> dict[str, Any]:
        """Build the per-response payload used for realtime response.create."""
        payload: dict[str, Any] = {
            "output_modalities": list(self._pending_output_modalities),
        }
        max_output_tokens = self._max_output_tokens()
        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens
        prompt = self._response_prompt_override()
        if prompt:
            payload["prompt"] = prompt
        return payload

    def _audio_input_transcription_config(self) -> dict[str, Any] | None:
        """Build the input audio transcription config when enabled."""
        model = str(self.config.openai_realtime_transcription_model or "").strip()
        if not model:
            return None
        config: dict[str, Any] = {"model": model}
        language = str(getattr(self.config, "openai_realtime_language", "") or "").strip()
        if language:
            config["language"] = language
        prompt = str(getattr(self.config, "openai_realtime_transcription_prompt", "") or "").strip()
        if prompt:
            config["prompt"] = prompt
        return config

    def _audio_input_noise_reduction_config(self) -> dict[str, Any] | None:
        """Resolve the configured input noise reduction mode."""
        configured = getattr(self.config, "openai_realtime_noise_reduction_type", None)
        noise_type = str(configured or "far_field").strip()
        if not noise_type or noise_type.lower() in {"none", "off", "null"}:
            return None
        return {"type": noise_type}

    def _turn_detection_config(self) -> dict[str, Any] | None:
        """Resolve optional turn-detection/VAD configuration."""
        explicit = getattr(self.config, "openai_realtime_turn_detection", None)
        if isinstance(explicit, Mapping):
            return copy.deepcopy(dict(explicit))
        configured_type = str(getattr(self.config, "openai_realtime_turn_detection_type", "") or "").strip()
        if not configured_type:
            return None
        if configured_type.lower() in {"none", "off", "null"}:
            return None

        turn_detection: dict[str, Any] = {"type": configured_type}

        create_response = getattr(self.config, "openai_realtime_turn_detection_create_response", None)
        if create_response is not None:
            turn_detection["create_response"] = bool(create_response)

        interrupt_response = getattr(self.config, "openai_realtime_turn_detection_interrupt_response", None)
        if interrupt_response is not None:
            turn_detection["interrupt_response"] = bool(interrupt_response)

        if configured_type == "server_vad":
            threshold = _coerce_optional_float(
                getattr(self.config, "openai_realtime_vad_threshold", None),
                default=None,
                minimum=0.0,
            )
            if threshold is not None and threshold <= 1.0:
                turn_detection["threshold"] = threshold
            prefix_padding_ms = _coerce_optional_int(
                getattr(self.config, "openai_realtime_vad_prefix_padding_ms", None),
                default=None,
                minimum=0,
            )
            if prefix_padding_ms is not None:
                turn_detection["prefix_padding_ms"] = prefix_padding_ms
            silence_duration_ms = _coerce_optional_int(
                getattr(self.config, "openai_realtime_vad_silence_duration_ms", None),
                default=None,
                minimum=0,
            )
            if silence_duration_ms is not None:
                turn_detection["silence_duration_ms"] = silence_duration_ms
        elif configured_type == "semantic_vad":
            eagerness = str(getattr(self.config, "openai_realtime_vad_eagerness", "") or "").strip()
            if eagerness:
                turn_detection["eagerness"] = eagerness
        return turn_detection

    def _include_fields_config(self) -> list[str] | None:
        """Resolve optional additional server output fields."""
        include = getattr(self.config, "openai_realtime_include", None)
        if isinstance(include, (list, tuple, set)):
            normalized = [str(item).strip() for item in include if str(item).strip()]
            return normalized or None
        if isinstance(include, str) and include.strip():
            return [include.strip()]
        return None

    def _prompt_config(self) -> dict[str, Any] | None:
        """Resolve optional hosted prompt configuration for session.update."""
        prompt_id = str(getattr(self.config, "openai_realtime_prompt_id", "") or "").strip()
        if not prompt_id:
            return None
        prompt: dict[str, Any] = {"id": prompt_id}
        version = str(getattr(self.config, "openai_realtime_prompt_version", "") or "").strip()
        if version:
            prompt["version"] = version
        variables = getattr(self.config, "openai_realtime_prompt_variables", None)
        if isinstance(variables, Mapping):
            prompt["variables"] = copy.deepcopy(dict(variables))
        return prompt

    def _response_prompt_override(self) -> dict[str, Any] | None:
        """Resolve an optional per-response prompt override."""
        prompt_id = str(getattr(self.config, "openai_realtime_response_prompt_id", "") or "").strip()
        if not prompt_id:
            return None
        prompt: dict[str, Any] = {"id": prompt_id}
        version = str(getattr(self.config, "openai_realtime_response_prompt_version", "") or "").strip()
        if version:
            prompt["version"] = version
        variables = getattr(self.config, "openai_realtime_response_prompt_variables", None)
        if isinstance(variables, Mapping):
            prompt["variables"] = copy.deepcopy(dict(variables))
        return prompt

    def _tracing_config(self) -> str | dict[str, Any] | None:
        """Resolve optional Realtime tracing configuration."""
        explicit = getattr(self.config, "openai_realtime_tracing", None)
        if explicit in {"auto", None}:
            if explicit == "auto":
                return "auto"
        elif isinstance(explicit, Mapping):
            return copy.deepcopy(dict(explicit))
        elif isinstance(explicit, str):
            normalized = explicit.strip().lower()
            if normalized in {"off", "none", "null"}:
                return None
            if normalized == "auto":
                return "auto"

        enable_auto = getattr(self.config, "openai_realtime_enable_tracing", None)
        if enable_auto is True:
            return "auto"
        if enable_auto is False:
            return None

        group_id = str(getattr(self.config, "openai_realtime_trace_group_id", "") or "").strip()
        workflow_name = str(getattr(self.config, "openai_realtime_trace_workflow_name", "") or "").strip()
        metadata = getattr(self.config, "openai_realtime_trace_metadata", None)
        if not group_id and not workflow_name and not isinstance(metadata, Mapping):
            return None

        tracing: dict[str, Any] = {}
        if group_id:
            tracing["group_id"] = group_id
        if workflow_name:
            tracing["workflow_name"] = workflow_name
        if isinstance(metadata, Mapping) and metadata:
            tracing["metadata"] = copy.deepcopy(dict(metadata))
        return tracing or None

    def _truncation_config(self) -> dict[str, Any] | None:
        """Resolve optional context truncation configuration."""
        explicit = getattr(self.config, "openai_realtime_truncation", None)
        if isinstance(explicit, Mapping):
            return copy.deepcopy(dict(explicit))
        if isinstance(explicit, str):
            normalized = explicit.strip().lower()
            if normalized in {"disabled", "off", "none", "null"}:
                return {"type": "disabled"}
        enable_cache_friendly = getattr(
            self.config,
            "openai_realtime_enable_cache_friendly_truncation",
            True,
        )
        if not bool(enable_cache_friendly):
            return None
        retention_ratio = _coerce_optional_float(
            getattr(self.config, "openai_realtime_retention_ratio", None),
            default=0.8,
            minimum=0.01,
        )
        if retention_ratio is None:
            return None
        if retention_ratio > 1.0:
            retention_ratio = 1.0
        return {
            "type": "retention_ratio",
            "retention_ratio": retention_ratio,
        }

    def _resolved_voice(self) -> str | dict[str, Any]:
        """Resolve either a named voice or a custom voice-id object."""
        voice_id = str(getattr(self.config, "openai_realtime_voice_id", "") or "").strip()
        if voice_id:
            return {"id": voice_id}
        configured = getattr(self.config, "openai_realtime_voice", None)
        if isinstance(configured, Mapping):
            return copy.deepcopy(dict(configured))
        return str(configured or "marin").strip() or "marin"

    def _speech_speed(self) -> float:
        """Return a provider-safe speech speed."""
        value = _coerce_optional_float(
            getattr(self.config, "openai_realtime_speed", None),
            default=1.0,
            minimum=0.25,
        ) or 1.0
        return min(value, 1.5)

    def _temperature(self) -> float | None:
        """Return a provider-safe sampling temperature when configured."""
        value = getattr(self.config, "openai_realtime_temperature", None)
        if value is None:
            return None
        temperature = _coerce_optional_float(value, default=None, minimum=0.0)
        if temperature is None:
            return None
        if temperature < 0.6 or temperature > 1.2:
            return None
        return temperature

    def _max_response_output_tokens(self) -> int | str | None:
        """Return the session-level maximum output tokens."""
        raw_value = getattr(self.config, "openai_realtime_max_response_output_tokens", None)
        if raw_value is None:
            return None
        if isinstance(raw_value, str) and raw_value.strip().lower() == "inf":
            return "inf"
        value = _coerce_optional_int(raw_value, default=None, minimum=1)
        if value is None:
            return None
        return min(value, 4096)

    def _max_output_tokens(self) -> int | str | None:
        """Return the per-response maximum output tokens override."""
        raw_value = getattr(self.config, "openai_realtime_max_output_tokens", None)
        if raw_value is None:
            return None
        if isinstance(raw_value, str) and raw_value.strip().lower() == "inf":
            return "inf"
        value = _coerce_optional_int(raw_value, default=None, minimum=1)
        if value is None:
            return None
        return min(value, 4096)

    def _tool_choice(self) -> str:
        """Return the configured tool-choice policy."""
        tool_choice = str(getattr(self.config, "openai_realtime_tool_choice", "") or "").strip()
        return tool_choice or "auto"

    def _compute_client_transport_fingerprint(self) -> str:
        """Serialize the config values that require rebuilding the client."""
        fingerprint = {
            "api_key": str(getattr(self.config, "openai_api_key", "") or ""),
            "project": str(getattr(self.config, "openai_project_id", "") or ""),
            "organization": str(getattr(self.config, "openai_organization_id", "") or ""),
            "base_url": str(getattr(self.config, "openai_base_url", "") or ""),
            "websocket_base_url": str(
                getattr(self.config, "openai_realtime_websocket_base_url", "") or ""
            ),
            "default_headers": dict(getattr(self.config, "openai_default_headers", {}) or {}),
            "timeout": getattr(self.config, "openai_realtime_client_timeout_seconds", None),
            "max_retries": getattr(self.config, "openai_realtime_client_max_retries", None),
        }
        return self._stable_json(fingerprint)

    def _refresh_client_if_needed_locked(self) -> None:
        """Rebuild the SDK client if transport config changed and client is internal."""
        if self._client_is_custom:
            return
        desired_fingerprint = self._compute_client_transport_fingerprint()
        if desired_fingerprint == self._client_transport_fingerprint:
            return
        if self._connection is not None:
            self._close_locked()
        self._client = self._client_factory(self.config)
        self._client_transport_fingerprint = desired_fingerprint

    def _immutable_session_fingerprint(self) -> str:
        """Serialize reconnect-worthy session fields."""
        fingerprint = {
            "model": str(self.config.openai_realtime_model),
            "voice": self._resolved_voice(),
            "tracing": self._tracing_config(),
            "input_audio_rate": _PROVIDER_PCM_SAMPLE_RATE,
            "output_audio_rate": _PROVIDER_PCM_SAMPLE_RATE,
        }
        return self._stable_json(fingerprint)

    def _prepare_input_audio(self, audio_pcm: bytes) -> bytes:
        """Normalize Twinr PCM16 mono input to the provider's required 24 kHz."""
        if not audio_pcm:
            return audio_pcm
        if len(audio_pcm) % 2 != 0:
            raise RuntimeError("Realtime PCM input must contain an even number of bytes")
        input_rate = self._input_audio_sample_rate()
        if input_rate == _PROVIDER_PCM_SAMPLE_RATE:
            return audio_pcm
        return self._resample_pcm16_mono(audio_pcm, from_rate=input_rate, to_rate=_PROVIDER_PCM_SAMPLE_RATE)

    def _input_audio_sample_rate(self) -> int:
        """Return the Twinr capture/input sample rate."""
        return _coerce_optional_int(
            getattr(self.config, "openai_realtime_input_sample_rate", None),
            default=_PROVIDER_PCM_SAMPLE_RATE,
            minimum=1,
        ) or _PROVIDER_PCM_SAMPLE_RATE

    def _resample_pcm16_mono(self, audio_pcm: bytes, *, from_rate: int, to_rate: int) -> bytes:
        """Resample mono little-endian PCM16 using linear interpolation."""
        if from_rate <= 0 or to_rate <= 0:
            raise RuntimeError("Audio sample rates must be positive integers")
        if len(audio_pcm) % 2 != 0:
            raise RuntimeError("Realtime PCM input must contain an even number of bytes")
        if from_rate == to_rate or not audio_pcm:
            return audio_pcm

        samples = array("h")
        samples.frombytes(audio_pcm)
        if sys.byteorder != "little":
            samples.byteswap()

        if len(samples) <= 1:
            output = samples
        else:
            output_length = max(1, int(round(len(samples) * float(to_rate) / float(from_rate))))
            output = array("h")
            if output_length == 1:
                output.append(int(samples[0]))
            else:
                source_last_index = len(samples) - 1
                output_last_index = output_length - 1
                for output_index in range(output_length):
                    source_position = (output_index * source_last_index) / output_last_index
                    left_index = int(source_position)
                    right_index = min(left_index + 1, source_last_index)
                    fraction = source_position - left_index
                    interpolated = int(
                        round(
                            (samples[left_index] * (1.0 - fraction))
                            + (samples[right_index] * fraction)
                        )
                    )
                    if interpolated < -32768:
                        interpolated = -32768
                    elif interpolated > 32767:
                        interpolated = 32767
                    output.append(interpolated)

        if sys.byteorder != "little":
            output.byteswap()
        return output.tobytes()

    def _stable_json(self, value: Any) -> str:
        """Serialize config state deterministically for change detection."""
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=self._json_default)

    def _truncate_text(self, value: str, *, max_chars: int) -> str:
        """Truncate a string without returning invalid text payloads."""
        if max_chars <= 0 or len(value) <= max_chars:
            return value
        if max_chars <= 1:
            return value[:max_chars]
        return value[: max_chars - 1] + "…"

    def _make_callback_dispatcher(
        self,
        name: str,
        callback: Callable[[Any], None] | None,
    ) -> _StreamingCallbackDispatcher | None:
        """Create a background dispatcher for streamed callbacks."""
        if callback is None:
            return None
        return _StreamingCallbackDispatcher(
            name,
            callback,
            max_queue_size=self._callback_queue_maxsize(),
            join_timeout_seconds=self._callback_join_timeout_seconds(),
        )

    def _close_callback_dispatcher(self, dispatcher: _StreamingCallbackDispatcher | None) -> None:
        """Close a callback dispatcher and surface any worker failure."""
        if dispatcher is None:
            return
        dispatcher.close()

    def _raise_callback_dispatcher_failure(self, dispatcher: _StreamingCallbackDispatcher | None) -> None:
        """Raise any worker-thread callback failure back into the turn."""
        if dispatcher is None:
            return
        dispatcher.raise_if_failed()

    def _tool_argument_max_chars(self) -> int:
        """Return the maximum allowed size of tool argument JSON payloads."""
        return _coerce_optional_int(
            getattr(self.config, "openai_realtime_tool_argument_max_chars", None),
            default=_DEFAULT_TOOL_ARGUMENT_MAX_CHARS,
            minimum=256,
        ) or _DEFAULT_TOOL_ARGUMENT_MAX_CHARS

    def _tool_output_max_chars(self) -> int:
        """Return the maximum allowed size of serialized tool outputs."""
        return _coerce_optional_int(
            getattr(self.config, "openai_realtime_tool_output_max_chars", None),
            default=_DEFAULT_TOOL_OUTPUT_MAX_CHARS,
            minimum=256,
        ) or _DEFAULT_TOOL_OUTPUT_MAX_CHARS

    def _max_tool_roundtrips_per_turn(self) -> int:
        """Return the maximum tool-execution loops allowed in one turn."""
        return _coerce_optional_int(
            getattr(self.config, "openai_realtime_max_tool_roundtrips_per_turn", None),
            default=_DEFAULT_MAX_TOOL_ROUNDTRIPS_PER_TURN,
            minimum=1,
        ) or _DEFAULT_MAX_TOOL_ROUNDTRIPS_PER_TURN

    def _callback_queue_maxsize(self) -> int:
        """Return the bounded queue size used for streamed callbacks."""
        return _coerce_optional_int(
            getattr(self.config, "openai_realtime_callback_queue_maxsize", None),
            default=_DEFAULT_CALLBACK_QUEUE_MAXSIZE,
            minimum=1,
        ) or _DEFAULT_CALLBACK_QUEUE_MAXSIZE

    def _callback_join_timeout_seconds(self) -> float:
        """Return the timeout used when draining callback workers on turn end."""
        return _coerce_optional_float(
            getattr(self.config, "openai_realtime_callback_join_timeout_seconds", None),
            default=_DEFAULT_CALLBACK_JOIN_TIMEOUT_SECONDS,
            minimum=0.1,
        ) or _DEFAULT_CALLBACK_JOIN_TIMEOUT_SECONDS

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
