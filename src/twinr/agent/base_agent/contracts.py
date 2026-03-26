"""Define provider contracts and composition helpers for the base agent.

This module is the structural contract layer between workflow/runtime code and
provider implementations. It keeps protocol surfaces, tool-call value objects,
and the composite provider glue in one import-stable place.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
import logging
import mimetypes
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from twinr.agent.base_agent.config import TwinrConfig
else:
    TwinrConfig = Any  # AUDIT-FIX(#6): Stellt einen Runtime-Fallback für Annotation-Resolution via get_type_hints bereit.

_LOGGER = logging.getLogger(__name__)


ConversationLike = Sequence[object]
ImageInputLike = object


@runtime_checkable
class ConfigurableProvider(Protocol):
    """Protocol for provider objects that expose a mutable ``config`` field."""

    config: TwinrConfig


@runtime_checkable
class TextResponse(Protocol):
    """Protocol for text-generation responses with shared metadata fields."""

    text: str
    response_id: str | None
    request_id: str | None
    model: str | None
    token_usage: object | None
    used_web_search: bool


@runtime_checkable
class SearchResponse(Protocol):
    """Protocol for search responses that include an answer and sources."""

    answer: str
    sources: tuple[str, ...]
    response_id: str | None
    request_id: str | None
    model: str | None
    token_usage: object | None
    used_web_search: bool


@dataclass(frozen=True, slots=True)
class AgentToolCall:
    """Represent one requested tool invocation from a model turn."""

    name: str
    call_id: str
    arguments: dict[str, Any]
    raw_arguments: str = "{}"

    def __post_init__(self) -> None:
        if not isinstance(self.arguments, Mapping):
            raise TypeError("AgentToolCall.arguments must be a mapping")
        object.__setattr__(
            self,
            "arguments",
            deepcopy(dict(self.arguments)),  # AUDIT-FIX(#4): Kapselt Tool-Argumente defensiv, damit externe Mutationen keinen shared mutable state erzeugen.
        )


@dataclass(frozen=True, slots=True)
class AgentToolResult:
    """Capture the serialized output returned to a tool-calling model."""

    call_id: str
    name: str
    output: Any
    serialized_output: str


@dataclass(frozen=True, slots=True)
class ToolCallingTurnResponse:
    """Store one tool-capable model turn and its continuation metadata."""

    text: str
    tool_calls: tuple[AgentToolCall, ...] = ()
    response_id: str | None = None
    request_id: str | None = None
    model: str | None = None
    token_usage: object | None = None
    used_web_search: bool = False
    continuation_token: str | None = None

    def __post_init__(self) -> None:
        normalized_tool_calls = tuple(self.tool_calls)
        if any(not isinstance(tool_call, AgentToolCall) for tool_call in normalized_tool_calls):
            raise TypeError("ToolCallingTurnResponse.tool_calls must contain AgentToolCall instances")
        object.__setattr__(
            self,
            "tool_calls",
            normalized_tool_calls,  # AUDIT-FIX(#5): Normalisiert auf Tuple und verhindert Alias-Bugs über mutierbare Sequenzen.
        )


@dataclass(frozen=True, slots=True)
class SupervisorDecision:
    """Represent the supervisor model's decision for streaming orchestration."""

    action: str
    spoken_ack: str | None = None
    spoken_reply: str | None = None
    kind: str | None = None
    goal: str | None = None
    prompt: str | None = None
    allow_web_search: bool | None = None
    location_hint: str | None = None
    date_context: str | None = None
    context_scope: str | None = None
    runtime_tool_name: str | None = None
    runtime_tool_arguments: dict[str, object] | None = None
    response_id: str | None = None
    request_id: str | None = None
    model: str | None = None
    token_usage: object | None = None

    def __post_init__(self) -> None:
        normalized = self.action.strip().lower()
        if normalized not in {"direct", "handoff", "end_conversation"}:
            raise ValueError(f"Unsupported supervisor action: {self.action}")
        object.__setattr__(self, "action", normalized)
        object.__setattr__(
            self,
            "context_scope",
            normalize_supervisor_decision_context_scope(self.context_scope),
        )
        object.__setattr__(
            self,
            "runtime_tool_name",
            normalize_supervisor_decision_runtime_tool_name(self.runtime_tool_name),
        )
        object.__setattr__(
            self,
            "runtime_tool_arguments",
            normalize_supervisor_decision_runtime_tool_arguments(self.runtime_tool_arguments),
        )


_SUPERVISOR_CONTEXT_SCOPES = frozenset({"tiny_recent", "full_context"})


def normalize_supervisor_decision_context_scope(value: object) -> str | None:
    """Return the validated supervisor decision context scope, if present."""

    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in _SUPERVISOR_CONTEXT_SCOPES:
        return normalized
    return None


def normalize_supervisor_decision_runtime_tool_name(value: object) -> str | None:
    """Return one stripped runtime-local tool name, if present."""

    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def normalize_supervisor_decision_runtime_tool_arguments(
    value: object,
) -> dict[str, object] | None:
    """Return JSON-like runtime-local tool arguments when the payload is a dict."""

    if not isinstance(value, dict):
        return None
    normalized: dict[str, object] = {}
    for key, item in value.items():
        key_text = str(key).strip()
        if not key_text:
            continue
        normalized[key_text] = item
    return normalized


def supervisor_decision_requires_full_context(decision: object | None) -> bool:
    """Return whether a supervisor decision says the fast lane lacks context."""

    if decision is None:
        return False
    return normalize_supervisor_decision_context_scope(
        getattr(decision, "context_scope", None)
    ) == "full_context"


@dataclass(frozen=True, slots=True)
class FirstWordReply:
    """Store the first-word lane reply chosen for low-latency speech output."""

    mode: str
    spoken_text: str
    response_id: str | None = None
    request_id: str | None = None
    model: str | None = None
    token_usage: object | None = None

    def __post_init__(self) -> None:
        normalized_mode = self.mode.strip().lower()
        if normalized_mode not in {"direct", "filler"}:
            raise ValueError(f"Unsupported first-word mode: {self.mode}")
        text = str(self.spoken_text or "").strip()
        if not text:
            raise ValueError("FirstWordReply.spoken_text must not be empty")
        object.__setattr__(self, "mode", normalized_mode)
        object.__setattr__(self, "spoken_text", text)


@dataclass(frozen=True, slots=True)
class ConversationClosureProviderDecision:
    """Store one structured closure-decision response from a provider.

    Attributes:
        close_now: Whether Twinr should stop automatic follow-up listening
            after the just-finished exchange.
        confidence: Normalized confidence score in the range ``0.0`` to
            ``1.0``.
        reason: Short provider-supplied reason string.
        matched_topics: Up to two matched steering-topic titles echoed from the
            current turn context.
        response_id: Provider response identifier when available.
        request_id: Transport request identifier when available.
        model: Provider model identifier when available.
        token_usage: Provider token-usage metadata when available.
    """

    close_now: bool
    confidence: float
    reason: str
    matched_topics: tuple[str, ...] = ()
    response_id: str | None = None
    request_id: str | None = None
    model: str | None = None
    token_usage: object | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "matched_topics",
            tuple(str(topic).strip() for topic in self.matched_topics if str(topic).strip()),
        )


@dataclass(frozen=True, slots=True)
class StreamingTranscriptionResult:
    """Store the current streaming transcription snapshot for a session."""

    transcript: str
    request_id: str | None = None
    saw_interim: bool = False
    saw_speech_final: bool = False
    saw_utterance_end: bool = False
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class StreamingSpeechEndpointEvent:
    """Represent a streaming endpoint signal emitted by an STT backend."""

    transcript: str
    event_type: str
    request_id: str | None = None
    is_final: bool = False
    speech_final: bool = False
    from_finalize: bool = False


@runtime_checkable
class SpeechToTextProvider(ConfigurableProvider, Protocol):
    """Protocol for providers that transcribe complete audio payloads."""

    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
        language: str | None = None,
        prompt: str | None = None,
    ) -> str:
        ...


@runtime_checkable
class PathSpeechToTextProvider(SpeechToTextProvider, Protocol):
    """Protocol for STT providers that can transcribe directly from a path."""

    def transcribe_path(
        self,
        path: str | Path,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> str:
        ...


@runtime_checkable
class StreamingSpeechToTextSession(Protocol):
    """Protocol for a live streaming transcription session instance."""

    def send_pcm(self, pcm_bytes: bytes) -> None:
        ...

    def snapshot(self) -> StreamingTranscriptionResult:
        ...

    def finalize(self) -> StreamingTranscriptionResult:
        ...

    def close(self) -> None:
        ...


@runtime_checkable
class StreamingSpeechToTextProvider(PathSpeechToTextProvider, Protocol):
    """Protocol for providers that open live streaming STT sessions."""

    def start_streaming_session(
        self,
        *,
        sample_rate: int,
        channels: int,
        language: str | None = None,
        prompt: str | None = None,
        on_interim: Callable[[str], None] | None = None,
        on_endpoint: Callable[[StreamingSpeechEndpointEvent], None] | None = None,
    ) -> StreamingSpeechToTextSession:
        ...


@runtime_checkable
class AgentTextProvider(ConfigurableProvider, Protocol):
    """Protocol for text-generation providers used by the base agent."""

    def respond_streaming(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> TextResponse:
        ...

    def respond_with_metadata(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
    ) -> TextResponse:
        ...

    def respond_to_images_with_metadata(
        self,
        prompt: str,
        *,
        images: Sequence[ImageInputLike],
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
    ) -> TextResponse:
        ...

    def search_live_info_with_metadata(
        self,
        question: str,
        *,
        conversation: ConversationLike | None = None,
        location_hint: str | None = None,
        date_context: str | None = None,
    ) -> SearchResponse:
        ...

    def compose_print_job_with_metadata(
        self,
        *,
        conversation: ConversationLike | None = None,
        focus_hint: str | None = None,
        direct_text: str | None = None,
        request_source: str = "button",
    ) -> TextResponse:
        ...

    def phrase_due_reminder_with_metadata(
        self,
        reminder: object,
        *,
        now: datetime | None = None,
    ) -> TextResponse:
        ...

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
        ...

    def fulfill_automation_prompt_with_metadata(
        self,
        prompt: str,
        *,
        allow_web_search: bool,
        delivery: str = "spoken",
    ) -> TextResponse:
        ...


@runtime_checkable
class ToolCallingAgentProvider(ConfigurableProvider, Protocol):
    """Protocol for agents that can emit tool calls and continue turns."""

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
        ...

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
        ...


@runtime_checkable
class SupervisorDecisionProvider(ConfigurableProvider, Protocol):
    """Protocol for providers that return a ``SupervisorDecision``."""

    def decide(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
    ) -> SupervisorDecision:
        ...


@runtime_checkable
class FirstWordProvider(ConfigurableProvider, Protocol):
    """Protocol for providers that produce a bounded first-word reply."""

    def reply(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
    ) -> FirstWordReply:
        ...


@runtime_checkable
class ConversationClosureProvider(ConfigurableProvider, Protocol):
    """Protocol for providers that return one closure-decision object."""

    def decide(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        timeout_seconds: float | None = None,
    ) -> ConversationClosureProviderDecision:
        ...


@runtime_checkable
class TextToSpeechProvider(ConfigurableProvider, Protocol):
    """Protocol for providers that synthesize spoken output."""

    def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        response_format: str | None = None,
        instructions: str | None = None,
    ) -> bytes:
        ...

    def synthesize_stream(
        self,
        text: str,
        *,
        voice: str | None = None,
        response_format: str | None = None,
        instructions: str | None = None,
        chunk_size: int = 4096,
    ) -> Iterator[bytes]:
        ...


@runtime_checkable
class CombinedSpeechAgentProvider(
    SpeechToTextProvider,
    AgentTextProvider,
    TextToSpeechProvider,
    Protocol,
):
    """Protocol for a single provider implementing STT, text, and TTS."""

    pass


def _ensure_runtime_protocol(name: str, provider: object, protocol: type[Any]) -> None:
    """Raise early when a configured provider misses a required protocol."""

    # AUDIT-FIX(#2): Erzwingt Fail-Fast-Validierung der injizierten Provider gegen die Runtime-Protokolle.
    if not isinstance(provider, protocol):
        raise TypeError(
            f"{name} must implement {protocol.__name__}; got {type(provider).__name__}"
        )


def _unique_configurable_providers(
    providers: Sequence[ConfigurableProvider],
) -> tuple[ConfigurableProvider, ...]:
    """Deduplicate provider objects while preserving their original order."""

    unique_providers: list[ConfigurableProvider] = []
    seen: set[int] = set()
    for provider in providers:
        provider_id = id(provider)
        if provider_id in seen:
            continue
        seen.add(provider_id)
        unique_providers.append(provider)
    return tuple(unique_providers)


def _apply_config_atomically(
    providers: Sequence[ConfigurableProvider],
    value: TwinrConfig,
) -> None:
    """Apply one config object across providers with best-effort rollback."""

    # AUDIT-FIX(#3): Wendet Konfigurationsänderungen rollback-fähig an, damit der Composite nicht halb umkonfiguriert zurückbleibt.
    previous_configs: list[tuple[ConfigurableProvider, TwinrConfig]] = []
    try:
        for provider in _unique_configurable_providers(providers):
            previous_configs.append((provider, provider.config))
            provider.config = value
    except Exception as exc:
        for provider, previous_value in reversed(previous_configs):
            try:
                provider.config = previous_value
            except Exception:
                _LOGGER.warning(
                    "Failed to roll back provider config for %s after composite apply_config failure.",
                    type(provider).__name__,
                    exc_info=True,
                )
        raise RuntimeError("Failed to apply config consistently across providers") from exc


def _guess_audio_content_type(path: str | Path) -> str:
    """Infer an audio content type from a file path with a WAV fallback."""

    guessed_type, _ = mimetypes.guess_type(str(path), strict=False)
    return guessed_type or "audio/wav"


def _read_audio_bytes_from_path(path: str | Path) -> bytes:
    """Read audio bytes from a path and raise clearer path-type errors."""

    audio_path = Path(path)
    # AUDIT-FIX(#1): Liest den Pfad in einem Schritt ein und vermeidet damit blindes Attributrouting plus unnötige Check-then-use-Rennen.
    try:
        return audio_path.read_bytes()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Audio file not found: {audio_path}") from exc
    except IsADirectoryError as exc:
        raise IsADirectoryError(f"Audio path is not a file: {audio_path}") from exc


@dataclass
class ProviderBundle:
    """Bundle the base agent's primary providers into one validated object."""

    stt: SpeechToTextProvider
    agent: AgentTextProvider
    tts: TextToSpeechProvider
    tool_agent: ToolCallingAgentProvider | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        _ensure_runtime_protocol("stt", self.stt, SpeechToTextProvider)  # AUDIT-FIX(#2): Validiert Provider bereits beim Bundling statt erst beim Live-Aufruf.
        _ensure_runtime_protocol("agent", self.agent, AgentTextProvider)  # AUDIT-FIX(#2): Verhindert späte Laufzeitcrashs durch fehlerhafte DI.
        _ensure_runtime_protocol("tts", self.tts, TextToSpeechProvider)  # AUDIT-FIX(#2): Verhindert späte Laufzeitcrashs durch fehlerhafte DI.
        if self.tool_agent is not None:
            _ensure_runtime_protocol("tool_agent", self.tool_agent, ToolCallingAgentProvider)  # AUDIT-FIX(#2): Optionaler Tool-Agent wird identisch fail-fast validiert.


class CompositeSpeechAgentProvider:
    """Delegate STT, text, and TTS calls to separate coordinated providers.

    The composite exposes one provider-like surface while keeping the three
    injected providers' configs synchronized and offering a path-based STT
    fallback when the selected STT provider only accepts raw bytes.
    """

    def __init__(
        self,
        *,
        stt: SpeechToTextProvider,
        agent: AgentTextProvider,
        tts: TextToSpeechProvider,
    ) -> None:
        _ensure_runtime_protocol("stt", stt, SpeechToTextProvider)  # AUDIT-FIX(#2): Stoppt Fehlverkabelung beim Systemstart statt beim ersten Senior-Request.
        _ensure_runtime_protocol("agent", agent, AgentTextProvider)  # AUDIT-FIX(#2): Stoppt Fehlverkabelung beim Systemstart statt beim ersten Senior-Request.
        _ensure_runtime_protocol("tts", tts, TextToSpeechProvider)  # AUDIT-FIX(#2): Stoppt Fehlverkabelung beim Systemstart statt beim ersten Senior-Request.
        self._stt = stt
        self._agent = agent
        self._tts = tts
        self._providers = _unique_configurable_providers((stt, agent, tts))
        self.config = agent.config  # AUDIT-FIX(#3): Synchronisiert die Initialkonfiguration über alle Provider, damit der Composite konsistent startet.

    @property
    def config(self) -> TwinrConfig:
        return self._agent.config

    @config.setter
    def config(self, value: TwinrConfig) -> None:
        _apply_config_atomically(self._providers, value)  # AUDIT-FIX(#3): Verhindert partielle Provider-Updates bei Setter-Fehlern.

    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
        language: str | None = None,
        prompt: str | None = None,
    ) -> str:
        return self._stt.transcribe(
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
        if isinstance(self._stt, PathSpeechToTextProvider):
            return self._stt.transcribe_path(path, language=language, prompt=prompt)

        audio_path = Path(path)
        audio_bytes = _read_audio_bytes_from_path(audio_path)
        return self._stt.transcribe(
            audio_bytes,
            filename=audio_path.name,
            content_type=_guess_audio_content_type(audio_path),  # AUDIT-FIX(#1): Fallback auf byte-basierte STT-Transkription statt blindem Attributzugriff.
            language=language,
            prompt=prompt,
        )

    def respond_streaming(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> TextResponse:
        return self._agent.respond_streaming(
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
        return self._agent.respond_with_metadata(
            prompt,
            conversation=conversation,
            instructions=instructions,
            allow_web_search=allow_web_search,
        )

    def respond_to_images_with_metadata(
        self,
        prompt: str,
        *,
        images: Sequence[ImageInputLike],
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
    ) -> TextResponse:
        return self._agent.respond_to_images_with_metadata(
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
        return self._agent.search_live_info_with_metadata(
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
        return self._agent.compose_print_job_with_metadata(
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
        return self._agent.phrase_due_reminder_with_metadata(reminder, now=now)

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
        return self._agent.phrase_proactive_prompt_with_metadata(
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
        return self._agent.fulfill_automation_prompt_with_metadata(
            prompt,
            allow_web_search=allow_web_search,
            delivery=delivery,
        )

    def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        response_format: str | None = None,
        instructions: str | None = None,
    ) -> bytes:
        return self._tts.synthesize(
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
    ) -> Iterator[bytes]:
        return self._tts.synthesize_stream(
            text,
            voice=voice,
            response_format=response_format,
            instructions=instructions,
            chunk_size=chunk_size,
        )


class FoundationModelProvider(Protocol):
    """Protocol for simple prompt-in, text-out model adapters."""

    def respond(self, prompt: str) -> str:
        ...


class PrintFormatter(Protocol):
    """Protocol for formatters that adapt text to the printer surface."""

    def format_for_print(self, text: str) -> str:
        ...
