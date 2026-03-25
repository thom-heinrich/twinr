"""Evaluate whether a finished exchange should keep follow-up listening open.

The closure layer keeps the post-response prompt bounded and can route the
decision through either the legacy tool-calling path or a faster structured
decision provider. Both paths are coerced into the same safe
``ConversationClosureDecision`` shape for workflow consumers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from threading import Event, Thread
import json
from typing import Any, Protocol, cast, runtime_checkable

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.conversation.decision_core import (
    coerce_probability as _coerce_probability,
    coerce_text as _coerce_text,
    compact_conversation as _compact_conversation,
    config_float as _config_float,
    config_int as _config_int,
    detect_provider_timeout_kwarg,
    extract_json_object as _extract_json_object,
)
from twinr.agent.base_agent.contracts import (
    ConversationClosureProvider,
    ConversationClosureProviderDecision,
    ConversationLike,
    ToolCallingAgentProvider,
)
from twinr.agent.base_agent.prompting.personality import load_conversation_closure_instructions
from twinr.agent.personality.steering import ConversationTurnSteeringCue, serialize_turn_steering_cues

_DEFAULT_CONTEXT_TURNS = 4
_DEFAULT_MAX_TRANSCRIPT_CHARS = 512
_DEFAULT_MAX_RESPONSE_CHARS = 512
_DEFAULT_MAX_REASON_CHARS = 256
_DEFAULT_PROVIDER_TIMEOUT_SECONDS = 2.0

_CLOSURE_DECISION_TOOL_SCHEMA: dict[str, object] = {
    "type": "function",
    "name": "submit_closure_decision",
    "description": "Decide whether Twinr should stop automatic follow-up listening after the just-finished exchange.",
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "close_now": {
                "type": "boolean",
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "reason": {
                "type": "string",
            },
            "matched_topics": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["close_now", "confidence", "reason"],
    },
}


@dataclass(frozen=True, slots=True)
class ConversationClosureDecision:
    """Describe whether Twinr should end automatic follow-up listening.

    Attributes:
        close_now: Whether follow-up listening should stop for the current
            exchange.
        confidence: Normalized confidence score in the range ``0.0`` to
            ``1.0``.
        reason: Short bounded reason code or summary from the evaluator.
        matched_topics: Topic titles from the provided turn-steering cues that
            the evaluator considered relevant to the exchange.
    """

    close_now: bool
    confidence: float
    reason: str
    matched_topics: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ConversationClosureEvaluation:
    """Capture one bounded closure-evaluation attempt for workflow consumers."""

    decision: ConversationClosureDecision | None = None
    error_type: str | None = None
    turn_steering_cues: tuple[ConversationTurnSteeringCue, ...] = ()


@runtime_checkable
class ConversationClosureEvaluator(Protocol):
    """Protocol for runtime evaluators that return closure decisions."""

    def evaluate(
        self,
        *,
        user_transcript: str,
        assistant_response: str,
        request_source: str,
        proactive_trigger: str | None = None,
        conversation: ConversationLike | None = None,
        turn_steering_cues: Sequence[ConversationTurnSteeringCue] = (),
    ) -> ConversationClosureDecision:
        ...


class _ConversationClosureEvaluatorBase:
    """Shared prompt assembly and coercion helpers for closure evaluators."""

    def __init__(self, *, config: TwinrConfig) -> None:
        self.config = config

    def _compact_conversation(
        self,
        conversation: ConversationLike | None,
    ) -> tuple[tuple[str, str], ...]:
        """Build the bounded recent-turn context for one closure decision."""

        return _compact_conversation(
            conversation,
            max_turns=_config_int(
                self.config,
                "conversation_closure_context_turns",
                _DEFAULT_CONTEXT_TURNS,
                minimum=0,
                maximum=32,
            ),
            max_item_chars=_config_int(
                self.config,
                "conversation_closure_max_transcript_chars",
                _DEFAULT_MAX_TRANSCRIPT_CHARS,
                minimum=64,
                maximum=4096,
            ),
            max_total_chars=8192,
        )

    def _build_prompt(
        self,
        *,
        user_transcript: str,
        assistant_response: str,
        request_source: str,
        proactive_trigger: str | None,
        conversation: tuple[tuple[str, str], ...],
        turn_steering_cues: Sequence[ConversationTurnSteeringCue],
    ) -> str:
        payload = {
            "task": "Decide whether Twinr should suppress any automatic follow-up listening because the exchange has clearly ended for now.",
            "exchange": {
                "user_transcript": _coerce_text(
                    user_transcript,
                    max_chars=_config_int(
                        self.config,
                        "conversation_closure_max_transcript_chars",
                        _DEFAULT_MAX_TRANSCRIPT_CHARS,
                        minimum=64,
                        maximum=4096,
                    ),
                ),
                "assistant_response": _coerce_text(
                    assistant_response,
                    max_chars=_config_int(
                        self.config,
                        "conversation_closure_max_response_chars",
                        _DEFAULT_MAX_RESPONSE_CHARS,
                        minimum=64,
                        maximum=4096,
                    ),
                ),
                "request_source": _coerce_text(request_source, default="button", max_chars=64) or "button",
                "proactive_trigger": _coerce_text(proactive_trigger, max_chars=128) or None,
                "recent_turn_count": len(conversation),
            },
            "turn_steering": {
                "topics": serialize_turn_steering_cues(turn_steering_cues),
                "instruction": (
                    "If one or two of these topics clearly match the exchange, echo exactly those provided titles in matched_topics. "
                    "Use both title and match_summary. Do not match merely because something sounds adjacent, local, communal, or loosely related. "
                    "Do not invent topic names."
                ),
            },
        }
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))

    def _coerce_decision(self, payload: dict[str, object]) -> ConversationClosureDecision:
        return ConversationClosureDecision(
            close_now=bool(payload.get("close_now", False)),
            confidence=_coerce_probability(payload.get("confidence", 0.0), default=0.0),
            reason=_coerce_text(
                payload.get("reason", ""),
                default="closure_controller_fallback",
                max_chars=_config_int(
                    self.config,
                    "conversation_closure_max_reason_chars",
                    _DEFAULT_MAX_REASON_CHARS,
                    minimum=32,
                    maximum=1024,
                ),
            )
            or "closure_controller_fallback",
            matched_topics=self._coerce_matched_topics(payload.get("matched_topics")),
        )

    def _coerce_provider_decision(
        self,
        decision: ConversationClosureProviderDecision,
    ) -> ConversationClosureDecision:
        """Normalize one provider-native decision into the runtime contract."""

        return ConversationClosureDecision(
            close_now=bool(decision.close_now),
            confidence=_coerce_probability(decision.confidence, default=0.0),
            reason=_coerce_text(
                decision.reason,
                default="closure_controller_fallback",
                max_chars=_config_int(
                    self.config,
                    "conversation_closure_max_reason_chars",
                    _DEFAULT_MAX_REASON_CHARS,
                    minimum=32,
                    maximum=1024,
                ),
            )
            or "closure_controller_fallback",
            matched_topics=self._coerce_matched_topics(decision.matched_topics),
        )

    def _coerce_matched_topics(self, value: object) -> tuple[str, ...]:
        """Coerce matched topic titles into a bounded normalized tuple."""

        if value is None:
            return ()
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            return ()
        topics: list[str] = []
        seen: set[str] = set()
        for raw_topic in value:
            normalized = _coerce_text(raw_topic, max_chars=96)
            if not normalized:
                continue
            key = normalized.casefold()
            if key in seen:
                continue
            seen.add(key)
            topics.append(normalized)
            if len(topics) >= 2:
                break
        return tuple(topics)

    def _provider_timeout_seconds(self) -> float:
        """Return the bounded provider budget for one closure request."""

        return _config_float(
            self.config,
            "conversation_closure_provider_timeout_seconds",
            _DEFAULT_PROVIDER_TIMEOUT_SECONDS,
            minimum=0.25,
            maximum=15.0,
        )

    def _call_with_watchdog(
        self,
        *,
        timeout_seconds: float,
        target_name: str,
        call,
    ):
        """Bound closure-model latency even when an adapter blocks too long."""

        done = Event()
        response_holder: list[object] = []
        error_holder: list[BaseException] = []

        def _worker() -> None:
            try:
                response_holder.append(call())
            except BaseException as exc:
                error_holder.append(exc)
            finally:
                done.set()

        worker = Thread(
            target=_worker,
            daemon=True,
            name=f"twinr-{target_name}",
        )
        worker.start()
        if not done.wait(timeout_seconds):
            raise TimeoutError(
                f"Conversation closure evaluation exceeded {timeout_seconds:.2f}s"
            )
        if error_holder:
            raise error_holder[0]
        if not response_holder:
            raise RuntimeError("Conversation closure evaluation returned no response")
        return response_holder[0]


class ToolCallingConversationClosureEvaluator(_ConversationClosureEvaluatorBase):
    """Ask a tool-calling provider for a structured closure decision.

    The evaluator summarizes the just-finished exchange, loads dedicated
    closure instructions, and accepts either tool arguments or JSON text as the
    provider response format.
    """

    def __init__(
        self,
        *,
        config: TwinrConfig,
        provider: ToolCallingAgentProvider,
    ) -> None:
        super().__init__(config=config)
        self.provider = provider
        self._provider_timeout_kwarg_name = detect_provider_timeout_kwarg(self.provider.start_turn_streaming)

    def evaluate(
        self,
        *,
        user_transcript: str,
        assistant_response: str,
        request_source: str,
        proactive_trigger: str | None = None,
        conversation: ConversationLike | None = None,
        turn_steering_cues: Sequence[ConversationTurnSteeringCue] = (),
    ) -> ConversationClosureDecision:
        """Evaluate whether Twinr should close follow-up listening now.

        Args:
            user_transcript: Final user transcript for the just-finished turn.
            assistant_response: Assistant response that was spoken or prepared.
            request_source: Origin of the exchange, such as ``button``.
            proactive_trigger: Optional proactive trigger name that opened the
                exchange.
            conversation: Optional recent conversation history for context.
            turn_steering_cues: Current machine-readable steering cues that may
                map the exchange onto a shared thread or a cooling topic.

        Returns:
            A bounded structured decision describing whether follow-up listening
            should stop after this response.
        """

        compact_conversation = self._compact_conversation(conversation)
        prompt = self._build_prompt(
            user_transcript=user_transcript,
            assistant_response=assistant_response,
            request_source=request_source,
            proactive_trigger=proactive_trigger,
            conversation=compact_conversation,
            turn_steering_cues=turn_steering_cues,
        )
        timeout_seconds = self._provider_timeout_seconds()
        provider_kwargs: dict[str, object] = {}
        if self._provider_timeout_kwarg_name is not None:
            provider_kwargs[self._provider_timeout_kwarg_name] = timeout_seconds
        response = self._call_with_watchdog(
            timeout_seconds=timeout_seconds,
            target_name="closure-evaluator",
            call=lambda: cast(Any, self.provider.start_turn_streaming)(
                prompt,
                conversation=compact_conversation,
                instructions=load_conversation_closure_instructions(self.config),
                tool_schemas=(_CLOSURE_DECISION_TOOL_SCHEMA,),
                allow_web_search=False,
                **provider_kwargs,
            ),
        )
        tool_calls = getattr(response, "tool_calls", ()) or ()
        for tool_call in tool_calls:
            if _coerce_text(getattr(tool_call, "name", ""), max_chars=64) != "submit_closure_decision":
                continue
            raw_arguments = getattr(tool_call, "arguments", {})
            if isinstance(raw_arguments, dict):
                payload = raw_arguments
            else:
                payload = _extract_json_object(raw_arguments) or {}
            return self._coerce_decision(payload)
        payload = _extract_json_object(getattr(response, "text", "")) or {}
        return self._coerce_decision(payload)

class StructuredConversationClosureEvaluator(_ConversationClosureEvaluatorBase):
    """Ask a structured decision provider for one fast closure decision."""

    def __init__(
        self,
        *,
        config: TwinrConfig,
        provider: ConversationClosureProvider,
    ) -> None:
        super().__init__(config=config)
        self.provider = provider

    def evaluate(
        self,
        *,
        user_transcript: str,
        assistant_response: str,
        request_source: str,
        proactive_trigger: str | None = None,
        conversation: ConversationLike | None = None,
        turn_steering_cues: Sequence[ConversationTurnSteeringCue] = (),
    ) -> ConversationClosureDecision:
        compact_conversation = self._compact_conversation(conversation)
        prompt = self._build_prompt(
            user_transcript=user_transcript,
            assistant_response=assistant_response,
            request_source=request_source,
            proactive_trigger=proactive_trigger,
            conversation=compact_conversation,
            turn_steering_cues=turn_steering_cues,
        )
        timeout_seconds = self._provider_timeout_seconds()
        decision = self._call_with_watchdog(
            timeout_seconds=timeout_seconds,
            target_name="closure-decision",
            call=lambda: self.provider.decide(
                prompt,
                conversation=compact_conversation,
                instructions=load_conversation_closure_instructions(self.config),
                timeout_seconds=timeout_seconds,
            ),
        )
        if not isinstance(decision, ConversationClosureProviderDecision):
            raise RuntimeError("Conversation closure decision provider returned an invalid result")
        return self._coerce_provider_decision(decision)
