# CHANGELOG: 2026-03-27
# BUG-1: Fixed bare speech_final deferral. A non-empty speech_final transcript is now evaluated immediately instead of waiting forever for a later finalize event.
# BUG-2: Fixed malformed/empty model outputs being treated as provider success. Invalid structured outputs now count as evaluator failures and can open the circuit breaker.
# SEC-1: Mitigated practical availability DoS from hung upstream evaluator calls. Timeouts/worker-busy states now fall back to local semantic turn decisions instead of disabling turn ending until the provider recovers.
# IMP-1: Added a hybrid semantic turn detector that fuses endpoint/finalization signals with lexical completion, backchannel, hesitation, and short-answer cues before escalating to the remote model.
# IMP-2: Upgraded prompt/tool handling with strict schema hints, richer bounded candidate metadata, and stronger model/local fallback blending aligned with 2026 streaming dialogue patterns.

"""Evaluate streaming turn boundaries for the base agent.

This module turns partial and final streaming speech events into bounded
``TurnDecision`` objects. It includes a hybrid evaluator and a controller that
shields the audio path from provider latency, parse errors, thread failures,
stale results, and weak endpoint signals.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Any, Literal, Protocol, cast
import json
import re
import time

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.conversation.decision_core import (
    coerce_probability as _coerce_probability,
    coerce_text as _coerce_text,
    compact_conversation as _compact_conversation,
    config_bool as _config_bool,
    config_float as _config_float,
    config_int as _config_int,
    detect_provider_timeout_kwarg,
    extract_json_object as _extract_json_object,
    normalize_turn_text,
    safe_getattr as _safe_getattr,
    sanitize_emit_value as _sanitize_emit_value,
)
from twinr.agent.base_agent.contracts import (
    ConversationLike,
    StreamingSpeechEndpointEvent,
    ToolCallingAgentProvider,
)
from twinr.agent.base_agent.prompting.personality import load_turn_controller_instructions

TurnDecisionName = Literal["continue_listening", "end_turn"]
TurnDecisionLabel = Literal["complete", "incomplete", "backchannel", "wait"]


class TurnControllerEvaluationError(RuntimeError):
    """Raised when the single turn-controller lane cannot produce a decision."""

_DEFAULT_CONTEXT_TURNS = 8
_DEFAULT_MAX_TRANSCRIPT_CHARS = 4096
_DEFAULT_MAX_REASON_CHARS = 256
_DEFAULT_MAX_CONVERSATION_ITEM_CHARS = 2000
_DEFAULT_MAX_CONVERSATION_TOTAL_CHARS = 12000
_DEFAULT_PROVIDER_TIMEOUT_SECONDS = 2.0
_DEFAULT_EVALUATION_TIMEOUT_SECONDS = 2.5
_DEFAULT_CIRCUIT_BREAKER_FAILURES = 3
_DEFAULT_CIRCUIT_BREAKER_COOLDOWN_SECONDS = 15.0
_DEFAULT_EMIT_VALUE_MAX_CHARS = 256
_DEFAULT_MODEL_TEXT_FALLBACK_MAX_CHARS = 8192
_DEFAULT_LOCAL_SHORT_CIRCUIT_CONFIDENCE = 0.90
_DEFAULT_LOCAL_FALLBACK_CONFIDENCE = 0.72
_DEFAULT_BACKCHANNEL_MAX_WORDS = 3
_DEFAULT_SHORT_ANSWER_MAX_WORDS = 4

_ACKNOWLEDGEMENT_NORMALIZED = {
    "mhm",
    "mmhm",
    "mmhmm",
    "uhhuh",
    "uh huh",
    "yeah",
    "yep",
    "ok",
    "okay",
    "right",
    "sure",
    "got it",
    "i see",
    "aha",
    "hmm",
    "hm",
    "ja",
    "jo",
    "klar",
    "genau",
    "stimmt",
    "verstanden",
    "verstehe",
    "alles klar",
}
_AFFIRMATIVE_NORMALIZED = {
    "yes",
    "yeah",
    "yep",
    "sure",
    "ok",
    "okay",
    "correct",
    "right",
    "ja",
    "klar",
    "genau",
    "stimmt",
}
_NEGATIVE_NORMALIZED = {
    "no",
    "nope",
    "nah",
    "nicht",
    "nein",
}
_CONTINUATION_TOKENS = {
    "and",
    "or",
    "but",
    "because",
    "if",
    "when",
    "while",
    "so",
    "then",
    "also",
    "though",
    "although",
    "that",
    "which",
    "who",
    "where",
    "with",
    "for",
    "to",
    "from",
    "about",
    "as",
    "und",
    "oder",
    "aber",
    "weil",
    "wenn",
    "während",
    "dann",
    "also",
    "dass",
    "mit",
    "für",
    "zu",
    "von",
    "über",
}
_FILLER_TOKENS = {
    "uh",
    "um",
    "er",
    "erm",
    "ah",
    "eh",
    "huh",
    "äh",
    "ähm",
    "hm",
    "hmm",
    "mhm",
}
_TERMINAL_PUNCTUATION = (".", "!", "?", "。", "！", "？")
_CONTINUATION_PUNCTUATION = (",", ";", ":", "-", "–", "—")
_ELLIPSIS_SUFFIXES = ("...", "…")
_WHITESPACE_RE = re.compile(r"\s+")

_TURN_DECISION_TOOL_SCHEMA: dict[str, object] = {
    "type": "function",
    "name": "submit_turn_decision",
    "description": "Submit the current turn-boundary decision for the active user utterance.",
    "strict": True,
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["continue_listening", "end_turn"],
            },
            "label": {
                "type": "string",
                "enum": ["complete", "incomplete", "backchannel", "wait"],
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "reason": {
                "type": "string",
            },
            "transcript": {
                "type": "string",
            },
        },
        "required": ["decision", "label", "confidence", "reason", "transcript"],
    },
}


@dataclass(frozen=True, slots=True)
class TurnEvaluationCandidate:
    """Capture the bounded endpoint snapshot sent to the evaluator.

    The candidate contains the latest transcript plus endpoint metadata needed
    to decide whether the user is done speaking.
    """

    transcript: str
    event_type: str
    request_id: str | None = None
    observed_at_monotonic: float = 0.0
    is_final: bool = False
    speech_final: bool = False
    from_finalize: bool = False
    source_revision: int = 0
    last_word_end_seconds: float | None = None
    silence_after_last_word_ms: float | None = None
    semantic_end_score: float | None = None
    backchannel_score: float | None = None
    interruption_score: float | None = None

    @classmethod
    def from_endpoint_event(
        cls,
        event: StreamingSpeechEndpointEvent,
        *,
        transcript: str,
        source_revision: int = 0,
    ) -> "TurnEvaluationCandidate":
        """Build a bounded evaluation candidate from an endpoint event."""

        last_word_end_raw = _safe_getattr(event, "last_word_end", None)
        last_word_end_seconds: float | None = None
        if isinstance(last_word_end_raw, (int, float)):
            last_word_end_seconds = float(last_word_end_raw)

        silence_after_last_word_raw = _safe_getattr(event, "silence_after_last_word_ms", None)
        if silence_after_last_word_raw is None:
            silence_after_last_word_raw = _safe_getattr(event, "gap_ms", None)
        silence_after_last_word_ms: float | None = None
        if isinstance(silence_after_last_word_raw, (int, float)):
            silence_after_last_word_ms = float(silence_after_last_word_raw)

        semantic_end_score: float | None = _coerce_probability(
            _safe_getattr(event, "semantic_end_score", _safe_getattr(event, "turn_end_score", None)),
            default=-1.0,
        )
        if semantic_end_score is not None and semantic_end_score < 0.0:
            semantic_end_score = None

        backchannel_score: float | None = _coerce_probability(
            _safe_getattr(event, "backchannel_score", _safe_getattr(event, "turn_backchannel_score", None)),
            default=-1.0,
        )
        if backchannel_score is not None and backchannel_score < 0.0:
            backchannel_score = None

        interruption_score: float | None = _coerce_probability(
            _safe_getattr(event, "interruption_score", None),
            default=-1.0,
        )
        if interruption_score is not None and interruption_score < 0.0:
            interruption_score = None

        return cls(
            transcript=_coerce_text(transcript),
            event_type=_coerce_text(_safe_getattr(event, "event_type", "endpoint"), default="endpoint", max_chars=64) or "endpoint",
            request_id=_coerce_text(_safe_getattr(event, "request_id", ""), max_chars=128) or None,
            observed_at_monotonic=time.monotonic(),
            is_final=bool(_safe_getattr(event, "is_final", False)),
            speech_final=bool(_safe_getattr(event, "speech_final", False)),
            from_finalize=bool(_safe_getattr(event, "from_finalize", False)),
            source_revision=source_revision,
            last_word_end_seconds=last_word_end_seconds,
            silence_after_last_word_ms=silence_after_last_word_ms,
            semantic_end_score=semantic_end_score,
            backchannel_score=backchannel_score,
            interruption_score=interruption_score,
        )

    @property
    def normalized_transcript(self) -> str:
        return _normalize_turn_text(self.transcript)

    @property
    def event_type_normalized(self) -> str:
        return _coerce_text(self.event_type, default="endpoint", max_chars=64).strip().lower() or "endpoint"

    @property
    def strong_final_signal(self) -> bool:
        return bool(
            self.speech_final
            or self.from_finalize
            or self.is_final
            or self.event_type_normalized == "utterance_end"
        )


@dataclass(frozen=True, slots=True)
class TurnDecision:
    """Describe whether capture should continue or end for the current turn."""

    decision: TurnDecisionName
    label: TurnDecisionLabel
    confidence: float
    reason: str
    transcript: str

    @property
    def ends_turn(self) -> bool:
        return self.decision == "end_turn"


class TurnDecisionEvaluator(Protocol):
    """Protocol for components that evaluate turn-boundary candidates."""

    def evaluate(
        self,
        *,
        candidate: TurnEvaluationCandidate,
        conversation: ConversationLike | None = None,
    ) -> TurnDecision:
        """Return a bounded turn decision for the current candidate."""
        ...


@dataclass(frozen=True, slots=True)
class _SemanticTurnFeatures:
    transcript: str
    normalized: str
    words: tuple[str, ...]
    word_count: int
    terminal_punctuation: bool
    continuation_punctuation: bool
    ellipsis: bool
    trailing_continuation: bool
    trailing_filler: bool
    short_acknowledgement: bool
    short_yes_no_answer: bool
    short_answer: bool
    strong_final_signal: bool
    endpoint_event_is_utterance_end: bool
    recent_assistant_question: bool
    last_assistant_turn: str
    silence_after_last_word_ms: float | None
    last_word_end_seconds: float | None
    semantic_end_score: float | None
    backchannel_score: float | None
    interruption_score: float | None


def _conversation_signal(
    conversation: ConversationLike | None,
) -> tuple[bool, str]:
    if conversation is None:
        return False, ""
    compact_conversation = _compact_conversation(
        conversation,
        max_turns=2,
        max_item_chars=240,
        max_total_chars=480,
    )
    for role, content in reversed(compact_conversation):
        if role != "assistant":
            continue
        last_assistant_turn = _coerce_text(content, max_chars=240)
        return last_assistant_turn.rstrip().endswith("?"), last_assistant_turn
    return False, ""


def _transcript_words(text: str) -> tuple[str, ...]:
    normalized = _coerce_text(_normalize_turn_text(text), max_chars=_DEFAULT_MAX_TRANSCRIPT_CHARS)
    if not normalized:
        return ()
    return tuple(part for part in normalized.split() if part)


def _build_semantic_turn_features(
    *,
    config: TwinrConfig,
    candidate: TurnEvaluationCandidate,
    conversation: ConversationLike | None,
) -> _SemanticTurnFeatures:
    transcript = _coerce_text(candidate.transcript, max_chars=_DEFAULT_MAX_TRANSCRIPT_CHARS).strip()
    normalized = _normalize_turn_text(transcript)
    words = tuple(part for part in normalized.split() if part)
    word_count = len(words)
    stripped = transcript.rstrip()
    terminal_punctuation = stripped.endswith(_TERMINAL_PUNCTUATION)
    continuation_punctuation = stripped.endswith(_CONTINUATION_PUNCTUATION)
    ellipsis = stripped.endswith(_ELLIPSIS_SUFFIXES)
    last_token = words[-1] if words else ""
    recent_assistant_question, last_assistant_turn = _conversation_signal(conversation)
    backchannel_max_words = _config_int(
        config,
        "turn_controller_backchannel_max_words",
        _DEFAULT_BACKCHANNEL_MAX_WORDS,
        minimum=1,
        maximum=16,
    )
    short_answer_max_words = _config_int(
        config,
        "turn_controller_short_answer_max_words",
        _DEFAULT_SHORT_ANSWER_MAX_WORDS,
        minimum=1,
        maximum=16,
    )
    short_acknowledgement = bool(
        normalized
        and word_count <= backchannel_max_words
        and normalized in _ACKNOWLEDGEMENT_NORMALIZED
    )
    short_yes_no_answer = bool(
        normalized
        and word_count <= 2
        and normalized in (_AFFIRMATIVE_NORMALIZED | _NEGATIVE_NORMALIZED)
    )
    short_answer = bool(
        word_count > 0
        and word_count <= short_answer_max_words
        and not continuation_punctuation
        and not ellipsis
        and last_token not in _CONTINUATION_TOKENS
        and last_token not in _FILLER_TOKENS
    )
    trailing_continuation = bool(last_token and last_token in _CONTINUATION_TOKENS) or continuation_punctuation or ellipsis
    trailing_filler = bool(last_token and last_token in _FILLER_TOKENS)
    return _SemanticTurnFeatures(
        transcript=transcript,
        normalized=normalized,
        words=words,
        word_count=word_count,
        terminal_punctuation=terminal_punctuation,
        continuation_punctuation=continuation_punctuation,
        ellipsis=ellipsis,
        trailing_continuation=trailing_continuation,
        trailing_filler=trailing_filler,
        short_acknowledgement=short_acknowledgement,
        short_yes_no_answer=short_yes_no_answer,
        short_answer=short_answer,
        strong_final_signal=candidate.strong_final_signal,
        endpoint_event_is_utterance_end=candidate.event_type_normalized == "utterance_end",
        recent_assistant_question=recent_assistant_question,
        last_assistant_turn=last_assistant_turn,
        silence_after_last_word_ms=candidate.silence_after_last_word_ms,
        last_word_end_seconds=candidate.last_word_end_seconds,
        semantic_end_score=candidate.semantic_end_score,
        backchannel_score=candidate.backchannel_score,
        interruption_score=candidate.interruption_score,
    )


def _local_semantic_turn_decision(
    *,
    config: TwinrConfig,
    candidate: TurnEvaluationCandidate,
    conversation: ConversationLike | None,
    best_effort: bool,
    reason_prefix: str,
) -> TurnDecision | None:
    """Return a fast, transcript-aware decision when the signal is strong enough."""

    if not _config_bool(config, "turn_controller_local_semantic_enabled", True):
        return None

    features = _build_semantic_turn_features(config=config, candidate=candidate, conversation=conversation)
    transcript = features.transcript
    if not transcript:
        return None

    local_short_circuit_confidence = _config_float(
        config,
        "turn_controller_local_semantic_short_circuit_confidence",
        _DEFAULT_LOCAL_SHORT_CIRCUIT_CONFIDENCE,
        minimum=0.5,
        maximum=1.0,
    )
    local_fallback_confidence = _config_float(
        config,
        "turn_controller_local_semantic_fallback_confidence",
        _DEFAULT_LOCAL_FALLBACK_CONFIDENCE,
        minimum=0.0,
        maximum=1.0,
    )
    required_confidence = local_fallback_confidence if best_effort else local_short_circuit_confidence

    decision: TurnDecision | None = None

    if (
        features.backchannel_score is not None
        and features.backchannel_score >= 0.85
        and not features.recent_assistant_question
    ):
        decision = TurnDecision(
            decision="continue_listening",
            label="backchannel",
            confidence=max(0.85, features.backchannel_score),
            reason=f"{reason_prefix}:upstream_backchannel_score",
            transcript=transcript,
        )
    elif (
        features.semantic_end_score is not None
        and features.semantic_end_score >= 0.85
        and not features.trailing_continuation
        and not features.trailing_filler
    ):
        decision = TurnDecision(
            decision="end_turn",
            label="complete",
            confidence=max(0.85, features.semantic_end_score),
            reason=f"{reason_prefix}:upstream_semantic_end_score",
            transcript=transcript,
        )
    elif features.short_acknowledgement and not features.recent_assistant_question:
        confidence = 0.92 if not features.strong_final_signal else 0.85
        decision = TurnDecision(
            decision="continue_listening",
            label="backchannel",
            confidence=confidence,
            reason=f"{reason_prefix}:local_backchannel",
            transcript=transcript,
        )
    elif features.trailing_continuation or features.trailing_filler:
        confidence = 0.95 if not features.strong_final_signal else 0.78
        decision = TurnDecision(
            decision="continue_listening",
            label="incomplete",
            confidence=confidence,
            reason=f"{reason_prefix}:local_incomplete",
            transcript=transcript,
        )
    elif features.recent_assistant_question and features.short_yes_no_answer:
        confidence = 0.97 if features.strong_final_signal else 0.90
        decision = TurnDecision(
            decision="end_turn",
            label="complete",
            confidence=confidence,
            reason=f"{reason_prefix}:local_short_answer_to_question",
            transcript=transcript,
        )
    elif features.recent_assistant_question and features.short_answer and features.word_count <= 4:
        confidence = 0.94 if features.strong_final_signal else 0.86
        decision = TurnDecision(
            decision="end_turn",
            label="complete",
            confidence=confidence,
            reason=f"{reason_prefix}:local_question_answer",
            transcript=transcript,
        )
    elif features.endpoint_event_is_utterance_end and not features.trailing_continuation and not features.trailing_filler:
        decision = TurnDecision(
            decision="end_turn",
            label="complete",
            confidence=0.93,
            reason=f"{reason_prefix}:local_utterance_end",
            transcript=transcript,
        )
    elif features.strong_final_signal and not features.short_acknowledgement and not features.trailing_continuation and not features.trailing_filler:
        confidence = 0.92
        if features.terminal_punctuation:
            confidence = 0.96
        decision = TurnDecision(
            decision="end_turn",
            label="complete",
            confidence=confidence,
            reason=f"{reason_prefix}:local_finalized_complete",
            transcript=transcript,
        )
    elif features.terminal_punctuation and features.word_count >= 2 and not features.trailing_continuation and not features.trailing_filler:
        decision = TurnDecision(
            decision="end_turn",
            label="complete",
            confidence=0.84,
            reason=f"{reason_prefix}:local_terminal_punctuation",
            transcript=transcript,
        )
    elif best_effort and features.word_count == 1 and not features.strong_final_signal:
        decision = TurnDecision(
            decision="continue_listening",
            label="wait",
            confidence=0.74,
            reason=f"{reason_prefix}:local_wait",
            transcript=transcript,
        )

    if decision is None:
        return None
    if decision.confidence < required_confidence:
        return None
    return decision


class ToolCallingTurnDecisionEvaluator:
    """Ask a tool-calling provider for structured turn-boundary decisions.

    The evaluator first applies a fast local semantic detector, then optionally
    asks the provider for a structured decision, and finally falls back to the
    local detector when the provider fails, times out, or returns invalid
    output.
    """

    def __init__(
        self,
        *,
        config: TwinrConfig,
        provider: ToolCallingAgentProvider,
    ) -> None:
        self.config = config
        self.provider = provider
        self._state_lock = Lock()
        self._consecutive_failures = 0
        self._circuit_open_until_monotonic = 0.0
        self._provider_timeout_kwarg_name = detect_provider_timeout_kwarg(self.provider.start_turn_streaming)

    def evaluate(
        self,
        *,
        candidate: TurnEvaluationCandidate,
        conversation: ConversationLike | None = None,
    ) -> TurnDecision:
        """Evaluate whether the current user turn should end now."""

        fallback_transcript = _coerce_text(
            candidate.transcript,
            max_chars=_config_int(
                self.config,
                "turn_controller_max_transcript_chars",
                _DEFAULT_MAX_TRANSCRIPT_CHARS,
                minimum=64,
                maximum=32768,
            ),
        )

        if self._is_circuit_open():
            raise TurnControllerEvaluationError("turn_controller_circuit_open")

        compact_conversation = _compact_conversation(
            conversation,
            max_turns=_config_int(
                self.config,
                "turn_controller_context_turns",
                _DEFAULT_CONTEXT_TURNS,
                minimum=0,
                maximum=128,
            ),
            max_item_chars=_config_int(
                self.config,
                "turn_controller_max_conversation_item_chars",
                _DEFAULT_MAX_CONVERSATION_ITEM_CHARS,
                minimum=128,
                maximum=8192,
            ),
            max_total_chars=_config_int(
                self.config,
                "turn_controller_max_conversation_total_chars",
                _DEFAULT_MAX_CONVERSATION_TOTAL_CHARS,
                minimum=512,
                maximum=65536,
            ),
        )
        prompt = self._build_prompt(candidate, compact_conversation)
        provider_kwargs: dict[str, object] = {}
        if self._provider_timeout_kwarg_name is not None:
            provider_kwargs[self._provider_timeout_kwarg_name] = _config_float(
                self.config,
                "turn_controller_provider_timeout_seconds",
                _DEFAULT_PROVIDER_TIMEOUT_SECONDS,
                minimum=0.25,
                maximum=30.0,
            )

        try:
            response = cast(Any, self.provider.start_turn_streaming)(
                prompt,
                conversation=compact_conversation,
                instructions=load_turn_controller_instructions(self.config),
                tool_schemas=(_TURN_DECISION_TOOL_SCHEMA,),
                allow_web_search=False,
                **provider_kwargs,
            )
        except Exception as exc:
            self._note_failure()
            raise TurnControllerEvaluationError(
                f"turn_controller_error:{type(exc).__name__}"
            ) from exc

        parsed_decision = self._extract_provider_decision(
            response=response,
            fallback_transcript=fallback_transcript,
        )
        if parsed_decision is None:
            self._note_failure()
            raise TurnControllerEvaluationError("turn_controller_invalid_provider_output")

        self._note_success()
        return parsed_decision

    def _extract_provider_decision(
        self,
        *,
        response: object,
        fallback_transcript: str,
    ) -> TurnDecision | None:
        tool_calls_value = _safe_getattr(response, "tool_calls", ())
        tool_calls = tool_calls_value if isinstance(tool_calls_value, (list, tuple)) else ()
        for tool_call in tool_calls:
            tool_name = _coerce_text(_safe_getattr(tool_call, "name", ""), max_chars=64)
            if tool_name != "submit_turn_decision":
                continue
            raw_arguments = _safe_getattr(tool_call, "arguments", {})
            if isinstance(raw_arguments, dict):
                payload = raw_arguments
            else:
                bounded_arguments = _coerce_text(
                    raw_arguments,
                    max_chars=_config_int(
                        self.config,
                        "turn_controller_model_text_fallback_max_chars",
                        _DEFAULT_MODEL_TEXT_FALLBACK_MAX_CHARS,
                        minimum=512,
                        maximum=65536,
                    ),
                )
                payload = _extract_json_object(bounded_arguments) or {}
            return self._coerce_decision(payload, fallback_transcript=fallback_transcript)

        return None

    def _build_prompt(
        self,
        candidate: TurnEvaluationCandidate,
        conversation: tuple[tuple[str, str], ...],
    ) -> str:
        recent_assistant_question, last_assistant_turn = _conversation_signal(conversation)
        transcript = _coerce_text(
            candidate.transcript,
            max_chars=_config_int(
                self.config,
                "turn_controller_max_transcript_chars",
                _DEFAULT_MAX_TRANSCRIPT_CHARS,
                minimum=64,
                maximum=32768,
            ),
        )
        semantic_features = _build_semantic_turn_features(
            config=self.config,
            candidate=candidate,
            conversation=conversation,
        )
        payload = {
            "task": "Decide whether the active user turn should end now or continue listening.",
            "candidate": {
                "transcript": transcript,
                "transcript_chars": len(transcript),
                "transcript_words": semantic_features.word_count,
                "normalized_transcript": semantic_features.normalized or None,
                "event_type": _coerce_text(candidate.event_type, default="endpoint", max_chars=64) or "endpoint",
                "request_id": _coerce_text(candidate.request_id, max_chars=128) or None,
                "is_final": bool(candidate.is_final),
                "speech_final": bool(candidate.speech_final),
                "from_finalize": bool(candidate.from_finalize),
                "last_word_end_seconds": candidate.last_word_end_seconds,
                "silence_after_last_word_ms": candidate.silence_after_last_word_ms,
                "semantic_end_score": candidate.semantic_end_score,
                "backchannel_score": candidate.backchannel_score,
                "interruption_score": candidate.interruption_score,
            },
            "dialogue_context": {
                "recent_assistant_question": recent_assistant_question,
                "recent_assistant_turn": last_assistant_turn or None,
                "recent_turn_count": len(conversation),
            },
            "semantic_hints": {
                "terminal_punctuation": semantic_features.terminal_punctuation,
                "continuation_punctuation": semantic_features.continuation_punctuation,
                "ellipsis": semantic_features.ellipsis,
                "trailing_continuation": semantic_features.trailing_continuation,
                "trailing_filler": semantic_features.trailing_filler,
                "short_acknowledgement": semantic_features.short_acknowledgement,
                "short_yes_no_answer": semantic_features.short_yes_no_answer,
            },
            "policy_hints": {
                "backchannel_max_chars": _config_int(
                    self.config,
                    "turn_controller_backchannel_max_chars",
                    24,
                    minimum=1,
                    maximum=512,
                ),
            },
        }
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))

    def _coerce_decision(
        self,
        payload: dict[str, object],
        *,
        fallback_transcript: str,
    ) -> TurnDecision:
        raw_decision = _coerce_text(payload.get("decision", ""), max_chars=32).lower()
        decision: TurnDecisionName = "continue_listening"
        if raw_decision == "end_turn":
            decision = "end_turn"
        raw_label = _coerce_text(payload.get("label", ""), max_chars=32).lower()
        label: TurnDecisionLabel = "wait"
        if raw_label in {"complete", "incomplete", "backchannel", "wait"}:
            label = cast(TurnDecisionLabel, raw_label)
        elif decision == "end_turn":
            label = "complete"
        elif fallback_transcript.strip():
            label = "incomplete"
        confidence_value = _coerce_probability(payload.get("confidence", 0.0), default=0.0)
        reason = _coerce_text(
            payload.get("reason", ""),
            max_chars=_config_int(
                self.config,
                "turn_controller_max_reason_chars",
                _DEFAULT_MAX_REASON_CHARS,
                minimum=32,
                maximum=2048,
            ),
        ) or "unspecified"
        transcript = _coerce_text(
            payload.get("transcript", ""),
            max_chars=_config_int(
                self.config,
                "turn_controller_max_transcript_chars",
                _DEFAULT_MAX_TRANSCRIPT_CHARS,
                minimum=64,
                maximum=32768,
            ),
        ) or fallback_transcript.strip()
        return TurnDecision(
            decision=decision,
            label=label,
            confidence=confidence_value,
            reason=reason,
            transcript=transcript,
        )

    def _is_circuit_open(self) -> bool:
        with self._state_lock:
            return self._circuit_open_until_monotonic > time.monotonic()

    def _note_success(self) -> None:
        with self._state_lock:
            self._consecutive_failures = 0
            self._circuit_open_until_monotonic = 0.0

    def _note_failure(self) -> None:
        failure_threshold = _config_int(
            self.config,
            "turn_controller_circuit_breaker_failures",
            _DEFAULT_CIRCUIT_BREAKER_FAILURES,
            minimum=1,
            maximum=100,
        )
        cooldown_seconds = _config_float(
            self.config,
            "turn_controller_circuit_breaker_cooldown_seconds",
            _DEFAULT_CIRCUIT_BREAKER_COOLDOWN_SECONDS,
            minimum=1.0,
            maximum=3600.0,
        )
        with self._state_lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= failure_threshold:
                self._circuit_open_until_monotonic = time.monotonic() + cooldown_seconds


class StreamingTurnController:
    """Coordinate streaming endpoint events and stop-capture decisions."""

    def __init__(
        self,
        *,
        config: TwinrConfig,
        evaluator: TurnDecisionEvaluator,
        conversation_factory: Callable[[], ConversationLike | None],
        emit: Callable[[str], None] | None = None,
    ) -> None:
        self._config = config
        self._evaluator = evaluator
        self._conversation_factory = conversation_factory
        self._emit = emit
        self._lock = Lock()
        self._latest_partial = ""
        self._latest_partial_normalized = ""
        self._pending_candidate: TurnEvaluationCandidate | None = None
        self._last_decision: TurnDecision | None = None
        self._fatal_error: TurnControllerEvaluationError | None = None
        self._evaluation_inflight = False
        self._stop_requested = False
        self._closed = False
        self._state_revision = 0
        self._evaluator_worker_active = False

    def on_interim(self, transcript: str) -> None:
        """Record the latest partial transcript for the active capture."""

        cleaned = _coerce_text(
            transcript,
            max_chars=_config_int(
                self._config,
                "turn_controller_max_transcript_chars",
                _DEFAULT_MAX_TRANSCRIPT_CHARS,
                minimum=64,
                maximum=32768,
            ),
        )
        if not cleaned:
            return
        with self._lock:
            if self._closed or self._stop_requested:
                return
            self._state_revision += 1
            self._latest_partial = cleaned
            self._latest_partial_normalized = _normalize_turn_text(cleaned)

    def on_endpoint(self, event: StreamingSpeechEndpointEvent) -> None:
        """Process a streaming endpoint event and update stop state."""

        emit_messages: list[str] = []
        should_start_runner = False
        transcript = _coerce_text(
            _safe_getattr(event, "transcript", ""),
            max_chars=_config_int(
                self._config,
                "turn_controller_max_transcript_chars",
                _DEFAULT_MAX_TRANSCRIPT_CHARS,
                minimum=64,
                maximum=32768,
            ),
        )
        with self._lock:
            if self._closed or self._stop_requested:
                return
            previous_partial_normalized = self._latest_partial_normalized
            if not transcript:
                transcript = self._latest_partial
            transcript = _coerce_text(
                transcript,
                max_chars=_config_int(
                    self._config,
                    "turn_controller_max_transcript_chars",
                    _DEFAULT_MAX_TRANSCRIPT_CHARS,
                    minimum=64,
                    maximum=32768,
                ),
            )
            if not transcript:
                return
            next_revision = self._state_revision + 1
            self._state_revision = next_revision
            self._latest_partial = transcript
            self._latest_partial_normalized = _normalize_turn_text(transcript)

            if self._should_defer_bare_speech_final_locked(
                event,
                transcript=transcript,
                previous_partial_normalized=previous_partial_normalized,
            ):
                emit_messages.append(
                    self._format_emit(
                        "turn_controller_reason",
                        "speech_final_wait_for_finalize",
                    )
                )
            else:
                fast_decision = self._fast_endpoint_decision_locked(
                    event,
                    transcript,
                    previous_partial_normalized=previous_partial_normalized,
                )
                if fast_decision is not None:
                    self._last_decision = fast_decision
                    self._latest_partial = fast_decision.transcript.strip()
                    self._latest_partial_normalized = _normalize_turn_text(self._latest_partial)
                    self._stop_requested = True
                    emit_messages.extend(self._decision_emit_messages(fast_decision))
                else:
                    candidate = TurnEvaluationCandidate.from_endpoint_event(
                        event,
                        transcript=transcript,
                        source_revision=next_revision,
                    )
                    self._pending_candidate = candidate
                    emit_messages.append(self._format_emit("turn_controller_candidate", candidate.event_type))
                    if not self._evaluation_inflight:
                        self._evaluation_inflight = True
                        should_start_runner = True

        if should_start_runner:
            try:
                Thread(
                    target=self._run_pending_evaluations,
                    daemon=True,
                    name="twinr-turn-controller",
                ).start()
            except Exception as exc:
                error = TurnControllerEvaluationError(
                    f"turn_controller_runner_start_error:{type(exc).__name__}"
                )
                with self._lock:
                    self._evaluation_inflight = False
                    self._pending_candidate = None
                    self._fatal_error = error
                    self._stop_requested = True
                emit_messages.append(self._format_emit("turn_controller_error", str(error)))

        self._emit_many(emit_messages)

    def _should_defer_bare_speech_final_locked(
        self,
        event: StreamingSpeechEndpointEvent,
        *,
        transcript: str,
        previous_partial_normalized: str,
    ) -> bool:
        if not bool(_safe_getattr(event, "speech_final", False)):
            return False
        if previous_partial_normalized:
            return False
        if bool(_safe_getattr(event, "from_finalize", False)):
            return False
        if bool(_safe_getattr(event, "is_final", False)):
            return False
        return bool(_normalize_turn_text(transcript))

    def should_stop_capture(self) -> bool:
        with self._lock:
            return self._stop_requested or self._fatal_error is not None

    def latest_transcript(self) -> str:
        with self._lock:
            if (
                self._stop_requested
                and self._last_decision is not None
                and self._last_decision.transcript.strip()
            ):
                return self._last_decision.transcript.strip()
            return self._latest_partial.strip()

    def last_decision(self) -> TurnDecision | None:
        with self._lock:
            return self._last_decision

    def fatal_error(self) -> TurnControllerEvaluationError | None:
        with self._lock:
            return self._fatal_error

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._stop_requested = True
            self._pending_candidate = None

    def _run_pending_evaluations(self) -> None:
        while True:
            with self._lock:
                if self._closed or self._stop_requested:
                    self._evaluation_inflight = False
                    return
                candidate = self._pending_candidate
                self._pending_candidate = None
            if candidate is None:
                with self._lock:
                    self._evaluation_inflight = False
                return

            try:
                decision = self._evaluate_candidate_with_watchdog(candidate)
            except TurnControllerEvaluationError as exc:
                with self._lock:
                    self._fatal_error = exc
                    self._evaluation_inflight = False
                    self._stop_requested = True
                self._emit_many([self._format_emit("turn_controller_error", str(exc))])
                return
            emit_messages: list[str] = []
            should_return = False

            with self._lock:
                if self._closed or self._stop_requested:
                    self._evaluation_inflight = False
                    return
                if self._pending_candidate is not None:
                    continue
                if candidate.source_revision != self._state_revision:
                    self._evaluation_inflight = False
                    should_return = True
                    emit_messages.append(self._format_emit("turn_controller_reason", "stale_candidate_discarded"))
                else:
                    self._last_decision = decision
                    if decision.transcript.strip():
                        self._latest_partial = decision.transcript.strip()
                        self._latest_partial_normalized = _normalize_turn_text(self._latest_partial)
                    if decision.decision == "end_turn":
                        self._stop_requested = True
                    emit_messages.extend(self._decision_emit_messages(decision))

            self._emit_many(emit_messages)

            if should_return:
                return

    def _evaluate_candidate_with_watchdog(
        self,
        candidate: TurnEvaluationCandidate,
    ) -> TurnDecision:
        timeout_seconds = _config_float(
            self._config,
            "turn_controller_eval_timeout_seconds",
            _DEFAULT_EVALUATION_TIMEOUT_SECONDS,
            minimum=0.25,
            maximum=30.0,
        )

        with self._lock:
            if self._evaluator_worker_active:
                raise TurnControllerEvaluationError("turn_controller_worker_busy")
            self._evaluator_worker_active = True

        done = Event()
        decision_holder: list[TurnDecision] = []
        error_holder: list[BaseException] = []

        def _worker() -> None:
            try:
                decision_holder.append(
                    self._sanitize_decision(
                        self._evaluator.evaluate(
                            candidate=candidate,
                            conversation=self._conversation_factory(),
                        ),
                        fallback_transcript=candidate.transcript,
                    )
                )
            except Exception as exc:
                error_holder.append(exc)
            finally:
                with self._lock:
                    self._evaluator_worker_active = False
                done.set()

        try:
            Thread(
                target=_worker,
                daemon=True,
                name="twinr-turn-controller-evaluator",
            ).start()
        except Exception as exc:
            with self._lock:
                self._evaluator_worker_active = False
            raise TurnControllerEvaluationError(
                f"turn_controller_worker_start_error:{type(exc).__name__}"
            ) from exc

        if not done.wait(timeout_seconds):
            raise TurnControllerEvaluationError("turn_controller_timeout")

        if error_holder:
            raise TurnControllerEvaluationError(
                f"turn_controller_error:{type(error_holder[0]).__name__}"
            ) from error_holder[0]

        if not decision_holder:
            raise TurnControllerEvaluationError("turn_controller_no_result")

        return decision_holder[0]

    def _sanitize_decision(
        self,
        decision: TurnDecision,
        *,
        fallback_transcript: str,
    ) -> TurnDecision:
        safe_decision: TurnDecisionName = "end_turn" if decision.decision == "end_turn" else "continue_listening"
        safe_transcript = _coerce_text(
            decision.transcript,
            max_chars=_config_int(
                self._config,
                "turn_controller_max_transcript_chars",
                _DEFAULT_MAX_TRANSCRIPT_CHARS,
                minimum=64,
                maximum=32768,
            ),
        ) or _coerce_text(
            fallback_transcript,
            max_chars=_config_int(
                self._config,
                "turn_controller_max_transcript_chars",
                _DEFAULT_MAX_TRANSCRIPT_CHARS,
                minimum=64,
                maximum=32768,
            ),
        )
        safe_label: TurnDecisionLabel
        if decision.label in {"complete", "incomplete", "backchannel", "wait"}:
            safe_label = cast(TurnDecisionLabel, decision.label)
        else:
            safe_label = "complete" if safe_decision == "end_turn" else "wait"

        return TurnDecision(
            decision=safe_decision,
            label=safe_label,
            confidence=_coerce_probability(decision.confidence, default=0.0),
            reason=_coerce_text(
                decision.reason,
                default="unspecified",
                max_chars=_config_int(
                    self._config,
                    "turn_controller_max_reason_chars",
                    _DEFAULT_MAX_REASON_CHARS,
                    minimum=32,
                    maximum=2048,
                ),
            ) or "unspecified",
            transcript=safe_transcript,
        )

    def _decision_emit_messages(self, decision: TurnDecision) -> list[str]:
        return [
            self._format_emit("turn_controller_decision", decision.decision),
            self._format_emit("turn_controller_label", decision.label),
            self._format_emit("turn_controller_confidence", f"{_coerce_probability(decision.confidence, default=0.0):.2f}"),
            self._format_emit("turn_controller_reason", decision.reason),
        ]

    def _emit_many(self, messages: list[str]) -> None:
        for message in messages:
            self._safe_emit(message)

    def _safe_emit(self, message: str) -> None:
        emit = self._emit
        if emit is None:
            return
        try:
            emit(message)
        except Exception:
            return

    def _format_emit(self, key: str, value: object) -> str:
        return f"{key}={_sanitize_emit_value(value, max_chars=_config_int(self._config, 'turn_controller_emit_value_max_chars', _DEFAULT_EMIT_VALUE_MAX_CHARS, minimum=32, maximum=4096))}"

    def _fast_endpoint_decision_locked(
        self,
        event: StreamingSpeechEndpointEvent,
        transcript: str,
        *,
        previous_partial_normalized: str,
    ) -> TurnDecision | None:
        if not _config_bool(self._config, "turn_controller_fast_endpoint_enabled", False):
            return None
        cleaned = _coerce_text(
            transcript,
            max_chars=_config_int(
                self._config,
                "turn_controller_max_transcript_chars",
                _DEFAULT_MAX_TRANSCRIPT_CHARS,
                minimum=64,
                maximum=32768,
            ),
        )
        if len(cleaned) < _config_int(
            self._config,
            "turn_controller_fast_endpoint_min_chars",
            1,
            minimum=1,
            maximum=32768,
        ):
            return None

        event_type = _coerce_text(_safe_getattr(event, "event_type", ""), max_chars=64).lower()
        speech_final = bool(_safe_getattr(event, "speech_final", False))
        if speech_final and cleaned.strip():
            local_complete = _local_semantic_turn_decision(
                config=self._config,
                candidate=TurnEvaluationCandidate.from_endpoint_event(
                    event,
                    transcript=cleaned,
                    source_revision=0,
                ),
                conversation=None,
                best_effort=True,
                reason_prefix="speech_final_fast_path",
            )
            if local_complete is not None and local_complete.decision == "end_turn":
                return TurnDecision(
                    decision=local_complete.decision,
                    label=local_complete.label,
                    confidence=local_complete.confidence,
                    reason="speech_final_fast_path",
                    transcript=local_complete.transcript,
                )
            return TurnDecision(
                decision="end_turn",
                label="complete",
                confidence=1.0,
                reason="speech_final_fast_path",
                transcript=cleaned,
            )
        if (
            event_type == "utterance_end"
            and _config_bool(self._config, "deepgram_streaming_stop_on_utterance_end", False)
        ):
            normalized = _normalize_turn_text(cleaned)
            if normalized and previous_partial_normalized and normalized == previous_partial_normalized:
                return TurnDecision(
                    decision="end_turn",
                    label="complete",
                    confidence=_coerce_probability(
                        _safe_getattr(
                            self._config,
                            "turn_controller_fast_endpoint_min_confidence",
                            0.0,
                        ),
                        default=0.0,
                    ),
                    reason="utterance_end_fast_path",
                    transcript=cleaned,
                )
        return None


def _normalize_turn_text(text: str) -> str:
    """Preserve the legacy turn-controller normalizer import path."""

    return normalize_turn_text(text)
