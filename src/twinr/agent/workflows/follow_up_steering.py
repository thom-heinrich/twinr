# CHANGELOG: 2026-03-27
# BUG-1: Avoid remote personality/state reads when closure evaluation is disabled or
#        follow-up listening is disallowed for the current request source.
# BUG-2: Closure and steering failures now surface immediately instead of being
#        replaced by local substitute decisions.
# BUG-3: Telemetry side effects are now best-effort so emit/_record_event failures do not
#        crash the voice runtime after a user-visible answer has already been produced.
# SEC-1: Telemetry no longer emits raw matched topic titles by default; hashes/counts are
#        recorded unless config.follow_up_telemetry_include_raw_topics=True.
# IMP-1: Add a short version-aware steering-cue cache for low-latency Raspberry-Pi
#        deployments without re-reading remote state on every turn.
# IMP-2: Forward optional structured cue payloads and hybrid conversation signals to
#        evaluators that can consume richer 2026 turn-taking context.

"""Translate personality steering state into runtime follow-up decisions.

This helper keeps follow-up reopening logic out of the large workflow loop
classes. It loads the current authoritative turn-steering cues from the
structured personality layer, passes them into the closure evaluator as
machine-readable context, and applies the resulting matched topics back onto
the runtime follow-up decision.

The runtime is intentionally defensive only around telemetry/export side
effects: follow-up orchestration sits on the live voice path, so diagnostic
emission must not break a turn after the authoritative decision already
exists. Steering and closure decisions themselves stay single-lane and fail
closed when their required path cannot produce an answer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from inspect import Signature, signature
from math import isfinite
import re
from time import monotonic

try:
    from twinr.agent.base_agent.conversation.closure import (
        ConversationClosureDecision,
        ConversationClosureEvaluation,
        assistant_expects_immediate_reply,
    )
except ImportError:
    from twinr.agent.base_agent.conversation.closure import (
        ConversationClosureEvaluation,
        assistant_expects_immediate_reply,
    )

    @dataclass(frozen=True, slots=True)
    class ConversationClosureDecision:
        close_now: bool
        confidence: float
        reason: str
        follow_up_action: str | None = None
        matched_topics: tuple[str, ...] = ()

from twinr.agent.base_agent.conversation.follow_up_context import (
    build_follow_up_context_hint,
    clear_pending_conversation_follow_up_hint,
    follow_up_context_hint_trace_details,
    pending_conversation_follow_up_hint_trace_details,
    remember_pending_conversation_follow_up_hint,
)
from twinr.agent.personality.service import PersonalityContextService
try:
    from twinr.agent.personality.steering import (
        ConversationTurnSteeringCue,
        FollowUpSteeringDecision,
        resolve_follow_up_steering,
        serialize_turn_steering_cues,
    )
except ImportError:
    from twinr.agent.personality.steering import (
        ConversationTurnSteeringCue,
        FollowUpSteeringDecision,
        resolve_follow_up_steering,
    )

    def serialize_turn_steering_cues(
        cues: Sequence[ConversationTurnSteeringCue],
    ) -> tuple[Mapping[str, object], ...]:
        return ()

_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
_FOLLOW_UP_ACTION_CONTINUE = "continue"
_FOLLOW_UP_ACTION_END = "end"


@dataclass(frozen=True, slots=True)
class FollowUpRuntimeDecision:
    """Describe whether the runtime should reopen automatic follow-up listening.

    Attributes:
        force_close: Whether the runtime should stop automatic follow-up
            listening after the current answer.
        source: Which policy path caused the decision, such as ``closure`` or
            ``steering``.
        reason: Short bounded reason code for telemetry and tests.
        matched_topics: Topic titles that the closure evaluator matched against
            the active steering cues.
        selected_topic: Strongest matched topic after steering resolution.
        positive_engagement_action: Current bounded action for the selected
            topic if one was matched.
    """

    force_close: bool = False
    source: str = "none"
    reason: str = "none"
    matched_topics: tuple[str, ...] = ()
    selected_topic: str | None = None
    positive_engagement_action: str = "silent"


@dataclass(frozen=True, slots=True)
class _SteeringCueCacheEntry:
    """Store one bounded steering-cue snapshot for short-term reuse."""

    fingerprint: tuple[str | None, str | None]
    expires_at: float
    cues: tuple[ConversationTurnSteeringCue, ...]


class FollowUpSteeringRuntime:
    """Bridge personality steering state into runtime follow-up orchestration."""

    _DEFAULT_STEERING_CUE_CACHE_TTL_SECONDS = 2.0
    _DEFAULT_MAX_MATCHED_TOPICS = 2
    _DEFAULT_MAX_TOPIC_CHARS = 96
    _DEFAULT_MAX_SIGNAL_ITEMS = 8
    _DEFAULT_MAX_SIGNAL_KEY_CHARS = 48
    _DEFAULT_MAX_SIGNAL_VALUE_CHARS = 128
    _DEFAULT_MAX_TELEMETRY_TOPIC_HASHES = 2

    def __init__(
        self,
        loop,
        *,
        personality_context_service: PersonalityContextService | None = None,
    ) -> None:
        self._loop = loop
        self._personality_context_service = personality_context_service or PersonalityContextService()
        self._steering_cue_cache: _SteeringCueCacheEntry | None = None
        self._evaluator_signature_key: int | None = None
        self._evaluator_signature: Signature | None = None
        self._evaluator_accepts_var_keyword = False
        self._evaluator_supported_kwargs: frozenset[str] = frozenset()

    def load_turn_steering_cues(
        self,
        *,
        force_refresh: bool = False,
        allow_remote: bool = True,
    ) -> tuple[ConversationTurnSteeringCue, ...]:
        """Load the current bounded turn-steering cues from personality state.

        Args:
            force_refresh: Ignore a warm cache and reload from the personality
                service when remote access is allowed.
            allow_remote: When ``False``, only cached cues may be returned.
                This prevents avoidable remote-state reads on fast paths where
                closure evaluation will be skipped anyway.
        """

        fingerprint = self._remote_state_fingerprint()
        cache = self._steering_cue_cache
        now = monotonic()
        if (
            cache is not None
            and cache.fingerprint == fingerprint
            and (not force_refresh or not allow_remote)
            and (cache.expires_at > now or not allow_remote)
        ):
            return cache.cues
        if not allow_remote:
            return cache.cues if cache is not None else ()

        try:
            cues = self._personality_context_service.load_turn_steering_cues(
                config=self._loop.config,
                remote_state=self._remote_state(),
            )
        except Exception as exc:
            self._safe_emit(
                f"conversation_follow_up_steering_cues_failed={type(exc).__name__}"
            )
            raise

        normalized_cues = self._normalize_cues(cues)
        ttl_seconds = self._steering_cue_cache_ttl_seconds()
        self._steering_cue_cache = _SteeringCueCacheEntry(
            fingerprint=fingerprint,
            expires_at=now + ttl_seconds,
            cues=normalized_cues,
        )
        return normalized_cues

    def evaluate_closure(
        self,
        *,
        user_transcript: str,
        assistant_response: str,
        request_source: str,
        proactive_trigger: str | None,
        conversation_signals: Mapping[str, object] | None = None,
        force_refresh_steering_cues: bool = False,
    ) -> ConversationClosureEvaluation:
        """Run the closure evaluator with current machine-readable steering cues.

        The optional ``conversation_signals`` hook is a forward-compatible path
        for richer turn-taking context such as end-of-turn, barge-in, or
        speaker-selection signals from a real-time voice front-end.
        """

        heuristic_assistant_expects_reply = assistant_expects_immediate_reply(
            assistant_response
        )
        follow_up_context_hint = build_follow_up_context_hint(
            conversation=self._loop.runtime.conversation_context(),
            user_transcript=user_transcript,
            assistant_response=assistant_response,
        )
        self._safe_trace_event(
            "conversation_follow_up_context_hint_built",
            kind="workflow",
            details={
                "request_source": self._normalized_label(request_source, default="unknown"),
                "proactive_trigger": self._normalized_label(proactive_trigger, default="none"),
                **follow_up_context_hint_trace_details(
                    conversation=self._loop.runtime.conversation_context(),
                    user_transcript=user_transcript,
                    assistant_response=assistant_response,
                    summary=follow_up_context_hint,
                ),
            },
        )
        evaluator = getattr(self._loop, "conversation_closure_evaluator", None)
        if evaluator is None or not self._coerce_bool(
            getattr(self._loop.config, "conversation_closure_guard_enabled", False)
        ):
            return ConversationClosureEvaluation(
                assistant_expects_reply=heuristic_assistant_expects_reply,
                follow_up_context_hint=follow_up_context_hint,
                turn_steering_cues=self.load_turn_steering_cues(allow_remote=False)
            )
        if not self._loop._follow_up_allowed_for_source(initial_source=request_source):
            return ConversationClosureEvaluation(
                assistant_expects_reply=heuristic_assistant_expects_reply,
                follow_up_context_hint=follow_up_context_hint,
                turn_steering_cues=self.load_turn_steering_cues(allow_remote=False)
            )

        fast_path_evaluation = self._reply_expected_fast_path_evaluation(
            user_transcript=user_transcript,
            assistant_response=assistant_response,
            request_source=request_source,
            proactive_trigger=proactive_trigger,
            follow_up_context_hint=follow_up_context_hint,
            heuristic_assistant_expects_reply=heuristic_assistant_expects_reply,
        )
        if fast_path_evaluation is not None:
            return fast_path_evaluation

        steering_cues = self.load_turn_steering_cues(
            force_refresh=force_refresh_steering_cues
        )
        decision = self._evaluate_with_optional_context(
            evaluator=evaluator,
            user_transcript=user_transcript,
            assistant_response=assistant_response,
            request_source=request_source,
            proactive_trigger=proactive_trigger,
            steering_cues=steering_cues,
            conversation_signals=conversation_signals,
        )
        follow_up_action = self._coerce_follow_up_action(
            getattr(decision, "follow_up_action", None)
        )
        return ConversationClosureEvaluation(
            decision=decision,
            assistant_expects_reply=self._assistant_expects_reply_from_action(
                follow_up_action,
                default=heuristic_assistant_expects_reply,
            ),
            follow_up_action=follow_up_action,
            follow_up_context_hint=follow_up_context_hint,
            turn_steering_cues=steering_cues,
        )

    def _reply_expected_fast_path_evaluation(
        self,
        *,
        user_transcript: str,
        assistant_response: str,
        request_source: str,
        proactive_trigger: str | None,
        follow_up_context_hint: str | None,
        heuristic_assistant_expects_reply: bool,
    ) -> ConversationClosureEvaluation | None:
        """Return an immediate keep-open decision for obvious reply-expected turns."""

        if not heuristic_assistant_expects_reply:
            return None

        steering_cues = self.load_turn_steering_cues(allow_remote=False)
        matched_topics = self._match_topics_locally(
            user_transcript=user_transcript,
            assistant_response=assistant_response,
            steering_cues=steering_cues,
        )
        self._safe_trace_event(
            "conversation_closure_fast_continue",
            kind="branch",
            details={
                "request_source": self._normalized_label(request_source, default="unknown"),
                "proactive_trigger": self._normalized_label(proactive_trigger, default="none"),
                "matched_topic_count": len(matched_topics),
                "steering_cache_cues": len(steering_cues),
            },
        )
        return ConversationClosureEvaluation(
            decision=ConversationClosureDecision(
                close_now=False,
                confidence=0.94,
                reason="assistant_expects_reply",
                follow_up_action=_FOLLOW_UP_ACTION_CONTINUE,
                matched_topics=matched_topics,
            ),
            assistant_expects_reply=True,
            follow_up_action=_FOLLOW_UP_ACTION_CONTINUE,
            follow_up_context_hint=follow_up_context_hint,
            turn_steering_cues=steering_cues,
        )

    def apply_closure_evaluation(
        self,
        *,
        evaluation: ConversationClosureEvaluation,
        request_source: str,
        proactive_trigger: str | None,
    ) -> FollowUpRuntimeDecision:
        """Apply closure plus steering state to the runtime follow-up decision."""

        follow_up_action = self._coerce_follow_up_action(
            getattr(evaluation, "follow_up_action", None)
        )
        follow_up_context_hint = self._normalized_text(
            getattr(evaluation, "follow_up_context_hint", None),
            max_chars=768,
        ) or None

        def _update_follow_up_context(*, force_close: bool) -> None:
            runtime = getattr(self._loop, "runtime", None)
            action = "clear" if force_close else "store"
            if force_close:
                clear_pending_conversation_follow_up_hint(runtime)
            else:
                remember_pending_conversation_follow_up_hint(
                    runtime,
                    summary=follow_up_context_hint,
                )
            self._safe_trace_decision(
                "conversation_follow_up_context_hint_update",
                question="What should Twinr do with the pending immediate follow-up carryover hint after this answer?",
                selected={
                    "id": action,
                    "justification": (
                        "Automatic follow-up is being force-closed."
                        if force_close
                        else "Automatic follow-up remains open for the next turn."
                    ),
                    "expected_outcome": (
                        "No carryover hint will be available to the next turn."
                        if force_close
                        else "The next explicit follow-up turn can inject the carryover hint."
                    ),
                },
                options=[
                    {
                        "id": "store",
                        "summary": "Keep one bounded carryover hint for the next immediate follow-up turn.",
                    },
                    {
                        "id": "clear",
                        "summary": "Drop the carryover hint because the thread is closing or cannot safely continue.",
                    },
                ],
                context={
                    "request_source": self._normalized_label(request_source, default="unknown"),
                    "proactive_trigger": self._normalized_label(proactive_trigger, default="none"),
                    "follow_up_action": follow_up_action or "none",
                    "assistant_expects_reply": bool(getattr(evaluation, "assistant_expects_reply", False)),
                    "error_type": self._normalized_label(getattr(evaluation, "error_type", None), default="none"),
                    "hint": follow_up_context_hint_trace_details(
                        conversation=(),
                        user_transcript="",
                        assistant_response="",
                        summary=follow_up_context_hint,
                    )["summary"],
                    "runtime_hint": pending_conversation_follow_up_hint_trace_details(runtime),
                },
                confidence=self._bounded_probability(getattr(getattr(evaluation, "decision", None), "confidence", 0.0)),
                guardrails=[
                    "Keep redaction-by-default; never trace raw carryover text.",
                    "Store the hint only while the next immediate follow-up is still expected.",
                ],
            )

        if follow_up_action is not None:
            self._safe_emit(f"conversation_follow_up_action={follow_up_action}")

        if self._reply_gate_applies(request_source) and not self._coerce_bool(
            getattr(evaluation, "assistant_expects_reply", False)
        ):
            _update_follow_up_context(force_close=True)
            self._safe_emit("conversation_follow_up_reply_gate=assistant_no_reply_expected")
            self._safe_record_event(
                "conversation_follow_up_reply_gate_closed",
                "Twinr closed automatic follow-up because the delivered answer did not ask for immediate user input.",
                request_source=request_source,
                proactive_trigger=proactive_trigger,
            )
            return FollowUpRuntimeDecision(
                force_close=True,
                source="reply_gate",
                reason="assistant_no_reply_expected",
            )

        if evaluation.error_type:
            raise RuntimeError(f"conversation_closure_error:{evaluation.error_type}")
        decision = evaluation.decision
        if decision is None:
            _update_follow_up_context(force_close=False)
            return FollowUpRuntimeDecision()

        self._safe_emit_closure_decision(decision)
        steering = self._resolve_follow_up_steering(
            evaluation.turn_steering_cues,
            matched_topics=getattr(decision, "matched_topics", ()),
        )
        if self._closure_decision_passes_threshold(decision):
            _update_follow_up_context(force_close=True)
            self._safe_record_event(
                "conversation_closure_detected",
                "Twinr suppressed automatic follow-up listening because the exchange clearly ended for now.",
                request_source=request_source,
                proactive_trigger=proactive_trigger,
                confidence=self._bounded_probability(getattr(decision, "confidence", 0.0)),
                reason=self._normalized_label(getattr(decision, "reason", "none")),
                **self._telemetry_topic_fields(
                    matched_topics=getattr(decision, "matched_topics", ()),
                    selected_topic=steering.selected_topic,
                ),
            )
            return FollowUpRuntimeDecision(
                force_close=True,
                source="closure",
                reason=self._normalized_label(getattr(decision, "reason", "none")),
                matched_topics=self._coerce_topics(getattr(decision, "matched_topics", ())),
                selected_topic=self._normalized_text(steering.selected_topic) or None,
                positive_engagement_action=self._normalized_label(
                    steering.positive_engagement_action,
                    default="silent",
                ),
            )

        if steering.force_close:
            _update_follow_up_context(force_close=True)
            self._emit_steering_signal(steering)
            self._safe_record_event(
                "conversation_follow_up_steering_vetoed",
                "Twinr released automatic follow-up listening because the matched topic should be answered briefly and then left alone.",
                request_source=request_source,
                proactive_trigger=proactive_trigger,
                steering_reason=self._normalized_label(steering.reason, default="neutral"),
                attention_state=self._normalized_label(
                    getattr(steering, "attention_state", "background"),
                    default="background",
                ),
                **self._telemetry_topic_fields(
                    matched_topics=steering.matched_topics,
                    selected_topic=steering.selected_topic,
                ),
            )
            return FollowUpRuntimeDecision(
                force_close=True,
                source="steering",
                reason=self._normalized_label(steering.reason, default="neutral"),
                matched_topics=self._coerce_topics(steering.matched_topics),
                selected_topic=self._normalized_text(steering.selected_topic) or None,
                positive_engagement_action=self._normalized_label(
                    steering.positive_engagement_action,
                    default="silent",
                ),
            )

        if steering.keep_open:
            _update_follow_up_context(force_close=False)
            self._emit_steering_signal(steering)
            self._safe_record_event(
                "conversation_follow_up_steering_kept_open",
                "Twinr kept automatic follow-up listening open because the matched topic still forms a shared conversational thread.",
                request_source=request_source,
                proactive_trigger=proactive_trigger,
                steering_reason=self._normalized_label(steering.reason, default="neutral"),
                attention_state=self._normalized_label(
                    getattr(steering, "attention_state", "background"),
                    default="background",
                ),
                **self._telemetry_topic_fields(
                    matched_topics=steering.matched_topics,
                    selected_topic=steering.selected_topic,
                ),
            )
            return FollowUpRuntimeDecision(
                force_close=False,
                source="steering",
                reason=self._normalized_label(steering.reason, default="neutral"),
                matched_topics=self._coerce_topics(steering.matched_topics),
                selected_topic=self._normalized_text(steering.selected_topic) or None,
                positive_engagement_action=self._normalized_label(
                    steering.positive_engagement_action,
                    default="silent",
                ),
            )
        _update_follow_up_context(force_close=False)
        return FollowUpRuntimeDecision(
            force_close=False,
            source="none",
            reason="none",
            matched_topics=self._coerce_topics(steering.matched_topics),
            selected_topic=self._normalized_text(steering.selected_topic) or None,
            positive_engagement_action=self._normalized_label(
                steering.positive_engagement_action,
                default="silent",
            ),
        )

    def _coerce_follow_up_action(self, value: object) -> str | None:
        """Normalize the closure contract's follow-up action enum."""

        normalized = self._normalized_label(value, default="", max_chars=32)
        if normalized == _FOLLOW_UP_ACTION_CONTINUE:
            return _FOLLOW_UP_ACTION_CONTINUE
        if normalized == _FOLLOW_UP_ACTION_END:
            return _FOLLOW_UP_ACTION_END
        return None

    def _assistant_expects_reply_from_action(
        self,
        follow_up_action: str | None,
        *,
        default: bool,
    ) -> bool:
        """Prefer the explicit closure-model action over local heuristics."""

        if follow_up_action == _FOLLOW_UP_ACTION_CONTINUE:
            return True
        if follow_up_action == _FOLLOW_UP_ACTION_END:
            return False
        return default

    def _evaluate_with_optional_context(
        self,
        *,
        evaluator,
        user_transcript: str,
        assistant_response: str,
        request_source: str,
        proactive_trigger: str | None,
        steering_cues: Sequence[ConversationTurnSteeringCue],
        conversation_signals: Mapping[str, object] | None,
    ):
        """Call the evaluator and opportunistically pass richer structured context."""

        kwargs: dict[str, object] = {
            "user_transcript": user_transcript,
            "assistant_response": assistant_response,
            "request_source": request_source,
            "proactive_trigger": proactive_trigger,
            "conversation": self._loop.runtime.conversation_context(),
            "turn_steering_cues": tuple(steering_cues),
        }

        if steering_cues:
            serialized_cues = serialize_turn_steering_cues(steering_cues)
            if self._evaluator_supports_argument(evaluator, "serialized_turn_steering_cues"):
                kwargs["serialized_turn_steering_cues"] = serialized_cues
            if self._evaluator_supports_argument(evaluator, "turn_steering_topics"):
                kwargs["turn_steering_topics"] = serialized_cues
        if conversation_signals:
            bounded_signals = self._bounded_conversation_signals(conversation_signals)
            if bounded_signals:
                if self._evaluator_supports_argument(evaluator, "conversation_signals"):
                    kwargs["conversation_signals"] = bounded_signals
                if self._evaluator_supports_argument(evaluator, "turn_signals"):
                    kwargs["turn_signals"] = bounded_signals
                if self._evaluator_supports_argument(evaluator, "closure_signals"):
                    kwargs["closure_signals"] = bounded_signals
        return evaluator.evaluate(**kwargs)

    def _match_topics_locally(
        self,
        *,
        user_transcript: str,
        assistant_response: str,
        steering_cues: Sequence[ConversationTurnSteeringCue],
    ) -> tuple[str, ...]:
        """Conservatively match steering topics from the current exchange.

        Matching is intentionally strict: exact title-phrase matches win first,
        followed by "all significant title tokens present" matching. This
        avoids the broad adjacent-topic over-matching that the steering layer's
        ``match_summary`` is designed to prevent.
        """

        haystack = self._normalized_text(f"{user_transcript}\n{assistant_response}", max_chars=2048)
        haystack_casefold = haystack.casefold()
        haystack_tokens = set(self._keyword_tokens(haystack))
        if not haystack_casefold:
            return ()

        max_topics = self._max_matched_topics()
        seen: set[str] = set()
        matched: list[str] = []
        ranked_cues = sorted(
            self._normalize_cues(steering_cues),
            key=lambda cue: float(getattr(cue, "salience", 0.0)),
            reverse=True,
        )
        for cue in ranked_cues:
            title = self._normalized_text(getattr(cue, "title", None), max_chars=self._DEFAULT_MAX_TOPIC_CHARS)
            if not title:
                continue
            title_key = title.casefold()
            if title_key in seen:
                continue

            title_tokens = self._keyword_tokens(title)
            exact_phrase_match = self._contains_phrase(haystack_casefold, title_key)
            all_title_tokens_present = bool(title_tokens) and title_tokens.issubset(haystack_tokens)
            if not exact_phrase_match and not all_title_tokens_present:
                continue

            seen.add(title_key)
            matched.append(title)
            if len(matched) >= max_topics:
                break
        return tuple(matched)

    def _resolve_follow_up_steering(
        self,
        cues: Sequence[ConversationTurnSteeringCue],
        *,
        matched_topics: Sequence[str],
    ) -> FollowUpSteeringDecision:
        """Resolve steering from the authoritative cue set."""

        return resolve_follow_up_steering(
            tuple(self._normalize_cues(cues)),
            matched_topics=self._coerce_topics(matched_topics),
        )

    def _closure_decision_passes_threshold(self, decision) -> bool:
        """Return whether the closure evaluator clearly asked to close now."""

        if not self._coerce_bool(getattr(decision, "close_now", False)):
            return False
        min_confidence = max(
            0.0,
            min(
                1.0,
                self._bounded_probability(
                    getattr(
                        self._loop.config,
                        "conversation_closure_min_confidence",
                        0.0,
                    )
                ),
            ),
        )
        confidence = self._bounded_probability(getattr(decision, "confidence", 0.0))
        if confidence < min_confidence:
            self._safe_emit("conversation_closure_below_threshold=true")
            return False
        return True

    def _emit_steering_signal(self, steering: FollowUpSteeringDecision) -> None:
        """Emit one bounded steering trace line for operator debugging."""

        self._safe_emit(
            f"conversation_follow_up_steering={self._normalized_label(steering.reason, default='neutral')}"
        )

    def _safe_emit(self, message: str) -> None:
        """Emit a trace line without allowing telemetry failures to break turns."""

        emit = getattr(self._loop, "emit", None)
        if emit is None:
            return
        try:
            emit(self._normalized_text(message, max_chars=256))
        except Exception:
            return

    def _safe_trace_event(
        self,
        message: str,
        *,
        kind: str,
        details: Mapping[str, object] | None = None,
        level: str = "INFO",
    ) -> None:
        """Emit one structured workflow event when the loop exposes tracing."""

        tracer = getattr(self._loop, "_trace_event", None)
        if not callable(tracer):
            return
        try:
            tracer(
                message,
                kind=kind,
                details=dict(details or {}),
                level=level,
            )
        except Exception:
            return

    def _safe_trace_decision(
        self,
        message: str,
        *,
        question: str,
        selected: Mapping[str, object],
        options: Sequence[Mapping[str, object]],
        context: Mapping[str, object] | None = None,
        confidence: object | None = None,
        guardrails: Sequence[str] | None = None,
    ) -> None:
        """Emit one structured workflow decision when the loop supports it."""

        tracer = getattr(self._loop, "_trace_decision", None)
        if not callable(tracer):
            return
        try:
            tracer(
                message,
                question=question,
                selected=dict(selected),
                options=[dict(option) for option in options],
                context=dict(context or {}),
                confidence=confidence,
                guardrails=list(guardrails or ()),
            )
        except Exception:
            return

    def _safe_emit_closure_decision(self, decision) -> None:
        """Emit closure telemetry without letting instrumentation crash runtime."""

        emitter = getattr(self._loop, "_emit_closure_decision", None)
        if emitter is None:
            return
        try:
            emitter(decision)
        except Exception as exc:
            self._safe_emit(
                f"conversation_closure_emit_failed={type(exc).__name__}"
            )

    def _safe_record_event(self, event_name: str, message: str, **kwargs: object) -> None:
        """Record an event on a best-effort basis."""

        recorder = getattr(self._loop, "_record_event", None)
        if recorder is None:
            return
        try:
            recorder(event_name, message, **kwargs)
        except Exception as exc:
            self._safe_emit(
                f"conversation_follow_up_record_event_failed={type(exc).__name__}"
            )

    def _telemetry_topic_fields(
        self,
        *,
        matched_topics: Sequence[str],
        selected_topic: str | None,
    ) -> dict[str, object]:
        """Return privacy-aware topic fields for telemetry payloads."""

        topics = self._coerce_topics(matched_topics)
        selected = self._normalized_text(selected_topic, max_chars=self._DEFAULT_MAX_TOPIC_CHARS) or None
        topic_hashes = tuple(
            self._hash_for_telemetry(topic)
            for topic in topics[: self._DEFAULT_MAX_TELEMETRY_TOPIC_HASHES]
        )

        # BREAKING: raw topic titles are now opt-in for telemetry because the
        # matched titles can encode sensitive senior-user context.
        if self._coerce_bool(
            getattr(self._loop.config, "follow_up_telemetry_include_raw_topics", False)
        ):
            payload: dict[str, object] = {
                "matched_topics": topics or None,
                "matched_topic_hashes": topic_hashes or None,
                "matched_topic_count": len(topics),
            }
            if selected is not None:
                payload["selected_topic"] = selected
                payload["selected_topic_hash"] = self._hash_for_telemetry(selected)
            return payload

        payload = {
            "matched_topics": None,
            "matched_topic_hashes": topic_hashes or None,
            "matched_topic_count": len(topics),
        }
        if selected is not None:
            payload["selected_topic"] = None
            payload["selected_topic_hash"] = self._hash_for_telemetry(selected)
        return payload

    def _bounded_conversation_signals(
        self,
        signals: Mapping[str, object],
    ) -> dict[str, object]:
        """Bound optional hybrid turn-taking signals for evaluator forwarding."""

        bounded: dict[str, object] = {}
        for raw_key, raw_value in signals.items():
            key = self._normalized_label(
                raw_key,
                default="signal",
                max_chars=self._DEFAULT_MAX_SIGNAL_KEY_CHARS,
            )
            if key in bounded:
                continue
            if isinstance(raw_value, bool):
                bounded[key] = raw_value
            elif isinstance(raw_value, int):
                bounded[key] = raw_value
            elif isinstance(raw_value, float):
                if isfinite(raw_value):
                    bounded[key] = raw_value
            else:
                value = self._normalized_text(
                    raw_value,
                    max_chars=self._DEFAULT_MAX_SIGNAL_VALUE_CHARS,
                )
                if value:
                    bounded[key] = value
            if len(bounded) >= self._DEFAULT_MAX_SIGNAL_ITEMS:
                break
        return bounded

    def _evaluator_supports_argument(self, evaluator, argument_name: str) -> bool:
        """Return whether the evaluator can accept one optional keyword arg."""

        evaluate_fn = getattr(evaluator, "evaluate", None)
        if evaluate_fn is None:
            return False
        signature_target = getattr(type(evaluator), "evaluate", evaluate_fn)
        signature_key = id(signature_target)
        if signature_key != self._evaluator_signature_key:
            self._evaluator_signature_key = signature_key
            try:
                evaluator_signature = signature(signature_target)
            except (TypeError, ValueError):
                self._evaluator_signature = None
                self._evaluator_accepts_var_keyword = False
                self._evaluator_supported_kwargs = frozenset()
            else:
                self._evaluator_signature = evaluator_signature
                self._evaluator_accepts_var_keyword = any(
                    parameter.kind == parameter.VAR_KEYWORD
                    for parameter in evaluator_signature.parameters.values()
                )
                self._evaluator_supported_kwargs = frozenset(
                    evaluator_signature.parameters.keys()
                )
        return self._evaluator_accepts_var_keyword or argument_name in self._evaluator_supported_kwargs

    def _normalize_cues(
        self,
        cues: Sequence[ConversationTurnSteeringCue] | None,
    ) -> tuple[ConversationTurnSteeringCue, ...]:
        """Coerce arbitrary cue collections into a bounded tuple."""

        if not cues:
            return ()
        normalized: list[ConversationTurnSteeringCue] = []
        for cue in cues:
            if isinstance(cue, ConversationTurnSteeringCue):
                normalized.append(cue)
        return tuple(normalized)

    def _coerce_topics(self, value: Sequence[str] | None) -> tuple[str, ...]:
        """Normalize topic titles into a bounded unique tuple."""

        if not value:
            return ()
        max_topics = self._max_matched_topics()
        topics: list[str] = []
        seen: set[str] = set()
        for raw_topic in value:
            topic = self._normalized_text(
                raw_topic,
                max_chars=self._DEFAULT_MAX_TOPIC_CHARS,
            )
            if not topic:
                continue
            key = topic.casefold()
            if key in seen:
                continue
            seen.add(key)
            topics.append(topic)
            if len(topics) >= max_topics:
                break
        return tuple(topics)

    def _reply_gate_applies(self, request_source: str) -> bool:
        """Return whether the assistant-reply gate should control auto-rearm."""

        normalized = self._normalized_label(request_source, default="")
        return normalized in {"voice_activation", "proactive"}

    def _keyword_tokens(self, text: object | None) -> set[str]:
        """Extract lower-cased significant tokens for strict local matching."""

        normalized = self._normalized_text(text, max_chars=256).casefold()
        return {
            token
            for token in _WORD_RE.findall(normalized)
            if len(token) >= 3 and not token.isdigit()
        }

    def _contains_phrase(self, haystack_casefold: str, phrase_casefold: str) -> bool:
        """Return whether one normalized phrase appears with token boundaries."""

        if not haystack_casefold or not phrase_casefold:
            return False
        return (
            re.search(
                rf"(?<!\w){re.escape(phrase_casefold)}(?!\w)",
                haystack_casefold,
                flags=re.UNICODE,
            )
            is not None
        )

    def _coerce_bool(self, value: object) -> bool:
        """Coerce common scalar representations into a bool."""

        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().casefold()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off", ""}:
                return False
        return bool(value)

    def _bounded_probability(self, value: object) -> float:
        """Clamp probabilities into the closed interval [0.0, 1.0]."""

        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not isfinite(number):
            return 0.0
        return max(0.0, min(1.0, number))

    def _normalized_text(
        self,
        value: object | None,
        *,
        max_chars: int = 128,
    ) -> str:
        """Collapse arbitrary text into one bounded single-line string."""

        normalized = " ".join(str(value or "").split()).strip()
        if len(normalized) <= max(0, max_chars):
            return normalized
        limit = max(0, max_chars - 3)
        return normalized[:limit].rstrip() + "..."

    def _normalized_label(
        self,
        value: object | None,
        *,
        default: str = "none",
        max_chars: int = 64,
    ) -> str:
        """Normalize one bounded telemetry/test code."""

        raw = self._normalized_text(value, max_chars=max_chars).casefold()
        if not raw:
            return default
        normalized = "".join(
            character if (character.isalnum() or character in {"_", "-", "."}) else "_"
            for character in raw
        ).strip("_")
        return normalized or default

    def _hash_for_telemetry(self, value: object | None) -> str:
        """Return a short stable hash for privacy-preserving telemetry."""

        normalized = self._normalized_text(value, max_chars=self._DEFAULT_MAX_TOPIC_CHARS)
        if not normalized:
            return ""
        return sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def _max_matched_topics(self) -> int:
        """Return the bounded matched-topic limit for runtime decisions."""

        raw_value = getattr(self._loop.config, "follow_up_runtime_max_matched_topics", None)
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            value = self._DEFAULT_MAX_MATCHED_TOPICS
        return max(1, min(8, value))

    def _steering_cue_cache_ttl_seconds(self) -> float:
        """Return the bounded steering-cue cache TTL."""

        raw_value = getattr(
            self._loop.config,
            "follow_up_steering_cue_cache_ttl_seconds",
            self._DEFAULT_STEERING_CUE_CACHE_TTL_SECONDS,
        )
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            value = self._DEFAULT_STEERING_CUE_CACHE_TTL_SECONDS
        if not isfinite(value):
            value = self._DEFAULT_STEERING_CUE_CACHE_TTL_SECONDS
        return max(0.0, min(30.0, value))

    def _remote_state_fingerprint(self) -> tuple[str | None, str | None]:
        """Return a short cache fingerprint for the active remote-state handle."""

        remote_state = self._remote_state()
        if remote_state is None:
            return (None, None)
        state_identity = (
            self._normalized_text(self._read_mapping_or_attr(remote_state, "cache_key"), max_chars=96)
            or self._normalized_text(self._read_mapping_or_attr(remote_state, "instance_id"), max_chars=96)
            or hex(id(remote_state))
        )
        state_version = None
        for name in (
            "state_version",
            "version",
            "revision",
            "snapshot_id",
            "etag",
            "updated_at",
            "last_updated_at",
        ):
            value = self._read_mapping_or_attr(remote_state, name)
            if value is None:
                continue
            state_version = self._normalized_text(value, max_chars=96) or None
            if state_version is not None:
                break
        return (state_identity, state_version)

    def _read_mapping_or_attr(self, value: object, name: str) -> object | None:
        """Read one field from either a mapping-like or attribute-based object."""

        if isinstance(value, Mapping):
            return value.get(name)
        return getattr(value, name, None)

    def _remote_state(self):
        """Return the shared remote-state instance used by long-term memory."""

        long_term_memory = getattr(self._loop.runtime, "long_term_memory", None)
        prompt_context_store = getattr(long_term_memory, "prompt_context_store", None)
        memory_store = getattr(prompt_context_store, "memory_store", None)
        return getattr(memory_store, "remote_state", None)
