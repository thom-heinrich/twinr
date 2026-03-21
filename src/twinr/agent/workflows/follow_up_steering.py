"""Translate personality steering state into runtime follow-up decisions.

This helper keeps follow-up reopening logic out of the large workflow loop
classes. It loads the current authoritative turn-steering cues from the
structured personality layer, passes them into the closure evaluator as
machine-readable context, and applies the resulting matched topics back onto
the runtime follow-up decision.
"""

from __future__ import annotations

from dataclasses import dataclass

from twinr.agent.base_agent.conversation.closure import ConversationClosureEvaluation
from twinr.agent.personality.service import PersonalityContextService
from twinr.agent.personality.steering import (
    ConversationTurnSteeringCue,
    FollowUpSteeringDecision,
    resolve_follow_up_steering,
)


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


class FollowUpSteeringRuntime:
    """Bridge personality steering state into runtime follow-up orchestration."""

    def __init__(
        self,
        loop,
        *,
        personality_context_service: PersonalityContextService | None = None,
    ) -> None:
        self._loop = loop
        self._personality_context_service = personality_context_service or PersonalityContextService()

    def load_turn_steering_cues(self) -> tuple[ConversationTurnSteeringCue, ...]:
        """Load the current bounded turn-steering cues from remote personality state."""

        return self._personality_context_service.load_turn_steering_cues(
            config=self._loop.config,
            remote_state=self._remote_state(),
        )

    def evaluate_closure(
        self,
        *,
        user_transcript: str,
        assistant_response: str,
        request_source: str,
        proactive_trigger: str | None,
    ) -> ConversationClosureEvaluation:
        """Run the closure evaluator with current machine-readable steering cues."""

        steering_cues = self.load_turn_steering_cues()
        evaluator = getattr(self._loop, "conversation_closure_evaluator", None)
        if evaluator is None or not self._loop.config.conversation_closure_guard_enabled:
            return ConversationClosureEvaluation(turn_steering_cues=steering_cues)
        if not self._loop._follow_up_allowed_for_source(initial_source=request_source):
            return ConversationClosureEvaluation(turn_steering_cues=steering_cues)
        try:
            decision = evaluator.evaluate(
                user_transcript=user_transcript,
                assistant_response=assistant_response,
                request_source=request_source,
                proactive_trigger=proactive_trigger,
                conversation=self._loop.runtime.conversation_context(),
                turn_steering_cues=steering_cues,
            )
        except Exception as exc:
            return ConversationClosureEvaluation(
                error_type=type(exc).__name__,
                turn_steering_cues=steering_cues,
            )
        return ConversationClosureEvaluation(
            decision=decision,
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

        if evaluation.error_type:
            self._loop.emit(f"conversation_closure_fallback={evaluation.error_type}")
            return FollowUpRuntimeDecision()
        decision = evaluation.decision
        if decision is None:
            return FollowUpRuntimeDecision()

        self._loop._emit_closure_decision(decision)
        steering = resolve_follow_up_steering(
            evaluation.turn_steering_cues,
            matched_topics=decision.matched_topics,
        )
        if self._closure_decision_passes_threshold(decision):
            self._loop._record_event(
                "conversation_closure_detected",
                "Twinr suppressed automatic follow-up listening because the exchange clearly ended for now.",
                request_source=request_source,
                proactive_trigger=proactive_trigger,
                confidence=decision.confidence,
                reason=decision.reason,
                matched_topics=decision.matched_topics or None,
            )
            return FollowUpRuntimeDecision(
                force_close=True,
                source="closure",
                reason=decision.reason,
                matched_topics=decision.matched_topics,
                selected_topic=steering.selected_topic,
                positive_engagement_action=steering.positive_engagement_action,
            )

        if steering.force_close:
            self._emit_steering_signal(steering)
            self._loop._record_event(
                "conversation_follow_up_steering_vetoed",
                "Twinr released automatic follow-up listening because the matched topic should be answered briefly and then left alone.",
                request_source=request_source,
                proactive_trigger=proactive_trigger,
                matched_topics=steering.matched_topics or None,
                selected_topic=steering.selected_topic,
                steering_reason=steering.reason,
                attention_state=steering.attention_state,
            )
            return FollowUpRuntimeDecision(
                force_close=True,
                source="steering",
                reason=steering.reason,
                matched_topics=steering.matched_topics,
                selected_topic=steering.selected_topic,
                positive_engagement_action=steering.positive_engagement_action,
            )

        if steering.keep_open:
            self._emit_steering_signal(steering)
            return FollowUpRuntimeDecision(
                force_close=False,
                source="steering",
                reason=steering.reason,
                matched_topics=steering.matched_topics,
                selected_topic=steering.selected_topic,
                positive_engagement_action=steering.positive_engagement_action,
            )
        return FollowUpRuntimeDecision(
            force_close=False,
            source="none",
            reason="none",
            matched_topics=steering.matched_topics,
            selected_topic=steering.selected_topic,
            positive_engagement_action=steering.positive_engagement_action,
        )

    def _closure_decision_passes_threshold(self, decision) -> bool:
        """Return whether the closure evaluator clearly asked to close now."""

        if not bool(getattr(decision, "close_now", False)):
            return False
        min_confidence = max(0.0, min(1.0, float(self._loop.config.conversation_closure_min_confidence)))
        if float(getattr(decision, "confidence", 0.0)) < min_confidence:
            self._loop.emit("conversation_closure_below_threshold=true")
            return False
        return True

    def _emit_steering_signal(self, steering: FollowUpSteeringDecision) -> None:
        """Emit one bounded steering trace line for operator debugging."""

        self._loop.emit(f"conversation_follow_up_steering={steering.reason}")

    def _remote_state(self):
        """Return the shared remote-state instance used by long-term memory."""

        long_term_memory = getattr(self._loop.runtime, "long_term_memory", None)
        prompt_context_store = getattr(long_term_memory, "prompt_context_store", None)
        memory_store = getattr(prompt_context_store, "memory_store", None)
        return getattr(memory_store, "remote_state", None)
