"""Follow-up continuation and closure helpers for the realtime workflow loop."""

from __future__ import annotations

from typing import Any

from twinr.agent.base_agent.conversation.closure import (
    ConversationClosureDecision,
    ConversationClosureEvaluation,
)


def follow_up_allowed_for_source(loop: Any, *, initial_source: str) -> bool:
    """Report whether the given turn source may reopen a follow-up listen."""

    if not loop.config.conversation_follow_up_enabled:
        return False
    if initial_source == "proactive":
        return bool(loop.config.conversation_follow_up_after_proactive_enabled)
    return True


def evaluate_follow_up_closure(
    loop: Any,
    *,
    user_transcript: str,
    assistant_response: str,
    request_source: str,
    proactive_trigger: str | None,
) -> ConversationClosureEvaluation:
    """Evaluate whether the just-finished turn should force-close follow-up."""

    return loop.follow_up_steering_runtime.evaluate_closure(
        user_transcript=user_transcript,
        assistant_response=assistant_response,
        request_source=request_source,
        proactive_trigger=proactive_trigger,
    )


def apply_follow_up_closure_evaluation(
    loop: Any,
    *,
    evaluation: ConversationClosureEvaluation,
    request_source: str,
    proactive_trigger: str | None,
) -> bool:
    """Apply one closure evaluation and return whether follow-up is vetoed."""

    decision = loop.follow_up_steering_runtime.apply_closure_evaluation(
        evaluation=evaluation,
        request_source=request_source,
        proactive_trigger=proactive_trigger,
    )
    return decision.force_close


def follow_up_vetoed_by_closure(
    loop: Any,
    *,
    user_transcript: str,
    assistant_response: str,
    request_source: str,
    proactive_trigger: str | None,
) -> bool:
    """Evaluate and apply closure steering for one finished answer."""

    evaluation = evaluate_follow_up_closure(
        loop,
        user_transcript=user_transcript,
        assistant_response=assistant_response,
        request_source=request_source,
        proactive_trigger=proactive_trigger,
    )
    return apply_follow_up_closure_evaluation(
        loop,
        evaluation=evaluation,
        request_source=request_source,
        proactive_trigger=proactive_trigger,
    )


def emit_closure_decision(loop: Any, decision: ConversationClosureDecision) -> None:
    """Emit the normalized closure decision fields for observability."""

    loop.emit(f"conversation_closure_close_now={str(decision.close_now).lower()}")
    loop.emit(f"conversation_closure_confidence={decision.confidence:.3f}")
    loop.emit(f"conversation_closure_reason={decision.reason}")
