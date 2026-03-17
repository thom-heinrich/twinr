"""Generate user-facing recovery replies through LLM providers only.

This module centralizes the "something went wrong or timed out, but the user
still needs a spoken reply" path for the streaming runtime. It intentionally
does not contain any canned spoken fallback strings. Every returned text must
originate from an LLM provider.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

from twinr.agent.base_agent.contracts import (
    ConversationLike,
    FirstWordProvider,
    SupervisorDecisionProvider,
)
from twinr.agent.base_agent.prompting.personality import merge_instructions


logger = logging.getLogger(__name__)

_RECOVERY_SUPERVISOR_INSTRUCTIONS = (
    "The previous runtime path for this same user turn failed or timed out before Twinr finished the spoken answer. "
    "Recover now with action direct and a non-empty spoken_reply. "
    "Use the actual user request and the provided context to answer as helpfully as you can right now. "
    "Do not choose handoff. "
    "Do not choose end_conversation. "
    "Do not leave spoken_reply empty. "
    "Do not use a generic stock phrase, generic apology, or tell the user to try again. "
    "Do not mention internal tools, workers, prompts, routing, or hidden context. "
    "If the full underlying task result is still unavailable, give one short, natural, request-grounded progress reply about this exact request instead of a template."
)

_RECOVERY_SUPERVISOR_RETRY_INSTRUCTIONS = (
    "You must obey this recovery contract strictly. "
    "Return action direct and a non-empty spoken_reply. "
    "No handoff. "
    "No end_conversation. "
    "No generic apology. "
    "No 'please try again'."
)

_RECOVERY_FIRST_WORD_INSTRUCTIONS = (
    "This is the final recovery reply for the current user turn. "
    "Return mode direct and one short natural spoken_text that is specifically about the current user request. "
    "Do not use a generic apology, a stock phrase, or 'please try again'."
)


@dataclass(frozen=True, slots=True)
class LLMRecoveryReply:
    """Capture one LLM-generated recovery reply and its provider metadata."""

    text: str
    response_id: str | None = None
    request_id: str | None = None
    model: str | None = None
    token_usage: object | None = None


class LLMRecoveryResponder:
    """Resolve a short direct recovery reply through existing LLM lanes."""

    def __init__(
        self,
        *,
        supervisor_decision_provider: SupervisorDecisionProvider | None,
        first_word_provider: FirstWordProvider | None,
        supervisor_instructions: str | None,
    ) -> None:
        self._supervisor_decision_provider = supervisor_decision_provider
        self._first_word_provider = first_word_provider
        self._supervisor_instructions = supervisor_instructions

    def recover(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None,
        instructions: str | None,
        failure_reason: str,
    ) -> LLMRecoveryReply:
        """Return one LLM-generated direct reply for a failed runtime path."""

        supervisor_reply = self._recover_via_supervisor_decision(
            prompt,
            conversation=conversation,
            instructions=instructions,
            failure_reason=failure_reason,
        )
        if supervisor_reply is not None:
            return supervisor_reply

        first_word_reply = self._recover_via_first_word(
            prompt,
            conversation=conversation,
            instructions=instructions,
            failure_reason=failure_reason,
        )
        if first_word_reply is not None:
            return first_word_reply

        raise RuntimeError(
            f"No LLM recovery reply available for failure_reason={failure_reason!r}."
        )

    def _recover_via_supervisor_decision(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None,
        instructions: str | None,
        failure_reason: str,
    ) -> LLMRecoveryReply | None:
        provider = self._supervisor_decision_provider
        if provider is None:
            return None
        instruction_variants = (
            merge_instructions(
                self._supervisor_instructions,
                instructions,
                _RECOVERY_SUPERVISOR_INSTRUCTIONS,
                f"Runtime recovery reason: {failure_reason}.",
            ),
            merge_instructions(
                self._supervisor_instructions,
                instructions,
                _RECOVERY_SUPERVISOR_INSTRUCTIONS,
                _RECOVERY_SUPERVISOR_RETRY_INSTRUCTIONS,
                f"Runtime recovery reason: {failure_reason}.",
            ),
        )
        for attempt, merged_instructions in enumerate(instruction_variants, start=1):
            try:
                decision = provider.decide(
                    prompt,
                    conversation=conversation,
                    instructions=merged_instructions,
                )
            except Exception:
                logger.exception(
                    "Supervisor recovery decision failed on attempt %s for %s.",
                    attempt,
                    failure_reason,
                )
                continue
            action = str(getattr(decision, "action", "") or "").strip().lower()
            reply = str(getattr(decision, "spoken_reply", "") or "").strip()
            if action == "direct" and reply:
                return LLMRecoveryReply(
                    text=reply,
                    response_id=getattr(decision, "response_id", None),
                    request_id=getattr(decision, "request_id", None),
                    model=getattr(decision, "model", None),
                    token_usage=getattr(decision, "token_usage", None),
                )
            logger.warning(
                "Supervisor recovery attempt %s returned action=%r reply_len=%s for %s.",
                attempt,
                action,
                len(reply),
                failure_reason,
            )
        return None

    def _recover_via_first_word(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None,
        instructions: str | None,
        failure_reason: str,
    ) -> LLMRecoveryReply | None:
        provider = self._first_word_provider
        if provider is None:
            return None
        try:
            reply = provider.reply(
                prompt,
                conversation=conversation,
                instructions=merge_instructions(
                    instructions,
                    _RECOVERY_FIRST_WORD_INSTRUCTIONS,
                    f"Runtime recovery reason: {failure_reason}.",
                ),
            )
        except Exception:
            logger.exception("First-word recovery reply failed for %s.", failure_reason)
            return None
        text = str(getattr(reply, "spoken_text", "") or "").strip()
        if not text:
            logger.warning("First-word recovery reply was empty for %s.", failure_reason)
            return None
        return LLMRecoveryReply(
            text=text,
            response_id=getattr(reply, "response_id", None),
            request_id=getattr(reply, "request_id", None),
            model=getattr(reply, "model", None),
            token_usage=getattr(reply, "token_usage", None),
        )
