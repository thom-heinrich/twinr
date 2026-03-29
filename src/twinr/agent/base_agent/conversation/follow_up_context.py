"""Carry structured immediate follow-up context across one open thread.

Twinr's follow-up window intentionally keeps the microphone open when the
assistant still expects an immediate reply. The next user utterance is often a
short clarification or repair such as "I meant the time" that only makes sense
when the just-established anchors from the previous exchange stay visible.

This module stores one bounded carryover hint on the runtime between the
assistant's follow-up-opening reply and the next follow-up turn. The hint is
derived from the immediate conversation state, not from remote memory, and is
only injected when the current turn is explicitly a follow-up turn.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from typing import Any


_RUNTIME_HINT_ATTR = "_twinr_pending_conversation_follow_up_hint"
_RUNTIME_HINT_ACTIVE_ATTR = "_twinr_pending_conversation_follow_up_hint_active"
_MAX_HINT_SUMMARY_CHARS = 720
_MAX_TURN_CHARS = 160


@dataclass(frozen=True, slots=True)
class PendingConversationFollowUpHint:
    """Describe one bounded immediate follow-up carryover summary."""

    summary: str


def _normalize_text(value: object, *, limit: int) -> str:
    """Return one compact single-line text fragment."""

    text = str(value or "").replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = " ".join(text.split()).strip()
    if not text or limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= 1:
        return text[:limit]
    return text[: limit - 1].rstrip() + "…"


def _text_trace_details(value: object | None, *, limit: int) -> dict[str, object]:
    """Return privacy-safe bounded trace fields for one text payload."""

    normalized = _normalize_text(value, limit=limit)
    if not normalized:
        return {
            "present": False,
            "chars": 0,
            "words": 0,
            "sha256_12": "",
        }
    return {
        "present": True,
        "chars": len(normalized),
        "words": len(normalized.split()),
        "sha256_12": hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12],
    }


def _coerce_conversation_turns(
    conversation: Sequence[tuple[str, str]] | None,
    *,
    user_transcript: str,
    assistant_response: str,
) -> tuple[tuple[str, str], ...]:
    """Return the newest bounded user/assistant turns, including the current exchange."""

    normalized: list[tuple[str, str]] = []
    for item in tuple(conversation or ()):
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        role = _normalize_text(item[0], limit=16).lower()
        content = _normalize_text(item[1], limit=_MAX_TURN_CHARS)
        if role not in {"user", "assistant"} or not content:
            continue
        normalized.append((role, content))

    current_user = _normalize_text(user_transcript, limit=_MAX_TURN_CHARS)
    current_assistant = _normalize_text(assistant_response, limit=_MAX_TURN_CHARS)
    latest_user = next((content for role, content in reversed(normalized) if role == "user"), "")
    latest_assistant = next((content for role, content in reversed(normalized) if role == "assistant"), "")
    if current_user and latest_user != current_user:
        normalized.append(("user", current_user))
    if current_assistant and latest_assistant != current_assistant:
        normalized.append(("assistant", current_assistant))
    return tuple(normalized[-4:])


def build_follow_up_context_hint(
    *,
    conversation: Sequence[tuple[str, str]] | None,
    user_transcript: str,
    assistant_response: str,
) -> str | None:
    """Build one bounded carryover summary for the next immediate follow-up turn."""

    recent_turns = _coerce_conversation_turns(
        conversation,
        user_transcript=user_transcript,
        assistant_response=assistant_response,
    )
    if not recent_turns:
        return None
    rendered_turns = "; ".join(
        f"{role}={content}" for role, content in recent_turns
    )
    summary = _normalize_text(
        (
            "Immediate open thread from the latest exchange. "
            f"Recent turns: {rendered_turns}. "
            "Continue this thread unless the user clearly changes topic."
        ),
        limit=_MAX_HINT_SUMMARY_CHARS,
    )
    return summary or None


def follow_up_context_hint_trace_details(
    *,
    conversation: Sequence[tuple[str, str]] | None,
    user_transcript: str,
    assistant_response: str,
    summary: str | None,
) -> dict[str, object]:
    """Return privacy-safe observability fields for one follow-up carryover hint."""

    recent_turns = _coerce_conversation_turns(
        conversation,
        user_transcript=user_transcript,
        assistant_response=assistant_response,
    )
    rendered_turns = "; ".join(f"{role}={content}" for role, content in recent_turns)
    return {
        "recent_turn_count": len(recent_turns),
        "recent_roles": [role for role, _content in recent_turns],
        "recent_turns_sha256_12": _text_trace_details(rendered_turns, limit=_MAX_HINT_SUMMARY_CHARS)["sha256_12"],
        "user_transcript": _text_trace_details(user_transcript, limit=_MAX_TURN_CHARS),
        "assistant_response": _text_trace_details(assistant_response, limit=_MAX_TURN_CHARS),
        "summary": _text_trace_details(summary, limit=_MAX_HINT_SUMMARY_CHARS),
    }


def clear_pending_conversation_follow_up_hint(runtime: Any) -> None:
    """Remove any pending follow-up carryover hint from the runtime."""

    if runtime is None:
        return
    try:
        delattr(runtime, _RUNTIME_HINT_ATTR)
    except AttributeError:
        return


def remember_pending_conversation_follow_up_hint(
    runtime: Any,
    *,
    summary: str | None,
) -> PendingConversationFollowUpHint | None:
    """Store one bounded follow-up carryover hint on the runtime."""

    if runtime is None:
        return None
    normalized_summary = _normalize_text(summary, limit=_MAX_HINT_SUMMARY_CHARS)
    if not normalized_summary:
        clear_pending_conversation_follow_up_hint(runtime)
        return None
    hint = PendingConversationFollowUpHint(summary=normalized_summary)
    setattr(runtime, _RUNTIME_HINT_ATTR, hint)
    return hint


def peek_pending_conversation_follow_up_hint(
    runtime: Any,
) -> PendingConversationFollowUpHint | None:
    """Return the current pending carryover hint without consuming it."""

    hint = getattr(runtime, _RUNTIME_HINT_ATTR, None)
    return hint if isinstance(hint, PendingConversationFollowUpHint) else None


def pending_conversation_follow_up_hint_trace_details(runtime: Any) -> dict[str, object]:
    """Return privacy-safe trace details for the current pending follow-up hint."""

    hint = peek_pending_conversation_follow_up_hint(runtime)
    return {
        "active": bool(getattr(runtime, _RUNTIME_HINT_ACTIVE_ATTR, False)) if runtime is not None else False,
        "summary": _text_trace_details(
            None if hint is None else hint.summary,
            limit=_MAX_HINT_SUMMARY_CHARS,
        ),
    }


@contextmanager
def pending_conversation_follow_up_hint_scope(runtime: Any, *, active: bool):
    """Temporarily enable or disable follow-up-hint injection for one turn."""

    if runtime is None:
        yield
        return
    previous = bool(getattr(runtime, _RUNTIME_HINT_ACTIVE_ATTR, False))
    setattr(runtime, _RUNTIME_HINT_ACTIVE_ATTR, bool(active))
    try:
        yield
    finally:
        if previous:
            setattr(runtime, _RUNTIME_HINT_ACTIVE_ATTR, True)
        else:
            try:
                delattr(runtime, _RUNTIME_HINT_ACTIVE_ATTR)
            except AttributeError:
                pass


def pending_conversation_follow_up_system_message(runtime: Any) -> str | None:
    """Return the current follow-up carryover system message when active."""

    if runtime is None or not bool(getattr(runtime, _RUNTIME_HINT_ACTIVE_ATTR, False)):
        return None
    hint = peek_pending_conversation_follow_up_hint(runtime)
    if hint is None:
        return None
    return _normalize_text(
        (
            "Immediate follow-up carryover for this turn. "
            "Treat the next user utterance as continuation of the still-open exchange unless the user clearly changes topic. "
            "Preserve explicit anchors already established in that exchange, such as place, date, or task scope, unless the user overrides them. "
            f"{hint.summary}"
        ),
        limit=1200,
    ) or None


__all__ = [
    "PendingConversationFollowUpHint",
    "build_follow_up_context_hint",
    "clear_pending_conversation_follow_up_hint",
    "follow_up_context_hint_trace_details",
    "peek_pending_conversation_follow_up_hint",
    "pending_conversation_follow_up_hint_trace_details",
    "pending_conversation_follow_up_hint_scope",
    "pending_conversation_follow_up_system_message",
    "remember_pending_conversation_follow_up_hint",
]
