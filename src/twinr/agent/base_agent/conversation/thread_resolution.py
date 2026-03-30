"""Resolve text turns against the immediate recent thread before routing.

Text surfaces can load broad hidden context that is useful overall but can
pollute short repairs such as "I meant the time". This module keeps text-turn
continuity grounded in the immediate user/assistant thread and can rewrite a
short repair into a standalone user request through a strict structured model
call when the recent turns already resolve the missing anchor.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from twinr.agent.base_agent.conversation.follow_up_context import recent_thread_carryover_system_message
from twinr.llm_json import request_structured_json_object


_MAX_RECENT_THREAD_TURNS = 4
_MAX_RECENT_TURN_CHARS = 220
_MAX_PROMPT_CHARS = 240
_REWRITE_MAX_OUTPUT_TOKENS = 120
_REWRITE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "resolution": {
            "type": "string",
            "enum": ["keep", "rewrite"],
        },
        "rewritten_user_text": {
            "type": "string",
            "maxLength": _MAX_PROMPT_CHARS,
        },
    },
    "required": ["resolution", "rewritten_user_text"],
}


@dataclass(frozen=True, slots=True)
class RecentThreadPromptResolution:
    """Hold one bounded prompt rewrite result for a text turn."""

    original_prompt: str
    effective_prompt: str
    resolution: str


def _normalize_text(value: object | None, *, limit: int) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = " ".join(text.split()).strip()
    if not text or limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= 1:
        return text[:limit]
    return text[: limit - 1].rstrip() + "..."


def _normalized_conversation(
    conversation: Sequence[tuple[str, str]] | None,
) -> tuple[tuple[str, str], ...]:
    normalized: list[tuple[str, str]] = []
    for item in tuple(conversation or ()):
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        role = _normalize_text(item[0], limit=16).lower()
        content = str(item[1] or "")
        if role not in {"system", "user", "assistant"} or not content.strip():
            continue
        normalized.append((role, content))
    return tuple(normalized)


def recent_thread_turns(
    conversation: Sequence[tuple[str, str]] | None,
    *,
    limit: int = _MAX_RECENT_THREAD_TURNS,
) -> tuple[tuple[str, str], ...]:
    """Return the newest bounded raw user/assistant turns from one conversation."""

    if limit <= 0:
        return ()
    turns: list[tuple[str, str]] = []
    for role, content in _normalized_conversation(conversation):
        if role not in {"user", "assistant"}:
            continue
        turns.append((role, content))
    return tuple(turns[-limit:])


def _leading_system_messages(
    conversation: Sequence[tuple[str, str]] | None,
) -> tuple[tuple[str, str], ...]:
    """Return the leading provider-authored system prelude from one context."""

    leading: list[tuple[str, str]] = []
    for role, content in _normalized_conversation(conversation):
        if role != "system":
            break
        leading.append((role, content))
    return tuple(leading)


def focus_recent_thread_conversation(
    conversation: Sequence[tuple[str, str]] | None,
    *,
    user_transcript: str,
) -> tuple[tuple[str, str], ...]:
    """Return a text-turn context focused on the immediate recent thread.

    Keep the provider-authored system prelude, drop older synthetic
    conversation-summary system turns, and keep only the recent raw
    user/assistant thread plus one carryover instruction.
    """

    leading = _leading_system_messages(conversation)
    recent_turns = recent_thread_turns(conversation)
    focused: list[tuple[str, str]] = list(leading)
    carryover = recent_thread_carryover_system_message(
        conversation=recent_turns,
        user_transcript=user_transcript,
    )
    if carryover:
        focused.append(("system", carryover))
    focused.extend(recent_turns)
    return tuple(focused)


def _rewrite_prompt_payload(
    *,
    recent_turns: Sequence[tuple[str, str]],
    user_transcript: str,
) -> str:
    rendered_turns = "\n".join(
        f"- {role}: {_normalize_text(content, limit=_MAX_RECENT_TURN_CHARS)}"
        for role, content in recent_turns
    )
    return (
        "Resolve the new user utterance against the immediate recent thread only.\n"
        "Use only the recent turns below. Ignore older memory summaries and broader history.\n"
        "If the new utterance is clearly a repair, clarification, or continuation of the same request, "
        "rewrite it into one short standalone user request in the same language.\n"
        "Carry forward explicit anchors that the immediate recent thread already established, such as place, "
        "date, timezone, or task scope.\n"
        "If the recent thread established a concrete place, date, or timezone for the same request, keep that "
        "anchor explicitly in the rewritten user request.\n"
        "If the immediate recent thread does not resolve the missing reference safely, return the original user "
        "utterance unchanged.\n"
        "Do not answer the user. Do not ask a follow-up question. Do not add facts not grounded in the recent turns.\n\n"
        f"Recent turns:\n{rendered_turns}\n\n"
        f"New user utterance:\n{_normalize_text(user_transcript, limit=_MAX_PROMPT_CHARS)}"
    )


def maybe_rewrite_prompt_against_recent_thread(
    backend: Any | None,
    *,
    conversation: Sequence[tuple[str, str]] | None,
    user_transcript: str,
) -> RecentThreadPromptResolution:
    """Return a bounded recent-thread rewrite when the backend can resolve one."""

    original_prompt = _normalize_text(user_transcript, limit=_MAX_PROMPT_CHARS)
    if not original_prompt:
        return RecentThreadPromptResolution(
            original_prompt="",
            effective_prompt="",
            resolution="keep",
        )
    recent_turns = recent_thread_turns(conversation)
    if not recent_turns or backend is None or not hasattr(backend, "config") or not hasattr(backend, "_client"):
        return RecentThreadPromptResolution(
            original_prompt=original_prompt,
            effective_prompt=original_prompt,
            resolution="keep",
        )
    try:
        payload = request_structured_json_object(
            backend,
            prompt=_rewrite_prompt_payload(
                recent_turns=recent_turns,
                user_transcript=original_prompt,
            ),
            instructions=(
                "Return one strict JSON object only. "
                "Do not emit markdown, code fences, or explanatory text."
            ),
            schema_name="twinr_recent_thread_prompt_rewrite_v1",
            schema=_REWRITE_SCHEMA,
            model=getattr(getattr(backend, "config", None), "default_model", None),
            reasoning_effort="low",
            max_output_tokens=_REWRITE_MAX_OUTPUT_TOKENS,
        )
    except Exception:
        return RecentThreadPromptResolution(
            original_prompt=original_prompt,
            effective_prompt=original_prompt,
            resolution="keep",
        )

    resolution = _normalize_text(payload.get("resolution"), limit=16).lower()
    rewritten_prompt = _normalize_text(payload.get("rewritten_user_text"), limit=_MAX_PROMPT_CHARS)
    if resolution != "rewrite" or not rewritten_prompt or rewritten_prompt == original_prompt:
        return RecentThreadPromptResolution(
            original_prompt=original_prompt,
            effective_prompt=original_prompt,
            resolution="keep",
        )
    return RecentThreadPromptResolution(
        original_prompt=original_prompt,
        effective_prompt=rewritten_prompt,
        resolution="rewrite",
    )


__all__ = [
    "RecentThreadPromptResolution",
    "focus_recent_thread_conversation",
    "maybe_rewrite_prompt_against_recent_thread",
    "recent_thread_turns",
]
