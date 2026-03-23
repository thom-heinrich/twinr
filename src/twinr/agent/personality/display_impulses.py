"""Derive positive ambient display impulses from Twinr's companion state.

This module turns the same structured state that already drives prompt-time
mindshare and positive-engagement behavior into short, silent display impulses.
The goal is not to fabricate a new personality layer, but to let Twinr's
evolving interests, co-attention, and tone become visible outside of spoken
turns.

Selection and wording stay generic:

- no topic, place, or benchmark-specific hardcoding
- only positive or gently inviting impulses
- bounded actions derived from existing appetite / co-attention state
- light tone variation from learned verbosity and humor, not from ad-hoc
  randomness in the runtime loop
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import hashlib

from twinr.agent.personality.display_impulse_copy import build_ambient_display_impulse_copy
from twinr.agent.personality.intelligence.models import WorldInterestSignal
from twinr.agent.personality.models import PersonalitySnapshot
from twinr.agent.personality.positive_engagement import (
    PositiveEngagementTopicPolicy,
    derive_positive_engagement_policy,
)
from twinr.agent.personality.self_expression import (
    CompanionMindshareItem,
    build_mindshare_items,
)

_ACTION_PRIORITY = {
    "hint": 0.08,
    "brief_update": 0.15,
    "ask_one": 0.22,
    "invite_follow_up": 0.28,
}
_ATTENTION_PRIORITY = {
    "background": 0.00,
    "growing": 0.03,
    "forming": 0.08,
    "shared_thread": 0.12,
}


@dataclass(frozen=True, slots=True)
class AmbientDisplayImpulseCandidate:
    """Describe one short silent impulse Twinr may show in the display reserve.

    Attributes:
        topic_key: Stable dedupe/cooldown key for this impulse topic.
        title: Human-readable topic title.
        source: Structured source family such as ``continuity`` or ``place``.
        action: Bounded engagement action this impulse reflects.
        attention_state: Durable co-attention/background label.
        salience: Relative strength of the current topic.
        eyebrow: Small top label for the HDMI reserve card.
        headline: Main short line for the card.
        body: Secondary calm conversational line.
        symbol: Emoji symbol token suitable for display rendering.
        accent: Visual accent token suitable for display rendering.
        reason: Short auditable explanation for selection/debugging.
        candidate_family: Generic reserve-bus family used for planning mix,
            such as ``world``, ``memory_follow_up``, or ``social``.
        generation_context: Optional extra structured context for the bounded
            reserve-lane LLM rewrite step. This should stay generic and
            storage-safe, for example a topic summary or a memory-follow-up
            rationale, never a hidden prompt string.
    """

    topic_key: str
    title: str
    source: str
    action: str
    attention_state: str
    salience: float
    eyebrow: str
    headline: str
    body: str
    symbol: str
    accent: str
    reason: str
    candidate_family: str = "general"
    generation_context: Mapping[str, object] | None = None


def _normalized_text(value: object | None) -> str:
    """Collapse one arbitrary text value into a trimmed single line."""

    return " ".join(str(value or "").split()).strip()


def _topic_key(value: object | None) -> str:
    """Return one stable topic key for cooldown and dedupe behavior."""

    return _normalized_text(value).casefold()


def _stable_fraction(*parts: object) -> float:
    """Return one deterministic 0..1 fraction for bounded daily variation."""

    digest = hashlib.sha1(
        "::".join(_normalized_text(part) for part in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") / 4_294_967_295.0


def _truncate_text(value: object | None, *, max_len: int) -> str:
    """Return one bounded display-safe line."""

    compact = _normalized_text(value)
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "…"


def _candidate_score(
    item: CompanionMindshareItem,
    policy: PositiveEngagementTopicPolicy,
    *,
    local_now: datetime | None,
) -> float:
    """Return one bounded ranking score for display-candidate ordering."""

    jitter = _stable_fraction(
        item.title,
        item.source,
        policy.action,
        local_now.date() if local_now else "",
    ) * 0.06
    return (
        float(item.salience)
        + _ACTION_PRIORITY.get(policy.action, 0.0)
        + _ATTENTION_PRIORITY.get(policy.attention_state, 0.0)
        + jitter
    )


def _candidate_family_for_source(source: object | None) -> str:
    """Map one generic mindshare source onto a reserve-planning family."""

    normalized = _normalized_text(source).casefold()
    if normalized in {"continuity", "relationship"}:
        return "memory_thread"
    if normalized == "place":
        return "place"
    if normalized in {"situational_awareness", "regional_news", "local_news", "world"}:
        return "world"
    return "general"


def _candidate_for_item(
    item: CompanionMindshareItem,
    policy: PositiveEngagementTopicPolicy,
    *,
    local_now: datetime | None,
) -> AmbientDisplayImpulseCandidate | None:
    """Convert one prompt-facing mindshare item into a display impulse."""

    if policy.action == "silent":
        return None
    if _normalized_text(item.source).casefold() == "live_search":
        return None
    topic_key = _topic_key(item.title)
    if not topic_key:
        return None
    copy = build_ambient_display_impulse_copy(
        item,
        policy,
        local_now=local_now,
    )
    summary = _truncate_text(item.summary, max_len=160)
    return AmbientDisplayImpulseCandidate(
        topic_key=topic_key,
        title=item.title,
        source=item.source,
        action=policy.action,
        attention_state=policy.attention_state,
        salience=float(item.salience),
        eyebrow=copy.eyebrow,
        headline=copy.headline,
        body=copy.body,
        symbol=copy.symbol,
        accent=copy.accent,
        reason=policy.reason,
        candidate_family=_candidate_family_for_source(item.source),
        generation_context={
            "candidate_family": "mindshare",
            "display_anchor": item.title,
            "hook_hint": summary,
            "topic_summary": summary,
            "topic_title": item.title,
            "conversation_depth": item.appetite.depth,
            "follow_up": item.appetite.follow_up,
            "proactivity": item.appetite.proactivity,
            "ongoing_interest": item.appetite.interest,
            "engagement_state": item.appetite.state,
            "attention_state": policy.attention_state,
            "display_personality_goal": "show_twinr_voice",
            "display_goal": "open_positive_conversation",
        },
    )


def build_ambient_display_impulse_candidates(
    snapshot: PersonalitySnapshot | None,
    *,
    engagement_signals: Sequence[WorldInterestSignal] = (),
    local_now: datetime | None = None,
    max_items: int = 4,
) -> tuple[AmbientDisplayImpulseCandidate, ...]:
    """Build the current positive display-impulse candidates.

    The returned candidates are already filtered to positive or gently inviting
    actions. Topics in ``cooling`` or ``avoid`` states naturally disappear
    because the underlying positive-engagement policy resolves them to
    ``silent``.
    """

    limited_max = max(1, int(max_items))
    items = build_mindshare_items(
        snapshot,
        engagement_signals=engagement_signals,
        max_items=max(limited_max, 6),
    )
    candidates: list[tuple[float, AmbientDisplayImpulseCandidate]] = []
    for item in items:
        policy = derive_positive_engagement_policy(
            item,
            engagement_signals=engagement_signals,
        )
        candidate = _candidate_for_item(
            item,
            policy,
            local_now=local_now,
        )
        if candidate is None:
            continue
        score = _candidate_score(item, policy, local_now=local_now)
        candidates.append((score, candidate))
    ranked = sorted(
        candidates,
        key=lambda entry: (entry[0], entry[1].salience, entry[1].headline, entry[1].topic_key),
        reverse=True,
    )
    return tuple(candidate for _score, candidate in ranked[:limited_max])
