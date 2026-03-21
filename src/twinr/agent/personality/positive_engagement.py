"""Derive explicit positive engagement actions from durable companion state.

This module turns prompt-time mindshare plus persisted engagement/co-attention
state into a small set of bounded actions that Twinr may use to foster
welcomed interaction:

- ``silent``: keep the topic in background awareness unless the user clearly
  asks for it
- ``hint``: allow one light mention when the current exchange naturally opens
  the door
- ``brief_update``: allow one short concrete update in open conversation
- ``ask_one``: allow one calm engaging question when the topic is already
  becoming a shared thread
- ``invite_follow_up``: allow a brief update plus a low-pressure invitation if
  the user wants to keep going

The policy must stay generic. It may depend on structured appetite,
co-attention, and durable user engagement, but never on hardcoded topic names
or benchmark-shaped examples.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from twinr.agent.personality.intelligence.models import WorldInterestSignal
from twinr.agent.personality.models import PersonalitySnapshot
from twinr.agent.personality.self_expression import (
    CompanionMindshareItem,
    build_mindshare_items,
)


@dataclass(frozen=True, slots=True)
class PositiveEngagementTopicPolicy:
    """Describe how Twinr may positively surface one current topic.

    Attributes:
        title: Human-readable topic title the policy applies to.
        salience: Relative importance within the current turn.
        attention_state: Coarser durable state such as ``shared_thread`` or
            ``cooling``.
        action: Bounded engagement action for the current turn. One of
            ``silent``, ``hint``, ``brief_update``, ``ask_one``, or
            ``invite_follow_up``.
        reason: Short auditable summary of why the action was selected.
    """

    title: str
    salience: float
    attention_state: str = "background"
    action: str = "silent"
    reason: str = "background_observe"


def _normalized_text(value: object | None) -> str:
    """Collapse arbitrary text into one trimmed single-line string."""

    return " ".join(str(value or "").split()).strip()


def _interest_key(value: object | None) -> str:
    """Normalize one topic label into a stable exact-match key."""

    return _normalized_text(value).casefold()


def _matching_signal(
    item: CompanionMindshareItem,
    *,
    engagement_signals: Sequence[WorldInterestSignal],
) -> WorldInterestSignal | None:
    """Return the strongest exact-match durable signal for one mindshare item."""

    item_key = _interest_key(item.title)
    if not item_key:
        return None
    ranked = sorted(
        (
            signal
            for signal in engagement_signals
            if _interest_key(signal.topic) == item_key
        ),
        key=lambda signal: (
            signal.co_attention_state == "shared_thread",
            signal.co_attention_state == "forming",
            signal.ongoing_interest == "active",
            signal.engagement_state == "resonant",
            signal.engagement_state == "warm",
            signal.engagement_score,
            signal.salience,
            signal.updated_at or "",
        ),
        reverse=True,
    )
    return next(iter(ranked), None)


_FOLLOW_UP_RANKS = {
    "do_not_push": 0,
    "wait_for_user_pull": 1,
    "one_gentle_follow_up": 2,
    "okay_to_explore": 3,
}
_PROACTIVITY_RANKS = {
    "do_not_volunteer": 0,
    "only_if_clearly_relevant": 1,
    "light_offer_if_open": 2,
    "brief_offer_if_open": 3,
}
_INTEREST_RANKS = {
    "peripheral": 0,
    "growing": 1,
    "active": 2,
}


def _attention_state(
    signal: WorldInterestSignal | None,
    *,
    appetite_state: str,
    interest: str,
) -> str:
    """Return one generic attention-state label for the policy output."""

    if appetite_state == "avoid":
        return "avoid"
    if appetite_state == "cooling":
        return "cooling"
    co_attention_state = _normalized_text(getattr(signal, "co_attention_state", None)).casefold()
    if co_attention_state == "shared_thread":
        return "shared_thread"
    if co_attention_state == "forming":
        return "forming"
    if interest in {"active", "growing"}:
        return "growing"
    return "background"


def derive_positive_engagement_policy(
    item: CompanionMindshareItem,
    *,
    engagement_signals: Sequence[WorldInterestSignal] = (),
) -> PositiveEngagementTopicPolicy:
    """Derive one bounded positive-engagement action for a surfaced topic.

    The policy is intentionally conservative: it should help Twinr create
    welcomed future interaction, not maximize immediate time-on-device. Topics
    only move into stronger actions when durable engagement, ongoing interest,
    and co-attention all point in the same direction.
    """

    appetite = item.appetite
    signal = _matching_signal(item, engagement_signals=engagement_signals)
    attention_state = _attention_state(
        signal,
        appetite_state=appetite.state,
        interest=appetite.interest,
    )
    follow_up_rank = _FOLLOW_UP_RANKS.get(appetite.follow_up, 1)
    proactivity_rank = _PROACTIVITY_RANKS.get(appetite.proactivity, 1)
    interest_rank = _INTEREST_RANKS.get(appetite.interest, 0)

    if appetite.state == "avoid" or appetite.proactivity == "do_not_volunteer":
        return PositiveEngagementTopicPolicy(
            title=item.title,
            salience=item.salience,
            attention_state=attention_state,
            action="silent",
            reason="respect_boundary",
        )
    if appetite.state == "cooling":
        return PositiveEngagementTopicPolicy(
            title=item.title,
            salience=item.salience,
            attention_state=attention_state,
            action="silent",
            reason="cooling_back_off",
        )
    if attention_state == "shared_thread":
        if follow_up_rank >= 3 and proactivity_rank >= 3 and interest_rank >= 2:
            return PositiveEngagementTopicPolicy(
                title=item.title,
                salience=item.salience,
                attention_state=attention_state,
                action="invite_follow_up",
                reason="shared_thread_invite",
            )
        if follow_up_rank >= 2 and interest_rank >= 1:
            return PositiveEngagementTopicPolicy(
                title=item.title,
                salience=item.salience,
                attention_state=attention_state,
                action="ask_one",
                reason="shared_thread_ask_one",
            )
        if proactivity_rank >= 2:
            return PositiveEngagementTopicPolicy(
                title=item.title,
                salience=item.salience,
                attention_state=attention_state,
                action="brief_update",
                reason="shared_thread_brief_update",
            )
        return PositiveEngagementTopicPolicy(
            title=item.title,
            salience=item.salience,
            attention_state=attention_state,
            action="hint",
            reason="shared_thread_hint",
        )
    if attention_state == "forming":
        if follow_up_rank >= 2 and interest_rank >= 1:
            return PositiveEngagementTopicPolicy(
                title=item.title,
                salience=item.salience,
                attention_state=attention_state,
                action="ask_one",
                reason="forming_thread_ask_one",
            )
        if proactivity_rank >= 2:
            return PositiveEngagementTopicPolicy(
                title=item.title,
                salience=item.salience,
                attention_state=attention_state,
                action="brief_update",
                reason="forming_thread_brief_update",
            )
        return PositiveEngagementTopicPolicy(
            title=item.title,
            salience=item.salience,
            attention_state=attention_state,
            action="hint",
            reason="forming_thread_hint",
        )
    if proactivity_rank >= 3 and interest_rank >= 1:
        return PositiveEngagementTopicPolicy(
            title=item.title,
            salience=item.salience,
            attention_state=attention_state,
            action="brief_update",
            reason="high_interest_brief_update",
        )
    if proactivity_rank >= 2 and interest_rank >= 1:
        return PositiveEngagementTopicPolicy(
            title=item.title,
            salience=item.salience,
            attention_state=attention_state,
            action="hint",
            reason="growing_interest_hint",
        )
    if interest_rank >= 1:
        return PositiveEngagementTopicPolicy(
            title=item.title,
            salience=item.salience,
            attention_state=attention_state,
            action="hint",
            reason="light_interest_hint",
        )
    return PositiveEngagementTopicPolicy(
        title=item.title,
        salience=item.salience,
        attention_state=attention_state,
        action="silent",
        reason="background_observe",
    )


def build_positive_engagement_policies(
    snapshot: PersonalitySnapshot | None,
    *,
    engagement_signals: Sequence[WorldInterestSignal] = (),
    max_items: int = 3,
) -> tuple[PositiveEngagementTopicPolicy, ...]:
    """Build bounded per-topic positive-engagement actions for the turn."""

    items = build_mindshare_items(
        snapshot,
        engagement_signals=engagement_signals,
        max_items=max_items,
    )
    return tuple(
        derive_positive_engagement_policy(
            item,
            engagement_signals=engagement_signals,
        )
        for item in items
    )


def _render_action(value: str) -> str:
    """Render one positive-engagement action into prompt-facing language."""

    if value == "invite_follow_up":
        return "give one short update and optionally invite the user to keep going if they want"
    if value == "ask_one":
        return "it is okay to ask one calm engaging question if the moment is open"
    if value == "brief_update":
        return "it is okay to offer one short concrete update in open conversation"
    if value == "hint":
        return "only give a light hint when the current exchange naturally opens it"
    return "keep this in background awareness unless the user clearly asks for it"


def render_positive_engagement_policy(
    snapshot: PersonalitySnapshot | None,
    *,
    engagement_signals: Sequence[WorldInterestSignal] = (),
) -> str | None:
    """Render authoritative positive-engagement guidance for the current turn."""

    policies = build_positive_engagement_policies(
        snapshot,
        engagement_signals=engagement_signals,
    )
    if not policies:
        return None
    lines = [
        "## Positive engagement policy",
        (
            "- Use these bounded actions to encourage welcomed conversation growth, not to push or trap attention."
        ),
        (
            "- Prefer calm, specific invitations over pressure: at most one topic should get an active positive-engagement move in a turn unless the user clearly keeps pulling."
        ),
        (
            "- If a topic is cooling or avoid, back off. If a topic is becoming a shared thread, it is fine to help it grow a little more naturally."
        ),
    ]
    for policy in policies:
        lines.append(
            f"- {policy.title}: {policy.attention_state}; action {policy.action}; {_render_action(policy.action)}."
        )
    return "\n".join(lines)
