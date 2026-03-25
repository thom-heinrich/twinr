"""Derive authoritative conversation-steering cues from durable companion state.

This module turns prompt-time mindshare plus persisted world-intelligence state
into compact turn-steering cues. Unlike ``MINDSHARE``, these cues are rendered
into the authoritative ``PERSONALITY`` layer so Twinr can make more stable
turn-level decisions about when to briefly update, when one calm follow-up is
appropriate, and when it should simply keep observing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from twinr.agent.personality.intelligence.models import WorldInterestSignal
from twinr.agent.personality.models import PersonalitySnapshot
from twinr.agent.personality.positive_engagement import (
    derive_positive_engagement_policy,
)
from twinr.agent.personality.self_expression import (
    CompanionMindshareItem,
    build_mindshare_items,
)


def _normalized_text(value: object | None) -> str:
    """Collapse arbitrary text into one trimmed single-line string."""

    return " ".join(str(value or "").split()).strip()


def _interest_key(value: object | None) -> str:
    """Normalize one topic title into a stable matching key."""

    return _normalized_text(value).casefold()


@dataclass(frozen=True, slots=True)
class ConversationTurnSteeringCue:
    """Describe how one topic may steer the current turn.

    Attributes:
        title: Human-readable topic title the cue applies to.
        salience: Relative importance for ranking within the current turn.
        attention_state: Coarser steering state such as ``shared_thread`` or
            ``cooling``.
        open_offer: Whether Twinr may proactively offer a short update when the
            conversation is open-ended.
        user_pull: What Twinr may do after the user clearly re-engages the
            topic inside the current exchange.
        observe_mode: How Twinr should behave when the topic is not being
            actively pulled forward by the user.
        positive_engagement_action: Explicit bounded action for fostering
            welcomed engagement around the topic during the current turn.
        match_summary: Short semantic description of what should count as a
            true match for this cue. The closure evaluator uses this to avoid
            over-matching adjacent local or community themes onto a broader
            topic such as politics.
    """

    title: str
    salience: float
    attention_state: str = "background"
    open_offer: str = "only_if_clearly_relevant"
    user_pull: str = "wait_for_user_pull"
    observe_mode: str = "keep_observing_in_background"
    positive_engagement_action: str = "silent"
    match_summary: str = ""


@dataclass(frozen=True, slots=True)
class FollowUpSteeringDecision:
    """Resolve one runtime follow-up stance from matched steering topics.

    Attributes:
        matched_topics: Topic titles that the closure evaluator said matched
            the just-finished exchange.
        selected_topic: Strongest matched topic after salience-aware ranking.
        attention_state: Steering-state label for the selected topic.
        force_close: Whether runtime should release the automatic follow-up
            window after the current answer.
        keep_open: Whether runtime may safely keep the follow-up window open
            because the user is still pulling a shared or forming thread.
        positive_engagement_action: Bounded positive-engagement action
            associated with the selected topic.
        reason: Short bounded reason code for telemetry and tests.
    """

    matched_topics: tuple[str, ...] = ()
    selected_topic: str | None = None
    attention_state: str = "background"
    force_close: bool = False
    keep_open: bool = False
    positive_engagement_action: str = "silent"
    reason: str = "neutral"


def _matching_signal(
    item: CompanionMindshareItem,
    *,
    engagement_signals: Sequence[WorldInterestSignal],
) -> WorldInterestSignal | None:
    """Return the strongest matching durable interest signal for one item."""

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
            signal.engagement_score,
            signal.salience,
            signal.updated_at or "",
        ),
        reverse=True,
    )
    return next(iter(ranked), None)


def _bounded_summary(value: object | None, *, limit: int = 180) -> str:
    """Collapse one semantic cue summary into bounded single-line text."""

    normalized = _normalized_text(value)
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)].rstrip() + "..."


def _cue_title(
    item: CompanionMindshareItem,
    *,
    matching_signal: WorldInterestSignal | None,
) -> str:
    """Return one prompt-facing cue title with optional regional disambiguation.

    Broad topics such as ``local politics`` are easy for a general model to
    over-match against nearby community or neighbourhood themes. When the
    durable signal already carries a concrete local or regional anchor, keep
    that anchor in the cue title so the evaluator sees the narrower intended
    topic.
    """

    title = _normalized_text(item.title)
    if matching_signal is None:
        return title
    region = _normalized_text(getattr(matching_signal, "region", None))
    scope = _normalized_text(getattr(matching_signal, "scope", None)).casefold()
    if not region or scope not in {"local", "regional"}:
        return title
    if region.casefold() in title.casefold():
        return title
    return f"{region} {title}"


def _cue_match_summary(
    item: CompanionMindshareItem,
    *,
    matching_signal: WorldInterestSignal | None,
) -> str:
    """Return one semantic description that bounds cue matching.

    The closure evaluator should not match purely on lexical overlap. This
    summary gives it compact topic meaning without introducing hardcoded
    user-specific rules.
    """

    if matching_signal is not None:
        summary = _bounded_summary(getattr(matching_signal, "summary", None))
        if summary:
            return summary
    return _bounded_summary(item.summary)


def _derive_turn_steering_cue(
    item: CompanionMindshareItem,
    *,
    engagement_signals: Sequence[WorldInterestSignal],
) -> ConversationTurnSteeringCue:
    """Convert one surfaced mindshare item into a turn-steering cue."""

    matching_signal = _matching_signal(item, engagement_signals=engagement_signals)
    appetite = item.appetite
    positive_engagement = derive_positive_engagement_policy(
        item,
        engagement_signals=engagement_signals,
    )
    co_attention_state = (
        _normalized_text(getattr(matching_signal, "co_attention_state", None)).casefold()
        or "latent"
    )
    cue_title = _cue_title(item, matching_signal=matching_signal)
    match_summary = _cue_match_summary(item, matching_signal=matching_signal)

    if appetite.state == "avoid":
        return ConversationTurnSteeringCue(
            title=cue_title,
            salience=item.salience,
            attention_state="avoid",
            open_offer="do_not_steer",
            user_pull="answer_briefly_then_release",
            observe_mode="stay_off_this_topic",
            positive_engagement_action=positive_engagement.action,
            match_summary=match_summary,
        )
    if appetite.state == "cooling":
        return ConversationTurnSteeringCue(
            title=cue_title,
            salience=item.salience,
            attention_state="cooling",
            open_offer="do_not_steer",
            user_pull="answer_briefly_then_release",
            observe_mode="keep_observing_without_steering",
            positive_engagement_action=positive_engagement.action,
            match_summary=match_summary,
        )
    if positive_engagement.action == "invite_follow_up":
        return ConversationTurnSteeringCue(
            title=cue_title,
            salience=item.salience,
            attention_state="shared_thread",
            open_offer="brief_update_if_open",
            user_pull="one_calm_follow_up",
            observe_mode="keep_observing_in_background",
            positive_engagement_action=positive_engagement.action,
            match_summary=match_summary,
        )
    if positive_engagement.action == "ask_one":
        return ConversationTurnSteeringCue(
            title=cue_title,
            salience=item.salience,
            attention_state="forming" if co_attention_state == "forming" else "growing",
            open_offer="mention_if_clearly_relevant",
            user_pull=(
                "one_calm_follow_up"
                if co_attention_state == "shared_thread"
                else "one_gentle_follow_up"
            ),
            observe_mode="mostly_observe_until_user_pull",
            positive_engagement_action=positive_engagement.action,
            match_summary=match_summary,
        )
    if positive_engagement.action == "brief_update":
        return ConversationTurnSteeringCue(
            title=cue_title,
            salience=item.salience,
            attention_state=(
                "shared_thread"
                if co_attention_state == "shared_thread"
                else ("forming" if co_attention_state == "forming" else "growing")
            ),
            open_offer="brief_update_if_open",
            user_pull="wait_for_user_pull",
            observe_mode="keep_observing_in_background",
            positive_engagement_action=positive_engagement.action,
            match_summary=match_summary,
        )
    if positive_engagement.action == "hint":
        return ConversationTurnSteeringCue(
            title=cue_title,
            salience=item.salience,
            attention_state=(
                "forming"
                if co_attention_state == "forming"
                else ("growing" if appetite.interest in {"active", "growing"} else "background")
            ),
            open_offer="mention_if_clearly_relevant",
            user_pull="wait_for_user_pull",
            observe_mode="mostly_observe_until_user_pull",
            positive_engagement_action=positive_engagement.action,
            match_summary=match_summary,
        )
    return ConversationTurnSteeringCue(
        title=cue_title,
        salience=item.salience,
        attention_state="background",
        open_offer="wait_for_user_pull",
        user_pull="wait_for_user_pull",
        observe_mode="keep_observing_in_background",
        positive_engagement_action=positive_engagement.action,
        match_summary=match_summary,
    )


def build_turn_steering_cues(
    snapshot: PersonalitySnapshot | None,
    *,
    engagement_signals: Sequence[WorldInterestSignal] = (),
    max_items: int = 3,
) -> tuple[ConversationTurnSteeringCue, ...]:
    """Build the bounded set of topics that may steer the current turn."""

    items = build_mindshare_items(
        snapshot,
        max_items=max_items,
        engagement_signals=engagement_signals,
    )
    return tuple(
        _derive_turn_steering_cue(
            item,
            engagement_signals=engagement_signals,
        )
        for item in items
    )


def serialize_turn_steering_cues(
    cues: Sequence[ConversationTurnSteeringCue],
) -> tuple[Mapping[str, object], ...]:
    """Serialize steering cues for compact evaluator prompts.

    Args:
        cues: Prompt-time steering cues selected for the current turn.

    Returns:
        Plain mappings that can be embedded into compact JSON payloads for
        turn-scoped tool evaluators.
    """

    serialized: list[Mapping[str, object]] = []
    for cue in cues:
        serialized.append(
            {
                "title": cue.title,
                "attention_state": cue.attention_state,
                "open_offer": cue.open_offer,
                "user_pull": cue.user_pull,
                "observe_mode": cue.observe_mode,
                "positive_engagement_action": cue.positive_engagement_action,
                "match_summary": cue.match_summary,
                "salience": round(float(cue.salience), 3),
            }
        )
    return tuple(serialized)


def resolve_follow_up_steering(
    cues: Sequence[ConversationTurnSteeringCue],
    *,
    matched_topics: Sequence[str] = (),
) -> FollowUpSteeringDecision:
    """Resolve runtime follow-up behavior from matched steering topics.

    Args:
        cues: Current authoritative steering cues for this turn.
        matched_topics: Topic titles that the closure evaluator matched to the
            just-finished exchange.

    Returns:
        A bounded steering decision. Topics marked for ``answer_briefly_then_release``
        or ``answer_then_pause`` close the automatic follow-up window; topics
        that allow one calm or gentle follow-up keep it open.
    """

    normalized_topics: list[str] = []
    seen_topics: set[str] = set()
    for topic in matched_topics:
        normalized = _normalized_text(topic)
        if not normalized:
            continue
        key = _interest_key(normalized)
        if key in seen_topics:
            continue
        seen_topics.add(key)
        normalized_topics.append(normalized)
    if not normalized_topics:
        return FollowUpSteeringDecision()

    cue_by_key = {
        _interest_key(cue.title): cue
        for cue in cues
        if _interest_key(cue.title)
    }
    matching_cues = [
        cue_by_key[key]
        for key in (_interest_key(topic) for topic in normalized_topics)
        if key in cue_by_key
    ]
    if not matching_cues:
        return FollowUpSteeringDecision(matched_topics=tuple(normalized_topics))

    selected = max(
        matching_cues,
        key=lambda cue: (
            cue.salience,
            cue.attention_state == "shared_thread",
            cue.attention_state == "forming",
            cue.attention_state == "growing",
        ),
    )
    if selected.user_pull in {"answer_briefly_then_release", "answer_then_pause"}:
        return FollowUpSteeringDecision(
            matched_topics=tuple(normalized_topics),
            selected_topic=selected.title,
            attention_state=selected.attention_state,
            force_close=True,
            positive_engagement_action=selected.positive_engagement_action,
            reason="release_after_answer",
        )
    if selected.user_pull in {"one_calm_follow_up", "one_gentle_follow_up"}:
        return FollowUpSteeringDecision(
            matched_topics=tuple(normalized_topics),
            selected_topic=selected.title,
            attention_state=selected.attention_state,
            keep_open=True,
            positive_engagement_action=selected.positive_engagement_action,
            reason="follow_up_allowed",
        )
    return FollowUpSteeringDecision(
        matched_topics=tuple(normalized_topics),
        selected_topic=selected.title,
        attention_state=selected.attention_state,
        positive_engagement_action=selected.positive_engagement_action,
        reason="neutral",
    )


def _render_attention_state(value: str) -> str:
    """Render one steering state into prompt-facing language."""

    if value == "shared_thread":
        return "shared thread"
    if value == "forming":
        return "forming shared thread"
    if value == "growing":
        return "growing focus"
    if value == "cooling":
        return "cooling topic"
    if value == "avoid":
        return "avoid topic"
    return "background topic"


def _render_open_offer(value: str) -> str:
    """Render one open-conversation steering action into plain language."""

    if value == "brief_update_if_open":
        return "in open conversation, one short concrete update is okay"
    if value == "mention_if_clearly_relevant":
        return "only mention it when the current exchange clearly invites it"
    if value == "wait_for_user_pull":
        return "do not volunteer it; wait for the user to pull it forward"
    if value == "do_not_steer":
        return "do not steer the conversation toward it"
    return "only mention it if clearly relevant"


def _render_user_pull(value: str) -> str:
    """Render one user-pull steering action into plain language."""

    if value == "one_calm_follow_up":
        return "after the user re-engages it, one calm follow-up question is okay"
    if value == "one_gentle_follow_up":
        return "after the user re-engages it, at most one gentle follow-up is okay"
    if value == "answer_briefly_then_release":
        return "if the user asks about it, answer briefly and then release it"
    if value == "answer_then_pause":
        return "if the user asks about it, answer and pause instead of pushing further"
    return "wait for the user to keep pulling before going further"


def _render_observe_mode(value: str) -> str:
    """Render one background observation mode into plain language."""

    if value == "keep_observing_in_background":
        return "otherwise keep watching it quietly in the background"
    if value == "mostly_observe_until_user_pull":
        return "otherwise mostly observe and let the user decide whether it grows"
    if value == "keep_observing_without_steering":
        return "otherwise keep it in view without steering back toward it"
    if value == "stay_off_this_topic":
        return "otherwise stay off this topic unless the user clearly insists"
    return "otherwise keep it in background awareness"


def _render_positive_engagement_action(value: str) -> str:
    """Render one explicit positive-engagement action into plain language."""

    if value == "invite_follow_up":
        return "if the turn is open, a brief update plus a low-pressure invitation is okay"
    if value == "ask_one":
        return "if the user is with it, one calm engaging question is okay"
    if value == "brief_update":
        return "in open conversation, one short concrete update is okay"
    if value == "hint":
        return "at most give one light hint when the exchange naturally opens it"
    return "otherwise keep this quiet unless the user clearly asks"


def render_turn_steering_policy(
    snapshot: PersonalitySnapshot | None,
    *,
    engagement_signals: Sequence[WorldInterestSignal] = (),
) -> str | None:
    """Render authoritative turn-steering guidance for the current state."""

    cues = build_turn_steering_cues(
        snapshot,
        engagement_signals=engagement_signals,
    )
    if not cues:
        return None
    lines = [
        "## Current conversation steering",
        (
            "- Use these cues to decide whether to briefly update, gently follow up, or simply observe during this turn."
        ),
        (
            "- Shared-thread topics may guide an open conversation with one short concrete update, but do not stack multiple unsolicited topic pivots."
        ),
        (
            "- After one calm follow-up on a shared-thread topic, return to observing unless the user clearly keeps pulling that thread forward."
        ),
    ]
    for cue in cues:
        lines.append(
            (
                f"- {cue.title}: {_render_attention_state(cue.attention_state)}; "
                f"{_render_open_offer(cue.open_offer)}; "
                f"{_render_user_pull(cue.user_pull)}; "
                f"{_render_observe_mode(cue.observe_mode)}; "
                f"{_render_positive_engagement_action(cue.positive_engagement_action)}."
            )
        )
    return "\n".join(lines)
