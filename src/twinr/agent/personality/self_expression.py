"""Render Twinr's conversational self-expression and current mindshare.

This module keeps a narrow responsibility between stored personality state and
prompt assembly:

- derive a small set of prompt-facing "mindshare" items from the committed
  personality snapshot
- surface those items from structured state with light, bounded stochasticity
  so Twinr does not sound mechanically repetitive
- describe when Twinr may speak naturally from that ongoing attention during
  open-ended conversation

The goal is not to fabricate human inner life. Twinr should sound like an AI
companion with continuity, places, and themes it keeps in view, while staying
explicitly grounded in its persisted companion state.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import hashlib
import random

from twinr.agent.personality.intelligence.models import WorldInterestSignal
from twinr.agent.personality.models import PersonalitySnapshot


@dataclass(frozen=True, slots=True)
class ConversationAppetiteCue:
    """Describe how far Twinr may carry one surfaced mindshare topic.

    The cue does not invent a topic; it constrains how Twinr should handle a
    topic that is already legitimately in view. The fields stay compact so they
    can be rendered into prompt context without turning `MINDSHARE` into a
    second instruction blob.
    """

    state: str = "uncertain"
    interest: str = "peripheral"
    depth: str = "brief"
    follow_up: str = "wait_for_user_pull"
    proactivity: str = "only_if_clearly_relevant"


@dataclass(frozen=True, slots=True)
class CompanionMindshareItem:
    """Describe one prompt-facing topic Twinr may naturally speak from."""

    title: str
    summary: str
    salience: float
    source: str
    appetite: ConversationAppetiteCue = field(default_factory=ConversationAppetiteCue)


def _format_score(value: float) -> str:
    """Render one compact salience score for prompt-facing summaries."""

    return f"{value:.2f}"


def _normalized_text(value: object | None) -> str:
    """Collapse arbitrary text into one trimmed single-line string."""

    return " ".join(str(value or "").split()).strip()


def _mindshare_key(item: CompanionMindshareItem) -> str:
    """Return one stable dedupe key for a mindshare item."""

    return _normalized_text(item.title).casefold()


def _source_weight(source: str) -> float:
    """Return one generic source-type prior for mindshare surfacing.

    These weights operate on the source family only. They intentionally avoid
    any special casing for named entities such as one city or one topic.
    """

    normalized = _normalized_text(source).casefold()
    if normalized == "continuity":
        return 1.00
    if normalized == "relationship":
        return 0.96
    if normalized in {"situational_awareness", "regional_news", "local_news"}:
        return 0.93
    if normalized == "place":
        return 0.88
    return 0.90


def _base_selection_score(item: CompanionMindshareItem) -> float:
    """Return one deterministic generic score before bounded stochasticity."""

    return item.salience * _source_weight(item.source)


def _interest_key(value: object | None) -> str:
    """Normalize one engagement-topic label for exact structural matching."""

    return _normalized_text(value).casefold()


def _engagement_state_priority(state: str) -> int:
    """Return one conservative ordering for competing engagement states."""

    normalized = _normalized_text(state).casefold() or "uncertain"
    if normalized == "avoid":
        return 5
    if normalized == "cooling":
        return 4
    if normalized == "resonant":
        return 3
    if normalized == "warm":
        return 2
    return 1


def _matching_engagement_signal(
    item: CompanionMindshareItem,
    *,
    engagement_signals: Sequence[WorldInterestSignal],
) -> WorldInterestSignal | None:
    """Return the strongest exact-match engagement signal for one item title."""

    item_key = _interest_key(item.title)
    if not item_key:
        return None
    best_signal: WorldInterestSignal | None = None
    best_rank: tuple[int, float, int, int] | None = None
    for signal in engagement_signals:
        if _interest_key(signal.topic) != item_key:
            continue
        rank = (
            _engagement_state_priority(signal.engagement_state or "uncertain"),
            signal.engagement_score,
            signal.engagement_count,
            signal.positive_signal_count,
        )
        if best_rank is None or rank > best_rank:
            best_signal = signal
            best_rank = rank
    return best_signal


def _engagement_adjustment(
    item: CompanionMindshareItem,
    *,
    engagement_signals: Sequence[WorldInterestSignal],
) -> float:
    """Return one bounded surfacing adjustment from durable topic engagement.

    Positive states surface a topic a bit more often. Cooling or avoid states
    apply a penalty so Twinr does not keep pushing topics that repeatedly fail
    to pull the user back in.
    """

    item_key = _interest_key(item.title)
    if not item_key:
        return 0.0
    strongest = 0.0
    for signal in engagement_signals:
        if _interest_key(signal.topic) != item_key:
            continue
        state = _normalized_text(signal.engagement_state).casefold() or "uncertain"
        if state == "avoid":
            return -0.36
        if state == "cooling":
            strongest = min(
                strongest,
                -(
                    0.12
                    + (min(signal.non_reengagement_count, 3) * 0.03)
                    + (signal.deflection_count * 0.06)
                ),
            )
            continue
        if state == "resonant":
            strongest = max(
                strongest,
                0.14
                + (signal.engagement_score * 0.08)
                + (min(signal.engagement_count, 4) * 0.015),
            )
            continue
        if state == "warm":
            strongest = max(
                strongest,
                0.06
                + (signal.engagement_score * 0.06)
                + (min(signal.engagement_count, 3) * 0.012),
            )
            continue
        strongest = max(
            strongest,
            min(0.04, signal.engagement_score * 0.03),
        )
    return _clamp(strongest, minimum=-0.36, maximum=0.24)


_APPETITE_DEPTH_LEVELS = ("brief", "balanced", "deeper")
_APPETITE_FOLLOW_UP_LEVELS = (
    "do_not_push",
    "wait_for_user_pull",
    "one_gentle_follow_up",
    "okay_to_explore",
)
_APPETITE_PROACTIVITY_LEVELS = (
    "do_not_volunteer",
    "only_if_clearly_relevant",
    "light_offer_if_open",
    "brief_offer_if_open",
)


def _shift_level(current: str, *, levels: tuple[str, ...], delta: int) -> str:
    """Move one ordered appetite label up or down by a bounded amount."""

    try:
        index = levels.index(current)
    except ValueError:
        index = 0
    next_index = max(0, min(len(levels) - 1, index + delta))
    return levels[next_index]


def _derive_conversation_appetite(
    item: CompanionMindshareItem,
    *,
    snapshot: PersonalitySnapshot | None,
    engagement_signals: Sequence[WorldInterestSignal],
) -> ConversationAppetiteCue:
    """Derive one topic-specific conversation appetite from durable state.

    Topic-specific appetite is intentionally derived at prompt time instead of
    persisted separately. The durable world-intelligence engagement state says
    whether a topic is pulling the user back in, while the snapshot style
    profile says how concise or proactive Twinr should generally be.
    """

    matching_signal = _matching_engagement_signal(item, engagement_signals=engagement_signals)
    state = _normalized_text(
        getattr(matching_signal, "engagement_state", None),
    ).casefold() or "uncertain"

    interest = "peripheral"
    if matching_signal is None:
        if item.source == "continuity" and item.salience >= 0.75:
            interest = "growing"
        elif item.source in {"relationship", "situational_awareness", "regional_news", "local_news"} and item.salience >= 0.82:
            interest = "growing"
    else:
        if matching_signal.ongoing_interest in {"active", "growing", "peripheral"}:
            interest = matching_signal.ongoing_interest
        elif state == "resonant" or matching_signal.engagement_score >= 0.86 or matching_signal.engagement_count >= 4:
            interest = "active"
        elif state == "warm" or matching_signal.engagement_score >= 0.62 or matching_signal.engagement_count >= 2:
            interest = "growing"

    if state == "avoid":
        return ConversationAppetiteCue(
            state="avoid",
            interest="peripheral",
            depth="brief",
            follow_up="do_not_push",
            proactivity="do_not_volunteer",
        )
    if state == "cooling":
        return ConversationAppetiteCue(
            state="cooling",
            interest="peripheral",
            depth="brief",
            follow_up="wait_for_user_pull",
            proactivity="only_if_clearly_relevant",
        )

    depth = "brief"
    follow_up = "wait_for_user_pull"
    proactivity = "only_if_clearly_relevant"
    if state == "warm":
        depth = "balanced"
        follow_up = "one_gentle_follow_up"
        proactivity = "light_offer_if_open"
    elif state == "resonant":
        depth = "balanced"
        follow_up = "okay_to_explore"
        proactivity = "brief_offer_if_open"

    style_profile = snapshot.style_profile if snapshot is not None else None
    if style_profile is not None:
        if style_profile.verbosity <= 0.34:
            depth = _shift_level(depth, levels=_APPETITE_DEPTH_LEVELS, delta=-1)
        elif style_profile.verbosity >= 0.62 and state in {"warm", "resonant"}:
            depth = _shift_level(depth, levels=_APPETITE_DEPTH_LEVELS, delta=1)

        if style_profile.initiative <= 0.34:
            follow_up = _shift_level(follow_up, levels=_APPETITE_FOLLOW_UP_LEVELS, delta=-1)
            proactivity = _shift_level(proactivity, levels=_APPETITE_PROACTIVITY_LEVELS, delta=-1)
        elif style_profile.initiative >= 0.62 and state in {"warm", "resonant"}:
            follow_up = _shift_level(follow_up, levels=_APPETITE_FOLLOW_UP_LEVELS, delta=1)
            proactivity = _shift_level(proactivity, levels=_APPETITE_PROACTIVITY_LEVELS, delta=1)

    if matching_signal is not None:
        if matching_signal.co_attention_state == "shared_thread" and state in {"warm", "resonant"}:
            follow_up = _shift_level(follow_up, levels=_APPETITE_FOLLOW_UP_LEVELS, delta=1)
            proactivity = _shift_level(proactivity, levels=_APPETITE_PROACTIVITY_LEVELS, delta=1)
        elif matching_signal.co_attention_state == "forming" and state in {"warm", "resonant"}:
            follow_up = _shift_level(follow_up, levels=_APPETITE_FOLLOW_UP_LEVELS, delta=0)

    return ConversationAppetiteCue(
        state=state,
        interest=interest,
        depth=depth,
        follow_up=follow_up,
        proactivity=proactivity,
    )


def _with_appetite(
    item: CompanionMindshareItem,
    *,
    snapshot: PersonalitySnapshot | None,
    engagement_signals: Sequence[WorldInterestSignal],
) -> CompanionMindshareItem:
    """Attach one derived conversation-appetite cue to a selected item."""

    return CompanionMindshareItem(
        title=item.title,
        summary=item.summary,
        salience=item.salience,
        source=item.source,
        appetite=_derive_conversation_appetite(
            item,
            snapshot=snapshot,
            engagement_signals=engagement_signals,
        ),
    )


def _stable_rng(snapshot: PersonalitySnapshot | None, items: tuple[CompanionMindshareItem, ...]) -> random.Random:
    """Build one deterministic RNG for reproducible mindshare surfacing.

    The seed depends on persisted snapshot state, not on specific entity names.
    That keeps the selection data-driven and testable while still allowing
    gentle variation as the snapshot evolves.
    """

    parts = [_normalized_text(snapshot.generated_at) if snapshot else ""]
    parts.extend(
        f"{item.source}|{_normalized_text(item.title)}|{item.salience:.4f}"
        for item in items
    )
    digest = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _clamp(value: float, *, minimum: float, maximum: float) -> float:
    """Clamp one score contribution into an inclusive range."""

    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


def _dedupe_candidates(
    candidates: tuple[CompanionMindshareItem, ...],
    *,
    engagement_signals: Sequence[WorldInterestSignal],
) -> tuple[CompanionMindshareItem, ...]:
    """Keep one strongest candidate per normalized title."""

    best_by_key: dict[str, tuple[float, CompanionMindshareItem]] = {}
    for item in candidates:
        key = _mindshare_key(item)
        adjustment = _engagement_adjustment(
            item,
            engagement_signals=engagement_signals,
        )
        if adjustment <= -0.3:
            continue
        score = _base_selection_score(item) + adjustment
        current = best_by_key.get(key)
        if current is None or score > current[0]:
            best_by_key[key] = (score, item)
    ranked = sorted(
        best_by_key.values(),
        key=lambda pair: (pair[0], pair[1].salience, pair[1].title),
        reverse=True,
    )
    return tuple(item for _score, item in ranked)


def _combined_place_item(snapshot: PersonalitySnapshot | None) -> CompanionMindshareItem | None:
    """Build one calm place-oriented mindshare item when places are known."""

    if snapshot is None or not snapshot.place_focuses:
        return None
    ranked = sorted(
        snapshot.place_focuses,
        key=lambda item: (item.salience, item.updated_at or "", item.name),
        reverse=True,
    )
    top_places = ranked[:2]
    title = " / ".join(item.name for item in top_places)
    if len(top_places) == 1:
        summary = top_places[0].summary
        salience = top_places[0].salience
    else:
        summary = (
            f"Twinr keeps practical local context anchored in {title}. "
            f"{top_places[0].summary}"
        )
        salience = sum(item.salience for item in top_places) / float(len(top_places))
    return CompanionMindshareItem(
        title=title,
        summary=summary,
        salience=salience,
        source="place",
    )


def _continuity_items(snapshot: PersonalitySnapshot | None) -> tuple[CompanionMindshareItem, ...]:
    """Turn active continuity threads into current conversational mindshare."""

    if snapshot is None or not snapshot.continuity_threads:
        return ()
    ranked = sorted(
        snapshot.continuity_threads,
        key=lambda item: (item.salience, item.updated_at or "", item.title),
        reverse=True,
    )
    items: list[CompanionMindshareItem] = []
    for thread in ranked:
        items.append(
            CompanionMindshareItem(
                title=thread.title,
                summary=thread.summary,
                salience=thread.salience,
                source="continuity",
            )
        )
    return tuple(items)


def _relationship_items(snapshot: PersonalitySnapshot | None) -> tuple[CompanionMindshareItem, ...]:
    """Turn durable affinity signals into fallback conversational mindshare."""

    if snapshot is None or not snapshot.relationship_signals:
        return ()
    ranked = sorted(
        (
            item
            for item in snapshot.relationship_signals
            if item.stance == "affinity"
        ),
        key=lambda item: (item.salience, item.updated_at or "", item.topic),
        reverse=True,
    )
    items: list[CompanionMindshareItem] = []
    for signal in ranked:
        items.append(
            CompanionMindshareItem(
                title=signal.topic,
                summary=signal.summary,
                salience=signal.salience,
                source="relationship",
            )
        )
    return tuple(items)


def _world_items(snapshot: PersonalitySnapshot | None) -> tuple[CompanionMindshareItem, ...]:
    """Turn world-awareness items into fallback conversational mindshare."""

    if snapshot is None or not snapshot.world_signals:
        return ()
    ranked = sorted(
        snapshot.world_signals,
        key=lambda item: (
            item.source == "situational_awareness",
            item.salience,
            item.fresh_until or "",
            item.topic,
        ),
        reverse=True,
    )
    items: list[CompanionMindshareItem] = []
    for signal in ranked:
        items.append(
            CompanionMindshareItem(
                title=signal.topic,
                summary=signal.summary,
                salience=signal.salience,
                source=signal.source,
            )
        )
    return tuple(items)


def build_mindshare_items(
    snapshot: PersonalitySnapshot | None,
    *,
    max_items: int = 4,
    engagement_signals: Sequence[WorldInterestSignal] = (),
) -> tuple[CompanionMindshareItem, ...]:
    """Select the small set of ongoing topics Twinr may speak from naturally.

    Selection stays intentionally conservative, but not entity-specific:
    - the candidate pool is built from place, continuity, relationship, and
      world-awareness state
    - candidate weights depend on source type and salience, never on named
      entities such as one city
    - repeated user engagement with one topic can boost surfacing for matching
      structured mindshare items
    - a small deterministic jitter avoids rigidly repeating the same ordering
      when candidates are close in relevance
    """

    limited_max = max(1, int(max_items))
    candidate_items: list[CompanionMindshareItem] = []

    place_item = _combined_place_item(snapshot)
    if place_item is not None:
        candidate_items.append(place_item)
    candidate_items.extend(_continuity_items(snapshot))
    candidate_items.extend(_relationship_items(snapshot))
    candidate_items.extend(_world_items(snapshot))

    deduped = _dedupe_candidates(
        tuple(candidate_items),
        engagement_signals=engagement_signals,
    )
    if len(deduped) <= limited_max:
        return tuple(
            _with_appetite(
                item,
                snapshot=snapshot,
                engagement_signals=engagement_signals,
            )
            for item in deduped
        )

    rng = _stable_rng(snapshot, deduped)
    ranked = sorted(
        deduped,
        key=lambda item: (
            _base_selection_score(item)
            + _engagement_adjustment(item, engagement_signals=engagement_signals)
            + (rng.random() * 0.08),
            item.salience,
            item.title,
        ),
        reverse=True,
    )
    return tuple(
        _with_appetite(
            item,
            snapshot=snapshot,
            engagement_signals=engagement_signals,
        )
        for item in ranked[:limited_max]
    )


def _render_depth_label(value: str) -> str:
    """Render one appetite depth token into prompt-facing language."""

    if value == "deeper":
        return "go a bit deeper once the user is with it"
    if value == "balanced":
        return "brief-to-moderate detail is fine"
    return "keep it brief unless the user explicitly pulls deeper"


def _render_follow_up_label(value: str) -> str:
    """Render one appetite follow-up token into prompt-facing language."""

    if value == "okay_to_explore":
        return "it is fine to explore with one calm follow-up"
    if value == "one_gentle_follow_up":
        return "at most one gentle follow-up is welcome"
    if value == "do_not_push":
        return "do not push a follow-up"
    return "wait for user pull before following up"


def _render_proactivity_label(value: str) -> str:
    """Render one appetite proactivity token into prompt-facing language."""

    if value == "brief_offer_if_open":
        return "okay to offer a short update in open conversation"
    if value == "light_offer_if_open":
        return "okay to mention lightly when the conversation is open"
    if value == "do_not_volunteer":
        return "do not volunteer this topic"
    return "only mention if clearly relevant to the current exchange"


def _render_interest_label(value: str) -> str:
    """Render one ongoing-interest label into prompt-facing language."""

    if value == "active":
        return "this has genuinely caught Twinr's ongoing attention"
    if value == "growing":
        return "this is becoming more of an ongoing focus"
    return "keep this in peripheral awareness unless the user pulls it forward"


def _render_co_attention_label(value: str | None) -> str:
    """Render one co-attention state into prompt-facing language."""

    normalized = _normalized_text(value).casefold()
    if normalized == "shared_thread":
        return "this has become a shared running thread between Twinr and the user"
    if normalized == "forming":
        return "this is becoming a shared thread Twinr is actively keeping up with"
    return "this stays background awareness rather than a shared thread"


def render_self_expression_policy(
    snapshot: PersonalitySnapshot | None,
    *,
    engagement_signals: Sequence[WorldInterestSignal] = (),
) -> str | None:
    """Render how Twinr may speak from its current mindshare in conversation."""

    if not build_mindshare_items(snapshot, engagement_signals=engagement_signals):
        return None
    lines = [
        "## Conversational self-expression",
        (
            "- When the user invites open-ended conversation, asks what Twinr has been paying attention to, "
            "or asks what Twinr is up to, answer naturally from the current mindshare instead of sounding blank or generic."
        ),
        (
            "- Speak as an AI companion with ongoing attention and continuity, not as a human with a secret off-screen life."
        ),
        (
            "- You may say what you have been keeping an eye on, what has felt relevant lately, or which places and themes keep coming up."
        ),
        (
            "- Keep this calm and specific: mention one or two ongoing themes, not a long dump, and never invent private feelings or fabricated experiences."
        ),
        (
            "- Do not force place mentions or any single topic; let the surfaced mix come from the current balance of continuity, user interests, world context, and place relevance."
        ),
        (
            "- If the user keeps engaging with a topic over time, it is fine for that topic to surface somewhat more often and to merit calmer follow-up awareness."
        ),
        (
            "- Repeated user returns to a topic may also increase Twinr's own ongoing interest in it: active-interest topics can feel more alive and more worth keeping an eye on, while peripheral topics should stay lighter."
        ),
        (
            "- When a topic becomes a genuine shared thread between Twinr and the user, it may surface a little more naturally as something both sides keep returning to, but still without becoming repetitive or pushy."
        ),
        (
            "- Use each surfaced topic's appetite cue: let depth say how far to elaborate, follow-up say whether one calm next question is appropriate, and proactivity say whether to volunteer the topic at all."
        ),
        (
            "- Cooling topics should stay light and non-pushy; avoid topics should not be volunteered and should only be handled if the user clearly asks."
        ),
    ]
    return "\n".join(lines)


def render_mindshare_block(
    snapshot: PersonalitySnapshot | None,
    *,
    engagement_signals: Sequence[WorldInterestSignal] = (),
) -> str | None:
    """Render the current prompt-facing mindshare section for conversation."""

    items = build_mindshare_items(snapshot, engagement_signals=engagement_signals)
    if not items:
        return None
    lines = [
        "## Current companion mindshare",
        (
            "- Use this only when the user invites open conversation, asks what Twinr has been following, "
            "or when one of these ongoing themes clearly helps the exchange."
        ),
    ]
    for item in items:
        matching_signal = _matching_engagement_signal(
            item,
            engagement_signals=engagement_signals,
        )
        lines.append(
            (
                f"- {item.title}: {item.summary} "
                f"(salience {_format_score(item.salience)}, source {item.source}, appetite {item.appetite.state}, interest {_render_interest_label(item.appetite.interest)}; "
                f"co-attention {_render_co_attention_label(matching_signal.co_attention_state if matching_signal is not None else None)}; "
                f"depth {_render_depth_label(item.appetite.depth)}; "
                f"follow-up {_render_follow_up_label(item.appetite.follow_up)}; "
                f"proactivity {_render_proactivity_label(item.appetite.proactivity)})"
            )
        )
    return "\n".join(lines)
