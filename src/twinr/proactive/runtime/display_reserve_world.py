"""Convert world-intelligence state into right-lane reserve candidates.

The right reserve lane should not only depend on already-collapsed prompt-time
mindshare. Twinr's RSS/world-intelligence layer carries a richer slow-moving
state:

- active feed subscriptions chosen by the installer or by later recalibration
- condensed situational-awareness threads built from repeated feed refreshes

This module translates that state into generic ambient reserve candidates
without coupling the runtime planner to the internals of world-intelligence
refreshing. The output stays bounded and topic-generic:

- awareness threads surface as calmer, richer world-context candidates
- active subscriptions provide topic breadth when awareness threads are still
  sparse
- no named-topic hardcoding, regex routing, or prompt text is embedded here
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from twinr.agent.personality.display_impulses import AmbientDisplayImpulseCandidate
from twinr.agent.personality.intelligence.models import (
    SituationalAwarenessThread,
    WorldFeedSubscription,
    WorldIntelligenceState,
)

_DEFAULT_AWARENESS_LIMIT = 8
_DEFAULT_SUBSCRIPTION_LIMIT = 16


def _compact_text(value: object | None, *, max_len: int) -> str:
    """Collapse arbitrary text into one bounded single line."""

    if value is None:
        return ""
    compact = " ".join(str(value).split()).strip()
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "…"


def _topic_key(value: object | None) -> str:
    """Return one stable normalized topic key."""

    return _compact_text(value, max_len=96).casefold()


def _awareness_action(thread: SituationalAwarenessThread) -> tuple[str, str]:
    """Return one calm action/attention pair for a world-awareness thread."""

    if thread.update_count >= 3 or thread.salience >= 0.78:
        return ("ask_one", "growing")
    return ("brief_update", "forming")


def _awareness_candidate(thread: SituationalAwarenessThread) -> AmbientDisplayImpulseCandidate | None:
    """Convert one awareness thread into a reserve-lane world candidate."""

    topic_key = _topic_key(thread.topic) or _topic_key(thread.title) or _topic_key(thread.thread_id)
    if not topic_key:
        return None
    action, attention_state = _awareness_action(thread)
    salience = min(0.96, max(0.48, float(thread.salience) + (min(thread.update_count, 4) * 0.04)))
    context: dict[str, object] = {
        "candidate_family": "world_awareness",
        "display_anchor": _compact_text(thread.title or thread.topic, max_len=72),
        "hook_hint": _compact_text(thread.summary, max_len=160),
        "topic_title": _compact_text(thread.title, max_len=72),
        "topic_summary": _compact_text(thread.summary, max_len=180),
        "topic": _compact_text(thread.topic, max_len=72),
        "scope": _compact_text(thread.scope, max_len=24),
        "update_count": int(thread.update_count),
        "source_labels": tuple(_compact_text(value, max_len=48) for value in thread.source_labels[:4]),
        "recent_titles": tuple(_compact_text(value, max_len=80) for value in thread.recent_titles[:4]),
    }
    if thread.region:
        context["region"] = _compact_text(thread.region, max_len=48)
    return AmbientDisplayImpulseCandidate(
        topic_key=topic_key,
        title=_compact_text(thread.title, max_len=72) or _compact_text(thread.topic, max_len=72),
        source="world_awareness",
        action=action,
        attention_state=attention_state,
        salience=salience,
        eyebrow="",
        headline=_compact_text(thread.title, max_len=112),
        body=_compact_text(_world_call_to_action(action=action, attention_state=attention_state), max_len=112),
        symbol="sparkles",
        accent="warm" if attention_state == "growing" else "info",
        reason="world_awareness_thread",
        candidate_family="world_awareness",
        generation_context=context,
    )


@dataclass(slots=True)
class _SubscriptionTopicAggregate:
    """Aggregate multiple active subscriptions that cover one semantic topic."""

    topic_key: str
    topic_title: str
    labels: list[str]
    scopes: set[str]
    regions: set[str]
    topics: set[str]
    max_priority: float
    total_priority: float
    source_count: int


def _subscription_topics(subscription: WorldFeedSubscription) -> tuple[str, ...]:
    """Return the semantic topic anchors for one active subscription."""

    if subscription.topics:
        return tuple(
            _compact_text(value, max_len=72)
            for value in subscription.topics
            if _compact_text(value, max_len=72)
        )
    label = _compact_text(subscription.label, max_len=72)
    return (label,) if label else ()


def _group_subscription_topics(
    subscriptions: Sequence[WorldFeedSubscription],
) -> dict[str, _SubscriptionTopicAggregate]:
    """Group active subscriptions by their declared topic anchors."""

    grouped: dict[str, _SubscriptionTopicAggregate] = {}
    for subscription in subscriptions:
        if not subscription.active:
            continue
        labels = _subscription_topics(subscription)
        if not labels:
            continue
        for topic_title in labels[:3]:
            key = _topic_key(topic_title)
            if not key:
                continue
            current = grouped.get(key)
            if current is None:
                current = _SubscriptionTopicAggregate(
                    topic_key=key,
                    topic_title=topic_title,
                    labels=[],
                    scopes=set(),
                    regions=set(),
                    topics=set(),
                    max_priority=0.0,
                    total_priority=0.0,
                    source_count=0,
                )
                grouped[key] = current
            label = _compact_text(subscription.label, max_len=56)
            if label and label not in current.labels:
                current.labels.append(label)
            if subscription.scope:
                current.scopes.add(_compact_text(subscription.scope, max_len=24))
            if subscription.region:
                current.regions.add(_compact_text(subscription.region, max_len=48))
            current.topics.add(topic_title)
            current.max_priority = max(current.max_priority, float(subscription.priority))
            current.total_priority += float(subscription.priority)
            current.source_count += 1
    return grouped


def _subscription_action(aggregate: _SubscriptionTopicAggregate) -> tuple[str, str]:
    """Return one bounded action/attention pair for a seeded world topic."""

    if aggregate.source_count >= 2 or aggregate.max_priority >= 0.82:
        return ("brief_update", "growing")
    return ("hint", "forming")


def _world_call_to_action(*, action: str, attention_state: str) -> str:
    """Return one short CTA line for world-derived reserve candidates."""

    if action == "brief_update":
        return "Wollen wir kurz darueber reden?"
    if attention_state == "growing":
        return "Was meinst du dazu?"
    return "Magst du kurz was dazu sagen?"


def _subscription_candidate(aggregate: _SubscriptionTopicAggregate) -> AmbientDisplayImpulseCandidate:
    """Convert one grouped world topic into a reserve-lane candidate."""

    action, attention_state = _subscription_action(aggregate)
    average_priority = aggregate.total_priority / float(max(1, aggregate.source_count))
    salience = min(
        0.9,
        max(
            0.42,
            0.24
            + (aggregate.max_priority * 0.34)
            + (average_priority * 0.12)
            + (min(aggregate.source_count, 4) * 0.05),
        ),
    )
    topic_title = _compact_text(aggregate.topic_title, max_len=72)
    source_labels = tuple(aggregate.labels[:4])
    context: dict[str, object] = {
        "candidate_family": "world_subscription",
        "display_anchor": topic_title,
        "hook_hint": _compact_text(
            f"Twinr verfolgt das gerade ueber {', '.join(source_labels[:2])}.",
            max_len=120,
        ),
        "topic_title": topic_title,
        "source_labels": source_labels,
        "source_count": int(aggregate.source_count),
        "topics": tuple(sorted(aggregate.topics))[:4],
        "scopes": tuple(sorted(aggregate.scopes))[:4],
    }
    if aggregate.regions:
        context["regions"] = tuple(sorted(aggregate.regions))[:4]
    return AmbientDisplayImpulseCandidate(
        topic_key=aggregate.topic_key,
        title=topic_title,
        source="world_subscription",
        action=action,
        attention_state=attention_state,
        salience=salience,
        eyebrow="",
        headline=_compact_text(f"Bei {topic_title} ist gerade etwas in Bewegung.", max_len=112),
        body=_compact_text(_world_call_to_action(action=action, attention_state=attention_state), max_len=112),
        symbol="sparkles",
        accent="info",
        reason="world_subscription_seed",
        candidate_family="world_subscription",
        generation_context=context,
    )


def _candidate_rank_key(candidate: AmbientDisplayImpulseCandidate) -> tuple[float, int, str]:
    """Return one stable descending rank key for world-derived candidates."""

    family_priority = {
        "world_awareness": 3,
        "world_subscription": 2,
    }.get(candidate.candidate_family, 0)
    return (float(candidate.salience), family_priority, candidate.topic_key)


def load_display_reserve_world_candidates(
    *,
    subscriptions: Sequence[WorldFeedSubscription],
    state: WorldIntelligenceState,
    max_items: int,
) -> tuple[AmbientDisplayImpulseCandidate, ...]:
    """Load world-derived reserve candidates from subscriptions and awareness.

    Awareness threads are the richer signal and therefore win per-topic dedupe.
    Active subscriptions provide topic breadth when the condensed awareness
    layer is still sparse. The right lane stays topic-led on purpose; raw
    source-label cards are suppressed because they read like feed internals
    instead of Twinr's own conversational openings.
    """

    limited_max = max(1, int(max_items))
    candidates: list[AmbientDisplayImpulseCandidate] = []
    for thread in state.awareness_threads[: max(limited_max, _DEFAULT_AWARENESS_LIMIT)]:
        candidate = _awareness_candidate(thread)
        if candidate is not None:
            candidates.append(candidate)
    grouped = _group_subscription_topics(subscriptions)
    candidates.extend(
        sorted(
            (_subscription_candidate(item) for item in grouped.values()),
            key=_candidate_rank_key,
            reverse=True,
        )[: max(limited_max, _DEFAULT_SUBSCRIPTION_LIMIT)]
    )
    deduped: dict[str, AmbientDisplayImpulseCandidate] = {}
    for candidate in candidates:
        current = deduped.get(candidate.topic_key)
        if current is None or _candidate_rank_key(candidate) > _candidate_rank_key(current):
            deduped[candidate.topic_key] = candidate
    ranked = sorted(deduped.values(), key=_candidate_rank_key, reverse=True)
    return tuple(ranked[:limited_max])


__all__ = ["load_display_reserve_world_candidates"]
