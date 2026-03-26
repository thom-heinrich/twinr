"""Backfill reserve-lane topics from latent personality snapshot state.

The visible reserve lane already receives strong candidates from explicit
world, memory, reflection, and discovery loaders. On sparse days the
personality snapshot still carries additional grounded topics that are real but
may not pass the stricter positive-engagement gate used by the live ambient
impulse path.

This module turns those latent snapshot topics into extra reserve seeds:

- active continuity threads that are still relevant but not surfaced elsewhere
- durable relationship affinities that can reopen a personal thread
- concrete place focuses that add local breadth without collapsing to one place
- snapshot world signals that add concrete public-topic variety beyond grouped
  subscription anchors

The output remains generic, bounded, and topic-grounded. It is a reserve-side
backfill path, not a second positive-engagement policy engine.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from twinr.agent.personality.display_impulses import AmbientDisplayImpulseCandidate
from twinr.agent.personality.intelligence.models import WorldInterestSignal
from twinr.agent.personality.models import (
    ContinuityThread,
    PersonalitySnapshot,
    PlaceFocus,
    RelationshipSignal,
    WorldSignal,
)

from .display_reserve_diversity import select_diverse_candidates
from .display_reserve_support import compact_text

_ASK_ONE_SALIENCE = 0.78
_BRIEF_UPDATE_SALIENCE = 0.60
_LOCAL_WORLD_SOURCES = frozenset({"local_news", "regional_news", "place"})
_DEFAULT_PLACE_LIMIT = 3
_SNAPSHOT_SOURCE_PRIORITY = {
    "continuity": 4,
    "relationship": 3,
    "place": 2,
}


def _topic_key(value: object | None) -> str:
    """Return one normalized semantic key."""

    return compact_text(value, max_len=96).casefold()


def _truncate_text(value: object | None, *, max_len: int) -> str:
    """Return one bounded display-safe text field."""

    return compact_text(value, max_len=max_len)


def _engagement_states_by_topic(
    engagement_signals: Sequence[WorldInterestSignal],
) -> dict[str, str]:
    """Return the strongest known engagement state per semantic topic."""

    states: dict[str, tuple[int, str]] = {}
    priority = {
        "avoid": 4,
        "cooling": 3,
        "resonant": 2,
        "warm": 1,
        "uncertain": 0,
    }
    for signal in engagement_signals:
        topic = _topic_key(signal.topic)
        state = compact_text(signal.engagement_state, max_len=24).casefold() or "uncertain"
        if not topic:
            continue
        rank = priority.get(state, 0)
        current = states.get(topic)
        if current is None or rank > current[0]:
            states[topic] = (rank, state)
    return {topic: state for topic, (_rank, state) in states.items()}


def _is_avoided(semantic_topic_key: str, *, engagement_states: Mapping[str, str]) -> bool:
    """Return whether one topic is explicitly avoided."""

    return engagement_states.get(semantic_topic_key, "") == "avoid"


def _salience_with_engagement(
    base_salience: float,
    *,
    semantic_topic_key: str,
    engagement_states: Mapping[str, str],
    minimum: float,
    maximum: float,
) -> float:
    """Apply one small generic engagement adjustment to salience."""

    state = engagement_states.get(semantic_topic_key, "")
    adjusted = float(base_salience)
    if state == "cooling":
        adjusted -= 0.12
    elif state == "warm":
        adjusted += 0.04
    elif state == "resonant":
        adjusted += 0.08
    return min(maximum, max(minimum, adjusted))


def _engagement_attention_state(
    base_salience: float,
    *,
    semantic_topic_key: str,
    engagement_states: Mapping[str, str],
) -> tuple[str, str]:
    """Return one bounded action/attention pair for a latent topic."""

    state = engagement_states.get(semantic_topic_key, "")
    if state in {"warm", "resonant"} or base_salience >= _ASK_ONE_SALIENCE:
        return ("ask_one", "shared_thread")
    if base_salience >= _BRIEF_UPDATE_SALIENCE:
        return ("brief_update", "growing")
    return ("hint", "forming")


def _continuity_card_intent(anchor: str) -> dict[str, str]:
    """Return semantic card intent for one continuity-style reserve seed."""

    return {
        "topic_semantics": f"gemeinsamer Faden zu {anchor}",
        "statement_intent": f"Twinr soll ruhig zeigen, dass {anchor} noch als gemeinsamer Faden im Blick ist.",
        "cta_intent": "Zu einem kurzen Update, Weiterreden oder Einordnen einladen.",
        "relationship_stance": "warm, konkret und persoenlich statt meta",
    }


def _place_card_intent(anchor: str) -> dict[str, str]:
    """Return semantic card intent for one place-oriented reserve seed."""

    return {
        "topic_semantics": f"Ort oder Region {anchor} im aktuellen Blick",
        "statement_intent": f"Twinr soll eine konkrete Beobachtung dazu machen, dass {anchor} gerade relevant wirkt.",
        "cta_intent": "Zu einer kurzen Reaktion, Erinnerung oder alltaeglichen Einordnung einladen.",
        "relationship_stance": "lokal, alltagsnah und ruhig",
    }


def _world_card_intent(anchor: str) -> dict[str, str]:
    """Return semantic card intent for one world-signal reserve seed."""

    return {
        "topic_semantics": f"oeffentliche Entwicklung zu {anchor}",
        "statement_intent": f"Twinr soll eine konkrete Beobachtung dazu machen, dass {anchor} gerade auffaellig oder relevant ist.",
        "cta_intent": "Zu einer kurzen Meinung, Haltung oder Einordnung einladen.",
        "relationship_stance": "ruhig beobachtend mit leichter eigener Haltung",
    }


def _continuity_copy(anchor: str, *, action: str) -> tuple[str, str]:
    """Return one fallback copy pair for a personal continuity seed."""

    headline = f"{anchor} ist bei mir noch im Blick."
    if action == "ask_one":
        return (headline, "Magst du mich kurz auf Stand bringen?")
    if action == "brief_update":
        return (headline, "Wollen wir kurz darauf schauen?")
    return (headline, "Magst du spaeter kurz dazu anknuepfen?")


def _place_copy(anchor: str, *, action: str) -> tuple[str, str]:
    """Return one fallback copy pair for a place-oriented seed."""

    headline = f"{anchor} ist bei mir gerade als Ort im Blick."
    if action == "ask_one":
        return (headline, "Wie fuehlt sich das dort fuer dich an?")
    if action == "brief_update":
        return (headline, "Magst du kurz draufschauen?")
    return (headline, "Ist das im Alltag gerade irgendwie praesent?")


def _world_copy(anchor: str, *, action: str) -> tuple[str, str]:
    """Return one fallback copy pair for a world-signal seed."""

    headline = f"Bei {anchor} ist mir gerade etwas Konkretes aufgefallen."
    if action == "ask_one":
        return (headline, "Was meinst du dazu?")
    if action == "brief_update":
        return (headline, "Magst du kurz draufschauen?")
    return (headline, "Wollen wir das spaeter kurz streifen?")


def _candidate_sort_key(candidate: AmbientDisplayImpulseCandidate) -> tuple[float, int, str, str]:
    """Return one stable rank key for latent snapshot candidates."""

    source = compact_text(candidate.source, max_len=32).casefold()
    return (
        float(candidate.salience),
        _SNAPSHOT_SOURCE_PRIORITY.get(source, 1),
        compact_text(candidate.attention_state, max_len=24).casefold(),
        candidate.semantic_key(),
    )


def _continuity_candidate(
    thread: ContinuityThread,
    *,
    engagement_states: Mapping[str, str],
) -> AmbientDisplayImpulseCandidate | None:
    """Convert one continuity thread into a reserve candidate."""

    semantic_topic_key = _topic_key(thread.title)
    if not semantic_topic_key or _is_avoided(semantic_topic_key, engagement_states=engagement_states):
        return None
    action, attention_state = _engagement_attention_state(
        float(thread.salience),
        semantic_topic_key=semantic_topic_key,
        engagement_states=engagement_states,
    )
    salience = _salience_with_engagement(
        float(thread.salience),
        semantic_topic_key=semantic_topic_key,
        engagement_states=engagement_states,
        minimum=0.42,
        maximum=0.92,
    )
    anchor = _truncate_text(thread.title, max_len=72)
    headline, body = _continuity_copy(anchor, action=action)
    return AmbientDisplayImpulseCandidate(
        topic_key=semantic_topic_key,
        semantic_topic_key=semantic_topic_key,
        title=anchor,
        source="continuity",
        action=action,
        attention_state=attention_state,
        salience=salience,
        eyebrow="",
        headline=_truncate_text(headline, max_len=112),
        body=_truncate_text(body, max_len=112),
        symbol="sparkles",
        accent="warm" if attention_state == "shared_thread" else "info",
        reason="snapshot_continuity_thread",
        candidate_family="snapshot_continuity",
        generation_context={
            "candidate_family": "snapshot_continuity",
            "display_anchor": anchor,
            "hook_hint": _truncate_text(thread.summary, max_len=160),
            "card_intent": _continuity_card_intent(anchor),
            "topic_title": anchor,
            "topic_summary": _truncate_text(thread.summary, max_len=180),
            "topic_origin": "personality_snapshot",
            "display_goal": "reopen_shared_thread",
            "updated_at": _truncate_text(thread.updated_at, max_len=40),
            "expires_at": _truncate_text(thread.expires_at, max_len=40),
        },
    )


def _relationship_candidate(
    signal: RelationshipSignal,
    *,
    engagement_states: Mapping[str, str],
) -> AmbientDisplayImpulseCandidate | None:
    """Convert one relationship affinity into a reserve candidate."""

    if signal.stance != "affinity":
        return None
    semantic_topic_key = _topic_key(signal.topic)
    if not semantic_topic_key or _is_avoided(semantic_topic_key, engagement_states=engagement_states):
        return None
    action, attention_state = _engagement_attention_state(
        float(signal.salience),
        semantic_topic_key=semantic_topic_key,
        engagement_states=engagement_states,
    )
    salience = _salience_with_engagement(
        float(signal.salience),
        semantic_topic_key=semantic_topic_key,
        engagement_states=engagement_states,
        minimum=0.40,
        maximum=0.88,
    )
    anchor = _truncate_text(signal.topic, max_len=72)
    headline, body = _continuity_copy(anchor, action=action)
    return AmbientDisplayImpulseCandidate(
        topic_key=semantic_topic_key,
        semantic_topic_key=semantic_topic_key,
        title=anchor,
        source="relationship",
        action=action,
        attention_state=attention_state,
        salience=salience,
        eyebrow="",
        headline=_truncate_text(headline, max_len=112),
        body=_truncate_text(body, max_len=112),
        symbol="sparkles",
        accent="warm" if attention_state == "shared_thread" else "info",
        reason="snapshot_relationship_signal",
        candidate_family="snapshot_relationship",
        generation_context={
            "candidate_family": "snapshot_relationship",
            "display_anchor": anchor,
            "hook_hint": _truncate_text(signal.summary, max_len=160),
            "card_intent": _continuity_card_intent(anchor),
            "topic_title": anchor,
            "topic_summary": _truncate_text(signal.summary, max_len=180),
            "topic_origin": "personality_snapshot",
            "display_goal": "reopen_shared_thread",
            "relationship_source": _truncate_text(signal.source, max_len=48),
            "updated_at": _truncate_text(signal.updated_at, max_len=40),
        },
    )


def _place_candidate(
    focus: PlaceFocus,
    *,
    engagement_states: Mapping[str, str],
) -> AmbientDisplayImpulseCandidate | None:
    """Convert one place focus into a reserve candidate."""

    semantic_topic_key = _topic_key(focus.name)
    if not semantic_topic_key or _is_avoided(semantic_topic_key, engagement_states=engagement_states):
        return None
    action, attention_state = _engagement_attention_state(
        float(focus.salience),
        semantic_topic_key=semantic_topic_key,
        engagement_states=engagement_states,
    )
    salience = _salience_with_engagement(
        float(focus.salience),
        semantic_topic_key=semantic_topic_key,
        engagement_states=engagement_states,
        minimum=0.38,
        maximum=0.82,
    )
    anchor = _truncate_text(focus.name, max_len=72)
    headline, body = _place_copy(anchor, action=action)
    return AmbientDisplayImpulseCandidate(
        topic_key=semantic_topic_key,
        semantic_topic_key=semantic_topic_key,
        title=anchor,
        source="place",
        action=action,
        attention_state=attention_state,
        salience=salience,
        eyebrow="",
        headline=_truncate_text(headline, max_len=112),
        body=_truncate_text(body, max_len=112),
        symbol="sparkles",
        accent="info",
        reason="snapshot_place_focus",
        candidate_family="snapshot_place",
        generation_context={
            "candidate_family": "snapshot_place",
            "display_anchor": anchor,
            "hook_hint": _truncate_text(focus.summary, max_len=160),
            "card_intent": _place_card_intent(anchor),
            "topic_title": anchor,
            "topic_summary": _truncate_text(focus.summary, max_len=180),
            "topic_origin": "personality_snapshot",
            "display_goal": "open_local_context",
            "geography": _truncate_text(focus.geography, max_len=48),
            "updated_at": _truncate_text(focus.updated_at, max_len=40),
        },
    )


def _world_candidate(
    signal: WorldSignal,
    *,
    engagement_states: Mapping[str, str],
) -> AmbientDisplayImpulseCandidate | None:
    """Convert one snapshot world signal into a reserve candidate."""

    semantic_topic_key = _topic_key(signal.topic)
    if not semantic_topic_key or _is_avoided(semantic_topic_key, engagement_states=engagement_states):
        return None
    action, attention_state = _engagement_attention_state(
        float(signal.salience),
        semantic_topic_key=semantic_topic_key,
        engagement_states=engagement_states,
    )
    salience = _salience_with_engagement(
        float(signal.salience) + (min(int(signal.evidence_count), 3) * 0.03),
        semantic_topic_key=semantic_topic_key,
        engagement_states=engagement_states,
        minimum=0.40,
        maximum=0.90,
    )
    anchor = _truncate_text(signal.topic, max_len=72)
    headline, body = _world_copy(anchor, action=action)
    return AmbientDisplayImpulseCandidate(
        topic_key=semantic_topic_key,
        semantic_topic_key=semantic_topic_key,
        title=anchor,
        source="situational_awareness",
        action=action,
        attention_state=attention_state,
        salience=salience,
        eyebrow="",
        headline=_truncate_text(headline, max_len=112),
        body=_truncate_text(body, max_len=112),
        symbol="sparkles",
        accent="info",
        reason="snapshot_world_signal",
        candidate_family="snapshot_world_signal",
        generation_context={
            "candidate_family": "snapshot_world_signal",
            "display_anchor": anchor,
            "hook_hint": _truncate_text(signal.summary, max_len=160),
            "card_intent": _world_card_intent(anchor),
            "topic_title": anchor,
            "topic_summary": _truncate_text(signal.summary, max_len=180),
            "topic_origin": "personality_snapshot",
            "display_goal": "open_public_topic",
            "source_kind": _truncate_text(signal.source, max_len=48),
            "region": _truncate_text(signal.region, max_len=48),
            "evidence_count": int(signal.evidence_count),
            "fresh_until": _truncate_text(signal.fresh_until, max_len=40),
        },
    )


def _best_by_semantic_topic(
    candidates: Sequence[AmbientDisplayImpulseCandidate],
    *,
    excluded_topics: set[str],
) -> tuple[AmbientDisplayImpulseCandidate, ...]:
    """Keep only the strongest latent candidate per semantic topic."""

    best: dict[str, AmbientDisplayImpulseCandidate] = {}
    for candidate in candidates:
        semantic_topic_key = candidate.semantic_key()
        if not semantic_topic_key or semantic_topic_key in excluded_topics:
            continue
        current = best.get(semantic_topic_key)
        if current is None or _candidate_sort_key(candidate) > _candidate_sort_key(current):
            best[semantic_topic_key] = candidate
    return tuple(
        sorted(
            best.values(),
            key=_candidate_sort_key,
            reverse=True,
        )
    )


def load_display_reserve_snapshot_candidates(
    snapshot: PersonalitySnapshot | None,
    *,
    engagement_signals: Sequence[WorldInterestSignal],
    exclude_topic_keys: Sequence[str] = (),
    max_items: int,
) -> tuple[AmbientDisplayImpulseCandidate, ...]:
    """Return bounded reserve candidates from latent snapshot topics."""

    if snapshot is None:
        return ()
    limited_max = max(1, int(max_items))
    excluded_topics = {_topic_key(value) for value in exclude_topic_keys if _topic_key(value)}
    engagement_states = _engagement_states_by_topic(engagement_signals)
    continuity_limit = max(6, limited_max)
    relationship_limit = max(4, (limited_max // 2) + 1)
    place_limit = _DEFAULT_PLACE_LIMIT
    world_limit = max(12, limited_max * 2)

    raw_candidates: list[AmbientDisplayImpulseCandidate] = []
    for thread in sorted(
        snapshot.continuity_threads,
        key=lambda item: (item.salience, item.updated_at or "", item.title),
        reverse=True,
    )[:continuity_limit]:
        candidate = _continuity_candidate(thread, engagement_states=engagement_states)
        if candidate is not None:
            raw_candidates.append(candidate)
    for relationship_signal in sorted(
        snapshot.relationship_signals,
        key=lambda item: (item.salience, item.updated_at or "", item.topic),
        reverse=True,
    )[:relationship_limit]:
        candidate = _relationship_candidate(
            relationship_signal,
            engagement_states=engagement_states,
        )
        if candidate is not None:
            raw_candidates.append(candidate)
    for focus in sorted(
        snapshot.place_focuses,
        key=lambda item: (item.salience, item.updated_at or "", item.name),
        reverse=True,
    )[:place_limit]:
        candidate = _place_candidate(focus, engagement_states=engagement_states)
        if candidate is not None:
            raw_candidates.append(candidate)
    for world_signal in sorted(
        snapshot.world_signals,
        key=lambda item: (
            item.source in _LOCAL_WORLD_SOURCES,
            item.salience,
            item.evidence_count,
            item.fresh_until or "",
            item.topic,
        ),
        reverse=True,
    )[:world_limit]:
        candidate = _world_candidate(
            world_signal,
            engagement_states=engagement_states,
        )
        if candidate is not None:
            raw_candidates.append(candidate)

    deduped = _best_by_semantic_topic(raw_candidates, excluded_topics=excluded_topics)
    if not deduped:
        return ()
    return select_diverse_candidates(
        deduped,
        max_items=min(limited_max, len(deduped)),
    )


__all__ = ["load_display_reserve_snapshot_candidates"]
