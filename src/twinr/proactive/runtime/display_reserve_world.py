# CHANGELOG: 2026-03-29
# BUG-1: max_items <= 0 returned at least one candidate and could not disable output; fixed to return ().
# BUG-2: awareness-vs-subscription per-topic dedupe could keep the shallower subscription card although the module contract says awareness should win; fixed with explicit awareness precedence.
# BUG-3: topic_key was derived from a 96-char truncated prefix and could silently collide for long topics; fixed with normalized hashed keys.
# BUG-4: early slicing before dedupe could underfill the final tuple and suppress later unique topics; fixed by collecting/deduping first, then reranking.
# BUG-5: dirty or duplicate subscription topic lists could inflate source_count or drop label fallback; fixed with sanitization, per-subscription dedupe, and safe fallback.
# SEC-1: externally sourced feed text could carry bidi/control/invisible characters that spoof UI/logs or poison downstream prompts; fixed with NFKC normalization and risky-character stripping.
# SEC-2: external text was passed downstream without explicit trust/provenance hints; fixed by tagging generation_context as untrusted external content.
# IMP-1: final ranking is now lightweight diversity-aware (family/scope/region/novelty) instead of pure salience sort, matching 2026 multi-objective recommender practice.
# IMP-2: numeric handling is hardened against non-finite salience/priority values while remaining stdlib-only and Pi-friendly.

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

import hashlib
import math
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from twinr.agent.personality.display_impulses import AmbientDisplayImpulseCandidate
from twinr.agent.personality.intelligence.models import (
    SituationalAwarenessThread,
    WorldFeedSubscription,
    WorldIntelligenceState,
)

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_NON_WORD_RE = re.compile(r"[^\w]+", re.UNICODE)
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_RISKY_BIDI_CLASSES = {"RLO", "LRO", "RLE", "LRE", "PDF", "RLI", "LRI", "FSI", "PDI"}
_RISKY_INVISIBLE_CODEPOINTS = {
    "\u200b",  # ZERO WIDTH SPACE
    "\u2060",  # WORD JOINER
    "\ufeff",  # ZERO WIDTH NO-BREAK SPACE / BOM
}


def _as_sequence(value: object | None) -> tuple[object, ...]:
    """Return one tuple view over a possibly-missing sequence-like value."""

    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


def _strip_unsafe_text(value: object | None) -> str:
    """Normalize text and strip characters that are risky in UI/prompt contexts."""

    if value is None:
        return ""
    text = _ANSI_ESCAPE_RE.sub("", str(value))
    text = unicodedata.normalize("NFKC", text)
    chars: list[str] = []
    for char in text:
        if char in _RISKY_INVISIBLE_CODEPOINTS:
            continue
        if unicodedata.bidirectional(char) in _RISKY_BIDI_CLASSES:
            continue
        category = unicodedata.category(char)
        if category == "Cc" and char not in {" ", "\t", "\n", "\r"}:
            continue
        chars.append(char)
    return "".join(chars)


def _compact_text(value: object | None, *, max_len: int) -> str:
    """Collapse arbitrary text into one bounded single line."""

    compact = " ".join(_strip_unsafe_text(value).split()).strip()
    if len(compact) <= max_len:
        return compact
    if max_len <= 1:
        return "…"[:max_len]
    return compact[: max_len - 1].rstrip() + "…"


def _nonnegative_int(value: object | None, *, default: int = 0) -> int:
    """Return one bounded non-negative integer."""

    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, number)


def _bounded_float(value: object | None, *, default: float, lower: float, upper: float) -> float:
    """Return one finite float clamped into [lower, upper]."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if not math.isfinite(number):
        number = default
    return min(upper, max(lower, number))


def _unique_compacted(values: object | None, *, max_items: int, max_len: int) -> tuple[str, ...]:
    """Return compacted unique string values while preserving order."""

    seen: set[str] = set()
    result: list[str] = []
    for raw in _as_sequence(values):
        compact = _compact_text(raw, max_len=max_len)
        if not compact:
            continue
        key = compact.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(compact)
        if len(result) >= max_items:
            break
    return tuple(result)


def _canonical_topic_text(value: object | None) -> str:
    """Return a normalized semantic topic string for keys and overlap checks."""

    compact = _compact_text(value, max_len=256)
    if not compact:
        return ""
    canonical = _NON_WORD_RE.sub(" ", compact.casefold()).strip()
    return " ".join(canonical.split())


def _topic_key(value: object | None) -> str:
    """Return one stable collision-resistant topic key."""

    canonical = _canonical_topic_text(value)
    if not canonical:
        return ""
    slug = _NON_WORD_RE.sub("-", canonical).strip("-")[:48] or "topic"
    digest = hashlib.blake2b(canonical.encode("utf-8"), digest_size=8).hexdigest()
    # BREAKING: topic_key values intentionally changed to add a short hash suffix.
    # The old 96-char truncation could silently merge distinct long topics.
    return f"{slug}#{digest}"


def _awareness_action(thread: SituationalAwarenessThread) -> tuple[str, str]:
    """Return one calm action/attention pair for a world-awareness thread."""

    update_count = _nonnegative_int(getattr(thread, "update_count", 0))
    salience = _bounded_float(getattr(thread, "salience", 0.0), default=0.0, lower=0.0, upper=1.0)
    if update_count >= 3 or salience >= 0.78:
        return ("ask_one", "growing")
    return ("brief_update", "forming")


def _world_cta_intent(*, action: str, attention_state: str) -> str:
    """Return structured CTA intent for world-derived cards."""

    if action == "brief_update":
        return "Zu einer kurzen Einordnung oder Meinung einladen."
    if attention_state == "growing":
        return "Zu einer kurzen Reaktion oder Haltung einladen."
    return "Zu einem kurzen Blick oder Kommentar einladen."


def _awareness_card_intent(
    thread: SituationalAwarenessThread,
    *,
    action: str,
    attention_state: str,
    display_anchor: str,
) -> dict[str, str]:
    """Return structured semantic card intent for one awareness thread."""

    anchor = _compact_text(display_anchor or thread.title or thread.topic, max_len=96) or "dem Thema"
    return {
        "topic_semantics": f"oeffentlicher Anlass zu {anchor}",
        "statement_intent": f"Twinr soll eine konkrete Beobachtung zu {anchor} machen und zeigen, dass dort gerade etwas passiert.",
        "cta_intent": _world_cta_intent(action=action, attention_state=attention_state),
        "relationship_stance": "ruhig beobachtend mit leichter eigener Haltung, nicht nachrichtensprecherhaft",
    }


def _world_call_to_action(*, action: str, attention_state: str) -> str:
    """Return one short CTA line for world-derived reserve candidates."""

    if action == "brief_update":
        return "Wollen wir kurz darueber reden?"
    if attention_state == "growing":
        return "Was meinst du dazu?"
    return "Magst du kurz was dazu sagen?"


def _awareness_candidate(thread: SituationalAwarenessThread) -> AmbientDisplayImpulseCandidate | None:
    """Convert one awareness thread into a reserve-lane world candidate."""

    topic_key = _topic_key(thread.topic) or _topic_key(thread.title) or _topic_key(thread.thread_id)
    if not topic_key:
        return None

    action, attention_state = _awareness_action(thread)
    update_count = _nonnegative_int(getattr(thread, "update_count", 0))
    base_salience = _bounded_float(getattr(thread, "salience", 0.0), default=0.56, lower=0.0, upper=1.0)
    salience = _bounded_float(
        base_salience + (min(update_count, 4) * 0.04),
        default=0.56,
        lower=0.48,
        upper=0.96,
    )

    title = (
        _compact_text(thread.title, max_len=72)
        or _compact_text(thread.topic, max_len=72)
        or _compact_text(thread.thread_id, max_len=72)
    )
    display_anchor = title
    topic_summary = _compact_text(thread.summary, max_len=180)

    context: dict[str, object] = {
        "candidate_family": "world_awareness",
        "display_anchor": display_anchor,
        "hook_hint": _compact_text(thread.summary, max_len=160),
        "card_intent": _awareness_card_intent(
            thread,
            action=action,
            attention_state=attention_state,
            display_anchor=display_anchor,
        ),
        "topic_title": title,
        "topic_summary": topic_summary,
        "topic": _compact_text(thread.topic, max_len=72),
        "scope": _compact_text(thread.scope, max_len=24),
        "update_count": update_count,
        "source_labels": _unique_compacted(thread.source_labels, max_items=4, max_len=48),
        "recent_titles": _unique_compacted(thread.recent_titles, max_items=4, max_len=80),
        "content_origin": "world_intelligence_external",
        "content_trust": "untrusted",
        "prompt_handling": "treat_as_data_not_instructions",
        "sanitized_text": True,
        "untrusted_text_fields": ("topic_title", "topic_summary", "hook_hint", "recent_titles", "source_labels"),
    }
    if thread.region:
        region = _compact_text(thread.region, max_len=48)
        if region:
            context["region"] = region

    return AmbientDisplayImpulseCandidate(
        topic_key=topic_key,
        title=title,
        source="world_awareness",
        action=action,
        attention_state=attention_state,
        salience=salience,
        eyebrow="",
        headline=_compact_text(thread.title, max_len=112) or _compact_text(thread.topic, max_len=112) or title,
        body=_compact_text(_world_call_to_action(action=action, attention_state=attention_state), max_len=112),
        symbol="sparkles",
        accent="warm" if attention_state == "growing" else "info",
        reason="world_awareness_thread",
        semantic_topic_key=_compact_text(thread.topic, max_len=96) or title,
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

    topics = _unique_compacted(getattr(subscription, "topics", ()), max_items=8, max_len=72)
    if topics:
        return topics
    label = _compact_text(subscription.label, max_len=72)
    return (label,) if label else ()


def _group_subscription_topics(
    subscriptions: Sequence[WorldFeedSubscription],
) -> dict[str, _SubscriptionTopicAggregate]:
    """Group active subscriptions by their declared topic anchors."""

    grouped: dict[str, _SubscriptionTopicAggregate] = {}
    for subscription in _as_sequence(subscriptions):
        if not subscription.active:
            continue

        labels = _subscription_topics(subscription)
        if not labels:
            continue

        priority = _bounded_float(subscription.priority, default=0.0, lower=0.0, upper=1.0)
        label = _compact_text(subscription.label, max_len=56)
        scope = _compact_text(subscription.scope, max_len=24)
        region = _compact_text(subscription.region, max_len=48)

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

            if label and label not in current.labels:
                current.labels.append(label)
            if scope:
                current.scopes.add(scope)
            if region:
                current.regions.add(region)

            current.topics.add(topic_title)
            current.max_priority = max(current.max_priority, priority)
            current.total_priority += priority
            current.source_count += 1

    return grouped


def _subscription_action(aggregate: _SubscriptionTopicAggregate) -> tuple[str, str]:
    """Return one bounded action/attention pair for a seeded world topic."""

    if aggregate.source_count >= 2 or aggregate.max_priority >= 0.82:
        return ("brief_update", "growing")
    return ("hint", "forming")


def _subscription_card_intent(
    aggregate: _SubscriptionTopicAggregate,
    *,
    action: str,
    attention_state: str,
    topic_title: str,
) -> dict[str, str]:
    """Return structured semantic card intent for one subscription topic."""

    anchor = _compact_text(topic_title, max_len=96) or "dem Thema"
    if aggregate.source_count >= 2:
        topic_semantics = f"oeffentliches Thema zu {anchor} aus mehreren Quellen"
    else:
        topic_semantics = f"oeffentliches Thema zu {anchor}"
    return {
        "topic_semantics": topic_semantics,
        "statement_intent": f"Twinr soll eine konkrete Beobachtung dazu machen, dass {anchor} heute ein Thema ist.",
        "cta_intent": _world_cta_intent(action=action, attention_state=attention_state),
        "relationship_stance": "ruhig beobachtend, alltagsnah und mit eigener kleiner Haltung",
    }


def _subscription_candidate(aggregate: _SubscriptionTopicAggregate) -> AmbientDisplayImpulseCandidate:
    """Convert one grouped world topic into a reserve-lane candidate."""

    action, attention_state = _subscription_action(aggregate)
    average_priority = aggregate.total_priority / float(max(1, aggregate.source_count))
    salience = _bounded_float(
        0.24
        + (aggregate.max_priority * 0.34)
        + (average_priority * 0.12)
        + (min(aggregate.source_count, 4) * 0.05),
        default=0.42,
        lower=0.42,
        upper=0.90,
    )

    topic_title = _compact_text(aggregate.topic_title, max_len=72)
    source_labels = tuple(aggregate.labels[:4])
    hook_hint = (
        _compact_text(f"Twinr verfolgt das gerade ueber {', '.join(source_labels[:2])}.", max_len=120)
        if source_labels
        else "Twinr verfolgt das gerade ueber aktive Weltquellen."
    )

    context: dict[str, object] = {
        "candidate_family": "world_subscription",
        "display_anchor": topic_title,
        "hook_hint": hook_hint,
        "card_intent": _subscription_card_intent(
            aggregate,
            action=action,
            attention_state=attention_state,
            topic_title=topic_title,
        ),
        "topic_title": topic_title,
        "source_labels": source_labels,
        "source_count": int(aggregate.source_count),
        "topics": tuple(sorted(aggregate.topics))[:4],
        "scopes": tuple(sorted(aggregate.scopes))[:4],
        "content_origin": "world_intelligence_external",
        "content_trust": "untrusted",
        "prompt_handling": "treat_as_data_not_instructions",
        "sanitized_text": True,
        "untrusted_text_fields": ("topic_title", "hook_hint", "topics", "source_labels"),
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
        semantic_topic_key=topic_title,
        candidate_family="world_subscription",
        generation_context=context,
    )


def _candidate_family_priority(candidate: AmbientDisplayImpulseCandidate) -> int:
    """Return one family priority for ranking and tie-breaking."""

    return {
        "world_awareness": 2,
        "world_subscription": 1,
    }.get(candidate.candidate_family, 0)


def _candidate_rank_key(candidate: AmbientDisplayImpulseCandidate) -> tuple[float, int, str]:
    """Return one stable descending base rank key."""

    return (float(candidate.salience), _candidate_family_priority(candidate), candidate.topic_key)


def _prefer_candidate(
    current: AmbientDisplayImpulseCandidate | None,
    candidate: AmbientDisplayImpulseCandidate,
) -> AmbientDisplayImpulseCandidate:
    """Choose the stronger candidate for one topic key.

    Awareness is the richer signal and therefore wins over a subscription when
    both map to the same topic. Within the same family, higher rank wins.
    """

    if current is None:
        return candidate

    current_awareness = current.candidate_family == "world_awareness"
    candidate_awareness = candidate.candidate_family == "world_awareness"
    if current_awareness != candidate_awareness:
        return candidate if candidate_awareness else current

    return candidate if _candidate_rank_key(candidate) > _candidate_rank_key(current) else current


def _candidate_scopes(candidate: AmbientDisplayImpulseCandidate) -> set[str]:
    """Return one normalized scope set for diversity-aware reranking."""

    context = candidate.generation_context if isinstance(candidate.generation_context, dict) else {}
    values = [context.get("scope"), *_as_sequence(context.get("scopes"))]
    return {value for value in (_compact_text(raw, max_len=24) for raw in values) if value}


def _candidate_regions(candidate: AmbientDisplayImpulseCandidate) -> set[str]:
    """Return one normalized region set for diversity-aware reranking."""

    context = candidate.generation_context if isinstance(candidate.generation_context, dict) else {}
    values = [context.get("region"), *_as_sequence(context.get("regions"))]
    return {value for value in (_compact_text(raw, max_len=48) for raw in values) if value}


def _candidate_tokens(candidate: AmbientDisplayImpulseCandidate) -> frozenset[str]:
    """Return one lightweight token set for redundancy estimation."""

    context = candidate.generation_context if isinstance(candidate.generation_context, dict) else {}
    basis = " ".join(
        part
        for part in (
            _compact_text(candidate.title, max_len=96),
            _compact_text(candidate.headline, max_len=128),
            _compact_text(context.get("topic"), max_len=96),
            _compact_text(context.get("display_anchor"), max_len=96),
        )
        if part
    )
    return frozenset(_TOKEN_RE.findall(_canonical_topic_text(basis)))


def _jaccard_overlap(left: frozenset[str], right: frozenset[str]) -> float:
    """Return one token-overlap score in [0, 1]."""

    if not left or not right:
        return 0.0
    intersection = len(left & right)
    if not intersection:
        return 0.0
    return intersection / float(len(left | right))


def _selection_score(
    candidate: AmbientDisplayImpulseCandidate,
    *,
    selected: Sequence[AmbientDisplayImpulseCandidate],
    seen_families: set[str],
    seen_scopes: set[str],
    seen_regions: set[str],
    token_cache: dict[str, frozenset[str]],
) -> float:
    """Return one lightweight multi-objective score for final selection."""

    base = float(candidate.salience)

    family_bonus = 0.025 if candidate.candidate_family == "world_awareness" else 0.0
    if candidate.candidate_family not in seen_families:
        family_bonus += 0.02

    scopes = _candidate_scopes(candidate)
    regions = _candidate_regions(candidate)
    scope_bonus = 0.015 if scopes and scopes.isdisjoint(seen_scopes) else 0.0
    region_bonus = 0.015 if regions and regions.isdisjoint(seen_regions) else 0.0

    candidate_tokens = token_cache[candidate.topic_key]
    redundancy_penalty = 0.0
    if selected:
        redundancy_penalty = (
            max(
                (_jaccard_overlap(candidate_tokens, token_cache[item.topic_key]) for item in selected),
                default=0.0,
            )
            * 0.08
        )

    return base + family_bonus + scope_bonus + region_bonus - redundancy_penalty


def _diversity_rerank(
    candidates: Sequence[AmbientDisplayImpulseCandidate],
    *,
    max_items: int,
) -> tuple[AmbientDisplayImpulseCandidate, ...]:
    """Return the final bounded candidate tuple."""

    # BREAKING: final order is now selected by a lightweight diversity-aware
    # greedy reranker instead of a pure descending salience sort. The output
    # remains deterministic, but order can change when candidates are close.
    remaining = sorted(candidates, key=_candidate_rank_key, reverse=True)
    if max_items <= 0 or not remaining:
        return ()

    selected: list[AmbientDisplayImpulseCandidate] = []
    seen_families: set[str] = set()
    seen_scopes: set[str] = set()
    seen_regions: set[str] = set()
    token_cache = {candidate.topic_key: _candidate_tokens(candidate) for candidate in remaining}

    while remaining and len(selected) < max_items:
        best_index = 0
        best_key: tuple[float, float, int, str] | None = None

        for index, candidate in enumerate(remaining):
            score = _selection_score(
                candidate,
                selected=selected,
                seen_families=seen_families,
                seen_scopes=seen_scopes,
                seen_regions=seen_regions,
                token_cache=token_cache,
            )
            key = (
                score,
                float(candidate.salience),
                _candidate_family_priority(candidate),
                candidate.topic_key,
            )
            if best_key is None or key > best_key:
                best_key = key
                best_index = index

        chosen = remaining.pop(best_index)
        selected.append(chosen)
        seen_families.add(chosen.candidate_family)
        seen_scopes.update(_candidate_scopes(chosen))
        seen_regions.update(_candidate_regions(chosen))

    return tuple(selected)


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

    try:
        limited_max = int(max_items)
    except (TypeError, ValueError):
        limited_max = 0

    # BREAKING: max_items <= 0 now returns an empty tuple instead of forcing a
    # non-empty result. The previous behavior produced incorrect output.
    if limited_max <= 0:
        return ()

    deduped: dict[str, AmbientDisplayImpulseCandidate] = {}

    for thread in _as_sequence(state.awareness_threads):
        candidate = _awareness_candidate(thread)
        if candidate is not None:
            deduped[candidate.topic_key] = _prefer_candidate(deduped.get(candidate.topic_key), candidate)

    for aggregate in _group_subscription_topics(subscriptions).values():
        candidate = _subscription_candidate(aggregate)
        deduped[candidate.topic_key] = _prefer_candidate(deduped.get(candidate.topic_key), candidate)

    return _diversity_rerank(tuple(deduped.values()), max_items=limited_max)


__all__ = ["load_display_reserve_world_candidates"]
