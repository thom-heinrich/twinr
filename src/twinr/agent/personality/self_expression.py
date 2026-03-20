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

from dataclasses import dataclass
import hashlib
import random

from twinr.agent.personality.models import PersonalitySnapshot


@dataclass(frozen=True, slots=True)
class CompanionMindshareItem:
    """Describe one prompt-facing topic Twinr may naturally speak from."""

    title: str
    summary: str
    salience: float
    source: str


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


def _stable_rng(snapshot: PersonalitySnapshot | None, items: tuple[CompanionMindshareItem, ...]) -> random.Random:
    """Build one deterministic RNG for reproducible mindshare surfacing.

    The seed depends on persisted snapshot state, not on specific entity names.
    That keeps the selection data-driven and testable while still allowing
    gentle variation as the snapshot evolves.
    """

    parts = [snapshot.generated_at if snapshot else ""]
    parts.extend(
        f"{item.source}|{_normalized_text(item.title)}|{item.salience:.4f}"
        for item in items
    )
    digest = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _dedupe_candidates(
    candidates: tuple[CompanionMindshareItem, ...],
) -> tuple[CompanionMindshareItem, ...]:
    """Keep one strongest candidate per normalized title."""

    best_by_key: dict[str, tuple[float, CompanionMindshareItem]] = {}
    for item in candidates:
        key = _mindshare_key(item)
        score = _base_selection_score(item)
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
                summary=f"Twinr has been keeping an eye on this thread. {thread.summary}",
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
                summary=f"This is part of Twinr's durable attention. {signal.summary}",
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
) -> tuple[CompanionMindshareItem, ...]:
    """Select the small set of ongoing topics Twinr may speak from naturally.

    Selection stays intentionally conservative, but not entity-specific:
    - the candidate pool is built from place, continuity, relationship, and
      world-awareness state
    - candidate weights depend on source type and salience, never on named
      entities such as one city
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

    deduped = _dedupe_candidates(tuple(candidate_items))
    if len(deduped) <= limited_max:
        return deduped

    rng = _stable_rng(snapshot, deduped)
    ranked = sorted(
        deduped,
        key=lambda item: (
            _base_selection_score(item) + (rng.random() * 0.08),
            item.salience,
            item.title,
        ),
        reverse=True,
    )
    return tuple(ranked[:limited_max])


def render_self_expression_policy(snapshot: PersonalitySnapshot | None) -> str | None:
    """Render how Twinr may speak from its current mindshare in conversation."""

    if not build_mindshare_items(snapshot):
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
    ]
    return "\n".join(lines)


def render_mindshare_block(snapshot: PersonalitySnapshot | None) -> str | None:
    """Render the current prompt-facing mindshare section for conversation."""

    items = build_mindshare_items(snapshot)
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
        lines.append(
            f"- {item.title}: {item.summary} (salience {_format_score(item.salience)}, source {item.source})"
        )
    return "\n".join(lines)
