"""Convert durable memory hooks into reserve-lane candidate families.

This module keeps long-term-memory specific reserve-candidate derivation away
from the broader ambient companion flow. It translates two existing durable
memory hooks into display candidates:

- open conflict items that would benefit from calm clarification
- gentle proactive follow-ups that can deepen continuity

The output stays generic and structured so higher-level planners can mix these
families with personality and reflection candidates without topic hardcoding.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
import hashlib

from twinr.agent.personality.display_impulses import AmbientDisplayImpulseCandidate
from twinr.memory.longterm.core.models import (
    LongTermConflictQueueItemV1,
    LongTermProactiveCandidateV1,
)

_DEFAULT_MEMORY_CONFLICT_LIMIT = 2
_DEFAULT_MEMORY_FOLLOW_UP_LIMIT = 2


def _compact_text(value: object | None) -> str:
    """Collapse arbitrary text into one trimmed single line."""

    return " ".join(str(value or "").split()).strip()


def _truncate_text(value: object | None, *, max_len: int) -> str:
    """Return one bounded display-safe text field."""

    compact = _compact_text(value)
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "…"


def _topic_key(value: object | None) -> str:
    """Return one stable cooldown/dedupe key."""

    return _compact_text(value).casefold()


def _stable_fraction(*parts: object) -> float:
    """Return one deterministic 0..1 fraction for bounded variation."""

    digest = hashlib.sha1(
        "::".join(_compact_text(part) for part in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") / 4_294_967_295.0


def _ensure_question(text: object | None, *, fallback: str) -> str:
    """Return one normalized question-shaped headline."""

    compact = _truncate_text(text, max_len=112) or fallback
    if compact.endswith("?"):
        return compact
    return compact.rstrip(".! ") + "?"


def _pick_variant(seed: float, variants: tuple[str, ...]) -> str:
    """Pick one deterministic variant from a bounded tuple."""

    if not variants:
        return ""
    index = min(len(variants) - 1, int(seed * len(variants)))
    return variants[index]


def _memory_conflict_body(item: LongTermConflictQueueItemV1) -> str:
    """Render one calm helper line for a memory-clarification question."""

    seed = _stable_fraction(item.slot_key, item.question, item.reason)
    if len(item.options) >= 2:
        variants = (
            "Da klingen fuer mich gerade noch zwei Versionen gleichzeitig an.",
            "Ich habe dazu im Moment noch zwei unterschiedliche Bilder im Kopf.",
        )
    else:
        variants = (
            "Da ist fuer mich noch ein kleines Fragezeichen offen.",
            "Da wuerde mir ein kurzer Anker helfen.",
        )
    return _truncate_text(_pick_variant(seed, variants), max_len=112)


def _follow_up_headline(candidate: LongTermProactiveCandidateV1) -> str:
    """Render one generic question-first headline for a memory follow-up."""

    seed = _stable_fraction(candidate.candidate_id, candidate.summary, candidate.confidence)
    if candidate.confidence >= 0.78:
        variants = (
            "Wie ist es damit weitergegangen?",
            "Magst du mir dazu noch etwas erzählen?",
            "Ist da inzwischen etwas klarer geworden?",
        )
    else:
        variants = (
            "Soll ich mir das etwas genauer merken?",
            "Magst du mir dazu spaeter noch etwas sagen?",
            "Willst du mir dazu noch einen kleinen Hinweis geben?",
        )
    return _ensure_question(_pick_variant(seed, variants), fallback="Magst du mir dazu noch etwas erzählen?")


def _clean_follow_up_summary(value: object | None) -> str:
    """Return one display-safe context line for a gentle memory follow-up."""

    summary = _compact_text(value)
    prefix = "If relevant, gently follow up on:"
    if summary.startswith(prefix):
        summary = summary.removeprefix(prefix).strip()
    return _truncate_text(summary, max_len=96)


def _follow_up_body(candidate: LongTermProactiveCandidateV1) -> str:
    """Render one calm helper/context line for a gentle follow-up."""

    seed = _stable_fraction(candidate.candidate_id, candidate.summary, candidate.rationale)
    context = _clean_follow_up_summary(candidate.summary)
    helper = _pick_variant(
        seed,
        (
            "Das ist mir gerade wieder eingefallen.",
            "Da bin ich innerlich noch nicht ganz fertig damit.",
            "Ich haenge da gedanklich noch ein wenig dran.",
        ),
    )
    if context:
        return _truncate_text(f"{context} {helper}", max_len=112)
    return _truncate_text(helper, max_len=112)


def _conflict_generation_context(item: LongTermConflictQueueItemV1) -> dict[str, object]:
    """Return generic structured context for LLM-written conflict copy."""

    options: list[dict[str, object]] = []
    for option in item.options[:3]:
        option_payload = {
            "summary": _truncate_text(option.summary, max_len=120),
            "status": _truncate_text(option.status, max_len=24),
        }
        if _compact_text(option.details):
            option_payload["details"] = _truncate_text(option.details, max_len=120)
        if _compact_text(option.value_key):
            option_payload["value_key"] = _truncate_text(option.value_key, max_len=80)
        options.append(option_payload)
    return {
        "candidate_family": "memory_conflict",
        "memory_goal": "clarify_conflict",
        "display_anchor": _truncate_text(item.question, max_len=120),
        "hook_hint": _truncate_text(item.reason, max_len=140),
        "question": _truncate_text(item.question, max_len=140),
        "reason": _truncate_text(item.reason, max_len=180),
        "options": options,
    }


def _follow_up_generation_context(candidate: LongTermProactiveCandidateV1) -> dict[str, object]:
    """Return generic structured context for LLM-written follow-up copy."""

    payload: dict[str, object] = {
        "candidate_family": "memory_follow_up",
        "memory_goal": "gentle_follow_up",
        "display_anchor": _truncate_text(_clean_follow_up_summary(candidate.summary), max_len=120),
        "hook_hint": _truncate_text(candidate.rationale, max_len=140),
        "summary": _truncate_text(_clean_follow_up_summary(candidate.summary), max_len=140),
        "rationale": _truncate_text(candidate.rationale, max_len=180),
        "confidence": round(float(candidate.confidence), 3),
        "sensitivity": _truncate_text(candidate.sensitivity, max_len=24),
    }
    if _compact_text(candidate.due_date):
        payload["due_date"] = _truncate_text(candidate.due_date, max_len=32)
    return payload


def build_memory_conflict_candidate(
    item: LongTermConflictQueueItemV1,
) -> AmbientDisplayImpulseCandidate | None:
    """Convert one durable conflict queue item into a reserve-lane question."""

    topic_key = _topic_key(item.slot_key) or _topic_key(item.question)
    if not topic_key:
        return None
    return AmbientDisplayImpulseCandidate(
        topic_key=topic_key,
        title=_compact_text(item.question) or item.slot_key,
        source="memory_conflict",
        action="ask_one",
        attention_state="forming",
        salience=0.98,
        eyebrow="",
        headline=_ensure_question(
            item.question,
            fallback="Magst du mir dabei kurz helfen?",
        ),
        body=_memory_conflict_body(item),
        symbol="question",
        accent="warm",
        reason="memory_conflict_clarify",
        candidate_family="memory_conflict",
        generation_context=_conflict_generation_context(item),
    )


def build_memory_follow_up_candidate(
    candidate: LongTermProactiveCandidateV1,
) -> AmbientDisplayImpulseCandidate | None:
    """Convert one gentle memory follow-up into a reserve-lane impulse."""

    if candidate.kind != "gentle_follow_up":
        return None
    if candidate.sensitivity == "critical":
        return None
    topic_key = _topic_key(candidate.candidate_id) or _topic_key(candidate.summary)
    if not topic_key:
        return None
    return AmbientDisplayImpulseCandidate(
        topic_key=topic_key,
        title=_clean_follow_up_summary(candidate.summary) or candidate.candidate_id,
        source="memory_follow_up",
        action="ask_one" if float(candidate.confidence) >= 0.72 else "hint",
        attention_state="forming",
        salience=min(0.95, 0.46 + float(candidate.confidence) * 0.42),
        eyebrow="",
        headline=_follow_up_headline(candidate),
        body=_follow_up_body(candidate),
        symbol="question",
        accent="warm" if float(candidate.confidence) >= 0.72 else "info",
        reason=_truncate_text(candidate.rationale, max_len=120) or "memory_follow_up",
        candidate_family="memory_follow_up",
        generation_context=_follow_up_generation_context(candidate),
    )


@dataclass(frozen=True, slots=True)
class DisplayReserveMemoryCandidateSet:
    """Collect memory-derived reserve candidates for one planning pass."""

    conflicts: tuple[AmbientDisplayImpulseCandidate, ...] = ()
    follow_ups: tuple[AmbientDisplayImpulseCandidate, ...] = ()

    def as_tuple(self) -> tuple[AmbientDisplayImpulseCandidate, ...]:
        """Return the candidates as one flat ordered tuple."""

        return tuple((*self.conflicts, *self.follow_ups))


def load_display_reserve_memory_candidates(
    *,
    conflicts: Sequence[LongTermConflictQueueItemV1],
    proactive_candidates: Sequence[LongTermProactiveCandidateV1],
    max_items: int,
) -> DisplayReserveMemoryCandidateSet:
    """Convert memory conflicts and follow-ups into reserve candidates."""

    limited_max = max(1, int(max_items))
    conflict_limit = min(limited_max, _DEFAULT_MEMORY_CONFLICT_LIMIT)
    follow_up_limit = min(limited_max, _DEFAULT_MEMORY_FOLLOW_UP_LIMIT)
    conflict_candidates = tuple(
        candidate
        for item in conflicts[:conflict_limit]
        if (candidate := build_memory_conflict_candidate(item)) is not None
    )
    follow_candidates: list[AmbientDisplayImpulseCandidate] = []
    for item in proactive_candidates:
        if len(follow_candidates) >= follow_up_limit:
            break
        candidate = build_memory_follow_up_candidate(item)
        if candidate is None:
            continue
        follow_candidates.append(candidate)
    return DisplayReserveMemoryCandidateSet(
        conflicts=conflict_candidates,
        follow_ups=tuple(follow_candidates),
    )


__all__ = [
    "DisplayReserveMemoryCandidateSet",
    "build_memory_conflict_candidate",
    "build_memory_follow_up_candidate",
    "load_display_reserve_memory_candidates",
]
