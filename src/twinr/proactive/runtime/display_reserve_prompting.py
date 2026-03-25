"""Shape compact reserve-card prompt payloads for the LLM rewrite step.

The reserve-copy generator should not pass full raw candidate context into one
large prompt. This module turns rich structured reserve candidates into a much
smaller prompt contract with two explicit user-facing semantics:

- ``topic_anchor``: what the card is about in a clear glanceable way
- ``hook_hint``: the concrete angle, follow-up, or tension to write from

Everything else is compressed into one short ``context_summary`` string so the
LLM has enough context to sound like Twinr without dragging around large nested
payloads from world, memory, or reflection sources.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
import json

from twinr.agent.personality.display_impulses import AmbientDisplayImpulseCandidate
from twinr.agent.personality.models import PersonalitySnapshot

_DEFAULT_CONTEXT_SUMMARY_MAX_CHARS = 220
_DEFAULT_LIST_ITEMS = 3


def _compact_text(value: object | None) -> str:
    """Collapse arbitrary text into one compact single line."""

    return " ".join(str(value or "").split()).strip()


def _truncate_text(value: object | None, *, max_len: int) -> str:
    """Return one bounded single-line string."""

    compact = _compact_text(value)
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "…"


def _coerce_mapping(value: object | None) -> Mapping[str, object]:
    """Return one mapping or an empty mapping."""

    return value if isinstance(value, Mapping) else {}


def _value_text(value: object | None, *, max_len: int) -> str:
    """Render one bounded text fragment from scalar, list, or mapping input."""

    if isinstance(value, Mapping):
        parts: list[str] = []
        for inner_key in ("title", "summary", "details", "value_key", "status"):
            text = _truncate_text(value.get(inner_key), max_len=max_len)
            if text:
                parts.append(text)
        return _truncate_text(" | ".join(parts), max_len=max_len)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parts = [
            _truncate_text(item, max_len=max_len // 2 if max_len > 24 else max_len)
            for item in list(value)[:_DEFAULT_LIST_ITEMS]
            if _compact_text(item)
        ]
        return _truncate_text(", ".join(part for part in parts if part), max_len=max_len)
    return _truncate_text(value, max_len=max_len)


def _first_text(mapping: Mapping[str, object], keys: Sequence[str], *, max_len: int) -> str:
    """Return the first non-empty bounded mapping value for the given keys."""

    for key in keys:
        text = _value_text(mapping.get(key), max_len=max_len)
        if text:
            return text
    return ""


def _topic_anchor(candidate: AmbientDisplayImpulseCandidate, context: Mapping[str, object]) -> str:
    """Derive one clear topical anchor for the visible reserve-card text."""

    del candidate
    return _first_text(
        context,
        (
            "display_anchor",
            "topic_title",
            "person_name",
            "environment_id",
            "source_title",
            "source_label",
            "question",
        ),
        max_len=84,
    )


def _hook_hint(candidate: AmbientDisplayImpulseCandidate, context: Mapping[str, object]) -> str:
    """Return one compact conversational angle for the current candidate."""

    del candidate
    return _first_text(
        context,
        (
            "hook_hint",
            "question",
            "topic_summary",
            "rationale",
            "reason",
        ),
        max_len=120,
    )


def _context_summary(
    candidate: AmbientDisplayImpulseCandidate,
    *,
    context: Mapping[str, object],
    topic_anchor: str,
    hook_hint: str,
) -> str:
    """Compress one rich candidate context into a short prompt summary."""

    ordered_keys = (
        "memory_goal",
        "display_goal",
        "display_personality_goal",
        "topic_summary",
        "summary",
        "rationale",
        "reason",
        "source_label",
        "source_labels",
        "source_title",
        "topics",
        "recent_titles",
        "regions",
        "region",
        "scope",
        "attributes",
        "options",
    )
    parts: list[str] = []
    seen: set[str] = {topic_anchor.casefold(), hook_hint.casefold()}
    for key in ordered_keys:
        text = _value_text(context.get(key), max_len=96)
        if not text:
            continue
        lowered = text.casefold()
        if lowered in seen:
            continue
        parts.append(text)
        seen.add(lowered)
        if len(" | ".join(parts)) >= _DEFAULT_CONTEXT_SUMMARY_MAX_CHARS:
            break
    if not parts:
        return ""
    return _truncate_text(" | ".join(parts), max_len=_DEFAULT_CONTEXT_SUMMARY_MAX_CHARS)


def build_candidate_prompt_payload(candidate: AmbientDisplayImpulseCandidate) -> dict[str, object]:
    """Serialize one reserve candidate into a compact LLM prompt payload."""

    context = _coerce_mapping(candidate.generation_context)
    topic_anchor = _topic_anchor(candidate, context)
    hook_hint = _hook_hint(candidate, context)
    return {
        "topic_key": candidate.topic_key,
        "candidate_family": _truncate_text(candidate.candidate_family, max_len=32) or "general",
        "source": _truncate_text(candidate.source, max_len=32),
        "action": _truncate_text(candidate.action, max_len=24),
        "attention_state": _truncate_text(candidate.attention_state, max_len=24),
        "topic_anchor": topic_anchor,
        "hook_hint": hook_hint,
        "context_summary": _context_summary(
            candidate,
            context=context,
            topic_anchor=topic_anchor,
            hook_hint=hook_hint,
        ),
    }


def _verbosity_label(value: float) -> str:
    """Describe the current verbosity band in compact prompt text."""

    if value <= 0.32:
        return "knapp"
    if value >= 0.68:
        return "etwas ausfuehrlicher"
    return "mittel"


def _initiative_label(value: float) -> str:
    """Describe the current initiative band in compact prompt text."""

    if value <= 0.32:
        return "eher abwartend"
    if value >= 0.68:
        return "sanft proaktiv"
    return "ausgewogen"


def _german_voice_summary(snapshot: PersonalitySnapshot | None) -> str:
    """Return one compact German companion-voice guide for reserve copy."""

    if snapshot is None:
        return "ruhig, warm, direkt, unaufgeregt"
    style_profile = snapshot.style_profile
    humor_profile = snapshot.humor_profile
    parts = [
        "ruhig",
        "warm",
        "aufmerksam",
        "unaufgeregt",
        "mehr Begleiter als Ansager",
    ]
    verbosity = style_profile.verbosity if style_profile is not None else 0.5
    initiative = style_profile.initiative if style_profile is not None else 0.45
    if verbosity <= 0.4:
        parts.append("eher knapp als ausschweifend")
    elif verbosity >= 0.7:
        parts.append("etwas ausfuehrlicher, aber klar")
    else:
        parts.append("klar und alltagsnah")
    if initiative >= 0.65:
        parts.append("leise proaktiv")
    else:
        parts.append("ohne zu draengeln")
    if humor_profile is not None and humor_profile.intensity >= 0.22:
        parts.append("mit trockenem, leisem Humor")
    parts.append("eher deeskalierend als alarmistisch")
    return _truncate_text("; ".join(parts), max_len=220)


def _snapshot_prompt_profile(snapshot: PersonalitySnapshot | None) -> dict[str, object]:
    """Summarize the current Twinr personality for prompt conditioning."""

    if snapshot is None:
        return {
            "traits": [],
            "humor": {"style": "subtil", "intensity": 0.0, "summary": ""},
            "style": {"verbosity": "mittel", "initiative": "ausgewogen"},
            "voice": "ruhig, klar, direkt, unaufgeregt",
            "voice_de": _german_voice_summary(None),
            "relationship_topics": [],
            "continuity_threads": [],
            "places": [],
            "world_topics": [],
        }
    traits = [
        _truncate_text(item.summary, max_len=88)
        for item in sorted(snapshot.core_traits, key=lambda entry: entry.weight, reverse=True)[:4]
        if _compact_text(item.summary)
    ]
    humor_profile = snapshot.humor_profile
    style_profile = snapshot.style_profile
    relationship_topics = [
        _truncate_text(item.topic, max_len=52)
        for item in sorted(snapshot.relationship_signals, key=lambda entry: entry.salience, reverse=True)[:4]
        if item.stance == "affinity"
    ]
    continuity_threads = [
        _truncate_text(item.title, max_len=52)
        for item in sorted(snapshot.continuity_threads, key=lambda entry: entry.salience, reverse=True)[:4]
    ]
    places = [
        _truncate_text(item.name, max_len=48)
        for item in sorted(snapshot.place_focuses, key=lambda entry: entry.salience, reverse=True)[:3]
    ]
    world_topics = [
        _truncate_text(item.topic, max_len=52)
        for item in sorted(snapshot.world_signals, key=lambda entry: entry.salience, reverse=True)[:4]
    ]
    voice_bits: list[str] = []
    if traits:
        voice_bits.append(traits[0])
    voice_bits.append(f"spricht {_verbosity_label(style_profile.verbosity if style_profile is not None else 0.5)}")
    voice_bits.append(f"ist {_initiative_label(style_profile.initiative if style_profile is not None else 0.45)}")
    if humor_profile is not None and humor_profile.intensity >= 0.2:
        humor_summary = _compact_text(humor_profile.summary)
        if humor_summary:
            voice_bits.append(humor_summary)
    return {
        "traits": traits,
        "humor": {
            "style": _compact_text(humor_profile.style if humor_profile is not None else "subtil"),
            "intensity": float(humor_profile.intensity if humor_profile is not None else 0.0),
            "summary": _truncate_text(humor_profile.summary if humor_profile is not None else "", max_len=88),
        },
        "style": {
            "verbosity": _verbosity_label(style_profile.verbosity if style_profile is not None else 0.5),
            "initiative": _initiative_label(style_profile.initiative if style_profile is not None else 0.45),
        },
        "voice": _truncate_text("; ".join(bit for bit in voice_bits if bit), max_len=180),
        "voice_de": _german_voice_summary(snapshot),
        "relationship_topics": relationship_topics,
        "continuity_threads": continuity_threads,
        "places": places,
        "world_topics": world_topics,
    }


def build_generation_prompt(
    *,
    snapshot: PersonalitySnapshot | None,
    candidates: Sequence[AmbientDisplayImpulseCandidate],
    local_now: datetime | None,
) -> str:
    """Build the compact user prompt for one reserve-copy generation batch."""

    payload = {
        "local_day": (local_now or datetime.now().astimezone()).date().isoformat(),
        "personality": _snapshot_prompt_profile(snapshot),
        "candidates": [build_candidate_prompt_payload(candidate) for candidate in candidates],
    }
    return (
        "Schreibe fuer jeden Kandidaten eine kurze, persoenlich klingende HDMI-Reserve-Karte.\n"
        "Rueckgabeformat: JSON mit einem Feld 'items'.\n"
        "Nutze exakt die topic_key-Werte aus den Kandidaten.\n"
        "Der Text soll positive Interaktion oeffnen und Twinrs Persoenlichkeit zeigen.\n"
        "Nutze besonders personality.voice_de fuer die deutsche Stimme, topic_anchor fuer die klare Themenbenennung, "
        "hook_hint fuer den eigentlichen Aufhaenger und context_summary fuer den konkreten Anlass.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )


__all__ = [
    "build_candidate_prompt_payload",
    "build_generation_prompt",
]
