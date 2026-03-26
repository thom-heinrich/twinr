"""Shape compact reserve-card prompt payloads for the LLM rewrite step.

The reserve-copy generator should not pass full raw candidate context into one
large prompt. This module turns rich structured reserve candidates into a much
smaller prompt contract with six explicit user-facing semantics:

- ``topic_anchor``: what the card is about in a clear glanceable way
- ``hook_hint``: the concrete angle, follow-up, or tension to write from
- ``card_intent``: structured meaning for headline statement, CTA, topic, and stance
- ``pickup_signal``: condensed evidence from real earlier reserve-card outcomes
- ``copy_family``: normalized family for copy examples and judging
- ``quality_rubric`` / ``family_examples``: small positive writer/judge assets

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

from .display_reserve_copy_contract import (
    reserve_copy_examples_payload,
    reserve_copy_rubric_payload,
    resolve_reserve_copy_family,
)

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


def _bounded_float(
    value: object | None,
    *,
    minimum: float,
    maximum: float,
    default: float = 0.0,
) -> float:
    """Clamp one prompt-facing numeric signal into a finite bounded range."""

    if value is None:
        return default
    try:
        if isinstance(value, (int, float, str, bytes, bytearray)):
            number = float(value)
        else:
            return default
    except (TypeError, ValueError):
        return default
    if number < minimum:
        return minimum
    if number > maximum:
        return maximum
    return number


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
            "topic_semantics",
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
            "statement_intent",
            "rationale",
            "reason",
        ),
        max_len=120,
    )


def _card_intent(context: Mapping[str, object]) -> dict[str, str] | None:
    """Return one structured semantic card-intent block when present."""

    raw = _coerce_mapping(context.get("card_intent"))
    if not raw:
        return None
    keys = (
        "topic_semantics",
        "statement_intent",
        "cta_intent",
        "relationship_stance",
        "source_semantics",
    )
    card_intent: dict[str, str] = {}
    for key in keys:
        max_len = 160 if key.endswith("_intent") else 96
        text = _truncate_text(raw.get(key), max_len=max_len)
        if text:
            card_intent[key] = text
    return card_intent or None


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


def _pickup_signal(context: Mapping[str, object]) -> dict[str, object] | None:
    """Return one compact real-outcome hint block for copy generation.

    ``ambient_learning`` is derived from actual reserve-card exposures and
    whether those cards were picked up, ignored, or cooled later. The prompt
    should only see a very small normalized slice of that evidence.
    """

    learning = _coerce_mapping(context.get("ambient_learning"))
    if not learning:
        return None
    topic_state = _truncate_text(learning.get("topic_state"), max_len=24).casefold() or "unknown"
    family_state = _truncate_text(learning.get("family_state"), max_len=24).casefold() or "unknown"
    topic_score = round(
        _bounded_float(learning.get("topic_score"), minimum=-1.0, maximum=1.0),
        3,
    )
    repetition_pressure = round(
        _bounded_float(learning.get("topic_repetition_pressure"), minimum=0.0, maximum=1.0),
        3,
    )
    family_score = round(
        _bounded_float(learning.get("family_score"), minimum=-1.0, maximum=1.0),
        3,
    )
    action_score = round(
        _bounded_float(learning.get("action_score"), minimum=-1.0, maximum=1.0),
        3,
    )
    if (
        topic_state == "unknown"
        and family_state == "unknown"
        and topic_score == 0.0
        and repetition_pressure == 0.0
        and family_score == 0.0
        and action_score == 0.0
    ):
        return None
    return {
        "topic_state": topic_state,
        "topic_score": topic_score,
        "topic_repetition_pressure": repetition_pressure,
        "family_state": family_state,
        "family_score": family_score,
        "action_score": action_score,
    }


def build_candidate_prompt_payload(candidate: AmbientDisplayImpulseCandidate) -> dict[str, object]:
    """Serialize one reserve candidate into a compact LLM prompt payload."""

    context = _coerce_mapping(candidate.generation_context)
    topic_anchor = _topic_anchor(candidate, context)
    hook_hint = _hook_hint(candidate, context)
    payload: dict[str, object] = {
        "topic_key": candidate.topic_key,
        "semantic_topic_key": candidate.semantic_key(),
        "expansion_angle": _truncate_text(candidate.expansion_angle, max_len=32),
        "copy_family": resolve_reserve_copy_family(candidate),
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
    support_sources = [
        _truncate_text(value, max_len=32)
        for value in candidate.support_sources
        if _truncate_text(value, max_len=32)
    ]
    if support_sources:
        payload["support_sources"] = support_sources
    card_intent = _card_intent(context)
    if card_intent is not None:
        payload["card_intent"] = card_intent
    pickup_signal = _pickup_signal(context)
    if pickup_signal is not None:
        payload["pickup_signal"] = pickup_signal
    return payload


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
    variants_per_candidate: int,
) -> str:
    """Build the compact user prompt for the first reserve-copy variant pass."""

    payload = {
        "local_day": (local_now or datetime.now().astimezone()).date().isoformat(),
        "variants_per_candidate": max(1, int(variants_per_candidate)),
        "personality": _snapshot_prompt_profile(snapshot),
        "quality_rubric": reserve_copy_rubric_payload(),
        "family_examples": reserve_copy_examples_payload(candidates),
        "candidates": [build_candidate_prompt_payload(candidate) for candidate in candidates],
    }
    return (
        "Erzeuge fuer jeden Kandidaten mehrere moegliche HDMI-Reserve-Karten.\n"
        "Rueckgabeformat: JSON mit einem Feld 'items'.\n"
        "Jedes item braucht 'topic_key' und ein Feld 'variants'.\n"
        "Jede Variante braucht genau 'headline' und 'body'.\n"
        "Nutze exakt die topic_key-Werte aus den Kandidaten.\n"
        f"Erzeuge pro Kandidat genau {max(1, int(variants_per_candidate))} Varianten.\n"
        "Die Varianten sollen unterschiedliche, aber plausible Aufhaenger haben und nicht nur dieselbe Formulierung leicht umstellen.\n"
        "Der Text soll positive Interaktion oeffnen und Twinrs Persoenlichkeit zeigen.\n"
        "family_examples enthaelt kleine Gold-Beispiele fuer gute Karten in der jeweiligen copy_family. "
        "Nutze sie als Stil- und Strukturreferenz, nicht als Satzschablone.\n"
        "quality_rubric zeigt, woran gute Karten gemessen werden. "
        "Sie gilt schon im Writer-Pass als stiller Selbstcheck.\n"
        "Nutze besonders personality.voice_de fuer die deutsche Stimme, topic_anchor fuer die klare Themenbenennung, "
        "hook_hint fuer den eigentlichen Aufhaenger, context_summary fuer den konkreten Anlass, "
        "card_intent als semantische Kartenspezifikation und pickup_signal fuer echte Reaktionsspuren aus frueheren Reserve-Karten.\n"
        "Wenn card_intent vorhanden ist, beschreibt statement_intent die Aussage der Headline, cta_intent die Bewegung der Body-Zeile, "
        "topic_semantics den Themenkern und relationship_stance Twinrs Haltung.\n"
        "Nutze diese Semantik als Primärquelle und spiegle weder topic_anchor noch card_intent wortwoertlich als Label zurück.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )


def build_selection_prompt(
    *,
    snapshot: PersonalitySnapshot | None,
    candidates: Sequence[AmbientDisplayImpulseCandidate],
    variants_by_topic: Mapping[str, Sequence[Mapping[str, str]]],
    local_now: datetime | None,
) -> str:
    """Build the compact user prompt for the second reserve-copy selection pass."""

    candidate_payloads: list[dict[str, object]] = []
    for candidate in candidates:
        payload = build_candidate_prompt_payload(candidate)
        payload["variants"] = [
            {
                "headline": _truncate_text(variant.get("headline"), max_len=128),
                "body": _truncate_text(variant.get("body"), max_len=128),
            }
            for variant in variants_by_topic.get(candidate.topic_key.casefold(), ())
            if _truncate_text(variant.get("headline"), max_len=128)
            and _truncate_text(variant.get("body"), max_len=128)
        ]
        candidate_payloads.append(payload)
    payload = {
        "local_day": (local_now or datetime.now().astimezone()).date().isoformat(),
        "personality": _snapshot_prompt_profile(snapshot),
        "quality_rubric": reserve_copy_rubric_payload(),
        "family_examples": reserve_copy_examples_payload(candidates),
        "candidates": candidate_payloads,
    }
    return (
        "Waehle fuer jeden Kandidaten aus mehreren Vorschlaegen die beste finale HDMI-Reserve-Karte.\n"
        "Rueckgabeformat: JSON mit einem Feld 'items'.\n"
        "Jedes item braucht genau 'topic_key', 'headline' und 'body'.\n"
        "Nutze exakt die topic_key-Werte aus den Kandidaten.\n"
        "Waehle die Variante, die am klarsten, engagingsten und am ehesten nach Twinr klingt.\n"
        "Du darfst eine gewaehlte Variante minimal straffen oder grammatisch glaetten, aber nicht den Anlass neu erfinden.\n"
        "Nutze family_examples als positive Stilreferenz fuer die jeweilige copy_family.\n"
        "Pruefe jede Variante still an quality_rubric. "
        "Wenn zwei Varianten gegeneinander antreten, gewinnt die, die in der Rubrik insgesamt besser abschneidet.\n"
        "Wenn ein Kandidat card_intent enthaelt, muessen statement_intent und cta_intent in der finalen Karte klar wiedererkennbar bleiben.\n"
        "Bevorzuge Varianten mit konkreter Beobachtung, klarem Themenanker und echter Einladung statt generischer Nettigkeit.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )


__all__ = [
    "build_candidate_prompt_payload",
    "build_generation_prompt",
    "build_selection_prompt",
]
