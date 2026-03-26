"""Expand grounded reserve topics into multiple right-lane card surfaces.

The companion flow already finds semantically meaningful topics, but the
product contract for the right lane is about card surfaces, not only raw topic
seeds. This module keeps that expansion step explicit and deterministic:

- merge overlapping seeds by semantic topic
- preserve one grouped semantic key for feedback and learning
- emit up to three distinct, grounded card surfaces per semantic topic
- keep expansion generic and family-driven instead of hardcoding named topics
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import sha1

from twinr.agent.personality.display_impulses import AmbientDisplayImpulseCandidate

from .display_reserve_diversity import select_diverse_candidates
from .display_reserve_support import compact_text

_DEFAULT_MAX_CARDS_PER_TOPIC = 3
_QUESTION_ANGLES = frozenset(
    {
        "example_probe",
        "preference_probe",
        "continuity_follow_up",
        "meaning_or_update",
        "gentle_status_check",
        "personal_reaction",
        "gentle_follow_up",
        "clarify",
        "public_reaction",
    }
)


def _topic_key(value: object | None) -> str:
    """Return one normalized semantic topic key."""

    return compact_text(value, max_len=96).casefold()


def _context(value: Mapping[str, object] | None) -> dict[str, object]:
    """Return one mutable plain mapping for generation context updates."""

    return dict(value or {})


def _candidate_sort_key(candidate: AmbientDisplayImpulseCandidate) -> tuple[float, str, str, str]:
    """Return one stable descending rank key for bundle selection."""

    return (
        float(candidate.salience),
        compact_text(candidate.attention_state, max_len=32).casefold(),
        compact_text(candidate.action, max_len=24).casefold(),
        candidate.semantic_key(),
    )


def _ordered_unique(values: Iterable[object]) -> tuple[str, ...]:
    """Return one bounded ordered unique tuple of compacted strings."""

    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        compact = compact_text(value, max_len=48)
        if not compact:
            continue
        key = compact.casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(compact)
    return tuple(ordered)


def _bundle_anchor(candidate: AmbientDisplayImpulseCandidate) -> str:
    """Return one human-facing anchor for fallback reserve copy."""

    context = _context(candidate.generation_context)
    for value in (
        context.get("display_anchor"),
        context.get("topic_title"),
        context.get("topic_semantics"),
        candidate.title,
        candidate.headline,
        candidate.semantic_key(),
    ):
        compact = compact_text(value, max_len=72)
        if compact:
            return compact
    return "dem Thema"


def _card_key(*, semantic_topic_key: str, angle: str) -> str:
    """Return one bounded unique key for a concrete expanded reserve card."""

    semantic = _topic_key(semantic_topic_key)
    normalized_angle = compact_text(angle, max_len=32).casefold().replace(" ", "_")
    if not semantic:
        semantic = "reserve_card"
    prefix = semantic[:56]
    digest = sha1(f"{semantic}::{normalized_angle}".encode("utf-8")).hexdigest()[:10]
    return f"{prefix}::{normalized_angle}::{digest}"


@dataclass(frozen=True, slots=True)
class DisplayReserveTopicBundle:
    """Group all raw reserve seeds that refer to one semantic topic."""

    semantic_topic_key: str
    primary_candidate: AmbientDisplayImpulseCandidate
    support_sources: tuple[str, ...]
    support_families: tuple[str, ...]
    candidate_count: int

    def representative_candidate(self) -> AmbientDisplayImpulseCandidate:
        """Return one ranked representative for bundle ordering/diversity."""

        context = _context(self.primary_candidate.generation_context)
        context["semantic_topic_key"] = self.semantic_topic_key
        context["support_sources"] = self.support_sources
        context["support_families"] = self.support_families
        context["support_count"] = self.candidate_count
        bonus = min(0.18, 0.04 * float(max(0, self.candidate_count - 1)))
        return replace(
            self.primary_candidate,
            semantic_topic_key=self.semantic_topic_key,
            support_sources=self.support_sources,
            generation_context=context,
            salience=min(1.25, float(self.primary_candidate.salience) + bonus),
        )


def bundle_display_reserve_candidates(
    candidates: Sequence[AmbientDisplayImpulseCandidate],
) -> tuple[DisplayReserveTopicBundle, ...]:
    """Return grouped topic bundles from raw reserve candidates."""

    grouped: dict[str, list[AmbientDisplayImpulseCandidate]] = {}
    for candidate in candidates:
        semantic_topic_key = candidate.semantic_key()
        if not semantic_topic_key:
            continue
        grouped.setdefault(semantic_topic_key, []).append(
            replace(candidate, semantic_topic_key=semantic_topic_key)
        )
    bundles: list[DisplayReserveTopicBundle] = []
    for semantic_topic_key, entries in grouped.items():
        ordered_entries = sorted(entries, key=_candidate_sort_key, reverse=True)
        primary = ordered_entries[0]
        support_sources = _ordered_unique(
            (
                *(primary.support_sources or ()),
                *(entry.source for entry in ordered_entries),
            )
        )
        support_families = _ordered_unique(entry.candidate_family for entry in ordered_entries)
        bundles.append(
            DisplayReserveTopicBundle(
                semantic_topic_key=semantic_topic_key,
                primary_candidate=primary,
                support_sources=support_sources,
                support_families=support_families,
                candidate_count=len(ordered_entries),
            )
        )
    return tuple(bundles)


def _bundle_kind(bundle: DisplayReserveTopicBundle) -> str:
    """Return one generic expansion family for a semantic topic bundle."""

    families = {compact_text(value, max_len=48).casefold() for value in bundle.support_families}
    sources = {compact_text(value, max_len=48).casefold() for value in bundle.support_sources}
    has_world = any(family.startswith("world") for family in families) or any(
        source.startswith("world") for source in sources
    )
    has_discovery = any("discovery" in family for family in families)
    has_conflict = any("conflict" in family for family in families)
    has_place = "place" in families
    has_personal = any(
        family.startswith(prefix)
        for family in families
        for prefix in ("memory", "reflection")
    ) or any(source in {"relationship", "continuity", "reflection_midterm"} for source in sources)
    if has_conflict:
        return "conflict"
    if has_discovery:
        return "discovery"
    if has_world and has_personal:
        return "mixed"
    if has_world:
        return "world"
    if has_place:
        return "place"
    if has_personal:
        return "continuity"
    return "generic"


def _angle_plan(bundle: DisplayReserveTopicBundle) -> tuple[str, ...]:
    """Return the deterministic angle plan for one semantic topic bundle."""

    kind = _bundle_kind(bundle)
    if kind == "conflict":
        return ("primary",)
    if kind == "discovery":
        return ("primary", "example_probe", "preference_probe")
    if kind == "mixed":
        return ("primary", "continuity_follow_up", "public_reaction")
    if kind == "world":
        return ("primary", "public_reaction", "broader_view")
    if kind == "place":
        return ("primary", "local_association", "gentle_status_check")
    if kind == "continuity":
        return ("primary", "meaning_or_update", "gentle_status_check")
    return ("primary", "personal_reaction", "gentle_follow_up")


def _secondary_copy(
    candidate: AmbientDisplayImpulseCandidate,
    *,
    angle: str,
) -> tuple[str, str]:
    """Return deterministic fallback copy for one expanded reserve-card angle."""

    anchor = _bundle_anchor(candidate)
    if angle == "example_probe":
        return (
            f"Bei {anchor} wuerde ich dich gern etwas genauer verstehen.",
            "Magst du mir ein kleines Beispiel dazu geben?",
        )
    if angle == "preference_probe":
        return (
            f"Bei {anchor} interessiert mich noch deine Seite.",
            "Was ist dir daran besonders wichtig?",
        )
    if angle == "continuity_follow_up":
        return (
            f"{anchor} ist bei mir noch nicht ganz weg.",
            "Magst du mich kurz auf Stand bringen?",
        )
    if angle == "meaning_or_update":
        return (
            f"Ich denke bei {anchor} noch einen Schritt weiter.",
            "Wie ist es damit gerade?",
        )
    if angle == "gentle_status_check":
        return (
            f"Zu {anchor} fehlt mir noch ein kleines Bild.",
            "Wollen wir kurz darauf schauen?",
        )
    if angle == "public_reaction":
        return (
            f"Bei {anchor} frage ich mich gerade nach deiner Sicht.",
            "Was meinst du dazu?",
        )
    if angle == "broader_view":
        return (
            f"Bei {anchor} steckt fuer mich noch mehr drin.",
            "Magst du kurz draufschauen?",
        )
    if angle == "local_association":
        return (
            f"Bei {anchor} denke ich gerade an deinen Alltag.",
            "Wie fuehlt sich das fuer dich dort an?",
        )
    if angle == "personal_reaction":
        return (
            f"An {anchor} bleibe ich gerade noch haengen.",
            "Wie schaust du darauf?",
        )
    if angle == "gentle_follow_up":
        return (
            f"Zu {anchor} habe ich noch einen ruhigen Nachfasser im Kopf.",
            "Magst du kurz weitermachen?",
        )
    return (
        compact_text(candidate.headline, max_len=128),
        compact_text(candidate.body, max_len=128),
    )


def _expanded_action(base_action: str, *, angle: str) -> str:
    """Return the bounded action token for one expanded card angle."""

    if angle == "primary":
        return compact_text(base_action, max_len=24).lower() or "hint"
    if angle in _QUESTION_ANGLES:
        return "ask_one"
    return "brief_update"


def _expanded_candidate(
    bundle: DisplayReserveTopicBundle,
    *,
    angle: str,
) -> AmbientDisplayImpulseCandidate:
    """Return one expanded reserve candidate for a concrete card angle."""

    candidate = bundle.representative_candidate()
    context = _context(candidate.generation_context)
    context["semantic_topic_key"] = bundle.semantic_topic_key
    context["support_sources"] = bundle.support_sources
    context["support_families"] = bundle.support_families
    context["support_count"] = bundle.candidate_count
    context["expansion_angle"] = angle
    context["bundle_kind"] = _bundle_kind(bundle)
    if angle == "primary":
        headline = compact_text(candidate.headline, max_len=128)
        body = compact_text(candidate.body, max_len=128)
    else:
        headline, body = _secondary_copy(candidate, angle=angle)
    return replace(
        candidate,
        topic_key=_card_key(semantic_topic_key=bundle.semantic_topic_key, angle=angle),
        semantic_topic_key=bundle.semantic_topic_key,
        action=_expanded_action(candidate.action, angle=angle),
        headline=headline,
        body=body,
        reason=compact_text(f"{candidate.reason}; expansion={angle}", max_len=120),
        generation_context=context,
        expansion_angle=angle,
        support_sources=bundle.support_sources,
    )


def expand_display_reserve_candidates(
    candidates: Sequence[AmbientDisplayImpulseCandidate],
    *,
    target_cards: int,
    max_cards_per_topic: int = _DEFAULT_MAX_CARDS_PER_TOPIC,
) -> tuple[AmbientDisplayImpulseCandidate, ...]:
    """Return one bounded expanded reserve-card set for the right lane."""

    limited_target = max(1, int(target_cards))
    bundles = bundle_display_reserve_candidates(candidates)
    if not bundles:
        return ()
    representatives = tuple(bundle.representative_candidate() for bundle in bundles)
    ordered_representatives = select_diverse_candidates(
        representatives,
        max_items=len(representatives),
    )
    bundle_by_topic = {bundle.semantic_topic_key: bundle for bundle in bundles}
    ordered_bundles = [
        bundle_by_topic[representative.semantic_key()]
        for representative in ordered_representatives
        if representative.semantic_key() in bundle_by_topic
    ]
    expanded: list[AmbientDisplayImpulseCandidate] = []
    max_passes = max(1, min(int(max_cards_per_topic), _DEFAULT_MAX_CARDS_PER_TOPIC))
    for pass_index in range(max_passes):
        for bundle in ordered_bundles:
            if len(expanded) >= limited_target:
                return tuple(expanded)
            angles = _angle_plan(bundle)
            if pass_index >= len(angles):
                continue
            expanded.append(_expanded_candidate(bundle, angle=angles[pass_index]))
    return tuple(expanded[:limited_target])


__all__ = [
    "DisplayReserveTopicBundle",
    "bundle_display_reserve_candidates",
    "expand_display_reserve_candidates",
]
