# CHANGELOG: 2026-03-29
# BUG-1: Fixed semantic bundling so overlapping topic variants (case, token order, aliases, upstream cluster keys)
# BUG-1: are merged into one bundle instead of producing duplicate right-lane cards.
# BUG-2: Fixed empty primary cards when upstream seeds provide title/semantic info but no headline/body.
# BUG-3: Fixed unstable selection on NaN/invalid salience and added transitive bundle merges for bridged aliases.
# SEC-1: Removed raw semantic text from topic_key and switched to opaque BLAKE2s card ids.
# SEC-1: # BREAKING: topic_key is no longer human-readable. This avoids leaking private semantic topics into logs/telemetry.
# SEC-2: Stripped bidi override/isolate characters and control characters from user-facing copy and identifiers.
# IMP-1: Added hybrid semantic grouping with canonical normalization, alias support, optional RapidFuzz fuzzy merging,
# IMP-1: and upstream cluster/embedding-key awareness.
# IMP-2: Added uncertainty-aware angle planning and multi-objective bundle ordering (salience + evidence + uncertainty + novelty).
# IMP-3: Added duplicate-surface suppression and richer bundle metadata in generation_context for downstream learning.

"""Expand grounded reserve topics into multiple right-lane card surfaces.

The companion flow already finds semantically meaningful topics, but the
product contract for the right lane is about card surfaces, not only raw topic
seeds. This module keeps that expansion step explicit and deterministic:

- merge overlapping seeds by semantic topic
- preserve one grouped semantic key for feedback and learning
- emit up to three distinct, grounded card surfaces per semantic topic
- keep expansion generic and family-driven instead of hardcoding named topics

2026 upgrade notes:
- hybrid semantic bundling combines exact ids, aliases, cluster ids, and lexical similarity
- angle planning uses uncertainty/evidence signals instead of a static family-only plan
- bundle ordering optimizes several objectives at once while remaining deterministic and Pi-friendly
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from hashlib import blake2s

try:
    from rapidfuzz import fuzz as _rapidfuzz_fuzz
except Exception:  # pragma: no cover - optional dependency
    _rapidfuzz_fuzz = None

from twinr.agent.personality.display_impulses import AmbientDisplayImpulseCandidate

from .display_reserve_diversity import select_diverse_candidates
from .display_reserve_support import compact_text

_DEFAULT_MAX_CARDS_PER_TOPIC = 3
_TOPIC_FUZZY_MATCH_THRESHOLD = 94.0
_TOPIC_SUBSET_MATCH_THRESHOLD = 97.0
_OPAQUE_CARD_KEY_PREFIX = "reserve_card"

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

_GENERIC_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "bei",
        "das",
        "dem",
        "den",
        "der",
        "des",
        "die",
        "ein",
        "eine",
        "einem",
        "einer",
        "eines",
        "for",
        "from",
        "im",
        "in",
        "mit",
        "of",
        "or",
        "the",
        "to",
        "und",
        "von",
        "zu",
    }
)

_BIDI_SPOOF_CHARS = frozenset(
    {
        "\u061c",
        "\u200e",
        "\u200f",
        "\u202a",
        "\u202b",
        "\u202c",
        "\u202d",
        "\u202e",
        "\u2066",
        "\u2067",
        "\u2068",
        "\u2069",
    }
)

_HIGH_CONFIDENCE_ID_KEYS = (
    "semantic_cluster",
    "semantic_cluster_id",
    "semantic_embedding_key",
    "topic_cluster",
    "topic_cluster_id",
    "cluster_id",
)

_ALIAS_KEYS = (
    "semantic_alias",
    "semantic_aliases",
    "topic_alias",
    "topic_aliases",
)


def _safe_float(value: object, *, default: float = 0.0) -> float:
    """Return one finite float or the provided default."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def _clip01(value: float) -> float:
    """Clamp one float into [0.0, 1.0]."""

    return min(1.0, max(0.0, value))


def _sanitize_text(value: object | None, *, max_len: int) -> str:
    """Return compacted text with control/bidi spoofing chars removed."""

    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text)
    cleaned: list[str] = []
    for char in text:
        if char in _BIDI_SPOOF_CHARS:
            continue
        category = unicodedata.category(char)
        if category in {"Cc", "Cs"}:
            if char.isspace():
                cleaned.append(" ")
            continue
        cleaned.append(char)
    return compact_text(re.sub(r"\s+", " ", "".join(cleaned)).strip(), max_len=max_len)


def _normalize_identifier(value: object | None, *, max_len: int = 128) -> str:
    """Return one normalized identifier-safe string."""

    return _sanitize_text(value, max_len=max_len).casefold()


def _topic_key(value: object | None) -> str:
    """Return one normalized semantic topic key."""

    return _sanitize_text(value, max_len=96).casefold()


def _topic_signature(value: object | None) -> str:
    """Return one lexical signature suitable for overlap matching."""

    normalized = _normalize_identifier(value, max_len=128)
    if not normalized:
        return ""
    normalized = re.sub(r"[_/|:;,+\-]+", " ", normalized)
    normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
    tokens = [token for token in normalized.split() if token and token not in _GENERIC_STOPWORDS]
    if not tokens:
        return normalized
    return " ".join(sorted(dict.fromkeys(tokens)))


def _context(value: Mapping[str, object] | None) -> dict[str, object]:
    """Return one mutable plain mapping for generation context updates."""

    return dict(value or {})


def _candidate_copy_quality(candidate: AmbientDisplayImpulseCandidate) -> int:
    """Return a tiny deterministic quality proxy for tie-breaking."""

    quality = 0
    if _sanitize_text(candidate.headline, max_len=128):
        quality += 2
    if _sanitize_text(candidate.body, max_len=128):
        quality += 2
    if _sanitize_text(candidate.title, max_len=96):
        quality += 1
    if candidate.support_sources:
        quality += 1
    return quality


def _candidate_sort_key(
    candidate: AmbientDisplayImpulseCandidate,
) -> tuple[float, int, str, str, str, str]:
    """Return one stable ascending rank key for bundle selection."""

    return (
        -_safe_float(getattr(candidate, "salience", 0.0), default=0.0),
        -_candidate_copy_quality(candidate),
        _normalize_identifier(getattr(candidate, "source", ""), max_len=48),
        _normalize_identifier(getattr(candidate, "candidate_family", ""), max_len=48),
        _normalize_identifier(getattr(candidate, "attention_state", ""), max_len=32),
        _normalize_identifier(candidate.semantic_key(), max_len=128),
    )


def _ordered_unique(values: Iterable[object], *, max_len: int = 64) -> tuple[str, ...]:
    """Return one ordered unique tuple of compacted strings."""

    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        compact = _sanitize_text(value, max_len=max_len)
        if not compact:
            continue
        key = compact.casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(compact)
    return tuple(ordered)


def _ordered_unique_identifiers(values: Iterable[str]) -> tuple[str, ...]:
    """Return ordered unique normalized identifiers without truncation collisions."""

    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        compact = _normalize_identifier(value, max_len=128)
        if not compact or compact in seen:
            continue
        seen.add(compact)
        ordered.append(compact)
    return tuple(ordered)


def _iter_alias_values(value: object | None) -> Iterable[object]:
    """Yield alias-like values as a flat iterable."""

    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(value)
    if isinstance(value, Iterable):
        return tuple(value)
    return (value,)


def _bundle_anchor(candidate: AmbientDisplayImpulseCandidate) -> str:
    """Return one human-facing anchor for fallback reserve copy."""

    context = _context(candidate.generation_context)
    for value in (
        context.get("display_anchor"),
        context.get("topic_title"),
        context.get("topic_semantics"),
        candidate.semantic_key(),
        candidate.title,
        candidate.headline,
    ):
        compact = _sanitize_text(value, max_len=72)
        if compact:
            return compact
    return "dem Thema"


def _bundle_id_from_signatures(signatures: Iterable[str]) -> str:
    """Return one opaque stable bundle id."""

    material = "||".join(sorted(signature for signature in signatures if signature))
    if not material:
        material = "reserve_topic"
    return blake2s(f"topic::{material}".encode("utf-8"), digest_size=10).hexdigest()


def _card_key(*, semantic_topic_key: str, angle: str, bundle_id: str) -> str:
    """Return one opaque unique key for a concrete expanded reserve card."""

    # BREAKING: topic_key no longer embeds semantic text. It is now opaque to avoid
    # leaking private user/topic information into logs, analytics, or feedback traces.
    semantic = _topic_key(semantic_topic_key)
    normalized_angle = _sanitize_text(angle, max_len=32).casefold().replace(" ", "_")
    digest_input = f"card::{bundle_id}::{semantic}::{normalized_angle}".encode("utf-8")
    digest = blake2s(digest_input, digest_size=12).hexdigest()
    return f"{_OPAQUE_CARD_KEY_PREFIX}::{digest}"


def _topic_similarity(left: str, right: str) -> float:
    """Return a lexical overlap score in [0, 100]."""

    if not left or not right:
        return 0.0
    if left == right:
        return 100.0
    if _rapidfuzz_fuzz is not None:
        return max(
            float(_rapidfuzz_fuzz.token_set_ratio(left, right)),
            float(_rapidfuzz_fuzz.partial_token_set_ratio(left, right)),
        )
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if left_tokens and right_tokens and (left_tokens <= right_tokens or right_tokens <= left_tokens):
        return 100.0
    token_overlap = 100.0 * len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
    seq_ratio = 100.0 * SequenceMatcher(None, left, right).ratio()
    return max(token_overlap, seq_ratio)


def _should_merge_topics(left: str, right: str) -> bool:
    """Return whether two lexical signatures are close enough to merge."""

    if not left or not right:
        return False
    if left == right:
        return True
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    overlap = left_tokens & right_tokens
    if not overlap:
        return False
    similarity = _topic_similarity(left, right)
    if left_tokens <= right_tokens or right_tokens <= left_tokens:
        return similarity >= _TOPIC_SUBSET_MATCH_THRESHOLD
    min_overlap = 1 if min(len(left_tokens), len(right_tokens)) <= 2 else 2
    return similarity >= _TOPIC_FUZZY_MATCH_THRESHOLD and len(overlap) >= min_overlap


def _semantic_material(candidate: AmbientDisplayImpulseCandidate) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Return display topic, exact ids, and fuzzy signatures for one candidate."""

    context = _context(candidate.generation_context)

    labels = _ordered_unique(
        (
            candidate.semantic_key(),
            context.get("semantic_topic_key"),
            context.get("topic_semantics"),
            context.get("topic_title"),
            context.get("display_anchor"),
            candidate.title,
            candidate.headline,
        ),
        max_len=96,
    )

    exact_ids: list[str] = []
    for key in _HIGH_CONFIDENCE_ID_KEYS:
        exact_ids.extend(_iter_alias_values(context.get(key)))
    for key in _ALIAS_KEYS:
        exact_ids.extend(_iter_alias_values(context.get(key)))
    exact_ids.extend(
        value
        for value in (
            getattr(candidate, "semantic_topic_key", None),
            context.get("semantic_topic_key"),
        )
        if _normalize_identifier(value, max_len=128)
    )

    normalized_exact = _ordered_unique_identifiers(_normalize_identifier(value) for value in exact_ids)
    signature_inputs = (
        candidate.semantic_key(),
        getattr(candidate, "semantic_topic_key", None),
        context.get("semantic_topic_key"),
        context.get("topic_semantics"),
        context.get("topic_title"),
        context.get("display_anchor"),
        *exact_ids,
    )
    signatures = _ordered_unique_identifiers(_topic_signature(value) for value in signature_inputs)
    if not signatures:
        signatures = _ordered_unique_identifiers(
            _topic_signature(value)
            for value in (
                candidate.topic_key,
                candidate.title,
                candidate.headline,
            )
        )

    display_topic = labels[0] if labels else _bundle_anchor(candidate)
    return display_topic, normalized_exact, signatures


def _extract_uncertainty(candidate: AmbientDisplayImpulseCandidate) -> float:
    """Return one normalized uncertainty score for follow-up planning."""

    context = _context(candidate.generation_context)
    raw_values: list[float] = []
    for key in (
        "uncertainty",
        "topic_uncertainty",
        "entropy",
        "topic_entropy",
        "intent_entropy",
        "ambiguity",
        "disambiguation_need",
        "preference_gap",
    ):
        value = context.get(key)
        number = _safe_float(value, default=-1.0)
        if number < 0.0:
            continue
        if number <= 1.0:
            raw_values.append(number)
        else:
            raw_values.append(math.tanh(number / 4.0))
    if raw_values:
        return _clip01(sum(raw_values) / len(raw_values))

    heuristic = 0.0
    if not _sanitize_text(candidate.headline, max_len=128):
        heuristic += 0.18
    if not _sanitize_text(candidate.body, max_len=128):
        heuristic += 0.12
    if "probe" in _normalize_identifier(candidate.action, max_len=32):
        heuristic += 0.20
    return _clip01(heuristic)


@dataclass(slots=True)
class _BundleAccumulator:
    """Mutable helper while building semantic bundles."""

    exact_ids: set[str] = field(default_factory=set)
    signatures: set[str] = field(default_factory=set)
    labels: list[str] = field(default_factory=list)
    candidates: list[AmbientDisplayImpulseCandidate] = field(default_factory=list)
    uncertainty_values: list[float] = field(default_factory=list)

    def matches(self, exact_ids: Sequence[str], signatures: Sequence[str]) -> bool:
        """Return whether the incoming candidate belongs to this accumulator."""

        if self.exact_ids.intersection(exact_ids):
            return True
        for left in self.signatures:
            for right in signatures:
                if _should_merge_topics(left, right):
                    return True
        return False

    def add(
        self,
        candidate: AmbientDisplayImpulseCandidate,
        *,
        display_topic: str,
        exact_ids: Sequence[str],
        signatures: Sequence[str],
        uncertainty: float,
    ) -> None:
        """Merge one candidate into this accumulator."""

        self.candidates.append(candidate)
        self.exact_ids.update(exact_ids)
        self.signatures.update(signatures)
        if display_topic:
            self.labels.append(display_topic)
        self.uncertainty_values.append(_clip01(uncertainty))


def _choose_bundle_label(
    labels: Sequence[str],
    primary: AmbientDisplayImpulseCandidate,
) -> str:
    """Return one stable human-readable semantic label for the bundle."""

    primary_semantic = _sanitize_text(primary.semantic_key(), max_len=96)
    candidates = _ordered_unique(
        (
            primary_semantic,
            *labels,
            _bundle_anchor(primary),
            primary.title,
            primary.headline,
        ),
        max_len=96,
    )
    if not candidates:
        return "reserve_topic"
    return min(
        candidates,
        key=lambda value: (
            0 if value == primary_semantic else 1,
            abs(len(value) - 28),
            len(value),
            value.casefold(),
        ),
    )


@dataclass(frozen=True, slots=True)
class DisplayReserveTopicBundle:
    """Group all raw reserve seeds that refer to one semantic topic."""

    semantic_topic_key: str
    primary_candidate: AmbientDisplayImpulseCandidate
    support_sources: tuple[str, ...]
    support_families: tuple[str, ...]
    candidate_count: int
    bundle_id: str = ""
    uncertainty: float = 0.0
    evidence_score: float = 0.0

    def representative_candidate(self) -> AmbientDisplayImpulseCandidate:
        """Return one ranked representative for bundle ordering/diversity."""

        context = _context(self.primary_candidate.generation_context)
        context["semantic_topic_key"] = self.semantic_topic_key
        context["semantic_bundle_id"] = self.bundle_id
        context["support_sources"] = self.support_sources
        context["support_families"] = self.support_families
        context["support_count"] = self.candidate_count
        context["bundle_uncertainty"] = self.uncertainty
        context["bundle_evidence"] = self.evidence_score
        bonus = min(0.22, 0.04 * float(max(0, self.candidate_count - 1)) + 0.08 * self.evidence_score)
        return replace(
            self.primary_candidate,
            semantic_topic_key=self.semantic_topic_key,
            support_sources=self.support_sources,
            generation_context=context,
            salience=min(1.35, _safe_float(self.primary_candidate.salience, default=0.0) + bonus),
        )


def bundle_display_reserve_candidates(
    candidates: Sequence[AmbientDisplayImpulseCandidate],
) -> tuple[DisplayReserveTopicBundle, ...]:
    """Return grouped topic bundles from raw reserve candidates."""

    accumulators: list[_BundleAccumulator] = []

    for candidate in candidates:
        display_topic, exact_ids, signatures = _semantic_material(candidate)
        if not exact_ids and not signatures and not display_topic:
            continue

        normalized_candidate = replace(
            candidate,
            semantic_topic_key=display_topic or _bundle_anchor(candidate),
        )
        uncertainty = _extract_uncertainty(normalized_candidate)

        matched = [accumulator for accumulator in accumulators if accumulator.matches(exact_ids, signatures)]
        if not matched:
            match = _BundleAccumulator()
            accumulators.append(match)
        else:
            match = matched[0]
            for other in matched[1:]:
                match.exact_ids.update(other.exact_ids)
                match.signatures.update(other.signatures)
                match.labels.extend(other.labels)
                match.candidates.extend(other.candidates)
                match.uncertainty_values.extend(other.uncertainty_values)
                accumulators.remove(other)
        match.add(
            normalized_candidate,
            display_topic=display_topic,
            exact_ids=exact_ids,
            signatures=signatures,
            uncertainty=uncertainty,
        )

    bundles: list[DisplayReserveTopicBundle] = []
    for accumulator in accumulators:
        if not accumulator.candidates:
            continue

        ordered_entries = sorted(accumulator.candidates, key=_candidate_sort_key)
        primary = ordered_entries[0]
        semantic_topic_key = _choose_bundle_label(accumulator.labels, primary)

        support_sources = _ordered_unique(
            (
                *(primary.support_sources or ()),
                *(_sanitize_text(entry.source, max_len=48) for entry in ordered_entries),
            ),
            max_len=48,
        )
        support_families = _ordered_unique(
            (_sanitize_text(entry.candidate_family, max_len=48) for entry in ordered_entries),
            max_len=48,
        )

        evidence_score = _clip01(
            0.14 * max(0, len(ordered_entries) - 1)
            + 0.09 * len(support_sources)
            + 0.07 * len(support_families)
        )
        mean_uncertainty = sum(accumulator.uncertainty_values) / len(accumulator.uncertainty_values)
        max_uncertainty = max(accumulator.uncertainty_values)
        uncertainty = _clip01(
            0.55 * max_uncertainty
            + 0.45 * mean_uncertainty
            + (0.08 if len(support_families) >= 2 else 0.0)
            + (0.06 if len(support_sources) >= 3 else 0.0)
        )

        bundles.append(
            DisplayReserveTopicBundle(
                semantic_topic_key=semantic_topic_key,
                primary_candidate=primary,
                support_sources=support_sources,
                support_families=support_families,
                candidate_count=len(ordered_entries),
                bundle_id=_bundle_id_from_signatures((*accumulator.signatures, semantic_topic_key)),
                uncertainty=uncertainty,
                evidence_score=evidence_score,
            )
        )

    return tuple(sorted(bundles, key=lambda bundle: _candidate_sort_key(bundle.primary_candidate)))


def _bundle_family_tokens(bundle: DisplayReserveTopicBundle) -> frozenset[str]:
    """Return normalized family/source tokens for multi-objective ordering."""

    tokens: set[str] = set()
    for value in (*bundle.support_families, *bundle.support_sources):
        normalized = _normalize_identifier(value, max_len=48)
        if not normalized:
            continue
        for token in re.split(r"[_\-/\s]+", normalized):
            if token:
                tokens.add(token)
    return frozenset(tokens)


def _bundle_kind(bundle: DisplayReserveTopicBundle) -> str:
    """Return one generic expansion family for a semantic topic bundle."""

    tokens = _bundle_family_tokens(bundle)

    has_world = any(token.startswith("world") or token in {"news", "public", "societal"} for token in tokens)
    has_discovery = "discovery" in tokens
    has_conflict = "conflict" in tokens
    has_place = any(token.startswith("place") or token in {"local", "location"} for token in tokens)
    has_personal = any(
        token.startswith(prefix)
        for token in tokens
        for prefix in ("memory", "reflection", "relationship", "continuity", "personal")
    )

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
    headline = _sanitize_text(candidate.headline or candidate.title or anchor, max_len=128)
    body = _sanitize_text(candidate.body, max_len=128)
    if not body:
        body = "Wollen wir kurz darauf schauen?"
    return headline, body


def _expanded_action(base_action: str, *, angle: str) -> str:
    """Return the bounded action token for one expanded card angle."""

    if angle == "primary":
        return _sanitize_text(base_action, max_len=24).lower() or "hint"
    if angle in _QUESTION_ANGLES:
        return "ask_one"
    return "brief_update"


def _angle_priority(bundle: DisplayReserveTopicBundle, angle: str) -> float:
    """Return one score for whether an angle belongs in the bundle plan."""

    kind = _bundle_kind(bundle)
    uncertainty = bundle.uncertainty
    evidence = bundle.evidence_score
    if angle == "primary":
        return 10.0

    score = 0.20 + 0.35 * evidence

    if angle in _QUESTION_ANGLES:
        score += 0.55 * uncertainty
    else:
        score += 0.10 * (1.0 - uncertainty)

    if kind == "conflict":
        if angle in {"meaning_or_update", "gentle_status_check"}:
            score += 0.30
        else:
            score -= 1.00
    elif kind == "discovery":
        if angle in {"example_probe", "preference_probe", "broader_view"}:
            score += 0.32
    elif kind == "mixed":
        if angle in {"continuity_follow_up", "public_reaction", "meaning_or_update"}:
            score += 0.28
    elif kind == "world":
        if angle == "public_reaction":
            score += 0.40 if uncertainty < 0.20 else 0.32
        elif angle == "broader_view":
            score += 0.28
    elif kind == "place":
        if angle in {"local_association", "gentle_status_check"}:
            score += 0.28
    elif kind == "continuity":
        if angle in {"meaning_or_update", "gentle_status_check", "continuity_follow_up"}:
            score += 0.28
    else:
        if angle in {"personal_reaction", "gentle_follow_up"}:
            score += 0.24

    if uncertainty < 0.20 and angle in {"example_probe", "preference_probe", "clarify"}:
        score -= 0.25
    if evidence < 0.15 and angle == "broader_view":
        score -= 0.20

    return score


def _angle_plan(bundle: DisplayReserveTopicBundle) -> tuple[str, ...]:
    """Return the deterministic, uncertainty-aware angle plan for one bundle."""

    candidates = (
        "primary",
        "example_probe",
        "preference_probe",
        "continuity_follow_up",
        "meaning_or_update",
        "gentle_status_check",
        "public_reaction",
        "broader_view",
        "local_association",
        "personal_reaction",
        "gentle_follow_up",
    )
    ordered = sorted(
        candidates,
        key=lambda angle: (
            -_angle_priority(bundle, angle),
            0 if angle == "primary" else 1,
            angle,
        ),
    )

    selected: list[str] = []
    question_count = 0
    for angle in ordered:
        if angle == "primary":
            selected.append(angle)
            continue
        is_question = angle in _QUESTION_ANGLES
        if is_question and question_count >= 2:
            continue
        selected.append(angle)
        if is_question:
            question_count += 1
        if len(selected) >= _DEFAULT_MAX_CARDS_PER_TOPIC:
            break
    if "primary" not in selected:
        selected.insert(0, "primary")
    return tuple(selected[:_DEFAULT_MAX_CARDS_PER_TOPIC])


def _expanded_candidate(
    bundle: DisplayReserveTopicBundle,
    *,
    angle: str,
) -> AmbientDisplayImpulseCandidate:
    """Return one expanded reserve candidate for a concrete card angle."""

    candidate = bundle.representative_candidate()
    context = _context(candidate.generation_context)
    context["semantic_topic_key"] = bundle.semantic_topic_key
    context["semantic_bundle_id"] = bundle.bundle_id
    context["support_sources"] = bundle.support_sources
    context["support_families"] = bundle.support_families
    context["support_count"] = bundle.candidate_count
    context["bundle_uncertainty"] = bundle.uncertainty
    context["bundle_evidence"] = bundle.evidence_score
    context["expansion_angle"] = angle
    context["bundle_kind"] = _bundle_kind(bundle)

    if angle == "primary":
        headline = _sanitize_text(candidate.headline or candidate.title or _bundle_anchor(candidate), max_len=128)
        body = _sanitize_text(candidate.body, max_len=128)
        if not body:
            _, body = _secondary_copy(candidate, angle="gentle_status_check")
    else:
        headline, body = _secondary_copy(candidate, angle=angle)

    headline = _sanitize_text(headline or _bundle_anchor(candidate), max_len=128)
    body = _sanitize_text(body, max_len=128)
    if not body:
        body = "Wollen wir kurz darauf schauen?"

    reason_parts = [candidate.reason, f"expansion={angle}", f"bundle={bundle.bundle_id}"]
    return replace(
        candidate,
        topic_key=_card_key(
            semantic_topic_key=bundle.semantic_topic_key,
            angle=angle,
            bundle_id=bundle.bundle_id,
        ),
        semantic_topic_key=bundle.semantic_topic_key,
        action=_expanded_action(candidate.action, angle=angle),
        headline=headline,
        body=body,
        reason=_sanitize_text("; ".join(part for part in reason_parts if part), max_len=160),
        generation_context=context,
        expansion_angle=angle,
        support_sources=bundle.support_sources,
    )


def _bundle_order_score(
    bundle: DisplayReserveTopicBundle,
    *,
    selected: Sequence[DisplayReserveTopicBundle],
    base_rank: int,
) -> float:
    """Return one multi-objective score for whole-page bundle ordering."""

    base = _safe_float(bundle.primary_candidate.salience, default=0.0)
    evidence = 0.15 * bundle.evidence_score
    uncertainty = 0.12 * bundle.uncertainty

    if not selected:
        novelty = 0.0
    else:
        seen_tokens = set().union(*(_bundle_family_tokens(entry) for entry in selected))
        bundle_tokens = _bundle_family_tokens(bundle)
        fresh = len(bundle_tokens - seen_tokens)
        overlap = len(bundle_tokens & seen_tokens)
        novelty = 0.04 * fresh - 0.02 * overlap

    return base + evidence + uncertainty + novelty - 0.01 * base_rank


def _order_bundles(
    bundles: Sequence[DisplayReserveTopicBundle],
) -> tuple[DisplayReserveTopicBundle, ...]:
    """Return one deterministic whole-page order for bundles."""

    representatives = tuple(bundle.representative_candidate() for bundle in bundles)
    diverse = select_diverse_candidates(representatives, max_items=len(representatives))
    bundle_by_id = {bundle.bundle_id: bundle for bundle in bundles}

    seeded: list[DisplayReserveTopicBundle] = []
    seen_ids: set[str] = set()
    for representative in diverse:
        context = _context(representative.generation_context)
        bundle_id = _sanitize_text(context.get("semantic_bundle_id"), max_len=32)
        if bundle_id and bundle_id in bundle_by_id and bundle_id not in seen_ids:
            seeded.append(bundle_by_id[bundle_id])
            seen_ids.add(bundle_id)

    for bundle in bundles:
        if bundle.bundle_id not in seen_ids:
            seeded.append(bundle)
            seen_ids.add(bundle.bundle_id)

    remaining = list(seeded)
    selected: list[DisplayReserveTopicBundle] = []
    while remaining:
        ranked = max(
            enumerate(remaining),
            key=lambda item: _bundle_order_score(item[1], selected=selected, base_rank=item[0]),
        )[0]
        selected.append(remaining.pop(ranked))
    return tuple(selected)


def _expanded_fingerprint(candidate: AmbientDisplayImpulseCandidate) -> str:
    """Return one near-duplicate fingerprint for expanded surfaces."""

    return "||".join(
        (
            _normalize_identifier(candidate.semantic_key(), max_len=96),
            _normalize_identifier(candidate.action, max_len=24),
            _normalize_identifier(candidate.headline, max_len=128),
            _normalize_identifier(candidate.body, max_len=128),
        )
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

    ordered_bundles = _order_bundles(bundles)
    expanded: list[AmbientDisplayImpulseCandidate] = []
    seen_cards: set[str] = set()
    max_passes = max(1, min(int(max_cards_per_topic), _DEFAULT_MAX_CARDS_PER_TOPIC))
    angle_cache = {bundle.bundle_id: _angle_plan(bundle) for bundle in ordered_bundles}

    for pass_index in range(max_passes):
        for bundle in ordered_bundles:
            if len(expanded) >= limited_target:
                return tuple(expanded)

            angles = angle_cache[bundle.bundle_id]
            if pass_index >= len(angles):
                continue

            candidate = _expanded_candidate(bundle, angle=angles[pass_index])
            fingerprint = _expanded_fingerprint(candidate)
            if fingerprint in seen_cards:
                continue
            seen_cards.add(fingerprint)
            expanded.append(candidate)

    return tuple(expanded[:limited_target])


__all__ = [
    "DisplayReserveTopicBundle",
    "bundle_display_reserve_candidates",
    "expand_display_reserve_candidates",
]
