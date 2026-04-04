"""Derive positive ambient display impulses from Twinr's companion state.

This module turns the same structured state that already drives prompt-time
mindshare and positive-engagement behavior into short, silent display impulses.
The goal is not to fabricate a new personality layer, but to let Twinr's
evolving interests, co-attention, and tone become visible outside of spoken
turns.

Selection and wording stay generic:

- no topic, place, or benchmark-specific hardcoding
- only positive or gently inviting impulses
- bounded actions derived from existing appetite / co-attention state
- fresh continuity threads may become low-pressure ambient hints even when the
  stricter spoken-turn policy still stays silent
- light tone variation from learned verbosity and humor, not from ad-hoc
  randomness in the runtime loop
"""

# CHANGELOG: 2026-03-27
# BUG-1: Fixed topic-key semantics. topic_key is now a concrete per-candidate key
#        and semantic_topic_key is now a grouped thread key, preventing cooldown /
#        dedupe collisions across unrelated sources.
# BUG-2: Removed day-dependent ranking jitter that caused silent candidate drift
#        unrelated to state, despite the module contract forbidding ad-hoc runtime
#        randomness in the selection loop.
# BUG-3: Added finite-score coercion, storage-safe context coercion, per-item
#        fault isolation, and diagnostics so malformed inputs do not blank the
#        whole ambient lane or poison ranking with NaN / inf.
# BUG-4: generation_context["candidate_family"] now carries the actual planning
#        family instead of the incorrect hard-coded "mindshare" value.
# SEC-1: Sanitized untrusted titles / summaries / reasons before forwarding them
#        to display fields and downstream rewrite context, stripping control and
#        bidi-format characters and bounding payload sizes.
# SEC-2: Hard-capped output and selection-pool sizes to protect Raspberry Pi 4
#        deployments and downstream LLM budgets from oversized requests.
# IMP-1: Upgraded ranking from plain top-k sorting to diversity-aware greedy
#        reranking with semantic dedupe and redundancy penalties.
# IMP-2: Added structured source-provenance / explanation metadata so downstream
#        rewrite steps can remain transparent about whether a card comes from
#        memory, place, or public-world context.

from __future__ import annotations

import logging
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from twinr.agent.personality._display_utils import (
    normalized_text as _normalized_text,
    stable_fraction as _stable_fraction,
    truncate_text as _truncate_text,
)
from twinr.agent.personality.display_impulse_copy import build_ambient_display_impulse_copy
from twinr.agent.personality.intelligence.models import WorldInterestSignal
from twinr.agent.personality.models import PersonalitySnapshot
from twinr.agent.personality.positive_engagement import (
    PositiveEngagementTopicPolicy,
    derive_positive_engagement_policy,
)
from twinr.agent.personality.self_expression import (
    CompanionMindshareItem,
    build_mindshare_items,
)

_LOG = logging.getLogger(__name__)

_ACTION_PRIORITY: Final[dict[str, float]] = {
    "hint": 0.08,
    "brief_update": 0.15,
    "ask_one": 0.22,
    "invite_follow_up": 0.28,
}
_ATTENTION_PRIORITY: Final[dict[str, float]] = {
    "background": 0.00,
    "growing": 0.03,
    "forming": 0.08,
    "shared_thread": 0.12,
}

# BREAKING: Ambient display output is now hard-capped. This protects Pi-class
# devices and downstream rewrite/token budgets from oversized caller requests.
_MAX_DISPLAY_ITEMS: Final[int] = 12
_MAX_SELECTION_POOL: Final[int] = 18
_MAX_TITLE_LEN: Final[int] = 96
_MAX_SUMMARY_LEN: Final[int] = 160
_MAX_EYEBROW_LEN: Final[int] = 32
_MAX_HEADLINE_LEN: Final[int] = 96
_MAX_BODY_LEN: Final[int] = 160
_MAX_REASON_LEN: Final[int] = 160
_MAX_CONTEXT_STR_LEN: Final[int] = 160
_AMBIENT_CONTINUITY_FRESHNESS: Final[frozenset[str]] = frozenset({"current", "recent"})

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"\w+", re.UNICODE)
_STRIP_TEXT_CODEPOINTS: Final[frozenset[int]] = frozenset(
    {
        0x00AD,  # soft hyphen
        0x034F,  # combining grapheme joiner
        0x061C,  # arabic letter mark
        0x180E,  # mongolian vowel separator
        0x200B,  # zero width space
        0x2060,  # word joiner
        0x2066,  # left-to-right isolate
        0x2067,  # right-to-left isolate
        0x2068,  # first strong isolate
        0x2069,  # pop directional isolate
        0x202A,  # left-to-right embedding
        0x202B,  # right-to-left embedding
        0x202C,  # pop directional formatting
        0x202D,  # left-to-right override
        0x202E,  # right-to-left override
        0xFEFF,  # zero width no-break space
    }
)


@dataclass(frozen=True, slots=True)
class AmbientDisplayImpulseCandidate:
    """Describe one short silent impulse Twinr may show in the display reserve.

    Attributes:
        topic_key: Stable unique key for this concrete reserve-card candidate.
        semantic_topic_key: Stable thread key shared by related card variants
            that should retire and learn together.
        title: Human-readable topic title.
        source: Structured source family such as ``continuity`` or ``place``.
        action: Bounded engagement action this impulse reflects.
        attention_state: Durable co-attention/background label.
        salience: Relative strength of the current topic.
        eyebrow: Small top label for the HDMI reserve card.
        headline: Main short line for the card.
        body: Secondary calm conversational line.
        symbol: Emoji symbol token suitable for display rendering.
        accent: Visual accent token suitable for display rendering.
        reason: Short auditable explanation for selection/debugging.
        candidate_family: Generic reserve-bus family used for planning mix,
            such as ``world``, ``memory_follow_up``, or ``social``.
        generation_context: Optional extra structured context for the bounded
            reserve-lane LLM rewrite step. This should stay generic and
            storage-safe, for example a topic summary or a memory-follow-up
            rationale, never a hidden prompt string.
    """

    topic_key: str
    title: str
    source: str
    action: str
    attention_state: str
    salience: float
    eyebrow: str
    headline: str
    body: str
    symbol: str
    accent: str
    reason: str
    semantic_topic_key: str = ""
    candidate_family: str = "general"
    generation_context: Mapping[str, object] | None = None
    expansion_angle: str = ""
    support_sources: tuple[str, ...] = ()

    def semantic_key(self) -> str:
        """Return the grouped feedback/learning key for this candidate."""

        return _topic_key(self.semantic_topic_key) or _topic_key(self.topic_key)


def _topic_key(value: object | None) -> str:
    """Return one stable topic key for cooldown and dedupe behavior."""

    return _normalized_text(value).casefold()


def _compose_topic_key(*parts: object | None) -> str:
    """Join stable key parts into one normalized cooldown / dedupe key."""

    normalized_parts = [part for part in (_topic_key(part) for part in parts) if part]
    return "::".join(normalized_parts)


def _safe_float(
    value: object,
    *,
    default: float = 0.0,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Return one finite float, optionally clamped to a bounded interval."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    if minimum is not None and number < minimum:
        number = minimum
    if maximum is not None and number > maximum:
        number = maximum
    return number


def _sanitize_text(
    value: object | None,
    *,
    max_len: int,
    fallback: str = "",
) -> str:
    """Normalize, bound, and de-risk text forwarded to display / LLM context."""

    text = _normalized_text(value)
    if not text:
        return fallback
    text = unicodedata.normalize("NFKC", text)
    cleaned: list[str] = []
    for char in text:
        codepoint = ord(char)
        category = unicodedata.category(char)
        if char in "\t\n\r":
            cleaned.append(" ")
            continue
        if codepoint in _STRIP_TEXT_CODEPOINTS:
            continue
        if category in {"Cc", "Cs"}:
            continue
        cleaned.append(char)
    collapsed = " ".join("".join(cleaned).split())
    return _truncate_text(collapsed, max_len=max_len) or fallback


def _sanitize_visual_token(
    value: object | None,
    *,
    max_len: int = 8,
    fallback: str = "",
) -> str:
    """Return one compact printable visual token for accents / emoji."""

    text = _normalized_text(value)
    if not text:
        return fallback
    cleaned: list[str] = []
    for char in text.strip():
        if char.isspace():
            continue
        category = unicodedata.category(char)
        if category in {"Cc", "Cs"}:
            continue
        cleaned.append(char)
    return "".join(cleaned)[:max_len] or fallback


def _storage_safe_value(
    value: object,
    *,
    max_len: int = _MAX_CONTEXT_STR_LEN,
) -> object:
    """Coerce values into JSON-/storage-safe primitives for generation_context."""

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(_safe_float(value, default=0.0, minimum=-1_000_000.0, maximum=1_000_000.0), 6)
    if isinstance(value, str):
        return _sanitize_text(value, max_len=max_len)
    enum_value = getattr(value, "value", None)
    if enum_value is not None and enum_value is not value:
        return _storage_safe_value(enum_value, max_len=max_len)
    if isinstance(value, Mapping):
        return {
            _sanitize_text(key, max_len=48): _storage_safe_value(item, max_len=max_len)
            for key, item in value.items()
            if _sanitize_text(key, max_len=48)
        }
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        items: list[object] = []
        for index, item in enumerate(value):
            if index >= 8:
                break
            items.append(_storage_safe_value(item, max_len=max_len))
        return tuple(items)
    return _sanitize_text(value, max_len=max_len)


def _candidate_score(
    item: CompanionMindshareItem,
    policy: PositiveEngagementTopicPolicy,
) -> float:
    """Return one bounded ranking score for display-candidate ordering."""

    # Stable epsilon tie-break only; no date-driven selection drift.
    epsilon = _safe_float(
        _stable_fraction(item.title, item.source, policy.action, policy.attention_state),
        default=0.0,
        minimum=0.0,
        maximum=1.0,
    ) * 0.0001
    return (
        _safe_float(getattr(item, "salience", 0.0), default=0.0, minimum=0.0, maximum=4.0)
        + _ACTION_PRIORITY.get(_topic_key(policy.action), 0.0)
        + _ATTENTION_PRIORITY.get(_topic_key(policy.attention_state), 0.0)
        + epsilon
    )


def _candidate_family_for_source(source: object | None) -> str:
    """Map one generic mindshare source onto a reserve-planning family."""

    normalized = _topic_key(source)
    if normalized in {"continuity", "relationship"}:
        return "memory_thread"
    if normalized == "place":
        return "place"
    if normalized in {"situational_awareness", "regional_news", "local_news", "world"}:
        return "world"
    return "general"


def _information_source_kind(source: object | None) -> str:
    """Describe the transparency / provenance class of one source."""

    normalized = _topic_key(source)
    if normalized in {"continuity", "relationship"}:
        return "conversation_history"
    if normalized == "place":
        return "place_context"
    if normalized in {"situational_awareness", "regional_news", "local_news", "world"}:
        return "public_world_context"
    return "general_context"


def _semantic_topic_key_for_item(item: CompanionMindshareItem) -> str:
    """Return a grouped thread key shared by related card variants."""

    for attribute in (
        "semantic_topic_key",
        "semantic_key",
        "thread_key",
        "memory_key",
        "group_key",
        "topic_key",
    ):
        key = _topic_key(getattr(item, attribute, None))
        if key:
            return key
    return _compose_topic_key(
        _candidate_family_for_source(getattr(item, "source", None)),
        _sanitize_text(getattr(item, "title", None), max_len=_MAX_TITLE_LEN),
    )


def _support_sources_for_item(
    item: CompanionMindshareItem,
    *,
    fallback_source: str,
) -> tuple[str, ...]:
    """Return one compact, sanitized provenance tuple."""

    raw_support_sources = getattr(item, "support_sources", None)
    if isinstance(raw_support_sources, Sequence) and not isinstance(raw_support_sources, (str, bytes, bytearray)):
        values = []
        for source in raw_support_sources:
            safe_source = _sanitize_text(source, max_len=48)
            if safe_source:
                values.append(safe_source)
        deduped = tuple(dict.fromkeys(values))
        if deduped:
            return deduped
    return (fallback_source,) if fallback_source else ()


def _mindshare_card_intent(
    item: CompanionMindshareItem,
    policy: PositiveEngagementTopicPolicy,
    *,
    summary: str,
) -> dict[str, str]:
    """Return structured semantic card intent for one mindshare candidate."""

    anchor = _sanitize_text(item.title, max_len=_MAX_TITLE_LEN, fallback="dem Thema")
    family = _candidate_family_for_source(item.source)
    if family == "memory_thread":
        return {
            "topic_semantics": f"gemeinsamer Faden zu {anchor}",
            "statement_intent": f"Twinr soll eine konkrete Beobachtung oder einen ruhigen Rueckbezug zu {anchor} machen.",
            "cta_intent": "Zu einer kurzen Meinung, Ergaenzung oder einem Weiterreden einladen.",
            "relationship_stance": "warm und aufmerksam statt behauptend",
        }
    if family == "world":
        return {
            "topic_semantics": f"oeffentliches Thema zu {anchor}",
            "statement_intent": f"Twinr soll eine konkrete Beobachtung dazu machen, was bei {anchor} gerade Thema ist.",
            "cta_intent": "Zu einer kurzen Meinung oder Einordnung einladen.",
            "relationship_stance": "ruhig beobachtend mit leichter Haltung",
        }
    if family == "place":
        return {
            "topic_semantics": f"Ort oder Region {anchor} im aktuellen Blick",
            "statement_intent": f"Twinr soll eine konkrete Beobachtung zu {anchor} machen.",
            "cta_intent": "Zu einer kurzen Reaktion oder Erinnerung einladen.",
            "relationship_stance": "alltagsnah und lokal statt abstrakt",
        }
    topic_semantics = summary or anchor
    return {
        "topic_semantics": topic_semantics,
        "statement_intent": f"Twinr soll zu {anchor} eine kurze konkrete Beobachtung machen.",
        "cta_intent": "Zu einer kurzen Reaktion oder Meinung einladen.",
        "relationship_stance": "ruhig, freundlich und leicht eigen",
    }


def _token_set(*values: object | None) -> frozenset[str]:
    """Return one compact lexical set for cheap similarity / redundancy checks."""

    tokens: set[str] = set()
    for value in values:
        text = _sanitize_text(value, max_len=160).casefold()
        if not text:
            continue
        for token in _TOKEN_RE.findall(text):
            if len(token) > 1:
                tokens.add(token)
    return frozenset(tokens)


def _jaccard_similarity(left: frozenset[str], right: frozenset[str]) -> float:
    """Return a cheap bounded lexical overlap score."""

    if not left or not right:
        return 0.0
    intersection = len(left & right)
    if intersection <= 0:
        return 0.0
    union = len(left | right)
    if union <= 0:
        return 0.0
    return intersection / union


def _candidate_similarity(
    left: AmbientDisplayImpulseCandidate,
    right: AmbientDisplayImpulseCandidate,
) -> float:
    """Return one cheap semantic / lexical redundancy score in [0, 1]."""

    if left.semantic_key() and left.semantic_key() == right.semantic_key():
        return 1.0
    if left.topic_key and left.topic_key == right.topic_key:
        return 0.98

    title_overlap = _jaccard_similarity(_token_set(left.title), _token_set(right.title))
    headline_overlap = _jaccard_similarity(_token_set(left.headline), _token_set(right.headline))
    body_overlap = _jaccard_similarity(_token_set(left.body), _token_set(right.body))

    similarity = (
        0.52 * title_overlap
        + 0.20 * headline_overlap
        + 0.08 * body_overlap
        + (0.12 if left.candidate_family == right.candidate_family else 0.0)
        + (0.05 if _topic_key(left.source) == _topic_key(right.source) else 0.0)
        + (0.03 if _topic_key(left.action) == _topic_key(right.action) else 0.0)
    )
    return min(1.0, similarity)


def _selection_utility(
    base_score: float,
    candidate: AmbientDisplayImpulseCandidate,
    selected: Sequence[AmbientDisplayImpulseCandidate],
) -> float:
    """Blend relevance with set-level diversity for one greedy selection step."""

    if not selected:
        return base_score

    max_similarity = max(_candidate_similarity(candidate, existing) for existing in selected)
    same_family_count = sum(1 for existing in selected if existing.candidate_family == candidate.candidate_family)
    same_action_count = sum(1 for existing in selected if _topic_key(existing.action) == _topic_key(candidate.action))
    same_source_count = sum(1 for existing in selected if _topic_key(existing.source) == _topic_key(candidate.source))

    utility = base_score
    utility -= 0.45 * max_similarity
    utility -= 0.11 * same_family_count
    utility -= 0.05 * same_action_count
    utility -= 0.04 * same_source_count
    return utility


def _collapse_semantic_duplicates(
    candidates: Sequence[tuple[float, AmbientDisplayImpulseCandidate]],
) -> list[tuple[float, AmbientDisplayImpulseCandidate]]:
    """Keep only the strongest card variant per semantic topic thread."""

    best_by_semantic: dict[str, tuple[float, AmbientDisplayImpulseCandidate]] = {}
    for score, candidate in candidates:
        semantic_key = candidate.semantic_key() or candidate.topic_key
        previous = best_by_semantic.get(semantic_key)
        if previous is None:
            best_by_semantic[semantic_key] = (score, candidate)
            continue
        previous_score, previous_candidate = previous
        if (
            score > previous_score
            or (
                math.isclose(score, previous_score)
                and (candidate.salience, candidate.headline, candidate.topic_key)
                > (previous_candidate.salience, previous_candidate.headline, previous_candidate.topic_key)
            )
        ):
            best_by_semantic[semantic_key] = (score, candidate)
    return sorted(
        best_by_semantic.values(),
        key=lambda entry: (entry[0], entry[1].salience, entry[1].headline, entry[1].topic_key),
        reverse=True,
    )


def _diversity_rerank(
    candidates: Sequence[tuple[float, AmbientDisplayImpulseCandidate]],
    *,
    limit: int,
) -> tuple[AmbientDisplayImpulseCandidate, ...]:
    """Greedily select a diverse, high-salience set under a fixed display budget."""

    pool = _collapse_semantic_duplicates(candidates)
    selected: list[AmbientDisplayImpulseCandidate] = []
    while pool and len(selected) < limit:
        best_index: int | None = None
        best_rank: tuple[float, float, float, str, str] | None = None
        for index, (base_score, candidate) in enumerate(pool):
            utility = _selection_utility(base_score, candidate, selected)
            rank = (
                utility,
                base_score,
                candidate.salience,
                candidate.headline,
                candidate.topic_key,
            )
            if best_rank is None or rank > best_rank:
                best_index = index
                best_rank = rank
        if best_index is None:
            break
        _score, chosen = pool.pop(best_index)
        selected.append(chosen)
    return tuple(selected)


def _candidate_for_item(
    item: CompanionMindshareItem,
    policy: PositiveEngagementTopicPolicy,
    *,
    local_now: datetime | None,
) -> AmbientDisplayImpulseCandidate | None:
    """Convert one prompt-facing mindshare item into a display impulse."""

    if _topic_key(policy.action) == "silent":
        return None
    source = _sanitize_text(getattr(item, "source", None), max_len=48, fallback="general")
    if _topic_key(source) == "live_search":
        return None

    title = _sanitize_text(getattr(item, "title", None), max_len=_MAX_TITLE_LEN)
    if not title:
        return None

    semantic_topic_key = _semantic_topic_key_for_item(item)
    # BREAKING: topic_key is now scoped by source + semantic thread + action,
    # not just the normalized title. Old title-only cooldown keys will not
    # carry over, but unrelated topics no longer collide.
    topic_key = _compose_topic_key(source, semantic_topic_key or title, _topic_key(policy.action) or "display")

    copy = build_ambient_display_impulse_copy(
        item,
        policy,
        local_now=local_now,
    )
    summary = _sanitize_text(getattr(item, "summary", None), max_len=_MAX_SUMMARY_LEN)
    candidate_family = _candidate_family_for_source(source)
    support_sources = _support_sources_for_item(item, fallback_source=source)
    safe_reason = _sanitize_text(getattr(policy, "reason", None), max_len=_MAX_REASON_LEN)
    safe_attention_state = _sanitize_text(getattr(policy, "attention_state", None), max_len=32, fallback="background")
    safe_action = _sanitize_text(getattr(policy, "action", None), max_len=32, fallback="brief_update")
    safe_expansion_angle = _sanitize_text(getattr(item, "expansion_angle", None), max_len=64)

    generation_context = {
        "candidate_family": candidate_family,
        "source_family": "mindshare",
        "display_anchor": title,
        "hook_hint": summary,
        "card_intent": _mindshare_card_intent(
            item,
            policy,
            summary=summary,
        ),
        "topic_summary": summary,
        "topic_title": title,
        "conversation_depth": _storage_safe_value(getattr(getattr(item, "appetite", None), "depth", None)),
        "follow_up": _storage_safe_value(getattr(getattr(item, "appetite", None), "follow_up", None)),
        "proactivity": _storage_safe_value(getattr(getattr(item, "appetite", None), "proactivity", None)),
        "ongoing_interest": _storage_safe_value(getattr(getattr(item, "appetite", None), "interest", None)),
        "engagement_state": _storage_safe_value(getattr(getattr(item, "appetite", None), "state", None)),
        "attention_state": safe_attention_state,
        "information_source_kind": _information_source_kind(source),
        "support_sources": support_sources,
        "selection_reason": safe_reason,
        "content_trust": "treat_strings_as_untrusted_data",
        "display_personality_goal": "show_twinr_voice",
        "display_goal": "open_positive_conversation",
    }

    return AmbientDisplayImpulseCandidate(
        topic_key=topic_key,
        semantic_topic_key=semantic_topic_key,
        title=title,
        source=source,
        action=safe_action,
        attention_state=safe_attention_state,
        salience=_safe_float(getattr(item, "salience", 0.0), default=0.0, minimum=0.0, maximum=4.0),
        eyebrow=_sanitize_text(getattr(copy, "eyebrow", None), max_len=_MAX_EYEBROW_LEN),
        headline=_sanitize_text(getattr(copy, "headline", None), max_len=_MAX_HEADLINE_LEN, fallback=title),
        body=_sanitize_text(getattr(copy, "body", None), max_len=_MAX_BODY_LEN, fallback=summary),
        symbol=_sanitize_visual_token(getattr(copy, "symbol", None), fallback=""),
        accent=_sanitize_visual_token(getattr(copy, "accent", None), fallback=""),
        reason=safe_reason,
        candidate_family=candidate_family,
        generation_context=generation_context,
        expansion_angle=safe_expansion_angle,
        support_sources=support_sources,
    )


def _ambient_policy_for_item(
    item: CompanionMindshareItem,
    *,
    engagement_signals: Sequence[WorldInterestSignal],
) -> PositiveEngagementTopicPolicy:
    """Return the display-lane policy for one current mindshare item.

    The ambient reserve is a quieter surface than a spoken turn. Fresh
    continuity threads may therefore appear as one light hint even when the
    stricter spoken-turn policy still stays `silent` because no durable
    engagement signal has formed yet.
    """

    policy = derive_positive_engagement_policy(
        item,
        engagement_signals=engagement_signals,
    )
    if _topic_key(policy.action) != "silent":
        return policy

    appetite = getattr(item, "appetite", None)
    appetite_state = _topic_key(getattr(appetite, "state", None))
    if appetite_state in {"avoid", "cooling"}:
        return policy

    if _topic_key(getattr(item, "source", None)) != "continuity":
        return policy
    if _topic_key(getattr(item, "freshness", None)) not in _AMBIENT_CONTINUITY_FRESHNESS:
        return policy

    return PositiveEngagementTopicPolicy(
        title=policy.title,
        salience=policy.salience,
        attention_state="growing",
        action="hint",
        reason="ambient_continuity_hint",
    )


def build_ambient_display_impulse_candidates(
    snapshot: PersonalitySnapshot | None,
    *,
    engagement_signals: Sequence[WorldInterestSignal] = (),
    local_now: datetime | None = None,
    max_items: int = 4,
) -> tuple[AmbientDisplayImpulseCandidate, ...]:
    """Build the current positive display-impulse candidates.

    The returned candidates are already filtered to positive or gently inviting
    actions. Topics in ``cooling`` or ``avoid`` states naturally disappear
    because the underlying positive-engagement policy resolves them to
    ``silent``.
    """

    try:
        requested_max = int(max_items)
    except (TypeError, ValueError):
        requested_max = 4
    limited_max = max(1, min(requested_max, _MAX_DISPLAY_ITEMS))
    selection_pool_size = min(max(limited_max * 3, 6), _MAX_SELECTION_POOL)

    try:
        items = build_mindshare_items(
            snapshot,
            engagement_signals=engagement_signals,
            max_items=selection_pool_size,
        )
    except Exception:
        _LOG.warning("ambient display candidate build failed in build_mindshare_items", exc_info=True)
        return ()

    candidates: list[tuple[float, AmbientDisplayImpulseCandidate]] = []
    for item in items:
        try:
            policy = _ambient_policy_for_item(
                item,
                engagement_signals=engagement_signals,
            )
            candidate = _candidate_for_item(
                item,
                policy,
                local_now=local_now,
            )
            if candidate is None:
                continue
            score = _candidate_score(item, policy)
            candidates.append((score, candidate))
        except Exception:
            _LOG.debug("skipping malformed ambient display candidate input", exc_info=True)
            continue

    return _diversity_rerank(candidates, limit=limited_max)
