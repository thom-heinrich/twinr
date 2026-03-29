# CHANGELOG: 2026-03-29
# BUG-1: Stop surfacing expired continuity threads and stale world signals. The old code stored
# freshness metadata in generation_context but never used it for selection, so outdated topics
# could still reach the reserve lane.
# BUG-2: Respect max_items=0. The old implementation forced at least one candidate via
# max(1, int(max_items)), which silently violated caller intent.
# BUG-3: Use full semantic normalization for dedupe / exclusion / engagement matching instead of
# display-length truncation. The old code could silently collide long topics that share a prefix.
# SEC-1: Sanitize untrusted snapshot text before forwarding it to display fields and downstream
# generation_context, stripping control / bidi characters, bounding payload sizes, and neutralizing
# obvious prompt-injection phrases from external memory/news inputs.
# IMP-1: Add trust-aware and time-aware scoring. Signals now honor optional trust_score plus
# freshness / recency decay, aligning reserve selection with 2026 memory-agent retrieval patterns.
# IMP-2: Replace full-list sorts with heapq.nlargest partial selection and add per-item fault
# isolation so one malformed snapshot item does not blank the whole reserve lane on Raspberry Pi 4.
# IMP-3: Add lightweight semantic near-duplicate suppression before diversity reranking so small
# wording variants do not crowd out topic variety.
# IMP-4: Align topic_key semantics with the newer display-impulse contract: topic_key is now a
# concrete per-candidate key and semantic_topic_key stays the grouped thread key.

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

import logging
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from heapq import nlargest
from typing import Final

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

_LOG = logging.getLogger(__name__)

_ASK_ONE_SALIENCE: Final[float] = 0.78
_BRIEF_UPDATE_SALIENCE: Final[float] = 0.60
_LOCAL_WORLD_SOURCES: Final[frozenset[str]] = frozenset({"local_news", "regional_news", "place"})
_DEFAULT_PLACE_LIMIT: Final[int] = 3
_MIN_TRUST_SCORE: Final[float] = 0.15
_MAX_TOPIC_KEY_LEN: Final[int] = 2048
_MAX_CONTEXT_ITEMS: Final[int] = 8
_NEAR_DUPLICATE_JACCARD: Final[float] = 0.82

_CONTINUITY_HALF_LIFE_HOURS: Final[float] = 24.0 * 30.0
_RELATIONSHIP_HALF_LIFE_HOURS: Final[float] = 24.0 * 45.0
_PLACE_HALF_LIFE_HOURS: Final[float] = 24.0 * 14.0
_WORLD_HALF_LIFE_HOURS: Final[float] = 18.0

_SNAPSHOT_SOURCE_PRIORITY: Final[dict[str, int]] = {
    "continuity": 4,
    "relationship": 3,
    "place": 2,
    "situational_awareness": 1,
}

_ACTION_PRIORITY: Final[dict[str, int]] = {
    "ask_one": 3,
    "brief_update": 2,
    "hint": 1,
}
_ATTENTION_PRIORITY: Final[dict[str, int]] = {
    "shared_thread": 3,
    "growing": 2,
    "forming": 1,
    "background": 0,
}

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"\w+", re.UNICODE)
_STRIP_TEXT_CODEPOINTS: Final[frozenset[int]] = frozenset(
    {
        0x00AD,  # soft hyphen
        0x034F,  # combining grapheme joiner
        0x061C,  # arabic letter mark
        0x180E,  # mongolian vowel separator
        0x200B,  # zero width space
        0x200C,  # zero width non-joiner
        0x200D,  # zero width joiner
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
_DANGEROUS_TEXT_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions?", re.IGNORECASE),
    re.compile(r"developer\s+mode", re.IGNORECASE),
    re.compile(r"reveal\s+.+prompt", re.IGNORECASE),
    re.compile(r"bypass\s+safety", re.IGNORECASE),
    re.compile(r"delete\s+.+data", re.IGNORECASE),
)


def _utc_now() -> datetime:
    """Return one aware UTC timestamp."""
    return datetime.now(tz=UTC)


def _safe_float(
    value: object | None,
    *,
    default: float = 0.0,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Return one finite float, optionally clamped."""
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


def _safe_int(
    value: object | None,
    *,
    default: int = 0,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Return one bounded integer."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    if minimum is not None and number < minimum:
        number = minimum
    if maximum is not None and number > maximum:
        number = maximum
    return number


def _normalize_text(value: object | None) -> str:
    """Return normalized text with control and bidi artifacts removed."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
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
    return " ".join("".join(cleaned).split())


def _contains_dangerous_text(value: object | None) -> bool:
    """Return whether text contains obvious instruction-like attack phrases."""
    text = _normalize_text(value)
    if not text:
        return False
    return any(pattern.search(text) is not None for pattern in _DANGEROUS_TEXT_PATTERNS)


def _neutralize_instruction_like_text(text: str) -> str:
    """Return one prompt-safer version of untrusted external text."""
    sanitized = text
    for pattern in _DANGEROUS_TEXT_PATTERNS:
        sanitized = pattern.sub("[redacted]", sanitized)
    return sanitized


def _semantic_text(value: object | None) -> str:
    """Return one normalized semantic text string for keys and matching."""
    text = _normalize_text(value)
    if not text:
        return ""
    text = _neutralize_instruction_like_text(text)
    return text[:_MAX_TOPIC_KEY_LEN]


def _topic_key(value: object | None) -> str:
    """Return one stable semantic key without display-length truncation artifacts."""
    return _semantic_text(value).casefold()


def _compose_topic_key(*parts: object | None) -> str:
    """Join normalized key parts into one concrete per-candidate key."""
    normalized_parts = [part for part in (_topic_key(part) for part in parts) if part]
    return "::".join(normalized_parts)


def _truncate_text(value: object | None, *, max_len: int, fallback: str = "") -> str:
    """Return one bounded display-safe text field."""
    text = _semantic_text(value)
    if not text:
        return fallback
    return compact_text(text, max_len=max_len) or fallback


def _storage_safe_value(value: object, *, max_len: int = 160) -> object:
    """Return one bounded JSON-/prompt-safe primitive tree."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(_safe_float(value, default=0.0, minimum=-1_000_000.0, maximum=1_000_000.0), 6)
    if isinstance(value, str):
        return _truncate_text(value, max_len=max_len)
    enum_value = getattr(value, "value", None)
    if enum_value is not None and enum_value is not value:
        return _storage_safe_value(enum_value, max_len=max_len)
    if isinstance(value, Mapping):
        safe_mapping: dict[str, object] = {}
        for raw_key, raw_item in value.items():
            key = _truncate_text(raw_key, max_len=48)
            if not key:
                continue
            safe_mapping[key] = _storage_safe_value(raw_item, max_len=max_len)
        return safe_mapping
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        items: list[object] = []
        for index, item in enumerate(value):
            if index >= _MAX_CONTEXT_ITEMS:
                break
            items.append(_storage_safe_value(item, max_len=max_len))
        return tuple(items)
    return _truncate_text(value, max_len=max_len)


def _semantic_tokens(value: object | None) -> frozenset[str]:
    """Return normalized lexical tokens used for lightweight near-duplicate checks."""
    return frozenset(_TOKEN_RE.findall(_topic_key(value)))


def _jaccard_similarity(left: frozenset[str], right: frozenset[str]) -> float:
    """Return token-set Jaccard similarity."""
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _is_near_duplicate(
    *,
    candidate_key: str,
    candidate_tokens: frozenset[str],
    other_key: str,
    other_tokens: frozenset[str],
) -> bool:
    """Return whether two semantic topics are effectively the same reserve subject."""
    if not candidate_key or not other_key:
        return False
    if candidate_key == other_key:
        return True
    if (
        min(len(candidate_key), len(other_key)) >= 24
        and (candidate_key in other_key or other_key in candidate_key)
    ):
        return True
    if min(len(candidate_tokens), len(other_tokens)) >= 2:
        return _jaccard_similarity(candidate_tokens, other_tokens) >= _NEAR_DUPLICATE_JACCARD
    return False


def _parse_datetime(value: object | None) -> datetime | None:
    """Parse one UTC-comparable datetime from common payload forms."""
    if value is None:
        return None
    parsed: datetime
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return None
        try:
            parsed = datetime.fromtimestamp(float(value), tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None
    else:
        text = _normalize_text(value)
        if not text:
            return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _timestamp_sort_key(value: object | None) -> float:
    """Return one monotonic sortable numeric timestamp key."""
    parsed = _parse_datetime(value)
    if parsed is None:
        return float("-inf")
    return parsed.timestamp()


def _call_temporal_guard(item: object, method_name: str, *, now_utc: datetime) -> bool | None:
    """Call one optional model freshness/expiry helper if present."""
    method = getattr(item, method_name, None)
    if not callable(method):
        return None
    for argument in (
        now_utc.isoformat().replace("+00:00", "Z"),
        now_utc.isoformat(),
        now_utc,
    ):
        try:
            return bool(method(argument))
        except TypeError:
            continue
        except Exception:
            _LOG.debug("Temporal guard %s failed for %r.", method_name, item, exc_info=True)
            return None
    try:
        return bool(method())
    except TypeError:
        return None
    except Exception:
        _LOG.debug("Temporal guard %s failed for %r.", method_name, item, exc_info=True)
        return None


def _is_expired(item: object, *, now_utc: datetime) -> bool:
    """Return whether one continuity-style item is expired."""
    guarded = _call_temporal_guard(item, "is_expired", now_utc=now_utc)
    if guarded is not None:
        return guarded
    expires_at = _parse_datetime(getattr(item, "expires_at", None))
    return expires_at is not None and expires_at <= now_utc


def _is_fresh(item: object, *, now_utc: datetime) -> bool:
    """Return whether one world-style item is still fresh."""
    guarded = _call_temporal_guard(item, "is_fresh", now_utc=now_utc)
    if guarded is not None:
        return guarded
    fresh_until = _parse_datetime(getattr(item, "fresh_until", None))
    return fresh_until is None or fresh_until > now_utc


def _half_life_multiplier(
    *,
    updated_at: object | None,
    now_utc: datetime,
    half_life_hours: float,
    floor: float,
) -> float:
    """Return one smooth recency multiplier with a non-zero floor."""
    parsed = _parse_datetime(updated_at)
    if parsed is None or parsed >= now_utc or half_life_hours <= 0.0:
        return 1.0
    age_hours = (now_utc - parsed).total_seconds() / 3600.0
    decay = 0.5 ** (age_hours / half_life_hours)
    bounded = min(1.0, max(0.0, decay))
    return floor + ((1.0 - floor) * bounded)


def _trust_score(item: object) -> float:
    """Return one normalized trust score, defaulting to trusted for old schemas."""
    return _safe_float(getattr(item, "trust_score", 1.0), default=1.0, minimum=0.0, maximum=1.0)


def _trust_multiplier(item: object) -> float:
    """Return one mild salience multiplier derived from trust."""
    return 0.60 + (0.40 * _trust_score(item))


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
        topic = _topic_key(getattr(signal, "topic", None))
        state = _truncate_text(getattr(signal, "engagement_state", None), max_len=24).casefold() or "uncertain"
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
    effective_salience: float,
    *,
    semantic_topic_key: str,
    engagement_states: Mapping[str, str],
) -> tuple[str, str]:
    """Return one bounded action/attention pair for a latent topic."""
    state = engagement_states.get(semantic_topic_key, "")
    if state in {"warm", "resonant"} or effective_salience >= _ASK_ONE_SALIENCE:
        return ("ask_one", "shared_thread")
    if effective_salience >= _BRIEF_UPDATE_SALIENCE:
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


def _candidate_sort_key(candidate: AmbientDisplayImpulseCandidate) -> tuple[float, int, int, int, str]:
    """Return one stable rank key for latent snapshot candidates."""
    source = _truncate_text(candidate.source, max_len=32).casefold()
    return (
        _safe_float(candidate.salience, default=0.0, minimum=0.0, maximum=1.0),
        _SNAPSHOT_SOURCE_PRIORITY.get(source, 1),
        _ACTION_PRIORITY.get(_truncate_text(candidate.action, max_len=24).casefold(), 0),
        _ATTENTION_PRIORITY.get(_truncate_text(candidate.attention_state, max_len=24).casefold(), 0),
        candidate.semantic_key(),
    )


def _item_sort_key(*parts: object) -> tuple[object, ...]:
    """Normalize heterogeneous tuple parts into a comparable ranking key."""
    normalized: list[object] = []
    for part in parts:
        if isinstance(part, (int, float)):
            normalized.append(part)
        else:
            normalized.append(_normalize_text(part).casefold())
    return tuple(normalized)


def _top_items(
    items: Sequence[object],
    *,
    limit: int,
    key_builder,
) -> tuple[object, ...]:
    """Return one resilient top-k subset without full-list sorting."""
    if limit <= 0:
        return ()
    scored: list[tuple[tuple[object, ...], object]] = []
    for item in items:
        try:
            scored.append((key_builder(item), item))
        except Exception:
            _LOG.debug("Skipping malformed reserve snapshot item %r during top-k selection.", item, exc_info=True)
    if not scored:
        return ()
    return tuple(item for _score, item in nlargest(limit, scored, key=lambda pair: pair[0]))


def _continuity_candidate(
    thread: ContinuityThread,
    *,
    engagement_states: Mapping[str, str],
    now_utc: datetime,
) -> AmbientDisplayImpulseCandidate | None:
    """Convert one continuity thread into a reserve candidate."""
    if _is_expired(thread, now_utc=now_utc):
        return None
    if _contains_dangerous_text(getattr(thread, "title", None)):
        return None
    if _trust_score(thread) < _MIN_TRUST_SCORE:
        return None
    semantic_topic_key = _topic_key(getattr(thread, "title", None))
    if not semantic_topic_key or _is_avoided(semantic_topic_key, engagement_states=engagement_states):
        return None

    salience = _safe_float(getattr(thread, "salience", None), default=0.0, minimum=0.0, maximum=1.0)
    salience *= _trust_multiplier(thread)
    salience *= _half_life_multiplier(
        updated_at=getattr(thread, "updated_at", None),
        now_utc=now_utc,
        half_life_hours=_CONTINUITY_HALF_LIFE_HOURS,
        floor=0.55,
    )
    salience = _salience_with_engagement(
        salience,
        semantic_topic_key=semantic_topic_key,
        engagement_states=engagement_states,
        minimum=0.42,
        maximum=0.92,
    )
    action, attention_state = _engagement_attention_state(
        salience,
        semantic_topic_key=semantic_topic_key,
        engagement_states=engagement_states,
    )
    anchor = _truncate_text(getattr(thread, "title", None), max_len=72)
    if not anchor:
        return None
    headline, body = _continuity_copy(anchor, action=action)
    trust_score = _trust_score(thread)
    return AmbientDisplayImpulseCandidate(
        topic_key=_compose_topic_key("continuity", semantic_topic_key),
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
            "hook_hint": _truncate_text(getattr(thread, "summary", None), max_len=160),
            "card_intent": _continuity_card_intent(anchor),
            "topic_title": anchor,
            "topic_summary": _truncate_text(getattr(thread, "summary", None), max_len=180),
            "topic_origin": "personality_snapshot",
            "display_goal": "reopen_shared_thread",
            "updated_at": _truncate_text(getattr(thread, "updated_at", None), max_len=40),
            "expires_at": _truncate_text(getattr(thread, "expires_at", None), max_len=40),
            "content_trust_score": round(trust_score, 3),
            "untrusted_source_data": True,
        },
    )


def _relationship_candidate(
    signal: RelationshipSignal,
    *,
    engagement_states: Mapping[str, str],
    now_utc: datetime,
) -> AmbientDisplayImpulseCandidate | None:
    """Convert one relationship affinity into a reserve candidate."""
    if _truncate_text(getattr(signal, "stance", None), max_len=24).casefold() != "affinity":
        return None
    if _contains_dangerous_text(getattr(signal, "topic", None)):
        return None
    if _trust_score(signal) < _MIN_TRUST_SCORE:
        return None
    semantic_topic_key = _topic_key(getattr(signal, "topic", None))
    if not semantic_topic_key or _is_avoided(semantic_topic_key, engagement_states=engagement_states):
        return None

    salience = _safe_float(getattr(signal, "salience", None), default=0.0, minimum=0.0, maximum=1.0)
    salience *= _trust_multiplier(signal)
    salience *= _half_life_multiplier(
        updated_at=getattr(signal, "updated_at", None),
        now_utc=now_utc,
        half_life_hours=_RELATIONSHIP_HALF_LIFE_HOURS,
        floor=0.60,
    )
    salience = _salience_with_engagement(
        salience,
        semantic_topic_key=semantic_topic_key,
        engagement_states=engagement_states,
        minimum=0.40,
        maximum=0.88,
    )
    action, attention_state = _engagement_attention_state(
        salience,
        semantic_topic_key=semantic_topic_key,
        engagement_states=engagement_states,
    )
    anchor = _truncate_text(getattr(signal, "topic", None), max_len=72)
    if not anchor:
        return None
    headline, body = _continuity_copy(anchor, action=action)
    trust_score = _trust_score(signal)
    return AmbientDisplayImpulseCandidate(
        topic_key=_compose_topic_key("relationship", semantic_topic_key),
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
            "hook_hint": _truncate_text(getattr(signal, "summary", None), max_len=160),
            "card_intent": _continuity_card_intent(anchor),
            "topic_title": anchor,
            "topic_summary": _truncate_text(getattr(signal, "summary", None), max_len=180),
            "topic_origin": "personality_snapshot",
            "display_goal": "reopen_shared_thread",
            "relationship_source": _truncate_text(getattr(signal, "source", None), max_len=48),
            "updated_at": _truncate_text(getattr(signal, "updated_at", None), max_len=40),
            "content_trust_score": round(trust_score, 3),
            "untrusted_source_data": True,
        },
    )


def _place_candidate(
    focus: PlaceFocus,
    *,
    engagement_states: Mapping[str, str],
    now_utc: datetime,
) -> AmbientDisplayImpulseCandidate | None:
    """Convert one place focus into a reserve candidate."""
    if _contains_dangerous_text(getattr(focus, "name", None)):
        return None
    if _trust_score(focus) < _MIN_TRUST_SCORE:
        return None
    semantic_topic_key = _topic_key(getattr(focus, "name", None))
    if not semantic_topic_key or _is_avoided(semantic_topic_key, engagement_states=engagement_states):
        return None

    salience = _safe_float(getattr(focus, "salience", None), default=0.0, minimum=0.0, maximum=1.0)
    salience *= _trust_multiplier(focus)
    salience *= _half_life_multiplier(
        updated_at=getattr(focus, "updated_at", None),
        now_utc=now_utc,
        half_life_hours=_PLACE_HALF_LIFE_HOURS,
        floor=0.62,
    )
    salience = _salience_with_engagement(
        salience,
        semantic_topic_key=semantic_topic_key,
        engagement_states=engagement_states,
        minimum=0.38,
        maximum=0.82,
    )
    action, attention_state = _engagement_attention_state(
        salience,
        semantic_topic_key=semantic_topic_key,
        engagement_states=engagement_states,
    )
    anchor = _truncate_text(getattr(focus, "name", None), max_len=72)
    if not anchor:
        return None
    headline, body = _place_copy(anchor, action=action)
    trust_score = _trust_score(focus)
    return AmbientDisplayImpulseCandidate(
        topic_key=_compose_topic_key("place", semantic_topic_key),
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
            "hook_hint": _truncate_text(getattr(focus, "summary", None), max_len=160),
            "card_intent": _place_card_intent(anchor),
            "topic_title": anchor,
            "topic_summary": _truncate_text(getattr(focus, "summary", None), max_len=180),
            "topic_origin": "personality_snapshot",
            "display_goal": "open_local_context",
            "geography": _truncate_text(getattr(focus, "geography", None), max_len=48),
            "updated_at": _truncate_text(getattr(focus, "updated_at", None), max_len=40),
            "content_trust_score": round(trust_score, 3),
            "untrusted_source_data": True,
        },
    )


def _world_candidate(
    signal: WorldSignal,
    *,
    engagement_states: Mapping[str, str],
    now_utc: datetime,
) -> AmbientDisplayImpulseCandidate | None:
    """Convert one snapshot world signal into a reserve candidate."""
    if not _is_fresh(signal, now_utc=now_utc):
        return None
    if _contains_dangerous_text(getattr(signal, "topic", None)):
        return None
    if _trust_score(signal) < _MIN_TRUST_SCORE:
        return None
    semantic_topic_key = _topic_key(getattr(signal, "topic", None))
    if not semantic_topic_key or _is_avoided(semantic_topic_key, engagement_states=engagement_states):
        return None

    evidence_count = _safe_int(getattr(signal, "evidence_count", None), default=0, minimum=0, maximum=6)
    salience = _safe_float(getattr(signal, "salience", None), default=0.0, minimum=0.0, maximum=1.0)
    salience += min(evidence_count, 3) * 0.03
    salience *= _trust_multiplier(signal)
    salience *= _half_life_multiplier(
        updated_at=getattr(signal, "updated_at", None),
        now_utc=now_utc,
        half_life_hours=_WORLD_HALF_LIFE_HOURS,
        floor=0.68,
    )
    salience = _salience_with_engagement(
        salience,
        semantic_topic_key=semantic_topic_key,
        engagement_states=engagement_states,
        minimum=0.40,
        maximum=0.90,
    )
    action, attention_state = _engagement_attention_state(
        salience,
        semantic_topic_key=semantic_topic_key,
        engagement_states=engagement_states,
    )
    anchor = _truncate_text(getattr(signal, "topic", None), max_len=72)
    if not anchor:
        return None
    headline, body = _world_copy(anchor, action=action)
    trust_score = _trust_score(signal)
    return AmbientDisplayImpulseCandidate(
        topic_key=_compose_topic_key("situational_awareness", semantic_topic_key),
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
            "hook_hint": _truncate_text(getattr(signal, "summary", None), max_len=160),
            "card_intent": _world_card_intent(anchor),
            "topic_title": anchor,
            "topic_summary": _truncate_text(getattr(signal, "summary", None), max_len=180),
            "topic_origin": "personality_snapshot",
            "display_goal": "open_public_topic",
            "source_kind": _truncate_text(getattr(signal, "source", None), max_len=48),
            "region": _truncate_text(getattr(signal, "region", None), max_len=48),
            "evidence_count": evidence_count,
            "fresh_until": _truncate_text(getattr(signal, "fresh_until", None), max_len=40),
            "content_trust_score": round(trust_score, 3),
            "untrusted_source_data": True,
        },
    )


def _safe_candidate_build(factory, item, *, engagement_states: Mapping[str, str], now_utc: datetime):
    """Build one candidate without letting one bad item fail the whole module."""
    try:
        return factory(item, engagement_states=engagement_states, now_utc=now_utc)
    except Exception:
        _LOG.debug("Skipping malformed reserve snapshot item %r during candidate build.", item, exc_info=True)
        return None


def _best_by_semantic_topic(
    candidates: Sequence[AmbientDisplayImpulseCandidate],
    *,
    excluded_topics: Sequence[tuple[str, frozenset[str]]],
) -> tuple[AmbientDisplayImpulseCandidate, ...]:
    """Keep only the strongest latent candidate per semantic / near-semantic topic."""
    ordered = sorted(candidates, key=_candidate_sort_key, reverse=True)
    selected: list[AmbientDisplayImpulseCandidate] = []
    selected_signatures: list[tuple[str, frozenset[str]]] = []

    for candidate in ordered:
        semantic_topic_key = candidate.semantic_key()
        if not semantic_topic_key:
            continue
        semantic_tokens = _semantic_tokens(semantic_topic_key)

        if any(
            _is_near_duplicate(
                candidate_key=semantic_topic_key,
                candidate_tokens=semantic_tokens,
                other_key=excluded_key,
                other_tokens=excluded_tokens,
            )
            for excluded_key, excluded_tokens in excluded_topics
        ):
            continue

        if any(
            _is_near_duplicate(
                candidate_key=semantic_topic_key,
                candidate_tokens=semantic_tokens,
                other_key=existing_key,
                other_tokens=existing_tokens,
            )
            for existing_key, existing_tokens in selected_signatures
        ):
            continue

        selected.append(candidate)
        selected_signatures.append((semantic_topic_key, semantic_tokens))

    return tuple(selected)


def load_display_reserve_snapshot_candidates(
    snapshot: PersonalitySnapshot | None,
    *,
    engagement_signals: Sequence[WorldInterestSignal],
    exclude_topic_keys: Sequence[str] = (),
    max_items: int,
    now_utc: datetime | None = None,
) -> tuple[AmbientDisplayImpulseCandidate, ...]:
    """Return bounded reserve candidates from latent snapshot topics."""
    if snapshot is None:
        return ()

    limited_max = _safe_int(max_items, default=0, minimum=0)
    if limited_max <= 0:
        return ()

    reference_now = now_utc if now_utc is not None else _utc_now()
    if reference_now.tzinfo is None:
        reference_now = reference_now.replace(tzinfo=UTC)
    else:
        reference_now = reference_now.astimezone(UTC)

    excluded_topics = tuple(
        (topic_key, _semantic_tokens(topic_key))
        for value in exclude_topic_keys
        if (topic_key := _topic_key(value))
    )
    engagement_states = _engagement_states_by_topic(engagement_signals)

    continuity_limit = max(6, limited_max)
    relationship_limit = max(4, (limited_max // 2) + 1)
    place_limit = min(_DEFAULT_PLACE_LIMIT, limited_max)
    world_limit = max(12, limited_max * 2)

    raw_candidates: list[AmbientDisplayImpulseCandidate] = []

    for thread in _top_items(
        snapshot.continuity_threads,
        limit=continuity_limit,
        key_builder=lambda item: _item_sort_key(
            _safe_float(getattr(item, "salience", None), default=0.0, minimum=0.0, maximum=1.0),
            _timestamp_sort_key(getattr(item, "updated_at", None)),
            getattr(item, "title", None),
        ),
    ):
        candidate = _safe_candidate_build(
            _continuity_candidate,
            thread,
            engagement_states=engagement_states,
            now_utc=reference_now,
        )
        if candidate is not None:
            raw_candidates.append(candidate)

    for relationship_signal in _top_items(
        snapshot.relationship_signals,
        limit=relationship_limit,
        key_builder=lambda item: _item_sort_key(
            _safe_float(getattr(item, "salience", None), default=0.0, minimum=0.0, maximum=1.0),
            _timestamp_sort_key(getattr(item, "updated_at", None)),
            getattr(item, "topic", None),
        ),
    ):
        candidate = _safe_candidate_build(
            _relationship_candidate,
            relationship_signal,
            engagement_states=engagement_states,
            now_utc=reference_now,
        )
        if candidate is not None:
            raw_candidates.append(candidate)

    for focus in _top_items(
        snapshot.place_focuses,
        limit=place_limit,
        key_builder=lambda item: _item_sort_key(
            _safe_float(getattr(item, "salience", None), default=0.0, minimum=0.0, maximum=1.0),
            _timestamp_sort_key(getattr(item, "updated_at", None)),
            getattr(item, "name", None),
        ),
    ):
        candidate = _safe_candidate_build(
            _place_candidate,
            focus,
            engagement_states=engagement_states,
            now_utc=reference_now,
        )
        if candidate is not None:
            raw_candidates.append(candidate)

    for world_signal in _top_items(
        snapshot.world_signals,
        limit=world_limit,
        key_builder=lambda item: _item_sort_key(
            bool(_truncate_text(getattr(item, "source", None), max_len=32).casefold() in _LOCAL_WORLD_SOURCES),
            _safe_float(getattr(item, "salience", None), default=0.0, minimum=0.0, maximum=1.0),
            _safe_int(getattr(item, "evidence_count", None), default=0, minimum=0, maximum=6),
            _timestamp_sort_key(getattr(item, "fresh_until", None)),
            getattr(item, "topic", None),
        ),
    ):
        candidate = _safe_candidate_build(
            _world_candidate,
            world_signal,
            engagement_states=engagement_states,
            now_utc=reference_now,
        )
        if candidate is not None:
            raw_candidates.append(candidate)

    deduped = _best_by_semantic_topic(raw_candidates, excluded_topics=excluded_topics)
    if not deduped:
        return ()

    bounded = select_diverse_candidates(
        deduped,
        max_items=min(limited_max, len(deduped)),
    )
    return tuple(
        AmbientDisplayImpulseCandidate(
            topic_key=_topic_key(candidate.topic_key),
            semantic_topic_key=_topic_key(candidate.semantic_topic_key),
            title=_truncate_text(candidate.title, max_len=72),
            source=_truncate_text(candidate.source, max_len=40),
            action=_truncate_text(candidate.action, max_len=24),
            attention_state=_truncate_text(candidate.attention_state, max_len=24),
            salience=_safe_float(candidate.salience, default=0.0, minimum=0.0, maximum=1.0),
            eyebrow=_truncate_text(candidate.eyebrow, max_len=48),
            headline=_truncate_text(candidate.headline, max_len=112),
            body=_truncate_text(candidate.body, max_len=112),
            symbol=_truncate_text(candidate.symbol, max_len=16),
            accent=_truncate_text(candidate.accent, max_len=16),
            reason=_truncate_text(candidate.reason, max_len=96),
            candidate_family=_truncate_text(candidate.candidate_family, max_len=48) or "general",
            generation_context=(
                _storage_safe_value(candidate.generation_context)
                if candidate.generation_context is not None
                else None
            ),
        )
        for candidate in bounded
    )


__all__ = ["load_display_reserve_snapshot_candidates"]
