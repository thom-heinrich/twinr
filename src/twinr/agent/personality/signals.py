# CHANGELOG: 2026-03-27
# BUG-1: Fixed extractor crashes on legacy/malformed attributes by routing all attribute access through safe mapping helpers.
# BUG-2: Fixed tool-history crashes when AgentToolCall.arguments or AgentToolResult.output are not mappings.
# BUG-3: Fixed event-id extraction crashes when memory_object.source or source.event_ids are missing/invalid.
# BUG-4: Fixed contradictory learning where explicit topic preference/feedback objects could also emit positive topic_affinity signals.
# BUG-5: Fixed style-feedback sign handling so negative preferences no longer produce positive impact scores.
# BUG-6: Fixed under-counted repeated evidence by honoring structured support_count in grouped topic/place aggregation.
# SEC-1: Hardened live-search ingestion against indirect prompt injection and memory poisoning by sanitizing/filtering untrusted external content before persistence.
# SEC-2: Suppressed topic/place/continuity signal extraction from sensitive contexts by default to reduce privacy leakage in senior-care deployments.
# SEC-3: Suppressed precise-address / coordinate-like place persistence and prefer coarse geographies (city/region/state/country) when available.
# IMP-1: Added conservative evidence thresholds so one-off mentions no longer become recurring topic/place signals by default.
# IMP-2: Added trust/provenance metadata, dynamic world-signal TTLs, and robust handling for richer live-search payloads and newer tool aliases.
#
# BREAKING: Sensitive-context memory objects no longer emit topic/place/continuity signals by default.
# BREAKING: Place extraction now prefers coarse geography labels and suppresses precise addresses/coordinates.
# BREAKING: topic_affinity/place signals now require repeated evidence (default support >= 2).

"""Extract structured personality-learning signals from Twinr runtime events.

This module sits between consolidated runtime artifacts and prompt-time
personalization. It converts structured long-term memory objects and structured
tool call/result payloads into conservative, policy-gated signals.

Compared with the original V1, this version adds three deployment-oriented
guardrails that matter in real assistants:

- schema-drift tolerance: malformed or partial payloads are skipped instead of
  crashing extraction for the whole batch
- memory governance: sensitive contexts and one-off mentions are not promoted
  into durable preference signals by default
- untrusted-input hardening: live-search outputs are treated as untrusted
  external content and sanitized before they become world/context signals
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from twinr.agent.base_agent.contracts import AgentToolCall, AgentToolResult
from twinr.agent.personality.models import ContinuityThread, InteractionSignal, PlaceSignal, WorldSignal
from twinr.memory.longterm.core.models import (
    LongTermConsolidationResultV1,
    LongTermConversationTurn,
    LongTermMemoryObjectV1,
)
from twinr.text_utils import slugify_identifier, truncate_text

INTERACTION_SIGNAL_TOPIC_AFFINITY = "topic_affinity"
INTERACTION_SIGNAL_TOPIC_ENGAGEMENT = "topic_engagement"
INTERACTION_SIGNAL_TOPIC_COOLING = "topic_cooling"
INTERACTION_SIGNAL_VERBOSITY_PREFERENCE = "verbosity_preference"
INTERACTION_SIGNAL_INITIATIVE_PREFERENCE = "initiative_preference"
INTERACTION_SIGNAL_HUMOR_FEEDBACK = "humor_feedback"
INTERACTION_SIGNAL_TOPIC_AVERSION = "topic_aversion"
PLACE_SIGNAL_SOURCE_CONVERSATION = "conversation_turn"
PLACE_SIGNAL_SOURCE_LIVE_SEARCH = "live_search"
WORLD_SIGNAL_SOURCE_LIVE_SEARCH = "live_search"
RELATIONSHIP_TOPIC_DELTA_PREFIX = "relationship.topic_affinity:"
RELATIONSHIP_TOPIC_AVERSION_DELTA_PREFIX = "relationship.topic_aversion:"
STYLE_VERBOSITY_DELTA_TARGET = "style.verbosity"
STYLE_INITIATIVE_DELTA_TARGET = "style.initiative"
SUPPORTED_TOOL_SIGNAL_NAMES = frozenset(
    {
        "search_live_info",
        "search_web",
        "web_search",
        "lookup_live_info",
    }
)

_TOPIC_ATTRIBUTE_KEYS = (
    "topic",
    "action",
    "plan_type",
    "event_domain",
    "fact_type",
    "observation_type",
    "pattern_type",
    "summary_type",
    "memory_domain",
)
_PLACE_ATTRIBUTE_PRIORITY_KEYS = (
    "city",
    "town",
    "district",
    "municipality",
    "county",
    "region",
    "state",
    "country",
    "place_name",
    "location_name",
    "place",
    "location",
)
_GENERIC_TOPIC_VALUES = frozenset(
    {
        "general",
        "stated_plan",
        "situational",
        "interaction",
        "presence",
        "thread",
        "social",
        "contact",
        "preference",
        "feedback",
        "planning",
        "topic",
    }
)
_SENSITIVE_DOMAIN_TOKENS = frozenset({"medical", "health", "distress", "crisis", "emergency", "care"})
_POSITIVE_SIGNAL_VALUES = frozenset({"positive", "more", "yes", "true", "liked", "helpful", "good"})
_NEGATIVE_SIGNAL_VALUES = frozenset({"negative", "less", "no", "false", "disliked", "too_much", "bad"})
_TOPIC_ENGAGEMENT_PREFERENCE_TYPES = frozenset(
    {"topic_engagement", "topic_follow_up", "topic_recurrence", "topic_deep_dive"}
)
_TOPIC_ENGAGEMENT_FEEDBACK_TARGETS = frozenset({"topic", "topic_engagement", "topic_follow_up"})
_TOPIC_ENGAGEMENT_POSITIVE_VALUES = _POSITIVE_SIGNAL_VALUES | frozenset(
    {"follow_up", "continue", "deeper", "deep_dive", "more_often", "revisit", "recurring"}
)
_TOPIC_COOLING_STRONG_PREFERENCE_TYPES = frozenset(
    {"topic_deflection", "topic_not_now", "topic_avoid"}
)
_TOPIC_COOLING_MILD_PREFERENCE_TYPES = frozenset(
    {"topic_less_often", "topic_pause", "topic_cooling"}
)
_TOPIC_COOLING_PREFERENCE_TYPES = (
    _TOPIC_COOLING_STRONG_PREFERENCE_TYPES | _TOPIC_COOLING_MILD_PREFERENCE_TYPES
)
_TOPIC_COOLING_FEEDBACK_TARGETS = _TOPIC_ENGAGEMENT_FEEDBACK_TARGETS
_TOPIC_COOLING_NEGATIVE_VALUES = _NEGATIVE_SIGNAL_VALUES | frozenset(
    {"skip", "move_on", "later", "not_now", "pause", "cooling", "less_often"}
)
_TOPIC_COOLING_OBSERVATION_TYPES = frozenset(
    {"topic_non_reengagement", "topic_deflection", "topic_cooling"}
)
_PREFERENCE_OBJECT_ATTRIBUTE_KEYS = (
    "preference_type",
    "engagement_type",
    "relationship_preference",
    "feedback_target",
    "style_dimension",
)
_TOOL_ARGUMENT_QUESTION_KEYS = ("question", "query", "prompt")
_TOOL_ARGUMENT_LOCATION_KEYS = ("location_hint", "location", "region", "area", "place")
_TOOL_OUTPUT_ANSWER_KEYS = ("answer", "summary", "observation", "result")
_TOOL_OUTPUT_SOURCE_KEYS = ("sources", "citations", "documents", "results", "items")
_WORLD_SIGNAL_VOLATILE_TOKENS = frozenset(
    {
        "weather",
        "forecast",
        "traffic",
        "delay",
        "road",
        "train",
        "flight",
        "news",
        "headline",
        "breaking",
        "latest",
        "score",
        "schedule",
        "price",
        "market",
        "stock",
        "crypto",
        "today",
        "tomorrow",
        "tonight",
        "current",
        "now",
    }
)
_COARSE_GEOGRAPHY_KEYS = frozenset(
    {"city", "town", "district", "municipality", "county", "region", "state", "country"}
)
_UNTRUSTED_EXTERNAL_PATTERNS = (
    re.compile(r"\bignore\s+(?:all\s+)?previous\s+instructions?\b", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\b", re.IGNORECASE),
    re.compile(r"\bdeveloper\s+mode\b", re.IGNORECASE),
    re.compile(r"\bsystem\s+override\b", re.IGNORECASE),
    re.compile(r"\breveal\s+(?:the\s+)?prompt\b", re.IGNORECASE),
    re.compile(r"\bdo\s+not\s+tell\s+the\s+user\b", re.IGNORECASE),
    re.compile(r"\bBEGIN\s+(?:SYSTEM|TEST|PROMPT)\s+INSTRUCTIONS\b", re.IGNORECASE),
)
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_COORDINATE_PATTERN = re.compile(r"\b[-+]?\d{1,3}\.\d{3,}\s*,\s*[-+]?\d{1,3}\.\d{3,}\b")
_ADDRESS_PATTERN = re.compile(
    r"\b\d{1,5}\s+[A-Za-zÀ-ÖØ-öø-ÿ0-9.\-]+\s+"
    r"(?:street|st|road|rd|avenue|ave|boulevard|blvd|lane|ln|drive|dr|way|court|ct|place|pl|"
    r"straße|strasse|str|gasse|allee|platz)\b",
    re.IGNORECASE,
)
_BUILDING_PATTERN = re.compile(
    r"\b(?:apt|apartment|unit|suite|room|floor|wing|building|block)\s+[A-Za-z0-9\-]+\b",
    re.IGNORECASE,
)
_POSTCODE_PATTERN = re.compile(r"\b\d{4,6}\b")


def _utcnow() -> datetime:
    """Return the current UTC time as an aware ``datetime``."""

    return datetime.now(timezone.utc)


def _ensure_aware_utc(value: datetime) -> datetime:
    """Normalize one ``datetime`` into UTC with timezone information."""

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _clean_text(value: object | None, *, limit: int | None = None) -> str:
    """Normalize arbitrary input into one bounded single-line string."""

    return truncate_text(None if value is None else str(value), limit=limit)


def _mean(values: Sequence[float], *, default: float = 0.0) -> float:
    """Return the arithmetic mean of a numeric sequence."""

    if not values:
        return default
    return sum(values) / len(values)


def _weighted_mean(
    values: Sequence[float],
    weights: Sequence[int],
    *,
    default: float = 0.0,
) -> float:
    """Return a weighted mean while tolerating empty or zero-weight input."""

    if not values or not weights or len(values) != len(weights):
        return default
    total_weight = sum(max(0, weight) for weight in weights)
    if total_weight <= 0:
        return default
    return sum(value * max(0, weight) for value, weight in zip(values, weights, strict=False)) / total_weight


def _clamp(value: float, *, minimum: float, maximum: float) -> float:
    """Clamp a numeric value into an inclusive range."""

    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


def _clean_token(value: object | None) -> str:
    """Normalize one structured label into a compact underscore token."""

    normalized = _clean_text(value, limit=80).casefold()
    for separator in (" ", "-", "/", ":"):
        normalized = normalized.replace(separator, "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def _safe_mapping(value: object | None) -> Mapping[str, object]:
    """Return one mapping-like payload or an empty mapping."""

    if isinstance(value, Mapping):
        return value
    return {}


def _safe_string_sequence(value: object | None, *, limit: int = 96) -> tuple[str, ...]:
    """Return one best-effort tuple of strings from arbitrary input."""

    if value is None:
        return ()
    if isinstance(value, str):
        normalized = _clean_text(value, limit=limit)
        return (normalized,) if normalized else ()
    if isinstance(value, Sequence):
        normalized_items: list[str] = []
        for item in value:
            normalized = _clean_text(item, limit=limit)
            if normalized:
                normalized_items.append(normalized)
        return tuple(normalized_items)
    return ()


def _coalesce_event_ids(event_id_groups: Iterable[object]) -> tuple[str, ...]:
    """Return de-duplicated event ids while preserving their first-seen order."""

    ordered: list[str] = []
    seen: set[str] = set()
    for group in event_id_groups:
        for event_id in _safe_string_sequence(group):
            if event_id in seen:
                continue
            seen.add(event_id)
            ordered.append(event_id)
    return tuple(ordered)


def _coerce_positive_int(value: object | None, *, default: int = 1) -> int:
    """Coerce one structured support-count-like value into a positive integer."""

    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def _safe_datetime(value: object | None) -> datetime | None:
    """Parse one datetime-like value into aware UTC time."""

    if value is None:
        return None
    if isinstance(value, datetime):
        return _ensure_aware_utc(value)
    normalized = _clean_text(value, limit=96)
    if not normalized:
        return None
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    return _ensure_aware_utc(parsed)


def _attributes(memory_object: LongTermMemoryObjectV1) -> Mapping[str, object]:
    """Return a safe attributes mapping for one memory object."""

    attributes = getattr(memory_object, "attributes", None)
    if isinstance(attributes, Mapping):
        return attributes
    return {}


def _memory_source_event_ids(memory_object: LongTermMemoryObjectV1) -> tuple[str, ...]:
    """Return safe source event ids for one memory object."""

    source = getattr(memory_object, "source", None)
    event_ids = getattr(source, "event_ids", ()) if source is not None else ()
    return _coalesce_event_ids((event_ids,))


def _first_structured_text(
    memory_object: LongTermMemoryObjectV1,
    *,
    keys: Sequence[str],
    fallback: object | None = None,
    limit: int = 96,
) -> str | None:
    """Return the first non-blank structured text value from one memory object."""

    attributes = _attributes(memory_object)
    for key in keys:
        normalized = _clean_text(attributes.get(key), limit=limit)
        if normalized:
            return normalized
    normalized_fallback = _clean_text(fallback, limit=limit)
    return normalized_fallback or None


def _first_structured_token(
    memory_object: LongTermMemoryObjectV1,
    *,
    keys: Sequence[str],
    fallback: object | None = None,
) -> str | None:
    """Return the first non-blank normalized token value from one memory object."""

    candidate = _first_structured_text(
        memory_object,
        keys=keys,
        fallback=fallback,
    )
    token = _clean_token(candidate)
    return token or None


def _first_mapping_text(
    payload: Mapping[str, object],
    *,
    keys: Sequence[str],
    limit: int = 220,
) -> str | None:
    """Return the first non-blank text value from one mapping."""

    for key in keys:
        normalized = _clean_text(payload.get(key), limit=limit)
        if normalized:
            return normalized
    return None


def _memory_support_count(memory_object: LongTermMemoryObjectV1) -> int:
    """Return the structured support count carried by one memory object."""

    return _coerce_positive_int(_attributes(memory_object).get("support_count"), default=1)


def _total_support_count(memory_objects: Sequence[LongTermMemoryObjectV1]) -> int:
    """Return total support count across grouped memory objects."""

    return sum(_memory_support_count(item) for item in memory_objects)


def _is_sensitive_context(memory_object: LongTermMemoryObjectV1) -> bool:
    """Return whether one memory object came from a sensitive learning context."""

    if _clean_token(getattr(memory_object, "sensitivity", None)) in {"private", "sensitive", "critical"}:
        return True
    attributes = _attributes(memory_object)
    explicit_flag = attributes.get("sensitive_context")
    if isinstance(explicit_flag, bool):
        return explicit_flag
    for key in (
        "memory_domain",
        "fact_type",
        "event_domain",
        "observation_type",
        "plan_type",
        "summary_type",
        "context_domain",
    ):
        if _clean_token(attributes.get(key)) in _SENSITIVE_DOMAIN_TOKENS:
            return True
    return False


def _is_preference_or_feedback_object(memory_object: LongTermMemoryObjectV1) -> bool:
    """Return whether one memory object encodes preference/feedback rather than neutral evidence."""

    attributes = _attributes(memory_object)
    if _clean_token(attributes.get("fact_type")) in {"preference", "feedback"}:
        return True
    if _clean_token(attributes.get("memory_domain")) == "preference":
        return True
    return any(_clean_token(attributes.get(key)) for key in _PREFERENCE_OBJECT_ATTRIBUTE_KEYS)


def _best_topic_label(memory_object: LongTermMemoryObjectV1) -> str | None:
    """Choose the strongest conversation topic label from one memory object."""

    attributes = _attributes(memory_object)
    for key in _TOPIC_ATTRIBUTE_KEYS:
        raw_value = attributes.get(key)
        normalized = _clean_text(raw_value, limit=80)
        if not normalized:
            continue
        if _clean_token(normalized) in _GENERIC_TOPIC_VALUES:
            continue
        return normalized
    return None


def _looks_like_precise_place(value: str) -> bool:
    """Return whether a place label looks like a precise address or coordinate."""

    if not value:
        return False
    if _COORDINATE_PATTERN.search(value):
        return True
    if _ADDRESS_PATTERN.search(value):
        return True
    if _BUILDING_PATTERN.search(value):
        return True
    lowered = value.casefold()
    if _POSTCODE_PATTERN.search(value) and any(
        marker in lowered
        for marker in (" street", " st ", " road", " rd ", " avenue", " ave", " boulevard", " blvd", " lane", " ln ", " drive", " dr ", " strasse", " straße", " gasse", " allee", " platz")
    ):
        return True
    return False


def _best_place_label(memory_object: LongTermMemoryObjectV1) -> tuple[str | None, str | None, str | None]:
    """Choose the safest strong place label, geography, and ref from one memory object."""

    attributes = _attributes(memory_object)
    place_ref = _clean_text(attributes.get("place_ref"), limit=96) or None
    for key in _PLACE_ATTRIBUTE_PRIORITY_KEYS:
        normalized = _clean_text(attributes.get(key), limit=80)
        if not normalized:
            continue
        geography = "place" if key not in _COARSE_GEOGRAPHY_KEYS else key
        if geography == "place" and _looks_like_precise_place(normalized):
            continue
        return normalized, geography, place_ref
    return None, None, place_ref


def _topic_delta_value(*, confidence: float, evidence_count: int) -> float:
    """Convert repeated topic evidence into one bounded salience delta."""

    weighted = confidence * min(max(evidence_count, 1), 3)
    return _clamp(weighted * 0.12, minimum=0.08, maximum=0.3)


def _strongest_object(memory_objects: Sequence[LongTermMemoryObjectV1]) -> LongTermMemoryObjectV1:
    """Return the strongest object from a group based on support count then confidence."""

    return max(
        memory_objects,
        key=lambda item: (_memory_support_count(item), _clamp(getattr(item, "confidence", 0.0), minimum=0.0, maximum=1.0)),
    )


def _dedupe_interaction_signals(signals: Sequence[InteractionSignal]) -> tuple[InteractionSignal, ...]:
    """Deduplicate interaction signals by stable signal id."""

    deduped: list[InteractionSignal] = []
    seen: set[str] = set()
    for signal in signals:
        signal_id = _clean_text(getattr(signal, "signal_id", None), limit=160)
        if signal_id and signal_id in seen:
            continue
        if signal_id:
            seen.add(signal_id)
        deduped.append(signal)
    return tuple(deduped)


def _dedupe_place_signals(signals: Sequence[PlaceSignal]) -> tuple[PlaceSignal, ...]:
    """Deduplicate place signals by stable signal id."""

    deduped: list[PlaceSignal] = []
    seen: set[str] = set()
    for signal in signals:
        signal_id = _clean_text(getattr(signal, "signal_id", None), limit=160)
        if signal_id and signal_id in seen:
            continue
        if signal_id:
            seen.add(signal_id)
        deduped.append(signal)
    return tuple(deduped)


def _dedupe_world_signals(signals: Sequence[WorldSignal]) -> tuple[WorldSignal, ...]:
    """Deduplicate world signals by their semantic identity."""

    deduped: list[WorldSignal] = []
    seen: set[tuple[str, str, str, str]] = set()
    for signal in signals:
        key = (
            _clean_text(getattr(signal, "topic", None), limit=160).casefold(),
            _clean_text(getattr(signal, "summary", None), limit=220).casefold(),
            _clean_text(getattr(signal, "region", None), limit=120).casefold(),
            _clean_text(getattr(signal, "source", None), limit=64).casefold(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(signal)
    return tuple(deduped)


def _dedupe_continuity_threads(threads: Sequence[ContinuityThread]) -> tuple[ContinuityThread, ...]:
    """Deduplicate continuity threads by title and expiry."""

    deduped: list[ContinuityThread] = []
    seen: set[tuple[str, str]] = set()
    for thread in threads:
        key = (
            _clean_text(getattr(thread, "title", None), limit=160).casefold(),
            _clean_text(getattr(thread, "expires_at", None), limit=96),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(thread)
    return tuple(deduped)


def _filter_untrusted_external_text(
    value: object | None,
    *,
    limit: int = 220,
) -> tuple[str, bool]:
    """Sanitize untrusted external text and flag whether risky patterns were removed."""

    cleaned = _clean_text(value, limit=max(limit * 2, limit))
    if not cleaned:
        return "", False
    cleaned = cleaned.replace("```", " ")
    cleaned = _HTML_TAG_PATTERN.sub(" ", cleaned)
    risky = False
    for pattern in _UNTRUSTED_EXTERNAL_PATTERNS:
        cleaned, replacements = pattern.subn("[filtered]", cleaned)
        risky = risky or replacements > 0
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = truncate_text(cleaned, limit=limit)
    remainder = cleaned.replace("[filtered]", "").strip()
    if risky and len(remainder) < 24:
        return "", True
    return cleaned, risky


def _dynamic_world_signal_ttl_hours(question: str, *, default_hours: int) -> int:
    """Choose a TTL based on volatility cues in the live-search question."""

    normalized = _clean_token(question)
    if any(token in normalized for token in _WORLD_SIGNAL_VOLATILE_TOKENS):
        return max(4, min(default_hours, 8))
    return max(4, int(default_hours))


@dataclass(frozen=True, slots=True)
class PersonalitySignalBatch:
    """Collect the signals extracted from one runtime event family."""

    interaction_signals: tuple[InteractionSignal, ...] = ()
    place_signals: tuple[PlaceSignal, ...] = ()
    world_signals: tuple[WorldSignal, ...] = ()
    continuity_threads: tuple[ContinuityThread, ...] = ()

    def has_any(self) -> bool:
        """Return whether the batch contains any learning evidence."""

        return bool(
            self.interaction_signals
            or self.place_signals
            or self.world_signals
            or self.continuity_threads
        )

    def normalized(self) -> "PersonalitySignalBatch":
        """Return the batch with duplicate signals removed."""

        return PersonalitySignalBatch(
            interaction_signals=_dedupe_interaction_signals(self.interaction_signals),
            place_signals=_dedupe_place_signals(self.place_signals),
            world_signals=_dedupe_world_signals(self.world_signals),
            continuity_threads=_dedupe_continuity_threads(self.continuity_threads),
        )

    def merged(self, other: "PersonalitySignalBatch") -> "PersonalitySignalBatch":
        """Return one batch containing both inputs in stable append order."""

        return PersonalitySignalBatch(
            interaction_signals=self.interaction_signals + other.interaction_signals,
            place_signals=self.place_signals + other.place_signals,
            world_signals=self.world_signals + other.world_signals,
            continuity_threads=self.continuity_threads + other.continuity_threads,
        ).normalized()


@dataclass(slots=True)
class PersonalitySignalExtractor:
    """Extract personality-learning signals from conversation and tool history.

    Attributes:
        now_provider: Clock used for freshness windows and deterministic tests.
        world_signal_ttl_hours: Default freshness horizon for live-search world
            signals before volatility-specific shortening is applied.
        continuity_thread_ttl_days: Default review horizon for continuity
            threads derived from repeated turn summaries.
        min_topic_affinity_support: Minimum repeated evidence required before a
            topic becomes a recurring-interest signal.
        min_place_signal_support: Minimum repeated evidence required before a
            conversational place becomes prompt-time situational context.
        suppress_sensitive_context_signals: Whether sensitive memory objects are
            blocked from emitting topic/place/continuity signals.
        suppress_precise_place_signals: Whether address-like or coordinate-like
            places are blocked from persistence.
        sanitize_external_content: Whether live-search answers are filtered for
            instruction-like external content before becoming world signals.
    """

    now_provider: Callable[[], datetime] = _utcnow
    world_signal_ttl_hours: int = 24
    continuity_thread_ttl_days: int = 21
    min_topic_affinity_support: int = 2  # BREAKING: single mentions no longer become recurring topic signals by default.
    min_place_signal_support: int = 2  # BREAKING: single mentions no longer become recurring conversational place signals by default.
    suppress_sensitive_context_signals: bool = True  # BREAKING: sensitive contexts no longer emit durable topic/place/continuity signals by default.
    suppress_precise_place_signals: bool = True  # BREAKING: precise places are suppressed instead of being persisted as prompt-time context.
    sanitize_external_content: bool = True

    def extract_from_consolidation(
        self,
        *,
        turn: LongTermConversationTurn,
        consolidation: LongTermConsolidationResultV1,
    ) -> PersonalitySignalBatch:
        """Extract interaction and place signals from one consolidated turn."""

        del turn  # carried for API compatibility and future use
        relevant_objects = tuple((*consolidation.durable_objects, *consolidation.deferred_objects))
        style_feedback_objects = tuple((*relevant_objects, *consolidation.episodic_objects))
        batch = PersonalitySignalBatch(
            interaction_signals=(
                self._extract_topic_affinity_signals(
                    turn_id=consolidation.turn_id,
                    memory_objects=relevant_objects,
                )
                + self._extract_topic_engagement_signals(
                    turn_id=consolidation.turn_id,
                    memory_objects=relevant_objects,
                )
                + self._extract_topic_cooling_signals(
                    turn_id=consolidation.turn_id,
                    memory_objects=relevant_objects,
                )
                + self._extract_style_and_feedback_signals(
                    turn_id=consolidation.turn_id,
                    memory_objects=style_feedback_objects,
                )
                + self._extract_topic_aversion_signals(
                    turn_id=consolidation.turn_id,
                    memory_objects=relevant_objects,
                )
            ),
            place_signals=self._extract_place_signals_from_objects(
                turn_id=consolidation.turn_id,
                memory_objects=relevant_objects,
                occurred_at=consolidation.occurred_at,
            ),
            continuity_threads=self._extract_continuity_threads_from_objects(
                memory_objects=relevant_objects,
                occurred_at=consolidation.occurred_at,
            ),
        )
        return batch.normalized()

    def extract_from_tool_history(
        self,
        *,
        tool_calls: Sequence[AgentToolCall],
        tool_results: Sequence[AgentToolResult],
    ) -> PersonalitySignalBatch:
        """Extract world and place signals from structured tool call history."""

        result_by_call_id: dict[str, AgentToolResult] = {}
        for result in tool_results:
            call_id = _clean_text(getattr(result, "call_id", None), limit=96)
            if call_id:
                result_by_call_id[call_id] = result

        now = _ensure_aware_utc(self.now_provider())
        place_signals: list[PlaceSignal] = []
        world_signals: list[WorldSignal] = []

        for tool_call in tool_calls:
            tool_name = _clean_text(getattr(tool_call, "name", None), limit=64)
            if tool_name not in SUPPORTED_TOOL_SIGNAL_NAMES:
                continue
            call_id = _clean_text(getattr(tool_call, "call_id", None), limit=96)
            if not call_id:
                continue
            tool_result = result_by_call_id.get(call_id)
            if tool_result is None:
                continue
            extracted = self._extract_live_search_signals(
                tool_call=tool_call,
                tool_result=tool_result,
                now=now,
            )
            place_signals.extend(extracted.place_signals)
            world_signals.extend(extracted.world_signals)

        return PersonalitySignalBatch(
            interaction_signals=(),
            place_signals=tuple(place_signals),
            world_signals=tuple(world_signals),
            continuity_threads=(),
        ).normalized()

    def _extract_topic_cooling_signals(
        self,
        *,
        turn_id: str,
        memory_objects: Sequence[LongTermMemoryObjectV1],
    ) -> tuple[InteractionSignal, ...]:
        """Build exposure-aware cooling signals from structured topic reactions."""

        signals: list[InteractionSignal] = []
        turn_slug = slugify_identifier(turn_id, fallback="turn")
        for memory_object in memory_objects:
            if self.suppress_sensitive_context_signals and _is_sensitive_context(memory_object):
                continue
            preference_type = _first_structured_token(
                memory_object,
                keys=("preference_type", "engagement_type", "relationship_preference"),
            )
            feedback_target = _first_structured_token(
                memory_object,
                keys=("feedback_target",),
            )
            observation_type = _first_structured_token(
                memory_object,
                keys=("observation_type", "pattern_type", "engagement_observation"),
            )
            engagement_value = _first_structured_token(
                memory_object,
                keys=("preference_value", "feedback_polarity", "reaction", "engagement_level"),
                fallback=getattr(memory_object, "value_key", None),
            )
            explicit_feedback = (
                preference_type in _TOPIC_COOLING_PREFERENCE_TYPES
                or (
                    feedback_target in _TOPIC_COOLING_FEEDBACK_TARGETS
                    and engagement_value in _TOPIC_COOLING_NEGATIVE_VALUES
                )
            )
            implicit_pattern = observation_type in _TOPIC_COOLING_OBSERVATION_TYPES
            if not explicit_feedback and not implicit_pattern:
                continue
            topic = _first_structured_text(
                memory_object,
                keys=("topic", "target_topic", "thread_title"),
                fallback=getattr(memory_object, "value_key", None),
            )
            if topic is None:
                continue

            attributes = _attributes(memory_object)
            support_count = _memory_support_count(memory_object)
            exposure_count = _coerce_positive_int(
                attributes.get("exposure_count"),
                default=max(2 if explicit_feedback else 1, support_count),
            )
            non_reengagement_count = 0
            deflection_count = 0
            negative_kind = "topic_cooling"

            if implicit_pattern:
                non_reengagement_count = max(
                    1,
                    _coerce_positive_int(attributes.get("non_reengagement_count"), default=1),
                )
                deflection_count = (
                    _coerce_positive_int(attributes.get("deflection_count"), default=1)
                    if observation_type == "topic_deflection"
                    else 0
                )
                exposure_aware = attributes.get("exposure_aware")
                if isinstance(exposure_aware, bool):
                    is_exposure_aware = exposure_aware
                else:
                    is_exposure_aware = exposure_count >= 2
                if not is_exposure_aware or exposure_count < 2:
                    continue
                negative_kind = observation_type or "topic_non_reengagement"
            else:
                if preference_type in _TOPIC_COOLING_STRONG_PREFERENCE_TYPES:
                    deflection_count = 1
                    non_reengagement_count = 1
                    negative_kind = preference_type or "topic_deflection"
                else:
                    non_reengagement_count = 1
                    negative_kind = preference_type or feedback_target or "topic_less_often"

            confidence = _clamp(getattr(memory_object, "confidence", 0.0), minimum=0.0, maximum=1.0)
            topic_slug = slugify_identifier(topic, fallback="topic")
            signals.append(
                InteractionSignal(
                    signal_id=f"signal:interaction:{turn_slug}:{topic_slug}:cooling",
                    signal_kind=INTERACTION_SIGNAL_TOPIC_COOLING,
                    target=topic,
                    summary=truncate_text(getattr(memory_object, "summary", ""), limit=180),
                    confidence=confidence,
                    impact=-(confidence * (0.55 if deflection_count else 0.4)),
                    evidence_count=max(support_count, non_reengagement_count, deflection_count, 1),
                    source_event_ids=_memory_source_event_ids(memory_object),
                    explicit_user_requested=explicit_feedback,
                    metadata={
                        "signal_source": PLACE_SIGNAL_SOURCE_CONVERSATION,
                        "memory_id": getattr(memory_object, "memory_id", ""),
                        "engagement_kind": negative_kind,
                        "engagement_direction": "negative",
                        "exposure_count": exposure_count,
                        "non_reengagement_count": non_reengagement_count,
                        "deflection_count": deflection_count,
                        "trust_tier": "internal_structured",
                    },
                )
            )
        return tuple(signals)

    def _extract_topic_engagement_signals(
        self,
        *,
        turn_id: str,
        memory_objects: Sequence[LongTermMemoryObjectV1],
    ) -> tuple[InteractionSignal, ...]:
        """Build stronger topic-engagement signals from structured reactions."""

        signals: list[InteractionSignal] = []
        turn_slug = slugify_identifier(turn_id, fallback="turn")
        for memory_object in memory_objects:
            if self.suppress_sensitive_context_signals and _is_sensitive_context(memory_object):
                continue
            preference_type = _first_structured_token(
                memory_object,
                keys=("preference_type", "engagement_type", "relationship_preference"),
            )
            feedback_target = _first_structured_token(
                memory_object,
                keys=("feedback_target",),
            )
            if (
                preference_type not in _TOPIC_ENGAGEMENT_PREFERENCE_TYPES
                and feedback_target not in _TOPIC_ENGAGEMENT_FEEDBACK_TARGETS
            ):
                continue
            engagement_value = _first_structured_token(
                memory_object,
                keys=("preference_value", "feedback_polarity", "reaction", "engagement_level"),
                fallback=getattr(memory_object, "value_key", None),
            )
            explicit_request = preference_type in _TOPIC_ENGAGEMENT_PREFERENCE_TYPES
            if not explicit_request and engagement_value not in _TOPIC_ENGAGEMENT_POSITIVE_VALUES:
                continue
            topic = _first_structured_text(
                memory_object,
                keys=("topic", "target_topic", "thread_title"),
                fallback=getattr(memory_object, "value_key", None),
            )
            if topic is None:
                continue
            confidence = _clamp(getattr(memory_object, "confidence", 0.0), minimum=0.0, maximum=1.0)
            support_count = _memory_support_count(memory_object)
            topic_slug = slugify_identifier(topic, fallback="topic")
            engagement_kind = preference_type or feedback_target or "topic_engagement"
            signals.append(
                InteractionSignal(
                    signal_id=f"signal:interaction:{turn_slug}:{topic_slug}:engagement",
                    signal_kind=INTERACTION_SIGNAL_TOPIC_ENGAGEMENT,
                    target=topic,
                    summary=truncate_text(getattr(memory_object, "summary", ""), limit=180),
                    confidence=confidence,
                    impact=confidence * (0.75 if explicit_request else 0.6),
                    evidence_count=max(2, support_count),
                    source_event_ids=_memory_source_event_ids(memory_object),
                    explicit_user_requested=explicit_request,
                    metadata={
                        "signal_source": PLACE_SIGNAL_SOURCE_CONVERSATION,
                        "memory_id": getattr(memory_object, "memory_id", ""),
                        "engagement_kind": engagement_kind,
                        "trust_tier": "internal_structured",
                    },
                )
            )
        return tuple(signals)

    def _extract_style_and_feedback_signals(
        self,
        *,
        turn_id: str,
        memory_objects: Sequence[LongTermMemoryObjectV1],
    ) -> tuple[InteractionSignal, ...]:
        """Build style and humor-feedback signals from structured preference objects."""

        signals: list[InteractionSignal] = []
        turn_slug = slugify_identifier(turn_id, fallback="turn")
        for memory_object in memory_objects:
            preference_type = _first_structured_token(
                memory_object,
                keys=("preference_type", "style_dimension", "feedback_target"),
            )
            preference_value = _first_structured_token(
                memory_object,
                keys=("preference_value", "preferred_level", "feedback_polarity", "reaction"),
                fallback=getattr(memory_object, "value_key", None),
            )
            if preference_type == "verbosity":
                delta_value = None
                delta_summary = None
                if preference_value in {"concise", "brief", "short", "compact"}:
                    delta_value = -0.18
                    delta_summary = "Keep answers a bit more concise by default."
                elif preference_value in {"detailed", "expansive", "full", "more_detail"}:
                    delta_value = 0.18
                    delta_summary = "Offer a bit more detail by default."
                if delta_value is None or delta_summary is None:
                    continue
                memory_slug = slugify_identifier(getattr(memory_object, "memory_id", None), fallback="verbosity")
                confidence = _clamp(getattr(memory_object, "confidence", 0.0), minimum=0.0, maximum=1.0)
                support_count = _memory_support_count(memory_object)
                signals.append(
                    InteractionSignal(
                        signal_id=f"signal:interaction:{turn_slug}:{memory_slug}",
                        signal_kind=INTERACTION_SIGNAL_VERBOSITY_PREFERENCE,
                        target="verbosity",
                        summary=truncate_text(getattr(memory_object, "summary", ""), limit=180),
                        confidence=confidence,
                        impact=confidence * (0.5 if delta_value > 0 else -0.5),
                        evidence_count=support_count,
                        source_event_ids=_memory_source_event_ids(memory_object),
                        explicit_user_requested=True,
                        delta_target=STYLE_VERBOSITY_DELTA_TARGET,
                        delta_value=delta_value,
                        delta_summary=delta_summary,
                        metadata={
                            "signal_source": PLACE_SIGNAL_SOURCE_CONVERSATION,
                            "memory_id": getattr(memory_object, "memory_id", ""),
                            "sensitive_context": _is_sensitive_context(memory_object),
                            "trust_tier": "internal_structured",
                        },
                    )
                )
                continue

            if preference_type == "initiative":
                delta_value = None
                delta_summary = None
                if preference_value in {"less_proactive", "less_initiative", "reactive", "minimal"}:
                    delta_value = -0.16
                    delta_summary = "Stay a bit more reactive unless the user invites a next step."
                elif preference_value in {"more_proactive", "more_initiative", "proactive", "gently_proactive"}:
                    delta_value = 0.16
                    delta_summary = "Take a slightly more proactive stance in relaxed turns."
                if delta_value is None or delta_summary is None:
                    continue
                memory_slug = slugify_identifier(getattr(memory_object, "memory_id", None), fallback="initiative")
                confidence = _clamp(getattr(memory_object, "confidence", 0.0), minimum=0.0, maximum=1.0)
                support_count = _memory_support_count(memory_object)
                signals.append(
                    InteractionSignal(
                        signal_id=f"signal:interaction:{turn_slug}:{memory_slug}",
                        signal_kind=INTERACTION_SIGNAL_INITIATIVE_PREFERENCE,
                        target="initiative",
                        summary=truncate_text(getattr(memory_object, "summary", ""), limit=180),
                        confidence=confidence,
                        impact=confidence * (0.4 if delta_value > 0 else -0.4),
                        evidence_count=support_count,
                        source_event_ids=_memory_source_event_ids(memory_object),
                        explicit_user_requested=True,
                        delta_target=STYLE_INITIATIVE_DELTA_TARGET,
                        delta_value=delta_value,
                        delta_summary=delta_summary,
                        metadata={
                            "signal_source": PLACE_SIGNAL_SOURCE_CONVERSATION,
                            "memory_id": getattr(memory_object, "memory_id", ""),
                            "sensitive_context": _is_sensitive_context(memory_object),
                            "trust_tier": "internal_structured",
                        },
                    )
                )
                continue

            if preference_type == "humor":
                delta_value = None
                delta_summary = None
                if preference_value in _POSITIVE_SIGNAL_VALUES:
                    delta_value = 0.12
                    delta_summary = "Increase humor slightly in relaxed turns."
                elif preference_value in _NEGATIVE_SIGNAL_VALUES:
                    delta_value = -0.12
                    delta_summary = "Dial humor back unless the user clearly invites it."
                if delta_value is None or delta_summary is None:
                    continue
                memory_slug = slugify_identifier(getattr(memory_object, "memory_id", None), fallback="humor")
                confidence = _clamp(getattr(memory_object, "confidence", 0.0), minimum=0.0, maximum=1.0)
                support_count = _memory_support_count(memory_object)
                signals.append(
                    InteractionSignal(
                        signal_id=f"signal:interaction:{turn_slug}:{memory_slug}",
                        signal_kind=INTERACTION_SIGNAL_HUMOR_FEEDBACK,
                        target="humor",
                        summary=truncate_text(getattr(memory_object, "summary", ""), limit=180),
                        confidence=confidence,
                        impact=confidence * (0.3 if delta_value > 0 else -0.3),
                        evidence_count=support_count,
                        source_event_ids=_memory_source_event_ids(memory_object),
                        explicit_user_requested=True,
                        delta_target="humor.intensity",
                        delta_value=delta_value,
                        delta_summary=delta_summary,
                        metadata={
                            "signal_source": PLACE_SIGNAL_SOURCE_CONVERSATION,
                            "memory_id": getattr(memory_object, "memory_id", ""),
                            "sensitive_context": _is_sensitive_context(memory_object),
                            "trust_tier": "internal_structured",
                        },
                    )
                )
        return tuple(signals)

    def _extract_topic_aversion_signals(
        self,
        *,
        turn_id: str,
        memory_objects: Sequence[LongTermMemoryObjectV1],
    ) -> tuple[InteractionSignal, ...]:
        """Build topic-aversion interaction signals from structured preference objects."""

        signals: list[InteractionSignal] = []
        turn_slug = slugify_identifier(turn_id, fallback="turn")
        for memory_object in memory_objects:
            if self.suppress_sensitive_context_signals and _is_sensitive_context(memory_object):
                continue
            preference_type = _first_structured_token(
                memory_object,
                keys=("preference_type", "relationship_preference"),
            )
            if preference_type != "topic_aversion":
                continue
            topic = _first_structured_text(
                memory_object,
                keys=("topic", "target_topic"),
                fallback=getattr(memory_object, "value_key", None),
            )
            if topic is None:
                continue
            confidence = _clamp(getattr(memory_object, "confidence", 0.0), minimum=0.0, maximum=1.0)
            support_count = _memory_support_count(memory_object)
            topic_slug = slugify_identifier(topic, fallback="topic")
            signals.append(
                InteractionSignal(
                    signal_id=f"signal:interaction:{turn_slug}:{topic_slug}:aversion",
                    signal_kind=INTERACTION_SIGNAL_TOPIC_AVERSION,
                    target=topic,
                    summary=truncate_text(getattr(memory_object, "summary", ""), limit=180),
                    confidence=confidence,
                    impact=-confidence * 0.5,
                    evidence_count=support_count,
                    source_event_ids=_memory_source_event_ids(memory_object),
                    delta_target=f"{RELATIONSHIP_TOPIC_AVERSION_DELTA_PREFIX}{topic}",
                    delta_value=_topic_delta_value(
                        confidence=confidence,
                        evidence_count=support_count,
                    ),
                    delta_summary=f"Avoid dwelling on {topic} unless the user explicitly asks.",
                    explicit_user_requested=True,
                    metadata={
                        "signal_source": PLACE_SIGNAL_SOURCE_CONVERSATION,
                        "memory_id": getattr(memory_object, "memory_id", ""),
                        "engagement_direction": "negative",
                        "engagement_kind": "topic_aversion",
                        "exposure_count": max(2, support_count),
                        "non_reengagement_count": 1,
                        "deflection_count": 2,
                        "trust_tier": "internal_structured",
                    },
                )
            )
        return tuple(signals)

    def _extract_continuity_threads_from_objects(
        self,
        *,
        memory_objects: Sequence[LongTermMemoryObjectV1],
        occurred_at: datetime,
    ) -> tuple[ContinuityThread, ...]:
        """Build prompt-facing continuity threads from repeated thread summaries."""

        threads: list[ContinuityThread] = []
        updated_at = _ensure_aware_utc(occurred_at)
        default_ttl_days = max(7, int(self.continuity_thread_ttl_days))
        max_ttl_days = max(default_ttl_days, default_ttl_days * 6)
        for memory_object in memory_objects:
            if self.suppress_sensitive_context_signals and _is_sensitive_context(memory_object):
                continue
            summary_type = _first_structured_token(memory_object, keys=("summary_type",))
            memory_domain = _first_structured_token(memory_object, keys=("memory_domain",))
            if summary_type != "thread" and memory_domain != "thread":
                continue
            support_count = _memory_support_count(memory_object)
            if support_count < 2:
                continue
            title = _first_structured_text(
                memory_object,
                keys=("thread_title", "person_name", "topic"),
                fallback=getattr(memory_object, "value_key", None),
            )
            if title is None:
                continue
            expires_candidate = (
                _safe_datetime(getattr(memory_object, "valid_to", None))
                or _safe_datetime(_attributes(memory_object).get("valid_to"))
            )
            if expires_candidate is None or expires_candidate <= updated_at:
                expires_at = updated_at + timedelta(days=default_ttl_days)
            else:
                max_allowed = updated_at + timedelta(days=max_ttl_days)
                expires_at = min(expires_candidate, max_allowed)
            threads.append(
                ContinuityThread(
                    title=title,
                    summary=truncate_text(getattr(memory_object, "summary", ""), limit=180),
                    salience=_clamp(
                        max(
                            _clamp(getattr(memory_object, "confidence", 0.0), minimum=0.0, maximum=1.0),
                            min(0.95, 0.35 + (support_count * 0.1)),
                        ),
                        minimum=0.0,
                        maximum=1.0,
                    ),
                    updated_at=updated_at.isoformat(),
                    expires_at=expires_at.isoformat(),
                )
            )
        return tuple(threads)

    def _extract_topic_affinity_signals(
        self,
        *,
        turn_id: str,
        memory_objects: Sequence[LongTermMemoryObjectV1],
    ) -> tuple[InteractionSignal, ...]:
        """Build topic-affinity interaction signals from structured memory objects."""

        grouped: dict[str, list[LongTermMemoryObjectV1]] = defaultdict(list)
        topic_labels: dict[str, str] = {}
        for memory_object in memory_objects:
            if self.suppress_sensitive_context_signals and _is_sensitive_context(memory_object):
                continue
            if _is_preference_or_feedback_object(memory_object):
                continue
            memory_attributes = _attributes(memory_object)
            if _clean_token(memory_attributes.get("memory_domain")) in {"preference", "thread"}:
                continue
            if _clean_token(memory_attributes.get("fact_type")) in {"preference", "feedback"}:
                continue
            if _clean_token(memory_attributes.get("summary_type")) == "thread":
                continue
            if _clean_token(memory_attributes.get("pattern_type")) in _TOPIC_COOLING_OBSERVATION_TYPES:
                continue
            if _clean_token(memory_attributes.get("observation_type")) in _TOPIC_COOLING_OBSERVATION_TYPES:
                continue
            topic = _best_topic_label(memory_object)
            if topic is None:
                continue
            topic_key = topic.casefold()
            topic_labels.setdefault(topic_key, topic)
            grouped[topic_key].append(memory_object)

        signals: list[InteractionSignal] = []
        turn_slug = slugify_identifier(turn_id, fallback="turn")
        min_support = max(1, int(self.min_topic_affinity_support))
        for topic_key, grouped_objects in grouped.items():
            evidence_count = _total_support_count(grouped_objects)
            if evidence_count < min_support:
                continue
            display_topic = topic_labels[topic_key]
            topic_slug = slugify_identifier(display_topic, fallback="topic")
            weights = [_memory_support_count(item) for item in grouped_objects]
            confidence = _weighted_mean(
                [_clamp(getattr(item, "confidence", 0.0), minimum=0.0, maximum=1.0) for item in grouped_objects],
                weights,
                default=0.5,
            )
            strongest_object = _strongest_object(grouped_objects)
            summary = truncate_text(
                getattr(strongest_object, "summary", ""),
                limit=180,
            )
            signals.append(
                InteractionSignal(
                    signal_id=f"signal:interaction:{turn_slug}:{topic_slug}",
                    signal_kind=INTERACTION_SIGNAL_TOPIC_AFFINITY,
                    target=display_topic,
                    summary=summary,
                    confidence=confidence,
                    impact=_clamp(confidence * 0.6, minimum=-1.0, maximum=1.0),
                    evidence_count=evidence_count,
                    source_event_ids=_coalesce_event_ids(
                        _memory_source_event_ids(item) for item in grouped_objects
                    ),
                    delta_target=f"{RELATIONSHIP_TOPIC_DELTA_PREFIX}{display_topic}",
                    delta_value=_topic_delta_value(
                        confidence=confidence,
                        evidence_count=evidence_count,
                    ),
                    delta_summary=f"Treat {display_topic} as a recurring user interest.",
                    metadata={
                        "signal_source": PLACE_SIGNAL_SOURCE_CONVERSATION,
                        "memory_ids": [getattr(item, "memory_id", "") for item in grouped_objects],
                        "trust_tier": "internal_structured",
                        "aggregation": "support_weighted",
                    },
                )
            )
        return tuple(signals)

    def _extract_place_signals_from_objects(
        self,
        *,
        turn_id: str,
        memory_objects: Sequence[LongTermMemoryObjectV1],
        occurred_at: datetime,
    ) -> tuple[PlaceSignal, ...]:
        """Build place signals from long-term objects that mention a location."""

        grouped: dict[str, list[LongTermMemoryObjectV1]] = defaultdict(list)
        place_labels: dict[str, str] = {}
        place_geographies: dict[str, str | None] = {}
        place_refs: dict[str, str | None] = {}
        for memory_object in memory_objects:
            if self.suppress_sensitive_context_signals and _is_sensitive_context(memory_object):
                continue
            place_name, geography, place_ref = _best_place_label(memory_object)
            if place_name is None:
                continue
            if self.suppress_precise_place_signals and geography == "place" and _looks_like_precise_place(place_name):
                continue
            place_key = place_name.casefold()
            place_labels.setdefault(place_key, place_name)
            place_geographies.setdefault(place_key, geography)
            place_refs.setdefault(place_key, place_ref)
            grouped[place_key].append(memory_object)

        signals: list[PlaceSignal] = []
        turn_slug = slugify_identifier(turn_id, fallback="turn")
        updated_at = _ensure_aware_utc(occurred_at).isoformat()
        min_support = max(1, int(self.min_place_signal_support))
        for place_key, grouped_objects in grouped.items():
            evidence_count = _total_support_count(grouped_objects)
            if evidence_count < min_support:
                continue
            display_place = place_labels[place_key]
            place_slug = slugify_identifier(display_place, fallback="place")
            weights = [_memory_support_count(item) for item in grouped_objects]
            confidence = _weighted_mean(
                [_clamp(getattr(item, "confidence", 0.0), minimum=0.0, maximum=1.0) for item in grouped_objects],
                weights,
                default=0.5,
            )
            strongest_object = _strongest_object(grouped_objects)
            signals.append(
                PlaceSignal(
                    signal_id=f"signal:place:{turn_slug}:{place_slug}",
                    place_name=display_place,
                    summary=truncate_text(getattr(strongest_object, "summary", ""), limit=180),
                    geography=place_geographies[place_key],
                    salience=_clamp(confidence, minimum=0.0, maximum=1.0),
                    confidence=confidence,
                    evidence_count=evidence_count,
                    source_event_ids=_coalesce_event_ids(
                        _memory_source_event_ids(item) for item in grouped_objects
                    ),
                    updated_at=updated_at,
                    metadata={
                        "signal_source": PLACE_SIGNAL_SOURCE_CONVERSATION,
                        "memory_ids": [getattr(item, "memory_id", "") for item in grouped_objects],
                        "place_ref": place_refs[place_key],
                        "trust_tier": "internal_structured",
                        "aggregation": "support_weighted",
                    },
                )
            )
        return tuple(signals)

    def _extract_live_search_signals(
        self,
        *,
        tool_call: AgentToolCall,
        tool_result: AgentToolResult,
        now: datetime,
    ) -> PersonalitySignalBatch:
        """Convert one successful live-search-like call into fresh context signals."""

        output = _safe_mapping(getattr(tool_result, "output", None))
        if not output:
            return PersonalitySignalBatch()
        status = _clean_text(output.get("status"), limit=32).lower()
        if status not in {"ok", "success"}:
            return PersonalitySignalBatch()

        arguments = _safe_mapping(getattr(tool_call, "arguments", None))
        question = _first_mapping_text(arguments, keys=_TOOL_ARGUMENT_QUESTION_KEYS, limit=96)
        if not question:
            return PersonalitySignalBatch()

        raw_answer = _first_mapping_text(output, keys=_TOOL_OUTPUT_ANSWER_KEYS, limit=320)
        if not raw_answer:
            return PersonalitySignalBatch()

        if self.sanitize_external_content:
            answer, risky_external_content = _filter_untrusted_external_text(raw_answer, limit=220)
        else:
            answer = _clean_text(raw_answer, limit=220)
            risky_external_content = False
        if not answer:
            return PersonalitySignalBatch()

        location_hint = _first_mapping_text(arguments, keys=_TOOL_ARGUMENT_LOCATION_KEYS, limit=80) or ""
        if self.suppress_precise_place_signals and location_hint and _looks_like_precise_place(location_hint):
            location_hint = ""

        response_id = _clean_text(output.get("response_id"), limit=96)
        source_event_ids = tuple(
            cleaned_id
            for cleaned_id in (
                _clean_text(getattr(tool_call, "call_id", None), limit=96),
                response_id,
            )
            if cleaned_id
        )

        source_count = 1
        for key in _TOOL_OUTPUT_SOURCE_KEYS:
            candidate = output.get(key)
            if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, bytearray)):
                source_count = max(1, len(candidate))
                break
        source_count = max(
            source_count,
            _coerce_positive_int(output.get("source_count"), default=source_count),
        )

        ttl_hours = _dynamic_world_signal_ttl_hours(question, default_hours=self.world_signal_ttl_hours)
        explicit_fresh_until = _safe_datetime(output.get("fresh_until"))
        if explicit_fresh_until is None or explicit_fresh_until <= now:
            fresh_until = now + timedelta(hours=ttl_hours)
        else:
            fresh_until = min(explicit_fresh_until, now + timedelta(hours=max(ttl_hours, 72)))

        salience = 0.58 + (0.04 * min(source_count, 4)) + (0.06 if location_hint else 0.0)
        if risky_external_content:
            salience -= 0.12
        salience = _clamp(salience, minimum=0.18, maximum=0.82)

        world_signal = WorldSignal(
            topic=question,
            summary=answer,
            region=location_hint or None,
            source=WORLD_SIGNAL_SOURCE_LIVE_SEARCH,
            salience=salience,
            fresh_until=fresh_until.isoformat(),
            evidence_count=source_count,
            source_event_ids=source_event_ids,
        )

        place_signals: tuple[PlaceSignal, ...] = ()
        if location_hint:
            place_signals = (
                PlaceSignal(
                    signal_id=(
                        "signal:place:"
                        f"{slugify_identifier(getattr(tool_call, 'call_id', None), fallback='tool')}:"
                        f"{slugify_identifier(location_hint, fallback='place')}"
                    ),
                    place_name=location_hint,
                    summary=f"Recent live information was requested about {location_hint}.",
                    geography="place",
                    salience=_clamp(0.62 if risky_external_content else 0.7, minimum=0.0, maximum=1.0),
                    confidence=0.62 if risky_external_content else 0.8,
                    evidence_count=1,
                    source_event_ids=source_event_ids,
                    updated_at=now.isoformat(),
                    metadata={
                        "signal_source": PLACE_SIGNAL_SOURCE_LIVE_SEARCH,
                        "question": question,
                        "trust_tier": "external_untrusted_sanitized" if risky_external_content else "external_untrusted",
                        "content_sanitized": self.sanitize_external_content,
                        "source_count": source_count,
                    },
                ),
            )

        return PersonalitySignalBatch(
            interaction_signals=(),
            place_signals=place_signals,
            world_signals=(world_signal,),
            continuity_threads=(),
        )