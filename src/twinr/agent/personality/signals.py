"""Extract structured personality-learning signals from Twinr runtime events.

This module keeps the early personality-learning taxonomy explicit and
separate from prompt rendering. V1 focuses on three concrete signal families:

- interaction signals: repeated topic/style evidence that may eventually
  become policy-gated relationship or character deltas
- place signals: geographic areas that keep surfacing in the user's life and
  therefore deserve prompt-time situational awareness
- world signals: fresh developments from live search or future news ingestion

The extraction logic stays deliberately conservative. It relies on structured
long-term memory objects and structured tool call/result payloads instead of
free-form text heuristics so Twinr can learn slowly without fragile parsing.
"""

from __future__ import annotations

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
SUPPORTED_TOOL_SIGNAL_NAMES = frozenset({"search_live_info"})

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
_PLACE_ATTRIBUTE_KEYS = (
    "place",
    "place_name",
    "location",
    "location_name",
    "city",
    "town",
    "district",
    "municipality",
    "county",
    "region",
    "state",
    "country",
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
    }
)
_SENSITIVE_DOMAIN_TOKENS = frozenset({"medical", "health", "distress", "crisis", "emergency", "care"})
_POSITIVE_SIGNAL_VALUES = frozenset({"positive", "more", "yes", "true", "liked", "helpful", "good"})
_NEGATIVE_SIGNAL_VALUES = frozenset({"negative", "less", "no", "false", "disliked", "too_much", "bad"})


def _utcnow() -> datetime:
    """Return the current UTC time as an aware ``datetime``."""

    return datetime.now(timezone.utc)


def _clean_text(value: object | None, *, limit: int | None = None) -> str:
    """Normalize arbitrary input into one bounded single-line string."""

    return truncate_text(None if value is None else str(value), limit=limit)


def _mean(values: Sequence[float], *, default: float = 0.0) -> float:
    """Return the arithmetic mean of a numeric sequence."""

    if not values:
        return default
    return sum(values) / len(values)


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


def _coalesce_event_ids(event_id_groups: Iterable[Sequence[str]]) -> tuple[str, ...]:
    """Return de-duplicated event ids while preserving their first-seen order."""

    ordered: list[str] = []
    seen: set[str] = set()
    for group in event_id_groups:
        for event_id in group:
            normalized = _clean_text(event_id, limit=96)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
    return tuple(ordered)


def _coerce_positive_int(value: object | None, *, default: int = 1) -> int:
    """Coerce one structured support-count-like value into a positive integer."""

    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def _attributes(memory_object: LongTermMemoryObjectV1) -> Mapping[str, object]:
    """Return a safe attributes mapping for one memory object."""

    attributes = memory_object.attributes
    if isinstance(attributes, Mapping):
        return attributes
    return {}


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


def _memory_support_count(memory_object: LongTermMemoryObjectV1) -> int:
    """Return the structured support count carried by one memory object."""

    return _coerce_positive_int(_attributes(memory_object).get("support_count"), default=1)


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


def _best_topic_label(memory_object: LongTermMemoryObjectV1) -> str | None:
    """Choose the strongest conversation topic label from one memory object."""

    attributes = memory_object.attributes or {}
    for key in _TOPIC_ATTRIBUTE_KEYS:
        raw_value = attributes.get(key)
        normalized = _clean_text(raw_value, limit=80)
        if not normalized:
            continue
        if normalized.replace(" ", "_").lower() in _GENERIC_TOPIC_VALUES:
            continue
        return normalized
    return None


def _best_place_label(memory_object: LongTermMemoryObjectV1) -> tuple[str | None, str | None, str | None]:
    """Choose the strongest place label, geography, and ref from one memory object."""

    attributes = memory_object.attributes or {}
    for key in _PLACE_ATTRIBUTE_KEYS:
        raw_value = attributes.get(key)
        normalized = _clean_text(raw_value, limit=80)
        if not normalized:
            continue
        geography = key
        if key in {"place", "place_name", "location", "location_name"}:
            geography = "place"
        place_ref = _clean_text(attributes.get("place_ref"), limit=96) or None
        return normalized, geography, place_ref
    return None, None, _clean_text(attributes.get("place_ref"), limit=96) or None


def _topic_delta_value(*, confidence: float, evidence_count: int) -> float:
    """Convert repeated topic evidence into one bounded salience delta."""

    weighted = confidence * min(max(evidence_count, 1), 3)
    return _clamp(weighted * 0.12, minimum=0.08, maximum=0.3)


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

    def merged(self, other: "PersonalitySignalBatch") -> "PersonalitySignalBatch":
        """Return one batch containing both inputs in stable append order."""

        return PersonalitySignalBatch(
            interaction_signals=self.interaction_signals + other.interaction_signals,
            place_signals=self.place_signals + other.place_signals,
            world_signals=self.world_signals + other.world_signals,
            continuity_threads=self.continuity_threads + other.continuity_threads,
        )


@dataclass(slots=True)
class PersonalitySignalExtractor:
    """Extract personality-learning signals from conversation and tool history.

    Attributes:
        now_provider: Clock used for freshness windows and deterministic tests.
        world_signal_ttl_hours: Freshness horizon for live-search world signals.
        continuity_thread_ttl_days: Default review horizon for continuity
            threads derived from repeated turn summaries.
    """

    now_provider: Callable[[], datetime] = _utcnow
    world_signal_ttl_hours: int = 24
    continuity_thread_ttl_days: int = 21

    def extract_from_consolidation(
        self,
        *,
        turn: LongTermConversationTurn,
        consolidation: LongTermConsolidationResultV1,
    ) -> PersonalitySignalBatch:
        """Extract interaction and place signals from one consolidated turn."""

        del turn  # Reserved for future explicit-style/humor extraction from structured turn context.
        relevant_objects = tuple((*consolidation.durable_objects, *consolidation.deferred_objects))
        return PersonalitySignalBatch(
            interaction_signals=(
                self._extract_topic_affinity_signals(
                    turn_id=consolidation.turn_id,
                    memory_objects=relevant_objects,
                )
                + self._extract_style_and_feedback_signals(
                    turn_id=consolidation.turn_id,
                    memory_objects=relevant_objects,
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

    def extract_from_tool_history(
        self,
        *,
        tool_calls: Sequence[AgentToolCall],
        tool_results: Sequence[AgentToolResult],
    ) -> PersonalitySignalBatch:
        """Extract world and place signals from structured tool call history."""

        result_by_call_id = {result.call_id: result for result in tool_results}
        now = self.now_provider().astimezone(timezone.utc)
        place_signals: list[PlaceSignal] = []
        world_signals: list[WorldSignal] = []

        for tool_call in tool_calls:
            if tool_call.name not in SUPPORTED_TOOL_SIGNAL_NAMES:
                continue
            tool_result = result_by_call_id.get(tool_call.call_id)
            if tool_result is None:
                continue
            if tool_call.name == "search_live_info":
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
        )

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
                memory_slug = slugify_identifier(memory_object.memory_id, fallback="verbosity")
                confidence = _clamp(memory_object.confidence, minimum=0.0, maximum=1.0)
                support_count = _memory_support_count(memory_object)
                signals.append(
                    InteractionSignal(
                        signal_id=f"signal:interaction:{turn_slug}:{memory_slug}",
                        signal_kind=INTERACTION_SIGNAL_VERBOSITY_PREFERENCE,
                        target="verbosity",
                        summary=truncate_text(memory_object.summary, limit=180),
                        confidence=confidence,
                        impact=confidence * 0.5,
                        evidence_count=support_count,
                        source_event_ids=_coalesce_event_ids((memory_object.source.event_ids,)),
                        delta_target=STYLE_VERBOSITY_DELTA_TARGET,
                        delta_value=delta_value,
                        delta_summary=delta_summary,
                        metadata={
                            "signal_source": PLACE_SIGNAL_SOURCE_CONVERSATION,
                            "memory_id": memory_object.memory_id,
                            "sensitive_context": _is_sensitive_context(memory_object),
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
                memory_slug = slugify_identifier(memory_object.memory_id, fallback="initiative")
                confidence = _clamp(memory_object.confidence, minimum=0.0, maximum=1.0)
                support_count = _memory_support_count(memory_object)
                signals.append(
                    InteractionSignal(
                        signal_id=f"signal:interaction:{turn_slug}:{memory_slug}",
                        signal_kind=INTERACTION_SIGNAL_INITIATIVE_PREFERENCE,
                        target="initiative",
                        summary=truncate_text(memory_object.summary, limit=180),
                        confidence=confidence,
                        impact=confidence * 0.4,
                        evidence_count=support_count,
                        source_event_ids=_coalesce_event_ids((memory_object.source.event_ids,)),
                        delta_target=STYLE_INITIATIVE_DELTA_TARGET,
                        delta_value=delta_value,
                        delta_summary=delta_summary,
                        metadata={
                            "signal_source": PLACE_SIGNAL_SOURCE_CONVERSATION,
                            "memory_id": memory_object.memory_id,
                            "sensitive_context": _is_sensitive_context(memory_object),
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
                memory_slug = slugify_identifier(memory_object.memory_id, fallback="humor")
                confidence = _clamp(memory_object.confidence, minimum=0.0, maximum=1.0)
                support_count = _memory_support_count(memory_object)
                signals.append(
                    InteractionSignal(
                        signal_id=f"signal:interaction:{turn_slug}:{memory_slug}",
                        signal_kind=INTERACTION_SIGNAL_HUMOR_FEEDBACK,
                        target="humor",
                        summary=truncate_text(memory_object.summary, limit=180),
                        confidence=confidence,
                        impact=confidence * 0.3,
                        evidence_count=support_count,
                        source_event_ids=_coalesce_event_ids((memory_object.source.event_ids,)),
                        delta_target="humor.intensity",
                        delta_value=delta_value,
                        delta_summary=delta_summary,
                        metadata={
                            "signal_source": PLACE_SIGNAL_SOURCE_CONVERSATION,
                            "memory_id": memory_object.memory_id,
                            "sensitive_context": _is_sensitive_context(memory_object),
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
            confidence = _clamp(memory_object.confidence, minimum=0.0, maximum=1.0)
            support_count = _memory_support_count(memory_object)
            topic_slug = slugify_identifier(topic, fallback="topic")
            signals.append(
                InteractionSignal(
                    signal_id=f"signal:interaction:{turn_slug}:{topic_slug}:aversion",
                    signal_kind=INTERACTION_SIGNAL_TOPIC_AVERSION,
                    target=topic,
                    summary=truncate_text(memory_object.summary, limit=180),
                    confidence=confidence,
                    impact=-confidence * 0.5,
                    evidence_count=support_count,
                    source_event_ids=_coalesce_event_ids((memory_object.source.event_ids,)),
                    delta_target=f"{RELATIONSHIP_TOPIC_AVERSION_DELTA_PREFIX}{topic}",
                    delta_value=_topic_delta_value(
                        confidence=confidence,
                        evidence_count=support_count,
                    ),
                    delta_summary=f"Avoid dwelling on {topic} unless the user explicitly asks.",
                    metadata={
                        "signal_source": PLACE_SIGNAL_SOURCE_CONVERSATION,
                        "memory_id": memory_object.memory_id,
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
        updated_at = occurred_at.astimezone(timezone.utc)
        for memory_object in memory_objects:
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
            expires_at = _clean_text(getattr(memory_object, "valid_to", None), limit=64) or (
                updated_at + timedelta(days=max(7, int(self.continuity_thread_ttl_days)))
            ).isoformat()
            threads.append(
                ContinuityThread(
                    title=title,
                    summary=truncate_text(memory_object.summary, limit=180),
                    salience=_clamp(
                        max(memory_object.confidence, min(0.95, 0.35 + (support_count * 0.1))),
                        minimum=0.0,
                        maximum=1.0,
                    ),
                    updated_at=updated_at.isoformat(),
                    expires_at=expires_at,
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
            memory_attributes = _attributes(memory_object)
            if _clean_token(memory_attributes.get("memory_domain")) in {"preference", "thread"}:
                continue
            if _clean_token(memory_attributes.get("fact_type")) in {"preference", "feedback"}:
                continue
            if _clean_token(memory_attributes.get("summary_type")) == "thread":
                continue
            topic = _best_topic_label(memory_object)
            if topic is None:
                continue
            topic_key = topic.casefold()
            topic_labels.setdefault(topic_key, topic)
            grouped[topic_key].append(memory_object)

        signals: list[InteractionSignal] = []
        turn_slug = slugify_identifier(turn_id, fallback="turn")
        for topic_key, grouped_objects in grouped.items():
            display_topic = topic_labels[topic_key]
            topic_slug = slugify_identifier(display_topic, fallback="topic")
            confidence = _mean([item.confidence for item in grouped_objects], default=0.5)
            evidence_count = len(grouped_objects)
            summary = truncate_text(
                grouped_objects[0].summary,
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
                        item.source.event_ids for item in grouped_objects
                    ),
                    delta_target=f"{RELATIONSHIP_TOPIC_DELTA_PREFIX}{display_topic}",
                    delta_value=_topic_delta_value(
                        confidence=confidence,
                        evidence_count=evidence_count,
                    ),
                    delta_summary=f"Treat {display_topic} as a recurring user interest.",
                    metadata={
                        "signal_source": PLACE_SIGNAL_SOURCE_CONVERSATION,
                        "memory_ids": [item.memory_id for item in grouped_objects],
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
            place_name, geography, place_ref = _best_place_label(memory_object)
            if place_name is None:
                continue
            place_key = place_name.casefold()
            place_labels.setdefault(place_key, place_name)
            place_geographies.setdefault(place_key, geography)
            place_refs.setdefault(place_key, place_ref)
            grouped[place_key].append(memory_object)

        signals: list[PlaceSignal] = []
        turn_slug = slugify_identifier(turn_id, fallback="turn")
        updated_at = occurred_at.astimezone(timezone.utc).isoformat()
        for place_key, grouped_objects in grouped.items():
            display_place = place_labels[place_key]
            place_slug = slugify_identifier(display_place, fallback="place")
            confidence = _mean([item.confidence for item in grouped_objects], default=0.5)
            signals.append(
                PlaceSignal(
                    signal_id=f"signal:place:{turn_slug}:{place_slug}",
                    place_name=display_place,
                    summary=truncate_text(grouped_objects[0].summary, limit=180),
                    geography=place_geographies[place_key],
                    salience=_clamp(confidence, minimum=0.0, maximum=1.0),
                    confidence=confidence,
                    evidence_count=len(grouped_objects),
                    source_event_ids=_coalesce_event_ids(
                        item.source.event_ids for item in grouped_objects
                    ),
                    updated_at=updated_at,
                    metadata={
                        "signal_source": PLACE_SIGNAL_SOURCE_CONVERSATION,
                        "memory_ids": [item.memory_id for item in grouped_objects],
                        "place_ref": place_refs[place_key],
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
        """Convert one successful ``search_live_info`` call into fresh context signals."""

        output = tool_result.output
        if not isinstance(output, Mapping):
            return PersonalitySignalBatch()
        status = _clean_text(output.get("status"), limit=32).lower()
        answer = _clean_text(output.get("answer"), limit=220)
        if status != "ok" or not answer:
            return PersonalitySignalBatch()

        question = _clean_text(tool_call.arguments.get("question"), limit=96)
        if not question:
            return PersonalitySignalBatch()
        location_hint = _clean_text(tool_call.arguments.get("location_hint"), limit=80)
        response_id = _clean_text(output.get("response_id"), limit=96)
        source_event_ids = tuple(
            item
            for item in (tool_call.call_id, response_id)
            if item
        )

        world_signal = WorldSignal(
            topic=question,
            summary=answer,
            region=location_hint or None,
            source=WORLD_SIGNAL_SOURCE_LIVE_SEARCH,
            salience=0.72 if location_hint else 0.64,
            fresh_until=(now + timedelta(hours=max(1, int(self.world_signal_ttl_hours)))).isoformat(),
            evidence_count=1,
            source_event_ids=source_event_ids,
        )
        place_signals: tuple[PlaceSignal, ...] = ()
        if location_hint:
            place_signals = (
                PlaceSignal(
                    signal_id=f"signal:place:{slugify_identifier(tool_call.call_id, fallback='tool')}:{slugify_identifier(location_hint, fallback='place')}",
                    place_name=location_hint,
                    summary=f"Recent live information was requested about {location_hint}.",
                    geography="place",
                    salience=0.7,
                    confidence=0.8,
                    evidence_count=1,
                    source_event_ids=source_event_ids,
                    updated_at=now.isoformat(),
                    metadata={
                        "signal_source": PLACE_SIGNAL_SOURCE_LIVE_SEARCH,
                        "question": question,
                    },
                ),
            )
        return PersonalitySignalBatch(
            interaction_signals=(),
            place_signals=place_signals,
            world_signals=(world_signal,),
            continuity_threads=(),
        )
