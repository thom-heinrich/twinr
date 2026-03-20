"""Extract world-intelligence calibration signals from structured runtime evidence.

This module keeps source-calibration learning separate from the generic
personality signal taxonomy. It turns existing structured personality evidence
and explicit tool usage into durable topic/region interest signals that the
RSS world-intelligence service can use during reflection-phase recalibration.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from twinr.agent.base_agent.contracts import AgentToolCall, AgentToolResult
from twinr.agent.personality.intelligence.models import WorldInterestSignal
from twinr.agent.personality.signals import (
    INTERACTION_SIGNAL_TOPIC_AFFINITY,
    INTERACTION_SIGNAL_TOPIC_AVERSION,
    PersonalitySignalBatch,
)
from twinr.text_utils import slugify_identifier, truncate_text

_SUPPORTED_TOOL_SIGNAL_NAMES = frozenset({"search_live_info", "configure_world_intelligence"})


def _utcnow() -> datetime:
    """Return the current UTC time."""

    return datetime.now(timezone.utc)


def _clean_text(value: object | None, *, limit: int = 120) -> str:
    """Normalize free-form input into bounded single-line text."""

    return truncate_text(None if value is None else str(value), limit=limit)


def _clamp(value: float, *, minimum: float, maximum: float) -> float:
    """Clamp one numeric value onto an inclusive range."""

    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


def _scope_from_region(region: str | None) -> str:
    """Choose the default calibration scope for one optional region."""

    return "local" if region else "topic"


@dataclass(frozen=True, slots=True)
class WorldIntelligenceSignalBatch:
    """Collect calibration signals emitted for the RSS intelligence service."""

    interest_signals: tuple[WorldInterestSignal, ...] = ()

    def has_any(self) -> bool:
        """Return whether the batch contains any calibration evidence."""

        return bool(self.interest_signals)


@dataclass(slots=True)
class WorldInterestSignalExtractor:
    """Build slow-changing world/place interest signals from runtime evidence."""

    now_provider: Callable[[], datetime] = _utcnow

    def extract_from_personality_batch(
        self,
        *,
        turn_id: str,
        batch: PersonalitySignalBatch,
        occurred_at: datetime,
    ) -> WorldIntelligenceSignalBatch:
        """Derive calibration signals from already-structured conversation evidence."""

        signals: list[WorldInterestSignal] = []
        occurred_at_iso = occurred_at.astimezone(timezone.utc).isoformat()
        strongest_place = next(iter(sorted(batch.place_signals, key=lambda item: item.salience, reverse=True)), None)

        for interaction_signal in batch.interaction_signals:
            if interaction_signal.signal_kind == INTERACTION_SIGNAL_TOPIC_AVERSION:
                continue
            if interaction_signal.signal_kind != INTERACTION_SIGNAL_TOPIC_AFFINITY:
                continue
            topic = _clean_text(interaction_signal.target, limit=96)
            if not topic:
                continue
            region = strongest_place.place_name if strongest_place is not None else None
            scope = _scope_from_region(region)
            topic_slug = slugify_identifier(topic, fallback="topic")
            region_slug = slugify_identifier(region or "global", fallback="region")
            signals.append(
                WorldInterestSignal(
                    signal_id=f"interest:conversation:{slugify_identifier(turn_id, fallback='turn')}:{region_slug}:{topic_slug}",
                    topic=topic,
                    summary=truncate_text(
                        f"Recurring conversation interest in {topic}."
                        if region is None
                        else f"Recurring conversation interest in {topic} around {region}.",
                        limit=180,
                    ),
                    region=region,
                    scope=scope,
                    salience=_clamp(interaction_signal.confidence * 0.75, minimum=0.35, maximum=0.92),
                    confidence=interaction_signal.confidence,
                    evidence_count=max(1, interaction_signal.evidence_count),
                    explicit=False,
                    source_event_ids=interaction_signal.source_event_ids,
                    updated_at=occurred_at_iso,
                )
            )

        for thread in batch.continuity_threads:
            title = _clean_text(thread.title, limit=96)
            if not title:
                continue
            signals.append(
                WorldInterestSignal(
                    signal_id=f"interest:thread:{slugify_identifier(title, fallback='thread')}",
                    topic=title,
                    summary=truncate_text(
                        f"Ongoing continuity around {title} suggests durable situational relevance.",
                        limit=180,
                    ),
                    region=strongest_place.place_name if strongest_place is not None else None,
                    scope=_scope_from_region(strongest_place.place_name if strongest_place is not None else None),
                    salience=_clamp(thread.salience * 0.8, minimum=0.3, maximum=0.9),
                    confidence=_clamp(thread.salience, minimum=0.0, maximum=1.0),
                    evidence_count=2,
                    explicit=False,
                    source_event_ids=(),
                    updated_at=occurred_at_iso,
                )
            )

        return WorldIntelligenceSignalBatch(interest_signals=tuple(signals))

    def extract_from_tool_history(
        self,
        *,
        tool_calls: Sequence[AgentToolCall],
        tool_results: Sequence[AgentToolResult],
    ) -> WorldIntelligenceSignalBatch:
        """Derive calibration signals from explicit search and RSS tool usage."""

        result_by_call_id = {result.call_id: result for result in tool_results}
        now_iso = self.now_provider().astimezone(timezone.utc).isoformat()
        signals: list[WorldInterestSignal] = []

        for tool_call in tool_calls:
            if tool_call.name not in _SUPPORTED_TOOL_SIGNAL_NAMES:
                continue
            tool_result = result_by_call_id.get(tool_call.call_id)
            if tool_result is None:
                continue
            if tool_call.name == "search_live_info":
                signals.extend(
                    self._from_live_search(
                        tool_call=tool_call,
                        tool_result=tool_result,
                        updated_at=now_iso,
                    )
                )
                continue
            if tool_call.name == "configure_world_intelligence":
                signals.extend(
                    self._from_world_intelligence_tool(
                        tool_call=tool_call,
                        tool_result=tool_result,
                        updated_at=now_iso,
                    )
                )

        return WorldIntelligenceSignalBatch(interest_signals=tuple(signals))

    def _from_live_search(
        self,
        *,
        tool_call: AgentToolCall,
        tool_result: AgentToolResult,
        updated_at: str,
    ) -> tuple[WorldInterestSignal, ...]:
        """Interpret successful live-search usage as topical situational interest."""

        output = tool_result.output
        if not isinstance(output, Mapping):
            return ()
        if _clean_text(output.get("status"), limit=32).casefold() != "ok":
            return ()
        question = _clean_text(tool_call.arguments.get("question"), limit=96)
        if not question:
            return ()
        region = _clean_text(tool_call.arguments.get("location_hint"), limit=80) or None
        return (
            WorldInterestSignal(
                signal_id=f"interest:tool:{tool_call.call_id}",
                topic=question,
                summary=truncate_text(
                    "Recent live search suggests the user wanted fresh situational awareness."
                    if region is None
                    else f"Recent live search suggests the user wanted fresh situational awareness about {region}.",
                    limit=180,
                ),
                region=region,
                scope=_scope_from_region(region),
                salience=0.62 if region is None else 0.72,
                confidence=0.7,
                evidence_count=1,
                explicit=False,
                source_event_ids=tuple(
                    item
                    for item in (tool_call.call_id, _clean_text(output.get("response_id"), limit=96))
                    if item
                ),
                updated_at=updated_at,
            ),
        )

    def _from_world_intelligence_tool(
        self,
        *,
        tool_call: AgentToolCall,
        tool_result: AgentToolResult,
        updated_at: str,
    ) -> tuple[WorldInterestSignal, ...]:
        """Interpret explicit RSS configuration as strong calibration evidence."""

        output = tool_result.output
        if not isinstance(output, Mapping):
            return ()
        if _clean_text(output.get("status"), limit=32).casefold() != "ok":
            return ()
        action = _clean_text(tool_call.arguments.get("action"), limit=32).casefold()
        if action not in {"subscribe", "discover", "refresh_now"}:
            return ()
        topics = tool_call.arguments.get("topics")
        if isinstance(topics, Sequence) and not isinstance(topics, (str, bytes, bytearray)):
            normalized_topics = tuple(
                topic
                for topic in (_clean_text(item, limit=96) for item in topics)
                if topic
            )
        else:
            normalized_topics = ()
        label = _clean_text(tool_call.arguments.get("label"), limit=96)
        region = _clean_text(tool_call.arguments.get("region") or tool_call.arguments.get("location_hint"), limit=80) or None
        fallback_topic = label or "configured world awareness"
        signal_topics = normalized_topics or (fallback_topic,)
        signals: list[WorldInterestSignal] = []
        for topic in signal_topics:
            signals.append(
                WorldInterestSignal(
                    signal_id=f"interest:tool-config:{tool_call.call_id}:{slugify_identifier(topic, fallback='topic')}",
                    topic=topic,
                    summary=truncate_text(
                        f"Explicit world-intelligence configuration for {topic}.",
                        limit=180,
                    ),
                    region=region,
                    scope=_scope_from_region(region),
                    salience=0.9,
                    confidence=0.95,
                    evidence_count=2,
                    explicit=True,
                    source_event_ids=(tool_call.call_id,),
                    updated_at=updated_at,
                )
            )
        return tuple(signals)
