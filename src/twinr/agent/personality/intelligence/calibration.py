"""Extract world-intelligence calibration signals from structured runtime evidence.

This module keeps source-calibration learning separate from the generic
personality signal taxonomy. It turns existing structured personality evidence
and explicit tool usage into durable topic/region interest signals that the
RSS world-intelligence service can use during reflection-phase recalibration.

Each interest signal also carries a bounded ``engagement_score``. That score is
not a guess about feelings; it is Twinr's generic estimate of how much a topic
is currently pulling the user back in via repeated conversation evidence,
follow-up search behavior, or explicit world-intelligence configuration.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from twinr.agent.base_agent.contracts import AgentToolCall, AgentToolResult
from twinr.agent.personality.intelligence.models import WorldInterestSignal
from twinr.agent.personality.signals import (
    INTERACTION_SIGNAL_TOPIC_AFFINITY,
    INTERACTION_SIGNAL_TOPIC_AVERSION,
    INTERACTION_SIGNAL_TOPIC_COOLING,
    INTERACTION_SIGNAL_TOPIC_ENGAGEMENT,
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


def _parse_iso(value: str | None) -> datetime | None:
    """Parse one optional ISO timestamp into aware UTC time."""

    text = _clean_text(value)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _topic_key(*, topic: str, region: str | None, scope: str) -> tuple[str, str, str]:
    """Build one stable topic key for same-topic comparisons."""

    return (
        _clean_text(topic, limit=96).casefold(),
        _clean_text(region, limit=96).casefold(),
        _clean_text(scope, limit=32).casefold() or "topic",
    )


def _engagement_score(*, confidence: float, evidence_count: int, base: float) -> float:
    """Convert repeated structured evidence into one bounded engagement score."""

    return _clamp(
        base + (confidence * 0.28) + (min(max(evidence_count, 1), 4) * 0.05),
        minimum=0.3,
        maximum=0.98,
    )


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
    follow_through_window_days: int = 21
    follow_through_candidate_limit: int = 2

    def extract_from_personality_batch(
        self,
        *,
        turn_id: str,
        batch: PersonalitySignalBatch,
        occurred_at: datetime,
        existing_interest_signals: Sequence[WorldInterestSignal] = (),
    ) -> WorldIntelligenceSignalBatch:
        """Derive calibration signals from already-structured conversation evidence."""

        signals: list[WorldInterestSignal] = []
        occurred_at_iso = occurred_at.astimezone(timezone.utc).isoformat()
        strongest_place = next(iter(sorted(batch.place_signals, key=lambda item: item.salience, reverse=True)), None)

        for interaction_signal in batch.interaction_signals:
            if interaction_signal.signal_kind == INTERACTION_SIGNAL_TOPIC_AVERSION:
                pass
            elif interaction_signal.signal_kind not in {
                INTERACTION_SIGNAL_TOPIC_AFFINITY,
                INTERACTION_SIGNAL_TOPIC_COOLING,
                INTERACTION_SIGNAL_TOPIC_ENGAGEMENT,
            }:
                continue
            topic = _clean_text(interaction_signal.target, limit=96)
            if not topic:
                continue
            region = strongest_place.place_name if strongest_place is not None else None
            scope = _scope_from_region(region)
            topic_slug = slugify_identifier(topic, fallback="topic")
            region_slug = slugify_identifier(region or "global", fallback="region")
            is_engagement = interaction_signal.signal_kind == INTERACTION_SIGNAL_TOPIC_ENGAGEMENT
            is_cooling = interaction_signal.signal_kind == INTERACTION_SIGNAL_TOPIC_COOLING
            is_avoid = interaction_signal.signal_kind == INTERACTION_SIGNAL_TOPIC_AVERSION
            engagement_base = 0.68 if interaction_signal.explicit_user_requested else 0.58
            if not is_engagement:
                engagement_base = 0.42
            if is_cooling:
                engagement_base = 0.2 if interaction_signal.explicit_user_requested else 0.28
            if is_avoid:
                engagement_base = 0.08 if interaction_signal.explicit_user_requested else 0.14
            metadata = interaction_signal.metadata or {}
            exposure_count = int(metadata.get("exposure_count", max(1, interaction_signal.evidence_count)))
            non_reengagement_count = int(metadata.get("non_reengagement_count", 0))
            deflection_count = int(metadata.get("deflection_count", 0))
            if is_avoid:
                exposure_count = max(exposure_count, 2)
                non_reengagement_count = max(non_reengagement_count, 1)
                deflection_count = max(deflection_count, 2 if interaction_signal.explicit_user_requested else 1)
            signals.append(
                WorldInterestSignal(
                    signal_id=(
                        f"interest:conversation:{slugify_identifier(turn_id, fallback='turn')}:{region_slug}:{topic_slug}"
                        if not is_engagement and not is_cooling and not is_avoid
                        else (
                            f"interest:engagement:{slugify_identifier(turn_id, fallback='turn')}:{region_slug}:{topic_slug}"
                            if is_engagement
                            else (
                                f"interest:cooling:{slugify_identifier(turn_id, fallback='turn')}:{region_slug}:{topic_slug}"
                                if is_cooling
                                else f"interest:avoid:{slugify_identifier(turn_id, fallback='turn')}:{region_slug}:{topic_slug}"
                            )
                        )
                    ),
                    topic=topic,
                    summary=truncate_text(
                        (
                            f"Recurring conversation interest in {topic}."
                            if region is None
                            else f"Recurring conversation interest in {topic} around {region}."
                        )
                        if not is_engagement and not is_cooling and not is_avoid
                        else (
                            f"The user showed strong follow-up engagement with {topic}."
                            if region is None
                            else f"The user showed strong follow-up engagement with {topic} around {region}."
                        )
                        if is_engagement
                        else (
                            f"{topic} did not pull the user back in after repeated exposure."
                            if region is None
                            else f"{topic} did not pull the user back in around {region} after repeated exposure."
                        )
                        if is_cooling
                        else (
                            f"The user explicitly wants Twinr to back off from {topic}."
                            if region is None
                            else f"The user explicitly wants Twinr to back off from {topic} around {region}."
                        ),
                        limit=180,
                    ),
                    region=region,
                    scope=scope,
                    salience=_clamp(
                        interaction_signal.confidence
                        * (
                            0.75
                            if not is_engagement and not is_cooling and not is_avoid
                            else (0.88 if is_engagement else (0.55 if is_cooling else 0.4))
                        ),
                        minimum=(
                            0.35
                            if not is_engagement and not is_cooling and not is_avoid
                            else (0.45 if is_engagement else (0.18 if is_cooling else 0.12))
                        ),
                        maximum=(
                            0.92
                            if not is_engagement and not is_cooling and not is_avoid
                            else (0.97 if is_engagement else (0.72 if is_cooling else 0.48))
                        ),
                    ),
                    confidence=interaction_signal.confidence,
                    engagement_score=(
                        _engagement_score(
                            confidence=interaction_signal.confidence,
                            evidence_count=interaction_signal.evidence_count,
                            base=engagement_base,
                        )
                        if not is_cooling and not is_avoid
                        else _clamp(
                            engagement_base
                            + (interaction_signal.confidence * (0.08 if is_cooling else 0.04))
                            - (non_reengagement_count * 0.08)
                            - (deflection_count * (0.12 if is_cooling else 0.16)),
                            minimum=0.01 if is_avoid else 0.05,
                            maximum=0.22 if is_avoid else 0.55,
                        )
                    ),
                    evidence_count=max(1, interaction_signal.evidence_count),
                    engagement_count=max(
                        0 if is_cooling or is_avoid else (1 if not is_engagement else 2),
                        interaction_signal.evidence_count
                        + (1 if is_engagement else 0)
                        - (1 if is_cooling else 0)
                        - (2 if is_avoid else 0),
                    ),
                    positive_signal_count=(
                        max(1, interaction_signal.evidence_count + (1 if is_engagement else 0))
                        if not is_cooling and not is_avoid
                        else 0
                    ),
                    exposure_count=max(exposure_count, 1 if not is_cooling else 2),
                    non_reengagement_count=non_reengagement_count,
                    deflection_count=deflection_count,
                    explicit=interaction_signal.explicit_user_requested,
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
                    engagement_score=_engagement_score(
                        confidence=_clamp(thread.salience, minimum=0.0, maximum=1.0),
                        evidence_count=2,
                        base=0.58,
                    ),
                    evidence_count=2,
                    engagement_count=2,
                    positive_signal_count=2,
                    exposure_count=2,
                    explicit=False,
                    source_event_ids=(),
                    updated_at=occurred_at_iso,
                )
            )

        signals.extend(
            self._derive_follow_through_interest_signals(
                turn_id=turn_id,
                batch=batch,
                occurred_at=occurred_at,
                strongest_region=strongest_place.place_name if strongest_place is not None else None,
                existing_interest_signals=existing_interest_signals,
            )
        )

        return WorldIntelligenceSignalBatch(interest_signals=tuple(signals))

    def _derive_follow_through_interest_signals(
        self,
        *,
        turn_id: str,
        batch: PersonalitySignalBatch,
        occurred_at: datetime,
        strongest_region: str | None,
        existing_interest_signals: Sequence[WorldInterestSignal],
    ) -> tuple[WorldInterestSignal, ...]:
        """Derive mild cooling evidence from repeated non-reengagement.

        This path stays conservative. It only emits negative evidence when the
        current turn shows clear topical pull elsewhere and Twinr already had
        an existing warm/resonant topic that did not get picked back up. One
        missed follow-up does not become dislike on its own; it merely starts a
        bounded, cross-session cooling trail.
        """

        current_topics = self._current_topic_strengths(
            batch=batch,
            strongest_region=strongest_region,
        )
        if not current_topics or not existing_interest_signals:
            return ()
        strongest_current = max(current_topics.values())
        if strongest_current < 0.55:
            return ()

        occurred_at_iso = occurred_at.astimezone(timezone.utc).isoformat()
        fresh_cutoff = occurred_at.astimezone(timezone.utc) - timedelta(days=max(1, self.follow_through_window_days))
        turn_slug = slugify_identifier(turn_id, fallback="turn")
        ranked_candidates = sorted(
            (
                signal
                for signal in existing_interest_signals
                if self._is_follow_through_candidate(
                    signal=signal,
                    current_topics=current_topics,
                    fresh_cutoff=fresh_cutoff,
                )
            ),
            key=lambda signal: (
                self._existing_interest_priority(signal),
                signal.engagement_score,
                signal.salience,
            ),
            reverse=True,
        )

        cooling_signals: list[WorldInterestSignal] = []
        current_label = self._dominant_current_topic_label(current_topics)
        for signal in ranked_candidates[: max(0, self.follow_through_candidate_limit)]:
            topic_slug = slugify_identifier(signal.topic, fallback="topic")
            cooling_signals.append(
                WorldInterestSignal(
                    signal_id=f"interest:follow-through:{turn_slug}:{topic_slug}",
                    topic=signal.topic,
                    summary=truncate_text(
                        (
                            f"This turn moved back toward {current_label} instead of reopening {signal.topic}."
                            if current_label
                            else f"This turn did not reopen {signal.topic}."
                        ),
                        limit=180,
                    ),
                    region=signal.region,
                    scope=signal.scope,
                    salience=_clamp(signal.salience * 0.36, minimum=0.18, maximum=0.46),
                    confidence=_clamp(0.42 + (strongest_current * 0.25), minimum=0.42, maximum=0.7),
                    engagement_score=_clamp(
                        signal.engagement_score - 0.22,
                        minimum=0.08,
                        maximum=0.48,
                    ),
                    evidence_count=1,
                    engagement_count=0,
                    positive_signal_count=0,
                    exposure_count=1,
                    non_reengagement_count=1,
                    explicit=False,
                    source_event_ids=(),
                    updated_at=occurred_at_iso,
                )
            )
        return tuple(cooling_signals)

    def _current_topic_strengths(
        self,
        *,
        batch: PersonalitySignalBatch,
        strongest_region: str | None,
    ) -> dict[tuple[str, str, str], float]:
        """Rank the topics that genuinely pulled the current turn forward."""

        strengths: dict[tuple[str, str, str], float] = {}
        scope = _scope_from_region(strongest_region)
        for signal in batch.interaction_signals:
            if signal.signal_kind not in {
                INTERACTION_SIGNAL_TOPIC_AFFINITY,
                INTERACTION_SIGNAL_TOPIC_ENGAGEMENT,
            }:
                continue
            key = _topic_key(
                topic=signal.target,
                region=strongest_region,
                scope=scope,
            )
            pull = _clamp(
                0.38
                + (signal.confidence * 0.32)
                + (0.14 if signal.signal_kind == INTERACTION_SIGNAL_TOPIC_ENGAGEMENT else 0.0)
                + (0.08 if signal.explicit_user_requested else 0.0),
                minimum=0.0,
                maximum=1.0,
            )
            strengths[key] = max(strengths.get(key, 0.0), pull)
        return strengths

    def _is_follow_through_candidate(
        self,
        *,
        signal: WorldInterestSignal,
        current_topics: Mapping[tuple[str, str, str], float],
        fresh_cutoff: datetime,
    ) -> bool:
        """Return whether one existing interest is eligible for mild cooling."""

        if signal.engagement_state not in {"resonant", "warm"}:
            return False
        if _topic_key(topic=signal.topic, region=signal.region, scope=signal.scope) in current_topics:
            return False
        updated_at = _parse_iso(signal.updated_at)
        if updated_at is not None and updated_at < fresh_cutoff:
            return False
        return signal.engagement_score >= 0.58 or signal.positive_signal_count >= 1

    def _existing_interest_priority(self, signal: WorldInterestSignal) -> float:
        """Score one existing interest for bounded follow-through cooling."""

        state_bonus = 0.22 if signal.engagement_state == "resonant" else 0.12
        return signal.engagement_score + (signal.salience * 0.25) + state_bonus

    def _dominant_current_topic_label(
        self,
        current_topics: Mapping[tuple[str, str, str], float],
    ) -> str | None:
        """Return a readable label for the strongest currently engaged topic."""

        if not current_topics:
            return None
        strongest_key = max(current_topics.items(), key=lambda item: item[1])[0]
        topic, region, _scope = strongest_key
        if region:
            return f"{topic} around {region}"
        return topic

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
                engagement_score=0.82 if region is not None else 0.76,
                evidence_count=1,
                engagement_count=2,
                positive_signal_count=2,
                exposure_count=1,
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
                    engagement_score=0.95,
                    evidence_count=2,
                    engagement_count=3,
                    positive_signal_count=3,
                    exposure_count=1,
                    explicit=True,
                    source_event_ids=(tool_call.call_id,),
                    updated_at=updated_at,
                )
            )
        return tuple(signals)
