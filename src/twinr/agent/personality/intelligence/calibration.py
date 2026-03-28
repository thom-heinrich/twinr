# CHANGELOG: 2026-03-27
# BUG-1: Hardened numeric/text/tool-argument coercion so malformed runtime evidence no longer crashes extraction.
# BUG-2: Fixed configure_world_intelligence topic parsing so a single string topic is preserved instead of silently collapsing to a generic fallback.
# BUG-3: Same-topic signals are now fused within a batch to prevent duplicate calibration inflation and incorrect downstream weighting.
# BUG-4: Invalid or naive timestamps no longer keep stale interests artificially eligible for follow-through cooling.
# SEC-1: Added caps for emitted signals, tool topics, source-event ids, and scanned runtime batches to resist signal-flood / memory-pressure DoS on Raspberry Pi-class deployments.
# SEC-2: Rejected arbitrary object stringification at trust boundaries to avoid expensive repr/str expansion from untrusted tool payloads.
# IMP-1: Added freshness-aware follow-through weighting aligned with 2025-2026 temporal-interest and long-term memory-update practice.
# IMP-2: Added deterministic semantic deduplication, stable ordering, and per-signal region override support for higher-quality world-intelligence calibration.

"""Extract world-intelligence calibration signals from structured runtime evidence.

This module keeps source-calibration learning separate from the generic
personality signal taxonomy. It turns existing structured personality evidence
and explicit tool usage into durable topic/region interest signals that the
RSS world-intelligence service can use during reflection-phase recalibration.

Each interest signal also carries a bounded ``engagement_score``. That score is
not a guess about feelings; it is Twinr's generic estimate of how much a topic
is currently pulling the user back in via repeated conversation evidence,
follow-up search behavior, or explicit world-intelligence configuration.

The 2026 upgrade in this file focuses on three edge-friendly properties:

1. Boundary hardening. Runtime payloads are coerced conservatively so malformed
   or oversized data no longer crashes or floods the calibrator.
2. Semantic fusion. Same-topic evidence inside one batch is merged before
   emission so downstream recalibration sees one bounded signal per topic/region
   and valence family instead of a stack of duplicates.
3. Freshness-aware cooling. Follow-through cooling now decays smoothly with
   interest freshness instead of relying on a hard cutoff only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import isfinite

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

_KIND_PRIORITY = {
    "avoid": 5,
    "tool-config": 4,
    "engagement": 3,
    "tool-search": 2,
    "cooling": 2,
    "follow-through-cooling": 2,
    "conversation": 1,
    "thread": 0,
}
_FAMILY_PRIORITY = {"avoid": 3, "positive": 2, "cooling": 1}
_COUNTER_MAX = 64


def _utcnow() -> datetime:
    """Return the current UTC time."""

    return datetime.now(timezone.utc)


def _ensure_utc(value: datetime) -> datetime:
    """Return one aware UTC datetime.

    Naive datetimes are treated as UTC to keep calibration deterministic across
    hosts. This is safer than relying on the process-local timezone.
    """

    # BREAKING: naive datetimes are normalized as UTC instead of inheriting the
    # host-local timezone implicitly.
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _clamp(value: float, *, minimum: float, maximum: float) -> float:
    """Clamp one numeric value onto an inclusive range."""

    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


def _clean_text(value: object | None, *, limit: int = 120) -> str:
    """Normalize scalar-ish input into bounded single-line text.

    Only cheap scalar values are accepted. Arbitrary containers and custom
    objects are rejected at the trust boundary to avoid expensive repr/str
    expansion from untrusted runtime payloads.
    """

    if value is None:
        return ""

    if isinstance(value, str):
        text = value[: max(limit * 4, limit)]
    elif isinstance(value, bytes):
        text = value[: max(limit * 4, limit)].decode("utf-8", errors="replace")
    elif isinstance(value, (int, bool)):
        text = str(value)
    elif isinstance(value, float):
        if not isfinite(value):
            return ""
        text = str(value)
    else:
        return ""

    text = " ".join(text.split())
    if not text:
        return ""
    return truncate_text(text, limit=limit)


def _clean_bool(value: object | None, *, default: bool = False) -> bool:
    """Coerce one optional loose boolean."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = _clean_text(value, limit=16).casefold()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _coerce_int(
    value: object | None,
    *,
    default: int = 0,
    minimum: int = 0,
    maximum: int = _COUNTER_MAX,
) -> int:
    """Coerce one loose integer onto a bounded range."""

    candidate = default
    if isinstance(value, bool):
        candidate = int(value)
    elif isinstance(value, int):
        candidate = value
    elif isinstance(value, float):
        if isfinite(value):
            candidate = int(value)
    elif isinstance(value, str):
        text = value.strip()[:32]
        if text:
            try:
                candidate = int(text)
            except ValueError:
                try:
                    candidate = int(float(text))
                except ValueError:
                    candidate = default
    return max(minimum, min(maximum, candidate))


def _coerce_float(
    value: object | None,
    *,
    default: float = 0.0,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    """Coerce one loose float onto a bounded range."""

    candidate = default
    if isinstance(value, bool):
        candidate = float(int(value))
    elif isinstance(value, (int, float)):
        if isfinite(float(value)):
            candidate = float(value)
    elif isinstance(value, str):
        text = value.strip()[:32]
        if text:
            try:
                parsed = float(text)
            except ValueError:
                parsed = None
            if parsed is not None and isfinite(parsed):
                candidate = parsed
    return _clamp(candidate, minimum=minimum, maximum=maximum)


def _as_mapping(value: object | None) -> Mapping[object, object]:
    """Return one mapping-like object or an empty mapping."""

    return value if isinstance(value, Mapping) else {}


def _limited_sequence_items(values: Sequence[object], *, limit: int) -> tuple[object, ...]:
    """Copy at most ``limit`` items from one sequence-like value."""

    if limit <= 0:
        return ()
    items: list[object] = []
    for index, item in enumerate(values):
        if index >= limit:
            break
        items.append(item)
    return tuple(items)


def _normalize_source_event_ids(values: object, *, limit: int) -> tuple[str, ...]:
    """Normalize one optional sequence of source-event ids."""

    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return ()
    normalized: list[str] = []
    seen: set[str] = set()
    for item in _limited_sequence_items(values, limit=max(0, limit * 2)):
        text = _clean_text(item, limit=96)
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
        if len(normalized) >= limit:
            break
    return tuple(normalized)


def _merge_source_event_ids(*groups: Sequence[str], limit: int) -> tuple[str, ...]:
    """Merge several source-event id groups into one bounded stable tuple."""

    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            text = _clean_text(item, limit=96)
            if not text or text in seen:
                continue
            seen.add(text)
            merged.append(text)
            if len(merged) >= limit:
                return tuple(merged)
    return tuple(merged)


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
    return _ensure_utc(parsed)


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


def _slug_component(
    value: object | None,
    *,
    fallback: str,
    text_limit: int = 96,
    slug_limit: int = 72,
) -> str:
    """Build one bounded slug from free-form input."""

    text = _clean_text(value, limit=text_limit)
    slug = slugify_identifier(text, fallback=fallback)
    if len(slug) > slug_limit:
        slug = slug[:slug_limit].rstrip("-_")
    return slug or fallback


@dataclass(frozen=True, slots=True)
class WorldIntelligenceSignalBatch:
    """Collect calibration signals emitted for the RSS intelligence service."""

    interest_signals: tuple[WorldInterestSignal, ...] = ()

    def has_any(self) -> bool:
        """Return whether the batch contains any calibration evidence."""

        return bool(self.interest_signals)


@dataclass(frozen=True, slots=True)
class _DraftInterestSignal:
    """Internal mutable-free draft used before semantic fusion."""

    family: str
    kind: str
    signal_id: str
    topic: str
    summary: str
    region: str | None
    scope: str
    salience: float
    confidence: float
    engagement_score: float
    evidence_count: int
    engagement_count: int
    positive_signal_count: int
    exposure_count: int
    non_reengagement_count: int = 0
    deflection_count: int = 0
    explicit: bool = False
    source_event_ids: tuple[str, ...] = ()
    updated_at: str = ""


@dataclass(frozen=True, slots=True)
class _CurrentTopicPull:
    """Normalized representation of current-turn topical pull."""

    topic: str
    region: str | None
    scope: str
    strength: float


@dataclass(slots=True)
class WorldInterestSignalExtractor:
    """Build slow-changing world/place interest signals from runtime evidence."""

    now_provider: Callable[[], datetime] = _utcnow
    follow_through_window_days: int = 21
    follow_through_candidate_limit: int = 2
    freshness_half_life_days: float = 7.0
    # BREAKING: Oversized runtime batches are now capped intentionally to keep
    # Pi-class deployments safe from signal-flooding and memory pressure.
    max_signals_per_batch: int = 24
    max_interaction_signals_per_batch: int = 64
    max_continuity_threads_per_batch: int = 12
    max_tool_calls_per_batch: int = 64
    max_topics_per_tool_call: int = 8
    max_source_event_ids: int = 6

    def extract_from_personality_batch(
        self,
        *,
        turn_id: str,
        batch: PersonalitySignalBatch,
        occurred_at: datetime,
        existing_interest_signals: Sequence[WorldInterestSignal] = (),
    ) -> WorldIntelligenceSignalBatch:
        """Derive calibration signals from already-structured conversation evidence."""

        occurred_at_utc = _ensure_utc(occurred_at)
        occurred_at_iso = occurred_at_utc.isoformat()

        strongest_place = max(
            batch.place_signals,
            key=lambda item: _coerce_float(getattr(item, "salience", 0.0), minimum=0.0, maximum=1.0),
            default=None,
        )
        strongest_region = (
            _clean_text(getattr(strongest_place, "place_name", None), limit=80) or None
            if strongest_place is not None
            else None
        )

        drafts: list[_DraftInterestSignal] = []
        turn_slug = _slug_component(turn_id, fallback="turn", text_limit=64, slug_limit=48)

        for interaction_signal in _limited_sequence_items(
            batch.interaction_signals,
            limit=self.max_interaction_signals_per_batch,
        ):
            kind = _clean_text(getattr(interaction_signal, "signal_kind", None), limit=48)
            if kind not in {
                INTERACTION_SIGNAL_TOPIC_AFFINITY,
                INTERACTION_SIGNAL_TOPIC_AVERSION,
                INTERACTION_SIGNAL_TOPIC_COOLING,
                INTERACTION_SIGNAL_TOPIC_ENGAGEMENT,
            }:
                continue

            topic = _clean_text(getattr(interaction_signal, "target", None), limit=96)
            if not topic:
                continue

            metadata = _as_mapping(getattr(interaction_signal, "metadata", None))
            region = self._interaction_region(
                interaction_signal=interaction_signal,
                metadata=metadata,
                strongest_region=strongest_region,
            )
            scope = _scope_from_region(region)

            confidence = _coerce_float(
                getattr(interaction_signal, "confidence", 0.0),
                default=0.0,
                minimum=0.0,
                maximum=1.0,
            )
            evidence_count = _coerce_int(
                getattr(interaction_signal, "evidence_count", 1),
                default=1,
                minimum=1,
                maximum=_COUNTER_MAX,
            )
            explicit = _clean_bool(getattr(interaction_signal, "explicit_user_requested", False))

            is_engagement = kind == INTERACTION_SIGNAL_TOPIC_ENGAGEMENT
            is_cooling = kind == INTERACTION_SIGNAL_TOPIC_COOLING
            is_avoid = kind == INTERACTION_SIGNAL_TOPIC_AVERSION

            engagement_base = 0.68 if explicit else 0.58
            if not is_engagement:
                engagement_base = 0.42
            if is_cooling:
                engagement_base = 0.2 if explicit else 0.28
            if is_avoid:
                engagement_base = 0.08 if explicit else 0.14

            exposure_count = _coerce_int(
                metadata.get("exposure_count"),
                default=max(1, evidence_count),
                minimum=0,
                maximum=_COUNTER_MAX,
            )
            non_reengagement_count = _coerce_int(
                metadata.get("non_reengagement_count"),
                default=0,
                minimum=0,
                maximum=_COUNTER_MAX,
            )
            deflection_count = _coerce_int(
                metadata.get("deflection_count"),
                default=0,
                minimum=0,
                maximum=_COUNTER_MAX,
            )

            if is_avoid:
                exposure_count = max(exposure_count, 2)
                non_reengagement_count = max(non_reengagement_count, 1)
                deflection_count = max(deflection_count, 2 if explicit else 1)

            topic_slug = _slug_component(topic, fallback="topic")
            region_slug = _slug_component(region or "global", fallback="region")

            drafts.append(
                _DraftInterestSignal(
                    family="avoid"
                    if is_avoid
                    else ("cooling" if is_cooling else "positive"),
                    kind="avoid"
                    if is_avoid
                    else ("cooling" if is_cooling else ("engagement" if is_engagement else "conversation")),
                    signal_id=(
                        f"interest:avoid:{turn_slug}:{region_slug}:{topic_slug}"
                        if is_avoid
                        else (
                            f"interest:cooling:{turn_slug}:{region_slug}:{topic_slug}"
                            if is_cooling
                            else (
                                f"interest:engagement:{turn_slug}:{region_slug}:{topic_slug}"
                                if is_engagement
                                else f"interest:conversation:{turn_slug}:{region_slug}:{topic_slug}"
                            )
                        )
                    ),
                    topic=topic,
                    summary=self._interaction_summary(
                        topic=topic,
                        region=region,
                        is_engagement=is_engagement,
                        is_cooling=is_cooling,
                        is_avoid=is_avoid,
                    ),
                    region=region,
                    scope=scope,
                    salience=_clamp(
                        confidence
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
                    confidence=confidence,
                    engagement_score=(
                        _engagement_score(
                            confidence=confidence,
                            evidence_count=evidence_count,
                            base=engagement_base,
                        )
                        if not is_cooling and not is_avoid
                        else _clamp(
                            engagement_base
                            + (confidence * (0.08 if is_cooling else 0.04))
                            - (non_reengagement_count * 0.08)
                            - (deflection_count * (0.12 if is_cooling else 0.16)),
                            minimum=0.01 if is_avoid else 0.05,
                            maximum=0.22 if is_avoid else 0.55,
                        )
                    ),
                    evidence_count=evidence_count,
                    engagement_count=max(
                        0 if is_cooling or is_avoid else (1 if not is_engagement else 2),
                        evidence_count
                        + (1 if is_engagement else 0)
                        - (1 if is_cooling else 0)
                        - (2 if is_avoid else 0),
                    ),
                    positive_signal_count=(
                        max(1, evidence_count + (1 if is_engagement else 0))
                        if not is_cooling and not is_avoid
                        else 0
                    ),
                    exposure_count=max(exposure_count, 1 if not is_cooling else 2),
                    non_reengagement_count=non_reengagement_count,
                    deflection_count=deflection_count,
                    explicit=explicit,
                    source_event_ids=_normalize_source_event_ids(
                        getattr(interaction_signal, "source_event_ids", ()),
                        limit=self.max_source_event_ids,
                    ),
                    updated_at=occurred_at_iso,
                )
            )

        for thread in _limited_sequence_items(
            batch.continuity_threads,
            limit=self.max_continuity_threads_per_batch,
        ):
            title = _clean_text(getattr(thread, "title", None), limit=96)
            if not title:
                continue
            thread_region = strongest_region
            thread_scope = _scope_from_region(thread_region)
            thread_salience = _coerce_float(
                getattr(thread, "salience", 0.0),
                default=0.0,
                minimum=0.0,
                maximum=1.0,
            )
            drafts.append(
                _DraftInterestSignal(
                    family="positive",
                    kind="thread",
                    signal_id=f"interest:thread:{_slug_component(title, fallback='thread')}",
                    topic=title,
                    summary=truncate_text(
                        f"Ongoing continuity around {title} suggests durable situational relevance.",
                        limit=180,
                    ),
                    region=thread_region,
                    scope=thread_scope,
                    salience=_clamp(thread_salience * 0.8, minimum=0.3, maximum=0.9),
                    confidence=thread_salience,
                    engagement_score=_engagement_score(
                        confidence=thread_salience,
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

        drafts.extend(
            self._derive_follow_through_interest_signals(
                turn_id=turn_id,
                batch=batch,
                occurred_at=occurred_at_utc,
                strongest_region=strongest_region,
                existing_interest_signals=existing_interest_signals,
            )
        )

        return WorldIntelligenceSignalBatch(
            interest_signals=self._merge_and_materialize(drafts),
        )

    def _interaction_region(
        self,
        *,
        interaction_signal: object,
        metadata: Mapping[object, object],
        strongest_region: str | None,
    ) -> str | None:
        """Choose the best available region for one interaction signal."""

        metadata_region = (
            _clean_text(
                metadata.get("region")
                or metadata.get("place_name")
                or metadata.get("location_hint")
                or metadata.get("location"),
                limit=80,
            )
            or None
        )
        signal_region = (
            _clean_text(getattr(interaction_signal, "region", None), limit=80)
            or _clean_text(getattr(interaction_signal, "place_name", None), limit=80)
            or None
        )
        return signal_region or metadata_region or strongest_region

    def _interaction_summary(
        self,
        *,
        topic: str,
        region: str | None,
        is_engagement: bool,
        is_cooling: bool,
        is_avoid: bool,
    ) -> str:
        """Build a readable interaction-derived summary."""

        return truncate_text(
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
        )

    def _merge_and_materialize(
        self,
        drafts: Sequence[_DraftInterestSignal],
    ) -> tuple[WorldInterestSignal, ...]:
        """Fuse same-topic drafts and materialize final signals."""

        grouped: dict[tuple[str, str, str, str], list[_DraftInterestSignal]] = {}
        for draft in drafts:
            key = (*_topic_key(topic=draft.topic, region=draft.region, scope=draft.scope), draft.family)
            grouped.setdefault(key, []).append(draft)

        merged: list[WorldInterestSignal] = []
        for group in grouped.values():
            dominant = max(
                group,
                key=lambda item: (
                    item.explicit,
                    _KIND_PRIORITY.get(item.kind, 0),
                    item.salience,
                    item.confidence,
                    item.engagement_score,
                ),
            )
            contribution_count = len(group)
            if dominant.family == "positive":
                engagement_minimum, engagement_maximum = 0.3, 0.98
                salience_minimum, salience_maximum = 0.3, 0.98
            elif dominant.family == "cooling":
                engagement_minimum, engagement_maximum = 0.05, 0.55
                salience_minimum, salience_maximum = 0.18, 0.72
            else:
                engagement_minimum, engagement_maximum = 0.01, 0.22
                salience_minimum, salience_maximum = 0.12, 0.48

            merge_bonus = min(0.12, 0.03 * max(0, contribution_count - 1))
            salience_values = [item.salience for item in group]
            confidence_values = [item.confidence for item in group]
            engagement_values = [item.engagement_score for item in group]

            if dominant.family == "positive":
                salience = _clamp(
                    max(salience_values) + merge_bonus,
                    minimum=salience_minimum,
                    maximum=salience_maximum,
                )
                engagement_score = _clamp(
                    max(engagement_values) + merge_bonus,
                    minimum=engagement_minimum,
                    maximum=engagement_maximum,
                )
            else:
                salience = _clamp(
                    max(salience_values) + min(0.08, merge_bonus),
                    minimum=salience_minimum,
                    maximum=salience_maximum,
                )
                engagement_score = _clamp(
                    min(engagement_values) - min(0.08, merge_bonus),
                    minimum=engagement_minimum,
                    maximum=engagement_maximum,
                )

            confidence = _clamp(
                max(confidence_values) + min(0.08, merge_bonus * 0.66),
                minimum=0.0,
                maximum=1.0,
            )

            summary = dominant.summary
            if contribution_count > 1:
                summary = truncate_text(
                    f"{summary.rstrip('.')}. Multiple signals in this batch reinforced the calibration.",
                    limit=180,
                )

            updated_at_candidates = [
                parsed
                for parsed in (_parse_iso(item.updated_at) for item in group)
                if parsed is not None
            ]
            updated_at = (
                max(updated_at_candidates).isoformat()
                if updated_at_candidates
                else dominant.updated_at
            )

            merged.append(
                WorldInterestSignal(
                    # BREAKING: Same-topic same-valence signals are now fused inside one
                    # batch, so downstream consumers should not expect one output signal
                    # per raw event anymore.
                    signal_id=dominant.signal_id,
                    topic=dominant.topic,
                    summary=summary,
                    region=dominant.region,
                    scope=dominant.scope,
                    salience=salience,
                    confidence=confidence,
                    engagement_score=engagement_score,
                    evidence_count=min(_COUNTER_MAX, sum(item.evidence_count for item in group)),
                    engagement_count=min(_COUNTER_MAX, sum(item.engagement_count for item in group)),
                    positive_signal_count=min(_COUNTER_MAX, sum(item.positive_signal_count for item in group)),
                    exposure_count=min(_COUNTER_MAX, sum(item.exposure_count for item in group)),
                    non_reengagement_count=min(_COUNTER_MAX, sum(item.non_reengagement_count for item in group)),
                    deflection_count=min(_COUNTER_MAX, sum(item.deflection_count for item in group)),
                    explicit=any(item.explicit for item in group),
                    source_event_ids=_merge_source_event_ids(
                        *(item.source_event_ids for item in group),
                        limit=self.max_source_event_ids,
                    ),
                    updated_at=updated_at,
                )
            )

        merged.sort(
            key=lambda item: (
                _FAMILY_PRIORITY.get(self._signal_family(item), 0),
                item.explicit,
                item.salience,
                item.confidence,
                item.engagement_score,
            ),
            reverse=True,
        )
        return tuple(merged[: max(0, self.max_signals_per_batch)])

    def _signal_family(self, signal: WorldInterestSignal) -> str:
        """Infer one coarse family for sort ordering."""

        signal_id = _clean_text(signal.signal_id, limit=128)
        if signal_id.startswith("interest:avoid:"):
            return "avoid"
        if signal_id.startswith("interest:cooling:") or signal_id.startswith("interest:follow-through:"):
            return "cooling"
        return "positive"

    def _derive_follow_through_interest_signals(
        self,
        *,
        turn_id: str,
        batch: PersonalitySignalBatch,
        occurred_at: datetime,
        strongest_region: str | None,
        existing_interest_signals: Sequence[WorldInterestSignal],
    ) -> tuple[_DraftInterestSignal, ...]:
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

        strongest_current = max(item.strength for item in current_topics.values())
        if strongest_current < 0.55:
            return ()

        occurred_at_utc = _ensure_utc(occurred_at)
        occurred_at_iso = occurred_at_utc.isoformat()
        fresh_cutoff = occurred_at_utc - timedelta(days=max(1, self.follow_through_window_days))
        turn_slug = _slug_component(turn_id, fallback="turn", text_limit=64, slug_limit=48)

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
            key=lambda signal: self._existing_interest_priority(
                signal=signal,
                reference_time=occurred_at_utc,
            ),
            reverse=True,
        )

        cooling_signals: list[_DraftInterestSignal] = []
        current_label = self._dominant_current_topic_label(current_topics)
        for signal in ranked_candidates[: max(0, self.follow_through_candidate_limit)]:
            updated_at = _parse_iso(signal.updated_at)
            freshness = self._freshness_weight(
                updated_at=updated_at,
                reference_time=occurred_at_utc,
            )
            cooling_signals.append(
                _DraftInterestSignal(
                    family="cooling",
                    kind="follow-through-cooling",
                    signal_id=f"interest:follow-through:{turn_slug}:{_slug_component(signal.topic, fallback='topic')}",
                    topic=signal.topic,
                    summary=truncate_text(
                        (
                            f"This turn moved back toward {current_label} instead of reopening {signal.topic}."
                            if current_label
                            else f"This turn did not reopen {signal.topic}."
                        ),
                        limit=180,
                    ),
                    region=_clean_text(signal.region, limit=80) or None,
                    scope=_clean_text(signal.scope, limit=32) or "topic",
                    salience=_clamp(
                        signal.salience * (0.24 + (0.22 * freshness)),
                        minimum=0.18,
                        maximum=0.46,
                    ),
                    confidence=_clamp(
                        0.42 + (strongest_current * 0.16) + (freshness * 0.12),
                        minimum=0.42,
                        maximum=0.74,
                    ),
                    engagement_score=_clamp(
                        signal.engagement_score - (0.14 + (0.10 * freshness)),
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
    ) -> dict[tuple[str, str, str], _CurrentTopicPull]:
        """Rank the topics that genuinely pulled the current turn forward."""

        strengths: dict[tuple[str, str, str], _CurrentTopicPull] = {}

        for signal in _limited_sequence_items(
            batch.interaction_signals,
            limit=self.max_interaction_signals_per_batch,
        ):
            kind = _clean_text(getattr(signal, "signal_kind", None), limit=48)
            if kind not in {
                INTERACTION_SIGNAL_TOPIC_AFFINITY,
                INTERACTION_SIGNAL_TOPIC_ENGAGEMENT,
            }:
                continue

            topic = _clean_text(getattr(signal, "target", None), limit=96)
            if not topic:
                continue

            metadata = _as_mapping(getattr(signal, "metadata", None))
            region = self._interaction_region(
                interaction_signal=signal,
                metadata=metadata,
                strongest_region=strongest_region,
            )
            scope = _scope_from_region(region)
            key = _topic_key(
                topic=topic,
                region=region,
                scope=scope,
            )

            confidence = _coerce_float(
                getattr(signal, "confidence", 0.0),
                default=0.0,
                minimum=0.0,
                maximum=1.0,
            )
            evidence_count = _coerce_int(
                getattr(signal, "evidence_count", 1),
                default=1,
                minimum=1,
                maximum=8,
            )
            pull = _clamp(
                0.34
                + (confidence * 0.32)
                + (min(evidence_count, 3) * 0.04)
                + (0.14 if kind == INTERACTION_SIGNAL_TOPIC_ENGAGEMENT else 0.0)
                + (0.08 if _clean_bool(getattr(signal, "explicit_user_requested", False)) else 0.0),
                minimum=0.0,
                maximum=1.0,
            )

            previous = strengths.get(key)
            if previous is None or pull > previous.strength:
                strengths[key] = _CurrentTopicPull(
                    topic=topic,
                    region=region,
                    scope=scope,
                    strength=pull,
                )

        for thread in _limited_sequence_items(
            batch.continuity_threads,
            limit=self.max_continuity_threads_per_batch,
        ):
            title = _clean_text(getattr(thread, "title", None), limit=96)
            if not title:
                continue
            region = strongest_region
            scope = _scope_from_region(region)
            key = _topic_key(topic=title, region=region, scope=scope)
            pull = _clamp(
                0.42 + (_coerce_float(getattr(thread, "salience", 0.0), minimum=0.0, maximum=1.0) * 0.28),
                minimum=0.0,
                maximum=0.86,
            )
            previous = strengths.get(key)
            if previous is None or pull > previous.strength:
                strengths[key] = _CurrentTopicPull(
                    topic=title,
                    region=region,
                    scope=scope,
                    strength=pull,
                )

        return strengths

    def _is_follow_through_candidate(
        self,
        *,
        signal: WorldInterestSignal,
        current_topics: Mapping[tuple[str, str, str], _CurrentTopicPull],
        fresh_cutoff: datetime,
    ) -> bool:
        """Return whether one existing interest is eligible for mild cooling."""

        if getattr(signal, "engagement_state", None) not in {"resonant", "warm"}:
            return False
        if _topic_key(topic=signal.topic, region=signal.region, scope=signal.scope) in current_topics:
            return False

        updated_at = _parse_iso(signal.updated_at)
        if updated_at is None:
            return False
        if updated_at < fresh_cutoff:
            return False

        return signal.engagement_score >= 0.58 or signal.positive_signal_count >= 1

    def _freshness_weight(
        self,
        *,
        updated_at: datetime | None,
        reference_time: datetime,
    ) -> float:
        """Convert recency into one smooth [0, 1] freshness weight."""

        if updated_at is None:
            return 0.0
        age_seconds = max(0.0, (reference_time - updated_at).total_seconds())
        half_life_days = max(self.freshness_half_life_days, 0.25)
        age_days = age_seconds / 86400.0
        return _clamp(0.5 ** (age_days / half_life_days), minimum=0.0, maximum=1.0)

    def _existing_interest_priority(
        self,
        signal: WorldInterestSignal,
        *,
        reference_time: datetime,
    ) -> float:
        """Score one existing interest for bounded follow-through cooling."""

        state_bonus = 0.22 if signal.engagement_state == "resonant" else 0.12
        freshness = self._freshness_weight(
            updated_at=_parse_iso(signal.updated_at),
            reference_time=reference_time,
        )
        return (
            (signal.engagement_score * 0.56)
            + (signal.salience * 0.18)
            + (freshness * 0.14)
            + state_bonus
        )

    def _dominant_current_topic_label(
        self,
        current_topics: Mapping[tuple[str, str, str], _CurrentTopicPull],
    ) -> str | None:
        """Return a readable label for the strongest currently engaged topic."""

        if not current_topics:
            return None
        strongest = max(current_topics.values(), key=lambda item: item.strength)
        if strongest.region:
            return f"{strongest.topic} around {strongest.region}"
        return strongest.topic

    def extract_from_tool_history(
        self,
        *,
        tool_calls: Sequence[AgentToolCall],
        tool_results: Sequence[AgentToolResult],
    ) -> WorldIntelligenceSignalBatch:
        """Derive calibration signals from explicit search and RSS tool usage."""

        result_by_call_id = {
            _clean_text(getattr(result, "call_id", None), limit=96): result
            for result in tool_results
            if _clean_text(getattr(result, "call_id", None), limit=96)
        }
        now_iso = _ensure_utc(self.now_provider()).isoformat()
        drafts: list[_DraftInterestSignal] = []

        for tool_call in _limited_sequence_items(tool_calls, limit=self.max_tool_calls_per_batch):
            tool_name = _clean_text(getattr(tool_call, "name", None), limit=64)
            if tool_name not in _SUPPORTED_TOOL_SIGNAL_NAMES:
                continue

            tool_call_id = _clean_text(getattr(tool_call, "call_id", None), limit=96)
            if not tool_call_id:
                continue
            tool_result = result_by_call_id.get(tool_call_id)
            if tool_result is None:
                continue

            if tool_name == "search_live_info":
                drafts.extend(
                    self._from_live_search(
                        tool_call=tool_call,
                        tool_result=tool_result,
                        updated_at=now_iso,
                    )
                )
                continue

            if tool_name == "configure_world_intelligence":
                drafts.extend(
                    self._from_world_intelligence_tool(
                        tool_call=tool_call,
                        tool_result=tool_result,
                        updated_at=now_iso,
                    )
                )

        return WorldIntelligenceSignalBatch(
            interest_signals=self._merge_and_materialize(drafts),
        )

    def _status_is_ok(self, output: Mapping[object, object]) -> bool:
        """Interpret one loose tool result status."""

        status = _clean_text(output.get("status"), limit=32).casefold()
        if status in {"ok", "success", "succeeded", "done"}:
            return True
        if _clean_bool(output.get("ok")):
            return True
        if _clean_bool(output.get("success")):
            return True
        return False

    def _tool_arguments(self, tool_call: AgentToolCall) -> Mapping[object, object]:
        """Return safe mapping-like tool arguments."""

        return _as_mapping(getattr(tool_call, "arguments", None))

    def _from_live_search(
        self,
        *,
        tool_call: AgentToolCall,
        tool_result: AgentToolResult,
        updated_at: str,
    ) -> tuple[_DraftInterestSignal, ...]:
        """Interpret successful live-search usage as topical situational interest."""

        output = _as_mapping(getattr(tool_result, "output", None))
        if not output or not self._status_is_ok(output):
            return ()

        arguments = self._tool_arguments(tool_call)
        question = _clean_text(arguments.get("question") or arguments.get("query"), limit=96)
        if not question:
            return ()

        region = _clean_text(arguments.get("location_hint") or arguments.get("region"), limit=80) or None
        explicit = _clean_bool(
            arguments.get("explicit_user_requested") or arguments.get("user_initiated"),
            default=False,
        )
        return (
            _DraftInterestSignal(
                family="positive",
                kind="tool-search",
                signal_id=f"interest:tool:{_slug_component(getattr(tool_call, 'call_id', None), fallback='call')}",
                topic=question,
                summary=truncate_text(
                    "Recent live search suggests the user wanted fresh situational awareness."
                    if region is None
                    else f"Recent live search suggests the user wanted fresh situational awareness about {region}.",
                    limit=180,
                ),
                region=region,
                scope=_scope_from_region(region),
                salience=0.64 if region is None else 0.74,
                confidence=0.78 if explicit else 0.7,
                engagement_score=(0.86 if region is not None else 0.8) if explicit else (0.82 if region is not None else 0.76),
                evidence_count=1,
                engagement_count=2,
                positive_signal_count=2,
                exposure_count=1,
                explicit=explicit,
                source_event_ids=_merge_source_event_ids(
                    (_clean_text(getattr(tool_call, "call_id", None), limit=96),),
                    (_clean_text(output.get("response_id"), limit=96),),
                    limit=self.max_source_event_ids,
                ),
                updated_at=updated_at,
            ),
        )

    def _normalize_tool_topics(self, arguments: Mapping[object, object]) -> tuple[str, ...]:
        """Normalize tool topic arguments from either list-like or scalar forms."""

        raw_topics = arguments.get("topics")
        if raw_topics is None:
            raw_topics = arguments.get("topic")

        values: list[str] = []

        if isinstance(raw_topics, str):
            topic = _clean_text(raw_topics, limit=96)
            if topic:
                values.append(topic)
        elif isinstance(raw_topics, Sequence) and not isinstance(raw_topics, (bytes, bytearray)):
            for item in _limited_sequence_items(raw_topics, limit=max(0, self.max_topics_per_tool_call * 2)):
                topic = _clean_text(item, limit=96)
                if topic:
                    values.append(topic)

        normalized: list[str] = []
        seen: set[str] = set()
        for topic in values:
            folded = topic.casefold()
            if folded in seen:
                continue
            seen.add(folded)
            normalized.append(topic)
            if len(normalized) >= self.max_topics_per_tool_call:
                break
        return tuple(normalized)

    def _from_world_intelligence_tool(
        self,
        *,
        tool_call: AgentToolCall,
        tool_result: AgentToolResult,
        updated_at: str,
    ) -> tuple[_DraftInterestSignal, ...]:
        """Interpret explicit RSS configuration as strong calibration evidence."""

        output = _as_mapping(getattr(tool_result, "output", None))
        if not output or not self._status_is_ok(output):
            return ()

        arguments = self._tool_arguments(tool_call)
        action = _clean_text(arguments.get("action"), limit=32).casefold()
        if action not in {"subscribe", "discover", "refresh_now"}:
            return ()

        normalized_topics = self._normalize_tool_topics(arguments)
        label = _clean_text(arguments.get("label"), limit=96)
        region = _clean_text(arguments.get("region") or arguments.get("location_hint"), limit=80) or None
        fallback_topic = label or "configured world awareness"
        signal_topics = normalized_topics or (fallback_topic,)

        signals: list[_DraftInterestSignal] = []
        for topic in signal_topics:
            signals.append(
                _DraftInterestSignal(
                    family="positive",
                    kind="tool-config",
                    signal_id=(
                        f"interest:tool-config:{_slug_component(getattr(tool_call, 'call_id', None), fallback='call')}"
                        f":{_slug_component(topic, fallback='topic')}"
                    ),
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
                    source_event_ids=(
                        _clean_text(getattr(tool_call, "call_id", None), limit=96),
                    ),
                    updated_at=updated_at,
                )
            )
        return tuple(signals)