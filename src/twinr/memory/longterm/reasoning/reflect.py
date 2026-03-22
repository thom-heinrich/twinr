"""Reflect over recent long-term memory state.

This module promotes repeated candidate facts, creates bounded person-thread
summaries, and optionally compiles midterm packets from the most recent
memory window without blocking the main memory pipeline.
"""

from __future__ import annotations

from collections.abc import Iterable  # AUDIT-FIX(#5): Robustly coerce list-like payload fields without exploding strings into character-tuples.
from dataclasses import dataclass
from datetime import date, datetime, timezone  # AUDIT-FIX(#4): Normalize naive/aware datetimes before sorting and min/max operations.
import json
import logging  # AUDIT-FIX(#2): Log degraded optional-reflection paths instead of crashing silently.
import math  # AUDIT-FIX(#6): Safely reject non-finite numeric config values.
from typing import Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError  # AUDIT-FIX(#4): Validate and apply the configured timezone for naive datetimes.

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.longterm.reasoning.midterm import (
    LongTermStructuredReflectionProgram,
    structured_reflection_program_from_config,
)
from twinr.memory.longterm.core.ontology import kind_matches
from twinr.memory.longterm.core.models import (
    LongTermMemoryObjectV1,
    LongTermMidtermPacketV1,
    LongTermReflectionResultV1,
)
from twinr.text_utils import collapse_whitespace, slugify_identifier

logger = logging.getLogger(__name__)  # AUDIT-FIX(#2): Preserve operational visibility when optional enrichment degrades.

_DEFAULT_TIMEZONE_NAME = "Europe/Berlin"  # AUDIT-FIX(#6): Keep a known-good fallback timezone for invalid/missing config.
_ALLOWED_ACTIVE_STATUSES = frozenset({"active", "candidate", "uncertain"})
_THREAD_SUMMARY_EVIDENCE_KINDS = frozenset({"event", "plan"})  # AUDIT-FIX(#1): Only evidence-bearing kinds may support thread summaries.
_SMART_HOME_ENVIRONMENT_DOMAIN = "smart_home_environment"
_ENVIRONMENT_DAY_PROFILE_SUMMARY = "environment_day_profile"
_ENVIRONMENT_DEVIATION_SUMMARY = "environment_deviation"
_ENVIRONMENT_NODE_SUMMARY = "environment_node"
_ENVIRONMENT_REFLECTION_SUMMARY = "environment_reflection"
_ENVIRONMENT_BASELINE_PATTERN = "environment_baseline"
_ENVIRONMENT_REFLECTION_PACKET_KIND = "recent_environment_pattern"
_ENVIRONMENT_MARKER_KEYS = (
    "active_epoch_count_day",
    "first_activity_minute_local",
    "last_activity_minute_local",
    "longest_daytime_inactivity_min",
    "night_activity_epoch_count",
    "unique_active_node_count_day",
    "transition_count_day",
    "fragmentation_index_day",
    "circadian_similarity_14d",
    "sensor_coverage_ratio_day",
)
_SENSITIVITY_RANK = {
    "low": 0,
    "normal": 1,
    "private": 2,
    "sensitive": 3,
    "critical": 4,
}


def _normalize_text(value: str | None) -> str:
    # AUDIT-FIX(#3): Treat None as empty text so null fields do not become the literal string "None".
    if value is None:
        return ""
    return collapse_whitespace(value)


def _normalize_object_text(value: object) -> str:
    # AUDIT-FIX(#3): Normalize arbitrary payload values safely while preserving None as empty text.
    if value is None:
        return ""
    if isinstance(value, str):
        return _normalize_text(value)
    if isinstance(value, datetime):
        return _normalize_text(value.isoformat())
    if isinstance(value, (bytes, bytearray)):
        return _normalize_text(bytes(value).decode("utf-8", errors="replace"))
    return _normalize_text(str(value))


def _safe_json_text(value: object) -> str:
    # AUDIT-FIX(#5): Fall back cleanly when payload objects are not JSON-serializable or contain circular references.
    try:
        return _normalize_text(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))
    except (TypeError, ValueError, OverflowError, RecursionError):
        return _normalize_object_text(value)


def _slugify(value: str, *, fallback: str) -> str:
    return slugify_identifier(value, fallback=fallback)


def _coerce_bounded_int(value: object, *, default: int, minimum: int) -> int:
    # AUDIT-FIX(#6): Sanitize config/runtime integer-like values so negative limits and bad env values cannot cause odd slicing behavior.
    if isinstance(value, bool):
        return default
    try:
        if isinstance(value, float) and not math.isfinite(value):
            return default
        coerced = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(minimum, coerced)


def _coerce_support_count(value: object) -> int:
    # AUDIT-FIX(#8): Parse numeric strings and non-finite values safely instead of relying on fragile isinstance checks.
    if isinstance(value, bool):
        return int(value)
    return _coerce_bounded_int(value, default=1, minimum=0)


def _normalize_timezone_name(value: str | None) -> str:
    # AUDIT-FIX(#3): Validate the configured timezone early and fall back to a known-good zone on invalid config.
    candidate = _normalize_text(value) or _DEFAULT_TIMEZONE_NAME
    try:
        ZoneInfo(candidate)
    except ZoneInfoNotFoundError:
        logger.warning(
            "Invalid long-term memory timezone %r; falling back to %s.",
            candidate,
            _DEFAULT_TIMEZONE_NAME,
        )
        return _DEFAULT_TIMEZONE_NAME
    return candidate


def _normalize_sensitivity(value: object) -> str:
    # AUDIT-FIX(#2): Keep packet sensitivities within the supported enum so invalid model output cannot fail downstream validation.
    normalized = _normalize_object_text(value).lower()
    if normalized in _SENSITIVITY_RANK:
        return normalized
    return "normal"


def _coerce_text_sequence(value: object) -> tuple[str, ...]:
    # AUDIT-FIX(#2): Treat strings as scalar values and reject non-sequence mappings so malformed payloads cannot explode into char-tuples.
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)):
        single_value = _normalize_object_text(value)
        return (single_value,) if single_value else ()
    if isinstance(value, Mapping):
        return ()
    if not isinstance(value, Iterable):
        return ()

    items: list[str] = []
    seen: set[str] = set()
    for item in value:
        normalized = _safe_json_text(item) if isinstance(item, (Mapping, list, tuple, set, frozenset)) else _normalize_object_text(item)
        if normalized and normalized not in seen:
            seen.add(normalized)
            items.append(normalized)
    return tuple(items)


def _normalize_midterm_attributes(value: object) -> dict[str, str] | None:
    if not isinstance(value, Mapping):
        return None
    normalized: dict[str, str] = {}
    for key, raw_value in value.items():
        clean_key = _normalize_object_text(key)
        if not clean_key or raw_value is None:
            continue
        if isinstance(raw_value, str):
            clean_value = _normalize_text(raw_value)
        elif isinstance(raw_value, (list, tuple, set, frozenset)):
            # AUDIT-FIX(#2): Join normalized list-like attribute values without double-normalizing each element.
            clean_value = _normalize_text(", ".join(_coerce_text_sequence(raw_value)))
        elif isinstance(raw_value, (bool, int, float)):
            clean_value = _normalize_object_text(raw_value)
        else:
            clean_value = _safe_json_text(raw_value)
        if clean_value:
            normalized[clean_key] = clean_value
    return normalized or None


def _coerce_mapping(value: object) -> Mapping[str, object]:
    """Return a mapping payload or an empty mapping."""

    if isinstance(value, Mapping):
        return value
    return {}


def _parse_iso_date(value: object) -> date | None:
    """Parse one date-like value into a calendar date."""

    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    text = _normalize_object_text(value)
    if not text:
        return None
    candidate = text[:10]
    try:
        return date.fromisoformat(candidate)
    except ValueError:
        return None


def _coerce_float(value: object) -> float | None:
    """Coerce one numeric value into a finite float."""

    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _unique_texts(*values: object, limit: int = 12) -> tuple[str, ...]:
    """Return bounded unique normalized text values."""

    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, (list, tuple, set, frozenset)):
            entries = value
        else:
            entries = (value,)
        for entry in entries:
            text = _normalize_object_text(entry)
            if not text or text in seen:
                continue
            seen.add(text)
            deduped.append(text)
            if len(deduped) >= limit:
                return tuple(deduped)
    return tuple(deduped)


def _minute_label(value: object) -> str:
    """Render one minute-of-day integer as ``HH:MM`` text."""

    minute = _coerce_bounded_int(value, default=-1, minimum=-1)
    if minute < 0:
        return ""
    hours, minutes = divmod(minute, 60)
    if hours < 0 or hours > 23 or minutes < 0 or minutes > 59:
        return ""
    return f"{hours:02d}:{minutes:02d}"


@dataclass(frozen=True, slots=True)
class LongTermMemoryReflector:
    """Coordinate post-consolidation reflection and optional midterm output.

    Attributes:
        min_support_count_for_promotion: Support count required before a
            repeated candidate or uncertain memory can auto-promote.
        min_support_count_for_thread_summary: Evidence count required before
            a person-thread summary can be created.
        program: Optional provider-backed midterm reflection compiler.
        midterm_packet_limit: Maximum number of midterm packets to accept.
        reflection_window_size: Maximum number of recent memories to include
            when compiling midterm packets.
        timezone_name: Local timezone used to normalize naive datetimes.
    """

    min_support_count_for_promotion: int = 2
    min_support_count_for_thread_summary: int = 2
    program: LongTermStructuredReflectionProgram | None = None
    midterm_packet_limit: int = 4
    reflection_window_size: int = 18
    timezone_name: str = _DEFAULT_TIMEZONE_NAME

    def __post_init__(self) -> None:
        # AUDIT-FIX(#3): Sanitize public constructor inputs too, not just from_config(), because this dataclass can be instantiated directly.
        object.__setattr__(
            self,
            "min_support_count_for_promotion",
            _coerce_bounded_int(self.min_support_count_for_promotion, default=2, minimum=1),
        )
        object.__setattr__(
            self,
            "min_support_count_for_thread_summary",
            _coerce_bounded_int(self.min_support_count_for_thread_summary, default=2, minimum=1),
        )
        object.__setattr__(
            self,
            "midterm_packet_limit",
            _coerce_bounded_int(self.midterm_packet_limit, default=4, minimum=0),
        )
        object.__setattr__(
            self,
            "reflection_window_size",
            _coerce_bounded_int(self.reflection_window_size, default=18, minimum=4),
        )
        object.__setattr__(self, "timezone_name", _normalize_timezone_name(self.timezone_name))

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "LongTermMemoryReflector":
        """Build a reflector from Twinr configuration.

        Args:
            config: Runtime configuration containing reflection thresholds,
                timezone settings, and optional midterm compiler settings.

        Returns:
            A configured reflector with bounded numeric settings and an
            optional midterm program when that feature initializes cleanly.
        """

        program: LongTermStructuredReflectionProgram | None = None
        if config.long_term_memory_midterm_enabled:
            try:
                # AUDIT-FIX(#3): Optional midterm reflection setup must not take the whole agent down on config/program init errors.
                program = structured_reflection_program_from_config(config)
            except Exception:
                logger.exception(
                    "Failed to initialize long-term memory midterm reflection program; disabling optional midterm reflection."
                )
                program = None
        return cls(
            program=program,
            midterm_packet_limit=_coerce_bounded_int(
                getattr(config, "long_term_memory_midterm_limit", 4),
                default=4,
                minimum=0,
            ),
            reflection_window_size=_coerce_bounded_int(
                getattr(config, "long_term_memory_reflection_window_size", 18),
                default=18,
                minimum=4,
            ),
            timezone_name=_normalize_timezone_name(getattr(config, "local_timezone_name", _DEFAULT_TIMEZONE_NAME)),
        )

    def reflect(
        self,
        *,
        objects: tuple[LongTermMemoryObjectV1, ...],
        include_midterm: bool = True,
    ) -> LongTermReflectionResultV1:
        """Reflect over the current long-term memory snapshot.

        Args:
            objects: Current long-term memory objects to canonicalize and
                inspect for promotions, summaries, and midterm compilation.
            include_midterm: Whether to run the optional midterm compiler after
                deterministic promotions and thread-summary synthesis.

        Returns:
            A reflection result containing promoted objects, newly created
            summaries, and any accepted midterm packets.
        """

        reflected: list[LongTermMemoryObjectV1] = []
        created_summaries: list[LongTermMemoryObjectV1] = []
        midterm_packets: list[LongTermMidtermPacketV1] = []

        canonical_objects: list[LongTermMemoryObjectV1] = []
        for item in objects:
            try:
                # AUDIT-FIX(#7): Corrupted file-backed memory records should be skipped, not allowed to abort the whole reflection batch.
                canonical_objects.append(item.canonicalized())
            except Exception:
                logger.exception("Skipping malformed long-term memory object during reflection.")
        normalized_objects = tuple(canonical_objects)

        for item in normalized_objects:
            support_count = self._support_count(item)
            if item.status in {"candidate", "uncertain"} and support_count >= self.min_support_count_for_promotion:
                try:
                    reflected.append(
                        item.with_updates(
                            status="active",
                            confidence=max(item.confidence, min(0.99, 0.55 + 0.12 * support_count)),
                        )
                    )
                except Exception:
                    logger.exception(
                        "Failed to promote long-term memory object %s during reflection.",
                        getattr(item, "memory_id", "<unknown>"),
                    )

        by_person: dict[str, list[LongTermMemoryObjectV1]] = {}
        for item in normalized_objects:
            person_ref = self._person_ref(item)
            if not person_ref:
                continue
            by_person.setdefault(person_ref, []).append(item)

        for person_ref, person_items in by_person.items():
            try:
                summary = self._build_thread_summary(person_ref=person_ref, items=tuple(person_items))
            except Exception:
                logger.exception(
                    "Failed to build thread summary for person_ref=%r; continuing without that summary.",
                    person_ref,
                )
                continue
            if summary is not None:
                created_summaries.append(summary)

        try:
            environment_summary, environment_packets = self._build_environment_reflection(objects=normalized_objects)
        except Exception:
            logger.exception("Failed to build smart-home environment reflection; continuing without that summary.")
            environment_summary = None
            environment_packets = ()
        if environment_summary is not None:
            created_summaries.append(environment_summary)
        if include_midterm and self.midterm_packet_limit > 0 and environment_packets:
            midterm_packets.extend(environment_packets[: self.midterm_packet_limit])

        if include_midterm and self.program is not None and self.midterm_packet_limit > 0:
            try:
                # AUDIT-FIX(#5): Optional midterm enrichment must degrade gracefully instead of crashing the caller on external/program errors.
                payload = self.program.compile_reflection(
                    objects=self._reflection_window(normalized_objects),
                    timezone_name=self.timezone_name,
                    packet_limit=max(0, self.midterm_packet_limit - len(midterm_packets)),
                )
            except Exception:
                logger.exception("Midterm reflection compilation failed; continuing without midterm packets.")
            else:
                midterm_packets.extend(self._midterm_packets_from_payload(payload))

        return LongTermReflectionResultV1(
            reflected_objects=tuple(reflected),
            created_summaries=tuple(created_summaries),
            midterm_packets=tuple(midterm_packets),
        )

    def _attributes_mapping(self, item: LongTermMemoryObjectV1) -> Mapping[str, object]:
        # AUDIT-FIX(#7): Defensively treat non-mapping attributes as empty so one malformed record does not poison the batch.
        attributes = getattr(item, "attributes", None)
        if isinstance(attributes, Mapping):
            return attributes
        return {}

    def _configured_timezone(self) -> ZoneInfo:
        try:
            return ZoneInfo(self.timezone_name)
        except ZoneInfoNotFoundError:
            return ZoneInfo(_DEFAULT_TIMEZONE_NAME)

    def _normalized_datetime(self, value: object) -> datetime | None:
        # AUDIT-FIX(#6): Convert naive datetimes to the configured local zone and compare/store them as aware UTC values.
        if not isinstance(value, datetime):
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            value = value.replace(tzinfo=self._configured_timezone())
        return value.astimezone(timezone.utc)

    def _latest_datetime(self, values: Iterable[object]) -> datetime | None:
        latest: datetime | None = None
        for value in values:
            normalized = self._normalized_datetime(value)
            if normalized is None:
                continue
            if latest is None or normalized > latest:
                latest = normalized
        return latest

    def _earliest_datetime(self, values: Iterable[object]) -> datetime | None:
        earliest: datetime | None = None
        for value in values:
            normalized = self._normalized_datetime(value)
            if normalized is None:
                continue
            if earliest is None or normalized < earliest:
                earliest = normalized
        return earliest

    def _item_updated_at(self, item: LongTermMemoryObjectV1) -> datetime:
        updated_at = self._normalized_datetime(getattr(item, "updated_at", None))
        if updated_at is None:
            return datetime.min.replace(tzinfo=timezone.utc)
        return updated_at

    def _support_count(self, item: LongTermMemoryObjectV1) -> int:
        raw = self._attributes_mapping(item).get("support_count")
        return _coerce_support_count(raw)

    def _environment_domain(self, item: LongTermMemoryObjectV1) -> str:
        """Return the normalized environment memory domain for one object."""

        return _normalize_object_text(self._attributes_mapping(item).get("memory_domain")).lower()

    def _environment_summary_type(self, item: LongTermMemoryObjectV1) -> str:
        """Return the normalized environment summary type for one object."""

        return _normalize_object_text(self._attributes_mapping(item).get("summary_type")).lower()

    def _environment_pattern_type(self, item: LongTermMemoryObjectV1) -> str:
        """Return the normalized environment pattern type for one object."""

        return _normalize_object_text(self._attributes_mapping(item).get("pattern_type")).lower()

    def _environment_id(self, item: LongTermMemoryObjectV1) -> str:
        """Return the stable environment identifier for one object."""

        return _normalize_object_text(self._attributes_mapping(item).get("environment_id")) or "home:main"

    def _environment_profile_day(self, item: LongTermMemoryObjectV1) -> date | None:
        """Return the profile day for one environment day-profile-like object."""

        attributes = self._attributes_mapping(item)
        return (
            _parse_iso_date(attributes.get("date"))
            or _parse_iso_date(attributes.get("profile_day"))
            or _parse_iso_date(getattr(item, "valid_from", None))
            or _parse_iso_date(getattr(item, "valid_to", None))
        )

    def _environment_deviation_day(self, item: LongTermMemoryObjectV1) -> date | None:
        """Return the observed day for one environment deviation object."""

        attributes = self._attributes_mapping(item)
        return (
            _parse_iso_date(attributes.get("observed_at"))
            or _parse_iso_date(getattr(item, "valid_from", None))
            or _parse_iso_date(getattr(item, "valid_to", None))
        )

    def _environment_marker_value(
        self,
        markers: Mapping[str, object],
        key: str,
    ) -> float | None:
        """Return one numeric marker value from a marker mapping."""

        return _coerce_float(markers.get(key))

    def _environment_baseline(self, *, profile: LongTermMemoryObjectV1, items: tuple[LongTermMemoryObjectV1, ...]) -> LongTermMemoryObjectV1 | None:
        """Select the best-matching baseline for one latest environment profile."""

        profile_environment_id = self._environment_id(profile)
        profile_weekday_class = _normalize_object_text(self._attributes_mapping(profile).get("weekday_class"))
        baselines = [
            item
            for item in items
            if self._environment_domain(item) == _SMART_HOME_ENVIRONMENT_DOMAIN
            and self._environment_pattern_type(item) == _ENVIRONMENT_BASELINE_PATTERN
            and self._environment_id(item) == profile_environment_id
        ]
        if not baselines:
            return None
        preferred = [
            item
            for item in baselines
            if _normalize_object_text(self._attributes_mapping(item).get("baseline_kind")) in {"", "short"}
            and _normalize_object_text(self._attributes_mapping(item).get("weekday_class")) == profile_weekday_class
        ]
        if not preferred:
            preferred = [
                item
                for item in baselines
                if _normalize_object_text(self._attributes_mapping(item).get("baseline_kind")) in {"", "short"}
                and _normalize_object_text(self._attributes_mapping(item).get("weekday_class")) == "all_days"
            ]
        pool = preferred or baselines
        return max(pool, key=lambda item: (self._item_updated_at(item), item.memory_id))

    def _build_environment_reflection(
        self,
        *,
        objects: tuple[LongTermMemoryObjectV1, ...],
    ) -> tuple[LongTermMemoryObjectV1 | None, tuple[LongTermMidtermPacketV1, ...]]:
        """Build one explainable smart-home environment reflection summary and packet."""

        environment_items = tuple(
            item
            for item in objects
            if self._environment_domain(item) == _SMART_HOME_ENVIRONMENT_DOMAIN
        )
        if not environment_items:
            return None, ()

        profile_items = tuple(
            item
            for item in environment_items
            if self._environment_summary_type(item) == _ENVIRONMENT_DAY_PROFILE_SUMMARY
        )
        if not profile_items:
            return None, ()

        latest_profile = max(
            profile_items,
            key=lambda item: (
                self._environment_profile_day(item) or date.min,
                self._item_updated_at(item),
                item.memory_id,
            ),
        )
        profile_day = self._environment_profile_day(latest_profile)
        if profile_day is None:
            return None, ()

        environment_id = self._environment_id(latest_profile)
        profile_attributes = self._attributes_mapping(latest_profile)
        profile_markers = _coerce_mapping(profile_attributes.get("markers"))
        baseline = self._environment_baseline(profile=latest_profile, items=environment_items)
        baseline_attributes = self._attributes_mapping(baseline) if baseline is not None else {}
        baseline_marker_stats = _coerce_mapping(baseline_attributes.get("marker_stats"))
        day_deviations = tuple(
            sorted(
                (
                    item
                    for item in environment_items
                    if self._environment_summary_type(item) in {_ENVIRONMENT_DEVIATION_SUMMARY, "environment_deviation_event"}
                    and self._environment_id(item) == environment_id
                    and self._environment_deviation_day(item) == profile_day
                ),
                key=lambda item: (
                    _normalize_object_text(self._attributes_mapping(item).get("severity")) == "high",
                    self._item_updated_at(item),
                    item.memory_id,
                ),
                reverse=True,
            )
        )
        quality_state = max(
            (
                item
                for item in environment_items
                if self._environment_summary_type(item) == "environment_quality_state"
                and self._environment_id(item) == environment_id
                and self._environment_deviation_day(item) == profile_day
            ),
            key=lambda item: (self._item_updated_at(item), item.memory_id),
            default=None,
        )
        change_point = max(
            (
                item
                for item in environment_items
                if self._environment_summary_type(item) == "environment_change_point"
                and self._environment_id(item) == environment_id
                and self._environment_deviation_day(item) == profile_day
            ),
            key=lambda item: (self._item_updated_at(item), item.memory_id),
            default=None,
        )
        regime = max(
            (
                item
                for item in environment_items
                if self._environment_pattern_type(item) == "environment_regime"
                and self._environment_id(item) == environment_id
                and (
                    _parse_iso_date(self._attributes_mapping(item).get("observed_at"))
                    or _parse_iso_date(getattr(item, "valid_to", None))
                ) == profile_day
            ),
            key=lambda item: (self._item_updated_at(item), item.memory_id),
            default=None,
        )
        node_count = len(
            {
                item.memory_id
                for item in environment_items
                if self._environment_summary_type(item) == _ENVIRONMENT_NODE_SUMMARY
                and self._environment_id(item) == environment_id
            }
        )

        deviation_labels = _unique_texts(
            *(
                _coerce_mapping(self._attributes_mapping(item).get("explanation")).get("short_label")
                for item in day_deviations
            ),
            limit=3,
        )
        deviation_types = _unique_texts(
            *(self._attributes_mapping(item).get("deviation_type") for item in day_deviations),
            limit=4,
        )
        quality_flags = _unique_texts(
            profile_attributes.get("quality_flags"),
            *(self._attributes_mapping(item).get("quality_flags") for item in day_deviations),
            self._attributes_mapping(quality_state).get("quality_flags") if quality_state is not None else (),
            self._attributes_mapping(change_point).get("quality_flags") if change_point is not None else (),
            self._attributes_mapping(regime).get("quality_flags") if regime is not None else (),
            limit=6,
        )
        blocked_by = _unique_texts(
            *(self._attributes_mapping(item).get("blocked_by") for item in day_deviations),
            self._attributes_mapping(quality_state).get("blocked_by") if quality_state is not None else (),
            self._attributes_mapping(change_point).get("blocked_by") if change_point is not None else (),
            limit=4,
        )
        quality_classification = _normalize_object_text(self._attributes_mapping(quality_state).get("classification"))

        active_epochs = self._environment_marker_value(profile_markers, "active_epoch_count_day")
        active_nodes = self._environment_marker_value(profile_markers, "unique_active_node_count_day")
        first_activity = _minute_label(profile_markers.get("first_activity_minute_local"))
        last_activity = _minute_label(profile_markers.get("last_activity_minute_local"))
        coverage_ratio = self._environment_marker_value(profile_markers, "sensor_coverage_ratio_day")

        if regime is not None:
            classification = "new_normal"
        elif change_point is not None:
            classification = "transition"
        elif deviation_labels:
            classification = "acute_deviation"
        elif quality_classification == "blocked":
            classification = "insufficient_quality"
        else:
            classification = "stable"

        summary_parts: list[str] = []
        if classification == "new_normal":
            summary_text = (
                f"Recent home activity on {profile_day.isoformat()} appears to have stabilized into a new normal pattern."
            )
        elif classification == "transition":
            summary_text = (
                f"Recent home activity on {profile_day.isoformat()} suggests an ongoing transition away from the older pattern."
            )
        elif classification == "acute_deviation":
            summary_parts.append(
                f"Recent home activity on {profile_day.isoformat()} differs from the usual pattern"
            )
            if blocked_by:
                summary_parts.append("and should be treated cautiously because sensor quality was limited")
            summary_text = " ".join(summary_parts).strip() + f": {'; '.join(deviation_labels)}."
        elif classification == "insufficient_quality":
            summary_text = (
                f"Recent home activity on {profile_day.isoformat()} is available, but interpretation is limited because sensor quality was reduced."
            )
        else:
            if first_activity and last_activity:
                summary_parts.append(f"activity ran from about {first_activity} to {last_activity}")
            elif first_activity:
                summary_parts.append(f"first activity was around {first_activity}")
            elif last_activity:
                summary_parts.append(f"last activity was around {last_activity}")
            if active_nodes is not None:
                summary_parts.append(f"{int(active_nodes)} motion nodes were active")
            if active_epochs is not None:
                summary_parts.append(f"{int(active_epochs)} activity epochs were observed")
            summary_text = (
                f"Recent home activity on {profile_day.isoformat()} is available for recall: "
                + "; ".join(summary_parts[:3])
                + "."
            )

        detail_parts = ["Room-agnostic smart-home environment reflection compiled from motion and device-health history."]
        if deviation_labels:
            deviation_explanations = _unique_texts(
                *(
                    _coerce_mapping(self._attributes_mapping(item).get("explanation")).get("human_readable")
                    or getattr(item, "details", None)
                    for item in day_deviations
                ),
                limit=2,
            )
            if deviation_explanations:
                detail_parts.append("Recent deviations: " + " ".join(explanation.rstrip(".") + "." for explanation in deviation_explanations))
        if change_point is not None:
            detail_parts.append((getattr(change_point, "details", None) or "An environment transition candidate was detected.").rstrip(".") + ".")
        if regime is not None:
            detail_parts.append((getattr(regime, "details", None) or "A new normal regime was accepted.").rstrip(".") + ".")
        if quality_state is not None:
            detail_parts.append((getattr(quality_state, "details", None) or "Environment quality state is available.").rstrip(".") + ".")
        profile_bits: list[str] = []
        if active_epochs is not None:
            profile_bits.append(f"active epochs {int(active_epochs)}")
        if active_nodes is not None:
            profile_bits.append(f"active nodes {int(active_nodes)}")
        if first_activity:
            profile_bits.append(f"first activity {first_activity}")
        if last_activity:
            profile_bits.append(f"last activity {last_activity}")
        if coverage_ratio is not None:
            profile_bits.append(f"sensor coverage {coverage_ratio:.2f}")
        if profile_bits:
            detail_parts.append("Latest daily profile: " + "; ".join(profile_bits) + ".")

        baseline_bits: list[str] = []
        baseline_weekday_class = _normalize_object_text(baseline_attributes.get("weekday_class"))
        baseline_kind = _normalize_object_text(baseline_attributes.get("baseline_kind"))
        if baseline_kind:
            baseline_bits.append(f"{baseline_kind} baseline")
        if baseline_weekday_class:
            baseline_bits.append(f"baseline bucket {baseline_weekday_class}")
        baseline_active_epochs = self._environment_marker_value(
            _coerce_mapping(baseline_marker_stats.get("active_epoch_count_day")),
            "median",
        )
        baseline_active_nodes = self._environment_marker_value(
            _coerce_mapping(baseline_marker_stats.get("unique_active_node_count_day")),
            "median",
        )
        if baseline_active_epochs is not None:
            baseline_bits.append(f"typical active epochs {baseline_active_epochs:.1f}")
        if baseline_active_nodes is not None:
            baseline_bits.append(f"typical active nodes {baseline_active_nodes:.1f}")
        if baseline_bits:
            detail_parts.append("Rolling baseline: " + "; ".join(baseline_bits) + ".")

        if node_count > 0:
            detail_parts.append(f"The current environment model includes {node_count} observed motion nodes.")
        if quality_flags:
            detail_parts.append("Quality flags: " + ", ".join(quality_flags) + ".")
        if blocked_by:
            detail_parts.append("Caution: " + ", ".join(blocked_by) + ".")
        details_text = " ".join(part for part in detail_parts if part)

        query_hints = _unique_texts(
            "home activity",
            "movement pattern",
            "daily routine",
            "recent activity change",
            "motion sensors",
            deviation_labels,
            deviation_types,
            summary_text,
            limit=10,
        )
        source_memory_ids = tuple(
            dict.fromkeys(
                item.memory_id
                for item in (
                    latest_profile,
                    baseline,
                    *day_deviations,
                    quality_state,
                    change_point,
                    regime,
                )
                if isinstance(item, LongTermMemoryObjectV1)
            )
        )
        source_object = day_deviations[0] if day_deviations else latest_profile
        summary_object = LongTermMemoryObjectV1(
            memory_id=f"environment_reflection:{_slugify(environment_id, fallback='environment')}:{profile_day.isoformat()}",
            kind="summary",
            summary=summary_text,
            details=details_text,
            source=source_object.source,
            status="active",
            confidence=min(0.95, 0.62 + (0.06 * len(day_deviations))),
            sensitivity="low",
            slot_key=f"environment_reflection:{environment_id}",
            value_key=profile_day.isoformat(),
            valid_from=profile_day.isoformat(),
            valid_to=profile_day.isoformat(),
            attributes={
                "memory_domain": _SMART_HOME_ENVIRONMENT_DOMAIN,
                "summary_type": _ENVIRONMENT_REFLECTION_SUMMARY,
                "environment_id": environment_id,
                "profile_day": profile_day.isoformat(),
                "classification": classification,
                "deviation_types": deviation_types,
                "deviation_labels": deviation_labels,
                "quality_flags": quality_flags,
                "blocked_by": blocked_by,
                "quality_classification": quality_classification or None,
                "active_node_count": int(active_nodes) if active_nodes is not None else None,
                "active_epoch_count": int(active_epochs) if active_epochs is not None else None,
                "baseline_weekday_class": baseline_weekday_class or None,
                "baseline_kind": baseline_kind or None,
                "query_hints": query_hints,
                "source_memory_ids": source_memory_ids,
            },
        )

        packet = LongTermMidtermPacketV1(
            packet_id=f"midterm:environment:{_slugify(environment_id, fallback='environment')}:{profile_day.isoformat()}",
            kind=_ENVIRONMENT_REFLECTION_PACKET_KIND,
            summary=summary_text,
            details=details_text,
            source_memory_ids=source_memory_ids,
            query_hints=query_hints,
            sensitivity="low",
            valid_from=profile_day.isoformat(),
            valid_to=profile_day.isoformat(),
            attributes={
                "memory_domain": _SMART_HOME_ENVIRONMENT_DOMAIN,
                "packet_scope": "recent_environment_reflection",
                "environment_id": environment_id,
                "profile_day": profile_day.isoformat(),
                "classification": classification,
                "quality_classification": quality_classification or None,
            },
        )
        return summary_object, (packet,)

    def _person_ref(self, item: LongTermMemoryObjectV1) -> str:
        """Resolve the person reference for one candidate thread item."""

        attributes = self._attributes_mapping(item)
        person_ref = _normalize_object_text(attributes.get("person_ref"))
        if person_ref:
            return person_ref
        subject_ref = _normalize_object_text(attributes.get("subject_ref"))
        if subject_ref.startswith("person:"):
            return subject_ref
        return ""

    def _reference_label(self, reference: str, *, title_case: bool = False) -> str:
        """Render one canonical graph reference into compact display text."""

        normalized = _normalize_text(reference.rsplit(":", 1)[-1].replace("_", " "))
        if not normalized:
            return ""
        if title_case:
            return normalized.title()
        return normalized

    def _is_relationship_fact(self, item: LongTermMemoryObjectV1) -> bool:
        """Return whether one fact describes the user's relationship to a person."""

        attributes = self._attributes_mapping(item)
        if kind_matches(item.kind, "fact", attributes, attr_key="fact_type", attr_value="relationship"):
            return True
        if item.kind != "fact" or not self._person_ref(item):
            return False
        object_ref = _normalize_object_text(attributes.get("object_ref"))
        return object_ref.startswith("user:")

    def _has_thread_summary_support(self, item: LongTermMemoryObjectV1) -> bool:
        if item.status == "active":
            return True
        return item.status in {"candidate", "uncertain"} and self._support_count(item) >= self.min_support_count_for_promotion

    def _is_thread_evidence(self, item: LongTermMemoryObjectV1) -> bool:
        if not self._person_ref(item):
            return False
        if self._is_relationship_fact(item):
            return True
        if item.kind == "fact":
            return True
        return item.kind in _THREAD_SUMMARY_EVIDENCE_KINDS

    def _resolve_person_name(
        self,
        *,
        person_ref: str,
        items: tuple[LongTermMemoryObjectV1, ...],
    ) -> str:
        # AUDIT-FIX(#1): Prefer the freshest explicit person_name instead of whichever item happened to be first in the tuple.
        for item in sorted(items, key=self._item_updated_at, reverse=True):
            person_name = _normalize_object_text(self._attributes_mapping(item).get("person_name"))
            if person_name:
                return person_name
        return self._reference_label(person_ref, title_case=True)

    def _build_thread_summary(
        self,
        *,
        person_ref: str,
        items: tuple[LongTermMemoryObjectV1, ...],
    ) -> LongTermMemoryObjectV1 | None:
        evidence_items = tuple(item for item in items if self._is_thread_evidence(item))
        # AUDIT-FIX(#1): Only real evidence items may count toward thread-summary support; existing summaries or unrelated kinds must not self-reinforce.
        if len(evidence_items) < self.min_support_count_for_thread_summary:
            return None
        supported_items = tuple(item for item in evidence_items if self._has_thread_summary_support(item))
        if len(supported_items) < self.min_support_count_for_thread_summary:
            return None

        person_name = self._resolve_person_name(person_ref=person_ref, items=supported_items)
        topic_bits: list[str] = []
        for item in supported_items:
            attributes = self._attributes_mapping(item)
            if self._is_relationship_fact(item):
                relation = _normalize_object_text(attributes.get("relation"))
                if relation:
                    topic_bits.append(f"{person_name} is the user's {relation}")
                    continue
                summary = _normalize_object_text(getattr(item, "summary", None))
                if summary:
                    topic_bits.append(summary.rstrip("."))
                    continue
                predicate = _normalize_object_text(attributes.get("predicate"))
                if predicate:
                    topic_bits.append(predicate)
            elif item.kind == "fact":
                summary = _normalize_object_text(getattr(item, "summary", None))
                if summary:
                    topic_bits.append(summary.rstrip("."))
                    continue
                predicate = _normalize_object_text(attributes.get("predicate"))
                value_text = _normalize_object_text(attributes.get("value_text"))
                object_ref = _normalize_object_text(attributes.get("object_ref"))
                if predicate and value_text:
                    topic_bits.append(f"{predicate} {value_text}")
                elif predicate and object_ref:
                    topic_bits.append(f"{predicate} {self._reference_label(object_ref)}")
                elif predicate:
                    topic_bits.append(predicate)
            elif item.kind in _THREAD_SUMMARY_EVIDENCE_KINDS:
                action = _normalize_object_text(attributes.get("action"))
                if not action:
                    action = _normalize_object_text(attributes.get("treatment"))
                if not action:
                    action = _normalize_object_text(attributes.get("purpose"))
                if not action:
                    action = _normalize_object_text(attributes.get("predicate"))
                place = _normalize_object_text(attributes.get("place"))
                if not place:
                    object_ref = _normalize_object_text(attributes.get("object_ref"))
                    if object_ref.startswith("place:"):
                        place = self._reference_label(object_ref)
                summary = _normalize_object_text(getattr(item, "summary", None))
                if action and place:
                    topic_bits.append(f"{action} at {place}")
                elif action:
                    topic_bits.append(action)
                elif place:
                    topic_bits.append(f"being at {place}")
                elif summary:
                    topic_bits.append(summary.rstrip("."))

        deduped_topics: list[str] = []
        seen_topics: set[str] = set()
        for topic in topic_bits:
            if topic not in seen_topics:
                seen_topics.add(topic)
                deduped_topics.append(topic)
        if not deduped_topics:
            return None

        summary_text = f"Ongoing thread about {person_name}: " + "; ".join(deduped_topics[:3]) + "."
        memory_id = f"thread:{_slugify(person_ref, fallback='person')}"
        latest = max(supported_items, key=self._item_updated_at)
        sensitivity = max(
            (_normalize_sensitivity(getattr(item, "sensitivity", "normal")) for item in supported_items),
            key=lambda value: _SENSITIVITY_RANK[value],
            default="normal",
        )
        return LongTermMemoryObjectV1(
            memory_id=memory_id,
            kind="summary",
            summary=summary_text,
            details="Reflected from multiple related long-term memory objects.",
            source=latest.source,
            status="active",
            confidence=min(0.99, 0.45 + 0.1 * len(supported_items)),
            sensitivity=sensitivity,
            slot_key=f"thread:{person_ref}",
            value_key=person_ref,
            valid_from=self._earliest_datetime(item.valid_from for item in supported_items),
            valid_to=self._latest_datetime(item.valid_to for item in supported_items),
            attributes={
                "person_ref": person_ref,
                "person_name": person_name,
                "support_count": len(supported_items),
                "topic_items": deduped_topics[:3],
                "summary_type": "thread",
                "memory_domain": "thread",
            },
        )

    def _reflection_window(
        self,
        objects: tuple[LongTermMemoryObjectV1, ...],
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        candidates = tuple(item for item in objects if item.status in _ALLOWED_ACTIVE_STATUSES)
        ranked = sorted(
            candidates,
            key=lambda item: (self._item_updated_at(item), item.confidence, item.memory_id),
            reverse=True,
        )
        return tuple(ranked[: self.reflection_window_size])

    def _midterm_packets_from_payload(
        self,
        payload: Mapping[str, object] | object,
    ) -> tuple[LongTermMidtermPacketV1, ...]:
        if not isinstance(payload, Mapping):
            logger.warning(
                "Unexpected midterm reflection payload type %s; ignoring payload.",
                type(payload).__name__,
            )
            return ()
        raw_packets = payload.get("midterm_packets")
        if not isinstance(raw_packets, (list, tuple)):
            return ()
        packets: list[LongTermMidtermPacketV1] = []
        for raw_packet in raw_packets[: self.midterm_packet_limit]:
            if not isinstance(raw_packet, Mapping):
                continue
            packet_id = _normalize_object_text(raw_packet.get("packet_id"))
            kind = _normalize_object_text(raw_packet.get("kind"))
            summary = _normalize_object_text(raw_packet.get("summary"))
            if not packet_id or not kind or not summary:
                continue

            try:
                packets.append(
                    LongTermMidtermPacketV1(
                        packet_id=packet_id,
                        kind=kind,
                        summary=summary,
                        details=_normalize_object_text(raw_packet.get("details")) or None,
                        source_memory_ids=_coerce_text_sequence(raw_packet.get("source_memory_ids")),
                        query_hints=_coerce_text_sequence(raw_packet.get("query_hints")),
                        sensitivity=_normalize_sensitivity(raw_packet.get("sensitivity", "normal")),
                        valid_from=_normalize_object_text(raw_packet.get("valid_from")) or None,
                        valid_to=_normalize_object_text(raw_packet.get("valid_to")) or None,
                        attributes=_normalize_midterm_attributes(raw_packet.get("attributes")),
                    )
                )
            except Exception:
                # AUDIT-FIX(#5): A single malformed packet should be dropped, not allowed to abort all packet generation.
                logger.exception("Skipping invalid midterm reflection packet %r.", packet_id)
                continue

        unique: dict[str, LongTermMidtermPacketV1] = {}
        for packet in packets:
            unique[packet.packet_id] = packet
        return tuple(unique.values())


__all__ = ["LongTermMemoryReflector"]
