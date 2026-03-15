from __future__ import annotations

from collections.abc import Iterable  # AUDIT-FIX(#5): Robustly coerce list-like payload fields without exploding strings into character-tuples.
from dataclasses import dataclass
from datetime import datetime, timezone  # AUDIT-FIX(#4): Normalize naive/aware datetimes before sorting and min/max operations.
import json
import logging  # AUDIT-FIX(#2): Log degraded optional-reflection paths instead of crashing silently.
import math  # AUDIT-FIX(#6): Safely reject non-finite numeric config values.
from typing import Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError  # AUDIT-FIX(#4): Validate and apply the configured timezone for naive datetimes.

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.longterm.midterm import (
    LongTermStructuredReflectionProgram,
    structured_reflection_program_from_config,
)
from twinr.memory.longterm.ontology import kind_matches
from twinr.memory.longterm.models import (
    LongTermMemoryObjectV1,
    LongTermMidtermPacketV1,
    LongTermReflectionResultV1,
)
from twinr.text_utils import collapse_whitespace, slugify_identifier

logger = logging.getLogger(__name__)  # AUDIT-FIX(#2): Preserve operational visibility when optional enrichment degrades.

_DEFAULT_TIMEZONE_NAME = "Europe/Berlin"  # AUDIT-FIX(#6): Keep a known-good fallback timezone for invalid/missing config.
_ALLOWED_ACTIVE_STATUSES = frozenset({"active", "candidate", "uncertain"})
_THREAD_SUMMARY_EVIDENCE_KINDS = frozenset({"event", "plan"})  # AUDIT-FIX(#1): Only evidence-bearing kinds may support thread summaries.
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


@dataclass(frozen=True, slots=True)
class LongTermMemoryReflector:
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
    ) -> LongTermReflectionResultV1:
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
            person_ref = _normalize_object_text(self._attributes_mapping(item).get("person_ref"))
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

        if self.program is not None and self.midterm_packet_limit > 0:
            try:
                # AUDIT-FIX(#5): Optional midterm enrichment must degrade gracefully instead of crashing the caller on external/program errors.
                payload = self.program.compile_reflection(
                    objects=self._reflection_window(normalized_objects),
                    timezone_name=self.timezone_name,
                    packet_limit=self.midterm_packet_limit,
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

    def _has_thread_summary_support(self, item: LongTermMemoryObjectV1) -> bool:
        if item.status == "active":
            return True
        return item.status in {"candidate", "uncertain"} and self._support_count(item) >= self.min_support_count_for_promotion

    def _is_thread_evidence(self, item: LongTermMemoryObjectV1) -> bool:
        attributes = self._attributes_mapping(item)
        return kind_matches(item.kind, "fact", attributes, attr_key="fact_type", attr_value="relationship") or item.kind in _THREAD_SUMMARY_EVIDENCE_KINDS

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
        return person_ref.rsplit(":", 1)[-1].replace("_", " ").title()

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
            if kind_matches(item.kind, "fact", attributes, attr_key="fact_type", attr_value="relationship"):
                relation = _normalize_object_text(attributes.get("relation"))
                if relation:
                    topic_bits.append(f"{person_name} is the user's {relation}")
            elif item.kind in _THREAD_SUMMARY_EVIDENCE_KINDS:
                action = _normalize_object_text(attributes.get("action"))
                if not action:
                    action = _normalize_object_text(attributes.get("treatment"))
                if not action:
                    action = _normalize_object_text(attributes.get("purpose"))
                place = _normalize_object_text(attributes.get("place"))
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