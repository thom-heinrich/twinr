from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Mapping

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

_SENSITIVITY_RANK = {
    "low": 0,
    "normal": 1,
    "private": 2,
    "sensitive": 3,
    "critical": 4,
}


def _normalize_text(value: str | None) -> str:
    return collapse_whitespace(value)


def _slugify(value: str, *, fallback: str) -> str:
    return slugify_identifier(value, fallback=fallback)


def _normalize_midterm_attributes(value: object) -> dict[str, str] | None:
    if not isinstance(value, Mapping):
        return None
    normalized: dict[str, str] = {}
    for key, raw_value in value.items():
        clean_key = _normalize_text(str(key))
        if not clean_key:
            continue
        if raw_value is None:
            continue
        if isinstance(raw_value, str):
            clean_value = _normalize_text(raw_value)
        elif isinstance(raw_value, (list, tuple)):
            clean_value = _normalize_text(
                ", ".join(_normalize_text(str(item)) for item in raw_value if _normalize_text(str(item)))
            )
        elif isinstance(raw_value, (bool, int, float)):
            clean_value = _normalize_text(str(raw_value))
        else:
            clean_value = _normalize_text(json.dumps(raw_value, ensure_ascii=False, sort_keys=True))
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
    timezone_name: str = "Europe/Berlin"

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "LongTermMemoryReflector":
        program = structured_reflection_program_from_config(config) if config.long_term_memory_midterm_enabled else None
        return cls(
            program=program,
            midterm_packet_limit=max(1, config.long_term_memory_midterm_limit),
            reflection_window_size=max(4, config.long_term_memory_reflection_window_size),
            timezone_name=config.local_timezone_name,
        )

    def reflect(
        self,
        *,
        objects: tuple[LongTermMemoryObjectV1, ...],
    ) -> LongTermReflectionResultV1:
        reflected: list[LongTermMemoryObjectV1] = []
        created_summaries: list[LongTermMemoryObjectV1] = []
        midterm_packets: list[LongTermMidtermPacketV1] = []

        for item in objects:
            item = item.canonicalized()
            support_count = self._support_count(item)
            if item.status in {"candidate", "uncertain"} and support_count >= self.min_support_count_for_promotion:
                reflected.append(
                    item.with_updates(
                        status="active",
                        confidence=max(item.confidence, min(0.99, 0.55 + 0.12 * support_count)),
                    )
                )

        by_person: dict[str, list[LongTermMemoryObjectV1]] = {}
        for item in objects:
            item = item.canonicalized()
            person_ref = _normalize_text(str((item.attributes or {}).get("person_ref", "")))
            if not person_ref:
                continue
            by_person.setdefault(person_ref, []).append(item)

        for person_ref, person_items in by_person.items():
            summary = self._build_thread_summary(person_ref=person_ref, items=tuple(person_items))
            if summary is not None:
                created_summaries.append(summary)

        if self.program is not None and self.midterm_packet_limit > 0:
            payload = self.program.compile_reflection(
                objects=self._reflection_window(objects),
                timezone_name=self.timezone_name,
                packet_limit=self.midterm_packet_limit,
            )
            midterm_packets.extend(self._midterm_packets_from_payload(payload))

        return LongTermReflectionResultV1(
            reflected_objects=tuple(reflected),
            created_summaries=tuple(created_summaries),
            midterm_packets=tuple(midterm_packets),
        )

    def _support_count(self, item: LongTermMemoryObjectV1) -> int:
        raw = (item.attributes or {}).get("support_count")
        if isinstance(raw, int):
            return raw
        if isinstance(raw, float):
            return int(raw)
        return 1

    def _build_thread_summary(
        self,
        *,
        person_ref: str,
        items: tuple[LongTermMemoryObjectV1, ...],
    ) -> LongTermMemoryObjectV1 | None:
        if len(items) < self.min_support_count_for_thread_summary:
            return None
        active_items = tuple(item for item in items if item.status in {"active", "candidate", "uncertain"})
        if len(active_items) < self.min_support_count_for_thread_summary:
            return None
        person_name = _normalize_text(str((active_items[0].attributes or {}).get("person_name", ""))) or person_ref.rsplit(":", 1)[-1].replace("_", " ").title()
        topic_bits: list[str] = []
        for item in active_items:
            if kind_matches(item.kind, "fact", item.attributes, attr_key="fact_type", attr_value="relationship"):
                relation = _normalize_text(str((item.attributes or {}).get("relation", "")))
                if relation:
                    topic_bits.append(f"{person_name} is the user's {relation}")
            elif item.kind in {"event", "plan"}:
                action = _normalize_text(str((item.attributes or {}).get("action", "")))
                if not action:
                    action = _normalize_text(str((item.attributes or {}).get("treatment", "")))
                if not action:
                    action = _normalize_text(str((item.attributes or {}).get("purpose", "")))
                place = _normalize_text(str((item.attributes or {}).get("place", "")))
                summary = _normalize_text(item.summary)
                if action and place:
                    topic_bits.append(f"{action} at {place}")
                elif action:
                    topic_bits.append(action)
                elif place:
                    topic_bits.append(f"being at {place}")
                elif summary:
                    topic_bits.append(summary.rstrip("."))
        deduped_topics: list[str] = []
        for item in topic_bits:
            if item not in deduped_topics:
                deduped_topics.append(item)
        if not deduped_topics:
            return None
        summary_text = f"Ongoing thread about {person_name}: " + "; ".join(deduped_topics[:3]) + "."
        memory_id = f"thread:{_slugify(person_ref, fallback='person')}"
        latest = max(active_items, key=lambda item: item.updated_at)
        sensitivity = max(active_items, key=lambda item: _SENSITIVITY_RANK.get(item.sensitivity, 0)).sensitivity
        return LongTermMemoryObjectV1(
            memory_id=memory_id,
            kind="summary",
            summary=summary_text,
            details="Reflected from multiple related long-term memory objects.",
            source=latest.source,
            status="active",
            confidence=min(0.99, 0.45 + 0.1 * len(active_items)),
            sensitivity=sensitivity,
            slot_key=f"thread:{person_ref}",
            value_key=person_ref,
            valid_from=min((item.valid_from for item in active_items if item.valid_from), default=None),
            valid_to=max((item.valid_to for item in active_items if item.valid_to), default=None),
            attributes={
                "person_ref": person_ref,
                "person_name": person_name,
                "support_count": len(active_items),
                "topic_items": deduped_topics[:3],
                "summary_type": "thread",
                "memory_domain": "thread",
            },
        )

    def _reflection_window(
        self,
        objects: tuple[LongTermMemoryObjectV1, ...],
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        candidates = tuple(
            item.canonicalized()
            for item in objects
            if item.status in {"active", "candidate", "uncertain"}
        )
        ranked = sorted(
            candidates,
            key=lambda item: (item.updated_at.isoformat(), item.confidence, item.memory_id),
            reverse=True,
        )
        return tuple(ranked[: self.reflection_window_size])

    def _midterm_packets_from_payload(
        self,
        payload: Mapping[str, object],
    ) -> tuple[LongTermMidtermPacketV1, ...]:
        raw_packets = payload.get("midterm_packets")
        if not isinstance(raw_packets, list):
            return ()
        packets: list[LongTermMidtermPacketV1] = []
        for raw_packet in raw_packets[: self.midterm_packet_limit]:
            if not isinstance(raw_packet, Mapping):
                continue
            packet_id = _normalize_text(str(raw_packet.get("packet_id", "")))
            kind = _normalize_text(str(raw_packet.get("kind", "")))
            summary = _normalize_text(str(raw_packet.get("summary", "")))
            if not packet_id or not kind or not summary:
                continue
            source_memory_ids = tuple(
                str(item).strip()
                for item in raw_packet.get("source_memory_ids", [])
                if str(item).strip()
            )
            query_hints = tuple(
                _normalize_text(str(item))
                for item in raw_packet.get("query_hints", [])
                if _normalize_text(str(item))
            )
            attributes = _normalize_midterm_attributes(raw_packet.get("attributes"))
            packets.append(
                LongTermMidtermPacketV1(
                    packet_id=packet_id,
                    kind=kind,
                    summary=summary,
                    details=_normalize_text(str(raw_packet.get("details", ""))) or None,
                    source_memory_ids=source_memory_ids,
                    query_hints=query_hints,
                    sensitivity=_normalize_text(str(raw_packet.get("sensitivity", "normal"))) or "normal",
                    valid_from=_normalize_text(str(raw_packet.get("valid_from", ""))) or None,
                    valid_to=_normalize_text(str(raw_packet.get("valid_to", ""))) or None,
                    attributes=attributes,
                )
            )
        unique: dict[str, LongTermMidtermPacketV1] = {}
        for packet in packets:
            unique[packet.packet_id] = packet
        return tuple(unique.values())


__all__ = ["LongTermMemoryReflector"]
