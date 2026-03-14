from __future__ import annotations

from dataclasses import dataclass

from twinr.memory.longterm.models import LongTermMemoryObjectV1, LongTermReflectionResultV1
from twinr.text_utils import collapse_whitespace, slugify_identifier

_SENSITIVITY_RANK = {
    "low": 0,
    "normal": 1,
    "private": 2,
    "medical": 3,
    "sensitive": 4,
}


def _normalize_text(value: str | None) -> str:
    return collapse_whitespace(value)


def _slugify(value: str, *, fallback: str) -> str:
    return slugify_identifier(value, fallback=fallback)


@dataclass(frozen=True, slots=True)
class LongTermMemoryReflector:
    min_support_count_for_promotion: int = 2
    min_support_count_for_thread_summary: int = 2

    def reflect(
        self,
        *,
        objects: tuple[LongTermMemoryObjectV1, ...],
    ) -> LongTermReflectionResultV1:
        reflected: list[LongTermMemoryObjectV1] = []
        created_summaries: list[LongTermMemoryObjectV1] = []

        for item in objects:
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
            person_ref = _normalize_text(str((item.attributes or {}).get("person_ref", "")))
            if not person_ref:
                continue
            by_person.setdefault(person_ref, []).append(item)

        for person_ref, person_items in by_person.items():
            summary = self._build_thread_summary(person_ref=person_ref, items=tuple(person_items))
            if summary is not None:
                created_summaries.append(summary)

        return LongTermReflectionResultV1(
            reflected_objects=tuple(reflected),
            created_summaries=tuple(created_summaries),
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
            if item.kind == "relationship_fact":
                relation = _normalize_text(str((item.attributes or {}).get("relation", "")))
                if relation:
                    topic_bits.append(f"{person_name} is the user's {relation}")
            elif item.kind in {"medical_event", "event_fact"}:
                treatment = _normalize_text(str((item.attributes or {}).get("treatment", "")))
                place = _normalize_text(str((item.attributes or {}).get("place", "")))
                if treatment and place:
                    topic_bits.append(f"{treatment} at {place}")
                elif treatment:
                    topic_bits.append(treatment)
                elif place:
                    topic_bits.append(f"being at {place}")
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
            kind="thread_summary",
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
            },
        )


__all__ = ["LongTermMemoryReflector"]
