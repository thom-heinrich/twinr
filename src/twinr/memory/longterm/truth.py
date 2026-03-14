from __future__ import annotations

from dataclasses import dataclass

from twinr.memory.longterm.models import LongTermMemoryConflictV1, LongTermMemoryObjectV1


def _normalize_text(value: str | None) -> str:
    return " ".join(str(value or "").split()).strip()


@dataclass(frozen=True, slots=True)
class LongTermTruthMaintainer:
    active_statuses: frozenset[str] = frozenset({"active", "uncertain", "candidate"})

    def detect_conflicts(
        self,
        *,
        existing_objects: tuple[LongTermMemoryObjectV1, ...],
        candidate: LongTermMemoryObjectV1,
    ) -> tuple[LongTermMemoryConflictV1, ...]:
        if not candidate.slot_key or not candidate.value_key:
            return ()
        conflicting = tuple(
            item
            for item in existing_objects
            if item.memory_id != candidate.memory_id
            and item.status in self.active_statuses
            and item.slot_key == candidate.slot_key
            and item.value_key
            and item.value_key != candidate.value_key
        )
        if not conflicting:
            return ()
        return (
            LongTermMemoryConflictV1(
                slot_key=candidate.slot_key,
                candidate_memory_id=candidate.memory_id,
                existing_memory_ids=tuple(item.memory_id for item in conflicting),
                question=self._clarification_question(candidate, conflicting),
                reason=f"Conflicting active memories exist for slot {candidate.slot_key}.",
            ),
        )

    def activate_candidate(
        self,
        *,
        candidate: LongTermMemoryObjectV1,
        existing_objects: tuple[LongTermMemoryObjectV1, ...],
    ) -> LongTermMemoryObjectV1:
        conflicts = self.detect_conflicts(existing_objects=existing_objects, candidate=candidate)
        if conflicts:
            return candidate.with_updates(
                status="uncertain",
                conflicts_with=tuple(
                    memory_id
                    for conflict in conflicts
                    for memory_id in conflict.existing_memory_ids
                ),
            )
        if candidate.status == "candidate":
            return candidate.with_updates(status="active")
        return candidate

    def _clarification_question(
        self,
        candidate: LongTermMemoryObjectV1,
        existing_objects: tuple[LongTermMemoryObjectV1, ...],
    ) -> str:
        label = _normalize_text(candidate.summary)
        if candidate.kind == "contact_method_fact":
            return "I have more than one contact detail for this person. Which one should I use?"
        if candidate.kind == "relationship_fact":
            return "I have conflicting relationship information. Which one is correct now?"
        if candidate.kind in {"medical_event", "event_fact"}:
            return "I have more than one memory about this event. Which one is correct now?"
        if label:
            return f"I have conflicting memories about this detail: {label} Which one is correct now?"
        return "I have conflicting memories here. Which one is correct now?"


__all__ = ["LongTermTruthMaintainer"]
