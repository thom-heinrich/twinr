from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from twinr.memory.longterm.models import (
    LongTermConflictOptionV1,
    LongTermConflictQueueItemV1,
    LongTermConflictResolutionV1,
    LongTermMemoryConflictV1,
    LongTermMemoryObjectV1,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class LongTermConflictResolver:
    user_confirmed_confidence_floor: float = 0.99

    def build_queue_items(
        self,
        *,
        conflicts: tuple[LongTermMemoryConflictV1, ...],
        objects: tuple[LongTermMemoryObjectV1, ...],
    ) -> tuple[LongTermConflictQueueItemV1, ...]:
        object_map = {item.memory_id: item for item in objects}
        queue: list[LongTermConflictQueueItemV1] = []
        for conflict in conflicts:
            option_ids = tuple(dict.fromkeys((*conflict.existing_memory_ids, conflict.candidate_memory_id)))
            options = [
                LongTermConflictOptionV1(
                    memory_id=item.memory_id,
                    summary=item.summary,
                    details=item.details,
                    status=item.status,
                    value_key=item.value_key,
                )
                for option_id in option_ids
                if (item := object_map.get(option_id)) is not None
            ]
            if not options:
                continue
            queue.append(
                LongTermConflictQueueItemV1(
                    slot_key=conflict.slot_key,
                    question=conflict.question,
                    reason=conflict.reason,
                    candidate_memory_id=conflict.candidate_memory_id,
                    options=tuple(sorted(options, key=self._option_sort_key)),
                )
            )
        return tuple(queue)

    def resolve(
        self,
        *,
        conflict: LongTermMemoryConflictV1,
        objects: tuple[LongTermMemoryObjectV1, ...],
        remaining_conflicts: tuple[LongTermMemoryConflictV1, ...],
        selected_memory_id: str,
        now: datetime | None = None,
    ) -> LongTermConflictResolutionV1:
        object_map = {item.memory_id: item for item in objects}
        option_ids = tuple(
            dict.fromkeys(
                (
                    *conflict.existing_memory_ids,
                    conflict.candidate_memory_id,
                    *(
                        item.memory_id
                        for item in objects
                        if item.slot_key == conflict.slot_key and item.memory_id != selected_memory_id
                    ),
                )
            )
        )
        if selected_memory_id not in option_ids:
            raise ValueError("selected_memory_id must match one of the conflict options.")
        selected = object_map.get(selected_memory_id)
        if selected is None:
            raise ValueError("selected memory object was not found in the structured store.")
        current_time = now or _utcnow()
        updated_objects: list[LongTermMemoryObjectV1] = []
        superseded_ids: list[str] = []
        for memory_id in option_ids:
            current = object_map.get(memory_id)
            if current is None:
                continue
            if memory_id == selected_memory_id:
                continue
            losing_status = self._losing_status(current)
            if losing_status == "superseded":
                superseded_ids.append(memory_id)
            updated_objects.append(
                current.with_updates(
                    status=losing_status,
                    conflicts_with=(),
                    updated_at=current_time,
                )
            )
        selected_attrs = dict(selected.attributes or {})
        selected_attrs["resolved_by_user"] = True
        selected_attrs["resolved_at"] = current_time.isoformat()
        updated_objects.append(
            selected.with_updates(
                status="active",
                confirmed_by_user=True,
                conflicts_with=(),
                supersedes=tuple(dict.fromkeys((*selected.supersedes, *superseded_ids))),
                confidence=max(selected.confidence, self.user_confirmed_confidence_floor),
                updated_at=current_time,
                attributes=selected_attrs,
            )
        )
        return LongTermConflictResolutionV1(
            slot_key=conflict.slot_key,
            selected_memory_id=selected_memory_id,
            updated_objects=tuple(sorted(updated_objects, key=lambda item: item.memory_id)),
            remaining_conflicts=tuple(item for item in remaining_conflicts if item.slot_key != conflict.slot_key),
        )

    def _option_sort_key(self, option: LongTermConflictOptionV1) -> tuple[int, str]:
        rank = {
            "active": 0,
            "uncertain": 1,
            "candidate": 2,
            "superseded": 3,
            "expired": 4,
            "invalid": 5,
        }
        return (rank.get(option.status, 99), option.memory_id)

    def _losing_status(self, item: LongTermMemoryObjectV1) -> str:
        if item.status == "active":
            return "superseded"
        return "invalid"


__all__ = ["LongTermConflictResolver"]
