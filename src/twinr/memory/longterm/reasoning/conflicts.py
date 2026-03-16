from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from twinr.memory.longterm.core.models import (
    LongTermConflictOptionV1,
    LongTermConflictQueueItemV1,
    LongTermConflictResolutionV1,
    LongTermMemoryConflictV1,
    LongTermMemoryObjectV1,
)

_STATUS_RANK: dict[str, int] = {
    "active": 0,
    "uncertain": 1,
    "candidate": 2,
    "superseded": 3,
    "expired": 4,
    "invalid": 5,
}
# AUDIT-FIX(#1): Only live statuses participate in slot-level conflict choice expansion.
_LIVE_CONFLICT_STATUSES = frozenset({"active", "uncertain", "candidate"})
# AUDIT-FIX(#1): Terminal statuses must be preserved instead of being flattened into "invalid".
_TERMINAL_STATUSES = frozenset({"superseded", "expired", "invalid"})


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_aware_utc(value: datetime | None) -> datetime:
    # AUDIT-FIX(#6): Reject naive datetimes and normalize all persisted timestamps to UTC.
    if value is None:
        return _utcnow()
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("now must be timezone-aware.")
    return value.astimezone(timezone.utc)


def _dedupe_ids(*memory_ids: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(memory_ids))


@dataclass(frozen=True, slots=True)
class LongTermConflictResolver:
    user_confirmed_confidence_floor: float = 0.99

    def __post_init__(self) -> None:
        # AUDIT-FIX(#5): Fail fast on invalid confidence floors so confidence stays in the 0..1 domain.
        if not 0.0 <= self.user_confirmed_confidence_floor <= 1.0:
            raise ValueError("user_confirmed_confidence_floor must be between 0.0 and 1.0 inclusive.")

    def build_queue_items(
        self,
        *,
        conflicts: tuple[LongTermMemoryConflictV1, ...],
        objects: tuple[LongTermMemoryObjectV1, ...],
    ) -> tuple[LongTermConflictQueueItemV1, ...]:
        # AUDIT-FIX(#3): Validate memory_id uniqueness before any conflict processing.
        object_map = self._build_object_map(objects)
        queue: list[LongTermConflictQueueItemV1] = []
        seen_slot_keys: set[str] = set()
        for conflict in conflicts:
            # AUDIT-FIX(#2): Emit at most one queue item per slot because resolution is slot-scoped.
            if conflict.slot_key in seen_slot_keys:
                continue
            option_ids = self._conflict_option_ids(conflict=conflict, objects=objects)
            try:
                # AUDIT-FIX(#4): Refuse malformed or cross-slot conflict options instead of showing partial choices.
                option_objects = self._validated_option_objects(
                    conflict=conflict,
                    object_map=object_map,
                    option_ids=option_ids,
                )
            except ValueError:
                continue
            # AUDIT-FIX(#4): Skip degenerate conflicts that no longer present a real user choice.
            if len(option_objects) < 2:
                continue
            options = [
                LongTermConflictOptionV1(
                    memory_id=item.memory_id,
                    summary=item.summary,
                    details=item.details,
                    status=item.status,
                    value_key=item.value_key,
                )
                for item in option_objects
            ]
            queue.append(
                LongTermConflictQueueItemV1(
                    slot_key=conflict.slot_key,
                    question=conflict.question,
                    reason=conflict.reason,
                    candidate_memory_id=conflict.candidate_memory_id,
                    options=tuple(sorted(options, key=self._option_sort_key)),
                )
            )
            seen_slot_keys.add(conflict.slot_key)
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
        # AUDIT-FIX(#3): Validate memory_id uniqueness before resolving against a possibly corrupted snapshot.
        object_map = self._build_object_map(objects)
        # AUDIT-FIX(#1): Use the same live-option expansion in queueing and resolution.
        option_ids = self._conflict_option_ids(conflict=conflict, objects=objects)
        # AUDIT-FIX(#4): Fail fast if the conflict references missing or cross-slot objects.
        option_objects = self._validated_option_objects(
            conflict=conflict,
            object_map=object_map,
            option_ids=option_ids,
        )
        if selected_memory_id not in option_ids:
            raise ValueError("selected_memory_id must match one of the conflict options.")
        selected = object_map[selected_memory_id]
        current_time = _coerce_aware_utc(now)
        updated_objects: list[LongTermMemoryObjectV1] = []
        superseded_ids: list[str] = []
        for current in option_objects:
            if current.memory_id == selected_memory_id:
                continue
            losing_status = self._losing_status(current)
            if losing_status == "superseded":
                superseded_ids.append(current.memory_id)
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
                supersedes=_dedupe_ids(*selected.supersedes, *superseded_ids),
                confidence=max(selected.confidence, self.user_confirmed_confidence_floor),
                updated_at=current_time,
                attributes=selected_attrs,
            )
        )
        # AUDIT-FIX(#2): Resolution remains slot-scoped, so same-slot conflicts are cleared together.
        return LongTermConflictResolutionV1(
            slot_key=conflict.slot_key,
            selected_memory_id=selected_memory_id,
            updated_objects=tuple(sorted(updated_objects, key=lambda item: item.memory_id)),
            remaining_conflicts=tuple(item for item in remaining_conflicts if item.slot_key != conflict.slot_key),
        )

    def _build_object_map(
        self,
        objects: tuple[LongTermMemoryObjectV1, ...],
    ) -> dict[str, LongTermMemoryObjectV1]:
        duplicate_ids: set[str] = set()
        object_map: dict[str, LongTermMemoryObjectV1] = {}
        for item in objects:
            if item.memory_id in object_map:
                duplicate_ids.add(item.memory_id)
                continue
            object_map[item.memory_id] = item
        if duplicate_ids:
            # AUDIT-FIX(#3): Hard-fail on duplicate IDs instead of silently overwriting state.
            raise ValueError(
                "duplicate memory_id values in structured store: "
                + ", ".join(sorted(duplicate_ids))
            )
        return object_map

    def _conflict_option_ids(
        self,
        *,
        conflict: LongTermMemoryConflictV1,
        objects: tuple[LongTermMemoryObjectV1, ...],
    ) -> tuple[str, ...]:
        # AUDIT-FIX(#1): Include all live same-slot memories so queueing and resolution operate on the same choice set.
        explicit_ids = _dedupe_ids(*conflict.existing_memory_ids, conflict.candidate_memory_id)
        live_same_slot_ids = tuple(
            item.memory_id
            for item in objects
            if item.slot_key == conflict.slot_key and item.status in _LIVE_CONFLICT_STATUSES
        )
        return _dedupe_ids(*explicit_ids, *live_same_slot_ids)

    def _validated_option_objects(
        self,
        *,
        conflict: LongTermMemoryConflictV1,
        object_map: dict[str, LongTermMemoryObjectV1],
        option_ids: tuple[str, ...],
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        option_objects: list[LongTermMemoryObjectV1] = []
        for memory_id in option_ids:
            current = object_map.get(memory_id)
            if current is None:
                # AUDIT-FIX(#4): Missing references indicate a corrupted or stale conflict snapshot.
                raise ValueError(f"conflict references unknown memory_id: {memory_id}")
            if current.slot_key != conflict.slot_key:
                # AUDIT-FIX(#4): Cross-slot references must never be presented as a valid choice.
                raise ValueError(
                    f"conflict slot_key mismatch for memory_id={memory_id}: "
                    f"{current.slot_key!r} != {conflict.slot_key!r}"
                )
            option_objects.append(current)
        return tuple(option_objects)

    def _option_sort_key(self, option: LongTermConflictOptionV1) -> tuple[int, str]:
        return (_STATUS_RANK.get(option.status, 99), option.memory_id)

    def _losing_status(self, item: LongTermMemoryObjectV1) -> str:
        # AUDIT-FIX(#1): Preserve terminal/archive statuses instead of flattening history to "invalid".
        if item.status == "active":
            return "superseded"
        if item.status in _TERMINAL_STATUSES:
            return item.status
        return "invalid"


__all__ = ["LongTermConflictResolver"]