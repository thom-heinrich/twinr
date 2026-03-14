from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from threading import Lock

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.chonkydb.client import chonkydb_data_path
from twinr.memory.fulltext import FullTextDocument, FullTextSelector
from twinr.memory.longterm.models import (
    LongTermConsolidationResultV1,
    LongTermMemoryConflictV1,
    LongTermMemoryObjectV1,
    LongTermConflictResolutionV1,
    LongTermMemoryMutationResultV1,
    LongTermMemoryReviewItemV1,
    LongTermMemoryReviewResultV1,
    LongTermRetentionResultV1,
    LongTermReflectionResultV1,
)
from twinr.text_utils import retrieval_terms


_OBJECT_STORE_SCHEMA = "twinr_memory_object_store"
_OBJECT_STORE_VERSION = 1
_CONFLICT_STORE_SCHEMA = "twinr_memory_conflict_store"
_CONFLICT_STORE_VERSION = 1


def _normalize_text(value: str | None) -> str:
    return " ".join(str(value or "").split()).strip()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


_NON_SEMANTIC_ATTRIBUTE_KEYS = frozenset(
    {
        "support_count",
        "event_names",
    }
)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        temp_path.replace(path)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


@dataclass(slots=True)
class LongTermStructuredStore:
    base_path: Path
    _lock: Lock = field(default_factory=Lock, repr=False)

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "LongTermStructuredStore":
        return cls(base_path=chonkydb_data_path(config))

    @property
    def objects_path(self) -> Path:
        return self.base_path / "twinr_memory_objects_v1.json"

    @property
    def conflicts_path(self) -> Path:
        return self.base_path / "twinr_memory_conflicts_v1.json"

    def load_objects(self) -> tuple[LongTermMemoryObjectV1, ...]:
        if not self.objects_path.exists():
            return ()
        payload = json.loads(self.objects_path.read_text(encoding="utf-8"))
        items = payload.get("objects", [])
        if not isinstance(items, list):
            return ()
        return tuple(
            LongTermMemoryObjectV1.from_payload(item)
            for item in items
            if isinstance(item, dict)
        )

    def load_conflicts(self) -> tuple[LongTermMemoryConflictV1, ...]:
        if not self.conflicts_path.exists():
            return ()
        payload = json.loads(self.conflicts_path.read_text(encoding="utf-8"))
        items = payload.get("conflicts", [])
        if not isinstance(items, list):
            return ()
        conflicts: list[LongTermMemoryConflictV1] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            existing_ids = item.get("existing_memory_ids")
            conflicts.append(
                LongTermMemoryConflictV1(
                    slot_key=str(item.get("slot_key", "")),
                    candidate_memory_id=str(item.get("candidate_memory_id", "")),
                    existing_memory_ids=tuple(str(value) for value in existing_ids if isinstance(value, str))
                    if isinstance(existing_ids, list)
                    else (),
                    question=str(item.get("question", "")),
                    reason=str(item.get("reason", "")),
                )
            )
        return tuple(conflicts)

    def get_object(self, memory_id: str) -> LongTermMemoryObjectV1 | None:
        normalized = _normalize_text(memory_id)
        if not normalized:
            return None
        return next((item for item in self.load_objects() if item.memory_id == normalized), None)

    def apply_consolidation(self, result: LongTermConsolidationResultV1) -> None:
        with self._lock:
            existing_objects = {item.memory_id: item for item in self.load_objects()}
            for item in (*result.episodic_objects, *result.durable_objects, *result.deferred_objects):
                existing = existing_objects.get(item.memory_id)
                existing_objects[item.memory_id] = self._merge_object(
                    existing=existing,
                    incoming=item,
                    increment_support=True,
                )
            existing_conflicts = {item.slot_key: item for item in self.load_conflicts()}
            for conflict in result.conflicts:
                existing_conflicts[conflict.slot_key] = conflict
            objects_payload = {
                "schema": _OBJECT_STORE_SCHEMA,
                "version": _OBJECT_STORE_VERSION,
                "objects": [item.to_payload() for item in sorted(existing_objects.values(), key=lambda row: row.memory_id)],
            }
            conflicts_payload = {
                "schema": _CONFLICT_STORE_SCHEMA,
                "version": _CONFLICT_STORE_VERSION,
                "conflicts": [
                    item.to_payload()
                    for item in sorted(existing_conflicts.values(), key=lambda row: (row.slot_key, row.candidate_memory_id))
                ],
            }
            _write_json_atomic(self.objects_path, objects_payload)
            _write_json_atomic(self.conflicts_path, conflicts_payload)

    def apply_reflection(self, result: LongTermReflectionResultV1) -> None:
        with self._lock:
            existing_objects = {item.memory_id: item for item in self.load_objects()}
            for item in (*result.reflected_objects, *result.created_summaries):
                existing_objects[item.memory_id] = self._merge_object(
                    existing=existing_objects.get(item.memory_id),
                    incoming=item,
                    increment_support=False,
                )
            payload = {
                "schema": _OBJECT_STORE_SCHEMA,
                "version": _OBJECT_STORE_VERSION,
                "objects": [item.to_payload() for item in sorted(existing_objects.values(), key=lambda row: row.memory_id)],
            }
            _write_json_atomic(self.objects_path, payload)

    def apply_retention(self, result: LongTermRetentionResultV1) -> None:
        with self._lock:
            objects = {item.memory_id: item for item in result.kept_objects}
            payload = {
                "schema": _OBJECT_STORE_SCHEMA,
                "version": _OBJECT_STORE_VERSION,
                "objects": [item.to_payload() for item in sorted(objects.values(), key=lambda row: row.memory_id)],
            }
            _write_json_atomic(self.objects_path, payload)

    def apply_conflict_resolution(self, result: LongTermConflictResolutionV1) -> None:
        with self._lock:
            existing_objects = {item.memory_id: item for item in self.load_objects()}
            for item in result.updated_objects:
                existing_objects[item.memory_id] = item
            objects_payload = {
                "schema": _OBJECT_STORE_SCHEMA,
                "version": _OBJECT_STORE_VERSION,
                "objects": [item.to_payload() for item in sorted(existing_objects.values(), key=lambda row: row.memory_id)],
            }
            conflicts_payload = {
                "schema": _CONFLICT_STORE_SCHEMA,
                "version": _CONFLICT_STORE_VERSION,
                "conflicts": [
                    item.to_payload()
                    for item in sorted(result.remaining_conflicts, key=lambda row: (row.slot_key, row.candidate_memory_id))
                ],
            }
            _write_json_atomic(self.objects_path, objects_payload)
            _write_json_atomic(self.conflicts_path, conflicts_payload)

    def apply_memory_mutation(self, result: LongTermMemoryMutationResultV1) -> None:
        with self._lock:
            existing_objects = {item.memory_id: item for item in self.load_objects()}
            for memory_id in result.deleted_memory_ids:
                existing_objects.pop(memory_id, None)
            for item in result.updated_objects:
                existing_objects[item.memory_id] = item
            objects_payload = {
                "schema": _OBJECT_STORE_SCHEMA,
                "version": _OBJECT_STORE_VERSION,
                "objects": [item.to_payload() for item in sorted(existing_objects.values(), key=lambda row: row.memory_id)],
            }
            conflicts_payload = {
                "schema": _CONFLICT_STORE_SCHEMA,
                "version": _CONFLICT_STORE_VERSION,
                "conflicts": [
                    item.to_payload()
                    for item in sorted(result.remaining_conflicts, key=lambda row: (row.slot_key, row.candidate_memory_id))
                ],
            }
            _write_json_atomic(self.objects_path, objects_payload)
            _write_json_atomic(self.conflicts_path, conflicts_payload)

    def review_objects(
        self,
        *,
        query_text: str | None = None,
        status: str | None = None,
        kind: str | None = None,
        include_episodes: bool = False,
        limit: int = 12,
    ) -> LongTermMemoryReviewResultV1:
        objects = [
            item
            for item in self.load_objects()
            if (include_episodes or item.kind != "episode")
            and (status is None or item.status == status)
            and (kind is None or item.kind == kind)
        ]
        if not objects:
            return LongTermMemoryReviewResultV1(
                items=(),
                total_count=0,
                query_text=query_text,
                status_filter=status,
                kind_filter=kind,
                include_episodes=include_episodes,
            )
        query_text = _normalize_text(query_text)
        if query_text:
            selector = self._object_selector(objects)
            selected_ids = selector.search(query_text, limit=max(1, limit))
            selected_by_id = {item.memory_id: item for item in objects}
            selected = [selected_by_id[memory_id] for memory_id in selected_ids if memory_id in selected_by_id]
            total_count = len(selected_ids)
        else:
            selected = sorted(
                objects,
                key=lambda item: (item.updated_at.isoformat(), item.memory_id),
                reverse=True,
            )[: max(1, limit)]
            total_count = len(objects)
        return LongTermMemoryReviewResultV1(
            items=tuple(self._to_review_item(item) for item in selected),
            total_count=total_count,
            query_text=query_text,
            status_filter=status,
            kind_filter=kind,
            include_episodes=include_episodes,
        )

    def confirm_object(self, memory_id: str) -> LongTermMemoryMutationResultV1:
        current = self.get_object(memory_id)
        if current is None:
            raise ValueError(f"No long-term memory object found for {memory_id!r}.")
        current_time = _utcnow()
        attrs = dict(current.attributes or {})
        attrs["review_confirmed_by_user"] = True
        attrs["review_confirmed_at"] = current_time.isoformat()
        updated = current.with_updates(
            status="active",
            confirmed_by_user=True,
            confidence=max(current.confidence, 0.99),
            updated_at=current_time,
            attributes=attrs,
        )
        return LongTermMemoryMutationResultV1(
            action="confirm",
            target_memory_id=current.memory_id,
            updated_objects=(updated,),
            remaining_conflicts=self.load_conflicts(),
        )

    def invalidate_object(
        self,
        memory_id: str,
        *,
        reason: str | None = None,
    ) -> LongTermMemoryMutationResultV1:
        current = self.get_object(memory_id)
        if current is None:
            raise ValueError(f"No long-term memory object found for {memory_id!r}.")
        current_time = _utcnow()
        attrs = dict(current.attributes or {})
        attrs["invalidated_by_user"] = True
        if reason:
            attrs["invalidation_reason"] = _normalize_text(reason)
        updated_target = current.with_updates(
            status="invalid",
            confirmed_by_user=True,
            conflicts_with=(),
            updated_at=current_time,
            attributes=attrs,
        )
        related_updates = self._cleanup_references_after_mutation(
            target_memory_id=current.memory_id,
            drop_supersedes=False,
        )
        remaining_conflicts = self._rewrite_conflicts_without_memory(current.memory_id)
        return LongTermMemoryMutationResultV1(
            action="invalidate",
            target_memory_id=current.memory_id,
            updated_objects=tuple(
                sorted((updated_target, *related_updates), key=lambda item: item.memory_id)
            ),
            remaining_conflicts=remaining_conflicts,
        )

    def delete_object(self, memory_id: str) -> LongTermMemoryMutationResultV1:
        current = self.get_object(memory_id)
        if current is None:
            raise ValueError(f"No long-term memory object found for {memory_id!r}.")
        related_updates = self._cleanup_references_after_mutation(
            target_memory_id=current.memory_id,
            drop_supersedes=True,
        )
        remaining_conflicts = self._rewrite_conflicts_without_memory(current.memory_id)
        return LongTermMemoryMutationResultV1(
            action="delete",
            target_memory_id=current.memory_id,
            updated_objects=tuple(sorted(related_updates, key=lambda item: item.memory_id)),
            deleted_memory_ids=(current.memory_id,),
            remaining_conflicts=remaining_conflicts,
        )

    def select_relevant_objects(
        self,
        *,
        query_text: str | None,
        limit: int = 4,
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        objects = tuple(
            item
            for item in self.load_objects()
            if item.kind != "episode" and item.status in {"active", "candidate", "uncertain"}
        )
        if not objects:
            return ()
        clean_query = _normalize_text(query_text)
        if not clean_query:
            return objects[:limit]
        selector = self._object_selector(objects)
        selected_ids = selector.search(clean_query, limit=max(1, limit))
        by_id = {item.memory_id: item for item in objects}
        selected = [by_id[memory_id] for memory_id in selected_ids if memory_id in by_id]
        return tuple(self._filter_query_relevant_objects(clean_query, selected=selected, limit=limit))

    def select_open_conflicts(
        self,
        *,
        query_text: str | None,
        limit: int = 3,
    ) -> tuple[LongTermMemoryConflictV1, ...]:
        conflicts = self.load_conflicts()
        if not conflicts:
            return ()
        clean_query = _normalize_text(query_text)
        if not clean_query:
            return conflicts[:limit]
        objects_by_id = {item.memory_id: item for item in self.load_objects()}
        selector = FullTextSelector(
            tuple(
                FullTextDocument(
                    doc_id=conflict.slot_key,
                    category="conflict",
                    content=self._conflict_search_text(conflict, objects_by_id=objects_by_id),
                )
                for conflict in conflicts
            )
        )
        selected_slots = selector.search(clean_query, limit=max(1, limit), category="conflict")
        by_slot = {item.slot_key: item for item in conflicts}
        return tuple(by_slot[slot_key] for slot_key in selected_slots if slot_key in by_slot)

    def _merge_object(
        self,
        *,
        existing: LongTermMemoryObjectV1 | None,
        incoming: LongTermMemoryObjectV1,
        increment_support: bool,
    ) -> LongTermMemoryObjectV1:
        if existing is None:
            attrs = dict(incoming.attributes or {})
            attrs["support_count"] = max(1, int(attrs.get("support_count", 1)))
            return incoming.with_updates(attributes=attrs)
        attrs = dict(existing.attributes or {})
        incoming_attrs = dict(incoming.attributes or {})
        existing_support = max(1, int(attrs.get("support_count", 1)))
        incoming_support = max(1, int(incoming_attrs.get("support_count", 1)))
        support_count = existing_support + 1 if increment_support else max(existing_support, incoming_support)
        attrs.update(incoming_attrs)
        attrs["support_count"] = support_count
        existing_event_ids = tuple(existing.source.event_ids)
        incoming_event_ids = tuple(incoming.source.event_ids)
        merged_event_ids = tuple(dict.fromkeys((*existing_event_ids, *incoming_event_ids)))
        merged_source = existing.source.to_payload()
        merged_source["event_ids"] = list(merged_event_ids)
        return incoming.with_updates(
            source=merged_source,
            attributes=attrs,
            created_at=min(existing.created_at, incoming.created_at),
            updated_at=max(existing.updated_at, incoming.updated_at),
            confidence=max(existing.confidence, incoming.confidence),
            status=self._preferred_status(existing.status, incoming.status),
        )

    def _preferred_status(self, existing: str, incoming: str) -> str:
        rank = {
            "active": 5,
            "candidate": 4,
            "uncertain": 3,
            "superseded": 2,
            "expired": 1,
            "invalid": 0,
        }
        return existing if rank.get(existing, -1) >= rank.get(incoming, -1) else incoming

    def _rewrite_conflicts_without_memory(
        self,
        memory_id: str,
    ) -> tuple[LongTermMemoryConflictV1, ...]:
        rewritten: list[LongTermMemoryConflictV1] = []
        for conflict in self.load_conflicts():
            if conflict.candidate_memory_id == memory_id:
                continue
            existing_ids = tuple(value for value in conflict.existing_memory_ids if value != memory_id)
            if not existing_ids:
                continue
            rewritten.append(
                LongTermMemoryConflictV1(
                    slot_key=conflict.slot_key,
                    candidate_memory_id=conflict.candidate_memory_id,
                    existing_memory_ids=existing_ids,
                    question=conflict.question,
                    reason=conflict.reason,
                )
            )
        return tuple(rewritten)

    def _cleanup_references_after_mutation(
        self,
        *,
        target_memory_id: str,
        drop_supersedes: bool,
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        updated_objects: list[LongTermMemoryObjectV1] = []
        for item in self.load_objects():
            if item.memory_id == target_memory_id:
                continue
            conflicts_with = tuple(value for value in item.conflicts_with if value != target_memory_id)
            supersedes = (
                tuple(value for value in item.supersedes if value != target_memory_id)
                if drop_supersedes
                else item.supersedes
            )
            if conflicts_with == item.conflicts_with and supersedes == item.supersedes:
                continue
            updated_objects.append(
                item.with_updates(
                    conflicts_with=conflicts_with,
                    supersedes=supersedes,
                    updated_at=_utcnow(),
                )
            )
        return tuple(updated_objects)

    def _to_review_item(self, item: LongTermMemoryObjectV1) -> LongTermMemoryReviewItemV1:
        return LongTermMemoryReviewItemV1(
            memory_id=item.memory_id,
            kind=item.kind,
            summary=item.summary,
            details=item.details,
            status=item.status,
            confidence=item.confidence,
            updated_at=item.updated_at,
            confirmed_by_user=item.confirmed_by_user,
            sensitivity=item.sensitivity,
            slot_key=item.slot_key,
            value_key=item.value_key,
        )

    def _object_selector(
        self,
        objects: tuple[LongTermMemoryObjectV1, ...] | list[LongTermMemoryObjectV1],
    ) -> FullTextSelector:
        return FullTextSelector(
            tuple(
                FullTextDocument(
                    doc_id=item.memory_id,
                    category="object",
                    content=self._object_search_text(item),
                )
                for item in objects
            )
        )

    def _object_search_text(self, item: LongTermMemoryObjectV1) -> str:
        parts = [
            item.kind,
            item.summary,
            item.details or "",
        ]
        for key, value in (item.attributes or {}).items():
            if key in _NON_SEMANTIC_ATTRIBUTE_KEYS:
                continue
            parts.append(key)
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, (list, tuple)):
                parts.extend(str(entry) for entry in value if isinstance(entry, str))
        return _normalize_text(" ".join(part for part in parts if part))

    def _filter_query_relevant_objects(
        self,
        query_text: str,
        *,
        selected: list[LongTermMemoryObjectV1],
        limit: int,
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        query_terms = set(retrieval_terms(query_text))
        semantic_query_terms = {term for term in query_terms if not term.isdigit()}
        if not semantic_query_terms:
            return tuple(selected[: max(1, limit)])
        filtered = [
            item
            for item in selected
            if semantic_query_terms.intersection(retrieval_terms(self._object_search_text(item)))
        ]
        return tuple(filtered[: max(1, limit)])

    def _conflict_search_text(
        self,
        conflict: LongTermMemoryConflictV1,
        *,
        objects_by_id: dict[str, LongTermMemoryObjectV1],
    ) -> str:
        related_parts = [
            conflict.slot_key,
            conflict.question,
            conflict.reason,
        ]
        candidate = objects_by_id.get(conflict.candidate_memory_id)
        if candidate is not None:
            related_parts.append(self._object_search_text(candidate))
        for memory_id in conflict.existing_memory_ids:
            existing = objects_by_id.get(memory_id)
            if existing is not None:
                related_parts.append(self._object_search_text(existing))
        return _normalize_text(" ".join(part for part in related_parts if part))


__all__ = ["LongTermStructuredStore"]
