"""Mutation, snapshot-write, and review flows for structured memory state."""


from __future__ import annotations

from twinr.memory.longterm.core.models import (
    LongTermConflictResolutionV1,
    LongTermConsolidationResultV1,
    LongTermMemoryMutationResultV1,
    LongTermMemoryReviewResultV1,
    LongTermMemoryConflictV1,
    LongTermMemoryObjectV1,
    LongTermReflectionResultV1,
    LongTermRetentionResultV1,
)

from .shared import (
    _ARCHIVE_STORE_SCHEMA,
    _ARCHIVE_STORE_VERSION,
    _CONFLICT_STORE_SCHEMA,
    _CONFLICT_STORE_VERSION,
    _OBJECT_STORE_SCHEMA,
    _OBJECT_STORE_VERSION,
    _coerce_aware_utc,
    _normalize_text,
    _utcnow,
)


class StructuredStoreMutationMixin:
    """Own review, mutation result generation, and snapshot-write flows."""

    def apply_consolidation(self, result: LongTermConsolidationResultV1) -> None:
        """Persist a consolidation result into object and conflict snapshots."""

        with self._lock:
            existing_objects = {item.memory_id: item for item in self.load_objects_fine_grained_for_write()}
            for item in (*result.episodic_objects, *result.durable_objects, *result.deferred_objects):
                existing = existing_objects.get(item.memory_id)
                existing_objects[item.memory_id] = self._merge_object(
                    existing=existing,
                    incoming=item,
                    increment_support=True,
                )
            existing_conflicts = {self._conflict_key(item): item for item in self.load_conflicts_fine_grained_for_write()}
            for conflict in result.conflicts:
                existing_conflicts[self._conflict_key(conflict)] = conflict
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
            self._persist_snapshot_payload(
                snapshot_kind="objects",
                local_path=self.objects_path,
                payload=objects_payload,
            )
            self._persist_snapshot_payload(
                snapshot_kind="conflicts",
                local_path=self.conflicts_path,
                payload=conflicts_payload,
            )

    def apply_reflection(self, result: LongTermReflectionResultV1) -> None:
        """Persist reflected objects and summaries into the object snapshot."""

        with self._lock:
            target_ids = tuple(
                dict.fromkeys(
                    item.memory_id
                    for item in (*result.reflected_objects, *result.created_summaries)
                    if isinstance(item, LongTermMemoryObjectV1)
                )
            )
            existing_objects = {item.memory_id: item for item in self.load_objects_by_ids(target_ids)}
            for item in (*result.reflected_objects, *result.created_summaries):
                existing_objects[item.memory_id] = self._merge_object(
                    existing=existing_objects.get(item.memory_id),
                    incoming=item,
                    increment_support=False,
                )
            self.commit_active_delta(
                object_upserts=tuple(
                    sorted(existing_objects.values(), key=lambda row: row.memory_id)
                ),
            )

    def apply_retention(self, result: LongTermRetentionResultV1) -> None:
        """Persist kept objects and archive retained-off objects."""

        with self._lock:
            archived_ids = tuple(item.memory_id for item in result.archived_objects)
            expired_ids = tuple(item.memory_id for item in result.expired_objects)
            if self._remote_catalog_enabled() and self._remote_catalog is not None:
                self.commit_active_delta(
                    object_upserts=result.kept_objects,
                    object_delete_ids=tuple(dict.fromkeys((*expired_ids, *archived_ids, *result.pruned_memory_ids))),
                    archive_upserts=result.archived_objects,
                )
                return
            archived_objects = {item.memory_id: item for item in self.load_archived_objects_fine_grained()}
            for item in result.archived_objects:
                archived_objects[item.memory_id] = item
            payload = {
                "schema": _OBJECT_STORE_SCHEMA,
                "version": _OBJECT_STORE_VERSION,
                "objects": [item.to_payload() for item in sorted(result.kept_objects, key=lambda row: row.memory_id)],
            }
            archive_payload = {
                "schema": _ARCHIVE_STORE_SCHEMA,
                "version": _ARCHIVE_STORE_VERSION,
                "objects": [item.to_payload() for item in sorted(archived_objects.values(), key=lambda row: row.memory_id)],
            }
            self._persist_snapshot_payload(
                snapshot_kind="objects",
                local_path=self.objects_path,
                payload=payload,
            )
            self._persist_snapshot_payload(
                snapshot_kind="archive",
                local_path=self.archive_path,
                payload=archive_payload,
            )

    def apply_conflict_resolution(self, result: LongTermConflictResolutionV1) -> None:
        """Persist updated objects and the remaining conflict queue."""

        with self._lock:
            self.commit_active_delta(
                object_upserts=tuple(sorted(result.updated_objects, key=lambda row: row.memory_id)),
                conflict_upserts=tuple(
                    sorted(result.remaining_conflicts, key=lambda row: (row.slot_key, row.candidate_memory_id))
                ),
                conflict_delete_slot_keys=result.deleted_conflict_slot_keys,
            )

    def write_snapshot(
        self,
        *,
        objects: tuple[LongTermMemoryObjectV1, ...],
        conflicts: tuple[LongTermMemoryConflictV1, ...] = (),
        archived_objects: tuple[LongTermMemoryObjectV1, ...] = (),
    ) -> None:
        """Write complete object, conflict, and archive snapshots at once."""

        with self._lock:
            objects_payload = {
                "schema": _OBJECT_STORE_SCHEMA,
                "version": _OBJECT_STORE_VERSION,
                "objects": [item.to_payload() for item in sorted(objects, key=lambda row: row.memory_id)],
            }
            conflicts_payload = {
                "schema": _CONFLICT_STORE_SCHEMA,
                "version": _CONFLICT_STORE_VERSION,
                "conflicts": [
                    item.to_payload()
                    for item in sorted(conflicts, key=lambda row: (row.slot_key, row.candidate_memory_id))
                ],
            }
            archive_payload = {
                "schema": _ARCHIVE_STORE_SCHEMA,
                "version": _ARCHIVE_STORE_VERSION,
                "objects": [item.to_payload() for item in sorted(archived_objects, key=lambda row: row.memory_id)],
            }
            self._persist_snapshot_payload(
                snapshot_kind="objects",
                local_path=self.objects_path,
                payload=objects_payload,
            )
            self._persist_snapshot_payload(
                snapshot_kind="conflicts",
                local_path=self.conflicts_path,
                payload=conflicts_payload,
            )
            self._persist_snapshot_payload(
                snapshot_kind="archive",
                local_path=self.archive_path,
                payload=archive_payload,
            )

    def apply_memory_mutation(self, result: LongTermMemoryMutationResultV1) -> None:
        """Persist a user-driven mutation result across all snapshots."""

        with self._lock:
            archive_delete_ids = tuple(
                dict.fromkeys(
                    (
                        *result.deleted_memory_ids,
                        *(item.memory_id for item in result.updated_objects),
                    )
                )
            )
            self.commit_active_delta(
                object_upserts=tuple(sorted(result.updated_objects, key=lambda row: row.memory_id)),
                object_delete_ids=result.deleted_memory_ids,
                conflict_upserts=tuple(
                    sorted(result.remaining_conflicts, key=lambda row: (row.slot_key, row.candidate_memory_id))
                ),
                conflict_delete_slot_keys=result.deleted_conflict_slot_keys,
                archive_delete_ids=archive_delete_ids,
            )

    def review_objects(
        self,
        *,
        query_text: str | None = None,
        status: str | None = None,
        kind: str | None = None,
        include_episodes: bool = False,
        limit: int = 12,
    ) -> LongTermMemoryReviewResultV1:
        """Build a bounded review page over stored memory objects."""

        with self._lock:
            bounded_limit = max(1, limit)
            remote_selected = self.select_review_objects_query_first(
                query_text=query_text,
                status=status,
                kind=kind,
                include_episodes=include_episodes,
                limit=bounded_limit,
            )
            if remote_selected is not None:
                selected, total_count = remote_selected
                return LongTermMemoryReviewResultV1(
                    items=tuple(self._to_review_item(item) for item in selected),
                    total_count=total_count,
                    query_text=_normalize_text(query_text),
                    status_filter=status,
                    kind_filter=kind,
                    include_episodes=include_episodes,
                )
            objects = [
                item
                for item in self.load_objects_fine_grained()
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
                all_selected_ids = selector.search(query_text, limit=len(objects))
                selected_by_id = {item.memory_id: item for item in objects}
                ordered_selected = [selected_by_id[memory_id] for memory_id in all_selected_ids if memory_id in selected_by_id]
                selected = ordered_selected[:bounded_limit]
                total_count = len(ordered_selected)
            else:
                selected = sorted(
                    objects,
                    key=lambda item: (_coerce_aware_utc(item.updated_at), item.memory_id),
                    reverse=True,
                )[:bounded_limit]
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
        """Build a mutation result that confirms one stored object."""

        with self._lock:
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
                remaining_conflicts=(),
            )

    def invalidate_object(
        self,
        memory_id: str,
        *,
        reason: str | None = None,
    ) -> LongTermMemoryMutationResultV1:
        """Build a mutation result that invalidates one stored object."""

        with self._lock:
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
            remaining_conflicts, deleted_conflict_slot_keys = self._conflict_delta_without_memory(current.memory_id)
            return LongTermMemoryMutationResultV1(
                action="invalidate",
                target_memory_id=current.memory_id,
                updated_objects=tuple(
                    sorted((updated_target, *related_updates), key=lambda item: item.memory_id)
                ),
                remaining_conflicts=remaining_conflicts,
                deleted_conflict_slot_keys=deleted_conflict_slot_keys,
            )

    def delete_object(self, memory_id: str) -> LongTermMemoryMutationResultV1:
        """Build a mutation result that removes one stored object."""

        with self._lock:
            current = self.get_object(memory_id)
            if current is None:
                raise ValueError(f"No long-term memory object found for {memory_id!r}.")
            related_updates = self._cleanup_references_after_mutation(
                target_memory_id=current.memory_id,
                drop_supersedes=True,
            )
            remaining_conflicts, deleted_conflict_slot_keys = self._conflict_delta_without_memory(current.memory_id)
            return LongTermMemoryMutationResultV1(
                action="delete",
                target_memory_id=current.memory_id,
                updated_objects=tuple(sorted(related_updates, key=lambda item: item.memory_id)),
                deleted_memory_ids=(current.memory_id,),
                remaining_conflicts=remaining_conflicts,
                deleted_conflict_slot_keys=deleted_conflict_slot_keys,
            )
