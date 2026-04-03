"""Selective working-set loading and delta commits for active structured-memory flows.

The active runtime paths should avoid hydrating and rewriting the complete
objects/conflicts/archive state when they only touch a bounded subset of
current long-term memory. This mixin provides two capabilities:

- load a bounded current working set for one persistence/backfill batch
- publish only the changed item documents plus an updated current head

Local-only stores still fall back to full local snapshot merges because they do
not have the remote catalog current-head contract available.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import inspect

from twinr.memory.chonkydb.models import ChonkyDBRecordItem
from twinr.memory.longterm.core.models import LongTermMemoryConflictV1, LongTermMemoryObjectV1
from twinr.memory.longterm.storage.remote_catalog import LongTermRemoteCatalogEntry
from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError

from .shared import (
    _ARCHIVE_STORE_SCHEMA,
    _ARCHIVE_STORE_VERSION,
    _CONFLICT_STORE_SCHEMA,
    _CONFLICT_STORE_VERSION,
    _OBJECT_STORE_SCHEMA,
    _OBJECT_STORE_VERSION,
    _normalize_text,
    _utcnow,
)


@dataclass(frozen=True, slots=True)
class StructuredStoreActiveWorkingSet:
    """Hold the bounded active object/conflict/archive state for one batch."""

    objects: tuple[LongTermMemoryObjectV1, ...]
    conflicts: tuple[LongTermMemoryConflictV1, ...]
    archived_objects: tuple[LongTermMemoryObjectV1, ...]


class StructuredStoreActiveDeltaMixin:
    """Own active-path selective loads and delta-style current-head commits."""

    def load_active_working_set(
        self,
        *,
        candidate_objects: Iterable[LongTermMemoryObjectV1] = (),
        event_ids: Iterable[str] = (),
    ) -> StructuredStoreActiveWorkingSet:
        """Load the bounded current state touched by one active persistence batch."""

        candidates = tuple(item for item in candidate_objects if isinstance(item, LongTermMemoryObjectV1))
        if not candidates:
            return StructuredStoreActiveWorkingSet(objects=(), conflicts=(), archived_objects=())
        if not self._remote_catalog_enabled() or self._remote_catalog is None:
            state = self.load_current_state_fine_grained_for_write()
            return StructuredStoreActiveWorkingSet(
                objects=tuple(state.objects),
                conflicts=tuple(state.conflicts),
                archived_objects=tuple(state.archived_objects),
            )

        selected_objects: dict[str, LongTermMemoryObjectV1] = {}

        def add_objects(items: Iterable[LongTermMemoryObjectV1]) -> None:
            for item in items:
                if not isinstance(item, LongTermMemoryObjectV1):
                    continue
                memory_id = _normalize_text(item.memory_id)
                if memory_id and memory_id not in selected_objects:
                    selected_objects[memory_id] = item

        normalized_event_ids = tuple(
            dict.fromkeys(
                normalized
                for normalized in (
                    _normalize_text(value)
                    for value in (
                        *event_ids,
                        *(event_id for item in candidates for event_id in item.source.event_ids),
                    )
                )
                if normalized
            )
        )
        target_memory_ids = tuple(
            dict.fromkeys(
                normalized
                for normalized in (
                    _normalize_text(value)
                    for value in (
                        *(item.memory_id for item in candidates),
                        *(memory_id for item in candidates for memory_id in item.conflicts_with),
                        *(memory_id for item in candidates for memory_id in item.supersedes),
                    )
                )
                if normalized
            )
        )
        target_slot_keys = tuple(
            dict.fromkeys(
                normalized
                for normalized in (_normalize_text(item.slot_key) for item in candidates)
                if normalized
            )
        )

        add_objects(self.load_objects_by_ids(target_memory_ids))
        add_objects(self.load_objects_by_slot_keys(target_slot_keys))
        add_objects(self.load_objects_by_event_ids(normalized_event_ids))

        related_memory_ids = tuple(selected_objects)
        if related_memory_ids:
            add_objects(self.load_objects_referencing_memory_ids(related_memory_ids))

        selected_conflicts: dict[str, LongTermMemoryConflictV1] = {}
        for conflict in self.load_conflicts_by_slot_keys(target_slot_keys):
            selected_conflicts[conflict.slot_key] = conflict
        for conflict in self.load_conflicts_for_memory_ids(tuple(selected_objects)):
            selected_conflicts[conflict.slot_key] = conflict

        related_conflict_ids = tuple(
            dict.fromkeys(
                memory_id
                for conflict in selected_conflicts.values()
                for memory_id in (conflict.candidate_memory_id, *conflict.existing_memory_ids)
                if isinstance(memory_id, str) and memory_id
            )
        )
        if related_conflict_ids:
            add_objects(self.load_objects_by_ids(related_conflict_ids))

        return StructuredStoreActiveWorkingSet(
            objects=tuple(sorted(selected_objects.values(), key=lambda item: item.memory_id)),
            conflicts=tuple(
                sorted(
                    selected_conflicts.values(),
                    key=lambda item: (item.slot_key, item.candidate_memory_id),
                )
            ),
            archived_objects=(),
        )

    def commit_active_delta(
        self,
        *,
        object_upserts: Iterable[LongTermMemoryObjectV1] = (),
        object_delete_ids: Iterable[str] = (),
        conflict_upserts: Iterable[LongTermMemoryConflictV1] = (),
        conflict_delete_slot_keys: Iterable[str] = (),
        archive_upserts: Iterable[LongTermMemoryObjectV1] = (),
        archive_delete_ids: Iterable[str] = (),
    ) -> None:
        """Persist only the changed active current-state items for one batch."""

        with self._lock:
            object_payloads = {
                item.memory_id: item.to_payload()
                for item in object_upserts
                if isinstance(item, LongTermMemoryObjectV1) and _normalize_text(item.memory_id)
            }
            archive_payloads = {
                item.memory_id: item.to_payload()
                for item in archive_upserts
                if isinstance(item, LongTermMemoryObjectV1) and _normalize_text(item.memory_id)
            }
            conflict_items = tuple(
                item
                for item in conflict_upserts
                if isinstance(item, LongTermMemoryConflictV1)
            )
            conflict_payloads = {
                self._conflict_doc_id(item): item.to_payload()
                for item in conflict_items
            }
            normalized_object_delete_ids = tuple(
                dict.fromkeys(
                    normalized
                    for normalized in (_normalize_text(value) for value in object_delete_ids)
                    if normalized
                )
            )
            normalized_archive_delete_ids = tuple(
                dict.fromkeys(
                    normalized
                    for normalized in (_normalize_text(value) for value in archive_delete_ids)
                    if normalized
                )
            )
            normalized_conflict_delete_slots = tuple(
                dict.fromkeys(
                    normalized
                    for normalized in (_normalize_text(value) for value in conflict_delete_slot_keys)
                    if normalized
                )
            )

            remote_catalog = self._remote_catalog
            if not self._remote_catalog_enabled() or remote_catalog is None:
                current_state = self.load_current_state_fine_grained_for_write()
                objects_by_id = {item.memory_id: item for item in current_state.objects}
                conflicts_by_slot = {item.slot_key: item for item in current_state.conflicts}
                archived_by_id = {item.memory_id: item for item in current_state.archived_objects}
                for memory_id in normalized_object_delete_ids:
                    objects_by_id.pop(memory_id, None)
                for item in object_upserts:
                    if isinstance(item, LongTermMemoryObjectV1):
                        objects_by_id[item.memory_id] = item
                for slot_key in normalized_conflict_delete_slots:
                    conflicts_by_slot.pop(slot_key, None)
                for item in conflict_items:
                    conflicts_by_slot[item.slot_key] = item
                for memory_id in normalized_archive_delete_ids:
                    archived_by_id.pop(memory_id, None)
                for item in archive_upserts:
                    if isinstance(item, LongTermMemoryObjectV1):
                        archived_by_id[item.memory_id] = item
                self._persist_snapshot_payload(
                    snapshot_kind="objects",
                    local_path=self.objects_path,
                    payload={
                        "schema": _OBJECT_STORE_SCHEMA,
                        "version": _OBJECT_STORE_VERSION,
                        "objects": [
                            item.to_payload()
                            for item in sorted(objects_by_id.values(), key=lambda row: row.memory_id)
                        ],
                    },
                )
                self._persist_snapshot_payload(
                    snapshot_kind="conflicts",
                    local_path=self.conflicts_path,
                    payload={
                        "schema": _CONFLICT_STORE_SCHEMA,
                        "version": _CONFLICT_STORE_VERSION,
                        "conflicts": [
                            item.to_payload()
                            for item in sorted(
                                conflicts_by_slot.values(),
                                key=lambda row: (row.slot_key, row.candidate_memory_id),
                            )
                        ],
                    },
                )
                self._persist_snapshot_payload(
                    snapshot_kind="archive",
                    local_path=self.archive_path,
                    payload={
                        "schema": _ARCHIVE_STORE_SCHEMA,
                        "version": _ARCHIVE_STORE_VERSION,
                        "objects": [
                            item.to_payload()
                            for item in sorted(archived_by_id.values(), key=lambda row: row.memory_id)
                        ],
                    },
                )
                return

            written_at = _utcnow().isoformat()
            self._commit_remote_collection_delta(
                snapshot_kind="objects",
                upsert_payloads=object_payloads,
                delete_item_ids=normalized_object_delete_ids,
                written_at=written_at,
            )
            self._commit_remote_collection_delta(
                snapshot_kind="conflicts",
                upsert_payloads=conflict_payloads,
                delete_item_ids=(),
                delete_slot_keys=normalized_conflict_delete_slots,
                written_at=written_at,
            )
            self._commit_remote_collection_delta(
                snapshot_kind="archive",
                upsert_payloads=archive_payloads,
                delete_item_ids=normalized_archive_delete_ids,
                written_at=written_at,
            )
            self._recent_local_snapshot_payloads.pop("objects", None)
            self._recent_local_snapshot_payloads.pop("conflicts", None)
            self._recent_local_snapshot_payloads.pop("archive", None)

    def _commit_remote_collection_delta(
        self,
        *,
        snapshot_kind: str,
        upsert_payloads: Mapping[str, Mapping[str, object]],
        delete_item_ids: Iterable[str] = (),
        delete_slot_keys: Iterable[str] = (),
        written_at: str,
    ) -> None:
        """Publish one changed current-head collection without hydrating unchanged payloads."""

        remote_catalog = self._remote_catalog
        if remote_catalog is None:
            raise RuntimeError("Fine-grained remote catalog store is required for active delta commits.")
        projection_complete_remote_write = snapshot_kind in {"objects", "conflicts", "archive"}
        definition = remote_catalog._require_definition(snapshot_kind)
        existing_entries = self._load_remote_collection_entries_for_active_delta(
            remote_catalog=remote_catalog,
            snapshot_kind=snapshot_kind,
        )
        delete_ids = {
            normalized
            for normalized in (_normalize_text(value) for value in delete_item_ids)
            if normalized
        }
        delete_slots = {
            normalized
            for normalized in (_normalize_text(value) for value in delete_slot_keys)
            if normalized
        }
        if delete_slots:
            delete_ids.update(
                entry.item_id
                for entry in existing_entries.values()
                if self._catalog_entry_text(entry, "slot_key") in delete_slots
            )
        for item_id in delete_ids:
            existing_entries.pop(item_id, None)
        if delete_ids:
            with remote_catalog._cache_lock:
                remote_catalog._item_payload_cache = {
                    cache_key: cached
                    for cache_key, cached in remote_catalog._item_payload_cache.items()
                    if not (cache_key[0] == snapshot_kind and cache_key[1] in delete_ids)
                }

        if upsert_payloads:
            write_client = remote_catalog._require_client(
                getattr(remote_catalog._require_remote_state(), "write_client", None),
                operation="write",
            )
            record_items: list[ChonkyDBRecordItem] = []
            staged: list[tuple[str, dict[str, object], Mapping[str, object]]] = []
            for item_id, raw_payload in upsert_payloads.items():
                normalized_item_id = _normalize_text(item_id)
                if not normalized_item_id:
                    continue
                payload = dict(raw_payload)
                payload_sha256 = remote_catalog._payload_sha256(payload)
                user_metadata = self._remote_item_metadata(snapshot_kind=snapshot_kind, payload=payload)
                catalog_metadata = remote_catalog._catalog_entry_metadata(
                    payload_sha256=payload_sha256,
                    user_metadata=user_metadata,
                )
                existing_entry = existing_entries.get(normalized_item_id)
                uri = remote_catalog.item_uri(snapshot_kind=snapshot_kind, item_id=normalized_item_id)
                if remote_catalog._can_reuse_existing_entry(
                    snapshot_kind=snapshot_kind,
                    entry=existing_entry,
                    item_id=normalized_item_id,
                    payload_sha256=payload_sha256,
                ):
                    existing_entries[normalized_item_id] = LongTermRemoteCatalogEntry(
                        snapshot_kind=snapshot_kind,
                        item_id=normalized_item_id,
                        document_id=None if existing_entry is None else existing_entry.document_id,
                        uri=uri,
                        metadata=dict(catalog_metadata),
                    )
                    remote_catalog._store_item_payload(
                        snapshot_kind=snapshot_kind,
                        item_id=normalized_item_id,
                        payload=payload,
                    )
                    continue
                metadata = remote_catalog._build_item_record_metadata(
                    definition=definition,
                    item_id=normalized_item_id,
                    payload=payload,
                    payload_sha256=payload_sha256,
                )
                if isinstance(user_metadata, Mapping):
                    for key, value in user_metadata.items():
                        normalized_key = str(key)
                        if value is None or normalized_key in remote_catalog._catalog_entry_object_fields():
                            continue
                        metadata[normalized_key] = value
                content_text = self._remote_item_search_text(snapshot_kind=snapshot_kind, payload=payload)
                record_items.append(
                    ChonkyDBRecordItem(
                        payload=remote_catalog._build_item_record_payload(
                            definition=definition,
                            item_id=normalized_item_id,
                            payload=payload,
                            content_text=content_text,
                        ),
                        metadata=metadata,
                        content=content_text,
                        uri=uri,
                        target_indexes=remote_catalog._searchable_write_target_indexes(),
                        enable_chunking=False,
                        include_insights_in_response=False,
                    )
                )
                staged.append((normalized_item_id, dict(catalog_metadata), payload))
            document_ids = remote_catalog._store_record_items(
                write_client,
                snapshot_kind=snapshot_kind,
                record_items=record_items,
                skip_async_document_id_wait=projection_complete_remote_write,
            )
            for index, (item_id, metadata, payload) in enumerate(staged):
                existing_entries[item_id] = LongTermRemoteCatalogEntry(
                    snapshot_kind=snapshot_kind,
                    item_id=item_id,
                    document_id=document_ids[index] if index < len(document_ids) else None,
                    uri=remote_catalog.item_uri(snapshot_kind=snapshot_kind, item_id=item_id),
                    metadata=dict(metadata),
                )
                remote_catalog._store_item_payload(
                    snapshot_kind=snapshot_kind,
                    item_id=item_id,
                    payload=payload,
                )

        ordered_entries = tuple(
            sorted(
                existing_entries.values(),
                key=lambda entry: entry.item_id,
            )
        )
        remote_catalog.persist_catalog_entries(
            snapshot_kind=snapshot_kind,
            entries=ordered_entries,
            written_at=written_at,
            skip_async_document_id_wait=projection_complete_remote_write,
        )

    @staticmethod
    def _load_remote_collection_entries_for_active_delta(
        *,
        remote_catalog: object,
        snapshot_kind: str,
    ) -> dict[str, LongTermRemoteCatalogEntry]:
        """Load reusable current entries, treating a proven-missing head as empty.

        Active-delta writes can be the first structured-memory write on a fresh
        namespace. The ordinary current-head load keeps remote failures
        fail-closed, but a fast probe may still prove that the fixed
        ``catalog/current`` document is genuinely absent rather than unhealthy.
        In that narrow case, start from an empty collection instead of aborting
        the first live delta write.
        """

        try:
            load_entries = getattr(remote_catalog, "_load_catalog_entries_for_write")
            entries = load_entries(snapshot_kind=snapshot_kind)
        except LongTermRemoteUnavailableError:
            probe_catalog_payload_result = getattr(remote_catalog, "probe_catalog_payload_result", None)
            if not callable(probe_catalog_payload_result):
                raise
            probe_kwargs: dict[str, object] = {"snapshot_kind": snapshot_kind}
            try:
                parameters = inspect.signature(probe_catalog_payload_result).parameters
            except (TypeError, ValueError):
                parameters = {}
            if "fast_fail" in parameters:
                probe_kwargs["fast_fail"] = True
            head_status, _payload = probe_catalog_payload_result(**probe_kwargs)
            if _normalize_text(head_status) == "not_found":
                entries = ()
            else:
                raise
        return {
            entry.item_id: entry
            for entry in entries
            if isinstance(entry, LongTermRemoteCatalogEntry)
        }
