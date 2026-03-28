"""Remote-aware selectors for episodic objects, durable objects, and conflicts."""

# mypy: disable-error-code=attr-defined

from __future__ import annotations

from collections.abc import Iterable, Mapping

from twinr.memory.longterm.core.models import LongTermMemoryConflictV1, LongTermMemoryObjectV1

from .shared import _LOG, _coerce_aware_utc, _normalize_text


class StructuredStoreRemoteSelectionMixin:
    """Own remote-first selection helpers for structured-store queries."""

    def _load_remote_objects_from_entries(
        self,
        *,
        entries: Iterable[object],
        snapshot_kind: str,
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        remote_catalog = self._remote_catalog
        if remote_catalog is None:
            return ()
        item_ids = [
            entry.item_id
            for entry in entries
            if hasattr(entry, "item_id") and isinstance(getattr(entry, "item_id"), str)
        ]
        payloads = remote_catalog.load_item_payloads(snapshot_kind=snapshot_kind, item_ids=item_ids)
        loaded: list[LongTermMemoryObjectV1] = []
        for payload in payloads:
            try:
                loaded.append(LongTermMemoryObjectV1.from_payload(payload))
            except Exception:
                _LOG.warning("Skipping invalid remote long-term object payload during selective load.", exc_info=True)
        by_id = {item.memory_id: item for item in loaded}
        return tuple(by_id[item_id] for item_id in item_ids if item_id in by_id)

    def _load_remote_objects_from_payloads(
        self,
        *,
        payloads: Iterable[Mapping[str, object]],
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        """Parse already-materialized remote object payloads."""

        loaded: list[LongTermMemoryObjectV1] = []
        for payload in payloads:
            try:
                loaded.append(LongTermMemoryObjectV1.from_payload(dict(payload)))
            except Exception:
                _LOG.warning("Skipping invalid remote long-term object payload during direct scope search.", exc_info=True)
        return tuple(loaded)

    def _remote_select_objects(
        self,
        *,
        query_text: str | None,
        limit: int,
        include_episodes: bool,
        fallback_limit: int,
        require_query_match: bool,
    ) -> tuple[LongTermMemoryObjectV1, ...] | None:
        remote_catalog = self._remote_catalog
        if not self._remote_catalog_enabled() or remote_catalog is None:
            return None
        bounded_limit = max(1, limit)
        clean_query = _normalize_text(query_text)

        def eligible(entry: object) -> bool:
            metadata = getattr(entry, "metadata", None)
            if not isinstance(metadata, Mapping):
                return False
            kind = _normalize_text(metadata.get("kind") if isinstance(metadata.get("kind"), str) else None)
            status = _normalize_text(metadata.get("status") if isinstance(metadata.get("status"), str) else None)
            if status not in {"active", "candidate", "uncertain"}:
                return False
            if include_episodes:
                return kind == "episode"
            return kind != "episode"

        if not clean_query:
            if require_query_match:
                return ()
            entries = remote_catalog.top_catalog_entries(
                snapshot_kind="objects",
                limit=bounded_limit if fallback_limit <= 0 else min(bounded_limit, max(1, fallback_limit)),
                eligible=eligible,
            )
            return self._load_remote_objects_from_entries(entries=entries, snapshot_kind="objects")

        try:
            direct_payloads = remote_catalog.search_current_item_payloads(
                snapshot_kind="objects",
                query_text=clean_query,
                limit=bounded_limit,
                eligible=eligible,
                allow_catalog_fallback=False,
            )
        except Exception:
            if self._remote_is_required():
                raise
            direct_payloads = None
        if direct_payloads is not None:
            if direct_payloads:
                selected = list(self._load_remote_objects_from_payloads(payloads=direct_payloads))
                filtered = list(self._filter_query_relevant_objects(clean_query, selected=selected, limit=bounded_limit))
                if filtered:
                    return self.rank_selected_objects(
                        query_texts=(clean_query,),
                        objects=filtered,
                        limit=bounded_limit,
                    )
                return ()
            if include_episodes:
                return ()

        try:
            if not remote_catalog.catalog_available(snapshot_kind="objects"):
                return None
        except Exception:
            if self._remote_is_required():
                raise
            return None

        entries = remote_catalog.search_catalog_entries(
            snapshot_kind="objects",
            query_text=clean_query,
            limit=bounded_limit,
            eligible=eligible,
        )
        selected = list(self._load_remote_objects_from_entries(entries=entries, snapshot_kind="objects"))
        filtered = list(self._filter_query_relevant_objects(clean_query, selected=selected, limit=bounded_limit))
        if filtered:
            return self.rank_selected_objects(
                query_texts=(clean_query,),
                objects=filtered,
                limit=bounded_limit,
            )
        if require_query_match or fallback_limit <= 0:
            return ()
        fallback_entries = remote_catalog.top_catalog_entries(
            snapshot_kind="objects",
            limit=min(bounded_limit, max(1, fallback_limit)),
            eligible=eligible,
        )
        return self._load_remote_objects_from_entries(entries=fallback_entries, snapshot_kind="objects")

    def _select_relevant_objects_from_loaded(
        self,
        *,
        objects: Iterable[LongTermMemoryObjectV1],
        query_text: str | None,
        include_episodes: bool,
        limit: int,
        require_query_match: bool,
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        """Search one already-loaded object pool without re-entering remote scope lookup."""

        bounded_limit = max(1, limit)
        eligible_objects = tuple(
            sorted(
                (
                    item
                    for item in objects
                    if item.status in {"active", "candidate", "uncertain"}
                    and ((item.kind == "episode") if include_episodes else (item.kind != "episode"))
                ),
                key=lambda item: (
                    _coerce_aware_utc(item.updated_at),
                    _coerce_aware_utc(item.created_at),
                    item.memory_id,
                ),
                reverse=True,
            )
        )
        if not eligible_objects:
            return ()
        clean_query = _normalize_text(query_text)
        if not clean_query:
            if require_query_match:
                return ()
            return eligible_objects[:bounded_limit]
        selector = self._object_selector(eligible_objects)
        selected_ids = selector.search(clean_query, limit=bounded_limit)
        by_id = {item.memory_id: item for item in eligible_objects}
        selected = [by_id[memory_id] for memory_id in selected_ids if memory_id in by_id]
        filtered = list(self._filter_query_relevant_objects(clean_query, selected=selected, limit=bounded_limit))
        if not filtered:
            return ()
        return self.rank_selected_objects(
            query_texts=(clean_query,),
            objects=filtered,
            limit=bounded_limit,
        )

    def _remote_select_conflicts(
        self,
        *,
        query_text: str | None,
        limit: int,
    ) -> tuple[LongTermMemoryConflictV1, ...] | None:
        remote_catalog = self._remote_catalog
        if not self._remote_catalog_enabled() or remote_catalog is None:
            return None
        bounded_limit = max(1, limit)
        clean_query = _normalize_text(query_text)
        try:
            if remote_catalog.catalog_item_count(snapshot_kind="conflicts") == 0:
                return ()
        except Exception:
            if self._remote_is_required():
                raise
            return None
        if not clean_query:
            try:
                if not remote_catalog.catalog_available(snapshot_kind="conflicts"):
                    return None
            except Exception:
                if self._remote_is_required():
                    raise
                return None
            entries = remote_catalog.top_catalog_entries(
                snapshot_kind="conflicts",
                limit=bounded_limit,
                preserve_order=True,
            )
            payloads = remote_catalog.load_item_payloads(
                snapshot_kind="conflicts",
                item_ids=(entry.item_id for entry in entries),
            )
        else:
            try:
                direct_payloads = remote_catalog.search_current_item_payloads(
                    snapshot_kind="conflicts",
                    query_text=clean_query,
                    limit=bounded_limit,
                    allow_catalog_fallback=False,
                )
            except Exception:
                if self._remote_is_required():
                    raise
                direct_payloads = None
            if direct_payloads:
                payloads = direct_payloads
            else:
                try:
                    if not remote_catalog.catalog_available(snapshot_kind="conflicts"):
                        return None
                except Exception:
                    if self._remote_is_required():
                        raise
                    return None
                entries = remote_catalog.search_catalog_entries(
                    snapshot_kind="conflicts",
                    query_text=clean_query,
                    limit=bounded_limit,
                )
                payloads = remote_catalog.load_item_payloads(
                    snapshot_kind="conflicts",
                    item_ids=(entry.item_id for entry in entries),
                )
        conflicts: list[LongTermMemoryConflictV1] = []
        for payload in payloads:
            try:
                conflicts.append(LongTermMemoryConflictV1.from_payload(payload))
            except Exception:
                _LOG.warning("Skipping invalid remote long-term conflict payload during selective load.", exc_info=True)
        if not clean_query:
            return tuple(conflicts[:bounded_limit])
        filtered_without_objects = self._filter_query_relevant_conflicts(
            clean_query,
            selected=conflicts,
            limit=bounded_limit,
        )
        if filtered_without_objects or not conflicts:
            return filtered_without_objects
        related_ids = tuple(
            dict.fromkeys(
                memory_id
                for conflict in conflicts
                for memory_id in (conflict.candidate_memory_id, *conflict.existing_memory_ids)
                if isinstance(memory_id, str) and memory_id
            )
        )
        objects_by_id: dict[str, LongTermMemoryObjectV1] = {}
        if related_ids:
            try:
                objects_by_id = {item.memory_id: item for item in self.load_objects_by_ids(related_ids)}
            except Exception:
                if self._remote_is_required():
                    raise
                objects_by_id = {}
        return self._filter_query_relevant_conflicts(
            clean_query,
            selected=conflicts,
            limit=bounded_limit,
            objects_by_id=objects_by_id,
        )

    def select_relevant_episodic_objects(
        self,
        *,
        query_text: str | None,
        limit: int = 4,
        fallback_limit: int = 2,
        require_query_match: bool = False,
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        """Select episodic memories relevant to one retrieval query."""

        remote_selected = self._remote_select_objects(
            query_text=query_text,
            limit=limit,
            include_episodes=True,
            fallback_limit=fallback_limit,
            require_query_match=require_query_match,
        )
        if remote_selected is not None:
            return remote_selected
        with self._lock:
            objects = tuple(
                sorted(
                    (
                        item
                        for item in self.load_objects()
                        if item.kind == "episode" and item.status in {"active", "candidate", "uncertain"}
                    ),
                    key=lambda item: (
                        _coerce_aware_utc(item.updated_at),
                        _coerce_aware_utc(item.created_at),
                        item.memory_id,
                    ),
                    reverse=True,
                )
            )
            if not objects:
                return ()
            bounded_limit = max(1, limit)
            clean_query = _normalize_text(query_text)
            if not clean_query:
                if require_query_match:
                    return ()
                return objects[:bounded_limit]
            selector = self._object_selector(objects)
            selected_ids = selector.search(
                clean_query,
                limit=bounded_limit,
                category="object",
                allow_fallback=not require_query_match and fallback_limit > 0,
            )
            by_id = {item.memory_id: item for item in objects}
            selected = [by_id[memory_id] for memory_id in selected_ids if memory_id in by_id]
            filtered = list(self._filter_query_relevant_objects(clean_query, selected=selected, limit=bounded_limit))
            if not filtered and not require_query_match and fallback_limit > 0:
                return objects[: min(bounded_limit, fallback_limit)]
            return tuple(filtered[:bounded_limit])
