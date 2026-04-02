"""Remote-aware selectors for episodic objects, durable objects, and conflicts."""

# mypy: disable-error-code=attr-defined

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

from twinr.memory.longterm.core.models import LongTermMemoryConflictV1, LongTermMemoryObjectV1
from twinr.memory.longterm.reasoning.conversation_recall import (
    conversation_recap_query_variants,
    is_conversation_episode_object,
    query_has_conversation_recap_semantics,
)

from .shared import _LOG, _coerce_aware_utc, _normalize_text

_LIVE_OBJECT_STATUSES = frozenset({"active", "candidate", "uncertain"})
_REMOTE_RECAP_EPISODIC_WINDOW = 48


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
        payloads = remote_catalog.load_selection_item_payloads(snapshot_kind=snapshot_kind, item_ids=item_ids)
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

    def _load_remote_objects_from_item_ids(
        self,
        *,
        item_ids: Iterable[str],
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        """Load query-time object payloads by item id without item-document fallback."""

        remote_catalog = self._remote_catalog
        if remote_catalog is None:
            return ()
        payloads = remote_catalog.load_selection_item_payloads(
            snapshot_kind="objects",
            item_ids=item_ids,
        )
        return self._load_remote_objects_from_payloads(payloads=payloads)

    def _load_remote_conflicts_from_payloads(
        self,
        *,
        payloads: Iterable[Mapping[str, object]],
    ) -> tuple[LongTermMemoryConflictV1, ...]:
        """Parse already-materialized remote conflict payloads."""

        loaded: list[LongTermMemoryConflictV1] = []
        for payload in payloads:
            try:
                loaded.append(LongTermMemoryConflictV1.from_payload(dict(payload)))
            except Exception:
                _LOG.warning("Skipping invalid remote long-term conflict payload during selective load.", exc_info=True)
        return tuple(loaded)

    def _load_remote_conflicts_from_item_ids(
        self,
        *,
        item_ids: Iterable[str],
    ) -> tuple[LongTermMemoryConflictV1, ...]:
        """Load current conflict payloads by item id from the fine-grained catalog."""

        remote_catalog = self._remote_catalog
        if remote_catalog is None:
            return ()
        payloads = remote_catalog.load_selection_item_payloads(
            snapshot_kind="conflicts",
            item_ids=item_ids,
        )
        return self._load_remote_conflicts_from_payloads(payloads=payloads)

    def _catalog_entry_projection(
        self,
        entry: object,
    ) -> Mapping[str, object]:
        metadata = getattr(entry, "metadata", None)
        if not isinstance(metadata, Mapping):
            return {}
        projection = metadata.get("selection_projection")
        if isinstance(projection, Mapping):
            return projection
        return metadata

    def _catalog_entry_text(self, entry: object, field: str) -> str:
        projection = self._catalog_entry_projection(entry)
        value = projection.get(field)
        return _normalize_text(value) if isinstance(value, str) else ""

    def _catalog_entry_list(self, entry: object, field: str) -> tuple[str, ...]:
        projection = self._catalog_entry_projection(entry)
        values = projection.get(field)
        if not isinstance(values, list):
            return ()
        return tuple(
            normalized
            for normalized in (_normalize_text(value) for value in values if isinstance(value, str))
            if normalized
        )

    def _review_entry_eligible(
        self,
        entry: object,
        *,
        status: str | None,
        kind: str | None,
        include_episodes: bool,
    ) -> bool:
        entry_kind = self._catalog_entry_text(entry, "kind")
        entry_status = self._catalog_entry_text(entry, "status")
        if not include_episodes and entry_kind == "episode":
            return False
        if status is not None and entry_status != _normalize_text(status):
            return False
        if kind is not None and entry_kind != _normalize_text(kind):
            return False
        return bool(entry_kind)

    def select_review_objects_query_first(
        self,
        *,
        query_text: str | None,
        status: str | None,
        kind: str | None,
        include_episodes: bool,
        limit: int,
    ) -> tuple[tuple[LongTermMemoryObjectV1, ...], int] | None:
        """Select review objects from catalog metadata before exact hydration."""

        remote_catalog = self._remote_catalog
        if not self._remote_catalog_enabled() or remote_catalog is None:
            return None
        bounded_limit = max(1, int(limit))
        clean_query = _normalize_text(query_text)

        def eligible(entry: object) -> bool:
            return self._review_entry_eligible(
                entry,
                status=status,
                kind=kind,
                include_episodes=include_episodes,
            )

        entries = tuple(
            entry
            for entry in remote_catalog.load_catalog_entries(snapshot_kind="objects")
            if eligible(entry)
        )
        if not entries:
            return (), 0
        if not clean_query:
            selected_entries = remote_catalog.top_catalog_entries(
                snapshot_kind="objects",
                limit=bounded_limit,
                eligible=eligible,
            )
            return self._load_remote_objects_from_entries(entries=selected_entries, snapshot_kind="objects"), len(entries)
        matched_entries = remote_catalog.search_catalog_entries(
            snapshot_kind="objects",
            query_text=clean_query,
            limit=max(bounded_limit, len(entries)),
            eligible=eligible,
        )
        return (
            self._load_remote_objects_from_entries(entries=matched_entries[:bounded_limit], snapshot_kind="objects"),
            len(matched_entries),
        )

    def load_conflicts_by_slot_keys(
        self,
        slot_keys: Iterable[str],
    ) -> tuple[LongTermMemoryConflictV1, ...]:
        """Load only conflicts for the requested slot keys."""

        normalized_keys = tuple(
            normalized
            for normalized in (_normalize_text(value) for value in slot_keys)
            if normalized
        )
        if not normalized_keys:
            return ()
        remote_catalog = self._remote_catalog
        if self._remote_catalog_enabled() and remote_catalog is not None:
            try:
                entries = tuple(
                    entry
                    for entry in remote_catalog.load_catalog_entries(snapshot_kind="conflicts")
                    if self._catalog_entry_text(entry, "slot_key") in normalized_keys
                )
                item_ids = tuple(
                    entry.item_id
                    for entry in entries
                    if isinstance(getattr(entry, "item_id", None), str)
                )
                if item_ids:
                    return self._load_remote_conflicts_from_item_ids(item_ids=item_ids)
                if self._remote_is_required():
                    return ()
            except Exception:
                if self._remote_is_required():
                    raise
                _LOG.warning("Failed loading fine-grained remote long-term conflicts by slot key.", exc_info=True)
        conflicts = self.load_conflicts_fine_grained()
        return tuple(conflict for conflict in conflicts if conflict.slot_key in normalized_keys)

    def load_objects_by_slot_keys(
        self,
        slot_keys: Iterable[str],
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        """Load only current objects for the requested slot keys."""

        normalized_keys = tuple(
            normalized
            for normalized in (_normalize_text(value) for value in slot_keys)
            if normalized
        )
        if not normalized_keys:
            return ()
        remote_catalog = self._remote_catalog
        if self._remote_catalog_enabled() and remote_catalog is not None:
            try:
                selected_ids = tuple(
                    dict.fromkeys(
                        entry.item_id
                        for entry in remote_catalog.load_catalog_entries(snapshot_kind="objects")
                        if isinstance(getattr(entry, "item_id", None), str)
                        and self._catalog_entry_text(entry, "slot_key") in normalized_keys
                    )
                )
                if selected_ids:
                    return self.load_objects_by_ids(selected_ids)
                if self._remote_is_required():
                    return ()
            except Exception:
                if self._remote_is_required():
                    raise
                _LOG.warning("Failed loading fine-grained remote long-term objects by slot key.", exc_info=True)
        objects = self.load_objects_fine_grained()
        return tuple(item for item in objects if item.slot_key in normalized_keys)

    def load_objects_by_event_ids(
        self,
        event_ids: Iterable[str],
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        """Load only current objects whose source provenance mentions one of the event ids."""

        normalized_ids = tuple(
            normalized
            for normalized in (_normalize_text(value) for value in event_ids)
            if normalized
        )
        if not normalized_ids:
            return ()
        target_ids = set(normalized_ids)
        remote_catalog = self._remote_catalog
        if self._remote_catalog_enabled() and remote_catalog is not None:
            try:
                selected_ids = tuple(
                    dict.fromkeys(
                        entry.item_id
                        for entry in remote_catalog.load_catalog_entries(snapshot_kind="objects")
                        if isinstance(getattr(entry, "item_id", None), str)
                        and bool(target_ids.intersection(self._catalog_entry_list(entry, "source_event_ids")))
                    )
                )
                if selected_ids:
                    return self.load_objects_by_ids(selected_ids)
                if self._remote_is_required():
                    return ()
            except Exception:
                if self._remote_is_required():
                    raise
                _LOG.warning("Failed loading fine-grained remote long-term objects by source event id.", exc_info=True)
        objects = self.load_objects_fine_grained()
        return tuple(
            item
            for item in objects
            if bool(target_ids.intersection(item.source.event_ids))
        )

    def load_objects_by_projection_filter(
        self,
        *,
        predicate: Callable[[Mapping[str, object]], bool],
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        """Load only current objects whose catalog projection matches ``predicate``."""

        remote_catalog = self._remote_catalog
        if self._remote_catalog_enabled() and remote_catalog is not None:
            try:
                selected_ids = tuple(
                    dict.fromkeys(
                        entry.item_id
                        for entry in remote_catalog.load_catalog_entries(snapshot_kind="objects")
                        if isinstance(getattr(entry, "item_id", None), str)
                        and predicate(self._catalog_entry_projection(entry))
                    )
                )
                if selected_ids:
                    return self.load_objects_by_ids(selected_ids)
                if self._remote_is_required():
                    return ()
            except Exception:
                if self._remote_is_required():
                    raise
                _LOG.warning("Failed loading fine-grained remote long-term objects by projection filter.", exc_info=True)
        objects = self.load_objects_fine_grained()
        return tuple(
            item
            for item in objects
            if predicate(self._remote_object_selection_projection(snapshot_kind="objects", payload=item.to_payload()))
        )

    def load_conflicts_for_memory_ids(
        self,
        memory_ids: Iterable[str],
    ) -> tuple[LongTermMemoryConflictV1, ...]:
        """Load only conflicts that reference one of the requested memory ids."""

        normalized_ids = tuple(
            normalized
            for normalized in (_normalize_text(value) for value in memory_ids)
            if normalized
        )
        if not normalized_ids:
            return ()
        target_ids = set(normalized_ids)
        remote_catalog = self._remote_catalog
        if self._remote_catalog_enabled() and remote_catalog is not None:
            try:
                entries = tuple(
                    entry
                    for entry in remote_catalog.load_catalog_entries(snapshot_kind="conflicts")
                    if self._catalog_entry_text(entry, "candidate_memory_id") in target_ids
                    or bool(target_ids.intersection(self._catalog_entry_list(entry, "existing_memory_ids")))
                )
                item_ids = tuple(
                    entry.item_id
                    for entry in entries
                    if isinstance(getattr(entry, "item_id", None), str)
                )
                if item_ids:
                    return self._load_remote_conflicts_from_item_ids(item_ids=item_ids)
                if self._remote_is_required():
                    return ()
            except Exception:
                if self._remote_is_required():
                    raise
                _LOG.warning("Failed loading fine-grained remote long-term conflicts by memory id.", exc_info=True)
        conflicts = self.load_conflicts_fine_grained()
        return tuple(
            conflict
            for conflict in conflicts
            if conflict.candidate_memory_id in target_ids
            or bool(target_ids.intersection(conflict.existing_memory_ids))
        )

    def load_objects_for_conflict(
        self,
        conflict: LongTermMemoryConflictV1,
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        """Load the exact object set needed to resolve one conflict."""

        explicit_ids = tuple(
            normalized
            for normalized in (
                _normalize_text(value)
                for value in (*conflict.existing_memory_ids, conflict.candidate_memory_id)
            )
            if normalized
        )
        slot_key = _normalize_text(conflict.slot_key)
        remote_catalog = self._remote_catalog
        if self._remote_catalog_enabled() and remote_catalog is not None:
            try:
                selected_ids = tuple(
                    dict.fromkeys(
                        entry.item_id
                        for entry in remote_catalog.load_catalog_entries(snapshot_kind="objects")
                        if isinstance(getattr(entry, "item_id", None), str)
                        and (
                            entry.item_id in explicit_ids
                            or (
                                slot_key
                                and self._catalog_entry_text(entry, "slot_key") == slot_key
                                and self._catalog_entry_text(entry, "status") in _LIVE_OBJECT_STATUSES
                            )
                        )
                    )
                )
                if selected_ids:
                    return self.load_objects_by_ids(selected_ids)
                if self._remote_is_required():
                    return ()
            except Exception:
                if self._remote_is_required():
                    raise
                _LOG.warning("Failed loading fine-grained remote long-term conflict objects.", exc_info=True)
        objects = self.load_objects_fine_grained()
        return tuple(
            item
            for item in objects
            if item.memory_id in explicit_ids
            or (slot_key and item.slot_key == slot_key and item.status in _LIVE_OBJECT_STATUSES)
        )

    def load_objects_referencing_memory_ids(
        self,
        memory_ids: Iterable[str],
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        """Load only objects whose reference fields mention one of the requested ids."""

        normalized_ids = tuple(
            normalized
            for normalized in (_normalize_text(value) for value in memory_ids)
            if normalized
        )
        if not normalized_ids:
            return ()
        target_ids = set(normalized_ids)
        remote_catalog = self._remote_catalog
        if self._remote_catalog_enabled() and remote_catalog is not None:
            try:
                selected_ids = tuple(
                    dict.fromkeys(
                        entry.item_id
                        for entry in remote_catalog.load_catalog_entries(snapshot_kind="objects")
                        if isinstance(getattr(entry, "item_id", None), str)
                        and (
                            bool(target_ids.intersection(self._catalog_entry_list(entry, "conflicts_with")))
                            or bool(target_ids.intersection(self._catalog_entry_list(entry, "supersedes")))
                        )
                    )
                )
                if selected_ids:
                    return self.load_objects_by_ids(selected_ids)
                if self._remote_is_required():
                    return ()
            except Exception:
                if self._remote_is_required():
                    raise
                _LOG.warning("Failed loading fine-grained remote long-term related objects.", exc_info=True)
        objects = self.load_objects_fine_grained()
        return tuple(
            item
            for item in objects
            if target_ids.intersection(item.conflicts_with) or target_ids.intersection(item.supersedes)
        )

    def select_query_time_objects_by_ids(
        self,
        *,
        query_text: str | None,
        memory_ids: Iterable[str],
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        """Load supporting objects for one retrieval section on the bounded query-time path.

        Retrieval callers such as conflict-queue assembly already know the
        target object ids from an upstream query-first selector. On required
        remote memory paths this helper must stay on that same bounded
        contract instead of escalating into exact item `documents/full` reads.
        Cold current-head and segment resolution is now acceptable here
        because it stays on the same current-head + retrieve-batch contract as
        the rest of query-time structured memory retrieval.
        """

        normalized_ids = tuple(
            normalized
            for normalized in (_normalize_text(value) for value in memory_ids)
            if normalized
        )
        if not normalized_ids:
            return ()
        bridge_objects = self._same_process_snapshot_bridge_objects()
        if bridge_objects is not None:
            by_id = {item.memory_id: item for item in bridge_objects}
            return tuple(by_id[memory_id] for memory_id in normalized_ids if memory_id in by_id)
        remote_catalog = self._remote_catalog
        clean_query = _normalize_text(query_text)
        if self._remote_catalog_enabled() and remote_catalog is not None:
            scoped_by_id: dict[str, LongTermMemoryObjectV1] = {}
            missing_ids = normalized_ids
            if clean_query:
                try:
                    payloads = remote_catalog.search_current_item_payloads(
                        snapshot_kind="objects",
                        query_text=clean_query,
                        limit=max(1, len(normalized_ids)),
                        eligible=lambda entry: entry.item_id in normalized_ids,
                        allow_catalog_fallback=False,
                    )
                except Exception:
                    if self._remote_is_required():
                        raise
                    payloads = None
                if payloads is not None:
                    scoped_by_id = {
                        item.memory_id: item
                        for item in self._load_remote_objects_from_payloads(payloads=payloads)
                    }
                    missing_ids = tuple(memory_id for memory_id in normalized_ids if memory_id not in scoped_by_id)
                    if not missing_ids:
                        return tuple(scoped_by_id[memory_id] for memory_id in normalized_ids if memory_id in scoped_by_id)
            by_id = dict(scoped_by_id)
            by_id.update(
                {
                    item.memory_id: item
                    for item in self._load_remote_objects_from_item_ids(item_ids=missing_ids)
                }
            )
            if by_id:
                return tuple(by_id[memory_id] for memory_id in normalized_ids if memory_id in by_id)
            if self._remote_is_required():
                return ()
        with self._lock:
            objects_by_id = {item.memory_id: item for item in self.load_objects_fine_grained()}
            return tuple(objects_by_id[memory_id] for memory_id in normalized_ids if memory_id in objects_by_id)

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
        recap_query = include_episodes and query_has_conversation_recap_semantics(clean_query)
        query_variants = (
            conversation_recap_query_variants(clean_query)
            if recap_query
            else ((clean_query,) if clean_query else ())
        )
        effective_query_text = " ".join(query_variants) if query_variants else clean_query

        def eligible(entry: object) -> bool:
            metadata = getattr(entry, "metadata", None)
            if not isinstance(metadata, Mapping):
                return False
            kind = _normalize_text(metadata.get("kind") if isinstance(metadata.get("kind"), str) else None)
            status = _normalize_text(metadata.get("status") if isinstance(metadata.get("status"), str) else None)
            if status not in {"active", "candidate", "uncertain"}:
                return False
            if include_episodes:
                if kind != "episode":
                    return False
                if recap_query:
                    projection = self._catalog_entry_projection(entry)
                    projection_attributes = (
                        projection.get("attributes")
                        if isinstance(projection.get("attributes"), Mapping)
                        else None
                    )
                    if projection and projection_attributes is not None:
                        return is_conversation_episode_object(
                            kind=projection.get("kind"),
                            attributes=projection_attributes,
                        )
                return True
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
            direct_payloads: tuple[dict[str, object], ...] | None = ()
            for candidate_query in query_variants:
                candidate_payloads = remote_catalog.search_current_item_payloads(
                    snapshot_kind="objects",
                    query_text=candidate_query,
                    limit=bounded_limit,
                    eligible=eligible,
                    allow_catalog_fallback=False,
                )
                if candidate_payloads is None:
                    direct_payloads = None
                    break
                if candidate_payloads:
                    direct_payloads = candidate_payloads
                    break
        except Exception:
            if self._remote_is_required():
                raise
            direct_payloads = None
        if direct_payloads is not None:
            if direct_payloads:
                selected = list(self._load_remote_objects_from_payloads(payloads=direct_payloads))
                filtered = list(
                    self._filter_query_relevant_objects(
                        effective_query_text,
                        selected=selected,
                        limit=bounded_limit,
                    )
                )
                if filtered:
                    return self.rank_selected_objects(
                        query_texts=query_variants or (clean_query,),
                        objects=filtered,
                        limit=bounded_limit,
                    )
                if recap_query:
                    rescued = self._remote_select_recent_recap_episodes(
                        remote_catalog=remote_catalog,
                        eligible=eligible,
                        query_text=effective_query_text,
                        query_texts=query_variants or (clean_query,),
                        limit=bounded_limit,
                    )
                    if rescued:
                        return rescued
                return ()
            if include_episodes:
                if recap_query:
                    rescued = self._remote_select_recent_recap_episodes(
                        remote_catalog=remote_catalog,
                        eligible=eligible,
                        query_text=effective_query_text,
                        query_texts=query_variants or (clean_query,),
                        limit=bounded_limit,
                    )
                    if rescued:
                        return rescued
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

    def _remote_select_recent_recap_episodes(
        self,
        *,
        remote_catalog,
        eligible: Callable[[object], bool],
        query_text: str,
        query_texts: tuple[str, ...],
        limit: int,
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        """Load a bounded recent episodic window for generic conversation recaps."""

        recent_entries = remote_catalog.top_catalog_entries(
            snapshot_kind="objects",
            limit=max(_REMOTE_RECAP_EPISODIC_WINDOW, max(1, int(limit))),
            eligible=eligible,
        )
        if not recent_entries:
            return ()
        selected = list(self._load_remote_objects_from_entries(entries=recent_entries, snapshot_kind="objects"))
        filtered = list(self._filter_query_relevant_objects(query_text, selected=selected, limit=max(1, int(limit))))
        if not filtered:
            return ()
        return self.rank_selected_objects(
            query_texts=query_texts or (query_text,),
            objects=filtered,
            limit=max(1, int(limit)),
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
            payloads = remote_catalog.load_selection_item_payloads(
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
                payloads = remote_catalog.load_selection_item_payloads(
                    snapshot_kind="conflicts",
                    item_ids=(entry.item_id for entry in entries),
                )
        conflicts = list(self._load_remote_conflicts_from_payloads(payloads=payloads))
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
                objects_by_id = {
                    item.memory_id: item
                    for item in self._load_remote_objects_from_item_ids(item_ids=related_ids)
                }
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
                        for item in self.load_objects_fine_grained()
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
