"""Retrieval entrypoints and shared context selection for structured memory."""


from __future__ import annotations

from collections.abc import Iterable
from typing import Mapping, cast

from twinr.agent.workflows.forensics import workflow_decision, workflow_event, workflow_span
from twinr.memory.fulltext import FullTextDocument, FullTextSelector
from twinr.memory.longterm.core.models import LongTermMemoryConflictV1, LongTermMemoryObjectV1
from twinr.memory.longterm.storage.remote_state import LongTermRemoteReadFailedError
from twinr.text_utils import retrieval_terms

from .shared import (
    _coerce_aware_utc,
    _normalize_text,
    _retrieval_trace_details,
    _run_timed_workflow_step,
)


class StructuredStoreRetrievalMixin:
    """Own retrieval entrypoints and shared context-object selection."""

    def select_relevant_objects(
        self,
        *,
        query_text: str | None,
        limit: int = 4,
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        """Select durable non-episodic objects relevant to a query."""

        bridge_objects = self._same_process_snapshot_bridge_objects()
        if bridge_objects is not None:
            workflow_event(
                kind="branch",
                msg="longterm_objects_same_process_snapshot_bridge",
                details={
                    "query_present": bool(_normalize_text(query_text)),
                    "limit": max(1, limit),
                },
            )
            return self._select_relevant_objects_from_loaded(
                objects=bridge_objects,
                query_text=query_text,
                include_episodes=False,
                limit=limit,
                require_query_match=False,
            )
        remote_selected = self._remote_select_objects(
            query_text=query_text,
            limit=limit,
            include_episodes=False,
            fallback_limit=0,
            require_query_match=False,
        )
        if remote_selected is not None:
            return remote_selected
        with self._lock:
            return self._select_relevant_objects_from_loaded(
                objects=self.load_objects_fine_grained(),
                query_text=query_text,
                include_episodes=False,
                limit=limit,
                require_query_match=False,
            )

    def select_fast_topic_objects(
        self,
        *,
        query_text: str | None,
        limit: int = 3,
        timeout_s: float | None = None,
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        """Select a tiny live object set for one bounded fast-topic hint block."""

        bounded_limit = max(1, int(limit))
        clean_query = _normalize_text(query_text)
        if not clean_query:
            return ()
        bridge_objects = self._same_process_snapshot_bridge_objects()
        if bridge_objects is not None:
            workflow_event(
                kind="branch",
                msg="longterm_fast_topic_same_process_snapshot_bridge",
                details={
                    "query_present": True,
                    "limit": bounded_limit,
                },
            )
            return self._select_fast_topic_objects_from_objects(
                clean_query=clean_query,
                objects=bridge_objects,
                limit=bounded_limit,
            )
        remote_catalog = self._remote_catalog
        if self._remote_catalog_enabled() and remote_catalog is not None:
            try:
                direct_payloads = _run_timed_workflow_step(
                    name="longterm_fast_topic_scope_payload_search",
                    kind="retrieval",
                    details=_retrieval_trace_details(
                        clean_query,
                        durable_limit=bounded_limit,
                        candidate_limit=bounded_limit,
                    ),
                    operation=lambda: remote_catalog.search_current_item_payloads_fast(
                        snapshot_kind="objects",
                        query_text=clean_query,
                        limit=bounded_limit,
                        timeout_s=timeout_s,
                    ),
                )
                return self._filter_fast_topic_objects_for_query(
                    clean_query=clean_query,
                    objects=self._load_remote_objects_from_payloads(payloads=direct_payloads),
                    limit=bounded_limit,
                )
            except LongTermRemoteReadFailedError:
                if self._remote_is_required():
                    raise
                local_selected = self._select_fast_topic_objects_from_local_snapshot(
                    clean_query=clean_query,
                    limit=bounded_limit,
                )
                return local_selected or ()
            except Exception:
                if self._remote_is_required():
                    raise
                local_selected = self._select_fast_topic_objects_from_local_snapshot(
                    clean_query=clean_query,
                    limit=bounded_limit,
                )
                return local_selected or ()
        local_selected = self._select_fast_topic_objects_from_local_snapshot(
            clean_query=clean_query,
            limit=bounded_limit,
        )
        return local_selected or ()

    def _select_fast_topic_objects_from_local_snapshot(
        self,
        *,
        clean_query: str,
        limit: int,
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        """Search the local object snapshot when remote fast-topic retrieval is not required."""

        with self._lock:
            return self._select_fast_topic_objects_from_objects(
                clean_query=clean_query,
                objects=self.load_objects_fine_grained(),
                limit=limit,
            )

    def select_open_conflicts(
        self,
        *,
        query_text: str | None,
        limit: int = 3,
    ) -> tuple[LongTermMemoryConflictV1, ...]:
        """Select unresolved conflicts relevant to a query."""

        bridge_conflicts = self._same_process_snapshot_bridge_conflicts()
        if bridge_conflicts is not None:
            bounded_limit = max(1, limit)
            clean_query = _normalize_text(query_text)
            workflow_event(
                kind="branch",
                msg="longterm_conflicts_same_process_snapshot_bridge",
                details={
                    "query_present": bool(clean_query),
                    "limit": bounded_limit,
                },
            )
            if not clean_query:
                return bridge_conflicts[:bounded_limit]
            bridge_objects = self._same_process_snapshot_bridge_objects() or ()
            objects_by_id = {item.memory_id: item for item in bridge_objects}
            filtered = self._filter_query_relevant_conflicts(
                clean_query,
                selected=bridge_conflicts,
                limit=bounded_limit,
                objects_by_id=objects_by_id,
            )
            if filtered or not bridge_conflicts:
                return filtered
            return self._filter_query_relevant_conflicts(
                clean_query,
                selected=bridge_conflicts,
                limit=bounded_limit,
            )
        remote_selected = self._remote_select_conflicts(
            query_text=query_text,
            limit=limit,
        )
        if remote_selected is not None:
            return remote_selected
        with self._lock:
            conflicts = self.load_conflicts_fine_grained()
            if not conflicts:
                return ()
            bounded_limit = max(1, limit)
            clean_query = _normalize_text(query_text)
            if not clean_query:
                return conflicts[:bounded_limit]
            related_memory_ids = tuple(
                dict.fromkeys(
                    memory_id
                    for conflict in conflicts
                    for memory_id in (conflict.candidate_memory_id, *conflict.existing_memory_ids)
                    if isinstance(memory_id, str) and memory_id
                )
            )
            objects_by_id = {
                item.memory_id: item
                for item in self.select_query_time_objects_by_ids(
                    query_text=clean_query,
                    memory_ids=related_memory_ids,
                )
            }
            selector = FullTextSelector(
                tuple(
                    FullTextDocument(
                        doc_id=self._conflict_doc_id(conflict),
                        category="conflict",
                        content=self._conflict_search_text(conflict, objects_by_id=objects_by_id),
                    )
                    for conflict in conflicts
                )
            )
            selected_doc_ids = selector.search(clean_query, limit=bounded_limit, category="conflict")
            by_doc_id = {self._conflict_doc_id(item): item for item in conflicts}
            selected = [by_doc_id[doc_id] for doc_id in selected_doc_ids if doc_id in by_doc_id]
            return self._filter_query_relevant_conflicts(
                clean_query,
                selected=selected,
                limit=bounded_limit,
                objects_by_id=objects_by_id,
            )

    def select_relevant_context_objects(
        self,
        *,
        query_text: str | None,
        episodic_limit: int,
        durable_limit: int,
    ) -> tuple[tuple[LongTermMemoryObjectV1, ...], tuple[LongTermMemoryObjectV1, ...]]:
        """Select episodic and durable objects from one shared query pass."""

        resolved_episodic_limit = max(0, int(episodic_limit))
        resolved_durable_limit = max(0, int(durable_limit))
        if resolved_episodic_limit <= 0 and resolved_durable_limit <= 0:
            return (), ()

        clean_query = _normalize_text(query_text)
        if not clean_query:
            return (), ()
        bridge_objects = self._same_process_snapshot_bridge_objects()
        if bridge_objects is not None:
            workflow_event(
                kind="branch",
                msg="longterm_context_objects_same_process_snapshot_bridge",
                details=_retrieval_trace_details(
                    clean_query,
                    episodic_limit=resolved_episodic_limit,
                    durable_limit=resolved_durable_limit,
                ),
            )
            return self._rescue_underfilled_context_sections(
                query_text=clean_query,
                partitioned=self._partition_context_objects(
                    query_text=clean_query,
                    objects=bridge_objects,
                    episodic_limit=resolved_episodic_limit,
                    durable_limit=resolved_durable_limit,
                ),
                durable_limit=resolved_durable_limit,
            )

        remote_catalog = self._remote_catalog
        shared_limit = self._shared_context_object_limit(
            episodic_limit=resolved_episodic_limit,
            durable_limit=resolved_durable_limit,
        )
        retry_limit = max(resolved_episodic_limit, resolved_durable_limit, 1)
        workflow_decision(
            msg="longterm_context_objects_selection_strategy",
            question="Which retrieval route should build the shared episodic and durable object context?",
            selected={
                "id": "remote_scope_then_catalog" if self._remote_catalog_enabled() and remote_catalog is not None else "local_selector",
                "summary": (
                    "Prefer current-scope remote retrieval, then catalog rescue."
                    if self._remote_catalog_enabled() and remote_catalog is not None
                    else "Use the local selector over already loaded object snapshots."
                ),
            },
            options=[
                {
                    "id": "remote_scope_then_catalog",
                    "summary": "Use ChonkyDB current-scope retrieval first and fall back to catalog-backed selection only when needed.",
                    "score_components": {
                        "remote_catalog_enabled": bool(self._remote_catalog_enabled() and remote_catalog is not None),
                        "shared_limit": shared_limit,
                        "retry_limit": retry_limit,
                    },
                    "constraints_violated": [] if self._remote_catalog_enabled() and remote_catalog is not None else ["remote_catalog_disabled"],
                },
                {
                    "id": "local_selector",
                    "summary": "Search the already loaded local object snapshot directly.",
                    "score_components": {
                        "remote_catalog_enabled": bool(self._remote_catalog_enabled() and remote_catalog is not None),
                        "shared_limit": shared_limit,
                    },
                    "constraints_violated": [] if not (self._remote_catalog_enabled() and remote_catalog is not None) else ["not_preferred_when_remote_catalog_available"],
                },
            ],
            context=_retrieval_trace_details(
                clean_query,
                episodic_limit=resolved_episodic_limit,
                durable_limit=resolved_durable_limit,
            ),
            confidence="high",
            guardrails=[
                "Use one shared object search so the Pi does not pay the same remote roundtrip twice.",
                "Keep current-scope retrieval authoritative before falling back to catalog hydration.",
            ],
            kpi_impact_estimate={
                "shared_limit": shared_limit,
                "retry_limit": retry_limit,
            },
        )
        if self._remote_catalog_enabled() and remote_catalog is not None:
            attempted_limits: list[int] = []
            for candidate_limit in (shared_limit, retry_limit):
                if candidate_limit in attempted_limits:
                    continue
                attempted_limits.append(candidate_limit)
                try:
                    def _scope_payload_search(candidate_limit: int = candidate_limit) -> object:
                        return remote_catalog.search_current_item_payloads(
                            snapshot_kind="objects",
                            query_text=clean_query,
                            limit=candidate_limit,
                            allow_catalog_fallback=False,
                        )

                    direct_payloads = cast(
                        tuple[Mapping[str, object], ...] | list[Mapping[str, object]] | None,
                        _run_timed_workflow_step(
                            name="longterm_context_objects_scope_payload_search",
                            kind="retrieval",
                            details=_retrieval_trace_details(
                                clean_query,
                                episodic_limit=resolved_episodic_limit,
                                durable_limit=resolved_durable_limit,
                                candidate_limit=candidate_limit,
                            ),
                            operation=_scope_payload_search,
                        ),
                    )
                except Exception:
                    if self._remote_is_required():
                        raise
                    direct_payloads = None
                if direct_payloads is None:
                    workflow_event(
                        kind="branch",
                        msg="longterm_context_objects_scope_unavailable",
                        details=_retrieval_trace_details(
                            clean_query,
                            episodic_limit=resolved_episodic_limit,
                            durable_limit=resolved_durable_limit,
                            candidate_limit=candidate_limit,
                        ),
                        reason={
                            "selected": {"id": "catalog_rescue", "summary": "Current-scope retrieval was unavailable; continue into catalog-backed selection."},
                            "options": [
                                {"id": "catalog_rescue", "summary": "Continue into catalog-backed selection.", "constraints_violated": []},
                                {"id": "abort_scope_retry", "summary": "Abort shared context lookup entirely.", "constraints_violated": ["would_hide_relevant_memory"]},
                            ],
                        },
                    )
                    continue
                with workflow_span(
                    name="longterm_context_objects_partition_shared_pool",
                    kind="retrieval",
                    details=_retrieval_trace_details(
                        clean_query,
                        episodic_limit=resolved_episodic_limit,
                        durable_limit=resolved_durable_limit,
                        candidate_limit=candidate_limit,
                        payload_count=len(direct_payloads),
                    ),
                ):
                    shared_objects = self._load_remote_objects_from_payloads(payloads=direct_payloads)
                    partitioned = self._partition_context_objects(
                        query_text=clean_query,
                        objects=shared_objects,
                        episodic_limit=resolved_episodic_limit,
                        durable_limit=resolved_durable_limit,
                    )
                if direct_payloads and not partitioned[0] and not partitioned[1]:
                    if not self._collapsed_scope_partition_needs_catalog_rescue(
                        query_text=clean_query,
                        objects=shared_objects,
                    ):
                        workflow_event(
                            kind="branch",
                            msg="longterm_context_objects_scope_authoritative_miss",
                            details=_retrieval_trace_details(
                                clean_query,
                                episodic_limit=resolved_episodic_limit,
                                durable_limit=resolved_durable_limit,
                                candidate_limit=candidate_limit,
                                payload_count=len(direct_payloads),
                            ),
                            reason={
                                "selected": {
                                    "id": "return_empty_partition",
                                    "summary": "Current-scope hits stayed off-topic after active-status filtering, so broader catalog rescue would only repeat the same miss work.",
                                },
                                "options": [
                                    {
                                        "id": "return_empty_partition",
                                        "summary": "Treat the active scope payloads as an authoritative semantic miss.",
                                        "constraints_violated": [],
                                    },
                                    {
                                        "id": "retry_or_catalog_rescue",
                                        "summary": "Continue into the next candidate window or catalog rescue.",
                                        "constraints_violated": ["would_repeat_off_topic_remote_work"],
                                    },
                                ],
                            },
                        )
                        return (), ()
                    workflow_event(
                        kind="branch",
                        msg="longterm_context_objects_scope_partition_empty",
                        details=_retrieval_trace_details(
                            clean_query,
                            episodic_limit=resolved_episodic_limit,
                            durable_limit=resolved_durable_limit,
                            candidate_limit=candidate_limit,
                            payload_count=len(direct_payloads),
                        ),
                        reason={
                            "selected": {"id": "retry_or_catalog_rescue", "summary": "Scope hits collapsed after partitioning; keep searching instead of returning empty context."},
                            "options": [
                                {"id": "retry_or_catalog_rescue", "summary": "Continue into the next candidate window or catalog rescue.", "constraints_violated": []},
                                {"id": "return_empty_partition", "summary": "Return no shared context objects.", "constraints_violated": ["would_hide_fresh_current_facts"]},
                            ],
                        },
                    )
                    continue
                with workflow_span(
                    name="longterm_context_objects_rescue_underfilled_sections",
                    kind="retrieval",
                    details=_retrieval_trace_details(
                        clean_query,
                        episodic_limit=resolved_episodic_limit,
                        durable_limit=resolved_durable_limit,
                        candidate_limit=candidate_limit,
                        payload_count=len(direct_payloads),
                    ),
                ):
                    return self._rescue_underfilled_context_sections(
                        query_text=clean_query,
                        partitioned=partitioned,
                        durable_limit=resolved_durable_limit,
                    )
            try:
                if not remote_catalog.catalog_available(snapshot_kind="objects"):
                    return (), ()
            except Exception:
                if self._remote_is_required():
                    raise
                return (), ()
            entries = cast(
                tuple[object, ...] | list[object],
                _run_timed_workflow_step(
                    name="longterm_context_objects_catalog_search",
                    kind="retrieval",
                    details=_retrieval_trace_details(
                        clean_query,
                        episodic_limit=resolved_episodic_limit,
                        durable_limit=resolved_durable_limit,
                        candidate_limit=shared_limit,
                    ),
                    operation=lambda: remote_catalog.search_catalog_entries(
                        snapshot_kind="objects",
                        query_text=clean_query,
                        limit=shared_limit,
                    ),
                ),
            )
            with workflow_span(
                name="longterm_context_objects_catalog_hydrate",
                kind="retrieval",
                details=_retrieval_trace_details(
                    clean_query,
                    episodic_limit=resolved_episodic_limit,
                    durable_limit=resolved_durable_limit,
                    entry_count=len(entries),
                ),
            ):
                shared_objects = self._load_remote_objects_from_entries(
                    entries=entries,
                    snapshot_kind="objects",
                )
                partitioned = self._partition_context_objects(
                    query_text=clean_query,
                    objects=shared_objects,
                    episodic_limit=resolved_episodic_limit,
                    durable_limit=resolved_durable_limit,
                )
            with workflow_span(
                name="longterm_context_objects_rescue_underfilled_sections",
                kind="retrieval",
                details=_retrieval_trace_details(
                    clean_query,
                    episodic_limit=resolved_episodic_limit,
                    durable_limit=resolved_durable_limit,
                    entry_count=len(entries),
                ),
            ):
                return self._rescue_underfilled_context_sections(
                    query_text=clean_query,
                    partitioned=partitioned,
                    durable_limit=resolved_durable_limit,
                )

        with self._lock:
            objects = tuple(
                sorted(
                    (
                        item
                        for item in self.load_objects_fine_grained()
                        if item.status in {"active", "candidate", "uncertain"}
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
                return (), ()
            selector = self._object_selector(objects)
            selected_ids = selector.search(clean_query, limit=shared_limit)
            by_id = {item.memory_id: item for item in objects}
            selected = tuple(by_id[memory_id] for memory_id in selected_ids if memory_id in by_id)
            return self._rescue_underfilled_context_sections(
                query_text=clean_query,
                partitioned=self._partition_context_objects(
                    query_text=clean_query,
                    objects=selected,
                    episodic_limit=resolved_episodic_limit,
                    durable_limit=resolved_durable_limit,
                ),
                durable_limit=resolved_durable_limit,
            )

    def _select_fast_topic_objects_from_loaded(
        self,
        *,
        objects: Iterable[LongTermMemoryObjectV1],
        limit: int,
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        """Keep fast-topic recall tiny and limited to current live objects."""

        selected: list[LongTermMemoryObjectV1] = []
        seen_memory_ids: set[str] = set()
        bounded_limit = max(1, int(limit))
        for item in objects:
            if item.memory_id in seen_memory_ids:
                continue
            if item.status not in {"active", "candidate", "uncertain"}:
                continue
            seen_memory_ids.add(item.memory_id)
            selected.append(item)
            if len(selected) >= bounded_limit:
                break
        return tuple(selected)

    def _filter_fast_topic_objects_for_query(
        self,
        *,
        clean_query: str,
        objects: Iterable[LongTermMemoryObjectV1],
        limit: int,
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        """Keep quick-memory hints topic-anchored for the current query."""

        eligible_objects = list(self._select_fast_topic_objects_from_loaded(objects=objects, limit=max(1, int(limit))))
        if not eligible_objects:
            return ()
        filtered = self._filter_query_relevant_objects(
            clean_query,
            selected=eligible_objects,
            limit=max(1, int(limit)),
        )
        if not filtered:
            return ()
        return self._select_fast_topic_objects_from_loaded(objects=filtered, limit=max(1, int(limit)))

    def _select_fast_topic_objects_from_objects(
        self,
        *,
        clean_query: str,
        objects: Iterable[LongTermMemoryObjectV1],
        limit: int,
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        """Run the bounded fast-topic selector over one already-loaded object pool."""

        eligible_objects = tuple(
            item
            for item in objects
            if item.status in {"active", "candidate", "uncertain"}
        )
        if not eligible_objects:
            return ()
        selector = self._object_selector(eligible_objects)
        selected_ids = selector.search(clean_query, limit=limit)
        by_id = {item.memory_id: item for item in eligible_objects}
        selected = tuple(by_id[memory_id] for memory_id in selected_ids if memory_id in by_id)
        return self._filter_fast_topic_objects_for_query(
            clean_query=clean_query,
            objects=selected,
            limit=limit,
        )

    def _shared_context_object_limit(
        self,
        *,
        episodic_limit: int,
        durable_limit: int,
    ) -> int:
        """Choose one ranked candidate window for shared context-object search."""

        max_limit = max(episodic_limit, durable_limit, 1)
        return max(12, max_limit * 4)

    def _partition_context_objects(
        self,
        *,
        query_text: str,
        objects: Iterable[LongTermMemoryObjectV1],
        episodic_limit: int,
        durable_limit: int,
    ) -> tuple[tuple[LongTermMemoryObjectV1, ...], tuple[LongTermMemoryObjectV1, ...]]:
        """Split one shared ranked object pool into episodic and durable sections."""

        eligible_objects = tuple(
            item
            for item in objects
            if item.status in {"active", "candidate", "uncertain"}
        )
        episodic_candidates = [item for item in eligible_objects if item.kind == "episode"]
        durable_candidates = [item for item in eligible_objects if item.kind != "episode"]
        episodic_filtered = self._filter_query_relevant_objects(
            query_text,
            selected=episodic_candidates,
            limit=max(episodic_limit, 1),
        )
        durable_filtered = self._filter_query_relevant_objects(
            query_text,
            selected=durable_candidates,
            limit=max(durable_limit, 1),
        )
        episodic_ranked = (
            self.rank_selected_objects(
                query_texts=(query_text,),
                objects=episodic_filtered,
                limit=episodic_limit,
            )
            if episodic_limit > 0
            else ()
        )
        durable_ranked = (
            self.rank_selected_objects(
                query_texts=(query_text,),
                objects=durable_filtered,
                limit=durable_limit,
            )
            if durable_limit > 0
            else ()
        )
        return episodic_ranked, durable_ranked

    def _collapsed_scope_partition_needs_catalog_rescue(
        self,
        *,
        query_text: str,
        objects: Iterable[LongTermMemoryObjectV1],
    ) -> bool:
        """Return whether an empty scope partition may still hide current facts."""

        eligible_objects = tuple(
            item
            for item in objects
            if item.status in {"active", "candidate", "uncertain"}
        )
        if not eligible_objects:
            return True
        query_terms = self._query_match_terms(retrieval_terms(query_text))
        if not query_terms:
            return False
        semantic_query_terms = self._semantic_query_terms(query_terms)
        for item in eligible_objects:
            if self._has_query_overlap(
                query_terms=semantic_query_terms or query_terms,
                document_terms=retrieval_terms(self._object_semantic_search_text(item)),
            ):
                return True
        return False

    def _rescue_underfilled_context_sections(
        self,
        *,
        query_text: str,
        partitioned: tuple[tuple[LongTermMemoryObjectV1, ...], tuple[LongTermMemoryObjectV1, ...]],
        durable_limit: int,
    ) -> tuple[tuple[LongTermMemoryObjectV1, ...], tuple[LongTermMemoryObjectV1, ...]]:
        """Top up durable context when a mixed shared pool starves durable hits."""

        episodic_ranked, durable_ranked = partitioned
        if durable_limit <= 0 or len(durable_ranked) >= durable_limit:
            return partitioned
        rescue_candidates = self.select_relevant_objects(
            query_text=query_text,
            limit=durable_limit,
        )
        if not rescue_candidates:
            return partitioned
        merged: list[LongTermMemoryObjectV1] = list(durable_ranked)
        seen_memory_ids = {item.memory_id for item in durable_ranked}
        for item in rescue_candidates:
            if item.memory_id in seen_memory_ids:
                continue
            seen_memory_ids.add(item.memory_id)
            merged.append(item)
        rescued_durable = self.rank_selected_objects(
            query_texts=(query_text,),
            objects=merged,
            limit=durable_limit,
        )
        return episodic_ranked, rescued_durable
