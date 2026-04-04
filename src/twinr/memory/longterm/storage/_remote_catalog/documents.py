"""Document hydration and payload-extraction helpers for remote catalog storage."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
import json
import time
from typing import Any

from twinr.agent.workflows.forensics import workflow_decision, workflow_event
from twinr.memory.chonkydb.models import ChonkyDBRetrieveRequest, ChonkyDBTopKRecordsRequest
from twinr.memory.longterm.core.models import (
    LongTermMemoryConflictV1,
    LongTermMemoryObjectV1,
    LongTermMidtermPacketV1,
    LongTermSourceRefV1,
)
from twinr.memory.longterm.storage.remote_read_diagnostics import (
    LongTermRemoteReadContext,
    record_remote_read_diagnostic,
)
from twinr.memory.longterm.storage.remote_read_observability import record_remote_read_observation
from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError

from ._typing import RemoteCatalogMixinBase
from .shared import (
    LongTermRemoteCatalogEntry,
    _ALLOWED_DOC_IDS_RETRIEVE_QUERY,
    _ITEM_VERSION,
    _RemoteCollectionDefinition,
    _iter_known_item_envelopes,
    _run_timed_workflow_step,
    _trace_search_details,
)


class RemoteCatalogDocumentMixin(RemoteCatalogMixinBase):
    def load_item_payload(
        self,
        *,
        snapshot_kind: str,
        item_id: str,
        document_id: str | None = None,
        uri: str | None = None,
    ) -> dict[str, object] | None:
        """Load one remote item document exactly by document id or URI."""

        remote_state = self._require_remote_state()
        read_client = self._require_client(getattr(remote_state, "read_client", None), operation="read")
        definition = self._require_definition(snapshot_kind)
        normalized_item_id = self._normalize_item_id(item_id)
        if not normalized_item_id:
            return None
        cached_payload = self._cached_item_payload(snapshot_kind=snapshot_kind, item_id=normalized_item_id)
        if cached_payload is not None and not self._is_projection_compat_payload(cached_payload):
            return cached_payload
        resolved_uri = uri or self.item_uri(snapshot_kind=snapshot_kind, item_id=normalized_item_id)
        context = LongTermRemoteReadContext(
            snapshot_kind=snapshot_kind,
            operation="fetch_item_document",
            request_method="GET",
            request_payload_kind="full_document_lookup",
            item_id=normalized_item_id,
            document_id_hint=document_id,
            uri_hint=resolved_uri,
            request_path="/v1/external/documents/full",
        )
        started_monotonic = time.monotonic()
        try:
            envelope = read_client.fetch_full_document(
                document_id=document_id,
                origin_uri=None if document_id else resolved_uri,
                include_content=True,
                max_content_chars=self._max_content_chars(),
            )
        except Exception as exc:
            if document_id:
                try:
                    envelope = read_client.fetch_full_document(
                        origin_uri=resolved_uri,
                        include_content=True,
                        max_content_chars=self._max_content_chars(),
                    )
                except Exception as fallback_exc:
                    record_remote_read_diagnostic(
                        remote_state=remote_state,
                        context=context,
                        exc=fallback_exc,
                        started_monotonic=started_monotonic,
                        outcome="failed",
                    )
                    raise LongTermRemoteUnavailableError(
                        f"Failed to read remote long-term {snapshot_kind!r} item {normalized_item_id!r}."
                    ) from fallback_exc
            else:
                record_remote_read_diagnostic(
                    remote_state=remote_state,
                    context=context,
                    exc=exc,
                    started_monotonic=started_monotonic,
                    outcome="failed",
                )
                raise LongTermRemoteUnavailableError(
                    f"Failed to read remote long-term {snapshot_kind!r} item {normalized_item_id!r}."
                ) from exc
        item_payload = self._extract_item_payload(
            definition=definition,
            item_id=normalized_item_id,
            payload=envelope,
        )
        if item_payload is None:
            return None
        record_remote_read_observation(
            remote_state=remote_state,
            context=context,
            latency_ms=max(0.0, (time.monotonic() - started_monotonic) * 1000.0),
            outcome="ok",
            classification="ok",
        )
        self._store_item_payload(
            snapshot_kind=snapshot_kind,
            item_id=normalized_item_id,
            payload=item_payload,
        )
        return dict(item_payload)

    def load_item_payloads(
        self,
        *,
        snapshot_kind: str,
        item_ids: Iterable[str],
    ) -> tuple[dict[str, object], ...]:
        """Load a bounded set of current item payloads by Twinr item id."""

        entry_by_id = {
            entry.item_id: entry
            for entry in self.load_catalog_entries(snapshot_kind=snapshot_kind)
        }
        ordered_item_ids: list[str] = []
        uncached_entries: list[LongTermRemoteCatalogEntry] = []
        loaded_by_item_id: dict[str, dict[str, object]] = {}
        for raw_item_id in item_ids:
            item_id = self._normalize_item_id(raw_item_id)
            if not item_id:
                continue
            entry = entry_by_id.get(item_id)
            if entry is None:
                continue
            ordered_item_ids.append(item_id)
            cached_payload = self._cached_item_payload(snapshot_kind=snapshot_kind, item_id=item_id)
            if cached_payload is not None and not self._is_projection_compat_payload(cached_payload):
                loaded_by_item_id[item_id] = cached_payload
                continue
            uncached_entries.append(entry)
        loaded_uncached = self._load_item_payloads_from_entries(
            snapshot_kind=snapshot_kind,
            entries=uncached_entries,
        )
        for entry, payload in zip(uncached_entries, loaded_uncached, strict=False):
            if payload is not None:
                loaded_by_item_id[entry.item_id] = payload
        return tuple(loaded_by_item_id[item_id] for item_id in ordered_item_ids if item_id in loaded_by_item_id)

    def load_selection_item_payloads(
        self,
        *,
        snapshot_kind: str,
        item_ids: Iterable[str],
    ) -> tuple[dict[str, object], ...]:
        """Load query-time payloads without falling through to full item documents.

        Context selection and catalog rescue only need the bounded item
        projection stored on catalog entries plus retrieve-batch recovery for
        older live shapes. They must not silently reintroduce expensive
        per-item `documents/full` reads once the query has already narrowed the
        candidate set.
        """

        entry_by_id = {
            entry.item_id: entry
            for entry in self.load_catalog_entries(snapshot_kind=snapshot_kind)
        }
        ordered_entries: list[LongTermRemoteCatalogEntry] = []
        for raw_item_id in item_ids:
            item_id = self._normalize_item_id(raw_item_id)
            if not item_id:
                continue
            entry = entry_by_id.get(item_id)
            if entry is not None:
                ordered_entries.append(entry)
        loaded = self._load_selection_item_payloads_from_entries(
            snapshot_kind=snapshot_kind,
            entries=tuple(ordered_entries),
        )
        resolved: list[dict[str, object]] = []
        for payload in loaded:
            if isinstance(payload, Mapping):
                resolved.append(dict(payload))
        return tuple(resolved)

    def prewarm_item_payload_cache(self, *, snapshot_kind: str) -> None:
        """Warm the full current snapshot payload set into the read cache."""

        if self._persistent_read_cache_ttl_s() <= 0.0:
            return
        entries = self.load_catalog_entries(snapshot_kind=snapshot_kind)
        if not entries:
            return
        self.load_item_payloads(
            snapshot_kind=snapshot_kind,
            item_ids=(entry.item_id for entry in entries),
        )

    def _load_selection_item_payloads_from_entries(
        self,
        *,
        snapshot_kind: str,
        entries: Iterable[LongTermRemoteCatalogEntry],
    ) -> tuple[dict[str, object] | None, ...]:
        """Load query-time payloads from projections or retrieve batches only."""

        ordered_entries = tuple(entries)
        if not ordered_entries:
            return ()
        loaded_by_item_id: dict[str, dict[str, object]] = {}
        unresolved_entries: list[LongTermRemoteCatalogEntry] = []
        for entry in ordered_entries:
            cached_payload = self._cached_item_payload(snapshot_kind=snapshot_kind, item_id=entry.item_id)
            if cached_payload is not None:
                loaded_by_item_id[entry.item_id] = cached_payload
                continue
            projection_payload = self._catalog_entry_item_payload(
                snapshot_kind=snapshot_kind,
                entry=entry,
            )
            if projection_payload is not None:
                loaded_by_item_id[entry.item_id] = dict(projection_payload)
                self._store_item_payload(
                    snapshot_kind=snapshot_kind,
                    item_id=entry.item_id,
                    payload=projection_payload,
                )
                continue
            unresolved_entries.append(entry)
        if unresolved_entries:
            loaded_by_item_id.update(
                self._load_item_payloads_via_retrieve(
                    snapshot_kind=snapshot_kind,
                    entries=tuple(unresolved_entries),
                )
            )
        return tuple(
            dict(loaded_by_item_id[entry.item_id]) if entry.item_id in loaded_by_item_id else None
            for entry in ordered_entries
        )

    def _load_item_payloads_from_entries(
        self,
        *,
        snapshot_kind: str,
        entries: Iterable[LongTermRemoteCatalogEntry],
    ) -> tuple[dict[str, object] | None, ...]:
        """Load multiple current item payloads with batch retrieve fallback."""

        ordered_entries = tuple(entries)
        if not ordered_entries:
            return ()
        loaded_by_item_id: dict[str, dict[str, object]] = {}
        unresolved_entries: list[LongTermRemoteCatalogEntry] = []
        for entry in ordered_entries:
            cached_payload = self._cached_item_payload(snapshot_kind=snapshot_kind, item_id=entry.item_id)
            if cached_payload is not None and not self._is_projection_compat_payload(cached_payload):
                loaded_by_item_id[entry.item_id] = cached_payload
                continue
            projection_payload = self._catalog_entry_item_payload(
                snapshot_kind=snapshot_kind,
                entry=entry,
            )
            if projection_payload is not None and not self._is_projection_compat_payload(projection_payload):
                loaded_by_item_id[entry.item_id] = dict(projection_payload)
                self._store_item_payload(
                    snapshot_kind=snapshot_kind,
                    item_id=entry.item_id,
                    payload=projection_payload,
                )
                continue
            unresolved_entries.append(entry)
        loaded_by_item_id.update(self._load_item_payloads_via_retrieve(
            snapshot_kind=snapshot_kind,
            entries=tuple(unresolved_entries),
        ))
        resolved: list[dict[str, object] | None] = []
        for entry in ordered_entries:
            payload = loaded_by_item_id.get(entry.item_id)
            if payload is None:
                payload = self.load_item_payload(
                    snapshot_kind=snapshot_kind,
                    item_id=entry.item_id,
                    document_id=entry.document_id,
                    uri=entry.uri,
                )
            elif isinstance(payload, Mapping):
                self._store_item_payload(
                    snapshot_kind=snapshot_kind,
                    item_id=entry.item_id,
                    payload=payload,
                )
            resolved.append(dict(payload) if isinstance(payload, Mapping) else None)
        return tuple(resolved)

    def _load_compat_item_payloads_from_entries(
        self,
        *,
        snapshot_kind: str,
        entries: Iterable[LongTermRemoteCatalogEntry],
    ) -> tuple[tuple[dict[str, object] | None, ...], bool]:
        """Load startup-compatible payloads, preferring direct catalog reconstruction."""

        ordered_entries = tuple(entries)
        if not ordered_entries:
            return (), snapshot_kind in {"objects", "archive", "conflicts"}
        loaded_by_item_id: dict[str, dict[str, object]] = {}
        unresolved_entries: list[LongTermRemoteCatalogEntry] = []
        direct_catalog_complete = snapshot_kind in {"objects", "archive", "conflicts"}
        if direct_catalog_complete:
            for entry in ordered_entries:
                payload = self._catalog_entry_item_payload(snapshot_kind=snapshot_kind, entry=entry)
                if payload is None:
                    direct_catalog_complete = False
                    unresolved_entries.append(entry)
                    continue
                loaded_by_item_id[entry.item_id] = payload
        else:
            unresolved_entries.extend(ordered_entries)
        if unresolved_entries:
            loaded_by_item_id.update(
                self._load_item_payloads_via_retrieve(
                    snapshot_kind=snapshot_kind,
                    entries=tuple(unresolved_entries),
                )
            )
        resolved: list[dict[str, object] | None] = []
        for entry in ordered_entries:
            payload = loaded_by_item_id.get(entry.item_id)
            if payload is None:
                payload = self.load_item_payload(
                    snapshot_kind=snapshot_kind,
                    item_id=entry.item_id,
                    document_id=entry.document_id,
                    uri=entry.uri,
                )
            resolved.append(dict(payload) if isinstance(payload, Mapping) else None)
        return tuple(resolved), direct_catalog_complete

    def _load_item_payloads_via_retrieve(
        self,
        *,
        snapshot_kind: str,
        entries: tuple[LongTermRemoteCatalogEntry, ...],
    ) -> dict[str, dict[str, object]]:
        """Load many current item payloads through ChonkyDB retrieve batches."""

        remote_state = self._require_remote_state()
        read_client = self._require_client(getattr(remote_state, "read_client", None), operation="read")
        definition = self._require_definition(snapshot_kind)
        entries_by_document_id: dict[str, list[LongTermRemoteCatalogEntry]] = {}
        for entry in entries:
            document_id = self._normalize_text(entry.document_id)
            if document_id:
                entries_by_document_id.setdefault(document_id, []).append(entry)
        if not entries_by_document_id:
            return {}
        loaded: dict[str, dict[str, object]] = {}
        for batch, candidates in self._load_retrieve_batch_candidates(
            snapshot_kind=snapshot_kind,
            read_client=read_client,
            batches=self._split_retrieve_doc_id_batches(tuple(entries_by_document_id)),
        ):
            for candidate in candidates:
                document_id = self._normalize_text(candidate.get("payload_id")) or self._normalize_text(candidate.get("chonky_id"))
                if not document_id:
                    continue
                batch_entries = entries_by_document_id.get(document_id)
                if not batch_entries:
                    continue
                for entry in batch_entries:
                    payload = self._extract_item_payload(
                        definition=definition,
                        item_id=entry.item_id,
                        payload=candidate,
                    )
                    if payload is not None:
                        loaded[entry.item_id] = dict(payload)
                        self._store_item_payload(
                            snapshot_kind=snapshot_kind,
                            item_id=entry.item_id,
                            payload=payload,
                        )
        return loaded

    def _load_retrieve_batch_candidates(
        self,
        *,
        snapshot_kind: str,
        read_client: Any,
        batches: tuple[tuple[str, ...], ...],
    ) -> tuple[tuple[tuple[str, ...], tuple[Mapping[str, object], ...]], ...]:
        """Load retrieve batches with bounded parallel reads."""

        if not batches:
            return ()
        self._require_remote_state()

        def load_one(batch: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[Mapping[str, object], ...]]:
            try:
                return batch, self._load_remote_batch_candidates(
                    snapshot_kind=snapshot_kind,
                    read_client=read_client,
                    batch=batch,
                )
            except Exception:
                return batch, ()

        max_workers = self._remote_read_max_workers(len(batches))
        if max_workers <= 1:
            return tuple(load_one(batch) for batch in batches)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return tuple(executor.map(load_one, batches))

    def _load_remote_batch_candidates(
        self,
        *,
        snapshot_kind: str,
        read_client: Any,
        batch: tuple[str, ...],
    ) -> tuple[Mapping[str, object], ...]:
        """Load one document-id batch, preferring one-shot structured reads."""

        remote_state = self._require_remote_state()
        topk_records = getattr(read_client, "topk_records", None)
        supports_topk_records = bool(getattr(read_client, "supports_topk_records", callable(topk_records)))
        allowed_indexes = self._selection_hydration_allowed_indexes()
        workflow_decision(
            msg="longterm_remote_catalog_batch_api",
            question="Which remote API contract should hydrate the selected document-id batch?",
            selected={
                "id": "topk_records" if supports_topk_records and callable(topk_records) else "retrieve",
                "summary": (
                    "Use one-shot top-k batch hydration."
                    if supports_topk_records and callable(topk_records)
                    else "Use retrieve batch hydration."
                ),
            },
            options=[
                {
                    "id": "topk_records",
                    "summary": "Use the structured top-k endpoint with allowed_doc_ids.",
                    "score_components": {"supports_topk_records": bool(supports_topk_records and callable(topk_records))},
                    "constraints_violated": [] if supports_topk_records and callable(topk_records) else ["topk_records_unavailable"],
                },
                {
                    "id": "retrieve",
                    "summary": "Use the legacy retrieve endpoint with allowed_doc_ids.",
                    "score_components": {"batch_size": len(batch)},
                    "constraints_violated": [],
                },
            ],
            context=_trace_search_details(
                snapshot_kind=snapshot_kind,
                result_limit=len(batch),
                allowed_doc_count=len(batch),
                batch_size=len(batch),
            ),
            confidence="high",
            guardrails=[
                "Prefer one-shot structured batch hydration when the backend supports it.",
                "Keep retrieve batch as the safe fallback for legacy or degraded paths.",
            ],
            kpi_impact_estimate={"batch_size": len(batch)},
        )
        if supports_topk_records and callable(topk_records):
            started_monotonic = time.monotonic()
            try:
                response = _run_timed_workflow_step(
                    name="longterm_remote_catalog_topk_batch_request",
                    kind="http",
                    details=_trace_search_details(
                        snapshot_kind=snapshot_kind,
                        result_limit=len(batch),
                        allowed_doc_count=len(batch),
                        batch_size=len(batch),
                    ),
                    operation=lambda: topk_records(
                        ChonkyDBTopKRecordsRequest(
                            query_text=_ALLOWED_DOC_IDS_RETRIEVE_QUERY,
                            result_limit=len(batch),
                            include_content=False,
                            include_metadata=True,
                            allowed_indexes=allowed_indexes,
                            allowed_doc_ids=batch,
                        )
                    ),
                )
                query_plan = getattr(response, "query_plan", None)
                if isinstance(query_plan, Mapping):
                    workflow_event(
                        kind="retrieval",
                        msg="longterm_remote_catalog_query_plan",
                        details={
                            **_trace_search_details(
                                snapshot_kind=snapshot_kind,
                                result_limit=len(batch),
                                allowed_doc_count=len(batch),
                                batch_size=len(batch),
                            ),
                            "operation": "topk_batch",
                            "query_plan": dict(query_plan),
                        },
                    )
                record_remote_read_observation(
                    remote_state=remote_state,
                    context=LongTermRemoteReadContext(
                        snapshot_kind=snapshot_kind,
                        operation="topk_batch",
                        request_method="POST",
                        request_payload_kind="topk_allowed_doc_batch",
                        allowed_doc_count=len(batch),
                        result_limit=len(batch),
                        batch_size=len(batch),
                        request_path="/v1/external/retrieve/topk_records",
                    ),
                    latency_ms=max(0.0, (time.monotonic() - started_monotonic) * 1000.0),
                    outcome="ok",
                    classification="ok",
                )
                return tuple(self._iter_retrieve_result_candidates(response))
            except Exception as exc:
                record_remote_read_diagnostic(
                    remote_state=remote_state,
                    context=LongTermRemoteReadContext(
                        snapshot_kind=snapshot_kind,
                        operation="topk_batch",
                        request_method="POST",
                        request_payload_kind="topk_allowed_doc_batch",
                        allowed_doc_count=len(batch),
                        result_limit=len(batch),
                        batch_size=len(batch),
                        request_path="/v1/external/retrieve/topk_records",
                    ),
                    exc=exc,
                    started_monotonic=started_monotonic,
                    outcome="fallback",
                )

        workflow_event(
            kind="branch",
            msg="longterm_remote_catalog_fallback_to_retrieve_batch",
            details=_trace_search_details(
                snapshot_kind=snapshot_kind,
                result_limit=len(batch),
                allowed_doc_count=len(batch),
                batch_size=len(batch),
            ),
            reason={
                "selected": {"id": "retrieve_batch", "summary": "Top-k batch hydration failed or is unavailable; fall back to retrieve batch."},
                "options": [
                    {"id": "retrieve_batch", "summary": "Use retrieve batch hydration.", "constraints_violated": []},
                    {"id": "abort_batch_hydration", "summary": "Abort hydration immediately.", "constraints_violated": ["fallback_is_available"]},
                ],
            },
        )
        started_monotonic = time.monotonic()
        try:
            response = _run_timed_workflow_step(
                name="longterm_remote_catalog_retrieve_batch_request",
                kind="http",
                details=_trace_search_details(
                    snapshot_kind=snapshot_kind,
                    result_limit=len(batch),
                    allowed_doc_count=len(batch),
                    batch_size=len(batch),
                ),
                operation=lambda: read_client.retrieve(
                    ChonkyDBRetrieveRequest(
                        query_text=_ALLOWED_DOC_IDS_RETRIEVE_QUERY,
                        result_limit=len(batch),
                        include_content=False,
                        include_metadata=True,
                        allowed_indexes=allowed_indexes,
                        allowed_doc_ids=batch,
                    )
                ),
            )
        except Exception as exc:
            record_remote_read_diagnostic(
                remote_state=remote_state,
                context=LongTermRemoteReadContext(
                    snapshot_kind=snapshot_kind,
                    operation="retrieve_batch",
                    request_method="POST",
                    request_payload_kind="retrieve_allowed_doc_batch",
                    allowed_doc_count=len(batch),
                    result_limit=len(batch),
                    batch_size=len(batch),
                    request_path="/v1/external/retrieve",
                ),
                exc=exc,
                started_monotonic=started_monotonic,
                outcome="degraded",
            )
            raise
        record_remote_read_observation(
            remote_state=remote_state,
            context=LongTermRemoteReadContext(
                snapshot_kind=snapshot_kind,
                operation="retrieve_batch",
                request_method="POST",
                request_payload_kind="retrieve_allowed_doc_batch",
                allowed_doc_count=len(batch),
                result_limit=len(batch),
                batch_size=len(batch),
                request_path="/v1/external/retrieve",
            ),
            latency_ms=max(0.0, (time.monotonic() - started_monotonic) * 1000.0),
            outcome="ok",
            classification="ok",
        )
        return tuple(self._iter_retrieve_result_candidates(response))

    def _iter_retrieve_result_candidates(self, payload: object) -> Iterable[Mapping[str, object]]:
        """Yield mapping-shaped raw retrieval hits from typed or fake clients."""

        raw = getattr(payload, "raw", None)
        if isinstance(raw, Mapping):
            raw_results = raw.get("results")
            if isinstance(raw_results, list):
                for item in raw_results:
                    if isinstance(item, Mapping):
                        yield item
                return
        results = getattr(payload, "results", None)
        if not isinstance(results, Iterable) or isinstance(results, (str, bytes, bytearray, Mapping)):
            return
        for hit in results:
            if isinstance(hit, Mapping):
                yield hit
                continue
            hit_raw = getattr(hit, "raw", None)
            if isinstance(hit_raw, Mapping):
                yield hit_raw
                continue
            candidate: dict[str, object] = {}
            payload_id = self._normalize_text(getattr(hit, "payload_id", None))
            if payload_id:
                candidate["payload_id"] = payload_id
            metadata = getattr(hit, "metadata", None)
            if isinstance(metadata, Mapping):
                candidate["metadata"] = dict(metadata)
            if candidate:
                yield candidate

    def _split_retrieve_doc_id_batches(
        self,
        document_ids: tuple[str, ...],
    ) -> tuple[tuple[str, ...], ...]:
        """Split retrieve doc ids into bounded batches."""

        if not document_ids:
            return ()
        batch_size = self._retrieve_batch_size()
        return tuple(
            tuple(document_ids[index : index + batch_size])
            for index in range(0, len(document_ids), batch_size)
        )

    def _candidate_catalog_entry(
        self,
        *,
        definition: _RemoteCollectionDefinition,
        payload: Mapping[str, object],
    ) -> LongTermRemoteCatalogEntry | None:
        """Build a minimal catalog entry from one live retrieval candidate."""

        item_id: str | None = None
        metadata: dict[str, object] = {}
        for candidate in self._iter_record_candidates(payload):
            if item_id is None:
                item_id = self._normalize_item_id(candidate.get("twinr_memory_item_id"))
            candidate_metadata = self._catalog_entry_metadata_from_mapping(candidate)
            if candidate_metadata:
                metadata.update(candidate_metadata)
        if not item_id:
            return None
        document_id = (
            self._normalize_text(payload.get("document_id"))
            or self._normalize_text(payload.get("payload_id"))
            or self._normalize_text(payload.get("chonky_id"))
        )
        return LongTermRemoteCatalogEntry(
            snapshot_kind=definition.snapshot_kind,
            item_id=item_id,
            document_id=document_id,
            uri=self.item_uri(snapshot_kind=definition.snapshot_kind, item_id=item_id),
            metadata=metadata,
        )

    def _extract_item_payload(
        self,
        *,
        definition: _RemoteCollectionDefinition,
        item_id: str,
        payload: Mapping[str, object],
    ) -> Mapping[str, object] | None:
        for candidate in self._iter_record_candidates(payload):
            normalized = self._extract_public_item_payload(
                definition=definition,
                item_id=item_id,
                candidate=candidate,
            )
            if normalized is not None:
                return normalized
            metadata_payload = self._extract_current_record_metadata_payload(
                definition=definition,
                item_id=item_id,
                candidate=candidate,
            )
            if metadata_payload is not None:
                return metadata_payload
            metadata_only = self._extract_legacy_metadata_only_item_payload(
                definition=definition,
                item_id=item_id,
                candidate=candidate,
            )
            if metadata_only is not None:
                return metadata_only
            if candidate.get("schema") != definition.item_schema:
                continue
            if candidate.get("version") != _ITEM_VERSION:
                continue
            if candidate.get("snapshot_kind") != definition.snapshot_kind:
                continue
            if self._normalize_item_id(candidate.get("item_id")) != item_id:
                continue
            raw_item = candidate.get(definition.envelope_key)
            if isinstance(raw_item, Mapping):
                return raw_item
        return None

    def _extract_public_item_payload(
        self,
        *,
        definition: _RemoteCollectionDefinition,
        item_id: str,
        candidate: Mapping[str, object],
    ) -> dict[str, object] | None:
        normalized_item_id = self._normalize_item_id(item_id)
        if not normalized_item_id:
            return None
        if definition.snapshot_kind in {"objects", "archive"}:
            try:
                obj = LongTermMemoryObjectV1.from_payload(candidate)
            except Exception:
                return None
            if self._normalize_text(obj.memory_id) != normalized_item_id:
                return None
            return obj.to_payload()
        if definition.snapshot_kind == "conflicts":
            try:
                conflict = LongTermMemoryConflictV1.from_payload(candidate)
            except Exception:
                return None
            if self._conflict_doc_id(conflict) != normalized_item_id:
                return None
            return conflict.to_payload()
        if definition.snapshot_kind == "midterm":
            try:
                packet = LongTermMidtermPacketV1.from_payload(candidate)
            except Exception:
                return None
            if self._normalize_text(packet.packet_id) != normalized_item_id:
                return None
            return packet.to_payload()
        return None

    def _extract_current_record_metadata_payload(
        self,
        *,
        definition: _RemoteCollectionDefinition,
        item_id: str,
        candidate: Mapping[str, object],
    ) -> dict[str, object] | None:
        """Accept live current-record metadata payloads for non-core memory types.

        The remote current-record write contract stores the authoritative public
        Twinr payload under `twinr_payload` inside document metadata. Live
        ChonkyDB reads may surface only that metadata, including in chunked
        `documents/full` responses where no top-level record wrapper survives.
        Older extraction only trusted envelope-shaped item records or legacy
        metadata-only objects, so personality/world-state collections hydrated
        zero payloads even though the live item metadata still carried the full
        public payload.
        """

        if definition.snapshot_kind in {"objects", "archive", "conflicts", "midterm"}:
            return None
        normalized_item_id = self._normalize_item_id(item_id)
        if not normalized_item_id:
            return None
        candidate_item_id = self._normalize_item_id(candidate.get("twinr_memory_item_id"))
        if candidate_item_id != normalized_item_id:
            return None
        if self._normalize_text(candidate.get("twinr_snapshot_kind")) != definition.snapshot_kind:
            return None
        nested_payload = candidate.get("twinr_payload")
        if not isinstance(nested_payload, Mapping):
            return None
        return dict(nested_payload)

    def _conflict_doc_id(self, conflict: LongTermMemoryConflictV1) -> str:
        """Return the canonical remote item id for one conflict payload."""

        return conflict.catalog_item_id()

    def _catalog_entry_item_payload(
        self,
        *,
        snapshot_kind: str,
        entry: LongTermRemoteCatalogEntry,
    ) -> dict[str, object] | None:
        definition = self._require_definition(snapshot_kind)
        candidate = dict(entry.metadata)
        candidate["twinr_snapshot_kind"] = definition.snapshot_kind
        candidate["twinr_memory_item_id"] = entry.item_id
        candidate["document_id"] = entry.document_id
        return self._projection_or_compat_item_payload(
            definition=definition,
            item_id=entry.item_id,
            candidate=candidate,
            projection_source_type="remote_catalog_entry",
            projection_attribute="remote_catalog_entry_compatibility",
            compat_source_type="remote_catalog_entry",
            compatibility_attribute="remote_catalog_entry_compatibility",
        )

    def _enrich_catalog_entry(
        self,
        *,
        snapshot_kind: str,
        entry: LongTermRemoteCatalogEntry,
        payload: Mapping[str, object],
    ) -> LongTermRemoteCatalogEntry:
        metadata = dict(entry.metadata)
        for field_name in self._catalog_entry_text_fields():
            value = self._normalize_text(payload.get(field_name))
            if value:
                metadata[field_name] = value
        for field_name in self._catalog_entry_list_fields():
            values = self._normalize_text_list(payload.get(field_name))
            if values:
                metadata[field_name] = list(values)
        payload_sha256 = self._normalize_text(entry.metadata.get("payload_sha256"))
        if payload_sha256:
            metadata["payload_sha256"] = payload_sha256
        return LongTermRemoteCatalogEntry(
            snapshot_kind=snapshot_kind,
            item_id=entry.item_id,
            document_id=entry.document_id,
            uri=entry.uri,
            metadata=metadata,
        )

    def _extract_legacy_metadata_only_item_payload(
        self,
        *,
        definition: _RemoteCollectionDefinition,
        item_id: str,
        candidate: Mapping[str, object],
    ) -> dict[str, object] | None:
        """Rebuild minimal object payloads from legacy metadata-only live reads."""

        if isinstance(candidate.get("twinr_payload"), Mapping):
            return None
        if self._normalize_text(candidate.get("twinr_snapshot_kind")) != definition.snapshot_kind:
            return None
        return self._projection_or_compat_item_payload(
            definition=definition,
            item_id=item_id,
            candidate=candidate,
            projection_source_type="legacy_remote_catalog_metadata",
            projection_attribute="legacy_remote_catalog_metadata_only",
            compat_source_type="legacy_remote_catalog_metadata",
            compatibility_attribute="legacy_remote_catalog_metadata_only",
        )

    def _projection_or_compat_item_payload(
        self,
        *,
        definition: _RemoteCollectionDefinition,
        item_id: str,
        candidate: Mapping[str, object],
        projection_source_type: str,
        projection_attribute: str,
        compat_source_type: str,
        compatibility_attribute: str,
    ) -> dict[str, object] | None:
        projection = self._selection_projection_from_candidate(candidate)
        if projection is not None:
            if definition.snapshot_kind in {"objects", "archive"}:
                payload = self._build_projection_object_payload(
                    definition=definition,
                    item_id=item_id,
                    projection=projection,
                    source_type=projection_source_type,
                    compatibility_attribute=projection_attribute,
                )
                if payload is not None:
                    return payload
            if definition.snapshot_kind == "conflicts":
                payload = self._build_projection_conflict_payload(
                    item_id=item_id,
                    projection=projection,
                )
                if payload is not None:
                    return payload
            if definition.snapshot_kind == "midterm":
                payload = self._build_projection_midterm_payload(
                    item_id=item_id,
                    projection=projection,
                )
                if payload is not None:
                    return payload
        if definition.snapshot_kind not in {"objects", "archive"}:
            return None
        return self._build_compat_object_payload_from_metadata(
            definition=definition,
            item_id=item_id,
            candidate=candidate,
            source_type=compat_source_type,
            compatibility_attribute=compatibility_attribute,
        )

    def _selection_projection_from_candidate(
        self,
        candidate: Mapping[str, object],
    ) -> dict[str, object] | None:
        projection = candidate.get("selection_projection")
        if isinstance(projection, Mapping):
            return dict(projection)
        if isinstance(projection, str):
            parsed = self._parse_json_mapping(projection)
            if isinstance(parsed, Mapping):
                return dict(parsed)
        return None

    def _build_projection_object_payload(
        self,
        *,
        definition: _RemoteCollectionDefinition,
        item_id: str,
        projection: Mapping[str, object],
        source_type: str,
        compatibility_attribute: str | None,
    ) -> dict[str, object] | None:
        if definition.snapshot_kind not in {"objects", "archive"}:
            return None
        normalized_item_id = self._normalize_item_id(projection.get("memory_id"))
        if normalized_item_id != item_id:
            return None
        kind = self._normalize_text(projection.get("kind"))
        summary = self._normalize_text(projection.get("summary"))
        created_at = self._normalize_text(projection.get("created_at"))
        updated_at = self._normalize_text(projection.get("updated_at")) or created_at
        if not kind or not summary or not created_at or not updated_at:
            return None
        raw_attributes = projection.get("attributes")
        attributes = dict(raw_attributes) if isinstance(raw_attributes, Mapping) else {}
        if compatibility_attribute:
            attributes[compatibility_attribute] = True
        raw_conflicts_with = projection.get("conflicts_with")
        conflicts_with = list(raw_conflicts_with) if isinstance(raw_conflicts_with, list) else None
        raw_supersedes = projection.get("supersedes")
        supersedes = list(raw_supersedes) if isinstance(raw_supersedes, list) else None
        raw_confidence = projection.get("confidence", 0.5)
        try:
            confidence = float(raw_confidence if isinstance(raw_confidence, (int, float, str)) else 0.5)
        except (TypeError, ValueError):
            confidence = 0.5
        try:
            return LongTermMemoryObjectV1(
                memory_id=item_id,
                kind=kind,
                summary=summary,
                source=LongTermSourceRefV1(
                    source_type=source_type,
                    event_ids=(item_id,),
                ),
                details=self._normalize_text(projection.get("details")),
                status=self._normalize_text(projection.get("status")) or "candidate",
                confidence=confidence,
                canonical_language=self._normalize_text(projection.get("canonical_language")) or "en",
                confirmed_by_user=bool(projection.get("confirmed_by_user")),
                sensitivity=self._normalize_text(projection.get("sensitivity")) or "normal",
                slot_key=self._normalize_text(projection.get("slot_key")),
                value_key=self._normalize_text(projection.get("value_key")),
                valid_from=self._normalize_text(projection.get("valid_from")),
                valid_to=self._normalize_text(projection.get("valid_to")),
                archived_at=self._normalize_text(projection.get("archived_at")),
                created_at=created_at,
                updated_at=updated_at,
                attributes=attributes or None,
                conflicts_with=tuple(value for value in conflicts_with or () if isinstance(value, str)),
                supersedes=tuple(value for value in supersedes or () if isinstance(value, str)),
            ).to_payload()
        except Exception:
            return None

    def _build_projection_conflict_payload(
        self,
        *,
        item_id: str,
        projection: Mapping[str, object],
    ) -> dict[str, object] | None:
        try:
            payload = LongTermMemoryConflictV1.from_payload(projection).to_payload()
            conflict = LongTermMemoryConflictV1.from_payload(payload)
        except Exception:
            return None
        return payload if self._conflict_doc_id(conflict) == item_id else None

    def _build_projection_midterm_payload(
        self,
        *,
        item_id: str,
        projection: Mapping[str, object],
    ) -> dict[str, object] | None:
        try:
            payload = LongTermMidtermPacketV1.from_payload(projection).to_payload()
            packet = LongTermMidtermPacketV1.from_payload(payload)
        except Exception:
            return None
        return payload if self._normalize_text(packet.packet_id) == item_id else None

    def _is_projection_compat_payload(
        self,
        payload: Mapping[str, object],
    ) -> bool:
        """Return whether a cached object payload came from a lossy compat projection."""

        raw_attributes = payload.get("attributes")
        if not isinstance(raw_attributes, Mapping):
            return False
        return bool(
            raw_attributes.get("remote_catalog_entry_compatibility")
            or raw_attributes.get("legacy_remote_catalog_metadata_only")
        )

    def _build_compat_object_payload_from_metadata(
        self,
        *,
        definition: _RemoteCollectionDefinition,
        item_id: str,
        candidate: Mapping[str, object],
        source_type: str,
        compatibility_attribute: str,
    ) -> dict[str, object] | None:
        if definition.snapshot_kind not in {"objects", "archive"}:
            return None
        normalized_item_id = self._normalize_item_id(candidate.get("twinr_memory_item_id"))
        if normalized_item_id != item_id:
            return None
        kind = self._normalize_text(candidate.get("kind"))
        summary = self._normalize_text(candidate.get("summary"))
        created_at = self._normalize_text(candidate.get("created_at"))
        updated_at = self._normalize_text(candidate.get("updated_at")) or created_at
        if not kind or not summary or not created_at or not updated_at:
            return None
        attributes: dict[str, object] = {compatibility_attribute: True}
        origin_uri = self._normalize_text(candidate.get("origin_uri"))
        if origin_uri:
            attributes["legacy_origin_uri"] = origin_uri
        raw_timestamp = candidate.get("timestamp")
        if isinstance(raw_timestamp, int):
            attributes["legacy_timestamp"] = raw_timestamp
        document_id = self._normalize_text(candidate.get("document_id"))
        if document_id:
            attributes["remote_document_id"] = document_id
        try:
            return LongTermMemoryObjectV1(
                memory_id=item_id,
                kind=kind,
                summary=summary,
                source=LongTermSourceRefV1(
                    source_type=source_type,
                    event_ids=(item_id,),
                ),
                status=self._normalize_text(candidate.get("status")) or "candidate",
                slot_key=self._normalize_text(candidate.get("slot_key")),
                value_key=self._normalize_text(candidate.get("value_key")),
                archived_at=self._normalize_text(candidate.get("archived_at")),
                created_at=created_at,
                updated_at=updated_at,
                attributes=attributes,
            ).to_payload()
        except Exception:
            return None

    def _iter_record_candidates(self, payload: Mapping[str, object]) -> Iterable[Mapping[str, object]]:
        yield payload
        yield from _iter_known_item_envelopes(payload)
        direct = payload.get("payload")
        if isinstance(direct, Mapping):
            yield direct
            yield from _iter_known_item_envelopes(direct)
        metadata = payload.get("metadata")
        if isinstance(metadata, Mapping):
            yield metadata
            yield from _iter_known_item_envelopes(metadata)
            nested_payload = metadata.get("twinr_payload")
            if isinstance(nested_payload, Mapping):
                yield nested_payload
        nested = payload.get("record")
        if isinstance(nested, Mapping):
            yield nested
            yield from _iter_known_item_envelopes(nested)
            nested_payload = nested.get("payload")
            if isinstance(nested_payload, Mapping):
                yield nested_payload
                yield from _iter_known_item_envelopes(nested_payload)
            nested_metadata = nested.get("metadata")
            if isinstance(nested_metadata, Mapping):
                yield nested_metadata
                yield from _iter_known_item_envelopes(nested_metadata)
                nested_inner_payload = nested_metadata.get("twinr_payload")
                if isinstance(nested_inner_payload, Mapping):
                    yield nested_inner_payload
            nested_content = nested.get("content")
            if isinstance(nested_content, str):
                parsed = self._parse_json_mapping(nested_content)
                if parsed is not None:
                    yield parsed
            nested_document = nested.get("document")
            if isinstance(nested_document, Mapping):
                yield from self._iter_record_candidates(nested_document)
        document = payload.get("document")
        if isinstance(document, Mapping):
            yield from self._iter_record_candidates(document)
        content = payload.get("content")
        if isinstance(content, str):
            parsed = self._parse_json_mapping(content)
            if parsed is not None:
                yield parsed
        chunks = payload.get("chunks")
        if isinstance(chunks, list):
            for chunk in chunks:
                if not isinstance(chunk, Mapping):
                    continue
                yield from _iter_known_item_envelopes(chunk)
                chunk_metadata = chunk.get("metadata")
                if isinstance(chunk_metadata, Mapping):
                    yield chunk_metadata
                    yield from _iter_known_item_envelopes(chunk_metadata)
                    chunk_inner_payload = chunk_metadata.get("twinr_payload")
                    if isinstance(chunk_inner_payload, Mapping):
                        yield chunk_inner_payload
                chunk_payload = chunk.get("payload")
                if isinstance(chunk_payload, Mapping):
                    yield chunk_payload
                    yield from _iter_known_item_envelopes(chunk_payload)
                chunk_content = chunk.get("content")
                if isinstance(chunk_content, str):
                    parsed = self._parse_json_mapping(chunk_content)
                    if parsed is not None:
                        yield parsed

    def _parse_json_mapping(self, value: str) -> Mapping[str, object] | None:
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return None
        return parsed if isinstance(parsed, Mapping) else None


__all__ = [
    "RemoteCatalogDocumentMixin",
]
