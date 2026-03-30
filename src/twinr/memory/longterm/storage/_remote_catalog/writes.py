"""Catalog persistence, segment IO, and write-attestation helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
import json
import math
import time
from uuid import uuid4

from twinr.memory.chonkydb.client import ChonkyDBError
from twinr.memory.chonkydb.models import (
    ChonkyDBBulkRecordRequest,
    ChonkyDBRecordItem,
    ChonkyDBRetrieveRequest,
    ChonkyDBTopKRecordsRequest,
)
from twinr.memory.longterm.storage._remote_retry import (
    clone_client_with_capped_timeout,
    remote_write_retry_delay_s,
    retryable_remote_write_attempts,
    should_fallback_async_job_resolution_error,
    should_retry_remote_write_error,
)
from twinr.memory.longterm.storage.remote_read_diagnostics import (
    LongTermRemoteReadContext,
    LongTermRemoteWriteContext,
    record_remote_read_diagnostic,
    record_remote_write_diagnostic,
)
from twinr.memory.longterm.storage.remote_read_observability import (
    record_remote_read_observation,
    record_remote_write_observation,
)
from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError

from .shared import (
    _ALLOWED_DOC_IDS_RETRIEVE_QUERY,
    LongTermRemoteCatalogEntry,
    _CATALOG_VERSION,
    _SEGMENT_VERSION,
    _DEFAULT_ASYNC_ATTESTATION_POLL_S,
    _ITEM_VERSION,
    _RemoteCollectionDefinition,
    _iter_known_item_envelopes,
    _run_timed_workflow_step,
)

_SYNC_SMALL_CONTROL_PLANE_WRITE_MAX_BYTES = 16_384
_SYNC_SMALL_CONTROL_PLANE_SNAPSHOT_KINDS = frozenset({"conflicts", "graph_nodes", "graph_edges"})


class RemoteCatalogWriteMixin:
    def build_catalog_payload(
        self,
        *,
        snapshot_kind: str,
        item_payloads: Iterable[Mapping[str, object]],
        item_id_getter,
        metadata_builder,
        content_builder,
    ) -> dict[str, object]:
        """Persist individual remote items and return the small current catalog."""

        self._clear_read_cache(snapshot_kind=snapshot_kind)
        remote_state = self._require_remote_state()
        write_client = self._require_client(getattr(remote_state, "write_client", None), operation="write")
        definition = self._require_definition(snapshot_kind)
        existing_entries = (
            {}
            if definition.snapshot_kind in {"graph_nodes", "graph_edges"}
            else {
                entry.item_id: entry
                for entry in self._load_catalog_entries_for_write(snapshot_kind=snapshot_kind)
            }
        )
        record_items: list[ChonkyDBRecordItem] = []
        staged: list[tuple[str, dict[str, object], str, dict[str, object]]] = []
        catalog_entries: list[dict[str, object]] = []
        for raw_payload in item_payloads:
            payload = dict(raw_payload)
            item_id = self._normalize_item_id(item_id_getter(payload))
            if not item_id:
                continue
            payload_sha256 = self._payload_sha256(payload)
            user_metadata = metadata_builder(payload)
            catalog_metadata = self._catalog_entry_metadata(
                payload_sha256=payload_sha256,
                user_metadata=user_metadata,
            )
            existing_entry = existing_entries.get(item_id)
            if self._can_reuse_existing_entry(
                snapshot_kind=definition.snapshot_kind,
                entry=existing_entry,
                item_id=item_id,
                payload_sha256=payload_sha256,
            ):
                catalog_entries.append(
                    self._build_catalog_entry(
                        item_id=item_id,
                        document_id=existing_entry.document_id if existing_entry is not None else None,
                        metadata=catalog_metadata,
                    )
                )
                continue
            uri = self.item_uri(snapshot_kind=snapshot_kind, item_id=item_id)
            metadata = self._build_item_record_metadata(
                definition=definition,
                item_id=item_id,
                payload=payload,
                payload_sha256=payload_sha256,
            )
            if isinstance(user_metadata, Mapping):
                for key, value in user_metadata.items():
                    normalized_key = str(key)
                    if value is None or normalized_key in self._catalog_entry_object_fields():
                        continue
                    metadata[normalized_key] = value
            content_text = content_builder(payload)
            record_items.append(
                ChonkyDBRecordItem(
                    payload=self._build_item_record_payload(
                        definition=definition,
                        item_id=item_id,
                        payload=payload,
                        content_text=content_text,
                    ),
                    metadata=metadata,
                    content=content_text,
                    uri=uri,
                    enable_chunking=False,
                    include_insights_in_response=False,
                )
            )
            staged.append((item_id, catalog_metadata, payload_sha256, payload))
        document_ids = self._store_record_items(
            write_client,
            snapshot_kind=definition.snapshot_kind,
            record_items=record_items,
        )
        for index, (item_id, metadata, _payload_sha256, payload) in enumerate(staged):
            catalog_entries.append(
                self._build_catalog_entry(
                    item_id=item_id,
                    document_id=document_ids[index] if index < len(document_ids) else None,
                    metadata=metadata,
                )
            )
        segment_refs = self._persist_catalog_segments(
            write_client,
            definition=definition,
            catalog_entries=catalog_entries,
        )
        self._store_catalog_entries(
            snapshot_kind=definition.snapshot_kind,
            entries=tuple(
                LongTermRemoteCatalogEntry(
                    snapshot_kind=definition.snapshot_kind,
                    item_id=str(entry.get("item_id") or ""),
                    document_id=self._normalize_text(entry.get("document_id")),
                    uri=self.item_uri(
                        snapshot_kind=definition.snapshot_kind,
                        item_id=str(entry.get("item_id") or ""),
                    ),
                    metadata=self._catalog_entry_metadata_from_mapping(entry),
                )
                for entry in catalog_entries
                if self._normalize_text(entry.get("item_id"))
            ),
        )
        return {
            "schema": definition.catalog_schema,
            "version": _CATALOG_VERSION,
            "items_count": len(catalog_entries),
            "segments": list(segment_refs),
        }

    def persist_catalog_entries(
        self,
        *,
        snapshot_kind: str,
        entries: Iterable[LongTermRemoteCatalogEntry],
        written_at: str | None = None,
    ) -> dict[str, object]:
        """Persist only the current catalog head/segments for existing item docs."""

        self._clear_read_cache(snapshot_kind=snapshot_kind)
        remote_state = self._require_remote_state()
        write_client = self._require_client(getattr(remote_state, "write_client", None), operation="write")
        definition = self._require_definition(snapshot_kind)
        ordered_entries = tuple(entries)
        catalog_entries = [
            self._build_catalog_entry(
                item_id=entry.item_id,
                document_id=entry.document_id,
                metadata=entry.metadata,
            )
            for entry in ordered_entries
        ]
        segment_refs = self._persist_catalog_segments(
            write_client,
            definition=definition,
            catalog_entries=catalog_entries,
        )
        self._store_catalog_entries(
            snapshot_kind=definition.snapshot_kind,
            entries=tuple(
                LongTermRemoteCatalogEntry(
                    snapshot_kind=definition.snapshot_kind,
                    item_id=entry.item_id,
                    document_id=entry.document_id,
                    uri=entry.uri,
                    metadata=dict(entry.metadata),
                )
                for entry in ordered_entries
            ),
        )
        payload = {
            "schema": definition.catalog_schema,
            "version": _CATALOG_VERSION,
            "items_count": len(catalog_entries),
            "segments": list(segment_refs),
        }
        if isinstance(written_at, str) and written_at.strip():
            payload["written_at"] = written_at
        self.persist_catalog_payload(snapshot_kind=snapshot_kind, payload=payload)
        return payload

    def persist_catalog_payload(
        self,
        *,
        snapshot_kind: str,
        payload: Mapping[str, object],
    ) -> dict[str, object]:
        """Persist the authoritative current catalog head as one fixed-URI record."""

        payload_dict = dict(payload)
        if not self.is_catalog_payload(snapshot_kind=snapshot_kind, payload=payload_dict):
            raise ValueError(f"Remote catalog head for {snapshot_kind!r} must be a supported catalog payload.")
        remote_state = self._require_remote_state()
        write_client = self._require_client(getattr(remote_state, "write_client", None), operation="write")
        definition = self._require_definition(snapshot_kind)
        written_at = self._normalize_text(payload_dict.get("written_at"))
        try:
            items_count = int(payload_dict.get("items_count") or 0)
        except (TypeError, ValueError):
            items_count = 0
        metadata: dict[str, object] = {
            "twinr_snapshot_kind": definition.snapshot_kind,
            "twinr_catalog_current_head": True,
            "twinr_catalog_schema": str(payload_dict.get("schema") or ""),
            "twinr_catalog_items_count": max(0, items_count),
        }
        if written_at:
            metadata["twinr_catalog_written_at"] = written_at
        self._store_record_items(
            write_client,
            snapshot_kind=definition.snapshot_kind,
            record_items=[
                ChonkyDBRecordItem(
                    payload=payload_dict,
                    metadata=metadata,
                    content=json.dumps(payload_dict, ensure_ascii=False),
                    uri=self._catalog_head_uri(snapshot_kind=definition.snapshot_kind),
                    enable_chunking=False,
                    include_insights_in_response=False,
                )
            ],
        )
        self._store_recent_catalog_head_payload(snapshot_kind=definition.snapshot_kind, payload=payload_dict)
        return payload_dict

    def _build_item_record_metadata(
        self,
        *,
        definition: _RemoteCollectionDefinition,
        item_id: str,
        payload: Mapping[str, object],
        payload_sha256: str,
    ) -> dict[str, object]:
        return {
            "twinr_snapshot_kind": definition.snapshot_kind,
            "twinr_memory_item_id": item_id,
            "twinr_payload_sha256": payload_sha256,
            "twinr_payload": dict(payload),
        }

    def _build_item_record_payload(
        self,
        *,
        definition: _RemoteCollectionDefinition,
        item_id: str,
        payload: Mapping[str, object],
        content_text: str,
    ) -> dict[str, object]:
        """Build the public ChonkyDB record payload for one fine-grained item.

        The record wrapper keeps ChonkyDB-facing schema/version fields, while
        the actual Twinr object/conflict payload lives under the collection's
        public envelope key. This lets live reads recover the authoritative
        item payload directly from the record body instead of relying solely on
        a metadata-embedded compatibility blob.
        """

        return {
            "schema": definition.item_schema,
            "version": _ITEM_VERSION,
            "snapshot_kind": definition.snapshot_kind,
            "item_id": item_id,
            definition.envelope_key: dict(payload),
            "metadata": {
                "twinr_snapshot_kind": definition.snapshot_kind,
                "twinr_memory_item_id": item_id,
            },
            "content": content_text,
        }

    def _catalog_entry_metadata(
        self,
        *,
        payload_sha256: str,
        user_metadata: Mapping[str, object] | None,
    ) -> dict[str, object]:
        metadata: dict[str, object] = {"payload_sha256": payload_sha256}
        if isinstance(user_metadata, Mapping):
            for field_name in self._catalog_entry_text_fields():
                value = self._normalize_text(user_metadata.get(field_name))
                if value:
                    metadata[field_name] = value
            for field_name in self._catalog_entry_list_fields():
                values = self._normalize_text_list(user_metadata.get(field_name))
                if values:
                    metadata[field_name] = list(values)
            for field_name in self._catalog_entry_object_fields():
                value = user_metadata.get(field_name)
                if isinstance(value, Mapping):
                    metadata[field_name] = dict(value)
        return metadata

    def _build_catalog_entry(
        self,
        *,
        item_id: str,
        document_id: str | None,
        metadata: Mapping[str, object],
    ) -> dict[str, object]:
        catalog_entry: dict[str, object] = {
            "item_id": item_id,
            "document_id": document_id,
        }
        for field_name in (*self._catalog_entry_text_fields(), "payload_sha256"):
            value = self._normalize_text(metadata.get(field_name))
            if value:
                catalog_entry[field_name] = value
        for field_name in self._catalog_entry_list_fields():
            values = self._normalize_text_list(metadata.get(field_name))
            if values:
                catalog_entry[field_name] = list(values)
        for field_name in self._catalog_entry_object_fields():
            value = metadata.get(field_name)
            if isinstance(value, Mapping):
                catalog_entry[field_name] = dict(value)
        return catalog_entry

    def _can_reuse_existing_entry(
        self,
        *,
        snapshot_kind: str,
        entry: LongTermRemoteCatalogEntry | None,
        item_id: str,
        payload_sha256: str,
    ) -> bool:
        # Graph current-view repairs must overwrite stale item documents even
        # when the logical payload is unchanged, because older live runs could
        # leave behind unreadable same-URI records while still advertising the
        # item through the current catalog head.
        if snapshot_kind in {"graph_nodes", "graph_edges"}:
            return False
        del snapshot_kind
        del item_id
        if entry is None:
            return False
        existing_sha256 = self._normalize_text(entry.metadata.get("payload_sha256"))
        if not existing_sha256 or existing_sha256 != payload_sha256:
            return False
        return bool(self._normalize_text(entry.document_id) or self._normalize_text(entry.uri))

    def _load_legacy_catalog_entries(
        self,
        *,
        definition: _RemoteCollectionDefinition,
        payload: Mapping[str, object],
    ) -> tuple[LongTermRemoteCatalogEntry, ...]:
        items = payload.get("items")
        assert isinstance(items, list)
        entries: list[LongTermRemoteCatalogEntry] = []
        for raw_entry in items:
            if not isinstance(raw_entry, Mapping):
                continue
            item_id = self._normalize_item_id(raw_entry.get("item_id"))
            uri = self._normalize_text(raw_entry.get("uri"))
            if not item_id or not uri:
                continue
            metadata = raw_entry.get("metadata")
            entries.append(
                LongTermRemoteCatalogEntry(
                    snapshot_kind=definition.snapshot_kind,
                    item_id=item_id,
                    document_id=self._normalize_text(raw_entry.get("document_id")),
                    uri=uri,
                    metadata=self._catalog_entry_metadata_from_mapping(metadata if isinstance(metadata, Mapping) else {}),
                )
            )
        return tuple(entries)

    def _load_segmented_catalog_entries(
        self,
        *,
        definition: _RemoteCollectionDefinition,
        payload: Mapping[str, object],
    ) -> tuple[LongTermRemoteCatalogEntry, ...]:
        segments = payload.get("segments")
        assert isinstance(segments, list)
        entries: list[LongTermRemoteCatalogEntry] = []
        segment_requests: list[tuple[int | None, str | None, str | None]] = []
        for raw_segment in segments:
            if not isinstance(raw_segment, Mapping):
                continue
            segment_index = self._normalize_segment_index(raw_segment.get("segment_index"))
            document_id = self._normalize_text(raw_segment.get("document_id"))
            uri = self._normalize_text(raw_segment.get("uri"))
            if document_id is None and uri is None:
                continue
            segment_requests.append((segment_index, document_id, uri))
        for envelope in self._load_catalog_segment_payloads(
            definition=definition,
            segment_requests=tuple(segment_requests),
        ):
            entries.extend(self._extract_segment_entries(definition=definition, payload=envelope))
        return tuple(entries)

    def _load_catalog_segment_payloads(
        self,
        *,
        definition: _RemoteCollectionDefinition,
        segment_requests: tuple[tuple[int | None, str | None, str | None], ...],
    ) -> tuple[Mapping[str, object], ...]:
        """Load segmented catalog documents with bounded parallel reads."""

        if not segment_requests:
            return ()

        remote_state = self._require_remote_state()
        read_client = self._require_client(getattr(remote_state, "read_client", None), operation="read")
        payloads_by_document_id = self._load_catalog_segment_payloads_via_query(
            definition=definition,
            read_client=read_client,
            segment_requests=segment_requests,
        )

        def load_one(request: tuple[int | None, str | None, str | None]) -> Mapping[str, object]:
            segment_index, document_id, uri = request
            normalized_document_id = self._normalize_text(document_id)
            if normalized_document_id:
                cached_payload = payloads_by_document_id.get(normalized_document_id)
                if cached_payload is not None:
                    return cached_payload
            return self._load_catalog_segment_payload(
                definition=definition,
                segment_index=segment_index,
                document_id=document_id,
                uri=uri,
            )

        remaining_requests = tuple(
            request
            for request in segment_requests
            if self._normalize_text(request[1]) not in payloads_by_document_id
        )
        if not remaining_requests:
            return tuple(
                dict(payloads_by_document_id[str(document_id)])
                for _segment_index, document_id, _uri in segment_requests
                if isinstance(document_id, str) and document_id in payloads_by_document_id
            )

        resolved_by_request: dict[tuple[int | None, str | None, str | None], Mapping[str, object]] = {}
        for request in segment_requests:
            normalized_document_id = self._normalize_text(request[1])
            if normalized_document_id and normalized_document_id in payloads_by_document_id:
                resolved_by_request[request] = dict(payloads_by_document_id[normalized_document_id])

        max_workers = self._remote_read_max_workers(len(remaining_requests))
        if max_workers <= 1:
            for request in remaining_requests:
                resolved_by_request[request] = load_one(request)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for request, payload in zip(
                    remaining_requests,
                    executor.map(load_one, remaining_requests),
                    strict=False,
                ):
                    resolved_by_request[request] = payload
        return tuple(dict(resolved_by_request[request]) for request in segment_requests if request in resolved_by_request)

    def _load_catalog_segment_payloads_via_query(
        self,
        *,
        definition: _RemoteCollectionDefinition,
        read_client: object,
        segment_requests: tuple[tuple[int | None, str | None, str | None], ...],
    ) -> dict[str, Mapping[str, object]]:
        """Load docid-backed catalog segments through the retrieve/topk batch path.

        Current catalog heads already carry exact segment document ids. Using
        the same query-first batch contract for those ids keeps query-time
        rescue off `documents/full` and aligned with the one-shot retrieval
        strategy used for item hydration.
        """

        document_ids = tuple(
            normalized_document_id
            for _segment_index, document_id, _uri in segment_requests
            if (normalized_document_id := self._normalize_text(document_id))
        )
        if not document_ids:
            return {}
        retry_attempts = self._remote_retry_attempts()
        retry_backoff_s = self._remote_retry_backoff_s()
        loaded: dict[str, Mapping[str, object]] = {}
        for batch in self._split_retrieve_doc_id_batches(tuple(dict.fromkeys(document_ids))):
            last_error: Exception | None = None
            for attempt_index in range(retry_attempts):
                try:
                    candidates = self._load_catalog_segment_batch_candidates(
                        snapshot_kind=definition.snapshot_kind,
                        read_client=read_client,
                        batch=batch,
                    )
                except Exception as exc:
                    last_error = exc
                    if attempt_index + 1 >= retry_attempts:
                        break
                    if retry_backoff_s > 0.0:
                        time.sleep(retry_backoff_s)
                    continue
                for candidate in candidates:
                    candidate_document_id = (
                        self._normalize_text(candidate.get("document_id"))
                        or self._normalize_text(candidate.get("payload_id"))
                        or self._normalize_text(candidate.get("chonky_id"))
                    )
                    if candidate_document_id and candidate_document_id in batch:
                        candidate_payload = dict(candidate)
                        if not self._extract_segment_entries(
                            definition=definition,
                            payload=candidate_payload,
                        ):
                            continue
                        loaded[candidate_document_id] = candidate_payload
                last_error = None
                break
            if last_error is not None:
                break
        return loaded

    def _load_catalog_segment_batch_candidates(
        self,
        *,
        snapshot_kind: str,
        read_client: object,
        batch: tuple[str, ...],
    ) -> tuple[Mapping[str, object], ...]:
        """Load segment docs through content-bearing query batches, not documents/full."""

        if not batch:
            return ()
        remote_state = self._require_remote_state()
        topk_records = getattr(read_client, "topk_records", None)
        supports_topk_records = bool(getattr(read_client, "supports_topk_records", callable(topk_records)))
        if supports_topk_records and callable(topk_records):
            started_monotonic = time.monotonic()
            try:
                response = _run_timed_workflow_step(
                    name="longterm_remote_catalog_topk_segment_batch_request",
                    kind="http",
                    details={
                        **self._trace_search_details_for_segment_batch(
                            snapshot_kind=snapshot_kind,
                            batch=batch,
                        ),
                        "content_bearing": True,
                    },
                    operation=lambda: topk_records(
                        ChonkyDBTopKRecordsRequest(
                            query_text=_ALLOWED_DOC_IDS_RETRIEVE_QUERY,
                            result_limit=len(batch),
                            include_content=True,
                            include_metadata=True,
                            max_content_chars=self._max_content_chars(),
                            allowed_doc_ids=batch,
                        )
                    ),
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
                return tuple(self._iter_catalog_segment_batch_result_candidates(response))
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

        started_monotonic = time.monotonic()
        try:
            response = _run_timed_workflow_step(
                name="longterm_remote_catalog_retrieve_segment_batch_request",
                kind="http",
                details={
                    **self._trace_search_details_for_segment_batch(
                        snapshot_kind=snapshot_kind,
                        batch=batch,
                    ),
                    "content_bearing": True,
                },
                operation=lambda: read_client.retrieve(
                    ChonkyDBRetrieveRequest(
                        query_text=_ALLOWED_DOC_IDS_RETRIEVE_QUERY,
                        result_limit=len(batch),
                        include_content=True,
                        include_metadata=True,
                        max_content_chars=self._max_content_chars(),
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
        return tuple(self._iter_catalog_segment_batch_result_candidates(response))

    @staticmethod
    def _trace_search_details_for_segment_batch(
        *,
        snapshot_kind: str,
        batch: tuple[str, ...],
    ) -> dict[str, object]:
        return {
            "snapshot_kind": str(snapshot_kind or "").strip(),
            "query_chars": 0,
            "scope_search": False,
            "has_namespace": False,
            "has_scope_ref": False,
            "result_limit": len(batch),
            "allowed_doc_count": len(batch),
            "batch_size": len(batch),
        }

    def _iter_catalog_segment_batch_result_candidates(self, payload: object) -> Iterable[Mapping[str, object]]:
        """Yield mapping-shaped raw retrieve/top-k hits for segment doc batches."""

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
            document_id = self._normalize_text(getattr(hit, "document_id", None))
            if document_id:
                candidate["document_id"] = document_id
            payload_data = getattr(hit, "payload", None)
            if isinstance(payload_data, Mapping):
                candidate["payload"] = dict(payload_data)
            metadata = getattr(hit, "metadata", None)
            if isinstance(metadata, Mapping):
                candidate["metadata"] = dict(metadata)
            content = getattr(hit, "content", None)
            if isinstance(content, str):
                candidate["content"] = content
            if candidate:
                yield candidate

    def _load_catalog_segment_payload(
        self,
        *,
        definition: _RemoteCollectionDefinition,
        segment_index: int | None,
        document_id: str | None,
        uri: str | None,
    ) -> Mapping[str, object]:
        remote_state = self._require_remote_state()
        read_client = self._require_client(getattr(remote_state, "read_client", None), operation="read")
        fallback_uri = uri or self._catalog_segment_uri(
            snapshot_kind=definition.snapshot_kind,
            segment_index=segment_index or 0,
        )
        retry_attempts = self._remote_retry_attempts()
        retry_backoff_s = self._remote_retry_backoff_s()
        context = LongTermRemoteReadContext(
            snapshot_kind=definition.snapshot_kind,
            operation="fetch_catalog_segment",
            request_method="GET",
            request_payload_kind="catalog_segment_document",
            document_id_hint=document_id,
            uri_hint=fallback_uri,
            segment_index=segment_index,
            request_path="/v1/external/documents/full",
        )
        started_monotonic = time.monotonic()
        last_error: Exception | None = None
        for attempt_index in range(retry_attempts):
            try:
                payload = self._fetch_catalog_segment_document(
                    read_client,
                    document_id=document_id,
                    fallback_uri=fallback_uri,
                )
                record_remote_read_observation(
                    remote_state=remote_state,
                    context=context,
                    latency_ms=max(0.0, (time.monotonic() - started_monotonic) * 1000.0),
                    outcome="ok",
                    classification="ok",
                )
                return payload
            except Exception as exc:
                last_error = exc
                if attempt_index + 1 >= retry_attempts:
                    break
                if retry_backoff_s > 0.0:
                    time.sleep(retry_backoff_s)
        assert last_error is not None
        record_remote_read_diagnostic(
            remote_state=remote_state,
            context=LongTermRemoteReadContext(
                snapshot_kind=definition.snapshot_kind,
                operation="fetch_catalog_segment",
                request_method="GET",
                request_payload_kind="catalog_segment_document",
                document_id_hint=document_id,
                uri_hint=fallback_uri,
                segment_index=segment_index,
                request_path="/v1/external/documents/full",
                attempt_index=retry_attempts,
                attempt_count=retry_attempts,
                retry_attempts_configured=retry_attempts,
                retry_backoff_s=retry_backoff_s,
                retry_mode="document_id_then_uri",
            ),
            exc=last_error,
            started_monotonic=started_monotonic,
            outcome="failed",
        )
        raise LongTermRemoteUnavailableError(
            f"Failed to read remote long-term {definition.snapshot_kind!r} catalog segment."
        ) from last_error

    def _fetch_catalog_segment_document(
        self,
        read_client: object,
        *,
        document_id: str | None,
        fallback_uri: str,
    ) -> Mapping[str, object]:
        """Read one segment by exact document id, then by the canonical segment URI."""

        try:
            return read_client.fetch_full_document(
                document_id=document_id,
                origin_uri=None if document_id else fallback_uri,
                include_content=True,
                max_content_chars=self._max_content_chars(),
            )
        except Exception:
            if not document_id:
                raise
        return read_client.fetch_full_document(
            origin_uri=fallback_uri,
            include_content=True,
            max_content_chars=self._max_content_chars(),
        )

    def _extract_segment_entries(
        self,
        *,
        definition: _RemoteCollectionDefinition,
        payload: Mapping[str, object],
    ) -> tuple[LongTermRemoteCatalogEntry, ...]:
        entries: list[LongTermRemoteCatalogEntry] = []
        for candidate in self._iter_record_candidates(payload):
            if candidate.get("schema") != definition.segment_schema:
                continue
            if candidate.get("version") != _SEGMENT_VERSION:
                continue
            if candidate.get("snapshot_kind") != definition.snapshot_kind:
                continue
            raw_items = candidate.get("items")
            if not isinstance(raw_items, list):
                continue
            for raw_entry in raw_items:
                if not isinstance(raw_entry, Mapping):
                    continue
                item_id = self._normalize_item_id(raw_entry.get("item_id"))
                if not item_id:
                    continue
                entries.append(
                    LongTermRemoteCatalogEntry(
                        snapshot_kind=definition.snapshot_kind,
                        item_id=item_id,
                        document_id=self._normalize_text(raw_entry.get("document_id")),
                        uri=self.item_uri(snapshot_kind=definition.snapshot_kind, item_id=item_id),
                        metadata=self._catalog_entry_metadata_from_mapping(raw_entry),
                    )
                )
            break
        return tuple(entries)

    def _persist_catalog_segments(
        self,
        write_client: object,
        *,
        definition: _RemoteCollectionDefinition,
        catalog_entries: list[dict[str, object]],
    ) -> tuple[dict[str, object], ...]:
        segment_batches = self._split_catalog_segment_entries(catalog_entries)
        if not segment_batches:
            return ()
        record_items: list[ChonkyDBRecordItem] = []
        for segment_index, segment_entries in enumerate(segment_batches):
            segment_payload = {
                "schema": definition.segment_schema,
                "version": _SEGMENT_VERSION,
                "snapshot_kind": definition.snapshot_kind,
                "segment_index": segment_index,
                "items": list(segment_entries),
            }
            record_items.append(
                ChonkyDBRecordItem(
                    payload=segment_payload,
                    metadata={
                        "twinr_snapshot_kind": definition.snapshot_kind,
                        "twinr_catalog_segment_index": segment_index,
                        "twinr_catalog_segment_items": len(segment_entries),
                    },
                    content=json.dumps(segment_payload, ensure_ascii=False),
                    uri=self._catalog_segment_uri(snapshot_kind=definition.snapshot_kind, segment_index=segment_index),
                    enable_chunking=False,
                    include_insights_in_response=False,
                )
            )
        document_ids = self._store_record_items(
            write_client,
            snapshot_kind=definition.snapshot_kind,
            record_items=record_items,
        )
        refs: list[dict[str, object]] = []
        for segment_index, segment_entries in enumerate(segment_batches):
            refs.append(
                {
                    "segment_index": segment_index,
                    "document_id": document_ids[segment_index] if segment_index < len(document_ids) else None,
                    "uri": self._catalog_segment_uri(
                        snapshot_kind=definition.snapshot_kind,
                        segment_index=segment_index,
                    ),
                    "entry_count": len(segment_entries),
                }
            )
        return tuple(refs)

    def _split_catalog_segment_entries(
        self,
        catalog_entries: list[dict[str, object]],
    ) -> tuple[tuple[dict[str, object], ...], ...]:
        if not catalog_entries:
            return ()
        max_segment_bytes = self._catalog_segment_max_bytes()
        empty_segment_bytes = len(
            json.dumps(
                {
                    "schema": "segment",
                    "version": _SEGMENT_VERSION,
                    "snapshot_kind": "snapshot",
                    "segment_index": 0,
                    "items": [],
                },
                ensure_ascii=False,
            ).encode("utf-8")
        )
        segments: list[tuple[dict[str, object], ...]] = []
        current_segment: list[dict[str, object]] = []
        current_bytes = empty_segment_bytes
        for entry in catalog_entries:
            entry_bytes = len(json.dumps(entry, ensure_ascii=False).encode("utf-8"))
            separator_bytes = 2 if current_segment else 0
            if current_segment and current_bytes + separator_bytes + entry_bytes > max_segment_bytes:
                segments.append(tuple(current_segment))
                current_segment = []
                current_bytes = empty_segment_bytes
                separator_bytes = 0
            current_segment.append(dict(entry))
            current_bytes += separator_bytes + entry_bytes
        if current_segment:
            segments.append(tuple(current_segment))
        return tuple(segments)

    def _store_record_items(
        self,
        write_client: object,
        *,
        snapshot_kind: str,
        record_items: list[ChonkyDBRecordItem],
    ) -> tuple[str | None, ...]:
        if not record_items:
            return ()
        batches = self._split_bulk_record_batches(record_items)
        document_ids: list[str | None] = []
        total_batches = len(batches)
        for index, batch in enumerate(batches):
            document_ids.extend(
                self._store_record_batch_with_retries(
                    write_client,
                    snapshot_kind=snapshot_kind,
                    batch=tuple(batch),
                    batch_index=index + 1,
                    batch_count=total_batches,
                    finalize_vector_segments=index + 1 >= len(batches),
                )
            )
        return tuple(document_ids)

    def _store_record_batch_with_retries(
        self,
        write_client: object,
        *,
        snapshot_kind: str,
        batch: tuple[ChonkyDBRecordItem, ...],
        batch_index: int,
        batch_count: int,
        finalize_vector_segments: bool,
    ) -> tuple[str | None, ...]:
        retry_attempts = max(1, self._remote_retry_attempts())
        retry_backoff_s = self._remote_retry_backoff_s()
        attempt_index = 0
        while attempt_index < retry_attempts:
            request_correlation_id = self._new_remote_write_correlation_id()
            async_job_timeout_s = self._remote_async_job_timeout_s()
            request_payload_kind = self._record_batch_payload_kind(batch=batch)
            access_classification = self._record_batch_access_classification(batch=batch)
            execution_mode = self._bulk_execution_mode(
                snapshot_kind=snapshot_kind,
                batch=batch,
                request_payload_kind=request_payload_kind,
            )
            request = ChonkyDBBulkRecordRequest(
                items=batch,
                execution_mode=execution_mode,
                timeout_seconds=async_job_timeout_s,
                client_request_id=request_correlation_id,
                finalize_vector_segments=finalize_vector_segments,
            )
            request_bytes = self._bulk_request_bytes(request)
            started_monotonic = time.monotonic()
            try:
                result = getattr(write_client, "store_records_bulk")(request)
                failure_detail = self._store_result_failure_detail(result)
                if failure_detail:
                    raise ChonkyDBError(
                        f"ChonkyDB rejected remote catalog write: {failure_detail}",
                        response_json=dict(result) if isinstance(result, Mapping) else None,
                    )
                extracted_document_ids = self._extract_document_ids(result, expected=len(batch))
                job_document_ids = self._await_async_job_document_ids(
                    write_client,
                    result=result,
                    expected=len(batch),
                )
                if job_document_ids is not None:
                    extracted_document_ids = job_document_ids
                if all(isinstance(value, str) and value for value in extracted_document_ids):
                    resolved_document_ids = extracted_document_ids
                else:
                    resolved_document_ids = self._attest_record_batch_readback(
                        snapshot_kind=snapshot_kind,
                        record_items=batch,
                        document_ids=extracted_document_ids,
                    )
                record_remote_write_observation(
                    remote_state=self._require_remote_state(),
                    context=LongTermRemoteWriteContext(
                        snapshot_kind=snapshot_kind,
                        operation="store_records_bulk",
                        request_method="POST",
                        request_payload_kind=request_payload_kind,
                        request_path="/v1/external/records/bulk",
                        timeout_s=async_job_timeout_s,
                        namespace=self._normalize_text(getattr(self._require_remote_state(), "namespace", None)),
                        access_classification=access_classification,
                        attempt_count=attempt_index + 1,
                        request_item_count=len(batch),
                        request_correlation_id=request_correlation_id,
                        batch_index=batch_index,
                        batch_count=batch_count,
                        request_bytes=request_bytes,
                        uri_hint=self._normalize_text(getattr(batch[0], "uri", None)) if len(batch) == 1 else None,
                    ),
                    latency_ms=max(0.0, (time.monotonic() - started_monotonic) * 1000.0),
                    outcome="ok",
                    classification="ok",
                )
                return resolved_document_ids
            except Exception as exc:
                retry_attempts = retryable_remote_write_attempts(retry_attempts, exc=exc)
                if should_retry_remote_write_error(exc) and attempt_index + 1 < retry_attempts:
                    delay_s = remote_write_retry_delay_s(
                        exc,
                        default_backoff_s=retry_backoff_s,
                        attempt_index=attempt_index,
                    )
                    if delay_s > 0.0:
                        time.sleep(delay_s)
                    attempt_index += 1
                    continue
                remote_state = self._require_remote_state()
                remote_write_context = {
                    "snapshot_kind": snapshot_kind,
                    "operation": "store_records_bulk",
                    "request_correlation_id": request_correlation_id,
                    "batch_index": batch_index,
                    "batch_count": batch_count,
                    "request_item_count": len(batch),
                    "request_bytes": request_bytes,
                }
                record_remote_write_diagnostic(
                    remote_state=remote_state,
                    context=LongTermRemoteWriteContext(
                        snapshot_kind=snapshot_kind,
                        operation="store_records_bulk",
                        request_method="POST",
                        request_payload_kind=request_payload_kind,
                        request_path="/v1/external/records/bulk",
                        timeout_s=async_job_timeout_s,
                        namespace=self._normalize_text(getattr(remote_state, "namespace", None)),
                        access_classification=access_classification,
                        attempt_count=attempt_index + 1,
                        request_item_count=len(batch),
                        request_correlation_id=request_correlation_id,
                        batch_index=batch_index,
                        batch_count=batch_count,
                        request_bytes=request_bytes,
                    ),
                    exc=exc,
                    started_monotonic=started_monotonic,
                    outcome="failed",
                )
                unavailable = LongTermRemoteUnavailableError(
                    "Failed to persist fine-grained remote long-term memory items "
                    f"(request_id={request_correlation_id}, batch={batch_index}/{batch_count}, "
                    f"items={len(batch)}, bytes={request_bytes})."
                )
                setattr(unavailable, "remote_write_context", remote_write_context)
                raise unavailable from exc

    def _record_batch_payload_kind(self, *, batch: tuple[ChonkyDBRecordItem, ...]) -> str:
        """Classify one bulk-write batch by the dominant URI contract it carries."""

        uris = [
            self._normalize_text(getattr(item, "uri", None))
            for item in batch
            if self._normalize_text(getattr(item, "uri", None))
        ]
        if uris and all(uri.endswith("/catalog/current") for uri in uris):
            return "catalog_current_head_record_batch"
        if uris and all("/catalog/segment/" in uri for uri in uris):
            return "catalog_segment_record_batch"
        return "fine_grained_record_batch"

    def _record_batch_access_classification(self, *, batch: tuple[ChonkyDBRecordItem, ...]) -> str:
        """Return the coarse live-path class for one bulk-write batch."""

        payload_kind = self._record_batch_payload_kind(batch=batch)
        if payload_kind == "catalog_current_head_record_batch":
            return "catalog_current_head_write"
        return "record_bulk_write"

    def _bulk_execution_mode(
        self,
        *,
        snapshot_kind: str,
        batch: tuple[ChonkyDBRecordItem, ...],
        request_payload_kind: str,
    ) -> str:
        """Choose sync only for tiny one-item control-plane batches.

        Conflict and graph current-view writes are tiny control-plane batches
        whose async job + readback visibility overhead can dominate the actual
        write. Keeping these batches synchronous removes that fragility without
        changing the broader async strategy for heavier object persistence.
        """

        if snapshot_kind not in _SYNC_SMALL_CONTROL_PLANE_SNAPSHOT_KINDS or len(batch) != 1:
            return "async"
        if request_payload_kind not in {
            "fine_grained_record_batch",
            "catalog_segment_record_batch",
            "catalog_current_head_record_batch",
        }:
            return "async"
        preview_request = ChonkyDBBulkRecordRequest(
            items=batch,
            execution_mode="sync",
            timeout_seconds=self._remote_async_job_timeout_s(),
            client_request_id="preview",
            finalize_vector_segments=True,
        )
        if self._bulk_request_bytes(preview_request) > _SYNC_SMALL_CONTROL_PLANE_WRITE_MAX_BYTES:
            return "async"
        return "sync"

    @staticmethod
    def _new_remote_write_correlation_id() -> str:
        return f"ltw-{uuid4().hex[:12]}"

    @staticmethod
    def _bulk_request_bytes(request: ChonkyDBBulkRecordRequest) -> int:
        return len(json.dumps(request.to_payload(), ensure_ascii=False).encode("utf-8"))

    def _attest_record_batch_readback(
        self,
        *,
        snapshot_kind: str,
        record_items: tuple[ChonkyDBRecordItem, ...],
        document_ids: tuple[str | None, ...],
    ) -> tuple[str | None, ...]:
        remote_state = self._require_remote_state()
        read_client = self._require_client(getattr(remote_state, "read_client", None), operation="read")
        indexed_items = tuple(enumerate(record_items))

        def attest_one(indexed_item: tuple[int, ChonkyDBRecordItem]) -> str | None:
            index, item = indexed_item
            document_id = document_ids[index] if index < len(document_ids) else None
            return self._attest_record_readback(
                read_client,
                snapshot_kind=snapshot_kind,
                record_item=item,
                document_id=document_id,
            )

        max_workers = self._remote_read_max_workers(len(indexed_items))
        if max_workers <= 1:
            return tuple(attest_one(indexed_item) for indexed_item in indexed_items)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return tuple(executor.map(attest_one, indexed_items))

    def _await_async_job_document_ids(
        self,
        client: object,
        *,
        result: object,
        expected: int,
    ) -> tuple[str | None, ...] | None:
        if not isinstance(result, Mapping):
            return None
        initial_document_ids = self._extract_document_ids(result, expected=expected)
        if all(isinstance(value, str) and value for value in initial_document_ids):
            return initial_document_ids
        job_id = self._normalize_text(result.get("job_id"))
        job_status = getattr(client, "job_status", None)
        if not job_id or not callable(job_status):
            return None
        poll_interval_s = max(self._remote_retry_backoff_s(), _DEFAULT_ASYNC_ATTESTATION_POLL_S)
        total_timeout_s = self._async_job_visibility_timeout_s()
        deadline = time.monotonic() + total_timeout_s
        while True:
            remaining_timeout_s = deadline - time.monotonic()
            if remaining_timeout_s <= 0.0:
                break
            capped_client = clone_client_with_capped_timeout(client, timeout_s=remaining_timeout_s)
            capped_job_status = getattr(capped_client, "job_status", None)
            try:
                payload = job_status(job_id) if not callable(capped_job_status) else capped_job_status(job_id)
            except Exception as exc:
                if not should_fallback_async_job_resolution_error(exc):
                    raise
            else:
                if isinstance(payload, Mapping):
                    status = self._normalize_text(payload.get("status")).lower()
                    if status in {"failed", "cancelled", "rejected"}:
                        detail = self._store_result_failure_detail(payload.get("result"))
                        if not detail:
                            detail = (
                                self._normalize_text(payload.get("error"))
                                or self._normalize_text(payload.get("error_type"))
                                or f"async job status={status}"
                            )
                        raise LongTermRemoteUnavailableError(
                            f"Accepted async remote write job {job_id!r} failed before readback: {detail}"
                        )
                    document_ids = self._extract_document_ids(payload.get("result"), expected=expected)
                    if not all(document_ids):
                        document_ids = self._extract_document_ids(payload, expected=expected)
                    if all(isinstance(value, str) and value for value in document_ids):
                        return document_ids
                    if status in {"succeeded", "done"}:
                        return None
            remaining_sleep_s = deadline - time.monotonic()
            if remaining_sleep_s <= 0.0:
                break
            if poll_interval_s > 0.0:
                time.sleep(min(poll_interval_s, remaining_sleep_s))
        return None

    def _attest_record_readback(
        self,
        read_client: object,
        *,
        snapshot_kind: str,
        record_item: ChonkyDBRecordItem,
        document_id: str | None,
    ) -> str | None:
        expected_payloads = self._expected_record_payload_candidates(record_item)
        if not expected_payloads:
            return document_id
        resolved_uri = self._normalize_text(getattr(record_item, "uri", None))
        if document_id is None and not resolved_uri:
            raise LongTermRemoteUnavailableError(
                f"Accepted remote long-term {snapshot_kind!r} write cannot be attested without document id or uri."
            )
        resolved_attempts = max(1, self._remote_retry_attempts())
        poll_interval_s = max(self._remote_retry_backoff_s(), _DEFAULT_ASYNC_ATTESTATION_POLL_S)
        if poll_interval_s > 0.0:
            resolved_attempts = max(
                resolved_attempts,
                int(math.ceil(self._async_attestation_visibility_timeout_s() / poll_interval_s)),
            )
        last_detail = "Remote write attestation did not observe the accepted payload."
        for attempt in range(resolved_attempts):
            for probe_document_id, probe_uri in self._attestation_probe_targets(
                document_id=document_id,
                resolved_uri=resolved_uri or None,
            ):
                try:
                    envelope = read_client.fetch_full_document(
                        document_id=probe_document_id,
                        origin_uri=probe_uri,
                        include_content=True,
                        max_content_chars=self._max_content_chars(),
                    )
                except Exception as exc:
                    last_detail = str(exc)
                    continue
                matched_document_id = self._match_attested_record_document_id(
                    envelope,
                    expected_payloads=expected_payloads,
                )
                if matched_document_id is not None:
                    return matched_document_id
                last_detail = "Remote write attestation read back a different same-uri document."
            if attempt + 1 >= resolved_attempts:
                break
            if poll_interval_s > 0.0:
                time.sleep(poll_interval_s)
        raise LongTermRemoteUnavailableError(
            f"Accepted remote long-term {snapshot_kind!r} write could not be read back: {last_detail}"
        )

    def _attestation_probe_targets(
        self,
        *,
        document_id: str | None,
        resolved_uri: str | None,
    ) -> tuple[tuple[str | None, str | None], ...]:
        """Return the bounded probe order for one accepted record attestation.

        ChonkyDB can acknowledge an async batch and even advertise the final
        ``document_id`` before that exact-id lookup becomes readable. Polling
        only once on the advertised id produces false-negative write failures on
        otherwise healthy backends. Keep the exact-id probe first, then fall
        back to the stable same-URI lookup when available.
        """

        targets: list[tuple[str | None, str | None]] = []
        if document_id:
            targets.append((document_id, None))
        if resolved_uri:
            targets.append((None, resolved_uri))
        if not targets:
            return ((None, None),)
        return tuple(targets)

    def _expected_record_payload_candidates(
        self,
        record_item: ChonkyDBRecordItem,
    ) -> tuple[Mapping[str, object], ...]:
        candidates: list[Mapping[str, object]] = []
        direct_payload = getattr(record_item, "payload", None)
        if isinstance(direct_payload, Mapping):
            candidates.append(dict(direct_payload))
            candidates.extend(_iter_known_item_envelopes(direct_payload))
        metadata = getattr(record_item, "metadata", None)
        if isinstance(metadata, Mapping):
            twinr_payload = metadata.get("twinr_payload")
            if isinstance(twinr_payload, Mapping):
                candidates.append(dict(twinr_payload))
        content = getattr(record_item, "content", None)
        if isinstance(content, str):
            parsed_content = self._parse_json_mapping(content)
            if isinstance(parsed_content, Mapping):
                candidates.append(dict(parsed_content))
        unique: list[Mapping[str, object]] = []
        seen: set[str] = set()
        for candidate in candidates:
            marker = json.dumps(candidate, ensure_ascii=False, sort_keys=True)
            if marker in seen:
                continue
            seen.add(marker)
            unique.append(candidate)
        return tuple(unique)

    def _match_attested_record_document_id(
        self,
        payload: object,
        *,
        expected_payloads: tuple[Mapping[str, object], ...],
    ) -> str | None:
        if not isinstance(payload, Mapping):
            return None
        for candidate, candidate_document_id in self._iter_attestation_candidates(payload):
            if any(candidate == expected for expected in expected_payloads):
                return candidate_document_id
        return None

    def _iter_attestation_candidates(
        self,
        payload: Mapping[str, object],
        *,
        inherited_document_id: str | None = None,
    ) -> Iterable[tuple[Mapping[str, object], str | None]]:
        current_document_id = (
            self._normalize_text(payload.get("document_id"))
            or self._normalize_text(payload.get("payload_id"))
            or self._normalize_text(payload.get("chonky_id"))
            or inherited_document_id
        )
        yield payload, current_document_id
        for item_payload in _iter_known_item_envelopes(payload):
            yield item_payload, current_document_id
        direct_payload = payload.get("payload")
        if isinstance(direct_payload, Mapping):
            yield direct_payload, current_document_id
            for item_payload in _iter_known_item_envelopes(direct_payload):
                yield item_payload, current_document_id
        metadata = payload.get("metadata")
        if isinstance(metadata, Mapping):
            yield metadata, current_document_id
            for item_payload in _iter_known_item_envelopes(metadata):
                yield item_payload, current_document_id
            nested_payload = metadata.get("twinr_payload")
            if isinstance(nested_payload, Mapping):
                yield nested_payload, current_document_id
        content = payload.get("content")
        if isinstance(content, str):
            parsed_content = self._parse_json_mapping(content)
            if isinstance(parsed_content, Mapping):
                yield parsed_content, current_document_id
        for field_name in ("record", "document"):
            nested = payload.get(field_name)
            if isinstance(nested, Mapping):
                yield from self._iter_attestation_candidates(
                    nested,
                    inherited_document_id=current_document_id,
                )
        chunks = payload.get("chunks")
        if isinstance(chunks, list):
            for chunk in chunks:
                if not isinstance(chunk, Mapping):
                    continue
                for item_payload in _iter_known_item_envelopes(chunk):
                    yield item_payload, current_document_id
                yield from self._iter_attestation_candidates(
                    chunk,
                    inherited_document_id=current_document_id,
                )

    @staticmethod
    def _store_result_failure_detail(result: object) -> str | None:
        if not isinstance(result, Mapping):
            return None
        failures: list[str] = []
        items = result.get("items")
        if isinstance(items, list):
            for index, item in enumerate(items):
                if not isinstance(item, Mapping):
                    continue
                if item.get("success") is False:
                    error_type = " ".join(str(item.get("error_type") or "").split()).strip()
                    error_text = " ".join(str(item.get("error") or "").split()).strip()
                    if error_type and error_text:
                        failures.append(f"item[{index}] {error_type}: {error_text}")
                    elif error_text:
                        failures.append(f"item[{index}] {error_text}")
                    else:
                        failures.append(f"item[{index}] rejected")
        if failures:
            return "; ".join(failures)
        if result.get("success") is False:
            error_type = " ".join(str(result.get("error_type") or "").split()).strip()
            error_text = " ".join(str(result.get("error") or "").split()).strip()
            if error_type and error_text:
                return f"{error_type}: {error_text}"
            if error_text:
                return error_text
            return "request rejected"
        return None

    def _split_bulk_record_batches(
        self,
        record_items: list[ChonkyDBRecordItem],
    ) -> tuple[tuple[ChonkyDBRecordItem, ...], ...]:
        if not record_items:
            return ()
        batch_size = self._bulk_write_batch_size()
        max_request_bytes = self._bulk_request_max_bytes()
        empty_request_bytes = len(
            json.dumps(
                ChonkyDBBulkRecordRequest(items=(), finalize_vector_segments=False).to_payload(),
                ensure_ascii=False,
            ).encode("utf-8")
        )
        batches: list[tuple[ChonkyDBRecordItem, ...]] = []
        current_batch: list[ChonkyDBRecordItem] = []
        current_bytes = empty_request_bytes
        for item in record_items:
            item_bytes = len(json.dumps(item.to_payload(), ensure_ascii=False).encode("utf-8"))
            separator_bytes = 2 if current_batch else 0
            if current_batch and (
                len(current_batch) >= batch_size
                or current_bytes + separator_bytes + item_bytes > max_request_bytes
            ):
                batches.append(tuple(current_batch))
                current_batch = []
                current_bytes = empty_request_bytes
                separator_bytes = 0
            current_batch.append(item)
            current_bytes += separator_bytes + item_bytes
        if current_batch:
            batches.append(tuple(current_batch))
        return tuple(batches)

    def _extract_document_ids(self, payload: object, *, expected: int) -> tuple[str | None, ...]:
        if not isinstance(payload, Mapping):
            return tuple(None for _ in range(expected))
        items = payload.get("items")
        if not isinstance(items, list):
            return tuple(None for _ in range(expected))
        document_ids: list[str | None] = []
        for item in items[:expected]:
            if not isinstance(item, Mapping):
                document_ids.append(None)
                continue
            document_ids.append(
                self._normalize_text(item.get("document_id"))
                or self._normalize_text(item.get("payload_id"))
                or self._normalize_text(item.get("chonky_id"))
            )
        while len(document_ids) < expected:
            document_ids.append(None)
        return tuple(document_ids)


__all__ = [
    "RemoteCatalogWriteMixin",
]
