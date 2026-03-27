"""Catalog persistence, segment IO, and write-attestation helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
import json
import math
import time
from uuid import uuid4

from twinr.memory.chonkydb.client import ChonkyDBError
from twinr.memory.chonkydb.models import ChonkyDBBulkRecordRequest, ChonkyDBRecordItem
from twinr.memory.longterm.storage.remote_read_diagnostics import (
    LongTermRemoteReadContext,
    LongTermRemoteWriteContext,
    record_remote_read_diagnostic,
    record_remote_write_diagnostic,
)
from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError

from .shared import (
    LongTermRemoteCatalogEntry,
    _CATALOG_VERSION,
    _SEGMENT_VERSION,
    _DEFAULT_ASYNC_ATTESTATION_POLL_S,
    _ITEM_VERSION,
    _RemoteCollectionDefinition,
)


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
        existing_entries = {
            entry.item_id: entry
            for entry in self.load_catalog_entries(snapshot_kind=snapshot_kind)
        }
        record_items: list[ChonkyDBRecordItem] = []
        staged: list[tuple[str, dict[str, object], str]] = []
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
                    if value is not None:
                        metadata[str(key)] = value
            content_text = content_builder(payload)
            record_items.append(
                ChonkyDBRecordItem(
                    payload={
                        "schema": definition.item_schema,
                        "version": _ITEM_VERSION,
                        "snapshot_kind": definition.snapshot_kind,
                        "item_id": item_id,
                        "metadata": {
                            "twinr_snapshot_kind": definition.snapshot_kind,
                            "twinr_memory_item_id": item_id,
                        },
                        "content": content_text,
                    },
                    metadata=metadata,
                    content=content_text,
                    uri=uri,
                    enable_chunking=False,
                    include_insights_in_response=False,
                )
            )
            staged.append((item_id, catalog_metadata, payload_sha256))
        document_ids = self._store_record_items(
            write_client,
            snapshot_kind=definition.snapshot_kind,
            record_items=record_items,
        )
        for index, (item_id, metadata, _payload_sha256) in enumerate(staged):
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
        catalog_entries = [
            self._build_catalog_entry(
                item_id=entry.item_id,
                document_id=entry.document_id,
                metadata=entry.metadata,
            )
            for entry in entries
        ]
        segment_refs = self._persist_catalog_segments(
            write_client,
            definition=definition,
            catalog_entries=catalog_entries,
        )
        payload = {
            "schema": definition.catalog_schema,
            "version": _CATALOG_VERSION,
            "items_count": len(catalog_entries),
            "segments": list(segment_refs),
        }
        if isinstance(written_at, str) and written_at.strip():
            payload["written_at"] = written_at
        remote_state.save_snapshot(snapshot_kind=snapshot_kind, payload=payload)
        return payload

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
        return catalog_entry

    def _can_reuse_existing_entry(
        self,
        *,
        snapshot_kind: str,
        entry: LongTermRemoteCatalogEntry | None,
        item_id: str,
        payload_sha256: str,
    ) -> bool:
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

        def load_one(request: tuple[int | None, str | None, str | None]) -> Mapping[str, object]:
            segment_index, document_id, uri = request
            return self._load_catalog_segment_payload(
                definition=definition,
                segment_index=segment_index,
                document_id=document_id,
                uri=uri,
            )

        max_workers = self._remote_read_max_workers(len(segment_requests))
        if max_workers <= 1:
            return tuple(load_one(request) for request in segment_requests)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return tuple(executor.map(load_one, segment_requests))

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
        started_monotonic = time.monotonic()
        last_error: Exception | None = None
        for attempt_index in range(retry_attempts):
            try:
                return self._fetch_catalog_segment_document(
                    read_client,
                    document_id=document_id,
                    fallback_uri=fallback_uri,
                )
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
            request_correlation_id = self._new_remote_write_correlation_id()
            async_job_timeout_s = self._remote_async_job_timeout_s()
            request = ChonkyDBBulkRecordRequest(
                items=tuple(batch),
                execution_mode="async",
                timeout_seconds=async_job_timeout_s,
                client_request_id=request_correlation_id,
                finalize_vector_segments=index + 1 >= len(batches),
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
                    document_ids.extend(extracted_document_ids)
                else:
                    document_ids.extend(
                        self._attest_record_batch_readback(
                            snapshot_kind=snapshot_kind,
                            record_items=batch,
                            document_ids=extracted_document_ids,
                        )
                    )
            except Exception as exc:
                remote_state = self._require_remote_state()
                remote_write_context = {
                    "snapshot_kind": snapshot_kind,
                    "operation": "store_records_bulk",
                    "request_correlation_id": request_correlation_id,
                    "batch_index": index + 1,
                    "batch_count": total_batches,
                    "request_item_count": len(batch),
                    "request_bytes": request_bytes,
                }
                record_remote_write_diagnostic(
                    remote_state=remote_state,
                    context=LongTermRemoteWriteContext(
                        snapshot_kind=snapshot_kind,
                        operation="store_records_bulk",
                        attempt_count=1,
                        request_item_count=len(batch),
                        request_correlation_id=request_correlation_id,
                        batch_index=index + 1,
                        batch_count=total_batches,
                        request_bytes=request_bytes,
                    ),
                    exc=exc,
                    started_monotonic=started_monotonic,
                    outcome="failed",
                )
                unavailable = LongTermRemoteUnavailableError(
                    "Failed to persist fine-grained remote long-term memory items "
                    f"(request_id={request_correlation_id}, batch={index + 1}/{total_batches}, "
                    f"items={len(batch)}, bytes={request_bytes})."
                )
                setattr(unavailable, "remote_write_context", remote_write_context)
                raise unavailable from exc
        return tuple(document_ids)

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
        resolved: list[str | None] = []
        for index, item in enumerate(record_items):
            document_id = document_ids[index] if index < len(document_ids) else None
            resolved.append(
                self._attest_record_readback(
                    read_client,
                    snapshot_kind=snapshot_kind,
                    record_item=item,
                    document_id=document_id,
                )
            )
        return tuple(resolved)

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
        attempts = max(1, int(math.ceil(total_timeout_s / poll_interval_s)))
        last_failure_detail = "async job did not expose document ids"
        for attempt in range(attempts):
            payload = job_status(job_id)
            if not isinstance(payload, Mapping):
                last_failure_detail = "async job status returned a non-mapping payload"
            else:
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
                    last_failure_detail = "async job finished without exposing document ids"
                elif status:
                    last_failure_detail = f"async job status={status}"
            if attempt + 1 >= attempts:
                break
            if poll_interval_s > 0.0:
                time.sleep(poll_interval_s)
        raise LongTermRemoteUnavailableError(
            f"Accepted async remote write job {job_id!r} could not be resolved to document ids: {last_failure_detail}"
        )

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
        resolved_attempts = 1 if document_id else self._remote_retry_attempts()
        poll_interval_s = 0.0 if document_id else max(self._remote_retry_backoff_s(), _DEFAULT_ASYNC_ATTESTATION_POLL_S)
        if document_id is None and poll_interval_s > 0.0:
            resolved_attempts = max(
                resolved_attempts,
                int(math.ceil(self._async_attestation_visibility_timeout_s() / poll_interval_s)),
            )
        last_detail = "Remote write attestation did not observe the accepted payload."
        for attempt in range(resolved_attempts):
            try:
                envelope = read_client.fetch_full_document(
                    document_id=document_id,
                    origin_uri=None if document_id else resolved_uri,
                    include_content=True,
                    max_content_chars=self._max_content_chars(),
                )
            except Exception as exc:
                last_detail = str(exc)
            else:
                matched_document_id = self._match_attested_record_document_id(
                    envelope,
                    expected_payloads=expected_payloads,
                )
                if matched_document_id is not None:
                    return matched_document_id
                last_detail = "Remote write attestation read back a different same-uri document."
            if document_id is not None or attempt + 1 >= resolved_attempts:
                break
            if poll_interval_s > 0.0:
                time.sleep(poll_interval_s)
        raise LongTermRemoteUnavailableError(
            f"Accepted remote long-term {snapshot_kind!r} write could not be read back: {last_detail}"
        )

    def _expected_record_payload_candidates(
        self,
        record_item: ChonkyDBRecordItem,
    ) -> tuple[Mapping[str, object], ...]:
        candidates: list[Mapping[str, object]] = []
        direct_payload = getattr(record_item, "payload", None)
        if isinstance(direct_payload, Mapping):
            candidates.append(dict(direct_payload))
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
        direct_payload = payload.get("payload")
        if isinstance(direct_payload, Mapping):
            yield direct_payload, current_document_id
        metadata = payload.get("metadata")
        if isinstance(metadata, Mapping):
            yield metadata, current_document_id
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
