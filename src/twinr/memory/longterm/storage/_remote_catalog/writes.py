"""Catalog persistence, segment IO, and write-attestation helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import time
from typing import Any, cast
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
    exception_chain,
    is_rate_limited_remote_write_error,
    remote_write_retry_delay_s,
    raise_if_remote_operation_cancelled,
    retryable_remote_write_attempts,
    sleep_with_remote_operation_abort,
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

from ._typing import RemoteCatalogMixinBase
from .shared import (
    _ALLOWED_DOC_IDS_RETRIEVE_QUERY,
    LongTermRemoteCatalogEntry,
    _CATALOG_VERSION,
    _DEFAULT_ASYNC_ATTESTATION_POLL_S,
    _DEFAULT_REMOTE_READ_TIMEOUT_S,
    _ITEM_VERSION,
    _RemoteCollectionDefinition,
    _SEGMENT_VERSION,
    _iter_known_item_envelopes,
    _segment_ref_supports_uri_only_contract,
    _run_timed_workflow_step,
)

_SYNC_SMALL_CONTROL_PLANE_WRITE_MAX_BYTES = 16_384
_SYNC_LARGE_GRAPH_CURRENT_HEAD_WRITE_MAX_BYTES = 131_072
_GRAPH_PROJECTION_SEGMENT_REQUEST_MAX_BYTES = 16_384
_GRAPH_SEGMENT_BULK_REQUEST_MAX_BYTES = 131_072
_SYNC_SMALL_SEGMENT_WRITE_MAX_BYTES = 36_864
_EXTENDED_ASYNC_CURRENT_HEAD_RETRY_ATTEMPTS = 9
_EXTENDED_ASYNC_CURRENT_HEAD_MAX_BACKOFF_S = 20.0
_EXTENDED_ASYNC_POST_SYNC_RESCUE_RETRY_ATTEMPTS = 9
_EXTENDED_ASYNC_POST_SYNC_RESCUE_MAX_BACKOFF_S = 20.0
# Default sync writes stay disabled. Fine-grained writes and fixed-URI catalog
# heads both start async so fresh namespaces do not burn their initial write on
# the inline lane before the jobs lane can absorb the cold-path work.
# Projection-complete control-plane writers may still take one tiny sync rescue
# after a proven async `429`, because they already have a bounded follow-up
# proof surface and the rescue path inherits the longer flush budget.
_SYNC_SMALL_CONTROL_PLANE_SNAPSHOT_KINDS = frozenset()
_SYNC_DEFERRED_ID_CURRENT_HEAD_SNAPSHOT_KINDS = frozenset(
    {
        "archive",
        "conflicts",
        "graph_edges",
        "graph_nodes",
        "midterm",
        "objects",
    }
)
_SYNC_DEFERRED_ID_SEGMENT_SNAPSHOT_KINDS = frozenset(
    {
        "archive",
        "conflicts",
        "graph_edges",
        "graph_nodes",
        "midterm",
        "objects",
    }
)
_SYNC_SMALL_PROJECTION_COMPLETE_SEGMENT_SNAPSHOT_KINDS = frozenset()
_PROJECTION_COMPLETE_FINE_GRAINED_SNAPSHOT_KINDS = frozenset(
    {
        "objects",
        "conflicts",
        "archive",
        "graph_edges",
        "graph_nodes",
        "midterm",
    }
)
_ASYNC_JOB_BACKPRESSURE_FINE_GRAINED_SNAPSHOT_KINDS = frozenset(
    {
        "objects",
        "conflicts",
        "archive",
    }
)
_ASYNC_JOB_BACKPRESSURE_SEGMENT_SNAPSHOT_KINDS = _SYNC_DEFERRED_ID_SEGMENT_SNAPSHOT_KINDS
_PROJECTION_ONLY_CATALOG_SNAPSHOT_KINDS = frozenset(
    {
        "graph_edges",
        "graph_nodes",
    }
)
_SYNC_DEFAULT_CURRENT_HEAD_SNAPSHOT_KINDS = frozenset(
    _SYNC_DEFERRED_ID_CURRENT_HEAD_SNAPSHOT_KINDS - _PROJECTION_ONLY_CATALOG_SNAPSHOT_KINDS
)


@dataclass(frozen=True, slots=True)
class _AttestedRecordMatch:
    """Represent one successful record attestation candidate.

    Same-URI ChonkyDB reads can return the expected Twinr payload without an
    exact `document_id`. The attestation path must treat that as a successful
    visibility proof instead of overloading `None` as both "matched without an
    id" and "no matching payload was found".
    """

    document_id: str | None


class RemoteCatalogWriteMixin(RemoteCatalogMixinBase):
    def repair_invalid_current_head_from_segment_uris(
        self,
        *,
        snapshot_kind: str,
    ) -> dict[str, object]:
        """Repair one invalid current head by re-attesting its existing segment URIs.

        Structured current heads can become unreadable when the fixed
        `.../catalog/current` document is visible but one or more segment refs
        still lack a stable `document_id`. The segment documents themselves are
        already authoritative at their versioned URIs, so repairing this case
        should not rebuild state from local snapshots or stale compatibility
        blobs. Instead, re-read the advertised segment URIs, recover the exact
        segment document ids from those envelopes, and republish the same head
        payload with completed segment refs.
        """

        definition = self._require_definition(snapshot_kind)
        if definition.snapshot_kind in _PROJECTION_ONLY_CATALOG_SNAPSHOT_KINDS:
            raise ValueError(f"Projection-only current head repair is not supported for {snapshot_kind!r}.")
        remote_state = self._require_remote_state()
        read_client = self._require_client(getattr(remote_state, "read_client", None), operation="read")
        envelope = self._fetch_catalog_head_envelope(
            read_client=self._catalog_head_read_client(read_client=read_client),
            snapshot_kind=snapshot_kind,
            metadata_only=False,
        )
        raw_payload = self._repairable_segmented_catalog_payload_candidate(
            definition=definition,
            envelope=envelope,
        )
        if raw_payload is None:
            raise LongTermRemoteUnavailableError(
                f"Required remote long-term {snapshot_kind!r} current catalog head is invalid and not repairable."
            )
        repaired_payload = self._repair_segment_ref_document_ids(
            definition=definition,
            payload=raw_payload,
        )
        self.persist_catalog_payload(snapshot_kind=snapshot_kind, payload=repaired_payload)
        self._clear_invalid_catalog_head(snapshot_kind=snapshot_kind)
        return repaired_payload

    def _repairable_segmented_catalog_payload_candidate(
        self,
        *,
        definition: _RemoteCollectionDefinition,
        envelope: Mapping[str, object],
    ) -> dict[str, object] | None:
        """Extract one segmented current-head payload even when read-contract invalid."""

        for candidate in self._iter_record_candidates(envelope):
            if candidate.get("schema") != definition.catalog_schema:
                continue
            if candidate.get("version") != _CATALOG_VERSION:
                continue
            segments = candidate.get("segments")
            if not isinstance(segments, list):
                continue
            items_count = self._normalize_segment_index(candidate.get("items_count"))
            if items_count is None:
                continue
            if items_count > 0 and not segments:
                continue
            return dict(candidate)
        return None

    def _repair_segment_ref_document_ids(
        self,
        *,
        definition: _RemoteCollectionDefinition,
        payload: Mapping[str, object],
    ) -> dict[str, object]:
        """Rebuild only the missing segment document ids of one current head."""

        segments = payload.get("segments")
        if not isinstance(segments, list):
            raise LongTermRemoteUnavailableError(
                f"Required remote long-term {definition.snapshot_kind!r} current catalog head is missing its segments."
            )
        segment_requests: list[tuple[int | None, str | None, str | None]] = []
        expected_entry_counts: list[int | None] = []
        for raw_segment in segments:
            if not isinstance(raw_segment, Mapping):
                raise LongTermRemoteUnavailableError(
                    f"Required remote long-term {definition.snapshot_kind!r} current catalog head contains a malformed segment ref."
                )
            segment_index = self._normalize_segment_index(raw_segment.get("segment_index"))
            uri = self._normalize_text(raw_segment.get("uri"))
            document_id = self._normalize_text(raw_segment.get("document_id"))
            if segment_index is None or not uri:
                raise LongTermRemoteUnavailableError(
                    f"Required remote long-term {definition.snapshot_kind!r} current catalog head contains an unreadable segment ref."
                )
            segment_requests.append((segment_index, document_id, uri))
            expected_entry_counts.append(self._normalize_segment_index(raw_segment.get("entry_count")))

        segment_envelopes = self._load_catalog_segment_payloads(
            definition=definition,
            segment_requests=tuple(segment_requests),
        )
        if len(segment_envelopes) != len(segment_requests):
            raise LongTermRemoteUnavailableError(
                f"Required remote long-term {definition.snapshot_kind!r} current catalog head repair could not reload every segment."
            )

        repaired_segments: list[dict[str, object]] = []
        total_entries = 0
        for raw_segment, expected_entry_count, envelope in zip(segments, expected_entry_counts, segment_envelopes, strict=False):
            assert isinstance(raw_segment, Mapping)
            segment_entries = self._extract_segment_entries(
                definition=definition,
                payload=envelope,
            )
            recovered_document_id = None
            for candidate_payload, candidate_document_id in self._iter_attestation_candidates(envelope):
                if candidate_payload.get("schema") != definition.segment_schema:
                    continue
                if candidate_payload.get("version") != _SEGMENT_VERSION:
                    continue
                if candidate_payload.get("snapshot_kind") != definition.snapshot_kind:
                    continue
                if not isinstance(candidate_payload.get("items"), list):
                    continue
                recovered_document_id = self._normalize_text(candidate_document_id)
                break
            if not recovered_document_id:
                raise LongTermRemoteUnavailableError(
                    f"Required remote long-term {definition.snapshot_kind!r} current catalog head repair could not attest a stable segment document id."
                )
            if expected_entry_count is not None and len(segment_entries) != expected_entry_count:
                raise LongTermRemoteUnavailableError(
                    f"Required remote long-term {definition.snapshot_kind!r} current catalog head repair observed a segment entry-count mismatch."
                )
            total_entries += len(segment_entries)
            repaired_segment = dict(raw_segment)
            repaired_segment["document_id"] = recovered_document_id
            repaired_segments.append(repaired_segment)

        items_count = self._normalize_segment_index(payload.get("items_count")) or 0
        if items_count != total_entries:
            raise LongTermRemoteUnavailableError(
                f"Required remote long-term {definition.snapshot_kind!r} current catalog head repair observed an item-count mismatch."
            )
        repaired_payload = dict(payload)
        repaired_payload["segments"] = repaired_segments
        if not self.is_catalog_payload(snapshot_kind=definition.snapshot_kind, payload=repaired_payload):
            raise LongTermRemoteUnavailableError(
                f"Required remote long-term {definition.snapshot_kind!r} current catalog head repair did not produce a readable head."
            )
        return repaired_payload

    def _catalog_payload_has_unreadable_segments(
        self,
        *,
        snapshot_kind: str,
        payload: Mapping[str, object] | None,
    ) -> bool:
        """Return whether one current head exists but its segment refs cannot hydrate."""

        if not self.is_catalog_payload(snapshot_kind=snapshot_kind, payload=payload):
            return False
        try:
            self.load_catalog_entries(
                snapshot_kind=snapshot_kind,
                payload=dict(cast(Mapping[str, object], payload)),
            )
        except LongTermRemoteUnavailableError:
            return True
        return False

    def build_catalog_payload(
        self,
        *,
        snapshot_kind: str,
        item_payloads: Iterable[Mapping[str, object]],
        item_id_getter,
        metadata_builder,
        content_builder,
        replace_invalid_current_head: bool = False,
        attest_readback: bool = True,
        skip_async_document_id_wait: bool = False,
    ) -> dict[str, object]:
        """Persist individual remote items and return the small current catalog."""

        self._clear_read_cache(snapshot_kind=snapshot_kind)
        remote_state = self._require_remote_state()
        write_client = self._require_client(getattr(remote_state, "write_client", None), operation="write")
        definition = self._require_definition(snapshot_kind)
        try:
            existing_entries = (
                {}
                if definition.snapshot_kind in {"graph_nodes", "graph_edges"}
                else {
                    entry.item_id: entry
                    for entry in self._load_catalog_entries_for_write(snapshot_kind=snapshot_kind)
                }
            )
        except LongTermRemoteUnavailableError:
            if not replace_invalid_current_head:
                raise
            head_status, head_payload = self.probe_catalog_payload_result(snapshot_kind=snapshot_kind)
            if head_status != "invalid" and not self._catalog_payload_has_unreadable_segments(
                snapshot_kind=snapshot_kind,
                payload=head_payload,
            ):
                raise
            existing_entries = {}
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
            if definition.snapshot_kind in _PROJECTION_ONLY_CATALOG_SNAPSHOT_KINDS:
                # Graph current-view reads hydrate nodes/edges from selection
                # projections carried on segment/current-head catalog entries.
                # Persisting a second fine-grained item-document lane on
                # `records/bulk` only duplicates the same payload and can
                # saturate the async queue without improving reader behavior.
                catalog_entries.append(
                    self._build_catalog_entry(
                        item_id=item_id,
                        document_id=None,
                        metadata=catalog_metadata,
                    )
                )
                continue
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
                    target_indexes=self._searchable_write_target_indexes(),
                    enable_chunking=False,
                    include_insights_in_response=False,
                )
            )
            staged.append((item_id, catalog_metadata, payload_sha256, payload))
        document_ids = self._store_record_items(
            write_client,
            snapshot_kind=definition.snapshot_kind,
            record_items=record_items,
            attest_readback=attest_readback,
            skip_async_document_id_wait=skip_async_document_id_wait,
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
            attest_readback=attest_readback,
            skip_async_document_id_wait=skip_async_document_id_wait,
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
        skip_async_document_id_wait: bool = False,
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
            skip_async_document_id_wait=skip_async_document_id_wait,
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
        self.persist_catalog_payload(
            snapshot_kind=snapshot_kind,
            payload=payload,
            skip_async_document_id_wait=skip_async_document_id_wait,
        )
        return payload

    def persist_catalog_payload(
        self,
        *,
        snapshot_kind: str,
        payload: Mapping[str, object],
        attest_readback: bool = True,
        skip_async_document_id_wait: bool = False,
    ) -> dict[str, object]:
        """Persist the authoritative current catalog head as one fixed-URI record."""

        payload_dict = dict(payload)
        if not self.is_catalog_payload(snapshot_kind=snapshot_kind, payload=payload_dict):
            raise ValueError(f"Remote catalog head for {snapshot_kind!r} must be a supported catalog payload.")
        remote_state = self._require_remote_state()
        write_client = self._require_client(getattr(remote_state, "write_client", None), operation="write")
        definition = self._require_definition(snapshot_kind)
        written_at = self._normalize_text(payload_dict.get("written_at"))
        items_count = self._normalize_segment_index(payload_dict.get("items_count")) or 0
        metadata: dict[str, object] = {
            "twinr_snapshot_kind": definition.snapshot_kind,
            "twinr_catalog_current_head": True,
            "twinr_catalog_schema": str(payload_dict.get("schema") or ""),
            "twinr_catalog_items_count": max(0, items_count),
        }
        if definition.snapshot_kind not in _PROJECTION_ONLY_CATALOG_SNAPSHOT_KINDS:
            metadata["twinr_payload"] = dict(payload_dict)
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
                    target_indexes=self._control_plane_write_target_indexes(),
                    enable_chunking=False,
                    include_insights_in_response=False,
                )
            ],
            attest_readback=attest_readback,
            skip_async_document_id_wait=skip_async_document_id_wait,
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
        include_selection_projection: bool = True,
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
                        metadata=self._catalog_entry_metadata_from_mapping(
                            metadata if isinstance(metadata, Mapping) else {},
                            include_selection_projection=include_selection_projection,
                        ),
                    )
                )
        return tuple(entries)

    def _load_segmented_catalog_entries(
        self,
        *,
        definition: _RemoteCollectionDefinition,
        payload: Mapping[str, object],
        fast_fail: bool = False,
        include_selection_projection: bool = True,
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
        segment_payloads = self._load_catalog_segment_payloads(
            definition=definition,
            segment_requests=tuple(segment_requests),
            fast_fail=fast_fail,
        )
        extracted_entries = _run_timed_workflow_step(
            name="longterm_remote_catalog_extract_segment_entries",
            kind="retrieval",
            details={
                "snapshot_kind": definition.snapshot_kind,
                "segment_count": len(segment_payloads),
                "include_selection_projection": include_selection_projection,
            },
            operation=lambda: tuple(
                entry
                for envelope in segment_payloads
                for entry in self._extract_segment_entries(
                    definition=definition,
                    payload=envelope,
                    include_selection_projection=include_selection_projection,
                )
            ),
        )
        entries.extend(extracted_entries)
        return tuple(entries)

    def _load_catalog_segment_payloads(
        self,
        *,
        definition: _RemoteCollectionDefinition,
        segment_requests: tuple[tuple[int | None, str | None, str | None], ...],
        fast_fail: bool = False,
    ) -> tuple[Mapping[str, object], ...]:
        """Load segmented catalog documents with bounded parallel reads."""

        if not segment_requests:
            return ()

        if len(segment_requests) == 1:
            segment_index, document_id, uri = segment_requests[0]
            # A one-segment current head already has the exact segment target.
            # Sending that through the query-batch lane adds an avoidable
            # retrieve/top-k roundtrip before the same direct document read we
            # can perform immediately here.
            return (
                self._load_catalog_segment_payload(
                    definition=definition,
                    segment_index=segment_index,
                    document_id=document_id,
                    uri=uri,
                    fast_fail=fast_fail,
                ),
            )

        remote_state = self._require_remote_state()
        read_client = self._require_client(getattr(remote_state, "read_client", None), operation="read")
        payloads_by_document_id = self._load_catalog_segment_payloads_via_query(
            definition=definition,
            read_client=read_client,
            segment_requests=segment_requests,
            fast_fail=fast_fail,
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
                fast_fail=fast_fail,
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
        fast_fail: bool = False,
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
        retry_attempts = 1 if fast_fail else self._remote_retry_attempts()
        retry_backoff_s = 0.0 if fast_fail else self._remote_retry_backoff_s()
        loaded: dict[str, Mapping[str, object]] = {}
        for batch in self._split_retrieve_doc_id_batches(tuple(dict.fromkeys(document_ids))):
            last_error: Exception | None = None
            for attempt_index in range(retry_attempts):
                try:
                    candidates = self._load_catalog_segment_batch_candidates(
                        snapshot_kind=definition.snapshot_kind,
                        read_client=read_client,
                        batch=batch,
                        fast_fail=fast_fail,
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
                raise LongTermRemoteUnavailableError(
                    f"Failed to query remote long-term {definition.snapshot_kind!r} catalog segments."
                ) from last_error
        return loaded

    def _load_catalog_segment_batch_candidates(
        self,
        *,
        snapshot_kind: str,
        read_client: Any,
        batch: tuple[str, ...],
        fast_fail: bool = False,
    ) -> tuple[Mapping[str, object], ...]:
        """Load segment docs through content-bearing query batches, not documents/full."""

        if not batch:
            return ()
        remote_state = self._require_remote_state()
        topk_records = getattr(read_client, "topk_records", None)
        supports_topk_records = bool(getattr(read_client, "supports_topk_records", callable(topk_records)))
        allowed_indexes = self._selection_hydration_allowed_indexes()
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
                            allowed_indexes=allowed_indexes,
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
                if fast_fail:
                    raise

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
        fast_fail: bool = False,
    ) -> Mapping[str, object]:
        remote_state = self._require_remote_state()
        read_client = self._require_client(getattr(remote_state, "read_client", None), operation="read")
        fallback_uri = uri or self._catalog_segment_uri(
            snapshot_kind=definition.snapshot_kind,
            segment_index=segment_index or 0,
        )
        retry_attempts = 1 if fast_fail else self._remote_retry_attempts()
        retry_backoff_s = 0.0 if fast_fail else self._remote_retry_backoff_s()
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
                    fast_fail=fast_fail,
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
                retry_mode="single_probe" if fast_fail else "document_id_then_uri",
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
        read_client: Any,
        *,
        document_id: str | None,
        fallback_uri: str,
        fast_fail: bool = False,
    ) -> Mapping[str, object]:
        """Read one segment by exact document id, then by the canonical segment URI."""

        try:
            return read_client.fetch_full_document(
                document_id=document_id,
                origin_uri=None if document_id else fallback_uri,
                include_content=True,
                max_content_chars=self._max_content_chars(),
            )
        except Exception as exc:
            if not document_id:
                raise
            if fast_fail and self._status_code_from_exception(exc) != 404:
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
        include_selection_projection: bool = True,
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
                        metadata=self._catalog_entry_metadata_from_mapping(
                            raw_entry,
                            include_selection_projection=include_selection_projection,
                        ),
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
        attest_readback: bool = True,
        skip_async_document_id_wait: bool = False,
    ) -> tuple[dict[str, object], ...]:
        segment_batches = self._split_catalog_segment_entries(
            definition=definition,
            catalog_entries=catalog_entries,
        )
        if not segment_batches:
            return ()
        record_items: list[ChonkyDBRecordItem] = []
        segment_tokens: list[str] = []
        for segment_index, segment_entries in enumerate(segment_batches):
            record_item, segment_token = self._build_catalog_segment_record_item(
                definition=definition,
                segment_index=segment_index,
                segment_entries=segment_entries,
            )
            record_items.append(record_item)
            segment_tokens.append(segment_token)
        document_ids = self._store_record_items(
            write_client,
            snapshot_kind=definition.snapshot_kind,
            record_items=record_items,
            attest_readback=attest_readback,
            skip_async_document_id_wait=skip_async_document_id_wait,
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
                        segment_token=segment_tokens[segment_index] if segment_index < len(segment_tokens) else None,
                    ),
                    "entry_count": len(segment_entries),
                }
            )
        return tuple(refs)

    def _split_catalog_segment_entries(
        self,
        *,
        definition: _RemoteCollectionDefinition,
        catalog_entries: list[dict[str, object]],
    ) -> tuple[tuple[dict[str, object], ...], ...]:
        if not catalog_entries:
            return ()
        max_segment_bytes = self._catalog_segment_request_budget_bytes(definition=definition)
        segments: list[tuple[dict[str, object], ...]] = []
        current_segment: list[dict[str, object]] = []
        for entry in catalog_entries:
            candidate_segment = [*current_segment, dict(entry)]
            candidate_bytes = self._catalog_segment_request_bytes(
                definition=definition,
                segment_index=len(segments),
                segment_entries=candidate_segment,
            )
            if current_segment and candidate_bytes > max_segment_bytes:
                segments.append(tuple(current_segment))
                current_segment = []
            current_segment.append(dict(entry))
        if current_segment:
            segments.append(tuple(current_segment))
        return tuple(segments)

    def _catalog_segment_request_budget_bytes(
        self,
        *,
        definition: _RemoteCollectionDefinition,
    ) -> int:
        """Return the per-segment request budget for one catalog definition.

        Projection-only graph catalogs still carry large
        `selection_projection` payloads inside one exact-read control-plane
        record. Live ChonkyDB runs continue to return `429 queue_saturated`
        for graph projection segment writes even in the mid-20s KB range
        even after removing redundant `content` duplication, and the one-item
        sync rescue path can then hit backend-busy or transport-timeout
        failures. Keep graph projection segments under a lower fixed ceiling
        so the normal async path stays below that proven saturation band
        instead of oscillating around it.
        """

        budget = self._catalog_segment_max_bytes()
        if definition.snapshot_kind in _PROJECTION_ONLY_CATALOG_SNAPSHOT_KINDS:
            return max(1, min(budget // 2, _GRAPH_PROJECTION_SEGMENT_REQUEST_MAX_BYTES))
        return budget

    def _build_catalog_segment_record_item(
        self,
        *,
        definition: _RemoteCollectionDefinition,
        segment_index: int,
        segment_entries: Iterable[dict[str, object]],
    ) -> tuple[ChonkyDBRecordItem, str]:
        """Build one immutable segment record plus its versioned token."""

        normalized_entries = tuple(dict(entry) for entry in segment_entries)
        segment_payload = {
            "schema": definition.segment_schema,
            "version": _SEGMENT_VERSION,
            "snapshot_kind": definition.snapshot_kind,
            "segment_index": segment_index,
            "items": list(normalized_entries),
        }
        segment_token = self._payload_sha256(segment_payload)[:24]
        content_text = None
        if definition.snapshot_kind not in _PROJECTION_ONLY_CATALOG_SNAPSHOT_KINDS:
            content_text = json.dumps(segment_payload, ensure_ascii=False)
        return (
            ChonkyDBRecordItem(
                payload=segment_payload,
                metadata={
                    "twinr_snapshot_kind": definition.snapshot_kind,
                    "twinr_catalog_segment_index": segment_index,
                    "twinr_catalog_segment_items": len(normalized_entries),
                    "twinr_catalog_segment_token": segment_token,
                },
                # Projection-only graph segment docs are exact-read control-plane
                # records. Their authoritative data already lives in `payload`,
                # so mirroring the same JSON again in `content` only bloats the
                # async request size and re-triggers queue saturation without
                # improving readers or attestation.
                content=content_text,
                uri=self._catalog_segment_uri(
                    snapshot_kind=definition.snapshot_kind,
                    segment_index=segment_index,
                    segment_token=segment_token,
                ),
                target_indexes=self._control_plane_write_target_indexes(),
                enable_chunking=False,
                include_insights_in_response=False,
            ),
            segment_token,
        )

    def _catalog_segment_request_bytes(
        self,
        *,
        definition: _RemoteCollectionDefinition,
        segment_index: int,
        segment_entries: Iterable[dict[str, object]],
    ) -> int:
        """Measure one single-segment bulk request by the real serialized payload.

        Segment entries can carry large `selection_projection` metadata. The
        actual ChonkyDB request can duplicate the segment JSON in both
        `payload` and `content`, and graph projection segments still carry the
        full metadata-rich payload even after dropping that redundant content
        copy. Size the segment against the real serialized one-item bulk
        request instead of the inner `items` array.
        """

        record_item, _segment_token = self._build_catalog_segment_record_item(
            definition=definition,
            segment_index=segment_index,
            segment_entries=segment_entries,
        )
        request = ChonkyDBBulkRecordRequest(
            items=(record_item,),
            execution_mode="async",
            timeout_seconds=self._remote_async_job_timeout_s(snapshot_kind=definition.snapshot_kind),
            client_request_id="ltw-segment-size-probe",
            finalize_vector_segments=False,
        )
        return self._bulk_request_bytes(request)

    def _store_record_items(
        self,
        write_client: object,
        *,
        snapshot_kind: str,
        record_items: list[ChonkyDBRecordItem],
        attest_readback: bool = True,
        skip_async_document_id_wait: bool = False,
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
                    attest_readback=attest_readback,
                    skip_async_document_id_wait=skip_async_document_id_wait,
                )
            )
        return tuple(document_ids)

    def _store_record_batch_with_retries(
        self,
        write_client: Any,
        *,
        snapshot_kind: str,
        batch: tuple[ChonkyDBRecordItem, ...],
        batch_index: int,
        batch_count: int,
        finalize_vector_segments: bool,
        attest_readback: bool,
        skip_async_document_id_wait: bool,
        forced_execution_mode: str | None = None,
        allow_single_item_sync_rescue: bool = True,
    ) -> tuple[str | None, ...]:
        retry_attempts = max(1, self._remote_retry_attempts())
        retry_backoff_s = self._remote_retry_backoff_s()
        attempt_index = 0
        while attempt_index < retry_attempts:
            raise_if_remote_operation_cancelled(operation="Remote catalog write")
            request_correlation_id = self._new_remote_write_correlation_id()
            async_job_timeout_s = self._remote_async_job_timeout_s(snapshot_kind=snapshot_kind)
            request_payload_kind = self._record_batch_payload_kind(batch=batch)
            access_classification = self._record_batch_access_classification(batch=batch)
            if self._needs_extended_async_current_head_retry_budget(
                snapshot_kind=snapshot_kind,
                batch=batch,
                request_payload_kind=request_payload_kind,
            ):
                retry_attempts = max(retry_attempts, _EXTENDED_ASYNC_CURRENT_HEAD_RETRY_ATTEMPTS)
            if self._needs_extended_async_post_sync_rescue_retry_budget(
                snapshot_kind=snapshot_kind,
                batch=batch,
                request_payload_kind=request_payload_kind,
                forced_execution_mode=forced_execution_mode,
                allow_single_item_sync_rescue=allow_single_item_sync_rescue,
            ):
                retry_attempts = max(
                    retry_attempts,
                    _EXTENDED_ASYNC_POST_SYNC_RESCUE_RETRY_ATTEMPTS,
                )
            # Projection-complete callers may defer fine-grained item ids because
            # the catalog metadata already carries the bounded follow-up proof.
            # Immutable control-plane docs still need the jobs endpoint when it
            # is available; otherwise repeated namespaces/retries can attach
            # readback to stale same-URI history instead of the accepted record.
            projection_complete_document_ids_optional = (
                skip_async_document_id_wait
                and request_payload_kind == "fine_grained_record_batch"
                and snapshot_kind in _PROJECTION_COMPLETE_FINE_GRAINED_SNAPSHOT_KINDS
            )
            execution_mode = forced_execution_mode or self._bulk_execution_mode(
                snapshot_kind=snapshot_kind,
                batch=batch,
                request_payload_kind=request_payload_kind,
                skip_async_document_id_wait=skip_async_document_id_wait,
            )
            write_transport_timeout_s = self._write_transport_timeout_s(
                execution_mode=execution_mode,
                snapshot_kind=snapshot_kind,
                request_payload_kind=request_payload_kind,
                forced_execution_mode=forced_execution_mode,
                skip_async_document_id_wait=skip_async_document_id_wait,
                async_job_timeout_s=async_job_timeout_s,
            )
            effective_write_client = self._client_with_timeout(
                write_client,
                timeout_s=write_transport_timeout_s,
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
            store_transport_ms: float | None = None
            async_job_wait_ms: float | None = None
            readback_attestation_ms: float | None = None
            async_job_resolution_source: str | None = None
            attestation_mode: str | None = None
            readback_required: bool | None = None
            try:
                result = getattr(effective_write_client, "store_records_bulk")(request)
                store_transport_ms = max(0.0, (time.monotonic() - started_monotonic) * 1000.0)
                failure_detail = self._store_result_failure_detail(result)
                if failure_detail:
                    raise ChonkyDBError(
                        f"ChonkyDB rejected remote catalog write: {failure_detail}",
                        response_json=dict(result) if isinstance(result, Mapping) else None,
                    )
                extracted_document_ids = self._extract_document_ids(result, expected=len(batch))
                fine_grained_readback_optional = self._can_defer_fine_grained_readback_attestation(
                    snapshot_kind=snapshot_kind,
                    batch=batch,
                    document_ids=extracted_document_ids,
                    skip_async_document_id_wait=skip_async_document_id_wait,
                )
                fine_grained_job_wait_optional = (
                    fine_grained_readback_optional
                    and snapshot_kind in _PROJECTION_COMPLETE_FINE_GRAINED_SNAPSHOT_KINDS
                )
                async_job_resolution_budget_s = self._async_job_visibility_timeout_s(snapshot_kind=snapshot_kind)
                async_job_poll_interval_s = max(self._remote_retry_backoff_s(), _DEFAULT_ASYNC_ATTESTATION_POLL_S)
                async_job_resolution_available = bool(
                    self._normalize_text(result.get("job_id")) if isinstance(result, Mapping) else None
                ) and callable(getattr(write_client, "job_status", None))
                projection_complete_catalog_attestation_optional = (
                    (
                        skip_async_document_id_wait
                        and request_payload_kind == "catalog_current_head_record_batch"
                        and snapshot_kind in _SYNC_DEFERRED_ID_CURRENT_HEAD_SNAPSHOT_KINDS
                    )
                    or (
                        projection_complete_document_ids_optional
                        and (
                            not async_job_resolution_available
                            or (
                                fine_grained_job_wait_optional
                                and async_job_resolution_budget_s < async_job_poll_interval_s
                            )
                        )
                    )
                )
                wait_for_async_job_completion = self._should_wait_for_async_job_completion(
                    snapshot_kind=snapshot_kind,
                    batch=batch,
                )
                skip_projection_complete_job_wait = (
                    projection_complete_document_ids_optional and not wait_for_async_job_completion
                )
                skip_projection_complete_budget_job_wait = (
                    fine_grained_job_wait_optional
                    and async_job_resolution_budget_s < async_job_poll_interval_s
                    and not wait_for_async_job_completion
                )
                skip_async_job_document_id_wait = (
                    not wait_for_async_job_completion
                    and self._can_skip_async_job_document_id_wait(
                        snapshot_kind=snapshot_kind,
                        batch=batch,
                        document_ids=extracted_document_ids,
                        skip_async_document_id_wait=skip_async_document_id_wait,
                    )
                )
                if all(isinstance(value, str) and value for value in extracted_document_ids):
                    async_job_resolution_source = "response"
                if (
                    not attest_readback
                    or skip_projection_complete_job_wait
                    or skip_async_job_document_id_wait
                    or (wait_for_async_job_completion and not async_job_resolution_available)
                    or skip_projection_complete_budget_job_wait
                    or not async_job_resolution_available
                ):
                    job_document_ids = None
                    if async_job_resolution_source is None:
                        if not attest_readback:
                            async_job_resolution_source = "skipped_attestation_disabled"
                        elif skip_projection_complete_job_wait:
                            async_job_resolution_source = "skipped_projection_complete"
                        elif skip_async_job_document_id_wait and snapshot_kind == "midterm":
                            async_job_resolution_source = "skipped_projection_complete"
                        elif not async_job_resolution_available:
                            async_job_resolution_source = "unavailable"
                        elif skip_projection_complete_budget_job_wait:
                            async_job_resolution_source = "skipped_projection_complete"
                        else:
                            async_job_resolution_source = "response"
                else:
                    job_wait_started = time.monotonic()
                    try:
                        job_document_ids = self._await_async_job_document_ids(
                            write_client,
                            result=result,
                            expected=len(batch),
                            snapshot_kind=snapshot_kind,
                            request_payload_kind=request_payload_kind,
                            require_job_completion=wait_for_async_job_completion,
                        )
                    finally:
                        async_job_wait_ms = max(0.0, (time.monotonic() - job_wait_started) * 1000.0)
                if job_document_ids is not None:
                    extracted_document_ids = job_document_ids
                    async_job_resolution_source = "job_status"
                if not attest_readback:
                    attestation_mode = "disabled"
                    readback_required = False
                    resolved_document_ids = extracted_document_ids
                elif all(isinstance(value, str) and value for value in extracted_document_ids):
                    if self._exact_document_ids_require_readback(
                        snapshot_kind=snapshot_kind,
                        batch=batch,
                        request_payload_kind=request_payload_kind,
                    ):
                        attestation_mode = "exact_document_ids_readback"
                        readback_required = True
                        readback_started = time.monotonic()
                        try:
                            resolved_document_ids = self._attest_record_batch_readback(
                                snapshot_kind=snapshot_kind,
                                record_items=batch,
                                document_ids=extracted_document_ids,
                            )
                        finally:
                            readback_attestation_ms = max(0.0, (time.monotonic() - readback_started) * 1000.0)
                    else:
                        attestation_mode = "exact_document_ids"
                        readback_required = False
                        resolved_document_ids = extracted_document_ids
                elif projection_complete_catalog_attestation_optional or fine_grained_readback_optional:
                    # Projection-complete writers serve fresh readers from the
                    # later segment/current-head contract, not from the
                    # transient visibility of each accepted item document. When
                    # async writes finish without stable document ids, live
                    # ChonkyDB can still 404 on per-item origin reads even
                    # though the subsequent catalog projection is the
                    # authoritative bounded proof surface. In that shape, defer
                    # attestation to the catalog projection instead of burning
                    # the write on item-document readback.
                    if async_job_resolution_source is None:
                        async_job_resolution_source = "skipped_projection_complete"
                    attestation_mode = "deferred_catalog_projection"
                    readback_required = False
                    resolved_document_ids = extracted_document_ids
                else:
                    attestation_mode = "readback"
                    readback_required = True
                    readback_started = time.monotonic()
                    try:
                        resolved_document_ids = self._attest_record_batch_readback(
                            snapshot_kind=snapshot_kind,
                            record_items=batch,
                            document_ids=extracted_document_ids,
                        )
                    finally:
                        readback_attestation_ms = max(0.0, (time.monotonic() - readback_started) * 1000.0)
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
                        request_execution_mode=execution_mode,
                        async_job_resolution_source=async_job_resolution_source,
                        attestation_mode=attestation_mode,
                        readback_required=readback_required,
                        store_transport_ms=store_transport_ms,
                        async_job_wait_ms=async_job_wait_ms,
                        readback_attestation_ms=readback_attestation_ms,
                        uri_hint=self._normalize_text(getattr(batch[0], "uri", None)) if len(batch) == 1 else None,
                    ),
                    latency_ms=max(0.0, (time.monotonic() - started_monotonic) * 1000.0),
                    outcome="ok",
                    classification="ok",
                )
                return resolved_document_ids
            except Exception as exc:
                if (
                    execution_mode == "async"
                    and allow_single_item_sync_rescue
                    and is_rate_limited_remote_write_error(exc)
                ):
                    if len(batch) > 1:
                        return self._store_rate_limited_record_subbatches(
                            write_client,
                            snapshot_kind=snapshot_kind,
                            batch=batch,
                            finalize_vector_segments=finalize_vector_segments,
                            attest_readback=attest_readback,
                            skip_async_document_id_wait=skip_async_document_id_wait,
                            allow_single_item_sync_rescue=allow_single_item_sync_rescue,
                        )
                    if self._can_rate_limited_single_item_sync_fallback(
                        snapshot_kind=snapshot_kind,
                        batch=batch,
                        request_payload_kind=request_payload_kind,
                    ):
                        return self._store_record_batch_with_retries(
                            write_client,
                            snapshot_kind=snapshot_kind,
                            batch=batch,
                            batch_index=batch_index,
                            batch_count=batch_count,
                            finalize_vector_segments=finalize_vector_segments,
                            attest_readback=attest_readback,
                            skip_async_document_id_wait=skip_async_document_id_wait,
                            forced_execution_mode="sync",
                            allow_single_item_sync_rescue=False,
                        )
                if (
                    not allow_single_item_sync_rescue
                    and forced_execution_mode == "sync"
                    and execution_mode == "sync"
                    and self._can_forced_sync_backpressure_async_fallback(
                        snapshot_kind=snapshot_kind,
                        batch=batch,
                        request_payload_kind=request_payload_kind,
                        exc=exc,
                    )
                ):
                    return self._store_record_batch_with_retries(
                        write_client,
                        snapshot_kind=snapshot_kind,
                        batch=batch,
                        batch_index=batch_index,
                        batch_count=batch_count,
                        finalize_vector_segments=finalize_vector_segments,
                        attest_readback=attest_readback,
                        skip_async_document_id_wait=skip_async_document_id_wait,
                        forced_execution_mode="async",
                        allow_single_item_sync_rescue=False,
                        )
                if (
                    not allow_single_item_sync_rescue
                    and forced_execution_mode == "sync"
                    and execution_mode == "sync"
                    and self._can_forced_sync_rescue_timeout_async_fallback(
                        snapshot_kind=snapshot_kind,
                        request_payload_kind=request_payload_kind,
                        batch=batch,
                        exc=exc,
                    )
                ):
                    return self._store_record_batch_with_retries(
                        write_client,
                        snapshot_kind=snapshot_kind,
                        batch=batch,
                        batch_index=batch_index,
                        batch_count=batch_count,
                        finalize_vector_segments=finalize_vector_segments,
                        attest_readback=attest_readback,
                        skip_async_document_id_wait=skip_async_document_id_wait,
                        forced_execution_mode="async",
                        allow_single_item_sync_rescue=False,
                    )
                if (
                    forced_execution_mode is None
                    and execution_mode == "sync"
                    and self._can_current_head_sync_backpressure_async_fallback(
                        snapshot_kind=snapshot_kind,
                        request_payload_kind=request_payload_kind,
                        batch=batch,
                        exc=exc,
                    )
                ):
                    return self._store_record_batch_with_retries(
                        write_client,
                        snapshot_kind=snapshot_kind,
                        batch=batch,
                        batch_index=batch_index,
                        batch_count=batch_count,
                        finalize_vector_segments=finalize_vector_segments,
                        attest_readback=attest_readback,
                        skip_async_document_id_wait=skip_async_document_id_wait,
                        forced_execution_mode="async",
                        allow_single_item_sync_rescue=False,
                    )
                if (
                    forced_execution_mode is None
                    and execution_mode == "sync"
                    and self._can_current_head_sync_timeout_async_fallback(
                        snapshot_kind=snapshot_kind,
                        request_payload_kind=request_payload_kind,
                        batch=batch,
                        exc=exc,
                    )
                ):
                    return self._store_record_batch_with_retries(
                        write_client,
                        snapshot_kind=snapshot_kind,
                        batch=batch,
                        batch_index=batch_index,
                        batch_count=batch_count,
                        finalize_vector_segments=finalize_vector_segments,
                        attest_readback=attest_readback,
                        skip_async_document_id_wait=skip_async_document_id_wait,
                        forced_execution_mode="async",
                        allow_single_item_sync_rescue=False,
                    )
                retry_attempts = retryable_remote_write_attempts(retry_attempts, exc=exc)
                if should_retry_remote_write_error(exc) and attempt_index + 1 < retry_attempts:
                    raise_if_remote_operation_cancelled(operation="Remote catalog write")
                    delay_s = remote_write_retry_delay_s(
                        exc,
                        default_backoff_s=retry_backoff_s,
                        attempt_index=attempt_index,
                        max_backoff_s=self._extended_async_retry_max_backoff_s(
                            snapshot_kind=snapshot_kind,
                            batch=batch,
                            request_payload_kind=request_payload_kind,
                            forced_execution_mode=forced_execution_mode,
                            allow_single_item_sync_rescue=allow_single_item_sync_rescue,
                        ),
                    )
                    if delay_s > 0.0:
                        sleep_with_remote_operation_abort(
                            delay_s,
                            operation="Remote catalog write retry",
                        )
                    attempt_index += 1
                    continue
                remote_state = self._require_remote_state()
                remote_write_context = {
                    "snapshot_kind": snapshot_kind,
                    "operation": "store_records_bulk",
                    "request_path": "/v1/external/records/bulk",
                    "request_payload_kind": request_payload_kind,
                    "request_execution_mode": execution_mode,
                    "timeout_s": async_job_timeout_s,
                    "attestation_mode": attestation_mode,
                    "readback_required": readback_required,
                    "attempt_count": attempt_index + 1,
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
                        request_execution_mode=execution_mode,
                        async_job_resolution_source=async_job_resolution_source,
                        attestation_mode=attestation_mode,
                        readback_required=readback_required,
                        store_transport_ms=store_transport_ms,
                        async_job_wait_ms=async_job_wait_ms,
                        readback_attestation_ms=readback_attestation_ms,
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
        raise LongTermRemoteUnavailableError(
            f"Failed to persist fine-grained remote long-term memory items for {snapshot_kind!r} after retry exhaustion."
        )

    def _can_rate_limited_single_item_sync_fallback(
        self,
        *,
        snapshot_kind: str,
        batch: tuple[ChonkyDBRecordItem, ...],
        request_payload_kind: str,
    ) -> bool:
        """Return whether one queue-saturated single item is safe to retry sync.

        Once an async batch already collapsed to one item, further batch
        splitting cannot reduce queue pressure any more. For a tiny payload the
        only bounded recovery path left is to bypass the async queue exactly
        once and let the backend process that record inline.

        Large segment docs can still overrun the sync transport budget, so
        retrying them inline would just convert one retryable async `429` into
        repeated sync/backend-busy failures. Tiny projection-complete graph,
        structured-memory, and midterm segments are different: once the segment
        already collapsed to a single very small record and the backend still
        says `queue_saturated`, re-submitting the same tiny job to the async
        queue has no smaller shape left to try. Those specific segments may
        bypass the queue exactly once through the sync lane. Larger graph
        segments stay on the async lane and use the normal transient backoff
        budget instead of forcing the inline path.
        Fine-grained item docs only use this sync bypass when the surrounding
        snapshot kind has a projection-complete follow-up proof. Midterm packet
        items now qualify as well because the authoritative current head
        already carries the full `selection_projection`, the sync rescue uses
        the longer flush transport budget, and a proven busy/timeout on that
        rescue still falls back to bounded async retries exactly once.
        """

        if len(batch) != 1:
            return False
        max_sync_request_bytes = _SYNC_SMALL_CONTROL_PLANE_WRITE_MAX_BYTES
        if request_payload_kind not in {
            "fine_grained_record_batch",
            "catalog_current_head_record_batch",
            "catalog_segment_record_batch",
        }:
            return False
        if request_payload_kind == "catalog_segment_record_batch":
            if snapshot_kind not in _SYNC_DEFERRED_ID_SEGMENT_SNAPSHOT_KINDS:
                return False
            preview_request = ChonkyDBBulkRecordRequest(
                items=batch,
                execution_mode="sync",
                timeout_seconds=self._remote_async_job_timeout_s(),
                client_request_id="preview",
                finalize_vector_segments=True,
            )
            return self._bulk_request_bytes(preview_request) <= _SYNC_SMALL_SEGMENT_WRITE_MAX_BYTES
        if request_payload_kind == "catalog_current_head_record_batch":
            # Only current-head writers with the explicit deferred-id contract
            # may bypass the async queue. Other same-URI heads still need the
            # normal async retry lane; forcing them into sync on one `429`
            # recreates the exact live failure where they cannot safely fall
            # back and die on the short inline transport timeout instead.
            if snapshot_kind not in _SYNC_DEFERRED_ID_CURRENT_HEAD_SNAPSHOT_KINDS:
                return False
            max_sync_request_bytes = self._current_head_sync_request_max_bytes(
                snapshot_kind=snapshot_kind,
                request_payload_kind=request_payload_kind,
            )
        if request_payload_kind == "fine_grained_record_batch":
            if snapshot_kind not in _PROJECTION_COMPLETE_FINE_GRAINED_SNAPSHOT_KINDS:
                return False
        if snapshot_kind in {"graph_nodes", "graph_edges"} and request_payload_kind != "catalog_current_head_record_batch":
            return False
        preview_request = ChonkyDBBulkRecordRequest(
            items=batch,
            execution_mode="sync",
            timeout_seconds=self._remote_async_job_timeout_s(),
            client_request_id="preview",
            finalize_vector_segments=True,
        )
        return self._bulk_request_bytes(preview_request) <= max_sync_request_bytes

    def _forced_sync_rescue_needs_flush_timeout(
        self,
        *,
        snapshot_kind: str,
        request_payload_kind: str,
        forced_execution_mode: str | None,
    ) -> bool:
        """Return whether one sync rescue write should inherit the flush budget."""

        if forced_execution_mode != "sync":
            return False
        if (
            request_payload_kind == "catalog_current_head_record_batch"
            and snapshot_kind in _SYNC_DEFERRED_ID_CURRENT_HEAD_SNAPSHOT_KINDS
        ):
            return True
        if (
            request_payload_kind == "catalog_segment_record_batch"
            and snapshot_kind in _SYNC_DEFERRED_ID_SEGMENT_SNAPSHOT_KINDS
        ):
            return True
        return (
            request_payload_kind == "fine_grained_record_batch"
            and snapshot_kind in _PROJECTION_COMPLETE_FINE_GRAINED_SNAPSHOT_KINDS
        )

    def _sync_write_needs_flush_timeout(
        self,
        *,
        execution_mode: str,
        snapshot_kind: str,
        request_payload_kind: str,
        forced_execution_mode: str | None,
        skip_async_document_id_wait: bool,
    ) -> bool:
        """Return whether one sync write needs the longer flush transport budget."""

        if execution_mode != "sync":
            return False
        if self._forced_sync_rescue_needs_flush_timeout(
            snapshot_kind=snapshot_kind,
            request_payload_kind=request_payload_kind,
            forced_execution_mode=forced_execution_mode,
        ):
            return True
        if forced_execution_mode is not None:
            return False
        if (
            request_payload_kind == "catalog_current_head_record_batch"
            and skip_async_document_id_wait
            and snapshot_kind in _SYNC_DEFERRED_ID_CURRENT_HEAD_SNAPSHOT_KINDS
        ):
            return True
        return (
            request_payload_kind == "catalog_segment_record_batch"
            and skip_async_document_id_wait
            and snapshot_kind in _SYNC_SMALL_PROJECTION_COMPLETE_SEGMENT_SNAPSHOT_KINDS
        )

    def _write_transport_timeout_s(
        self,
        *,
        execution_mode: str,
        snapshot_kind: str,
        request_payload_kind: str,
        forced_execution_mode: str | None,
        skip_async_document_id_wait: bool,
        async_job_timeout_s: float | None,
    ) -> float | None:
        """Return the client transport timeout for one bulk write attempt.

        The HTTP client timeout must never undercut the bounded server-side
        execution window that Twinr already grants to accepted async jobs for
        the same snapshot kind. Live `objects/catalog/segment` canary failures
        proved that one tiny `async -> sync` rescue can legitimately finish
        after the generic 60 s flush budget yet still well inside the
        floored 180 s async job budget. Reusing that broader budget on the
        specific direct-sync control-plane lanes keeps the rescue honest
        without broadening ordinary write timeouts.
        """

        write_transport_timeout_s = self._remote_write_timeout_s()
        if execution_mode == "async":
            return max(
                self._remote_flush_timeout_s(),
                write_transport_timeout_s or 0.0,
            )
        if not self._sync_write_needs_flush_timeout(
            execution_mode=execution_mode,
            snapshot_kind=snapshot_kind,
            request_payload_kind=request_payload_kind,
            forced_execution_mode=forced_execution_mode,
            skip_async_document_id_wait=skip_async_document_id_wait,
        ):
            return write_transport_timeout_s
        return max(
            self._remote_flush_timeout_s(),
            async_job_timeout_s or 0.0,
            write_transport_timeout_s or 0.0,
        )

    def _can_forced_sync_backpressure_async_fallback(
        self,
        *,
        snapshot_kind: str,
        batch: tuple[ChonkyDBRecordItem, ...],
        request_payload_kind: str,
        exc: BaseException,
    ) -> bool:
        """Return whether one busy sync rescue should fall back to async retries.

        A single-item batch may rescue from `async -> sync` once after
        `queue_saturated`. If the sync lane itself immediately reports
        `payload_sync_bulk_busy`, keeping the rest of the retry budget pinned to
        sync just hammers the wrong lane. In that proven shape the batch may
        return to bounded async retries exactly once, without re-enabling
        another sync rescue.
        """

        if len(batch) != 1:
            return False
        if not is_rate_limited_remote_write_error(exc):
            return False
        if (
            request_payload_kind == "catalog_segment_record_batch"
            and snapshot_kind in _SYNC_DEFERRED_ID_SEGMENT_SNAPSHOT_KINDS
        ):
            return True
        if (
            request_payload_kind == "catalog_current_head_record_batch"
            and snapshot_kind in _SYNC_DEFERRED_ID_CURRENT_HEAD_SNAPSHOT_KINDS
        ):
            return True
        return (
            request_payload_kind == "fine_grained_record_batch"
            and snapshot_kind in _PROJECTION_COMPLETE_FINE_GRAINED_SNAPSHOT_KINDS
        )

    def _can_forced_sync_rescue_timeout_async_fallback(
        self,
        *,
        snapshot_kind: str,
        request_payload_kind: str,
        batch: tuple[ChonkyDBRecordItem, ...],
        exc: BaseException,
    ) -> bool:
        """Return whether one timed-out forced sync rescue may retry async once.

        Single-item writes may rescue from `async -> sync` once after live
        `queue_saturated` backpressure. ChonkyDB can still let that inline
        request sit behind earlier accepted work long enough to overrun the
        sync transport timeout without surfacing an explicit busy status. In
        that proven timeout shape the write should return to bounded async
        retries exactly once instead of burning the rest of its retry budget on
        a congested sync lane.
        """

        if len(batch) != 1:
            return False
        if request_payload_kind == "catalog_current_head_record_batch":
            if snapshot_kind not in _SYNC_DEFERRED_ID_CURRENT_HEAD_SNAPSHOT_KINDS:
                return False
        elif request_payload_kind == "fine_grained_record_batch":
            if snapshot_kind not in _PROJECTION_COMPLETE_FINE_GRAINED_SNAPSHOT_KINDS:
                return False
        elif request_payload_kind == "catalog_segment_record_batch":
            if snapshot_kind not in _SYNC_DEFERRED_ID_SEGMENT_SNAPSHOT_KINDS:
                return False
        else:
            return False
        for item in exception_chain(exc):
            if isinstance(item, ChonkyDBError):
                if item.status_code is not None:
                    continue
                message = " ".join(str(item).lower().split())
                if "timed out" in message or "timeout" in message:
                    return True
                continue
            if isinstance(item, (TimeoutError, OSError)):
                return True
        return False

    def _can_current_head_sync_timeout_async_fallback(
        self,
        *,
        snapshot_kind: str,
        request_payload_kind: str,
        batch: tuple[ChonkyDBRecordItem, ...],
        exc: BaseException,
    ) -> bool:
        """Return whether one tiny current-head sync timeout may retry async."""

        if snapshot_kind not in _SYNC_DEFAULT_CURRENT_HEAD_SNAPSHOT_KINDS:
            return False
        if request_payload_kind != "catalog_current_head_record_batch":
            return False
        return self._can_forced_sync_rescue_timeout_async_fallback(
            snapshot_kind=snapshot_kind,
            request_payload_kind=request_payload_kind,
            batch=batch,
            exc=exc,
        )

    def _can_current_head_sync_backpressure_async_fallback(
        self,
        *,
        snapshot_kind: str,
        request_payload_kind: str,
        batch: tuple[ChonkyDBRecordItem, ...],
        exc: BaseException,
    ) -> bool:
        """Return whether one default sync current-head busy response may retry async.

        Small structured current-head writes still prefer the sync lane when the
        backend is healthy because the payload is tiny and the projection
        contract already bounds readback. If that inline lane itself reports
        temporary backpressure, keeping the rest of the retry budget pinned to
        sync only hammers the congested path. In that proven shape, fall back
        once to async and keep the sync rescue disabled so the write cannot
        ping-pong back to the same busy inline lane.
        """

        if snapshot_kind not in _SYNC_DEFERRED_ID_CURRENT_HEAD_SNAPSHOT_KINDS:
            return False
        if request_payload_kind != "catalog_current_head_record_batch":
            return False
        if snapshot_kind not in _SYNC_DEFAULT_CURRENT_HEAD_SNAPSHOT_KINDS:
            return False
        if len(batch) != 1:
            return False
        return is_rate_limited_remote_write_error(exc)

    def _store_rate_limited_record_subbatches(
        self,
        write_client: object,
        *,
        snapshot_kind: str,
        batch: tuple[ChonkyDBRecordItem, ...],
        finalize_vector_segments: bool,
        attest_readback: bool,
        skip_async_document_id_wait: bool,
        allow_single_item_sync_rescue: bool,
    ) -> tuple[str | None, ...]:
        """Retry one rejected async batch by splitting it into smaller pieces.

        Live ChonkyDB can reject multi-item async writes with `429 queue_saturated`
        even when the individual items succeed immediately once the queue burden
        per job drops. Splitting only the already rejected batch keeps the normal
        high-throughput path unchanged while reacting to proven backend
        backpressure with a smaller request shape.
        """

        if len(batch) <= 1:
            raise LongTermRemoteUnavailableError(
                "Cannot split a single-item async batch after a queue_saturated rejection."
            )
        midpoint = max(1, len(batch) // 2)
        subbatches = tuple(part for part in (batch[:midpoint], batch[midpoint:]) if part)
        document_ids: list[str | None] = []
        subbatch_count = len(subbatches)
        for index, subbatch in enumerate(subbatches, start=1):
            document_ids.extend(
                self._store_record_batch_with_retries(
                    write_client,
                    snapshot_kind=snapshot_kind,
                    batch=subbatch,
                    batch_index=index,
                    batch_count=subbatch_count,
                    finalize_vector_segments=finalize_vector_segments and index >= subbatch_count,
                    attest_readback=attest_readback,
                    skip_async_document_id_wait=skip_async_document_id_wait,
                    allow_single_item_sync_rescue=allow_single_item_sync_rescue,
                )
            )
        return tuple(document_ids)

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

    def _current_head_sync_request_max_bytes(
        self,
        *,
        snapshot_kind: str,
        request_payload_kind: str,
    ) -> int:
        """Return the bounded sync ceiling for one rescued current-head write.

        Projection-only graph heads legitimately carry many segment refs plus
        topology refs in the authoritative `catalog/current` record. When an
        async head write already proved `429 queue_saturated`, those larger
        graph heads may still need one bounded sync rescue around 121 KB.
        """

        if (
            request_payload_kind == "catalog_current_head_record_batch"
            and snapshot_kind in _PROJECTION_ONLY_CATALOG_SNAPSHOT_KINDS
        ):
            return _SYNC_LARGE_GRAPH_CURRENT_HEAD_WRITE_MAX_BYTES
        return _SYNC_SMALL_CONTROL_PLANE_WRITE_MAX_BYTES

    def _segment_sync_request_max_bytes(
        self,
        *,
        snapshot_kind: str,
        request_payload_kind: str,
        skip_async_document_id_wait: bool,
    ) -> int:
        """Return the bounded sync ceiling for one direct catalog-segment write.

        Direct sync segment writes stay reserved for explicit graph rescue
        paths. Structured `objects/conflicts/archive` catalog segments remain
        async-first; only the matching current-head control-plane write may use
        sync when the deferred-id contract allows it.
        """

        if (
            request_payload_kind == "catalog_segment_record_batch"
            and skip_async_document_id_wait
            and snapshot_kind in _SYNC_SMALL_PROJECTION_COMPLETE_SEGMENT_SNAPSHOT_KINDS
        ):
            return _SYNC_SMALL_SEGMENT_WRITE_MAX_BYTES
        return _SYNC_SMALL_CONTROL_PLANE_WRITE_MAX_BYTES

    def _needs_extended_async_current_head_retry_budget(
        self,
        *,
        snapshot_kind: str,
        batch: tuple[ChonkyDBRecordItem, ...],
        request_payload_kind: str,
    ) -> bool:
        """Return whether one async current-head write needs a longer 429 budget.

        Live ChonkyDB can keep its async queue saturated for longer than the
        generic transient-write budget while accepted cold-path jobs drain. The
        non-deferred same-URI current-head writers cannot safely use the sync
        rescue lane, so they need a larger async-only retry window instead of
        failing before the queue has any realistic chance to clear.
        """

        return (
            len(batch) == 1
            and request_payload_kind == "catalog_current_head_record_batch"
            and snapshot_kind not in _SYNC_DEFERRED_ID_CURRENT_HEAD_SNAPSHOT_KINDS
        )

    def _current_head_async_retry_max_backoff_s(
        self,
        *,
        snapshot_kind: str,
        batch: tuple[ChonkyDBRecordItem, ...],
        request_payload_kind: str,
    ) -> float | None:
        """Return one optional higher backoff cap for async-only current heads."""

        if not self._needs_extended_async_current_head_retry_budget(
            snapshot_kind=snapshot_kind,
            batch=batch,
            request_payload_kind=request_payload_kind,
        ):
            return None
        return _EXTENDED_ASYNC_CURRENT_HEAD_MAX_BACKOFF_S

    def _needs_extended_async_post_sync_rescue_retry_budget(
        self,
        *,
        snapshot_kind: str,
        batch: tuple[ChonkyDBRecordItem, ...],
        request_payload_kind: str,
        forced_execution_mode: str | None,
        allow_single_item_sync_rescue: bool,
    ) -> bool:
        """Return whether one post-sync-rescue async item needs a longer retry tail."""

        return (
            forced_execution_mode == "async"
            and not allow_single_item_sync_rescue
            and len(batch) == 1
            and (
                (
                    request_payload_kind == "fine_grained_record_batch"
                    and snapshot_kind in _PROJECTION_COMPLETE_FINE_GRAINED_SNAPSHOT_KINDS
                )
                or (
                    request_payload_kind == "catalog_current_head_record_batch"
                    and snapshot_kind in _SYNC_DEFERRED_ID_CURRENT_HEAD_SNAPSHOT_KINDS
                )
                or (
                    request_payload_kind == "catalog_segment_record_batch"
                    and snapshot_kind in _SYNC_DEFERRED_ID_SEGMENT_SNAPSHOT_KINDS
                )
            )
        )

    def _extended_async_retry_max_backoff_s(
        self,
        *,
        snapshot_kind: str,
        batch: tuple[ChonkyDBRecordItem, ...],
        request_payload_kind: str,
        forced_execution_mode: str | None,
        allow_single_item_sync_rescue: bool,
    ) -> float | None:
        """Return one optional higher backoff cap for proven longer async tails."""

        current_head_backoff_s = self._current_head_async_retry_max_backoff_s(
            snapshot_kind=snapshot_kind,
            batch=batch,
            request_payload_kind=request_payload_kind,
        )
        if current_head_backoff_s is not None:
            return current_head_backoff_s
        if self._needs_extended_async_post_sync_rescue_retry_budget(
            snapshot_kind=snapshot_kind,
            batch=batch,
            request_payload_kind=request_payload_kind,
            forced_execution_mode=forced_execution_mode,
            allow_single_item_sync_rescue=allow_single_item_sync_rescue,
        ):
            return _EXTENDED_ASYNC_POST_SYNC_RESCUE_MAX_BACKOFF_S
        return None

    def _can_defer_fine_grained_readback_attestation(
        self,
        *,
        snapshot_kind: str,
        batch: tuple[ChonkyDBRecordItem, ...],
        document_ids: tuple[str | None, ...],
        skip_async_document_id_wait: bool,
    ) -> bool:
        """Return whether one accepted batch may skip per-item readback attestation.

        Live ChonkyDB can finish async writes without exposing stable payload ids.
        In that shape the fallback `documents/full?origin_uri=...` path is not an
        indexed exact lookup; it can 404 on large stores even though the later
        catalog/segment/current-head writes that Twinr actually serves from are
        valid. Objects/conflicts/archive keep the bounded runtime projection on
        catalog entries, so their authoritative proof is the catalog write path,
        not a URI-only item-document reread. Midterm packet writes with
        ``skip_async_document_id_wait=True`` use the same projection-complete
        proof surface: the later current-head payload already carries the full
        ``selection_projection`` that fresh readers hydrate from, so waiting for
        per-item origin visibility only burns async-job budget without proving a
        stronger contract.
        """

        if self._record_batch_payload_kind(batch=batch) != "fine_grained_record_batch":
            return False
        if all(isinstance(value, str) and value for value in document_ids):
            return False
        if snapshot_kind in _PROJECTION_COMPLETE_FINE_GRAINED_SNAPSHOT_KINDS:
            return True
        return snapshot_kind == "midterm" and skip_async_document_id_wait

    def _exact_document_ids_require_readback(
        self,
        *,
        snapshot_kind: str,
        batch: tuple[ChonkyDBRecordItem, ...],
        request_payload_kind: str,
    ) -> bool:
        """Return whether exact ids still need document-body visibility proof.

        Immutable `catalog/segment` records are immediately referenced by the
        subsequent `catalog/current` publish. A jobs-endpoint `document_id`
        alone is therefore not enough: fresh readers must be able to hydrate
        the segment body itself before Twinr advertises that ref as current.
        """

        del snapshot_kind
        del batch
        return request_payload_kind == "catalog_segment_record_batch"

    def _can_skip_async_job_document_id_wait(
        self,
        *,
        snapshot_kind: str,
        batch: tuple[ChonkyDBRecordItem, ...],
        document_ids: tuple[str | None, ...],
        skip_async_document_id_wait: bool,
    ) -> bool:
        """Return whether one accepted batch may skip async job-status doc-id polling.

        Exact document ids from ``job_status(...)`` are useful even for mutable
        current heads: they let same-URI attestation distinguish the newly
        accepted document from older history without trusting the exact-id read
        alone as authoritative visibility. Graph segment/control-plane writes
        already use projection-complete follow-up proofs, and midterm writes
        may do the same when the caller explicitly opted into deferred item-id
        resolution for the projection-complete current head. Midterm item,
        segment, and current-head payloads all carry enough authoritative state
        that the bounded readback path can prove visibility directly from the
        accepted payload, without burning the async jobs budget on exact-id
        polling first.
        """

        if snapshot_kind in {"graph_nodes", "graph_edges"}:
            return True
        if snapshot_kind == "midterm" and skip_async_document_id_wait:
            return True
        return all(isinstance(value, str) and value for value in document_ids)

    def _should_wait_for_async_job_completion(
        self,
        *,
        snapshot_kind: str,
        batch: tuple[ChonkyDBRecordItem, ...],
    ) -> bool:
        """Return whether one accepted async batch should wait for job completion.

        Live projection-complete segment publishes can enqueue several accepted
        versioned `catalog/segment` jobs back-to-back. ChonkyDB may expose the
        first tokenized segment URI before the async queue has drained enough
        for the next equally shaped batch, so same-URI readback alone is not a
        strong enough backpressure proof there. Deferred-id segment writers
        therefore keep the jobs endpoint as the authoritative queue-drain
        contract before Twinr submits the next segment write.

        Large structured snapshot publishes (`objects/conflicts/archive`) can
        self-induce the same queue flood: `write_snapshot(...)` emits many
        fine-grained async batches in sequence, and the async jobs lane can
        still be busy with the accepted earlier batches when the next batch
        arrives. Those structured fine-grained batches therefore wait for job
        completion as well, even when batching has already collapsed to
        single-item requests under the request-byte ceiling.
        """

        request_payload_kind = self._record_batch_payload_kind(batch=batch)
        if (
            snapshot_kind in _ASYNC_JOB_BACKPRESSURE_SEGMENT_SNAPSHOT_KINDS
            and request_payload_kind == "catalog_segment_record_batch"
        ):
            return True
        return (
            snapshot_kind in _ASYNC_JOB_BACKPRESSURE_FINE_GRAINED_SNAPSHOT_KINDS
            and request_payload_kind == "fine_grained_record_batch"
        )

    def _bulk_execution_mode(
        self,
        *,
        snapshot_kind: str,
        batch: tuple[ChonkyDBRecordItem, ...],
        request_payload_kind: str,
        skip_async_document_id_wait: bool,
    ) -> str:
        """Choose async by default and reserve sync for proven rescue paths.

        After the control-plane empty-target-index fix on ChonkyDB, live
        messy-corpus runs still showed default inline `graph_*/catalog/current`
        writes blocking for 30-44 seconds even without the earlier
        vector/fulltext regression. Keep only those graph current heads
        async-first. Small structured `objects/conflicts/archive/midterm`
        current heads stay on the existing bounded sync fast path, while graph
        segment sync remains a narrow rescue-only exception.
        """

        if len(batch) != 1:
            return "async"
        if request_payload_kind not in {
            "fine_grained_record_batch",
            "catalog_current_head_record_batch",
            "catalog_segment_record_batch",
        }:
            return "async"
        sync_enabled = snapshot_kind in _SYNC_SMALL_CONTROL_PLANE_SNAPSHOT_KINDS
        max_sync_request_bytes = _SYNC_SMALL_CONTROL_PLANE_WRITE_MAX_BYTES
        segment_sync_request_max_bytes = self._segment_sync_request_max_bytes(
            snapshot_kind=snapshot_kind,
            request_payload_kind=request_payload_kind,
            skip_async_document_id_wait=skip_async_document_id_wait,
        )
        if segment_sync_request_max_bytes > _SYNC_SMALL_CONTROL_PLANE_WRITE_MAX_BYTES:
            sync_enabled = True
            max_sync_request_bytes = segment_sync_request_max_bytes
        if (
            skip_async_document_id_wait
            and request_payload_kind == "catalog_current_head_record_batch"
            and snapshot_kind in _SYNC_DEFAULT_CURRENT_HEAD_SNAPSHOT_KINDS
        ):
            sync_enabled = True
            max_sync_request_bytes = self._current_head_sync_request_max_bytes(
                snapshot_kind=snapshot_kind,
                request_payload_kind=request_payload_kind,
            )
        if not sync_enabled:
            return "async"
        preview_request = ChonkyDBBulkRecordRequest(
            items=batch,
            execution_mode="sync",
            timeout_seconds=self._remote_async_job_timeout_s(),
            client_request_id="preview",
            finalize_vector_segments=True,
        )
        if self._bulk_request_bytes(preview_request) > max_sync_request_bytes:
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
        snapshot_kind: str,
        request_payload_kind: str,
        require_job_completion: bool = False,
    ) -> tuple[str | None, ...] | None:
        if not isinstance(result, Mapping):
            return None
        initial_document_ids = self._extract_document_ids(result, expected=expected)
        if all(isinstance(value, str) and value for value in initial_document_ids) and not require_job_completion:
            return initial_document_ids
        job_id = self._normalize_text(result.get("job_id"))
        job_status = getattr(client, "job_status", None)
        if not job_id or not callable(job_status):
            return initial_document_ids if all(isinstance(value, str) and value for value in initial_document_ids) else None
        poll_interval_s = max(self._remote_retry_backoff_s(), _DEFAULT_ASYNC_ATTESTATION_POLL_S)
        total_timeout_s = self._async_job_completion_wait_timeout_s(
            snapshot_kind=snapshot_kind,
            request_payload_kind=request_payload_kind,
            require_job_completion=require_job_completion,
        )
        deadline = time.monotonic() + total_timeout_s
        while True:
            raise_if_remote_operation_cancelled(operation="Remote catalog async-job wait")
            remaining_timeout_s = deadline - time.monotonic()
            if remaining_timeout_s <= 0.0:
                break
            capped_client = clone_client_with_capped_timeout(
                client,
                timeout_s=self._job_status_transport_timeout_s(
                    snapshot_kind=snapshot_kind,
                    request_payload_kind=request_payload_kind,
                    require_job_completion=require_job_completion,
                    remaining_timeout_s=remaining_timeout_s,
                ),
            )
            capped_job_status = getattr(capped_client, "job_status", None)
            try:
                payload = job_status(job_id) if not callable(capped_job_status) else capped_job_status(job_id)
            except Exception as exc:
                if not should_fallback_async_job_resolution_error(exc):
                    raise
                if self._can_fallback_job_status_error_to_uri_attestation(
                    snapshot_kind=snapshot_kind,
                    request_payload_kind=request_payload_kind,
                    require_job_completion=require_job_completion,
                ):
                    return None
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
                    if all(isinstance(value, str) and value for value in document_ids) and not require_job_completion:
                        return document_ids
                    if self._can_fallback_pending_job_status_to_uri_attestation(
                        snapshot_kind=snapshot_kind,
                        request_payload_kind=request_payload_kind,
                        require_job_completion=require_job_completion,
                        status=status,
                        document_ids=document_ids,
                    ):
                        return None
                    if status in {"succeeded", "done"}:
                        if all(isinstance(value, str) and value for value in document_ids):
                            return document_ids
                        if all(isinstance(value, str) and value for value in initial_document_ids):
                            return initial_document_ids
                        return None
            remaining_sleep_s = deadline - time.monotonic()
            if remaining_sleep_s <= 0.0:
                break
            if poll_interval_s > 0.0:
                sleep_with_remote_operation_abort(
                    min(poll_interval_s, remaining_sleep_s),
                    operation="Remote catalog async-job wait",
                )
        return None

    def _can_fallback_job_status_error_to_uri_attestation(
        self,
        *,
        snapshot_kind: str,
        request_payload_kind: str,
        require_job_completion: bool,
    ) -> bool:
        """Return whether one jobs-endpoint error may fall back to URI readback.

        Versioned catalog-segment URI visibility only proves that one accepted
        payload became readable. It does not prove that the async jobs queue
        drained enough for the next dependent segment write. Completion-
        required segment batches therefore keep polling the jobs lane instead
        of short-circuiting to URI attestation on the first transport hiccup.
        """

        if request_payload_kind == "catalog_segment_record_batch":
            if require_job_completion:
                return False
            return snapshot_kind in _SYNC_DEFERRED_ID_SEGMENT_SNAPSHOT_KINDS
        return (
            not require_job_completion
            and request_payload_kind == "catalog_current_head_record_batch"
            and snapshot_kind in _SYNC_DEFERRED_ID_CURRENT_HEAD_SNAPSHOT_KINDS
        )

    def _can_fallback_pending_job_status_to_uri_attestation(
        self,
        *,
        snapshot_kind: str,
        request_payload_kind: str,
        require_job_completion: bool,
        status: str,
        document_ids: tuple[str | None, ...],
    ) -> bool:
        """Return whether one non-terminal jobs payload may defer to URI readback.

        Versioned catalog-segment URIs are already authoritative visibility
        proofs. When the jobs endpoint keeps returning ``pending`` without
        document ids, waiting through the full async-job deadline only delays
        the stronger same-URI readback attestation that follows anyway.
        """

        if any(isinstance(value, str) and value for value in document_ids):
            return False
        if status in {"done", "succeeded", "failed", "cancelled", "rejected"}:
            return False
        return self._can_fallback_job_status_error_to_uri_attestation(
            snapshot_kind=snapshot_kind,
            request_payload_kind=request_payload_kind,
            require_job_completion=require_job_completion,
        )

    def _job_status_transport_timeout_s(
        self,
        *,
        snapshot_kind: str,
        request_payload_kind: str,
        require_job_completion: bool,
        remaining_timeout_s: float,
    ) -> float:
        """Return the per-call HTTP timeout for one jobs-endpoint poll.

        Completion-required segment writes still need bounded polling, but the
        old speculative timeout only made sense when Twinr was willing to
        abandon jobs polling immediately and trust same-URI visibility. Once
        completion is the actual backpressure surface, use the normal remote
        read timeout as the per-poll cap instead.
        """

        capped_timeout_s = max(_DEFAULT_ASYNC_ATTESTATION_POLL_S, float(remaining_timeout_s))
        config = getattr(getattr(self, "remote_state", None), "config", None)
        try:
            read_timeout_s = float(
                getattr(
                    config,
                    "long_term_memory_remote_read_timeout_s",
                    _DEFAULT_REMOTE_READ_TIMEOUT_S,
                )
            )
        except (TypeError, ValueError):
            read_timeout_s = _DEFAULT_REMOTE_READ_TIMEOUT_S
        read_timeout_cap_s = min(capped_timeout_s, max(_DEFAULT_ASYNC_ATTESTATION_POLL_S, read_timeout_s))
        if (
            require_job_completion
            and request_payload_kind == "catalog_segment_record_batch"
            and snapshot_kind in _ASYNC_JOB_BACKPRESSURE_SEGMENT_SNAPSHOT_KINDS
        ):
            return read_timeout_cap_s
        if not self._can_fallback_job_status_error_to_uri_attestation(
            snapshot_kind=snapshot_kind,
            request_payload_kind=request_payload_kind,
            require_job_completion=require_job_completion,
        ):
            return capped_timeout_s
        return read_timeout_cap_s

    def _async_job_completion_wait_timeout_s(
        self,
        *,
        snapshot_kind: str,
        request_payload_kind: str,
        require_job_completion: bool,
    ) -> float:
        """Return the total jobs-wait budget for one accepted async write.

        Completion-required writes need a real queue-drain proof before Twinr
        submits the next dependent control-plane record. Live retention-canary
        runs proved that the old 15 s cap on structured fine-grained batches
        could let `objects` item jobs keep draining in the background while
        Twinr already moved on to the next `catalog/segment` write, which then
        failed with `429 queue_saturated` or `503 payload_sync_bulk_busy`.
        Use the full bounded visibility window whenever completion itself is
        the backpressure contract, even if later item-document ids are not the
        authoritative read surface.
        """

        total_timeout_s = self._async_job_visibility_timeout_s(snapshot_kind=snapshot_kind)
        return total_timeout_s

    def _attest_record_readback(
        self,
        read_client: Any,
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
        poll_interval_s = max(self._remote_retry_backoff_s(), _DEFAULT_ASYNC_ATTESTATION_POLL_S)
        visibility_timeout_s = self._attestation_visibility_timeout_s(
            snapshot_kind=snapshot_kind,
            record_item=record_item,
        )
        require_document_id = self._attestation_requires_non_null_document_id(record_item)
        last_detail = "Remote write attestation did not observe the accepted payload."
        probe_targets = self._attestation_probe_targets(
            document_id=document_id,
            resolved_uri=resolved_uri or None,
            exact_document_id_authoritative=bool(document_id) and not self._is_mutable_catalog_record(record_item),
        )
        deadline = time.monotonic() + visibility_timeout_s
        while True:
            raise_if_remote_operation_cancelled(operation="Remote catalog readback attestation")
            remaining_timeout_s = deadline - time.monotonic()
            if remaining_timeout_s <= 0.0:
                break
            for probe_document_id, probe_uri in probe_targets:
                raise_if_remote_operation_cancelled(operation="Remote catalog readback attestation")
                remaining_timeout_s = deadline - time.monotonic()
                if remaining_timeout_s <= 0.0:
                    break
                capped_read_client: Any = clone_client_with_capped_timeout(read_client, timeout_s=remaining_timeout_s)
                try:
                    envelope = capped_read_client.fetch_full_document(
                        document_id=probe_document_id,
                        origin_uri=probe_uri,
                        include_content=True,
                        max_content_chars=self._max_content_chars(),
                    )
                except Exception as exc:
                    last_detail = str(exc)
                    continue
                matched_record = self._match_attested_record_document_id(
                    envelope,
                    expected_payloads=expected_payloads,
                )
                if matched_record is not None:
                    matched_document_id = self._normalize_text(matched_record.document_id)
                    if matched_document_id or not require_document_id:
                        return matched_document_id
                    # Unversioned segment refs still need a stable document id,
                    # but tokenized `.../catalog/segment/<index>/<token>` URIs
                    # are already immutable same-record selectors. Those may
                    # stay URI-only when ChonkyDB omits `document_id` on
                    # same-URI readback envelopes.
                    last_detail = "Remote write attestation observed the accepted payload without a stable document id."
                    continue
                last_detail = "Remote write attestation read back a different same-uri document."
            remaining_sleep_s = deadline - time.monotonic()
            if remaining_sleep_s <= 0.0:
                break
            if poll_interval_s > 0.0:
                sleep_with_remote_operation_abort(
                    min(poll_interval_s, remaining_sleep_s),
                    operation="Remote catalog readback attestation",
                )
        raise LongTermRemoteUnavailableError(
            f"Accepted remote long-term {snapshot_kind!r} write could not be read back: {last_detail}"
        )

    def _attestation_visibility_timeout_s(
        self,
        *,
        snapshot_kind: str,
        record_item: ChonkyDBRecordItem,
    ) -> float:
        """Return the visibility window for one accepted-write attestation.

        Mutable fixed catalog URIs like ``.../catalog/current`` and
        ``.../catalog/segment/<n>`` can lag on same-URI visibility longer than
        ordinary item records even after the async job is already accepted.
        When the backend has not produced a stable exact document id for those
        mutable URIs yet, retrying the whole write after the smaller generic
        attestation window just republishes the same moving catalog surface and
        can make the lag worse. Mutable catalog records therefore inherit the
        broader accepted-job visibility budget before we declare the write
        unavailable.
        """

        visibility_timeout_s = self._async_attestation_visibility_timeout_s()
        if not self._is_mutable_catalog_record(record_item):
            return visibility_timeout_s
        return max(
            visibility_timeout_s,
            self._async_job_visibility_timeout_s(snapshot_kind=snapshot_kind),
        )

    def _is_mutable_catalog_record(self, record_item: ChonkyDBRecordItem) -> bool:
        """Return whether one record item targets a fixed mutable catalog URI."""

        resolved_uri = self._normalize_text(getattr(record_item, "uri", None))
        metadata = getattr(record_item, "metadata", None)
        metadata_mapping = metadata if isinstance(metadata, Mapping) else None
        if isinstance(metadata_mapping, Mapping) and metadata_mapping.get("twinr_catalog_current_head") is True:
            return True
        if isinstance(metadata_mapping, Mapping) and metadata_mapping.get("twinr_catalog_segment_token") is not None:
            return False
        if resolved_uri.endswith("/catalog/current"):
            return True
        if "/catalog/segment/" in resolved_uri:
            suffix = resolved_uri.rsplit("/catalog/segment/", 1)[-1]
            return len([part for part in suffix.split("/") if part]) <= 1
        return bool(
            isinstance(metadata_mapping, Mapping)
            and metadata_mapping.get("twinr_catalog_segment_index") is not None
        )

    def _attestation_requires_non_null_document_id(
        self,
        record_item: ChonkyDBRecordItem,
    ) -> bool:
        """Return whether one attested record must surface a stable document id.

        Unversioned structured segment refs still require a stable
        `document_id`, but tokenized `.../catalog/segment/<index>/<token>`
        URIs are immutable selectors and remain safe to publish even when the
        backend omits `document_id` on `documents/full?origin_uri=...`.
        Graph projection segments keep their historical URI-first contract.
        """

        metadata = getattr(record_item, "metadata", None)
        if not isinstance(metadata, Mapping):
            return False
        if metadata.get("twinr_catalog_segment_token") is None:
            return False
        snapshot_kind = self._normalize_text(metadata.get("twinr_snapshot_kind"))
        resolved_uri = self._normalize_text(getattr(record_item, "uri", None))
        return not _segment_ref_supports_uri_only_contract(
            snapshot_kind=snapshot_kind,
            uri=resolved_uri,
        )

    def _attestation_probe_targets(
        self,
        *,
        document_id: str | None,
        resolved_uri: str | None,
        exact_document_id_authoritative: bool = False,
    ) -> tuple[tuple[str | None, str | None], ...]:
        """Return the bounded probe order for one accepted record attestation.

        ChonkyDB can acknowledge an async batch and even advertise the final
        ``document_id`` before that exact-id lookup becomes readable. Polling
        only once on the advertised id produces false-negative write failures on
        otherwise healthy backends. For immutable item/segment documents, that
        exact id is still the authoritative proof once it exists; a same-URI
        follow-up only burns visibility budget on a weaker surface. Mutable
        current heads keep the exact-id probe first, then fall back to the
        stable same-URI lookup when available.
        """

        targets: list[tuple[str | None, str | None]] = []
        if document_id:
            targets.append((document_id, None))
            if exact_document_id_authoritative:
                return tuple(targets)
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
    ) -> _AttestedRecordMatch | None:
        if not isinstance(payload, Mapping):
            return None
        for candidate, candidate_document_id in self._iter_attestation_candidates(payload):
            if any(candidate == expected for expected in expected_payloads):
                return _AttestedRecordMatch(document_id=candidate_document_id)
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
        if self._record_items_are_projection_only_graph_segments(record_items):
            # Mixed-corpus graph seeds can build many ~39-41 KB projection
            # segments. Packing them back into ~500 KB `/records/bulk` jobs
            # reintroduces the same backend queue saturation that segment-level
            # splitting was supposed to avoid. Keep graph segment bulks small
            # enough that they stay in the lighter async-overload envelope.
            max_request_bytes = min(max_request_bytes, _GRAPH_SEGMENT_BULK_REQUEST_MAX_BYTES)
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

    def _record_items_are_projection_only_graph_segments(
        self,
        record_items: list[ChonkyDBRecordItem],
    ) -> bool:
        """Return whether one bulk-write call only carries graph segment docs."""

        batch = tuple(record_items)
        if self._record_batch_payload_kind(batch=batch) != "catalog_segment_record_batch":
            return False
        snapshot_kinds = {
            self._normalize_text(
                (
                    getattr(item, "metadata", {}).get("twinr_snapshot_kind")
                    if isinstance(getattr(item, "metadata", None), Mapping)
                    else None
                )
                or (
                    getattr(item, "payload", {}).get("snapshot_kind")
                    if isinstance(getattr(item, "payload", None), Mapping)
                    else None
                )
            )
            for item in batch
        }
        snapshot_kinds.discard("")
        return bool(snapshot_kinds) and snapshot_kinds <= _PROJECTION_ONLY_CATALOG_SNAPSHOT_KINDS

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
