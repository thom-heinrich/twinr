from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import stat
import socket
import sys
import tempfile
from threading import Barrier, Event, Lock, Thread
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test.longterm_test_program import make_test_extractor
from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.chonkydb.client import ChonkyDBError
from twinr.memory.longterm.reasoning.consolidator import LongTermMemoryConsolidator
from twinr.memory.longterm.core.models import (
    LongTermConsolidationResultV1,
    LongTermGraphEdgeCandidateV1,
    LongTermMemoryConflictV1,
    LongTermMemoryObjectV1,
    LongTermSourceRefV1,
)
from twinr.memory.longterm.storage import remote_read_observability
from twinr.memory.longterm.storage.remote_read_diagnostics import (
    LongTermRemoteReadContext,
    _classify_remote_read_exception,
    record_remote_read_diagnostic,
)
from twinr.memory.longterm.storage.remote_state import (
    LongTermRemoteReadFailedError,
    LongTermRemoteUnavailableError,
)
from twinr.memory.longterm.storage.store import LongTermStructuredStore, _write_json_atomic
from twinr.memory.longterm.reasoning.truth import LongTermTruthMaintainer
from twinr.ops.events import TwinrOpsEventStore
from twinr.text_utils import retrieval_terms

_TEST_CORINNA_PHONE_OLD = "+15555551234"
_TEST_CORINNA_PHONE_NEW = "+15555558877"
_TEST_MARTA_PHONE_OLD = "+15555551122"
_TEST_MARTA_PHONE_NEW = "+15555553456"


class _FakeRemoteState:
    def __init__(self) -> None:
        self.client = _FakeChonkyClient()
        self.enabled = True
        self.required = False
        self.namespace = "test-namespace"
        self.read_client = self.client
        self.write_client = self.client
        self.config = SimpleNamespace(
            long_term_memory_migration_enabled=False,
            long_term_memory_migration_batch_size=64,
            long_term_memory_remote_read_timeout_s=8.0,
            long_term_memory_remote_write_timeout_s=15.0,
            long_term_memory_remote_flush_timeout_s=60.0,
            long_term_memory_remote_bulk_request_max_bytes=512 * 1024,
            long_term_memory_remote_shard_max_content_chars=1000,
            long_term_memory_remote_max_content_chars=2_000_000,
        )
        self.snapshots: dict[str, dict[str, object]] = {}

    def load_snapshot(self, *, snapshot_kind: str, local_path=None):
        del local_path
        payload = self.snapshots.get(snapshot_kind)
        return dict(payload) if isinstance(payload, dict) else None

    def save_snapshot(self, *, snapshot_kind: str, payload):
        self.snapshots[snapshot_kind] = dict(payload)


class _ProbeOnlyCatalogRemoteState(_FakeRemoteState):
    def __init__(self) -> None:
        super().__init__()
        self.required = True
        self.load_calls: list[str] = []
        self.probe_calls: list[dict[str, object]] = []

    def load_snapshot(self, *, snapshot_kind: str, local_path=None):
        del local_path
        self.load_calls.append(snapshot_kind)
        return super().load_snapshot(snapshot_kind=snapshot_kind)

    def probe_snapshot_load(
        self,
        *,
        snapshot_kind: str,
        local_path=None,
        prefer_cached_document_id: bool = False,
        prefer_metadata_only: bool = False,
        fast_fail: bool = False,
    ):
        del local_path
        self.probe_calls.append(
            {
                "snapshot_kind": snapshot_kind,
                "prefer_cached_document_id": prefer_cached_document_id,
                "prefer_metadata_only": prefer_metadata_only,
                "fast_fail": fast_fail,
            }
        )
        payloads = {
            "objects": {
                "schema": "twinr_memory_object_catalog_v3",
                "version": 3,
                "items_count": 0,
                "segments": [],
                "written_at": "2026-03-29T16:00:00+00:00",
            },
            "conflicts": {
                "schema": "twinr_memory_conflict_catalog_v3",
                "version": 3,
                "items_count": 0,
                "segments": [],
                "written_at": "2026-03-29T16:00:00+00:00",
            },
            "archive": {
                "schema": "twinr_memory_archive_catalog_v3",
                "version": 3,
                "items_count": 0,
                "segments": [],
                "written_at": "2026-03-29T16:00:00+00:00",
            },
        }
        return SimpleNamespace(
            snapshot_kind=snapshot_kind,
            status="found",
            detail=None,
            payload=dict(payloads[snapshot_kind]),
        )


class _ConcurrentEnsureRemoteState(_FakeRemoteState):
    def __init__(self) -> None:
        super().__init__()
        self.required = True
        self.load_calls: list[str] = []
        self.max_concurrent_loads = 0
        self._active_loads = 0
        self._active_lock = Lock()
        self._release_loads = Event()

    def load_snapshot(self, *, snapshot_kind: str, local_path=None):
        del local_path
        with self._active_lock:
            self.load_calls.append(snapshot_kind)
            self._active_loads += 1
            self.max_concurrent_loads = max(self.max_concurrent_loads, self._active_loads)
            if self._active_loads >= 3:
                self._release_loads.set()
        try:
            if not self._release_loads.wait(timeout=1.0):
                raise AssertionError("expected object/conflict/archive remote snapshot loads to overlap")
            return super().load_snapshot(snapshot_kind=snapshot_kind)
        finally:
            with self._active_lock:
                self._active_loads -= 1


class _SerialEnsureRemoteState(_FakeRemoteState):
    def __init__(self) -> None:
        super().__init__()
        self.required = True
        self.load_calls: list[str] = []
        self.max_concurrent_loads = 0
        self._active_loads = 0
        self._active_lock = Lock()

    def load_snapshot(self, *, snapshot_kind: str, local_path=None):
        del local_path
        with self._active_lock:
            self.load_calls.append(snapshot_kind)
            self._active_loads += 1
            self.max_concurrent_loads = max(self.max_concurrent_loads, self._active_loads)
        try:
            time.sleep(0.01)
            return super().load_snapshot(snapshot_kind=snapshot_kind)
        finally:
            with self._active_lock:
                self._active_loads -= 1


class _LaggingSnapshotVisibilityRemoteState(_FakeRemoteState):
    def __init__(self) -> None:
        super().__init__()
        self.required = True
        self._pending_snapshots: dict[str, dict[str, object]] = {}

    def save_snapshot(self, *, snapshot_kind: str, payload):
        normalized = dict(payload)
        if snapshot_kind in self.snapshots:
            self._pending_snapshots[snapshot_kind] = normalized
            return
        self.snapshots[snapshot_kind] = normalized

    def promote_pending(self, snapshot_kind: str) -> None:
        pending = self._pending_snapshots.pop(snapshot_kind, None)
        if pending is not None:
            self.snapshots[snapshot_kind] = pending


class _FakeChonkyClient:
    def __init__(self, *, max_items_per_bulk: int | None = None, max_request_bytes: int | None = None) -> None:
        self._next_document_id = 1
        self.max_items_per_bulk = max_items_per_bulk
        self.max_request_bytes = max_request_bytes
        self.supports_topk_records = False
        self.bulk_calls = 0
        self.retrieve_calls = 0
        self.topk_records_calls = 0
        self.topk_records_payloads: list[dict[str, object]] = []
        self.fetch_full_document_calls = 0
        self.bulk_request_bytes: list[int] = []
        self.bulk_request_schemas: list[tuple[str, ...]] = []
        self.records_by_document_id: dict[str, dict[str, object]] = {}
        self.records_by_uri: dict[str, dict[str, object]] = {}

    def store_records_bulk(self, request):
        items = tuple(getattr(request, "items", ()))
        self.bulk_calls += 1
        self.bulk_request_schemas.append(
            tuple(
                str(payload.get("schema"))
                for payload in (getattr(item, "payload", None) for item in items)
                if isinstance(payload, dict)
            )
        )
        if self.max_items_per_bulk is not None and len(items) > self.max_items_per_bulk:
            raise LongTermRemoteUnavailableError("bulk request too large")
        payload = request.to_payload() if hasattr(request, "to_payload") else dict(request)
        request_bytes = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        self.bulk_request_bytes.append(request_bytes)
        if self.max_request_bytes is not None and request_bytes > self.max_request_bytes:
            raise LongTermRemoteUnavailableError("bulk request exceeds byte limit")
        response_items = []
        for item in items:
            document_id = f"doc-{self._next_document_id}"
            self._next_document_id += 1
            record = {
                "document_id": document_id,
                "payload": dict(getattr(item, "payload", {}) or {}),
                "metadata": dict(getattr(item, "metadata", {}) or {}),
                "content": getattr(item, "content", None),
                "uri": getattr(item, "uri", None),
            }
            self.records_by_document_id[document_id] = record
            uri = record.get("uri")
            if isinstance(uri, str) and uri:
                self.records_by_uri[uri] = record
            response_items.append({"document_id": document_id})
        return {"items": response_items}

    def fetch_full_document(self, *, document_id=None, origin_uri=None, include_content=True, max_content_chars=4000):
        del include_content
        del max_content_chars
        self.fetch_full_document_calls += 1
        if isinstance(document_id, str) and document_id:
            record = self.records_by_document_id.get(document_id)
            if record is not None:
                return dict(record)
        if isinstance(origin_uri, str) and origin_uri:
            record = self.records_by_uri.get(origin_uri)
            if record is not None:
                return dict(record)
        raise LongTermRemoteUnavailableError("remote document unavailable")

    def retrieve(self, request):
        payload = request.to_payload() if hasattr(request, "to_payload") else dict(request)
        query_text = str(payload.get("query_text") or "").lower()
        allowed = set(str(value) for value in payload.get("allowed_doc_ids", ()) if str(value))
        self.retrieve_calls += 1
        ranked = []
        for document_id, record in self.records_by_document_id.items():
            if allowed and document_id not in allowed:
                continue
            content = str(record.get("content") or "").lower()
            if query_text == "__allowed_doc_ids__" and allowed:
                pass
            elif query_text and query_text not in content:
                continue
            ranked.append(
                {
                    "payload_id": document_id,
                    "relevance_score": 1.0,
                    "metadata": dict(record.get("metadata") or {}),
                    "content": record.get("content"),
                    "source_index": "fulltext",
                    "candidate_origin": "fulltext",
                }
            )
        return SimpleNamespace(
            success=True,
            mode="advanced",
            results=tuple(SimpleNamespace(**item) for item in ranked),
            indexes_used=("fulltext",),
            raw={"results": [dict(item) for item in ranked]},
        )

    def topk_records(self, request):
        payload = request.to_payload() if hasattr(request, "to_payload") else dict(request)
        query_text = str(payload.get("query_text") or "").lower()
        allowed = set(str(value) for value in payload.get("allowed_doc_ids", ()) if str(value))
        self.topk_records_calls += 1
        self.topk_records_payloads.append(dict(payload))
        # Preserve legacy retrieve counters for tests that only assert that a remote read happened.
        self.retrieve_calls += 1
        ranked = []
        for document_id, record in self.records_by_document_id.items():
            if allowed and document_id not in allowed:
                continue
            content = str(record.get("content") or "").lower()
            if query_text == "__allowed_doc_ids__" and allowed:
                pass
            elif query_text and query_text not in content:
                continue
            ranked.append(
                {
                    "payload_id": document_id,
                    "document_id": document_id,
                    "relevance_score": 1.0,
                    "metadata": dict(record.get("metadata") or {}),
                    "payload": dict(record.get("payload") or {}),
                    "payload_source": "record.payload",
                    "source_index": "fulltext",
                    "candidate_origin": "fulltext",
                }
            )
        return SimpleNamespace(
            success=True,
            mode="advanced",
            results=tuple(SimpleNamespace(**item) for item in ranked),
            indexes_used=("fulltext",),
            scope_ref=payload.get("scope_ref"),
            query_plan={"latency_ms": {"search": 1.0, "materialize": 0.2}},
            raw={"results": [dict(item) for item in ranked]},
        )


class _TimeoutingScopeTopKClient(_FakeChonkyClient):
    def __init__(self) -> None:
        super().__init__()
        self.supports_topk_records = True
        self._failed_scope_queries = 0

    def topk_records(self, request):
        payload = request.to_payload() if hasattr(request, "to_payload") else dict(request)
        if payload.get("scope_ref") and self._failed_scope_queries == 0:
            self._failed_scope_queries += 1
            raise ChonkyDBError(
                "ChonkyDB request failed for POST /v1/external/retrieve/topk_records: The read operation timed out"
            )
        return super().topk_records(request)


class _AlwaysTimeoutingScopeTopKClient(_FakeChonkyClient):
    def __init__(self) -> None:
        super().__init__()
        self.supports_topk_records = True

    def topk_records(self, request):
        payload = request.to_payload() if hasattr(request, "to_payload") else dict(request)
        if payload.get("scope_ref"):
            self.topk_records_calls += 1
            self.topk_records_payloads.append(dict(payload))
            raise ChonkyDBError(
                "ChonkyDB request failed for POST /v1/external/retrieve/topk_records: The read operation timed out"
            )
        return super().topk_records(request)


class _Http503ScopeTopKClient(_FakeChonkyClient):
    def __init__(self) -> None:
        super().__init__()
        self.supports_topk_records = True

    def topk_records(self, request):
        payload = request.to_payload() if hasattr(request, "to_payload") else dict(request)
        if payload.get("scope_ref"):
            self.topk_records_calls += 1
            self.topk_records_payloads.append(dict(payload))
            raise ChonkyDBError(
                "ChonkyDB request failed for POST /v1/external/retrieve/topk_records",
                status_code=503,
                response_json={
                    "detail": "payload_sync_bulk_busy",
                    "error": "payload_sync_bulk_busy",
                    "error_type": "ServerBusy",
                },
            )
        return super().topk_records(request)


class _TransientSegmentTimeoutClient(_FakeChonkyClient):
    def __init__(self, *, failing_document_id: str, failing_uri: str) -> None:
        super().__init__()
        self.fetch_attempts_by_document_id: dict[str, int] = {}
        self.fetch_attempts_by_uri: dict[str, int] = {}
        self._remaining_document_failures = {failing_document_id: 1}
        self._remaining_uri_failures = {failing_uri: 1}

    def fetch_full_document(self, *, document_id=None, origin_uri=None, include_content=True, max_content_chars=4000):
        if isinstance(document_id, str) and document_id:
            self.fetch_attempts_by_document_id[document_id] = self.fetch_attempts_by_document_id.get(document_id, 0) + 1
            remaining_failures = self._remaining_document_failures.get(document_id, 0)
            if remaining_failures > 0:
                self._remaining_document_failures[document_id] = remaining_failures - 1
                raise ChonkyDBError(
                    "ChonkyDB request failed for GET /v1/external/documents/full: The read operation timed out"
                )
        if isinstance(origin_uri, str) and origin_uri:
            self.fetch_attempts_by_uri[origin_uri] = self.fetch_attempts_by_uri.get(origin_uri, 0) + 1
            remaining_failures = self._remaining_uri_failures.get(origin_uri, 0)
            if remaining_failures > 0:
                self._remaining_uri_failures[origin_uri] = remaining_failures - 1
                raise ChonkyDBError(
                    "ChonkyDB request failed for GET /v1/external/documents/full: The read operation timed out"
                )
        return super().fetch_full_document(
            document_id=document_id,
            origin_uri=origin_uri,
            include_content=include_content,
            max_content_chars=max_content_chars,
        )


class _AsyncBulkEventuallyVisibleClient(_FakeChonkyClient):
    def __init__(self, *, stale_reads_before_visible: int = 2) -> None:
        super().__init__()
        self.stale_reads_before_visible = stale_reads_before_visible
        self.fetch_attempts_by_uri: dict[str, int] = {}
        self.bulk_execution_modes: list[str] = []
        self.pending_records_by_uri: dict[str, dict[str, object]] = {}

    def store_records_bulk(self, request):
        items = tuple(getattr(request, "items", ()))
        self.bulk_execution_modes.append(str(getattr(request, "execution_mode", "")))
        for item in items:
            document_id = f"doc-{self._next_document_id}"
            self._next_document_id += 1
            record = {
                "document_id": document_id,
                "payload": dict(getattr(item, "payload", {}) or {}),
                "metadata": dict(getattr(item, "metadata", {}) or {}),
                "content": getattr(item, "content", None),
                "uri": getattr(item, "uri", None),
            }
            uri = str(record.get("uri") or "")
            if uri:
                self.pending_records_by_uri[uri] = record
        return {"success": True, "job_id": "job-bulk-1", "status": "pending"}

    def fetch_full_document(self, *, document_id=None, origin_uri=None, include_content=True, max_content_chars=4000):
        del include_content
        del max_content_chars
        self.fetch_full_document_calls += 1
        if isinstance(document_id, str) and document_id:
            return super().fetch_full_document(
                document_id=document_id,
                origin_uri=origin_uri,
                include_content=True,
                max_content_chars=4000,
            )
        uri = str(origin_uri or "")
        if uri and uri in self.pending_records_by_uri:
            attempts = self.fetch_attempts_by_uri.get(uri, 0) + 1
            self.fetch_attempts_by_uri[uri] = attempts
            if attempts > self.stale_reads_before_visible:
                record = dict(self.pending_records_by_uri[uri])
                self.records_by_document_id[str(record["document_id"])] = dict(record)
                self.records_by_uri[uri] = dict(record)
                return record
        return super().fetch_full_document(
            document_id=document_id,
            origin_uri=origin_uri,
            include_content=True,
            max_content_chars=4000,
        )


class _AsyncBulkNeverVisibleClient(_FakeChonkyClient):
    def __init__(self) -> None:
        super().__init__()
        self.bulk_execution_modes: list[str] = []

    def store_records_bulk(self, request):
        self.bulk_execution_modes.append(str(getattr(request, "execution_mode", "")))
        return {"success": True, "job_id": "job-bulk-1", "status": "pending"}


class _AsyncBulkJobStatusClient(_AsyncBulkEventuallyVisibleClient):
    def __init__(self) -> None:
        super().__init__(stale_reads_before_visible=999)
        self.job_status_calls: list[str] = []
        self._job_records_by_id: dict[str, list[dict[str, object]]] = {}
        self._next_job_id = 1
        self.bulk_timeout_seconds: list[float | None] = []

    def store_records_bulk(self, request):
        items = tuple(getattr(request, "items", ()))
        self.bulk_execution_modes.append(str(getattr(request, "execution_mode", "")))
        timeout_seconds = getattr(request, "timeout_seconds", None)
        self.bulk_timeout_seconds.append(None if timeout_seconds is None else float(timeout_seconds))
        job_id = f"job-bulk-{self._next_job_id}"
        self._next_job_id += 1
        job_records: list[dict[str, object]] = []
        for item in items:
            document_id = f"doc-{self._next_document_id}"
            self._next_document_id += 1
            record = {
                "document_id": document_id,
                "payload": dict(getattr(item, "payload", {}) or {}),
                "metadata": dict(getattr(item, "metadata", {}) or {}),
                "content": getattr(item, "content", None),
                "uri": getattr(item, "uri", None),
            }
            uri = str(record.get("uri") or "")
            if uri:
                self.pending_records_by_uri[uri] = record
            self.records_by_document_id[document_id] = dict(record)
            if uri:
                self.records_by_uri[uri] = dict(record)
            job_records.append(record)
        self._job_records_by_id[job_id] = job_records
        return {"success": True, "job_id": job_id, "status": "pending"}

    def job_status(self, job_id):
        self.job_status_calls.append(str(job_id))
        records = self._job_records_by_id[str(job_id)]
        return {
            "success": True,
            "job_id": str(job_id),
            "status": "done",
            "result": {
                "success": True,
                "items": [{"document_id": str(record["document_id"])} for record in records],
            },
        }


class _LargeWindowTimeoutingScopeTopKClient(_FakeChonkyClient):
    def __init__(self, *, max_ok_limit: int) -> None:
        super().__init__()
        self.supports_topk_records = True
        self.max_ok_limit = max_ok_limit
        self.attempted_limits: list[int] = []

    def topk_records(self, request):
        payload = request.to_payload() if hasattr(request, "to_payload") else dict(request)
        self.attempted_limits.append(int(payload.get("result_limit") or 0))
        if payload.get("scope_ref") and int(payload.get("result_limit") or 0) > self.max_ok_limit:
            raise ChonkyDBError(
                "ChonkyDB request failed for POST /v1/external/retrieve/topk_records: The read operation timed out"
            )
        return super().topk_records(request)


class _EmptyScopeTopKClient(_FakeChonkyClient):
    def __init__(self) -> None:
        super().__init__()
        self.supports_topk_records = True

    def topk_records(self, request):
        payload = request.to_payload() if hasattr(request, "to_payload") else dict(request)
        if payload.get("scope_ref"):
            self.topk_records_calls += 1
            self.topk_records_payloads.append(dict(payload))
            return SimpleNamespace(
                success=True,
                mode="advanced",
                results=(),
                indexes_used=("fulltext",),
                scope_ref=payload.get("scope_ref"),
                query_plan={"latency_ms": {"search": 1.0, "materialize": 0.2}},
                raw={"results": []},
            )
        return super().topk_records(request)


class _StaleScopeTopKClient(_FakeChonkyClient):
    def __init__(self) -> None:
        super().__init__()
        self.supports_topk_records = True
        self.scope_records: tuple[dict[str, object], ...] = ()

    def snapshot_scope_records(self) -> None:
        self.scope_records = tuple(
            {
                "document_id": str(record.get("document_id") or ""),
                "payload": dict(record.get("payload") or {}),
                "metadata": dict(record.get("metadata") or {}),
                "content": record.get("content"),
                "uri": record.get("uri"),
            }
            for record in self.records_by_document_id.values()
        )

    def topk_records(self, request):
        payload = request.to_payload() if hasattr(request, "to_payload") else dict(request)
        if payload.get("scope_ref"):
            self.topk_records_calls += 1
            self.topk_records_payloads.append(dict(payload))
            query_text = str(payload.get("query_text") or "").lower()
            ranked = []
            for record in self.scope_records:
                content = str(record.get("content") or "").lower()
                if query_text and query_text not in content:
                    continue
                ranked.append(
                    {
                        "payload_id": str(record.get("document_id") or ""),
                        "document_id": str(record.get("document_id") or ""),
                        "relevance_score": 1.0,
                        "metadata": dict(record.get("metadata") or {}),
                        "payload": dict(record.get("payload") or {}),
                        "payload_source": "record.payload",
                        "source_index": "fulltext",
                        "candidate_origin": "fulltext",
                    }
                )
            return SimpleNamespace(
                success=True,
                mode="advanced",
                results=tuple(SimpleNamespace(**item) for item in ranked),
                indexes_used=("fulltext",),
                scope_ref=payload.get("scope_ref"),
                query_plan={"latency_ms": {"search": 1.0, "materialize": 0.2}},
                raw={"results": [dict(item) for item in ranked]},
            )
        return super().topk_records(request)


class _LiveShapeChonkyClient(_FakeChonkyClient):
    def __init__(self) -> None:
        super().__init__()
        self.supports_topk_records = False

    def fetch_full_document(self, *, document_id=None, origin_uri=None, include_content=True, max_content_chars=4000):
        del max_content_chars
        record = None
        if isinstance(document_id, str) and document_id:
            record = self.records_by_document_id.get(document_id)
        if record is None and isinstance(origin_uri, str) and origin_uri:
            record = self.records_by_uri.get(origin_uri)
        if record is None:
            raise LongTermRemoteUnavailableError("remote document unavailable")
        payload_id = str(record.get("document_id") or "")
        metadata = record.get("metadata")
        content = record.get("content")
        chunk = {
            "payload_id": payload_id,
            "chonky_id": payload_id,
            "doc_id_int": 1,
            "metadata": dict(metadata) if isinstance(metadata, dict) else {},
        }
        if include_content:
            chunk["content"] = content
            chunk["content_summary"] = content
        return {
            "success": True,
            "document_id": payload_id,
            "origin_uri": str(record.get("uri") or ""),
            "chunk_count": 1,
            "chunks": [chunk],
        }


class _MetadataOnlyLiveShapeChonkyClient(_LiveShapeChonkyClient):
    def retrieve(self, request):
        response = super().retrieve(request)
        normalized_results = []
        for result in getattr(response, "results", ()):
            metadata = dict(getattr(result, "metadata", {}) or {})
            metadata.pop("twinr_payload", None)
            metadata.pop("twinr_payload_sha256", None)
            normalized_results.append(
                {
                    "payload_id": getattr(result, "payload_id", None),
                    "relevance_score": getattr(result, "relevance_score", None),
                    "metadata": metadata,
                    "content": "",
                    "source_index": getattr(result, "source_index", None),
                    "candidate_origin": getattr(result, "candidate_origin", None),
                }
            )
        return SimpleNamespace(
            success=getattr(response, "success", True),
            mode=getattr(response, "mode", "advanced"),
            results=tuple(SimpleNamespace(**item) for item in normalized_results),
            indexes_used=getattr(response, "indexes_used", ("fulltext",)),
            raw={"results": normalized_results},
        )

    def fetch_full_document(self, *, document_id=None, origin_uri=None, include_content=True, max_content_chars=4000):
        record = None
        if isinstance(document_id, str) and document_id:
            record = self.records_by_document_id.get(document_id)
        if record is None and isinstance(origin_uri, str) and origin_uri:
            record = self.records_by_uri.get(origin_uri)
        if record is None:
            raise LongTermRemoteUnavailableError("remote document unavailable")
        payload = dict(record.get("payload") or {})
        if not (
            isinstance(payload, dict)
            and payload.get("schema") == "twinr_memory_object_record_v2"
            and isinstance(payload.get("item_id"), str)
        ):
            return super().fetch_full_document(
                document_id=document_id,
                origin_uri=origin_uri,
                include_content=include_content,
                max_content_chars=max_content_chars,
            )
        payload_id = str(record.get("document_id") or "")
        metadata = dict(record.get("metadata") or {})
        metadata.pop("twinr_payload", None)
        metadata.pop("twinr_payload_sha256", None)
        chunk = {
            "payload_id": payload_id,
            "chonky_id": payload_id,
            "doc_id_int": 1,
            "metadata": metadata,
        }
        if include_content:
            chunk["content"] = ""
            chunk["content_summary"] = ""
        return {
            "success": True,
            "document_id": payload_id,
            "origin_uri": str(record.get("uri") or ""),
            "chunk_count": 1,
            "chunks": [chunk],
        }


class _NoItemFetchChonkyClient(_FakeChonkyClient):
    def __init__(self) -> None:
        super().__init__()
        self.item_fetch_attempts = 0

    def fetch_full_document(self, *, document_id=None, origin_uri=None, include_content=True, max_content_chars=4000):
        record = None
        if isinstance(document_id, str) and document_id:
            record = self.records_by_document_id.get(document_id)
        if record is None and isinstance(origin_uri, str) and origin_uri:
            record = self.records_by_uri.get(origin_uri)
        if record is None:
            raise LongTermRemoteUnavailableError("remote document unavailable")
        payload = dict(record.get("payload") or {})
        if payload.get("schema") == "twinr_memory_object_record_v2":
            self.item_fetch_attempts += 1
            raise AssertionError("item fetch should use retrieve batching before full-document fallback")
        return super().fetch_full_document(
            document_id=document_id,
            origin_uri=origin_uri,
            include_content=include_content,
            max_content_chars=max_content_chars,
        )


class _RetrieveErrorChonkyClient:
    def __init__(self, exc: BaseException, *, fallback_client: object | None = None) -> None:
        self.exc = exc
        self.fallback_client = fallback_client

    def retrieve(self, request):
        del request
        raise self.exc

    def fetch_full_document(self, **kwargs):
        if self.fallback_client is None:
            raise AttributeError("fetch_full_document")
        return self.fallback_client.fetch_full_document(**kwargs)


class _CorruptingRewriteChonkyClient(_LiveShapeChonkyClient):
    def store_records_bulk(self, request):
        items = tuple(getattr(request, "items", ()))
        self.bulk_calls += 1
        self.bulk_request_schemas.append(
            tuple(
                str(payload.get("schema"))
                for payload in (getattr(item, "payload", None) for item in items)
                if isinstance(payload, dict)
            )
        )
        payload = request.to_payload() if hasattr(request, "to_payload") else dict(request)
        request_bytes = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        self.bulk_request_bytes.append(request_bytes)
        response_items = []
        for item in items:
            uri = getattr(item, "uri", None)
            existing = self.records_by_uri.get(uri) if isinstance(uri, str) and uri else None
            normalized_payload = dict(getattr(item, "payload", {}) or {})
            normalized_metadata = dict(getattr(item, "metadata", {}) or {})
            normalized_content = getattr(item, "content", None)
            if (
                existing is not None
                and existing.get("payload") == normalized_payload
                and existing.get("metadata") == normalized_metadata
                and existing.get("content") == normalized_content
            ):
                document_id = str(existing.get("document_id") or "")
                corrupted = {
                    "document_id": document_id,
                    "payload": {},
                    "metadata": {},
                    "content": "",
                    "uri": uri,
                }
                self.records_by_document_id[document_id] = corrupted
                if isinstance(uri, str) and uri:
                    self.records_by_uri[uri] = corrupted
                response_items.append({"document_id": document_id})
                continue
            document_id = f"doc-{self._next_document_id}"
            self._next_document_id += 1
            record = {
                "document_id": document_id,
                "payload": normalized_payload,
                "metadata": normalized_metadata,
                "content": normalized_content,
                "uri": uri,
            }
            self.records_by_document_id[document_id] = record
            if isinstance(uri, str) and uri:
                self.records_by_uri[uri] = record
            response_items.append({"document_id": document_id})
        return {"items": response_items}


class _ExistingItemFetchFailingChonkyClient(_LiveShapeChonkyClient):
    def fetch_full_document(self, *, document_id=None, origin_uri=None, include_content=True, max_content_chars=4000):
        record = None
        if isinstance(document_id, str) and document_id:
            record = self.records_by_document_id.get(document_id)
        if record is None and isinstance(origin_uri, str) and origin_uri:
            record = self.records_by_uri.get(origin_uri)
        payload = dict(record.get("payload") or {}) if isinstance(record, dict) else {}
        if payload.get("schema") == "twinr_memory_object_record_v2":
            raise TimeoutError("item readback timed out")
        return super().fetch_full_document(
            document_id=document_id,
            origin_uri=origin_uri,
            include_content=include_content,
            max_content_chars=max_content_chars,
        )


class _TermOverlapChonkyClient(_FakeChonkyClient):
    def retrieve(self, request):
        payload = request.to_payload() if hasattr(request, "to_payload") else dict(request)
        query_terms = {
            term
            for term in retrieval_terms(payload.get("query_text"))
            if any(char.isalpha() for char in term)
        }
        allowed = set(str(value) for value in payload.get("allowed_doc_ids", ()) if str(value))
        self.retrieve_calls += 1
        ranked = []
        for document_id, record in self.records_by_document_id.items():
            if allowed and document_id not in allowed:
                continue
            content_terms = {
                term
                for term in retrieval_terms(record.get("content"))
                if any(char.isalpha() for char in term)
            }
            overlap = len(query_terms.intersection(content_terms))
            if query_terms and overlap <= 0:
                continue
            ranked.append(
                {
                    "payload_id": document_id,
                    "relevance_score": float(overlap or 1.0),
                    "metadata": dict(record.get("metadata") or {}),
                    "content": record.get("content"),
                    "source_index": "fulltext",
                    "candidate_origin": "fulltext",
                }
            )
        ranked.sort(key=lambda item: (item["relevance_score"], str(item["payload_id"])), reverse=True)
        return SimpleNamespace(
            success=True,
            mode="advanced",
            results=tuple(SimpleNamespace(**item) for item in ranked),
            indexes_used=("fulltext",),
            raw={"results": [dict(item) for item in ranked]},
        )


class _SemanticDriftChonkyClient(_FakeChonkyClient):
    def retrieve(self, request):
        payload = request.to_payload() if hasattr(request, "to_payload") else dict(request)
        allowed = set(str(value) for value in payload.get("allowed_doc_ids", ()) if str(value))
        self.retrieve_calls += 1
        ranked = []
        for document_id, record in self.records_by_document_id.items():
            if allowed and document_id not in allowed:
                continue
            ranked.append(
                {
                    "payload_id": document_id,
                    "relevance_score": 1.0,
                    "metadata": dict(record.get("metadata") or {}),
                    "content": record.get("content"),
                    "source_index": "semantic",
                    "candidate_origin": "semantic",
                }
            )
        return SimpleNamespace(
            success=True,
            mode="advanced",
            results=tuple(SimpleNamespace(**item) for item in ranked),
            indexes_used=("semantic",),
            raw={"results": [dict(item) for item in ranked]},
        )


class _SemanticDriftScopeTopKClient(_FakeChonkyClient):
    def __init__(self) -> None:
        super().__init__()
        self.supports_topk_records = True

    def topk_records(self, request):
        payload = request.to_payload() if hasattr(request, "to_payload") else dict(request)
        if payload.get("scope_ref"):
            self.topk_records_calls += 1
            self.topk_records_payloads.append(dict(payload))
            self.retrieve_calls += 1
            ranked = []
            for document_id, record in self.records_by_document_id.items():
                ranked.append(
                    {
                        "payload_id": document_id,
                        "document_id": document_id,
                        "relevance_score": 1.0,
                        "metadata": dict(record.get("metadata") or {}),
                        "payload": dict(record.get("payload") or {}),
                        "payload_source": "record.payload",
                        "source_index": "semantic",
                        "candidate_origin": "semantic",
                    }
                )
            return SimpleNamespace(
                success=True,
                mode="advanced",
                results=tuple(SimpleNamespace(**item) for item in ranked),
                indexes_used=("semantic",),
                scope_ref=payload.get("scope_ref"),
                query_plan={"latency_ms": {"search": 1.0, "materialize": 0.2}},
                raw={"results": [dict(item) for item in ranked]},
            )
        return super().topk_records(request)


class _FailingRemoteState(_FakeRemoteState):
    def __init__(self) -> None:
        super().__init__()
        self.required = True

    def load_snapshot(self, *, snapshot_kind: str, local_path=None):
        del snapshot_kind
        del local_path
        raise LongTermRemoteUnavailableError("remote unavailable")


class _FailingRemoteSaveState(_FakeRemoteState):
    def __init__(self) -> None:
        super().__init__()
        self.required = True

    def save_snapshot(self, *, snapshot_kind: str, payload):
        del snapshot_kind
        del payload
        raise LongTermRemoteUnavailableError("remote write unavailable")


def _config(root: str) -> TwinrConfig:
    return TwinrConfig(
        project_root=root,
        personality_dir="personality",
        memory_markdown_path=str(Path(root) / "state" / "MEMORY.md"),
        long_term_memory_path=str(Path(root) / "state" / "chonkydb"),
    )


def _source() -> LongTermSourceRefV1:
    return LongTermSourceRefV1(
        source_type="conversation_turn",
        event_ids=("turn:test",),
        speaker="user",
        modality="voice",
    )


def _count_remote_records_with_schema(client: _FakeChonkyClient, schema: str) -> int:
    return sum(
        1
        for record in client.records_by_document_id.values()
        if isinstance(record.get("payload"), dict) and record["payload"].get("schema") == schema
    )


def _count_bulk_calls_with_schema(client: _FakeChonkyClient, schema: str) -> int:
    return sum(1 for schemas in client.bulk_request_schemas if schemas and all(value == schema for value in schemas))


def _sparsify_catalog_segment_entries(client: _FakeChonkyClient, catalog_payload: dict[str, object]) -> None:
    for raw_segment in catalog_payload.get("segments", ()):
        if not isinstance(raw_segment, dict):
            continue
        document_id = raw_segment.get("document_id")
        if not isinstance(document_id, str) or not document_id:
            continue
        record = client.records_by_document_id.get(document_id)
        if not isinstance(record, dict):
            continue
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        items = payload.get("items")
        if not isinstance(items, list):
            continue
        sparse_items = []
        for raw_item in items:
            if not isinstance(raw_item, dict):
                continue
            sparse_items.append(
                {
                    key: raw_item[key]
                    for key in ("item_id", "document_id", "kind", "status", "updated_at", "payload_sha256")
                    if key in raw_item
                }
            )
        payload["items"] = sparse_items
        record["content"] = json.dumps(payload, ensure_ascii=False)


class LongTermStructuredStoreTests(unittest.TestCase):
    def test_atomic_json_write_survives_concurrent_writers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "race.json"
            errors: list[BaseException] = []

            def worker(index: int) -> None:
                try:
                    _write_json_atomic(target, {"writer": index})
                except BaseException as exc:
                    errors.append(exc)

            threads = [Thread(target=worker, args=(index,)) for index in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual(errors, [])
            payload = json.loads(target.read_text(encoding="utf-8"))

        self.assertIn(payload["writer"], range(8))

    def test_atomic_json_write_makes_structured_snapshot_world_readable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "objects.json"

            _write_json_atomic(target, {"writer": 1})

            mode = stat.S_IMODE(target.stat().st_mode)

        self.assertEqual(mode, 0o644)

    def test_apply_consolidation_persists_objects_and_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LongTermStructuredStore.from_config(_config(temp_dir))
            result = LongTermConsolidationResultV1(
                turn_id="turn:test",
                occurred_at=datetime(2026, 3, 14, 12, 0, tzinfo=ZoneInfo("Europe/Berlin")),
                episodic_objects=(
                    LongTermMemoryObjectV1(
                        memory_id="episode:turn_test",
                        kind="episode",
                        summary="Conversation turn recorded for long-term memory.",
                        source=_source(),
                        status="active",
                        confidence=1.0,
                        slot_key="episode:turn:test",
                        value_key="turn:test",
                    ),
                ),
                durable_objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:janina_spouse",
                        kind="relationship_fact",
                        summary="Janina is the user's wife.",
                        source=_source(),
                        status="active",
                        confidence=0.98,
                        slot_key="relationship:user:main:wife",
                        value_key="person:janina",
                        attributes={"person_ref": "person:janina"},
                    ),
                ),
                deferred_objects=(),
                conflicts=(
                    LongTermMemoryConflictV1(
                        slot_key="contact:person:corinna_maier:phone",
                        candidate_memory_id="fact:corinna_phone_new",
                        existing_memory_ids=("fact:corinna_phone_old",),
                        question="I have more than one contact detail for this person. Which one should I use?",
                        reason="Conflicting active memories exist for slot contact:person:corinna_maier:phone.",
                    ),
                ),
                graph_edges=(
                    LongTermGraphEdgeCandidateV1(
                        source_ref="user:main",
                        edge_type="social_family_of",
                        target_ref="person:janina",
                        confidence=0.98,
                    ),
                ),
            )

            store.apply_consolidation(result)

            objects = store.load_objects()
            conflicts = store.load_conflicts()

        self.assertEqual(len(objects), 2)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].slot_key, "contact:person:corinna_maier:phone")

    def test_remote_primary_store_keeps_snapshots_off_disk(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            result = LongTermConsolidationResultV1(
                turn_id="turn:test",
                occurred_at=datetime(2026, 3, 14, 12, 0, tzinfo=ZoneInfo("Europe/Berlin")),
                episodic_objects=(
                    LongTermMemoryObjectV1(
                        memory_id="episode:turn_test",
                        kind="episode",
                        summary="Conversation turn recorded for long-term memory.",
                        source=_source(),
                        status="active",
                        confidence=1.0,
                    ),
                ),
                durable_objects=(),
                deferred_objects=(),
                conflicts=(),
                graph_edges=(),
            )

            store.apply_consolidation(result)
            loaded = store.load_objects()

        self.assertFalse(store.objects_path.exists())
        self.assertEqual(len(loaded), 1)
        self.assertEqual(remote_state.snapshots["objects"]["schema"], "twinr_memory_object_catalog_v3")
        self.assertEqual(remote_state.snapshots["objects"]["items_count"], 1)
        self.assertEqual(len(remote_state.snapshots["objects"]["segments"]), 1)
        self.assertEqual(_count_remote_records_with_schema(remote_state.client, "twinr_memory_object_record_v2"), 1)
        self.assertEqual(_count_remote_records_with_schema(remote_state.client, "twinr_memory_object_catalog_segment_v1"), 1)

    def test_remote_primary_store_persists_conflicts_as_raw_snapshot_body(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            writer_store = LongTermStructuredStore(
                base_path=Path(temp_dir) / "writer" / "state" / "chonkydb",
                remote_state=remote_state,
            )
            reader_store = LongTermStructuredStore(
                base_path=Path(temp_dir) / "reader" / "state" / "chonkydb",
                remote_state=remote_state,
            )

            old_phone = LongTermMemoryObjectV1(
                memory_id="fact:corinna_phone_old",
                kind="contact_method_fact",
                summary=f"Corinna Maier can be reached at {_TEST_CORINNA_PHONE_OLD}.",
                source=_source(),
                status="active",
                confidence=0.95,
                slot_key="contact:person:corinna_maier:phone",
                value_key=_TEST_CORINNA_PHONE_OLD,
            )
            new_phone = LongTermMemoryObjectV1(
                memory_id="fact:corinna_phone_new",
                kind="contact_method_fact",
                summary=f"Corinna Maier can be reached at {_TEST_CORINNA_PHONE_NEW}.",
                source=_source(),
                status="uncertain",
                confidence=0.92,
                slot_key="contact:person:corinna_maier:phone",
                value_key=_TEST_CORINNA_PHONE_NEW,
            )
            conflict = LongTermMemoryConflictV1(
                slot_key="contact:person:corinna_maier:phone",
                candidate_memory_id="fact:corinna_phone_new",
                existing_memory_ids=("fact:corinna_phone_old",),
                question="Which phone number should I use for Corinna Maier?",
                reason="Conflicting phone numbers exist.",
            )

            writer_store.write_snapshot(
                objects=(old_phone, new_phone),
                conflicts=(conflict,),
                archived_objects=(),
            )

            conflicts = reader_store.load_conflicts()

        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].slot_key, "contact:person:corinna_maier:phone")
        self.assertEqual(remote_state.snapshots["conflicts"]["schema"], "twinr_memory_conflict_store")
        self.assertEqual(remote_state.snapshots["conflicts"]["version"], 1)
        self.assertEqual(len(remote_state.snapshots["conflicts"]["conflicts"]), 1)
        self.assertEqual(_count_remote_records_with_schema(remote_state.client, "twinr_memory_conflict_record_v2"), 0)
        self.assertEqual(_count_remote_records_with_schema(remote_state.client, "twinr_memory_conflict_catalog_segment_v1"), 0)

    def test_remote_read_diagnostics_classify_timeout_backend_and_contract_failures(self) -> None:
        timeout_exc = TimeoutError("retrieve timed out")
        backend_exc = ChonkyDBError("ChonkyDB request failed for POST /v1/external/retrieve", status_code=503)
        contract_exc = ChonkyDBError(
            "ChonkyDB returned an invalid payload for retrieve()",
            response_json={"results": []},
        )
        try:
            classified_dns = ""
            try:
                raise socket.gaierror(-3, "Temporary failure in name resolution")
            except socket.gaierror as inner:
                raise ChonkyDBError("ChonkyDB request failed for POST /v1/external/retrieve: dns") from inner
        except ChonkyDBError as dns_exc:
            classified_dns = _classify_remote_read_exception(dns_exc)

        self.assertEqual(_classify_remote_read_exception(timeout_exc), "timeout")
        self.assertEqual(_classify_remote_read_exception(backend_exc), "backend_http_error")
        self.assertEqual(_classify_remote_read_exception(contract_exc), "client_contract_error")
        self.assertEqual(classified_dns, "dns_resolution_error")

    def test_remote_read_diagnostics_can_override_ops_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with tempfile.TemporaryDirectory() as override_dir:
                remote_state = _FakeRemoteState()
                remote_state.config.project_root = str(Path(temp_dir))
                previous_override = os.environ.get("TWINR_REMOTE_READ_DIAGNOSTICS_PROJECT_ROOT")
                os.environ["TWINR_REMOTE_READ_DIAGNOSTICS_PROJECT_ROOT"] = override_dir
                try:
                    record_remote_read_diagnostic(
                        remote_state=remote_state,
                        context=LongTermRemoteReadContext(
                            snapshot_kind="conflicts",
                            operation="retrieve_search",
                            query_text="Corinna phone number",
                            allowed_doc_count=1,
                            catalog_entry_count=1,
                            result_limit=1,
                        ),
                        exc=TimeoutError("retrieve timed out"),
                        started_monotonic=time.monotonic(),
                        outcome="failed",
                    )
                finally:
                    if previous_override is None:
                        os.environ.pop("TWINR_REMOTE_READ_DIAGNOSTICS_PROJECT_ROOT", None)
                    else:
                        os.environ["TWINR_REMOTE_READ_DIAGNOSTICS_PROJECT_ROOT"] = previous_override

                default_events = TwinrOpsEventStore.from_project_root(temp_dir).tail(limit=5)
                override_events = TwinrOpsEventStore.from_project_root(override_dir).tail(limit=5)

        self.assertFalse(default_events)
        self.assertTrue(override_events)
        failed_event = next(
            event
            for event in override_events
            if event.get("event") == "longterm_remote_read_failed"
        )
        self.assertEqual(failed_event["data"]["snapshot_kind"], "conflicts")

    def test_remote_read_histogram_observation_creates_cross_service_lock_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.config.project_root = temp_dir

            remote_read_observability.record_remote_read_observation(
                remote_state=remote_state,
                context=LongTermRemoteReadContext(
                    snapshot_kind="objects",
                    operation="topk_search",
                    request_method="POST",
                    request_path="/v1/external/retrieve/topk_records",
                    request_payload_kind="topk_scope_query",
                ),
                latency_ms=125.0,
                outcome="ok",
                classification="ok",
            )

            lock_path = Path(temp_dir) / "artifacts" / "stores" / "ops" / ".longterm_remote_read_histograms.json.lock"
            lock_mode = stat.S_IMODE(lock_path.stat().st_mode)

        self.assertEqual(lock_mode, 0o666)

    def test_remote_read_histogram_observation_tolerates_foreign_owned_lock_file_mode_refresh_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.config.project_root = temp_dir
            lock_path = Path(temp_dir) / "artifacts" / "stores" / "ops" / ".longterm_remote_read_histograms.json.lock"
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_path.write_text("", encoding="utf-8")
            os.chmod(lock_path, 0o600)
            real_fchmod = remote_read_observability.os.fchmod
            call_count = 0

            def flaky_fchmod(fd: int, mode: int) -> None:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise PermissionError(1, "operation not permitted")
                real_fchmod(fd, mode)

            with patch.object(remote_read_observability.os, "fchmod", side_effect=flaky_fchmod):
                remote_read_observability.record_remote_read_observation(
                    remote_state=remote_state,
                    context=LongTermRemoteReadContext(
                        snapshot_kind="objects",
                        operation="topk_search",
                        request_method="POST",
                        request_path="/v1/external/retrieve/topk_records",
                        request_payload_kind="topk_scope_query",
                    ),
                    latency_ms=125.0,
                    outcome="ok",
                    classification="ok",
                )

            histogram_path = Path(temp_dir) / "artifacts" / "stores" / "ops" / "longterm_remote_read_histograms.json"
            histogram_exists = histogram_path.exists()

        self.assertGreaterEqual(call_count, 2)
        self.assertTrue(histogram_exists)

    def test_remote_read_histogram_atomic_write_uses_unique_temp_paths_for_concurrent_writers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            histogram_path = Path(temp_dir) / "artifacts" / "stores" / "ops" / "longterm_remote_read_histograms.json"
            real_replace = remote_read_observability.os.replace
            barrier = Barrier(2)
            replace_sources: list[str] = []
            errors: list[BaseException] = []

            def gated_replace(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
                del destination
                replace_sources.append(os.fspath(source))
                barrier.wait(timeout=2.0)
                real_replace(source, histogram_path)

            def worker(writer_id: int) -> None:
                try:
                    remote_read_observability._write_json_atomic(histogram_path, {"writer": writer_id})
                except BaseException as exc:  # pragma: no cover - assertion carries details on failure
                    errors.append(exc)

            with patch.object(remote_read_observability.os, "replace", side_effect=gated_replace):
                first = Thread(target=worker, args=(1,))
                second = Thread(target=worker, args=(2,))
                first.start()
                second.start()
                first.join()
                second.join()

        self.assertFalse(errors)
        self.assertEqual(len(replace_sources), 2)
        self.assertEqual(len({Path(source).name for source in replace_sources}), 2)

    def test_select_open_conflicts_logs_remote_retrieve_failure_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            remote_state = _FakeRemoteState()
            remote_state.config.project_root = str(project_root)
            remote_state.config.long_term_memory_remote_read_timeout_s = 9.0
            writer_store = LongTermStructuredStore(
                base_path=project_root / "writer" / "state" / "chonkydb",
                remote_state=remote_state,
            )
            reader_store = LongTermStructuredStore(
                base_path=project_root / "reader" / "state" / "chonkydb",
                remote_state=remote_state,
            )

            old_phone = LongTermMemoryObjectV1(
                memory_id="fact:corinna_phone_old",
                kind="contact_method_fact",
                summary=f"Corinna Maier can be reached at {_TEST_CORINNA_PHONE_OLD}.",
                source=_source(),
                status="active",
                confidence=0.95,
                slot_key="contact:person:corinna_maier:phone",
                value_key=_TEST_CORINNA_PHONE_OLD,
            )
            new_phone = LongTermMemoryObjectV1(
                memory_id="fact:corinna_phone_new",
                kind="contact_method_fact",
                summary=f"Corinna Maier can be reached at {_TEST_CORINNA_PHONE_NEW}.",
                source=_source(),
                status="uncertain",
                confidence=0.92,
                slot_key="contact:person:corinna_maier:phone",
                value_key=_TEST_CORINNA_PHONE_NEW,
            )
            conflict = LongTermMemoryConflictV1(
                slot_key="contact:person:corinna_maier:phone",
                candidate_memory_id="fact:corinna_phone_new",
                existing_memory_ids=("fact:corinna_phone_old",),
                question="Which phone number should I use for Corinna Maier?",
                reason="Conflicting phone numbers exist.",
            )
            old_marta_phone = LongTermMemoryObjectV1(
                memory_id="fact:marta_phone_old",
                kind="contact_method_fact",
                summary=f"Marta Schulz can be reached at {_TEST_MARTA_PHONE_OLD}.",
                source=_source(),
                status="active",
                confidence=0.95,
                slot_key="contact:person:marta_schulz:phone",
                value_key=_TEST_MARTA_PHONE_OLD,
            )
            new_marta_phone = LongTermMemoryObjectV1(
                memory_id="fact:marta_phone_new",
                kind="contact_method_fact",
                summary=f"Marta Schulz can be reached at {_TEST_MARTA_PHONE_NEW}.",
                source=_source(),
                status="uncertain",
                confidence=0.9,
                slot_key="contact:person:marta_schulz:phone",
                value_key=_TEST_MARTA_PHONE_NEW,
            )
            second_conflict = LongTermMemoryConflictV1(
                slot_key="contact:person:marta_schulz:phone",
                candidate_memory_id="fact:marta_phone_new",
                existing_memory_ids=("fact:marta_phone_old",),
                question="Which phone number should I use for Marta Schulz?",
                reason="Conflicting phone numbers exist.",
            )

            writer_store.write_snapshot(
                objects=(old_phone, new_phone, old_marta_phone, new_marta_phone),
                conflicts=(conflict, second_conflict),
                archived_objects=(),
            )
            remote_state.read_client = _RetrieveErrorChonkyClient(
                TimeoutError("retrieve timed out"),
                fallback_client=remote_state.client,
            )

            conflicts = reader_store.select_open_conflicts(query_text="phone number", limit=1)
            events = TwinrOpsEventStore.from_project_root(project_root).tail(limit=5)

        self.assertTrue(events)
        self.assertEqual(len(conflicts), 1)
        search_event = next(
            event
            for event in events
            if event.get("event") == "longterm_remote_read_degraded"
            and dict(event.get("data") or {}).get("operation") == "retrieve_search"
        )
        self.assertEqual(search_event["level"], "warning")
        self.assertIn("conflicts", search_event["message"])
        data = dict(search_event["data"])
        self.assertEqual(data["snapshot_kind"], "conflicts")
        self.assertEqual(data["operation"], "retrieve_search")
        self.assertEqual(data["classification"], "timeout")
        self.assertEqual(data["allowed_doc_count"], 2)
        self.assertEqual(data["catalog_entry_count"], 2)
        self.assertEqual(data["result_limit"], 1)
        self.assertEqual(data["read_timeout_s"], 9.0)
        self.assertEqual(data["error_type"], "TimeoutError")
        self.assertEqual(data["root_cause_type"], "TimeoutError")
        alert_event = next(
            event
            for event in events
            if event.get("event") == "longterm_remote_read_alert"
            and dict(event.get("data") or {}).get("operation") == "retrieve_search"
        )
        self.assertEqual(alert_event["level"], "warning")
        alert_data = dict(alert_event["data"])
        self.assertEqual(alert_data["classification"], "timeout")
        self.assertEqual(alert_data["alert_kind"], "timeout")

    def test_select_open_conflicts_short_circuits_when_remote_conflict_catalog_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.client.supports_topk_records = True
            store = LongTermStructuredStore(
                base_path=Path(temp_dir) / "state" / "chonkydb",
                remote_state=remote_state,
            )
            store.write_snapshot(objects=(), conflicts=(), archived_objects=())

            conflicts = store.select_open_conflicts(query_text="phone number", limit=3)

        self.assertEqual(conflicts, ())
        self.assertEqual(remote_state.client.topk_records_calls, 0)

    def test_remote_catalog_records_endpoint_and_payload_kind_histograms_for_topk_search_and_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.client.supports_topk_records = True
            remote_state.config.project_root = str(project_root)
            store = LongTermStructuredStore(
                base_path=project_root / "state" / "chonkydb",
                remote_state=remote_state,
            )
            store.write_snapshot(
                objects=tuple(
                    LongTermMemoryObjectV1(
                        memory_id=f"fact:{index}",
                        kind="fact",
                        summary=f"Früher stand die rote Thermoskanne im Flurschrank Fach {index}.",
                        details="Historische Ortsangabe zur roten Thermoskanne.",
                        source=_source(),
                        status="active",
                        confidence=0.99,
                        confirmed_by_user=True,
                        slot_key=f"object:red_thermos:location:{index}",
                        value_key=f"hallway_cupboard_{index}",
                    )
                    for index in range(4)
                ),
                conflicts=(),
                archived_objects=(),
            )
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None

            relevant = remote_catalog.search_current_item_payloads(
                snapshot_kind="objects",
                query_text="Thermoskanne",
                limit=3,
            )
            hydrated = remote_catalog.load_item_payloads(
                snapshot_kind="objects",
                item_ids=("fact:0", "fact:1"),
            )
            histogram_path = project_root / "artifacts" / "stores" / "ops" / "longterm_remote_read_histograms.json"
            payload = json.loads(histogram_path.read_text(encoding="utf-8"))

        self.assertEqual(len(relevant), 3)
        self.assertEqual(len(hydrated), 2)
        operations = dict(payload.get("operations") or {})
        self.assertIn("objects:topk_search", operations)
        self.assertIn("objects:topk_batch", operations)
        search_entry = dict(operations["objects:topk_search"])
        batch_entry = dict(operations["objects:topk_batch"])
        self.assertEqual(search_entry["last_outcome"], "ok")
        self.assertEqual(search_entry["last_classification"], "ok")
        self.assertEqual(search_entry["last_request_endpoint"], "POST /v1/external/retrieve/topk_records")
        self.assertEqual(search_entry["last_request_payload_kind"], "topk_scope_query")
        self.assertGreaterEqual(int(dict(search_entry["request_endpoint_counts"])["POST /v1/external/retrieve/topk_records"]), 1)
        self.assertGreaterEqual(int(dict(search_entry["request_payload_kind_counts"])["topk_scope_query"]), 1)
        self.assertGreaterEqual(int(search_entry["total_count"]), 1)
        self.assertEqual(batch_entry["last_outcome"], "ok")
        self.assertEqual(batch_entry["last_classification"], "ok")
        self.assertEqual(batch_entry["last_request_endpoint"], "POST /v1/external/retrieve/topk_records")
        self.assertEqual(batch_entry["last_request_payload_kind"], "topk_allowed_doc_batch")
        self.assertGreaterEqual(int(dict(batch_entry["request_endpoint_counts"])["POST /v1/external/retrieve/topk_records"]), 1)
        self.assertGreaterEqual(int(dict(batch_entry["request_payload_kind_counts"])["topk_allowed_doc_batch"]), 1)
        self.assertGreaterEqual(int(batch_entry["total_count"]), 1)

    def test_select_relevant_objects_degrades_remote_search_timeout_to_local_catalog_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.config.project_root = str(project_root)
            remote_state.config.long_term_memory_remote_read_timeout_s = 9.0
            writer_store = LongTermStructuredStore(
                base_path=project_root / "writer" / "state" / "chonkydb",
                remote_state=remote_state,
            )
            reader_store = LongTermStructuredStore(
                base_path=project_root / "reader" / "state" / "chonkydb",
                remote_state=remote_state,
            )

            thermos = LongTermMemoryObjectV1(
                memory_id="fact:thermos_location_old",
                kind="fact",
                summary="Früher stand deine rote Thermoskanne im Flurschrank.",
                source=_source(),
                status="active",
                confidence=0.96,
                slot_key="location:thermos",
                value_key="flurschrank",
            )
            jam_old = LongTermMemoryObjectV1(
                memory_id="fact:jam_preference_old",
                kind="fact",
                summary="Deine Lieblingsmarmelade ist Erdbeermarmelade.",
                source=_source(),
                status="active",
                confidence=0.94,
                slot_key="preference:breakfast:jam",
                value_key="strawberry",
            )
            jam_new = LongTermMemoryObjectV1(
                memory_id="fact:jam_preference_new",
                kind="fact",
                summary="Inzwischen magst du lieber Aprikosenmarmelade.",
                source=_source(),
                status="uncertain",
                confidence=0.95,
                slot_key="preference:breakfast:jam",
                value_key="apricot",
            )
            breakfast = LongTermMemoryObjectV1(
                memory_id="fact:jam_breakfast",
                kind="fact",
                summary="Du isst gern Marmelade auf Brot zum Frühstück.",
                source=_source(),
                status="active",
                confidence=0.84,
                slot_key="fact:user:breakfast:jam",
                value_key="jam_on_bread_at_breakfast",
            )

            writer_store.write_snapshot(
                objects=(thermos, jam_old, jam_new, breakfast),
                conflicts=(),
                archived_objects=(),
            )
            remote_state.read_client = _RetrieveErrorChonkyClient(
                TimeoutError("retrieve timed out"),
                fallback_client=remote_state.client,
            )

            relevant = reader_store.select_relevant_objects(
                query_text="Wo stand früher meine rote Thermoskanne?",
                limit=3,
            )
            events = TwinrOpsEventStore.from_project_root(project_root).tail(limit=8)

        self.assertEqual([item.memory_id for item in relevant], ["fact:thermos_location_old"])
        search_event = next(
            event
            for event in events
            if event.get("event") == "longterm_remote_read_degraded"
            and dict(event.get("data") or {}).get("operation") == "retrieve_search"
            and dict(event.get("data") or {}).get("snapshot_kind") == "objects"
        )
        data = dict(search_event["data"])
        self.assertEqual(data["classification"], "timeout")
        self.assertEqual(data["allowed_doc_count"], 4)
        self.assertEqual(data["catalog_entry_count"], 4)
        self.assertEqual(data["result_limit"], 3)

    def test_select_open_conflicts_skips_remote_search_when_candidate_set_already_fits_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.config.project_root = str(project_root)
            writer_store = LongTermStructuredStore(
                base_path=project_root / "writer" / "state" / "chonkydb",
                remote_state=remote_state,
            )
            reader_store = LongTermStructuredStore(
                base_path=project_root / "reader" / "state" / "chonkydb",
                remote_state=remote_state,
            )

            old_phone = LongTermMemoryObjectV1(
                memory_id="fact:corinna_phone_old",
                kind="contact_method_fact",
                summary=f"Corinna Maier can be reached at {_TEST_CORINNA_PHONE_OLD}.",
                source=_source(),
                status="active",
                confidence=0.95,
                slot_key="contact:person:corinna_maier:phone",
                value_key=_TEST_CORINNA_PHONE_OLD,
            )
            new_phone = LongTermMemoryObjectV1(
                memory_id="fact:corinna_phone_new",
                kind="contact_method_fact",
                summary=f"Corinna Maier can be reached at {_TEST_CORINNA_PHONE_NEW}.",
                source=_source(),
                status="uncertain",
                confidence=0.92,
                slot_key="contact:person:corinna_maier:phone",
                value_key=_TEST_CORINNA_PHONE_NEW,
            )
            conflict = LongTermMemoryConflictV1(
                slot_key="contact:person:corinna_maier:phone",
                candidate_memory_id="fact:corinna_phone_new",
                existing_memory_ids=("fact:corinna_phone_old",),
                question="Which phone number should I use for Corinna Maier?",
                reason="Conflicting phone numbers exist.",
            )

            writer_store.write_snapshot(
                objects=(old_phone, new_phone),
                conflicts=(conflict,),
                archived_objects=(),
            )
            remote_state.read_client = _RetrieveErrorChonkyClient(
                TimeoutError("retrieve timed out"),
                fallback_client=remote_state.client,
            )

            conflicts = reader_store.select_open_conflicts(query_text="Corinna phone number", limit=3)
            events = TwinrOpsEventStore.from_project_root(project_root).tail(limit=5)

        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].slot_key, "contact:person:corinna_maier:phone")
        self.assertFalse(any(event.get("event") == "longterm_remote_read_failed" for event in events))

    def test_select_open_conflicts_falls_back_when_scope_topk_returns_false_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.client = _TermOverlapChonkyClient()
            remote_state.client.supports_topk_records = True
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_preference_old",
                        kind="fact",
                        summary="Deine Lieblingsmarmelade ist Erdbeermarmelade.",
                        details="Aeltere Vorliebe fuer das Fruehstueck.",
                        source=_source(),
                        status="active",
                        confidence=0.94,
                        slot_key="preference:breakfast:jam",
                        value_key="strawberry",
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_preference_new",
                        kind="fact",
                        summary="Inzwischen magst du lieber Aprikosenmarmelade.",
                        details="Neuere Vorliebe fuer das Fruehstueck.",
                        source=_source(),
                        status="uncertain",
                        confidence=0.95,
                        slot_key="preference:breakfast:jam",
                        value_key="apricot",
                    ),
                ),
                conflicts=(
                    LongTermMemoryConflictV1(
                        slot_key="preference:breakfast:jam",
                        candidate_memory_id="fact:jam_preference_new",
                        existing_memory_ids=("fact:jam_preference_old",),
                        question="Welche Marmelade stimmt gerade?",
                        reason="Widerspruechliche Marmeladenpraeferenzen liegen vor.",
                    ),
                ),
                archived_objects=(),
            )

            conflicts = store.select_open_conflicts(
                query_text="Welche Marmeladen stehen gerade im Widerspruch?",
                limit=3,
            )

        self.assertGreater(remote_state.client.topk_records_calls, 0)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].slot_key, "preference:breakfast:jam")

    def test_search_catalog_entries_falls_back_when_scope_topk_returns_false_empty_for_objects(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.client = _TermOverlapChonkyClient()
            remote_state.client.supports_topk_records = True
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:thermos_location_old",
                        kind="fact",
                        summary="Früher stand die rote Thermoskanne im Flurschrank.",
                        details="Historische Ortsangabe zur roten Thermoskanne.",
                        source=_source(),
                        status="active",
                        confidence=0.99,
                        confirmed_by_user=True,
                        slot_key="object:red_thermos:location",
                        value_key="hallway_cupboard",
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:tea_preference",
                        kind="fact",
                        summary="Du trinkst gern schwarzen Tee.",
                        source=_source(),
                        status="active",
                        confidence=0.91,
                        slot_key="preference:drink:tea",
                        value_key="black_tea",
                    ),
                ),
                conflicts=(),
                archived_objects=(),
            )
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None

            relevant = remote_catalog.search_catalog_entries(
                snapshot_kind="objects",
                query_text="Wo stand früher meine rote Thermoskanne?",
                limit=1,
                eligible=lambda entry: dict(getattr(entry, "metadata", {}) or {}).get("status") in {"active", "candidate", "uncertain"},
            )

        self.assertEqual(tuple(entry.item_id for entry in relevant), ("fact:thermos_location_old",))

    def test_write_snapshot_logs_remote_bulk_write_failure_diagnostic(self) -> None:
        class _BulkWriteFailingClient(_FakeChonkyClient):
            def store_records_bulk(self, request):
                del request
                try:
                    raise socket.gaierror(-3, "Temporary failure in name resolution")
                except socket.gaierror as inner:
                    raise ChonkyDBError(
                        "ChonkyDB request failed for POST /v1/external/records/bulk: dns"
                    ) from inner

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.config.project_root = str(project_root)
            remote_state.config.long_term_memory_remote_write_timeout_s = 15.0
            failing_client = _BulkWriteFailingClient()
            remote_state.client = failing_client
            remote_state.read_client = failing_client
            remote_state.write_client = failing_client
            store = LongTermStructuredStore(
                base_path=project_root / "state" / "chonkydb",
                remote_state=remote_state,
            )
            object_fact = LongTermMemoryObjectV1(
                memory_id="fact:jam_preference_old",
                kind="fact",
                summary="Deine Lieblingsmarmelade ist Erdbeermarmelade.",
                source=_source(),
                status="active",
                confidence=0.94,
                slot_key="preference:breakfast:jam",
                value_key="strawberry",
            )

            with self.assertRaises(LongTermRemoteUnavailableError):
                store.write_snapshot(objects=(object_fact,), conflicts=(), archived_objects=())

            events = TwinrOpsEventStore.from_project_root(project_root).tail(limit=5)

        self.assertTrue(events)
        last_event = events[-1]
        self.assertEqual(last_event["event"], "longterm_remote_write_failed")
        self.assertEqual(last_event["level"], "warning")
        self.assertIn("dns_resolution_error", last_event["message"])
        data = dict(last_event["data"])
        self.assertEqual(data["snapshot_kind"], "objects")
        self.assertEqual(data["operation"], "store_records_bulk")
        self.assertEqual(data["classification"], "dns_resolution_error")
        self.assertEqual(data["request_kind"], "write")
        self.assertEqual(data["request_item_count"], 1)
        self.assertEqual(data["write_timeout_s"], 15.0)
        self.assertEqual(data["error_type"], "ChonkyDBError")
        self.assertEqual(data["root_cause_type"], "gaierror")
        self.assertTrue(data["request_correlation_id"])
        self.assertEqual(data["batch_index"], 1)
        self.assertEqual(data["batch_count"], 1)
        self.assertGreater(data["request_bytes"], 0)

    def test_write_snapshot_bulk_failure_preserves_elapsed_latency_and_correlation_context(self) -> None:
        class _DelayedBulkWriteFailingClient(_FakeChonkyClient):
            def store_records_bulk(self, request):
                del request
                time.sleep(0.02)
                try:
                    raise TimeoutError("write timed out")
                except TimeoutError as inner:
                    raise ChonkyDBError(
                        "ChonkyDB request failed for POST /v1/external/records/bulk: timed out"
                    ) from inner

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.config.project_root = str(project_root)
            remote_state.config.long_term_memory_remote_write_timeout_s = 15.0
            failing_client = _DelayedBulkWriteFailingClient()
            remote_state.client = failing_client
            remote_state.read_client = failing_client
            remote_state.write_client = failing_client
            store = LongTermStructuredStore(
                base_path=project_root / "state" / "chonkydb",
                remote_state=remote_state,
            )
            object_fact = LongTermMemoryObjectV1(
                memory_id="fact:jam_preference_old",
                kind="fact",
                summary="Deine Lieblingsmarmelade ist Erdbeermarmelade.",
                source=_source(),
                status="active",
                confidence=0.94,
                slot_key="preference:breakfast:jam",
                value_key="strawberry",
            )

            with self.assertRaises(LongTermRemoteUnavailableError) as raised:
                store.write_snapshot(objects=(object_fact,), conflicts=(), archived_objects=())

            events = TwinrOpsEventStore.from_project_root(project_root).tail(limit=5)

        self.assertTrue(events)
        last_event = events[-1]
        data = dict(last_event["data"])
        self.assertGreaterEqual(float(data["latency_ms"]), 15.0)
        self.assertTrue(data["request_correlation_id"])
        self.assertEqual(data["batch_index"], 1)
        self.assertEqual(data["batch_count"], 1)
        self.assertGreater(data["request_bytes"], 0)
        raised_context = getattr(raised.exception, "remote_write_context", None)
        self.assertIsInstance(raised_context, dict)
        assert isinstance(raised_context, dict)
        self.assertEqual(raised_context["request_correlation_id"], data["request_correlation_id"])
        self.assertIn(data["request_correlation_id"], str(raised.exception))

    def test_write_snapshot_bulk_failure_records_problem_detail_fields(self) -> None:
        class _Http503BulkWriteFailingClient(_FakeChonkyClient):
            def store_records_bulk(self, request):
                del request
                raise ChonkyDBError(
                    "ChonkyDB request failed for POST /v1/external/records/bulk",
                    status_code=503,
                    response_json={
                        "type": "about:blank",
                        "title": "ServerBusy",
                        "status": 503,
                        "detail": "payload_sync_bulk_busy",
                        "error": "payload_sync_bulk_busy",
                        "error_type": "ServerBusy",
                        "instance": "/v1/external/records/bulk",
                        "success": False,
                    },
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.config.project_root = str(project_root)
            failing_client = _Http503BulkWriteFailingClient()
            remote_state.client = failing_client
            remote_state.read_client = failing_client
            remote_state.write_client = failing_client
            store = LongTermStructuredStore(
                base_path=project_root / "state" / "chonkydb",
                remote_state=remote_state,
            )
            object_fact = LongTermMemoryObjectV1(
                memory_id="fact:jam_preference_old",
                kind="fact",
                summary="Deine Lieblingsmarmelade ist Erdbeermarmelade.",
                source=_source(),
                status="active",
                confidence=0.94,
                slot_key="preference:breakfast:jam",
                value_key="strawberry",
            )

            with self.assertRaises(LongTermRemoteUnavailableError):
                store.write_snapshot(objects=(object_fact,), conflicts=(), archived_objects=())

            events = TwinrOpsEventStore.from_project_root(project_root).tail(limit=5)

        self.assertTrue(events)
        data = dict(events[-1]["data"])
        self.assertEqual(data["status_code"], 503)
        self.assertEqual(data["response_detail"], "payload_sync_bulk_busy")
        self.assertEqual(data["response_error"], "payload_sync_bulk_busy")
        self.assertEqual(data["response_error_type"], "ServerBusy")

    def test_remote_catalog_bulk_write_attests_async_same_uri_visibility(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.config.long_term_memory_remote_retry_attempts = 4
            remote_state.config.long_term_memory_remote_retry_backoff_s = 0.0
            remote_state.client = _AsyncBulkEventuallyVisibleClient(stale_reads_before_visible=2)
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None
            uri = remote_catalog.item_uri(snapshot_kind="objects", item_id="fact:jam_preference")
            remote_state.client.records_by_document_id["doc-stale"] = {
                "document_id": "doc-stale",
                "payload": {
                    "schema": "twinr_memory_object_record_v2",
                    "version": 1,
                    "snapshot_kind": "objects",
                    "item_id": "fact:jam_preference",
                    "metadata": {
                        "twinr_snapshot_kind": "objects",
                        "twinr_memory_item_id": "fact:jam_preference",
                    },
                    "content": "Du magst Erdbeermarmelade.",
                },
                "metadata": {
                    "twinr_snapshot_kind": "objects",
                    "twinr_memory_item_id": "fact:jam_preference",
                    "twinr_payload_sha256": "stale",
                    "twinr_payload": {
                        "schema": "twinr_memory_object",
                        "version": 1,
                        "memory_id": "fact:jam_preference",
                        "kind": "fact",
                        "summary": "Du magst Erdbeermarmelade.",
                        "source": _source().to_payload(),
                        "status": "active",
                        "confidence": 0.91,
                        "slot_key": "preference:breakfast:jam",
                        "value_key": "strawberry",
                    },
                },
                "content": "Du magst Erdbeermarmelade.",
                "uri": uri,
            }
            remote_state.client.records_by_uri[uri] = dict(remote_state.client.records_by_document_id["doc-stale"])
            objects = (
                LongTermMemoryObjectV1(
                    memory_id="fact:jam_preference",
                    kind="fact",
                    summary="Du magst Aprikosenmarmelade.",
                    source=_source(),
                    status="active",
                    confidence=0.95,
                    slot_key="preference:breakfast:jam",
                    value_key="apricot",
                ),
            )

            payload = remote_catalog.build_catalog_payload(
                snapshot_kind="objects",
                item_payloads=(obj.to_payload() for obj in objects),
                item_id_getter=lambda item: item.get("memory_id"),
                metadata_builder=lambda item: item,
                content_builder=lambda item: str(item.get("summary") or ""),
            )

        self.assertEqual(remote_state.client.bulk_execution_modes, ["async", "async"])
        self.assertGreaterEqual(remote_state.client.fetch_attempts_by_uri[uri], 3)
        catalog_entries = remote_catalog._load_segmented_catalog_entries(
            definition=remote_catalog._require_definition("objects"),
            payload=payload,
        )
        self.assertEqual(len(catalog_entries), 1)
        self.assertEqual(catalog_entries[0].document_id, "doc-1")

    def test_remote_catalog_segment_read_retries_after_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.config.long_term_memory_remote_retry_attempts = 2
            remote_state.config.long_term_memory_remote_retry_backoff_s = 0.0
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None
            segment_document_id = "segment-doc-1"
            segment_uri = remote_catalog._catalog_segment_uri(snapshot_kind="objects", segment_index=0)
            client = _TransientSegmentTimeoutClient(
                failing_document_id=segment_document_id,
                failing_uri=segment_uri,
            )
            remote_state.client = client
            remote_state.read_client = client
            remote_state.write_client = client
            client.records_by_document_id[segment_document_id] = {
                "document_id": segment_document_id,
                "payload": {
                    "schema": "twinr_memory_object_catalog_segment_v1",
                    "version": 1,
                    "snapshot_kind": "objects",
                    "segment_index": 0,
                    "items": [
                        {
                            "item_id": "fact:jam_preference",
                            "document_id": "doc-1",
                            "summary": "Du magst Aprikosenmarmelade.",
                            "kind": "fact",
                            "status": "active",
                        }
                    ],
                },
                "metadata": {},
                "content": "segment",
                "uri": segment_uri,
            }
            client.records_by_uri[segment_uri] = dict(client.records_by_document_id[segment_document_id])

            entries = remote_catalog._load_segmented_catalog_entries(
                definition=remote_catalog._require_definition("objects"),
                payload={
                    "segments": [
                        {
                            "segment_index": 0,
                            "document_id": segment_document_id,
                            "uri": segment_uri,
                        }
                    ]
                },
            )

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].item_id, "fact:jam_preference")
        self.assertEqual(entries[0].document_id, "doc-1")
        self.assertEqual(client.fetch_attempts_by_document_id[segment_document_id], 2)
        self.assertEqual(client.fetch_attempts_by_uri[segment_uri], 1)

    def test_remote_catalog_bulk_write_prefers_async_job_document_ids_over_same_uri_head(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.config.long_term_memory_remote_retry_attempts = 2
            remote_state.config.long_term_memory_remote_retry_backoff_s = 0.0
            remote_state.client = _AsyncBulkJobStatusClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None
            uri = remote_catalog.item_uri(snapshot_kind="objects", item_id="fact:jam_preference")
            remote_state.client.records_by_document_id["doc-stale"] = {
                "document_id": "doc-stale",
                "payload": {"schema": "stale"},
                "metadata": {"twinr_memory_item_id": "fact:jam_preference"},
                "content": "stale",
                "uri": uri,
            }
            remote_state.client.records_by_uri[uri] = dict(remote_state.client.records_by_document_id["doc-stale"])
            objects = (
                LongTermMemoryObjectV1(
                    memory_id="fact:jam_preference",
                    kind="fact",
                    summary="Du magst Aprikosenmarmelade.",
                    source=_source(),
                    status="active",
                    confidence=0.95,
                    slot_key="preference:breakfast:jam",
                    value_key="apricot",
                ),
            )

            payload = remote_catalog.build_catalog_payload(
                snapshot_kind="objects",
                item_payloads=(obj.to_payload() for obj in objects),
                item_id_getter=lambda item: item.get("memory_id"),
                metadata_builder=lambda item: item,
                content_builder=lambda item: str(item.get("summary") or ""),
            )

        self.assertEqual(remote_state.client.bulk_execution_modes, ["async", "async"])
        self.assertTrue(remote_state.client.job_status_calls)
        self.assertEqual(remote_state.client.fetch_attempts_by_uri.get(uri, 0), 0)
        catalog_entries = remote_catalog._load_segmented_catalog_entries(
            definition=remote_catalog._require_definition("objects"),
            payload=payload,
        )
        self.assertEqual(len(catalog_entries), 1)
        self.assertEqual(catalog_entries[0].document_id, "doc-1")
        self.assertTrue(all(value == 60.0 for value in remote_state.client.bulk_timeout_seconds))

    def test_remote_catalog_async_job_timeout_uses_flush_budget_floor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.config.long_term_memory_remote_write_timeout_s = 15.0
            remote_state.config.long_term_memory_remote_flush_timeout_s = 60.0
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None

            self.assertEqual(remote_catalog._remote_async_job_timeout_s(), 60.0)
            self.assertEqual(remote_catalog._async_job_visibility_timeout_s(), 68.0)
            self.assertEqual(remote_catalog._async_attestation_visibility_timeout_s(), 30.0)

            remote_state.config.long_term_memory_remote_flush_timeout_s = 5.0
            self.assertEqual(remote_catalog._remote_async_job_timeout_s(), 15.0)
            self.assertEqual(remote_catalog._async_job_visibility_timeout_s(), 23.0)
            self.assertEqual(remote_catalog._async_attestation_visibility_timeout_s(), 30.0)

    def test_remote_catalog_bulk_write_raises_when_async_readback_never_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.config.project_root = str(project_root)
            remote_state.config.long_term_memory_remote_retry_attempts = 2
            remote_state.config.long_term_memory_remote_retry_backoff_s = 0.0
            remote_state.client = _AsyncBulkNeverVisibleClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=project_root / "state" / "chonkydb", remote_state=remote_state)
            object_fact = LongTermMemoryObjectV1(
                memory_id="fact:jam_preference_old",
                kind="fact",
                summary="Deine Lieblingsmarmelade ist Erdbeermarmelade.",
                source=_source(),
                status="active",
                confidence=0.94,
                slot_key="preference:breakfast:jam",
                value_key="strawberry",
            )

            with self.assertRaises(LongTermRemoteUnavailableError) as raised:
                store.write_snapshot(objects=(object_fact,), conflicts=(), archived_objects=())

            events = TwinrOpsEventStore.from_project_root(project_root).tail(limit=5)

        self.assertIn("Failed to persist fine-grained remote long-term memory items", str(raised.exception))
        self.assertEqual(remote_state.client.bulk_execution_modes, ["async"])
        self.assertTrue(events)
        self.assertEqual(events[-1]["event"], "longterm_remote_write_failed")

    def test_remote_primary_store_reuses_unchanged_documents_without_item_readback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.config.project_root = str(project_root)
            remote_state.client = _ExistingItemFetchFailingChonkyClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=project_root / "state" / "chonkydb", remote_state=remote_state)
            objects = (
                LongTermMemoryObjectV1(
                    memory_id="fact:jam_preference",
                    kind="fact",
                    summary="Du magst Aprikosenmarmelade.",
                    source=_source(),
                    status="active",
                    confidence=0.95,
                    slot_key="preference:breakfast:jam",
                    value_key="apricot",
                ),
            )

            store.write_snapshot(objects=objects, conflicts=(), archived_objects=())
            first_record_count = _count_remote_records_with_schema(remote_state.client, "twinr_memory_object_record_v2")

            store.write_snapshot(objects=objects, conflicts=(), archived_objects=())
            second_record_count = _count_remote_records_with_schema(remote_state.client, "twinr_memory_object_record_v2")
            events = TwinrOpsEventStore.from_project_root(project_root).tail(limit=5)

        self.assertEqual(first_record_count, 1)
        self.assertEqual(second_record_count, 1)
        self.assertFalse(any(event.get("event") == "longterm_remote_read_failed" for event in events))

    def test_ensure_remote_snapshots_seeds_empty_remote_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)

            ensured = store.ensure_remote_snapshots()

        self.assertEqual(set(ensured), {"objects", "conflicts", "archive"})
        self.assertEqual(remote_state.snapshots["conflicts"]["schema"], "twinr_memory_conflict_store")
        self.assertEqual(remote_state.snapshots["archive"]["schema"], "twinr_memory_archive_store")
        self.assertEqual(remote_state.snapshots["objects"]["schema"], "twinr_memory_object_catalog_v3")
        self.assertEqual(remote_state.snapshots["conflicts"]["version"], 1)
        self.assertEqual(remote_state.snapshots["archive"]["version"], 1)
        self.assertEqual(remote_state.snapshots["objects"]["version"], 3)
        self.assertEqual(remote_state.snapshots["conflicts"]["conflicts"], [])
        self.assertEqual(remote_state.snapshots["archive"]["objects"], [])
        self.assertEqual(remote_state.snapshots["objects"]["items_count"], 0)
        self.assertEqual(remote_state.snapshots["objects"]["segments"], [])
        self.assertNotIn("written_at", remote_state.snapshots["conflicts"])
        self.assertNotIn("written_at", remote_state.snapshots["archive"])
        self.assertIn("written_at", remote_state.snapshots["objects"])

    def test_ensure_remote_snapshots_bootstraps_empty_remote_documents_for_fresh_required_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.required = True
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)

            ensured = store.ensure_remote_snapshots()

        self.assertEqual(set(ensured), {"objects", "conflicts", "archive"})
        self.assertEqual(remote_state.snapshots["objects"]["schema"], "twinr_memory_object_catalog_v3")
        self.assertEqual(remote_state.snapshots["objects"]["items_count"], 0)
        self.assertEqual(remote_state.snapshots["objects"]["segments"], [])
        self.assertEqual(remote_state.snapshots["conflicts"]["schema"], "twinr_memory_conflict_store")
        self.assertEqual(remote_state.snapshots["conflicts"]["conflicts"], [])
        self.assertEqual(remote_state.snapshots["archive"]["schema"], "twinr_memory_archive_store")
        self.assertEqual(remote_state.snapshots["archive"]["objects"], [])
        self.assertFalse(store.objects_path.exists())

    def test_ensure_remote_snapshots_serializes_required_remote_reads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _SerialEnsureRemoteState()
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)

            ensured = store.ensure_remote_snapshots()

        collapsed_load_calls: list[str] = []
        for snapshot_kind in remote_state.load_calls:
            if not collapsed_load_calls or collapsed_load_calls[-1] != snapshot_kind:
                collapsed_load_calls.append(snapshot_kind)
        self.assertEqual(set(ensured), {"objects", "conflicts", "archive"})
        self.assertEqual(collapsed_load_calls, ["objects", "conflicts", "archive"])
        self.assertEqual(remote_state.max_concurrent_loads, 1)

    def test_ensure_remote_snapshots_skips_catalog_hydration_when_probe_already_proves_remote_heads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _ProbeOnlyCatalogRemoteState()
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)

            ensured = store.ensure_remote_snapshots()

        self.assertEqual(ensured, ())
        self.assertEqual(remote_state.load_calls, [])
        self.assertEqual(
            remote_state.probe_calls,
            [
                {"snapshot_kind": "objects", "prefer_cached_document_id": True, "prefer_metadata_only": True, "fast_fail": True},
                {"snapshot_kind": "conflicts", "prefer_cached_document_id": True, "prefer_metadata_only": True, "fast_fail": True},
                {"snapshot_kind": "archive", "prefer_cached_document_id": True, "prefer_metadata_only": True, "fast_fail": True},
            ],
        )

    def test_ensure_remote_snapshots_still_fails_closed_when_required_remote_write_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FailingRemoteSaveState()
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)

            with self.assertRaises(LongTermRemoteUnavailableError):
                store.ensure_remote_snapshots()

    def test_remote_primary_store_persists_one_document_per_object(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            objects = tuple(
                LongTermMemoryObjectV1(
                    memory_id=f"fact:{index}",
                    kind="fact",
                    summary=f"Fact number {index} " + ("x" * 400),
                    source=_source(),
                    status="active",
                    confidence=0.9,
                )
                for index in range(4)
            )

            store.write_snapshot(objects=objects)
            loaded = store.load_objects()

        self.assertEqual(len(loaded), 4)
        self.assertEqual(remote_state.snapshots["objects"]["items_count"], 4)
        self.assertEqual(_count_remote_records_with_schema(remote_state.client, "twinr_memory_object_record_v2"), 4)
        self.assertEqual(_count_remote_records_with_schema(remote_state.client, "twinr_memory_object_catalog_segment_v1"), 1)

    def test_remote_primary_store_batches_fine_grained_remote_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.client = _FakeChonkyClient(max_items_per_bulk=2)
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            remote_state.config.long_term_memory_migration_batch_size = 2
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            objects = tuple(
                LongTermMemoryObjectV1(
                    memory_id=f"fact:{index}",
                    kind="fact",
                    summary=f"Fact number {index}",
                    source=_source(),
                    status="active",
                    confidence=0.9,
                )
                for index in range(5)
            )

            store.write_snapshot(objects=objects)

        self.assertEqual(remote_state.snapshots["objects"]["items_count"], 5)
        self.assertEqual(_count_remote_records_with_schema(remote_state.client, "twinr_memory_object_record_v2"), 5)
        self.assertEqual(_count_bulk_calls_with_schema(remote_state.client, "twinr_memory_object_record_v2"), 3)

    def test_remote_primary_store_splits_fine_grained_remote_writes_by_request_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.client = _FakeChonkyClient(max_request_bytes=4000)
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            remote_state.config.long_term_memory_migration_batch_size = 64
            remote_state.config.long_term_memory_remote_bulk_request_max_bytes = 4000
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            objects = tuple(
                LongTermMemoryObjectV1(
                    memory_id=f"fact:{index}",
                    kind="fact",
                    summary=f"Fact number {index} " + ("x" * 250),
                    source=_source(),
                    status="active",
                    confidence=0.9,
                )
                for index in range(5)
            )

            store.write_snapshot(objects=objects)

        self.assertEqual(remote_state.snapshots["objects"]["items_count"], 5)
        self.assertEqual(_count_remote_records_with_schema(remote_state.client, "twinr_memory_object_record_v2"), 5)
        self.assertGreater(_count_bulk_calls_with_schema(remote_state.client, "twinr_memory_object_record_v2"), 1)
        self.assertTrue(remote_state.client.bulk_request_bytes)
        self.assertLessEqual(max(remote_state.client.bulk_request_bytes), 4000)

    def test_remote_primary_store_round_trips_through_live_document_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.client = _LiveShapeChonkyClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            objects = (
                LongTermMemoryObjectV1(
                    memory_id="fact:1",
                    kind="fact",
                    summary="Fact number 1",
                    source=_source(),
                    status="active",
                    confidence=0.9,
                    details="Stored as a real ChonkyDB document.",
                    slot_key="fact:1",
                    value_key="fact_1",
                    attributes={"favorite_color": "blue"},
                ),
            )

            store.write_snapshot(objects=objects)
            loaded = store.load_objects()

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].memory_id, "fact:1")
        self.assertEqual(loaded[0].details, "Stored as a real ChonkyDB document.")
        self.assertEqual(dict(loaded[0].attributes or {}).get("favorite_color"), "blue")

    def test_remote_primary_store_assembles_catalog_snapshots_via_retrieve_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.client = _NoItemFetchChonkyClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            store.write_snapshot(
                objects=tuple(
                    LongTermMemoryObjectV1(
                        memory_id=f"fact:{index}",
                        kind="fact",
                        summary=f"Fact number {index}",
                        source=_source(),
                        status="active",
                        confidence=0.9,
                    )
                    for index in range(4)
                )
            )

            self.assertEqual(store.ensure_remote_snapshots(), ())
            loaded = store.load_objects()

        self.assertEqual(len(loaded), 4)
        self.assertGreater(remote_state.client.retrieve_calls, 0)
        self.assertEqual(remote_state.client.item_fetch_attempts, 0)

    def test_remote_primary_store_loads_rich_catalog_entries_without_retrieve(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.client = _NoItemFetchChonkyClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            store.write_snapshot(
                objects=tuple(
                    LongTermMemoryObjectV1(
                        memory_id=f"fact:{index}",
                        kind="fact",
                        summary=f"Fact number {index}",
                        source=_source(),
                        status="active",
                        confidence=0.9,
                        slot_key=f"fact:{index}",
                        value_key=f"fact_{index}",
                    )
                    for index in range(4)
                )
            )

            self.assertEqual(store.ensure_remote_snapshots(), ())

        self.assertEqual(remote_state.client.retrieve_calls, 0)
        self.assertEqual(remote_state.client.item_fetch_attempts, 0)

    def test_remote_primary_store_recovers_metadata_only_live_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.client = _MetadataOnlyLiveShapeChonkyClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            objects = (
                LongTermMemoryObjectV1(
                    memory_id="pattern:button:green:start_listening:evening",
                    kind="pattern",
                    summary="The green button is often used in the evening.",
                    source=_source(),
                    status="active",
                    confidence=0.9,
                    details="Recovered from a metadata-only remote document.",
                    slot_key="pattern:button:green:start_listening:evening",
                    value_key="start_listening_evening",
                ),
            )

            store.write_snapshot(objects=objects)

            self.assertEqual(store.ensure_remote_snapshots(), ())
            loaded = store.load_objects()

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].memory_id, "pattern:button:green:start_listening:evening")
        self.assertEqual(loaded[0].summary, "The green button is often used in the evening.")
        self.assertEqual(loaded[0].source.source_type, "legacy_remote_catalog_metadata")
        self.assertTrue(dict(loaded[0].attributes or {}).get("legacy_remote_catalog_metadata_only"))

    def test_remote_primary_store_keeps_sparse_catalog_recovery_read_only_during_required_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.config.long_term_memory_migration_enabled = True
            remote_state.client = _MetadataOnlyLiveShapeChonkyClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            objects = (
                LongTermMemoryObjectV1(
                    memory_id="pattern:button:green:start_listening:evening",
                    kind="pattern",
                    summary="The green button is often used in the evening.",
                    source=_source(),
                    status="active",
                    confidence=0.9,
                    details="Recovered from a metadata-only remote document.",
                    slot_key="pattern:button:green:start_listening:evening",
                    value_key="start_listening_evening",
                ),
            )

            store.write_snapshot(objects=objects)
            _sparsify_catalog_segment_entries(remote_state.client, remote_state.snapshots["objects"])
            bulk_calls_before_probe = remote_state.client.bulk_calls
            remote_state.client.retrieve_calls = 0

            self.assertEqual(store.ensure_remote_snapshots(), ())
            first_retrieve_calls = remote_state.client.retrieve_calls
            remote_state.client.retrieve_calls = 0
            self.assertEqual(store.ensure_remote_snapshots(), ())
            second_retrieve_calls = remote_state.client.retrieve_calls
            sparse_entries = store._remote_catalog.load_catalog_entries(snapshot_kind="objects")

        self.assertGreater(first_retrieve_calls, 0)
        self.assertGreater(second_retrieve_calls, 0)
        self.assertEqual(remote_state.client.bulk_calls, bulk_calls_before_probe)
        self.assertIsNone(sparse_entries[0].metadata.get("summary"))

    def test_remote_primary_store_reuses_unchanged_documents_on_subsequent_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.client = _CorruptingRewriteChonkyClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            first_objects = (
                LongTermMemoryObjectV1(
                    memory_id="episode:1",
                    kind="episode",
                    summary="First episodic turn.",
                    details="The user mentioned tomato soup.",
                    source=_source(),
                    status="active",
                    confidence=1.0,
                    slot_key="episode:1",
                    value_key="episode_1",
                    attributes={"raw_transcript": "Meine Lieblingssuppe ist Tomatensuppe."},
                ),
                LongTermMemoryObjectV1(
                    memory_id="fact:soup",
                    kind="fact",
                    summary="Favorite soup is tomato soup.",
                    details="Direct user preference.",
                    source=_source(),
                    status="active",
                    confidence=0.98,
                    slot_key="favorite_soup",
                    value_key="tomato_soup",
                ),
            )
            second_objects = first_objects + (
                LongTermMemoryObjectV1(
                    memory_id="episode:2",
                    kind="episode",
                    summary="Second episodic turn.",
                    details="The user prefers Sunday midday.",
                    source=_source(),
                    status="active",
                    confidence=1.0,
                    slot_key="episode:2",
                    value_key="episode_2",
                ),
                LongTermMemoryObjectV1(
                    memory_id="fact:time",
                    kind="fact",
                    summary="Preferred soup time is Sunday midday.",
                    details="Direct user preference.",
                    source=_source(),
                    status="candidate",
                    confidence=0.74,
                    slot_key="favorite_time",
                    value_key="sunday_midday",
                ),
            )

            store.write_snapshot(objects=first_objects)
            first_entries = {
                entry.item_id: entry.document_id
                for entry in store._remote_catalog.load_catalog_entries(snapshot_kind="objects")
            }

            store.write_snapshot(objects=second_objects)
            loaded = store.load_objects()
            second_entries = {
                entry.item_id: entry.document_id
                for entry in store._remote_catalog.load_catalog_entries(snapshot_kind="objects")
            }

        self.assertEqual(set(item.memory_id for item in loaded), {"episode:1", "episode:2", "fact:soup", "fact:time"})
        self.assertEqual(_count_remote_records_with_schema(remote_state.client, "twinr_memory_object_record_v2"), 4)
        self.assertEqual(second_entries["episode:1"], first_entries["episode:1"])
        self.assertEqual(second_entries["fact:soup"], first_entries["fact:soup"])

    def test_remote_primary_store_does_not_fallback_to_local_objects_when_remote_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "state" / "chonkydb"
            base_path.mkdir(parents=True, exist_ok=True)
            objects_path = base_path / "twinr_memory_objects_v1.json"
            objects_path.write_text(
                json.dumps({"schema": "twinr_memory_object_store", "version": 1, "objects": [{"memory_id": "fact:local"}]}),
                encoding="utf-8",
            )
            store = LongTermStructuredStore(base_path=base_path, remote_state=_FailingRemoteState())

            with self.assertRaises(LongTermRemoteUnavailableError):
                store.load_objects()

    def test_remote_primary_store_selects_relevant_objects_via_fine_grained_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:janina_spouse",
                        kind="relationship_fact",
                        summary="Janina is the user's wife.",
                        source=_source(),
                        status="active",
                        confidence=0.98,
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:tea_preference",
                        kind="preference_fact",
                        summary="The user likes black tea.",
                        source=_source(),
                        status="active",
                        confidence=0.9,
                    ),
                )
            )

            relevant = store.select_relevant_objects(query_text="Janina", limit=1)

        self.assertEqual(tuple(item.memory_id for item in relevant), ("fact:janina_spouse",))

    def test_remote_primary_store_reuses_remote_catalog_search_and_item_reads_when_cache_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.config.long_term_memory_remote_read_cache_ttl_s = 60.0
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:janina_spouse",
                        kind="relationship_fact",
                        summary="Janina is the user's wife.",
                        source=_source(),
                        status="active",
                        confidence=0.98,
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:tea_preference",
                        kind="preference_fact",
                        summary="The user likes black tea.",
                        source=_source(),
                        status="active",
                        confidence=0.9,
                    ),
                )
            )
            remote_state.client.retrieve_calls = 0

            first = store.select_relevant_objects(query_text="Janina", limit=1)
            retrieve_calls_after_first = remote_state.client.retrieve_calls
            second = store.select_relevant_objects(query_text="Janina", limit=1)

        self.assertEqual(tuple(item.memory_id for item in first), ("fact:janina_spouse",))
        self.assertEqual(tuple(item.memory_id for item in second), ("fact:janina_spouse",))
        self.assertGreater(retrieve_calls_after_first, 0)
        self.assertEqual(remote_state.client.retrieve_calls, retrieve_calls_after_first)

    def test_remote_primary_store_prefers_one_shot_topk_records_without_extra_document_fetches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:janina_spouse",
                        kind="relationship_fact",
                        summary="Janina is the user's wife.",
                        source=_source(),
                        status="active",
                        confidence=0.98,
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:tea_preference",
                        kind="preference_fact",
                        summary="The user likes black tea.",
                        source=_source(),
                        status="active",
                        confidence=0.9,
                    ),
                )
            )
            remote_catalog = store._remote_catalog
            self.assertIsNotNone(remote_catalog)
            entries = remote_catalog.load_catalog_entries(snapshot_kind="objects")
            janina_entries = tuple(entry for entry in entries if entry.item_id == "fact:janina_spouse")
            remote_state.client.supports_topk_records = True
            remote_state.client.fetch_full_document_calls = 0
            remote_state.client.topk_records_calls = 0

            selected = remote_catalog._search_remote_candidates(
                snapshot_kind="objects",
                read_client=remote_state.client,
                query_text="Janina",
                result_limit=1,
                allowed_doc_ids=tuple(entry.document_id for entry in entries if entry.document_id),
                catalog_entry_count=len(entries),
            )
            payloads = remote_catalog._load_item_payloads_from_entries(
                snapshot_kind="objects",
                entries=janina_entries,
            )

        self.assertEqual(len(selected), 1)
        self.assertEqual(payloads[0]["memory_id"], "fact:janina_spouse")
        self.assertGreater(remote_state.client.topk_records_calls, 1)
        self.assertEqual(remote_state.client.fetch_full_document_calls, 0)

    def test_remote_primary_store_selects_relevant_objects_via_direct_scope_search_without_catalog_hydration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="episode:doctor",
                        kind="episode",
                        summary="Janina had an eye doctor appointment yesterday.",
                        source=_source(),
                        status="active",
                        confidence=0.95,
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:janina_spouse",
                        kind="relationship_fact",
                        summary="Janina is the user's wife.",
                        source=_source(),
                        status="active",
                        confidence=0.98,
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:tea_preference",
                        kind="preference_fact",
                        summary="The user likes black tea.",
                        source=_source(),
                        status="active",
                        confidence=0.9,
                    ),
                )
            )
            remote_state.client.supports_topk_records = True
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None

            def _fail_catalog_hydration(*args, **kwargs):
                del args
                del kwargs
                raise AssertionError("full catalog hydration should not run for direct scope search")

            remote_catalog.load_catalog_entries = _fail_catalog_hydration  # type: ignore[method-assign]
            remote_catalog.catalog_available = _fail_catalog_hydration  # type: ignore[method-assign]

            relevant = store.select_relevant_objects(query_text="Janina", limit=1)

        self.assertEqual(tuple(item.memory_id for item in relevant), ("fact:janina_spouse",))
        payload = remote_state.client.topk_records_payloads[-1]
        self.assertEqual(payload["namespace"], "test-namespace")
        self.assertEqual(payload["scope_ref"], "longterm:objects:current")
        self.assertEqual(remote_state.client.fetch_full_document_calls, 0)

    def test_remote_primary_store_skips_remote_recent_fallback_after_query_scope_miss(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="episode:doctor",
                        kind="episode",
                        summary="Janina had an eye doctor appointment yesterday.",
                        source=_source(),
                        status="active",
                        confidence=0.95,
                    ),
                )
            )
            remote_state.client.supports_topk_records = True
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None

            def _fail_catalog_hydration(*args, **kwargs):
                del args
                del kwargs
                raise AssertionError("remote query miss must not hydrate the full episodic catalog")

            remote_catalog.load_catalog_entries = _fail_catalog_hydration  # type: ignore[method-assign]
            remote_catalog.catalog_available = _fail_catalog_hydration  # type: ignore[method-assign]

            relevant = store.select_relevant_episodic_objects(
                query_text="Hallo",
                limit=2,
                fallback_limit=2,
                require_query_match=False,
            )

        self.assertEqual(relevant, ())
        payload = remote_state.client.topk_records_payloads[-1]
        self.assertEqual(payload["scope_ref"], "longterm:objects:current")

    def test_search_current_item_payloads_oversamples_initial_scope_window_for_client_filters(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="episode:doctor",
                        kind="episode",
                        summary="Janina had an eye doctor appointment yesterday.",
                        source=_source(),
                        status="active",
                        confidence=0.95,
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:janina_spouse",
                        kind="relationship_fact",
                        summary="Janina is the user's wife.",
                        source=_source(),
                        status="active",
                        confidence=0.98,
                    ),
                )
            )
            remote_state.client.supports_topk_records = True
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None

            payloads = remote_catalog.search_current_item_payloads(
                snapshot_kind="objects",
                query_text="Janina",
                limit=1,
                eligible=lambda entry: entry.metadata.get("kind") != "episode",
            )

        self.assertEqual(tuple(payload["memory_id"] for payload in payloads), ("fact:janina_spouse",))
        self.assertEqual(remote_state.client.topk_records_payloads[0]["result_limit"], 16)

    def test_search_current_item_payloads_returns_none_when_scope_topk_times_out(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.client = _TimeoutingScopeTopKClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:janina_spouse",
                        kind="relationship_fact",
                        summary="Janina is the user's wife.",
                        source=_source(),
                        status="active",
                        confidence=0.98,
                    ),
                )
            )
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None

            payloads = remote_catalog.search_current_item_payloads(
                snapshot_kind="objects",
                query_text="Janina",
                limit=1,
            )

        self.assertIsNone(payloads)
        self.assertEqual(remote_state.client.retrieve_calls, 0)

    def test_select_fast_topic_objects_uses_one_shot_scope_search_with_bounded_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.client.supports_topk_records = True
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:janina_spouse",
                        kind="relationship_fact",
                        summary="Janina is the user's wife.",
                        source=_source(),
                        status="active",
                        confidence=0.98,
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="episode:doctor",
                        kind="episode",
                        summary="Janina had an eye doctor appointment yesterday.",
                        source=_source(),
                        status="active",
                        confidence=0.95,
                    ),
                )
            )
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None

            def _fail_catalog_hydration(*args, **kwargs):
                del args
                del kwargs
                raise AssertionError("fast topic lookup must not hydrate the full catalog")

            remote_catalog.load_catalog_entries = _fail_catalog_hydration  # type: ignore[method-assign]
            remote_catalog.catalog_available = _fail_catalog_hydration  # type: ignore[method-assign]

            relevant = store.select_fast_topic_objects(
                query_text="Janina",
                limit=2,
                timeout_s=0.45,
            )

        self.assertEqual(
            {item.memory_id for item in relevant},
            {"fact:janina_spouse", "episode:doctor"},
        )
        self.assertEqual(len(relevant), 2)
        payload = remote_state.client.topk_records_payloads[-1]
        self.assertEqual(payload["namespace"], "test-namespace")
        self.assertEqual(payload["scope_ref"], "longterm:objects:current")
        self.assertEqual(payload["result_limit"], 2)
        self.assertEqual(payload["timeout_seconds"], 0.45)
        self.assertEqual(remote_state.client.topk_records_calls, 1)
        self.assertEqual(remote_state.client.fetch_full_document_calls, 0)

    def test_required_remote_reads_prefer_same_process_snapshot_after_remote_visibility_lag(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _LaggingSnapshotVisibilityRemoteState()
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            old_object = LongTermMemoryObjectV1(
                memory_id="fact:jam_preference_new",
                kind="fact",
                summary="Inzwischen magst du lieber Aprikosenmarmelade.",
                details="Neuere Vorliebe fuer das Fruehstueck.",
                source=_source(),
                status="uncertain",
                confidence=0.95,
                slot_key="preference:breakfast:jam",
                value_key="apricot",
            )
            confirmed_object = old_object.with_updates(
                status="active",
                confirmed_by_user=True,
                confidence=0.99,
            )

            store.write_snapshot(objects=(old_object,))
            store.write_snapshot(objects=(confirmed_object,))

            remote_payload = store._load_remote_snapshot_payload(snapshot_kind="objects")
            loaded = store.load_objects()

            remote_state.promote_pending("objects")
            loaded_after_remote_catchup = store.load_objects()

        assert remote_payload is not None
        remote_objects = tuple(remote_payload.get("objects") or ())
        self.assertEqual(remote_objects[0]["status"], "uncertain")
        self.assertFalse(remote_objects[0]["confirmed_by_user"])
        self.assertEqual(tuple(item.memory_id for item in loaded), ("fact:jam_preference_new",))
        self.assertEqual(loaded[0].status, "active")
        self.assertTrue(loaded[0].confirmed_by_user)
        self.assertEqual(loaded_after_remote_catchup[0].status, "active")
        self.assertTrue(loaded_after_remote_catchup[0].confirmed_by_user)

    def test_select_relevant_context_objects_prefer_same_process_snapshot_after_remote_visibility_lag(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _LaggingSnapshotVisibilityRemoteState()
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_generic",
                        kind="fact",
                        summary="User usually likes some jam on bread at breakfast.",
                        details='User said: "Ich mag beim Frühstück meistens etwas Marmelade auf dem Brot."',
                        source=_source(),
                        status="active",
                        confidence=0.84,
                        slot_key="fact:user:breakfast:jam",
                        value_key="jam_on_bread_at_breakfast",
                        attributes={"fact_type": "general", "memory_domain": "general"},
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_preference_old",
                        kind="fact",
                        summary="Deine Lieblingsmarmelade ist Erdbeermarmelade.",
                        details="Aeltere Vorliebe fuer das Fruehstueck.",
                        source=_source(),
                        status="active",
                        confidence=0.94,
                        slot_key="preference:breakfast:jam",
                        value_key="strawberry",
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_preference_new",
                        kind="fact",
                        summary="Inzwischen magst du lieber Aprikosenmarmelade.",
                        details="Neuere Vorliebe fuer das Fruehstueck.",
                        source=_source(),
                        status="uncertain",
                        confidence=0.95,
                        slot_key="preference:breakfast:jam",
                        value_key="apricot",
                    ),
                ),
            )
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_generic",
                        kind="fact",
                        summary="User usually likes some jam on bread at breakfast.",
                        details='User said: "Ich mag beim Frühstück meistens etwas Marmelade auf dem Brot."',
                        source=_source(),
                        status="active",
                        confidence=0.84,
                        slot_key="fact:user:breakfast:jam",
                        value_key="jam_on_bread_at_breakfast",
                        attributes={"fact_type": "general", "memory_domain": "general"},
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_preference_old",
                        kind="fact",
                        summary="Deine Lieblingsmarmelade ist Erdbeermarmelade.",
                        details="Aeltere Vorliebe fuer das Fruehstueck.",
                        source=_source(),
                        status="superseded",
                        confidence=0.94,
                        slot_key="preference:breakfast:jam",
                        value_key="strawberry",
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_preference_new",
                        kind="fact",
                        summary="Inzwischen magst du lieber Aprikosenmarmelade.",
                        details="Neuere Vorliebe fuer das Fruehstueck.",
                        source=_source(),
                        status="active",
                        confirmed_by_user=True,
                        confidence=0.99,
                        slot_key="preference:breakfast:jam",
                        value_key="apricot",
                    ),
                ),
            )

            episodic, durable = store.select_relevant_context_objects(
                query_text="Welche Marmelade ist jetzt als bestaetigt gespeichert?",
                episodic_limit=3,
                durable_limit=3,
            )

        self.assertEqual(tuple(item.memory_id for item in episodic), ())
        self.assertEqual(durable[0].memory_id, "fact:jam_preference_new")
        self.assertEqual(durable[0].status, "active")
        self.assertTrue(durable[0].confirmed_by_user)

    def test_select_fast_topic_objects_prefer_same_process_snapshot_after_remote_visibility_lag(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _LaggingSnapshotVisibilityRemoteState()
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_generic",
                        kind="fact",
                        summary="User usually likes some jam on bread at breakfast.",
                        details='User said: "Ich mag beim Frühstück meistens etwas Marmelade auf dem Brot."',
                        source=_source(),
                        status="active",
                        confidence=0.84,
                        slot_key="fact:user:breakfast:jam",
                        value_key="jam_on_bread_at_breakfast",
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_preference_new",
                        kind="fact",
                        summary="Inzwischen magst du lieber Aprikosenmarmelade.",
                        details="Neuere Vorliebe fuer das Fruehstueck.",
                        source=_source(),
                        status="uncertain",
                        confidence=0.95,
                        slot_key="preference:breakfast:jam",
                        value_key="apricot",
                    ),
                ),
            )
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_generic",
                        kind="fact",
                        summary="User usually likes some jam on bread at breakfast.",
                        details='User said: "Ich mag beim Frühstück meistens etwas Marmelade auf dem Brot."',
                        source=_source(),
                        status="active",
                        confidence=0.84,
                        slot_key="fact:user:breakfast:jam",
                        value_key="jam_on_bread_at_breakfast",
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_preference_new",
                        kind="fact",
                        summary="Inzwischen magst du lieber Aprikosenmarmelade.",
                        details="Neuere Vorliebe fuer das Fruehstueck.",
                        source=_source(),
                        status="active",
                        confirmed_by_user=True,
                        confidence=0.99,
                        slot_key="preference:breakfast:jam",
                        value_key="apricot",
                    ),
                ),
            )

            relevant = store.select_fast_topic_objects(
                query_text="Welche Marmeladensorte ist aktuell gespeichert?",
                limit=2,
                timeout_s=0.45,
            )

        self.assertEqual(relevant[0].memory_id, "fact:jam_preference_new")
        self.assertEqual(relevant[0].status, "active")
        self.assertTrue(relevant[0].confirmed_by_user)

    def test_select_fast_topic_objects_raises_precise_remote_read_failure_for_scope_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.config.project_root = str(project_root)
            remote_state.client = _TimeoutingScopeTopKClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=project_root / "state" / "chonkydb", remote_state=remote_state)

            with self.assertRaises(LongTermRemoteReadFailedError) as raised:
                store.select_fast_topic_objects(
                    query_text="Janina",
                    limit=2,
                    timeout_s=0.45,
                )

            events = TwinrOpsEventStore.from_project_root(project_root).tail(limit=8)

        details = dict(raised.exception.details)
        self.assertEqual(details["operation"], "fast_topic_topk_search")
        self.assertEqual(details["classification"], "timeout")
        self.assertEqual(details["timeout_reason"], "read_operation_timed_out")
        self.assertEqual(details["request_path"], "/v1/external/retrieve/topk_records")
        self.assertEqual(details["request_timeout_s"], 0.45)
        self.assertEqual(details["scope_ref"], "longterm:objects:current")
        self.assertEqual(details["retry_attempts_configured"], 1)
        self.assertEqual(details["retry_backoff_s"], 0.0)
        self.assertEqual(details["retry_mode"], "disabled_for_fast_topic_budget")
        self.assertFalse(details["retry_enabled"])
        self.assertEqual(details["attempt_index"], 1)
        self.assertEqual(details["attempt_count"], 1)
        failed_event = next(
            event
            for event in events
            if event.get("event") == "longterm_remote_read_failed"
            and dict(event.get("data") or {}).get("operation") == "fast_topic_topk_search"
        )
        failed_data = dict(failed_event["data"])
        self.assertEqual(failed_data["classification"], "timeout")
        self.assertEqual(failed_data["timeout_reason"], "read_operation_timed_out")
        self.assertEqual(failed_data["retry_mode"], "disabled_for_fast_topic_budget")

    def test_select_fast_topic_objects_records_http_status_and_problem_detail_for_503(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.config.project_root = str(project_root)
            remote_state.client = _Http503ScopeTopKClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=project_root / "state" / "chonkydb", remote_state=remote_state)

            with self.assertRaises(LongTermRemoteReadFailedError) as raised:
                store.select_fast_topic_objects(
                    query_text="Thermoskanne",
                    limit=1,
                    timeout_s=0.6,
                )

            events = TwinrOpsEventStore.from_project_root(project_root).tail(limit=8)

        details = dict(raised.exception.details)
        self.assertEqual(details["operation"], "fast_topic_topk_search")
        self.assertEqual(details["classification"], "backend_http_error")
        self.assertEqual(details["status_code"], 503)
        self.assertEqual(details["response_status_code"], 503)
        self.assertEqual(details["response_detail"], "payload_sync_bulk_busy")
        self.assertEqual(details["response_error"], "payload_sync_bulk_busy")
        self.assertEqual(details["response_error_type"], "ServerBusy")
        failed_event = next(
            event
            for event in events
            if event.get("event") == "longterm_remote_read_failed"
            and dict(event.get("data") or {}).get("operation") == "fast_topic_topk_search"
        )
        failed_data = dict(failed_event["data"])
        self.assertEqual(failed_data["status_code"], 503)
        self.assertEqual(failed_data["response_error_type"], "ServerBusy")

    def test_search_current_item_payloads_fast_rescues_timeout_with_current_catalog_projection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.client = _AlwaysTimeoutingScopeTopKClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:thermos_location_old",
                        kind="fact",
                        summary="Die rote Thermoskanne steht im Flurschrank.",
                        details="Aktueller Ort der roten Thermoskanne.",
                        source=_source(),
                        status="active",
                        confidence=0.99,
                        confirmed_by_user=True,
                        slot_key="object:red_thermos:location",
                        value_key="hallway_cupboard",
                    ),
                ),
            )
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None

            payloads = remote_catalog.search_current_item_payloads_fast(
                snapshot_kind="objects",
                query_text="Thermoskanne",
                limit=1,
                timeout_s=0.45,
            )

        self.assertEqual(tuple(payload["memory_id"] for payload in payloads), ("fact:thermos_location_old",))

    def test_search_current_item_payloads_fast_rescues_503_with_current_catalog_projection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.client = _Http503ScopeTopKClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:thermos_location_old",
                        kind="fact",
                        summary="Die rote Thermoskanne steht im Flurschrank.",
                        details="Aktueller Ort der roten Thermoskanne.",
                        source=_source(),
                        status="active",
                        confidence=0.99,
                        confirmed_by_user=True,
                        slot_key="object:red_thermos:location",
                        value_key="hallway_cupboard",
                    ),
                ),
            )
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None

            payloads = remote_catalog.search_current_item_payloads_fast(
                snapshot_kind="objects",
                query_text="Thermoskanne",
                limit=1,
                timeout_s=0.6,
            )

        self.assertEqual(tuple(payload["memory_id"] for payload in payloads), ("fact:thermos_location_old",))

    def test_search_current_item_payloads_prefers_remote_topk_even_with_cached_catalog_projection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.client.supports_topk_records = True
            remote_state.config.long_term_memory_remote_read_cache_ttl_s = 60.0
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="episode:doctor",
                        kind="episode",
                        summary="Janina had an eye doctor appointment yesterday.",
                        source=_source(),
                        status="active",
                        confidence=0.95,
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:janina_spouse",
                        kind="relationship_fact",
                        summary="Janina is the user's wife.",
                        source=_source(),
                        status="active",
                        confidence=0.98,
                    ),
                )
            )
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None
            self.assertGreater(len(remote_catalog.load_catalog_entries(snapshot_kind="objects")), 0)

            def _fail_local(*args, **kwargs):
                del args
                del kwargs
                raise AssertionError("current-scope reads must prefer remote topk before the local cached selector")

            remote_catalog._local_search_catalog_entries = _fail_local  # type: ignore[method-assign]

            payloads = remote_catalog.search_current_item_payloads(
                snapshot_kind="objects",
                query_text="Janina",
                limit=2,
            )

        self.assertEqual(tuple(payload["memory_id"] for payload in payloads), ("episode:doctor", "fact:janina_spouse"))
        self.assertEqual(remote_state.client.topk_records_calls, 1)

    def test_search_current_item_payloads_falls_back_to_current_catalog_after_scope_false_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.client = _EmptyScopeTopKClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:thermos_location_old",
                        kind="fact",
                        summary="Früher stand die rote Thermoskanne im Flurschrank.",
                        details="Historische Ortsangabe zur roten Thermoskanne.",
                        source=_source(),
                        status="active",
                        confidence=0.99,
                        confirmed_by_user=True,
                        slot_key="object:red_thermos:location",
                        value_key="hallway_cupboard",
                    ),
                ),
            )
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None

            payloads = remote_catalog.search_current_item_payloads(
                snapshot_kind="objects",
                query_text="Thermoskanne",
                limit=2,
                allow_catalog_fallback=True,
            )

        self.assertEqual(tuple(payload["memory_id"] for payload in payloads), ("fact:thermos_location_old",))
        self.assertEqual(remote_state.client.topk_records_calls, 2)
        self.assertEqual(remote_state.client.retrieve_calls, 1)

    def test_search_current_item_payloads_reconciles_stale_scope_payloads_with_current_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.client = _StaleScopeTopKClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_preference_old",
                        kind="fact",
                        summary="Deine Lieblingsmarmelade ist Erdbeermarmelade.",
                        source=_source(),
                        status="active",
                        confidence=0.94,
                        slot_key="preference:breakfast:jam",
                        value_key="strawberry",
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_preference_new",
                        kind="fact",
                        summary="Inzwischen magst du lieber Aprikosenmarmelade.",
                        source=_source(),
                        status="uncertain",
                        confidence=0.95,
                        slot_key="preference:breakfast:jam",
                        value_key="apricot",
                    ),
                ),
            )
            remote_state.client.snapshot_scope_records()
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_preference_old",
                        kind="fact",
                        summary="Deine Lieblingsmarmelade ist Erdbeermarmelade.",
                        source=_source(),
                        status="superseded",
                        confidence=0.94,
                        slot_key="preference:breakfast:jam",
                        value_key="strawberry",
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_preference_new",
                        kind="fact",
                        summary="Inzwischen magst du lieber Aprikosenmarmelade.",
                        source=_source(),
                        status="active",
                        confidence=0.99,
                        confirmed_by_user=True,
                        slot_key="preference:breakfast:jam",
                        value_key="apricot",
                        attributes={"resolved_by_user": True},
                    ),
                ),
            )
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None

            payloads = remote_catalog.search_current_item_payloads(
                snapshot_kind="objects",
                query_text="Aprikosenmarmelade bestaetigt",
                limit=2,
                allow_catalog_fallback=True,
            )
            payload_by_id = {payload["memory_id"]: payload for payload in payloads}

        self.assertIn("fact:jam_preference_new", payload_by_id)
        self.assertEqual(payload_by_id["fact:jam_preference_new"]["status"], "active")
        self.assertTrue(payload_by_id["fact:jam_preference_new"]["confirmed_by_user"])
        self.assertNotEqual(
            payload_by_id["fact:jam_preference_new"]["status"],
            "uncertain",
        )

    def test_select_relevant_context_objects_uses_one_scope_search_for_both_sections(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="episode:doctor",
                        kind="episode",
                        summary="Janina had an eye doctor appointment yesterday.",
                        source=_source(),
                        status="active",
                        confidence=0.95,
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:janina_spouse",
                        kind="relationship_fact",
                        summary="Janina is the user's wife.",
                        source=_source(),
                        status="active",
                        confidence=0.98,
                    ),
                )
            )
            remote_state.client.supports_topk_records = True

            episodic, durable = store.select_relevant_context_objects(
                query_text="Janina",
                episodic_limit=1,
                durable_limit=1,
            )

        self.assertEqual(tuple(item.memory_id for item in episodic), ("episode:doctor",))
        self.assertEqual(tuple(item.memory_id for item in durable), ("fact:janina_spouse",))
        self.assertEqual(remote_state.client.topk_records_calls, 1)

    def test_select_relevant_context_objects_falls_back_after_scope_false_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.client = _EmptyScopeTopKClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:thermos_location_old",
                        kind="fact",
                        summary="Früher stand die rote Thermoskanne im Flurschrank.",
                        details="Historische Ortsangabe zur roten Thermoskanne.",
                        source=_source(),
                        status="active",
                        confidence=0.99,
                        confirmed_by_user=True,
                        slot_key="object:red_thermos:location",
                        value_key="hallway_cupboard",
                    ),
                ),
            )

            episodic, durable = store.select_relevant_context_objects(
                query_text="Wo stand früher meine rote Thermoskanne?",
                episodic_limit=1,
                durable_limit=1,
            )

        self.assertEqual(tuple(item.memory_id for item in episodic), ())
        self.assertEqual(tuple(item.memory_id for item in durable), ("fact:thermos_location_old",))
        self.assertGreaterEqual(remote_state.client.topk_records_calls, 2)

    def test_select_relevant_context_objects_retries_with_smaller_scope_window_after_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.client = _LargeWindowTimeoutingScopeTopKClient(max_ok_limit=3)
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="episode:doctor",
                        kind="episode",
                        summary="Janina had an eye doctor appointment yesterday.",
                        source=_source(),
                        status="active",
                        confidence=0.95,
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:janina_spouse",
                        kind="relationship_fact",
                        summary="Janina is the user's wife.",
                        source=_source(),
                        status="active",
                        confidence=0.98,
                    ),
                )
            )

            episodic, durable = store.select_relevant_context_objects(
                query_text="Janina",
                episodic_limit=3,
                durable_limit=3,
            )

        self.assertEqual(tuple(item.memory_id for item in episodic), ("episode:doctor",))
        self.assertEqual(tuple(item.memory_id for item in durable), ("fact:janina_spouse",))
        self.assertEqual(
            tuple(remote_state.client.attempted_limits[:2]),
            (12, 3),
        )

    def test_select_relevant_context_objects_rescues_collapsed_stale_scope_hits_for_confirmed_meta_query(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.client = _StaleScopeTopKClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_generic",
                        kind="fact",
                        summary="User usually likes some jam on bread at breakfast.",
                        details='User said: "Ich mag beim Frühstück meistens etwas Marmelade auf dem Brot."',
                        source=_source(),
                        status="active",
                        confidence=0.84,
                        slot_key="fact:user:breakfast:jam",
                        value_key="jam_on_bread_at_breakfast",
                        attributes={
                            "fact_type": "general",
                            "memory_domain": "general",
                            "value_text": "jam on bread at breakfast",
                        },
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_preference_old",
                        kind="fact",
                        summary="Deine Lieblingsmarmelade ist Erdbeermarmelade.",
                        details="Aeltere Vorliebe fuer das Fruehstueck.",
                        source=_source(),
                        status="active",
                        confidence=0.94,
                        slot_key="preference:breakfast:jam",
                        value_key="strawberry",
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_preference_new",
                        kind="fact",
                        summary="Inzwischen magst du lieber Aprikosenmarmelade.",
                        details="Neuere Vorliebe fuer das Fruehstueck.",
                        source=_source(),
                        status="uncertain",
                        confidence=0.95,
                        slot_key="preference:breakfast:jam",
                        value_key="apricot",
                    ),
                ),
            )
            remote_state.client.snapshot_scope_records()
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_generic",
                        kind="fact",
                        summary="User usually likes some jam on bread at breakfast.",
                        details='User said: "Ich mag beim Frühstück meistens etwas Marmelade auf dem Brot."',
                        source=_source(),
                        status="active",
                        confidence=0.84,
                        slot_key="fact:user:breakfast:jam",
                        value_key="jam_on_bread_at_breakfast",
                        attributes={
                            "fact_type": "general",
                            "memory_domain": "general",
                            "value_text": "jam on bread at breakfast",
                        },
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_preference_old",
                        kind="fact",
                        summary="Deine Lieblingsmarmelade ist Erdbeermarmelade.",
                        details="Aeltere Vorliebe fuer das Fruehstueck.",
                        source=_source(),
                        status="superseded",
                        confidence=0.94,
                        slot_key="preference:breakfast:jam",
                        value_key="strawberry",
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_preference_new",
                        kind="fact",
                        summary="Inzwischen magst du lieber Aprikosenmarmelade.",
                        details="Neuere Vorliebe fuer das Fruehstueck.",
                        source=_source(),
                        status="active",
                        confidence=0.99,
                        confirmed_by_user=True,
                        slot_key="preference:breakfast:jam",
                        value_key="apricot",
                        attributes={
                            "fact_type": "general",
                            "memory_domain": "general",
                            "resolved_by_user": True,
                        },
                    ),
                ),
            )

            episodic, durable = store.select_relevant_context_objects(
                query_text="Welche Marmelade ist jetzt als bestaetigt gespeichert?",
                episodic_limit=3,
                durable_limit=3,
            )

        self.assertEqual(tuple(item.memory_id for item in episodic), ())
        self.assertEqual(tuple(item.memory_id for item in durable[:2]), ("fact:jam_preference_new", "fact:jam_generic"))
        self.assertTrue(durable[0].confirmed_by_user)

    def test_select_relevant_context_objects_rescues_collapsed_stale_scope_hits_for_current_query(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.client = _StaleScopeTopKClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_generic",
                        kind="fact",
                        summary="User usually likes some jam on bread at breakfast.",
                        details='User said: "Ich mag beim Frühstück meistens etwas Marmelade auf dem Brot."',
                        source=_source(),
                        status="active",
                        confidence=0.84,
                        slot_key="fact:user:breakfast:jam",
                        value_key="jam_on_bread_at_breakfast",
                        attributes={
                            "fact_type": "general",
                            "memory_domain": "general",
                            "value_text": "jam on bread at breakfast",
                        },
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_preference_old",
                        kind="fact",
                        summary="Deine Lieblingsmarmelade ist Erdbeermarmelade.",
                        details="Aeltere Vorliebe fuer das Fruehstueck.",
                        source=_source(),
                        status="active",
                        confidence=0.94,
                        slot_key="preference:breakfast:jam",
                        value_key="strawberry",
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_preference_new",
                        kind="fact",
                        summary="Inzwischen magst du lieber Aprikosenmarmelade.",
                        details="Neuere Vorliebe fuer das Fruehstueck.",
                        source=_source(),
                        status="uncertain",
                        confidence=0.95,
                        slot_key="preference:breakfast:jam",
                        value_key="apricot",
                    ),
                ),
            )
            remote_state.client.snapshot_scope_records()
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_generic",
                        kind="fact",
                        summary="User usually likes some jam on bread at breakfast.",
                        details='User said: "Ich mag beim Frühstück meistens etwas Marmelade auf dem Brot."',
                        source=_source(),
                        status="active",
                        confidence=0.84,
                        slot_key="fact:user:breakfast:jam",
                        value_key="jam_on_bread_at_breakfast",
                        attributes={
                            "fact_type": "general",
                            "memory_domain": "general",
                            "value_text": "jam on bread at breakfast",
                        },
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_preference_old",
                        kind="fact",
                        summary="Deine Lieblingsmarmelade ist Erdbeermarmelade.",
                        details="Aeltere Vorliebe fuer das Fruehstueck.",
                        source=_source(),
                        status="superseded",
                        confidence=0.94,
                        slot_key="preference:breakfast:jam",
                        value_key="strawberry",
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_preference_new",
                        kind="fact",
                        summary="Inzwischen magst du lieber Aprikosenmarmelade.",
                        details="Neuere Vorliebe fuer das Fruehstueck.",
                        source=_source(),
                        status="active",
                        confidence=0.99,
                        confirmed_by_user=True,
                        slot_key="preference:breakfast:jam",
                        value_key="apricot",
                        attributes={
                            "fact_type": "general",
                            "memory_domain": "general",
                            "resolved_by_user": True,
                        },
                    ),
                ),
            )

            episodic, durable = store.select_relevant_context_objects(
                query_text="Welche Marmeladensorte ist aktuell gespeichert?",
                episodic_limit=3,
                durable_limit=3,
            )

        self.assertEqual(tuple(item.memory_id for item in episodic), ())
        self.assertEqual(tuple(item.memory_id for item in durable[:2]), ("fact:jam_preference_new", "fact:jam_generic"))
        self.assertTrue(durable[0].confirmed_by_user)

    def test_select_relevant_context_objects_skips_catalog_rescue_after_offtopic_scope_hits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.client = _SemanticDriftScopeTopKClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_preference_old",
                        kind="fact",
                        summary="Deine Lieblingsmarmelade ist Erdbeermarmelade.",
                        details="Aeltere Vorliebe fuer das Fruehstueck.",
                        source=_source(),
                        status="active",
                        confidence=0.94,
                        slot_key="preference:breakfast:jam",
                        value_key="strawberry",
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:janina_spouse",
                        kind="relationship_fact",
                        summary="Janina is the user's wife.",
                        source=_source(),
                        status="active",
                        confidence=0.98,
                    ),
                ),
            )

            episodic, durable = store.select_relevant_context_objects(
                query_text="Was ist ein Regenbogen?",
                episodic_limit=3,
                durable_limit=3,
            )

        self.assertEqual(tuple(item.memory_id for item in episodic), ())
        self.assertEqual(tuple(item.memory_id for item in durable), ())
        self.assertEqual(remote_state.client.topk_records_calls, 1)
        self.assertEqual(remote_state.client.retrieve_calls, 1)

    def test_select_relevant_objects_falls_back_to_catalog_search_after_scope_topk_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.client = _TimeoutingScopeTopKClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:janina_spouse",
                        kind="relationship_fact",
                        summary="Janina is the user's wife.",
                        source=_source(),
                        status="active",
                        confidence=0.98,
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:tea_preference",
                        kind="preference_fact",
                        summary="The user likes black tea.",
                        source=_source(),
                        status="active",
                        confidence=0.9,
                    ),
                )
            )

            relevant = store.select_relevant_objects(query_text="Janina", limit=1)

        self.assertEqual(tuple(item.memory_id for item in relevant), ("fact:janina_spouse",))
        self.assertGreaterEqual(remote_state.client.topk_records_calls, 2)
        self.assertEqual(remote_state.client.retrieve_calls, remote_state.client.topk_records_calls)

    def test_select_relevant_objects_falls_back_when_scope_topk_returns_false_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.client = _TermOverlapChonkyClient()
            remote_state.client.supports_topk_records = True
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:thermos_location_old",
                        kind="fact",
                        summary="Früher stand die rote Thermoskanne im Flurschrank.",
                        details="Historische Ortsangabe zur roten Thermoskanne.",
                        source=_source(),
                        status="active",
                        confidence=0.99,
                        confirmed_by_user=True,
                        slot_key="object:red_thermos:location",
                        value_key="hallway_cupboard",
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:tea_preference",
                        kind="fact",
                        summary="Du trinkst gern schwarzen Tee.",
                        source=_source(),
                        status="active",
                        confidence=0.91,
                        slot_key="preference:drink:tea",
                        value_key="black_tea",
                    ),
                ),
                conflicts=(),
                archived_objects=(),
            )

            relevant = store.select_relevant_objects(
                query_text="Wo stand früher meine rote Thermoskanne?",
                limit=1,
            )

        self.assertEqual(tuple(item.memory_id for item in relevant), ("fact:thermos_location_old",))

    def test_search_catalog_entries_uses_scope_ref_for_full_current_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:janina_spouse",
                        kind="relationship_fact",
                        summary="Janina is the user's wife.",
                        source=_source(),
                        status="active",
                        confidence=0.98,
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:tea_preference",
                        kind="preference_fact",
                        summary="The user likes black tea.",
                        source=_source(),
                        status="active",
                        confidence=0.9,
                    ),
                )
            )
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None
            remote_state.client.supports_topk_records = True

            relevant = remote_catalog.search_catalog_entries(
                snapshot_kind="objects",
                query_text="Janina",
                limit=1,
            )

        self.assertEqual(tuple(entry.item_id for entry in relevant), ("fact:janina_spouse",))
        payload = remote_state.client.topk_records_payloads[-1]
        self.assertEqual(payload["namespace"], "test-namespace")
        self.assertEqual(payload["scope_ref"], "longterm:objects:current")
        self.assertNotIn("allowed_doc_ids", payload)

    def test_search_catalog_entries_keeps_explicit_allowlist_for_filtered_subsets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:janina_spouse",
                        kind="relationship_fact",
                        summary="Janina is the user's wife.",
                        source=_source(),
                        status="active",
                        confidence=0.98,
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:tea_preference",
                        kind="preference_fact",
                        summary="The user likes black tea.",
                        source=_source(),
                        status="active",
                        confidence=0.9,
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:morning_walk",
                        kind="routine_fact",
                        summary="The user walks every morning.",
                        source=_source(),
                        status="active",
                        confidence=0.88,
                    ),
                )
            )
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None
            remote_state.client.supports_topk_records = True

            relevant = remote_catalog.search_catalog_entries(
                snapshot_kind="objects",
                query_text="tea",
                limit=1,
                eligible=lambda entry: entry.item_id != "fact:morning_walk",
            )

        self.assertEqual(tuple(entry.item_id for entry in relevant), ("fact:tea_preference",))
        payload = remote_state.client.topk_records_payloads[-1]
        self.assertNotIn("namespace", payload)
        self.assertNotIn("scope_ref", payload)
        self.assertEqual(len(payload["allowed_doc_ids"]), 2)
    def test_select_relevant_objects_prefers_query_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _config(temp_dir)
            store = LongTermStructuredStore.from_config(config)
            extractor = make_test_extractor()
            consolidator = LongTermMemoryConsolidator(truth_maintainer=LongTermTruthMaintainer())
            extraction = extractor.extract_conversation_turn(
                transcript=(
                    "Today is a beautiful Sunday, it is really warm. "
                    "My wife Janina is at the eye doctor and is getting eye laser treatment."
                ),
                response="I hope Janina's appointment goes smoothly.",
                occurred_at=datetime(2026, 3, 14, 10, 30, tzinfo=ZoneInfo("Europe/Berlin")),
                turn_id="turn:test",
            )
            result = consolidator.consolidate(extraction=extraction)
            store.apply_consolidation(result)

            relevant = store.select_relevant_objects(query_text="How is Janina today?", limit=3)

        summaries = [item.summary for item in relevant]
        self.assertIn("Janina is the user's wife.", summaries)
        self.assertTrue(any("eye laser treatment" in summary for summary in summaries))

    def test_select_relevant_objects_ignores_auxiliary_only_overlap_for_control_queries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LongTermStructuredStore.from_config(_config(temp_dir))
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_preference_old",
                        kind="fact",
                        summary="Deine Lieblingsmarmelade ist Erdbeermarmelade.",
                        details="Aeltere Vorliebe fuer das Fruehstueck.",
                        source=_source(),
                        status="active",
                        confidence=0.94,
                        slot_key="preference:breakfast:jam",
                        value_key="strawberry",
                    ),
                ),
                conflicts=(),
                archived_objects=(),
            )

            relevant = store.select_relevant_objects(query_text="Was ist ein Regenbogen?", limit=3)

        self.assertEqual(relevant, ())

    def test_select_relevant_objects_prefers_confirmed_fact_for_meta_memory_queries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LongTermStructuredStore.from_config(_config(temp_dir))
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_generic",
                        kind="fact",
                        summary="User usually likes some jam on bread at breakfast.",
                        details='User said: "Ich mag beim Frühstück meistens etwas Marmelade auf dem Brot."',
                        source=_source(),
                        status="active",
                        confidence=0.84,
                        slot_key="fact:user:breakfast:jam",
                        value_key="jam_on_bread_at_breakfast",
                        attributes={
                            "fact_type": "general",
                            "memory_domain": "general",
                            "value_text": "jam on bread at breakfast",
                        },
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_preference_old",
                        kind="fact",
                        summary="Deine Lieblingsmarmelade ist Erdbeermarmelade.",
                        details="Aeltere Vorliebe fuer das Fruehstueck.",
                        source=_source(),
                        status="superseded",
                        confidence=0.9,
                        slot_key="preference:breakfast:jam",
                        value_key="strawberry",
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_preference_new",
                        kind="fact",
                        summary="Inzwischen magst du lieber Aprikosenmarmelade.",
                        details="Neuere Vorliebe fuer das Fruehstueck.",
                        source=_source(),
                        status="active",
                        confidence=0.99,
                        confirmed_by_user=True,
                        slot_key="preference:breakfast:jam",
                        value_key="apricot",
                        attributes={
                            "fact_type": "general",
                            "memory_domain": "general",
                            "resolved_by_user": True,
                        },
                    ),
                ),
                conflicts=(),
                archived_objects=(),
            )

            relevant = store.select_relevant_objects(
                query_text="Welche Marmelade ist jetzt als bestaetigt gespeichert?",
                limit=3,
            )

        self.assertEqual(relevant[0].memory_id, "fact:jam_preference_new")
        self.assertTrue(relevant[0].confirmed_by_user)

    def test_remote_primary_store_filters_semantic_drift_for_control_queries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.client = _SemanticDriftChonkyClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_preference_old",
                        kind="fact",
                        summary="Deine Lieblingsmarmelade ist Erdbeermarmelade.",
                        details="Aeltere Vorliebe fuer das Fruehstueck.",
                        source=_source(),
                        status="active",
                        confidence=0.94,
                        slot_key="preference:breakfast:jam",
                        value_key="strawberry",
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_preference_new",
                        kind="fact",
                        summary="Inzwischen magst du lieber Aprikosenmarmelade.",
                        details="Neuere Vorliebe fuer das Fruehstueck.",
                        source=_source(),
                        status="uncertain",
                        confidence=0.95,
                        slot_key="preference:breakfast:jam",
                        value_key="apricot",
                    ),
                ),
                conflicts=(
                    LongTermMemoryConflictV1(
                        slot_key="preference:breakfast:jam",
                        candidate_memory_id="fact:jam_preference_new",
                        existing_memory_ids=("fact:jam_preference_old",),
                        question="Welche Marmelade stimmt gerade?",
                        reason="Widerspruechliche Marmeladenpraeferenzen liegen vor.",
                    ),
                ),
                archived_objects=(),
            )

            relevant = store.select_relevant_objects(query_text="Was ist ein Regenbogen?", limit=3)
            conflicts = store.select_open_conflicts(query_text="Was ist ein Regenbogen?", limit=3)

        self.assertEqual(relevant, ())
        self.assertEqual(conflicts, ())

    def test_remote_primary_store_prefers_confirmed_fact_for_meta_memory_queries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.client = _TermOverlapChonkyClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_generic",
                        kind="fact",
                        summary="User usually likes some jam on bread at breakfast.",
                        details='User said: "Ich mag beim Frühstück meistens etwas Marmelade auf dem Brot."',
                        source=_source(),
                        status="active",
                        confidence=0.84,
                        slot_key="fact:user:breakfast:jam",
                        value_key="jam_on_bread_at_breakfast",
                        attributes={
                            "fact_type": "general",
                            "memory_domain": "general",
                            "value_text": "jam on bread at breakfast",
                        },
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_preference_new",
                        kind="fact",
                        summary="Inzwischen magst du lieber Aprikosenmarmelade.",
                        details="Neuere Vorliebe fuer das Fruehstueck.",
                        source=_source(),
                        status="active",
                        confidence=0.99,
                        confirmed_by_user=True,
                        slot_key="preference:breakfast:jam",
                        value_key="apricot",
                        attributes={
                            "fact_type": "general",
                            "memory_domain": "general",
                            "resolved_by_user": True,
                        },
                    ),
                )
            )

            relevant = store.select_relevant_objects(
                query_text="Which marmalade is currently saved as confirmed?",
                limit=3,
            )

        self.assertEqual(relevant[0].memory_id, "fact:jam_preference_new")
        self.assertTrue(relevant[0].confirmed_by_user)

    def test_remote_primary_store_prefers_confirmed_fact_for_meta_memory_queries_after_scope_false_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.client = _TermOverlapChonkyClient()
            remote_state.client.supports_topk_records = True
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_generic",
                        kind="fact",
                        summary="User usually likes some jam on bread at breakfast.",
                        details='User said: "Ich mag beim Frühstück meistens etwas Marmelade auf dem Brot."',
                        source=_source(),
                        status="active",
                        confidence=0.84,
                        slot_key="fact:user:breakfast:jam",
                        value_key="jam_on_bread_at_breakfast",
                        attributes={
                            "fact_type": "general",
                            "memory_domain": "general",
                            "value_text": "jam on bread at breakfast",
                        },
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_preference_old",
                        kind="fact",
                        summary="Deine Lieblingsmarmelade ist Erdbeermarmelade.",
                        details="Aeltere Vorliebe fuer das Fruehstueck.",
                        source=_source(),
                        status="superseded",
                        confidence=0.94,
                        slot_key="preference:breakfast:jam",
                        value_key="strawberry",
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_preference_new",
                        kind="fact",
                        summary="Inzwischen magst du lieber Aprikosenmarmelade.",
                        details="Neuere Vorliebe fuer das Fruehstueck.",
                        source=_source(),
                        status="active",
                        confidence=0.99,
                        confirmed_by_user=True,
                        slot_key="preference:breakfast:jam",
                        value_key="apricot",
                        attributes={
                            "fact_type": "general",
                            "memory_domain": "general",
                            "resolved_by_user": True,
                        },
                    ),
                )
            )

            relevant = store.select_relevant_objects(
                query_text="Welche Marmeladensorte ist aktuell gespeichert?",
                limit=3,
            )

        self.assertEqual(relevant[0].memory_id, "fact:jam_preference_new")
        self.assertTrue(relevant[0].confirmed_by_user)

    def test_remote_primary_store_meta_query_ignores_unrelated_state_only_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.client = _SemanticDriftScopeTopKClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:thermos_location_old",
                        kind="fact",
                        summary="Früher stand die rote Thermoskanne im Flurschrank.",
                        details="Historische Ortsangabe zur roten Thermoskanne.",
                        source=_source(),
                        status="active",
                        confidence=0.99,
                        confirmed_by_user=True,
                        slot_key="object:red_thermos:location",
                        value_key="hallway_cupboard",
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_generic",
                        kind="fact",
                        summary="User usually likes some jam on bread at breakfast.",
                        details='User said: "Ich mag beim Frühstück meistens etwas Marmelade auf dem Brot."',
                        source=_source(),
                        status="active",
                        confidence=0.84,
                        slot_key="fact:user:breakfast:jam",
                        value_key="jam_on_bread_at_breakfast",
                        attributes={
                            "fact_type": "general",
                            "memory_domain": "general",
                            "value_text": "jam on bread at breakfast",
                        },
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_preference_new",
                        kind="fact",
                        summary="Inzwischen magst du lieber Aprikosenmarmelade.",
                        details="Neuere Vorliebe fuer das Fruehstueck.",
                        source=_source(),
                        status="active",
                        confidence=0.99,
                        confirmed_by_user=True,
                        slot_key="preference:breakfast:jam",
                        value_key="apricot",
                        attributes={
                            "fact_type": "general",
                            "memory_domain": "general",
                            "resolved_by_user": True,
                        },
                    ),
                )
            )

            relevant = store.select_relevant_objects(
                query_text="Welche Marmelade ist jetzt als bestaetigt gespeichert?",
                limit=3,
            )

        self.assertEqual(tuple(item.memory_id for item in relevant[:2]), ("fact:jam_preference_new", "fact:jam_generic"))
        self.assertNotIn("fact:thermos_location_old", tuple(item.memory_id for item in relevant))

    def test_store_merges_repeated_memory_ids_into_support_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LongTermStructuredStore.from_config(_config(temp_dir))
            first = LongTermConsolidationResultV1(
                turn_id="turn:1",
                occurred_at=datetime(2026, 3, 14, 10, 0, tzinfo=ZoneInfo("Europe/Berlin")),
                episodic_objects=(),
                durable_objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:janina_wife",
                        kind="relationship_fact",
                        summary="Janina is the user's wife.",
                        source=_source(),
                        status="active",
                        confidence=0.98,
                        slot_key="relationship:user:main:wife",
                        value_key="person:janina",
                        attributes={"person_ref": "person:janina", "person_name": "Janina", "relation": "wife"},
                    ),
                ),
                deferred_objects=(),
                conflicts=(),
                graph_edges=(),
            )
            second = LongTermConsolidationResultV1(
                turn_id="turn:2",
                occurred_at=datetime(2026, 3, 14, 11, 0, tzinfo=ZoneInfo("Europe/Berlin")),
                episodic_objects=(),
                durable_objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:janina_wife",
                        kind="relationship_fact",
                        summary="Janina is the user's wife.",
                        source=LongTermSourceRefV1(
                            source_type="conversation_turn",
                            event_ids=("turn:2",),
                            speaker="user",
                            modality="voice",
                        ),
                        status="active",
                        confidence=0.98,
                        slot_key="relationship:user:main:wife",
                        value_key="person:janina",
                        attributes={"person_ref": "person:janina", "person_name": "Janina", "relation": "wife"},
                    ),
                ),
                deferred_objects=(),
                conflicts=(),
                graph_edges=(),
            )

            store.apply_consolidation(first)
            store.apply_consolidation(second)

            objects = store.load_objects()

        relation = next(item for item in objects if item.memory_id == "fact:janina_wife")
        self.assertEqual((relation.attributes or {}).get("support_count"), 2)

    def test_select_relevant_objects_ignores_internal_numeric_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LongTermStructuredStore.from_config(_config(temp_dir))
            result = LongTermConsolidationResultV1(
                turn_id="turn:test",
                occurred_at=datetime(2026, 3, 14, 12, 0, tzinfo=ZoneInfo("Europe/Berlin")),
                episodic_objects=(),
                durable_objects=(
                    LongTermMemoryObjectV1(
                        memory_id="pattern:print:button:afternoon",
                        kind="interaction_pattern_fact",
                        summary="Printed Twinr output was used in the afternoon.",
                        details="Low-confidence print usage pattern derived from a button print completion event.",
                        source=_source(),
                        status="active",
                        confidence=0.61,
                        slot_key="pattern:print:button:afternoon:2026-03-14",
                        value_key="printed_output",
                        attributes={"request_source": "button", "daypart": "afternoon", "support_count": 20},
                    ),
                ),
                deferred_objects=(),
                conflicts=(),
                graph_edges=(),
            )
            store.apply_consolidation(result)

            relevant = store.select_relevant_objects(query_text="calculate 27 times 14", limit=3)
            more = store.select_relevant_objects(query_text="convert 20 celsius to fahrenheit", limit=3)

        self.assertEqual(relevant, ())
        self.assertEqual(more, ())

    def test_review_objects_can_filter_and_rank_durable_items(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _config(temp_dir)
            store = LongTermStructuredStore.from_config(config)
            extractor = make_test_extractor()
            consolidator = LongTermMemoryConsolidator(truth_maintainer=LongTermTruthMaintainer())
            extraction = extractor.extract_conversation_turn(
                transcript=(
                    "Today is a beautiful Sunday, it is really warm. "
                    "My wife Janina is at the eye doctor and is getting eye laser treatment."
                ),
                response="I hope Janina's appointment goes smoothly.",
                occurred_at=datetime(2026, 3, 14, 10, 30, tzinfo=ZoneInfo("Europe/Berlin")),
                turn_id="turn:test",
            )
            store.apply_consolidation(consolidator.consolidate(extraction=extraction))

            review = store.review_objects(
                query_text="What happened to Janina today?",
                status="active",
                include_episodes=False,
                limit=4,
            )

        self.assertGreaterEqual(review.total_count, 2)
        self.assertTrue(all(item.kind != "episode" for item in review.items))
        self.assertTrue(any("Janina is the user's wife." == item.summary for item in review.items))
        self.assertTrue(any("eye laser treatment" in item.summary for item in review.items))

    def test_invalidate_object_marks_memory_invalid_and_drops_related_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LongTermStructuredStore.from_config(_config(temp_dir))
            result = LongTermConsolidationResultV1(
                turn_id="turn:test",
                occurred_at=datetime(2026, 3, 14, 12, 0, tzinfo=ZoneInfo("Europe/Berlin")),
                episodic_objects=(),
                durable_objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:corinna_phone_old",
                        kind="contact_method_fact",
                        summary=f"Corinna Maier can be reached at {_TEST_CORINNA_PHONE_OLD}.",
                        source=_source(),
                        status="active",
                        confidence=0.95,
                        slot_key="contact:person:corinna_maier:phone",
                        value_key=_TEST_CORINNA_PHONE_OLD,
                    ),
                ),
                deferred_objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:corinna_phone_new",
                        kind="contact_method_fact",
                        summary=f"Corinna Maier can be reached at {_TEST_CORINNA_PHONE_NEW}.",
                        source=_source(),
                        status="uncertain",
                        confidence=0.92,
                        slot_key="contact:person:corinna_maier:phone",
                        value_key=_TEST_CORINNA_PHONE_NEW,
                    ),
                ),
                conflicts=(
                    LongTermMemoryConflictV1(
                        slot_key="contact:person:corinna_maier:phone",
                        candidate_memory_id="fact:corinna_phone_new",
                        existing_memory_ids=("fact:corinna_phone_old",),
                        question="Which phone number should I use for Corinna Maier?",
                        reason="Conflicting phone numbers exist.",
                    ),
                ),
                graph_edges=(),
            )
            store.apply_consolidation(result)

            mutation = store.invalidate_object("fact:corinna_phone_new", reason="User said this is outdated.")
            store.apply_memory_mutation(mutation)
            objects = {item.memory_id: item for item in store.load_objects()}
            conflicts = store.load_conflicts()

        self.assertEqual(mutation.action, "invalidate")
        self.assertEqual(objects["fact:corinna_phone_new"].status, "invalid")
        self.assertEqual(objects["fact:corinna_phone_new"].attributes["invalidation_reason"], "User said this is outdated.")
        self.assertEqual(conflicts, ())

    def test_delete_object_removes_memory_and_cleans_reference_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LongTermStructuredStore.from_config(_config(temp_dir))
            result = LongTermConsolidationResultV1(
                turn_id="turn:test",
                occurred_at=datetime(2026, 3, 14, 12, 0, tzinfo=ZoneInfo("Europe/Berlin")),
                episodic_objects=(),
                durable_objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:janina_wife",
                        kind="relationship_fact",
                        summary="Janina is the user's wife.",
                        source=_source(),
                        status="active",
                        confidence=0.99,
                        slot_key="relationship:user:main:wife",
                        value_key="person:janina",
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="summary:janina_thread",
                        kind="thread_summary",
                        summary="Ongoing thread about Janina and her appointments.",
                        source=_source(),
                        status="active",
                        confidence=0.88,
                        conflicts_with=("fact:janina_wife",),
                        supersedes=("fact:janina_wife",),
                    ),
                ),
                deferred_objects=(),
                conflicts=(),
                graph_edges=(),
            )
            store.apply_consolidation(result)

            mutation = store.delete_object("fact:janina_wife")
            store.apply_memory_mutation(mutation)
            objects = {item.memory_id: item for item in store.load_objects()}

        self.assertEqual(mutation.deleted_memory_ids, ("fact:janina_wife",))
        self.assertNotIn("fact:janina_wife", objects)
        self.assertEqual(objects["summary:janina_thread"].conflicts_with, ())
        self.assertEqual(objects["summary:janina_thread"].supersedes, ())


if __name__ == "__main__":
    unittest.main()
