from __future__ import annotations

from datetime import datetime, timezone
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
from twinr.memory.chonkydb.models import ChonkyDBRecordItem
from twinr.memory.longterm.reasoning.consolidator import LongTermMemoryConsolidator
from twinr.memory.longterm.reasoning.conflicts import LongTermConflictResolver
from twinr.memory.longterm.reasoning.retention import LongTermRetentionPolicy
from twinr.memory.longterm.core.cooperative_abort import (
    LongTermOperationCancelledError,
    longterm_operation_abort_scope,
)
from twinr.memory.longterm.core.models import (
    LongTermConsolidationResultV1,
    LongTermConflictResolutionV1,
    LongTermGraphEdgeCandidateV1,
    LongTermMemoryConflictV1,
    LongTermMemoryContext,
    LongTermMidtermPacketV1,
    LongTermMemoryMutationResultV1,
    LongTermMemoryObjectV1,
    LongTermSourceRefV1,
)
from twinr.memory.longterm.retrieval.retriever import LongTermRetriever
from twinr.memory.longterm.storage import remote_read_diagnostics, remote_read_observability
from twinr.memory.longterm.storage.midterm_store import LongTermMidtermStore
from twinr.memory.longterm.storage.provider_answer_front_store import LongTermProviderAnswerFrontStore
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
from twinr.memory.query_normalization import LongTermQueryProfile
from twinr.ops.events import TwinrOpsEventStore
from twinr.text_utils import retrieval_terms

_TEST_CORINNA_PHONE_OLD = "+15555551234"
_TEST_CORINNA_PHONE_NEW = "+15555558877"
_TEST_MARTA_PHONE_OLD = "+15555551122"
_TEST_MARTA_PHONE_NEW = "+15555553456"


def _raise_missing_current_head_or_unavailable(origin_uri: object) -> None:
    """Mirror the live backend: missing fixed current heads are explicit 404s."""

    if isinstance(origin_uri, str) and origin_uri.endswith("/catalog/current"):
        raise ChonkyDBError(
            "ChonkyDB request failed for GET /v1/external/documents/full",
            status_code=404,
            response_json={
                "detail": "document_not_found",
                "error": "document_not_found",
                "error_type": "KeyError",
                "success": False,
            },
        )
    raise LongTermRemoteUnavailableError("remote document unavailable")


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
        self.client = _LaggingCurrentHeadVisibilityClient()
        self.read_client = self.client
        self.write_client = self.client

    def promote_pending(self, snapshot_kind: str) -> None:
        self.client.promote_pending(snapshot_kind)


class _FakeChonkyClient:
    def __init__(self, *, max_items_per_bulk: int | None = None, max_request_bytes: int | None = None) -> None:
        self._next_document_id = 1
        self.max_items_per_bulk = max_items_per_bulk
        self.max_request_bytes = max_request_bytes
        self.supports_topk_records = False
        self.bulk_calls = 0
        self.retrieve_calls = 0
        self.retrieve_payloads: list[dict[str, object]] = []
        self.topk_records_calls = 0
        self.topk_records_payloads: list[dict[str, object]] = []
        self.fetch_full_document_calls = 0
        self.bulk_request_bytes: list[int] = []
        self.bulk_request_payloads: list[dict[str, object]] = []
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
        self.bulk_request_payloads.append(dict(payload))
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
        _raise_missing_current_head_or_unavailable(origin_uri)

    def retrieve(self, request):
        payload = request.to_payload() if hasattr(request, "to_payload") else dict(request)
        query_text = str(payload.get("query_text") or "").lower()
        allowed = set(str(value) for value in payload.get("allowed_doc_ids", ()) if str(value))
        self.retrieve_calls += 1
        self.retrieve_payloads.append(dict(payload))
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


class _CurrentHead404Client(_FakeChonkyClient):
    """Report missing current-head reads as explicit 404s, not generic transport failures."""

    def fetch_full_document(self, *, document_id=None, origin_uri=None, include_content=True, max_content_chars=4000):
        if isinstance(origin_uri, str) and origin_uri.endswith("/catalog/current") and origin_uri not in self.records_by_uri:
            self.fetch_full_document_calls += 1
            raise ChonkyDBError(
                "ChonkyDB request failed for GET /v1/external/documents/full",
                status_code=404,
                response_json={
                    "detail": "document_not_found",
                    "error": "document_not_found",
                    "error_type": "KeyError",
                    "success": False,
                },
            )
        return super().fetch_full_document(
            document_id=document_id,
            origin_uri=origin_uri,
            include_content=include_content,
            max_content_chars=max_content_chars,
        )


class _TimeoutCloningReadClient(_FakeChonkyClient):
    def __init__(self, *, timeout_s: float = 8.0) -> None:
        super().__init__()
        self.config = SimpleNamespace(timeout_s=float(timeout_s))
        self.clone_timeout_history: list[float] = []
        self.fetch_timeout_history: list[float] = []

    def clone_with_timeout(self, timeout_s: float):
        self.clone_timeout_history.append(float(timeout_s))
        return _TimeoutCloningReadClientView(parent=self, timeout_s=float(timeout_s))


class _TimeoutCloningReadClientView:
    def __init__(self, *, parent: _TimeoutCloningReadClient, timeout_s: float) -> None:
        self._parent = parent
        self.config = SimpleNamespace(timeout_s=float(timeout_s))

    def clone_with_timeout(self, timeout_s: float):
        return self._parent.clone_with_timeout(timeout_s)

    def fetch_full_document(self, *, document_id=None, origin_uri=None, include_content=True, max_content_chars=4000):
        del document_id
        del origin_uri
        del include_content
        del max_content_chars
        self._parent.fetch_timeout_history.append(float(self.config.timeout_s))
        raise TimeoutError(f"timed out after {self.config.timeout_s:.3f}s")


class _TimeoutCloningWriteClient(_FakeChonkyClient):
    def __init__(self, *, timeout_s: float = 15.0) -> None:
        super().__init__()
        self.config = SimpleNamespace(timeout_s=float(timeout_s))
        self.clone_timeout_history: list[float] = []
        self.store_timeout_history: list[float] = []

    def clone_with_timeout(self, timeout_s: float):
        self.clone_timeout_history.append(float(timeout_s))
        return _TimeoutCloningWriteClientView(parent=self, timeout_s=float(timeout_s))


class _TimeoutCloningWriteClientView:
    def __init__(self, *, parent: _TimeoutCloningWriteClient, timeout_s: float) -> None:
        self._parent = parent
        self.config = SimpleNamespace(timeout_s=float(timeout_s))

    def clone_with_timeout(self, timeout_s: float):
        return self._parent.clone_with_timeout(timeout_s)

    def store_records_bulk(self, request):
        self._parent.store_timeout_history.append(float(self.config.timeout_s))
        return _FakeChonkyClient.store_records_bulk(self._parent, request)


class _TransientCurrentHead503Client(_CurrentHead404Client):
    """Emit one transient 503 for one current-head URI before falling back to 404/real data."""

    def __init__(self, *, snapshot_kind: str, failure_count: int = 1) -> None:
        super().__init__()
        self.snapshot_kind = str(snapshot_kind)
        self.remaining_failures = max(0, int(failure_count))
        self.current_head_attempt_uris: list[str] = []

    def fetch_full_document(self, *, document_id=None, origin_uri=None, include_content=True, max_content_chars=4000):
        if (
            self.remaining_failures > 0
            and isinstance(origin_uri, str)
            and f"/{self.snapshot_kind}/catalog/current" in origin_uri
            and origin_uri not in self.records_by_uri
        ):
            self.fetch_full_document_calls += 1
            self.current_head_attempt_uris.append(origin_uri)
            self.remaining_failures -= 1
            raise ChonkyDBError(
                "ChonkyDB request failed for GET /v1/external/documents/full",
                status_code=503,
                response_json={
                    "detail": "Upstream unavailable or restarting",
                    "status": 503,
                    "success": False,
                    "title": "Service Unavailable",
                    "type": "about:blank",
                },
            )
        return super().fetch_full_document(
            document_id=document_id,
            origin_uri=origin_uri,
            include_content=include_content,
            max_content_chars=max_content_chars,
        )


class _LaggingCurrentHeadVisibilityClient(_FakeChonkyClient):
    def __init__(self) -> None:
        super().__init__()
        self._pending_head_records_by_snapshot_kind: dict[str, dict[str, object]] = {}

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
            metadata = record.get("metadata")
            snapshot_kind = (
                str(metadata.get("twinr_snapshot_kind") or "")
                if isinstance(metadata, dict)
                else ""
            )
            is_current_head = bool(
                isinstance(metadata, dict) and metadata.get("twinr_catalog_current_head") is True
            )
            if isinstance(uri, str) and uri:
                if is_current_head and uri in self.records_by_uri:
                    self._pending_head_records_by_snapshot_kind[snapshot_kind] = dict(record)
                else:
                    self.records_by_uri[uri] = dict(record)
            response_items.append({"document_id": document_id})
        return {"items": response_items}

    def promote_pending(self, snapshot_kind: str) -> None:
        pending = self._pending_head_records_by_snapshot_kind.pop(snapshot_kind, None)
        if not isinstance(pending, dict):
            return
        uri = pending.get("uri")
        if isinstance(uri, str) and uri:
            self.records_by_uri[uri] = dict(pending)


class _FailingBulkWriteChonkyClient(_FakeChonkyClient):
    def store_records_bulk(self, request):
        del request
        raise LongTermRemoteUnavailableError("remote write unavailable")


class _TimeoutingScopeTopKClient(_FakeChonkyClient):
    def __init__(self) -> None:
        super().__init__()
        self.supports_topk_records = True
        self._failed_scope_queries = 0

    def topk_records(self, request):
        payload = request.to_payload() if hasattr(request, "to_payload") else dict(request)
        if payload.get("scope_ref") and self._failed_scope_queries == 0:
            self._failed_scope_queries += 1
            self.topk_records_calls += 1
            self.topk_records_payloads.append(dict(payload))
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


class _TransientHttp503ScopeTopKClient(_FakeChonkyClient):
    def __init__(self) -> None:
        super().__init__()
        self.supports_topk_records = True
        self._failed_scope_queries = 0

    def topk_records(self, request):
        payload = request.to_payload() if hasattr(request, "to_payload") else dict(request)
        if payload.get("scope_ref") and self._failed_scope_queries == 0:
            self._failed_scope_queries += 1
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


class _UnsupportedScopeRefTopKClient(_FakeChonkyClient):
    def __init__(self) -> None:
        super().__init__()
        self.supports_topk_records = True

    def topk_records(self, request):
        payload = request.to_payload() if hasattr(request, "to_payload") else dict(request)
        scope_ref = str(payload.get("scope_ref") or "").strip()
        if scope_ref:
            self.topk_records_calls += 1
            self.topk_records_payloads.append(dict(payload))
            self.retrieve_calls += 1
            raise ChonkyDBError(
                "ChonkyDB request failed for POST /v1/external/retrieve/topk_records",
                status_code=400,
                response_json={
                    "detail": f"unsupported scope_ref: {scope_ref}",
                    "error": f"unsupported scope_ref: {scope_ref}",
                    "success": False,
                },
            )
        return super().topk_records(request)


class _TransientSegmentTimeoutClient(_FakeChonkyClient):
    def __init__(self, *, failing_document_id: str, failing_uri: str) -> None:
        super().__init__()
        self.retrieve_attempts_by_document_id: dict[str, int] = {}
        self.fetch_attempts_by_uri: dict[str, int] = {}
        self._failing_document_id = failing_document_id
        self._remaining_document_failures = {failing_document_id: 1}
        self._remaining_uri_failures = {failing_uri: 1}

    def retrieve(self, request):
        payload = request.to_payload() if hasattr(request, "to_payload") else dict(request)
        allowed_doc_ids = tuple(str(value) for value in payload.get("allowed_doc_ids", ()) if str(value))
        if self._failing_document_id in allowed_doc_ids:
            self.retrieve_attempts_by_document_id[self._failing_document_id] = (
                self.retrieve_attempts_by_document_id.get(self._failing_document_id, 0) + 1
            )
            remaining_failures = self._remaining_document_failures.get(self._failing_document_id, 0)
            if remaining_failures > 0:
                self._remaining_document_failures[self._failing_document_id] = remaining_failures - 1
                raise ChonkyDBError(
                    "ChonkyDB request failed for POST /v1/external/retrieve: The read operation timed out"
                )
        return super().retrieve(request)

    def fetch_full_document(self, *, document_id=None, origin_uri=None, include_content=True, max_content_chars=4000):
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


class _AsyncBulkQueueSaturatedSplitClient(_FakeChonkyClient):
    def __init__(self) -> None:
        super().__init__()
        self.bulk_execution_modes: list[str] = []
        self.bulk_request_item_counts: list[int] = []

    def store_records_bulk(self, request):
        items = tuple(getattr(request, "items", ()))
        execution_mode = str(getattr(request, "execution_mode", ""))
        self.bulk_execution_modes.append(execution_mode)
        self.bulk_request_item_counts.append(len(items))
        if execution_mode == "async" and len(items) > 1:
            raise ChonkyDBError(
                "ChonkyDB request failed for POST /v1/external/records/bulk",
                status_code=429,
                response_json={
                    "detail": "queue_saturated",
                    "error": "queue_saturated",
                    "error_type": "RuntimeError",
                    "success": False,
                    "status": 429,
                },
            )
        return super().store_records_bulk(request)


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


class _AsyncBulkJobStatusTimeoutClient(_AsyncBulkEventuallyVisibleClient):
    def __init__(self) -> None:
        super().__init__(stale_reads_before_visible=0)
        self.job_status_calls: list[str] = []

    def job_status(self, job_id):
        self.job_status_calls.append(str(job_id))
        raise ChonkyDBError(
            f"ChonkyDB request failed for GET /v1/external/jobs/{job_id}: The read operation timed out"
        )


class _AsyncBulkJobStatusNoDocumentIdsClient(_AsyncBulkEventuallyVisibleClient):
    def __init__(self, *, stale_reads_before_visible: int = 2) -> None:
        super().__init__(stale_reads_before_visible=stale_reads_before_visible)
        self.job_status_calls: list[str] = []

    def job_status(self, job_id):
        self.job_status_calls.append(str(job_id))
        return {
            "success": True,
            "job_id": str(job_id),
            "status": "done",
            "result": {
                "success": True,
                "items": [{}],
            },
        }


class _AsyncBulkJobStatusNoDocumentIdsItemOrigin404Client(_AsyncBulkJobStatusNoDocumentIdsClient):
    def __init__(self, *, stale_reads_before_visible: int = 2) -> None:
        super().__init__(stale_reads_before_visible=stale_reads_before_visible)
        self.item_origin_404_attempts: dict[str, int] = {}

    def fetch_full_document(self, *, document_id=None, origin_uri=None, include_content=True, max_content_chars=4000):
        uri = str(origin_uri or "")
        if uri and "/catalog/" not in uri:
            self.item_origin_404_attempts[uri] = self.item_origin_404_attempts.get(uri, 0) + 1
            raise ChonkyDBError(
                "ChonkyDB request failed for GET /v1/external/documents/full",
                status_code=404,
                response_json={"detail": "document_not_found"},
            )
        return super().fetch_full_document(
            document_id=document_id,
            origin_uri=origin_uri,
            include_content=include_content,
            max_content_chars=max_content_chars,
        )


class _AsyncBulkDocumentIdEventuallyVisibleClient(_FakeChonkyClient):
    def __init__(self, *, stale_document_reads_before_visible: int = 2) -> None:
        super().__init__()
        self.bulk_execution_modes: list[str] = []
        self.fetch_attempts_by_document_id: dict[str, int] = {}
        self.fetch_attempts_by_uri: dict[str, int] = {}
        self._stale_document_reads_before_visible = max(0, stale_document_reads_before_visible)
        self._job_records_by_id: dict[str, list[dict[str, object]]] = {}
        self._next_job_id = 1

    def store_records_bulk(self, request):
        items = tuple(getattr(request, "items", ()))
        self.bulk_execution_modes.append(str(getattr(request, "execution_mode", "")))
        job_id = f"job-bulk-{self._next_job_id}"
        self._next_job_id += 1
        job_records: list[dict[str, object]] = []
        for item in items:
            document_id = f"doc-{self._next_document_id}"
            self._next_document_id += 1
            job_records.append(
                {
                    "document_id": document_id,
                    "payload": dict(getattr(item, "payload", {}) or {}),
                    "metadata": dict(getattr(item, "metadata", {}) or {}),
                    "content": getattr(item, "content", None),
                    "uri": getattr(item, "uri", None),
                }
            )
        self._job_records_by_id[job_id] = job_records
        return {"success": True, "job_id": job_id, "status": "pending"}

    def job_status(self, job_id):
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

    def fetch_full_document(self, *, document_id=None, origin_uri=None, include_content=True, max_content_chars=4000):
        del include_content
        del max_content_chars
        self.fetch_full_document_calls += 1
        if isinstance(document_id, str) and document_id:
            attempts = self.fetch_attempts_by_document_id.get(document_id, 0) + 1
            self.fetch_attempts_by_document_id[document_id] = attempts
            record = next(
                (
                    candidate
                    for records in self._job_records_by_id.values()
                    for candidate in records
                    if str(candidate.get("document_id") or "") == document_id
                ),
                None,
            )
            if record is not None and attempts > self._stale_document_reads_before_visible:
                uri = str(record.get("uri") or "")
                self.records_by_document_id[document_id] = dict(record)
                if uri:
                    self.records_by_uri[uri] = dict(record)
                return dict(record)
            raise ChonkyDBError(
                "ChonkyDB request failed for GET /v1/external/documents/full",
                status_code=404,
                response_json={"detail": "document_not_found"},
            )
        if isinstance(origin_uri, str) and origin_uri:
            self.fetch_attempts_by_uri[origin_uri] = self.fetch_attempts_by_uri.get(origin_uri, 0) + 1
        return super().fetch_full_document(
            document_id=document_id,
            origin_uri=origin_uri,
            include_content=True,
            max_content_chars=4000,
        )


class _AsyncBulkSegmentJobStatusPreferredClient(_AsyncBulkDocumentIdEventuallyVisibleClient):
    def __init__(self, *, clock: dict[str, float]) -> None:
        super().__init__(stale_document_reads_before_visible=0)
        self.clock = clock
        self.job_status_calls: list[str] = []

    def job_status(self, job_id):
        self.job_status_calls.append(str(job_id))
        return super().job_status(job_id)

    def fetch_full_document(self, *, document_id=None, origin_uri=None, include_content=True, max_content_chars=4000):
        if isinstance(origin_uri, str) and origin_uri:
            self.fetch_attempts_by_uri[origin_uri] = self.fetch_attempts_by_uri.get(origin_uri, 0) + 1
            # Simulate a stale same-URI history hit that consumes the remaining attestation window.
            self.clock["now"] += 1.0
            return {
                "document_id": "doc-stale",
                "payload": {
                    "schema": "stale_segment",
                    "version": 1,
                    "snapshot_kind": "objects",
                    "segment_index": 0,
                    "items": [],
                },
                "metadata": {
                    "twinr_snapshot_kind": "objects",
                    "twinr_catalog_segment_index": 0,
                },
                "content": json.dumps(
                    {
                        "schema": "stale_segment",
                        "version": 1,
                        "snapshot_kind": "objects",
                        "segment_index": 0,
                        "items": [],
                    },
                    ensure_ascii=False,
                ),
                "uri": origin_uri,
            }
        return super().fetch_full_document(
            document_id=document_id,
            origin_uri=origin_uri,
            include_content=include_content,
            max_content_chars=max_content_chars,
        )


class _AsyncBulkCurrentHeadJobStatusPreferredClient(_AsyncBulkDocumentIdEventuallyVisibleClient):
    def __init__(self, *, clock: dict[str, float]) -> None:
        super().__init__(stale_document_reads_before_visible=0)
        self.clock = clock
        self.job_status_calls: list[str] = []

    def job_status(self, job_id):
        self.job_status_calls.append(str(job_id))
        return super().job_status(job_id)

    def fetch_full_document(self, *, document_id=None, origin_uri=None, include_content=True, max_content_chars=4000):
        if isinstance(origin_uri, str) and origin_uri:
            self.fetch_attempts_by_uri[origin_uri] = self.fetch_attempts_by_uri.get(origin_uri, 0) + 1
            self.clock["now"] += 1.0
            fresh_document_id = next(
                (
                    str(candidate.get("document_id") or "")
                    for records in self._job_records_by_id.values()
                    for candidate in records
                    if str(candidate.get("uri") or "") == origin_uri
                ),
                "doc-1",
            )
            stale_payload = {
                "schema": "twinr_memory_object_catalog_v3",
                "version": 3,
                "items_count": 0,
                "segments": [],
                "written_at": "2026-03-30T09:00:00+00:00",
            }
            return {
                "success": True,
                "origin_uri": origin_uri,
                "chunks": [
                    {
                        "document_id": "doc-stale",
                        "metadata": {
                            "twinr_snapshot_kind": "objects",
                            "twinr_catalog_current_head": True,
                            "twinr_payload": dict(stale_payload),
                        },
                        "content": json.dumps(stale_payload, ensure_ascii=False),
                    },
                    {
                        "document_id": fresh_document_id,
                        "metadata": {
                            "twinr_snapshot_kind": "objects",
                            "twinr_catalog_current_head": True,
                        },
                    },
                ],
            }
        return super().fetch_full_document(
            document_id=document_id,
            origin_uri=origin_uri,
            include_content=include_content,
            max_content_chars=max_content_chars,
        )


class _ConflictAsyncFailingClient(_FakeChonkyClient):
    def __init__(self) -> None:
        super().__init__()
        self.bulk_execution_modes: list[str] = []
        self.batch_snapshot_kinds: list[tuple[str, ...]] = []

    def store_records_bulk(self, request):
        items = tuple(getattr(request, "items", ()))
        execution_mode = str(getattr(request, "execution_mode", ""))
        snapshot_kinds = tuple(
            str((getattr(item, "metadata", {}) or {}).get("twinr_snapshot_kind") or "")
            for item in items
        )
        self.bulk_execution_modes.append(execution_mode)
        self.batch_snapshot_kinds.append(snapshot_kinds)
        if execution_mode == "async" and "conflicts" in snapshot_kinds:
            raise LongTermRemoteUnavailableError("conflict async write visibility failed")
        return super().store_records_bulk(request)


class _ProjectionCompleteControlPlaneAsyncFailingClient(_FakeChonkyClient):
    def __init__(self) -> None:
        super().__init__()
        self.bulk_execution_modes: list[str] = []
        self.batch_snapshot_kinds: list[tuple[str, ...]] = []
        self.batch_uris: list[tuple[str, ...]] = []

    def store_records_bulk(self, request):
        items = tuple(getattr(request, "items", ()))
        execution_mode = str(getattr(request, "execution_mode", ""))
        snapshot_kinds = tuple(
            str((getattr(item, "metadata", {}) or {}).get("twinr_snapshot_kind") or "")
            for item in items
        )
        uris = tuple(str(getattr(item, "uri", "") or "") for item in items)
        self.bulk_execution_modes.append(execution_mode)
        self.batch_snapshot_kinds.append(snapshot_kinds)
        self.batch_uris.append(uris)
        if execution_mode == "async":
            for snapshot_kind, uri in zip(snapshot_kinds, uris, strict=False):
                if snapshot_kind not in {"objects", "conflicts", "archive", "midterm"}:
                    continue
                if uri.endswith("/catalog/current"):
                    raise LongTermRemoteUnavailableError("projection-complete control-plane write stayed async")
        return super().store_records_bulk(request)


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


class _NoItemFetchEmptyScopeTopKClient(_FakeChonkyClient):
    def __init__(self) -> None:
        super().__init__()
        self.supports_topk_records = True
        self.item_fetch_attempts = 0

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

    def fetch_full_document(self, *, document_id=None, origin_uri=None, include_content=True, max_content_chars=4000):
        record = None
        if isinstance(document_id, str) and document_id:
            record = self.records_by_document_id.get(document_id)
        if record is None and isinstance(origin_uri, str) and origin_uri:
            record = self.records_by_uri.get(origin_uri)
        if record is None:
            _raise_missing_current_head_or_unavailable(origin_uri)
        payload = dict(record.get("payload") or {})
        if payload.get("schema") == "twinr_memory_object_record_v2":
            self.item_fetch_attempts += 1
            raise AssertionError("item fetch should not run for false-empty catalog-projection rescue")
        return super().fetch_full_document(
            document_id=document_id,
            origin_uri=origin_uri,
            include_content=include_content,
            max_content_chars=max_content_chars,
        )


class _NoDocumentsFullScope404ChonkyClient(_FakeChonkyClient):
    def __init__(self) -> None:
        super().__init__()
        self.supports_topk_records = True
        self.item_fetch_attempts = 0
        self.segment_fetch_attempts = 0

    def topk_records(self, request):
        payload = request.to_payload() if hasattr(request, "to_payload") else dict(request)
        if payload.get("scope_ref"):
            self.topk_records_calls += 1
            self.topk_records_payloads.append(dict(payload))
            raise ChonkyDBError(
                "ChonkyDB request failed for POST /v1/external/retrieve/topk_records",
                status_code=404,
                response_json={
                    "detail": "document_not_found",
                    "error": "document_not_found",
                    "error_type": "KeyError",
                    "success": False,
                },
            )
        return super().topk_records(request)

    def fetch_full_document(self, *, document_id=None, origin_uri=None, include_content=True, max_content_chars=4000):
        record = None
        if isinstance(document_id, str) and document_id:
            record = self.records_by_document_id.get(document_id)
        if record is None and isinstance(origin_uri, str) and origin_uri:
            record = self.records_by_uri.get(origin_uri)
        if record is None:
            _raise_missing_current_head_or_unavailable(origin_uri)
        payload = dict(record.get("payload") or {})
        schema = str(payload.get("schema") or "")
        if schema in {
            "twinr_memory_object_record_v2",
            "twinr_memory_conflict_record_v2",
        }:
            self.item_fetch_attempts += 1
            raise AssertionError("item fetch should stay on retrieve batching once scope top-k fails")
        if schema in {
            "twinr_memory_object_catalog_segment_v1",
            "twinr_memory_conflict_catalog_segment_v1",
        }:
            self.segment_fetch_attempts += 1
            raise AssertionError("catalog segment fetch should stay on retrieve batching once scope top-k fails")
        return super().fetch_full_document(
            document_id=document_id,
            origin_uri=origin_uri,
            include_content=include_content,
            max_content_chars=max_content_chars,
        )


class _SegmentContentBearingTopKClient(_FakeChonkyClient):
    def __init__(self) -> None:
        super().__init__()
        self.supports_topk_records = True
        self.segment_fetch_attempts = 0
        self.segment_batch_include_content_values: list[bool] = []

    def topk_records(self, request):
        payload = request.to_payload() if hasattr(request, "to_payload") else dict(request)
        allowed_doc_ids = tuple(str(value) for value in payload.get("allowed_doc_ids", ()) if str(value))
        if allowed_doc_ids:
            self.topk_records_calls += 1
            self.topk_records_payloads.append(dict(payload))
            self.retrieve_calls += 1
            self.segment_batch_include_content_values.append(bool(payload.get("include_content")))
            ranked = []
            for document_id in allowed_doc_ids:
                record = self.records_by_document_id.get(document_id)
                if not isinstance(record, dict):
                    continue
                metadata = dict(record.get("metadata") or {})
                ranked.append(
                    {
                        "payload_id": document_id,
                        "document_id": document_id,
                        "relevance_score": 1.0,
                        "metadata": metadata,
                        "payload": {
                            "metadata": metadata,
                            "payload_data": {},
                        },
                        "payload_source": "service.payload_blob",
                        "content": str(record.get("content") or ""),
                        "source_index": "fulltext,vector",
                        "candidate_origin": "fulltext,vector",
                    }
                )
            return SimpleNamespace(
                success=True,
                mode="advanced",
                results=tuple(SimpleNamespace(**item) for item in ranked),
                indexes_used=("fulltext", "vector"),
                scope_ref=payload.get("scope_ref"),
                query_plan={"scope_cache_hit": False},
                raw={"results": [dict(item) for item in ranked]},
            )
        return super().topk_records(request)

    def fetch_full_document(self, *, document_id=None, origin_uri=None, include_content=True, max_content_chars=4000):
        record = None
        if isinstance(document_id, str) and document_id:
            record = self.records_by_document_id.get(document_id)
        if record is None and isinstance(origin_uri, str) and origin_uri:
            record = self.records_by_uri.get(origin_uri)
        if record is None:
            _raise_missing_current_head_or_unavailable(origin_uri)
        payload = dict(record.get("payload") or {})
        if payload.get("schema") in {
            "twinr_memory_object_catalog_segment_v1",
            "twinr_memory_conflict_catalog_segment_v1",
        }:
            self.segment_fetch_attempts += 1
            raise AssertionError("segment docs should be parsed from content-bearing batch results")
        return super().fetch_full_document(
            document_id=document_id,
            origin_uri=origin_uri,
            include_content=include_content,
            max_content_chars=max_content_chars,
        )


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
            _raise_missing_current_head_or_unavailable(origin_uri)
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
            _raise_missing_current_head_or_unavailable(origin_uri)
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


class _PayloadEnvelopeLiveShapeChonkyClient(_LiveShapeChonkyClient):
    def fetch_full_document(self, *, document_id=None, origin_uri=None, include_content=True, max_content_chars=4000):
        del max_content_chars
        record = None
        if isinstance(document_id, str) and document_id:
            record = self.records_by_document_id.get(document_id)
        if record is None and isinstance(origin_uri, str) and origin_uri:
            record = self.records_by_uri.get(origin_uri)
        if record is None:
            _raise_missing_current_head_or_unavailable(origin_uri)
        payload_id = str(record.get("document_id") or "")
        metadata = dict(record.get("metadata") or {})
        metadata.pop("twinr_payload", None)
        metadata.pop("twinr_payload_sha256", None)
        chunk = {
            "payload_id": payload_id,
            "chonky_id": payload_id,
            "doc_id_int": 1,
            "payload": dict(record.get("payload") or {}),
            "metadata": metadata,
        }
        if include_content:
            content = record.get("content")
            chunk["content"] = content
            chunk["content_summary"] = content
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
            _raise_missing_current_head_or_unavailable(origin_uri)
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


class _ConflictSelectionProjectionOnlyClient(_NoItemFetchChonkyClient):
    def __init__(self) -> None:
        super().__init__()
        self.supports_topk_records = True

    def fetch_full_document(self, *, document_id=None, origin_uri=None, include_content=True, max_content_chars=4000):
        if isinstance(origin_uri, str) and origin_uri.endswith("/catalog/current") and origin_uri not in self.records_by_uri:
            self.fetch_full_document_calls += 1
            raise ChonkyDBError(
                "ChonkyDB request failed for GET /v1/external/documents/full",
                status_code=404,
                response_json={
                    "detail": "document_not_found",
                    "error": "document_not_found",
                    "error_type": "KeyError",
                    "success": False,
                },
            )
        return super().fetch_full_document(
            document_id=document_id,
            origin_uri=origin_uri,
            include_content=include_content,
            max_content_chars=max_content_chars,
        )

    def _allowed_docs_are_object_records(self, allowed_doc_ids: tuple[str, ...]) -> bool:
        if not allowed_doc_ids:
            return False
        for document_id in allowed_doc_ids:
            record = self.records_by_document_id.get(document_id)
            payload = dict(record.get("payload") or {}) if isinstance(record, dict) else {}
            if payload.get("schema") != "twinr_memory_object_record_v2":
                return False
        return True

    def retrieve(self, request):
        payload = request.to_payload() if hasattr(request, "to_payload") else dict(request)
        allowed_doc_ids = tuple(str(value) for value in payload.get("allowed_doc_ids", ()) if str(value))
        if self._allowed_docs_are_object_records(allowed_doc_ids):
            self.retrieve_calls += 1
            return SimpleNamespace(
                success=True,
                mode="advanced",
                results=(),
                indexes_used=("fulltext",),
                raw={"results": []},
            )
        return super().retrieve(request)

    def topk_records(self, request):
        payload = request.to_payload() if hasattr(request, "to_payload") else dict(request)
        allowed_doc_ids = tuple(str(value) for value in payload.get("allowed_doc_ids", ()) if str(value))
        if self._allowed_docs_are_object_records(allowed_doc_ids):
            self.topk_records_calls += 1
            self.topk_records_payloads.append(dict(payload))
            self.retrieve_calls += 1
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
        self.client = _FailingBulkWriteChonkyClient()
        self.read_client = self.client
        self.write_client = self.client


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


def _current_catalog_head_uri(store: LongTermStructuredStore, *, snapshot_kind: str) -> str:
    remote_catalog = store._remote_catalog
    assert remote_catalog is not None
    return remote_catalog._catalog_head_uri(snapshot_kind=snapshot_kind)


def _current_catalog_head_payload(store: LongTermStructuredStore, *, snapshot_kind: str) -> dict[str, object] | None:
    remote_state = store.remote_state
    assert remote_state is not None
    uri = _current_catalog_head_uri(store, snapshot_kind=snapshot_kind)
    record = remote_state.client.records_by_uri.get(uri)
    if not isinstance(record, dict):
        return None
    payload = record.get("payload")
    return dict(payload) if isinstance(payload, dict) else None


def _count_bulk_calls_with_schema(client: _FakeChonkyClient, schema: str) -> int:
    return sum(1 for schemas in client.bulk_request_schemas if schemas and all(value == schema for value in schemas))


def _bulk_item_target_indexes(client: _FakeChonkyClient, schema: str) -> tuple[tuple[str, ...], ...]:
    targets: list[tuple[str, ...]] = []
    for request_payload in client.bulk_request_payloads:
        items = request_payload.get("items")
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            payload = item.get("payload")
            if not isinstance(payload, dict) or payload.get("schema") != schema:
                continue
            raw_targets = item.get("target_indexes")
            if isinstance(raw_targets, list):
                targets.append(tuple(str(value) for value in raw_targets if str(value).strip()))
            else:
                targets.append(())
    return tuple(targets)


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

    def test_apply_consolidation_bootstraps_fresh_required_remote_namespace_without_snapshot_blob_reads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.config.project_root = str(project_root)
            remote_state.client = _CurrentHead404Client()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            load_snapshot_calls: list[dict[str, object]] = []

            def _load_snapshot(
                *,
                snapshot_kind: str,
                local_path=None,
                prefer_cached_document_id: bool = True,
            ):
                del local_path
                load_snapshot_calls.append(
                    {
                        "snapshot_kind": snapshot_kind,
                        "prefer_cached_document_id": prefer_cached_document_id,
                    }
                )
                raise AssertionError("Fresh required-remote consolidation must not revive snapshot blob reads.")

            remote_state.load_snapshot = _load_snapshot  # type: ignore[assignment]
            store = LongTermStructuredStore(
                base_path=project_root / "state" / "chonkydb",
                remote_state=remote_state,
            )
            result = LongTermConsolidationResultV1(
                turn_id="turn:test",
                occurred_at=datetime(2026, 3, 14, 12, 0, tzinfo=ZoneInfo("Europe/Berlin")),
                episodic_objects=(),
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
                    ),
                ),
                deferred_objects=(),
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
            store_type = type(store)

            with patch.object(
                store_type,
                "load_objects",
                side_effect=AssertionError("Fresh required-remote consolidation must not hydrate object snapshots."),
            ), patch.object(
                store_type,
                "load_conflicts",
                side_effect=AssertionError("Fresh required-remote consolidation must not hydrate conflict snapshots."),
            ):
                store.apply_consolidation(result)
                objects = store.load_objects_fine_grained()
                conflicts = store.load_conflicts_fine_grained()

        self.assertEqual(load_snapshot_calls, [])
        self.assertEqual(tuple(item.memory_id for item in objects), ("fact:janina_spouse",))
        self.assertEqual(tuple(item.slot_key for item in conflicts), ("contact:person:corinna_maier:phone",))
        self.assertIsNotNone(_current_catalog_head_payload(store, snapshot_kind="objects"))
        self.assertIsNotNone(_current_catalog_head_payload(store, snapshot_kind="conflicts"))

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
        objects_head = _current_catalog_head_payload(store, snapshot_kind="objects")
        assert objects_head is not None
        self.assertEqual(objects_head["schema"], "twinr_memory_object_catalog_v3")
        self.assertEqual(objects_head["items_count"], 1)
        self.assertEqual(len(objects_head["segments"]), 1)
        self.assertEqual(remote_state.snapshots, {})
        self.assertEqual(_count_remote_records_with_schema(remote_state.client, "twinr_memory_object_record_v2"), 1)
        self.assertEqual(_count_remote_records_with_schema(remote_state.client, "twinr_memory_object_catalog_segment_v1"), 1)
        self.assertEqual(_count_remote_records_with_schema(remote_state.client, "twinr_memory_object_catalog_v3"), 1)
        self.assertEqual(
            _bulk_item_target_indexes(remote_state.client, "twinr_memory_object_record_v2"),
            (("fulltext", "temporal", "tags"),),
        )
        self.assertEqual(
            _bulk_item_target_indexes(remote_state.client, "twinr_memory_object_catalog_segment_v1"),
            ((),),
        )
        self.assertEqual(
            _bulk_item_target_indexes(remote_state.client, "twinr_memory_object_catalog_v3"),
            ((),),
        )

    def test_write_snapshot_uses_explicit_non_vector_targets_for_remote_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.config.project_root = str(project_root)
            store = LongTermStructuredStore(
                base_path=project_root / "state" / "chonkydb",
                remote_state=remote_state,
            )
            object_fact = LongTermMemoryObjectV1(
                memory_id="fact:tea_preference",
                kind="fact",
                summary="The user drinks black tea in the morning.",
                source=_source(),
                status="active",
                confidence=0.95,
                slot_key="preference:breakfast:drink",
                value_key="black_tea",
            )

            store.write_snapshot(objects=(object_fact,), conflicts=(), archived_objects=())

        self.assertEqual(
            _bulk_item_target_indexes(remote_state.client, "twinr_memory_object_record_v2"),
            (("fulltext", "temporal", "tags"),),
        )
        self.assertEqual(
            _bulk_item_target_indexes(remote_state.client, "twinr_memory_object_catalog_segment_v1"),
            ((),),
        )
        self.assertEqual(
            _bulk_item_target_indexes(remote_state.client, "twinr_memory_object_catalog_v3"),
            ((),),
        )

    def test_remote_catalog_build_payload_treats_missing_current_head_as_empty_without_legacy_snapshot_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.config.project_root = str(project_root)
            remote_state.client = _CurrentHead404Client()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            load_snapshot_calls: list[dict[str, object]] = []

            def _load_snapshot(
                *,
                snapshot_kind: str,
                local_path=None,
                prefer_cached_document_id: bool = True,
            ):
                del local_path
                load_snapshot_calls.append(
                    {
                        "snapshot_kind": snapshot_kind,
                        "prefer_cached_document_id": prefer_cached_document_id,
                    }
                )
                raise AssertionError("build_catalog_payload must not revive legacy snapshot reads when catalog/current is missing")

            remote_state.load_snapshot = _load_snapshot  # type: ignore[assignment]
            store = LongTermStructuredStore(
                base_path=project_root / "state" / "chonkydb",
                remote_state=remote_state,
            )
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None
            object_fact = LongTermMemoryObjectV1(
                memory_id="fact:jam_preference",
                kind="fact",
                summary="Du magst Aprikosenmarmelade.",
                source=_source(),
                status="active",
                confidence=0.95,
                slot_key="preference:breakfast:jam",
                value_key="apricot",
            )

            payload = remote_catalog.build_catalog_payload(
                snapshot_kind="objects",
                item_payloads=(object_fact.to_payload(),),
                item_id_getter=lambda item: item.get("memory_id"),
                metadata_builder=lambda item: store._remote_item_metadata(snapshot_kind="objects", payload=item),
                content_builder=lambda item: store._remote_item_search_text(snapshot_kind="objects", payload=item),
            )
            histogram_path = project_root / "artifacts" / "stores" / "ops" / "longterm_remote_read_histograms.json"
            histogram_payload = json.loads(histogram_path.read_text(encoding="utf-8"))

        self.assertEqual(load_snapshot_calls, [])
        self.assertEqual(payload["schema"], "twinr_memory_object_catalog_v3")
        self.assertEqual(payload["items_count"], 1)
        self.assertEqual(_count_remote_records_with_schema(remote_state.client, "twinr_memory_object_record_v2"), 1)
        operations = dict(histogram_payload.get("operations") or {})
        self.assertIn("objects:load_catalog_current_head", operations)
        head_entry = dict(operations["objects:load_catalog_current_head"])
        self.assertEqual(head_entry["last_outcome"], "missing")
        self.assertEqual(head_entry["last_classification"], "not_found")
        self.assertEqual(head_entry["last_access_classification"], "catalog_current_head")

    def test_remote_primary_store_persists_conflicts_as_fine_grained_catalog(self) -> None:
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
        conflicts_head = _current_catalog_head_payload(writer_store, snapshot_kind="conflicts")
        assert conflicts_head is not None
        self.assertEqual(conflicts_head["schema"], "twinr_memory_conflict_catalog_v3")
        self.assertEqual(conflicts_head["version"], 3)
        self.assertEqual(conflicts_head["items_count"], 1)
        self.assertEqual(len(conflicts_head["segments"]), 1)
        self.assertEqual(remote_state.snapshots, {})
        self.assertEqual(_count_remote_records_with_schema(remote_state.client, "twinr_memory_conflict_record_v2"), 1)
        self.assertEqual(_count_remote_records_with_schema(remote_state.client, "twinr_memory_conflict_catalog_segment_v1"), 1)
        self.assertEqual(_count_remote_records_with_schema(remote_state.client, "twinr_memory_conflict_catalog_v3"), 1)

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

    def test_remote_read_histogram_alert_permission_denied_stays_quiet(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.config.project_root = temp_dir
            fake_store = SimpleNamespace(
                append=lambda **kwargs: (_ for _ in ()).throw(PermissionError(13, "permission denied"))
            )

            with (
                patch.object(remote_read_observability, "_ops_event_store", return_value=fake_store),
                patch.object(remote_read_observability, "_LOG") as log,
            ):
                remote_read_observability.record_remote_read_observation(
                    remote_state=remote_state,
                    context=LongTermRemoteReadContext(
                        snapshot_kind="objects",
                        operation="topk_search",
                        request_method="POST",
                        request_path="/v1/external/retrieve/topk_records",
                        request_payload_kind="topk_scope_query",
                    ),
                    latency_ms=2500.0,
                    outcome="ok",
                    classification="ok",
                )

            histogram_exists = (
                Path(temp_dir) / "artifacts" / "stores" / "ops" / "longterm_remote_read_histograms.json"
            ).exists()

        self.assertTrue(histogram_exists)
        log.warning.assert_not_called()
        log.debug.assert_called_once()
        self.assertIn("Skipping remote long-term", log.debug.call_args.args[0])
        self.assertEqual(log.debug.call_args.args[1], "read")
        self.assertEqual(log.debug.call_args.args[2], "alert")

    def test_remote_read_diagnostic_permission_denied_stays_quiet(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.config.project_root = temp_dir
            fake_store = SimpleNamespace(
                append=lambda **kwargs: (_ for _ in ()).throw(PermissionError(13, "permission denied"))
            )

            with (
                patch.object(remote_read_diagnostics, "_ops_event_store", return_value=fake_store),
                patch.object(remote_read_diagnostics, "record_remote_read_observation") as record_observation,
                patch.object(remote_read_diagnostics, "_LOG") as log,
            ):
                record_remote_read_diagnostic(
                    remote_state=remote_state,
                    context=LongTermRemoteReadContext(
                        snapshot_kind="objects",
                        operation="retrieve_search",
                        request_method="POST",
                        request_path="/v1/external/retrieve",
                        request_payload_kind="retrieve_scope_query",
                    ),
                    exc=TimeoutError("timed out"),
                    started_monotonic=time.monotonic() - 0.05,
                    outcome="fallback",
                )

        record_observation.assert_called_once()
        log.warning.assert_not_called()
        log.debug.assert_called_once()
        self.assertIn("Skipping remote long-term", log.debug.call_args.args[0])
        self.assertEqual(log.debug.call_args.args[1], "read")
        self.assertEqual(log.debug.call_args.args[2], "diagnostic")

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
            events = TwinrOpsEventStore.from_project_root(project_root).tail(limit=20)

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
        self.assertIn("objects:load_catalog_current_head", operations)
        self.assertIn("objects:store_records_bulk", operations)
        self.assertIn("objects:topk_search", operations)
        self.assertIn("objects:topk_batch", operations)
        head_entry = dict(operations["objects:load_catalog_current_head"])
        write_entry = dict(operations["objects:store_records_bulk"])
        search_entry = dict(operations["objects:topk_search"])
        batch_entry = dict(operations["objects:topk_batch"])
        self.assertEqual(head_entry["last_request_kind"], "read")
        self.assertEqual(head_entry["last_access_classification"], "catalog_current_head")
        self.assertGreaterEqual(int(dict(head_entry["access_classification_counts"])["catalog_current_head"]), 1)
        self.assertEqual(write_entry["last_request_kind"], "write")
        self.assertGreaterEqual(int(dict(write_entry["request_kind_counts"])["write"]), 1)
        self.assertGreaterEqual(int(dict(write_entry["request_payload_kind_counts"])["fine_grained_record_batch"]), 1)
        self.assertGreaterEqual(int(dict(write_entry["request_payload_kind_counts"])["catalog_segment_record_batch"]), 1)
        self.assertGreaterEqual(int(dict(write_entry["request_payload_kind_counts"])["catalog_current_head_record_batch"]), 1)
        self.assertEqual(search_entry["last_outcome"], "ok")
        self.assertEqual(search_entry["last_classification"], "ok")
        self.assertEqual(search_entry["last_request_kind"], "read")
        self.assertEqual(search_entry["last_access_classification"], "topk_scope_query")
        self.assertEqual(search_entry["last_request_endpoint"], "POST /v1/external/retrieve/topk_records")
        self.assertEqual(search_entry["last_request_payload_kind"], "topk_scope_query")
        self.assertGreaterEqual(int(dict(search_entry["request_endpoint_counts"])["POST /v1/external/retrieve/topk_records"]), 1)
        self.assertGreaterEqual(int(dict(search_entry["request_payload_kind_counts"])["topk_scope_query"]), 1)
        self.assertGreaterEqual(int(dict(search_entry["request_kind_counts"])["read"]), 1)
        self.assertGreaterEqual(int(dict(search_entry["access_classification_counts"])["topk_scope_query"]), 1)
        self.assertGreaterEqual(int(search_entry["total_count"]), 1)
        self.assertEqual(batch_entry["last_outcome"], "ok")
        self.assertEqual(batch_entry["last_classification"], "ok")
        self.assertEqual(batch_entry["last_request_kind"], "read")
        self.assertEqual(batch_entry["last_access_classification"], "retrieve_batch")
        self.assertEqual(batch_entry["last_request_endpoint"], "POST /v1/external/retrieve/topk_records")
        self.assertEqual(batch_entry["last_request_payload_kind"], "topk_allowed_doc_batch")
        self.assertGreaterEqual(int(dict(batch_entry["request_endpoint_counts"])["POST /v1/external/retrieve/topk_records"]), 1)
        self.assertGreaterEqual(int(dict(batch_entry["request_payload_kind_counts"])["topk_allowed_doc_batch"]), 1)
        self.assertGreaterEqual(int(dict(batch_entry["request_kind_counts"])["read"]), 1)
        self.assertGreaterEqual(int(dict(batch_entry["access_classification_counts"])["retrieve_batch"]), 1)
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

    def test_select_open_conflicts_false_empty_catalog_projection_avoids_item_document_fetches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.client = _NoItemFetchEmptyScopeTopKClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
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
            store.write_snapshot(
                objects=(old_phone, new_phone),
                conflicts=(conflict,),
                archived_objects=(),
            )

            conflicts = store.select_open_conflicts(query_text=_TEST_CORINNA_PHONE_NEW, limit=1)

        self.assertEqual(tuple(item.slot_key for item in conflicts), ("contact:person:corinna_maier:phone",))
        self.assertEqual(remote_state.client.item_fetch_attempts, 0)

    def test_retriever_conflict_queue_keeps_query_time_object_hydration_off_documents_full(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.client = _ConflictSelectionProjectionOnlyClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            remote_state.config.project_root = str(project_root)
            store = LongTermStructuredStore(
                base_path=project_root / "state" / "chonkydb",
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
            store.write_snapshot(
                objects=(old_phone, new_phone),
                conflicts=(conflict,),
                archived_objects=(),
            )
            retriever = LongTermRetriever(
                config=TwinrConfig(
                    project_root=str(project_root),
                    personality_dir="personality",
                    memory_markdown_path=str(project_root / "state" / "MEMORY.md"),
                    long_term_memory_path=str(project_root / "state" / "chonkydb"),
                    long_term_memory_recall_limit=2,
                ),
                prompt_context_store=SimpleNamespace(),
                graph_store=SimpleNamespace(),
                object_store=store,
                midterm_store=SimpleNamespace(),
                conflict_resolver=LongTermConflictResolver(),
                subtext_builder=SimpleNamespace(),
            )

            queue = retriever.select_conflict_queue(
                query=LongTermQueryProfile.from_text("Which phone number should I use for Corinna Maier?"),
                limit=1,
            )
            histogram_path = project_root / "artifacts" / "stores" / "ops" / "longterm_remote_read_histograms.json"
            payload = json.loads(histogram_path.read_text(encoding="utf-8"))

        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0].slot_key, "contact:person:corinna_maier:phone")
        self.assertEqual({option.memory_id for option in queue[0].options}, {"fact:corinna_phone_old", "fact:corinna_phone_new"})
        self.assertEqual(remote_state.client.item_fetch_attempts, 0)
        operations = dict(payload.get("operations") or {})
        self.assertIn("objects:load_catalog_current_head", operations)
        self.assertNotIn("objects:fetch_catalog_segment", operations)
        self.assertNotIn("objects:topk_batch", operations)
        self.assertNotIn("objects:retrieve_batch", operations)
        self.assertNotIn("objects:fetch_item_document", operations)

    def test_select_query_time_objects_by_ids_backfills_missing_exact_ids_after_partial_scope_search(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.client = _TermOverlapChonkyClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(
                base_path=project_root / "state" / "chonkydb",
                remote_state=remote_state,
            )
            old_jam = LongTermMemoryObjectV1(
                memory_id="fact:jam_preference_old",
                kind="fact",
                summary="Deine Lieblingsmarmelade ist Erdbeermarmelade.",
                details="Aeltere Vorliebe fuer das Fruehstueck.",
                source=_source(),
                status="active",
                confidence=0.94,
                slot_key="preference:breakfast:jam",
                value_key="strawberry",
            )
            new_jam = LongTermMemoryObjectV1(
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
            conflict = LongTermMemoryConflictV1(
                slot_key="preference:breakfast:jam",
                candidate_memory_id="fact:jam_preference_new",
                existing_memory_ids=("fact:jam_preference_old",),
                question="Welche Marmelade stimmt gerade?",
                reason="Widerspruechliche Marmeladenpraeferenzen liegen vor.",
            )
            store.write_snapshot(
                objects=(old_jam, new_jam),
                conflicts=(conflict,),
                archived_objects=(),
            )
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None
            retriever = LongTermRetriever(
                config=TwinrConfig(
                    project_root=str(project_root),
                    personality_dir="personality",
                    memory_markdown_path=str(project_root / "state" / "MEMORY.md"),
                    long_term_memory_path=str(project_root / "state" / "chonkydb"),
                    long_term_memory_recall_limit=2,
                ),
                prompt_context_store=SimpleNamespace(),
                graph_store=SimpleNamespace(),
                object_store=store,
                midterm_store=SimpleNamespace(),
                conflict_resolver=LongTermConflictResolver(),
                subtext_builder=SimpleNamespace(),
            )

            with patch.object(
                remote_catalog,
                "search_current_item_payloads",
                return_value=(new_jam.to_payload(),),
            ):
                objects = store.select_query_time_objects_by_ids(
                    query_text="Welche Marmeladen stehen gerade im Widerspruch?",
                    memory_ids=("fact:jam_preference_old", "fact:jam_preference_new"),
                )
                queue = retriever.select_conflict_queue(
                    query=LongTermQueryProfile.from_text("Welche Marmeladen stehen gerade im Widerspruch?"),
                    limit=1,
                )

        self.assertEqual(
            tuple(item.memory_id for item in objects),
            ("fact:jam_preference_old", "fact:jam_preference_new"),
        )
        self.assertEqual(len(queue), 1)
        self.assertEqual(
            {option.memory_id for option in queue[0].options},
            {"fact:jam_preference_old", "fact:jam_preference_new"},
        )

    def test_scope_topk_404_object_fallback_keeps_segment_and_item_reads_off_documents_full(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.client = _NoDocumentsFullScope404ChonkyClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            remote_state.config.project_root = str(project_root)
            writer_store = LongTermStructuredStore(
                base_path=project_root / "writer" / "state" / "chonkydb",
                remote_state=remote_state,
            )
            writer_store.write_snapshot(
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
            reader_store = LongTermStructuredStore(
                base_path=project_root / "reader" / "state" / "chonkydb",
                remote_state=remote_state,
            )

            relevant = reader_store.select_relevant_objects(query_text="Janina", limit=1)
            histogram_path = project_root / "artifacts" / "stores" / "ops" / "longterm_remote_read_histograms.json"
            payload = json.loads(histogram_path.read_text(encoding="utf-8"))

        self.assertEqual(tuple(item.memory_id for item in relevant), ("fact:janina_spouse",))
        self.assertEqual(remote_state.client.segment_fetch_attempts, 0)
        self.assertEqual(remote_state.client.item_fetch_attempts, 0)
        operations = dict(payload.get("operations") or {})
        self.assertIn("objects:load_catalog_current_head", operations)
        self.assertGreaterEqual(int(dict(operations["objects:topk_search"]).get("total_count", 0)), 1)
        self.assertNotIn("objects:fetch_catalog_segment", operations)
        self.assertTrue(
            "objects:topk_batch" in operations or "objects:retrieve_batch" in operations
        )

    def test_scope_topk_404_conflict_queue_keeps_query_time_catalog_reads_off_documents_full(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.client = _NoDocumentsFullScope404ChonkyClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            remote_state.config.project_root = str(project_root)
            writer_store = LongTermStructuredStore(
                base_path=project_root / "writer" / "state" / "chonkydb",
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
            reader_store = LongTermStructuredStore(
                base_path=project_root / "reader" / "state" / "chonkydb",
                remote_state=remote_state,
            )
            retriever = LongTermRetriever(
                config=TwinrConfig(
                    project_root=str(project_root),
                    personality_dir="personality",
                    memory_markdown_path=str(project_root / "state" / "MEMORY.md"),
                    long_term_memory_path=str(project_root / "reader" / "state" / "chonkydb"),
                    long_term_memory_recall_limit=2,
                ),
                prompt_context_store=SimpleNamespace(),
                graph_store=SimpleNamespace(),
                object_store=reader_store,
                midterm_store=SimpleNamespace(),
                conflict_resolver=LongTermConflictResolver(),
                subtext_builder=SimpleNamespace(),
            )

            queue = retriever.select_conflict_queue(
                query=LongTermQueryProfile.from_text("Which phone number should I use for Corinna Maier?"),
                limit=1,
            )
            histogram_path = project_root / "artifacts" / "stores" / "ops" / "longterm_remote_read_histograms.json"
            payload = json.loads(histogram_path.read_text(encoding="utf-8"))

        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0].slot_key, "contact:person:corinna_maier:phone")
        self.assertEqual(remote_state.client.segment_fetch_attempts, 0)
        self.assertEqual(remote_state.client.item_fetch_attempts, 0)
        operations = dict(payload.get("operations") or {})
        self.assertNotIn("objects:fetch_catalog_segment", operations)
        self.assertNotIn("conflicts:fetch_catalog_segment", operations)
        self.assertTrue(
            "objects:topk_batch" in operations or "objects:retrieve_batch" in operations
        )
        self.assertTrue(
            "conflicts:topk_batch" in operations or "conflicts:retrieve_batch" in operations
        )

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
                "content": json.dumps(
                    {
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
                    ensure_ascii=False,
                ),
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
        self.assertEqual(client.retrieve_attempts_by_document_id[segment_document_id], 2)
        self.assertEqual(client.fetch_attempts_by_uri.get(segment_uri, 0), 0)

    def test_remote_catalog_segment_batch_topk_reads_content_bearing_live_shape_without_documents_full(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.required = True
            client = _SegmentContentBearingTopKClient()
            remote_state.client = client
            remote_state.read_client = client
            remote_state.write_client = client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None
            segment_document_id = "segment-doc-1"
            segment_uri = remote_catalog._catalog_segment_uri(snapshot_kind="objects", segment_index=0)
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
                "metadata": {
                    "twinr_snapshot_kind": "objects",
                    "twinr_catalog_segment_index": 0,
                    "twinr_catalog_segment_items": 1,
                },
                "content": json.dumps(
                    {
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
                    ensure_ascii=False,
                ),
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
        self.assertEqual(client.segment_batch_include_content_values, [True])
        self.assertEqual(client.segment_fetch_attempts, 0)

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
        self.assertTrue(all(value == 180.0 for value in remote_state.client.bulk_timeout_seconds))

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
            self.assertEqual(remote_catalog._remote_async_job_timeout_s(snapshot_kind="objects"), 180.0)
            self.assertEqual(remote_catalog._async_job_visibility_timeout_s(snapshot_kind="objects"), 188.0)
            self.assertEqual(remote_catalog._remote_async_job_timeout_s(snapshot_kind="conflicts"), 180.0)
            self.assertEqual(remote_catalog._async_job_visibility_timeout_s(snapshot_kind="conflicts"), 188.0)
            self.assertEqual(remote_catalog._remote_async_job_timeout_s(snapshot_kind="archive"), 180.0)
            self.assertEqual(remote_catalog._async_job_visibility_timeout_s(snapshot_kind="archive"), 188.0)
            self.assertEqual(remote_catalog._remote_async_job_timeout_s(snapshot_kind="graph_nodes"), 180.0)
            self.assertEqual(remote_catalog._async_job_visibility_timeout_s(snapshot_kind="graph_nodes"), 188.0)
            self.assertEqual(remote_catalog._remote_async_job_timeout_s(snapshot_kind="graph_edges"), 180.0)
            self.assertEqual(remote_catalog._remote_async_job_timeout_s(snapshot_kind="provider_answer_fronts"), 180.0)
            self.assertEqual(remote_catalog._async_job_visibility_timeout_s(snapshot_kind="provider_answer_fronts"), 188.0)

            remote_state.config.long_term_memory_remote_flush_timeout_s = 5.0
            self.assertEqual(remote_catalog._remote_async_job_timeout_s(), 15.0)
            self.assertEqual(remote_catalog._async_job_visibility_timeout_s(), 23.0)
            self.assertEqual(remote_catalog._async_attestation_visibility_timeout_s(), 30.0)
            self.assertEqual(remote_catalog._remote_async_job_timeout_s(snapshot_kind="objects"), 180.0)
            self.assertEqual(remote_catalog._async_job_visibility_timeout_s(snapshot_kind="objects"), 188.0)
            self.assertEqual(remote_catalog._remote_async_job_timeout_s(snapshot_kind="graph_nodes"), 180.0)
            self.assertEqual(remote_catalog._async_job_visibility_timeout_s(snapshot_kind="graph_nodes"), 188.0)

    def test_provider_answer_front_save_skips_readback_attestation_and_records_phase_breakdown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.config.project_root = str(project_root)
            remote_state.client = _AsyncBulkJobStatusClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermProviderAnswerFrontStore(remote_state)

            store.save_front(
                query_keys=("Was bringt Lea heute Abend vorbei?",),
                context=LongTermMemoryContext(
                    durable_context="Lea bringt heute Abend eine Thermoskanne vorbei.",
                    midterm_context="Lea bringt selbstgemachte Linsensuppe vorbei.",
                ),
            )

            histogram_path = project_root / "artifacts" / "stores" / "ops" / "longterm_remote_read_histograms.json"
            histogram = json.loads(histogram_path.read_text(encoding="utf-8"))

        self.assertEqual(remote_state.client.job_status_calls, [])
        entry = dict(histogram["operations"]["provider_answer_fronts:store_records_bulk"])
        self.assertEqual(entry["last_request_execution_mode"], "async")
        self.assertEqual(entry["async_job_resolution_source_counts"]["skipped_attestation_disabled"], 3)
        self.assertEqual(entry["last_attestation_mode"], "disabled")
        self.assertFalse(entry["last_readback_required"])
        self.assertIsInstance(entry["last_store_transport_ms"], float)
        self.assertIsNone(entry["last_async_job_wait_ms"])
        self.assertIsNone(entry["last_readback_attestation_ms"])
        self.assertEqual(entry["attestation_mode_counts"]["disabled"], 3)

    def test_projection_complete_write_only_skips_item_async_job_status_and_item_uri_attestation_when_document_ids_are_optional(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.client = _AsyncBulkJobStatusTimeoutClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None
            uri = remote_catalog.item_uri(snapshot_kind="objects", item_id="fact:jam_preference")
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

            with patch.object(remote_catalog, "_async_job_visibility_timeout_s", return_value=0.01):
                payload = remote_catalog.build_catalog_payload(
                    snapshot_kind="objects",
                    item_payloads=(obj.to_payload() for obj in objects),
                    item_id_getter=lambda item: item.get("memory_id"),
                    metadata_builder=lambda item: {
                        "summary": item.get("summary"),
                        "selection_projection": dict(item),
                    },
                    content_builder=lambda item: str(item.get("summary") or ""),
                    skip_async_document_id_wait=True,
                )

        segment_uri = str(payload["segments"][0]["uri"])
        self.assertEqual(remote_state.client.job_status_calls, ["job-bulk-1"])
        self.assertEqual(remote_state.client.fetch_attempts_by_uri.get(uri, 0), 0)
        self.assertGreaterEqual(remote_state.client.fetch_attempts_by_uri.get(segment_uri, 0), 1)
        catalog_entries = remote_catalog._load_segmented_catalog_entries(
            definition=remote_catalog._require_definition("objects"),
            payload=payload,
        )
        self.assertEqual(len(catalog_entries), 1)
        self.assertIsNone(catalog_entries[0].document_id)

    def test_write_snapshot_only_skips_async_job_status_for_projection_complete_fine_grained_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.config.project_root = str(project_root)
            remote_state.client = _AsyncBulkJobStatusClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=project_root / "state" / "chonkydb", remote_state=remote_state)
            object_fact = LongTermMemoryObjectV1(
                memory_id="fact:jam_preference",
                kind="fact",
                summary="Du magst Aprikosenmarmelade.",
                source=_source(),
                status="active",
                confidence=0.95,
                slot_key="preference:breakfast:jam",
                value_key="apricot",
            )
            first_conflict = LongTermMemoryConflictV1(
                slot_key="preference:breakfast:jam",
                candidate_memory_id="fact:jam_preference_candidate_1",
                existing_memory_ids=("fact:jam_preference",),
                question="Welche Marmelade stimmt gerade?",
                reason="Eine neue Aussage widerspricht der gespeicherten Vorliebe.",
            )
            second_conflict = LongTermMemoryConflictV1(
                slot_key="preference:breakfast:jam",
                candidate_memory_id="fact:jam_preference_candidate_2",
                existing_memory_ids=("fact:jam_preference",),
                question="Welche Marmelade stimmt gerade?",
                reason="Eine weitere Aussage widerspricht der gespeicherten Vorliebe.",
            )

            store.write_snapshot(
                objects=(object_fact,),
                conflicts=(first_conflict, second_conflict),
                archived_objects=(),
            )
            histogram_path = project_root / "artifacts" / "stores" / "ops" / "longterm_remote_read_histograms.json"
            histogram = json.loads(histogram_path.read_text(encoding="utf-8"))
            fresh_store = LongTermStructuredStore(
                base_path=project_root / "fresh_state" / "chonkydb",
                remote_state=remote_state,
            )

        self.assertEqual(len(remote_state.client.job_status_calls), 5)
        object_entry = dict(histogram["operations"]["objects:store_records_bulk"])
        self.assertEqual(object_entry["async_job_resolution_source_counts"]["skipped_projection_complete"], 1)
        self.assertEqual(object_entry["async_job_resolution_source_counts"]["job_status"], 2)
        self.assertEqual(object_entry["attestation_mode_counts"]["deferred_projection_complete"], 1)
        self.assertEqual(object_entry["attestation_mode_counts"]["exact_document_ids"], 2)
        conflict_entry = dict(histogram["operations"]["conflicts:store_records_bulk"])
        self.assertEqual(conflict_entry["async_job_resolution_source_counts"]["skipped_projection_complete"], 1)
        self.assertEqual(conflict_entry["async_job_resolution_source_counts"]["job_status"], 2)
        self.assertEqual(conflict_entry["attestation_mode_counts"]["deferred_projection_complete"], 1)
        self.assertEqual(conflict_entry["attestation_mode_counts"]["exact_document_ids"], 2)
        archive_entry = dict(histogram["operations"]["archive:store_records_bulk"])
        self.assertEqual(archive_entry["async_job_resolution_source_counts"]["job_status"], 1)
        self.assertEqual(archive_entry["attestation_mode_counts"]["exact_document_ids"], 1)
        loaded_objects = fresh_store.load_objects_fine_grained()
        self.assertEqual(tuple(item.memory_id for item in loaded_objects), ("fact:jam_preference",))
        loaded_conflicts = fresh_store.load_conflicts_fine_grained()
        self.assertEqual(
            tuple(item.candidate_memory_id for item in loaded_conflicts),
            ("fact:jam_preference_candidate_1", "fact:jam_preference_candidate_2"),
        )

    def test_midterm_store_only_skips_async_job_status_for_projection_complete_fine_grained_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.config.project_root = str(project_root)
            remote_state.client = _AsyncBulkJobStatusClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermMidtermStore(base_path=project_root / "state" / "chonkydb", remote_state=remote_state)
            packet = LongTermMidtermPacketV1(
                packet_id="midterm:test_packet",
                kind="restart_recall",
                summary="Lea bringt heute Abend eine Thermoskanne mit Linsensuppe vorbei.",
                updated_at=datetime(2026, 4, 3, 7, 20, 0, tzinfo=timezone.utc),
                attributes={"persistence_scope": "restart_recall"},
            )

            store.save_packets(packets=(packet,))
            histogram_path = project_root / "artifacts" / "stores" / "ops" / "longterm_remote_read_histograms.json"
            histogram = json.loads(histogram_path.read_text(encoding="utf-8"))
            remote_packets = store._remote_midterm.load_packets()

        self.assertEqual(len(remote_state.client.job_status_calls), 2)
        entry = dict(histogram["operations"]["midterm:store_records_bulk"])
        self.assertEqual(entry["async_job_resolution_source_counts"]["skipped_projection_complete"], 1)
        self.assertEqual(entry["async_job_resolution_source_counts"]["job_status"], 2)
        self.assertEqual(entry["attestation_mode_counts"]["deferred_projection_complete"], 1)
        self.assertEqual(entry["attestation_mode_counts"]["exact_document_ids"], 2)
        self.assertEqual(tuple(packet.packet_id for packet in remote_packets), ("midterm:test_packet",))

    def test_midterm_store_keeps_projection_complete_segments_async_but_current_head_sync(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.client = _ProjectionCompleteControlPlaneAsyncFailingClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermMidtermStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            packet = LongTermMidtermPacketV1(
                packet_id="midterm:test_packet",
                kind="restart_recall",
                summary="Lea bringt heute Abend eine Thermoskanne mit Linsensuppe vorbei.",
                updated_at=datetime(2026, 4, 3, 7, 20, 0, tzinfo=timezone.utc),
                attributes={"persistence_scope": "restart_recall"},
            )

            store.save_packets(packets=(packet,))

        midterm_segment_modes = [
            mode
            for mode, uris in zip(
                remote_state.client.bulk_execution_modes,
                remote_state.client.batch_uris,
                strict=False,
            )
            if any("/midterm/catalog/segment/" in uri for uri in uris)
        ]
        current_head_modes = [
            mode
            for mode, uris in zip(
                remote_state.client.bulk_execution_modes,
                remote_state.client.batch_uris,
                strict=False,
            )
            if any(uri.endswith("/catalog/current") for uri in uris)
        ]
        fine_grained_modes = [
            mode
            for mode, uris in zip(
                remote_state.client.bulk_execution_modes,
                remote_state.client.batch_uris,
                strict=False,
            )
            if uris and all("/catalog/" not in uri for uri in uris)
        ]
        self.assertTrue(midterm_segment_modes)
        self.assertTrue(all(mode == "async" for mode in midterm_segment_modes))
        self.assertTrue(current_head_modes)
        self.assertTrue(all(mode == "sync" for mode in current_head_modes))
        self.assertIn("async", fine_grained_modes)

    def test_catalog_segment_write_falls_back_to_same_uri_attestation_after_job_status_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.client = _AsyncBulkJobStatusTimeoutClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None
            definition = remote_catalog._require_definition("objects")
            base_segment_uri = remote_catalog._catalog_segment_uri(snapshot_kind="objects", segment_index=0)

            with patch.object(remote_catalog, "_async_job_visibility_timeout_s", return_value=0.01):
                refs = remote_catalog._persist_catalog_segments(
                    remote_state.write_client,
                    definition=definition,
                    catalog_entries=[
                        remote_catalog._build_catalog_entry(
                            item_id="fact:jam_preference",
                            document_id="doc-existing",
                            metadata={"summary": "Du magst Aprikosenmarmelade."},
                        )
                    ],
                )

        segment_uri = str(refs[0]["uri"])
        self.assertTrue(remote_state.client.job_status_calls)
        self.assertGreaterEqual(remote_state.client.fetch_attempts_by_uri.get(segment_uri, 0), 1)
        self.assertTrue(segment_uri.startswith(base_segment_uri + "/"))
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0]["document_id"], "doc-1")

    def test_catalog_segment_write_uses_versioned_uri_to_avoid_fixed_uri_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.client = _AsyncBulkJobStatusTimeoutClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None
            base_segment_uri = remote_catalog._catalog_segment_uri(snapshot_kind="objects", segment_index=0)
            with patch.object(remote_catalog, "_async_job_visibility_timeout_s", return_value=0.01):
                refs = remote_catalog._persist_catalog_segments(
                    remote_state.write_client,
                    definition=remote_catalog._require_definition("objects"),
                    catalog_entries=[
                        remote_catalog._build_catalog_entry(
                            item_id="fact:jam_preference",
                            document_id="doc-existing",
                            metadata={"summary": "Du magst Aprikosenmarmelade."},
                        )
                    ],
                )

        segment_uri = str(refs[0]["uri"])
        self.assertTrue(remote_state.client.job_status_calls)
        self.assertGreaterEqual(remote_state.client.fetch_attempts_by_uri.get(segment_uri, 0), 1)
        self.assertNotEqual(segment_uri, base_segment_uri)
        self.assertTrue(segment_uri.startswith(base_segment_uri + "/"))
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0]["document_id"], "doc-1")

    def test_catalog_segment_write_prefers_job_status_document_id_over_stale_same_uri_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            clock = {"now": 0.0}
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.client = _AsyncBulkSegmentJobStatusPreferredClient(clock=clock)
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None

            with (
                patch("twinr.memory.longterm.storage._remote_catalog.writes.time.monotonic", side_effect=lambda: clock["now"]),
                patch.object(remote_catalog, "_async_attestation_visibility_timeout_s", return_value=0.5),
                patch.object(remote_catalog, "_async_job_visibility_timeout_s", return_value=0.5),
                patch.object(remote_catalog, "_remote_retry_backoff_s", return_value=0.0),
            ):
                refs = remote_catalog._persist_catalog_segments(
                    remote_state.write_client,
                    definition=remote_catalog._require_definition("objects"),
                    catalog_entries=[
                        remote_catalog._build_catalog_entry(
                            item_id="fact:jam_preference",
                            document_id="doc-existing",
                            metadata={"summary": "Du magst Aprikosenmarmelade."},
                        )
                    ],
                )

        segment_uri = str(refs[0]["uri"])
        self.assertTrue(remote_state.client.job_status_calls)
        self.assertEqual(remote_state.client.fetch_attempts_by_uri.get(segment_uri, 0), 0)
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0]["document_id"], "doc-1")

    def test_projection_complete_catalog_segment_write_still_prefers_job_status_document_id_over_stale_same_uri_history(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            clock = {"now": 0.0}
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.client = _AsyncBulkSegmentJobStatusPreferredClient(clock=clock)
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None

            with (
                patch("twinr.memory.longterm.storage._remote_catalog.writes.time.monotonic", side_effect=lambda: clock["now"]),
                patch.object(remote_catalog, "_async_attestation_visibility_timeout_s", return_value=0.5),
                patch.object(remote_catalog, "_async_job_visibility_timeout_s", return_value=0.5),
                patch.object(remote_catalog, "_remote_retry_backoff_s", return_value=0.0),
            ):
                refs = remote_catalog._persist_catalog_segments(
                    remote_state.write_client,
                    definition=remote_catalog._require_definition("objects"),
                    catalog_entries=[
                        remote_catalog._build_catalog_entry(
                            item_id="fact:jam_preference",
                            document_id="doc-existing",
                            metadata={"summary": "Du magst Aprikosenmarmelade."},
                        )
                    ],
                    skip_async_document_id_wait=True,
                )

        segment_uri = str(refs[0]["uri"])
        self.assertTrue(remote_state.client.job_status_calls)
        self.assertEqual(remote_state.client.fetch_attempts_by_uri.get(segment_uri, 0), 0)
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0]["document_id"], "doc-1")

    def test_catalog_current_head_write_prefers_job_status_document_id_over_stale_same_uri_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            clock = {"now": 0.0}
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.client = _AsyncBulkCurrentHeadJobStatusPreferredClient(clock=clock)
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None
            fresh_payload = {
                "schema": remote_catalog._require_definition("objects").catalog_schema,
                "version": 3,
                "items_count": 0,
                "segments": [],
                "written_at": "2026-03-31T09:00:00+00:00",
            }

            with (
                patch("twinr.memory.longterm.storage._remote_catalog.writes.time.monotonic", side_effect=lambda: clock["now"]),
                patch.object(remote_catalog, "_async_attestation_visibility_timeout_s", return_value=0.5),
                patch.object(remote_catalog, "_async_job_visibility_timeout_s", return_value=0.5),
                patch.object(remote_catalog, "_remote_retry_backoff_s", return_value=0.0),
            ):
                persisted = remote_catalog.persist_catalog_payload(snapshot_kind="objects", payload=fresh_payload)

        current_head_uri = remote_catalog._catalog_head_uri(snapshot_kind="objects")
        self.assertEqual(persisted["written_at"], "2026-03-31T09:00:00+00:00")
        self.assertTrue(remote_state.client.job_status_calls)
        self.assertEqual(remote_state.client.fetch_attempts_by_document_id.get("doc-1", 0), 0)
        self.assertEqual(remote_state.client.fetch_attempts_by_uri.get(current_head_uri, 0), 0)

    def test_non_projection_complete_write_falls_back_to_uri_attestation_when_async_job_status_times_out(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.client = _AsyncBulkJobStatusTimeoutClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None
            uri = remote_catalog.item_uri(snapshot_kind="midterm", item_id="packet:test")
            record_item = ChonkyDBRecordItem(
                payload={
                    "schema": "twinr_memory_midterm_packet_record_v1",
                    "version": 1,
                    "snapshot_kind": "midterm",
                    "item_id": "packet:test",
                    "packet": {
                        "packet_id": "packet:test",
                        "summary": "Lea mochte die rote Thermoskanne.",
                    },
                },
                metadata={
                    "twinr_snapshot_kind": "midterm",
                    "twinr_memory_item_id": "packet:test",
                    "twinr_payload": {
                        "packet_id": "packet:test",
                        "summary": "Lea mochte die rote Thermoskanne.",
                    },
                },
                content=json.dumps({"packet_id": "packet:test", "summary": "Lea mochte die rote Thermoskanne."}),
                uri=uri,
                enable_chunking=False,
                include_insights_in_response=False,
            )

            with patch.object(remote_catalog, "_async_job_visibility_timeout_s", return_value=0.01):
                document_ids = remote_catalog._store_record_items(
                    remote_state.write_client,
                    snapshot_kind="midterm",
                    record_items=[record_item],
                )

        self.assertEqual(remote_state.client.job_status_calls, ["job-bulk-1"])
        self.assertEqual(remote_state.client.fetch_attempts_by_uri.get(uri, 0), 1)
        self.assertEqual(document_ids, ("doc-1",))

    def test_remote_catalog_current_head_attestation_uses_flush_window_for_same_uri_visibility(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.config.long_term_memory_remote_retry_attempts = 1
            remote_state.config.long_term_memory_remote_retry_backoff_s = 0.0
            remote_state.client = _AsyncBulkJobStatusNoDocumentIdsClient(stale_reads_before_visible=2)
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None
            current_head_uri = remote_catalog._catalog_head_uri(snapshot_kind="objects")
            stale_payload = {
                "schema": remote_catalog._require_definition("objects").catalog_schema,
                "version": 3,
                "items_count": 0,
                "segments": [],
                "written_at": "2026-03-30T09:00:00+00:00",
            }
            stale_record = {
                "document_id": "doc-stale",
                "payload": dict(stale_payload),
                "metadata": {
                    "twinr_snapshot_kind": "objects",
                    "twinr_catalog_current_head": True,
                    "twinr_payload": dict(stale_payload),
                },
                "content": json.dumps(stale_payload, ensure_ascii=False),
                "uri": current_head_uri,
            }
            remote_state.client.records_by_document_id["doc-stale"] = dict(stale_record)
            remote_state.client.records_by_uri[current_head_uri] = dict(stale_record)
            fresh_payload = {
                "schema": remote_catalog._require_definition("objects").catalog_schema,
                "version": 3,
                "items_count": 0,
                "segments": [],
                "written_at": "2026-03-31T09:00:00+00:00",
            }

            with (
                patch.object(remote_catalog, "_async_attestation_visibility_timeout_s", return_value=0.05),
                patch.object(remote_catalog, "_async_job_visibility_timeout_s", return_value=0.15),
            ):
                persisted = remote_catalog.persist_catalog_payload(snapshot_kind="objects", payload=fresh_payload)

        self.assertEqual(persisted["written_at"], "2026-03-31T09:00:00+00:00")
        self.assertEqual(remote_state.client.job_status_calls, ["job-bulk-1"])
        self.assertGreaterEqual(remote_state.client.fetch_attempts_by_uri[current_head_uri], 3)
        self.assertEqual(
            dict(remote_state.client.records_by_uri[current_head_uri]).get("payload", {}).get("written_at"),
            "2026-03-31T09:00:00+00:00",
        )

    def test_remote_catalog_write_aborts_retry_when_shutdown_requested(self) -> None:
        class _RateLimitedCurrentHeadClient(_FakeChonkyClient):
            def __init__(self, stop_event: Event) -> None:
                super().__init__()
                self._stop_event = stop_event

            def store_records_bulk(self, request):
                del request
                self.bulk_calls += 1
                self._stop_event.set()
                raise ChonkyDBError(
                    "ChonkyDB request failed for POST /v1/external/records/bulk",
                    status_code=429,
                    response_json={
                        "detail": "queue_saturated",
                        "error": "queue_saturated",
                        "error_type": "RuntimeError",
                    },
                )

        stop_event = Event()
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.config.long_term_memory_remote_retry_attempts = 4
            remote_state.config.long_term_memory_remote_retry_backoff_s = 5.0
            client = _RateLimitedCurrentHeadClient(stop_event)
            remote_state.client = client
            remote_state.read_client = client
            remote_state.write_client = client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None
            fresh_payload = {
                "schema": remote_catalog._require_definition("objects").catalog_schema,
                "version": 3,
                "items_count": 0,
                "segments": [],
                "written_at": "2026-04-03T08:30:00+00:00",
            }

            with self.assertRaises(LongTermOperationCancelledError):
                with longterm_operation_abort_scope(
                    should_abort=stop_event.is_set,
                    wait_for_abort=stop_event.wait,
                    label="test_remote_catalog_write_aborts_retry_when_shutdown_requested",
                ):
                    remote_catalog.persist_catalog_payload(snapshot_kind="objects", payload=fresh_payload)

        self.assertEqual(client.bulk_calls, 1)

    def test_remote_catalog_attestation_caps_read_timeout_to_remaining_visibility_window(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.client = _TimeoutCloningReadClient(timeout_s=8.0)
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None
            head_payload = {
                "schema": remote_catalog._require_definition("objects").catalog_schema,
                "version": 3,
                "items_count": 0,
                "segments": [],
            }
            record_item = ChonkyDBRecordItem(
                payload=dict(head_payload),
                metadata={
                    "twinr_snapshot_kind": "objects",
                    "twinr_catalog_current_head": True,
                    "twinr_payload": dict(head_payload),
                },
                content=json.dumps(head_payload, ensure_ascii=False),
                uri=remote_catalog._catalog_head_uri(snapshot_kind="objects"),
                enable_chunking=False,
                include_insights_in_response=False,
            )
            monotonic_values = iter((0.0, 0.02, 0.04, 0.06))

            with (
                patch("twinr.memory.longterm.storage._remote_catalog.writes.time.monotonic", side_effect=lambda: next(monotonic_values)),
                patch.object(remote_catalog, "_async_attestation_visibility_timeout_s", return_value=0.05),
                patch.object(remote_catalog, "_async_job_visibility_timeout_s", return_value=0.05),
                patch.object(remote_catalog, "_remote_retry_backoff_s", return_value=0.0),
            ):
                with self.assertRaises(LongTermRemoteUnavailableError):
                    remote_catalog._attest_record_readback(
                        remote_state.read_client,
                        snapshot_kind="objects",
                        record_item=record_item,
                        document_id=None,
                    )

        self.assertEqual(len(remote_state.client.fetch_timeout_history), 1)
        self.assertEqual(len(remote_state.client.clone_timeout_history), 1)
        self.assertLessEqual(remote_state.client.fetch_timeout_history[0], 0.1)
        self.assertLessEqual(remote_state.client.clone_timeout_history[0], 0.1)
        self.assertLess(remote_state.client.fetch_timeout_history[0], remote_state.client.config.timeout_s)
        self.assertLess(remote_state.client.clone_timeout_history[0], remote_state.client.config.timeout_s)

    def test_commit_active_delta_floors_async_write_transport_timeout_to_flush_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.client = _TimeoutCloningWriteClient(timeout_s=15.0)
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            relationship = LongTermMemoryObjectV1(
                memory_id="fact:janina_wife_live",
                kind="fact",
                summary="Janina is the user's wife.",
                source=_source(),
                status="active",
                confidence=0.98,
                slot_key="relationship:user:main:wife",
                value_key="person:janina",
            )
            appointment = LongTermMemoryObjectV1(
                memory_id="event:janina_eye_laser_live",
                kind="event",
                summary="Janina has eye laser treatment at the eye doctor on 2026-04-04.",
                source=_source(),
                status="active",
                confidence=0.93,
                slot_key="event:person:janina:eye_laser_treatment:2026-04-04",
                value_key="event:janina_eye_laser_2026_04_04",
                valid_from="2026-04-04",
                valid_to="2026-04-04",
            )

            store.commit_active_delta(object_upserts=(relationship, appointment))

        self.assertEqual(remote_state.client.config.timeout_s, 15.0)
        self.assertEqual(remote_state.client.clone_timeout_history, [60.0, 60.0])
        self.assertEqual(remote_state.client.store_timeout_history, [60.0, 60.0])

    def test_commit_active_delta_splits_rate_limited_async_batch_into_single_item_retries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.client = _AsyncBulkQueueSaturatedSplitClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=root / "writer" / "state" / "chonkydb", remote_state=remote_state)
            relationship = LongTermMemoryObjectV1(
                memory_id="fact:janina_wife_live",
                kind="fact",
                summary="Janina is the user's wife.",
                source=_source(),
                status="active",
                confidence=0.98,
                slot_key="relationship:user:main:wife",
                value_key="person:janina",
            )
            appointment = LongTermMemoryObjectV1(
                memory_id="event:janina_eye_laser_live",
                kind="event",
                summary="Janina has eye laser treatment at the eye doctor on 2026-04-04.",
                source=_source(),
                status="active",
                confidence=0.93,
                slot_key="event:person:janina:eye_laser_treatment:2026-04-04",
                value_key="event:janina_eye_laser_2026_04_04",
                valid_from="2026-04-04",
                valid_to="2026-04-04",
            )

            store.commit_active_delta(object_upserts=(relationship, appointment))
            reader_store = LongTermStructuredStore(base_path=root / "reader" / "state" / "chonkydb", remote_state=remote_state)
            remote_catalog = reader_store._remote_catalog
            assert remote_catalog is not None
            payloads = remote_catalog.load_selection_item_payloads(
                snapshot_kind="objects",
                item_ids=("fact:janina_wife_live", "event:janina_eye_laser_live"),
            )

        async_item_counts = [
            item_count
            for item_count, mode in zip(
                remote_state.client.bulk_request_item_counts,
                remote_state.client.bulk_execution_modes,
                strict=False,
            )
            if mode == "async"
        ]
        self.assertEqual(async_item_counts[0], 2)
        self.assertGreaterEqual(async_item_counts.count(1), 3)
        self.assertEqual(len(payloads), 2)

    def test_remote_catalog_readback_attestation_retries_known_document_id_without_same_uri_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.config.long_term_memory_remote_retry_backoff_s = 0.0
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None
            client = _AsyncBulkDocumentIdEventuallyVisibleClient(stale_document_reads_before_visible=2)
            remote_state.client = client
            remote_state.read_client = client
            remote_state.write_client = client
            uri = remote_catalog.item_uri(snapshot_kind="objects", item_id="fact:jam_preference")
            payload = LongTermMemoryObjectV1(
                memory_id="fact:jam_preference",
                kind="fact",
                summary="Du magst Aprikosenmarmelade.",
                source=_source(),
                status="active",
                confidence=0.95,
                slot_key="preference:breakfast:jam",
                value_key="apricot",
            ).to_payload()
            client._job_records_by_id["job-bulk-1"] = [
                {
                    "document_id": "doc-1",
                    "payload": {
                        "schema": "twinr_memory_object_record_v2",
                        "version": 1,
                        "snapshot_kind": "objects",
                        "item_id": "fact:jam_preference",
                        "object": dict(payload),
                        "metadata": {
                            "twinr_snapshot_kind": "objects",
                            "twinr_memory_item_id": "fact:jam_preference",
                        },
                        "content": payload["summary"],
                    },
                    "metadata": {
                        "twinr_snapshot_kind": "objects",
                        "twinr_memory_item_id": "fact:jam_preference",
                    },
                    "content": payload["summary"],
                    "uri": uri,
                }
            ]

            attested_document_id = remote_catalog._attest_record_readback(
                client,
                snapshot_kind="objects",
                record_item=SimpleNamespace(
                    payload=dict(client._job_records_by_id["job-bulk-1"][0]["payload"]),
                    metadata=dict(client._job_records_by_id["job-bulk-1"][0]["metadata"]),
                    content=payload["summary"],
                    uri=uri,
                ),
                document_id="doc-1",
            )

        self.assertEqual(attested_document_id, "doc-1")
        self.assertGreaterEqual(client.fetch_attempts_by_document_id["doc-1"], 3)
        self.assertEqual(client.fetch_attempts_by_uri.get(uri, 0), 0)

    def test_catalog_segment_attestation_accepts_same_uri_payload_match_without_document_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.config.long_term_memory_remote_retry_attempts = 1
            remote_state.config.long_term_memory_remote_retry_backoff_s = 0.0
            remote_state.client = _FakeChonkyClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None
            definition = remote_catalog._require_definition("graph_nodes")
            uri = remote_catalog._catalog_segment_uri(
                snapshot_kind="graph_nodes",
                segment_index=0,
                segment_token="segmenttoken1234567890abcd",
            )
            segment_payload = {
                "schema": definition.segment_schema,
                "version": 1,
                "snapshot_kind": "graph_nodes",
                "segment_index": 0,
                "items": [
                    {
                        "item_id": "brand:melitta",
                        "document_id": None,
                        "kind": "brand",
                        "summary": "Melitta",
                        "created_at": "2026-04-03T17:41:24+00:00",
                        "updated_at": "2026-04-03T17:41:24+00:00",
                        "payload_sha256": "182cd308e452f5c47c898b02367ca5d40e00a69ffffd313a21dd5a8409e8b7d6",
                        "selection_projection": {
                            "id": "brand:melitta",
                            "type": "brand",
                            "label": "Melitta",
                            "attributes": {"category": "brand"},
                            "status": "active",
                        },
                    }
                ],
            }
            remote_state.client.records_by_uri[uri] = {
                "success": True,
                "origin_uri": uri,
                "metadata": {
                    "twinr_snapshot_kind": "graph_nodes",
                    "twinr_catalog_segment_index": 0,
                    "twinr_catalog_segment_items": 1,
                    "twinr_catalog_segment_token": "segmenttoken1234567890abcd",
                },
                "content": json.dumps(segment_payload, ensure_ascii=False),
            }
            record_item = ChonkyDBRecordItem(
                payload=dict(segment_payload),
                metadata={
                    "twinr_snapshot_kind": "graph_nodes",
                    "twinr_catalog_segment_index": 0,
                    "twinr_catalog_segment_items": 1,
                    "twinr_catalog_segment_token": "segmenttoken1234567890abcd",
                },
                content=json.dumps(segment_payload, ensure_ascii=False),
                uri=uri,
                enable_chunking=False,
                include_insights_in_response=False,
            )

            attested_document_id = remote_catalog._attest_record_readback(
                remote_state.read_client,
                snapshot_kind="graph_nodes",
                record_item=record_item,
                document_id=None,
            )

        self.assertIsNone(attested_document_id)
        self.assertEqual(remote_state.client.fetch_full_document_calls, 1)

    def test_projection_complete_write_skips_item_origin_attestation_when_async_job_omits_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.config.long_term_memory_remote_retry_attempts = 1
            remote_state.config.long_term_memory_remote_retry_backoff_s = 0.0
            remote_state.client = _AsyncBulkJobStatusNoDocumentIdsItemOrigin404Client(stale_reads_before_visible=0)
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            writer_store = LongTermStructuredStore(base_path=root / "writer" / "state" / "chonkydb", remote_state=remote_state)
            writer_store.write_snapshot(
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
                ),
                conflicts=(),
                archived_objects=(),
            )
            reader_store = LongTermStructuredStore(base_path=root / "reader" / "state" / "chonkydb", remote_state=remote_state)
            remote_catalog = reader_store._remote_catalog
            assert remote_catalog is not None
            item_uri = remote_catalog.item_uri(snapshot_kind="objects", item_id="fact:jam_preference_old")
            entries = remote_catalog.load_catalog_entries(snapshot_kind="objects")
            payloads = remote_catalog.load_selection_item_payloads(
                snapshot_kind="objects",
                item_ids=("fact:jam_preference_old",),
            )

        self.assertEqual(remote_state.client.item_origin_404_attempts.get(item_uri, 0), 0)
        self.assertEqual(len(entries), 1)
        self.assertIsNone(entries[0].document_id)
        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["memory_id"], "fact:jam_preference_old")
        self.assertEqual(payloads[0]["summary"], "Deine Lieblingsmarmelade ist Erdbeermarmelade.")

    def test_midterm_write_still_requires_item_origin_attestation_when_async_job_omits_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.config.long_term_memory_remote_retry_attempts = 1
            remote_state.config.long_term_memory_remote_retry_backoff_s = 0.0
            remote_state.client = _AsyncBulkJobStatusNoDocumentIdsItemOrigin404Client(stale_reads_before_visible=0)
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None
            item_uri = remote_catalog.item_uri(snapshot_kind="midterm", item_id="packet:test")
            record_item = ChonkyDBRecordItem(
                payload={
                    "schema": "twinr_memory_midterm_packet_record_v1",
                    "version": 1,
                    "snapshot_kind": "midterm",
                    "item_id": "packet:test",
                    "packet": {
                        "packet_id": "packet:test",
                        "summary": "Lea mochte die rote Thermoskanne.",
                    },
                },
                metadata={
                    "twinr_snapshot_kind": "midterm",
                    "twinr_memory_item_id": "packet:test",
                    "twinr_payload": {
                        "packet_id": "packet:test",
                        "summary": "Lea mochte die rote Thermoskanne.",
                    },
                },
                content=json.dumps({"packet_id": "packet:test", "summary": "Lea mochte die rote Thermoskanne."}),
                uri=item_uri,
                enable_chunking=False,
                include_insights_in_response=False,
            )

            with self.assertRaises(LongTermRemoteUnavailableError):
                remote_catalog._store_record_items(
                    remote_state.write_client,
                    snapshot_kind="midterm",
                    record_items=[record_item],
                )

        self.assertGreaterEqual(remote_state.client.item_origin_404_attempts.get(item_uri, 0), 1)

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
        self.assertEqual(remote_state.client.bulk_execution_modes, ["async", "async", "async"])
        self.assertTrue(events)
        self.assertEqual(events[-1]["event"], "longterm_remote_write_failed")

    def test_write_snapshot_uses_sync_for_tiny_single_conflict_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.client = _ConflictAsyncFailingClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            conflict = LongTermMemoryConflictV1(
                slot_key="preference:breakfast:jam",
                candidate_memory_id="fact:jam_preference_new",
                existing_memory_ids=("fact:jam_preference_old",),
                question="Welche Marmelade stimmt gerade?",
                reason="Widerspruechliche Marmeladenpraeferenzen liegen vor.",
            )

            store.write_snapshot(objects=(), conflicts=(conflict,), archived_objects=())

        conflict_modes = [
            mode
            for mode, snapshot_kinds in zip(
                remote_state.client.bulk_execution_modes,
                remote_state.client.batch_snapshot_kinds,
                strict=False,
            )
            if "conflicts" in snapshot_kinds
        ]
        self.assertTrue(conflict_modes)
        self.assertTrue(all(mode == "sync" for mode in conflict_modes))

    def test_write_snapshot_keeps_projection_complete_segments_async_but_current_head_sync(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.client = _ProjectionCompleteControlPlaneAsyncFailingClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
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
            conflict = LongTermMemoryConflictV1(
                slot_key="preference:breakfast:jam",
                candidate_memory_id="fact:jam_preference_new",
                existing_memory_ids=("fact:jam_preference_old",),
                question="Welche Marmelade stimmt gerade?",
                reason="Widerspruechliche Marmeladenpraeferenzen liegen vor.",
            )

            store.write_snapshot(objects=(object_fact,), conflicts=(conflict,), archived_objects=())

        object_segment_modes = [
            mode
            for mode, uris in zip(
                remote_state.client.bulk_execution_modes,
                remote_state.client.batch_uris,
                strict=False,
            )
            if any("/objects/catalog/segment/" in uri for uri in uris)
        ]
        conflict_segment_modes = [
            mode
            for mode, uris in zip(
                remote_state.client.bulk_execution_modes,
                remote_state.client.batch_uris,
                strict=False,
            )
            if any("/conflicts/catalog/segment/" in uri for uri in uris)
        ]
        current_head_modes = [
            mode
            for mode, uris in zip(
                remote_state.client.bulk_execution_modes,
                remote_state.client.batch_uris,
                strict=False,
            )
            if any(uri.endswith("/catalog/current") for uri in uris)
        ]
        fine_grained_modes = [
            mode
            for mode, uris in zip(
                remote_state.client.bulk_execution_modes,
                remote_state.client.batch_uris,
                strict=False,
            )
            if uris and all("/catalog/" not in uri for uri in uris)
        ]
        self.assertTrue(object_segment_modes)
        self.assertTrue(all(mode == "async" for mode in object_segment_modes))
        self.assertTrue(conflict_segment_modes)
        self.assertTrue(all(mode == "sync" for mode in conflict_segment_modes))
        self.assertTrue(current_head_modes)
        self.assertTrue(all(mode == "sync" for mode in current_head_modes))
        self.assertIn("async", fine_grained_modes)

    def test_commit_active_delta_skips_item_origin_attestation_when_async_job_omits_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.config.long_term_memory_remote_retry_attempts = 1
            remote_state.config.long_term_memory_remote_retry_backoff_s = 0.0
            remote_state.client = _AsyncBulkJobStatusNoDocumentIdsItemOrigin404Client(stale_reads_before_visible=0)
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=root / "writer" / "state" / "chonkydb", remote_state=remote_state)
            updated_object = LongTermMemoryObjectV1(
                memory_id="fact:janina_phone_new",
                kind="fact",
                summary="Janina's new phone number ends with 44.",
                details="Confirmed during the latest clarification turn.",
                source=LongTermSourceRefV1(
                    source_type="conversation_turn",
                    event_ids=("turn:janina-phone-20260330",),
                ),
                status="active",
                confidence=0.99,
                confirmed_by_user=True,
                slot_key="contact:janina:phone",
                value_key="ends_with_44",
            )

            store.commit_active_delta(object_upserts=(updated_object,))
            reader_store = LongTermStructuredStore(base_path=root / "reader" / "state" / "chonkydb", remote_state=remote_state)
            remote_catalog = reader_store._remote_catalog
            assert remote_catalog is not None
            item_uri = remote_catalog.item_uri(snapshot_kind="objects", item_id="fact:janina_phone_new")
            entries = remote_catalog.load_catalog_entries(snapshot_kind="objects")
            payloads = remote_catalog.load_selection_item_payloads(
                snapshot_kind="objects",
                item_ids=("fact:janina_phone_new",),
            )

        self.assertEqual(len(remote_state.client.job_status_calls), 4)
        self.assertEqual(remote_state.client.item_origin_404_attempts.get(item_uri, 0), 0)
        self.assertEqual(len(entries), 1)
        self.assertIsNone(entries[0].document_id)
        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["memory_id"], "fact:janina_phone_new")

    def test_commit_active_delta_keeps_projection_complete_segments_async_but_current_head_sync(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.client = _ProjectionCompleteControlPlaneAsyncFailingClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            updated_object = LongTermMemoryObjectV1(
                memory_id="fact:janina_phone_new",
                kind="fact",
                summary="Janina's new phone number ends with 44.",
                details="Confirmed during the latest clarification turn.",
                source=LongTermSourceRefV1(
                    source_type="conversation_turn",
                    event_ids=("turn:janina-phone-20260330",),
                ),
                status="active",
                confidence=0.99,
                confirmed_by_user=True,
                slot_key="contact:janina:phone",
                value_key="ends_with_44",
            )

            store.commit_active_delta(object_upserts=(updated_object,))

        segment_modes = [
            mode
            for mode, uris in zip(
                remote_state.client.bulk_execution_modes,
                remote_state.client.batch_uris,
                strict=False,
            )
            if any("/catalog/segment/" in uri for uri in uris)
        ]
        current_head_modes = [
            mode
            for mode, uris in zip(
                remote_state.client.bulk_execution_modes,
                remote_state.client.batch_uris,
                strict=False,
            )
            if any(uri.endswith("/catalog/current") for uri in uris)
        ]
        fine_grained_modes = [
            mode
            for mode, uris in zip(
                remote_state.client.bulk_execution_modes,
                remote_state.client.batch_uris,
                strict=False,
            )
            if uris and all("/catalog/" not in uri for uri in uris)
        ]
        self.assertTrue(segment_modes)
        self.assertTrue(all(mode == "async" for mode in segment_modes))
        self.assertTrue(current_head_modes)
        self.assertTrue(all(mode == "sync" for mode in current_head_modes))
        self.assertIn("async", fine_grained_modes)

    def test_commit_active_delta_treats_fast_probe_not_found_as_empty_current_head(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.required = True
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None
            updated_object = LongTermMemoryObjectV1(
                memory_id="fact:janina_phone_new",
                kind="fact",
                summary="Janina's new phone number ends with 44.",
                details="Confirmed during the latest clarification turn.",
                source=LongTermSourceRefV1(
                    source_type="conversation_turn",
                    event_ids=("turn:janina-phone-20260330",),
                ),
                status="active",
                confidence=0.99,
                confirmed_by_user=True,
                slot_key="contact:janina:phone",
                value_key="ends_with_44",
            )
            probe_calls: list[dict[str, object]] = []

            def _probe_catalog_payload_result(*, snapshot_kind: str, fast_fail: bool = False):
                probe_calls.append({"snapshot_kind": snapshot_kind, "fast_fail": fast_fail})
                return ("not_found", None)

            with (
                patch.object(
                    remote_catalog,
                    "_load_catalog_entries_for_write",
                    side_effect=LongTermRemoteUnavailableError("current head unavailable"),
                ),
                patch.object(
                    remote_catalog,
                    "probe_catalog_payload_result",
                    new=_probe_catalog_payload_result,
                ),
            ):
                store.commit_active_delta(object_upserts=(updated_object,))
                loaded = store.load_objects_by_ids(("fact:janina_phone_new",))

        self.assertIn({"snapshot_kind": "objects", "fast_fail": True}, probe_calls)
        self.assertEqual(tuple(item.memory_id for item in loaded), ("fact:janina_phone_new",))

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
        objects_head = _current_catalog_head_payload(store, snapshot_kind="objects")
        conflicts_head = _current_catalog_head_payload(store, snapshot_kind="conflicts")
        archive_head = _current_catalog_head_payload(store, snapshot_kind="archive")
        assert objects_head is not None
        assert conflicts_head is not None
        assert archive_head is not None
        self.assertEqual(objects_head["schema"], "twinr_memory_object_catalog_v3")
        self.assertEqual(objects_head["version"], 3)
        self.assertEqual(objects_head["items_count"], 0)
        self.assertEqual(objects_head["segments"], [])
        self.assertIn("written_at", objects_head)
        self.assertEqual(conflicts_head["schema"], "twinr_memory_conflict_catalog_v3")
        self.assertEqual(conflicts_head["version"], 3)
        self.assertEqual(conflicts_head["items_count"], 0)
        self.assertEqual(conflicts_head["segments"], [])
        self.assertIn("written_at", conflicts_head)
        self.assertEqual(archive_head["schema"], "twinr_memory_archive_catalog_v3")
        self.assertEqual(archive_head["version"], 3)
        self.assertEqual(archive_head["items_count"], 0)
        self.assertEqual(archive_head["segments"], [])
        self.assertIn("written_at", archive_head)
        self.assertEqual(remote_state.snapshots, {})

    def test_ensure_remote_snapshots_bootstraps_empty_remote_documents_for_fresh_required_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.required = True
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)

            ensured = store.ensure_remote_snapshots()

        self.assertEqual(set(ensured), {"objects", "conflicts", "archive"})
        objects_head = _current_catalog_head_payload(store, snapshot_kind="objects")
        conflicts_head = _current_catalog_head_payload(store, snapshot_kind="conflicts")
        archive_head = _current_catalog_head_payload(store, snapshot_kind="archive")
        assert objects_head is not None
        assert conflicts_head is not None
        assert archive_head is not None
        self.assertEqual(objects_head["schema"], "twinr_memory_object_catalog_v3")
        self.assertEqual(objects_head["items_count"], 0)
        self.assertEqual(objects_head["segments"], [])
        self.assertEqual(conflicts_head["schema"], "twinr_memory_conflict_catalog_v3")
        self.assertEqual(conflicts_head["items_count"], 0)
        self.assertEqual(conflicts_head["segments"], [])
        self.assertIn("written_at", conflicts_head)
        self.assertEqual(archive_head["schema"], "twinr_memory_archive_catalog_v3")
        self.assertEqual(archive_head["items_count"], 0)
        self.assertEqual(archive_head["segments"], [])
        self.assertIn("written_at", archive_head)
        self.assertEqual(remote_state.snapshots, {})
        self.assertFalse(store.objects_path.exists())

    def test_structured_store_readiness_bootstrap_keeps_fresh_required_namespace_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.client = _CurrentHead404Client()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            load_snapshot_calls: list[str] = []

            def _load_snapshot(*, snapshot_kind: str, local_path=None):
                del local_path
                load_snapshot_calls.append(snapshot_kind)
                raise AssertionError("Fresh readiness bootstrap must not revive legacy snapshot blob reads.")

            remote_state.load_snapshot = _load_snapshot  # type: ignore[assignment]
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)

            ensured = store.ensure_remote_snapshots_for_readiness()
            objects_payload = store.probe_remote_current_snapshot_for_readiness(snapshot_kind="objects")
            conflicts_payload = store.probe_remote_current_snapshot_for_readiness(snapshot_kind="conflicts")
            archive_payload = store.probe_remote_current_snapshot_for_readiness(snapshot_kind="archive")

        self.assertEqual(ensured, ())
        assert objects_payload is not None
        assert conflicts_payload is not None
        assert archive_payload is not None
        self.assertEqual(objects_payload["schema"], "twinr_memory_object_store")
        self.assertEqual(objects_payload["objects"], [])
        self.assertEqual(conflicts_payload["schema"], "twinr_memory_conflict_store")
        self.assertEqual(conflicts_payload["conflicts"], [])
        self.assertEqual(archive_payload["schema"], "twinr_memory_archive_store")
        self.assertEqual(archive_payload["objects"], [])
        self.assertEqual(objects_payload["written_at"], "1970-01-01T00:00:00+00:00")
        self.assertEqual(conflicts_payload["written_at"], "1970-01-01T00:00:00+00:00")
        self.assertEqual(archive_payload["written_at"], "1970-01-01T00:00:00+00:00")
        self.assertEqual(remote_state.client.bulk_calls, 0)
        self.assertEqual(load_snapshot_calls, [])
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
        self.assertEqual(collapsed_load_calls, [])
        self.assertEqual(remote_state.max_concurrent_loads, 0)

    def test_ensure_remote_snapshots_reuses_probed_legacy_catalog_heads_without_hydration_or_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _ProbeOnlyCatalogRemoteState()
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)

            ensured = store.ensure_remote_snapshots()

        self.assertEqual(ensured, ())
        self.assertEqual(remote_state.load_calls, [])
        self.assertEqual(
            remote_state.probe_calls,
            [
                {"snapshot_kind": "objects", "prefer_cached_document_id": False, "prefer_metadata_only": True, "fast_fail": True},
                {"snapshot_kind": "conflicts", "prefer_cached_document_id": False, "prefer_metadata_only": True, "fast_fail": True},
                {"snapshot_kind": "archive", "prefer_cached_document_id": False, "prefer_metadata_only": True, "fast_fail": True},
            ],
        )
        self.assertEqual(remote_state.client.bulk_calls, 0)
        self.assertEqual(_count_remote_records_with_schema(remote_state.client, "twinr_memory_object_catalog_v3"), 0)
        self.assertEqual(_count_remote_records_with_schema(remote_state.client, "twinr_memory_conflict_catalog_v3"), 0)
        self.assertEqual(_count_remote_records_with_schema(remote_state.client, "twinr_memory_archive_catalog_v3"), 0)

    def test_ensure_and_write_snapshot_keep_catalog_current_heads_off_legacy_snapshot_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.config.project_root = str(project_root)
            store = LongTermStructuredStore(
                base_path=project_root / "state" / "chonkydb",
                remote_state=remote_state,
            )
            object_fact = LongTermMemoryObjectV1(
                memory_id="fact:janina_spouse",
                kind="relationship_fact",
                summary="Janina is the user's wife.",
                source=_source(),
                status="active",
                confidence=0.98,
            )
            conflict = LongTermMemoryConflictV1(
                slot_key="contact:person:corinna_maier:phone",
                candidate_memory_id="fact:corinna_phone_new",
                existing_memory_ids=("fact:corinna_phone_old",),
                question="Which phone number should I use for Corinna Maier?",
                reason="Conflicting phone numbers exist.",
            )

            store.ensure_remote_snapshots()
            store.write_snapshot(objects=(object_fact,), conflicts=(conflict,), archived_objects=())
            histogram_path = project_root / "artifacts" / "stores" / "ops" / "longterm_remote_read_histograms.json"
            payload = json.loads(histogram_path.read_text(encoding="utf-8"))

        operations = dict(payload.get("operations") or {})
        self.assertNotIn("objects:snapshot_load", operations)
        self.assertNotIn("conflicts:snapshot_load", operations)
        self.assertNotIn("archive:snapshot_load", operations)
        self.assertNotIn("objects:store_snapshot_record", operations)
        self.assertNotIn("conflicts:store_snapshot_record", operations)
        self.assertNotIn("archive:store_snapshot_record", operations)
        self.assertIn("objects:store_records_bulk", operations)
        self.assertIn("conflicts:store_records_bulk", operations)
        self.assertIn("archive:store_records_bulk", operations)

    def test_load_catalog_payload_prefers_newest_current_head_chunk_when_origin_lookup_returns_multiple_versions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.required = True
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None
            current_head_uri = _current_catalog_head_uri(store, snapshot_kind="conflicts")
            older_payload = {
                "schema": "twinr_memory_conflict_catalog_v3",
                "version": 3,
                "items_count": 0,
                "segments": [],
                "written_at": "2026-03-29T16:00:00+00:00",
            }
            newer_payload = {
                "schema": "twinr_memory_conflict_catalog_v3",
                "version": 3,
                "items_count": 1,
                "segments": [
                    {
                        "segment_index": 0,
                        "document_id": "segment-doc-new",
                        "uri": "twinr://longterm/test/conflicts/catalog/segment/0000",
                        "entry_count": 1,
                    }
                ],
                "written_at": "2026-03-29T16:05:00+00:00",
            }

            def _fetch_full_document(*, document_id=None, origin_uri=None, include_content=True, max_content_chars=4000):
                del document_id
                del include_content
                del max_content_chars
                if origin_uri == current_head_uri:
                    return {
                        "success": True,
                        "origin_uri": origin_uri,
                        "chunks": [
                            {
                                "payload_id": "doc-old",
                                "document_id": "doc-old",
                                "metadata": {
                                    "twinr_snapshot_kind": "conflicts",
                                    "twinr_catalog_current_head": True,
                                    "twinr_catalog_items_count": 0,
                                },
                                "content": json.dumps(older_payload, ensure_ascii=False),
                            },
                            {
                                "payload_id": "doc-new",
                                "document_id": "doc-new",
                                "metadata": {
                                    "twinr_snapshot_kind": "conflicts",
                                    "twinr_catalog_current_head": True,
                                    "twinr_catalog_items_count": 1,
                                },
                                "content": json.dumps(newer_payload, ensure_ascii=False),
                            },
                        ],
                    }
                _raise_missing_current_head_or_unavailable(origin_uri)

            with patch.object(remote_state.client, "fetch_full_document", side_effect=_fetch_full_document):
                payload = remote_catalog.load_catalog_payload(snapshot_kind="conflicts")
                item_count = remote_catalog.catalog_item_count(snapshot_kind="conflicts")

        assert payload is not None
        self.assertEqual(payload["written_at"], "2026-03-29T16:05:00+00:00")
        self.assertEqual(payload["items_count"], 1)
        self.assertEqual(item_count, 1)

    def test_is_catalog_payload_rejects_nonempty_current_head_without_segment_refs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.required = True
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None

            self.assertFalse(
                remote_catalog.is_catalog_payload(
                    snapshot_kind="objects",
                    payload={
                        "schema": "twinr_memory_object_catalog_v3",
                        "version": 3,
                        "items_count": 2,
                        "segments": [],
                        "written_at": "2026-03-31T06:21:22.063093+00:00",
                    },
                )
            )

    def test_load_catalog_payload_recovers_nonempty_current_head_from_metadata_embedded_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.required = True
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None
            head_payload = {
                "schema": "twinr_memory_object_catalog_v3",
                "version": 3,
                "items_count": 1,
                "segments": [
                    {
                        "segment_index": 0,
                        "document_id": "segment-doc-1",
                        "uri": "twinr://longterm/test/objects/catalog/segment/0000",
                        "entry_count": 1,
                    }
                ],
                "written_at": "2026-03-31T06:21:22.063093+00:00",
            }

            def _fetch_full_document(*, document_id=None, origin_uri=None, include_content=True, max_content_chars=4000):
                del document_id
                del origin_uri
                del include_content
                del max_content_chars
                return {
                    "metadata": {
                        "twinr_snapshot_kind": "objects",
                        "twinr_catalog_current_head": True,
                        "twinr_catalog_schema": "twinr_memory_object_catalog_v3",
                        "twinr_catalog_items_count": 1,
                        "twinr_catalog_written_at": "2026-03-31T06:21:22.063093+00:00",
                        "twinr_payload": dict(head_payload),
                    }
                }

            with patch.object(remote_state.client, "fetch_full_document", side_effect=_fetch_full_document):
                payload = remote_catalog.load_catalog_payload(snapshot_kind="objects")

        assert payload is not None
        self.assertEqual(payload["items_count"], 1)
        self.assertEqual(payload["segments"], head_payload["segments"])

    def test_persist_catalog_payload_embeds_full_payload_in_current_head_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            result = LongTermConsolidationResultV1(
                turn_id="turn:test",
                occurred_at=datetime(2026, 3, 31, 7, 0, tzinfo=ZoneInfo("Europe/Berlin")),
                episodic_objects=(),
                durable_objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:doctor_appointment",
                        kind="event",
                        summary="A doctor appointment is scheduled.",
                        source=_source(),
                        status="active",
                        confidence=0.97,
                        confirmed_by_user=True,
                        slot_key="event:doctor_appointment",
                        value_key="doctor_appointment",
                    ),
                ),
                deferred_objects=(),
                conflicts=(),
                graph_edges=(),
            )

            store.apply_consolidation(result)
            head_payload = _current_catalog_head_payload(store, snapshot_kind="objects")
            head_record = remote_state.client.records_by_uri[_current_catalog_head_uri(store, snapshot_kind="objects")]

        assert head_payload is not None
        metadata = dict(head_record.get("metadata") or {})
        self.assertEqual(metadata.get("twinr_payload"), head_payload)

    def test_probe_catalog_payload_metadata_only_uses_backend_compatible_min_content_chars(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.required = True
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None
            head_payload = {
                "schema": "twinr_memory_object_catalog_v3",
                "version": 3,
                "items_count": 0,
                "segments": [],
                "written_at": "2026-03-29T16:00:00+00:00",
            }

            def _fetch_full_document(*, document_id=None, origin_uri=None, include_content=True, max_content_chars=4000):
                del document_id
                del origin_uri
                self.assertFalse(include_content)
                self.assertGreaterEqual(max_content_chars, 100)
                return dict(head_payload)

            with patch.object(remote_state.client, "fetch_full_document", side_effect=_fetch_full_document):
                payload = remote_catalog.probe_catalog_payload(snapshot_kind="objects")

        assert payload is not None
        self.assertEqual(payload["schema"], "twinr_memory_object_catalog_v3")
        self.assertEqual(payload["items_count"], 0)

    def test_probe_catalog_payload_retries_full_content_when_metadata_only_current_head_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.required = True
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None
            head_payload = {
                "schema": "twinr_memory_object_catalog_v3",
                "version": 3,
                "items_count": 0,
                "segments": [],
                "written_at": "2026-03-29T16:00:00+00:00",
            }
            fetch_calls: list[dict[str, object]] = []

            def _fetch_full_document(*, document_id=None, origin_uri=None, include_content=True, max_content_chars=4000):
                del document_id
                fetch_calls.append(
                    {
                        "origin_uri": origin_uri,
                        "include_content": include_content,
                        "max_content_chars": max_content_chars,
                    }
                )
                if not include_content:
                    raise ChonkyDBError(
                        "ChonkyDB request failed for GET /v1/external/documents/full",
                        status_code=400,
                        response_json={
                            "detail": "Request validation failed",
                            "success": False,
                        },
                    )
                return dict(head_payload)

            with patch.object(remote_state.client, "fetch_full_document", side_effect=_fetch_full_document):
                payload = remote_catalog.probe_catalog_payload(snapshot_kind="objects")

        assert payload is not None
        self.assertEqual(payload["schema"], "twinr_memory_object_catalog_v3")
        self.assertEqual(payload["items_count"], 0)
        self.assertEqual(len(fetch_calls), 2)
        self.assertFalse(bool(fetch_calls[0]["include_content"]))
        self.assertTrue(bool(fetch_calls[1]["include_content"]))
        self.assertGreater(int(fetch_calls[1]["max_content_chars"]), int(fetch_calls[0]["max_content_chars"]))

    def test_probe_catalog_payload_retries_full_content_when_metadata_only_head_lacks_nonempty_segments(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.required = True
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None
            metadata_only_head = {
                "success": True,
                "document_id": "graph-head-doc-1",
                "origin_uri": remote_catalog._catalog_head_uri(snapshot_kind="graph_nodes"),
                "chunk_count": 1,
                "chunks": [
                    {
                        "metadata": {
                            "twinr_snapshot_kind": "graph_nodes",
                            "twinr_catalog_current_head": True,
                            "twinr_catalog_schema": "twinr_graph_node_catalog_v3",
                            "twinr_catalog_items_count": 5,
                        }
                    }
                ],
            }
            full_head = {
                "schema": "twinr_graph_node_catalog_v3",
                "version": 3,
                "items_count": 5,
                "segments": [
                    {
                        "segment_index": 0,
                        "document_id": "graph-segment-doc-1",
                        "uri": remote_catalog._catalog_segment_uri(snapshot_kind="graph_nodes", segment_index=0),
                        "entry_count": 5,
                    }
                ],
                "subject_node_id": "user:main",
                "graph_id": "graph:user_main",
            }
            fetch_calls: list[dict[str, object]] = []

            def _fetch_full_document(*, document_id=None, origin_uri=None, include_content=True, max_content_chars=4000):
                del document_id
                fetch_calls.append(
                    {
                        "origin_uri": origin_uri,
                        "include_content": include_content,
                        "max_content_chars": max_content_chars,
                    }
                )
                return dict(metadata_only_head if not include_content else full_head)

            with patch.object(remote_state.client, "fetch_full_document", side_effect=_fetch_full_document):
                payload = remote_catalog.probe_catalog_payload(snapshot_kind="graph_nodes")

        assert payload is not None
        self.assertEqual(payload["schema"], "twinr_graph_node_catalog_v3")
        self.assertEqual(payload["items_count"], 5)
        self.assertEqual(len(fetch_calls), 2)
        self.assertFalse(bool(fetch_calls[0]["include_content"]))
        self.assertTrue(bool(fetch_calls[1]["include_content"]))
        self.assertGreater(int(fetch_calls[1]["max_content_chars"]), int(fetch_calls[0]["max_content_chars"]))

    def test_probe_catalog_payload_uses_bootstrap_timeout_for_fixed_current_head_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            head_payload = {
                "schema": "twinr_memory_archive_catalog_v3",
                "version": 3,
                "items_count": 0,
                "segments": [],
                "written_at": "2026-03-30T19:00:00+00:00",
            }
            fetch_calls: list[dict[str, object]] = []

            class _TimeoutSensitiveCurrentHeadClient:
                def __init__(self, timeout_s: float) -> None:
                    self.config = SimpleNamespace(timeout_s=timeout_s)

                def clone_with_timeout(self, timeout_s: float):
                    return _TimeoutSensitiveCurrentHeadClient(timeout_s)

                def fetch_full_document(
                    self,
                    *,
                    document_id=None,
                    origin_uri=None,
                    include_content=True,
                    max_content_chars=4000,
                ):
                    del document_id
                    fetch_calls.append(
                        {
                            "origin_uri": origin_uri,
                            "include_content": include_content,
                            "max_content_chars": max_content_chars,
                            "timeout_s": float(self.config.timeout_s),
                        }
                    )
                    if not include_content:
                        raise ChonkyDBError(
                            "ChonkyDB request failed for GET /v1/external/documents/full",
                            status_code=400,
                            response_json={
                                "detail": "Request validation failed",
                                "success": False,
                            },
                        )
                    if float(self.config.timeout_s) < 20.0:
                        raise ChonkyDBError(
                            "ChonkyDB request failed for GET /v1/external/documents/full: "
                            "HTTPSConnectionPool(host='memory.test', port=443): Read timed out. (read timeout=8.0)"
                        )
                    return {
                        "document_id": "archive-head-1",
                        "content": json.dumps(head_payload),
                    }

            class _CatalogHeadBootstrapRemoteState(_FakeRemoteState):
                def __init__(self) -> None:
                    super().__init__()
                    self.required = True
                    client = _TimeoutSensitiveCurrentHeadClient(timeout_s=8.0)
                    self.client = client
                    self.read_client = client
                    self.write_client = client

                def _origin_resolution_client(self, client):
                    return client.clone_with_timeout(24.0)

            remote_state = _CatalogHeadBootstrapRemoteState()
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None

            payload = remote_catalog.probe_catalog_payload(snapshot_kind="archive")

        assert payload is not None
        self.assertEqual(payload["schema"], "twinr_memory_archive_catalog_v3")
        self.assertEqual(payload["items_count"], 0)
        self.assertEqual(
            [
                {
                    "include_content": bool(call["include_content"]),
                    "timeout_s": float(call["timeout_s"]),
                }
                for call in fetch_calls
            ],
            [
                {"include_content": False, "timeout_s": 24.0},
                {"include_content": True, "timeout_s": 24.0},
            ],
        )

    def test_probe_remote_current_snapshot_prefers_catalog_current_head_without_legacy_snapshot_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _ProbeOnlyCatalogRemoteState()
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)

            class _CatalogProbeOnly:
                def enabled(self) -> bool:
                    return True

                def probe_catalog_payload(self, *, snapshot_kind: str):
                    return {
                        "schema": {
                            "objects": "twinr_memory_object_catalog_v3",
                            "conflicts": "twinr_memory_conflict_catalog_v3",
                            "archive": "twinr_memory_archive_catalog_v3",
                        }[snapshot_kind],
                        "version": 3,
                        "items_count": 0,
                        "segments": [],
                        "written_at": "2026-03-29T16:00:00+00:00",
                    }

                def is_catalog_payload(self, *, snapshot_kind: str, payload) -> bool:
                    del snapshot_kind
                    return isinstance(payload, dict) and payload.get("version") == 3 and isinstance(payload.get("segments"), list)

            store._remote_catalog = _CatalogProbeOnly()

            payload = store.probe_remote_current_snapshot(snapshot_kind="objects")

        assert payload is not None
        self.assertEqual(payload["schema"], "twinr_memory_object_catalog_v3")
        self.assertEqual(remote_state.probe_calls, [])

    def test_probe_remote_current_snapshot_prefers_recent_same_process_snapshot_when_catalog_head_lags(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _ProbeOnlyCatalogRemoteState()
            remote_state.required = True
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)

            class _CatalogHeadInvisible:
                def enabled(self) -> bool:
                    return True

                def probe_catalog_payload(self, *, snapshot_kind: str):
                    del snapshot_kind
                    return None

                def is_catalog_payload(self, *, snapshot_kind: str, payload) -> bool:
                    del snapshot_kind
                    del payload
                    return False

            store._remote_catalog = _CatalogHeadInvisible()
            store._recent_local_snapshot_payloads["objects"] = store._stamp_snapshot_payload(store._empty_objects_payload())

            with patch.object(
                type(remote_state),
                "probe_snapshot_load",
                side_effect=AssertionError("Legacy remote snapshot probes must not run for catalog-backed current heads."),
            ):
                payload = store.probe_remote_current_snapshot(snapshot_kind="objects")

        assert payload is not None
        self.assertEqual(payload["schema"], "twinr_memory_object_store")
        self.assertEqual(payload["objects"], [])

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
        objects_head = _current_catalog_head_payload(store, snapshot_kind="objects")
        assert objects_head is not None
        self.assertEqual(objects_head["items_count"], 4)
        self.assertEqual(_count_remote_records_with_schema(remote_state.client, "twinr_memory_object_record_v2"), 4)
        self.assertEqual(_count_remote_records_with_schema(remote_state.client, "twinr_memory_object_catalog_segment_v1"), 1)
        self.assertEqual(_count_remote_records_with_schema(remote_state.client, "twinr_memory_object_catalog_v3"), 1)

    def test_remote_primary_store_retries_transient_current_head_503_before_conflict_write(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.client = _TransientCurrentHead503Client(snapshot_kind="conflicts")
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            remote_state.config.long_term_memory_remote_retry_attempts = 2
            remote_state.config.long_term_memory_remote_retry_backoff_s = 0.0
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            conflict = LongTermMemoryConflictV1(
                slot_key="contact:person:lea:phone",
                candidate_memory_id="fact:lea_phone_current",
                existing_memory_ids=("fact:lea_phone_old",),
                question="Welche Telefonnummer soll ich fuer Lea verwenden?",
                reason="Es gibt widerspruechliche Telefonnummern fuer Lea.",
            )

            store.write_snapshot(objects=(), conflicts=(conflict,), archived_objects=())
            loaded_conflicts = store.load_conflicts()

        self.assertEqual(len(loaded_conflicts), 1)
        self.assertEqual(loaded_conflicts[0].slot_key, conflict.slot_key)
        self.assertEqual(remote_state.client.remaining_failures, 0)
        self.assertEqual(len(remote_state.client.current_head_attempt_uris), 1)

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

        objects_head = _current_catalog_head_payload(store, snapshot_kind="objects")
        assert objects_head is not None
        self.assertEqual(objects_head["items_count"], 5)
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

        objects_head = _current_catalog_head_payload(store, snapshot_kind="objects")
        assert objects_head is not None
        self.assertEqual(objects_head["items_count"], 5)
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

    def test_remote_primary_store_round_trips_through_public_item_envelope(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.client = _PayloadEnvelopeLiveShapeChonkyClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            objects = (
                LongTermMemoryObjectV1(
                    memory_id="fact:payload_envelope",
                    kind="fact",
                    summary="Payload envelopes now carry the real memory object.",
                    source=_source(),
                    status="active",
                    confidence=0.93,
                    details="Recovered from the record payload without the metadata blob.",
                    slot_key="fact:payload_envelope",
                    value_key="payload_envelope",
                    attributes={"transport": "public_payload"},
                ),
            )

            store.write_snapshot(objects=objects)
            document = next(iter(remote_state.client.records_by_document_id.values()))
            stored_payload = dict(document.get("payload") or {})
            stored_metadata = dict(document.get("metadata") or {})
            loaded = store.load_objects()

        self.assertEqual(stored_payload.get("schema"), "twinr_memory_object_record_v2")
        self.assertEqual(dict(stored_payload.get("object") or {}).get("memory_id"), "fact:payload_envelope")
        self.assertEqual(dict(stored_payload.get("object") or {}).get("details"), objects[0].details)
        self.assertIsInstance(stored_metadata.get("twinr_payload"), dict)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].memory_id, "fact:payload_envelope")
        self.assertEqual(loaded[0].details, "Recovered from the record payload without the metadata blob.")
        self.assertEqual(dict(loaded[0].attributes or {}).get("transport"), "public_payload")

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
            writer_store = LongTermStructuredStore(
                base_path=Path(temp_dir) / "state" / "chonkydb",
                remote_state=remote_state,
            )
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

            writer_store.write_snapshot(objects=objects)

            self.assertEqual(writer_store.ensure_remote_snapshots(), ())
            reader_store = LongTermStructuredStore(
                base_path=Path(temp_dir) / "reader" / "state" / "chonkydb",
                remote_state=remote_state,
            )
            loaded = reader_store.load_objects()

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
            objects_head = _current_catalog_head_payload(store, snapshot_kind="objects")
            assert objects_head is not None
            _sparsify_catalog_segment_entries(remote_state.client, objects_head)
            bulk_calls_before_probe = remote_state.client.bulk_calls
            remote_state.client.retrieve_calls = 0

            self.assertEqual(store.ensure_remote_snapshots(), ())
            first_retrieve_calls = remote_state.client.retrieve_calls
            remote_state.client.retrieve_calls = 0
            self.assertEqual(store.ensure_remote_snapshots(), ())
            second_retrieve_calls = remote_state.client.retrieve_calls
            sparse_entries = store._remote_catalog.load_catalog_entries(snapshot_kind="objects")

        self.assertEqual(first_retrieve_calls, 0)
        self.assertEqual(second_retrieve_calls, 0)
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

    def test_remote_catalog_promotes_legacy_snapshot_head_to_current_head_document(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None
            object_fact = LongTermMemoryObjectV1(
                memory_id="fact:janina_spouse",
                kind="relationship_fact",
                summary="Janina is the user's wife.",
                source=_source(),
                status="active",
                confidence=0.98,
            ).to_payload()
            legacy_catalog = remote_catalog.build_catalog_payload(
                snapshot_kind="objects",
                item_payloads=(object_fact,),
                item_id_getter=lambda item: item.get("memory_id"),
                metadata_builder=lambda item: store._remote_item_metadata(snapshot_kind="objects", payload=item),
                content_builder=lambda item: store._remote_item_search_text(snapshot_kind="objects", payload=item),
            )
            remote_state.save_snapshot(snapshot_kind="objects", payload=legacy_catalog)

            self.assertIsNone(_current_catalog_head_payload(store, snapshot_kind="objects"))
            entries = remote_catalog.load_catalog_entries(snapshot_kind="objects")

        promoted_head = _current_catalog_head_payload(store, snapshot_kind="objects")
        assert promoted_head is not None
        self.assertEqual(tuple(entry.item_id for entry in entries), ("fact:janina_spouse",))
        self.assertEqual(promoted_head["schema"], "twinr_memory_object_catalog_v3")
        self.assertEqual(promoted_head["items_count"], 1)
        self.assertEqual(_count_remote_records_with_schema(remote_state.client, "twinr_memory_object_catalog_v3"), 1)

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
            remote_state.client.fetch_full_document_calls = 0

            first = store.select_relevant_objects(query_text="Janina", limit=1)
            retrieve_calls_after_first = remote_state.client.retrieve_calls
            fetch_calls_after_first = remote_state.client.fetch_full_document_calls
            second = store.select_relevant_objects(query_text="Janina", limit=1)

        self.assertEqual(tuple(item.memory_id for item in first), ("fact:janina_spouse",))
        self.assertEqual(tuple(item.memory_id for item in second), ("fact:janina_spouse",))
        self.assertEqual(retrieve_calls_after_first, 0)
        self.assertEqual(remote_state.client.retrieve_calls, retrieve_calls_after_first)
        self.assertEqual(remote_state.client.fetch_full_document_calls, fetch_calls_after_first)

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
            remote_state.client.fetch_full_document_calls = 0
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

    def test_generic_recap_query_prefers_conversation_turn_episodes_over_sensor_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LongTermStructuredStore.from_config(_config(temp_dir))
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="episode:turn_recap",
                        kind="episode",
                        summary="Conversation turn recorded for long-term memory.",
                        details='User said: "Lea brings lentil soup tonight." Assistant answered: "I will remember Lea and the soup."',
                        source=_source(),
                        status="active",
                        confidence=1.0,
                        slot_key="episode:turn:recap",
                        value_key="turn:recap",
                        attributes={
                            "raw_transcript": "Lea bringt heute Abend Linsensuppe vorbei.",
                            "raw_response": "Ich merke mir Lea und die Suppe.",
                            "request_source": "conversation",
                            "input_modality": "voice",
                        },
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="episode:sensor_noise",
                        kind="episode",
                        summary="Multimodal device event recorded: sensor_observation.",
                        details="Structured multimodal evidence from PIR and room-audio state.",
                        source=_source(),
                        status="active",
                        confidence=0.92,
                        slot_key="episode:multimodal:sensor",
                        value_key="sensor:observation",
                        attributes={
                            "event_names": ["sensor.state"],
                            "memory_domain": "smart_home_environment",
                        },
                    ),
                )
            )

            relevant = store.select_relevant_episodic_objects(
                query_text="Worüber haben wir heute gesprochen?",
                limit=2,
                fallback_limit=0,
                require_query_match=False,
            )

        self.assertEqual(tuple(item.memory_id for item in relevant), ("episode:turn_recap",))

    def test_remote_primary_store_matches_generic_recap_query_for_new_conversation_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.client.supports_topk_records = True
            store = LongTermStructuredStore(
                base_path=Path(temp_dir) / "state" / "chonkydb",
                remote_state=remote_state,
            )
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="episode:turn_remote_recap",
                        kind="episode",
                        summary="Conversation turn recorded for long-term memory.",
                        details='User said: "Lea brings lentil soup tonight." Assistant answered: "I will remember Lea and the soup."',
                        source=_source(),
                        status="active",
                        confidence=1.0,
                        slot_key="episode:turn:remote_recap",
                        value_key="turn:remote_recap",
                        attributes={
                            "raw_transcript": "Lea bringt heute Abend Linsensuppe vorbei.",
                            "raw_response": "Ich merke mir Lea und die Suppe.",
                            "request_source": "conversation",
                            "input_modality": "voice",
                        },
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="episode:multimodal_sensor_remote",
                        kind="episode",
                        summary="Multimodal device event recorded: sensor_observation.",
                        details="Structured multimodal evidence from PIR and room-audio state.",
                        source=_source(),
                        status="active",
                        confidence=0.92,
                        slot_key="episode:multimodal:remote_sensor",
                        value_key="sensor:remote_observation",
                        attributes={
                            "event_names": ["sensor.state"],
                            "memory_domain": "smart_home_environment",
                        },
                    ),
                )
            )

            relevant = store.select_relevant_episodic_objects(
                query_text="What did we talk about today?",
                limit=2,
                fallback_limit=0,
                require_query_match=False,
            )

        self.assertEqual(tuple(item.memory_id for item in relevant), ("episode:turn_remote_recap",))

    def test_remote_primary_recap_query_hits_direct_scope_search_before_recent_catalog_rescue(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.client.supports_topk_records = True
            store = LongTermStructuredStore(
                base_path=Path(temp_dir) / "state" / "chonkydb",
                remote_state=remote_state,
            )
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="episode:turn_remote_direct_recap",
                        kind="episode",
                        summary="Conversation turn recorded for long-term memory.",
                        details='User said: "Lea brings lentil soup tonight." Assistant answered: "I will remember Lea and the soup."',
                        source=_source(),
                        status="active",
                        confidence=1.0,
                        slot_key="episode:turn:remote_direct_recap",
                        value_key="turn:remote_direct_recap",
                        attributes={
                            "raw_transcript": "Lea bringt heute Abend Linsensuppe vorbei.",
                            "raw_response": "Ich merke mir Lea und die Suppe.",
                            "request_source": "conversation",
                            "input_modality": "voice",
                        },
                    ),
                )
            )
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None

            def _fail_catalog_rescue(*args, **kwargs):
                del args
                del kwargs
                raise AssertionError("generic recap direct scope search must not hydrate the recent episodic catalog")

            remote_catalog.load_catalog_entries = _fail_catalog_rescue  # type: ignore[method-assign]
            remote_catalog.top_catalog_entries = _fail_catalog_rescue  # type: ignore[method-assign]

            relevant = store.select_relevant_episodic_objects(
                query_text="Worüber haben wir heute gesprochen?",
                limit=2,
                fallback_limit=0,
                require_query_match=False,
            )

        self.assertEqual(tuple(item.memory_id for item in relevant), ("episode:turn_remote_direct_recap",))
        self.assertEqual(
            tuple(payload["query_text"] for payload in remote_state.client.topk_records_payloads[-2:]),
            ("Worüber haben wir heute gesprochen?", "worueber gesprochen"),
        )

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
            remote_state.client.fetch_full_document_calls = 0

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
        self.assertGreaterEqual(remote_state.client.topk_records_calls, 1)
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

    def test_select_fast_topic_objects_keeps_same_process_snapshot_bridge_when_remote_written_at_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.required = True
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

            loaded = store.load_objects()

            def _fail_remote_fast_topic(*args, **kwargs):
                del args
                del kwargs
                raise AssertionError("same-process snapshot bridge should satisfy fast-topic before remote search")

            assert store._remote_catalog is not None
            store._remote_catalog.search_current_item_payloads_fast = _fail_remote_fast_topic  # type: ignore[method-assign]
            relevant = store.select_fast_topic_objects(
                query_text="Wo stand früher meine rote Thermoskanne?",
                limit=1,
                timeout_s=0.45,
            )

        self.assertEqual(tuple(item.memory_id for item in loaded), ("fact:thermos_location_old",))
        self.assertEqual(tuple(item.memory_id for item in relevant), ("fact:thermos_location_old",))

    def test_select_open_conflicts_keeps_same_process_snapshot_bridge_when_remote_visibility_lags(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _LaggingSnapshotVisibilityRemoteState()
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
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
            store.write_snapshot(
                objects=(old_phone, new_phone),
                conflicts=(conflict,),
                archived_objects=(),
            )

            remote_catalog = store._remote_catalog
            assert remote_catalog is not None

            def _fail_remote_conflict_lookup(*args, **kwargs):
                del args
                del kwargs
                raise AssertionError("same-process conflict bridge should satisfy select_open_conflicts before remote reads")

            remote_catalog.catalog_item_count = _fail_remote_conflict_lookup  # type: ignore[method-assign]
            remote_catalog.search_current_item_payloads = _fail_remote_conflict_lookup  # type: ignore[method-assign]
            conflicts = store.select_open_conflicts(
                query_text="Which phone number should I use for Corinna Maier?",
                limit=1,
            )

        self.assertEqual(tuple(item.slot_key for item in conflicts), ("contact:person:corinna_maier:phone",))

    def test_select_fast_topic_objects_raises_precise_remote_read_failure_for_scope_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.config.project_root = str(project_root)
            remote_state.config.long_term_memory_remote_retry_attempts = 2
            remote_state.config.long_term_memory_remote_retry_backoff_s = 0.0
            remote_state.client = _AlwaysTimeoutingScopeTopKClient()
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
        self.assertEqual(details["retry_attempts_configured"], 2)
        self.assertEqual(details["retry_backoff_s"], 0.0)
        self.assertEqual(details["retry_mode"], "bounded_transient_retry")
        self.assertTrue(details["retry_enabled"])
        self.assertEqual(details["attempt_index"], 2)
        self.assertEqual(details["attempt_count"], 2)
        self.assertEqual(remote_state.client.topk_records_calls, 2)
        failed_event = next(
            event
            for event in events
            if event.get("event") == "longterm_remote_read_failed"
            and dict(event.get("data") or {}).get("operation") == "fast_topic_topk_search"
        )
        failed_data = dict(failed_event["data"])
        self.assertEqual(failed_data["classification"], "timeout")
        self.assertEqual(failed_data["timeout_reason"], "read_operation_timed_out")
        self.assertEqual(failed_data["retry_mode"], "bounded_transient_retry")

    def test_select_fast_topic_objects_records_http_status_and_problem_detail_for_503(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.config.project_root = str(project_root)
            remote_state.config.long_term_memory_remote_retry_attempts = 2
            remote_state.config.long_term_memory_remote_retry_backoff_s = 0.0
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
        self.assertEqual(details["retry_attempts_configured"], 2)
        self.assertEqual(details["retry_backoff_s"], 0.0)
        self.assertEqual(details["retry_mode"], "bounded_transient_retry")
        self.assertTrue(details["retry_enabled"])
        self.assertEqual(details["attempt_index"], 2)
        self.assertEqual(details["attempt_count"], 2)
        self.assertEqual(remote_state.client.topk_records_calls, 2)
        failed_event = next(
            event
            for event in events
            if event.get("event") == "longterm_remote_read_failed"
            and dict(event.get("data") or {}).get("operation") == "fast_topic_topk_search"
        )
        failed_data = dict(failed_event["data"])
        self.assertEqual(failed_data["status_code"], 503)
        self.assertEqual(failed_data["response_error_type"], "ServerBusy")

    def test_select_fast_topic_objects_retries_transient_scope_timeout_before_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.config.project_root = str(project_root)
            remote_state.config.long_term_memory_remote_retry_attempts = 2
            remote_state.config.long_term_memory_remote_retry_backoff_s = 0.0
            remote_state.client = _TimeoutingScopeTopKClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=project_root / "state" / "chonkydb", remote_state=remote_state)
            store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:janina_spouse",
                        kind="relationship_fact",
                        summary="Janina ist die Ehefrau des Nutzers.",
                        source=_source(),
                        status="active",
                        confidence=0.98,
                    ),
                ),
            )
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None

            payloads = remote_catalog.search_current_item_payloads_fast(
                snapshot_kind="objects",
                query_text="Janina",
                limit=1,
                timeout_s=0.45,
            )
            events = TwinrOpsEventStore.from_project_root(project_root).tail(limit=8)

        self.assertEqual(tuple(payload["memory_id"] for payload in payloads), ("fact:janina_spouse",))
        self.assertEqual(remote_state.client.topk_records_calls, 2)
        self.assertFalse(
            any(
                event.get("event") == "longterm_remote_read_failed"
                and dict(event.get("data") or {}).get("operation") == "fast_topic_topk_search"
                for event in events
            )
        )

    def test_select_fast_topic_objects_retries_transient_503_before_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.config.project_root = str(project_root)
            remote_state.config.long_term_memory_remote_retry_attempts = 2
            remote_state.config.long_term_memory_remote_retry_backoff_s = 0.0
            remote_state.client = _TransientHttp503ScopeTopKClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=project_root / "state" / "chonkydb", remote_state=remote_state)
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
            events = TwinrOpsEventStore.from_project_root(project_root).tail(limit=8)

        self.assertEqual(tuple(payload["memory_id"] for payload in payloads), ("fact:thermos_location_old",))
        self.assertEqual(remote_state.client.topk_records_calls, 2)
        self.assertFalse(
            any(
                event.get("event") == "longterm_remote_read_failed"
                and dict(event.get("data") or {}).get("operation") == "fast_topic_topk_search"
                for event in events
            )
        )

    def test_search_current_item_payloads_fast_rescues_timeout_with_current_catalog_projection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.config.long_term_memory_remote_retry_attempts = 2
            remote_state.config.long_term_memory_remote_retry_backoff_s = 0.0
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
            remote_state.config.long_term_memory_remote_retry_attempts = 2
            remote_state.config.long_term_memory_remote_retry_backoff_s = 0.0
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

    def test_search_current_item_payloads_fast_skips_repeated_unsupported_scope_ref_roundtrips(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.config.long_term_memory_remote_read_cache_ttl_s = 60.0
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
            failing_client = _UnsupportedScopeRefTopKClient()
            failing_client.records_by_document_id = {
                document_id: dict(record)
                for document_id, record in remote_state.client.records_by_document_id.items()
            }
            failing_client.records_by_uri = {
                uri: dict(record)
                for uri, record in remote_state.client.records_by_uri.items()
            }
            failing_client._next_document_id = remote_state.client._next_document_id
            remote_state.client = failing_client
            remote_state.read_client = failing_client
            remote_state.write_client = failing_client
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None

            first_payloads = remote_catalog.search_current_item_payloads_fast(
                snapshot_kind="objects",
                query_text="Thermoskanne",
                limit=1,
                timeout_s=0.6,
            )
            second_payloads = remote_catalog.search_current_item_payloads_fast(
                snapshot_kind="objects",
                query_text="Thermoskanne",
                limit=1,
                timeout_s=0.6,
            )

        self.assertEqual(tuple(payload["memory_id"] for payload in first_payloads), ("fact:thermos_location_old",))
        self.assertEqual(tuple(payload["memory_id"] for payload in second_payloads), ("fact:thermos_location_old",))
        self.assertEqual(remote_state.client.topk_records_calls, 1)

    def test_search_current_item_payloads_fast_skips_cold_catalog_rescue_when_projection_is_uncached(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.config.long_term_memory_remote_retry_attempts = 2
            remote_state.config.long_term_memory_remote_retry_backoff_s = 0.0
            remote_state.client = _AlwaysTimeoutingScopeTopKClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None

            with self.assertRaises(LongTermRemoteReadFailedError):
                remote_catalog.search_current_item_payloads_fast(
                    snapshot_kind="objects",
                    query_text="Thermoskanne",
                    limit=1,
                    timeout_s=0.45,
                )

        self.assertEqual(remote_state.client.topk_records_calls, 2)
        self.assertEqual(remote_state.client.fetch_full_document_calls, 0)

    def test_search_current_item_payloads_fast_returns_empty_for_missing_current_head(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.client = _NoDocumentsFullScope404ChonkyClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None

            payloads = remote_catalog.search_current_item_payloads_fast(
                snapshot_kind="objects",
                query_text="Thermoskanne",
                limit=1,
                timeout_s=0.45,
            )

        self.assertEqual(payloads, ())
        self.assertEqual(remote_state.client.topk_records_calls, 1)
        self.assertTrue(remote_catalog._scope_search_supported(snapshot_kind="objects"))

    def test_search_current_item_payloads_fast_fallback_ignores_stale_document_id_hint_for_mutable_current_head(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
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
            fresh_payload = _current_catalog_head_payload(store, snapshot_kind="objects")
            assert fresh_payload is not None
            stale_payload = {
                "schema": "twinr_memory_object_catalog_v3",
                "version": 3,
                "items_count": 0,
                "segments": [],
            }
            failing_client = _Http503ScopeTopKClient()
            failing_client.records_by_document_id = {
                document_id: dict(record)
                for document_id, record in remote_state.client.records_by_document_id.items()
            }
            failing_client.records_by_uri = {
                uri: dict(record)
                for uri, record in remote_state.client.records_by_uri.items()
            }
            failing_client._next_document_id = remote_state.client._next_document_id
            remote_state.client = failing_client
            remote_state.read_client = failing_client
            remote_state.write_client = failing_client
            load_snapshot_calls: list[dict[str, object]] = []

            def _load_snapshot(
                *,
                snapshot_kind: str,
                local_path=None,
                prefer_cached_document_id: bool = True,
            ):
                del local_path
                load_snapshot_calls.append(
                    {
                        "snapshot_kind": snapshot_kind,
                        "prefer_cached_document_id": prefer_cached_document_id,
                    }
                )
                return dict(stale_payload if prefer_cached_document_id else fresh_payload)

            remote_state.load_snapshot = _load_snapshot  # type: ignore[assignment]
            remote_catalog._load_catalog_head_result = (  # type: ignore[method-assign]
                lambda **kwargs: SimpleNamespace(status="unavailable", payload=None)
            )

            payloads = remote_catalog.search_current_item_payloads_fast(
                snapshot_kind="objects",
                query_text="Thermoskanne",
                limit=1,
                timeout_s=0.6,
            )

        self.assertTrue(
            not load_snapshot_calls
            or all(
                call == {"snapshot_kind": "objects", "prefer_cached_document_id": False}
                for call in load_snapshot_calls
            )
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

    def test_search_current_item_payloads_skips_repeated_unsupported_scope_ref_roundtrips(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.config.long_term_memory_remote_read_cache_ttl_s = 60.0
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
            failing_client = _UnsupportedScopeRefTopKClient()
            failing_client.records_by_document_id = {
                document_id: dict(record)
                for document_id, record in remote_state.client.records_by_document_id.items()
            }
            failing_client.records_by_uri = {
                uri: dict(record)
                for uri, record in remote_state.client.records_by_uri.items()
            }
            failing_client._next_document_id = remote_state.client._next_document_id
            remote_state.client = failing_client
            remote_state.read_client = failing_client
            remote_state.write_client = failing_client
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None

            first_payloads = remote_catalog.search_current_item_payloads(
                snapshot_kind="objects",
                query_text="Thermoskanne",
                limit=1,
            )
            second_payloads = remote_catalog.search_current_item_payloads(
                snapshot_kind="objects",
                query_text="Thermoskanne",
                limit=1,
            )

        self.assertEqual(tuple(payload["memory_id"] for payload in first_payloads), ("fact:thermos_location_old",))
        self.assertEqual(tuple(payload["memory_id"] for payload in second_payloads), ("fact:thermos_location_old",))
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
        self.assertGreaterEqual(remote_state.client.topk_records_calls, 1)
        self.assertGreaterEqual(remote_state.client.retrieve_calls, 0)

    def test_search_current_item_payloads_false_empty_catalog_projection_avoids_item_document_fetches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.client = _NoItemFetchEmptyScopeTopKClient()
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
        self.assertEqual(remote_state.client.item_fetch_attempts, 0)

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

    def test_search_current_item_payloads_reconciles_stale_scope_payloads_without_catalog_fallback_flag(self) -> None:
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
                query_text="Aprikosenmarmelade",
                limit=2,
                allow_catalog_fallback=False,
            )
            payload_by_id = {payload["memory_id"]: payload for payload in payloads}

        self.assertIn("fact:jam_preference_new", payload_by_id)
        self.assertEqual(payload_by_id["fact:jam_preference_new"]["status"], "active")
        self.assertTrue(payload_by_id["fact:jam_preference_new"]["confirmed_by_user"])
        self.assertNotEqual(
            payload_by_id["fact:jam_preference_new"]["status"],
            "uncertain",
        )

    def test_search_current_item_payloads_fast_reconciles_stale_scope_payloads_with_current_catalog(self) -> None:
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

            payloads = remote_catalog.search_current_item_payloads_fast(
                snapshot_kind="objects",
                query_text="Aprikosenmarmelade",
                limit=2,
            )
            payload_by_id = {payload["memory_id"]: payload for payload in payloads}

        self.assertIn("fact:jam_preference_new", payload_by_id)
        self.assertEqual(payload_by_id["fact:jam_preference_new"]["status"], "active")
        self.assertTrue(payload_by_id["fact:jam_preference_new"]["confirmed_by_user"])
        self.assertNotEqual(
            payload_by_id["fact:jam_preference_new"]["status"],
            "uncertain",
        )

    def test_load_catalog_payload_skips_repeated_invalid_current_head_reads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None
            remote_state.snapshots["user_context"] = {
                "schema": "twinr_user_context_catalog_v3",
                "version": 3,
                "items_count": 0,
                "segments": [],
            }
            invalid_uri = remote_catalog._catalog_head_uri(snapshot_kind="user_context")
            remote_state.client.records_by_uri[invalid_uri] = {
                "document_id": "doc-invalid-user-context",
                "payload": {
                    "schema": "twinr_user_context_snapshot_v1",
                    "body": {"entries": []},
                },
                "metadata": {},
                "content": "{}",
                "uri": invalid_uri,
            }

            first_payload = remote_catalog.load_catalog_payload(snapshot_kind="user_context")
            first_fetch_calls = remote_state.client.fetch_full_document_calls
            second_payload = remote_catalog.load_catalog_payload(snapshot_kind="user_context")
            second_fetch_calls = remote_state.client.fetch_full_document_calls

        self.assertEqual(first_payload, remote_state.snapshots["user_context"])
        self.assertEqual(second_payload, remote_state.snapshots["user_context"])
        self.assertGreaterEqual(first_fetch_calls, 1)
        self.assertEqual(second_fetch_calls, first_fetch_calls)

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
        self.assertGreaterEqual(remote_state.client.topk_records_calls, 1)
        self.assertGreaterEqual(remote_state.client.retrieve_calls, 1)

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
        self.assertEqual(payload["allowed_indexes"], ["fulltext", "temporal", "tags"])
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
        self.assertEqual(payload["allowed_indexes"], ["fulltext", "temporal", "tags"])
        self.assertNotIn("namespace", payload)
        self.assertNotIn("scope_ref", payload)
        self.assertEqual(len(payload["allowed_doc_ids"]), 2)

    def test_search_catalog_entries_retrieve_fallback_keeps_lightweight_non_ann_indexes(self) -> None:
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

            relevant = remote_catalog.search_catalog_entries(
                snapshot_kind="objects",
                query_text="Janina",
                limit=1,
            )

        self.assertEqual(tuple(entry.item_id for entry in relevant), ("fact:janina_spouse",))
        payload = remote_state.client.retrieve_payloads[-1]
        self.assertEqual(payload["allowed_indexes"], ["fulltext", "temporal", "tags"])

    def test_load_selection_item_payloads_topk_batch_uses_fulltext_only_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            remote_state = _FakeRemoteState()
            remote_state.client.supports_topk_records = True
            store = LongTermStructuredStore(base_path=root / "state" / "chonkydb", remote_state=remote_state)
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
                ),
                conflicts=(),
                archived_objects=(),
            )
            reader_store = LongTermStructuredStore(base_path=root / "reader" / "state" / "chonkydb", remote_state=remote_state)
            remote_catalog = reader_store._remote_catalog
            assert remote_catalog is not None

            payloads = remote_catalog.load_selection_item_payloads(
                snapshot_kind="objects",
                item_ids=("fact:jam_preference_old",),
            )

        self.assertEqual(len(payloads), 1)
        payload = remote_state.client.topk_records_payloads[-1]
        self.assertEqual(payload["query_text"], "__allowed_doc_ids__")
        self.assertEqual(payload["allowed_indexes"], ["fulltext"])

    def test_load_selection_item_payloads_retrieve_batch_uses_fulltext_only_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            remote_state = _FakeRemoteState()
            store = LongTermStructuredStore(base_path=root / "state" / "chonkydb", remote_state=remote_state)
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
                ),
                conflicts=(),
                archived_objects=(),
            )
            reader_store = LongTermStructuredStore(base_path=root / "reader" / "state" / "chonkydb", remote_state=remote_state)
            remote_catalog = reader_store._remote_catalog
            assert remote_catalog is not None

            payloads = remote_catalog.load_selection_item_payloads(
                snapshot_kind="objects",
                item_ids=("fact:jam_preference_old",),
            )

        self.assertEqual(len(payloads), 1)
        payload = remote_state.client.retrieve_payloads[-1]
        self.assertEqual(payload["query_text"], "__allowed_doc_ids__")
        self.assertEqual(payload["allowed_indexes"], ["fulltext"])
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
            original_load_objects = store.load_objects
            store_type = type(store)
            with patch.object(
                store_type,
                "load_objects",
                side_effect=AssertionError("Review must not hydrate the full object snapshot."),
            ), patch.object(
                store_type,
                "load_objects_fine_grained",
                new=lambda _self: original_load_objects(),
            ):
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
            original_load_objects = store.load_objects
            original_load_conflicts = store.load_conflicts
            store_type = type(store)
            with patch.object(
                store_type,
                "load_objects",
                side_effect=AssertionError("Invalidate must not hydrate the full object snapshot."),
            ), patch.object(
                store_type,
                "load_conflicts",
                side_effect=AssertionError("Invalidate must not hydrate the full conflict snapshot."),
            ), patch.object(
                store_type,
                "load_objects_fine_grained",
                new=lambda _self: original_load_objects(),
            ), patch.object(
                store_type,
                "load_conflicts_fine_grained",
                new=lambda _self: original_load_conflicts(),
            ), patch.object(
                store_type,
                "load_objects_fine_grained_for_write",
                new=lambda _self: original_load_objects(),
            ), patch.object(
                store_type,
                "load_conflicts_fine_grained_for_write",
                new=lambda _self: original_load_conflicts(),
            ), patch.object(
                store_type,
                "load_archived_objects_fine_grained_for_write",
                new=lambda _self: (),
            ):
                mutation = store.invalidate_object("fact:corinna_phone_new", reason="User said this is outdated.")
                store.apply_memory_mutation(mutation)
                objects = {item.memory_id: item for item in original_load_objects()}
                conflicts = original_load_conflicts()

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
            original_load_objects = store.load_objects
            original_load_conflicts = store.load_conflicts
            store_type = type(store)
            with patch.object(
                store_type,
                "load_objects",
                side_effect=AssertionError("Delete must not hydrate the full object snapshot."),
            ), patch.object(
                store_type,
                "load_conflicts",
                side_effect=AssertionError("Delete must not hydrate the full conflict snapshot."),
            ), patch.object(
                store_type,
                "load_objects_fine_grained",
                new=lambda _self: original_load_objects(),
            ), patch.object(
                store_type,
                "load_conflicts_fine_grained",
                new=lambda _self: original_load_conflicts(),
            ), patch.object(
                store_type,
                "load_objects_fine_grained_for_write",
                new=lambda _self: original_load_objects(),
            ), patch.object(
                store_type,
                "load_conflicts_fine_grained_for_write",
                new=lambda _self: original_load_conflicts(),
            ), patch.object(
                store_type,
                "load_archived_objects_fine_grained_for_write",
                new=lambda _self: (),
            ):
                mutation = store.delete_object("fact:janina_wife")
                store.apply_memory_mutation(mutation)
                objects = {item.memory_id: item for item in original_load_objects()}

        self.assertEqual(mutation.deleted_memory_ids, ("fact:janina_wife",))
        self.assertNotIn("fact:janina_wife", objects)
        self.assertEqual(objects["summary:janina_thread"].conflicts_with, ())
        self.assertEqual(objects["summary:janina_thread"].supersedes, ())

    def test_apply_memory_mutation_bootstraps_fresh_required_remote_namespace_without_snapshot_blob_reads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.config.project_root = str(project_root)
            remote_state.client = _CurrentHead404Client()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            load_snapshot_calls: list[dict[str, object]] = []

            def _load_snapshot(
                *,
                snapshot_kind: str,
                local_path=None,
                prefer_cached_document_id: bool = True,
            ):
                del local_path
                load_snapshot_calls.append(
                    {
                        "snapshot_kind": snapshot_kind,
                        "prefer_cached_document_id": prefer_cached_document_id,
                    }
                )
                raise AssertionError("Fresh required-remote mutation must not revive snapshot blob reads.")

            remote_state.load_snapshot = _load_snapshot  # type: ignore[assignment]
            store = LongTermStructuredStore(
                base_path=project_root / "state" / "chonkydb",
                remote_state=remote_state,
            )
            mutation = LongTermMemoryMutationResultV1(
                action="confirm",
                target_memory_id="fact:janina_spouse",
                updated_objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:janina_spouse",
                        kind="relationship_fact",
                        summary="Janina is the user's wife.",
                        source=_source(),
                        status="active",
                        confidence=0.99,
                        slot_key="relationship:user:main:wife",
                        value_key="person:janina",
                        confirmed_by_user=True,
                    ),
                ),
                remaining_conflicts=(),
            )
            store_type = type(store)

            with patch.object(
                store_type,
                "load_objects",
                side_effect=AssertionError("Fresh required-remote mutation must not hydrate object snapshots."),
            ), patch.object(
                store_type,
                "load_current_state_fine_grained_for_write",
                side_effect=AssertionError("Fresh required-remote mutation must not hydrate the full current state."),
            ), patch.object(
                store_type,
                "load_conflicts",
                side_effect=AssertionError("Fresh required-remote mutation must not hydrate conflict snapshots."),
            ), patch.object(
                store_type,
                "load_archived_objects",
                side_effect=AssertionError("Fresh required-remote mutation must not hydrate archive snapshots."),
            ), patch.object(
                store_type,
                "write_snapshot",
                side_effect=AssertionError("Fresh required-remote mutation must not rewrite full snapshots."),
            ):
                store.apply_memory_mutation(mutation)
                current_state = store.load_current_state_fine_grained()

        self.assertEqual(load_snapshot_calls, [])
        self.assertEqual(tuple(item.memory_id for item in current_state.objects), ("fact:janina_spouse",))
        self.assertEqual(current_state.conflicts, ())
        self.assertEqual(current_state.archived_objects, ())
        self.assertIsNotNone(_current_catalog_head_payload(store, snapshot_kind="objects"))
        self.assertIsNotNone(_current_catalog_head_payload(store, snapshot_kind="conflicts"))
        self.assertIsNotNone(_current_catalog_head_payload(store, snapshot_kind="archive"))

    def test_apply_conflict_resolution_bootstraps_fresh_required_remote_namespace_without_full_state_rewrites(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.config.project_root = str(project_root)
            remote_state.client = _CurrentHead404Client()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            load_snapshot_calls: list[dict[str, object]] = []

            def _load_snapshot(
                *,
                snapshot_kind: str,
                local_path=None,
                prefer_cached_document_id: bool = True,
            ):
                del local_path
                load_snapshot_calls.append(
                    {
                        "snapshot_kind": snapshot_kind,
                        "prefer_cached_document_id": prefer_cached_document_id,
                    }
                )
                raise AssertionError("Fresh required-remote conflict resolution must not revive snapshot blob reads.")

            remote_state.load_snapshot = _load_snapshot  # type: ignore[assignment]
            store = LongTermStructuredStore(
                base_path=project_root / "state" / "chonkydb",
                remote_state=remote_state,
            )
            resolution = LongTermConflictResolutionV1(
                slot_key="contact:person:corinna_maier:phone",
                selected_memory_id="fact:corinna_phone_new",
                updated_objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:corinna_phone_old",
                        kind="contact_method_fact",
                        summary=f"Corinna Maier can be reached at {_TEST_CORINNA_PHONE_OLD}.",
                        source=_source(),
                        status="superseded",
                        confidence=0.95,
                        slot_key="contact:person:corinna_maier:phone",
                        value_key=_TEST_CORINNA_PHONE_OLD,
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:corinna_phone_new",
                        kind="contact_method_fact",
                        summary=f"Corinna Maier can be reached at {_TEST_CORINNA_PHONE_NEW}.",
                        source=_source(),
                        status="active",
                        confidence=0.99,
                        confirmed_by_user=True,
                        slot_key="contact:person:corinna_maier:phone",
                        value_key=_TEST_CORINNA_PHONE_NEW,
                    ),
                ),
                remaining_conflicts=(),
                deleted_conflict_slot_keys=("contact:person:corinna_maier:phone",),
            )
            store_type = type(store)

            with patch.object(
                store_type,
                "load_objects",
                side_effect=AssertionError("Fresh required-remote conflict resolution must not hydrate object snapshots."),
            ), patch.object(
                store_type,
                "load_current_state_fine_grained_for_write",
                side_effect=AssertionError("Fresh required-remote conflict resolution must not hydrate the full current state."),
            ), patch.object(
                store_type,
                "load_conflicts",
                side_effect=AssertionError("Fresh required-remote conflict resolution must not hydrate conflict snapshots."),
            ), patch.object(
                store_type,
                "load_archived_objects",
                side_effect=AssertionError("Fresh required-remote conflict resolution must not hydrate archive snapshots."),
            ), patch.object(
                store_type,
                "write_snapshot",
                side_effect=AssertionError("Fresh required-remote conflict resolution must not rewrite full snapshots."),
            ):
                store.apply_conflict_resolution(resolution)
                current_state = store.load_current_state_fine_grained()

        self.assertEqual(load_snapshot_calls, [])
        self.assertEqual(
            tuple(item.memory_id for item in current_state.objects),
            ("fact:corinna_phone_new", "fact:corinna_phone_old"),
        )
        self.assertEqual(current_state.conflicts, ())
        self.assertEqual(current_state.archived_objects, ())
        self.assertIsNotNone(_current_catalog_head_payload(store, snapshot_kind="objects"))
        self.assertIsNotNone(_current_catalog_head_payload(store, snapshot_kind="conflicts"))

    def test_fine_grained_current_state_loaders_use_remote_catalog_without_snapshot_blob_reads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LongTermStructuredStore.from_config(_config(temp_dir))
            object_payload = LongTermMemoryObjectV1(
                memory_id="fact:janina_wife",
                kind="relationship_fact",
                summary="Janina is the user's wife.",
                source=_source(),
                status="active",
                confidence=0.99,
                slot_key="relationship:user:main:wife",
                value_key="person:janina",
            ).to_payload()
            conflict = LongTermMemoryConflictV1(
                slot_key="contact:person:corinna_maier:phone",
                candidate_memory_id="fact:corinna_phone_new",
                existing_memory_ids=("fact:corinna_phone_old",),
                question="Which phone number should I use for Corinna Maier?",
                reason="Conflicting phone numbers exist.",
            )
            archived_payload = LongTermMemoryObjectV1(
                memory_id="episode:old_weather",
                kind="episode",
                summary="We talked about the weather last week.",
                source=_source(),
                status="expired",
                confidence=0.61,
                archived_at="2026-03-29T18:00:00+00:00",
            ).to_payload()

            class _FineGrainedCatalog:
                def __init__(self) -> None:
                    self.calls: list[tuple[str, str, tuple[str, ...]]] = []

                def enabled(self) -> bool:
                    return True

                def catalog_available(self, *, snapshot_kind: str) -> bool:
                    self.calls.append(("catalog_available", snapshot_kind, ()))
                    return True

                def load_catalog_entries(self, *, snapshot_kind: str):
                    self.calls.append(("load_catalog_entries", snapshot_kind, ()))
                    item_ids_by_kind = {
                        "objects": (object_payload["memory_id"],),
                        "conflicts": (conflict.catalog_item_id(),),
                        "archive": (archived_payload["memory_id"],),
                    }
                    return tuple(SimpleNamespace(item_id=item_id) for item_id in item_ids_by_kind[snapshot_kind])

                def load_item_payloads(self, *, snapshot_kind: str, item_ids):
                    ordered_item_ids = tuple(item_ids)
                    self.calls.append(("load_item_payloads", snapshot_kind, ordered_item_ids))
                    payloads_by_kind = {
                        "objects": {object_payload["memory_id"]: object_payload},
                        "conflicts": {conflict.catalog_item_id(): conflict.to_payload()},
                        "archive": {archived_payload["memory_id"]: archived_payload},
                    }
                    return tuple(payloads_by_kind[snapshot_kind][item_id] for item_id in ordered_item_ids)

            catalog = _FineGrainedCatalog()
            store._remote_catalog = catalog
            store_type = type(store)

            with patch.object(
                store_type,
                "load_objects",
                side_effect=AssertionError("Objects snapshot blob read is forbidden."),
            ), patch.object(
                store_type,
                "load_conflicts",
                side_effect=AssertionError("Conflicts snapshot blob read is forbidden."),
            ), patch.object(
                store_type,
                "load_archived_objects",
                side_effect=AssertionError("Archive snapshot blob read is forbidden."),
            ):
                objects = store.load_objects_fine_grained()
                conflicts = store.load_conflicts_fine_grained()
                archived = store.load_archived_objects_fine_grained()

        self.assertEqual(tuple(item.memory_id for item in objects), ("fact:janina_wife",))
        self.assertEqual(tuple(item.slot_key for item in conflicts), ("contact:person:corinna_maier:phone",))
        self.assertEqual(tuple(item.memory_id for item in archived), ("episode:old_weather",))
        self.assertTrue(any(call[0] == "load_item_payloads" and call[1] == "objects" for call in catalog.calls))
        self.assertTrue(any(call[0] == "load_item_payloads" and call[1] == "conflicts" for call in catalog.calls))
        self.assertTrue(any(call[0] == "load_item_payloads" and call[1] == "archive" for call in catalog.calls))

    def test_fine_grained_archive_loader_accepts_legitimate_empty_raw_archive_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.snapshots["archive"] = {
                "schema": "twinr_memory_archive_store",
                "version": 1,
                "objects": [],
            }
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)

            class _ArchiveCatalogUnavailable:
                def enabled(self) -> bool:
                    return True

                def catalog_available(self, *, snapshot_kind: str) -> bool:
                    return snapshot_kind != "archive"

                def _load_catalog_head_payload(self, *, snapshot_kind: str):
                    del snapshot_kind
                    return None

                def is_catalog_payload(self, *, snapshot_kind: str, payload) -> bool:
                    del snapshot_kind
                    del payload
                    return False

            store._remote_catalog = _ArchiveCatalogUnavailable()
            store_type = type(store)

            with patch.object(
                store_type,
                "load_archived_objects",
                side_effect=AssertionError("Archive snapshot blob read is forbidden."),
            ):
                archived = store.load_archived_objects_fine_grained()

        self.assertEqual(archived, ())

    def test_load_current_state_fine_grained_uses_fine_grained_loaders_without_snapshot_blob_reads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LongTermStructuredStore.from_config(_config(temp_dir))
            expected_object = LongTermMemoryObjectV1(
                memory_id="fact:janina_wife",
                kind="relationship_fact",
                summary="Janina is the user's wife.",
                source=_source(),
                status="active",
                confidence=0.99,
            )
            expected_conflict = LongTermMemoryConflictV1(
                slot_key="contact:person:corinna_maier:phone",
                candidate_memory_id="fact:corinna_phone_new",
                existing_memory_ids=("fact:corinna_phone_old",),
                question="Which phone number should I use for Corinna Maier?",
                reason="Conflicting phone numbers exist.",
            )
            expected_archive = LongTermMemoryObjectV1(
                memory_id="episode:old_weather",
                kind="episode",
                summary="We talked about the weather last week.",
                source=_source(),
                status="expired",
                confidence=0.61,
                archived_at="2026-03-29T18:00:00+00:00",
            )
            store_type = type(store)

            with patch.object(
                store_type,
                "load_objects",
                side_effect=AssertionError("Objects snapshot blob read is forbidden."),
            ), patch.object(
                store_type,
                "load_conflicts",
                side_effect=AssertionError("Conflicts snapshot blob read is forbidden."),
            ), patch.object(
                store_type,
                "load_archived_objects",
                side_effect=AssertionError("Archive snapshot blob read is forbidden."),
            ), patch.object(
                store_type,
                "load_objects_fine_grained",
                new=lambda _self: (expected_object,),
            ), patch.object(
                store_type,
                "load_conflicts_fine_grained",
                new=lambda _self: (expected_conflict,),
            ), patch.object(
                store_type,
                "load_archived_objects_fine_grained",
                new=lambda _self: (expected_archive,),
            ):
                current_state = store.load_current_state_fine_grained()

        self.assertEqual(current_state.objects, (expected_object,))
        self.assertEqual(current_state.conflicts, (expected_conflict,))
        self.assertEqual(current_state.archived_objects, (expected_archive,))

    def test_load_current_state_fine_grained_accepts_missing_current_heads_for_fresh_required_remote_namespace(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.required = True
            remote_state.client = _NoDocumentsFullScope404ChonkyClient()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            store_type = type(store)

            with patch.object(
                store_type,
                "load_objects",
                side_effect=AssertionError("Fresh required-remote namespaces must not hydrate object snapshots."),
            ), patch.object(
                store_type,
                "load_conflicts",
                side_effect=AssertionError("Fresh required-remote namespaces must not hydrate conflict snapshots."),
            ), patch.object(
                store_type,
                "load_archived_objects",
                side_effect=AssertionError("Fresh required-remote namespaces must not hydrate archive snapshots."),
            ):
                current_state = store.load_current_state_fine_grained()

        self.assertEqual(current_state.objects, ())
        self.assertEqual(current_state.conflicts, ())
        self.assertEqual(current_state.archived_objects, ())
        self.assertEqual(remote_state.client.topk_records_calls, 0)
        self.assertEqual(remote_state.client.item_fetch_attempts, 0)
        self.assertEqual(remote_state.client.segment_fetch_attempts, 0)

    def test_load_current_state_fine_grained_prefers_same_process_catalog_cache_after_remote_head_visibility_lag(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _LaggingSnapshotVisibilityRemoteState()
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            updated_object = LongTermMemoryObjectV1(
                memory_id="fact:jam_preference_new",
                kind="fact",
                summary="Inzwischen magst du lieber Aprikosenmarmelade.",
                details="Neuere Vorliebe fuer das Fruehstueck.",
                source=_source(),
                status="active",
                confidence=0.99,
                slot_key="preference:breakfast:jam",
                value_key="apricot",
                confirmed_by_user=True,
            )

            store.ensure_remote_snapshots()
            store.write_snapshot(objects=(updated_object,))
            remote_catalog = store._remote_catalog
            assert remote_catalog is not None

            with patch.object(
                type(remote_catalog),
                "catalog_available",
                side_effect=AssertionError("Same-process fine-grained current-state reads must not depend on refetching the remote head."),
            ):
                current_state = store.load_current_state_fine_grained()

            remote_state.promote_pending("objects")
            loaded_after_remote_catchup = store.load_current_state_fine_grained()

        self.assertEqual(tuple(item.memory_id for item in current_state.objects), ("fact:jam_preference_new",))
        self.assertEqual(current_state.objects[0].status, "active")
        self.assertTrue(current_state.objects[0].confirmed_by_user)
        self.assertEqual(current_state.conflicts, ())
        self.assertEqual(current_state.archived_objects, ())
        self.assertEqual(tuple(item.memory_id for item in loaded_after_remote_catchup.objects), ("fact:jam_preference_new",))
        self.assertEqual(loaded_after_remote_catchup.objects[0].status, "active")
        self.assertTrue(loaded_after_remote_catchup.objects[0].confirmed_by_user)

    def test_load_active_working_set_uses_source_event_id_projection_without_snapshot_blob_reads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            existing_object = LongTermMemoryObjectV1(
                memory_id="event:janina_eye_treatment",
                kind="event",
                summary="Janina has eye laser treatment today.",
                source=LongTermSourceRefV1(
                    source_type="conversation_turn",
                    event_ids=("turn:janina-eye-20260330",),
                ),
                status="active",
                confidence=0.98,
                slot_key="event:janina_eye_treatment",
                value_key="2026-03-30",
                confirmed_by_user=True,
            )
            store.write_snapshot(objects=(existing_object,))
            candidate = LongTermMemoryObjectV1(
                memory_id="summary:janina_eye_follow_up",
                kind="summary",
                summary="Follow up on Janina's eye treatment.",
                source=LongTermSourceRefV1(
                    source_type="conversation_turn",
                    event_ids=("turn:new-summary",),
                ),
                status="candidate",
                confidence=0.72,
                slot_key="summary:janina_eye_follow_up",
                value_key="follow_up",
            )
            store_type = type(store)

            with patch.object(
                store_type,
                "load_objects",
                side_effect=AssertionError("Active working-set loads must not hydrate the full object snapshot."),
            ), patch.object(
                store_type,
                "load_conflicts",
                side_effect=AssertionError("Active working-set loads must not hydrate the full conflict snapshot."),
            ), patch.object(
                store_type,
                "load_archived_objects",
                side_effect=AssertionError("Active working-set loads must not hydrate the full archive snapshot."),
            ):
                working_set = store.load_active_working_set(
                    candidate_objects=(candidate,),
                    event_ids=("turn:janina-eye-20260330",),
                )

        self.assertEqual(
            tuple(item.memory_id for item in working_set.objects),
            ("event:janina_eye_treatment",),
        )
        self.assertEqual(working_set.conflicts, ())
        self.assertEqual(working_set.archived_objects, ())

    def test_load_objects_by_projection_filter_uses_catalog_projection_without_snapshot_blob_reads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            old_episode = LongTermMemoryObjectV1(
                memory_id="episode:old_weather",
                kind="episode",
                summary="We talked about the weather a month ago.",
                source=LongTermSourceRefV1(
                    source_type="conversation_turn",
                    event_ids=("turn:old-weather",),
                ),
                status="active",
                confidence=0.72,
                created_at="2026-02-01T10:00:00+00:00",
                updated_at="2026-02-01T10:00:00+00:00",
            )
            fresh_fact = LongTermMemoryObjectV1(
                memory_id="fact:janina_eye_treatment",
                kind="event",
                summary="Janina has eye laser treatment today.",
                source=LongTermSourceRefV1(
                    source_type="conversation_turn",
                    event_ids=("turn:janina-eye-20260330",),
                ),
                status="active",
                confidence=0.98,
                slot_key="event:janina_eye_treatment",
                value_key="2026-03-30",
                confirmed_by_user=True,
            )
            store.write_snapshot(objects=(old_episode, fresh_fact))
            retention_policy = LongTermRetentionPolicy()
            store_type = type(store)

            with patch.object(
                store_type,
                "load_objects",
                side_effect=AssertionError("Projection-filtered retention loads must not hydrate the full object snapshot."),
            ), patch.object(
                store_type,
                "load_objects_fine_grained",
                side_effect=AssertionError("Projection-filtered retention loads must not sweep the full fine-grained object state."),
            ):
                selected = store.load_objects_by_projection_filter(
                    predicate=lambda projection: retention_policy.projection_requires_action(
                        projection,
                        now=datetime(2026, 3, 16, 10, 0, tzinfo=ZoneInfo("UTC")),
                    )
                )

        self.assertEqual(tuple(item.memory_id for item in selected), ("episode:old_weather",))

    def test_commit_active_delta_updates_remote_current_head_without_full_snapshot_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.read_client = remote_state.client
            remote_state.write_client = remote_state.client
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)
            updated_object = LongTermMemoryObjectV1(
                memory_id="fact:janina_phone_new",
                kind="fact",
                summary="Janina's new phone number ends with 44.",
                details="Confirmed during the latest clarification turn.",
                source=LongTermSourceRefV1(
                    source_type="conversation_turn",
                    event_ids=("turn:janina-phone-20260330",),
                ),
                status="active",
                confidence=0.99,
                confirmed_by_user=True,
                slot_key="contact:janina:phone",
                value_key="ends_with_44",
            )
            store_type = type(store)

            with patch.object(
                store_type,
                "write_snapshot",
                side_effect=AssertionError("Active delta commits must not rewrite the full current state."),
            ):
                store.commit_active_delta(object_upserts=(updated_object,))
                loaded = store.load_objects_by_ids(("fact:janina_phone_new",))

        self.assertEqual(tuple(item.memory_id for item in loaded), ("fact:janina_phone_new",))
        self.assertEqual(loaded[0].status, "active")
        self.assertTrue(loaded[0].confirmed_by_user)


if __name__ == "__main__":
    unittest.main()
