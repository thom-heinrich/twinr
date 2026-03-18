from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys
import tempfile
from threading import Event, Lock, Thread
import time
from types import SimpleNamespace
import unittest
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test.longterm_test_program import make_test_extractor
from twinr.config import TwinrConfig
from twinr.memory.longterm.reasoning.consolidator import LongTermMemoryConsolidator
from twinr.memory.longterm.core.models import (
    LongTermConsolidationResultV1,
    LongTermGraphEdgeCandidateV1,
    LongTermMemoryConflictV1,
    LongTermMemoryObjectV1,
    LongTermSourceRefV1,
)
from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError
from twinr.memory.longterm.storage.store import LongTermStructuredStore, _write_json_atomic
from twinr.memory.longterm.reasoning.truth import LongTermTruthMaintainer
from twinr.text_utils import retrieval_terms


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


class _FakeChonkyClient:
    def __init__(self, *, max_items_per_bulk: int | None = None, max_request_bytes: int | None = None) -> None:
        self._next_document_id = 1
        self.max_items_per_bulk = max_items_per_bulk
        self.max_request_bytes = max_request_bytes
        self.bulk_calls = 0
        self.retrieve_calls = 0
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


class _LiveShapeChonkyClient(_FakeChonkyClient):
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

    def test_remote_primary_store_persists_conflicts_into_remote_catalog(self) -> None:
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
                summary="Corinna Maier can be reached at +491761234.",
                source=_source(),
                status="active",
                confidence=0.95,
                slot_key="contact:person:corinna_maier:phone",
                value_key="+491761234",
            )
            new_phone = LongTermMemoryObjectV1(
                memory_id="fact:corinna_phone_new",
                kind="contact_method_fact",
                summary="Corinna Maier can be reached at +4940998877.",
                source=_source(),
                status="uncertain",
                confidence=0.92,
                slot_key="contact:person:corinna_maier:phone",
                value_key="+4940998877",
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
        self.assertEqual(remote_state.snapshots["conflicts"]["schema"], "twinr_memory_conflict_catalog_v3")
        self.assertEqual(remote_state.snapshots["conflicts"]["items_count"], 1)
        self.assertEqual(len(remote_state.snapshots["conflicts"]["segments"]), 1)
        self.assertEqual(_count_remote_records_with_schema(remote_state.client, "twinr_memory_conflict_record_v2"), 1)
        self.assertEqual(_count_remote_records_with_schema(remote_state.client, "twinr_memory_conflict_catalog_segment_v1"), 1)

    def test_ensure_remote_snapshots_seeds_empty_remote_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb", remote_state=remote_state)

            ensured = store.ensure_remote_snapshots()

        self.assertEqual(set(ensured), {"objects", "conflicts", "archive"})
        self.assertEqual(remote_state.snapshots["conflicts"]["schema"], "twinr_memory_conflict_catalog_v3")
        self.assertEqual(remote_state.snapshots["archive"]["schema"], "twinr_memory_archive_catalog_v3")
        self.assertEqual(remote_state.snapshots["objects"]["schema"], "twinr_memory_object_catalog_v3")
        self.assertEqual(remote_state.snapshots["conflicts"]["version"], 3)
        self.assertEqual(remote_state.snapshots["archive"]["version"], 3)
        self.assertEqual(remote_state.snapshots["objects"]["version"], 3)
        self.assertEqual(remote_state.snapshots["conflicts"]["items_count"], 0)
        self.assertEqual(remote_state.snapshots["archive"]["items_count"], 0)
        self.assertEqual(remote_state.snapshots["objects"]["items_count"], 0)
        self.assertEqual(remote_state.snapshots["conflicts"]["segments"], [])
        self.assertEqual(remote_state.snapshots["archive"]["segments"], [])
        self.assertEqual(remote_state.snapshots["objects"]["segments"], [])
        self.assertIn("written_at", remote_state.snapshots["conflicts"])
        self.assertIn("written_at", remote_state.snapshots["archive"])
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
                        summary="Corinna Maier can be reached at +491761234.",
                        source=_source(),
                        status="active",
                        confidence=0.95,
                        slot_key="contact:person:corinna_maier:phone",
                        value_key="+491761234",
                    ),
                ),
                deferred_objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:corinna_phone_new",
                        kind="contact_method_fact",
                        summary="Corinna Maier can be reached at +4940998877.",
                        source=_source(),
                        status="uncertain",
                        confidence=0.92,
                        slot_key="contact:person:corinna_maier:phone",
                        value_key="+4940998877",
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
