from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import types
import unittest


_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _seed_namespace_package(name: str, path: Path) -> ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    module.__path__ = [str(path)]
    module.__package__ = name
    return module


def _bootstrap_remote_catalog_imports() -> tuple[type[object], type[object] | None, type[object], type[object]]:
    """Import remote-catalog modules without executing the heavy memory package root.

    The shared worktree currently contains an unrelated in-progress config
    refactor that breaks package-root imports before these tests even reach the
    storage layer. Seed the minimum package + config stubs needed so this suite
    can freeze `remote_catalog` behavior in isolation.
    """

    _seed_namespace_package("twinr.memory", _SRC / "twinr" / "memory")
    _seed_namespace_package("twinr.memory.longterm", _SRC / "twinr" / "memory" / "longterm")
    _seed_namespace_package("twinr.memory.longterm.core", _SRC / "twinr" / "memory" / "longterm" / "core")
    _seed_namespace_package("twinr.memory.longterm.storage", _SRC / "twinr" / "memory" / "longterm" / "storage")
    _seed_namespace_package("twinr.memory.chonkydb", _SRC / "twinr" / "memory" / "chonkydb")
    _seed_namespace_package("twinr.agent.base_agent", _SRC / "twinr" / "agent" / "base_agent")

    config_module = types.ModuleType("twinr.agent.base_agent.config")

    @dataclass(frozen=True, slots=True)
    class TwinrConfig:
        chonkydb_base_url: str | None = None
        chonkydb_api_key: str | None = None
        chonkydb_api_key_header: str | None = None
        chonkydb_allow_bearer_auth: bool = False
        chonkydb_timeout_s: float = 10.0
        chonkydb_max_response_bytes: int = 32 * 1024 * 1024

    config_module.TwinrConfig = TwinrConfig
    sys.modules["twinr.agent.base_agent.config"] = config_module

    client_module = importlib.import_module("twinr.memory.chonkydb.client")
    models_module = importlib.import_module("twinr.memory.chonkydb.models")
    chonkydb_package = sys.modules["twinr.memory.chonkydb"]
    for export in ("ChonkyDBClient", "ChonkyDBError", "chonkydb_data_path"):
        setattr(chonkydb_package, export, getattr(client_module, export))
    for export in (
        "ChonkyDBAuthInfo",
        "ChonkyDBBulkRecordRequest",
        "ChonkyDBConnectionConfig",
        "ChonkyDBRecordItem",
        "ChonkyDBRetrieveRequest",
        "ChonkyDBTopKRecordsRequest",
    ):
        setattr(chonkydb_package, export, getattr(models_module, export))

    remote_catalog_module = importlib.import_module("twinr.memory.longterm.storage.remote_catalog")
    core_models_module = importlib.import_module("twinr.memory.longterm.core.models")
    remote_state_module = importlib.import_module("twinr.memory.longterm.storage.remote_state")
    public_store = remote_catalog_module.LongTermRemoteCatalogStore
    try:
        base_module = importlib.import_module("twinr.memory.longterm.storage._remote_catalog.base")
        base_store = base_module.LongTermRemoteCatalogStoreBase
    except Exception:
        base_store = None
    return (
        public_store,
        base_store,
        core_models_module.LongTermMemoryObjectV1,
        remote_state_module.LongTermRemoteUnavailableError,
    )


LongTermRemoteCatalogStore, LongTermRemoteCatalogStoreBase, LongTermMemoryObjectV1, LongTermRemoteUnavailableError = (
    _bootstrap_remote_catalog_imports()
)

LongTermSourceRefV1 = importlib.import_module("twinr.memory.longterm.core.models").LongTermSourceRefV1


class _FakeChonkyClient:
    def __init__(self, *, supports_topk_records: bool = False) -> None:
        self._next_document_id = 1
        self.supports_topk_records = supports_topk_records
        self.records_by_document_id: dict[str, dict[str, object]] = {}
        self.records_by_uri: dict[str, dict[str, object]] = {}
        self.topk_records_calls = 0
        self.topk_records_payloads: list[dict[str, object]] = []

    def store_records_bulk(self, request):
        items = tuple(getattr(request, "items", ()))
        response_items: list[dict[str, object]] = []
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
        allowed = {str(value) for value in payload.get("allowed_doc_ids", ()) if str(value)}
        ranked: list[dict[str, object]] = []
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
                    "metadata": dict(record.get("metadata") or {}),
                    "payload": dict(record.get("payload") or {}),
                    "content": record.get("content"),
                }
            )
        return SimpleNamespace(
            results=tuple(SimpleNamespace(**item) for item in ranked),
            raw={"results": [dict(item) for item in ranked]},
        )

    def topk_records(self, request):
        payload = request.to_payload() if hasattr(request, "to_payload") else dict(request)
        query_text = str(payload.get("query_text") or "").lower()
        allowed = {str(value) for value in payload.get("allowed_doc_ids", ()) if str(value)}
        self.topk_records_calls += 1
        self.topk_records_payloads.append(dict(payload))
        ranked: list[dict[str, object]] = []
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
                    "metadata": dict(record.get("metadata") or {}),
                    "payload": dict(record.get("payload") or {}),
                    "content": record.get("content"),
                }
            )
        return SimpleNamespace(
            results=tuple(SimpleNamespace(**item) for item in ranked),
            raw={"results": [dict(item) for item in ranked]},
        )


class _MetadataOnlyRetrieveClient(_FakeChonkyClient):
    def retrieve(self, request):
        response = super().retrieve(request)
        normalized_results: list[dict[str, object]] = []
        for result in getattr(response, "results", ()):
            metadata = dict(getattr(result, "metadata", {}) or {})
            metadata.pop("twinr_payload", None)
            metadata.pop("twinr_payload_sha256", None)
            normalized_results.append(
                {
                    "payload_id": getattr(result, "payload_id", None),
                    "metadata": metadata,
                    "content": "",
                }
            )
        return SimpleNamespace(
            results=tuple(SimpleNamespace(**item) for item in normalized_results),
            raw={"results": normalized_results},
        )


class _FakeRemoteState:
    def __init__(self, client: _FakeChonkyClient | None = None) -> None:
        self.client = client or _FakeChonkyClient()
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
            long_term_memory_remote_read_cache_ttl_s=60.0,
        )
        self.snapshots: dict[str, dict[str, object]] = {}

    def load_snapshot(self, *, snapshot_kind: str, local_path=None):
        del local_path
        payload = self.snapshots.get(snapshot_kind)
        return dict(payload) if isinstance(payload, dict) else None

    def save_snapshot(self, *, snapshot_kind: str, payload):
        self.snapshots[snapshot_kind] = dict(payload)


def _make_object_payload(
    *,
    memory_id: str,
    summary: str,
    status: str = "active",
    created_at: str = "2026-03-27T10:00:00+00:00",
    updated_at: str = "2026-03-27T10:00:00+00:00",
) -> dict[str, object]:
    return LongTermMemoryObjectV1(
        memory_id=memory_id,
        kind="fact",
        summary=summary,
        source=LongTermSourceRefV1(source_type="manual", event_ids=(memory_id,)),
        status=status,
        slot_key="preference",
        value_key="jam",
        created_at=created_at,
        updated_at=updated_at,
        attributes={"household_member": "Corinna"},
    ).to_payload()


def _item_id_getter(payload: dict[str, object]) -> str | None:
    return str(payload.get("memory_id") or "")


def _metadata_builder(payload: dict[str, object]) -> dict[str, object]:
    return {
        "kind": payload.get("kind"),
        "status": payload.get("status"),
        "summary": payload.get("summary"),
        "slot_key": payload.get("slot_key"),
        "value_key": payload.get("value_key"),
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
        "existing_memory_ids": ["fact:seed"],
    }


def _content_builder(payload: dict[str, object]) -> str:
    return f"{payload.get('summary', '')} {payload.get('status', '')}"


def _capture_remote_catalog_behavior(store_cls: type[object]) -> dict[str, object]:
    roundtrip_state = _FakeRemoteState()
    roundtrip_store = store_cls(roundtrip_state)
    payloads = (
        _make_object_payload(memory_id="fact:jam", summary="Corinna liebt Aprikosenmarmelade"),
        _make_object_payload(memory_id="fact:tea", summary="Die Thermoskanne steht im Regal"),
    )
    catalog = roundtrip_store.build_catalog_payload(
        snapshot_kind="objects",
        item_payloads=payloads,
        item_id_getter=_item_id_getter,
        metadata_builder=_metadata_builder,
        content_builder=_content_builder,
    )
    roundtrip_state.save_snapshot(snapshot_kind="objects", payload=catalog)
    entries = roundtrip_store.load_catalog_entries(snapshot_kind="objects")
    assembled = roundtrip_store.assemble_snapshot_from_catalog(snapshot_kind="objects", payload=catalog)
    compat = roundtrip_store.assemble_snapshot_from_catalog_result(snapshot_kind="objects", payload=catalog)

    search_client = _FakeChonkyClient(supports_topk_records=True)
    search_state = _FakeRemoteState(client=search_client)
    search_store = store_cls(search_state)
    search_catalog = search_store.build_catalog_payload(
        snapshot_kind="objects",
        item_payloads=payloads,
        item_id_getter=_item_id_getter,
        metadata_builder=_metadata_builder,
        content_builder=_content_builder,
    )
    search_state.save_snapshot(snapshot_kind="objects", payload=search_catalog)
    search_entries = search_store.search_catalog_entries(snapshot_kind="objects", query_text="Thermoskanne", limit=1)
    scope_payloads = search_store.search_current_item_payloads(
        snapshot_kind="objects",
        query_text="Thermoskanne",
        limit=1,
    )

    metadata_client = _MetadataOnlyRetrieveClient()
    metadata_state = _FakeRemoteState(client=metadata_client)
    metadata_store = store_cls(metadata_state)
    metadata_catalog = metadata_store.build_catalog_payload(
        snapshot_kind="objects",
        item_payloads=payloads[:1],
        item_id_getter=_item_id_getter,
        metadata_builder=_metadata_builder,
        content_builder=_content_builder,
    )
    metadata_state.save_snapshot(snapshot_kind="objects", payload=metadata_catalog)
    metadata_only_payloads = metadata_store.load_item_payloads(
        snapshot_kind="objects",
        item_ids=("fact:jam",),
    )

    compat_objects = tuple(compat.payload.get("objects", ())) if isinstance(compat.payload, dict) else ()
    scope_details = search_client.topk_records_payloads[0] if search_client.topk_records_payloads else {}
    metadata_attributes = {}
    if metadata_only_payloads:
        metadata_attributes = dict(metadata_only_payloads[0].get("attributes") or {})
    return {
        "catalog_schema": catalog.get("schema"),
        "catalog_items_count": catalog.get("items_count"),
        "segment_entry_counts": tuple(
            int(segment.get("entry_count") or 0)
            for segment in catalog.get("segments", ())
            if isinstance(segment, dict)
        ),
        "entry_ids": tuple(entry.item_id for entry in entries),
        "entry_doc_ids": tuple(entry.document_id for entry in entries),
        "assembled_memory_ids": tuple(
            str(item.get("memory_id") or "")
            for item in assembled.get("objects", ())
            if isinstance(item, dict)
        )
        if isinstance(assembled, dict)
        else (),
        "compat_complete": compat.direct_catalog_complete,
        "compat_source_types": tuple(
            str(item.get("source", {}).get("type") or "")
            for item in compat_objects
            if isinstance(item, dict)
        ),
        "compat_remote_flags": tuple(
            bool((item.get("attributes") or {}).get("remote_catalog_entry_compatibility"))
            for item in compat_objects
            if isinstance(item, dict)
        ),
        "catalog_search_ids": tuple(entry.item_id for entry in search_entries),
        "scope_search_summaries": tuple(
            str(item.get("summary") or "")
            for item in (scope_payloads or ())
            if isinstance(item, dict)
        ),
        "scope_search_scope_ref": scope_details.get("scope_ref"),
        "scope_search_namespace": scope_details.get("namespace"),
        "scope_search_allowed_doc_ids": scope_details.get("allowed_doc_ids"),
        "metadata_only_source_types": tuple(
            str(item.get("source", {}).get("type") or "")
            for item in metadata_only_payloads
            if isinstance(item, dict)
        ),
        "metadata_only_flag": bool(metadata_attributes.get("legacy_remote_catalog_metadata_only")),
    }


class RemoteCatalogRefactorParityTests(unittest.TestCase):
    def test_public_remote_catalog_behavior_matches_golden_master(self) -> None:
        observed = _capture_remote_catalog_behavior(LongTermRemoteCatalogStore)
        self.assertEqual(
            observed,
            {
                "catalog_schema": "twinr_memory_object_catalog_v3",
                "catalog_items_count": 2,
                "segment_entry_counts": (2,),
                "entry_ids": ("fact:jam", "fact:tea"),
                "entry_doc_ids": ("doc-1", "doc-2"),
                "assembled_memory_ids": ("fact:jam", "fact:tea"),
                "compat_complete": True,
                "compat_source_types": ("remote_catalog_entry", "remote_catalog_entry"),
                "compat_remote_flags": (True, True),
                "catalog_search_ids": ("fact:tea",),
                "scope_search_summaries": ("Die Thermoskanne steht im Regal",),
                "scope_search_scope_ref": "longterm:objects:current",
                "scope_search_namespace": "test-namespace",
                "scope_search_allowed_doc_ids": None,
                "metadata_only_source_types": ("remote_catalog_entry",),
                "metadata_only_flag": False,
            },
        )

    def test_internal_base_matches_public_wrapper_when_available(self) -> None:
        if LongTermRemoteCatalogStoreBase is None:
            self.skipTest("internal remote_catalog base package not available before refactor extraction")
        self.assertEqual(
            _capture_remote_catalog_behavior(LongTermRemoteCatalogStore),
            _capture_remote_catalog_behavior(LongTermRemoteCatalogStoreBase),
        )


if __name__ == "__main__":
    unittest.main()
