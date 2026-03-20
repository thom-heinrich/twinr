from __future__ import annotations

import json
import pytest
from fastapi.testclient import TestClient

try:
    from agent_framework.chonkDB.data_management.api.app import create_app
    from agent_framework.chonkDB.data_management.api.routers import (
        admin as admin_router,
        external as external_router,
    )
    from agent_framework.chonkDB.data_management.api.routers import (
        indexes as indexes_router,
    )
except ImportError as exc:  # pragma: no cover - environment-gated import
    pytest.skip(
        f"external facade tests require full chonkydb runtime: {exc}",
        allow_module_level=True,
    )


def _disable_startup_warmups(monkeypatch) -> None:
    monkeypatch.setenv("CHONKY_API_WARMUP_WAIT_FOR_READY", "0")
    monkeypatch.setenv("CHONKY_API_WARMUP_HASH_INDEXES", "0")
    monkeypatch.setenv("CHONKY_API_WARMUP_EMBEDDINGS", "0")
    monkeypatch.setenv("CHONKY_API_WARMUP_CORE_INDEXES", "0")
    monkeypatch.setenv("CHONKY_API_WARMUP_WRITE_PATH", "0")
    monkeypatch.setenv("CHONKY_API_WARMUP_ADMIN_STATS", "0")
    monkeypatch.setenv("CHONKY_API_WARMUP_DOCID_MAPPING", "0")
    monkeypatch.delenv("CHONKDB_API_KEY", raising=False)
    monkeypatch.delenv("CHONKDB_API_KEY_HEADER", raising=False)
    monkeypatch.delenv("CHONKDB_API_KEY_ALLOW_BEARER", raising=False)
    monkeypatch.delenv("CHONKDB_API_KEY_EXEMPT_PATHS", raising=False)
    monkeypatch.setattr(
        external_router.health_router,
        "_warmup_pending_payload",
        lambda _request, readiness_scope="full": {
            "pending": False,
            "pending_count": 0,
            "pending_tasks": [],
            "blocking_pending": False,
            "blocking_count": 0,
            "blocking_tasks": [],
            "readiness_scope": str(readiness_scope),
            "tracked_count": 0,
            "fulltext_pending_indexes": [],
            "fulltext_states": {},
            "vector_pending_indexes": [],
            "vector_states": {},
            "graph_pending_indexes": [],
            "graph_states": {},
        },
    )


class _FacadeFakeService:
    def __init__(self) -> None:
        self.calls: dict[str, list[dict[str, object]]] = {
            "store_payload": [],
            "start_ingest_job": [],
            "sync_bulk": [],
            "async_bulk": [],
            "query": [],
            "advanced": [],
            "graph": [],
            "add_edge": [],
            "add_edge_smart": [],
            "neighbors": [],
            "path": [],
            "patterns": [],
            "info": [],
            "full_document": [],
            "list": [],
            "delete": [],
            "job_status": [],
            "cancel": [],
            "warm_start": [],
            "optimize": [],
            "reload": [],
            "flush": [],
            "clear_all": [],
            "health": [],
        }
        self._payload_blob_by_payload_id = {
            "payload-blob-1": {
                "schema": "twinr_memory_object",
                "version": 1,
                "memory_id": "fact:blob",
                "summary": "Blob payload hydrated.",
            }
        }
        self._full_documents_by_origin_uri = {
            "twinr://longterm/test-namespace/__pointer__%3Aobjects": {
                "success": True,
                "document_id": "pointer-objects-current",
                "origin_uri": "twinr://longterm/test-namespace/__pointer__%3Aobjects",
                "content": json.dumps(
                    {
                        "schema": "twinr_remote_snapshot_v1",
                        "namespace": "test-namespace",
                        "snapshot_kind": "__pointer__:objects",
                        "updated_at": "2026-03-20T00:00:01Z",
                        "body": {
                            "schema": "twinr_remote_snapshot_pointer_v1",
                            "version": 1,
                            "snapshot_kind": "objects",
                            "document_id": "snapshot-objects-current",
                        },
                    },
                    ensure_ascii=False,
                ),
            },
            "twinr://longterm/test-namespace/objects": {
                "success": True,
                "document_id": "snapshot-objects-current",
                "origin_uri": "twinr://longterm/test-namespace/objects",
                "chunk_count": 2,
                "chunks": [
                    {
                        "payload_id": "snapshot-objects-old",
                        "document_id": "snapshot-objects-old",
                        "metadata": {
                            "twinr_namespace": "test-namespace",
                            "twinr_snapshot_kind": "objects",
                            "twinr_snapshot_updated_at": "2026-03-19T00:00:00Z",
                            "origin_uri": "twinr://longterm/test-namespace/objects",
                        },
                        "content": json.dumps(
                            {
                                "schema": "twinr_remote_snapshot_v1",
                                "namespace": "test-namespace",
                                "snapshot_kind": "objects",
                                "updated_at": "2026-03-19T00:00:00Z",
                                "body": {
                                    "schema": "twinr_memory_object_store_manifest",
                                    "version": 1,
                                    "shards": ["objects__part_0000"],
                                },
                            },
                            ensure_ascii=False,
                        ),
                    },
                    {
                        "payload_id": "snapshot-objects-current",
                        "document_id": "snapshot-objects-current",
                        "metadata": {
                            "twinr_namespace": "test-namespace",
                            "twinr_snapshot_kind": "objects",
                            "twinr_snapshot_updated_at": "2026-03-20T00:00:00Z",
                            "origin_uri": "twinr://longterm/test-namespace/objects",
                        },
                        "content": json.dumps(
                            {
                                "schema": "twinr_remote_snapshot_v1",
                                "namespace": "test-namespace",
                                "snapshot_kind": "objects",
                                "updated_at": "2026-03-20T00:00:00Z",
                                "body": {
                                    "schema": "twinr_memory_object_catalog_v3",
                                    "version": 3,
                                    "segments": [
                                        {
                                            "segment_index": 0,
                                            "document_id": "segment-objects-0",
                                            "uri": "twinr://longterm/test-namespace/objects/catalog/segment/0000",
                                            "entry_count": 1,
                                        }
                                    ],
                                },
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
            },
            "twinr://longterm/test-namespace/objects/catalog/segment/0000": {
                "success": True,
                "document_id": "segment-objects-0",
                "content": json.dumps(
                    {
                        "schema": "twinr_memory_object_catalog_segment_v1",
                        "version": 1,
                        "snapshot_kind": "objects",
                        "segment_index": 0,
                        "items": [
                            {
                                "item_id": "fact:ada",
                                "document_id": "doc-advanced-1",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
            },
        }
        self._full_documents_by_document_id = {
            "snapshot-objects-current": {
                "success": True,
                "document_id": "snapshot-objects-current",
                "content": json.dumps(
                    {
                        "schema": "twinr_remote_snapshot_v1",
                        "namespace": "test-namespace",
                        "snapshot_kind": "objects",
                        "updated_at": "2026-03-20T00:00:00Z",
                        "body": {
                            "schema": "twinr_memory_object_catalog_v3",
                            "version": 3,
                            "segments": [
                                {
                                    "segment_index": 0,
                                    "document_id": "segment-objects-0",
                                    "uri": "twinr://longterm/test-namespace/objects/catalog/segment/0000",
                                    "entry_count": 1,
                                }
                            ],
                        },
                    },
                    ensure_ascii=False,
                ),
            },
            "segment-objects-0": dict(
                self._full_documents_by_origin_uri[
                    "twinr://longterm/test-namespace/objects/catalog/segment/0000"
                ]
            )
        }

    async def store_payload(self, **kwargs):  # noqa: ANN003
        self.calls["store_payload"].append(dict(kwargs))
        return {"success": True, "payload_id": "payload-sync-1"}

    async def start_ingest_job(self, **kwargs):  # noqa: ANN003
        self.calls["start_ingest_job"].append(dict(kwargs))
        return {"success": True, "job_id": "job-single-1", "status": "pending"}

    async def store_payloads_sync_bulk(self, **kwargs):  # noqa: ANN003
        self.calls["sync_bulk"].append(dict(kwargs))
        items = list(kwargs.get("items") or [])
        return {
            "success": True,
            "all_succeeded": True,
            "count": len(items),
            "succeeded": len(items),
            "failed": 0,
            "items": [
                {
                    "success": True,
                    "payload_id": f"payload-{idx}",
                    "document_id": f"payload-{idx}",
                    "execution_time_ms": 12.5,
                }
                for idx, _ in enumerate(items)
            ],
        }

    async def start_ingest_job_bulk(
        self, payloads, timeout_seconds=None, finalize_vector_segments=True
    ):  # noqa: ANN001
        self.calls["async_bulk"].append(
            {
                "payloads": list(payloads or []),
                "timeout_seconds": timeout_seconds,
                "finalize_vector_segments": finalize_vector_segments,
            }
        )
        return {
            "success": True,
            "job_id": "job-bulk-1",
            "status": "pending",
            "items": len(list(payloads or [])),
        }

    async def ingest_document(self, **kwargs):  # noqa: ANN003
        return {
            "success": True,
            "chonky_id": "doc-42",
            "indexed": [["fulltext", "doc-42"]],
        }

    async def query_payloads(self, **kwargs):  # noqa: ANN003
        self.calls["query"].append(dict(kwargs))
        return {
            "success": True,
            "results": [{"payload_id": "payload-query-1"}],
            "indexes_used": ["fulltext"],
        }

    async def query_payloads_advanced(self, **kwargs):  # noqa: ANN003
        self.calls["advanced"].append(dict(kwargs))
        if str(kwargs.get("query_text") or "").strip() == "blob-fallback":
            return {
                "success": True,
                "results": [
                    {
                        "payload_id": "payload-blob-1",
                        "document_id": "payload-blob-1",
                        "relevance_score": 0.77,
                        "source_index": "fulltext",
                        "candidate_origin": "fulltext",
                        "metadata": {
                            "twinr_memory_item_id": "fact:blob",
                            "summary": "Blob payload hydrated.",
                        },
                    }
                ],
                "indexes_used": ["fulltext"],
                "debug": {"latency_breakdown_ms": {"search": 9.0}},
            }
        return {
            "success": True,
            "results": [
                {
                    "payload_id": "payload-advanced-1",
                    "document_id": "doc-advanced-1",
                    "relevance_score": 0.91,
                    "source_index": "hnsw",
                    "candidate_origin": "vector",
                    "metadata": {
                        "twinr_memory_item_id": "fact:ada",
                        "twinr_payload": {
                            "schema": "twinr_memory_object",
                            "version": 1,
                            "memory_id": "fact:ada",
                            "summary": "Ada likes tea.",
                        },
                    },
                    "score_breakdown": {"vector": 0.91},
                }
            ],
            "indexes_used": ["fulltext", "hnsw"],
            "debug": {"latency_breakdown_ms": {"search": 12.4, "materialize": 1.2}},
        }

    async def graph_semantic_filter(self, **kwargs):  # noqa: ANN003
        self.calls["graph"].append(dict(kwargs))
        return {
            "success": True,
            "results": [{"payload_id": "payload-graph-1"}],
            "indexes_used": ["graph_index"],
        }

    async def add_graph_edge(self, **kwargs):  # noqa: ANN003
        self.calls["add_edge"].append(dict(kwargs))
        return {"success": True, "edge_id": "edge-1"}

    async def add_graph_edge_smart(self, **kwargs):  # noqa: ANN003
        self.calls["add_edge_smart"].append(dict(kwargs))
        return {"success": True, "edge_id": "edge-smart-1"}

    async def graph_neighbors(self, **kwargs):  # noqa: ANN003
        self.calls["neighbors"].append(dict(kwargs))
        return {
            "success": True,
            "neighbors": [{"label": "Analytical Engine"}],
            "count": 1,
        }

    async def graph_path(self, **kwargs):  # noqa: ANN003
        self.calls["path"].append(dict(kwargs))
        return {
            "success": True,
            "path": [
                str(kwargs.get("source_label") or ""),
                str(kwargs.get("target_label") or ""),
            ],
        }

    async def query_graph_patterns(self, **kwargs):  # noqa: ANN003
        self.calls["patterns"].append(dict(kwargs))
        return {
            "success": True,
            "patterns": [{"nodes": ["Ada", "Analytical Engine"]}],
            "total_results": 1,
        }

    async def get_payload_info(self, **kwargs):  # noqa: ANN003
        self.calls["info"].append(dict(kwargs))
        record_id = (
            kwargs.get("payload_id") or kwargs.get("chonky_id") or "payload-sync-1"
        )
        return {
            "success": True,
            "payload_id": str(record_id),
            "chonky_id": "42",
            "metadata": {"origin_uri": "memory://payload-sync-1"},
        }

    async def get_full_document(self, **kwargs):  # noqa: ANN003
        self.calls["full_document"].append(dict(kwargs))
        document_id = str(kwargs.get("document_id") or "").strip()
        origin_uri = str(kwargs.get("origin_uri") or "").strip()
        if document_id and document_id in self._full_documents_by_document_id:
            return dict(self._full_documents_by_document_id[document_id])
        if origin_uri and origin_uri in self._full_documents_by_origin_uri:
            return dict(self._full_documents_by_origin_uri[origin_uri])
        return {
            "success": True,
            "document_id": str(kwargs.get("document_id") or "42"),
            "chunks": [{"content": "hello world"}],
        }

    async def list_all_payloads(self, **kwargs):  # noqa: ANN003
        self.calls["list"].append(dict(kwargs))
        return {
            "success": True,
            "items": [{"payload_id": "payload-sync-1"}],
            "count": 1,
        }

    async def delete_payload(self, **kwargs):  # noqa: ANN003
        self.calls["delete"].append(dict(kwargs))
        return {"success": True, "deleted": True}

    async def job_status(self, job_id):  # noqa: ANN001
        self.calls["job_status"].append({"job_id": job_id})
        return {"success": True, "job_id": str(job_id), "status": "done"}

    async def cancel_job(self, job_id):  # noqa: ANN001
        self.calls["cancel"].append({"job_id": job_id})
        return {"success": True, "job_id": str(job_id), "status": "cancelled"}

    async def list_available_indexes(self):
        return {
            "success": True,
            "indexes": ["fulltext_index", "graph_index", "temporal_index"],
        }

    async def get_system_stats(self):
        return {
            "success": True,
            "backend": "fake",
            "data_dir": "/tmp/chonkydb-fake",
            "disk": {"total_bytes": 1000, "used_bytes": 400, "free_bytes": 600},
            "index_files": {
                "fulltext_index.chonk": {"bytes": 111},
                "graph_index.chonk": {"bytes": 222},
            },
        }

    async def get_basic_metrics(self):
        return {"success": True, "embedding_cache": {"hits": 1, "misses": 0}}

    async def warm_start_models(self, **kwargs):  # noqa: ANN003
        self.calls["warm_start"].append(dict(kwargs))
        return {"success": True, "status": "ok", "warmed": True}

    async def optimize_indexes(self, **kwargs):  # noqa: ANN003
        self.calls["optimize"].append(dict(kwargs))
        return {"success": True, "status": "ok", "optimized": True}

    async def reload_from_disk(self):
        self.calls["reload"].append({})
        return {"success": True, "status": "reloaded"}

    async def flush_all(self, **kwargs):  # noqa: ANN003
        self.calls["flush"].append(dict(kwargs))
        return {"success": True, "status": "flushed"}

    async def clear_all(self, **kwargs):  # noqa: ANN003
        self.calls["clear_all"].append(dict(kwargs))
        return {"success": True, "status": "cleared"}

    async def health_check(self):
        self.calls["health"].append({})
        return {"success": True, "ok": True, "status": "ok", "backend": "fake"}

    def _ensure_api_server_main_api_for_request(self, **kwargs):  # noqa: ANN003
        return object()

    def _ensure_engine_docid_router_ready(self, _api, require_router=False):  # noqa: ANN001, ANN003
        service = self

        class _FakeDocIdManager:
            def get_doc_id_for_uuid(self, value):  # noqa: ANN001
                if str(value or "").strip() == "payload-blob-1":
                    return 42
                return None

            def read_document_data(
                self,
                doc_id,
                subindex_key,
                raise_on_not_found=False,
                allow_header_reload_on_miss=False,
            ):  # noqa: ANN001, ANN003
                del raise_on_not_found
                del allow_header_reload_on_miss
                if int(doc_id) == 42 and str(subindex_key) == "payload":
                    return dict(service._payload_blob_by_payload_id["payload-blob-1"])
                return None

        return object(), _FakeDocIdManager(), object()

    def _resolve_doc_int_for_payload_ref(self, dm, *, payload_id, chonky_id):  # noqa: ANN001
        return dm.get_doc_id_for_uuid(payload_id or chonky_id)

    def _decode_payload_object(self, raw):  # noqa: ANN001
        return raw


def test_external_facade_dispatch_and_openapi(monkeypatch) -> None:
    _disable_startup_warmups(monkeypatch)
    fake = _FacadeFakeService()
    monkeypatch.setattr(indexes_router, "svc", fake)
    monkeypatch.setattr(external_router, "_get_svc", lambda _request=None: fake)

    app = create_app()
    with TestClient(app) as client:
        instance_response = client.get("/v1/external/instance")
        assert instance_response.status_code == 200, instance_response.text
        assert instance_response.json()["service"] == "ccodex_memory"

        sync_response = client.post(
            "/v1/external/records",
            json={
                "content": "memory body",
                "metadata": {"source": "external"},
                "timeout_seconds": 41,
                "client_request_id": "single-req-1",
            },
        )
        assert sync_response.status_code == 200, sync_response.text
        assert sync_response.json()["payload_id"] == "payload-0"
        assert sync_response.json()["document_id"] == "payload-0"
        assert sync_response.json()["client_request_id"] == "single-req-1"

        async_response = client.post(
            "/v1/external/records",
            json={
                "content": "memory body",
                "execution_mode": "async",
            },
        )
        assert async_response.status_code == 202, async_response.text
        assert async_response.json()["job_id"] == "job-single-1"

        bulk_sync_response = client.post(
            "/v1/external/records/bulk",
            json={
                "client_request_id": "bulk-req-1",
                "items": [
                    {"content": "item-1"},
                    {"content": "item-2"},
                ],
            },
        )
        assert bulk_sync_response.status_code == 200, bulk_sync_response.text
        assert bulk_sync_response.json()["count"] == 2

        bulk_async_response = client.post(
            "/v1/external/records/bulk",
            json={
                "execution_mode": "async",
                "timeout_seconds": 42,
                "finalize_vector_segments": False,
                "items": [{"content": "item-a"}],
            },
        )
        assert bulk_async_response.status_code == 202, bulk_async_response.text
        assert bulk_async_response.json()["job_id"] == "job-bulk-1"

        basic_query_response = client.post(
            "/v1/external/retrieve",
            json={"mode": "basic", "query_text": "hello world", "result_limit": 3},
        )
        assert basic_query_response.status_code == 200, basic_query_response.text
        assert basic_query_response.json()["mode"] == "basic"

        advanced_query_response = client.post(
            "/v1/external/retrieve",
            json={
                "mode": "advanced",
                "query_text": "hello world",
                "allowed_indexes": ["fulltext", "hnsw"],
                "graph_index_name": "graph_index",
                "seed_label": "Ada",
            },
        )
        assert advanced_query_response.status_code == 200, advanced_query_response.text
        assert advanced_query_response.json()["mode"] == "advanced"

        topk_query_response = client.post(
            "/v1/external/retrieve/topk_records",
            json={
                "query_text": "Ada",
                "namespace": "test-namespace",
                "scope_ref": "longterm:objects:current",
            },
        )
        assert topk_query_response.status_code == 200, topk_query_response.text
        assert topk_query_response.json()["mode"] == "advanced"
        assert topk_query_response.json()["results"][0]["payload"]["memory_id"] == "fact:ada"
        assert topk_query_response.json()["results"][0]["payload_source"] == "metadata.twinr_payload"
        assert topk_query_response.json()["query_plan"]["latency_ms"]["search"] == 12.4

        graph_query_response = client.post(
            "/v1/external/retrieve",
            json={
                "mode": "graph",
                "seed_label": "Ada Lovelace",
                "graph_index_name": "graph_index",
                "graph_edge_types": ["describes", "mentions"],
                "graph_direction": "outbound",
                "graph_weight": 0.35,
                "result_limit": 7,
            },
        )
        assert graph_query_response.status_code == 200, graph_query_response.text
        assert graph_query_response.json()["mode"] == "graph"

        graph_edge_response = client.post(
            "/v1/external/graph/edges",
            json={"from_id": 1, "to_id": 2, "edge_type": "describes"},
        )
        assert graph_edge_response.status_code == 200, graph_edge_response.text
        assert graph_edge_response.json()["edge_id"] == "edge-1"

        graph_edge_smart_response = client.post(
            "/v1/external/graph/edges/smart",
            json={
                "from_ref": "Ada Lovelace",
                "to_ref": "Analytical Engine",
                "edge_type": "mentions",
            },
        )
        assert (
            graph_edge_smart_response.status_code == 200
        ), graph_edge_smart_response.text
        assert graph_edge_smart_response.json()["edge_id"] == "edge-smart-1"

        graph_neighbors_response = client.post(
            "/v1/external/graph/neighbors",
            json={
                "label_or_id": "Ada Lovelace",
                "direction": "outbound",
                "with_edges": True,
                "return_ids": True,
                "edge_types": ["describes"],
                "limit": 5,
                "timeout_seconds": 2,
            },
        )
        assert (
            graph_neighbors_response.status_code == 200
        ), graph_neighbors_response.text
        assert graph_neighbors_response.json()["count"] == 1

        graph_path_response = client.post(
            "/v1/external/graph/path",
            json={
                "source_label": "Ada Lovelace",
                "target_label": "Analytical Engine",
                "edge_types": ["mentions"],
                "return_ids": True,
            },
        )
        assert graph_path_response.status_code == 200, graph_path_response.text
        assert graph_path_response.json()["path"] == [
            "Ada Lovelace",
            "Analytical Engine",
        ]

        graph_patterns_response = client.post(
            "/v1/external/graph/patterns",
            json={
                "patterns": [
                    {
                        "start_label": "Ada Lovelace",
                        "end_label": "Analytical Engine",
                        "edge_types": ["mentions"],
                    }
                ],
                "limit": 3,
                "max_depth": 4,
                "include_content": False,
            },
        )
        assert graph_patterns_response.status_code == 200, graph_patterns_response.text
        assert graph_patterns_response.json()["total_results"] == 1

        list_response = client.get(
            "/v1/external/records?offset=5&limit=25&include_metadata=false"
        )
        assert list_response.status_code == 200, list_response.text

        record_response = client.get(
            "/v1/external/records/payload-sync-1?include_document=true"
        )
        assert record_response.status_code == 200, record_response.text
        assert record_response.json()["document"]["document_id"] == "42"

        delete_response = client.delete("/v1/external/records/payload-sync-1")
        assert delete_response.status_code == 200, delete_response.text

        job_response = client.get("/v1/external/jobs/job-bulk-1")
        assert job_response.status_code == 200, job_response.text
        cancel_response = client.post("/v1/external/jobs/job-bulk-1/cancel")
        assert cancel_response.status_code == 200, cancel_response.text

        admin_auth_response = client.get("/v1/external/admin/auth")
        assert admin_auth_response.status_code == 200, admin_auth_response.text
        assert admin_auth_response.json()["api_key_configured"] is False

        admin_docs_response = client.get("/v1/external/admin/docs")
        assert admin_docs_response.status_code == 200, admin_docs_response.text
        assert (
            admin_docs_response.json()["guide_endpoint"]
            == "/v1/external/admin/docs/guide"
        )
        assert admin_docs_response.json()["swagger_ui_url"] == "http://testserver/docs"

        admin_docs_guide_response = client.get("/v1/external/admin/docs/guide")
        assert (
            admin_docs_guide_response.status_code == 200
        ), admin_docs_guide_response.text
        assert admin_docs_guide_response.text
        assert "API" in admin_docs_guide_response.text

        admin_docs_openapi_response = client.get("/v1/external/admin/docs/openapi.json")
        assert (
            admin_docs_openapi_response.status_code == 200
        ), admin_docs_openapi_response.text

        admin_health_response = client.get("/v1/external/admin/health")
        assert admin_health_response.status_code == 200, admin_health_response.text
        assert admin_health_response.json()["ok"] is True
        assert admin_health_response.json()["warmup"]["readiness_scope"] == "full"

        admin_ready_response = client.get("/v1/external/admin/ready")
        assert admin_ready_response.status_code == 200, admin_ready_response.text
        assert admin_ready_response.json()["status"] == "ok"
        assert admin_ready_response.json()["warmup"]["readiness_scope"] == "full"

        admin_stats_response = client.get("/v1/external/admin/stats")
        assert admin_stats_response.status_code == 200, admin_stats_response.text
        assert admin_stats_response.json()["backend"] == "fake"

        admin_metrics_response = client.get("/v1/external/admin/metrics/basic")
        assert admin_metrics_response.status_code == 200, admin_metrics_response.text
        assert admin_metrics_response.json()["embedding_cache"]["hits"] == 1

        admin_indexes_response = client.get("/v1/external/admin/indexes")
        assert admin_indexes_response.status_code == 200, admin_indexes_response.text
        assert "graph_index" in admin_indexes_response.json()["indexes"]

        admin_storage_response = client.get("/v1/external/admin/storage")
        assert admin_storage_response.status_code == 200, admin_storage_response.text
        assert admin_storage_response.json()["tracked_index_bytes"] == 333

        admin_provision_response = client.post(
            "/v1/external/admin/provision",
            json={
                "warm_start": True,
                "components": ["embeddings"],
                "load_embeddings": True,
                "flush_after": True,
                "strongest_flush": False,
                "timeout_seconds": 17,
            },
        )
        assert (
            admin_provision_response.status_code == 200
        ), admin_provision_response.text
        assert admin_provision_response.json()["provisioned"] is True
        assert admin_provision_response.json()["warm_start"]["warmed"] is True
        assert admin_provision_response.json()["flush"]["status"] == "flushed"

        admin_warm_start_response = client.post(
            "/v1/external/admin/warm-start",
            json={"components": ["graph"], "load_information_extraction": True},
        )
        assert (
            admin_warm_start_response.status_code == 200
        ), admin_warm_start_response.text
        assert admin_warm_start_response.json()["warmed"] is True

        admin_optimize_response = client.post(
            "/v1/external/admin/optimize",
            json={"reindex": True, "compact": True},
        )
        assert admin_optimize_response.status_code == 200, admin_optimize_response.text
        assert admin_optimize_response.json()["optimized"] is True

        admin_reload_response = client.post("/v1/external/admin/reload")
        assert admin_reload_response.status_code == 200, admin_reload_response.text
        assert admin_reload_response.json()["status"] == "reloaded"

        admin_flush_response = client.post(
            "/v1/external/admin/flush?strongest=false&timeout_seconds=9"
        )
        assert admin_flush_response.status_code == 200, admin_flush_response.text
        assert admin_flush_response.json()["status"] == "flushed"

        admin_clear_response = client.delete("/v1/external/admin/data?confirm=true")
        assert admin_clear_response.status_code == 200, admin_clear_response.text
        assert admin_clear_response.json()["status"] == "cleared"

        openapi_response = client.get("/openapi.json")
        assert openapi_response.status_code == 200, openapi_response.text
        paths = openapi_response.json()["paths"]
        assert "/v1/external/records" in paths
        assert "/v1/external/retrieve" in paths
        assert "/v1/external/retrieve/topk_records" in paths
        assert "/v1/external/graph/edges" in paths
        assert "/v1/external/graph/edges/smart" in paths
        assert "/v1/external/graph/neighbors" in paths
        assert "/v1/external/graph/path" in paths
        assert "/v1/external/graph/patterns" in paths
        assert "/v1/external/jobs/{job_id}" in paths
        assert "/v1/external/admin/docs" in paths
        assert "/v1/external/admin/docs/guide" in paths
        assert "/v1/external/admin/docs/openapi.json" in paths
        assert "/v1/external/admin/health" in paths
        assert "/v1/external/admin/provision" in paths
        assert "/v1/external/admin/data" in paths
        security_schemes = openapi_response.json()["components"]["securitySchemes"]
        assert security_schemes["APIKeyHeaderAuth"]["type"] == "apiKey"
        assert security_schemes["APIKeyHeaderAuth"]["name"] == "x-api-key"
        assert security_schemes["BearerAuth"]["scheme"] == "bearer"

    assert fake.calls["store_payload"] == []
    assert fake.calls["start_ingest_job"][0]["content"] == "memory body"
    assert fake.calls["sync_bulk"][0]["client_request_id"] == "single-req-1"
    assert fake.calls["sync_bulk"][0]["timeout_seconds"] == 41
    assert "timeout_seconds" not in fake.calls["sync_bulk"][0]["items"][0]
    assert fake.calls["sync_bulk"][0]["items"][0]["content"] == "memory body"
    assert fake.calls["sync_bulk"][1]["client_request_id"] == "bulk-req-1"
    assert fake.calls["async_bulk"][0]["timeout_seconds"] == 42
    assert fake.calls["async_bulk"][0]["finalize_vector_segments"] is False
    assert fake.calls["query"][0]["query_text"] == "hello world"
    assert fake.calls["advanced"][0]["graph_seed_label"] == "Ada"
    assert fake.calls["graph"][0]["seed_label"] == "Ada Lovelace"
    assert fake.calls["graph"][0]["index_name"] == "graph_index"
    assert fake.calls["graph"][0]["graph_edge_types"] == ["describes", "mentions"]
    assert fake.calls["graph"][0]["graph_direction"] == "outbound"
    assert fake.calls["graph"][0]["graph_weight"] == 0.35
    assert fake.calls["add_edge"][0]["edge_type"] == "describes"
    assert fake.calls["add_edge_smart"][0]["edge_type"] == "mentions"
    assert fake.calls["neighbors"][0]["label"] == "Ada Lovelace"
    assert fake.calls["neighbors"][0]["edge_types"] == ["describes"]
    assert fake.calls["path"][0]["edge_types"] == ["mentions"]
    assert fake.calls["patterns"][0]["patterns"][0]["start_label"] == "Ada Lovelace"
    assert fake.calls["list"][0]["offset"] == 5
    assert fake.calls["list"][0]["limit"] == 25
    assert fake.calls["list"][0]["include_metadata"] is False
    assert fake.calls["info"][0]["payload_id"] == "payload-sync-1"
    assert fake.calls["full_document"][0]["document_id"] == "42"
    assert fake.calls["delete"][0]["payload_id"] == "payload-sync-1"
    assert fake.calls["job_status"][0]["job_id"] == "job-bulk-1"
    assert fake.calls["cancel"][0]["job_id"] == "job-bulk-1"
    assert fake.calls["health"]
    assert fake.calls["warm_start"][0]["components"] == ["embeddings"]
    assert fake.calls["warm_start"][1]["components"] == ["graph"]
    assert fake.calls["optimize"][0]["reindex"] is True
    assert fake.calls["reload"]
    assert fake.calls["flush"][0]["strongest"] is False
    assert fake.calls["flush"][0]["timeout_seconds"] == 17
    assert fake.calls["flush"][1]["strongest"] is False
    assert fake.calls["clear_all"][0]["confirm"] is True


def test_external_topk_records_route_materializes_structured_payload(monkeypatch) -> None:
    _disable_startup_warmups(monkeypatch)
    fake = _FacadeFakeService()
    monkeypatch.setattr(indexes_router, "svc", fake)
    monkeypatch.setattr(external_router, "_get_svc", lambda _request=None: fake)

    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/v1/external/retrieve/topk_records",
            json={
                "query_text": "Ada",
                "namespace": "test-namespace",
                "scope_ref": "longterm:objects:current",
            },
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["mode"] == "advanced"
    assert payload["scope_ref"] == "longterm:objects:current"
    assert payload["results"][0]["payload"]["memory_id"] == "fact:ada"
    assert payload["results"][0]["payload_source"] == "metadata.twinr_payload"
    assert payload["query_plan"]["latency_ms"]["search"] == 12.4
    assert payload["query_plan"]["latency_ms"]["scope_resolve"] >= 0.0
    assert payload["query_plan"]["scope_cache_hit"] is False
    assert fake.calls["advanced"][0]["allowed_doc_ids"] == ["doc-advanced-1"]
    assert "scope_ref" not in fake.calls["advanced"][0]
    assert "namespace" not in fake.calls["advanced"][0]
    assert fake.calls["full_document"][0]["origin_uri"] == "twinr://longterm/test-namespace/__pointer__%3Aobjects"
    assert fake.calls["full_document"][1]["document_id"] == "snapshot-objects-current"
    assert fake.calls["full_document"][2]["document_id"] == "segment-objects-0"


def test_external_topk_records_scope_ref_keeps_warm_cache_for_unrelated_nested_and_non_scope_writes(
    monkeypatch,
) -> None:
    _disable_startup_warmups(monkeypatch)
    fake = _FacadeFakeService()
    monkeypatch.setattr(indexes_router, "svc", fake)
    monkeypatch.setattr(external_router, "_get_svc", lambda _request=None: fake)
    external_router._invalidate_scope_allowed_doc_ids_cache()

    app = create_app()
    with TestClient(app) as client:
        first = client.post(
            "/v1/external/retrieve/topk_records",
            json={
                "query_text": "Ada",
                "namespace": "test-namespace",
                "scope_ref": "longterm:objects:current",
            },
        )
        assert first.status_code == 200, first.text
        assert first.json()["query_plan"]["scope_cache_hit"] is False
        assert len(fake.calls["full_document"]) == 3

        second = client.post(
            "/v1/external/retrieve/topk_records",
            json={
                "query_text": "Ada",
                "namespace": "test-namespace",
                "scope_ref": "longterm:objects:current",
            },
        )
        assert second.status_code == 200, second.text
        assert second.json()["query_plan"]["scope_cache_hit"] is True
        assert len(fake.calls["full_document"]) == 3

        unrelated_write_response = client.post(
            "/v1/external/records",
            json={
                "payload": {
                    "schema": "demo",
                    "namespace": "other-namespace",
                    "value": "invalidate",
                },
                "metadata": {"kind": "demo"},
                "content": "invalidate scope cache",
            },
        )
        assert unrelated_write_response.status_code == 200, unrelated_write_response.text

        third = client.post(
            "/v1/external/retrieve/topk_records",
            json={
                "query_text": "Ada",
                "namespace": "test-namespace",
                "scope_ref": "longterm:objects:current",
            },
        )
        assert third.status_code == 200, third.text
        assert third.json()["query_plan"]["scope_cache_hit"] is True
        assert len(fake.calls["full_document"]) == 3

        no_namespace_unrelated_write = client.post(
            "/v1/external/records",
            json={
                "payload": {"schema": "demo", "value": "ignore"},
                "metadata": {"kind": "demo"},
                "content": "ignore unrelated write without namespace",
            },
        )
        assert no_namespace_unrelated_write.status_code == 200, no_namespace_unrelated_write.text

        fourth = client.post(
            "/v1/external/retrieve/topk_records",
            json={
                "query_text": "Ada",
                "namespace": "test-namespace",
                "scope_ref": "longterm:objects:current",
            },
        )
        assert fourth.status_code == 200, fourth.text
        assert fourth.json()["query_plan"]["scope_cache_hit"] is True
        assert len(fake.calls["full_document"]) == 3

        non_scope_same_namespace_write = client.post(
            "/v1/external/records",
            json={
                "payload": {
                    "schema": "twinr_remote_snapshot_v1",
                    "namespace": "test-namespace",
                    "snapshot_kind": "graph",
                    "updated_at": "2026-03-20T00:00:00Z",
                    "body": {"schema": "demo_graph_snapshot", "value": "keep-hot"},
                },
                "metadata": {
                    "kind": "demo",
                    "twinr_namespace": "test-namespace",
                    "twinr_snapshot_kind": "graph",
                    "twinr_snapshot_schema": "twinr_remote_snapshot_v1",
                    "origin_uri": "twinr://longterm/test-namespace/graph",
                },
                "uri": "twinr://longterm/test-namespace/graph",
                "content": "keep scope cache hot",
            },
        )
        assert non_scope_same_namespace_write.status_code == 200, non_scope_same_namespace_write.text

        fifth = client.post(
            "/v1/external/retrieve/topk_records",
            json={
                "query_text": "Ada",
                "namespace": "test-namespace",
                "scope_ref": "longterm:objects:current",
            },
        )
        assert fifth.status_code == 200, fifth.text
        assert fifth.json()["query_plan"]["scope_cache_hit"] is True
        assert len(fake.calls["full_document"]) == 3

        related_write_response = client.post(
            "/v1/external/records",
            json={
                "payload": {
                    "schema": "twinr_remote_snapshot_v1",
                    "namespace": "test-namespace",
                    "snapshot_kind": "objects",
                    "updated_at": "2026-03-20T00:00:01Z",
                    "body": {"schema": "demo_object_snapshot", "value": "invalidate"},
                },
                "metadata": {
                    "kind": "demo",
                    "twinr_namespace": "test-namespace",
                    "twinr_snapshot_kind": "objects",
                    "twinr_snapshot_schema": "twinr_remote_snapshot_v1",
                    "origin_uri": "twinr://longterm/test-namespace/objects",
                },
                "uri": "twinr://longterm/test-namespace/objects",
                "content": "invalidate scope cache",
            },
        )
        assert related_write_response.status_code == 200, related_write_response.text

        sixth = client.post(
            "/v1/external/retrieve/topk_records",
            json={
                "query_text": "Ada",
                "namespace": "test-namespace",
                "scope_ref": "longterm:objects:current",
            },
        )
        assert sixth.status_code == 200, sixth.text
        assert sixth.json()["query_plan"]["scope_cache_hit"] is True
        assert len(fake.calls["full_document"]) == 6


def test_external_topk_records_route_requires_namespace_for_scope_only_queries(monkeypatch) -> None:
    _disable_startup_warmups(monkeypatch)
    fake = _FacadeFakeService()
    monkeypatch.setattr(indexes_router, "svc", fake)
    monkeypatch.setattr(external_router, "_get_svc", lambda _request=None: fake)

    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/v1/external/retrieve/topk_records",
            json={
                "query_text": "Ada",
                "scope_ref": "longterm:objects:current",
            },
        )

    assert response.status_code == 400, response.text
    assert response.json()["error"] == "namespace required when scope_ref is used without allowed_doc_ids"


def test_external_topk_records_scope_ref_falls_back_to_latest_compatible_catalog_when_pointer_missing(monkeypatch) -> None:
    _disable_startup_warmups(monkeypatch)
    fake = _FacadeFakeService()
    fake._full_documents_by_origin_uri.pop(
        "twinr://longterm/test-namespace/__pointer__%3Aobjects",
        None,
    )
    object_history = fake._full_documents_by_origin_uri["twinr://longterm/test-namespace/objects"]
    object_history["chunks"].append(
        {
            "payload_id": "snapshot-objects-manifest-newer",
            "document_id": "snapshot-objects-manifest-newer",
            "metadata": {
                "twinr_namespace": "test-namespace",
                "twinr_snapshot_kind": "objects",
                "twinr_snapshot_updated_at": "2026-03-21T00:00:00Z",
                "origin_uri": "twinr://longterm/test-namespace/objects",
            },
            "content": json.dumps(
                {
                    "schema": "twinr_remote_snapshot_v1",
                    "namespace": "test-namespace",
                    "snapshot_kind": "objects",
                    "updated_at": "2026-03-21T00:00:00Z",
                    "body": {
                        "schema": "twinr_memory_object_store_manifest",
                        "version": 1,
                        "shards": ["objects__part_0000"],
                    },
                },
                ensure_ascii=False,
            ),
        }
    )
    monkeypatch.setattr(indexes_router, "svc", fake)
    monkeypatch.setattr(external_router, "_get_svc", lambda _request=None: fake)

    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/v1/external/retrieve/topk_records",
            json={
                "query_text": "Ada",
                "namespace": "test-namespace",
                "scope_ref": "longterm:objects:current",
            },
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["results"][0]["payload"]["memory_id"] == "fact:ada"
    assert fake.calls["advanced"][0]["allowed_doc_ids"] == ["doc-advanced-1"]
    assert fake.calls["full_document"][0]["origin_uri"] == "twinr://longterm/test-namespace/__pointer__%3Aobjects"
    assert fake.calls["full_document"][1]["origin_uri"] == "twinr://longterm/test-namespace/objects"


def test_external_topk_records_scope_ref_uses_latest_pointer_snapshot(monkeypatch) -> None:
    _disable_startup_warmups(monkeypatch)
    fake = _FacadeFakeService()
    fake._full_documents_by_origin_uri["twinr://longterm/test-namespace/__pointer__%3Aobjects"] = {
        "success": True,
        "document_id": "pointer-objects-current",
        "origin_uri": "twinr://longterm/test-namespace/__pointer__%3Aobjects",
        "chunk_count": 2,
        "chunks": [
            {
                "payload_id": "pointer-objects-old",
                "document_id": "pointer-objects-old",
                "metadata": {
                    "twinr_namespace": "test-namespace",
                    "twinr_snapshot_kind": "__pointer__:objects",
                    "twinr_snapshot_updated_at": "2026-03-19T00:00:00Z",
                    "origin_uri": "twinr://longterm/test-namespace/__pointer__%3Aobjects",
                },
                "content": json.dumps(
                    {
                        "schema": "twinr_remote_snapshot_v1",
                        "namespace": "test-namespace",
                        "snapshot_kind": "__pointer__:objects",
                        "updated_at": "2026-03-19T00:00:00Z",
                        "body": {
                            "schema": "twinr_remote_snapshot_pointer_v1",
                            "version": 1,
                            "snapshot_kind": "objects",
                            "document_id": "snapshot-objects-old",
                        },
                    },
                    ensure_ascii=False,
                ),
            },
            {
                "payload_id": "pointer-objects-current",
                "document_id": "pointer-objects-current",
                "metadata": {
                    "twinr_namespace": "test-namespace",
                    "twinr_snapshot_kind": "__pointer__:objects",
                    "twinr_snapshot_updated_at": "2026-03-20T00:00:01Z",
                    "origin_uri": "twinr://longterm/test-namespace/__pointer__%3Aobjects",
                },
                "content": json.dumps(
                    {
                        "schema": "twinr_remote_snapshot_v1",
                        "namespace": "test-namespace",
                        "snapshot_kind": "__pointer__:objects",
                        "updated_at": "2026-03-20T00:00:01Z",
                        "body": {
                            "schema": "twinr_remote_snapshot_pointer_v1",
                            "version": 1,
                            "snapshot_kind": "objects",
                            "document_id": "snapshot-objects-current",
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }
    monkeypatch.setattr(indexes_router, "svc", fake)
    monkeypatch.setattr(external_router, "_get_svc", lambda _request=None: fake)

    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/v1/external/retrieve/topk_records",
            json={
                "query_text": "Ada",
                "namespace": "test-namespace",
                "scope_ref": "longterm:objects:current",
            },
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["results"][0]["payload"]["memory_id"] == "fact:ada"
    assert fake.calls["advanced"][0]["allowed_doc_ids"] == ["doc-advanced-1"]
    assert fake.calls["full_document"][0]["origin_uri"] == "twinr://longterm/test-namespace/__pointer__%3Aobjects"
    assert fake.calls["full_document"][1]["document_id"] == "snapshot-objects-current"


def test_external_topk_records_route_hydrates_payload_blob_when_search_hit_lacks_payload(monkeypatch) -> None:
    _disable_startup_warmups(monkeypatch)
    fake = _FacadeFakeService()
    monkeypatch.setattr(indexes_router, "svc", fake)
    monkeypatch.setattr(external_router, "_get_svc", lambda _request=None: fake)

    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/v1/external/retrieve/topk_records",
            json={
                "query_text": "blob-fallback",
                "allowed_doc_ids": ["payload-blob-1"],
                "result_limit": 1,
            },
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["results"][0]["payload_source"] == "service.payload_blob"
    assert payload["results"][0]["payload"] == {
        "schema": "twinr_memory_object",
        "version": 1,
        "memory_id": "fact:blob",
        "summary": "Blob payload hydrated.",
    }


def test_single_item_bulk_result_or_http_error_raises_for_failed_item() -> None:
    with pytest.raises(external_router.HTTPException) as excinfo:
        external_router._single_item_bulk_result_or_http_error(
            {
                "success": True,
                "all_succeeded": False,
                "count": 1,
                "succeeded": 0,
                "failed": 1,
                "items": [
                    {
                        "success": False,
                        "error": "payload_sync_bulk_runtime_timeout stage=payload",
                        "error_type": "TimeoutError",
                        "item_index": 0,
                    }
                ],
            }
        )

    assert excinfo.value.status_code == 504
    assert excinfo.value.detail["error_type"] == "TimeoutError"


def test_external_auth_middleware_requires_api_key(monkeypatch) -> None:
    _disable_startup_warmups(monkeypatch)
    monkeypatch.setenv("CHONKDB_API_KEY", "top-secret")
    monkeypatch.setenv("CHONKDB_API_KEY_ALLOW_BEARER", "1")
    monkeypatch.setenv(
        "CHONKDB_API_KEY_EXEMPT_PATHS", "/v1/health,/v1/ready,/openapi.json"
    )

    fake = _FacadeFakeService()
    monkeypatch.setattr(indexes_router, "svc", fake)
    monkeypatch.setattr(external_router, "_get_svc", lambda _request=None: fake)

    app = create_app()
    with TestClient(app) as client:
        unauthorized = client.get("/v1/external/instance")
        assert unauthorized.status_code == 401, unauthorized.text
        assert unauthorized.json()["detail"] == "unauthorized"

        unauthorized_admin = client.get("/v1/external/admin/auth")
        assert unauthorized_admin.status_code == 401, unauthorized_admin.text

        unauthorized_docs = client.get("/v1/external/admin/docs")
        assert unauthorized_docs.status_code == 401, unauthorized_docs.text

        exempt_openapi = client.get("/openapi.json")
        assert exempt_openapi.status_code == 200, exempt_openapi.text
        exempt_openapi_schema = exempt_openapi.json()
        assert exempt_openapi_schema["paths"]["/v1/external/instance"]["get"][
            "security"
        ] == [{"APIKeyHeaderAuth": []}, {"BearerAuth": []}]

        authorized = client.get(
            "/v1/external/instance",
            headers={"Authorization": "Bearer top-secret"},
        )
        assert authorized.status_code == 200, authorized.text
        assert authorized.json()["auth_enabled"] is True

        authorized_admin = client.get(
            "/v1/external/admin/auth",
            headers={"Authorization": "Bearer top-secret"},
        )
        assert authorized_admin.status_code == 200, authorized_admin.text
        assert authorized_admin.json()["api_key_configured"] is True
        assert authorized_admin.json()["allow_bearer"] is True

        authorized_docs = client.get(
            "/v1/external/admin/docs",
            headers={"Authorization": "Bearer top-secret"},
        )
        assert authorized_docs.status_code == 200, authorized_docs.text
        assert (
            authorized_docs.json()["openapi_alias_endpoint"]
            == "/v1/external/admin/docs/openapi.json"
        )


def test_external_stats_and_instance_reuse_admin_stats_cache(monkeypatch) -> None:
    _disable_startup_warmups(monkeypatch)
    fake = _FacadeFakeService()
    cached_stats = {
        "success": True,
        "backend": "cached",
        "data_dir": "/tmp/cached",
        "disk": {"total_bytes": 10, "used_bytes": 4, "free_bytes": 6},
        "index_files": {"graph_index.chonk": {"bytes": 222}},
    }
    stats_calls: list[dict[str, object]] = []

    async def _timed_out_stats():
        stats_calls.append({"called": True})
        return {
            "success": False,
            "error": "admin_stats_timeout timeout_s=2.500",
            "error_type": "TimeoutError",
        }

    fake.get_system_stats = _timed_out_stats  # type: ignore[method-assign]
    monkeypatch.setattr(admin_router, "_STATS_CACHE", None)
    monkeypatch.setattr(admin_router, "_STATS_TS", None)
    monkeypatch.setattr(
        admin_router, "_shared_read_stats_cache", lambda _ttl: dict(cached_stats)
    )
    monkeypatch.setattr(indexes_router, "svc", fake)
    monkeypatch.setattr(external_router, "_get_svc", lambda _request=None: fake)

    app = create_app()
    with TestClient(app) as client:
        admin_stats_response = client.get("/v1/external/admin/stats")
        assert admin_stats_response.status_code == 200, admin_stats_response.text
        assert admin_stats_response.json()["backend"] == "cached"

        instance_response = client.get("/v1/external/instance")
        assert instance_response.status_code == 200, instance_response.text
        assert instance_response.json()["success"] is True
        assert instance_response.json()["system_stats"]["backend"] == "cached"

    assert not stats_calls


def test_external_admin_ready_fails_closed_when_service_not_ready(monkeypatch) -> None:
    _disable_startup_warmups(monkeypatch)
    fake = _FacadeFakeService()

    async def _not_ready_health_check():
        fake.calls["health"].append({"mode": "not_ready"})
        return {"success": True, "ok": False, "status": "starting", "backend": "fake"}

    fake.health_check = _not_ready_health_check  # type: ignore[method-assign]
    monkeypatch.setattr(indexes_router, "svc", fake)
    monkeypatch.setattr(external_router, "_get_svc", lambda _request=None: fake)

    app = create_app()
    with TestClient(app) as client:
        response = client.get("/v1/external/admin/ready")
        assert response.status_code == 503, response.text
        assert "starting" in response.text
