from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

try:
    from chonkydb.api.app import create_app
    from chonkydb.api.routers import (
        admin as admin_router,
        external as external_router,
    )
    from chonkydb.api.routers import (
        indexes as indexes_router,
    )
except ImportError as exc:  # pragma: no cover - environment-gated import
    pytest.skip(
        f"external facade tests require full chonkydb runtime: {exc}",
        allow_module_level=True,
    )


_BEHAVIOR_FREEZE_CASES_PATH = (
    Path(__file__).resolve().parents[4]
    / "tests"
    / "fixtures"
    / "chonkydb_productive_lanes"
    / "behavior_freeze_cases.json"
)


def _load_behavior_freeze_case(*keys: str) -> dict[str, object]:
    node: object = json.loads(_BEHAVIOR_FREEZE_CASES_PATH.read_text(encoding="utf-8"))
    for key in keys:
        assert isinstance(node, dict)
        node = node[key]
    assert isinstance(node, dict)
    return dict(node)


def _response_detail_mapping(response) -> dict[str, object]:
    detail = response.json()["detail"]
    if isinstance(detail, dict):
        return dict(detail)
    if isinstance(detail, str):
        try:
            parsed = json.loads(detail)
        except json.JSONDecodeError:
            return {"error": detail}
        if isinstance(parsed, dict):
            return dict(parsed)
    return {"error": str(detail)}


def _http_error_class(status_code: int) -> str | None:
    return None if int(status_code) < 400 else f"http_{int(status_code)}"


def _external_latency_class(*, status_code: int, body: dict[str, object]) -> str:
    if int(status_code) in {202, 303} or str(body.get("execution_mode") or "") == "async":
        return "async_background"
    query_plan = body.get("query_plan")
    if isinstance(query_plan, dict) and isinstance(query_plan.get("latency_ms"), dict):
        return "interactive_measured"
    return "sync_inline"


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


def _clear_external_router_caches() -> None:
    external_router._clear_twinr_current_document_cache()
    external_router._invalidate_scope_allowed_doc_ids_cache(invalidation_targets=None)


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
        return {
            "success": True,
            "results": [{"payload_id": "payload-advanced-1"}],
            "indexes_used": ["fulltext", "hnsw"],
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
        admin_stats_body = admin_stats_response.json()
        assert admin_stats_body["backend"] in {"fake", "bootstrap_cache"}
        if admin_stats_body["backend"] == "fake":
            assert admin_stats_body["data_dir"] == "/tmp/chonkydb-fake"
        else:
            assert admin_stats_body["stats_bootstrap"] is True

        admin_metrics_response = client.get("/v1/external/admin/metrics/basic")
        assert admin_metrics_response.status_code == 200, admin_metrics_response.text
        assert admin_metrics_response.json()["embedding_cache"]["hits"] == 1

        admin_indexes_response = client.get("/v1/external/admin/indexes")
        assert admin_indexes_response.status_code == 200, admin_indexes_response.text
        assert "graph_index" in admin_indexes_response.json()["indexes"]

        admin_storage_response = client.get("/v1/external/admin/storage")
        assert admin_storage_response.status_code == 200, admin_storage_response.text
        assert int(admin_storage_response.json()["tracked_index_bytes"]) >= 0
        assert int(admin_storage_response.json()["index_file_count"]) >= 0

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


def test_external_record_behavior_freeze_sync_record_write(monkeypatch) -> None:
    _disable_startup_warmups(monkeypatch)
    _clear_external_router_caches()
    fake = _FacadeFakeService()
    case = _load_behavior_freeze_case("external", "sync_record_write")
    request_payload = dict(case["input"])
    request_meta = dict(case["http_request"])
    expected_response = dict(case["response"])

    monkeypatch.setattr(indexes_router, "svc", fake)
    monkeypatch.setattr(external_router, "_get_svc", lambda _request=None: fake)

    app = create_app()
    with TestClient(app) as client:
        response = client.post(str(request_meta["path"]), json=request_payload)

    assert response.status_code == int(expected_response["status_code"]), response.text
    body = response.json()
    observed = {
        "success": bool(body.get("success", True)),
        "payload_id": body.get("payload_id"),
        "document_id": body.get("document_id"),
        "operation": body.get("operation"),
        "execution_mode": body.get("execution_mode"),
        "client_request_id": body.get("client_request_id"),
    }
    assert observed == expected_response["body"]
    assert [str(body.get("document_id") or "")] == list(case["doc_id_list"])
    assert _http_error_class(response.status_code) == case["error_class"]
    assert _external_latency_class(status_code=response.status_code, body=body) == case["latency_class"]


def test_external_routes_use_direct_router_logic(monkeypatch) -> None:
    _disable_startup_warmups(monkeypatch)
    fake = _FacadeFakeService()
    monkeypatch.setattr(indexes_router, "svc", fake)
    monkeypatch.setattr(external_router, "_get_svc", lambda _request=None: fake)

    app = create_app()
    with TestClient(app) as client:
        write_response = client.post(
            "/v1/external/records",
            json={"content": "direct body", "client_request_id": "direct-write"},
        )
        read_response = client.post(
            "/v1/external/retrieve/topk_records",
            json={"query_text": "direct query"},
        )

    assert write_response.status_code == 200, write_response.text
    assert write_response.json()["payload_id"] == "payload-0"
    assert write_response.json()["client_request_id"] == "direct-write"
    assert read_response.status_code == 200, read_response.text
    assert read_response.json()["results"][0]["payload_id"] == "payload-advanced-1"
    assert read_response.json()["query_plan"]["scope_cache_hit"] is False
    assert not hasattr(external_router, "_EXTERNAL_WRITER_CONTRACT")
    assert not hasattr(external_router, "_EXTERNAL_READER_CONTRACT")
    assert fake.calls["sync_bulk"][0]["items"][0]["content"] == "direct body"
    assert fake.calls["advanced"][0]["query_text"] == "direct query"


def test_external_topk_payload_blob_loader_contract_mismatch_fails_closed(
    monkeypatch,
) -> None:
    _disable_startup_warmups(monkeypatch)

    class _LegacyTopKDM:
        @staticmethod
        def get_doc_id_for_uuid(value: str) -> int | None:
            return 7001 if str(value) == "payload-advanced-1" else None

        @staticmethod
        def read_document_data(  # noqa: ANN205 - test stub
            doc_id: int,
            subindex_key: str,
            *,
            raise_on_not_found: bool = True,
        ):
            _ = (doc_id, subindex_key, raise_on_not_found)
            return {"payload": {"title": "should-not-succeed"}}

    class _TopKContractService(_FacadeFakeService):
        def __init__(self) -> None:
            super().__init__()
            engine = SimpleNamespace(
                chonk_router=SimpleNamespace(),
                docid_mapping_manager=_LegacyTopKDM(),
            )
            self._api = SimpleNamespace(_engine=engine)

        def _ensure_api_server_main_api_for_request(self, *, timeout_seconds=None):  # noqa: ANN001
            _ = timeout_seconds
            return self._api

        @staticmethod
        def _ensure_engine_docid_router_ready(api, require_router=True):  # noqa: ANN001
            _ = require_router
            return (api._engine, api._engine.docid_mapping_manager, api._engine.chonk_router)

        @staticmethod
        def _resolve_doc_int_for_payload_ref(dm, *, payload_id, chonky_id):  # noqa: ANN001
            _ = dm
            for raw in (payload_id, chonky_id):
                if str(raw) == "payload-advanced-1":
                    return 7001
            return None

        @staticmethod
        def _decode_payload_object(raw, *, on_error_reload=None):  # noqa: ANN001
            _ = on_error_reload
            return raw

        async def query_payloads_advanced(self, **kwargs):  # noqa: ANN003
            self.calls["advanced"].append(dict(kwargs))
            return {
                "success": True,
                "results": [{"payload_id": "payload-advanced-1", "score": 0.75}],
                "indexes_used": ["fulltext"],
            }

    fake = _TopKContractService()
    monkeypatch.setattr(indexes_router, "svc", fake)
    monkeypatch.setattr(external_router, "_get_svc", lambda _request=None: fake)

    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/v1/external/retrieve/topk_records",
            json={"query_text": "contract check"},
        )

    assert response.status_code == 503, response.text
    assert _response_detail_mapping(response)["error"] == (
        "docid_mapping.read_document_data must accept allow_header_reload_on_miss for top-k payload blob loads"
    )


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


@pytest.mark.parametrize(
    ("result", "expected_status"),
    [
        ({"success": False, "status_code": 429, "error_type": "TimeoutError"}, 429),
        ({"success": False, "error_type": "not_found"}, 404),
        ({"success": False, "error_type": "TimeoutError"}, 504),
        ({"success": False, "error_type": "ValidationError"}, 400),
        ({"success": False, "error_type": "ServerBusy"}, 503),
        ({"success": False, "error_type": "TooManyRequests"}, 429),
    ],
)
def test_error_status_from_result_uses_explicit_contract(
    result: dict[str, object],
    expected_status: int,
) -> None:
    assert (
        external_router._error_status_from_result(result, default_status=500)
        == expected_status
    )


def test_error_status_from_result_does_not_infer_from_free_text() -> None:
    assert (
        external_router._error_status_from_result(
            {
                "success": False,
                "error": "document not found during secondary projection rebuild",
                "error_type": "RuntimeError",
            },
            default_status=500,
        )
        == 500
    )


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

    async def _timed_out_stats(*_args, **_kwargs):
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


def test_external_instance_bootstraps_admin_stats_cache_on_cold_cache_miss(
    monkeypatch,
) -> None:
    _disable_startup_warmups(monkeypatch)
    fake = _FacadeFakeService()
    monkeypatch.setattr(admin_router, "_STATS_CACHE", None)
    monkeypatch.setattr(admin_router, "_STATS_TS", None)
    monkeypatch.setattr(admin_router, "_shared_read_stats_cache", lambda _ttl: None)
    monkeypatch.setattr(admin_router, "_shared_read_stats_cache_any", lambda: None)
    monkeypatch.setenv("CHONKDB_DATA_DIR", "/tmp/chonkydb-bootstrap-cache")

    async def _unexpected_stats(*_args, **_kwargs):
        raise AssertionError("live system stats should not run on a cold cache miss")

    fake.get_system_stats = _unexpected_stats  # type: ignore[method-assign]
    monkeypatch.setattr(indexes_router, "svc", fake)
    monkeypatch.setattr(external_router, "_get_svc", lambda _request=None: fake)

    app = create_app()
    with TestClient(app) as client:
        response = client.get("/v1/external/instance")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["success"] is True
    assert body["ready"] is True
    assert body["degraded"] is False
    assert body["system_stats"]["backend"] == "bootstrap_cache"
    assert body["system_stats"]["stats_bootstrap"] is True
    assert body["system_stats"]["stats_bootstrap_reason"] == "cold_cache_miss"


def test_external_instance_avoids_slow_system_stats_on_cold_cache_miss(
    monkeypatch,
) -> None:
    _disable_startup_warmups(monkeypatch)
    fake = _FacadeFakeService()
    monkeypatch.setenv("CHONKY_EXTERNAL_INSTANCE_COMPONENT_TIMEOUT_S", "0.1")
    monkeypatch.setattr(admin_router, "_STATS_CACHE", None)
    monkeypatch.setattr(admin_router, "_STATS_TS", None)
    monkeypatch.setattr(admin_router, "_shared_read_stats_cache", lambda _ttl: None)
    monkeypatch.setattr(admin_router, "_shared_read_stats_cache_any", lambda: None)

    async def _slow_stats(*_args, **_kwargs):
        await asyncio.sleep(0.25)
        return {
            "success": True,
            "backend": "slow",
            "data_dir": "/tmp/slow",
            "disk": {},
            "index_files": {},
            "indexes": {},
        }

    fake.get_system_stats = _slow_stats  # type: ignore[method-assign]
    monkeypatch.setattr(indexes_router, "svc", fake)
    monkeypatch.setattr(external_router, "_get_svc", lambda _request=None: fake)

    app = create_app()
    started = time.perf_counter()
    with TestClient(app) as client:
        response = client.get("/v1/external/instance")
    elapsed_s = time.perf_counter() - started

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["success"] is True
    assert body["ready"] is True
    assert body["degraded"] is False
    assert body.get("degraded_components") is None
    assert body["system_stats"]["success"] is True
    assert body["system_stats"]["backend"] == "bootstrap_cache"
    assert body["system_stats"]["stats_bootstrap"] is True
    assert body["indexes"]["success"] is True
    assert body["basic_metrics"]["success"] is True
    assert body["component_timeout_seconds"] == pytest.approx(0.1)
    assert elapsed_s < 0.5


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


def test_external_topk_scope_ref_uses_scope_cache_and_reports_hits(monkeypatch) -> None:
    _disable_startup_warmups(monkeypatch)
    _clear_external_router_caches()
    monkeypatch.setenv("CHONKY_SCOPE_DOC_IDS_CACHE_TTL_S", "120")
    case = _load_behavior_freeze_case("external", "topk_scope_ref")
    request_meta = dict(case["http_request"])
    request_payload = dict(case["input"])
    expected_response = dict(case["response"])

    snapshot_uri = external_router._snapshot_origin_uri(
        namespace="acme",
        snapshot_kind="objects",
    )
    pointer_uri = external_router._pointer_origin_uri(
        namespace="acme",
        snapshot_kind="objects",
    )
    segment_uri = external_router._segment_origin_uri(
        namespace="acme",
        uri_segment="objects",
        segment_index=0,
    )

    class _ScopeAwareService(_FacadeFakeService):
        async def get_full_document(self, **kwargs):  # noqa: ANN003
            self.calls["full_document"].append(dict(kwargs))
            origin_uri = kwargs.get("origin_uri")
            document_id = kwargs.get("document_id")
            if origin_uri == pointer_uri:
                return {
                    "success": True,
                    "origin_uri": pointer_uri,
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_pointer_v1",
                            "version": 1,
                            "snapshot_kind": "objects",
                            "document_id": "snapshot-doc-1",
                        }
                    ),
                }
            if document_id == "snapshot-doc-1":
                return {
                    "success": True,
                    "document_id": "snapshot-doc-1",
                    "origin_uri": snapshot_uri,
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "acme",
                            "snapshot_kind": "objects",
                            "updated_at": "2026-03-20T17:00:00Z",
                            "body": {
                                "schema": "twinr_memory_object_catalog_v3",
                                "version": 3,
                                "segments": [
                                    {"segment_index": 0, "document_id": "segment-doc-1"}
                                ],
                            },
                        }
                    ),
                }
            if document_id == "segment-doc-1" or origin_uri == segment_uri:
                return {
                    "success": True,
                    "document_id": "segment-doc-1",
                    "origin_uri": segment_uri,
                    "content": json.dumps(
                        {
                            "schema": "twinr_memory_object_catalog_segment_v1",
                            "version": 1,
                            "snapshot_kind": "objects",
                            "items": [
                                {"document_id": "doc-a"},
                                {"document_id": "doc-b"},
                            ],
                        }
                    ),
                }
            raise AssertionError(f"unexpected get_full_document call: {kwargs}")

        async def query_payloads_advanced(self, **kwargs):  # noqa: ANN003
            self.calls["advanced"].append(dict(kwargs))
            return {
                "success": True,
                "results": [],
                "indexes_used": ["fulltext"],
                "debug": {"latency_breakdown_ms": {"fulltext": 1.2}},
            }

    fake = _ScopeAwareService()
    monkeypatch.setattr(indexes_router, "svc", fake)
    monkeypatch.setattr(external_router, "_get_svc", lambda _request=None: fake)

    app = create_app()
    with TestClient(app) as client:
        first = client.post(str(request_meta["path"]), json=request_payload)
        assert first.status_code == 200, first.text
        first_body = first.json()
        observed_first = {
            "success": bool(first_body.get("success")),
            "mode": first_body.get("mode"),
            "results": list(first_body.get("results") or []),
            "indexes_used": list(first_body.get("indexes_used") or []),
            "scope_ref": first_body.get("scope_ref"),
            "query_plan": {
                "scope_cache_hit": bool((first_body.get("query_plan") or {}).get("scope_cache_hit")),
                "latency_ms_keys": sorted(
                    str(key)
                    for key in ((first_body.get("query_plan") or {}).get("latency_ms") or {}).keys()
                ),
            },
        }
        assert observed_first == expected_response["body"]
        assert _http_error_class(first.status_code) == case["error_class"]
        assert _external_latency_class(status_code=first.status_code, body=first_body) == case["latency_class"]
        assert first.json()["query_plan"]["scope_cache_hit"] is False

        second = client.post(str(request_meta["path"]), json=request_payload)
        assert second.status_code == 200, second.text
        assert second.json()["query_plan"]["scope_cache_hit"] is True

    assert fake.calls["advanced"][0]["allowed_doc_ids"] == list(case["doc_id_list"])
    assert fake.calls["advanced"][1]["allowed_doc_ids"] == list(case["doc_id_list"])
    assert len(fake.calls["full_document"]) == 3
    assert fake.calls["full_document"][0]["max_content_chars"] == 32_768
    assert fake.calls["full_document"][1]["max_content_chars"] == 512_000
    assert fake.calls["full_document"][2]["max_content_chars"] == 512_000


def test_external_get_full_document_caps_twinr_pointer_origin_reads(monkeypatch) -> None:
    _disable_startup_warmups(monkeypatch)
    _clear_external_router_caches()

    pointer_uri = external_router._pointer_origin_uri(
        namespace="acme",
        snapshot_kind="objects",
    )

    class _PointerService(_FacadeFakeService):
        async def get_full_document(self, **kwargs):  # noqa: ANN003
            self.calls["full_document"].append(dict(kwargs))
            return {
                "success": True,
                "document_id": "pointer-doc",
                "origin_uri": pointer_uri,
                "content": json.dumps(
                    {
                        "schema": "twinr_remote_snapshot_pointer_v1",
                        "version": 1,
                        "snapshot_kind": "objects",
                        "document_id": "snapshot-doc-1",
                    }
                ),
            }

    fake = _PointerService()
    monkeypatch.setattr(indexes_router, "svc", fake)
    monkeypatch.setattr(external_router, "_get_svc", lambda _request=None: fake)

    app = create_app()
    with TestClient(app) as client:
        response = client.get(
            f"/v1/external/documents/full?origin_uri={pointer_uri}&max_content_chars=2000000"
        )
        assert response.status_code == 200, response.text

    assert len(fake.calls["full_document"]) == 1
    assert fake.calls["full_document"][0]["max_content_chars"] == 32_768


def test_external_get_full_document_uses_pointer_head_for_twinr_snapshot_origin(
    monkeypatch,
) -> None:
    _disable_startup_warmups(monkeypatch)
    _clear_external_router_caches()

    snapshot_uri = external_router._snapshot_origin_uri(
        namespace="acme",
        snapshot_kind="prompt_memory",
    )
    pointer_uri = external_router._pointer_origin_uri(
        namespace="acme",
        snapshot_kind="prompt_memory",
    )
    snapshot_payload = {
        "schema": "twinr_remote_snapshot_v1",
        "namespace": "acme",
        "snapshot_kind": "prompt_memory",
        "updated_at": "2026-03-29T16:36:00Z",
        "body": {"messages": ["hello"]},
    }

    class _SnapshotHeadService(_FacadeFakeService):
        async def get_full_document(self, **kwargs):  # noqa: ANN003
            self.calls["full_document"].append(dict(kwargs))
            if kwargs.get("origin_uri") == pointer_uri:
                return {
                    "success": True,
                    "document_id": "pointer-doc",
                    "origin_uri": pointer_uri,
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_pointer_v1",
                            "version": 1,
                            "snapshot_kind": "prompt_memory",
                            "document_id": "snapshot-doc-1",
                        }
                    ),
                }
            if kwargs.get("document_id") == "snapshot-doc-1":
                return {
                    "success": True,
                    "document_id": "snapshot-doc-1",
                    "origin_uri": snapshot_uri,
                    "content": json.dumps(snapshot_payload),
                }
            raise AssertionError(f"unexpected full_document kwargs: {kwargs}")

    fake = _SnapshotHeadService()
    monkeypatch.setattr(indexes_router, "svc", fake)
    monkeypatch.setattr(external_router, "_get_svc", lambda _request=None: fake)

    app = create_app()
    with TestClient(app) as client:
        response = client.get(
            f"/v1/external/documents/full?origin_uri={snapshot_uri}&max_content_chars=2000000"
        )
        assert response.status_code == 200, response.text
        assert json.loads(response.json()["content"]) == snapshot_payload

    assert len(fake.calls["full_document"]) == 2
    assert fake.calls["full_document"][0]["origin_uri"] == pointer_uri
    assert fake.calls["full_document"][0]["max_content_chars"] == 32_768
    assert fake.calls["full_document"][1]["document_id"] == "snapshot-doc-1"
    assert fake.calls["full_document"][1]["max_content_chars"] == 512_000


def test_external_get_full_document_uses_origin_lookup_heads_before_origin_uri_fallback(
    monkeypatch,
) -> None:
    _disable_startup_warmups(monkeypatch)
    _clear_external_router_caches()

    snapshot_uri = external_router._snapshot_origin_uri(
        namespace="acme",
        snapshot_kind="prompt_memory",
    )
    pointer_uri = external_router._pointer_origin_uri(
        namespace="acme",
        snapshot_kind="prompt_memory",
    )
    snapshot_payload = {
        "schema": "twinr_remote_snapshot_v1",
        "namespace": "acme",
        "snapshot_kind": "prompt_memory",
        "updated_at": "2026-03-29T16:36:00Z",
        "body": {"messages": ["hello"]},
    }

    class _OriginLookupRouter:
        @staticmethod
        def _origin_lookup_payload_key(value: str) -> str:
            digest = hashlib.sha256(
                str(value).encode("utf-8", errors="ignore")
            ).hexdigest()
            return f"origin_lookup_{digest}"

        def __init__(self) -> None:
            self.payload_store = {
                self._origin_lookup_payload_key(pointer_uri): {
                    "doc_ids": [1001, 1002],
                }
            }

        def read_payload(self, *, key: str, allow_header_reload_on_miss: bool = True):  # noqa: ANN201
            _ = allow_header_reload_on_miss
            if key not in self.payload_store:
                raise KeyError(key)
            return self.payload_store[key]

    class _OriginLookupDM:
        def get_uuid_for_doc_id(self, doc_id: int) -> str | None:
            return {
                1001: "pointer-doc-old",
                1002: "pointer-doc-new",
            }.get(int(doc_id))

    class _OriginLookupHeadService(_FacadeFakeService):
        def __init__(self) -> None:
            super().__init__()
            engine = SimpleNamespace(
                chonk_router=_OriginLookupRouter(),
                docid_mapping_manager=_OriginLookupDM(),
            )
            self._api = SimpleNamespace(_engine=engine)

        def _ensure_api_server_main_api_for_request(self, *, timeout_seconds=None):  # noqa: ANN001
            _ = timeout_seconds
            return self._api

        @staticmethod
        def _ensure_engine_docid_router_ready(api, require_router=True):  # noqa: ANN001
            _ = require_router
            return (api._engine, api._engine.docid_mapping_manager, api._engine.chonk_router)

        @staticmethod
        def _decode_payload_object(raw, *, on_error_reload=None):  # noqa: ANN001
            _ = on_error_reload
            return raw

        @staticmethod
        def _payload_uuid_for_doc_int(dm, doc_int):  # noqa: ANN001
            return dm.get_uuid_for_doc_id(int(doc_int))

        async def get_full_document(self, **kwargs):  # noqa: ANN003
            self.calls["full_document"].append(dict(kwargs))
            if kwargs.get("document_id") == "pointer-doc-new":
                return {
                    "success": True,
                    "document_id": "pointer-doc-new",
                    "origin_uri": pointer_uri,
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_pointer_v1",
                            "version": 1,
                            "snapshot_kind": "prompt_memory",
                            "document_id": "snapshot-doc-1",
                        }
                    ),
                }
            if kwargs.get("document_id") == "snapshot-doc-1":
                return {
                    "success": True,
                    "document_id": "snapshot-doc-1",
                    "origin_uri": snapshot_uri,
                    "content": json.dumps(snapshot_payload),
                }
            raise AssertionError(f"unexpected full_document kwargs: {kwargs}")

    fake = _OriginLookupHeadService()
    monkeypatch.setattr(indexes_router, "svc", fake)
    monkeypatch.setattr(external_router, "_get_svc", lambda _request=None: fake)

    app = create_app()
    with TestClient(app) as client:
        response = client.get(
            f"/v1/external/documents/full?origin_uri={snapshot_uri}&max_content_chars=2000000"
        )
        assert response.status_code == 200, response.text
        assert json.loads(response.json()["content"]) == snapshot_payload

    assert len(fake.calls["full_document"]) == 2
    assert fake.calls["full_document"][0]["document_id"] == "pointer-doc-new"
    assert fake.calls["full_document"][1]["document_id"] == "snapshot-doc-1"
    assert fake.calls["full_document"][0]["origin_uri"] == pointer_uri
    assert fake.calls["full_document"][1]["origin_uri"] == snapshot_uri


def test_external_get_full_document_uses_snapshot_origin_lookup_head_when_pointer_missing(
    monkeypatch,
) -> None:
    _disable_startup_warmups(monkeypatch)
    _clear_external_router_caches()

    snapshot_uri = external_router._snapshot_origin_uri(
        namespace="acme",
        snapshot_kind="objects",
    )
    pointer_uri = external_router._pointer_origin_uri(
        namespace="acme",
        snapshot_kind="objects",
    )
    newer_snapshot_payload = {
        "schema": "twinr_remote_snapshot_v1",
        "namespace": "acme",
        "snapshot_kind": "objects",
        "updated_at": "2026-03-29T16:40:00Z",
        "body": {
            "schema": "twinr_memory_object_catalog_v2",
            "version": 2,
            "items": [{"document_id": "doc-new"}],
        },
    }

    class _OriginLookupRouter:
        @staticmethod
        def _origin_lookup_payload_key(value: str) -> str:
            digest = hashlib.sha256(
                str(value).encode("utf-8", errors="ignore")
            ).hexdigest()
            return f"origin_lookup_{digest}"

        def __init__(self) -> None:
            self.payload_store = {
                self._origin_lookup_payload_key(snapshot_uri): {
                    "doc_ids": [2001, 2002],
                }
            }

        def read_payload(self, *, key: str, allow_header_reload_on_miss: bool = True):  # noqa: ANN201
            _ = allow_header_reload_on_miss
            if key not in self.payload_store:
                raise KeyError(key)
            return self.payload_store[key]

    class _OriginLookupDM:
        def get_uuid_for_doc_id(self, doc_id: int) -> str | None:
            return {
                2001: "snapshot-doc-old",
                2002: "snapshot-doc-new",
            }.get(int(doc_id))

    class _SnapshotFallbackService(_FacadeFakeService):
        def __init__(self) -> None:
            super().__init__()
            engine = SimpleNamespace(
                chonk_router=_OriginLookupRouter(),
                docid_mapping_manager=_OriginLookupDM(),
            )
            self._api = SimpleNamespace(_engine=engine)

        def _ensure_api_server_main_api_for_request(self, *, timeout_seconds=None):  # noqa: ANN001
            _ = timeout_seconds
            return self._api

        @staticmethod
        def _ensure_engine_docid_router_ready(api, require_router=True):  # noqa: ANN001
            _ = require_router
            return (api._engine, api._engine.docid_mapping_manager, api._engine.chonk_router)

        @staticmethod
        def _decode_payload_object(raw, *, on_error_reload=None):  # noqa: ANN001
            _ = on_error_reload
            return raw

        @staticmethod
        def _payload_uuid_for_doc_int(dm, doc_int):  # noqa: ANN001
            return dm.get_uuid_for_doc_id(int(doc_int))

        async def get_full_document(self, **kwargs):  # noqa: ANN003
            self.calls["full_document"].append(dict(kwargs))
            if kwargs.get("document_id") == "snapshot-doc-new":
                return {
                    "success": True,
                    "document_id": "snapshot-doc-new",
                    "origin_uri": snapshot_uri,
                    "content": json.dumps(newer_snapshot_payload),
                }
            raise AssertionError(f"unexpected full_document kwargs: {kwargs}")

    fake = _SnapshotFallbackService()
    monkeypatch.setattr(indexes_router, "svc", fake)
    monkeypatch.setattr(external_router, "_get_svc", lambda _request=None: fake)

    app = create_app()
    with TestClient(app) as client:
        response = client.get(
            f"/v1/external/documents/full?origin_uri={snapshot_uri}&max_content_chars=2000000"
        )
        assert response.status_code == 200, response.text
        assert json.loads(response.json()["content"]) == newer_snapshot_payload

    assert len(fake.calls["full_document"]) == 1
    assert fake.calls["full_document"][0]["document_id"] == "snapshot-doc-new"
    assert fake.calls["full_document"][0]["origin_uri"] == snapshot_uri


def test_external_origin_lookup_head_uses_batch_uuid_resolution_and_prunes_stale_doc_ids(
    monkeypatch,
) -> None:
    _disable_startup_warmups(monkeypatch)
    _clear_external_router_caches()

    snapshot_uri = external_router._snapshot_origin_uri(
        namespace="acme",
        snapshot_kind="conflicts",
    )

    class _BatchRouter:
        @staticmethod
        def _origin_lookup_payload_key(value: str) -> str:
            digest = hashlib.sha256(
                str(value).encode("utf-8", errors="ignore")
            ).hexdigest()
            return f"origin_lookup_{digest}"

        def __init__(self) -> None:
            self.payload_store = {
                self._origin_lookup_payload_key(snapshot_uri): {
                    "v": 1,
                    "lookup_value": snapshot_uri,
                    "doc_ids": [4001, 4002],
                    "doc_count": 2,
                }
            }
            self.store_calls: list[dict[str, object]] = []

        def read_payload(self, *, key: str, allow_header_reload_on_miss: bool = True):  # noqa: ANN201
            _ = allow_header_reload_on_miss
            if key not in self.payload_store:
                raise KeyError(key)
            return self.payload_store[key]

        @staticmethod
        def _pack_chonkbin(payload):  # noqa: ANN001, ANN205 - test stub
            return dict(payload)

        def store_payload(self, key, data, data_type=None, tx_id=None):  # noqa: ANN001, ANN201
            self.store_calls.append(
                {
                    "key": str(key),
                    "data": dict(data),
                    "data_type": data_type,
                    "tx_id": tx_id,
                }
            )
            self.payload_store[str(key)] = dict(data)
            return (0, 0)

    class _BatchDM:
        def __init__(self) -> None:
            self.batch_calls: list[list[int]] = []
            self.single_calls: list[int] = []

        def get_uuid_for_doc_ids_batch(self, doc_ids: list[int]) -> dict[int, str | None]:  # noqa: D401
            normalized = [int(doc_id) for doc_id in list(doc_ids or [])]
            self.batch_calls.append(list(normalized))
            return {
                4001: None,
                4002: "snapshot-doc-live",
            }

        def get_uuid_for_doc_id(self, doc_id: int) -> str | None:  # noqa: D401
            self.single_calls.append(int(doc_id))
            raise AssertionError("batch uuid resolution should be used for stale head pruning")

    class _BatchHeadService(_FacadeFakeService):
        def __init__(self) -> None:
            super().__init__()
            engine = SimpleNamespace(
                chonk_router=_BatchRouter(),
                docid_mapping_manager=_BatchDM(),
            )
            self._api = SimpleNamespace(_engine=engine)

        def _ensure_api_server_main_api_for_request(self, *, timeout_seconds=None):  # noqa: ANN001
            _ = timeout_seconds
            return self._api

        @staticmethod
        def _ensure_engine_docid_router_ready(api, require_router=True):  # noqa: ANN001
            _ = require_router
            return (api._engine, api._engine.docid_mapping_manager, api._engine.chonk_router)

        @staticmethod
        def _decode_payload_object(raw, *, on_error_reload=None):  # noqa: ANN001
            _ = on_error_reload
            return raw

    fake = _BatchHeadService()

    resolution = external_router._resolve_latest_live_origin_lookup_document_id(
        fake,
        origin_uri=snapshot_uri,
        scope_phase="snapshot_catalog",
        namespace="acme",
        snapshot_kind="conflicts",
    )

    router = fake._api._engine.chonk_router
    dm = fake._api._engine.docid_mapping_manager
    lookup_key = router._origin_lookup_payload_key(snapshot_uri)

    assert resolution["available"] is True
    assert resolution["document_id"] == "snapshot-doc-live"
    assert dm.batch_calls == [[4002, 4001]]
    assert dm.single_calls == []
    assert len(router.store_calls) == 1
    assert router.store_calls[0]["key"] == lookup_key
    assert router.store_calls[0]["data"]["doc_ids"] == [4002]
    assert router.payload_store[lookup_key]["doc_count"] == 1


def test_external_origin_lookup_head_contract_mismatch_fails_closed() -> None:
    snapshot_uri = external_router._snapshot_origin_uri(
        namespace="acme",
        snapshot_kind="conflicts",
    )

    class _LegacyRouter:
        @staticmethod
        def _origin_lookup_payload_key(value: str) -> str:
            digest = hashlib.sha256(
                str(value).encode("utf-8", errors="ignore")
            ).hexdigest()
            return f"origin_lookup_{digest}"

        def __init__(self) -> None:
            self.payload_store = {
                self._origin_lookup_payload_key(snapshot_uri): {
                    "doc_ids": [4101],
                }
            }

        def read_payload(self, *, key: str):  # noqa: ANN201
            if key not in self.payload_store:
                raise KeyError(key)
            return self.payload_store[key]

    class _LookupDM:
        @staticmethod
        def get_uuid_for_doc_ids_batch(doc_ids: list[int]) -> dict[int, str | None]:
            return {int(doc_id): "snapshot-doc-live" for doc_id in list(doc_ids or [])}

    class _LegacyLookupService(_FacadeFakeService):
        def __init__(self) -> None:
            super().__init__()
            engine = SimpleNamespace(
                chonk_router=_LegacyRouter(),
                docid_mapping_manager=_LookupDM(),
            )
            self._api = SimpleNamespace(_engine=engine)

        def _ensure_api_server_main_api_for_request(self, *, timeout_seconds=None):  # noqa: ANN001
            _ = timeout_seconds
            return self._api

        @staticmethod
        def _ensure_engine_docid_router_ready(api, require_router=True):  # noqa: ANN001
            _ = require_router
            return (api._engine, api._engine.docid_mapping_manager, api._engine.chonk_router)

        @staticmethod
        def _decode_payload_object(raw, *, on_error_reload=None):  # noqa: ANN001
            _ = on_error_reload
            return raw

    with pytest.raises(external_router.HTTPException) as excinfo:
        external_router._resolve_latest_live_origin_lookup_document_id(
            _LegacyLookupService(),
            origin_uri=snapshot_uri,
            scope_phase="snapshot_catalog",
            namespace="acme",
            snapshot_kind="conflicts",
        )

    assert excinfo.value.status_code == 503
    assert (
        excinfo.value.detail["error_code"]
        == "origin_lookup_read_payload_contract_mismatch"
    )


def test_external_get_full_document_reads_snapshot_head_via_payload_blob_fastpath(
    monkeypatch,
) -> None:
    _disable_startup_warmups(monkeypatch)
    _clear_external_router_caches()

    snapshot_uri = external_router._snapshot_origin_uri(
        namespace="acme",
        snapshot_kind="conflicts",
    )
    snapshot_payload = {
        "schema": "twinr_remote_snapshot_v1",
        "namespace": "acme",
        "snapshot_kind": "conflicts",
        "updated_at": "2026-03-29T17:58:00Z",
        "body": {
            "schema": "twinr_memory_conflict_catalog_v3",
            "version": 3,
            "items": [{"document_id": "conflict-a"}],
        },
    }

    class _BlobRouter:
        @staticmethod
        def _origin_lookup_payload_key(value: str) -> str:
            digest = hashlib.sha256(
                str(value).encode("utf-8", errors="ignore")
            ).hexdigest()
            return f"origin_lookup_{digest}"

        def __init__(self) -> None:
            self.payload_store = {
                self._origin_lookup_payload_key(snapshot_uri): {
                    "doc_ids": [5002],
                }
            }

        def read_payload(self, *, key: str, allow_header_reload_on_miss: bool = True):  # noqa: ANN201
            _ = allow_header_reload_on_miss
            if key not in self.payload_store:
                raise KeyError(key)
            return self.payload_store[key]

    class _BlobDM:
        def get_uuid_for_doc_ids_batch(self, doc_ids: list[int]) -> dict[int, str | None]:  # noqa: D401
            return {
                int(doc_id): "snapshot-doc-blob" if int(doc_id) == 5002 else None
                for doc_id in list(doc_ids or [])
            }

        @staticmethod
        def get_doc_id_for_uuid(value: str) -> int | None:
            return 5002 if str(value) == "snapshot-doc-blob" else None

        @staticmethod
        def read_document_data(  # noqa: ANN205 - test stub
            doc_id: int,
            subindex_key: str,
            *,
            raise_on_not_found: bool = True,
            allow_header_reload_on_miss: bool = True,
            **_kwargs,
        ):
            _ = (raise_on_not_found, allow_header_reload_on_miss)
            if int(doc_id) == 5002 and str(subindex_key) == "payload":
                return dict(snapshot_payload)
            return None

    class _BlobFastPathService(_FacadeFakeService):
        def __init__(self) -> None:
            super().__init__()
            engine = SimpleNamespace(
                chonk_router=_BlobRouter(),
                docid_mapping_manager=_BlobDM(),
            )
            self._api = SimpleNamespace(_engine=engine)

        def _ensure_api_server_main_api_for_request(self, *, timeout_seconds=None):  # noqa: ANN001
            _ = timeout_seconds
            return self._api

        @staticmethod
        def _ensure_engine_docid_router_ready(api, require_router=True):  # noqa: ANN001
            _ = require_router
            return (api._engine, api._engine.docid_mapping_manager, api._engine.chonk_router)

        @staticmethod
        def _resolve_doc_int_for_payload_ref(dm, *, payload_id, chonky_id):  # noqa: ANN001
            for raw in (payload_id, chonky_id):
                if str(raw) == "snapshot-doc-blob":
                    return 5002
            return None

        @staticmethod
        def _decode_payload_object(raw, *, on_error_reload=None):  # noqa: ANN001
            _ = on_error_reload
            return raw

        async def get_full_document(self, **kwargs):  # noqa: ANN003
            self.calls["full_document"].append(dict(kwargs))
            raise AssertionError(f"payload blob fastpath should bypass get_full_document: {kwargs}")

    fake = _BlobFastPathService()
    monkeypatch.setattr(indexes_router, "svc", fake)
    monkeypatch.setattr(external_router, "_get_svc", lambda _request=None: fake)

    app = create_app()
    with TestClient(app) as client:
        response = client.get(f"/v1/external/documents/full?origin_uri={snapshot_uri}")
        assert response.status_code == 200, response.text
        assert response.json()["document_id"] == "snapshot-doc-blob"
        assert response.json()["origin_uri"] == snapshot_uri
        assert json.loads(response.json()["content"]) == snapshot_payload

    assert fake.calls["full_document"] == []


def test_external_get_full_document_payload_blob_contract_mismatch_fails_closed(
    monkeypatch,
) -> None:
    _disable_startup_warmups(monkeypatch)
    _clear_external_router_caches()

    snapshot_uri = external_router._snapshot_origin_uri(
        namespace="acme",
        snapshot_kind="conflicts",
    )

    class _BlobRouter:
        @staticmethod
        def _origin_lookup_payload_key(value: str) -> str:
            digest = hashlib.sha256(
                str(value).encode("utf-8", errors="ignore")
            ).hexdigest()
            return f"origin_lookup_{digest}"

        def __init__(self) -> None:
            self.payload_store = {
                self._origin_lookup_payload_key(snapshot_uri): {
                    "doc_ids": [5004],
                }
            }

        def read_payload(self, *, key: str, allow_header_reload_on_miss: bool = True):  # noqa: ANN201
            _ = allow_header_reload_on_miss
            if key not in self.payload_store:
                raise KeyError(key)
            return self.payload_store[key]

    class _LegacyBlobDM:
        @staticmethod
        def get_uuid_for_doc_ids_batch(doc_ids: list[int]) -> dict[int, str | None]:
            return {
                int(doc_id): "snapshot-doc-legacy" if int(doc_id) == 5004 else None
                for doc_id in list(doc_ids or [])
            }

        @staticmethod
        def get_doc_id_for_uuid(value: str) -> int | None:
            return 5004 if str(value) == "snapshot-doc-legacy" else None

        @staticmethod
        def read_document_data(  # noqa: ANN205 - test stub
            doc_id: int,
            subindex_key: str,
            *,
            raise_on_not_found: bool = True,
        ):
            _ = (doc_id, subindex_key, raise_on_not_found)
            return {"schema": external_router._TWINR_REMOTE_SNAPSHOT_SCHEMA}

    class _BlobContractService(_FacadeFakeService):
        def __init__(self) -> None:
            super().__init__()
            engine = SimpleNamespace(
                chonk_router=_BlobRouter(),
                docid_mapping_manager=_LegacyBlobDM(),
            )
            self._api = SimpleNamespace(_engine=engine)

        def _ensure_api_server_main_api_for_request(self, *, timeout_seconds=None):  # noqa: ANN001
            _ = timeout_seconds
            return self._api

        @staticmethod
        def _ensure_engine_docid_router_ready(api, require_router=True):  # noqa: ANN001
            _ = require_router
            return (api._engine, api._engine.docid_mapping_manager, api._engine.chonk_router)

        @staticmethod
        def _resolve_doc_int_for_payload_ref(dm, *, payload_id, chonky_id):  # noqa: ANN001
            _ = dm
            for raw in (payload_id, chonky_id):
                if str(raw) == "snapshot-doc-legacy":
                    return 5004
            return None

        @staticmethod
        def _decode_payload_object(raw, *, on_error_reload=None):  # noqa: ANN001
            _ = on_error_reload
            return raw

        async def get_full_document(self, **kwargs):  # noqa: ANN003
            self.calls["full_document"].append(dict(kwargs))
            raise AssertionError(f"payload blob contract mismatch should fail before service fallback: {kwargs}")

    fake = _BlobContractService()
    monkeypatch.setattr(indexes_router, "svc", fake)
    monkeypatch.setattr(external_router, "_get_svc", lambda _request=None: fake)

    app = create_app()
    with TestClient(app) as client:
        response = client.get(f"/v1/external/documents/full?origin_uri={snapshot_uri}")

    assert response.status_code == 503, response.text
    assert _response_detail_mapping(response)["error"] == (
        "docid_mapping.read_document_data must accept allow_header_reload_on_miss for payload blob fastpaths"
    )
    assert fake.calls["full_document"] == []


def test_external_get_full_document_reads_snapshot_head_via_component_text_fastpath(
    monkeypatch,
) -> None:
    _disable_startup_warmups(monkeypatch)
    _clear_external_router_caches()

    snapshot_uri = external_router._snapshot_origin_uri(
        namespace="acme",
        snapshot_kind="conflicts",
    )
    snapshot_payload = {
        "schema": "twinr_remote_snapshot_v1",
        "namespace": "acme",
        "snapshot_kind": "conflicts",
        "updated_at": "2026-03-29T18:02:00Z",
        "body": {
            "schema": "twinr_memory_conflict_catalog_v3",
            "version": 3,
            "items": [{"document_id": "conflict-b"}],
        },
    }

    class _ComponentRouter:
        @staticmethod
        def _origin_lookup_payload_key(value: str) -> str:
            digest = hashlib.sha256(
                str(value).encode("utf-8", errors="ignore")
            ).hexdigest()
            return f"origin_lookup_{digest}"

        def __init__(self) -> None:
            self.payload_store = {
                self._origin_lookup_payload_key(snapshot_uri): {
                    "doc_ids": [5003],
                }
            }

        def read_payload(self, *, key: str, allow_header_reload_on_miss: bool = True):  # noqa: ANN201
            _ = allow_header_reload_on_miss
            if key not in self.payload_store:
                raise KeyError(key)
            return self.payload_store[key]

    class _ComponentDM:
        def get_uuid_for_doc_ids_batch(self, doc_ids: list[int]) -> dict[int, str | None]:  # noqa: D401
            return {
                int(doc_id): "snapshot-doc-components"
                if int(doc_id) == 5003
                else None
                for doc_id in list(doc_ids or [])
            }

        @staticmethod
        def get_doc_id_for_uuid(value: str) -> int | None:
            return 5003 if str(value) == "snapshot-doc-components" else None

        @staticmethod
        def read_document_data(  # noqa: ANN205 - test stub
            doc_id: int,
            subindex_key: str,
            *,
            raise_on_not_found: bool = True,
            allow_header_reload_on_miss: bool = True,
            **_kwargs,
        ):
            _ = (raise_on_not_found, allow_header_reload_on_miss)
            if int(doc_id) == 5003 and str(subindex_key) == "payload":
                return {"schema": "plain_payload", "metadata": {"note": "non_cacheable"}}
            return None

    class _ComponentFastPathService(_FacadeFakeService):
        def __init__(self) -> None:
            super().__init__()
            self.component_reads: list[dict[str, object]] = []
            engine = SimpleNamespace(
                chonk_router=_ComponentRouter(),
                docid_mapping_manager=_ComponentDM(),
            )
            self._api = SimpleNamespace(_engine=engine)

        def _ensure_api_server_main_api_for_request(self, *, timeout_seconds=None):  # noqa: ANN001
            _ = timeout_seconds
            return self._api

        @staticmethod
        def _ensure_engine_docid_router_ready(api, require_router=True):  # noqa: ANN001
            _ = require_router
            return (api._engine, api._engine.docid_mapping_manager, api._engine.chonk_router)

        @staticmethod
        def _resolve_doc_int_for_payload_ref(dm, *, payload_id, chonky_id):  # noqa: ANN001
            _ = dm
            for raw in (payload_id, chonky_id):
                if str(raw) == "snapshot-doc-components":
                    return 5003
            return None

        @staticmethod
        def _decode_payload_object(raw, *, on_error_reload=None):  # noqa: ANN001
            _ = on_error_reload
            return raw

        def _read_payload_components_fast(  # noqa: ANN001
            self,
            *,
            dm,
            router,
            doc_int,
            include_content,
            content_mode=None,
            content_char_limit=None,
        ):
            _ = (dm, router)
            self.component_reads.append(
                {
                    "doc_int": int(doc_int),
                    "include_content": bool(include_content),
                    "content_mode": content_mode,
                    "content_char_limit": content_char_limit,
                }
            )
            return (
                {"origin_uri": snapshot_uri},
                json.dumps(snapshot_payload),
            )

        async def get_full_document(self, **kwargs):  # noqa: ANN003
            self.calls["full_document"].append(dict(kwargs))
            raise AssertionError(
                f"component text fastpath should bypass get_full_document: {kwargs}"
            )

    fake = _ComponentFastPathService()
    monkeypatch.setattr(indexes_router, "svc", fake)
    monkeypatch.setattr(external_router, "_get_svc", lambda _request=None: fake)

    app = create_app()
    with TestClient(app) as client:
        response = client.get(f"/v1/external/documents/full?origin_uri={snapshot_uri}")
        assert response.status_code == 200, response.text
        assert response.json()["document_id"] == "snapshot-doc-components"
        assert response.json()["origin_uri"] == snapshot_uri
        assert response.json()["metadata"] == {"origin_uri": snapshot_uri}
        assert json.loads(response.json()["content"]) == snapshot_payload

    assert fake.calls["full_document"] == []
    assert fake.component_reads == [
        {
            "doc_int": 5003,
            "include_content": True,
            "content_mode": "full",
            "content_char_limit": None,
        }
    ]


def test_external_get_full_document_retries_snapshot_origin_during_warmup(
    monkeypatch,
) -> None:
    _disable_startup_warmups(monkeypatch)
    _clear_external_router_caches()

    monkeypatch.setattr(
        external_router.health_router,
        "_warmup_pending_payload",
        lambda _request, readiness_scope="full": {
            "pending": True,
            "pending_count": 1,
            "pending_tasks": ["chonky_warmup_docid_mapping"],
            "blocking_pending": True,
            "blocking_count": 1,
            "blocking_tasks": ["chonky_warmup_docid_mapping"],
            "nonblocking_pending": False,
            "nonblocking_count": 0,
            "nonblocking_tasks": [],
            "readiness_scope": str(readiness_scope),
            "tracked_count": 1,
            "fulltext_pending_indexes": [],
            "fulltext_states": {},
            "vector_pending_indexes": [],
            "vector_states": {},
            "temporal_pending_indexes": [],
            "temporal_states": {},
            "graph_pending_indexes": [],
            "graph_states": {},
        },
    )

    snapshot_uri = external_router._snapshot_origin_uri(
        namespace="acme",
        snapshot_kind="conflicts",
    )
    pointer_uri = external_router._pointer_origin_uri(
        namespace="acme",
        snapshot_kind="conflicts",
    )
    snapshot_payload = {
        "schema": "twinr_remote_snapshot_v1",
        "namespace": "acme",
        "snapshot_kind": "conflicts",
        "updated_at": "2026-03-29T18:46:00Z",
        "body": {"items": [{"document_id": "conflict-c"}]},
    }

    class _WarmupRetryService(_FacadeFakeService):
        def __init__(self) -> None:
            super().__init__()
            self.snapshot_origin_attempts = 0

        async def get_full_document(self, **kwargs):  # noqa: ANN003
            self.calls["full_document"].append(dict(kwargs))
            if kwargs.get("origin_uri") == pointer_uri:
                return {
                    "success": False,
                    "error": "document_not_found",
                    "error_type": "not_found",
                }
            if kwargs.get("origin_uri") == snapshot_uri:
                self.snapshot_origin_attempts += 1
                if self.snapshot_origin_attempts == 1:
                    return {
                        "success": False,
                        "error": "document_not_found",
                        "error_type": "not_found",
                    }
                return {
                    "success": True,
                    "document_id": "snapshot-doc-after-warmup",
                    "origin_uri": snapshot_uri,
                    "content": json.dumps(snapshot_payload),
                }
            raise AssertionError(f"unexpected full_document kwargs: {kwargs}")

    fake = _WarmupRetryService()
    monkeypatch.setattr(indexes_router, "svc", fake)
    monkeypatch.setattr(external_router, "_get_svc", lambda _request=None: fake)

    app = create_app()
    with TestClient(app) as client:
        response = client.get(f"/v1/external/documents/full?origin_uri={snapshot_uri}")
        assert response.status_code == 200, response.text
        assert response.json()["document_id"] == "snapshot-doc-after-warmup"
        assert json.loads(response.json()["content"]) == snapshot_payload

    assert fake.snapshot_origin_attempts == 2
    assert [call.get("origin_uri") for call in fake.calls["full_document"]] == [
        pointer_uri,
        snapshot_uri,
        pointer_uri,
        snapshot_uri,
    ]


def test_external_get_full_document_returns_503_while_snapshot_warmup_pending(
    monkeypatch,
) -> None:
    _disable_startup_warmups(monkeypatch)
    _clear_external_router_caches()

    monkeypatch.setattr(
        external_router.health_router,
        "_warmup_pending_payload",
        lambda _request, readiness_scope="full": {
            "pending": True,
            "pending_count": 1,
            "pending_tasks": ["chonky_warmup_docid_mapping"],
            "blocking_pending": True,
            "blocking_count": 1,
            "blocking_tasks": ["chonky_warmup_docid_mapping"],
            "nonblocking_pending": False,
            "nonblocking_count": 0,
            "nonblocking_tasks": [],
            "readiness_scope": str(readiness_scope),
            "tracked_count": 1,
            "fulltext_pending_indexes": [],
            "fulltext_states": {},
            "vector_pending_indexes": [],
            "vector_states": {},
            "temporal_pending_indexes": [],
            "temporal_states": {},
            "graph_pending_indexes": [],
            "graph_states": {},
        },
    )

    snapshot_uri = external_router._snapshot_origin_uri(
        namespace="acme",
        snapshot_kind="conflicts",
    )
    pointer_uri = external_router._pointer_origin_uri(
        namespace="acme",
        snapshot_kind="conflicts",
    )

    class _Warmup503Service(_FacadeFakeService):
        async def get_full_document(self, **kwargs):  # noqa: ANN003
            self.calls["full_document"].append(dict(kwargs))
            if kwargs.get("origin_uri") in {pointer_uri, snapshot_uri}:
                return {
                    "success": False,
                    "error": "document_not_found",
                    "error_type": "not_found",
                }
            raise AssertionError(f"unexpected full_document kwargs: {kwargs}")

    fake = _Warmup503Service()
    monkeypatch.setattr(indexes_router, "svc", fake)
    monkeypatch.setattr(external_router, "_get_svc", lambda _request=None: fake)

    app = create_app()
    with TestClient(app) as client:
        response = client.get(f"/v1/external/documents/full?origin_uri={snapshot_uri}")
        assert response.status_code == 503, response.text
        body = response.json()
        assert body["error"] == "warmup_pending"
        assert body["detail"] == "warmup_pending"
        assert body["message"] == "twinr_snapshot_scope_warmup_pending"
        assert body["status"] == 503
        assert body["instance"] == "/v1/external/documents/full"

    assert len(fake.calls["full_document"]) == 8
    assert all(
        call.get("origin_uri") in {pointer_uri, snapshot_uri}
        for call in fake.calls["full_document"]
    )


def test_scope_warmup_pending_http_exception_carries_snapshot_contract(
    monkeypatch,
) -> None:
    _disable_startup_warmups(monkeypatch)

    request = SimpleNamespace()
    snapshot_uri = external_router._snapshot_origin_uri(
        namespace="acme",
        snapshot_kind="conflicts",
    )
    last_exc = external_router.HTTPException(
        status_code=404,
        detail={
            "success": False,
            "error": "document_not_found",
            "error_type": "not_found",
        },
    )

    exc = external_router._scope_warmup_pending_http_exception(
        request=request,
        namespace="acme",
        snapshot_kind="conflicts",
        origin_uri=snapshot_uri,
        attempts=external_router._TWINR_SCOPE_WARMUP_RETRY_ATTEMPTS,
        last_exc=last_exc,
        pointer_head_available=False,
        snapshot_head_available=False,
    )

    assert exc.status_code == 503
    detail = dict(exc.detail or {})
    assert detail["error"] == "warmup_pending"
    assert detail["error_type"] == "warmup_pending"
    assert detail["namespace"] == "acme"
    assert detail["snapshot_kind"] == "conflicts"
    assert detail["origin_uri"] == snapshot_uri
    assert detail["attempts"] == external_router._TWINR_SCOPE_WARMUP_RETRY_ATTEMPTS
    assert detail["pointer_head_available"] is False
    assert detail["snapshot_head_available"] is False
    assert detail["retryable_error"] == "document_not_found"


def test_external_get_full_document_preserves_404_outside_snapshot_warmup(
    monkeypatch,
) -> None:
    _disable_startup_warmups(monkeypatch)
    _clear_external_router_caches()

    snapshot_uri = external_router._snapshot_origin_uri(
        namespace="acme",
        snapshot_kind="conflicts",
    )
    pointer_uri = external_router._pointer_origin_uri(
        namespace="acme",
        snapshot_kind="conflicts",
    )

    class _PermanentMissingService(_FacadeFakeService):
        async def get_full_document(self, **kwargs):  # noqa: ANN003
            self.calls["full_document"].append(dict(kwargs))
            if kwargs.get("origin_uri") in {pointer_uri, snapshot_uri}:
                return {
                    "success": False,
                    "error": "document_not_found",
                    "error_type": "not_found",
                }
            raise AssertionError(f"unexpected full_document kwargs: {kwargs}")

    fake = _PermanentMissingService()
    monkeypatch.setattr(indexes_router, "svc", fake)
    monkeypatch.setattr(external_router, "_get_svc", lambda _request=None: fake)

    app = create_app()
    with TestClient(app) as client:
        response = client.get(f"/v1/external/documents/full?origin_uri={snapshot_uri}")
        assert response.status_code == 404, response.text
        body = response.json()
        assert body["error"] == "document_not_found"
        assert body["status"] == 404

    assert len(fake.calls["full_document"]) == 2
    assert [call.get("origin_uri") for call in fake.calls["full_document"]] == [
        pointer_uri,
        snapshot_uri,
    ]


def test_external_get_full_document_reads_through_twinr_cache(monkeypatch) -> None:
    _disable_startup_warmups(monkeypatch)
    _clear_external_router_caches()

    snapshot_uri = external_router._snapshot_origin_uri(
        namespace="acme",
        snapshot_kind="objects",
    )

    class _ReadThroughService(_FacadeFakeService):
        async def get_full_document(self, **kwargs):  # noqa: ANN003
            self.calls["full_document"].append(dict(kwargs))
            return {
                "success": True,
                "document_id": "snapshot-doc-rt",
                "origin_uri": snapshot_uri,
                "chunk_count": 1,
                "chunks": [
                    {
                        "payload_id": "snapshot-doc-rt",
                        "chonky_id": "123",
                        "metadata": {"origin_uri": snapshot_uri},
                        "content": json.dumps(
                            {
                                "schema": "twinr_remote_snapshot_v1",
                                "namespace": "acme",
                                "snapshot_kind": "objects",
                                "updated_at": "2026-03-20T17:10:00Z",
                                "body": {
                                    "schema": "twinr_memory_object_catalog_v2",
                                    "version": 2,
                                    "items": [{"document_id": "doc-a"}],
                                },
                            }
                        ),
                    }
                ],
            }

    fake = _ReadThroughService()
    monkeypatch.setattr(indexes_router, "svc", fake)
    monkeypatch.setattr(external_router, "_get_svc", lambda _request=None: fake)

    app = create_app()
    with TestClient(app) as client:
        first = client.get(
            "/v1/external/documents/full?document_id=snapshot-doc-rt"
        )
        assert first.status_code == 200, first.text
        assert first.json()["chunk_count"] == 1
        second = client.get(
            "/v1/external/documents/full?document_id=snapshot-doc-rt"
        )
        assert second.status_code == 200, second.text
        assert second.json()["origin_uri"] == snapshot_uri
        assert json.loads(second.json()["content"]) == {
            "schema": "twinr_remote_snapshot_v1",
            "namespace": "acme",
            "snapshot_kind": "objects",
            "updated_at": "2026-03-20T17:10:00Z",
            "body": {
                "schema": "twinr_memory_object_catalog_v2",
                "version": 2,
                "items": [{"document_id": "doc-a"}],
            },
        }

    assert len(fake.calls["full_document"]) == 1


def test_external_get_full_document_cache_prefers_latest_chunk(monkeypatch) -> None:
    _disable_startup_warmups(monkeypatch)
    _clear_external_router_caches()

    snapshot_uri = external_router._snapshot_origin_uri(
        namespace="acme",
        snapshot_kind="objects",
    )
    older_snapshot_payload = {
        "schema": "twinr_remote_snapshot_v1",
        "namespace": "acme",
        "snapshot_kind": "objects",
        "updated_at": "2026-03-20T17:10:00Z",
        "body": {
            "schema": "twinr_memory_object_catalog_v2",
            "version": 2,
            "items": [{"document_id": "doc-old"}],
        },
    }
    newer_snapshot_payload = {
        "schema": "twinr_remote_snapshot_v1",
        "namespace": "acme",
        "snapshot_kind": "objects",
        "updated_at": "2026-03-20T17:11:00Z",
        "body": {
            "schema": "twinr_memory_object_catalog_v2",
            "version": 2,
            "items": [{"document_id": "doc-new"}],
        },
    }

    class _LatestChunkService(_FacadeFakeService):
        async def get_full_document(self, **kwargs):  # noqa: ANN003
            self.calls["full_document"].append(dict(kwargs))
            return {
                "success": True,
                "origin_uri": snapshot_uri,
                "chunk_count": 2,
                "chunks": [
                    {
                        "payload_id": "snapshot-doc-old",
                        "chonky_id": "101",
                        "metadata": {"origin_uri": snapshot_uri},
                        "content": json.dumps(older_snapshot_payload),
                    },
                    {
                        "payload_id": "snapshot-doc-new",
                        "chonky_id": "102",
                        "metadata": {"origin_uri": snapshot_uri},
                        "content": json.dumps(newer_snapshot_payload),
                    },
                ],
            }

    fake = _LatestChunkService()
    monkeypatch.setattr(indexes_router, "svc", fake)
    monkeypatch.setattr(external_router, "_get_svc", lambda _request=None: fake)

    app = create_app()
    with TestClient(app) as client:
        first = client.get(
            f"/v1/external/documents/full?origin_uri={snapshot_uri}"
        )
        assert first.status_code == 200, first.text
        second = client.get(
            f"/v1/external/documents/full?origin_uri={snapshot_uri}"
        )
        assert second.status_code == 200, second.text

    assert len(fake.calls["full_document"]) == 2
    assert fake.calls["full_document"][0]["origin_uri"] == external_router._pointer_origin_uri(
        namespace="acme",
        snapshot_kind="objects",
    )
    assert fake.calls["full_document"][1]["origin_uri"] == snapshot_uri
    assert second.json()["document_id"] == "snapshot-doc-new"
    assert second.json()["origin_uri"] == snapshot_uri
    assert json.loads(second.json()["content"]) == newer_snapshot_payload


def test_external_write_prime_and_delete_clear_twinr_document_cache(monkeypatch) -> None:
    _disable_startup_warmups(monkeypatch)
    _clear_external_router_caches()

    snapshot_uri = external_router._snapshot_origin_uri(
        namespace="acme",
        snapshot_kind="objects",
    )
    snapshot_payload = {
        "schema": "twinr_remote_snapshot_v1",
        "namespace": "acme",
        "snapshot_kind": "objects",
        "updated_at": "2026-03-20T17:20:00Z",
        "body": {
            "schema": "twinr_memory_object_catalog_v2",
            "version": 2,
            "items": [{"document_id": "doc-a"}],
        },
    }

    class _WritePrimeService(_FacadeFakeService):
        async def store_payloads_sync_bulk(self, **kwargs):  # noqa: ANN003
            self.calls["sync_bulk"].append(dict(kwargs))
            return {
                "success": True,
                "all_succeeded": True,
                "count": 1,
                "succeeded": 1,
                "failed": 0,
                "items": [
                    {
                        "success": True,
                        "payload_id": "payload-0",
                        "document_id": "doc-0",
                    }
                ],
            }

        async def get_full_document(self, **kwargs):  # noqa: ANN003
            self.calls["full_document"].append(dict(kwargs))
            return {
                "success": True,
                "document_id": "doc-0",
                "origin_uri": snapshot_uri,
                "content": json.dumps(snapshot_payload),
            }

    fake = _WritePrimeService()
    monkeypatch.setattr(indexes_router, "svc", fake)
    monkeypatch.setattr(external_router, "_get_svc", lambda _request=None: fake)

    app = create_app()
    with TestClient(app) as client:
        create_response = client.post(
            "/v1/external/records",
            json={
                "uri": snapshot_uri,
                "payload": snapshot_payload,
                "metadata": {"origin_uri": snapshot_uri},
            },
        )
        assert create_response.status_code == 200, create_response.text
        refresh_call_count = len(fake.calls["full_document"])

        cached_response = client.get(
            f"/v1/external/documents/full?origin_uri={snapshot_uri}"
        )
        assert cached_response.status_code == 200, cached_response.text
        assert len(fake.calls["full_document"]) == refresh_call_count

        delete_response = client.delete("/v1/external/records/payload-0")
        assert delete_response.status_code == 200, delete_response.text

        uncached_response = client.get(
            "/v1/external/documents/full?document_id=doc-0"
        )
        assert uncached_response.status_code == 200, uncached_response.text
        assert len(fake.calls["full_document"]) == refresh_call_count + 1


def test_external_async_write_primes_twinr_document_cache(monkeypatch) -> None:
    _disable_startup_warmups(monkeypatch)
    _clear_external_router_caches()

    snapshot_uri = external_router._snapshot_origin_uri(
        namespace="acme",
        snapshot_kind="objects",
    )
    snapshot_payload = {
        "schema": "twinr_remote_snapshot_v1",
        "namespace": "acme",
        "snapshot_kind": "objects",
        "updated_at": "2026-03-27T15:31:00Z",
        "body": {
            "schema": "twinr_memory_object_catalog_v2",
            "version": 2,
            "items": [{"document_id": "doc-a"}],
        },
    }

    class _AsyncWritePrimeService(_FacadeFakeService):
        async def get_full_document(self, **kwargs):  # noqa: ANN003
            self.calls["full_document"].append(dict(kwargs))
            return {
                "success": True,
                "document_id": "doc-async-1",
                "origin_uri": snapshot_uri,
                "content": json.dumps(snapshot_payload),
            }

    fake = _AsyncWritePrimeService()
    monkeypatch.setattr(indexes_router, "svc", fake)
    monkeypatch.setattr(external_router, "_get_svc", lambda _request=None: fake)

    app = create_app()
    with TestClient(app) as client:
        create_response = client.post(
            "/v1/external/records",
            json={
                "execution_mode": "async",
                "uri": snapshot_uri,
                "payload": snapshot_payload,
                "metadata": {"origin_uri": snapshot_uri},
            },
        )
        assert create_response.status_code == 202, create_response.text
        assert create_response.json()["job_id"] == "job-single-1"

        cached_response = client.get(
            f"/v1/external/documents/full?origin_uri={snapshot_uri}"
        )
        assert cached_response.status_code == 200, cached_response.text
        assert cached_response.json()["origin_uri"] == snapshot_uri
        assert json.loads(cached_response.json()["content"]) == snapshot_payload
        assert len(fake.calls["full_document"]) == 0


def test_external_async_bulk_write_primes_twinr_document_cache(monkeypatch) -> None:
    _disable_startup_warmups(monkeypatch)
    _clear_external_router_caches()

    snapshot_uri = external_router._snapshot_origin_uri(
        namespace="acme-bulk",
        snapshot_kind="objects",
    )
    snapshot_payload = {
        "schema": "twinr_remote_snapshot_v1",
        "namespace": "acme-bulk",
        "snapshot_kind": "objects",
        "updated_at": "2026-03-27T15:32:00Z",
        "body": {
            "schema": "twinr_memory_object_catalog_v2",
            "version": 2,
            "items": [{"document_id": "doc-b"}],
        },
    }

    class _AsyncBulkWritePrimeService(_FacadeFakeService):
        async def get_full_document(self, **kwargs):  # noqa: ANN003
            self.calls["full_document"].append(dict(kwargs))
            return {
                "success": True,
                "document_id": "doc-async-bulk-1",
                "origin_uri": snapshot_uri,
                "content": json.dumps(snapshot_payload),
            }

    fake = _AsyncBulkWritePrimeService()
    monkeypatch.setattr(indexes_router, "svc", fake)
    monkeypatch.setattr(external_router, "_get_svc", lambda _request=None: fake)

    app = create_app()
    with TestClient(app) as client:
        create_response = client.post(
            "/v1/external/records/bulk",
            json={
                "execution_mode": "async",
                "items": [
                    {
                        "uri": snapshot_uri,
                        "payload": snapshot_payload,
                        "metadata": {"origin_uri": snapshot_uri},
                    }
                ],
            },
        )
        assert create_response.status_code == 202, create_response.text
        assert create_response.json()["job_id"] == "job-bulk-1"

        cached_response = client.get(
            f"/v1/external/documents/full?origin_uri={snapshot_uri}"
        )
        assert cached_response.status_code == 200, cached_response.text
        assert cached_response.json()["origin_uri"] == snapshot_uri
        assert json.loads(cached_response.json()["content"]) == snapshot_payload
        assert len(fake.calls["full_document"]) == 0


def test_external_admin_clear_all_clears_twinr_document_cache(monkeypatch) -> None:
    _disable_startup_warmups(monkeypatch)
    _clear_external_router_caches()

    snapshot_uri = external_router._snapshot_origin_uri(
        namespace="acme",
        snapshot_kind="objects",
    )
    pointer_uri = external_router._pointer_origin_uri(
        namespace="acme",
        snapshot_kind="objects",
    )
    snapshot_payload = {
        "schema": "twinr_remote_snapshot_v1",
        "namespace": "acme",
        "snapshot_kind": "objects",
        "updated_at": "2026-03-20T17:30:00Z",
        "body": {
            "schema": "twinr_memory_object_catalog_v2",
            "version": 2,
            "items": [{"document_id": "doc-a"}],
        },
    }

    class _ClearAllService(_FacadeFakeService):
        async def get_full_document(self, **kwargs):  # noqa: ANN003
            self.calls["full_document"].append(dict(kwargs))
            return {
                "success": True,
                "document_id": "doc-clear-1",
                "origin_uri": snapshot_uri,
                "content": json.dumps(snapshot_payload),
            }

    fake = _ClearAllService()
    monkeypatch.setattr(indexes_router, "svc", fake)
    monkeypatch.setattr(external_router, "_get_svc", lambda _request=None: fake)

    app = create_app()
    with TestClient(app) as client:
        first = client.get(
            f"/v1/external/documents/full?origin_uri={snapshot_uri}"
        )
        assert first.status_code == 200, first.text
        assert len(fake.calls["full_document"]) == 2
        assert fake.calls["full_document"][0]["origin_uri"] == pointer_uri
        assert fake.calls["full_document"][1]["origin_uri"] == snapshot_uri

        clear_response = client.delete("/v1/external/admin/data?confirm=true")
        assert clear_response.status_code == 200, clear_response.text

        second = client.get(
            f"/v1/external/documents/full?origin_uri={snapshot_uri}"
        )
        assert second.status_code == 200, second.text
        assert len(fake.calls["full_document"]) == 4
        assert fake.calls["full_document"][2]["origin_uri"] == pointer_uri
        assert fake.calls["full_document"][3]["origin_uri"] == snapshot_uri
