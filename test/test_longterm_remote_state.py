from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from io import BytesIO
import json
import logging
from pathlib import Path
import socket
from threading import Event
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import cast
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test.longterm_test_program import make_test_extractor
from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.chonkydb import ChonkyDBClient, ChonkyDBConnectionConfig
from twinr.memory.context_store import PersistentMemoryMarkdownStore
from twinr.memory.longterm.core.cooperative_abort import (
    LongTermOperationCancelledError,
    longterm_operation_abort_scope,
)
from twinr.memory.longterm.storage.remote_state import (
    LongTermRemoteReadFailedError,
    LongTermRemoteSnapshotProbe,
    LongTermRemoteStatus,
    LongTermRemoteStateStore,
    LongTermRemoteUnavailableError,
)
from twinr.memory.longterm.runtime.service import LongTermMemoryService
from twinr.ops.events import TwinrOpsEventStore


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class FakeOpener:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.responses: list[object] = []

    def queue_json(self, payload: dict[str, object]) -> None:
        self.responses.append(FakeHTTPResponse(payload))

    def queue_http_error(
        self,
        status_code: int,
        payload: dict[str, object],
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.responses.append(
            HTTPError(
                url="https://memory.test/fail",
                code=status_code,
                msg="bad request",
                hdrs=headers,
                fp=BytesIO(json.dumps(payload).encode("utf-8")),
            )
        )

    def queue_exception(self, exc: Exception) -> None:
        self.responses.append(exc)

    def __call__(self, request, timeout: float):
        self.calls.append(
            {
                "full_url": request.full_url,
                "method": request.get_method(),
                "headers": dict(request.header_items()),
                "body": request.data.decode("utf-8") if request.data else None,
                "timeout": timeout,
            }
        )
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class _CacheAwareFakeRemoteState:
    def __init__(self, payloads: dict[str, dict[str, object]]) -> None:
        self.enabled = True
        self.required = True
        self._payloads = {key: dict(value) for key, value in payloads.items()}
        self.load_calls: list[str] = []
        self.probe_calls: list[dict[str, object]] = []
        self.status_calls = 0
        self._cache_depth = 0
        self._cache: dict[str, dict[str, object]] = {}

    @contextmanager
    def cache_probe_reads(self):
        self._cache_depth += 1
        if self._cache_depth == 1:
            self._cache.clear()
        try:
            yield
        finally:
            self._cache_depth -= 1
            if self._cache_depth == 0:
                self._cache.clear()

    def status(self):
        self.status_calls += 1
        return SimpleNamespace(mode="remote_primary", ready=True, detail=None)

    def load_snapshot(self, *, snapshot_kind: str, local_path: Path | None = None):
        del local_path
        cached = self._cache.get(snapshot_kind)
        if cached is not None:
            return dict(cached)
        self.load_calls.append(snapshot_kind)
        payload = dict(self._payloads[snapshot_kind])
        if self._cache_depth > 0:
            self._cache[snapshot_kind] = dict(payload)
        return payload

    def probe_snapshot_load(
        self,
        *,
        snapshot_kind: str,
        local_path: Path | None = None,
        prefer_cached_document_id: bool = False,
        prefer_metadata_only: bool = False,
        fast_fail: bool = False,
    ):
        del local_path, fast_fail
        cached = self._cache.get(snapshot_kind)
        if cached is not None:
            return LongTermRemoteSnapshotProbe(
                snapshot_kind=snapshot_kind,
                status="found",
                latency_ms=0.0,
                selected_source="pointer_document",
                payload=dict(cached),
            )
        self.probe_calls.append(
            {
                "snapshot_kind": snapshot_kind,
                "prefer_cached_document_id": prefer_cached_document_id,
                "prefer_metadata_only": prefer_metadata_only,
            }
        )
        payload = dict(self._payloads[snapshot_kind])
        if self._cache_depth > 0:
            self._cache[snapshot_kind] = dict(payload)
        return LongTermRemoteSnapshotProbe(
            snapshot_kind=snapshot_kind,
            status="found",
            latency_ms=1.0,
            selected_source="pointer_document",
            payload=payload,
        )


class _EnsureSnapshotStore:
    def __init__(self, remote_state: _CacheAwareFakeRemoteState, snapshot_kind: str) -> None:
        self.remote_state = remote_state
        self.remote_snapshot_kind = snapshot_kind

    def ensure_remote_snapshot(self) -> bool:
        self.remote_state.load_snapshot(snapshot_kind=self.remote_snapshot_kind)
        return False


class _PromptContextStoreDouble:
    def __init__(self, remote_state: _CacheAwareFakeRemoteState) -> None:
        self.memory_store = _EnsureSnapshotStore(remote_state, "prompt_memory")
        self.user_store = _EnsureSnapshotStore(remote_state, "user_context")
        self.personality_store = _EnsureSnapshotStore(remote_state, "personality_context")

    def ensure_remote_snapshots(self) -> tuple[str, ...]:
        self.memory_store.ensure_remote_snapshot()
        self.user_store.ensure_remote_snapshot()
        self.personality_store.ensure_remote_snapshot()
        return ()


class _GraphStoreDouble:
    def __init__(self, remote_state: _CacheAwareFakeRemoteState) -> None:
        self.remote_state = remote_state

    def ensure_remote_snapshot(self) -> bool:
        self.remote_state.load_snapshot(snapshot_kind="graph")
        return False


class _ObjectStoreDouble:
    def __init__(self, remote_state: _CacheAwareFakeRemoteState) -> None:
        self.remote_state = remote_state
        self.fast_topic_calls: list[dict[str, object]] = []

    def ensure_remote_snapshots(self) -> tuple[str, ...]:
        self.remote_state.load_snapshot(snapshot_kind="objects")
        self.remote_state.load_snapshot(snapshot_kind="conflicts")
        self.remote_state.load_snapshot(snapshot_kind="archive")
        return ()

    def select_fast_topic_objects(
        self,
        *,
        query_text: str | None = None,
        limit: int = 3,
        timeout_s: float | None = None,
    ):
        self.fast_topic_calls.append(
            {
                "query_text": query_text,
                "limit": limit,
                "timeout_s": timeout_s,
            }
        )
        return ()


class _MidtermStoreDouble:
    def __init__(self, remote_state: _CacheAwareFakeRemoteState) -> None:
        self.remote_state = remote_state

    def ensure_remote_snapshot(self) -> bool:
        self.remote_state.load_snapshot(snapshot_kind="midterm")
        return False


class LongTermRemoteStateStoreTests(unittest.TestCase):
    def _config(self, root: str) -> TwinrConfig:
        return TwinrConfig(
            project_root=root,
            long_term_memory_enabled=True,
            long_term_memory_mode="remote_primary",
            long_term_memory_remote_required=True,
            long_term_memory_migration_enabled=True,
            long_term_memory_remote_namespace="test-namespace",
            chonkydb_base_url="https://memory.test",
            chonkydb_api_key="secret-key",
        )

    def test_probe_snapshot_load_reuses_cached_result_within_explicit_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(temp_dir)
            read_opener = FakeOpener()
            for document_id, memory_id in (
                ("doc-123", "fact:first"),
                ("doc-456", "fact:second"),
            ):
                read_opener.queue_json(
                    {
                        "success": True,
                        "document_id": f"pointer-{document_id}",
                        "origin_uri": "twinr://longterm/test-namespace/__pointer__:objects",
                        "content": json.dumps(
                            {
                                "schema": "twinr_remote_snapshot_v1",
                                "namespace": "test-namespace",
                                "snapshot_kind": "__pointer__:objects",
                                "updated_at": "2026-03-18T11:00:00+00:00",
                                "body": {
                                    "schema": "twinr_remote_snapshot_pointer_v1",
                                    "version": 1,
                                    "snapshot_kind": "objects",
                                    "document_id": document_id,
                                },
                            }
                        ),
                    }
                )
                read_opener.queue_json(
                    {
                        "success": True,
                        "document_id": document_id,
                        "content": json.dumps(
                            {
                                "schema": "twinr_remote_snapshot_v1",
                                "namespace": "test-namespace",
                                "snapshot_kind": "objects",
                                "updated_at": "2026-03-18T11:00:01+00:00",
                                "body": {"schema": "object_store", "objects": [{"memory_id": memory_id}]},
                            }
                        ),
                    }
                )
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=FakeOpener(),
                ),
            )

            with state.cache_probe_reads():
                first = state.probe_snapshot_load(snapshot_kind="objects", local_path=Path(temp_dir) / "objects.json")
                cached = state.probe_snapshot_load(snapshot_kind="objects")
            fresh = state.probe_snapshot_load(snapshot_kind="objects")

        self.assertEqual(len(read_opener.calls), 4)
        self.assertEqual(first.payload, {"schema": "object_store", "objects": [{"memory_id": "fact:first"}]})
        self.assertEqual(cached.payload, first.payload)
        self.assertEqual(cached.latency_ms, 0.0)
        self.assertEqual(fresh.payload, {"schema": "object_store", "objects": [{"memory_id": "fact:second"}]})

    def test_save_snapshot_seeds_cached_probe_within_explicit_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(temp_dir)
            read_opener = FakeOpener()
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "doc-123",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "objects",
                            "updated_at": "2026-03-19T13:00:00+00:00",
                            "body": {"schema": "object_store", "objects": []},
                        }
                    ),
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "pointer-doc-123",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "__pointer__:objects",
                            "updated_at": "2026-03-19T13:00:01+00:00",
                            "body": {
                                "schema": "twinr_remote_snapshot_pointer_v1",
                                "version": 1,
                                "snapshot_kind": "objects",
                                "document_id": "doc-123",
                            },
                        }
                    ),
                }
            )
            write_opener = FakeOpener()
            write_opener.queue_json(
                {
                    "success": True,
                    "items": [
                        {
                            "success": True,
                            "document_id": "doc-123",
                            "payload_id": "doc-123",
                        }
                    ],
                }
            )
            write_opener.queue_json({"success": True, "stored": 1})
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=write_opener,
                ),
            )

            with state.cache_probe_reads():
                state.save_snapshot(snapshot_kind="objects", payload={"schema": "object_store", "objects": []})
                cached = state.probe_snapshot_load(snapshot_kind="objects")

        self.assertEqual(len(read_opener.calls), 2)
        self.assertEqual(cached.payload, {"schema": "object_store", "objects": []})
        self.assertEqual(cached.document_id, "doc-123")
        self.assertEqual(cached.selected_source, "saved_document")
        self.assertEqual(cached.latency_ms, 0.0)

    def test_save_snapshot_keeps_other_cached_probes_within_explicit_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(temp_dir)
            read_opener = FakeOpener()
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "doc-objects",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "objects",
                            "updated_at": "2026-03-19T13:00:00+00:00",
                            "body": {"schema": "object_store", "objects": []},
                        }
                    ),
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "pointer-objects",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "__pointer__:objects",
                            "updated_at": "2026-03-19T13:00:00+00:00",
                            "body": {
                                "schema": "twinr_remote_snapshot_pointer_v1",
                                "version": 1,
                                "snapshot_kind": "objects",
                                "document_id": "doc-objects",
                            },
                        }
                    ),
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "doc-user",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "user_context",
                            "updated_at": "2026-03-19T13:00:01+00:00",
                            "body": {"schema": "managed_context", "entries": []},
                        }
                    ),
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "pointer-user",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "__pointer__:user_context",
                            "updated_at": "2026-03-19T13:00:01+00:00",
                            "body": {
                                "schema": "twinr_remote_snapshot_pointer_v1",
                                "version": 1,
                                "snapshot_kind": "user_context",
                                "document_id": "doc-user",
                            },
                        }
                    ),
                }
            )
            write_opener = FakeOpener()
            for document_id in ("doc-objects", "doc-pointer-objects", "doc-user", "doc-pointer-user"):
                write_opener.queue_json(
                    {
                        "success": True,
                        "items": [
                            {
                                "success": True,
                                "document_id": document_id,
                                "payload_id": document_id,
                            }
                        ],
                    }
                )
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=write_opener,
                ),
            )

            with state.cache_probe_reads():
                state.save_snapshot(snapshot_kind="objects", payload={"schema": "object_store", "objects": []})
                state.save_snapshot(snapshot_kind="user_context", payload={"schema": "managed_context", "entries": []})
                cached_objects = state.probe_snapshot_load(snapshot_kind="objects")
                cached_user_context = state.probe_snapshot_load(snapshot_kind="user_context")

        self.assertEqual(len(read_opener.calls), 4)
        self.assertEqual(cached_objects.payload, {"schema": "object_store", "objects": []})
        self.assertEqual(cached_objects.document_id, "doc-objects")
        self.assertEqual(cached_objects.selected_source, "saved_document")
        self.assertEqual(cached_user_context.payload, {"schema": "managed_context", "entries": []})
        self.assertEqual(cached_user_context.document_id, "doc-user")
        self.assertEqual(cached_user_context.selected_source, "saved_document")

    def test_probe_remote_ready_reuses_snapshot_reads_within_one_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            prompt_state = _CacheAwareFakeRemoteState(
                {
                    "prompt_memory": {"schema": "prompt_memory", "entries": []},
                    "user_context": {"schema": "managed_context", "entries": []},
                    "personality_context": {"schema": "managed_context", "entries": []},
                }
            )
            object_state = _CacheAwareFakeRemoteState(
                {
                    "objects": {"schema": "twinr_memory_object_catalog_v2", "version": 2, "items": []},
                    "conflicts": {"schema": "conflicts", "conflicts": []},
                    "archive": {"schema": "twinr_memory_archive_catalog_v2", "version": 2, "items": []},
                }
            )
            graph_state = _CacheAwareFakeRemoteState({"graph": {"schema": "graph", "nodes": [], "edges": []}})
            midterm_state = _CacheAwareFakeRemoteState({"midterm": {"schema": "midterm", "packets": []}})
            service = LongTermMemoryService(
                config=config,
                prompt_context_store=_PromptContextStoreDouble(prompt_state),
                graph_store=_GraphStoreDouble(graph_state),
                object_store=_ObjectStoreDouble(object_state),
                midterm_store=_MidtermStoreDouble(midterm_state),
                query_rewriter=SimpleNamespace(),
                retriever=SimpleNamespace(),
                extractor=SimpleNamespace(),
                multimodal_extractor=SimpleNamespace(),
                truth_maintainer=SimpleNamespace(),
                consolidator=SimpleNamespace(),
                conflict_resolver=SimpleNamespace(),
                reflector=SimpleNamespace(),
                sensor_memory=SimpleNamespace(),
                ops_backfiller=SimpleNamespace(),
                planner=SimpleNamespace(),
                proactive_policy=SimpleNamespace(),
                retention_policy=SimpleNamespace(),
            )

            result = service.probe_remote_ready()

        self.assertTrue(result.ready)
        self.assertEqual(prompt_state.load_calls, ["prompt_memory", "user_context", "personality_context"])
        self.assertEqual(prompt_state.probe_calls, [])
        self.assertEqual(object_state.load_calls, ["objects", "conflicts", "archive"])
        self.assertEqual(object_state.probe_calls, [])
        self.assertEqual(graph_state.load_calls, ["graph"])
        self.assertEqual(graph_state.probe_calls, [])
        self.assertEqual(midterm_state.load_calls, ["midterm"])
        self.assertEqual(midterm_state.probe_calls, [])

    def test_probe_remote_ready_skips_redundant_per_store_status_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            prompt_state = _CacheAwareFakeRemoteState(
                {
                    "prompt_memory": {"schema": "prompt_memory", "entries": []},
                    "user_context": {"schema": "managed_context", "entries": []},
                    "personality_context": {"schema": "managed_context", "entries": []},
                }
            )
            object_state = _CacheAwareFakeRemoteState(
                {
                    "objects": {"schema": "twinr_memory_object_catalog_v2", "version": 2, "items": []},
                    "conflicts": {"schema": "conflicts", "conflicts": []},
                    "archive": {"schema": "twinr_memory_archive_catalog_v2", "version": 2, "items": []},
                }
            )
            graph_state = _CacheAwareFakeRemoteState({"graph": {"schema": "graph", "nodes": [], "edges": []}})
            midterm_state = _CacheAwareFakeRemoteState({"midterm": {"schema": "midterm", "packets": []}})
            service = LongTermMemoryService(
                config=config,
                prompt_context_store=_PromptContextStoreDouble(prompt_state),
                graph_store=_GraphStoreDouble(graph_state),
                object_store=_ObjectStoreDouble(object_state),
                midterm_store=_MidtermStoreDouble(midterm_state),
                query_rewriter=SimpleNamespace(),
                retriever=SimpleNamespace(),
                extractor=SimpleNamespace(),
                multimodal_extractor=SimpleNamespace(),
                truth_maintainer=SimpleNamespace(),
                consolidator=SimpleNamespace(),
                conflict_resolver=SimpleNamespace(),
                reflector=SimpleNamespace(),
                sensor_memory=SimpleNamespace(),
                ops_backfiller=SimpleNamespace(),
                planner=SimpleNamespace(),
                proactive_policy=SimpleNamespace(),
                retention_policy=SimpleNamespace(),
            )

            result = service.probe_remote_ready()

        self.assertTrue(result.ready)
        self.assertEqual(prompt_state.status_calls, 1)
        self.assertEqual(object_state.status_calls, 0)
        self.assertEqual(graph_state.status_calls, 0)
        self.assertEqual(midterm_state.status_calls, 0)

    def test_probe_remote_ready_accepts_archive_safe_deep_readiness_when_instance_flag_stays_false(self) -> None:
        class _InstanceFlagFalseRemoteState(_CacheAwareFakeRemoteState):
            def status(self):
                self.status_calls += 1
                return LongTermRemoteStatus(
                    mode="remote_primary",
                    ready=False,
                    detail="ChonkyDB instance responded but is not ready.",
                    operational_probe_allowed=True,
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            prompt_state = _InstanceFlagFalseRemoteState(
                {
                    "prompt_memory": {"schema": "prompt_memory", "entries": []},
                    "user_context": {"schema": "managed_context", "entries": []},
                    "personality_context": {"schema": "managed_context", "entries": []},
                }
            )
            object_state = _CacheAwareFakeRemoteState(
                {
                    "objects": {"schema": "twinr_memory_object_catalog_v2", "version": 2, "items": []},
                    "conflicts": {"schema": "conflicts", "conflicts": []},
                    "archive": {"schema": "twinr_memory_archive_catalog_v2", "version": 2, "items": []},
                }
            )
            graph_state = _CacheAwareFakeRemoteState({"graph": {"schema": "graph", "nodes": [], "edges": []}})
            midterm_state = _CacheAwareFakeRemoteState({"midterm": {"schema": "midterm", "packets": []}})
            service = LongTermMemoryService(
                config=config,
                prompt_context_store=_PromptContextStoreDouble(prompt_state),
                graph_store=_GraphStoreDouble(graph_state),
                object_store=_ObjectStoreDouble(object_state),
                midterm_store=_MidtermStoreDouble(midterm_state),
                query_rewriter=SimpleNamespace(),
                retriever=SimpleNamespace(),
                extractor=SimpleNamespace(),
                multimodal_extractor=SimpleNamespace(),
                truth_maintainer=SimpleNamespace(),
                consolidator=SimpleNamespace(),
                conflict_resolver=SimpleNamespace(),
                reflector=SimpleNamespace(),
                sensor_memory=SimpleNamespace(),
                ops_backfiller=SimpleNamespace(),
                planner=SimpleNamespace(),
                proactive_policy=SimpleNamespace(),
                retention_policy=SimpleNamespace(),
            )

            result = service.probe_remote_ready()

        self.assertTrue(result.ready)
        self.assertFalse(result.remote_status.ready)
        self.assertTrue(result.remote_status.operational_probe_allowed)
        self.assertEqual(result.steps[0].name, "remote_status")
        self.assertEqual(result.steps[0].status, "warn")
        self.assertIn("query_surface_ready_despite_instance_flag_false", result.detail or "")
        self.assertEqual(prompt_state.status_calls, 1)
        self.assertEqual(prompt_state.load_calls, ["prompt_memory", "user_context", "personality_context"])

    def test_probe_remote_ready_accepts_current_only_deep_readiness_when_instance_flag_stays_false(self) -> None:
        class _InstanceFlagFalseRemoteState(_CacheAwareFakeRemoteState):
            def status(self):
                self.status_calls += 1
                return LongTermRemoteStatus(
                    mode="remote_primary",
                    ready=False,
                    detail="ChonkyDB instance responded but is not ready.",
                    operational_probe_allowed=True,
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            prompt_state = _InstanceFlagFalseRemoteState(
                {
                    "prompt_memory": {"schema": "prompt_memory", "entries": []},
                    "user_context": {"schema": "managed_context", "entries": []},
                    "personality_context": {"schema": "managed_context", "entries": []},
                }
            )
            object_state = _CacheAwareFakeRemoteState(
                {
                    "objects": {"schema": "twinr_memory_object_catalog_v2", "version": 2, "items": []},
                    "conflicts": {"schema": "conflicts", "conflicts": []},
                    "archive": {"schema": "twinr_memory_archive_catalog_v2", "version": 2, "items": []},
                }
            )
            graph_state = _CacheAwareFakeRemoteState({"graph": {"schema": "graph", "nodes": [], "edges": []}})
            midterm_state = _CacheAwareFakeRemoteState({"midterm": {"schema": "midterm", "packets": []}})
            service = LongTermMemoryService(
                config=config,
                prompt_context_store=_PromptContextStoreDouble(prompt_state),
                graph_store=_GraphStoreDouble(graph_state),
                object_store=_ObjectStoreDouble(object_state),
                midterm_store=_MidtermStoreDouble(midterm_state),
                query_rewriter=SimpleNamespace(),
                retriever=SimpleNamespace(),
                extractor=SimpleNamespace(),
                multimodal_extractor=SimpleNamespace(),
                truth_maintainer=SimpleNamespace(),
                consolidator=SimpleNamespace(),
                conflict_resolver=SimpleNamespace(),
                reflector=SimpleNamespace(),
                sensor_memory=SimpleNamespace(),
                ops_backfiller=SimpleNamespace(),
                planner=SimpleNamespace(),
                proactive_policy=SimpleNamespace(),
                retention_policy=SimpleNamespace(),
            )

            result = service.probe_remote_ready(bootstrap=False, include_archive=False)

        self.assertTrue(result.ready)
        self.assertFalse(result.remote_status.ready)
        self.assertEqual(result.warm_result.health_tier, "degraded")
        self.assertIn("query_surface_ready_despite_instance_flag_false", result.detail or "")

    def test_probe_remote_ready_attests_external_remote_ready_when_deep_readiness_overrides_instance_flag(self) -> None:
        class _AttestingRemoteState(_CacheAwareFakeRemoteState):
            def __init__(
                self,
                payloads: dict[str, dict[str, object]],
                *,
                status_response: LongTermRemoteStatus | None = None,
            ) -> None:
                super().__init__(payloads)
                self.attest_count = 0
                self._status_response = status_response or LongTermRemoteStatus(
                    mode="remote_primary",
                    ready=True,
                    detail=None,
                )

            def status(self):
                self.status_calls += 1
                return self._status_response

            def attest_external_readiness(self) -> None:
                self.attest_count += 1

        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            prompt_state = _AttestingRemoteState(
                {
                    "prompt_memory": {"schema": "prompt_memory", "entries": []},
                    "user_context": {"schema": "managed_context", "entries": []},
                    "personality_context": {"schema": "managed_context", "entries": []},
                },
                status_response=LongTermRemoteStatus(
                    mode="remote_primary",
                    ready=False,
                    detail="Transient shallow probe miss.",
                    operational_probe_allowed=True,
                ),
            )
            object_state = _AttestingRemoteState(
                {
                    "objects": {"schema": "twinr_memory_object_catalog_v2", "version": 2, "items": []},
                    "conflicts": {"schema": "conflicts", "conflicts": []},
                    "archive": {"schema": "twinr_memory_archive_catalog_v2", "version": 2, "items": []},
                }
            )
            graph_state = _AttestingRemoteState({"graph": {"schema": "graph", "nodes": [], "edges": []}})
            midterm_state = _AttestingRemoteState({"midterm": {"schema": "midterm", "packets": []}})
            service = LongTermMemoryService(
                config=config,
                prompt_context_store=_PromptContextStoreDouble(prompt_state),
                graph_store=_GraphStoreDouble(graph_state),
                object_store=_ObjectStoreDouble(object_state),
                midterm_store=_MidtermStoreDouble(midterm_state),
                query_rewriter=SimpleNamespace(),
                retriever=SimpleNamespace(),
                extractor=SimpleNamespace(),
                multimodal_extractor=SimpleNamespace(),
                truth_maintainer=SimpleNamespace(),
                consolidator=SimpleNamespace(),
                conflict_resolver=SimpleNamespace(),
                reflector=SimpleNamespace(),
                sensor_memory=SimpleNamespace(),
                ops_backfiller=SimpleNamespace(),
                planner=SimpleNamespace(),
                proactive_policy=SimpleNamespace(),
                retention_policy=SimpleNamespace(),
            )

            result = service.probe_remote_ready()

        self.assertTrue(result.ready)
        self.assertIn("query_surface_ready_despite_instance_flag_false", result.detail or "")
        self.assertEqual(prompt_state.attest_count, 1)
        self.assertEqual(object_state.attest_count, 1)
        self.assertEqual(graph_state.attest_count, 1)
        self.assertEqual(midterm_state.attest_count, 1)

    def test_probe_remote_ready_external_attestation_probe_reopens_local_cooldown_for_recovery(self) -> None:
        class _CooldownRecoveringRemoteState(_CacheAwareFakeRemoteState):
            def __init__(self, payloads: dict[str, dict[str, object]]) -> None:
                super().__init__(payloads)
                self.attest_count = 0
                self._cooldown_open = True

            def status(self):
                self.status_calls += 1
                if self._cooldown_open:
                    return LongTermRemoteStatus(
                        mode="remote_primary",
                        ready=False,
                        detail="Remote long-term memory is temporarily cooling down after recent failures.",
                        operational_probe_allowed=False,
                    )
                return LongTermRemoteStatus(
                    mode="remote_primary",
                    ready=True,
                    detail=None,
                    operational_probe_allowed=True,
                )

            def attest_external_readiness(self) -> None:
                self.attest_count += 1
                self._cooldown_open = False
                self._cache.clear()

        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            prompt_state = _CooldownRecoveringRemoteState(
                {
                    "prompt_memory": {"schema": "prompt_memory", "entries": []},
                    "user_context": {"schema": "managed_context", "entries": []},
                    "personality_context": {"schema": "managed_context", "entries": []},
                }
            )
            object_state = _CacheAwareFakeRemoteState(
                {
                    "objects": {"schema": "twinr_memory_object_catalog_v2", "version": 2, "items": []},
                    "conflicts": {"schema": "conflicts", "conflicts": []},
                    "archive": {"schema": "twinr_memory_archive_catalog_v2", "version": 2, "items": []},
                }
            )
            graph_state = _CacheAwareFakeRemoteState({"graph": {"schema": "graph", "nodes": [], "edges": []}})
            midterm_state = _CacheAwareFakeRemoteState({"midterm": {"schema": "midterm", "packets": []}})
            service = LongTermMemoryService(
                config=config,
                prompt_context_store=_PromptContextStoreDouble(prompt_state),
                graph_store=_GraphStoreDouble(graph_state),
                object_store=_ObjectStoreDouble(object_state),
                midterm_store=_MidtermStoreDouble(midterm_state),
                query_rewriter=SimpleNamespace(),
                retriever=SimpleNamespace(),
                extractor=SimpleNamespace(),
                multimodal_extractor=SimpleNamespace(),
                truth_maintainer=SimpleNamespace(),
                consolidator=SimpleNamespace(),
                conflict_resolver=SimpleNamespace(),
                reflector=SimpleNamespace(),
                sensor_memory=SimpleNamespace(),
                ops_backfiller=SimpleNamespace(),
                planner=SimpleNamespace(),
                proactive_policy=SimpleNamespace(),
                retention_policy=SimpleNamespace(),
            )

            result = service.probe_remote_ready(external_attestation_probe=True)

        self.assertTrue(result.ready)
        self.assertEqual(prompt_state.status_calls, 2)
        self.assertEqual(prompt_state.attest_count, 2)
        self.assertEqual(result.steps[0].name, "remote_status")
        self.assertEqual(result.steps[0].status, "fail")
        self.assertEqual(result.steps[1].name, "remote_status_external_attestation_reset")
        self.assertEqual(result.steps[2].name, "remote_status_post_attestation")
        self.assertEqual(result.steps[2].status, "ok")

    def test_probe_remote_ready_steady_state_skips_bootstrap_and_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            prompt_state = _CacheAwareFakeRemoteState(
                {
                    "prompt_memory": {"schema": "prompt_memory", "entries": []},
                    "user_context": {"schema": "managed_context", "entries": []},
                    "personality_context": {"schema": "managed_context", "entries": []},
                }
            )
            object_state = _CacheAwareFakeRemoteState(
                {
                    "objects": {"schema": "twinr_memory_object_catalog_v2", "version": 2, "items": []},
                    "conflicts": {"schema": "conflicts", "conflicts": []},
                    "archive": {"schema": "twinr_memory_archive_catalog_v2", "version": 2, "items": []},
                }
            )
            graph_state = _CacheAwareFakeRemoteState({"graph": {"schema": "graph", "nodes": [], "edges": []}})
            midterm_state = _CacheAwareFakeRemoteState({"midterm": {"schema": "midterm", "packets": []}})
            service = LongTermMemoryService(
                config=config,
                prompt_context_store=_PromptContextStoreDouble(prompt_state),
                graph_store=_GraphStoreDouble(graph_state),
                object_store=_ObjectStoreDouble(object_state),
                midterm_store=_MidtermStoreDouble(midterm_state),
                query_rewriter=SimpleNamespace(),
                retriever=SimpleNamespace(),
                extractor=SimpleNamespace(),
                multimodal_extractor=SimpleNamespace(),
                truth_maintainer=SimpleNamespace(),
                consolidator=SimpleNamespace(),
                conflict_resolver=SimpleNamespace(),
                reflector=SimpleNamespace(),
                sensor_memory=SimpleNamespace(),
                ops_backfiller=SimpleNamespace(),
                planner=SimpleNamespace(),
                proactive_policy=SimpleNamespace(),
                retention_policy=SimpleNamespace(),
            )

            result = service.probe_remote_ready(bootstrap=False, include_archive=False)

        self.assertTrue(result.ready)
        self.assertEqual(prompt_state.load_calls, [])
        self.assertEqual(prompt_state.probe_calls, [
            {"snapshot_kind": "prompt_memory", "prefer_cached_document_id": True, "prefer_metadata_only": True},
            {"snapshot_kind": "user_context", "prefer_cached_document_id": True, "prefer_metadata_only": True},
            {"snapshot_kind": "personality_context", "prefer_cached_document_id": True, "prefer_metadata_only": True},
        ])
        self.assertEqual(object_state.load_calls, [])
        self.assertEqual(object_state.probe_calls, [
            {"snapshot_kind": "objects", "prefer_cached_document_id": True, "prefer_metadata_only": True},
            {"snapshot_kind": "conflicts", "prefer_cached_document_id": True, "prefer_metadata_only": True},
        ])
        self.assertEqual(graph_state.load_calls, [])
        self.assertEqual(graph_state.probe_calls, [
            {"snapshot_kind": "graph", "prefer_cached_document_id": True, "prefer_metadata_only": True},
        ])
        self.assertEqual(midterm_state.load_calls, [])
        self.assertEqual(midterm_state.probe_calls, [
            {"snapshot_kind": "midterm", "prefer_cached_document_id": True, "prefer_metadata_only": True},
        ])

    def test_probe_remote_ready_requires_fast_topic_route_when_enabled(self) -> None:
        class _FailingFastTopicObjectStore(_ObjectStoreDouble):
            def select_fast_topic_objects(
                self,
                *,
                query_text: str | None = None,
                limit: int = 3,
                timeout_s: float | None = None,
            ):
                super().select_fast_topic_objects(
                    query_text=query_text,
                    limit=limit,
                    timeout_s=timeout_s,
                )
                raise LongTermRemoteReadFailedError(
                    "Required remote long-term fast-topic retrieval failed."
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            prompt_state = _CacheAwareFakeRemoteState(
                {
                    "prompt_memory": {"schema": "prompt_memory", "entries": []},
                    "user_context": {"schema": "managed_context", "entries": []},
                    "personality_context": {"schema": "managed_context", "entries": []},
                }
            )
            object_state = _CacheAwareFakeRemoteState(
                {
                    "objects": {"schema": "twinr_memory_object_catalog_v2", "version": 2, "items": []},
                    "conflicts": {"schema": "conflicts", "conflicts": []},
                    "archive": {"schema": "twinr_memory_archive_catalog_v2", "version": 2, "items": []},
                }
            )
            graph_state = _CacheAwareFakeRemoteState({"graph": {"schema": "graph", "nodes": [], "edges": []}})
            midterm_state = _CacheAwareFakeRemoteState({"midterm": {"schema": "midterm", "packets": []}})
            object_store = _FailingFastTopicObjectStore(object_state)
            service = LongTermMemoryService(
                config=config,
                prompt_context_store=_PromptContextStoreDouble(prompt_state),
                graph_store=_GraphStoreDouble(graph_state),
                object_store=object_store,
                midterm_store=_MidtermStoreDouble(midterm_state),
                query_rewriter=SimpleNamespace(),
                retriever=SimpleNamespace(),
                extractor=SimpleNamespace(),
                multimodal_extractor=SimpleNamespace(),
                truth_maintainer=SimpleNamespace(),
                consolidator=SimpleNamespace(),
                conflict_resolver=SimpleNamespace(),
                reflector=SimpleNamespace(),
                sensor_memory=SimpleNamespace(),
                ops_backfiller=SimpleNamespace(),
                planner=SimpleNamespace(),
                proactive_policy=SimpleNamespace(),
                retention_policy=SimpleNamespace(),
            )

            result = service.probe_remote_ready()

        self.assertFalse(result.ready)
        self.assertEqual(
            result.detail,
            "LongTermRemoteReadFailedError: Required remote long-term fast-topic retrieval failed.",
        )
        assert result.warm_result is not None
        self.assertTrue(result.warm_result.fast_topic_checked)
        self.assertFalse(result.warm_result.fast_topic_ready)
        self.assertEqual(result.warm_result.failed_snapshot_kind, "fast_topic_route")
        self.assertEqual(result.warm_result.checks[-1].snapshot_kind, "fast_topic_route")
        self.assertEqual(result.warm_result.checks[-1].selected_source, "current_scope_topk_contract")
        self.assertEqual(result.steps[-1].name, "object_store.select_fast_topic_objects_readiness")
        self.assertEqual(result.steps[-1].status, "fail")
        proof_contract = result.warm_result.proof_contract()
        self.assertTrue(proof_contract["fast_topic_checked"])
        self.assertFalse(proof_contract["fast_topic_ready"])
        self.assertIn(
            "configured namespace current-scope fast-topic retrieval route",
            proof_contract["operations_not_proved"],
        )
        self.assertEqual(len(object_store.fast_topic_calls), 1)
        self.assertEqual(object_store.fast_topic_calls[0]["timeout_s"], 8.0)

    def test_probe_remote_ready_uses_readiness_budget_for_fast_topic_route(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                long_term_memory_fast_topic_timeout_s=0.45,
                long_term_memory_remote_read_timeout_s=6.0,
                long_term_memory_remote_watchdog_probe_timeout_s=2.4,
                long_term_memory_remote_watchdog_startup_probe_timeout_s=5.5,
            )
            prompt_state = _CacheAwareFakeRemoteState(
                {
                    "prompt_memory": {"schema": "prompt_memory", "entries": []},
                    "user_context": {"schema": "managed_context", "entries": []},
                    "personality_context": {"schema": "managed_context", "entries": []},
                }
            )
            object_state = _CacheAwareFakeRemoteState(
                {
                    "objects": {"schema": "twinr_memory_object_catalog_v2", "version": 2, "items": []},
                    "conflicts": {"schema": "conflicts", "conflicts": []},
                    "archive": {"schema": "twinr_memory_archive_catalog_v2", "version": 2, "items": []},
                }
            )
            graph_state = _CacheAwareFakeRemoteState({"graph": {"schema": "graph", "nodes": [], "edges": []}})
            midterm_state = _CacheAwareFakeRemoteState({"midterm": {"schema": "midterm", "packets": []}})
            object_store = _ObjectStoreDouble(object_state)
            service = LongTermMemoryService(
                config=config,
                prompt_context_store=_PromptContextStoreDouble(prompt_state),
                graph_store=_GraphStoreDouble(graph_state),
                object_store=object_store,
                midterm_store=_MidtermStoreDouble(midterm_state),
                query_rewriter=SimpleNamespace(),
                retriever=SimpleNamespace(),
                extractor=SimpleNamespace(),
                multimodal_extractor=SimpleNamespace(),
                truth_maintainer=SimpleNamespace(),
                consolidator=SimpleNamespace(),
                conflict_resolver=SimpleNamespace(),
                reflector=SimpleNamespace(),
                sensor_memory=SimpleNamespace(),
                ops_backfiller=SimpleNamespace(),
                planner=SimpleNamespace(),
                proactive_policy=SimpleNamespace(),
                retention_policy=SimpleNamespace(),
            )

            startup_result = service.probe_remote_ready()
            steady_result = service.probe_remote_ready(bootstrap=False, include_archive=False)

        self.assertTrue(startup_result.ready)
        self.assertTrue(steady_result.ready)
        self.assertEqual(
            tuple(call["timeout_s"] for call in object_store.fast_topic_calls),
            (5.5, 2.4),
        )

    def test_probe_remote_ready_skips_fast_topic_route_when_disabled(self) -> None:
        class _DisabledFastTopicObjectStore(_ObjectStoreDouble):
            def select_fast_topic_objects(
                self,
                *,
                query_text: str | None = None,
                limit: int = 3,
                timeout_s: float | None = None,
            ):
                raise AssertionError("fast-topic readiness must not run when the lane is disabled")

        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                long_term_memory_fast_topic_enabled=False,
            )
            prompt_state = _CacheAwareFakeRemoteState(
                {
                    "prompt_memory": {"schema": "prompt_memory", "entries": []},
                    "user_context": {"schema": "managed_context", "entries": []},
                    "personality_context": {"schema": "managed_context", "entries": []},
                }
            )
            object_state = _CacheAwareFakeRemoteState(
                {
                    "objects": {"schema": "twinr_memory_object_catalog_v2", "version": 2, "items": []},
                    "conflicts": {"schema": "conflicts", "conflicts": []},
                    "archive": {"schema": "twinr_memory_archive_catalog_v2", "version": 2, "items": []},
                }
            )
            graph_state = _CacheAwareFakeRemoteState({"graph": {"schema": "graph", "nodes": [], "edges": []}})
            midterm_state = _CacheAwareFakeRemoteState({"midterm": {"schema": "midterm", "packets": []}})
            service = LongTermMemoryService(
                config=config,
                prompt_context_store=_PromptContextStoreDouble(prompt_state),
                graph_store=_GraphStoreDouble(graph_state),
                object_store=_DisabledFastTopicObjectStore(object_state),
                midterm_store=_MidtermStoreDouble(midterm_state),
                query_rewriter=SimpleNamespace(),
                retriever=SimpleNamespace(),
                extractor=SimpleNamespace(),
                multimodal_extractor=SimpleNamespace(),
                truth_maintainer=SimpleNamespace(),
                consolidator=SimpleNamespace(),
                conflict_resolver=SimpleNamespace(),
                reflector=SimpleNamespace(),
                sensor_memory=SimpleNamespace(),
                ops_backfiller=SimpleNamespace(),
                planner=SimpleNamespace(),
                proactive_policy=SimpleNamespace(),
                retention_policy=SimpleNamespace(),
            )

            result = service.probe_remote_ready()

        self.assertTrue(result.ready)
        assert result.warm_result is not None
        self.assertFalse(result.warm_result.fast_topic_checked)

    def test_probe_remote_ready_bootstrap_prefers_graph_readiness_bootstrap_contract(self) -> None:
        class _ReadinessGraphStore:
            def __init__(self, remote_state: _CacheAwareFakeRemoteState) -> None:
                self.remote_state = remote_state
                self.readiness_bootstrap_calls = 0
                self.legacy_bootstrap_calls = 0

            def ensure_remote_snapshot_for_readiness(self) -> bool:
                self.readiness_bootstrap_calls += 1
                return False

            def ensure_remote_snapshot(self) -> bool:
                self.legacy_bootstrap_calls += 1
                raise AssertionError("readiness bootstrap must not seed the graph current view")

            def probe_remote_current_view_for_readiness(self) -> dict[str, object]:
                return {
                    "generation_id": "gen-empty",
                    "topology_index_name": "twinr_graph_test_bootstrap_empty",
                    "subject_node_id": "user:main",
                    "graph_id": "graph:user_main",
                    "created_at": "2026-04-01T10:00:00Z",
                    "updated_at": "2026-04-01T10:00:00Z",
                    "topology_refs": {"user:main": "bootstrap_empty:user:main"},
                    "synthetic_empty": True,
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            prompt_state = _CacheAwareFakeRemoteState(
                {
                    "prompt_memory": {"schema": "prompt_memory", "entries": []},
                    "user_context": {"schema": "managed_context", "entries": []},
                    "personality_context": {"schema": "managed_context", "entries": []},
                }
            )
            object_state = _CacheAwareFakeRemoteState(
                {
                    "objects": {"schema": "twinr_memory_object_catalog_v2", "version": 2, "items": []},
                    "conflicts": {"schema": "conflicts", "conflicts": []},
                    "archive": {"schema": "twinr_memory_archive_catalog_v2", "version": 2, "items": []},
                }
            )
            graph_state = _CacheAwareFakeRemoteState({"graph": {"schema": "graph", "nodes": [], "edges": []}})
            graph_store = _ReadinessGraphStore(graph_state)
            midterm_state = _CacheAwareFakeRemoteState({"midterm": {"schema": "midterm", "packets": []}})
            service = LongTermMemoryService(
                config=config,
                prompt_context_store=_PromptContextStoreDouble(prompt_state),
                graph_store=graph_store,
                object_store=_ObjectStoreDouble(object_state),
                midterm_store=_MidtermStoreDouble(midterm_state),
                query_rewriter=SimpleNamespace(),
                retriever=SimpleNamespace(),
                extractor=SimpleNamespace(),
                multimodal_extractor=SimpleNamespace(),
                truth_maintainer=SimpleNamespace(),
                consolidator=SimpleNamespace(),
                conflict_resolver=SimpleNamespace(),
                reflector=SimpleNamespace(),
                sensor_memory=SimpleNamespace(),
                ops_backfiller=SimpleNamespace(),
                planner=SimpleNamespace(),
                proactive_policy=SimpleNamespace(),
                retention_policy=SimpleNamespace(),
            )

            result = service.probe_remote_ready()

        self.assertTrue(result.ready)
        self.assertEqual(graph_store.readiness_bootstrap_calls, 1)
        self.assertEqual(graph_store.legacy_bootstrap_calls, 0)
        self.assertEqual(graph_state.load_calls, [])
        assert result.warm_result is not None
        selected_sources = {check.snapshot_kind: check.selected_source for check in result.warm_result.checks}
        self.assertEqual(selected_sources["graph"], "graph_current_view")

    def test_probe_remote_ready_bootstrap_prefers_object_store_readiness_bootstrap_contract(self) -> None:
        class _ReadinessObjectStore:
            def __init__(self, remote_state: _CacheAwareFakeRemoteState) -> None:
                self.remote_state = remote_state
                self.readiness_bootstrap_calls = 0
                self.legacy_bootstrap_calls = 0

            def ensure_remote_snapshots_for_readiness(self) -> tuple[str, ...]:
                self.readiness_bootstrap_calls += 1
                return ()

            def ensure_remote_snapshots(self) -> tuple[str, ...]:
                self.legacy_bootstrap_calls += 1
                raise AssertionError("readiness bootstrap must not seed empty structured snapshots")

            def probe_remote_current_snapshot_for_readiness(self, *, snapshot_kind: str) -> dict[str, object]:
                payloads = {
                    "objects": {
                        "schema": "twinr_memory_object_store",
                        "version": 2,
                        "objects": [],
                        "written_at": "1970-01-01T00:00:00+00:00",
                    },
                    "conflicts": {
                        "schema": "twinr_memory_conflict_store",
                        "version": 2,
                        "conflicts": [],
                        "written_at": "1970-01-01T00:00:00+00:00",
                    },
                    "archive": {
                        "schema": "twinr_memory_archive_store",
                        "version": 2,
                        "objects": [],
                        "written_at": "1970-01-01T00:00:00+00:00",
                    },
                }
                return dict(payloads[snapshot_kind])

            def select_fast_topic_objects(
                self,
                *,
                query_text: str | None = None,
                limit: int = 3,
                timeout_s: float | None = None,
            ):
                del query_text, limit, timeout_s
                return ()

        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            prompt_state = _CacheAwareFakeRemoteState(
                {
                    "prompt_memory": {"schema": "prompt_memory", "entries": []},
                    "user_context": {"schema": "managed_context", "entries": []},
                    "personality_context": {"schema": "managed_context", "entries": []},
                }
            )
            object_state = _CacheAwareFakeRemoteState(
                {
                    "objects": {"schema": "twinr_memory_object_catalog_v2", "version": 2, "items": []},
                    "conflicts": {"schema": "conflicts", "conflicts": []},
                    "archive": {"schema": "twinr_memory_archive_catalog_v2", "version": 2, "items": []},
                }
            )
            graph_state = _CacheAwareFakeRemoteState({"graph": {"schema": "graph", "nodes": [], "edges": []}})
            midterm_state = _CacheAwareFakeRemoteState({"midterm": {"schema": "midterm", "packets": []}})
            object_store = _ReadinessObjectStore(object_state)
            service = LongTermMemoryService(
                config=config,
                prompt_context_store=_PromptContextStoreDouble(prompt_state),
                graph_store=_GraphStoreDouble(graph_state),
                object_store=object_store,
                midterm_store=_MidtermStoreDouble(midterm_state),
                query_rewriter=SimpleNamespace(),
                retriever=SimpleNamespace(),
                extractor=SimpleNamespace(),
                multimodal_extractor=SimpleNamespace(),
                truth_maintainer=SimpleNamespace(),
                consolidator=SimpleNamespace(),
                conflict_resolver=SimpleNamespace(),
                reflector=SimpleNamespace(),
                sensor_memory=SimpleNamespace(),
                ops_backfiller=SimpleNamespace(),
                planner=SimpleNamespace(),
                proactive_policy=SimpleNamespace(),
                retention_policy=SimpleNamespace(),
            )

            result = service.probe_remote_ready()

        self.assertTrue(result.ready)
        self.assertEqual(object_store.readiness_bootstrap_calls, 1)
        self.assertEqual(object_store.legacy_bootstrap_calls, 0)
        assert result.warm_result is not None
        selected_sources = {check.snapshot_kind: check.selected_source for check in result.warm_result.checks}
        self.assertEqual(selected_sources["objects"], "object_store_current_contract")
        self.assertEqual(selected_sources["conflicts"], "object_store_current_contract")
        self.assertEqual(selected_sources["archive"], "object_store_current_contract")

    def test_probe_remote_ready_bootstrap_prefers_midterm_readiness_bootstrap_contract(self) -> None:
        class _ReadinessMidtermStore:
            def __init__(self, remote_state: _CacheAwareFakeRemoteState) -> None:
                self.remote_state = remote_state
                self.readiness_bootstrap_calls = 0
                self.legacy_bootstrap_calls = 0

            def ensure_remote_snapshot_for_readiness(self) -> bool:
                self.readiness_bootstrap_calls += 1
                return False

            def ensure_remote_snapshot(self) -> bool:
                self.legacy_bootstrap_calls += 1
                raise AssertionError("readiness bootstrap must not seed empty midterm state")

            def probe_remote_current_head_for_readiness(self) -> dict[str, object]:
                return {
                    "schema": "twinr_memory_midterm_store",
                    "version": 2,
                    "packets": [],
                    "written_at": "1970-01-01T00:00:00+00:00",
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            prompt_state = _CacheAwareFakeRemoteState(
                {
                    "prompt_memory": {"schema": "prompt_memory", "entries": []},
                    "user_context": {"schema": "managed_context", "entries": []},
                    "personality_context": {"schema": "managed_context", "entries": []},
                }
            )
            object_state = _CacheAwareFakeRemoteState(
                {
                    "objects": {"schema": "twinr_memory_object_catalog_v2", "version": 2, "items": []},
                    "conflicts": {"schema": "conflicts", "conflicts": []},
                    "archive": {"schema": "twinr_memory_archive_catalog_v2", "version": 2, "items": []},
                }
            )
            graph_state = _CacheAwareFakeRemoteState({"graph": {"schema": "graph", "nodes": [], "edges": []}})
            midterm_state = _CacheAwareFakeRemoteState({"midterm": {"schema": "midterm", "packets": []}})
            midterm_store = _ReadinessMidtermStore(midterm_state)
            service = LongTermMemoryService(
                config=config,
                prompt_context_store=_PromptContextStoreDouble(prompt_state),
                graph_store=_GraphStoreDouble(graph_state),
                object_store=_ObjectStoreDouble(object_state),
                midterm_store=midterm_store,
                query_rewriter=SimpleNamespace(),
                retriever=SimpleNamespace(),
                extractor=SimpleNamespace(),
                multimodal_extractor=SimpleNamespace(),
                truth_maintainer=SimpleNamespace(),
                consolidator=SimpleNamespace(),
                conflict_resolver=SimpleNamespace(),
                reflector=SimpleNamespace(),
                sensor_memory=SimpleNamespace(),
                ops_backfiller=SimpleNamespace(),
                planner=SimpleNamespace(),
                proactive_policy=SimpleNamespace(),
                retention_policy=SimpleNamespace(),
            )

            result = service.probe_remote_ready()

        self.assertTrue(result.ready)
        self.assertEqual(midterm_store.readiness_bootstrap_calls, 1)
        self.assertEqual(midterm_store.legacy_bootstrap_calls, 0)
        assert result.warm_result is not None
        selected_sources = {check.snapshot_kind: check.selected_source for check in result.warm_result.checks}
        self.assertEqual(selected_sources["midterm"], "catalog_current_head")

    def test_remote_snapshot_save_and_load_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(temp_dir)
            read_opener = FakeOpener()
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "doc-1",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "objects",
                            "updated_at": "2026-03-19T13:00:00+00:00",
                            "body": {"schema": "object_store", "objects": []},
                        }
                    ),
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "pointer-doc-1",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "__pointer__:objects",
                            "updated_at": "2026-03-19T13:00:01+00:00",
                            "body": {
                                "schema": "twinr_remote_snapshot_pointer_v1",
                                "version": 1,
                                "snapshot_kind": "objects",
                                "document_id": "doc-1",
                            },
                        }
                    ),
                }
            )
            read_opener.queue_http_error(404, {"detail": "not found"})
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "pointer-doc-2",
                    "origin_uri": "twinr://longterm/test-namespace/__pointer__:objects",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "__pointer__:objects",
                            "updated_at": "2026-03-19T13:00:01+00:00",
                            "body": {
                                "schema": "twinr_remote_snapshot_pointer_v1",
                                "version": 1,
                                "snapshot_kind": "objects",
                                "document_id": "doc-2",
                            },
                        }
                    ),
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "doc-2",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "objects",
                            "body": {"schema": "object_store", "objects": [{"memory_id": "fact:1"}]},
                        }
                    ),
                }
            )
            write_opener = FakeOpener()
            write_opener.queue_json(
                {
                    "success": True,
                    "items": [
                        {
                            "success": True,
                            "document_id": "doc-1",
                            "payload_id": "doc-1",
                        }
                    ],
                }
            )
            write_opener.queue_json({"success": True, "stored": 1})
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=write_opener,
                ),
            )

            state.save_snapshot(snapshot_kind="objects", payload={"schema": "object_store", "objects": []})
            payload = state.load_snapshot(snapshot_kind="objects", local_path=Path(temp_dir) / "objects.json")

        self.assertEqual(payload, {"schema": "object_store", "objects": [{"memory_id": "fact:1"}]})
        body = json.loads(write_opener.calls[0]["body"])
        self.assertEqual(body["items"][0]["payload"]["namespace"], "test-namespace")
        self.assertEqual(body["items"][0]["payload"]["snapshot_kind"], "objects")
        self.assertEqual(body["items"][0]["metadata"]["twinr_snapshot_kind"], "objects")

    def test_remote_snapshot_save_persists_pointer_when_store_returns_document_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(temp_dir)
            read_opener = FakeOpener()
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "doc-123",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "objects",
                            "updated_at": "2026-03-19T13:00:00+00:00",
                            "body": {"schema": "object_store", "objects": []},
                        }
                    ),
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "pointer-doc-123",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "__pointer__:objects",
                            "updated_at": "2026-03-19T13:00:01+00:00",
                            "body": {
                                "schema": "twinr_remote_snapshot_pointer_v1",
                                "version": 1,
                                "snapshot_kind": "objects",
                                "document_id": "doc-123",
                            },
                        }
                    ),
                }
            )
            write_opener = FakeOpener()
            write_opener.queue_json(
                {
                    "success": True,
                    "items": [
                        {
                            "success": True,
                            "document_id": "doc-123",
                            "payload_id": "doc-123",
                        }
                    ],
                }
            )
            write_opener.queue_json({"success": True, "stored": 1})
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=write_opener,
                ),
            )

            state.save_snapshot(snapshot_kind="objects", payload={"schema": "object_store", "objects": []})

        self.assertEqual(len(write_opener.calls), 2)
        first_body = json.loads(write_opener.calls[0]["body"])
        second_body = json.loads(write_opener.calls[1]["body"])
        self.assertEqual(first_body["items"][0]["payload"]["snapshot_kind"], "objects")
        self.assertEqual(second_body["items"][0]["payload"]["snapshot_kind"], "__pointer__:objects")
        self.assertEqual(second_body["items"][0]["payload"]["body"]["schema"], "twinr_remote_snapshot_pointer_v1")
        self.assertEqual(second_body["items"][0]["payload"]["body"]["document_id"], "doc-123")

    def test_write_learned_snapshot_document_id_hint_persists_across_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(temp_dir)
            first_state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=FakeOpener(),
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=FakeOpener(),
                ),
            )
            first_state._remember_snapshot_document_id(
                snapshot_kind="objects",
                document_id="doc-persisted-123",
                persist=True,
            )

            read_opener = FakeOpener()
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "doc-persisted-123",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "objects",
                            "updated_at": "2026-03-20T15:25:01+00:00",
                            "body": {"schema": "object_store", "objects": [{"memory_id": "fact:persisted"}]},
                        }
                    ),
                }
            )
            second_state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=FakeOpener(),
                ),
            )

            payload = second_state.load_snapshot(snapshot_kind="objects", local_path=Path(temp_dir) / "objects.json")

        self.assertEqual(payload, {"schema": "object_store", "objects": [{"memory_id": "fact:persisted"}]})
        self.assertEqual(len(read_opener.calls), 1)
        first_query = parse_qs(urlparse(read_opener.calls[0]["full_url"]).query)
        self.assertEqual(first_query["document_id"], ["doc-persisted-123"])
        self.assertNotIn("origin_uri", first_query)

    def test_remote_snapshot_save_persists_pointer_when_attested_origin_read_supplies_document_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(temp_dir)
            read_opener = FakeOpener()
            read_opener.queue_json(
                {
                    "success": True,
                    "origin_uri": "twinr://longterm/test-namespace/conflicts",
                    "chunk_count": 1,
                    "chunks": [
                        {
                            "payload_id": "conflicts-doc-123",
                            "content": json.dumps(
                                {
                                    "schema": "twinr_remote_snapshot_v1",
                                    "namespace": "test-namespace",
                                    "snapshot_kind": "conflicts",
                                    "updated_at": "2026-03-19T13:00:00+00:00",
                                    "body": {
                                        "schema": "twinr_memory_conflict_catalog_v3",
                                        "version": 3,
                                        "items_count": 0,
                                        "segments": [],
                                    },
                                }
                            ),
                        }
                    ],
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "pointer-doc-123",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "__pointer__:conflicts",
                            "updated_at": "2026-03-19T13:00:01+00:00",
                            "body": {
                                "schema": "twinr_remote_snapshot_pointer_v1",
                                "version": 1,
                                "snapshot_kind": "conflicts",
                                "document_id": "conflicts-doc-123",
                            },
                        }
                    ),
                }
            )
            write_opener = FakeOpener()
            write_opener.queue_json({"success": True, "stored": 1})
            write_opener.queue_json({"success": True, "stored": 1})
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=write_opener,
                ),
            )

            state.save_snapshot(
                snapshot_kind="conflicts",
                payload={"schema": "twinr_memory_conflict_catalog_v3", "version": 3, "items_count": 0, "segments": []},
            )

        self.assertEqual(state._cached_snapshot_document_id(snapshot_kind="conflicts"), "conflicts-doc-123")
        self.assertEqual(len(write_opener.calls), 2)
        pointer_body = json.loads(write_opener.calls[1]["body"])
        self.assertEqual(pointer_body["items"][0]["payload"]["snapshot_kind"], "__pointer__:conflicts")
        self.assertEqual(pointer_body["items"][0]["payload"]["body"]["document_id"], "conflicts-doc-123")

    def test_remote_snapshot_load_prefers_pointer_document_id_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(temp_dir)
            read_opener = FakeOpener()
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "pointer-doc",
                    "origin_uri": "twinr://longterm/test-namespace/__pointer__:objects",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "__pointer__:objects",
                            "updated_at": "2026-03-16T20:00:00+00:00",
                            "body": {
                                "schema": "twinr_remote_snapshot_pointer_v1",
                                "version": 1,
                                "snapshot_kind": "objects",
                                "document_id": "doc-123",
                            },
                        }
                    ),
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "doc-123",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "objects",
                            "updated_at": "2026-03-16T20:00:01+00:00",
                            "body": {"schema": "object_store", "objects": [{"memory_id": "fact:pointer"}]},
                        }
                    ),
                }
            )
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=FakeOpener(),
                ),
            )

            payload = state.load_snapshot(snapshot_kind="objects", local_path=Path(temp_dir) / "objects.json")

        self.assertEqual(payload, {"schema": "object_store", "objects": [{"memory_id": "fact:pointer"}]})
        first_query = parse_qs(urlparse(read_opener.calls[0]["full_url"]).query)
        second_query = parse_qs(urlparse(read_opener.calls[1]["full_url"]).query)
        self.assertEqual(first_query["origin_uri"], ["twinr://longterm/test-namespace/__pointer__%3Aobjects"])
        self.assertEqual(second_query["document_id"], ["doc-123"])
        self.assertNotIn("origin_uri", second_query)

    def test_remote_snapshot_load_uses_bootstrap_timeout_for_uncached_origin_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                self._config(temp_dir),
                long_term_memory_remote_read_timeout_s=8.0,
            )
            read_opener = FakeOpener()
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "pointer-doc",
                    "origin_uri": "twinr://longterm/test-namespace/__pointer__:objects",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "__pointer__:objects",
                            "updated_at": "2026-03-20T15:30:00+00:00",
                            "body": {
                                "schema": "twinr_remote_snapshot_pointer_v1",
                                "version": 1,
                                "snapshot_kind": "objects",
                                "document_id": "doc-bootstrap-123",
                            },
                        }
                    ),
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "doc-bootstrap-123",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "objects",
                            "updated_at": "2026-03-20T15:30:01+00:00",
                            "body": {"schema": "object_store", "objects": [{"memory_id": "fact:bootstrap"}]},
                        }
                    ),
                }
            )
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=FakeOpener(),
                ),
            )

            payload = state.load_snapshot(snapshot_kind="objects", local_path=Path(temp_dir) / "objects.json")

        self.assertEqual(payload, {"schema": "object_store", "objects": [{"memory_id": "fact:bootstrap"}]})
        self.assertEqual(len(read_opener.calls), 2)
        first_query = parse_qs(urlparse(read_opener.calls[0]["full_url"]).query)
        second_query = parse_qs(urlparse(read_opener.calls[1]["full_url"]).query)
        self.assertEqual(first_query["origin_uri"], ["twinr://longterm/test-namespace/__pointer__%3Aobjects"])
        self.assertEqual(read_opener.calls[0]["timeout"], 24.0)
        self.assertEqual(second_query["document_id"], ["doc-bootstrap-123"])
        self.assertEqual(read_opener.calls[1]["timeout"], 20.0)

    def test_remote_snapshot_load_reuses_explicit_document_id_hint_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(temp_dir)
            read_opener = FakeOpener()
            graph_payload = {
                "schema": "twinr_remote_snapshot_v1",
                "namespace": "test-namespace",
                "snapshot_kind": "graph",
                "updated_at": "2026-03-18T13:00:01+00:00",
                "body": {"schema": "graph", "nodes": [{"id": "user:main"}], "edges": []},
            }
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "graph-doc-123",
                    "content": json.dumps(graph_payload),
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "graph-doc-123",
                    "content": json.dumps(graph_payload),
                }
            )
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=FakeOpener(),
                ),
            )
            state._remember_snapshot_document_id(snapshot_kind="graph", document_id="graph-doc-123")

            first = state.load_snapshot(snapshot_kind="graph", local_path=Path(temp_dir) / "graph.json")
            second = state.load_snapshot(snapshot_kind="graph", local_path=Path(temp_dir) / "graph.json")

        self.assertEqual(first, {"schema": "graph", "nodes": [{"id": "user:main"}], "edges": []})
        self.assertEqual(second, first)
        self.assertEqual(len(read_opener.calls), 2)
        first_query = parse_qs(urlparse(read_opener.calls[0]["full_url"]).query)
        second_query = parse_qs(urlparse(read_opener.calls[1]["full_url"]).query)
        self.assertEqual(first_query["document_id"], ["graph-doc-123"])
        self.assertEqual(second_query["document_id"], ["graph-doc-123"])
        self.assertNotIn("origin_uri", first_query)
        self.assertNotIn("origin_uri", second_query)

    def test_remote_snapshot_load_reuses_cached_document_id_hint_when_opted_in(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(temp_dir)
            read_opener = FakeOpener()
            graph_payload = {
                "schema": "twinr_remote_snapshot_v1",
                "namespace": "test-namespace",
                "snapshot_kind": "graph",
                "updated_at": "2026-03-18T13:00:01+00:00",
                "body": {"schema": "graph", "nodes": [{"id": "user:main"}], "edges": []},
            }
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "graph-doc-123",
                    "content": json.dumps(graph_payload),
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "graph-doc-123",
                    "content": json.dumps(graph_payload),
                }
            )
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=FakeOpener(),
                ),
            )
            state._remember_snapshot_document_id(snapshot_kind="graph", document_id="graph-doc-123")

            first = state.load_snapshot(
                snapshot_kind="graph",
                local_path=Path(temp_dir) / "graph.json",
                prefer_cached_document_id=True,
            )
            second = state.load_snapshot(
                snapshot_kind="graph",
                local_path=Path(temp_dir) / "graph.json",
                prefer_cached_document_id=True,
            )

        self.assertEqual(first, {"schema": "graph", "nodes": [{"id": "user:main"}], "edges": []})
        self.assertEqual(second, first)
        self.assertEqual(len(read_opener.calls), 2)
        first_query = parse_qs(urlparse(read_opener.calls[0]["full_url"]).query)
        second_query = parse_qs(urlparse(read_opener.calls[1]["full_url"]).query)
        self.assertEqual(first_query["document_id"], ["graph-doc-123"])
        self.assertEqual(second_query["document_id"], ["graph-doc-123"])
        self.assertNotIn("origin_uri", first_query)
        self.assertNotIn("origin_uri", second_query)

    def test_probe_snapshot_load_reuses_cached_document_id_hint_when_opted_in(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(temp_dir)
            read_opener = FakeOpener()
            graph_payload = {
                "schema": "twinr_remote_snapshot_v1",
                "namespace": "test-namespace",
                "snapshot_kind": "graph",
                "updated_at": "2026-03-18T13:00:01+00:00",
                "body": {"schema": "graph", "nodes": [{"id": "user:main"}], "edges": []},
            }
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "graph-doc-123",
                    "content": json.dumps(graph_payload),
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "graph-doc-123",
                    "content": json.dumps(graph_payload),
                }
            )
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=FakeOpener(),
                ),
            )
            state._remember_snapshot_document_id(snapshot_kind="graph", document_id="graph-doc-123")

            first = state.probe_snapshot_load(
                snapshot_kind="graph",
                local_path=Path(temp_dir) / "graph.json",
                prefer_cached_document_id=True,
            )
            second = state.probe_snapshot_load(
                snapshot_kind="graph",
                local_path=Path(temp_dir) / "graph.json",
                prefer_cached_document_id=True,
            )

        self.assertEqual(first.payload, {"schema": "graph", "nodes": [{"id": "user:main"}], "edges": []})
        self.assertEqual(second.payload, first.payload)
        self.assertEqual(len(read_opener.calls), 2)
        first_query = parse_qs(urlparse(read_opener.calls[0]["full_url"]).query)
        second_query = parse_qs(urlparse(read_opener.calls[1]["full_url"]).query)
        self.assertEqual(first_query["document_id"], ["graph-doc-123"])
        self.assertEqual(second_query["document_id"], ["graph-doc-123"])
        self.assertNotIn("origin_uri", first_query)
        self.assertNotIn("origin_uri", second_query)

    def test_remote_snapshot_load_reuses_inprocess_read_cache_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(self._config(temp_dir), long_term_memory_remote_read_cache_ttl_s=60.0)
            read_opener = FakeOpener()
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "pointer-doc",
                    "origin_uri": "twinr://longterm/test-namespace/__pointer__:objects",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "__pointer__:objects",
                            "updated_at": "2026-03-16T20:00:00+00:00",
                            "body": {
                                "schema": "twinr_remote_snapshot_pointer_v1",
                                "version": 1,
                                "snapshot_kind": "objects",
                                "document_id": "doc-123",
                            },
                        }
                    ),
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "doc-123",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "objects",
                            "updated_at": "2026-03-16T20:00:01+00:00",
                            "body": {"schema": "object_store", "objects": [{"memory_id": "fact:pointer"}]},
                        }
                    ),
                }
            )
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=FakeOpener(),
                ),
            )

            first = state.load_snapshot(snapshot_kind="objects", local_path=Path(temp_dir) / "objects.json")
            second = state.load_snapshot(snapshot_kind="objects", local_path=Path(temp_dir) / "objects.json")

        self.assertEqual(first, {"schema": "object_store", "objects": [{"memory_id": "fact:pointer"}]})
        self.assertEqual(second, first)
        self.assertEqual(len(read_opener.calls), 2)

    def test_remote_snapshot_load_reuses_read_learned_document_id_after_cache_clear_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(self._config(temp_dir), long_term_memory_remote_read_cache_ttl_s=60.0)
            read_opener = FakeOpener()
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "pointer-doc",
                    "origin_uri": "twinr://longterm/test-namespace/__pointer__:objects",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "__pointer__:objects",
                            "updated_at": "2026-03-16T20:00:00+00:00",
                            "body": {
                                "schema": "twinr_remote_snapshot_pointer_v1",
                                "version": 1,
                                "snapshot_kind": "objects",
                                "document_id": "doc-123",
                            },
                        }
                    ),
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "doc-123",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "objects",
                            "updated_at": "2026-03-16T20:00:01+00:00",
                            "body": {"schema": "object_store", "objects": [{"memory_id": "fact:pointer"}]},
                        }
                    ),
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "doc-123",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "objects",
                            "updated_at": "2026-03-16T20:00:01+00:00",
                            "body": {"schema": "object_store", "objects": [{"memory_id": "fact:pointer"}]},
                        }
                    ),
                }
            )
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=FakeOpener(),
                ),
            )

            first = state.load_snapshot(snapshot_kind="objects", local_path=Path(temp_dir) / "objects.json")
            state._clear_snapshot_read(snapshot_kind="objects")
            second = state.load_snapshot(snapshot_kind="objects", local_path=Path(temp_dir) / "objects.json")

        self.assertEqual(first, {"schema": "object_store", "objects": [{"memory_id": "fact:pointer"}]})
        self.assertEqual(second, first)
        self.assertEqual(len(read_opener.calls), 3)
        first_query = parse_qs(urlparse(read_opener.calls[0]["full_url"]).query)
        second_query = parse_qs(urlparse(read_opener.calls[1]["full_url"]).query)
        third_query = parse_qs(urlparse(read_opener.calls[2]["full_url"]).query)
        self.assertEqual(first_query["origin_uri"], ["twinr://longterm/test-namespace/__pointer__%3Aobjects"])
        self.assertEqual(second_query["document_id"], ["doc-123"])
        self.assertEqual(third_query["document_id"], ["doc-123"])
        self.assertNotIn("origin_uri", second_query)
        self.assertNotIn("origin_uri", third_query)

    def test_remote_snapshot_load_revalidates_after_external_update_instead_of_reusing_read_learned_hint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(temp_dir)
            read_opener = FakeOpener()
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "pointer-doc-1",
                    "origin_uri": "twinr://longterm/test-namespace/__pointer__:objects",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "__pointer__:objects",
                            "updated_at": "2026-03-18T13:00:00+00:00",
                            "body": {
                                "schema": "twinr_remote_snapshot_pointer_v1",
                                "version": 1,
                                "snapshot_kind": "objects",
                                "document_id": "doc-123",
                            },
                        }
                    ),
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "doc-123",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "objects",
                            "updated_at": "2026-03-18T13:00:01+00:00",
                            "body": {"schema": "object_store", "objects": [{"memory_id": "fact:stale"}]},
                        }
                    ),
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "pointer-doc-2",
                    "origin_uri": "twinr://longterm/test-namespace/__pointer__:objects",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "__pointer__:objects",
                            "updated_at": "2026-03-18T13:05:00+00:00",
                            "body": {
                                "schema": "twinr_remote_snapshot_pointer_v1",
                                "version": 1,
                                "snapshot_kind": "objects",
                                "document_id": "doc-456",
                            },
                        }
                    ),
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "doc-456",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "objects",
                            "updated_at": "2026-03-18T13:05:01+00:00",
                            "body": {"schema": "object_store", "objects": [{"memory_id": "fact:fresh"}]},
                        }
                    ),
                }
            )
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=FakeOpener(),
                ),
            )

            first = state.load_snapshot(snapshot_kind="objects", local_path=Path(temp_dir) / "objects.json")
            second = state.load_snapshot(snapshot_kind="objects", local_path=Path(temp_dir) / "objects.json")

        self.assertEqual(first, {"schema": "object_store", "objects": [{"memory_id": "fact:stale"}]})
        self.assertEqual(second, {"schema": "object_store", "objects": [{"memory_id": "fact:fresh"}]})
        self.assertEqual(len(read_opener.calls), 4)
        third_query = parse_qs(urlparse(read_opener.calls[2]["full_url"]).query)
        fourth_query = parse_qs(urlparse(read_opener.calls[3]["full_url"]).query)
        self.assertEqual(third_query["origin_uri"], ["twinr://longterm/test-namespace/__pointer__%3Aobjects"])
        self.assertEqual(fourth_query["document_id"], ["doc-456"])

    def test_remote_snapshot_load_clears_stale_explicit_document_id_hint_after_stale_hit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(temp_dir)
            read_opener = FakeOpener()
            read_opener.queue_http_error(404, {"detail": "not found"})
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "pointer-doc-2",
                    "origin_uri": "twinr://longterm/test-namespace/__pointer__:graph",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "__pointer__:graph",
                            "updated_at": "2026-03-18T13:05:00+00:00",
                            "body": {
                                "schema": "twinr_remote_snapshot_pointer_v1",
                                "version": 1,
                                "snapshot_kind": "graph",
                                "document_id": "graph-doc-456",
                            },
                        }
                    ),
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "graph-doc-456",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "graph",
                            "updated_at": "2026-03-18T13:05:01+00:00",
                            "body": {"schema": "graph", "nodes": [{"id": "user:next"}], "edges": []},
                        }
                    ),
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "pointer-doc-2",
                    "origin_uri": "twinr://longterm/test-namespace/__pointer__:graph",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "__pointer__:graph",
                            "updated_at": "2026-03-18T13:05:00+00:00",
                            "body": {
                                "schema": "twinr_remote_snapshot_pointer_v1",
                                "version": 1,
                                "snapshot_kind": "graph",
                                "document_id": "graph-doc-456",
                            },
                        }
                    ),
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "graph-doc-456",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "graph",
                            "updated_at": "2026-03-18T13:05:01+00:00",
                            "body": {"schema": "graph", "nodes": [{"id": "user:next"}], "edges": []},
                        }
                    ),
                }
            )
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=FakeOpener(),
                ),
            )
            state._remember_snapshot_document_id(snapshot_kind="graph", document_id="graph-doc-123")

            first = state.load_snapshot(
                snapshot_kind="graph",
                local_path=Path(temp_dir) / "graph.json",
                prefer_cached_document_id=True,
            )
            second = state.load_snapshot(
                snapshot_kind="graph",
                local_path=Path(temp_dir) / "graph.json",
                prefer_cached_document_id=True,
            )

        self.assertEqual(first, {"schema": "graph", "nodes": [{"id": "user:next"}], "edges": []})
        self.assertEqual(second, {"schema": "graph", "nodes": [{"id": "user:next"}], "edges": []})
        self.assertEqual(len(read_opener.calls), 5)
        first_query = parse_qs(urlparse(read_opener.calls[0]["full_url"]).query)
        second_query = parse_qs(urlparse(read_opener.calls[1]["full_url"]).query)
        third_query = parse_qs(urlparse(read_opener.calls[2]["full_url"]).query)
        fourth_query = parse_qs(urlparse(read_opener.calls[3]["full_url"]).query)
        fifth_query = parse_qs(urlparse(read_opener.calls[4]["full_url"]).query)
        self.assertEqual(first_query["document_id"], ["graph-doc-123"])
        self.assertEqual(second_query["origin_uri"], ["twinr://longterm/test-namespace/__pointer__%3Agraph"])
        self.assertEqual(third_query["document_id"], ["graph-doc-456"])
        self.assertEqual(fourth_query["origin_uri"], ["twinr://longterm/test-namespace/__pointer__%3Agraph"])
        self.assertEqual(fifth_query["document_id"], ["graph-doc-456"])

    def test_remote_snapshot_load_falls_back_after_unreadable_cached_document_id_hint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(self._config(temp_dir), long_term_memory_remote_retry_attempts=1)
            read_opener = FakeOpener()
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "doc-123",
                    "origin_uri": "twinr://longterm/test-namespace/objects",
                    "chunks": [{"payload_id": "doc-123", "content": "{\"schema\":\"broken\""}],
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "pointer-doc-2",
                    "origin_uri": "twinr://longterm/test-namespace/__pointer__:objects",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "__pointer__:objects",
                            "updated_at": "2026-03-18T13:05:00+00:00",
                            "body": {
                                "schema": "twinr_remote_snapshot_pointer_v1",
                                "version": 1,
                                "snapshot_kind": "objects",
                                "document_id": "doc-456",
                            },
                        }
                    ),
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "doc-456",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "objects",
                            "updated_at": "2026-03-18T13:05:01+00:00",
                            "body": {"schema": "object_store", "objects": [{"memory_id": "fact:fresh"}]},
                        }
                    ),
                }
            )
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=FakeOpener(),
                ),
            )
            state._remember_snapshot_document_id(snapshot_kind="objects", document_id="doc-123")

            payload = state.load_snapshot(
                snapshot_kind="objects",
                local_path=Path(temp_dir) / "objects.json",
                prefer_cached_document_id=True,
            )

        self.assertEqual(payload, {"schema": "object_store", "objects": [{"memory_id": "fact:fresh"}]})
        self.assertEqual(len(read_opener.calls), 3)
        first_query = parse_qs(urlparse(read_opener.calls[0]["full_url"]).query)
        second_query = parse_qs(urlparse(read_opener.calls[1]["full_url"]).query)
        third_query = parse_qs(urlparse(read_opener.calls[2]["full_url"]).query)
        self.assertEqual(first_query["document_id"], ["doc-123"])
        self.assertEqual(second_query["origin_uri"], ["twinr://longterm/test-namespace/__pointer__%3Aobjects"])
        self.assertEqual(third_query["document_id"], ["doc-456"])

    def test_remote_snapshot_load_repairs_pointer_from_latest_origin_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(temp_dir)
            read_opener = FakeOpener()
            read_opener.queue_http_error(404, {"detail": "not found"})
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "old-doc",
                    "origin_uri": "twinr://longterm/test-namespace/objects",
                    "chunk_count": 2,
                    "chunks": [
                        {
                            "payload_id": "old-doc",
                            "content": json.dumps(
                                {
                                    "schema": "twinr_remote_snapshot_v1",
                                    "namespace": "test-namespace",
                                    "snapshot_kind": "objects",
                                    "updated_at": "2026-03-16T18:00:00+00:00",
                                    "body": {"schema": "object_store", "objects": [{"memory_id": "fact:old"}]},
                                }
                            ),
                        },
                        {
                            "payload_id": "new-doc",
                            "content": json.dumps(
                                {
                                    "schema": "twinr_remote_snapshot_v1",
                                    "namespace": "test-namespace",
                                    "snapshot_kind": "objects",
                                    "updated_at": "2026-03-16T19:00:00+00:00",
                                    "body": {"schema": "object_store", "objects": [{"memory_id": "fact:new"}]},
                                }
                            ),
                        },
                    ],
                }
            )
            write_opener = FakeOpener()
            write_opener.queue_json({"success": True, "stored": 1})
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=write_opener,
                ),
            )

            payload = state.load_snapshot(snapshot_kind="objects", local_path=Path(temp_dir) / "objects.json")

        self.assertEqual(payload, {"schema": "object_store", "objects": [{"memory_id": "fact:new"}]})
        self.assertEqual(len(write_opener.calls), 1)
        pointer_body = json.loads(write_opener.calls[0]["body"])
        self.assertEqual(pointer_body["items"][0]["payload"]["snapshot_kind"], "__pointer__:objects")
        self.assertEqual(pointer_body["items"][0]["payload"]["body"]["document_id"], "new-doc")

    def test_remote_snapshot_load_repairs_stale_pointer_document_from_origin_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(temp_dir)
            read_opener = FakeOpener()
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "pointer-doc",
                    "origin_uri": "twinr://longterm/test-namespace/__pointer__:objects",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "__pointer__:objects",
                            "updated_at": "2026-03-17T17:29:03+00:00",
                            "body": {
                                "schema": "twinr_remote_snapshot_pointer_v1",
                                "version": 1,
                                "snapshot_kind": "objects",
                                "document_id": "stale-doc",
                            },
                        }
                    ),
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "stale-doc",
                    "chunk_count": 1,
                    "chunks": [
                        {
                            "payload_id": "stale-doc",
                            "content": "",
                            "content_summary": "",
                        }
                    ],
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "fresh-doc",
                    "origin_uri": "twinr://longterm/test-namespace/objects",
                    "chunk_count": 1,
                    "chunks": [
                        {
                            "payload_id": "fresh-doc",
                            "content": json.dumps(
                                {
                                    "schema": "twinr_remote_snapshot_v1",
                                    "namespace": "test-namespace",
                                    "snapshot_kind": "objects",
                                    "updated_at": "2026-03-17T17:29:04+00:00",
                                    "body": {"schema": "object_store", "objects": [{"memory_id": "fact:fresh"}]},
                                }
                            ),
                        }
                    ],
                }
            )
            write_opener = FakeOpener()
            write_opener.queue_json({"success": True, "stored": 1})
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=write_opener,
                ),
            )

            payload = state.load_snapshot(snapshot_kind="objects", local_path=Path(temp_dir) / "objects.json")

        self.assertEqual(payload, {"schema": "object_store", "objects": [{"memory_id": "fact:fresh"}]})
        self.assertEqual(len(read_opener.calls), 3)
        first_query = parse_qs(urlparse(read_opener.calls[0]["full_url"]).query)
        second_query = parse_qs(urlparse(read_opener.calls[1]["full_url"]).query)
        third_query = parse_qs(urlparse(read_opener.calls[2]["full_url"]).query)
        self.assertEqual(first_query["origin_uri"], ["twinr://longterm/test-namespace/__pointer__%3Aobjects"])
        self.assertEqual(second_query["document_id"], ["stale-doc"])
        self.assertEqual(third_query["origin_uri"], ["twinr://longterm/test-namespace/objects"])
        pointer_body = json.loads(write_opener.calls[0]["body"])
        self.assertEqual(pointer_body["items"][0]["payload"]["snapshot_kind"], "__pointer__:objects")
        self.assertEqual(pointer_body["items"][0]["payload"]["body"]["document_id"], "fresh-doc")

    def test_remote_snapshot_load_retries_malformed_origin_response_before_failing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                self._config(temp_dir),
                long_term_memory_remote_retry_attempts=2,
                long_term_memory_remote_retry_backoff_s=0.0,
            )
            read_opener = FakeOpener()
            read_opener.queue_http_error(404, {"detail": "not found"})
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "objects-doc",
                    "chunk_count": 1,
                    "chunks": [
                        {
                            "payload_id": "objects-doc",
                            "content": "",
                            "content_summary": "",
                        }
                    ],
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "objects-doc",
                    "chunk_count": 1,
                    "chunks": [
                        {
                            "payload_id": "objects-doc",
                            "content": json.dumps(
                                {
                                    "schema": "twinr_remote_snapshot_v1",
                                    "namespace": "test-namespace",
                                    "snapshot_kind": "objects",
                                    "updated_at": "2026-03-18T12:00:00+00:00",
                                    "body": {"schema": "object_store", "objects": [{"memory_id": "fact:recovered"}]},
                                }
                            ),
                        }
                    ],
                }
            )
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=FakeOpener(),
                ),
            )

            payload = state.load_snapshot(snapshot_kind="objects", local_path=Path(temp_dir) / "objects.json")

        self.assertEqual(payload, {"schema": "object_store", "objects": [{"memory_id": "fact:recovered"}]})
        self.assertEqual(len(read_opener.calls), 3)

    def test_remote_snapshot_save_raises_when_pointer_persist_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                self._config(temp_dir),
                long_term_memory_remote_retry_attempts=1,
                long_term_memory_remote_retry_backoff_s=0.0,
            )
            read_opener = FakeOpener()
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "doc-123",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "objects",
                            "updated_at": "2026-03-19T13:00:00+00:00",
                            "body": {"schema": "object_store", "objects": []},
                        }
                    ),
                }
            )
            write_opener = FakeOpener()
            write_opener.queue_json(
                {
                    "success": True,
                    "items": [
                        {
                            "success": True,
                            "document_id": "doc-123",
                            "payload_id": "doc-123",
                        }
                    ],
                }
            )
            write_opener.queue_exception(TimeoutError("pointer write timed out"))
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=write_opener,
                ),
            )

            with self.assertRaises(LongTermRemoteUnavailableError):
                state.save_snapshot(snapshot_kind="objects", payload={"schema": "object_store", "objects": []})

        self.assertEqual(len(write_opener.calls), 2)
        first_body = json.loads(write_opener.calls[0]["body"])
        second_body = json.loads(write_opener.calls[1]["body"])
        self.assertEqual(first_body["items"][0]["payload"]["snapshot_kind"], "objects")
        self.assertEqual(second_body["items"][0]["payload"]["snapshot_kind"], "__pointer__:objects")

    def test_remote_snapshot_save_raises_when_pointer_readback_stays_unreadable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                self._config(temp_dir),
                long_term_memory_remote_retry_attempts=2,
                long_term_memory_remote_retry_backoff_s=0.0,
            )
            read_opener = FakeOpener()
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "doc-123",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "objects",
                            "updated_at": "2026-03-19T13:00:00+00:00",
                            "body": {"schema": "object_store", "objects": []},
                        }
                    ),
                }
            )
            read_opener.queue_http_error(404, {"detail": "not found"})
            read_opener.queue_http_error(404, {"detail": "not found"})
            write_opener = FakeOpener()
            write_opener.queue_json(
                {
                    "success": True,
                    "items": [
                        {
                            "success": True,
                            "document_id": "doc-123",
                            "payload_id": "doc-123",
                        }
                    ],
                }
            )
            write_opener.queue_json({"success": True, "stored": 1})
            write_opener.queue_json({"success": True, "stored": 1})
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=write_opener,
                ),
            )

        with self.assertRaises(LongTermRemoteUnavailableError):
            state.save_snapshot(snapshot_kind="objects", payload={"schema": "object_store", "objects": []})

        self.assertEqual(len(write_opener.calls), 2)

    def test_remote_snapshot_save_retries_after_transient_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                self._config(temp_dir),
                long_term_memory_remote_retry_attempts=2,
                long_term_memory_remote_retry_backoff_s=0.0,
            )
            read_opener = FakeOpener()
            read_opener.queue_json(
                {
                    "success": True,
                    "origin_uri": "twinr://longterm/test-namespace/objects",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "objects",
                            "updated_at": "2026-03-19T13:00:00+00:00",
                            "body": {"schema": "object_store", "objects": []},
                        }
                    ),
                }
            )
            write_opener = FakeOpener()
            write_opener.queue_exception(TimeoutError("timed out"))
            write_opener.queue_json({"success": True, "stored": 1})
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=write_opener,
                ),
            )

            state.save_snapshot(snapshot_kind="objects", payload={"schema": "object_store", "objects": []})

        self.assertEqual(len(write_opener.calls), 2)

    def test_remote_snapshot_save_extends_retries_for_queue_saturation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                self._config(temp_dir),
                long_term_memory_remote_retry_attempts=2,
                long_term_memory_remote_retry_backoff_s=0.0,
            )
            read_opener = FakeOpener()
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "doc-123",
                    "origin_uri": "twinr://longterm/test-namespace/objects",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "objects",
                            "updated_at": "2026-03-19T13:00:00+00:00",
                            "body": {"schema": "object_store", "objects": []},
                        }
                    ),
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "doc-pointer",
                    "origin_uri": "twinr://longterm/test-namespace/__pointer__:objects",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "__pointer__:objects",
                            "updated_at": "2026-03-19T13:00:01+00:00",
                            "body": {
                                "schema": "twinr_remote_snapshot_pointer_v1",
                                "version": 1,
                                "snapshot_kind": "objects",
                                "document_id": "doc-123",
                            },
                        }
                    ),
                }
            )
            write_opener = FakeOpener()
            for _ in range(4):
                write_opener.queue_http_error(
                    429,
                    {
                        "type": "about:blank",
                        "title": "RuntimeError",
                        "status": 429,
                        "detail": "queue_saturated",
                        "error_type": "RuntimeError",
                        "error": "queue_saturated",
                    },
                    headers={"Retry-After": "0"},
                )
            write_opener.queue_json({"success": True, "stored": 1})
            write_opener.queue_json({"success": True, "stored": 1})
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=write_opener,
                ),
            )

            state.save_snapshot(snapshot_kind="objects", payload={"schema": "object_store", "objects": []})

        self.assertEqual(len(write_opener.calls), 6)

    def test_remote_snapshot_save_aborts_retry_when_shutdown_requested(self) -> None:
        class _AbortAfterFirstWriteOpener(FakeOpener):
            def __init__(self, stop_event: Event) -> None:
                super().__init__()
                self._stop_event = stop_event

            def __call__(self, request, timeout: float):
                self._stop_event.set()
                return super().__call__(request, timeout)

        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                self._config(temp_dir),
                long_term_memory_remote_retry_attempts=4,
                long_term_memory_remote_retry_backoff_s=5.0,
            )
            stop_event = Event()
            read_opener = FakeOpener()
            write_opener = _AbortAfterFirstWriteOpener(stop_event)
            write_opener.queue_http_error(
                429,
                {
                    "type": "about:blank",
                    "title": "RuntimeError",
                    "status": 429,
                    "detail": "queue_saturated",
                    "error_type": "RuntimeError",
                    "error": "queue_saturated",
                },
                headers={"Retry-After": "0"},
            )
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=write_opener,
                ),
            )

            with self.assertRaises(LongTermOperationCancelledError):
                with longterm_operation_abort_scope(
                    should_abort=stop_event.is_set,
                    wait_for_abort=stop_event.wait,
                    label="test_remote_snapshot_save_aborts_retry_when_shutdown_requested",
                ):
                    state.save_snapshot(snapshot_kind="objects", payload={"schema": "object_store", "objects": []})

        self.assertEqual(len(write_opener.calls), 1)

    def test_remote_snapshot_save_retries_when_attested_readback_is_malformed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                self._config(temp_dir),
                long_term_memory_remote_retry_attempts=2,
                long_term_memory_remote_retry_backoff_s=0.0,
            )
            read_opener = FakeOpener()
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "doc-bad",
                    "chunk_count": 1,
                    "chunks": [
                        {
                            "payload_id": "doc-bad",
                            "content": "",
                            "content_summary": "",
                        }
                    ],
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "doc-bad",
                    "chunk_count": 1,
                    "chunks": [
                        {
                            "payload_id": "doc-bad",
                            "content": "",
                            "content_summary": "",
                        }
                    ],
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "doc-good",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "user_context",
                            "updated_at": "2026-03-19T13:00:01+00:00",
                            "body": {"schema": "twinr_managed_context", "version": 1, "entries": []},
                        }
                    ),
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "pointer-doc-good",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "__pointer__:user_context",
                            "updated_at": "2026-03-19T13:00:02+00:00",
                            "body": {
                                "schema": "twinr_remote_snapshot_pointer_v1",
                                "version": 1,
                                "snapshot_kind": "user_context",
                                "document_id": "doc-good",
                            },
                        }
                    ),
                }
            )
            write_opener = FakeOpener()
            write_opener.queue_json(
                {
                    "success": True,
                    "items": [
                        {
                            "success": True,
                            "document_id": "doc-bad",
                            "payload_id": "doc-bad",
                        }
                    ],
                }
            )
            write_opener.queue_json(
                {
                    "success": True,
                    "items": [
                        {
                            "success": True,
                            "document_id": "doc-good",
                            "payload_id": "doc-good",
                        }
                    ],
                }
            )
            write_opener.queue_json({"success": True, "stored": 1})
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=write_opener,
                ),
            )

            state.save_snapshot(
                snapshot_kind="user_context",
                payload={"schema": "twinr_managed_context", "version": 1, "entries": []},
            )

        self.assertEqual(len(write_opener.calls), 3)
        first_body = json.loads(write_opener.calls[0]["body"])
        second_body = json.loads(write_opener.calls[1]["body"])
        third_body = json.loads(write_opener.calls[2]["body"])
        self.assertEqual(first_body["items"][0]["payload"]["snapshot_kind"], "user_context")
        self.assertEqual(second_body["items"][0]["payload"]["snapshot_kind"], "user_context")
        self.assertEqual(third_body["items"][0]["payload"]["snapshot_kind"], "__pointer__:user_context")
        self.assertEqual(third_body["items"][0]["payload"]["body"]["document_id"], "doc-good")

    def test_remote_snapshot_save_raises_when_attested_readback_stays_malformed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                self._config(temp_dir),
                long_term_memory_remote_retry_attempts=2,
                long_term_memory_remote_retry_backoff_s=0.0,
            )
            read_opener = FakeOpener()
            for document_id in ("doc-bad-1", "doc-bad-1", "doc-bad-2", "doc-bad-2"):
                read_opener.queue_json(
                    {
                        "success": True,
                        "document_id": document_id,
                        "chunk_count": 1,
                        "chunks": [
                            {
                                "payload_id": document_id,
                                "content": "",
                                "content_summary": "",
                            }
                        ],
                    }
                )
            write_opener = FakeOpener()
            for document_id in ("doc-bad-1", "doc-bad-2"):
                write_opener.queue_json(
                    {
                        "success": True,
                        "items": [
                            {
                                "success": True,
                                "document_id": document_id,
                                "payload_id": document_id,
                            }
                        ],
                    }
                )
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=write_opener,
                ),
            )

            with self.assertRaises(LongTermRemoteUnavailableError):
                state.save_snapshot(
                    snapshot_kind="user_context",
                    payload={"schema": "twinr_managed_context", "version": 1, "entries": []},
                )

    def test_remote_snapshot_save_uses_async_write_and_retries_not_found_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                self._config(temp_dir),
                long_term_memory_remote_retry_attempts=2,
                long_term_memory_remote_retry_backoff_s=0.0,
            )
            payload = {"schema": "object_store", "objects": [{"memory_id": "fact:async"}]}
            read_opener = FakeOpener()
            read_opener.queue_http_error(404, {"detail": "not found"})
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "doc-async",
                    "origin_uri": "twinr://longterm/test-namespace/objects",
                    "chunk_count": 1,
                    "chunks": [
                        {
                            "payload_id": "doc-async",
                            "content": json.dumps(
                                {
                                    "schema": "twinr_remote_snapshot_v1",
                                    "namespace": "test-namespace",
                                    "snapshot_kind": "objects",
                                    "updated_at": "2026-03-19T14:30:00+00:00",
                                    "body": payload,
                                }
                            ),
                        }
                    ],
                }
            )
            read_opener.queue_http_error(404, {"detail": "not found"})
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "ptr-async",
                    "origin_uri": "twinr://longterm/test-namespace/__pointer__:objects",
                    "chunk_count": 1,
                    "chunks": [
                        {
                            "payload_id": "ptr-async",
                            "content": json.dumps(
                                {
                                    "schema": "twinr_remote_snapshot_v1",
                                    "namespace": "test-namespace",
                                    "snapshot_kind": "__pointer__:objects",
                                    "updated_at": "2026-03-19T14:30:01+00:00",
                                    "body": {
                                        "schema": "twinr_remote_snapshot_pointer_v1",
                                        "version": 1,
                                        "snapshot_kind": "objects",
                                        "document_id": "doc-async",
                                    },
                                }
                            ),
                        }
                    ],
                }
            )
            write_opener = FakeOpener()
            write_opener.queue_json(
                {
                    "success": True,
                    "job_id": "job-snapshot",
                    "status": "pending",
                    "items": 1,
                    "operation": "store_payload",
                    "execution_mode": "async",
                }
            )
            write_opener.queue_json(
                {
                    "success": True,
                    "job_id": "job-snapshot",
                    "status": "succeeded",
                    "result": {"success": True, "items": []},
                }
            )
            write_opener.queue_json(
                {
                    "success": True,
                    "job_id": "job-pointer",
                    "status": "pending",
                    "items": 1,
                    "operation": "store_payload",
                    "execution_mode": "async",
                }
            )
            write_opener.queue_json(
                {
                    "success": True,
                    "job_id": "job-pointer",
                    "status": "succeeded",
                    "result": {"success": True, "items": []},
                }
            )
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=write_opener,
                ),
            )

            state.save_snapshot(snapshot_kind="objects", payload=payload)

        self.assertEqual(state._document_id_hints["objects"], "doc-async")
        self.assertEqual(len(read_opener.calls), 4)
        self.assertEqual(len(write_opener.calls), 4)
        first_body = json.loads(write_opener.calls[0]["body"])
        second_body = json.loads(write_opener.calls[2]["body"])
        self.assertEqual(first_body["execution_mode"], "async")
        self.assertEqual(second_body["execution_mode"], "async")
        self.assertEqual(first_body["items"][0]["payload"]["snapshot_kind"], "objects")
        self.assertEqual(second_body["items"][0]["payload"]["snapshot_kind"], "__pointer__:objects")
        self.assertTrue(write_opener.calls[1]["full_url"].endswith("/v1/external/jobs/job-snapshot"))
        self.assertTrue(write_opener.calls[3]["full_url"].endswith("/v1/external/jobs/job-pointer"))

    def test_remote_snapshot_save_resolves_async_job_document_id_before_snapshot_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                self._config(temp_dir),
                long_term_memory_remote_retry_attempts=2,
                long_term_memory_remote_retry_backoff_s=0.0,
            )
            payload = {"schema": "archive_store", "items": []}
            read_opener = FakeOpener()
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "archive-doc-123",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "archive",
                            "updated_at": "2026-03-27T10:10:00+00:00",
                            "body": payload,
                        }
                    ),
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "ptr-archive-123",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "__pointer__:archive",
                            "updated_at": "2026-03-27T10:10:01+00:00",
                            "body": {
                                "schema": "twinr_remote_snapshot_pointer_v1",
                                "version": 1,
                                "snapshot_kind": "archive",
                                "document_id": "archive-doc-123",
                            },
                        }
                    ),
                }
            )
            write_opener = FakeOpener()
            write_opener.queue_json(
                {
                    "success": True,
                    "job_id": "job-archive",
                    "status": "pending",
                    "items": 1,
                    "operation": "store_payload",
                    "execution_mode": "async",
                }
            )
            write_opener.queue_json(
                {
                    "success": True,
                    "job_id": "job-archive",
                    "status": "succeeded",
                    "result": {
                        "success": True,
                        "items": [
                            {
                                "success": True,
                                "document_id": "archive-doc-123",
                                "payload_id": "archive-doc-123",
                            }
                        ],
                    },
                }
            )
            write_opener.queue_json(
                {
                    "success": True,
                    "items": [
                        {
                            "success": True,
                            "document_id": "ptr-archive-123",
                            "payload_id": "ptr-archive-123",
                        }
                    ],
                    "operation": "store_payload",
                    "execution_mode": "async",
                }
            )
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=write_opener,
                ),
            )

            state.save_snapshot(snapshot_kind="archive", payload=payload)

        self.assertEqual(state._document_id_hints["archive"], "archive-doc-123")
        self.assertEqual(len(read_opener.calls), 2)
        self.assertEqual(len(write_opener.calls), 3)
        first_query = parse_qs(urlparse(read_opener.calls[0]["full_url"]).query)
        second_query = parse_qs(urlparse(read_opener.calls[1]["full_url"]).query)
        self.assertEqual(first_query["document_id"], ["archive-doc-123"])
        self.assertEqual(second_query["document_id"], ["ptr-archive-123"])
        self.assertNotIn("origin_uri", first_query)
        self.assertNotIn("origin_uri", second_query)
        self.assertTrue(write_opener.calls[1]["full_url"].endswith("/v1/external/jobs/job-archive"))

    def test_remote_snapshot_save_falls_back_to_origin_attestation_when_async_job_status_times_out(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                self._config(temp_dir),
                long_term_memory_remote_retry_attempts=1,
                long_term_memory_remote_retry_backoff_s=0.0,
            )
            payload = {"schema": "object_store", "objects": [{"memory_id": "fact:new"}]}
            read_opener = FakeOpener()
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "new-doc",
                    "origin_uri": "twinr://longterm/test-namespace/objects",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "objects",
                            "updated_at": "2026-03-30T09:10:00+00:00",
                            "body": payload,
                        }
                    ),
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "ptr-new",
                    "origin_uri": "twinr://longterm/test-namespace/__pointer__:objects",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "__pointer__:objects",
                            "updated_at": "2026-03-30T09:10:01+00:00",
                            "body": {
                                "schema": "twinr_remote_snapshot_pointer_v1",
                                "version": 1,
                                "snapshot_kind": "objects",
                                "document_id": "new-doc",
                            },
                        }
                    ),
                }
            )
            write_opener = FakeOpener()
            write_opener.queue_json(
                {
                    "success": True,
                    "job_id": "job-snapshot",
                    "status": "pending",
                    "items": 1,
                    "operation": "store_payload",
                    "execution_mode": "async",
                }
            )
            write_opener.queue_exception(socket.timeout("job status slow"))
            write_opener.queue_json(
                {
                    "success": True,
                    "job_id": "job-pointer",
                    "status": "pending",
                    "items": 1,
                    "operation": "store_payload",
                    "execution_mode": "async",
                }
            )
            write_opener.queue_exception(socket.timeout("job status slow"))
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=write_opener,
                ),
            )

            with patch.object(state, "_async_job_visibility_timeout_s", return_value=0.01):
                state.save_snapshot(snapshot_kind="objects", payload=payload)

        self.assertEqual(state._document_id_hints["objects"], "new-doc")
        self.assertEqual(len(read_opener.calls), 2)
        self.assertEqual(len(write_opener.calls), 4)
        first_query = parse_qs(urlparse(read_opener.calls[0]["full_url"]).query)
        second_query = parse_qs(urlparse(read_opener.calls[1]["full_url"]).query)
        self.assertEqual(first_query["origin_uri"], ["twinr://longterm/test-namespace/objects"])
        self.assertEqual(second_query["origin_uri"], ["twinr://longterm/test-namespace/__pointer__%3Aobjects"])
        self.assertNotIn("document_id", first_query)
        self.assertNotIn("document_id", second_query)
        self.assertTrue(write_opener.calls[1]["full_url"].endswith("/v1/external/jobs/job-snapshot"))
        self.assertTrue(write_opener.calls[3]["full_url"].endswith("/v1/external/jobs/job-pointer"))

    def test_remote_snapshot_save_retries_stale_async_origin_attestation_without_rewriting_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                self._config(temp_dir),
                long_term_memory_remote_retry_attempts=2,
                long_term_memory_remote_retry_backoff_s=0.0,
            )
            payload = {"schema": "object_store", "objects": [{"memory_id": "fact:new"}]}
            stale_payload = {"schema": "object_store", "objects": [{"memory_id": "fact:old"}]}
            read_opener = FakeOpener()
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "old-doc",
                    "origin_uri": "twinr://longterm/test-namespace/objects",
                    "chunk_count": 1,
                    "chunks": [
                        {
                            "payload_id": "old-doc",
                            "content": json.dumps(
                                {
                                    "schema": "twinr_remote_snapshot_v1",
                                    "namespace": "test-namespace",
                                    "snapshot_kind": "objects",
                                    "updated_at": "2026-03-20T18:00:00+00:00",
                                    "body": stale_payload,
                                }
                            ),
                        }
                    ],
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "old-doc",
                    "origin_uri": "twinr://longterm/test-namespace/objects",
                    "chunk_count": 2,
                    "chunks": [
                        {
                            "payload_id": "old-doc",
                            "content": json.dumps(
                                {
                                    "schema": "twinr_remote_snapshot_v1",
                                    "namespace": "test-namespace",
                                    "snapshot_kind": "objects",
                                    "updated_at": "2026-03-20T18:00:00+00:00",
                                    "body": stale_payload,
                                }
                            ),
                        },
                        {
                            "payload_id": "new-doc",
                            "content": json.dumps(
                                {
                                    "schema": "twinr_remote_snapshot_v1",
                                    "namespace": "test-namespace",
                                    "snapshot_kind": "objects",
                                    "updated_at": "2026-03-20T18:00:01+00:00",
                                    "body": payload,
                                }
                            ),
                        },
                    ],
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "ptr-new",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "__pointer__:objects",
                            "updated_at": "2026-03-20T18:00:02+00:00",
                            "body": {
                                "schema": "twinr_remote_snapshot_pointer_v1",
                                "version": 1,
                                "snapshot_kind": "objects",
                                "document_id": "new-doc",
                            },
                        }
                    ),
                }
            )
            write_opener = FakeOpener()
            write_opener.queue_json(
                {
                    "success": True,
                    "job_id": "job-snapshot",
                    "status": "pending",
                    "items": 1,
                    "operation": "store_payload",
                    "execution_mode": "async",
                }
            )
            write_opener.queue_json(
                {
                    "success": True,
                    "job_id": "job-snapshot",
                    "status": "succeeded",
                    "result": {"success": True, "items": []},
                }
            )
            write_opener.queue_json(
                {
                    "success": True,
                    "items": [
                        {
                            "success": True,
                            "document_id": "ptr-new",
                            "payload_id": "ptr-new",
                        }
                    ],
                    "operation": "store_payload",
                    "execution_mode": "async",
                }
            )
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=write_opener,
                ),
            )

            state.save_snapshot(snapshot_kind="objects", payload=payload)

        self.assertEqual(state._document_id_hints["objects"], "new-doc")
        self.assertEqual(len(write_opener.calls), 3)
        self.assertEqual(len(read_opener.calls), 3)
        first_body = json.loads(write_opener.calls[0]["body"])
        second_body = json.loads(write_opener.calls[2]["body"])
        self.assertEqual(first_body["items"][0]["payload"]["snapshot_kind"], "objects")
        self.assertEqual(second_body["items"][0]["payload"]["snapshot_kind"], "__pointer__:objects")
        self.assertEqual(second_body["items"][0]["payload"]["body"]["document_id"], "new-doc")
        self.assertTrue(write_opener.calls[1]["full_url"].endswith("/v1/external/jobs/job-snapshot"))

    def test_remote_snapshot_save_waits_through_multiple_stale_async_origin_reads_without_rewriting_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                self._config(temp_dir),
                long_term_memory_remote_retry_attempts=2,
                long_term_memory_remote_retry_backoff_s=0.0,
            )
            payload = {"schema": "object_store", "objects": [{"memory_id": "fact:new"}]}
            stale_payload = {"schema": "object_store", "objects": [{"memory_id": "fact:old"}]}
            read_opener = FakeOpener()
            for _ in range(3):
                read_opener.queue_json(
                    {
                        "success": True,
                        "document_id": "old-doc",
                        "origin_uri": "twinr://longterm/test-namespace/objects",
                        "chunk_count": 1,
                        "chunks": [
                            {
                                "payload_id": "old-doc",
                                "content": json.dumps(
                                    {
                                        "schema": "twinr_remote_snapshot_v1",
                                        "namespace": "test-namespace",
                                        "snapshot_kind": "objects",
                                        "updated_at": "2026-03-20T18:00:00+00:00",
                                        "body": stale_payload,
                                    }
                                ),
                            }
                        ],
                    }
                )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "old-doc",
                    "origin_uri": "twinr://longterm/test-namespace/objects",
                    "chunk_count": 2,
                    "chunks": [
                        {
                            "payload_id": "old-doc",
                            "content": json.dumps(
                                {
                                    "schema": "twinr_remote_snapshot_v1",
                                    "namespace": "test-namespace",
                                    "snapshot_kind": "objects",
                                    "updated_at": "2026-03-20T18:00:00+00:00",
                                    "body": stale_payload,
                                }
                            ),
                        },
                        {
                            "payload_id": "new-doc",
                            "content": json.dumps(
                                {
                                    "schema": "twinr_remote_snapshot_v1",
                                    "namespace": "test-namespace",
                                    "snapshot_kind": "objects",
                                    "updated_at": "2026-03-20T18:00:01+00:00",
                                    "body": payload,
                                }
                            ),
                        },
                    ],
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "ptr-new",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "__pointer__:objects",
                            "updated_at": "2026-03-20T18:00:02+00:00",
                            "body": {
                                "schema": "twinr_remote_snapshot_pointer_v1",
                                "version": 1,
                                "snapshot_kind": "objects",
                                "document_id": "new-doc",
                            },
                        }
                    ),
                }
            )
            write_opener = FakeOpener()
            write_opener.queue_json(
                {
                    "success": True,
                    "job_id": "job-snapshot",
                    "status": "pending",
                    "items": 1,
                    "operation": "store_payload",
                    "execution_mode": "async",
                }
            )
            write_opener.queue_json(
                {
                    "success": True,
                    "job_id": "job-snapshot",
                    "status": "succeeded",
                    "result": {"success": True, "items": []},
                }
            )
            write_opener.queue_json(
                {
                    "success": True,
                    "items": [
                        {
                            "success": True,
                            "document_id": "ptr-new",
                            "payload_id": "ptr-new",
                        }
                    ],
                    "operation": "store_payload",
                    "execution_mode": "async",
                }
            )
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=write_opener,
                ),
            )

            state.save_snapshot(snapshot_kind="objects", payload=payload)

        self.assertEqual(state._document_id_hints["objects"], "new-doc")
        self.assertEqual(len(write_opener.calls), 3)
        self.assertEqual(len(read_opener.calls), 5)
        self.assertTrue(write_opener.calls[1]["full_url"].endswith("/v1/external/jobs/job-snapshot"))

    def test_remote_snapshot_save_escalates_deferred_midterm_attestation_to_job_status_after_stale_origin_mismatch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                self._config(temp_dir),
                long_term_memory_remote_retry_attempts=1,
                long_term_memory_remote_retry_backoff_s=0.0,
            )
            payload = {
                "schema": "twinr_memory_midterm_catalog_v1",
                "version": 1,
                "items_count": 1,
                "segments": [
                    {
                        "uri": "https://example.invalid/midterm/catalog/segment/0000/token",
                        "items_count": 1,
                    }
                ],
                "written_at": "2026-04-09T07:45:39+00:00",
            }
            stale_payload = {
                "schema": "twinr_memory_midterm_catalog_v1",
                "version": 1,
                "items_count": 1,
                "segments": [
                    {
                        "uri": "https://example.invalid/midterm/catalog/segment/0000/old-token",
                        "items_count": 1,
                    }
                ],
                "written_at": "2026-04-09T07:45:30+00:00",
            }
            read_opener = FakeOpener()
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "old-doc",
                    "origin_uri": "twinr://longterm/test-namespace/midterm",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "midterm",
                            "updated_at": "2026-04-09T07:45:30+00:00",
                            "body": stale_payload,
                        }
                    ),
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "new-doc",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "midterm",
                            "updated_at": "2026-04-09T07:45:39+00:00",
                            "body": payload,
                        }
                    ),
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "ptr-new",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "__pointer__:midterm",
                            "updated_at": "2026-04-09T07:45:40+00:00",
                            "body": {
                                "schema": "twinr_remote_snapshot_pointer_v1",
                                "version": 1,
                                "snapshot_kind": "midterm",
                                "document_id": "new-doc",
                            },
                        }
                    ),
                }
            )
            write_opener = FakeOpener()
            write_opener.queue_json(
                {
                    "success": True,
                    "job_id": "job-snapshot",
                    "status": "pending",
                    "items": 1,
                    "operation": "store_payload",
                    "execution_mode": "async",
                }
            )
            write_opener.queue_json(
                {
                    "success": True,
                    "job_id": "job-snapshot",
                    "status": "succeeded",
                    "result": {
                        "success": True,
                        "items": [{"document_id": "new-doc"}],
                    },
                }
            )
            write_opener.queue_json(
                {
                    "success": True,
                    "items": [
                        {
                            "success": True,
                            "document_id": "ptr-new",
                            "payload_id": "ptr-new",
                        }
                    ],
                    "operation": "store_payload",
                    "execution_mode": "async",
                }
            )
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=write_opener,
                ),
                namespace="test-namespace",
            )

            with patch.object(state, "_async_attestation_visibility_timeout_s", return_value=0.01):
                state.save_snapshot(
                    snapshot_kind="midterm",
                    payload=payload,
                    skip_async_document_id_wait=True,
                )

        self.assertEqual(state._document_id_hints["midterm"], "new-doc")
        self.assertEqual(len(write_opener.calls), 3)
        self.assertEqual(len(read_opener.calls), 3)
        first_query = parse_qs(urlparse(read_opener.calls[0]["full_url"]).query)
        second_query = parse_qs(urlparse(read_opener.calls[1]["full_url"]).query)
        self.assertEqual(first_query["origin_uri"], ["twinr://longterm/test-namespace/midterm"])
        self.assertNotIn("document_id", first_query)
        self.assertEqual(second_query["document_id"], ["new-doc"])
        self.assertNotIn("origin_uri", second_query)
        self.assertTrue(write_opener.calls[1]["full_url"].endswith("/v1/external/jobs/job-snapshot"))

    def test_remote_snapshot_save_waits_for_exact_document_visibility_before_rewriting_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                self._config(temp_dir),
                long_term_memory_remote_retry_attempts=2,
                long_term_memory_remote_retry_backoff_s=0.0,
            )
            payload = {"schema": "object_store", "objects": [{"memory_id": "fact:new"}]}
            stale_payload = {"schema": "object_store", "objects": [{"memory_id": "fact:old"}]}
            read_opener = FakeOpener()
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "doc-async",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "objects",
                            "updated_at": "2026-03-20T18:00:00+00:00",
                            "body": stale_payload,
                        }
                    ),
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "doc-async",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "objects",
                            "updated_at": "2026-03-20T18:00:01+00:00",
                            "body": payload,
                        }
                    ),
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "ptr-new",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "__pointer__:objects",
                            "updated_at": "2026-03-20T18:00:02+00:00",
                            "body": {
                                "schema": "twinr_remote_snapshot_pointer_v1",
                                "version": 1,
                                "snapshot_kind": "objects",
                                "document_id": "doc-async",
                            },
                        }
                    ),
                }
            )
            write_opener = FakeOpener()
            write_opener.queue_json(
                {
                    "success": True,
                    "items": [
                        {
                            "success": True,
                            "document_id": "doc-async",
                            "payload_id": "doc-async",
                        }
                    ],
                }
            )
            write_opener.queue_json(
                {
                    "success": True,
                    "items": [
                        {
                            "success": True,
                            "document_id": "ptr-new",
                            "payload_id": "ptr-new",
                        }
                    ],
                }
            )
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=write_opener,
                ),
            )

            state.save_snapshot(snapshot_kind="objects", payload=payload)

        self.assertEqual(state._document_id_hints["objects"], "doc-async")
        self.assertEqual(len(write_opener.calls), 2)
        self.assertEqual(len(read_opener.calls), 3)

    def test_remote_snapshot_save_waits_for_exact_pointer_visibility_before_rewriting_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                self._config(temp_dir),
                long_term_memory_remote_retry_attempts=2,
                long_term_memory_remote_retry_backoff_s=0.0,
            )
            payload = {"schema": "object_store", "objects": [{"memory_id": "fact:new"}]}
            read_opener = FakeOpener()
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "new-doc",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "objects",
                            "updated_at": "2026-03-20T18:10:00+00:00",
                            "body": payload,
                        }
                    ),
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "ptr-new",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "__pointer__:objects",
                            "updated_at": "2026-03-20T18:09:59+00:00",
                            "body": {
                                "schema": "twinr_remote_snapshot_pointer_v1",
                                "version": 1,
                                "snapshot_kind": "objects",
                                "document_id": "old-doc",
                            },
                        }
                    ),
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "ptr-new",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "__pointer__:objects",
                            "updated_at": "2026-03-20T18:10:01+00:00",
                            "body": {
                                "schema": "twinr_remote_snapshot_pointer_v1",
                                "version": 1,
                                "snapshot_kind": "objects",
                                "document_id": "new-doc",
                            },
                        }
                    ),
                }
            )
            write_opener = FakeOpener()
            write_opener.queue_json(
                {
                    "success": True,
                    "items": [
                        {
                            "success": True,
                            "document_id": "new-doc",
                            "payload_id": "new-doc",
                        }
                    ],
                }
            )
            write_opener.queue_json(
                {
                    "success": True,
                    "items": [
                        {
                            "success": True,
                            "document_id": "ptr-new",
                            "payload_id": "ptr-new",
                        }
                    ],
                }
            )
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=write_opener,
                ),
            )

            state.save_snapshot(snapshot_kind="objects", payload=payload)

        self.assertEqual(state._document_id_hints["objects"], "new-doc")
        self.assertEqual(len(write_opener.calls), 2)
        self.assertEqual(len(read_opener.calls), 3)

    def test_remote_snapshot_save_retries_stale_async_pointer_attestation_without_rewriting_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                self._config(temp_dir),
                long_term_memory_remote_retry_attempts=2,
                long_term_memory_remote_retry_backoff_s=0.0,
            )
            payload = {"schema": "object_store", "objects": [{"memory_id": "fact:new"}]}
            read_opener = FakeOpener()
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "new-doc",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "objects",
                            "updated_at": "2026-03-20T18:10:00+00:00",
                            "body": payload,
                        }
                    ),
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "ptr-old",
                    "origin_uri": "twinr://longterm/test-namespace/__pointer__:objects",
                    "chunk_count": 1,
                    "chunks": [
                        {
                            "payload_id": "ptr-old",
                            "content": json.dumps(
                                {
                                    "schema": "twinr_remote_snapshot_v1",
                                    "namespace": "test-namespace",
                                    "snapshot_kind": "__pointer__:objects",
                                    "updated_at": "2026-03-20T18:09:59+00:00",
                                    "body": {
                                        "schema": "twinr_remote_snapshot_pointer_v1",
                                        "version": 1,
                                        "snapshot_kind": "objects",
                                        "document_id": "old-doc",
                                    },
                                }
                            ),
                        }
                    ],
                }
            )
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "ptr-old",
                    "origin_uri": "twinr://longterm/test-namespace/__pointer__:objects",
                    "chunk_count": 2,
                    "chunks": [
                        {
                            "payload_id": "ptr-old",
                            "content": json.dumps(
                                {
                                    "schema": "twinr_remote_snapshot_v1",
                                    "namespace": "test-namespace",
                                    "snapshot_kind": "__pointer__:objects",
                                    "updated_at": "2026-03-20T18:09:59+00:00",
                                    "body": {
                                        "schema": "twinr_remote_snapshot_pointer_v1",
                                        "version": 1,
                                        "snapshot_kind": "objects",
                                        "document_id": "old-doc",
                                    },
                                }
                            ),
                        },
                        {
                            "payload_id": "ptr-new",
                            "content": json.dumps(
                                {
                                    "schema": "twinr_remote_snapshot_v1",
                                    "namespace": "test-namespace",
                                    "snapshot_kind": "__pointer__:objects",
                                    "updated_at": "2026-03-20T18:10:01+00:00",
                                    "body": {
                                        "schema": "twinr_remote_snapshot_pointer_v1",
                                        "version": 1,
                                        "snapshot_kind": "objects",
                                        "document_id": "new-doc",
                                    },
                                }
                            ),
                        },
                    ],
                }
            )
            write_opener = FakeOpener()
            write_opener.queue_json(
                {
                    "success": True,
                    "items": [
                        {
                            "success": True,
                            "document_id": "new-doc",
                            "payload_id": "new-doc",
                        }
                    ],
                    "operation": "store_payload",
                    "execution_mode": "sync",
                }
            )
            write_opener.queue_json(
                {
                    "success": True,
                    "job_id": "job-pointer",
                    "status": "pending",
                    "items": 1,
                    "operation": "store_payload",
                    "execution_mode": "async",
                }
            )
            write_opener.queue_json(
                {
                    "success": True,
                    "job_id": "job-pointer",
                    "status": "succeeded",
                    "result": {"success": True, "items": []},
                }
            )
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=write_opener,
                ),
            )

            state.save_snapshot(snapshot_kind="objects", payload=payload)

        self.assertEqual(state._document_id_hints["objects"], "new-doc")
        self.assertEqual(len(write_opener.calls), 3)
        self.assertEqual(len(read_opener.calls), 3)
        pointer_body = json.loads(write_opener.calls[1]["body"])
        self.assertEqual(pointer_body["execution_mode"], "async")
        self.assertEqual(pointer_body["items"][0]["payload"]["snapshot_kind"], "__pointer__:objects")
        self.assertEqual(pointer_body["items"][0]["payload"]["body"]["document_id"], "new-doc")
        self.assertTrue(write_opener.calls[2]["full_url"].endswith("/v1/external/jobs/job-pointer"))

    def test_remote_snapshot_save_rejects_item_level_store_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                self._config(temp_dir),
                long_term_memory_remote_retry_attempts=1,
                long_term_memory_remote_retry_backoff_s=0.0,
            )
            read_opener = FakeOpener()
            write_opener = FakeOpener()
            write_opener.queue_json(
                {
                    "success": True,
                    "all_succeeded": False,
                    "count": 1,
                    "succeeded": 0,
                    "failed": 1,
                    "items": [
                        {
                            "success": False,
                            "error": "content required",
                            "error_type": "ValueError",
                            "item_index": 0,
                        }
                    ],
                    "operation": "store_payload",
                    "execution_mode": "sync",
                }
            )
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=write_opener,
                ),
            )

            with self.assertRaises(LongTermRemoteUnavailableError):
                state.save_snapshot(snapshot_kind="objects", payload={"schema": "object_store", "objects": []})

        self.assertEqual(read_opener.calls, [])

    def test_remote_snapshot_save_logs_dns_write_diagnostic_on_final_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                self._config(temp_dir),
                long_term_memory_remote_retry_attempts=1,
                long_term_memory_remote_retry_backoff_s=0.0,
                long_term_memory_remote_write_timeout_s=17.0,
            )
            write_opener = FakeOpener()
            write_opener.queue_exception(socket.gaierror(-3, "Temporary failure in name resolution"))
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=FakeOpener(),
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=write_opener,
                ),
            )

            with self.assertRaises(LongTermRemoteUnavailableError):
                state.save_snapshot(snapshot_kind="objects", payload={"schema": "object_store", "objects": []})

            events = TwinrOpsEventStore.from_project_root(temp_dir).tail(limit=5)

        self.assertTrue(events)
        last_event = events[-1]
        self.assertEqual(last_event["event"], "longterm_remote_write_failed")
        self.assertEqual(last_event["level"], "warning")
        self.assertIn("dns_resolution_error", last_event["message"])
        data = dict(last_event["data"])
        self.assertEqual(data["request_kind"], "write")
        self.assertEqual(data["snapshot_kind"], "objects")
        self.assertEqual(data["operation"], "store_snapshot_record")
        self.assertEqual(data["classification"], "dns_resolution_error")
        self.assertEqual(data["attempt_count"], 1)
        self.assertEqual(data["request_item_count"], 1)
        self.assertEqual(data["write_timeout_s"], 17.0)
        self.assertEqual(data["error_type"], "ChonkyDBError")
        self.assertEqual(data["root_cause_type"], "gaierror")
        self.assertEqual(data["request_execution_mode"], "async")
        self.assertTrue(data["request_correlation_id"])
        self.assertGreater(data["request_bytes"], 0)
        self.assertIsNotNone(data["store_transport_ms"])
        self.assertIsNone(data["async_job_wait_ms"])

    def test_remote_snapshot_timeout_records_endpoint_payload_type_and_histogram(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                self._config(temp_dir),
                long_term_memory_remote_retry_attempts=1,
                long_term_memory_remote_retry_backoff_s=0.0,
            )
            read_opener = FakeOpener()
            read_opener.queue_exception(TimeoutError("Read operation timed out"))
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=FakeOpener(),
                ),
            )
            state._remember_snapshot_document_id(snapshot_kind="archive", document_id="archive-doc-1")

            with self.assertRaises(LongTermRemoteUnavailableError):
                state.load_snapshot(snapshot_kind="archive")

            events = TwinrOpsEventStore.from_project_root(temp_dir).tail(limit=5)
            histogram_path = Path(temp_dir) / "artifacts" / "stores" / "ops" / "longterm_remote_read_histograms.json"
            payload = json.loads(histogram_path.read_text(encoding="utf-8"))

        failed_event = next(event for event in events if event["event"] == "longterm_remote_read_failed")
        data = dict(failed_event["data"])
        self.assertEqual(data["snapshot_kind"], "archive")
        self.assertEqual(data["operation"], "snapshot_load")
        self.assertEqual(data["classification"], "timeout")
        self.assertEqual(data["request_method"], "GET")
        self.assertEqual(data["request_path"], "/v1/external/documents/full")
        self.assertEqual(data["request_endpoint"], "GET /v1/external/documents/full")
        self.assertEqual(data["request_payload_kind"], "document_id_cached_head")
        self.assertEqual(data["access_classification"], "legacy_snapshot_compat")
        self.assertEqual(data["timeout_reason"], "read_operation_timed_out")
        operations = dict(payload.get("operations") or {})
        entry = dict(operations["archive:snapshot_load"])
        self.assertEqual(entry["last_request_endpoint"], "GET /v1/external/documents/full")
        self.assertEqual(entry["last_request_payload_kind"], "document_id_cached_head")
        self.assertEqual(entry["last_request_kind"], "read")
        self.assertEqual(entry["last_access_classification"], "legacy_snapshot_compat")
        self.assertEqual(entry["last_classification"], "timeout")
        self.assertEqual(dict(entry["request_endpoint_counts"])["GET /v1/external/documents/full"], 1)
        self.assertEqual(dict(entry["request_payload_kind_counts"])["document_id_cached_head"], 1)
        self.assertEqual(dict(entry["request_kind_counts"])["read"], 1)
        self.assertEqual(dict(entry["access_classification_counts"])["legacy_snapshot_compat"], 1)

    def test_remote_snapshot_loads_from_documents_full_chunk_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(temp_dir)
            read_opener = FakeOpener()
            read_opener.queue_http_error(404, {"detail": "not found"})
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "doc-1",
                    "origin_uri": "twinr://longterm/test-namespace/objects",
                    "chunk_count": 1,
                    "chunks": [
                        {
                            "payload_id": "chunk-1",
                            "metadata": {
                                "twinr_namespace": "test-namespace",
                                "twinr_snapshot_kind": "objects",
                                "origin_uri": "twinr://longterm/test-namespace/objects",
                            },
                            "content": json.dumps(
                                {
                                    "schema": "twinr_remote_snapshot_v1",
                                    "namespace": "test-namespace",
                                    "snapshot_kind": "objects",
                                    "body": {"schema": "object_store", "objects": [{"memory_id": "fact:2"}]},
                                }
                            ),
                        }
                    ],
                    "origin_scan_mode": "origin_lookup_index",
                    "origin_lookup_hits": 1,
                }
            )
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=FakeOpener(),
                ),
            )

            payload = state.load_snapshot(snapshot_kind="objects", local_path=Path(temp_dir) / "objects.json")

        self.assertEqual(payload, {"schema": "object_store", "objects": [{"memory_id": "fact:2"}]})

    def test_remote_snapshot_load_retries_after_transient_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                self._config(temp_dir),
                long_term_memory_remote_retry_attempts=2,
                long_term_memory_remote_retry_backoff_s=0.0,
            )
            read_opener = FakeOpener()
            read_opener.queue_http_error(404, {"detail": "not found"})
            read_opener.queue_exception(TimeoutError("timed out"))
            read_opener.queue_json(
                {
                    "origin_uri": "twinr://longterm/test-namespace/objects",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "objects",
                            "body": {"schema": "object_store", "objects": [{"memory_id": "fact:retry"}]},
                        }
                    ),
                }
            )
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=FakeOpener(),
                ),
            )

            payload = state.load_snapshot(snapshot_kind="objects", local_path=Path(temp_dir) / "objects.json")

        self.assertEqual(payload, {"schema": "object_store", "objects": [{"memory_id": "fact:retry"}]})
        self.assertEqual(len(read_opener.calls), 3)

    def test_remote_snapshot_uses_configured_max_content_chars(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                self._config(temp_dir),
                long_term_memory_remote_max_content_chars=7654321,
            )
            local_path = Path(temp_dir) / "state" / "chonkydb" / "twinr_memory_objects_v1.json"
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text(json.dumps({"schema": "object_store", "objects": []}), encoding="utf-8")
            read_opener = FakeOpener()
            read_opener.queue_http_error(404, {"detail": "not found"})
            read_opener.queue_json(
                {
                    "origin_uri": "twinr://longterm/test-namespace/objects",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "objects",
                            "body": {"schema": "object_store", "objects": []},
                        }
                    ),
                }
            )
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=FakeOpener(),
                ),
            )

            payload = state.load_snapshot(snapshot_kind="objects", local_path=local_path)

        self.assertEqual(payload, {"schema": "object_store", "objects": []})
        query = parse_qs(urlparse(read_opener.calls[1]["full_url"]).query)
        self.assertEqual(query["max_content_chars"], ["7654321"])

    def test_remote_snapshot_ensure_seeds_missing_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(temp_dir)
            read_opener = FakeOpener()
            read_opener.queue_http_error(404, {"detail": "not found"})
            read_opener.queue_http_error(404, {"detail": "not found"})
            read_opener.queue_json(
                {
                    "success": True,
                    "origin_uri": "twinr://longterm/test-namespace/user_context",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "user_context",
                            "updated_at": "2026-03-19T13:00:00+00:00",
                            "body": {"schema": "twinr_managed_context", "version": 1, "entries": []},
                        }
                    ),
                }
            )
            write_opener = FakeOpener()
            write_opener.queue_json({"success": True, "stored": 1})
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=write_opener,
                ),
            )

            created = state.ensure_snapshot(
                snapshot_kind="user_context",
                payload={"schema": "twinr_managed_context", "version": 1, "entries": []},
            )

        self.assertTrue(created)
        write_body = json.loads(write_opener.calls[0]["body"])
        self.assertEqual(write_body["items"][0]["payload"]["snapshot_kind"], "user_context")

    def test_remote_snapshot_migrates_from_local_when_remote_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(temp_dir)
            local_path = Path(temp_dir) / "state" / "chonkydb" / "twinr_memory_midterm_v1.json"
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_payload = {"schema": "midterm_store", "packets": [{"packet_id": "midterm:1"}]}
            local_path.write_text(json.dumps(local_payload), encoding="utf-8")

            read_opener = FakeOpener()
            read_opener.queue_http_error(404, {"detail": "not found"})
            read_opener.queue_http_error(404, {"detail": "not found"})
            read_opener.queue_json(
                {
                    "success": True,
                    "origin_uri": "twinr://longterm/test-namespace/midterm",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "midterm",
                            "updated_at": "2026-03-19T13:00:00+00:00",
                            "body": local_payload,
                        }
                    ),
                }
            )
            write_opener = FakeOpener()
            write_opener.queue_json({"success": True, "stored": 1})
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=write_opener,
                ),
            )

            payload = state.load_snapshot(snapshot_kind="midterm", local_path=local_path)

        self.assertEqual(payload, local_payload)
        write_body = json.loads(write_opener.calls[0]["body"])
        self.assertEqual(write_body["items"][0]["payload"]["snapshot_kind"], "midterm")
        self.assertEqual(write_body["items"][0]["payload"]["body"], local_payload)

    def test_remote_snapshot_load_does_not_warn_for_probe_path_outside_root_when_remote_payload_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(temp_dir)
            outside_dir = Path(temp_dir).parent / f"{Path(temp_dir).name}_outside"
            outside_dir.mkdir(parents=True, exist_ok=True)
            outside_path = outside_dir / "twinr_memory_midterm_v1.json"
            read_opener = FakeOpener()
            read_opener.queue_http_error(404, {"detail": "not found"})
            read_opener.queue_json(
                {
                    "origin_uri": "twinr://longterm/test-namespace/midterm",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "midterm",
                            "body": {"schema": "twinr_memory_midterm_store", "version": 1, "packets": []},
                        }
                    ),
                }
            )
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=FakeOpener(),
                ),
            )

            with self.assertNoLogs("twinr.memory.longterm.storage.remote_state", level=logging.WARNING):
                payload = state.load_snapshot(snapshot_kind="midterm", local_path=outside_path)

        self.assertEqual(payload, {"schema": "twinr_memory_midterm_store", "version": 1, "packets": []})

    def test_remote_snapshot_warns_clearly_when_local_fallback_path_is_outside_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                self._config(temp_dir),
                long_term_memory_remote_required=False,
                long_term_memory_migration_enabled=False,
            )
            outside_dir = Path(temp_dir).parent / f"{Path(temp_dir).name}_outside"
            outside_dir.mkdir(parents=True, exist_ok=True)
            outside_path = outside_dir / "twinr_memory_objects_v1.json"
            read_opener = FakeOpener()
            read_opener.queue_http_error(404, {"detail": "not found"})
            read_opener.queue_http_error(404, {"detail": "not found"})
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=FakeOpener(),
                ),
            )

            with self.assertLogs("twinr.memory.longterm.storage.remote_state", level=logging.WARNING) as captured:
                payload = state.load_snapshot(snapshot_kind="objects", local_path=outside_path)

        self.assertIsNone(payload)
        self.assertTrue(
            any(
                "caller supplied a path outside the configured Twinr memory root" in message
                and "not corrupted memory data" in message
                for message in captured.output
            )
        )

    def test_remote_snapshot_raises_when_migration_write_times_out_in_required_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                self._config(temp_dir),
                long_term_memory_remote_retry_attempts=1,
            )
            local_path = Path(temp_dir) / "state" / "chonkydb" / "twinr_memory_objects_v1.json"
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_payload = {"schema": "object_store", "objects": [{"memory_id": "fact:local"}]}
            local_path.write_text(json.dumps(local_payload), encoding="utf-8")

            read_opener = FakeOpener()
            read_opener.queue_http_error(404, {"detail": "not found"})
            read_opener.queue_http_error(404, {"detail": "not found"})
            write_opener = FakeOpener()
            write_opener.queue_exception(TimeoutError("timed out"))
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=write_opener,
                ),
            )

            with self.assertRaisesRegex(RuntimeError, "Failed to write remote long-term snapshot 'objects'"):
                state.load_snapshot(snapshot_kind="objects", local_path=local_path)

        self.assertEqual(len(read_opener.calls), 2)
        self.assertEqual(len(write_opener.calls), 1)

    def test_remote_snapshot_does_not_fallback_to_local_cache_when_migration_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir) / "state" / "chonkydb" / "twinr_memory_objects_v1.json"
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text(
                json.dumps({"schema": "object_store", "objects": [{"memory_id": "fact:local"}]}),
                encoding="utf-8",
            )
            config = TwinrConfig(
                project_root=temp_dir,
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_remote_required=False,
                long_term_memory_migration_enabled=False,
                long_term_memory_remote_namespace="test-namespace",
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                chonkydb_base_url="https://memory.test",
                chonkydb_api_key="secret-key",
            )
            read_opener = FakeOpener()
            read_opener.queue_http_error(404, {"detail": "not found"})
            read_opener.queue_http_error(404, {"detail": "not found"})
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=FakeOpener(),
                ),
            )

            payload = state.load_snapshot(snapshot_kind="objects", local_path=local_path)

        self.assertIsNone(payload)
        self.assertEqual(len(read_opener.calls), 2)

    def test_remote_snapshot_returns_none_when_uri_is_missing_and_no_local_cache_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(temp_dir)
            read_opener = FakeOpener()
            read_opener.queue_http_error(404, {"detail": "not found"})
            read_opener.queue_http_error(404, {"detail": "not found"})
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=FakeOpener(),
                ),
            )

            payload = state.load_snapshot(snapshot_kind="objects", local_path=Path(temp_dir) / "missing.json")

        self.assertIsNone(payload)
        self.assertEqual(len(read_opener.calls), 2)

    def test_remote_namespace_is_derived_when_not_explicitly_configured(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
            )
            state = LongTermRemoteStateStore.from_config(config)

        self.assertIsNotNone(state.namespace)
        self.assertTrue(str(state.namespace).startswith("twinr_longterm_v1:"))

    def test_status_reserves_liveness_budget_from_extended_probe_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            opener = FakeOpener()
            opener.queue_json({"success": True, "service": "ccodex_memory", "ready": True, "auth_enabled": True})
            config = TwinrConfig(
                project_root=temp_dir,
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_remote_required=True,
                long_term_memory_remote_read_timeout_s=8.0,
                chonkydb_timeout_s=20.0,
                chonkydb_base_url="https://memory.test",
                chonkydb_api_key="secret-key",
            )
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(
                        base_url="https://memory.test",
                        api_key="secret-key",
                        timeout_s=8.0,
                    ),
                    opener=opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=FakeOpener(),
                ),
            )

            status = state.status()

        self.assertTrue(status.ready)
        self.assertEqual(state._status_probe_timeout_s(), 20.0)
        self.assertEqual(state._status_probe_instance_timeout_s(), 12.0)
        self.assertEqual(state._status_probe_liveness_timeout_s(), 8.0)
        self.assertAlmostEqual(opener.calls[0]["timeout"], 12.0, places=3)

    def test_status_uses_document_liveness_probe_when_instance_times_out(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            opener = FakeOpener()
            opener.queue_exception(socket.timeout("instance slow"))
            opener.queue_http_error(404, {"detail": "document_not_found"})
            config = TwinrConfig(
                project_root=temp_dir,
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_remote_required=True,
                long_term_memory_remote_read_timeout_s=8.0,
                chonkydb_timeout_s=20.0,
                chonkydb_base_url="https://memory.test",
                chonkydb_api_key="secret-key",
            )
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(
                        base_url="https://memory.test",
                        api_key="secret-key",
                        timeout_s=8.0,
                    ),
                    opener=opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=FakeOpener(),
                ),
            )

            status = state.status()

        self.assertTrue(status.ready)
        self.assertIsNone(status.detail)
        self.assertEqual(len(opener.calls), 2)
        self.assertAlmostEqual(opener.calls[0]["timeout"], 12.0, places=3)
        self.assertAlmostEqual(opener.calls[1]["timeout"], 8.0, places=3)
        self.assertIn("/v1/external/instance", str(opener.calls[0]["full_url"]))
        self.assertIn("/v1/external/documents/full", str(opener.calls[1]["full_url"]))
        self.assertIn("document_id=00000000-0000-0000-0000-000000000000", str(opener.calls[1]["full_url"]))

    def test_status_accepts_bounded_400_liveness_probe_when_instance_times_out(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            opener = FakeOpener()
            opener.queue_exception(socket.timeout("instance slow"))
            opener.queue_http_error(400, {"detail": "selector_invalid"})
            config = TwinrConfig(
                project_root=temp_dir,
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_remote_required=True,
                long_term_memory_remote_read_timeout_s=8.0,
                chonkydb_timeout_s=20.0,
                chonkydb_base_url="https://memory.test",
                chonkydb_api_key="secret-key",
            )
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(
                        base_url="https://memory.test",
                        api_key="secret-key",
                        timeout_s=8.0,
                    ),
                    opener=opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=FakeOpener(),
                ),
            )

            status = state.status()

        self.assertTrue(status.ready)
        self.assertIsNone(status.detail)
        self.assertEqual(len(opener.calls), 2)

    def test_status_reserves_liveness_budget_even_when_instance_stalls(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            opener = FakeOpener()
            opener.queue_exception(socket.timeout("instance slow"))
            opener.queue_http_error(404, {"detail": "document_not_found"})
            config = TwinrConfig(
                project_root=temp_dir,
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_remote_required=True,
                long_term_memory_remote_read_timeout_s=8.0,
                chonkydb_timeout_s=20.0,
                chonkydb_base_url="https://memory.test",
                chonkydb_api_key="secret-key",
            )
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(
                        base_url="https://memory.test",
                        api_key="secret-key",
                        timeout_s=8.0,
                    ),
                    opener=opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=FakeOpener(),
                ),
            )

            status = state.status()

        self.assertTrue(status.ready)
        self.assertEqual(len(opener.calls), 2)
        self.assertAlmostEqual(opener.calls[0]["timeout"], state._status_probe_instance_timeout_s(), places=3)
        self.assertAlmostEqual(opener.calls[1]["timeout"], state._status_probe_liveness_timeout_s(), places=3)

    def test_status_retries_transient_instance_503_before_reporting_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            opener = FakeOpener()
            opener.queue_http_error(503, {"detail": "busy"})
            opener.queue_http_error(503, {"detail": "busy"})
            opener.queue_json({"success": True, "service": "ccodex_memory", "ready": True, "auth_enabled": True})
            config = TwinrConfig(
                project_root=temp_dir,
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_remote_required=True,
                long_term_memory_remote_read_timeout_s=8.0,
                chonkydb_timeout_s=20.0,
                chonkydb_base_url="https://memory.test",
                chonkydb_api_key="secret-key",
                long_term_memory_remote_retry_attempts=3,
                long_term_memory_remote_retry_backoff_s=0.0,
            )
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(
                        base_url="https://memory.test",
                        api_key="secret-key",
                        timeout_s=8.0,
                    ),
                    opener=opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=FakeOpener(),
                ),
            )

            status = state.status()

        self.assertTrue(status.ready)
        self.assertIsNone(status.detail)
        self.assertEqual(len(opener.calls), 3)
        self.assertFalse(state._circuit_is_open())
        self.assertTrue(all("/v1/external/instance" in str(call["full_url"]) for call in opener.calls))

    def test_status_opens_local_cooldown_only_after_failure_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_remote_required=True,
                long_term_memory_remote_read_timeout_s=8.0,
                chonkydb_timeout_s=20.0,
                chonkydb_base_url="https://memory.test",
                chonkydb_api_key="secret-key",
                long_term_memory_remote_retry_attempts=1,
                long_term_memory_remote_retry_backoff_s=0.0,
                long_term_memory_remote_required_failure_threshold=2,
            )
            opener = FakeOpener()
            opener.queue_http_error(503, {"detail": "busy"})
            opener.queue_http_error(503, {"detail": "busy"})
            opener.queue_http_error(503, {"detail": "busy"})
            opener.queue_http_error(503, {"detail": "busy"})
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(
                        base_url="https://memory.test",
                        api_key="secret-key",
                        timeout_s=8.0,
                    ),
                    opener=opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=FakeOpener(),
                ),
            )

            first = state.status()
            circuit_open_after_first = state._circuit_is_open()
            second = state.status()
            circuit_open_after_second = state._circuit_is_open()
            cooldown = state.status()

        self.assertFalse(first.ready)
        self.assertEqual(first.detail, "ChonkyDB health check failed (ChonkyDBError).")
        self.assertTrue(first.operational_probe_allowed)
        self.assertFalse(circuit_open_after_first)
        self.assertFalse(second.ready)
        self.assertEqual(second.detail, "ChonkyDB health check failed (ChonkyDBError).")
        self.assertFalse(second.operational_probe_allowed)
        self.assertTrue(circuit_open_after_second)
        self.assertEqual(
            cooldown.detail,
            "Remote long-term memory is temporarily cooling down after recent failures.",
        )

    def test_status_preserves_explicit_upstream_restart_detail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_remote_required=True,
                long_term_memory_remote_read_timeout_s=8.0,
                chonkydb_timeout_s=20.0,
                chonkydb_base_url="https://memory.test",
                chonkydb_api_key="secret-key",
                long_term_memory_remote_retry_attempts=1,
                long_term_memory_remote_retry_backoff_s=0.0,
            )
            opener = FakeOpener()
            opener.queue_http_error(503, {"detail": "Upstream unavailable or restarting"})
            opener.queue_http_error(503, {"detail": "Upstream unavailable or restarting"})
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(
                        base_url="https://memory.test",
                        api_key="secret-key",
                        timeout_s=8.0,
                    ),
                    opener=opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=FakeOpener(),
                ),
            )

            status = state.status()

        self.assertFalse(status.ready)
        self.assertEqual(status.detail, "Upstream unavailable or restarting")
        self.assertTrue(status.operational_probe_allowed)

    def test_status_probe_clamps_an_overlarge_client_timeout_to_the_probe_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_remote_required=True,
                long_term_memory_remote_read_timeout_s=8.0,
                chonkydb_timeout_s=20.0,
                chonkydb_base_url="https://memory.test",
                chonkydb_api_key="secret-key",
            )

            class _StatusClient:
                def __init__(self, timeout_s: float, calls: list[float]) -> None:
                    self.config = SimpleNamespace(timeout_s=timeout_s)
                    self._calls = calls

                def instance(self):
                    self._calls.append(float(self.config.timeout_s))
                    return SimpleNamespace(ready=True)

                def clone_with_timeout(self, timeout_s: float):
                    return _StatusClient(timeout_s, self._calls)

            observed_timeouts: list[float] = []
            read_client = _StatusClient(timeout_s=90.0, calls=observed_timeouts)
            state = LongTermRemoteStateStore(
                config=config,
                read_client=cast(ChonkyDBClient, read_client),
                write_client=cast(ChonkyDBClient, read_client),
            )

            status = state.status()

        self.assertTrue(status.ready)
        self.assertEqual(len(observed_timeouts), 1)
        self.assertAlmostEqual(observed_timeouts[0], state._status_probe_instance_timeout_s(), places=3)

    def test_origin_resolution_client_clamps_an_overlarge_timeout_to_the_bootstrap_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(temp_dir)

            class _ClonableClient:
                def __init__(self, timeout_s: float) -> None:
                    self.config = SimpleNamespace(timeout_s=timeout_s)

                def clone_with_timeout(self, timeout_s: float):
                    return _ClonableClient(timeout_s)

            state = LongTermRemoteStateStore(
                config=config,
                read_client=cast(ChonkyDBClient, _ClonableClient(timeout_s=90.0)),
                write_client=cast(ChonkyDBClient, _ClonableClient(timeout_s=90.0)),
            )

            tuned = state._origin_resolution_client(cast(ChonkyDBClient, _ClonableClient(timeout_s=90.0)))

        self.assertEqual(tuned.config.timeout_s, state._origin_resolution_bootstrap_timeout_s())

    def test_origin_resolution_client_respects_explicit_client_timeout_caps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(temp_dir)

            class _ClonableClient:
                def __init__(self, timeout_s: float) -> None:
                    self.config = SimpleNamespace(timeout_s=timeout_s)

                def clone_with_timeout(self, timeout_s: float):
                    clone = _ClonableClient(timeout_s)
                    timeout_cap_s = getattr(self, "_twinr_timeout_cap_s", None)
                    if timeout_cap_s is not None:
                        setattr(clone, "_twinr_timeout_cap_s", timeout_cap_s)
                    return clone

            state = LongTermRemoteStateStore(
                config=config,
                read_client=cast(ChonkyDBClient, _ClonableClient(timeout_s=2.0)),
                write_client=cast(ChonkyDBClient, _ClonableClient(timeout_s=2.0)),
            )
            capped = _ClonableClient(timeout_s=2.0)
            setattr(capped, "_twinr_timeout_cap_s", 2.0)

            tuned = state._origin_resolution_client(cast(ChonkyDBClient, capped))

        self.assertIs(tuned, capped)
        self.assertEqual(tuned.config.timeout_s, 2.0)

    def test_load_snapshot_via_uri_can_resolve_snapshot_from_metadata_without_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(temp_dir)
            read_opener = FakeOpener()
            read_opener.queue_json(
                {
                    "success": True,
                    "record": {
                        "document_id": "doc-123",
                        "payload": {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "objects",
                            "updated_at": "2026-03-29T10:00:00+00:00",
                            "body": {"schema": "object_store", "objects": [{"memory_id": "fact:meta"}]},
                        },
                    },
                }
            )
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=FakeOpener(),
                ),
            )

            result = state._load_snapshot_via_uri(
                state.read_client,
                snapshot_kind="objects",
                source="origin_uri",
                prefer_metadata_only=True,
            )

        self.assertEqual(result.payload, {"schema": "object_store", "objects": [{"memory_id": "fact:meta"}]})
        self.assertEqual(len(read_opener.calls), 1)

    def test_load_snapshot_via_uri_falls_back_to_content_when_metadata_has_no_snapshot_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(temp_dir)
            read_opener = FakeOpener()
            read_opener.queue_json({"success": True, "record": {"document_id": "doc-123", "metadata": {"status": "ok"}}})
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "doc-123",
                    "content": json.dumps(
                        {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "objects",
                            "updated_at": "2026-03-29T10:00:01+00:00",
                            "body": {"schema": "object_store", "objects": [{"memory_id": "fact:content"}]},
                        }
                    ),
                }
            )
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=FakeOpener(),
                ),
            )

            result = state._load_snapshot_via_uri(
                state.read_client,
                snapshot_kind="objects",
                source="origin_uri",
                prefer_metadata_only=True,
            )

        self.assertEqual(result.payload, {"schema": "object_store", "objects": [{"memory_id": "fact:content"}]})
        self.assertEqual(len(read_opener.calls), 2)

    def test_probe_snapshot_load_accepts_prefer_metadata_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(temp_dir)
            read_opener = FakeOpener()
            read_opener.queue_json(
                {
                    "success": True,
                    "record": {
                        "document_id": "pointer-doc-123",
                        "payload": {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "__pointer__:objects",
                            "updated_at": "2026-03-29T10:00:00+00:00",
                            "body": {
                                "schema": "twinr_remote_snapshot_pointer_v1",
                                "version": 1,
                                "snapshot_kind": "objects",
                                "document_id": "doc-123",
                            },
                        },
                    },
                }
            )
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=FakeOpener(),
                ),
            )

            result = state.probe_snapshot_load(
                snapshot_kind="__pointer__:objects",
                prefer_metadata_only=True,
            )

        self.assertEqual(
            result.payload,
            {
                "schema": "twinr_remote_snapshot_pointer_v1",
                "version": 1,
                "snapshot_kind": "objects",
                "document_id": "doc-123",
            },
        )
        self.assertEqual(len(read_opener.calls), 1)

    def test_probe_snapshot_load_fast_fail_avoids_bootstrap_timeout_inflation_and_retries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(temp_dir)
            read_opener = FakeOpener()
            read_opener.queue_http_error(503, {"detail": "busy"})
            read_opener.queue_http_error(503, {"detail": "busy"})
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key", timeout_s=8.0),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=FakeOpener(),
                ),
            )

            result = state.probe_snapshot_load(
                snapshot_kind="conflicts",
                prefer_cached_document_id=True,
                prefer_metadata_only=True,
                fast_fail=True,
            )

        self.assertEqual(result.status, "unavailable")
        self.assertEqual(len(read_opener.calls), 2)
        self.assertTrue(all(call["timeout"] == 8.0 for call in read_opener.calls))

    def test_probe_snapshot_load_tries_origin_after_pointer_failure_opens_circuit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                self._config(temp_dir),
                long_term_memory_remote_retry_attempts=1,
                long_term_memory_remote_retry_backoff_s=0.0,
            )
            read_opener = FakeOpener()
            write_opener = FakeOpener()
            read_opener.queue_http_error(503, {"detail": "pointer busy"})
            read_opener.queue_json(
                {
                    "success": True,
                    "document_id": "doc-origin",
                    "record": {
                        "document_id": "doc-origin",
                        "payload": {
                            "schema": "twinr_remote_snapshot_v1",
                            "namespace": "test-namespace",
                            "snapshot_kind": "objects",
                            "updated_at": "2026-03-29T10:00:02+00:00",
                            "body": {"schema": "object_store", "objects": [{"memory_id": "fact:origin"}]},
                        },
                    },
                }
            )
            state = LongTermRemoteStateStore(
                config=config,
                read_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=read_opener,
                ),
                write_client=ChonkyDBClient(
                    ChonkyDBConnectionConfig(base_url="https://memory.test", api_key="secret-key"),
                    opener=write_opener,
                ),
            )

            result = state.probe_snapshot_load(snapshot_kind="objects", prefer_metadata_only=True)

        self.assertEqual(result.status, "found")
        self.assertEqual(result.selected_source, "origin_uri")
        self.assertEqual(result.payload, {"schema": "object_store", "objects": [{"memory_id": "fact:origin"}]})
        self.assertFalse(state._circuit_is_open())
        self.assertEqual(len(read_opener.calls), 2)
        self.assertEqual(len(write_opener.calls), 0)
        self.assertIn("__pointer__%253Aobjects", read_opener.calls[0]["full_url"])
        self.assertIn("%2Fobjects&include_content=false", read_opener.calls[1]["full_url"])

    def test_attest_external_readiness_clears_local_cooldown_and_probe_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state = LongTermRemoteStateStore(config=self._config(temp_dir))
            with state._state_lock:
                state._consecutive_failures = 3
                state._circuit_open_until_monotonic = time.monotonic() + 30.0
                state._probe_cache["objects"] = LongTermRemoteSnapshotProbe(
                    snapshot_kind="objects",
                    status="missing",
                    latency_ms=1.0,
                    detail="stale failure",
                )

            self.assertEqual(
                state.status().detail,
                "Remote long-term memory is temporarily cooling down after recent failures.",
            )

            state.attest_external_readiness()

            self.assertEqual(state.status().detail, "ChonkyDB is not configured.")
            self.assertIsNone(state._cached_probe(snapshot_kind="objects"))

    def test_service_fails_closed_when_remote_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_remote_required=True,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
            )
            service = LongTermMemoryService.from_config(config, extractor=make_test_extractor())
            try:
                with self.assertRaisesRegex(RuntimeError, "Remote-primary long-term memory is enabled"):
                    service.build_provider_context("What did we talk about earlier?")
                enqueue_result = service.enqueue_conversation_turn(
                    transcript="Please remember this for later.",
                    response="I will keep our conversation context available.",
                )
                drained = service.flush(timeout_s=2.0)
                entries = PersistentMemoryMarkdownStore(config.memory_markdown_path).load_entries()
            finally:
                service.shutdown()

        self.assertIsNotNone(enqueue_result)
        self.assertTrue(enqueue_result.accepted)
        self.assertFalse(drained)
        self.assertEqual(entries, ())
