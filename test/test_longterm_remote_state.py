from __future__ import annotations

from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test.longterm_test_program import make_test_extractor
from twinr.config import TwinrConfig
from twinr.memory.chonkydb import ChonkyDBClient, ChonkyDBConnectionConfig
from twinr.memory.context_store import PersistentMemoryMarkdownStore
from twinr.memory.longterm.storage.remote_state import LongTermRemoteStateStore
from twinr.memory.longterm.runtime.service import LongTermMemoryService


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

    def queue_http_error(self, status_code: int, payload: dict[str, object]) -> None:
        self.responses.append(
            HTTPError(
                url="https://memory.test/fail",
                code=status_code,
                msg="bad request",
                hdrs=None,
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

    def test_remote_snapshot_save_and_load_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._config(temp_dir)
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
                            "body": {"schema": "object_store", "objects": [{"memory_id": "fact:1"}]},
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
                    opener=FakeOpener(),
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

    def test_remote_snapshot_save_retries_after_transient_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                self._config(temp_dir),
                long_term_memory_remote_retry_attempts=2,
                long_term_memory_remote_retry_backoff_s=0.0,
            )
            write_opener = FakeOpener()
            write_opener.queue_exception(TimeoutError("timed out"))
            write_opener.queue_json({"success": True, "stored": 1})
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

            state.save_snapshot(snapshot_kind="objects", payload={"schema": "object_store", "objects": []})

        self.assertEqual(len(write_opener.calls), 2)

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
