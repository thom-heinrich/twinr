from pathlib import Path
import stat
import sys
import tempfile
from threading import Lock
import time
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.memory.chonkydb.client import ChonkyDBError
from twinr.memory.context_store import ManagedContextFileStore, PersistentMemoryMarkdownStore, PromptContextStore
from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError


class _FakeCatalogClient:
    def __init__(self, *, error_getter, remote_state=None) -> None:
        self._error_getter = error_getter
        self._remote_state = remote_state
        self._next_document_id = 1
        self.bulk_calls = 0
        self.bulk_execution_modes: list[str] = []
        self.records_by_document_id: dict[str, dict[str, object]] = {}
        self.records_by_uri: dict[str, dict[str, object]] = {}

    def _maybe_raise(self) -> None:
        error = self._error_getter()
        if error is not None:
            raise error

    def store_records_bulk(self, request):
        self.bulk_calls += 1
        self.bulk_execution_modes.append(str(getattr(request, "execution_mode", "")))
        self._maybe_raise()
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
            metadata = record.get("metadata")
            if (
                isinstance(metadata, dict)
                and metadata.get("twinr_catalog_current_head") is True
                and self._remote_state is not None
            ):
                snapshot_kind = metadata.get("twinr_snapshot_kind")
                payload = metadata.get("twinr_payload")
                if isinstance(snapshot_kind, str) and snapshot_kind and isinstance(payload, dict):
                    self._remote_state.snapshots[snapshot_kind] = dict(payload)
            response_items.append({"document_id": document_id})
        return {"items": response_items}

    def fetch_full_document(self, *, document_id=None, origin_uri=None, include_content=True, max_content_chars=4000):
        del include_content, max_content_chars
        self._maybe_raise()
        if isinstance(document_id, str) and document_id:
            record = self.records_by_document_id.get(document_id)
            if record is not None:
                return dict(record)
        if isinstance(origin_uri, str) and origin_uri:
            record = self.records_by_uri.get(origin_uri)
            if record is not None:
                return dict(record)
        raise LongTermRemoteUnavailableError("remote document unavailable")


class _FakeRemoteState:
    def __init__(self, *, migration_enabled: bool = False) -> None:
        self.enabled = True
        self.required = False
        self.namespace = "test-namespace"
        self.load_error: Exception | None = None
        self.config = SimpleNamespace(
            long_term_memory_migration_enabled=migration_enabled,
            long_term_memory_migration_batch_size=64,
            long_term_memory_remote_read_timeout_s=8.0,
            long_term_memory_remote_write_timeout_s=15.0,
            long_term_memory_remote_flush_timeout_s=60.0,
            long_term_memory_remote_bulk_request_max_bytes=512 * 1024,
            long_term_memory_remote_shard_max_content_chars=1000,
            long_term_memory_remote_max_content_chars=2_000_000,
            long_term_memory_remote_read_cache_ttl_s=0.0,
        )
        self.snapshots: dict[str, dict[str, object]] = {}
        self.probe_calls: list[dict[str, object]] = []
        self.client = _FakeCatalogClient(error_getter=lambda: self.load_error, remote_state=self)
        self.read_client = self.client
        self.write_client = self.client

    def load_snapshot(self, *, snapshot_kind: str, local_path=None):
        del local_path
        if self.load_error is not None:
            raise self.load_error
        payload = self.snapshots.get(snapshot_kind)
        return dict(payload) if isinstance(payload, dict) else None

    def probe_snapshot_load(
        self,
        *,
        snapshot_kind: str,
        local_path=None,
        prefer_cached_document_id: bool = False,
        prefer_metadata_only: bool = False,
        fast_fail: bool = False,
    ):
        del local_path, fast_fail
        self.probe_calls.append(
            {
                "snapshot_kind": snapshot_kind,
                "prefer_cached_document_id": prefer_cached_document_id,
                "prefer_metadata_only": prefer_metadata_only,
            }
        )
        if self.load_error is not None:
            raise self.load_error
        payload = self.snapshots.get(snapshot_kind)
        return SimpleNamespace(
            status="found" if isinstance(payload, dict) else "not_found",
            payload=dict(payload) if isinstance(payload, dict) else None,
            detail=None,
        )

    def save_snapshot(self, *, snapshot_kind: str, payload):
        self.snapshots[snapshot_kind] = dict(payload)


class _FailingRemoteState(_FakeRemoteState):
    def __init__(self) -> None:
        super().__init__()
        self.load_error = LongTermRemoteUnavailableError(
            "Failed to read remote long-term current head: status=503"
        )


class _ConcurrentPromptSnapshotTracker:
    def __init__(self) -> None:
        self.max_concurrent_calls = 0
        self._active_calls = 0
        self._lock = Lock()
        self.call_order: list[str] = []

    def enter(self, snapshot_kind: str) -> None:
        with self._lock:
            self._active_calls += 1
            self.max_concurrent_calls = max(self.max_concurrent_calls, self._active_calls)
            self.call_order.append(snapshot_kind)

    def leave(self) -> None:
        with self._lock:
            self._active_calls -= 1


class _ConcurrentPromptSnapshotComponent:
    def __init__(
        self,
        *,
        tracker: _ConcurrentPromptSnapshotTracker,
        remote_snapshot_kind: str,
        created: bool,
    ) -> None:
        self.remote_snapshot_kind = remote_snapshot_kind
        self._tracker = tracker
        self._created = created

    def ensure_remote_snapshot(self) -> bool:
        self._tracker.enter(self.remote_snapshot_kind)
        try:
            time.sleep(0.01)
            return self._created
        finally:
            self._tracker.leave()


class ContextStoreTests(unittest.TestCase):
    def test_memory_store_writes_markdown_and_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "MEMORY.md"
            store = PersistentMemoryMarkdownStore(path, max_entries=3)

            first = store.remember(
                kind="appointment",
                summary="Arzttermin am Montag um 14 Uhr.",
                details="Bei Dr. Meyer in Hamburg.",
            )
            second = store.remember(
                kind="appointment",
                summary="Arzttermin am Montag um 14 Uhr.",
                details="Bei Dr. Meyer in Hamburg.",
            )

            entries = store.load_entries()
            rendered = path.read_text(encoding="utf-8")

        self.assertEqual(len(entries), 1)
        self.assertEqual(first.entry_id, second.entry_id)
        self.assertEqual(entries[0].kind, "appointment")
        self.assertIn("Arzttermin am Montag um 14 Uhr.", rendered)

    def test_memory_store_writes_world_readable_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "MEMORY.md"
            store = PersistentMemoryMarkdownStore(path, max_entries=3)

            store.remember(
                kind="appointment",
                summary="Augenarzt am Dienstag um 10:30.",
            )

            mode = stat.S_IMODE(path.stat().st_mode)

        self.assertEqual(mode, 0o644)

    def test_memory_store_renders_compact_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "MEMORY.md"
            store = PersistentMemoryMarkdownStore(path)
            store.remember(
                kind="contact",
                summary="Telefonnummer vom Rathaus Schwarzenbek: 04151 8810.",
            )

            context = store.render_context()

        self.assertIsNotNone(context)
        self.assertIn("Durable remembered items", context)
        self.assertIn("04151 8810", context)

    def test_managed_context_store_upserts_by_category(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "PERSONALITY.md"
            path.write_text("Base personality text.\n", encoding="utf-8")
            store = ManagedContextFileStore(path, section_title="Twinr managed personality updates")

            store.upsert(category="response_style", instruction="Keep answers short and direct.")
            store.upsert(category="response_style", instruction="Keep answers very short and calm.")
            store.upsert(category="humor", instruction="Use only light humor.")

            entries = store.load_entries()
            rendered = path.read_text(encoding="utf-8")

        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].key, "response_style")
        self.assertIn("Base personality text.", rendered)
        self.assertIn("Twinr managed personality updates", rendered)
        self.assertIn("response_style: Keep answers very short and calm.", rendered)
        self.assertIn("humor: Use only light humor.", rendered)

    def test_managed_context_store_replaces_base_text_and_keeps_managed_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "USER.md"
            store = ManagedContextFileStore(path, section_title="Twinr managed user updates")
            store.replace_base_text("Base user facts.")
            store.upsert(category="pets", instruction="Thom has two dogs.")
            store.replace_base_text("Updated user facts.")

            rendered = path.read_text(encoding="utf-8")
            base_text = store.load_base_text()
            entries = store.load_entries()

        self.assertEqual(base_text, "Updated user facts.")
        self.assertEqual(len(entries), 1)
        self.assertIn("Updated user facts.", rendered)
        self.assertIn("pets: Thom has two dogs.", rendered)

    def test_memory_store_remote_primary_keeps_explicit_memories_off_disk(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "MEMORY.md"
            remote_state = _FakeRemoteState()
            store = PersistentMemoryMarkdownStore(path, remote_state=remote_state)

            store.remember(
                kind="appointment",
                summary="Eye doctor on Tuesday at 10:30.",
            )
            entries = store.load_entries()
            context = store.render_context()

        self.assertFalse(path.exists())
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].summary, "Eye doctor on Tuesday at 10:30.")
        self.assertIn("Eye doctor on Tuesday at 10:30.", context or "")
        self.assertIn("prompt_memory", remote_state.snapshots)
        self.assertEqual(remote_state.snapshots["prompt_memory"]["schema"], "twinr_prompt_memory_catalog_v3")
        self.assertTrue(remote_state.client.bulk_execution_modes)
        self.assertTrue(all(mode == "async" for mode in remote_state.client.bulk_execution_modes))

    def test_managed_context_store_remote_primary_keeps_updates_off_disk(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "USER.md"
            path.write_text("Base user facts.\n", encoding="utf-8")
            remote_state = _FakeRemoteState()
            store = ManagedContextFileStore(
                path,
                section_title="Twinr managed user updates",
                remote_state=remote_state,
                remote_snapshot_kind="user_context",
            )

            store.upsert(category="pets", instruction="Thom has two dogs.")
            rendered = store.render_context()
            entries = store.load_entries()
            on_disk = path.read_text(encoding="utf-8")

        self.assertEqual(on_disk, "Base user facts.\n")
        self.assertEqual(len(entries), 1)
        self.assertIn("Base user facts.", rendered or "")
        self.assertIn("pets: Thom has two dogs.", rendered or "")
        self.assertIn("user_context", remote_state.snapshots)
        self.assertEqual(remote_state.snapshots["user_context"]["schema"], "twinr_user_context_catalog_v3")
        self.assertTrue(remote_state.client.bulk_execution_modes)
        self.assertTrue(all(mode == "async" for mode in remote_state.client.bulk_execution_modes))

    def test_managed_context_store_ensure_remote_snapshot_keeps_empty_remote_document_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "USER.md"
            path.write_text("Base user facts.\n", encoding="utf-8")
            remote_state = _FakeRemoteState()
            store = ManagedContextFileStore(
                path,
                section_title="Twinr managed user updates",
                remote_state=remote_state,
                remote_snapshot_kind="user_context",
            )

            created = store.ensure_remote_snapshot()

        self.assertFalse(created)
        self.assertEqual(
            store.probe_remote_current_head(),
            {"schema": "twinr_user_context_catalog_v3", "version": 3, "items_count": 0, "segments": []},
        )
        self.assertEqual(remote_state.snapshots, {})
        self.assertEqual(remote_state.client.bulk_calls, 0)

    def test_managed_context_store_ensure_remote_snapshot_repairs_invalid_blank_current_head(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "USER.md"
            path.write_text("Base user facts.\n", encoding="utf-8")
            remote_state = _FakeRemoteState()
            store = ManagedContextFileStore(
                path,
                section_title="Twinr managed user updates",
                remote_state=remote_state,
                remote_snapshot_kind="user_context",
            )
            current_head_uri = store._remote_records._catalog._catalog_head_uri(snapshot_kind="user_context")
            invalid_record = {
                "document_id": "doc-invalid-user-context",
                "payload": {},
                "metadata": {},
                "content": "",
                "uri": current_head_uri,
            }
            remote_state.client.records_by_document_id[invalid_record["document_id"]] = invalid_record
            remote_state.client.records_by_uri[current_head_uri] = invalid_record

            created = store.ensure_remote_snapshot()

        self.assertTrue(created)
        self.assertEqual(
            store.probe_remote_current_head(),
            {"schema": "twinr_user_context_catalog_v3", "version": 3, "items_count": 0, "segments": []},
        )
        self.assertEqual(
            remote_state.snapshots["user_context"],
            {"schema": "twinr_user_context_catalog_v3", "version": 3, "items_count": 0, "segments": []},
        )
        self.assertEqual(remote_state.client.bulk_calls, 1)

    def test_managed_context_store_retries_transient_queue_saturation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "USER.md"
            path.write_text(
                "\n".join(
                    [
                        "Base user facts.",
                        "",
                        "<!-- TWINR_MANAGED_CONTEXT_START -->",
                        "## Twinr managed user updates",
                        "_This section is managed by Twinr. Keep entries short and stable._",
                        "- pets: Thom has two dogs.",
                        "<!-- TWINR_MANAGED_CONTEXT_END -->",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            remote_state = _FakeRemoteState()
            remote_state.config.long_term_memory_remote_retry_backoff_s = 0.0
            remote_state.client._error_getter = lambda: (
                ChonkyDBError(
                    "ChonkyDB request failed for POST /v1/external/records/bulk",
                    status_code=429,
                    response_json={"detail": "queue_saturated", "error": "queue_saturated"},
                )
                if 0 < remote_state.client.bulk_calls < 3
                else None
            )
            store = ManagedContextFileStore(
                path,
                section_title="Twinr managed user updates",
                remote_state=remote_state,
                remote_snapshot_kind="user_context",
            )

            created = store.ensure_remote_snapshot()

        self.assertTrue(created)
        self.assertEqual(remote_state.client.bulk_calls, 5)
        self.assertEqual(remote_state.snapshots["user_context"]["schema"], "twinr_user_context_catalog_v3")
        self.assertEqual(remote_state.snapshots["user_context"]["version"], 3)
        self.assertEqual(remote_state.snapshots["user_context"]["items_count"], 1)
        self.assertEqual(len(remote_state.snapshots["user_context"]["segments"]), 1)
        self.assertIsInstance(remote_state.snapshots["user_context"].get("written_at"), str)

    def test_memory_store_ensure_remote_snapshot_keeps_empty_prompt_memory_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "MEMORY.md"
            remote_state = _FakeRemoteState()
            store = PersistentMemoryMarkdownStore(path, remote_state=remote_state)

            created = store.ensure_remote_snapshot()

        self.assertFalse(created)
        self.assertEqual(
            store.probe_remote_current_head(),
            {"schema": "twinr_prompt_memory_catalog_v3", "version": 3, "items_count": 0, "segments": []},
        )
        self.assertEqual(remote_state.snapshots, {})
        self.assertEqual(remote_state.client.bulk_calls, 0)

    def test_memory_store_ensure_remote_snapshot_repairs_invalid_blank_current_head(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "MEMORY.md"
            remote_state = _FakeRemoteState()
            store = PersistentMemoryMarkdownStore(path, remote_state=remote_state)
            current_head_uri = store._remote_records._catalog._catalog_head_uri(snapshot_kind="prompt_memory")
            invalid_record = {
                "document_id": "doc-invalid-prompt-memory",
                "payload": {},
                "metadata": {},
                "content": "",
                "uri": current_head_uri,
            }
            remote_state.client.records_by_document_id[invalid_record["document_id"]] = invalid_record
            remote_state.client.records_by_uri[current_head_uri] = invalid_record

            created = store.ensure_remote_snapshot()

        self.assertTrue(created)
        self.assertEqual(
            store.probe_remote_current_head(),
            {"schema": "twinr_prompt_memory_catalog_v3", "version": 3, "items_count": 0, "segments": []},
        )
        self.assertEqual(
            remote_state.snapshots["prompt_memory"],
            {"schema": "twinr_prompt_memory_catalog_v3", "version": 3, "items_count": 0, "segments": []},
        )
        self.assertEqual(remote_state.client.bulk_calls, 1)

    def test_memory_store_load_entries_uses_legacy_head_when_current_head_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "MEMORY.md"
            remote_state = _FakeRemoteState()
            writer_store = PersistentMemoryMarkdownStore(path, remote_state=remote_state)

            writer_store.remember(
                kind="appointment",
                summary="Eye doctor on Tuesday at 10:30.",
            )
            current_head_uri = writer_store._remote_records._catalog._catalog_head_uri(snapshot_kind="prompt_memory")
            remote_state.client.records_by_uri.pop(current_head_uri, None)
            remote_state.probe_calls.clear()
            baseline_bulk_calls = remote_state.client.bulk_calls
            reader_store = PersistentMemoryMarkdownStore(path, remote_state=remote_state)

            head = reader_store.load_remote_current_head()
            entries = reader_store.load_entries()

        self.assertEqual(
            head,
            remote_state.snapshots["prompt_memory"],
        )
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].summary, "Eye doctor on Tuesday at 10:30.")
        self.assertEqual(remote_state.client.bulk_calls, baseline_bulk_calls)
        self.assertEqual(
            remote_state.probe_calls[-1],
            {
                "snapshot_kind": "prompt_memory",
                "prefer_cached_document_id": True,
                "prefer_metadata_only": True,
            },
        )

    def test_memory_store_current_head_only_read_skips_legacy_head_when_current_head_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "MEMORY.md"
            remote_state = _FakeRemoteState()
            writer_store = PersistentMemoryMarkdownStore(path, remote_state=remote_state)

            writer_store.remember(
                kind="appointment",
                summary="Eye doctor on Tuesday at 10:30.",
            )
            current_head_uri = writer_store._remote_records._catalog._catalog_head_uri(snapshot_kind="prompt_memory")
            remote_state.client.records_by_uri.pop(current_head_uri, None)
            remote_state.probe_calls.clear()
            reader_store = PersistentMemoryMarkdownStore(
                path,
                remote_state=remote_state,
                allow_legacy_head_reads=False,
            )

            head = reader_store.load_remote_current_head()
            entries = reader_store.load_entries()

        self.assertEqual(
            head,
            {"schema": "twinr_prompt_memory_catalog_v3", "version": 3, "items_count": 0, "segments": []},
        )
        self.assertEqual(entries, ())
        self.assertEqual(remote_state.probe_calls, [])

    def test_managed_context_store_load_entries_uses_legacy_head_when_current_head_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "USER.md"
            path.write_text("Base user facts.\n", encoding="utf-8")
            remote_state = _FakeRemoteState()
            writer_store = ManagedContextFileStore(
                path,
                section_title="Twinr managed user updates",
                remote_state=remote_state,
                remote_snapshot_kind="user_context",
            )

            writer_store.upsert(category="pets", instruction="Thom has two dogs.")
            current_head_uri = writer_store._remote_records._catalog._catalog_head_uri(snapshot_kind="user_context")
            remote_state.client.records_by_uri.pop(current_head_uri, None)
            remote_state.probe_calls.clear()
            baseline_bulk_calls = remote_state.client.bulk_calls
            reader_store = ManagedContextFileStore(
                path,
                section_title="Twinr managed user updates",
                remote_state=remote_state,
                remote_snapshot_kind="user_context",
            )

            head = reader_store.load_remote_current_head()
            entries = reader_store.load_entries()

        self.assertEqual(
            head,
            remote_state.snapshots["user_context"],
        )
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].instruction, "Thom has two dogs.")
        self.assertEqual(remote_state.client.bulk_calls, baseline_bulk_calls)
        self.assertEqual(
            remote_state.probe_calls[-1],
            {
                "snapshot_kind": "user_context",
                "prefer_cached_document_id": True,
                "prefer_metadata_only": True,
            },
        )

    def test_managed_context_store_current_head_only_read_skips_legacy_head_when_current_head_missing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "USER.md"
            path.write_text("Base user facts.\n", encoding="utf-8")
            remote_state = _FakeRemoteState()
            writer_store = ManagedContextFileStore(
                path,
                section_title="Twinr managed user updates",
                remote_state=remote_state,
                remote_snapshot_kind="user_context",
            )

            writer_store.upsert(category="pets", instruction="Thom has two dogs.")
            current_head_uri = writer_store._remote_records._catalog._catalog_head_uri(snapshot_kind="user_context")
            remote_state.client.records_by_uri.pop(current_head_uri, None)
            remote_state.probe_calls.clear()
            reader_store = ManagedContextFileStore(
                path,
                section_title="Twinr managed user updates",
                remote_state=remote_state,
                remote_snapshot_kind="user_context",
                allow_legacy_head_reads=False,
            )

            head = reader_store.load_remote_current_head()
            entries = reader_store.load_entries()

        self.assertEqual(
            head,
            {"schema": "twinr_user_context_catalog_v3", "version": 3, "items_count": 0, "segments": []},
        )
        self.assertEqual(entries, ())
        self.assertEqual(remote_state.probe_calls, [])

    def test_prompt_context_store_serializes_remote_snapshot_checks(self) -> None:
        tracker = _ConcurrentPromptSnapshotTracker()
        store = PromptContextStore(
            memory_store=_ConcurrentPromptSnapshotComponent(
                tracker=tracker,
                remote_snapshot_kind="prompt_memory",
                created=True,
            ),
            user_store=_ConcurrentPromptSnapshotComponent(
                tracker=tracker,
                remote_snapshot_kind="user_context",
                created=False,
            ),
            personality_store=_ConcurrentPromptSnapshotComponent(
                tracker=tracker,
                remote_snapshot_kind="personality_context",
                created=True,
            ),
        )

        ensured = store.ensure_remote_snapshots()

        self.assertEqual(ensured, ("prompt_memory", "personality_context"))
        self.assertEqual(tracker.call_order, ["prompt_memory", "user_context", "personality_context"])
        self.assertEqual(tracker.max_concurrent_calls, 1)

    def test_managed_context_store_remote_primary_raises_when_remote_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "PERSONALITY.md"
            path.write_text(
                "\n".join(
                    [
                        "Base personality text.",
                        "",
                        "<!-- TWINR_MANAGED_CONTEXT_START -->",
                        "## Twinr managed personality updates",
                        "_This section is managed by Twinr. Keep entries short and stable._",
                        "- response_style: Keep answers calm.",
                        "<!-- TWINR_MANAGED_CONTEXT_END -->",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            store = ManagedContextFileStore(
                path,
                section_title="Twinr managed personality updates",
                remote_state=_FailingRemoteState(),
                remote_snapshot_kind="personality_context",
            )

            with self.assertRaises(LongTermRemoteUnavailableError):
                store.load_entries()
            with self.assertRaises(LongTermRemoteUnavailableError):
                store.render_context()


if __name__ == "__main__":
    unittest.main()
