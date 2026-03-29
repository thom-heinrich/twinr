from pathlib import Path
import stat
import sys
import tempfile
from threading import Lock
import time
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.memory.context_store import ManagedContextFileStore, PersistentMemoryMarkdownStore, PromptContextStore
from twinr.memory.longterm.storage.remote_state import LongTermRemoteUnavailableError


class _FakeRemoteState:
    def __init__(self, *, migration_enabled: bool = False) -> None:
        self.enabled = True
        self.config = SimpleNamespace(long_term_memory_migration_enabled=migration_enabled)
        self.snapshots: dict[str, dict[str, object]] = {}
        self.probe_calls: list[dict[str, object]] = []

    def load_snapshot(self, *, snapshot_kind: str, local_path=None):
        del local_path
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
        payload = self.snapshots.get(snapshot_kind)
        return SimpleNamespace(
            status="found" if isinstance(payload, dict) else "not_found",
            payload=dict(payload) if isinstance(payload, dict) else None,
            detail=None,
        )

    def save_snapshot(self, *, snapshot_kind: str, payload):
        self.snapshots[snapshot_kind] = dict(payload)


class _FailingRemoteState(_FakeRemoteState):
    def load_snapshot(self, *, snapshot_kind: str, local_path=None):
        del local_path
        raise LongTermRemoteUnavailableError(
            f"Failed to read remote long-term snapshot {snapshot_kind!r}: status=503"
        )

    def probe_snapshot_load(
        self,
        *,
        snapshot_kind: str,
        local_path=None,
        prefer_cached_document_id: bool = False,
        prefer_metadata_only: bool = False,
        fast_fail: bool = False,
    ):
        del local_path, prefer_cached_document_id, prefer_metadata_only, fast_fail
        raise LongTermRemoteUnavailableError(
            f"Failed to read remote long-term snapshot {snapshot_kind!r}: status=503"
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

    def test_managed_context_store_ensure_remote_snapshot_seeds_empty_remote_document(self) -> None:
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

        self.assertTrue(created)
        self.assertEqual(
            remote_state.snapshots["user_context"],
            {"schema": "twinr_managed_context", "version": 1, "entries": []},
        )

    def test_memory_store_ensure_remote_snapshot_seeds_empty_prompt_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "MEMORY.md"
            remote_state = _FakeRemoteState()
            store = PersistentMemoryMarkdownStore(path, remote_state=remote_state)

            created = store.ensure_remote_snapshot()

        self.assertTrue(created)
        self.assertEqual(
            remote_state.snapshots["prompt_memory"],
            {"schema": "twinr_prompt_memory", "version": 1, "entries": []},
        )

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
