from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.memory.context_store import ManagedContextFileStore, PersistentMemoryMarkdownStore


class _FakeRemoteState:
    def __init__(self, *, migration_enabled: bool = False) -> None:
        self.enabled = True
        self.config = SimpleNamespace(long_term_memory_migration_enabled=migration_enabled)
        self.snapshots: dict[str, dict[str, object]] = {}

    def load_snapshot(self, *, snapshot_kind: str, local_path=None):
        del local_path
        payload = self.snapshots.get(snapshot_kind)
        return dict(payload) if isinstance(payload, dict) else None

    def save_snapshot(self, *, snapshot_kind: str, payload):
        self.snapshots[snapshot_kind] = dict(payload)


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


if __name__ == "__main__":
    unittest.main()
