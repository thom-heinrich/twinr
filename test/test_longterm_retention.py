from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.memory.longterm.core.models import LongTermMemoryObjectV1, LongTermSourceRefV1
from twinr.memory.longterm.reasoning.retention import LongTermRetentionPolicy
from twinr.memory.longterm.storage.store import LongTermStructuredStore


class LongTermRetentionPolicyTests(unittest.TestCase):
    def _source(self) -> LongTermSourceRefV1:
        return LongTermSourceRefV1(source_type="conversation_turn", event_ids=("turn:test",), speaker="user", modality="voice")

    def test_retention_archives_old_episodes_and_prunes_old_observations(self) -> None:
        now = datetime(2026, 3, 15, 12, 0, tzinfo=timezone.utc)
        episode = LongTermMemoryObjectV1(
            memory_id="episode:1",
            kind="episode",
            summary="Episode summary",
            source=self._source(),
            updated_at=now - timedelta(days=10),
            created_at=now - timedelta(days=10),
        )
        observation = LongTermMemoryObjectV1(
            memory_id="observation:1",
            kind="observation",
            summary="It is warm today.",
            source=self._source(),
            attributes={"observation_type": "weather"},
            updated_at=now - timedelta(days=4),
            created_at=now - timedelta(days=4),
        )
        expiring = LongTermMemoryObjectV1(
            memory_id="event:1",
            kind="event",
            summary="Appointment is today.",
            source=self._source(),
            valid_to="2026-03-14",
            status="active",
            updated_at=now - timedelta(days=1),
            created_at=now - timedelta(days=1),
        )

        result = LongTermRetentionPolicy(timezone_name="UTC", archive_enabled=True).apply(
            objects=(episode, observation, expiring),
            now=now,
        )

        self.assertEqual({item.memory_id for item in result.archived_objects}, {"episode:1"})
        self.assertEqual(result.pruned_memory_ids, ("observation:1",))
        self.assertEqual({item.memory_id for item in result.expired_objects}, {"event:1"})

    def test_retention_keeps_active_patterns_when_valid_to_is_only_last_observed_day(self) -> None:
        now = datetime(2026, 3, 21, 12, 0, tzinfo=timezone.utc)
        pattern = LongTermMemoryObjectV1(
            memory_id="pattern:button:green:start_listening:morning",
            kind="pattern",
            summary="The green button was used to start a conversation in the morning.",
            details="Low-confidence interaction pattern derived from repeated button use.",
            source=self._source(),
            status="active",
            valid_from="2026-03-14",
            valid_to="2026-03-14",
            updated_at=now - timedelta(minutes=2),
            created_at=now - timedelta(minutes=2),
            attributes={
                "pattern_type": "interaction",
                "memory_domain": "interaction",
                "daypart": "morning",
            },
        )

        result = LongTermRetentionPolicy(timezone_name="UTC", archive_enabled=True).apply(
            objects=(pattern,),
            now=now,
        )

        self.assertEqual(result.expired_objects, ())
        self.assertEqual(result.archived_objects, ())
        self.assertEqual(result.pruned_memory_ids, ())
        self.assertEqual(result.kept_objects[0].status, "active")

    def test_store_retention_writes_archive_snapshot(self) -> None:
        now = datetime(2026, 3, 15, 12, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                long_term_memory_enabled=True,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
            )
            store = LongTermStructuredStore.from_config(config)
            episode = LongTermMemoryObjectV1(
                memory_id="episode:1",
                kind="episode",
                summary="Episode summary",
                source=self._source(),
                updated_at=now - timedelta(days=10),
                created_at=now - timedelta(days=10),
            )
            store.write_snapshot(objects=(episode,))

            result = LongTermRetentionPolicy(timezone_name="UTC", archive_enabled=True).apply(
                objects=store.load_objects(),
                now=now,
            )
            store.apply_retention(result)

            active_ids = {item.memory_id for item in store.load_objects()}
            archived_ids = {item.memory_id for item in store.load_archived_objects()}

        self.assertEqual(active_ids, set())
        self.assertEqual(archived_ids, {"episode:1"})
