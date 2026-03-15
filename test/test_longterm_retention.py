from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.memory.longterm.models import LongTermMemoryObjectV1, LongTermSourceRefV1
from twinr.memory.longterm.retention import LongTermRetentionPolicy


def _source() -> LongTermSourceRefV1:
    return LongTermSourceRefV1(
        source_type="conversation_turn",
        event_ids=("turn:test",),
        speaker="user",
        modality="voice",
    )


class LongTermRetentionPolicyTests(unittest.TestCase):
    def test_retention_expires_past_time_bound_event(self) -> None:
        policy = LongTermRetentionPolicy()
        item = LongTermMemoryObjectV1(
            memory_id="event:appointment",
            kind="event",
            summary="Janina has eye laser treatment on 2026-03-14.",
            source=_source(),
            status="active",
            confidence=0.9,
            valid_from="2026-03-14",
            valid_to="2026-03-14",
            sensitivity="sensitive",
            attributes={
                "memory_domain": "appointment",
                "event_domain": "appointment",
            },
        )

        result = policy.apply(
            objects=(item,),
            now=datetime(2026, 3, 16, 10, 0, tzinfo=timezone.utc),
        )

        self.assertEqual(len(result.expired_objects), 1)
        self.assertEqual(result.expired_objects[0].status, "expired")

    def test_retention_prunes_old_episode_and_old_observation(self) -> None:
        policy = LongTermRetentionPolicy(ephemeral_episode_days=14, ephemeral_observation_days=2)
        old_episode = LongTermMemoryObjectV1(
            memory_id="episode:old",
            kind="episode",
            summary="Conversation turn recorded for long-term memory.",
            source=_source(),
            status="active",
            confidence=1.0,
            created_at=datetime(2026, 2, 1, 10, 0, tzinfo=timezone.utc),
            updated_at=datetime(2026, 2, 1, 10, 0, tzinfo=timezone.utc),
        )
        old_observation = LongTermMemoryObjectV1(
            memory_id="observation:old",
            kind="situational_observation",
            summary="The user described the day as warm.",
            source=_source(),
            status="active",
            confidence=0.7,
            created_at=datetime(2026, 3, 10, 10, 0, tzinfo=timezone.utc),
            updated_at=datetime(2026, 3, 10, 10, 0, tzinfo=timezone.utc),
        )
        durable = LongTermMemoryObjectV1(
            memory_id="fact:janina_wife",
            kind="relationship_fact",
            summary="Janina is the user's wife.",
            source=_source(),
            status="active",
            confidence=0.98,
        )

        result = policy.apply(
            objects=(old_episode, old_observation, durable),
            now=datetime(2026, 3, 16, 10, 0, tzinfo=timezone.utc),
        )

        self.assertIn("episode:old", result.pruned_memory_ids)
        self.assertIn("observation:old", result.pruned_memory_ids)
        kept_ids = {item.memory_id for item in result.kept_objects}
        self.assertIn("fact:janina_wife", kept_ids)


if __name__ == "__main__":
    unittest.main()
