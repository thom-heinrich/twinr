from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
import tempfile
import unittest
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.memory.longterm.reasoning.conflicts import LongTermConflictResolver
from twinr.memory.longterm.core.models import (
    LongTermConsolidationResultV1,
    LongTermMemoryConflictV1,
    LongTermMemoryObjectV1,
    LongTermSourceRefV1,
)
from twinr.memory.longterm.runtime.service import LongTermMemoryService
from twinr.memory.longterm.storage.store import LongTermStructuredStore


_TEST_CORINNA_PHONE_OLD = "+15555551234"
_TEST_CORINNA_PHONE_NEW = "+15555558877"


def _config(root: str) -> TwinrConfig:
    return TwinrConfig(
        project_root=root,
        personality_dir="personality",
        memory_markdown_path=str(Path(root) / "state" / "MEMORY.md"),
        long_term_memory_path=str(Path(root) / "state" / "chonkydb"),
        long_term_memory_enabled=True,
    )


def _source(event_id: str = "turn:test") -> LongTermSourceRefV1:
    return LongTermSourceRefV1(
        source_type="conversation_turn",
        event_ids=(event_id,),
        speaker="user",
        modality="voice",
    )


def _existing_phone() -> LongTermMemoryObjectV1:
    return LongTermMemoryObjectV1(
        memory_id="fact:corinna_phone_old",
        kind="contact_method_fact",
        summary=f"Corinna Maier can be reached at {_TEST_CORINNA_PHONE_OLD}.",
        details="Use the mobile number ending in 1234.",
        source=_source("turn:1"),
        status="active",
        confidence=0.95,
        slot_key="contact:person:corinna_maier:phone",
        value_key=_TEST_CORINNA_PHONE_OLD,
        attributes={"person_ref": "person:corinna_maier"},
    )


def _candidate_phone() -> LongTermMemoryObjectV1:
    return LongTermMemoryObjectV1(
        memory_id="fact:corinna_phone_new",
        kind="contact_method_fact",
        summary=f"Corinna Maier can be reached at {_TEST_CORINNA_PHONE_NEW}.",
        details="Use the office number ending in 8877.",
        source=_source("turn:2"),
        status="uncertain",
        confidence=0.92,
        slot_key="contact:person:corinna_maier:phone",
        value_key=_TEST_CORINNA_PHONE_NEW,
        attributes={"person_ref": "person:corinna_maier"},
    )


def _phone_conflict() -> LongTermMemoryConflictV1:
    return LongTermMemoryConflictV1(
        slot_key="contact:person:corinna_maier:phone",
        candidate_memory_id="fact:corinna_phone_new",
        existing_memory_ids=("fact:corinna_phone_old",),
        question="I have more than one contact detail for this person. Which one should I use?",
        reason="Conflicting active memories exist for slot contact:person:corinna_maier:phone.",
    )


class LongTermConflictResolverTests(unittest.TestCase):
    def test_build_queue_items_includes_concrete_options(self) -> None:
        resolver = LongTermConflictResolver()

        queue = resolver.build_queue_items(
            conflicts=(_phone_conflict(),),
            objects=(_existing_phone(), _candidate_phone()),
        )

        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0].slot_key, "contact:person:corinna_maier:phone")
        self.assertEqual([item.memory_id for item in queue[0].options], ["fact:corinna_phone_old", "fact:corinna_phone_new"])
        self.assertIn("1234", queue[0].options[0].summary)
        self.assertIn("8877", queue[0].options[1].summary)

    def test_resolve_can_activate_candidate_and_supersede_existing_value(self) -> None:
        resolver = LongTermConflictResolver()

        result = resolver.resolve(
            conflict=_phone_conflict(),
            objects=(_existing_phone(), _candidate_phone()),
            remaining_conflicts=(_phone_conflict(),),
            selected_memory_id="fact:corinna_phone_new",
            now=datetime(2026, 3, 14, 15, 0, tzinfo=ZoneInfo("Europe/Berlin")),
        )

        updated = {item.memory_id: item for item in result.updated_objects}
        self.assertEqual(updated["fact:corinna_phone_new"].status, "active")
        self.assertTrue(updated["fact:corinna_phone_new"].confirmed_by_user)
        self.assertIn("fact:corinna_phone_old", updated["fact:corinna_phone_new"].supersedes)
        self.assertEqual(updated["fact:corinna_phone_old"].status, "superseded")
        self.assertEqual(result.remaining_conflicts, ())
        self.assertEqual(result.deleted_conflict_slot_keys, ("contact:person:corinna_maier:phone",))


class LongTermConflictStoreTests(unittest.TestCase):
    def test_apply_consolidation_keeps_unrelated_existing_conflicts(self) -> None:
        first_conflict = LongTermMemoryConflictV1(
            slot_key="contact:person:corinna_maier:phone",
            candidate_memory_id="fact:corinna_phone_new",
            existing_memory_ids=("fact:corinna_phone_old",),
            question="Which phone number should I use for Corinna Maier?",
            reason="Conflicting phone numbers exist.",
        )
        second_conflict = LongTermMemoryConflictV1(
            slot_key="relationship:user:main:spouse",
            candidate_memory_id="fact:spouse_name_new",
            existing_memory_ids=("fact:spouse_name_old",),
            question="Which spouse name is correct now?",
            reason="Conflicting spouse names exist.",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LongTermStructuredStore.from_config(_config(temp_dir))
            store.apply_consolidation(
                LongTermConsolidationResultV1(
                    turn_id="turn:1",
                    occurred_at=datetime(2026, 3, 14, 10, 0, tzinfo=ZoneInfo("Europe/Berlin")),
                    episodic_objects=(),
                    durable_objects=(),
                    deferred_objects=(),
                    conflicts=(first_conflict,),
                    graph_edges=(),
                )
            )
            store.apply_consolidation(
                LongTermConsolidationResultV1(
                    turn_id="turn:2",
                    occurred_at=datetime(2026, 3, 14, 11, 0, tzinfo=ZoneInfo("Europe/Berlin")),
                    episodic_objects=(),
                    durable_objects=(),
                    deferred_objects=(),
                    conflicts=(second_conflict,),
                    graph_edges=(),
                )
            )

            conflicts = store.load_conflicts()

        self.assertEqual({item.slot_key for item in conflicts}, {first_conflict.slot_key, second_conflict.slot_key})


class LongTermConflictServiceTests(unittest.TestCase):
    def test_service_can_select_and_resolve_open_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _config(temp_dir)
            service = LongTermMemoryService.from_config(config)
            service.object_store.apply_consolidation(
                LongTermConsolidationResultV1(
                    turn_id="turn:2",
                    occurred_at=datetime(2026, 3, 14, 12, 0, tzinfo=ZoneInfo("Europe/Berlin")),
                    episodic_objects=(),
                    durable_objects=(_existing_phone(),),
                    deferred_objects=(_candidate_phone(),),
                    conflicts=(_phone_conflict(),),
                    graph_edges=(),
                )
            )

            queue = service.select_conflict_queue("What is Corinna's number?")
            resolution = service.resolve_conflict(
                slot_key="contact:person:corinna_maier:phone",
                selected_memory_id="fact:corinna_phone_new",
            )
            context = service.build_provider_context("What is Corinna's number?")
            objects = {item.memory_id: item for item in service.object_store.load_objects()}
            conflicts = service.object_store.load_conflicts()
            service.shutdown()

        self.assertEqual(len(queue), 1)
        self.assertIn("8877", queue[0].options[1].summary)
        self.assertEqual(resolution.selected_memory_id, "fact:corinna_phone_new")
        self.assertEqual(objects["fact:corinna_phone_new"].status, "active")
        self.assertEqual(objects["fact:corinna_phone_old"].status, "superseded")
        self.assertEqual(conflicts, ())
        self.assertIsNone(context.conflict_context)


if __name__ == "__main__":
    unittest.main()
