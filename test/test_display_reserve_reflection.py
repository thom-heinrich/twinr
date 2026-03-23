from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.memory.longterm.core.models import (
    LongTermMemoryObjectV1,
    LongTermMidtermPacketV1,
    LongTermSourceRefV1,
)
from twinr.proactive.runtime.display_reserve_reflection import (
    load_display_reserve_reflection_candidates,
)


class _FakeObjectStore:
    def __init__(self, objects):
        self._objects = tuple(objects)

    def load_objects(self):
        return self._objects


class _FakeMidtermStore:
    def __init__(self, packets):
        self._packets = tuple(packets)

    def load_packets(self):
        return self._packets


class _FakeMemoryService:
    def __init__(self, *, objects=(), packets=()):
        self.object_store = _FakeObjectStore(objects)
        self.midterm_store = _FakeMidtermStore(packets)


class DisplayReserveReflectionTests(unittest.TestCase):
    def test_reflection_candidates_surface_continuity_and_suppress_operational_packets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                display_reserve_bus_reflection_candidate_limit=5,
            )
            now = datetime(2026, 3, 22, 15, 0, tzinfo=timezone.utc)
            summary = LongTermMemoryObjectV1(
                memory_id="thread:person:janina",
                kind="summary",
                summary="Ongoing thread about Janina: Arzttermin; neue Schule.",
                details="Reflected from multiple related long-term memory objects.",
                source=LongTermSourceRefV1(source_type="reflection", event_ids=("turn:1",)),
                status="active",
                confidence=0.84,
                sensitivity="normal",
                slot_key="thread:person:janina",
                value_key="person:janina",
                updated_at=now - timedelta(hours=6),
                attributes={
                    "summary_type": "thread",
                    "person_name": "Janina",
                    "memory_domain": "thread",
                },
            )
            preference_packet = LongTermMidtermPacketV1(
                packet_id="midterm:user_preference_morning_coffee_melitta",
                kind="preference",
                summary="User likes to drink Melitta coffee in the morning.",
                details="This can help with relevant small talk about morning routines.",
                query_hints=("Melitta", "morning coffee"),
                sensitivity="normal",
                updated_at=now - timedelta(hours=3),
            )
            continuity_packet = LongTermMidtermPacketV1(
                packet_id="midterm:turn:recent",
                kind="recent_turn_continuity",
                summary="You asked how the doctor appointment went yesterday.",
                details="Immediate continuity only.",
                query_hints=("doctor appointment", "yesterday"),
                sensitivity="normal",
                updated_at=now - timedelta(minutes=20),
                attributes={"persistence_scope": "turn_continuity"},
            )
            conversation_packet = LongTermMidtermPacketV1(
                packet_id="midterm:conversation:check_in",
                kind="conversation_context",
                summary="The user recently checked in and asked how Twinr is doing.",
                details="A warm small-talk opener may fit as shared continuity.",
                query_hints=("small talk", "check in"),
                sensitivity="normal",
                updated_at=now - timedelta(hours=1),
            )
            quality_packet = LongTermMidtermPacketV1(
                packet_id="midterm:quality:session",
                kind="interaction_quality",
                summary="The last session felt warm and calm.",
                details="Operational interaction quality only.",
                query_hints=("warm", "calm"),
                sensitivity="normal",
                updated_at=now - timedelta(minutes=10),
            )
            conversation_state_packet = LongTermMidtermPacketV1(
                packet_id="midterm:conversation:recent_farewells_20260322",
                kind="conversation_state",
                summary="The user recently ended the interaction with brief German farewells and closure cues.",
                details="No explicit follow-up tasks or open questions were left in these turns.",
                query_hints=("German farewell", "until next time"),
                sensitivity="normal",
                updated_at=now - timedelta(minutes=12),
            )
            device_packet = LongTermMidtermPacketV1(
                packet_id="midterm:device:room_guard",
                kind="device_context",
                summary="Ambiguous room guard should keep watching the room state.",
                details="Operational device/watch-state only.",
                query_hints=("ambiguous room guard",),
                sensitivity="normal",
                updated_at=now - timedelta(minutes=8),
            )
            memory_service = _FakeMemoryService(
                objects=(summary,),
                packets=(
                    preference_packet,
                    continuity_packet,
                    conversation_packet,
                    quality_packet,
                    conversation_state_packet,
                    device_packet,
                ),
            )

            candidates = load_display_reserve_reflection_candidates(
                memory_service,
                config=config,
                local_now=now,
                max_items=4,
            )

        by_topic = {candidate.topic_key: candidate for candidate in candidates}
        sources = {candidate.source for candidate in candidates}
        self.assertEqual(len(candidates), 4)
        self.assertIn("thread:person:janina", by_topic)
        self.assertIn("morning coffee", by_topic)
        self.assertIn("small talk", by_topic)
        self.assertIn("doctor appointment", by_topic)
        self.assertIn("reflection_summary", sources)
        self.assertIn("reflection_midterm", sources)
        self.assertEqual(by_topic["small talk"].attention_state, "shared_thread")
        self.assertEqual(by_topic["doctor appointment"].attention_state, "shared_thread")
        self.assertEqual(by_topic["small talk"].candidate_family, "reflection_thread")
        self.assertEqual(by_topic["doctor appointment"].candidate_family, "reflection_thread")
        self.assertNotIn("Recent conversation continuity", by_topic["doctor appointment"].headline)
        self.assertNotIn("Immediate continuity only", by_topic["doctor appointment"].body)
        self.assertEqual(
            (by_topic["doctor appointment"].generation_context or {}).get("display_anchor"),
            "Doctor appointment",
        )
        self.assertNotIn("warm", by_topic)

    def test_continuity_packets_do_not_surface_internal_midterm_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                display_reserve_bus_reflection_candidate_limit=5,
            )
            now = datetime(2026, 3, 22, 20, 0, tzinfo=timezone.utc)
            continuity_packet = LongTermMidtermPacketV1(
                packet_id="midterm:turn:f922982944b6f3370039",
                kind="recent_turn_continuity",
                summary="Recent conversation continuity from the latest user-assistant turn.",
                details=(
                    "This packet preserves immediate continuity until slower durable-memory enrichment finishes. "
                    "User said: wie geht's dir Assistant. Assistant answered: Mir geht's gut."
                ),
                query_hints=("wie", "geht's", "dir", "assistant"),
                sensitivity="normal",
                updated_at=now - timedelta(minutes=5),
                attributes={
                    "persistence_scope": "turn_continuity",
                    "display_anchor": "Wie geht's dir",
                },
            )
            memory_service = _FakeMemoryService(packets=(continuity_packet,))

            candidates = load_display_reserve_reflection_candidates(
                memory_service,
                config=config,
                local_now=now,
                max_items=3,
            )

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate.topic_key, "wie geht's dir")
        self.assertNotIn("Recent conversation continuity", candidate.headline)
        self.assertNotIn("This packet preserves immediate continuity", candidate.body)
        self.assertIn("Wie geht's dir", candidate.headline)
        self.assertEqual((candidate.generation_context or {}).get("display_anchor"), "Wie geht's dir")
        self.assertEqual((candidate.generation_context or {}).get("hook_hint"), "wie geht's dir")

    def test_continuity_packets_without_displayable_anchor_are_suppressed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                display_reserve_bus_reflection_candidate_limit=5,
            )
            now = datetime(2026, 3, 22, 20, 0, tzinfo=timezone.utc)
            continuity_packet = LongTermMidtermPacketV1(
                packet_id="midterm:turn:farewell",
                kind="recent_turn_continuity",
                summary="Recent conversation continuity from the latest user-assistant turn.",
                details="User said: bis zum nächsten Mal! Assistant answered: Bis zum nächsten Mal! Pass auf dich auf!",
                query_hints=("bis", "zum", "naechsten", "mal", "pass", "auf", "dich"),
                sensitivity="normal",
                updated_at=now - timedelta(minutes=5),
                attributes={"persistence_scope": "turn_continuity"},
            )
            memory_service = _FakeMemoryService(packets=(continuity_packet,))

            candidates = load_display_reserve_reflection_candidates(
                memory_service,
                config=config,
                local_now=now,
                max_items=3,
            )

        self.assertEqual(candidates, ())

    def test_policy_context_packets_do_not_surface_in_right_lane(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                display_reserve_bus_reflection_candidate_limit=5,
            )
            now = datetime(2026, 3, 22, 20, 0, tzinfo=timezone.utc)
            packet = LongTermMidtermPacketV1(
                packet_id="midterm:policy:guard",
                kind="policy_context",
                summary="Ambiguous-room conditions triggered guardrails that blocked targeted inference.",
                details="Operational policy state only.",
                query_hints=("ambiguous_room_guard", "guard active"),
                sensitivity="normal",
                updated_at=now - timedelta(minutes=5),
            )
            memory_service = _FakeMemoryService(packets=(packet,))

            candidates = load_display_reserve_reflection_candidates(
                memory_service,
                config=config,
                local_now=now,
                max_items=3,
            )

        self.assertEqual(candidates, ())


if __name__ == "__main__":
    unittest.main()
