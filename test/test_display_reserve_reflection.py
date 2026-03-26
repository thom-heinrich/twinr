from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile
import unittest
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.longterm.core.models import (
    LongTermMemoryObjectV1,
    LongTermMidtermPacketV1,
    LongTermSourceRefV1,
)
from twinr.memory.longterm.runtime.service import LongTermMemoryService
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


def _load_candidates(
    memory_service: _FakeMemoryService,
    *,
    config: TwinrConfig,
    local_now: datetime,
    max_items: int,
):
    return load_display_reserve_reflection_candidates(
        cast(LongTermMemoryService, memory_service),
        config=config,
        local_now=local_now,
        max_items=max_items,
    )


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
            grounded_preference_packet = LongTermMidtermPacketV1(
                packet_id="midterm:user_preference_coffee_grounded",
                kind="preference",
                summary="User likes to drink Melitta coffee in the morning.",
                details="This can help with relevant small talk about morning routines.",
                query_hints=("Melitta", "morning coffee"),
                sensitivity="normal",
                updated_at=now - timedelta(hours=2),
                attributes={
                    "display_anchor": "Dein Kaffee am Morgen",
                    "transcript_excerpt": "Mein Kaffee am Morgen ist mir wichtig.",
                },
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
            interaction_packet = LongTermMidtermPacketV1(
                packet_id="midterm:interaction:green_button",
                kind="interaction",
                summary="The user asked about the green button.",
                details="Generic interaction residue only.",
                query_hints=("green button",),
                sensitivity="normal",
                updated_at=now - timedelta(minutes=15),
            )
            memory_service = _FakeMemoryService(
                objects=(summary,),
                packets=(
                    preference_packet,
                    grounded_preference_packet,
                    conversation_packet,
                    quality_packet,
                    conversation_state_packet,
                    device_packet,
                    interaction_packet,
                ),
            )

            candidates = _load_candidates(
                memory_service,
                config=config,
                local_now=now,
                max_items=4,
            )

        by_topic = {candidate.topic_key: candidate for candidate in candidates}
        sources = {candidate.source for candidate in candidates}
        self.assertEqual(len(candidates), 2)
        self.assertIn("thread:person:janina", by_topic)
        self.assertIn("dein kaffee am morgen", by_topic)
        self.assertIn("reflection_summary", sources)
        self.assertIn("reflection_midterm", sources)
        self.assertEqual(by_topic["dein kaffee am morgen"].candidate_family, "reflection_preference")
        self.assertFalse(by_topic["dein kaffee am morgen"].headline.endswith("?"))
        self.assertTrue(by_topic["dein kaffee am morgen"].body.endswith("?"))
        self.assertNotIn("morning coffee", by_topic)
        self.assertNotIn("small talk", by_topic)
        self.assertNotIn("green button", by_topic)
        self.assertNotIn("warm", by_topic)

    def test_recent_turn_continuity_packets_are_hidden_from_visible_lane(self) -> None:
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

            candidates = _load_candidates(
                memory_service,
                config=config,
                local_now=now,
                max_items=3,
            )

        self.assertEqual(candidates, ())

    def test_conversation_context_packets_with_structured_anchor_still_surface(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                display_reserve_bus_reflection_candidate_limit=5,
            )
            now = datetime(2026, 3, 22, 20, 0, tzinfo=timezone.utc)
            continuity_packet = LongTermMidtermPacketV1(
                packet_id="midterm:conversation:arzttermin",
                kind="conversation_context",
                summary="Shared conversation context around the doctor appointment yesterday.",
                details="A slower continuity packet with a stable shared thread.",
                query_hints=("doctor appointment", "yesterday"),
                sensitivity="normal",
                updated_at=now - timedelta(hours=4),
                attributes={
                    "packet_scope": "shared_thread",
                    "display_anchor": "Arzttermin gestern",
                    "transcript_excerpt": "Wie war der Arzttermin gestern?",
                },
            )
            memory_service = _FakeMemoryService(packets=(continuity_packet,))

            candidates = _load_candidates(
                memory_service,
                config=config,
                local_now=now,
                max_items=3,
            )

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate.topic_key, "arzttermin gestern")
        self.assertEqual(candidate.attention_state, "shared_thread")
        self.assertEqual(candidate.candidate_family, "reflection_thread")
        self.assertEqual((candidate.generation_context or {}).get("display_anchor"), "Arzttermin gestern")
        self.assertEqual((candidate.generation_context or {}).get("display_goal"), "call_back_to_earlier_conversation")

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

            candidates = _load_candidates(
                memory_service,
                config=config,
                local_now=now,
                max_items=3,
            )

        self.assertEqual(candidates, ())

    def test_continuity_packets_with_only_transcript_excerpt_are_suppressed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                display_reserve_bus_reflection_candidate_limit=5,
            )
            now = datetime(2026, 3, 26, 8, 30, tzinfo=timezone.utc)
            continuity_packet = LongTermMidtermPacketV1(
                packet_id="midterm:turn:unfinished_sentence",
                kind="recent_turn_continuity",
                summary="Recent conversation continuity from the latest user-assistant turn.",
                details="Immediate continuity only.",
                query_hints=("angefangener Satz", "vorhin"),
                sensitivity="normal",
                updated_at=now - timedelta(minutes=4),
                attributes={
                    "persistence_scope": "turn_continuity",
                    "transcript_excerpt": "Ich hatte vorhin noch etwas angefangen.",
                    "response_excerpt": "Wenn du magst, kannst du da spaeter weitermachen.",
                },
            )
            memory_service = _FakeMemoryService(packets=(continuity_packet,))

            candidates = _load_candidates(
                memory_service,
                config=config,
                local_now=now,
                max_items=3,
            )

        self.assertEqual(candidates, ())

    def test_summary_objects_without_structured_display_anchor_are_suppressed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                display_reserve_bus_reflection_candidate_limit=5,
            )
            now = datetime(2026, 3, 24, 15, 0, tzinfo=timezone.utc)
            summary = LongTermMemoryObjectV1(
                memory_id="summary:place:schwarzen_weg",
                kind="summary",
                summary="The assistant answered a weather query for Schwarzen Weg, Berlin.",
                details="Internal reflected event summary only.",
                source=LongTermSourceRefV1(source_type="reflection", event_ids=("turn:99",)),
                status="active",
                confidence=0.91,
                sensitivity="normal",
                slot_key="summary:place:schwarzen_weg",
                value_key="weather_query",
                updated_at=now - timedelta(hours=2),
                attributes={
                    "summary_type": "thread",
                    "memory_domain": "thread",
                },
            )
            memory_service = _FakeMemoryService(objects=(summary,))

            candidates = _load_candidates(
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

            candidates = _load_candidates(
                memory_service,
                config=config,
                local_now=now,
                max_items=3,
            )

        self.assertEqual(candidates, ())


if __name__ == "__main__":
    unittest.main()
