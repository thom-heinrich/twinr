from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.memory.longterm import (
    LongTermConversationTurn,
    LongTermMemoryObjectV1,
    LongTermMemoryReflector,
    LongTermMidtermStore,
    LongTermSourceRefV1,
)
from twinr.memory.longterm.reasoning.midterm import (
    _memory_object_to_prompt_payload,
    _midterm_reflection_schema,
    _normalize_reflection_result,
)
from twinr.memory.longterm.reasoning.turn_continuity import LongTermTurnContinuityCompiler
from twinr.memory.longterm.retrieval.retriever import LongTermRetriever
from twinr.memory.longterm.storage.store import LongTermStructuredStore
from twinr.memory.context_store import PromptContextStore
from twinr.memory.query_normalization import LongTermQueryProfile
from twinr.memory.longterm.reasoning.conflicts import LongTermConflictResolver
from twinr.memory.longterm.retrieval.subtext import LongTermSubtextBuilder
from twinr.memory.chonkydb.personal_graph import TwinrPersonalGraphStore


def _config(root: str) -> TwinrConfig:
    return TwinrConfig(
        project_root=root,
        personality_dir="personality",
        memory_markdown_path=str(Path(root) / "state" / "MEMORY.md"),
        long_term_memory_enabled=True,
        long_term_memory_path=str(Path(root) / "state" / "chonkydb"),
        long_term_memory_recall_limit=4,
        long_term_memory_midterm_enabled=True,
        long_term_memory_midterm_limit=3,
        user_display_name="Erika",
    )


def _source(event_id: str) -> LongTermSourceRefV1:
    return LongTermSourceRefV1(
        source_type="conversation_turn",
        event_ids=(event_id,),
        speaker="user",
        modality="voice",
    )


class StubReflectionProgram:
    def compile_reflection(
        self,
        *,
        objects,
        timezone_name: str,
        packet_limit: int,
    ):
        del timezone_name
        del packet_limit
        object_ids = {item.memory_id for item in objects}
        if {"fact:janina_wife", "event:janina_eye_doctor_today"}.issubset(object_ids):
            return {
                "midterm_packets": [
                    {
                        "packet_id": "midterm:janina_today",
                        "kind": "recent_life_bundle",
                        "summary": "Janina is the user's wife and has an eye doctor appointment today.",
                        "details": "Recent context suggests that questions about Janina today may relate to that appointment.",
                        "source_memory_ids": [
                            "fact:janina_wife",
                            "event:janina_eye_doctor_today",
                        ],
                        "query_hints": [
                            "janina",
                            "wife",
                            "eye doctor",
                            "today",
                        ],
                        "sensitivity": "sensitive",
                        "valid_from": "2026-03-15",
                        "valid_to": "2026-03-15",
                        "attributes": {
                            "scope": "recent_window",
                            "focus_refs": ["person:janina"],
                        },
                    }
                ]
            }
        return {"midterm_packets": []}


class _EnglishPacketFromGermanSourceProgram:
    def compile_reflection(
        self,
        *,
        objects,
        timezone_name: str,
        packet_limit: int,
    ):
        del timezone_name
        del packet_limit
        object_ids = {item.memory_id for item in objects}
        if "event:lea_soup_dropoff" not in object_ids:
            return {"midterm_packets": []}
        return {
            "midterm_packets": [
                {
                    "packet_id": "midterm:lea_soup_dropoff",
                    "kind": "upcoming_event",
                    "summary": "Lea will bring a thermos of lentil soup this evening.",
                    "details": "At 7 PM Lea will drop off homemade lentil soup in a thermos.",
                    "source_memory_ids": ["event:lea_soup_dropoff"],
                    "query_hints": ["lea", "thermos", "lentil soup", "7 pm"],
                    "sensitivity": "normal",
                    "valid_from": None,
                    "valid_to": None,
                }
            ]
        }


class _FakeRemoteState:
    def __init__(self) -> None:
        self.enabled = True
        self.snapshots: dict[str, dict[str, object]] = {}

    def load_snapshot(self, *, snapshot_kind: str, local_path=None):
        del local_path
        payload = self.snapshots.get(snapshot_kind)
        return dict(payload) if isinstance(payload, dict) else None

    def save_snapshot(self, *, snapshot_kind: str, payload):
        self.snapshots[snapshot_kind] = dict(payload)


class LongTermMidtermTests(unittest.TestCase):
    def test_midterm_schema_is_openai_strict_compatible(self) -> None:
        schema = _midterm_reflection_schema(max_packets=3)
        packet_schema = schema["properties"]["midterm_packets"]["items"]
        properties = packet_schema["properties"]
        required = packet_schema["required"]

        self.assertEqual(set(required), set(properties))
        self.assertEqual(properties["details"]["anyOf"][1]["type"], "null")
        self.assertEqual(properties["valid_from"]["anyOf"][1]["type"], "null")
        self.assertEqual(properties["valid_to"]["anyOf"][1]["type"], "null")

    def test_reflector_can_compile_midterm_packets(self) -> None:
        reflector = LongTermMemoryReflector(program=StubReflectionProgram(), midterm_packet_limit=3)
        wife = LongTermMemoryObjectV1(
            memory_id="fact:janina_wife",
            kind="fact",
            summary="Janina is the user's wife.",
            source=_source("turn:1"),
            status="active",
            confidence=0.98,
            slot_key="relationship:user:main:wife",
            value_key="person:janina",
            sensitivity="private",
            attributes={
                "person_ref": "person:janina",
                "person_name": "Janina",
                "relation": "wife",
                "fact_type": "relationship",
                "support_count": 2,
            },
        )
        appointment = LongTermMemoryObjectV1(
            memory_id="event:janina_eye_doctor_today",
            kind="event",
            summary="Janina has an eye doctor appointment today.",
            source=_source("turn:2"),
            status="active",
            confidence=0.93,
            slot_key="event:person:janina:eye_doctor:2026-03-15",
            value_key="appointment:janina:eye_doctor:2026-03-15",
            sensitivity="sensitive",
            valid_from="2026-03-15",
            valid_to="2026-03-15",
            attributes={
                "person_ref": "person:janina",
                "person_name": "Janina",
                "action": "eye doctor appointment",
                "memory_domain": "appointment",
            },
        )

        result = reflector.reflect(objects=(wife, appointment))

        self.assertEqual(len(result.midterm_packets), 1)
        packet = result.midterm_packets[0]
        self.assertEqual(packet.packet_id, "midterm:janina_today")
        self.assertEqual(packet.kind, "recent_life_bundle")
        self.assertIn("Janina", packet.summary)
        self.assertIn("eye doctor", packet.summary)
        self.assertEqual(packet.source_memory_ids, ("fact:janina_wife", "event:janina_eye_doctor_today"))

    def test_reflector_preserves_source_language_hints_for_canonical_midterm_packets(self) -> None:
        dropoff = LongTermMemoryObjectV1(
            memory_id="event:lea_soup_dropoff",
            kind="event",
            summary="Lea bringt heute Abend um 19 Uhr eine Thermoskanne mit Linsensuppe vorbei.",
            details="Heute Abend kommt Lea vorbei und bringt dir eine Thermoskanne mit selbstgemachter Linsensuppe.",
            source=_source("turn:lea"),
            status="active",
            confidence=0.95,
            slot_key="event:lea:soup_dropoff:2026-03-21T19:00:00+01:00",
            value_key="event:lea:soup_dropoff",
            sensitivity="normal",
            valid_from="2026-03-21T19:00:00+01:00",
            valid_to="2026-03-21T20:00:00+01:00",
            attributes={
                "person_name": "Lea",
                "item": "Thermoskanne mit Linsensuppe",
                "memory_domain": "delivery",
            },
        )

        prompt_payload = _memory_object_to_prompt_payload(dropoff)
        normalized = _normalize_reflection_result(
            _EnglishPacketFromGermanSourceProgram().compile_reflection(
                objects=(dropoff,),
                timezone_name="Europe/Berlin",
                packet_limit=3,
            ),
            valid_memory_ids={"event:lea_soup_dropoff"},
            packet_limit=3,
            source_payload_by_memory_id={"event:lea_soup_dropoff": prompt_payload},
        )

        self.assertEqual(len(normalized["midterm_packets"]), 1)
        packet = LongTermMidtermStore.packet_type.from_payload(normalized["midterm_packets"][0])
        self.assertIn("thermos", packet.query_hints)
        self.assertTrue(
            any("Thermoskanne" in hint or "Linsensuppe" in hint for hint in packet.query_hints),
            packet.query_hints,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            store = LongTermMidtermStore.from_config(_config(temp_dir))
            store.save_packets(packets=(packet,))
            matches = store.select_relevant_packets("Was bringt Lea heute Abend vorbei?", limit=2)

        self.assertEqual([item.packet_id for item in matches], ["midterm:lea_soup_dropoff"])

    def test_turn_continuity_compiler_preserves_source_language_turn_recall(self) -> None:
        packet = LongTermTurnContinuityCompiler().compile_packet(
            turn=LongTermConversationTurn(
                transcript="Meine Tochter Lea bringt mir heute Abend eine Thermoskanne mit Linsensuppe vorbei.",
                response="Ich merke mir, dass Lea dir heute Abend die Thermoskanne mit Linsensuppe bringt.",
                source="conversation",
            )
        )

        self.assertIsNotNone(packet)
        assert packet is not None
        self.assertEqual(packet.kind, "recent_turn_continuity")
        self.assertIn("latest user-assistant turn", packet.summary)
        self.assertIn("Thermoskanne", packet.details or "")
        self.assertIn("Linsensuppe", packet.details or "")
        self.assertIn("lea", packet.query_hints)
        self.assertTrue(
            any("thermoskanne" in hint.lower() or "linsensuppe" in hint.lower() for hint in packet.query_hints),
            packet.query_hints,
        )
        self.assertEqual(packet.attributes["persistence_scope"], "turn_continuity")

    def test_midterm_store_selects_query_relevant_packets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _config(temp_dir)
            store = LongTermMidtermStore.from_config(config)
            store.save_packets(
                packets=(
                    store.packet_type(
                        packet_id="midterm:janina_today",
                        kind="recent_life_bundle",
                        summary="Janina has an eye doctor appointment today.",
                        details="Useful when the user asks about Janina today.",
                        source_memory_ids=("event:janina_eye_doctor_today",),
                        query_hints=("janina", "eye doctor", "today"),
                        sensitivity="sensitive",
                    ),
                    store.packet_type(
                        packet_id="midterm:tea_shopping",
                        kind="shopping_bundle",
                        summary="The user usually buys tea at Laden Seidel.",
                        details="Useful for nearby tea suggestions.",
                        source_memory_ids=("fact:tea_store",),
                        query_hints=("tea", "laden seidel"),
                        sensitivity="normal",
                    ),
                )
            )

            janina_packets = store.select_relevant_packets("How is Janina doing today?", limit=2)
            math_packets = store.select_relevant_packets("What is 27 times 14?", limit=2)

        self.assertEqual([item.packet_id for item in janina_packets], ["midterm:janina_today"])
        self.assertEqual(math_packets, ())

    def test_midterm_store_matches_compound_terms_without_off_topic_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _config(temp_dir)
            store = LongTermMidtermStore.from_config(config)
            store.save_packets(
                packets=(
                    store.packet_type(
                        packet_id="midterm:jam_restart",
                        kind="adaptive_restart_recall_policy",
                        summary="Persistent restart recall for this stable durable memory: Inzwischen magst du lieber Aprikosenmarmelade.",
                        details="Use this packet as direct grounding after fresh runtime restarts when the current turn overlaps the same topic.",
                        source_memory_ids=("fact:jam_preference_new",),
                        query_hints=("preference:breakfast:jam", "bestaetigt"),
                        attributes={"persistence_scope": "restart_recall"},
                    ),
                )
            )

            jam_packets = store.select_relevant_packets(
                "Welche Marmelade ist jetzt als bestaetigt gespeichert?",
                limit=2,
            )
            control_packets = store.select_relevant_packets("Was ist ein Regenbogen?", limit=2)

        self.assertEqual([item.packet_id for item in jam_packets], ["midterm:jam_restart"])
        self.assertEqual(control_packets, ())

    def test_ensure_remote_snapshot_seeds_empty_midterm_store(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            store = LongTermMidtermStore(
                base_path=Path(temp_dir) / "state" / "chonkydb",
                remote_state=remote_state,
            )

            with self.assertNoLogs("twinr.memory.longterm.storage.midterm_store", level=logging.WARNING):
                created = store.ensure_remote_snapshot()

        self.assertTrue(created)
        self.assertEqual(
            remote_state.snapshots["midterm"],
            {"schema": "twinr_memory_midterm_store", "version": 1, "packets": []},
        )

    def test_save_packets_writes_world_readable_midterm_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LongTermMidtermStore(base_path=Path(temp_dir) / "state" / "chonkydb")

            store.save_packets(
                packets=(
                    store.packet_type(
                        packet_id="midterm:tea_preference",
                        kind="preference_bundle",
                        summary="The user prefers Oolong tea in the afternoon.",
                        source_memory_ids=("fact:drink_preference",),
                        query_hints=("oolong", "tea", "afternoon"),
                        sensitivity="normal",
                    ),
                )
            )

            mode = stat.S_IMODE(store.packets_path.stat().st_mode)

        self.assertEqual(mode, 0o644)

    def test_load_packets_ignores_missing_local_snapshot_without_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LongTermMidtermStore(base_path=Path(temp_dir) / "state" / "chonkydb")

            with self.assertNoLogs("twinr.memory.longterm.storage.midterm_store", level=logging.WARNING):
                packets = store.load_packets()

        self.assertEqual(packets, ())

    def test_load_packets_uses_remote_snapshot_without_warning_when_local_cache_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            remote_state = _FakeRemoteState()
            remote_state.snapshots["midterm"] = {
                "schema": "twinr_memory_midterm_store",
                "version": 1,
                "packets": [
                    {
                        "packet_id": "midterm:tea_preference",
                        "kind": "preference_bundle",
                        "summary": "The user prefers Oolong tea in the afternoon.",
                        "details": "Useful for drink recommendations and recall questions.",
                        "source_memory_ids": ["fact:drink_preference"],
                        "query_hints": ["oolong", "tea", "afternoon"],
                        "sensitivity": "normal",
                    }
                ],
            }
            store = LongTermMidtermStore(
                base_path=Path(temp_dir) / "state" / "chonkydb",
                remote_state=remote_state,
            )

            with self.assertNoLogs("twinr.memory.longterm.storage.midterm_store", level=logging.WARNING):
                packets = store.load_packets()

        self.assertEqual([item.packet_id for item in packets], ["midterm:tea_preference"])

    def test_replace_packets_with_attribute_preserves_other_packet_scopes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _config(temp_dir)
            store = LongTermMidtermStore.from_config(config)
            store.save_packets(
                packets=(
                    store.packet_type(
                        packet_id="midterm:reflection",
                        kind="recent_life_bundle",
                        summary="Reflection packet",
                        source_memory_ids=("fact:reflection",),
                        query_hints=("reflection",),
                        sensitivity="normal",
                    ),
                    store.packet_type(
                        packet_id="adaptive:restart:old",
                        kind="adaptive_restart_recall_policy",
                        summary="Old restart packet",
                        source_memory_ids=("fact:old",),
                        query_hints=("old",),
                        sensitivity="normal",
                        attributes={"persistence_scope": "restart_recall"},
                    ),
                )
            )

            store.replace_packets_with_attribute(
                packets=(
                    store.packet_type(
                        packet_id="adaptive:restart:new",
                        kind="adaptive_restart_recall_policy",
                        summary="New restart packet",
                        source_memory_ids=("fact:new",),
                        query_hints=("new",),
                        sensitivity="normal",
                        attributes={"persistence_scope": "restart_recall"},
                    ),
                ),
                attribute_key="persistence_scope",
                attribute_value="restart_recall",
            )
            loaded = store.load_packets()

        self.assertEqual(
            [item.packet_id for item in loaded],
            ["adaptive:restart:new", "midterm:reflection"],
        )

    def test_retriever_includes_midterm_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _config(temp_dir)
            prompt_context_store = PromptContextStore.from_config(config)
            graph_store = TwinrPersonalGraphStore.from_config(config)
            object_store = LongTermStructuredStore.from_config(config)
            midterm_store = LongTermMidtermStore.from_config(config)
            midterm_store.save_packets(
                packets=(
                    midterm_store.packet_type(
                        packet_id="midterm:janina_today",
                        kind="recent_life_bundle",
                        summary="Janina has an eye doctor appointment today.",
                        details="Useful when the user asks about Janina today.",
                        source_memory_ids=("event:janina_eye_doctor_today",),
                        query_hints=("janina", "eye doctor", "today"),
                        sensitivity="sensitive",
                    ),
                )
            )
            retriever = LongTermRetriever(
                config=config,
                prompt_context_store=prompt_context_store,
                graph_store=graph_store,
                object_store=object_store,
                midterm_store=midterm_store,
                conflict_resolver=LongTermConflictResolver(),
                subtext_builder=LongTermSubtextBuilder(config=config, graph_store=graph_store),
            )

            context = retriever.build_context(
                query=LongTermQueryProfile.from_text(
                    "Wie geht es Janina heute?",
                    canonical_english_text="How is Janina doing today?",
                ),
                original_query_text="Wie geht es Janina heute?",
            )

        self.assertIsNotNone(context.midterm_context)
        self.assertIn("twinr_long_term_midterm_context_v1", context.midterm_context or "")
        self.assertIn("Janina has an eye doctor appointment today.", context.midterm_context or "")


if __name__ == "__main__":
    unittest.main()
