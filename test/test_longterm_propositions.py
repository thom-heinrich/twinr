from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
import unittest
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.memory.longterm.core.models import LongTermSourceRefV1
from twinr.memory.longterm.ingestion.propositions import (
    LongTermTurnPropositionBundleV1,
    LongTermTurnPropositionCompiler,
    _turn_proposition_schema,
    structured_turn_program_from_config,
)
from twinr.config import TwinrConfig


class LongTermTurnPropositionTests(unittest.TestCase):
    def test_turn_proposition_schema_is_openai_strict_compatible(self) -> None:
        schema = _turn_proposition_schema()
        proposition_schema = schema["properties"]["propositions"]["items"]
        edge_schema = schema["properties"]["graph_edges"]["items"]

        self.assertEqual(set(proposition_schema["required"]), set(proposition_schema["properties"]))
        self.assertEqual(set(edge_schema["required"]), set(edge_schema["properties"]))
        self.assertEqual(proposition_schema["properties"]["details"]["anyOf"][1]["type"], "null")
        self.assertEqual(set(proposition_schema["properties"]["source_channel"]["enum"]), {"user_transcript", "assistant_response", "both"})
        self.assertEqual(proposition_schema["properties"]["subject_ref"]["anyOf"][1]["type"], "null")
        self.assertEqual(proposition_schema["properties"]["object_ref"]["anyOf"][1]["type"], "null")
        self.assertEqual(proposition_schema["properties"]["value_text"]["anyOf"][1]["type"], "null")

    def test_compiler_turns_generic_propositions_into_memory_objects(self) -> None:
        occurred_at = datetime(2026, 3, 15, 10, 30, tzinfo=ZoneInfo("Europe/Berlin"))
        payload = {
            "propositions": [
                {
                    "proposition_id": "wife_relation",
                    "kind": "fact",
                    "summary": "Janina is the user's wife.",
                    "details": "Directly stated by the user.",
                    "predicate": "relationship_spouse",
                    "confidence": 0.98,
                    "sensitivity": "private",
                    "source_channel": "user_transcript",
                    "subject_ref": "user:main",
                    "object_ref": "person:janina",
                    "value_text": None,
                    "valid_from": None,
                    "valid_to": None,
                    "attributes": [
                        {"key": "relation", "value": "wife"},
                        {"key": "person_name", "value": "Janina"},
                        {"key": "fact_type", "value": "relationship"},
                    ],
                },
                {
                    "proposition_id": "warm_today",
                    "kind": "observation",
                    "summary": "The user described the day as warm.",
                    "details": "Directly stated by the user.",
                    "predicate": "weather_condition",
                    "confidence": 0.72,
                    "sensitivity": "low",
                    "source_channel": "user_transcript",
                    "subject_ref": "day:2026-03-15",
                    "object_ref": None,
                    "value_text": "warm",
                    "valid_from": "2026-03-15",
                    "valid_to": "2026-03-15",
                    "attributes": [
                        {"key": "memory_domain", "value": "situational"},
                        {"key": "observation_type", "value": "situational"},
                    ],
                },
                {
                    "proposition_id": "laser_event",
                    "kind": "event",
                    "summary": "Janina is getting eye laser treatment on 2026-03-15.",
                    "details": "Directly stated by the user.",
                    "predicate": "eye_laser_treatment",
                    "confidence": 0.93,
                    "sensitivity": "sensitive",
                    "source_channel": "user_transcript",
                    "subject_ref": "person:janina",
                    "object_ref": "event:janina_eye_laser_2026_03_15",
                    "value_text": None,
                    "valid_from": "2026-03-15",
                    "valid_to": "2026-03-15",
                    "attributes": [
                        {"key": "memory_domain", "value": "appointment"},
                        {"key": "event_domain", "value": "appointment"},
                        {"key": "action", "value": "eye laser treatment"},
                    ],
                },
            ],
            "graph_edges": [
                {
                    "source_ref": "user:main",
                    "edge_type": "social_related_to_user",
                    "target_ref": "person:janina",
                    "confidence": 0.98,
                    "confirmed_by_user": True,
                    "valid_from": None,
                    "valid_to": None,
                    "attributes": [
                        {"key": "relation", "value": "wife"},
                    ],
                },
                {
                    "source_ref": "event:janina_eye_laser_2026_03_15",
                    "edge_type": "temporal_occurs_on",
                    "target_ref": "day:2026-03-15",
                    "confidence": 0.9,
                    "confirmed_by_user": True,
                    "valid_from": "2026-03-15",
                    "valid_to": "2026-03-15",
                    "attributes": [],
                },
            ],
        }
        bundle = LongTermTurnPropositionBundleV1.from_payload(payload)
        compiler = LongTermTurnPropositionCompiler()
        objects, edges = compiler.compile(
            bundle=bundle,
            source_ref=LongTermSourceRefV1(
                source_type="conversation_turn",
                event_ids=("turn:test",),
                speaker="user",
                modality="voice",
            ),
        )

        self.assertEqual(len(objects), 3)
        self.assertEqual(len(edges), 2)
        self.assertEqual(objects[0].slot_key, "fact:user:main:relationship_spouse")
        self.assertEqual(objects[0].value_key, "person:janina")
        self.assertEqual(objects[1].slot_key, "observation:day:2026-03-15:weather_condition:2026-03-15")
        self.assertEqual(objects[1].value_key, "warm")
        self.assertEqual(objects[2].slot_key, "event:person:janina:eye_laser_treatment:2026-03-15")
        self.assertEqual(objects[2].value_key, "event:janina_eye_laser_2026_03_15")
        self.assertEqual(objects[2].valid_from, occurred_at.date().isoformat())

    def test_compiler_skips_assistant_only_propositions(self) -> None:
        bundle = LongTermTurnPropositionBundleV1.from_payload(
            {
                "propositions": [
                    {
                        "proposition_id": "assistant_only",
                        "kind": "event",
                        "summary": "Janina has an appointment today.",
                        "details": "Only stated by the assistant.",
                        "predicate": "has_appointment",
                        "confidence": 0.7,
                        "sensitivity": "normal",
                        "source_channel": "assistant_response",
                        "subject_ref": "person:janina",
                        "object_ref": None,
                        "value_text": None,
                        "valid_from": "2026-03-15",
                        "valid_to": "2026-03-15",
                        "attributes": [],
                    }
                ],
                "graph_edges": [
                    {
                        "source_ref": "person:janina",
                        "edge_type": "temporal_occurs_on",
                        "target_ref": "day:2026-03-15",
                        "confidence": 0.7,
                        "confirmed_by_user": False,
                        "valid_from": "2026-03-15",
                        "valid_to": "2026-03-15",
                        "attributes": [
                            {"key": "proposition_id", "value": "assistant_only"},
                        ],
                    }
                ],
            }
        )
        compiler = LongTermTurnPropositionCompiler()
        objects, edges = compiler.compile(
            bundle=bundle,
            source_ref=LongTermSourceRefV1(source_type="conversation_turn"),
        )

        self.assertEqual(objects, ())
        self.assertEqual(edges, ())

    def test_compiler_enriches_style_and_humor_semantics_from_canonical_predicates(self) -> None:
        bundle = LongTermTurnPropositionBundleV1.from_payload(
            {
                "propositions": [
                    {
                        "proposition_id": "pref_verbosity",
                        "kind": "fact",
                        "summary": "The user says shorter and calmer answers usually help more.",
                        "details": None,
                        "predicate": "user_prefers_answer_style",
                        "confidence": 0.88,
                        "sensitivity": "normal",
                        "source_channel": "user_transcript",
                        "subject_ref": "user:main",
                        "object_ref": None,
                        "value_text": "Shorter and calmer answers",
                        "valid_from": None,
                        "valid_to": None,
                        "attributes": [],
                    },
                    {
                        "proposition_id": "pref_initiative",
                        "kind": "fact",
                        "summary": "The user says one small follow-up can help when it is really useful.",
                        "details": None,
                        "predicate": "user_prefers_small_follow_up_when_helpful",
                        "confidence": 0.82,
                        "sensitivity": "normal",
                        "source_channel": "user_transcript",
                        "subject_ref": "user:main",
                        "object_ref": None,
                        "value_text": "One small follow-up question when it really helps",
                        "valid_from": None,
                        "valid_to": None,
                        "attributes": [],
                    },
                    {
                        "proposition_id": "feedback_humor",
                        "kind": "observation",
                        "summary": "A small dry joke worked for the user.",
                        "details": None,
                        "predicate": "responds_well_to_dry_humor",
                        "confidence": 0.79,
                        "sensitivity": "normal",
                        "source_channel": "user_transcript",
                        "subject_ref": "user:main",
                        "object_ref": "concept:dry_humor",
                        "value_text": None,
                        "valid_from": None,
                        "valid_to": None,
                        "attributes": [],
                    },
                ],
                "graph_edges": [],
            }
        )

        objects, _edges = LongTermTurnPropositionCompiler().compile(
            bundle=bundle,
            source_ref=LongTermSourceRefV1(source_type="conversation_turn"),
        )
        object_by_predicate = {
            object_.attributes["predicate"]: object_
            for object_ in objects
            if object_.attributes is not None and "predicate" in object_.attributes
        }

        verbosity = object_by_predicate["user_prefers_answer_style"]
        self.assertEqual(verbosity.attributes["memory_domain"], "preference")
        self.assertEqual(verbosity.attributes["preference_type"], "verbosity")
        self.assertEqual(verbosity.attributes["preference_value"], "concise")

        initiative = object_by_predicate["user_prefers_small_follow_up_when_helpful"]
        self.assertEqual(initiative.attributes["memory_domain"], "preference")
        self.assertEqual(initiative.attributes["preference_type"], "initiative")
        self.assertEqual(initiative.attributes["preference_value"], "gently_proactive")

        humor = object_by_predicate["responds_well_to_dry_humor"]
        self.assertEqual(humor.attributes["memory_domain"], "preference")
        self.assertEqual(humor.attributes["feedback_target"], "humor")
        self.assertEqual(humor.attributes["feedback_polarity"], "positive")

    def test_turn_program_forces_canonical_english_backend_language(self) -> None:
        config = TwinrConfig(
            openai_api_key="sk-test",
            openai_realtime_language="de",
            long_term_memory_turn_extractor_model="gpt-5.2-mini",
            long_term_memory_turn_extractor_max_output_tokens=2600,
        )

        program = structured_turn_program_from_config(config)

        self.assertIsNotNone(program)
        assert program is not None
        self.assertEqual(program.backend.config.openai_realtime_language, "en")
        self.assertEqual(program.model, "gpt-5.2-mini")
        self.assertEqual(program.max_output_tokens, 2600)


if __name__ == "__main__":
    unittest.main()
