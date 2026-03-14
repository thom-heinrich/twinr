from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.memory.longterm.models import LongTermMemoryObjectV1, LongTermSourceRefV1
from twinr.memory.longterm.reflect import LongTermMemoryReflector


def _source() -> LongTermSourceRefV1:
    return LongTermSourceRefV1(
        source_type="conversation_turn",
        event_ids=("turn:test",),
        speaker="user",
        modality="voice",
    )


class LongTermMemoryReflectorTests(unittest.TestCase):
    def test_reflector_can_promote_repeated_uncertain_fact(self) -> None:
        reflector = LongTermMemoryReflector()
        item = LongTermMemoryObjectV1(
            memory_id="fact:corinna_role",
            kind="relationship_fact",
            summary="Corinna Maier is the user's physiotherapist.",
            source=_source(),
            status="uncertain",
            confidence=0.62,
            slot_key="relationship:user:main:physiotherapist",
            value_key="person:corinna_maier",
            attributes={
                "person_ref": "person:corinna_maier",
                "person_name": "Corinna Maier",
                "relation": "physiotherapist",
                "support_count": 2,
            },
        )

        result = reflector.reflect(objects=(item,))

        self.assertEqual(len(result.reflected_objects), 1)
        self.assertEqual(result.reflected_objects[0].status, "active")
        self.assertGreaterEqual(result.reflected_objects[0].confidence, 0.79)

    def test_reflector_can_create_thread_summary_for_person(self) -> None:
        reflector = LongTermMemoryReflector()
        relationship = LongTermMemoryObjectV1(
            memory_id="fact:janina_wife",
            kind="relationship_fact",
            summary="Janina is the user's wife.",
            source=_source(),
            status="active",
            confidence=0.98,
            slot_key="relationship:user:main:wife",
            value_key="person:janina",
            attributes={
                "person_ref": "person:janina",
                "person_name": "Janina",
                "relation": "wife",
                "support_count": 2,
            },
        )
        medical_event = LongTermMemoryObjectV1(
            memory_id="fact:janina_eye_laser",
            kind="medical_event",
            summary="Janina has eye laser treatment at the eye doctor on 2026-03-14.",
            source=_source(),
            status="active",
            confidence=0.92,
            slot_key="event:person:janina:eye_laser_treatment:2026-03-14",
            value_key="event:janina_eye_laser_2026_03_14",
            valid_from="2026-03-14",
            valid_to="2026-03-14",
            sensitivity="medical",
            attributes={
                "person_ref": "person:janina",
                "person_name": "Janina",
                "treatment": "eye laser treatment",
                "place": "the eye doctor",
                "support_count": 1,
            },
        )

        result = reflector.reflect(objects=(relationship, medical_event))

        self.assertEqual(len(result.created_summaries), 1)
        summary = result.created_summaries[0]
        self.assertEqual(summary.kind, "thread_summary")
        self.assertIn("Ongoing thread about Janina", summary.summary)
        self.assertIn("eye laser treatment", summary.summary)
        self.assertEqual(summary.sensitivity, "medical")


if __name__ == "__main__":
    unittest.main()
