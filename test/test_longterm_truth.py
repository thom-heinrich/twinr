from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.memory.longterm.core.models import LongTermMemoryObjectV1, LongTermSourceRefV1
from twinr.memory.longterm.reasoning.truth import LongTermTruthMaintainer


def _source() -> LongTermSourceRefV1:
    return LongTermSourceRefV1(
        source_type="conversation_turn",
        event_ids=("turn:test",),
        speaker="user",
        modality="voice",
    )


class LongTermTruthMaintainerTests(unittest.TestCase):
    def test_detect_conflicts_when_same_slot_has_different_value(self) -> None:
        maintainer = LongTermTruthMaintainer()
        existing = LongTermMemoryObjectV1(
            memory_id="fact:corinna_phone_old",
            kind="contact_method_fact",
            summary="Corinna Maier can be reached at +491761234.",
            source=_source(),
            status="active",
            confidence=0.95,
            slot_key="contact:person:corinna_maier:phone",
            value_key="+491761234",
            attributes={"person_ref": "person:corinna_maier"},
        )
        candidate = LongTermMemoryObjectV1(
            memory_id="fact:corinna_phone_new",
            kind="contact_method_fact",
            summary="Corinna Maier can be reached at +4940998877.",
            source=_source(),
            status="candidate",
            confidence=0.92,
            slot_key="contact:person:corinna_maier:phone",
            value_key="+4940998877",
            attributes={"person_ref": "person:corinna_maier"},
        )

        conflicts = maintainer.detect_conflicts(existing_objects=(existing,), candidate=candidate)

        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].slot_key, "contact:person:corinna_maier:phone")
        self.assertIn("contact detail", conflicts[0].question.lower())

    def test_activate_candidate_marks_clean_candidate_active(self) -> None:
        maintainer = LongTermTruthMaintainer()
        candidate = LongTermMemoryObjectV1(
            memory_id="fact:janina_spouse",
            kind="relationship_fact",
            summary="Janina is the user's wife.",
            source=_source(),
            status="candidate",
            confidence=0.98,
            slot_key="relationship:user:main:wife",
            value_key="person:janina",
            attributes={"person_ref": "person:janina"},
        )

        activated = maintainer.activate_candidate(existing_objects=(), candidate=candidate)

        self.assertEqual(activated.status, "active")
        self.assertEqual(activated.value_key, "person:janina")


if __name__ == "__main__":
    unittest.main()
