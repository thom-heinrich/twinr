from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
import unittest
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.memory.longterm.models import LongTermMemoryObjectV1, LongTermSourceRefV1
from twinr.memory.longterm.planner import LongTermProactivePlanner


def _source() -> LongTermSourceRefV1:
    return LongTermSourceRefV1(
        source_type="conversation_turn",
        event_ids=("turn:test",),
        speaker="user",
        modality="voice",
    )


class LongTermProactivePlannerTests(unittest.TestCase):
    def test_planner_emits_same_day_reminder_for_active_medical_event(self) -> None:
        planner = LongTermProactivePlanner(timezone_name="Europe/Berlin")
        event = LongTermMemoryObjectV1(
            memory_id="fact:janina_eye_laser",
            kind="medical_event",
            summary="Janina has eye laser treatment at the eye doctor on 2026-03-14.",
            source=_source(),
            status="active",
            confidence=0.92,
            valid_from="2026-03-14",
            valid_to="2026-03-14",
            sensitivity="medical",
            attributes={"person_ref": "person:janina", "person_name": "Janina"},
        )

        plan = planner.plan(
            objects=(event,),
            now=datetime(2026, 3, 14, 9, 0, tzinfo=ZoneInfo("Europe/Berlin")),
        )

        self.assertEqual(len(plan.candidates), 1)
        self.assertEqual(plan.candidates[0].kind, "same_day_reminder")
        self.assertIn("Janina has eye laser treatment", plan.candidates[0].summary)

    def test_planner_emits_gentle_follow_up_for_thread_summary(self) -> None:
        planner = LongTermProactivePlanner(timezone_name="Europe/Berlin")
        summary = LongTermMemoryObjectV1(
            memory_id="thread:person_janina",
            kind="thread_summary",
            summary="Ongoing thread about Janina: Janina is the user's wife; eye laser treatment at the eye doctor.",
            source=_source(),
            status="active",
            confidence=0.76,
            sensitivity="medical",
            attributes={
                "person_ref": "person:janina",
                "person_name": "Janina",
                "support_count": 3,
            },
        )

        plan = planner.plan(
            objects=(summary,),
            now=datetime(2026, 3, 14, 15, 0, tzinfo=ZoneInfo("Europe/Berlin")),
        )

        self.assertEqual(len(plan.candidates), 1)
        self.assertEqual(plan.candidates[0].kind, "gentle_follow_up")
        self.assertIn("Ongoing thread about Janina", plan.candidates[0].summary)


if __name__ == "__main__":
    unittest.main()
