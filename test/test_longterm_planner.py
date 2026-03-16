from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
import unittest
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.memory.longterm.core.models import LongTermMemoryObjectV1, LongTermSourceRefV1
from twinr.memory.longterm.proactive.planner import LongTermProactivePlanner


def _source() -> LongTermSourceRefV1:
    return LongTermSourceRefV1(
        source_type="conversation_turn",
        event_ids=("turn:test",),
        speaker="user",
        modality="voice",
    )


def _sensor_live_facts(
    *,
    person_visible: bool = False,
    looking_toward_device: bool = False,
    hand_or_object_near_camera: bool = False,
    body_pose: str = "upright",
    speech_detected: bool = False,
    quiet: bool = True,
    last_response_available: bool = False,
    recent_print_completed: bool = False,
) -> dict[str, object]:
    return {
        "camera": {
            "person_visible": person_visible,
            "looking_toward_device": looking_toward_device,
            "hand_or_object_near_camera": hand_or_object_near_camera,
            "body_pose": body_pose,
        },
        "vad": {
            "speech_detected": speech_detected,
            "quiet": quiet,
        },
        "last_response_available": last_response_available,
        "recent_print_completed": recent_print_completed,
    }


class LongTermProactivePlannerTests(unittest.TestCase):
    def test_planner_emits_same_day_reminder_for_active_same_day_event(self) -> None:
        planner = LongTermProactivePlanner(timezone_name="Europe/Berlin")
        event = LongTermMemoryObjectV1(
            memory_id="event:janina_eye_laser",
            kind="event",
            summary="Janina has eye laser treatment at the eye doctor on 2026-03-14.",
            source=_source(),
            status="active",
            confidence=0.92,
            valid_from="2026-03-14",
            valid_to="2026-03-14",
            sensitivity="sensitive",
            attributes={
                "person_ref": "person:janina",
                "person_name": "Janina",
                "memory_domain": "appointment",
                "event_domain": "appointment",
                "action": "eye laser treatment",
                "place": "eye doctor",
            },
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
            kind="summary",
            summary="Ongoing thread about Janina: Janina is the user's wife; eye laser treatment at the eye doctor.",
            source=_source(),
            status="active",
            confidence=0.76,
            sensitivity="sensitive",
            attributes={
                "person_ref": "person:janina",
                "person_name": "Janina",
                "support_count": 3,
                "summary_type": "thread",
            },
        )

        plan = planner.plan(
            objects=(summary,),
            now=datetime(2026, 3, 14, 15, 0, tzinfo=ZoneInfo("Europe/Berlin")),
        )

        self.assertEqual(len(plan.candidates), 1)
        self.assertEqual(plan.candidates[0].kind, "gentle_follow_up")
        self.assertIn("Ongoing thread about Janina", plan.candidates[0].summary)

    def test_planner_emits_routine_check_in_only_with_live_confirmation(self) -> None:
        planner = LongTermProactivePlanner(timezone_name="Europe/Berlin")
        deviation = LongTermMemoryObjectV1(
            memory_id="deviation:presence:weekday:morning:2026-03-18",
            kind="summary",
            summary="Presence seems unusually low in the morning for weekdays.",
            source=_source(),
            status="candidate",
            confidence=0.81,
            sensitivity="low",
            valid_from="2026-03-18",
            valid_to="2026-03-18",
            attributes={
                "memory_domain": "sensor_routine",
                "summary_type": "sensor_deviation",
                "deviation_type": "missing_presence",
                "weekday_class": "weekday",
                "daypart": "morning",
                "date": "2026-03-18",
                "requires_live_confirmation": True,
            },
        )

        without_live = planner.plan(
            objects=(deviation,),
            now=datetime(2026, 3, 18, 9, 0, tzinfo=ZoneInfo("Europe/Berlin")),
        )
        with_live = planner.plan(
            objects=(deviation,),
            now=datetime(2026, 3, 18, 9, 0, tzinfo=ZoneInfo("Europe/Berlin")),
            live_facts=_sensor_live_facts(person_visible=True, quiet=True),
        )

        self.assertEqual(without_live.candidates, ())
        self.assertEqual(len(with_live.candidates), 1)
        self.assertEqual(with_live.candidates[0].kind, "routine_check_in")

    def test_planner_emits_routine_camera_offer_when_camera_pattern_and_live_showing_match(self) -> None:
        planner = LongTermProactivePlanner(timezone_name="Europe/Berlin")
        routine = LongTermMemoryObjectV1(
            memory_id="routine:interaction:camera_showing:weekday:morning",
            kind="pattern",
            summary="Camera showing is typical in the morning on weekdays.",
            source=_source(),
            status="active",
            confidence=0.8,
            sensitivity="low",
            valid_from="2026-03-03",
            valid_to="2026-03-17",
            attributes={
                "memory_domain": "sensor_routine",
                "routine_type": "interaction",
                "interaction_type": "camera_showing",
                "weekday_class": "weekday",
                "daypart": "morning",
            },
        )

        plan = planner.plan(
            objects=(routine,),
            now=datetime(2026, 3, 18, 9, 0, tzinfo=ZoneInfo("Europe/Berlin")),
            live_facts=_sensor_live_facts(
                person_visible=True,
                hand_or_object_near_camera=True,
                quiet=True,
            ),
        )

        self.assertEqual(len(plan.candidates), 1)
        self.assertEqual(plan.candidates[0].kind, "routine_camera_offer")

    def test_planner_emits_routine_print_offer_only_if_answer_available_and_not_recently_printed(self) -> None:
        planner = LongTermProactivePlanner(timezone_name="Europe/Berlin")
        routine = LongTermMemoryObjectV1(
            memory_id="routine:interaction:print:weekday:morning",
            kind="pattern",
            summary="Printing is typical in the morning on weekdays.",
            source=_source(),
            status="active",
            confidence=0.82,
            sensitivity="low",
            valid_from="2026-03-03",
            valid_to="2026-03-17",
            attributes={
                "memory_domain": "sensor_routine",
                "routine_type": "interaction",
                "interaction_type": "print",
                "weekday_class": "weekday",
                "daypart": "morning",
            },
        )

        blocked = planner.plan(
            objects=(routine,),
            now=datetime(2026, 3, 18, 9, 0, tzinfo=ZoneInfo("Europe/Berlin")),
            live_facts=_sensor_live_facts(
                person_visible=True,
                looking_toward_device=True,
                last_response_available=True,
                recent_print_completed=True,
            ),
        )
        allowed = planner.plan(
            objects=(routine,),
            now=datetime(2026, 3, 18, 9, 0, tzinfo=ZoneInfo("Europe/Berlin")),
            live_facts=_sensor_live_facts(
                person_visible=True,
                looking_toward_device=True,
                last_response_available=True,
                recent_print_completed=False,
            ),
        )

        self.assertEqual(blocked.candidates, ())
        self.assertEqual(len(allowed.candidates), 1)
        self.assertEqual(allowed.candidates[0].kind, "routine_print_offer")


if __name__ == "__main__":
    unittest.main()
