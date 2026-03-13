from datetime import timedelta
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.automations import (
    AutomationAction,
    AutomationCondition,
    AutomationStore,
    IfThenAutomationTrigger,
    TimeAutomationTrigger,
    build_sensor_trigger,
    describe_sensor_trigger,
)
from twinr.memory.reminders import now_in_timezone


class AutomationStoreTests(unittest.TestCase):
    def test_create_time_automation_persists_and_lists_tool_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = AutomationStore(Path(temp_dir) / "automations.json", timezone_name="Europe/Berlin")
            due_at = (now_in_timezone("Europe/Berlin") + timedelta(hours=2)).isoformat()

            created = store.create_time_automation(
                name="Morning briefing",
                description="Speak a short morning reminder.",
                schedule="once",
                due_at=due_at,
                actions=(
                    AutomationAction(kind="say", text="Guten Morgen. Heute steht der Arzttermin an."),
                ),
                tags=("morning", "care"),
            )

            loaded = store.get(created.automation_id)
            tool_records = store.list_tool_records(now=now_in_timezone("Europe/Berlin"))

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.name, "Morning briefing")
        self.assertIsInstance(loaded.trigger, TimeAutomationTrigger)
        self.assertEqual(len(tool_records), 1)
        self.assertEqual(tool_records[0]["trigger_kind"], "time")
        self.assertFalse(tool_records[0]["due_now"])
        self.assertIsNotNone(tool_records[0]["next_run_at"])

    def test_daily_time_automation_becomes_due_once_per_occurrence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = AutomationStore(Path(temp_dir) / "automations.json", timezone_name="Europe/Berlin")
            now = now_in_timezone("Europe/Berlin")
            run_time = (now - timedelta(minutes=5)).strftime("%H:%M")
            created = store.create_time_automation(
                name="Daily medicine prompt",
                schedule="daily",
                time_of_day=run_time,
                actions=(AutomationAction(kind="say", text="Bitte jetzt an die Tabletten denken."),),
            )

            due_before_mark = store.due_time_automations(now=now)
            store.mark_triggered(created.automation_id, triggered_at=now)
            due_after_mark = store.due_time_automations(now=now + timedelta(minutes=1))

        self.assertEqual([entry.automation_id for entry in due_before_mark], [created.automation_id])
        self.assertEqual(due_after_mark, ())

    def test_if_then_automation_matches_facts_and_respects_cooldown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = AutomationStore(Path(temp_dir) / "automations.json", timezone_name="Europe/Berlin")
            now = now_in_timezone("Europe/Berlin")
            created = store.create_if_then_automation(
                name="Print after request",
                event_name="conversation_completed",
                all_conditions=(
                    AutomationCondition(key="conversation.print_requested", operator="truthy"),
                    AutomationCondition(key="conversation.status", operator="eq", value="ready"),
                ),
                cooldown_seconds=600,
                actions=(
                    AutomationAction(kind="tool_call", tool_name="print_receipt", payload={"mode": "summary"}),
                ),
            )

            matches_first = store.matching_if_then_automations(
                facts={"conversation": {"print_requested": True, "status": "ready"}},
                event_name="conversation_completed",
                now=now,
            )
            store.mark_triggered(created.automation_id, triggered_at=now)
            matches_second = store.matching_if_then_automations(
                facts={"conversation": {"print_requested": True, "status": "ready"}},
                event_name="conversation_completed",
                now=now + timedelta(minutes=5),
            )

        self.assertEqual([entry.automation_id for entry in matches_first], [created.automation_id])
        self.assertEqual(matches_second, ())

    def test_update_and_delete_automation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = AutomationStore(Path(temp_dir) / "automations.json", timezone_name="Europe/Berlin")
            created = store.create_if_then_automation(
                name="Follow-up check",
                event_name="reminder_delivered",
                all_conditions=(AutomationCondition(key="reminder.kind", operator="eq", value="appointment"),),
                actions=(AutomationAction(kind="llm_prompt", text="Ask whether the user needs the appointment printed."),),
            )

            updated = store.update(
                created.automation_id,
                name="Appointment follow-up",
                enabled=False,
                trigger=IfThenAutomationTrigger(
                    event_name="reminder_delivered",
                    all_conditions=(AutomationCondition(key="reminder.kind", operator="eq", value="appointment"),),
                    any_conditions=(AutomationCondition(key="reminder.summary", operator="contains", value="Arzt"),),
                    cooldown_seconds=120.0,
                ),
            )
            removed = store.delete(created.automation_id)

        self.assertEqual(updated.name, "Appointment follow-up")
        self.assertFalse(updated.enabled)
        self.assertEqual(removed.automation_id, created.automation_id)
        self.assertIsNone(store.get(created.automation_id))

    def test_render_context_lists_active_automations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = AutomationStore(Path(temp_dir) / "automations.json", timezone_name="Europe/Berlin")
            store.create_time_automation(
                name="Daily weather",
                schedule="daily",
                time_of_day="08:00",
                actions=(
                    AutomationAction(
                        kind="llm_prompt",
                        text="Give the daily weather report.",
                        payload={"delivery": "spoken", "allow_web_search": True},
                    ),
                ),
            )

            context = store.render_context()

        self.assertIsNotNone(context)
        assert context is not None
        self.assertIn("Active automations:", context)
        self.assertIn("Daily weather", context)
        self.assertIn("spoken llm prompt with web search", context)

    def test_sensor_trigger_helpers_build_and_describe_supported_kinds(self) -> None:
        immediate = build_sensor_trigger("camera_person_visible", cooldown_seconds=90)
        delayed = build_sensor_trigger("vad_quiet", hold_seconds=30, cooldown_seconds=120)

        immediate_spec = describe_sensor_trigger(immediate)
        delayed_spec = describe_sensor_trigger(delayed)

        self.assertIsNotNone(immediate_spec)
        self.assertEqual(immediate.event_name, "camera.person_visible")
        self.assertEqual(immediate_spec.trigger_kind, "camera_person_visible")
        self.assertEqual(immediate_spec.hold_seconds, 0.0)
        self.assertIsNotNone(delayed_spec)
        self.assertIsNone(delayed.event_name)
        self.assertEqual(delayed_spec.trigger_kind, "vad_quiet")
        self.assertEqual(delayed_spec.hold_seconds, 30.0)

    def test_if_then_tool_records_expose_sensor_trigger_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = AutomationStore(Path(temp_dir) / "automations.json", timezone_name="Europe/Berlin")
            created = store.create_if_then_automation(
                name="Watch for visitors",
                actions=(AutomationAction(kind="say", text="Jemand ist da."),),
                event_name=build_sensor_trigger("camera_person_visible").event_name,
                all_conditions=build_sensor_trigger("camera_person_visible").all_conditions,
                cooldown_seconds=45,
                tags=("sensor", "camera_person_visible"),
            )

            record = store.list_tool_records()[0]

        self.assertEqual(created.automation_id, record["automation_id"])
        self.assertEqual(record["trigger_kind"], "if_then")
        self.assertEqual(record["sensor_trigger_kind"], "camera_person_visible")
        self.assertEqual(record["sensor_hold_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
