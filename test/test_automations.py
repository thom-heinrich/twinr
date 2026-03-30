from datetime import datetime, timedelta
from pathlib import Path
import sys
import tempfile
import unittest
from zoneinfo import ZoneInfo

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

    def test_weekly_due_time_matches_respect_misfire_grace_and_coalesce_policy(self) -> None:
        zone = ZoneInfo("Europe/Berlin")
        now = datetime(2026, 3, 25, 10, 30, tzinfo=zone)
        previous_run_at = datetime(2026, 3, 2, 8, 0, tzinfo=zone)
        previous_triggered_at = datetime(2026, 3, 2, 8, 1, tzinfo=zone)
        expected_backlog = (
            datetime(2026, 3, 23, 8, 0, tzinfo=zone),
            datetime(2026, 3, 25, 8, 0, tzinfo=zone),
        )
        grace_seconds = 3 * 24 * 60 * 60

        with tempfile.TemporaryDirectory() as temp_dir:
            store = AutomationStore(Path(temp_dir) / "automations.json", timezone_name="Europe/Berlin")
            entries_by_policy = {}
            for policy in ("latest", "earliest", "all"):
                entry = store.create_time_automation(
                    name=f"Weekly {policy}",
                    schedule="weekly",
                    time_of_day="08:00",
                    weekdays=(0, 2),
                    misfire_grace_seconds=grace_seconds,
                    coalesce_policy=policy,
                    actions=(AutomationAction(kind="say", text=f"Weekly backlog {policy}."),),
                )
                store.mark_triggered(
                    entry.automation_id,
                    triggered_at=previous_triggered_at,
                    scheduled_for_at=previous_run_at,
                )
                reloaded = store.get(entry.automation_id)
                assert reloaded is not None
                entries_by_policy[policy] = reloaded

            matches = store.due_time_matches(now=now)
            matches_by_automation_id: dict[str, list[object]] = {}
            for match in matches:
                matches_by_automation_id.setdefault(match.entry.automation_id, []).append(match)

            latest_matches = matches_by_automation_id[entries_by_policy["latest"].automation_id]
            earliest_matches = matches_by_automation_id[entries_by_policy["earliest"].automation_id]
            all_matches = matches_by_automation_id[entries_by_policy["all"].automation_id]
            tool_records = {
                record["automation_id"]: record
                for record in store.list_tool_records(now=now)
            }

        self.assertEqual(len(latest_matches), 1)
        self.assertEqual(latest_matches[0].scheduled_for_at, expected_backlog[-1])
        self.assertEqual(latest_matches[0].pending_run_count, 2)

        self.assertEqual(len(earliest_matches), 1)
        self.assertEqual(earliest_matches[0].scheduled_for_at, expected_backlog[0])
        self.assertEqual(earliest_matches[0].pending_run_count, 2)

        self.assertEqual(
            [match.scheduled_for_at for match in all_matches],
            list(expected_backlog),
        )
        self.assertEqual([match.pending_run_count for match in all_matches], [1, 1])

        latest_record = tool_records[entries_by_policy["latest"].automation_id]
        earliest_record = tool_records[entries_by_policy["earliest"].automation_id]
        all_record = tool_records[entries_by_policy["all"].automation_id]
        self.assertEqual(latest_record["scheduled_for_at"], expected_backlog[-1].isoformat())
        self.assertEqual(earliest_record["scheduled_for_at"], expected_backlog[0].isoformat())
        self.assertEqual(all_record["scheduled_for_at"], expected_backlog[0].isoformat())
        self.assertEqual(latest_record["pending_run_count"], 2)
        self.assertEqual(earliest_record["pending_run_count"], 2)
        self.assertEqual(all_record["pending_run_count"], 2)
        self.assertEqual(latest_record["coalesce_policy"], "latest")
        self.assertEqual(earliest_record["coalesce_policy"], "earliest")
        self.assertEqual(all_record["coalesce_policy"], "all")
        self.assertEqual(latest_record["misfire_grace_seconds"], float(grace_seconds))

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

    def test_store_files_use_owner_only_runtime_modes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = AutomationStore(Path(temp_dir) / "automations.json", timezone_name="Europe/Berlin")
            due_at = (now_in_timezone("Europe/Berlin") + timedelta(hours=1)).isoformat()
            store.create_time_automation(
                name="Shared state check",
                schedule="once",
                due_at=due_at,
                actions=(AutomationAction(kind="say", text="Hallo."),),
            )
            store.load_entries()

            store_mode = store.path.stat().st_mode & 0o777
            backup_mode = store.backup_path.stat().st_mode & 0o777
            lock_mode = store.lock_path.stat().st_mode & 0o777

        self.assertEqual(store_mode, 0o600)
        self.assertEqual(backup_mode, 0o600)
        self.assertEqual(lock_mode, 0o600)

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
