from datetime import timedelta
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.memory.reminders import ReminderStore, now_in_timezone


class ReminderStoreTests(unittest.TestCase):
    def test_schedule_persists_and_deduplicates_pending_reminder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "reminders.json"
            store = ReminderStore(path, timezone_name="Europe/Berlin")
            due_at = (now_in_timezone("Europe/Berlin") + timedelta(hours=2)).isoformat()

            first = store.schedule(
                due_at=due_at,
                summary="Arzttermin",
                details="Bei Dr. Meyer",
                kind="appointment",
                original_request="Erinnere mich bitte spaeter an den Arzttermin.",
            )
            second = store.schedule(
                due_at=due_at,
                summary="Arzttermin",
                details="Bei Dr. Meyer in Hamburg",
                kind="appointment",
            )

            entries = store.load_entries()

        self.assertEqual(first.reminder_id, second.reminder_id)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].details, "Bei Dr. Meyer in Hamburg")
        self.assertEqual(entries[0].kind, "appointment")

    def test_reserve_mark_failed_and_mark_delivered_update_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "reminders.json"
            store = ReminderStore(path, timezone_name="Europe/Berlin", retry_delay_s=30.0)
            due_at = (now_in_timezone("Europe/Berlin") + timedelta(seconds=1)).isoformat()
            created = store.schedule(due_at=due_at, summary="Medikament nehmen")

            reserved = store.reserve_due(now=created.due_at + timedelta(seconds=1))
            failed = store.mark_failed(created.reminder_id, error="speaker offline", failed_at=created.due_at + timedelta(seconds=2))
            delivered = store.mark_delivered(
                created.reminder_id,
                delivered_at=created.due_at + timedelta(seconds=40),
            )

        self.assertEqual(len(reserved), 1)
        self.assertEqual(reserved[0].delivery_attempts, 1)
        self.assertIsNotNone(reserved[0].next_attempt_at)
        self.assertEqual(failed.last_error, "speaker offline")
        self.assertIsNotNone(delivered.delivered_at)
        self.assertTrue(delivered.delivered)

    def test_render_context_lists_pending_reminders(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "reminders.json"
            store = ReminderStore(path, timezone_name="Europe/Berlin")
            due_at = (now_in_timezone("Europe/Berlin") + timedelta(hours=1)).isoformat()
            store.schedule(due_at=due_at, summary="Buskarte mitnehmen", kind="task")

            context = store.render_context()

        self.assertIsNotNone(context)
        self.assertIn("Scheduled reminders and timers:", context)
        self.assertIn("Buskarte mitnehmen", context)


if __name__ == "__main__":
    unittest.main()
