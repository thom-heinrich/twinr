from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.agent.personality.display_impulses import AmbientDisplayImpulseCandidate
from twinr.proactive.runtime.display_reserve_day_plan import (
    DisplayReserveDayPlanStore,
    DisplayReserveDayPlanner,
)


def _candidate(
    *,
    topic_key: str,
    action: str,
    attention_state: str,
    salience: float,
) -> AmbientDisplayImpulseCandidate:
    return AmbientDisplayImpulseCandidate(
        topic_key=topic_key,
        title=topic_key.title(),
        source="world",
        action=action,
        attention_state=attention_state,
        salience=salience,
        eyebrow="IM BLICK",
        headline=topic_key.title(),
        body="Da tut sich gerade etwas.",
        symbol="sparkles",
        accent="info",
        reason=f"seed:{topic_key}",
    )


class DisplayReserveDayPlanTests(unittest.TestCase):
    def test_planner_builds_and_persists_one_local_day_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                display_reserve_bus_items_per_day=6,
                display_reserve_bus_candidate_limit=3,
                display_reserve_bus_topic_gap=1,
            )
            planner = DisplayReserveDayPlanner.from_config(config)
            planner.candidate_loader = lambda _config, *, local_now, max_items: (
                _candidate(topic_key="ai companions", action="invite_follow_up", attention_state="shared_thread", salience=0.9),
                _candidate(topic_key="world politics", action="brief_update", attention_state="growing", salience=0.7),
            )[:max_items]

            now = datetime(2026, 3, 22, 10, 0, tzinfo=timezone.utc)
            plan = planner.ensure_plan(config=config, local_now=now)
            loaded = DisplayReserveDayPlanStore.from_config(config).load()

        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(plan.local_day, "2026-03-22")
        self.assertEqual(len(plan.items), 6)
        self.assertEqual(loaded.cursor, 0)
        self.assertEqual(loaded.current_item().topic_key, plan.current_item().topic_key)

    def test_planner_advances_cursor_without_rebuilding_same_day_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir, display_reserve_bus_items_per_day=4)
            planner = DisplayReserveDayPlanner.from_config(config)
            planner.candidate_loader = lambda _config, *, local_now, max_items: (
                _candidate(topic_key="ai companions", action="invite_follow_up", attention_state="shared_thread", salience=0.9),
                _candidate(topic_key="hamburg local politics", action="ask_one", attention_state="forming", salience=0.8),
            )[:max_items]

            now = datetime(2026, 3, 22, 11, 0, tzinfo=timezone.utc)
            first = planner.ensure_plan(config=config, local_now=now)
            planner.mark_published(config=config, local_now=now)
            advanced = planner.ensure_plan(config=config, local_now=now + timedelta(minutes=10))

        self.assertEqual(first.local_day, advanced.local_day)
        self.assertEqual(advanced.cursor, 1)
        self.assertNotEqual(first.current_item().topic_key, advanced.current_item().topic_key)

    def test_planner_rolls_to_new_day_after_refresh_window(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            berlin = timezone(timedelta(hours=1))
            config = TwinrConfig(
                project_root=temp_dir,
                display_reserve_bus_refresh_after_local="05:30",
                display_reserve_bus_items_per_day=4,
            )
            planner = DisplayReserveDayPlanner.from_config(config)
            planner.candidate_loader = lambda _config, *, local_now, max_items: (
                _candidate(topic_key=f"topic-{local_now.date().isoformat()}", action="brief_update", attention_state="growing", salience=0.8),
            )[:max_items]

            late_night = datetime(2026, 3, 22, 23, 0, tzinfo=berlin)
            first = planner.ensure_plan(config=config, local_now=late_night)
            before_refresh = planner.ensure_plan(
                config=config,
                local_now=datetime(2026, 3, 23, 4, 30, tzinfo=berlin),
            )
            after_refresh = planner.ensure_plan(
                config=config,
                local_now=datetime(2026, 3, 23, 6, 0, tzinfo=berlin),
            )

        self.assertEqual(first.local_day, "2026-03-22")
        self.assertEqual(before_refresh.local_day, "2026-03-22")
        self.assertEqual(after_refresh.local_day, "2026-03-23")

    def test_planner_retries_same_day_empty_plan_after_short_backoff(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                display_reserve_bus_items_per_day=4,
            )
            planner = DisplayReserveDayPlanner.from_config(config)
            now = datetime(2026, 3, 22, 10, 0, tzinfo=timezone.utc)
            planner.candidate_loader = lambda _config, *, local_now, max_items: ()

            first = planner.ensure_plan(config=config, local_now=now)
            planner.candidate_loader = lambda _config, *, local_now, max_items: (
                _candidate(topic_key="ai companions", action="invite_follow_up", attention_state="shared_thread", salience=0.9),
            )[:max_items]
            still_empty = planner.ensure_plan(config=config, local_now=now + timedelta(seconds=30))
            rebuilt = planner.ensure_plan(config=config, local_now=now + timedelta(seconds=61))

        self.assertEqual(len(first.items), 0)
        self.assertEqual(len(still_empty.items), 0)
        self.assertEqual(len(rebuilt.items), 4)
        self.assertEqual(rebuilt.current_item().topic_key, "ai companions")


if __name__ == "__main__":
    unittest.main()
