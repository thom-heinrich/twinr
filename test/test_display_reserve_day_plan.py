from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.agent.personality.display_impulses import AmbientDisplayImpulseCandidate
from twinr.display.reserve_bus_feedback import DisplayReserveBusFeedbackStore
from twinr.proactive.runtime.display_reserve_day_plan import (
    DisplayReserveDayPlanStore,
    DisplayReserveDayPlanner,
    DisplayReservePlannedItem,
)


def _candidate(
    *,
    topic_key: str,
    action: str,
    attention_state: str,
    salience: float,
    source: str = "world",
    candidate_family: str = "world",
) -> AmbientDisplayImpulseCandidate:
    return AmbientDisplayImpulseCandidate(
        topic_key=topic_key,
        title=topic_key.title(),
        source=source,
        action=action,
        attention_state=attention_state,
        salience=salience,
        eyebrow="IM BLICK",
        headline=topic_key.title(),
        body="Da tut sich gerade etwas.",
        symbol="sparkles",
        accent="info",
        reason=f"seed:{topic_key}",
        candidate_family=candidate_family,
    )


class DisplayReserveDayPlanTests(unittest.TestCase):
    def test_planned_item_keeps_question_headlines_beyond_old_64_char_limit(self) -> None:
        candidate = AmbientDisplayImpulseCandidate(
            topic_key="ai companions",
            title="AI companions",
            source="world",
            action="ask_one",
            attention_state="forming",
            salience=0.88,
            eyebrow="",
            headline="Ich habe dazu heute etwas gelesen. Was meinst du dazu, wenn wir spaeter kurz auf die Entwicklung schauen?",
            body="Es geht um AI companions. Dann weiss ich besser, ob ich weiter darauf achten soll.",
            symbol="question",
            accent="warm",
            reason="test",
        )

        item = DisplayReservePlannedItem.from_candidate(
            candidate,
            hold_seconds=1200.0,
            reason="test",
        )

        self.assertGreater(len(candidate.headline), 64)
        self.assertEqual(item.headline, candidate.headline)

    def test_planner_builds_and_persists_one_unique_cycle(self) -> None:
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
        self.assertEqual(len(plan.items), 2)
        self.assertEqual(loaded.cursor, 0)
        self.assertEqual(loaded.current_item().topic_key, plan.current_item().topic_key)

    def test_planner_keeps_same_day_cards_rotating_after_one_full_cycle(self) -> None:
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
            second = planner.ensure_plan(config=config, local_now=now + timedelta(minutes=10))
            planner.mark_published(config=config, local_now=now + timedelta(minutes=11))
            wrapped = planner.ensure_plan(config=config, local_now=now + timedelta(minutes=20))

        self.assertEqual(first.local_day, second.local_day)
        self.assertEqual(second.cursor, 1)
        self.assertNotEqual(first.current_item().topic_key, second.current_item().topic_key)
        self.assertEqual(wrapped.cursor, 2)
        self.assertEqual(wrapped.current_item().topic_key, first.current_item().topic_key)

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
        self.assertEqual(len(rebuilt.items), 1)
        self.assertEqual(rebuilt.current_item().topic_key, "ai companions")

    def test_planner_keeps_same_day_cycle_when_upstream_candidates_change(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir, display_reserve_bus_items_per_day=6)
            planner = DisplayReserveDayPlanner.from_config(config)
            now = datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc)
            planner.candidate_loader = lambda _config, *, local_now, max_items: (
                _candidate(topic_key="ai companions", action="invite_follow_up", attention_state="shared_thread", salience=0.9),
                _candidate(topic_key="world politics", action="brief_update", attention_state="growing", salience=0.7),
            )[:max_items]

            first = planner.ensure_plan(config=config, local_now=now)
            planner.mark_published(config=config, local_now=now + timedelta(minutes=1))
            planner.mark_published(config=config, local_now=now + timedelta(minutes=2))
            planner.candidate_loader = lambda _config, *, local_now, max_items: (
                _candidate(topic_key="hamburg civic life", action="ask_one", attention_state="forming", salience=0.95),
            )[:max_items]

            continued = planner.ensure_plan(config=config, local_now=now + timedelta(minutes=3))

        self.assertEqual(len(first.items), 2)
        self.assertEqual(first.current_item().topic_key, "ai companions")
        self.assertEqual(continued.cursor, 2)
        self.assertEqual(len(continued.items), 2)
        self.assertEqual(continued.current_item().topic_key, "ai companions")

    def test_recent_immediate_pickup_of_current_topic_retires_it_and_promotes_next(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir, display_reserve_bus_items_per_day=6)
            planner = DisplayReserveDayPlanner.from_config(config)
            planner.candidate_loader = lambda _config, *, local_now, max_items: (
                _candidate(topic_key="world politics", action="brief_update", attention_state="growing", salience=0.94),
                _candidate(topic_key="ai companions", action="invite_follow_up", attention_state="shared_thread", salience=0.42),
            )[:max_items]
            feedback_store = DisplayReserveBusFeedbackStore.from_config(config)
            now = datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc)

            first = planner.ensure_plan(config=config, local_now=now)
            planner.mark_published(config=config, local_now=now + timedelta(minutes=1))
            feedback_store.record_reaction(
                topic_key="world politics",
                reaction="immediate_engagement",
                intensity=1.0,
                reason="The user immediately spoke about the shown world politics card.",
                now=now + timedelta(minutes=5),
            )
            rebuilt = planner.ensure_plan(config=config, local_now=now + timedelta(minutes=6))

        self.assertEqual(first.current_item().topic_key, "world politics")
        self.assertEqual(rebuilt.current_item().topic_key, "ai companions")
        self.assertEqual(rebuilt.retired_topic_keys, ("world politics",))
        self.assertEqual([item.topic_key for item in rebuilt.active_items()], ["ai companions"])

    def test_feedback_rebuild_retires_only_answered_topic_from_same_day_rotation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir, display_reserve_bus_items_per_day=6)
            planner = DisplayReserveDayPlanner.from_config(config)
            planner.candidate_loader = lambda _config, *, local_now, max_items: (
                _candidate(topic_key="world politics", action="brief_update", attention_state="growing", salience=0.94),
                _candidate(topic_key="ai companions", action="hint", attention_state="background", salience=0.52),
                _candidate(topic_key="peace and diplomacy", action="ask_one", attention_state="forming", salience=0.68),
            )[:max_items]
            feedback_store = DisplayReserveBusFeedbackStore.from_config(config)
            now = datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc)

            first = planner.ensure_plan(config=config, local_now=now)
            planner.mark_published(config=config, local_now=now + timedelta(minutes=1))
            feedback_store.record_reaction(
                topic_key="world politics",
                reaction="immediate_engagement",
                intensity=1.0,
                reason="The user answered the shown world politics card.",
                now=now + timedelta(minutes=2),
            )
            rebuilt = planner.ensure_plan(config=config, local_now=now + timedelta(minutes=3))

        self.assertEqual(first.current_item().topic_key, "world politics")
        self.assertEqual(rebuilt.retired_topic_keys, ("world politics",))
        self.assertNotIn("world politics", [item.topic_key for item in rebuilt.active_items()])
        self.assertEqual([item.topic_key for item in rebuilt.active_items()], ["peace and diplomacy", "ai companions"])

    def test_ignored_feedback_keeps_topic_in_same_day_rotation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir, display_reserve_bus_items_per_day=4)
            planner = DisplayReserveDayPlanner.from_config(config)
            planner.candidate_loader = lambda _config, *, local_now, max_items: (
                _candidate(topic_key="ai companions", action="invite_follow_up", attention_state="shared_thread", salience=0.9),
            )[:max_items]
            feedback_store = DisplayReserveBusFeedbackStore.from_config(config)
            now = datetime(2026, 3, 22, 13, 0, tzinfo=timezone.utc)

            planner.ensure_plan(config=config, local_now=now)
            planner.mark_published(config=config, local_now=now + timedelta(minutes=1))
            feedback_store.record_reaction(
                topic_key="ai companions",
                reaction="ignored",
                intensity=1.0,
                reason="The user did not pick up the card yet.",
                now=now + timedelta(minutes=2),
            )
            rebuilt = planner.ensure_plan(config=config, local_now=now + timedelta(minutes=3))

        self.assertEqual(rebuilt.retired_topic_keys, ())
        self.assertEqual([item.topic_key for item in rebuilt.active_items()], ["ai companions"])
        self.assertEqual(rebuilt.current_item().topic_key, "ai companions")

    def test_planner_spreads_sources_when_multiple_families_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir, display_reserve_bus_items_per_day=6)
            planner = DisplayReserveDayPlanner.from_config(config)
            planner.candidate_loader = lambda _config, *, local_now, max_items: (
                AmbientDisplayImpulseCandidate(
                    topic_key="memory follow up",
                    title="Memory follow up",
                    source="memory_follow_up",
                    action="ask_one",
                    attention_state="forming",
                    salience=0.88,
                    eyebrow="",
                    headline="Wie ist es damit weitergegangen?",
                    body="Da fehlt mir noch ein kleines Stueck.",
                    symbol="question",
                    accent="warm",
                    reason="memory",
                ),
                _candidate(topic_key="ai companions", action="invite_follow_up", attention_state="shared_thread", salience=0.82),
                _candidate(topic_key="world politics", action="brief_update", attention_state="growing", salience=0.78),
            )[:max_items]
            now = datetime(2026, 3, 22, 14, 0, tzinfo=timezone.utc)

            plan = planner.ensure_plan(config=config, local_now=now)

        first_sources = [item.source for item in plan.items[:3]]
        self.assertEqual(len(first_sources), 3)
        self.assertNotEqual(first_sources[0], first_sources[1])
        self.assertNotEqual(first_sources[1], first_sources[2])

    def test_planner_spreads_candidate_families_even_when_sources_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir, display_reserve_bus_items_per_day=6)
            planner = DisplayReserveDayPlanner.from_config(config)
            planner.candidate_loader = lambda _config, *, local_now, max_items: (
                _candidate(
                    topic_key="ai companions",
                    action="invite_follow_up",
                    attention_state="shared_thread",
                    salience=0.92,
                    source="world",
                    candidate_family="world",
                ),
                _candidate(
                    topic_key="janina follow up",
                    action="ask_one",
                    attention_state="forming",
                    salience=0.86,
                    source="world",
                    candidate_family="memory_follow_up",
                ),
                _candidate(
                    topic_key="hamburg civic life",
                    action="brief_update",
                    attention_state="growing",
                    salience=0.80,
                    source="world",
                    candidate_family="place",
                ),
            )[:max_items]
            now = datetime(2026, 3, 22, 15, 0, tzinfo=timezone.utc)

            plan = planner.ensure_plan(config=config, local_now=now)

        first_families = [item.candidate_family for item in plan.items[:3]]
        self.assertEqual(first_families, ["world", "memory_follow_up", "place"])

    def test_default_hold_seconds_rotate_more_often_but_still_remain_calm(self) -> None:
        config = TwinrConfig(project_root=".")
        planner = DisplayReserveDayPlanner.from_config(config)
        candidate = _candidate(
            topic_key="ai companions",
            action="invite_follow_up",
            attention_state="shared_thread",
            salience=0.92,
        )

        hold_seconds = planner._hold_seconds_for_candidate(config=config, candidate=candidate)

        self.assertGreaterEqual(hold_seconds, 240.0)
        self.assertLessEqual(hold_seconds, 720.0)
        self.assertLessEqual(hold_seconds, config.display_reserve_bus_max_hold_s)


if __name__ == "__main__":
    unittest.main()
