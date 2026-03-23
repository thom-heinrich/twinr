from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.agent.personality.display_impulses import AmbientDisplayImpulseCandidate
from twinr.display.ambient_impulse_history import DisplayAmbientImpulseHistoryStore
from twinr.proactive.runtime.display_reserve_companion_planner import (
    DisplayReserveCompanionPlanner,
)
from twinr.proactive.runtime.display_reserve_day_plan import DisplayReserveDayPlan
from twinr.memory.longterm.core.models import LongTermReflectionResultV1
from twinr.memory.longterm.core.models import LongTermMemoryObjectV1, LongTermSourceRefV1


def _candidate(topic_key: str) -> AmbientDisplayImpulseCandidate:
    return AmbientDisplayImpulseCandidate(
        topic_key=topic_key,
        title=topic_key.title(),
        source="world",
        action="ask_one",
        attention_state="forming",
        salience=0.86,
        eyebrow="",
        headline=f"Was denkst du zu {topic_key.title()}?",
        body="Wenn du magst, schaue ich weiter darauf.",
        symbol="sparkles",
        accent="warm",
        reason=f"seed:{topic_key}",
        candidate_family="world",
    )


class _FakeLongTermMemoryService:
    def __init__(self, *, reflection: LongTermReflectionResultV1 | None = None) -> None:
        self.calls: list[object | None] = []
        self.reflection = reflection or LongTermReflectionResultV1(
            reflected_objects=(
                LongTermMemoryObjectV1(
                    memory_id="reflection:1",
                    kind="summary",
                    summary="Reflection object.",
                    source=LongTermSourceRefV1(source_type="reflection", event_ids=("turn:reflection",)),
                    status="active",
                    confidence=0.8,
                    sensitivity="normal",
                ),
            ),
            created_summaries=(
                LongTermMemoryObjectV1(
                    memory_id="summary:1",
                    kind="summary",
                    summary="Created summary.",
                    source=LongTermSourceRefV1(source_type="reflection", event_ids=("turn:summary",)),
                    status="active",
                    confidence=0.82,
                    sensitivity="normal",
                ),
            ),
            midterm_packets=(),
        )

    def run_reflection(self, *, search_backend=None) -> LongTermReflectionResultV1:
        self.calls.append(search_backend)
        return self.reflection


class DisplayReserveCompanionPlannerTests(unittest.TestCase):
    def test_nightly_maintenance_prepares_next_day_plan_and_records_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                display_reserve_bus_refresh_after_local="05:30",
                display_reserve_bus_nightly_after_local="00:30",
            )
            planner = DisplayReserveCompanionPlanner.from_config(config)
            fake_memory = _FakeLongTermMemoryService()
            planner.long_term_memory_factory = lambda _config: fake_memory
            planner.candidate_loader = lambda _config, *, local_now, max_items: (
                _candidate("ai companions"),
                _candidate("world politics"),
            )[:max_items]
            history = DisplayAmbientImpulseHistoryStore.from_config(config)
            shown_at = datetime(2026, 3, 22, 20, 0, tzinfo=timezone.utc)
            exposure = history.append_exposure(
                source="world",
                topic_key="ai companions",
                title="AI companions",
                headline="Was denkst du zu AI companions?",
                body="Wenn du magst, schaue ich weiter darauf.",
                action="ask_one",
                attention_state="forming",
                shown_at=shown_at,
                expires_at=shown_at + timedelta(minutes=10),
                metadata={"candidate_family": "world"},
            )
            history.resolve_feedback(
                exposure_id=exposure.exposure_id,
                response_status="engaged",
                response_sentiment="positive",
                response_at=shown_at + timedelta(minutes=2),
                response_mode="voice_immediate_pickup",
                response_latency_seconds=12.0,
                response_turn_id="turn:1",
                response_target="ai companions",
                response_summary="Immediate pickup.",
            )

            result = planner.maybe_run_nightly_maintenance(
                config=config,
                local_now=datetime(2026, 3, 23, 1, 0, tzinfo=timezone.utc),
                search_backend="search-backend",
            )
            prepared = planner.prepared_store.load()
            state = planner.state_store.load()

        self.assertEqual(result.action, "prepared")
        self.assertEqual(fake_memory.calls, ["search-backend"])
        self.assertIsNotNone(prepared)
        self.assertIsNotNone(state)
        assert prepared is not None
        assert state is not None
        self.assertEqual(prepared.local_day, "2026-03-23")
        self.assertCountEqual(
            [item.topic_key for item in prepared.items],
            ["ai companions", "world politics"],
        )
        self.assertEqual(state.reflection_reflected_object_count, 1)
        self.assertEqual(state.reflection_created_summary_count, 1)
        self.assertEqual(state.outcome_summary.engaged_count, 1)
        self.assertEqual(state.outcome_summary.immediate_pickup_count, 1)
        self.assertEqual(state.outcome_summary.positive_topics, ("ai companions",))
        self.assertIn("ai companions", state.positive_topics)

    def test_ensure_plan_adopts_prepared_day_after_refresh_window(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                display_reserve_bus_refresh_after_local="05:30",
                display_reserve_bus_nightly_after_local="00:30",
            )
            planner = DisplayReserveCompanionPlanner.from_config(config)
            planner.long_term_memory_factory = lambda _config: _FakeLongTermMemoryService()
            planner.candidate_loader = lambda _config, *, local_now, max_items: (
                _candidate(f"topic-{local_now.date().isoformat()}"),
            )[:max_items]
            planner.store.save(
                DisplayReserveDayPlan(
                    local_day="2026-03-22",
                    generated_at=datetime(2026, 3, 22, 22, 0, tzinfo=timezone.utc).isoformat(),
                    cursor=0,
                    items=planner.day_planner.build_plan_for_day(
                        config=config,
                        local_day=datetime(2026, 3, 22, 22, 0, tzinfo=timezone.utc).date(),
                        local_now=datetime(2026, 3, 22, 22, 0, tzinfo=timezone.utc),
                    ).items,
                    candidate_count=1,
                )
            )

            planner.maybe_run_nightly_maintenance(
                config=config,
                local_now=datetime(2026, 3, 23, 1, 0, tzinfo=timezone.utc),
            )
            before_refresh = planner.ensure_plan(
                config=config,
                local_now=datetime(2026, 3, 23, 4, 0, tzinfo=timezone.utc),
            )
            after_refresh = planner.ensure_plan(
                config=config,
                local_now=datetime(2026, 3, 23, 6, 0, tzinfo=timezone.utc),
            )

        self.assertEqual(before_refresh.local_day, "2026-03-22")
        self.assertEqual(after_refresh.local_day, "2026-03-23")
        self.assertEqual(after_refresh.current_item().topic_key, "topic-2026-03-23")
        self.assertIsNone(planner.prepared_store.load())

    def test_nightly_maintenance_does_not_repeat_after_same_day_adoption(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                display_reserve_bus_refresh_after_local="05:30",
                display_reserve_bus_nightly_after_local="00:30",
            )
            planner = DisplayReserveCompanionPlanner.from_config(config)
            fake_memory = _FakeLongTermMemoryService()
            planner.long_term_memory_factory = lambda _config: fake_memory
            planner.candidate_loader = lambda _config, *, local_now, max_items: (
                _candidate("peace and diplomacy"),
            )[:max_items]

            first = planner.maybe_run_nightly_maintenance(
                config=config,
                local_now=datetime(2026, 3, 23, 1, 0, tzinfo=timezone.utc),
            )
            second = planner.maybe_run_nightly_maintenance(
                config=config,
                local_now=datetime(2026, 3, 23, 2, 0, tzinfo=timezone.utc),
            )
            adopted = planner.ensure_plan(
                config=config,
                local_now=datetime(2026, 3, 23, 6, 0, tzinfo=timezone.utc),
            )

        self.assertEqual(first.action, "prepared")
        self.assertEqual(second.action, "skipped")
        self.assertEqual(second.reason, "already_prepared")
        self.assertEqual(adopted.local_day, "2026-03-23")
        self.assertEqual(len(fake_memory.calls), 1)


if __name__ == "__main__":
    unittest.main()
