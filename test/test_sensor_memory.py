from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test.longterm_test_program import make_test_extractor
from twinr.config import TwinrConfig
from twinr.memory.longterm import (
    LongTermConsolidationResultV1,
    LongTermMemoryObjectV1,
    LongTermMemoryService,
    LongTermSensorMemoryCompiler,
    LongTermSourceRefV1,
)


def _event_id(day: date, slug: str) -> str:
    return f"multimodal:{day.strftime('%Y%m%d')}T080000+0000:{slug}"


def _pattern(
    *,
    memory_id: str,
    daypart: str,
    event_days: tuple[date, ...],
    pattern_type: str,
) -> LongTermMemoryObjectV1:
    return LongTermMemoryObjectV1(
        memory_id=memory_id,
        kind="pattern",
        summary=f"Raw {pattern_type} pattern for {daypart}.",
        details="Synthetic multimodal pattern seed for tests.",
        source=LongTermSourceRefV1(
            source_type="proactive_monitor",
            event_ids=tuple(_event_id(day, memory_id.replace(":", "_")) for day in event_days),
            modality="sensor",
        ),
        status="active",
        confidence=0.64,
        sensitivity="low",
        slot_key=memory_id,
        value_key=pattern_type,
        valid_from=min(day.isoformat() for day in event_days),
        valid_to=max(day.isoformat() for day in event_days),
        attributes={
            "pattern_type": pattern_type,
            "daypart": daypart,
        },
    )


def _config(root: str, **overrides) -> TwinrConfig:
    return TwinrConfig(
        project_root=root,
        personality_dir="personality",
        memory_markdown_path=str(Path(root) / "state" / "MEMORY.md"),
        long_term_memory_enabled=True,
        long_term_memory_path=str(Path(root) / "state" / "chonkydb"),
        long_term_memory_sensor_memory_enabled=True,
        long_term_memory_sensor_baseline_days=7,
        long_term_memory_sensor_min_days_observed=4,
        long_term_memory_sensor_min_routine_ratio=0.6,
        long_term_memory_sensor_deviation_min_delta=0.5,
        **overrides,
    )


class LongTermSensorMemoryCompilerTests(unittest.TestCase):
    def test_builds_weekday_presence_routine_from_repeated_signal_days(self) -> None:
        compiler = LongTermSensorMemoryCompiler(
            enabled=True,
            baseline_days=7,
            min_days_observed=4,
            min_routine_ratio=0.6,
            deviation_min_delta=0.5,
        )
        reference = datetime(2026, 3, 16, 10, 0, tzinfo=timezone.utc)
        raw = _pattern(
            memory_id="pattern:presence:morning:near_device",
            daypart="morning",
            event_days=(
                date(2026, 3, 9),
                date(2026, 3, 10),
                date(2026, 3, 11),
                date(2026, 3, 13),
            ),
            pattern_type="presence",
        )

        result = compiler.compile(objects=(raw,), now=reference)
        routines = {item.memory_id: item for item in result.created_summaries}

        self.assertIn("routine:presence:weekday:morning", routines)
        attrs = dict(routines["routine:presence:weekday:morning"].attributes or {})
        self.assertEqual(attrs["days_observed"], 5)
        self.assertEqual(attrs["days_with_presence"], 4)
        self.assertEqual(attrs["typical_presence_ratio"], 0.8)

    def test_builds_weekend_interaction_routine_separately(self) -> None:
        compiler = LongTermSensorMemoryCompiler(
            enabled=True,
            baseline_days=14,
            min_days_observed=4,
            min_routine_ratio=0.6,
            deviation_min_delta=0.5,
        )
        reference = datetime(2026, 3, 16, 14, 0, tzinfo=timezone.utc)
        raw = _pattern(
            memory_id="pattern:print:button:afternoon",
            daypart="afternoon",
            event_days=(
                date(2026, 3, 7),
                date(2026, 3, 8),
                date(2026, 3, 14),
            ),
            pattern_type="interaction",
        )

        result = compiler.compile(objects=(raw,), now=reference)
        routines = {item.memory_id: item for item in result.created_summaries}

        self.assertIn("routine:interaction:print:weekend:afternoon", routines)
        attrs = dict(routines["routine:interaction:print:weekend:afternoon"].attributes or {})
        self.assertEqual(attrs["days_with_interaction"], 3)
        self.assertEqual(attrs["weekday_class"], "weekend")

    def test_builds_presence_deviation_when_current_daypart_is_missing_expected_presence(self) -> None:
        compiler = LongTermSensorMemoryCompiler(
            enabled=True,
            baseline_days=7,
            min_days_observed=4,
            min_routine_ratio=0.6,
            deviation_min_delta=0.5,
        )
        reference = datetime(2026, 3, 18, 9, 30, tzinfo=timezone.utc)
        raw = _pattern(
            memory_id="pattern:presence:morning:near_device",
            daypart="morning",
            event_days=(
                date(2026, 3, 11),
                date(2026, 3, 12),
                date(2026, 3, 13),
                date(2026, 3, 17),
            ),
            pattern_type="presence",
        )

        result = compiler.compile(objects=(raw,), now=reference)
        items = {item.memory_id: item for item in result.created_summaries}

        self.assertIn("deviation:presence:weekday:morning:2026-03-18", items)
        attrs = dict(items["deviation:presence:weekday:morning:2026-03-18"].attributes or {})
        self.assertEqual(attrs["deviation_type"], "missing_presence")
        self.assertTrue(attrs["requires_live_confirmation"])


class LongTermSensorMemoryServiceTests(unittest.TestCase):
    def test_service_run_sensor_memory_persists_routine_objects(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = LongTermMemoryService.from_config(_config(temp_dir), extractor=make_test_extractor())
            raw = _pattern(
                memory_id="pattern:presence:morning:near_device",
                daypart="morning",
                event_days=(
                    date(2026, 3, 9),
                    date(2026, 3, 10),
                    date(2026, 3, 11),
                    date(2026, 3, 13),
                ),
                pattern_type="presence",
            )
            service.object_store.apply_consolidation(
                LongTermConsolidationResultV1(
                    turn_id="turn:sensor-memory",
                    occurred_at=datetime(2026, 3, 16, 10, 0, tzinfo=timezone.utc),
                    episodic_objects=(),
                    durable_objects=(raw,),
                    deferred_objects=(),
                    conflicts=(),
                    graph_edges=(),
                )
            )

            result = service.run_sensor_memory(now=datetime(2026, 3, 16, 10, 0, tzinfo=timezone.utc))
            objects = {item.memory_id: item for item in service.object_store.load_objects()}
            service.shutdown()

        self.assertTrue(result.created_summaries)
        self.assertIn("routine:presence:weekday:morning", objects)


if __name__ == "__main__":
    unittest.main()
