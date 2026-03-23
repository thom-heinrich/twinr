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
    LongTermEnvironmentProfileCompiler,
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


def _audio_pattern(
    *,
    interaction_type: str,
    daypart: str,
    event_days: tuple[date, ...],
) -> LongTermMemoryObjectV1:
    return LongTermMemoryObjectV1(
        memory_id=f"pattern:audio_interaction:{interaction_type}:{daypart}",
        kind="pattern",
        summary=f"Raw audio interaction pattern for {interaction_type} in the {daypart}.",
        details="Synthetic ReSpeaker audio interaction pattern seed for tests.",
        source=LongTermSourceRefV1(
            source_type="proactive_monitor",
            event_ids=tuple(_event_id(day, f"{interaction_type}_{daypart}") for day in event_days),
            modality="sensor",
        ),
        status="active",
        confidence=0.66,
        sensitivity="low",
        slot_key=f"pattern:audio_interaction:{interaction_type}:{daypart}",
        value_key="audio_interaction",
        valid_from=min(day.isoformat() for day in event_days),
        valid_to=max(day.isoformat() for day in event_days),
        attributes={
            "pattern_type": "interaction",
            "interaction_type": interaction_type,
            "daypart": daypart,
            "memory_domain": "respeaker_audio_routine",
            "memory_class": "session_memory",
            "source_sensor": "respeaker_xvf3800",
            "source_type": "observed",
            "requires_confirmation": True,
        },
    )


def _smart_home_event_id(occurred_at: datetime, *, node_token: str, suffix: str) -> str:
    return f"smart_home_env:{occurred_at.strftime('%Y%m%dT%H%M%S%f%z')}:{node_token}:{suffix}"


def _smart_home_motion_pattern(
    *,
    day: date,
    node_id: str,
    event_hours: tuple[int, ...],
) -> LongTermMemoryObjectV1:
    node_token = node_id.replace(":", "_").replace("-", "_")
    event_datetimes = tuple(
        datetime(day.year, day.month, day.day, hour, 0, tzinfo=timezone.utc)
        for hour in event_hours
    )
    return LongTermMemoryObjectV1(
        memory_id=f"pattern:smart_home_node_activity:{node_token}:{day.isoformat()}",
        kind="pattern",
        summary="Synthetic smart-home motion node activity pattern.",
        details="Synthetic room-agnostic smart-home motion seed for tests.",
        source=LongTermSourceRefV1(
            source_type="smart_home_sensor",
            event_ids=tuple(
                _smart_home_event_id(occurred_at, node_token=node_token, suffix=f"{index:02d}")
                for index, occurred_at in enumerate(event_datetimes)
            ),
            modality="sensor",
        ),
        status="active",
        confidence=0.66,
        sensitivity="low",
        slot_key=f"pattern:smart_home_node_activity:{node_id}:{day.isoformat()}",
        value_key="smart_home_motion_node_activity",
        valid_from=day.isoformat(),
        valid_to=day.isoformat(),
        attributes={
            "memory_domain": "smart_home_environment",
            "environment_id": "home:main",
            "environment_signal_type": "motion_node_activity",
            "node_id": node_id,
            "provider": "hue",
            "route_id": "192.168.178.22",
            "source_entity_id": node_id,
            "provider_label": node_id,
            "provider_area_label": "Erdgeschoss",
        },
    )


def _smart_home_health_pattern(
    *,
    day: date,
    node_id: str,
    state: str,
    hour: int,
) -> LongTermMemoryObjectV1:
    node_token = node_id.replace(":", "_").replace("-", "_")
    occurred_at = datetime(day.year, day.month, day.day, hour, 30, tzinfo=timezone.utc)
    return LongTermMemoryObjectV1(
        memory_id=f"pattern:smart_home_node_health:{node_token}:{state}:{day.isoformat()}",
        kind="pattern",
        summary="Synthetic smart-home node health pattern.",
        details="Synthetic room-agnostic smart-home health seed for tests.",
        source=LongTermSourceRefV1(
            source_type="smart_home_sensor",
            event_ids=(
                _smart_home_event_id(occurred_at, node_token=node_token, suffix=state),
            ),
            modality="sensor",
        ),
        status="active",
        confidence=0.68,
        sensitivity="low",
        slot_key=f"pattern:smart_home_node_health:{node_id}:{day.isoformat()}",
        value_key="smart_home_node_health",
        valid_from=day.isoformat(),
        valid_to=day.isoformat(),
        attributes={
            "memory_domain": "smart_home_environment",
            "environment_id": "home:main",
            "environment_signal_type": "node_health",
            "health_state": state,
            "node_id": node_id,
            "provider": "hue",
            "route_id": "192.168.178.22",
            "source_entity_id": node_id,
            "provider_label": node_id,
            "provider_area_label": "Erdgeschoss",
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

    def test_builds_voice_conversation_start_routine_from_respeaker_audio_patterns(self) -> None:
        compiler = LongTermSensorMemoryCompiler(
            enabled=True,
            baseline_days=7,
            min_days_observed=4,
            min_routine_ratio=0.6,
            deviation_min_delta=0.5,
        )
        reference = datetime(2026, 3, 16, 10, 0, tzinfo=timezone.utc)
        raw = _audio_pattern(
            interaction_type="conversation_start_audio",
            daypart="morning",
            event_days=(
                date(2026, 3, 9),
                date(2026, 3, 10),
                date(2026, 3, 11),
                date(2026, 3, 13),
            ),
        )

        result = compiler.compile(objects=(raw,), now=reference)
        routines = {item.memory_id: item for item in result.created_summaries}

        self.assertIn("routine:interaction:conversation_start_audio:weekday:morning", routines)
        attrs = dict(routines["routine:interaction:conversation_start_audio:weekday:morning"].attributes or {})
        self.assertEqual(attrs["interaction_type"], "conversation_start_audio")
        self.assertEqual(attrs["days_with_interaction"], 4)

    def test_builds_observed_voice_response_channel_preference_from_repeated_audio_patterns(self) -> None:
        compiler = LongTermSensorMemoryCompiler(
            enabled=True,
            baseline_days=7,
            min_days_observed=4,
            min_routine_ratio=0.6,
            deviation_min_delta=0.5,
        )
        reference = datetime(2026, 3, 16, 10, 0, tzinfo=timezone.utc)
        start_pattern = _audio_pattern(
            interaction_type="conversation_start_audio",
            daypart="morning",
            event_days=(
                date(2026, 3, 9),
                date(2026, 3, 10),
                date(2026, 3, 11),
                date(2026, 3, 13),
            ),
        )
        resume_pattern = _audio_pattern(
            interaction_type="resume_follow_up",
            daypart="morning",
            event_days=(
                date(2026, 3, 10),
                date(2026, 3, 11),
                date(2026, 3, 13),
            ),
        )

        result = compiler.compile(objects=(start_pattern, resume_pattern), now=reference)
        items = {item.memory_id: item for item in result.created_summaries}

        preference = items["preference:response_channel:voice:weekday:morning"]
        attrs = dict(preference.attributes or {})
        self.assertEqual(preference.status, "candidate")
        self.assertEqual(attrs["memory_class"], "observed_preference")
        self.assertEqual(attrs["preferred_channel"], "voice")
        self.assertTrue(attrs["requires_confirmation"])
        self.assertEqual(attrs["days_with_voice_signal"], 4)

    def test_builds_resume_follow_up_routine_from_respeaker_audio_patterns(self) -> None:
        compiler = LongTermSensorMemoryCompiler(
            enabled=True,
            baseline_days=14,
            min_days_observed=4,
            min_routine_ratio=0.6,
            deviation_min_delta=0.5,
        )
        reference = datetime(2026, 3, 16, 18, 0, tzinfo=timezone.utc)
        raw = _audio_pattern(
            interaction_type="resume_follow_up",
            daypart="evening",
            event_days=(
                date(2026, 3, 2),
                date(2026, 3, 3),
                date(2026, 3, 4),
                date(2026, 3, 5),
                date(2026, 3, 9),
                date(2026, 3, 10),
                date(2026, 3, 12),
            ),
        )

        result = compiler.compile(objects=(raw,), now=reference)
        routines = {item.memory_id: item for item in result.created_summaries}

        self.assertIn("routine:interaction:resume_follow_up:weekday:evening", routines)
        attrs = dict(routines["routine:interaction:resume_follow_up:weekday:evening"].attributes or {})
        self.assertEqual(attrs["interaction_type"], "resume_follow_up")
        self.assertEqual(attrs["days_with_interaction"], 7)

    def test_environment_profile_compiler_builds_day_profile_baseline_and_activity_drop_deviation(self) -> None:
        compiler = LongTermEnvironmentProfileCompiler(
            enabled=True,
            baseline_days=7,
            history_days=21,
            min_baseline_days=4,
        )
        reference = datetime(2026, 3, 18, 18, 0, tzinfo=timezone.utc)
        history = (
            _smart_home_motion_pattern(day=date(2026, 3, 11), node_id="route:192.168.178.22:node-a", event_hours=(7, 8, 12, 13, 18, 19)),
            _smart_home_motion_pattern(day=date(2026, 3, 11), node_id="route:192.168.178.22:node-b", event_hours=(7, 12, 18)),
            _smart_home_motion_pattern(day=date(2026, 3, 12), node_id="route:192.168.178.22:node-a", event_hours=(7, 8, 12, 13, 18, 19)),
            _smart_home_motion_pattern(day=date(2026, 3, 12), node_id="route:192.168.178.22:node-b", event_hours=(7, 12, 18)),
            _smart_home_motion_pattern(day=date(2026, 3, 13), node_id="route:192.168.178.22:node-a", event_hours=(7, 8, 12, 13, 18, 19)),
            _smart_home_motion_pattern(day=date(2026, 3, 13), node_id="route:192.168.178.22:node-b", event_hours=(7, 12, 18)),
            _smart_home_motion_pattern(day=date(2026, 3, 17), node_id="route:192.168.178.22:node-a", event_hours=(7, 8, 12, 13, 18, 19)),
            _smart_home_motion_pattern(day=date(2026, 3, 17), node_id="route:192.168.178.22:node-b", event_hours=(7, 12, 18)),
            _smart_home_motion_pattern(day=date(2026, 3, 18), node_id="route:192.168.178.22:node-a", event_hours=(7,)),
            _smart_home_health_pattern(day=date(2026, 3, 18), node_id="route:192.168.178.22:node-b", state="offline", hour=9),
        )

        result = compiler.compile(objects=history, now=reference)
        items = {item.memory_id: item for item in result.created_summaries}

        profile = items["environment_profile:home_main:day:2026-03-18"]
        baseline = items["environment_baseline:home_main:weekday:rolling_7d"]
        long_baseline = items["environment_baseline:home_main:long:weekday:rolling_56d"]
        deviation = items["environment_deviation:home_main:daily_activity_drop:2026-03-18"]
        deviation_event = items["environment_deviation_event:home_main:acute_deviation:2026-03-18"]
        quality_state = items["environment_quality_state:home_main:2026-03-18"]

        profile_attrs = dict(profile.attributes or {})
        baseline_attrs = dict(baseline.attributes or {})
        long_baseline_attrs = dict(long_baseline.attributes or {})
        deviation_attrs = dict(deviation.attributes or {})
        deviation_event_attrs = dict(deviation_event.attributes or {})
        quality_attrs = dict(quality_state.attributes or {})

        self.assertEqual(profile_attrs["summary_type"], "environment_day_profile")
        self.assertEqual(profile_attrs["markers"]["active_epoch_count_day"], 1)
        self.assertEqual(profile_attrs["markers"]["unique_active_node_count_day"], 1)
        self.assertIn("device_offline_present", profile_attrs["quality_flags"])
        self.assertEqual(baseline_attrs["pattern_type"], "environment_baseline")
        self.assertEqual(baseline_attrs["baseline_kind"], "short")
        self.assertEqual(long_baseline_attrs["baseline_kind"], "long")
        self.assertEqual(baseline_attrs["sample_count"], 4)
        self.assertIn("active_epoch_count_day", baseline_attrs["marker_stats"])
        self.assertIn("mad", baseline_attrs["marker_stats"]["active_epoch_count_day"])
        self.assertEqual(deviation_attrs["deviation_type"], "daily_activity_drop")
        self.assertEqual(deviation_attrs["markers"][0]["name"], "active_epoch_count_day")
        self.assertEqual(deviation_event_attrs["summary_type"], "environment_deviation_event")
        self.assertEqual(deviation_event_attrs["classification"], "acute_deviation")
        self.assertEqual(quality_attrs["summary_type"], "environment_quality_state")
        self.assertEqual(quality_attrs["classification"], "blocked")

    def test_sensor_memory_compiler_merges_environment_profile_outputs(self) -> None:
        compiler = LongTermSensorMemoryCompiler(
            enabled=True,
            baseline_days=7,
            min_days_observed=4,
            min_routine_ratio=0.6,
            deviation_min_delta=0.5,
        )
        reference = datetime(2026, 3, 18, 18, 0, tzinfo=timezone.utc)
        history = (
            _smart_home_motion_pattern(day=date(2026, 3, 11), node_id="route:192.168.178.22:node-a", event_hours=(7, 8, 12, 13, 18, 19)),
            _smart_home_motion_pattern(day=date(2026, 3, 12), node_id="route:192.168.178.22:node-a", event_hours=(7, 8, 12, 13, 18, 19)),
            _smart_home_motion_pattern(day=date(2026, 3, 13), node_id="route:192.168.178.22:node-a", event_hours=(7, 8, 12, 13, 18, 19)),
            _smart_home_motion_pattern(day=date(2026, 3, 17), node_id="route:192.168.178.22:node-a", event_hours=(7, 8, 12, 13, 18, 19)),
            _smart_home_motion_pattern(day=date(2026, 3, 18), node_id="route:192.168.178.22:node-a", event_hours=(7,)),
        )

        result = compiler.compile(objects=history, now=reference)
        items = {item.memory_id: item for item in result.created_summaries}

        self.assertIn("environment_profile:home_main:day:2026-03-18", items)
        self.assertIn("environment_baseline:home_main:weekday:rolling_7d", items)

    def test_environment_profile_compiler_detects_change_point_and_regime(self) -> None:
        compiler = LongTermEnvironmentProfileCompiler(
            enabled=True,
            baseline_days=14,
            short_baseline_days=14,
            long_baseline_days=56,
            history_days=56,
            min_baseline_days=7,
            drift_min_days=5,
            regime_accept_days=8,
        )
        reference = datetime(2026, 3, 30, 18, 0, tzinfo=timezone.utc)
        high_days = tuple(date(2026, 3, day) for day in range(1, 21))
        low_days = tuple(date(2026, 3, day) for day in range(21, 31))
        history = tuple(
            _smart_home_motion_pattern(
                day=day,
                node_id="route:192.168.178.22:node-a",
                event_hours=(7, 8, 12, 13, 18, 19),
            )
            for day in high_days
        ) + tuple(
            _smart_home_motion_pattern(
                day=day,
                node_id="route:192.168.178.22:node-a",
                event_hours=(7, 18),
            )
            for day in low_days
        )

        result = compiler.compile(objects=history, now=reference)
        items = {item.memory_id: item for item in result.created_summaries}

        change_point = items["environment_change_point:home_main:2026-03-30"]
        regime = items["environment_regime:home_main:2026-03-21"]
        change_attrs = dict(change_point.attributes or {})
        regime_attrs = dict(regime.attributes or {})

        self.assertEqual(change_attrs["summary_type"], "environment_change_point")
        self.assertIn("active_epoch_count_day", {marker["name"] for marker in change_attrs["markers"]})
        self.assertEqual(regime_attrs["pattern_type"], "environment_regime")
        self.assertEqual(regime_attrs["regime_started_on"], "2026-03-21")


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

    def test_service_run_sensor_memory_persists_environment_profile_objects(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = LongTermMemoryService.from_config(_config(temp_dir), extractor=make_test_extractor())
            raw_objects = (
                _smart_home_motion_pattern(day=date(2026, 3, 11), node_id="route:192.168.178.22:node-a", event_hours=(7, 8, 12, 13, 18, 19)),
                _smart_home_motion_pattern(day=date(2026, 3, 12), node_id="route:192.168.178.22:node-a", event_hours=(7, 8, 12, 13, 18, 19)),
                _smart_home_motion_pattern(day=date(2026, 3, 13), node_id="route:192.168.178.22:node-a", event_hours=(7, 8, 12, 13, 18, 19)),
                _smart_home_motion_pattern(day=date(2026, 3, 17), node_id="route:192.168.178.22:node-a", event_hours=(7, 8, 12, 13, 18, 19)),
                _smart_home_motion_pattern(day=date(2026, 3, 18), node_id="route:192.168.178.22:node-a", event_hours=(7,)),
            )
            service.object_store.apply_consolidation(
                LongTermConsolidationResultV1(
                    turn_id="turn:smart-home-environment",
                    occurred_at=datetime(2026, 3, 18, 18, 0, tzinfo=timezone.utc),
                    episodic_objects=(),
                    durable_objects=raw_objects,
                    deferred_objects=(),
                    conflicts=(),
                    graph_edges=(),
                )
            )

            result = service.run_sensor_memory(now=datetime(2026, 3, 18, 18, 0, tzinfo=timezone.utc))
            objects = {item.memory_id: item for item in service.object_store.load_objects()}
            service.shutdown()

        self.assertTrue(result.created_summaries)
        self.assertIn("environment_profile:home_main:day:2026-03-18", objects)
        self.assertIn("environment_baseline:home_main:weekday:rolling_7d", objects)


if __name__ == "__main__":
    unittest.main()
