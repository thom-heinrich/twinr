from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.memory.longterm import (
    LongTermMemoryObjectV1,
    LongTermReflectionResultV1,
    LongTermSourceRefV1,
)
from twinr.memory.longterm.runtime.service_impl.lifecycle import LongTermMemoryServiceLifecycleMixin
from twinr.memory.longterm.runtime.service_impl.maintenance import LongTermMemoryServiceMaintenanceMixin


def _source(event_id: str) -> LongTermSourceRefV1:
    return LongTermSourceRefV1(
        source_type="conversation_turn",
        event_ids=(event_id,),
        speaker="user",
        modality="voice",
    )


def _empty_reflection_result() -> LongTermReflectionResultV1:
    return LongTermReflectionResultV1(
        reflected_objects=(),
        created_summaries=(),
        midterm_packets=(),
    )


class _QueryOnlyObjectStore:
    def __init__(
        self,
        *,
        reflection_objects: tuple[LongTermMemoryObjectV1, ...] = (),
        sensor_objects: tuple[LongTermMemoryObjectV1, ...] = (),
        restart_objects: tuple[LongTermMemoryObjectV1, ...] = (),
    ) -> None:
        self.reflection_objects = reflection_objects
        self.sensor_objects = sensor_objects
        self.restart_objects = restart_objects
        self.queries: list[tuple[str, int]] = []
        self.applied_reflections: list[LongTermReflectionResultV1] = []

    def select_fast_topic_objects(self, *, query_text=None, limit=0):
        normalized_query = str(query_text or "")
        self.queries.append((normalized_query, int(limit)))
        if "confirmed_by_user confirmed active fact preference relationship" in normalized_query:
            return self.restart_objects
        if "event appointment reminder calendar visit meeting active" in normalized_query:
            return self.restart_objects
        if "summary_type thread memory_domain thread thread_summary active" in normalized_query:
            return self.restart_objects
        if "memory_domain sensor_routine routine_type interaction" in normalized_query and "confirmed_by_user" in normalized_query:
            return self.restart_objects
        if "pattern_type presence interaction daypart event_names" in normalized_query:
            return self.sensor_objects
        if "memory_domain sensor_routine routine_type presence interaction" in normalized_query:
            return self.sensor_objects
        if "environment_reflection" in normalized_query:
            return self.reflection_objects
        if "memory_domain smart_home_environment summary_type environment_day_profile" in normalized_query:
            return self.sensor_objects
        if "person_ref person_name relation relationship" in normalized_query:
            return self.reflection_objects
        if "event appointment reminder calendar visit meeting trip follow_up" in normalized_query:
            return self.reflection_objects
        return ()

    def load_objects(self):
        raise AssertionError("Live-near runtime selectors must not hydrate the full object snapshot.")

    def apply_reflection(self, result: LongTermReflectionResultV1) -> None:
        self.applied_reflections.append(result)


class _RecordingReflector:
    def __init__(self) -> None:
        self.calls: list[tuple[LongTermMemoryObjectV1, ...]] = []

    def reflect(
        self,
        *,
        objects: tuple[LongTermMemoryObjectV1, ...],
        include_midterm: bool = True,
    ) -> LongTermReflectionResultV1:
        del include_midterm
        self.calls.append(tuple(objects))
        return _empty_reflection_result()


class _RecordingSensorMemory:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def compile(
        self,
        *,
        objects: tuple[LongTermMemoryObjectV1, ...],
        now: datetime | None = None,
    ) -> LongTermReflectionResultV1:
        self.calls.append({"objects": tuple(objects), "now": now})
        return _empty_reflection_result()


class _FakeMidtermStore:
    def __init__(self) -> None:
        self.applied_reflections: list[LongTermReflectionResultV1] = []
        self.replace_calls: list[dict[str, object]] = []

    def apply_reflection(self, result: LongTermReflectionResultV1) -> None:
        self.applied_reflections.append(result)

    def replace_packets_with_attribute(
        self,
        *,
        packets,
        attribute_key: str,
        attribute_value: str,
    ) -> None:
        self.replace_calls.append(
            {
                "packets": tuple(packets),
                "attribute_key": attribute_key,
                "attribute_value": attribute_value,
            }
        )


class _RecordingRestartRecallCompiler:
    def __init__(self) -> None:
        self.calls: list[tuple[LongTermMemoryObjectV1, ...]] = []

    def build_packets(
        self,
        *,
        objects,
    ) -> tuple[str, ...]:
        object_tuple = tuple(objects)
        self.calls.append(object_tuple)
        return ("restart-packet",)


class _MaintenanceHarness(LongTermMemoryServiceMaintenanceMixin):
    def __init__(self, *, object_store: _QueryOnlyObjectStore) -> None:
        self._store_lock = threading.RLock()
        self.object_store = object_store
        self.reflector = _RecordingReflector()
        self.sensor_memory = _RecordingSensorMemory()
        self.midterm_store = _FakeMidtermStore()
        self.personality_learning = None
        self.config = SimpleNamespace(local_timezone_name="Europe/Berlin")

    @staticmethod
    def _empty_reflection_result() -> LongTermReflectionResultV1:
        return _empty_reflection_result()

    @staticmethod
    def _has_reflection_payload(result: LongTermReflectionResultV1) -> bool:
        return bool(result.reflected_objects or result.created_summaries or result.midterm_packets)


class _LifecycleHarness(LongTermMemoryServiceLifecycleMixin):
    def __init__(self, *, object_store: _QueryOnlyObjectStore) -> None:
        self.object_store = object_store
        self.restart_recall_policy_compiler = _RecordingRestartRecallCompiler()
        self.midterm_store = _FakeMidtermStore()


def _fact(
    memory_id: str,
    *,
    summary: str,
    attributes: dict[str, object] | None = None,
) -> LongTermMemoryObjectV1:
    return LongTermMemoryObjectV1(
        memory_id=memory_id,
        kind="fact",
        summary=summary,
        source=_source(f"turn:{memory_id}"),
        status="active",
        attributes=attributes or {},
    )


def _pattern(memory_id: str, *, summary: str) -> LongTermMemoryObjectV1:
    return LongTermMemoryObjectV1(
        memory_id=memory_id,
        kind="pattern",
        summary=summary,
        source=_source(f"turn:{memory_id}"),
        status="active",
        attributes={
            "memory_domain": "smart_home_environment",
            "pattern_type": "presence",
            "daypart": "morning",
        },
    )


def _environment_profile(memory_id: str, *, summary: str) -> LongTermMemoryObjectV1:
    return LongTermMemoryObjectV1(
        memory_id=memory_id,
        kind="summary",
        summary=summary,
        source=_source(f"turn:{memory_id}"),
        status="active",
        attributes={
            "memory_domain": "smart_home_environment",
            "summary_type": "environment_day_profile",
            "environment_id": "home:main",
            "date": "2026-03-29",
        },
    )


class LongTermRuntimeQuerySelectorTests(unittest.TestCase):
    def test_run_reflection_uses_targeted_queries_without_full_snapshot_reads(self) -> None:
        relationship = _fact(
            "fact:janina_wife",
            summary="Janina is the user's wife.",
            attributes={"person_ref": "person:janina", "relation": "wife", "support_count": 3},
        )
        environment_profile = _environment_profile(
            "environment_profile:home_main:day:2026-03-29",
            summary="Home activity profile for the day.",
        )
        raw_pattern = _pattern(
            "pattern:presence:2026-03-29:morning",
            summary="Presence was observed in the kitchen this morning.",
        )
        object_store = _QueryOnlyObjectStore(
            reflection_objects=(relationship, environment_profile),
            sensor_objects=(raw_pattern, environment_profile),
        )
        service = _MaintenanceHarness(object_store=object_store)

        result = service.run_reflection()

        self.assertFalse(result.reflected_objects)
        self.assertEqual(
            {item.memory_id for item in service.reflector.calls[0]},
            {"fact:janina_wife", "environment_profile:home_main:day:2026-03-29"},
        )
        self.assertEqual(
            {item.memory_id for item in service.sensor_memory.calls[0]["objects"]},
            {
                "pattern:presence:2026-03-29:morning",
                "environment_profile:home_main:day:2026-03-29",
            },
        )
        self.assertGreaterEqual(len(object_store.queries), 6)

    def test_run_sensor_memory_uses_targeted_queries_without_full_snapshot_reads(self) -> None:
        raw_pattern = _pattern(
            "pattern:presence:2026-03-29:afternoon",
            summary="Presence was observed in the living room this afternoon.",
        )
        routine = _fact(
            "routine:presence:weekday:afternoon",
            summary="The user is often active in the afternoon.",
            attributes={"memory_domain": "sensor_routine", "routine_type": "presence"},
        )
        object_store = _QueryOnlyObjectStore(sensor_objects=(raw_pattern, routine))
        service = _MaintenanceHarness(object_store=object_store)

        service.run_sensor_memory(now=datetime(2026, 3, 29, 14, 30, tzinfo=timezone.utc))

        self.assertEqual(
            {item.memory_id for item in service.sensor_memory.calls[0]["objects"]},
            {"pattern:presence:2026-03-29:afternoon", "routine:presence:weekday:afternoon"},
        )
        self.assertIsNotNone(service.sensor_memory.calls[0]["now"])

    def test_restart_recall_refresh_uses_targeted_queries_without_full_snapshot_reads(self) -> None:
        stable_fact = _fact(
            "fact:favorite_bread",
            summary="The user prefers rye bread.",
            attributes={"support_count": 3},
        )
        stable_event = LongTermMemoryObjectV1(
            memory_id="event:doctor_visit",
            kind="event",
            summary="The user has a doctor visit tomorrow.",
            source=_source("turn:event:doctor_visit"),
            status="active",
            confirmed_by_user=True,
        )
        object_store = _QueryOnlyObjectStore(restart_objects=(stable_fact, stable_event))
        service = _LifecycleHarness(object_store=object_store)

        service._refresh_restart_recall_packets_locked()

        self.assertEqual(
            {item.memory_id for item in service.restart_recall_policy_compiler.calls[0]},
            {"fact:favorite_bread", "event:doctor_visit"},
        )
        self.assertEqual(
            service.midterm_store.replace_calls,
            [
                {
                    "packets": ("restart-packet",),
                    "attribute_key": "persistence_scope",
                    "attribute_value": "restart_recall",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
