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
from twinr.memory.longterm.runtime.live_object_selectors import (
    select_reflection_neighborhood_objects,
    select_sensor_memory_neighborhood_objects,
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
        projection_objects: tuple[LongTermMemoryObjectV1, ...] = (),
        event_objects: tuple[LongTermMemoryObjectV1, ...] = (),
    ) -> None:
        self.reflection_objects = reflection_objects
        self.sensor_objects = sensor_objects
        self.restart_objects = restart_objects
        self.projection_objects = projection_objects
        self.event_objects = event_objects
        self.queries: list[tuple[str, int]] = []
        self.applied_reflections: list[LongTermReflectionResultV1] = []
        self.event_allow_cold_calls: list[bool] = []
        self.projection_allow_cold_calls: list[bool] = []

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

    def load_objects_by_projection_filter(
        self,
        *,
        predicate,
        allow_cold_remote_catalog_scan: bool = True,
    ):
        self.projection_allow_cold_calls.append(bool(allow_cold_remote_catalog_scan))
        selected: list[LongTermMemoryObjectV1] = []
        for item in self.projection_objects:
            payload = item.to_payload()
            projection = {
                "memory_id": payload.get("memory_id"),
                "kind": payload.get("kind"),
                "summary": payload.get("summary"),
                "details": payload.get("details"),
                "status": payload.get("status"),
                "value_key": payload.get("value_key"),
                "attributes": dict(payload.get("attributes") or {}),
            }
            source = payload.get("source")
            if isinstance(source, dict) and isinstance(source.get("event_ids"), list):
                projection["source_event_ids"] = list(source["event_ids"])
            if predicate(projection):
                selected.append(item)
        return tuple(selected)

    def load_objects_by_event_ids(
        self,
        event_ids,
        *,
        allow_cold_remote_catalog_scan: bool = True,
    ):
        self.event_allow_cold_calls.append(bool(allow_cold_remote_catalog_scan))
        target_ids = {str(item).strip() for item in event_ids if str(item).strip()}
        return tuple(
            item
            for item in self.event_objects
            if target_ids.intersection(item.source.event_ids)
        )

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
    def test_reflection_neighborhood_selector_prefers_touched_person_and_event_neighbors(self) -> None:
        seed = _fact(
            "fact:janina_wife",
            summary="Janina is the user's wife.",
            attributes={"person_ref": "person:janina", "relation": "wife", "support_count": 3},
        )
        thread_summary = LongTermMemoryObjectV1(
            memory_id="thread:person_janina",
            kind="summary",
            summary="Ongoing thread about Janina.",
            source=_source("turn:thread:janina"),
            status="active",
            attributes={"summary_type": "thread", "person_ref": "person:janina"},
            value_key="person:janina",
        )
        same_event = LongTermMemoryObjectV1(
            memory_id="event:janina_eye_doctor",
            kind="event",
            summary="Janina has an eye-doctor appointment.",
            source=LongTermSourceRefV1(
                source_type="conversation_turn",
                event_ids=seed.source.event_ids,
            ),
            status="active",
            attributes={"memory_domain": "planning"},
        )
        unrelated = _fact(
            "fact:mario_friend",
            summary="Mario is a family friend.",
            attributes={"person_ref": "person:mario", "support_count": 2},
        )
        object_store = _QueryOnlyObjectStore(
            projection_objects=(thread_summary, unrelated),
            event_objects=(same_event,),
        )

        selected = select_reflection_neighborhood_objects(
            object_store,
            seed_objects=(seed,),
        )

        self.assertEqual(
            {item.memory_id for item in selected},
            {"fact:janina_wife", "thread:person_janina", "event:janina_eye_doctor"},
        )

    def test_reflection_neighborhood_selector_does_not_fall_back_to_broad_compile_union(self) -> None:
        seed = _fact(
            "fact:bread_preference",
            summary="The user prefers rye bread.",
            attributes={"preference_type": "food"},
        )
        broad_union_only = _fact(
            "fact:janina_wife",
            summary="Janina is the user's wife.",
            attributes={"person_ref": "person:janina", "relation": "wife", "support_count": 3},
        )
        object_store = _QueryOnlyObjectStore(reflection_objects=(broad_union_only,))

        selected = select_reflection_neighborhood_objects(
            object_store,
            seed_objects=(seed,),
        )

        self.assertEqual({item.memory_id for item in selected}, {"fact:bread_preference"})
        self.assertEqual(object_store.event_allow_cold_calls, [False])
        self.assertEqual(object_store.projection_allow_cold_calls, [False])

    def test_sensor_neighborhood_selector_prefers_touched_domain_neighbors(self) -> None:
        seed_pattern = _pattern(
            "pattern:presence:2026-03-29:morning",
            summary="Presence was observed in the kitchen this morning.",
        )
        environment_profile = _environment_profile(
            "environment_profile:home_main:day:2026-03-29",
            summary="Home activity profile for the day.",
        )
        routine = _fact(
            "routine:presence:weekday:morning",
            summary="The user is usually active in the morning.",
            attributes={"memory_domain": "sensor_routine", "routine_type": "presence"},
        )
        unrelated = _fact(
            "fact:favorite_bread",
            summary="The user prefers rye bread.",
            attributes={"support_count": 3},
        )
        object_store = _QueryOnlyObjectStore(
            projection_objects=(environment_profile, routine, unrelated),
        )

        selected = select_sensor_memory_neighborhood_objects(
            object_store,
            seed_objects=(seed_pattern,),
        )

        self.assertEqual(
            {item.memory_id for item in selected},
            {
                "pattern:presence:2026-03-29:morning",
                "environment_profile:home_main:day:2026-03-29",
                "routine:presence:weekday:morning",
            },
        )

    def test_sensor_neighborhood_selector_does_not_fall_back_to_broad_compile_union(self) -> None:
        seed_pattern = _pattern(
            "pattern:presence:2026-03-29:morning",
            summary="Presence was observed in the kitchen this morning.",
        )
        broad_union_only = _fact(
            "routine:interaction:weekday:morning",
            summary="Voice interactions usually happen in the morning.",
            attributes={"memory_domain": "sensor_routine", "routine_type": "interaction"},
        )
        object_store = _QueryOnlyObjectStore(sensor_objects=(broad_union_only,))

        selected = select_sensor_memory_neighborhood_objects(
            object_store,
            seed_objects=(seed_pattern,),
        )

        self.assertEqual(
            {item.memory_id for item in selected},
            {"pattern:presence:2026-03-29:morning"},
        )
        self.assertEqual(object_store.event_allow_cold_calls, [False])
        self.assertEqual(object_store.projection_allow_cold_calls, [False])

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
