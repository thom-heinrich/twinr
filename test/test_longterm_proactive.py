from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
from threading import Thread
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test.longterm_test_program import make_test_extractor
from twinr.agent.base_agent import TwinrConfig
from twinr.memory.longterm import (
    LongTermConsolidationResultV1,
    LongTermMemoryObjectV1,
    LongTermMemoryService,
    LongTermSourceRefV1,
)
from twinr.memory.longterm.runtime.service_impl.proactive import _select_proactive_planner_objects
from twinr.memory.longterm.storage.remote_state import LongTermRemoteReadFailedError
from twinr.memory.longterm.proactive.state import _write_json_atomic


def _source(event_id: str) -> LongTermSourceRefV1:
    return LongTermSourceRefV1(
        source_type="conversation_turn",
        event_ids=(event_id,),
        speaker="user",
        modality="voice",
    )


def _config(root: str, **overrides) -> TwinrConfig:
    return TwinrConfig(
        project_root=root,
        personality_dir="personality",
        memory_markdown_path=str(Path(root) / "state" / "MEMORY.md"),
        long_term_memory_enabled=True,
        long_term_memory_path=str(Path(root) / "state" / "chonkydb"),
        long_term_memory_proactive_enabled=True,
        long_term_memory_proactive_poll_interval_s=0.0,
        long_term_memory_proactive_min_confidence=0.7,
        long_term_memory_proactive_repeat_cooldown_s=3600.0,
        long_term_memory_proactive_skip_cooldown_s=600.0,
        long_term_memory_proactive_reservation_ttl_s=90.0,
        **overrides,
    )


class LongTermProactiveIntegrationTests(unittest.TestCase):
    def test_proactive_planner_selection_uses_targeted_queries_without_full_snapshot_reads(self) -> None:
        event_object = LongTermMemoryObjectV1(
            memory_id="event:doctor",
            kind="event",
            summary="Doctor appointment tomorrow.",
            source=_source("turn:event"),
            status="active",
        )
        summary_object = LongTermMemoryObjectV1(
            memory_id="thread:walk_weather",
            kind="thread_summary",
            summary="Ongoing thread about the user's walk plans.",
            source=_source("turn:thread"),
            status="active",
            attributes={"support_count": 3},
        )
        routine_object = LongTermMemoryObjectV1(
            memory_id="routine:print_offer",
            kind="fact",
            summary="The user often wants a printed agenda after breakfast.",
            source=_source("turn:routine"),
            status="active",
            attributes={"memory_domain": "sensor_routine", "routine_type": "interaction"},
        )

        class QueryOnlyObjectStore:
            def __init__(self) -> None:
                self.queries: list[tuple[str | None, int]] = []

            def select_fast_topic_objects(self, *, query_text=None, limit=0):
                self.queries.append((query_text, limit))
                normalized_query = str(query_text or "")
                if "event" in normalized_query and "plan" in normalized_query:
                    return (event_object,)
                if "thread_summary" in normalized_query and "summary_type" in normalized_query:
                    return (summary_object,)
                if "sensor_routine" in normalized_query and "interaction" in normalized_query:
                    return (routine_object,)
                return ()

            def load_objects(self):
                raise AssertionError("Proactive planning must not hydrate the full object snapshot.")

        object_store = QueryOnlyObjectStore()

        selected = _select_proactive_planner_objects(object_store)

        self.assertEqual(
            {item.memory_id for item in selected},
            {"event:doctor", "thread:walk_weather", "routine:print_offer"},
        )
        self.assertEqual(len(object_store.queries), 3)

    def test_preview_proactive_candidate_skips_fast_topic_read_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = LongTermMemoryService.from_config(_config(temp_dir))

            def _fail_fast_topic(**_kwargs):
                raise LongTermRemoteReadFailedError(
                    "Required remote long-term fast-topic retrieval failed.",
                    details={
                        "operation": "fast_topic_topk_search",
                        "classification": "client_contract_error",
                        "status_code": 400,
                    },
                )

            service.object_store.select_fast_topic_objects = _fail_fast_topic  # type: ignore[method-assign]

            candidate = service.preview_proactive_candidate(
                now=datetime(2026, 3, 14, 9, 0, tzinfo=timezone.utc)
            )
            service.shutdown()

        self.assertIsNone(candidate)

    def test_plan_proactive_candidates_skips_fast_topic_read_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = LongTermMemoryService.from_config(_config(temp_dir))

            def _fail_fast_topic(**_kwargs):
                raise LongTermRemoteReadFailedError(
                    "Required remote long-term fast-topic retrieval failed.",
                    details={
                        "operation": "fast_topic_topk_search",
                        "classification": "client_contract_error",
                        "status_code": 400,
                    },
                )

            service.object_store.select_fast_topic_objects = _fail_fast_topic  # type: ignore[method-assign]

            plan = service.plan_proactive_candidates(
                now=datetime(2026, 3, 14, 9, 0, tzinfo=timezone.utc)
            )
            service.shutdown()

        self.assertEqual(plan.candidates, ())

    def test_proactive_state_atomic_write_survives_concurrent_writers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "proactive.json"
            errors: list[BaseException] = []

            def worker(index: int) -> None:
                try:
                    _write_json_atomic(target, {"writer": index})
                except BaseException as exc:
                    errors.append(exc)

            threads = [Thread(target=worker, args=(index,)) for index in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual(errors, [])
            payload = json.loads(target.read_text(encoding="utf-8"))

        self.assertIn(payload["writer"], range(8))

    def test_service_reserves_and_cools_down_same_candidate_after_delivery(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = LongTermMemoryService.from_config(
                _config(temp_dir, long_term_memory_proactive_allow_sensitive=True),
                extractor=make_test_extractor(),
            )
            service.object_store.apply_consolidation(
                service.consolidator.consolidate(
                    extraction=service.extractor.extract_conversation_turn(
                        transcript="My wife Janina is at the eye doctor today.",
                        response="I hope Janina's appointment goes smoothly.",
                        occurred_at=datetime(2026, 3, 14, 8, 0, tzinfo=timezone.utc),
                    )
                )
            )

            reservation = service.reserve_proactive_candidate(
                now=datetime(2026, 3, 14, 8, 5, tzinfo=timezone.utc)
            )
            self.assertIsNotNone(reservation)
            service.mark_proactive_candidate_delivered(
                reservation,
                delivered_at=datetime(2026, 3, 14, 8, 6, tzinfo=timezone.utc),
                prompt_text="I hope Janina's appointment goes smoothly today.",
            )
            second = service.reserve_proactive_candidate(
                now=datetime(2026, 3, 14, 8, 20, tzinfo=timezone.utc)
            )
            history = service.proactive_policy.state_store.load_entries()
            service.shutdown()

        self.assertIsNone(second)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].delivery_count, 1)
        self.assertEqual(history[0].last_prompt_text, "I hope Janina's appointment goes smoothly today.")

    def test_sensitive_candidates_require_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _config(temp_dir, long_term_memory_proactive_allow_sensitive=False)
            service = LongTermMemoryService.from_config(config, extractor=make_test_extractor())
            service.object_store.apply_consolidation(
                service.consolidator.consolidate(
                    extraction=service.extractor.extract_conversation_turn(
                        transcript="My wife Janina is at the eye doctor today.",
                        response="I hope Janina's appointment goes smoothly.",
                        occurred_at=datetime(2026, 3, 14, 8, 0, tzinfo=timezone.utc),
                    )
                )
            )

            reservation = service.reserve_proactive_candidate(
                now=datetime(2026, 3, 14, 8, 5, tzinfo=timezone.utc)
            )
            service.shutdown()

        self.assertIsNone(reservation)

    def test_non_sensitive_thread_summary_can_be_reserved_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = LongTermMemoryService.from_config(_config(temp_dir))
            service.object_store.apply_consolidation(
                LongTermConsolidationResultV1(
                    turn_id="turn:thread",
                    occurred_at=datetime(2026, 3, 14, 8, 0, tzinfo=timezone.utc),
                    episodic_objects=(),
                    durable_objects=(
                        LongTermMemoryObjectV1(
                            memory_id="thread:walk_weather",
                            kind="thread_summary",
                            summary="Ongoing thread about the user's plan to walk if the weather is nice.",
                            details="Reflected from multiple related turns.",
                            source=_source("turn:thread"),
                            status="active",
                            confidence=0.82,
                            sensitivity="normal",
                            slot_key="thread:user:main:walk_weather",
                            value_key="walk_weather",
                            attributes={"support_count": 4},
                        ),
                    ),
                    deferred_objects=(),
                    conflicts=(),
                    graph_edges=(),
                )
            )

            reservation = service.reserve_proactive_candidate(
                now=datetime(2026, 3, 14, 9, 0, tzinfo=timezone.utc)
            )
            service.shutdown()

        self.assertIsNotNone(reservation)
        self.assertEqual(reservation.candidate.kind, "gentle_follow_up")

    def test_recent_past_event_generates_gentle_follow_up_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = LongTermMemoryService.from_config(_config(temp_dir))
            service.object_store.apply_consolidation(
                LongTermConsolidationResultV1(
                    turn_id="turn:doctor",
                    occurred_at=datetime(2026, 3, 15, 8, 0, tzinfo=timezone.utc),
                    episodic_objects=(),
                    durable_objects=(
                        LongTermMemoryObjectV1(
                            memory_id="event:doctor_yesterday",
                            kind="event",
                            summary="The user had a doctor appointment yesterday.",
                            details="A recent appointment may deserve a soft follow-up about how it went.",
                            source=_source("turn:doctor"),
                            status="active",
                            confidence=0.79,
                            sensitivity="normal",
                            valid_from="2026-03-14",
                            valid_to="2026-03-14",
                            slot_key="event:user:doctor:2026-03-14",
                            value_key="event:doctor_yesterday",
                        ),
                    ),
                    deferred_objects=(),
                    conflicts=(),
                    graph_edges=(),
                )
            )

            plan = service.plan_proactive_candidates(
                now=datetime(2026, 3, 15, 9, 0, tzinfo=timezone.utc)
            )
            service.shutdown()

        follow_up_candidates = [item for item in plan.candidates if item.kind == "gentle_follow_up"]
        self.assertEqual(len(follow_up_candidates), 1)
        self.assertIn("doctor appointment yesterday", follow_up_candidates[0].summary.lower())

    def test_skip_history_temporarily_suppresses_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = LongTermMemoryService.from_config(_config(temp_dir))
            service.object_store.apply_consolidation(
                LongTermConsolidationResultV1(
                    turn_id="turn:thread",
                    occurred_at=datetime(2026, 3, 14, 8, 0, tzinfo=timezone.utc),
                    episodic_objects=(),
                    durable_objects=(
                        LongTermMemoryObjectV1(
                            memory_id="thread:walk_weather",
                            kind="thread_summary",
                            summary="Ongoing thread about the user's plan to walk if the weather is nice.",
                            details="Reflected from multiple related turns.",
                            source=_source("turn:thread"),
                            status="active",
                            confidence=0.82,
                            sensitivity="normal",
                            slot_key="thread:user:main:walk_weather",
                            value_key="walk_weather",
                            attributes={"support_count": 4},
                        ),
                    ),
                    deferred_objects=(),
                    conflicts=(),
                    graph_edges=(),
                )
            )

            reservation = service.reserve_proactive_candidate(
                now=datetime(2026, 3, 14, 9, 0, tzinfo=timezone.utc)
            )
            self.assertIsNotNone(reservation)
            service.mark_proactive_candidate_skipped(
                reservation,
                reason="delivery_failed: speaker busy",
                skipped_at=datetime(2026, 3, 14, 9, 1, tzinfo=timezone.utc),
            )
            second = service.reserve_proactive_candidate(
                now=datetime(2026, 3, 14, 9, 5, tzinfo=timezone.utc)
            )
            third = service.reserve_proactive_candidate(
                now=datetime(2026, 3, 14, 9, 20, tzinfo=timezone.utc)
            )
            service.shutdown()

        self.assertIsNone(second)
        self.assertIsNotNone(third)


if __name__ == "__main__":
    unittest.main()
