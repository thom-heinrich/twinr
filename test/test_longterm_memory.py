"""Regression and integration tests for the long-term memory runtime service."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.contracts import AgentToolCall, AgentToolResult
from test.longterm_test_program import make_test_extractor
from twinr.config import TwinrConfig
from twinr.memory.chonkydb import TwinrPersonalGraphStore
from twinr.memory.context_store import PromptContextStore
from twinr.memory.longterm import (
    LongTermConsolidationResultV1,
    LongTermConversationTurn,
    LongTermMemoryConflictV1,
    LongTermMemoryContext,
    LongTermMemoryObjectV1,
    LongTermMemoryReflector,
    LongTermReflectionResultV1,
    LongTermMemoryService,
    LongTermMidtermStore,
    LongTermSourceRefV1,
    LongTermStructuredStore,
)
from twinr.memory.longterm.runtime.worker import AsyncLongTermMemoryWriter, AsyncLongTermWriterState
from twinr.memory.query_normalization import LongTermQueryProfile


class _StaticQueryRewriter:
    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = mapping

    def profile(self, query_text: str | None) -> LongTermQueryProfile:
        canonical = self._mapping.get(str(query_text or ""))
        return LongTermQueryProfile.from_text(query_text, canonical_english_text=canonical)


class _StubReflectionProgram:
    def compile_reflection(
        self,
        *,
        objects: tuple[LongTermMemoryObjectV1, ...],
        timezone_name: str,
        packet_limit: int,
    ):
        del timezone_name
        del packet_limit
        if any("eye laser treatment" in item.summary.lower() for item in objects):
            return {
                "midterm_packets": [
                    {
                        "packet_id": "midterm:janina_today",
                        "kind": "recent_life_bundle",
                        "summary": "Janina has eye laser treatment today.",
                        "details": "This is near-term context for follow-up questions about Janina.",
                        "source_memory_ids": [item.memory_id for item in objects if "eye laser treatment" in item.summary.lower()],
                        "query_hints": ["janina", "today", "eye laser treatment"],
                        "sensitivity": "sensitive",
                        "valid_from": "2026-03-15",
                        "valid_to": "2026-03-15",
                        "attributes": {"scope": "recent_window"},
                    }
                ]
            }
        return {"midterm_packets": []}


class _FailingReflectionProgram:
    def compile_reflection(
        self,
        *,
        objects: tuple[LongTermMemoryObjectV1, ...],
        timezone_name: str,
        packet_limit: int,
    ):
        del objects
        del timezone_name
        del packet_limit
        raise RuntimeError("reflection compiler failed")


class _BudgetRecordingWriter:
    def __init__(
        self,
        *,
        worker_name: str,
        pending_count: int,
        last_error_message: str | None = None,
        flush_result: bool = True,
        sleep_s: float = 0.0,
    ) -> None:
        self.recorded_timeouts: list[float] = []
        self._state = AsyncLongTermWriterState(
            worker_name=worker_name,
            pending_count=pending_count,
            inflight_count=0,
            dropped_count=0,
            last_error_message=last_error_message,
            accepting=True,
            worker_alive=True,
        )
        self._flush_result = flush_result
        self._sleep_s = max(0.0, float(sleep_s))

    def snapshot_state(self) -> AsyncLongTermWriterState:
        return self._state

    def flush(self, *, timeout_s: float) -> bool:
        self.recorded_timeouts.append(timeout_s)
        if self._sleep_s > 0.0:
            time.sleep(self._sleep_s)
        return self._flush_result


class _CountingRemoteState:
    def __init__(self) -> None:
        self.enter_count = 0

    @contextmanager
    def cache_probe_reads(self):
        self.enter_count += 1
        yield


class _ProbeCachingRetriever:
    def __init__(self, remote_state: _CountingRemoteState) -> None:
        self.object_store = type("ObjectStoreStub", (), {"remote_state": remote_state})()

    def build_context(self, *, query: LongTermQueryProfile, original_query_text: str | None = None) -> LongTermMemoryContext:
        del query
        del original_query_text
        return LongTermMemoryContext(episodic_context="remembered")

    def select_conflict_queue(self, *, query: LongTermQueryProfile, limit: int | None = None):
        del query
        del limit
        return ()

    def select_durable_objects(self, *, query: LongTermQueryProfile, limit: int | None = None):
        del query
        del limit
        return ()

    def _render_durable_context(self, durable_objects):
        del durable_objects
        return None


class _PrewarmObjectStore:
    def __init__(self) -> None:
        self.remote_state = _CountingRemoteState()
        self.calls: list[tuple[str, object]] = []
        self.catalog_calls: list[str] = []
        self._remote_catalog = SimpleNamespace(
            load_catalog_entries=lambda *, snapshot_kind: self.catalog_calls.append(snapshot_kind) or ()
        )

    def select_relevant_episodic_objects(self, *, query_text: str | None, limit: int, fallback_limit: int, require_query_match: bool):
        self.calls.append(("episodic", query_text, limit, fallback_limit, require_query_match))
        return ()

    def select_relevant_objects(self, *, query_text: str | None, limit: int):
        self.calls.append(("durable", query_text, limit))
        return ()

    def select_open_conflicts(self, *, query_text: str | None, limit: int):
        self.calls.append(("conflicts", query_text, limit))
        return ()


class _FailOnEnterLock:
    def __enter__(self):
        raise AssertionError("_store_lock must not gate foreground provider-context reads")

    def __exit__(self, exc_type, exc, tb):
        return False


class _RecordingPersonalityLearningService:
    def __init__(self) -> None:
        self.conversation_calls: list[tuple[LongTermConversationTurn, LongTermConsolidationResultV1]] = []
        self.tool_history_calls: list[tuple[tuple[AgentToolCall, ...], tuple[AgentToolResult, ...]]] = []

    def record_conversation_consolidation(
        self,
        *,
        turn: LongTermConversationTurn,
        consolidation: LongTermConsolidationResultV1,
    ) -> None:
        self.conversation_calls.append((turn, consolidation))

    def record_tool_history(
        self,
        *,
        tool_calls: tuple[AgentToolCall, ...],
        tool_results: tuple[AgentToolResult, ...],
    ) -> None:
        self.tool_history_calls.append((tool_calls, tool_results))


class _StubThreadSummaryReflector:
    def __init__(self, result: LongTermReflectionResultV1) -> None:
        self._result = result

    def reflect(
        self,
        *,
        objects: tuple[LongTermMemoryObjectV1, ...],
        include_midterm: bool = True,
    ) -> LongTermReflectionResultV1:
        del objects
        del include_midterm
        return self._result


class _BlockingExtractor:
    def __init__(self, delegate) -> None:
        self._delegate = delegate
        self.started = threading.Event()
        self.release = threading.Event()

    def extract_conversation_turn(self, **kwargs):
        self.started.set()
        self.release.wait(timeout=5.0)
        return self._delegate.extract_conversation_turn(**kwargs)


class LongTermMemoryServiceTests(unittest.TestCase):
    def _source(self, event_id: str = "turn:test") -> LongTermSourceRefV1:
        return LongTermSourceRefV1(
            source_type="conversation_turn",
            event_ids=(event_id,),
            speaker="user",
            modality="voice",
        )

    def _ops_entry(
        self,
        *,
        event: str,
        created_at: str,
        data: dict[str, object] | None = None,
    ) -> dict[str, object]:
        return {
            "event": event,
            "created_at": created_at,
            "data": dict(data or {}),
        }

    def test_background_worker_persists_episodic_turns_in_memory_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                long_term_memory_write_queue_size=4,
            )
            service = LongTermMemoryService.from_config(config, extractor=make_test_extractor())
            result = service.enqueue_conversation_turn(
                transcript="Wie wird das Wetter heute?",
                response="Heute ist es sonnig und mild.",
            )
            drained = service.flush(timeout_s=2.0)
            entries = service.prompt_context_store.memory_store.load_entries()
            stored_objects = service.object_store.load_objects()
            service.shutdown()

        self.assertIsNotNone(result)
        self.assertTrue(result.accepted)
        self.assertTrue(drained)
        self.assertEqual(entries[0].kind, "episodic_turn")
        self.assertIn('Conversation about "Wie wird das Wetter heute?"', entries[0].summary)
        self.assertIn('Twinr answered: "Heute ist es sonnig und mild."', entries[0].details or "")
        self.assertTrue(any(item.kind == "episode" for item in stored_objects))

    def test_background_worker_persists_immediate_midterm_packet_before_slow_extraction_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                long_term_memory_midterm_enabled=True,
                long_term_memory_reflection_compiler_enabled=False,
                long_term_memory_write_queue_size=4,
            )
            extractor = _BlockingExtractor(make_test_extractor())
            service = LongTermMemoryService.from_config(config, extractor=extractor)
            try:
                result = service.enqueue_conversation_turn(
                    transcript="Lea bringt mir heute Abend eine Thermoskanne mit Linsensuppe vorbei.",
                    response="Ich merke mir, dass Lea dir heute Abend die Thermoskanne mit Linsensuppe bringt.",
                )
                self.assertIsNotNone(result)
                assert result is not None
                self.assertTrue(result.accepted)
                self.assertTrue(extractor.started.wait(timeout=1.0))

                deadline = time.monotonic() + 2.0
                packets = ()
                while time.monotonic() < deadline:
                    packets = service.midterm_store.load_packets()
                    if packets:
                        break
                    time.sleep(0.02)

                self.assertFalse(service.flush(timeout_s=0.05))
                self.assertTrue(packets)
                self.assertEqual(packets[0].kind, "recent_turn_continuity")
                self.assertEqual(packets[0].attributes["persistence_scope"], "turn_continuity")
                self.assertIn("Thermoskanne", packets[0].details or "")
            finally:
                extractor.release.set()
                service.flush(timeout_s=2.0)
                service.shutdown()

    def test_build_provider_context_uses_remote_probe_cache_within_one_turn(self) -> None:
        remote_state = _CountingRemoteState()
        retriever = _ProbeCachingRetriever(remote_state)
        service = object.__new__(LongTermMemoryService)
        service.config = TwinrConfig(project_root=".", personality_dir="personality")
        service.query_rewriter = _StaticQueryRewriter({})
        service.retriever = retriever
        service.object_store = retriever.object_store
        service.prompt_context_store = SimpleNamespace(
            memory_store=SimpleNamespace(remote_state=None),
            user_store=SimpleNamespace(remote_state=None),
            personality_store=SimpleNamespace(remote_state=None),
        )
        service.graph_store = SimpleNamespace(remote_state=None)
        service.midterm_store = SimpleNamespace(remote_state=None)
        service._store_lock = threading.RLock()

        context = service.build_provider_context("na alles klar")

        self.assertEqual(context.episodic_context, "remembered")
        self.assertEqual(remote_state.enter_count, 1)

    def test_build_provider_context_does_not_wait_on_shared_store_lock(self) -> None:
        remote_state = _CountingRemoteState()
        retriever = _ProbeCachingRetriever(remote_state)
        service = object.__new__(LongTermMemoryService)
        service.config = TwinrConfig(project_root=".", personality_dir="personality")
        service.query_rewriter = _StaticQueryRewriter({})
        service.retriever = retriever
        service.object_store = retriever.object_store
        service.prompt_context_store = SimpleNamespace(
            memory_store=SimpleNamespace(remote_state=None),
            user_store=SimpleNamespace(remote_state=None),
            personality_store=SimpleNamespace(remote_state=None),
        )
        service.graph_store = SimpleNamespace(remote_state=None)
        service.midterm_store = SimpleNamespace(remote_state=None)
        service.writer = None
        service.multimodal_writer = None
        service._store_lock = _FailOnEnterLock()

        context = service.build_provider_context("na alles klar")

        self.assertEqual(context.episodic_context, "remembered")
        self.assertEqual(remote_state.enter_count, 1)

    def test_build_tool_provider_context_does_not_wait_on_shared_store_lock(self) -> None:
        remote_state = _CountingRemoteState()
        retriever = _ProbeCachingRetriever(remote_state)
        service = object.__new__(LongTermMemoryService)
        service.config = TwinrConfig(project_root=".", personality_dir="personality")
        service.query_rewriter = _StaticQueryRewriter({})
        service.retriever = retriever
        service.object_store = retriever.object_store
        service.prompt_context_store = SimpleNamespace(
            memory_store=SimpleNamespace(remote_state=None),
            user_store=SimpleNamespace(remote_state=None),
            personality_store=SimpleNamespace(remote_state=None),
        )
        service.graph_store = SimpleNamespace(
            remote_state=None,
            build_prompt_context=lambda *_args, **_kwargs: "graph",
        )
        service.midterm_store = SimpleNamespace(remote_state=None)
        service.writer = None
        service.multimodal_writer = None
        service._store_lock = _FailOnEnterLock()

        context = service.build_tool_provider_context("na alles klar")

        self.assertEqual(context.episodic_context, "remembered")
        self.assertEqual(context.graph_context, "graph")
        self.assertEqual(remote_state.enter_count, 1)

    def test_prewarm_foreground_read_cache_primes_lightweight_remote_indexes(self) -> None:
        object_store = _PrewarmObjectStore()
        graph_calls: list[str] = []
        midterm_calls: list[str] = []
        service = object.__new__(LongTermMemoryService)
        service.config = TwinrConfig(project_root=".", personality_dir="personality", long_term_memory_recall_limit=3)
        service.object_store = object_store
        service.prompt_context_store = SimpleNamespace(
            memory_store=SimpleNamespace(remote_state=None),
            user_store=SimpleNamespace(remote_state=None),
            personality_store=SimpleNamespace(remote_state=None),
        )
        service.graph_store = SimpleNamespace(remote_state=None, load_document=lambda: graph_calls.append("graph"))
        service.midterm_store = SimpleNamespace(remote_state=None, load_packets=lambda: midterm_calls.append("midterm"))
        service._store_lock = threading.RLock()

        service.prewarm_foreground_read_cache()

        self.assertEqual(object_store.remote_state.enter_count, 1)
        self.assertEqual(object_store.calls, [])
        self.assertEqual(object_store.catalog_calls, ["objects", "conflicts"])
        self.assertEqual(graph_calls, ["graph"])
        self.assertEqual(midterm_calls, ["midterm"])

    def test_background_worker_persists_extracted_graph_edges(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                user_display_name="Erika",
            )
            service = LongTermMemoryService.from_config(config, extractor=make_test_extractor())
            service.enqueue_conversation_turn(
                transcript=(
                    "Today is a beautiful Sunday, it is really warm. "
                    "My wife Janina is at the eye doctor and is getting eye laser treatment."
                ),
                response="I hope Janina's appointment goes smoothly.",
            )
            drained = service.flush(timeout_s=2.0)
            graph = service.graph_store.load_document()
            service.shutdown()

        edge_types = {edge.edge_type for edge in graph.edges}
        node_ids = {node.node_id for node in graph.nodes}

        self.assertTrue(drained)
        self.assertIn("social_related_to_user", edge_types)
        self.assertIn("temporal_occurs_on", edge_types)
        self.assertIn("user:main", node_ids)
        self.assertIn("person:janina", node_ids)

    def test_service_flush_continues_when_background_reflection_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                long_term_memory_write_queue_size=4,
            )
            service = LongTermMemoryService.from_config(config, extractor=make_test_extractor())
            if service.writer is not None:
                service.writer.shutdown(timeout_s=1.0)
            service.writer = AsyncLongTermMemoryWriter(
                write_callback=lambda item: LongTermMemoryService._persist_longterm_turn(
                    config=config,
                    store=service.prompt_context_store,
                    graph_store=service.graph_store,
                    object_store=service.object_store,
                    midterm_store=service.midterm_store,
                    extractor=service.extractor,
                    consolidator=service.consolidator,
                    reflector=LongTermMemoryReflector(program=_FailingReflectionProgram()),
                    sensor_memory=service.sensor_memory,
                    retention_policy=service.retention_policy,
                    item=item,
                ),
                max_queue_size=4,
                poll_interval_s=0.01,
            )
            try:
                result = service.enqueue_conversation_turn(
                    transcript="Bitte merk dir etwas zu Janina.",
                    response="Ich habe den Kontext aufgenommen.",
                )
                drained = service.flush(timeout_s=1.0)
                error_message = None if service.writer is None else service.writer.last_error_message
            finally:
                service.shutdown(timeout_s=1.0)

        self.assertIsNotNone(result)
        self.assertTrue(result.accepted)
        self.assertTrue(drained)
        self.assertIsNone(error_message)

    def test_service_flush_clamps_later_writers_to_the_remaining_total_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
            )
            service = LongTermMemoryService.from_config(config, extractor=make_test_extractor())
            if service.writer is not None:
                service.writer.shutdown(timeout_s=1.0)
            if service.multimodal_writer is not None:
                service.multimodal_writer.shutdown(timeout_s=1.0)
            conversation_writer = _BudgetRecordingWriter(
                worker_name="twinr-longterm-memory",
                pending_count=1,
                sleep_s=0.12,
            )
            multimodal_writer = _BudgetRecordingWriter(
                worker_name="twinr-longterm-multimodal",
                pending_count=1,
            )
            service.writer = conversation_writer  # type: ignore[assignment]
            service.multimodal_writer = multimodal_writer  # type: ignore[assignment]

            drained = service.flush(timeout_s=0.6)

        self.assertTrue(drained)
        self.assertEqual(len(conversation_writer.recorded_timeouts), 1)
        self.assertEqual(len(multimodal_writer.recorded_timeouts), 1)
        self.assertAlmostEqual(conversation_writer.recorded_timeouts[0], 0.6, delta=0.01)
        self.assertLess(multimodal_writer.recorded_timeouts[0], conversation_writer.recorded_timeouts[0])
        self.assertAlmostEqual(multimodal_writer.recorded_timeouts[0], 0.48, delta=0.05)

    def test_service_flush_skips_idle_writer_and_preserves_full_budget_for_active_writer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
            )
            service = LongTermMemoryService.from_config(config, extractor=make_test_extractor())
            if service.writer is not None:
                service.writer.shutdown(timeout_s=1.0)
            if service.multimodal_writer is not None:
                service.multimodal_writer.shutdown(timeout_s=1.0)
            conversation_writer = _BudgetRecordingWriter(
                worker_name="twinr-longterm-memory",
                pending_count=1,
            )
            multimodal_writer = _BudgetRecordingWriter(
                worker_name="twinr-longterm-multimodal",
                pending_count=0,
            )
            service.writer = conversation_writer  # type: ignore[assignment]
            service.multimodal_writer = multimodal_writer  # type: ignore[assignment]

            drained = service.flush(timeout_s=0.6)

        self.assertTrue(drained)
        self.assertEqual(len(conversation_writer.recorded_timeouts), 1)
        self.assertAlmostEqual(conversation_writer.recorded_timeouts[0], 0.6, delta=0.01)
        self.assertEqual(multimodal_writer.recorded_timeouts, [])

    def test_remote_primary_turn_persistence_skips_memory_markdown_after_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
            )
            prompt_context_store = PromptContextStore.from_config(config)
            graph_store = TwinrPersonalGraphStore(
                Path(temp_dir) / "state" / "chonkydb" / "twinr_graph_v1.json",
                user_label="Erika",
            )
            object_store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb")
            midterm_store = LongTermMidtermStore(base_path=Path(temp_dir) / "state" / "chonkydb")
            helper = LongTermMemoryService.from_config(config, extractor=make_test_extractor())
            try:
                LongTermMemoryService._persist_longterm_turn(
                    config=config,
                    store=prompt_context_store,
                    graph_store=graph_store,
                    object_store=object_store,
                    midterm_store=midterm_store,
                    extractor=make_test_extractor(),
                    consolidator=helper.consolidator,
                    reflector=LongTermMemoryReflector(program=_StubReflectionProgram()),
                    sensor_memory=helper.sensor_memory,
                    retention_policy=helper.retention_policy,
                    item=LongTermConversationTurn(
                        transcript="Today I want to go for a walk if the weather is nice.",
                        response="I can keep the weather in mind for your walk.",
                    ),
                )
            finally:
                helper.shutdown(timeout_s=1.0)
            objects = object_store.load_objects()
            memory_path = Path(config.memory_markdown_path)

        self.assertFalse(memory_path.exists())
        self.assertTrue(any(item.kind == "episode" for item in objects))

    def test_persist_longterm_turn_records_personality_learning_from_consolidation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
            )
            prompt_context_store = PromptContextStore.from_config(config)
            graph_store = TwinrPersonalGraphStore(
                Path(temp_dir) / "state" / "chonkydb" / "twinr_graph_v1.json",
                user_label="Erika",
            )
            object_store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb")
            midterm_store = LongTermMidtermStore(base_path=Path(temp_dir) / "state" / "chonkydb")
            helper = LongTermMemoryService.from_config(config, extractor=make_test_extractor())
            personality_learning = _RecordingPersonalityLearningService()
            turn = LongTermConversationTurn(
                transcript="Today I want to go for a walk if the weather is nice.",
                response="I can keep the weather in mind for your walk.",
            )
            try:
                LongTermMemoryService._persist_longterm_turn(
                    config=config,
                    store=prompt_context_store,
                    graph_store=graph_store,
                    object_store=object_store,
                    midterm_store=midterm_store,
                    extractor=make_test_extractor(),
                    consolidator=helper.consolidator,
                    reflector=LongTermMemoryReflector(program=_StubReflectionProgram()),
                    sensor_memory=helper.sensor_memory,
                    retention_policy=helper.retention_policy,
                    personality_learning=personality_learning,
                    item=turn,
                )
            finally:
                helper.shutdown(timeout_s=1.0)

        self.assertEqual(len(personality_learning.conversation_calls), 1)
        recorded_turn, recorded_result = personality_learning.conversation_calls[0]
        self.assertEqual(recorded_turn.transcript, turn.transcript)
        self.assertEqual(recorded_result.turn_id[:5], "turn:")

    def test_persist_longterm_turn_passes_reflection_thread_summaries_to_personality_learning(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                long_term_memory_mode="remote_primary",
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
            )
            prompt_context_store = PromptContextStore.from_config(config)
            graph_store = TwinrPersonalGraphStore(
                Path(temp_dir) / "state" / "chonkydb" / "twinr_graph_v1.json",
                user_label="Erika",
            )
            object_store = LongTermStructuredStore(base_path=Path(temp_dir) / "state" / "chonkydb")
            midterm_store = LongTermMidtermStore(base_path=Path(temp_dir) / "state" / "chonkydb")
            helper = LongTermMemoryService.from_config(config, extractor=make_test_extractor())
            personality_learning = _RecordingPersonalityLearningService()
            turn = LongTermConversationTurn(
                transcript="Janina is my wife and she has an eye doctor appointment.",
                response="I will keep that Janina thread in mind.",
            )
            reflection_summary = LongTermMemoryObjectV1(
                memory_id="thread:person_janina",
                kind="summary",
                summary="Ongoing thread about Janina: Janina is the user's wife; eye doctor appointment.",
                source=self._source("turn:janina"),
                status="active",
                confidence=0.78,
                slot_key="thread:person:janina",
                value_key="person:janina",
                attributes={
                    "person_ref": "person:janina",
                    "person_name": "Janina",
                    "support_count": 2,
                    "summary_type": "thread",
                    "memory_domain": "thread",
                },
            )
            reflector = _StubThreadSummaryReflector(
                LongTermReflectionResultV1(
                    reflected_objects=(),
                    created_summaries=(reflection_summary,),
                )
            )
            try:
                LongTermMemoryService._persist_longterm_turn(
                    config=config,
                    store=prompt_context_store,
                    graph_store=graph_store,
                    object_store=object_store,
                    midterm_store=midterm_store,
                    extractor=make_test_extractor(),
                    consolidator=helper.consolidator,
                    reflector=reflector,
                    sensor_memory=helper.sensor_memory,
                    retention_policy=helper.retention_policy,
                    personality_learning=personality_learning,
                    item=turn,
                )
            finally:
                helper.shutdown(timeout_s=1.0)

        self.assertEqual(len(personality_learning.conversation_calls), 1)
        _recorded_turn, recorded_result = personality_learning.conversation_calls[0]
        summary_objects = [
            item
            for item in (*recorded_result.durable_objects, *recorded_result.deferred_objects)
            if (item.attributes or {}).get("summary_type") == "thread"
        ]
        self.assertEqual(len(summary_objects), 1)
        self.assertEqual(summary_objects[0].memory_id, reflection_summary.memory_id)

    def test_record_personality_tool_history_routes_tool_calls_to_learning_service(self) -> None:
        service = object.__new__(LongTermMemoryService)
        service.personality_learning = _RecordingPersonalityLearningService()
        service._store_lock = threading.RLock()
        tool_call = AgentToolCall(
            name="search_live_info",
            call_id="call:search:1",
            arguments={"question": "What changed today?"},
        )
        tool_result = AgentToolResult(
            call_id="call:search:1",
            name="search_live_info",
            output={"status": "ok", "answer": "Fresh answer"},
            serialized_output='{"status":"ok"}',
        )

        service.record_personality_tool_history(
            tool_calls=(tool_call,),
            tool_results=(tool_result,),
        )

        self.assertEqual(len(service.personality_learning.tool_history_calls), 1)
        recorded_calls, recorded_results = service.personality_learning.tool_history_calls[0]
        self.assertEqual(recorded_calls[0].name, "search_live_info")
        self.assertEqual(recorded_results[0].call_id, "call:search:1")

    def test_provider_context_combines_recent_episodes_and_graph_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                long_term_memory_recall_limit=2,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                user_display_name="Erika",
            )
            graph_store = TwinrPersonalGraphStore.from_config(config)
            graph_store.remember_preference(
                category="brand",
                value="Melitta",
                for_product="coffee",
            )
            service = LongTermMemoryService.from_config(
                config,
                graph_store=graph_store,
                extractor=make_test_extractor(),
            )
            service.enqueue_conversation_turn(
                transcript="Heute wollte ich spazieren gehen und vorher noch einmal auf das Wetter schauen.",
                response="Dann schaue ich heute gern noch einmal auf das Wetter für den Spaziergang.",
            )
            service.flush(timeout_s=2.0)

            context = service.build_provider_context("Wie wird das Wetter für meinen Spaziergang heute?")
            service.shutdown()

        messages = context.system_messages()
        self.assertEqual(len(messages), 4)
        self.assertIn("Silent personalization background for this turn.", messages[0])
        self.assertIn("twinr_silent_personalization_context_v1", messages[0])
        self.assertIn("twinr_long_term_midterm_context_v1", messages[1])
        self.assertIn("Heute wollte ich spazieren gehen", messages[1])
        self.assertIn("Structured long-term episodic memory for this turn.", messages[2])
        self.assertIn("Heute wollte ich spazieren gehen", messages[2])
        self.assertIn("twinr_graph_memory_context_v1", messages[3])
        self.assertIn("Melitta", messages[3])

    def test_provider_context_does_not_fallback_to_irrelevant_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                long_term_memory_recall_limit=2,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                user_display_name="Erika",
            )
            service = LongTermMemoryService.from_config(config, extractor=make_test_extractor())
            service.enqueue_conversation_turn(
                transcript="Today I want to go for a walk in the park.",
                response="Then the weather matters for the walk.",
            )
            service.flush(timeout_s=2.0)

            context = service.build_provider_context("Was ist 27 mal 14?")
            service.shutdown()

        self.assertIsNone(context.episodic_context)

    def test_subtext_context_surfaces_personalization_without_explicit_memory_language(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                long_term_memory_recall_limit=3,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                user_display_name="Erika",
            )
            graph_store = TwinrPersonalGraphStore.from_config(config)
            graph_store.remember_preference(
                category="brand",
                value="Melitta",
                for_product="coffee",
            )
            graph_store.remember_plan(
                summary="Go for a walk in the park",
                when_text="today",
                details="The user wanted good weather for the walk.",
            )
            service = LongTermMemoryService.from_config(
                config,
                graph_store=graph_store,
                extractor=make_test_extractor(),
            )
            service.enqueue_conversation_turn(
                transcript="Tomorrow I want to go for a walk if the weather is nice.",
                response="I can keep the weather in mind for that walk.",
            )
            service.flush(timeout_s=2.0)

            context = service.build_provider_context("Where can I buy coffee today, and is the weather good?")
            service.shutdown()

        self.assertIsNotNone(context.subtext_context)
        subtext = context.subtext_context or ""
        self.assertIn("twinr_silent_personalization_context_v1", subtext)
        self.assertIn("Melitta", subtext)
        self.assertIn("Go for a walk in the park", subtext)
        self.assertIn("Use it as conversational subtext", subtext)
        self.assertNotIn("I remember", subtext)
        self.assertIn("Do not say earlier, before, last time, neulich", subtext)

    def test_query_rewrite_can_bridge_german_query_to_english_memory_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                long_term_memory_recall_limit=3,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                user_display_name="Erika",
            )
            graph_store = TwinrPersonalGraphStore.from_config(config)
            graph_store.remember_preference(
                category="brand",
                value="Melitta",
                for_product="coffee",
            )
            service = LongTermMemoryService.from_config(config, graph_store=graph_store)
            service.query_rewriter = _StaticQueryRewriter(
                {"Wo kann ich heute Kaffee kaufen?": "Where can I buy coffee today?"}
            )

            context = service.build_provider_context("Wo kann ich heute Kaffee kaufen?")
            service.shutdown()

        self.assertIsNotNone(context.subtext_context)
        self.assertIn("Melitta", context.subtext_context or "")

    def test_query_rewrite_does_not_hide_same_language_historical_fact_recall(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                long_term_memory_recall_limit=3,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                user_display_name="Erika",
            )
            service = LongTermMemoryService.from_config(config, extractor=make_test_extractor())
            service.query_rewriter = _StaticQueryRewriter(
                {"Wo stand früher meine rote Thermoskanne?": "Where did my red thermos flask used to be kept?"}
            )
            service.object_store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:thermos_location_old",
                        kind="fact",
                        summary="Früher stand die rote Thermoskanne im Flurschrank.",
                        details="Historische Ortsangabe zur roten Thermoskanne.",
                        source=self._source("turn:thermos"),
                        status="active",
                        confidence=0.98,
                        confirmed_by_user=True,
                        slot_key="object:red_thermos:location",
                        value_key="hallway_cupboard",
                    ),
                ),
                conflicts=(),
                archived_objects=(),
            )

            context = service.build_provider_context("Wo stand früher meine rote Thermoskanne?")
            tool_context = service.build_tool_provider_context("Wo stand früher meine rote Thermoskanne?")
            service.shutdown()

        self.assertIsNotNone(context.durable_context)
        self.assertIn("Flurschrank", context.durable_context or "")
        self.assertIsNotNone(tool_context.durable_context)
        self.assertIn("Flurschrank", tool_context.durable_context or "")

    def test_query_rewrite_can_surface_confirmed_fact_for_meta_memory_query(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                long_term_memory_recall_limit=3,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                user_display_name="Erika",
            )
            service = LongTermMemoryService.from_config(config, extractor=make_test_extractor())
            service.query_rewriter = _StaticQueryRewriter(
                {
                    "Welche Marmelade ist jetzt als bestaetigt gespeichert?": "Which marmalade is currently saved as confirmed?"
                }
            )
            service.object_store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_generic",
                        kind="fact",
                        summary="User usually likes some jam on bread at breakfast.",
                        details='User said: "Ich mag beim Frühstück meistens etwas Marmelade auf dem Brot."',
                        source=self._source("turn:jam_generic"),
                        status="active",
                        confidence=0.84,
                        slot_key="fact:user:breakfast:jam",
                        value_key="jam_on_bread_at_breakfast",
                        attributes={
                            "fact_type": "general",
                            "memory_domain": "general",
                            "value_text": "jam on bread at breakfast",
                        },
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_preference_new",
                        kind="fact",
                        summary="Inzwischen magst du lieber Aprikosenmarmelade.",
                        details="Neuere Vorliebe fuer das Fruehstueck.",
                        source=self._source("turn:jam_new"),
                        status="active",
                        confidence=0.99,
                        confirmed_by_user=True,
                        slot_key="preference:breakfast:jam",
                        value_key="apricot",
                        attributes={
                            "fact_type": "general",
                            "memory_domain": "general",
                            "resolved_by_user": True,
                        },
                    ),
                ),
                conflicts=(),
                archived_objects=(),
            )

            context = service.build_provider_context("Welche Marmelade ist jetzt als bestaetigt gespeichert?")
            tool_context = service.build_tool_provider_context("Welche Marmelade ist jetzt als bestaetigt gespeichert?")
            service.shutdown()

        self.assertIsNotNone(context.durable_context)
        self.assertIn("Aprikosenmarmelade", context.durable_context or "")
        self.assertIn('"confirmed_by_user": true', context.durable_context or "")
        self.assertIsNotNone(tool_context.durable_context)
        self.assertIn("Aprikosenmarmelade", tool_context.durable_context or "")
        self.assertIn('"slot_key": "preference:breakfast:jam"', tool_context.durable_context or "")

    def test_confirm_memory_persists_restart_recall_packets_for_fresh_service(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                long_term_memory_recall_limit=3,
                long_term_memory_midterm_enabled=True,
                long_term_memory_midterm_limit=3,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                user_display_name="Erika",
            )
            service = LongTermMemoryService.from_config(config, extractor=make_test_extractor())
            service.object_store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:thermos_location_old",
                        kind="fact",
                        summary="Früher stand die rote Thermoskanne im Flurschrank.",
                        details="Historische Ortsangabe zur roten Thermoskanne.",
                        source=self._source("turn:thermos"),
                        status="active",
                        confidence=0.99,
                        confirmed_by_user=True,
                        slot_key="object:red_thermos:location",
                        value_key="hallway_cupboard",
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_generic",
                        kind="fact",
                        summary="User usually likes some jam on bread at breakfast.",
                        details='User said: "Ich mag beim Frühstück meistens etwas Marmelade auf dem Brot."',
                        source=self._source("turn:jam_generic"),
                        status="active",
                        confidence=0.84,
                        slot_key="fact:user:breakfast:jam",
                        value_key="jam_on_bread_at_breakfast",
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_preference_old",
                        kind="fact",
                        summary="Deine Lieblingsmarmelade ist Erdbeermarmelade.",
                        details="Aeltere Vorliebe fuer das Fruehstueck.",
                        source=self._source("turn:jam_old"),
                        status="active",
                        confidence=0.94,
                        slot_key="preference:breakfast:jam",
                        value_key="strawberry",
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_preference_new",
                        kind="fact",
                        summary="Inzwischen magst du lieber Aprikosenmarmelade.",
                        details="Neuere Vorliebe fuer das Fruehstueck.",
                        source=self._source("turn:jam_new"),
                        status="uncertain",
                        confidence=0.95,
                        slot_key="preference:breakfast:jam",
                        value_key="apricot",
                    ),
                ),
                conflicts=(
                    LongTermMemoryConflictV1(
                        slot_key="preference:breakfast:jam",
                        candidate_memory_id="fact:jam_preference_new",
                        existing_memory_ids=("fact:jam_preference_old",),
                        question="Welche Marmelade stimmt gerade?",
                        reason="Widerspruechliche Marmeladenpraeferenzen liegen vor.",
                    ),
                ),
                archived_objects=(),
            )
            service.confirm_memory(memory_id="fact:jam_preference_new")
            service.shutdown()

            fresh_service = LongTermMemoryService.from_config(config, extractor=make_test_extractor())
            stored_packets = fresh_service.midterm_store.load_packets()
            jam_context = fresh_service.build_provider_context("Welche Marmelade ist jetzt als bestaetigt gespeichert?")
            thermos_context = fresh_service.build_provider_context("Wo stand früher meine rote Thermoskanne?")
            control_context = fresh_service.build_provider_context("Was ist ein Regenbogen?")
            fresh_service.shutdown()

        self.assertEqual(
            [item.packet_id for item in stored_packets],
            [
                "adaptive:restart:fact_jam_preference_new",
                "adaptive:restart:fact_thermos_location_old",
            ],
        )
        self.assertIn("Aprikosenmarmelade", jam_context.midterm_context or "")
        self.assertIn("restart_recall", jam_context.midterm_context or "")
        self.assertIn("Flurschrank", thermos_context.midterm_context or "")
        self.assertNotIn("Aprikosenmarmelade", control_context.midterm_context or "")

    def test_provider_context_omits_off_topic_conflict_memory_for_control_query(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                long_term_memory_recall_limit=3,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                user_display_name="Erika",
            )
            service = LongTermMemoryService.from_config(config, extractor=make_test_extractor())
            service.query_rewriter = _StaticQueryRewriter(
                {
                    "Was ist ein Regenbogen?": "What is a rainbow?",
                }
            )
            service.object_store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_preference_old",
                        kind="fact",
                        summary="Deine Lieblingsmarmelade ist Erdbeermarmelade.",
                        details="Aeltere Vorliebe fuer das Fruehstueck.",
                        source=self._source("turn:jam_old"),
                        status="active",
                        confidence=0.94,
                        slot_key="preference:breakfast:jam",
                        value_key="strawberry",
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:jam_preference_new",
                        kind="fact",
                        summary="Inzwischen magst du lieber Aprikosenmarmelade.",
                        details="Neuere Vorliebe fuer das Fruehstueck.",
                        source=self._source("turn:jam_new"),
                        status="uncertain",
                        confidence=0.95,
                        slot_key="preference:breakfast:jam",
                        value_key="apricot",
                    ),
                ),
                conflicts=(
                    LongTermMemoryConflictV1(
                        slot_key="preference:breakfast:jam",
                        candidate_memory_id="fact:jam_preference_new",
                        existing_memory_ids=("fact:jam_preference_old",),
                        question="Welche Marmelade stimmt gerade?",
                        reason="Widerspruechliche Marmeladenpraeferenzen liegen vor.",
                    ),
                ),
                archived_objects=(),
            )

            context = service.build_provider_context("Was ist ein Regenbogen?")
            service.shutdown()

        self.assertIsNone(context.durable_context)
        self.assertIsNone(context.conflict_context)

    def test_explicit_memory_and_profile_updates_route_through_service(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            personality_dir = Path(temp_dir) / "personality"
            personality_dir.mkdir(parents=True, exist_ok=True)
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
            )
            service = LongTermMemoryService.from_config(config, extractor=make_test_extractor())

            memory_entry = service.store_explicit_memory(
                kind="fact",
                summary="Preferred pharmacy is Linden Apotheke.",
                details="Use this as a stable reference.",
            )
            user_entry = service.update_user_profile(
                category="preferred_name",
                instruction="Call the user Erika.",
            )
            personality_entry = service.update_personality(
                category="response_style",
                instruction="Keep answers calm and short.",
            )

            memory_text = Path(config.memory_markdown_path).read_text(encoding="utf-8")
            user_text = (personality_dir / "USER.md").read_text(encoding="utf-8")
            personality_text = (personality_dir / "PERSONALITY.md").read_text(encoding="utf-8")

        self.assertEqual(memory_entry.kind, "fact")
        self.assertIn("Preferred pharmacy is Linden Apotheke.", memory_text)
        self.assertEqual(user_entry.key, "preferred_name")
        self.assertIn("preferred_name: Call the user Erika.", user_text)
        self.assertEqual(personality_entry.key, "response_style")
        self.assertIn("response_style: Keep answers calm and short.", personality_text)

    def test_service_can_analyze_turn_into_consolidated_memory_objects(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                user_display_name="Erika",
            )
            service = LongTermMemoryService.from_config(config, extractor=make_test_extractor())

            result = service.analyze_conversation_turn(
                transcript=(
                    "Today is a beautiful Sunday, it is really warm. "
                    "My wife Janina is at the eye doctor and is getting eye laser treatment."
                ),
                response="I hope Janina's appointment goes smoothly.",
            )

        durable_summaries = [item.summary for item in result.durable_objects]
        self.assertIn("Janina is the user's wife.", durable_summaries)
        self.assertTrue(any("eye laser treatment" in summary for summary in durable_summaries))
        self.assertFalse(result.clarification_needed)

    def test_provider_context_can_include_structured_durable_memory_from_background_store(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                user_display_name="Erika",
            )
            service = LongTermMemoryService.from_config(config, extractor=make_test_extractor())
            service.enqueue_conversation_turn(
                transcript=(
                    "Today is a beautiful Sunday, it is really warm. "
                    "My wife Janina is at the eye doctor and is getting eye laser treatment."
                ),
                response="I hope Janina's appointment goes smoothly.",
            )
            service.flush(timeout_s=2.0)

            context = service.build_provider_context("How is Janina today?")
            service.shutdown()

        self.assertIsNotNone(context.durable_context)
        self.assertIn("twinr_long_term_durable_context_v1", context.durable_context or "")
        self.assertIn("Janina is the user's wife.", context.durable_context or "")
        self.assertIn("eye laser treatment", context.durable_context or "")
        self.assertIn("Ongoing thread about Janina", context.durable_context or "")

    def test_service_can_plan_bounded_proactive_candidates_from_stored_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                user_display_name="Erika",
            )
            service = LongTermMemoryService.from_config(config, extractor=make_test_extractor())
            service.enqueue_conversation_turn(
                transcript=(
                    "Today is a beautiful Sunday, it is really warm. "
                    "My wife Janina is at the eye doctor and is getting eye laser treatment."
                ),
                response="I hope Janina's appointment goes smoothly.",
            )
            service.flush(timeout_s=2.0)

            plan = service.plan_proactive_candidates()
            service.shutdown()

        candidate_kinds = {item.kind for item in plan.candidates}
        self.assertIn("same_day_reminder", candidate_kinds)
        self.assertIn("gentle_follow_up", candidate_kinds)

    def test_service_reflection_can_persist_midterm_packets_and_expose_them(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                long_term_memory_midterm_enabled=True,
                long_term_memory_midterm_limit=3,
                user_display_name="Erika",
            )
            service = LongTermMemoryService.from_config(config, extractor=make_test_extractor())
            service.reflector = LongTermMemoryReflector(
                program=_StubReflectionProgram(),
                midterm_packet_limit=3,
                reflection_window_size=12,
                timezone_name=config.local_timezone_name,
            )
            service.object_store.apply_consolidation(
                service.consolidator.consolidate(
                    extraction=service.extractor.extract_conversation_turn(
                        transcript="My wife Janina is getting eye laser treatment today.",
                        response="I hope Janina's appointment goes smoothly.",
                        occurred_at=datetime(2026, 3, 15, 10, 0, tzinfo=timezone.utc),
                    ),
                    existing_objects=service.object_store.load_objects(),
                )
            )
            reflection = service.run_reflection()
            context = service.build_provider_context("How is Janina today?")
            stored_packets = service.midterm_store.load_packets()
            service.shutdown()

        self.assertEqual(len(reflection.midterm_packets), 1)
        self.assertEqual(len(stored_packets), 1)
        self.assertIsNotNone(context.midterm_context)
        self.assertIn("twinr_long_term_midterm_context_v1", context.midterm_context or "")
        self.assertIn("Janina has eye laser treatment today.", context.midterm_context or "")

    def test_service_reflection_triggers_world_intelligence_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                user_display_name="Erika",
            )
            service = LongTermMemoryService.from_config(config, extractor=make_test_extractor())
            calls: list[str] = []

            class _TrackingPersonalityLearning:
                def maybe_refresh_world_intelligence(self_nonlocal, *, search_backend=None) -> None:
                    calls.append("refresh" if search_backend is None else "refresh_with_backend")
                    return None

            service.personality_learning = _TrackingPersonalityLearning()
            service.reflector = LongTermMemoryReflector(
                program=_StubReflectionProgram(),
                midterm_packet_limit=3,
                reflection_window_size=12,
                timezone_name=config.local_timezone_name,
            )
            service.run_reflection()
            service.shutdown()

        self.assertEqual(calls, ["refresh"])

    def test_service_reflection_passes_optional_search_backend_into_world_intelligence_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                user_display_name="Erika",
            )
            service = LongTermMemoryService.from_config(config, extractor=make_test_extractor())
            calls: list[object | None] = []
            backend = object()

            class _TrackingPersonalityLearning:
                def maybe_refresh_world_intelligence(self_nonlocal, *, search_backend=None) -> None:
                    calls.append(search_backend)
                    return None

            service.personality_learning = _TrackingPersonalityLearning()
            service.reflector = LongTermMemoryReflector(
                program=_StubReflectionProgram(),
                midterm_packet_limit=3,
                reflection_window_size=12,
                timezone_name=config.local_timezone_name,
            )
            service.run_reflection(search_backend=backend)
            service.shutdown()

        self.assertEqual(calls, [backend])

    def test_service_can_run_retention_and_remove_old_ephemeral_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                user_display_name="Erika",
            )
            service = LongTermMemoryService.from_config(config, extractor=make_test_extractor())
            service.object_store.apply_consolidation(
                service.consolidator.consolidate(
                    extraction=service.extractor.extract_conversation_turn(
                        transcript="Today is warm.",
                        response="It sounds pleasant outside.",
                    )
                )
            )
            objects = service.object_store.load_objects()
            old_episode = next(item for item in objects if item.kind == "episode")
            old_observation = next(item for item in objects if item.kind == "observation")
            service.object_store.apply_consolidation(
                service.consolidator.consolidate(
                    extraction=service.extractor.extract_conversation_turn(
                        transcript="My wife Janina is getting eye laser treatment today.",
                        response="I hope it goes smoothly.",
                    )
                )
            )
            rewritten = []
            for item in service.object_store.load_objects():
                if item.memory_id in {old_episode.memory_id, old_observation.memory_id}:
                    rewritten.append(
                        item.with_updates(
                            created_at=datetime(2026, 2, 1, 10, 0, tzinfo=timezone.utc),
                            updated_at=datetime(2026, 2, 1, 10, 0, tzinfo=timezone.utc),
                        )
                    )
                else:
                    rewritten.append(item)
            service.object_store.apply_retention(
                service.retention_policy.apply(
                    objects=tuple(rewritten),
                    now=datetime(2026, 3, 16, 10, 0, tzinfo=timezone.utc),
                )
            )

            kept = service.object_store.load_objects()
            service.shutdown()

        kept_ids = {item.memory_id for item in kept}
        self.assertNotIn(old_episode.memory_id, kept_ids)
        self.assertNotIn(old_observation.memory_id, kept_ids)
        self.assertTrue(
            any(
                item.kind == "event"
                and item.status == "active"
                and (item.attributes or {}).get("event_domain") == "appointment"
                for item in kept
            )
        )

    def test_service_can_review_memory_without_surface_noise(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                user_display_name="Erika",
            )
            service = LongTermMemoryService.from_config(config, extractor=make_test_extractor())
            service.enqueue_conversation_turn(
                transcript="My wife Janina is at the eye doctor today.",
                response="I hope Janina's appointment goes smoothly.",
            )
            service.flush(timeout_s=2.0)

            review = service.review_memory(query_text="Janina eye doctor", include_episodes=False, limit=5)
            service.shutdown()

        self.assertGreaterEqual(review.total_count, 2)
        self.assertTrue(all(item.kind != "episode" for item in review.items))
        self.assertTrue(any("Janina" in item.summary for item in review.items))

    def test_backfill_ops_history_builds_patterns_routines_and_deviation_objects(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                long_term_memory_sensor_memory_enabled=True,
                long_term_memory_sensor_baseline_days=7,
                long_term_memory_sensor_min_days_observed=4,
                long_term_memory_sensor_min_routine_ratio=0.6,
                long_term_memory_sensor_deviation_min_delta=0.5,
                user_display_name="Erika",
            )
            service = LongTermMemoryService.from_config(config, extractor=make_test_extractor())
            entries = (
                self._ops_entry(
                    event="proactive_observation",
                    created_at="2026-03-09T08:00:00+00:00",
                    data={
                        "inspected": True,
                        "pir_motion_detected": True,
                        "low_motion": False,
                        "person_visible": True,
                        "looking_toward_device": False,
                        "body_pose": "upright",
                        "smiling": False,
                        "hand_or_object_near_camera": False,
                        "speech_detected": False,
                        "distress_detected": False,
                    },
                ),
                self._ops_entry(
                    event="turn_started",
                    created_at="2026-03-09T08:01:00+00:00",
                    data={"request_source": "button"},
                ),
                self._ops_entry(
                    event="print_started",
                    created_at="2026-03-09T13:00:00+00:00",
                    data={"button": "yellow", "request_source": "button", "queue": "Thermal_GP58"},
                ),
                self._ops_entry(
                    event="print_job_sent",
                    created_at="2026-03-09T13:00:02+00:00",
                    data={"queue": "Thermal_GP58", "job": "Thermal_GP58-1"},
                ),
                self._ops_entry(
                    event="proactive_observation",
                    created_at="2026-03-10T08:05:00+00:00",
                    data={
                        "inspected": True,
                        "pir_motion_detected": True,
                        "low_motion": False,
                        "person_visible": True,
                        "looking_toward_device": True,
                        "body_pose": "upright",
                        "smiling": False,
                        "hand_or_object_near_camera": False,
                        "speech_detected": False,
                        "distress_detected": False,
                    },
                ),
                self._ops_entry(
                    event="turn_started",
                    created_at="2026-03-10T08:06:00+00:00",
                    data={"request_source": "button"},
                ),
                self._ops_entry(
                    event="print_started",
                    created_at="2026-03-10T13:00:00+00:00",
                    data={"button": "yellow", "request_source": "button", "queue": "Thermal_GP58"},
                ),
                self._ops_entry(
                    event="print_job_sent",
                    created_at="2026-03-10T13:00:02+00:00",
                    data={"queue": "Thermal_GP58", "job": "Thermal_GP58-2"},
                ),
                self._ops_entry(
                    event="proactive_observation",
                    created_at="2026-03-11T08:10:00+00:00",
                    data={
                        "inspected": True,
                        "pir_motion_detected": True,
                        "low_motion": False,
                        "person_visible": True,
                        "looking_toward_device": False,
                        "body_pose": "upright",
                        "smiling": False,
                        "hand_or_object_near_camera": True,
                        "speech_detected": False,
                        "distress_detected": False,
                    },
                ),
                self._ops_entry(
                    event="turn_started",
                    created_at="2026-03-11T08:11:00+00:00",
                    data={"request_source": "button"},
                ),
                self._ops_entry(
                    event="print_started",
                    created_at="2026-03-11T13:00:00+00:00",
                    data={"button": "yellow", "request_source": "button", "queue": "Thermal_GP58"},
                ),
                self._ops_entry(
                    event="print_job_sent",
                    created_at="2026-03-11T13:00:02+00:00",
                    data={"queue": "Thermal_GP58", "job": "Thermal_GP58-3"},
                ),
                self._ops_entry(
                    event="proactive_observation",
                    created_at="2026-03-13T08:15:00+00:00",
                    data={
                        "inspected": True,
                        "pir_motion_detected": True,
                        "low_motion": False,
                        "person_visible": True,
                        "looking_toward_device": False,
                        "body_pose": "upright",
                        "smiling": False,
                        "hand_or_object_near_camera": True,
                        "speech_detected": False,
                        "distress_detected": False,
                    },
                ),
                self._ops_entry(
                    event="turn_started",
                    created_at="2026-03-13T08:16:00+00:00",
                    data={"request_source": "button"},
                ),
                self._ops_entry(
                    event="print_started",
                    created_at="2026-03-13T13:00:00+00:00",
                    data={"button": "yellow", "request_source": "button", "queue": "Thermal_GP58"},
                ),
                self._ops_entry(
                    event="print_job_sent",
                    created_at="2026-03-13T13:00:02+00:00",
                    data={"queue": "Thermal_GP58", "job": "Thermal_GP58-4"},
                ),
            )

            result = service.backfill_ops_multimodal_history(
                entries=entries,
                now=datetime(2026, 3, 16, 9, 30, tzinfo=timezone.utc),
            )
            objects = {item.memory_id: item for item in service.object_store.load_objects()}
            service.shutdown()

        self.assertEqual(result.scanned_events, len(entries))
        self.assertEqual(result.sensor_observations, 4)
        self.assertEqual(result.button_interactions, 8)
        self.assertEqual(result.print_completions, 4)
        self.assertEqual(result.applied_evidence, result.generated_evidence)
        self.assertIn("pattern:presence:morning:near_device", objects)
        self.assertIn("pattern:camera_interaction:morning", objects)
        self.assertIn("pattern:button:green:start_listening:morning", objects)
        self.assertIn("pattern:print:button:afternoon", objects)
        self.assertIn("routine:presence:weekday:morning", objects)
        self.assertIn("routine:interaction:conversation_start:weekday:morning", objects)
        self.assertIn("routine:interaction:print:weekday:afternoon", objects)
        self.assertIn("deviation:presence:weekday:morning:2026-03-16", objects)

    def test_backfill_ops_history_is_idempotent_when_replayed_twice(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                long_term_memory_sensor_memory_enabled=True,
                long_term_memory_sensor_baseline_days=7,
                long_term_memory_sensor_min_days_observed=4,
                long_term_memory_sensor_min_routine_ratio=0.6,
                long_term_memory_sensor_deviation_min_delta=0.5,
            )
            service = LongTermMemoryService.from_config(config, extractor=make_test_extractor())
            entries = (
                self._ops_entry(
                    event="proactive_observation",
                    created_at="2026-03-09T08:00:00+00:00",
                    data={
                        "inspected": True,
                        "pir_motion_detected": True,
                        "low_motion": False,
                        "person_visible": True,
                        "looking_toward_device": False,
                        "body_pose": "upright",
                        "smiling": False,
                        "hand_or_object_near_camera": False,
                        "speech_detected": False,
                        "distress_detected": False,
                    },
                ),
                self._ops_entry(
                    event="turn_started",
                    created_at="2026-03-09T08:01:00+00:00",
                    data={"request_source": "button"},
                ),
                self._ops_entry(
                    event="print_started",
                    created_at="2026-03-09T13:00:00+00:00",
                    data={"button": "yellow", "request_source": "button", "queue": "Thermal_GP58"},
                ),
                self._ops_entry(
                    event="print_job_sent",
                    created_at="2026-03-09T13:00:02+00:00",
                    data={"queue": "Thermal_GP58", "job": "Thermal_GP58-1"},
                ),
            )

            first = service.backfill_ops_multimodal_history(
                entries=entries,
                now=datetime(2026, 3, 10, 10, 0, tzinfo=timezone.utc),
            )
            source_ids_before = objects = {
                item.memory_id: tuple(item.source.event_ids)
                for item in service.object_store.load_objects()
            }
            second = service.backfill_ops_multimodal_history(
                entries=entries,
                now=datetime(2026, 3, 10, 10, 0, tzinfo=timezone.utc),
            )
            source_ids_after = {
                item.memory_id: tuple(item.source.event_ids)
                for item in service.object_store.load_objects()
            }
            service.shutdown()

        self.assertGreater(first.applied_evidence, 0)
        self.assertEqual(second.applied_evidence, 0)
        self.assertEqual(second.skipped_existing, second.generated_evidence)
        self.assertEqual(source_ids_before, source_ids_after)

    def test_backfill_ops_history_still_compiles_sensor_memory_when_reflection_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                long_term_memory_sensor_memory_enabled=True,
                long_term_memory_sensor_baseline_days=7,
                long_term_memory_sensor_min_days_observed=4,
                long_term_memory_sensor_min_routine_ratio=0.6,
                long_term_memory_sensor_deviation_min_delta=0.5,
            )
            service = LongTermMemoryService.from_config(config, extractor=make_test_extractor())
            service.reflector = LongTermMemoryReflector(program=_FailingReflectionProgram())
            entries = (
                self._ops_entry(
                    event="proactive_observation",
                    created_at="2026-03-09T08:00:00+00:00",
                    data={
                        "inspected": True,
                        "pir_motion_detected": True,
                        "person_visible": True,
                        "body_pose": "upright",
                        "speech_detected": False,
                        "distress_detected": False,
                    },
                ),
                self._ops_entry(
                    event="proactive_observation",
                    created_at="2026-03-10T08:00:00+00:00",
                    data={
                        "inspected": True,
                        "pir_motion_detected": True,
                        "person_visible": True,
                        "body_pose": "upright",
                        "speech_detected": False,
                        "distress_detected": False,
                    },
                ),
                self._ops_entry(
                    event="proactive_observation",
                    created_at="2026-03-11T08:00:00+00:00",
                    data={
                        "inspected": True,
                        "pir_motion_detected": True,
                        "person_visible": True,
                        "body_pose": "upright",
                        "speech_detected": False,
                        "distress_detected": False,
                    },
                ),
                self._ops_entry(
                    event="proactive_observation",
                    created_at="2026-03-13T08:00:00+00:00",
                    data={
                        "inspected": True,
                        "pir_motion_detected": True,
                        "person_visible": True,
                        "body_pose": "upright",
                        "speech_detected": False,
                        "distress_detected": False,
                    },
                ),
            )

            result = service.backfill_ops_multimodal_history(
                entries=entries,
                now=datetime(2026, 3, 16, 9, 30, tzinfo=timezone.utc),
            )
            objects = {item.memory_id: item for item in service.object_store.load_objects()}
            service.shutdown()

        self.assertIsNone(result.reflection_error)
        self.assertIn("routine:presence:weekday:morning", objects)

    def test_confirm_memory_can_resolve_open_conflict_via_service(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                user_display_name="Erika",
            )
            service = LongTermMemoryService.from_config(config, extractor=make_test_extractor())
            service.object_store.apply_consolidation(
                LongTermConsolidationResultV1(
                    turn_id="turn:2",
                    occurred_at=datetime(2026, 3, 14, 12, 0, tzinfo=timezone.utc),
                    episodic_objects=(),
                    durable_objects=(
                        LongTermMemoryObjectV1(
                            memory_id="fact:corinna_phone_old",
                            kind="contact_method_fact",
                            summary="Corinna Maier can be reached at +491761234.",
                            source=self._source("turn:1"),
                            status="active",
                            confidence=0.95,
                            slot_key="contact:person:corinna_maier:phone",
                            value_key="+491761234",
                            attributes={"person_ref": "person:corinna_maier"},
                        ),
                    ),
                    deferred_objects=(
                        LongTermMemoryObjectV1(
                            memory_id="fact:corinna_phone_new",
                            kind="contact_method_fact",
                            summary="Corinna Maier can be reached at +4940998877.",
                            source=self._source("turn:2"),
                            status="uncertain",
                            confidence=0.92,
                            slot_key="contact:person:corinna_maier:phone",
                            value_key="+4940998877",
                            attributes={"person_ref": "person:corinna_maier"},
                        ),
                    ),
                    conflicts=(
                        LongTermMemoryConflictV1(
                            slot_key="contact:person:corinna_maier:phone",
                            candidate_memory_id="fact:corinna_phone_new",
                            existing_memory_ids=("fact:corinna_phone_old",),
                            question="Which phone number should I use for Corinna Maier?",
                            reason="Conflicting phone numbers exist.",
                        ),
                    ),
                    graph_edges=(),
                )
            )
            queue_before = service.select_conflict_queue("What is Corinna's number?")

            resolution = service.confirm_memory(memory_id=queue_before[0].candidate_memory_id)
            objects = {item.memory_id: item for item in service.object_store.load_objects()}
            queue_after = service.select_conflict_queue("What is Corinna's number?")
            service.shutdown()

        self.assertEqual(len(queue_before), 1)
        self.assertEqual(resolution.selected_memory_id, queue_before[0].candidate_memory_id)
        self.assertEqual(queue_after, ())
        self.assertTrue(any(item.status == "active" and item.confirmed_by_user for item in objects.values()))
        self.assertTrue(any(item.status in {"superseded", "invalid"} for item in objects.values()))

    def test_service_can_invalidate_and_delete_memory_objects(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                user_display_name="Erika",
            )
            service = LongTermMemoryService.from_config(config, extractor=make_test_extractor())
            service.object_store.apply_consolidation(
                service.consolidator.consolidate(
                    extraction=service.extractor.extract_conversation_turn(
                        transcript="My wife Janina is at the eye doctor today.",
                        response="I hope Janina's appointment goes smoothly.",
                    )
                )
            )
            current_objects = tuple(item for item in service.object_store.load_objects() if item.kind != "episode")
            relationship = next(
                item
                for item in current_objects
                if item.kind == "fact" and (item.attributes or {}).get("fact_type") == "relationship"
            )
            event = next(
                item
                for item in current_objects
                if item.kind == "event" and (item.attributes or {}).get("event_domain") == "appointment"
            )

            invalidation = service.invalidate_memory(memory_id=event.memory_id, reason="This appointment was canceled.")
            deletion = service.delete_memory(memory_id=relationship.memory_id)
            objects = {item.memory_id: item for item in service.object_store.load_objects()}
            service.shutdown()

        self.assertEqual(invalidation.action, "invalidate")
        self.assertEqual(objects[event.memory_id].status, "invalid")
        self.assertEqual(objects[event.memory_id].attributes["invalidation_reason"], "This appointment was canceled.")
        self.assertEqual(deletion.action, "delete")
        self.assertNotIn(relationship.memory_id, objects)


if __name__ == "__main__":
    unittest.main()
