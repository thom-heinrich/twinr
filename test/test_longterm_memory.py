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
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.contracts import AgentToolCall, AgentToolResult
from test.longterm_test_program import make_test_extractor
from twinr.agent.base_agent import TwinrConfig
from twinr.agent.base_agent.runtime.memory import TwinrRuntimeMemoryMixin
from twinr.memory.chonkydb import TwinrPersonalGraphStore
from twinr.memory.chonkydb.client import ChonkyDBError
from twinr.memory.context_store import PromptContextStore
from twinr.memory.longterm import (
    LongTermConsolidationResultV1,
    LongTermConversationTurn,
    LongTermMemoryConflictV1,
    LongTermMemoryContext,
    LongTermMemoryObjectV1,
    LongTermMultimodalEvidence,
    LongTermMemoryReflector,
    LongTermReflectionResultV1,
    LongTermMemoryService,
    LongTermMidtermStore,
    LongTermSourceRefV1,
    LongTermStructuredStore,
)
from twinr.memory.longterm.storage.remote_state import (
    LongTermRemoteReadFailedError,
)
from twinr.memory.longterm.runtime.prepared_context import PreparedLongTermContextFront
from twinr.memory.longterm.runtime.worker import AsyncLongTermMemoryWriter, AsyncLongTermWriterState
from twinr.memory.longterm.retrieval.retriever import LongTermRetriever, _LongTermContextInputs
from twinr.memory.query_normalization import LongTermQueryProfile
from test.test_longterm_store import _FakeRemoteState

_TEST_CORINNA_PHONE_OLD = "+15555551234"
_TEST_CORINNA_PHONE_NEW = "+15555558877"


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
        self.attest_count = 0

    @contextmanager
    def cache_probe_reads(self):
        self.enter_count += 1
        yield

    def attest_external_readiness(self) -> None:
        self.attest_count += 1


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

    def build_tool_context(
        self,
        *,
        query: LongTermQueryProfile,
        original_query_text: str | None = None,
        include_graph_fallback: bool = True,
    ) -> LongTermMemoryContext:
        del query
        del original_query_text
        del include_graph_fallback
        return LongTermMemoryContext(episodic_context="remembered", graph_context="graph")


class _BlockingPreparedRetriever:
    def __init__(self) -> None:
        self.object_store = type("ObjectStoreStub", (), {"remote_state": _CountingRemoteState()})()
        self.started = threading.Event()
        self.release = threading.Event()
        self.provider_calls: list[str] = []

    def build_context(self, *, query: LongTermQueryProfile, original_query_text: str | None = None) -> LongTermMemoryContext:
        del original_query_text
        self.provider_calls.append(query.retrieval_text)
        self.started.set()
        self.release.wait(timeout=5.0)
        return LongTermMemoryContext(durable_context=f"prepared:{query.retrieval_text}:{len(self.provider_calls)}")

    def build_tool_context(
        self,
        *,
        query: LongTermQueryProfile,
        original_query_text: str | None = None,
        include_graph_fallback: bool = True,
    ) -> LongTermMemoryContext:
        del original_query_text
        del include_graph_fallback
        return LongTermMemoryContext(graph_context=f"tool:{query.retrieval_text}")


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
        self.queued_tool_history_calls: list[tuple[tuple[AgentToolCall, ...], tuple[AgentToolResult, ...]]] = []
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

    def enqueue_tool_history(
        self,
        *,
        tool_calls: tuple[AgentToolCall, ...],
        tool_results: tuple[AgentToolResult, ...],
    ) -> None:
        self.queued_tool_history_calls.append((tool_calls, tool_results))


class _RecordingFlushService:
    def __init__(self, *, flush_result: bool = True) -> None:
        self.flush_result = flush_result
        self.flush_timeouts: list[float] = []

    def flush(self, *, timeout_s: float) -> bool:
        self.flush_timeouts.append(timeout_s)
        return self.flush_result


class _RecordingPromptContextMutationService(_RecordingFlushService):
    def __init__(self) -> None:
        super().__init__(flush_result=True)
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def store_explicit_memory(self, *, kind: str, summary: str, details: str | None = None):
        self.calls.append(("store_explicit_memory", (kind, summary, details)))
        return SimpleNamespace(kind=kind, summary=summary, details=details)

    def delete_explicit_memory(self, *, entry_id: str):
        self.calls.append(("delete_explicit_memory", (entry_id,)))
        return SimpleNamespace(entry_id=entry_id)

    def update_user_profile(self, *, category: str, instruction: str):
        self.calls.append(("update_user_profile", (category, instruction)))
        return SimpleNamespace(key=category, instruction=instruction)

    def remove_user_profile(self, *, category: str):
        self.calls.append(("remove_user_profile", (category,)))
        return SimpleNamespace(key=category)

    def update_personality(self, *, category: str, instruction: str):
        self.calls.append(("update_personality", (category, instruction)))
        return SimpleNamespace(key=category, instruction=instruction)

    def remove_personality(self, *, category: str):
        self.calls.append(("remove_personality", (category,)))
        return SimpleNamespace(key=category)


class _RuntimeMemoryProbe(TwinrRuntimeMemoryMixin):
    def __init__(self, *, config: TwinrConfig, long_term_memory: object) -> None:
        self.config = config
        self.long_term_memory = long_term_memory


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
    def __init__(self, delegate, *, wait_timeout_s: float = 10.0) -> None:
        self._delegate = delegate
        self.started = threading.Event()
        self.release = threading.Event()
        self._wait_timeout_s = max(0.1, float(wait_timeout_s))

    def extract_conversation_turn(self, **kwargs):
        self.started.set()
        self.release.wait(timeout=self._wait_timeout_s)
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
            prepared_generation = service.prepared_context_front.generation() if service.prepared_context_front is not None else -1
            service.shutdown()

        self.assertIsNotNone(result)
        self.assertTrue(result.accepted)
        self.assertTrue(drained)
        self.assertGreater(prepared_generation, 0)
        self.assertEqual(entries[0].kind, "episodic_turn")
        self.assertIn('Conversation about "Wie wird das Wetter heute?"', entries[0].summary)
        self.assertIn('Twinr answered: "Heute ist es sonnig und mild."', entries[0].details or "")
        self.assertTrue(any(item.kind == "episode" for item in stored_objects))

    def test_background_multimodal_writer_invalidates_prepared_contexts_without_worker_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                user_display_name="Erika",
                long_term_memory_write_queue_size=4,
            )
            service = LongTermMemoryService.from_config(config, extractor=make_test_extractor())
            try:
                result = service.enqueue_multimodal_evidence(
                    event_name="print_completed",
                    modality="printer",
                    source="realtime_print",
                    message="Printed Twinr output was delivered from the realtime loop.",
                    data={"request_source": "button", "queue": "Thermal_GP58"},
                )
                drained = service.flush(timeout_s=2.0)
                writer_state = service.multimodal_writer.snapshot_state() if service.multimodal_writer is not None else None
                prepared_generation = service.prepared_context_front.generation() if service.prepared_context_front is not None else -1
            finally:
                service.shutdown()

        self.assertIsNotNone(result)
        self.assertTrue(result.accepted)
        self.assertTrue(drained)
        assert writer_state is not None
        self.assertIsNone(writer_state.last_error_message)
        self.assertGreater(prepared_generation, 0)

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
            extractor = _BlockingExtractor(make_test_extractor(), wait_timeout_s=10.0)
            service = LongTermMemoryService.from_config(config, extractor=extractor)
            try:
                result = service.enqueue_conversation_turn(
                    transcript="Lea bringt mir heute Abend eine Thermoskanne mit Linsensuppe vorbei.",
                    response="Ich merke mir, dass Lea dir heute Abend die Thermoskanne mit Linsensuppe bringt.",
                )
                self.assertIsNotNone(result)
                assert result is not None
                self.assertTrue(result.accepted)
                self.assertTrue(extractor.started.wait(timeout=5.0))

                deadline = time.monotonic() + 5.0
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

    def test_attest_external_remote_ready_visits_each_unique_remote_state_once(self) -> None:
        shared_remote_state = _CountingRemoteState()
        distinct_remote_state = _CountingRemoteState()
        service = object.__new__(LongTermMemoryService)
        service.prompt_context_store = SimpleNamespace(
            memory_store=SimpleNamespace(remote_state=shared_remote_state),
            user_store=SimpleNamespace(remote_state=shared_remote_state),
            personality_store=SimpleNamespace(remote_state=distinct_remote_state),
        )
        service.graph_store = SimpleNamespace(remote_state=shared_remote_state)
        service.object_store = SimpleNamespace(remote_state=distinct_remote_state)
        service.midterm_store = SimpleNamespace(remote_state=None)

        service.attest_external_remote_ready()

        self.assertEqual(shared_remote_state.attest_count, 1)
        self.assertEqual(distinct_remote_state.attest_count, 1)

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

    def test_prewarmed_provider_context_reuses_inflight_full_context_without_duplicate_retrieval(self) -> None:
        retriever = _BlockingPreparedRetriever()
        service = object.__new__(LongTermMemoryService)
        service.config = TwinrConfig(
            project_root=".",
            personality_dir="personality",
            long_term_memory_enabled=True,
            long_term_memory_fast_topic_enabled=False,
        )
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
        service.fast_topic_builder = None
        service.prepared_context_front = PreparedLongTermContextFront()
        service._store_lock = threading.RLock()
        result: dict[str, LongTermMemoryContext] = {}

        try:
            scheduled = service.prewarm_provider_context("Wie geht es Janina?", rewrite_query=False)
            self.assertTrue(scheduled)
            self.assertTrue(retriever.started.wait(timeout=2.0))

            worker = threading.Thread(
                target=lambda: result.setdefault(
                    "context",
                    service.build_provider_context("Wie geht es Janina?"),
                ),
                daemon=True,
            )
            worker.start()
            time.sleep(0.05)
            self.assertEqual(len(retriever.provider_calls), 1)

            retriever.release.set()
            worker.join(timeout=2.0)
            self.assertIn("context", result)
            self.assertEqual(result["context"].durable_context, "prepared:Wie geht es Janina?:1")
            self.assertEqual(len(retriever.provider_calls), 1)
        finally:
            retriever.release.set()
            service.shutdown()

    def test_confirm_memory_invalidates_prepared_provider_context_front(self) -> None:
        remote_state = _CountingRemoteState()
        retriever = _ProbeCachingRetriever(remote_state)
        service = object.__new__(LongTermMemoryService)
        service.config = TwinrConfig(
            project_root=".",
            personality_dir="personality",
            long_term_memory_enabled=True,
            long_term_memory_fast_topic_enabled=False,
        )
        service.query_rewriter = _StaticQueryRewriter({})
        service.retriever = retriever
        service.object_store = SimpleNamespace(
            remote_state=remote_state,
            load_conflicts_for_memory_ids=lambda memory_ids: (),
            confirm_object=lambda memory_id: {"memory_id": memory_id},
            apply_memory_mutation=lambda result: None,
        )
        service.prompt_context_store = SimpleNamespace(
            memory_store=SimpleNamespace(remote_state=None),
            user_store=SimpleNamespace(remote_state=None),
            personality_store=SimpleNamespace(remote_state=None),
        )
        service.graph_store = SimpleNamespace(remote_state=None)
        service.midterm_store = SimpleNamespace(remote_state=None)
        service.writer = None
        service.multimodal_writer = None
        service.fast_topic_builder = None
        service.prepared_context_front = PreparedLongTermContextFront()
        service._refresh_restart_recall_packets_locked = lambda: None
        service._store_lock = threading.RLock()
        build_calls: list[str] = []

        def _build_context(*, query: LongTermQueryProfile, original_query_text: str | None = None) -> LongTermMemoryContext:
            del original_query_text
            build_calls.append(query.retrieval_text)
            return LongTermMemoryContext(durable_context=f"durable:{len(build_calls)}")

        retriever.build_context = _build_context  # type: ignore[method-assign]
        try:
            first = service.build_provider_context("Wer ist Janina?")
            second = service.build_provider_context("Wer ist Janina?")
            self.assertEqual(first.durable_context, "durable:1")
            self.assertEqual(second.durable_context, "durable:1")
            self.assertEqual(build_calls, ["Wer ist Janina?"])

            service.confirm_memory(memory_id="fact:janina")

            third = service.build_provider_context("Wer ist Janina?")
            self.assertEqual(third.durable_context, "durable:2")
            self.assertEqual(build_calls, ["Wer ist Janina?", "Wer ist Janina?"])
        finally:
            service.shutdown()

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

    def test_build_tool_provider_context_records_latest_snapshot(self) -> None:
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
        service._store_lock = threading.RLock()

        context = service.build_tool_provider_context("na alles klar")
        snapshot = service.latest_context_snapshot(profile="tool")

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot.profile, "tool")
        self.assertEqual(snapshot.query_profile.retrieval_text, "na alles klar")
        self.assertEqual(snapshot.context, context)
        self.assertEqual(snapshot.source, "built_sync")

    def test_longterm_retriever_build_tool_context_reuses_loaded_inputs(self) -> None:
        retriever = LongTermRetriever(
            config=TwinrConfig(project_root=".", personality_dir="personality"),
            prompt_context_store=SimpleNamespace(),
            graph_store=SimpleNamespace(render_prompt_context_selection=lambda *_args, **_kwargs: "graph:redacted"),
            object_store=SimpleNamespace(),
            midterm_store=SimpleNamespace(),
            conflict_resolver=SimpleNamespace(),
            subtext_builder=SimpleNamespace(),
        )
        safe_object = SimpleNamespace(
            memory_id="fact:safe",
            kind="fact",
            attributes={},
        )
        conflicting_object = SimpleNamespace(
            memory_id="fact:conflicting",
            kind="fact",
            attributes={},
        )
        contact_method_object = SimpleNamespace(
            memory_id="fact:contact",
            kind="fact",
            attributes={"fact_type": "contact_method"},
        )
        context_inputs = _LongTermContextInputs(
            episodic_entries=[],
            midterm_packets=("midterm-packet",),
            adaptive_packets=("adaptive-packet",),
            durable_objects=(safe_object, conflicting_object, contact_method_object),
            conflict_queue=(
                SimpleNamespace(options=(SimpleNamespace(memory_id="fact:conflicting"),)),
            ),
            graph_selection=SimpleNamespace(document="graph-doc", query_plan={"source": "test"}),
            graph_context="graph:full",
            unified_query_plan={"source": "test"},
        )
        query = LongTermQueryProfile.from_text("Worueber haben wir heute gesprochen?")

        with (
            patch.object(LongTermRetriever, "_load_context_inputs", return_value=context_inputs) as load_mock,
            patch.object(LongTermRetriever, "_render_durable_context", return_value="durable:redacted") as durable_mock,
            patch.object(LongTermRetriever, "_render_episodic_context", return_value="episodic") as episodic_mock,
            patch.object(LongTermRetriever, "_render_conflict_context", return_value="conflicts") as conflict_mock,
            patch.object(LongTermRetriever, "_render_midterm_context", return_value="midterm") as midterm_mock,
            patch.object(LongTermRetriever, "_build_subtext_context", return_value="subtext") as subtext_mock,
        ):
            context = retriever.build_tool_context(query=query, original_query_text=query.original_text)

        self.assertEqual(load_mock.call_count, 1)
        self.assertEqual(
            load_mock.call_args.kwargs,
            {
                "query_texts": ("Worueber haben wir heute gesprochen?",),
                "retrieval_text": "Worueber haben wir heute gesprochen?",
                "include_graph": False,
            },
        )
        self.assertEqual(durable_mock.call_count, 1)
        self.assertEqual(durable_mock.call_args.args[0], (safe_object,))
        self.assertEqual(episodic_mock.call_count, 1)
        self.assertEqual(conflict_mock.call_count, 0)
        self.assertEqual(midterm_mock.call_count, 1)
        self.assertEqual(subtext_mock.call_count, 0)
        self.assertEqual(context.durable_context, "durable:redacted")
        self.assertIsNone(context.graph_context)
        self.assertIsNone(context.subtext_context)
        self.assertIsNone(context.conflict_context)

    def test_longterm_retriever_build_tool_context_uses_graph_only_as_empty_fallback(self) -> None:
        retriever = LongTermRetriever(
            config=TwinrConfig(project_root=".", personality_dir="personality"),
            prompt_context_store=SimpleNamespace(),
            graph_store=SimpleNamespace(),
            object_store=SimpleNamespace(),
            midterm_store=SimpleNamespace(),
            conflict_resolver=SimpleNamespace(),
            subtext_builder=SimpleNamespace(),
        )
        context_inputs = _LongTermContextInputs(
            episodic_entries=[],
            midterm_packets=(),
            adaptive_packets=(),
            durable_objects=(),
            conflict_queue=(),
            graph_selection=None,
            graph_context=None,
            unified_query_plan={"source": "test"},
        )
        query = LongTermQueryProfile.from_text("Worueber haben wir heute gesprochen?")

        with (
            patch.object(LongTermRetriever, "_load_context_inputs", return_value=context_inputs),
            patch.object(LongTermRetriever, "_build_graph_context", return_value="graph:fallback") as graph_mock,
            patch.object(LongTermRetriever, "_render_durable_context", return_value=None),
            patch.object(LongTermRetriever, "_render_episodic_context", return_value=None),
            patch.object(LongTermRetriever, "_render_midterm_context", return_value=None),
            patch.object(LongTermRetriever, "_build_subtext_context", return_value="subtext") as subtext_mock,
        ):
            context = retriever.build_tool_context(query=query, original_query_text=query.original_text)

        self.assertEqual(graph_mock.call_count, 1)
        self.assertEqual(graph_mock.call_args.args[0], "Worueber haben wir heute gesprochen?")
        self.assertEqual(subtext_mock.call_count, 0)
        self.assertEqual(context.graph_context, "graph:fallback")
        self.assertIsNone(context.subtext_context)

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

    def test_persist_longterm_turn_uses_fine_grained_current_state_without_snapshot_blob_reads(self) -> None:
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
            turn = LongTermConversationTurn(
                transcript="Today is warm and Janina has an appointment.",
                response="I will remember Janina's appointment.",
            )
            try:
                original_load_objects = service.object_store.load_objects
                original_load_conflicts = service.object_store.load_conflicts
                original_load_archived_objects = service.object_store.load_archived_objects
                original_write_snapshot = service.object_store.write_snapshot
                store_type = type(service.object_store)

                with patch.object(
                    store_type,
                    "load_objects",
                    side_effect=AssertionError("Conversation persistence must not hydrate the full object snapshot."),
                ), patch.object(
                    store_type,
                    "load_conflicts",
                    side_effect=AssertionError("Conversation persistence must not hydrate the full conflict snapshot."),
                ), patch.object(
                    store_type,
                    "load_archived_objects",
                    side_effect=AssertionError("Conversation persistence must not hydrate the full archive snapshot."),
                ), patch.object(
                    store_type,
                    "load_active_working_set",
                    new=lambda _self, **_kwargs: SimpleNamespace(
                        objects=original_load_objects(),
                        conflicts=original_load_conflicts(),
                        archived_objects=original_load_archived_objects(),
                    ),
                ), patch.object(
                    store_type,
                    "commit_active_delta",
                    new=lambda _self, **kwargs: original_write_snapshot(
                        objects=tuple(kwargs.get("object_upserts", ())),
                        conflicts=tuple(kwargs.get("conflict_upserts", ())),
                        archived_objects=tuple(kwargs.get("archive_upserts", ())),
                    ),
                ), patch.object(
                    store_type,
                    "write_snapshot",
                    side_effect=AssertionError("Conversation persistence must not rewrite the full current state."),
                ):
                    LongTermMemoryService._persist_longterm_turn(
                        config=config,
                        store=service.prompt_context_store,
                        graph_store=service.graph_store,
                        object_store=service.object_store,
                        midterm_store=service.midterm_store,
                        extractor=service.extractor,
                        consolidator=service.consolidator,
                        reflector=service.reflector,
                        sensor_memory=service.sensor_memory,
                        retention_policy=service.retention_policy,
                        item=turn,
                    )
                    stored_objects = original_load_objects()
            finally:
                service.shutdown()

        self.assertTrue(any(item.kind == "episode" for item in stored_objects))

    def test_persist_multimodal_evidence_uses_fine_grained_current_state_without_snapshot_blob_reads(self) -> None:
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
            evidence = LongTermMultimodalEvidence(
                event_name="print_completed",
                modality="printer",
                source="realtime_print",
                message="Printed Twinr output was delivered from the realtime loop.",
                data={"request_source": "button", "queue": "Thermal_GP58"},
            )
            try:
                original_load_objects = service.object_store.load_objects
                original_load_conflicts = service.object_store.load_conflicts
                original_load_archived_objects = service.object_store.load_archived_objects
                original_write_snapshot = service.object_store.write_snapshot
                store_type = type(service.object_store)

                with patch.object(
                    store_type,
                    "load_objects",
                    side_effect=AssertionError("Multimodal persistence must not hydrate the full object snapshot."),
                ), patch.object(
                    store_type,
                    "load_conflicts",
                    side_effect=AssertionError("Multimodal persistence must not hydrate the full conflict snapshot."),
                ), patch.object(
                    store_type,
                    "load_archived_objects",
                    side_effect=AssertionError("Multimodal persistence must not hydrate the full archive snapshot."),
                ), patch.object(
                    store_type,
                    "load_active_working_set",
                    new=lambda _self, **_kwargs: SimpleNamespace(
                        objects=original_load_objects(),
                        conflicts=original_load_conflicts(),
                        archived_objects=original_load_archived_objects(),
                    ),
                ), patch.object(
                    store_type,
                    "commit_active_delta",
                    new=lambda _self, **kwargs: original_write_snapshot(
                        objects=tuple(kwargs.get("object_upserts", ())),
                        conflicts=tuple(kwargs.get("conflict_upserts", ())),
                        archived_objects=tuple(kwargs.get("archive_upserts", ())),
                    ),
                ), patch.object(
                    store_type,
                    "write_snapshot",
                    side_effect=AssertionError("Multimodal persistence must not rewrite the full current state."),
                ):
                    LongTermMemoryService._persist_multimodal_evidence(
                        object_store=service.object_store,
                        midterm_store=service.midterm_store,
                        multimodal_extractor=service.multimodal_extractor,
                        consolidator=service.consolidator,
                        reflector=service.reflector,
                        sensor_memory=service.sensor_memory,
                        retention_policy=service.retention_policy,
                        store_lock=service._store_lock,
                        timezone_name=service.config.local_timezone_name,
                        item=evidence,
                    )
                    stored_objects = original_load_objects()
            finally:
                service.shutdown()

        self.assertTrue(any(item.kind == "episode" for item in stored_objects))

    def test_persist_longterm_turn_records_personality_learning_from_consolidation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
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

    def test_persist_longterm_turn_bootstraps_fresh_remote_primary_object_namespace(self) -> None:
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
            remote_state = _FakeRemoteState()
            object_store = LongTermStructuredStore(
                base_path=Path(temp_dir) / "state" / "chonkydb",
                remote_state=remote_state,
            )
            midterm_store = LongTermMidtermStore(
                base_path=Path(temp_dir) / "state" / "chonkydb",
                remote_state=remote_state,
            )
            helper = LongTermMemoryService.from_config(config, extractor=make_test_extractor())
            turn = LongTermConversationTurn(
                transcript="Lea bringt heute Abend Suppe vorbei.",
                response="Ich merke mir Lea und die Suppe.",
            )
            try:
                result = LongTermMemoryService._persist_longterm_turn(
                    config=config,
                    store=prompt_context_store,
                    graph_store=graph_store,
                    object_store=object_store,
                    midterm_store=midterm_store,
                    extractor=make_test_extractor(),
                    consolidator=helper.consolidator,
                    reflector=helper.reflector,
                    sensor_memory=helper.sensor_memory,
                    retention_policy=helper.retention_policy,
                    personality_learning=None,
                    item=turn,
                )
                remote_catalog = object_store._remote_catalog
                object_entries = (
                    remote_catalog.load_catalog_entries(snapshot_kind="objects")
                    if remote_catalog is not None
                    else ()
                )
            finally:
                helper.shutdown(timeout_s=1.0)

        self.assertIsNone(result)
        self.assertEqual(len(object_entries), 1)
        self.assertTrue(object_entries[0].item_id.startswith("episode:turn"))

    def test_persist_longterm_turn_passes_reflection_thread_summaries_to_personality_learning(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
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

    def test_persist_longterm_turn_preserves_text_channel_provenance_for_memory_and_personality(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
            )
            service = LongTermMemoryService.from_config(config, extractor=make_test_extractor())
            personality_learning = _RecordingPersonalityLearningService()
            turn = LongTermConversationTurn(
                transcript="Janina hat mir bei WhatsApp geschrieben, dass der Arzttermin verschoben wurde.",
                response="Ich merke mir die WhatsApp-Nachricht zu Janinas Termin.",
                source="whatsapp",
                modality="text",
            )
            try:
                LongTermMemoryService._persist_longterm_turn(
                    config=config,
                    store=service.prompt_context_store,
                    graph_store=service.graph_store,
                    object_store=service.object_store,
                    midterm_store=service.midterm_store,
                    extractor=service.extractor,
                    consolidator=service.consolidator,
                    reflector=service.reflector,
                    sensor_memory=service.sensor_memory,
                    retention_policy=service.retention_policy,
                    personality_learning=personality_learning,
                    item=turn,
                )
                stored_objects = service.object_store.load_objects()
            finally:
                service.shutdown()

        self.assertEqual(len(personality_learning.conversation_calls), 1)
        recorded_turn, _recorded_result = personality_learning.conversation_calls[0]
        self.assertEqual(recorded_turn.source, "whatsapp")
        self.assertEqual(recorded_turn.modality, "text")
        episode = next(item for item in stored_objects if item.kind == "episode")
        self.assertEqual(episode.source.source_type, "whatsapp")
        self.assertEqual(episode.source.modality, "text")
        self.assertEqual(episode.attributes["request_source"], "whatsapp")
        self.assertEqual(episode.attributes["input_modality"], "text")

    def test_import_external_conversation_turn_preserves_timestamp_without_personality_learning(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
            )
            service = LongTermMemoryService.from_config(config, extractor=make_test_extractor())
            personality_learning = _RecordingPersonalityLearningService()
            service.personality_learning = personality_learning
            occurred_at = datetime(2026, 3, 1, 9, 30, tzinfo=timezone.utc)
            try:
                service.import_external_conversation_turn(
                    transcript="Ich komme heute etwas später.",
                    response="WhatsApp contact Anna replied: Alles klar.",
                    source="whatsapp_history",
                    modality="text",
                    created_at=occurred_at,
                )
                service.run_retention()
                stored_objects = service.object_store.load_objects()
            finally:
                service.shutdown()

        self.assertEqual(personality_learning.conversation_calls, [])
        episode = next(item for item in stored_objects if item.kind == "episode")
        self.assertEqual(episode.source.source_type, "whatsapp_history")
        self.assertEqual(episode.attributes["request_source"], "whatsapp_history")
        self.assertEqual(episode.attributes["input_modality"], "text")
        self.assertEqual(episode.attributes["retention_policy"], "preserve")
        self.assertEqual(episode.created_at.date().isoformat(), "2026-03-01")

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

    def test_enqueue_personality_tool_history_routes_tool_calls_to_queueing_path(self) -> None:
        service = object.__new__(LongTermMemoryService)
        service.personality_learning = _RecordingPersonalityLearningService()
        service._store_lock = _FailOnEnterLock()
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

        service.enqueue_personality_tool_history(
            tool_calls=(tool_call,),
            tool_results=(tool_result,),
        )

        self.assertEqual(len(service.personality_learning.queued_tool_history_calls), 1)
        self.assertEqual(service.personality_learning.tool_history_calls, [])

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
        self.assertEqual(len(messages), 5)
        self.assertIn("Silent personalization background for this turn.", messages[0])
        self.assertIn("twinr_silent_personalization_context_v1", messages[0])
        self.assertIn("twinr_fast_topic_context_v1", messages[1])
        self.assertIn("current thread hint", messages[1])
        self.assertIn("twinr_long_term_midterm_context_v1", messages[2])
        self.assertIn("Heute wollte ich spazieren gehen", messages[2])
        self.assertIn("Structured long-term episodic memory for this turn.", messages[3])
        self.assertIn("Heute wollte ich spazieren gehen", messages[3])
        self.assertIn("twinr_graph_memory_context_v1", messages[4])
        self.assertIn("Melitta", messages[4])

    def test_fast_provider_context_builds_compact_topic_hints(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                long_term_memory_fast_topic_enabled=True,
                long_term_memory_fast_topic_limit=2,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                user_display_name="Erika",
            )
            service = LongTermMemoryService.from_config(config, extractor=make_test_extractor())
            service.object_store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:janina_spouse",
                        kind="relationship_fact",
                        summary="Janina is the user's wife.",
                        source=self._source("turn:fast:1"),
                        status="active",
                        confidence=0.98,
                        confirmed_by_user=True,
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="episode:janina_doctor",
                        kind="episode",
                        summary="Janina had an eye doctor appointment yesterday.",
                        source=self._source("turn:fast:2"),
                        status="active",
                        confidence=0.94,
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="fact:tea_preference",
                        kind="preference_fact",
                        summary="The user likes black tea.",
                        source=self._source("turn:fast:3"),
                        status="active",
                        confidence=0.9,
                    ),
                )
            )
            service.query_rewriter = _StaticQueryRewriter({})  # type: ignore[assignment]

            context = service.build_fast_provider_context("Was weisst du ueber Janina?")
            service.shutdown()

        self.assertIsNotNone(context.topic_context)
        self.assertIn("twinr_fast_topic_context_v1", context.topic_context or "")
        self.assertIn("confirmed relationship hint", context.topic_context or "")
        self.assertIn("current thread hint", context.topic_context or "")
        self.assertIn("Janina is the user's wife.", context.topic_context or "")
        self.assertIn("Janina had an eye doctor appointment yesterday.", context.topic_context or "")
        self.assertNotIn("black tea", context.topic_context or "")

    def test_provider_context_includes_quick_memory_topic_hints_before_main_recall(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                long_term_memory_fast_topic_enabled=True,
                long_term_memory_fast_topic_limit=2,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                user_display_name="Erika",
            )
            service = LongTermMemoryService.from_config(config, extractor=make_test_extractor())
            service.object_store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:janina_spouse",
                        kind="relationship_fact",
                        summary="Janina is the user's wife.",
                        source=self._source("turn:provider-fast:1"),
                        status="active",
                        confidence=0.98,
                        confirmed_by_user=True,
                    ),
                    LongTermMemoryObjectV1(
                        memory_id="episode:janina_doctor",
                        kind="episode",
                        summary="Janina had an eye doctor appointment yesterday.",
                        source=self._source("turn:provider-fast:2"),
                        status="active",
                        confidence=0.94,
                    ),
                )
            )
            service.query_rewriter = _StaticQueryRewriter({})  # type: ignore[assignment]

            context = service.build_provider_context("Was weisst du ueber Janina?")
            service.shutdown()

        self.assertIsNotNone(context.topic_context)
        self.assertIn("Janina is the user's wife.", context.topic_context or "")
        self.assertIn("Janina had an eye doctor appointment yesterday.", context.topic_context or "")
        messages = context.system_messages()
        self.assertGreaterEqual(len(messages), 2)
        self.assertEqual(messages[0], context.topic_context)
        self.assertIn("twinr_long_term_midterm_context_v1", messages[1])
        self.assertTrue(any(message is not context.topic_context for message in messages[1:]))

    def test_fast_provider_context_skips_query_rewrite_network_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                long_term_memory_fast_topic_enabled=True,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                user_display_name="Erika",
            )
            service = LongTermMemoryService.from_config(config, extractor=make_test_extractor())
            service.object_store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:janina_spouse",
                        kind="relationship_fact",
                        summary="Janina is the user's wife.",
                        source=self._source("turn:fast:no-rewrite"),
                        status="active",
                        confidence=0.98,
                    ),
                )
            )

            class _ExplodingRewriter:
                def profile(self, query_text):
                    del query_text
                    raise AssertionError("fast provider context must not invoke the normal query rewriter")

            service.query_rewriter = _ExplodingRewriter()  # type: ignore[assignment]
            context = service.build_fast_provider_context("Was weisst du ueber Janina?")
            service.shutdown()

        self.assertIsNotNone(context.topic_context)
        self.assertIn("Janina is the user's wife.", context.topic_context or "")

    def test_fast_provider_context_skips_probe_read_cache_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                long_term_memory_fast_topic_enabled=True,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                user_display_name="Erika",
            )
            service = LongTermMemoryService.from_config(config, extractor=make_test_extractor())
            service.object_store.write_snapshot(
                objects=(
                    LongTermMemoryObjectV1(
                        memory_id="fact:janina_spouse",
                        kind="relationship_fact",
                        summary="Janina is the user's wife.",
                        source=self._source("turn:fast:no-probe-cache"),
                        status="active",
                        confidence=0.98,
                    ),
                )
            )

            @contextmanager
            def _fail_probe_cache():
                raise AssertionError("fast provider context must not wrap the hot path in probe-read caching")
                yield

            with patch.object(
                LongTermMemoryService,
                "_temporary_remote_probe_cache",
                _fail_probe_cache,
            ):
                context = service.build_fast_provider_context("Was weisst du ueber Janina?")
            service.shutdown()

        self.assertIsNotNone(context.topic_context)
        self.assertIn("Janina is the user's wife.", context.topic_context or "")

    def test_fast_provider_context_raises_remote_read_failed_when_fast_topic_builder_times_out(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                long_term_memory_fast_topic_enabled=True,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                user_display_name="Erika",
            )
            service = LongTermMemoryService.from_config(config, extractor=make_test_extractor())

            class _TimeoutingFastTopicBuilder:
                def build(self, *, query_profile):
                    del query_profile
                    raise ChonkyDBError("timed out")

            service.fast_topic_builder = _TimeoutingFastTopicBuilder()  # type: ignore[assignment]
            with self.assertRaises(LongTermRemoteReadFailedError) as raised:
                service.build_fast_provider_context("Was weisst du ueber Janina?")
            service.shutdown()

        self.assertEqual(dict(raised.exception.details), {
            "operation": "fast_provider_context",
            "request_kind": "read",
            "outcome": "failed",
            "classification": "unexpected_error",
            "error_type": "ChonkyDBError",
            "error_message": "timed out",
        })

    def test_provider_context_skips_quick_memory_when_fast_topic_builder_times_out(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                long_term_memory_fast_topic_enabled=True,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                user_display_name="Erika",
            )
            service = LongTermMemoryService.from_config(config, extractor=make_test_extractor())

            class _TimeoutingFastTopicBuilder:
                def build(self, *, query_profile):
                    del query_profile
                    raise ChonkyDBError("timed out")

            service.fast_topic_builder = _TimeoutingFastTopicBuilder()  # type: ignore[assignment]
            context = service.build_provider_context("Was weisst du ueber Janina?")
            service.shutdown()

        self.assertIsNone(context.topic_context)

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
        self.assertIn('"confirmation_state": "explicitly_confirmed_by_user"', context.durable_context or "")
        self.assertIn("prefer active user-confirmed facts", context.durable_context or "")
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

        self.assertIsNone(context.topic_context)
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

    def test_prompt_context_mutations_do_not_wait_on_shared_store_lock(self) -> None:
        service = object.__new__(LongTermMemoryService)
        service.prompt_context_store = SimpleNamespace(
            memory_store=SimpleNamespace(
                remember=lambda **kwargs: ("remember", kwargs),
                delete=lambda **kwargs: ("delete", kwargs),
            ),
            user_store=SimpleNamespace(
                upsert=lambda **kwargs: ("user_upsert", kwargs),
                delete=lambda **kwargs: ("user_delete", kwargs),
            ),
            personality_store=SimpleNamespace(
                upsert=lambda **kwargs: ("personality_upsert", kwargs),
                delete=lambda **kwargs: ("personality_delete", kwargs),
            ),
        )
        service._store_lock = _FailOnEnterLock()

        self.assertEqual(
            service.store_explicit_memory(kind="fact", summary="summary", details="details"),
            ("remember", {"kind": "fact", "summary": "summary", "details": "details"}),
        )
        self.assertEqual(
            service.delete_explicit_memory(entry_id="MEM-1"),
            ("delete", {"entry_id": "MEM-1"}),
        )
        self.assertEqual(
            service.update_user_profile(category="nickname", instruction="Use Erika."),
            ("user_upsert", {"category": "nickname", "instruction": "Use Erika."}),
        )
        self.assertEqual(
            service.remove_user_profile(category="nickname"),
            ("user_delete", {"category": "nickname"}),
        )
        self.assertEqual(
            service.update_personality(category="tone", instruction="Stay calm."),
            ("personality_upsert", {"category": "tone", "instruction": "Stay calm."}),
        )
        self.assertEqual(
            service.remove_personality(category="tone"),
            ("personality_delete", {"category": "tone"}),
        )

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

    def test_service_analyze_conversation_turn_uses_active_working_set_without_snapshot_blob_reads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                user_display_name="Erika",
            )
            service = LongTermMemoryService.from_config(config, extractor=make_test_extractor())
            try:
                original_load_objects = service.object_store.load_objects
                store_type = type(service.object_store)

                with patch.object(
                    store_type,
                    "load_objects",
                    side_effect=AssertionError("Conversation analysis must not hydrate the full object snapshot."),
                ), patch.object(
                    store_type,
                    "load_active_working_set",
                    new=lambda _self, **_kwargs: SimpleNamespace(
                        objects=original_load_objects(),
                        conflicts=(),
                        archived_objects=(),
                    ),
                ):
                    result = service.analyze_conversation_turn(
                        transcript=(
                            "Today is a beautiful Sunday, it is really warm. "
                            "My wife Janina is at the eye doctor and is getting eye laser treatment."
                        ),
                        response="I hope Janina's appointment goes smoothly.",
                    )
            finally:
                service.shutdown()

        self.assertFalse(result.clarification_needed)
        self.assertTrue(result.durable_objects)

    def test_service_analyze_multimodal_evidence_uses_active_working_set_without_snapshot_blob_reads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                user_display_name="Erika",
            )
            service = LongTermMemoryService.from_config(config, extractor=make_test_extractor())
            try:
                original_load_objects = service.object_store.load_objects
                store_type = type(service.object_store)

                with patch.object(
                    store_type,
                    "load_objects",
                    side_effect=AssertionError("Multimodal analysis must not hydrate the full object snapshot."),
                ), patch.object(
                    store_type,
                    "load_active_working_set",
                    new=lambda _self, **_kwargs: SimpleNamespace(
                        objects=original_load_objects(),
                        conflicts=(),
                        archived_objects=(),
                    ),
                ):
                    result = service.analyze_multimodal_evidence(
                        event_name="sensor_observation",
                        modality="sensor",
                        source="proactive_monitor",
                        message="Changed multimodal sensor observation recorded.",
                        data={
                            "facts": {
                                "pir": {"motion_detected": True},
                                "camera": {"person_visible": True},
                            },
                            "event_names": ["pir.motion_detected", "camera.person_visible"],
                        },
                    )
            finally:
                service.shutdown()

        self.assertTrue(result.episodic_objects or result.durable_objects or result.deferred_objects)

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
                def maybe_refresh_world_intelligence(self, *, search_backend=None) -> None:
                    del self
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
                def maybe_refresh_world_intelligence(self, *, search_backend=None) -> None:
                    del self
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

    def test_service_run_retention_uses_projection_selector_without_snapshot_blob_reads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                user_display_name="Erika",
            )
            service = LongTermMemoryService.from_config(config, extractor=make_test_extractor())
            try:
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
                service.object_store.write_snapshot(objects=tuple(rewritten))

                original_load_objects = service.object_store.load_objects
                original_load_archived_objects = service.object_store.load_archived_objects
                original_write_snapshot = service.object_store.write_snapshot
                store_type = type(service.object_store)
                service.object_store._remote_catalog = object()

                with patch.object(
                    store_type,
                    "load_objects",
                    side_effect=AssertionError("Retention must not hydrate the full object snapshot."),
                ), patch.object(
                    store_type,
                    "load_objects_fine_grained",
                    side_effect=AssertionError("Retention must not sweep the full fine-grained object state."),
                ), patch.object(
                    store_type,
                    "load_archived_objects",
                    side_effect=AssertionError("Retention must not hydrate the full archive snapshot."),
                ), patch.object(
                    store_type,
                    "load_archived_objects_fine_grained",
                    new=lambda _self: original_load_archived_objects(),
                ), patch.object(
                    store_type,
                    "_remote_catalog_enabled",
                    new=lambda _self: True,
                ), patch.object(
                    store_type,
                    "load_objects_by_projection_filter",
                    new=lambda _self, *, predicate: tuple(
                        item
                        for item in original_load_objects()
                        if predicate(
                            _self._remote_object_selection_projection(
                                snapshot_kind="objects",
                                payload=item.to_payload(),
                            )
                        )
                    ),
                ), patch.object(
                    store_type,
                    "commit_active_delta",
                    new=lambda _self, **kwargs: original_write_snapshot(
                        objects=tuple(kwargs.get("object_upserts", ())),
                        conflicts=(),
                        archived_objects=tuple(kwargs.get("archive_upserts", ())),
                    ),
                ):
                    service.run_retention()
                    kept = original_load_objects()
                    archived = original_load_archived_objects()
            finally:
                service.shutdown()

        kept_ids = {item.memory_id for item in kept}
        archived_ids = {item.memory_id for item in archived}
        self.assertNotIn(old_episode.memory_id, kept_ids)
        self.assertNotIn(old_observation.memory_id, kept_ids)
        self.assertIn(old_episode.memory_id, archived_ids)
        self.assertNotIn(old_observation.memory_id, archived_ids)

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
            original_load_objects = service.object_store.load_objects
            store_type = type(service.object_store)
            with patch.object(
                store_type,
                "load_objects",
                side_effect=AssertionError("review_memory must not hydrate the full object snapshot."),
            ), patch.object(
                store_type,
                "load_objects_fine_grained",
                new=lambda _self: original_load_objects(),
            ):
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
            source_ids_before = {
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

    def test_backfill_ops_history_uses_fine_grained_state_loaders_without_snapshot_blob_reads(self) -> None:
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
            try:
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

                original_load_objects = service.object_store.load_objects
                original_load_conflicts = service.object_store.load_conflicts
                original_load_archived_objects = service.object_store.load_archived_objects
                original_write_snapshot = service.object_store.write_snapshot
                store_type = type(service.object_store)

                with patch.object(
                    store_type,
                    "load_objects",
                    side_effect=AssertionError("Backfill must not hydrate the full object snapshot."),
                ), patch.object(
                    store_type,
                    "load_conflicts",
                    side_effect=AssertionError("Backfill must not hydrate the full conflict snapshot."),
                ), patch.object(
                    store_type,
                    "load_archived_objects",
                    side_effect=AssertionError("Backfill must not hydrate the full archive snapshot."),
                ), patch.object(
                    store_type,
                    "load_active_working_set",
                    new=lambda _self, **_kwargs: SimpleNamespace(
                        objects=original_load_objects(),
                        conflicts=original_load_conflicts(),
                        archived_objects=original_load_archived_objects(),
                    ),
                ), patch.object(
                    store_type,
                    "commit_active_delta",
                    new=lambda _self, **kwargs: original_write_snapshot(
                        objects=tuple(kwargs.get("object_upserts", ())),
                        conflicts=tuple(kwargs.get("conflict_upserts", ())),
                        archived_objects=tuple(kwargs.get("archive_upserts", ())),
                    ),
                ), patch.object(
                    store_type,
                    "write_snapshot",
                    side_effect=AssertionError("Backfill must not rewrite the full current state."),
                ):
                    result = service.backfill_ops_multimodal_history(
                        entries=entries,
                        now=datetime(2026, 3, 10, 10, 0, tzinfo=timezone.utc),
                    )
                    objects = {item.memory_id: item for item in original_load_objects()}
            finally:
                service.shutdown()

        self.assertGreater(result.applied_evidence, 0)
        self.assertIn("pattern:presence:morning:near_device", objects)
        self.assertIn("pattern:button:green:start_listening:morning", objects)

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
                            summary=f"Corinna Maier can be reached at {_TEST_CORINNA_PHONE_OLD}.",
                            source=self._source("turn:1"),
                            status="active",
                            confidence=0.95,
                            slot_key="contact:person:corinna_maier:phone",
                            value_key=_TEST_CORINNA_PHONE_OLD,
                            attributes={"person_ref": "person:corinna_maier"},
                        ),
                    ),
                    deferred_objects=(
                        LongTermMemoryObjectV1(
                            memory_id="fact:corinna_phone_new",
                            kind="contact_method_fact",
                            summary=f"Corinna Maier can be reached at {_TEST_CORINNA_PHONE_NEW}.",
                            source=self._source("turn:2"),
                            status="uncertain",
                            confidence=0.92,
                            slot_key="contact:person:corinna_maier:phone",
                            value_key=_TEST_CORINNA_PHONE_NEW,
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
            original_load_objects = service.object_store.load_objects
            original_load_conflicts = service.object_store.load_conflicts
            store_type = type(service.object_store)
            with patch.object(
                store_type,
                "load_objects",
                side_effect=AssertionError("confirm_memory conflict resolution must not hydrate the full object snapshot."),
            ), patch.object(
                store_type,
                "load_conflicts",
                side_effect=AssertionError("confirm_memory conflict resolution must not hydrate the full conflict snapshot."),
            ), patch.object(
                store_type,
                "load_objects_fine_grained",
                new=lambda _self: original_load_objects(),
            ), patch.object(
                store_type,
                "load_conflicts_fine_grained",
                side_effect=AssertionError("confirm_memory conflict resolution must not sweep the full conflict state."),
            ), patch.object(
                store_type,
                "load_conflicts_for_memory_ids",
                new=lambda _self, memory_ids: tuple(
                    conflict
                    for conflict in original_load_conflicts()
                    if conflict.candidate_memory_id in set(memory_ids)
                    or bool(set(conflict.existing_memory_ids).intersection(memory_ids))
                ),
            ), patch.object(
                store_type,
                "load_objects_fine_grained_for_write",
                new=lambda _self: original_load_objects(),
            ), patch.object(
                store_type,
                "load_conflicts_fine_grained_for_write",
                new=lambda _self: original_load_conflicts(),
            ), patch.object(
                store_type,
                "load_archived_objects_fine_grained_for_write",
                new=lambda _self: (),
            ):
                resolution = service.confirm_memory(memory_id=queue_before[0].candidate_memory_id)
                objects = {item.memory_id: item for item in original_load_objects()}
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
            original_load_objects = service.object_store.load_objects
            original_load_conflicts = service.object_store.load_conflicts
            store_type = type(service.object_store)
            with patch.object(
                store_type,
                "load_objects",
                side_effect=AssertionError("invalidate/delete must not hydrate the full object snapshot."),
            ), patch.object(
                store_type,
                "load_conflicts",
                side_effect=AssertionError("invalidate/delete must not hydrate the full conflict snapshot."),
            ), patch.object(
                store_type,
                "load_objects_fine_grained",
                new=lambda _self: original_load_objects(),
            ), patch.object(
                store_type,
                "load_conflicts_fine_grained",
                side_effect=AssertionError("invalidate/delete must not sweep the full conflict state."),
            ), patch.object(
                store_type,
                "load_conflicts_for_memory_ids",
                new=lambda _self, memory_ids: tuple(
                    conflict
                    for conflict in original_load_conflicts()
                    if conflict.candidate_memory_id in set(memory_ids)
                    or bool(set(conflict.existing_memory_ids).intersection(memory_ids))
                ),
            ), patch.object(
                store_type,
                "load_objects_fine_grained_for_write",
                new=lambda _self: original_load_objects(),
            ), patch.object(
                store_type,
                "load_conflicts_fine_grained_for_write",
                new=lambda _self: original_load_conflicts(),
            ), patch.object(
                store_type,
                "load_archived_objects_fine_grained_for_write",
                new=lambda _self: (),
            ):
                invalidation = service.invalidate_memory(memory_id=event.memory_id, reason="This appointment was canceled.")
                deletion = service.delete_memory(memory_id=relationship.memory_id)
                objects = {item.memory_id: item for item in original_load_objects()}
            service.shutdown()

        self.assertEqual(invalidation.action, "invalidate")
        self.assertEqual(objects[event.memory_id].status, "invalid")
        self.assertEqual(objects[event.memory_id].attributes["invalidation_reason"], "This appointment was canceled.")
        self.assertEqual(deletion.action, "delete")
        self.assertNotIn(relationship.memory_id, objects)


class RuntimeMemoryFlushTimeoutTests(unittest.TestCase):
    def test_best_effort_flush_respects_caller_timeout_in_remote_primary_mode(self) -> None:
        config = TwinrConfig(
            openai_api_key="test-key",
            project_root=".",
            long_term_memory_mode="remote_primary",
            long_term_memory_remote_flush_timeout_s=60.0,
        )
        long_term_memory = _RecordingFlushService(flush_result=True)
        runtime = _RuntimeMemoryProbe(config=config, long_term_memory=long_term_memory)

        flushed = runtime.flush_long_term_memory(timeout_s=5.0)

        self.assertTrue(flushed)
        self.assertEqual(long_term_memory.flush_timeouts, [5.0])

    def test_strict_flush_keeps_remote_primary_minimum_timeout(self) -> None:
        config = TwinrConfig(
            openai_api_key="test-key",
            project_root=".",
            long_term_memory_mode="remote_primary",
            long_term_memory_remote_flush_timeout_s=60.0,
        )
        long_term_memory = _RecordingFlushService(flush_result=False)
        runtime = _RuntimeMemoryProbe(config=config, long_term_memory=long_term_memory)

        with self.assertRaises(TimeoutError):
            runtime._flush_long_term_memory_strict(operation="test", timeout_s=5.0)

        self.assertEqual(long_term_memory.flush_timeouts, [60.0])

    def test_prompt_context_runtime_writes_do_not_trigger_long_term_flush(self) -> None:
        config = TwinrConfig(
            openai_api_key="test-key",
            project_root=".",
            long_term_memory_mode="remote_primary",
            long_term_memory_remote_flush_timeout_s=60.0,
        )
        long_term_memory = _RecordingPromptContextMutationService()
        runtime = _RuntimeMemoryProbe(config=config, long_term_memory=long_term_memory)

        memory_entry = runtime.store_durable_memory(kind="fact", summary="summary", details="details")
        deleted_entry = runtime.delete_durable_memory_entry(entry_id="MEM-1")
        user_entry = runtime.update_user_profile_context(category="nickname", instruction="Use Erika.")
        removed_user_entry = runtime.remove_user_profile_context(category="nickname")
        personality_entry = runtime.update_personality_context(category="tone", instruction="Stay calm.")
        removed_personality_entry = runtime.remove_personality_context(category="tone")

        self.assertEqual(memory_entry.kind, "fact")
        self.assertEqual(deleted_entry.entry_id, "MEM-1")
        self.assertEqual(user_entry.key, "nickname")
        self.assertEqual(removed_user_entry.key, "nickname")
        self.assertEqual(personality_entry.key, "tone")
        self.assertEqual(removed_personality_entry.key, "tone")
        self.assertEqual(long_term_memory.flush_timeouts, [])
        self.assertEqual(
            [name for name, _args in long_term_memory.calls],
            [
                "store_explicit_memory",
                "delete_explicit_memory",
                "update_user_profile",
                "remove_user_profile",
                "update_personality",
                "remove_personality",
            ],
        )


if __name__ == "__main__":
    unittest.main()
