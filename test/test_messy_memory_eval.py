from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from typing import cast
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.chonkydb.client import ChonkyDBError
from twinr.memory.chonkydb.personal_graph import TwinrPersonalGraphStore
from twinr.memory.longterm.core.models import (
    LongTermMemoryConflictV1,
    LongTermMemoryObjectV1,
    LongTermMidtermPacketV1,
    LongTermSourceRefV1,
)
from twinr.memory.longterm.evaluation._unified_retrieval_shared import UnifiedRetrievalGoldsetCaseResult
from twinr.memory.longterm.evaluation.messy_memory_eval import (
    MessyMemoryEvalResult,
    MessyMemoryEvalSeedStats,
    MessyMemoryPhaseResult,
    MessyMemoryRestartSummary,
    _MaterializedMessyCorpus,
    _build_eval_runtime_config,
    _merge_midterm_packets,
    _merge_structured_current_states,
    _prepare_local_seed_service,
    _publish_materialized_messy_corpus,
    _run_phase,
    _seed_synthetic_graph_locally_then_sync_remote,
    _build_restart_summary,
    default_messy_memory_eval_path,
    run_messy_memory_eval,
    write_messy_memory_eval_artifacts,
)
from twinr.memory.longterm.evaluation.multimodal_eval import (
    MultimodalEvalCase,
    MultimodalEvalCaseResult,
    MultimodalEvalSummary,
)
from twinr.memory.longterm.evaluation.eval import LongTermEvalCase, LongTermEvalCaseResult, LongTermEvalSummary
from twinr.memory.longterm.ingestion.extract import LongTermStructuredTurnProgram, LongTermTurnExtractor
from twinr.memory.longterm.reasoning.midterm import LongTermStructuredReflectionProgram
from twinr.memory.longterm.reasoning.reflect import LongTermMemoryReflector
from twinr.memory.longterm.evaluation.unified_retrieval_benchmark import UnifiedRetrievalBenchmarkSummary
from twinr.memory.longterm.runtime.service import LongTermMemoryService
from twinr.memory.longterm.storage._structured_store.snapshots import StructuredStoreCurrentState


def _phase(
    *,
    phase: str,
    synthetic_results: tuple[LongTermEvalCaseResult, ...],
    multimodal_results: tuple[MultimodalEvalCaseResult, ...],
    unified_results: tuple[UnifiedRetrievalGoldsetCaseResult, ...],
    synthetic_passed: int,
    multimodal_passed: int,
    unified_passed: int,
) -> MessyMemoryPhaseResult:
    return MessyMemoryPhaseResult(
        phase=phase,
        synthetic_summary=LongTermEvalSummary(
            total_cases=len(synthetic_results),
            passed_cases=synthetic_passed,
            category_case_counts={"contact_exact_lookup": len(synthetic_results)},
            category_pass_counts={"contact_exact_lookup": synthetic_passed} if synthetic_passed else {},
        ),
        multimodal_summary=MultimodalEvalSummary(
            total_cases=len(multimodal_results),
            passed_cases=multimodal_passed,
            category_case_counts={"combined_context": len(multimodal_results)},
            category_pass_counts={"combined_context": multimodal_passed} if multimodal_passed else {},
        ),
        unified_summary=UnifiedRetrievalBenchmarkSummary(
            total_cases=len(unified_results),
            passed_cases=unified_passed,
            candidate_source_exact_cases=unified_passed,
            access_path_exact_cases=unified_passed,
            selected_id_exact_cases=unified_passed,
            join_anchor_exact_cases=unified_passed,
            source_precision_macro=1.0,
            source_recall_macro=1.0,
            access_path_precision_macro=1.0,
            access_path_recall_macro=1.0,
            selected_id_precision_macro=1.0,
            selected_id_recall_macro=1.0,
            join_anchor_precision_macro=1.0,
            join_anchor_recall_macro=1.0,
            path_safety_rate=1.0,
            overall_quality_score=1.0 if unified_passed == len(unified_results) else 0.75,
        ),
        synthetic_case_results=synthetic_results,
        multimodal_case_results=multimodal_results,
        unified_case_results=unified_results,
    )


class MessyMemoryEvalTests(unittest.TestCase):
    @staticmethod
    def _memory_object(memory_id: str) -> LongTermMemoryObjectV1:
        return LongTermMemoryObjectV1(
            memory_id=memory_id,
            kind="fact",
            summary=memory_id,
            source=LongTermSourceRefV1(source_type="test"),
        )

    @staticmethod
    def _memory_conflict(slot_key: str, candidate_memory_id: str) -> LongTermMemoryConflictV1:
        return LongTermMemoryConflictV1(
            slot_key=slot_key,
            candidate_memory_id=candidate_memory_id,
            existing_memory_ids=("existing:memory",),
            question="Clarify conflict?",
            reason="test",
        )

    def test_build_eval_runtime_config_disables_query_rewrite_and_subtext_compiler(self) -> None:
        @dataclass(frozen=True, slots=True)
        class _FakeConfig:
            long_term_memory_query_rewrite_enabled: bool = True
            long_term_memory_subtext_compiler_enabled: bool = True

        with patch(
            "twinr.memory.longterm.evaluation.messy_memory_eval._build_isolated_config",
            return_value=_FakeConfig(),
        ):
            config = _build_eval_runtime_config(
                base_config=cast(TwinrConfig, _FakeConfig()),
                base_project_root=Path("/tmp/project-root"),
                runtime_root=Path("/tmp/runtime-root"),
                remote_namespace="messy-test",
            )

        self.assertFalse(config.long_term_memory_query_rewrite_enabled)
        self.assertFalse(config.long_term_memory_subtext_compiler_enabled)

    def test_prepare_local_seed_service_replaces_frozen_extractor_and_reflector_programs(self) -> None:
        class _FakeService:
            def __init__(self) -> None:
                self.extractor = LongTermTurnExtractor(
                    timezone_name="Europe/Berlin",
                    program=cast(LongTermStructuredTurnProgram, object()),
                )
                self.reflector = LongTermMemoryReflector(
                    program=cast(LongTermStructuredReflectionProgram, object()),
                    midterm_packet_limit=4,
                )
                self.personality_learning = object()

        service = _FakeService()

        _prepare_local_seed_service(cast(LongTermMemoryService, service))

        self.assertIsNone(service.extractor.program)
        self.assertIsNone(service.reflector.program)
        self.assertEqual(service.reflector.midterm_packet_limit, 0)
        self.assertIsNone(service.personality_learning)

    def test_publish_materialized_messy_corpus_persists_final_remote_state_once(self) -> None:
        class _FakeRemoteGraph:
            def __init__(self) -> None:
                self.persisted_documents: list[object] = []

            def enabled(self) -> bool:
                return True

            def persist_document(self, *, document) -> None:
                self.persisted_documents.append(document)

        class _FakeGraphStore:
            def __init__(self) -> None:
                self._remote_graph = _FakeRemoteGraph()

        class _FakeObjectStore:
            def __init__(self) -> None:
                self.snapshot_calls: list[tuple[tuple[object, ...], tuple[object, ...], tuple[object, ...]]] = []

            def write_snapshot(self, *, objects, conflicts=(), archived_objects=()) -> None:
                self.snapshot_calls.append((tuple(objects), tuple(conflicts), tuple(archived_objects)))

        class _FakeMidtermStore:
            def __init__(self) -> None:
                self.saved_packets: list[tuple[object, ...]] = []

            def save_packets(self, *, packets) -> None:
                self.saved_packets.append(tuple(packets))

        class _FakeService:
            def __init__(self) -> None:
                self.graph_store = _FakeGraphStore()
                self.object_store = _FakeObjectStore()
                self.midterm_store = _FakeMidtermStore()

        service = _FakeService()
        packet_a = LongTermMidtermPacketV1(
            packet_id="packet-a",
            kind="recent_contact_bundle",
            summary="Packet A",
            details="Synthetic packet A",
            source_memory_ids=(),
            query_hints=("packet-a",),
            sensitivity="normal",
        )
        packet_b = LongTermMidtermPacketV1(
            packet_id="packet-b",
            kind="recent_contact_bundle",
            summary="Packet B",
            details="Synthetic packet B",
            source_memory_ids=(),
            query_hints=("packet-b",),
            sensitivity="normal",
        )
        materialized = _MaterializedMessyCorpus(
            contacts=(),
            preferences=(),
            preference_memory_count=0,
            plans=(),
            episodes=(),
            multimodal_events=0,
            multimodal_episodic_turns=0,
            unified_episodic_objects=0,
            unified_durable_objects=0,
            unified_conflict_count=0,
            unified_midterm_packets=0,
            graph_document="graph-document",
            current_state=StructuredStoreCurrentState(
                objects=(self._memory_object("object-a"), self._memory_object("object-b")),
                conflicts=(self._memory_conflict("slot-a", "candidate-a"),),
                archived_objects=(self._memory_object("archive-a"),),
            ),
            midterm_packets=(packet_a, packet_b),
        )

        _publish_materialized_messy_corpus(
            service=cast(LongTermMemoryService, service),
            materialized=materialized,
        )

        self.assertEqual(service.graph_store._remote_graph.persisted_documents, ["graph-document"])
        self.assertEqual(len(service.object_store.snapshot_calls), 1)
        objects, conflicts, archived_objects = service.object_store.snapshot_calls[0]
        self.assertEqual(tuple(item.memory_id for item in objects), ("object-a", "object-b"))
        self.assertEqual(
            tuple((item.slot_key, item.candidate_memory_id) for item in conflicts),
            (("slot-a", "candidate-a"),),
        )
        self.assertEqual(tuple(item.memory_id for item in archived_objects), ("archive-a",))
        self.assertEqual(service.midterm_store.saved_packets, [(packet_a, packet_b)])

    def test_run_phase_uses_static_rewriter_and_disables_context_fronts(self) -> None:
        class _FakeService:
            def __init__(self) -> None:
                self.graph_store = object()
                self.query_rewriter = object()
                self.prepared_context_front = object()
                self.provider_answer_front = object()

        service = _FakeService()
        synthetic_case = LongTermEvalCase(
            case_id="synthetic-1",
            category="temporal_multihop",
            query_text="Wer hat mich letzten Mittwoch angerufen?",
            canonical_query_text="who called me last Wednesday",
            kind="provider_context",
            expected_contains=("Corinna",),
        )
        multimodal_case = MultimodalEvalCase(
            case_id="multimodal-1",
            category="combined_context",
            query_text="Was habe ich gestern Abend gemacht?",
            canonical_query_text="what did i do yesterday evening",
            expected_durable_contains=("Wohnzimmer",),
        )
        synthetic_result = LongTermEvalCaseResult(
            case_id=synthetic_case.case_id,
            category=synthetic_case.category,
            kind=synthetic_case.kind,
            passed=True,
            matched_contains=("Corinna",),
            missing_contains=(),
            present_forbidden=(),
        )
        multimodal_result = MultimodalEvalCaseResult(
            case_id=multimodal_case.case_id,
            category=multimodal_case.category,
            passed=True,
            durable_context_present=True,
            episodic_context_present=True,
            missing_durable=(),
            missing_episodic=(),
            present_forbidden_durable=(),
            present_forbidden_episodic=(),
        )
        unified_summary = UnifiedRetrievalBenchmarkSummary(
            total_cases=0,
            passed_cases=0,
            candidate_source_exact_cases=0,
            access_path_exact_cases=0,
            selected_id_exact_cases=0,
            join_anchor_exact_cases=0,
            source_precision_macro=1.0,
            source_recall_macro=1.0,
            access_path_precision_macro=1.0,
            access_path_recall_macro=1.0,
            selected_id_precision_macro=1.0,
            selected_id_recall_macro=1.0,
            join_anchor_precision_macro=1.0,
            join_anchor_recall_macro=1.0,
            path_safety_rate=1.0,
            overall_quality_score=1.0,
        )

        def _fake_run_eval_case_safely(*, service, graph_store, case):
            self.assertIs(graph_store, service.graph_store)
            self.assertIsNone(service.prepared_context_front)
            self.assertIsNone(service.provider_answer_front)
            profile = service.query_rewriter.profile(case.query_text)
            self.assertEqual(profile.canonical_english_text, case.canonical_query_text)
            return synthetic_result

        def _fake_run_case(service, case):
            self.assertIsNone(service.prepared_context_front)
            self.assertIsNone(service.provider_answer_front)
            profile = service.query_rewriter.profile(case.query_text)
            self.assertEqual(profile.canonical_english_text, case.canonical_query_text)
            return multimodal_result

        with (
            patch(
                "twinr.memory.longterm.evaluation.messy_memory_eval._build_eval_cases",
                return_value=(synthetic_case,),
            ),
            patch(
                "twinr.memory.longterm.evaluation.messy_memory_eval._build_multimodal_eval_cases",
                return_value=(multimodal_case,),
            ),
            patch(
                "twinr.memory.longterm.evaluation.messy_memory_eval._run_eval_case_safely",
                side_effect=_fake_run_eval_case_safely,
            ),
            patch(
                "twinr.memory.longterm.evaluation.messy_memory_eval._run_case",
                side_effect=_fake_run_case,
            ),
            patch(
                "twinr.memory.longterm.evaluation.messy_memory_eval.unified_retrieval_goldset_cases",
                return_value=(),
            ),
            patch(
                "twinr.memory.longterm.evaluation.messy_memory_eval.wait_for_unified_retrieval_cases",
                return_value=(),
            ),
            patch(
                "twinr.memory.longterm.evaluation.messy_memory_eval.benchmark_unified_retrieval_cases",
                return_value=SimpleNamespace(summary=unified_summary),
            ),
        ):
            phase_result, analysis = _run_phase(
                service=cast(LongTermMemoryService, service),
                phase="writer",
                unified_case_profile="expanded",
                contacts=(),
                preferences=(),
                plans=(),
                episodes=(),
            )

        self.assertEqual(phase_result.synthetic_summary.passed_cases, 1)
        self.assertEqual(phase_result.multimodal_summary.passed_cases, 1)
        self.assertEqual(analysis.summary.total_cases, 0)

    def test_seed_synthetic_graph_locally_then_sync_remote_persists_once(self) -> None:
        class _FakeRemoteGraph:
            def __init__(self) -> None:
                self.persisted_documents: list[object] = []

            def enabled(self) -> bool:
                return True

            def persist_document(self, *, document) -> None:
                self.persisted_documents.append(document)

        class _FakeGraphStore:
            def __init__(self, root: Path) -> None:
                self.path = root / "state" / "chonkydb" / "twinr_graph_v1.json"
                self.user_label = "Erika"
                self.timezone_name = "Europe/Berlin"
                self._lock_path = root / "state" / "locks" / "twinr_graph_v1.json.lock"
                self._remote_graph = _FakeRemoteGraph()

        with tempfile.TemporaryDirectory() as temp_dir:
            fake_graph_store = _FakeGraphStore(Path(temp_dir))

            contacts, preferences, preference_count, plans = _seed_synthetic_graph_locally_then_sync_remote(
                cast(TwinrPersonalGraphStore, fake_graph_store)
            )

        self.assertEqual(len(contacts), 150)
        self.assertEqual(len(preferences), 100)
        self.assertEqual(preference_count, 150)
        self.assertEqual(len(plans), 100)
        self.assertEqual(len(fake_graph_store._remote_graph.persisted_documents), 1)
        persisted_document = fake_graph_store._remote_graph.persisted_documents[0]
        self.assertGreater(len(getattr(persisted_document, "nodes")), 0)
        self.assertGreater(len(getattr(persisted_document, "edges")), 0)

    def test_merge_structured_current_states_and_midterm_packets_keep_composed_fixtures(self) -> None:
        earlier_state = StructuredStoreCurrentState(
            objects=(
                self._memory_object("episode:synthetic"),
                self._memory_object("durable:multimodal"),
            ),
            conflicts=(
                self._memory_conflict("contact:anna:email", "fact:anna_email_old"),
            ),
            archived_objects=(
                self._memory_object("archive:weather_old"),
            ),
        )
        later_state = StructuredStoreCurrentState(
            objects=(
                self._memory_object("episode:corinna_called"),
                self._memory_object("durable:multimodal"),
            ),
            conflicts=(
                self._memory_conflict("contact:anna:email", "fact:anna_email_current"),
            ),
            archived_objects=(
                self._memory_object("archive:weather_old"),
                self._memory_object("archive:shopping_old"),
            ),
        )
        earlier_packet = LongTermMidtermPacketV1(
            packet_id="midterm:multimodal",
            kind="routine_bundle",
            summary="Earlier packet",
            details="Earlier packet",
            source_memory_ids=(),
            query_hints=("routine",),
            sensitivity="normal",
        )
        later_packet = LongTermMidtermPacketV1(
            packet_id="midterm:corinna_today",
            kind="recent_contact_bundle",
            summary="Later packet",
            details="Later packet",
            source_memory_ids=(),
            query_hints=("corinna",),
            sensitivity="normal",
        )

        merged_state = _merge_structured_current_states(earlier_state, later_state)
        merged_packets = _merge_midterm_packets((earlier_packet,), (later_packet,))

        self.assertEqual(
            tuple(item.memory_id for item in merged_state.objects),
            ("durable:multimodal", "episode:corinna_called", "episode:synthetic"),
        )
        self.assertEqual(
            tuple((item.slot_key, item.candidate_memory_id) for item in merged_state.conflicts),
            (
                ("contact:anna:email", "fact:anna_email_current"),
                ("contact:anna:email", "fact:anna_email_old"),
            ),
        )
        self.assertEqual(
            tuple(item.memory_id for item in merged_state.archived_objects),
            ("archive:shopping_old", "archive:weather_old"),
        )
        self.assertEqual(
            tuple(packet.packet_id for packet in merged_packets),
            ("midterm:corinna_today", "midterm:multimodal"),
        )

    def test_phase_result_aggregates_counts_and_failed_case_ids(self) -> None:
        phase = _phase(
            phase="writer",
            synthetic_results=(
                LongTermEvalCaseResult(
                    case_id="contact_exact_01",
                    category="contact_exact_lookup",
                    kind="contact_lookup",
                    passed=False,
                    matched_contains=(),
                    missing_contains=("+49 151 0000500",),
                    present_forbidden=(),
                ),
                LongTermEvalCaseResult(
                    case_id="contact_exact_02",
                    category="contact_exact_lookup",
                    kind="contact_lookup",
                    passed=True,
                    matched_contains=("Paula Adler000",),
                    missing_contains=(),
                    present_forbidden=(),
                ),
            ),
            multimodal_results=(
                MultimodalEvalCaseResult(
                    case_id="combined_weather_1",
                    category="combined_context",
                    passed=False,
                    durable_context_present=True,
                    episodic_context_present=True,
                    missing_durable=(),
                    missing_episodic=("After breakfast I usually ask Twinr about the weather",),
                    present_forbidden_durable=(),
                    present_forbidden_episodic=(),
                ),
            ),
            unified_results=(
                UnifiedRetrievalGoldsetCaseResult(
                    case_id="corinna_phone_full_stack",
                    phase="writer",
                    query_text="What is Corinna Maier's phone number?",
                    candidate_sources=("adaptive", "conflict", "durable", "episodic", "graph", "midterm"),
                    access_path=(
                        "structured_query_first",
                        "catalog_current_head",
                        "topk_scope_query",
                        "retrieve_batch",
                        "graph_path_query",
                        "graph_neighbors_query",
                    ),
                ),
            ),
            synthetic_passed=1,
            multimodal_passed=0,
            unified_passed=1,
        )

        self.assertEqual(phase.total_cases, 4)
        self.assertEqual(phase.passed_cases, 2)
        self.assertAlmostEqual(phase.accuracy, 0.5)
        self.assertEqual(phase.suite_case_counts, {"synthetic": 2, "multimodal": 1, "unified": 1})
        self.assertEqual(phase.suite_pass_counts, {"synthetic": 1, "multimodal": 0, "unified": 1})
        self.assertEqual(
            phase.failed_case_ids,
            ("multimodal:combined_weather_1", "synthetic:contact_exact_01"),
        )

    def test_build_restart_summary_detects_regressions(self) -> None:
        writer_phase = _phase(
            phase="writer",
            synthetic_results=(
                LongTermEvalCaseResult(
                    case_id="contact_exact_01",
                    category="contact_exact_lookup",
                    kind="contact_lookup",
                    passed=True,
                    matched_contains=("Paula Adler000",),
                    missing_contains=(),
                    present_forbidden=(),
                ),
            ),
            multimodal_results=(
                MultimodalEvalCaseResult(
                    case_id="combined_weather_1",
                    category="combined_context",
                    passed=True,
                    durable_context_present=True,
                    episodic_context_present=True,
                    missing_durable=(),
                    missing_episodic=(),
                    present_forbidden_durable=(),
                    present_forbidden_episodic=(),
                ),
            ),
            unified_results=(
                UnifiedRetrievalGoldsetCaseResult(
                    case_id="corinna_recent_call_continuity",
                    phase="writer",
                    query_text="Did Corinna call earlier today?",
                ),
            ),
            synthetic_passed=1,
            multimodal_passed=1,
            unified_passed=1,
        )
        fresh_reader_phase = _phase(
            phase="fresh_reader",
            synthetic_results=(
                LongTermEvalCaseResult(
                    case_id="contact_exact_01",
                    category="contact_exact_lookup",
                    kind="contact_lookup",
                    passed=True,
                    matched_contains=("Paula Adler000",),
                    missing_contains=(),
                    present_forbidden=(),
                ),
            ),
            multimodal_results=(
                MultimodalEvalCaseResult(
                    case_id="combined_weather_1",
                    category="combined_context",
                    passed=False,
                    durable_context_present=True,
                    episodic_context_present=True,
                    missing_durable=(),
                    missing_episodic=("After breakfast I usually ask Twinr about the weather",),
                    present_forbidden_durable=(),
                    present_forbidden_episodic=(),
                ),
            ),
            unified_results=(
                UnifiedRetrievalGoldsetCaseResult(
                    case_id="corinna_recent_call_continuity",
                    phase="fresh_reader",
                    query_text="Did Corinna call earlier today?",
                    missing_context_terms=(
                        ("episodic_context", ("Corinna called earlier today",)),
                    ),
                ),
            ),
            synthetic_passed=1,
            multimodal_passed=0,
            unified_passed=0,
        )

        restart = _build_restart_summary(
            writer_phase=writer_phase,
            fresh_reader_phase=fresh_reader_phase,
        )

        self.assertLess(restart.accuracy_delta, 0.0)
        self.assertLess(restart.unified_quality_delta, 0.0)
        self.assertEqual(
            restart.regressed_case_ids,
            (
                "multimodal:combined_weather_1",
                "unified:corinna_recent_call_continuity",
            ),
        )
        self.assertEqual(restart.recovered_case_ids, ())

    def test_write_messy_memory_eval_artifacts(self) -> None:
        writer_phase = _phase(
            phase="writer",
            synthetic_results=(
                LongTermEvalCaseResult(
                    case_id="contact_exact_01",
                    category="contact_exact_lookup",
                    kind="contact_lookup",
                    passed=True,
                    matched_contains=("Paula Adler000",),
                    missing_contains=(),
                    present_forbidden=(),
                ),
            ),
            multimodal_results=(
                MultimodalEvalCaseResult(
                    case_id="combined_weather_1",
                    category="combined_context",
                    passed=True,
                    durable_context_present=True,
                    episodic_context_present=True,
                    missing_durable=(),
                    missing_episodic=(),
                    present_forbidden_durable=(),
                    present_forbidden_episodic=(),
                ),
            ),
            unified_results=(
                UnifiedRetrievalGoldsetCaseResult(
                    case_id="corinna_phone_full_stack",
                    phase="writer",
                    query_text="What is Corinna Maier's phone number?",
                    candidate_sources=("adaptive", "conflict", "durable", "episodic", "graph", "midterm"),
                    access_path=(
                        "structured_query_first",
                        "catalog_current_head",
                        "topk_scope_query",
                        "retrieve_batch",
                        "graph_path_query",
                        "graph_neighbors_query",
                    ),
                ),
            ),
            synthetic_passed=1,
            multimodal_passed=1,
            unified_passed=1,
        )
        fresh_reader_phase = _phase(
            phase="fresh_reader",
            synthetic_results=writer_phase.synthetic_case_results,
            multimodal_results=writer_phase.multimodal_case_results,
            unified_results=(
                UnifiedRetrievalGoldsetCaseResult(
                    case_id="corinna_phone_full_stack",
                    phase="fresh_reader",
                    query_text="What is Corinna Maier's phone number?",
                    candidate_sources=("adaptive", "conflict", "durable", "episodic", "graph", "midterm"),
                    access_path=(
                        "structured_query_first",
                        "catalog_current_head",
                        "topk_scope_query",
                        "retrieve_batch",
                        "graph_path_query",
                        "graph_neighbors_query",
                    ),
                ),
            ),
            synthetic_passed=1,
            multimodal_passed=1,
            unified_passed=1,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            result = MessyMemoryEvalResult(
                probe_id="messy_memory_eval_20260404t070000z",
                status="ok",
                started_at="2026-04-04T07:00:00Z",
                finished_at="2026-04-04T07:00:30Z",
                env_path=str(project_root / ".env"),
                base_project_root=str(project_root),
                runtime_namespace="twinr_messy_memory_eval_messy_memory_eval_20260404t070000z",
                writer_root=str(project_root / "writer"),
                fresh_reader_root=str(project_root / "reader"),
                seed_stats=MessyMemoryEvalSeedStats(
                    synthetic_contacts=150,
                    synthetic_preferences=150,
                    synthetic_plans=100,
                    synthetic_episodic_turns=100,
                    multimodal_events=200,
                    multimodal_episodic_turns=300,
                    unified_episodic_objects=2,
                    unified_durable_objects=3,
                    unified_conflict_count=1,
                    unified_midterm_packets=2,
                    combined_graph_nodes=512,
                    combined_graph_edges=644,
                ),
                writer_phase=writer_phase,
                fresh_reader_phase=fresh_reader_phase,
                restart_summary=MessyMemoryRestartSummary(
                    writer_accuracy=1.0,
                    fresh_reader_accuracy=1.0,
                    accuracy_delta=0.0,
                    writer_unified_quality_score=1.0,
                    fresh_reader_unified_quality_score=1.0,
                    unified_quality_delta=0.0,
                ),
            )

            persisted = write_messy_memory_eval_artifacts(result, project_root=project_root)
            artifact_path = default_messy_memory_eval_path(project_root)
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            report_exists = Path(persisted.report_path or "").exists()

        self.assertTrue(persisted.executed)
        self.assertTrue(persisted.ready)
        self.assertEqual(payload["probe_id"], result.probe_id)
        self.assertEqual(payload["seed_stats"]["total_seed_entries"], 1008)
        self.assertTrue(payload["ready"])
        self.assertTrue(report_exists)

    def test_run_messy_memory_eval_persists_failure_context_and_temp_roots(self) -> None:
        class _FakeService:
            @classmethod
            def from_config(cls, _config):
                return cls()

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            env_path = project_root / ".env"
            env_path.write_text(
                "\n".join(
                    (
                        "TWINR_CHONKYDB_BASE_URL=https://example.invalid",
                        "TWINR_CHONKYDB_API_KEY=test-key",
                        "TWINR_CHONKYDB_API_KEY_HEADER=x-api-key",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            remote_exc = ChonkyDBError(
                "queue_saturated",
                status_code=429,
                response_json={"error": "queue_saturated", "error_type": "TooManyRequests"},
            )
            unavailable = RuntimeError("Failed to persist fine-grained remote long-term memory items")
            unavailable.__cause__ = remote_exc
            setattr(
                unavailable,
                "remote_write_context",
                {
                    "snapshot_kind": "objects",
                    "operation": "store_records_bulk",
                    "request_path": "/v1/external/records/bulk",
                    "request_payload_kind": "fine_grained_record_batch",
                    "request_execution_mode": "async",
                    "request_correlation_id": "ltw-test123",
                    "batch_index": 1,
                    "batch_count": 1,
                    "request_item_count": 1,
                    "request_bytes": 10847,
                },
            )
            with (
                patch(
                    "twinr.memory.longterm.evaluation.messy_memory_eval.LongTermMemoryService",
                    _FakeService,
                ),
                patch(
                    "twinr.memory.longterm.evaluation.messy_memory_eval.ensure_unified_retrieval_remote_ready",
                ),
                patch(
                    "twinr.memory.longterm.evaluation.messy_memory_eval._build_eval_runtime_config",
                    return_value=object(),
                ),
                patch(
                    "twinr.memory.longterm.evaluation.messy_memory_eval._seed_messy_corpus",
                    side_effect=unavailable,
                ),
                patch("twinr.memory.longterm.evaluation.messy_memory_eval._shutdown_service"),
            ):
                result = run_messy_memory_eval(
                    env_path=env_path,
                    probe_id="messy_failure_context_test",
                )

            self.assertEqual(result.status, "failed")
            remote_write_context = result.error_remote_write_context
            self.assertIsNotNone(remote_write_context)
            assert remote_write_context is not None
            self.assertEqual(remote_write_context["snapshot_kind"], "objects")
            self.assertEqual(remote_write_context["request_bytes"], 10847)
            self.assertTrue(result.failure_temp_roots_preserved)
            self.assertEqual(result.error_exception_chain[0]["type"], "RuntimeError")
            self.assertEqual(result.error_exception_chain[1]["type"], "ChonkyDBError")
            self.assertEqual(result.error_exception_chain[1]["status_code"], 429)
            self.assertTrue(Path(result.writer_root or "").exists())
            self.assertTrue(Path(result.fresh_reader_root or "").exists())


if __name__ == "__main__":
    unittest.main()
