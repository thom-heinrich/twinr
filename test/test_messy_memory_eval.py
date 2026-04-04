from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.memory.longterm.evaluation._unified_retrieval_shared import UnifiedRetrievalGoldsetCaseResult
from twinr.memory.longterm.evaluation.messy_memory_eval import (
    MessyMemoryEvalResult,
    MessyMemoryEvalSeedStats,
    MessyMemoryPhaseResult,
    MessyMemoryRestartSummary,
    _seed_synthetic_graph_locally_then_sync_remote,
    _build_restart_summary,
    default_messy_memory_eval_path,
    write_messy_memory_eval_artifacts,
)
from twinr.memory.longterm.evaluation.multimodal_eval import (
    MultimodalEvalCaseResult,
    MultimodalEvalSummary,
)
from twinr.memory.longterm.evaluation.eval import LongTermEvalCaseResult, LongTermEvalSummary
from twinr.memory.longterm.evaluation.unified_retrieval_benchmark import UnifiedRetrievalBenchmarkSummary


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
                fake_graph_store
            )

        self.assertEqual(len(contacts), 150)
        self.assertEqual(len(preferences), 100)
        self.assertEqual(preference_count, 150)
        self.assertEqual(len(plans), 100)
        self.assertEqual(len(fake_graph_store._remote_graph.persisted_documents), 1)
        persisted_document = fake_graph_store._remote_graph.persisted_documents[0]
        self.assertGreater(len(persisted_document.nodes), 0)
        self.assertGreater(len(persisted_document.edges), 0)

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
                    missing_context_terms={
                        "episodic_context": ("Corinna called earlier today",),
                    },
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


if __name__ == "__main__":
    unittest.main()
