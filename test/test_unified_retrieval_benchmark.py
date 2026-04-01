from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.memory.longterm.evaluation._unified_retrieval_shared import (
    UnifiedRetrievalGoldsetCaseResult,
    default_unified_retrieval_goldset_cases,
)
from twinr.memory.longterm.evaluation.unified_retrieval_benchmark import (
    UnifiedRetrievalBenchmarkResult,
    benchmark_unified_retrieval_cases,
    default_unified_retrieval_benchmark_path,
    write_unified_retrieval_benchmark_artifacts,
)
from twinr.memory.longterm.evaluation.unified_retrieval_goldset import UnifiedRetrievalGoldsetResult


class UnifiedRetrievalBenchmarkTests(unittest.TestCase):
    def test_benchmark_scores_exact_unified_selection_as_perfect(self) -> None:
        cases = default_unified_retrieval_goldset_cases()
        continuity = next(case for case in cases if case.case_id == "corinna_recent_call_continuity")
        result = UnifiedRetrievalGoldsetCaseResult(
            case_id=continuity.case_id,
            phase="goldset",
            query_text=continuity.query_text,
            candidate_sources=continuity.required_candidate_sources,
            access_path=continuity.required_access_path,
            selected_ids=continuity.required_selected_ids,
            join_anchors=continuity.required_join_anchors,
        )

        analysis = benchmark_unified_retrieval_cases(cases=cases, case_results=(result,))
        case_metric = analysis.case_metrics[0]

        self.assertEqual(case_metric.candidate_source_precision, 1.0)
        self.assertEqual(case_metric.candidate_source_recall, 1.0)
        self.assertEqual(case_metric.selected_id_precision, 1.0)
        self.assertEqual(case_metric.join_anchor_precision, 1.0)
        self.assertEqual(case_metric.unexpected_candidate_sources, ())
        self.assertEqual(case_metric.unexpected_selected_ids, ())
        self.assertEqual(case_metric.unexpected_join_anchors, ())
        self.assertEqual(analysis.summary.overall_quality_score, 1.0)

    def test_write_unified_retrieval_benchmark_artifacts(self) -> None:
        cases = default_unified_retrieval_goldset_cases()
        graph_case = next(case for case in cases if case.case_id == "anna_email_graph_only")
        analysis = benchmark_unified_retrieval_cases(
            cases=cases,
            case_results=(
                UnifiedRetrievalGoldsetCaseResult(
                    case_id=graph_case.case_id,
                    phase="goldset",
                    query_text=graph_case.query_text,
                    candidate_sources=("graph",),
                    access_path=graph_case.required_access_path,
                    selected_ids={"graph_node_ids": ("person:anna_becker",)},
                    present_sections=("graph_context",),
                ),
            ),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            benchmark_result = UnifiedRetrievalBenchmarkResult(
                probe_id="unified_retrieval_benchmark_20260330t190000z",
                status="ok",
                started_at="2026-03-30T19:00:00Z",
                finished_at="2026-03-30T19:00:10Z",
                env_path=str(project_root / ".env"),
                base_project_root=str(project_root),
                goldset_result=UnifiedRetrievalGoldsetResult(
                    probe_id="unified_retrieval_benchmark_20260330t190000z",
                    status="ok",
                    started_at="2026-03-30T19:00:00Z",
                    finished_at="2026-03-30T19:00:05Z",
                    env_path=str(project_root / ".env"),
                    base_project_root=str(project_root),
                    runtime_namespace="twinr_unified_retrieval_goldset_unified_retrieval_benchmark_20260330t190000z",
                    case_results=(
                        UnifiedRetrievalGoldsetCaseResult(
                            case_id=graph_case.case_id,
                            phase="goldset",
                            query_text=graph_case.query_text,
                            candidate_sources=("graph",),
                            access_path=graph_case.required_access_path,
                        ),
                    ),
                ),
                analysis=analysis,
            )

            persisted = write_unified_retrieval_benchmark_artifacts(
                benchmark_result,
                project_root=project_root,
            )
            artifact_path = default_unified_retrieval_benchmark_path(project_root)
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            report_exists = Path(persisted.report_path or "").exists()

        self.assertTrue(persisted.ready)
        self.assertEqual(payload["probe_id"], benchmark_result.probe_id)
        self.assertEqual(payload["analysis"]["summary"]["total_cases"], 1)
        self.assertEqual(payload["analysis"]["summary"]["passed_cases"], 1)
        self.assertTrue(report_exists)


if __name__ == "__main__":
    unittest.main()
