from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.memory.longterm.evaluation._unified_retrieval_shared import (
    UnifiedRetrievalFixtureSeedStats,
    UnifiedRetrievalGoldsetCaseResult,
)
from twinr.memory.longterm.evaluation.live_unified_retrieval_acceptance import (
    LiveUnifiedRetrievalAcceptanceResult,
    default_live_unified_retrieval_acceptance_path,
    write_live_unified_retrieval_acceptance_artifacts,
)


class LiveUnifiedRetrievalAcceptanceArtifactTests(unittest.TestCase):
    def test_write_live_unified_retrieval_acceptance_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            result = LiveUnifiedRetrievalAcceptanceResult(
                probe_id="unified_retrieval_live_20260330t180000z",
                status="ok",
                started_at="2026-03-30T18:00:00Z",
                finished_at="2026-03-30T18:00:12Z",
                env_path=str(project_root / ".env"),
                base_project_root=str(project_root),
                runtime_namespace="twinr_unified_retrieval_live_unified_retrieval_live_20260330t180000z",
                writer_root=str(project_root / "writer"),
                fresh_reader_root=str(project_root / "reader"),
                seed_stats=UnifiedRetrievalFixtureSeedStats(
                    episodic_objects=2,
                    durable_objects=3,
                    conflict_count=1,
                    midterm_packets=2,
                    graph_nodes=6,
                    graph_edges=5,
                ),
                case_results=(
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
            )

            persisted = write_live_unified_retrieval_acceptance_artifacts(
                result,
                project_root=project_root,
            )
            artifact_path = default_live_unified_retrieval_acceptance_path(project_root)
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            report_exists = Path(persisted.report_path or "").exists()
            artifact_exists = artifact_path.exists()

        self.assertTrue(persisted.ready)
        self.assertEqual(payload["probe_id"], result.probe_id)
        self.assertEqual(payload["passed_cases"], 2)
        self.assertTrue(payload["ready"])
        self.assertTrue(report_exists)
        self.assertTrue(artifact_exists)


if __name__ == "__main__":
    unittest.main()
