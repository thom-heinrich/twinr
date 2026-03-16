from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.memory.longterm.evaluation.eval import run_synthetic_longterm_eval


class LongTermMemoryEvalTests(unittest.TestCase):
    def test_synthetic_eval_runs_with_expected_shape(self) -> None:
        result = run_synthetic_longterm_eval()

        self.assertEqual(result.seed_stats.total_memories, 500)
        self.assertEqual(result.summary.total_cases, 50)
        self.assertEqual(result.summary.category_case_counts["contact_exact_lookup"], 10)
        self.assertEqual(result.summary.category_case_counts["contact_disambiguation"], 10)
        self.assertEqual(result.summary.category_case_counts["shopping_recall"], 10)
        self.assertEqual(result.summary.category_case_counts["temporal_multihop"], 10)
        self.assertEqual(result.summary.category_case_counts["episodic_recall"], 10)
        self.assertTrue(result.memory_path.endswith("state/MEMORY.md"))
        self.assertTrue(result.graph_path.endswith("state/chonkydb/twinr_graph_v1.json"))
        self.assertGreaterEqual(result.summary.accuracy, 0.0)
        self.assertLessEqual(result.summary.accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
