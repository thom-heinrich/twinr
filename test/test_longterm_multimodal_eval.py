from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.memory.longterm.evaluation.multimodal_eval import run_multimodal_longterm_eval


class LongTermMultimodalEvalTests(unittest.TestCase):
    def test_multimodal_eval_runs_with_expected_shape(self) -> None:
        result = run_multimodal_longterm_eval()
        failed_cases = [case.case_id for case in result.cases if not case.passed]

        self.assertEqual(result.seed_stats.total_seed_entries, 500)
        self.assertEqual(result.seed_stats.multimodal_events, 200)
        self.assertEqual(result.seed_stats.episodic_turns, 300)
        self.assertEqual(result.summary.total_cases, 50)
        self.assertEqual(result.summary.passed_cases, 50, failed_cases)
        self.assertEqual(result.summary.category_case_counts["presence_routine"], 10)
        self.assertEqual(result.summary.category_case_counts["button_print_routine"], 10)
        self.assertEqual(result.summary.category_case_counts["camera_interaction"], 10)
        self.assertEqual(result.summary.category_case_counts["combined_context"], 10)
        self.assertEqual(result.summary.category_case_counts["control_irrelevant"], 10)
        self.assertEqual(result.summary.category_pass_counts["presence_routine"], 10)
        self.assertEqual(result.summary.category_pass_counts["button_print_routine"], 10)
        self.assertEqual(result.summary.category_pass_counts["camera_interaction"], 10)
        self.assertEqual(result.summary.category_pass_counts["combined_context"], 10)
        self.assertEqual(result.summary.category_pass_counts["control_irrelevant"], 10)
        self.assertTrue(all(case.error is None for case in result.cases))
        self.assertEqual(result.summary.accuracy, 1.0)
        self.assertGreater(result.seed_stats.consolidated_object_count, 0)
        self.assertGreater(result.seed_stats.episodic_entry_count, 0)
        self.assertTrue(result.object_store_path.endswith("state/chonkydb/twinr_memory_objects_v1.json"))
        self.assertTrue(result.memory_path.endswith("state/MEMORY.md"))


if __name__ == "__main__":
    unittest.main()
