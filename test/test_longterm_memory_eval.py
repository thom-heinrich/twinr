from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.memory.chonkydb.personal_graph import TwinrPersonalGraphStore
from twinr.memory.longterm.evaluation.eval import (
    LongTermEvalCase,
    _run_eval_case,
    _seed_contacts,
    run_synthetic_longterm_eval,
)


class LongTermMemoryEvalTests(unittest.TestCase):
    def test_contact_lookup_eval_accepts_canonical_phone_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = TwinrPersonalGraphStore(
                path=Path(temp_dir) / "graph.json",
                user_label="Erika",
                timezone_name="Europe/Berlin",
            )
            contact = next(item for item in _seed_contacts(store) if not item.ambiguous)
            case = LongTermEvalCase(
                case_id="contact_exact_01",
                category="contact_exact_lookup",
                query_text=contact.given_name,
                kind="contact_lookup",
                lookup_family_name=contact.family_name,
                lookup_role=contact.role.replace("_", " ").title(),
                expected_lookup_status="found",
                expected_contains=(contact.label, contact.phone),
                expected_option_count=0,
            )

            result = _run_eval_case(
                service=None,  # type: ignore[arg-type]
                graph_store=store,
                case=case,
            )

        self.assertTrue(result.passed)
        self.assertEqual(result.missing_contains, ())

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
