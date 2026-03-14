from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
import unittest
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test.longterm_test_program import make_test_extractor


class LongTermTurnExtractorTests(unittest.TestCase):
    def test_single_turn_can_expand_into_multiple_memory_candidates(self) -> None:
        extractor = make_test_extractor()
        result = extractor.extract_conversation_turn(
            transcript=(
                "Today is a beautiful Sunday, it is really warm. "
                "My wife Janina is at the eye doctor and is getting eye laser treatment."
            ),
            response="I hope Janina's appointment goes smoothly.",
            occurred_at=datetime(2026, 3, 14, 10, 30, tzinfo=ZoneInfo("Europe/Berlin")),
            turn_id="turn:test",
        )

        summaries = [item.summary for item in result.candidate_objects]
        edge_types = {edge.edge_type for edge in result.graph_edges}
        rendered = "\n".join(
            [result.episode.summary, result.episode.details or "", *summaries]
        ).lower()

        self.assertEqual(result.turn_id, "turn:test")
        self.assertEqual(result.episode.kind, "episode")
        self.assertIn("Janina is the user's wife.", summaries)
        self.assertIn("The user described the day as sunday.", summaries)
        self.assertIn("The user described the day as warm.", summaries)
        self.assertTrue(any("eye laser treatment" in summary for summary in summaries))
        self.assertIn("social_family_of", edge_types)
        self.assertIn("general_related_to", edge_types)
        self.assertIn("temporal_occurs_on", edge_types)
        self.assertNotIn("vision problem", rendered)
        self.assertNotIn("sehproblem", rendered)


if __name__ == "__main__":
    unittest.main()
