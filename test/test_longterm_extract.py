from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
import unittest
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test.longterm_test_program import make_test_extractor
from twinr.memory.longterm.extract import _turn_extraction_schema


class LongTermTurnExtractorTests(unittest.TestCase):
    def test_turn_extraction_schema_is_openai_strict_compatible(self) -> None:
        schema = _turn_extraction_schema()
        proposition_schema = schema["properties"]["propositions"]["items"]
        edge_schema = schema["properties"]["graph_edges"]["items"]

        self.assertEqual(set(proposition_schema["required"]), set(proposition_schema["properties"]))
        self.assertEqual(set(edge_schema["required"]), set(edge_schema["properties"]))
        self.assertEqual(proposition_schema["properties"]["details"]["anyOf"][1]["type"], "null")
        self.assertEqual(proposition_schema["properties"]["subject_ref"]["anyOf"][1]["type"], "null")
        self.assertEqual(proposition_schema["properties"]["object_ref"]["anyOf"][1]["type"], "null")
        self.assertEqual(proposition_schema["properties"]["value_text"]["anyOf"][1]["type"], "null")
        self.assertEqual(proposition_schema["properties"]["valid_from"]["anyOf"][1]["type"], "null")
        self.assertEqual(proposition_schema["properties"]["valid_to"]["anyOf"][1]["type"], "null")
        self.assertEqual(edge_schema["properties"]["valid_from"]["anyOf"][1]["type"], "null")
        self.assertEqual(edge_schema["properties"]["valid_to"]["anyOf"][1]["type"], "null")
        attribute_schema = proposition_schema["properties"]["attributes"]["items"]
        self.assertEqual(set(attribute_schema["required"]), {"key", "value"})
        self.assertEqual(set(attribute_schema["properties"]), {"key", "value"})

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
        self.assertIn("social_related_to_user", edge_types)
        self.assertIn("general_related_to", edge_types)
        self.assertIn("temporal_occurs_on", edge_types)
        self.assertNotIn("vision problem", rendered)
        self.assertNotIn("sehproblem", rendered)


if __name__ == "__main__":
    unittest.main()
