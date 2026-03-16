from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.memory.longterm.evaluation.subtext_eval import (
    _extract_json_object,
    contains_explicit_memory_announcement,
    default_subtext_eval_cases,
)


class SubtextResponseEvalTests(unittest.TestCase):
    def test_default_cases_cover_expected_shape(self) -> None:
        cases = default_subtext_eval_cases()
        self.assertEqual(len(cases), 8)
        self.assertGreaterEqual(sum(1 for case in cases if case.should_use_personal_context), 4)
        self.assertGreaterEqual(sum(1 for case in cases if not case.should_use_personal_context), 3)

    def test_explicit_memory_detection_catches_common_phrases(self) -> None:
        self.assertTrue(contains_explicit_memory_announcement("Wenn ich mich richtig erinnere, magst du Melitta."))
        self.assertTrue(contains_explicit_memory_announcement("I remember that you said this yesterday."))
        self.assertFalse(contains_explicit_memory_announcement("Melitta könnte heute gut passen."))

    def test_extract_json_object_accepts_fenced_or_wrapped_content(self) -> None:
        payload = _extract_json_object("```json\n{\"passed\": true, \"reason\": \"ok\"}\n```")
        self.assertEqual(payload["passed"], True)
        self.assertEqual(payload["reason"], "ok")


if __name__ == "__main__":
    unittest.main()
