from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.memory.longterm.core.models import (
    LongTermConflictOptionV1,
    LongTermConflictQueueItemV1,
    LongTermProactiveCandidateV1,
)
from twinr.proactive.runtime.display_reserve_memory import (
    build_memory_conflict_candidate,
    build_memory_follow_up_candidate,
)


class DisplayReserveMemoryTests(unittest.TestCase):
    def test_memory_conflict_candidate_uses_statement_headline_and_cta_body(self) -> None:
        question = "Which phone number should I use for Corinna Maier?"
        reason = "Conflicting phone numbers exist."
        candidate = build_memory_conflict_candidate(
            LongTermConflictQueueItemV1(
                slot_key="contact:person:corinna_maier:phone",
                question=question,
                reason=reason,
                candidate_memory_id="fact:1",
                options=(
                    LongTermConflictOptionV1(
                        memory_id="fact:1",
                        summary="0170 1234567",
                        status="active",
                    ),
                    LongTermConflictOptionV1(
                        memory_id="fact:2",
                        summary="040 123456",
                        status="active",
                    ),
                ),
            )
        )

        assert candidate is not None
        self.assertFalse(candidate.headline.endswith("?"))
        self.assertTrue(candidate.body.endswith("?"))
        self.assertNotEqual(candidate.headline, reason)
        self.assertNotEqual(candidate.body, question)
        self.assertEqual(
            (candidate.generation_context or {}).get("card_intent"),
            {
                "topic_semantics": f"alltaegliche Klaerung zu {question}",
                "statement_intent": f"Twinr soll ruhig sagen, dass zu {question} gerade zwei moegliche Versionen offen sind.",
                "cta_intent": "Den Nutzer bitten, kurz zu sagen, was stimmt oder was gemeint ist.",
                "relationship_stance": "ruhige Klaerung statt Datenpflege- oder Systemton",
            },
        )

    def test_memory_follow_up_candidate_uses_statement_headline_and_cta_body(self) -> None:
        candidate = build_memory_follow_up_candidate(
            LongTermProactiveCandidateV1(
                candidate_id="follow-up:1",
                kind="gentle_follow_up",
                summary="The doctor appointment from yesterday is still open.",
                rationale="This likely matters for continuity.",
                confidence=0.83,
                sensitivity="normal",
                due_date=None,
            )
        )

        assert candidate is not None
        self.assertFalse(candidate.headline.endswith("?"))
        self.assertTrue(candidate.body.endswith("?"))
        self.assertEqual(
            (candidate.generation_context or {}).get("card_intent"),
            {
                "topic_semantics": "persoenlicher Nachfasser zu The doctor appointment from yesterday is still open.",
                "statement_intent": "Twinr soll ruhig an The doctor appointment from yesterday is still open. anknuepfen und zeigen, dass dazu noch etwas offen ist.",
                "cta_intent": "Zu einem kurzen Update oder Weiterreden einladen.",
                "relationship_stance": "ruhiges persoenliches Nachfassen statt Erinnerungs- oder Speicherton",
            },
        )


if __name__ == "__main__":
    unittest.main()
