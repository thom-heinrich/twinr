from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
import unittest
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test.longterm_test_program import make_test_extractor
from twinr.memory.longterm.reasoning.consolidator import LongTermMemoryConsolidator
from twinr.memory.longterm.core.models import (
    LongTermGraphEdgeCandidateV1,
    LongTermMemoryObjectV1,
    LongTermSourceRefV1,
    LongTermTurnExtractionV1,
)
from twinr.memory.longterm.reasoning.truth import LongTermTruthMaintainer


def _source() -> LongTermSourceRefV1:
    return LongTermSourceRefV1(
        source_type="conversation_turn",
        event_ids=("turn:test",),
        speaker="user",
        modality="voice",
    )


class LongTermMemoryConsolidatorTests(unittest.TestCase):
    def test_consolidator_promotes_durable_candidates_and_keeps_contextual_ones_episodic(self) -> None:
        extractor = make_test_extractor()
        consolidator = LongTermMemoryConsolidator(truth_maintainer=LongTermTruthMaintainer())
        extraction = extractor.extract_conversation_turn(
            transcript=(
                "Today is a beautiful Sunday, it is really warm. "
                "My wife Janina is at the eye doctor and is getting eye laser treatment."
            ),
            response="I hope Janina's appointment goes smoothly.",
            occurred_at=datetime(2026, 3, 14, 10, 30, tzinfo=ZoneInfo("Europe/Berlin")),
            turn_id="turn:test",
        )

        result = consolidator.consolidate(extraction=extraction)

        durable_summaries = [item.summary for item in result.durable_objects]
        episodic_kinds = {item.kind for item in result.episodic_objects}
        edge_types = {edge.edge_type for edge in result.graph_edges}

        self.assertFalse(result.clarification_needed)
        self.assertIn("Janina is the user's wife.", durable_summaries)
        self.assertTrue(any("eye laser treatment" in summary for summary in durable_summaries))
        self.assertIn("episode", episodic_kinds)
        self.assertIn("observation", episodic_kinds)
        self.assertIn("social_related_to_user", edge_types)
        self.assertIn("temporal_occurs_on", edge_types)

    def test_consolidator_keeps_conflicting_candidate_deferred_and_blocks_its_graph_edge(self) -> None:
        consolidator = LongTermMemoryConsolidator(truth_maintainer=LongTermTruthMaintainer())
        episode = LongTermMemoryObjectV1(
            memory_id="episode:turn_test",
            kind="episode",
            summary="Conversation turn recorded for long-term memory.",
            details="User said: \"Use Corinna's new phone number.\" Assistant answered: \"I will keep that in mind.\"",
            source=_source(),
            status="candidate",
            confidence=1.0,
            slot_key="episode:turn:test",
            value_key="turn:test",
        )
        candidate = LongTermMemoryObjectV1(
            memory_id="fact:corinna_phone_new",
            kind="contact_method_fact",
            summary="Corinna Maier can be reached at +4940998877.",
            source=_source(),
            status="candidate",
            confidence=0.95,
            slot_key="contact:person:corinna_maier:phone",
            value_key="+4940998877",
            attributes={"person_ref": "person:corinna_maier"},
        )
        extraction = LongTermTurnExtractionV1(
            turn_id="turn:test",
            occurred_at=datetime(2026, 3, 14, 11, 0, tzinfo=ZoneInfo("Europe/Berlin")),
            episode=episode,
            candidate_objects=(candidate,),
            graph_edges=(
                LongTermGraphEdgeCandidateV1(
                    source_ref="person:corinna_maier",
                    edge_type="general_has_contact_method",
                    target_ref="phone:+4940998877",
                    confidence=0.95,
                    attributes={"origin_memory_id": "fact:corinna_phone_new"},
                ),
            ),
        )
        existing = (
            LongTermMemoryObjectV1(
                memory_id="fact:corinna_phone_old",
                kind="contact_method_fact",
                summary="Corinna Maier can be reached at +491761234.",
                source=_source(),
                status="active",
                confidence=0.95,
                slot_key="contact:person:corinna_maier:phone",
                value_key="+491761234",
                attributes={"person_ref": "person:corinna_maier"},
            ),
        )

        result = consolidator.consolidate(extraction=extraction, existing_objects=existing)

        self.assertEqual(len(result.conflicts), 1)
        self.assertEqual(len(result.durable_objects), 0)
        self.assertEqual(result.deferred_objects[0].status, "uncertain")
        self.assertEqual(len(result.graph_edges), 0)


if __name__ == "__main__":
    unittest.main()
