from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.personality.intelligence.models import WorldInterestSignal
from twinr.agent.personality.models import (
    ContinuityThread,
    PersonalitySnapshot,
    PlaceFocus,
    RelationshipSignal,
    WorldSignal,
)
from twinr.proactive.runtime.display_reserve_snapshot_topics import (
    load_display_reserve_snapshot_candidates,
)


class DisplayReserveSnapshotTopicsTests(unittest.TestCase):
    def test_snapshot_loader_backfills_new_semantic_topics_and_skips_existing_ones(self) -> None:
        snapshot = PersonalitySnapshot(
            continuity_threads=(
                ContinuityThread(
                    title="Arzttermin gestern",
                    summary="Da ist noch offen, wie der Termin gelaufen ist.",
                    salience=0.82,
                ),
                ContinuityThread(
                    title="Janina und die neue Schule",
                    summary="Das bleibt ein gemeinsamer Familienfaden.",
                    salience=0.76,
                ),
            ),
            relationship_signals=(
                RelationshipSignal(
                    topic="AI companions",
                    summary="Das ist als gemeinsames Interessensfeld mehrfach aufgetaucht.",
                    salience=0.72,
                ),
            ),
            place_focuses=(
                PlaceFocus(
                    name="Schwarzenbek",
                    summary="Der Ort bleibt als lokaler Bezug relevant.",
                    geography="local",
                    salience=0.64,
                ),
            ),
            world_signals=(
                WorldSignal(
                    topic="OpenAI baut einen automatisierten Researcher",
                    summary="Der Forschungs- und Produktwinkel bleibt konkret spannend.",
                    source="world",
                    salience=0.74,
                    evidence_count=2,
                ),
            ),
        )

        candidates = load_display_reserve_snapshot_candidates(
            snapshot,
            engagement_signals=(),
            exclude_topic_keys=("ai companions",),
            max_items=8,
        )

        semantic_topics = {candidate.semantic_key() for candidate in candidates}
        self.assertNotIn("ai companions", semantic_topics)
        self.assertIn("arzttermin gestern", semantic_topics)
        self.assertIn("janina und die neue schule", semantic_topics)
        self.assertIn("schwarzenbek", semantic_topics)
        self.assertIn("openai baut einen automatisierten researcher", semantic_topics)
        world_candidate = next(
            candidate
            for candidate in candidates
            if candidate.semantic_key() == "openai baut einen automatisierten researcher"
        )
        self.assertEqual(world_candidate.source, "situational_awareness")

    def test_snapshot_loader_skips_avoided_topics_but_keeps_cooling_topics(self) -> None:
        snapshot = PersonalitySnapshot(
            continuity_threads=(
                ContinuityThread(
                    title="Agentic AI",
                    summary="Das waere ohne Guard ein weiterer technischer Themenfaden.",
                    salience=0.83,
                ),
                ContinuityThread(
                    title="Janina",
                    summary="Der persoenliche Faden bleibt trotz abgekuehlter Reaktion relevant.",
                    salience=0.77,
                ),
            ),
        )
        engagement_signals = (
            WorldInterestSignal(
                signal_id="signal:agentic-ai",
                topic="Agentic AI",
                summary="Da soll Twinr im Moment deutlich zurueckgehen.",
                salience=0.81,
                engagement_score=0.12,
                engagement_state="avoid",
            ),
            WorldInterestSignal(
                signal_id="signal:janina",
                topic="Janina",
                summary="Das Thema kuehlt gerade etwas ab, ist aber nicht tabu.",
                salience=0.68,
                engagement_score=0.34,
                engagement_state="cooling",
            ),
        )

        candidates = load_display_reserve_snapshot_candidates(
            snapshot,
            engagement_signals=engagement_signals,
            max_items=4,
        )

        semantic_topics = {candidate.semantic_key() for candidate in candidates}
        self.assertNotIn("agentic ai", semantic_topics)
        self.assertIn("janina", semantic_topics)
        janina_candidate = next(candidate for candidate in candidates if candidate.semantic_key() == "janina")
        self.assertLess(janina_candidate.salience, 0.77)

    def test_snapshot_loader_caps_place_backfill_to_top_three_focuses(self) -> None:
        snapshot = PersonalitySnapshot(
            place_focuses=tuple(
                PlaceFocus(
                    name=name,
                    summary=f"{name} bleibt als lokaler Bezug relevant.",
                    geography="local",
                    salience=salience,
                )
                for name, salience in (
                    ("Schwarzenbek", 0.82),
                    ("Hamburg", 0.78),
                    ("Berlin", 0.73),
                    ("Schwarzen Weg", 0.51),
                    ("Schwarzenbeek", 0.49),
                )
            ),
        )

        candidates = load_display_reserve_snapshot_candidates(
            snapshot,
            engagement_signals=(),
            max_items=10,
        )

        place_topics = {
            candidate.semantic_key()
            for candidate in candidates
            if candidate.source == "place"
        }
        self.assertEqual(
            place_topics,
            {"schwarzenbek", "hamburg", "berlin"},
        )


if __name__ == "__main__":
    unittest.main()
