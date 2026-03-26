from datetime import datetime
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.personality.display_impulses import (
    build_ambient_display_impulse_candidates,
)
from twinr.agent.personality.intelligence.models import WorldInterestSignal
from twinr.agent.personality.models import (
    ContinuityThread,
    ConversationStyleProfile,
    HumorProfile,
    RelationshipSignal,
    PersonalitySnapshot,
    WorldSignal,
)


class PersonalityDisplayImpulseTests(unittest.TestCase):
    def test_shared_thread_topic_becomes_positive_display_impulse(self) -> None:
        snapshot = PersonalitySnapshot(
            style_profile=ConversationStyleProfile(verbosity=0.72, initiative=0.78),
            humor_profile=HumorProfile(
                style="dry",
                summary="Dry, light observational humor.",
                intensity=0.56,
            ),
            continuity_threads=(
                ContinuityThread(
                    title="AI companions",
                    summary="Da gibt es heute wieder neue Entwicklungen.",
                    salience=0.92,
                ),
            ),
        )
        signal = WorldInterestSignal(
            signal_id="world:ai_companions",
            topic="AI companions",
            summary="Thema mit wiederholtem Nutzerzug.",
            engagement_score=0.94,
            engagement_state="resonant",
            ongoing_interest="active",
            co_attention_state="shared_thread",
            co_attention_count=3,
            engagement_count=5,
            positive_signal_count=4,
        )

        candidates = build_ambient_display_impulse_candidates(
            snapshot,
            engagement_signals=(signal,),
            local_now=datetime(2026, 3, 22, 9, 30),
        )

        self.assertTrue(candidates)
        first = candidates[0]
        self.assertEqual(first.topic_key, "ai companions")
        self.assertIn(first.action, {"ask_one", "invite_follow_up"})
        self.assertEqual(first.accent, "warm")
        self.assertEqual(first.eyebrow, "")
        self.assertIn("AI companions", first.headline)
        self.assertTrue(first.body)
        self.assertIsNotNone(first.generation_context)
        assert first.generation_context is not None
        self.assertEqual(first.generation_context.get("candidate_family"), "mindshare")
        self.assertEqual(
            first.generation_context.get("card_intent"),
            {
                "topic_semantics": "gemeinsamer Faden zu AI companions",
                "statement_intent": "Twinr soll eine konkrete Beobachtung oder einen ruhigen Rueckbezug zu AI companions machen.",
                "cta_intent": "Zu einer kurzen Meinung, Ergaenzung oder einem Weiterreden einladen.",
                "relationship_stance": "warm und aufmerksam statt behauptend",
            },
        )

    def test_continuity_topic_prefers_memory_clarifying_copy(self) -> None:
        snapshot = PersonalitySnapshot(
            style_profile=ConversationStyleProfile(verbosity=0.64, initiative=0.68),
            humor_profile=HumorProfile(
                style="dry",
                summary="Dry, light observational humor.",
                intensity=0.38,
            ),
            continuity_threads=(
                ContinuityThread(
                    title="Arzttermin",
                    summary="Da gibt es wohl noch offene Punkte.",
                    salience=0.94,
                ),
            ),
        )
        signal = WorldInterestSignal(
            signal_id="world:arzttermin",
            topic="Arzttermin",
            summary="Persoenlicher Faden mit weiterem Klaerungsbedarf.",
            engagement_score=0.88,
            engagement_state="resonant",
            ongoing_interest="active",
            co_attention_state="shared_thread",
            co_attention_count=2,
            engagement_count=3,
            positive_signal_count=3,
        )

        candidates = build_ambient_display_impulse_candidates(
            snapshot,
            engagement_signals=(signal,),
            local_now=datetime(2026, 3, 22, 9, 45),
        )

        self.assertTrue(candidates)
        first = candidates[0]
        self.assertIn("Arzttermin", first.headline)
        self.assertTrue(first.body)

    def test_cooling_topic_does_not_surface_as_ambient_impulse(self) -> None:
        snapshot = PersonalitySnapshot(
            style_profile=ConversationStyleProfile(verbosity=0.55, initiative=0.55),
            continuity_threads=(
                ContinuityThread(
                    title="World politics",
                    summary="Bleibt ein Thema, aber gerade ohne guten Zug.",
                    salience=0.9,
                ),
            ),
        )
        signal = WorldInterestSignal(
            signal_id="world:world_politics",
            topic="World politics",
            summary="Abklingendes Thema.",
            engagement_score=0.42,
            engagement_state="cooling",
            ongoing_interest="peripheral",
            co_attention_state="latent",
            non_reengagement_count=3,
            deflection_count=0,
            exposure_count=4,
            positive_signal_count=1,
        )

        candidates = build_ambient_display_impulse_candidates(
            snapshot,
            engagement_signals=(signal,),
            local_now=datetime(2026, 3, 22, 11, 0),
        )

        self.assertEqual(candidates, ())

    def test_relationship_summary_does_not_add_internal_durable_attention_prefix(self) -> None:
        snapshot = PersonalitySnapshot(
            style_profile=ConversationStyleProfile(verbosity=0.52, initiative=0.61),
            relationship_signals=(
                RelationshipSignal(
                    topic="AI companions",
                    summary="Der Nutzer kommt darauf immer wieder zurueck.",
                    salience=0.91,
                    stance="affinity",
                    source="conversation",
                ),
            ),
        )
        signal = WorldInterestSignal(
            signal_id="world:ai_companions",
            topic="AI companions",
            summary="Wiederkehrender gemeinsamer Faden.",
            engagement_score=0.94,
            engagement_state="resonant",
            ongoing_interest="active",
            co_attention_state="shared_thread",
            co_attention_count=3,
            engagement_count=5,
            positive_signal_count=4,
        )

        candidates = build_ambient_display_impulse_candidates(
            snapshot,
            engagement_signals=(signal,),
            local_now=datetime(2026, 3, 22, 9, 50),
        )

        self.assertTrue(candidates)
        context = candidates[0].generation_context or {}
        self.assertEqual(context.get("hook_hint"), "Der Nutzer kommt darauf immer wieder zurueck.")
        self.assertEqual(
            context.get("card_intent"),
            {
                "topic_semantics": "gemeinsamer Faden zu AI companions",
                "statement_intent": "Twinr soll eine konkrete Beobachtung oder einen ruhigen Rueckbezug zu AI companions machen.",
                "cta_intent": "Zu einer kurzen Meinung, Ergaenzung oder einem Weiterreden einladen.",
                "relationship_stance": "warm und aufmerksam statt behauptend",
            },
        )

    def test_live_search_mindshare_does_not_surface_as_display_impulse(self) -> None:
        snapshot = PersonalitySnapshot(
            style_profile=ConversationStyleProfile(verbosity=0.5, initiative=0.56),
            world_signals=(
                WorldSignal(
                    topic="Donald Trump heute.",
                    summary="Suchrest aus einer frischen Live-Suche.",
                    salience=0.82,
                    source="live_search",
                ),
            ),
        )
        signal = WorldInterestSignal(
            signal_id="world:donald_trump_heute",
            topic="Donald Trump heute.",
            summary="Frisches Suchthema ohne durable Co-Attention.",
            engagement_score=0.66,
            engagement_state="warm",
            ongoing_interest="growing",
            co_attention_state="latent",
            engagement_count=1,
            positive_signal_count=1,
        )

        candidates = build_ambient_display_impulse_candidates(
            snapshot,
            engagement_signals=(signal,),
            local_now=datetime(2026, 3, 22, 10, 10),
        )

        self.assertEqual(candidates, ())


if __name__ == "__main__":
    unittest.main()
