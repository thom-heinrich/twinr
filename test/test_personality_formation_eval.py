"""Test pure report helpers for the bounded personality-formation eval.

These tests deliberately avoid live provider or remote-memory calls. They
verify the deterministic helper logic that scores whether a bounded eval run
actually proved richer reflection, personality drift, and downstream behavior
change.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_personality_formation_eval import (
    _build_evaluation_summary,
    _behavior_changes,
    _default_eval_days,
    _snapshot_metrics,
)
from twinr.agent.personality.intelligence.models import WorldInterestSignal, WorldIntelligenceState
from twinr.agent.personality.models import (
    ConversationStyleProfile,
    HumorProfile,
    PlaceFocus,
    PersonalitySnapshot,
    RelationshipSignal,
)


class PersonalityFormationEvalTests(unittest.TestCase):
    def test_default_eval_days_cover_multiple_learning_channels(self) -> None:
        now = datetime(2026, 3, 21, 12, 0, tzinfo=timezone.utc)
        days = _default_eval_days(now=now)

        self.assertEqual(len(days), 2)
        self.assertTrue(any(day.raw_turns for day in days))
        self.assertTrue(any(day.structured_learning for day in days))
        self.assertTrue(any(day.tool_history for day in days))
        appointment_turn = next(turn for turn in days[0].raw_turns if turn.name == "janina_appointment")
        expected_date = (days[0].raw_turns[0].occurred_at + timedelta(days=1)).date().isoformat()
        self.assertIn(expected_date, appointment_turn.transcript)
        self.assertGreater(appointment_turn.occurred_at.date(), now.date())
        raw_turn_names = {turn.name for day in days for turn in day.raw_turns}
        self.assertIn("style_verbosity_free_1", raw_turn_names)
        self.assertIn("style_initiative_free_2", raw_turn_names)
        self.assertIn("humor_feedback_free_2", raw_turn_names)

    def test_snapshot_metrics_extract_personality_drift_fields(self) -> None:
        snapshot = PersonalitySnapshot(
            humor_profile=HumorProfile(
                style="dry observational humor",
                summary="Use dry humor sparingly.",
                intensity=0.34,
            ),
            style_profile=ConversationStyleProfile(
                verbosity=0.42,
                initiative=0.57,
            ),
            relationship_signals=(
                RelationshipSignal(
                    topic="AI companions",
                    summary="Recurring interest in AI companions.",
                    salience=0.8,
                ),
            ),
            place_focuses=(
                PlaceFocus(
                    name="Hamburg",
                    summary="Hamburg stays relevant context.",
                    salience=0.7,
                ),
            ),
        )

        metrics = _snapshot_metrics(snapshot)

        self.assertTrue(metrics["exists"])
        self.assertEqual(metrics["humor_intensity"], 0.34)
        self.assertEqual(metrics["verbosity"], 0.42)
        self.assertEqual(metrics["initiative"], 0.57)
        self.assertEqual(metrics["place_focuses"], ["Hamburg"])
        self.assertEqual(metrics["relationship_topics"], ["AI companions"])

    def test_behavior_changes_only_reports_actual_differences(self) -> None:
        initial = {
            "AI companions": {
                "in_mindshare": False,
                "positive_engagement_action": None,
                "attention_state": None,
            },
            "world politics": {
                "in_mindshare": True,
                "positive_engagement_action": "hint",
                "attention_state": "growing",
            },
        }
        final = {
            "AI companions": {
                "in_mindshare": True,
                "positive_engagement_action": "invite_follow_up",
                "attention_state": "shared_thread",
            },
            "world politics": {
                "in_mindshare": True,
                "positive_engagement_action": "hint",
                "attention_state": "growing",
            },
        }

        changes = _behavior_changes(initial=initial, final=final)

        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["topic"], "AI companions")
        self.assertIn("positive_engagement_action", changes[0]["fields"])
        self.assertIn("attention_state", changes[0]["fields"])

    def test_build_evaluation_summary_requires_rich_reflection_and_behavior_change(self) -> None:
        initial_snapshot = {
            "exists": True,
            "humor_intensity": 0.2,
            "verbosity": 0.5,
            "initiative": 0.4,
            "relationship_topics": [],
            "place_focuses": [],
            "continuity_titles": [],
        }
        final_snapshot = {
            "exists": True,
            "humor_intensity": 0.32,
            "verbosity": 0.44,
            "initiative": 0.52,
            "relationship_topics": ["AI companions"],
            "place_focuses": [],
            "continuity_titles": ["Ongoing thread about Janina"],
        }
        initial_world = WorldIntelligenceState()
        final_world = WorldIntelligenceState(
            interest_signals=(
                WorldInterestSignal(
                    signal_id="interest:ai_companions",
                    topic="AI companions",
                    summary="Recurring shared interest in AI companions.",
                    salience=0.8,
                    engagement_state="resonant",
                    ongoing_interest="active",
                    co_attention_state="shared_thread",
                ),
            ),
        )
        initial_behavior = {
            "AI companions": {
                "in_mindshare": False,
                "positive_engagement_action": None,
                "attention_state": None,
            },
        }
        final_behavior = {
            "AI companions": {
                "in_mindshare": True,
                "positive_engagement_action": "invite_follow_up",
                "attention_state": "shared_thread",
            },
        }

        evaluation = _build_evaluation_summary(
            initial_snapshot_metrics=initial_snapshot,
            final_snapshot_metrics=final_snapshot,
            initial_memory_state={"summary_count": 0},
            final_memory_state={"summary_count": 1},
            initial_world_state=initial_world,
            final_world_state=final_world,
            initial_behavior=initial_behavior,
            final_behavior=final_behavior,
            total_reflected_objects=2,
            total_created_summaries=1,
            raw_style_signal_counts={
                "verbosity_preference": 2,
                "initiative_preference": 2,
                "humor_feedback": 2,
            },
        )

        self.assertTrue(evaluation["reflection_activity_nonzero"])
        self.assertTrue(evaluation["reflection_semantic_richness_nonzero"])
        self.assertEqual(evaluation["inline_summary_growth"], 1)
        self.assertTrue(evaluation["humor_evolution_nonzero"])
        self.assertTrue(evaluation["style_evolution_nonzero"])
        self.assertTrue(evaluation["personality_forming_proven"])
        self.assertTrue(evaluation["verbosity_forming_proven"])
        self.assertTrue(evaluation["initiative_forming_proven"])
        self.assertTrue(evaluation["humor_forming_proven"])
        self.assertTrue(evaluation["style_axes_forming_proven"])
        self.assertEqual(evaluation["behavior_change_topics"], ["AI companions"])

    def test_build_evaluation_summary_marks_thin_reflection_when_no_summaries_exist(self) -> None:
        evaluation = _build_evaluation_summary(
            initial_snapshot_metrics={
                "exists": True,
                "humor_intensity": 0.2,
                "verbosity": 0.5,
                "initiative": 0.4,
                "relationship_topics": [],
                "place_focuses": [],
                "continuity_titles": [],
            },
            final_snapshot_metrics={
                "exists": True,
                "humor_intensity": 0.2,
                "verbosity": 0.5,
                "initiative": 0.4,
                "relationship_topics": [],
                "place_focuses": [],
                "continuity_titles": [],
            },
            initial_memory_state={"summary_count": 0},
            final_memory_state={"summary_count": 0},
            initial_world_state=WorldIntelligenceState(),
            final_world_state=WorldIntelligenceState(),
            initial_behavior={},
            final_behavior={},
            total_reflected_objects=3,
            total_created_summaries=0,
            raw_style_signal_counts={},
        )

        self.assertTrue(evaluation["reflection_activity_nonzero"])
        self.assertFalse(evaluation["reflection_semantic_richness_nonzero"])
        self.assertFalse(evaluation["personality_forming_proven"])
        self.assertFalse(evaluation["style_axes_forming_proven"])

    def test_build_evaluation_summary_keeps_style_axes_red_without_raw_turn_style_learning(self) -> None:
        evaluation = _build_evaluation_summary(
            initial_snapshot_metrics={
                "exists": True,
                "humor_intensity": 0.2,
                "verbosity": 0.5,
                "initiative": 0.4,
                "relationship_topics": [],
                "place_focuses": [],
                "continuity_titles": [],
            },
            final_snapshot_metrics={
                "exists": True,
                "humor_intensity": 0.3,
                "verbosity": 0.42,
                "initiative": 0.52,
                "relationship_topics": ["AI companions"],
                "place_focuses": [],
                "continuity_titles": ["Janina"],
            },
            initial_memory_state={"summary_count": 0},
            final_memory_state={"summary_count": 1},
            initial_world_state=WorldIntelligenceState(),
            final_world_state=WorldIntelligenceState(),
            initial_behavior={},
            final_behavior={},
            total_reflected_objects=2,
            total_created_summaries=1,
            raw_style_signal_counts={"verbosity_preference": 0, "initiative_preference": 0, "humor_feedback": 0},
        )

        self.assertTrue(evaluation["humor_evolution_nonzero"])
        self.assertTrue(evaluation["style_evolution_nonzero"])
        self.assertFalse(evaluation["verbosity_forming_proven"])
        self.assertFalse(evaluation["initiative_forming_proven"])
        self.assertFalse(evaluation["humor_forming_proven"])
        self.assertFalse(evaluation["style_axes_forming_proven"])


if __name__ == "__main__":
    unittest.main()
