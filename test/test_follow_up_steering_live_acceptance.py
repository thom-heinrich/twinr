"""Validate helper logic for the live follow-up steering acceptance harness."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.personality.intelligence import (
    WorldFeedSubscription,
    WorldIntelligenceState,
    WorldInterestSignal,
)
from twinr.agent.personality.steering import ConversationTurnSteeringCue


_HARNESS_PATH = Path(__file__).resolve().parent / "run_follow_up_steering_live_acceptance.py"
_HARNESS_SPEC = importlib.util.spec_from_file_location(
    "twinr_follow_up_steering_live_acceptance_test_module",
    _HARNESS_PATH,
)
assert _HARNESS_SPEC is not None and _HARNESS_SPEC.loader is not None
_HARNESS = importlib.util.module_from_spec(_HARNESS_SPEC)
sys.modules[_HARNESS_SPEC.name] = _HARNESS
_HARNESS_SPEC.loader.exec_module(_HARNESS)


class FollowUpSteeringLiveAcceptanceTests(unittest.TestCase):
    def test_turn_expectation_uses_current_user_pull_policy(self) -> None:
        keep_open = ConversationTurnSteeringCue(
            title="AI companions",
            salience=0.84,
            attention_state="shared_thread",
            user_pull="one_calm_follow_up",
            positive_engagement_action="invite_follow_up",
        )
        release = ConversationTurnSteeringCue(
            title="local politics",
            salience=0.71,
            attention_state="cooling",
            user_pull="answer_briefly_then_release",
            positive_engagement_action="silent",
        )

        keep_open_expectation = _HARNESS._turn_expectation_for_topic(cue=keep_open)
        release_expectation = _HARNESS._turn_expectation_for_topic(cue=release)
        explicit_close = _HARNESS._turn_expectation_for_topic(cue=keep_open, explicit_close=True)

        self.assertEqual(keep_open_expectation.source, "steering")
        self.assertFalse(keep_open_expectation.force_close)
        self.assertEqual(keep_open_expectation.selected_topic, "AI companions")
        self.assertEqual(release_expectation.source, "steering")
        self.assertTrue(release_expectation.force_close)
        self.assertEqual(release_expectation.selected_topic, "local politics")
        self.assertEqual(explicit_close.source, "closure")
        self.assertTrue(explicit_close.force_close)

    def test_multi_day_summary_counts_transition_checks(self) -> None:
        turn_results = (
            {
                "duration_ms": 1200.0,
                "passed": True,
                "error_type": None,
                "matched_topics": ["AI companions"],
                "runtime_source": "steering",
                "runtime_force_close": False,
            },
        )
        scenario_results = (
            {
                "name": "scenario",
                "passed": True,
                "transition_checks": (
                    {"name": "a", "passed": True},
                    {"name": "b", "passed": False},
                ),
            },
        )

        summary = _HARNESS._multi_day_summary(
            turn_results=turn_results,
            scenario_results=scenario_results,
        )

        self.assertEqual(summary["case_count"], 1)
        self.assertEqual(summary["scenario_count"], 1)
        self.assertEqual(summary["transition_check_count"], 2)
        self.assertEqual(summary["transition_pass_count"], 1)
        self.assertEqual(summary["transition_pass_rate"], 0.5)

    def test_choose_multiday_focus_signal_prefers_feed_covered_topic(self) -> None:
        ai_signal = WorldInterestSignal(
            signal_id="interest:ai_companions",
            topic="AI companions",
            summary="AI companions are a strong running thread.",
            salience=0.9,
            confidence=0.82,
            engagement_score=0.9,
            engagement_state="resonant",
            ongoing_interest="active",
            ongoing_interest_score=0.92,
            co_attention_state="forming",
            co_attention_score=0.66,
            co_attention_count=1,
            updated_at="2026-03-21T09:00:00+00:00",
        )
        world_signal = WorldInterestSignal(
            signal_id="interest:world_politics",
            topic="world politics",
            summary="World politics is also relevant but not feed-covered here.",
            salience=0.88,
            confidence=0.8,
            engagement_score=0.88,
            engagement_state="resonant",
            ongoing_interest="active",
            ongoing_interest_score=0.9,
            co_attention_state="forming",
            co_attention_score=0.62,
            co_attention_count=1,
            updated_at="2026-03-21T09:00:00+00:00",
        )
        context = _HARNESS.LoadedLiveAcceptanceContext(
            snapshot=None,
            cues=(
                ConversationTurnSteeringCue(
                    title="AI companions",
                    salience=0.9,
                    attention_state="forming",
                    user_pull="one_gentle_follow_up",
                    positive_engagement_action="ask_one",
                ),
                ConversationTurnSteeringCue(
                    title="world politics",
                    salience=0.91,
                    attention_state="forming",
                    user_pull="one_gentle_follow_up",
                    positive_engagement_action="ask_one",
                ),
            ),
            state=WorldIntelligenceState(interest_signals=(world_signal, ai_signal)),
            subscriptions=(
                WorldFeedSubscription(
                    subscription_id="feed:ai",
                    label="AI companions feed",
                    feed_url="https://example.com/ai.xml",
                    topics=("AI companions",),
                    priority=0.8,
                    created_by="installer",
                    created_at="2026-03-21T09:00:00+00:00",
                ),
            ),
            remote_state=object(),
        )

        selected = _HARNESS._choose_multiday_focus_signal(context)

        self.assertEqual(selected.topic, "AI companions")


if __name__ == "__main__":
    unittest.main()
