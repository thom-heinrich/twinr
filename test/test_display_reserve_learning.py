from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.agent.personality.display_impulses import AmbientDisplayImpulseCandidate
from twinr.display.ambient_impulse_history import DisplayAmbientImpulseHistoryStore
from twinr.proactive.runtime.display_reserve_learning import DisplayReserveLearningProfileBuilder


class DisplayReserveLearningTests(unittest.TestCase):
    def test_learning_profile_boosts_topics_with_immediate_pickup_and_cools_repeated_ignores(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            history = DisplayAmbientImpulseHistoryStore.from_config(config)
            now = datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc)

            engaged = history.append_exposure(
                source="world",
                topic_key="ai companions",
                title="AI companions",
                headline="AI companions",
                body="Da ist gerade etwas interessant.",
                action="ask_one",
                attention_state="shared_thread",
                shown_at=now - timedelta(days=1, minutes=10),
                expires_at=now - timedelta(days=1, minutes=2),
                metadata={"candidate_family": "world"},
            )
            history.resolve_feedback(
                exposure_id=engaged.exposure_id,
                response_status="engaged",
                response_sentiment="positive",
                response_at=now - timedelta(days=1, minutes=8),
                response_mode="voice_immediate_pickup",
                response_latency_seconds=12.0,
                response_turn_id="turn:1",
                response_target="AI companions",
                response_summary="The user picked up the displayed thread immediately.",
            )

            ignored_one = history.append_exposure(
                source="memory_follow_up",
                topic_key="janina",
                title="Janina",
                headline="Wie ist es damit weitergegangen?",
                body="Da fehlt mir noch ein Stueck.",
                action="ask_one",
                attention_state="forming",
                shown_at=now - timedelta(hours=28),
                expires_at=now - timedelta(hours=27, minutes=50),
                metadata={"candidate_family": "memory_follow_up"},
            )
            history.resolve_feedback(
                exposure_id=ignored_one.exposure_id,
                response_status="ignored",
                response_sentiment="neutral",
                response_at=now - timedelta(hours=25),
                response_mode="no_voice_pickup",
                response_latency_seconds=3600.0,
                response_turn_id="turn:2",
                response_target="Janina",
                response_summary="The user did not return to the displayed topic.",
            )

            ignored_two = history.append_exposure(
                source="memory_follow_up",
                topic_key="janina",
                title="Janina",
                headline="Wie ist es damit weitergegangen?",
                body="Da fehlt mir noch ein Stueck.",
                action="ask_one",
                attention_state="forming",
                shown_at=now - timedelta(hours=8),
                expires_at=now - timedelta(hours=7, minutes=50),
                metadata={"candidate_family": "memory_follow_up"},
            )
            history.resolve_feedback(
                exposure_id=ignored_two.exposure_id,
                response_status="ignored",
                response_sentiment="neutral",
                response_at=now - timedelta(hours=5),
                response_mode="no_voice_pickup",
                response_latency_seconds=3600.0,
                response_turn_id="turn:3",
                response_target="Janina",
                response_summary="The user again did not return to the displayed topic.",
            )

            profile = DisplayReserveLearningProfileBuilder.from_config(config).build(now=now)

            ai_signal = profile.topic_signal("ai companions")
            janina_signal = profile.topic_signal("janina")
            ask_one_signal = profile.action_signal("ask_one")
            memory_family_signal = profile.family_signal("memory_follow_up")

        self.assertIsNotNone(ai_signal)
        self.assertIsNotNone(janina_signal)
        assert ai_signal is not None
        assert janina_signal is not None
        self.assertGreater(ai_signal.normalized_score, 0.4)
        self.assertEqual(ai_signal.topic_state, "pulling")
        self.assertLess(janina_signal.normalized_score, -0.2)
        self.assertEqual(janina_signal.topic_state, "cooling")
        self.assertGreater(janina_signal.repetition_pressure, 0.0)
        self.assertIsNotNone(ask_one_signal)
        self.assertIsNotNone(memory_family_signal)
        assert ask_one_signal is not None
        assert memory_family_signal is not None
        self.assertLess(memory_family_signal.normalized_score, 0.0)

    def test_candidate_adjustment_uses_topic_family_and_action_learning(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            history = DisplayAmbientImpulseHistoryStore.from_config(config)
            now = datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc)
            exposure = history.append_exposure(
                source="world",
                topic_key="world politics",
                title="World politics",
                headline="Weltpolitik",
                body="Da ist heute Bewegung drin.",
                action="brief_update",
                attention_state="growing",
                shown_at=now - timedelta(hours=18),
                expires_at=now - timedelta(hours=17, minutes=50),
                metadata={"candidate_family": "world"},
            )
            history.resolve_feedback(
                exposure_id=exposure.exposure_id,
                response_status="engaged",
                response_sentiment="positive",
                response_at=now - timedelta(hours=17, minutes=55),
                response_mode="voice_immediate_pickup",
                response_latency_seconds=15.0,
                response_turn_id="turn:4",
                response_target="World politics",
                response_summary="Immediate pickup.",
            )
            profile = DisplayReserveLearningProfileBuilder.from_config(config).build(now=now)
            candidate = AmbientDisplayImpulseCandidate(
                topic_key="world politics",
                title="World politics",
                source="world",
                action="brief_update",
                attention_state="growing",
                salience=0.55,
                eyebrow="",
                headline="Weltpolitik",
                body="Da ist heute Bewegung drin.",
                symbol="sparkles",
                accent="info",
                reason="test",
                candidate_family="world",
            )

        self.assertGreater(profile.candidate_adjustment(candidate), 0.0)


if __name__ == "__main__":
    unittest.main()
