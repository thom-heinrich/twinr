from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.agent.personality.ambient_feedback import AmbientImpulseFeedbackExtractor
from twinr.agent.personality.models import ContinuityThread, InteractionSignal
from twinr.agent.personality.signals import (
    INTERACTION_SIGNAL_TOPIC_AFFINITY,
    INTERACTION_SIGNAL_TOPIC_AVERSION,
    INTERACTION_SIGNAL_TOPIC_COOLING,
    INTERACTION_SIGNAL_TOPIC_ENGAGEMENT,
    PersonalitySignalBatch,
)
from twinr.display.ambient_impulse_history import DisplayAmbientImpulseHistoryStore
from twinr.display.reserve_bus_feedback import DisplayReserveBusFeedbackStore
from twinr.memory.longterm.core.models import (
    LongTermConsolidationResultV1,
    LongTermConversationTurn,
)


def _turn_and_consolidation(*, turn_id: str, occurred_at: datetime) -> tuple[LongTermConversationTurn, LongTermConsolidationResultV1]:
    turn = LongTermConversationTurn(
        transcript="Kurze Rueckmeldung.",
        response="Danke.",
        created_at=occurred_at,
    )
    consolidation = LongTermConsolidationResultV1(
        turn_id=turn_id,
        occurred_at=occurred_at,
        episodic_objects=(),
        durable_objects=(),
        deferred_objects=(),
        conflicts=(),
        graph_edges=(),
    )
    return turn, consolidation


class DisplayAmbientFeedbackExtractorTests(unittest.TestCase):
    def test_positive_pickup_of_recent_card_creates_engagement_signal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            store = DisplayAmbientImpulseHistoryStore.from_config(config)
            feedback_store = DisplayReserveBusFeedbackStore.from_config(config)
            shown_at = datetime(2026, 3, 22, 10, 0, tzinfo=timezone.utc)
            store.append_exposure(
                source="world",
                topic_key="ai companions",
                title="AI companions",
                headline="Wie entwickeln sich AI companions gerade?",
                body="Mich interessiert, was dir daran wichtig ist.",
                action="invite_follow_up",
                attention_state="shared_thread",
                shown_at=shown_at,
                expires_at=shown_at + timedelta(minutes=10),
                match_anchors=("AI companions", "AI companions gerade"),
            )
            extractor = AmbientImpulseFeedbackExtractor(
                history_store=store,
                reserve_bus_feedback_store=feedback_store,
            )
            turn, consolidation = _turn_and_consolidation(
                turn_id="turn:display:positive",
                occurred_at=shown_at + timedelta(minutes=4),
            )

            batch = PersonalitySignalBatch(
                interaction_signals=(
                    InteractionSignal(
                        signal_id="signal:interaction:test:engagement",
                        signal_kind=INTERACTION_SIGNAL_TOPIC_ENGAGEMENT,
                        target="AI companions",
                        summary="The user explicitly wanted to stay with AI companions.",
                        confidence=0.82,
                        impact=0.55,
                        evidence_count=2,
                        source_event_ids=("turn:display:positive",),
                    ),
                ),
            )

            feedback_batch = extractor.extract_from_consolidation(
                turn=turn,
                consolidation=consolidation,
                extracted_batch=batch,
            )
            history = store.load()
            reserve_feedback = feedback_store.load_active(now=shown_at + timedelta(minutes=4))

        self.assertEqual(len(feedback_batch.interaction_signals), 1)
        self.assertEqual(feedback_batch.interaction_signals[0].signal_kind, INTERACTION_SIGNAL_TOPIC_ENGAGEMENT)
        self.assertEqual(feedback_batch.interaction_signals[0].metadata["signal_source"], "display_reserve_card")
        self.assertEqual(feedback_batch.interaction_signals[0].metadata["response_mode"], "voice_immediate_pickup")
        self.assertEqual(history[0].response_status, "engaged")
        self.assertEqual(history[0].response_sentiment, "positive")
        self.assertEqual(history[0].response_mode, "voice_immediate_pickup")
        self.assertIsNotNone(reserve_feedback)
        assert reserve_feedback is not None
        self.assertEqual(reserve_feedback.reaction, "immediate_engagement")
        self.assertEqual(reserve_feedback.topic_key, "ai companions")

    def test_negative_pickup_of_recent_card_creates_negative_signal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            store = DisplayAmbientImpulseHistoryStore.from_config(config)
            feedback_store = DisplayReserveBusFeedbackStore.from_config(config)
            shown_at = datetime(2026, 3, 22, 11, 0, tzinfo=timezone.utc)
            store.append_exposure(
                source="world",
                topic_key="world politics",
                title="World politics",
                headline="Was tut sich gerade in der Weltpolitik?",
                body="Wenn du willst, schauen wir kurz drauf.",
                action="brief_update",
                attention_state="growing",
                shown_at=shown_at,
                expires_at=shown_at + timedelta(minutes=10),
                match_anchors=("world politics",),
            )
            extractor = AmbientImpulseFeedbackExtractor(
                history_store=store,
                reserve_bus_feedback_store=feedback_store,
            )
            turn, consolidation = _turn_and_consolidation(
                turn_id="turn:display:negative",
                occurred_at=shown_at + timedelta(minutes=3),
            )

            batch = PersonalitySignalBatch(
                interaction_signals=(
                    InteractionSignal(
                        signal_id="signal:interaction:test:aversion",
                        signal_kind=INTERACTION_SIGNAL_TOPIC_AVERSION,
                        target="world politics",
                        summary="The user wants less world-politics prompting right now.",
                        confidence=0.79,
                        impact=-0.42,
                        evidence_count=2,
                        source_event_ids=("turn:display:negative",),
                    ),
                ),
            )

            feedback_batch = extractor.extract_from_consolidation(
                turn=turn,
                consolidation=consolidation,
                extracted_batch=batch,
            )
            history = store.load()
            reserve_feedback = feedback_store.load_active(now=shown_at + timedelta(minutes=3))

        self.assertEqual(len(feedback_batch.interaction_signals), 1)
        self.assertEqual(feedback_batch.interaction_signals[0].signal_kind, INTERACTION_SIGNAL_TOPIC_AVERSION)
        self.assertEqual(history[0].response_status, "avoided")
        self.assertEqual(history[0].response_sentiment, "negative")
        self.assertEqual(history[0].response_mode, "voice_immediate_pushback")
        self.assertIsNotNone(reserve_feedback)
        assert reserve_feedback is not None
        self.assertEqual(reserve_feedback.reaction, "avoided")

    def test_repeated_non_pickup_cools_topic_after_other_topics_pull_user(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            store = DisplayAmbientImpulseHistoryStore.from_config(config)
            feedback_store = DisplayReserveBusFeedbackStore.from_config(config)
            shown_at = datetime(2026, 3, 22, 9, 0, tzinfo=timezone.utc)
            for offset_minutes in (0, 90):
                store.append_exposure(
                    source="world",
                    topic_key="local politics",
                    title="Local politics",
                    headline="Wie siehst du die lokalen Entscheidungen gerade?",
                    body="Da ist fuer mich noch ein kleiner Faden offen.",
                    action="ask_one",
                    attention_state="forming",
                    shown_at=shown_at + timedelta(minutes=offset_minutes),
                    expires_at=shown_at + timedelta(minutes=offset_minutes + 10),
                    match_anchors=("local politics",),
                )
            extractor = AmbientImpulseFeedbackExtractor(
                history_store=store,
                reserve_bus_feedback_store=feedback_store,
            )
            turn, consolidation = _turn_and_consolidation(
                turn_id="turn:display:ignored",
                occurred_at=shown_at + timedelta(hours=3),
            )
            batch = PersonalitySignalBatch(
                interaction_signals=(
                    InteractionSignal(
                        signal_id="signal:interaction:test:affinity",
                        signal_kind=INTERACTION_SIGNAL_TOPIC_AFFINITY,
                        target="AI companions",
                        summary="The user kept going with AI companions instead.",
                        confidence=0.83,
                        impact=0.28,
                        evidence_count=2,
                        source_event_ids=("turn:display:ignored",),
                    ),
                ),
                continuity_threads=(
                    ContinuityThread(
                        title="AI companions",
                        summary="The conversation kept deepening there.",
                        salience=0.8,
                    ),
                ),
            )

            feedback_batch = extractor.extract_from_consolidation(
                turn=turn,
                consolidation=consolidation,
                extracted_batch=batch,
            )
            history = store.load()
            reserve_feedback = feedback_store.load_active(now=shown_at + timedelta(hours=3))

        self.assertEqual(len(feedback_batch.interaction_signals), 1)
        self.assertEqual(feedback_batch.interaction_signals[0].signal_kind, INTERACTION_SIGNAL_TOPIC_COOLING)
        self.assertEqual(feedback_batch.interaction_signals[0].metadata["engagement_kind"], "display_non_reengagement")
        self.assertEqual(history[-1].response_status, "ignored")
        self.assertEqual(history[-1].response_mode, "no_voice_pickup")
        self.assertIsNotNone(reserve_feedback)
        assert reserve_feedback is not None
        self.assertEqual(reserve_feedback.reaction, "ignored")


if __name__ == "__main__":
    unittest.main()
