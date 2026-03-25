from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.personality.intelligence.models import (
    SituationalAwarenessThread,
    WorldFeedSubscription,
    WorldIntelligenceState,
)
from twinr.proactive.runtime.display_reserve_world import load_display_reserve_world_candidates


class DisplayReserveWorldTests(unittest.TestCase):
    def test_world_candidates_surface_awareness_threads_and_subscription_breadth(self) -> None:
        subscriptions = (
            WorldFeedSubscription(
                subscription_id="feed:hamburg-local",
                label="Hamburg Lokalpolitik",
                feed_url="https://example.test/hamburg.xml",
                scope="regional",
                priority=0.84,
                topics=("hamburg local politics",),
            ),
            WorldFeedSubscription(
                subscription_id="feed:schwarzenbek-local",
                label="Schwarzenbek / Herzogtum Lauenburg",
                feed_url="https://example.test/schwarzenbek.xml",
                scope="regional",
                priority=0.81,
                topics=("schwarzenbek civic life",),
            ),
            WorldFeedSubscription(
                subscription_id="feed:mit-ai",
                label="MIT AI",
                feed_url="https://example.test/mit-ai.xml",
                scope="topic",
                priority=0.86,
                topics=("AI companions", "agentic ai"),
            ),
            WorldFeedSubscription(
                subscription_id="feed:verge-ai",
                label="The Verge AI",
                feed_url="https://example.test/verge-ai.xml",
                scope="topic",
                priority=0.82,
                topics=("AI companions", "agentic ai"),
            ),
        )
        state = WorldIntelligenceState(
            awareness_threads=(
                SituationalAwarenessThread(
                    thread_id="aware:hamburg",
                    title="Hamburg plant neue Schulwege",
                    summary="Mehrere neue Meldungen drehen sich um sichere Wege und Umbauten.",
                    topic="hamburg local politics",
                    scope="regional",
                    salience=0.78,
                    update_count=3,
                    source_labels=("Hamburg Lokalpolitik",),
                ),
            ),
        )

        candidates = load_display_reserve_world_candidates(
            subscriptions=subscriptions,
            state=state,
            max_items=10,
        )

        by_topic = {candidate.topic_key: candidate for candidate in candidates}
        self.assertIn("hamburg local politics", by_topic)
        self.assertIn("schwarzenbek civic life", by_topic)
        self.assertIn("agentic ai", by_topic)
        self.assertEqual(by_topic["hamburg local politics"].candidate_family, "world_awareness")
        self.assertEqual(by_topic["agentic ai"].candidate_family, "world_subscription")
        self.assertFalse(by_topic["agentic ai"].headline.endswith("?"))
        self.assertTrue(by_topic["agentic ai"].body.endswith("?"))
        self.assertGreater(
            by_topic["hamburg local politics"].salience,
            by_topic["schwarzenbek civic life"].salience,
        )

    def test_subscription_topics_are_grouped_across_multiple_feeds(self) -> None:
        subscriptions = (
            WorldFeedSubscription(
                subscription_id="feed:one",
                label="Source One",
                feed_url="https://example.test/one.xml",
                scope="topic",
                priority=0.78,
                topics=("AI companions",),
            ),
            WorldFeedSubscription(
                subscription_id="feed:two",
                label="Source Two",
                feed_url="https://example.test/two.xml",
                scope="topic",
                priority=0.84,
                topics=("AI companions",),
            ),
        )

        candidates = load_display_reserve_world_candidates(
            subscriptions=subscriptions,
            state=WorldIntelligenceState(),
            max_items=10,
        )

        by_topic = {candidate.topic_key: candidate for candidate in candidates}
        self.assertIn("ai companions", by_topic)
        grouped = by_topic["ai companions"]
        self.assertEqual(grouped.candidate_family, "world_subscription")
        self.assertEqual(grouped.action, "brief_update")
        self.assertEqual(grouped.attention_state, "growing")
        self.assertIn("source_count", grouped.generation_context or {})
        self.assertEqual(len(candidates), 1)


if __name__ == "__main__":
    unittest.main()
