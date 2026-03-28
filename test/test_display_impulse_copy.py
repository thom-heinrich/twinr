from datetime import datetime, timezone
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.personality.display_impulse_copy import build_ambient_display_impulse_copy
from twinr.agent.personality.positive_engagement import PositiveEngagementTopicPolicy
from twinr.agent.personality.self_expression import CompanionMindshareItem, ConversationAppetiteCue


class DisplayImpulseCopyTests(unittest.TestCase):
    def test_world_copy_uses_statement_headline_and_cta_body(self) -> None:
        copy = build_ambient_display_impulse_copy(
            CompanionMindshareItem(
                title="AI companions",
                summary="Personal AI companions are getting more useful.",
                salience=0.82,
                source="world",
                appetite=ConversationAppetiteCue(),
            ),
            PositiveEngagementTopicPolicy(
                title="AI companions",
                salience=0.82,
                attention_state="shared_thread",
                action="ask_one",
                reason="shared_thread_ask_one",
            ),
            local_now=datetime(2026, 3, 25, 10, 0, tzinfo=timezone.utc),
        )

        self.assertFalse(copy.headline.endswith("?"))
        self.assertTrue(copy.body)
        self.assertNotEqual(copy.body, copy.headline)

    def test_memory_copy_uses_statement_headline_and_cta_body(self) -> None:
        copy = build_ambient_display_impulse_copy(
            CompanionMindshareItem(
                title="Dein Kaffee am Morgen",
                summary="This matters for continuity.",
                salience=0.76,
                source="continuity",
                appetite=ConversationAppetiteCue(),
            ),
            PositiveEngagementTopicPolicy(
                title="Dein Kaffee am Morgen",
                salience=0.76,
                attention_state="forming",
                action="ask_one",
                reason="forming_thread_ask_one",
            ),
            local_now=datetime(2026, 3, 25, 10, 0, tzinfo=timezone.utc),
        )

        self.assertFalse(copy.headline.endswith("?"))
        self.assertTrue(copy.body)
        self.assertNotEqual(copy.body, copy.headline)


if __name__ == "__main__":
    unittest.main()
