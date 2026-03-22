from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.display.ambient_impulse_cues import DisplayAmbientImpulseCue
from twinr.display.emoji_cues import DisplayEmojiCue
from twinr.display.reserve_bus import resolve_display_reserve_bus


class DisplayReserveBusTests(unittest.TestCase):
    def test_resolve_prefers_emoji_over_ambient_impulse(self) -> None:
        state = resolve_display_reserve_bus(
            emoji_cue=DisplayEmojiCue(symbol="thumbs_up", accent="success"),
            ambient_impulse_cue=DisplayAmbientImpulseCue(
                topic_key="ai companions",
                headline="AI companions",
                body="Da gibt es gerade etwas Neues.",
            ),
        )

        self.assertEqual(state.owner, "emoji")
        self.assertIsNotNone(state.emoji_cue)
        self.assertIsNone(state.ambient_impulse_cue)

    def test_resolve_uses_ambient_impulse_when_no_emoji_is_active(self) -> None:
        state = resolve_display_reserve_bus(
            emoji_cue=None,
            ambient_impulse_cue=DisplayAmbientImpulseCue(
                topic_key="world politics",
                headline="Weltpolitik",
                body="Da lohnt sich heute ein kurzer Blick.",
            ),
        )

        self.assertEqual(state.owner, "ambient_impulse")
        self.assertIsNone(state.emoji_cue)
        self.assertIsNotNone(state.ambient_impulse_cue)

    def test_resolve_returns_explicit_empty_state_when_nothing_is_active(self) -> None:
        state = resolve_display_reserve_bus(
            emoji_cue=None,
            ambient_impulse_cue=None,
        )

        self.assertEqual(state.owner, "empty")
        self.assertEqual(state.reason, "empty")
        self.assertEqual(state.signature()[0], "empty")


if __name__ == "__main__":
    unittest.main()
