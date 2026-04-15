from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.display.ambient_impulse_cues import DisplayAmbientImpulseCue
from twinr.display.emoji_cues import DisplayEmojiCue
from twinr.display.reserve_bus import resolve_display_reserve_bus
from twinr.display.service_connect_cues import DisplayServiceConnectCue


class DisplayReserveBusTests(unittest.TestCase):
    def test_signature_ignores_reserve_cue_lifetime_refresh(self) -> None:
        cases = (
            (
                "service_connect",
                lambda updated_at, expires_at: resolve_display_reserve_bus(
                    service_connect_cue=DisplayServiceConnectCue(
                        source="service_connect",
                        updated_at=updated_at,
                        expires_at=expires_at,
                        service_id="whatsapp",
                        service_label="WhatsApp",
                        phase="qr",
                        summary="Scan QR",
                        detail="Use WhatsApp Linked Devices.",
                    ),
                    emoji_cue=None,
                    ambient_impulse_cue=None,
                ),
            ),
            (
                "emoji",
                lambda updated_at, expires_at: resolve_display_reserve_bus(
                    service_connect_cue=None,
                    emoji_cue=DisplayEmojiCue(
                        source="gesture_ack",
                        updated_at=updated_at,
                        expires_at=expires_at,
                        symbol="thumbs_up",
                        accent="success",
                    ),
                    ambient_impulse_cue=None,
                ),
            ),
            (
                "ambient_impulse",
                lambda updated_at, expires_at: resolve_display_reserve_bus(
                    service_connect_cue=None,
                    emoji_cue=None,
                    ambient_impulse_cue=DisplayAmbientImpulseCue(
                        source="ambient",
                        updated_at=updated_at,
                        expires_at=expires_at,
                        topic_key="world politics",
                        headline="Weltpolitik",
                        body="Da lohnt sich heute ein kurzer Blick.",
                    ),
                ),
            ),
        )

        for owner, builder in cases:
            with self.subTest(owner=owner):
                first = builder("2026-04-07T08:00:00+00:00", "2026-04-07T08:00:04+00:00")
                refreshed = builder("2026-04-07T08:00:03+00:00", "2026-04-07T08:00:07+00:00")
                self.assertEqual(first.signature(), refreshed.signature())

    def test_resolve_prefers_service_connect_over_other_cues(self) -> None:
        state = resolve_display_reserve_bus(
            service_connect_cue=DisplayServiceConnectCue(
                service_id="whatsapp",
                service_label="WhatsApp",
                phase="qr",
                summary="Scan QR",
                detail="Use WhatsApp Linked Devices.",
            ),
            emoji_cue=DisplayEmojiCue(symbol="thumbs_up", accent="success"),
            ambient_impulse_cue=DisplayAmbientImpulseCue(
                topic_key="ai companions",
                headline="AI companions",
                body="Da gibt es gerade etwas Neues.",
            ),
        )

        self.assertEqual(state.owner, "service_connect")
        self.assertIsNotNone(state.service_connect_cue)
        self.assertIsNone(state.emoji_cue)
        self.assertIsNone(state.ambient_impulse_cue)

    def test_resolve_prefers_emoji_over_ambient_impulse(self) -> None:
        state = resolve_display_reserve_bus(
            service_connect_cue=None,
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
            service_connect_cue=None,
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
            service_connect_cue=None,
            emoji_cue=None,
            ambient_impulse_cue=None,
        )

        self.assertEqual(state.owner, "empty")
        self.assertEqual(state.reason, "empty")
        self.assertEqual(state.signature()[0], "empty")


if __name__ == "__main__":
    unittest.main()
