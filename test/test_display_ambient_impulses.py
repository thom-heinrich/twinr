from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.display.ambient_impulse_cues import (
    DisplayAmbientImpulseController,
    DisplayAmbientImpulseCue,
    DisplayAmbientImpulseCueStore,
)
from twinr.agent.personality.display_impulses import AmbientDisplayImpulseCandidate
from twinr.proactive.runtime.display_ambient_impulses import DisplayAmbientImpulsePublisher


class DisplayAmbientImpulseCueTests(unittest.TestCase):
    def test_from_dict_normalizes_text_symbol_and_timestamps(self) -> None:
        cue = DisplayAmbientImpulseCue.from_dict(
            {
                "source": " proactive ",
                "topic_key": " AI companions ",
                "eyebrow": " gemeinsamer faden ",
                "headline": "AI companions",
                "body": "Wenn du magst, schauen wir spaeter kurz darauf.",
                "symbol": " HEART ",
                "accent": "WARM",
                "action": "INVITE FOLLOW UP",
            },
            fallback_updated_at=datetime(2026, 3, 22, 9, 0, tzinfo=timezone.utc),
            default_ttl_s=18.0,
        )

        self.assertEqual(cue.source, "proactive")
        self.assertEqual(cue.topic_key, "ai companions")
        self.assertEqual(cue.symbol, "heart")
        self.assertEqual(cue.accent, "warm")
        self.assertEqual(cue.action, "invite_follow_up")
        self.assertEqual(cue.glyph(), "❤️")
        self.assertIsNotNone(cue.updated_at)
        self.assertIsNotNone(cue.expires_at)

    def test_store_roundtrip_and_expiry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            store = DisplayAmbientImpulseCueStore.from_config(config)
            now = datetime(2026, 3, 22, 9, 5, tzinfo=timezone.utc)

            saved = store.save(
                DisplayAmbientImpulseCue(
                    source="ambient",
                    topic_key="ai companions",
                    eyebrow="IM BLICK",
                    headline="AI companions",
                    body="Da gibt es gerade etwas Neues.",
                ),
                hold_seconds=4.0,
                now=now,
            )
            loaded = store.load_active(now=now + timedelta(seconds=1))
            expired = store.load_active(now=now + timedelta(seconds=5))

        self.assertEqual(loaded, saved)
        self.assertIsNone(expired)

    def test_controller_persists_one_active_impulse(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            controller = DisplayAmbientImpulseController.from_config(config, default_source="runtime")
            now = datetime(2026, 3, 22, 9, 10, tzinfo=timezone.utc)

            saved = controller.show_impulse(
                topic_key="hamburg local politics",
                eyebrow="IM BLICK",
                headline="Hamburg local politics",
                body="Da lohnt sich ein kurzer Blick.",
                symbol="sparkles",
                accent="info",
                hold_seconds=8.0,
                now=now,
            )
            loaded = controller.store.load_active(now=now + timedelta(seconds=1))

        self.assertEqual(saved.source, "runtime")
        self.assertEqual(saved.topic_key, "hamburg local politics")
        self.assertEqual(saved.symbol, "sparkles")
        self.assertEqual(saved.accent, "info")
        self.assertEqual(loaded, saved)


class DisplayAmbientImpulsePublisherTests(unittest.TestCase):
    def test_publisher_publishes_next_planned_item_when_surface_is_free(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                display_driver="hdmi_wayland",
                display_reserve_bus_items_per_day=4,
            )
            publisher = DisplayAmbientImpulsePublisher.from_config(config)
            publisher.candidate_loader = lambda _config, *, local_now, max_items: (
                AmbientDisplayImpulseCandidate(
                    topic_key="ai companions",
                    title="AI companions",
                    source="world",
                    action="invite_follow_up",
                    attention_state="shared_thread",
                    salience=0.92,
                    eyebrow="GEMEINSAMER FADEN",
                    headline="AI companions",
                    body="Wenn du magst, schauen wir spaeter kurz darauf.",
                    symbol="heart",
                    accent="warm",
                    reason=f"test@{local_now.date().isoformat()}",
                ),
            )[:max_items]
            now = datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc)

            result = publisher.publish_if_due(
                config=config,
                monotonic_now=100.0,
                runtime_status="waiting",
                presence_active=True,
                local_now=now,
            )
            cue = publisher.active_store.load_active(now=now + timedelta(seconds=1))
            plan = publisher.planner.store.load()

        self.assertEqual(result.action, "published")
        self.assertIsNotNone(cue)
        self.assertIsNotNone(plan)
        assert cue is not None
        assert plan is not None
        self.assertEqual(cue.topic_key, "ai companions")
        self.assertEqual(plan.cursor, 1)

    def test_publisher_blocks_when_an_ambient_impulse_is_already_active(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                display_driver="hdmi_wayland",
                display_reserve_bus_items_per_day=4,
            )
            publisher = DisplayAmbientImpulsePublisher.from_config(config)
            now = datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc)
            publisher.active_store.save(
                DisplayAmbientImpulseCue(
                    source="proactive_ambient_impulse",
                    topic_key="world politics",
                    headline="Weltpolitik",
                    body="Das bleibt gerade im Blick.",
                ),
                hold_seconds=600.0,
                now=now,
            )

            result = publisher.publish_if_due(
                config=config,
                monotonic_now=100.0,
                runtime_status="waiting",
                presence_active=True,
                local_now=now + timedelta(seconds=30),
            )

        self.assertEqual(result.action, "blocked")
        self.assertEqual(result.reason, "ambient_impulse_active")


if __name__ == "__main__":
    unittest.main()
