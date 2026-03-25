from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.display.presentation_cues import (
    DisplayPresentationCardCue,
    DisplayPresentationController,
    DisplayPresentationCue,
    DisplayPresentationStore,
)


class DisplayPresentationCueTests(unittest.TestCase):
    def test_from_dict_normalizes_kind_text_and_lines(self) -> None:
        cue = DisplayPresentationCue.from_dict(
            {
                "kind": "RICH CARD",
                "title": "  Family Call  ",
                "subtitle": "Marta is waiting",
                "body_lines": ["Tap green and speak", "", "Say hello to Marta"],
                "accent": "warm",
            },
            fallback_updated_at=datetime(2026, 3, 18, 17, 0, tzinfo=timezone.utc),
            default_ttl_s=12.0,
        )

        self.assertEqual(cue.kind, "rich_card")
        self.assertEqual(cue.title, "Family Call")
        self.assertEqual(cue.body_lines, ("Tap green and speak", "Say hello to Marta"))
        self.assertEqual(cue.accent, "warm")

    def test_store_resolves_relative_path_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = DisplayPresentationStore.from_config(TwinrConfig(project_root=temp_dir))

        self.assertEqual(store.path, Path(temp_dir) / "artifacts" / "stores" / "ops" / "display_presentation.json")

    def test_store_roundtrip_and_progress_bucket(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            store = DisplayPresentationStore.from_config(config)
            now = datetime(2026, 3, 18, 17, 5, tzinfo=timezone.utc)

            saved = store.save(
                DisplayPresentationCue(kind="rich_card", title="Reminder", body_lines=("Tea time",)),
                hold_seconds=6.0,
                now=now,
            )
            loaded = store.load_active(now=now + timedelta(seconds=1))
            expired = store.load_active(now=now + timedelta(seconds=7))

        self.assertEqual(loaded, saved)
        assert loaded is not None
        self.assertGreaterEqual(loaded.transition_bucket(now=now + timedelta(milliseconds=200)), 1)
        self.assertIsNone(expired)

    def test_active_card_prefers_highest_priority_scene_card(self) -> None:
        cue = DisplayPresentationCue(
            cards=(
                DisplayPresentationCardCue(key="summary", title="Summary", priority=40, accent="info"),
                DisplayPresentationCardCue(key="photo", kind="image", title="Photo", priority=90, accent="success"),
                DisplayPresentationCardCue(key="alert", title="Alert", priority=70, accent="alert"),
            )
        )

        active = cue.active_card()
        queued = cue.queued_cards()

        self.assertIsNotNone(active)
        assert active is not None
        self.assertEqual(active.key, "photo")
        self.assertEqual([card.key for card in queued], ["alert", "summary"])
        self.assertEqual(cue.telemetry_signature(now=datetime(2026, 3, 18, 17, 7, tzinfo=timezone.utc))[:3], ("photo", "image", 90))

    def test_controller_persists_image_presentation_without_raw_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            controller = DisplayPresentationController.from_config(config, default_source="camera_surface")
            now = datetime(2026, 3, 18, 17, 10, tzinfo=timezone.utc)

            saved = controller.show_image(
                image_path="/tmp/test-card.png",
                title="Marta",
                subtitle="Photo sent now",
                body_lines=("Tap green to answer",),
                accent="success",
                hold_seconds=10.0,
                now=now,
            )
            loaded = controller.store.load_active(now=now + timedelta(seconds=1))

        self.assertEqual(saved.source, "camera_surface")
        self.assertEqual(saved.kind, "image")
        self.assertEqual(saved.image_path, "/tmp/test-card.png")
        self.assertEqual(saved.accent, "success")
        self.assertEqual(loaded, saved)

    def test_controller_persists_prioritized_scene_without_raw_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            controller = DisplayPresentationController.from_config(config, default_source="operator")
            now = datetime(2026, 3, 18, 17, 12, tzinfo=timezone.utc)

            saved = controller.show_scene(
                cards=(
                    DisplayPresentationCardCue(key="summary", title="Summary", priority=20, accent="info"),
                    DisplayPresentationCardCue(
                        key="family_photo",
                        kind="image",
                        title="Family Photo",
                        image_path="/tmp/family.png",
                        priority=80,
                        accent="warm",
                        face_emotion="happy",
                    ),
                ),
                hold_seconds=10.0,
                now=now,
            )
            loaded = controller.store.load_active(now=now + timedelta(seconds=1))

        self.assertEqual(saved.active_card_key, "family_photo")
        self.assertEqual(saved.active_card().key if saved.active_card() is not None else None, "family_photo")
        self.assertEqual(len(saved.normalized_cards()), 2)
        self.assertEqual(loaded, saved)


if __name__ == "__main__":
    unittest.main()
