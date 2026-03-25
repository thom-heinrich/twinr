from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.display.emoji_cues import DisplayEmojiController, DisplayEmojiCue, DisplayEmojiCueStore, DisplayEmojiSymbol


class DisplayEmojiCueTests(unittest.TestCase):
    def test_from_dict_normalizes_symbol_accent_and_timestamps(self) -> None:
        cue = DisplayEmojiCue.from_dict(
            {
                "source": " camera_surface ",
                "symbol": "WAVING HAND",
                "accent": "SUCCESS",
            },
            fallback_updated_at=datetime(2026, 3, 20, 17, 0, tzinfo=timezone.utc),
            default_ttl_s=8.0,
        )

        self.assertEqual(cue.source, "camera_surface")
        self.assertEqual(cue.symbol, "waving_hand")
        self.assertEqual(cue.accent, "success")
        self.assertEqual(cue.glyph(), "👋")
        self.assertIsNotNone(cue.updated_at)
        self.assertIsNotNone(cue.expires_at)

    def test_store_resolves_relative_path_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = DisplayEmojiCueStore.from_config(TwinrConfig(project_root=temp_dir))

        self.assertEqual(store.path, Path(temp_dir) / "artifacts" / "stores" / "ops" / "display_emoji.json")

    def test_store_roundtrip_and_expiry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            store = DisplayEmojiCueStore.from_config(config)
            now = datetime(2026, 3, 20, 17, 5, tzinfo=timezone.utc)

            saved = store.save(
                DisplayEmojiCue(symbol="thumbs_up", accent="success"),
                hold_seconds=4.0,
                now=now,
            )
            loaded = store.load_active(now=now + timedelta(seconds=1))
            expired = store.load_active(now=now + timedelta(seconds=5))

        self.assertEqual(loaded, saved)
        self.assertIsNone(expired)

    def test_controller_persists_explicit_non_face_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            controller = DisplayEmojiController.from_config(config, default_source="runtime")
            now = datetime(2026, 3, 20, 17, 10, tzinfo=timezone.utc)

            saved = controller.show_symbol(
                DisplayEmojiSymbol.THUMBS_UP,
                accent="success",
                hold_seconds=6.0,
                now=now,
            )
            loaded = controller.store.load_active(now=now + timedelta(seconds=1))

        self.assertEqual(saved.source, "runtime")
        self.assertEqual(saved.symbol, "thumbs_up")
        self.assertEqual(saved.accent, "success")
        self.assertEqual(loaded, saved)


if __name__ == "__main__":
    unittest.main()
