from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.display.face_cues import DisplayFaceCue, DisplayFaceCueStore
from twinr.display.face_expressions import (
    DisplayFaceBrowStyle,
    DisplayFaceEmotion,
    DisplayFaceExpression,
    DisplayFaceExpressionController,
    DisplayFaceGazeDirection,
    DisplayFaceMouthStyle,
)


class DisplayFaceCueTests(unittest.TestCase):
    def test_from_dict_clamps_axes_and_normalizes_styles(self) -> None:
        cue = DisplayFaceCue.from_dict(
            {
                "gaze_x": 9,
                "gaze_y": -9,
                "head_dx": 4,
                "head_dy": -4,
                "mouth": "SMILE",
                "brows": "focus",
                "blink": "true",
            },
            fallback_updated_at=datetime(2026, 3, 18, 15, 0, tzinfo=timezone.utc),
            default_ttl_s=6.0,
        )

        self.assertEqual(cue.gaze_x, 3)
        self.assertEqual(cue.gaze_y, -3)
        self.assertEqual(cue.head_dx, 2)
        self.assertEqual(cue.head_dy, -2)
        self.assertEqual(cue.mouth, "smile")
        self.assertEqual(cue.brows, "inward_tilt")
        self.assertTrue(cue.blink)
        self.assertIsNotNone(cue.updated_at)
        self.assertIsNotNone(cue.expires_at)

    def test_from_dict_normalizes_legacy_alias_styles_to_canonical_values(self) -> None:
        cue = DisplayFaceCue.from_dict(
            {
                "mouth": "line",
                "brows": "flat",
            },
            fallback_updated_at=datetime(2026, 3, 18, 15, 0, tzinfo=timezone.utc),
            default_ttl_s=4.0,
        )

        self.assertEqual(cue.mouth, "neutral")
        self.assertEqual(cue.brows, "straight")

    def test_store_resolves_relative_path_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = DisplayFaceCueStore.from_config(TwinrConfig(project_root=temp_dir))

        self.assertEqual(store.path, Path(temp_dir) / "artifacts" / "stores" / "ops" / "display_face_cue.json")

    def test_store_roundtrip_and_expiry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            store = DisplayFaceCueStore.from_config(config)
            now = datetime(2026, 3, 18, 16, 0, tzinfo=timezone.utc)

            saved = store.save(
                DisplayFaceCue(gaze_x=1, mouth="open", brows="raised"),
                hold_seconds=3.0,
                now=now,
            )
            loaded = store.load_active(now=now + timedelta(seconds=1))
            expired = store.load_active(now=now + timedelta(seconds=4))

        self.assertEqual(loaded, saved)
        self.assertIsNone(expired)

    def test_clear_removes_existing_cue_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            store = DisplayFaceCueStore.from_config(config)
            store.save(DisplayFaceCue(mouth="neutral"), hold_seconds=5.0)

            store.clear()

            self.assertFalse(store.path.exists())

    def test_expression_to_cue_maps_discrete_gaze_and_styles(self) -> None:
        expression = DisplayFaceExpression(
            gaze=DisplayFaceGazeDirection.UP_RIGHT,
            mouth=DisplayFaceMouthStyle.THINKING,
            brows=DisplayFaceBrowStyle.ROOF,
            blink=True,
            head_dx=1,
            head_dy=-1,
        )

        cue = expression.to_cue(source="camera_surface")

        self.assertEqual(cue.source, "camera_surface")
        self.assertEqual(cue.gaze_x, 2)
        self.assertEqual(cue.gaze_y, -2)
        self.assertEqual(cue.mouth, "thinking")
        self.assertEqual(cue.brows, "roof")
        self.assertTrue(cue.blink)
        self.assertEqual(cue.head_dx, 1)
        self.assertEqual(cue.head_dy, -1)

    def test_expression_controller_builds_emotion_and_override_without_raw_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            controller = DisplayFaceExpressionController.from_config(config, default_source="camera_surface")
            now = datetime(2026, 3, 18, 16, 30, tzinfo=timezone.utc)

            saved = controller.show(
                emotion=DisplayFaceEmotion.HAPPY,
                gaze=DisplayFaceGazeDirection.LEFT,
                mouth=DisplayFaceMouthStyle.SCRUNCHED,
                hold_seconds=8.0,
                now=now,
            )
            loaded = controller.store.load_active(now=now + timedelta(seconds=1))

        self.assertEqual(saved.source, "camera_surface")
        self.assertEqual(saved.gaze_x, -2)
        self.assertEqual(saved.gaze_y, 0)
        self.assertEqual(saved.mouth, "scrunched")
        self.assertEqual(saved.brows, "raised")
        self.assertEqual(loaded, saved)


if __name__ == "__main__":
    unittest.main()
