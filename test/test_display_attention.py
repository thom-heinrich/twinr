from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.display.face_cues import DisplayFaceCue
from twinr.display.face_expressions import DisplayFaceGazeDirection
from twinr.proactive.runtime.display_attention import (
    DisplayAttentionCuePublisher,
    derive_display_attention_cue,
)


class DisplayAttentionCueTests(unittest.TestCase):
    def test_derives_leftward_gaze_from_primary_person_anchor(self) -> None:
        decision = derive_display_attention_cue(
            config=TwinrConfig(proactive_capture_interval_s=6.0, display_attention_refresh_interval_s=1.25),
            live_facts={
                "camera": {
                    "person_visible": True,
                    "primary_person_center_x": 0.12,
                    "primary_person_center_y": 0.45,
                    "looking_toward_device": True,
                },
                "vad": {
                    "speech_detected": False,
                },
            },
        )

        self.assertTrue(decision.active)
        self.assertEqual(decision.reason, "visible_person")
        self.assertEqual(decision.gaze, DisplayFaceGazeDirection.LEFT)
        self.assertLess(decision.hold_seconds, 6.0)

    def test_derives_speaking_mouth_when_visible_person_is_current_speaker(self) -> None:
        decision = derive_display_attention_cue(
            config=TwinrConfig(proactive_capture_interval_s=1.0),
            live_facts={
                "camera": {
                    "person_visible": True,
                    "primary_person_zone": "center",
                    "engaged_with_device": True,
                },
                "vad": {
                    "speech_detected": True,
                },
                "speaker_association": {
                    "associated": True,
                },
            },
        )

        self.assertTrue(decision.active)
        self.assertEqual(decision.reason, "speaker_visible_person")
        self.assertEqual(decision.expression().mouth, "speak")

    def test_publisher_does_not_overwrite_active_foreign_cue(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir, proactive_capture_interval_s=1.0)
            publisher = DisplayAttentionCuePublisher.from_config(config)
            publisher.store.save(
                DisplayFaceCue(source="operator", gaze_x=2, mouth="smile"),
                hold_seconds=10.0,
                now=datetime(2026, 3, 20, 8, 0, tzinfo=timezone.utc),
            )

            result = publisher.publish_from_facts(
                config=config,
                live_facts={
                    "camera": {
                        "person_visible": True,
                        "primary_person_center_x": 0.1,
                    },
                    "vad": {
                        "speech_detected": False,
                    },
                },
                now=datetime(2026, 3, 20, 8, 0, 1, tzinfo=timezone.utc),
            )
            loaded = publisher.store.load_active(now=datetime(2026, 3, 20, 8, 0, 1, tzinfo=timezone.utc))

        self.assertEqual(result.action, "blocked_foreign_cue")
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.source, "operator")
        self.assertEqual(loaded.gaze_x, 2)

    def test_publisher_clears_its_own_cue_when_no_person_is_visible(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir, proactive_capture_interval_s=1.0)
            publisher = DisplayAttentionCuePublisher.from_config(config)
            publisher.publish_from_facts(
                config=config,
                live_facts={
                    "camera": {
                        "person_visible": True,
                        "primary_person_center_x": 0.82,
                    },
                    "vad": {
                        "speech_detected": False,
                    },
                },
                now=datetime(2026, 3, 20, 8, 2, tzinfo=timezone.utc),
            )

            result = publisher.publish_from_facts(
                config=config,
                live_facts={
                    "camera": {
                        "person_visible": False,
                    },
                    "vad": {
                        "speech_detected": False,
                    },
                },
                now=datetime(2026, 3, 20, 8, 2, 1, tzinfo=timezone.utc),
            )

            loaded = publisher.store.load_active(now=datetime(2026, 3, 20, 8, 2, 1, tzinfo=timezone.utc))

        self.assertEqual(result.action, "cleared")
        self.assertIsNone(loaded)


if __name__ == "__main__":
    unittest.main()
