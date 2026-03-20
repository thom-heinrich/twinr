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
                    "primary_person_center_y": 0.12,
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

    def test_derives_session_focus_cue_from_serialized_attention_target(self) -> None:
        decision = derive_display_attention_cue(
            config=TwinrConfig(proactive_capture_interval_s=1.0),
            live_facts={
                "camera": {
                    "person_visible": False,
                },
                "attention_target": {
                    "state": "holding_session_focus",
                    "active": True,
                    "target_horizontal": "left",
                    "target_vertical": "up",
                    "focus_source": "speaker_association",
                    "session_focus_active": True,
                    "speaker_locked": False,
                    "showing_intent_active": False,
                    "confidence": 0.81,
                },
            },
        )

        self.assertTrue(decision.active)
        self.assertEqual(decision.reason, "holding_session_focus")
        self.assertEqual(decision.gaze, DisplayFaceGazeDirection.LEFT)
        self.assertEqual(decision.expression().brows, "soft")

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

    def test_publisher_refreshes_matching_cue_before_it_expires(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                proactive_capture_interval_s=6.0,
                display_attention_refresh_interval_s=1.25,
                display_face_cue_ttl_s=4.0,
            )
            publisher = DisplayAttentionCuePublisher.from_config(config)
            first_result = publisher.publish_from_facts(
                config=config,
                live_facts={
                    "camera": {
                        "person_visible": True,
                        "primary_person_center_x": 0.12,
                    },
                    "vad": {
                        "speech_detected": False,
                    },
                },
                now=datetime(2026, 3, 20, 8, 3, tzinfo=timezone.utc),
            )
            initial = publisher.store.load_active(now=datetime(2026, 3, 20, 8, 3, 0, 1000, tzinfo=timezone.utc))
            assert initial is not None

            refresh_result = publisher.publish_from_facts(
                config=config,
                live_facts={
                    "camera": {
                        "person_visible": True,
                        "primary_person_center_x": 0.12,
                    },
                    "vad": {
                        "speech_detected": False,
                    },
                },
                now=datetime(2026, 3, 20, 8, 3, 2, 700000, tzinfo=timezone.utc),
            )
            refreshed = publisher.store.load_active(now=datetime(2026, 3, 20, 8, 3, 4, 200000, tzinfo=timezone.utc))

        self.assertEqual(first_result.action, "updated")
        self.assertEqual(refresh_result.action, "refreshed")
        self.assertIsNotNone(refreshed)
        assert refreshed is not None
        self.assertEqual(refreshed.gaze_x, -2)
        self.assertGreater(str(refreshed.expires_at), str(initial.expires_at))


if __name__ == "__main__":
    unittest.main()
