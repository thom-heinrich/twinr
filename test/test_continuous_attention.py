from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.proactive.runtime.continuous_attention import ContinuousAttentionTracker


class ContinuousAttentionTrackerTests(unittest.TestCase):
    def test_prefers_visible_person_anchor_over_primary_body_center_for_target(self) -> None:
        tracker = ContinuousAttentionTracker.from_config(TwinrConfig())

        snapshot = tracker.observe(
            observed_at=0.0,
            live_facts={
                "camera": {
                    "person_visible": True,
                    "person_count": 1,
                    "visible_persons": [
                        {
                            "box": {"top": 0.14, "left": 0.44, "bottom": 0.34, "right": 0.58},
                            "zone": "center",
                            "confidence": 0.93,
                        },
                    ],
                    "primary_person_center_x": 0.51,
                    "primary_person_center_y": 0.78,
                },
                "vad": {"speech_detected": False},
            },
        )

        self.assertTrue(snapshot.active)
        self.assertEqual(snapshot.state, "active_visible_person")
        self.assertAlmostEqual(snapshot.target_center_x, 0.51, places=2)
        self.assertAlmostEqual(snapshot.target_center_y, 0.24, places=2)

    def test_keeps_recent_unmatched_second_track_through_brief_detector_drop(self) -> None:
        tracker = ContinuousAttentionTracker.from_config(TwinrConfig())

        tracker.observe(
            observed_at=0.0,
            live_facts={
                "camera": {
                    "person_visible": True,
                    "person_count": 2,
                    "visible_persons": [
                        {
                            "box": {"top": 0.1, "left": 0.05, "bottom": 0.9, "right": 0.35},
                            "zone": "left",
                            "confidence": 0.91,
                        },
                        {
                            "box": {"top": 0.1, "left": 0.62, "bottom": 0.9, "right": 0.9},
                            "zone": "right",
                            "confidence": 0.9,
                        },
                    ],
                },
                "vad": {"speech_detected": False},
            },
        )

        tracker.observe(
            observed_at=0.3,
            live_facts={
                "camera": {
                    "person_visible": True,
                    "person_count": 2,
                    "visible_persons": [
                        {
                            "box": {"top": 0.1, "left": 0.1, "bottom": 0.9, "right": 0.4},
                            "zone": "left",
                            "confidence": 0.91,
                        },
                        {
                            "box": {"top": 0.1, "left": 0.62, "bottom": 0.9, "right": 0.9},
                            "zone": "right",
                            "confidence": 0.9,
                        },
                    ],
                },
                "vad": {"speech_detected": False},
            },
        )

        snapshot = tracker.observe(
            observed_at=0.6,
            live_facts={
                "camera": {
                    "person_visible": True,
                    "person_count": 1,
                    "visible_persons": [
                        {
                            "box": {"top": 0.1, "left": 0.62, "bottom": 0.9, "right": 0.9},
                            "zone": "right",
                            "confidence": 0.9,
                        },
                    ],
                },
                "vad": {"speech_detected": True},
                "respeaker": {
                    "azimuth_deg": None,
                    "direction_confidence": None,
                },
                "audio_policy": {"speaker_direction_stable": None},
            },
        )

        self.assertEqual(snapshot.visible_track_count, 2)
        self.assertEqual(snapshot.state, "last_moved_visible_person")
        self.assertEqual(snapshot.focus_source, "last_motion_track")
        self.assertFalse(snapshot.speaker_locked)


if __name__ == "__main__":
    unittest.main()
