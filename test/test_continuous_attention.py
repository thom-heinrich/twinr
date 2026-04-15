from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent import TwinrConfig
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
        self.assertAlmostEqual(snapshot.target_center_x or 0.0, 0.51, places=2)
        self.assertAlmostEqual(snapshot.target_center_y or 0.0, 0.24, places=2)

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

    def test_single_visible_speaker_blends_audio_direction_into_target_center(self) -> None:
        tracker = ContinuousAttentionTracker.from_config(TwinrConfig())

        tracker.observe(
            observed_at=0.0,
            live_facts={
                "camera": {
                    "person_visible": True,
                    "person_count": 1,
                    "visible_persons": [
                        {
                            "box": {"top": 0.1, "left": 0.7, "bottom": 0.9, "right": 0.9},
                            "zone": "right",
                            "confidence": 0.92,
                        },
                    ],
                },
                "vad": {"speech_detected": False},
            },
        )

        snapshot = tracker.observe(
            observed_at=0.5,
            live_facts={
                "camera": {
                    "person_visible": True,
                    "person_count": 1,
                    "visible_persons": [
                        {
                            "box": {"top": 0.1, "left": 0.7, "bottom": 0.9, "right": 0.9},
                            "zone": "right",
                            "confidence": 0.92,
                        },
                    ],
                },
                "vad": {"speech_detected": True},
                "respeaker": {
                    "azimuth_deg": 74,
                    "direction_confidence": None,
                },
                "audio_policy": {"speaker_direction_stable": None},
            },
        )

        self.assertEqual(snapshot.state, "active_visible_speaker_track")
        self.assertTrue(snapshot.speaker_locked)
        self.assertAlmostEqual(snapshot.target_center_x or 0.0, 0.748, places=3)
        self.assertGreaterEqual(snapshot.confidence, 0.6)

    def test_does_not_hold_stale_selected_track_over_fresh_visible_anchor(self) -> None:
        tracker = ContinuousAttentionTracker.from_config(TwinrConfig())

        initial = tracker.observe(
            observed_at=0.0,
            live_facts={
                "camera": {
                    "person_visible": True,
                    "person_count": 2,
                    "visible_persons": [
                        {
                            "box": {"top": 0.1, "left": 0.05, "bottom": 0.9, "right": 0.25},
                            "zone": "left",
                            "confidence": 0.85,
                        },
                        {
                            "box": {"top": 0.1, "left": 0.8, "bottom": 0.9, "right": 0.98},
                            "zone": "right",
                            "confidence": 0.95,
                        },
                    ],
                },
                "vad": {"speech_detected": False},
            },
        )

        follow = tracker.observe(
            observed_at=0.4,
            live_facts={
                "camera": {
                    "person_visible": True,
                    "person_count": 1,
                    "visible_persons": [
                        {
                            "box": {"top": 0.1, "left": 0.44, "bottom": 0.9, "right": 0.66},
                            "zone": "center",
                            "confidence": 0.9,
                        },
                    ],
                    "primary_person_center_x": 0.55,
                    "primary_person_center_y": 0.5,
                },
                "vad": {"speech_detected": False},
            },
        )

        self.assertEqual(initial.target_track_id, "visible_track_2")
        self.assertEqual(initial.state, "active_visible_person")
        self.assertEqual(follow.state, "active_visible_person")
        self.assertEqual(follow.focus_source, "primary_visible_person")
        self.assertAlmostEqual(follow.target_center_x or 0.0, 0.55, places=2)

    def test_single_visible_person_prefers_stabilized_primary_anchor_over_raw_box_center(self) -> None:
        tracker = ContinuousAttentionTracker.from_config(TwinrConfig())

        snapshot = tracker.observe(
            observed_at=0.0,
            live_facts={
                "camera": {
                    "person_visible": True,
                    "person_count": 1,
                    "visible_persons": [
                        {
                            "box": {"top": 0.1, "left": 0.82, "bottom": 0.92, "right": 0.98},
                            "zone": "right",
                            "confidence": 0.92,
                        },
                    ],
                    "primary_person_center_x": 0.76,
                    "primary_person_center_y": 0.66,
                    "primary_person_zone": "right",
                    "visual_attention_score": 0.88,
                },
                "vad": {"speech_detected": False},
            },
        )

        self.assertTrue(snapshot.active)
        self.assertEqual(snapshot.state, "active_visible_person")
        self.assertEqual(snapshot.focus_source, "primary_visible_person")
        self.assertAlmostEqual(snapshot.target_center_x or 0.0, 0.76, places=2)
        self.assertAlmostEqual(snapshot.target_center_y or 0.0, 0.51, places=2)


if __name__ == "__main__":
    unittest.main()
