from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.proactive.runtime.attention_targeting import (
    MultimodalAttentionTargetTracker,
)
from twinr.proactive.runtime.claim_metadata import RuntimeClaimMetadata
from twinr.proactive.runtime.identity_fusion import MultimodalIdentityFusionSnapshot
from twinr.proactive.runtime.speaker_association import ReSpeakerSpeakerAssociationSnapshot


class MultimodalAttentionTargetTests(unittest.TestCase):
    def test_prioritizes_visible_speaker_and_records_focus(self) -> None:
        tracker = MultimodalAttentionTargetTracker.from_config(TwinrConfig())

        snapshot = tracker.observe(
            observed_at=10.0,
            live_facts={
                "camera": {
                    "person_visible": True,
                    "primary_person_center_x": 0.14,
                    "primary_person_center_y": 0.14,
                },
                "vad": {
                    "speech_detected": True,
                },
            },
            runtime_status="listening",
            presence_session_id=7,
            speaker_association=ReSpeakerSpeakerAssociationSnapshot(
                state="primary_visible_person_associated",
                associated=True,
                confidence=0.87,
            ),
        )

        self.assertEqual(snapshot.state, "active_visible_speaker_track")
        self.assertTrue(snapshot.active)
        self.assertTrue(snapshot.speaker_locked)
        self.assertEqual(snapshot.target_horizontal, "left")
        self.assertEqual(snapshot.target_vertical, "center")
        self.assertEqual(snapshot.focus_source, "speaker_association")
        self.assertIsNotNone(snapshot.target_center_x)

    def test_visible_speaker_confidence_does_not_drop_below_active_track_floor(self) -> None:
        tracker = MultimodalAttentionTargetTracker.from_config(TwinrConfig())

        tracker.observe(
            observed_at=10.0,
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
                    "primary_person_center_x": 0.8,
                    "primary_person_center_y": 0.5,
                    "visual_attention_score": 0.18,
                },
                "vad": {"speech_detected": False},
            },
            runtime_status="waiting",
            presence_session_id=7,
        )

        snapshot = tracker.observe(
            observed_at=10.5,
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
                    "primary_person_center_x": 0.8,
                    "primary_person_center_y": 0.5,
                    "visual_attention_score": 0.18,
                },
                "vad": {"speech_detected": True},
                "respeaker": {
                    "azimuth_deg": 74,
                    "direction_confidence": None,
                },
                "audio_policy": {"speaker_direction_stable": None},
            },
            runtime_status="listening",
            presence_session_id=7,
            speaker_association=ReSpeakerSpeakerAssociationSnapshot(
                state="primary_visible_person_associated",
                associated=True,
                confidence=0.2,
            ),
        )

        self.assertEqual(snapshot.state, "active_visible_speaker_track")
        self.assertTrue(snapshot.active)
        self.assertTrue(snapshot.speaker_locked)
        self.assertGreaterEqual(snapshot.confidence, 0.6)

    def test_prefers_recently_moving_visible_person_when_room_is_quiet(self) -> None:
        tracker = MultimodalAttentionTargetTracker.from_config(TwinrConfig())

        first = tracker.observe(
            observed_at=10.0,
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
                            "confidence": 0.88,
                        },
                    ],
                    "primary_person_center_x": 0.2,
                    "primary_person_center_y": 0.5,
                },
                "vad": {"speech_detected": False},
            },
            runtime_status="waiting",
            presence_session_id=15,
        )

        second = tracker.observe(
            observed_at=10.3,
            live_facts={
                "camera": {
                    "person_visible": True,
                    "person_count": 2,
                    "visible_persons": [
                        {
                            "box": {"top": 0.1, "left": 0.10, "bottom": 0.9, "right": 0.40},
                            "zone": "left",
                            "confidence": 0.91,
                        },
                        {
                            "box": {"top": 0.1, "left": 0.62, "bottom": 0.9, "right": 0.9},
                            "zone": "right",
                            "confidence": 0.88,
                        },
                    ],
                    "primary_person_center_x": 0.25,
                    "primary_person_center_y": 0.5,
                },
                "vad": {"speech_detected": False},
            },
            runtime_status="waiting",
            presence_session_id=15,
        )

        self.assertEqual(first.state, "visible_primary_person")
        self.assertEqual(second.state, "last_moved_visible_person")
        self.assertEqual(second.focus_source, "last_motion_track")
        self.assertEqual(second.target_horizontal, "left")

    def test_prefers_recent_session_focus_over_new_primary_anchor_in_multi_person_context(self) -> None:
        tracker = MultimodalAttentionTargetTracker.from_config(TwinrConfig())
        tracker.observe(
            observed_at=10.0,
            live_facts={
                "camera": {
                    "person_visible": True,
                    "primary_person_center_x": 0.14,
                    "primary_person_center_y": 0.5,
                },
                "vad": {
                    "speech_detected": True,
                },
            },
            runtime_status="listening",
            presence_session_id=9,
            speaker_association=ReSpeakerSpeakerAssociationSnapshot(
                state="primary_visible_person_associated",
                associated=True,
                confidence=0.88,
            ),
        )

        snapshot = tracker.observe(
            observed_at=11.0,
            live_facts={
                "camera": {
                    "person_visible": True,
                    "person_count": 2,
                    "primary_person_center_x": 0.82,
                    "primary_person_center_y": 0.52,
                    "person_recently_visible": True,
                },
                "vad": {
                    "speech_detected": False,
                },
            },
            runtime_status="processing",
            presence_session_id=9,
        )

        self.assertEqual(snapshot.state, "session_focus_locked")
        self.assertTrue(snapshot.session_focus_active)
        self.assertEqual(snapshot.target_horizontal, "left")
        self.assertEqual(snapshot.focus_source, "speaker_association")

    def test_holds_session_focus_when_visual_anchor_temporarily_missing(self) -> None:
        tracker = MultimodalAttentionTargetTracker.from_config(TwinrConfig())
        tracker.observe(
            observed_at=20.0,
            live_facts={
                "camera": {
                    "person_visible": True,
                    "primary_person_center_x": 0.78,
                    "primary_person_center_y": 0.42,
                    "showing_intent_likely": True,
                },
                "vad": {
                    "speech_detected": False,
                },
            },
            runtime_status="listening",
            presence_session_id=11,
        )

        snapshot = tracker.observe(
            observed_at=22.0,
            live_facts={
                "camera": {
                    "person_visible": False,
                    "person_recently_visible": True,
                },
                "vad": {
                    "speech_detected": False,
                },
            },
            runtime_status="answering",
            presence_session_id=11,
        )

        self.assertEqual(snapshot.state, "holding_session_focus")
        self.assertTrue(snapshot.session_focus_active)
        self.assertEqual(snapshot.target_horizontal, "right")
        self.assertEqual(snapshot.focus_source, "showing_intent")

    def test_does_not_hold_session_focus_when_camera_is_explicitly_unavailable(self) -> None:
        tracker = MultimodalAttentionTargetTracker.from_config(TwinrConfig())
        tracker.observe(
            observed_at=20.0,
            live_facts={
                "camera": {
                    "person_visible": True,
                    "primary_person_center_x": 0.78,
                    "primary_person_center_y": 0.42,
                    "showing_intent_likely": True,
                },
                "vad": {
                    "speech_detected": False,
                },
            },
            runtime_status="listening",
            presence_session_id=11,
        )

        snapshot = tracker.observe(
            observed_at=22.0,
            live_facts={
                "camera": {
                    "person_visible": False,
                    "person_recently_visible": True,
                    "camera_online": False,
                    "camera_ready": False,
                },
                "vad": {
                    "speech_detected": False,
                },
            },
            runtime_status="answering",
            presence_session_id=11,
        )

        self.assertEqual(snapshot.state, "inactive")
        self.assertFalse(snapshot.active)

    def test_uses_identity_fusion_track_as_focus_seed_in_multi_person_scene(self) -> None:
        tracker = MultimodalAttentionTargetTracker.from_config(TwinrConfig())

        snapshot = tracker.observe(
            observed_at=30.0,
            live_facts={
                "camera": {
                    "person_visible": True,
                    "person_count": 2,
                    "primary_person_center_x": 0.82,
                    "primary_person_center_y": 0.46,
                },
                "vad": {
                    "speech_detected": False,
                },
            },
            runtime_status="processing",
            presence_session_id=13,
            identity_fusion=MultimodalIdentityFusionSnapshot(
                state="stable_main_user_multimodal",
                temporal_state="stable_multimodal_match",
                track_consistency_state="stable_anchor",
                track_anchor_zone="left",
                presence_session_id=13,
                claim=RuntimeClaimMetadata(
                    confidence=0.91,
                    source="voice_profile_plus_temporal_portrait_match_plus_track_history_plus_presence_session_memory",
                    requires_confirmation=True,
                ),
            ),
        )

        self.assertEqual(snapshot.state, "session_focus_locked")
        self.assertTrue(snapshot.session_focus_active)
        self.assertEqual(snapshot.target_horizontal, "left")
        self.assertEqual(snapshot.focus_source, "identity_fusion_track")

    def test_showing_intent_does_not_override_last_moved_person_in_multi_person_scene(self) -> None:
        tracker = MultimodalAttentionTargetTracker.from_config(TwinrConfig())

        tracker.observe(
            observed_at=10.0,
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
                            "confidence": 0.88,
                        },
                    ],
                    "primary_person_center_x": 0.2,
                    "primary_person_center_y": 0.5,
                },
                "vad": {"speech_detected": False},
            },
            runtime_status="waiting",
            presence_session_id=21,
        )

        snapshot = tracker.observe(
            observed_at=10.3,
            live_facts={
                "camera": {
                    "person_visible": True,
                    "person_count": 2,
                    "showing_intent_likely": True,
                    "visible_persons": [
                        {
                            "box": {"top": 0.1, "left": 0.1, "bottom": 0.9, "right": 0.4},
                            "zone": "left",
                            "confidence": 0.91,
                        },
                        {
                            "box": {"top": 0.1, "left": 0.62, "bottom": 0.9, "right": 0.9},
                            "zone": "right",
                            "confidence": 0.88,
                        },
                    ],
                    "primary_person_center_x": 0.25,
                    "primary_person_center_y": 0.5,
                },
                "vad": {"speech_detected": False},
            },
            runtime_status="waiting",
            presence_session_id=21,
        )

        self.assertEqual(snapshot.state, "last_moved_visible_person")
        self.assertEqual(snapshot.focus_source, "last_motion_track")
        self.assertEqual(snapshot.target_horizontal, "left")
        self.assertTrue(snapshot.showing_intent_active)


if __name__ == "__main__":
    unittest.main()
