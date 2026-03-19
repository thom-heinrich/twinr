from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.proactive.runtime.multimodal_initiative import derive_respeaker_multimodal_initiative
from twinr.proactive.runtime.speaker_association import derive_respeaker_speaker_association


class ReSpeakerSpeakerAssociationTests(unittest.TestCase):
    def test_associates_single_primary_visible_person_when_audio_and_camera_are_strong(self) -> None:
        facts = {
            "camera": {
                "person_visible": True,
                "person_count": 1,
                "person_count_unknown": False,
                "primary_person_zone": "center",
                "looking_toward_device": True,
            },
            "respeaker": {
                "azimuth_deg": 277,
                "direction_confidence": 0.91,
            },
            "audio_policy": {
                "speaker_direction_stable": True,
            },
        }

        snapshot = derive_respeaker_speaker_association(
            observed_at=12.0,
            live_facts=facts,
        )

        self.assertEqual(snapshot.state, "primary_visible_person_associated")
        self.assertTrue(snapshot.associated)
        self.assertEqual(snapshot.target_id, "primary_visible_person")
        self.assertGreater(snapshot.confidence or 0.0, 0.8)

    def test_multi_person_context_fails_closed(self) -> None:
        snapshot = derive_respeaker_speaker_association(
            observed_at=12.0,
            live_facts={
                "camera": {
                    "person_visible": True,
                    "person_count": 2,
                    "person_count_unknown": False,
                    "primary_person_zone": "left",
                },
                "respeaker": {
                    "azimuth_deg": 90,
                    "direction_confidence": 0.93,
                },
                "audio_policy": {
                    "speaker_direction_stable": True,
                },
            },
        )

        self.assertEqual(snapshot.state, "multi_person_context")
        self.assertFalse(snapshot.associated)

    def test_low_direction_confidence_fails_closed(self) -> None:
        snapshot = derive_respeaker_speaker_association(
            observed_at=12.0,
            live_facts={
                "camera": {
                    "person_visible": True,
                    "person_count": 1,
                    "person_count_unknown": False,
                    "primary_person_zone": "center",
                },
                "respeaker": {
                    "azimuth_deg": 270,
                    "direction_confidence": 0.41,
                },
                "audio_policy": {
                    "speaker_direction_stable": False,
                },
            },
        )

        self.assertEqual(snapshot.state, "audio_direction_unstable")
        self.assertFalse(snapshot.associated)


class ReSpeakerMultimodalInitiativeTests(unittest.TestCase):
    def test_multimodal_initiative_becomes_ready_for_clear_single_person_context(self) -> None:
        facts = {
            "camera": {
                "person_visible": True,
                "person_count": 1,
                "person_count_unknown": False,
                "primary_person_zone": "center",
                "looking_toward_device": True,
                "visual_attention_score": 0.88,
            },
            "respeaker": {
                "azimuth_deg": 277,
                "direction_confidence": 0.9,
            },
            "audio_policy": {
                "speaker_direction_stable": True,
                "presence_audio_active": True,
                "quiet_window_open": False,
            },
        }

        snapshot = derive_respeaker_multimodal_initiative(
            observed_at=12.0,
            live_facts=facts,
        )

        self.assertTrue(snapshot.ready)
        self.assertEqual(snapshot.recommended_channel, "speech")
        self.assertGreater(snapshot.confidence or 0.0, 0.8)

    def test_multimodal_initiative_defers_to_display_when_association_is_weak(self) -> None:
        facts = {
            "camera": {
                "person_visible": True,
                "person_count": 1,
                "person_count_unknown": False,
                "primary_person_zone": "center",
            },
            "respeaker": {
                "azimuth_deg": 270,
                "direction_confidence": 0.42,
            },
            "audio_policy": {
                "speaker_direction_stable": False,
                "presence_audio_active": True,
            },
        }

        snapshot = derive_respeaker_multimodal_initiative(
            observed_at=12.0,
            live_facts=facts,
        )

        self.assertFalse(snapshot.ready)
        self.assertEqual(snapshot.recommended_channel, "display")
        self.assertEqual(snapshot.block_reason, "low_confidence_speaker_association")

    def test_multimodal_initiative_defers_to_display_when_background_media_is_active(self) -> None:
        facts = {
            "camera": {
                "person_visible": True,
                "person_count": 1,
                "person_count_unknown": False,
                "primary_person_zone": "center",
                "looking_toward_device": True,
            },
            "respeaker": {
                "azimuth_deg": 277,
                "direction_confidence": 0.93,
            },
            "audio_policy": {
                "speaker_direction_stable": True,
                "presence_audio_active": True,
                "speech_delivery_defer_reason": "background_media_active",
            },
        }

        snapshot = derive_respeaker_multimodal_initiative(
            observed_at=12.0,
            live_facts=facts,
        )

        self.assertFalse(snapshot.ready)
        self.assertEqual(snapshot.recommended_channel, "display")
        self.assertEqual(snapshot.block_reason, "background_media_active")


if __name__ == "__main__":
    unittest.main()
