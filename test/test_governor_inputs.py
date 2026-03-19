from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.proactive.runtime.audio_policy import ReSpeakerAudioPolicySnapshot
from twinr.proactive.runtime.governor_inputs import build_respeaker_governor_inputs
from twinr.proactive.runtime.multimodal_initiative import ReSpeakerMultimodalInitiativeSnapshot
from twinr.proactive.runtime.presence import PresenceSessionSnapshot


class GovernorInputTests(unittest.TestCase):
    def test_build_respeaker_governor_inputs_packages_presence_and_audio_policy_state(self) -> None:
        inputs = build_respeaker_governor_inputs(
            requested_channel="display",
            presence_snapshot=PresenceSessionSnapshot(
                armed=True,
                reason="recent_person_visible",
                person_visible=False,
                session_id=7,
            ),
            audio_policy_snapshot=ReSpeakerAudioPolicySnapshot(
                observed_at=10.0,
                room_busy_or_overlapping=True,
                quiet_window_open=False,
                resume_window_open=True,
                mute_blocks_voice_capture=False,
                initiative_block_reason="room_busy_or_overlapping",
                speech_delivery_defer_reason="background_media_active",
                runtime_alert_code="ready",
            ),
            multimodal_initiative_snapshot=ReSpeakerMultimodalInitiativeSnapshot(
                observed_at=10.0,
                ready=False,
                confidence=0.61,
                block_reason="low_confidence_speaker_association",
                recommended_channel="display",
            ),
        )

        self.assertEqual(inputs.channel, "display")
        self.assertEqual(inputs.presence_session_id, 7)
        self.assertEqual(inputs.runtime_alert_code, "ready")
        self.assertEqual(inputs.initiative_block_reason, "room_busy_or_overlapping")
        self.assertEqual(inputs.event_data()["audio_resume_window_open"], True)
        self.assertEqual(inputs.event_data()["multimodal_initiative_ready"], False)
        self.assertEqual(
            inputs.event_data()["multimodal_initiative_block_reason"],
            "low_confidence_speaker_association",
        )
