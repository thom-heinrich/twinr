from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.proactive.runtime.audio_policy import ReSpeakerAudioPolicyTracker
from twinr.proactive.social import SocialAudioObservation


class ReSpeakerAudioPolicyTests(unittest.TestCase):
    def test_presence_audio_active_is_false_for_background_media(self) -> None:
        tracker = ReSpeakerAudioPolicyTracker()

        snapshot = tracker.observe(
            now=10.0,
            audio=SocialAudioObservation(
                speech_detected=True,
                background_media_likely=True,
                non_speech_audio_likely=True,
                room_quiet=False,
                assistant_output_active=False,
                device_runtime_mode="audio_ready",
                host_control_ready=True,
            ),
        )

        self.assertFalse(snapshot.presence_audio_active)
        self.assertEqual(snapshot.speech_delivery_defer_reason, "background_media_active")


if __name__ == "__main__":
    unittest.main()
