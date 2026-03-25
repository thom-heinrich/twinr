from array import array
import io
import math
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.hardware.household_identity import HouseholdIdentityFeedbackStore, HouseholdIdentityManager
from twinr.hardware.household_voice_identity import HouseholdVoiceIdentityMonitor, HouseholdVoiceIdentityStore
from twinr.hardware.portrait_identity import PortraitIdentityProfile, PortraitReferenceImage
from twinr.hardware.portrait_match import PortraitMatchObservation


def _voice_sample_wav_bytes(*, frequency_hz: float = 175.0, amplitude: float = 0.35, duration_s: float = 1.8) -> bytes:
    sample_rate = 16000
    total_frames = int(sample_rate * duration_s)
    frames = array("h")
    for index in range(total_frames):
        t = index / sample_rate
        envelope = min(1.0, index / (sample_rate * 0.2), (total_frames - index) / (sample_rate * 0.2))
        sample = amplitude * envelope * (
            (0.70 * math.sin(2.0 * math.pi * frequency_hz * t))
            + (0.20 * math.sin(2.0 * math.pi * frequency_hz * 2.0 * t))
            + (0.10 * math.sin(2.0 * math.pi * (frequency_hz + 35.0) * t))
        )
        frames.append(max(-32767, min(32767, int(sample * 32767))))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames.tobytes())
    return buffer.getvalue()


class _FakePortraitProvider:
    def __init__(self, observation: PortraitMatchObservation, *, user_id: str, display_name: str) -> None:
        self._observation = observation
        self._profile = PortraitIdentityProfile(
            user_id=user_id,
            display_name=display_name,
            primary_user=False,
            created_at="2026-03-19T12:00:00+00:00",
            updated_at="2026-03-19T12:00:00+00:00",
            reference_images=(
                PortraitReferenceImage(
                    reference_id="ref_1",
                    relative_path=f"{user_id}/ref_1.jpg",
                    image_sha256="abc123",
                    source="manual_import",
                    added_at="2026-03-19T12:00:00+00:00",
                    embedding=(1.0, 0.0, 0.0),
                    detector_confidence=0.95,
                ),
            ),
        )

    def list_profiles(self):
        return (self._profile,)

    def observe(self):
        return self._observation

    def capture_and_enroll_reference(self, **_kwargs):
        raise AssertionError("capture should not be called in this test")


class HouseholdIdentityManagerTests(unittest.TestCase):
    def test_observe_combines_face_voice_and_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir, personality_dir="personality")
            portrait_provider = _FakePortraitProvider(
                PortraitMatchObservation(
                    checked_at=12.0,
                    state="known_other_user",
                    matches_reference_user=False,
                    confidence=0.88,
                    fused_confidence=0.91,
                    temporal_state="stable_match",
                    temporal_observation_count=3,
                    matched_user_id="guest_user",
                    matched_user_display_name="Guest",
                    reference_image_count=1,
                    live_face_count=1,
                    backend_name="fake_portrait_backend",
                ),
                user_id="guest_user",
                display_name="Guest",
            )
            voice_monitor = HouseholdVoiceIdentityMonitor(
                store=HouseholdVoiceIdentityStore(Path(temp_dir) / "household_voice_identities.json"),
                primary_user_id="main_user",
                likely_threshold=0.72,
                uncertain_threshold=0.55,
                identity_margin=0.06,
                min_sample_ms=1200,
                max_enrollment_samples=6,
            )
            guest_voice = _voice_sample_wav_bytes(frequency_hz=240.0)
            voice_monitor.enroll_wav_bytes(guest_voice, user_id="guest_user", display_name="Guest")
            feedback_store = HouseholdIdentityFeedbackStore(Path(temp_dir) / "household_identity_feedback.json")
            manager = HouseholdIdentityManager(
                config=config,
                portrait_provider=portrait_provider,
                voice_monitor=voice_monitor,
                feedback_store=feedback_store,
            )

            observation = manager.observe(
                audio_pcm=wave.open(io.BytesIO(guest_voice), "rb").readframes(16000 * 2),  # type: ignore[arg-type]
                sample_rate=16000,
                channels=1,
            )
            event, member = manager.record_feedback(outcome="confirm", user_id="guest_user")

        self.assertEqual(observation.state, "multimodal_match")
        self.assertEqual(observation.matched_user_id, "guest_user")
        self.assertEqual(event.outcome, "confirm")
        self.assertIsNotNone(member)
        self.assertEqual(member.user_id, "guest_user")
        self.assertGreater(member.quality.score, 0.5)


if __name__ == "__main__":
    unittest.main()
