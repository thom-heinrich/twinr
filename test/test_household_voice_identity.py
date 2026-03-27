from array import array
import io
import math
from pathlib import Path
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.household_voice_identity import (
    HouseholdVoiceIdentityMonitor,
    HouseholdVoiceIdentityStore,
    assess_household_voice_pcm16,
    household_voice_profiles_revision,
)


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


class HouseholdVoiceIdentityTests(unittest.TestCase):
    def make_monitor(self, temp_dir: str) -> HouseholdVoiceIdentityMonitor:
        return HouseholdVoiceIdentityMonitor(
            store=HouseholdVoiceIdentityStore(Path(temp_dir) / "household_voice_identities.json"),
            primary_user_id="main_user",
            likely_threshold=0.72,
            uncertain_threshold=0.55,
            identity_margin=0.06,
            min_sample_ms=1200,
            max_enrollment_samples=6,
        )

    def test_assess_distinguishes_other_enrolled_user(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = self.make_monitor(temp_dir)
            main_voice = _voice_sample_wav_bytes(frequency_hz=175.0)
            guest_voice = _voice_sample_wav_bytes(frequency_hz=240.0)

            monitor.enroll_wav_bytes(main_voice, user_id="main_user", display_name="Theo")
            monitor.enroll_wav_bytes(guest_voice, user_id="guest_user", display_name="Guest")
            assessment = monitor.assess_wav_bytes(guest_voice)

        self.assertEqual(assessment.status, "known_other_user")
        self.assertEqual(assessment.matched_user_id, "guest_user")
        self.assertEqual(assessment.matched_user_display_name, "Guest")
        self.assertGreater(assessment.confidence or 0.0, 0.72)

    def test_reset_clears_one_household_voice_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = self.make_monitor(temp_dir)
            main_voice = _voice_sample_wav_bytes(frequency_hz=175.0)

            monitor.enroll_wav_bytes(main_voice, user_id="main_user", display_name="Theo")
            summary = monitor.reset(user_id="main_user")

        self.assertFalse(summary.enrolled)
        self.assertEqual(summary.sample_count, 0)

    def test_assess_household_voice_pcm16_matches_exported_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = self.make_monitor(temp_dir)
            guest_voice = _voice_sample_wav_bytes(frequency_hz=240.0)

            monitor.enroll_wav_bytes(guest_voice, user_id="guest_user", display_name="Guest")
            with wave.open(io.BytesIO(guest_voice), "rb") as wav_file:
                guest_pcm = wav_file.readframes(wav_file.getnframes())
            assessment = assess_household_voice_pcm16(
                guest_pcm,
                sample_rate=16000,
                channels=1,
                checked_at="2026-03-27T11:00:00+00:00",
                profiles=monitor.voice_profiles(),
                primary_user_id="main_user",
                likely_threshold=0.72,
                uncertain_threshold=0.55,
                identity_margin=0.06,
                min_sample_ms=1200,
            )

        self.assertEqual(assessment.status, "known_other_user")
        self.assertEqual(assessment.matched_user_id, "guest_user")

    def test_household_voice_profiles_revision_changes_when_profiles_change(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = self.make_monitor(temp_dir)
            main_voice = _voice_sample_wav_bytes(frequency_hz=175.0)
            revision_before = household_voice_profiles_revision(monitor.voice_profiles())

            monitor.enroll_wav_bytes(main_voice, user_id="main_user", display_name="Theo")
            revision_after = household_voice_profiles_revision(monitor.voice_profiles())

        self.assertNotEqual(revision_before, revision_after)


if __name__ == "__main__":
    unittest.main()
