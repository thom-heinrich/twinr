from array import array
import io
import math
from pathlib import Path
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.voice_profile import VoiceProfileMonitor, VoiceProfileStore


def _voice_sample_wav_bytes(*, kind: str = "voice", frequency_hz: float = 175.0, amplitude: float = 0.35, duration_s: float = 1.8) -> bytes:
    sample_rate = 16000
    total_frames = int(sample_rate * duration_s)
    frames = array("h")
    for index in range(total_frames):
        t = index / sample_rate
        envelope = min(1.0, index / (sample_rate * 0.2), (total_frames - index) / (sample_rate * 0.2))
        if kind == "voice":
            sample = amplitude * envelope * (
                (0.70 * math.sin(2.0 * math.pi * frequency_hz * t))
                + (0.20 * math.sin(2.0 * math.pi * frequency_hz * 2.0 * t))
                + (0.10 * math.sin(2.0 * math.pi * (frequency_hz + 35.0) * t))
            )
        elif kind == "noise":
            sample = amplitude * envelope * (math.sin(2.0 * math.pi * 1400.0 * t) * math.sin(2.0 * math.pi * 37.0 * t))
        else:
            raise ValueError(f"Unsupported sample kind: {kind}")
        frames.append(max(-32767, min(32767, int(sample * 32767))))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames.tobytes())
    return buffer.getvalue()


class VoiceProfileTests(unittest.TestCase):
    def make_monitor(self, temp_dir: str) -> VoiceProfileMonitor:
        return VoiceProfileMonitor(
            store=VoiceProfileStore(Path(temp_dir) / "voice_profile.json"),
            likely_threshold=0.72,
            uncertain_threshold=0.55,
            max_enrollment_samples=6,
            min_sample_ms=1200,
        )

    def test_enroll_and_assess_same_sample_likely_user(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = self.make_monitor(temp_dir)
            sample = _voice_sample_wav_bytes()

            template = monitor.enroll_wav_bytes(sample)
            assessment = monitor.assess_wav_bytes(sample)

        self.assertEqual(template.sample_count, 1)
        self.assertEqual(assessment.status, "likely_user")
        self.assertIsNotNone(assessment.confidence)
        self.assertGreaterEqual(assessment.confidence or 0.0, 0.99)

    def test_assess_noise_reports_unknown_voice(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = self.make_monitor(temp_dir)
            monitor.enroll_wav_bytes(_voice_sample_wav_bytes(frequency_hz=175.0))

            assessment = monitor.assess_wav_bytes(_voice_sample_wav_bytes(kind="noise", amplitude=0.5))

        self.assertEqual(assessment.status, "unknown_voice")
        self.assertIsNotNone(assessment.confidence)
        self.assertLess(assessment.confidence or 1.0, 0.55)

    def test_store_contains_only_template_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = self.make_monitor(temp_dir)
            monitor.enroll_wav_bytes(_voice_sample_wav_bytes())
            store_path = Path(temp_dir) / "voice_profile.json"
            payload = store_path.read_text(encoding="utf-8")

        self.assertIn('"embedding"', payload)
        self.assertIn('"sample_count"', payload)
        self.assertNotIn("RIFF", payload)
        self.assertNotIn("WAVE", payload)
        self.assertNotIn("data", payload)

    def test_reset_clears_local_template(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = self.make_monitor(temp_dir)
            sample = _voice_sample_wav_bytes()
            monitor.enroll_wav_bytes(sample)

            summary = monitor.reset()

        self.assertFalse(summary.enrolled)
        self.assertEqual(summary.sample_count, 0)


if __name__ == "__main__":
    unittest.main()
