import audioop
import io
from pathlib import Path
import sys
import unittest
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.workflows.working_feedback_tone import (
    PROCESSING_SWELL_TONE_SPEC,
    build_swelling_feedback_tone_pcm16,
    build_swelling_feedback_tone_wav_bytes,
)


class WorkingFeedbackToneTests(unittest.TestCase):
    def test_processing_swell_tone_pcm_swells_toward_the_middle(self) -> None:
        sample_rate = 24000
        pcm_bytes = build_swelling_feedback_tone_pcm16(
            PROCESSING_SWELL_TONE_SPEC,
            sample_rate=sample_rate,
            peak_gain=0.15,
        )

        segment_bytes = (sample_rate // 5) * 2
        first_rms = audioop.rms(pcm_bytes[:segment_bytes], 2)
        middle_start = max(0, (len(pcm_bytes) // 2) - (segment_bytes // 2))
        middle_rms = audioop.rms(pcm_bytes[middle_start : middle_start + segment_bytes], 2)
        last_rms = audioop.rms(pcm_bytes[-segment_bytes:], 2)

        self.assertLess(first_rms, middle_rms)
        self.assertLess(last_rms, middle_rms)

    def test_processing_swell_tone_lasts_long_enough_to_feel_like_a_slow_hum(self) -> None:
        sample_rate = 24000
        pcm_bytes = build_swelling_feedback_tone_pcm16(
            PROCESSING_SWELL_TONE_SPEC,
            sample_rate=sample_rate,
            peak_gain=0.15,
        )

        self.assertGreaterEqual(len(pcm_bytes) // 2, sample_rate * 4)

    def test_processing_swell_tone_stays_in_low_humming_frequency_range(self) -> None:
        sample_rate = 24000
        pcm_bytes = build_swelling_feedback_tone_pcm16(
            PROCESSING_SWELL_TONE_SPEC,
            sample_rate=sample_rate,
            peak_gain=0.15,
        )

        analysis_window_frames = sample_rate // 3
        center_frame = (len(pcm_bytes) // 2) // 2
        start_frame = max(0, center_frame - (analysis_window_frames // 2))
        center_window = pcm_bytes[start_frame * 2 : (start_frame + analysis_window_frames) * 2]
        estimated_frequency_hz = audioop.cross(center_window, 2) / (
            2.0 * (analysis_window_frames / sample_rate)
        )

        self.assertGreater(estimated_frequency_hz, 110.0)
        self.assertLess(estimated_frequency_hz, 210.0)

    def test_processing_swell_tone_wav_bytes_wrap_pcm(self) -> None:
        sample_rate = 24000
        wav_bytes = build_swelling_feedback_tone_wav_bytes(
            PROCESSING_SWELL_TONE_SPEC,
            sample_rate=sample_rate,
            peak_gain=0.15,
        )

        with wave.open(io.BytesIO(wav_bytes), "rb") as reader:
            self.assertEqual(reader.getnchannels(), 1)
            self.assertEqual(reader.getframerate(), sample_rate)
            self.assertGreaterEqual(reader.getnframes(), sample_rate * 4)


if __name__ == "__main__":
    unittest.main()
