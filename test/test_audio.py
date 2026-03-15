from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.audio import (
    resolve_dynamic_pause_thresholds,
    resolve_pause_resume_confirmation,
)


class DynamicPauseThresholdTests(unittest.TestCase):
    def test_single_resume_spike_does_not_confirm_resume(self) -> None:
        confirmed, consecutive_chunks = resolve_pause_resume_confirmation(
            consecutive_resume_chunks=0,
            required_resume_chunks=2,
        )

        self.assertFalse(confirmed)
        self.assertEqual(consecutive_chunks, 1)

    def test_second_resume_chunk_confirms_resume(self) -> None:
        confirmed, consecutive_chunks = resolve_pause_resume_confirmation(
            consecutive_resume_chunks=1,
            required_resume_chunks=2,
        )

        self.assertTrue(confirmed)
        self.assertEqual(consecutive_chunks, 0)

    def test_short_utterance_gets_extra_tail(self) -> None:
        pause_ms, pause_grace_ms = resolve_dynamic_pause_thresholds(
            base_pause_ms=1200,
            base_pause_grace_ms=450,
            speech_window_ms=800,
            enabled=True,
            short_utterance_max_ms=1000,
            long_utterance_min_ms=5000,
            short_pause_bonus_ms=120,
            short_pause_grace_bonus_ms=0,
            medium_pause_penalty_ms=120,
            medium_pause_grace_penalty_ms=250,
            long_pause_penalty_ms=320,
            long_pause_grace_penalty_ms=220,
        )

        self.assertEqual((pause_ms, pause_grace_ms), (1320, 450))

    def test_medium_utterance_keeps_baseline(self) -> None:
        pause_ms, pause_grace_ms = resolve_dynamic_pause_thresholds(
            base_pause_ms=1200,
            base_pause_grace_ms=450,
            speech_window_ms=2400,
            enabled=True,
            short_utterance_max_ms=1000,
            long_utterance_min_ms=5000,
            short_pause_bonus_ms=120,
            short_pause_grace_bonus_ms=0,
            medium_pause_penalty_ms=120,
            medium_pause_grace_penalty_ms=250,
            long_pause_penalty_ms=320,
            long_pause_grace_penalty_ms=220,
        )

        self.assertEqual((pause_ms, pause_grace_ms), (1080, 200))

    def test_long_utterance_ends_sooner(self) -> None:
        pause_ms, pause_grace_ms = resolve_dynamic_pause_thresholds(
            base_pause_ms=1200,
            base_pause_grace_ms=450,
            speech_window_ms=6400,
            enabled=True,
            short_utterance_max_ms=1000,
            long_utterance_min_ms=5000,
            short_pause_bonus_ms=120,
            short_pause_grace_bonus_ms=0,
            medium_pause_penalty_ms=120,
            medium_pause_grace_penalty_ms=250,
            long_pause_penalty_ms=320,
            long_pause_grace_penalty_ms=220,
        )

        self.assertEqual((pause_ms, pause_grace_ms), (880, 230))


if __name__ == "__main__":
    unittest.main()
