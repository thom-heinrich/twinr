from pathlib import Path
from tempfile import TemporaryFile
from unittest import mock
import os
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.audio import (
    WaveAudioPlayer,
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


class _FakePlaybackProcess:
    def __init__(self) -> None:
        self.stdin = TemporaryFile()
        self.stderr = TemporaryFile()
        self.stdout = None
        self.returncode = None
        self.terminate_calls = 0
        self.wait_calls = 0
        self.wait_timeouts: list[float | None] = []

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.wait_calls += 1
        self.wait_timeouts.append(timeout)
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


class WaveAudioPlayerTests(unittest.TestCase):
    def test_preempted_stream_terminates_aplay_without_waiting_for_drain(self) -> None:
        player = WaveAudioPlayer(device="default")
        process = _FakePlaybackProcess()
        stop_requested = [False]
        original_os_write = os.write

        def fake_write(fd, view):
            written = original_os_write(fd, view)
            stop_requested[0] = True
            return written

        with (
            mock.patch("twinr.hardware.audio._spawn_audio_process", return_value=process),
            mock.patch("twinr.hardware.audio._wait_for_writable", return_value=True),
            mock.patch("twinr.hardware.audio.os.set_blocking"),
            mock.patch("twinr.hardware.audio.os.write", side_effect=fake_write),
        ):
            player.play_wav_chunks(
                [b"chunk-1", b"chunk-2"],
                should_stop=lambda: stop_requested[0],
            )

        self.assertGreaterEqual(process.terminate_calls, 1)
        self.assertEqual(process.wait_timeouts, [1.0])

    def test_stop_playback_terminates_active_process(self) -> None:
        player = WaveAudioPlayer(device="default")
        process = _FakePlaybackProcess()

        player._set_active_process(process)
        player.stop_playback()

        self.assertGreaterEqual(process.terminate_calls, 1)


if __name__ == "__main__":
    unittest.main()
