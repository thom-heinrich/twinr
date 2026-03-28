from pathlib import Path
from threading import Event, Thread
import sys
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.workflows.playback_coordinator import PlaybackCoordinator, PlaybackPriority


class _BlockingTonePlayer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.tone_started = Event()
        self.release = Event()
        self.stop_calls = 0

    def play_tone(self, *, frequency_hz: int, duration_ms: int, volume: float, sample_rate: int) -> None:
        del duration_ms, volume, sample_rate
        self.calls.append(("tone", frequency_hz))
        if frequency_hz == 440:
            self.tone_started.set()
            self.release.wait(timeout=2.0)

    def stop_playback(self) -> None:
        self.stop_calls += 1
        self.release.set()


class _ChunkPlayer:
    def __init__(self) -> None:
        self.played: list[bytes] = []
        self.stop_calls = 0

    def play_wav_chunks(self, chunks, *, should_stop=None) -> None:
        payload = bytearray()
        for chunk in chunks:
            payload.extend(chunk)
            if callable(should_stop) and should_stop():
                break
        self.played.append(bytes(payload))

    def stop_playback(self) -> None:
        self.stop_calls += 1


class _PCMChunkPlayer:
    def __init__(self) -> None:
        self.played: list[bytes] = []
        self.sample_rates: list[int] = []
        self.channels: list[int] = []
        self.stop_calls = 0

    def play_pcm16_chunks(self, chunks, *, sample_rate: int, channels: int = 1, should_stop=None) -> None:
        payload = bytearray()
        for chunk in chunks:
            payload.extend(chunk)
            if callable(should_stop) and should_stop():
                break
        self.played.append(bytes(payload))
        self.sample_rates.append(sample_rate)
        self.channels.append(channels)

    def stop_playback(self) -> None:
        self.stop_calls += 1


class _InterruptingPCMPlayer:
    def __init__(self) -> None:
        self.played: list[bytes] = []
        self.feedback_started = Event()
        self.stop_calls = 0
        self._stop_current = Event()

    def play_pcm16_chunks(self, chunks, *, sample_rate: int, channels: int = 1, should_stop=None) -> None:
        del sample_rate, channels
        payload = bytearray()
        first_chunk: bytes | None = None
        stop_current = Event()
        self._stop_current = stop_current
        for chunk in chunks:
            chunk_bytes = bytes(chunk)
            if first_chunk is None:
                first_chunk = chunk_bytes
                if first_chunk.startswith(b"FEED"):
                    self.feedback_started.set()
            payload.extend(chunk_bytes)
            if first_chunk is not None and first_chunk.startswith(b"FEED"):
                while not stop_current.is_set():
                    if callable(should_stop) and should_stop():
                        break
                    time.sleep(0.005)
                raise RuntimeError(
                    "Audio playback failed: aplay: pcm_write:2127: write error: Interrupted system call"
                )
        self.played.append(bytes(payload))

    def stop_playback(self) -> None:
        self.stop_calls += 1
        self._stop_current.set()


class _CoordinatorRequestDouble:
    def __init__(self, *, owner: str, priority: int, stop) -> None:
        self.owner = owner
        self.priority = priority
        self.atomic = False
        self.done = Event()
        self.cancel_reason: str | None = None
        self.cancel_event = Event()
        self.stop = stop


class PlaybackCoordinatorTests(unittest.TestCase):
    def test_higher_priority_request_preempts_active_feedback(self) -> None:
        player = _BlockingTonePlayer()
        coordinator = PlaybackCoordinator(player)
        results: list[object] = []

        def run_feedback() -> None:
            results.append(
                coordinator.play_tone(
                    owner="feedback",
                    priority=PlaybackPriority.FEEDBACK,
                    frequency_hz=440,
                    duration_ms=500,
                    volume=0.1,
                    sample_rate=24000,
                )
            )

        thread = Thread(target=run_feedback, daemon=True)
        thread.start()
        self.assertTrue(player.tone_started.wait(timeout=1.0))

        beep_result = coordinator.play_tone(
            owner="beep",
            priority=PlaybackPriority.BEEP,
            frequency_hz=880,
            duration_ms=50,
            volume=0.1,
            sample_rate=24000,
        )
        thread.join(timeout=1.0)
        coordinator.close(timeout_s=1.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual([call[1] for call in player.calls], [440, 880])
        self.assertGreaterEqual(player.stop_calls, 1)
        self.assertFalse(beep_result.preempted)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].preempted)

    def test_streamed_wav_playback_runs_through_single_queue(self) -> None:
        player = _ChunkPlayer()
        coordinator = PlaybackCoordinator(player)

        result = coordinator.play_wav_chunks(
            owner="speech",
            priority=PlaybackPriority.SPEECH,
            chunks=iter((b"A", b"B", b"C")),
        )
        coordinator.close(timeout_s=1.0)

        self.assertEqual(player.played, [b"ABC"])
        self.assertFalse(result.preempted)
        self.assertGreaterEqual(result.runtime_ms, 0.0)

    def test_streamed_pcm16_playback_runs_through_single_queue(self) -> None:
        player = _PCMChunkPlayer()
        coordinator = PlaybackCoordinator(player)

        result = coordinator.play_pcm16_chunks(
            owner="feedback",
            priority=PlaybackPriority.FEEDBACK,
            chunks=iter((b"\x01\x00", b"\x02\x00", b"\x03\x00")),
            sample_rate=24000,
            channels=1,
        )
        coordinator.close(timeout_s=1.0)

        self.assertEqual(player.played, [b"\x01\x00\x02\x00\x03\x00"])
        self.assertEqual(player.sample_rates, [24000])
        self.assertEqual(player.channels, [1])
        self.assertFalse(result.preempted)
        self.assertGreaterEqual(result.runtime_ms, 0.0)

    def test_preempted_pcm_stop_error_returns_preempted_result(self) -> None:
        player = _InterruptingPCMPlayer()
        coordinator = PlaybackCoordinator(player)
        results: list[object] = []

        def run_feedback() -> None:
            results.append(
                coordinator.play_pcm16_chunks(
                    owner="feedback",
                    priority=PlaybackPriority.FEEDBACK,
                    chunks=iter((b"FEED-1", b"FEED-2")),
                    sample_rate=24000,
                    channels=1,
                )
            )

        thread = Thread(target=run_feedback, daemon=True)
        thread.start()
        self.assertTrue(player.feedback_started.wait(timeout=1.0))

        speech_result = coordinator.play_pcm16_chunks(
            owner="speech",
            priority=PlaybackPriority.SPEECH,
            chunks=iter((b"SPEAK-1",)),
            sample_rate=24000,
            channels=1,
        )
        thread.join(timeout=1.0)
        coordinator.close(timeout_s=1.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(player.stop_calls, 1)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].preempted)
        self.assertFalse(speech_result.preempted)
        self.assertIn(b"SPEAK-1", player.played)

    def test_bound_preemption_callback_does_not_stop_replacement_request(self) -> None:
        player = _ChunkPlayer()
        coordinator = PlaybackCoordinator(player)
        current = coordinator._current = _CoordinatorRequestDouble(
            owner="feedback",
            priority=int(PlaybackPriority.FEEDBACK),
            stop=player.stop_playback,
        )
        current.cancel_event.set()

        with coordinator._condition:
            stop_callback = coordinator._build_bound_stop_callback_locked(current)
            replacement = _CoordinatorRequestDouble(
                owner="speech",
                priority=int(PlaybackPriority.SPEECH),
                stop=player.stop_playback,
            )
            coordinator._current = replacement

        stop_callback()
        stop_calls_after_callback = player.stop_calls
        coordinator.close(timeout_s=1.0)

        self.assertEqual(stop_calls_after_callback, 0)

    def test_bound_preemption_callback_stops_original_request_when_still_current(self) -> None:
        player = _ChunkPlayer()
        coordinator = PlaybackCoordinator(player)
        current = coordinator._current = _CoordinatorRequestDouble(
            owner="feedback",
            priority=int(PlaybackPriority.FEEDBACK),
            stop=player.stop_playback,
        )
        current.cancel_event.set()

        with coordinator._condition:
            stop_callback = coordinator._build_bound_stop_callback_locked(current)

        stop_callback()
        stop_calls_after_callback = player.stop_calls
        coordinator.close(timeout_s=1.0)

        self.assertEqual(stop_calls_after_callback, 1)

    def test_stop_owner_only_stops_matching_active_request(self) -> None:
        player = _ChunkPlayer()
        coordinator = PlaybackCoordinator(player)
        coordinator._current = _CoordinatorRequestDouble(
            owner="feedback",
            priority=int(PlaybackPriority.FEEDBACK),
            stop=player.stop_playback,
        )

        coordinator.stop_owner("speech")
        stop_calls_after_miss = player.stop_calls
        coordinator.stop_owner("feedback")
        stop_calls_after_hit = player.stop_calls
        coordinator.close(timeout_s=1.0)

        self.assertEqual(stop_calls_after_miss, 0)
        self.assertEqual(stop_calls_after_hit, 1)


if __name__ == "__main__":
    unittest.main()
