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

    def test_bound_preemption_callback_does_not_stop_replacement_request(self) -> None:
        player = _ChunkPlayer()
        coordinator = PlaybackCoordinator(player)
        current = coordinator._current = type("_Req", (), {})()
        current.owner = "feedback"
        current.priority = int(PlaybackPriority.FEEDBACK)
        current.atomic = False
        current.cancel_event = Event()
        current.stop = player.stop_playback
        current.cancel_event.set()

        with coordinator._condition:
            stop_callback = coordinator._build_bound_stop_callback_locked(current)
            replacement = type("_Req", (), {})()
            replacement.owner = "speech"
            replacement.priority = int(PlaybackPriority.SPEECH)
            replacement.atomic = False
            replacement.cancel_event = Event()
            replacement.stop = player.stop_playback
            coordinator._current = replacement

        stop_callback()
        stop_calls_after_callback = player.stop_calls
        coordinator.close(timeout_s=1.0)

        self.assertEqual(stop_calls_after_callback, 0)

    def test_bound_preemption_callback_stops_original_request_when_still_current(self) -> None:
        player = _ChunkPlayer()
        coordinator = PlaybackCoordinator(player)
        current = coordinator._current = type("_Req", (), {})()
        current.owner = "feedback"
        current.priority = int(PlaybackPriority.FEEDBACK)
        current.atomic = False
        current.cancel_event = Event()
        current.stop = player.stop_playback
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
        current = coordinator._current = type("_Req", (), {})()
        current.owner = "feedback"
        current.priority = int(PlaybackPriority.FEEDBACK)
        current.atomic = False
        current.cancel_event = Event()
        current.stop = player.stop_playback

        coordinator.stop_owner("speech")
        stop_calls_after_miss = player.stop_calls
        coordinator.stop_owner("feedback")
        stop_calls_after_hit = player.stop_calls
        coordinator.close(timeout_s=1.0)

        self.assertEqual(stop_calls_after_miss, 0)
        self.assertEqual(stop_calls_after_hit, 1)


if __name__ == "__main__":
    unittest.main()
