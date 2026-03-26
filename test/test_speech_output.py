from pathlib import Path
from threading import Event
from time import sleep
import time
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.tools import SpeechLaneDelta
from twinr.agent.workflows.playback_coordinator import PlaybackCoordinator
from twinr.agent.workflows import speech_output as speech_output_module
from twinr.agent.workflows.speech_output import InterruptibleSpeechOutput


class SlowTTSProvider:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.started = Event()

    def synthesize_stream(self, text: str, **kwargs):
        del kwargs
        self.calls.append(text)
        if "nach" in text:
            self.started.set()
            yield b"FILLER-1"
            sleep(0.2)
            yield b"FILLER-2"
            sleep(0.2)
            yield b"FILLER-3"
            return
        yield b"FINAL-1"
        yield b"FINAL-2"


class GatedFirstChunkTTSProvider:
    def __init__(self) -> None:
        self.started = Event()
        self.release = Event()

    def synthesize_stream(self, text: str, **kwargs):
        del text, kwargs
        self.started.set()
        self.release.wait(timeout=5.0)
        if not self.release.is_set():
            return
        yield b"READY-1"
        yield b"READY-2"


class ReentrantCloseTTSProvider:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.blocking_second_chunk = Event()
        self.release_second_chunk = Event()

    def synthesize_stream(self, text: str, **kwargs):
        del kwargs
        self.calls.append(text)
        if "nach" not in text:
            yield b"FINAL-1"
            yield b"FINAL-2"
            return
        yield b"FILLER-1"
        self.blocking_second_chunk.set()
        self.release_second_chunk.wait(timeout=5.0)
        if not self.release_second_chunk.is_set():
            return
        yield b"FILLER-2"


class CloseRaisesGeneratorExecutingIterator:
    def __init__(self) -> None:
        self.blocking_second_chunk = Event()
        self.release_second_chunk = Event()
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self) -> bytes:
        if self._index == 0:
            self._index = 1
            return b"FILLER-1"
        if self._index == 1:
            self.blocking_second_chunk.set()
            self.release_second_chunk.wait(timeout=5.0)
            self._index = 2
            if not self.release_second_chunk.is_set():
                raise StopIteration
            return b"FILLER-2"
        raise StopIteration

    def close(self) -> None:
        raise ValueError("generator already executing")


class CloseRaisesGeneratorExecutingTTSProvider:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.iterator = CloseRaisesGeneratorExecutingIterator()

    def synthesize_stream(self, text: str, **kwargs):
        del kwargs
        self.calls.append(text)
        if "nach" in text:
            return self.iterator
        return iter((b"FINAL-1", b"FINAL-2"))


class BlockingCloseIterator:
    def __init__(self) -> None:
        self.close_started = Event()
        self.release_close = Event()
        self._yielded = False

    def __iter__(self):
        return self

    def __next__(self) -> bytes:
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return b"FINAL-1"

    def close(self) -> None:
        self.close_started.set()
        self.release_close.wait(timeout=5.0)


class BlockingCloseTTSProvider:
    def __init__(self) -> None:
        self.iterator = BlockingCloseIterator()

    def synthesize_stream(self, text: str, **kwargs):
        del text, kwargs
        return self.iterator


class InterruptiblePlayer:
    def __init__(self) -> None:
        self.played: list[bytes] = []
        self.stopped = False

    def play_wav_chunks(self, chunks, *, should_stop=None) -> None:
        payload = bytearray()
        for chunk in chunks:
            payload.extend(chunk)
            if should_stop is not None and should_stop():
                self.stopped = True
                break
        self.played.append(bytes(payload))

    def stop_playback(self) -> None:
        self.stopped = True


class CoordinatorPreemptObservationPlayer:
    def __init__(self) -> None:
        self.feedback_started = Event()
        self.feedback_stopped = Event()
        self.speech_started = Event()
        self.played: list[bytes] = []

    def play_wav_chunks(self, chunks, *, should_stop=None) -> None:
        payload = bytearray()
        first_chunk: bytes | None = None
        for chunk in chunks:
            if first_chunk is None:
                first_chunk = bytes(chunk)
                if first_chunk.startswith(b"PROCESS"):
                    self.feedback_started.set()
                else:
                    self.speech_started.set()
            payload.extend(chunk)
            if should_stop is not None and should_stop():
                if first_chunk is not None and first_chunk.startswith(b"PROCESS"):
                    self.feedback_stopped.set()
                break
        self.played.append(bytes(payload))

    def stop_playback(self) -> None:
        return


class InterruptibleSpeechOutputTests(unittest.TestCase):
    def test_flush_enqueues_pending_non_atomic_segment(self) -> None:
        tts_provider = SlowTTSProvider()
        player = InterruptiblePlayer()

        output = InterruptibleSpeechOutput(
            tts_provider=tts_provider,
            player=player,
            chunk_size=512,
            segment_boundary=lambda text: None,
        )

        output.submit_text_delta("Hallo zusammen")
        output.flush()
        output.close(timeout_s=2.0)
        output.raise_if_error()

        self.assertEqual(tts_provider.calls, ["Hallo zusammen"])
        self.assertTrue(any(b"FINAL-1" in payload for payload in player.played))

    def test_preempted_filler_stops_and_final_audio_plays(self) -> None:
        tts_provider = SlowTTSProvider()
        player = InterruptiblePlayer()
        first_audio = []
        started = []
        preempted = []

        output = InterruptibleSpeechOutput(
            tts_provider=tts_provider,
            player=player,
            chunk_size=512,
            segment_boundary=lambda text: len(text) if text.strip() else None,
            on_speaking_started=lambda: started.append("started"),
            on_first_audio=lambda: first_audio.append("audio"),
            on_preempt=lambda: preempted.append("preempted"),
        )

        output.submit_lane_delta(
            SpeechLaneDelta(
                text="Ich schaue kurz nach.",
                lane="filler",
            )
        )
        self.assertTrue(output.wait_for_first_audio(timeout_s=1.0))
        output.submit_lane_delta(
            SpeechLaneDelta(
                text="Heute wird es trocken und kühl.",
                lane="final",
                replace_current=True,
                atomic=True,
            )
        )
        output.flush()
        output.close(timeout_s=2.0)
        output.raise_if_error()

        self.assertEqual(len(started), 1)
        self.assertEqual(len(first_audio), 1)
        self.assertEqual(preempted, ["preempted"])
        self.assertTrue(player.played[0].startswith(b"FILLER-1"))
        self.assertLess(len(player.played[0]), len(b"FILLER-1FILLER-2FILLER-3"))
        self.assertEqual(tts_provider.calls, ["Ich schaue kurz nach.", "Heute wird es trocken und kühl."])
        self.assertTrue(any(b"FINAL-1" in payload for payload in player.played))

    def test_preempted_filler_does_not_log_generator_reentrancy_warning(self) -> None:
        tts_provider = CloseRaisesGeneratorExecutingTTSProvider()
        player = InterruptiblePlayer()

        output = InterruptibleSpeechOutput(
            tts_provider=tts_provider,
            player=player,
            chunk_size=512,
            segment_boundary=lambda text: len(text) if text.strip() else None,
        )

        with mock.patch.object(speech_output_module._LOGGER, "warning") as warning:
            output.submit_lane_delta(
                SpeechLaneDelta(
                    text="Ich schaue kurz nach.",
                    lane="filler",
                )
            )
            self.assertTrue(output.wait_for_first_audio(timeout_s=1.0))
            self.assertTrue(tts_provider.iterator.blocking_second_chunk.wait(timeout=1.0))
            output.submit_lane_delta(
                SpeechLaneDelta(
                    text="Heute kommt die eigentliche Antwort.",
                    lane="final",
                    replace_current=True,
                    atomic=True,
                )
            )
            tts_provider.iterator.release_second_chunk.set()
            output.close(timeout_s=2.0)
            output.raise_if_error()

        self.assertFalse(
            any("failed to close the live TTS stream" in str(call.args[0]) for call in warning.call_args_list)
        )
        self.assertEqual(tts_provider.calls, ["Ich schaue kurz nach.", "Heute kommt die eigentliche Antwort."])
        self.assertTrue(any(b"FINAL-1" in payload for payload in player.played))

    def test_wait_for_first_audio_observes_first_chunk(self) -> None:
        tts_provider = SlowTTSProvider()
        player = InterruptiblePlayer()

        output = InterruptibleSpeechOutput(
            tts_provider=tts_provider,
            player=player,
            chunk_size=512,
            segment_boundary=lambda text: len(text) if text.strip() else None,
        )

        output.submit_lane_delta(
            SpeechLaneDelta(
                text="Ich schaue kurz nach.",
                lane="filler",
            )
        )

        self.assertTrue(output.wait_for_first_audio(timeout_s=1.0))
        output.close(timeout_s=2.0)
        output.raise_if_error()

    def test_wait_until_idle_observes_playback_completion(self) -> None:
        tts_provider = SlowTTSProvider()
        player = InterruptiblePlayer()

        output = InterruptibleSpeechOutput(
            tts_provider=tts_provider,
            player=player,
            chunk_size=512,
            segment_boundary=lambda text: len(text) if text.strip() else None,
        )

        output.submit_lane_delta(
            SpeechLaneDelta(
                text="Ich schaue kurz nach.",
                lane="filler",
            )
        )

        self.assertTrue(output.wait_for_first_audio(timeout_s=1.0))
        self.assertFalse(output.wait_until_idle(timeout_s=0.05))
        self.assertTrue(output.wait_until_idle(timeout_s=1.0))
        output.close(timeout_s=2.0)
        output.raise_if_error()

    def test_external_stop_interrupts_current_playback(self) -> None:
        tts_provider = SlowTTSProvider()
        player = InterruptiblePlayer()
        stop_event = Event()

        output = InterruptibleSpeechOutput(
            tts_provider=tts_provider,
            player=player,
            chunk_size=512,
            segment_boundary=lambda text: len(text) if text.strip() else None,
            should_stop=stop_event.is_set,
        )

        output.submit_lane_delta(
            SpeechLaneDelta(
                text="Ich schaue kurz nach.",
                lane="filler",
            )
        )

        self.assertTrue(output.wait_for_first_audio(timeout_s=1.0))
        stop_event.set()
        output.close(timeout_s=2.0)
        output.raise_if_error()

        self.assertTrue(player.played)
        self.assertTrue(len(player.played[0]) < len(b"FILLER-1FILLER-2FILLER-3"))

    def test_speaking_started_waits_for_real_first_audio(self) -> None:
        tts_provider = GatedFirstChunkTTSProvider()
        player = InterruptiblePlayer()
        started: list[str] = []

        output = InterruptibleSpeechOutput(
            tts_provider=tts_provider,
            player=player,
            chunk_size=512,
            segment_boundary=lambda text: len(text) if text.strip() else None,
            on_speaking_started=lambda: started.append("started"),
        )

        output.submit_text_delta("Hallo zusammen.")
        self.assertTrue(tts_provider.started.wait(timeout=1.0))
        sleep(0.05)
        self.assertEqual(started, [])
        self.assertFalse(output.wait_for_first_audio(timeout_s=0.05))

        tts_provider.release.set()
        self.assertTrue(output.wait_for_first_audio(timeout_s=1.0))
        output.close(timeout_s=2.0)
        output.raise_if_error()

        self.assertEqual(started, ["started"])

    def test_output_can_play_through_playback_coordinator(self) -> None:
        tts_provider = SlowTTSProvider()
        player = InterruptiblePlayer()
        coordinator = PlaybackCoordinator(player)

        output = InterruptibleSpeechOutput(
            tts_provider=tts_provider,
            player=player,
            chunk_size=512,
            segment_boundary=lambda text: len(text) if text.strip() else None,
            playback_coordinator=coordinator,
        )

        output.submit_text_delta("Hallo zusammen.")
        output.flush()
        output.close(timeout_s=2.0)
        output.raise_if_error()
        coordinator.close(timeout_s=1.0)

        self.assertTrue(player.played)
        self.assertIn(b"FINAL-1", player.played[0])

    def test_playback_coordinator_waits_for_first_tts_chunk_before_preempting_feedback(self) -> None:
        tts_provider = GatedFirstChunkTTSProvider()
        player = CoordinatorPreemptObservationPlayer()
        coordinator = PlaybackCoordinator(player)

        def _processing_chunks():
            while True:
                yield b"PROCESS"
                sleep(0.01)

        feedback_done = Event()

        def _run_feedback() -> None:
            coordinator.play_wav_chunks(
                owner="working_feedback:processing",
                priority=10,
                chunks=_processing_chunks(),
            )
            feedback_done.set()

        feedback_thread = speech_output_module.Thread(target=_run_feedback, daemon=True)
        feedback_thread.start()
        self.assertTrue(player.feedback_started.wait(timeout=1.0))

        output = InterruptibleSpeechOutput(
            tts_provider=tts_provider,
            player=player,
            chunk_size=512,
            segment_boundary=lambda text: len(text) if text.strip() else None,
            playback_coordinator=coordinator,
        )

        output.submit_text_delta("Hallo zusammen.")
        self.assertTrue(tts_provider.started.wait(timeout=1.0))
        self.assertFalse(player.feedback_stopped.wait(timeout=0.1))
        self.assertFalse(feedback_done.is_set())

        tts_provider.release.set()
        self.assertTrue(output.wait_for_first_audio(timeout_s=1.0))
        self.assertTrue(player.feedback_stopped.wait(timeout=1.0))
        output.close(timeout_s=2.0)
        output.raise_if_error()
        coordinator.close(timeout_s=1.0)
        feedback_thread.join(timeout=1.0)

        self.assertTrue(player.speech_started.is_set())
        self.assertTrue(feedback_done.is_set())
        self.assertTrue(any(b"READY-1" in payload for payload in player.played))

    def test_abort_returns_quickly_when_first_chunk_stalls(self) -> None:
        tts_provider = GatedFirstChunkTTSProvider()
        player = InterruptiblePlayer()

        output = InterruptibleSpeechOutput(
            tts_provider=tts_provider,
            player=player,
            chunk_size=512,
            segment_boundary=lambda text: len(text) if text.strip() else None,
        )

        output.submit_text_delta("Ich schaue kurz nach.")
        self.assertTrue(tts_provider.started.wait(timeout=1.0))

        started_at = time.monotonic()
        stopped = output.abort(timeout_s=0.2)
        elapsed = time.monotonic() - started_at
        tts_provider.release.set()

        self.assertIn(stopped, {True, False})
        self.assertLess(elapsed, 0.5)
        self.assertFalse(output.wait_for_first_audio(timeout_s=0.05))

    def test_close_returns_quickly_when_stream_close_blocks(self) -> None:
        tts_provider = BlockingCloseTTSProvider()
        player = InterruptiblePlayer()

        output = InterruptibleSpeechOutput(
            tts_provider=tts_provider,
            player=player,
            chunk_size=512,
            segment_boundary=lambda text: len(text) if text.strip() else None,
        )

        output.submit_text_delta("Hallo zusammen.")
        self.assertTrue(output.wait_for_first_audio(timeout_s=1.0))

        started_at = time.monotonic()
        output.close(timeout_s=0.2)
        elapsed = time.monotonic() - started_at
        self.assertTrue(tts_provider.iterator.close_started.wait(timeout=1.0))
        tts_provider.iterator.release_close.set()
        output.raise_if_error()

        self.assertLess(elapsed, 0.5)
        self.assertTrue(player.played)

    def test_interrupt_path_uses_short_pump_join_timeout_with_playback_coordinator(self) -> None:
        tts_provider = GatedFirstChunkTTSProvider()
        player = InterruptiblePlayer()
        coordinator = PlaybackCoordinator(player)
        recorded_timeouts: list[float | None] = []
        original_join = speech_output_module._TTSChunkPump.join

        output = InterruptibleSpeechOutput(
            tts_provider=tts_provider,
            player=player,
            chunk_size=512,
            segment_boundary=lambda text: len(text) if text.strip() else None,
            playback_coordinator=coordinator,
        )

        def _record_join(pump, *, timeout_s=None):
            recorded_timeouts.append(timeout_s)
            return original_join(pump, timeout_s=0.0)

        with mock.patch.object(speech_output_module._TTSChunkPump, "join", autospec=True, side_effect=_record_join):
            output.submit_text_delta("Ich schaue kurz nach.")
            self.assertTrue(tts_provider.started.wait(timeout=1.0))
            output.abort(timeout_s=0.2)
            tts_provider.release.set()

        coordinator.close(timeout_s=1.0)

        self.assertTrue(recorded_timeouts)
        self.assertLessEqual(recorded_timeouts[0] or 0.0, 0.05)


if __name__ == "__main__":
    unittest.main()
