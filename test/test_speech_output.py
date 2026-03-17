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
        self.assertTrue(tts_provider.started.wait(timeout=1.0))
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
