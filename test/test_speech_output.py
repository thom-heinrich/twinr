from pathlib import Path
from threading import Event
from time import sleep
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.tools import SpeechLaneDelta
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
