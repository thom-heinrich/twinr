from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.providers.openai_backend import OpenAITextResponse
from twinr.providers.openai_realtime import OpenAIRealtimeTurn
from twinr.realtime_runner import TwinrRealtimeHardwareLoop
from twinr.runtime import TwinrRuntime


class FakeRealtimeSession:
    def __init__(self) -> None:
        self.calls: list[bytes] = []
        self.conversations: list[tuple[tuple[str, str], ...] | None] = []
        self.entered = False
        self.exited = False
        self.turns = [
            OpenAIRealtimeTurn(
                transcript="Hallo Twinr",
                response_text="Guten Tag",
                response_id="resp_rt_123",
                end_conversation=False,
            )
        ]

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.exited = True

    def run_audio_turn(
        self,
        audio_pcm: bytes,
        *,
        conversation=None,
        on_audio_chunk=None,
        on_output_text_delta=None,
    ) -> OpenAIRealtimeTurn:
        self.calls.append(audio_pcm)
        self.conversations.append(conversation)
        if on_output_text_delta is not None:
            on_output_text_delta("Guten ")
            on_output_text_delta("Tag")
        if on_audio_chunk is not None:
            on_audio_chunk(b"PCM")
        if self.turns:
            return self.turns.pop(0)
        return OpenAIRealtimeTurn(
            transcript="Hallo Twinr",
            response_text="Guten Tag",
            response_id="resp_rt_123",
            end_conversation=False,
        )


class FakePrintBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[tuple[str, str], ...] | None, str | None, str | None, str]] = []

    def compose_print_job_with_metadata(
        self,
        *,
        conversation=None,
        focus_hint: str | None = None,
        direct_text: str | None = None,
        request_source: str = "button",
    ) -> OpenAITextResponse:
        self.calls.append((conversation, focus_hint, direct_text, request_source))
        return OpenAITextResponse(text="GUTEN TAG")


class FakeRecorder:
    def __init__(self, recordings: list[bytes | Exception] | None = None) -> None:
        self.pause_values: list[int] = []
        self.start_timeouts: list[float | None] = []
        self.speech_start_chunks: list[int | None] = []
        self.ignore_initial_ms: list[int] = []
        self.recordings = list(recordings or [b"PCMINPUT"])

    def record_pcm_until_pause_with_options(
        self,
        *,
        pause_ms: int,
        start_timeout_s: float | None = None,
        speech_start_chunks: int | None = None,
        ignore_initial_ms: int = 0,
    ) -> bytes:
        self.pause_values.append(pause_ms)
        self.start_timeouts.append(start_timeout_s)
        self.speech_start_chunks.append(speech_start_chunks)
        self.ignore_initial_ms.append(ignore_initial_ms)
        if not self.recordings:
            return b"PCMINPUT"
        value = self.recordings.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


class FakePlayer:
    def __init__(self) -> None:
        self.played: list[bytes] = []
        self.tones: list[tuple[int, int, float, int]] = []

    def play_tone(
        self,
        *,
        frequency_hz: int = 1046,
        duration_ms: int = 180,
        volume: float = 0.8,
        sample_rate: int = 24000,
    ) -> None:
        self.tones.append((frequency_hz, duration_ms, volume, sample_rate))

    def play_pcm16_chunks(self, chunks, *, sample_rate: int, channels: int = 1) -> None:
        self.played.append(b"".join(chunks))
        self.sample_rate = sample_rate
        self.channels = channels


class FakePrinter:
    def __init__(self) -> None:
        self.printed: list[str] = []

    def print_text(self, text: str) -> str:
        self.printed.append(text)
        return "request id is Test-2 (1 file(s))"


class RealtimeHardwareLoopTests(unittest.TestCase):
    def make_loop(
        self,
        *,
        config: TwinrConfig | None = None,
        recorder: FakeRecorder | None = None,
    ) -> tuple[TwinrRealtimeHardwareLoop, list[str], FakeRealtimeSession, FakePrintBackend, FakeRecorder, FakePlayer, FakePrinter]:
        config = config or TwinrConfig()
        lines: list[str] = []
        realtime_session = FakeRealtimeSession()
        print_backend = FakePrintBackend()
        recorder = recorder or FakeRecorder()
        player = FakePlayer()
        printer = FakePrinter()
        loop = TwinrRealtimeHardwareLoop(
            config=config,
            runtime=TwinrRuntime(config=config),
            realtime_session=realtime_session,
            print_backend=print_backend,
            button_monitor=SimpleNamespace(__enter__=lambda self: self, __exit__=lambda self, exc_type, exc, tb: None),
            recorder=recorder,
            player=player,
            printer=printer,
            emit=lines.append,
            sleep=lambda _seconds: None,
            error_reset_seconds=0.0,
        )
        return loop, lines, realtime_session, print_backend, recorder, player, printer

    def test_green_button_runs_realtime_audio_turn(self) -> None:
        loop, lines, realtime_session, _print_backend, recorder, player, _printer = self.make_loop()

        loop.handle_button_press("green")

        self.assertEqual(recorder.pause_values, [1200])
        self.assertEqual(recorder.start_timeouts, [None])
        self.assertEqual(recorder.speech_start_chunks, [None])
        self.assertEqual(recorder.ignore_initial_ms, [0])
        self.assertEqual(realtime_session.calls, [b"PCMINPUT"])
        self.assertEqual(realtime_session.conversations, [()])
        self.assertTrue(realtime_session.entered)
        self.assertTrue(realtime_session.exited)
        self.assertEqual(len(player.tones), 1)
        self.assertEqual(player.played, [b"PCM"])
        self.assertEqual(loop.runtime.last_transcript, "Hallo Twinr")
        self.assertEqual(loop.runtime.last_response, "Guten Tag")
        self.assertIn("status=listening", lines)
        self.assertIn("status=processing", lines)
        self.assertIn("status=answering", lines)
        self.assertIn("status=waiting", lines)
        self.assertTrue(any(line.startswith("timing_realtime_ms=") for line in lines))

    def test_yellow_button_uses_print_backend(self) -> None:
        loop, lines, _realtime_session, print_backend, _recorder, _player, printer = self.make_loop()
        loop.runtime.last_response = "Guten Tag"

        loop.handle_button_press("yellow")

        self.assertEqual(print_backend.calls, [((), None, "Guten Tag", "button")])
        self.assertEqual(printer.printed, ["GUTEN TAG"])
        self.assertIn("status=printing", lines)
        self.assertEqual(lines[-1], "status=waiting")

    def test_follow_up_turn_beeps_again_and_stops_on_timeout(self) -> None:
        config = TwinrConfig(
            conversation_follow_up_enabled=True,
            conversation_follow_up_timeout_s=3.5,
            audio_follow_up_speech_start_chunks=5,
            audio_follow_up_ignore_ms=420,
        )
        recorder = FakeRecorder(
            recordings=[
                b"PCMINPUT",
                RuntimeError("No speech detected before timeout"),
            ]
        )
        loop, lines, realtime_session, _print_backend, recorder, player, _printer = self.make_loop(
            config=config,
            recorder=recorder,
        )

        loop.handle_button_press("green")

        self.assertEqual(recorder.pause_values, [1200, 1200])
        self.assertEqual(recorder.start_timeouts, [None, 3.5])
        self.assertEqual(recorder.speech_start_chunks, [None, 5])
        self.assertEqual(recorder.ignore_initial_ms, [0, 420])
        self.assertEqual(realtime_session.calls, [b"PCMINPUT"])
        self.assertEqual(realtime_session.conversations[0], ())
        self.assertEqual(len(player.tones), 2)
        self.assertIn("follow_up_timeout=true", lines)
        self.assertEqual(loop.runtime.status.value, "waiting")

    def test_follow_up_turn_receives_previous_conversation_history(self) -> None:
        config = TwinrConfig(
            conversation_follow_up_enabled=True,
            conversation_follow_up_timeout_s=3.5,
            audio_follow_up_speech_start_chunks=5,
            audio_follow_up_ignore_ms=420,
        )
        recorder = FakeRecorder(
            recordings=[
                b"FIRST",
                b"SECOND",
                RuntimeError("No speech detected before timeout"),
            ]
        )
        loop, _lines, realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(
            config=config,
            recorder=recorder,
        )
        realtime_session.turns = [
            OpenAIRealtimeTurn(
                transcript="Erste Frage",
                response_text="Erste Antwort",
                response_id="resp_one",
                end_conversation=False,
            ),
            OpenAIRealtimeTurn(
                transcript="Zweite Frage",
                response_text="Zweite Antwort",
                response_id="resp_two",
                end_conversation=False,
            ),
        ]

        loop.handle_button_press("green")

        self.assertEqual(realtime_session.conversations[0], ())
        self.assertEqual(
            realtime_session.conversations[1],
            (
                ("user", "Erste Frage"),
                ("assistant", "Erste Antwort"),
            ),
        )

    def test_end_conversation_tool_stops_follow_up_loop(self) -> None:
        config = TwinrConfig(
            conversation_follow_up_enabled=True,
            conversation_follow_up_timeout_s=3.5,
        )
        loop, lines, realtime_session, _print_backend, recorder, player, _printer = self.make_loop(
            config=config,
            recorder=FakeRecorder(recordings=[b"PCMINPUT"]),
        )
        realtime_session.turns = [
            OpenAIRealtimeTurn(
                transcript="Danke, das war's",
                response_text="Gern. Bis spaeter.",
                response_id="resp_end",
                end_conversation=True,
            )
        ]

        loop.handle_button_press("green")

        self.assertEqual(recorder.pause_values, [1200])
        self.assertEqual(len(player.tones), 1)
        self.assertIn("conversation_ended=true", lines)
        self.assertEqual(loop.runtime.status.value, "waiting")

    def test_print_tool_call_prints_without_formatter(self) -> None:
        loop, lines, _realtime_session, print_backend, _recorder, _player, printer = self.make_loop()
        loop.runtime.press_green_button()
        loop.runtime.submit_transcript("Bitte drucke das")

        result = loop._handle_print_tool_call({"text": "Wichtige Info"})

        self.assertEqual(
            print_backend.calls,
            [((), None, "Wichtige Info", "tool")],
        )
        self.assertEqual(printer.printed, ["GUTEN TAG"])
        self.assertEqual(result["status"], "printed")
        self.assertEqual(result["text"], "GUTEN TAG")
        self.assertIn("status=printing", lines)
        self.assertIn("print_tool_call=true", lines)


if __name__ == "__main__":
    unittest.main()
