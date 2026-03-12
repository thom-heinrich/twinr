from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.providers.openai_backend import OpenAITextResponse
from twinr.runner import TwinrHardwareLoop
from twinr.runtime import TwinrRuntime


class FakeBackend:
    def __init__(self) -> None:
        self.transcribe_calls: list[tuple[bytes, str, str]] = []
        self.respond_calls: list[tuple[str, tuple[tuple[str, str], ...] | None, bool | None]] = []
        self.synthesize_calls: list[str] = []
        self.print_calls: list[tuple[tuple[tuple[str, str], ...] | None, str | None, str | None, str]] = []
        self.transcript = "Hello Twinr"
        self.answer = "Hello back."

    def transcribe(self, audio_bytes: bytes, *, filename: str, content_type: str) -> str:
        self.transcribe_calls.append((audio_bytes, filename, content_type))
        return self.transcript

    def respond_streaming(self, prompt: str, *, conversation=None, allow_web_search=None, on_text_delta=None) -> OpenAITextResponse:
        self.respond_calls.append((prompt, conversation, allow_web_search))
        if on_text_delta is not None:
            for chunk in ("Hello ", "back."):
                on_text_delta(chunk)
        return OpenAITextResponse(
            text=self.answer,
            response_id="resp_123",
            request_id="req_123",
            used_web_search=True,
        )

    def synthesize(self, text: str) -> bytes:
        self.synthesize_calls.append(text)
        return b"RIFF"

    def synthesize_stream(self, text: str):
        self.synthesize_calls.append(text)
        yield b"RI"
        yield b"FF"

    def compose_print_job_with_metadata(
        self,
        *,
        conversation=None,
        focus_hint: str | None = None,
        direct_text: str | None = None,
        request_source: str = "button",
    ) -> OpenAITextResponse:
        self.print_calls.append((conversation, focus_hint, direct_text, request_source))
        return OpenAITextResponse(text="HELLO BACK")


class FakeRecorder:
    def __init__(self) -> None:
        self.pause_values: list[int] = []

    def record_until_pause(self, *, pause_ms: int) -> bytes:
        self.pause_values.append(pause_ms)
        return b"WAVDATA"


class FakePlayer:
    def __init__(self) -> None:
        self.played: list[bytes] = []

    def play_wav_bytes(self, audio_bytes: bytes) -> None:
        self.played.append(audio_bytes)

    def play_wav_chunks(self, chunks) -> None:
        self.played.append(b"".join(chunks))


class FakePrinter:
    def __init__(self) -> None:
        self.printed: list[str] = []

    def print_text(self, text: str) -> str:
        self.printed.append(text)
        return "request id is Test-1 (1 file(s))"


class HardwareLoopTests(unittest.TestCase):
    def make_loop(self, *, backend=None) -> tuple[TwinrHardwareLoop, list[str], FakeRecorder, FakePlayer, FakePrinter]:
        lines: list[str] = []
        recorder = FakeRecorder()
        player = FakePlayer()
        printer = FakePrinter()
        loop = TwinrHardwareLoop(
            config=TwinrConfig(),
            runtime=TwinrRuntime(config=TwinrConfig()),
            backend=backend or FakeBackend(),
            button_monitor=SimpleNamespace(__enter__=lambda self: self, __exit__=lambda self, exc_type, exc, tb: None),
            recorder=recorder,
            player=player,
            printer=printer,
            emit=lines.append,
            sleep=lambda _seconds: None,
            error_reset_seconds=0.0,
        )
        return loop, lines, recorder, player, printer

    def test_green_button_runs_full_audio_turn(self) -> None:
        backend = FakeBackend()
        loop, lines, recorder, player, _printer = self.make_loop(backend=backend)

        loop.handle_button_press("green")

        self.assertEqual(recorder.pause_values, [1200])
        self.assertEqual(backend.transcribe_calls[0][0], b"WAVDATA")
        self.assertEqual(backend.transcribe_calls[0][1], "twinr-listen.wav")
        self.assertEqual(backend.respond_calls[0][0], "Hello Twinr")
        self.assertFalse(backend.respond_calls[0][2])
        self.assertEqual(player.played, [b"RIFF"])
        self.assertEqual(loop.runtime.last_response, "Hello back.")
        self.assertIn("status=listening", lines)
        self.assertIn("status=processing", lines)
        self.assertIn("status=answering", lines)
        self.assertIn("status=waiting", lines)
        self.assertIn("timing_playback_ms=streamed", lines)

    def test_yellow_button_formats_and_prints_last_answer(self) -> None:
        backend = FakeBackend()
        loop, lines, _recorder, _player, printer = self.make_loop(backend=backend)
        loop.runtime.last_response = "Hello back"

        loop.handle_button_press("yellow")

        self.assertEqual(
            backend.print_calls,
            [((), None, "Hello back", "button")],
        )
        self.assertEqual(printer.printed, ["HELLO BACK"])
        self.assertIn("status=printing", lines)
        self.assertEqual(lines[-1], "status=waiting")

    def test_errors_reset_runtime_to_waiting(self) -> None:
        loop, lines, _recorder, _player, _printer = self.make_loop(backend=FakeBackend())

        loop.handle_button_press("yellow")

        self.assertIn("status=error", lines)
        self.assertTrue(any(line.startswith("error=") for line in lines))
        self.assertEqual(loop.runtime.status.value, "waiting")

    def test_auto_web_search_enables_search_for_freshness_queries(self) -> None:
        backend = FakeBackend()
        backend.transcript = "What is the weather today in Berlin?"
        loop, _lines, _recorder, _player, _printer = self.make_loop(backend=backend)

        loop.handle_button_press("green")

        self.assertTrue(backend.respond_calls[0][2])


if __name__ == "__main__":
    unittest.main()
