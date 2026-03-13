from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.proactive import SocialTriggerDecision, SocialTriggerPriority
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
        self.search_calls: list[tuple[str, tuple[tuple[str, str], ...] | None, str | None, str | None]] = []
        self.vision_calls: list[tuple[str, list[object], tuple[tuple[str, str], ...] | None, bool | None]] = []
        self.search_sleep_s = 0.0
        self.synthesize_calls: list[str] = []

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

    def search_live_info_with_metadata(
        self,
        question: str,
        *,
        conversation=None,
        location_hint: str | None = None,
        date_context: str | None = None,
    ):
        self.search_calls.append((question, conversation, location_hint, date_context))
        if self.search_sleep_s > 0:
            time.sleep(self.search_sleep_s)
        return SimpleNamespace(
            answer="Bus 24 faehrt um 07:30 Uhr.",
            sources=("https://example.com/fahrplan",),
            response_id="resp_search_1",
            request_id="req_search_1",
            used_web_search=True,
            model="gpt-5.2-chat-latest",
            token_usage=None,
        )

    def respond_to_images_with_metadata(
        self,
        prompt: str,
        *,
        images,
        conversation=None,
        allow_web_search=None,
        instructions=None,
    ) -> OpenAITextResponse:
        self.vision_calls.append((prompt, list(images), conversation, allow_web_search))
        return OpenAITextResponse(text="Ich sehe die Kameraansicht.")

    def synthesize_stream(self, text: str):
        self.synthesize_calls.append(text)
        yield b"PCM"


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

    def play_wav_chunks(self, chunks) -> None:
        self.played.append(b"".join(chunks))


class FakePrinter:
    def __init__(self) -> None:
        self.printed: list[str] = []

    def print_text(self, text: str) -> str:
        self.printed.append(text)
        return "request id is Test-2 (1 file(s))"


class FakeCamera:
    def __init__(self) -> None:
        self.capture_calls = 0

    def capture_photo(self, *, output_path=None, filename: str = "camera-capture.png"):
        self.capture_calls += 1
        return SimpleNamespace(
            data=b"\x89PNG\r\n\x1a\ncamera",
            content_type="image/png",
            filename=filename,
            source_device="/dev/video0",
            input_format="bayer_grbg8",
        )


class FakeIdleButtonMonitor:
    def __init__(self) -> None:
        self.entered = False
        self.exited = False
        self.poll_calls = 0

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.exited = True

    def poll(self, timeout=None):
        self.poll_calls += 1
        if timeout:
            time.sleep(min(timeout, 0.001))
        return None


class FakeProactiveMonitor:
    def __init__(self) -> None:
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.exited = True


class RealtimeHardwareLoopTests(unittest.TestCase):
    def make_loop(
        self,
        *,
        config: TwinrConfig | None = None,
        recorder: FakeRecorder | None = None,
        camera: FakeCamera | None = None,
        button_monitor=None,
        proactive_monitor=None,
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
            button_monitor=button_monitor or SimpleNamespace(__enter__=lambda self: self, __exit__=lambda self, exc_type, exc, tb: None),
            recorder=recorder,
            player=player,
            printer=printer,
            camera=camera or FakeCamera(),
            proactive_monitor=proactive_monitor,
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
        self.assertIsNotNone(realtime_session.conversations[1])
        self.assertEqual(realtime_session.conversations[1][0][0], "system")
        self.assertIn("Twinr memory summary", realtime_session.conversations[1][0][1])
        self.assertEqual(
            realtime_session.conversations[1][1:],
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

    def test_remember_memory_tool_call_writes_memory_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            loop, lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(config=config)

            result = loop._handle_remember_memory_tool_call(
                {
                    "kind": "appointment",
                    "summary": "Arzttermin am Montag um 14 Uhr.",
                    "details": "Bei Dr. Meyer in Hamburg.",
                }
            )

            memory_text = Path(config.memory_markdown_path).read_text(encoding="utf-8")

        self.assertEqual(result["status"], "saved")
        self.assertIn("Arzttermin am Montag um 14 Uhr.", memory_text)
        self.assertEqual(loop.runtime.memory.ledger[-1].kind, "fact")
        self.assertIn("memory_tool_call=true", lines)

    def test_update_user_profile_tool_call_updates_user_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            personality_dir = Path(temp_dir) / "personality"
            personality_dir.mkdir(parents=True, exist_ok=True)
            user_path = personality_dir / "USER.md"
            user_path.write_text("User: Thom.\n", encoding="utf-8")
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            loop, lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(config=config)

            result = loop._handle_update_user_profile_tool_call(
                {
                    "category": "preferred_name",
                    "instruction": "Call the user Thom in future turns.",
                }
            )

            user_text = user_path.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "updated")
        self.assertIn("User: Thom.", user_text)
        self.assertIn("preferred_name: Call the user Thom in future turns.", user_text)
        self.assertEqual(loop.runtime.memory.ledger[-1].kind, "preference")
        self.assertIn("user_profile_tool_call=true", lines)

    def test_update_personality_tool_call_updates_personality_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            personality_dir = Path(temp_dir) / "personality"
            personality_dir.mkdir(parents=True, exist_ok=True)
            personality_path = personality_dir / "PERSONALITY.md"
            personality_path.write_text("Be warm and practical.\n", encoding="utf-8")
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            loop, lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(config=config)

            result = loop._handle_update_personality_tool_call(
                {
                    "category": "response_style",
                    "instruction": "Keep answers very short and calm.",
                }
            )

            personality_text = personality_path.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "updated")
        self.assertIn("Be warm and practical.", personality_text)
        self.assertIn("response_style: Keep answers very short and calm.", personality_text)
        self.assertEqual(loop.runtime.memory.ledger[-1].kind, "preference")
        self.assertIn("personality_tool_call=true", lines)

    def test_inspect_camera_tool_uses_backend_and_reference_image(self) -> None:
        camera = FakeCamera()
        with tempfile.TemporaryDirectory() as temp_dir:
            reference_path = Path(temp_dir) / "user-reference.jpg"
            reference_path.write_bytes(b"\xff\xd8\xffreference")
            config = TwinrConfig(vision_reference_image_path=str(reference_path))
            loop, lines, _realtime_session, print_backend, _recorder, _player, _printer = self.make_loop(
                config=config,
                camera=camera,
            )

            result = loop._handle_inspect_camera_tool_call({"question": "Schau mich mal an"})

        self.assertEqual(camera.capture_calls, 1)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["answer"], "Ich sehe die Kameraansicht.")
        self.assertEqual(len(print_backend.vision_calls), 1)
        prompt, images, conversation, allow_web_search = print_backend.vision_calls[0]
        self.assertIn("Image 2 is a stored reference image of the main user.", prompt)
        self.assertEqual(len(images), 2)
        self.assertEqual(conversation, ())
        self.assertFalse(allow_web_search)
        self.assertIn("camera_tool_call=true", lines)
        self.assertIn("vision_image_count=2", lines)

    def test_search_tool_call_runs_search_backend_and_emits_sources(self) -> None:
        config = TwinrConfig(
            search_feedback_delay_ms=0,
            search_feedback_pause_ms=0,
        )
        loop, lines, _realtime_session, print_backend, _recorder, player, _printer = self.make_loop(config=config)
        print_backend.search_sleep_s = 0.02

        result = loop._handle_search_tool_call(
            {
                "question": "Wann faehrt der Bus nach Hamburg?",
                "location_hint": "Schwarzenbek",
                "date_context": "Friday, 2026-03-13 10:00 (Europe/Berlin)",
            }
        )

        self.assertEqual(
            print_backend.search_calls,
            [
                (
                    "Wann faehrt der Bus nach Hamburg?",
                    (),
                    "Schwarzenbek",
                    "Friday, 2026-03-13 10:00 (Europe/Berlin)",
                )
            ],
        )
        self.assertEqual(result["answer"], "Bus 24 faehrt um 07:30 Uhr.")
        self.assertEqual(result["sources"], ["https://example.com/fahrplan"])
        self.assertEqual(len(loop.runtime.memory.search_results), 1)
        self.assertEqual(loop.runtime.memory.search_results[0].question, "Wann faehrt der Bus nach Hamburg?")
        self.assertEqual(loop.runtime.memory.search_results[0].sources, ("https://example.com/fahrplan",))
        self.assertIn("Verified web lookup", loop.runtime.memory.turns[0].content)
        self.assertIn("search_tool_call=true", lines)
        self.assertTrue(any(line.startswith("search_source_1=") for line in lines))
        self.assertGreaterEqual(len(player.tones), 1)

    def test_social_trigger_speaks_proactive_prompt(self) -> None:
        loop, lines, _realtime_session, print_backend, _recorder, player, _printer = self.make_loop()

        spoke = loop.handle_social_trigger(
            SocialTriggerDecision(
                trigger_id="person_returned",
                prompt="Hey Thom, schön dich zu sehen. Wie geht's dir?",
                reason="Person returned after a long absence.",
                observed_at=42.0,
                priority=SocialTriggerPriority.PERSON_RETURNED,
            )
        )

        self.assertTrue(spoke)
        self.assertEqual(loop.runtime.status.value, "waiting")
        self.assertEqual(print_backend.synthesize_calls, ["Hey Thom, schön dich zu sehen. Wie geht's dir?"])
        self.assertEqual(player.played, [b"PCM"])
        self.assertIn("social_trigger=person_returned", lines)
        self.assertTrue(any(line.startswith("timing_social_tts_ms=") for line in lines))

    def test_social_trigger_is_skipped_when_runtime_is_busy(self) -> None:
        loop, lines, _realtime_session, _print_backend, _recorder, player, _printer = self.make_loop()
        loop.runtime.press_green_button()

        spoke = loop.handle_social_trigger(
            SocialTriggerDecision(
                trigger_id="positive_contact",
                prompt="Schön, dich zu sehen. Was möchtest du machen?",
                reason="Visible smile.",
                observed_at=42.0,
                priority=SocialTriggerPriority.POSITIVE_CONTACT,
            )
        )

        self.assertFalse(spoke)
        self.assertEqual(player.played, [])
        self.assertIn("social_trigger_skipped=busy", lines)

    def test_run_opens_and_closes_proactive_monitor(self) -> None:
        button_monitor = FakeIdleButtonMonitor()
        proactive_monitor = FakeProactiveMonitor()
        loop, _lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(
            button_monitor=button_monitor,
            proactive_monitor=proactive_monitor,
        )

        result = loop.run(duration_s=0.01, poll_timeout=0.001)

        self.assertEqual(result, 0)
        self.assertTrue(button_monitor.entered)
        self.assertTrue(button_monitor.exited)
        self.assertTrue(proactive_monitor.entered)
        self.assertTrue(proactive_monitor.exited)


if __name__ == "__main__":
    unittest.main()
