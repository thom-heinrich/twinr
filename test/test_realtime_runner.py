from array import array
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import math
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.automations import AutomationAction
from twinr.config import TwinrConfig
from twinr.hardware import VoiceAssessment
from twinr.memory.reminders import now_in_timezone
from twinr.proactive import SocialTriggerDecision, SocialTriggerPriority, WakewordMatch
from twinr.providers.openai import OpenAITextResponse
from twinr.providers.openai.realtime import OpenAIRealtimeTurn
from twinr.realtime_runner import TwinrRealtimeHardwareLoop
from twinr.runtime import TwinrRuntime


def _voice_sample_pcm_bytes(*, frequency_hz: float = 175.0, amplitude: float = 0.35, duration_s: float = 1.8) -> bytes:
    sample_rate = 24000
    total_frames = int(sample_rate * duration_s)
    frames = array("h")
    for index in range(total_frames):
        t = index / sample_rate
        envelope = min(1.0, index / (sample_rate * 0.2), (total_frames - index) / (sample_rate * 0.2))
        sample = amplitude * envelope * (
            (0.70 * math.sin(2.0 * math.pi * frequency_hz * t))
            + (0.20 * math.sin(2.0 * math.pi * frequency_hz * 2.0 * t))
            + (0.10 * math.sin(2.0 * math.pi * (frequency_hz + 35.0) * t))
        )
        frames.append(max(-32767, min(32767, int(sample * 32767))))
    return frames.tobytes()


class FakeRealtimeSession:
    def __init__(self) -> None:
        self.calls: list[bytes] = []
        self.text_calls: list[str] = []
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

    def run_text_turn(
        self,
        prompt: str,
        *,
        conversation=None,
        on_audio_chunk=None,
        on_output_text_delta=None,
    ) -> OpenAIRealtimeTurn:
        self.text_calls.append(prompt)
        self.conversations.append(conversation)
        if on_output_text_delta is not None:
            on_output_text_delta("Guten ")
            on_output_text_delta("Tag")
        if on_audio_chunk is not None:
            on_audio_chunk(b"PCM")
        if self.turns:
            return self.turns.pop(0)
        return OpenAIRealtimeTurn(
            transcript=prompt,
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
        self.reminder_calls: list[object] = []
        self.generic_reminder_calls: list[tuple[str, str | None, bool | None]] = []
        self.automation_calls: list[tuple[str, bool, str]] = []
        self.proactive_calls: list[tuple[str, str, str, int, tuple[tuple[str, str], ...] | None, tuple[str, ...]]] = []
        self.transcribe_calls: list[tuple[bytes, str | None, str | None]] = []
        self.transcribe_result = "hey twinr"

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

    def phrase_due_reminder_with_metadata(self, reminder):
        self.reminder_calls.append(reminder)
        return OpenAITextResponse(
            text=f"Erinnerung: {reminder.summary}",
            response_id="resp_reminder_1",
            request_id="req_reminder_1",
            used_web_search=False,
        )

    def phrase_proactive_prompt_with_metadata(
        self,
        *,
        trigger_id: str,
        reason: str,
        default_prompt: str,
        priority: int,
        conversation=None,
        recent_prompts=(),
        observation_facts=(),
    ) -> OpenAITextResponse:
        self.proactive_calls.append(
            (
                trigger_id,
                reason,
                default_prompt,
                priority,
                conversation,
                tuple(recent_prompts),
                tuple(observation_facts),
            )
        )
        return OpenAITextResponse(
            text=f"Proaktiv: {trigger_id}",
            response_id="resp_social_1",
            request_id="req_social_1",
            model="gpt-5.2",
            token_usage=None,
            used_web_search=False,
        )

    def fulfill_automation_prompt_with_metadata(
        self,
        prompt: str,
        *,
        allow_web_search: bool,
        delivery: str = "spoken",
    ) -> OpenAITextResponse:
        self.automation_calls.append((prompt, allow_web_search, delivery))
        text = "Die Wettervorhersage ist trocken und mild."
        if delivery == "printed":
            text = "Schlagzeilen: Markt ruhig. Wetter mild."
        return OpenAITextResponse(
            text=text,
            response_id="resp_auto_1",
            request_id="req_auto_1",
            used_web_search=allow_web_search,
            model="gpt-5.2",
            token_usage=None,
        )

    def respond_with_metadata(
        self,
        prompt: str,
        *,
        conversation=None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
    ) -> OpenAITextResponse:
        self.generic_reminder_calls.append((prompt, instructions, allow_web_search))
        return OpenAITextResponse(
            text="Erinnerung: Medikament nehmen",
            response_id="resp_generic_1",
            request_id="req_generic_1",
            used_web_search=False,
            model="gpt-5.2",
            token_usage=None,
        )

    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "audio.wav",
        content_type: str = "audio/wav",
        language: str | None = None,
        prompt: str | None = None,
    ) -> str:
        self.transcribe_calls.append((audio_bytes, language, prompt))
        return self.transcribe_result


class FakeRecorder:
    def __init__(self, recordings: list[bytes | Exception] | None = None) -> None:
        self.pause_values: list[int] = []
        self.start_timeouts: list[float | None] = []
        self.speech_start_chunks: list[int | None] = []
        self.ignore_initial_ms: list[int] = []
        self.pause_grace_values: list[int] = []
        self.recordings = list(recordings or [b"PCMINPUT"])

    def capture_pcm_until_pause_with_options(
        self,
        *,
        pause_ms: int,
        start_timeout_s: float | None = None,
        max_record_seconds: float | None = None,
        speech_start_chunks: int | None = None,
        ignore_initial_ms: int = 0,
        pause_grace_ms: int = 0,
    ):
        del max_record_seconds
        self.pause_values.append(pause_ms)
        self.start_timeouts.append(start_timeout_s)
        self.speech_start_chunks.append(speech_start_chunks)
        self.ignore_initial_ms.append(ignore_initial_ms)
        self.pause_grace_values.append(pause_grace_ms)
        if not self.recordings:
            value: bytes | Exception = b"PCMINPUT"
        else:
            value = self.recordings.pop(0)
        if isinstance(value, Exception):
            raise value
        if hasattr(value, "pcm_bytes"):
            return value
        return SimpleNamespace(
            pcm_bytes=value,
            speech_started_after_ms=2100,
            resumed_after_pause_count=0,
        )

    def record_pcm_until_pause_with_options(
        self,
        *,
        pause_ms: int,
        start_timeout_s: float | None = None,
        max_record_seconds: float | None = None,
        speech_start_chunks: int | None = None,
        ignore_initial_ms: int = 0,
        pause_grace_ms: int = 0,
    ) -> bytes:
        result = self.capture_pcm_until_pause_with_options(
            pause_ms=pause_ms,
            start_timeout_s=start_timeout_s,
            max_record_seconds=max_record_seconds,
            speech_start_chunks=speech_start_chunks,
            ignore_initial_ms=ignore_initial_ms,
            pause_grace_ms=pause_grace_ms,
        )
        return result.pcm_bytes


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
    _LANGUAGE_CONTRACT = "All user-facing spoken and written replies for this turn must be in German."

    def make_loop(
        self,
        *,
        config: TwinrConfig | None = None,
        recorder: FakeRecorder | None = None,
        camera: FakeCamera | None = None,
        button_monitor=None,
        print_backend: FakePrintBackend | None = None,
        stt_provider=None,
        agent_provider=None,
        tts_provider=None,
        voice_profile_monitor=None,
        proactive_monitor=None,
    ) -> tuple[TwinrRealtimeHardwareLoop, list[str], FakeRealtimeSession, FakePrintBackend, FakeRecorder, FakePlayer, FakePrinter]:
        temp_dir_handle = tempfile.TemporaryDirectory()
        temp_root = Path(temp_dir_handle.name)
        config = config or TwinrConfig()
        config = replace(
            config,
            project_root=str(temp_root if config.project_root == "." else Path(config.project_root)),
            runtime_state_path=self._sandbox_path(
                config.runtime_state_path,
                temp_root / "runtime-state.json",
                default="/tmp/twinr-runtime-state.json",
            ),
            reminder_store_path=self._sandbox_path(
                config.reminder_store_path,
                temp_root / "state" / "reminders.json",
                default="state/reminders.json",
            ),
            automation_store_path=self._sandbox_path(
                config.automation_store_path,
                temp_root / "state" / "automations.json",
                default="state/automations.json",
            ),
            adaptive_timing_store_path=self._sandbox_path(
                config.adaptive_timing_store_path,
                temp_root / "state" / "adaptive-timing.json",
                default="state/adaptive_timing.json",
            ),
        )
        lines: list[str] = []
        realtime_session = FakeRealtimeSession()
        resolved_print_backend = print_backend
        if resolved_print_backend is None:
            for provider in (agent_provider, stt_provider, tts_provider):
                if provider is not None:
                    resolved_print_backend = provider
                    break
        if resolved_print_backend is None:
            resolved_print_backend = FakePrintBackend()
        recorder = recorder or FakeRecorder()
        player = FakePlayer()
        printer = FakePrinter()
        loop = TwinrRealtimeHardwareLoop(
            config=config,
            runtime=TwinrRuntime(config=config),
            realtime_session=realtime_session,
            print_backend=resolved_print_backend,
            stt_provider=stt_provider,
            agent_provider=agent_provider,
            tts_provider=tts_provider,
            button_monitor=button_monitor or SimpleNamespace(__enter__=lambda self: self, __exit__=lambda self, exc_type, exc, tb: None),
            recorder=recorder,
            player=player,
            printer=printer,
            camera=camera or FakeCamera(),
            voice_profile_monitor=voice_profile_monitor,
            proactive_monitor=proactive_monitor,
            emit=lines.append,
            sleep=lambda _seconds: None,
            error_reset_seconds=0.0,
        )
        loop._test_temp_dir = temp_dir_handle
        return loop, lines, realtime_session, resolved_print_backend, recorder, player, printer

    @staticmethod
    def _sandbox_path(current: str, isolated: Path, *, default: str) -> str:
        path = Path(current)
        if current == default or not path.is_absolute():
            return str(isolated)
        return current

    def _assert_language_contract_only(self, conversation: tuple[tuple[str, str], ...] | None) -> None:
        self.assertIsNotNone(conversation)
        assert conversation is not None
        self.assertEqual(len(conversation), 1)
        self.assertEqual(conversation[0][0], "system")
        self.assertIn(self._LANGUAGE_CONTRACT, conversation[0][1])

    def test_green_button_runs_realtime_audio_turn(self) -> None:
        loop, lines, realtime_session, _print_backend, recorder, player, _printer = self.make_loop()

        loop.handle_button_press("green")

        self.assertEqual(recorder.pause_values, [1200])
        self.assertEqual(recorder.start_timeouts, [8.0])
        self.assertEqual(recorder.speech_start_chunks, [None])
        self.assertEqual(recorder.ignore_initial_ms, [0])
        self.assertEqual(recorder.pause_grace_values, [900])
        self.assertEqual(realtime_session.calls, [b"PCMINPUT"])
        self.assertEqual(len(realtime_session.conversations), 1)
        self._assert_language_contract_only(realtime_session.conversations[0])
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

    def test_yellow_button_accepts_split_support_providers(self) -> None:
        backend = FakePrintBackend()
        loop, _lines, _realtime_session, _print_backend, _recorder, _player, printer = self.make_loop(
            print_backend=None,
            stt_provider=backend,
            agent_provider=backend,
            tts_provider=backend,
        )
        loop.runtime.last_response = "Guten Tag"

        loop.handle_button_press("yellow")

        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(printer.printed, ["GUTEN TAG"])

    def test_wakeword_with_remaining_text_runs_direct_text_turn(self) -> None:
        loop, lines, realtime_session, _print_backend, recorder, player, _printer = self.make_loop()

        handled = loop.handle_wakeword_match(
            WakewordMatch(
                detected=True,
                transcript="Hey Twinr wie spaet ist es",
                matched_phrase="hey twinr",
                remaining_text="wie spaet ist es",
                normalized_transcript="hey twinr wie spaet ist es",
            )
        )

        self.assertTrue(handled)
        self.assertEqual(realtime_session.text_calls, ["wie spaet ist es"])
        self.assertEqual(realtime_session.calls, [])
        self.assertEqual(recorder.pause_values, [])
        self.assertEqual(len(player.tones), 0)
        self.assertIn("wakeword_mode=direct_text", lines)
        self.assertEqual(loop.runtime.last_transcript, "Hallo Twinr")

    def test_wakeword_without_remaining_text_opens_listening_window(self) -> None:
        loop, lines, realtime_session, _print_backend, recorder, player, _printer = self.make_loop()

        handled = loop.handle_wakeword_match(
            WakewordMatch(
                detected=True,
                transcript="Hey Twinr",
                matched_phrase="hey twinr",
                remaining_text="",
                normalized_transcript="hey twinr",
            )
        )

        self.assertTrue(handled)
        self.assertEqual(realtime_session.calls, [b"PCMINPUT"])
        self.assertEqual(realtime_session.text_calls, [])
        self.assertEqual(recorder.pause_values, [1200])
        self.assertEqual(recorder.start_timeouts, [loop.config.conversation_follow_up_timeout_s])
        self.assertEqual(recorder.speech_start_chunks, [loop.config.audio_follow_up_speech_start_chunks])
        self.assertEqual(recorder.ignore_initial_ms, [loop.config.audio_follow_up_ignore_ms])
        self.assertEqual(recorder.pause_grace_values, [loop.config.adaptive_timing_pause_grace_ms])
        self.assertEqual(len(player.tones), 1)
        self.assertEqual(player.played, [b"PCM", b"PCM"])
        self.assertIn("wakeword_mode=listen", lines)
        self.assertIn("wakeword_ack=Ja?", lines)

    def test_yellow_button_uses_print_backend(self) -> None:
        loop, lines, _realtime_session, print_backend, _recorder, _player, printer = self.make_loop()
        loop.runtime.last_response = "Guten Tag"

        loop.handle_button_press("yellow")

        self.assertEqual(len(print_backend.calls), 1)
        self._assert_language_contract_only(print_backend.calls[0][0])
        self.assertEqual(print_backend.calls[0][1:], (None, "Guten Tag", "button"))
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
        self.assertEqual(recorder.start_timeouts, [8.0, 3.5])
        self.assertEqual(recorder.speech_start_chunks, [None, 5])
        self.assertEqual(recorder.ignore_initial_ms, [0, 420])
        self.assertEqual(recorder.pause_grace_values, [900, 900])
        self.assertEqual(realtime_session.calls, [b"PCMINPUT"])
        self._assert_language_contract_only(realtime_session.conversations[0])
        self.assertEqual(len(player.tones), 2)
        self.assertIn("follow_up_timeout=true", lines)
        self.assertEqual(loop.runtime.status.value, "waiting")

    def test_button_listening_timeout_adapts_next_start_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
                adaptive_timing_store_path=str(Path(temp_dir) / "adaptive-timing.json"),
            )
            recorder = FakeRecorder(
                recordings=[
                    RuntimeError("No speech detected before timeout"),
                    b"PCMINPUT",
                ]
            )
            loop, lines, realtime_session, _print_backend, recorder, player, _printer = self.make_loop(
                config=config,
                recorder=recorder,
            )

            loop.handle_button_press("green")
            loop.handle_button_press("green")

        self.assertEqual(recorder.start_timeouts, [8.0, 8.75])
        self.assertEqual(realtime_session.calls, [b"PCMINPUT"])
        self.assertEqual(len(player.tones), 2)
        self.assertIn("listen_timeout=true", lines)

    def test_resumed_pause_adapts_next_pause_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
                adaptive_timing_store_path=str(Path(temp_dir) / "adaptive-timing.json"),
            )
            recorder = FakeRecorder(
                recordings=[
                    SimpleNamespace(
                        pcm_bytes=b"FIRST",
                        speech_started_after_ms=2100,
                        resumed_after_pause_count=1,
                    ),
                    b"SECOND",
                ]
            )
            loop, _lines, realtime_session, _print_backend, recorder, _player, _printer = self.make_loop(
                config=config,
                recorder=recorder,
            )

            loop.handle_button_press("green")
            loop.handle_button_press("green")

        self.assertEqual(recorder.pause_values, [1200, 1340])
        self.assertEqual(recorder.pause_grace_values, [900, 990])
        self.assertEqual(realtime_session.calls, [b"FIRST", b"SECOND"])

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

        self._assert_language_contract_only(realtime_session.conversations[0])
        self.assertIsNotNone(realtime_session.conversations[1])
        self.assertEqual(realtime_session.conversations[1][0][0], "system")
        self.assertIn(self._LANGUAGE_CONTRACT, realtime_session.conversations[1][0][1])
        self.assertEqual(realtime_session.conversations[1][1][0], "system")
        self.assertIn("Twinr memory summary", realtime_session.conversations[1][1][1])
        self.assertEqual(
            realtime_session.conversations[1][2:],
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
        self.assertEqual(recorder.pause_grace_values, [900])
        self.assertEqual(len(player.tones), 1)
        self.assertIn("conversation_ended=true", lines)
        self.assertEqual(loop.runtime.status.value, "waiting")

    def test_green_button_updates_runtime_voice_assessment(self) -> None:
        class FakeVoiceProfileMonitor:
            def assess_pcm16(self, audio_pcm: bytes, *, sample_rate: int, channels: int) -> VoiceAssessment:
                self.audio_pcm = audio_pcm
                self.sample_rate = sample_rate
                self.channels = channels
                return VoiceAssessment(
                    status="uncertain",
                    label="Uncertain",
                    detail="Partial match.",
                    confidence=0.63,
                    checked_at="2026-03-13T12:15:00+00:00",
                )

        monitor = FakeVoiceProfileMonitor()
        loop, lines, realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(
            voice_profile_monitor=monitor,
        )

        loop.handle_button_press("green")

        self.assertEqual(monitor.audio_pcm, b"PCMINPUT")
        self.assertEqual(monitor.sample_rate, loop.config.openai_realtime_input_sample_rate)
        self.assertEqual(monitor.channels, loop.config.audio_channels)
        self.assertEqual(loop.runtime.user_voice_status, "uncertain")
        self.assertEqual(loop.runtime.user_voice_confidence, 0.63)
        self.assertEqual(loop.runtime.user_voice_checked_at, "2026-03-13T12:15:00+00:00")
        self.assertIsNotNone(realtime_session.conversations[0])
        self.assertTrue(
            any(
                role == "system" and "Speaker signal: partial match" in content
                for role, content in realtime_session.conversations[0]
            )
        )
        self.assertIn("voice_profile_status=uncertain", lines)
        self.assertIn("voice_profile_confidence=0.63", lines)

    def test_print_tool_call_prints_without_formatter(self) -> None:
        loop, lines, _realtime_session, print_backend, _recorder, _player, printer = self.make_loop()
        loop.runtime.press_green_button()
        loop.runtime.submit_transcript("Bitte drucke das")

        result = loop._handle_print_tool_call({"text": "Wichtige Info"})

        self.assertEqual(len(print_backend.calls), 1)
        self._assert_language_contract_only(print_backend.calls[0][0])
        self.assertEqual(print_backend.calls[0][1:], (None, "Wichtige Info", "tool"))
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

    def test_schedule_reminder_tool_call_writes_reminder_store(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reminder_path = Path(temp_dir) / "state" / "reminders.json"
            config = TwinrConfig(
                project_root=temp_dir,
                reminder_store_path=str(reminder_path),
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            loop, lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(config=config)

            result = loop._handle_schedule_reminder_tool_call(
                {
                    "due_at": "2026-03-15T12:00:00+01:00",
                    "summary": "Arzttermin",
                    "details": "Bei Dr. Meyer",
                    "kind": "appointment",
                }
            )

            reminder_text = reminder_path.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "scheduled")
        self.assertIn("Arzttermin", reminder_text)
        self.assertIn("appointment", reminder_text)
        self.assertEqual(loop.runtime.memory.ledger[-1].kind, "reminder")
        self.assertIn("reminder_tool_call=true", lines)

    def test_list_create_update_and_delete_time_automation_tool_calls(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                automation_store_path=str(Path(temp_dir) / "state" / "automations.json"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            loop, lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(config=config)

            created = loop._handle_create_time_automation_tool_call(
                {
                    "name": "Daily weather",
                    "schedule": "daily",
                    "time_of_day": "08:00",
                    "delivery": "spoken",
                    "content_mode": "llm_prompt",
                    "content": "Give the morning weather report.",
                    "allow_web_search": True,
                }
            )
            listed = loop._handle_list_automations_tool_call({})
            updated = loop._handle_update_time_automation_tool_call(
                {
                    "automation_ref": created["automation"]["automation_id"],
                    "time_of_day": "09:15",
                    "delivery": "printed",
                    "content": "Print the morning weather report.",
                }
            )
            deleted = loop._handle_delete_automation_tool_call(
                {"automation_ref": created["automation"]["automation_id"]}
            )

        self.assertEqual(created["status"], "created")
        self.assertEqual(created["automation"]["schedule"], "daily")
        self.assertEqual(created["automation"]["delivery"], "spoken")
        self.assertEqual(listed["count"], 1)
        self.assertEqual(updated["status"], "updated")
        self.assertEqual(updated["automation"]["time_of_day"], "09:15")
        self.assertEqual(updated["automation"]["delivery"], "printed")
        self.assertEqual(deleted["status"], "deleted")
        self.assertIn("automation_tool_call=true", lines)

    def test_list_create_update_and_delete_sensor_automation_tool_calls(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                automation_store_path=str(Path(temp_dir) / "state" / "automations.json"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            loop, lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(config=config)

            created = loop._handle_create_sensor_automation_tool_call(
                {
                    "name": "Visitor hello",
                    "trigger_kind": "camera_person_visible",
                    "delivery": "spoken",
                    "content_mode": "llm_prompt",
                    "content": "Say hello to the visitor.",
                    "allow_web_search": False,
                    "cooldown_seconds": 90,
                }
            )
            listed = loop._handle_list_automations_tool_call({})
            updated = loop._handle_update_sensor_automation_tool_call(
                {
                    "automation_ref": created["automation"]["automation_id"],
                    "trigger_kind": "vad_quiet",
                    "hold_seconds": 45,
                    "delivery": "printed",
                    "content": "Print a quiet-room note.",
                }
            )
            deleted = loop._handle_delete_automation_tool_call(
                {"automation_ref": created["automation"]["automation_id"]}
            )

        self.assertEqual(created["status"], "created")
        self.assertEqual(created["automation"]["trigger_kind"], "if_then")
        self.assertEqual(created["automation"]["sensor_trigger_kind"], "camera_person_visible")
        self.assertEqual(listed["count"], 1)
        self.assertEqual(updated["status"], "updated")
        self.assertEqual(updated["automation"]["sensor_trigger_kind"], "vad_quiet")
        self.assertEqual(updated["automation"]["sensor_hold_seconds"], 45.0)
        self.assertEqual(updated["automation"]["delivery"], "printed")
        self.assertEqual(deleted["status"], "deleted")
        self.assertIn("automation_tool_call=true", lines)

    def test_idle_loop_executes_due_spoken_automation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                automation_store_path=str(Path(temp_dir) / "state" / "automations.json"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
                automation_poll_interval_s=0.0,
            )
            loop, lines, _realtime_session, print_backend, _recorder, player, _printer = self.make_loop(config=config)
            entry = loop.runtime.create_time_automation(
                name="Daily weather",
                schedule="daily",
                time_of_day=now_in_timezone(config.local_timezone_name).strftime("%H:%M"),
                actions=(
                    AutomationAction(
                        kind="llm_prompt",
                        text="Give the morning weather report.",
                        payload={"delivery": "spoken", "allow_web_search": True},
                        enabled=True,
                    ),
                ),
                source="test",
            )

            executed = loop._maybe_run_due_automation()
            stored = loop.runtime.automation_store.get(entry.automation_id)

        self.assertTrue(executed)
        self.assertEqual(print_backend.automation_calls, [("Give the morning weather report.", True, "spoken")])
        self.assertEqual(print_backend.synthesize_calls, ["Die Wettervorhersage ist trocken und mild."])
        self.assertEqual(player.played, [b"PCM"])
        assert stored is not None
        self.assertIsNotNone(stored.last_triggered_at)
        self.assertIn("automation_executed=true", lines)

    def test_idle_loop_executes_due_printed_automation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                automation_store_path=str(Path(temp_dir) / "state" / "automations.json"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
                automation_poll_interval_s=0.0,
            )
            loop, lines, _realtime_session, print_backend, _recorder, _player, printer = self.make_loop(config=config)
            entry = loop.runtime.create_time_automation(
                name="Daily headlines",
                schedule="daily",
                time_of_day=now_in_timezone(config.local_timezone_name).strftime("%H:%M"),
                actions=(
                    AutomationAction(
                        kind="llm_prompt",
                        text="Print the main headlines of the day.",
                        payload={"delivery": "printed", "allow_web_search": True},
                        enabled=True,
                    ),
                ),
                source="test",
            )

            executed = loop._maybe_run_due_automation()
            stored = loop.runtime.automation_store.get(entry.automation_id)

        self.assertTrue(executed)
        self.assertEqual(print_backend.automation_calls, [("Print the main headlines of the day.", True, "printed")])
        self.assertEqual(printer.printed, ["GUTEN TAG"])
        assert stored is not None
        self.assertIsNotNone(stored.last_triggered_at)
        self.assertIn("automation_print_job=request id is Test-2 (1 file(s))", lines)

    def test_idle_loop_executes_sensor_automation_from_camera_event(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                automation_store_path=str(Path(temp_dir) / "state" / "automations.json"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            loop, lines, _realtime_session, print_backend, _recorder, player, _printer = self.make_loop(config=config)
            loop._handle_create_sensor_automation_tool_call(
                {
                    "name": "Visitor hello",
                    "trigger_kind": "camera_person_visible",
                    "delivery": "spoken",
                    "content_mode": "llm_prompt",
                    "content": "Say hello to the visitor.",
                    "allow_web_search": False,
                }
            )

            loop.handle_sensor_observation(
                {
                    "sensor": {"inspected": True, "observed_at": 5.0},
                    "pir": {"motion_detected": True, "low_motion": False, "no_motion_for_s": 0.0},
                    "camera": {
                        "person_visible": True,
                        "person_visible_for_s": 0.0,
                        "looking_toward_device": True,
                        "body_pose": "upright",
                        "smiling": False,
                        "hand_or_object_near_camera": False,
                        "hand_or_object_near_camera_for_s": 0.0,
                    },
                    "vad": {
                        "speech_detected": False,
                        "speech_detected_for_s": 0.0,
                        "quiet": True,
                        "quiet_for_s": 12.0,
                        "distress_detected": False,
                    },
                },
                ("camera.person_visible",),
            )
            executed = loop._maybe_run_sensor_automation()

        self.assertTrue(executed)
        self.assertEqual(print_backend.automation_calls, [("Say hello to the visitor.", False, "spoken")])
        self.assertEqual(player.played, [b"PCM"])
        self.assertIn("automation_trigger_source=sensor", lines)
        self.assertIn("automation_event_name=camera.person_visible", lines)

    def test_idle_loop_executes_sensor_automation_from_duration_facts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                automation_store_path=str(Path(temp_dir) / "state" / "automations.json"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            loop, lines, _realtime_session, _print_backend, _recorder, _player, printer = self.make_loop(config=config)
            loop._handle_create_sensor_automation_tool_call(
                {
                    "name": "Quiet room print",
                    "trigger_kind": "vad_quiet",
                    "hold_seconds": 30,
                    "delivery": "printed",
                    "content_mode": "static_text",
                    "content": "Bitte leise bleiben.",
                    "cooldown_seconds": 120,
                }
            )

            loop.handle_sensor_observation(
                {
                    "sensor": {"inspected": False, "observed_at": 60.0},
                    "pir": {"motion_detected": False, "low_motion": True, "no_motion_for_s": 45.0},
                    "camera": {
                        "person_visible": False,
                        "person_visible_for_s": 0.0,
                        "looking_toward_device": False,
                        "body_pose": "unknown",
                        "smiling": False,
                        "hand_or_object_near_camera": False,
                        "hand_or_object_near_camera_for_s": 0.0,
                    },
                    "vad": {
                        "speech_detected": False,
                        "speech_detected_for_s": 0.0,
                        "quiet": True,
                        "quiet_for_s": 45.0,
                        "distress_detected": False,
                    },
                },
                (),
            )
            executed = loop._maybe_run_sensor_automation()

        self.assertTrue(executed)
        self.assertEqual(printer.printed, ["Bitte leise bleiben."])
        self.assertIn("automation_trigger_source=sensor", lines)

    def test_idle_loop_delivers_due_reminder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                reminder_store_path=str(Path(temp_dir) / "state" / "reminders.json"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
                reminder_poll_interval_s=0.0,
            )
            loop, lines, _realtime_session, print_backend, _recorder, player, _printer = self.make_loop(config=config)
            loop.runtime.schedule_reminder(
                due_at=now_in_timezone(config.local_timezone_name).isoformat(),
                summary="Medikament nehmen",
                kind="medication",
                source="test",
            )

            delivered = loop._maybe_deliver_due_reminder()
            stored_entries = loop.runtime.reminder_store.load_entries()

        self.assertTrue(delivered)
        self.assertEqual(len(print_backend.reminder_calls), 1)
        self.assertEqual(print_backend.synthesize_calls, ["Erinnerung: Medikament nehmen"])
        self.assertEqual(player.played, [b"PCM"])
        self.assertIn("reminder_delivered=true", lines)
        self.assertTrue(stored_entries[0].delivered)

    def test_idle_loop_delivers_due_reminder_with_generic_backend_fallback(self) -> None:
        class LegacyReminderBackend(FakePrintBackend):
            phrase_due_reminder_with_metadata = None

        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                reminder_store_path=str(Path(temp_dir) / "state" / "reminders.json"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
                reminder_poll_interval_s=0.0,
            )
            print_backend = LegacyReminderBackend()
            loop, lines, _realtime_session, _print_backend, _recorder, player, _printer = self.make_loop(
                config=config,
                print_backend=print_backend,
            )
            loop.runtime.schedule_reminder(
                due_at=now_in_timezone(config.local_timezone_name).isoformat(),
                summary="Medikament nehmen",
                kind="medication",
                source="test",
            )

            delivered = loop._maybe_deliver_due_reminder()
            stored_entries = loop.runtime.reminder_store.load_entries()

        self.assertTrue(delivered)
        self.assertEqual(len(print_backend.reminder_calls), 0)
        self.assertEqual(len(print_backend.generic_reminder_calls), 1)
        self.assertIn("Speak the reminder now.", print_backend.generic_reminder_calls[0][0])
        self.assertEqual(player.played, [b"PCM"])
        self.assertIn("reminder_backend_fallback=generic", lines)
        self.assertIn("reminder_delivered=true", lines)
        self.assertTrue(stored_entries[0].delivered)

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

    def test_remember_contact_and_lookup_contact_tool_calls_handle_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            loop, lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(config=config)

            first = loop._handle_remember_contact_tool_call(
                {
                    "given_name": "Corinna",
                    "family_name": "Maier",
                    "phone": "01761234",
                    "role": "Physiotherapeutin",
                }
            )
            second = loop._handle_remember_contact_tool_call(
                {
                    "given_name": "Corinna",
                    "family_name": "Schmidt",
                    "phone": "0309988",
                    "role": "Nachbarin",
                }
            )
            lookup = loop._handle_lookup_contact_tool_call({"name": "Corinna"})
            resolved = loop._handle_lookup_contact_tool_call({"name": "Corinna", "role": "Physiotherapeutin"})

        self.assertEqual(first["status"], "created")
        self.assertEqual(second["status"], "created")
        self.assertEqual(lookup["status"], "needs_clarification")
        self.assertEqual(len(lookup["options"]), 2)
        self.assertEqual(resolved["status"], "found")
        self.assertEqual(resolved["label"], "Corinna Maier")
        self.assertEqual(resolved["phones"], ["01761234"])
        self.assertIn("graph_contact_tool_call=true", lines)
        self.assertIn("graph_contact_lookup=true", lines)

    def test_remember_preference_and_plan_tool_calls_feed_provider_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
                openai_web_search_timezone="Europe/Berlin",
            )
            loop, lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(config=config)

            preference = loop._handle_remember_preference_tool_call(
                {
                    "category": "brand",
                    "value": "Melitta",
                    "for_product": "coffee",
                }
            )
            plan = loop._handle_remember_plan_tool_call(
                {
                    "summary": "go for a walk",
                    "when": "today",
                }
            )
            loop.runtime.last_transcript = "Wie wird das Wetter heute?"
            provider_context = loop.runtime.provider_conversation_context()
            system_contexts = [content for role, content in provider_context if role == "system"]

        self.assertEqual(preference["edge_type"], "user_prefers_brand")
        self.assertEqual(plan["edge_type"], "user_plans")
        self.assertTrue(any("All user-facing spoken and written replies for this turn must be in German." in content for content in system_contexts))
        self.assertTrue(any("Melitta" in content for content in system_contexts))
        self.assertTrue(any("go for a walk" in content for content in system_contexts))
        self.assertIn("graph_preference_tool_call=true", lines)
        self.assertIn("graph_plan_tool_call=true", lines)

    def test_update_user_profile_requires_confirmation_for_unknown_voice(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            personality_dir = Path(temp_dir) / "personality"
            personality_dir.mkdir(parents=True, exist_ok=True)
            (personality_dir / "USER.md").write_text("User: Thom.\n", encoding="utf-8")
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            loop, _lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(config=config)
            loop.runtime.update_user_voice_assessment(
                status="unknown_voice",
                confidence=0.22,
                checked_at="2026-03-13T12:30:00+00:00",
            )

            with self.assertRaisesRegex(RuntimeError, "confirmed=true"):
                loop._handle_update_user_profile_tool_call(
                    {
                        "category": "preferred_name",
                        "instruction": "Call the user Thom in future turns.",
                    }
                )

            result = loop._handle_update_user_profile_tool_call(
                {
                    "category": "preferred_name",
                    "instruction": "Call the user Thom in future turns.",
                    "confirmed": True,
                }
            )

        self.assertEqual(result["status"], "updated")

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

    def test_update_simple_setting_tool_call_persists_and_applies_memory_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "TWINR_MEMORY_MAX_TURNS=20\nTWINR_MEMORY_KEEP_RECENT=10\n",
                encoding="utf-8",
            )
            config = TwinrConfig.from_env(env_path)
            loop, lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(config=config)

            result = loop._handle_update_simple_setting_tool_call(
                {
                    "setting": "memory_capacity",
                    "action": "increase",
                }
            )

            env_text = env_path.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "updated")
        self.assertEqual(result["memory_max_turns"], 28)
        self.assertEqual(result["memory_keep_recent"], 12)
        self.assertEqual(loop.config.memory_max_turns, 28)
        self.assertEqual(loop.runtime.memory.max_turns, 28)
        self.assertIn("TWINR_MEMORY_MAX_TURNS=28", env_text)
        self.assertIn("TWINR_MEMORY_KEEP_RECENT=12", env_text)
        self.assertEqual(loop.runtime.memory.ledger[-1].kind, "preference")
        self.assertIn("simple_setting_tool_call=true", lines)

    def test_update_simple_setting_requires_confirmation_for_unknown_voice(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "TWINR_MEMORY_MAX_TURNS=20\nTWINR_MEMORY_KEEP_RECENT=10\n",
                encoding="utf-8",
            )
            config = TwinrConfig.from_env(env_path)
            loop, _lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(config=config)
            loop.runtime.update_user_voice_assessment(
                status="unknown_voice",
                confidence=0.18,
                checked_at="2026-03-13T21:00:00+00:00",
            )

            with self.assertRaisesRegex(RuntimeError, "confirmed=true"):
                loop._handle_update_simple_setting_tool_call(
                    {
                        "setting": "memory_capacity",
                        "action": "increase",
                    }
                )

            result = loop._handle_update_simple_setting_tool_call(
                {
                    "setting": "memory_capacity",
                    "action": "increase",
                    "confirmed": True,
                }
            )

        self.assertEqual(result["status"], "updated")

    def test_update_simple_setting_can_change_voice_and_speed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "OPENAI_TTS_VOICE=marin",
                        "OPENAI_REALTIME_VOICE=sage",
                        "OPENAI_TTS_SPEED=1.00",
                        "OPENAI_REALTIME_SPEED=1.00",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config = TwinrConfig.from_env(env_path)
            loop, _lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(config=config)

            voice_result = loop._handle_update_simple_setting_tool_call(
                {
                    "setting": "spoken_voice",
                    "action": "set",
                    "value": "männliche Stimme",
                }
            )
            speed_result = loop._handle_update_simple_setting_tool_call(
                {
                    "setting": "speech_speed",
                    "action": "decrease",
                }
            )

            env_text = env_path.read_text(encoding="utf-8")

        self.assertEqual(voice_result["status"], "updated")
        self.assertEqual(voice_result["voice"], "cedar")
        self.assertEqual(loop.config.openai_tts_voice, "cedar")
        self.assertEqual(loop.config.openai_realtime_voice, "cedar")
        self.assertEqual(speed_result["status"], "updated")
        self.assertEqual(speed_result["speech_speed"], 0.9)
        self.assertEqual(loop.config.openai_tts_speed, 0.9)
        self.assertEqual(loop.config.openai_realtime_speed, 0.9)
        self.assertIn("OPENAI_TTS_VOICE=cedar", env_text)
        self.assertIn("OPENAI_REALTIME_VOICE=cedar", env_text)
        self.assertIn("OPENAI_TTS_SPEED=0.90", env_text)
        self.assertIn("OPENAI_REALTIME_SPEED=0.90", env_text)

    def test_voice_profile_tools_can_enroll_status_and_reset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
                voice_profile_store_path=str(Path(temp_dir) / "state" / "voice_profile.json"),
            )
            loop, lines, _realtime_session, _print_backend, _recorder, _player, _printer = self.make_loop(config=config)
            loop._current_turn_audio_pcm = _voice_sample_pcm_bytes()

            enroll = loop._handle_enroll_voice_profile_tool_call({})
            status = loop._handle_get_voice_profile_status_tool_call({})
            reset = loop._handle_reset_voice_profile_tool_call({})

        self.assertEqual(enroll["status"], "enrolled")
        self.assertGreaterEqual(enroll["sample_count"], 1)
        self.assertEqual(status["status"], "ok")
        self.assertTrue(status["enrolled"])
        self.assertEqual(status["current_signal"], "likely_user")
        self.assertEqual(reset["status"], "reset")
        self.assertFalse(reset["enrolled"])
        self.assertIn("voice_profile_tool_call=true", lines)

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
        self._assert_language_contract_only(conversation)
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

        self.assertEqual(len(print_backend.search_calls), 1)
        question, conversation, location_hint, date_context = print_backend.search_calls[0]
        self.assertEqual(question, "Wann faehrt der Bus nach Hamburg?")
        self._assert_language_contract_only(conversation)
        self.assertEqual(location_hint, "Schwarzenbek")
        self.assertEqual(date_context, "Friday, 2026-03-13 10:00 (Europe/Berlin)")
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
        self.assertEqual(len(print_backend.proactive_calls), 1)
        self.assertEqual(loop.runtime.last_transcript, "Hallo Twinr")
        self.assertEqual(loop.runtime.last_response, "Guten Tag")
        proactive_call = print_backend.proactive_calls[0]
        self.assertEqual(
            proactive_call[:5],
            (
                "person_returned",
                "Person returned after a long absence.",
                "Hey Thom, schön dich zu sehen. Wie geht's dir?",
                int(SocialTriggerPriority.PERSON_RETURNED),
                (),
            ),
        )
        self.assertIsInstance(proactive_call[5], tuple)
        self.assertIsInstance(proactive_call[6], tuple)
        self.assertEqual(print_backend.synthesize_calls, ["Proaktiv: person_returned"])
        self.assertEqual(len(player.tones), 1)
        self.assertEqual(player.played, [b"PCM", b"PCM"])
        self.assertIn("social_trigger=person_returned", lines)
        self.assertIn("social_prompt_mode=llm", lines)
        self.assertIn("proactive_listen=true", lines)
        self.assertIn("status=listening", lines)
        self.assertIn("social_response_id=resp_social_1", lines)
        self.assertTrue(any(line.startswith("timing_social_tts_ms=") for line in lines))
        social_events = [
            entry
            for entry in loop.runtime.ops_events.tail(limit=40)
            if entry["event"] == "social_trigger_prompted"
            and entry.get("data", {}).get("trigger") == "person_returned"
        ]
        self.assertTrue(social_events)
        self.assertEqual(social_events[-1]["data"]["prompt"], "Proaktiv: person_returned")
        self.assertEqual(
            social_events[-1]["data"]["default_prompt"],
            "Hey Thom, schön dich zu sehen. Wie geht's dir?",
        )
        self.assertEqual(social_events[-1]["data"]["prompt_mode"], "llm")

    def test_social_trigger_opens_hands_free_listening_window_and_times_out(self) -> None:
        recorder = FakeRecorder(
            recordings=[
                RuntimeError("No speech detected before timeout"),
            ]
        )
        loop, lines, realtime_session, print_backend, recorder, player, _printer = self.make_loop(
            config=TwinrConfig(
                conversation_follow_up_timeout_s=3.5,
                audio_follow_up_speech_start_chunks=5,
                audio_follow_up_ignore_ms=420,
            ),
            recorder=recorder,
        )

        spoke = loop.handle_social_trigger(
            SocialTriggerDecision(
                trigger_id="attention_window",
                prompt="Kann ich dir bei etwas helfen?",
                reason="Person was visible and quiet.",
                observed_at=42.0,
                priority=SocialTriggerPriority.ATTENTION_WINDOW,
            )
        )

        self.assertTrue(spoke)
        self.assertEqual(realtime_session.calls, [])
        self.assertEqual(recorder.pause_values, [1200])
        self.assertEqual(recorder.start_timeouts, [3.5])
        self.assertEqual(recorder.speech_start_chunks, [5])
        self.assertEqual(recorder.ignore_initial_ms, [420])
        self.assertEqual(len(print_backend.proactive_calls), 1)
        self.assertEqual(len(player.tones), 1)
        self.assertEqual(player.played, [b"PCM"])
        self.assertIn("proactive_listen_timeout=true", lines)
        self.assertEqual(loop.runtime.status.value, "waiting")

    def test_social_trigger_uses_direct_prompt_for_safety_events(self) -> None:
        loop, lines, _realtime_session, print_backend, _recorder, player, _printer = self.make_loop()

        spoke = loop.handle_social_trigger(
            SocialTriggerDecision(
                trigger_id="possible_fall",
                prompt="Brauchst du Hilfe?",
                reason="Person disappeared after a fall-like transition.",
                observed_at=42.0,
                priority=SocialTriggerPriority.POSSIBLE_FALL,
            )
        )

        self.assertTrue(spoke)
        self.assertEqual(print_backend.proactive_calls, [])
        self.assertEqual(print_backend.synthesize_calls, ["Brauchst du Hilfe?"])
        self.assertIn("social_prompt_mode=direct_safety", lines)
        social_events = [
            entry
            for entry in loop.runtime.ops_events.tail(limit=40)
            if entry["event"] == "social_trigger_prompted"
            and entry.get("data", {}).get("trigger") == "possible_fall"
        ]
        self.assertTrue(social_events)
        self.assertEqual(social_events[-1]["data"]["prompt"], "Brauchst du Hilfe?")
        self.assertEqual(social_events[-1]["data"]["prompt_mode"], "direct_safety")

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
        social_events = [entry for entry in loop.runtime.ops_events.tail(limit=20) if entry["event"] == "social_trigger_skipped"]
        self.assertEqual(len(social_events), 1)
        self.assertEqual(social_events[0]["data"]["prompt"], "Schön, dich zu sehen. Was möchtest du machen?")

    def test_social_trigger_is_skipped_while_follow_up_session_is_active(self) -> None:
        loop, lines, _realtime_session, _print_backend, _recorder, player, _printer = self.make_loop()
        loop._conversation_session_active = True

        spoke = loop.handle_social_trigger(
            SocialTriggerDecision(
                trigger_id="showing_intent",
                prompt="Möchtest du mir etwas zeigen?",
                reason="Object near the camera.",
                observed_at=42.0,
                priority=SocialTriggerPriority.SHOWING_INTENT,
            )
        )

        self.assertFalse(spoke)
        self.assertEqual(player.played, [])
        self.assertIn("social_trigger_skipped=conversation_active", lines)
        social_events = [entry for entry in loop.runtime.ops_events.tail(limit=20) if entry["event"] == "social_trigger_skipped"]
        self.assertGreaterEqual(len(social_events), 1)
        self.assertEqual(social_events[-1]["data"]["skip_reason"], "conversation_active")

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
