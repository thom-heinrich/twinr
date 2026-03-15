from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.agent.base_agent.contracts import (
    AgentToolCall,
    AgentToolResult,
    StreamingSpeechEndpointEvent,
    StreamingTranscriptionResult,
    ToolCallingTurnResponse,
)
from twinr.hardware import VoiceAssessment
from twinr.memory.longterm import LongTermConsolidationResultV1, LongTermMemoryObjectV1, LongTermSourceRefV1
from twinr.memory.reminders import now_in_timezone
from twinr.proactive import SocialTriggerDecision, SocialTriggerPriority
from twinr.providers.openai import OpenAIImageInput, OpenAITextResponse
from twinr.runner import TwinrHardwareLoop
from twinr.runtime import TwinrRuntime


def _system_messages(conversation: tuple[tuple[str, str], ...] | None) -> tuple[str, ...]:
    return tuple(content for role, content in (conversation or ()) if role == "system")


def _assert_contains_system_message(testcase: unittest.TestCase, conversation, needle: str) -> None:
    testcase.assertTrue(
        any(needle in content for content in _system_messages(conversation)),
        msg=f"Expected a system message containing {needle!r}, got {conversation!r}",
    )


def _longterm_source(event_id: str) -> LongTermSourceRefV1:
    return LongTermSourceRefV1(
        source_type="conversation_turn",
        event_ids=(event_id,),
        speaker="user",
        modality="voice",
    )


def _fresh_checked_at() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class FakeBackend:
    def __init__(self) -> None:
        self.config = TwinrConfig()
        self.transcribe_calls: list[tuple[bytes, str, str]] = []
        self.respond_calls: list[tuple[str, tuple[tuple[str, str], ...] | None, bool | None]] = []
        self.respond_to_images_calls: list[
            tuple[str, list[OpenAIImageInput], tuple[tuple[str, str], ...] | None, bool | None]
        ] = []
        self.search_calls: list[tuple[str, tuple[tuple[str, str], ...] | None, str | None, str | None]] = []
        self.synthesize_calls: list[str] = []
        self.print_calls: list[tuple[tuple[tuple[str, str], ...] | None, str | None, str | None, str]] = []
        self.reminder_calls: list[object] = []
        self.proactive_calls: list[tuple[str, str, str, int, tuple[tuple[str, str], ...] | None, tuple[str, ...], tuple[str, ...]]] = []
        self.transcript = "Hello Twinr"
        self.answer = "Hello back."
        self.used_web_search = False

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
            used_web_search=self.used_web_search,
        )

    def respond_with_metadata(
        self,
        prompt: str,
        *,
        conversation=None,
        instructions=None,
        allow_web_search=None,
    ) -> OpenAITextResponse:
        del instructions
        self.respond_calls.append((prompt, conversation, allow_web_search))
        return OpenAITextResponse(
            text=self.answer,
            response_id="resp_123",
            request_id="req_123",
            used_web_search=self.used_web_search,
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
        self.respond_to_images_calls.append((prompt, list(images), conversation, allow_web_search))
        return OpenAITextResponse(
            text=self.answer,
            response_id="resp_vision_123",
            request_id="req_vision_123",
            used_web_search=False,
        )

    def search_live_info_with_metadata(
        self,
        question: str,
        *,
        conversation=None,
        location_hint: str | None = None,
        date_context: str | None = None,
    ):
        self.search_calls.append((question, conversation, location_hint, date_context))
        return SimpleNamespace(
            answer=self.answer,
            sources=("https://example.com/result",),
            response_id="resp_search_123",
            request_id="req_search_123",
            model="gpt-5.2",
            token_usage=None,
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

    def phrase_due_reminder_with_metadata(self, reminder) -> OpenAITextResponse:
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
        del delivery
        self.respond_calls.append((prompt, None, allow_web_search))
        return OpenAITextResponse(
            text=self.answer,
            response_id="resp_auto_1",
            request_id="req_auto_1",
            model="gpt-5.2",
            token_usage=None,
            used_web_search=allow_web_search,
        )


class FakeRecorder:
    def __init__(self, recordings: list[bytes | Exception] | None = None) -> None:
        self.pause_values: list[int] = []
        self.start_timeouts: list[float | None] = []
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
        on_chunk=None,
        should_stop=None,
    ):
        del max_record_seconds, speech_start_chunks, ignore_initial_ms
        self.pause_values.append(pause_ms)
        self.start_timeouts.append(start_timeout_s)
        self.pause_grace_values.append(pause_grace_ms)
        if on_chunk is not None:
            on_chunk(b"PCM-A")
            on_chunk(b"PCM-B")
        if should_stop is not None:
            for _ in range(50):
                if should_stop():
                    break
                time.sleep(0.001)
        if not self.recordings:
            value: bytes | Exception = b"PCMINPUT"
        else:
            value = self.recordings.pop(0)
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(
            pcm_bytes=value,
            speech_started_after_ms=2300,
            resumed_after_pause_count=0,
        )


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


class FakeToolCallingProvider:
    def __init__(self, *, tool_name: str, arguments: dict[str, object], final_text: str = "Hello back.") -> None:
        self.tool_name = tool_name
        self.arguments = dict(arguments)
        self.final_text = final_text
        self.start_calls: list[tuple[str, tuple[tuple[str, str], ...] | None, bool | None]] = []
        self.continue_calls: list[tuple[str, tuple[AgentToolResult, ...], bool | None]] = []

    def start_turn_streaming(
        self,
        prompt: str,
        *,
        conversation=None,
        instructions=None,
        tool_schemas=(),
        allow_web_search=None,
        on_text_delta=None,
    ) -> ToolCallingTurnResponse:
        del instructions, tool_schemas, on_text_delta
        self.start_calls.append((prompt, conversation, allow_web_search))
        return ToolCallingTurnResponse(
            text="",
            tool_calls=(
                AgentToolCall(
                    name=self.tool_name,
                    call_id="call_1",
                    arguments=dict(self.arguments),
                ),
            ),
            response_id="resp_tool_start",
            request_id="req_tool_start",
            model="gpt-5.2",
            token_usage=None,
            used_web_search=False,
            continuation_token="resp_tool_start",
        )

    def continue_turn_streaming(
        self,
        *,
        continuation_token: str,
        tool_results,
        instructions=None,
        tool_schemas=(),
        allow_web_search=None,
        on_text_delta=None,
    ) -> ToolCallingTurnResponse:
        del instructions, tool_schemas
        self.continue_calls.append((continuation_token, tuple(tool_results), allow_web_search))
        if on_text_delta is not None:
            for chunk in ("Hello ", "back."):
                on_text_delta(chunk)
        return ToolCallingTurnResponse(
            text=self.final_text,
            tool_calls=(),
            response_id="resp_tool_finish",
            request_id="req_tool_finish",
            model="gpt-5.2",
            token_usage=None,
            used_web_search=False,
            continuation_token="resp_tool_finish",
        )


class FakeTurnStreamingSpeechSession:
    def __init__(self) -> None:
        self.sent: list[bytes] = []
        self.closed = False
        self._on_endpoint = None
        self.finalize_calls = 0

    def send_pcm(self, pcm_bytes: bytes) -> None:
        self.sent.append(pcm_bytes)
        if len(self.sent) == 2 and self._on_endpoint is not None:
            self._on_endpoint(
                StreamingSpeechEndpointEvent(
                    transcript="ich bin immernoch am programmieren, nur damit du es weisst",
                    event_type="speech_final",
                    speech_final=True,
                )
            )

    def snapshot(self) -> StreamingTranscriptionResult:
        return StreamingTranscriptionResult(
            transcript="ich bin immernoch am programmieren, nur damit du es weisst",
            request_id="turn-stt-1",
            saw_interim=True,
            saw_speech_final=True,
            saw_utterance_end=False,
        )

    def finalize(self) -> StreamingTranscriptionResult:
        self.finalize_calls += 1
        return StreamingTranscriptionResult(
            transcript="ich bin immernoch am programmieren, nur damit du es weisst",
            request_id="turn-stt-1",
            saw_interim=True,
            saw_speech_final=True,
            saw_utterance_end=False,
        )

    def close(self) -> None:
        self.closed = True


class FakeTurnStreamingSpeechToTextProvider:
    def __init__(self, config: TwinrConfig) -> None:
        self.config = config
        self.session = FakeTurnStreamingSpeechSession()
        self.start_calls: list[dict[str, object]] = []

    def transcribe(self, audio_bytes: bytes, **kwargs) -> str:
        del audio_bytes, kwargs
        return "ich bin immernoch am programmieren, nur damit du es weisst"

    def transcribe_path(self, path, **kwargs) -> str:
        del path, kwargs
        return "ich bin immernoch am programmieren, nur damit du es weisst"

    def start_streaming_session(
        self,
        *,
        sample_rate: int,
        channels: int,
        language: str | None = None,
        prompt: str | None = None,
        on_interim=None,
        on_endpoint=None,
    ):
        del prompt
        self.start_calls.append(
            {
                "sample_rate": sample_rate,
                "channels": channels,
                "language": language,
            }
        )
        self.session._on_endpoint = on_endpoint
        if on_interim is not None:
            on_interim("ich bin immernoch")
        return self.session


class FakeTurnToolAgentProvider:
    def __init__(self, config: TwinrConfig) -> None:
        self.config = config
        self.start_calls: list[dict[str, object]] = []

    def start_turn_streaming(
        self,
        prompt: str,
        *,
        conversation=None,
        instructions=None,
        tool_schemas=(),
        allow_web_search=None,
        on_text_delta=None,
    ) -> ToolCallingTurnResponse:
        del on_text_delta
        self.start_calls.append(
            {
                "prompt": prompt,
                "conversation": conversation,
                "instructions": instructions,
                "tool_schemas": list(tool_schemas),
                "allow_web_search": allow_web_search,
            }
        )
        return ToolCallingTurnResponse(
            text="",
            tool_calls=(
                AgentToolCall(
                    name="submit_turn_decision",
                    call_id="call_turn_1",
                    arguments={
                        "decision": "end_turn",
                        "confidence": 0.94,
                        "reason": "complete_request",
                        "transcript": "ich bin immernoch am programmieren, nur damit du es weisst",
                    },
                ),
            ),
            response_id="resp_turn_controller_1",
            request_id="req_turn_controller_1",
            model="gpt-5.2",
            token_usage=None,
            used_web_search=False,
            continuation_token="resp_turn_controller_1",
        )


class HardwareLoopTests(unittest.TestCase):
    def make_loop(
        self,
        *,
        backend=None,
        stt_provider=None,
        agent_provider=None,
        tts_provider=None,
        tool_agent_provider=None,
        turn_stt_provider=None,
        turn_tool_agent_provider=None,
        config: TwinrConfig | None = None,
        recorder: FakeRecorder | None = None,
        camera=None,
        button_monitor=None,
        voice_profile_monitor=None,
        proactive_monitor=None,
    ) -> tuple[TwinrHardwareLoop, list[str], FakeRecorder, FakePlayer, FakePrinter]:
        lines: list[str] = []
        recorder = recorder or FakeRecorder()
        player = FakePlayer()
        printer = FakePrinter()
        temp_dir_handle = tempfile.TemporaryDirectory()
        temp_root = Path(temp_dir_handle.name)
        config = config or TwinrConfig()
        original_project_root = Path(config.project_root).resolve()
        sandbox_project_default = config.project_root == "."
        config = replace(
            config,
            project_root=str(temp_root if config.project_root == "." else Path(config.project_root)),
            runtime_state_path=self._sandbox_path(
                config.runtime_state_path,
                temp_root / "runtime-state.json",
                default="/tmp/twinr-runtime-state.json",
                project_root=original_project_root,
                sandbox_project_default=sandbox_project_default,
            ),
            reminder_store_path=self._sandbox_path(
                config.reminder_store_path,
                temp_root / "state" / "reminders.json",
                default="state/reminders.json",
                project_root=original_project_root,
                sandbox_project_default=sandbox_project_default,
            ),
            memory_markdown_path=self._sandbox_path(
                config.memory_markdown_path,
                temp_root / "state" / "MEMORY.md",
                default="state/MEMORY.md",
                project_root=original_project_root,
                sandbox_project_default=sandbox_project_default,
            ),
            adaptive_timing_store_path=self._sandbox_path(
                config.adaptive_timing_store_path,
                temp_root / "state" / "adaptive-timing.json",
                default="state/adaptive_timing.json",
                project_root=original_project_root,
                sandbox_project_default=sandbox_project_default,
            ),
            long_term_memory_path=self._sandbox_path(
                config.long_term_memory_path,
                temp_root / "state" / "chonkydb",
                default="state/chonkydb",
                project_root=original_project_root,
                sandbox_project_default=sandbox_project_default,
            ),
        )
        resolved_backend = (
            backend
            if backend is not None or any(provider is not None for provider in (stt_provider, agent_provider, tts_provider))
            else FakeBackend()
        )
        loop = TwinrHardwareLoop(
            config=config,
            runtime=TwinrRuntime(config=config),
            backend=resolved_backend,
            stt_provider=stt_provider,
            agent_provider=agent_provider,
            tts_provider=tts_provider,
            tool_agent_provider=tool_agent_provider,
            turn_stt_provider=turn_stt_provider,
            turn_tool_agent_provider=turn_tool_agent_provider,
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
        return loop, lines, recorder, player, printer

    @staticmethod
    def _sandbox_path(
        current: str,
        isolated: Path,
        *,
        default: str,
        project_root: Path,
        sandbox_project_default: bool,
    ) -> str:
        path = Path(current)
        default_path = Path(default)
        project_default = project_root / default_path if not default_path.is_absolute() else default_path
        if current == default or not path.is_absolute() or (sandbox_project_default and path == project_default):
            return str(isolated)
        return current

    def test_green_button_runs_full_audio_turn(self) -> None:
        backend = FakeBackend()
        loop, lines, recorder, player, _printer = self.make_loop(backend=backend)

        loop.handle_button_press("green")

        self.assertEqual(recorder.pause_values, [1200])
        self.assertEqual(recorder.start_timeouts, [8.0])
        self.assertEqual(recorder.pause_grace_values, [450])
        self.assertTrue(backend.transcribe_calls[0][0].startswith(b"RIFF"))
        self.assertEqual(backend.transcribe_calls[0][1], "twinr-listen.wav")
        self.assertEqual(backend.respond_calls[0][0], "Hello Twinr")
        self.assertEqual(backend.respond_to_images_calls, [])
        self.assertIsNone(backend.respond_calls[0][2])
        self.assertEqual(player.played, [b"RIFF"])
        self.assertEqual(loop.runtime.last_response, "Hello back.")
        self.assertIn("status=listening", lines)

    def test_green_button_accepts_split_providers(self) -> None:
        backend = FakeBackend()
        loop, lines, _recorder, player, _printer = self.make_loop(
            backend=None,
            stt_provider=backend,
            agent_provider=backend,
            tts_provider=backend,
        )

        loop.handle_button_press("green")

        self.assertEqual(len(backend.transcribe_calls), 1)
        self.assertEqual(len(backend.respond_calls), 1)
        self.assertEqual(player.played, [b"RIFF"])
        self.assertIn("status=processing", lines)
        self.assertIn("status=answering", lines)
        self.assertIn("status=waiting", lines)
        self.assertIn("timing_playback_ms=streamed", lines)

    def test_green_button_uses_processing_feedback_before_answering(self) -> None:
        backend = FakeBackend()
        loop, _lines, _recorder, _player, _printer = self.make_loop(backend=backend)
        feedback_kinds: list[str] = []

        def fake_start(kind: str):
            feedback_kinds.append(kind)
            return lambda: None

        loop._start_working_feedback_loop = fake_start  # type: ignore[method-assign]

        loop.handle_button_press("green")

        self.assertEqual(feedback_kinds[0], "processing")

    def test_green_button_can_end_turn_from_streaming_turn_controller(self) -> None:
        config = TwinrConfig(
            turn_controller_enabled=True,
            turn_controller_fast_endpoint_enabled=False,
            openai_api_key="test-key",
            project_root=".",
            personality_dir="personality",
        )
        backend = FakeBackend()
        turn_stt_provider = FakeTurnStreamingSpeechToTextProvider(config)
        turn_tool_agent_provider = FakeTurnToolAgentProvider(config)
        loop, lines, _recorder, _player, _printer = self.make_loop(
            backend=backend,
            config=config,
            turn_stt_provider=turn_stt_provider,
            turn_tool_agent_provider=turn_tool_agent_provider,
        )

        loop.handle_button_press("green")

        self.assertEqual(turn_stt_provider.start_calls[0]["sample_rate"], config.audio_sample_rate)
        self.assertEqual(turn_stt_provider.session.sent, [b"PCM-A", b"PCM-B"])
        self.assertTrue(turn_stt_provider.session.closed)
        self.assertEqual(len(turn_tool_agent_provider.start_calls), 1)
        self.assertEqual(backend.transcribe_calls, [])
        self.assertIn("turn_controller_candidate=speech_final", lines)
        self.assertIn("turn_controller_decision=end_turn", lines)
        self.assertIn("stt_partial=ich bin immernoch", lines)
        self.assertIn("timing_stt_ms=", "\n".join(lines))

    def test_green_button_recovers_when_streaming_stt_hears_speech_before_local_threshold(self) -> None:
        config = TwinrConfig(
            turn_controller_enabled=True,
            turn_controller_fast_endpoint_enabled=False,
            openai_api_key="test-key",
            project_root=".",
            personality_dir="personality",
        )
        backend = FakeBackend()
        recorder = FakeRecorder(recordings=[RuntimeError("No speech detected before timeout")])
        turn_stt_provider = FakeTurnStreamingSpeechToTextProvider(config)
        turn_tool_agent_provider = FakeTurnToolAgentProvider(config)
        loop, lines, _recorder, _player, _printer = self.make_loop(
            backend=backend,
            config=config,
            recorder=recorder,
            turn_stt_provider=turn_stt_provider,
            turn_tool_agent_provider=turn_tool_agent_provider,
        )

        loop.handle_button_press("green")

        self.assertIn("turn_controller_capture_recovered=true", lines)
        self.assertEqual(backend.transcribe_calls, [])
        self.assertEqual(backend.respond_calls[0][0], "ich bin immernoch am programmieren, nur damit du es weisst")
        self.assertEqual(loop.runtime.last_response, "Hello back.")

    def test_green_button_no_speech_timeout_returns_to_waiting(self) -> None:
        backend = FakeBackend()
        recorder = FakeRecorder(recordings=[RuntimeError("No speech detected before timeout")])
        loop, lines, recorder, player, _printer = self.make_loop(
            backend=backend,
            config=TwinrConfig(),
            recorder=recorder,
        )

        loop.handle_button_press("green")

        self.assertEqual(recorder.pause_values, [1200])
        self.assertEqual(backend.transcribe_calls, [])
        self.assertEqual(player.played, [])
        self.assertIn("listen_timeout=true", lines)
        self.assertEqual(loop.runtime.status.value, "waiting")

    def test_yellow_button_formats_and_prints_last_answer(self) -> None:
        backend = FakeBackend()
        loop, lines, _recorder, _player, printer = self.make_loop(backend=backend)
        loop.runtime.last_response = "Hello back"

        loop.handle_button_press("yellow")

        self.assertEqual(len(backend.print_calls), 1)
        conversation, focus_hint, direct_text, request_source = backend.print_calls[0]
        self.assertEqual(focus_hint, None)
        self.assertEqual(direct_text, "Hello back")
        self.assertEqual(request_source, "button")
        _assert_contains_system_message(self, conversation, "All user-facing spoken and written replies")
        self.assertEqual(printer.printed, ["HELLO BACK"])
        self.assertIn("status=printing", lines)
        self.assertEqual(lines[-1], "status=waiting")

    def test_idle_loop_delivers_due_reminder(self) -> None:
        backend = FakeBackend()
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                reminder_store_path=str(Path(temp_dir) / "state" / "reminders.json"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
                reminder_poll_interval_s=0.0,
            )
            loop, lines, _recorder, player, _printer = self.make_loop(backend=backend, config=config)
            loop.runtime.schedule_reminder(
                due_at=now_in_timezone(config.local_timezone_name).isoformat(),
                summary="Tabletten nehmen",
                kind="medication",
                source="test",
            )

            delivered = loop._maybe_deliver_due_reminder()
            stored_entries = loop.runtime.reminder_store.load_entries()

        self.assertTrue(delivered)
        self.assertEqual(len(backend.reminder_calls), 1)
        self.assertEqual(backend.synthesize_calls, ["Erinnerung: Tabletten nehmen"])
        self.assertEqual(player.played, [b"RIFF"])
        self.assertIn("reminder_delivered=true", lines)
        self.assertTrue(stored_entries[0].delivered)

    def test_social_trigger_is_governor_blocked_after_recent_reminder(self) -> None:
        backend = FakeBackend()
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                reminder_store_path=str(Path(temp_dir) / "state" / "reminders.json"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
                reminder_poll_interval_s=0.0,
                proactive_governor_global_prompt_cooldown_s=300.0,
                proactive_governor_source_repeat_cooldown_s=0.0,
            )
            loop, lines, _recorder, player, _printer = self.make_loop(backend=backend, config=config)
            loop.runtime.schedule_reminder(
                due_at=now_in_timezone(config.local_timezone_name).isoformat(),
                summary="Tabletten nehmen",
                kind="medication",
                source="test",
            )

            delivered = loop._maybe_deliver_due_reminder()
            spoke = loop.handle_social_trigger(
                SocialTriggerDecision(
                    trigger_id="attention_window",
                    prompt="Soll ich dir helfen?",
                    reason="User seems attentive and quiet.",
                    observed_at=12.0,
                    priority=SocialTriggerPriority.ATTENTION_WINDOW,
                    score=0.92,
                    threshold=0.86,
                    evidence=(),
                )
            )

        self.assertTrue(delivered)
        self.assertFalse(spoke)
        self.assertEqual(player.played, [b"RIFF"])
        self.assertIn("social_trigger_skipped=governor_global_prompt_cooldown_active", lines)

    def test_idle_loop_delivers_longterm_proactive_candidate(self) -> None:
        backend = FakeBackend()
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                long_term_memory_enabled=True,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                long_term_memory_proactive_enabled=True,
                long_term_memory_proactive_poll_interval_s=0.0,
                long_term_memory_proactive_min_confidence=0.7,
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            loop, lines, _recorder, player, _printer = self.make_loop(backend=backend, config=config)
            loop.runtime.long_term_memory.object_store.apply_consolidation(
                LongTermConsolidationResultV1(
                    turn_id="turn:thread",
                    occurred_at=now_in_timezone(config.local_timezone_name),
                    episodic_objects=(),
                    durable_objects=(
                        LongTermMemoryObjectV1(
                            memory_id="thread:walk_weather",
                            kind="thread_summary",
                            summary="Ongoing thread about the user's plan to walk if the weather is nice.",
                            details="Reflected from multiple related turns.",
                            source=_longterm_source("turn:thread"),
                            status="active",
                            confidence=0.82,
                            sensitivity="normal",
                            slot_key="thread:user:main:walk_weather",
                            value_key="walk_weather",
                            attributes={"support_count": 4},
                        ),
                    ),
                    deferred_objects=(),
                    conflicts=(),
                    graph_edges=(),
                )
            )

            delivered = loop._maybe_run_long_term_memory_proactive()
            history = loop.runtime.long_term_memory.proactive_policy.state_store.load_entries()

        self.assertTrue(delivered)
        self.assertEqual(len(backend.proactive_calls), 1)
        self.assertEqual(backend.proactive_calls[0][0], "longterm:candidate:thread_walk_weather:followup")
        self.assertEqual(player.played, [b"RIFF"])
        self.assertIn("longterm_proactive_candidate=candidate:thread_walk_weather:followup", lines)
        self.assertIn("longterm_proactive_prompt_mode=llm", lines)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].delivery_count, 1)

    def test_longterm_proactive_is_governor_blocked_after_recent_social_prompt(self) -> None:
        backend = FakeBackend()
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                long_term_memory_enabled=True,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                long_term_memory_proactive_enabled=True,
                long_term_memory_proactive_poll_interval_s=0.0,
                long_term_memory_proactive_min_confidence=0.7,
                proactive_governor_global_prompt_cooldown_s=300.0,
                proactive_governor_source_repeat_cooldown_s=0.0,
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            loop, lines, _recorder, player, _printer = self.make_loop(backend=backend, config=config)
            loop.runtime.long_term_memory.object_store.apply_consolidation(
                LongTermConsolidationResultV1(
                    turn_id="turn:thread",
                    occurred_at=now_in_timezone(config.local_timezone_name),
                    episodic_objects=(),
                    durable_objects=(
                        LongTermMemoryObjectV1(
                            memory_id="thread:walk_weather",
                            kind="thread_summary",
                            summary="Ongoing thread about the user's plan to walk if the weather is nice.",
                            details="Reflected from multiple related turns.",
                            source=_longterm_source("turn:thread"),
                            status="active",
                            confidence=0.82,
                            sensitivity="normal",
                            slot_key="thread:user:main:walk_weather",
                            value_key="walk_weather",
                            attributes={"support_count": 4},
                        ),
                    ),
                    deferred_objects=(),
                    conflicts=(),
                    graph_edges=(),
                )
            )

            spoke = loop.handle_social_trigger(
                SocialTriggerDecision(
                    trigger_id="attention_window",
                    prompt="Soll ich dir helfen?",
                    reason="User seems attentive and quiet.",
                    observed_at=12.0,
                    priority=SocialTriggerPriority.ATTENTION_WINDOW,
                    score=0.92,
                    threshold=0.86,
                    evidence=(),
                )
            )
            delivered = loop._maybe_run_long_term_memory_proactive()
            history = loop.runtime.long_term_memory.proactive_policy.state_store.load_entries()

        self.assertTrue(spoke)
        self.assertFalse(delivered)
        self.assertEqual(player.played, [b"RIFF"])
        self.assertIn("longterm_proactive_skipped=governor_global_prompt_cooldown_active", lines)
        self.assertEqual(len(history), 0)

    def test_errors_reset_runtime_to_waiting(self) -> None:
        loop, lines, _recorder, _player, _printer = self.make_loop(backend=FakeBackend())

        loop.handle_button_press("yellow")

        self.assertIn("status=error", lines)
        self.assertTrue(any(line.startswith("error=") for line in lines))
        self.assertEqual(loop.runtime.status.value, "waiting")

    def test_auto_web_search_defers_to_backend_default_config(self) -> None:
        backend = FakeBackend()
        backend.used_web_search = True
        backend.transcript = "What is the weather today in Berlin?"
        loop, _lines, _recorder, _player, _printer = self.make_loop(
            backend=backend,
            config=TwinrConfig(openai_enable_web_search=True),
        )

        loop.handle_button_press("green")

        self.assertIsNone(backend.respond_calls[0][2])
        self.assertEqual(len(loop.runtime.memory.search_results), 1)
        self.assertEqual(loop.runtime.memory.search_results[0].question, "What is the weather today in Berlin?")
        self.assertEqual(loop.runtime.memory.search_results[0].answer, "Hello back.")

    def test_visual_queries_use_camera_and_multimodal_request(self) -> None:
        backend = FakeBackend()
        backend.transcript = "Schau mich mal an"
        camera = FakeCamera()
        tool_agent = FakeToolCallingProvider(
            tool_name="inspect_camera",
            arguments={"question": "Schau mich mal an"},
        )
        loop, lines, _recorder, player, _printer = self.make_loop(
            backend=backend,
            camera=camera,
            tool_agent_provider=tool_agent,
        )

        loop.handle_button_press("green")

        self.assertEqual(camera.capture_calls, 1)
        self.assertEqual(tool_agent.start_calls[0][0], "Schau mich mal an")
        self.assertEqual(len(backend.respond_to_images_calls), 1)
        prompt, images, conversation, allow_web_search = backend.respond_to_images_calls[0]
        self.assertIn("Image 1 is the current live camera frame", prompt)
        _assert_contains_system_message(self, conversation, "All user-facing spoken and written replies")
        self.assertFalse(allow_web_search)
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].label, "Image 1: live camera frame from the device.")
        self.assertEqual(player.played, [b"RIFF"])
        self.assertIn("camera_tool_call=true", lines)
        self.assertIn("vision_image_count=1", lines)

    def test_visual_queries_attach_reference_image_when_configured(self) -> None:
        backend = FakeBackend()
        backend.transcript = "Wie sehe ich heute aus?"
        camera = FakeCamera()
        tool_agent = FakeToolCallingProvider(
            tool_name="inspect_camera",
            arguments={"question": "Wie sehe ich heute aus?"},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            reference_path = Path(temp_dir) / "user-reference.jpg"
            reference_path.write_bytes(b"\xff\xd8\xffreference")
            config = TwinrConfig(
                vision_reference_image_path=str(reference_path),
            )
            loop, lines, _recorder, _player, _printer = self.make_loop(
                backend=backend,
                config=config,
                camera=camera,
                tool_agent_provider=tool_agent,
            )

            loop.handle_button_press("green")

        self.assertEqual(len(backend.respond_to_images_calls), 1)
        prompt, images, _conversation, _allow_web_search = backend.respond_to_images_calls[0]
        self.assertIn("Image 2 is a stored reference image of the main user.", prompt)
        self.assertEqual(len(images), 2)
        self.assertEqual(images[1].label, "Image 2: stored reference image of the main user. Use it only for person or identity comparison.")
        self.assertTrue(any(line.startswith("vision_reference_image=") for line in lines))

    def test_tool_agent_routes_fresh_search_requests_through_search_tool(self) -> None:
        backend = FakeBackend()
        backend.transcript = "What is the weather today in Berlin?"
        tool_agent = FakeToolCallingProvider(
            tool_name="search_live_info",
            arguments={
                "question": "What is the weather today in Berlin?",
                "location_hint": "Berlin",
                "date_context": "2026-03-14",
            },
        )
        loop, lines, _recorder, _player, _printer = self.make_loop(
            backend=backend,
            tool_agent_provider=tool_agent,
        )

        loop.handle_button_press("green")

        self.assertEqual(tool_agent.start_calls[0][0], "What is the weather today in Berlin?")
        self.assertEqual(len(backend.search_calls), 1)
        question, conversation, location_hint, date_context = backend.search_calls[0]
        self.assertEqual(question, "What is the weather today in Berlin?")
        self.assertEqual(location_hint, "Berlin")
        self.assertEqual(date_context, "2026-03-14")
        _assert_contains_system_message(self, conversation, "All user-facing spoken and written replies")
        self.assertIn("search_tool_call=true", lines)
        self.assertEqual(len(loop.runtime.memory.search_results), 1)
        self.assertEqual(loop.runtime.memory.search_results[0].question, "What is the weather today in Berlin?")

    def test_green_button_updates_runtime_voice_assessment(self) -> None:
        class FakeVoiceProfileMonitor:
            def assess_wav_bytes(self, audio_bytes: bytes) -> VoiceAssessment:
                self.audio_bytes = audio_bytes
                checked_at = _fresh_checked_at()
                return VoiceAssessment(
                    status="likely_user",
                    label="Likely user",
                    detail="Close to the enrolled template.",
                    confidence=0.81,
                    checked_at=checked_at,
                )

        monitor = FakeVoiceProfileMonitor()
        backend = FakeBackend()
        loop, lines, _recorder, _player, _printer = self.make_loop(
            backend=backend,
            voice_profile_monitor=monitor,
        )

        loop.handle_button_press("green")

        self.assertTrue(monitor.audio_bytes.startswith(b"RIFF"))
        self.assertEqual(loop.runtime.user_voice_status, "likely_user")
        self.assertEqual(loop.runtime.user_voice_confidence, 0.81)
        self.assertIsNotNone(loop.runtime.user_voice_checked_at)
        assert loop.runtime.user_voice_checked_at is not None
        self.assertTrue(loop.runtime.user_voice_checked_at.endswith("Z"))
        self.assertIsNotNone(backend.respond_calls[0][1])
        _assert_contains_system_message(self, backend.respond_calls[0][1], "Speaker signal: likely match")
        self.assertIn("voice_profile_status=likely_user", lines)
        self.assertIn("voice_profile_confidence=0.81", lines)

    def test_social_trigger_speaks_proactive_prompt(self) -> None:
        backend = FakeBackend()
        loop, lines, _recorder, player, _printer = self.make_loop(backend=backend)

        spoke = loop.handle_social_trigger(
            SocialTriggerDecision(
                trigger_id="attention_window",
                prompt="Kann ich dir bei etwas helfen?",
                reason="Person looked toward the device and stayed quiet.",
                observed_at=42.0,
                priority=SocialTriggerPriority.ATTENTION_WINDOW,
            )
        )

        self.assertTrue(spoke)
        self.assertEqual(loop.runtime.status.value, "waiting")
        self.assertEqual(player.played, [b"RIFF"])
        self.assertEqual(loop.runtime.last_response, None)
        self.assertEqual(len(backend.proactive_calls), 1)
        trigger_id, reason, default_prompt, priority, conversation, recent_prompts, observation_facts = backend.proactive_calls[0]
        self.assertEqual(
            (trigger_id, reason, default_prompt, priority),
            (
                "attention_window",
                "Person looked toward the device and stayed quiet.",
                "Kann ich dir bei etwas helfen?",
                int(SocialTriggerPriority.ATTENTION_WINDOW),
            ),
        )
        _assert_contains_system_message(self, conversation, "All user-facing spoken and written replies")
        self.assertEqual(recent_prompts, ())
        self.assertEqual(observation_facts, ())
        self.assertIn("status=answering", lines)
        self.assertIn("social_trigger=attention_window", lines)
        self.assertIn("social_prompt_mode=llm", lines)
        self.assertIn("social_prompt=Proaktiv: attention_window", lines)
        social_events = [
            entry
            for entry in loop.runtime.ops_events.tail(limit=20)
            if entry["event"] == "social_trigger_prompted"
            and entry.get("data", {}).get("trigger") == "attention_window"
        ]
        self.assertEqual(len(social_events), 1)
        self.assertEqual(social_events[0]["data"]["prompt"], "Proaktiv: attention_window")
        self.assertEqual(social_events[0]["data"]["default_prompt"], "Kann ich dir bei etwas helfen?")
        self.assertEqual(social_events[0]["data"]["prompt_mode"], "llm")

    def test_social_trigger_uses_direct_prompt_for_safety_events(self) -> None:
        backend = FakeBackend()
        loop, lines, _recorder, player, _printer = self.make_loop(backend=backend)

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
        self.assertEqual(backend.proactive_calls, [])
        self.assertEqual(player.played, [b"RIFF"])
        self.assertIn("social_prompt_mode=direct_safety", lines)
        self.assertIn("social_prompt=Brauchst du Hilfe?", lines)
        social_events = [
            entry
            for entry in loop.runtime.ops_events.tail(limit=20)
            if entry["event"] == "social_trigger_prompted"
            and entry.get("data", {}).get("trigger") == "possible_fall"
        ]
        self.assertEqual(len(social_events), 1)
        self.assertEqual(social_events[0]["data"]["prompt"], "Brauchst du Hilfe?")
        self.assertEqual(social_events[0]["data"]["prompt_mode"], "direct_safety")

    def test_social_trigger_is_skipped_when_runtime_is_busy(self) -> None:
        backend = FakeBackend()
        loop, lines, _recorder, player, _printer = self.make_loop(backend=backend)
        loop.runtime.press_green_button()

        spoke = loop.handle_social_trigger(
            SocialTriggerDecision(
                trigger_id="showing_intent",
                prompt="Möchtest du mir etwas zeigen?",
                reason="Hand or object near the camera.",
                observed_at=42.0,
                priority=SocialTriggerPriority.SHOWING_INTENT,
            )
        )

        self.assertFalse(spoke)
        self.assertEqual(player.played, [])
        self.assertIn("social_trigger_skipped=busy", lines)
        social_events = [entry for entry in loop.runtime.ops_events.tail(limit=20) if entry["event"] == "social_trigger_skipped"]
        self.assertEqual(len(social_events), 1)
        self.assertEqual(social_events[0]["data"]["prompt"], "Möchtest du mir etwas zeigen?")

    def test_run_opens_and_closes_proactive_monitor(self) -> None:
        backend = FakeBackend()
        button_monitor = FakeIdleButtonMonitor()
        proactive_monitor = FakeProactiveMonitor()
        loop, _lines, _recorder, _player, _printer = self.make_loop(
            backend=backend,
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
