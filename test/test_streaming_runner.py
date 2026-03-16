from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.contracts import (
    AgentToolCall,
    FirstWordReply,
    StreamingTranscriptionResult,
    ToolCallingTurnResponse,
)
from twinr.agent.tools.dual_lane_loop import DualLaneToolLoop
from twinr.agent.workflows.streaming_runner import TwinrStreamingHardwareLoop
from twinr.config import TwinrConfig
from twinr.memory.longterm.models import (
    LongTermConsolidationResultV1,
    LongTermMemoryConflictV1,
    LongTermMemoryObjectV1,
    LongTermSourceRefV1,
)
from twinr.providers.openai import (
    OpenAIBackend,
    OpenAIFirstWordProvider,
    OpenAIToolCallingAgentProvider,
    OpenAITextResponse,
)
from twinr.runtime import TwinrRuntime


class FakeToolAgentProvider:
    def __init__(self, config: TwinrConfig) -> None:
        self.config = config
        self.start_calls: list[dict[str, object]] = []
        self.continue_calls: list[dict[str, object]] = []

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
        self.start_calls.append(
            {
                "prompt": prompt,
                "conversation": conversation,
                "instructions": instructions,
                "tool_schemas": list(tool_schemas),
                "allow_web_search": allow_web_search,
            }
        )
        if on_text_delta is not None:
            on_text_delta("Ich drucke das.")
        return ToolCallingTurnResponse(
            text="Ich drucke das.",
            tool_calls=(
                AgentToolCall(
                    name="print_receipt",
                    call_id="call_print_1",
                    arguments={"text": "Termine"},
                    raw_arguments='{"text":"Termine"}',
                ),
            ),
            response_id="resp_start_1",
            continuation_token="resp_start_1",
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
        self.continue_calls.append(
            {
                "continuation_token": continuation_token,
                "tool_results": list(tool_results),
                "instructions": instructions,
                "tool_schemas": list(tool_schemas),
                "allow_web_search": allow_web_search,
            }
        )
        if on_text_delta is not None:
            on_text_delta(" Ist erledigt.")
        return ToolCallingTurnResponse(
            text="Ist erledigt.",
            response_id="resp_done_1",
        )


class FakePrintBackend:
    def __init__(self, config: TwinrConfig) -> None:
        self.config = config
        self.print_calls: list[tuple[str | None, str | None, str]] = []

    def compose_print_job_with_metadata(
        self,
        *,
        conversation=None,
        focus_hint: str | None = None,
        direct_text: str | None = None,
        request_source: str = "button",
    ) -> OpenAITextResponse:
        del conversation
        self.print_calls.append((focus_hint, direct_text, request_source))
        return OpenAITextResponse(text="AUSDRUCK")

    def phrase_due_reminder_with_metadata(self, reminder, *, now=None) -> OpenAITextResponse:
        del reminder, now
        return OpenAITextResponse(text="Erinnerung")

    def phrase_proactive_prompt_with_metadata(self, **kwargs) -> OpenAITextResponse:
        del kwargs
        return OpenAITextResponse(text="Proaktiv")

    def search_live_info_with_metadata(self, question: str, **kwargs):
        del question, kwargs
        return SimpleNamespace(
            answer="Antwort",
            sources=(),
            response_id="resp_search",
            request_id="req_search",
            model="gpt-5.2",
            token_usage=None,
            used_web_search=True,
        )

    def respond_to_images_with_metadata(self, prompt: str, **kwargs) -> OpenAITextResponse:
        del prompt, kwargs
        return OpenAITextResponse(text="Kamera")

    def fulfill_automation_prompt_with_metadata(self, prompt: str, **kwargs) -> OpenAITextResponse:
        del prompt, kwargs
        return OpenAITextResponse(text="Automation")


class FakeSpeechToTextProvider:
    def __init__(self, config: TwinrConfig) -> None:
        self.config = config

    def transcribe(self, audio_bytes: bytes, **kwargs) -> str:
        del audio_bytes, kwargs
        return "Hallo Twinr"

    def transcribe_path(self, path, **kwargs) -> str:
        del path, kwargs
        return "Hallo Twinr"


class FakeStreamingSpeechSession:
    def __init__(self) -> None:
        self.sent: list[bytes] = []
        self.closed = False
        self.finalize_calls = 0

    def send_pcm(self, pcm_bytes: bytes) -> None:
        self.sent.append(pcm_bytes)

    def snapshot(self) -> StreamingTranscriptionResult:
        return StreamingTranscriptionResult(
            transcript="Streaming Hallo Twinr",
            request_id="dg-stream-1",
            saw_interim=True,
            saw_speech_final=True,
            saw_utterance_end=False,
        )

    def finalize(self) -> StreamingTranscriptionResult:
        self.finalize_calls += 1
        return StreamingTranscriptionResult(
            transcript="Streaming Hallo Twinr",
            request_id="dg-stream-1",
            saw_interim=True,
            saw_speech_final=True,
            saw_utterance_end=False,
        )

    def close(self) -> None:
        self.closed = True


class FakeStreamingSpeechToTextProvider(FakeSpeechToTextProvider):
    def __init__(self, config: TwinrConfig) -> None:
        super().__init__(config)
        self.session = FakeStreamingSpeechSession()
        self.start_calls: list[dict[str, object]] = []
        self.interim_callback = None

    def transcribe(self, audio_bytes: bytes, **kwargs) -> str:
        raise AssertionError("streaming STT path should not fall back to file transcription")

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
        del prompt, on_endpoint
        self.start_calls.append(
            {
                "sample_rate": sample_rate,
                "channels": channels,
                "language": language,
            }
        )
        self.interim_callback = on_interim
        if on_interim is not None:
            on_interim("Stream partiell")
        return self.session


class FakeFirstWordProvider:
    def __init__(self, config: TwinrConfig, *, reply: FirstWordReply | None = None) -> None:
        self.config = config
        self.reply_value = reply or FirstWordReply(mode="filler", spoken_text="Ich schaue kurz nach.")
        self.calls: list[dict[str, object]] = []

    def reply(self, prompt: str, *, conversation=None, instructions=None) -> FirstWordReply:
        self.calls.append(
            {
                "prompt": prompt,
                "conversation": conversation,
                "instructions": instructions,
            }
        )
        return self.reply_value


class FakeTextToSpeechProvider:
    def __init__(self, config: TwinrConfig) -> None:
        self.config = config
        self.calls: list[str] = []
        self.stream_calls: list[str] = []
        self.stream_chunk_sizes: list[int] = []

    def synthesize(self, text: str, **kwargs) -> bytes:
        del kwargs
        self.calls.append(text)
        return b"RIFF"

    def synthesize_stream(self, text: str, **kwargs):
        self.stream_chunk_sizes.append(int(kwargs.get("chunk_size", 4096)))
        self.calls.append(text)
        self.stream_calls.append(text)
        yield b"RI"
        yield b"FF"


class TraceTextToSpeechProvider(FakeTextToSpeechProvider):
    def __init__(self, config: TwinrConfig, trace: list[str]) -> None:
        super().__init__(config)
        self.trace = trace

    def synthesize_stream(self, text: str, **kwargs):
        self.trace.append(f"tts_start:{text}")
        yield from super().synthesize_stream(text, **kwargs)


class FakePlayer:
    def __init__(self) -> None:
        self.played: list[bytes] = []
        self.played_wav_bytes: list[bytes] = []

    def play_wav_chunks(self, chunks, *, should_stop=None) -> None:
        payload = bytearray()
        for chunk in chunks:
            payload.extend(chunk)
            if should_stop is not None and should_stop():
                break
        self.played.append(bytes(payload))

    def play_wav_bytes(self, audio_bytes: bytes) -> None:
        self.played_wav_bytes.append(audio_bytes)

    def play_tone(self, **kwargs) -> None:
        del kwargs


class FakePrinter:
    def __init__(self) -> None:
        self.printed: list[str] = []

    def print_text(self, text: str) -> str:
        self.printed.append(text)
        return "job-1"


class FakeVoiceProfileMonitor:
    def summary(self):
        return SimpleNamespace(enrolled=False, sample_count=0, updated_at=None, average_duration_ms=None)

    def assess_pcm16(self, *args, **kwargs):
        del args, kwargs
        return SimpleNamespace(should_persist=False, status=None, confidence=None, checked_at=None)


class FakeUsageStore:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def append(self, **kwargs) -> None:
        self.calls.append(kwargs)


class FakeRecorder:
    def capture_pcm_until_pause_with_options(self, **kwargs):
        on_chunk = kwargs.get("on_chunk")
        if on_chunk is not None:
            on_chunk(b"PCM-A")
            on_chunk(b"PCM-B")
        return SimpleNamespace(
            pcm_bytes=b"PCM-AB",
            speech_started_after_ms=120,
            resumed_after_pause_count=1,
        )


class CapturingDualLaneLoop(DualLaneToolLoop):
    def __init__(self) -> None:
        self.run_calls: list[dict[str, object]] = []
        self.run_handoff_calls: list[dict[str, object]] = []

    def run(self, prompt: str, **kwargs):
        self.run_calls.append({"prompt": prompt, **kwargs})
        return SimpleNamespace(
            text="Ich schaue kurz nach.\nMorgen wird es sonnig.",
            response_id="resp_dual_lane",
            request_id="req_dual_lane",
            rounds=1,
            tool_calls=(),
            used_web_search=False,
            model="gpt-4o-mini",
            token_usage=None,
        )

    def run_handoff_only(self, prompt: str, **kwargs):
        self.run_handoff_calls.append({"prompt": prompt, **kwargs})
        on_lane_text_delta = kwargs.get("on_lane_text_delta")
        if on_lane_text_delta is not None:
            on_lane_text_delta(
                SimpleNamespace(
                    text="Heute wird es sonnig.",
                    lane="final",
                    replace_current=True,
                    atomic=True,
                )
            )
        return SimpleNamespace(
            text="Heute wird es sonnig.",
            response_id="resp_handoff",
            request_id="req_handoff",
            rounds=2,
            tool_calls=(),
            used_web_search=True,
            model="gpt-4o-mini",
            token_usage=None,
        )


class StubSupervisorDecision:
    def __init__(self, *, action: str, spoken_reply: str | None = None) -> None:
        self.action = action
        self.spoken_reply = spoken_reply
        self.spoken_ack = None
        self.kind = None
        self.goal = None
        self.allow_web_search = None
        self.response_id = "prefetch_resp"
        self.request_id = "prefetch_req"
        self.model = "gpt-4o-mini"
        self.token_usage = None


class StreamingRunnerTests(unittest.TestCase):
    def test_openai_dual_lane_uses_separate_backends_per_lane(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                llm_provider="openai",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            loop = object.__new__(TwinrStreamingHardwareLoop)
            loop.config = config
            loop.tool_agent_provider = OpenAIToolCallingAgentProvider(
                OpenAIBackend(config=config, client=SimpleNamespace()),
            )
            loop._tool_handlers = {}
            loop.first_word_provider = None
            tool_schemas = ()
            streaming_turn_loop = TwinrStreamingHardwareLoop._build_streaming_turn_loop(
                loop,
                tool_schemas=tool_schemas,
            )

        self.assertIsInstance(streaming_turn_loop, DualLaneToolLoop)
        self.assertIsInstance(loop.first_word_provider, OpenAIFirstWordProvider)
        self.assertIsInstance(streaming_turn_loop.supervisor_provider, OpenAIToolCallingAgentProvider)
        self.assertIsInstance(streaming_turn_loop.specialist_provider, OpenAIToolCallingAgentProvider)
        self.assertIsNot(
            loop.first_word_provider.backend,
            streaming_turn_loop.specialist_provider.backend,
        )
        self.assertIsNot(
            loop.first_word_provider.backend,
            streaming_turn_loop.supervisor_provider.backend,
        )
        self.assertIsNot(
            streaming_turn_loop.supervisor_provider.backend,
            streaming_turn_loop.specialist_provider.backend,
        )
        self.assertIsNotNone(streaming_turn_loop.supervisor_decision_provider)
        self.assertIsNot(
            streaming_turn_loop.supervisor_decision_provider.backend,
            streaming_turn_loop.specialist_provider.backend,
        )

    def test_text_turn_executes_tool_calls_and_streams_tts(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            tool_agent = FakeToolAgentProvider(config)
            support_provider = FakePrintBackend(config)
            tts_provider = FakeTextToSpeechProvider(config)
            player = FakePlayer()
            printer = FakePrinter()
            usage_store = FakeUsageStore()

            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=tool_agent,
                print_backend=support_provider,
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=support_provider,
                tts_provider=tts_provider,
                player=player,
                printer=printer,
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=usage_store,
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )

            keep_listening = loop._run_single_text_turn(
                transcript="Bitte druck das aus",
                listen_source="button",
                proactive_trigger=None,
            )

        self.assertTrue(keep_listening)
        self.assertEqual(printer.printed, ["AUSDRUCK"])
        self.assertIn("Ich drucke das.", runtime.last_response or "")
        self.assertIn("Ist erledigt.", runtime.last_response or "")
        self.assertEqual(tool_agent.continue_calls[0]["continuation_token"], "resp_start_1")
        self.assertEqual(tool_agent.continue_calls[0]["tool_results"][0].name, "print_receipt")
        self.assertEqual(tool_agent.start_calls[0]["allow_web_search"], False)
        self.assertTrue(any(call["request_kind"] == "print" for call in usage_store.calls))
        self.assertTrue(any(call["request_kind"] == "conversation" for call in usage_store.calls))

    def test_audio_turn_uses_streaming_stt_session_when_available(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            tool_agent = FakeToolAgentProvider(config)
            support_provider = FakePrintBackend(config)
            tts_provider = FakeTextToSpeechProvider(config)
            player = FakePlayer()
            printer = FakePrinter()
            usage_store = FakeUsageStore()
            stt_provider = FakeStreamingSpeechToTextProvider(config)
            lines: list[str] = []

            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=tool_agent,
                print_backend=support_provider,
                stt_provider=stt_provider,
                agent_provider=support_provider,
                tts_provider=tts_provider,
                recorder=FakeRecorder(),
                player=player,
                printer=printer,
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=usage_store,
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
                emit=lines.append,
            )

            keep_listening = loop._run_single_audio_turn(
                initial_source="button",
                follow_up=False,
                listening_window=runtime.listening_window(initial_source="button", follow_up=False),
                listen_source="button",
                proactive_trigger=None,
                speech_start_chunks=None,
                ignore_initial_ms=0,
                timeout_emit_key="listen_timeout",
                timeout_message="Listening timed out before speech started.",
                play_initial_beep=False,
            )

        self.assertTrue(keep_listening)
        self.assertEqual(stt_provider.start_calls[0]["sample_rate"], config.audio_sample_rate)
        self.assertEqual(stt_provider.start_calls[0]["channels"], config.audio_channels)
        self.assertEqual(stt_provider.start_calls[0]["language"], config.deepgram_stt_language)
        self.assertEqual(stt_provider.session.sent, [b"PCM-A", b"PCM-B"])
        self.assertEqual(stt_provider.session.finalize_calls, 0)
        self.assertTrue(stt_provider.session.closed)
        self.assertIn("transcript=Streaming Hallo Twinr", lines)
        self.assertIn("stt_streaming_early=true", lines)

    def test_segment_boundary_prefers_clause_and_soft_wrap(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                streaming_tts_clause_min_chars=20,
                streaming_tts_soft_segment_chars=40,
                streaming_tts_hard_segment_chars=60,
                long_term_memory_query_rewrite_enabled=False,
            )
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=TwinrRuntime(config=config),
                tool_agent_provider=FakeToolAgentProvider(config),
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                recorder=FakeRecorder(),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )

            clause_boundary = loop._segment_boundary("Das Wetter morgen ist wechselhaft, aber am Nachmittag trockener")
            soft_wrap_boundary = loop._segment_boundary(
                "Das Wetter morgen bleibt insgesamt wechselhaft und am Nachmittag wieder trockener"
            )

        self.assertEqual(clause_boundary, len("Das Wetter morgen ist wechselhaft,"))
        self.assertIsNotNone(soft_wrap_boundary)
        self.assertLess(soft_wrap_boundary, len("Das Wetter morgen bleibt insgesamt wechselhaft und am Nachmittag wieder trockener"))

    def test_streaming_tts_uses_configured_chunk_size(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                openai_tts_stream_chunk_size=1536,
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            tool_agent = FakeToolAgentProvider(config)
            tts_provider = FakeTextToSpeechProvider(config)
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=tool_agent,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=tts_provider,
                recorder=FakeRecorder(),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )

            loop._run_single_text_turn(
                transcript="Bitte druck das aus",
                listen_source="button",
                proactive_trigger=None,
            )

        self.assertIn(1536, tts_provider.stream_chunk_sizes)

    def test_dual_lane_filler_uses_streamed_tts_not_prefetched_audio(self) -> None:
        class AckOnlyLoop(DualLaneToolLoop):
            def __init__(self) -> None:
                pass

            def run(self, *args, **kwargs):
                on_lane_text_delta = kwargs.get("on_lane_text_delta")
                if on_lane_text_delta is not None:
                    on_lane_text_delta(
                        SimpleNamespace(
                            text="Ich schaue kurz nach.",
                            lane="filler",
                            replace_current=False,
                        )
                    )
                return SimpleNamespace(
                    text="Ich schaue kurz nach.",
                    response_id="resp_ack",
                    request_id="req_ack",
                    rounds=1,
                    tool_calls=(),
                    used_web_search=False,
                    model="gpt-4o-mini",
                    token_usage=None,
                )

        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            tts_provider = FakeTextToSpeechProvider(config)
            player = FakePlayer()
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=AckOnlyLoop(),
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=tts_provider,
                player=player,
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )

            loop._run_single_text_turn(
                transcript="Wie wird das Wetter morgen?",
                listen_source="button",
                proactive_trigger=None,
            )

        self.assertEqual(player.played_wav_bytes, [])
        self.assertIn("Ich schaue kurz nach.", tts_provider.stream_calls)

    def test_prefetched_search_handoff_uses_speculative_result_before_full_run(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            tts_provider = FakeTextToSpeechProvider(config)
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=tts_provider,
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            decision = SimpleNamespace(
                action="handoff",
                spoken_ack="Ich schaue kurz nach.",
                spoken_reply=None,
                kind="search",
                goal="Check the weather.",
                allow_web_search=True,
                response_id="prefetch_resp",
                request_id="prefetch_req",
                model="gpt-4o-mini",
                token_usage=None,
            )
            speculative_result = SimpleNamespace(
                text="Heute wird es sonnig.",
                response_id="resp_spec",
                request_id="req_spec",
                rounds=2,
                tool_calls=(),
                used_web_search=True,
                model="gpt-4o-mini",
                token_usage=None,
            )

            loop._consume_speculative_supervisor_decision = lambda transcript: decision  # type: ignore[method-assign]
            loop._consume_speculative_handoff_result = (  # type: ignore[method-assign]
                lambda transcript, handoff, wait_for_completion: speculative_result
            )

            keep_listening = loop._run_single_text_turn(
                transcript="Wie ist das Wetter heute?",
                listen_source="button",
                proactive_trigger=None,
            )

        self.assertTrue(keep_listening)
        self.assertEqual(dual_lane.run_calls, [])
        self.assertEqual(dual_lane.run_handoff_calls, [])
        self.assertEqual(tts_provider.stream_calls, ["Heute wird es sonnig."])
        self.assertEqual(runtime.last_response, "Heute wird es sonnig.")

    def test_prefetched_search_handoff_falls_back_to_handoff_only_lane(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            tts_provider = FakeTextToSpeechProvider(config)
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=tts_provider,
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            loop.first_word_provider = FakeFirstWordProvider(
                config,
                reply=FirstWordReply(mode="filler", spoken_text="Ich schaue kurz nach."),
            )
            decision = SimpleNamespace(
                action="handoff",
                spoken_ack="Ich schaue kurz nach.",
                spoken_reply=None,
                kind="search",
                goal="Check the weather.",
                allow_web_search=True,
                response_id="prefetch_resp",
                request_id="prefetch_req",
                model="gpt-4o-mini",
                token_usage=None,
            )

            loop._consume_speculative_supervisor_decision = lambda transcript: decision  # type: ignore[method-assign]
            loop._consume_speculative_handoff_result = (  # type: ignore[method-assign]
                lambda transcript, handoff, wait_for_completion: None
            )

            keep_listening = loop._run_single_text_turn(
                transcript="Wie ist das Wetter heute?",
                listen_source="button",
                proactive_trigger=None,
            )

        self.assertTrue(keep_listening)
        self.assertEqual(dual_lane.run_calls, [])
        self.assertEqual(len(dual_lane.run_handoff_calls), 1)
        self.assertFalse(dual_lane.run_handoff_calls[0]["emit_filler"])
        self.assertEqual(tts_provider.stream_calls[-1], "Heute wird es sonnig.")
        self.assertEqual(runtime.last_response, "Heute wird es sonnig.")

    def test_speculative_first_word_prefetch_is_used_before_final_lane(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            tts_provider = FakeTextToSpeechProvider(config)
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=tts_provider,
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            loop._consume_speculative_first_word = lambda transcript: FirstWordReply(  # type: ignore[method-assign]
                mode="filler",
                spoken_text="Ich schaue kurz nach.",
            )
            loop._run_dual_lane_final_response = lambda transcript, turn_instructions: SimpleNamespace(  # type: ignore[method-assign]
                text="Heute wird es sonnig.",
                response_id="resp_final",
                request_id="req_final",
                rounds=2,
                tool_calls=(),
                used_web_search=True,
                model="gpt-4o-mini",
                token_usage=None,
            )

            keep_listening = loop._run_single_text_turn(
                transcript="Wie ist das Wetter heute?",
                listen_source="button",
                proactive_trigger=None,
            )

        self.assertTrue(keep_listening)
        self.assertEqual(tts_provider.stream_calls[-1], "Heute wird es sonnig.")
        self.assertEqual(runtime.last_response, "Heute wird es sonnig.")

    def test_sync_first_word_direct_is_not_repeated_when_final_matches(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            tts_provider = FakeTextToSpeechProvider(config)
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=tts_provider,
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            loop.first_word_provider = FakeFirstWordProvider(
                config,
                reply=FirstWordReply(mode="direct", spoken_text="Ja, alles gut."),
            )
            loop._run_dual_lane_final_response = lambda transcript, turn_instructions: SimpleNamespace(  # type: ignore[method-assign]
                text="Ja, alles gut.",
                response_id="resp_final",
                request_id="req_final",
                rounds=1,
                tool_calls=(),
                used_web_search=False,
                model="gpt-4o-mini",
                token_usage=None,
            )

            keep_listening = loop._run_single_text_turn(
                transcript="Alles ok bei dir?",
                listen_source="button",
                proactive_trigger=None,
            )

        self.assertTrue(keep_listening)
        self.assertEqual(tts_provider.stream_calls, ["Ja, alles gut."])
        self.assertEqual(runtime.last_response, "Ja, alles gut.")

    def test_final_lane_waits_for_first_audio_gate(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            dual_lane = CapturingDualLaneLoop()
            trace: list[str] = []
            tts_provider = TraceTextToSpeechProvider(config, trace)
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=tts_provider,
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            loop.first_word_provider = FakeFirstWordProvider(
                config,
                reply=FirstWordReply(mode="filler", spoken_text="Ich schaue kurz nach."),
            )

            def fake_final_response(transcript: str, *, turn_instructions: str | None):
                del transcript, turn_instructions
                trace.append("final_start")
                return SimpleNamespace(
                    text="Heute wird es sonnig.",
                    response_id="resp_final",
                    request_id="req_final",
                    rounds=1,
                    tool_calls=(),
                    used_web_search=True,
                    model="gpt-4o-mini",
                    token_usage=None,
                )

            loop._run_dual_lane_final_response = fake_final_response  # type: ignore[method-assign]

            keep_listening = loop._run_single_text_turn(
                transcript="Wie ist das Wetter heute?",
                listen_source="button",
                proactive_trigger=None,
            )

        self.assertTrue(keep_listening)
        self.assertLess(
            trace.index("tts_start:Ich schaue kurz nach."),
            trace.index("final_start"),
        )

    def test_dual_lane_streaming_runner_passes_slim_supervisor_context(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                streaming_supervisor_context_turns=2,
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            runtime.memory.remember("user", "Alter Turn eins")
            runtime.memory.remember("assistant", "Alter Turn zwei")
            runtime.memory.remember("user", "Letzte Frage")
            runtime.memory.remember("assistant", "Letzte Antwort")
            dual_lane = CapturingDualLaneLoop()

            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=FakeToolAgentProvider(config),
                streaming_turn_loop=dual_lane,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )

            loop._run_single_text_turn(
                transcript="Wie wird das Wetter morgen?",
                listen_source="button",
                proactive_trigger=None,
            )

        call = dual_lane.run_calls[0]
        supervisor_context = call["supervisor_conversation"]
        specialist_context = call["conversation"]
        self.assertEqual(
            [(role, content) for role, content in supervisor_context if role != "system"],
            [("user", "Letzte Frage"), ("assistant", "Letzte Antwort")],
        )
        self.assertGreater(len(specialist_context), len(supervisor_context))

    def test_consume_speculative_supervisor_decision_accepts_direct_reply(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=TwinrRuntime(config=config),
                tool_agent_provider=FakeToolAgentProvider(config),
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            loop._speculative_supervisor_started = True
            loop._speculative_supervisor_transcript = "Alles ok"
            loop._speculative_supervisor_decision = StubSupervisorDecision(
                action="direct",
                spoken_reply="Mir geht's gut.",
            )
            loop._speculative_supervisor_done.set()

            decision = loop._consume_speculative_supervisor_decision("Alles ok bei dir?")

        self.assertIsNotNone(decision)
        self.assertEqual(decision.action, "direct")

    def test_streaming_endpoint_can_prime_speculative_supervisor(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=TwinrRuntime(config=config),
                tool_agent_provider=FakeToolAgentProvider(config),
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            seen: list[str] = []
            loop._maybe_start_speculative_supervisor_decision = seen.append  # type: ignore[method-assign]

            loop._on_streaming_stt_endpoint(SimpleNamespace(transcript="Alles okay bei dir", event_type="speech_final"))

        self.assertEqual(seen, ["Alles okay bei dir"])

    def test_streaming_endpoint_can_prime_speculative_first_word(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=TwinrRuntime(config=config),
                tool_agent_provider=FakeToolAgentProvider(config),
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            seen: list[str] = []
            loop._maybe_start_speculative_first_word = seen.append  # type: ignore[method-assign]

            loop._on_streaming_stt_endpoint(SimpleNamespace(transcript="Alles okay bei dir", event_type="speech_final"))

        self.assertEqual(seen, ["Alles okay bei dir"])

    def test_groq_config_uses_compact_tool_schemas_and_instructions(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                groq_api_key="groq-key",
                llm_provider="groq",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            tool_agent = FakeToolAgentProvider(config)
            support_provider = FakePrintBackend(config)

            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=tool_agent,
                print_backend=support_provider,
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=support_provider,
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )

            loop._run_single_text_turn(
                transcript="Bitte druck das aus",
                listen_source="button",
                proactive_trigger=None,
            )

        schema_properties = tool_agent.start_calls[0]["tool_schemas"][0]["parameters"]["properties"]
        schemas_by_name = {
            schema["name"]: schema
            for schema in tool_agent.start_calls[0]["tool_schemas"]
        }
        self.assertLess(len(tool_agent.start_calls[0]["tool_schemas"][0]["description"]), 80)
        self.assertTrue(all("description" not in value for value in schema_properties.values()))
        self.assertEqual(set(schema_properties), {"focus_hint", "text"})
        self.assertIn("question", schemas_by_name["search_live_info"]["parameters"]["properties"])
        self.assertIn("Available Twinr spoken voices", tool_agent.start_calls[0]["instructions"])
        self.assertNotIn("Current bounded simple settings", tool_agent.start_calls[0]["instructions"])

    def test_groq_config_streams_final_text_only(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                groq_api_key="groq-key",
                llm_provider="groq",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            tool_agent = FakeToolAgentProvider(config)
            support_provider = FakePrintBackend(config)
            tts_provider = FakeTextToSpeechProvider(config)

            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=tool_agent,
                print_backend=support_provider,
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=support_provider,
                tts_provider=tts_provider,
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )

            loop._run_single_text_turn(
                transcript="Bitte druck das aus",
                listen_source="button",
                proactive_trigger=None,
            )

        self.assertEqual(tts_provider.calls, ["Ist erledigt."])
        self.assertEqual(runtime.last_response, "Ist erledigt.")

    def test_text_turn_uses_processing_feedback_before_answering(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            tool_agent = FakeToolAgentProvider(config)
            loop = TwinrStreamingHardwareLoop(
                config=config,
                runtime=runtime,
                tool_agent_provider=tool_agent,
                print_backend=FakePrintBackend(config),
                stt_provider=FakeSpeechToTextProvider(config),
                agent_provider=FakePrintBackend(config),
                tts_provider=FakeTextToSpeechProvider(config),
                player=FakePlayer(),
                printer=FakePrinter(),
                voice_profile_monitor=FakeVoiceProfileMonitor(),
                usage_store=FakeUsageStore(),
                button_monitor=SimpleNamespace(),
                proactive_monitor=SimpleNamespace(),
            )
            feedback_kinds: list[str] = []

            def fake_start(kind: str):
                feedback_kinds.append(kind)
                return lambda: None

            loop._start_working_feedback_loop = fake_start  # type: ignore[method-assign]

            loop._run_single_text_turn(
                transcript="Bitte druck das aus",
                listen_source="button",
                proactive_trigger=None,
            )

        self.assertEqual(feedback_kinds[0], "processing")

    def test_tool_provider_context_hides_exact_contact_methods_and_conflicts(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                long_term_memory_query_rewrite_enabled=False,
            )
            runtime = TwinrRuntime(config=config)
            runtime.remember_contact(given_name="Anna", family_name="Schulz", phone="040 1234567")
            runtime.long_term_memory.object_store.apply_consolidation(
                LongTermConsolidationResultV1(
                    turn_id="turn:2",
                    occurred_at=datetime(2026, 3, 14, 12, 0, tzinfo=ZoneInfo("Europe/Berlin")),
                    episodic_objects=(),
                    durable_objects=(
                        LongTermMemoryObjectV1(
                            memory_id="fact:corinna_phone_old",
                            kind="contact_method_fact",
                            summary="Corinna Maier can be reached at +491761234.",
                            source=LongTermSourceRefV1(
                                source_type="conversation_turn",
                                event_ids=("turn:1",),
                                speaker="user",
                                modality="voice",
                            ),
                            status="active",
                            confidence=0.95,
                            slot_key="contact:person:corinna_maier:phone",
                            value_key="+491761234",
                        ),
                    ),
                    deferred_objects=(
                        LongTermMemoryObjectV1(
                            memory_id="fact:corinna_phone_new",
                            kind="contact_method_fact",
                            summary="Corinna Maier can be reached at +4940998877.",
                            source=LongTermSourceRefV1(
                                source_type="conversation_turn",
                                event_ids=("turn:2",),
                                speaker="user",
                                modality="voice",
                            ),
                            status="uncertain",
                            confidence=0.92,
                            slot_key="contact:person:corinna_maier:phone",
                            value_key="+4940998877",
                        ),
                    ),
                    conflicts=(
                        LongTermMemoryConflictV1(
                            slot_key="contact:person:corinna_maier:phone",
                            candidate_memory_id="fact:corinna_phone_new",
                            existing_memory_ids=("fact:corinna_phone_old",),
                            question="Which phone number should I use for Corinna Maier?",
                            reason="Conflicting phone numbers exist.",
                        ),
                    ),
                    graph_edges=(),
                )
            )
            runtime.last_transcript = "Wie ist die Telefonnummer von Anna Schulz?"
            provider_contact_context = runtime.provider_conversation_context()
            tool_contact_context = runtime.tool_provider_conversation_context()
            supervisor_contact_context = runtime.supervisor_provider_conversation_context()
            runtime.last_transcript = "Gibt es bei Corinna Maier offene Erinnerungskonflikte?"
            provider_conflict_context = runtime.provider_conversation_context()
            tool_conflict_context = runtime.tool_provider_conversation_context()
            supervisor_conflict_context = runtime.supervisor_provider_conversation_context()

        provider_contact_system = "\n".join(content for role, content in provider_contact_context if role == "system")
        tool_contact_system = "\n".join(content for role, content in tool_contact_context if role == "system")
        supervisor_contact_system = "\n".join(content for role, content in supervisor_contact_context if role == "system")
        provider_conflict_system = "\n".join(content for role, content in provider_conflict_context if role == "system")
        tool_conflict_system = "\n".join(content for role, content in tool_conflict_context if role == "system")
        supervisor_conflict_system = "\n".join(content for role, content in supervisor_conflict_context if role == "system")
        self.assertIn("040 1234567", provider_contact_system)
        self.assertIn("Structured unresolved long-term memory conflicts", provider_conflict_system)
        self.assertIn("+4940998877", provider_conflict_system)
        self.assertNotIn("040 1234567", tool_contact_system)
        self.assertNotIn("Structured unresolved long-term memory conflicts", tool_conflict_system)
        self.assertNotIn("+4940998877", tool_conflict_system)
        self.assertNotIn("040 1234567", supervisor_contact_system)
        self.assertNotIn("Structured unresolved long-term memory conflicts", supervisor_conflict_system)
        self.assertNotIn("+4940998877", supervisor_conflict_system)

    def test_supervisor_context_uses_recent_raw_tail_only(self) -> None:
        with TemporaryDirectory() as temp_dir:
            runtime = TwinrRuntime(
                TwinrConfig(
                    openai_api_key="test-key",
                    project_root=temp_dir,
                    personality_dir="personality",
                    memory_max_turns=8,
                    memory_keep_recent=4,
                    streaming_supervisor_context_turns=2,
                )
            )
            runtime.memory.remember("user", "Turn one")
            runtime.memory.remember("assistant", "Turn two")
            runtime.memory.remember("user", "Turn three")
            runtime.memory.remember("assistant", "Turn four")
            context = runtime.supervisor_provider_conversation_context()

        non_system_messages = [(role, content) for role, content in context if role != "system"]
        self.assertEqual(
            non_system_messages,
            [("user", "Turn three"), ("assistant", "Turn four")],
        )


if __name__ == "__main__":
    unittest.main()
