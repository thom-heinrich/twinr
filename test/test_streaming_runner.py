from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.contracts import AgentToolCall, ToolCallingTurnResponse
from twinr.agent.workflows.streaming_runner import TwinrStreamingHardwareLoop
from twinr.config import TwinrConfig
from twinr.providers.openai import OpenAITextResponse
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


class FakeTextToSpeechProvider:
    def __init__(self, config: TwinrConfig) -> None:
        self.config = config
        self.calls: list[str] = []

    def synthesize(self, text: str, **kwargs) -> bytes:
        del kwargs
        self.calls.append(text)
        return b"RIFF"

    def synthesize_stream(self, text: str, **kwargs):
        del kwargs
        self.calls.append(text)
        yield b"RI"
        yield b"FF"


class FakePlayer:
    def __init__(self) -> None:
        self.played: list[bytes] = []

    def play_wav_chunks(self, chunks) -> None:
        self.played.append(b"".join(chunks))

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


class StreamingRunnerTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
