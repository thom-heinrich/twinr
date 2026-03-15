from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.agent.base_agent.contracts import AgentToolResult
from twinr.providers.openai import (
    OpenAIProviderBundle,
    OpenAISupervisorDecisionProvider,
    OpenAIToolCallingAgentProvider,
    OpenAITextResponse,
)


class FakeAdapterBackend:
    def __init__(self, config: TwinrConfig) -> None:
        self.config = config
        self.calls: list[tuple[str, object]] = []

    def transcribe(self, audio_bytes: bytes, **kwargs) -> str:
        self.calls.append(("transcribe", (audio_bytes, kwargs)))
        return "transcribed"

    def transcribe_path(self, path, **kwargs) -> str:
        self.calls.append(("transcribe_path", (path, kwargs)))
        return "transcribed-path"

    def respond_streaming(self, prompt: str, **kwargs) -> OpenAITextResponse:
        self.calls.append(("respond_streaming", (prompt, kwargs)))
        return OpenAITextResponse(text="hello")

    def respond_with_metadata(self, prompt: str, **kwargs) -> OpenAITextResponse:
        self.calls.append(("respond_with_metadata", (prompt, kwargs)))
        return OpenAITextResponse(text="hello")

    def respond_to_images_with_metadata(self, prompt: str, **kwargs) -> OpenAITextResponse:
        self.calls.append(("respond_to_images_with_metadata", (prompt, kwargs)))
        return OpenAITextResponse(text="vision")

    def search_live_info_with_metadata(self, question: str, **kwargs):
        self.calls.append(("search_live_info_with_metadata", (question, kwargs)))
        return type("SearchResult", (), {"answer": "answer", "sources": ("https://example.com",), "response_id": None, "request_id": None, "model": "gpt", "token_usage": None, "used_web_search": True})()

    def compose_print_job_with_metadata(self, **kwargs) -> OpenAITextResponse:
        self.calls.append(("compose_print_job_with_metadata", kwargs))
        return OpenAITextResponse(text="print me")

    def phrase_due_reminder_with_metadata(self, reminder, **kwargs) -> OpenAITextResponse:
        self.calls.append(("phrase_due_reminder_with_metadata", (reminder, kwargs)))
        return OpenAITextResponse(text="remember")

    def phrase_proactive_prompt_with_metadata(self, **kwargs) -> OpenAITextResponse:
        self.calls.append(("phrase_proactive_prompt_with_metadata", kwargs))
        return OpenAITextResponse(text="proactive")

    def fulfill_automation_prompt_with_metadata(self, prompt: str, **kwargs) -> OpenAITextResponse:
        self.calls.append(("fulfill_automation_prompt_with_metadata", (prompt, kwargs)))
        return OpenAITextResponse(text="automation")

    def synthesize(self, text: str, **kwargs) -> bytes:
        self.calls.append(("synthesize", (text, kwargs)))
        return b"AUDIO"

    def synthesize_stream(self, text: str, **kwargs):
        self.calls.append(("synthesize_stream", (text, kwargs)))
        yield b"A"
        yield b"UDIO"


class FakeResponseStream:
    def __init__(self, events, final_response) -> None:
        self._events = list(events)
        self._final_response = final_response

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def __iter__(self):
        return iter(self._events)

    def get_final_response(self):
        return self._final_response


class FakeResponsesClient:
    def __init__(self) -> None:
        self.create_requests: list[dict[str, object]] = []
        self.create_results: list[object] = []
        self.stream_requests: list[dict[str, object]] = []
        self.stream_results: list[tuple[list[object], object]] = []

    def create(self, **request):
        self.create_requests.append(request)
        if not self.create_results:
            raise AssertionError("No fake create result configured")
        return self.create_results.pop(0)

    def stream(self, **request):
        self.stream_requests.append(request)
        if not self.stream_results:
            raise AssertionError("No fake stream result configured")
        events, final_response = self.stream_results.pop(0)
        return FakeResponseStream(events, final_response)


class FakeToolBackend:
    def __init__(self, config: TwinrConfig) -> None:
        self.config = config
        self._client = SimpleNamespace(responses=FakeResponsesClient())

    def _resolve_base_instructions(self) -> str:
        return "Base instructions"

    def _resolve_tool_loop_base_instructions(self) -> str:
        return "Tool-loop base instructions"

    def _build_response_request(
        self,
        prompt: str,
        *,
        conversation=None,
        instructions=None,
        allow_web_search=None,
        model: str,
        reasoning_effort: str,
        max_output_tokens=None,
        extra_user_content=None,
        prompt_cache_scope=None,
    ) -> dict[str, object]:
        del extra_user_content
        request: dict[str, object] = {
            "model": model,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            "reasoning": {"effort": reasoning_effort},
            "store": False,
        }
        if conversation is not None:
            request["conversation"] = conversation
        if instructions is not None:
            request["instructions"] = instructions
        if max_output_tokens is not None:
            request["max_output_tokens"] = max_output_tokens
        if allow_web_search:
            request["tools"] = [{"type": "web_search"}]
            request["tool_choice"] = "auto"
        if prompt_cache_scope:
            request["prompt_cache_key"] = f"twinr:{prompt_cache_scope}:{model}:de"
        return request

    def _build_tools(self, use_web_search: bool):
        return [{"type": "web_search"}] if use_web_search else []

    def _apply_prompt_cache(self, request: dict[str, object], *, scope: str | None, model: str) -> None:
        if scope:
            request["prompt_cache_key"] = f"twinr:{scope}:{model}:de"

    def _extract_output_text(self, response) -> str:
        return str(getattr(response, "output_text", "")).strip()

    def _used_web_search(self, response) -> bool:
        return any(getattr(item, "type", None) == "web_search_call" for item in getattr(response, "output", []) or [])


class OpenAIProviderBundleTests(unittest.TestCase):
    def test_bundle_adapters_delegate_and_share_config(self) -> None:
        backend = FakeAdapterBackend(TwinrConfig(openai_api_key="test-key"))
        bundle = OpenAIProviderBundle.from_backend(backend)

        self.assertEqual(bundle.stt.transcribe(b"PCM"), "transcribed")
        self.assertEqual(bundle.agent.respond_with_metadata("hello").text, "hello")
        self.assertEqual(bundle.agent.compose_print_job_with_metadata().text, "print me")
        self.assertEqual(tuple(bundle.tts.synthesize_stream("answer")), (b"A", b"UDIO"))
        self.assertEqual(bundle.combined.phrase_due_reminder_with_metadata(object()).text, "remember")
        self.assertIsNotNone(bundle.tool_agent)

        updated_config = replace(backend.config, default_model="gpt-4o-mini")
        bundle.combined.config = updated_config

        self.assertEqual(backend.config.default_model, "gpt-4o-mini")
        self.assertEqual(
            [name for name, _payload in backend.calls[:5]],
            [
                "transcribe",
                "respond_with_metadata",
                "compose_print_job_with_metadata",
                "synthesize_stream",
                "phrase_due_reminder_with_metadata",
            ],
        )


class OpenAIToolCallingAgentProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = TwinrConfig(openai_api_key="test-key", default_model="gpt-5.2")

    def test_start_turn_streaming_merges_function_tools_and_parses_calls(self) -> None:
        backend = FakeToolBackend(self.config)
        backend._client.responses.stream_results.append(
            (
                [SimpleNamespace(type="response.output_text.delta", delta="Ich prüfe das.")],
                SimpleNamespace(
                    id="resp_start_1",
                    _request_id="req_start_1",
                    model="gpt-5.2",
                    output_text="Ich prüfe das.",
                    output=[
                        SimpleNamespace(
                            type="function_call",
                            name="print_receipt",
                            call_id="call_1",
                            arguments='{"focus_hint":"arzttermin"}',
                        )
                    ],
                    usage=None,
                ),
            )
        )
        provider = OpenAIToolCallingAgentProvider(backend)
        text_deltas: list[str] = []

        response = provider.start_turn_streaming(
            "Bitte druck das aus",
            conversation=(("system", "Stay concise"),),
            instructions="Extra guidance",
            tool_schemas=[{"type": "function", "name": "print_receipt"}],
            allow_web_search=True,
            on_text_delta=text_deltas.append,
        )

        request = backend._client.responses.stream_requests[0]
        self.assertEqual(response.response_id, "resp_start_1")
        self.assertEqual(response.tool_calls[0].name, "print_receipt")
        self.assertEqual(response.tool_calls[0].arguments["focus_hint"], "arzttermin")
        self.assertEqual(text_deltas, ["Ich prüfe das."])
        self.assertEqual(request["model"], "gpt-5.2")
        self.assertIn("Tool-loop base instructions", request["instructions"])
        self.assertTrue(request["store"])
        self.assertEqual(request["tool_choice"], "auto")
        self.assertEqual(request["prompt_cache_key"], "twinr:tool_loop_start:gpt-5.2:de")
        self.assertEqual(
            [tool["type"] for tool in request["tools"]],
            ["web_search", "function"],
        )

    def test_continue_turn_streaming_sends_function_call_outputs(self) -> None:
        backend = FakeToolBackend(self.config)
        backend._client.responses.stream_results.append(
            (
                [],
                SimpleNamespace(
                    id="resp_continue_1",
                    _request_id="req_continue_1",
                    model="gpt-5.2",
                    output_text="Ist erledigt.",
                    output=[],
                    usage=None,
                ),
            )
        )
        provider = OpenAIToolCallingAgentProvider(backend)

        response = provider.continue_turn_streaming(
            continuation_token="resp_start_1",
            tool_results=(
                AgentToolResult(
                    call_id="call_1",
                    name="print_receipt",
                    output={"status": "printed"},
                    serialized_output='{"status":"printed"}',
                ),
            ),
            tool_schemas=[{"type": "function", "name": "print_receipt"}],
            allow_web_search=False,
        )

        request = backend._client.responses.stream_requests[0]
        self.assertEqual(response.text, "Ist erledigt.")
        self.assertEqual(request["previous_response_id"], "resp_start_1")
        self.assertIn("Tool-loop base instructions", request["instructions"])
        self.assertTrue(request["store"])
        self.assertEqual(request["prompt_cache_key"], "twinr:tool_loop_continue:gpt-5.2:de")
        self.assertEqual(
            request["input"],
            [{"type": "function_call_output", "call_id": "call_1", "output": '{"status":"printed"}'}],
        )
        self.assertEqual(request["tools"], [{"type": "function", "name": "print_receipt"}])

    def test_provider_overrides_model_and_base_instructions_for_fast_model(self) -> None:
        backend = FakeToolBackend(self.config)
        backend._client.responses.stream_results.append(
            (
                [],
                SimpleNamespace(
                    id="resp_override_1",
                    _request_id="req_override_1",
                    model="gpt-4o-mini",
                    output_text="Kurzantwort.",
                    output=[],
                    usage=None,
                ),
            )
        )
        provider = OpenAIToolCallingAgentProvider(
            backend,
            model_override="gpt-4o-mini",
            reasoning_effort_override="low",
            base_instructions_override="Supervisor role instructions",
        )

        response = provider.start_turn_streaming("Hallo")

        request = backend._client.responses.stream_requests[0]
        self.assertEqual(response.text, "Kurzantwort.")
        self.assertEqual(request["model"], "gpt-4o-mini")
        self.assertNotIn("reasoning", request)
        self.assertIn("Supervisor role instructions", request["instructions"])

    def test_provider_can_replace_default_tool_loop_base_instructions(self) -> None:
        backend = FakeToolBackend(self.config)
        backend._client.responses.stream_results.append(
            (
                [],
                SimpleNamespace(
                    id="resp_override_2",
                    _request_id="req_override_2",
                    model="gpt-4o-mini",
                    output_text="Kurzantwort.",
                    output=[],
                    usage=None,
                ),
            )
        )
        provider = OpenAIToolCallingAgentProvider(
            backend,
            model_override="gpt-4o-mini",
            reasoning_effort_override="low",
            base_instructions_override="Supervisor-only base instructions",
            replace_base_instructions=True,
        )

        provider.start_turn_streaming("Hallo", instructions="Runtime extra")

        request = backend._client.responses.stream_requests[0]
        self.assertIn("Supervisor-only base instructions", request["instructions"])
        self.assertIn("Runtime extra", request["instructions"])
        self.assertNotIn("Tool-loop base instructions", request["instructions"])

    def test_provider_strips_top_level_combinators_from_function_parameters(self) -> None:
        backend = FakeToolBackend(self.config)
        backend._client.responses.stream_results.append(
            (
                [],
                SimpleNamespace(
                    id="resp_tools_1",
                    _request_id="req_tools_1",
                    model="gpt-5.2",
                    output_text="Alles klar.",
                    output=[],
                    usage=None,
                ),
            )
        )
        provider = OpenAIToolCallingAgentProvider(backend)

        provider.start_turn_streaming(
            "Bitte druck das aus",
            tool_schemas=[
                {
                    "type": "function",
                    "name": "print_receipt",
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "anyOf": [{"required": ["text"]}],
                        "allOf": [{"properties": {"text": {"minLength": 1}}}],
                    },
                }
            ],
        )

        request = backend._client.responses.stream_requests[0]
        parameters = request["tools"][0]["parameters"]
        self.assertNotIn("anyOf", parameters)
        self.assertNotIn("allOf", parameters)


class OpenAISupervisorDecisionProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = TwinrConfig(openai_api_key="test-key", default_model="gpt-5.2")

    def test_decide_uses_structured_json_schema_and_replaces_base_instructions(self) -> None:
        backend = FakeToolBackend(self.config)
        backend._client.responses.create_results.append(
            SimpleNamespace(
                id="resp_decide_1",
                _request_id="req_decide_1",
                model="gpt-4o-mini",
                output_text=(
                    '{"action":"handoff","spoken_ack":"Ich schaue kurz nach.","spoken_reply":null,'
                    '"kind":"search","goal":"weather tomorrow","allow_web_search":true}'
                ),
                output=[],
                usage=None,
            )
        )
        provider = OpenAISupervisorDecisionProvider(
            backend,
            model_override="gpt-4o-mini",
            reasoning_effort_override="low",
            base_instructions_override="Supervisor decision base instructions",
            replace_base_instructions=True,
        )

        decision = provider.decide(
            "Wie wird das Wetter morgen?",
            conversation=(("user", "Voriger Turn"),),
            instructions="Runtime extra",
        )

        request = backend._client.responses.create_requests[0]
        self.assertEqual(decision.action, "handoff")
        self.assertEqual(decision.spoken_ack, "Ich schaue kurz nach.")
        self.assertEqual(decision.kind, "search")
        self.assertTrue(decision.allow_web_search)
        self.assertEqual(request["model"], "gpt-4o-mini")
        self.assertIn("Supervisor decision base instructions", request["instructions"])
        self.assertIn("Runtime extra", request["instructions"])
        self.assertNotIn("Tool-loop base instructions", request["instructions"])
        self.assertEqual(request["prompt_cache_key"], "twinr:supervisor_decision:gpt-4o-mini:de")
        self.assertEqual(request["max_output_tokens"], 80)
        self.assertEqual(request["text"]["format"]["type"], "json_schema")
