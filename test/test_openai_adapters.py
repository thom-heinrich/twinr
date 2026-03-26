from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.contracts import AgentToolResult
from twinr.providers.openai import (
    OpenAIConversationClosureDecisionProvider,
    OpenAIFirstWordProvider,
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
        self.build_tools_calls: list[tuple[bool, str]] = []

    def _resolve_base_instructions(self) -> str:
        return "Base instructions"

    def _resolve_tool_loop_base_instructions(self) -> str:
        return "Tool-loop base instructions"

    def _call_with_model_fallback(self, preferred_model: str, fallback_models, call):
        attempted: list[str] = []
        for model in (preferred_model, *fallback_models):
            if not model or model in attempted:
                continue
            attempted.append(model)
            try:
                return call(model), model
            except Exception:
                continue
        raise AssertionError("No fake create result configured")

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

    def _build_tools(self, use_web_search: bool, *, model: str):
        self.build_tools_calls.append((use_web_search, model))
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

    def test_first_word_provider_requests_structured_fast_reply(self) -> None:
        backend = FakeToolBackend(
            TwinrConfig(
                openai_api_key="test-key",
                streaming_first_word_model="gpt-4.1-nano",
                streaming_first_word_max_output_tokens=36,
            )
        )
        backend._client.responses.create_results.append(
            SimpleNamespace(
                id="resp_first_word_1",
                _request_id="req_first_word_1",
                model="gpt-4.1-nano",
                output_text='{"mode":"filler","spoken_text":"Ich schaue kurz nach."}',
                output=[],
                usage=None,
            )
        )
        provider = OpenAIFirstWordProvider(
            backend,
            model_override="gpt-4.1-nano",
            base_instructions_override="Fast first-word instructions",
            replace_base_instructions=True,
        )

        reply = provider.reply(
            "Wie ist das Wetter heute?",
            conversation=(("assistant", "Vorherige Antwort"),),
        )

        request = backend._client.responses.create_requests[0]
        self.assertEqual(reply.mode, "filler")
        self.assertEqual(reply.spoken_text, "Ich schaue kurz nach.")
        self.assertEqual(request["model"], "gpt-4.1-nano")
        self.assertEqual(request["max_output_tokens"], 36)
        self.assertEqual(request["prompt_cache_key"], "twinr:first_word:gpt-4.1-nano:de")
        self.assertIn("Fast first-word instructions", request["instructions"])
        self.assertEqual(request["text"]["format"]["name"], "twinr_first_word_reply")

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
        self.assertEqual(backend.build_tools_calls, [(False, "gpt-5.2")])

    def test_continue_turn_streaming_passes_model_into_web_search_tool_build(self) -> None:
        backend = FakeToolBackend(self.config)
        backend._client.responses.stream_results.append(
            (
                [],
                SimpleNamespace(
                    id="resp_continue_search_1",
                    _request_id="req_continue_search_1",
                    model="gpt-5.2",
                    output_text="Ich habe es gefunden.",
                    output=[],
                    usage=None,
                ),
            )
        )
        provider = OpenAIToolCallingAgentProvider(backend)

        response = provider.continue_turn_streaming(
            continuation_token="resp_start_2",
            tool_results=(
                AgentToolResult(
                    call_id="call_2",
                    name="web_lookup",
                    output={"status": "ok"},
                    serialized_output='{"status":"ok"}',
                ),
            ),
            allow_web_search=True,
        )

        request = backend._client.responses.stream_requests[0]
        self.assertEqual(response.text, "Ich habe es gefunden.")
        self.assertEqual(backend.build_tools_calls, [(True, "gpt-5.2")])
        self.assertEqual(request["tools"], [{"type": "web_search"}])

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
                    '"kind":"search","goal":"weather tomorrow","prompt":"What is new in Schwarzenbek weather tomorrow?",'
                    '"allow_web_search":true,'
                    '"location_hint":"Schwarzenbek","date_context":"Tuesday, 2026-03-17 (Europe/Berlin)",'
                    '"context_scope":"full_context"}'
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
        self.assertEqual(decision.prompt, "What is new in Schwarzenbek weather tomorrow?")
        self.assertTrue(decision.allow_web_search)
        self.assertEqual(decision.location_hint, "Schwarzenbek")
        self.assertEqual(decision.date_context, "Tuesday, 2026-03-17 (Europe/Berlin)")
        self.assertEqual(decision.context_scope, "full_context")
        self.assertEqual(request["model"], "gpt-4o-mini")
        self.assertIn("Supervisor decision base instructions", request["instructions"])
        self.assertIn("Runtime extra", request["instructions"])
        self.assertNotIn("Tool-loop base instructions", request["instructions"])
        self.assertEqual(request["prompt_cache_key"], "twinr:supervisor_decision:gpt-4o-mini:de")
        self.assertEqual(request["max_output_tokens"], 80)
        self.assertEqual(request["text"]["format"]["type"], "json_schema")
        self.assertIn("location_hint", request["text"]["format"]["schema"]["properties"])
        self.assertIn("runtime_tool_name", request["text"]["format"]["schema"]["properties"])
        self.assertIn("runtime_tool_arguments_json", request["text"]["format"]["schema"]["properties"])
        self.assertEqual(
            set(request["text"]["format"]["schema"]["required"]),
            set(request["text"]["format"]["schema"]["properties"]),
        )

    def test_decide_accepts_handoff_without_spoken_ack(self) -> None:
        backend = FakeToolBackend(self.config)
        backend._client.responses.create_results.append(
            SimpleNamespace(
                id="resp_decide_2",
                _request_id="req_decide_2",
                model="gpt-4o-mini",
                output_text=(
                    '{"action":"handoff","spoken_ack":null,"spoken_reply":null,'
                    '"kind":"search","goal":"weather tomorrow","prompt":null,"allow_web_search":true,'
                    '"location_hint":"Schwarzenbek","date_context":"Tuesday, 2026-03-17 (Europe/Berlin)",'
                    '"context_scope":"full_context"}'
                ),
                output=[],
                usage=None,
            )
        )
        provider = OpenAISupervisorDecisionProvider(
            backend,
            model_override="gpt-4o-mini",
        )

        decision = provider.decide("Wie wird das Wetter morgen?")
        request = backend._client.responses.create_requests[0]

        self.assertEqual(decision.action, "handoff")
        self.assertIsNone(decision.spoken_ack)
        self.assertEqual(decision.kind, "search")
        self.assertIsNone(decision.prompt)
        self.assertIn("date_context", request["text"]["format"]["schema"]["properties"])
        self.assertIn("context_scope", request["text"]["format"]["schema"]["properties"])
        self.assertIn("prompt", request["text"]["format"]["schema"]["properties"])

    def test_decide_parses_runtime_local_tool_handoff_fields(self) -> None:
        backend = FakeToolBackend(self.config)
        backend._client.responses.create_results.append(
            SimpleNamespace(
                id="resp_decide_runtime_tool",
                _request_id="req_decide_runtime_tool",
                model="gpt-4o-mini",
                output_text=(
                    '{"action":"handoff","spoken_ack":"Ich schalte mich kurz stumm.","spoken_reply":null,'
                    '"kind":"automation","goal":"Set temporary quiet mode.","prompt":null,"allow_web_search":false,'
                    '"location_hint":null,"date_context":null,"context_scope":"tiny_recent",'
                    '"runtime_tool_name":"manage_voice_quiet_mode",'
                    '"runtime_tool_arguments_json":"{\\"action\\":\\"set\\",\\"duration_minutes\\":2}"}'
                ),
                output=[],
                usage=None,
            )
        )
        provider = OpenAISupervisorDecisionProvider(
            backend,
            model_override="gpt-4o-mini",
        )

        decision = provider.decide("Sei bitte 2 Minuten ruhig.")

        self.assertEqual(decision.action, "handoff")
        self.assertEqual(decision.kind, "automation")
        self.assertEqual(decision.context_scope, "tiny_recent")
        self.assertEqual(decision.runtime_tool_name, "manage_voice_quiet_mode")
        self.assertEqual(decision.runtime_tool_arguments, {"action": "set", "duration_minutes": 2})

    def test_decide_retries_once_when_structured_response_hits_max_output_tokens(self) -> None:
        backend = FakeToolBackend(self.config)
        backend._client.responses.create_results.extend(
            [
                SimpleNamespace(
                    id="resp_decide_incomplete",
                    _request_id="req_decide_incomplete",
                    model="gpt-4o-mini",
                    status="incomplete",
                    incomplete_details=SimpleNamespace(reason="max_output_tokens"),
                    output_text="",
                    output=[],
                    usage=None,
                ),
                SimpleNamespace(
                    id="resp_decide_retry",
                    _request_id="req_decide_retry",
                    model="gpt-4o-mini",
                    status="completed",
                    output_text=(
                        '{"action":"handoff","spoken_ack":"Ich schaue kurz nach.","spoken_reply":null,'
                        '"kind":"search","goal":"latest headlines","allow_web_search":true,'
                        '"location_hint":null,"date_context":null,"context_scope":"tiny_recent"}'
                    ),
                    output=[],
                    usage=None,
                ),
            ]
        )
        provider = OpenAISupervisorDecisionProvider(
            backend,
            model_override="gpt-4o-mini",
            reasoning_effort_override="low",
        )

        decision = provider.decide("Was sind die Nachrichten heute?")

        self.assertEqual(decision.action, "handoff")
        self.assertEqual(decision.kind, "search")
        self.assertEqual(len(backend._client.responses.create_requests), 2)
        self.assertEqual(backend._client.responses.create_requests[0]["max_output_tokens"], 80)
        self.assertEqual(backend._client.responses.create_requests[1]["max_output_tokens"], 160)

    def test_decide_floors_gpt5_supervisor_budget_to_512_tokens(self) -> None:
        backend = FakeToolBackend(
            TwinrConfig(
                openai_api_key="test-key",
                default_model="gpt-5.2",
                streaming_supervisor_max_output_tokens=80,
            )
        )
        backend._client.responses.create_results.append(
            SimpleNamespace(
                id="resp_decide_budget",
                _request_id="req_decide_budget",
                model="gpt-5.2-chat-latest",
                output_text=(
                    '{"action":"handoff","spoken_ack":"Ich schaue kurz nach.","spoken_reply":null,'
                    '"kind":"search","goal":"latest headlines","allow_web_search":true,'
                    '"location_hint":null,"date_context":null,"context_scope":"tiny_recent"}'
                ),
                output=[],
                usage=None,
            )
        )
        provider = OpenAISupervisorDecisionProvider(
            backend,
            model_override="gpt-5.2-chat-latest",
        )

        provider.decide("Was gibt es Neues?")

        request = backend._client.responses.create_requests[0]
        self.assertEqual(request["model"], "gpt-5.2-chat-latest")
        self.assertEqual(request["max_output_tokens"], 512)

    def test_decide_uses_extended_retry_ladder_for_gpt5_supervisor_budget_failures(self) -> None:
        backend = FakeToolBackend(
            TwinrConfig(
                openai_api_key="test-key",
                default_model="gpt-5.4-mini",
                streaming_supervisor_max_output_tokens=80,
            )
        )
        backend._client.responses.create_results.extend(
            [
                SimpleNamespace(
                    id="resp_decide_gpt5_incomplete_1",
                    _request_id="req_decide_gpt5_incomplete_1",
                    model="gpt-5.4-mini-2026-03-17",
                    status="incomplete",
                    incomplete_details=SimpleNamespace(reason="max_output_tokens"),
                    output_text="",
                    output=[],
                    usage=None,
                ),
                SimpleNamespace(
                    id="resp_decide_gpt5_incomplete_2",
                    _request_id="req_decide_gpt5_incomplete_2",
                    model="gpt-5.4-mini-2026-03-17",
                    status="incomplete",
                    incomplete_details=SimpleNamespace(reason="max_output_tokens"),
                    output_text="",
                    output=[],
                    usage=None,
                ),
                SimpleNamespace(
                    id="resp_decide_gpt5_retry_3",
                    _request_id="req_decide_gpt5_retry_3",
                    model="gpt-5.4-mini-2026-03-17",
                    status="completed",
                    output_text=(
                        '{"action":"handoff","spoken_ack":"Ich schaue kurz nach.","spoken_reply":null,'
                        '"kind":"search","goal":"latest headlines","allow_web_search":true,'
                        '"location_hint":null,"date_context":null,"context_scope":"tiny_recent"}'
                    ),
                    output=[],
                    usage=None,
                ),
            ]
        )
        provider = OpenAISupervisorDecisionProvider(
            backend,
            model_override="gpt-5.4-mini",
            reasoning_effort_override="low",
        )

        decision = provider.decide("Was gibt es Neues?")

        self.assertEqual(decision.action, "handoff")
        self.assertEqual(decision.kind, "search")
        self.assertEqual(len(backend._client.responses.create_requests), 3)
        self.assertEqual(
            [request["max_output_tokens"] for request in backend._client.responses.create_requests],
            [512, 768, 1024],
        )

    def test_decide_extracts_first_balanced_json_object_from_text_fallback(self) -> None:
        backend = FakeToolBackend(self.config)
        backend._client.responses.create_results.append(
            SimpleNamespace(
                id="resp_decide_extra",
                _request_id="req_decide_extra",
                model="gpt-5.4-mini-2026-03-17",
                status="completed",
                output_parsed=None,
                output_text=(
                    '{"action":"handoff","spoken_ack":"Ich schaue kurz nach.","spoken_reply":null,'
                    '"kind":"search","goal":"latest headlines","prompt":null,"allow_web_search":true,'
                    '"location_hint":null,"date_context":null,"context_scope":"tiny_recent"}'
                    '\nHinweis: structured output completed.'
                ),
                output=[],
                usage=None,
            )
        )
        provider = OpenAISupervisorDecisionProvider(
            backend,
            model_override="gpt-5.4-mini",
        )

        decision = provider.decide("Was gibt es Neues?")

        self.assertEqual(decision.action, "handoff")
        self.assertEqual(decision.kind, "search")
        self.assertEqual(decision.context_scope, "tiny_recent")

    def test_decide_ignores_following_json_objects_in_text_fallback(self) -> None:
        backend = FakeToolBackend(self.config)
        backend._client.responses.create_results.append(
            SimpleNamespace(
                id="resp_decide_extra_objects",
                _request_id="req_decide_extra_objects",
                model="gpt-5.4-mini-2026-03-17",
                status="completed",
                output_parsed=None,
                output_text=(
                    '{"action":"handoff","spoken_ack":"Ich schaue kurz nach.","spoken_reply":null,'
                    '"kind":"memory","goal":"check remembered contact details","prompt":null,'
                    '"allow_web_search":false,"location_hint":null,"date_context":null,'
                    '"context_scope":"full_context"}'
                    '\n'
                    '{"debug":"duplicate fragment"}'
                ),
                output=[],
                usage=None,
            )
        )
        provider = OpenAISupervisorDecisionProvider(
            backend,
            model_override="gpt-5.4-mini",
        )

        decision = provider.decide("Wie ist die Telefonnummer von Anna Schulz?")

        self.assertEqual(decision.action, "handoff")
        self.assertEqual(decision.kind, "memory")
        self.assertFalse(decision.allow_web_search)
        self.assertEqual(decision.context_scope, "full_context")

    def test_decide_prefers_sdk_parsed_mapping_when_output_text_is_invalid(self) -> None:
        backend = FakeToolBackend(self.config)
        backend._client.responses.create_results.append(
            SimpleNamespace(
                id="resp_decide_parsed",
                _request_id="req_decide_parsed",
                model="gpt-5.4-mini-2026-03-17",
                status="completed",
                output_parsed={
                    "action": "handoff",
                    "spoken_ack": "Ich schaue kurz nach.",
                    "spoken_reply": None,
                    "kind": "memory",
                    "goal": "Check stored conflicts.",
                    "prompt": None,
                    "allow_web_search": False,
                    "location_hint": None,
                    "date_context": None,
                    "context_scope": "full_context",
                },
                output_text='{"broken": true} trailing',
                output=[],
                usage=None,
            )
        )
        provider = OpenAISupervisorDecisionProvider(
            backend,
            model_override="gpt-5.4-mini",
        )

        decision = provider.decide("Gibt es offene Erinnerungskonflikte?")

        self.assertEqual(decision.action, "handoff")
        self.assertEqual(decision.kind, "memory")
        self.assertEqual(decision.context_scope, "full_context")


class OpenAIConversationClosureDecisionProviderTests(unittest.TestCase):
    def test_decide_uses_structured_json_schema_and_closure_config(self) -> None:
        backend = FakeToolBackend(
            TwinrConfig(
                openai_api_key="test-key",
                default_model="gpt-5.2",
                conversation_closure_model="gpt-4o-mini",
                conversation_closure_max_output_tokens=48,
            )
        )
        backend._client.responses.create_results.append(
            SimpleNamespace(
                id="resp_closure_1",
                _request_id="req_closure_1",
                model="gpt-4o-mini",
                output_text='{"close_now":false,"confidence":0.84,"reason":"still_engaged","matched_topics":["AI companions"]}',
                output=[],
                usage=None,
            )
        )
        provider = OpenAIConversationClosureDecisionProvider(
            backend,
            model_override="gpt-4o-mini",
            base_instructions_override="Closure controller instructions",
            replace_base_instructions=True,
        )

        decision = provider.decide(
            "closure prompt",
            conversation=(("assistant", "Voriger Turn"),),
            instructions="Runtime closure extra",
            timeout_seconds=7.5,
        )

        request = backend._client.responses.create_requests[0]
        self.assertFalse(decision.close_now)
        self.assertAlmostEqual(decision.confidence, 0.84)
        self.assertEqual(decision.reason, "still_engaged")
        self.assertEqual(decision.matched_topics, ("AI companions",))
        self.assertEqual(request["model"], "gpt-4o-mini")
        self.assertEqual(request["max_output_tokens"], 48)
        self.assertEqual(request["timeout"], 7.5)
        self.assertIn("Closure controller instructions", request["instructions"])
        self.assertIn("Runtime closure extra", request["instructions"])
        self.assertEqual(request["prompt_cache_key"], "twinr:conversation_closure:gpt-4o-mini:de")
        self.assertEqual(request["text"]["format"]["type"], "json_schema")
        self.assertEqual(request["text"]["format"]["name"], "twinr_conversation_closure_decision")
