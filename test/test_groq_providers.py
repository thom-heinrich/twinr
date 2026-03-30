from pathlib import Path
from types import SimpleNamespace
import sys
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.contracts import AgentToolResult
from twinr.providers.groq import GroqAgentTextProvider, GroqToolCallingAgentProvider


class FakeSupportProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    @property
    def config(self):
        return None

    @config.setter
    def config(self, value) -> None:
        del value

    def respond_streaming(self, prompt: str, **kwargs):
        self.calls.append(("respond_streaming", (prompt, kwargs)))
        return SimpleNamespace(text="support streaming", response_id=None, request_id=None, model="openai", token_usage=None, used_web_search=True)

    def respond_with_metadata(self, prompt: str, **kwargs):
        self.calls.append(("respond_with_metadata", (prompt, kwargs)))
        return SimpleNamespace(text="support metadata", response_id=None, request_id=None, model="openai", token_usage=None, used_web_search=bool(kwargs.get("allow_web_search")))

    def respond_to_images_with_metadata(self, prompt: str, **kwargs):
        self.calls.append(("respond_to_images_with_metadata", (prompt, kwargs)))
        return SimpleNamespace(text="vision", response_id=None, request_id=None, model="openai", token_usage=None, used_web_search=False)

    def search_live_info_with_metadata(self, question: str, **kwargs):
        self.calls.append(("search_live_info_with_metadata", (question, kwargs)))
        return SimpleNamespace(answer="answer", sources=(), response_id=None, request_id=None, model="openai", token_usage=None, used_web_search=True)

    def compose_print_job_with_metadata(self, **kwargs):
        self.calls.append(("compose_print_job_with_metadata", kwargs))
        return SimpleNamespace(text="print", response_id=None, request_id=None, model="openai", token_usage=None, used_web_search=False)

    def phrase_due_reminder_with_metadata(self, reminder, **kwargs):
        self.calls.append(("phrase_due_reminder_with_metadata", (reminder, kwargs)))
        return SimpleNamespace(text="reminder", response_id=None, request_id=None, model="openai", token_usage=None, used_web_search=False)

    def phrase_proactive_prompt_with_metadata(self, **kwargs):
        self.calls.append(("phrase_proactive_prompt_with_metadata", kwargs))
        return SimpleNamespace(text="proactive", response_id=None, request_id=None, model="openai", token_usage=None, used_web_search=False)

    def fulfill_automation_prompt_with_metadata(self, prompt: str, **kwargs):
        self.calls.append(("fulfill_automation_prompt_with_metadata", (prompt, kwargs)))
        return SimpleNamespace(text="automation", response_id=None, request_id=None, model="openai", token_usage=None, used_web_search=False)


class FakeGroqChatCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.results: list[object] = []
        self.delay_seconds = 0.0

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)
        if not self.results:
            raise AssertionError("No fake Groq result configured")
        return self.results.pop(0)


class FakeGroqClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=FakeGroqChatCompletions())


class GroqProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = TwinrConfig(
            groq_api_key="groq-key",
            groq_model="llama-3.1-8b-instant",
        )
        self.support = FakeSupportProvider()
        self.client = FakeGroqClient()

    def test_agent_text_provider_streams_content(self) -> None:
        self.client.chat.completions.results.append(
            iter(
                [
                    SimpleNamespace(
                        id="groq_resp_1",
                        _request_id="groq_req_1",
                        model="llama-3.1-8b-instant",
                        choices=[SimpleNamespace(delta=SimpleNamespace(content="Hallo "))],
                    ),
                    SimpleNamespace(
                        id="groq_resp_1",
                        _request_id="groq_req_1",
                        model="llama-3.1-8b-instant",
                        choices=[SimpleNamespace(delta=SimpleNamespace(content="Thom"))],
                    ),
                ]
            )
        )
        provider = GroqAgentTextProvider(self.config, support_provider=self.support, client=self.client)
        deltas: list[str] = []

        response = provider.respond_streaming("Sag hallo", on_text_delta=deltas.append)

        self.assertEqual(response.text, "Hallo Thom")
        self.assertEqual(deltas, ["Hallo ", "Thom"])
        self.assertEqual(self.client.chat.completions.calls[0]["model"], "llama-3.1-8b-instant")
        self.assertTrue(self.client.chat.completions.calls[0]["stream"])

    def test_agent_text_provider_uses_native_groq_web_search_by_default(self) -> None:
        self.client.chat.completions.results.append(
            SimpleNamespace(
                id="groq_search_1",
                _request_id="groq_req_search_1",
                model="groq/compound-mini",
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="Heute wird es sonnig.",
                            executed_tools=[{"type": "web_search"}],
                        )
                    )
                ],
            )
        )
        provider = GroqAgentTextProvider(self.config, support_provider=self.support, client=self.client)

        response = provider.respond_with_metadata("Wie ist das Wetter?", allow_web_search=True)

        self.assertEqual(response.text, "Heute wird es sonnig.")
        self.assertTrue(response.used_web_search)
        self.assertEqual(self.support.calls, [])
        self.assertEqual(self.client.chat.completions.calls[0]["model"], "groq/compound-mini")

    def test_agent_text_provider_omits_service_tier_when_not_configured(self) -> None:
        self.client.chat.completions.results.append(
            SimpleNamespace(
                id="groq_resp_omit_tier",
                _request_id="groq_req_omit_tier",
                model="llama-3.1-8b-instant",
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            )
        )
        provider = GroqAgentTextProvider(self.config, support_provider=self.support, client=self.client)

        response = provider.respond_with_metadata("Antworte nur mit: ok.")

        self.assertEqual(response.text, "ok")
        self.assertNotIn("service_tier", self.client.chat.completions.calls[0])

    def test_agent_text_provider_marks_explicit_search_path_as_web_search_without_sdk_metadata(self) -> None:
        self.client.chat.completions.results.append(
            SimpleNamespace(
                id="groq_search_2",
                _request_id="groq_req_search_2",
                model="groq/compound-mini",
                choices=[SimpleNamespace(message=SimpleNamespace(content="Es ist 2026."))],
            )
        )
        provider = GroqAgentTextProvider(self.config, support_provider=self.support, client=self.client)

        response = provider.respond_with_metadata("Welches Jahr haben wir?", allow_web_search=True)

        self.assertEqual(response.text, "Es ist 2026.")
        self.assertTrue(response.used_web_search)

    def test_agent_text_provider_can_opt_in_to_support_fallback_for_web_search(self) -> None:
        provider = GroqAgentTextProvider(
            TwinrConfig(
                groq_api_key="groq-key",
                groq_model="llama-3.1-8b-instant",
                groq_allow_search_fallback=True,
            ),
            support_provider=self.support,
            client=self.client,
        )

        response = provider.respond_with_metadata("Wie ist das Wetter?", allow_web_search=True)

        self.assertEqual(response.text, "support metadata")
        self.assertEqual(self.support.calls[0][0], "respond_with_metadata")

    def test_agent_text_provider_enforces_wall_clock_timeout_for_native_search(self) -> None:
        self.client.chat.completions.delay_seconds = 0.2
        self.client.chat.completions.results.append(
            SimpleNamespace(
                id="groq_search_timeout",
                _request_id="groq_req_search_timeout",
                model="groq/compound-mini",
                choices=[SimpleNamespace(message=SimpleNamespace(content="zu spät"))],
            )
        )
        provider = GroqAgentTextProvider(
            TwinrConfig(
                groq_api_key="groq-key",
                groq_model="llama-3.1-8b-instant",
                groq_request_timeout_seconds=0.05,
            ),
            support_provider=self.support,
            client=self.client,
        )

        started = time.monotonic()
        response = provider.respond_with_metadata("Wie spät ist es?", allow_web_search=True)
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.15)
        self.assertEqual(response.text, "I am having trouble right now. Please try again.")
        self.assertEqual(self.support.calls, [])

    def test_tool_calling_provider_streams_and_continues(self) -> None:
        self.client.chat.completions.results.append(
            SimpleNamespace(
                id="groq_tool_1",
                _request_id="groq_req_2",
                model="llama-3.1-8b-instant",
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="Ich prüfe das.",
                            tool_calls=[
                                SimpleNamespace(
                                    id="call_print_1",
                                    function=SimpleNamespace(
                                        name="print_receipt",
                                        arguments='{"focus_hint":"Arzttermin"}',
                                    ),
                                )
                            ],
                        )
                    )
                ],
            )
        )
        self.client.chat.completions.results.append(
            SimpleNamespace(
                id="groq_tool_2",
                _request_id="groq_req_3",
                model="llama-3.1-8b-instant",
                choices=[SimpleNamespace(message=SimpleNamespace(content="Ist erledigt.", tool_calls=[]))],
            )
        )
        provider = GroqToolCallingAgentProvider(self.config, client=self.client)
        deltas: list[str] = []

        start = provider.start_turn_streaming(
            "Bitte drucke das",
            tool_schemas=[{"type": "function", "name": "print_receipt", "parameters": {"type": "object"}}],
            on_text_delta=deltas.append,
        )
        self.assertEqual(start.text, "Ich prüfe das.")
        self.assertEqual(start.tool_calls[0].name, "print_receipt")
        self.assertEqual(start.tool_calls[0].arguments["focus_hint"], "Arzttermin")
        self.assertTrue(start.continuation_token)

        finish = provider.continue_turn_streaming(
            continuation_token=start.continuation_token or "",
            tool_results=(
                AgentToolResult(
                    call_id="call_print_1",
                    name="print_receipt",
                    output={"status": "printed"},
                    serialized_output='{"status":"printed"}',
                ),
            ),
            tool_schemas=[{"type": "function", "name": "print_receipt", "parameters": {"type": "object"}}],
            on_text_delta=deltas.append,
        )

        self.assertEqual(finish.text, "Ist erledigt.")
        self.assertEqual(deltas, ["Ist erledigt."])
        self.assertNotIn("stream", self.client.chat.completions.calls[0])
        continue_request = self.client.chat.completions.calls[1]
        self.assertEqual(continue_request["messages"][-1]["role"], "tool")
        self.assertEqual(continue_request["messages"][-1]["tool_call_id"], "call_print_1")

    def test_groq_messages_fold_conversation_system_entries_into_one_system_prompt(self) -> None:
        self.client.chat.completions.results.append(
            SimpleNamespace(
                id="groq_tool_3",
                _request_id="groq_req_4",
                model="llama-3.1-8b-instant",
                choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=[]))],
            )
        )
        provider = GroqToolCallingAgentProvider(self.config, client=self.client)

        provider.start_turn_streaming(
            "Bitte drucke das",
            conversation=(
                ("system", "Twinr memory summary:\\n- Arzttermin morgen um 12 Uhr"),
                ("assistant", "Vorherige Antwort"),
            ),
            instructions="Speak German.",
            tool_schemas=[{"type": "function", "name": "print_receipt", "description": "Print", "parameters": {"type": "object"}}],
        )

        request_messages = self.client.chat.completions.calls[0]["messages"]
        self.assertEqual(request_messages[0]["role"], "system")
        self.assertIn("Speak German.", request_messages[0]["content"])
        self.assertIn("Twinr memory summary", request_messages[0]["content"])
        self.assertEqual([message["role"] for message in request_messages[1:]], ["assistant", "user"])


if __name__ == "__main__":
    unittest.main()
