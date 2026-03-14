from pathlib import Path
from types import SimpleNamespace
import sys
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

    def create(self, **kwargs):
        self.calls.append(kwargs)
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

    def test_agent_text_provider_delegates_web_search_to_support_provider(self) -> None:
        provider = GroqAgentTextProvider(self.config, support_provider=self.support, client=self.client)

        response = provider.respond_with_metadata("Wie ist das Wetter?", allow_web_search=True)

        self.assertEqual(response.text, "support metadata")
        self.assertEqual(self.support.calls[0][0], "respond_with_metadata")

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
