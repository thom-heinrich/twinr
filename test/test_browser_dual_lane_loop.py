from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.contracts import AgentToolCall, ToolCallingTurnResponse
from twinr.agent.tools.runtime.dual_lane_loop import DualLaneToolLoop
from twinr.agent.tools.runtime.speech_lane import SpeechLaneDelta


class _BrowserAwareSupervisorProvider:
    def __init__(self) -> None:
        self.config = TwinrConfig(
            openai_api_key="test-key",
            project_root="/tmp/test-browser-dual-lane",
            personality_dir="personality",
        )

    def start_turn_streaming(self, prompt: str, **kwargs) -> ToolCallingTurnResponse:
        del prompt, kwargs
        raise AssertionError("run_handoff_only should not enter the supervisor provider")

    def continue_turn_streaming(self, **kwargs) -> ToolCallingTurnResponse:
        del kwargs
        raise AssertionError("run_handoff_only should not continue the supervisor provider")


class _BrowserAwareSpecialistProvider:
    def __init__(self) -> None:
        self.config = TwinrConfig(
            openai_api_key="test-key",
            project_root="/tmp/test-browser-dual-lane",
            personality_dir="personality",
        )
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
                    name="browser_automation",
                    call_id="browser_1",
                    arguments={
                        "goal": "Check whether the practice website lists today's opening hours.",
                        "start_url": "https://example.org/hours",
                        "allowed_domains": ["example.org"],
                    },
                ),
            ),
            response_id="worker_browser_start",
            continuation_token="worker_browser_start",
            model="gpt-5.2-chat-latest",
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
        del on_text_delta
        self.continue_calls.append(
            {
                "continuation_token": continuation_token,
                "tool_results": list(tool_results),
                "instructions": instructions,
                "tool_schemas": list(tool_schemas),
                "allow_web_search": allow_web_search,
            }
        )
        return ToolCallingTurnResponse(
            text="Die Website nennt heute geänderte Öffnungszeiten bis sechzehn Uhr.",
            response_id="worker_browser_done",
            model="gpt-5.2-chat-latest",
            used_web_search=True,
        )


class BrowserDualLaneLoopTests(unittest.TestCase):
    def test_search_handoff_enters_specialist_loop_when_browser_tool_is_available(self) -> None:
        specialist = _BrowserAwareSpecialistProvider()
        browser_calls: list[dict[str, object]] = []
        search_calls: list[dict[str, object]] = []
        lane_events: list[SpeechLaneDelta] = []

        def _handle_search(arguments: dict[str, object] | AgentToolCall) -> dict[str, object]:
            if isinstance(arguments, AgentToolCall):
                search_calls.append(dict(arguments.arguments))
            else:
                search_calls.append(dict(arguments))
            return {"answer": "8 Grad"}

        def _handle_browser(arguments: dict[str, object] | AgentToolCall) -> dict[str, object]:
            if isinstance(arguments, AgentToolCall):
                browser_calls.append(dict(arguments.arguments))
            else:
                browser_calls.append(dict(arguments))
            return {
                "status": "completed",
                "ok": True,
                "summary": "The site lists today's hours.",
                "final_url": "https://example.org/hours",
                "used_web_search": True,
            }

        loop = DualLaneToolLoop(
            supervisor_provider=_BrowserAwareSupervisorProvider(),
            specialist_provider=specialist,
            tool_handlers={
                "search_live_info": _handle_search,
                "browser_automation": _handle_browser,
            },
            tool_schemas=[
                {"type": "function", "name": "search_live_info"},
                {"type": "function", "name": "browser_automation"},
            ],
            supervisor_instructions="Supervisor instructions",
            specialist_instructions="Specialist instructions",
        )

        result = loop.run_handoff_only(
            "Hat die Praxis heute offen?",
            handoff=SimpleNamespace(
                action="handoff",
                spoken_ack="Das kann einen Moment dauern, ich melde mich gleich.",
                kind="search",
                goal="Check the practice website for today's opening hours.",
                allow_web_search=True,
                response_id="prefetch_resp",
                request_id="prefetch_req",
                model="gpt-4o-mini",
                token_usage=None,
            ),
            on_lane_text_delta=lane_events.append,
        )

        self.assertEqual(search_calls, [])
        self.assertEqual(
            browser_calls,
            [
                {
                    "goal": "Check whether the practice website lists today's opening hours.",
                    "start_url": "https://example.org/hours",
                    "allowed_domains": ["example.org"],
                }
            ],
        )
        self.assertEqual(len(specialist.start_calls), 1)
        self.assertFalse(specialist.start_calls[0]["allow_web_search"])
        self.assertFalse(specialist.continue_calls[0]["allow_web_search"])
        self.assertEqual(result.text, "Die Website nennt heute geänderte Öffnungszeiten bis sechzehn Uhr.")
        self.assertEqual(
            lane_events,
            [
                SpeechLaneDelta(
                    text="Das kann einen Moment dauern, ich melde mich gleich.",
                    lane="filler",
                    replace_current=False,
                ),
                SpeechLaneDelta(
                    text="Die Website nennt heute geänderte Öffnungszeiten bis sechzehn Uhr.",
                    lane="final",
                    replace_current=True,
                    atomic=True,
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
