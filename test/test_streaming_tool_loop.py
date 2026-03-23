from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.contracts import AgentToolCall, ToolCallingTurnResponse
from twinr.agent.tools.runtime.streaming_loop import ToolCallingStreamingLoop


class FakeToolCallingProvider:
    def __init__(self) -> None:
        self.config = object()
        self.start_calls: list[tuple[str, tuple[tuple[str, str], ...] | None]] = []
        self.continue_calls: list[tuple[str, tuple[str, ...]]] = []

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
        del instructions, tool_schemas, allow_web_search
        self.start_calls.append((prompt, conversation))
        if on_text_delta is not None:
            on_text_delta("Ich prüfe das. ")
        return ToolCallingTurnResponse(
            text="Ich prüfe das.",
            tool_calls=(
                AgentToolCall(
                    name="search_live_info",
                    call_id="call_search_1",
                    arguments={"question": prompt},
                    raw_arguments='{"question":"%s"}' % prompt,
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
        del instructions, tool_schemas, allow_web_search
        self.continue_calls.append(
            (
                continuation_token,
                tuple(result.serialized_output for result in tool_results),
            )
        )
        if on_text_delta is not None:
            on_text_delta("Der Bus fährt um 07:30 Uhr.")
        return ToolCallingTurnResponse(
            text="Der Bus fährt um 07:30 Uhr.",
            response_id="resp_continue_1",
            used_web_search=True,
        )


class ToolCallingStreamingLoopTests(unittest.TestCase):
    def test_runs_tool_calls_and_continues_generation(self) -> None:
        provider = FakeToolCallingProvider()
        seen_arguments: list[dict[str, object]] = []

        loop = ToolCallingStreamingLoop(
            provider,
            tool_handlers={
                "search_live_info": lambda arguments: seen_arguments.append(arguments)
                or {"status": "ok", "answer": "Bus 24 fährt um 07:30 Uhr."}
            },
            tool_schemas=[{"type": "function", "name": "search_live_info"}],
        )

        text_deltas: list[str] = []
        result = loop.run(
            "Wann fährt der Bus?",
            conversation=(("system", "Stay helpful"),),
            on_text_delta=text_deltas.append,
        )

        self.assertEqual(provider.start_calls[0][0], "Wann fährt der Bus?")
        self.assertEqual(seen_arguments, [{"question": "Wann fährt der Bus?"}])
        self.assertEqual(provider.continue_calls[0][0], "resp_start_1")
        self.assertEqual(result.rounds, 2)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(len(result.tool_results), 1)
        self.assertTrue(result.used_web_search)
        self.assertIn("Ich prüfe das.", result.text)
        self.assertIn("Der Bus fährt um 07:30 Uhr.", result.text)
        self.assertEqual(
            text_deltas,
            ["Ich prüfe das. ", "Der Bus fährt um 07:30 Uhr."],
        )

    def test_handler_can_opt_in_to_full_tool_call_object(self) -> None:
        provider = FakeToolCallingProvider()
        seen_call_ids: list[str] = []

        def handler(tool_call: AgentToolCall):
            seen_call_ids.append(tool_call.call_id)
            return {"status": "ok", "answer": "Bus 24 fährt um 07:30 Uhr."}

        setattr(handler, "_twinr_accepts_tool_call", True)

        loop = ToolCallingStreamingLoop(
            provider,
            tool_handlers={"search_live_info": handler},
            tool_schemas=[{"type": "function", "name": "search_live_info"}],
        )

        loop.run("Wann fährt der Bus?")

        self.assertEqual(seen_call_ids, ["call_search_1"])

    def test_should_stop_prevents_tool_execution_after_provider_return(self) -> None:
        provider = FakeToolCallingProvider()
        seen_arguments: list[dict[str, object]] = []
        stop_requested = {"value": False}

        class _StoppingProvider(FakeToolCallingProvider):
            def start_turn_streaming(self, prompt: str, **kwargs) -> ToolCallingTurnResponse:
                response = super().start_turn_streaming(prompt, **kwargs)
                stop_requested["value"] = True
                return response

        provider = _StoppingProvider()
        loop = ToolCallingStreamingLoop(
            provider,
            tool_handlers={
                "search_live_info": lambda arguments: seen_arguments.append(arguments)
                or {"status": "ok", "answer": "Bus 24 fährt um 07:30 Uhr."}
            },
            tool_schemas=[{"type": "function", "name": "search_live_info"}],
        )

        with self.assertRaises(InterruptedError):
            loop.run(
                "Wann fährt der Bus?",
                should_stop=lambda: stop_requested["value"],
            )

        self.assertEqual(seen_arguments, [])


if __name__ == "__main__":
    unittest.main()
