from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.contracts import AgentToolCall, ToolCallingTurnResponse
from twinr.agent.tools.dual_lane_loop import DualLaneToolLoop, SpeechLaneDelta


class FakeSupervisorProvider:
    def __init__(self) -> None:
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
            on_text_delta("Ich schaue kurz nach.")
        return ToolCallingTurnResponse(
            text="Ich schaue kurz nach.",
            tool_calls=(
                AgentToolCall(
                    name="handoff_specialist_worker",
                    call_id="handoff_1",
                    arguments={
                        "kind": "search",
                        "goal": "Find the weather for tomorrow in Schwarzenbek.",
                        "spoken_ack": "Ich schaue kurz nach.",
                        "allow_web_search": True,
                    },
                ),
            ),
            response_id="sup_start",
            continuation_token="sup_start",
            model="gpt-4o-mini",
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
            on_text_delta(" Morgen wird es kühler und trocken.")
        return ToolCallingTurnResponse(
            text="Morgen wird es kühler und trocken.",
            response_id="sup_done",
            model="gpt-4o-mini",
        )


class FakeSpecialistProvider:
    def __init__(self) -> None:
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
        return ToolCallingTurnResponse(
            text="",
            tool_calls=(
                AgentToolCall(
                    name="search_live_info",
                    call_id="search_1",
                    arguments={"question": "weather tomorrow in Schwarzenbek"},
                ),
            ),
            response_id="worker_start",
            continuation_token="worker_start",
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
            text="Morgen 8 Grad, trocken und leicht bewölkt.",
            response_id="worker_done",
            model="gpt-5.2-chat-latest",
            used_web_search=True,
        )


class FakeSupervisorDecisionProvider:
    def __init__(self, decision: dict[str, object]) -> None:
        self.decision = decision
        self.calls: list[dict[str, object]] = []

    def decide(self, prompt: str, *, conversation=None, instructions=None):
        self.calls.append(
            {
                "prompt": prompt,
                "conversation": conversation,
                "instructions": instructions,
            }
        )
        return SimpleNamespace(
            action=self.decision.get("action", "handoff"),
            spoken_ack=self.decision.get("spoken_ack"),
            spoken_reply=self.decision.get("spoken_reply"),
            kind=self.decision.get("kind"),
            goal=self.decision.get("goal"),
            allow_web_search=self.decision.get("allow_web_search"),
            response_id="decision_1",
            request_id="req_decision_1",
            model="gpt-4o-mini",
            token_usage=None,
        )


class DualLaneLoopTests(unittest.TestCase):
    def test_prefetched_handoff_emits_filler_then_preempting_final_lane(self) -> None:
        supervisor = FakeSupervisorProvider()
        specialist = FakeSpecialistProvider()
        lane_events: list[SpeechLaneDelta] = []

        loop = DualLaneToolLoop(
            supervisor_provider=supervisor,
            specialist_provider=specialist,
            tool_handlers={
                "search_live_info": lambda arguments: {"answer": "8 Grad", "arguments": arguments},
            },
            tool_schemas=[{"type": "function", "name": "search_live_info"}],
            supervisor_instructions="Supervisor instructions",
            specialist_instructions="Specialist instructions",
        )

        result = loop.run(
            "Wie wird das Wetter morgen in Schwarzenbek?",
            prefetched_decision=SimpleNamespace(
                action="handoff",
                spoken_ack="Ich schaue kurz nach und sage dir gleich Bescheid.",
                kind="search",
                goal="Find the weather for tomorrow in Schwarzenbek.",
                allow_web_search=True,
                response_id="decision_1",
                request_id="req_decision_1",
                model="gpt-4o-mini",
                token_usage=None,
            ),
            on_lane_text_delta=lane_events.append,
        )

        self.assertEqual(
            lane_events,
            [
                SpeechLaneDelta(
                    text="Ich schaue kurz nach und sage dir gleich Bescheid.",
                    lane="filler",
                    replace_current=False,
                ),
                SpeechLaneDelta(
                    text="8 Grad",
                    lane="final",
                    replace_current=True,
                    atomic=True,
                ),
            ],
        )
        self.assertEqual(result.text, "8 Grad")
        self.assertTrue(result.used_web_search)
        self.assertEqual(specialist.start_calls, [])

    def test_search_handoff_runs_direct_search_worker_and_preserves_streamed_ack(self) -> None:
        supervisor = FakeSupervisorProvider()
        specialist = FakeSpecialistProvider()
        streamed: list[str] = []
        search_calls: list[dict[str, object]] = []

        loop = DualLaneToolLoop(
            supervisor_provider=supervisor,
            specialist_provider=specialist,
            tool_handlers={
                "search_live_info": lambda arguments: search_calls.append(arguments) or {"answer": "8 Grad"},
            },
            tool_schemas=[{"type": "function", "name": "search_live_info"}],
            supervisor_instructions="Supervisor instructions",
            specialist_instructions="Specialist instructions",
        )

        result = loop.run(
            "Wie wird das Wetter morgen in Schwarzenbek?",
            conversation=(("system", "Stay calm"),),
            instructions="Extra instructions",
            allow_web_search=False,
            on_text_delta=streamed.append,
        )

        self.assertEqual(streamed, ["Ich schaue kurz nach.", " Morgen wird es kühler und trocken."])
        self.assertEqual(result.text, "Ich schaue kurz nach.\nMorgen wird es kühler und trocken.")
        self.assertTrue(result.used_web_search)
        self.assertEqual(len(search_calls), 1)
        self.assertEqual(supervisor.start_calls[0]["tool_schemas"][-1]["name"], "handoff_specialist_worker")
        self.assertEqual(specialist.start_calls, [])
        self.assertGreaterEqual(len(result.tool_calls), 2)

    def test_handoff_tool_can_emit_immediate_ack_without_supervisor_text(self) -> None:
        class HandoffOnlySupervisor(FakeSupervisorProvider):
            def start_turn_streaming(self, prompt: str, **kwargs) -> ToolCallingTurnResponse:
                self.start_calls.append(
                    {
                        "prompt": prompt,
                        "conversation": kwargs.get("conversation"),
                        "instructions": kwargs.get("instructions"),
                        "tool_schemas": list(kwargs.get("tool_schemas", ())),
                        "allow_web_search": kwargs.get("allow_web_search"),
                    }
                )
                return ToolCallingTurnResponse(
                    text="",
                    tool_calls=(
                        AgentToolCall(
                            name="handoff_specialist_worker",
                            call_id="handoff_2",
                            arguments={
                                "kind": "search",
                                "goal": "Find the weather for tomorrow in Schwarzenbek.",
                                "spoken_ack": "Ich schaue kurz nach.",
                                "allow_web_search": True,
                            },
                        ),
                    ),
                    response_id="sup_start_2",
                    continuation_token="sup_start_2",
                    model="gpt-4o-mini",
                )

        supervisor = HandoffOnlySupervisor()
        specialist = FakeSpecialistProvider()
        streamed: list[str] = []

        loop = DualLaneToolLoop(
            supervisor_provider=supervisor,
            specialist_provider=specialist,
            tool_handlers={
                "search_live_info": lambda arguments: {"answer": "8 Grad", "arguments": arguments},
            },
            tool_schemas=[{"type": "function", "name": "search_live_info"}],
            supervisor_instructions="Supervisor instructions",
            specialist_instructions="Specialist instructions",
        )

        result = loop.run(
            "Wie wird das Wetter morgen in Schwarzenbek?",
            conversation=(("system", "Stay calm"),),
            instructions="Extra instructions",
            allow_web_search=False,
            on_text_delta=streamed.append,
        )

        self.assertEqual(streamed[0], "Ich schaue kurz nach.")
        self.assertTrue(result.used_web_search)
        self.assertEqual(result.text, "Morgen wird es kühler und trocken.")
        self.assertEqual(len(specialist.start_calls), 0)

    def test_handoff_tool_ack_does_not_duplicate_existing_supervisor_text(self) -> None:
        supervisor = FakeSupervisorProvider()
        specialist = FakeSpecialistProvider()
        streamed: list[str] = []

        loop = DualLaneToolLoop(
            supervisor_provider=supervisor,
            specialist_provider=specialist,
            tool_handlers={
                "search_live_info": lambda arguments: {"answer": "8 Grad", "arguments": arguments},
            },
            tool_schemas=[{"type": "function", "name": "search_live_info"}],
            supervisor_instructions="Supervisor instructions",
            specialist_instructions="Specialist instructions",
        )

        result = loop.run(
            "Wie wird das Wetter morgen in Schwarzenbek?",
            on_text_delta=streamed.append,
        )

        self.assertEqual(streamed.count("Ich schaue kurz nach."), 1)
        self.assertIn("Morgen wird es kühler und trocken.", result.text)

    def test_dual_lane_loop_can_use_distinct_supervisor_and_specialist_contexts(self) -> None:
        supervisor = FakeSupervisorProvider()
        specialist = FakeSpecialistProvider()
        loop = DualLaneToolLoop(
            supervisor_provider=supervisor,
            specialist_provider=specialist,
            tool_handlers={
                "search_live_info": lambda arguments: {"answer": "8 Grad", "arguments": arguments},
            },
            tool_schemas=[{"type": "function", "name": "search_live_info"}],
            supervisor_instructions="Supervisor instructions",
            specialist_instructions="Specialist instructions",
        )

        loop.run(
            "Wie wird das Wetter morgen in Schwarzenbek?",
            conversation=(("user", "fallback"),),
            supervisor_conversation=(("user", "kurzer supervisor-kontext"),),
            specialist_conversation=(("system", "reicher specialist-kontext"), ("user", "voller suchkontext")),
        )

        self.assertEqual(supervisor.start_calls[0]["conversation"], (("user", "kurzer supervisor-kontext"),))
        self.assertEqual(specialist.start_calls, [])

    def test_direct_supervisor_answer_skips_specialist(self) -> None:
        class DirectSupervisor(FakeSupervisorProvider):
            def start_turn_streaming(self, *args, **kwargs) -> ToolCallingTurnResponse:
                on_text_delta = kwargs.get("on_text_delta")
                if on_text_delta is not None:
                    on_text_delta("Gerne.")
                return ToolCallingTurnResponse(
                    text="Gerne.",
                    response_id="sup_direct",
                    model="gpt-4o-mini",
                )

        supervisor = DirectSupervisor()
        specialist = FakeSpecialistProvider()
        loop = DualLaneToolLoop(
            supervisor_provider=supervisor,
            specialist_provider=specialist,
            tool_handlers={},
            tool_schemas=[],
            supervisor_instructions="Supervisor instructions",
            specialist_instructions="Specialist instructions",
        )

        result = loop.run("Sag einfach hallo.")

        self.assertEqual(result.text, "Gerne.")
        self.assertEqual(len(specialist.start_calls), 0)

    def test_structured_supervisor_decision_can_handoff_without_tool_loop_supervisor(self) -> None:
        supervisor = FakeSupervisorProvider()
        specialist = FakeSpecialistProvider()
        decision_provider = FakeSupervisorDecisionProvider(
            {
                "action": "handoff",
                "spoken_ack": "Einen Moment bitte.",
                "kind": "search",
                "goal": "Weather tomorrow in Schwarzenbek.",
                "allow_web_search": True,
            }
        )
        streamed: list[str] = []
        loop = DualLaneToolLoop(
            supervisor_provider=supervisor,
            specialist_provider=specialist,
            supervisor_decision_provider=decision_provider,
            tool_handlers={
                "search_live_info": lambda arguments: {"answer": "8 Grad", "arguments": arguments},
            },
            tool_schemas=[{"type": "function", "name": "search_live_info"}],
            supervisor_instructions="Supervisor instructions",
            specialist_instructions="Specialist instructions",
        )

        result = loop.run(
            "Wie wird das Wetter morgen in Schwarzenbek?",
            supervisor_conversation=(("user", "kurzer kontext"),),
            specialist_conversation=(("user", "voller suchkontext"),),
            on_text_delta=streamed.append,
        )

        self.assertEqual(streamed[0], "Einen Moment bitte.")
        self.assertEqual(len(supervisor.start_calls), 0)
        self.assertEqual(decision_provider.calls[0]["conversation"], (("user", "kurzer kontext"),))
        self.assertEqual(specialist.start_calls, [])
        self.assertEqual(result.tool_calls[0].name, "handoff_specialist_worker")
        self.assertTrue(result.used_web_search)

    def test_structured_supervisor_direct_reply_falls_back_to_spoken_ack_field(self) -> None:
        supervisor = FakeSupervisorProvider()
        specialist = FakeSpecialistProvider()
        decision_provider = FakeSupervisorDecisionProvider(
            {
                "action": "direct",
                "spoken_ack": "Hallo!",
                "spoken_reply": None,
            }
        )
        streamed: list[str] = []
        loop = DualLaneToolLoop(
            supervisor_provider=supervisor,
            specialist_provider=specialist,
            supervisor_decision_provider=decision_provider,
            tool_handlers={},
            tool_schemas=[],
            supervisor_instructions="Supervisor instructions",
            specialist_instructions="Specialist instructions",
        )

        result = loop.run("Sag bitte kurz Hallo.", on_text_delta=streamed.append)

        self.assertEqual(result.text, "Hallo!")
        self.assertEqual(streamed, ["Hallo!"])
        self.assertEqual(len(specialist.start_calls), 0)

    def test_prefetched_decision_skips_supervisor_decision_provider_roundtrip(self) -> None:
        supervisor = FakeSupervisorProvider()
        specialist = FakeSpecialistProvider()
        decision_provider = FakeSupervisorDecisionProvider(
            {
                "action": "direct",
                "spoken_reply": "Hallo!",
            }
        )
        prefetched_decision = SimpleNamespace(
            action="direct",
            spoken_reply="Schon da.",
            spoken_ack=None,
            kind=None,
            goal=None,
            allow_web_search=None,
            response_id="prefetched_1",
            request_id="prefetched_req_1",
            model="gpt-4o-mini",
            token_usage=None,
        )
        streamed: list[str] = []
        loop = DualLaneToolLoop(
            supervisor_provider=supervisor,
            specialist_provider=specialist,
            supervisor_decision_provider=decision_provider,
            tool_handlers={},
            tool_schemas=[],
            supervisor_instructions="Supervisor instructions",
            specialist_instructions="Specialist instructions",
        )

        result = loop.run(
            "Alles ok bei dir?",
            prefetched_decision=prefetched_decision,
            on_text_delta=streamed.append,
        )

        self.assertEqual(result.text, "Schon da.")
        self.assertEqual(streamed, ["Schon da."])
        self.assertEqual(decision_provider.calls, [])
        self.assertEqual(supervisor.start_calls, [])
        self.assertEqual(specialist.start_calls, [])

    def test_run_handoff_only_emits_filler_then_atomic_final(self) -> None:
        supervisor = FakeSupervisorProvider()
        specialist = FakeSpecialistProvider()
        lane_events: list[SpeechLaneDelta] = []
        search_calls: list[dict[str, object]] = []
        loop = DualLaneToolLoop(
            supervisor_provider=supervisor,
            specialist_provider=specialist,
            tool_handlers={
                "search_live_info": lambda arguments: search_calls.append(arguments) or {"answer": "8 Grad"},
            },
            tool_schemas=[{"type": "function", "name": "search_live_info"}],
            supervisor_instructions="Supervisor instructions",
            specialist_instructions="Specialist instructions",
        )

        result = loop.run_handoff_only(
            "Wie wird das Wetter morgen in Schwarzenbek?",
            handoff=SimpleNamespace(
                action="handoff",
                spoken_ack="Ich schaue kurz nach.",
                kind="search",
                goal="Find the weather.",
                allow_web_search=True,
                response_id="prefetch_resp",
                request_id="prefetch_req",
                model="gpt-4o-mini",
                token_usage=None,
            ),
            on_lane_text_delta=lane_events.append,
        )

        self.assertEqual(
            lane_events,
            [
                SpeechLaneDelta(
                    text="Ich schaue kurz nach.",
                    lane="filler",
                    replace_current=False,
                ),
                SpeechLaneDelta(
                    text="8 Grad",
                    lane="final",
                    replace_current=True,
                    atomic=True,
                ),
            ],
        )
        self.assertEqual(result.text, "8 Grad")
        self.assertEqual(search_calls, [{"question": "Wie wird das Wetter morgen in Schwarzenbek?"}])
        self.assertEqual(specialist.start_calls, [])


if __name__ == "__main__":
    unittest.main()
