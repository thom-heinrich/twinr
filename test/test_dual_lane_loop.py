from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.contracts import AgentToolCall, ToolCallingTurnResponse
from twinr.agent.tools.runtime.dual_lane_loop import DualLaneToolLoop, SpeechLaneDelta


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
            location_hint=self.decision.get("location_hint"),
            date_context=self.decision.get("date_context"),
            context_scope=self.decision.get("context_scope"),
            response_id="decision_1",
            request_id="req_decision_1",
            model="gpt-4o-mini",
            token_usage=None,
        )


class DualLaneLoopTests(unittest.TestCase):
    def test_resolve_supervisor_decision_merges_per_turn_instructions(self) -> None:
        supervisor_decision_provider = FakeSupervisorDecisionProvider(
            {
                "action": "handoff",
                "kind": "search",
                "goal": "Find current Hamburg local politics updates.",
                "spoken_ack": "Ich schaue kurz nach.",
                "allow_web_search": True,
            }
        )
        loop = DualLaneToolLoop(
            supervisor_provider=FakeSupervisorProvider(),
            specialist_provider=FakeSpecialistProvider(),
            tool_handlers={},
            tool_schemas=(),
            supervisor_decision_provider=supervisor_decision_provider,
            supervisor_instructions="Supervisor instructions",
            specialist_instructions="Specialist instructions",
        )

        decision = loop.resolve_supervisor_decision(
            "Was ist denn heute in der Hamburger Lokalpolitik los?",
            conversation=(("system", "Stay calm"),),
            instructions="DISPLAY OVERLAY",
        )

        self.assertEqual(decision.action, "handoff")
        self.assertEqual(len(supervisor_decision_provider.calls), 1)
        self.assertIn("Supervisor instructions", supervisor_decision_provider.calls[0]["instructions"])
        self.assertIn("DISPLAY OVERLAY", supervisor_decision_provider.calls[0]["instructions"])

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
        search_calls: list[dict[str, object]] = []
        decision_provider = FakeSupervisorDecisionProvider(
            {
                "action": "handoff",
                "spoken_ack": "Einen Moment bitte.",
                "kind": "search",
                "goal": "Weather tomorrow in Schwarzenbek.",
                "allow_web_search": True,
                "location_hint": "Schwarzenbek",
                "date_context": "Tuesday, 2026-03-17 (Europe/Berlin)",
                "context_scope": "full_context",
            }
        )
        streamed: list[str] = []
        loop = DualLaneToolLoop(
            supervisor_provider=supervisor,
            specialist_provider=specialist,
            supervisor_decision_provider=decision_provider,
            tool_handlers={
                "search_live_info": lambda arguments: search_calls.append(arguments) or {"answer": "8 Grad", "arguments": arguments},
            },
            tool_schemas=[{"type": "function", "name": "search_live_info"}],
            supervisor_instructions="Supervisor instructions",
            specialist_instructions="Specialist instructions",
        )

        result = loop.run(
            "Wie wird das Wetter morgen in Schwarzenbek?",
            supervisor_conversation=(("user", "kurzer kontext"),),
            specialist_conversation=(("user", "voller suchkontext"),),
            instructions="General tool-agent instructions that must not leak into the decision lane.",
            on_text_delta=streamed.append,
        )

        self.assertEqual(streamed[0], "Einen Moment bitte.")
        self.assertEqual(len(supervisor.start_calls), 0)
        self.assertEqual(decision_provider.calls[0]["conversation"], (("user", "kurzer kontext"),))
        self.assertIn("Supervisor instructions", decision_provider.calls[0]["instructions"])
        self.assertIn(
            "General tool-agent instructions that must not leak into the decision lane.",
            decision_provider.calls[0]["instructions"],
        )
        self.assertEqual(specialist.start_calls, [])
        self.assertEqual(
            search_calls,
            [
                {
                    "question": "Wie wird das Wetter morgen in Schwarzenbek?",
                    "location_hint": "Schwarzenbek",
                    "date_context": "Tuesday, 2026-03-17 (Europe/Berlin)",
                }
            ],
        )

    def test_structured_supervisor_decision_allows_handoff_without_spoken_ack(self) -> None:
        supervisor = FakeSupervisorProvider()
        specialist = FakeSpecialistProvider()
        search_calls: list[dict[str, object]] = []
        decision_provider = FakeSupervisorDecisionProvider(
            {
                "action": "handoff",
                "spoken_ack": None,
                "kind": "search",
                "goal": "Weather tomorrow in Schwarzenbek.",
                "allow_web_search": True,
                "location_hint": "Schwarzenbek",
                "date_context": "Tuesday, 2026-03-17 (Europe/Berlin)",
                "context_scope": "full_context",
            }
        )
        streamed: list[str] = []
        loop = DualLaneToolLoop(
            supervisor_provider=supervisor,
            specialist_provider=specialist,
            supervisor_decision_provider=decision_provider,
            tool_handlers={
                "search_live_info": lambda arguments: search_calls.append(arguments) or {"answer": "8 Grad", "arguments": arguments},
            },
            tool_schemas=[{"type": "function", "name": "search_live_info"}],
            supervisor_instructions="Supervisor instructions",
            specialist_instructions="Specialist instructions",
        )

        result = loop.run(
            "Wie wird das Wetter morgen in Schwarzenbek?",
            supervisor_conversation=(("user", "kurzer kontext"),),
            specialist_conversation=(("user", "voller suchkontext"),),
            instructions="General tool-agent instructions that must not leak into the decision lane.",
            on_text_delta=streamed.append,
        )

        self.assertEqual(streamed, ["8 Grad"])
        self.assertEqual(len(supervisor.start_calls), 0)
        self.assertEqual(decision_provider.calls[0]["conversation"], (("user", "kurzer kontext"),))
        self.assertEqual(specialist.start_calls, [])
        self.assertEqual(
            search_calls,
            [
                {
                    "question": "Wie wird das Wetter morgen in Schwarzenbek?",
                    "location_hint": "Schwarzenbek",
                    "date_context": "Tuesday, 2026-03-17 (Europe/Berlin)",
                }
            ],
        )
        self.assertEqual(result.text, "8 Grad")
        self.assertEqual(result.tool_calls[0].name, "handoff_specialist_worker")
        self.assertTrue(result.used_web_search)

    def test_structured_supervisor_direct_reply_uses_spoken_reply_field(self) -> None:
        supervisor = FakeSupervisorProvider()
        specialist = FakeSpecialistProvider()
        decision_provider = FakeSupervisorDecisionProvider(
            {
                "action": "direct",
                "spoken_ack": None,
                "spoken_reply": "Hallo!",
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

    def test_structured_supervisor_direct_full_context_downgrades_to_memory_handoff(self) -> None:
        supervisor = FakeSupervisorProvider()

        class MemorySpecialistProvider:
            def __init__(self) -> None:
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
                    on_text_delta("Vorhin haben wir über das Wetter gesprochen.")
                return ToolCallingTurnResponse(
                    text="Vorhin haben wir über das Wetter gesprochen.",
                    response_id="memory_done",
                    model="gpt-5.2",
                )

        specialist = MemorySpecialistProvider()
        trace_decisions: list[tuple[str, dict[str, object]]] = []
        decision_provider = FakeSupervisorDecisionProvider(
            {
                "action": "direct",
                "spoken_reply": "Ich kann mich nicht erinnern.",
                "spoken_ack": "Ich hole kurz unser Gespräch zusammen.",
                "kind": "memory",
                "goal": "Recall what Twinr and the user discussed earlier today.",
                "context_scope": "full_context",
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
            trace_decision=lambda name, **kwargs: trace_decisions.append((name, kwargs)),
        )

        result = loop.run(
            "Worüber haben wir heute geredet?",
            specialist_conversation=(("system", "reicher erinnerungskontext"),),
            on_text_delta=streamed.append,
        )

        self.assertEqual(streamed[0], "Ich hole kurz unser Gespräch zusammen.")
        self.assertEqual(result.text, "Vorhin haben wir über das Wetter gesprochen.")
        self.assertEqual(len(specialist.start_calls), 1)
        self.assertEqual(specialist.start_calls[0]["conversation"], (("system", "reicher erinnerungskontext"),))
        self.assertTrue(
            any(
                name == "dual_lane_direct_downgraded_to_handoff"
                and payload["context"]["decision"]["context_scope"] == "full_context"
                and payload["context"]["fallback_handoff"]["kind"] == "memory"
                for name, payload in trace_decisions
            )
        )

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

    def test_supervisor_loop_failure_raises_explicit_error(self) -> None:
        class ExplodingSupervisorProvider(FakeSupervisorProvider):
            def start_turn_streaming(self, *args, **kwargs):  # type: ignore[override]
                raise RuntimeError("supervisor failed")

        supervisor = ExplodingSupervisorProvider()
        specialist = FakeSpecialistProvider()

        class FailThenRecoverDecisionProvider:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def decide(self, prompt: str, *, conversation=None, instructions=None):
                self.calls.append(
                    {
                        "prompt": prompt,
                        "conversation": conversation,
                        "instructions": instructions,
                    }
                )
                if len(self.calls) == 1:
                    raise RuntimeError("decision failed")
                return SimpleNamespace(
                    action="direct",
                    spoken_ack=None,
                    spoken_reply="Ich beantworte das jetzt direkt fuer dich.",
                    kind=None,
                    goal=None,
                    allow_web_search=None,
                    location_hint=None,
                    date_context=None,
                    context_scope=None,
                    response_id="decision_2",
                    request_id="req_decision_2",
                    model="gpt-4o-mini",
                    token_usage=None,
                )

        decision_provider = FailThenRecoverDecisionProvider()
        streamed: list[str] = []
        trace_events: list[tuple[str, dict[str, object]]] = []
        loop = DualLaneToolLoop(
            supervisor_provider=supervisor,
            specialist_provider=specialist,
            supervisor_decision_provider=decision_provider,
            tool_handlers={},
            tool_schemas=[],
            supervisor_instructions="Supervisor instructions",
            specialist_instructions="Specialist instructions",
            trace_event=lambda name, **kwargs: trace_events.append((name, kwargs)),
        )

        with self.assertRaises(RuntimeError):
            loop.run(
                "Worueber haben wir heute gesprochen?",
                supervisor_conversation=(("system", "voller kontext"),),
                on_text_delta=streamed.append,
            )

        self.assertEqual(streamed, [])
        self.assertEqual(len(decision_provider.calls), 1)
        self.assertEqual(decision_provider.calls[0]["conversation"], (("system", "voller kontext"),))
        self.assertTrue(
            any(
                name == "dual_lane_supervisor_loop_failed"
                and payload["details"]["error_type"] == "RuntimeError"
                for name, payload in trace_events
            )
        )

    def test_handoff_only_specialist_failure_raises_explicit_error(self) -> None:
        supervisor = FakeSupervisorProvider()
        specialist = FakeSpecialistProvider()
        decision_provider = FakeSupervisorDecisionProvider(
            {
                "action": "direct",
                "spoken_reply": "Ich hole dir das kurz direkt aus unserem Kontext.",
            }
        )
        lane_events: list[SpeechLaneDelta] = []
        loop = DualLaneToolLoop(
            supervisor_provider=supervisor,
            specialist_provider=specialist,
            supervisor_decision_provider=decision_provider,
            tool_handlers={
                "search_live_info": lambda arguments: (_ for _ in ()).throw(RuntimeError(f"boom: {arguments}")),
            },
            tool_schemas=[{"type": "function", "name": "search_live_info"}],
            supervisor_instructions="Supervisor instructions",
            specialist_instructions="Specialist instructions",
        )

        with self.assertRaises(RuntimeError):
            loop.run_handoff_only(
                "Worueber haben wir heute gesprochen?",
                conversation=(("system", "reicher kontext"),),
                handoff=SimpleNamespace(
                    action="handoff",
                    spoken_ack="Ich schaue kurz nach.",
                    kind="search",
                    goal="Recall today's conversation.",
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
            ],
        )
        self.assertEqual(decision_provider.calls, [])

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

    def test_run_handoff_only_passes_location_and_date_hints_to_direct_search(self) -> None:
        supervisor = FakeSupervisorProvider()
        specialist = FakeSpecialistProvider()
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

        loop.run_handoff_only(
            "Wie wird das Wetter morgen dort?",
            handoff=SimpleNamespace(
                action="handoff",
                spoken_ack="Ich schaue kurz nach.",
                kind="search",
                goal="Find the weather.",
                allow_web_search=True,
                location_hint="Schwarzenbek",
                date_context="Tuesday, 2026-03-17 (Europe/Berlin)",
                response_id="prefetch_resp",
                request_id="prefetch_req",
                model="gpt-4o-mini",
                token_usage=None,
            ),
        )

        self.assertEqual(
            search_calls,
            [
                {
                    "question": "Wie wird das Wetter morgen dort?",
                    "location_hint": "Schwarzenbek",
                    "date_context": "Tuesday, 2026-03-17 (Europe/Berlin)",
                }
            ],
        )

    def test_run_handoff_only_prefers_handoff_prompt_for_direct_search(self) -> None:
        supervisor = FakeSupervisorProvider()
        specialist = FakeSpecialistProvider()
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

        loop.run_handoff_only(
            "Was ist denn heute so in der schwarzen Lokalpolitik?",
            handoff=SimpleNamespace(
                action="handoff",
                spoken_ack="Ich schaue kurz nach.",
                kind="search",
                goal="Find the latest Hamburg local politics update.",
                prompt="Was ist heute in der Hamburger Lokalpolitik wichtig?",
                allow_web_search=True,
                location_hint="Hamburg",
                date_context="Sunday, 2026-03-23 (Europe/Berlin)",
                response_id="prefetch_resp",
                request_id="prefetch_req",
                model="gpt-4o-mini",
                token_usage=None,
            ),
        )

        self.assertEqual(
            search_calls,
            [
                {
                    "question": "Was ist heute in der Hamburger Lokalpolitik wichtig?",
                    "location_hint": "Hamburg",
                    "date_context": "Sunday, 2026-03-23 (Europe/Berlin)",
                }
            ],
        )

    def test_run_handoff_only_skips_direct_search_when_turn_was_stopped(self) -> None:
        supervisor = FakeSupervisorProvider()
        specialist = FakeSpecialistProvider()
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

        with self.assertRaises(InterruptedError):
            loop.run_handoff_only(
                "Wie wird das Wetter morgen dort?",
                handoff=SimpleNamespace(
                    action="handoff",
                    spoken_ack="Ich schaue kurz nach.",
                    kind="search",
                    goal="Find the weather.",
                    allow_web_search=True,
                    location_hint="Schwarzenbek",
                    date_context="Tuesday, 2026-03-17 (Europe/Berlin)",
                    response_id="prefetch_resp",
                    request_id="prefetch_req",
                    model="gpt-4o-mini",
                    token_usage=None,
                ),
                should_stop=lambda: True,
            )

        self.assertEqual(search_calls, [])


if __name__ == "__main__":
    unittest.main()
