from pathlib import Path
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.conversation.decision_core import normalize_turn_text
from twinr.agent.base_agent.conversation.turn_controller import (
    StreamingTurnController,
    ToolCallingTurnDecisionEvaluator,
    TurnEvaluationCandidate,
    _normalize_turn_text,
)
from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.contracts import (
    AgentToolCall,
    StreamingSpeechEndpointEvent,
    ToolCallingTurnResponse,
)


class FakeTurnToolAgentProvider:
    def __init__(self, config: TwinrConfig) -> None:
        self.config = config
        self.calls: list[dict[str, object]] = []

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
        self.calls.append(
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
                    name="submit_turn_decision",
                    call_id="call_turn_1",
                    arguments={
                        "decision": "end_turn",
                        "label": "complete",
                        "confidence": 0.91,
                        "reason": "complete_request",
                        "transcript": "ich bin noch am programmieren",
                    },
                    raw_arguments='{"decision":"end_turn","label":"complete","confidence":0.91,"reason":"complete_request","transcript":"ich bin noch am programmieren"}',
                ),
            ),
        )

    def continue_turn_streaming(
        self,
        *,
        continuation_token: str,
        tool_results=(),
        instructions=None,
        tool_schemas=(),
        allow_web_search=None,
        on_text_delta=None,
    ) -> ToolCallingTurnResponse:
        del continuation_token, tool_results, instructions, tool_schemas, allow_web_search, on_text_delta
        raise NotImplementedError


class TurnControllerTests(unittest.TestCase):
    def test_public_normalize_turn_text_matches_legacy_turn_controller_alias(self) -> None:
        transcript = "  Hallo \n  Welt  "

        self.assertEqual(normalize_turn_text(transcript), "hallo welt")
        self.assertEqual(_normalize_turn_text(transcript), normalize_turn_text(transcript))

    def test_tool_call_evaluator_returns_structured_decision(self) -> None:
        config = TwinrConfig(
            openai_api_key="test-key",
            project_root=".",
            personality_dir="personality",
        )
        provider = FakeTurnToolAgentProvider(config)
        evaluator = ToolCallingTurnDecisionEvaluator(config=config, provider=provider)

        decision = evaluator.evaluate(
            candidate=TurnEvaluationCandidate(
                transcript="ich bin noch am programmieren",
                event_type="speech_final",
                speech_final=True,
            ),
            conversation=(("assistant", "Woran arbeitest du?"),),
        )

        self.assertEqual(decision.decision, "end_turn")
        self.assertEqual(decision.label, "complete")
        self.assertAlmostEqual(decision.confidence, 0.91)
        self.assertEqual(decision.reason, "complete_request")
        self.assertEqual(decision.transcript, "ich bin noch am programmieren")
        self.assertEqual(provider.calls[0]["allow_web_search"], False)

    def test_streaming_turn_controller_requests_stop_after_end_turn_decision(self) -> None:
        config = TwinrConfig(
            openai_api_key="test-key",
            project_root=".",
            personality_dir="personality",
        )
        provider = FakeTurnToolAgentProvider(config)
        evaluator = ToolCallingTurnDecisionEvaluator(config=config, provider=provider)
        lines: list[str] = []
        controller = StreamingTurnController(
            config=config,
            evaluator=evaluator,
            conversation_factory=lambda: (("assistant", "Woran arbeitest du?"),),
            emit=lines.append,
        )

        controller.on_interim("ich bin noch")
        controller.on_endpoint(
            StreamingSpeechEndpointEvent(
                transcript="ich bin noch am programmieren",
                event_type="speech_final",
                speech_final=True,
            )
        )
        for _ in range(50):
            if controller.should_stop_capture():
                break
            time.sleep(0.001)

        self.assertTrue(controller.should_stop_capture())
        self.assertEqual(controller.latest_transcript(), "ich bin noch am programmieren")
        self.assertIn("turn_controller_decision=end_turn", lines)
        self.assertIn("turn_controller_label=complete", lines)

    def test_streaming_turn_controller_fast_path_stops_on_speech_final(self) -> None:
        config = TwinrConfig(
            openai_api_key="test-key",
            project_root=".",
            personality_dir="personality",
            turn_controller_fast_endpoint_enabled=True,
            turn_controller_fast_endpoint_min_chars=6,
        )
        provider = FakeTurnToolAgentProvider(config)
        evaluator = ToolCallingTurnDecisionEvaluator(config=config, provider=provider)
        lines: list[str] = []
        controller = StreamingTurnController(
            config=config,
            evaluator=evaluator,
            conversation_factory=lambda: (),
            emit=lines.append,
        )
        controller.on_interim("ich bin")

        controller.on_endpoint(
            StreamingSpeechEndpointEvent(
                transcript="ich bin fertig",
                event_type="speech_final",
                speech_final=True,
            )
        )

        self.assertTrue(controller.should_stop_capture())
        self.assertEqual(controller.latest_transcript(), "ich bin fertig")
        self.assertEqual(provider.calls, [])
        self.assertIn("turn_controller_reason=speech_final_fast_path", lines)
        self.assertIn("turn_controller_label=complete", lines)

    def test_streaming_turn_controller_waits_for_finalize_on_bare_speech_final(self) -> None:
        config = TwinrConfig(
            openai_api_key="test-key",
            project_root=".",
            personality_dir="personality",
            turn_controller_fast_endpoint_enabled=True,
            turn_controller_fast_endpoint_min_chars=6,
        )
        provider = FakeTurnToolAgentProvider(config)
        evaluator = ToolCallingTurnDecisionEvaluator(config=config, provider=provider)
        lines: list[str] = []
        controller = StreamingTurnController(
            config=config,
            evaluator=evaluator,
            conversation_factory=lambda: (),
            emit=lines.append,
        )

        controller.on_endpoint(
            StreamingSpeechEndpointEvent(
                transcript="bitte gut",
                event_type="speech_final",
                speech_final=True,
            )
        )

        self.assertFalse(controller.should_stop_capture())
        self.assertEqual(controller.latest_transcript(), "bitte gut")
        self.assertEqual(provider.calls, [])
        self.assertIn("turn_controller_reason=speech_final_wait_for_finalize", lines)

    def test_streaming_turn_controller_fast_path_stops_on_stable_utterance_end(self) -> None:
        config = TwinrConfig(
            openai_api_key="test-key",
            project_root=".",
            personality_dir="personality",
            turn_controller_fast_endpoint_enabled=True,
            turn_controller_fast_endpoint_min_chars=6,
            deepgram_streaming_stop_on_utterance_end=True,
        )
        provider = FakeTurnToolAgentProvider(config)
        evaluator = ToolCallingTurnDecisionEvaluator(config=config, provider=provider)
        lines: list[str] = []
        controller = StreamingTurnController(
            config=config,
            evaluator=evaluator,
            conversation_factory=lambda: (),
            emit=lines.append,
        )

        controller.on_interim("ich bin fertig")
        controller.on_endpoint(
            StreamingSpeechEndpointEvent(
                transcript="ich bin fertig",
                event_type="utterance_end",
            )
        )

        self.assertTrue(controller.should_stop_capture())
        self.assertEqual(controller.latest_transcript(), "ich bin fertig")
        self.assertEqual(provider.calls, [])
        self.assertIn("turn_controller_reason=utterance_end_fast_path", lines)
        self.assertIn("turn_controller_label=complete", lines)

    def test_text_fallback_can_parse_backchannel_label(self) -> None:
        config = TwinrConfig(
            openai_api_key="test-key",
            project_root=".",
            personality_dir="personality",
        )

        class BackchannelProvider(FakeTurnToolAgentProvider):
            def start_turn_streaming(self, *args, **kwargs) -> ToolCallingTurnResponse:
                del args, kwargs
                return ToolCallingTurnResponse(
                    text='{"decision":"end_turn","label":"backchannel","confidence":0.82,"reason":"short_answer","transcript":"ja"}'
                )

        evaluator = ToolCallingTurnDecisionEvaluator(config=config, provider=BackchannelProvider(config))

        decision = evaluator.evaluate(
            candidate=TurnEvaluationCandidate(
                transcript="ja",
                event_type="speech_final",
                speech_final=True,
            ),
            conversation=(("assistant", "Moechtest du, dass ich es drucke?"),),
        )

        self.assertEqual(decision.decision, "end_turn")
        self.assertEqual(decision.label, "backchannel")

    def test_turn_controller_only_passes_lane_specific_instructions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            personality_dir = Path(temp_dir) / "personality"
            personality_dir.mkdir()
            (personality_dir / "TURN_CONTROLLER.md").write_text(
                "Turn-controller lane instructions",
                encoding="utf-8",
            )
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
            )
            provider = FakeTurnToolAgentProvider(config)
            evaluator = ToolCallingTurnDecisionEvaluator(config=config, provider=provider)

            evaluator.evaluate(
                candidate=TurnEvaluationCandidate(
                    transcript="ich bin noch am programmieren",
                    event_type="speech_final",
                    speech_final=True,
                ),
                conversation=(("assistant", "Woran arbeitest du?"),),
            )

        self.assertEqual(provider.calls[0]["instructions"], "Turn-controller lane instructions")


if __name__ == "__main__":
    unittest.main()
