from pathlib import Path
from threading import Event
import json
import sys
import time
import unittest
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.conversation.closure import (
    ConversationClosureDecision,
    StructuredConversationClosureEvaluator,
    ToolCallingConversationClosureEvaluator,
    assistant_expects_immediate_reply,
)
from twinr.agent.base_agent.contracts import (
    AgentToolCall,
    ConversationClosureProviderDecision,
    ToolCallingTurnResponse,
)
from twinr.agent.personality.steering import ConversationTurnSteeringCue
from twinr.agent.base_agent.config import TwinrConfig


class FakeClosureToolAgentProvider:
    def __init__(
        self,
        config: TwinrConfig,
        *,
        close_now: bool = True,
        follow_up_action: str = "end",
        matched_topics: tuple[str, ...] = (),
    ) -> None:
        self.config = config
        self.close_now = close_now
        self.follow_up_action = follow_up_action
        self.matched_topics = matched_topics
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
        **kwargs,
    ) -> ToolCallingTurnResponse:
        del on_text_delta, kwargs
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
            continuation_token="closure-test-token",
            tool_calls=(
                AgentToolCall(
                    name="submit_closure_decision",
                    call_id="closure-1",
                    arguments={
                        "close_now": self.close_now,
                        "confidence": 0.93 if self.close_now else 0.22,
                        "reason": "explicit_goodbye" if self.close_now else "still_engaged",
                        "follow_up_action": self.follow_up_action,
                        "matched_topics": list(self.matched_topics),
                    },
                    raw_arguments=json.dumps(
                        {
                            "close_now": self.close_now,
                            "confidence": 0.93 if self.close_now else 0.22,
                            "reason": "explicit_goodbye" if self.close_now else "still_engaged",
                            "follow_up_action": self.follow_up_action,
                            "matched_topics": list(self.matched_topics),
                        }
                    ),
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


class BlockingClosureToolAgentProvider:
    def __init__(self, config: TwinrConfig) -> None:
        self.config = config
        self.started = Event()
        self.release = Event()

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
        del prompt, conversation, instructions, tool_schemas, allow_web_search, on_text_delta
        self.started.set()
        self.release.wait(timeout=5.0)
        return ToolCallingTurnResponse(text="", tool_calls=())

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


class FakeStructuredClosureProvider:
    def __init__(
        self,
        config: TwinrConfig,
        *,
        close_now: bool = False,
        follow_up_action: str = "end",
        matched_topics: tuple[str, ...] = (),
    ) -> None:
        self.config = config
        self.close_now = close_now
        self.follow_up_action = follow_up_action
        self.matched_topics = matched_topics
        self.calls: list[dict[str, object]] = []

    def decide(
        self,
        prompt: str,
        *,
        conversation=None,
        instructions=None,
        timeout_seconds=None,
    ) -> ConversationClosureProviderDecision:
        self.calls.append(
            {
                "prompt": prompt,
                "conversation": conversation,
                "instructions": instructions,
                "timeout_seconds": timeout_seconds,
            }
        )
        return ConversationClosureProviderDecision(
            close_now=self.close_now,
            confidence=0.87 if self.close_now else 0.34,
            reason="explicit_goodbye" if self.close_now else "still_engaged",
            matched_topics=self.matched_topics,
            follow_up_action=self.follow_up_action,
        )


class ConversationClosureEvaluatorTests(unittest.TestCase):
    def test_assistant_expects_immediate_reply_is_question_gated(self) -> None:
        self.assertTrue(assistant_expects_immediate_reply("Moechtest du noch etwas wissen?"))
        self.assertTrue(assistant_expects_immediate_reply("Soll ich das gleich drucken?"))
        self.assertTrue(
            assistant_expects_immediate_reply(
                "Das haengt von der Zeitzone ab. Welche Zeitzone meinst du"
            )
        )
        self.assertFalse(
            assistant_expects_immediate_reply(
                "Ich brauche deine Zeitzone oder deinen Ort, um dir die genaue Uhrzeit zu sagen. Wenn du mir das nennst, sage ich dir sofort die aktuelle Zeit."
            )
        )
        self.assertFalse(assistant_expects_immediate_reply("Es ist zehn Uhr."))

    def test_evaluator_returns_structured_close_decision(self) -> None:
        config = TwinrConfig(
            openai_api_key="test-key",
            project_root=".",
            personality_dir="personality",
        )
        provider = FakeClosureToolAgentProvider(config, close_now=True)
        evaluator = ToolCallingConversationClosureEvaluator(config=config, provider=provider)

        decision = evaluator.evaluate(
            user_transcript="Danke, bis bald.",
            assistant_response="Gern, bis bald.",
            request_source="button",
            conversation=(("assistant", "Kann ich dir sonst noch helfen?"),),
        )

        self.assertEqual(
            decision,
            ConversationClosureDecision(
                close_now=True,
                confidence=0.93,
                reason="explicit_goodbye",
                follow_up_action="end",
            ),
        )
        self.assertEqual(provider.calls[0]["allow_web_search"], False)

    def test_evaluator_can_keep_conversation_open(self) -> None:
        config = TwinrConfig(
            openai_api_key="test-key",
            project_root=".",
            personality_dir="personality",
        )
        provider = FakeClosureToolAgentProvider(config, close_now=False)
        evaluator = ToolCallingConversationClosureEvaluator(config=config, provider=provider)

        decision = evaluator.evaluate(
            user_transcript="Wie wird morgen das Wetter?",
            assistant_response="Ich schaue kurz nach.",
            request_source="button",
        )

        self.assertFalse(decision.close_now)
        self.assertEqual(decision.reason, "still_engaged")

    def test_structured_follow_up_action_can_keep_statement_style_clarification_open(self) -> None:
        config = TwinrConfig(
            openai_api_key="test-key",
            project_root=".",
            personality_dir="personality",
        )
        provider = FakeStructuredClosureProvider(
            config,
            close_now=False,
            follow_up_action="continue",
        )
        evaluator = StructuredConversationClosureEvaluator(config=config, provider=provider)

        decision = evaluator.evaluate(
            user_transcript="Wie spaet ist es in New York?",
            assistant_response=(
                "Ich brauche deine Zeitzone oder deinen Ort, um dir die genaue Uhrzeit zu sagen. "
                "Wenn du mir das nennst, sage ich dir sofort die aktuelle Zeit."
            ),
            request_source="voice_activation",
        )

        self.assertFalse(decision.close_now)
        self.assertEqual(decision.follow_up_action, "continue")

    def test_evaluator_serializes_turn_steering_and_parses_matched_topics(self) -> None:
        config = TwinrConfig(
            openai_api_key="test-key",
            project_root=".",
            personality_dir="personality",
        )
        provider = FakeClosureToolAgentProvider(
            config,
            close_now=False,
            matched_topics=("AI companions",),
        )
        evaluator = ToolCallingConversationClosureEvaluator(config=config, provider=provider)

        decision = evaluator.evaluate(
            user_transcript="Und bei KI-Begleitern gerade?",
            assistant_response="Kurz gesagt geht gerade viel in Richtung alltagstauglicher Begleiter.",
            request_source="button",
            turn_steering_cues=(
                ConversationTurnSteeringCue(
                    title="AI companions",
                    salience=0.92,
                    attention_state="shared_thread",
                    open_offer="brief_update_if_open",
                    user_pull="one_calm_follow_up",
                    positive_engagement_action="invite_follow_up",
                ),
            ),
        )

        self.assertEqual(decision.matched_topics, ("AI companions",))
        prompt_payload = json.loads(cast(str, provider.calls[0]["prompt"]))
        self.assertIn("turn_steering", prompt_payload)
        self.assertEqual(prompt_payload["turn_steering"]["topics"][0]["title"], "AI companions")
        self.assertEqual(
            prompt_payload["turn_steering"]["topics"][0]["positive_engagement_action"],
            "invite_follow_up",
        )
        self.assertEqual(
            prompt_payload["turn_steering"]["topics"][0]["match_summary"],
            "",
        )
        self.assertIn("Use both title and match_summary", prompt_payload["turn_steering"]["instruction"])

    def test_evaluator_serializes_semantic_match_summary_for_disambiguation(self) -> None:
        config = TwinrConfig(
            openai_api_key="test-key",
            project_root=".",
            personality_dir="personality",
        )
        provider = FakeClosureToolAgentProvider(
            config,
            close_now=False,
            matched_topics=("Hamburg local politics",),
        )
        evaluator = ToolCallingConversationClosureEvaluator(config=config, provider=provider)

        decision = evaluator.evaluate(
            user_transcript="What changed in Hamburg local politics today?",
            assistant_response="Mostly concrete civic and municipal decisions.",
            request_source="button",
            turn_steering_cues=(
                ConversationTurnSteeringCue(
                    title="Hamburg local politics",
                    salience=0.81,
                    attention_state="growing",
                    open_offer="mention_if_clearly_relevant",
                    user_pull="wait_for_user_pull",
                    observe_mode="mostly_observe_until_user_pull",
                    positive_engagement_action="hint",
                    match_summary="Municipal decisions, civic policy, and local public changes that affect daily life in Hamburg.",
                ),
            ),
        )

        self.assertEqual(decision.matched_topics, ("Hamburg local politics",))
        prompt_payload = json.loads(cast(str, provider.calls[0]["prompt"]))
        self.assertEqual(
            prompt_payload["turn_steering"]["topics"][0]["match_summary"],
            "Municipal decisions, civic policy, and local public changes that affect daily life in Hamburg.",
        )

    def test_evaluator_enforces_wall_clock_timeout_without_adapter_timeout_kwarg(self) -> None:
        config = TwinrConfig(
            openai_api_key="test-key",
            project_root=".",
            personality_dir="personality",
            conversation_closure_provider_timeout_seconds=0.25,
        )
        provider = BlockingClosureToolAgentProvider(config)
        evaluator = ToolCallingConversationClosureEvaluator(config=config, provider=provider)

        started = time.monotonic()
        try:
            with self.assertRaises(TimeoutError):
                evaluator.evaluate(
                    user_transcript="Erzaehl mir mehr dazu.",
                    assistant_response="Ich kann noch etwas dazu sagen.",
                    request_source="button",
                )
        finally:
            provider.release.set()
        elapsed = time.monotonic() - started

        self.assertTrue(provider.started.is_set())
        self.assertLess(elapsed, 1.0)

    def test_evaluator_short_circuits_clear_signoff_without_provider_call(self) -> None:
        config = TwinrConfig(
            openai_api_key="test-key",
            project_root=".",
            personality_dir="personality",
        )
        provider = FakeClosureToolAgentProvider(config, close_now=False)
        evaluator = ToolCallingConversationClosureEvaluator(config=config, provider=provider)

        decision = evaluator.evaluate(
            user_transcript="Danke.",
            assistant_response="Gern geschehen.",
            request_source="button",
        )

        self.assertTrue(decision.close_now)
        self.assertEqual(decision.reason, "user_terminal_signoff")
        self.assertEqual(provider.calls, [])

    def test_structured_evaluator_uses_fast_provider_contract(self) -> None:
        config = TwinrConfig(
            openai_api_key="test-key",
            project_root=".",
            personality_dir="personality",
            conversation_closure_provider_timeout_seconds=7.5,
        )
        provider = FakeStructuredClosureProvider(
            config,
            close_now=False,
            matched_topics=("AI companions",),
        )
        evaluator = StructuredConversationClosureEvaluator(config=config, provider=provider)

        decision = evaluator.evaluate(
            user_transcript="Und bei KI-Begleitern gerade?",
            assistant_response="Kurz gesagt geht gerade viel in Richtung alltagstauglicher Begleiter.",
            request_source="button",
            turn_steering_cues=(
                ConversationTurnSteeringCue(
                    title="AI companions",
                    salience=0.92,
                    attention_state="shared_thread",
                    open_offer="brief_update_if_open",
                    user_pull="one_calm_follow_up",
                    positive_engagement_action="invite_follow_up",
                ),
            ),
        )

        self.assertEqual(decision.close_now, False)
        self.assertEqual(decision.reason, "still_engaged")
        self.assertEqual(decision.matched_topics, ("AI companions",))
        self.assertEqual(provider.calls[0]["timeout_seconds"], 7.5)
        self.assertIn("AI companions", cast(str, provider.calls[0]["prompt"]))


if __name__ == "__main__":
    unittest.main()
