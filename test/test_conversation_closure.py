from pathlib import Path
from threading import Event
import sys
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.conversation.closure import (
    ConversationClosureDecision,
    ToolCallingConversationClosureEvaluator,
)
from twinr.agent.base_agent.contracts import AgentToolCall, ToolCallingTurnResponse
from twinr.config import TwinrConfig


class FakeClosureToolAgentProvider:
    def __init__(self, config: TwinrConfig, *, close_now: bool = True) -> None:
        self.config = config
        self.close_now = close_now
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
            tool_calls=(
                AgentToolCall(
                    name="submit_closure_decision",
                    call_id="closure-1",
                    arguments={
                        "close_now": self.close_now,
                        "confidence": 0.93 if self.close_now else 0.22,
                        "reason": "explicit_goodbye" if self.close_now else "still_engaged",
                    },
                    raw_arguments='{"close_now":true,"confidence":0.93,"reason":"explicit_goodbye"}'
                    if self.close_now
                    else '{"close_now":false,"confidence":0.22,"reason":"still_engaged"}',
                ),
            ),
        )


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


class ConversationClosureEvaluatorTests(unittest.TestCase):
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
                    user_transcript="Danke.",
                    assistant_response="Gern geschehen.",
                    request_source="button",
                )
        finally:
            provider.release.set()
        elapsed = time.monotonic() - started

        self.assertTrue(provider.started.is_set())
        self.assertLess(elapsed, 1.0)


if __name__ == "__main__":
    unittest.main()
