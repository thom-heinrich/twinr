from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.agent.base_agent.conversation.closure import ConversationClosureDecision
from twinr.agent.workflows.follow_up_steering import FollowUpSteeringRuntime


class _FailingClosureEvaluator:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, **kwargs) -> ConversationClosureDecision:
        del kwargs
        self.calls += 1
        raise AssertionError("reply-expected fast path should not call closure evaluator")


class _RecordingClosureEvaluator:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def evaluate(self, **kwargs) -> ConversationClosureDecision:
        self.calls.append(dict(kwargs))
        return ConversationClosureDecision(
            close_now=False,
            confidence=0.31,
            reason="still_engaged",
            follow_up_action="end",
        )


class _FailingPersonalityContextService:
    def __init__(self) -> None:
        self.calls = 0

    def load_turn_steering_cues(self, *, config, remote_state):
        del config, remote_state
        self.calls += 1
        raise AssertionError("reply-expected fast path should not load remote steering cues")


class _RecordingPersonalityContextService:
    def __init__(self) -> None:
        self.calls = 0

    def load_turn_steering_cues(self, *, config, remote_state):
        del config, remote_state
        self.calls += 1
        return ()


class FollowUpSteeringRuntimeTests(unittest.TestCase):
    def _build_loop(self, *, evaluator) -> SimpleNamespace:
        config = TwinrConfig(
            openai_api_key="test-key",
            project_root=".",
            personality_dir="personality",
            conversation_closure_guard_enabled=True,
        )
        runtime = SimpleNamespace(
            conversation_context=lambda: (),
            long_term_memory=SimpleNamespace(
                prompt_context_store=SimpleNamespace(
                    memory_store=SimpleNamespace(remote_state=SimpleNamespace(cache_key="remote-state"))
                )
            ),
        )
        return SimpleNamespace(
            config=config,
            runtime=runtime,
            conversation_closure_evaluator=evaluator,
            emit=lambda _message: None,
            _trace_event=lambda *args, **kwargs: None,
            _follow_up_allowed_for_source=lambda initial_source: True,
        )

    def test_reply_expected_fast_path_skips_remote_steering_and_provider(self) -> None:
        evaluator = _FailingClosureEvaluator()
        personality_context_service = _FailingPersonalityContextService()
        helper = FollowUpSteeringRuntime(
            self._build_loop(evaluator=evaluator),
            personality_context_service=personality_context_service,
        )

        evaluation = helper.evaluate_closure(
            user_transcript="Wie geht es dir?",
            assistant_response="Mir geht's gut, danke! Und dir?",
            request_source="button",
            proactive_trigger=None,
        )

        self.assertEqual(evaluator.calls, 0)
        self.assertEqual(personality_context_service.calls, 0)
        self.assertEqual(evaluation.follow_up_action, "continue")
        self.assertTrue(evaluation.assistant_expects_reply)
        self.assertIsNotNone(evaluation.decision)
        assert evaluation.decision is not None
        self.assertEqual(evaluation.decision.reason, "assistant_expects_reply")

    def test_non_reply_turn_still_uses_full_closure_path(self) -> None:
        evaluator = _RecordingClosureEvaluator()
        personality_context_service = _RecordingPersonalityContextService()
        helper = FollowUpSteeringRuntime(
            self._build_loop(evaluator=evaluator),
            personality_context_service=personality_context_service,
        )

        evaluation = helper.evaluate_closure(
            user_transcript="Danke.",
            assistant_response="Gern geschehen.",
            request_source="button",
            proactive_trigger=None,
        )

        self.assertEqual(personality_context_service.calls, 1)
        self.assertEqual(len(evaluator.calls), 1)
        self.assertEqual(evaluation.follow_up_action, "end")
        self.assertFalse(evaluation.assistant_expects_reply)


if __name__ == "__main__":
    unittest.main()
