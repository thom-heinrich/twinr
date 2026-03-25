"""Targeted regression tests for extracted realtime-runner helper modules."""

from threading import RLock
from types import SimpleNamespace
import unittest

from twinr.agent.base_agent.conversation.closure import ConversationClosureDecision
from twinr.agent.workflows import realtime_follow_up, voice_orchestrator_runtime
from twinr.agent.workflows.remote_transcript_commit import RemoteTranscriptCommitCoordinator


class _FakeVoiceOrchestrator:
    def __init__(self) -> None:
        self.states: list[tuple[str, str | None, bool]] = []
        self.intent_contexts: list[dict[str, object]] = []
        self.pauses: list[str] = []
        self.resumes: list[str] = []

    def notify_runtime_state(
        self,
        *,
        state: str,
        detail: str | None = None,
        follow_up_allowed: bool = False,
        **kwargs: object,
    ) -> None:
        self.states.append((state, detail, follow_up_allowed))
        self.intent_contexts.append(dict(kwargs))

    def seed_runtime_state(
        self,
        *,
        state: str,
        detail: str | None = None,
        follow_up_allowed: bool = False,
        **kwargs: object,
    ) -> None:
        self.notify_runtime_state(
            state=state,
            detail=detail,
            follow_up_allowed=follow_up_allowed,
            **kwargs,
        )

    def pause_capture(self, *, reason: str) -> None:
        self.pauses.append(reason)

    def resume_capture(self, *, reason: str) -> None:
        self.resumes.append(reason)


class _FollowUpHarness:
    def __init__(self, *, enabled: bool = True, proactive_enabled: bool = False) -> None:
        self.config = SimpleNamespace(
            conversation_follow_up_enabled=enabled,
            conversation_follow_up_after_proactive_enabled=proactive_enabled,
        )
        self.emitted: list[str] = []

    def emit(self, line: str) -> None:
        self.emitted.append(line)


class _VoiceHarness:
    def __init__(self) -> None:
        self.voice_orchestrator = _FakeVoiceOrchestrator()
        self._voice_orchestrator_runtime_state_lock = RLock()
        self._last_voice_orchestrator_runtime_state: tuple[str, str | None, bool] | None = None
        self._last_voice_orchestrator_intent_context = None
        self._latest_sensor_observation_facts = {
            "camera": {"person_visible": True},
            "person_state": {
                "interaction_ready": True,
                "targeted_inference_blocked": False,
                "recommended_channel": "speech",
                "attention_state": {"state": "attending_to_device"},
                "interaction_intent_state": {"state": "showing_intent"},
            },
        }
        self.runtime = SimpleNamespace(status=SimpleNamespace(value="waiting"))
        self._remote_transcript_commits = RemoteTranscriptCommitCoordinator()
        self.emitted: list[str] = []
        self.traces: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.refresh_requests = 0
        self.session_calls: list[dict[str, object]] = []

    def emit(self, line: str) -> None:
        self.emitted.append(line)

    def _trace_event(self, *args: object, **kwargs: object) -> None:
        self.traces.append((args, kwargs))

    def _handle_error(self, exc: Exception) -> None:
        raise exc

    def _required_remote_dependency_current_ready(self) -> bool:
        return True

    def _request_required_remote_dependency_refresh(self) -> None:
        self.refresh_requests += 1

    def _background_work_allowed(self) -> bool:
        return True

    def _run_conversation_session(self, **kwargs: object) -> bool:
        self.session_calls.append(dict(kwargs))
        return True

    def _follow_up_allowed_for_source(self, *, initial_source: str) -> bool:
        return realtime_follow_up.follow_up_allowed_for_source(
            SimpleNamespace(
                config=SimpleNamespace(
                    conversation_follow_up_enabled=True,
                    conversation_follow_up_after_proactive_enabled=False,
                )
            ),
            initial_source=initial_source,
        )


class RealtimeFollowUpHelpersTests(unittest.TestCase):
    def test_follow_up_allowed_for_source_respects_proactive_gate(self) -> None:
        proactive_loop = _FollowUpHarness(enabled=True, proactive_enabled=False)
        normal_loop = _FollowUpHarness(enabled=True, proactive_enabled=False)

        self.assertFalse(
            realtime_follow_up.follow_up_allowed_for_source(
                proactive_loop,
                initial_source="proactive",
            )
        )
        self.assertTrue(
            realtime_follow_up.follow_up_allowed_for_source(
                normal_loop,
                initial_source="button",
            )
        )

    def test_emit_closure_decision_writes_expected_lines(self) -> None:
        loop = _FollowUpHarness()

        realtime_follow_up.emit_closure_decision(
            loop,
            ConversationClosureDecision(
                close_now=True,
                confidence=0.875,
                reason="explicit_goodbye",
            ),
        )

        self.assertEqual(
            loop.emitted,
            [
                "conversation_closure_close_now=true",
                "conversation_closure_confidence=0.875",
                "conversation_closure_reason=explicit_goodbye",
            ],
        )


class VoiceOrchestratorRuntimeHelpersTests(unittest.TestCase):
    def test_notify_voice_orchestrator_state_updates_cached_state_and_context(self) -> None:
        loop = _VoiceHarness()

        voice_orchestrator_runtime.notify_voice_orchestrator_state(
            loop,
            "waiting",
            detail="idle",
            follow_up_allowed=False,
        )

        self.assertEqual(loop._last_voice_orchestrator_runtime_state, ("waiting", "idle", False))
        self.assertEqual(loop.voice_orchestrator.states[-1], ("waiting", "idle", False))
        self.assertTrue(loop.voice_orchestrator.intent_contexts[-1]["person_visible"])

    def test_handle_remote_transcript_committed_reopens_follow_up_turn(self) -> None:
        loop = _VoiceHarness()

        handled = voice_orchestrator_runtime.handle_remote_transcript_committed(
            loop,
            "wie geht es dir",
            "follow_up",
        )

        self.assertTrue(handled)
        self.assertEqual(loop.emitted[-1], "voice_orchestrator_follow_up=true")
        self.assertEqual(
            loop.session_calls,
            [
                {
                    "initial_source": "follow_up",
                    "seed_transcript": "wie geht es dir",
                    "play_initial_beep": False,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
