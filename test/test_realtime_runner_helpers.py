"""Targeted regression tests for extracted realtime-runner helper modules."""

from pathlib import Path
from threading import RLock
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import patch

from twinr.agent.base_agent.conversation.closure import ConversationClosureDecision
from twinr.agent.workflows import realtime_follow_up, voice_orchestrator_runtime
from twinr.agent.workflows.forensics import WorkflowForensics
from twinr.agent.workflows.remote_transcript_commit import RemoteTranscriptCommitCoordinator
from twinr.agent.workflows.realtime_runtime.support import TwinrRealtimeSupportMixin


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
    def __init__(
        self,
        *,
        enabled: bool = True,
        proactive_enabled: bool = False,
        voice_quiet_active: bool = False,
    ) -> None:
        self.config = SimpleNamespace(
            conversation_follow_up_enabled=enabled,
            conversation_follow_up_after_proactive_enabled=proactive_enabled,
        )
        self.runtime = SimpleNamespace(voice_quiet_active=lambda: voice_quiet_active)
        self.emitted: list[str] = []

    def emit(self, line: str) -> None:
        self.emitted.append(line)


class _VoiceHarness:
    def __init__(self) -> None:
        self.voice_orchestrator = _FakeVoiceOrchestrator()
        self._voice_orchestrator_runtime_state_lock = RLock()
        self._last_voice_orchestrator_runtime_state: tuple[str, str | None, bool] | None = None
        self._last_voice_orchestrator_intent_context = None
        self._last_voice_orchestrator_quiet_until_utc: str | None = None
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
        self.runtime = SimpleNamespace(
            status=SimpleNamespace(value="waiting"),
            voice_quiet_until_utc=lambda: None,
            voice_quiet_active=lambda: False,
        )
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


class _FakeWorkflowForensics(WorkflowForensics):
    def __init__(
        self,
        *,
        enabled: bool,
        service: str,
        run_id: str,
        remaining_budget: int,
        can_accept: bool,
    ) -> None:
        self.enabled = enabled
        self.service = service
        self.run_id = run_id
        self._remaining_budget = remaining_budget
        self._can_accept = can_accept
        self.closed = False

    def remaining_event_budget(self) -> int:
        return self._remaining_budget

    def can_accept_events(self) -> bool:
        return self._can_accept

    def close(self) -> None:
        self.closed = True


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

    def test_follow_up_allowed_for_source_respects_voice_quiet_gate(self) -> None:
        quiet_loop = _FollowUpHarness(enabled=True, proactive_enabled=True, voice_quiet_active=True)

        self.assertFalse(
            realtime_follow_up.follow_up_allowed_for_source(
                quiet_loop,
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

    def test_emit_closure_decision_includes_follow_up_action_when_present(self) -> None:
        loop = _FollowUpHarness()

        realtime_follow_up.emit_closure_decision(
            loop,
            ConversationClosureDecision(
                close_now=False,
                confidence=0.210,
                reason="still_engaged",
                follow_up_action="continue",
            ),
        )

        self.assertEqual(
            loop.emitted,
            [
                "conversation_closure_close_now=false",
                "conversation_closure_confidence=0.210",
                "conversation_closure_reason=still_engaged",
                "conversation_closure_follow_up_action=continue",
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
        self.assertIsNone(loop.voice_orchestrator.intent_contexts[-1]["voice_quiet_until_utc"])

    def test_refresh_voice_orchestrator_sensor_context_replays_when_voice_quiet_changes(self) -> None:
        loop = _VoiceHarness()

        voice_orchestrator_runtime.notify_voice_orchestrator_state(
            loop,
            "waiting",
            detail="idle",
            follow_up_allowed=True,
        )
        self.assertEqual(len(loop.voice_orchestrator.states), 1)

        loop.runtime = SimpleNamespace(
            status=SimpleNamespace(value="waiting"),
            voice_quiet_until_utc=lambda: "2026-03-25T12:15:00Z",
            voice_quiet_active=lambda: True,
        )
        voice_orchestrator_runtime.refresh_voice_orchestrator_sensor_context(loop)

        self.assertEqual(len(loop.voice_orchestrator.states), 2)
        self.assertEqual(loop.voice_orchestrator.states[-1], ("waiting", "idle", False))
        self.assertEqual(
            loop.voice_orchestrator.intent_contexts[-1]["voice_quiet_until_utc"],
            "2026-03-25T12:15:00Z",
        )

    def test_handle_remote_transcript_committed_reopens_follow_up_turn(self) -> None:
        loop = _VoiceHarness()
        loop._last_voice_orchestrator_runtime_state = ("follow_up_open", "voice_activation", True)

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
        trace_messages = [args[0] for args, _kwargs in loop.traces]
        self.assertIn("voice_orchestrator_transcript_committed_payload", trace_messages)
        self.assertIn("voice_orchestrator_follow_up_transcript_committed", trace_messages)
        payload_kwargs = next(
            kwargs
            for args, kwargs in loop.traces
            if args and args[0] == "voice_orchestrator_transcript_committed_payload"
        )
        self.assertEqual(payload_kwargs["details"]["source"], "follow_up")
        self.assertEqual(payload_kwargs["details"]["normalized"]["chars"], len("wie geht es dir"))

    def test_workflow_trace_capacity_helper_rotates_exhausted_run(self) -> None:
        old_tracer = _FakeWorkflowForensics(
            enabled=True,
            service="TwinrStreamingHardwareLoop",
            run_id="old-run",
            remaining_budget=12,
            can_accept=False,
        )
        new_tracer = _FakeWorkflowForensics(
            enabled=True,
            service="TwinrStreamingHardwareLoop",
            run_id="new-run",
            remaining_budget=5000,
            can_accept=True,
        )
        harness = SimpleNamespace(
            workflow_forensics=old_tracer,
            voice_orchestrator=SimpleNamespace(_forensics=old_tracer),
            _project_root=Path(tempfile.gettempdir()),
            traces=[],
            emitted=[],
        )

        def _trace_event(*args: object, **kwargs: object) -> None:
            harness.traces.append((args, kwargs))

        def _try_emit(line: str) -> None:
            harness.emitted.append(line)

        harness._trace_event = _trace_event
        harness._try_emit = _try_emit
        harness._safe_error_text = lambda exc: str(exc)
        harness._workflow_trace_min_session_event_budget = lambda: 512
        harness._close_workflow_trace_replacement = lambda tracer: tracer.close()

        with patch(
            "twinr.agent.workflows.realtime_runtime.support.WorkflowForensics.from_env",
            return_value=new_tracer,
        ):
            TwinrRealtimeSupportMixin._ensure_workflow_trace_capacity_for_session(
                harness,
                initial_source="follow_up",
                proactive_trigger=None,
                seed_present=True,
            )

        deadline = time.time() + 0.5
        while not old_tracer.closed and time.time() < deadline:
            time.sleep(0.01)

        self.assertIs(harness.workflow_forensics, new_tracer)
        self.assertIs(harness.voice_orchestrator._forensics, new_tracer)
        self.assertTrue(old_tracer.closed)
        self.assertIn(
            "workflow_trace_rotated_for_conversation_session",
            [args[0] for args, _kwargs in harness.traces],
        )


if __name__ == "__main__":
    unittest.main()
