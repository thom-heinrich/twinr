from pathlib import Path
import sys
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.contracts import FirstWordReply
from twinr.agent.tools.runtime.speech_lane import SpeechLaneDelta
from twinr.agent.tools.runtime.streaming_loop import StreamingToolLoopResult
from twinr.agent.workflows.streaming_turn_orchestrator import (
    FinalLaneTimeoutError,
    StreamingTurnOrchestrator,
    StreamingTurnTimeoutPolicy,
)


class StreamingTurnOrchestratorTests(unittest.TestCase):
    def test_bridge_completion_traces_still_running_final_lane_snapshot(self) -> None:
        trace_events: list[tuple[str, dict[str, object]]] = []
        idle_after = [False]
        orchestrator = StreamingTurnOrchestrator(
            timeout_policy=StreamingTurnTimeoutPolicy(
                bridge_reply_timeout_ms=10,
                final_lane_watchdog_timeout_ms=200,
                final_lane_hard_timeout_ms=1000,
                first_audio_gate_ms=0,
            ),
            queue_lane_delta=lambda delta: None,
            wait_for_first_audio=lambda *, timeout_s=None: True,
            wait_until_idle=lambda *, timeout_s=None: True,
            is_output_idle=lambda: idle_after[0],
            ensure_processing_feedback=lambda: None,
            resume_processing_after_bridge=lambda: None,
            trace_event=lambda name, **kwargs: trace_events.append((name, kwargs)),
        )

        def _slow_final_lane() -> StreamingToolLoopResult:
            idle_after[0] = True
            time.sleep(0.05)
            return StreamingToolLoopResult(
                text="Hier ist die finale Antwort.",
                rounds=1,
                tool_calls=(),
                tool_results=(),
                response_id="resp_final",
                request_id="req_final",
                model="gpt-4o-mini",
                token_usage=None,
                used_web_search=True,
            )

        outcome = orchestrator.execute(
            prefetched_first_word=FirstWordReply(mode="filler", spoken_text="Ich schaue kurz nach."),
            prefetched_first_word_source="prefetched",
            generate_first_word=None,
            bridge_fallback_reply=None,
            run_final_lane=_slow_final_lane,
            recover_final_lane_response=None,
        )

        self.assertEqual(outcome.response.text, "Hier ist die finale Antwort.")
        event = next(
            payload
            for name, payload in trace_events
            if name == "streaming_final_lane_still_running_after_bridge"
        )
        final_lane = event["details"]["final_lane"]
        self.assertEqual(final_lane["name"], "final-lane")
        self.assertFalse(final_lane["done"])
        self.assertEqual(final_lane["thread"]["name"], "twinr-final-lane")
        self.assertTrue(final_lane["thread"]["stack_present"])

    def test_final_response_waits_for_bridge_idle_after_first_audio(self) -> None:
        lane_events: list[SpeechLaneDelta] = []
        waits: list[tuple[str, float | None]] = []
        orchestrator = StreamingTurnOrchestrator(
            timeout_policy=StreamingTurnTimeoutPolicy(
                bridge_reply_timeout_ms=250,
                final_lane_watchdog_timeout_ms=4000,
                final_lane_hard_timeout_ms=15000,
                first_audio_gate_ms=900,
            ),
            queue_lane_delta=lane_events.append,
            wait_for_first_audio=lambda *, timeout_s=None: waits.append(("audio", timeout_s)) or True,
            wait_until_idle=lambda *, timeout_s=None: waits.append(("idle", timeout_s)) or True,
            ensure_processing_feedback=lambda: None,
        )

        orchestrator._queue_final_response(
            StreamingToolLoopResult(
                text="Vorhin haben wir über das Wetter gesprochen.",
                rounds=1,
                tool_calls=(),
                tool_results=(),
                response_id="resp_final",
                request_id="req_final",
                model="gpt-4o-mini",
                token_usage=None,
                used_web_search=False,
            ),
            bridge_reply=FirstWordReply(mode="filler", spoken_text="Einen Moment bitte."),
            wait_for_bridge_audio=True,
        )

        self.assertEqual(waits, [("audio", 0.9), ("idle", 0.9)])
        self.assertEqual(
            lane_events,
            [
                SpeechLaneDelta(
                    text="Vorhin haben wir über das Wetter gesprochen.",
                    lane="final",
                    replace_current=True,
                    atomic=True,
                )
            ],
        )

    def test_final_lane_error_raises_instead_of_recovering(self) -> None:
        lane_events: list[SpeechLaneDelta] = []
        orchestrator = StreamingTurnOrchestrator(
            timeout_policy=StreamingTurnTimeoutPolicy(
                bridge_reply_timeout_ms=10,
                final_lane_watchdog_timeout_ms=50,
                final_lane_hard_timeout_ms=200,
                first_audio_gate_ms=0,
            ),
            queue_lane_delta=lane_events.append,
            wait_for_first_audio=lambda *, timeout_s=None: True,
            ensure_processing_feedback=lambda: None,
        )

        def _failing_final_lane() -> StreamingToolLoopResult:
            raise RuntimeError("boom")

        with self.assertRaisesRegex(RuntimeError, "boom"):
            orchestrator.execute(
                prefetched_first_word=None,
                prefetched_first_word_source="none",
                generate_first_word=None,
                bridge_fallback_reply=None,
                run_final_lane=_failing_final_lane,
                recover_final_lane_response=None,
            )

        self.assertEqual(lane_events, [])

    def test_final_lane_error_can_bypass_recovery_for_fatal_errors(self) -> None:
        orchestrator = StreamingTurnOrchestrator(
            timeout_policy=StreamingTurnTimeoutPolicy(
                bridge_reply_timeout_ms=10,
                final_lane_watchdog_timeout_ms=50,
                final_lane_hard_timeout_ms=200,
                first_audio_gate_ms=0,
            ),
            queue_lane_delta=lambda delta: None,
            wait_for_first_audio=lambda *, timeout_s=None: True,
            ensure_processing_feedback=lambda: None,
        )

        class _FatalRemoteError(RuntimeError):
            pass

        def _failing_final_lane() -> StreamingToolLoopResult:
            raise _FatalRemoteError("remote down")

        with self.assertRaises(_FatalRemoteError):
            orchestrator.execute(
                prefetched_first_word=None,
                prefetched_first_word_source="none",
                generate_first_word=None,
                bridge_fallback_reply=None,
                run_final_lane=_failing_final_lane,
                recover_final_lane_response=lambda failure_reason: StreamingToolLoopResult(
                    text=failure_reason,
                    rounds=1,
                    tool_calls=(),
                    tool_results=(),
                    response_id="resp_recovery",
                    request_id="req_recovery",
                    model="gpt-4o-mini",
                    token_usage=None,
                    used_web_search=False,
                ),
                should_recover_final_lane_error=lambda exc: not isinstance(exc, _FatalRemoteError),
            )

    def test_bridge_filler_completion_resumes_processing_while_final_lane_waits(self) -> None:
        lane_events: list[SpeechLaneDelta] = []
        resumed: list[str] = []
        idle_after = [False]

        orchestrator = StreamingTurnOrchestrator(
            timeout_policy=StreamingTurnTimeoutPolicy(
                bridge_reply_timeout_ms=10,
                final_lane_watchdog_timeout_ms=200,
                final_lane_hard_timeout_ms=1000,
                first_audio_gate_ms=0,
            ),
            queue_lane_delta=lane_events.append,
            wait_for_first_audio=lambda *, timeout_s=None: True,
            wait_until_idle=lambda *, timeout_s=None: True,
            is_output_idle=lambda: idle_after[0],
            ensure_processing_feedback=lambda: None,
            resume_processing_after_bridge=lambda: resumed.append("processing"),
        )

        def _slow_final_lane() -> StreamingToolLoopResult:
            idle_after[0] = True
            time.sleep(0.05)
            return StreamingToolLoopResult(
                text="Hier ist die finale Antwort.",
                rounds=1,
                tool_calls=(),
                tool_results=(),
                response_id="resp_final",
                request_id="req_final",
                model="gpt-4o-mini",
                token_usage=None,
                used_web_search=True,
            )

        outcome = orchestrator.execute(
            prefetched_first_word=FirstWordReply(mode="filler", spoken_text="Ich schaue kurz nach."),
            prefetched_first_word_source="prefetched",
            generate_first_word=None,
            bridge_fallback_reply=None,
            run_final_lane=_slow_final_lane,
            recover_final_lane_response=None,
        )

        self.assertEqual(resumed, ["processing"])
        self.assertEqual(outcome.response.text, "Hier ist die finale Antwort.")

    def test_final_lane_timeout_raises_error_instead_of_recovery_reply(self) -> None:
        orchestrator = StreamingTurnOrchestrator(
            timeout_policy=StreamingTurnTimeoutPolicy(
                bridge_reply_timeout_ms=10,
                final_lane_watchdog_timeout_ms=20,
                final_lane_hard_timeout_ms=40,
                first_audio_gate_ms=0,
            ),
            queue_lane_delta=lambda delta: None,
            wait_for_first_audio=lambda *, timeout_s=None: True,
            ensure_processing_feedback=lambda: None,
        )

        def _blocking_final_lane() -> StreamingToolLoopResult:
            time.sleep(0.2)
            return StreamingToolLoopResult(
                text="Zu spät.",
                rounds=1,
                tool_calls=(),
                tool_results=(),
                response_id="resp_final",
                request_id="req_final",
                model="gpt-4o-mini",
                token_usage=None,
                used_web_search=True,
            )

        with self.assertRaises(FinalLaneTimeoutError):
            orchestrator.execute(
                prefetched_first_word=None,
                prefetched_first_word_source="none",
                generate_first_word=None,
                bridge_fallback_reply=None,
                run_final_lane=_blocking_final_lane,
                recover_final_lane_response=None,
            )

    def test_final_lane_timeout_requests_cooperative_stop_signal(self) -> None:
        requested: list[str] = []
        trace_events: list[tuple[str, dict[str, object]]] = []
        orchestrator = StreamingTurnOrchestrator(
            timeout_policy=StreamingTurnTimeoutPolicy(
                bridge_reply_timeout_ms=10,
                final_lane_watchdog_timeout_ms=20,
                final_lane_hard_timeout_ms=40,
                first_audio_gate_ms=0,
            ),
            queue_lane_delta=lambda delta: None,
            wait_for_first_audio=lambda *, timeout_s=None: True,
            ensure_processing_feedback=lambda: None,
            request_final_lane_stop=requested.append,
            trace_event=lambda name, **kwargs: trace_events.append((name, kwargs)),
        )

        def _blocking_final_lane() -> StreamingToolLoopResult:
            time.sleep(0.2)
            return StreamingToolLoopResult(
                text="Zu spät.",
                rounds=1,
                tool_calls=(),
                tool_results=(),
                response_id="resp_final",
                request_id="req_final",
                model="gpt-4o-mini",
                token_usage=None,
                used_web_search=True,
            )

        with self.assertRaises(FinalLaneTimeoutError):
            orchestrator.execute(
                prefetched_first_word=None,
                prefetched_first_word_source="none",
                generate_first_word=None,
                bridge_fallback_reply=None,
                run_final_lane=_blocking_final_lane,
                recover_final_lane_response=None,
            )

        self.assertEqual(requested, ["timeout"])
        watchdog_event = next(
            payload
            for name, payload in trace_events
            if name == "streaming_final_lane_watchdog_triggered"
        )
        timeout_event = next(
            payload for name, payload in trace_events if name == "streaming_final_lane_timeout"
        )
        self.assertEqual(watchdog_event["level"], "WARN")
        self.assertEqual(timeout_event["level"], "ERROR")
        final_lane = timeout_event["details"]["final_lane"]
        self.assertEqual(final_lane["name"], "final-lane")
        self.assertFalse(final_lane["done"])
        self.assertEqual(final_lane["thread"]["name"], "twinr-final-lane")
        self.assertTrue(final_lane["thread"]["stack_present"])
        self.assertEqual(final_lane["thread"]["top_frame"]["func"], "_blocking_final_lane")


if __name__ == "__main__":
    unittest.main()
