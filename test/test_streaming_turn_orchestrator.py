from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.contracts import FirstWordReply
from twinr.agent.tools.runtime.dual_lane_loop import SpeechLaneDelta
from twinr.agent.tools.runtime.streaming_loop import StreamingToolLoopResult
from twinr.agent.workflows.streaming_turn_orchestrator import (
    StreamingTurnOrchestrator,
    StreamingTurnTimeoutPolicy,
)


class StreamingTurnOrchestratorTests(unittest.TestCase):
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

    def test_final_lane_error_uses_recovery_callback(self) -> None:
        lane_events: list[SpeechLaneDelta] = []
        recovery_calls: list[str] = []
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

        outcome = orchestrator.execute(
            prefetched_first_word=None,
            prefetched_first_word_source="none",
            generate_first_word=None,
            bridge_fallback_reply=None,
            run_final_lane=_failing_final_lane,
            recover_final_lane_response=lambda failure_reason: recovery_calls.append(failure_reason) or StreamingToolLoopResult(
                text="Ich antworte dir darauf jetzt direkt.",
                rounds=1,
                tool_calls=(),
                tool_results=(),
                response_id="resp_recovery",
                request_id="req_recovery",
                model="gpt-4o-mini",
                token_usage=None,
                used_web_search=False,
            ),
        )

        self.assertEqual(recovery_calls, ["final_lane_error"])
        self.assertEqual(outcome.response.text, "Ich antworte dir darauf jetzt direkt.")
        self.assertEqual(
            lane_events,
            [
                SpeechLaneDelta(
                    text="Ich antworte dir darauf jetzt direkt.",
                    lane="direct",
                    replace_current=False,
                    atomic=True,
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
