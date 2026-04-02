from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.orchestrator.local_bridge_target import LocalOrchestratorBridgeTarget
from twinr.orchestrator.remote_tool_timeout import DEFAULT_REMOTE_TOOL_TIMEOUT_SECONDS
from twinr.orchestrator.probe_turn import run_orchestrator_probe_turn


class _FakeProbeLoop:
    last_instance = None

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self._tool_handlers = {
            "search_live_info": lambda arguments: {"status": "ok", "question": arguments.get("question")}
        }
        self.close_calls: list[float] = []
        self.recorded_events: list[dict[str, object]] = []
        type(self).last_instance = self

    def _record_event(self, event, message, *, level="info", **payload) -> None:
        self.recorded_events.append(
            {
                "event": event,
                "message": message,
                "level": level,
                "payload": payload,
            }
        )

    def close(self, *, timeout_s: float = 1.0) -> None:
        self.close_calls.append(timeout_s)


class _FakeOrchestratorClient:
    last_instance = None

    def __init__(
        self,
        url: str,
        *,
        shared_secret: str | None = None,
        tool_timeout_seconds: float | None = None,
        require_tls: bool = True,
    ) -> None:
        self.url = url
        self.shared_secret = shared_secret
        self.tool_timeout_seconds = tool_timeout_seconds
        self.require_tls = require_tls
        self.run_calls: list[dict[str, object]] = []
        type(self).last_instance = self

    def run_turn(self, request, *, tool_handlers, on_ack=None, on_text_delta=None, on_transport_event=None):
        self.run_calls.append(
            {
                "request": request,
                "tool_handlers": dict(tool_handlers),
            }
        )
        if on_transport_event is not None:
            on_transport_event("ws_connected", {"request_id": "req-1"})
            on_transport_event("turn_request_sent", {"request_id": "req-1"})
            on_transport_event("first_server_event", {"message_type": "ack"})
            on_transport_event("tool_request_received", {"call_id": "call-1", "tool_name": "search_live_info"})
        if on_ack is not None:
            on_ack(SimpleNamespace(ack_id="checking_now", text="Ich schaue kurz nach."))
        if on_text_delta is not None:
            on_text_delta("Teil ")
            on_text_delta("Antwort")
        if on_transport_event is not None:
            on_transport_event(
                "tool_response_sent",
                {"call_id": "call-1", "tool_name": "search_live_info", "ok": True, "error": None},
            )
            on_transport_event(
                "turn_complete_received",
                {
                    "request_id": "req-1",
                    "response_id": "resp-1",
                    "model": "gpt-5.4-mini",
                    "rounds": 2,
                    "used_web_search": True,
                },
            )
        return SimpleNamespace(
            text="Teil Antwort",
            rounds=2,
            used_web_search=True,
            response_id="resp-1",
            request_id="req-1",
            model="gpt-5.4-mini",
        )


class OrchestratorProbeTurnTests(unittest.TestCase):
    def test_probe_turn_uses_lightweight_realtime_runtime_and_emits_stage_timings(self) -> None:
        config = TwinrConfig(
            orchestrator_ws_url="ws://127.0.0.1:8797/ws/orchestrator",
            orchestrator_shared_secret="secret",
            voice_orchestrator_enabled=True,
            voice_orchestrator_ws_url="ws://127.0.0.1:8798/ws/voice",
        )
        runtime_events: list[dict[str, object]] = []
        lines: list[str] = []
        runtime = SimpleNamespace(
            tool_provider_conversation_context=lambda: [{"role": "system", "content": "tool"}],
            supervisor_provider_conversation_context=lambda: [{"role": "system", "content": "supervisor"}],
            _record_event=lambda event, message, level="info", **payload: runtime_events.append(
                {
                    "event": event,
                    "message": message,
                    "level": level,
                    "payload": payload,
                }
            ),
        )
        fake_bundle = SimpleNamespace(
            print_backend=object(),
            stt=object(),
            verification_stt=object(),
            agent=object(),
            tts=object(),
            tool_agent=object(),
        )

        with patch("twinr.orchestrator.probe_turn.build_streaming_provider_bundle", return_value=fake_bundle):
            with patch("twinr.orchestrator.probe_turn.TwinrRealtimeHardwareLoop", _FakeProbeLoop):
                with patch("twinr.orchestrator.probe_turn.OrchestratorWebSocketClient", _FakeOrchestratorClient):
                    with patch(
                        "twinr.orchestrator.probe_turn.resolve_local_orchestrator_probe_target",
                        return_value=LocalOrchestratorBridgeTarget(
                            url="ws://127.0.0.1:8797/ws/orchestrator",
                            rewritten=True,
                            reason="host_loopback_bridge_override",
                        ),
                    ):
                        outcome = run_orchestrator_probe_turn(
                            config=config,
                            runtime=runtime,
                            backend=object(),
                            prompt="Wie gehts dir denn",
                            emit_line=lines.append,
                        )

        self.assertEqual(outcome.deltas, ("Teil ", "Antwort"))
        self.assertEqual(outcome.result.text, "Teil Antwort")
        self.assertEqual(outcome.tool_handler_count, 1)
        self.assertIsNotNone(_FakeProbeLoop.last_instance)
        self.assertFalse(_FakeProbeLoop.last_instance.kwargs["config"].voice_orchestrator_enabled)
        self.assertEqual(_FakeProbeLoop.last_instance.close_calls, [0.2])
        self.assertIsNotNone(_FakeOrchestratorClient.last_instance)
        self.assertEqual(
            _FakeOrchestratorClient.last_instance.run_calls[0]["request"].prompt,
            "Wie gehts dir denn",
        )
        self.assertTrue(_FakeOrchestratorClient.last_instance.require_tls)
        self.assertEqual(
            _FakeOrchestratorClient.last_instance.tool_timeout_seconds,
            DEFAULT_REMOTE_TOOL_TIMEOUT_SECONDS,
        )
        self.assertEqual(
            set(_FakeOrchestratorClient.last_instance.run_calls[0]["tool_handlers"]),
            {"search_live_info"},
        )
        self.assertTrue(any(line.startswith("probe_decision=lightweight_realtime_runtime") for line in lines))
        self.assertIn("probe_tool_handler_count=1", lines)
        self.assertIn("probe_orchestrator_target=ws://127.0.0.1:8797/ws/orchestrator", lines)
        self.assertIn("probe_orchestrator_target_reason=host_loopback_bridge_override", lines)
        self.assertTrue(any(line.startswith("probe_stage=provider_bundle status=ok") for line in lines))
        self.assertTrue(any(line.startswith("probe_stage=tool_runtime_bootstrap status=ok") for line in lines))
        self.assertTrue(any(line.startswith("probe_stage=ws_connect status=ok") for line in lines))
        self.assertTrue(any(line.startswith("probe_stage=turn_submit status=ok") for line in lines))
        self.assertTrue(any(line.startswith("probe_stage=first_server_event status=ok") for line in lines))
        self.assertTrue(any(line.startswith("probe_stage=tool_call status=ok") for line in lines))
        self.assertTrue(any(line.startswith("probe_stage=turn_complete status=ok") for line in lines))
        self.assertTrue(any(line.startswith("probe_stage=websocket_turn status=ok") for line in lines))
        self.assertTrue(any(line.startswith("probe_stage=cleanup_complete status=ok") for line in lines))
        self.assertTrue(any(line.startswith("ack=checking_now:Ich schaue kurz nach.") for line in lines))
        self.assertTrue(any(event["event"] == "orchestrator_probe_decision" for event in runtime_events))
        self.assertTrue(
            any(event["event"] == "orchestrator_probe_stage" for event in _FakeProbeLoop.last_instance.recorded_events)
        )
        self.assertGreaterEqual(len(outcome.stage_results), 7)

    def test_probe_turn_can_explicitly_allow_insecure_non_loopback_ws(self) -> None:
        config = TwinrConfig(
            orchestrator_ws_url="ws://192.168.1.154:8797/ws/orchestrator",
            orchestrator_allow_insecure_ws=True,
            orchestrator_shared_secret="secret",
        )
        runtime = SimpleNamespace(
            tool_provider_conversation_context=lambda: [],
            supervisor_provider_conversation_context=lambda: [],
            _record_event=lambda *args, **kwargs: None,
        )
        fake_bundle = SimpleNamespace(
            print_backend=object(),
            stt=object(),
            verification_stt=object(),
            agent=object(),
            tts=object(),
            tool_agent=object(),
        )

        with patch("twinr.orchestrator.probe_turn.build_streaming_provider_bundle", return_value=fake_bundle):
            with patch("twinr.orchestrator.probe_turn.TwinrRealtimeHardwareLoop", _FakeProbeLoop):
                with patch("twinr.orchestrator.probe_turn.OrchestratorWebSocketClient", _FakeOrchestratorClient):
                    with patch(
                        "twinr.orchestrator.probe_turn.resolve_local_orchestrator_probe_target",
                        return_value=LocalOrchestratorBridgeTarget(
                            url="ws://192.168.1.154:8797/ws/orchestrator",
                        ),
                    ):
                        run_orchestrator_probe_turn(
                            config=config,
                            runtime=runtime,
                            backend=object(),
                            prompt="Ping",
                            emit_line=lambda line: None,
                        )

        self.assertIsNotNone(_FakeOrchestratorClient.last_instance)
        self.assertFalse(_FakeOrchestratorClient.last_instance.require_tls)


if __name__ == "__main__":
    unittest.main()
