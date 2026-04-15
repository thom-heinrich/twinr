from contextlib import contextmanager, nullcontext
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.contracts import SupervisorDecision
from twinr.agent.base_agent.config import TwinrConfig
from twinr.display.render_state import DisplayVisibleStateAssessment
from twinr.orchestrator.local_bridge_target import LocalOrchestratorBridgeTarget
from twinr.orchestrator.remote_tool_timeout import DEFAULT_REMOTE_TOOL_TIMEOUT_SECONDS
from twinr.orchestrator.probe_turn import (
    _PROBE_LONG_TERM_FLUSH_TIMEOUT_S,
    _PROBE_RUNTIME_CLOSE_TIMEOUT_S,
    _safe_record_event,
    run_orchestrator_probe_turn,
)


class _FakeProbeLoop:
    last_instance = None

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self._tool_handlers = {
            "search_live_info": lambda arguments: {"status": "ok", "question": arguments.get("question")}
        }
        self.close_calls: list[float] = []
        self.recorded_events: list[dict[str, object]] = []
        self.authorize_calls: list[str] = []
        self.trace_events: list[dict[str, object]] = []
        self.trace_decisions: list[dict[str, object]] = []
        self.trace_spans: list[dict[str, object]] = []
        self.active_trace_ids: list[str | None] = []
        self.workflow_forensics = SimpleNamespace(_run_dir="/tmp/probe-trace-run")
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

    def authorize_realtime_sensitive_tools(self, reason: str = "explicit") -> tuple[str, ...]:
        self.authorize_calls.append(reason)
        self._tool_handlers["remember_memory"] = lambda arguments: {
            "status": "ok",
            "memory": arguments.get("memory"),
        }
        return tuple(self._tool_handlers)

    def _workflow_trace_set_active(self, trace_id: str | None) -> None:
        self.active_trace_ids.append(trace_id)

    def _trace_event(self, msg: str, **kwargs) -> None:
        self.trace_events.append({"msg": msg, "kwargs": dict(kwargs)})

    def _trace_decision(self, msg: str, **kwargs) -> None:
        self.trace_decisions.append({"msg": msg, "kwargs": dict(kwargs)})

    def _trace_span(self, *, name: str, kind: str = "span", details: dict[str, object] | None = None, trace_id: str | None = None):
        self.trace_spans.append(
            {
                "name": name,
                "kind": kind,
                "details": dict(details or {}),
                "trace_id": trace_id,
            }
        )
        return nullcontext()

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
            on_transport_event("request_prepare_tool_handlers", {"handler_count": len(tool_handlers)})
            on_transport_event("request_prepare_headers", {"has_headers": bool(self.shared_secret)})
            on_transport_event("request_prepare_connector_kwargs", {"kwarg_count": 3})
            on_transport_event("request_prepare_deadline", {"bounded": True})
            on_transport_event(
                "request_prepare_payload",
                {
                    "request_id": "req-1",
                    "conversation_messages": len(getattr(request, "conversation", ())),
                    "supervisor_messages": len(getattr(request, "supervisor_conversation", ())),
                },
            )
            on_transport_event("request_prepared", {"request_id": "req-1"})
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


def _prefetched_runtime_local_decision() -> SupervisorDecision:
    return SupervisorDecision(
        action="handoff",
        spoken_ack="Ich schaue kurz nach.",
        spoken_reply=None,
        kind="automation",
        goal="Inspect Twinr quiet mode.",
        allow_web_search=False,
        context_scope="tiny_recent",
        runtime_tool_name="manage_voice_quiet_mode",
        runtime_tool_arguments={"action": "status"},
        response_id="probe-decision-1",
        request_id="probe-request-1",
        model="gpt-5.4-mini",
        token_usage=None,
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
        listening_sources: list[str] = []
        submitted_transcripts: list[str] = []
        flush_timeouts: list[float] = []
        runtime = SimpleNamespace(
            tool_provider_conversation_context=lambda: [{"role": "system", "content": "tool"}],
            tool_provider_tiny_recent_conversation_context=lambda: [
                {"role": "system", "content": "tool_tiny_recent"}
            ],
            supervisor_provider_conversation_context=lambda: [{"role": "system", "content": "supervisor"}],
            begin_listening=lambda request_source: listening_sources.append(request_source),
            submit_transcript=lambda transcript: submitted_transcripts.append(transcript),
            flush_long_term_memory=lambda *, timeout_s: flush_timeouts.append(timeout_s) or True,
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
                        with patch(
                            "twinr.orchestrator.probe_turn.assess_visible_display_state",
                            return_value=DisplayVisibleStateAssessment(
                                verdict="proved",
                                source="render_state",
                                reason="render_state_loaded",
                                visible_runtime_status="error",
                                visible_operator_status="error",
                                rendered_at="2026-04-11T09:05:00+00:00",
                            ),
                        ):
                            with patch(
                                "twinr.orchestrator.probe_turn._probe_prefetched_supervisor_decision",
                                return_value=_prefetched_runtime_local_decision(),
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
        self.assertEqual(outcome.tool_handler_count, 2)
        self.assertIsNotNone(_FakeProbeLoop.last_instance)
        self.assertFalse(_FakeProbeLoop.last_instance.kwargs["config"].voice_orchestrator_enabled)
        self.assertEqual(_FakeProbeLoop.last_instance.close_calls, [_PROBE_RUNTIME_CLOSE_TIMEOUT_S])
        self.assertEqual(_FakeProbeLoop.last_instance.authorize_calls, ["orchestrator_probe_turn"])
        self.assertEqual(listening_sources, ["orchestrator_probe_turn"])
        self.assertEqual(submitted_transcripts, ["Wie gehts dir denn"])
        self.assertEqual(flush_timeouts, [_PROBE_LONG_TERM_FLUSH_TIMEOUT_S])
        self.assertIsNotNone(_FakeOrchestratorClient.last_instance)
        self.assertEqual(
            _FakeOrchestratorClient.last_instance.run_calls[0]["request"].prompt,
            "Wie gehts dir denn",
        )
        self.assertEqual(
            _FakeOrchestratorClient.last_instance.run_calls[0]["request"].prefetched_supervisor_decision,
            {
                "action": "handoff",
                "spoken_ack": "Ich schaue kurz nach.",
                "spoken_reply": None,
                "kind": "automation",
                "goal": "Inspect Twinr quiet mode.",
                "prompt": None,
                "allow_web_search": False,
                "location_hint": None,
                "date_context": None,
                "context_scope": "tiny_recent",
                "runtime_tool_name": "manage_voice_quiet_mode",
                "runtime_tool_arguments": {"action": "status"},
                "response_id": "probe-decision-1",
                "request_id": "probe-request-1",
                "model": "gpt-5.4-mini",
                "token_usage": None,
            },
        )
        self.assertTrue(_FakeOrchestratorClient.last_instance.require_tls)
        self.assertEqual(
            _FakeOrchestratorClient.last_instance.tool_timeout_seconds,
            DEFAULT_REMOTE_TOOL_TIMEOUT_SECONDS,
        )
        self.assertEqual(
            set(_FakeOrchestratorClient.last_instance.run_calls[0]["tool_handlers"]),
            {"search_live_info", "remember_memory"},
        )
        self.assertTrue(any(line.startswith("probe_decision=lightweight_realtime_runtime") for line in lines))
        self.assertIn("probe_sensitive_tools_authorized=true", lines)
        self.assertIn("probe_tool_handler_count=2", lines)
        self.assertIn("probe_orchestrator_target=ws://127.0.0.1:8797/ws/orchestrator", lines)
        self.assertIn("probe_orchestrator_target_reason=host_loopback_bridge_override", lines)
        self.assertIn("display_visible_state=error display_visible_operator_status=error display_visible_state_verdict=proved display_visible_state_source=render_state display_visible_state_reason=render_state_loaded display_visible_rendered_at=2026-04-11T09:05:00+00:00", lines)
        self.assertTrue(any(line.startswith("probe_stage=provider_bundle status=ok") for line in lines))
        self.assertTrue(any(line.startswith("probe_stage=tool_runtime_bootstrap status=ok") for line in lines))
        self.assertTrue(any(line.startswith("probe_stage=prompt_seed status=ok") for line in lines))
        self.assertTrue(any(line.startswith("probe_stage=request_prepare_supervisor_context status=ok") for line in lines))
        self.assertTrue(any(line.startswith("probe_stage=request_prepare_supervisor_decision status=ok") for line in lines))
        self.assertTrue(
            any(
                line.startswith("probe_stage=request_prepare_tool_context status=ok")
                and "context_scope=tiny_recent" in line
                for line in lines
            )
        )
        self.assertTrue(any(line.startswith("probe_stage=request_prepare_request_object status=ok") for line in lines))
        self.assertTrue(any(line.startswith("probe_stage=request_prepare_tool_handlers status=ok") for line in lines))
        self.assertTrue(any(line.startswith("probe_stage=request_prepare_headers status=ok") for line in lines))
        self.assertTrue(any(line.startswith("probe_stage=request_prepare_connector_kwargs status=ok") for line in lines))
        self.assertTrue(any(line.startswith("probe_stage=request_prepare_deadline status=ok") for line in lines))
        self.assertTrue(any(line.startswith("probe_stage=request_prepare_payload status=ok") for line in lines))
        self.assertTrue(any(line.startswith("probe_stage=request_prepare status=ok") for line in lines))
        self.assertTrue(any(line.startswith("probe_stage=ws_connect status=ok") for line in lines))
        self.assertTrue(any(line.startswith("probe_stage=turn_submit status=ok") for line in lines))
        self.assertTrue(any(line.startswith("probe_stage=first_server_event status=ok") for line in lines))
        self.assertTrue(any(line.startswith("probe_stage=tool_call status=ok") for line in lines))
        self.assertTrue(any(line.startswith("probe_stage=turn_complete status=ok") for line in lines))
        self.assertTrue(any(line.startswith("probe_stage=websocket_turn status=ok") for line in lines))
        self.assertTrue(any(line.startswith("probe_stage=long_term_flush status=ok") for line in lines))
        self.assertTrue(any(line.startswith("probe_stage=cleanup_complete status=ok") for line in lines))
        self.assertTrue(any(line.startswith("ack=checking_now:Ich schaue kurz nach.") for line in lines))
        self.assertTrue(any(event["event"] == "orchestrator_probe_decision" for event in runtime_events))
        self.assertTrue(
            any(event["event"] == "orchestrator_probe_stage" for event in _FakeProbeLoop.last_instance.recorded_events)
        )
        self.assertGreaterEqual(len(outcome.stage_results), 7)

    def test_probe_turn_uses_tiny_recent_context_when_prefetched_decision_allows_it(self) -> None:
        config = TwinrConfig(
            orchestrator_ws_url="ws://127.0.0.1:8797/ws/orchestrator",
            orchestrator_shared_secret="secret",
        )
        lines: list[str] = []
        tool_context_calls = {"full": 0, "tiny_recent": 0}

        def _full_context():
            tool_context_calls["full"] += 1
            return [("system", "tool_full")]

        def _tiny_recent_context():
            tool_context_calls["tiny_recent"] += 1
            return [("system", "tool_tiny_recent")]

        runtime = SimpleNamespace(
            tool_provider_conversation_context=_full_context,
            tool_provider_tiny_recent_conversation_context=_tiny_recent_context,
            supervisor_provider_conversation_context=lambda: [("system", "supervisor")],
            begin_listening=lambda request_source: None,
            submit_transcript=lambda transcript: None,
            flush_long_term_memory=lambda *, timeout_s: True,
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
                        return_value=LocalOrchestratorBridgeTarget(url="ws://127.0.0.1:8797/ws/orchestrator"),
                    ):
                        with patch(
                            "twinr.orchestrator.probe_turn._probe_prefetched_supervisor_decision",
                            return_value=_prefetched_runtime_local_decision(),
                        ):
                            run_orchestrator_probe_turn(
                                config=config,
                                runtime=runtime,
                                backend=object(),
                                prompt="Bist du gerade ruhig?",
                                emit_line=lines.append,
                            )

        self.assertEqual(tool_context_calls["full"], 0)
        self.assertEqual(tool_context_calls["tiny_recent"], 1)
        self.assertEqual(
            _FakeOrchestratorClient.last_instance.run_calls[0]["request"].conversation,
            (("system", "tool_tiny_recent"),),
        )
        self.assertTrue(
            any(
                line.startswith("probe_stage=request_prepare_tool_context status=ok")
                and "context_scope=tiny_recent" in line
                for line in lines
            )
        )

    def test_safe_record_event_surfaces_owner_failures(self) -> None:
        owner = SimpleNamespace(_record_event=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

        with self.assertRaisesRegex(RuntimeError, "boom"):
            _safe_record_event(owner, "event", "message")

    def test_probe_turn_binds_workflow_forensics_and_bridges_runtime_trace_hooks(self) -> None:
        config = TwinrConfig(
            orchestrator_ws_url="ws://127.0.0.1:8797/ws/orchestrator",
            orchestrator_shared_secret="secret",
        )
        runtime_events: list[dict[str, object]] = []
        lines: list[str] = []
        runtime = SimpleNamespace(
            tool_provider_conversation_context=lambda: runtime._trace_event(  # type: ignore[attr-defined]
                "runtime_context_probe",
                kind="workflow",
                details={"source": "tool"},
            ) or [("system", "tool")],
            supervisor_provider_conversation_context=lambda: [("system", "supervisor")],
            begin_listening=lambda request_source: None,
            submit_transcript=lambda transcript: None,
            flush_long_term_memory=lambda *, timeout_s: True,
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

        @contextmanager
        def _bound_trace(_tracer):
            yield "trace-probe-1"

        with patch("twinr.orchestrator.probe_turn.build_streaming_provider_bundle", return_value=fake_bundle):
            with patch("twinr.orchestrator.probe_turn.TwinrRealtimeHardwareLoop", _FakeProbeLoop):
                with patch("twinr.orchestrator.probe_turn.OrchestratorWebSocketClient", _FakeOrchestratorClient):
                    with patch(
                        "twinr.orchestrator.probe_turn.resolve_local_orchestrator_probe_target",
                        return_value=LocalOrchestratorBridgeTarget(url="ws://127.0.0.1:8797/ws/orchestrator"),
                    ):
                        with patch("twinr.orchestrator.probe_turn.bind_workflow_forensics", _bound_trace):
                            with patch(
                                "twinr.orchestrator.probe_turn._probe_prefetched_supervisor_decision",
                                return_value=_prefetched_runtime_local_decision(),
                            ):
                                outcome = run_orchestrator_probe_turn(
                                    config=config,
                                    runtime=runtime,
                                    backend=object(),
                                    prompt="Ping",
                                    emit_line=lines.append,
                                )

        self.assertEqual(outcome.result.text, "Teil Antwort")
        self.assertIsNotNone(_FakeProbeLoop.last_instance)
        self.assertEqual(_FakeProbeLoop.last_instance.active_trace_ids, ["trace-probe-1", None])
        self.assertTrue(any(line == "probe_workflow_trace_id=trace-probe-1" for line in lines))
        self.assertTrue(any(line.startswith("probe_workflow_run_dir=") for line in lines))
        self.assertTrue(
            any(event["msg"] == "runtime_context_probe" for event in _FakeProbeLoop.last_instance.trace_events)
        )
        runtime_context_event = next(
            event for event in _FakeProbeLoop.last_instance.trace_events if event["msg"] == "runtime_context_probe"
        )
        self.assertEqual(runtime_context_event["kwargs"]["trace_id"], "trace-probe-1")

    def test_probe_turn_fails_closed_when_long_term_flush_misses_budget(self) -> None:
        config = TwinrConfig(
            orchestrator_ws_url="ws://127.0.0.1:8797/ws/orchestrator",
            orchestrator_shared_secret="secret",
        )
        runtime = SimpleNamespace(
            tool_provider_conversation_context=lambda: [],
            supervisor_provider_conversation_context=lambda: [],
            begin_listening=lambda request_source: None,
            submit_transcript=lambda transcript: None,
            flush_long_term_memory=lambda *, timeout_s: False,
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
                        return_value=LocalOrchestratorBridgeTarget(url="ws://127.0.0.1:8797/ws/orchestrator"),
                    ):
                        with patch(
                            "twinr.orchestrator.probe_turn._probe_prefetched_supervisor_decision",
                            return_value=_prefetched_runtime_local_decision(),
                        ):
                            with self.assertRaisesRegex(
                                RuntimeError,
                                "Probe long-term memory flush did not finish within 10.00s.",
                            ):
                                run_orchestrator_probe_turn(
                                    config=config,
                                    runtime=runtime,
                                    backend=object(),
                                    prompt="Ping",
                                    emit_line=lambda line: None,
                                )

    def test_probe_turn_can_explicitly_allow_insecure_non_loopback_ws(self) -> None:
        config = TwinrConfig(
            orchestrator_ws_url="ws://192.168.1.154:8797/ws/orchestrator",
            orchestrator_allow_insecure_ws=True,
            orchestrator_shared_secret="secret",
        )
        runtime = SimpleNamespace(
            tool_provider_conversation_context=lambda: [],
            supervisor_provider_conversation_context=lambda: [],
            begin_listening=lambda request_source: None,
            submit_transcript=lambda transcript: None,
            flush_long_term_memory=lambda *, timeout_s: True,
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
                        with patch(
                            "twinr.orchestrator.probe_turn._probe_prefetched_supervisor_decision",
                            return_value=_prefetched_runtime_local_decision(),
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
