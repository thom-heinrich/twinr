from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.contracts import AgentToolCall, ToolCallingTurnResponse
from twinr.agent.base_agent.config import TwinrConfig
from twinr.orchestrator.acks import ack_text_for_id
from twinr.orchestrator.client import OrchestratorWebSocketClient
from twinr.orchestrator.contracts import OrchestratorToolResponse, OrchestratorTurnCompleteEvent
from twinr.orchestrator.contracts import OrchestratorTurnRequest
from twinr.orchestrator.server import EdgeOrchestratorServer
from twinr.orchestrator.session import EdgeOrchestratorSession, RemoteToolBridge


class _FakeDecisionProvider:
    def decide(self, prompt: str, *, conversation=None, instructions=None):
        del prompt, conversation, instructions
        return type(
            "Decision",
            (),
            {
                "action": "handoff",
                "spoken_ack": "Ich schaue kurz nach.",
                "spoken_reply": None,
                "kind": "search",
                "goal": "Check the weather.",
                "allow_web_search": True,
                "response_id": "decision-1",
                "request_id": "req-1",
                "model": "gpt-4o-mini",
                "token_usage": None,
            },
        )()


class _FakeSpecialistProvider:
    def __init__(self) -> None:
        self.start_calls: list[dict[str, object]] = []
        self.continue_calls: list[dict[str, object]] = []

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
        self.start_calls.append({})
        return ToolCallingTurnResponse(
            text="",
            tool_calls=(
                AgentToolCall(
                    name="search_live_info",
                    call_id="call-search-1",
                    arguments={"question": "Wie wird das Wetter morgen?"},
                    raw_arguments='{"question":"Wie wird das Wetter morgen?"}',
                ),
            ),
            response_id="worker-start",
            continuation_token="worker-start",
        )

    def continue_turn_streaming(
        self,
        *,
        continuation_token: str,
        tool_results,
        instructions=None,
        tool_schemas=(),
        allow_web_search=None,
        on_text_delta=None,
    ) -> ToolCallingTurnResponse:
        del continuation_token, tool_results, instructions, tool_schemas, allow_web_search, on_text_delta
        self.continue_calls.append({})
        return ToolCallingTurnResponse(
            text="Morgen wird es sonnig.",
            response_id="worker-done",
            used_web_search=True,
        )


class _FakeServerSession:
    def run_turn(self, prompt, *, conversation, supervisor_conversation, emit_event, tool_bridge):
        del prompt, conversation, supervisor_conversation
        emit_event({"type": "ack", "ack_id": "checking_now", "text": "Ich schaue kurz nach."})
        handler = tool_bridge.build_handlers(("search_live_info",))["search_live_info"]
        tool_call = AgentToolCall(
            name="search_live_info",
            call_id="server-call-1",
            arguments={"question": "Wie wird das Wetter morgen?"},
            raw_arguments='{"question":"Wie wird das Wetter morgen?"}',
        )
        output = handler(tool_call)
        self.last_output = output
        from twinr.orchestrator.contracts import OrchestratorTurnCompleteEvent

        return OrchestratorTurnCompleteEvent(
            text="Morgen wird es sonnig.",
            rounds=2,
            used_web_search=True,
            response_id="resp-1",
            request_id="req-1",
            model="gpt-4o-mini",
        )


class OrchestratorSessionTests(unittest.TestCase):
    def test_edge_orchestrator_session_emits_ack_and_remote_tool_request(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
            )
            session = EdgeOrchestratorSession(
                config,
                supervisor_decision_provider=_FakeDecisionProvider(),
                specialist_provider=_FakeSpecialistProvider(),
                tool_names=("search_live_info",),
            )
            events: list[dict[str, object]] = []
            tool_bridge = None

            def emit(payload: dict[str, object]) -> None:
                events.append(payload)
                if payload.get("type") == "tool_request":
                    tool_bridge.submit_result(
                        str(payload["call_id"]),
                        output={"answer": "Morgen wird es sonnig."},
                        error=None,
                    )

            tool_bridge = RemoteToolBridge(emit)

            result = session.run_turn(
                "Wie wird das Wetter morgen?",
                conversation=(),
                supervisor_conversation=(),
                emit_event=emit,
                tool_bridge=tool_bridge,
            )

        self.assertEqual(events[0]["type"], "ack")
        self.assertEqual(events[0]["ack_id"], "checking_now")
        self.assertEqual(events[1]["type"], "tool_request")
        self.assertEqual(events[1]["name"], "search_live_info")
        self.assertEqual(result.text, "Morgen wird es sonnig.")
        self.assertTrue(result.used_web_search)


class OrchestratorServerTests(unittest.TestCase):
    def test_server_websocket_bridges_tool_request_and_result(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
            )
            app = EdgeOrchestratorServer(config, session_factory=lambda _config: _FakeServerSession()).create_app()
            with TestClient(app) as client:
                with client.websocket_connect("/ws/orchestrator") as websocket:
                    websocket.send_json(OrchestratorTurnRequest(prompt="Wie wird das Wetter morgen?").to_payload())
                    ack = websocket.receive_json()
                    tool_request = websocket.receive_json()
                    websocket.send_json(
                        {
                            "type": "tool_result",
                            "call_id": tool_request["call_id"],
                            "ok": True,
                            "output": {"answer": "Morgen wird es sonnig."},
                        }
                    )
                    completed = websocket.receive_json()

        self.assertEqual(ack["type"], "ack")
        self.assertEqual(ack_text_for_id(ack["ack_id"]), "Ich schaue kurz nach.")
        self.assertEqual(tool_request["type"], "tool_request")
        self.assertEqual(completed["type"], "turn_complete")
        self.assertEqual(completed["text"], "Morgen wird es sonnig.")


class OrchestratorClientTests(unittest.TestCase):
    def test_client_handles_ack_tool_request_and_completion(self) -> None:
        class _FakeSocket:
            def __init__(self):
                self.sent: list[str] = []
                self.messages = iter(
                    [
                        '{"type":"ack","ack_id":"checking_now","text":"Ich schaue kurz nach."}',
                        '{"type":"tool_request","call_id":"call-1","name":"search_live_info","arguments":{"question":"Wie wird das Wetter morgen?"}}',
                        '{"type":"turn_complete","text":"Morgen wird es sonnig.","rounds":2,"used_web_search":true,"response_id":"resp-1","request_id":"req-1","model":"gpt-4o-mini","tool_calls":[],"tool_results":[]}',
                    ]
                )

            def send(self, payload: str) -> None:
                self.sent.append(payload)

            def recv(self) -> str:
                return next(self.messages)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        ack_events = []
        connector = lambda *args, **kwargs: _FakeSocket()
        client = OrchestratorWebSocketClient("ws://test/ws", connector=connector)

        result = client.run_turn(
            OrchestratorTurnRequest(prompt="Wie wird das Wetter morgen?"),
            tool_handlers={"search_live_info": lambda arguments: {"answer": arguments["question"]}},
            on_ack=ack_events.append,
        )

        self.assertEqual(len(ack_events), 1)
        self.assertEqual(ack_events[0].ack_id, "checking_now")
        self.assertEqual(result.text, "Morgen wird es sonnig.")


class OrchestratorContractTests(unittest.TestCase):
    def test_turn_complete_payload_serializes_model_like_token_usage(self) -> None:
        token_usage = type(
            "TokenUsage",
            (),
            {
                "__init__": lambda self: setattr(self, "total_tokens", 42),
            },
        )()
        payload = OrchestratorTurnCompleteEvent(
            text="Hallo!",
            rounds=1,
            used_web_search=False,
            token_usage=token_usage,
        ).to_payload()
        self.assertEqual(payload["token_usage"], {"total_tokens": 42})

    def test_tool_response_payload_sanitizes_nested_non_json_output(self) -> None:
        payload = OrchestratorToolResponse(
            call_id="call-1",
            ok=True,
            output={"meta": {"path": Path("/tmp/test.txt")}},
        ).to_payload()
        self.assertEqual(payload["output"], {"meta": {"path": "/tmp/test.txt"}})


if __name__ == "__main__":
    unittest.main()
