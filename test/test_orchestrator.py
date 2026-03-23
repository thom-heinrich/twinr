from dataclasses import dataclass
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest
from urllib import error as urllib_error
from unittest.mock import patch
from types import SimpleNamespace

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.contracts import AgentToolCall, ToolCallingTurnResponse
from twinr.agent.base_agent.config import TwinrConfig
from twinr.orchestrator.acks import ack_text_for_id
from twinr.orchestrator.client import OrchestratorWebSocketClient
from twinr.orchestrator.contracts import OrchestratorToolResponse, OrchestratorTurnCompleteEvent
from twinr.orchestrator.contracts import OrchestratorTurnRequest
from twinr.orchestrator.local_stt import LocalSttBackendAdapter
from twinr.orchestrator.server import EdgeOrchestratorServer, create_app as create_orchestrator_app
from twinr.orchestrator.session import EdgeOrchestratorSession, RemoteToolBridge
from twinr.orchestrator.voice_client import OrchestratorVoiceWebSocketClient
from twinr.orchestrator.voice_contracts import (
    OrchestratorVoiceAudioFrame,
    OrchestratorVoiceHelloRequest,
    OrchestratorVoiceRuntimeStateEvent,
)
from twinr.orchestrator.voice_session import EdgeOrchestratorVoiceSession


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


class _FakeVoiceServerSession:
    def handle_hello(self, request):
        del request
        return [{"type": "voice_ready", "session_id": "voice-1", "backend": "wekws"}]

    def handle_runtime_state(self, event):
        del event
        return []

    def handle_audio_frame(self, frame):
        del frame
        return [
            {
                "type": "wake_confirmed",
                "matched_phrase": "twinna",
                "remaining_text": "schau mal im web",
                "backend": "wekws",
                "detector_label": "twinna",
                "score": 0.81,
            }
        ]


class _FakeFrameSpotter:
    def __init__(self) -> None:
        self.calls = 0

    def process_pcm_bytes(self, pcm_bytes: bytes, *, channels: int = 1):
        del pcm_bytes, channels
        self.calls += 1
        if self.calls == 1:
            from twinr.proactive.wakeword.matching import WakewordMatch

            return WakewordMatch(
                detected=True,
                transcript="",
                matched_phrase="twinna",
                remaining_text="",
                backend="wekws",
                detector_label="twinna",
                score=0.81,
            )
        return None

    def reset(self) -> None:
        return None


class _NeverMatchFrameSpotter:
    def process_pcm_bytes(self, pcm_bytes: bytes, *, channels: int = 1):
        del pcm_bytes, channels
        return None

    def reset(self) -> None:
        return None


class _FakeWakePhraseSpotter:
    def detect(self, capture):
        del capture
        from twinr.proactive.wakeword.matching import WakewordMatch

        return WakewordMatch(
            detected=True,
            transcript="Twinna schau mal im Web",
            matched_phrase="twinna",
            remaining_text="schau mal im web",
            backend="stt",
            detector_label="twinna",
            score=0.92,
        )


class _FakeWakeTailExtractor:
    def extract(self, capture):
        del capture
        return "schau mal im web"


class _CountingTranscriptBackend:
    def __init__(self, transcript: str) -> None:
        self.transcript = transcript
        self.calls = 0

    def transcribe(self, *args, **kwargs):
        del args, kwargs
        self.calls += 1
        return self.transcript


class _MinDurationWakePhraseSpotter:
    def __init__(self, *, min_duration_ms: int, remaining_text: str = "wie geht es dir") -> None:
        self.min_duration_ms = min_duration_ms
        self.remaining_text = remaining_text
        self.capture_durations_ms: list[int] = []

    def detect(self, capture):
        self.capture_durations_ms.append(int(capture.sample.duration_ms))
        from twinr.proactive.wakeword.matching import WakewordMatch

        if int(capture.sample.duration_ms) < self.min_duration_ms:
            return WakewordMatch(
                detected=False,
                transcript="",
                backend="stt",
            )
        return WakewordMatch(
            detected=True,
            transcript=f"Twinna {self.remaining_text}",
            matched_phrase="twinna",
            remaining_text=self.remaining_text,
            backend="stt",
            detector_label="twinna",
            score=0.93,
        )


class _TranscriptOnlyWakeTailExtractor:
    def __init__(self, transcript: str) -> None:
        self.transcript = transcript

    def extract(self, capture):
        del capture
        return self.transcript


class _RejectingWakePhraseSpotter:
    def detect(self, capture):
        del capture
        from twinr.proactive.wakeword.matching import WakewordMatch

        return WakewordMatch(
            detected=False,
            transcript="Maurizio Puppe",
            normalized_transcript="maurizio puppe",
            backend="stt",
        )


class _ExplodingWakePhraseSpotter:
    def detect(self, capture):
        del capture
        raise AssertionError("backend-led wake confirmation must not call wake phrase STT")


class _LengthSensitiveWakePhraseSpotter:
    def __init__(self, *, min_duration_ms: int) -> None:
        self.min_duration_ms = int(min_duration_ms)
        self.capture_durations_ms: list[int] = []

    def detect(self, capture):
        self.capture_durations_ms.append(int(capture.sample.duration_ms))
        from twinr.proactive.wakeword.matching import WakewordMatch

        if int(capture.sample.duration_ms) < self.min_duration_ms:
            return WakewordMatch(
                detected=False,
                transcript="",
                backend="stt",
            )
        return WakewordMatch(
            detected=True,
            transcript="Twinna",
            matched_phrase="twinna",
            remaining_text="",
            normalized_transcript="twinna",
            backend="stt",
            detector_label="twinna",
            score=0.92,
        )


class _ShortPrefixWakePhraseSpotter:
    def __init__(self, *, min_duration_ms: int, max_duration_ms: int) -> None:
        self.min_duration_ms = int(min_duration_ms)
        self.max_duration_ms = int(max_duration_ms)
        self.capture_durations_ms: list[int] = []

    def detect(self, capture):
        self.capture_durations_ms.append(int(capture.sample.duration_ms))
        from twinr.proactive.wakeword.matching import WakewordMatch

        duration_ms = int(capture.sample.duration_ms)
        if duration_ms < self.min_duration_ms or duration_ms > self.max_duration_ms:
            return WakewordMatch(
                detected=False,
                transcript="schau mal im web nach dem wetter in berlin",
                normalized_transcript="schau mal im web nach dem wetter in berlin",
                backend="stt",
            )
        return WakewordMatch(
            detected=True,
            transcript="Twinner",
            matched_phrase="twinna",
            remaining_text="",
            normalized_transcript="twinner",
            backend="stt",
            detector_label="twinna",
            score=0.92,
        )


class _GrowingWakePhraseSpotter:
    def __init__(self) -> None:
        self.capture_durations_ms: list[int] = []

    def detect(self, capture):
        self.capture_durations_ms.append(int(capture.sample.duration_ms))
        from twinr.proactive.wakeword.matching import WakewordMatch

        duration_ms = int(capture.sample.duration_ms)
        if duration_ms < 100:
            return WakewordMatch(
                detected=False,
                transcript="",
                backend="stt",
            )
        if duration_ms < 300:
            return WakewordMatch(
                detected=True,
                transcript="Twinner",
                matched_phrase="twinna",
                remaining_text="",
                normalized_transcript="twinner",
                backend="stt",
                detector_label="twinna",
                score=0.92,
            )
        return WakewordMatch(
            detected=True,
            transcript="Twinner schau mal im web",
            matched_phrase="twinna",
            remaining_text="schau mal im web",
            normalized_transcript="twinner schau mal im web",
            backend="stt",
            detector_label="twinna",
            score=0.92,
        )


class _BurstAnchoredWakePhraseSpotter:
    def __init__(self, *, wake_prefix: bytes, min_duration_ms: int, max_duration_ms: int) -> None:
        self.wake_prefix = wake_prefix
        self.min_duration_ms = int(min_duration_ms)
        self.max_duration_ms = int(max_duration_ms)
        self.capture_prefixes: list[bytes] = []
        self.capture_durations_ms: list[int] = []

    def detect(self, capture):
        self.capture_prefixes.append(bytes(capture.pcm_bytes[: len(self.wake_prefix)]))
        self.capture_durations_ms.append(int(capture.sample.duration_ms))
        from twinr.proactive.wakeword.matching import WakewordMatch

        duration_ms = int(capture.sample.duration_ms)
        if (
            duration_ms < self.min_duration_ms
            or duration_ms > self.max_duration_ms
            or not bytes(capture.pcm_bytes).startswith(self.wake_prefix)
        ):
            return WakewordMatch(
                detected=False,
                transcript="schau mal im web nach dem wetter in berlin",
                normalized_transcript="schau mal im web nach dem wetter in berlin",
                backend="stt",
            )
        return WakewordMatch(
            detected=True,
            transcript="Twinner",
            matched_phrase="twinna",
            remaining_text="",
            normalized_transcript="twinner",
            backend="stt",
            detector_label="twinna",
            score=0.92,
        )


def _pcm_frame(value: int) -> bytes:
    """Build one mono PCM16 frame with a stable amplitude for orchestrator tests."""

    sample = int(value).to_bytes(2, byteorder="little", signed=True)
    return sample * 1600


class _FakeUrlOpenResponse:
    def __init__(self, payload: bytes, *, status: int = 200) -> None:
        self._payload = payload
        self.status = status

    def read(self) -> bytes:
        return self._payload

    def getcode(self) -> int:
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


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
                orchestrator_shared_secret="secret-token",
            )
            app = EdgeOrchestratorServer(config, session_factory=lambda _config: _FakeServerSession()).create_app()
            with TestClient(app) as client:
                with client.websocket_connect(
                    "/ws/orchestrator",
                    headers={"x-twinr-secret": "secret-token"},
                ) as websocket:
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

    def test_server_voice_websocket_emits_ready_and_wake_confirmed(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                orchestrator_shared_secret="secret-token",
            )
            app = EdgeOrchestratorServer(
                config,
                voice_session_factory=lambda _config: _FakeVoiceServerSession(),
            ).create_app()
            with TestClient(app) as client:
                with client.websocket_connect(
                    "/ws/orchestrator/voice",
                    headers={"x-twinr-secret": "secret-token"},
                ) as websocket:
                    websocket.send_json(
                        OrchestratorVoiceHelloRequest(
                            session_id="voice-1",
                            sample_rate=16000,
                            channels=1,
                            chunk_ms=100,
                        ).to_payload()
                    )
                    ready = websocket.receive_json()
                    websocket.send_json(
                        OrchestratorVoiceAudioFrame(
                            sequence=0,
                            pcm_bytes=b"\x00\x00" * 1600,
                        ).to_payload()
                    )
                    wake = websocket.receive_json()

        self.assertEqual(ready["type"], "voice_ready")
        self.assertEqual(wake["type"], "wake_confirmed")
        self.assertEqual(wake["remaining_text"], "schau mal im web")

    def test_create_app_rejects_non_transcript_first_voice_gateway(self) -> None:
        with TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "OPENAI_API_KEY=test-key",
                        "TWINR_ORCHESTRATOR_SHARED_SECRET=secret-token",
                        "TWINR_VOICE_ORCHESTRATOR_WAKE_STAGE1_MODE=backend",
                        "TWINR_VOICE_ORCHESTRATOR_LOCAL_STT_URL=http://127.0.0.1:18090",
                        "TWINR_WAKEWORD_ENABLED=false",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "transcript-first local_stt stage1 only"):
                create_orchestrator_app(env_path)

    def test_create_app_rejects_generic_gateway_wakeword_backends(self) -> None:
        with TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "OPENAI_API_KEY=test-key",
                        "TWINR_ORCHESTRATOR_SHARED_SECRET=secret-token",
                        "TWINR_VOICE_ORCHESTRATOR_WAKE_STAGE1_MODE=local_stt",
                        "TWINR_VOICE_ORCHESTRATOR_LOCAL_STT_URL=http://127.0.0.1:18090",
                        "TWINR_WAKEWORD_ENABLED=true",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "generic wakeword backends disabled"):
                create_orchestrator_app(env_path)


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

            def recv(self, timeout=None) -> str:
                del timeout
                return next(self.messages)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        ack_events = []
        connector = lambda *args, **kwargs: _FakeSocket()
        client = OrchestratorWebSocketClient("ws://127.0.0.1/ws", connector=connector, require_tls=False)

        result = client.run_turn(
            OrchestratorTurnRequest(prompt="Wie wird das Wetter morgen?"),
            tool_handlers={"search_live_info": lambda arguments: {"answer": arguments["question"]}},
            on_ack=ack_events.append,
        )

        self.assertEqual(len(ack_events), 1)
        self.assertEqual(ack_events[0].ack_id, "checking_now")
        self.assertEqual(result.text, "Morgen wird es sonnig.")

    def test_voice_client_decodes_ready_and_wake_events(self) -> None:
        class _FakeSocket:
            def __init__(self):
                self.sent: list[str] = []
                self.messages = iter(
                    [
                        '{"type":"voice_ready","session_id":"voice-1","backend":"wekws"}',
                        '{"type":"wake_confirmed","matched_phrase":"twinna","remaining_text":"schau mal im web","backend":"wekws","detector_label":"twinna","score":0.9}',
                    ]
                )

            def send(self, payload: str) -> None:
                self.sent.append(payload)

            def recv(self, timeout=None) -> str:
                del timeout
                return next(self.messages)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        events = []
        client = OrchestratorVoiceWebSocketClient(
            "ws://127.0.0.1/ws",
            connector=lambda *args, **kwargs: _FakeSocket(),
            on_event=events.append,
            require_tls=False,
        )

        client.open()
        client.send_hello(
            OrchestratorVoiceHelloRequest(
                session_id="voice-1",
                sample_rate=16000,
                channels=1,
                chunk_ms=100,
            )
        )
        client.send_runtime_state(
            OrchestratorVoiceRuntimeStateEvent(
                state="waiting",
                follow_up_allowed=False,
            )
        )
        client.close()

        self.assertTrue(any(getattr(event, "backend", "") == "wekws" for event in events))
        self.assertTrue(any(getattr(event, "remaining_text", "") == "schau mal im web" for event in events))

    def test_voice_client_close_tolerates_receiver_thread_starting_late(self) -> None:
        class _FakeSocket:
            def __init__(self):
                self.sent: list[str] = []

            def send(self, payload: str) -> None:
                self.sent.append(payload)

            def recv(self, timeout=None) -> str:
                raise TimeoutError()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class _ImmediateWaitEvent:
            def clear(self) -> None:
                return None

            def set(self) -> None:
                return None

            def wait(self, timeout=None) -> bool:
                del timeout
                return True

            def is_set(self) -> bool:
                return False

        class _DeferredThread:
            latest = None

            def __init__(self, *, target, args=(), daemon=None, name=None):
                del daemon, name
                self._target = target
                self._args = args
                _DeferredThread.latest = self

            def start(self) -> None:
                return None

            def join(self, timeout=None) -> None:
                del timeout

            def is_alive(self) -> bool:
                return True

            def run_now(self) -> None:
                self._target(*self._args)

        client = OrchestratorVoiceWebSocketClient(
            "ws://127.0.0.1/ws",
            connector=lambda *args, **kwargs: _FakeSocket(),
            on_event=lambda event: None,
            require_tls=False,
        )
        client._receiver_started = _ImmediateWaitEvent()

        with patch("twinr.orchestrator.voice_client.Thread", _DeferredThread):
            client.open()
            client.send_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            client.close()
            self.assertIsNotNone(_DeferredThread.latest)
            _DeferredThread.latest.run_now()


class OrchestratorVoiceSessionTests(unittest.TestCase):
    def test_local_stt_backend_adapter_posts_audio_and_returns_text(self) -> None:
        captured = {}

        def _fake_urlopen(request, timeout=0):
            captured["url"] = request.full_url
            captured["headers"] = dict(request.header_items())
            captured["body"] = request.data
            captured["timeout"] = timeout
            return _FakeUrlOpenResponse(
                b'{"text":"Hallo, Twinner!","language":"de","segments":[{"start":0.0,"end":1.0,"text":"Hallo, Twinner!"}],"duration_sec":1.0}'
            )

        adapter = LocalSttBackendAdapter(
            base_url="http://127.0.0.1:18090",
            bearer_token="secret",
            language="de",
            mode="active_listening",
            timeout_s=2.5,
        )

        with patch("urllib.request.urlopen", _fake_urlopen):
            text = adapter.transcribe(b"RIFFdata", filename="wakeword.wav", content_type="audio/wav")

        self.assertEqual(text, "Hallo, Twinner!")
        self.assertEqual(captured["url"], "http://127.0.0.1:18090/v1/transcribe")
        self.assertEqual(captured["timeout"], 2.5)
        self.assertEqual(captured["headers"]["Authorization"], "Bearer secret")
        self.assertIn(b'name="language"', captured["body"])
        self.assertIn(b"active_listening", captured["body"])
        self.assertIn(b'filename="wakeword.wav"', captured["body"])
        self.assertIn(b"RIFFdata", captured["body"])

    def test_local_stt_backend_adapter_retries_busy_service_once(self) -> None:
        attempts: list[int] = []
        sleep_calls: list[float] = []

        def _fake_urlopen(request, timeout=0):
            del request, timeout
            attempts.append(1)
            if len(attempts) == 1:
                raise urllib_error.HTTPError(
                    url="http://127.0.0.1:18090/v1/transcribe",
                    code=429,
                    msg="Too Many Requests",
                    hdrs=None,
                    fp=io.BytesIO(b'{"detail":"stt busy"}'),
                )
            return _FakeUrlOpenResponse(
                b'{"text":"Hey Twinna","language":"de","segments":[],"duration_sec":1.0}'
            )

        adapter = LocalSttBackendAdapter(
            base_url="http://127.0.0.1:18090",
            language="de",
            mode="active_listening",
            timeout_s=2.5,
            retry_attempts=1,
            retry_backoff_s=0.4,
        )

        with patch("urllib.request.urlopen", _fake_urlopen), patch(
            "time.sleep", lambda seconds: sleep_calls.append(seconds)
        ):
            text = adapter.transcribe(b"RIFFdata", filename="wakeword.wav", content_type="audio/wav")

        self.assertEqual(text, "Hey Twinna")
        self.assertEqual(len(attempts), 2)
        self.assertEqual(sleep_calls, [0.4])

    def test_voice_session_confirms_backend_wakeword_after_postroll(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                voice_orchestrator_wake_postroll_ms=100,
                voice_orchestrator_wake_tail_max_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "Twinna schau mal im Web"),
                frame_spotter=_FakeFrameSpotter(),
                wake_phrase_spotter=_ExplodingWakePhraseSpotter(),
                backend_tail_transcript_extractor=_FakeWakeTailExtractor(),
            )

            ready = session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            self.assertEqual(ready[0]["type"], "voice_ready")
            first = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=b"\x01\x00" * 1600)
            )
            second = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=b"\x01\x00" * 1600)
            )

        self.assertEqual(first, [])
        self.assertEqual(second[0]["type"], "wake_confirmed")
        self.assertEqual(second[0]["matched_phrase"], "twinna")
        self.assertEqual(second[0]["remaining_text"], "schau mal im web")

    def test_voice_session_backend_stage1_waits_for_tail_endpoint_before_confirming(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
                voice_orchestrator_wake_postroll_ms=100,
                voice_orchestrator_wake_tail_endpoint_silence_ms=200,
                voice_orchestrator_wake_tail_max_ms=800,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "Twinna schau mal im Web"),
                frame_spotter=_FakeFrameSpotter(),
                wake_phrase_spotter=_ExplodingWakePhraseSpotter(),
                backend_tail_transcript_extractor=_FakeWakeTailExtractor(),
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            first = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(2))
            )
            second = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=_pcm_frame(2))
            )
            third = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=2, pcm_bytes=_pcm_frame(0))
            )
            fourth = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=3, pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(first, [])
        self.assertEqual(second, [])
        self.assertEqual(third, [])
        self.assertEqual(fourth[0]["type"], "wake_confirmed")
        self.assertEqual(fourth[0]["remaining_text"], "schau mal im web")

    def test_voice_session_backend_stage1_confirms_wake_only_after_post_wake_activity_and_silence_endpoint(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
                voice_orchestrator_wake_postroll_ms=100,
                voice_orchestrator_wake_tail_endpoint_silence_ms=200,
                voice_orchestrator_wake_tail_max_ms=800,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "Twinna"),
                frame_spotter=_FakeFrameSpotter(),
                wake_phrase_spotter=_ExplodingWakePhraseSpotter(),
                backend_tail_transcript_extractor=_TranscriptOnlyWakeTailExtractor(""),
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            first = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(2))
            )
            second = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=_pcm_frame(2))
            )
            third = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=2, pcm_bytes=_pcm_frame(0))
            )
            fourth = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=3, pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(first, [])
        self.assertEqual(second, [])
        self.assertEqual(third, [])
        self.assertEqual(fourth[0]["type"], "wake_confirmed")
        self.assertEqual(fourth[0]["remaining_text"], "")

    def test_voice_session_follow_up_window_routes_repeated_wake_phrase_as_fresh_wake(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
                voice_orchestrator_follow_up_window_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "Twinna wie geht es dir"),
                frame_spotter=_NeverMatchFrameSpotter(),
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            session.handle_runtime_state(
                OrchestratorVoiceRuntimeStateEvent(
                    state="follow_up_open",
                    detail="wakeword",
                    follow_up_allowed=True,
                )
            )
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(2))
            )

        self.assertEqual(events[0]["type"], "wake_confirmed")
        self.assertEqual(events[0]["matched_phrase"], "twinna")
        self.assertEqual(events[0]["remaining_text"], "wie geht es dir")

    def test_voice_session_follow_up_window_keeps_stage1_wake_detection_active(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
                voice_orchestrator_wake_postroll_ms=100,
                voice_orchestrator_wake_tail_max_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                frame_spotter=_FakeFrameSpotter(),
                wake_phrase_spotter=_ExplodingWakePhraseSpotter(),
                backend_tail_transcript_extractor=_FakeWakeTailExtractor(),
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            session.handle_runtime_state(
                OrchestratorVoiceRuntimeStateEvent(
                    state="follow_up_open",
                    detail="wakeword",
                    follow_up_allowed=True,
                )
            )
            first = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(2))
            )
            second = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=_pcm_frame(2))
            )

        self.assertEqual(first, [])
        self.assertEqual(second[0]["type"], "wake_confirmed")
        self.assertEqual(second[0]["matched_phrase"], "twinna")
        self.assertEqual(second[0]["remaining_text"], "schau mal im web")

    def test_voice_session_follow_up_window_requests_local_capture_without_wake_phrase(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
                voice_orchestrator_follow_up_window_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "wie geht es dir"),
                frame_spotter=_NeverMatchFrameSpotter(),
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            session.handle_runtime_state(
                OrchestratorVoiceRuntimeStateEvent(
                    state="follow_up_open",
                    detail="wakeword",
                    follow_up_allowed=True,
                )
            )
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(2))
            )

        self.assertEqual(events[0]["type"], "follow_up_capture_requested")

    def test_voice_session_follow_up_window_waits_for_full_window_before_generic_capture(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

            def step(self, seconds: float) -> None:
                self.value += seconds

        with TemporaryDirectory() as temp_dir:
            clock = _FakeClock()
            backend = _CountingTranscriptBackend("Untertitel der Amara.org-Community")
            wake_phrase_spotter = _MinDurationWakePhraseSpotter(min_duration_ms=300, remaining_text="wie geht es dir")
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
                voice_orchestrator_follow_up_window_ms=300,
                voice_orchestrator_wake_stage1_mode="local_stt",
                voice_orchestrator_local_stt_url="http://127.0.0.1:18090",
                voice_orchestrator_wake_postroll_ms=100,
                voice_orchestrator_wake_tail_max_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=backend,
                wake_phrase_spotter=wake_phrase_spotter,
                backend_tail_transcript_extractor=_TranscriptOnlyWakeTailExtractor("wie geht es dir"),
                monotonic_fn=clock,
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            session.handle_runtime_state(
                OrchestratorVoiceRuntimeStateEvent(
                    state="follow_up_open",
                    detail="wakeword",
                    follow_up_allowed=True,
                )
            )
            first = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(2))
            )
            clock.step(0.1)
            second = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=_pcm_frame(2))
            )
            clock.step(0.1)
            third = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=2, pcm_bytes=_pcm_frame(2))
            )
            clock.step(0.1)
            fourth = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=3, pcm_bytes=_pcm_frame(2))
            )

        self.assertEqual(first, [])
        self.assertEqual(second, [])
        self.assertEqual(third, [])
        self.assertEqual(fourth[0]["type"], "wake_confirmed")
        self.assertEqual(fourth[0]["matched_phrase"], "twinna")
        self.assertEqual(fourth[0]["remaining_text"], "wie geht es dir")
        self.assertEqual(backend.calls, 0)

    def test_voice_session_backend_stage1_confirms_at_tail_max_without_endpoint(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
                voice_orchestrator_wake_postroll_ms=100,
                voice_orchestrator_wake_tail_endpoint_silence_ms=300,
                voice_orchestrator_wake_tail_max_ms=300,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "Twinna schau mal im Web"),
                frame_spotter=_FakeFrameSpotter(),
                wake_phrase_spotter=_ExplodingWakePhraseSpotter(),
                backend_tail_transcript_extractor=_FakeWakeTailExtractor(),
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            first = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(2))
            )
            second = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=_pcm_frame(2))
            )
            third = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=2, pcm_bytes=_pcm_frame(2))
            )
            fourth = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=3, pcm_bytes=_pcm_frame(2))
            )

        self.assertEqual(first, [])
        self.assertEqual(second, [])
        self.assertEqual(third, [])
        self.assertEqual(fourth[0]["type"], "wake_confirmed")
        self.assertEqual(fourth[0]["remaining_text"], "schau mal im web")

    def test_voice_session_backend_stage1_waits_for_tail_max_until_post_wake_activity_is_seen(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1000,
                voice_orchestrator_wake_postroll_ms=100,
                voice_orchestrator_wake_tail_endpoint_silence_ms=200,
                voice_orchestrator_wake_tail_max_ms=400,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                frame_spotter=_FakeFrameSpotter(),
                wake_phrase_spotter=_ExplodingWakePhraseSpotter(),
                backend_tail_transcript_extractor=_FakeWakeTailExtractor(),
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            first = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(0))
            )
            second = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=_pcm_frame(0))
            )
            third = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=2, pcm_bytes=_pcm_frame(0))
            )
            fourth = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=3, pcm_bytes=_pcm_frame(0))
            )
            fifth = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=4, pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(first, [])
        self.assertEqual(second, [])
        self.assertEqual(third, [])
        self.assertEqual(fourth, [])
        self.assertEqual(fifth[0]["type"], "wake_confirmed")
        self.assertEqual(fifth[0]["remaining_text"], "schau mal im web")

    def test_voice_session_backend_stage1_uses_trimmed_tail_transcript_after_confirmed_wake(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                voice_orchestrator_wake_postroll_ms=100,
                voice_orchestrator_wake_tail_max_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "Maurizio Puppe"),
                frame_spotter=_FakeFrameSpotter(),
                wake_phrase_spotter=_ExplodingWakePhraseSpotter(),
                backend_tail_transcript_extractor=_TranscriptOnlyWakeTailExtractor(
                    "schau mal im web nach dem wetter in berlin"
                ),
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            first = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=b"\x01\x00" * 1600)
            )
            second = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=b"\x01\x00" * 1600)
            )

        self.assertEqual(first, [])
        self.assertEqual(second[0]["type"], "wake_confirmed")
        self.assertEqual(second[0]["matched_phrase"], "twinna")
        self.assertEqual(second[0]["remaining_text"], "schau mal im web nach dem wetter in berlin")
        self.assertEqual(second[0]["backend"], "wekws")

    def test_voice_session_uses_local_stt_stage1_for_wake_candidates(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

            def step(self, seconds: float) -> None:
                self.value += seconds

        with TemporaryDirectory() as temp_dir:
            clock = _FakeClock()
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1000,
                voice_orchestrator_wake_stage1_mode="local_stt",
                voice_orchestrator_local_stt_url="http://127.0.0.1:18090",
                voice_orchestrator_wake_candidate_window_ms=100,
                voice_orchestrator_wake_postroll_ms=100,
                voice_orchestrator_wake_tail_max_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=_FakeWakePhraseSpotter(),
                backend_tail_transcript_extractor=_FakeWakeTailExtractor(),
                monotonic_fn=clock,
            )
            self.assertEqual(session.wake_candidate_min_active_ratio, 0.0)

            ready = session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            first = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=b"\x01\x00" * 1600)
            )
            clock.step(1.0)
            second = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=b"\x01\x00" * 1600)
            )

        self.assertEqual(ready[0]["backend"], "local_stt")
        self.assertEqual(first, [])
        self.assertEqual(second[0]["type"], "wake_confirmed")
        self.assertEqual(second[0]["remaining_text"], "schau mal im web")

    def test_voice_session_local_stt_stage1_still_requires_transcript_match(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

            def step(self, seconds: float) -> None:
                self.value += seconds

        with TemporaryDirectory() as temp_dir:
            clock = _FakeClock()
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1000,
                voice_orchestrator_wake_stage1_mode="local_stt",
                voice_orchestrator_local_stt_url="http://127.0.0.1:18090",
                voice_orchestrator_wake_candidate_window_ms=100,
                voice_orchestrator_wake_postroll_ms=100,
                voice_orchestrator_wake_tail_max_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=_RejectingWakePhraseSpotter(),
                backend_tail_transcript_extractor=_FakeWakeTailExtractor(),
                monotonic_fn=clock,
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            first = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=b"\x01\x00" * 1600)
            )
            clock.step(1.0)
            second = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=b"\x01\x00" * 1600)
            )

        self.assertEqual(first, [])
        self.assertEqual(second, [])

    def test_voice_session_local_stt_stage1_falls_back_to_frame_spotter_for_short_wake_only_turns(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

            def step(self, seconds: float) -> None:
                self.value += seconds

        with TemporaryDirectory() as temp_dir:
            clock = _FakeClock()
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1000,
                voice_orchestrator_wake_stage1_mode="local_stt",
                voice_orchestrator_local_stt_url="http://127.0.0.1:18090",
                voice_orchestrator_wake_candidate_window_ms=100,
                voice_orchestrator_wake_postroll_ms=100,
                voice_orchestrator_wake_tail_max_ms=100,
            )
            frame_spotter = _FakeFrameSpotter()
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                frame_spotter=frame_spotter,
                wake_phrase_spotter=_RejectingWakePhraseSpotter(),
                backend_tail_transcript_extractor=_FakeWakeTailExtractor(),
                monotonic_fn=clock,
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            first = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=b"\x01\x00" * 1600)
            )
            clock.step(0.1)
            second = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=b"\x01\x00" * 1600)
            )

        self.assertEqual(first, [])
        self.assertEqual(second[0]["type"], "wake_confirmed")
        self.assertEqual(second[0]["matched_phrase"], "twinna")
        self.assertEqual(second[0]["remaining_text"], "schau mal im web")
        self.assertEqual(second[0]["backend"], "wekws")
        self.assertEqual(frame_spotter.calls, 1)

    def test_voice_session_local_stt_frame_spotter_fallback_beats_follow_up_capture_for_short_rewake(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

            def step(self, seconds: float) -> None:
                self.value += seconds

        with TemporaryDirectory() as temp_dir:
            clock = _FakeClock()
            backend = _CountingTranscriptBackend("Untertitel der Amara.org-Community")
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
                voice_orchestrator_follow_up_window_ms=100,
                voice_orchestrator_wake_stage1_mode="local_stt",
                voice_orchestrator_local_stt_url="http://127.0.0.1:18090",
                voice_orchestrator_wake_candidate_window_ms=100,
                voice_orchestrator_wake_postroll_ms=100,
                voice_orchestrator_wake_tail_max_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=backend,
                frame_spotter=_FakeFrameSpotter(),
                wake_phrase_spotter=_RejectingWakePhraseSpotter(),
                backend_tail_transcript_extractor=_FakeWakeTailExtractor(),
                monotonic_fn=clock,
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            session.handle_runtime_state(
                OrchestratorVoiceRuntimeStateEvent(
                    state="follow_up_open",
                    detail="wakeword",
                    follow_up_allowed=True,
                )
            )
            first = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(2))
            )
            clock.step(0.1)
            second = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=_pcm_frame(2))
            )

        self.assertEqual(first, [])
        self.assertEqual(second[0]["type"], "wake_confirmed")
        self.assertEqual(second[0]["matched_phrase"], "twinna")
        self.assertEqual(second[0]["remaining_text"], "schau mal im web")
        self.assertEqual(backend.calls, 0)

    def test_voice_session_local_stt_stage1_expands_with_available_history_up_to_short_scan_window(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

            def step(self, seconds: float) -> None:
                self.value += seconds

        with TemporaryDirectory() as temp_dir:
            clock = _FakeClock()
            wake_phrase_spotter = _LengthSensitiveWakePhraseSpotter(min_duration_ms=200)
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1000,
                voice_orchestrator_history_ms=300,
                voice_orchestrator_wake_stage1_mode="local_stt",
                voice_orchestrator_local_stt_url="http://127.0.0.1:18090",
                voice_orchestrator_wake_candidate_window_ms=100,
                voice_orchestrator_wake_postroll_ms=100,
                voice_orchestrator_wake_tail_max_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=wake_phrase_spotter,
                backend_tail_transcript_extractor=_FakeWakeTailExtractor(),
                monotonic_fn=clock,
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            first = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=b"\x01\x00" * 1600)
            )
            clock.step(1.0)
            second = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=b"\x01\x00" * 1600)
            )
            third = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=2, pcm_bytes=b"\x01\x00" * 1600)
            )

        self.assertEqual(first, [])
        self.assertEqual(second, [])
        self.assertEqual(third[0]["type"], "wake_confirmed")
        self.assertEqual(wake_phrase_spotter.capture_durations_ms, [100, 200])

    def test_voice_session_local_stt_stage1_uses_short_prefix_before_full_sentence_capture(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

            def step(self, seconds: float) -> None:
                self.value += seconds

        with TemporaryDirectory() as temp_dir:
            clock = _FakeClock()
            wake_phrase_spotter = _ShortPrefixWakePhraseSpotter(
                min_duration_ms=800,
                max_duration_ms=1000,
            )
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1000,
                voice_orchestrator_history_ms=2400,
                voice_orchestrator_wake_stage1_mode="local_stt",
                voice_orchestrator_local_stt_url="http://127.0.0.1:18090",
                voice_orchestrator_wake_candidate_window_ms=2200,
                voice_orchestrator_wake_postroll_ms=100,
                voice_orchestrator_wake_tail_max_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=wake_phrase_spotter,
                backend_tail_transcript_extractor=_FakeWakeTailExtractor(),
                monotonic_fn=clock,
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            for sequence in range(10):
                session.handle_audio_frame(
                    OrchestratorVoiceAudioFrame(sequence=sequence, pcm_bytes=b"\x01\x00" * 1600)
                )
            clock.step(1.0)
            wake_candidate = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=10, pcm_bytes=b"\x01\x00" * 1600)
            )
            confirmed = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=11, pcm_bytes=b"\x01\x00" * 1600)
            )

        self.assertEqual(wake_candidate, [])
        self.assertEqual(confirmed[0]["type"], "wake_confirmed")
        self.assertEqual(confirmed[0]["matched_phrase"], "twinna")
        self.assertEqual(confirmed[0]["remaining_text"], "schau mal im web")
        self.assertEqual(wake_phrase_spotter.capture_durations_ms[:2], [100, 1000])

    def test_voice_session_local_stt_stage1_caps_scans_to_short_window(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

            def step(self, seconds: float) -> None:
                self.value += seconds

        with TemporaryDirectory() as temp_dir:
            clock = _FakeClock()
            wake_phrase_spotter = _ShortPrefixWakePhraseSpotter(
                min_duration_ms=900,
                max_duration_ms=1000,
            )
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1000,
                voice_orchestrator_history_ms=3200,
                voice_orchestrator_wake_stage1_mode="local_stt",
                voice_orchestrator_local_stt_url="http://127.0.0.1:18090",
                voice_orchestrator_wake_candidate_window_ms=2200,
                voice_orchestrator_wake_postroll_ms=100,
                voice_orchestrator_wake_tail_max_ms=100,
                voice_orchestrator_candidate_cooldown_s=0.9,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=wake_phrase_spotter,
                backend_tail_transcript_extractor=_FakeWakeTailExtractor(),
                monotonic_fn=clock,
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            final_events: list[dict[str, object]] = []
            for sequence in range(27):
                events = session.handle_audio_frame(
                    OrchestratorVoiceAudioFrame(sequence=sequence, pcm_bytes=b"\x01\x00" * 1600)
                )
                if events:
                    final_events = events
                clock.step(0.1)

        self.assertEqual(final_events[0]["type"], "wake_confirmed")
        self.assertEqual(final_events[0]["matched_phrase"], "twinna")
        self.assertEqual(final_events[0]["remaining_text"], "schau mal im web")
        self.assertTrue(wake_phrase_spotter.capture_durations_ms)
        self.assertLessEqual(max(wake_phrase_spotter.capture_durations_ms), 1000)

    def test_voice_session_local_stt_stage1_keeps_pending_wake_across_postroll_frames(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

            def step(self, seconds: float) -> None:
                self.value += seconds

        with TemporaryDirectory() as temp_dir:
            clock = _FakeClock()
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1000,
                voice_orchestrator_wake_stage1_mode="local_stt",
                voice_orchestrator_local_stt_url="http://127.0.0.1:18090",
                voice_orchestrator_wake_candidate_window_ms=100,
                voice_orchestrator_wake_postroll_ms=250,
                voice_orchestrator_wake_tail_max_ms=250,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=_FakeWakePhraseSpotter(),
                backend_tail_transcript_extractor=_FakeWakeTailExtractor(),
                monotonic_fn=clock,
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            events: list[dict[str, object]] = []
            for sequence in range(4):
                events = session.handle_audio_frame(
                    OrchestratorVoiceAudioFrame(sequence=sequence, pcm_bytes=b"\x01\x00" * 1600)
                )
                clock.step(0.1)

        self.assertEqual(events[0]["type"], "wake_confirmed")
        self.assertEqual(events[0]["remaining_text"], "schau mal im web")

    def test_voice_session_local_stt_stage1_delays_next_scan_until_slow_scan_finishes(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

            def step(self, seconds: float) -> None:
                self.value += seconds

        class _SlowRejectingWakePhraseSpotter:
            def __init__(self, clock: _FakeClock) -> None:
                self.clock = clock
                self.calls = 0

            def detect(self, capture):
                del capture
                self.calls += 1
                self.clock.step(0.5)
                from twinr.proactive.wakeword.matching import WakewordMatch

                return WakewordMatch(
                    detected=False,
                    transcript="",
                    backend="stt",
                )

        with TemporaryDirectory() as temp_dir:
            clock = _FakeClock()
            wake_phrase_spotter = _SlowRejectingWakePhraseSpotter(clock)
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1000,
                voice_orchestrator_wake_stage1_mode="local_stt",
                voice_orchestrator_local_stt_url="http://127.0.0.1:18090",
                voice_orchestrator_wake_candidate_window_ms=2200,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=wake_phrase_spotter,
                backend_tail_transcript_extractor=_FakeWakeTailExtractor(),
                monotonic_fn=clock,
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            first = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=b"\x01\x00" * 1600)
            )
            second = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=b"\x01\x00" * 1600)
            )

        self.assertEqual(first, [])
        self.assertEqual(second, [])
        self.assertEqual(wake_phrase_spotter.calls, 1)

    def test_voice_session_local_stt_stage1_uses_tail_extractor_without_pending_rescan(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

            def step(self, seconds: float) -> None:
                self.value += seconds

        with TemporaryDirectory() as temp_dir:
            clock = _FakeClock()
            wake_phrase_spotter = _GrowingWakePhraseSpotter()
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1000,
                voice_orchestrator_wake_stage1_mode="local_stt",
                voice_orchestrator_local_stt_url="http://127.0.0.1:18090",
                voice_orchestrator_wake_candidate_window_ms=100,
                voice_orchestrator_wake_postroll_ms=250,
                voice_orchestrator_wake_tail_max_ms=250,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=wake_phrase_spotter,
                backend_tail_transcript_extractor=_TranscriptOnlyWakeTailExtractor("schau mal im web"),
                monotonic_fn=clock,
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            events: list[dict[str, object]] = []
            for sequence in range(4):
                events = session.handle_audio_frame(
                    OrchestratorVoiceAudioFrame(sequence=sequence, pcm_bytes=b"\x01\x00" * 1600)
                )
                clock.step(0.1)

        self.assertEqual(events[0]["type"], "wake_confirmed")
        self.assertEqual(events[0]["remaining_text"], "schau mal im web")
        self.assertEqual(wake_phrase_spotter.capture_durations_ms, [100])

    def test_voice_session_local_stt_stage1_anchors_short_scan_at_latest_speech_burst_start(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

            def step(self, seconds: float) -> None:
                self.value += seconds

        with TemporaryDirectory() as temp_dir:
            clock = _FakeClock()
            wake_frame = (1000).to_bytes(2, "little", signed=True) * 1600
            tail_frame = (2000).to_bytes(2, "little", signed=True) * 1600
            silence_frame = b"\x00\x00" * 1600
            wake_phrase_spotter = _BurstAnchoredWakePhraseSpotter(
                wake_prefix=wake_frame[:16],
                min_duration_ms=900,
                max_duration_ms=1000,
            )
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=800,
                voice_orchestrator_history_ms=3200,
                voice_orchestrator_wake_stage1_mode="local_stt",
                voice_orchestrator_local_stt_url="http://127.0.0.1:18090",
                voice_orchestrator_wake_candidate_window_ms=2200,
                voice_orchestrator_wake_postroll_ms=100,
                voice_orchestrator_wake_tail_max_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=wake_phrase_spotter,
                backend_tail_transcript_extractor=_FakeWakeTailExtractor(),
                monotonic_fn=clock,
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            frames = [silence_frame] * 4 + [wake_frame] * 3 + [tail_frame] * 15
            for sequence, pcm_bytes in enumerate(frames):
                session.handle_audio_frame(
                    OrchestratorVoiceAudioFrame(sequence=sequence, pcm_bytes=pcm_bytes)
                )
            clock.step(1.0)
            wake_candidate = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=len(frames), pcm_bytes=tail_frame)
            )
            confirmed = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=len(frames) + 1, pcm_bytes=tail_frame)
            )

        self.assertEqual(wake_candidate, [])
        self.assertEqual(confirmed[0]["type"], "wake_confirmed")
        self.assertEqual(confirmed[0]["matched_phrase"], "twinna")
        self.assertEqual(confirmed[0]["remaining_text"], "schau mal im web")
        self.assertTrue(wake_phrase_spotter.capture_prefixes)
        self.assertEqual(wake_phrase_spotter.capture_prefixes[-1], wake_frame[:16])
        self.assertTrue(wake_phrase_spotter.capture_durations_ms)
        self.assertLessEqual(wake_phrase_spotter.capture_durations_ms[-1], 1000)

    def test_voice_session_local_stt_stage1_preserves_quiet_onset_before_active_burst(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

            def step(self, seconds: float) -> None:
                self.value += seconds

        with TemporaryDirectory() as temp_dir:
            clock = _FakeClock()
            quiet_onset_frame = (600).to_bytes(2, "little", signed=True) * 1600
            active_wake_frame = (1200).to_bytes(2, "little", signed=True) * 1600
            tail_frame = (2000).to_bytes(2, "little", signed=True) * 1600
            silence_frame = b"\x00\x00" * 1600
            wake_phrase_spotter = _BurstAnchoredWakePhraseSpotter(
                wake_prefix=quiet_onset_frame[:16],
                min_duration_ms=900,
                max_duration_ms=1000,
            )
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=800,
                voice_orchestrator_history_ms=3200,
                voice_orchestrator_wake_stage1_mode="local_stt",
                voice_orchestrator_local_stt_url="http://127.0.0.1:18090",
                voice_orchestrator_wake_candidate_window_ms=2200,
                voice_orchestrator_wake_postroll_ms=100,
                voice_orchestrator_wake_tail_max_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=wake_phrase_spotter,
                backend_tail_transcript_extractor=_FakeWakeTailExtractor(),
                monotonic_fn=clock,
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            frames = [silence_frame] * 4 + [quiet_onset_frame] + [active_wake_frame] * 2 + [tail_frame] * 15
            for sequence, pcm_bytes in enumerate(frames):
                session.handle_audio_frame(
                    OrchestratorVoiceAudioFrame(sequence=sequence, pcm_bytes=pcm_bytes)
                )
            clock.step(1.0)
            wake_candidate = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=len(frames), pcm_bytes=tail_frame)
            )
            confirmed = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=len(frames) + 1, pcm_bytes=tail_frame)
            )

        self.assertEqual(wake_candidate, [])
        self.assertEqual(confirmed[0]["type"], "wake_confirmed")
        self.assertEqual(confirmed[0]["matched_phrase"], "twinna")
        self.assertEqual(confirmed[0]["remaining_text"], "schau mal im web")
        self.assertTrue(wake_phrase_spotter.capture_prefixes)
        self.assertEqual(wake_phrase_spotter.capture_prefixes[-1], quiet_onset_frame[:16])
        self.assertTrue(wake_phrase_spotter.capture_durations_ms)
        self.assertLessEqual(wake_phrase_spotter.capture_durations_ms[-1], 1000)

    def test_voice_session_local_stt_mode_requires_service_url(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                voice_orchestrator_wake_stage1_mode="local_stt",
            )

            with self.assertRaises(ValueError):
                EdgeOrchestratorVoiceSession(config)

    def test_voice_session_local_stt_mode_does_not_require_openai_backend(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                voice_orchestrator_wake_stage1_mode="local_stt",
                voice_orchestrator_local_stt_url="http://127.0.0.1:18090",
            )

            session = EdgeOrchestratorVoiceSession(config)
            ready = session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )

        self.assertEqual(ready[0]["type"], "voice_ready")
        self.assertEqual(ready[0]["backend"], "local_stt")


class OrchestratorContractTests(unittest.TestCase):
    def test_turn_complete_payload_serializes_model_like_token_usage(self) -> None:
        @dataclass
        class TokenUsage:
            total_tokens: int = 42

        token_usage = TokenUsage()
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
