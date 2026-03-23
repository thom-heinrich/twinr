from dataclasses import dataclass
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
from threading import Event, Thread
import time
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
from twinr.orchestrator.remote_asr import RemoteAsrBackendAdapter
from twinr.orchestrator.server import EdgeOrchestratorServer, create_app as create_orchestrator_app
from twinr.orchestrator.session import EdgeOrchestratorSession, RemoteToolBridge
from twinr.orchestrator.voice_activation import VoiceActivationMatch
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
        return [{"type": "voice_ready", "session_id": "voice-1", "backend": "remote_asr"}]

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
                "backend": "remote_asr",
                "detector_label": "twinna",
                "score": 0.81,
            }
        ]


class _FakeWakePhraseSpotter:
    def detect(self, capture):
        del capture

        return VoiceActivationMatch(
            detected=True,
            transcript="Twinna schau mal im Web",
            matched_phrase="twinna",
            remaining_text="schau mal im web",
            backend="remote_asr",
            detector_label="twinna",
            score=0.92,
        )

    def match_transcript(self, transcript: str):
        return VoiceActivationMatch(
            detected=True,
            transcript=transcript,
            matched_phrase="twinna",
            remaining_text="wie geht es dir" if "wie geht es dir" in transcript.lower() else "schau mal im web",
            normalized_transcript=str(transcript or "").strip().lower(),
            backend="remote_asr",
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

        if int(capture.sample.duration_ms) < self.min_duration_ms:
            return VoiceActivationMatch(
                detected=False,
                transcript="",
                backend="remote_asr",
            )
        return VoiceActivationMatch(
            detected=True,
            transcript=f"Twinna {self.remaining_text}",
            matched_phrase="twinna",
            remaining_text=self.remaining_text,
            backend="remote_asr",
            detector_label="twinna",
            score=0.93,
        )


class _TranscriptOnlyWakeTailExtractor:
    def __init__(self, transcript: str) -> None:
        self.transcript = transcript

    def extract(self, capture):
        del capture
        return self.transcript


class _ExplodingWakeTailExtractor:
    def extract(self, capture):
        del capture
        raise AssertionError("same-stream remote_asr wake path must not use tail extractor")


class _RejectingWakePhraseSpotter:
    def detect(self, capture):
        del capture

        return VoiceActivationMatch(
            detected=False,
            transcript="Maurizio Puppe",
            normalized_transcript="maurizio puppe",
            backend="remote_asr",
        )

    def match_transcript(self, transcript: str):
        return VoiceActivationMatch(
            detected=False,
            transcript=transcript,
            normalized_transcript=str(transcript or "").strip().lower(),
            backend="remote_asr",
        )


class _TranscriptOnlyNonWakePhraseSpotter:
    def __init__(self, transcript: str) -> None:
        self.transcript = transcript

    def detect(self, capture):
        del capture

        return VoiceActivationMatch(
            detected=False,
            transcript=self.transcript,
            normalized_transcript=str(self.transcript or "").strip().lower(),
            backend="remote_asr",
        )


class _BusyWakePhraseSpotter:
    def __init__(self, message: str = 'Remote ASR service returned HTTP 429: {"detail":"stt busy"}') -> None:
        self.message = message

    def detect(self, capture):
        del capture
        raise RuntimeError(self.message)


class _LengthSensitiveWakePhraseSpotter:
    def __init__(self, *, min_duration_ms: int) -> None:
        self.min_duration_ms = int(min_duration_ms)
        self.capture_durations_ms: list[int] = []

    def detect(self, capture):
        self.capture_durations_ms.append(int(capture.sample.duration_ms))

        if int(capture.sample.duration_ms) < self.min_duration_ms:
            return VoiceActivationMatch(
                detected=False,
                transcript="",
                backend="remote_asr",
            )
        return VoiceActivationMatch(
            detected=True,
            transcript="Twinna",
            matched_phrase="twinna",
            remaining_text="",
            normalized_transcript="twinna",
            backend="remote_asr",
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

        duration_ms = int(capture.sample.duration_ms)
        if duration_ms < self.min_duration_ms or duration_ms > self.max_duration_ms:
            return VoiceActivationMatch(
                detected=False,
                transcript="schau mal im web nach dem wetter in berlin",
                normalized_transcript="schau mal im web nach dem wetter in berlin",
                backend="remote_asr",
            )
        return VoiceActivationMatch(
            detected=True,
            transcript="Twinner",
            matched_phrase="twinna",
            remaining_text="",
            normalized_transcript="twinner",
            backend="remote_asr",
            detector_label="twinna",
            score=0.92,
        )


class _GrowingWakePhraseSpotter:
    def __init__(self) -> None:
        self.capture_durations_ms: list[int] = []

    def detect(self, capture):
        self.capture_durations_ms.append(int(capture.sample.duration_ms))

        duration_ms = int(capture.sample.duration_ms)
        if duration_ms < 100:
            return VoiceActivationMatch(
                detected=False,
                transcript="",
                backend="remote_asr",
            )
        if duration_ms < 300:
            return VoiceActivationMatch(
                detected=True,
                transcript="Twinner",
                matched_phrase="twinna",
                remaining_text="",
                normalized_transcript="twinner",
                backend="remote_asr",
                detector_label="twinna",
                score=0.92,
            )
        return VoiceActivationMatch(
            detected=True,
            transcript="Twinner schau mal im web",
            matched_phrase="twinna",
            remaining_text="schau mal im web",
            normalized_transcript="twinner schau mal im web",
            backend="remote_asr",
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

        duration_ms = int(capture.sample.duration_ms)
        if (
            duration_ms < self.min_duration_ms
            or duration_ms > self.max_duration_ms
            or not bytes(capture.pcm_bytes).startswith(self.wake_prefix)
        ):
            return VoiceActivationMatch(
                detected=False,
                transcript="schau mal im web nach dem wetter in berlin",
                normalized_transcript="schau mal im web nach dem wetter in berlin",
                backend="remote_asr",
            )
        return VoiceActivationMatch(
            detected=True,
            transcript="Twinner",
            matched_phrase="twinna",
            remaining_text="",
            normalized_transcript="twinner",
            backend="remote_asr",
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
                        "TWINR_VOICE_ORCHESTRATOR_REMOTE_ASR_URL=http://127.0.0.1:18090",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Unsupported voice orchestrator wake stage1 mode: backend"):
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
                        '{"type":"voice_ready","session_id":"voice-1","backend":"remote_asr"}',
                        '{"type":"wake_confirmed","matched_phrase":"twinna","remaining_text":"schau mal im web","backend":"remote_asr","detector_label":"twinna","score":0.9}',
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

        self.assertTrue(any(getattr(event, "backend", "") == "remote_asr" for event in events))
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

    def test_voice_client_serializes_concurrent_runtime_state_and_audio_sends(self) -> None:
        class _BlockingSocket:
            def __init__(self):
                self.sent: list[str] = []
                self._first_send_entered = Event()
                self._release_first_send = Event()
                self._active_sends = 0
                self.max_active_sends = 0
                self._send_calls = 0

            def send(self, payload: str) -> None:
                self._send_calls += 1
                self._active_sends += 1
                self.max_active_sends = max(self.max_active_sends, self._active_sends)
                try:
                    if self._send_calls == 1:
                        self._first_send_entered.set()
                        self._release_first_send.wait(timeout=1.0)
                    time.sleep(0.01)
                    self.sent.append(payload)
                finally:
                    self._active_sends -= 1

            def recv(self, timeout=None) -> str:
                raise TimeoutError()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        socket = _BlockingSocket()
        client = OrchestratorVoiceWebSocketClient(
            "ws://127.0.0.1/ws",
            connector=lambda *args, **kwargs: socket,
            on_event=lambda event: None,
            require_tls=False,
        )

        client.open()
        audio_thread = Thread(
            target=client.send_audio_frame,
            args=(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=b"\x00\x00" * 160),
            ),
        )
        audio_thread.start()
        self.assertTrue(socket._first_send_entered.wait(timeout=1.0))

        state_thread = Thread(
            target=client.send_runtime_state,
            args=(
                OrchestratorVoiceRuntimeStateEvent(state="listening", follow_up_allowed=False),
            ),
        )
        state_thread.start()
        time.sleep(0.05)
        self.assertEqual(socket.max_active_sends, 1)

        socket._release_first_send.set()
        audio_thread.join(timeout=1.0)
        state_thread.join(timeout=1.0)
        client.close()

        self.assertEqual(len(socket.sent), 2)


class OrchestratorVoiceSessionTests(unittest.TestCase):
    def test_remote_asr_backend_adapter_posts_audio_and_returns_text(self) -> None:
        captured = {}

        def _fake_urlopen(request, timeout=0):
            captured["url"] = request.full_url
            captured["headers"] = dict(request.header_items())
            captured["body"] = request.data
            captured["timeout"] = timeout
            return _FakeUrlOpenResponse(
                b'{"text":"Hallo, Twinner!","language":"de","segments":[{"start":0.0,"end":1.0,"text":"Hallo, Twinner!"}],"duration_sec":1.0}'
            )

        adapter = RemoteAsrBackendAdapter(
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

    def test_remote_asr_backend_adapter_retries_busy_service_once(self) -> None:
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

        adapter = RemoteAsrBackendAdapter(
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
                voice_orchestrator_wake_stage1_mode="remote_asr",
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=100,
                voice_orchestrator_follow_up_window_ms=100,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "Twinna wie geht es dir"),
                wake_phrase_spotter=_MinDurationWakePhraseSpotter(
                    min_duration_ms=100,
                    remaining_text="wie geht es dir",
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
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(first, [])
        self.assertEqual(events[0]["type"], "wake_confirmed")
        self.assertEqual(events[0]["matched_phrase"], "twinna")
        self.assertEqual(events[0]["remaining_text"], "wie geht es dir")

    def test_voice_session_follow_up_window_commits_transcript_without_wake_phrase(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
                voice_orchestrator_wake_stage1_mode="remote_asr",
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=100,
                voice_orchestrator_follow_up_window_ms=100,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "wie geht es dir"),
                wake_phrase_spotter=_TranscriptOnlyNonWakePhraseSpotter("wie geht es dir"),
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
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(first, [])
        self.assertEqual(events[0]["type"], "transcript_committed")
        self.assertEqual(events[0]["source"], "follow_up")
        self.assertEqual(events[0]["transcript"], "wie geht es dir")

    def test_voice_session_listening_window_commits_same_stream_transcript(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
                voice_orchestrator_wake_stage1_mode="remote_asr",
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=100,
                voice_orchestrator_follow_up_window_ms=100,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "wie geht es dir"),
                wake_phrase_spotter=_TranscriptOnlyNonWakePhraseSpotter("wie geht es dir"),
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
                    state="listening",
                    detail="wakeword",
                    follow_up_allowed=False,
                )
            )
            first = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(2))
            )
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(first, [])
        self.assertEqual(events[0]["type"], "transcript_committed")
        self.assertEqual(events[0]["source"], "listening")
        self.assertEqual(events[0]["transcript"], "wie geht es dir")

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
                voice_orchestrator_wake_stage1_mode="remote_asr",
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=100,
                voice_orchestrator_wake_postroll_ms=100,
                voice_orchestrator_wake_tail_max_ms=100,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
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
                OrchestratorVoiceAudioFrame(sequence=3, pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(first, [])
        self.assertEqual(second, [])
        self.assertEqual(third, [])
        self.assertEqual(fourth[0]["type"], "wake_confirmed")
        self.assertEqual(fourth[0]["matched_phrase"], "twinna")
        self.assertEqual(fourth[0]["remaining_text"], "wie geht es dir")
        self.assertEqual(backend.calls, 0)

    def test_voice_session_remote_asr_context_bias_accepts_shorter_wake_utterance(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
                voice_orchestrator_wake_stage1_mode="remote_asr",
                voice_orchestrator_remote_asr_min_wake_duration_ms=300,
                voice_orchestrator_intent_min_wake_duration_relief_ms=100,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
            )
            wake_phrase_spotter = _MinDurationWakePhraseSpotter(min_duration_ms=200)
            biased_session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=wake_phrase_spotter,
            )
            strict_session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=_MinDurationWakePhraseSpotter(min_duration_ms=200),
            )

            for session in (biased_session, strict_session):
                session.handle_hello(
                    OrchestratorVoiceHelloRequest(
                        session_id="voice-1",
                        sample_rate=16000,
                        channels=1,
                        chunk_ms=100,
                    )
                )

            strict_session.handle_runtime_state(
                OrchestratorVoiceRuntimeStateEvent(
                    state="waiting",
                    detail="idle",
                    follow_up_allowed=False,
                )
            )
            biased_session.handle_runtime_state(
                OrchestratorVoiceRuntimeStateEvent(
                    state="waiting",
                    detail="idle",
                    follow_up_allowed=False,
                    attention_state="attending_to_device",
                    interaction_intent_state="showing_intent",
                    person_visible=True,
                    interaction_ready=True,
                    targeted_inference_blocked=False,
                    recommended_channel="speech",
                )
            )

            strict_session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(2))
            )
            strict_session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=_pcm_frame(2))
            )
            strict_events = strict_session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=2, pcm_bytes=_pcm_frame(0))
            )

            biased_session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(2))
            )
            biased_session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=_pcm_frame(2))
            )
            biased_events = biased_session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=2, pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(strict_events, [])
        self.assertEqual(biased_events[0]["type"], "wake_confirmed")
        self.assertEqual(biased_events[0]["matched_phrase"], "twinna")

    def test_voice_session_follow_up_context_refresh_uses_open_time_for_bonus_deadline(self) -> None:
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
                audio_speech_threshold=1,
                voice_orchestrator_follow_up_timeout_s=6.0,
                voice_orchestrator_intent_follow_up_timeout_bonus_s=2.0,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=_RejectingWakePhraseSpotter(),
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

            clock.step(5.0)
            session.handle_runtime_state(
                OrchestratorVoiceRuntimeStateEvent(
                    state="follow_up_open",
                    detail="wakeword",
                    follow_up_allowed=True,
                    attention_state="attending_to_device",
                    interaction_intent_state="showing_intent",
                    person_visible=True,
                    interaction_ready=True,
                    targeted_inference_blocked=False,
                    recommended_channel="speech",
                )
            )
            clock.step(2.0)
            before_deadline = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=b"")
            )
            clock.step(1.1)
            after_deadline = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=b"")
            )

        self.assertEqual(before_deadline, [])
        self.assertEqual(after_deadline[0]["type"], "follow_up_closed")
        self.assertEqual(after_deadline[0]["reason"], "timeout")

    def test_voice_session_uses_remote_asr_stage1_for_wake_candidates(self) -> None:
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
                audio_speech_threshold=1,
                voice_orchestrator_wake_stage1_mode="remote_asr",
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=100,
                voice_orchestrator_wake_candidate_window_ms=100,
                voice_orchestrator_wake_postroll_ms=100,
                voice_orchestrator_wake_tail_max_ms=100,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
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
            third = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=2, pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(ready[0]["backend"], "remote_asr")
        self.assertEqual(first, [])
        self.assertEqual(second, [])
        self.assertEqual(third[0]["type"], "wake_confirmed")
        self.assertEqual(third[0]["remaining_text"], "schau mal im web")

    def test_voice_session_remote_asr_stage1_still_requires_transcript_match(self) -> None:
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
                audio_speech_threshold=1,
                voice_orchestrator_wake_stage1_mode="remote_asr",
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=100,
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

    def test_voice_session_remote_asr_stage1_scans_quiet_nonzero_audio_without_active_chunks(self) -> None:
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
                audio_speech_threshold=10,
                voice_orchestrator_wake_stage1_mode="remote_asr",
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=100,
                voice_orchestrator_wake_candidate_window_ms=100,
                voice_orchestrator_wake_postroll_ms=100,
                voice_orchestrator_wake_tail_max_ms=100,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
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
            first = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(2))
            )
            clock.step(1.0)
            second = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=_pcm_frame(2))
            )
            third = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=2, pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(first, [])
        self.assertEqual(second, [])
        self.assertEqual(third[0]["type"], "wake_confirmed")
        self.assertEqual(third[0]["matched_phrase"], "twinna")
        self.assertEqual(third[0]["remaining_text"], "schau mal im web")

    def test_voice_session_remote_asr_stage1_capture_keeps_wake_and_tail_in_one_utterance(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=10,
                voice_orchestrator_wake_stage1_mode="remote_asr",
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_wake_candidate_window_ms=2200,
                voice_orchestrator_follow_up_window_ms=900,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=_FakeWakePhraseSpotter(),
                backend_tail_transcript_extractor=_FakeWakeTailExtractor(),
            )

            for frame in (
                _pcm_frame(20),
                _pcm_frame(20),
                _pcm_frame(0),
                _pcm_frame(0),
                _pcm_frame(0),
                _pcm_frame(0),
                _pcm_frame(20),
                _pcm_frame(20),
                _pcm_frame(20),
            ):
                session._remember_frame(frame)

            capture = session._recent_remote_asr_stage1_capture()

        self.assertEqual(capture.sample.duration_ms, 900)
        self.assertEqual(capture.sample.chunk_count, 9)
        self.assertEqual(capture.pcm_bytes[: len(_pcm_frame(20))], _pcm_frame(20))

    def test_voice_session_remote_asr_stage1_waits_for_minimum_wake_burst(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

            def step(self, seconds: float) -> None:
                self.value += seconds

        with TemporaryDirectory() as temp_dir:
            clock = _FakeClock()
            wake_phrase_spotter = _MinDurationWakePhraseSpotter(min_duration_ms=800)
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
                voice_orchestrator_wake_stage1_mode="remote_asr",
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=800,
                voice_orchestrator_wake_candidate_window_ms=2200,
                voice_orchestrator_candidate_cooldown_s=0.0,
                voice_orchestrator_wake_postroll_ms=100,
                voice_orchestrator_wake_tail_max_ms=100,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
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
            events: list[dict[str, object]] = []
            for sequence in range(7):
                events = session.handle_audio_frame(
                    OrchestratorVoiceAudioFrame(sequence=sequence, pcm_bytes=_pcm_frame(2))
                )
                self.assertEqual(events, [])
                clock.step(0.1)
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=7, pcm_bytes=_pcm_frame(2))
            )
            self.assertEqual(events, [])
            clock.step(0.1)
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=8, pcm_bytes=_pcm_frame(2))
            )
            self.assertEqual(events, [])
            clock.step(0.1)
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=9, pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(events[0]["type"], "wake_confirmed")
        self.assertEqual(events[0]["matched_phrase"], "twinna")
        self.assertEqual(events[0]["remaining_text"], "wie geht es dir")
        self.assertEqual(wake_phrase_spotter.capture_durations_ms, [1000])

    def test_voice_session_remote_asr_stage1_persists_buffering_debug_entry(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
                voice_orchestrator_wake_stage1_mode="remote_asr",
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=300,
                voice_orchestrator_wake_candidate_window_ms=2200,
                voice_orchestrator_candidate_cooldown_s=0.0,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=_FakeWakePhraseSpotter(),
                backend_tail_transcript_extractor=_FakeWakeTailExtractor(),
                monotonic_fn=_FakeClock(),
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(2))
            )

            transcript_log_path = (
                Path(temp_dir) / "artifacts" / "stores" / "ops" / "voice_gateway_transcripts.jsonl"
            )
            entries = [
                json.loads(line)
                for line in transcript_log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(events, [])
        self.assertEqual(entries[-1]["stage"], "activation_utterance")
        self.assertEqual(entries[-1]["outcome"], "buffering_short_utterance")
        self.assertEqual(entries[-1]["sample"]["duration_ms"], 100)
        self.assertEqual(entries[-1]["details"]["required_active_ms"], 300)

    def test_voice_session_remote_asr_stage1_persists_raw_match_debug_entry(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

        with TemporaryDirectory() as temp_dir:
            wake_phrase_spotter = _ShortPrefixWakePhraseSpotter(min_duration_ms=100, max_duration_ms=200)
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
                voice_orchestrator_wake_stage1_mode="remote_asr",
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=100,
                voice_orchestrator_wake_candidate_window_ms=100,
                voice_orchestrator_candidate_cooldown_s=0.0,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=wake_phrase_spotter,
                backend_tail_transcript_extractor=_TranscriptOnlyWakeTailExtractor(""),
                monotonic_fn=_FakeClock(),
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(2))
            )
            self.assertEqual(events, [])
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=_pcm_frame(0))
            )

            transcript_log_path = (
                Path(temp_dir) / "artifacts" / "stores" / "ops" / "voice_gateway_transcripts.jsonl"
            )
            entries = [
                json.loads(line)
                for line in transcript_log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(events[0]["type"], "wake_confirmed")
        self.assertEqual(entries[-1]["stage"], "activation_utterance")
        self.assertEqual(entries[-1]["outcome"], "matched")
        self.assertEqual(entries[-1]["transcript"], "Twinner")
        self.assertEqual(entries[-1]["matched_phrase"], "twinna")
        self.assertEqual(entries[-1]["remaining_text"], "")

    def test_voice_session_remote_asr_stage1_persists_raw_no_match_debug_entry(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
                voice_orchestrator_wake_stage1_mode="remote_asr",
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=100,
                voice_orchestrator_wake_candidate_window_ms=100,
                voice_orchestrator_candidate_cooldown_s=0.0,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=_RejectingWakePhraseSpotter(),
                backend_tail_transcript_extractor=_TranscriptOnlyWakeTailExtractor(""),
                monotonic_fn=_FakeClock(),
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(2))
            )
            self.assertEqual(events, [])
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=_pcm_frame(0))
            )

            transcript_log_path = (
                Path(temp_dir) / "artifacts" / "stores" / "ops" / "voice_gateway_transcripts.jsonl"
            )
            entries = [
                json.loads(line)
                for line in transcript_log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(events, [])
        self.assertEqual(entries[-1]["stage"], "activation_utterance")
        self.assertEqual(entries[-1]["outcome"], "no_match")
        self.assertEqual(entries[-1]["transcript"], "Maurizio Puppe")

    def test_voice_session_remote_asr_stage1_persists_backend_error_debug_entry(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
                voice_orchestrator_wake_stage1_mode="remote_asr",
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=100,
                voice_orchestrator_wake_candidate_window_ms=100,
                voice_orchestrator_candidate_cooldown_s=0.0,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=_BusyWakePhraseSpotter(),
                backend_tail_transcript_extractor=_TranscriptOnlyWakeTailExtractor(""),
                monotonic_fn=_FakeClock(),
            )

            session.handle_hello(
                OrchestratorVoiceHelloRequest(
                    session_id="voice-1",
                    sample_rate=16000,
                    channels=1,
                    chunk_ms=100,
                )
            )
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=0, pcm_bytes=_pcm_frame(2))
            )
            self.assertEqual(events, [])
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=1, pcm_bytes=_pcm_frame(0))
            )

            transcript_log_path = (
                Path(temp_dir) / "artifacts" / "stores" / "ops" / "voice_gateway_transcripts.jsonl"
            )
            entries = [
                json.loads(line)
                for line in transcript_log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(events, [])
        self.assertEqual(entries[-1]["stage"], "activation_utterance")
        self.assertEqual(entries[-1]["outcome"], "backend_error")
        self.assertEqual(entries[-1]["details"]["error_type"], "RuntimeError")
        self.assertIn("stt busy", entries[-1]["details"]["error_message"])

    def test_voice_session_remote_asr_stage1_confirms_wake_only_turn_after_endpoint_silence(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

            def step(self, seconds: float) -> None:
                self.value += seconds

        with TemporaryDirectory() as temp_dir:
            clock = _FakeClock()
            wake_phrase_spotter = _MinDurationWakePhraseSpotter(min_duration_ms=800, remaining_text="")
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1,
                voice_orchestrator_wake_stage1_mode="remote_asr",
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=800,
                voice_orchestrator_wake_candidate_window_ms=2200,
                voice_orchestrator_candidate_cooldown_s=0.0,
                voice_orchestrator_wake_postroll_ms=250,
                voice_orchestrator_wake_tail_endpoint_silence_ms=300,
                voice_orchestrator_wake_tail_max_ms=2200,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=wake_phrase_spotter,
                backend_tail_transcript_extractor=_TranscriptOnlyWakeTailExtractor(""),
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
            for sequence in range(8):
                events = session.handle_audio_frame(
                    OrchestratorVoiceAudioFrame(sequence=sequence, pcm_bytes=_pcm_frame(2))
                )
                self.assertEqual(events, [])
                clock.step(0.1)
            for silence_index in range(3):
                events = session.handle_audio_frame(
                    OrchestratorVoiceAudioFrame(sequence=8 + silence_index, pcm_bytes=_pcm_frame(0))
                )
                clock.step(0.1)

        self.assertEqual(events[0]["type"], "wake_confirmed")
        self.assertEqual(events[0]["matched_phrase"], "twinna")
        self.assertEqual(events[0]["remaining_text"], "")
        self.assertEqual(wake_phrase_spotter.capture_durations_ms, [1100])

    def test_voice_session_remote_asr_stage1_expands_with_available_history_up_to_short_scan_window(self) -> None:
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
                audio_speech_threshold=1,
                voice_orchestrator_history_ms=300,
                voice_orchestrator_wake_stage1_mode="remote_asr",
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=100,
                voice_orchestrator_wake_candidate_window_ms=100,
                voice_orchestrator_wake_postroll_ms=100,
                voice_orchestrator_wake_tail_max_ms=100,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
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
            fourth = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=3, pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(first, [])
        self.assertEqual(second, [])
        self.assertEqual(third, [])
        self.assertEqual(fourth[0]["type"], "wake_confirmed")
        self.assertEqual(wake_phrase_spotter.capture_durations_ms, [400])

    def test_voice_session_remote_asr_stage1_uses_short_prefix_before_full_sentence_capture(self) -> None:
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
                audio_speech_threshold=1,
                voice_orchestrator_history_ms=2400,
                voice_orchestrator_wake_stage1_mode="remote_asr",
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=100,
                voice_orchestrator_wake_candidate_window_ms=2200,
                voice_orchestrator_wake_postroll_ms=100,
                voice_orchestrator_wake_tail_max_ms=100,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
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
            wake_candidate = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=10, pcm_bytes=b"\x01\x00" * 1600)
            )
            confirmed = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=11, pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(wake_candidate, [])
        self.assertEqual(confirmed[0]["type"], "wake_confirmed")
        self.assertEqual(confirmed[0]["matched_phrase"], "twinna")
        self.assertEqual(confirmed[0]["remaining_text"], "schau mal im web")
        self.assertEqual(wake_phrase_spotter.capture_durations_ms, [1200])

    def test_voice_session_remote_asr_stage1_caps_scans_to_short_window(self) -> None:
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
                audio_speech_threshold=1,
                voice_orchestrator_history_ms=3200,
                voice_orchestrator_wake_stage1_mode="remote_asr",
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=100,
                voice_orchestrator_wake_candidate_window_ms=2200,
                voice_orchestrator_wake_postroll_ms=100,
                voice_orchestrator_wake_tail_max_ms=100,
                voice_orchestrator_candidate_cooldown_s=0.9,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
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
            final_events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=27, pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(final_events[0]["type"], "wake_confirmed")
        self.assertEqual(final_events[0]["matched_phrase"], "twinna")
        self.assertEqual(final_events[0]["remaining_text"], "schau mal im web")
        self.assertTrue(wake_phrase_spotter.capture_durations_ms)
        self.assertEqual(wake_phrase_spotter.capture_durations_ms, [2800])

    def test_voice_session_remote_asr_stage1_keeps_pending_wake_across_postroll_frames(self) -> None:
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
                audio_speech_threshold=1,
                voice_orchestrator_wake_stage1_mode="remote_asr",
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=100,
                voice_orchestrator_wake_candidate_window_ms=100,
                voice_orchestrator_wake_postroll_ms=250,
                voice_orchestrator_wake_tail_max_ms=250,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
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
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=4, pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(events[0]["type"], "wake_confirmed")
        self.assertEqual(events[0]["remaining_text"], "schau mal im web")

    def test_voice_session_remote_asr_stage1_delays_next_scan_until_slow_scan_finishes(self) -> None:
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
                return VoiceActivationMatch(
                    detected=False,
                    transcript="",
                    backend="remote_asr",
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
                audio_speech_threshold=1,
                voice_orchestrator_wake_stage1_mode="remote_asr",
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=100,
                voice_orchestrator_wake_candidate_window_ms=2200,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
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
            third = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=2, pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(first, [])
        self.assertEqual(second, [])
        self.assertEqual(third, [])
        self.assertEqual(wake_phrase_spotter.calls, 1)

    def test_voice_session_remote_asr_stage1_uses_tail_extractor_without_pending_rescan(self) -> None:
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
                audio_speech_threshold=1,
                voice_orchestrator_wake_stage1_mode="remote_asr",
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=100,
                voice_orchestrator_wake_candidate_window_ms=100,
                voice_orchestrator_wake_postroll_ms=250,
                voice_orchestrator_wake_tail_max_ms=250,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
            )
            session = EdgeOrchestratorVoiceSession(
                config,
                backend=SimpleNamespace(transcribe=lambda *args, **kwargs: "ignored"),
                wake_phrase_spotter=wake_phrase_spotter,
                backend_tail_transcript_extractor=_ExplodingWakeTailExtractor(),
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
            events = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=4, pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(events[0]["type"], "wake_confirmed")
        self.assertEqual(events[0]["remaining_text"], "schau mal im web")
        self.assertEqual(wake_phrase_spotter.capture_durations_ms, [500])

    def test_voice_session_remote_asr_stage1_anchors_short_scan_at_latest_speech_burst_start(self) -> None:
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
                max_duration_ms=2500,
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
                voice_orchestrator_wake_stage1_mode="remote_asr",
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=100,
                voice_orchestrator_wake_candidate_window_ms=2200,
                voice_orchestrator_wake_postroll_ms=100,
                voice_orchestrator_wake_tail_max_ms=100,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
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
            wake_candidate = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=len(frames), pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(wake_candidate[0]["type"], "wake_confirmed")
        self.assertEqual(wake_candidate[0]["matched_phrase"], "twinna")
        self.assertEqual(wake_candidate[0]["remaining_text"], "")
        self.assertTrue(wake_phrase_spotter.capture_prefixes)
        self.assertEqual(wake_phrase_spotter.capture_prefixes[-1], wake_frame[:16])
        self.assertEqual(wake_phrase_spotter.capture_durations_ms, [1900])

    def test_voice_session_remote_asr_stage1_preserves_quiet_onset_before_active_burst(self) -> None:
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
                max_duration_ms=2500,
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
                voice_orchestrator_wake_stage1_mode="remote_asr",
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=100,
                voice_orchestrator_wake_candidate_window_ms=2200,
                voice_orchestrator_wake_postroll_ms=100,
                voice_orchestrator_wake_tail_max_ms=100,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
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
            wake_candidate = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=len(frames), pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(wake_candidate[0]["type"], "wake_confirmed")
        self.assertEqual(wake_candidate[0]["matched_phrase"], "twinna")
        self.assertEqual(wake_candidate[0]["remaining_text"], "")
        self.assertTrue(wake_phrase_spotter.capture_prefixes)
        self.assertEqual(wake_phrase_spotter.capture_prefixes[-1], quiet_onset_frame[:16])
        self.assertEqual(wake_phrase_spotter.capture_durations_ms, [1900])

    def test_voice_session_remote_asr_stage1_preserves_nonzero_onset_below_half_threshold(self) -> None:
        class _FakeClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                return self.value

            def step(self, seconds: float) -> None:
                self.value += seconds

        with TemporaryDirectory() as temp_dir:
            clock = _FakeClock()
            quiet_onset_frame = (624).to_bytes(2, "little", signed=True) * 1600
            active_wake_frame = (2783).to_bytes(2, "little", signed=True) * 1600
            tail_frame = (3853).to_bytes(2, "little", signed=True) * 1600
            silence_frame = b"\x00\x00" * 1600
            wake_phrase_spotter = _BurstAnchoredWakePhraseSpotter(
                wake_prefix=quiet_onset_frame[:16],
                min_duration_ms=900,
                max_duration_ms=2500,
            )
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                audio_sample_rate=16000,
                audio_channels=1,
                audio_chunk_ms=100,
                audio_speech_threshold=1500,
                voice_orchestrator_history_ms=3200,
                voice_orchestrator_wake_stage1_mode="remote_asr",
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=100,
                voice_orchestrator_wake_candidate_window_ms=2200,
                voice_orchestrator_wake_postroll_ms=100,
                voice_orchestrator_wake_tail_max_ms=100,
                voice_orchestrator_wake_tail_endpoint_silence_ms=100,
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
            wake_candidate = session.handle_audio_frame(
                OrchestratorVoiceAudioFrame(sequence=len(frames), pcm_bytes=_pcm_frame(0))
            )

        self.assertEqual(wake_candidate[0]["type"], "wake_confirmed")
        self.assertEqual(wake_candidate[0]["matched_phrase"], "twinna")
        self.assertEqual(wake_candidate[0]["remaining_text"], "")
        self.assertTrue(wake_phrase_spotter.capture_prefixes)
        self.assertEqual(wake_phrase_spotter.capture_prefixes[-1], quiet_onset_frame[:16])
        self.assertEqual(wake_phrase_spotter.capture_durations_ms, [1900])

    def test_voice_session_remote_asr_mode_requires_service_url(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=temp_dir,
                personality_dir="personality",
                voice_orchestrator_wake_stage1_mode="remote_asr",
            )

            with self.assertRaises(ValueError):
                EdgeOrchestratorVoiceSession(config)

    def test_voice_session_remote_asr_mode_does_not_require_openai_backend(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                voice_orchestrator_wake_stage1_mode="remote_asr",
                voice_orchestrator_remote_asr_url="http://127.0.0.1:18090",
                voice_orchestrator_remote_asr_min_wake_duration_ms=100,
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
        self.assertEqual(ready[0]["backend"], "remote_asr")


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
