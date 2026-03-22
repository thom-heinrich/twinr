"""Run the websocket client for Twinr's streaming voice orchestrator path.

The workflow-local edge helper owns microphone capture and runtime-state
mapping. This module only owns the websocket transport: connect, send bounded
audio/runtime events, receive server decisions on a background thread, and
surface sanitized failures back to the caller.
"""

from __future__ import annotations

from collections.abc import Callable
from ipaddress import ip_address
import json
from threading import Event, Lock, Thread, current_thread
from typing import Any
from urllib.parse import urlsplit

from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidURI
from websockets.sync.client import connect as websocket_connect

from .voice_contracts import (
    OrchestratorVoiceAudioFrame,
    OrchestratorVoiceHelloRequest,
    OrchestratorVoiceRuntimeStateEvent,
    VoiceServerEvent,
    decode_voice_server_event,
)


def _validate_url(url: str, *, require_tls: bool) -> str:
    """Reject invalid or insecure websocket URLs before connecting."""

    normalized = str(url or "").strip()
    if not normalized:
        raise ValueError("voice orchestrator websocket url must not be empty")
    parts = urlsplit(normalized)
    if parts.scheme not in {"ws", "wss"}:
        raise ValueError("voice orchestrator websocket url must use ws:// or wss://")
    if parts.scheme != "wss":
        host = (parts.hostname or "").strip()
        if require_tls and host not in {"127.0.0.1", "localhost", "::1"}:
            try:
                if not ip_address(host).is_loopback:
                    raise ValueError("voice orchestrator websocket must use wss:// outside loopback")
            except ValueError:
                raise ValueError("voice orchestrator websocket must use wss:// outside loopback") from None
    return normalized


class OrchestratorVoiceWebSocketClient:
    """Keep one bounded websocket connection to the remote voice orchestrator."""

    _DEFAULT_OPEN_TIMEOUT_SECONDS = 10.0
    _DEFAULT_RECV_TIMEOUT_SECONDS = 2.0
    _DEFAULT_CLOSE_TIMEOUT_SECONDS = 5.0
    _DEFAULT_PING_INTERVAL_SECONDS = 20.0
    _DEFAULT_PING_TIMEOUT_SECONDS = 20.0
    _DEFAULT_MAX_MESSAGE_BYTES = 1_048_576
    _DEFAULT_MAX_QUEUE = 16

    def __init__(
        self,
        url: str,
        *,
        shared_secret: str | None = None,
        connector: Callable[..., Any] | None = None,
        on_event: Callable[[VoiceServerEvent], None] | None = None,
        open_timeout_seconds: float = _DEFAULT_OPEN_TIMEOUT_SECONDS,
        recv_timeout_seconds: float = _DEFAULT_RECV_TIMEOUT_SECONDS,
        close_timeout_seconds: float = _DEFAULT_CLOSE_TIMEOUT_SECONDS,
        ping_interval_seconds: float | None = _DEFAULT_PING_INTERVAL_SECONDS,
        ping_timeout_seconds: float | None = _DEFAULT_PING_TIMEOUT_SECONDS,
        max_message_bytes: int = _DEFAULT_MAX_MESSAGE_BYTES,
        max_queue: int = _DEFAULT_MAX_QUEUE,
        require_tls: bool = True,
    ) -> None:
        self.url = _validate_url(url, require_tls=require_tls)
        self.shared_secret = str(shared_secret or "").strip() or None
        self._connector = connector or websocket_connect
        self._on_event = on_event
        self.open_timeout_seconds = float(open_timeout_seconds)
        self.recv_timeout_seconds = float(recv_timeout_seconds)
        self.close_timeout_seconds = float(close_timeout_seconds)
        self.ping_interval_seconds = ping_interval_seconds
        self.ping_timeout_seconds = ping_timeout_seconds
        self.max_message_bytes = int(max_message_bytes)
        self.max_queue = int(max_queue)
        self._socket = None
        self._socket_lock = Lock()
        self._receiver_stop = Event()
        self._receiver_thread: Thread | None = None
        self._opened = False

    def open(self) -> "OrchestratorVoiceWebSocketClient":
        """Establish the websocket and start the background receiver."""

        with self._socket_lock:
            if self._opened and self._socket is not None:
                return self
            headers = None
            if self.shared_secret is not None:
                headers = {"x-twinr-secret": self.shared_secret}
            connector_kwargs = {
                "additional_headers": headers,
                "open_timeout": self.open_timeout_seconds,
                "close_timeout": self.close_timeout_seconds,
                "ping_interval": self.ping_interval_seconds,
                "ping_timeout": self.ping_timeout_seconds,
                "max_size": self.max_message_bytes,
                "max_queue": self.max_queue,
            }
            try:
                self._socket = self._connector(self.url, **connector_kwargs)
                self._socket.__enter__()
            except (InvalidURI, InvalidHandshake, OSError) as exc:
                raise ConnectionError("Failed to connect to voice orchestrator websocket") from exc
            self._receiver_stop.clear()
            self._receiver_thread = Thread(
                target=self._receiver_loop,
                daemon=True,
                name="twinr-voice-orchestrator-recv",
            )
            self._receiver_thread.start()
            self._opened = True
        return self

    def close(self) -> None:
        """Stop background receiving and close the websocket."""

        with self._socket_lock:
            socket = self._socket
            thread = self._receiver_thread
            self._receiver_stop.set()
            self._socket = None
            self._receiver_thread = None
            self._opened = False
        if thread is not None and thread is not current_thread():
            thread.join(timeout=self.close_timeout_seconds)
        if socket is not None:
            try:
                socket.__exit__(None, None, None)
            except Exception:
                pass

    def send_hello(self, request: OrchestratorVoiceHelloRequest) -> None:
        """Send the opening session metadata to the server."""

        self._send_payload(request.to_payload(), context="voice hello")

    def send_audio_frame(self, frame: OrchestratorVoiceAudioFrame) -> None:
        """Send one PCM frame to the server."""

        self._send_payload(frame.to_payload(), context="voice audio frame")

    def send_runtime_state(self, event: OrchestratorVoiceRuntimeStateEvent) -> None:
        """Send one runtime-state update to the server."""

        self._send_payload(event.to_payload(), context="voice runtime state")

    def _send_payload(self, payload: dict[str, Any], *, context: str) -> None:
        socket = self._require_socket()
        try:
            socket.send(json.dumps(payload, ensure_ascii=False))
        except ConnectionClosed as exc:
            raise ConnectionError(f"Voice orchestrator websocket closed while sending {context}") from exc

    def _require_socket(self):
        with self._socket_lock:
            if self._socket is None:
                raise RuntimeError("Voice orchestrator websocket is not open")
            return self._socket

    def _receiver_loop(self) -> None:
        socket = self._require_socket()
        while not self._receiver_stop.is_set():
            try:
                raw_message = socket.recv(timeout=self.recv_timeout_seconds)
            except TimeoutError:
                continue
            except ConnectionClosed:
                self._emit_error("Voice orchestrator websocket closed unexpectedly.")
                return
            except Exception as exc:
                self._emit_error(f"Voice orchestrator receive failed: {type(exc).__name__}")
                return
            if isinstance(raw_message, bytes):
                try:
                    raw_text = raw_message.decode("utf-8")
                except UnicodeDecodeError:
                    self._emit_error("Voice orchestrator sent non-UTF-8 payloads.")
                    return
            else:
                raw_text = str(raw_message)
            try:
                payload = json.loads(raw_text)
            except json.JSONDecodeError:
                self._emit_error("Voice orchestrator sent malformed JSON.")
                return
            try:
                event = decode_voice_server_event(payload)
            except Exception as exc:
                self._emit_error(f"Voice orchestrator sent an unsupported event: {type(exc).__name__}")
                return
            if self._on_event is not None:
                self._on_event(event)

    def _emit_error(self, text: str) -> None:
        from .voice_contracts import OrchestratorVoiceErrorEvent

        if self._on_event is not None:
            self._on_event(OrchestratorVoiceErrorEvent(error=text))


__all__ = ["OrchestratorVoiceWebSocketClient"]
