"""Run the websocket client for Twinr's streaming voice orchestrator path.

The workflow-local edge helper owns microphone capture and runtime-state
mapping. This module owns the websocket transport: connect, send bounded
audio/runtime events, receive server decisions on a background thread, and
surface sanitized transport failures back to the caller.

# CHANGELOG: 2026-03-29
# BUG-1: Fixed false error reporting on normal server close (1000/1001) and
#        now differentiate graceful vs abnormal disconnects.
# BUG-2: Fixed stale half-open state after remote disconnect by clearing the
#        active socket when the receiver loop exits.
# BUG-3: Fixed silent receiver death when the caller's on_event callback raises;
#        callback crashes are now logged and the connection is closed cleanly.
# BUG-4: Fixed shutdown ordering so close() unblocks recv() before joining,
#        avoiding reconnect races and leaked receiver threads.
# BUG-5: Normalized open() transport failures, including handshake timeouts, to
#        ConnectionError for a stable caller-facing API.
# SEC-1: # BREAKING: disabled automatic proxy use by default. websockets 15+
#        auto-discovers HTTP/SOCKS proxies from OS / environment; that is a bad
#        default for an embedded voice appliance unless explicitly opted in.
# SEC-2: # BREAKING: removed the default User-Agent header to reduce passive
#        version fingerprinting of Python / websockets on production devices.
# IMP-1: Added a bytes-first JSON hot path: recv(decode=False) plus send(...,
#        text=True) for already-encoded UTF-8 JSON, with optional msgspec
#        acceleration and stdlib-json fallback.
# IMP-2: Added 2026-tuned permessage-deflate settings for small repetitive JSON
#        traffic on resource-constrained devices, while keeping wire
#        compatibility with standard WebSocket servers.
# IMP-3: Added optional Bearer auth, custom TLS context / SNI override,
#        structured headers, logger integration, and support for websockets 16
#        tuple-based max_size / max_queue limits.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from ipaddress import ip_address
import inspect
import json
import logging
import ssl as ssl_module
from threading import Event, Lock, Thread, current_thread
from typing import Any
from urllib.parse import urlsplit

from websockets.exceptions import (
    ConnectionClosed,
    ConnectionClosedOK,
    InvalidHandshake,
    InvalidURI,
)
from websockets.sync.client import connect as websocket_connect

try:
    from websockets.extensions import permessage_deflate
except Exception:  # pragma: no cover - optional import / older websockets build
    permessage_deflate = None

try:
    import msgspec
except Exception:  # pragma: no cover - optional dependency
    msgspec = None

from .voice_contracts import (
    OrchestratorVoiceAudioFrame,
    OrchestratorVoiceHelloRequest,
    OrchestratorVoiceIdentityProfilesEvent,
    OrchestratorVoiceRuntimeStateEvent,
    VoiceServerEvent,
    decode_voice_server_event,
)


_DEFAULT_LOGGER = logging.getLogger("twinr.voice.orchestrator.ws")


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


def _validate_timeout(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    normalized = float(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be > 0 or None")
    return normalized


def _validate_int_or_pair(
    name: str,
    value: int | tuple[int | None, int | None] | None,
) -> int | tuple[int | None, int | None] | None:
    if value is None:
        return None
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"{name} tuple must contain exactly two entries")
        high, low = value
        for index, item in enumerate((high, low), start=1):
            if item is not None and int(item) <= 0:
                raise ValueError(f"{name}[{index}] must be > 0 or None")
        return (None if high is None else int(high), None if low is None else int(low))
    if int(value) <= 0:
        raise ValueError(f"{name} must be > 0 or None")
    return int(value)


def _filter_connector_kwargs(connector: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep compatibility with simple test doubles that don't accept new kwargs."""

    try:
        signature = inspect.signature(connector)
    except (TypeError, ValueError):
        return kwargs

    parameters = signature.parameters.values()
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return kwargs

    accepted = {
        parameter.name
        for parameter in parameters
        if parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return {key: value for key, value in kwargs.items() if key in accepted}


def _close_connection(owner: Any | None, socket: Any | None, *, code: int, reason: str) -> None:
    """Close a websocket / connector resource without leaking transport state."""

    if socket is not None:
        close = getattr(socket, "close", None)
        if callable(close):
            try:
                close(code=code, reason=reason)
            except TypeError:
                close()
            except Exception:
                pass

    if owner is not None and owner is not socket:
        exit_method = getattr(owner, "__exit__", None)
        if callable(exit_method):
            try:
                exit_method(None, None, None)
            except Exception:
                pass


def _close_code_and_reason(exc: ConnectionClosed) -> tuple[int | None, str | None]:
    close_frame = getattr(exc, "rcvd", None) or getattr(exc, "sent", None)
    if close_frame is None:
        return None, None
    return getattr(close_frame, "code", None), getattr(close_frame, "reason", None)


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
        auth_token: str | None = None,
        connector: Callable[..., Any] | None = None,
        on_event: Callable[[VoiceServerEvent], None] | None = None,
        logger: logging.Logger | logging.LoggerAdapter[Any] | None = None,
        ssl_context: ssl_module.SSLContext | None = None,
        server_hostname: str | None = None,
        # BREAKING: default is None in order to avoid fingerprinting Python/websockets versions.
        user_agent_header: str | None = None,
        # BREAKING: websockets 15+ auto-uses OS/environment proxies by default; disable unless explicitly requested.
        proxy: str | bool | None = None,
        compression: str | None = "tuned",
        handshake_headers: Mapping[str, str] | None = None,
        shared_secret_header_name: str = "x-twinr-secret",
        open_timeout_seconds: float | None = _DEFAULT_OPEN_TIMEOUT_SECONDS,
        recv_timeout_seconds: float | None = _DEFAULT_RECV_TIMEOUT_SECONDS,
        close_timeout_seconds: float | None = _DEFAULT_CLOSE_TIMEOUT_SECONDS,
        ping_interval_seconds: float | None = _DEFAULT_PING_INTERVAL_SECONDS,
        ping_timeout_seconds: float | None = _DEFAULT_PING_TIMEOUT_SECONDS,
        max_message_bytes: int | tuple[int | None, int | None] | None = _DEFAULT_MAX_MESSAGE_BYTES,
        max_queue: int | tuple[int | None, int | None] | None = _DEFAULT_MAX_QUEUE,
        require_tls: bool = True,
    ) -> None:
        self.url = _validate_url(url, require_tls=require_tls)
        self.shared_secret = str(shared_secret or "").strip() or None
        self.auth_token = str(auth_token or "").strip() or None
        self._connector = connector or websocket_connect
        self._on_event = on_event
        self._logger = logger or _DEFAULT_LOGGER

        self.ssl_context = ssl_context
        self.server_hostname = str(server_hostname or "").strip() or None
        self.user_agent_header = user_agent_header
        self.proxy = proxy
        self.compression = compression
        self.handshake_headers = {
            str(key): str(value)
            for key, value in (handshake_headers or {}).items()
            if str(key).strip()
        }
        self.shared_secret_header_name = str(shared_secret_header_name or "").strip() or "x-twinr-secret"

        self.open_timeout_seconds = _validate_timeout("open_timeout_seconds", open_timeout_seconds)
        self.recv_timeout_seconds = _validate_timeout("recv_timeout_seconds", recv_timeout_seconds)
        self.close_timeout_seconds = _validate_timeout("close_timeout_seconds", close_timeout_seconds)
        self.ping_interval_seconds = _validate_timeout("ping_interval_seconds", ping_interval_seconds)
        self.ping_timeout_seconds = _validate_timeout("ping_timeout_seconds", ping_timeout_seconds)
        self.max_message_bytes = _validate_int_or_pair("max_message_bytes", max_message_bytes)
        self.max_queue = _validate_int_or_pair("max_queue", max_queue)

        self._socket: Any | None = None
        self._socket_owner: Any | None = None
        self._socket_lock = Lock()
        self._send_lock = Lock()
        self._receiver_stop = Event()
        self._receiver_started = Event()
        self._receiver_thread: Thread | None = None
        self._opened = False

    @property
    def is_open(self) -> bool:
        with self._socket_lock:
            return self._socket is not None and self._opened

    @property
    def latency_seconds(self) -> float | None:
        socket = self._require_socket(raise_when_closed=False)
        if socket is None:
            return None
        latency = getattr(socket, "latency", None)
        return None if latency is None else float(latency)

    def open(self) -> "OrchestratorVoiceWebSocketClient":
        """Establish the websocket and start the background receiver."""

        with self._socket_lock:
            if self._opened and self._socket is not None:
                return self

            headers = dict(self.handshake_headers)
            if self.auth_token is not None:
                headers.setdefault("Authorization", f"Bearer {self.auth_token}")
            if self.shared_secret is not None:
                headers.setdefault(self.shared_secret_header_name, self.shared_secret)

            connect_kwargs: dict[str, Any] = {
                "additional_headers": headers or None,
                "open_timeout": self.open_timeout_seconds,
                "close_timeout": self.close_timeout_seconds,
                "ping_interval": self.ping_interval_seconds,
                "ping_timeout": self.ping_timeout_seconds,
                "max_size": self.max_message_bytes,
                "max_queue": self.max_queue,
                "proxy": self.proxy,
                "user_agent_header": self.user_agent_header,
                "logger": self._logger,
            }

            if self.server_hostname is not None:
                connect_kwargs["server_hostname"] = self.server_hostname

            if urlsplit(self.url).scheme == "wss" and self.ssl_context is not None:
                connect_kwargs["ssl"] = self.ssl_context

            if self.compression == "tuned" and permessage_deflate is not None:
                connect_kwargs["compression"] = None
                connect_kwargs["extensions"] = [
                    permessage_deflate.ClientPerMessageDeflateFactory(
                        server_max_window_bits=11,
                        client_max_window_bits=11,
                        compress_settings={"memLevel": 4},
                    )
                ]
            else:
                connect_kwargs["compression"] = self.compression

            try:
                connector_result = self._connector(self.url, **_filter_connector_kwargs(self._connector, connect_kwargs))
                entered_socket = connector_result.__enter__() if hasattr(connector_result, "__enter__") else connector_result
                self._socket_owner = connector_result
                self._socket = entered_socket if entered_socket is not None else connector_result
            except (InvalidURI, InvalidHandshake, OSError, TimeoutError) as exc:
                raise ConnectionError("Failed to connect to voice orchestrator websocket") from exc

            self._receiver_stop.clear()
            self._receiver_started.clear()
            socket = self._socket
            self._receiver_thread = Thread(
                target=self._receiver_loop,
                args=(socket,),
                daemon=True,
                name="twinr-voice-orchestrator-recv",
            )
            self._opened = True
            self._receiver_thread.start()

        if not self._receiver_started.wait(timeout=self.open_timeout_seconds):
            self.close(code=1011, reason="receiver thread failed to start")
            raise ConnectionError("Voice orchestrator receiver thread did not start")

        return self

    def close(self, *, code: int = 1000, reason: str = "") -> None:
        """Stop background receiving and close the websocket."""

        with self._socket_lock:
            socket = self._socket
            owner = self._socket_owner
            thread = self._receiver_thread

            self._receiver_stop.set()
            self._socket = None
            self._socket_owner = None
            self._receiver_thread = None
            self._opened = False
            self._receiver_started.clear()

        _close_connection(owner, socket, code=code, reason=reason)

        if thread is not None and thread is not current_thread():
            thread.join(timeout=self.close_timeout_seconds)

    def send_hello(self, request: OrchestratorVoiceHelloRequest) -> None:
        """Send the opening session metadata to the server."""

        self._send_payload(request.to_payload(), context="voice hello")

    def send_audio_frame(self, frame: OrchestratorVoiceAudioFrame) -> None:
        """Send one PCM frame to the server."""

        self._send_payload(frame.to_payload(), context="voice audio frame")

    def send_runtime_state(self, event: OrchestratorVoiceRuntimeStateEvent) -> None:
        """Send one runtime-state update to the server."""

        self._send_payload(event.to_payload(), context="voice runtime state")

    def send_identity_profiles(self, event: OrchestratorVoiceIdentityProfilesEvent) -> None:
        """Send the current read-only household voice profile snapshot."""

        self._send_payload(event.to_payload(), context="voice identity profiles")

    def _send_payload(self, payload: dict[str, Any], *, context: str) -> None:
        socket = self._require_socket()

        try:
            message = self._encode_payload(payload)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Failed to serialize {context}") from exc

        try:
            with self._send_lock:
                self._socket_send(socket, message)
        except ConnectionClosed as exc:
            self._mark_socket_closed(socket)
            raise ConnectionError(f"Voice orchestrator websocket closed while sending {context}") from exc

    def _encode_payload(self, payload: dict[str, Any]) -> bytes:
        if msgspec is not None:
            return msgspec.json.encode(payload)
        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

    def _decode_payload(self, raw_message: bytes) -> Any:
        if msgspec is not None:
            return msgspec.json.decode(raw_message)
        return json.loads(raw_message)

    def _require_socket(self, *, raise_when_closed: bool = True):
        with self._socket_lock:
            socket = self._socket
        if socket is None and raise_when_closed:
            raise RuntimeError("Voice orchestrator websocket is not open")
        return socket

    def _mark_socket_closed(self, socket: Any) -> None:
        with self._socket_lock:
            if self._socket is socket:
                self._socket = None
                self._socket_owner = None
                self._opened = False
            if self._receiver_thread is current_thread():
                self._receiver_thread = None
                self._receiver_started.clear()

    def _socket_send(self, socket: Any, message: bytes) -> None:
        """Send one encoded JSON payload across websocket implementations/test doubles."""

        try:
            socket.send(message, text=True)
            return
        except TypeError:
            pass
        try:
            socket.send(message)
            return
        except TypeError:
            socket.send(message.decode("utf-8"))

    def _socket_recv(self, socket: Any) -> Any:
        """Receive one websocket payload across websocket implementations/test doubles."""

        try:
            return socket.recv(timeout=self.recv_timeout_seconds, decode=False)
        except TypeError:
            return socket.recv(timeout=self.recv_timeout_seconds)

    def _receiver_loop(self, socket: Any) -> None:
        self._receiver_started.set()

        try:
            while not self._receiver_stop.is_set():
                try:
                    raw_message = self._socket_recv(socket)
                except TimeoutError:
                    continue
                except ConnectionClosedOK as exc:
                    if self._receiver_stop.is_set():
                        return
                    self._emit_disconnect(exc)
                    return
                except ConnectionClosed as exc:
                    if self._receiver_stop.is_set():
                        return
                    self._emit_error(self._format_abnormal_close_message(exc))
                    return
                except Exception as exc:
                    if self._receiver_stop.is_set():
                        return
                    self._emit_error(f"Voice orchestrator receive failed: {type(exc).__name__}")
                    return

                raw_bytes = raw_message.encode("utf-8") if isinstance(raw_message, str) else bytes(raw_message)

                try:
                    payload = self._decode_payload(raw_bytes)
                except UnicodeDecodeError:
                    self._emit_error("Voice orchestrator sent non-UTF-8 payloads.")
                    return
                except Exception:
                    self._emit_error("Voice orchestrator sent malformed JSON.")
                    return

                try:
                    event = decode_voice_server_event(payload)
                except Exception as exc:
                    self._emit_error(f"Voice orchestrator sent an unsupported event: {type(exc).__name__}")
                    return

                self._dispatch_event(event)
        finally:
            self._mark_socket_closed(socket)

    def _format_abnormal_close_message(self, exc: ConnectionClosed) -> str:
        code, reason = _close_code_and_reason(exc)
        if code is None:
            return "Voice orchestrator websocket closed unexpectedly."
        if reason:
            return f"Voice orchestrator websocket closed unexpectedly (code {code}: {reason})."
        return f"Voice orchestrator websocket closed unexpectedly (code {code})."

    def _emit_disconnect(self, exc: ConnectionClosed) -> None:
        code, reason = _close_code_and_reason(exc)
        if code is None:
            text = "Voice orchestrator websocket closed by server."
        elif reason:
            text = f"Voice orchestrator websocket closed by server (code {code}: {reason})."
        else:
            text = f"Voice orchestrator websocket closed by server (code {code})."
        self._emit_error(text)

    def _dispatch_event(self, event: VoiceServerEvent) -> None:
        if self._on_event is None:
            return

        try:
            self._on_event(event)
        except Exception:
            self._logger.exception(
                "Voice orchestrator event callback crashed; closing websocket client",
            )
            self._receiver_stop.set()
            try:
                self.close(code=1011, reason="client callback failure")
            except Exception:
                pass

    def _emit_error(self, text: str) -> None:
        from .voice_contracts import OrchestratorVoiceErrorEvent

        if self._on_event is not None:
            self._dispatch_event(OrchestratorVoiceErrorEvent(error=text))


__all__ = ["OrchestratorVoiceWebSocketClient"]
