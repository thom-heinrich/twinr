"""Run websocket turns against a remote Twinr orchestrator.

This module provides a blocking websocket client plus an asyncio-friendly
wrapper that can execute remote tool requests while enforcing transport bounds.
"""

from __future__ import annotations

import asyncio  # AUDIT-FIX(#7): Provide an async-safe wrapper for use from the uvicorn event loop.
from collections.abc import Callable, Mapping  # AUDIT-FIX(#5): Validate tool argument/output mappings before executing or returning them.
from typing import Any
import inspect  # AUDIT-FIX(#3): Preserve compatibility with injected connectors by passing only supported kwargs.
import json
from ipaddress import ip_address  # AUDIT-FIX(#1): Enforce secure transport rules while allowing loopback ws:// for local deployments.
from time import monotonic  # AUDIT-FIX(#3): Bound total turn duration and per-receive waits.
from urllib.parse import urlsplit  # AUDIT-FIX(#1): Validate websocket URL scheme and host before connecting.

from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidURI  # AUDIT-FIX(#3): Normalize network/protocol failures into deterministic exceptions.
from websockets.sync.client import connect as websocket_connect

from twinr.orchestrator.contracts import (
    OrchestratorAckEvent,
    OrchestratorClientTurnResult,
    OrchestratorToolRequest,
    OrchestratorToolResponse,
    OrchestratorTurnCompleteEvent,
    OrchestratorTurnRequest,
)


class OrchestratorWebSocketClient:
    """Run one Twinr turn through the edge-orchestrator websocket protocol."""

    _DEFAULT_OPEN_TIMEOUT_SECONDS = 10.0
    _DEFAULT_RECV_TIMEOUT_SECONDS = 90.0
    _DEFAULT_TURN_TIMEOUT_SECONDS = 300.0
    _DEFAULT_CLOSE_TIMEOUT_SECONDS = 10.0
    _DEFAULT_PING_INTERVAL_SECONDS = 20.0
    _DEFAULT_PING_TIMEOUT_SECONDS = 20.0
    _DEFAULT_MAX_MESSAGE_BYTES = 1_048_576
    _DEFAULT_MAX_QUEUE = 16
    _DEFAULT_MAX_ACK_EVENTS = 128
    _GENERIC_TOOL_FAILURE = "Tool execution failed"
    _MALFORMED_TOOL_REQUEST = "Malformed tool request"
    _UNSUPPORTED_TOOL_REQUEST = "Unsupported remote tool"
    _INVALID_TOOL_ARGUMENTS = "Tool arguments must be a JSON object"
    _INVALID_TOOL_OUTPUT = "Tool output must be a JSON object"
    _NON_SERIALIZABLE_TOOL_OUTPUT = "Tool output is not JSON-serializable"

    def __init__(
        self,
        url: str,
        *,
        shared_secret: str | None = None,
        connector: Callable[..., Any] | None = None,
        open_timeout_seconds: float = _DEFAULT_OPEN_TIMEOUT_SECONDS,
        recv_timeout_seconds: float = _DEFAULT_RECV_TIMEOUT_SECONDS,
        turn_timeout_seconds: float | None = _DEFAULT_TURN_TIMEOUT_SECONDS,
        close_timeout_seconds: float = _DEFAULT_CLOSE_TIMEOUT_SECONDS,
        ping_interval_seconds: float | None = _DEFAULT_PING_INTERVAL_SECONDS,
        ping_timeout_seconds: float | None = _DEFAULT_PING_TIMEOUT_SECONDS,
        max_message_bytes: int = _DEFAULT_MAX_MESSAGE_BYTES,
        max_queue: int = _DEFAULT_MAX_QUEUE,
        max_ack_events: int = _DEFAULT_MAX_ACK_EVENTS,
        require_tls: bool = True,
    ) -> None:
        self.url = self._validate_url(url, require_tls=require_tls)  # AUDIT-FIX(#1): Reject invalid or insecure non-loopback websocket URLs at construction time.
        self.shared_secret = self._normalize_shared_secret(shared_secret)  # AUDIT-FIX(#9): Reject CR/LF/NUL in secrets before turning them into headers.
        self._connector = connector or websocket_connect
        self.open_timeout_seconds = self._validate_positive_float(open_timeout_seconds, "open_timeout_seconds")  # AUDIT-FIX(#3): Validate timeout configuration eagerly to avoid undefined runtime behavior.
        self.recv_timeout_seconds = self._validate_positive_float(recv_timeout_seconds, "recv_timeout_seconds")  # AUDIT-FIX(#3): Validate timeout configuration eagerly to avoid undefined runtime behavior.
        self.turn_timeout_seconds = self._validate_optional_positive_float(turn_timeout_seconds, "turn_timeout_seconds")  # AUDIT-FIX(#3): Bound worst-case hangs even if partial traffic continues.
        self.close_timeout_seconds = self._validate_positive_float(close_timeout_seconds, "close_timeout_seconds")  # AUDIT-FIX(#3): Ensure clean close has a deterministic upper bound.
        self.ping_interval_seconds = self._validate_optional_positive_float(ping_interval_seconds, "ping_interval_seconds")  # AUDIT-FIX(#3): Keep transport liveness detection configurable and valid.
        self.ping_timeout_seconds = self._validate_optional_positive_float(ping_timeout_seconds, "ping_timeout_seconds")  # AUDIT-FIX(#3): Keep transport liveness detection configurable and valid.
        self.max_message_bytes = self._validate_positive_int(max_message_bytes, "max_message_bytes")  # AUDIT-FIX(#4): Bound inbound payload size on low-memory hardware.
        self.max_queue = self._validate_positive_int(max_queue, "max_queue")  # AUDIT-FIX(#4): Bound queued inbound frames on low-memory hardware.
        self.max_ack_events = self._validate_positive_int(max_ack_events, "max_ack_events")  # AUDIT-FIX(#8): Prevent unbounded ack retention from exhausting memory.

    async def arun_turn(
        self,
        request: OrchestratorTurnRequest,
        *,
        tool_handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]],
        on_text_delta: Callable[[str], None] | None = None,
        on_ack: Callable[[OrchestratorAckEvent], None] | None = None,
    ) -> OrchestratorClientTurnResult:
        """Run one orchestrator turn without blocking the active event loop."""

        loop = asyncio.get_running_loop()

        def _threadsafe_text_delta(delta: str) -> None:
            if on_text_delta is None:
                return
            loop.call_soon_threadsafe(on_text_delta, delta)  # AUDIT-FIX(#7): Marshal callbacks back onto the event loop thread when using the async wrapper.

        def _threadsafe_ack(event: OrchestratorAckEvent) -> None:
            if on_ack is None:
                return
            loop.call_soon_threadsafe(on_ack, event)  # AUDIT-FIX(#7): Marshal callbacks back onto the event loop thread when using the async wrapper.

        return await asyncio.to_thread(  # AUDIT-FIX(#7): Keep the blocking sync websocket client off the main async event loop.
            self.run_turn,
            request,
            tool_handlers=tool_handlers,
            on_text_delta=_threadsafe_text_delta if on_text_delta is not None else None,
            on_ack=_threadsafe_ack if on_ack is not None else None,
        )

    def run_turn(
        self,
        request: OrchestratorTurnRequest,
        *,
        tool_handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]],
        on_text_delta: Callable[[str], None] | None = None,
        on_ack: Callable[[OrchestratorAckEvent], None] | None = None,
    ) -> OrchestratorClientTurnResult:
        """Run one blocking websocket turn against the orchestrator server."""

        handlers = self._snapshot_tool_handlers(tool_handlers)  # AUDIT-FIX(#6): Snapshot handler mappings once per turn to avoid mid-turn races from external mutation.
        headers = None
        if self.shared_secret is not None:
            headers = {"x-twinr-secret": self.shared_secret}
        ack_events: list[OrchestratorAckEvent] = []
        connector_kwargs = self._build_connector_kwargs(headers)
        deadline = self._compute_deadline()  # AUDIT-FIX(#3): Apply a total turn budget in addition to per-receive timeouts.

        try:
            with self._connector(self.url, **connector_kwargs) as websocket:
                self._send_json(websocket, request.to_payload(), context="turn request")
                while True:
                    payload = self._recv_payload(websocket, deadline=deadline)
                    message_type = self._sanitize_text(payload.get("type", ""), limit=64)
                    if message_type == "ack":
                        event = OrchestratorAckEvent(
                            ack_id=self._sanitize_text(payload.get("ack_id", ""), limit=256),
                            text=str(payload.get("text", "") or ""),
                        )
                        ack_events.append(event)
                        if len(ack_events) > self.max_ack_events:
                            ack_events.pop(0)  # AUDIT-FIX(#8): Retain only the most recent ack events to cap memory growth on the RPi.
                        if on_ack is not None:
                            on_ack(event)
                        elif on_text_delta is not None:
                            on_text_delta(event.text)
                        continue
                    if message_type == "text_delta":
                        delta = str(payload.get("delta", "") or "")
                        if on_text_delta is not None and delta:
                            on_text_delta(delta)
                        continue
                    if message_type == "tool_request":
                        response = self._handle_tool_request(payload, handlers)
                        self._send_json(websocket, response.to_payload(), context="tool response")
                        continue
                    if message_type == "turn_complete":
                        try:
                            completed = OrchestratorTurnCompleteEvent.from_payload(payload)
                        except Exception as exc:  # AUDIT-FIX(#4): Convert malformed completion payloads into clear protocol errors.
                            raise RuntimeError("Received malformed turn_complete payload from orchestrator") from exc
                        return OrchestratorClientTurnResult(
                            text=completed.text,
                            rounds=completed.rounds,
                            used_web_search=completed.used_web_search,
                            response_id=completed.response_id,
                            request_id=completed.request_id,
                            model=completed.model,
                            token_usage=completed.token_usage,
                            tool_calls=completed.tool_calls,
                            tool_results=completed.tool_results,
                            ack_events=ack_events,
                        )
                    if message_type == "turn_error":
                        remote_error = self._sanitize_text(payload.get("error", "Orchestrator turn failed"), limit=512)  # AUDIT-FIX(#10): Strip control characters and bound remote error size before surfacing it locally.
                        raise RuntimeError(remote_error or "Orchestrator turn failed")
                    raise RuntimeError(  # AUDIT-FIX(#4): Fail fast on unknown protocol messages instead of silently looping forever.
                        f"Unsupported orchestrator message type: {message_type or '<empty>'}"
                    )
        except TimeoutError as exc:  # AUDIT-FIX(#3): Surface deterministic timeout failures for upper-layer retry/fallback logic.
            raise TimeoutError("Timed out while waiting for orchestrator websocket traffic") from exc
        except ConnectionClosed as exc:  # AUDIT-FIX(#3): Surface mid-turn disconnects consistently for callers.
            raise ConnectionError("Orchestrator websocket connection closed before turn completion") from exc
        except (InvalidURI, InvalidHandshake, OSError) as exc:  # AUDIT-FIX(#3): Normalize connection-establishment failures into a stable exception type.
            raise ConnectionError("Failed to connect to orchestrator websocket") from exc

    def _handle_tool_request(
        self,
        payload: dict[str, Any],
        tool_handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]],
    ) -> OrchestratorToolResponse:
        """Execute one remote tool request through the local handler map."""

        raw_call_id = self._sanitize_text(payload.get("call_id", ""), limit=256)
        try:
            request_event = OrchestratorToolRequest.from_payload(payload)
        except Exception as exc:
            if raw_call_id:
                return OrchestratorToolResponse(
                    call_id=raw_call_id,
                    ok=False,
                    error=self._MALFORMED_TOOL_REQUEST,
                )  # AUDIT-FIX(#5): Respond to malformed tool requests when a call_id is available instead of aborting the entire turn.
            raise RuntimeError("Received malformed tool_request payload from orchestrator") from exc

        handler = tool_handlers.get(request_event.name)
        if handler is None:
            safe_tool_name = self._sanitize_text(request_event.name, limit=128)
            return OrchestratorToolResponse(
                call_id=request_event.call_id,
                ok=False,
                error=f"{self._UNSUPPORTED_TOOL_REQUEST}: {safe_tool_name or '<unknown>'}",
            )

        if not isinstance(request_event.arguments, Mapping):
            return OrchestratorToolResponse(
                call_id=request_event.call_id,
                ok=False,
                error=self._INVALID_TOOL_ARGUMENTS,
            )  # AUDIT-FIX(#5): Reject malformed tool arguments explicitly instead of relying on dict(...) coercion side-effects.

        try:
            output = handler(dict(request_event.arguments))
        except Exception:
            return OrchestratorToolResponse(
                call_id=request_event.call_id,
                ok=False,
                error=self._GENERIC_TOOL_FAILURE,
            )  # AUDIT-FIX(#2): Do not leak local exception text, paths, or secrets back to the remote orchestrator.

        if not isinstance(output, Mapping):
            return OrchestratorToolResponse(
                call_id=request_event.call_id,
                ok=False,
                error=self._INVALID_TOOL_OUTPUT,
            )  # AUDIT-FIX(#5): Enforce the declared tool contract before building a success response.

        output_payload = dict(output)
        try:
            json.dumps(output_payload, ensure_ascii=False)
        except (TypeError, ValueError):
            return OrchestratorToolResponse(
                call_id=request_event.call_id,
                ok=False,
                error=self._NON_SERIALIZABLE_TOOL_OUTPUT,
            )  # AUDIT-FIX(#5): Prevent mid-send serialization crashes from invalid tool outputs.

        return OrchestratorToolResponse(
            call_id=request_event.call_id,
            ok=True,
            output=output_payload,
        )

    def _build_connector_kwargs(self, headers: dict[str, str] | None) -> dict[str, Any]:
        """Build the subset of websocket connector kwargs this client uses."""

        candidate_kwargs: dict[str, Any] = {
            "additional_headers": headers,
            "open_timeout": self.open_timeout_seconds,
            "close_timeout": self.close_timeout_seconds,
            "ping_interval": self.ping_interval_seconds,
            "ping_timeout": self.ping_timeout_seconds,
            "max_size": self.max_message_bytes,
            "max_queue": self.max_queue,
        }
        return self._filter_supported_kwargs(self._connector, candidate_kwargs)  # AUDIT-FIX(#3): Avoid breaking custom injected connectors that accept only a subset of websocket kwargs.

    @staticmethod
    def _filter_supported_kwargs(func: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
        """Drop connector kwargs that the injected connector does not accept."""

        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):
            return kwargs

        parameters = signature.parameters.values()
        if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
            return kwargs

        supported_names = {
            name
            for name, parameter in signature.parameters.items()
            if parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        if "additional_headers" in kwargs and "additional_headers" not in supported_names and "extra_headers" in supported_names:
            kwargs = {**kwargs, "extra_headers": kwargs["additional_headers"]}  # AUDIT-FIX(#3): Preserve auth header delivery for injected connectors that still use the older extra_headers name.
            kwargs.pop("additional_headers", None)
        return {name: value for name, value in kwargs.items() if name in supported_names}

    def _recv_payload(self, websocket: Any, *, deadline: float | None) -> dict[str, Any]:
        """Receive one JSON object payload from the websocket transport."""

        timeout = self.recv_timeout_seconds
        if deadline is not None:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError("Orchestrator turn exceeded the configured timeout budget")  # AUDIT-FIX(#3): Enforce total turn timeout even if the connection stays open.
            timeout = min(timeout, remaining)

        raw_message = websocket.recv(timeout=timeout)
        if isinstance(raw_message, bytes):
            try:
                raw_text = raw_message.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise RuntimeError("Received non-UTF-8 payload from orchestrator") from exc  # AUDIT-FIX(#4): Reject binary/non-UTF-8 protocol frames explicitly.
        else:
            raw_text = str(raw_message)

        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Received malformed JSON from orchestrator") from exc  # AUDIT-FIX(#4): Normalize bad JSON into a clear protocol error.

        if not isinstance(payload, dict):
            raise RuntimeError("Received non-object message from orchestrator")  # AUDIT-FIX(#4): Enforce object-shaped protocol messages before field access.
        return payload

    @staticmethod
    def _send_json(websocket: Any, payload: Any, *, context: str) -> None:
        """Serialize and send one JSON payload over the websocket."""

        try:
            message = json.dumps(payload, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Failed to serialize {context}") from exc  # AUDIT-FIX(#5): Convert serialization failures into explicit local errors.
        websocket.send(message)

    def _compute_deadline(self) -> float | None:
        """Compute the absolute deadline for the current turn, if bounded."""

        if self.turn_timeout_seconds is None:
            return None
        return monotonic() + self.turn_timeout_seconds

    @staticmethod
    def _snapshot_tool_handlers(
        tool_handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]],
    ) -> dict[str, Callable[[dict[str, Any]], dict[str, Any]]]:
        """Copy and validate the per-turn local tool handler mapping."""

        handlers = dict(tool_handlers)
        for name, handler in handlers.items():
            if not callable(handler):
                raise TypeError(f"tool_handlers[{name!r}] must be callable")  # AUDIT-FIX(#6): Fail early on invalid handler registration instead of crashing mid-turn.
        return handlers

    @staticmethod
    def _sanitize_text(value: Any, *, limit: int) -> str:
        """Strip control characters and bound transport text length."""

        text = "" if value is None else str(value)
        cleaned = "".join(character if character.isprintable() else " " for character in text).strip()
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[: limit - 1].rstrip() + "…"

    @staticmethod
    def _normalize_shared_secret(shared_secret: str | None) -> str | None:
        """Normalize the shared secret used for websocket authentication."""

        secret = (shared_secret or "").strip()
        if not secret:
            return None
        if any(character in secret for character in ("\r", "\n", "\x00")):
            raise ValueError("shared_secret contains invalid control characters")  # AUDIT-FIX(#9): Block header injection and malformed request generation from bad secret values.
        return secret

    @classmethod
    def _validate_url(cls, url: str, *, require_tls: bool) -> str:
        """Normalize and validate the orchestrator websocket URL."""

        normalized_url = url.strip()
        if not normalized_url:
            raise ValueError("url must not be empty")  # AUDIT-FIX(#1): Fail fast on missing websocket endpoints.

        parsed = urlsplit(normalized_url)
        scheme = parsed.scheme.lower()
        host = parsed.hostname or ""
        if scheme not in {"ws", "wss"}:
            raise ValueError("url must use ws:// or wss://")  # AUDIT-FIX(#1): Restrict the client to websocket transports only.
        if not host:
            raise ValueError("url must include a hostname")  # AUDIT-FIX(#1): Reject ambiguous or malformed websocket URLs.
        if require_tls and scheme != "wss" and not cls._is_loopback_host(host):
            raise ValueError("Non-loopback orchestrator URLs must use wss://")  # AUDIT-FIX(#1): Prevent sending senior data or secrets over plaintext network links.
        return normalized_url

    @staticmethod
    def _is_loopback_host(host: str) -> bool:
        """Return whether the host resolves to a loopback endpoint."""

        normalized_host = host.strip().rstrip(".").lower()
        if normalized_host == "localhost":
            return True
        try:
            return ip_address(normalized_host).is_loopback
        except ValueError:
            return False

    @staticmethod
    def _validate_positive_float(value: float, name: str) -> float:
        """Validate a required positive floating-point setting."""

        if value <= 0:
            raise ValueError(f"{name} must be > 0")
        return float(value)

    @staticmethod
    def _validate_optional_positive_float(value: float | None, name: str) -> float | None:
        """Validate an optional positive floating-point setting."""

        if value is None:
            return None
        if value <= 0:
            raise ValueError(f"{name} must be > 0 when provided")
        return float(value)

    @staticmethod
    def _validate_positive_int(value: int, name: str) -> int:
        """Validate a required positive integer setting."""

        if value <= 0:
            raise ValueError(f"{name} must be > 0")
        return int(value)
