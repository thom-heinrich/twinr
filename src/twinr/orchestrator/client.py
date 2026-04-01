# CHANGELOG: 2026-03-29
# BUG-1: Deduplicated tool_request call_id handling so retransmits/replays don't execute physical side-effecting tools twice.
# BUG-2: Added bounded tool execution with async-handler support; hung tools can no longer stall a turn forever.
# BUG-3: Ack retention is now bounded by both count and bytes; ack text no longer pollutes streamed model text by default.
# BUG-4: Enforced strict JSON (reject NaN/Infinity, non-finite config values, and mismatched request_id) to avoid silent protocol corruption.
# SEC-1: Disabled implicit env/system proxying by default and exposed explicit proxy/TLS knobs to avoid leaking auth headers through surprise proxies.
# SEC-2: Disabled websocket compression and User-Agent fingerprinting by default, added outbound size limits, and enforced event/tool budgets for Pi-safe operation.
# IMP-1: arun_turn now prefers the native websockets asyncio client instead of always hopping through a worker thread.
# IMP-2: Added schema-first execution budgets plus optional handler context kwargs (call_id/tool_name/turn_request_id/deadline_monotonic/timeout_seconds).

"""Run websocket turns against a remote Twinr orchestrator.

This module provides blocking and asyncio-native websocket clients that can
execute remote tool requests while enforcing transport, execution, and memory
budgets suitable for Raspberry Pi 4 deployments.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import inspect
import json
from ipaddress import ip_address
import math
import ssl
import threading
from time import monotonic
from typing import Any
from urllib.parse import urlsplit

from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidURI

try:
    from websockets.exceptions import InvalidProxy
except Exception:  # pragma: no cover - depends on websockets version.
    InvalidProxy = None  # type: ignore[assignment]

try:
    from websockets.asyncio.client import connect as websocket_connect_async
except Exception:  # pragma: no cover - native asyncio client may be unavailable on older websockets releases.
    websocket_connect_async = None  # type: ignore[assignment]

from websockets.sync.client import connect as websocket_connect

from twinr.orchestrator.contracts import (
    OrchestratorAckEvent,
    OrchestratorClientTurnResult,
    OrchestratorToolRequest,
    OrchestratorToolResponse,
    OrchestratorTurnCompleteEvent,
    OrchestratorTurnRequest,
    _json_safe,
)

_TransportEventCallback = Callable[[str, dict[str, Any]], Any]


class OrchestratorProtocolError(RuntimeError):
    """Raised when the orchestrator violates the websocket turn protocol."""


class OrchestratorBudgetExceeded(RuntimeError):
    """Raised when the orchestrator exceeds a local client-side safety budget."""


@dataclass(slots=True)
class _ToolTimeoutContext:
    timeout_seconds: float | None
    timeout_scope: str | None  # "tool", "turn", or None.


@dataclass(slots=True)
class _PreparedToolRequest:
    request_event: OrchestratorToolRequest | None
    signature: str
    cached_response: OrchestratorToolResponse | None = None


def _is_connect_failure_exception(exc: Exception) -> bool:
    if isinstance(exc, (InvalidURI, InvalidHandshake, OSError)):
        return True
    return InvalidProxy is not None and isinstance(exc, InvalidProxy)


class OrchestratorWebSocketClient:
    """Run one Twinr turn through the edge-orchestrator websocket protocol."""

    _DEFAULT_OPEN_TIMEOUT_SECONDS = 10.0
    _DEFAULT_RECV_TIMEOUT_SECONDS = 90.0
    _DEFAULT_TURN_TIMEOUT_SECONDS = 300.0
    _DEFAULT_TOOL_TIMEOUT_SECONDS = 60.0
    _DEFAULT_CLOSE_TIMEOUT_SECONDS = 10.0
    _DEFAULT_PING_INTERVAL_SECONDS = 20.0
    _DEFAULT_PING_TIMEOUT_SECONDS = 20.0
    _DEFAULT_MAX_MESSAGE_BYTES = 1_048_576
    _DEFAULT_MAX_OUTBOUND_MESSAGE_BYTES = 1_048_576
    _DEFAULT_MAX_QUEUE = 16
    _DEFAULT_MAX_ACK_EVENTS = 128
    _DEFAULT_MAX_ACK_TEXT_CHARS = 4096
    _DEFAULT_MAX_ACK_BUFFER_BYTES = 65_536
    _DEFAULT_MAX_PROTOCOL_EVENTS = 2048
    _DEFAULT_MAX_TOOL_REQUESTS_PER_TURN = 32
    _DEFAULT_MAX_STREAMED_TEXT_BYTES = 1_048_576

    _GENERIC_TOOL_FAILURE = "Tool execution failed"
    _MALFORMED_TOOL_REQUEST = "Malformed tool request"
    _UNSUPPORTED_TOOL_REQUEST = "Unsupported remote tool"
    _INVALID_TOOL_ARGUMENTS = "Tool arguments must be a JSON object"
    _INVALID_TOOL_OUTPUT = "Tool output must be a JSON object"
    _NON_SERIALIZABLE_TOOL_OUTPUT = "Tool output is not strict JSON-serializable"
    _NON_SERIALIZABLE_REQUEST = "Turn request is not strict JSON-serializable"
    _TOOL_OUTPUT_TOO_LARGE = "Tool output exceeds the configured size limit"
    _TOOL_TIMEOUT = "Tool execution timed out"
    _CONFLICTING_DUPLICATE_TOOL_REQUEST = "Conflicting duplicate tool request"
    _TURN_TIMEOUT = "Orchestrator turn exceeded the configured timeout budget"
    _PROTOCOL_EVENT_BUDGET_EXCEEDED = "Orchestrator exceeded the configured protocol-event budget"
    _TOOL_REQUEST_BUDGET_EXCEEDED = "Orchestrator exceeded the configured tool-request budget"
    _STREAMED_TEXT_BUDGET_EXCEEDED = "Orchestrator streamed too much text before completion"

    def __init__(
        self,
        url: str,
        *,
        shared_secret: str | None = None,
        connector: Callable[..., Any] | None = None,
        async_connector: Callable[..., Any] | None = None,
        open_timeout_seconds: float = _DEFAULT_OPEN_TIMEOUT_SECONDS,
        recv_timeout_seconds: float = _DEFAULT_RECV_TIMEOUT_SECONDS,
        turn_timeout_seconds: float | None = _DEFAULT_TURN_TIMEOUT_SECONDS,
        tool_timeout_seconds: float | None = _DEFAULT_TOOL_TIMEOUT_SECONDS,
        close_timeout_seconds: float = _DEFAULT_CLOSE_TIMEOUT_SECONDS,
        ping_interval_seconds: float | None = _DEFAULT_PING_INTERVAL_SECONDS,
        ping_timeout_seconds: float | None = _DEFAULT_PING_TIMEOUT_SECONDS,
        max_message_bytes: int = _DEFAULT_MAX_MESSAGE_BYTES,
        max_outbound_message_bytes: int = _DEFAULT_MAX_OUTBOUND_MESSAGE_BYTES,
        max_queue: int = _DEFAULT_MAX_QUEUE,
        max_ack_events: int = _DEFAULT_MAX_ACK_EVENTS,
        max_ack_text_chars: int = _DEFAULT_MAX_ACK_TEXT_CHARS,
        max_ack_buffer_bytes: int = _DEFAULT_MAX_ACK_BUFFER_BYTES,
        max_protocol_events: int = _DEFAULT_MAX_PROTOCOL_EVENTS,
        max_tool_requests_per_turn: int = _DEFAULT_MAX_TOOL_REQUESTS_PER_TURN,
        max_streamed_text_bytes: int = _DEFAULT_MAX_STREAMED_TEXT_BYTES,
        require_tls: bool = True,
        proxy: str | bool | None = None,
        compression: str | None = None,
        user_agent_header: str | None = None,
        ssl_context: ssl.SSLContext | None = None,
        server_hostname: str | None = None,
        origin: str | None = None,
        subprotocols: Sequence[str] | None = None,
        mirror_ack_text_to_text_delta: bool = False,
    ) -> None:
        self.url = self._validate_url(url, require_tls=require_tls)
        self.shared_secret = self._normalize_shared_secret(shared_secret)
        self._connector = connector or websocket_connect
        self._async_connector = async_connector
        if self._async_connector is None and connector is None:
            self._async_connector = websocket_connect_async

        self.open_timeout_seconds = self._validate_positive_float(open_timeout_seconds, "open_timeout_seconds")
        self.recv_timeout_seconds = self._validate_positive_float(recv_timeout_seconds, "recv_timeout_seconds")
        self.turn_timeout_seconds = self._validate_optional_positive_float(turn_timeout_seconds, "turn_timeout_seconds")
        self.tool_timeout_seconds = self._validate_optional_positive_float(tool_timeout_seconds, "tool_timeout_seconds")  # BREAKING: tool execution is now bounded by default to prevent indefinite hangs.
        self.close_timeout_seconds = self._validate_positive_float(close_timeout_seconds, "close_timeout_seconds")
        self.ping_interval_seconds = self._validate_optional_positive_float(ping_interval_seconds, "ping_interval_seconds")
        self.ping_timeout_seconds = self._validate_optional_positive_float(ping_timeout_seconds, "ping_timeout_seconds")
        self.max_message_bytes = self._validate_positive_int(max_message_bytes, "max_message_bytes")
        self.max_outbound_message_bytes = self._validate_positive_int(max_outbound_message_bytes, "max_outbound_message_bytes")
        self.max_queue = self._validate_positive_int(max_queue, "max_queue")
        self.max_ack_events = self._validate_positive_int(max_ack_events, "max_ack_events")
        self.max_ack_text_chars = self._validate_positive_int(max_ack_text_chars, "max_ack_text_chars")
        self.max_ack_buffer_bytes = self._validate_positive_int(max_ack_buffer_bytes, "max_ack_buffer_bytes")
        self.max_protocol_events = self._validate_positive_int(max_protocol_events, "max_protocol_events")
        self.max_tool_requests_per_turn = self._validate_positive_int(max_tool_requests_per_turn, "max_tool_requests_per_turn")
        self.max_streamed_text_bytes = self._validate_positive_int(max_streamed_text_bytes, "max_streamed_text_bytes")

        self.proxy = self._normalize_proxy(proxy)  # BREAKING: implicit env/system proxy auto-discovery is disabled unless proxy=True or a proxy URL is provided explicitly.
        self.compression = self._normalize_compression(compression)  # BREAKING: websocket compression is disabled by default to reduce Pi RAM/CPU pressure and compression-related risk.
        self.user_agent_header = self._normalize_optional_header_value(user_agent_header, "user_agent_header")  # BREAKING: default removes the User-Agent header to reduce fingerprinting.
        self.ssl_context = self._validate_ssl_context(ssl_context)
        self.server_hostname = self._normalize_optional_host(server_hostname, "server_hostname")
        self.origin = self._normalize_optional_header_value(origin, "origin")
        self.subprotocols = self._normalize_subprotocols(subprotocols)
        self.mirror_ack_text_to_text_delta = bool(mirror_ack_text_to_text_delta)  # BREAKING: ack text no longer feeds on_text_delta unless this is explicitly enabled.

    async def arun_turn(
        self,
        request: OrchestratorTurnRequest,
        *,
        tool_handlers: Mapping[str, Callable[..., Any]],
        on_text_delta: Callable[[str], Any] | None = None,
        on_ack: Callable[[OrchestratorAckEvent], Any] | None = None,
        on_transport_event: _TransportEventCallback | None = None,
    ) -> OrchestratorClientTurnResult:
        """Run one orchestrator turn without blocking the active event loop."""

        handlers = self._snapshot_tool_handlers(tool_handlers)
        if self._async_connector is None:
            loop = asyncio.get_running_loop()

            def _threadsafe_text_delta(delta: str) -> None:
                if on_text_delta is None:
                    return
                loop.call_soon_threadsafe(on_text_delta, delta)

            def _threadsafe_ack(event: OrchestratorAckEvent) -> None:
                if on_ack is None:
                    return
                loop.call_soon_threadsafe(on_ack, event)

            return await asyncio.to_thread(
                self.run_turn,
                request,
                tool_handlers=handlers,
                on_text_delta=_threadsafe_text_delta if on_text_delta is not None else None,
                on_ack=_threadsafe_ack if on_ack is not None else None,
            )

        headers = self._build_headers()
        connector_kwargs = self._build_connector_kwargs(self._async_connector, headers=headers)
        deadline = self._compute_deadline()
        request_payload = self._build_request_payload(request)
        expected_request_id = self._extract_expected_request_id(request_payload)
        ack_records: deque[tuple[OrchestratorAckEvent, int]] = deque()
        ack_buffer_bytes = 0
        streamed_text_bytes = 0
        protocol_events = 0
        handled_tool_calls: dict[str, tuple[str, OrchestratorToolResponse]] = {}
        first_server_event_emitted = False

        try:
            async with self._async_connector(self.url, **connector_kwargs) as websocket:
                await self._emit_transport_event_async(
                    on_transport_event,
                    "ws_connected",
                    request_id=expected_request_id or None,
                )
                await self._async_send_json(websocket, request_payload, context="turn request")
                await self._emit_transport_event_async(
                    on_transport_event,
                    "turn_request_sent",
                    request_id=expected_request_id or None,
                )
                while True:
                    payload = await self._async_recv_payload(websocket, deadline=deadline)
                    protocol_events = self._count_protocol_event(protocol_events)
                    message_type = self._sanitize_text(payload.get("type", ""), limit=64)
                    if not first_server_event_emitted:
                        first_server_event_emitted = True
                        await self._emit_transport_event_async(
                            on_transport_event,
                            "first_server_event",
                            message_type=message_type or None,
                        )

                    if message_type == "ack":
                        event = self._build_ack_event(payload)
                        ack_buffer_bytes = self._append_ack_event(ack_records, ack_buffer_bytes, event)
                        if on_ack is not None:
                            await self._maybe_await_callback(on_ack(event))
                        elif on_text_delta is not None and self.mirror_ack_text_to_text_delta and event.text:
                            await self._maybe_await_callback(on_text_delta(event.text))
                        continue

                    if message_type == "text_delta":
                        delta = self._coerce_stream_text(payload.get("delta", ""))
                        streamed_text_bytes = self._count_streamed_text(streamed_text_bytes, delta)
                        if on_text_delta is not None and delta:
                            await self._maybe_await_callback(on_text_delta(delta))
                        continue

                    if message_type == "tool_request":
                        tool_name = self._sanitize_text(payload.get("name", ""), limit=128)
                        call_id = self._sanitize_text(payload.get("call_id", ""), limit=256)
                        await self._emit_transport_event_async(
                            on_transport_event,
                            "tool_request_received",
                            tool_name=tool_name or None,
                            call_id=call_id or None,
                        )
                        response = await self._handle_tool_request_async(
                            payload,
                            handlers,
                            deadline=deadline,
                            turn_request_id=expected_request_id or None,
                            handled_tool_calls=handled_tool_calls,
                        )
                        await self._async_send_json(websocket, response.to_payload(), context="tool response")
                        await self._emit_transport_event_async(
                            on_transport_event,
                            "tool_response_sent",
                            tool_name=tool_name or None,
                            call_id=response.call_id or None,
                            ok=response.ok,
                            error=response.error,
                        )
                        continue

                    if message_type == "turn_complete":
                        completed = self._parse_turn_complete(payload)
                        self._validate_completed_request_id(completed.request_id, expected_request_id)
                        await self._emit_transport_event_async(
                            on_transport_event,
                            "turn_complete_received",
                            request_id=completed.request_id,
                            response_id=completed.response_id,
                            model=completed.model,
                            rounds=completed.rounds,
                            used_web_search=completed.used_web_search,
                        )
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
                            ack_events=[event for event, _ in ack_records],
                        )

                    if message_type == "turn_error":
                        remote_error = self._sanitize_text(payload.get("error", "Orchestrator turn failed"), limit=512)
                        await self._emit_transport_event_async(
                            on_transport_event,
                            "turn_error_received",
                            error=remote_error or None,
                        )
                        raise RuntimeError(remote_error or "Orchestrator turn failed")

                    raise OrchestratorProtocolError(
                        f"Unsupported orchestrator message type: {message_type or '<empty>'}"
                    )
        except TimeoutError as exc:
            message = str(exc).strip() or "Timed out while waiting for orchestrator websocket traffic"
            raise TimeoutError(message) from exc
        except ConnectionClosed as exc:
            raise ConnectionError("Orchestrator websocket connection closed before turn completion") from exc
        except Exception as exc:
            if _is_connect_failure_exception(exc):
                raise ConnectionError("Failed to connect to orchestrator websocket") from exc
            raise

    def run_turn(
        self,
        request: OrchestratorTurnRequest,
        *,
        tool_handlers: Mapping[str, Callable[..., Any]],
        on_text_delta: Callable[[str], Any] | None = None,
        on_ack: Callable[[OrchestratorAckEvent], Any] | None = None,
        on_transport_event: _TransportEventCallback | None = None,
    ) -> OrchestratorClientTurnResult:
        """Run one blocking websocket turn against the orchestrator server."""

        handlers = self._snapshot_tool_handlers(tool_handlers)
        headers = self._build_headers()
        connector_kwargs = self._build_connector_kwargs(self._connector, headers=headers)
        deadline = self._compute_deadline()
        request_payload = self._build_request_payload(request)
        expected_request_id = self._extract_expected_request_id(request_payload)
        ack_records: deque[tuple[OrchestratorAckEvent, int]] = deque()
        ack_buffer_bytes = 0
        streamed_text_bytes = 0
        protocol_events = 0
        handled_tool_calls: dict[str, tuple[str, OrchestratorToolResponse]] = {}
        first_server_event_emitted = False

        try:
            with self._connector(self.url, **connector_kwargs) as websocket:
                self._emit_transport_event_blocking(
                    on_transport_event,
                    "ws_connected",
                    request_id=expected_request_id or None,
                )
                self._send_json(websocket, request_payload, context="turn request")
                self._emit_transport_event_blocking(
                    on_transport_event,
                    "turn_request_sent",
                    request_id=expected_request_id or None,
                )
                while True:
                    payload = self._recv_payload(websocket, deadline=deadline)
                    protocol_events = self._count_protocol_event(protocol_events)
                    message_type = self._sanitize_text(payload.get("type", ""), limit=64)
                    if not first_server_event_emitted:
                        first_server_event_emitted = True
                        self._emit_transport_event_blocking(
                            on_transport_event,
                            "first_server_event",
                            message_type=message_type or None,
                        )

                    if message_type == "ack":
                        event = self._build_ack_event(payload)
                        ack_buffer_bytes = self._append_ack_event(ack_records, ack_buffer_bytes, event)
                        if on_ack is not None:
                            self._await_callback_blocking(on_ack(event))
                        elif on_text_delta is not None and self.mirror_ack_text_to_text_delta and event.text:
                            self._await_callback_blocking(on_text_delta(event.text))
                        continue

                    if message_type == "text_delta":
                        delta = self._coerce_stream_text(payload.get("delta", ""))
                        streamed_text_bytes = self._count_streamed_text(streamed_text_bytes, delta)
                        if on_text_delta is not None and delta:
                            self._await_callback_blocking(on_text_delta(delta))
                        continue

                    if message_type == "tool_request":
                        tool_name = self._sanitize_text(payload.get("name", ""), limit=128)
                        call_id = self._sanitize_text(payload.get("call_id", ""), limit=256)
                        self._emit_transport_event_blocking(
                            on_transport_event,
                            "tool_request_received",
                            tool_name=tool_name or None,
                            call_id=call_id or None,
                        )
                        response = self._handle_tool_request_sync(
                            payload,
                            handlers,
                            deadline=deadline,
                            turn_request_id=expected_request_id or None,
                            handled_tool_calls=handled_tool_calls,
                        )
                        self._send_json(websocket, response.to_payload(), context="tool response")
                        self._emit_transport_event_blocking(
                            on_transport_event,
                            "tool_response_sent",
                            tool_name=tool_name or None,
                            call_id=response.call_id or None,
                            ok=response.ok,
                            error=response.error,
                        )
                        continue

                    if message_type == "turn_complete":
                        completed = self._parse_turn_complete(payload)
                        self._validate_completed_request_id(completed.request_id, expected_request_id)
                        self._emit_transport_event_blocking(
                            on_transport_event,
                            "turn_complete_received",
                            request_id=completed.request_id,
                            response_id=completed.response_id,
                            model=completed.model,
                            rounds=completed.rounds,
                            used_web_search=completed.used_web_search,
                        )
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
                            ack_events=[event for event, _ in ack_records],
                        )

                    if message_type == "turn_error":
                        remote_error = self._sanitize_text(payload.get("error", "Orchestrator turn failed"), limit=512)
                        self._emit_transport_event_blocking(
                            on_transport_event,
                            "turn_error_received",
                            error=remote_error or None,
                        )
                        raise RuntimeError(remote_error or "Orchestrator turn failed")

                    raise OrchestratorProtocolError(
                        f"Unsupported orchestrator message type: {message_type or '<empty>'}"
                    )
        except TimeoutError as exc:
            message = str(exc).strip() or "Timed out while waiting for orchestrator websocket traffic"
            raise TimeoutError(message) from exc
        except ConnectionClosed as exc:
            raise ConnectionError("Orchestrator websocket connection closed before turn completion") from exc
        except Exception as exc:
            if _is_connect_failure_exception(exc):
                raise ConnectionError("Failed to connect to orchestrator websocket") from exc
            raise

    async def _handle_tool_request_async(
        self,
        payload: dict[str, Any],
        tool_handlers: Mapping[str, Callable[..., Any]],
        *,
        deadline: float | None,
        turn_request_id: str | None,
        handled_tool_calls: dict[str, tuple[str, OrchestratorToolResponse]],
    ) -> OrchestratorToolResponse:
        """Execute one remote tool request through the local handler map."""

        prepared = self._prepare_tool_request(payload, handled_tool_calls=handled_tool_calls)
        if prepared.cached_response is not None:
            return prepared.cached_response

        request_event = prepared.request_event
        if request_event is None:
            raise OrchestratorProtocolError("Prepared tool request unexpectedly missing request_event")

        handler = tool_handlers.get(request_event.name)
        if handler is None:
            response = OrchestratorToolResponse(
                call_id=request_event.call_id,
                ok=False,
                error=f"{self._UNSUPPORTED_TOOL_REQUEST}: {self._sanitize_text(request_event.name, limit=128) or '<unknown>'}",
            )
            handled_tool_calls[request_event.call_id] = (prepared.signature, response)
            return response

        if not isinstance(request_event.arguments, Mapping):
            response = OrchestratorToolResponse(
                call_id=request_event.call_id,
                ok=False,
                error=self._INVALID_TOOL_ARGUMENTS,
            )
            handled_tool_calls[request_event.call_id] = (prepared.signature, response)
            return response

        timeout_context = self._compute_tool_timeout_context(deadline)
        context_kwargs = self._build_tool_context_kwargs(
            request_event,
            timeout_seconds=timeout_context.timeout_seconds,
            deadline=deadline,
            turn_request_id=turn_request_id,
        )

        try:
            output = await self._run_tool_handler_async(
                handler,
                dict(request_event.arguments),
                context_kwargs=context_kwargs,
                timeout_seconds=timeout_context.timeout_seconds,
            )
        except TimeoutError as exc:
            if timeout_context.timeout_scope == "turn":
                raise TimeoutError(self._TURN_TIMEOUT) from exc
            response = OrchestratorToolResponse(
                call_id=request_event.call_id,
                ok=False,
                error=self._TOOL_TIMEOUT,
            )
            handled_tool_calls[request_event.call_id] = (prepared.signature, response)
            return response
        except Exception:
            response = OrchestratorToolResponse(
                call_id=request_event.call_id,
                ok=False,
                error=self._GENERIC_TOOL_FAILURE,
            )
            handled_tool_calls[request_event.call_id] = (prepared.signature, response)
            return response

        response = self._build_tool_success_response(request_event.call_id, output)
        handled_tool_calls[request_event.call_id] = (prepared.signature, response)
        return response

    def _handle_tool_request_sync(
        self,
        payload: dict[str, Any],
        tool_handlers: Mapping[str, Callable[..., Any]],
        *,
        deadline: float | None,
        turn_request_id: str | None,
        handled_tool_calls: dict[str, tuple[str, OrchestratorToolResponse]],
    ) -> OrchestratorToolResponse:
        """Execute one remote tool request through the local handler map."""

        prepared = self._prepare_tool_request(payload, handled_tool_calls=handled_tool_calls)
        if prepared.cached_response is not None:
            return prepared.cached_response

        request_event = prepared.request_event
        if request_event is None:
            raise OrchestratorProtocolError("Prepared tool request unexpectedly missing request_event")

        handler = tool_handlers.get(request_event.name)
        if handler is None:
            response = OrchestratorToolResponse(
                call_id=request_event.call_id,
                ok=False,
                error=f"{self._UNSUPPORTED_TOOL_REQUEST}: {self._sanitize_text(request_event.name, limit=128) or '<unknown>'}",
            )
            handled_tool_calls[request_event.call_id] = (prepared.signature, response)
            return response

        if not isinstance(request_event.arguments, Mapping):
            response = OrchestratorToolResponse(
                call_id=request_event.call_id,
                ok=False,
                error=self._INVALID_TOOL_ARGUMENTS,
            )
            handled_tool_calls[request_event.call_id] = (prepared.signature, response)
            return response

        timeout_context = self._compute_tool_timeout_context(deadline)
        context_kwargs = self._build_tool_context_kwargs(
            request_event,
            timeout_seconds=timeout_context.timeout_seconds,
            deadline=deadline,
            turn_request_id=turn_request_id,
        )

        try:
            output = self._run_tool_handler_sync(
                handler,
                dict(request_event.arguments),
                context_kwargs=context_kwargs,
                timeout_seconds=timeout_context.timeout_seconds,
            )
        except TimeoutError as exc:
            if timeout_context.timeout_scope == "turn":
                raise TimeoutError(self._TURN_TIMEOUT) from exc
            response = OrchestratorToolResponse(
                call_id=request_event.call_id,
                ok=False,
                error=self._TOOL_TIMEOUT,
            )
            handled_tool_calls[request_event.call_id] = (prepared.signature, response)
            return response
        except Exception:
            response = OrchestratorToolResponse(
                call_id=request_event.call_id,
                ok=False,
                error=self._GENERIC_TOOL_FAILURE,
            )
            handled_tool_calls[request_event.call_id] = (prepared.signature, response)
            return response

        response = self._build_tool_success_response(request_event.call_id, output)
        handled_tool_calls[request_event.call_id] = (prepared.signature, response)
        return response

    def _build_tool_success_response(self, call_id: str, output: Any) -> OrchestratorToolResponse:
        """Validate tool output and build a success response."""

        if not isinstance(output, Mapping):
            return OrchestratorToolResponse(
                call_id=call_id,
                ok=False,
                error=self._INVALID_TOOL_OUTPUT,
            )

        safe_output = _json_safe(dict(output))
        if not isinstance(safe_output, Mapping):
            return OrchestratorToolResponse(
                call_id=call_id,
                ok=False,
                error=self._INVALID_TOOL_OUTPUT,
            )
        output_payload = dict(safe_output)
        output_error = self._validate_tool_output_payload(output_payload)
        if output_error is not None:
            return OrchestratorToolResponse(
                call_id=call_id,
                ok=False,
                error=output_error,
            )

        return OrchestratorToolResponse(
            call_id=call_id,
            ok=True,
            output=output_payload,
        )

    def _prepare_tool_request(
        self,
        payload: dict[str, Any],
        *,
        handled_tool_calls: dict[str, tuple[str, OrchestratorToolResponse]],
    ) -> _PreparedToolRequest:
        """Parse, validate, and deduplicate one incoming tool request."""

        raw_call_id = self._sanitize_text(payload.get("call_id", ""), limit=256)
        try:
            request_event = OrchestratorToolRequest.from_payload(payload)
        except Exception as exc:
            if raw_call_id:
                return _PreparedToolRequest(
                    request_event=None,
                    signature="",
                    cached_response=OrchestratorToolResponse(
                        call_id=raw_call_id,
                        ok=False,
                        error=self._MALFORMED_TOOL_REQUEST,
                    ),
                )
            raise OrchestratorProtocolError("Received malformed tool_request payload from orchestrator") from exc

        signature = self._tool_request_signature(request_event)
        cached = handled_tool_calls.get(request_event.call_id)
        if cached is not None:
            cached_signature, cached_response = cached
            if cached_signature == signature:
                return _PreparedToolRequest(
                    request_event=request_event,
                    signature=signature,
                    cached_response=cached_response,
                )
            return _PreparedToolRequest(
                request_event=request_event,
                signature=signature,
                cached_response=OrchestratorToolResponse(
                    call_id=request_event.call_id,
                    ok=False,
                    error=self._CONFLICTING_DUPLICATE_TOOL_REQUEST,
                ),
            )

        if len(handled_tool_calls) >= self.max_tool_requests_per_turn:
            raise OrchestratorBudgetExceeded(self._TOOL_REQUEST_BUDGET_EXCEEDED)

        return _PreparedToolRequest(
            request_event=request_event,
            signature=signature,
        )

    async def _run_tool_handler_async(
        self,
        handler: Callable[..., Any],
        arguments: dict[str, Any],
        *,
        context_kwargs: dict[str, Any],
        timeout_seconds: float | None,
    ) -> Any:
        """Run a tool handler inside the asyncio turn loop."""

        if self._is_async_callable(handler):
            coro = self._invoke_async_handler(handler, arguments, context_kwargs=context_kwargs)
            if timeout_seconds is None:
                return await coro
            try:
                return await asyncio.wait_for(coro, timeout=timeout_seconds)
            except asyncio.TimeoutError as exc:
                raise TimeoutError(self._TOOL_TIMEOUT) from exc

        worker = asyncio.to_thread(self._invoke_handler_blocking, handler, arguments, context_kwargs)
        if timeout_seconds is None:
            return await worker
        try:
            return await asyncio.wait_for(worker, timeout=timeout_seconds)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(self._TOOL_TIMEOUT) from exc

    async def _invoke_async_handler(
        self,
        handler: Callable[..., Any],
        arguments: dict[str, Any],
        *,
        context_kwargs: dict[str, Any],
    ) -> Any:
        """Invoke an async-capable handler directly on the running event loop."""

        filtered_kwargs = self._filter_supported_kwargs(handler, context_kwargs)
        result = handler(arguments, **filtered_kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    def _run_tool_handler_sync(
        self,
        handler: Callable[..., Any],
        arguments: dict[str, Any],
        *,
        context_kwargs: dict[str, Any],
        timeout_seconds: float | None,
    ) -> Any:
        """Run a tool handler in blocking mode with an optional timeout."""

        return self._call_in_daemon_thread(
            lambda: self._invoke_handler_blocking(handler, arguments, context_kwargs),
            timeout_seconds=timeout_seconds,
            thread_name=f"twinr-tool-{self._sanitize_text(context_kwargs.get('tool_name', 'worker'), limit=32) or 'worker'}",
        )

    def _invoke_handler_blocking(
        self,
        handler: Callable[..., Any],
        arguments: dict[str, Any],
        context_kwargs: dict[str, Any],
    ) -> Any:
        """Invoke a handler in a regular thread, resolving awaitables when needed."""

        filtered_kwargs = self._filter_supported_kwargs(handler, context_kwargs)
        result = handler(arguments, **filtered_kwargs)
        if inspect.isawaitable(result):
            return asyncio.run(result)
        return result

    @staticmethod
    def _call_in_daemon_thread(
        func: Callable[[], Any],
        *,
        timeout_seconds: float | None,
        thread_name: str,
    ) -> Any:
        """Run a blocking callable in a daemon thread so timeouts can't freeze the turn loop."""

        box: dict[str, Any] = {}
        done = threading.Event()

        def _runner() -> None:
            try:
                box["value"] = func()
            except BaseException as exc:  # pragma: no cover - exercised indirectly via callers.
                box["error"] = exc
            finally:
                done.set()

        thread = threading.Thread(target=_runner, name=thread_name, daemon=True)
        thread.start()
        finished = done.wait(timeout_seconds)
        if not finished:
            raise TimeoutError("Timed out while executing local tool handler")
        if "error" in box:
            raise box["error"]
        return box.get("value")

    def _build_tool_context_kwargs(
        self,
        request_event: OrchestratorToolRequest,
        *,
        timeout_seconds: float | None,
        deadline: float | None,
        turn_request_id: str | None,
    ) -> dict[str, Any]:
        """Build optional context kwargs for advanced tool handlers."""

        kwargs: dict[str, Any] = {
            "call_id": request_event.call_id,
            "tool_name": request_event.name,
        }
        if turn_request_id:
            kwargs["turn_request_id"] = turn_request_id
        if deadline is not None:
            kwargs["deadline_monotonic"] = deadline
        if timeout_seconds is not None:
            kwargs["timeout_seconds"] = timeout_seconds
        return kwargs

    def _compute_tool_timeout_context(self, deadline: float | None) -> _ToolTimeoutContext:
        """Compute the effective timeout for one tool execution."""

        remaining_turn_budget = self._remaining_deadline_seconds(deadline)
        if remaining_turn_budget is not None and remaining_turn_budget <= 0:
            raise TimeoutError(self._TURN_TIMEOUT)

        if self.tool_timeout_seconds is None:
            return _ToolTimeoutContext(
                timeout_seconds=remaining_turn_budget,
                timeout_scope="turn" if remaining_turn_budget is not None else None,
            )

        if remaining_turn_budget is None:
            return _ToolTimeoutContext(
                timeout_seconds=self.tool_timeout_seconds,
                timeout_scope="tool",
            )

        if remaining_turn_budget <= self.tool_timeout_seconds:
            return _ToolTimeoutContext(
                timeout_seconds=remaining_turn_budget,
                timeout_scope="turn",
            )

        return _ToolTimeoutContext(
            timeout_seconds=self.tool_timeout_seconds,
            timeout_scope="tool",
        )

    def _validate_tool_output_payload(self, output_payload: dict[str, Any]) -> str | None:
        """Return a safe remote-facing error if tool output can't be sent."""

        try:
            self._serialize_json(
                output_payload,
                context="tool output",
                max_bytes=self.max_outbound_message_bytes,
            )
        except ValueError:
            return self._NON_SERIALIZABLE_TOOL_OUTPUT
        except RuntimeError:
            return self._TOOL_OUTPUT_TOO_LARGE
        return None

    def _build_request_payload(self, request: OrchestratorTurnRequest) -> dict[str, Any]:
        """Build and validate the outbound turn request payload."""

        payload = request.to_payload()
        if not isinstance(payload, Mapping):
            raise TypeError("request.to_payload() must return a mapping")
        request_payload = dict(payload)
        try:
            self._serialize_json(
                request_payload,
                context="turn request",
                max_bytes=self.max_outbound_message_bytes,
            )
        except ValueError as exc:
            raise RuntimeError(self._NON_SERIALIZABLE_REQUEST) from exc
        return request_payload

    def _extract_expected_request_id(self, request_payload: Mapping[str, Any]) -> str:
        """Extract the request_id used to correlate the turn response."""

        return self._sanitize_text(request_payload.get("request_id", ""), limit=256)

    def _validate_completed_request_id(self, actual_request_id: Any, expected_request_id: str) -> None:
        """Ensure the completion event belongs to the request we sent."""

        if not expected_request_id:
            return
        actual = self._sanitize_text(actual_request_id, limit=256)
        if actual and actual != expected_request_id:
            raise OrchestratorProtocolError(
                f"Mismatched turn_complete request_id: expected {expected_request_id!r}, got {actual!r}"
            )

    def _parse_turn_complete(self, payload: dict[str, Any]) -> OrchestratorTurnCompleteEvent:
        """Parse the final turn_complete event."""

        try:
            return OrchestratorTurnCompleteEvent.from_payload(payload)
        except Exception as exc:
            raise OrchestratorProtocolError("Received malformed turn_complete payload from orchestrator") from exc

    def _build_ack_event(self, payload: dict[str, Any]) -> OrchestratorAckEvent:
        """Build a bounded ack event suitable for local storage/UI."""

        ack_id = self._sanitize_text(payload.get("ack_id", ""), limit=256)
        text = self._truncate_text_by_utf8_bytes(
            self._sanitize_transport_text(payload.get("text", ""), max_chars=self.max_ack_text_chars),
            self.max_ack_buffer_bytes,
        )
        return OrchestratorAckEvent(
            ack_id=ack_id,
            text=text,
        )

    def _append_ack_event(
        self,
        ack_records: deque[tuple[OrchestratorAckEvent, int]],
        ack_buffer_bytes: int,
        event: OrchestratorAckEvent,
    ) -> int:
        """Append an ack while respecting both count and byte budgets."""

        event_bytes = len(event.text.encode("utf-8"))
        while ack_records and (
            len(ack_records) >= self.max_ack_events
            or ack_buffer_bytes + event_bytes > self.max_ack_buffer_bytes
        ):
            _, old_size = ack_records.popleft()
            ack_buffer_bytes -= old_size

        if event_bytes > self.max_ack_buffer_bytes:
            truncated_text = self._truncate_text_by_utf8_bytes(event.text, self.max_ack_buffer_bytes)
            event = OrchestratorAckEvent(
                ack_id=event.ack_id,
                text=truncated_text,
            )
            event_bytes = len(event.text.encode("utf-8"))

        ack_records.append((event, event_bytes))
        return ack_buffer_bytes + event_bytes

    def _count_protocol_event(self, current_count: int) -> int:
        """Increment and enforce the per-turn protocol-event budget."""

        current_count += 1
        if current_count > self.max_protocol_events:
            raise OrchestratorBudgetExceeded(self._PROTOCOL_EVENT_BUDGET_EXCEEDED)
        return current_count

    def _count_streamed_text(self, current_bytes: int, delta: str) -> int:
        """Increment and enforce the per-turn streamed-text budget."""

        current_bytes += len(delta.encode("utf-8"))
        if current_bytes > self.max_streamed_text_bytes:
            raise OrchestratorBudgetExceeded(self._STREAMED_TEXT_BUDGET_EXCEEDED)
        return current_bytes

    def _tool_request_signature(self, request_event: OrchestratorToolRequest) -> str:
        """Build a stable signature for duplicate tool_request detection."""

        return self._serialize_json(
            {
                "name": request_event.name,
                "arguments": dict(request_event.arguments) if isinstance(request_event.arguments, Mapping) else request_event.arguments,
            },
            context="tool request signature",
            max_bytes=self.max_message_bytes,
        )

    def _build_headers(self) -> dict[str, str] | None:
        """Build the authenticated websocket request headers."""

        if self.shared_secret is None:
            return None
        return {"x-twinr-secret": self.shared_secret}

    def _build_connector_kwargs(
        self,
        connector: Callable[..., Any],
        *,
        headers: dict[str, str] | None,
    ) -> dict[str, Any]:
        """Build the subset of websocket connector kwargs this client uses."""

        candidate_kwargs: dict[str, Any] = {
            "additional_headers": headers,
            "open_timeout": self.open_timeout_seconds,
            "close_timeout": self.close_timeout_seconds,
            "ping_interval": self.ping_interval_seconds,
            "ping_timeout": self.ping_timeout_seconds,
            "max_size": self.max_message_bytes,
            "max_queue": self.max_queue,
            "compression": self.compression,
            "user_agent_header": self.user_agent_header,
            "proxy": self.proxy,
            "ssl": self.ssl_context,
            "server_hostname": self.server_hostname,
            "origin": self.origin,
            "subprotocols": self.subprotocols,
        }
        return self._filter_supported_kwargs(connector, candidate_kwargs)

    @staticmethod
    def _filter_supported_kwargs(func: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
        """Drop connector kwargs that the injected callable does not accept."""

        explicit_none_kwargs = {"compression", "proxy", "user_agent_header"}

        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):
            return {
                name: value
                for name, value in kwargs.items()
                if value is not None or name in explicit_none_kwargs
            }

        parameters = signature.parameters.values()
        if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
            return {
                name: value
                for name, value in kwargs.items()
                if value is not None or name in explicit_none_kwargs
            }

        supported_names = {
            name
            for name, parameter in signature.parameters.items()
            if parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        filtered_kwargs = dict(kwargs)
        if "additional_headers" in filtered_kwargs and "additional_headers" not in supported_names and "extra_headers" in supported_names:
            filtered_kwargs["extra_headers"] = filtered_kwargs["additional_headers"]
            filtered_kwargs.pop("additional_headers", None)
        if "ssl" in filtered_kwargs and "ssl" not in supported_names and "ssl_context" in supported_names:
            filtered_kwargs["ssl_context"] = filtered_kwargs["ssl"]
            filtered_kwargs.pop("ssl", None)
        return {
            name: value
            for name, value in filtered_kwargs.items()
            if name in supported_names and (value is not None or name in explicit_none_kwargs)
        }

    async def _async_recv_payload(self, websocket: Any, *, deadline: float | None) -> dict[str, Any]:
        """Receive one JSON object payload from the websocket transport."""

        timeout = self._compute_recv_timeout(deadline)
        try:
            raw_message = await asyncio.wait_for(websocket.recv(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(self._TURN_TIMEOUT if deadline is not None and self._remaining_deadline_seconds(deadline) == 0 else "Timed out while waiting for orchestrator websocket traffic") from exc
        return self._decode_payload(raw_message)

    def _recv_payload(self, websocket: Any, *, deadline: float | None) -> dict[str, Any]:
        """Receive one JSON object payload from the websocket transport."""

        raw_message = websocket.recv(timeout=self._compute_recv_timeout(deadline))
        return self._decode_payload(raw_message)

    @staticmethod
    async def _maybe_await_callback(result: Any) -> None:
        """Await a callback result if it is awaitable."""

        if inspect.isawaitable(result):
            await result

    async def _emit_transport_event_async(
        self,
        callback: _TransportEventCallback | None,
        event: str,
        **payload: Any,
    ) -> None:
        """Emit one bounded transport event for probe/acceptance instrumentation."""

        if callback is None:
            return
        safe_payload = _json_safe(payload)
        details = dict(safe_payload) if isinstance(safe_payload, Mapping) else {}
        await self._maybe_await_callback(callback(event, details))

    @staticmethod
    def _await_callback_blocking(result: Any) -> None:
        """Resolve an awaitable callback result when running in blocking mode."""

        if inspect.isawaitable(result):
            asyncio.run(result)

    def _emit_transport_event_blocking(
        self,
        callback: _TransportEventCallback | None,
        event: str,
        **payload: Any,
    ) -> None:
        """Emit one bounded transport event when the sync client is active."""

        if callback is None:
            return
        safe_payload = _json_safe(payload)
        details = dict(safe_payload) if isinstance(safe_payload, Mapping) else {}
        self._await_callback_blocking(callback(event, details))

    def _decode_payload(self, raw_message: Any) -> dict[str, Any]:
        """Decode and validate one JSON protocol message."""

        if isinstance(raw_message, bytes):
            try:
                raw_text = raw_message.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise OrchestratorProtocolError("Received non-UTF-8 payload from orchestrator") from exc
        else:
            raw_text = str(raw_message)

        try:
            payload = json.loads(raw_text, parse_constant=self._reject_invalid_json_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            raise OrchestratorProtocolError("Received malformed JSON from orchestrator") from exc

        if not isinstance(payload, dict):
            raise OrchestratorProtocolError("Received non-object message from orchestrator")
        return payload

    async def _async_send_json(self, websocket: Any, payload: Any, *, context: str) -> None:
        """Serialize and send one JSON payload over the websocket."""

        message = self._serialize_json(payload, context=context, max_bytes=self.max_outbound_message_bytes)
        await websocket.send(message)

    def _send_json(self, websocket: Any, payload: Any, *, context: str) -> None:
        """Serialize and send one JSON payload over the websocket."""

        message = self._serialize_json(payload, context=context, max_bytes=self.max_outbound_message_bytes)
        websocket.send(message)

    def _serialize_json(self, payload: Any, *, context: str, max_bytes: int) -> str:
        """Serialize one payload as strict JSON and enforce an outbound size budget."""

        try:
            message = json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Failed to serialize {context}") from exc

        encoded = message.encode("utf-8")
        if len(encoded) > max_bytes:
            raise RuntimeError(f"Serialized {context} exceeds size limit of {max_bytes} bytes")
        return message

    def _compute_deadline(self) -> float | None:
        """Compute the absolute deadline for the current turn, if bounded."""

        if self.turn_timeout_seconds is None:
            return None
        return monotonic() + self.turn_timeout_seconds

    def _compute_recv_timeout(self, deadline: float | None) -> float:
        """Compute the receive timeout for the next websocket recv()."""

        timeout = self.recv_timeout_seconds
        remaining = self._remaining_deadline_seconds(deadline)
        if remaining is None:
            return timeout
        if remaining <= 0:
            raise TimeoutError(self._TURN_TIMEOUT)
        return min(timeout, remaining)

    @staticmethod
    def _remaining_deadline_seconds(deadline: float | None) -> float | None:
        """Return the remaining turn budget in seconds."""

        if deadline is None:
            return None
        return max(0.0, deadline - monotonic())

    @staticmethod
    def _snapshot_tool_handlers(
        tool_handlers: Mapping[str, Callable[..., Any]],
    ) -> dict[str, Callable[..., Any]]:
        """Copy and validate the per-turn local tool handler mapping."""

        handlers = dict(tool_handlers)
        for name, handler in handlers.items():
            if not callable(handler):
                raise TypeError(f"tool_handlers[{name!r}] must be callable")
        return handlers

    @staticmethod
    def _sanitize_text(value: Any, *, limit: int) -> str:
        """Strip control characters and bound transport metadata length."""

        text = "" if value is None else str(value)
        cleaned = "".join(character if character.isprintable() else " " for character in text).strip()
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[: limit - 1].rstrip() + "…"

    @staticmethod
    def _sanitize_transport_text(value: Any, *, max_chars: int) -> str:
        """Normalize human-readable transport text without destroying newlines or tabs."""

        text = "" if value is None else str(value)
        cleaned = []
        for character in text:
            if character in ("\n", "\r", "\t") or character.isprintable():
                cleaned.append(character)
            else:
                cleaned.append(" ")
        normalized = "".join(cleaned).strip()
        if len(normalized) <= max_chars:
            return normalized
        return normalized[: max_chars - 1].rstrip() + "…"

    def _coerce_stream_text(self, value: Any) -> str:
        """Coerce streamed model text to a string while preserving human-readable formatting."""

        return self._sanitize_transport_text(value, max_chars=self.max_message_bytes)

    @staticmethod
    def _truncate_text_by_utf8_bytes(text: str, max_bytes: int) -> str:
        """Truncate text by encoded UTF-8 byte size without producing invalid UTF-8."""

        encoded = text.encode("utf-8")
        if len(encoded) <= max_bytes:
            return text
        if max_bytes <= 0:
            return ""
        ellipsis = "…"
        ellipsis_bytes = len(ellipsis.encode("utf-8"))
        if max_bytes <= ellipsis_bytes:
            return ellipsis[:1]

        budget = max_bytes - ellipsis_bytes
        pieces: list[str] = []
        used = 0
        for character in text:
            char_bytes = len(character.encode("utf-8"))
            if used + char_bytes > budget:
                break
            pieces.append(character)
            used += char_bytes
        return "".join(pieces).rstrip() + "…"

    @staticmethod
    def _reject_invalid_json_constant(value: str) -> Any:
        """Reject NaN/Infinity constants accepted by Python's permissive JSON parser."""

        raise ValueError(f"Invalid JSON constant: {value}")

    @staticmethod
    def _normalize_shared_secret(shared_secret: str | None) -> str | None:
        """Normalize the shared secret used for websocket authentication."""

        secret = (shared_secret or "").strip()
        if not secret:
            return None
        if any(character in secret for character in ("\r", "\n", "\x00")):
            raise ValueError("shared_secret contains invalid control characters")
        return secret

    @classmethod
    def _validate_url(cls, url: str, *, require_tls: bool) -> str:
        """Normalize and validate the orchestrator websocket URL."""

        normalized_url = url.strip()
        if not normalized_url:
            raise ValueError("url must not be empty")

        parsed = urlsplit(normalized_url)
        scheme = parsed.scheme.lower()
        host = parsed.hostname or ""
        if scheme not in {"ws", "wss"}:
            raise ValueError("url must use ws:// or wss://")
        if not host:
            raise ValueError("url must include a hostname")
        if require_tls and scheme != "wss" and not cls._is_loopback_host(host):
            raise ValueError("Non-loopback orchestrator URLs must use wss://")
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
        """Validate a required positive finite floating-point setting."""

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a finite number")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0:
            raise ValueError(f"{name} must be > 0 and finite")
        return numeric

    @classmethod
    def _validate_optional_positive_float(cls, value: float | None, name: str) -> float | None:
        """Validate an optional positive finite floating-point setting."""

        if value is None:
            return None
        return cls._validate_positive_float(value, name)

    @staticmethod
    def _validate_positive_int(value: int, name: str) -> int:
        """Validate a required positive integer setting."""

        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be > 0")
        return value

    @staticmethod
    def _normalize_proxy(proxy: str | bool | None) -> str | bool | None:
        """Normalize proxy configuration for websockets.connect()."""

        if proxy is False or proxy is None:
            return None
        if proxy is True:
            return True
        if not isinstance(proxy, str):
            raise TypeError("proxy must be None, a bool, or a string URL")
        normalized = proxy.strip()
        if not normalized:
            return None
        if any(character in normalized for character in ("\r", "\n", "\x00")):
            raise ValueError("proxy contains invalid control characters")
        return normalized

    @staticmethod
    def _normalize_compression(compression: str | None) -> str | None:
        """Normalize websocket compression settings."""

        if compression is None:
            return None
        normalized = compression.strip().lower()
        if not normalized:
            return None
        if normalized != "deflate":
            raise ValueError("compression must be None or 'deflate'")
        return normalized

    @staticmethod
    def _normalize_optional_header_value(value: str | None, name: str) -> str | None:
        """Normalize an optional HTTP header value."""

        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        if any(character in normalized for character in ("\r", "\n", "\x00")):
            raise ValueError(f"{name} contains invalid control characters")
        return normalized

    @staticmethod
    def _normalize_optional_host(value: str | None, name: str) -> str | None:
        """Normalize an optional host value."""

        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        if any(character in normalized for character in ("\r", "\n", "\x00", " ")):
            raise ValueError(f"{name} contains invalid characters")
        return normalized

    @staticmethod
    def _normalize_subprotocols(subprotocols: Sequence[str] | None) -> tuple[str, ...] | None:
        """Normalize websocket subprotocol negotiation settings."""

        if subprotocols is None:
            return None
        normalized: list[str] = []
        for index, value in enumerate(subprotocols):
            if not isinstance(value, str):
                raise TypeError(f"subprotocols[{index}] must be a string")
            token = value.strip()
            if not token:
                raise ValueError(f"subprotocols[{index}] must not be empty")
            if any(character in token for character in ("\r", "\n", "\x00", ",")):
                raise ValueError(f"subprotocols[{index}] contains invalid characters")
            normalized.append(token)
        return tuple(normalized)

    @staticmethod
    def _validate_ssl_context(value: ssl.SSLContext | None) -> ssl.SSLContext | None:
        """Validate an optional SSL context."""

        if value is None:
            return None
        if not isinstance(value, ssl.SSLContext):
            raise TypeError("ssl_context must be an ssl.SSLContext instance")
        return value

    @staticmethod
    def _is_async_callable(func: Callable[..., Any]) -> bool:
        """Return whether func is an async callable or async __call__ object."""

        if inspect.iscoroutinefunction(func):
            return True
        call = getattr(func, "__call__", None)
        return inspect.iscoroutinefunction(call)
