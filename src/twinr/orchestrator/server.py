# CHANGELOG: 2026-03-29
# BUG-1: Track active run_turn workers and defer session teardown until the worker stops, fixing disconnect/shutdown races and orphaned turn execution.
# BUG-2: Add bounded ingress/egress pressure control so blocked clients and bursty voice frames no longer create unbounded latency or memory growth.
# SEC-1: Reject oversized frames, enforce per-connection rate limits, cap concurrent websockets, and validate browser Origin in insecure-loopback mode.
# IMP-1: Switch websocket ingress to raw-frame parsing with explicit denial responses, idle/size/rate guardrails, and sanitized protocol errors.
# IMP-2: Upgrade the voice path to a bounded real-time pipeline with stale-audio shedding, plus richer lifecycle cleanup and cancellation hooks.
# BREAKING: Default concurrent websocket connections are now capped (8 total) to protect Raspberry Pi deployments; set ORCHESTRATOR_MAX_CONNECTIONS=0 to restore unlimited behavior.
# BREAKING: Oversized websocket frames are now rejected by default (4 MiB orchestrator / 1 MiB voice); tune with ORCHESTRATOR_MAX_MESSAGE_BYTES and ORCHESTRATOR_VOICE_MAX_MESSAGE_BYTES.

"""Expose the FastAPI websocket server for Twinr's edge orchestrator.

This module accepts websocket turn requests, bridges remote tool results into a
session object, and keeps transport auth, queueing, and teardown bounded.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import ipaddress
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock
from typing import Any, Callable, Iterable
from urllib.parse import urlparse

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.responses import PlainTextResponse

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.workflows.forensics import WorkflowForensics
from twinr.orchestrator.contracts import (
    OrchestratorErrorEvent,
    OrchestratorToolResponse,
    OrchestratorTurnRequest,
)
from twinr.orchestrator.remote_asr_service import (
    RemoteAsrHttpService,
    remote_asr_url_targets_local_orchestrator,
)
from twinr.orchestrator.session import EdgeOrchestratorSession, RemoteToolBridge
from twinr.orchestrator.voice_contracts import (
    OrchestratorVoiceAudioFrame,
    OrchestratorVoiceErrorEvent,
    OrchestratorVoiceHelloRequest,
    OrchestratorVoiceIdentityProfilesEvent,
    OrchestratorVoiceRuntimeStateEvent,
)
from twinr.orchestrator.voice_gateway_guardrails import (
    assert_transcript_first_voice_gateway_contract,
)
from twinr.orchestrator.voice_session import EdgeOrchestratorVoiceSession


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SocketPolicy:
    """Runtime limits for one websocket class."""

    outgoing_queue_maxsize: int
    max_message_bytes: int
    idle_timeout_seconds: float | None
    message_rate_per_second: float
    message_burst: int


@dataclass(frozen=True)
class _VoicePolicy(_SocketPolicy):
    """Runtime limits for the voice websocket path."""

    ingress_queue_maxsize: int


class _TurnGate:
    """Serialize turn execution so one websocket has at most one live turn."""

    def __init__(self) -> None:
        self._active_task: asyncio.Task[Any] | None = None

    def try_start(self, task: asyncio.Task[Any]) -> bool:
        """Register one task as the active turn if no turn is currently running."""

        if self.is_active():
            return False
        self._active_task = task
        return True

    def finish(self, task: asyncio.Task[Any] | None) -> None:
        """Release the active turn if it matches the completed task."""

        if task is not None and self._active_task is task:
            self._active_task = None

    def is_active(self) -> bool:
        """Return whether a turn task is still running."""

        return self._active_task is not None and not self._active_task.done()

    def active_task(self) -> asyncio.Task[Any] | None:
        """Return the currently active task, if any."""

        return self._active_task


class _ConnectionGate:
    """Bound the number of concurrently open websocket sessions."""

    def __init__(self, *, max_connections: int) -> None:
        self._max_connections = max_connections
        self._active_connections = 0
        self._lock = Lock()

    async def try_acquire(self) -> bool:
        """Reserve one connection slot if capacity remains."""

        with self._lock:
            if self._max_connections > 0 and self._active_connections >= self._max_connections:
                return False
            self._active_connections += 1
            return True

    async def release(self) -> None:
        """Return one connection slot to the pool."""

        with self._lock:
            if self._active_connections > 0:
                self._active_connections -= 1


class _TokenBucket:
    """Simple per-connection token bucket for websocket ingress."""

    def __init__(self, *, rate_per_second: float, burst: int) -> None:
        self._rate_per_second = max(0.0, float(rate_per_second))
        self._capacity = max(0, int(burst))
        self._tokens = float(self._capacity)
        self._updated_at = time.monotonic()

    def allow(self, *, tokens: float = 1.0) -> bool:
        """Consume tokens if capacity permits, or allow everything when disabled."""

        if self._rate_per_second <= 0.0 or self._capacity <= 0:
            return True

        now = time.monotonic()
        elapsed = max(0.0, now - self._updated_at)
        self._updated_at = now
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate_per_second)
        if self._tokens < tokens:
            return False
        self._tokens -= tokens
        return True


class _ProtocolMessageError(Exception):
    """Raised when a websocket message is malformed or violates local policy."""

    def __init__(self, user_message: str, *, close_code: int | None = None, close_reason: str | None = None) -> None:
        super().__init__(user_message)
        self.user_message = user_message
        self.close_code = close_code
        self.close_reason = close_reason or user_message


class _IdleTimeoutExceeded(Exception):
    """Raised when a websocket stays idle beyond the configured timeout."""


class _VoiceIngressClosed(Exception):
    """Raised when the bounded voice ingress queue has been closed."""


class _VoiceIngressQueue:
    """Bounded FIFO queue for voice messages with stale-audio shedding."""

    def __init__(self, *, maxsize: int) -> None:
        self._maxsize = max(1, maxsize)
        self._items: deque[dict[str, Any]] = deque()
        self._closed = False
        self._condition = asyncio.Condition()

    async def put(self, payload: dict[str, Any]) -> bool:
        """Insert one message, shedding the oldest audio frame when the queue is full."""

        async with self._condition:
            if self._closed:
                return False

            if len(self._items) >= self._maxsize:
                if str(payload.get("type", "") or "") == "voice_audio_frame":
                    if not self._drop_oldest_audio_frame_locked():
                        return False
                    logger.warning("Shedding stale voice_audio_frame due to voice ingress backpressure")
                else:
                    return False

            self._items.append(payload)
            self._condition.notify()
            return True

    async def get(self) -> dict[str, Any]:
        """Remove and return the oldest queued message."""

        async with self._condition:
            while not self._items:
                if self._closed:
                    raise _VoiceIngressClosed()
                await self._condition.wait()
            return self._items.popleft()

    async def close(self) -> None:
        """Wake blocked consumers and reject future producers."""

        async with self._condition:
            self._closed = True
            self._condition.notify_all()

    def _drop_oldest_audio_frame_locked(self) -> bool:
        """Drop the oldest queued audio frame while preserving control messages."""

        for index, item in enumerate(self._items):
            if str(item.get("type", "") or "") == "voice_audio_frame":
                del self._items[index]
                return True
        return False


class EdgeOrchestratorServer:
    """Serve Twinr orchestrator turns over websocket endpoints."""

    def __init__(
        self,
        config: TwinrConfig,
        *,
        session_factory: Callable[[TwinrConfig], EdgeOrchestratorSession] | None = None,
        voice_session_factory: Callable[[TwinrConfig], EdgeOrchestratorVoiceSession] | None = None,
    ) -> None:
        self.config = config
        self._session_factory = session_factory or EdgeOrchestratorSession
        self._voice_session_factory = voice_session_factory or EdgeOrchestratorVoiceSession
        self._voice_forensics = WorkflowForensics.from_env(
            project_root=Path(self.config.project_root),
            service="EdgeOrchestratorVoiceServer",
        )
        self._remote_asr_service = (
            RemoteAsrHttpService(self.config, forensics=self._voice_forensics)
            if remote_asr_url_targets_local_orchestrator(self.config)
            else None
        )
        self._orchestrator_policy = _SocketPolicy(
            outgoing_queue_maxsize=_get_outgoing_queue_maxsize(self.config),
            max_message_bytes=_get_orchestrator_max_message_bytes(self.config),
            idle_timeout_seconds=_get_orchestrator_idle_timeout_seconds(self.config),
            message_rate_per_second=_get_orchestrator_message_rate_per_second(self.config),
            message_burst=_get_orchestrator_message_burst(self.config),
        )
        self._voice_policy = _VoicePolicy(
            outgoing_queue_maxsize=_get_outgoing_queue_maxsize(self.config),
            max_message_bytes=_get_voice_max_message_bytes(self.config),
            idle_timeout_seconds=_get_voice_idle_timeout_seconds(self.config),
            message_rate_per_second=_get_voice_message_rate_per_second(self.config),
            message_burst=_get_voice_message_burst(self.config),
            ingress_queue_maxsize=_get_voice_ingress_queue_maxsize(self.config),
        )
        self._turn_shutdown_grace_seconds = _get_turn_shutdown_grace_seconds(self.config)
        self._max_connections = _get_max_connections(self.config)
        self._allowed_origins = _get_allowed_origins(self.config)
        self._connection_gate = _ConnectionGate(max_connections=self._max_connections)

    def create_app(self) -> FastAPI:
        """Build the FastAPI application that exposes the orchestrator sockets."""

        server = self

        @contextlib.asynccontextmanager
        async def _app_lifespan(_app: FastAPI):
            try:
                yield
            finally:
                await _best_effort_close(server._remote_asr_service, label="remote ASR service")
                await _best_effort_close(server._voice_forensics, label="voice orchestrator forensics")

        app = FastAPI(title="Twinr Orchestrator", version="0.2.0", lifespan=_app_lifespan)
        if server._remote_asr_service is not None:
            app.include_router(server._remote_asr_service.build_router())

        @app.websocket("/ws/orchestrator")
        async def orchestrator_socket(websocket: WebSocket) -> None:
            if not await server._connection_gate.try_acquire():
                logger.warning("Rejected orchestrator websocket from %s because connection capacity is exhausted", _client_host(websocket))
                await _deny_websocket(websocket, status_code=503, detail="orchestrator_busy")
                return

            await _serve_orchestrator_socket(server, websocket)

        @app.websocket("/ws/orchestrator/voice")
        async def orchestrator_voice_socket(websocket: WebSocket) -> None:
            if not await server._connection_gate.try_acquire():
                logger.warning("Rejected voice websocket from %s because connection capacity is exhausted", _client_host(websocket))
                await _deny_websocket(websocket, status_code=503, detail="voice_orchestrator_busy")
                return

            await _serve_voice_socket(server, websocket)

        return app

    def _authorize(self, websocket: WebSocket) -> bool:
        """Authorize one websocket based on shared-secret or loopback policy."""

        expected = str(getattr(self.config, "orchestrator_shared_secret", "") or "").strip()
        actual = (websocket.headers.get("x-twinr-secret") or "").strip()
        if expected:
            return hmac.compare_digest(actual, expected)

        return _allow_insecure_loopback(self.config) and _is_loopback_host(_client_host(websocket))

    def _origin_allowed(self, websocket: WebSocket) -> bool:
        """Validate browser Origin where it materially improves security."""

        origin = (websocket.headers.get("origin") or "").strip()
        if not origin:
            return True

        normalized = _normalize_origin(origin)
        if normalized is None:
            return False

        if self._allowed_origins:
            return normalized in self._allowed_origins

        # In insecure loopback mode, only accept browser origins that are themselves loopback.
        if _allow_insecure_loopback(self.config):
            return _is_loopback_origin(normalized)

        # When shared-secret auth is enabled, browser JS typically cannot send the required custom
        # header anyway, so we preserve compatibility with native/non-browser clients by only
        # enforcing an explicit allowlist.
        return True


async def _serve_orchestrator_socket(server: EdgeOrchestratorServer, websocket: WebSocket) -> None:
    """Serve one non-voice orchestrator websocket connection."""

    try:
        if not server._authorize(websocket):
            logger.warning("Rejected unauthorized orchestrator websocket from %s", _client_host(websocket))
            await _deny_websocket(websocket, status_code=401, detail="unauthorized")
            return

        if not server._origin_allowed(websocket):
            logger.warning(
                "Rejected orchestrator websocket from %s due to disallowed Origin %r",
                _client_host(websocket),
                websocket.headers.get("origin"),
            )
            await _deny_websocket(websocket, status_code=403, detail="disallowed_origin")
            return

        await websocket.accept()
        loop = asyncio.get_running_loop()
        closed_event = Event()
        outgoing: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=server._orchestrator_policy.outgoing_queue_maxsize
        )
        rate_limiter = _TokenBucket(
            rate_per_second=server._orchestrator_policy.message_rate_per_second,
            burst=server._orchestrator_policy.message_burst,
        )
        turn_gate = _TurnGate()

        def emit_event(payload: dict[str, Any]) -> None:
            _enqueue_payload(loop, outgoing, payload, closed_event, websocket)

        sender: asyncio.Task[Any] | None = None
        turn_task: asyncio.Task[Any] | None = None
        session: EdgeOrchestratorSession | None = None
        tool_bridge: RemoteToolBridge | None = None

        try:
            try:
                tool_bridge = RemoteToolBridge(emit_event=emit_event)
                session = server._session_factory(server.config)
            except Exception:
                logger.exception("Failed to initialize orchestrator websocket session")
                with contextlib.suppress(RuntimeError, WebSocketDisconnect):
                    await websocket.send_json(_error_payload("The orchestrator could not start. Please reconnect."))
                await _close_websocket_quietly(websocket, code=1011, reason="startup_failed")
                return

            sender = asyncio.create_task(_sender_loop(websocket, outgoing, closed_event))

            while not closed_event.is_set():
                try:
                    payload = await _receive_json_payload(
                        websocket,
                        max_message_bytes=server._orchestrator_policy.max_message_bytes,
                        idle_timeout_seconds=server._orchestrator_policy.idle_timeout_seconds,
                    )
                except WebSocketDisconnect:
                    break
                except _IdleTimeoutExceeded:
                    logger.info("Closing idle orchestrator websocket from %s", _client_host(websocket))
                    await _close_websocket_quietly(websocket, code=1001, reason="idle_timeout")
                    break
                except _ProtocolMessageError as exc:
                    logger.warning(
                        "Rejected orchestrator websocket message from %s: %s",
                        _client_host(websocket),
                        exc.user_message,
                    )
                    emit_event(_error_payload(exc.user_message))
                    if exc.close_code is not None:
                        await _close_websocket_quietly(websocket, code=exc.close_code, reason=exc.close_reason)
                        break
                    continue

                if not rate_limiter.allow():
                    logger.warning("Rate-limited orchestrator websocket from %s", _client_host(websocket))
                    emit_event(_error_payload("Too many websocket messages."))
                    await _close_websocket_quietly(websocket, code=1013, reason="rate_limited")
                    break

                message_type = str(payload.get("type", "") or "")
                if message_type == "run_turn":
                    try:
                        request = OrchestratorTurnRequest.from_payload(payload)
                    except Exception:
                        logger.exception("Invalid run_turn payload received")
                        emit_event(_error_payload("Invalid run_turn payload."))
                        continue

                    if turn_gate.is_active():
                        emit_event(_error_payload("A turn is already in progress."))
                        continue

                    turn_task = asyncio.create_task(
                        _run_turn(
                            session=session,
                            request=request,
                            emit_event=emit_event,
                            tool_bridge=tool_bridge,
                            closed_event=closed_event,
                            turn_gate=turn_gate,
                        ),
                        name="twinr-orchestrator-turn",
                    )
                    if not turn_gate.try_start(turn_task):
                        turn_task.cancel()
                        emit_event(_error_payload("A turn is already in progress."))
                    continue

                if message_type == "tool_result":
                    if not turn_gate.is_active():
                        emit_event(_error_payload("No active turn is waiting for a tool result."))
                        continue

                    try:
                        response = OrchestratorToolResponse.from_payload(payload)
                    except Exception:
                        logger.exception("Invalid tool_result payload received")
                        emit_event(_error_payload("Invalid tool_result payload."))
                        continue

                    try:
                        tool_bridge.submit_result(
                            response.call_id,
                            output=response.output,
                            error=response.error if not response.ok else None,
                        )
                    except Exception:
                        logger.exception("Failed to apply tool_result to remote bridge")
                        emit_event(_error_payload("The tool result could not be applied."))
                    continue

                emit_event(_error_payload("Unsupported message type."))

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Unhandled orchestrator websocket failure")
            closed_event.set()
            await _cancel_task(sender)
            sender = None
            with contextlib.suppress(RuntimeError, WebSocketDisconnect):
                await websocket.send_json(_error_payload("The connection failed. Please reconnect."))
        finally:
            closed_event.set()
            await _best_effort_close(tool_bridge, label="remote tool bridge")
            session = await _drain_or_defer_turn_shutdown(
                turn_gate=turn_gate,
                session=session,
                grace_seconds=server._turn_shutdown_grace_seconds,
            )
            await _best_effort_close(session, label="orchestrator session")
            await _cancel_task(sender)
    finally:
        await server._connection_gate.release()


async def _serve_voice_socket(server: EdgeOrchestratorServer, websocket: WebSocket) -> None:
    """Serve one voice orchestrator websocket connection."""

    try:
        if not server._authorize(websocket):
            logger.warning("Rejected unauthorized voice orchestrator websocket from %s", _client_host(websocket))
            await _deny_websocket(websocket, status_code=401, detail="unauthorized")
            return

        if not server._origin_allowed(websocket):
            logger.warning(
                "Rejected voice websocket from %s due to disallowed Origin %r",
                _client_host(websocket),
                websocket.headers.get("origin"),
            )
            await _deny_websocket(websocket, status_code=403, detail="disallowed_origin")
            return

        await websocket.accept()
        loop = asyncio.get_running_loop()
        closed_event = Event()
        outgoing: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=server._voice_policy.outgoing_queue_maxsize
        )
        voice_ingress = _VoiceIngressQueue(maxsize=server._voice_policy.ingress_queue_maxsize)
        rate_limiter = _TokenBucket(
            rate_per_second=server._voice_policy.message_rate_per_second,
            burst=server._voice_policy.message_burst,
        )

        def emit_event(payload: dict[str, Any]) -> None:
            _enqueue_payload(loop, outgoing, payload, closed_event, websocket)

        sender: asyncio.Task[Any] | None = None
        worker: asyncio.Task[Any] | None = None
        voice_session: EdgeOrchestratorVoiceSession | None = None

        try:
            try:
                voice_session = server._voice_session_factory(server.config)
                configure_forensics = getattr(voice_session, "set_forensics", None)
                if callable(configure_forensics):
                    configure_forensics(server._voice_forensics)
            except Exception:
                logger.exception("Failed to initialize voice orchestrator websocket session")
                with contextlib.suppress(RuntimeError, WebSocketDisconnect):
                    await websocket.send_json(
                        OrchestratorVoiceErrorEvent(
                            error="The voice orchestrator could not start. Please reconnect."
                        ).to_payload()
                    )
                await _close_websocket_quietly(websocket, code=1011, reason="startup_failed")
                return

            sender = asyncio.create_task(_sender_loop(websocket, outgoing, closed_event))
            worker = asyncio.create_task(
                _voice_worker(
                    voice_session=voice_session,
                    voice_ingress=voice_ingress,
                    emit_event=emit_event,
                    closed_event=closed_event,
                ),
                name="twinr-orchestrator-voice-worker",
            )

            while not closed_event.is_set():
                try:
                    payload = await _receive_json_payload(
                        websocket,
                        max_message_bytes=server._voice_policy.max_message_bytes,
                        idle_timeout_seconds=server._voice_policy.idle_timeout_seconds,
                    )
                except WebSocketDisconnect:
                    break
                except _IdleTimeoutExceeded:
                    logger.info("Closing idle voice websocket from %s", _client_host(websocket))
                    await _close_websocket_quietly(websocket, code=1001, reason="idle_timeout")
                    break
                except _ProtocolMessageError as exc:
                    logger.warning(
                        "Rejected voice websocket message from %s: %s",
                        _client_host(websocket),
                        exc.user_message,
                    )
                    emit_event(OrchestratorVoiceErrorEvent(error=exc.user_message).to_payload())
                    if exc.close_code is not None:
                        await _close_websocket_quietly(websocket, code=exc.close_code, reason=exc.close_reason)
                        break
                    continue

                if not rate_limiter.allow():
                    logger.warning("Rate-limited voice websocket from %s", _client_host(websocket))
                    emit_event(OrchestratorVoiceErrorEvent(error="Too many websocket messages.").to_payload())
                    await _close_websocket_quietly(websocket, code=1013, reason="rate_limited")
                    break

                accepted = await voice_ingress.put(payload)
                if not accepted:
                    logger.error("Voice ingress queue overflow for client %s; closing websocket", _client_host(websocket))
                    emit_event(
                        OrchestratorVoiceErrorEvent(
                            error="The voice session is overloaded. Please reconnect."
                        ).to_payload()
                    )
                    await _close_websocket_quietly(websocket, code=1013, reason="voice_backpressure")
                    break

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Unhandled voice orchestrator websocket failure")
            closed_event.set()
            await _cancel_task(sender)
            sender = None
            with contextlib.suppress(RuntimeError, WebSocketDisconnect):
                await websocket.send_json(
                    OrchestratorVoiceErrorEvent(
                        error="The voice connection failed. Please reconnect."
                    ).to_payload()
                )
        finally:
            closed_event.set()
            await voice_ingress.close()
            await _cancel_task(worker)
            await _best_effort_close(voice_session, label="voice orchestrator session")
            await _cancel_task(sender)
    finally:
        await server._connection_gate.release()


async def _run_turn(
    *,
    session: EdgeOrchestratorSession,
    request: OrchestratorTurnRequest,
    emit_event: Callable[[dict[str, Any]], None],
    tool_bridge: RemoteToolBridge,
    closed_event: Event,
    turn_gate: _TurnGate,
) -> None:
    """Run one blocking orchestrator turn in a worker thread."""

    try:
        if closed_event.is_set():
            return

        def thread_emit(payload: dict[str, Any]) -> None:
            if not closed_event.is_set():
                emit_event(payload)

        result = await asyncio.to_thread(
            session.run_turn,
            request.prompt,
            conversation=request.conversation,
            supervisor_conversation=request.supervisor_conversation,
            emit_event=thread_emit,
            tool_bridge=tool_bridge,
        )
        thread_emit(result.to_payload())
    except Exception:
        logger.exception("Orchestrator turn failed")
        emit_event(_error_payload("The request could not be completed."))
    finally:
        turn_gate.finish(asyncio.current_task())


async def _voice_worker(
    *,
    voice_session: EdgeOrchestratorVoiceSession,
    voice_ingress: _VoiceIngressQueue,
    emit_event: Callable[[dict[str, Any]], None],
    closed_event: Event,
) -> None:
    """Consume voice websocket messages from the bounded ingress queue."""

    try:
        while not closed_event.is_set():
            try:
                payload = await voice_ingress.get()
            except _VoiceIngressClosed:
                return

            message_type = str(payload.get("type", "") or "")
            try:
                if message_type == "voice_hello":
                    request = OrchestratorVoiceHelloRequest.from_payload(payload)
                    for event_payload in await asyncio.to_thread(voice_session.handle_hello, request):
                        emit_event(event_payload)
                    continue
                if message_type == "voice_runtime_state":
                    runtime_event = OrchestratorVoiceRuntimeStateEvent.from_payload(payload)
                    for event_payload in await asyncio.to_thread(
                        voice_session.handle_runtime_state,
                        runtime_event,
                    ):
                        emit_event(event_payload)
                    continue
                if message_type == "voice_identity_profiles":
                    identity_event = OrchestratorVoiceIdentityProfilesEvent.from_payload(payload)
                    for event_payload in await asyncio.to_thread(
                        voice_session.handle_identity_profiles,
                        identity_event,
                    ):
                        emit_event(event_payload)
                    continue
                if message_type == "voice_audio_frame":
                    frame = OrchestratorVoiceAudioFrame.from_payload(payload)
                    for event_payload in await asyncio.to_thread(voice_session.handle_audio_frame, frame):
                        emit_event(event_payload)
                    continue
            except Exception:
                logger.exception("Voice orchestrator websocket message handling failed")
                emit_event(
                    OrchestratorVoiceErrorEvent(
                        error="The voice session could not process the message."
                    ).to_payload()
                )
                continue

            emit_event(OrchestratorVoiceErrorEvent(error="Unsupported message type.").to_payload())
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Voice worker failed unexpectedly")


async def _sender_loop(
    websocket: WebSocket,
    outgoing: asyncio.Queue[dict[str, Any]],
    closed_event: Event,
) -> None:
    """Forward queued payloads from background workers to the websocket."""

    try:
        while not closed_event.is_set():
            payload = await outgoing.get()
            await websocket.send_json(payload)
    except asyncio.CancelledError:
        raise
    except (RuntimeError, WebSocketDisconnect):
        closed_event.set()
    except Exception:
        closed_event.set()
        logger.exception("Sender loop failed unexpectedly")


def create_app(env_file: str | Path) -> FastAPI:
    """Load Twinr config from disk and build the orchestrator FastAPI app."""

    env_path = Path(env_file)
    config = TwinrConfig.from_env(env_path)
    assert_transcript_first_voice_gateway_contract(config, env_file=str(env_path))
    return EdgeOrchestratorServer(config).create_app()


async def _receive_json_payload(
    websocket: WebSocket,
    *,
    max_message_bytes: int,
    idle_timeout_seconds: float | None,
) -> dict[str, Any]:
    """Receive one websocket frame, enforce limits, and parse it as a JSON object."""

    try:
        if idle_timeout_seconds is not None and idle_timeout_seconds > 0:
            message = await asyncio.wait_for(websocket.receive(), timeout=idle_timeout_seconds)
        else:
            message = await websocket.receive()
    except asyncio.TimeoutError as exc:
        raise _IdleTimeoutExceeded() from exc

    message_type = str(message.get("type", "") or "")
    if message_type == "websocket.disconnect":
        raise WebSocketDisconnect(message.get("code", 1000))
    if message_type != "websocket.receive":
        raise _ProtocolMessageError("Invalid websocket message.")

    raw_bytes = _extract_message_bytes(message)
    if max_message_bytes > 0 and len(raw_bytes) > max_message_bytes:
        raise _ProtocolMessageError(
            "Websocket message too large.",
            close_code=1009,
            close_reason="message_too_large",
        )

    try:
        payload = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise _ProtocolMessageError("Invalid websocket message.") from exc

    if not isinstance(payload, dict):
        raise _ProtocolMessageError("Invalid message payload.")

    return payload


def _extract_message_bytes(message: dict[str, Any]) -> bytes:
    """Return the raw bytes of one websocket receive message."""

    data = message.get("bytes")
    if data is not None:
        if isinstance(data, bytes):
            return data
        raise _ProtocolMessageError("Invalid websocket message payload.")
    text = message.get("text")
    if text is None:
        raise _ProtocolMessageError("Invalid websocket message payload.")
    return str(text).encode("utf-8")


def _enqueue_payload(
    event_loop: asyncio.AbstractEventLoop,
    outgoing: asyncio.Queue[dict[str, Any]],
    payload: dict[str, Any],
    closed_event: Event,
    websocket: WebSocket | None,
) -> None:
    """Schedule one outgoing payload onto the connection-local event loop."""

    if closed_event.is_set():
        return

    try:
        event_loop.call_soon_threadsafe(
            _enqueue_payload_nowait,
            outgoing,
            payload,
            closed_event,
            websocket,
        )
    except RuntimeError:
        closed_event.set()


def _enqueue_payload_nowait(
    outgoing: asyncio.Queue[dict[str, Any]],
    payload: dict[str, Any],
    closed_event: Event,
    websocket: WebSocket | None,
) -> None:
    """Push one payload into the bounded outgoing queue without blocking."""

    if closed_event.is_set():
        return

    try:
        outgoing.put_nowait(payload)
    except asyncio.QueueFull:
        closed_event.set()
        logger.error("Outgoing websocket queue overflow; closing websocket")
        if websocket is not None:
            asyncio.create_task(_close_websocket_quietly(websocket, code=1013, reason="backpressure"))


async def _drain_or_defer_turn_shutdown(
    *,
    turn_gate: _TurnGate,
    session: EdgeOrchestratorSession | None,
    grace_seconds: float,
) -> EdgeOrchestratorSession | None:
    """Wait briefly for the active turn to stop, or defer session close until it does."""

    if session is None:
        return None

    task = turn_gate.active_task()
    if task is None:
        return session

    await _best_effort_invoke(session, ("cancel_current_turn", "cancel_turn", "cancel", "abort"), label="orchestrator session")

    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=grace_seconds)
        return session
    except asyncio.TimeoutError:
        logger.warning(
            "Active orchestrator turn did not stop within %.2fs; deferring session close until it completes",
            grace_seconds,
        )

        def _close_when_done(_task: asyncio.Task[Any]) -> None:
            async def _close_later() -> None:
                await _best_effort_close(session, label="orchestrator session")

            with contextlib.suppress(RuntimeError):
                asyncio.create_task(_close_later())

        task.add_done_callback(_close_when_done)
        return None


async def _cancel_task(task: asyncio.Task[Any] | None) -> None:
    """Cancel and await a background task if it still exists."""

    if task is None:
        return

    if not task.done():
        task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        return
    except Exception:
        logger.exception("Background task cleanup failed")


async def _best_effort_invoke(resource: Any | None, method_names: Iterable[str], *, label: str) -> None:
    """Invoke the first matching callable on a resource and ignore teardown noise."""

    if resource is None:
        return

    for method_name in method_names:
        method = getattr(resource, method_name, None)
        if not callable(method):
            continue

        try:
            result = method()
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            logger.exception("Failed to invoke %s on %s", method_name, label)
        return


async def _best_effort_close(resource: Any | None, *, label: str) -> None:
    """Call the first supported close-like method on a resource."""

    await _best_effort_invoke(resource, ("close", "shutdown", "stop"), label=label)


async def _deny_websocket(websocket: WebSocket, *, status_code: int, detail: str) -> None:
    """Deny a websocket upgrade with an HTTP response when supported."""

    response = PlainTextResponse(detail, status_code=status_code)
    send_denial_response = getattr(websocket, "send_denial_response", None)
    with contextlib.suppress(RuntimeError, WebSocketDisconnect):
        if callable(send_denial_response):
            try:
                await send_denial_response(response)
                return
            except RuntimeError:
                pass
        await websocket.close(code=1008, reason=_safe_close_reason(detail))


async def _close_websocket_quietly(websocket: WebSocket, *, code: int, reason: str) -> None:
    """Close a websocket while suppressing teardown noise."""

    with contextlib.suppress(RuntimeError, WebSocketDisconnect):
        await websocket.close(code=code, reason=_safe_close_reason(reason))


def _safe_close_reason(reason: str) -> str:
    """Clamp a close reason to the RFC6455 practical limit."""

    encoded = reason.encode("utf-8")
    if len(encoded) <= 123:
        return reason
    return encoded[:123].decode("utf-8", errors="ignore")


def _error_payload(message: str) -> dict[str, Any]:
    """Build a normalized websocket error payload."""

    return OrchestratorErrorEvent(error=message).to_payload()


def _normalize_origin(origin: str) -> str | None:
    """Normalize an Origin header into a comparable scheme://host[:port] form."""

    raw = origin.strip()
    if not raw or raw.lower() == "null":
        return None

    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None

    host = parsed.hostname.lower()
    port = parsed.port
    default_port = 80 if parsed.scheme == "http" else 443
    if port is None or port == default_port:
        return f"{parsed.scheme}://{host}"
    return f"{parsed.scheme}://{host}:{port}"


def _is_loopback_origin(origin: str) -> bool:
    """Return whether a normalized origin resolves to a loopback host."""

    parsed = urlparse(origin)
    host = parsed.hostname or ""
    return _is_loopback_host(host)


def _get_outgoing_queue_maxsize(config: TwinrConfig) -> int:
    """Resolve the bounded outgoing queue size for one websocket."""

    raw = getattr(config, "orchestrator_outgoing_queue_maxsize", None)
    if raw is None:
        raw = os.getenv("ORCHESTRATOR_OUTGOING_QUEUE_MAXSIZE", "256")
    # Never allow 0 here: asyncio.Queue(0) is unbounded, which is the opposite of what this
    # transport layer needs on a memory-constrained Raspberry Pi.
    return _coerce_positive_int(raw, default=256)


def _get_orchestrator_max_message_bytes(config: TwinrConfig) -> int:
    """Resolve the maximum accepted message size for the non-voice websocket."""

    raw = getattr(config, "orchestrator_max_message_bytes", None)
    if raw is None:
        raw = os.getenv("ORCHESTRATOR_MAX_MESSAGE_BYTES", str(4 * 1024 * 1024))
    return _coerce_positive_int(raw, default=4 * 1024 * 1024)


def _get_voice_max_message_bytes(config: TwinrConfig) -> int:
    """Resolve the maximum accepted message size for the voice websocket."""

    raw = getattr(config, "orchestrator_voice_max_message_bytes", None)
    if raw is None:
        raw = os.getenv("ORCHESTRATOR_VOICE_MAX_MESSAGE_BYTES", str(1024 * 1024))
    return _coerce_positive_int(raw, default=1024 * 1024)


def _get_orchestrator_idle_timeout_seconds(config: TwinrConfig) -> float | None:
    """Resolve an optional inactivity timeout for the non-voice websocket."""

    raw = getattr(config, "orchestrator_idle_timeout_seconds", None)
    if raw is None:
        raw = os.getenv("ORCHESTRATOR_IDLE_TIMEOUT_SECONDS", "0")
    return _coerce_optional_positive_float(raw)


def _get_voice_idle_timeout_seconds(config: TwinrConfig) -> float | None:
    """Resolve an optional inactivity timeout for the voice websocket."""

    raw = getattr(config, "orchestrator_voice_idle_timeout_seconds", None)
    if raw is None:
        raw = os.getenv("ORCHESTRATOR_VOICE_IDLE_TIMEOUT_SECONDS", "0")
    return _coerce_optional_positive_float(raw)


def _get_orchestrator_message_rate_per_second(config: TwinrConfig) -> float:
    """Resolve the per-connection ingress rate for the non-voice websocket."""

    raw = getattr(config, "orchestrator_message_rate_per_second", None)
    if raw is None:
        raw = os.getenv("ORCHESTRATOR_MESSAGE_RATE_PER_SECOND", "8")
    return _coerce_non_negative_float(raw, default=8.0)


def _get_orchestrator_message_burst(config: TwinrConfig) -> int:
    """Resolve the burst capacity for the non-voice websocket rate limiter."""

    raw = getattr(config, "orchestrator_message_burst", None)
    if raw is None:
        raw = os.getenv("ORCHESTRATOR_MESSAGE_BURST", "16")
    return _coerce_positive_int(raw, default=16)


def _get_voice_message_rate_per_second(config: TwinrConfig) -> float:
    """Resolve the per-connection ingress rate for the voice websocket."""

    raw = getattr(config, "orchestrator_voice_message_rate_per_second", None)
    if raw is None:
        raw = os.getenv("ORCHESTRATOR_VOICE_MESSAGE_RATE_PER_SECOND", "120")
    return _coerce_non_negative_float(raw, default=120.0)


def _get_voice_message_burst(config: TwinrConfig) -> int:
    """Resolve the burst capacity for the voice websocket rate limiter."""

    raw = getattr(config, "orchestrator_voice_message_burst", None)
    if raw is None:
        raw = os.getenv("ORCHESTRATOR_VOICE_MESSAGE_BURST", "240")
    return _coerce_positive_int(raw, default=240)


def _get_voice_ingress_queue_maxsize(config: TwinrConfig) -> int:
    """Resolve the bounded application-level voice ingress queue size."""

    raw = getattr(config, "orchestrator_voice_ingress_queue_maxsize", None)
    if raw is None:
        raw = os.getenv("ORCHESTRATOR_VOICE_INGRESS_QUEUE_MAXSIZE", "32")
    return _coerce_positive_int(raw, default=32)


def _get_turn_shutdown_grace_seconds(config: TwinrConfig) -> float:
    """Resolve how long teardown should wait for a turn worker to stop."""

    raw = getattr(config, "orchestrator_turn_shutdown_grace_seconds", None)
    if raw is None:
        raw = os.getenv("ORCHESTRATOR_TURN_SHUTDOWN_GRACE_SECONDS", "5")
    return _coerce_positive_float(raw, default=5.0)


def _get_max_connections(config: TwinrConfig) -> int:
    """Resolve the maximum concurrent websocket connections for the whole app."""

    raw = getattr(config, "orchestrator_max_connections", None)
    if raw is None:
        raw = os.getenv("ORCHESTRATOR_MAX_CONNECTIONS", "8")
    return _coerce_non_negative_int(raw, default=8)


def _get_allowed_origins(config: TwinrConfig) -> frozenset[str]:
    """Resolve the explicit browser Origin allowlist."""

    raw = getattr(config, "orchestrator_allowed_origins", None)
    if raw is None:
        raw = os.getenv("ORCHESTRATOR_ALLOWED_ORIGINS", "")
    parts = raw if isinstance(raw, (list, tuple, set, frozenset)) else str(raw).split(",")
    origins = set[str]()
    for part in parts:
        normalized = _normalize_origin(str(part))
        if normalized is not None:
            origins.add(normalized)
    return frozenset(origins)


def _allow_insecure_loopback(config: TwinrConfig) -> bool:
    """Return whether plaintext loopback websocket auth is explicitly allowed."""

    raw = getattr(config, "orchestrator_allow_insecure_loopback", None)
    if raw is None:
        raw = os.getenv("ORCHESTRATOR_ALLOW_INSECURE_LOOPBACK", "0")
    return _coerce_bool(raw)


def _coerce_non_negative_int(value: Any, *, default: int) -> int:
    """Parse a non-negative integer or fall back to a safe default."""

    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number >= 0 else default


def _coerce_positive_int(value: Any, *, default: int) -> int:
    """Parse a positive integer or fall back to a safe default."""

    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


def _coerce_positive_float(value: Any, *, default: float) -> float:
    """Parse a positive float or fall back to a safe default."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


def _coerce_non_negative_float(value: Any, *, default: float) -> float:
    """Parse a non-negative float or fall back to a safe default."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number >= 0 else default


def _coerce_optional_positive_float(value: Any) -> float | None:
    """Parse a positive float, or None when unset/disabled."""

    if value is None:
        return None
    raw = str(value).strip()
    if not raw or raw in {"0", "0.0", "false", "False", "off", "none", "None"}:
        return None
    try:
        number = float(raw)
    except ValueError:
        return None
    return number if number > 0 else None


def _coerce_bool(value: Any) -> bool:
    """Parse a transport-style boolean token."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _client_host(websocket: WebSocket) -> str:
    """Extract the client host string from a websocket connection."""

    if websocket.client is None or websocket.client.host is None:
        return ""
    return websocket.client.host


def _is_loopback_host(host: str) -> bool:
    """Return whether a host string refers to loopback."""

    if not host:
        return False

    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host.strip().lower() == "localhost"