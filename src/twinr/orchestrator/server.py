"""Expose the FastAPI websocket server for Twinr's edge orchestrator.

This module accepts websocket turn requests, bridges remote tool results into a
session object, and keeps transport auth, queueing, and teardown bounded.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import ipaddress
import logging
import os
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Callable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.workflows.forensics import WorkflowForensics
from twinr.orchestrator.contracts import (
    OrchestratorErrorEvent,
    OrchestratorToolResponse,
    OrchestratorTurnRequest,
)
from twinr.orchestrator.session import EdgeOrchestratorSession, RemoteToolBridge
from twinr.orchestrator.remote_asr_service import (
    RemoteAsrHttpService,
    remote_asr_url_targets_local_orchestrator,
)
from twinr.orchestrator.voice_contracts import (
    OrchestratorVoiceAudioFrame,
    OrchestratorVoiceErrorEvent,
    OrchestratorVoiceHelloRequest,
    OrchestratorVoiceRuntimeStateEvent,
)
from twinr.orchestrator.voice_gateway_guardrails import (
    assert_transcript_first_voice_gateway_contract,
)
from twinr.orchestrator.voice_session import EdgeOrchestratorVoiceSession


logger = logging.getLogger(__name__)


class _TurnGate:
    """Serialize turn execution so one websocket has at most one live turn."""

    # AUDIT-FIX(#1): Serialize run_turn handling per websocket so one session/bridge cannot be mutated by overlapping turns.
    def __init__(self) -> None:
        self._lock = Lock()
        self._active = False

    def try_start(self) -> bool:
        """Mark the gate active if no turn is currently running."""

        with self._lock:
            if self._active:
                return False
            self._active = True
            return True

    def finish(self) -> None:
        """Release the active-turn marker."""

        with self._lock:
            self._active = False

    def is_active(self) -> bool:
        """Return whether the gate currently tracks an active turn."""

        with self._lock:
            return self._active


class EdgeOrchestratorServer:
    """Serve Twinr orchestrator turns over a websocket endpoint."""

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

    def create_app(self) -> FastAPI:
        """Build the FastAPI application that exposes the orchestrator socket."""

        server = self

        @contextlib.asynccontextmanager
        async def _app_lifespan(_app: FastAPI):
            try:
                yield
            finally:
                await _best_effort_close(server._remote_asr_service, label="remote ASR service")
                await _best_effort_close(server._voice_forensics, label="voice orchestrator forensics")

        app = FastAPI(title="Twinr Orchestrator", version="0.1.0", lifespan=_app_lifespan)
        if server._remote_asr_service is not None:
            app.include_router(server._remote_asr_service.build_router())

        @app.websocket("/ws/orchestrator")
        async def orchestrator_socket(websocket: WebSocket) -> None:
            if not server._authorize(websocket):
                logger.warning("Rejected unauthorized orchestrator websocket from %s", _client_host(websocket))
                await _close_websocket_quietly(websocket, code=1008, reason="unauthorized")  # AUDIT-FIX(#2): Fail closed on missing/invalid auth.
                return

            await websocket.accept()
            loop = asyncio.get_running_loop()  # AUDIT-FIX(#3): Use the connection-local event loop instead of app-global mutable state.
            closed_event = Event()  # AUDIT-FIX(#5): Stop background emission and cleanup work once the socket is gone.
            turn_gate = _TurnGate()
            outgoing: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
                maxsize=_get_outgoing_queue_maxsize(server.config)
            )  # AUDIT-FIX(#8): Bound outbound buffering to avoid unbounded memory growth on a blocked websocket.

            def emit_event(payload: dict[str, Any]) -> None:
                _enqueue_payload(loop, outgoing, payload, closed_event, websocket)  # AUDIT-FIX(#3): Route every emit through the connection-local loop.

            sender: asyncio.Task[Any] | None = None
            session: EdgeOrchestratorSession | None = None
            tool_bridge: RemoteToolBridge | None = None

            try:
                try:
                    tool_bridge = RemoteToolBridge(emit_event=emit_event)
                    session = server._session_factory(server.config)
                except Exception:
                    logger.exception("Failed to initialize orchestrator websocket session")
                    with contextlib.suppress(RuntimeError, WebSocketDisconnect):
                        await websocket.send_json(
                            _error_payload("The orchestrator could not start. Please reconnect.")
                        )  # AUDIT-FIX(#4): Report startup failures as protocol errors instead of crashing the handler.
                    await _close_websocket_quietly(websocket, code=1011, reason="startup_failed")
                    return

                sender = asyncio.create_task(_sender_loop(websocket, outgoing, closed_event))

                while not closed_event.is_set():
                    try:
                        payload = await websocket.receive_json()
                    except WebSocketDisconnect:
                        break
                    except Exception:
                        logger.exception("Failed to receive JSON from orchestrator websocket")
                        emit_event(_error_payload("Invalid websocket message."))  # AUDIT-FIX(#4): Keep malformed frames from killing the session.
                        continue

                    if not isinstance(payload, dict):
                        emit_event(_error_payload("Invalid message payload."))  # AUDIT-FIX(#4): Reject non-object JSON before calling payload.get().
                        continue

                    message_type = str(payload.get("type", "") or "")
                    if message_type == "run_turn":
                        try:
                            request = OrchestratorTurnRequest.from_payload(payload)
                        except Exception:
                            logger.exception("Invalid run_turn payload received")
                            emit_event(_error_payload("Invalid run_turn payload."))  # AUDIT-FIX(#4): Contain request validation failures.
                            continue

                        if not turn_gate.try_start():
                            emit_event(
                                _error_payload("A turn is already in progress.")
                            )  # AUDIT-FIX(#1): Reject overlapping turns on the same session/bridge.
                            continue

                        thread = Thread(
                            target=_run_turn_thread,
                            args=(session, request, outgoing, tool_bridge, loop, closed_event, turn_gate),
                            daemon=True,
                            name="twinr-orchestrator-turn",
                        )
                        try:
                            thread.start()
                        except Exception:
                            turn_gate.finish()
                            logger.exception("Failed to start orchestrator turn thread")
                            emit_event(_error_payload("The request could not be started."))  # AUDIT-FIX(#4): Handle thread startup failure without dropping the socket.
                        continue

                    if message_type == "tool_result":
                        if not turn_gate.is_active():
                            emit_event(
                                _error_payload("No active turn is waiting for a tool result.")
                            )  # AUDIT-FIX(#1): Reject stale or out-of-order tool results.
                            continue

                        try:
                            response = OrchestratorToolResponse.from_payload(payload)
                        except Exception:
                            logger.exception("Invalid tool_result payload received")
                            emit_event(_error_payload("Invalid tool_result payload."))  # AUDIT-FIX(#4): Contain tool_result validation failures.
                            continue

                        try:
                            tool_bridge.submit_result(
                                response.call_id,
                                output=response.output,
                                error=response.error if not response.ok else None,
                            )
                        except Exception:
                            logger.exception("Failed to apply tool_result to remote bridge")
                            emit_event(_error_payload("The tool result could not be applied."))  # AUDIT-FIX(#4): Prevent bridge failures from tearing down the websocket.
                        continue

                    emit_event(_error_payload("Unsupported message type."))  # AUDIT-FIX(#4): Explicitly reject unknown protocol messages.

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Unhandled orchestrator websocket failure")
                closed_event.set()  # AUDIT-FIX(#4): Stop queued sender traffic before sending a terminal error directly.
                await _cancel_task(sender)  # AUDIT-FIX(#4): Avoid concurrent websocket sends from the sender task during fatal error handling.
                sender = None
                with contextlib.suppress(RuntimeError, WebSocketDisconnect):
                    await websocket.send_json(
                        _error_payload("The connection failed. Please reconnect.")
                    )  # AUDIT-FIX(#4): Surface fatal handler errors without exposing internals.
            finally:
                closed_event.set()  # AUDIT-FIX(#5): Ensure post-disconnect emits are dropped immediately.
                await _best_effort_close(tool_bridge, label="remote tool bridge")  # AUDIT-FIX(#5): Best-effort teardown to unblock pending tool waits.
                await _best_effort_close(session, label="orchestrator session")  # AUDIT-FIX(#5): Release session resources on disconnect/restart.
                await _cancel_task(sender)  # AUDIT-FIX(#5): Await sender cancellation to avoid orphaned task warnings.

        @app.websocket("/ws/orchestrator/voice")
        async def orchestrator_voice_socket(websocket: WebSocket) -> None:
            if not server._authorize(websocket):
                logger.warning("Rejected unauthorized voice orchestrator websocket from %s", _client_host(websocket))
                await _close_websocket_quietly(websocket, code=1008, reason="unauthorized")
                return

            await websocket.accept()
            loop = asyncio.get_running_loop()
            closed_event = Event()
            outgoing: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
                maxsize=_get_outgoing_queue_maxsize(server.config)
            )

            def emit_event(payload: dict[str, Any]) -> None:
                _enqueue_payload(loop, outgoing, payload, closed_event, websocket)

            sender: asyncio.Task[Any] | None = None
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

                while not closed_event.is_set():
                    try:
                        payload = await websocket.receive_json()
                    except WebSocketDisconnect:
                        break
                    except Exception:
                        logger.exception("Failed to receive JSON from voice orchestrator websocket")
                        emit_event(OrchestratorVoiceErrorEvent(error="Invalid websocket message.").to_payload())
                        continue

                    if not isinstance(payload, dict):
                        emit_event(OrchestratorVoiceErrorEvent(error="Invalid message payload.").to_payload())
                        continue

                    message_type = str(payload.get("type", "") or "")
                    try:
                        if message_type == "voice_hello":
                            request = OrchestratorVoiceHelloRequest.from_payload(payload)
                            for event_payload in await asyncio.to_thread(voice_session.handle_hello, request):
                                emit_event(event_payload)
                            continue
                        if message_type == "voice_runtime_state":
                            event = OrchestratorVoiceRuntimeStateEvent.from_payload(payload)
                            for event_payload in await asyncio.to_thread(voice_session.handle_runtime_state, event):
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
                await _best_effort_close(voice_session, label="voice orchestrator session")
                await _cancel_task(sender)

        return app

    def _authorize(self, websocket: WebSocket) -> bool:
        """Authorize one websocket based on shared-secret or loopback policy."""

        expected = str(getattr(self.config, "orchestrator_shared_secret", "") or "").strip()
        actual = (websocket.headers.get("x-twinr-secret") or "").strip()
        if expected:
            return hmac.compare_digest(actual, expected)  # AUDIT-FIX(#2): Use constant-time comparison for shared-secret checks.

        return _allow_insecure_loopback(self.config) and _is_loopback_host(
            _client_host(websocket)
        )  # AUDIT-FIX(#2): Default to deny when no secret is configured; only allow explicit local-loopback opt-out.


def _run_turn_thread(
    session: EdgeOrchestratorSession,
    request: OrchestratorTurnRequest,
    outgoing: asyncio.Queue[dict[str, Any]],
    tool_bridge: RemoteToolBridge,
    event_loop: asyncio.AbstractEventLoop,
    closed_event: Event,
    turn_gate: _TurnGate,
) -> None:
    """Run one blocking orchestrator turn on a worker thread."""

    def emit(payload: dict[str, Any]) -> None:
        _enqueue_payload(event_loop, outgoing, payload, closed_event, None)  # AUDIT-FIX(#5): Drop thread emissions once the connection has closed.

    try:
        if closed_event.is_set():
            return

        result = session.run_turn(
            request.prompt,
            conversation=request.conversation,
            supervisor_conversation=request.supervisor_conversation,
            emit_event=emit,
            tool_bridge=tool_bridge,
        )
        emit(result.to_payload())
    except Exception:
        logger.exception("Orchestrator turn failed")
        emit(
            _error_payload("The request could not be completed.")
        )  # AUDIT-FIX(#6): Log full exception details server-side and return a sanitized client error.
    finally:
        turn_gate.finish()  # AUDIT-FIX(#1): Always release the single-turn gate, even on crashes.


async def _sender_loop(
    websocket: WebSocket,
    outgoing: asyncio.Queue[dict[str, Any]],
    closed_event: Event,
) -> None:
    """Forward queued payloads from the session thread to the websocket."""

    try:
        while not closed_event.is_set():
            payload = await outgoing.get()
            await websocket.send_json(payload)
    except asyncio.CancelledError:
        raise
    except (RuntimeError, WebSocketDisconnect):
        closed_event.set()  # AUDIT-FIX(#7): Convert sender-side websocket failures into clean connection shutdown.
    except Exception:
        closed_event.set()
        logger.exception("Sender loop failed unexpectedly")  # AUDIT-FIX(#7): Avoid silent task failure in the background sender.


def create_app(env_file: str | Path) -> FastAPI:
    """Load Twinr config from disk and build the orchestrator FastAPI app."""

    env_path = Path(env_file)
    config = TwinrConfig.from_env(env_path)
    assert_transcript_first_voice_gateway_contract(config, env_file=str(env_path))
    return EdgeOrchestratorServer(config).create_app()


def _enqueue_payload(
    event_loop: asyncio.AbstractEventLoop,
    outgoing: asyncio.Queue[dict[str, Any]],
    payload: dict[str, Any],
    closed_event: Event,
    websocket: WebSocket | None,
) -> None:
    """Schedule one outgoing payload onto the websocket event loop."""

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
        closed_event.set()  # AUDIT-FIX(#5): Ignore late emits against a closed event loop during teardown.


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
        logger.error("Outgoing orchestrator queue overflow; closing websocket")  # AUDIT-FIX(#8): Fail fast on backpressure instead of growing memory without bound.
        if websocket is not None:
            asyncio.create_task(_close_websocket_quietly(websocket, code=1013, reason="backpressure"))


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
        logger.exception("Background task cleanup failed")  # AUDIT-FIX(#5): Cleanup must not crash websocket teardown.


async def _best_effort_close(resource: Any | None, *, label: str) -> None:
    """Call the first supported close-like method on a resource."""

    if resource is None:
        return

    for method_name in ("close", "shutdown", "stop"):
        method = getattr(resource, method_name, None)
        if not callable(method):
            continue

        try:
            result = method()
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            logger.exception("Failed to close %s via %s", label, method_name)  # AUDIT-FIX(#5): Log teardown failures but continue cleanup.
        return


async def _close_websocket_quietly(websocket: WebSocket, *, code: int, reason: str) -> None:
    """Close a websocket while suppressing teardown noise."""

    with contextlib.suppress(RuntimeError, WebSocketDisconnect):
        await websocket.close(code=code, reason=reason)


def _error_payload(message: str) -> dict[str, Any]:
    """Build a normalized websocket error payload."""

    return OrchestratorErrorEvent(error=message).to_payload()


def _get_outgoing_queue_maxsize(config: TwinrConfig) -> int:
    """Resolve the bounded outgoing queue size for one websocket."""

    raw = getattr(config, "orchestrator_outgoing_queue_maxsize", None)
    if raw is None:
        raw = os.getenv("ORCHESTRATOR_OUTGOING_QUEUE_MAXSIZE", "256")
    return _coerce_non_negative_int(raw, default=256)


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
