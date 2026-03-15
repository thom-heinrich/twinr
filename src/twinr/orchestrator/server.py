from __future__ import annotations

import asyncio
from pathlib import Path
from threading import Thread
from typing import Any, Callable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from twinr.agent.base_agent.config import TwinrConfig
from twinr.orchestrator.contracts import (
    OrchestratorErrorEvent,
    OrchestratorToolResponse,
    OrchestratorTurnRequest,
)
from twinr.orchestrator.session import EdgeOrchestratorSession, RemoteToolBridge


class EdgeOrchestratorServer:
    def __init__(
        self,
        config: TwinrConfig,
        *,
        session_factory: Callable[[TwinrConfig], EdgeOrchestratorSession] | None = None,
    ) -> None:
        self.config = config
        self._session_factory = session_factory or EdgeOrchestratorSession

    def create_app(self) -> FastAPI:
        app = FastAPI(title="Twinr Orchestrator", version="0.1.0")
        server = self

        @app.websocket("/ws/orchestrator")
        async def orchestrator_socket(websocket: WebSocket) -> None:
            if not server._authorize(websocket):
                await websocket.close(code=1008)
                return
            await websocket.accept()
            outgoing: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
            tool_bridge = RemoteToolBridge(
                emit_event=lambda payload: websocket.app.state.loop.call_soon_threadsafe(outgoing.put_nowait, payload)
            )
            app.state.loop = asyncio.get_running_loop()
            session = server._session_factory(server.config)
            sender = asyncio.create_task(_sender_loop(websocket, outgoing))
            try:
                while True:
                    payload = await websocket.receive_json()
                    message_type = str(payload.get("type", "") or "")
                    if message_type == "run_turn":
                        request = OrchestratorTurnRequest.from_payload(payload)
                        thread = Thread(
                            target=_run_turn_thread,
                            args=(session, request, outgoing, tool_bridge, app.state.loop),
                            daemon=True,
                        )
                        thread.start()
                        continue
                    if message_type == "tool_result":
                        response = OrchestratorToolResponse.from_payload(payload)
                        tool_bridge.submit_result(
                            response.call_id,
                            output=response.output,
                            error=response.error if not response.ok else None,
                        )
                        continue
            except WebSocketDisconnect:
                return
            finally:
                sender.cancel()

        return app

    def _authorize(self, websocket: WebSocket) -> bool:
        expected = (self.config.orchestrator_shared_secret or "").strip()
        if not expected:
            return True
        actual = (websocket.headers.get("x-twinr-secret") or "").strip()
        return actual == expected


def _run_turn_thread(
    session: EdgeOrchestratorSession,
    request: OrchestratorTurnRequest,
    outgoing: asyncio.Queue[dict[str, Any]],
    tool_bridge: RemoteToolBridge,
    event_loop: asyncio.AbstractEventLoop,
) -> None:
    def emit(payload: dict[str, Any]) -> None:
        event_loop.call_soon_threadsafe(outgoing.put_nowait, payload)

    try:
        result = session.run_turn(
            request.prompt,
            conversation=request.conversation,
            supervisor_conversation=request.supervisor_conversation,
            emit_event=emit,
            tool_bridge=tool_bridge,
        )
        emit(result.to_payload())
    except Exception as exc:
        emit(OrchestratorErrorEvent(error=str(exc)).to_payload())


async def _sender_loop(websocket: WebSocket, outgoing: asyncio.Queue[dict[str, Any]]) -> None:
    while True:
        payload = await outgoing.get()
        await websocket.send_json(payload)


def create_app(env_file: str | Path) -> FastAPI:
    config = TwinrConfig.from_env(Path(env_file))
    return EdgeOrchestratorServer(config).create_app()
