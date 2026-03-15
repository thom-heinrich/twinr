from __future__ import annotations

from collections.abc import Callable
from typing import Any
import json

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
    def __init__(
        self,
        url: str,
        *,
        shared_secret: str | None = None,
        connector: Callable[..., Any] | None = None,
    ) -> None:
        self.url = url
        self.shared_secret = shared_secret
        self._connector = connector or websocket_connect

    def run_turn(
        self,
        request: OrchestratorTurnRequest,
        *,
        tool_handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]],
        on_text_delta: Callable[[str], None] | None = None,
        on_ack: Callable[[OrchestratorAckEvent], None] | None = None,
    ) -> OrchestratorClientTurnResult:
        headers = None
        if (self.shared_secret or "").strip():
            headers = {"x-twinr-secret": self.shared_secret.strip()}
        ack_events: list[OrchestratorAckEvent] = []
        with self._connector(self.url, additional_headers=headers) as websocket:
            websocket.send(json.dumps(request.to_payload(), ensure_ascii=False))
            while True:
                payload = json.loads(websocket.recv())
                message_type = str(payload.get("type", "") or "")
                if message_type == "ack":
                    event = OrchestratorAckEvent(
                        ack_id=str(payload.get("ack_id", "") or "").strip(),
                        text=str(payload.get("text", "") or ""),
                    )
                    ack_events.append(event)
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
                    request_event = OrchestratorToolRequest.from_payload(payload)
                    handler = tool_handlers.get(request_event.name)
                    if handler is None:
                        response = OrchestratorToolResponse(
                            call_id=request_event.call_id,
                            ok=False,
                            error=f"Unsupported remote tool: {request_event.name}",
                        )
                    else:
                        try:
                            output = handler(dict(request_event.arguments))
                        except Exception as exc:
                            response = OrchestratorToolResponse(
                                call_id=request_event.call_id,
                                ok=False,
                                error=str(exc),
                            )
                        else:
                            response = OrchestratorToolResponse(
                                call_id=request_event.call_id,
                                ok=True,
                                output=output,
                            )
                    websocket.send(json.dumps(response.to_payload(), ensure_ascii=False))
                    continue
                if message_type == "turn_complete":
                    completed = OrchestratorTurnCompleteEvent.from_payload(payload)
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
                    raise RuntimeError(str(payload.get("error", "Orchestrator turn failed")))
