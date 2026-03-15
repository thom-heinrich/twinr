from __future__ import annotations

from dataclasses import dataclass
from queue import Queue
from threading import Event
from typing import Any, Callable, Sequence
import json

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.contracts import AgentToolCall, ConversationLike, ToolCallingAgentProvider
from twinr.agent.base_agent.personality import load_supervisor_loop_instructions
from twinr.agent.tools import (
    DualLaneToolLoop,
    build_agent_tool_schemas,
    build_specialist_tool_agent_instructions,
    build_supervisor_decision_instructions,
    realtime_tool_names,
)
from twinr.orchestrator.acks import ack_id_for_text
from twinr.orchestrator.contracts import (
    OrchestratorAckEvent,
    OrchestratorTextDeltaEvent,
    OrchestratorToolRequest,
    OrchestratorTurnCompleteEvent,
)
from twinr.providers.openai import OpenAIBackend, OpenAISupervisorDecisionProvider, OpenAIToolCallingAgentProvider


class _UnusedSupervisorProvider:
    def start_turn_streaming(self, *args, **kwargs):  # pragma: no cover - should not run
        raise RuntimeError("Structured supervisor decision path should not call the tool-loop supervisor provider")

    def continue_turn_streaming(self, *args, **kwargs):  # pragma: no cover - should not run
        raise RuntimeError("Structured supervisor decision path should not call the tool-loop supervisor provider")


@dataclass
class _PendingToolCall:
    request: OrchestratorToolRequest
    done: Event
    output: dict[str, Any] | None = None
    error: str | None = None


class RemoteToolBridge:
    def __init__(self, emit_event: Callable[[dict[str, Any]], None]) -> None:
        self._emit_event = emit_event
        self._pending: dict[str, _PendingToolCall] = {}

    def build_handlers(self, tool_names: Sequence[str]) -> dict[str, Callable[[AgentToolCall], Any]]:
        return {name: self._make_handler(name) for name in tool_names}

    def submit_result(self, call_id: str, *, output: dict[str, Any] | None, error: str | None) -> None:
        pending = self._pending.get(call_id)
        if pending is None:
            raise KeyError(f"Unknown tool result call_id: {call_id}")
        pending.output = output
        pending.error = error
        pending.done.set()

    def _make_handler(self, name: str) -> Callable[[AgentToolCall], Any]:
        def _handler(tool_call: AgentToolCall) -> dict[str, Any]:
            request = OrchestratorToolRequest(
                call_id=tool_call.call_id,
                name=name,
                arguments=dict(tool_call.arguments),
            )
            pending = _PendingToolCall(request=request, done=Event())
            self._pending[request.call_id] = pending
            try:
                self._emit_event(request.to_payload())
                if not pending.done.wait(timeout=60.0):
                    raise RuntimeError(f"Timed out waiting for remote tool result: {name}")
                if pending.error:
                    raise RuntimeError(pending.error)
                return pending.output or {}
            finally:
                self._pending.pop(request.call_id, None)

        setattr(_handler, "_twinr_accepts_tool_call", True)
        return _handler


class EdgeOrchestratorSession:
    def __init__(
        self,
        config: TwinrConfig,
        *,
        supervisor_decision_provider=None,
        specialist_provider: ToolCallingAgentProvider | None = None,
        tool_names: Sequence[str] | None = None,
    ) -> None:
        self.config = config
        backend = OpenAIBackend(config=config)
        self.supervisor_decision_provider = supervisor_decision_provider or OpenAISupervisorDecisionProvider(
            backend,
            model_override=config.streaming_supervisor_model,
            reasoning_effort_override=config.streaming_supervisor_reasoning_effort,
            base_instructions_override=load_supervisor_loop_instructions(config),
            replace_base_instructions=True,
        )
        self.specialist_provider = specialist_provider or OpenAIToolCallingAgentProvider(
            backend,
            model_override=config.streaming_specialist_model,
            reasoning_effort_override=config.streaming_specialist_reasoning_effort,
        )
        self.tool_names = tuple(tool_names or realtime_tool_names())

    def run_turn(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None,
        supervisor_conversation: ConversationLike | None,
        emit_event: Callable[[dict[str, Any]], None],
        tool_bridge: RemoteToolBridge,
    ) -> OrchestratorTurnCompleteEvent:
        remote_handlers = tool_bridge.build_handlers(self.tool_names)
        tool_loop = DualLaneToolLoop(
            supervisor_provider=_UnusedSupervisorProvider(),
            specialist_provider=self.specialist_provider,
            supervisor_decision_provider=self.supervisor_decision_provider,
            tool_handlers=remote_handlers,
            tool_schemas=build_agent_tool_schemas(self.tool_names),
            supervisor_instructions=build_supervisor_decision_instructions(
                self.config,
                extra_instructions=self.config.openai_realtime_instructions,
            ),
            specialist_instructions=build_specialist_tool_agent_instructions(
                self.config,
                extra_instructions=self.config.openai_realtime_instructions,
            ),
            max_rounds=6,
        )

        first_delta = True

        def on_text_delta(delta: str) -> None:
            nonlocal first_delta
            cleaned = delta.strip()
            if first_delta and cleaned:
                ack_id = ack_id_for_text(cleaned)
                if ack_id is not None:
                    emit_event(OrchestratorAckEvent(ack_id=ack_id, text=cleaned).to_payload())
                    first_delta = False
                    return
            emit_event(OrchestratorTextDeltaEvent(delta=delta).to_payload())
            first_delta = False

        result = tool_loop.run(
            prompt,
            conversation=conversation,
            supervisor_conversation=supervisor_conversation,
            instructions=None,
            allow_web_search=False,
            on_text_delta=on_text_delta,
        )
        return OrchestratorTurnCompleteEvent(
            text=result.text,
            rounds=result.rounds,
            used_web_search=result.used_web_search,
            response_id=result.response_id,
            request_id=result.request_id,
            model=result.model,
            token_usage=result.token_usage,
            tool_calls=result.tool_calls,
            tool_results=result.tool_results,
        )
