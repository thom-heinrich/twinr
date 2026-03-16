"""Bridge Twinr's dual-lane tool loop into the edge-orchestrator transport.

This module turns websocket requests into local tool-loop turns, emits ack/text
deltas back to the client, and correlates remote tool calls with returned
results.
"""

from __future__ import annotations

from dataclasses import dataclass
from queue import Queue
from threading import Event, Lock
from typing import Any, Callable, Sequence
import asyncio
import json
import logging
import os
from collections.abc import Mapping

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.contracts import AgentToolCall, ConversationLike, ToolCallingAgentProvider
from twinr.agent.base_agent.prompting.personality import load_supervisor_loop_instructions
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


LOGGER = logging.getLogger(__name__)  # AUDIT-FIX(#4): Surface stale/late tool-result problems without crashing the session.
_DEFAULT_REMOTE_TOOL_TIMEOUT_SECONDS = 90.0  # AUDIT-FIX(#5): Use a more forgiving default for intermittent home-WiFi conditions.
_REMOTE_TOOL_TIMEOUT_ENV = "TWINR_REMOTE_TOOL_TIMEOUT_SECONDS"
_TURN_FAILURE_TEXT_ENV = "TWINR_ORCHESTRATOR_FAILURE_TEXT"
_DEFAULT_TURN_FAILURE_TEXT = "Sorry, I had trouble finishing that. Please try again."


def _read_positive_float_env(name: str, default: float) -> float:
    """Read a positive float from the environment or fall back to ``default``."""

    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        parsed = float(raw_value)
    except ValueError:
        LOGGER.warning("Invalid %s=%r; using default %.1f", name, raw_value, default)  # AUDIT-FIX(#5): Invalid timeout config degrades safely.
        return default
    if parsed <= 0:
        LOGGER.warning("Non-positive %s=%r; using default %.1f", name, raw_value, default)  # AUDIT-FIX(#5): Reject unusable timeout values.
        return default
    return parsed


def _read_turn_failure_text() -> str:
    """Read the senior-facing fallback phrase for orchestrator failures."""

    text = os.getenv(_TURN_FAILURE_TEXT_ENV, _DEFAULT_TURN_FAILURE_TEXT).strip()  # AUDIT-FIX(#6): Allow deployment-specific, senior-safe fallback phrasing.
    return text or _DEFAULT_TURN_FAILURE_TEXT


def _normalize_tool_names(tool_names: Sequence[str] | None) -> tuple[str, ...]:
    """Normalize and deduplicate the tool names exposed to the remote client."""

    raw_tool_names = realtime_tool_names() if tool_names is None else tuple(tool_names)  # AUDIT-FIX(#2): Preserve an explicit empty tool list instead of re-enabling defaults.
    normalized: list[str] = []
    seen: set[str] = set()
    for name in raw_tool_names:
        if not isinstance(name, str) or not name:
            raise ValueError("Tool names must be non-empty strings")  # AUDIT-FIX(#2): Fail fast on invalid tool registrations.
        if name in seen:
            continue  # AUDIT-FIX(#2): Deduplicate tool names to keep handler/schema maps deterministic.
        seen.add(name)
        normalized.append(name)
    return tuple(normalized)


def _assert_not_running_on_event_loop_thread() -> None:
    """Refuse blocking bridge execution on an active asyncio event-loop thread."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError(
        "EdgeOrchestratorSession.run_turn() must execute off the asyncio event-loop thread because remote tool waits are blocking"
    )  # AUDIT-FIX(#3): Fail fast instead of risking a whole-process deadlock on the uvicorn loop thread.


def _compose_failure_text(streamed_chunks: Sequence[str], fallback_text: str) -> str:
    """Append the fallback phrase to any already-streamed partial text."""

    streamed_text = "".join(streamed_chunks)
    if not streamed_text:
        return fallback_text
    if streamed_text.endswith((" ", "\n")):
        return f"{streamed_text}{fallback_text}"
    return f"{streamed_text} {fallback_text}"


def _build_error_turn_complete_event(text: str) -> OrchestratorTurnCompleteEvent:
    """Build a well-formed completion event for a failed turn."""

    return OrchestratorTurnCompleteEvent(
        text=text,
        rounds=0,
        used_web_search=False,
        response_id="",
        request_id="",
        model="",
        token_usage={},
        tool_calls=[],
        tool_results=[],
    )  # AUDIT-FIX(#6): Return a well-formed completion object on failure so the caller can recover cleanly.


class _UnusedSupervisorProvider:
    """Guard the dual-lane loop from calling the wrong supervisor API."""

    def start_turn_streaming(self, *args, **kwargs):  # pragma: no cover - should not run
        """Reject streaming supervisor calls on the structured decision path."""

        raise RuntimeError("Structured supervisor decision path should not call the tool-loop supervisor provider")

    def continue_turn_streaming(self, *args, **kwargs):  # pragma: no cover - should not run
        """Reject continued supervisor streaming on the structured decision path."""

        raise RuntimeError("Structured supervisor decision path should not call the tool-loop supervisor provider")


@dataclass
class _PendingToolCall:
    """Track one in-flight remote tool request awaiting a result."""

    request: OrchestratorToolRequest
    done: Event
    output: dict[str, Any] | None = None
    error: str | None = None


class RemoteToolBridge:
    """Correlate local tool calls with remote websocket tool results."""

    def __init__(
        self,
        emit_event: Callable[[dict[str, Any]], None],
        *,
        tool_result_timeout_seconds: float | None = None,
    ) -> None:
        self._emit_event = emit_event
        self._pending: dict[str, _PendingToolCall] = {}
        self._pending_lock = Lock()  # AUDIT-FIX(#1): Protect the in-flight call map across handler and result threads.
        self._tool_result_timeout_seconds = self._resolve_tool_result_timeout(tool_result_timeout_seconds)  # AUDIT-FIX(#5): Make the wait budget configurable.

    @staticmethod
    def _resolve_tool_result_timeout(timeout_seconds: float | None) -> float:
        """Resolve the blocking wait budget for remote tool results."""

        if timeout_seconds is None:
            return _read_positive_float_env(_REMOTE_TOOL_TIMEOUT_ENV, _DEFAULT_REMOTE_TOOL_TIMEOUT_SECONDS)
        if timeout_seconds <= 0:
            raise ValueError("tool_result_timeout_seconds must be > 0")
        return timeout_seconds

    @staticmethod
    def _normalize_tool_arguments(arguments: Any) -> dict[str, Any]:
        """Normalize tool-call arguments into a plain mapping."""

        if arguments is None:
            return {}
        if not isinstance(arguments, Mapping):
            raise TypeError("Remote tool arguments must be a mapping or None")
        return dict(arguments)  # AUDIT-FIX(#7): Reject malformed model output with a controlled error.

    @staticmethod
    def _normalize_tool_output(output: dict[str, Any] | None) -> dict[str, Any] | None:
        """Normalize remote tool output into a plain mapping."""

        if output is None:
            return None
        if not isinstance(output, Mapping):
            raise TypeError("Remote tool output must be a mapping or None")
        return dict(output)  # AUDIT-FIX(#7): Prevent non-dict outputs from surfacing as late transport/type failures.

    @staticmethod
    def _normalize_call_id(call_id: Any) -> str:
        """Validate and normalize the transport correlation identifier."""

        if not isinstance(call_id, str) or not call_id.strip():
            raise TypeError("Remote tool call_id must be a non-empty string")
        return call_id

    def build_handlers(self, tool_names: Sequence[str]) -> dict[str, Callable[[AgentToolCall], Any]]:
        """Build local tool handlers that forward work to the remote client."""

        return {name: self._make_handler(name) for name in tool_names}

    def submit_result(self, call_id: str, *, output: dict[str, Any] | None, error: str | None) -> None:
        """Apply one returned tool result to the matching pending request."""

        try:
            normalized_call_id = self._normalize_call_id(call_id)
        except TypeError as exc:
            LOGGER.warning("Discarding remote tool result with invalid call_id: %s", exc)  # AUDIT-FIX(#7): Bad external correlation IDs are dropped safely instead of crashing the endpoint.
            return
        try:
            normalized_output = self._normalize_tool_output(output)
        except TypeError as exc:
            LOGGER.warning("Discarding invalid remote tool output payload for call_id=%s: %s", normalized_call_id, exc)  # AUDIT-FIX(#7): Bad remote outputs become tool errors instead of endpoint crashes without leaking payload content.
            normalized_output = None
            error = error or "Remote tool returned an invalid output payload"

        with self._pending_lock:  # AUDIT-FIX(#1): Make lookup/update/set atomic with respect to concurrent waiter cleanup.
            pending = self._pending.get(normalized_call_id)
            if pending is None:
                LOGGER.warning("Discarding stale or unknown remote tool result for call_id=%s", normalized_call_id)  # AUDIT-FIX(#4): Late results are normal under retries/timeouts and must not crash the process.
                return
            if pending.done.is_set():
                LOGGER.warning("Discarding duplicate remote tool result for call_id=%s", normalized_call_id)  # AUDIT-FIX(#4): Duplicate result deliveries become idempotent no-ops.
                return
            pending.output = normalized_output
            pending.error = error
            pending.done.set()

    def _make_handler(self, name: str) -> Callable[[AgentToolCall], Any]:
        """Build one forwarding handler for a named remote tool."""

        def _handler(tool_call: AgentToolCall) -> dict[str, Any]:
            call_id = self._normalize_call_id(tool_call.call_id)  # AUDIT-FIX(#7): Guard against malformed provider output before it reaches bridge state.
            request = OrchestratorToolRequest(
                call_id=call_id,
                name=name,
                arguments=self._normalize_tool_arguments(tool_call.arguments),
            )
            pending = _PendingToolCall(request=request, done=Event())
            with self._pending_lock:  # AUDIT-FIX(#1): Register the waiter atomically and reject in-flight call_id collisions.
                if call_id in self._pending:
                    raise RuntimeError(f"Duplicate in-flight remote tool call_id: {call_id}")
                self._pending[call_id] = pending
            try:
                payload = request.to_payload()
                json.dumps(payload)  # AUDIT-FIX(#7): Fail fast on non-JSON-serializable payloads before handing them to the transport.
                self._emit_event(payload)
                if not pending.done.wait(timeout=self._tool_result_timeout_seconds):
                    raise RuntimeError(
                        f"Remote tool '{name}' did not return within {self._tool_result_timeout_seconds:.0f} seconds"
                    )  # AUDIT-FIX(#5): Use the configurable timeout budget rather than a brittle hard-coded 60s.
                if pending.error:
                    raise RuntimeError(pending.error)
                return pending.output or {}
            finally:
                with self._pending_lock:  # AUDIT-FIX(#1): Remove the waiter under lock so submit_result cannot race with cleanup.
                    self._pending.pop(call_id, None)

        setattr(_handler, "_twinr_accepts_tool_call", True)
        return _handler


class EdgeOrchestratorSession:
    """Run one Twinr tool-loop turn behind the websocket orchestrator server."""

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
        self.tool_names = _normalize_tool_names(tool_names)  # AUDIT-FIX(#2): Keep explicit empty tool lists empty and normalize duplicates deterministically.
        self._turn_failure_text = _read_turn_failure_text()  # AUDIT-FIX(#6): Keep the senior-facing recovery phrase simple and overrideable per deployment.

    def run_turn(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None,
        supervisor_conversation: ConversationLike | None,
        emit_event: Callable[[dict[str, Any]], None],
        tool_bridge: RemoteToolBridge,
    ) -> OrchestratorTurnCompleteEvent:
        """Run one orchestrated tool-loop turn and stream events to the caller."""

        streamed_chunks: list[str] = []
        first_delta = True

        def on_text_delta(delta: str) -> None:
            nonlocal first_delta
            cleaned = delta.strip()
            if first_delta and cleaned:
                try:
                    ack_id = ack_id_for_text(cleaned)
                except Exception as exc:
                    LOGGER.warning("Ack classification failed; falling back to plain text delta (%s)", type(exc).__name__)  # AUDIT-FIX(#6): Ack helper failures must not abort the turn.
                    ack_id = None
                if ack_id is not None:
                    try:
                        emit_event(OrchestratorAckEvent(ack_id=ack_id, text=cleaned).to_payload())
                    except Exception as exc:
                        LOGGER.warning("Ack event emission failed; retrying as text delta (%s)", type(exc).__name__)  # AUDIT-FIX(#6): Gracefully degrade when the specialized ack path fails.
                    else:
                        streamed_chunks.append(cleaned)
                        first_delta = False
                        return
            emit_event(OrchestratorTextDeltaEvent(delta=delta).to_payload())
            streamed_chunks.append(delta)
            first_delta = False

        try:
            _assert_not_running_on_event_loop_thread()  # AUDIT-FIX(#3): Prevent a blocking wait from freezing the shared asyncio loop.
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
        except Exception as exc:
            LOGGER.error("Edge orchestrator turn failed (%s)", type(exc).__name__)  # AUDIT-FIX(#6): Convert provider/transport/tool failures into a controlled recovery path without spilling internals into logs.
            try:
                emit_event(OrchestratorTextDeltaEvent(delta=self._turn_failure_text).to_payload())
            except Exception as fallback_exc:
                LOGGER.debug("Unable to emit fallback failure text (%s)", type(fallback_exc).__name__)  # AUDIT-FIX(#6): A broken transport should not mask the primary failure.
            return _build_error_turn_complete_event(
                _compose_failure_text(streamed_chunks, self._turn_failure_text)
            )
