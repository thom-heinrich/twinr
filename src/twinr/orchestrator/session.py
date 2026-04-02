"""Bridge Twinr's dual-lane tool loop into the edge-orchestrator transport.

This module turns websocket requests into local tool-loop turns, emits ack/text
deltas back to the client, and correlates remote tool calls with returned
results.
"""

# CHANGELOG: 2026-03-29
# BUG-1: Ack events could be skipped when the first streamed delta was whitespace-only; ack classification now waits for the first meaningful token.
# BUG-2: Concurrent run_turn() calls on the same session could interleave deltas/tool calls and corrupt a conversation; turns are now serialized per session.
# BUG-3: Tool payload validation accepted permissive JSON (e.g. NaN/Infinity) and only validated outbound requests, causing brittle late transport/runtime failures.
# SEC-1: Remote tool arguments/results/errors were effectively unbounded, enabling practical RAM/bandwidth/log-spam DoS on a Raspberry Pi 4; byte/length budgets are now enforced.
# SEC-2: Pending remote tool waiters could remain blocked after transport failure/disconnect; fail_all_pending() now releases them deterministically.
# IMP-1: Tool schemas are upgraded toward explicit strict structured outputs (closed objects + nullable optionals) so behavior is less provider/API-mode dependent.
# IMP-2: Static schemas/instructions are precomputed per session, tool-call latency is observable, and max rounds / payload budgets are env-configurable for Pi deployments.

from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Lock
from time import monotonic
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
from twinr.orchestrator.remote_tool_timeout import read_remote_tool_timeout_seconds
from twinr.providers.openai import OpenAIBackend, OpenAISupervisorDecisionProvider, OpenAIToolCallingAgentProvider


LOGGER = logging.getLogger(__name__)

_DEFAULT_REMOTE_TOOL_REQUEST_MAX_BYTES = 128 * 1024
_REMOTE_TOOL_REQUEST_MAX_BYTES_ENV = "TWINR_REMOTE_TOOL_REQUEST_MAX_BYTES"

_DEFAULT_REMOTE_TOOL_RESULT_MAX_BYTES = 512 * 1024
_REMOTE_TOOL_RESULT_MAX_BYTES_ENV = "TWINR_REMOTE_TOOL_RESULT_MAX_BYTES"

_DEFAULT_REMOTE_TOOL_ERROR_MAX_CHARS = 512
_REMOTE_TOOL_ERROR_MAX_CHARS_ENV = "TWINR_REMOTE_TOOL_ERROR_MAX_CHARS"

_DEFAULT_REMOTE_TOOL_CALL_ID_MAX_CHARS = 256
_REMOTE_TOOL_CALL_ID_MAX_CHARS_ENV = "TWINR_REMOTE_TOOL_CALL_ID_MAX_CHARS"

_DEFAULT_TOOL_NAME_MAX_CHARS = 128

_TURN_FAILURE_TEXT_ENV = "TWINR_ORCHESTRATOR_FAILURE_TEXT"
_DEFAULT_TURN_FAILURE_TEXT = "Sorry, I had trouble finishing that. Please try again."

_DEFAULT_MAX_ROUNDS = 6
_MAX_ROUNDS_ENV = "TWINR_ORCHESTRATOR_MAX_ROUNDS"


def _read_positive_int_env(name: str, default: int) -> int:
    """Read a positive integer from the environment or fall back to ``default``."""

    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        parsed = int(raw_value)
    except ValueError:
        LOGGER.warning("Invalid %s=%r; using default %d", name, raw_value, default)
        return default
    if parsed <= 0:
        LOGGER.warning("Non-positive %s=%r; using default %d", name, raw_value, default)
        return default
    return parsed


def _read_turn_failure_text() -> str:
    """Read the senior-facing fallback phrase for orchestrator failures."""

    text = os.getenv(_TURN_FAILURE_TEXT_ENV, _DEFAULT_TURN_FAILURE_TEXT).strip()
    return text or _DEFAULT_TURN_FAILURE_TEXT


def _json_size_bytes(payload: Any, *, context: str) -> int:
    """Return the UTF-8 JSON size of ``payload`` after strict JSON validation."""

    try:
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{context} must be valid JSON-serializable data") from exc
    return len(serialized.encode("utf-8"))


def _enforce_json_payload_budget(payload: Any, *, context: str, max_bytes: int) -> None:
    """Reject payloads that are not strict JSON or exceed ``max_bytes``."""

    size_bytes = _json_size_bytes(payload, context=context)
    if size_bytes > max_bytes:
        raise ValueError(f"{context} exceeds the {max_bytes}-byte limit")


def _normalize_error_text(error: Any, *, max_chars: int) -> str | None:
    """Normalize a remote-tool error into a bounded single-line string."""

    if error is None:
        return None
    if not isinstance(error, str):
        error = str(error)
    cleaned = " ".join(error.split())
    if not cleaned:
        return None
    if len(cleaned) > max_chars:
        cleaned = f"{cleaned[: max_chars - 1].rstrip()}…"
    return cleaned


def _normalize_tool_names(tool_names: Sequence[str] | None) -> tuple[str, ...]:
    """Normalize and deduplicate the tool names exposed to the remote client."""

    raw_tool_names = realtime_tool_names() if tool_names is None else tuple(tool_names)
    normalized: list[str] = []
    seen: set[str] = set()
    for name in raw_tool_names:
        if not isinstance(name, str) or not name:
            raise ValueError("Tool names must be non-empty strings")
        if len(name) > _DEFAULT_TOOL_NAME_MAX_CHARS:
            raise ValueError(
                f"Tool name {name!r} exceeds the {_DEFAULT_TOOL_NAME_MAX_CHARS}-character limit"
            )
        if name in seen:
            continue
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
    )


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
    )


def _schema_allows_null(schema: Mapping[str, Any]) -> bool:
    """Return ``True`` if a JSON schema already permits ``null``."""

    schema_type = schema.get("type")
    if schema_type == "null":
        return True
    if isinstance(schema_type, (list, tuple)) and "null" in schema_type:
        return True
    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and None in enum_values:
        return True
    if schema.get("const") is None and "const" in schema:
        return True
    for branch_key in ("anyOf", "oneOf"):
        branches = schema.get(branch_key)
        if isinstance(branches, list):
            for branch in branches:
                if isinstance(branch, Mapping) and _schema_allows_null(branch):
                    return True
    return False


def _make_nullable_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a JSON schema into a nullable schema while preserving semantics."""

    upgraded = dict(schema)
    if _schema_allows_null(upgraded):
        return upgraded

    schema_type = upgraded.get("type")
    if isinstance(schema_type, str):
        upgraded["type"] = [schema_type, "null"] if schema_type != "null" else "null"
        enum_values = upgraded.get("enum")
        if isinstance(enum_values, list) and None not in enum_values:
            upgraded["enum"] = [*enum_values, None]
        return upgraded

    if isinstance(schema_type, (list, tuple)):
        updated_types = list(schema_type)
        if "null" not in updated_types:
            updated_types.append("null")
        upgraded["type"] = updated_types
        enum_values = upgraded.get("enum")
        if isinstance(enum_values, list) and None not in enum_values:
            upgraded["enum"] = [*enum_values, None]
        return upgraded

    enum_values = upgraded.get("enum")
    if isinstance(enum_values, list):
        if None not in enum_values:
            upgraded["enum"] = [*enum_values, None]
        return upgraded

    return {"anyOf": [upgraded, {"type": "null"}]}


def _strictify_json_schema(schema: Mapping[str, Any], *, optional: bool) -> dict[str, Any]:
    """Best-effort upgrade of a JSON schema to a strict-structured-output-friendly form."""

    upgraded = dict(schema)

    for defs_key in ("$defs", "definitions"):
        definitions = upgraded.get(defs_key)
        if isinstance(definitions, Mapping):
            upgraded[defs_key] = {
                definition_name: _strictify_json_schema(definition_schema, optional=False)
                if isinstance(definition_schema, Mapping)
                else definition_schema
                for definition_name, definition_schema in definitions.items()
            }

    for key in ("items", "contains", "if", "then", "else", "not", "propertyNames"):
        nested = upgraded.get(key)
        if isinstance(nested, Mapping):
            upgraded[key] = _strictify_json_schema(nested, optional=False)
        elif isinstance(nested, list):
            upgraded[key] = [
                _strictify_json_schema(item, optional=False) if isinstance(item, Mapping) else item
                for item in nested
            ]

    for key in ("prefixItems", "allOf", "anyOf", "oneOf"):
        branches = upgraded.get(key)
        if isinstance(branches, list):
            upgraded[key] = [
                _strictify_json_schema(branch, optional=False) if isinstance(branch, Mapping) else branch
                for branch in branches
            ]

    properties = upgraded.get("properties")
    if isinstance(properties, Mapping):
        required_names = upgraded.get("required")
        required_set = set(required_names) if isinstance(required_names, list) else set()
        strict_properties: dict[str, Any] = {}
        ordered_property_names: list[str] = []
        for property_name, property_schema in properties.items():
            ordered_property_names.append(property_name)
            if isinstance(property_schema, Mapping):
                strict_properties[property_name] = _strictify_json_schema(
                    property_schema,
                    optional=property_name not in required_set,
                )
            else:
                strict_properties[property_name] = property_schema
        upgraded["properties"] = strict_properties
        upgraded["required"] = ordered_property_names
        upgraded["additionalProperties"] = False

    schema_type = upgraded.get("type")
    if schema_type == "object" or (isinstance(schema_type, (list, tuple)) and "object" in schema_type):
        upgraded["additionalProperties"] = False

    if optional:
        upgraded = _make_nullable_schema(upgraded)

    return upgraded


def _upgrade_tool_schema_for_strict_outputs(tool_schema: Any) -> Any:
    """Upgrade one tool schema toward explicit strict structured outputs."""

    if not isinstance(tool_schema, Mapping):
        return tool_schema

    upgraded = dict(tool_schema)
    if upgraded.get("type") != "function":
        return upgraded

    parameters = upgraded.get("parameters")
    if isinstance(parameters, Mapping):
        upgraded["parameters"] = _strictify_json_schema(parameters, optional=False)
        # BREAKING: tools now request explicit strict structured outputs when the provider honors OpenAI-style function schemas.
        upgraded["strict"] = True
        return upgraded

    function_payload = upgraded.get("function")
    if isinstance(function_payload, Mapping):
        function_copy = dict(function_payload)
        parameters = function_copy.get("parameters")
        if isinstance(parameters, Mapping):
            function_copy["parameters"] = _strictify_json_schema(parameters, optional=False)
        # BREAKING: nested function schemas are made explicit/strict to reduce provider-mode drift.
        function_copy["strict"] = True
        upgraded["function"] = function_copy
        return upgraded

    return upgraded


def _upgrade_tool_schemas_for_frontier(tool_schemas: Sequence[Any]) -> list[Any]:
    """Upgrade tool schemas to a more explicit, 2026-style strict form."""

    return [_upgrade_tool_schema_for_strict_outputs(tool_schema) for tool_schema in tool_schemas]


class _UnusedSupervisorProvider:
    """Guard the dual-lane loop from calling the wrong supervisor API."""

    def start_turn_streaming(self, *args, **kwargs):  # pragma: no cover - should not run
        """Reject streaming supervisor calls on the structured decision path."""

        raise RuntimeError("Structured supervisor decision path should not call the tool-loop supervisor provider")

    def continue_turn_streaming(self, *args, **kwargs):  # pragma: no cover - should not run
        """Reject continued supervisor streaming on the structured decision path."""

        raise RuntimeError("Structured supervisor decision path should not call the tool-loop supervisor provider")


@dataclass(slots=True)
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
        request_max_bytes: int | None = None,
        result_max_bytes: int | None = None,
        error_max_chars: int | None = None,
        call_id_max_chars: int | None = None,
    ) -> None:
        self._emit_event = emit_event
        self._pending: dict[str, _PendingToolCall] = {}
        self._pending_lock = Lock()
        self._tool_result_timeout_seconds = self._resolve_tool_result_timeout(tool_result_timeout_seconds)
        self._request_max_bytes = self._resolve_positive_int(
            request_max_bytes,
            env_name=_REMOTE_TOOL_REQUEST_MAX_BYTES_ENV,
            default=_DEFAULT_REMOTE_TOOL_REQUEST_MAX_BYTES,
            field_name="request_max_bytes",
        )
        self._result_max_bytes = self._resolve_positive_int(
            result_max_bytes,
            env_name=_REMOTE_TOOL_RESULT_MAX_BYTES_ENV,
            default=_DEFAULT_REMOTE_TOOL_RESULT_MAX_BYTES,
            field_name="result_max_bytes",
        )
        self._error_max_chars = self._resolve_positive_int(
            error_max_chars,
            env_name=_REMOTE_TOOL_ERROR_MAX_CHARS_ENV,
            default=_DEFAULT_REMOTE_TOOL_ERROR_MAX_CHARS,
            field_name="error_max_chars",
        )
        self._call_id_max_chars = self._resolve_positive_int(
            call_id_max_chars,
            env_name=_REMOTE_TOOL_CALL_ID_MAX_CHARS_ENV,
            default=_DEFAULT_REMOTE_TOOL_CALL_ID_MAX_CHARS,
            field_name="call_id_max_chars",
        )

    @staticmethod
    def _resolve_positive_int(
        value: int | None,
        *,
        env_name: str,
        default: int,
        field_name: str,
    ) -> int:
        """Resolve a positive integer from an override or environment."""

        if value is None:
            return _read_positive_int_env(env_name, default)
        if value <= 0:
            raise ValueError(f"{field_name} must be > 0")
        return value

    @staticmethod
    def _resolve_tool_result_timeout(timeout_seconds: float | None) -> float:
        """Resolve the blocking wait budget for remote tool results."""

        if timeout_seconds is None:
            return read_remote_tool_timeout_seconds(logger=LOGGER)
        if timeout_seconds <= 0:
            raise ValueError("tool_result_timeout_seconds must be > 0")
        return timeout_seconds

    def _normalize_tool_arguments(self, arguments: Any) -> dict[str, Any]:
        """Normalize tool-call arguments into a plain JSON mapping."""

        if arguments is None:
            return {}
        if not isinstance(arguments, Mapping):
            raise TypeError("Remote tool arguments must be a mapping or None")
        normalized = dict(arguments)
        _enforce_json_payload_budget(
            normalized,
            context="Remote tool arguments",
            max_bytes=self._request_max_bytes,
        )
        return normalized

    def _normalize_tool_output(self, output: Any) -> dict[str, Any] | None:
        """Normalize remote tool output into a plain JSON mapping."""

        if output is None:
            return None
        if not isinstance(output, Mapping):
            raise TypeError("Remote tool output must be a mapping or None")
        normalized = dict(output)
        _enforce_json_payload_budget(
            normalized,
            context="Remote tool output",
            max_bytes=self._result_max_bytes,
        )
        return normalized

    def _normalize_call_id(self, call_id: Any) -> str:
        """Validate and normalize the transport correlation identifier."""

        if not isinstance(call_id, str) or not call_id.strip():
            raise TypeError("Remote tool call_id must be a non-empty string")
        if len(call_id) > self._call_id_max_chars:
            raise ValueError(
                f"Remote tool call_id exceeds the {self._call_id_max_chars}-character limit"
            )
        return call_id

    def _normalize_error(self, error: Any) -> str | None:
        """Normalize and bound remote-tool errors."""

        return _normalize_error_text(error, max_chars=self._error_max_chars)

    def build_handlers(self, tool_names: Sequence[str]) -> dict[str, Callable[[AgentToolCall], Any]]:
        """Build local tool handlers that forward work to the remote client."""

        return {name: self._make_handler(name) for name in tool_names}

    def fail_all_pending(self, error: str = "Remote tool bridge closed") -> int:
        """Release all waiters with a synthetic error, e.g. on disconnect."""

        normalized_error = self._normalize_error(error) or "Remote tool bridge closed"
        released = 0
        with self._pending_lock:
            for pending in self._pending.values():
                if pending.done.is_set():
                    continue
                pending.error = normalized_error
                pending.done.set()
                released += 1
        if released:
            LOGGER.debug("Released %d pending remote tool waiters", released)
        return released

    def submit_result(self, call_id: str, *, output: dict[str, Any] | None, error: str | None) -> None:
        """Apply one returned tool result to the matching pending request."""

        try:
            normalized_call_id = self._normalize_call_id(call_id)
        except (TypeError, ValueError) as exc:
            LOGGER.warning("Discarding remote tool result with invalid call_id: %s", exc)
            return

        normalized_error = self._normalize_error(error)

        try:
            normalized_output = self._normalize_tool_output(output)
        except (TypeError, ValueError) as exc:
            LOGGER.warning(
                "Discarding invalid remote tool output payload for call_id=%s: %s",
                normalized_call_id,
                exc,
            )
            normalized_output = None
            normalized_error = normalized_error or "Remote tool returned an invalid output payload"

        with self._pending_lock:
            pending = self._pending.get(normalized_call_id)
            if pending is None:
                LOGGER.warning("Discarding stale or unknown remote tool result for call_id=%s", normalized_call_id)
                return
            if pending.done.is_set():
                LOGGER.warning("Discarding duplicate remote tool result for call_id=%s", normalized_call_id)
                return
            pending.output = normalized_output
            pending.error = normalized_error
            pending.done.set()

    def _make_handler(self, name: str) -> Callable[[AgentToolCall], Any]:
        """Build one forwarding handler for a named remote tool."""

        def _handler(tool_call: AgentToolCall) -> dict[str, Any]:
            call_id = self._normalize_call_id(tool_call.call_id)
            request = OrchestratorToolRequest(
                call_id=call_id,
                name=name,
                arguments=self._normalize_tool_arguments(tool_call.arguments),
            )
            pending = _PendingToolCall(request=request, done=Event())
            with self._pending_lock:
                if call_id in self._pending:
                    raise RuntimeError(f"Duplicate in-flight remote tool call_id: {call_id}")
                self._pending[call_id] = pending

            started_at = monotonic()
            try:
                payload = request.to_payload()
                # BREAKING: oversized / non-strict-JSON remote tool payloads now fail fast instead of risking transport breakage or Pi memory pressure.
                _enforce_json_payload_budget(
                    payload,
                    context=f"Remote tool request payload for '{name}'",
                    max_bytes=self._request_max_bytes,
                )
                self._emit_event(payload)
                if not pending.done.wait(timeout=self._tool_result_timeout_seconds):
                    raise RuntimeError(
                        f"Remote tool '{name}' did not return within {self._tool_result_timeout_seconds:.0f} seconds"
                    )
                if pending.error:
                    raise RuntimeError(pending.error)
                return pending.output or {}
            finally:
                elapsed_ms = (monotonic() - started_at) * 1000.0
                LOGGER.debug("Remote tool call finished name=%s call_id=%s latency_ms=%.1f", name, call_id, elapsed_ms)
                with self._pending_lock:
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
        self.tool_names = _normalize_tool_names(tool_names)
        self._turn_failure_text = _read_turn_failure_text()
        self._turn_lock = Lock()
        self._max_rounds = _read_positive_int_env(_MAX_ROUNDS_ENV, _DEFAULT_MAX_ROUNDS)

        # Frontier-upgrade: these values are static for the lifetime of a session, so precompute them once for lower Pi-side CPU/memory churn.
        self._tool_schemas = _upgrade_tool_schemas_for_frontier(build_agent_tool_schemas(self.tool_names))
        self._supervisor_instructions = build_supervisor_decision_instructions(
            self.config,
            extra_instructions=self.config.openai_realtime_instructions,
        )
        self._specialist_instructions = build_specialist_tool_agent_instructions(
            self.config,
            extra_instructions=self.config.openai_realtime_instructions,
        )

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

        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")

        _assert_not_running_on_event_loop_thread()

        with self._turn_lock:
            streamed_chunks: list[str] = []
            first_meaningful_delta = True

            def on_text_delta(delta: str) -> None:
                nonlocal first_meaningful_delta

                if not isinstance(delta, str):
                    raise TypeError("Text delta must be a string")

                cleaned = delta.strip()
                if cleaned and first_meaningful_delta:
                    try:
                        ack_id = ack_id_for_text(cleaned)
                    except Exception as exc:
                        LOGGER.warning(
                            "Ack classification failed; falling back to plain text delta (%s)",
                            type(exc).__name__,
                        )
                        ack_id = None
                    if ack_id is not None:
                        try:
                            emit_event(OrchestratorAckEvent(ack_id=ack_id, text=cleaned).to_payload())
                        except Exception as exc:
                            LOGGER.warning(
                                "Ack event emission failed; retrying as text delta (%s)",
                                type(exc).__name__,
                            )
                        else:
                            streamed_chunks.append(cleaned)
                            first_meaningful_delta = False
                            return

                emit_event(OrchestratorTextDeltaEvent(delta=delta).to_payload())
                streamed_chunks.append(delta)
                if cleaned:
                    first_meaningful_delta = False

            turn_started_at = monotonic()
            try:
                remote_handlers = tool_bridge.build_handlers(self.tool_names)
                tool_loop = DualLaneToolLoop(
                    supervisor_provider=_UnusedSupervisorProvider(),
                    specialist_provider=self.specialist_provider,
                    supervisor_decision_provider=self.supervisor_decision_provider,
                    tool_handlers=remote_handlers,
                    tool_schemas=self._tool_schemas,
                    supervisor_instructions=self._supervisor_instructions,
                    specialist_instructions=self._specialist_instructions,
                    max_rounds=self._max_rounds,
                )

                result = tool_loop.run(
                    prompt,
                    conversation=conversation,
                    supervisor_conversation=supervisor_conversation,
                    instructions=None,
                    allow_web_search=False,
                    on_text_delta=on_text_delta,
                )
                LOGGER.debug(
                    "Edge orchestrator turn completed rounds=%s tool_calls=%s latency_ms=%.1f",
                    result.rounds,
                    len(result.tool_calls),
                    (monotonic() - turn_started_at) * 1000.0,
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
                tool_bridge.fail_all_pending("Orchestrator turn aborted before tool results completed")
                LOGGER.error("Edge orchestrator turn failed (%s)", type(exc).__name__)
                try:
                    emit_event(OrchestratorTextDeltaEvent(delta=self._turn_failure_text).to_payload())
                except Exception as fallback_exc:
                    LOGGER.debug("Unable to emit fallback failure text (%s)", type(fallback_exc).__name__)
                return _build_error_turn_complete_event(
                    _compose_failure_text(streamed_chunks, self._turn_failure_text)
                )
