from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence
import json
import logging  # AUDIT-FIX(#1,#7,#13): Log guarded failures instead of letting provider/tool/callback errors crash the turn.
from uuid import uuid4  # AUDIT-FIX(#6): Generate unique fallback call IDs for deterministic tool/result correlation.

from twinr.agent.base_agent.contracts import (
    AgentToolCall,
    AgentToolResult,
    ConversationLike,
    SupervisorDecisionProvider,
    ToolCallingAgentProvider,
)
from twinr.agent.base_agent.personality import merge_instructions
from twinr.agent.tools.streaming_loop import StreamingToolLoopResult, ToolCallingStreamingLoop


logger = logging.getLogger(__name__)  # AUDIT-FIX(#1): Preserve root-cause visibility while returning senior-safe fallbacks.
_ALLOWED_HANDOFF_KINDS = frozenset({"general", "search", "memory", "automation"})  # AUDIT-FIX(#12): Normalize handoff kind across both supervisor paths.


_HANDOFF_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "name": "handoff_specialist_worker",
    "description": (
        "Use the slower specialist worker when the answer needs fresh web research, more synthesis, "
        "or a deeper multi-step tool pass than the fast supervisor should handle directly."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["general", "search", "memory", "automation"],
                "description": "Short handoff category that best describes why the specialist is needed.",
            },
            "goal": {
                "type": "string",
                "description": "Short description of what the specialist should achieve for this turn.",
            },
            "spoken_ack": {
                "type": "string",
                "description": (
                    "A short user-facing acknowledgement in the configured language that can be spoken immediately "
                    "before the specialist work starts, for example that Twinr is checking something now."
                ),
            },
            "prompt": {
                "type": "string",
                "description": "Optional rewritten task for the specialist worker. Omit to reuse the original user prompt.",
            },
            "allow_web_search": {
                "type": "boolean",
                "description": "Set true when the specialist should be allowed to use live web search.",
            },
        },
        "required": ["kind", "goal", "spoken_ack"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True, slots=True)
class _SpecialistRecord:
    result: StreamingToolLoopResult  # AUDIT-FIX(#14): Remove the unused always-empty trigger field so merge state cannot lie about provenance.


class DualLaneToolLoop:
    def __init__(
        self,
        *,
        supervisor_provider: ToolCallingAgentProvider,
        specialist_provider: ToolCallingAgentProvider,
        tool_handlers: dict[str, Callable[[dict[str, Any]], Any]],
        tool_schemas: Sequence[dict[str, Any]],
        supervisor_decision_provider: SupervisorDecisionProvider | None = None,
        supervisor_tool_handlers: dict[str, Callable[[dict[str, Any]], Any]] | None = None,
        supervisor_tool_schemas: Sequence[dict[str, Any]] | None = None,
        supervisor_instructions: str,
        specialist_instructions: str,
        max_rounds: int = 6,
        default_spoken_ack: str = "Einen Moment bitte.",  # AUDIT-FIX(#9): Keep current behaviour by default but allow locale-safe deployment configuration.
        default_end_reply: str = "Bis bald.",
        default_error_reply: str = "Das hat gerade nicht geklappt. Bitte versuche es noch einmal.",
    ) -> None:
        self.supervisor_provider = supervisor_provider
        self.specialist_provider = specialist_provider
        self.supervisor_decision_provider = supervisor_decision_provider
        self.tool_handlers = dict(tool_handlers)
        self.tool_schemas = tuple(tool_schemas)
        self._has_supervisor_tool_handlers_override = supervisor_tool_handlers is not None  # AUDIT-FIX(#3): Preserve whether the caller intentionally supplied an override, even when it is empty.
        self.supervisor_tool_handlers = dict(supervisor_tool_handlers) if supervisor_tool_handlers is not None else dict(tool_handlers)  # AUDIT-FIX(#3): Respect intentionally empty supervisor overrides.
        self.supervisor_tool_schemas = tuple(supervisor_tool_schemas) if supervisor_tool_schemas is not None else tuple(tool_schemas)  # AUDIT-FIX(#3): Respect intentionally empty supervisor schema overrides.
        self.supervisor_instructions = supervisor_instructions
        self.specialist_instructions = specialist_instructions
        if not isinstance(max_rounds, int) or max_rounds < 1:
            logger.warning("Invalid max_rounds=%r; defaulting to 1.", max_rounds)  # AUDIT-FIX(#11): Clamp invalid runtime config instead of entering undefined loop behaviour.
            max_rounds = 1
        self.max_rounds = max_rounds
        self.default_spoken_ack = _normalize_default_text(default_spoken_ack, "Einen Moment bitte.")  # AUDIT-FIX(#9): Remove hard-coded locale lock-in from user-facing fallbacks.
        self.default_end_reply = _normalize_default_text(default_end_reply, "Bis bald.")
        self.default_error_reply = _normalize_default_text(
            default_error_reply,
            "Das hat gerade nicht geklappt. Bitte versuche es noch einmal.",
        )

    def run(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        supervisor_conversation: ConversationLike | None = None,
        specialist_conversation: ConversationLike | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> StreamingToolLoopResult:
        specialist_records: list[_SpecialistRecord] = []
        supervisor_text_emitted = False  # AUDIT-FIX(#15): Track supervisor output with a boolean sentinel instead of storing and re-joining the full stream.
        resolved_supervisor_conversation = supervisor_conversation if supervisor_conversation is not None else conversation
        resolved_specialist_conversation = specialist_conversation if specialist_conversation is not None else conversation

        def _emit_user_text(text: str) -> None:
            _safe_emit_text_delta(on_text_delta, text)  # AUDIT-FIX(#1): Guard TTS/UI callback failures so they cannot abort the turn.

        def handoff_specialist_worker(arguments: dict[str, Any]) -> dict[str, Any]:
            normalized_arguments = _normalize_handoff_arguments(
                arguments,
                fallback_prompt=prompt,
                default_spoken_ack=self.default_spoken_ack,
            )  # AUDIT-FIX(#12): Validate and normalize supervisor-provided handoff payloads consistently.
            spoken_ack = normalized_arguments["spoken_ack"]
            if spoken_ack and not supervisor_text_emitted:
                _emit_user_text(spoken_ack)  # AUDIT-FIX(#8): Keep the senior informed even when the fast lane has not produced text yet.

            specialist_prompt = _strip_text(normalized_arguments.get("prompt")) or prompt
            try:
                specialist_result = ToolCallingStreamingLoop(
                    provider=self.specialist_provider,
                    tool_handlers=self.tool_handlers,
                    tool_schemas=self.tool_schemas,
                    max_rounds=self.max_rounds,
                ).run(
                    specialist_prompt,
                    conversation=resolved_specialist_conversation,
                    instructions=merge_instructions(
                        self.specialist_instructions,
                        instructions,
                        _specialist_handoff_context(normalized_arguments),
                    ),
                    allow_web_search=_handoff_allow_web_search(
                        normalized_arguments,
                        allow_web_search,
                    ),
                    on_text_delta=None,
                )
                status = "ok"
                error_code: str | None = None
            except Exception:
                logger.exception("Specialist handoff failed.")  # AUDIT-FIX(#1): Convert provider/tool failures into safe handoff error results.
                specialist_result = _make_loop_result(
                    text=self.default_error_reply,
                    rounds=0,
                    tool_calls=(),
                    tool_results=(),
                    response_id=None,
                    request_id=None,
                    model=None,
                    token_usage=None,
                    used_web_search=False,
                )  # AUDIT-FIX(#2): Always materialize a specialist result so downstream code never indexes an empty record list.
                status = "error"
                error_code = "specialist_worker_failed"

            specialist_records.append(_SpecialistRecord(result=specialist_result))
            output = {
                "status": status,
                "kind": normalized_arguments["kind"],
                "goal": normalized_arguments["goal"],
                "spoken_ack": spoken_ack,
                "answer_text": _strip_text(specialist_result.text),
                "used_web_search": bool(specialist_result.used_web_search),
                "tool_calls": len(specialist_result.tool_calls),
                "rounds": specialist_result.rounds,
            }
            if error_code is not None:
                output["error"] = error_code
            return output

        decision = None
        if self.supervisor_decision_provider is not None:
            try:
                decision = self.supervisor_decision_provider.decide(
                    prompt,
                    conversation=resolved_supervisor_conversation,
                    instructions=merge_instructions(self.supervisor_instructions, instructions),
                )
            except Exception:
                logger.exception(
                    "Supervisor decision provider failed; falling back to the supervisor loop."
                )  # AUDIT-FIX(#1): Do not let a decision-router failure take down the whole turn.

        if decision is not None:
            action = _normalize_decision_action(getattr(decision, "action", None))
            if action == "direct":
                reply = _strip_text(getattr(decision, "spoken_reply", None) or getattr(decision, "spoken_ack", None))
                if not reply:
                    reply = self.default_error_reply  # AUDIT-FIX(#8): Never return a silent direct turn to a senior user.
                _emit_user_text(reply)
                return _make_loop_result(
                    text=reply,
                    rounds=1,
                    tool_calls=(),
                    tool_results=(),
                    response_id=getattr(decision, "response_id", None),
                    request_id=getattr(decision, "request_id", None),
                    model=getattr(decision, "model", None),
                    token_usage=getattr(decision, "token_usage", None),
                    used_web_search=False,
                )

            if action == "end_conversation":
                end_handler = self.supervisor_tool_handlers.get("end_conversation")
                if end_handler is None and not self._has_supervisor_tool_handlers_override:
                    end_handler = self.tool_handlers.get("end_conversation")  # AUDIT-FIX(#3): Do not silently re-enable shared handlers when an empty supervisor override was intentional.
                end_result: Any = {"status": "ok"}
                reply = _strip_text(getattr(decision, "spoken_reply", None)) or self.default_end_reply
                try:
                    if end_handler is not None:
                        end_result = end_handler({})
                except Exception:
                    logger.exception("end_conversation handler failed.")
                    end_result = {"status": "error", "error": "end_conversation_failed"}
                    reply = self.default_error_reply
                call_id = _first_non_none(
                    getattr(decision, "response_id", None),
                    _make_call_id("end_conversation"),
                )  # AUDIT-FIX(#6): Ensure a unique fallback ID for tool/result pairing.
                _emit_user_text(reply)
                return _make_loop_result(
                    text=reply,
                    rounds=1,
                    tool_calls=(
                        AgentToolCall(
                            name="end_conversation",
                            call_id=call_id,
                            arguments={},
                            raw_arguments="{}",
                        ),
                    ),
                    tool_results=(
                        AgentToolResult(
                            call_id=call_id,
                            name="end_conversation",
                            output=end_result,
                            serialized_output=_safe_json_dumps(end_result),
                        ),
                    ),  # AUDIT-FIX(#7): Serialize arbitrary handler outputs safely.
                    response_id=getattr(decision, "response_id", None),
                    request_id=getattr(decision, "request_id", None),
                    model=getattr(decision, "model", None),
                    token_usage=getattr(decision, "token_usage", None),
                    used_web_search=False,
                )

            handoff_arguments = _normalize_handoff_arguments(
                {
                    "kind": getattr(decision, "kind", None),
                    "goal": getattr(decision, "goal", None),
                    "spoken_ack": getattr(decision, "spoken_ack", None),
                    "prompt": getattr(decision, "prompt", None),
                    "allow_web_search": getattr(decision, "allow_web_search", None),
                },
                fallback_prompt=prompt,
                default_spoken_ack=self.default_spoken_ack,
            )  # AUDIT-FIX(#4,#12): Normalize the handoff payload and preserve the original allow_web_search value for shared parsing.
            call_id = _first_non_none(
                getattr(decision, "response_id", None),
                _make_call_id("handoff_specialist_worker"),
            )  # AUDIT-FIX(#6): Ensure the synthetic handoff call ID cannot collide across turns.
            handoff_output = handoff_specialist_worker(handoff_arguments)
            specialist_result = specialist_records[-1].result  # AUDIT-FIX(#2): Safe because handoff_specialist_worker now always appends a record.
            final_text = _strip_text(specialist_result.text) or self.default_error_reply
            _emit_user_text(final_text)  # AUDIT-FIX(#8): Stream the specialist answer on the decision path instead of speaking only the acknowledgement.
            return _make_loop_result(
                text=final_text,
                rounds=1 + specialist_result.rounds,
                tool_calls=(
                    AgentToolCall(
                        name="handoff_specialist_worker",
                        call_id=call_id,
                        arguments=handoff_arguments,
                        raw_arguments=_safe_json_dumps(handoff_arguments),
                    ),
                    *specialist_result.tool_calls,
                ),  # AUDIT-FIX(#6): Persist valid raw tool arguments for auditing and deterministic replay.
                tool_results=(
                    AgentToolResult(
                        call_id=call_id,
                        name="handoff_specialist_worker",
                        output=handoff_output,
                        serialized_output=_safe_json_dumps(handoff_output),
                    ),
                    *specialist_result.tool_results,
                ),  # AUDIT-FIX(#7): Avoid secondary crashes when tool outputs are not JSON-serializable by default.
                response_id=_first_non_none(
                    specialist_result.response_id,
                    getattr(decision, "response_id", None),
                ),
                request_id=_first_non_none(
                    specialist_result.request_id,
                    getattr(decision, "request_id", None),
                ),
                model=_first_non_none(
                    specialist_result.model,
                    getattr(decision, "model", None),
                ),
                token_usage=_merge_token_usage(
                    getattr(decision, "token_usage", None),
                    specialist_result.token_usage,
                ),  # AUDIT-FIX(#10): Accumulate token usage instead of dropping specialist cost/accounting data.
                used_web_search=bool(specialist_result.used_web_search),
            )

        supervisor_handlers = dict(self.supervisor_tool_handlers)
        supervisor_handlers["handoff_specialist_worker"] = handoff_specialist_worker
        supervisor_loop = ToolCallingStreamingLoop(
            provider=self.supervisor_provider,
            tool_handlers=supervisor_handlers,
            tool_schemas=_merge_tool_schemas(
                self.supervisor_tool_schemas,
                (_HANDOFF_TOOL_SCHEMA,),
            ),
            max_rounds=self.max_rounds,
        )  # AUDIT-FIX(#13): De-duplicate tool schemas so the supervisor cannot receive the same handoff tool twice.

        def _forward_supervisor_delta(delta: str) -> None:
            nonlocal supervisor_text_emitted
            if _strip_text(delta):
                supervisor_text_emitted = True
            _safe_emit_text_delta(on_text_delta, delta)  # AUDIT-FIX(#1,#15): Keep streaming robust without buffering the full supervisor text in memory.

        try:
            supervisor_result = supervisor_loop.run(
                prompt,
                conversation=resolved_supervisor_conversation,
                instructions=merge_instructions(self.supervisor_instructions, instructions),
                allow_web_search=allow_web_search,
                on_text_delta=_forward_supervisor_delta,
            )
        except Exception:
            logger.exception("Supervisor loop failed.")  # AUDIT-FIX(#1): Return the best available fallback instead of crashing the request.
            fallback_text = self.default_error_reply
            used_web_search = False
            token_usage = None
            if specialist_records:
                latest_specialist_result = specialist_records[-1].result
                fallback_text = _strip_text(latest_specialist_result.text) or fallback_text
                used_web_search = bool(latest_specialist_result.used_web_search)
                token_usage = latest_specialist_result.token_usage
            if not supervisor_text_emitted:
                _emit_user_text(fallback_text)
            return _make_loop_result(
                text=fallback_text,
                rounds=1 + sum(record.result.rounds for record in specialist_records),
                tool_calls=tuple(
                    _merge_handoff_items((), specialist_records, "tool_calls")
                ),
                tool_results=tuple(
                    _merge_handoff_items((), specialist_records, "tool_results")
                ),
                response_id=_first_non_none(
                    *[record.result.response_id for record in specialist_records],
                ),
                request_id=_first_non_none(
                    *[record.result.request_id for record in specialist_records],
                ),
                model=_first_non_none(
                    *[record.result.model for record in specialist_records],
                ),
                token_usage=token_usage,
                used_web_search=used_web_search,
            )

        merged_tool_calls = _merge_handoff_items(
            supervisor_result.tool_calls,
            specialist_records,
            "tool_calls",
        )  # AUDIT-FIX(#5): Interleave specialist calls immediately after each handoff instead of appending them out of order.
        merged_tool_results = _merge_handoff_items(
            supervisor_result.tool_results,
            specialist_records,
            "tool_results",
        )  # AUDIT-FIX(#5): Preserve chronological tool/result order for multi-handoff turns.

        used_web_search = bool(supervisor_result.used_web_search)
        response_id = supervisor_result.response_id
        request_id = supervisor_result.request_id
        model = supervisor_result.model
        token_usage = supervisor_result.token_usage
        rounds = supervisor_result.rounds
        fallback_text = ""

        for record in specialist_records:
            fallback_text = _strip_text(record.result.text) or fallback_text
            used_web_search = used_web_search or bool(record.result.used_web_search)
            rounds += record.result.rounds
            response_id = _first_non_none(response_id, record.result.response_id)
            request_id = _first_non_none(request_id, record.result.request_id)
            model = _first_non_none(model, record.result.model)
            token_usage = _merge_token_usage(
                token_usage,
                record.result.token_usage,
            )  # AUDIT-FIX(#10): Aggregate specialist token usage into the final accounting result.

        final_text = _strip_text(supervisor_result.text) or fallback_text
        if not final_text:
            final_text = self.default_error_reply  # AUDIT-FIX(#8): Avoid silent empty completions on supervisor loop failures or blank model outputs.
        if final_text and not supervisor_text_emitted:
            _emit_user_text(final_text)

        return _make_loop_result(
            text=final_text,
            rounds=rounds,
            tool_calls=tuple(merged_tool_calls),
            tool_results=tuple(merged_tool_results),
            response_id=response_id,
            request_id=request_id,
            model=model,
            token_usage=token_usage,
            used_web_search=used_web_search,
        )


def _make_loop_result(
    *,
    text: str,
    rounds: int,
    tool_calls: Sequence[AgentToolCall],
    tool_results: Sequence[AgentToolResult],
    response_id: str | None,
    request_id: str | None,
    model: str | None,
    token_usage: Any,
    used_web_search: bool,
) -> StreamingToolLoopResult:
    return StreamingToolLoopResult(
        text=text,
        rounds=rounds,
        tool_calls=tuple(tool_calls),
        tool_results=tuple(tool_results),
        response_id=response_id,
        request_id=request_id,
        model=model,
        token_usage=token_usage,
        used_web_search=used_web_search,
    )


def _safe_emit_text_delta(
    on_text_delta: Callable[[str], None] | None,
    text: str,
) -> None:
    raw = _coerce_text(text)
    if not raw or on_text_delta is None:
        return
    try:
        on_text_delta(raw)
    except Exception:
        logger.exception("on_text_delta callback failed.")  # AUDIT-FIX(#1): Treat output-channel faults as non-fatal so the loop can still return a final result.


def _safe_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        logger.exception("Failed to serialize tool output to JSON.")  # AUDIT-FIX(#7): Prevent diagnostics serialization from crashing otherwise successful turns.
        return json.dumps(
            {"status": "serialization_error", "type": type(value).__name__},
            ensure_ascii=False,
        )


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _make_call_id(prefix: str) -> str:
    safe_prefix = _strip_text(prefix) or "tool"
    return f"{safe_prefix}_{uuid4().hex}"


def _merge_token_usage(base: Any, extra: Any) -> Any:
    if base is None:
        return extra
    if extra is None:
        return base
    if isinstance(base, (int, float)) and isinstance(extra, (int, float)):
        return base + extra
    if isinstance(base, dict) and isinstance(extra, dict):
        merged = dict(base)
        for key, value in extra.items():
            if key in merged:
                merged[key] = _merge_token_usage(merged[key], value)
            else:
                merged[key] = value
        return merged
    return base


def _merge_tool_schemas(
    primary: Sequence[dict[str, Any]],
    extra: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    merged: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for schema in (*primary, *extra):
        name = _tool_schema_name(schema)
        if name is not None:
            if name in seen_names:
                continue
            seen_names.add(name)
        merged.append(schema)
    return tuple(merged)


def _tool_schema_name(schema: dict[str, Any]) -> str | None:
    name = schema.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    function_schema = schema.get("function")
    if isinstance(function_schema, dict):
        function_name = function_schema.get("name")
        if isinstance(function_name, str) and function_name.strip():
            return function_name.strip()
    return None


def _merge_handoff_items(
    supervisor_items: Sequence[Any],
    specialist_records: Sequence[_SpecialistRecord],
    attribute: str,
) -> list[Any]:
    merged: list[Any] = []
    specialist_index = 0
    for item in supervisor_items:
        merged.append(item)
        if getattr(item, "name", None) == "handoff_specialist_worker":
            if specialist_index < len(specialist_records):
                merged.extend(getattr(specialist_records[specialist_index].result, attribute))
            else:
                logger.warning(
                    "Missing specialist record for supervisor handoff item %d.",
                    specialist_index,
                )
            specialist_index += 1

    if specialist_index < len(specialist_records):
        logger.warning(
            "Found %d specialist record(s) without a matching supervisor handoff item.",
            len(specialist_records) - specialist_index,
        )
        for record in specialist_records[specialist_index:]:
            merged.extend(getattr(record.result, attribute))
    return merged


def _normalize_decision_action(value: Any) -> str:
    normalized = _strip_text(value).lower()
    if normalized in {"direct", "end_conversation"}:
        return normalized
    return "handoff"


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _strip_text(value: Any) -> str:
    return _coerce_text(value).strip()


def _normalize_default_text(value: Any, fallback: str) -> str:
    normalized = _strip_text(value)
    if normalized:
        return normalized
    return fallback


def _normalize_handoff_kind(value: Any) -> str:
    normalized = _strip_text(value).lower()
    if normalized in _ALLOWED_HANDOFF_KINDS:
        return normalized
    return "general"


def _normalize_handoff_arguments(
    arguments: dict[str, Any],
    *,
    fallback_prompt: str,
    default_spoken_ack: str,
) -> dict[str, Any]:
    normalized: dict[str, Any] = {
        "kind": _normalize_handoff_kind(arguments.get("kind")),
        "goal": _strip_text(arguments.get("goal")) or fallback_prompt,
        "spoken_ack": _normalize_spoken_ack(arguments, default_spoken_ack),
    }
    prompt = _strip_text(arguments.get("prompt"))
    if prompt:
        normalized["prompt"] = prompt
    if "allow_web_search" in arguments:
        normalized["allow_web_search"] = arguments.get("allow_web_search")
    return normalized


def _handoff_allow_web_search(arguments: dict[str, Any], default: bool | None) -> bool | None:
    if "allow_web_search" not in arguments:
        return default
    value = arguments.get("allow_web_search")
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    normalized = _strip_text(value).lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _specialist_handoff_context(arguments: dict[str, Any]) -> str:
    goal = _strip_text(arguments.get("goal"))
    kind = _normalize_handoff_kind(arguments.get("kind"))
    prompt = _strip_text(arguments.get("prompt"))
    context_parts = [f"Specialist handoff kind: {kind}."]
    if goal:
        context_parts.append(f"Specialist goal: {goal}")
    if prompt:
        context_parts.append(f"Specialist rewritten prompt: {prompt}")
    return " ".join(context_parts)  # AUDIT-FIX(#16): This helper always returns a string, so the type now matches runtime behaviour.


def _normalize_spoken_ack(arguments: dict[str, Any], default: str) -> str:
    raw = _strip_text(arguments.get("spoken_ack"))
    if raw:
        return raw
    return default