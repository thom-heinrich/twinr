"""Run Twinr's dual-lane supervisor and specialist tool loop.

This module extends the generic streaming tool loop with a fast supervisor lane
and an optional specialist lane. It preserves the speech-lane semantics used by
the runtime speech output pipeline: quick filler acknowledgements can be spoken
first, then replaced atomically by a final answer once specialist work
finishes.
"""

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
    supervisor_decision_requires_full_context,
)
from twinr.agent.base_agent.prompting.personality import merge_instructions
from .streaming_loop import StreamingToolLoopResult, ToolCallingStreamingLoop


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
            "location_hint": {
                "type": "string",
                "description": (
                    "Optional explicit place already named by the user, for example a city, district, or street. "
                    "Use this for search handoffs when the target location matters."
                ),
            },
            "date_context": {
                "type": "string",
                "description": (
                    "Optional absolute date or local date/time context for search handoffs when the user referred "
                    "to relative dates such as today or tomorrow."
                ),
            },
        },
        "required": ["kind", "goal", "spoken_ack"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True, slots=True)
class _SpecialistRecord:
    """Store one specialist result for later chronological merging."""

    result: StreamingToolLoopResult  # AUDIT-FIX(#14): Remove the unused always-empty trigger field so merge state cannot lie about provenance.


@dataclass(frozen=True, slots=True)
class SpeechLaneDelta:
    """Describe one speech-output event emitted by the dual-lane loop.

    Attributes:
        text: Spoken text fragment to emit.
        lane: Output lane label such as ``direct``, ``filler``, or ``final``.
        replace_current: Whether this delta should replace the currently spoken
            filler content.
        atomic: Whether the delta should be treated as one indivisible segment.
    """

    text: str
    lane: str
    replace_current: bool = False
    atomic: bool = False


class DualLaneToolLoop:
    """Coordinate a fast supervisor lane with an optional specialist lane.

    The supervisor may answer directly, end the conversation, or hand work to a
    slower specialist loop. The class keeps the spoken acknowledgement flow
    bounded and merges all tool calls/results back into one chronological turn
    record for downstream consumers.
    """

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
        """Store providers, tool surfaces, and locale-safe fallback strings."""
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

    def resolve_supervisor_decision(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        prefetched_decision: Any | None = None,
        instructions: str | None = None,
    ) -> Any | None:
        """Resolve the supervisor decision, preferring prefetched state when present."""
        if prefetched_decision is not None:
            return prefetched_decision
        if self.supervisor_decision_provider is None:
            return None
        try:
            return self.supervisor_decision_provider.decide(
                prompt,
                conversation=conversation,
                instructions=merge_instructions(self.supervisor_instructions, instructions),
            )
        except Exception:
            logger.exception(
                "Supervisor decision provider failed; falling back to the supervisor loop."
            )
            return None

    def _run_direct_search_handoff(
        self,
        prompt: str,
        arguments: dict[str, Any],
    ) -> StreamingToolLoopResult:
        """Run the search handoff through the direct search handler shortcut."""
        search_handler = self.tool_handlers.get("search_live_info")
        if search_handler is None:
            raise RuntimeError("search_live_info handler is not configured")
        search_prompt = _strip_text(arguments.get("prompt")) or prompt
        call_id = _make_call_id("search_live_info")
        tool_arguments = {"question": search_prompt}
        location_hint = _strip_text(arguments.get("location_hint"))
        if location_hint:
            tool_arguments["location_hint"] = location_hint
        date_context = _strip_text(arguments.get("date_context"))
        if date_context:
            tool_arguments["date_context"] = date_context
        output = search_handler(tool_arguments)
        answer_text = _strip_text(
            output.get("answer")
            or output.get("spoken_answer")
            or output.get("text")
        )
        if not answer_text:
            raise RuntimeError("direct search handoff returned empty answer text")
        return _make_loop_result(
            text=answer_text,
            rounds=1,
            tool_calls=(
                AgentToolCall(
                    name="search_live_info",
                    call_id=call_id,
                    arguments=tool_arguments,
                    raw_arguments=_safe_json_dumps(tool_arguments),
                ),
            ),
            tool_results=(
                AgentToolResult(
                    call_id=call_id,
                    name="search_live_info",
                    output=output,
                    serialized_output=_safe_json_dumps(output),
                ),
            ),
            response_id=_strip_text(output.get("response_id")) or None,
            request_id=_strip_text(output.get("request_id")) or None,
            model=_strip_text(output.get("model")) or None,
            token_usage=output.get("token_usage"),
            used_web_search=bool(output.get("used_web_search", True)),
        )

    def _resolve_specialist_result(
        self,
        prompt: str,
        *,
        normalized_arguments: dict[str, Any],
        specialist_conversation: ConversationLike | None,
        instructions: str | None,
        allow_web_search: bool | None,
    ) -> StreamingToolLoopResult:
        """Resolve the specialist result through either direct search or the worker loop."""
        specialist_prompt = _strip_text(normalized_arguments.get("prompt")) or prompt
        if normalized_arguments["kind"] == "search":
            return self._run_direct_search_handoff(prompt, normalized_arguments)
        return ToolCallingStreamingLoop(
            provider=self.specialist_provider,
            tool_handlers=self.tool_handlers,
            tool_schemas=self.tool_schemas,
            max_rounds=self.max_rounds,
        ).run(
            specialist_prompt,
            conversation=specialist_conversation,
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

    def run_handoff_only(
        self,
        prompt: str,
        *,
        handoff: Any,
        conversation: ConversationLike | None = None,
        specialist_conversation: ConversationLike | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
        on_text_delta: Callable[[str], None] | None = None,
        on_lane_text_delta: Callable[[SpeechLaneDelta], None] | None = None,
        emit_filler: bool = True,
    ) -> StreamingToolLoopResult:
        """Run only the specialist side of a handoff.

        This path is used when a handoff has already been decided outside the
        normal supervisor loop, for example by a prefetched decision.
        """

        resolved_specialist_conversation = (
            specialist_conversation if specialist_conversation is not None else conversation
        )
        if isinstance(handoff, dict):
            raw_arguments = dict(handoff)
            response_id = raw_arguments.get("response_id")
            request_id = raw_arguments.get("request_id")
            model = raw_arguments.get("model")
            token_usage = raw_arguments.get("token_usage")
        else:
            raw_arguments = {
                "kind": getattr(handoff, "kind", None),
                "goal": getattr(handoff, "goal", None),
                "spoken_ack": getattr(handoff, "spoken_ack", None),
                "prompt": getattr(handoff, "prompt", None),
                "allow_web_search": getattr(handoff, "allow_web_search", None),
                "location_hint": getattr(handoff, "location_hint", None),
                "date_context": getattr(handoff, "date_context", None),
            }
            response_id = getattr(handoff, "response_id", None)
            request_id = getattr(handoff, "request_id", None)
            model = getattr(handoff, "model", None)
            token_usage = getattr(handoff, "token_usage", None)

        normalized_arguments = _normalize_handoff_arguments(
            raw_arguments,
            fallback_prompt=prompt,
            default_spoken_ack=self.default_spoken_ack,
        )

        def _emit_user_text(
            text: str,
            *,
            lane: str = "direct",
            replace_current: bool = False,
            atomic: bool = False,
        ) -> None:
            _safe_emit_speech_delta(
                on_lane_text_delta,
                on_text_delta,
                SpeechLaneDelta(
                    text=text,
                    lane=lane,
                    replace_current=replace_current,
                    atomic=atomic,
                ),
            )

        spoken_ack = normalized_arguments["spoken_ack"]
        if emit_filler and spoken_ack:
            _emit_user_text(
                spoken_ack,
                lane="filler",
            )

        try:
            specialist_result = self._resolve_specialist_result(
                prompt,
                normalized_arguments=normalized_arguments,
                specialist_conversation=resolved_specialist_conversation,
                instructions=instructions,
                allow_web_search=allow_web_search,
            )
            status = "ok"
            error_code: str | None = None
        except Exception:
            logger.exception("Specialist handoff failed.")
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
            )
            status = "error"
            error_code = "specialist_worker_failed"

        handoff_output = {
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
            handoff_output["error"] = error_code

        final_text = _strip_text(specialist_result.text) or self.default_error_reply
        if on_text_delta is not None or on_lane_text_delta is not None:
            _emit_user_text(
                final_text,
                lane="final",
                replace_current=emit_filler and bool(spoken_ack),
                atomic=True,
            )

        call_id = _first_non_none(
            response_id,
            _make_call_id("handoff_specialist_worker"),
        )
        return _make_loop_result(
            text=final_text,
            rounds=1 + specialist_result.rounds,
            tool_calls=(
                AgentToolCall(
                    name="handoff_specialist_worker",
                    call_id=call_id,
                    arguments=normalized_arguments,
                    raw_arguments=_safe_json_dumps(normalized_arguments),
                ),
                *specialist_result.tool_calls,
            ),
            tool_results=(
                AgentToolResult(
                    call_id=call_id,
                    name="handoff_specialist_worker",
                    output=handoff_output,
                    serialized_output=_safe_json_dumps(handoff_output),
                ),
                *specialist_result.tool_results,
            ),
            response_id=_first_non_none(specialist_result.response_id, response_id),
            request_id=_first_non_none(specialist_result.request_id, request_id),
            model=_first_non_none(specialist_result.model, model),
            token_usage=_merge_token_usage(token_usage, specialist_result.token_usage),
            used_web_search=bool(specialist_result.used_web_search),
        )

    def run(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        supervisor_conversation: ConversationLike | None = None,
        specialist_conversation: ConversationLike | None = None,
        prefetched_decision: Any | None = None,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
        on_text_delta: Callable[[str], None] | None = None,
        on_lane_text_delta: Callable[[SpeechLaneDelta], None] | None = None,
    ) -> StreamingToolLoopResult:
        """Run the full supervisor/specialist loop for one turn.

        The supervisor may answer directly, emit an ``end_conversation`` tool
        result, or call the synthetic ``handoff_specialist_worker`` tool to
        delegate work to the specialist lane.
        """

        specialist_records: list[_SpecialistRecord] = []
        supervisor_text_emitted = False  # AUDIT-FIX(#15): Track supervisor output with a boolean sentinel instead of storing and re-joining the full stream.
        resolved_supervisor_conversation = supervisor_conversation if supervisor_conversation is not None else conversation
        resolved_specialist_conversation = specialist_conversation if specialist_conversation is not None else conversation

        def _emit_user_text(
            text: str,
            *,
            lane: str = "direct",
            replace_current: bool = False,
            atomic: bool = False,
        ) -> None:
            _safe_emit_speech_delta(
                on_lane_text_delta,
                on_text_delta,
                SpeechLaneDelta(
                    text=text,
                    lane=lane,
                    replace_current=replace_current,
                    atomic=atomic,
                ),
            )  # AUDIT-FIX(#1): Guard TTS/UI callback failures so they cannot abort the turn.

        def handoff_specialist_worker(arguments: dict[str, Any]) -> dict[str, Any]:
            normalized_arguments = _normalize_handoff_arguments(
                arguments,
                fallback_prompt=prompt,
                default_spoken_ack=self.default_spoken_ack,
            )  # AUDIT-FIX(#12): Validate and normalize supervisor-provided handoff payloads consistently.
            spoken_ack = normalized_arguments["spoken_ack"]
            if spoken_ack and not supervisor_text_emitted:
                _emit_user_text(
                    spoken_ack,
                    lane="filler",
                )  # AUDIT-FIX(#8): Keep the senior informed even when the fast lane has not produced text yet.

            try:
                specialist_result = self._resolve_specialist_result(
                    prompt,
                    normalized_arguments=normalized_arguments,
                    specialist_conversation=resolved_specialist_conversation,
                    instructions=instructions,
                    allow_web_search=allow_web_search,
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

        decision = self.resolve_supervisor_decision(
            prompt,
            conversation=resolved_supervisor_conversation,
            prefetched_decision=prefetched_decision,
            instructions=instructions,
        )

        if decision is not None:
            action = _normalize_decision_action(getattr(decision, "action", None))
            if action == "direct" and supervisor_decision_requires_full_context(decision):
                return self.run_handoff_only(
                    prompt,
                    handoff=_decision_fallback_handoff(
                        decision,
                        prompt=prompt,
                        default_spoken_ack=self.default_spoken_ack,
                    ),
                    conversation=conversation,
                    specialist_conversation=resolved_specialist_conversation,
                    instructions=instructions,
                    allow_web_search=allow_web_search,
                    on_text_delta=on_text_delta,
                    on_lane_text_delta=on_lane_text_delta,
                    emit_filler=True,
                )
            if action == "direct":
                reply = _strip_text(getattr(decision, "spoken_reply", None) or getattr(decision, "spoken_ack", None))
                if not reply:
                    reply = self.default_error_reply  # AUDIT-FIX(#8): Never return a silent direct turn to a senior user.
                _emit_user_text(reply, lane="direct")
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
                _emit_user_text(reply, lane="direct")
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

            return self.run_handoff_only(
                prompt,
                handoff=decision,
                conversation=conversation,
                specialist_conversation=resolved_specialist_conversation,
                instructions=instructions,
                allow_web_search=allow_web_search,
                on_text_delta=on_text_delta,
                on_lane_text_delta=on_lane_text_delta,
                emit_filler=True,
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
    """Build an immutable loop result from the collected turn state."""
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
    """Emit a plain text delta while swallowing callback failures."""
    raw = _coerce_text(text)
    if not raw or on_text_delta is None:
        return
    try:
        on_text_delta(raw)
    except Exception:
        logger.exception("on_text_delta callback failed.")  # AUDIT-FIX(#1): Treat output-channel faults as non-fatal so the loop can still return a final result.


def _safe_emit_speech_delta(
    on_lane_text_delta: Callable[[SpeechLaneDelta], None] | None,
    on_text_delta: Callable[[str], None] | None,
    delta: SpeechLaneDelta,
) -> None:
    """Prefer lane-aware speech emission and fall back to plain text callbacks."""
    raw = _coerce_text(delta.text)
    if not raw:
        return
    if on_lane_text_delta is not None:
        try:
            on_lane_text_delta(
                SpeechLaneDelta(
                    text=raw,
                    lane=_strip_text(delta.lane) or "direct",
                    replace_current=bool(delta.replace_current),
                    atomic=bool(delta.atomic),
                )
            )
            return
        except Exception:
            logger.exception("on_lane_text_delta callback failed.")
    _safe_emit_text_delta(on_text_delta, raw)


def _safe_json_dumps(value: Any) -> str:
    """Serialize arbitrary diagnostic payloads into JSON for tool envelopes."""
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        logger.exception("Failed to serialize tool output to JSON.")  # AUDIT-FIX(#7): Prevent diagnostics serialization from crashing otherwise successful turns.
        return json.dumps(
            {"status": "serialization_error", "type": type(value).__name__},
            ensure_ascii=False,
        )


def _first_non_none(*values: Any) -> Any:
    """Return the first non-``None`` value from the provided candidates."""
    for value in values:
        if value is not None:
            return value
    return None


def _make_call_id(prefix: str) -> str:
    """Create a deterministic-looking unique tool call identifier."""
    safe_prefix = _strip_text(prefix) or "tool"
    return f"{safe_prefix}_{uuid4().hex}"


def _merge_token_usage(base: Any, extra: Any) -> Any:
    """Merge provider token-usage payloads across supervisor and specialist work."""
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
    """Merge tool schemas while de-duplicating by declared tool name."""
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
    """Extract a tool name from either accepted schema format."""
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
    """Interleave specialist items immediately after their handoff marker."""
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
    """Normalize the supervisor action into one of Twinr's supported modes."""
    normalized = _strip_text(value).lower()
    if normalized in {"direct", "end_conversation"}:
        return normalized
    return "handoff"


def _coerce_text(value: Any) -> str:
    """Coerce arbitrary callback payloads into text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _strip_text(value: Any) -> str:
    """Coerce text and strip surrounding whitespace."""
    return _coerce_text(value).strip()


def _normalize_default_text(value: Any, fallback: str) -> str:
    """Return a stripped non-empty string or the configured fallback."""
    normalized = _strip_text(value)
    if normalized:
        return normalized
    return fallback


def _normalize_handoff_kind(value: Any) -> str:
    """Normalize the handoff kind to the supported specialist categories."""
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
    """Normalize a supervisor handoff payload into the canonical shape."""
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
    location_hint = _strip_text(arguments.get("location_hint"))
    if location_hint:
        normalized["location_hint"] = location_hint
    date_context = _strip_text(arguments.get("date_context"))
    if date_context:
        normalized["date_context"] = date_context
    return normalized


def _handoff_allow_web_search(arguments: dict[str, Any], default: bool | None) -> bool | None:
    """Resolve the handoff web-search flag from booleans or common string forms."""
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
    """Build a short instruction suffix that explains the handoff intent."""
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
    """Return the spoken acknowledgement or a configured default."""
    raw = _strip_text(arguments.get("spoken_ack"))
    if raw:
        return raw
    return default


def _decision_fallback_handoff(
    decision: Any,
    *,
    prompt: str,
    default_spoken_ack: str,
) -> dict[str, Any]:
    """Convert a full-context direct decision into a safe handoff payload."""

    kind = _normalize_handoff_kind(getattr(decision, "kind", None))
    if kind == "general":
        kind = "memory"
    fallback_arguments = {
        "kind": kind,
        "goal": _strip_text(getattr(decision, "goal", None)) or prompt,
        "spoken_ack": _strip_text(getattr(decision, "spoken_ack", None)) or default_spoken_ack,
        "prompt": _strip_text(getattr(decision, "prompt", None)),
        "allow_web_search": getattr(decision, "allow_web_search", None),
        "location_hint": _strip_text(getattr(decision, "location_hint", None)),
        "date_context": _strip_text(getattr(decision, "date_context", None)),
        "response_id": getattr(decision, "response_id", None),
        "request_id": getattr(decision, "request_id", None),
        "model": getattr(decision, "model", None),
        "token_usage": getattr(decision, "token_usage", None),
    }
    return fallback_arguments
