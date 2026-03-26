"""Run Twinr's dual-lane supervisor and specialist tool loop.

This module extends the generic streaming tool loop with a fast supervisor lane
and an optional specialist lane. It preserves the speech-lane semantics used by
the runtime speech output pipeline: quick filler acknowledgements can be spoken
first, then replaced atomically by a final answer once specialist work
finishes.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Sequence
import logging  # AUDIT-FIX(#1,#7,#13): Log guarded failures instead of letting provider/tool/callback errors crash the turn.

from twinr.agent.base_agent.contracts import (
    AgentToolCall,
    AgentToolResult,
    ConversationLike,
    FirstWordProvider,
    SupervisorDecisionProvider,
    ToolCallingAgentProvider,
    supervisor_decision_requires_full_context,
)
from twinr.agent.base_agent.prompting.personality import merge_instructions
from .forensics import (
    conversation_summary as _conversation_summary,
    decision_summary as _decision_summary,
    handoff_summary as _handoff_summary,
    loop_result_summary as _loop_result_summary,
    text_summary as _text_summary,
)
from .handoff import (
    _HANDOFF_TOOL_SCHEMA,
    decision_fallback_handoff as _decision_fallback_handoff,
    handoff_allow_web_search as _handoff_allow_web_search,
    merge_handoff_items as _merge_handoff_items,
    merge_tool_schemas as _merge_tool_schemas,
    normalize_decision_action as _normalize_decision_action,
    normalize_handoff_arguments as _normalize_handoff_arguments,
    specialist_handoff_context as _specialist_handoff_context,
    supervisor_decision_has_required_user_reply as _supervisor_decision_has_required_user_reply,
)
from .loop_support import (
    first_non_none as _first_non_none,
    make_call_id as _make_call_id,
    make_loop_result as _make_loop_result,
    merge_token_usage as _merge_token_usage,
    raise_if_should_stop,
    safe_json_dumps as _safe_json_dumps,
    strip_text as _strip_text,
)
from .recovery_reply import LLMRecoveryResponder
from .runtime_local_handoff import (
    decision_runtime_local_tool_call as _decision_runtime_local_tool_call,
    runtime_local_tool_reply_text as _runtime_local_tool_reply_text,
)
from .speech_lane import (
    SpeechLaneDelta,
    safe_emit_speech_delta as _safe_emit_speech_delta,
    safe_emit_text_delta as _safe_emit_text_delta,
)
from .streaming_loop import StreamingToolLoopResult, ToolCallingStreamingLoop, ToolHandler


logger = logging.getLogger(__name__)  # AUDIT-FIX(#1): Preserve root-cause visibility while returning senior-safe fallbacks.
_raise_if_should_stop = partial(raise_if_should_stop, actor="dual-lane work")


@dataclass(frozen=True, slots=True)
class _SpecialistRecord:
    """Store one specialist result for later chronological merging."""

    result: StreamingToolLoopResult  # AUDIT-FIX(#14): Remove the unused always-empty trigger field so merge state cannot lie about provenance.


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
        tool_handlers: dict[str, ToolHandler],
        tool_schemas: Sequence[dict[str, Any]],
        supervisor_decision_provider: SupervisorDecisionProvider | None = None,
        first_word_provider: FirstWordProvider | None = None,
        supervisor_tool_handlers: dict[str, ToolHandler] | None = None,
        supervisor_tool_schemas: Sequence[dict[str, Any]] | None = None,
        supervisor_instructions: str,
        specialist_instructions: str,
        max_rounds: int = 6,
        trace_event: Callable[..., None] | None = None,
        trace_decision: Callable[..., None] | None = None,
    ) -> None:
        """Store providers, tool surfaces, and LLM-only recovery state."""
        self.supervisor_provider = supervisor_provider
        self.specialist_provider = specialist_provider
        self.supervisor_decision_provider = supervisor_decision_provider
        self.first_word_provider = first_word_provider
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
        self._trace_event_callback = trace_event
        self._trace_decision_callback = trace_decision
        self._recovery_responder = LLMRecoveryResponder(
            supervisor_decision_provider=self.supervisor_decision_provider,
            first_word_provider=self.first_word_provider,
            supervisor_instructions=self.supervisor_instructions,
        )

    def _trace_event(
        self,
        name: str,
        *,
        kind: str,
        details: dict[str, Any] | None = None,
        level: str | None = None,
        kpi: dict[str, Any] | None = None,
    ) -> None:
        """Emit a best-effort forensic event without exposing raw user text."""

        callback = self._trace_event_callback
        if callback is None:
            return
        try:
            callback(
                name,
                kind=kind,
                details=dict(details or {}),
                level=level,
                kpi=kpi,
            )
        except Exception:
            logger.exception("dual-lane trace_event callback failed.")

    def _trace_decision(
        self,
        name: str,
        *,
        question: str,
        selected: dict[str, Any],
        options: Sequence[dict[str, Any]],
        context: dict[str, Any] | None = None,
        guardrails: Sequence[str] | None = None,
    ) -> None:
        """Emit a best-effort forensic decision without aborting the turn."""

        callback = self._trace_decision_callback
        if callback is None:
            return
        try:
            callback(
                name,
                question=question,
                selected=dict(selected),
                options=[dict(option) for option in options],
                context=dict(context or {}),
                guardrails=list(guardrails or ()),
            )
        except Exception:
            logger.exception("dual-lane trace_decision callback failed.")

    def resolve_supervisor_decision(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        prefetched_decision: Any | None = None,
        instructions: str | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> Any | None:
        """Resolve the supervisor decision, preferring prefetched state when present."""
        _raise_if_should_stop(should_stop, context="supervisor_decision_start")
        if prefetched_decision is not None:
            if _supervisor_decision_has_required_user_reply(prefetched_decision):
                self._trace_decision(
                    "dual_lane_supervisor_decision_resolved",
                    question="Which supervisor decision is active for this turn?",
                    selected={
                        "id": _normalize_decision_action(getattr(prefetched_decision, "action", None)),
                        "summary": "Reuse prefetched supervisor decision",
                    },
                    options=[
                        {"id": "prefetched", "summary": "Reuse prefetched decision"},
                        {"id": "provider", "summary": "Resolve decision from provider"},
                        {"id": "none", "summary": "Fall back to supervisor loop"},
                    ],
                    context={
                        "source": "prefetched",
                        "decision": _decision_summary(prefetched_decision),
                        "conversation": _conversation_summary(conversation),
                        "prompt": _text_summary(prompt),
                    },
                )
                return prefetched_decision
            logger.warning(
                "Ignoring prefetched supervisor decision without the required user-facing reply field."
            )
            self._trace_event(
                "dual_lane_prefetched_decision_rejected",
                kind="exception",
                level="WARN",
                details={
                    "reason": "missing_user_reply",
                    "decision": _decision_summary(prefetched_decision),
                    "conversation": _conversation_summary(conversation),
                    "prompt": _text_summary(prompt),
                },
            )
        if self.supervisor_decision_provider is None:
            return None
        try:
            _raise_if_should_stop(should_stop, context="supervisor_decision_provider_call")
            decision = self.supervisor_decision_provider.decide(
                prompt,
                conversation=conversation,
                instructions=merge_instructions(self.supervisor_instructions, instructions),
            )
            _raise_if_should_stop(should_stop, context="supervisor_decision_provider_return")
            if not _supervisor_decision_has_required_user_reply(decision):
                raise ValueError("Supervisor decision omitted the required user-facing reply field.")
            self._trace_decision(
                "dual_lane_supervisor_decision_resolved",
                question="Which supervisor decision is active for this turn?",
                selected={
                    "id": _normalize_decision_action(getattr(decision, "action", None)),
                    "summary": "Use provider-resolved supervisor decision",
                },
                options=[
                    {"id": "prefetched", "summary": "Reuse prefetched decision"},
                    {"id": "provider", "summary": "Resolve decision from provider"},
                    {"id": "none", "summary": "Fall back to supervisor loop"},
                ],
                context={
                    "source": "provider",
                    "decision": _decision_summary(decision),
                    "conversation": _conversation_summary(conversation),
                    "prompt": _text_summary(prompt),
                },
            )
            return decision
        except InterruptedError:
            raise
        except Exception as exc:
            logger.exception(
                "Supervisor decision provider failed; falling back to the supervisor loop."
            )
            self._trace_event(
                "dual_lane_supervisor_decision_failed",
                kind="exception",
                level="WARN",
                details={
                    "error_type": type(exc).__name__,
                    "conversation": _conversation_summary(conversation),
                    "prompt": _text_summary(prompt),
                },
            )
            return None

    def recover_with_llm(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        instructions: str | None = None,
        failure_reason: str,
        rounds: int = 1,
        tool_calls: Sequence[AgentToolCall] = (),
        tool_results: Sequence[AgentToolResult] = (),
        used_web_search: bool = False,
    ) -> StreamingToolLoopResult:
        """Generate a recovery reply through LLM lanes only."""

        reply = self._recovery_responder.recover(
            prompt,
            conversation=conversation,
            instructions=instructions,
            failure_reason=failure_reason,
        )
        return _make_loop_result(
            text=reply.text,
            rounds=max(1, int(rounds)),
            tool_calls=tuple(tool_calls),
            tool_results=tuple(tool_results),
            response_id=reply.response_id,
            request_id=reply.request_id,
            model=reply.model,
            token_usage=reply.token_usage,
            used_web_search=bool(used_web_search),
        )

    def _run_direct_search_handoff(
        self,
        prompt: str,
        arguments: dict[str, Any],
        *,
        should_stop: Callable[[], bool] | None = None,
    ) -> StreamingToolLoopResult:
        """Run the search handoff through the direct search handler shortcut."""
        _raise_if_should_stop(should_stop, context="direct_search_handoff_start")
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
        if getattr(search_handler, "_twinr_accepts_tool_call", False):
            output = search_handler(
                AgentToolCall(
                    name="search_live_info",
                    call_id=call_id,
                    arguments=tool_arguments,
                    raw_arguments=_safe_json_dumps(tool_arguments),
                )
            )
        else:
            output = search_handler(tool_arguments)
        _raise_if_should_stop(should_stop, context="direct_search_handoff_after_handler")
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
        should_stop: Callable[[], bool] | None = None,
    ) -> StreamingToolLoopResult:
        """Resolve the specialist result through either direct search or the worker loop."""
        _raise_if_should_stop(should_stop, context="specialist_resolution_start")
        specialist_prompt = _strip_text(normalized_arguments.get("prompt")) or prompt
        if normalized_arguments["kind"] == "search" and "browser_automation" not in self.tool_handlers:
            return self._run_direct_search_handoff(
                prompt,
                normalized_arguments,
                should_stop=should_stop,
            )
        specialist_allow_web_search = _handoff_allow_web_search(
            normalized_arguments,
            allow_web_search,
        )
        if (
            normalized_arguments["kind"] == "search"
            and "browser_automation" in self.tool_handlers
            and "search_live_info" in self.tool_handlers
        ):
            # Keep generic web research and browser work on explicit Twinr tools
            # once the optional browser lane is available. Otherwise the provider's
            # own web-search mode can bypass the intended search-vs-browser
            # boundary and skip the permission step entirely.
            specialist_allow_web_search = False
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
            allow_web_search=specialist_allow_web_search,
            on_text_delta=None,
            should_stop=should_stop,
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
        should_stop: Callable[[], bool] | None = None,
    ) -> StreamingToolLoopResult:
        """Run only the specialist side of a handoff.

        This path is used when a handoff has already been decided outside the
        normal supervisor loop, for example by a prefetched decision.
        """

        resolved_specialist_conversation = (
            specialist_conversation if specialist_conversation is not None else conversation
        )
        _raise_if_should_stop(should_stop, context="handoff_only_start")
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
        )
        self._trace_event(
            "dual_lane_handoff_only_started",
            kind="branch",
            details={
                "handoff": _handoff_summary(normalized_arguments),
                "conversation": _conversation_summary(conversation),
                "specialist_conversation": _conversation_summary(resolved_specialist_conversation),
                "prompt": _text_summary(prompt),
                "emit_filler": bool(emit_filler),
            },
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
                should_stop=should_stop,
            )
        except InterruptedError:
            raise
        except Exception as exc:
            logger.exception("Specialist handoff failed.")
            self._trace_event(
                "dual_lane_handoff_only_specialist_failed",
                kind="exception",
                level="ERROR",
                details={
                    "error_type": type(exc).__name__,
                    "handoff": _handoff_summary(normalized_arguments),
                    "specialist_conversation": _conversation_summary(resolved_specialist_conversation),
                },
            )
            raise
        _raise_if_should_stop(should_stop, context="handoff_only_after_specialist")
        status = "ok"
        error_code: str | None = None

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

        final_text = _strip_text(specialist_result.text)
        if on_text_delta is not None or on_lane_text_delta is not None:
            _emit_user_text(
                final_text,
                lane="final",
                replace_current=emit_filler and bool(spoken_ack),
                atomic=True,
            )
        self._trace_event(
            "dual_lane_handoff_only_completed",
            kind="observation",
            details={
                "status": status,
                "handoff": _handoff_summary(normalized_arguments),
                "result": _loop_result_summary(specialist_result),
                "final_text": _text_summary(final_text),
                "replaced_filler": bool(emit_filler and spoken_ack),
            },
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

    def run_runtime_local_tool_only(
        self,
        prompt: str,
        *,
        decision: Any,
        on_text_delta: Callable[[str], None] | None = None,
        on_lane_text_delta: Callable[[SpeechLaneDelta], None] | None = None,
        emit_filler: bool = True,
        should_stop: Callable[[], bool] | None = None,
    ) -> StreamingToolLoopResult:
        """Execute one supervisor-resolved runtime-local tool without a specialist hop."""

        resolved_tool_call = _decision_runtime_local_tool_call(decision)
        if resolved_tool_call is None:
            raise ValueError("runtime-local tool handoff requires runtime_tool_name and runtime_tool_arguments")
        tool_name, tool_arguments = resolved_tool_call
        handler = self.tool_handlers.get(tool_name)
        if handler is None:
            raise ValueError(f"runtime-local tool handoff requested unavailable tool: {tool_name}")
        _raise_if_should_stop(should_stop, context="runtime_local_tool_start")
        self._trace_event(
            "dual_lane_runtime_local_tool_started",
            kind="branch",
            details={
                "prompt": _text_summary(prompt),
                "decision": _decision_summary(decision),
                "tool_name": tool_name,
                "argument_keys": sorted(tool_arguments.keys()),
                "emit_filler": bool(emit_filler),
            },
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

        spoken_ack = _strip_text(getattr(decision, "spoken_ack", None))
        if emit_filler and spoken_ack:
            _emit_user_text(spoken_ack, lane="filler")

        try:
            tool_output = handler(tool_arguments)
        except InterruptedError:
            raise
        except Exception as exc:
            logger.exception("Runtime-local tool handoff failed.")
            self._trace_event(
                "dual_lane_runtime_local_tool_failed",
                kind="exception",
                level="ERROR",
                details={
                    "error_type": type(exc).__name__,
                    "decision": _decision_summary(decision),
                    "tool_name": tool_name,
                    "argument_keys": sorted(tool_arguments.keys()),
                },
            )
            raise
        _raise_if_should_stop(should_stop, context="runtime_local_tool_after_handler")

        final_text = _runtime_local_tool_reply_text(tool_output) or spoken_ack
        if final_text and (on_text_delta is not None or on_lane_text_delta is not None):
            _emit_user_text(
                final_text,
                lane="final",
                replace_current=emit_filler and bool(spoken_ack),
                atomic=True,
            )
        self._trace_event(
            "dual_lane_runtime_local_tool_completed",
            kind="observation",
            details={
                "decision": _decision_summary(decision),
                "tool_name": tool_name,
                "final_text": _text_summary(final_text),
                "output_present": tool_output is not None,
            },
        )
        call_id = _first_non_none(
            getattr(decision, "response_id", None),
            _make_call_id(tool_name),
        )
        return _make_loop_result(
            text=final_text,
            rounds=1,
            tool_calls=(
                AgentToolCall(
                    name=tool_name,
                    call_id=call_id,
                    arguments=tool_arguments,
                    raw_arguments=_safe_json_dumps(tool_arguments),
                ),
            ),
            tool_results=(
                AgentToolResult(
                    call_id=call_id,
                    name=tool_name,
                    output=tool_output,
                    serialized_output=_safe_json_dumps(tool_output),
                ),
            ),
            response_id=getattr(decision, "response_id", None),
            request_id=getattr(decision, "request_id", None),
            model=getattr(decision, "model", None),
            token_usage=getattr(decision, "token_usage", None),
            used_web_search=False,
        )

    def run(
        self,
        prompt: str,
        *,
        conversation: ConversationLike | None = None,
        supervisor_conversation: ConversationLike | None = None,
        specialist_conversation: ConversationLike | None = None,
        prefetched_decision: Any | None = None,
        skip_supervisor_decision: bool = False,
        instructions: str | None = None,
        allow_web_search: bool | None = None,
        on_text_delta: Callable[[str], None] | None = None,
        on_lane_text_delta: Callable[[SpeechLaneDelta], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
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
        _raise_if_should_stop(should_stop, context="dual_lane_run_start")

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

        def handoff_specialist_worker(arguments: dict[str, Any] | AgentToolCall) -> dict[str, Any]:
            _raise_if_should_stop(should_stop, context="handoff_worker_start")
            normalized_arguments = _normalize_handoff_arguments(
                arguments.arguments if isinstance(arguments, AgentToolCall) else arguments,
                fallback_prompt=prompt,
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
                    should_stop=should_stop,
                )
            except InterruptedError:
                raise
            except Exception as exc:
                logger.exception("Specialist handoff failed.")
                self._trace_event(
                    "dual_lane_supervisor_handoff_specialist_failed",
                    kind="exception",
                    level="ERROR",
                    details={
                        "error_type": type(exc).__name__,
                        "handoff": _handoff_summary(normalized_arguments),
                        "specialist_conversation": _conversation_summary(resolved_specialist_conversation),
                    },
                )
                raise
            status = "ok"
            error_code: str | None = None

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
        if not skip_supervisor_decision:
            decision = self.resolve_supervisor_decision(
                prompt,
                conversation=resolved_supervisor_conversation,
                prefetched_decision=prefetched_decision,
                instructions=instructions,
                should_stop=should_stop,
            )
        _raise_if_should_stop(should_stop, context="dual_lane_post_decision")

        if decision is not None:
            action = _normalize_decision_action(getattr(decision, "action", None))
            if action == "direct" and supervisor_decision_requires_full_context(decision):
                fallback_handoff = _decision_fallback_handoff(
                    decision,
                    prompt=prompt,
                )
                self._trace_decision(
                    "dual_lane_direct_downgraded_to_handoff",
                    question="Why was a direct supervisor answer not spoken directly?",
                    selected={
                        "id": "handoff_with_full_context",
                        "summary": "The direct answer required the full tool/memory context",
                    },
                    options=[
                        {"id": "direct", "summary": "Speak the direct supervisor answer"},
                        {"id": "handoff_with_full_context", "summary": "Re-run through the specialist with full context"},
                    ],
                    context={
                        "decision": _decision_summary(decision),
                        "fallback_handoff": _handoff_summary(fallback_handoff),
                        "conversation": _conversation_summary(conversation),
                        "specialist_conversation": _conversation_summary(resolved_specialist_conversation),
                    },
                    guardrails=["preserve_full_context"],
                )
                return self.run_handoff_only(
                    prompt,
                    handoff=fallback_handoff,
                    conversation=conversation,
                    specialist_conversation=resolved_specialist_conversation,
                    instructions=instructions,
                    allow_web_search=allow_web_search,
                    on_text_delta=on_text_delta,
                    on_lane_text_delta=on_lane_text_delta,
                    emit_filler=True,
                    should_stop=should_stop,
                )
            if action == "direct":
                reply = _strip_text(getattr(decision, "spoken_reply", None))
                self._trace_event(
                    "dual_lane_direct_reply_selected",
                    kind="decision",
                    details={
                        "decision": _decision_summary(decision),
                        "reply": _text_summary(reply),
                        "conversation": _conversation_summary(resolved_supervisor_conversation),
                    },
                )
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
                reply = _strip_text(getattr(decision, "spoken_reply", None))
                try:
                    if end_handler is not None:
                        end_result = end_handler({})
                except Exception as exc:
                    logger.exception("end_conversation handler failed.")
                    self._trace_event(
                        "dual_lane_end_conversation_handler_failed",
                        kind="exception",
                        level="ERROR",
                        details={
                            "error_type": type(exc).__name__,
                            "decision": _decision_summary(decision),
                            "conversation": _conversation_summary(resolved_supervisor_conversation),
                        },
                    )
                    raise
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
                should_stop=should_stop,
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
                should_stop=should_stop,
            )
            _raise_if_should_stop(should_stop, context="dual_lane_supervisor_loop_complete")
        except InterruptedError:
            raise
        except Exception as exc:
            logger.exception("Supervisor loop failed.")
            self._trace_event(
                "dual_lane_supervisor_loop_failed",
                kind="exception",
                level="ERROR",
                details={
                    "error_type": type(exc).__name__,
                    "conversation": _conversation_summary(resolved_supervisor_conversation),
                    "prompt": _text_summary(prompt),
                    "specialist_count": len(specialist_records),
                    "specialist_tail": [_loop_result_summary(record.result) for record in specialist_records[-2:]],
                },
            )
            raise

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
            self._trace_event(
                "dual_lane_supervisor_result_empty",
                kind="exception",
                level="ERROR",
                details={
                    "supervisor_result": _loop_result_summary(supervisor_result),
                    "specialist_count": len(specialist_records),
                    "merged_tool_calls": len(merged_tool_calls),
                    "merged_tool_results": len(merged_tool_results),
                },
            )
            raise RuntimeError("dual-lane supervisor returned no final text")
        if final_text and not supervisor_text_emitted:
            _emit_user_text(final_text)
        self._trace_event(
            "dual_lane_run_completed",
            kind="observation",
            details={
                "decision": _decision_summary(decision),
                "supervisor_result": _loop_result_summary(supervisor_result),
                "final_text": _text_summary(final_text),
                "specialist_count": len(specialist_records),
                "used_web_search": used_web_search,
                "supervisor_text_emitted": supervisor_text_emitted,
            },
        )

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
