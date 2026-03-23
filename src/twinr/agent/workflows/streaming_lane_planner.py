"""Build streaming lane plans and select the final dual-lane execution path."""

from __future__ import annotations

from hashlib import sha1
from typing import Any

from twinr.agent.tools import (
    DualLaneToolLoop,
    build_compact_tool_agent_instructions,
    build_tool_agent_instructions,
)
from twinr.agent.base_agent.contracts import supervisor_decision_requires_full_context
from twinr.agent.workflows.streaming_turn_coordinator import StreamingTurnLanePlan
from twinr.agent.workflows.streaming_turn_orchestrator import StreamingTurnTimeoutPolicy


class StreamingLanePlanner:
    """Construct lane plans while keeping `streaming_runner.py` thin."""

    def __init__(self, loop) -> None:
        self._loop = loop

    def streaming_turn_timeout_policy(
        self,
        *,
        decision_hint=None,
    ) -> StreamingTurnTimeoutPolicy:
        """Build the bounded timeout policy for one parallel dual-lane turn.

        Search handoffs can legitimately take longer than parametric direct
        replies because they still need to reach the search tool path and wait
        for live results. Keep those turns bounded, but give them a dedicated
        wider envelope instead of reusing the generic 15s final-lane deadline.
        """

        loop = self._loop
        use_search_budget = _decision_requires_search_timeout(decision_hint)
        watchdog_ms = max(
            25,
            int(
                loop.config.streaming_search_final_lane_watchdog_timeout_ms
                if use_search_budget
                else loop.config.streaming_final_lane_watchdog_timeout_ms
            ),
        )
        hard_timeout_ms = max(
            watchdog_ms,
            int(
                loop.config.streaming_search_final_lane_hard_timeout_ms
                if use_search_budget
                else loop.config.streaming_final_lane_hard_timeout_ms
            ),
        )
        return StreamingTurnTimeoutPolicy(
            bridge_reply_timeout_ms=max(0, int(loop.config.streaming_bridge_reply_timeout_ms)),
            final_lane_watchdog_timeout_ms=watchdog_ms,
            final_lane_hard_timeout_ms=hard_timeout_ms,
            first_audio_gate_ms=max(0, int(loop.config.streaming_first_word_final_lane_wait_ms)),
        )

    def build_turn_lane_plan(self, transcript: str) -> StreamingTurnLanePlan:
        """Build the full execution plan for one streaming turn."""

        loop = self._loop
        with loop._trace_span(
            name="streaming_lane_plan_build",
            kind="span",
            details={
                "dual_lane": isinstance(loop.streaming_turn_loop, DualLaneToolLoop),
                "text_len": len(transcript),
            },
        ):
            if isinstance(loop.streaming_turn_loop, DualLaneToolLoop):
                turn_instructions = None
                loop._maybe_start_speculative_first_word(transcript)
                loop._maybe_start_speculative_supervisor_decision(transcript)
                local_route_resolution = loop._resolve_local_semantic_route(transcript)
                prefetched_decision = (
                    getattr(local_route_resolution, "supervisor_decision", None)
                    if local_route_resolution is not None
                    else None
                )
                if prefetched_decision is None:
                    prefetched_decision = loop._consume_speculative_supervisor_decision(transcript)
                first_word_reply = (
                    getattr(local_route_resolution, "bridge_reply", None)
                    if local_route_resolution is not None
                    else None
                )
                first_word_source = "local_semantic_router" if first_word_reply is not None else "none"
                if first_word_reply is None:
                    first_word_reply = loop._dual_lane_bridge_reply_from_decision(prefetched_decision)
                    first_word_source = "supervisor_prefetched" if first_word_reply is not None else "none"
                local_authoritative = bool(
                    local_route_resolution is not None
                    and getattr(local_route_resolution, "supervisor_decision", None) is not None
                )
                if (
                    first_word_reply is None
                    and loop.config.streaming_first_word_enabled
                    and not loop._dual_lane_prefers_supervisor_bridge()
                ):
                    first_word_reply = loop._consume_speculative_first_word(transcript)
                    if first_word_reply is not None:
                        first_word_source = "first_word_prefetched"
                loop._trace_event(
                    "streaming_dual_lane_starting",
                    kind="decision",
                    details={
                        "local_authoritative": local_authoritative,
                        "prefetched_decision": getattr(prefetched_decision, "action", None)
                        if prefetched_decision is not None
                        else None,
                        "prefetched_first_word": first_word_reply is not None,
                    },
                )
                return StreamingTurnLanePlan(
                    turn_instructions=turn_instructions,
                    run_final_lane=lambda: loop._run_dual_lane_final_response(
                        transcript,
                        turn_instructions=turn_instructions,
                        prefetched_decision=prefetched_decision,
                    ),
                    prefetched_first_word=first_word_reply,
                    prefetched_first_word_source=first_word_source,
                    generate_first_word=(
                        (
                            lambda: loop._generate_supervisor_bridge_reply(
                                transcript,
                                instructions=turn_instructions,
                            )
                        )
                        if (
                            first_word_reply is None
                            and loop._dual_lane_prefers_supervisor_bridge()
                            and not local_authoritative
                        )
                        else (
                            (lambda: loop._generate_first_word_reply(transcript))
                            if first_word_reply is None and loop.first_word_provider is not None
                            else None
                        )
                    ),
                    bridge_fallback_reply=None,
                    timeout_policy=loop._streaming_turn_timeout_policy(
                        decision_hint=prefetched_decision,
                    ),
                    recover_final_lane_response=None,
                )

            turn_instructions = (
                build_compact_tool_agent_instructions(
                    loop.config,
                    extra_instructions=loop.config.openai_realtime_instructions,
                )
                if (loop.config.llm_provider or "").strip().lower() == "groq"
                else build_tool_agent_instructions(
                    loop.config,
                    extra_instructions=loop.config.openai_realtime_instructions,
                )
            )
            return StreamingTurnLanePlan(
                turn_instructions=turn_instructions,
                run_single_lane=lambda on_text_delta: loop.streaming_turn_loop.run(
                    transcript,
                    conversation=loop.runtime.tool_provider_conversation_context(),
                    instructions=turn_instructions,
                    allow_web_search=False,
                    on_text_delta=on_text_delta,
                    should_stop=loop._active_turn_stop_requested,
                ),
            )

    def run_dual_lane_final_response(
        self,
        transcript: str,
        *,
        turn_instructions: str | None,
        prefetched_decision=None,
    ):
        """Execute the bounded final-lane path for one dual-lane turn."""

        loop = self._loop
        if prefetched_decision is None:
            local_route_resolution = loop._resolve_local_semantic_route(transcript)
            if local_route_resolution is not None:
                prefetched_decision = getattr(local_route_resolution, "supervisor_decision", None)
            if prefetched_decision is None:
                prefetched_decision = loop._consume_speculative_supervisor_decision(transcript)
        resolved_decision = prefetched_decision
        search_context = None
        supervisor_context = None
        supervisor_direct_context = None
        tool_context = None
        skip_structured_supervisor_decision = False

        def _raise_if_turn_stopped(stage: str) -> None:
            if not loop._active_turn_stop_requested():
                return
            loop._trace_event(
                "dual_lane_final_path_aborted",
                kind="branch",
                details={"stage": stage, "transcript": _text_summary(transcript)},
            )
            raise InterruptedError(f"dual-lane final path stopped during {stage}")

        def _materialize_context(source: str, value):
            loop._trace_event(
                "dual_lane_context_materialized",
                kind="observation",
                details={
                    "source": source,
                    "transcript": _text_summary(transcript),
                    "context": _conversation_summary(value),
                },
            )
            return value

        def _search_context():
            nonlocal search_context
            if search_context is None:
                search_context = _materialize_context(
                    "search",
                    loop.runtime.search_provider_conversation_context(),
                )
            return search_context

        def _supervisor_context():
            nonlocal supervisor_context
            if supervisor_context is None:
                supervisor_context = _materialize_context(
                    "supervisor",
                    loop.runtime.supervisor_provider_conversation_context(),
                )
            return supervisor_context

        def _supervisor_direct_context():
            nonlocal supervisor_direct_context
            if supervisor_direct_context is None:
                supervisor_direct_context = _materialize_context(
                    "supervisor_direct",
                    loop.runtime.supervisor_direct_provider_conversation_context(transcript),
                )
            return supervisor_direct_context

        def _tool_context():
            nonlocal tool_context
            if tool_context is None:
                tool_context = _materialize_context(
                    "tool",
                    loop.runtime.tool_provider_conversation_context(),
                )
            return tool_context

        def _handoff_context(decision):
            kind = str(getattr(decision, "kind", "") or "").strip().lower()
            return _search_context() if kind == "search" else _tool_context()

        def _run_with_search_feedback(decision, callback):
            stop_search_feedback = lambda: None
            if _decision_wants_search_feedback(decision):
                stop_working_feedback = getattr(loop, "_stop_working_feedback", None)
                if callable(stop_working_feedback):
                    stop_working_feedback()
                stop_search_feedback = loop._start_search_feedback_loop()
            try:
                _raise_if_turn_stopped("search_feedback_start")
                return callback()
            finally:
                stop_search_feedback()

        _raise_if_turn_stopped("final_path_start")
        if prefetched_decision is not None and getattr(prefetched_decision, "action", None) == "handoff":
            handoff_kind = str(getattr(prefetched_decision, "kind", "") or "").strip().lower() or "general"
            handoff_context = _handoff_context(prefetched_decision)
            loop._trace_decision(
                "dual_lane_final_path_selected",
                question="Which final-lane execution path should run?",
                selected={"id": "prefetched_handoff", "summary": f"Run one direct handoff-only {handoff_kind} path"},
                options=[
                    {"id": "prefetched_handoff", "summary": "Run one direct handoff-only path"},
                    {"id": "resolve_supervisor", "summary": "Resolve supervisor decision synchronously"},
                    {"id": "generic_tool_loop", "summary": "Run generic tool loop"},
                ],
                context={
                    "transcript": _text_summary(transcript),
                    "handoff_kind": handoff_kind,
                    "prefetched": True,
                    "decision": _decision_summary(prefetched_decision),
                    "context_source": "search" if handoff_kind == "search" else "tool",
                    "context": _conversation_summary(handoff_context),
                },
                guardrails=["single_search_execution"],
            )
            return _run_with_search_feedback(
                prefetched_decision,
                lambda: loop.streaming_turn_loop.run_handoff_only(
                    transcript,
                    conversation=handoff_context,
                    specialist_conversation=handoff_context,
                    handoff=prefetched_decision,
                    instructions=turn_instructions,
                    allow_web_search=False,
                    on_text_delta=None,
                    on_lane_text_delta=None,
                    emit_filler=False,
                    should_stop=loop._active_turn_stop_requested,
                ),
            )
        if getattr(loop.streaming_turn_loop, "supervisor_decision_provider", None) is not None:
            shared_supervisor_decision = (
                prefetched_decision is None
                and loop._has_shared_speculative_supervisor_decision(transcript)
            )
            if shared_supervisor_decision:
                prefetched_decision = loop._wait_for_speculative_supervisor_decision(
                    transcript,
                    wait_ms=_shared_supervisor_decision_wait_ms(loop),
                )
                if prefetched_decision is None:
                    loop._trace_event(
                        "dual_lane_shared_supervisor_decision_unavailable",
                        kind="branch",
                        details={
                            "transcript": _text_summary(transcript),
                            "fallback_to_sync_resolve": True,
                        },
                    )
            _raise_if_turn_stopped("supervisor_decision_initial")
            resolved_decision = prefetched_decision
            if resolved_decision is None and not skip_structured_supervisor_decision:
                resolved_decision = loop.streaming_turn_loop.resolve_supervisor_decision(
                    transcript,
                    conversation=_supervisor_context(),
                    instructions=turn_instructions,
                    should_stop=loop._active_turn_stop_requested,
                )
            if resolved_decision is not None:
                action = str(getattr(resolved_decision, "action", "") or "").strip().lower()
                if action == "direct":
                    _raise_if_turn_stopped("supervisor_decision_reresolve")
                    loop._trace_event(
                        "dual_lane_direct_reply_reresolved_with_memory_context",
                        kind="branch",
                        details={"prefetched": prefetched_decision is not None},
                    )
                    resolved_decision = loop.streaming_turn_loop.resolve_supervisor_decision(
                        transcript,
                        conversation=_supervisor_direct_context(),
                        instructions=turn_instructions,
                        should_stop=loop._active_turn_stop_requested,
                    )
                    action = str(getattr(resolved_decision, "action", "") or "").strip().lower()
                if action == "handoff" and str(getattr(resolved_decision, "kind", "") or "").strip().lower() == "search":
                    loop._trace_decision(
                        "dual_lane_final_path_selected",
                        question="Which final-lane execution path should run?",
                        selected={"id": "resolved_search_handoff", "summary": "Run direct search handoff"},
                        options=[
                            {"id": "resolved_search_handoff", "summary": "Run search handoff only"},
                            {"id": "resolved_direct", "summary": "Run resolved direct reply"},
                            {"id": "generic_tool_loop", "summary": "Run generic tool loop"},
                        ],
                        context={
                            "transcript": _text_summary(transcript),
                            "action": action,
                            "decision": _decision_summary(resolved_decision),
                            "context_source": "search",
                            "context": _conversation_summary(_search_context()),
                        },
                        guardrails=["search_handoff_short_path"],
                    )
                    return _run_with_search_feedback(
                        resolved_decision,
                        lambda: loop.streaming_turn_loop.run_handoff_only(
                            transcript,
                            conversation=_search_context(),
                            specialist_conversation=_search_context(),
                            handoff=resolved_decision,
                            instructions=turn_instructions,
                            allow_web_search=False,
                            on_text_delta=None,
                            on_lane_text_delta=None,
                            emit_filler=False,
                            should_stop=loop._active_turn_stop_requested,
                        ),
                    )
                if action in {"direct", "end_conversation"}:
                    loop._trace_decision(
                        "dual_lane_final_path_selected",
                        question="Which final-lane execution path should run?",
                        selected={"id": action, "summary": "Run resolved direct supervisor path"},
                        options=[
                            {"id": "direct", "summary": "Speak direct answer"},
                            {"id": "end_conversation", "summary": "Speak answer and close session"},
                            {"id": "generic_tool_loop", "summary": "Run generic tool loop"},
                        ],
                        context={
                            "transcript": _text_summary(transcript),
                            "action": action,
                            "decision": _decision_summary(resolved_decision),
                            "context_source": (
                                "tool"
                                if supervisor_decision_requires_full_context(resolved_decision)
                                else "search"
                            ),
                        },
                    )
                    final_context = (
                        _tool_context()
                        if supervisor_decision_requires_full_context(resolved_decision)
                        else _search_context()
                    )
                    return loop.streaming_turn_loop.run(
                        transcript,
                        conversation=final_context,
                        supervisor_conversation=(
                            _supervisor_direct_context() if action == "direct" else _supervisor_context()
                        ),
                        prefetched_decision=resolved_decision,
                        instructions=turn_instructions,
                        allow_web_search=False,
                        on_text_delta=None,
                        on_lane_text_delta=None,
                        should_stop=loop._active_turn_stop_requested,
                    )
                if _decision_wants_search_feedback(resolved_decision):
                    return _run_with_search_feedback(
                        resolved_decision,
                        lambda: loop.streaming_turn_loop.run(
                            transcript,
                            conversation=_tool_context(),
                            supervisor_conversation=_supervisor_context(),
                            prefetched_decision=resolved_decision,
                            skip_supervisor_decision=skip_structured_supervisor_decision,
                            instructions=turn_instructions,
                            allow_web_search=False,
                            on_text_delta=None,
                            on_lane_text_delta=None,
                            should_stop=loop._active_turn_stop_requested,
                        ),
                    )
        loop._trace_decision(
            "dual_lane_final_path_selected",
            question="Which final-lane execution path should run?",
            selected={"id": "generic_tool_loop", "summary": "Run generic tool loop"},
            options=[
                {"id": "prefetched_handoff", "summary": "Reuse search handoff"},
                {"id": "resolved_supervisor", "summary": "Resolve supervisor route"},
                {"id": "generic_tool_loop", "summary": "Run generic tool loop"},
            ],
            context={
                "transcript": _text_summary(transcript),
                "prefetched": prefetched_decision is not None,
                "decision": _decision_summary(resolved_decision),
                "context_source": "tool",
                "context": _conversation_summary(_tool_context()),
            },
        )
        return loop.streaming_turn_loop.run(
            transcript,
            conversation=_tool_context(),
            supervisor_conversation=_supervisor_context(),
            prefetched_decision=resolved_decision,
            skip_supervisor_decision=skip_structured_supervisor_decision,
            instructions=turn_instructions,
            allow_web_search=False,
            on_text_delta=None,
            on_lane_text_delta=None,
            should_stop=loop._active_turn_stop_requested,
        )


def _text_summary(value: Any) -> dict[str, Any]:
    """Describe text safely for workflow forensics without raw content."""

    normalized = str(value or "").strip()
    if not normalized:
        return {"present": False, "chars": 0, "words": 0, "sha12": None}
    return {
        "present": True,
        "chars": len(normalized),
        "words": len(normalized.split()),
        "sha12": sha1(normalized.encode("utf-8")).hexdigest()[:12],
    }


def _conversation_summary(conversation) -> dict[str, Any]:
    """Summarize one provider conversation context without leaking text."""

    if not conversation:
        return {"present": False, "messages": 0, "tail": []}
    tail: list[dict[str, Any]] = []
    total_chars = 0
    for item in conversation:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            role = str(item[0] or "").strip() or "unknown"
            content = str(item[1] or "")
        else:
            role = "unknown"
            content = str(item or "")
        total_chars += len(content.strip())
        tail.append({"role": role, "content": _text_summary(content)})
    return {
        "present": True,
        "messages": len(tail),
        "total_chars": total_chars,
        "tail": tail[-3:],
    }


def _decision_summary(decision) -> dict[str, Any]:
    """Summarize the final-lane decision shape without raw prompt leakage."""

    if decision is None:
        return {"present": False}
    return {
        "present": True,
        "action": str(getattr(decision, "action", "") or "").strip().lower() or None,
        "kind": str(getattr(decision, "kind", "") or "").strip().lower() or None,
        "context_scope": str(getattr(decision, "context_scope", "") or "").strip() or None,
        "allow_web_search": getattr(decision, "allow_web_search", None),
        "spoken_ack": _text_summary(getattr(decision, "spoken_ack", None)),
        "spoken_reply": _text_summary(getattr(decision, "spoken_reply", None)),
        "goal": _text_summary(getattr(decision, "goal", None)),
        "location_hint": _text_summary(getattr(decision, "location_hint", None)),
        "date_context": _text_summary(getattr(decision, "date_context", None)),
    }


def _decision_wants_search_feedback(decision) -> bool:
    """Return whether a supervisor handoff should play search progress tones."""

    if decision is None:
        return False
    action = str(getattr(decision, "action", "") or "").strip().lower()
    if action != "handoff":
        return False
    kind = str(getattr(decision, "kind", "") or "").strip().lower()
    if kind == "search":
        return True
    return bool(getattr(decision, "allow_web_search", False))


def _decision_requires_search_timeout(decision) -> bool:
    """Return whether the final lane should use the wider search timeout budget."""

    if decision is None:
        return False
    action = str(getattr(decision, "action", "") or "").strip().lower()
    if action != "handoff":
        return False
    kind = str(getattr(decision, "kind", "") or "").strip().lower()
    if kind == "search":
        return True
    return bool(getattr(decision, "allow_web_search", False))


def _shared_supervisor_decision_wait_ms(loop) -> int:
    """Bound how long the final lane waits for the shared supervisor-decision worker."""

    prefetch_wait_ms = max(0, int(getattr(loop.config, "streaming_supervisor_prefetch_wait_ms", 0)))
    bridge_wait_ms = max(0, int(getattr(loop.config, "streaming_bridge_reply_timeout_ms", 0)))
    return max(prefetch_wait_ms, bridge_wait_ms)
