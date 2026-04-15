"""Build streaming lane plans and select the final dual-lane execution path."""

from __future__ import annotations

# CHANGELOG: 2026-03-28
# BUG-1: Prevent one-shot/generator conversation contexts from being consumed by tracing
#        before execution by freezing them exactly once at materialization time.
# BUG-2: Fix supervisor-decision handling for dict/Pydantic/dataclass-shaped outputs and
#        avoid accidental duplicate supervisor-resolution work when a decision is already known.
# BUG-3: Preserve a valid prefetched direct decision if the full-context re-resolve returns
#        no decision instead of silently falling back to an unrelated generic path.
# SEC-1: Replace stable unsalted SHA-1 trace fingerprints with keyed BLAKE2s forensic
#        fingerprints to reduce offline reconstruction/correlation risk from stolen logs.
#        # BREAKING: trace field `sha12` is now a keyed BLAKE2s fingerprint alias, not a
#        literal SHA-1 digest. Set TWINR_TRACE_FINGERPRINT_KEY for stable cross-process fingerprints.
# SEC-2: Direct runtime-local tool shortcuts no longer bypass authorization/allowlist hooks.
#        Unauthorized shortcuts fall back to the generic tool loop instead of executing immediately.
#        # BREAKING: old "always execute runtime-local shortcut" behavior now requires
#        `streaming_allow_unapproved_runtime_local_tool_shortcuts=True`.
# IMP-1: Freeze context snapshots and summarize only bounded tails to keep tracing cheap on Pi 4
#        while removing time-of-check/time-of-use drift in long streaming turns.
# IMP-2: Add typed decision normalization and common-format message extraction so the planner
#        works with 2026-style structured outputs and richer conversation item shapes.
# IMP-3: Planner falls back to its own methods when loop monkey-patch helper methods are absent.

import os
from dataclasses import dataclass
from hashlib import blake2s
from typing import Any, Mapping, Sequence

from twinr.agent.tools import (
    DualLaneToolLoop,
    build_compact_tool_agent_instructions,
    build_tool_agent_instructions,
)
from twinr.agent.base_agent.prompting.personality import load_tool_loop_instructions, merge_instructions
from twinr.agent.base_agent.contracts import supervisor_decision_requires_full_context
from twinr.agent.base_agent.contracts import normalize_supervisor_decision_context_scope
from twinr.agent.base_agent.contracts import normalize_supervisor_decision_runtime_tool_name
from twinr.agent.tools.runtime.runtime_local_handoff import (
    has_executable_runtime_local_tool_call as _has_executable_runtime_local_tool_call,
)
from twinr.agent.workflows.streaming_turn_coordinator import StreamingTurnLanePlan
from twinr.agent.workflows.streaming_turn_orchestrator import StreamingTurnTimeoutPolicy
from twinr.agent.workflows.voice_turn_latency import mark_voice_turn_supervisor_ready


_TRACE_FINGERPRINT_ENV = "TWINR_TRACE_FINGERPRINT_KEY"
_TRACE_FINGERPRINT_DIGEST_SIZE = 8
_TRACE_DEFAULT_SUMMARY_SAMPLE_LIMIT = 64
_TRACE_DEFAULT_SUMMARY_TAIL_LIMIT = 3
_TRACE_MAX_TEXT_ITEMS = 8
_TRACE_PERSON = b"twlnv2__"
_DEFAULT_SHARED_SUPERVISOR_WAIT_MS = 2000
_PROCESS_TRACE_FINGERPRINT_KEY: str | bytes | None = os.environ.get(_TRACE_FINGERPRINT_ENV)
if _PROCESS_TRACE_FINGERPRINT_KEY is None:
    _PROCESS_TRACE_FINGERPRINT_KEY = os.urandom(16)


@dataclass(frozen=True)
class _DecisionView:
    """Normalized read-only projection of a supervisor decision."""

    raw: Any
    present: bool
    action: str | None
    kind: str | None
    context_scope: str | None
    runtime_tool_name: str | None
    runtime_tool_arguments_present: bool
    runtime_tool_shortcut_ready: bool
    allow_web_search: bool
    spoken_ack: Any
    spoken_reply: Any
    goal: Any
    location_hint: Any
    date_context: Any

    @classmethod
    def from_any(cls, decision: Any) -> "_DecisionView":
        if decision is None:
            return cls(
                raw=None,
                present=False,
                action=None,
                kind=None,
                context_scope=None,
                runtime_tool_name=None,
                runtime_tool_arguments_present=False,
                runtime_tool_shortcut_ready=False,
                allow_web_search=False,
                spoken_ack=None,
                spoken_reply=None,
                goal=None,
                location_hint=None,
                date_context=None,
            )
        runtime_tool_arguments = _read_field(decision, "runtime_tool_arguments", None)
        return cls(
            raw=decision,
            present=True,
            action=_normalized_lower(_read_field(decision, "action", None)),
            kind=_normalized_lower(_read_field(decision, "kind", None)),
            context_scope=normalize_supervisor_decision_context_scope(
                _read_field(decision, "context_scope", None)
            ),
            runtime_tool_name=normalize_supervisor_decision_runtime_tool_name(
                _read_field(decision, "runtime_tool_name", None)
            ),
            runtime_tool_arguments_present=isinstance(runtime_tool_arguments, Mapping),
            runtime_tool_shortcut_ready=_has_executable_runtime_local_tool_call(decision),
            allow_web_search=bool(_read_field(decision, "allow_web_search", False)),
            spoken_ack=_read_field(decision, "spoken_ack", None),
            spoken_reply=_read_field(decision, "spoken_reply", None),
            goal=_read_field(decision, "goal", None),
            location_hint=_read_field(decision, "location_hint", None),
            date_context=_read_field(decision, "date_context", None),
        )

    @property
    def is_runtime_local_tool(self) -> bool:
        return self.runtime_tool_shortcut_ready

    @property
    def wants_search_feedback(self) -> bool:
        return self.action == "handoff" and (self.kind == "search" or self.allow_web_search)


class StreamingLanePlanner:
    """Construct lane plans while keeping `streaming_runner.py` thin."""

    def __init__(self, loop) -> None:
        self._loop = loop

    def streaming_turn_timeout_policy(
        self,
        *,
        decision_hint=None,
        assume_unresolved_supervisor_handoff: bool = False,
    ) -> StreamingTurnTimeoutPolicy:
        """Build the bounded timeout policy for one parallel dual-lane turn.

        Tool/search handoffs can legitimately take longer than parametric
        direct replies because they still need to reach a specialist/tool path
        and wait for the result. Keep those turns bounded, but give them a
        dedicated wider envelope instead of reusing the generic 15s final-lane
        deadline.
        """

        loop = self._loop
        use_search_budget = _decision_requires_extended_timeout(
            decision_hint,
            assume_unresolved_supervisor_handoff=assume_unresolved_supervisor_handoff,
        )
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
                prefetched_decision_view = _DecisionView.from_any(prefetched_decision)

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
                        "prefetched_decision": prefetched_decision_view.action,
                        "prefetched_first_word": first_word_reply is not None,
                    },
                )
                timeout_builder = getattr(loop, "_streaming_turn_timeout_policy", None)
                # Live spoken turns can miss the speculative supervisor window while
                # still resolving into a real handoff/search final lane.
                unresolved_supervisor_handoff = (
                    not prefetched_decision_view.present
                    and getattr(loop.streaming_turn_loop, "supervisor_decision_provider", None)
                    is not None
                )
                timeout_policy = (
                    timeout_builder(
                        decision_hint=prefetched_decision,
                        assume_unresolved_supervisor_handoff=unresolved_supervisor_handoff,
                    )
                    if callable(timeout_builder)
                    else self.streaming_turn_timeout_policy(
                        decision_hint=prefetched_decision,
                        assume_unresolved_supervisor_handoff=unresolved_supervisor_handoff,
                    )
                )
                mark_voice_turn_supervisor_ready()
                return StreamingTurnLanePlan(
                    turn_instructions=turn_instructions,
                    run_final_lane=lambda: (
                        loop._run_dual_lane_final_response(
                            transcript,
                            turn_instructions=turn_instructions,
                            prefetched_decision=prefetched_decision,
                        )
                        if callable(getattr(loop, "_run_dual_lane_final_response", None))
                        else self.run_dual_lane_final_response(
                            transcript,
                            turn_instructions=turn_instructions,
                            prefetched_decision=prefetched_decision,
                        )
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
                    timeout_policy=timeout_policy,
                    recover_final_lane_response=None,
                )

            turn_extra_instructions = merge_instructions(
                loop.config.openai_realtime_instructions,
                load_tool_loop_instructions(loop.config),
            )
            turn_instructions = (
                build_compact_tool_agent_instructions(
                    loop.config,
                    extra_instructions=turn_extra_instructions,
                )
                if (loop.config.llm_provider or "").strip().lower() == "groq"
                else build_tool_agent_instructions(
                    loop.config,
                    extra_instructions=turn_extra_instructions,
                )
            )
            mark_voice_turn_supervisor_ready()
            return StreamingTurnLanePlan(
                turn_instructions=turn_instructions,
                run_single_lane=lambda on_text_delta: loop.streaming_turn_loop.run(
                    transcript,
                    conversation=loop.runtime.tool_provider_tiny_recent_conversation_context(),
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
        prefetched_decision_view = _DecisionView.from_any(prefetched_decision)
        resolved_decision = prefetched_decision
        resolved_decision_view = prefetched_decision_view
        search_context = None
        supervisor_context = None
        tool_context = None
        tool_tiny_recent_context = None
        skip_structured_supervisor_decision = prefetched_decision_view.present

        def _raise_if_turn_stopped(stage: str) -> None:
            if not loop._active_turn_stop_requested():
                return
            loop._trace_event(
                "dual_lane_final_path_aborted",
                kind="branch",
                details={"stage": stage, "transcript": _text_summary(transcript, loop=loop)},
            )
            raise InterruptedError(f"dual-lane final path stopped during {stage}")

        def _materialize_context(source: str, provider):
            _raise_if_turn_stopped(f"{source}_context_fetch")
            value = provider() if callable(provider) else provider
            frozen = _freeze_conversation_context(value)
            _raise_if_turn_stopped(f"{source}_context_fetched")
            loop._trace_event(
                "dual_lane_context_materialized",
                kind="observation",
                details={
                    "source": source,
                    "transcript": _text_summary(transcript, loop=loop),
                    "context": _conversation_summary(frozen, loop=loop),
                },
            )
            return frozen

        def _search_context():
            nonlocal search_context
            if search_context is None:
                search_context = _materialize_context(
                    "search",
                    loop.runtime.search_provider_conversation_context,
                )
            return search_context

        def _supervisor_context():
            nonlocal supervisor_context
            if supervisor_context is None:
                supervisor_context = _materialize_context(
                    "supervisor",
                    loop.runtime.supervisor_provider_conversation_context,
                )
            return supervisor_context

        def _tool_context():
            nonlocal tool_context
            if tool_context is None:
                tool_context = _materialize_context(
                    "tool",
                    loop.runtime.tool_provider_conversation_context,
                )
            return tool_context

        def _tool_tiny_recent_context():
            nonlocal tool_tiny_recent_context
            if tool_tiny_recent_context is None:
                tiny_recent_reader = getattr(
                    loop.runtime,
                    "tool_provider_tiny_recent_conversation_context",
                    None,
                )
                tool_tiny_recent_context = _materialize_context(
                    "tool_tiny_recent",
                    tiny_recent_reader
                    if callable(tiny_recent_reader)
                    else loop.runtime.tool_provider_conversation_context,
                )
            return tool_tiny_recent_context

        def _handoff_context(decision_view: _DecisionView):
            if decision_view.kind == "search":
                return _search_context()
            return _tool_tiny_recent_context() if decision_view.context_scope == "tiny_recent" else _tool_context()

        def _run_with_search_feedback(decision_view: _DecisionView, callback):
            def stop_search_feedback() -> None:
                return None

            if decision_view.wants_search_feedback and not _working_feedback_active(loop):
                stop_search_feedback = loop._start_search_feedback_loop()
            try:
                _raise_if_turn_stopped("search_feedback_start")
                return callback()
            finally:
                stop_search_feedback()

        def _run_generic_supervisor_loop():
            return loop.streaming_turn_loop.run(
                transcript,
                conversation=_tool_tiny_recent_context(),
                supervisor_conversation=_supervisor_context(),
                prefetched_decision=None,
                skip_supervisor_decision=True,
                instructions=turn_instructions,
                allow_web_search=False,
                on_text_delta=None,
                on_lane_text_delta=None,
                should_stop=loop._active_turn_stop_requested,
            )

        def _run_runtime_local_tool_only_if_authorized(
            decision_view: _DecisionView,
            *,
            source: str,
        ):
            if decision_view.runtime_tool_name is not None and not decision_view.runtime_tool_shortcut_ready:
                loop._trace_event(
                    "dual_lane_runtime_local_tool_shortcut_incomplete",
                    kind="branch",
                    details={
                        "source": source,
                        "transcript": _text_summary(transcript, loop=loop),
                        "decision": _decision_summary(decision_view.raw, loop=loop),
                        "fallback": "specialist_handoff",
                    },
                )
            if not decision_view.is_runtime_local_tool:
                return None
            if not _runtime_local_tool_shortcut_authorized(loop, transcript, decision_view.raw):
                loop._trace_event(
                    "dual_lane_runtime_local_tool_shortcut_blocked",
                    kind="security",
                    details={
                        "source": source,
                        "transcript": _text_summary(transcript, loop=loop),
                        "decision": _decision_summary(decision_view.raw, loop=loop),
                        "fallback": "generic_tool_loop",
                    },
                )
                return None
            loop._trace_decision(
                "dual_lane_final_path_selected",
                question="Which final-lane execution path should run?",
                selected={
                    "id": f"{source}_runtime_local_tool",
                    "summary": "Execute the runtime-local tool directly",
                },
                options=[
                    {
                        "id": f"{source}_runtime_local_tool",
                        "summary": "Execute the runtime-local tool directly",
                    },
                    {"id": f"{source}_generic", "summary": "Run generic tool loop with prefetched decision"},
                    {"id": "resolve_supervisor", "summary": "Resolve supervisor decision synchronously"},
                ],
                context={
                    "transcript": _text_summary(transcript, loop=loop),
                    "decision": _decision_summary(decision_view.raw, loop=loop),
                },
                guardrails=["runtime_local_direct_tool_authorized"],
            )
            return loop.streaming_turn_loop.run_runtime_local_tool_only(
                transcript,
                decision=decision_view.raw,
                on_text_delta=None,
                on_lane_text_delta=None,
                emit_filler=False,
                should_stop=loop._active_turn_stop_requested,
            )

        _raise_if_turn_stopped("final_path_start")
        if prefetched_decision_view.action == "handoff":
            runtime_local_result = _run_runtime_local_tool_only_if_authorized(
                prefetched_decision_view,
                source="prefetched",
            )
            if runtime_local_result is not None:
                return runtime_local_result
            if not prefetched_decision_view.is_runtime_local_tool:
                handoff_kind = prefetched_decision_view.kind or "general"
                handoff_context = _handoff_context(prefetched_decision_view)
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
                        "transcript": _text_summary(transcript, loop=loop),
                        "handoff_kind": handoff_kind,
                        "prefetched": True,
                        "decision": _decision_summary(prefetched_decision, loop=loop),
                        "context_source": "search" if handoff_kind == "search" else "tool",
                        "context": _conversation_summary(handoff_context, loop=loop),
                    },
                    guardrails=["single_search_execution"],
                )
                return _run_with_search_feedback(
                    prefetched_decision_view,
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
                prefetched_decision_view = _DecisionView.from_any(prefetched_decision)
                if prefetched_decision is None:
                    loop._trace_event(
                        "dual_lane_shared_supervisor_decision_unavailable",
                        kind="branch",
                        details={
                            "transcript": _text_summary(transcript, loop=loop),
                            "fallback_to_generic_supervisor_loop": True,
                        },
                    )
            _raise_if_turn_stopped("supervisor_decision_initial")
            resolved_decision = prefetched_decision
            resolved_decision_view = prefetched_decision_view
            skip_structured_supervisor_decision = (
                skip_structured_supervisor_decision or resolved_decision_view.present
            )
            if resolved_decision_view.present:
                action = resolved_decision_view.action or ""
                if action == "handoff":
                    runtime_local_result = _run_runtime_local_tool_only_if_authorized(
                        resolved_decision_view,
                        source="resolved",
                    )
                    if runtime_local_result is not None:
                        return runtime_local_result
                    if not resolved_decision_view.is_runtime_local_tool:
                        handoff_kind = resolved_decision_view.kind or "general"
                        handoff_context = _handoff_context(resolved_decision_view)
                        context_source = "search" if handoff_kind == "search" else "tool"
                        loop._trace_decision(
                            "dual_lane_final_path_selected",
                            question="Which final-lane execution path should run?",
                            selected={
                                "id": f"resolved_{handoff_kind}_handoff",
                                "summary": f"Run direct {handoff_kind} handoff",
                            },
                            options=[
                                {"id": "resolved_handoff", "summary": "Run specialist handoff only"},
                                {"id": "resolved_direct", "summary": "Run resolved direct reply"},
                                {"id": "generic_tool_loop", "summary": "Run generic tool loop"},
                            ],
                            context={
                                "transcript": _text_summary(transcript, loop=loop),
                                "action": action,
                                "decision": _decision_summary(resolved_decision, loop=loop),
                                "context_source": context_source,
                                "context": _conversation_summary(handoff_context, loop=loop),
                            },
                            guardrails=["handoff_short_path"],
                        )
                        return _run_with_search_feedback(
                            resolved_decision_view,
                            lambda: loop.streaming_turn_loop.run_handoff_only(
                                transcript,
                                conversation=handoff_context,
                                specialist_conversation=handoff_context,
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
                            "transcript": _text_summary(transcript, loop=loop),
                            "action": action,
                            "decision": _decision_summary(resolved_decision, loop=loop),
                            "context_source": "tool"
                            if supervisor_decision_requires_full_context(resolved_decision)
                            else "supervisor",
                        },
                    )
                    final_context = (
                        _tool_context() if supervisor_decision_requires_full_context(resolved_decision) else None
                    )
                    return loop.streaming_turn_loop.run(
                        transcript,
                        conversation=final_context,
                        supervisor_conversation=_supervisor_context(),
                        prefetched_decision=resolved_decision,
                        skip_supervisor_decision=skip_structured_supervisor_decision,
                        instructions=turn_instructions,
                        allow_web_search=False,
                        on_text_delta=None,
                        on_lane_text_delta=None,
                        should_stop=loop._active_turn_stop_requested,
                    )
                if resolved_decision_view.wants_search_feedback:
                    return _run_with_search_feedback(
                        resolved_decision_view,
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
            selected={"id": "generic_supervisor_loop", "summary": "Run generic supervisor loop without structured decision"},
            options=[
                {"id": "prefetched_handoff", "summary": "Reuse search handoff"},
                {"id": "resolved_supervisor", "summary": "Resolve supervisor route"},
                {"id": "generic_supervisor_loop", "summary": "Run generic supervisor loop without structured decision"},
            ],
            context={
                "transcript": _text_summary(transcript, loop=loop),
                "prefetched": prefetched_decision_view.present,
                "decision": _decision_summary(resolved_decision, loop=loop),
                "context_source": "supervisor+tool_tiny_recent",
                "conversation": _conversation_summary(_tool_tiny_recent_context(), loop=loop),
                "supervisor_conversation": _conversation_summary(_supervisor_context(), loop=loop),
            },
            guardrails=["skip_duplicate_structured_supervisor_resolution"],
        )
        return _run_generic_supervisor_loop()


def _read_field(value: Any, name: str, default: Any = None) -> Any:
    """Read a field from mappings, objects, or dataclass-like values."""

    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _normalized_lower(value: Any) -> str | None:
    normalized = str(value or "").strip().lower()
    return normalized or None


def _freeze_conversation_context(value):
    """Snapshot a possibly one-shot context iterable into a reusable container."""

    if value is None:
        return None
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return list(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    try:
        return list(value)
    except TypeError:
        return value


def _trace_summary_sample_limit(loop=None) -> int:
    configured = getattr(getattr(loop, "config", None), "streaming_trace_summary_sample_limit", None)
    if configured is None:
        configured = _TRACE_DEFAULT_SUMMARY_SAMPLE_LIMIT
    return max(1, int(configured))


def _trace_summary_tail_limit(loop=None) -> int:
    configured = getattr(getattr(loop, "config", None), "streaming_trace_summary_tail_limit", None)
    if configured is None:
        configured = _TRACE_DEFAULT_SUMMARY_TAIL_LIMIT
    return max(1, int(configured))


def _trace_fingerprint_key(loop=None) -> bytes:
    configured = getattr(getattr(loop, "config", None), "streaming_trace_fingerprint_key", None)
    if configured is not None:
        key = configured if isinstance(configured, bytes) else str(configured).encode("utf-8", errors="ignore")
    else:
        key = (
            _PROCESS_TRACE_FINGERPRINT_KEY
            if isinstance(_PROCESS_TRACE_FINGERPRINT_KEY, bytes)
            else str(_PROCESS_TRACE_FINGERPRINT_KEY).encode("utf-8", errors="ignore")
        )
    if len(key) > 32:
        key = blake2s(key, digest_size=32, person=b"twrkeyv2").digest()
    return key


def _trace_fingerprint(value: str, *, loop=None) -> str:
    digest = blake2s(
        value.encode("utf-8", errors="ignore"),
        digest_size=_TRACE_FINGERPRINT_DIGEST_SIZE,
        key=_trace_fingerprint_key(loop),
        person=_TRACE_PERSON,
    ).hexdigest()
    return digest[:12]


def _flatten_text(value: Any, *, max_items: int = _TRACE_MAX_TEXT_ITEMS) -> str:
    """Best-effort extraction of text-ish content from common response item shapes."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, Mapping):
        for key in ("text", "content", "value", "message"):
            if key in value:
                return _flatten_text(value.get(key), max_items=max_items)
        return str(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parts: list[str] = []
        for item in list(value)[:max_items]:
            flattened = _flatten_text(item, max_items=max_items)
            if flattened:
                parts.append(flattened)
        return " ".join(parts)
    for attr in ("text", "content", "value", "message"):
        nested = getattr(value, attr, None)
        if nested is not None:
            return _flatten_text(nested, max_items=max_items)
    return str(value)


def _message_role_and_content(item: Any) -> tuple[str, str]:
    """Extract a role/content view from common tuple, dict, or object message shapes."""

    if isinstance(item, (list, tuple)) and len(item) >= 2:
        role = str(item[0] or "").strip() or "unknown"
        content = _flatten_text(item[1])
        return role, content
    if isinstance(item, Mapping):
        role = str(item.get("role") or item.get("type") or "unknown").strip() or "unknown"
        content = _flatten_text(
            item.get("content", item.get("text", item.get("value", item.get("message", ""))))
        )
        return role, content
    role = str(getattr(item, "role", None) or getattr(item, "type", None) or "unknown").strip() or "unknown"
    content = _flatten_text(
        getattr(item, "content", None)
        or getattr(item, "text", None)
        or getattr(item, "value", None)
        or getattr(item, "message", None)
        or item
    )
    return role, content


def _text_summary(value: Any, *, loop=None) -> dict[str, Any]:
    """Describe text safely for workflow forensics without raw content."""

    normalized = _flatten_text(value).strip()
    if not normalized:
        return {"present": False, "chars": 0, "words": 0, "sha12": None, "fingerprint12": None}
    fingerprint = _trace_fingerprint(normalized, loop=loop)
    return {
        "present": True,
        "chars": len(normalized),
        "words": len(normalized.split()),
        "sha12": fingerprint,
        "fingerprint12": fingerprint,
        "fingerprint_alg": "blake2s-keyed-trunc12hex",
    }


def _conversation_summary(conversation, *, loop=None) -> dict[str, Any]:
    """Summarize one provider conversation context without leaking text."""

    if not conversation:
        return {"present": False, "messages": 0, "tail": []}

    sample_limit = _trace_summary_sample_limit(loop)
    tail_limit = _trace_summary_tail_limit(loop)

    if isinstance(conversation, Sequence) and not isinstance(conversation, (str, bytes, bytearray)):
        items = conversation
    else:
        items = tuple(conversation)

    total_messages = len(items)
    sampled_items = items[-sample_limit:]
    tail_items = sampled_items[-tail_limit:]

    sampled_total_chars = 0
    tail: list[dict[str, Any]] = []
    for item in tail_items:
        role, content = _message_role_and_content(item)
        content_summary = _text_summary(content, loop=loop)
        sampled_total_chars += content_summary["chars"]
        tail.append({"role": role, "content": content_summary})

    if len(sampled_items) > len(tail_items):
        for item in sampled_items[:-len(tail_items)]:
            _, content = _message_role_and_content(item)
            sampled_total_chars += len(content.strip())

    return {
        "present": True,
        "messages": total_messages,
        "sampled_messages": len(sampled_items),
        "sampled_total_chars": sampled_total_chars,
        "truncated": total_messages > len(sampled_items),
        "tail": tail,
    }


def _decision_summary(decision, *, loop=None) -> dict[str, Any]:
    """Summarize the final-lane decision shape without raw prompt leakage."""

    view = _DecisionView.from_any(decision)
    if not view.present:
        return {"present": False}
    return {
        "present": True,
        "action": view.action,
        "kind": view.kind,
        "context_scope": view.context_scope,
        "runtime_tool_name": view.runtime_tool_name,
        "runtime_tool_arguments_present": view.runtime_tool_arguments_present,
        "allow_web_search": view.allow_web_search,
        "spoken_ack": _text_summary(view.spoken_ack, loop=loop),
        "spoken_reply": _text_summary(view.spoken_reply, loop=loop),
        "goal": _text_summary(view.goal, loop=loop),
        "location_hint": _text_summary(view.location_hint, loop=loop),
        "date_context": _text_summary(view.date_context, loop=loop),
    }


def _decision_wants_search_feedback(decision) -> bool:
    """Return whether a supervisor handoff should play search progress tones."""

    return _DecisionView.from_any(decision).wants_search_feedback


def _decision_requires_extended_timeout(
    decision,
    *,
    assume_unresolved_supervisor_handoff: bool,
) -> bool:
    """Return whether the final lane should use the wider handoff timeout budget."""

    view = _DecisionView.from_any(decision)
    if not view.present:
        return bool(assume_unresolved_supervisor_handoff)
    return view.action == "handoff"


def _working_feedback_active(loop) -> bool:
    """Return whether the default processing-feedback loop is still active."""

    return callable(getattr(loop, "_working_feedback_stop", None))


def _shared_supervisor_decision_wait_ms(loop) -> int:
    """Bound how long the final lane waits for the shared supervisor-decision worker."""

    prefetch_wait_ms = max(0, int(getattr(loop.config, "streaming_supervisor_prefetch_wait_ms", 0)))
    bridge_wait_ms = max(0, int(getattr(loop.config, "streaming_bridge_reply_timeout_ms", 0)))
    watchdog_timeout_ms = max(0, int(getattr(loop.config, "streaming_final_lane_watchdog_timeout_ms", 0)))
    configured_hard_timeout_ms = getattr(
        loop.config,
        "streaming_supervisor_prefetch_hard_timeout_ms",
        _DEFAULT_SHARED_SUPERVISOR_WAIT_MS,
    )
    if configured_hard_timeout_ms is None:
        return max(
            prefetch_wait_ms,
            bridge_wait_ms,
            watchdog_timeout_ms or _DEFAULT_SHARED_SUPERVISOR_WAIT_MS,
        )

    hard_timeout_ms = max(0, int(configured_hard_timeout_ms))
    target_wait_ms = max(prefetch_wait_ms, bridge_wait_ms, hard_timeout_ms)
    if watchdog_timeout_ms > 0:
        return min(target_wait_ms, watchdog_timeout_ms)
    return target_wait_ms


def _runtime_local_tool_shortcut_authorized(loop, transcript: str, decision) -> bool:
    """Return whether this layer may bypass the generic tool loop and execute a local tool directly."""

    decision_view = _DecisionView.from_any(decision)
    if not decision_view.is_runtime_local_tool:
        return True

    approval_hook = getattr(loop, "_authorize_runtime_local_tool_execution", None)
    if callable(approval_hook):
        try:
            return bool(
                approval_hook(
                    transcript=transcript,
                    decision=decision_view.raw,
                    runtime_tool_name=decision_view.runtime_tool_name,
                )
            )
        except TypeError:
            try:
                return bool(approval_hook(transcript, decision_view.raw, decision_view.runtime_tool_name))
            except TypeError:
                return bool(approval_hook(decision_view.raw))

    runtime_tool_name = decision_view.runtime_tool_name
    if runtime_tool_name:
        runtime_tool_names = tuple(getattr(loop, "_runtime_tool_names", ()) or ())
        if not runtime_tool_names or runtime_tool_name in runtime_tool_names:
            sensitive_check = getattr(loop, "_is_sensitive_tool_name", None)
            is_sensitive = bool(sensitive_check(runtime_tool_name)) if callable(sensitive_check) else False
            if not is_sensitive:
                return True
            sensitive_authorized = getattr(loop, "_is_sensitive_tool_access_authorized", None)
            if callable(sensitive_authorized):
                try:
                    return bool(sensitive_authorized())
                except Exception:
                    return False

    allowlist = getattr(getattr(loop, "config", None), "streaming_runtime_local_tool_shortcut_allowlist", None)
    if allowlist:
        try:
            normalized_allowlist = {str(item).strip() for item in allowlist if str(item).strip()}
        except TypeError:
            normalized_allowlist = {str(allowlist).strip()}
        return decision_view.runtime_tool_name in normalized_allowlist

    return bool(
        getattr(
            getattr(loop, "config", None),
            "streaming_allow_unapproved_runtime_local_tool_shortcuts",
            False,
        )
    )
