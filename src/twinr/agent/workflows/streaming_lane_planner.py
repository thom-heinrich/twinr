"""Build streaming lane plans and select the final dual-lane execution path."""

from __future__ import annotations

from twinr.agent.tools import (
    DualLaneToolLoop,
    build_compact_tool_agent_instructions,
    build_tool_agent_instructions,
)
from twinr.agent.workflows.streaming_turn_coordinator import StreamingTurnLanePlan
from twinr.agent.workflows.streaming_turn_orchestrator import StreamingTurnTimeoutPolicy


class StreamingLanePlanner:
    """Construct lane plans while keeping `streaming_runner.py` thin."""

    def __init__(self, loop) -> None:
        self._loop = loop

    def streaming_turn_timeout_policy(self) -> StreamingTurnTimeoutPolicy:
        """Build the bounded timeout policy for one parallel dual-lane turn."""

        loop = self._loop
        watchdog_ms = max(25, int(loop.config.streaming_final_lane_watchdog_timeout_ms))
        hard_timeout_ms = max(watchdog_ms, int(loop.config.streaming_final_lane_hard_timeout_ms))
        return StreamingTurnTimeoutPolicy(
            bridge_reply_timeout_ms=max(0, int(loop.config.streaming_bridge_reply_timeout_ms)),
            final_lane_watchdog_timeout_ms=watchdog_ms,
            final_lane_hard_timeout_ms=hard_timeout_ms,
            first_audio_gate_ms=max(0, int(loop.config.streaming_first_word_final_lane_wait_ms)),
        )

    def dual_lane_error_reply(self) -> str:
        """Return the localized fallback reply for bounded lane failures."""

        loop = self._loop
        if isinstance(loop.streaming_turn_loop, DualLaneToolLoop):
            return str(getattr(loop.streaming_turn_loop, "default_error_reply", "") or "").strip()
        return "Das hat gerade nicht geklappt. Bitte versuche es noch einmal."

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
                prefetched_decision = loop._consume_speculative_supervisor_decision(transcript)
                first_word_reply = loop._dual_lane_bridge_reply_from_decision(prefetched_decision)
                first_word_source = "supervisor_prefetched" if first_word_reply is not None else "none"
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
                        if first_word_reply is None and loop._dual_lane_prefers_supervisor_bridge()
                        else (
                            (lambda: loop._generate_first_word_reply(transcript))
                            if first_word_reply is None and loop.first_word_provider is not None
                            else None
                        )
                    ),
                    bridge_fallback_reply=None,
                    timeout_policy=loop._streaming_turn_timeout_policy(),
                    final_timeout_reply=loop._dual_lane_error_reply(),
                    final_error_reply=loop._dual_lane_error_reply(),
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
            prefetched_decision = loop._consume_speculative_supervisor_decision(transcript)
        search_context = loop.runtime.search_provider_conversation_context()
        supervisor_context = loop.runtime.supervisor_provider_conversation_context()
        if prefetched_decision is not None and getattr(prefetched_decision, "action", None) == "handoff":
            loop._trace_decision(
                "dual_lane_final_path_selected",
                question="Which final-lane execution path should run?",
                selected={"id": "prefetched_handoff", "summary": "Run one direct handoff-only search"},
                options=[
                    {"id": "prefetched_handoff", "summary": "Run one direct handoff-only search"},
                    {"id": "resolve_supervisor", "summary": "Resolve supervisor decision synchronously"},
                    {"id": "generic_tool_loop", "summary": "Run generic tool loop"},
                ],
                context={"transcript_len": len(transcript), "prefetched": True},
                guardrails=["single_search_execution"],
            )
            return loop.streaming_turn_loop.run_handoff_only(
                transcript,
                conversation=search_context,
                specialist_conversation=search_context,
                handoff=prefetched_decision,
                instructions=turn_instructions,
                allow_web_search=False,
                on_text_delta=None,
                on_lane_text_delta=None,
                emit_filler=False,
            )
        if getattr(loop.streaming_turn_loop, "supervisor_decision_provider", None) is not None:
            resolved_decision = prefetched_decision or loop.streaming_turn_loop.resolve_supervisor_decision(
                transcript,
                conversation=supervisor_context,
                instructions=turn_instructions,
            )
            if resolved_decision is not None:
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
                        context={"transcript_len": len(transcript), "action": action},
                        guardrails=["search_handoff_short_path"],
                    )
                    return loop.streaming_turn_loop.run_handoff_only(
                        transcript,
                        conversation=search_context,
                        specialist_conversation=search_context,
                        handoff=resolved_decision,
                        instructions=turn_instructions,
                        allow_web_search=False,
                        on_text_delta=None,
                        on_lane_text_delta=None,
                        emit_filler=False,
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
                        context={"transcript_len": len(transcript), "action": action},
                    )
                    return loop.streaming_turn_loop.run(
                        transcript,
                        conversation=search_context,
                        supervisor_conversation=supervisor_context,
                        prefetched_decision=resolved_decision,
                        instructions=turn_instructions,
                        allow_web_search=False,
                        on_text_delta=None,
                        on_lane_text_delta=None,
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
            context={"transcript_len": len(transcript), "prefetched": prefetched_decision is not None},
        )
        return loop.streaming_turn_loop.run(
            transcript,
            conversation=loop.runtime.tool_provider_conversation_context(),
            supervisor_conversation=supervisor_context,
            prefetched_decision=prefetched_decision,
            instructions=turn_instructions,
            allow_web_search=False,
            on_text_delta=None,
            on_lane_text_delta=None,
        )
