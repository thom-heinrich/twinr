"""Own turn-controller context shaping and label-specific guidance messages."""

from __future__ import annotations

from dataclasses import dataclass

from twinr.agent.base_agent.conversation.turn_controller import StreamingTurnController


@dataclass(frozen=True, slots=True)
class TurnGuidanceContext:
    """Capture the bounded context selected for one guided turn."""

    conversation: tuple[tuple[str, str], ...]
    max_context_turns: int
    available_context_turns: int
    selected_context_turns: int
    guidance_message_count: int
    turn_label: str | None


class TurnGuidanceRuntime:
    """Build bounded turn-controller context and label-aware guidance."""

    def __init__(self, loop) -> None:
        self._loop = loop

    def controller_conversation(self) -> tuple[tuple[str, str], ...]:
        """Return the bounded conversation slice used by the turn controller."""

        loop = self._loop
        max_turns = max(0, int(loop.config.turn_controller_context_turns))
        conversation = loop.runtime.conversation_context()
        if max_turns <= 0 or len(conversation) <= max_turns:
            selected = conversation
        else:
            selected = conversation[-max_turns:]
        loop._trace_event(
            "turn_guidance_controller_context_gate",
            kind="decision",
            details={
                "available_turns": len(conversation),
                "selected_turns": len(selected),
                "max_context_turns": max_turns,
            },
            kpi={"selected_turns": len(selected)},
        )
        return selected

    def build_streaming_turn_controller(self) -> StreamingTurnController | None:
        """Build the bounded turn controller when the runtime gate is open."""

        loop = self._loop
        enabled = loop.turn_decision_evaluator is not None and bool(loop.config.turn_controller_enabled)
        loop._trace_event(
            "turn_guidance_controller_build_gate",
            kind="decision",
            details={"enabled": enabled},
        )
        if not enabled:
            return None
        return StreamingTurnController(
            config=loop.config,
            evaluator=loop.turn_decision_evaluator,
            conversation_factory=self.controller_conversation,
            emit=loop.emit,
        )

    def guidance_messages(self, turn_label: str | None) -> tuple[tuple[str, str], ...]:
        """Return the label-specific guidance messages for one turn."""

        normalized = str(turn_label or "").strip().lower()
        if normalized != "backchannel":
            return ()
        return (
            (
                "system",
                "The current user turn is a short backchannel or direct answer to the latest assistant prompt. "
                "If you answer, keep it very short, direct, and do not restate the whole context.",
            ),
        )

    def context_for_turn_label(self, turn_label: str | None) -> TurnGuidanceContext:
        """Return the provider conversation plus any label-specific guidance."""

        loop = self._loop
        base_conversation = loop.runtime.provider_conversation_context()
        guidance = self.guidance_messages(turn_label)
        context = TurnGuidanceContext(
            conversation=base_conversation + guidance,
            max_context_turns=max(0, int(loop.config.turn_controller_context_turns)),
            available_context_turns=len(base_conversation),
            selected_context_turns=len(base_conversation),
            guidance_message_count=len(guidance),
            turn_label=str(turn_label or "").strip().lower() or None,
        )
        loop._trace_event(
            "turn_guidance_context_gate",
            kind="decision",
            details={
                "turn_label": context.turn_label,
                "available_turns": context.available_context_turns,
                "guidance_message_count": context.guidance_message_count,
                "max_context_turns": context.max_context_turns,
            },
            kpi={"conversation_turns": len(context.conversation)},
        )
        return context

    def conversation_context_for_turn_label(self, turn_label: str | None) -> tuple[tuple[str, str], ...]:
        """Return only the composed conversation payload for one turn label."""

        return self.context_for_turn_label(turn_label).conversation
