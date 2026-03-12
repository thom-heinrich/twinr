from __future__ import annotations

from dataclasses import dataclass, field

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.state_machine import (
    InvalidTransitionError,
    TwinrEvent,
    TwinrStateMachine,
    TwinrStatus,
)
from twinr.memory import OnDeviceMemory


@dataclass(slots=True)
class TwinrRuntime:
    config: TwinrConfig
    state_machine: TwinrStateMachine = field(default_factory=TwinrStateMachine)
    memory: OnDeviceMemory = field(init=False)
    last_transcript: str | None = None
    last_response: str | None = None

    def __post_init__(self) -> None:
        self.memory = OnDeviceMemory(
            max_turns=self.config.memory_max_turns,
            keep_recent=self.config.memory_keep_recent,
        )

    @property
    def status(self) -> TwinrStatus:
        return self.state_machine.status

    def conversation_context(self) -> tuple[tuple[str, str], ...]:
        return tuple((turn.role, turn.content) for turn in self.memory.turns)

    def press_green_button(self) -> TwinrStatus:
        return self.state_machine.transition(TwinrEvent.GREEN_BUTTON_PRESSED)

    def submit_transcript(self, transcript: str) -> TwinrStatus:
        self.last_transcript = transcript.strip()
        return self.state_machine.transition(TwinrEvent.SPEECH_PAUSE_DETECTED)

    def begin_answering(self) -> TwinrStatus:
        return self.state_machine.transition(TwinrEvent.RESPONSE_READY)

    def cancel_listening(self) -> TwinrStatus:
        return self.state_machine.reset()

    def finalize_agent_turn(self, answer: str) -> str:
        if not self.last_transcript:
            raise RuntimeError("No transcript is available for response generation")
        response = answer.strip()
        self.memory.remember("user", self.last_transcript)
        self.memory.remember("assistant", response)
        self.last_response = response
        return response

    def complete_agent_turn(self, answer: str) -> str:
        self.begin_answering()
        return self.finalize_agent_turn(answer)

    def finish_speaking(self) -> TwinrStatus:
        return self.state_machine.transition(TwinrEvent.TTS_FINISHED)

    def press_yellow_button(self) -> str:
        if not self.last_response:
            raise RuntimeError("No assistant response is available for printing")
        self.state_machine.transition(TwinrEvent.YELLOW_BUTTON_PRESSED)
        return self.last_response

    def begin_tool_print(self) -> TwinrStatus:
        return self.state_machine.transition(TwinrEvent.PRINT_REQUESTED)

    def resume_answering_after_print(self) -> TwinrStatus:
        return self.state_machine.transition(TwinrEvent.RESPONSE_READY)

    def maybe_begin_tool_print(self) -> TwinrStatus | None:
        try:
            return self.begin_tool_print()
        except InvalidTransitionError:
            return None

    def finish_printing(self) -> TwinrStatus:
        return self.state_machine.transition(TwinrEvent.PRINT_FINISHED)

    def fail(self, message: str) -> TwinrStatus:
        return self.state_machine.fail(message)

    def reset_error(self) -> TwinrStatus:
        return self.state_machine.reset()
