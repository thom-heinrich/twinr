from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.runtime_state import RuntimeSnapshotStore
from twinr.agent.base_agent.state_machine import (
    InvalidTransitionError,
    TwinrEvent,
    TwinrStateMachine,
    TwinrStatus,
)
from twinr.memory import OnDeviceMemory
from twinr.memory import ConversationTurn


@dataclass(slots=True)
class TwinrRuntime:
    config: TwinrConfig
    state_machine: TwinrStateMachine = field(default_factory=TwinrStateMachine)
    memory: OnDeviceMemory = field(init=False)
    snapshot_store: RuntimeSnapshotStore = field(init=False)
    last_transcript: str | None = None
    last_response: str | None = None

    def __post_init__(self) -> None:
        self.memory = OnDeviceMemory(
            max_turns=self.config.memory_max_turns,
            keep_recent=self.config.memory_keep_recent,
        )
        self.snapshot_store = RuntimeSnapshotStore(self.config.runtime_state_path)
        if self.config.restore_runtime_state_on_startup:
            self._restore_snapshot_context()
        self._persist_snapshot()

    @property
    def status(self) -> TwinrStatus:
        return self.state_machine.status

    def conversation_context(self) -> tuple[tuple[str, str], ...]:
        return tuple((turn.role, turn.content) for turn in self.memory.turns)

    def press_green_button(self) -> TwinrStatus:
        status = self.state_machine.transition(TwinrEvent.GREEN_BUTTON_PRESSED)
        self._persist_snapshot()
        return status

    def submit_transcript(self, transcript: str) -> TwinrStatus:
        self.last_transcript = transcript.strip()
        status = self.state_machine.transition(TwinrEvent.SPEECH_PAUSE_DETECTED)
        self._persist_snapshot()
        return status

    def begin_answering(self) -> TwinrStatus:
        status = self.state_machine.transition(TwinrEvent.RESPONSE_READY)
        self._persist_snapshot()
        return status

    def cancel_listening(self) -> TwinrStatus:
        status = self.state_machine.reset()
        self._persist_snapshot()
        return status

    def finalize_agent_turn(self, answer: str) -> str:
        if not self.last_transcript:
            raise RuntimeError("No transcript is available for response generation")
        response = answer.strip()
        self.memory.remember("user", self.last_transcript)
        self.memory.remember("assistant", response)
        self.last_response = response
        self._persist_snapshot()
        return response

    def complete_agent_turn(self, answer: str) -> str:
        self.begin_answering()
        return self.finalize_agent_turn(answer)

    def finish_speaking(self) -> TwinrStatus:
        status = self.state_machine.transition(TwinrEvent.TTS_FINISHED)
        self._persist_snapshot()
        return status

    def press_yellow_button(self) -> str:
        if not self.last_response:
            raise RuntimeError("No assistant response is available for printing")
        self.state_machine.transition(TwinrEvent.YELLOW_BUTTON_PRESSED)
        self._persist_snapshot()
        return self.last_response

    def begin_tool_print(self) -> TwinrStatus:
        status = self.state_machine.transition(TwinrEvent.PRINT_REQUESTED)
        self._persist_snapshot()
        return status

    def resume_answering_after_print(self) -> TwinrStatus:
        status = self.state_machine.transition(TwinrEvent.RESPONSE_READY)
        self._persist_snapshot()
        return status

    def maybe_begin_tool_print(self) -> TwinrStatus | None:
        try:
            return self.begin_tool_print()
        except InvalidTransitionError:
            return None

    def finish_printing(self) -> TwinrStatus:
        status = self.state_machine.transition(TwinrEvent.PRINT_FINISHED)
        self._persist_snapshot()
        return status

    def fail(self, message: str) -> TwinrStatus:
        status = self.state_machine.fail(message)
        self._persist_snapshot(error_message=message)
        return status

    def reset_error(self) -> TwinrStatus:
        status = self.state_machine.reset()
        self._persist_snapshot()
        return status

    def _persist_snapshot(self, *, error_message: str | None = None) -> None:
        self.snapshot_store.save(
            status=self.status.value,
            memory_turns=self.memory.turns,
            last_transcript=self.last_transcript,
            last_response=self.last_response,
            error_message=error_message,
        )

    def _restore_snapshot_context(self) -> None:
        snapshot = self.snapshot_store.load()
        self.last_transcript = snapshot.last_transcript
        self.last_response = snapshot.last_response or None
        restored_turns: list[ConversationTurn] = []
        for turn in snapshot.memory_turns:
            created_at = self._parse_snapshot_timestamp(turn.created_at)
            restored_turns.append(
                ConversationTurn(
                    role=turn.role,
                    content=turn.content,
                    created_at=created_at,
                )
            )
        self.memory.restore(tuple(restored_turns))
        if not self.last_response:
            self.last_response = self.memory.last_assistant_message()

    @staticmethod
    def _parse_snapshot_timestamp(value: str) -> datetime:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return datetime.now().astimezone()
