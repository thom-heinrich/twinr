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
from twinr.memory import ConversationTurn, MemoryLedgerItem, MemoryState, OnDeviceMemory, SearchMemoryEntry
from twinr.ops.events import TwinrOpsEventStore, compact_text


@dataclass(slots=True)
class TwinrRuntime:
    config: TwinrConfig
    state_machine: TwinrStateMachine = field(default_factory=TwinrStateMachine)
    memory: OnDeviceMemory = field(init=False)
    snapshot_store: RuntimeSnapshotStore = field(init=False)
    ops_events: TwinrOpsEventStore = field(init=False)
    last_transcript: str | None = None
    last_response: str | None = None

    def __post_init__(self) -> None:
        self.memory = OnDeviceMemory(
            max_turns=self.config.memory_max_turns,
            keep_recent=self.config.memory_keep_recent,
        )
        self.snapshot_store = RuntimeSnapshotStore(self.config.runtime_state_path)
        self.ops_events = TwinrOpsEventStore.from_config(self.config)
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
        self.ops_events.append(
            event="turn_started",
            message="Green button started a conversation turn.",
            data={"button": "green", "status": status.value},
        )
        return status

    def submit_transcript(self, transcript: str) -> TwinrStatus:
        self.last_transcript = transcript.strip()
        status = self.state_machine.transition(TwinrEvent.SPEECH_PAUSE_DETECTED)
        self._persist_snapshot()
        self.ops_events.append(
            event="transcript_submitted",
            message="Transcript captured for the active turn.",
            data={"status": status.value, "transcript_preview": compact_text(self.last_transcript)},
        )
        return status

    def begin_answering(self) -> TwinrStatus:
        status = self.state_machine.transition(TwinrEvent.RESPONSE_READY)
        self._persist_snapshot()
        return status

    def begin_proactive_prompt(self, prompt: str) -> str:
        spoken_prompt = prompt.strip()
        if not spoken_prompt:
            raise RuntimeError("Proactive prompt text must not be empty")
        status = self.state_machine.transition(TwinrEvent.PROACTIVE_PROMPT_READY)
        self.memory.remember("assistant", spoken_prompt)
        self._persist_snapshot()
        self.ops_events.append(
            event="proactive_prompt_started",
            message="Twinr started a proactive spoken prompt.",
            data={
                "status": status.value,
                "response_preview": compact_text(spoken_prompt),
            },
        )
        return spoken_prompt

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
        self.ops_events.append(
            event="turn_completed",
            message="Assistant response stored in runtime memory.",
            data={
                "status": self.status.value,
                "transcript_preview": compact_text(self.last_transcript),
                "response_preview": compact_text(response),
            },
        )
        return response

    def remember_search_result(
        self,
        *,
        question: str,
        answer: str,
        sources: tuple[str, ...] = (),
        location_hint: str | None = None,
        date_context: str | None = None,
    ) -> SearchMemoryEntry:
        entry = self.memory.remember_search(
            question=question,
            answer=answer,
            sources=sources,
            location_hint=location_hint,
            date_context=date_context,
        )
        self._persist_snapshot()
        self.ops_events.append(
            event="search_result_stored",
            message="Search result stored in structured on-device memory.",
            data={
                "question_preview": compact_text(question),
                "answer_preview": compact_text(answer),
                "sources": len(sources),
            },
        )
        return entry

    def remember_note(
        self,
        *,
        kind: str,
        content: str,
        source: str = "tool",
        metadata: dict[str, str] | None = None,
    ) -> MemoryLedgerItem:
        item = self.memory.remember_note(
            kind=kind,
            content=content,
            source=source,
            metadata=metadata,
        )
        self._persist_snapshot()
        self.ops_events.append(
            event="memory_note_stored",
            message="Structured memory note stored in on-device memory.",
            data={
                "kind": kind,
                "content_preview": compact_text(content),
            },
        )
        return item

    def complete_agent_turn(self, answer: str) -> str:
        self.begin_answering()
        return self.finalize_agent_turn(answer)

    def finish_speaking(self) -> TwinrStatus:
        status = self.state_machine.transition(TwinrEvent.TTS_FINISHED)
        self._persist_snapshot()
        self.ops_events.append(
            event="tts_finished",
            message="Twinr finished speaking the response.",
            data={"status": status.value},
        )
        return status

    def press_yellow_button(self) -> str:
        if not self.last_response:
            raise RuntimeError("No assistant response is available for printing")
        self.state_machine.transition(TwinrEvent.YELLOW_BUTTON_PRESSED)
        self._persist_snapshot()
        self.ops_events.append(
            event="print_started",
            message="Yellow button requested a print.",
            data={
                "button": "yellow",
                "status": self.status.value,
                "response_preview": compact_text(self.last_response),
            },
        )
        return self.last_response

    def begin_tool_print(self) -> TwinrStatus:
        status = self.state_machine.transition(TwinrEvent.PRINT_REQUESTED)
        self._persist_snapshot()
        self.ops_events.append(
            event="print_started",
            message="Assistant tool requested a print.",
            data={"request_source": "tool", "status": status.value},
        )
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
        self.ops_events.append(
            event="print_finished",
            message="Twinr returned to idle after printing.",
            data={"status": status.value},
        )
        return status

    def fail(self, message: str) -> TwinrStatus:
        status = self.state_machine.fail(message)
        self._persist_snapshot(error_message=message)
        self.ops_events.append(
            event="error",
            level="error",
            message="Twinr runtime entered the error state.",
            data={"status": status.value, "error": message},
        )
        return status

    def reset_error(self) -> TwinrStatus:
        status = self.state_machine.reset()
        self._persist_snapshot()
        self.ops_events.append(
            event="error_reset",
            message="Twinr runtime left the error state.",
            data={"status": status.value},
        )
        return status

    def _persist_snapshot(self, *, error_message: str | None = None) -> None:
        self.snapshot_store.save(
            status=self.status.value,
            memory_turns=self.memory.turns,
            memory_raw_tail=self.memory.raw_tail,
            memory_ledger=self.memory.ledger,
            memory_search_results=self.memory.search_results,
            memory_state=self.memory.state,
            last_transcript=self.last_transcript,
            last_response=self.last_response,
            error_message=error_message,
        )

    def _restore_snapshot_context(self) -> None:
        snapshot = self.snapshot_store.load()
        self.last_transcript = snapshot.last_transcript
        self.last_response = snapshot.last_response or None
        if snapshot.memory_raw_tail or snapshot.memory_ledger or snapshot.memory_search_results:
            self.memory.restore_structured(
                raw_tail=tuple(
                    ConversationTurn(
                        role=turn.role,
                        content=turn.content,
                        created_at=self._parse_snapshot_timestamp(turn.created_at),
                    )
                    for turn in snapshot.memory_raw_tail
                ),
                ledger=tuple(
                    MemoryLedgerItem(
                        kind=item.kind,
                        content=item.content,
                        created_at=self._parse_snapshot_timestamp(item.created_at),
                        source=item.source,
                        metadata=dict(item.metadata),
                    )
                    for item in snapshot.memory_ledger
                ),
                search_results=tuple(
                    SearchMemoryEntry(
                        question=item.question,
                        answer=item.answer,
                        sources=tuple(item.sources),
                        created_at=self._parse_snapshot_timestamp(item.created_at),
                        location_hint=item.location_hint,
                        date_context=item.date_context,
                    )
                    for item in snapshot.memory_search_results
                ),
                state=MemoryState(
                    active_topic=snapshot.memory_state.active_topic,
                    last_user_goal=snapshot.memory_state.last_user_goal,
                    pending_printable=snapshot.memory_state.pending_printable,
                    last_search_summary=snapshot.memory_state.last_search_summary,
                    open_loops=tuple(snapshot.memory_state.open_loops),
                ),
            )
        else:
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
