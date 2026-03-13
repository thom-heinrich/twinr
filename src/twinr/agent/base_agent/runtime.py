from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from twinr.automations import AutomationCondition, AutomationDefinition, AutomationStore
from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.runtime_state import RuntimeSnapshotStore
from twinr.agent.base_agent.state_machine import (
    InvalidTransitionError,
    TwinrEvent,
    TwinrStateMachine,
    TwinrStatus,
)
from twinr.memory import ConversationTurn, MemoryLedgerItem, MemoryState, OnDeviceMemory, SearchMemoryEntry
from twinr.memory.reminders import ReminderEntry, ReminderStore, format_due_label
from twinr.ops.events import TwinrOpsEventStore, compact_text


@dataclass(slots=True)
class TwinrRuntime:
    config: TwinrConfig
    state_machine: TwinrStateMachine = field(default_factory=TwinrStateMachine)
    memory: OnDeviceMemory = field(init=False)
    reminder_store: ReminderStore = field(init=False)
    automation_store: AutomationStore = field(init=False)
    snapshot_store: RuntimeSnapshotStore = field(init=False)
    ops_events: TwinrOpsEventStore = field(init=False)
    last_transcript: str | None = None
    last_response: str | None = None
    user_voice_status: str | None = None
    user_voice_confidence: float | None = None
    user_voice_checked_at: str | None = None

    def __post_init__(self) -> None:
        self.memory = OnDeviceMemory(
            max_turns=self.config.memory_max_turns,
            keep_recent=self.config.memory_keep_recent,
        )
        self.reminder_store = ReminderStore(
            self.config.reminder_store_path,
            timezone_name=self.config.local_timezone_name,
            retry_delay_s=self.config.reminder_retry_delay_s,
            max_entries=self.config.reminder_max_entries,
        )
        self.automation_store = AutomationStore(
            self.config.automation_store_path,
            timezone_name=self.config.local_timezone_name,
            max_entries=self.config.automation_max_entries,
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

    def provider_conversation_context(self) -> tuple[tuple[str, str], ...]:
        guidance = self._voice_guidance_message()
        if not guidance:
            return self.conversation_context()
        return (("system", guidance), *self.conversation_context())

    def begin_listening(
        self,
        *,
        request_source: str,
        button: str | None = None,
        proactive_trigger: str | None = None,
    ) -> TwinrStatus:
        status = self.state_machine.transition(TwinrEvent.GREEN_BUTTON_PRESSED)
        self._persist_snapshot()
        data = {
            "status": status.value,
            "request_source": request_source,
        }
        if button:
            data["button"] = button
        if proactive_trigger:
            data["proactive_trigger"] = proactive_trigger
        self.ops_events.append(
            event="turn_started",
            message="Conversation listening window started.",
            data=data,
        )
        return status

    def press_green_button(self) -> TwinrStatus:
        return self.begin_listening(request_source="button", button="green")

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
        return self._begin_background_prompt(
            prompt,
            event="proactive_prompt_started",
            message="Twinr started a proactive spoken prompt.",
        )

    def begin_reminder_prompt(self, prompt: str) -> str:
        return self._begin_background_prompt(
            prompt,
            event="reminder_prompt_started",
            message="Twinr started speaking a due reminder.",
        )

    def begin_automation_prompt(self, prompt: str) -> str:
        return self._begin_background_prompt(
            prompt,
            event="automation_prompt_started",
            message="Twinr started speaking a due automation prompt.",
        )

    def _begin_background_prompt(self, prompt: str, *, event: str, message: str) -> str:
        spoken_prompt = prompt.strip()
        if not spoken_prompt:
            raise RuntimeError("Background prompt text must not be empty")
        status = self.state_machine.transition(TwinrEvent.PROACTIVE_PROMPT_READY)
        self.memory.remember("assistant", spoken_prompt)
        self._persist_snapshot()
        self.ops_events.append(
            event=event,
            message=message,
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

    def update_user_voice_assessment(
        self,
        *,
        status: str | None,
        confidence: float | None,
        checked_at: str | None,
    ) -> None:
        self.user_voice_status = (status or "").strip() or None
        self.user_voice_confidence = confidence
        self.user_voice_checked_at = (checked_at or "").strip() or None
        self._persist_snapshot()

    def _voice_guidance_message(self) -> str | None:
        status = (self.user_voice_status or "").strip().lower()
        if not status:
            return None
        if status == "likely_user":
            signal = "likely match to the enrolled main-user voice profile"
        elif status == "uncertain":
            signal = "partial match to the enrolled main-user voice profile"
        elif status == "unknown_voice":
            signal = "does not match the enrolled main-user voice profile closely enough"
        else:
            signal = status.replace("_", " ")

        parts = [
            "Live speaker signal for this turn. Treat it as a local heuristic, not proof of identity.",
            f"Speaker signal: {signal}.",
        ]
        if self.user_voice_confidence is not None:
            parts.append(f"Confidence: {self.user_voice_confidence * 100:.0f}%.")
        if status in {"uncertain", "unknown_voice"}:
            parts.append(
                "For persistent or security-sensitive changes, first ask for explicit confirmation. "
                "Only call tools with confirmed=true after the user clearly confirms in the current conversation."
            )
        else:
            parts.append(
                "You may use this signal for calmer personalization, but never as the only authorization for a sensitive action."
            )
        return " ".join(parts)

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

    def schedule_reminder(
        self,
        *,
        due_at: str,
        summary: str,
        details: str | None = None,
        kind: str = "reminder",
        source: str = "tool",
        original_request: str | None = None,
    ) -> ReminderEntry:
        entry = self.reminder_store.schedule(
            due_at=due_at,
            summary=summary,
            details=details,
            kind=kind,
            source=source,
            original_request=original_request,
        )
        due_label = format_due_label(entry.due_at, timezone_name=self.config.local_timezone_name)
        self.remember_note(
            kind="reminder",
            content=f"Reminder scheduled for {due_label}: {entry.summary}",
            source=source,
            metadata={"reminder_id": entry.reminder_id, "reminder_kind": entry.kind},
        )
        self.ops_events.append(
            event="reminder_scheduled",
            message="A reminder or timer was scheduled.",
            data={
                "reminder_id": entry.reminder_id,
                "kind": entry.kind,
                "due_at": entry.due_at.isoformat(),
                "summary": compact_text(entry.summary),
            },
        )
        return entry

    def list_automation_records(self, *, now: datetime | None = None) -> tuple[dict[str, object], ...]:
        return self.automation_store.list_tool_records(now=now)

    def create_time_automation(
        self,
        *,
        name: str,
        actions,
        description: str | None = None,
        enabled: bool = True,
        schedule: str = "once",
        due_at: str | None = None,
        time_of_day: str | None = None,
        weekdays: tuple[int, ...] | list[int] = (),
        timezone_name: str | None = None,
        source: str = "tool",
        tags: tuple[str, ...] | list[str] = (),
    ) -> AutomationDefinition:
        entry = self.automation_store.create_time_automation(
            name=name,
            description=description,
            enabled=enabled,
            schedule=schedule,
            due_at=due_at,
            time_of_day=time_of_day,
            weekdays=weekdays,
            timezone_name=timezone_name or self.config.local_timezone_name,
            actions=actions,
            source=source,
            tags=tags,
        )
        self.remember_note(
            kind="automation",
            content=f"Time automation created: {entry.name}",
            source=source,
            metadata={"automation_id": entry.automation_id, "trigger_kind": entry.trigger.kind},
        )
        self.ops_events.append(
            event="automation_created",
            message="A time-based automation was created.",
            data={
                "automation_id": entry.automation_id,
                "name": entry.name,
                "schedule": getattr(entry.trigger, "schedule", entry.trigger.kind),
            },
        )
        return entry

    def create_if_then_automation(
        self,
        *,
        name: str,
        actions,
        description: str | None = None,
        enabled: bool = True,
        event_name: str | None = None,
        all_conditions: tuple[AutomationCondition, ...] | list[AutomationCondition] = (),
        any_conditions: tuple[AutomationCondition, ...] | list[AutomationCondition] = (),
        cooldown_seconds: float = 0.0,
        source: str = "tool",
        tags: tuple[str, ...] | list[str] = (),
    ) -> AutomationDefinition:
        entry = self.automation_store.create_if_then_automation(
            name=name,
            description=description,
            enabled=enabled,
            event_name=event_name,
            all_conditions=all_conditions,
            any_conditions=any_conditions,
            cooldown_seconds=cooldown_seconds,
            actions=actions,
            source=source,
            tags=tags,
        )
        self.remember_note(
            kind="automation",
            content=f"Sensor automation created: {entry.name}",
            source=source,
            metadata={"automation_id": entry.automation_id, "trigger_kind": entry.trigger.kind},
        )
        self.ops_events.append(
            event="automation_created",
            message="An if-then automation was created.",
            data={
                "automation_id": entry.automation_id,
                "name": entry.name,
                "event_name": getattr(entry.trigger, "event_name", None),
            },
        )
        return entry

    def update_automation(
        self,
        automation_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        enabled: bool | None = None,
        trigger=None,
        actions=None,
        source: str | None = None,
        tags: tuple[str, ...] | list[str] | None = None,
    ) -> AutomationDefinition:
        entry = self.automation_store.update(
            automation_id,
            name=name,
            description=description,
            enabled=enabled,
            trigger=trigger,
            actions=actions,
            source=source,
            tags=tags,
        )
        self.remember_note(
            kind="automation",
            content=f"Automation updated: {entry.name}",
            source=source or "automation_update",
            metadata={"automation_id": entry.automation_id, "trigger_kind": entry.trigger.kind},
        )
        self.ops_events.append(
            event="automation_updated",
            message="An automation was updated.",
            data={"automation_id": entry.automation_id, "name": entry.name},
        )
        return entry

    def delete_automation(self, automation_id: str, *, source: str = "tool") -> AutomationDefinition:
        entry = self.automation_store.delete(automation_id)
        self.remember_note(
            kind="automation",
            content=f"Automation deleted: {entry.name}",
            source=source,
            metadata={"automation_id": entry.automation_id},
        )
        self.ops_events.append(
            event="automation_deleted",
            message="An automation was deleted.",
            data={"automation_id": entry.automation_id, "name": entry.name},
        )
        return entry

    def due_time_automations(self, *, now: datetime | None = None) -> tuple[AutomationDefinition, ...]:
        return self.automation_store.due_time_automations(now=now)

    def matching_if_then_automations(
        self,
        *,
        facts: dict[str, object],
        event_name: str | None = None,
        now: datetime | None = None,
    ) -> tuple[AutomationDefinition, ...]:
        return self.automation_store.matching_if_then_automations(
            facts=facts,
            event_name=event_name,
            now=now,
        )

    def mark_automation_triggered(
        self,
        automation_id: str,
        *,
        triggered_at: datetime | None = None,
        source: str = "automation_execution",
    ) -> AutomationDefinition:
        entry = self.automation_store.mark_triggered(automation_id, triggered_at=triggered_at)
        self.remember_note(
            kind="automation",
            content=f"Automation ran: {entry.name}",
            source=source,
            metadata={"automation_id": entry.automation_id},
        )
        self.ops_events.append(
            event="automation_triggered",
            message="A scheduled automation was executed.",
            data={"automation_id": entry.automation_id, "name": entry.name},
        )
        return entry

    def reserve_due_reminders(self, *, limit: int = 1) -> tuple[ReminderEntry, ...]:
        return self.reminder_store.reserve_due(limit=limit)

    def mark_reminder_delivered(self, reminder_id: str) -> ReminderEntry:
        entry = self.reminder_store.mark_delivered(reminder_id)
        self.remember_note(
            kind="reminder",
            content=f"Reminder delivered: {entry.summary}",
            source="reminder_delivery",
            metadata={"reminder_id": entry.reminder_id},
        )
        self.ops_events.append(
            event="reminder_delivered",
            message="A due reminder was delivered successfully.",
            data={"reminder_id": entry.reminder_id, "summary": compact_text(entry.summary)},
        )
        return entry

    def mark_reminder_failed(self, reminder_id: str, *, error: str) -> ReminderEntry:
        entry = self.reminder_store.mark_failed(reminder_id, error=error)
        self.ops_events.append(
            event="reminder_delivery_failed",
            level="error",
            message="A due reminder could not be delivered.",
            data={
                "reminder_id": entry.reminder_id,
                "summary": compact_text(entry.summary),
                "error": error,
            },
        )
        return entry

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

    def begin_automation_print(self) -> TwinrStatus:
        try:
            status = self.state_machine.transition(TwinrEvent.PRINT_REQUESTED)
        except InvalidTransitionError:
            if self.state_machine.status != TwinrStatus.WAITING:
                raise
            status = self.state_machine.transition(TwinrEvent.YELLOW_BUTTON_PRESSED)
        self._persist_snapshot()
        self.ops_events.append(
            event="print_started",
            message="Scheduled automation requested a print.",
            data={"request_source": "automation", "status": status.value},
        )
        return status

    def maybe_begin_automation_print(self) -> TwinrStatus | None:
        try:
            return self.begin_automation_print()
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
            user_voice_status=self.user_voice_status,
            user_voice_confidence=self.user_voice_confidence,
            user_voice_checked_at=self.user_voice_checked_at,
        )

    def _restore_snapshot_context(self) -> None:
        snapshot = self.snapshot_store.load()
        self.last_transcript = snapshot.last_transcript
        self.last_response = snapshot.last_response or None
        self.user_voice_status = snapshot.user_voice_status or None
        self.user_voice_confidence = snapshot.user_voice_confidence
        self.user_voice_checked_at = snapshot.user_voice_checked_at or None
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
