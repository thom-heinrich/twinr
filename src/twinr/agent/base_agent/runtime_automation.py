from __future__ import annotations

from datetime import datetime

from twinr.automations import AutomationCondition, AutomationDefinition
from twinr.memory.reminders import ReminderEntry, format_due_label
from twinr.ops.events import compact_text


class TwinrRuntimeAutomationMixin:
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

    def peek_due_reminders(self, *, limit: int = 1) -> tuple[ReminderEntry, ...]:
        return self.reminder_store.peek_due(limit=limit)

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
