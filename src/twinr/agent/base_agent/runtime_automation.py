from __future__ import annotations

import logging
import math
import re
from copy import deepcopy
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from twinr.automations import AutomationCondition, AutomationDefinition
from twinr.memory.reminders import ReminderEntry, format_due_label
from twinr.ops.events import compact_text


logger = logging.getLogger(__name__)

_SECRET_KV_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|token|secret|password|passwd|authorization)\b\s*[:=]\s*([^\s,;]+)"
)
_BEARER_TOKEN_PATTERN = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_BASIC_TOKEN_PATTERN = re.compile(r"(?i)\bbasic\s+[A-Za-z0-9._~+/=-]+")


class TwinrRuntimeAutomationMixin:
    # AUDIT-FIX(#5,#6,#7): Add 3.11-compatible validation/snapshot helpers at the module boundary
    # so naive datetimes, mutable caller-owned inputs, and blank identifiers are handled before they
    # reach file-backed stores.
    def _local_timezone_name(self) -> str | None:
        timezone_name = getattr(getattr(self, "config", None), "local_timezone_name", None)
        if not isinstance(timezone_name, str):
            return None
        timezone_name = timezone_name.strip()
        return timezone_name or None

    def _normalize_datetime_arg(self, value: datetime | None, *, field_name: str) -> datetime | None:
        if value is None:
            return None
        if not isinstance(value, datetime):
            raise TypeError(f"{field_name} must be a datetime or None.")
        if value.tzinfo is not None and value.utcoffset() is not None:
            return value

        timezone_name = self._local_timezone_name()
        if timezone_name is None:
            raise ValueError(f"{field_name} must be timezone-aware when no local timezone is configured.")
        try:
            return value.replace(tzinfo=ZoneInfo(timezone_name))
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Invalid local timezone configuration: {timezone_name!r}.") from exc

    def _snapshot_value(self, value: Any) -> Any:
        if value is None:
            return None
        try:
            return deepcopy(value)
        except Exception:
            logger.exception("Failed to snapshot runtime automation input; using original object.")
            return value

    def _require_non_empty_identifier(self, value: str, *, field_name: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string.")
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{field_name} must not be empty.")
        return normalized

    def _require_positive_int(self, value: int, *, field_name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{field_name} must be an integer.")
        if value < 1:
            raise ValueError(f"{field_name} must be >= 1.")
        return value

    def _require_non_negative_seconds(self, value: float, *, field_name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{field_name} must be a real number.")
        numeric_value = float(value)
        if not math.isfinite(numeric_value) or numeric_value < 0.0:
            raise ValueError(f"{field_name} must be a finite number >= 0.")
        return numeric_value

    # AUDIT-FIX(#4): Redact obvious secret-bearing substrings and collapse multiline exception text
    # before persisting it into state or ops logs.
    def _sanitize_error_text(self, error: object) -> str:
        text = "" if error is None else str(error)
        text = text.replace("\r", " ").replace("\n", " ").strip()
        text = _SECRET_KV_PATTERN.sub(lambda match: f"{match.group(1)}=[REDACTED]", text)
        text = _BEARER_TOKEN_PATTERN.sub("Bearer [REDACTED]", text)
        text = _BASIC_TOKEN_PATTERN.sub("Basic [REDACTED]", text)
        return self._compact_text_value(text)

    def _compact_text_value(self, value: object) -> str:
        text = "" if value is None else str(value)
        try:
            return compact_text(text)
        except Exception:
            return text

    def _safe_isoformat(self, value: object) -> str | None:
        if value is None:
            return None
        isoformat = getattr(value, "isoformat", None)
        if callable(isoformat):
            try:
                return isoformat()
            except Exception:
                logger.exception("Failed to format datetime-like value for ops event payload.")
        return self._compact_text_value(value)

    def _format_due_label_or_fallback(self, due_at: object) -> str:
        if isinstance(due_at, datetime):
            timezone_name = self._local_timezone_name()
            if timezone_name is not None:
                try:
                    return format_due_label(due_at, timezone_name=timezone_name)
                except Exception:
                    logger.exception("Failed to format reminder due label.")
            try:
                return due_at.isoformat()
            except Exception:
                logger.exception("Failed to fallback-format reminder due timestamp.")
        return "the requested time"

    # AUDIT-FIX(#1): Never let note/event side effects turn a successful store mutation into a visible
    # failure. Downstream retries would otherwise duplicate reminders/automations.
    def _safe_remember_note(
        self,
        *,
        kind: str,
        content: str,
        source: str,
        metadata: dict[str, object] | None = None,
    ) -> None:
        try:
            self.remember_note(
                kind=kind,
                content=content,
                source=source,
                metadata=metadata,
            )
        except Exception:
            logger.exception("Failed to persist runtime note after successful state mutation.")

    def _safe_append_ops_event(
        self,
        *,
        event: str,
        message: str,
        data: dict[str, object],
        level: str | None = None,
    ) -> None:
        kwargs: dict[str, object] = {
            "event": event,
            "message": message,
            "data": data,
        }
        if level is not None:
            kwargs["level"] = level
        try:
            self.ops_events.append(**kwargs)
        except Exception:
            logger.exception("Failed to append ops event after successful state mutation: %s", event)

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
        # AUDIT-FIX(#1): Build note/event payloads defensively and emit them best-effort only.
        due_label = self._format_due_label_or_fallback(getattr(entry, "due_at", None))
        self._safe_remember_note(
            kind="reminder",
            content=f"Reminder scheduled for {due_label}: {getattr(entry, 'summary', summary)}",
            source=source,
            metadata={
                "reminder_id": getattr(entry, "reminder_id", None),
                "reminder_kind": getattr(entry, "kind", kind),
            },
        )
        self._safe_append_ops_event(
            event="reminder_scheduled",
            message="A reminder or timer was scheduled.",
            data={
                "reminder_id": getattr(entry, "reminder_id", None),
                "kind": getattr(entry, "kind", kind),
                "due_at": self._safe_isoformat(getattr(entry, "due_at", None)),
                "summary": self._compact_text_value(getattr(entry, "summary", summary)),
            },
        )
        return entry

    def list_automation_records(self, *, now: datetime | None = None) -> tuple[dict[str, object], ...]:
        # AUDIT-FIX(#5): Normalize boundary datetimes so store queries are timezone-stable on Python 3.11.
        return self.automation_store.list_tool_records(
            now=self._normalize_datetime_arg(now, field_name="now")
        )

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
        # AUDIT-FIX(#6): Snapshot mutable caller inputs before handing them to the automation store.
        entry = self.automation_store.create_time_automation(
            name=name,
            description=description,
            enabled=enabled,
            schedule=schedule,
            due_at=due_at,
            time_of_day=time_of_day,
            weekdays=self._snapshot_value(weekdays),
            timezone_name=timezone_name or self._local_timezone_name(),
            actions=self._snapshot_value(actions),
            source=source,
            tags=self._snapshot_value(tags),
        )
        trigger = getattr(entry, "trigger", None)
        # AUDIT-FIX(#1): Note/event writes are post-commit side effects and must not bubble exceptions.
        self._safe_remember_note(
            kind="automation",
            content=f"Time automation created: {getattr(entry, 'name', name)}",
            source=source,
            metadata={
                "automation_id": getattr(entry, "automation_id", None),
                "trigger_kind": getattr(trigger, "kind", None),
            },
        )
        self._safe_append_ops_event(
            event="automation_created",
            message="A time-based automation was created.",
            data={
                "automation_id": getattr(entry, "automation_id", None),
                "name": self._compact_text_value(getattr(entry, "name", name)),
                "schedule": getattr(trigger, "schedule", getattr(trigger, "kind", None)),
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
        # AUDIT-FIX(#2): Reject negative/NaN/infinite cooldowns before they disable debounce semantics.
        cooldown_seconds = self._require_non_negative_seconds(
            cooldown_seconds,
            field_name="cooldown_seconds",
        )
        # AUDIT-FIX(#6): Snapshot mutable caller inputs before handing them to the automation store.
        entry = self.automation_store.create_if_then_automation(
            name=name,
            description=description,
            enabled=enabled,
            event_name=event_name,
            all_conditions=self._snapshot_value(all_conditions),
            any_conditions=self._snapshot_value(any_conditions),
            cooldown_seconds=cooldown_seconds,
            actions=self._snapshot_value(actions),
            source=source,
            tags=self._snapshot_value(tags),
        )
        trigger = getattr(entry, "trigger", None)
        # AUDIT-FIX(#1): Note/event writes are post-commit side effects and must not bubble exceptions.
        self._safe_remember_note(
            kind="automation",
            content=f"Sensor automation created: {getattr(entry, 'name', name)}",
            source=source,
            metadata={
                "automation_id": getattr(entry, "automation_id", None),
                "trigger_kind": getattr(trigger, "kind", None),
            },
        )
        self._safe_append_ops_event(
            event="automation_created",
            message="An if-then automation was created.",
            data={
                "automation_id": getattr(entry, "automation_id", None),
                "name": self._compact_text_value(getattr(entry, "name", name)),
                "event_name": self._compact_text_value(getattr(trigger, "event_name", event_name)),
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
        # AUDIT-FIX(#7): Reject blank identifiers before they hit file-backed store resolution.
        automation_id = self._require_non_empty_identifier(automation_id, field_name="automation_id")
        # AUDIT-FIX(#6): Snapshot mutable caller inputs before handing them to the automation store.
        entry = self.automation_store.update(
            automation_id,
            name=name,
            description=description,
            enabled=enabled,
            trigger=self._snapshot_value(trigger),
            actions=self._snapshot_value(actions),
            source=source,
            tags=self._snapshot_value(tags),
        )
        trigger_value = getattr(entry, "trigger", None)
        # AUDIT-FIX(#1): Note/event writes are post-commit side effects and must not bubble exceptions.
        self._safe_remember_note(
            kind="automation",
            content=f"Automation updated: {getattr(entry, 'name', name) or automation_id}",
            source=source or "automation_update",
            metadata={
                "automation_id": getattr(entry, "automation_id", automation_id),
                "trigger_kind": getattr(trigger_value, "kind", None),
            },
        )
        self._safe_append_ops_event(
            event="automation_updated",
            message="An automation was updated.",
            data={
                "automation_id": getattr(entry, "automation_id", automation_id),
                "name": self._compact_text_value(getattr(entry, "name", name) or automation_id),
            },
        )
        return entry

    def delete_automation(self, automation_id: str, *, source: str = "tool") -> AutomationDefinition:
        # AUDIT-FIX(#7): Reject blank identifiers before they hit file-backed store resolution.
        automation_id = self._require_non_empty_identifier(automation_id, field_name="automation_id")
        entry = self.automation_store.delete(automation_id)
        # AUDIT-FIX(#1): Note/event writes are post-commit side effects and must not bubble exceptions.
        self._safe_remember_note(
            kind="automation",
            content=f"Automation deleted: {getattr(entry, 'name', automation_id)}",
            source=source,
            metadata={"automation_id": getattr(entry, "automation_id", automation_id)},
        )
        self._safe_append_ops_event(
            event="automation_deleted",
            message="An automation was deleted.",
            data={
                "automation_id": getattr(entry, "automation_id", automation_id),
                "name": self._compact_text_value(getattr(entry, "name", automation_id)),
            },
        )
        return entry

    def due_time_automations(self, *, now: datetime | None = None) -> tuple[AutomationDefinition, ...]:
        # AUDIT-FIX(#5): Normalize boundary datetimes so store queries are timezone-stable on Python 3.11.
        return self.automation_store.due_time_automations(
            now=self._normalize_datetime_arg(now, field_name="now")
        )

    def matching_if_then_automations(
        self,
        *,
        facts: dict[str, object],
        event_name: str | None = None,
        now: datetime | None = None,
    ) -> tuple[AutomationDefinition, ...]:
        # AUDIT-FIX(#5,#6): Normalize datetimes and snapshot fact payloads before evaluation.
        return self.automation_store.matching_if_then_automations(
            facts=self._snapshot_value(facts),
            event_name=event_name,
            now=self._normalize_datetime_arg(now, field_name="now"),
        )

    def mark_automation_triggered(
        self,
        automation_id: str,
        *,
        triggered_at: datetime | None = None,
        source: str = "automation_execution",
    ) -> AutomationDefinition:
        # AUDIT-FIX(#5,#7): Validate identifiers and normalize runtime timestamps before persistence.
        automation_id = self._require_non_empty_identifier(automation_id, field_name="automation_id")
        entry = self.automation_store.mark_triggered(
            automation_id,
            triggered_at=self._normalize_datetime_arg(triggered_at, field_name="triggered_at"),
        )
        # AUDIT-FIX(#1): Note/event writes are post-commit side effects and must not bubble exceptions.
        self._safe_remember_note(
            kind="automation",
            content=f"Automation ran: {getattr(entry, 'name', automation_id)}",
            source=source,
            metadata={"automation_id": getattr(entry, "automation_id", automation_id)},
        )
        self._safe_append_ops_event(
            event="automation_triggered",
            message="A scheduled automation was executed.",
            data={
                "automation_id": getattr(entry, "automation_id", automation_id),
                "name": self._compact_text_value(getattr(entry, "name", automation_id)),
            },
        )
        return entry

    def reserve_due_reminders(self, *, limit: int = 1) -> tuple[ReminderEntry, ...]:
        # AUDIT-FIX(#3): Prevent undefined reserve semantics caused by zero/negative/non-int limits.
        return self.reminder_store.reserve_due(limit=self._require_positive_int(limit, field_name="limit"))

    def peek_due_reminders(self, *, limit: int = 1) -> tuple[ReminderEntry, ...]:
        # AUDIT-FIX(#3): Prevent undefined peek semantics caused by zero/negative/non-int limits.
        return self.reminder_store.peek_due(limit=self._require_positive_int(limit, field_name="limit"))

    def mark_reminder_delivered(self, reminder_id: str) -> ReminderEntry:
        # AUDIT-FIX(#7): Reject blank identifiers before they hit file-backed store resolution.
        reminder_id = self._require_non_empty_identifier(reminder_id, field_name="reminder_id")
        entry = self.reminder_store.mark_delivered(reminder_id)
        # AUDIT-FIX(#1): Note/event writes are post-commit side effects and must not bubble exceptions.
        self._safe_remember_note(
            kind="reminder",
            content=f"Reminder delivered: {getattr(entry, 'summary', reminder_id)}",
            source="reminder_delivery",
            metadata={"reminder_id": getattr(entry, "reminder_id", reminder_id)},
        )
        self._safe_append_ops_event(
            event="reminder_delivered",
            message="A due reminder was delivered successfully.",
            data={
                "reminder_id": getattr(entry, "reminder_id", reminder_id),
                "summary": self._compact_text_value(getattr(entry, "summary", reminder_id)),
            },
        )
        return entry

    def mark_reminder_failed(self, reminder_id: str, *, error: str) -> ReminderEntry:
        # AUDIT-FIX(#4,#7): Validate IDs and redact error text before writing it into persistent state.
        reminder_id = self._require_non_empty_identifier(reminder_id, field_name="reminder_id")
        sanitized_error = self._sanitize_error_text(error)
        entry = self.reminder_store.mark_failed(reminder_id, error=sanitized_error)
        # AUDIT-FIX(#1): Ops-event emission is post-commit and must not mask the store outcome.
        self._safe_append_ops_event(
            event="reminder_delivery_failed",
            level="error",
            message="A due reminder could not be delivered.",
            data={
                "reminder_id": getattr(entry, "reminder_id", reminder_id),
                "summary": self._compact_text_value(getattr(entry, "summary", reminder_id)),
                "error": sanitized_error,
            },
        )
        return entry