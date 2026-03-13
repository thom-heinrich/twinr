from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Literal
import json

from twinr.memory.reminders import now_in_timezone, parse_due_at, resolve_timezone

ConditionOperator = Literal["eq", "ne", "contains", "gt", "gte", "lt", "lte", "truthy", "falsy"]
TimeSchedule = Literal["once", "daily", "weekly"]
ActionKind = Literal["say", "print", "tool_call", "llm_prompt"]

_SUPPORTED_OPERATORS = {"eq", "ne", "contains", "gt", "gte", "lt", "lte", "truthy", "falsy"}
_SUPPORTED_ACTIONS = {"say", "print", "tool_call", "llm_prompt"}
_SUPPORTED_SCHEDULES = {"once", "daily", "weekly"}


def _normalize_text(value: Any, *, limit: int) -> str:
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + "…"


def _slugify(value: Any, *, fallback: str) -> str:
    text = "".join(character if str(character).isalnum() else "_" for character in str(value or "").strip().lower())
    normalized = "_".join(part for part in text.split("_") if part)
    return normalized or fallback


def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    text = str(value).strip()
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def _parse_iso_timestamp(value: Any, *, timezone_name: str | None) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return parse_due_at(text, timezone_name=timezone_name)
    except ValueError:
        return None


def _parse_time_of_day(value: str) -> time:
    text = str(value or "").strip()
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError("time_of_day must use HH:MM")
    hour = int(parts[0])
    minute = int(parts[1])
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError("time_of_day must use HH:MM")
    return time(hour=hour, minute=minute)


def _normalize_weekdays(values: tuple[int, ...] | list[int] | None) -> tuple[int, ...]:
    if not values:
        return ()
    normalized = tuple(sorted({int(value) for value in values}))
    for value in normalized:
        if value < 0 or value > 6:
            raise ValueError("weekdays must use integers 0-6")
    return normalized


def _fact_value(facts: dict[str, Any], key: str) -> Any:
    current: Any = facts
    for part in key.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        return None
    return current


def _values_match(*, operator: str, actual: Any, expected: Any) -> bool:
    if operator == "truthy":
        return bool(actual)
    if operator == "falsy":
        return not bool(actual)
    if operator == "contains":
        if isinstance(actual, (list, tuple, set)):
            return any(str(item).strip().lower() == str(expected).strip().lower() for item in actual)
        return str(expected).strip().lower() in str(actual or "").strip().lower()

    left = _coerce_scalar(actual)
    right = _coerce_scalar(expected)

    if operator == "eq":
        return left == right
    if operator == "ne":
        return left != right
    if operator in {"gt", "gte", "lt", "lte"}:
        try:
            left_value = float(left)
            right_value = float(right)
        except (TypeError, ValueError):
            left_value = str(left or "")
            right_value = str(right or "")
        if operator == "gt":
            return left_value > right_value
        if operator == "gte":
            return left_value >= right_value
        if operator == "lt":
            return left_value < right_value
        return left_value <= right_value
    raise ValueError(f"Unsupported operator: {operator}")


@dataclass(frozen=True, slots=True)
class AutomationCondition:
    key: str
    operator: ConditionOperator
    value: str | int | float | bool | None = None

    def __post_init__(self) -> None:
        clean_key = _normalize_text(self.key, limit=120)
        clean_operator = str(self.operator).strip().lower()
        if not clean_key:
            raise ValueError("condition key must not be empty")
        if clean_operator not in _SUPPORTED_OPERATORS:
            raise ValueError(f"Unsupported condition operator: {self.operator}")
        object.__setattr__(self, "key", clean_key)
        object.__setattr__(self, "operator", clean_operator)
        if clean_operator in {"truthy", "falsy"}:
            object.__setattr__(self, "value", None)

    def matches(self, facts: dict[str, Any]) -> bool:
        return _values_match(
            operator=self.operator,
            actual=_fact_value(facts, self.key),
            expected=self.value,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "operator": self.operator,
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class TimeAutomationTrigger:
    kind: Literal["time"] = "time"
    schedule: TimeSchedule = "once"
    due_at: str | None = None
    time_of_day: str | None = None
    weekdays: tuple[int, ...] = ()
    timezone_name: str | None = None

    def __post_init__(self) -> None:
        clean_schedule = str(self.schedule).strip().lower()
        if clean_schedule not in _SUPPORTED_SCHEDULES:
            raise ValueError(f"Unsupported time schedule: {self.schedule}")
        object.__setattr__(self, "schedule", clean_schedule)
        object.__setattr__(self, "weekdays", _normalize_weekdays(self.weekdays))
        timezone_name = str(self.timezone_name or "").strip() or None
        object.__setattr__(self, "timezone_name", timezone_name)
        if clean_schedule == "once":
            if not str(self.due_at or "").strip():
                raise ValueError("due_at is required for once schedules")
            parse_due_at(str(self.due_at), timezone_name=timezone_name)
            object.__setattr__(self, "time_of_day", None)
            object.__setattr__(self, "weekdays", ())
            return
        parsed = _parse_time_of_day(str(self.time_of_day or ""))
        object.__setattr__(self, "time_of_day", parsed.strftime("%H:%M"))
        if clean_schedule == "daily":
            object.__setattr__(self, "due_at", None)
            object.__setattr__(self, "weekdays", ())
            return
        if not self.weekdays:
            raise ValueError("weekdays are required for weekly schedules")
        object.__setattr__(self, "due_at", None)

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "schedule": self.schedule,
            "due_at": self.due_at,
            "time_of_day": self.time_of_day,
            "weekdays": list(self.weekdays),
            "timezone_name": self.timezone_name,
        }


@dataclass(frozen=True, slots=True)
class IfThenAutomationTrigger:
    kind: Literal["if_then"] = "if_then"
    event_name: str | None = None
    all_conditions: tuple[AutomationCondition, ...] = ()
    any_conditions: tuple[AutomationCondition, ...] = ()
    cooldown_seconds: float = 0.0

    def __post_init__(self) -> None:
        event_name = _normalize_text(self.event_name, limit=120) or None
        cooldown_seconds = max(0.0, float(self.cooldown_seconds))
        if event_name is None and not self.all_conditions and not self.any_conditions:
            raise ValueError("if_then trigger requires an event_name or at least one condition")
        object.__setattr__(self, "event_name", event_name)
        object.__setattr__(self, "cooldown_seconds", cooldown_seconds)

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "event_name": self.event_name,
            "all_conditions": [condition.to_payload() for condition in self.all_conditions],
            "any_conditions": [condition.to_payload() for condition in self.any_conditions],
            "cooldown_seconds": self.cooldown_seconds,
        }


AutomationTrigger = TimeAutomationTrigger | IfThenAutomationTrigger


@dataclass(frozen=True, slots=True)
class AutomationAction:
    kind: ActionKind
    text: str | None = None
    tool_name: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def __post_init__(self) -> None:
        clean_kind = str(self.kind).strip().lower()
        if clean_kind not in _SUPPORTED_ACTIONS:
            raise ValueError(f"Unsupported action kind: {self.kind}")
        object.__setattr__(self, "kind", clean_kind)
        object.__setattr__(self, "text", _normalize_text(self.text, limit=420) or None)
        object.__setattr__(self, "tool_name", _normalize_text(self.tool_name, limit=120) or None)
        object.__setattr__(self, "payload", dict(self.payload or {}))
        if clean_kind in {"say", "print", "llm_prompt"} and not self.text:
            raise ValueError(f"text is required for action kind: {self.kind}")
        if clean_kind == "tool_call" and not self.tool_name:
            raise ValueError("tool_name is required for action kind: tool_call")

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "text": self.text,
            "tool_name": self.tool_name,
            "payload": dict(self.payload),
            "enabled": self.enabled,
        }


@dataclass(frozen=True, slots=True)
class AutomationDefinition:
    automation_id: str
    name: str
    trigger: AutomationTrigger
    actions: tuple[AutomationAction, ...]
    description: str | None = None
    enabled: bool = True
    created_at: datetime = field(default_factory=lambda: now_in_timezone("UTC"))
    updated_at: datetime = field(default_factory=lambda: now_in_timezone("UTC"))
    last_triggered_at: datetime | None = None
    source: str = "manual"
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        clean_name = _normalize_text(self.name, limit=120)
        if not clean_name:
            raise ValueError("automation name must not be empty")
        if not self.actions:
            raise ValueError("automation must include at least one action")
        clean_source = _slugify(self.source, fallback="manual")
        clean_tags = tuple(_slugify(tag, fallback="tag") for tag in self.tags if _normalize_text(tag, limit=40))
        object.__setattr__(self, "automation_id", _normalize_text(self.automation_id, limit=64))
        object.__setattr__(self, "name", clean_name)
        object.__setattr__(self, "description", _normalize_text(self.description, limit=220) or None)
        object.__setattr__(self, "source", clean_source)
        object.__setattr__(self, "tags", clean_tags)

    def to_payload(self) -> dict[str, Any]:
        return {
            "automation_id": self.automation_id,
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "trigger": self.trigger.to_payload(),
            "actions": [action.to_payload() for action in self.actions],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_triggered_at": self.last_triggered_at.isoformat() if self.last_triggered_at else None,
            "source": self.source,
            "tags": list(self.tags),
        }


class AutomationEngine:
    def due_time_automations(
        self,
        entries: tuple[AutomationDefinition, ...] | list[AutomationDefinition],
        *,
        now: datetime | None = None,
    ) -> tuple[AutomationDefinition, ...]:
        current_time = now or now_in_timezone("UTC")
        due: list[AutomationDefinition] = []
        for entry in entries:
            if not entry.enabled or not isinstance(entry.trigger, TimeAutomationTrigger):
                continue
            if self._time_trigger_due(entry, now=current_time):
                due.append(entry)
        return tuple(due)

    def matching_if_then_automations(
        self,
        entries: tuple[AutomationDefinition, ...] | list[AutomationDefinition],
        *,
        facts: dict[str, Any],
        event_name: str | None = None,
        now: datetime | None = None,
    ) -> tuple[AutomationDefinition, ...]:
        current_time = now or now_in_timezone("UTC")
        matches: list[AutomationDefinition] = []
        for entry in entries:
            if not entry.enabled or not isinstance(entry.trigger, IfThenAutomationTrigger):
                continue
            if self._if_then_trigger_matches(entry, facts=facts, event_name=event_name, now=current_time):
                matches.append(entry)
        return tuple(matches)

    def next_run_at(
        self,
        entry: AutomationDefinition,
        *,
        now: datetime | None = None,
    ) -> datetime | None:
        current_time = now or now_in_timezone("UTC")
        if isinstance(entry.trigger, TimeAutomationTrigger):
            return self._next_time_run(entry.trigger, last_triggered_at=entry.last_triggered_at, now=current_time)
        return None

    def tool_record(
        self,
        entry: AutomationDefinition,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        current_time = now or now_in_timezone("UTC")
        next_run = self.next_run_at(entry, now=current_time)
        due_now = False
        if isinstance(entry.trigger, TimeAutomationTrigger):
            due_now = self._time_trigger_due(entry, now=current_time)
        record = {
            **entry.to_payload(),
            "trigger_kind": entry.trigger.kind,
            "due_now": due_now,
            "next_run_at": next_run.isoformat() if next_run is not None else None,
        }
        if isinstance(entry.trigger, IfThenAutomationTrigger):
            from twinr.automations.sensors import describe_sensor_trigger

            sensor_trigger = describe_sensor_trigger(entry.trigger)
            record.update(
                {
                    "event_name": entry.trigger.event_name,
                    "cooldown_seconds": entry.trigger.cooldown_seconds,
                    "all_conditions": [condition.to_payload() for condition in entry.trigger.all_conditions],
                    "any_conditions": [condition.to_payload() for condition in entry.trigger.any_conditions],
                }
            )
            if sensor_trigger is not None:
                record.update(
                    {
                        "sensor_trigger_kind": sensor_trigger.trigger_kind,
                        "sensor_hold_seconds": sensor_trigger.hold_seconds,
                        "sensor_label": sensor_trigger.label,
                    }
                )
        return record

    def _time_trigger_due(self, entry: AutomationDefinition, *, now: datetime) -> bool:
        trigger = entry.trigger
        if not isinstance(trigger, TimeAutomationTrigger):
            return False
        zone = resolve_timezone(trigger.timezone_name)
        local_now = now.astimezone(zone)
        last_triggered_at = entry.last_triggered_at.astimezone(zone) if entry.last_triggered_at else None
        if trigger.schedule == "once":
            if last_triggered_at is not None:
                return False
            due_at = parse_due_at(str(trigger.due_at), timezone_name=trigger.timezone_name)
            return due_at <= now
        run_time = _parse_time_of_day(str(trigger.time_of_day or "00:00"))
        scheduled_at = local_now.replace(hour=run_time.hour, minute=run_time.minute, second=0, microsecond=0)
        if trigger.schedule == "daily":
            if scheduled_at > local_now:
                return False
            if last_triggered_at is not None and last_triggered_at >= scheduled_at:
                return False
            return True
        if local_now.weekday() not in trigger.weekdays:
            return False
        if scheduled_at > local_now:
            return False
        if last_triggered_at is not None and last_triggered_at >= scheduled_at:
            return False
        return True

    def _next_time_run(
        self,
        trigger: TimeAutomationTrigger,
        *,
        last_triggered_at: datetime | None,
        now: datetime,
    ) -> datetime | None:
        zone = resolve_timezone(trigger.timezone_name)
        local_now = now.astimezone(zone)
        local_last = last_triggered_at.astimezone(zone) if last_triggered_at is not None else None
        if trigger.schedule == "once":
            due_at = parse_due_at(str(trigger.due_at), timezone_name=trigger.timezone_name)
            if local_last is not None:
                return None
            return due_at
        run_time = _parse_time_of_day(str(trigger.time_of_day or "00:00"))
        today_at = local_now.replace(hour=run_time.hour, minute=run_time.minute, second=0, microsecond=0)
        if trigger.schedule == "daily":
            if today_at > local_now:
                return today_at
            if local_last is None or local_last < today_at:
                return today_at
            return today_at + timedelta(days=1)
        for offset in range(0, 8):
            candidate_day = local_now + timedelta(days=offset)
            if candidate_day.weekday() not in trigger.weekdays:
                continue
            candidate_at = candidate_day.replace(hour=run_time.hour, minute=run_time.minute, second=0, microsecond=0)
            if offset == 0 and candidate_at < local_now and (local_last is not None and local_last >= candidate_at):
                continue
            if offset == 0 and candidate_at < local_now and local_last is None:
                continue
            if local_last is not None and candidate_at <= local_last:
                continue
            return candidate_at
        return None

    def _if_then_trigger_matches(
        self,
        entry: AutomationDefinition,
        *,
        facts: dict[str, Any],
        event_name: str | None,
        now: datetime,
    ) -> bool:
        trigger = entry.trigger
        if not isinstance(trigger, IfThenAutomationTrigger):
            return False
        if trigger.event_name and trigger.event_name != _normalize_text(event_name, limit=120):
            return False
        if entry.last_triggered_at is not None and trigger.cooldown_seconds > 0:
            cooldown_until = entry.last_triggered_at + timedelta(seconds=trigger.cooldown_seconds)
            if cooldown_until > now:
                return False
        if trigger.all_conditions and not all(condition.matches(facts) for condition in trigger.all_conditions):
            return False
        if trigger.any_conditions and not any(condition.matches(facts) for condition in trigger.any_conditions):
            return False
        return True


class AutomationStore:
    def __init__(
        self,
        path: str | Path,
        *,
        timezone_name: str | None = None,
        max_entries: int = 96,
        engine: AutomationEngine | None = None,
    ) -> None:
        self.path = Path(path)
        self.timezone_name = timezone_name or "Europe/Berlin"
        self.max_entries = max(8, int(max_entries))
        self.engine = engine or AutomationEngine()

    def load_entries(self) -> tuple[AutomationDefinition, ...]:
        if not self.path.exists():
            return ()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ()
        items = payload.get("entries", [])
        if not isinstance(items, list):
            return ()
        entries: list[AutomationDefinition] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            entry = self._entry_from_payload(item)
            if entry is not None:
                entries.append(entry)
        return tuple(sorted(entries, key=lambda entry: (entry.name.lower(), entry.created_at)))

    def list_tool_records(self, *, now: datetime | None = None) -> tuple[dict[str, Any], ...]:
        return tuple(self.engine.tool_record(entry, now=now) for entry in self.load_entries())

    def render_context(self, *, limit: int = 8, now: datetime | None = None) -> str | None:
        current_time = now or now_in_timezone(self.timezone_name)
        enabled_entries = [entry for entry in self.load_entries() if entry.enabled]
        if not enabled_entries:
            return None
        lines = ["Active automations:"]
        for entry in enabled_entries[:limit]:
            schedule_text = self._describe_trigger(entry, now=current_time)
            action_text = self._describe_actions(entry.actions)
            lines.append(f"- {entry.name}: {schedule_text}. Action: {action_text}.")
        return "\n".join(lines).strip()

    def get(self, automation_id: str) -> AutomationDefinition | None:
        lookup = _normalize_text(automation_id, limit=64)
        for entry in self.load_entries():
            if entry.automation_id == lookup:
                return entry
        return None

    def create_time_automation(
        self,
        *,
        name: str,
        actions: tuple[AutomationAction, ...] | list[AutomationAction],
        description: str | None = None,
        enabled: bool = True,
        schedule: TimeSchedule = "once",
        due_at: str | None = None,
        time_of_day: str | None = None,
        weekdays: tuple[int, ...] | list[int] = (),
        timezone_name: str | None = None,
        source: str = "manual",
        tags: tuple[str, ...] | list[str] = (),
    ) -> AutomationDefinition:
        now = now_in_timezone(self.timezone_name)
        trigger = TimeAutomationTrigger(
            schedule=schedule,
            due_at=due_at,
            time_of_day=time_of_day,
            weekdays=tuple(weekdays),
            timezone_name=timezone_name or self.timezone_name,
        )
        entry = AutomationDefinition(
            automation_id=f"AUTO-{now.strftime('%Y%m%dT%H%M%S%fZ')}",
            name=name,
            description=description,
            enabled=enabled,
            trigger=trigger,
            actions=tuple(actions),
            created_at=now,
            updated_at=now,
            source=source,
            tags=tuple(tags),
        )
        return self.upsert(entry)

    def create_if_then_automation(
        self,
        *,
        name: str,
        actions: tuple[AutomationAction, ...] | list[AutomationAction],
        description: str | None = None,
        enabled: bool = True,
        event_name: str | None = None,
        all_conditions: tuple[AutomationCondition, ...] | list[AutomationCondition] = (),
        any_conditions: tuple[AutomationCondition, ...] | list[AutomationCondition] = (),
        cooldown_seconds: float = 0.0,
        source: str = "manual",
        tags: tuple[str, ...] | list[str] = (),
    ) -> AutomationDefinition:
        now = now_in_timezone(self.timezone_name)
        trigger = IfThenAutomationTrigger(
            event_name=event_name,
            all_conditions=tuple(all_conditions),
            any_conditions=tuple(any_conditions),
            cooldown_seconds=cooldown_seconds,
        )
        entry = AutomationDefinition(
            automation_id=f"AUTO-{now.strftime('%Y%m%dT%H%M%S%fZ')}",
            name=name,
            description=description,
            enabled=enabled,
            trigger=trigger,
            actions=tuple(actions),
            created_at=now,
            updated_at=now,
            source=source,
            tags=tuple(tags),
        )
        return self.upsert(entry)

    def upsert(self, entry: AutomationDefinition) -> AutomationDefinition:
        entries = list(self.load_entries())
        normalized = replace(entry, updated_at=now_in_timezone(self.timezone_name))
        for index, existing in enumerate(entries):
            if existing.automation_id != normalized.automation_id:
                continue
            entries[index] = normalized
            self._write_entries(tuple(entries))
            return normalized
        entries.append(normalized)
        self._write_entries(tuple(entries))
        return normalized

    def update(
        self,
        automation_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        enabled: bool | None = None,
        trigger: AutomationTrigger | None = None,
        actions: tuple[AutomationAction, ...] | list[AutomationAction] | None = None,
        source: str | None = None,
        tags: tuple[str, ...] | list[str] | None = None,
    ) -> AutomationDefinition:
        existing = self.get(automation_id)
        if existing is None:
            raise KeyError(f"Unknown automation_id: {automation_id}")
        updated = replace(
            existing,
            name=name if name is not None else existing.name,
            description=description if description is not None else existing.description,
            enabled=enabled if enabled is not None else existing.enabled,
            trigger=trigger if trigger is not None else existing.trigger,
            actions=tuple(actions) if actions is not None else existing.actions,
            source=source if source is not None else existing.source,
            tags=tuple(tags) if tags is not None else existing.tags,
            updated_at=now_in_timezone(self.timezone_name),
        )
        return self.upsert(updated)

    def delete(self, automation_id: str) -> AutomationDefinition:
        entries = list(self.load_entries())
        lookup = _normalize_text(automation_id, limit=64)
        for index, entry in enumerate(entries):
            if entry.automation_id != lookup:
                continue
            removed = entries.pop(index)
            self._write_entries(tuple(entries))
            return removed
        raise KeyError(f"Unknown automation_id: {automation_id}")

    def mark_triggered(self, automation_id: str, *, triggered_at: datetime | None = None) -> AutomationDefinition:
        existing = self.get(automation_id)
        if existing is None:
            raise KeyError(f"Unknown automation_id: {automation_id}")
        return self.upsert(
            replace(
                existing,
                last_triggered_at=triggered_at or now_in_timezone(self.timezone_name),
                updated_at=triggered_at or now_in_timezone(self.timezone_name),
            )
        )

    def due_time_automations(self, *, now: datetime | None = None) -> tuple[AutomationDefinition, ...]:
        return self.engine.due_time_automations(self.load_entries(), now=now)

    def matching_if_then_automations(
        self,
        *,
        facts: dict[str, Any],
        event_name: str | None = None,
        now: datetime | None = None,
    ) -> tuple[AutomationDefinition, ...]:
        return self.engine.matching_if_then_automations(
            self.load_entries(),
            facts=facts,
            event_name=event_name,
            now=now,
        )

    def _write_entries(self, entries: tuple[AutomationDefinition, ...]) -> None:
        normalized_entries = sorted(entries, key=lambda entry: (entry.name.lower(), entry.created_at))[: self.max_entries]
        payload = {
            "updated_at": now_in_timezone(self.timezone_name).isoformat(),
            "entries": [entry.to_payload() for entry in normalized_entries],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _entry_from_payload(self, payload: dict[str, Any]) -> AutomationDefinition | None:
        automation_id = _normalize_text(payload.get("automation_id"), limit=64)
        name = _normalize_text(payload.get("name"), limit=120)
        trigger_payload = payload.get("trigger")
        actions_payload = payload.get("actions")
        if not automation_id or not name or not isinstance(trigger_payload, dict) or not isinstance(actions_payload, list):
            return None
        trigger = self._trigger_from_payload(trigger_payload)
        if trigger is None:
            return None
        actions = tuple(action for action in (self._action_from_payload(item) for item in actions_payload) if action is not None)
        if not actions:
            return None
        return AutomationDefinition(
            automation_id=automation_id,
            name=name,
            description=_normalize_text(payload.get("description"), limit=220) or None,
            enabled=bool(payload.get("enabled", True)),
            trigger=trigger,
            actions=actions,
            created_at=_parse_iso_timestamp(payload.get("created_at"), timezone_name=self.timezone_name)
            or now_in_timezone(self.timezone_name),
            updated_at=_parse_iso_timestamp(payload.get("updated_at"), timezone_name=self.timezone_name)
            or now_in_timezone(self.timezone_name),
            last_triggered_at=_parse_iso_timestamp(payload.get("last_triggered_at"), timezone_name=self.timezone_name),
            source=_slugify(payload.get("source"), fallback="manual"),
            tags=tuple(str(item).strip() for item in payload.get("tags", []) if str(item).strip()),
        )

    def _trigger_from_payload(self, payload: dict[str, Any]) -> AutomationTrigger | None:
        kind = str(payload.get("kind", "")).strip().lower()
        if kind == "time":
            try:
                return TimeAutomationTrigger(
                    schedule=str(payload.get("schedule", "once")),
                    due_at=payload.get("due_at"),
                    time_of_day=payload.get("time_of_day"),
                    weekdays=tuple(int(item) for item in payload.get("weekdays", []) if str(item).strip()),
                    timezone_name=payload.get("timezone_name") or self.timezone_name,
                )
            except (TypeError, ValueError):
                return None
        if kind == "if_then":
            all_conditions = tuple(
                condition
                for condition in (self._condition_from_payload(item) for item in payload.get("all_conditions", []))
                if condition is not None
            )
            any_conditions = tuple(
                condition
                for condition in (self._condition_from_payload(item) for item in payload.get("any_conditions", []))
                if condition is not None
            )
            try:
                return IfThenAutomationTrigger(
                    event_name=payload.get("event_name"),
                    all_conditions=all_conditions,
                    any_conditions=any_conditions,
                    cooldown_seconds=float(payload.get("cooldown_seconds", 0.0) or 0.0),
                )
            except ValueError:
                return None
        return None

    def _condition_from_payload(self, payload: Any) -> AutomationCondition | None:
        if not isinstance(payload, dict):
            return None
        try:
            return AutomationCondition(
                key=str(payload.get("key", "")),
                operator=str(payload.get("operator", "eq")),
                value=payload.get("value"),
            )
        except ValueError:
            return None

    def _action_from_payload(self, payload: Any) -> AutomationAction | None:
        if not isinstance(payload, dict):
            return None
        try:
            return AutomationAction(
                kind=str(payload.get("kind", "")),
                text=payload.get("text"),
                tool_name=payload.get("tool_name"),
                payload=dict(payload.get("payload", {}) or {}),
                enabled=bool(payload.get("enabled", True)),
            )
        except (TypeError, ValueError):
            return None

    def _describe_trigger(self, entry: AutomationDefinition, *, now: datetime) -> str:
        trigger = entry.trigger
        from twinr.automations.sensors import describe_sensor_trigger_text

        sensor_text = describe_sensor_trigger_text(trigger)
        if sensor_text:
            return sensor_text
        if not isinstance(trigger, TimeAutomationTrigger):
            return f"{trigger.kind} trigger"
        if trigger.schedule == "once":
            due_at = parse_due_at(str(trigger.due_at), timezone_name=trigger.timezone_name)
            return f"once at {due_at.isoformat()}"
        if trigger.schedule == "daily":
            return f"every day at {trigger.time_of_day}"
        weekday_names = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
        weekdays = ", ".join(weekday_names[index] for index in trigger.weekdays)
        next_run = self.engine.next_run_at(entry, now=now)
        if next_run is not None:
            return f"weekly on {weekdays} at {trigger.time_of_day}; next {next_run.isoformat()}"
        return f"weekly on {weekdays} at {trigger.time_of_day}"

    def _describe_actions(self, actions: tuple[AutomationAction, ...]) -> str:
        parts: list[str] = []
        for action in actions:
            if not action.enabled:
                continue
            if action.kind == "llm_prompt":
                delivery = str(action.payload.get("delivery", "spoken")).strip().lower() or "spoken"
                search = bool(action.payload.get("allow_web_search"))
                detail = f"{delivery} llm prompt"
                if search:
                    detail += " with web search"
                if action.text:
                    detail += f" ({_normalize_text(action.text, limit=80)})"
                parts.append(detail)
                continue
            if action.kind in {"say", "print"}:
                parts.append(f"{action.kind} {_normalize_text(action.text or '', limit=80)}".strip())
                continue
            parts.append(f"{action.kind} {action.tool_name or ''}".strip())
        return "; ".join(part for part in parts if part) or "no enabled action"
