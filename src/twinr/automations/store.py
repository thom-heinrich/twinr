# CHANGELOG: 2026-03-28
# BUG-1: Fixed real cross-process lost-update races in the JSON store by adding an advisory file lock
# BUG-2: Fixed missed daily/weekly runs after downtime/restart by tracking scheduled fire times and computing pending occurrences
# BUG-3: Fixed backup recovery when the primary store file is missing but the backup still exists
# SEC-1: Added bounded store-file reads/writes to avoid practical memory/disk exhaustion on Raspberry Pi deployments
# SEC-2: Hardened lock/store file creation to private permissions and safe lock-file path validation
# IMP-1: Added APScheduler-style time semantics: misfire_grace_seconds, coalesce_policy, scheduled-for tracking, and due match metadata
# IMP-2: Added optional msgspec read fast-path plus cache invalidation/signatures for lower-overhead Pi 4 operation
"""Define Twinr automation models, scheduling logic, and file-backed storage.

This module owns the canonical automation data model used across the runtime,
web UI, and tool handlers. It validates operator-managed automations, evaluates
time and fact-based triggers, and persists them to a bounded JSON store.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import Any, ClassVar, Iterator, Literal
from uuid import uuid4
import json
import math
import os
import tempfile

try:  # Linux/Raspberry Pi path for real inter-process locking.
    import fcntl  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - non-Unix fallback
    fcntl = None  # type: ignore[assignment]

try:  # Optional frontier fast-path; keep the module drop-in without extra deps.
    import msgspec  # type: ignore[import-not-found]  # pylint: disable=import-error
except Exception:  # pragma: no cover - optional dependency
    msgspec = None  # type: ignore[assignment]

from twinr.memory.reminders import now_in_timezone, parse_due_at, resolve_timezone

ConditionOperator = Literal["eq", "ne", "contains", "gt", "gte", "lt", "lte", "truthy", "falsy"]
TimeSchedule = Literal["once", "daily", "weekly"]
ActionKind = Literal["say", "print", "tool_call", "llm_prompt"]
CoalescePolicy = Literal["latest", "earliest", "all"]

_SUPPORTED_OPERATORS = {"eq", "ne", "contains", "gt", "gte", "lt", "lte", "truthy", "falsy"}
_SUPPORTED_ACTIONS = {"say", "print", "tool_call", "llm_prompt"}
_SUPPORTED_SCHEDULES = {"once", "daily", "weekly"}
_SUPPORTED_COALESCE_POLICIES = {"latest", "earliest", "all"}
_DEFAULT_MAX_STORE_BYTES = 2 * 1024 * 1024
_MAX_PENDING_OCCURRENCES = 512
_PAYLOAD_SCHEMA_VERSION = 2
_UTC = timezone.utc
_CROSS_SERVICE_STORE_FILE_MODE = 0o600
_CROSS_SERVICE_LOCK_FILE_MODE = 0o600


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


def _normalize_event_name(value: Any) -> str | None:
    text = _normalize_text(value, limit=120)
    if not text:
        return None
    parts: list[str] = []
    for character in text.lower():
        if character.isalnum():
            parts.append(character)
            continue
        if character in {".", "_"}:
            if parts and parts[-1] != character:
                parts.append(character)
            continue
        if parts and parts[-1] != "_":
            parts.append("_")
    normalized = "".join(parts).strip("._")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    while ".." in normalized:
        normalized = normalized.replace("..", ".")
    normalized = normalized.replace("._", ".").replace("_.", ".")
    return normalized or None


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


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            return default
        return value != 0
    text = str(value).strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _coerce_int(value: Any, *, default: int) -> int:
    try:
        return int(str(value).strip())
    except (AttributeError, TypeError, ValueError):
        return default


def _parse_non_negative_float(value: Any, *, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite number") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field_name} must be a finite number")
    return number


def _parse_optional_non_negative_float(value: Any, *, field_name: str) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return _parse_non_negative_float(value, field_name=field_name)


def _canonical_timezone_name(timezone_name: str | None, *, fallback: str) -> str:
    candidate = str(timezone_name or "").strip() or fallback
    try:
        zone = resolve_timezone(candidate)
    except Exception:
        zone = resolve_timezone(fallback)
        candidate = fallback
    return str(getattr(zone, "key", candidate) or candidate)


def _ensure_aware_datetime(value: datetime, *, timezone_name: str) -> datetime:
    zone = resolve_timezone(_canonical_timezone_name(timezone_name, fallback="UTC"))
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=zone)
    return value


def _to_utc_datetime(value: datetime, *, timezone_name: str) -> datetime:
    return _ensure_aware_datetime(value, timezone_name=timezone_name).astimezone(_UTC)


def _parse_due_at_absolute(value: Any, *, timezone_name: str | None) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError("due_at must not be empty")
    zone_name = _canonical_timezone_name(timezone_name, fallback="UTC")
    parsed = parse_due_at(text, timezone_name=zone_name)
    return _ensure_aware_datetime(parsed, timezone_name=zone_name)


def _parse_iso_timestamp(value: Any, *, timezone_name: str | None) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = _parse_due_at_absolute(text, timezone_name=timezone_name)
    except (TypeError, ValueError):
        return None
    return _to_utc_datetime(parsed, timezone_name=timezone_name or "UTC")


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


def _normalize_coalesce_policy(value: Any, *, default: str = "latest") -> CoalescePolicy:
    text = str(value or default).strip().lower() or default
    if text not in _SUPPORTED_COALESCE_POLICIES:
        raise ValueError(f"Unsupported coalesce policy: {value}")
    return text  # type: ignore[return-value]


def _json_safe_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 8:
        return _normalize_text(repr(value), limit=240)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, datetime):
        return _to_utc_datetime(value, timezone_name="UTC").isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item, depth=depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_value(item, depth=depth + 1) for item in value]
    return _normalize_text(repr(value), limit=240)


def _normalize_store_path(path: str | Path) -> Path:
    return Path(os.path.abspath(str(Path(path).expanduser())))


def _path_signature(path: Path) -> tuple[bool, int, int]:
    try:
        stat = path.stat()
    except OSError:
        return (False, 0, 0)
    return (True, int(stat.st_mtime_ns), int(stat.st_size))


def _valid_local_candidates(local_naive: datetime, *, zone: Any) -> tuple[datetime, ...]:
    candidates: list[datetime] = []
    seen: set[str] = set()
    for fold in (0, 1):
        candidate = local_naive.replace(tzinfo=zone, fold=fold)
        roundtrip = candidate.astimezone(_UTC).astimezone(zone)
        if roundtrip.replace(tzinfo=None) != local_naive:
            continue
        key = roundtrip.isoformat()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(roundtrip)
    candidates.sort(key=lambda item: item.astimezone(_UTC))
    return tuple(candidates)


def _resolve_local_schedule_datetime(schedule_date: date, run_time: time, *, zone: Any) -> datetime:
    local_naive = datetime.combine(schedule_date, run_time)
    candidates = _valid_local_candidates(local_naive, zone=zone)
    if candidates:
        return candidates[0]
    for minute_offset in range(1, 181):
        shifted = local_naive + timedelta(minutes=minute_offset)
        shifted_candidates = _valid_local_candidates(shifted, zone=zone)
        if shifted_candidates:
            return shifted_candidates[0]
    raise ValueError(f"Could not resolve scheduled local time for {schedule_date.isoformat()} {run_time.isoformat()}")


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


def _safe_describe_sensor_trigger(trigger: AutomationTrigger) -> Any | None:
    try:
        from twinr.automations.sensors import describe_sensor_trigger
    except Exception:
        return None
    try:
        return describe_sensor_trigger(trigger)
    except Exception:
        return None


def _safe_describe_sensor_trigger_text(trigger: AutomationTrigger) -> str | None:
    try:
        from twinr.automations.sensors import describe_sensor_trigger_text
    except Exception:
        return None
    try:
        sensor_text = describe_sensor_trigger_text(trigger)
    except Exception:
        return None
    return _normalize_text(sensor_text, limit=220) or None


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
        return _values_match(operator=self.operator, actual=_fact_value(facts, self.key), expected=self.value)

    def to_payload(self) -> dict[str, Any]:
        return {"key": self.key, "operator": self.operator, "value": self.value}


@dataclass(frozen=True, slots=True)
class TimeAutomationTrigger:
    kind: Literal["time"] = "time"
    schedule: TimeSchedule = "once"
    due_at: str | None = None
    time_of_day: str | None = None
    weekdays: tuple[int, ...] = ()
    timezone_name: str | None = None
    misfire_grace_seconds: float | None = None
    coalesce_policy: CoalescePolicy = "latest"

    def __post_init__(self) -> None:
        clean_kind = str(self.kind).strip().lower()
        if clean_kind != "time":
            raise ValueError("TimeAutomationTrigger.kind must be 'time'")
        clean_schedule = str(self.schedule).strip().lower()
        if clean_schedule not in _SUPPORTED_SCHEDULES:
            raise ValueError(f"Unsupported time schedule: {self.schedule}")
        normalized_weekdays = _normalize_weekdays(self.weekdays)
        timezone_name = _canonical_timezone_name(self.timezone_name, fallback="UTC")
        misfire_grace_seconds = _parse_optional_non_negative_float(
            self.misfire_grace_seconds,
            field_name="misfire_grace_seconds",
        )
        coalesce_policy = _normalize_coalesce_policy(self.coalesce_policy)
        object.__setattr__(self, "kind", "time")
        object.__setattr__(self, "schedule", clean_schedule)
        object.__setattr__(self, "weekdays", normalized_weekdays)
        object.__setattr__(self, "timezone_name", timezone_name)
        object.__setattr__(self, "misfire_grace_seconds", misfire_grace_seconds)
        object.__setattr__(self, "coalesce_policy", coalesce_policy)
        if clean_schedule == "once":
            if not str(self.due_at or "").strip():
                raise ValueError("due_at is required for once schedules")
            parsed_due_at = _parse_due_at_absolute(self.due_at, timezone_name=timezone_name)
            object.__setattr__(self, "due_at", parsed_due_at.isoformat())
            object.__setattr__(self, "time_of_day", None)
            object.__setattr__(self, "weekdays", ())
            return
        parsed = _parse_time_of_day(str(self.time_of_day or ""))
        object.__setattr__(self, "time_of_day", parsed.strftime("%H:%M"))
        object.__setattr__(self, "due_at", None)
        if clean_schedule == "daily":
            object.__setattr__(self, "weekdays", ())
            return
        if not normalized_weekdays:
            raise ValueError("weekdays are required for weekly schedules")

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "schedule": self.schedule,
            "due_at": self.due_at,
            "time_of_day": self.time_of_day,
            "weekdays": list(self.weekdays),
            "timezone_name": self.timezone_name,
            "misfire_grace_seconds": self.misfire_grace_seconds,
            "coalesce_policy": self.coalesce_policy,
        }


@dataclass(frozen=True, slots=True)
class IfThenAutomationTrigger:
    kind: Literal["if_then"] = "if_then"
    event_name: str | None = None
    all_conditions: tuple[AutomationCondition, ...] = ()
    any_conditions: tuple[AutomationCondition, ...] = ()
    cooldown_seconds: float = 0.0

    def __post_init__(self) -> None:
        clean_kind = str(self.kind).strip().lower()
        if clean_kind != "if_then":
            raise ValueError("IfThenAutomationTrigger.kind must be 'if_then'")
        all_conditions = tuple(self.all_conditions or ())
        any_conditions = tuple(self.any_conditions or ())
        if not all(isinstance(condition, AutomationCondition) for condition in all_conditions):
            raise ValueError("all_conditions must contain AutomationCondition values")
        if not all(isinstance(condition, AutomationCondition) for condition in any_conditions):
            raise ValueError("any_conditions must contain AutomationCondition values")
        event_name = _normalize_event_name(self.event_name)
        cooldown_seconds = _parse_non_negative_float(self.cooldown_seconds, field_name="cooldown_seconds")
        if event_name is None and not all_conditions and not any_conditions:
            raise ValueError("if_then trigger requires an event_name or at least one condition")
        object.__setattr__(self, "kind", "if_then")
        object.__setattr__(self, "event_name", event_name)
        object.__setattr__(self, "all_conditions", all_conditions)
        object.__setattr__(self, "any_conditions", any_conditions)
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
        try:
            raw_payload = dict(self.payload or {})
        except (TypeError, ValueError) as exc:
            raise ValueError("payload must be a mapping") from exc
        object.__setattr__(self, "payload", _json_safe_value(raw_payload))
        object.__setattr__(self, "enabled", _coerce_bool(self.enabled, default=True))
        if clean_kind in {"say", "print", "llm_prompt"} and not self.text:
            raise ValueError(f"text is required for action kind: {self.kind}")
        if clean_kind == "tool_call" and not self.tool_name:
            raise ValueError("tool_name is required for action kind: tool_call")

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "text": self.text,
            "tool_name": self.tool_name,
            "payload": _json_safe_value(self.payload),
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
    last_scheduled_at: datetime | None = None
    source: str = "manual"
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        clean_automation_id = _normalize_text(self.automation_id, limit=64)
        clean_name = _normalize_text(self.name, limit=120)
        actions = tuple(self.actions or ())
        if not clean_automation_id:
            raise ValueError("automation_id must not be empty")
        if not clean_name:
            raise ValueError("automation name must not be empty")
        if not isinstance(self.trigger, (TimeAutomationTrigger, IfThenAutomationTrigger)):
            raise ValueError("trigger must be a supported automation trigger")
        if not actions:
            raise ValueError("automation must include at least one action")
        if not all(isinstance(action, AutomationAction) for action in actions):
            raise ValueError("actions must contain AutomationAction values")
        clean_source = _slugify(self.source, fallback="manual")
        clean_tags = tuple(_slugify(tag, fallback="tag") for tag in self.tags if _normalize_text(tag, limit=40))
        object.__setattr__(self, "automation_id", clean_automation_id)
        object.__setattr__(self, "name", clean_name)
        object.__setattr__(self, "trigger", self.trigger)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "description", _normalize_text(self.description, limit=220) or None)
        object.__setattr__(self, "enabled", _coerce_bool(self.enabled, default=True))
        object.__setattr__(self, "created_at", _to_utc_datetime(self.created_at, timezone_name="UTC"))
        object.__setattr__(self, "updated_at", _to_utc_datetime(self.updated_at, timezone_name="UTC"))
        if self.last_triggered_at is not None:
            object.__setattr__(self, "last_triggered_at", _to_utc_datetime(self.last_triggered_at, timezone_name="UTC"))
        if self.last_scheduled_at is not None:
            object.__setattr__(self, "last_scheduled_at", _to_utc_datetime(self.last_scheduled_at, timezone_name="UTC"))
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
            "last_scheduled_at": self.last_scheduled_at.isoformat() if self.last_scheduled_at else None,
            "source": self.source,
            "tags": list(self.tags),
        }


@dataclass(frozen=True, slots=True)
class TimeAutomationMatch:
    entry: AutomationDefinition
    scheduled_for_at: datetime
    pending_run_count: int = 1


class AutomationEngine:
    """Evaluate automation triggers without owning persistence."""

    def due_time_matches(
        self,
        entries: tuple[AutomationDefinition, ...] | list[AutomationDefinition],
        *,
        now: datetime | None = None,
    ) -> tuple[TimeAutomationMatch, ...]:
        current_time = _ensure_aware_datetime(now or now_in_timezone("UTC"), timezone_name="UTC")
        matches: list[TimeAutomationMatch] = []
        for entry in entries:
            if not entry.enabled or not isinstance(entry.trigger, TimeAutomationTrigger):
                continue
            matches.extend(self._time_trigger_matches(entry, now=current_time))
        matches.sort(key=lambda match: (match.scheduled_for_at, match.entry.name.lower(), match.entry.automation_id))
        return tuple(matches)

    def due_time_automations(
        self,
        entries: tuple[AutomationDefinition, ...] | list[AutomationDefinition],
        *,
        now: datetime | None = None,
    ) -> tuple[AutomationDefinition, ...]:
        matches = self.due_time_matches(entries, now=now)
        seen: set[str] = set()
        due: list[AutomationDefinition] = []
        for match in matches:
            if match.entry.automation_id in seen:
                continue
            seen.add(match.entry.automation_id)
            due.append(match.entry)
        return tuple(due)

    def matching_if_then_automations(
        self,
        entries: tuple[AutomationDefinition, ...] | list[AutomationDefinition],
        *,
        facts: dict[str, Any],
        event_name: str | None = None,
        now: datetime | None = None,
    ) -> tuple[AutomationDefinition, ...]:
        current_time = _ensure_aware_datetime(now or now_in_timezone("UTC"), timezone_name="UTC")
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
        current_time = _ensure_aware_datetime(now or now_in_timezone("UTC"), timezone_name="UTC")
        if isinstance(entry.trigger, TimeAutomationTrigger):
            matches = self._time_trigger_matches(entry, now=current_time)
            if matches:
                return matches[0].scheduled_for_at
            return self._next_future_time_run(entry, now=current_time)
        return None

    def tool_record(
        self,
        entry: AutomationDefinition,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        current_time = _ensure_aware_datetime(now or now_in_timezone("UTC"), timezone_name="UTC")
        next_run = self.next_run_at(entry, now=current_time)
        pending_times: tuple[datetime, ...] = ()
        time_matches: tuple[TimeAutomationMatch, ...] = ()
        due_now = False
        if isinstance(entry.trigger, TimeAutomationTrigger):
            pending_times = self._pending_time_occurrences(entry, now=current_time)
            time_matches = self._time_trigger_matches(entry, now=current_time)
            due_now = bool(pending_times)
        record = {
            **entry.to_payload(),
            "trigger_kind": entry.trigger.kind,
            "due_now": due_now,
            "next_run_at": next_run.isoformat() if next_run is not None else None,
        }
        if isinstance(entry.trigger, TimeAutomationTrigger):
            record.update(
                {
                    "pending_run_count": len(pending_times),
                    "scheduled_for_at": time_matches[0].scheduled_for_at.isoformat() if time_matches else None,
                    "misfire_grace_seconds": entry.trigger.misfire_grace_seconds,
                    "coalesce_policy": entry.trigger.coalesce_policy,
                }
            )
        if isinstance(entry.trigger, IfThenAutomationTrigger):
            sensor_trigger = _safe_describe_sensor_trigger(entry.trigger)
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
                        "sensor_trigger_kind": getattr(sensor_trigger, "trigger_kind", None),
                        "sensor_hold_seconds": getattr(sensor_trigger, "hold_seconds", None),
                        "sensor_label": getattr(sensor_trigger, "label", None),
                    }
                )
        return record

    def _effective_schedule_anchor(self, entry: AutomationDefinition, *, zone_name: str) -> datetime | None:
        source = entry.last_scheduled_at or entry.last_triggered_at
        if source is None:
            return None
        return _ensure_aware_datetime(source, timezone_name=zone_name)

    def _occurrence_within_misfire_grace(self, trigger: TimeAutomationTrigger, occurrence_at: datetime, *, now: datetime) -> bool:
        grace = trigger.misfire_grace_seconds
        if grace is None:
            return True
        return occurrence_at + timedelta(seconds=grace) >= now

    def _pending_time_occurrences(self, entry: AutomationDefinition, *, now: datetime) -> tuple[datetime, ...]:
        trigger = entry.trigger
        if not isinstance(trigger, TimeAutomationTrigger):
            return ()
        zone = resolve_timezone(trigger.timezone_name)
        local_now = _ensure_aware_datetime(now, timezone_name=trigger.timezone_name).astimezone(zone)
        acknowledged_at = self._effective_schedule_anchor(entry, zone_name=trigger.timezone_name)
        created_at = entry.created_at.astimezone(zone)
        if trigger.schedule == "once":
            if acknowledged_at is not None:
                return ()
            due_at = _parse_due_at_absolute(trigger.due_at, timezone_name=trigger.timezone_name)
            if due_at > local_now:
                return ()
            if not self._occurrence_within_misfire_grace(trigger, due_at, now=local_now):
                return ()
            return (due_at,)

        run_time = _parse_time_of_day(str(trigger.time_of_day or "00:00"))
        anchor = acknowledged_at.astimezone(zone) if acknowledged_at is not None else None
        start_date = anchor.date() if anchor is not None else created_at.date()
        end_date = local_now.date()
        pending: list[datetime] = []
        days = max((end_date - start_date).days, 0)
        if days > _MAX_PENDING_OCCURRENCES:
            start_date = end_date - timedelta(days=_MAX_PENDING_OCCURRENCES)
        current_date = start_date
        while current_date <= end_date:
            if trigger.schedule == "weekly" and current_date.weekday() not in trigger.weekdays:
                current_date += timedelta(days=1)
                continue
            scheduled_at = _resolve_local_schedule_datetime(current_date, run_time, zone=zone)
            if anchor is not None and scheduled_at <= anchor:
                current_date += timedelta(days=1)
                continue
            if scheduled_at > local_now:
                break
            if self._occurrence_within_misfire_grace(trigger, scheduled_at, now=local_now):
                pending.append(scheduled_at)
            current_date += timedelta(days=1)
        return tuple(pending)

    def _time_trigger_matches(self, entry: AutomationDefinition, *, now: datetime) -> tuple[TimeAutomationMatch, ...]:
        trigger = entry.trigger
        if not isinstance(trigger, TimeAutomationTrigger):
            return ()
        pending = self._pending_time_occurrences(entry, now=now)
        if not pending:
            return ()
        if trigger.coalesce_policy == "all":
            return tuple(TimeAutomationMatch(entry=entry, scheduled_for_at=scheduled_for, pending_run_count=1) for scheduled_for in pending)
        if trigger.coalesce_policy == "earliest":
            return (TimeAutomationMatch(entry=entry, scheduled_for_at=pending[0], pending_run_count=len(pending)),)
        return (TimeAutomationMatch(entry=entry, scheduled_for_at=pending[-1], pending_run_count=len(pending)),)

    def _next_future_time_run(self, entry: AutomationDefinition, *, now: datetime) -> datetime | None:
        trigger = entry.trigger
        if not isinstance(trigger, TimeAutomationTrigger):
            return None
        zone = resolve_timezone(trigger.timezone_name)
        local_now = _ensure_aware_datetime(now, timezone_name=trigger.timezone_name).astimezone(zone)
        acknowledged_at = self._effective_schedule_anchor(entry, zone_name=trigger.timezone_name)
        anchor = acknowledged_at.astimezone(zone) if acknowledged_at is not None else entry.created_at.astimezone(zone)
        if trigger.schedule == "once":
            due_at = _parse_due_at_absolute(trigger.due_at, timezone_name=trigger.timezone_name)
            if acknowledged_at is not None:
                return None
            if due_at <= local_now and not self._occurrence_within_misfire_grace(trigger, due_at, now=local_now):
                return None
            return due_at

        run_time = _parse_time_of_day(str(trigger.time_of_day or "00:00"))
        for offset in range(0, 370):
            candidate_date = local_now.date() + timedelta(days=offset)
            if trigger.schedule == "weekly" and candidate_date.weekday() not in trigger.weekdays:
                continue
            candidate_at = _resolve_local_schedule_datetime(candidate_date, run_time, zone=zone)
            if candidate_at <= local_now:
                continue
            if candidate_at <= anchor:
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
        normalized_event_name = _normalize_event_name(event_name)
        if trigger.event_name and trigger.event_name != normalized_event_name:
            return False
        if entry.last_triggered_at is not None and trigger.cooldown_seconds > 0:
            last_triggered_at = _ensure_aware_datetime(entry.last_triggered_at, timezone_name="UTC")
            cooldown_until = last_triggered_at + timedelta(seconds=trigger.cooldown_seconds)
            if cooldown_until > _ensure_aware_datetime(now, timezone_name="UTC"):
                return False
        if trigger.all_conditions and not all(condition.matches(facts) for condition in trigger.all_conditions):
            return False
        if trigger.any_conditions and not any(condition.matches(facts) for condition in trigger.any_conditions):
            return False
        return True


class AutomationStore:
    """Persist and query Twinr automations in a bounded JSON file."""

    _LOCKS: ClassVar[dict[str, RLock]] = {}
    _LOCKS_GUARD: ClassVar[RLock] = RLock()

    def __init__(
        self,
        path: str | Path,
        *,
        timezone_name: str | None = None,
        max_entries: int = 96,
        engine: AutomationEngine | None = None,
        max_store_bytes: int = _DEFAULT_MAX_STORE_BYTES,
    ) -> None:
        self.path = _normalize_store_path(path)
        self.backup_path = self.path.with_name(f"{self.path.name}.bak")
        self.lock_path = self.path.with_name(f"{self.path.name}.lock")
        self.timezone_name = _canonical_timezone_name(timezone_name, fallback="Europe/Berlin")
        self.max_entries = max(8, _coerce_int(max_entries, default=96))
        self.max_store_bytes = max(64 * 1024, _coerce_int(max_store_bytes, default=_DEFAULT_MAX_STORE_BYTES))
        self.engine = engine or AutomationEngine()
        self._lock = self._lock_for_path(self.path)
        self._cached_entries: tuple[AutomationDefinition, ...] = ()
        self._cached_signature: tuple[tuple[bool, int, int], tuple[bool, int, int]] | None = None

    @classmethod
    def _lock_for_path(cls, path: Path) -> RLock:
        key = str(path)
        with cls._LOCKS_GUARD:
            lock = cls._LOCKS.get(key)
            if lock is None:
                lock = RLock()
                cls._LOCKS[key] = lock
            return lock

    @contextmanager
    def _guarded_store_access(self) -> Iterator[None]:
        with self._lock:
            with self._interprocess_lock():
                yield

    @contextmanager
    def _interprocess_lock(self) -> Iterator[None]:
        self._assert_safe_store_path(self.lock_path)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(str(self.lock_path), os.O_CREAT | os.O_RDWR, _CROSS_SERVICE_LOCK_FILE_MODE)
        try:
            try:
                # The Pi runtime and operator tools can touch the same store under different users.
                os.fchmod(fd, _CROSS_SERVICE_LOCK_FILE_MODE)
            except OSError:
                pass
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def load_entries(self) -> tuple[AutomationDefinition, ...]:
        with self._guarded_store_access():
            entries = self._load_entries_unlocked()
            self._cached_entries = entries
            return entries

    def list_tool_records(self, *, now: datetime | None = None) -> tuple[dict[str, Any], ...]:
        return tuple(self.engine.tool_record(entry, now=now) for entry in self.load_entries())

    def render_context(self, *, limit: int = 8, now: datetime | None = None) -> str | None:
        current_time = _ensure_aware_datetime(now or now_in_timezone(self.timezone_name), timezone_name=self.timezone_name)
        enabled_entries = [entry for entry in self.load_entries() if entry.enabled]
        if not enabled_entries:
            return None
        safe_limit = max(0, _coerce_int(limit, default=8))
        lines = ["Active automations:"]
        for entry in enabled_entries[:safe_limit]:
            schedule_text = self._describe_trigger(entry, now=current_time)
            action_text = self._describe_actions(entry.actions)
            lines.append(f"- {entry.name}: {schedule_text}. Action: {action_text}.")
        return "\n".join(lines).strip()

    def get(self, automation_id: str) -> AutomationDefinition | None:
        lookup = _normalize_text(automation_id, limit=64)
        with self._guarded_store_access():
            entries = self._load_entries_unlocked()
            return self._find_entry_unlocked(entries, lookup)

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
        misfire_grace_seconds: float | None = None,
        coalesce_policy: CoalescePolicy = "latest",
        source: str = "manual",
        tags: tuple[str, ...] | list[str] = (),
    ) -> AutomationDefinition:
        now = _ensure_aware_datetime(now_in_timezone(self.timezone_name), timezone_name=self.timezone_name)
        trigger = TimeAutomationTrigger(
            schedule=schedule,
            due_at=due_at,
            time_of_day=time_of_day,
            weekdays=tuple(weekdays),
            timezone_name=timezone_name or self.timezone_name,
            misfire_grace_seconds=misfire_grace_seconds,
            coalesce_policy=coalesce_policy,
        )
        entry = AutomationDefinition(
            automation_id=f"AUTO-{now.astimezone(_UTC).strftime('%Y%m%dT%H%M%S%fZ')}-{uuid4().hex[:8]}",
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
        now = _ensure_aware_datetime(now_in_timezone(self.timezone_name), timezone_name=self.timezone_name)
        trigger = IfThenAutomationTrigger(
            event_name=event_name,
            all_conditions=tuple(all_conditions),
            any_conditions=tuple(any_conditions),
            cooldown_seconds=cooldown_seconds,
        )
        entry = AutomationDefinition(
            automation_id=f"AUTO-{now.astimezone(_UTC).strftime('%Y%m%dT%H%M%S%fZ')}-{uuid4().hex[:8]}",
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
        with self._guarded_store_access():
            entries = list(self._load_entries_unlocked())
            normalized = replace(
                entry,
                updated_at=_to_utc_datetime(now_in_timezone(self.timezone_name), timezone_name=self.timezone_name),
            )
            for index, existing in enumerate(entries):
                if existing.automation_id != normalized.automation_id:
                    continue
                entries[index] = normalized
                self._write_entries_unlocked(tuple(entries))
                return normalized
            if len(entries) >= self.max_entries:
                raise ValueError(f"Automation store is full ({self.max_entries} entries)")
            entries.append(normalized)
            self._write_entries_unlocked(tuple(entries))
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
        lookup = _normalize_text(automation_id, limit=64)
        with self._guarded_store_access():
            entries = list(self._load_entries_unlocked())
            existing = self._find_entry_unlocked(tuple(entries), lookup)
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
                updated_at=_to_utc_datetime(now_in_timezone(self.timezone_name), timezone_name=self.timezone_name),
            )
            for index, current in enumerate(entries):
                if current.automation_id == lookup:
                    entries[index] = updated
                    self._write_entries_unlocked(tuple(entries))
                    return updated
        raise KeyError(f"Unknown automation_id: {automation_id}")

    def delete(self, automation_id: str) -> AutomationDefinition:
        lookup = _normalize_text(automation_id, limit=64)
        with self._guarded_store_access():
            entries = list(self._load_entries_unlocked())
            for index, entry in enumerate(entries):
                if entry.automation_id != lookup:
                    continue
                removed = entries.pop(index)
                self._write_entries_unlocked(tuple(entries))
                return removed
        raise KeyError(f"Unknown automation_id: {automation_id}")

    def mark_triggered(
        self,
        automation_id: str,
        *,
        triggered_at: datetime | None = None,
        scheduled_for_at: datetime | None = None,
    ) -> AutomationDefinition:
        lookup = _normalize_text(automation_id, limit=64)
        with self._guarded_store_access():
            entries = list(self._load_entries_unlocked())
            existing = self._find_entry_unlocked(tuple(entries), lookup)
            if existing is None:
                raise KeyError(f"Unknown automation_id: {automation_id}")
            effective_triggered_at = _to_utc_datetime(
                triggered_at or now_in_timezone(self.timezone_name),
                timezone_name=self.timezone_name,
            )
            effective_scheduled_for_at = _to_utc_datetime(
                scheduled_for_at or effective_triggered_at,
                timezone_name=self.timezone_name,
            )
            updated = replace(
                existing,
                last_triggered_at=effective_triggered_at,
                last_scheduled_at=effective_scheduled_for_at,
                updated_at=effective_triggered_at,
            )
            for index, current in enumerate(entries):
                if current.automation_id == lookup:
                    entries[index] = updated
                    self._write_entries_unlocked(tuple(entries))
                    return updated
        raise KeyError(f"Unknown automation_id: {automation_id}")

    def due_time_matches(self, *, now: datetime | None = None) -> tuple[TimeAutomationMatch, ...]:
        return self.engine.due_time_matches(self.load_entries(), now=now)

    def due_time_automations(self, *, now: datetime | None = None) -> tuple[AutomationDefinition, ...]:
        return self.engine.due_time_automations(self.load_entries(), now=now)

    def matching_if_then_automations(
        self,
        *,
        facts: dict[str, Any],
        event_name: str | None = None,
        now: datetime | None = None,
    ) -> tuple[AutomationDefinition, ...]:
        return self.engine.matching_if_then_automations(self.load_entries(), facts=facts, event_name=event_name, now=now)

    def _sorted_entries(self, entries: tuple[AutomationDefinition, ...] | list[AutomationDefinition]) -> tuple[AutomationDefinition, ...]:
        return tuple(sorted(entries, key=lambda entry: (entry.name.lower(), entry.created_at, entry.automation_id)))

    def _find_entry_unlocked(
        self,
        entries: tuple[AutomationDefinition, ...] | list[AutomationDefinition],
        automation_id: str,
    ) -> AutomationDefinition | None:
        for entry in entries:
            if entry.automation_id == automation_id:
                return entry
        return None

    def _assert_safe_store_path(self, path: Path) -> None:
        normalized_path = _normalize_store_path(path)
        if normalized_path.name in {"", ".", ".."}:
            raise ValueError("Automation store path must point to a file")
        if normalized_path.exists() and normalized_path.is_dir():
            raise ValueError("Automation store path must be a file, not a directory")
        for candidate in (normalized_path, *normalized_path.parents):
            if candidate.exists() and candidate.is_symlink():
                raise ValueError(f"Refusing symlinked automation store path: {normalized_path}")

    def _fsync_directory(self, directory: Path) -> None:
        try:
            dir_fd = os.open(str(directory), os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    def _decode_json_payload(self, raw: bytes) -> dict[str, Any] | None:
        try:
            if msgspec is not None:
                payload = msgspec.json.decode(raw)
            else:
                payload = json.loads(raw.decode("utf-8"))
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _read_payload_file(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            self._assert_safe_store_path(path)
            stat = path.stat()
            if stat.st_size > self.max_store_bytes:
                return None
            raw = path.read_bytes()
        except OSError:
            return None
        return self._decode_json_payload(raw)

    def _store_signature(self) -> tuple[tuple[bool, int, int], tuple[bool, int, int]]:
        return (_path_signature(self.path), _path_signature(self.backup_path))

    def _load_entries_unlocked(self) -> tuple[AutomationDefinition, ...]:
        signature = self._store_signature()
        if self._cached_signature == signature:
            return self._cached_entries
        payload: dict[str, Any] | None = None
        if self.path.exists():
            payload = self._read_payload_file(self.path)
        if payload is None:
            payload = self._read_payload_file(self.backup_path)
        if payload is None:
            self._cached_signature = signature
            return self._cached_entries
        items = payload.get("entries", [])
        if not isinstance(items, list):
            self._cached_signature = signature
            return self._cached_entries
        entries: list[AutomationDefinition] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            entry = self._entry_from_payload(item)
            if entry is not None:
                entries.append(entry)
        normalized = self._sorted_entries(entries)
        self._cached_entries = normalized
        self._cached_signature = signature
        return normalized

    def _atomic_write_json_file(self, path: Path, payload: dict[str, Any]) -> None:
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        encoded = serialized.encode("utf-8")
        if len(encoded) > self.max_store_bytes:
            raise ValueError(f"Automation payload exceeds {self.max_store_bytes} bytes")
        parent = path.parent
        self._assert_safe_store_path(path)
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(parent))
        temporary_path = Path(temporary_name)
        try:
            try:
                os.fchmod(fd, _CROSS_SERVICE_STORE_FILE_MODE)
            except OSError:
                pass
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
            self._fsync_directory(parent)
        except Exception:
            try:
                if temporary_path.exists():
                    temporary_path.unlink()
            except OSError:
                pass
            raise

    def _write_entries_unlocked(self, entries: tuple[AutomationDefinition, ...]) -> None:
        normalized_entries = self._sorted_entries(entries)
        payload = {
            "schema_version": _PAYLOAD_SCHEMA_VERSION,
            "updated_at": _to_utc_datetime(now_in_timezone(self.timezone_name), timezone_name=self.timezone_name).isoformat(),
            "entries": [entry.to_payload() for entry in normalized_entries],
        }
        self._atomic_write_json_file(self.path, payload)
        try:
            self._atomic_write_json_file(self.backup_path, payload)
        except (OSError, ValueError):
            pass
        self._cached_entries = normalized_entries
        self._cached_signature = self._store_signature()

    def _entry_from_payload(self, payload: dict[str, Any]) -> AutomationDefinition | None:
        automation_id = _normalize_text(payload.get("automation_id"), limit=64)
        name = _normalize_text(payload.get("name"), limit=120)
        trigger_payload = payload.get("trigger")
        actions_payload = payload.get("actions")
        tags_payload = payload.get("tags", ())
        if not automation_id or not name or not isinstance(trigger_payload, dict) or not isinstance(actions_payload, list):
            return None
        trigger = self._trigger_from_payload(trigger_payload)
        if trigger is None:
            return None
        actions = tuple(action for action in (self._action_from_payload(item) for item in actions_payload) if action is not None)
        if not actions:
            return None
        if not isinstance(tags_payload, (list, tuple)):
            tags_payload = ()
        now_utc = _to_utc_datetime(now_in_timezone(self.timezone_name), timezone_name=self.timezone_name)
        return AutomationDefinition(
            automation_id=automation_id,
            name=name,
            description=_normalize_text(payload.get("description"), limit=220) or None,
            enabled=_coerce_bool(payload.get("enabled", True), default=True),
            trigger=trigger,
            actions=actions,
            created_at=_parse_iso_timestamp(payload.get("created_at"), timezone_name=self.timezone_name) or now_utc,
            updated_at=_parse_iso_timestamp(payload.get("updated_at"), timezone_name=self.timezone_name) or now_utc,
            last_triggered_at=_parse_iso_timestamp(payload.get("last_triggered_at"), timezone_name=self.timezone_name),
            last_scheduled_at=_parse_iso_timestamp(payload.get("last_scheduled_at"), timezone_name=self.timezone_name),
            source=_slugify(payload.get("source"), fallback="manual"),
            tags=tuple(str(item).strip() for item in tags_payload if str(item).strip()),
        )

    def _trigger_from_payload(self, payload: dict[str, Any]) -> AutomationTrigger | None:
        kind = str(payload.get("kind", "")).strip().lower()
        if kind == "time":
            try:
                weekdays_payload = payload.get("weekdays", [])
                if not isinstance(weekdays_payload, (list, tuple)):
                    return None
                return TimeAutomationTrigger(
                    schedule=str(payload.get("schedule", "once")),
                    due_at=payload.get("due_at"),
                    time_of_day=payload.get("time_of_day"),
                    weekdays=tuple(int(item) for item in weekdays_payload if str(item).strip()),
                    timezone_name=payload.get("timezone_name") or self.timezone_name,
                    misfire_grace_seconds=payload.get("misfire_grace_seconds"),
                    coalesce_policy=payload.get("coalesce_policy", "latest"),
                )
            except (TypeError, ValueError):
                return None
        if kind == "if_then":
            all_conditions_payload = payload.get("all_conditions", [])
            any_conditions_payload = payload.get("any_conditions", [])
            if not isinstance(all_conditions_payload, (list, tuple)) or not isinstance(any_conditions_payload, (list, tuple)):
                return None
            all_conditions = tuple(condition for condition in (self._condition_from_payload(item) for item in all_conditions_payload) if condition is not None)
            any_conditions = tuple(condition for condition in (self._condition_from_payload(item) for item in any_conditions_payload) if condition is not None)
            try:
                return IfThenAutomationTrigger(
                    event_name=payload.get("event_name"),
                    all_conditions=all_conditions,
                    any_conditions=any_conditions,
                    cooldown_seconds=payload.get("cooldown_seconds", 0.0) or 0.0,
                )
            except ValueError:
                return None
        return None

    def _condition_from_payload(self, payload: Any) -> AutomationCondition | None:
        if not isinstance(payload, dict):
            return None
        try:
            return AutomationCondition(key=str(payload.get("key", "")), operator=str(payload.get("operator", "eq")), value=payload.get("value"))
        except (TypeError, ValueError):
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
                enabled=_coerce_bool(payload.get("enabled", True), default=True),
            )
        except (TypeError, ValueError):
            return None

    def _describe_trigger(self, entry: AutomationDefinition, *, now: datetime) -> str:
        trigger = entry.trigger
        sensor_text = _safe_describe_sensor_trigger_text(trigger)
        if sensor_text:
            return sensor_text
        if not isinstance(trigger, TimeAutomationTrigger):
            return f"{trigger.kind} trigger"
        if trigger.schedule == "once":
            due_at = _parse_due_at_absolute(trigger.due_at, timezone_name=trigger.timezone_name)
            return f"once at {due_at.isoformat()}"
        if trigger.schedule == "daily":
            next_run = self.engine.next_run_at(entry, now=now)
            if next_run is not None:
                return f"every day at {trigger.time_of_day}; next {next_run.isoformat()}"
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
                search = _coerce_bool(action.payload.get("allow_web_search"), default=False)
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
