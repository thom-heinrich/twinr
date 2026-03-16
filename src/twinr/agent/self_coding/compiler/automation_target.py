"""Compile raw Codex automation drafts into activation-ready Twinr manifests.

This module keeps the `automation_manifest` target deterministic. It accepts a
bounded JSON object from Codex, validates it against Twinr's automation model,
and emits a canonical manifest that later activation code can apply without
guessing or reparsing free-form text.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from math import isfinite
from typing import Any, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from twinr.agent.self_coding.contracts import CompileJobRecord, RequirementsDialogueSession
from twinr.automations.store import (
    AutomationAction,
    AutomationCondition,
    AutomationDefinition,
    IfThenAutomationTrigger,
    TimeAutomationTrigger,
)
from twinr.text_utils import extract_json_object, slugify_identifier

_AUTOMATION_MANIFEST_SCHEMA = "twinr_self_coding_automation_manifest_v1"
_MAX_RAW_CONTENT_CHARS = 64_000
_MAX_JSON_DEPTH = 12
_MAX_JSON_COLLECTION_ITEMS = 256
_MAX_JSON_STRING_LENGTH = 8_192
_MAX_ACTIONS = 16
_MAX_CONDITIONS = 32
_TRIGGER_KIND_ALIASES = frozenset({"time", "if_then"})
_TIME_TRIGGER_KEYS = frozenset({"schedule", "due_at", "time_of_day", "weekdays", "timezone_name"})
_EVENT_TRIGGER_KEYS = frozenset({"event_name", "all_conditions", "any_conditions", "cooldown_seconds"})
_TRUE_STRINGS = frozenset({"1", "true", "yes", "on"})
_FALSE_STRINGS = frozenset({"0", "false", "no", "off"})
_WEEKDAY_NAME_TO_INDEX = {
    "monday": 0,
    "mon": 0,
    "tuesday": 1,
    "tue": 1,
    "wednesday": 2,
    "wed": 2,
    "thursday": 3,
    "thu": 3,
    "friday": 4,
    "fri": 4,
    "saturday": 5,
    "sat": 5,
    "sunday": 6,
    "sun": 6,
}


class AutomationManifestCompilerError(ValueError):
    """Raise when a raw Codex automation draft cannot become a safe manifest."""


@dataclass(frozen=True, slots=True)
class CompiledAutomationManifest:
    """Carry one canonical automation manifest plus metadata for persistence."""

    automation: AutomationDefinition
    content: str
    summary: str
    metadata: dict[str, Any]


def compile_automation_manifest_content(
    *,
    job: CompileJobRecord,
    session: RequirementsDialogueSession,
    raw_content: str,
) -> CompiledAutomationManifest:
    """Validate and canonicalize one raw Codex automation draft.

    Args:
        job: Compile job metadata used to stamp the resulting manifest.
        session: Approved requirements session that scoped the compile request.
        raw_content: Raw JSON text returned inside the Codex artifact.

    Returns:
        A canonical manifest document and its normalized automation definition.

    Raises:
        AutomationManifestCompilerError: If the raw draft is not valid JSON or
            cannot be converted into a valid Twinr automation definition.
    """

    try:
        # AUDIT-FIX(#11): Enforce explicit input bounds before parsing to keep the RPi single-process compiler responsive.
        _validate_raw_content(raw_content)
        payload_object = extract_json_object(raw_content)
        # AUDIT-FIX(#4): Require a JSON object root instead of assuming extract_json_object() always returns a mapping.
        payload = _require_mapping(
            _canonicalize_json_value(payload_object, context="root automation manifest"),
            field_name="root automation manifest",
        )
        automation_payload = _extract_automation_payload(payload)
        compiled_at = datetime.now(UTC)
        automation = _build_automation(job=job, session=session, payload=automation_payload, now=compiled_at)
        # AUDIT-FIX(#5): Validate the compiled automation payload is JSON-safe before json.dumps() so failures surface as compiler errors.
        automation_document = _require_mapping(
            _canonicalize_json_value(automation.to_payload(), context="compiled automation payload"),
            field_name="compiled automation payload",
        )
        job_id = _require_non_empty_string(getattr(job, "job_id", None), field_name="job.job_id")
        skill_id = _require_non_empty_string(getattr(job, "skill_id", None), field_name="job.skill_id")
        skill_name = _normalize_optional_string(getattr(job, "skill_name", None), field_name="job.skill_name")
        document = {
            "schema": _AUTOMATION_MANIFEST_SCHEMA,
            "target": "automation_manifest",
            "job_id": job_id,
            "skill_id": skill_id,
            "skill_name": skill_name,
            "compiled_at": compiled_at.isoformat().replace("+00:00", "Z"),
            "activation_policy": {
                "requires_confirmation": True,
                "initial_enabled": False,
            },
            "automation": automation_document,
        }
        action_kinds = tuple(action.kind for action in automation.actions)
        metadata = {
            "automation_id": automation.automation_id,
            "manifest_schema": _AUTOMATION_MANIFEST_SCHEMA,
            "trigger_kind": automation.trigger.kind,
            "action_kinds": list(action_kinds),
            "activatable": True,
        }
        return CompiledAutomationManifest(
            automation=automation,
            # AUDIT-FIX(#5): Emit strict JSON only; allow_nan=False prevents non-standard NaN/Infinity tokens from leaking into manifests.
            content=json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
            summary=f"Validated automation manifest for {automation.name}.",
            metadata=metadata,
        )
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        # AUDIT-FIX(#4): Normalize all validation/serialization failures into the module's public compiler error contract.
        raise AutomationManifestCompilerError(f"Invalid automation manifest: {exc}") from exc


def _extract_automation_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    automation_payload = payload.get("automation")
    if automation_payload is not None:
        # AUDIT-FIX(#4): Reject non-object `automation` payloads explicitly instead of falling through into confusing downstream errors.
        return _require_mapping(automation_payload, field_name="automation")

    has_actions = _payload_contains_actions(payload)
    has_trigger = "trigger" in payload or _has_time_trigger_fields(payload) or _has_event_trigger_fields(payload)
    # AUDIT-FIX(#8): Accept top-level singular `action` shorthand consistently with _resolve_actions_payload().
    if has_actions and has_trigger:
        return payload
    raise ValueError("expected an `automation` object with trigger and actions/action")


def _build_automation(
    *,
    job: CompileJobRecord,
    session: RequirementsDialogueSession,
    payload: Mapping[str, Any],
    now: datetime,
) -> AutomationDefinition:
    job_skill_id = _require_non_empty_string(getattr(job, "skill_id", None), field_name="job.skill_id")
    session_skill_name = _normalize_optional_string(getattr(session, "skill_name", None), field_name="session.skill_name")
    job_skill_name = _normalize_optional_string(getattr(job, "skill_name", None), field_name="job.skill_name")
    request_summary = _normalize_optional_string(
        getattr(session, "request_summary", None),
        field_name="session.request_summary",
    )
    # AUDIT-FIX(#9): Avoid stringifying None/non-strings into literal "None"; require real strings for user-visible automation names.
    name = _normalize_optional_string(payload.get("name"), field_name="automation.name") or session_skill_name or job_skill_name
    if name is None:
        raise ValueError("automation name must not be empty")

    actions_payload = _resolve_actions_payload(payload)
    if not actions_payload:
        raise ValueError("automation must include at least one action")

    trigger = _resolve_trigger(payload)
    actions = tuple(_build_action(item) for item in actions_payload)
    automation_id = f"ase_{slugify_identifier(job_skill_id, fallback='skill')}"
    return AutomationDefinition(
        automation_id=automation_id,
        name=name,
        description=_normalize_optional_string(payload.get("description"), field_name="automation.description") or request_summary,
        enabled=False,
        trigger=trigger,
        actions=actions,
        # AUDIT-FIX(#10): Reuse one UTC timestamp so manifest-level and model-level timestamps stay consistent.
        created_at=now,
        updated_at=now,
        source="self_coding",
        tags=("self_coding", job_skill_id),
    )


def _build_trigger(payload: Mapping[str, Any]) -> TimeAutomationTrigger | IfThenAutomationTrigger:
    raw_kind = _normalize_optional_string(payload.get("kind"), field_name="automation trigger kind")
    raw_type = _normalize_optional_string(payload.get("type"), field_name="automation trigger type")

    # AUDIT-FIX(#1): Treat `type` as a trigger-kind alias before falling back to legacy event-name shorthand.
    if raw_kind is not None:
        kind = raw_kind.casefold()
    elif raw_type is not None and raw_type.casefold() in _TRIGGER_KIND_ALIASES:
        kind = raw_type.casefold()
    elif raw_type is not None:
        legacy_payload = dict(payload)
        legacy_payload.setdefault("event_name", raw_type)
        return _build_if_then_trigger(legacy_payload)
    else:
        raise ValueError("automation trigger kind must be provided")

    if kind == "time":
        return _build_time_trigger(payload)
    if kind == "if_then":
        return _build_if_then_trigger(payload)
    raise ValueError(f"unsupported automation trigger kind: {kind}")


def _resolve_trigger(payload: Mapping[str, Any]) -> TimeAutomationTrigger | IfThenAutomationTrigger:
    trigger_payload = payload.get("trigger")
    if trigger_payload is not None:
        # AUDIT-FIX(#4): Reject non-object trigger payloads with a deterministic error instead of relying on attribute access.
        return _build_trigger(_require_mapping(trigger_payload, field_name="automation.trigger"))

    has_event_fields = _has_event_trigger_fields(payload)
    has_time_fields = _has_time_trigger_fields(payload)
    # AUDIT-FIX(#2): Reject ambiguous shorthand triggers instead of guessing based on branch order.
    if has_event_fields and has_time_fields:
        raise ValueError("automation trigger is ambiguous; choose either event fields or time fields")
    if has_event_fields:
        return _build_if_then_trigger(payload)
    if has_time_fields:
        return _build_time_trigger(payload)
    raise ValueError("automation trigger must be an object or a supported shorthand")


def _build_time_trigger(payload: Mapping[str, Any]) -> TimeAutomationTrigger:
    if _has_event_trigger_fields(payload):
        # AUDIT-FIX(#2): Block mixed trigger families inside explicit trigger objects to avoid silent miscompilation.
        raise ValueError("time trigger cannot include event trigger fields")

    schedule = _normalize_optional_string(payload.get("schedule"), field_name="time trigger schedule")
    due_at = _normalize_optional_string(payload.get("due_at"), field_name="time trigger due_at")
    time_of_day = _normalize_optional_string(payload.get("time_of_day"), field_name="time trigger time_of_day")
    timezone_name = _normalize_optional_string(payload.get("timezone_name"), field_name="time trigger timezone_name")
    weekdays = _parse_weekdays(payload.get("weekdays"))

    if schedule is None:
        schedule = "once"
    else:
        schedule = schedule.casefold()

    if due_at is None and time_of_day is None:
        # AUDIT-FIX(#6): Require enough temporal information to schedule a reminder deterministically.
        raise ValueError("time trigger must include due_at or time_of_day")
    if weekdays and time_of_day is None:
        raise ValueError("time trigger weekdays require time_of_day")
    if timezone_name is not None:
        _validate_timezone_name(timezone_name)
    if due_at is not None:
        _validate_due_at(due_at, timezone_name=timezone_name)
    if time_of_day is not None:
        time_of_day = _normalize_time_of_day(time_of_day)

    return TimeAutomationTrigger(
        schedule=schedule,
        due_at=due_at,
        time_of_day=time_of_day,
        weekdays=weekdays,
        timezone_name=timezone_name,
    )


def _build_if_then_trigger(payload: Mapping[str, Any]) -> IfThenAutomationTrigger:
    if _has_time_trigger_fields(payload):
        # AUDIT-FIX(#2): Block mixed trigger families inside explicit trigger objects to avoid silent miscompilation.
        raise ValueError("if_then trigger cannot include time trigger fields")

    all_conditions_payload = payload.get("all_conditions", ())
    any_conditions_payload = payload.get("any_conditions", ())
    if all_conditions_payload is None:
        all_conditions_payload = ()
    if any_conditions_payload is None:
        any_conditions_payload = ()
    if not isinstance(all_conditions_payload, (list, tuple)):
        raise ValueError("if_then all_conditions must be a list")
    if not isinstance(any_conditions_payload, (list, tuple)):
        raise ValueError("if_then any_conditions must be a list")
    if len(all_conditions_payload) > _MAX_CONDITIONS:
        raise ValueError(f"if_then all_conditions must contain at most {_MAX_CONDITIONS} items")
    if len(any_conditions_payload) > _MAX_CONDITIONS:
        raise ValueError(f"if_then any_conditions must contain at most {_MAX_CONDITIONS} items")

    event_name = _normalize_optional_string(payload.get("event_name"), field_name="if_then trigger event_name")
    if event_name is None:
        raw_type = _normalize_optional_string(payload.get("type"), field_name="if_then trigger type")
        if raw_type is not None and raw_type.casefold() not in _TRIGGER_KIND_ALIASES:
            event_name = raw_type

    all_conditions = tuple(_build_condition(item) for item in all_conditions_payload)
    any_conditions = tuple(_build_condition(item) for item in any_conditions_payload)
    if event_name is None and not all_conditions and not any_conditions:
        # AUDIT-FIX(#2): Reject empty event triggers; they force the executor to guess or fail later.
        raise ValueError("if_then trigger must include event_name or at least one condition")

    return IfThenAutomationTrigger(
        event_name=event_name,
        all_conditions=all_conditions,
        any_conditions=any_conditions,
        # AUDIT-FIX(#6): Parse cooldowns strictly so invalid strings/negative values cannot silently disable debounce logic.
        cooldown_seconds=_parse_non_negative_float(
            payload.get("cooldown_seconds"),
            field_name="if_then trigger cooldown_seconds",
            default=0.0,
        ),
    )


def _build_condition(payload: Any) -> AutomationCondition:
    if not isinstance(payload, Mapping):
        raise ValueError("automation condition must be an object")
    key = _require_non_empty_string(payload.get("key"), field_name="automation condition key")
    operator = _require_non_empty_string(payload.get("operator", "eq"), field_name="automation condition operator")
    return AutomationCondition(
        # AUDIT-FIX(#7): Require meaningful condition keys/operators now instead of deferring malformed conditions to runtime.
        key=key,
        operator=operator,
        value=_canonicalize_json_value(payload.get("value"), context=f"automation condition `{key}` value"),
    )


def _build_action(payload: Any) -> AutomationAction:
    if not isinstance(payload, Mapping):
        raise ValueError("automation action must be an object")
    action_kind = _require_non_empty_string(
        payload.get("kind") or payload.get("type"),
        field_name="automation action kind",
    )
    action_payload = payload.get("payload", {})
    if action_payload is None:
        action_payload = {}
    if not isinstance(action_payload, Mapping):
        raise ValueError("automation action payload must be an object")
    text = _normalize_optional_string(payload.get("text"), field_name=f"automation action `{action_kind}` text")
    if text is None:
        text = _normalize_optional_string(payload.get("message"), field_name=f"automation action `{action_kind}` message")
    return AutomationAction(
        # AUDIT-FIX(#7): Enforce a non-empty action kind and JSON-safe payload values so unsupported actions fail at compile time.
        kind=action_kind,
        text=text,
        tool_name=_normalize_optional_string(payload.get("tool_name"), field_name=f"automation action `{action_kind}` tool_name"),
        payload=_require_mapping(
            _canonicalize_json_value(action_payload, context=f"automation action `{action_kind}` payload"),
            field_name=f"automation action `{action_kind}` payload",
        ),
        # AUDIT-FIX(#3): Parse booleans strictly; bool("false") == True was enabling actions that should stay disabled.
        enabled=_parse_bool(payload.get("enabled", True), field_name=f"automation action `{action_kind}` enabled"),
    )


def _resolve_actions_payload(payload: Mapping[str, Any]) -> tuple[Any, ...]:
    actions_payload = payload.get("actions")
    if actions_payload is not None:
        if not isinstance(actions_payload, (list, tuple)):
            raise ValueError("automation actions must be a list")
        if len(actions_payload) > _MAX_ACTIONS:
            # AUDIT-FIX(#11): Bound action fan-out so one malformed draft cannot explode runtime complexity on the device.
            raise ValueError(f"automation actions must contain at most {_MAX_ACTIONS} items")
        if actions_payload:
            return tuple(actions_payload)
    singular_action = payload.get("action")
    if singular_action is not None:
        if not isinstance(singular_action, Mapping):
            raise ValueError("automation action must be an object")
        return (singular_action,)
    return ()


def _validate_raw_content(raw_content: str) -> None:
    if not isinstance(raw_content, str):
        raise TypeError("raw_content must be a string")
    if not raw_content.strip():
        raise ValueError("raw_content must not be empty")
    if len(raw_content) > _MAX_RAW_CONTENT_CHARS:
        raise ValueError(f"raw_content exceeds {_MAX_RAW_CONTENT_CHARS} characters")


def _payload_contains_actions(payload: Mapping[str, Any]) -> bool:
    actions_payload = payload.get("actions")
    if isinstance(actions_payload, (list, tuple)) and len(actions_payload) > 0:
        return True
    return isinstance(payload.get("action"), Mapping)


def _has_time_trigger_fields(payload: Mapping[str, Any]) -> bool:
    return any(_value_is_present(payload.get(key)) for key in _TIME_TRIGGER_KEYS if key in payload)


def _has_event_trigger_fields(payload: Mapping[str, Any]) -> bool:
    return any(_value_is_present(payload.get(key)) for key in _EVENT_TRIGGER_KEYS if key in payload)


def _value_is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return bool(value)
    return True


def _require_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


def _normalize_optional_string(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.strip()
    return normalized or None


def _require_non_empty_string(value: Any, *, field_name: str) -> str:
    normalized = _normalize_optional_string(value, field_name=field_name)
    if normalized is None:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _parse_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        if value in (0, 1):
            return bool(value)
        raise ValueError(f"{field_name} must be a boolean")
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
    raise ValueError(f"{field_name} must be a boolean")


def _parse_non_negative_float(value: Any, *, field_name: str, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative number")
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{field_name} must be a non-negative number")
        try:
            parsed = float(normalized)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be a non-negative number") from exc
    else:
        raise ValueError(f"{field_name} must be a non-negative number")
    if not isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{field_name} must be a non-negative finite number")
    return parsed


def _parse_weekdays(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("time trigger weekdays must be a list")

    numeric_days: list[int] = []
    normalized_days: list[int] = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError("time trigger weekdays must contain integers or weekday names")
        if isinstance(item, int):
            numeric_days.append(item)
            continue
        if isinstance(item, str):
            candidate = item.strip().casefold()
            if not candidate:
                raise ValueError("time trigger weekdays must not contain empty strings")
            if candidate in _WEEKDAY_NAME_TO_INDEX:
                normalized_days.append(_WEEKDAY_NAME_TO_INDEX[candidate])
                continue
            try:
                numeric_days.append(int(candidate))
            except ValueError as exc:
                raise ValueError("time trigger weekdays must contain integers or weekday names") from exc
            continue
        raise ValueError("time trigger weekdays must contain integers or weekday names")

    if numeric_days:
        # AUDIT-FIX(#6): Normalize common 1..7 weekday output into Python's 0..6 convention instead of passing ambiguous raw integers downstream.
        use_iso_weekdays = 0 not in numeric_days
        for day in numeric_days:
            if use_iso_weekdays:
                if day < 1 or day > 7:
                    raise ValueError("time trigger weekdays must use 1..7 or 0..6 values")
                normalized_days.append(day - 1)
            else:
                if day < 0 or day > 6:
                    raise ValueError("time trigger weekdays must use 0..6 values")
                normalized_days.append(day)

    return tuple(sorted(set(normalized_days)))


def _validate_timezone_name(timezone_name: str) -> None:
    try:
        ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"time trigger timezone_name must be a valid IANA timezone: {timezone_name}") from exc


def _validate_due_at(due_at: str, *, timezone_name: str | None) -> None:
    normalized = due_at.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("time trigger due_at must be an ISO-8601 datetime") from exc
    if parsed.tzinfo is None and timezone_name is None:
        # AUDIT-FIX(#6): Refuse timezone-naive absolute times; otherwise the executor has to guess around DST boundaries.
        raise ValueError("time trigger due_at must include timezone information or timezone_name")


def _normalize_time_of_day(time_of_day: str) -> str:
    for fmt, out_fmt in (("%H:%M:%S", "%H:%M:%S"), ("%H:%M", "%H:%M")):
        try:
            parsed = datetime.strptime(time_of_day, fmt)
        except ValueError:
            continue
        return parsed.strftime(out_fmt)
    raise ValueError("time trigger time_of_day must use HH:MM or HH:MM:SS")


def _canonicalize_json_value(value: Any, *, context: str, depth: int = 0) -> Any:
    if depth > _MAX_JSON_DEPTH:
        raise ValueError(f"{context} exceeds maximum nesting depth of {_MAX_JSON_DEPTH}")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{context} must not contain NaN or Infinity")
        return value
    if isinstance(value, str):
        if len(value) > _MAX_JSON_STRING_LENGTH:
            raise ValueError(f"{context} exceeds maximum string length of {_MAX_JSON_STRING_LENGTH}")
        return value
    if isinstance(value, Mapping):
        if len(value) > _MAX_JSON_COLLECTION_ITEMS:
            raise ValueError(f"{context} exceeds maximum object size of {_MAX_JSON_COLLECTION_ITEMS}")
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{context} keys must be strings")
            normalized[key] = _canonicalize_json_value(item, context=f"{context}.{key}", depth=depth + 1)
        return normalized
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_JSON_COLLECTION_ITEMS:
            raise ValueError(f"{context} exceeds maximum array size of {_MAX_JSON_COLLECTION_ITEMS}")
        return [_canonicalize_json_value(item, context=f"{context}[]", depth=depth + 1) for item in value]
    raise ValueError(f"{context} must contain only JSON-compatible values")