from __future__ import annotations

from typing import Any

from twinr.agent.tools.support import optional_bool, optional_float, require_sensitive_voice_confirmation
from twinr.automations import (
    AutomationAction,
    AutomationDefinition,
    TimeAutomationTrigger,
    build_sensor_trigger,
    describe_sensor_trigger,
)


def handle_list_automations(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    include_disabled = bool(arguments.get("include_disabled", False))
    records = list(owner.runtime.list_automation_records())
    if not include_disabled:
        records = [record for record in records if bool(record.get("enabled", True))]
    serialised = [serialize_automation_record(record) for record in records]
    owner.emit(f"automation_list_count={len(serialised)}")
    owner._record_event(
        "automation_listed",
        "Realtime tool listed configured automations.",
        count=len(serialised),
        include_disabled=include_disabled,
    )
    return {
        "status": "ok",
        "count": len(serialised),
        "automations": serialised,
    }


def handle_create_time_automation(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    require_sensitive_voice_confirmation(owner, arguments, action_label="create or change scheduled automations")
    name = str(arguments.get("name", "")).strip()
    schedule = str(arguments.get("schedule", "")).strip().lower()
    if not name or not schedule:
        raise RuntimeError("create_time_automation requires `name` and `schedule`")
    action = build_automation_action(owner, arguments)
    weekdays = parse_weekdays(arguments.get("weekdays"))
    entry = owner.runtime.create_time_automation(
        name=name,
        description=str(arguments.get("description", "")).strip() or None,
        enabled=optional_bool(arguments, "enabled", default=True),
        schedule=schedule,
        due_at=str(arguments.get("due_at", "")).strip() or None,
        time_of_day=str(arguments.get("time_of_day", "")).strip() or None,
        weekdays=weekdays,
        timezone_name=str(arguments.get("timezone_name", "")).strip() or owner.config.local_timezone_name,
        actions=(action,),
        source="create_time_automation",
        tags=parse_tags(arguments.get("tags")),
    )
    owner.emit("automation_tool_call=true")
    owner.emit(f"automation_created={entry.name}")
    owner.emit(f"automation_id={entry.automation_id}")
    owner._record_event(
        "automation_tool_created",
        "Realtime tool created a time-based automation.",
        automation_id=entry.automation_id,
        name=entry.name,
        schedule=schedule,
    )
    return {
        "status": "created",
        "automation": serialize_automation_record(owner.runtime.automation_store.engine.tool_record(entry)),
    }


def handle_create_sensor_automation(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    require_sensitive_voice_confirmation(owner, arguments, action_label="create or change sensor automations")
    name = str(arguments.get("name", "")).strip()
    trigger_kind = str(arguments.get("trigger_kind", "")).strip().lower()
    if not name or not trigger_kind:
        raise RuntimeError("create_sensor_automation requires `name` and `trigger_kind`")
    trigger = build_sensor_automation_trigger(arguments)
    action = build_automation_action(owner, arguments)
    entry = owner.runtime.create_if_then_automation(
        name=name,
        description=str(arguments.get("description", "")).strip() or None,
        enabled=optional_bool(arguments, "enabled", default=True),
        event_name=trigger.event_name,
        all_conditions=trigger.all_conditions,
        any_conditions=trigger.any_conditions,
        cooldown_seconds=trigger.cooldown_seconds,
        actions=(action,),
        source="create_sensor_automation",
        tags=build_sensor_automation_tags(trigger_kind, parse_tags(arguments.get("tags"))),
    )
    owner.emit("automation_tool_call=true")
    owner.emit(f"automation_created={entry.name}")
    owner.emit(f"automation_id={entry.automation_id}")
    owner.emit(f"automation_sensor_trigger={trigger_kind}")
    owner._record_event(
        "automation_tool_created",
        "Realtime tool created a sensor-triggered automation.",
        automation_id=entry.automation_id,
        name=entry.name,
        trigger_kind=trigger_kind,
    )
    return {
        "status": "created",
        "automation": serialize_automation_record(owner.runtime.automation_store.engine.tool_record(entry)),
    }


def handle_update_time_automation(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    require_sensitive_voice_confirmation(owner, arguments, action_label="change scheduled automations")
    automation_ref = str(arguments.get("automation_ref", "")).strip()
    if not automation_ref:
        raise RuntimeError("update_time_automation requires `automation_ref`")
    entry = resolve_automation_reference(owner, automation_ref)
    if not isinstance(entry.trigger, TimeAutomationTrigger):
        raise RuntimeError("Only time-based automations can be updated with update_time_automation")
    trigger = build_updated_time_trigger(owner, entry, arguments)
    actions = build_updated_automation_actions(owner, entry, arguments)
    updated = owner.runtime.update_automation(
        entry.automation_id,
        name=(str(arguments.get("name", "")).strip() or None) if "name" in arguments else None,
        description=(str(arguments.get("description", "")).strip() or None) if "description" in arguments else None,
        enabled=optional_bool(arguments, "enabled", default=None),
        trigger=trigger,
        actions=actions,
        source="update_time_automation",
        tags=parse_tags(arguments.get("tags")) if "tags" in arguments else None,
    )
    owner.emit("automation_tool_call=true")
    owner.emit(f"automation_updated={updated.name}")
    owner.emit(f"automation_id={updated.automation_id}")
    owner._record_event(
        "automation_tool_updated",
        "Realtime tool updated a time-based automation.",
        automation_id=updated.automation_id,
        name=updated.name,
    )
    return {
        "status": "updated",
        "automation": serialize_automation_record(owner.runtime.automation_store.engine.tool_record(updated)),
    }


def handle_update_sensor_automation(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    require_sensitive_voice_confirmation(owner, arguments, action_label="change sensor automations")
    automation_ref = str(arguments.get("automation_ref", "")).strip()
    if not automation_ref:
        raise RuntimeError("update_sensor_automation requires `automation_ref`")
    entry = resolve_automation_reference(owner, automation_ref)
    sensor_spec = describe_sensor_trigger(entry.trigger)
    if sensor_spec is None:
        raise RuntimeError("Only supported sensor-triggered automations can be updated with update_sensor_automation")
    trigger = build_updated_sensor_trigger(entry, arguments)
    actions = build_updated_automation_actions(owner, entry, arguments)
    updated = owner.runtime.update_automation(
        entry.automation_id,
        name=(str(arguments.get("name", "")).strip() or None) if "name" in arguments else None,
        description=(str(arguments.get("description", "")).strip() or None) if "description" in arguments else None,
        enabled=optional_bool(arguments, "enabled", default=None),
        trigger=trigger,
        actions=actions,
        source="update_sensor_automation",
        tags=build_updated_sensor_tags(entry, arguments),
    )
    updated_spec = describe_sensor_trigger(updated.trigger)
    owner.emit("automation_tool_call=true")
    owner.emit(f"automation_updated={updated.name}")
    owner.emit(f"automation_id={updated.automation_id}")
    if updated_spec is not None:
        owner.emit(f"automation_sensor_trigger={updated_spec.trigger_kind}")
    owner._record_event(
        "automation_tool_updated",
        "Realtime tool updated a sensor-triggered automation.",
        automation_id=updated.automation_id,
        name=updated.name,
        trigger_kind=updated_spec.trigger_kind if updated_spec is not None else None,
    )
    return {
        "status": "updated",
        "automation": serialize_automation_record(owner.runtime.automation_store.engine.tool_record(updated)),
    }


def handle_delete_automation(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    require_sensitive_voice_confirmation(owner, arguments, action_label="delete automations")
    automation_ref = str(arguments.get("automation_ref", "")).strip()
    if not automation_ref:
        raise RuntimeError("delete_automation requires `automation_ref`")
    entry = resolve_automation_reference(owner, automation_ref)
    removed = owner.runtime.delete_automation(entry.automation_id, source="delete_automation")
    owner.emit("automation_tool_call=true")
    owner.emit(f"automation_deleted={removed.name}")
    owner.emit(f"automation_id={removed.automation_id}")
    owner._record_event(
        "automation_tool_deleted",
        "Realtime tool deleted an automation.",
        automation_id=removed.automation_id,
        name=removed.name,
    )
    return {
        "status": "deleted",
        "automation_id": removed.automation_id,
        "name": removed.name,
    }


def build_automation_action(
    owner: Any,
    arguments: dict[str, object],
    *,
    fallback: AutomationAction | None = None,
) -> AutomationAction:
    fallback_delivery = "spoken"
    if fallback is not None:
        if fallback.kind == "print":
            fallback_delivery = "printed"
        elif fallback.kind == "llm_prompt":
            fallback_delivery = normalize_delivery(fallback.payload.get("delivery"))
    allow_web_search = optional_bool(
        arguments,
        "allow_web_search",
        default=bool(fallback.payload.get("allow_web_search")) if fallback is not None else False,
    )
    delivery = normalize_delivery(arguments.get("delivery") if "delivery" in arguments else fallback_delivery)
    raw_mode = (
        str(arguments.get("content_mode", "")).strip().lower()
        if "content_mode" in arguments
        else ("llm_prompt" if fallback is None or fallback.kind == "llm_prompt" else "static_text")
    )
    content_mode = "static_text" if raw_mode == "static_text" else "llm_prompt"
    content = (
        str(arguments.get("content", "")).strip()
        if "content" in arguments
        else (fallback.text or "" if fallback is not None else "")
    )
    if not content:
        raise RuntimeError("Automation content must not be empty")
    if content_mode == "llm_prompt":
        return AutomationAction(
            kind="llm_prompt",
            text=content,
            payload={
                "delivery": delivery,
                "allow_web_search": allow_web_search,
            },
        )
    return AutomationAction(
        kind="print" if delivery == "printed" else "say",
        text=content,
    )


def build_updated_time_trigger(
    owner: Any,
    entry: AutomationDefinition,
    arguments: dict[str, object],
) -> TimeAutomationTrigger | None:
    existing = entry.trigger
    if not isinstance(existing, TimeAutomationTrigger):
        raise RuntimeError("Only time-based automations can be updated")
    trigger_fields = {"schedule", "due_at", "time_of_day", "weekdays", "timezone_name"}
    if not any(field in arguments for field in trigger_fields):
        return None
    schedule = str(arguments.get("schedule", existing.schedule)).strip().lower() or existing.schedule
    weekdays = parse_weekdays(arguments.get("weekdays")) if "weekdays" in arguments else existing.weekdays
    timezone_name = str(arguments.get("timezone_name", existing.timezone_name or "")).strip() or (
        existing.timezone_name or owner.config.local_timezone_name
    )
    due_at = str(arguments.get("due_at", existing.due_at or "")).strip() or None
    time_of_day = str(arguments.get("time_of_day", existing.time_of_day or "")).strip() or None
    return TimeAutomationTrigger(
        schedule=schedule,
        due_at=due_at,
        time_of_day=time_of_day,
        weekdays=weekdays,
        timezone_name=timezone_name,
    )


def build_updated_automation_actions(
    owner: Any,
    entry: AutomationDefinition,
    arguments: dict[str, object],
) -> tuple[AutomationAction, ...] | None:
    action_fields = {"delivery", "content_mode", "content", "allow_web_search"}
    if not any(field in arguments for field in action_fields):
        return None
    if not entry.actions:
        raise RuntimeError("Automation has no actions to update")
    primary = entry.actions[0]
    return (build_automation_action(owner, arguments, fallback=primary),)


def build_sensor_automation_trigger(arguments: dict[str, object], *, fallback: AutomationDefinition | None = None):
    fallback_spec = describe_sensor_trigger(fallback.trigger) if fallback is not None else None
    trigger_kind = (
        str(arguments.get("trigger_kind", "")).strip().lower()
        if "trigger_kind" in arguments
        else (fallback_spec.trigger_kind if fallback_spec is not None else "")
    )
    if not trigger_kind:
        raise RuntimeError("Sensor automation trigger_kind must not be empty")
    hold_default = fallback_spec.hold_seconds if fallback_spec is not None else 0.0
    cooldown_default = fallback_spec.cooldown_seconds if fallback_spec is not None else 0.0
    hold_seconds = optional_float(arguments, "hold_seconds", default=hold_default)
    cooldown_seconds = optional_float(arguments, "cooldown_seconds", default=cooldown_default)
    try:
        return build_sensor_trigger(
            trigger_kind,
            hold_seconds=hold_seconds,
            cooldown_seconds=cooldown_seconds,
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def build_updated_sensor_trigger(entry: AutomationDefinition, arguments: dict[str, object]):
    sensor_fields = {"trigger_kind", "hold_seconds", "cooldown_seconds"}
    if not any(field in arguments for field in sensor_fields):
        return None
    return build_sensor_automation_trigger(arguments, fallback=entry)


def build_sensor_automation_tags(trigger_kind: str, tags: tuple[str, ...]) -> tuple[str, ...]:
    normalized_tags = list(tags)
    normalized_tags.extend(("sensor", trigger_kind))
    return tuple(dict.fromkeys(tag for tag in normalized_tags if tag))


def build_updated_sensor_tags(entry: AutomationDefinition, arguments: dict[str, object]) -> tuple[str, ...] | None:
    sensor_spec = describe_sensor_trigger(entry.trigger)
    if sensor_spec is None:
        return None
    if "tags" in arguments:
        base_tags = parse_tags(arguments.get("tags"))
    elif "trigger_kind" not in arguments:
        return None
    else:
        base_tags = tuple(tag for tag in entry.tags if tag not in {"sensor", sensor_spec.trigger_kind})
    next_kind = (
        str(arguments.get("trigger_kind", "")).strip().lower()
        if "trigger_kind" in arguments
        else sensor_spec.trigger_kind
    )
    return build_sensor_automation_tags(next_kind, tuple(base_tags))


def resolve_automation_reference(owner: Any, automation_ref: str) -> AutomationDefinition:
    exact = owner.runtime.automation_store.get(automation_ref)
    if exact is not None:
        return exact
    normalized_ref = automation_ref.strip().lower()
    if not normalized_ref:
        raise RuntimeError("Automation reference must not be empty")
    entries = owner.runtime.automation_store.load_entries()
    exact_name_matches = [entry for entry in entries if entry.name.strip().lower() == normalized_ref]
    if len(exact_name_matches) == 1:
        return exact_name_matches[0]
    partial_matches = [entry for entry in entries if normalized_ref in entry.name.strip().lower()]
    if len(partial_matches) == 1:
        return partial_matches[0]
    if len(exact_name_matches) > 1 or len(partial_matches) > 1:
        raise RuntimeError("Automation reference is ambiguous; call list_automations first.")
    raise RuntimeError("No automation matched that reference.")


def serialize_automation_record(record: dict[str, object]) -> dict[str, object]:
    actions = tuple(record.get("actions", ()) or ())
    primary_action = actions[0] if actions else {}
    if not isinstance(primary_action, dict):
        primary_action = {}
    payload = primary_action.get("payload", {})
    if not isinstance(payload, dict):
        payload = {}
    content_preview = str(primary_action.get("text", "")).strip()
    serialised = {
        "automation_id": record.get("automation_id"),
        "name": record.get("name"),
        "description": record.get("description"),
        "enabled": record.get("enabled"),
        "trigger_kind": record.get("trigger_kind"),
        "schedule": (record.get("trigger") or {}).get("schedule") if isinstance(record.get("trigger"), dict) else None,
        "due_at": (record.get("trigger") or {}).get("due_at") if isinstance(record.get("trigger"), dict) else None,
        "time_of_day": (record.get("trigger") or {}).get("time_of_day") if isinstance(record.get("trigger"), dict) else None,
        "weekdays": (record.get("trigger") or {}).get("weekdays") if isinstance(record.get("trigger"), dict) else None,
        "timezone_name": (record.get("trigger") or {}).get("timezone_name") if isinstance(record.get("trigger"), dict) else None,
        "next_run_at": record.get("next_run_at"),
        "due_now": record.get("due_now"),
        "delivery": payload.get("delivery", "printed" if primary_action.get("kind") == "print" else "spoken"),
        "content_mode": "llm_prompt" if primary_action.get("kind") == "llm_prompt" else "static_text",
        "content_preview": content_preview[:120] + ("…" if len(content_preview) > 120 else ""),
    }
    if record.get("trigger_kind") == "if_then":
        serialised.update(
            {
                "event_name": record.get("event_name"),
                "cooldown_seconds": record.get("cooldown_seconds"),
                "sensor_trigger_kind": record.get("sensor_trigger_kind"),
                "sensor_hold_seconds": record.get("sensor_hold_seconds"),
                "sensor_label": record.get("sensor_label"),
            }
        )
    return serialised


def normalize_delivery(raw_value: object) -> str:
    normalized = str(raw_value or "").strip().lower()
    if normalized in {"print", "printed", "printer"}:
        return "printed"
    return "spoken"


def parse_weekdays(raw_value: object) -> tuple[int, ...]:
    if raw_value is None or raw_value == "":
        return ()
    if not isinstance(raw_value, (list, tuple)):
        raise RuntimeError("weekdays must be an array of weekday numbers 0-6")
    weekdays: list[int] = []
    for item in raw_value:
        try:
            weekday = int(item)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("weekdays must be integers 0-6") from exc
        if weekday < 0 or weekday > 6:
            raise RuntimeError("weekdays must use integers 0-6")
        weekdays.append(weekday)
    return tuple(sorted(set(weekdays)))


def parse_tags(raw_value: object) -> tuple[str, ...]:
    if raw_value is None or raw_value == "":
        return ()
    if not isinstance(raw_value, (list, tuple)):
        raise RuntimeError("tags must be an array of short strings")
    return tuple(str(item).strip() for item in raw_value if str(item).strip())
