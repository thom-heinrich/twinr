"""Handle automation tool calls for realtime Twinr sessions.

Exports synchronous handler entrypoints for listing, creating, updating, and
deleting scheduled and sensor-triggered automations at the runtime boundary.
"""

from __future__ import annotations

from math import isfinite
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .support import optional_bool, optional_float, require_sensitive_voice_confirmation
from twinr.automations import (
    AutomationAction,
    AutomationDefinition,
    TimeAutomationTrigger,
    build_sensor_trigger,
    describe_sensor_trigger,
)

_UNSET = object()
_PRINT_DELIVERY_ALIASES = {"print", "printed", "printer"}
_SPOKEN_DELIVERY_ALIASES = {"say", "speak", "spoken", "speech", "voice", "audio"}
_MAX_TELEMETRY_VALUE_LENGTH = 256
_MAX_TAG_LENGTH = 64


def handle_list_automations(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """List configured automations from the current runtime.

    Args:
        owner: Tool executor owner exposing runtime access and telemetry hooks.
        arguments: Tool payload. Supports ``include_disabled`` to include
            disabled automation records.

    Returns:
        JSON-safe payload with ``status``, ``count``, and serialized
        ``automations``.

    Raises:
        RuntimeError: If the runtime rejects the request or arguments cannot be
            parsed safely.
    """
    include_disabled = optional_bool(arguments, "include_disabled", default=False)  # AUDIT-FIX(#9): Parse tool booleans explicitly instead of using bool("false")==True.
    records = _call_runtime(
        "list automations right now",
        lambda: list(owner.runtime.list_automation_records()),
    )  # AUDIT-FIX(#2): Convert low-level runtime failures into stable tool errors.
    filtered_records: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        if include_disabled or _coerce_bool_like(record.get("enabled", True), default=True):  # AUDIT-FIX(#9): Harden bool parsing against corrupted string-backed state.
            filtered_records.append(record)
    serialised = [serialize_automation_record(record) for record in filtered_records]
    _safe_emit_kv(owner, "automation_list_count", len(serialised))  # AUDIT-FIX(#3): Telemetry is best-effort and sanitized so it cannot fail the tool or inject log content.
    _safe_record_event(
        owner,
        "automation_listed",
        "Realtime tool listed configured automations.",
        count=len(serialised),
        include_disabled=include_disabled,
    )  # AUDIT-FIX(#3): Event recording must never turn a successful read into an error.
    return {
        "status": "ok",
        "count": len(serialised),
        "automations": serialised,
    }


def handle_create_time_automation(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """Create a time-based automation from a tool payload.

    Args:
        owner: Tool executor owner exposing runtime access and telemetry hooks.
        arguments: Tool payload with required ``name`` and ``schedule`` plus
            optional trigger, action, tag, and confirmation fields.

    Returns:
        JSON-safe payload with ``status="created"`` and the created
        ``automation`` record.

    Raises:
        SensitiveActionConfirmationRequired: If spoken confirmation is required
            before changing automations.
        RuntimeError: If required fields are missing, invalid, duplicated, or
            the runtime rejects creation.
    """
    require_sensitive_voice_confirmation(owner, arguments, action_label="create or change scheduled automations")
    name = _text_or_empty(arguments.get("name"))  # AUDIT-FIX(#10): Treat None as empty input instead of the literal string "None".
    schedule = _lower_text_or_empty(arguments.get("schedule"))  # AUDIT-FIX(#10): Avoid stringifying null schedule values into invalid persisted data.
    if not name or not schedule:
        raise RuntimeError("create_time_automation requires `name` and `schedule`")
    ensure_unique_automation_name(owner, name)  # AUDIT-FIX(#6): Prevent duplicate case-insensitive names that later make voice references ambiguous.
    action = build_automation_action(owner, arguments)
    weekdays = parse_weekdays(arguments.get("weekdays"))
    timezone_name = resolve_timezone_name(owner, arguments.get("timezone_name"))  # AUDIT-FIX(#11): Validate timezone_name early and fall back to UTC-safe defaults.
    entry = _call_runtime(
        "create the scheduled automation",
        lambda: owner.runtime.create_time_automation(
            name=name,
            description=_text_or_empty(arguments.get("description")) or None,  # AUDIT-FIX(#10): Do not persist "None" as a description.
            enabled=optional_bool(arguments, "enabled", default=True),
            schedule=schedule,
            due_at=_text_or_empty(arguments.get("due_at")) or None,  # AUDIT-FIX(#10): Keep nullable time fields truly nullable.
            time_of_day=_text_or_empty(arguments.get("time_of_day")) or None,  # AUDIT-FIX(#10): Keep nullable time fields truly nullable.
            weekdays=weekdays,
            timezone_name=timezone_name,
            actions=(action,),
            source="create_time_automation",
            tags=parse_tags(arguments.get("tags")),
        ),
    )  # AUDIT-FIX(#2): Runtime/store failures should surface as stable tool errors, not raw internals.
    _safe_emit(owner, "automation_tool_call=true")  # AUDIT-FIX(#3): Best-effort telemetry prevents duplicate retries after successful mutation.
    _safe_emit_kv(owner, "automation_created", entry.name)  # AUDIT-FIX(#3): Sanitize user-controlled values before emitting key=value telemetry.
    _safe_emit_kv(owner, "automation_id", entry.automation_id)  # AUDIT-FIX(#3): Telemetry must not break the mutation response path.
    _safe_record_event(
        owner,
        "automation_tool_created",
        "Realtime tool created a time-based automation.",
        automation_id=entry.automation_id,
        name=entry.name,
        schedule=schedule,
    )  # AUDIT-FIX(#3): Event logging is intentionally non-fatal after state changes.
    return {
        "status": "created",
        "automation": serialize_automation_response(owner, entry),  # AUDIT-FIX(#4): Fall back to entry-based serialization if tool_record fails post-commit.
    }


def handle_create_sensor_automation(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """Create a sensor-triggered automation from a tool payload.

    Args:
        owner: Tool executor owner exposing runtime access and telemetry hooks.
        arguments: Tool payload with required ``name`` and ``trigger_kind``
            plus optional action, timing, tag, and confirmation fields.

    Returns:
        JSON-safe payload with ``status="created"`` and the created
        ``automation`` record.

    Raises:
        SensitiveActionConfirmationRequired: If spoken confirmation is required
            before changing automations.
        RuntimeError: If required fields are missing, invalid, duplicated, or
            the runtime rejects creation.
    """
    require_sensitive_voice_confirmation(owner, arguments, action_label="create or change sensor automations")
    name = _text_or_empty(arguments.get("name"))  # AUDIT-FIX(#10): Treat None as empty input instead of the literal string "None".
    trigger_kind = _lower_text_or_empty(arguments.get("trigger_kind"))  # AUDIT-FIX(#10): Avoid stringifying null trigger kinds into invalid values.
    if not name or not trigger_kind:
        raise RuntimeError("create_sensor_automation requires `name` and `trigger_kind`")
    ensure_unique_automation_name(owner, name)  # AUDIT-FIX(#6): Prevent duplicate names before creating new automations.
    trigger = build_sensor_automation_trigger(arguments)
    sensor_spec = describe_sensor_trigger(trigger)
    canonical_trigger_kind = sensor_spec.trigger_kind if sensor_spec is not None else trigger_kind  # AUDIT-FIX(#12): Use the canonical trigger kind returned by the trigger builder.
    action = build_automation_action(owner, arguments)
    entry = _call_runtime(
        "create the sensor automation",
        lambda: owner.runtime.create_if_then_automation(
            name=name,
            description=_text_or_empty(arguments.get("description")) or None,  # AUDIT-FIX(#10): Do not persist "None" as a description.
            enabled=optional_bool(arguments, "enabled", default=True),
            event_name=trigger.event_name,
            all_conditions=trigger.all_conditions,
            any_conditions=trigger.any_conditions,
            cooldown_seconds=trigger.cooldown_seconds,
            actions=(action,),
            source="create_sensor_automation",
            tags=build_sensor_automation_tags(canonical_trigger_kind, parse_tags(arguments.get("tags"))),  # AUDIT-FIX(#12): Keep stored sensor tags aligned with the canonical trigger kind.
        ),
    )  # AUDIT-FIX(#2): Runtime/store failures should surface as stable tool errors, not raw internals.
    _safe_emit(owner, "automation_tool_call=true")  # AUDIT-FIX(#3): Best-effort telemetry prevents false-negative mutation results.
    _safe_emit_kv(owner, "automation_created", entry.name)  # AUDIT-FIX(#3): Sanitize user-controlled values before emitting key=value telemetry.
    _safe_emit_kv(owner, "automation_id", entry.automation_id)  # AUDIT-FIX(#3): Telemetry must not fail the command after commit.
    _safe_emit_kv(owner, "automation_sensor_trigger", canonical_trigger_kind)  # AUDIT-FIX(#12): Emit the normalized trigger kind, not the raw caller input.
    _safe_record_event(
        owner,
        "automation_tool_created",
        "Realtime tool created a sensor-triggered automation.",
        automation_id=entry.automation_id,
        name=entry.name,
        trigger_kind=canonical_trigger_kind,
    )  # AUDIT-FIX(#3): Event logging is intentionally non-fatal after state changes.
    return {
        "status": "created",
        "automation": serialize_automation_response(owner, entry),  # AUDIT-FIX(#4): Fall back to entry-based serialization if tool_record fails post-commit.
    }


def handle_update_time_automation(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """Update a time-based automation by reference.

    Args:
        owner: Tool executor owner exposing runtime access and telemetry hooks.
        arguments: Tool payload with required ``automation_ref`` and one or
            more mutable fields such as schedule, content, tags, or enabled
            state.

    Returns:
        JSON-safe payload with ``status="updated"`` and the updated
        ``automation`` record.

    Raises:
        SensitiveActionConfirmationRequired: If spoken confirmation is required
            before changing automations.
        RuntimeError: If the reference is missing, does not resolve to a
            time-based automation, no fields changed, or the runtime rejects
            the update.
    """
    require_sensitive_voice_confirmation(owner, arguments, action_label="change scheduled automations")
    automation_ref = _text_or_empty(arguments.get("automation_ref"))  # AUDIT-FIX(#10): Treat None as empty input instead of the literal string "None".
    if not automation_ref:
        raise RuntimeError("update_time_automation requires `automation_ref`")
    entry = resolve_automation_reference(owner, automation_ref, allow_partial_match=False)  # AUDIT-FIX(#1): Mutating operations must not target partial-name matches.
    if not isinstance(entry.trigger, TimeAutomationTrigger):
        raise RuntimeError("Only time-based automations can be updated with update_time_automation")
    name_update = _optional_non_empty_argument(arguments, "name")
    description_update = _optional_non_empty_argument(arguments, "description")
    if name_update is not _UNSET:
        ensure_unique_automation_name(owner, name_update, excluding_automation_id=entry.automation_id)  # AUDIT-FIX(#6): Preserve unique names during rename operations.
    enabled_update = optional_bool(arguments, "enabled", default=None) if "enabled" in arguments else None
    trigger = build_updated_time_trigger(owner, entry, arguments)
    actions = build_updated_automation_actions(owner, entry, arguments)
    tags_update = parse_tags(arguments.get("tags")) if "tags" in arguments else None
    if (
        name_update is _UNSET
        and description_update is _UNSET
        and enabled_update is None
        and trigger is None
        and actions is None
        and tags_update is None
    ):
        raise RuntimeError("update_time_automation requires at least one field to change")  # AUDIT-FIX(#5): Reject silent no-op updates that would falsely report success.
    updated = _call_runtime(
        "update the scheduled automation",
        lambda: owner.runtime.update_automation(
            entry.automation_id,
            name=None if name_update is _UNSET else name_update,
            description=None if description_update is _UNSET else description_update,
            enabled=enabled_update,
            trigger=trigger,
            actions=actions,
            source="update_time_automation",
            tags=tags_update,
        ),
    )  # AUDIT-FIX(#2): Runtime/store failures should surface as stable tool errors, not raw internals.
    _safe_emit(owner, "automation_tool_call=true")  # AUDIT-FIX(#3): Best-effort telemetry prevents false-negative mutation results.
    _safe_emit_kv(owner, "automation_updated", updated.name)  # AUDIT-FIX(#3): Sanitize user-controlled values before emitting key=value telemetry.
    _safe_emit_kv(owner, "automation_id", updated.automation_id)  # AUDIT-FIX(#3): Telemetry must not fail the command after commit.
    _safe_record_event(
        owner,
        "automation_tool_updated",
        "Realtime tool updated a time-based automation.",
        automation_id=updated.automation_id,
        name=updated.name,
    )  # AUDIT-FIX(#3): Event logging is intentionally non-fatal after state changes.
    return {
        "status": "updated",
        "automation": serialize_automation_response(owner, updated),  # AUDIT-FIX(#4): Fall back to entry-based serialization if tool_record fails post-commit.
    }


def handle_update_sensor_automation(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """Update a sensor-triggered automation by reference.

    Args:
        owner: Tool executor owner exposing runtime access and telemetry hooks.
        arguments: Tool payload with required ``automation_ref`` and one or
            more mutable fields such as trigger timing, content, tags, or
            enabled state.

    Returns:
        JSON-safe payload with ``status="updated"`` and the updated
        ``automation`` record.

    Raises:
        SensitiveActionConfirmationRequired: If spoken confirmation is required
            before changing automations.
        RuntimeError: If the reference is missing, does not resolve to a
            supported sensor automation, no fields changed, or the runtime
            rejects the update.
    """
    require_sensitive_voice_confirmation(owner, arguments, action_label="change sensor automations")
    automation_ref = _text_or_empty(arguments.get("automation_ref"))  # AUDIT-FIX(#10): Treat None as empty input instead of the literal string "None".
    if not automation_ref:
        raise RuntimeError("update_sensor_automation requires `automation_ref`")
    entry = resolve_automation_reference(owner, automation_ref, allow_partial_match=False)  # AUDIT-FIX(#1): Mutating operations must not target partial-name matches.
    sensor_spec = describe_sensor_trigger(entry.trigger)
    if sensor_spec is None:
        raise RuntimeError("Only supported sensor-triggered automations can be updated with update_sensor_automation")
    name_update = _optional_non_empty_argument(arguments, "name")
    description_update = _optional_non_empty_argument(arguments, "description")
    if name_update is not _UNSET:
        ensure_unique_automation_name(owner, name_update, excluding_automation_id=entry.automation_id)  # AUDIT-FIX(#6): Preserve unique names during rename operations.
    enabled_update = optional_bool(arguments, "enabled", default=None) if "enabled" in arguments else None
    trigger = build_updated_sensor_trigger(entry, arguments)
    actions = build_updated_automation_actions(owner, entry, arguments)
    tags_update = build_updated_sensor_tags(entry, arguments, next_trigger=trigger)
    if (
        name_update is _UNSET
        and description_update is _UNSET
        and enabled_update is None
        and trigger is None
        and actions is None
        and tags_update is None
    ):
        raise RuntimeError("update_sensor_automation requires at least one field to change")  # AUDIT-FIX(#5): Reject silent no-op updates that would falsely report success.
    updated = _call_runtime(
        "update the sensor automation",
        lambda: owner.runtime.update_automation(
            entry.automation_id,
            name=None if name_update is _UNSET else name_update,
            description=None if description_update is _UNSET else description_update,
            enabled=enabled_update,
            trigger=trigger,
            actions=actions,
            source="update_sensor_automation",
            tags=tags_update,
        ),
    )  # AUDIT-FIX(#2): Runtime/store failures should surface as stable tool errors, not raw internals.
    updated_spec = describe_sensor_trigger(updated.trigger)
    _safe_emit(owner, "automation_tool_call=true")  # AUDIT-FIX(#3): Best-effort telemetry prevents false-negative mutation results.
    _safe_emit_kv(owner, "automation_updated", updated.name)  # AUDIT-FIX(#3): Sanitize user-controlled values before emitting key=value telemetry.
    _safe_emit_kv(owner, "automation_id", updated.automation_id)  # AUDIT-FIX(#3): Telemetry must not fail the command after commit.
    if updated_spec is not None:
        _safe_emit_kv(owner, "automation_sensor_trigger", updated_spec.trigger_kind)  # AUDIT-FIX(#12): Emit the normalized trigger kind, not the raw caller input.
    _safe_record_event(
        owner,
        "automation_tool_updated",
        "Realtime tool updated a sensor-triggered automation.",
        automation_id=updated.automation_id,
        name=updated.name,
        trigger_kind=updated_spec.trigger_kind if updated_spec is not None else None,
    )  # AUDIT-FIX(#3): Event logging is intentionally non-fatal after state changes.
    return {
        "status": "updated",
        "automation": serialize_automation_response(owner, updated),  # AUDIT-FIX(#4): Fall back to entry-based serialization if tool_record fails post-commit.
    }


def handle_delete_automation(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """Delete an automation by reference.

    Args:
        owner: Tool executor owner exposing runtime access and telemetry hooks.
        arguments: Tool payload with required ``automation_ref`` and optional
            confirmation fields.

    Returns:
        JSON-safe payload with ``status="deleted"``, the removed
        ``automation_id``, and its ``name``.

    Raises:
        SensitiveActionConfirmationRequired: If spoken confirmation is required
            before deleting automations.
        RuntimeError: If the reference is missing, ambiguous, or the runtime
            rejects deletion.
    """
    require_sensitive_voice_confirmation(owner, arguments, action_label="delete automations")
    automation_ref = _text_or_empty(arguments.get("automation_ref"))  # AUDIT-FIX(#10): Treat None as empty input instead of the literal string "None".
    if not automation_ref:
        raise RuntimeError("delete_automation requires `automation_ref`")
    entry = resolve_automation_reference(owner, automation_ref, allow_partial_match=False)  # AUDIT-FIX(#1): Destructive operations must not target partial-name matches.
    removed = _call_runtime(
        "delete the automation",
        lambda: owner.runtime.delete_automation(entry.automation_id, source="delete_automation"),
    )  # AUDIT-FIX(#2): Runtime/store failures should surface as stable tool errors, not raw internals.
    _safe_emit(owner, "automation_tool_call=true")  # AUDIT-FIX(#3): Best-effort telemetry prevents false-negative mutation results.
    _safe_emit_kv(owner, "automation_deleted", removed.name)  # AUDIT-FIX(#3): Sanitize user-controlled values before emitting key=value telemetry.
    _safe_emit_kv(owner, "automation_id", removed.automation_id)  # AUDIT-FIX(#3): Telemetry must not fail the command after commit.
    _safe_record_event(
        owner,
        "automation_tool_deleted",
        "Realtime tool deleted an automation.",
        automation_id=removed.automation_id,
        name=removed.name,
    )  # AUDIT-FIX(#3): Event logging is intentionally non-fatal after state changes.
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
    fallback_payload = action_payload_dict(fallback)  # AUDIT-FIX(#7): Legacy/non-LLM actions may have payload=None or a non-dict payload.
    if fallback is not None:
        if fallback.kind == "print":
            fallback_delivery = "printed"
        elif fallback.kind == "llm_prompt":
            fallback_delivery = normalize_delivery(fallback_payload.get("delivery"))
    allow_web_search = optional_bool(
        arguments,
        "allow_web_search",
        default=_coerce_bool_like(fallback_payload.get("allow_web_search"), default=False) if fallback is not None else False,  # AUDIT-FIX(#7): Read fallback payload only after normalizing it to a dict.
    )
    default_content_mode = "llm_prompt" if fallback is None or fallback.kind == "llm_prompt" else "static_text"
    delivery = parse_delivery(
        arguments.get("delivery") if "delivery" in arguments else None,
        default=fallback_delivery,
    )  # AUDIT-FIX(#13): Reject invalid delivery values instead of silently changing output modality.
    content_mode = parse_content_mode(
        arguments.get("content_mode") if "content_mode" in arguments else None,
        default=default_content_mode,
    )  # AUDIT-FIX(#13): Reject invalid content_mode values instead of silently switching to llm_prompt.
    content = (
        _text_or_empty(arguments.get("content"))
        if "content" in arguments
        else _text_or_empty(fallback.text if fallback is not None else "")
    )  # AUDIT-FIX(#10): Treat None as empty content instead of the literal string "None".
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
    schedule = _lower_text_or_empty(arguments.get("schedule", existing.schedule)) or existing.schedule  # AUDIT-FIX(#10): Avoid stringifying null schedule values into invalid persisted data.
    weekdays = parse_weekdays(arguments.get("weekdays")) if "weekdays" in arguments else tuple(existing.weekdays or ())
    timezone_name = resolve_timezone_name(
        owner,
        arguments.get("timezone_name", existing.timezone_name or ""),
        fallback=existing.timezone_name or None,
    )  # AUDIT-FIX(#11): Validate timezone_name early and fall back to UTC-safe defaults.
    due_at = _text_or_empty(arguments.get("due_at", existing.due_at or "")) or None  # AUDIT-FIX(#10): Keep nullable time fields truly nullable.
    time_of_day = _text_or_empty(arguments.get("time_of_day", existing.time_of_day or "")) or None  # AUDIT-FIX(#10): Keep nullable time fields truly nullable.
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


def build_sensor_automation_trigger(
    arguments: dict[str, object],
    *,
    fallback: AutomationDefinition | None = None,
) -> Any:
    fallback_spec = describe_sensor_trigger(fallback.trigger) if fallback is not None else None
    trigger_kind = (
        _lower_text_or_empty(arguments.get("trigger_kind"))
        if "trigger_kind" in arguments
        else (fallback_spec.trigger_kind if fallback_spec is not None else "")
    )  # AUDIT-FIX(#10): Avoid stringifying null trigger kinds into invalid values.
    if not trigger_kind:
        raise RuntimeError("Sensor automation trigger_kind must not be empty")
    hold_default = fallback_spec.hold_seconds if fallback_spec is not None else 0.0
    cooldown_default = fallback_spec.cooldown_seconds if fallback_spec is not None else 0.0
    hold_seconds = validate_non_negative_finite(
        "hold_seconds",
        optional_float(arguments, "hold_seconds", default=hold_default),
    )  # AUDIT-FIX(#8): Reject negative, NaN, and infinite sensor timing values before they reach the runtime.
    cooldown_seconds = validate_non_negative_finite(
        "cooldown_seconds",
        optional_float(arguments, "cooldown_seconds", default=cooldown_default),
    )  # AUDIT-FIX(#8): Reject negative, NaN, and infinite sensor timing values before they reach the runtime.
    try:
        return build_sensor_trigger(
            trigger_kind,
            hold_seconds=hold_seconds,
            cooldown_seconds=cooldown_seconds,
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def build_updated_sensor_trigger(entry: AutomationDefinition, arguments: dict[str, object]) -> Any:
    sensor_fields = {"trigger_kind", "hold_seconds", "cooldown_seconds"}
    if not any(field in arguments for field in sensor_fields):
        return None
    return build_sensor_automation_trigger(arguments, fallback=entry)


def build_sensor_automation_tags(trigger_kind: str, tags: tuple[str, ...]) -> tuple[str, ...]:
    normalized_tags = list(tags)
    normalized_tags.extend(("sensor", trigger_kind))
    return tuple(dict.fromkeys(tag for tag in normalized_tags if tag))


def build_updated_sensor_tags(
    entry: AutomationDefinition,
    arguments: dict[str, object],
    *,
    next_trigger: Any = None,
) -> tuple[str, ...] | None:
    current_spec = describe_sensor_trigger(entry.trigger)
    if current_spec is None:
        return None
    if "tags" not in arguments and "trigger_kind" not in arguments:
        return None
    next_spec = describe_sensor_trigger(next_trigger if next_trigger is not None else entry.trigger)
    next_kind = next_spec.trigger_kind if next_spec is not None else current_spec.trigger_kind  # AUDIT-FIX(#12): Rebuild tags from the canonical trigger kind, not raw caller input.
    if "tags" in arguments:
        base_tags = parse_tags(arguments.get("tags"))
    else:
        base_tags = tuple(tag for tag in tuple(entry.tags or ()) if isinstance(tag, str))
    reserved_tags = {"sensor", current_spec.trigger_kind, next_kind}
    cleaned_tags = tuple(tag for tag in base_tags if tag not in reserved_tags)  # AUDIT-FIX(#12): Remove stale reserved sensor tags before rebuilding canonical tags.
    return build_sensor_automation_tags(next_kind, cleaned_tags)


def resolve_automation_reference(
    owner: Any,
    automation_ref: str,
    *,
    allow_partial_match: bool = True,
) -> AutomationDefinition:
    normalized_input = _text_or_empty(automation_ref)  # AUDIT-FIX(#10): Treat None-like values and surrounding whitespace consistently before lookup.
    if not normalized_input:
        raise RuntimeError("Automation reference must not be empty")
    exact = _call_runtime(
        "look up the automation",
        lambda: owner.runtime.automation_store.get(normalized_input),
    )  # AUDIT-FIX(#2): Convert low-level store failures into stable tool errors.
    if exact is not None:
        return exact
    normalized_ref = normalized_input.lower()
    entries = _call_runtime(
        "load automations",
        lambda: owner.runtime.automation_store.load_entries(),
    )  # AUDIT-FIX(#2): Convert low-level store failures into stable tool errors.
    exact_name_matches = [entry for entry in entries if automation_name_key(getattr(entry, "name", "")) == normalized_ref]
    if len(exact_name_matches) == 1:
        return exact_name_matches[0]
    if len(exact_name_matches) > 1:
        raise RuntimeError("Automation name is not unique; use the automation_id from list_automations.")  # AUDIT-FIX(#6): Duplicate names must be resolved by id, not guessed by the tool.
    if allow_partial_match:
        partial_matches = [entry for entry in entries if normalized_ref in automation_name_key(getattr(entry, "name", ""))]
        if len(partial_matches) == 1:
            return partial_matches[0]
        if len(partial_matches) > 1:
            raise RuntimeError("Automation reference is ambiguous; call list_automations first.")
        raise RuntimeError("No automation matched that reference.")
    raise RuntimeError("No automation matched that exact reference; use the automation_id or exact automation name.")  # AUDIT-FIX(#1): Require exact references for mutating operations.


def serialize_automation_response(owner: Any, entry: AutomationDefinition) -> dict[str, object]:
    try:
        record = owner.runtime.automation_store.engine.tool_record(entry)
    except Exception:
        record = automation_definition_to_record(entry)  # AUDIT-FIX(#4): Fall back to a local serializer when tool_record is unavailable post-commit.
    try:
        return serialize_automation_record(record)
    except Exception:
        return automation_definition_to_record(entry)  # AUDIT-FIX(#4): Return a minimal safe record instead of failing after a successful mutation.


def automation_definition_to_record(entry: AutomationDefinition) -> dict[str, object]:
    trigger = getattr(entry, "trigger", None)
    record: dict[str, object] = {
        "automation_id": getattr(entry, "automation_id", None),
        "name": getattr(entry, "name", None),
        "description": getattr(entry, "description", None),
        "enabled": getattr(entry, "enabled", None),
        "actions": tuple(action_to_record(action) for action in tuple(getattr(entry, "actions", ()) or ())),
        "next_run_at": getattr(entry, "next_run_at", None),
        "due_now": getattr(entry, "due_now", None),
    }
    if isinstance(trigger, TimeAutomationTrigger):
        record["trigger_kind"] = "time"
        record["trigger"] = {
            "schedule": trigger.schedule,
            "due_at": trigger.due_at,
            "time_of_day": trigger.time_of_day,
            "weekdays": tuple(trigger.weekdays or ()),
            "timezone_name": trigger.timezone_name,
        }
        return record
    sensor_spec = describe_sensor_trigger(trigger)
    if sensor_spec is not None:
        record.update(
            {
                "trigger_kind": "if_then",
                "event_name": getattr(trigger, "event_name", getattr(entry, "event_name", None)),
                "cooldown_seconds": sensor_spec.cooldown_seconds,
                "sensor_trigger_kind": sensor_spec.trigger_kind,
                "sensor_hold_seconds": sensor_spec.hold_seconds,
                "sensor_label": getattr(sensor_spec, "label", None),
            }
        )
        return record
    if isinstance(trigger, dict):
        record["trigger_kind"] = trigger.get("kind")
        record["trigger"] = trigger
    return record


def action_to_record(action: AutomationAction) -> dict[str, object]:
    return {
        "kind": getattr(action, "kind", None),
        "text": getattr(action, "text", None),
        "payload": action_payload_dict(action),
    }


def action_payload_dict(action: AutomationAction | None) -> dict[str, object]:
    if action is None:
        return {}
    payload = getattr(action, "payload", None)
    return payload if isinstance(payload, dict) else {}


def _call_runtime(operation_label: str, callback: Any) -> Any:
    try:
        return callback()
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Could not {operation_label}.") from exc


def _safe_emit(owner: Any, event: str) -> None:
    try:
        owner.emit(event)
    except Exception:
        return


def _safe_emit_kv(owner: Any, key: str, value: object) -> None:
    _safe_emit(owner, f"{key}={sanitize_telemetry_value(value)}")


def _safe_record_event(owner: Any, event_name: str, description: str, **payload: object) -> None:
    safe_payload: dict[str, object] = {}
    for key, value in payload.items():
        if value is None or isinstance(value, (bool, int)):
            safe_payload[key] = value
        elif isinstance(value, float):
            safe_payload[key] = value if isfinite(value) else None
        else:
            safe_payload[key] = sanitize_telemetry_value(value)
    try:
        owner._record_event(event_name, description, **safe_payload)
    except Exception:
        return


def sanitize_telemetry_value(raw_value: object) -> str:
    value = _text_or_empty(raw_value)
    cleaned = "".join(" " if char in {"\x00", "\r", "\n"} else char for char in value)
    cleaned = " ".join(cleaned.split())
    return cleaned[:_MAX_TELEMETRY_VALUE_LENGTH]


def ensure_unique_automation_name(
    owner: Any,
    name: str,
    *,
    excluding_automation_id: str | None = None,
) -> None:
    normalized_name = automation_name_key(name)
    if not normalized_name:
        raise RuntimeError("Automation name must not be empty")
    entries = _call_runtime(
        "load automations",
        lambda: owner.runtime.automation_store.load_entries(),
    )  # AUDIT-FIX(#2): Convert low-level store failures into stable tool errors.
    for entry in entries:
        if getattr(entry, "automation_id", None) == excluding_automation_id:
            continue
        if automation_name_key(getattr(entry, "name", "")) == normalized_name:
            raise RuntimeError("An automation with that name already exists; choose a different name.")  # AUDIT-FIX(#6): Prevent future ambiguous lookups in voice flows.


def automation_name_key(raw_value: object) -> str:
    return _text_or_empty(raw_value).lower()


def resolve_timezone_name(owner: Any, raw_value: object, *, fallback: str | None = None) -> str:
    candidate = _text_or_empty(raw_value) or _text_or_empty(fallback) or _text_or_empty(getattr(owner.config, "local_timezone_name", "")) or "UTC"
    try:
        ZoneInfo(candidate)
    except ZoneInfoNotFoundError as exc:
        raise RuntimeError(f"Unknown timezone_name `{candidate}`") from exc
    return candidate


def validate_non_negative_finite(field_name: str, raw_value: object) -> float:
    value = float(raw_value)
    if not isfinite(value):
        raise RuntimeError(f"{field_name} must be a finite number")
    if value < 0:
        raise RuntimeError(f"{field_name} must be zero or greater")
    return value


def parse_delivery(raw_value: object, *, default: str) -> str:
    normalized = _lower_text_or_empty(raw_value)
    if not normalized:
        return default
    if normalized in _PRINT_DELIVERY_ALIASES:
        return "printed"
    if normalized in _SPOKEN_DELIVERY_ALIASES:
        return "spoken"
    raise RuntimeError("delivery must be `spoken` or `printed`")


def parse_content_mode(raw_value: object, *, default: str) -> str:
    normalized = _lower_text_or_empty(raw_value)
    if not normalized:
        return default
    if normalized in {"static_text", "llm_prompt"}:
        return normalized
    raise RuntimeError("content_mode must be `static_text` or `llm_prompt`")


def _optional_non_empty_argument(arguments: dict[str, object], key: str) -> object:
    if key not in arguments:
        return _UNSET
    value = _text_or_empty(arguments.get(key))
    return value if value else _UNSET


def _coerce_bool_like(raw_value: object, *, default: bool) -> bool:
    if raw_value is None:
        return default
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, int):
        if raw_value in (0, 1):
            return bool(raw_value)
        return default
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on", "enabled"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off", "disabled"}:
            return False
    return default


def _text_or_empty(raw_value: object) -> str:
    if raw_value is None:
        return ""
    return str(raw_value).strip()


def _lower_text_or_empty(raw_value: object) -> str:
    return _text_or_empty(raw_value).lower()


def serialize_automation_record(record: dict[str, object]) -> dict[str, object]:
    actions = tuple(record.get("actions", ()) or ())
    primary_action = actions[0] if actions else {}
    if not isinstance(primary_action, dict):
        primary_action = {}
    payload = primary_action.get("payload", {})
    if not isinstance(payload, dict):
        payload = {}
    content_preview = _text_or_empty(primary_action.get("text"))
    trigger = record.get("trigger")
    trigger_dict = trigger if isinstance(trigger, dict) else {}
    serialised = {
        "automation_id": record.get("automation_id"),
        "name": record.get("name"),
        "description": record.get("description"),
        "enabled": record.get("enabled"),
        "trigger_kind": record.get("trigger_kind"),
        "schedule": trigger_dict.get("schedule"),
        "due_at": trigger_dict.get("due_at"),
        "time_of_day": trigger_dict.get("time_of_day"),
        "weekdays": trigger_dict.get("weekdays"),
        "timezone_name": trigger_dict.get("timezone_name"),
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
    normalized = _lower_text_or_empty(raw_value)
    if normalized in _PRINT_DELIVERY_ALIASES:
        return "printed"
    if normalized in _SPOKEN_DELIVERY_ALIASES:
        return "spoken"
    return "spoken"


def parse_weekdays(raw_value: object) -> tuple[int, ...]:
    if raw_value is None or raw_value == "":
        return ()
    if not isinstance(raw_value, (list, tuple)):
        raise RuntimeError("weekdays must be an array of weekday numbers 0-6")
    weekdays: list[int] = []
    for item in raw_value:
        if isinstance(item, bool):
            raise RuntimeError("weekdays must be integers 0-6")  # AUDIT-FIX(#14): Reject bools so True/False cannot silently become weekday 1/0.
        if isinstance(item, int):
            weekday = item
        elif isinstance(item, str):
            normalized = item.strip()
            if not normalized or not normalized.lstrip("+-").isdigit():
                raise RuntimeError("weekdays must be integers 0-6")  # AUDIT-FIX(#14): Reject non-integral numeric strings such as "1.5".
            weekday = int(normalized)
        else:
            raise RuntimeError("weekdays must be integers 0-6")  # AUDIT-FIX(#14): Reject silent coercions from floats and arbitrary objects.
        if weekday < 0 or weekday > 6:
            raise RuntimeError("weekdays must use integers 0-6")
        weekdays.append(weekday)
    return tuple(sorted(set(weekdays)))


def parse_tags(raw_value: object) -> tuple[str, ...]:
    if raw_value is None or raw_value == "":
        return ()
    if not isinstance(raw_value, (list, tuple)):
        raise RuntimeError("tags must be an array of short strings")
    tags: list[str] = []
    for item in raw_value:
        if item is None:
            continue
        if not isinstance(item, str):
            raise RuntimeError("tags must be an array of short strings")  # AUDIT-FIX(#15): Reject non-string tag values instead of persisting str(dict(...)) garbage.
        tag = item.strip()
        if not tag:
            continue
        if len(tag) > _MAX_TAG_LENGTH:
            raise RuntimeError(f"tags must be at most {_MAX_TAG_LENGTH} characters long")  # AUDIT-FIX(#15): Enforce the documented 'short strings' contract.
        if any(ord(char) < 32 for char in tag):
            raise RuntimeError("tags must not contain control characters")  # AUDIT-FIX(#15): Prevent control-character pollution in state and logs.
        tags.append(tag)
    return tuple(dict.fromkeys(tags))  # AUDIT-FIX(#15): Deduplicate tags while preserving the caller's order.
