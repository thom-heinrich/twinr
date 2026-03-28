"""Automation-related tool schema families."""

from __future__ import annotations

from typing import Any

from .context import SchemaBuildContext
from .shared import (
    array_property,
    boolean_property,
    create_time_schedule_rules,
    iso8601_datetime_property,
    number_property,
    string_property,
    time_of_day_property,
    update_sensor_rules,
    update_time_schedule_rules,
    weekdays_property,
)


def build_list_automations_schema(_context: SchemaBuildContext) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "list_automations",
        "description": (
            "List the currently configured time-based and sensor-triggered automations. "
            "Use this for exact questions about current automations and before updating or deleting one when hidden context may be stale."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "include_disabled": boolean_property(
                    "Set true if disabled automations should also be included."
                )
            },
            "required": [],
            "additionalProperties": False,
        },
    }


def build_create_time_automation_schema(_context: SchemaBuildContext) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "create_time_automation",
        "description": (
            "Create a time-based automation for one-off or recurring actions such as daily weather, "
            "daily news, or printed headlines."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": string_property(
                    "Short operator-friendly name for the automation.",
                    min_length=1,
                ),
                "description": string_property(
                    "Optional short description of what the automation does.",
                    min_length=1,
                ),
                "schedule": string_property(
                    "Time schedule type.",
                    enum=["once", "daily", "weekly"],
                ),
                "due_at": iso8601_datetime_property(
                    "Absolute ISO 8601 local datetime with timezone offset for once schedules."
                ),
                "time_of_day": time_of_day_property(
                    "Local time in HH:MM for daily or weekly schedules."
                ),
                "weekdays": weekdays_property(
                    "Weekday numbers for weekly schedules, where Monday is 0 and Sunday is 6."
                ),
                "delivery": string_property(
                    "Whether the automation should speak or print when it runs.",
                    enum=["spoken", "printed"],
                ),
                "content_mode": string_property(
                    "Use llm_prompt for generated content or static_text for fixed wording.",
                    enum=["llm_prompt", "static_text"],
                ),
                "content": string_property(
                    "The prompt or static text the automation should use.",
                    min_length=1,
                ),
                "allow_web_search": boolean_property(
                    "Set true when the automation needs fresh live information from the web."
                ),
                "enabled": boolean_property(
                    "Whether the automation should be active immediately."
                ),
                "tags": array_property(
                    "Optional short tags for operator organization.",
                    string_property("Tag.", min_length=1),
                    unique_items=True,
                ),
                "timezone_name": string_property(
                    "Optional timezone name. Use the local Twinr timezone unless there is a clear reason not to.",
                    min_length=1,
                ),
                "confirmed": boolean_property(
                    "Set true only after the user clearly confirmed this persistent automation change when extra confirmation is needed."
                ),
            },
            "allOf": create_time_schedule_rules(),
            "required": ["name", "schedule", "delivery", "content_mode", "content"],
            "additionalProperties": False,
        },
    }


def build_create_sensor_automation_schema(context: SchemaBuildContext) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "create_sensor_automation",
        "description": (
            "Create an automation triggered by PIR motion, camera visibility/object readings, "
            "or background microphone/VAD state."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": string_property(
                    "Short operator-friendly name for the automation.",
                    min_length=1,
                ),
                "description": string_property(
                    "Optional short description of what the automation does.",
                    min_length=1,
                ),
                "trigger_kind": string_property(
                    context.sensor_trigger_kind_help,
                    enum=context.sensor_trigger_kinds,
                ),
                "hold_seconds": number_property(
                    "Optional required hold duration before firing. Required for quiet/no-motion triggers.",
                    minimum=0,
                ),
                "cooldown_seconds": number_property(
                    "Optional cooldown after the automation fired.",
                    minimum=0,
                ),
                "delivery": string_property(
                    "Whether the automation should speak or print when it runs.",
                    enum=["spoken", "printed"],
                ),
                "content_mode": string_property(
                    "Use llm_prompt for generated content or static_text for fixed wording.",
                    enum=["llm_prompt", "static_text"],
                ),
                "content": string_property(
                    "The prompt or static text the automation should use.",
                    min_length=1,
                ),
                "allow_web_search": boolean_property(
                    "Set true when the automation needs fresh live information from the web."
                ),
                "enabled": boolean_property(
                    "Whether the automation should be active immediately."
                ),
                "tags": array_property(
                    "Optional short tags for operator organization.",
                    string_property("Tag.", min_length=1),
                    unique_items=True,
                ),
                "confirmed": boolean_property(
                    "Set true only after the user clearly confirmed this persistent automation change when extra confirmation is needed."
                ),
            },
            "required": ["name", "trigger_kind", "delivery", "content_mode", "content"],
            "additionalProperties": False,
        },
    }


def build_update_time_automation_schema(_context: SchemaBuildContext) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "update_time_automation",
        "description": "Update an existing time-based automation. Use list_automations first if you need to identify it.",
        "parameters": {
            "type": "object",
            "properties": {
                "automation_ref": string_property(
                    "Automation id or a clear automation name.",
                    min_length=1,
                ),
                "name": string_property("Optional new automation name.", min_length=1),
                "description": string_property(
                    "Optional new description.",
                    min_length=1,
                ),
                "schedule": string_property(
                    "Optional new time schedule type.",
                    enum=["once", "daily", "weekly"],
                ),
                "due_at": iso8601_datetime_property(
                    "Absolute ISO 8601 local datetime with timezone offset for once schedules."
                ),
                "time_of_day": time_of_day_property(
                    "Local time in HH:MM for daily or weekly schedules."
                ),
                "weekdays": weekdays_property(
                    "Weekday numbers for weekly schedules, where Monday is 0 and Sunday is 6."
                ),
                "delivery": string_property(
                    "Optional new delivery mode.",
                    enum=["spoken", "printed"],
                ),
                "content_mode": string_property(
                    "Optional new content mode. If omitted, keep the current mode.",
                    enum=["llm_prompt", "static_text"],
                ),
                "content": string_property(
                    "Optional new prompt or static text. If omitted, keep the current wording.",
                    min_length=1,
                ),
                "allow_web_search": boolean_property(
                    "Optional new live-search flag for llm_prompt content."
                ),
                "enabled": boolean_property("Optional enabled toggle."),
                "tags": array_property(
                    "Optional full replacement tag list.",
                    string_property("Tag.", min_length=1),
                    unique_items=True,
                ),
                "timezone_name": string_property(
                    "Optional new timezone name.",
                    min_length=1,
                ),
                "confirmed": boolean_property(
                    "Set true only after the user clearly confirmed this persistent automation change when extra confirmation is needed."
                ),
            },
            "allOf": update_time_schedule_rules(),
            "required": ["automation_ref"],
            "additionalProperties": False,
        },
    }


def build_update_sensor_automation_schema(context: SchemaBuildContext) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "update_sensor_automation",
        "description": (
            "Update an existing supported sensor-triggered automation. "
            "You may replace the trigger kind, hold_seconds, delivery, and content in one update. "
            "Resolve pir_no_motion as inactivity or no motion/presence, and use vad_quiet only for room-audio silence. "
            "Use list_automations first if you need to identify it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "automation_ref": string_property(
                    "Automation id or a clear automation name.",
                    min_length=1,
                ),
                "name": string_property("Optional new automation name.", min_length=1),
                "description": string_property(
                    "Optional new description.",
                    min_length=1,
                ),
                "trigger_kind": string_property(
                    context.sensor_trigger_kind_help,
                    enum=context.sensor_trigger_kinds,
                ),
                "hold_seconds": number_property(
                    "Optional hold duration before firing.",
                    minimum=0,
                ),
                "cooldown_seconds": number_property(
                    "Optional cooldown after the automation fired.",
                    minimum=0,
                ),
                "delivery": string_property(
                    "Optional new delivery mode.",
                    enum=["spoken", "printed"],
                ),
                "content_mode": string_property(
                    "Optional new content mode.",
                    enum=["llm_prompt", "static_text"],
                ),
                "content": string_property(
                    "Optional new prompt or static text.",
                    min_length=1,
                ),
                "allow_web_search": boolean_property(
                    "Optional new live-search flag for llm_prompt content."
                ),
                "enabled": boolean_property("Optional enabled toggle."),
                "tags": array_property(
                    "Optional full replacement tag list.",
                    string_property("Tag.", min_length=1),
                    unique_items=True,
                ),
                "confirmed": boolean_property(
                    "Set true only after the user clearly confirmed this persistent automation change when extra confirmation is needed."
                ),
            },
            "allOf": update_sensor_rules(),
            "required": ["automation_ref"],
            "additionalProperties": False,
        },
    }


def build_delete_automation_schema(_context: SchemaBuildContext) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "delete_automation",
        "description": "Delete an existing scheduled automation.",
        "parameters": {
            "type": "object",
            "properties": {
                "automation_ref": string_property(
                    "Automation id or a clear automation name.",
                    min_length=1,
                ),
                "confirmed": boolean_property(
                    "Set true only after the user clearly confirmed the deletion when extra confirmation is needed."
                ),
            },
            "required": ["automation_ref"],
            "additionalProperties": False,
        },
    }


TOOL_BUILDERS = (
    ("list_automations", build_list_automations_schema),
    ("create_time_automation", build_create_time_automation_schema),
    ("create_sensor_automation", build_create_sensor_automation_schema),
    ("update_time_automation", build_update_time_automation_schema),
    ("update_sensor_automation", build_update_sensor_automation_schema),
    ("delete_automation", build_delete_automation_schema),
)
