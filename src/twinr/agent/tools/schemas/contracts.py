"""Build canonical and adapted JSON schemas for Twinr agent tools.

Exports the public schema builders for the full tool surface plus the compact
and realtime-safe variants derived from it. Import the public builders from
``twinr.agent.tools.schemas`` or ``twinr.agent.tools`` rather than depending on
the private helper functions in this module.

.. note::
   Realtime-safe schemas intentionally strip unsupported top-level JSON Schema
   keys and rely on Twinr runtime validation for some cross-field constraints.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Sequence

from twinr.agent.base_agent.settings.simple_settings import (
    spoken_voice_options_context,
    supported_setting_names,
    supported_spoken_voices,
)
from twinr.automations import supported_sensor_trigger_kinds
from twinr.integrations.smarthome.models import (
    SmartHomeCommand,
    SmartHomeEntityAggregateField,
    SmartHomeEntityClass,
    SmartHomeEventAggregateField,
    SmartHomeEventKind,
)

_CANONICAL_ENGLISH_MEMORY_NOTE = (
    "All semantic text fields must be canonical English, even if the user spoke another language. "
    "Keep names, phone numbers, email addresses, IDs, codes, and exact quoted text verbatim."
)

# AUDIT-FIX(#2): Encode the documented HH:MM local-time contract as machine-readable validation
# instead of leaving it to free-form descriptions.
_LOCAL_TIME_HHMM_PATTERN = r"^(?:[01]\d|2[0-3]):[0-5]\d$"
_REALTIME_TOP_LEVEL_UNSUPPORTED_SCHEMA_KEYS: frozenset[str] = frozenset(
    {"allOf", "anyOf", "oneOf", "not", "enum", "if", "then", "else"}
)
_REALTIME_VALIDATION_NOTE = (
    "Realtime compatibility note: some cross-field validation rules are enforced by Twinr at tool execution time "
    "instead of the API schema. Follow the field descriptions exactly and only call this tool with complete, valid arguments."
)


# AUDIT-FIX(#1): Normalize accidental bare-string inputs and deduplicate tool names so callers do
# not silently lose every capability because Python treats str as Iterable[str].
def _normalize_tool_names(tool_names: Iterable[str] | str | bytes | bytearray | None) -> tuple[str, ...]:
    if tool_names is None:
        return ()

    raw_items: Iterable[Any]
    if isinstance(tool_names, str):
        raw_items = (tool_names,)
    elif isinstance(tool_names, (bytes, bytearray)):
        raw_items = (tool_names.decode("utf-8", errors="ignore"),)
    else:
        raw_items = tool_names

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_item in raw_items:
        if raw_item is None:
            continue
        if isinstance(raw_item, (bytes, bytearray)):
            item = raw_item.decode("utf-8", errors="ignore").strip()
        else:
            item = str(raw_item).strip()
        if not item or item in seen:
            continue
        normalized.append(item)
        seen.add(item)
    return tuple(normalized)


def _unique_strings(values: Iterable[Any]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        unique.append(item)
        seen.add(item)
    return sorted(unique, key=str.casefold)


def _string_property(
    description: str,
    *,
    min_length: int | None = None,
    enum: Sequence[str] | None = None,
    pattern: str | None = None,
    string_format: str | None = None,
) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "string", "description": description}
    if min_length is not None:
        schema["minLength"] = min_length
    if enum is not None:
        schema["enum"] = list(enum)
    if pattern is not None:
        schema["pattern"] = pattern
    if string_format is not None:
        schema["format"] = string_format
    return schema


def _number_property(
    description: str,
    *,
    minimum: int | float | None = None,
    maximum: int | float | None = None,
    integer: bool = False,
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "integer" if integer else "number",
        "description": description,
    }
    if minimum is not None:
        schema["minimum"] = minimum
    if maximum is not None:
        schema["maximum"] = maximum
    return schema


def _boolean_property(description: str) -> dict[str, Any]:
    return {"type": "boolean", "description": description}


def _array_property(
    description: str,
    items: dict[str, Any],
    *,
    min_items: int | None = None,
    max_items: int | None = None,
    unique_items: bool | None = None,
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "array",
        "description": description,
        "items": items,
    }
    if min_items is not None:
        schema["minItems"] = min_items
    if max_items is not None:
        schema["maxItems"] = max_items
    if unique_items is not None:
        schema["uniqueItems"] = unique_items
    return schema


# AUDIT-FIX(#2): Reuse strict date/time and weekday validators across reminder and automation
# tools so malformed schedules are rejected before any side effects happen.
def _iso8601_datetime_property(description: str) -> dict[str, Any]:
    return _string_property(description, string_format="date-time")


def _time_of_day_property(description: str) -> dict[str, Any]:
    return _string_property(description, pattern=_LOCAL_TIME_HHMM_PATTERN)


def _weekdays_property(description: str) -> dict[str, Any]:
    return _array_property(
        description,
        _number_property(
            "Weekday number where Monday is 0 and Sunday is 6.",
            minimum=0,
            maximum=6,
            integer=True,
        ),
        min_items=1,
        max_items=7,
        unique_items=True,
    )


# AUDIT-FIX(#2): Express schedule-dependent requirements directly in the schema so
# impossible automation payloads fail validation before execution.
def _create_time_schedule_rules() -> list[dict[str, Any]]:
    return [
        {
            "if": {"properties": {"schedule": {"const": "once"}}, "required": ["schedule"]},
            "then": {"required": ["due_at"]},
        },
        {
            "if": {"properties": {"schedule": {"const": "daily"}}, "required": ["schedule"]},
            "then": {"required": ["time_of_day"]},
        },
        {
            "if": {"properties": {"schedule": {"const": "weekly"}}, "required": ["schedule"]},
            "then": {"required": ["time_of_day", "weekdays"]},
        },
    ]


# AUDIT-FIX(#2): Prevent invalid schedule edits and no-op update calls from slipping
# through to the automation executor.
def _update_time_schedule_rules() -> list[dict[str, Any]]:
    return [
        {
            "if": {"properties": {"schedule": {"const": "once"}}, "required": ["schedule"]},
            "then": {"required": ["due_at"]},
        },
        {
            "if": {"properties": {"schedule": {"const": "daily"}}, "required": ["schedule"]},
            "then": {"required": ["time_of_day"]},
        },
        {
            "if": {"properties": {"schedule": {"const": "weekly"}}, "required": ["schedule"]},
            "then": {"required": ["time_of_day", "weekdays"]},
        },
        {
            "anyOf": [
                {"required": ["name"]},
                {"required": ["description"]},
                {"required": ["schedule"]},
                {"required": ["due_at"]},
                {"required": ["time_of_day"]},
                {"required": ["weekdays"]},
                {"required": ["delivery"]},
                {"required": ["content_mode"]},
                {"required": ["content"]},
                {"required": ["allow_web_search"]},
                {"required": ["enabled"]},
                {"required": ["tags"]},
                {"required": ["timezone_name"]},
            ]
        },
    ]


# AUDIT-FIX(#2): Require at least one mutable sensor field on update so a bare
# automation_ref does not produce a confusing no-op tool call.
def _update_sensor_rules() -> list[dict[str, Any]]:
    return [
        {
            "anyOf": [
                {"required": ["name"]},
                {"required": ["description"]},
                {"required": ["trigger_kind"]},
                {"required": ["hold_seconds"]},
                {"required": ["cooldown_seconds"]},
                {"required": ["delivery"]},
                {"required": ["content_mode"]},
                {"required": ["content"]},
                {"required": ["allow_web_search"]},
                {"required": ["enabled"]},
                {"required": ["tags"]},
            ]
        }
    ]


# AUDIT-FIX(#2): Constrain setting values by setting kind so the model cannot emit
# unsupported voices or out-of-range runtime settings.
def _simple_setting_rules(spoken_voices: Sequence[str]) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = [
        {
            "if": {"properties": {"action": {"const": "set"}}, "required": ["action"]},
            "then": {"required": ["value"]},
        },
        {
            "if": {"properties": {"setting": {"const": "memory_capacity"}}, "required": ["setting"]},
            "then": {"properties": {"value": {"type": "integer", "minimum": 1, "maximum": 4}}},
        },
        {
            "if": {"properties": {"setting": {"const": "speech_speed"}}, "required": ["setting"]},
            "then": {"properties": {"value": {"type": "number", "minimum": 0.75, "maximum": 1.15}}},
        },
        {
            "if": {"properties": {"setting": {"const": "speech_pause_ms"}}, "required": ["setting"]},
            "then": {"properties": {"value": {"type": "integer", "minimum": 0}}},
        },
        {
            "if": {"properties": {"setting": {"const": "follow_up_timeout_s"}}, "required": ["setting"]},
            "then": {"properties": {"value": {"type": "number", "minimum": 0}}},
        },
    ]
    if spoken_voices:
        rules.append(
            {
                "if": {"properties": {"setting": {"const": "spoken_voice"}}, "required": ["setting"]},
                "then": {
                    "properties": {
                        "action": {"const": "set"},
                        "value": {
                            "type": "string",
                            "minLength": 1,
                            "enum": list(spoken_voices),
                        },
                    },
                    "required": ["value"],
                },
            }
        )
    return rules


def build_agent_tool_schemas(tool_names: Iterable[str] | str | bytes | bytearray | None) -> list[dict[str, Any]]:
    """Build the canonical JSON schemas for the requested tool names.

    Args:
        tool_names: Tool names to include. Accepts iterables plus single string
            or bytes inputs and ignores duplicates and empty values.

    Returns:
        A list of OpenAI-style function tool schema dictionaries ordered by the
        canonical tool registry in this module.
    """
    normalized_tool_names = _normalize_tool_names(tool_names)
    available = set(normalized_tool_names)

    # AUDIT-FIX(#4): Snapshot dynamic capability providers once per build so every schema in the
    # returned list is internally consistent even if the provider is mutable or generator-backed.
    sensor_trigger_kinds = _unique_strings(supported_sensor_trigger_kinds())
    sensor_trigger_kind_help = (
        "Supported sensor trigger type. "
        "Use pir_no_motion when no motion or no presence/activity should be detected for hold_seconds. "
        "Use vad_quiet only when the room microphone should stay quiet for hold_seconds. "
        "Use camera_person_visible when a person should be visible in the camera view."
    )
    setting_names = _unique_strings(supported_setting_names())
    spoken_voices = _unique_strings(supported_spoken_voices())
    spoken_voice_catalog = str(spoken_voice_options_context()).strip()
    smart_home_entity_classes = _unique_strings(item.value for item in SmartHomeEntityClass)
    smart_home_commands = _unique_strings(item.value for item in SmartHomeCommand)
    smart_home_entity_aggregate_fields = _unique_strings(item.value for item in SmartHomeEntityAggregateField)
    smart_home_event_aggregate_fields = _unique_strings(item.value for item in SmartHomeEventAggregateField)
    smart_home_event_kinds = _unique_strings(item.value for item in SmartHomeEventKind)

    tools: list[dict[str, Any]] = []
    if "print_receipt" in available:
        tools.append(
            {
                "type": "function",
                "name": "print_receipt",
                "description": (
                    "Print short, user-facing content on the thermal receipt printer "
                    "when the user explicitly asks for a printout. "
                    "Use focus_hint to describe what from the recent context should be printed. "
                    "When the user asks for exact wording, quoted text, or a literal string to print, "
                    "you must pass that exact printable wording in text."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "focus_hint": _string_property(
                            "Short hint describing what from the recent conversation should be printed.",
                            min_length=1,
                        ),
                        "text": _string_property(
                            "Exact printable wording. Required when the user asked to print exact text, "
                            "quoted text, or a literal string.",
                            min_length=1,
                        ),
                    },
                    # AUDIT-FIX(#2): Reject empty print jobs by requiring either focus_hint or text.
                    "anyOf": [{"required": ["focus_hint"]}, {"required": ["text"]}],
                    "required": [],
                    "additionalProperties": False,
                },
            }
        )
    if "search_live_info" in available:
        tools.append(
            {
                "type": "function",
                "name": "search_live_info",
                "description": (
                    "Look up fresh or externally verifiable web information for the user. "
                    "Use this for broad web research, not only a fixed list of example domains. "
                    "Do not use it for the user's own smart-home inventory, room/device state, or recent in-home smart-home events; those belong to the smart-home tools."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": _string_property(
                            "The exact question to research on the web.",
                            min_length=1,
                        ),
                        "location_hint": _string_property(
                            "Optional location such as a city or district relevant to the search.",
                            min_length=1,
                        ),
                        "date_context": _string_property(
                            "Optional absolute date or time context if the user referred to relative dates.",
                            min_length=1,
                        ),
                    },
                    "required": ["question"],
                    "additionalProperties": False,
                },
            }
        )
    if "schedule_reminder" in available:
        tools.append(
            {
                "type": "function",
                "name": "schedule_reminder",
                "description": (
                    "Schedule a future reminder or timer when the user asks to be reminded later or to set a timer. "
                    "Always send due_at as an absolute ISO 8601 local datetime with timezone offset."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "due_at": _iso8601_datetime_property(
                            "Absolute local due time in ISO 8601 format, for example 2026-03-14T12:00:00+01:00."
                        ),
                        "summary": _string_property(
                            "Short summary of what Twinr should remind the user about.",
                            min_length=1,
                        ),
                        "details": _string_property(
                            "Optional extra detail to include when the reminder is spoken.",
                            min_length=1,
                        ),
                        "kind": _string_property(
                            "Short type such as reminder, timer, appointment, medication, task, or alarm.",
                            min_length=1,
                        ),
                        "original_request": _string_property(
                            "Optional short quote or paraphrase of the user's original reminder request.",
                            min_length=1,
                        ),
                    },
                    "required": ["due_at", "summary"],
                    "additionalProperties": False,
                },
            }
        )
    if "list_automations" in available:
        tools.append(
            {
                "type": "function",
                "name": "list_automations",
                "description": (
                    "List the currently configured time-based and sensor-triggered automations. "
                    "Use this for exact questions about current automations and before updating or deleting one when hidden context may be stale."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "include_disabled": _boolean_property(
                            "Set true if disabled automations should also be included."
                        )
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            }
        )
    if "create_time_automation" in available:
        tools.append(
            {
                "type": "function",
                "name": "create_time_automation",
                "description": (
                    "Create a time-based automation for one-off or recurring actions such as daily weather, "
                    "daily news, or printed headlines."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": _string_property(
                            "Short operator-friendly name for the automation.",
                            min_length=1,
                        ),
                        "description": _string_property(
                            "Optional short description of what the automation does.",
                            min_length=1,
                        ),
                        "schedule": _string_property(
                            "Time schedule type.",
                            enum=["once", "daily", "weekly"],
                        ),
                        "due_at": _iso8601_datetime_property(
                            "Absolute ISO 8601 local datetime with timezone offset for once schedules."
                        ),
                        "time_of_day": _time_of_day_property(
                            "Local time in HH:MM for daily or weekly schedules."
                        ),
                        "weekdays": _weekdays_property(
                            "Weekday numbers for weekly schedules, where Monday is 0 and Sunday is 6."
                        ),
                        "delivery": _string_property(
                            "Whether the automation should speak or print when it runs.",
                            enum=["spoken", "printed"],
                        ),
                        "content_mode": _string_property(
                            "Use llm_prompt for generated content or static_text for fixed wording.",
                            enum=["llm_prompt", "static_text"],
                        ),
                        "content": _string_property(
                            "The prompt or static text the automation should use.",
                            min_length=1,
                        ),
                        "allow_web_search": _boolean_property(
                            "Set true when the automation needs fresh live information from the web."
                        ),
                        "enabled": _boolean_property(
                            "Whether the automation should be active immediately."
                        ),
                        "tags": _array_property(
                            "Optional short tags for operator organization.",
                            _string_property("Tag.", min_length=1),
                            unique_items=True,
                        ),
                        "timezone_name": _string_property(
                            "Optional timezone name. Use the local Twinr timezone unless there is a clear reason not to.",
                            min_length=1,
                        ),
                        "confirmed": _boolean_property(
                            "Set true only after the user clearly confirmed this persistent automation change when extra confirmation is needed."
                        ),
                    },
                    "allOf": _create_time_schedule_rules(),
                    "required": ["name", "schedule", "delivery", "content_mode", "content"],
                    "additionalProperties": False,
                },
            }
        )
    if "create_sensor_automation" in available:
        tools.append(
            {
                "type": "function",
                "name": "create_sensor_automation",
                "description": (
                    "Create an automation triggered by PIR motion, camera visibility/object readings, "
                    "or background microphone/VAD state."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": _string_property(
                            "Short operator-friendly name for the automation.",
                            min_length=1,
                        ),
                        "description": _string_property(
                            "Optional short description of what the automation does.",
                            min_length=1,
                        ),
                        "trigger_kind": _string_property(
                            sensor_trigger_kind_help,
                            enum=sensor_trigger_kinds,
                        ),
                        "hold_seconds": _number_property(
                            "Optional required hold duration before firing. Required for quiet/no-motion triggers.",
                            minimum=0,
                        ),
                        "cooldown_seconds": _number_property(
                            "Optional cooldown after the automation fired.",
                            minimum=0,
                        ),
                        "delivery": _string_property(
                            "Whether the automation should speak or print when it runs.",
                            enum=["spoken", "printed"],
                        ),
                        "content_mode": _string_property(
                            "Use llm_prompt for generated content or static_text for fixed wording.",
                            enum=["llm_prompt", "static_text"],
                        ),
                        "content": _string_property(
                            "The prompt or static text the automation should use.",
                            min_length=1,
                        ),
                        "allow_web_search": _boolean_property(
                            "Set true when the automation needs fresh live information from the web."
                        ),
                        "enabled": _boolean_property(
                            "Whether the automation should be active immediately."
                        ),
                        "tags": _array_property(
                            "Optional short tags for operator organization.",
                            _string_property("Tag.", min_length=1),
                            unique_items=True,
                        ),
                        "confirmed": _boolean_property(
                            "Set true only after the user clearly confirmed this persistent automation change when extra confirmation is needed."
                        ),
                    },
                    "required": ["name", "trigger_kind", "delivery", "content_mode", "content"],
                    "additionalProperties": False,
                },
            }
        )
    if "update_time_automation" in available:
        tools.append(
            {
                "type": "function",
                "name": "update_time_automation",
                "description": "Update an existing time-based automation. Use list_automations first if you need to identify it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "automation_ref": _string_property(
                            "Automation id or a clear automation name.",
                            min_length=1,
                        ),
                        "name": _string_property("Optional new automation name.", min_length=1),
                        "description": _string_property(
                            "Optional new description.",
                            min_length=1,
                        ),
                        "schedule": _string_property(
                            "Optional new time schedule type.",
                            enum=["once", "daily", "weekly"],
                        ),
                        "due_at": _iso8601_datetime_property(
                            "Absolute ISO 8601 local datetime with timezone offset for once schedules."
                        ),
                        "time_of_day": _time_of_day_property(
                            "Local time in HH:MM for daily or weekly schedules."
                        ),
                        "weekdays": _weekdays_property(
                            "Weekday numbers for weekly schedules, where Monday is 0 and Sunday is 6."
                        ),
                        "delivery": _string_property(
                            "Optional new delivery mode.",
                            enum=["spoken", "printed"],
                        ),
                        "content_mode": _string_property(
                            "Optional new content mode. If omitted, keep the current mode.",
                            enum=["llm_prompt", "static_text"],
                        ),
                        "content": _string_property(
                            "Optional new prompt or static text. If omitted, keep the current wording.",
                            min_length=1,
                        ),
                        "allow_web_search": _boolean_property(
                            "Optional new live-search flag for llm_prompt content."
                        ),
                        "enabled": _boolean_property("Optional enabled toggle."),
                        "tags": _array_property(
                            "Optional full replacement tag list.",
                            _string_property("Tag.", min_length=1),
                            unique_items=True,
                        ),
                        "timezone_name": _string_property(
                            "Optional new timezone name.",
                            min_length=1,
                        ),
                        "confirmed": _boolean_property(
                            "Set true only after the user clearly confirmed this persistent automation change when extra confirmation is needed."
                        ),
                    },
                    "allOf": _update_time_schedule_rules(),
                    "required": ["automation_ref"],
                    "additionalProperties": False,
                },
            }
        )
    if "update_sensor_automation" in available:
        tools.append(
            {
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
                        "automation_ref": _string_property(
                            "Automation id or a clear automation name.",
                            min_length=1,
                        ),
                        "name": _string_property("Optional new automation name.", min_length=1),
                        "description": _string_property(
                            "Optional new description.",
                            min_length=1,
                        ),
                        "trigger_kind": _string_property(
                            sensor_trigger_kind_help,
                            enum=sensor_trigger_kinds,
                        ),
                        "hold_seconds": _number_property(
                            "Optional hold duration before firing.",
                            minimum=0,
                        ),
                        "cooldown_seconds": _number_property(
                            "Optional cooldown after the automation fired.",
                            minimum=0,
                        ),
                        "delivery": _string_property(
                            "Optional new delivery mode.",
                            enum=["spoken", "printed"],
                        ),
                        "content_mode": _string_property(
                            "Optional new content mode.",
                            enum=["llm_prompt", "static_text"],
                        ),
                        "content": _string_property(
                            "Optional new prompt or static text.",
                            min_length=1,
                        ),
                        "allow_web_search": _boolean_property(
                            "Optional new live-search flag for llm_prompt content."
                        ),
                        "enabled": _boolean_property("Optional enabled toggle."),
                        "tags": _array_property(
                            "Optional full replacement tag list.",
                            _string_property("Tag.", min_length=1),
                            unique_items=True,
                        ),
                        "confirmed": _boolean_property(
                            "Set true only after the user clearly confirmed this persistent automation change when extra confirmation is needed."
                        ),
                    },
                    "allOf": _update_sensor_rules(),
                    "required": ["automation_ref"],
                    "additionalProperties": False,
                },
            }
        )
    if "delete_automation" in available:
        tools.append(
            {
                "type": "function",
                "name": "delete_automation",
                "description": "Delete an existing scheduled automation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "automation_ref": _string_property(
                            "Automation id or a clear automation name.",
                            min_length=1,
                        ),
                        "confirmed": _boolean_property(
                            "Set true only after the user clearly confirmed the deletion when extra confirmation is needed."
                        ),
                    },
                    "required": ["automation_ref"],
                    "additionalProperties": False,
                },
            }
        )
    if "list_smart_home_entities" in available:
        tools.append(
            {
                "type": "function",
                "name": "list_smart_home_entities",
                "description": (
                    "List bounded smart-home entities such as lights, scenes, motion sensors, and device-health endpoints. "
                    "This tool supports generic selectors, exact scalar state filters, pagination, and simple aggregations, so it can answer both exact device discovery questions and broader house-status queries without a special-case summary tool. "
                    "Use this instead of web search when the user asks about the devices, areas, grouped counts, or filtered current state in their own smart home. "
                    "Broad live status answers will often call this tool more than once with different selectors or aggregations, for example lights that are on, offline devices, grouped counts by area, or alarm/device-health entities. "
                    "Avoid using one truncated catch-all listing as the whole house status when narrower filters or grouped counts are available. "
                    "When a later exact state read is needed, use the returned entity_id values verbatim."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_ids": _array_property(
                            "Optional exact entity IDs to limit the listing to a known subset.",
                            _string_property("Exact smart-home entity ID.", min_length=1),
                            min_items=1,
                            max_items=8,
                            unique_items=True,
                        ),
                        "entity_class": _string_property(
                            "Optional legacy single entity-class filter such as light, scene, motion_sensor, or device_health.",
                            enum=smart_home_entity_classes,
                        ),
                        "entity_classes": _array_property(
                            "Optional entity-class filter set. Prefer this over entity_class when more than one class is relevant.",
                            _string_property(
                                "Smart-home entity class.",
                                enum=smart_home_entity_classes,
                            ),
                            min_items=1,
                            max_items=8,
                            unique_items=True,
                        ),
                        "areas": _array_property(
                            "Optional exact area labels to keep, such as Wohnzimmer or Flur.",
                            _string_property("Exact smart-home area label.", min_length=1),
                            min_items=1,
                            max_items=8,
                            unique_items=True,
                        ),
                        "providers": _array_property(
                            "Optional provider IDs to keep, such as hue.",
                            _string_property("Exact smart-home provider ID.", min_length=1),
                            min_items=1,
                            max_items=8,
                            unique_items=True,
                        ),
                        "online": _boolean_property(
                            "Optional online-status filter. Set false to focus on offline or unavailable entities."
                        ),
                        "controllable": _boolean_property(
                            "Optional controllability filter."
                        ),
                        "readable": _boolean_property(
                            "Optional readability filter."
                        ),
                        "include_unavailable": _boolean_property(
                            "Set true to include currently unavailable or offline entities."
                        ),
                        "state_filters": _array_property(
                            "Optional exact scalar state filters. Use keys such as on, brightness, motion, or nested dotted paths when the provider exposes nested state.",
                            {
                                "type": "object",
                                "properties": {
                                    "key": _string_property(
                                        "Exact state key or dotted state path to inspect.",
                                        min_length=1,
                                    ),
                                    "value": {
                                        "anyOf": [
                                            {"type": "string", "minLength": 1},
                                            {"type": "number"},
                                            {"type": "boolean"},
                                        ],
                                        "description": "Exact scalar value that the selected state key must equal.",
                                    },
                                },
                                "required": ["key", "value"],
                                "additionalProperties": False,
                            },
                            min_items=1,
                            max_items=8,
                        ),
                        "aggregate_by": _array_property(
                            "Optional entity fields to aggregate counts by. Prefer this first for broader house-status scans in larger homes before narrowing to exact entity names.",
                            _string_property(
                                "Aggregate field.",
                                enum=smart_home_entity_aggregate_fields,
                            ),
                            min_items=1,
                            max_items=6,
                            unique_items=True,
                        ),
                        "limit": _number_property(
                            "Maximum number of matching entities to return after filtering.",
                            minimum=1,
                            maximum=32,
                            integer=True,
                        ),
                        "cursor": _string_property(
                            "Optional opaque offset cursor from the previous entity list response.",
                            min_length=1,
                        ),
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            }
        )
    if "read_smart_home_state" in available:
        tools.append(
            {
                "type": "function",
                "name": "read_smart_home_state",
                "description": (
                    "Read the current state for one or more exact smart-home entities after they are already known. "
                    "Use this for precise questions like whether a light is on, how bright it is, or whether a motion sensor recently fired. "
                    "entity_ids must be exact routed identifiers copied verbatim from smart-home tool results or directly from the user's exact routed IDs, not IDs invented from labels or classes."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_ids": _array_property(
                            "One or more exact smart-home entity IDs to inspect.",
                            _string_property("Exact smart-home entity ID.", min_length=1),
                            min_items=1,
                            max_items=8,
                            unique_items=True,
                        ),
                        "include_unavailable": _boolean_property(
                            "Set true to include entities even if they are currently unavailable."
                        ),
                    },
                    "required": ["entity_ids"],
                    "additionalProperties": False,
                },
            }
        )
    if "control_smart_home_entities" in available:
        tools.append(
            {
                "type": "function",
                "name": "control_smart_home_entities",
                "description": (
                    "Control allowed low-risk smart-home targets such as lights, grouped lights, switches, and scenes. "
                    "Use this only for explicit user requests to change device state."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_ids": _array_property(
                            "One or more exact smart-home entity IDs to control.",
                            _string_property("Exact smart-home entity ID.", min_length=1),
                            min_items=1,
                            max_items=8,
                            unique_items=True,
                        ),
                        "command": _string_property(
                            "Generic control command to execute.",
                            enum=smart_home_commands,
                        ),
                        "brightness": _number_property(
                            "Brightness percentage from 0 to 100. Required only for set_brightness.",
                            minimum=0,
                            maximum=100,
                        ),
                        "confirmed": _boolean_property(
                            "Set true only after the user clearly confirmed this control action when explicit confirmation is needed."
                        ),
                    },
                    "required": ["entity_ids", "command"],
                    "additionalProperties": False,
                },
            }
        )
    if "read_smart_home_sensor_stream" in available:
        tools.append(
            {
                "type": "function",
                "name": "read_smart_home_sensor_stream",
                "description": (
                    "Read a bounded batch of recent normalized smart-home events such as motion detections, button presses, connectivity changes, or alarm state changes. "
                    "Use this for explicit inspection or debugging of the current stream. "
                    "Use this instead of web search when the user asks about recent in-home smart-home activity. "
                    "It supports generic event selectors and simple aggregations instead of a hardcoded status-summary path. "
                    "For broader live house-status answers, pair this with list_smart_home_entities instead of treating the recent event batch as the whole current state."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_ids": _array_property(
                            "Optional exact smart-home entity IDs to keep in the event stream.",
                            _string_property("Exact smart-home entity ID.", min_length=1),
                            min_items=1,
                            max_items=8,
                            unique_items=True,
                        ),
                        "event_kinds": _array_property(
                            "Optional event-kind filters such as motion_detected, device_offline, or alarm_triggered.",
                            _string_property(
                                "Normalized smart-home event kind.",
                                enum=smart_home_event_kinds,
                            ),
                            min_items=1,
                            max_items=8,
                            unique_items=True,
                        ),
                        "areas": _array_property(
                            "Optional exact area labels to keep in the event stream.",
                            _string_property("Exact smart-home area label.", min_length=1),
                            min_items=1,
                            max_items=8,
                            unique_items=True,
                        ),
                        "providers": _array_property(
                            "Optional provider IDs to keep in the event stream.",
                            _string_property("Exact smart-home provider ID.", min_length=1),
                            min_items=1,
                            max_items=8,
                            unique_items=True,
                        ),
                        "aggregate_by": _array_property(
                            "Optional event fields to aggregate counts by.",
                            _string_property(
                                "Aggregate field.",
                                enum=smart_home_event_aggregate_fields,
                            ),
                            min_items=1,
                            max_items=4,
                            unique_items=True,
                        ),
                        "cursor": _string_property(
                            "Optional provider cursor from the previous stream read when available.",
                            min_length=1,
                        ),
                        "limit": _number_property(
                            "Maximum number of stream events to return.",
                            minimum=1,
                            maximum=20,
                            integer=True,
                        ),
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            }
        )
    if "propose_skill_learning" in available:
        tools.append(
            {
                "type": "function",
                "name": "propose_skill_learning",
                "description": (
                    "Start Twinr's self-coding learning flow for a new persistent capability the current tool surface cannot already satisfy. "
                    "Use this only for new repeatable behaviors, not for one-off answers."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": _string_property(
                            "Short human-readable name for the skill Twinr should learn.",
                            min_length=1,
                        ),
                        "action": _string_property(
                            "Plain-language summary of what the learned skill should do.",
                            min_length=1,
                        ),
                        "request_summary": _string_property(
                            "Optional short paraphrase of the user's request in user-facing language.",
                            min_length=1,
                        ),
                        "skill_id": _string_property(
                            "Optional stable identifier if one is already known.",
                            min_length=1,
                        ),
                        "trigger_mode": _string_property(
                            "Optional preliminary trigger mode.",
                            enum=["push", "pull"],
                        ),
                        "trigger_conditions": _array_property(
                            "Optional preliminary trigger conditions as stable identifiers.",
                            _string_property("Trigger condition identifier.", min_length=1),
                            unique_items=True,
                        ),
                        "scope": {
                            "type": "object",
                            "description": "Optional preliminary structured scope object.",
                            "additionalProperties": True,
                        },
                        "constraints": _array_property(
                            "Optional preliminary constraints in plain language.",
                            _string_property("Constraint.", min_length=1),
                            unique_items=True,
                        ),
                        "capabilities": _array_property(
                            "Required ASE capabilities such as camera, pir, speaker, llm_call, memory, scheduler, rules, safety, email, or calendar.",
                            _string_property("Capability identifier.", min_length=1),
                            min_items=1,
                            unique_items=True,
                        ),
                    },
                    "required": ["name", "action", "capabilities"],
                    "additionalProperties": False,
                },
            }
        )
    if "answer_skill_question" in available:
        tools.append(
            {
                "type": "function",
                "name": "answer_skill_question",
                "description": (
                    "Continue an active self-coding requirements dialogue after Twinr has already asked one of its short follow-up questions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": _string_property(
                            "Dialogue session identifier previously returned by propose_skill_learning.",
                            min_length=1,
                        ),
                        "use_default": _boolean_property(
                            "Set true when the user explicitly says to do whatever makes sense or to use the default."
                        ),
                        "trigger_mode": _string_property(
                            "Optional updated trigger mode for the current answer.",
                            enum=["push", "pull"],
                        ),
                        "trigger_conditions": _array_property(
                            "Optional extra trigger conditions to merge into the draft skill.",
                            _string_property("Trigger condition identifier.", min_length=1),
                            unique_items=True,
                        ),
                        "scope": {
                            "type": "object",
                            "description": "Optional shallow scope patch to merge into the draft skill scope.",
                            "additionalProperties": True,
                        },
                        "constraints": _array_property(
                            "Optional extra constraints to merge into the draft skill.",
                            _string_property("Constraint.", min_length=1),
                            unique_items=True,
                        ),
                        "action": _string_property(
                            "Optional refined action wording for how the skill should behave.",
                            min_length=1,
                        ),
                        "answer_summary": _string_property(
                            "Optional short summary of the user's answer for auditability.",
                            min_length=1,
                        ),
                        "confirmed": _boolean_property(
                            "Set true or false only when Twinr is at the final confirmation step."
                        ),
                    },
                    "required": ["session_id"],
                    "additionalProperties": False,
                },
            }
        )
    if "confirm_skill_activation" in available:
        tools.append(
            {
                "type": "function",
                "name": "confirm_skill_activation",
                "description": (
                    "Enable a compiled self-coding skill only after the user explicitly agrees that the soft-launch version should be turned on."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "job_id": _string_property(
                            "Compile job identifier that is already soft-launch ready.",
                            min_length=1,
                        ),
                        "confirmed": _boolean_property(
                            "Set true only after the user clearly approved enabling this learned behavior."
                        ),
                    },
                    "required": ["job_id", "confirmed"],
                    "additionalProperties": False,
                },
            }
        )
    if "rollback_skill_activation" in available:
        tools.append(
            {
                "type": "function",
                "name": "rollback_skill_activation",
                "description": (
                    "Roll a learned self-coding skill back to an earlier version when the user wants the new behavior undone."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_id": _string_property(
                            "Stable identifier of the learned skill family to roll back.",
                            min_length=1,
                        ),
                        "target_version": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Optional explicit older version to restore. If omitted, restore the newest earlier stable version.",
                        },
                    },
                    "required": ["skill_id"],
                    "additionalProperties": False,
                },
            }
        )
    if "pause_skill_activation" in available:
        tools.append(
            {
                "type": "function",
                "name": "pause_skill_activation",
                "description": (
                    "Pause one active learned self-coding skill version when the user or operator wants it temporarily disabled."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_id": _string_property(
                            "Stable identifier of the learned skill family to pause.",
                            min_length=1,
                        ),
                        "version": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Active learned skill version to pause.",
                        },
                        "reason": _string_property(
                            "Optional short pause reason such as operator_pause.",
                            min_length=1,
                        ),
                    },
                    "required": ["skill_id", "version"],
                    "additionalProperties": False,
                },
            }
        )
    if "reactivate_skill_activation" in available:
        tools.append(
            {
                "type": "function",
                "name": "reactivate_skill_activation",
                "description": (
                    "Re-enable one paused learned self-coding skill version after the user or operator wants it active again."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_id": _string_property(
                            "Stable identifier of the learned skill family to reactivate.",
                            min_length=1,
                        ),
                        "version": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Paused learned skill version to reactivate.",
                        },
                    },
                    "required": ["skill_id", "version"],
                    "additionalProperties": False,
                },
            }
        )
    if "remember_memory" in available:
        tools.append(
            {
                "type": "function",
                "name": "remember_memory",
                "description": (
                    "Store an important memory for future turns when the user explicitly asks you to remember something. "
                    "Use only for clear remember/save requests, not for ordinary conversation. "
                    f"{_CANONICAL_ENGLISH_MEMORY_NOTE}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kind": _string_property(
                            "Short type such as appointment, contact, reminder, preference, fact, or task.",
                            min_length=1,
                        ),
                        "summary": _string_property(
                            "Short factual summary of what should be remembered.",
                            min_length=1,
                        ),
                        "details": _string_property(
                            "Optional extra detail that helps later recall.",
                            min_length=1,
                        ),
                        "confirmed": _boolean_property(
                            "Set true only after the user clearly confirmed the persistent save when extra confirmation is needed."
                        ),
                    },
                    "required": ["summary"],
                    "additionalProperties": False,
                },
            }
        )
    if "remember_contact" in available:
        tools.append(
            {
                "type": "function",
                "name": "remember_contact",
                "description": (
                    "Store or refine a remembered contact in Twinr's structured graph memory. "
                    "Use this when the user explicitly wants Twinr to remember a person with a phone number, email, relation, or role. "
                    f"{_CANONICAL_ENGLISH_MEMORY_NOTE}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "given_name": _string_property(
                            "First name or main short name of the contact.",
                            min_length=1,
                        ),
                        "family_name": _string_property(
                            "Optional family name if known.",
                            min_length=1,
                        ),
                        "phone": _string_property(
                            "Optional phone number if the user gave one.",
                            min_length=1,
                        ),
                        "email": _string_property(
                            "Optional email if the user gave one.",
                            min_length=1,
                            string_format="email",
                        ),
                        "role": _string_property(
                            "Optional role such as physiotherapist, daughter, neighbor, or friend.",
                            min_length=1,
                        ),
                        "relation": _string_property(
                            "Optional relationship wording such as daughter, family, or helper.",
                            min_length=1,
                        ),
                        "notes": _string_property(
                            "Optional short detail that helps future disambiguation.",
                            min_length=1,
                        ),
                        "confirmed": _boolean_property(
                            "Set true only after the user clearly confirmed the persistent save when extra confirmation is needed."
                        ),
                    },
                    "required": ["given_name"],
                    "additionalProperties": False,
                },
            }
        )
    if "lookup_contact" in available:
        tools.append(
            {
                "type": "function",
                "name": "lookup_contact",
                "description": (
                    "Look up a remembered contact and return the stored phone number or email. "
                    "Use this for exact contact details instead of relying on hidden memory context, and ask for clarification when multiple matches exist."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": _string_property(
                            "Name or short name of the contact to look up.",
                            min_length=1,
                        ),
                        "family_name": _string_property(
                            "Optional family name if the user gave one.",
                            min_length=1,
                        ),
                        "role": _string_property(
                            "Optional role such as physiotherapist, daughter, or neighbor.",
                            min_length=1,
                        ),
                    },
                    "required": ["name"],
                    "additionalProperties": False,
                },
            }
        )
    if "get_memory_conflicts" in available:
        tools.append(
            {
                "type": "function",
                "name": "get_memory_conflicts",
                "description": (
                    "Inspect open long-term memory conflicts when the user asks what Twinr is unsure about, "
                    "or when you need the current conflict option IDs before resolving one. "
                    "Use this tool for exact conflict inspection instead of answering from hidden conflict summaries."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query_text": _string_property(
                            "Optional short query describing the current topic, such as Corinna, physiotherapist, "
                            "phone number, spouse, or coffee brand.",
                            min_length=1,
                        ),
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            }
        )
    if "resolve_memory_conflict" in available:
        tools.append(
            {
                "type": "function",
                "name": "resolve_memory_conflict",
                "description": (
                    "Resolve one open long-term memory conflict after the user clearly identified which stored option is correct. "
                    "Use the slot_key and selected_memory_id from conflict context or from get_memory_conflicts."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "slot_key": _string_property(
                            "The conflict slot key to resolve, for example contact:person:corinna_maier:phone.",
                            min_length=1,
                        ),
                        "selected_memory_id": _string_property(
                            "The chosen memory_id from the conflict options that should become active.",
                            min_length=1,
                        ),
                        "confirmed": _boolean_property(
                            "Set true only after the user clearly confirmed the persistent memory correction when extra confirmation is needed."
                        ),
                    },
                    "required": ["slot_key", "selected_memory_id"],
                    "additionalProperties": False,
                },
            }
        )
    if "remember_preference" in available:
        tools.append(
            {
                "type": "function",
                "name": "remember_preference",
                "description": (
                    "Store a stable personal preference in Twinr's structured graph memory, such as a preferred brand, favorite shop, or disliked food. "
                    f"{_CANONICAL_ENGLISH_MEMORY_NOTE}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": _string_property(
                            "Short category such as brand, store, food, drink, activity, music, or thing.",
                            min_length=1,
                        ),
                        "value": _string_property(
                            "The preferred or disliked thing, place, or brand.",
                            min_length=1,
                        ),
                        "for_product": _string_property(
                            "Optional product context, for example coffee.",
                            min_length=1,
                        ),
                        "sentiment": _string_property(
                            "Use prefer, like, dislike, or usually_buy_at.",
                            min_length=1,
                        ),
                        "details": _string_property(
                            "Optional short detail for later recall.",
                            min_length=1,
                        ),
                        "confirmed": _boolean_property(
                            "Set true only after the user clearly confirmed the persistent save when extra confirmation is needed."
                        ),
                    },
                    "required": ["category", "value"],
                    "additionalProperties": False,
                },
            }
        )
    if "remember_plan" in available:
        tools.append(
            {
                "type": "function",
                "name": "remember_plan",
                "description": (
                    "Store a short future intention or plan in Twinr's structured graph memory, such as wanting to go for a walk today. "
                    f"{_CANONICAL_ENGLISH_MEMORY_NOTE}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": _string_property("Short plan summary.", min_length=1),
                        "when": _string_property(
                            "Optional time wording such as today, tomorrow, 2026-03-14, or next Monday.",
                            min_length=1,
                        ),
                        "details": _string_property(
                            "Optional short detail for later recall.",
                            min_length=1,
                        ),
                        "confirmed": _boolean_property(
                            "Set true only after the user clearly confirmed the persistent save when extra confirmation is needed."
                        ),
                    },
                    "required": ["summary"],
                    "additionalProperties": False,
                },
            }
        )
    if "update_user_profile" in available:
        tools.append(
            {
                "type": "function",
                "name": "update_user_profile",
                "description": (
                    "Update stable user profile or preference context for future turns when the user explicitly asks you to remember it. "
                    f"{_CANONICAL_ENGLISH_MEMORY_NOTE}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": _string_property(
                            "Short category such as preferred_name, location, preference, contact, or routine.",
                            min_length=1,
                        ),
                        "instruction": _string_property(
                            "Short, durable instruction or fact to store in the user profile.",
                            min_length=1,
                        ),
                        "confirmed": _boolean_property(
                            "Set true only after the user clearly confirmed this persistent profile change when extra confirmation is needed."
                        ),
                    },
                    "required": ["category", "instruction"],
                    "additionalProperties": False,
                },
            }
        )
    if "update_personality" in available:
        tools.append(
            {
                "type": "function",
                "name": "update_personality",
                "description": (
                    "Update how Twinr should speak or behave in future turns when the user explicitly asks for a behavior change. "
                    f"{_CANONICAL_ENGLISH_MEMORY_NOTE}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": _string_property(
                            "Short category such as response_style, humor, language, verbosity, or greeting_style.",
                            min_length=1,
                        ),
                        "instruction": _string_property(
                            "Short future-behavior instruction to store in Twinr personality context.",
                            min_length=1,
                        ),
                        "confirmed": _boolean_property(
                            "Set true only after the user clearly confirmed this persistent behavior change when extra confirmation is needed."
                        ),
                    },
                    "required": ["category", "instruction"],
                    "additionalProperties": False,
                },
            }
        )
    if "manage_user_discovery" in available:
        tools.append(
            {
                "type": "function",
                "name": "manage_user_discovery",
                "description": (
                    "Manage Twinr's guided get-to-know-you flow across the initial setup and later short lifelong-learning follow-ups. "
                    "Use this when the user wants to start or continue the setup, offers to tell Twinr something about themselves, freely volunteers stable profile details that should enter the bounded discovery flow, answers an active get-to-know-you question, asks to pause or skip a topic, or says not now to a visible discovery invitation. "
                    "When the user answered, include compact learned_facts in canonical English as durable summaries, not raw transcript quotes, and use one learned_fact or memory_route per distinct learned detail. "
                    "Direct first-person profile statements, wish-form self-statements about preferred name or form of address, or direct profile corrections from an identified speaker already count as approval for discovery saves or mutations. "
                    "If the user explicitly corrects or deletes a previously learned detail, use review_profile and then replace_fact or delete_fact in the same turn when needed, even if discovery setup is still active."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": _string_property(
                            "Discovery action.",
                            enum=[
                                "start_or_resume",
                                "answer",
                                "skip_topic",
                                "pause_session",
                                "snooze",
                                "status",
                                "review_profile",
                                "replace_fact",
                                "delete_fact",
                            ],
                        ),
                        "topic_id": _string_property(
                            "Optional predefined topic such as basics, companion_style, social, interests, hobbies, routines, pets, no_goes, or health.",
                            min_length=1,
                        ),
                        "fact_id": _string_property(
                            "Required for replace_fact and delete_fact. Use the exact fact_id from a prior review_profile result and omit it for other actions.",
                            min_length=1,
                        ),
                        "learned_facts": _array_property(
                            "Optional compact durable facts learned from the user's answer. Use canonical English, not raw transcript quotes.",
                            {
                                "type": "object",
                                "properties": {
                                    "storage": _string_property(
                                        "Use user_profile for stable personal facts and personality for how Twinr should address or behave toward the user.",
                                        enum=["user_profile", "personality"],
                                    ),
                                    "text": _string_property(
                                        "Short durable fact or behavior preference in canonical English.",
                                        min_length=1,
                                    ),
                                },
                                "required": ["storage", "text"],
                                "additionalProperties": False,
                            },
                            max_items=8,
                        ),
                        "memory_routes": _array_property(
                            "Optional structured durable routes learned from the user's answer or used as replacements. Use canonical English semantic text, create one route per distinct learned detail, and only fill fields relevant to that route_kind.",
                            {
                                "type": "object",
                                "properties": {
                                    "route_kind": _string_property(
                                        "Structured discovery route kind.",
                                        enum=["user_profile", "personality", "contact", "preference", "plan", "durable_memory"],
                                    ),
                                    "text": _string_property("Managed-context fact text.", min_length=1),
                                    "category": _string_property("Preference category.", min_length=1),
                                    "given_name": _string_property("Contact given name.", min_length=1),
                                    "family_name": _string_property("Contact family name.", min_length=1),
                                    "phone": _string_property("Contact phone number.", min_length=1),
                                    "email": _string_property("Contact email address.", min_length=1),
                                    "role": _string_property("Contact role.", min_length=1),
                                    "relation": _string_property("Contact relation to the user.", min_length=1),
                                    "notes": _string_property("Short contact notes.", min_length=1),
                                    "value": _string_property("Preference value.", min_length=1),
                                    "sentiment": _string_property("Preference sentiment.", enum=["prefer", "like", "dislike", "avoid"]),
                                    "for_product": _string_property("Optional preference scope or product.", min_length=1),
                                    "summary": _string_property("Plan or durable-memory summary.", min_length=1),
                                    "when_text": _string_property("Optional natural-language timing text for a plan.", min_length=1),
                                    "details": _string_property("Optional additional details.", min_length=1),
                                    "kind": _string_property("Durable-memory kind label.", min_length=1),
                                },
                                "required": ["route_kind"],
                                "additionalProperties": False,
                            },
                            max_items=8,
                        ),
                        "topic_complete": _boolean_property(
                            "Set true only when the current topic is sufficiently covered for now and Twinr may move on or wrap up."
                        ),
                        "permission_granted": _boolean_property(
                            "For sensitive topics such as health, set true only after the user clearly agreed to continue on that topic, or false when the user declined."
                        ),
                        "snooze_days": _number_property(
                            "Optional whole-number snooze length in days for not-now responses.",
                            minimum=1,
                            maximum=14,
                            integer=True,
                        ),
                        "confirmed": _boolean_property(
                            "Set true when the user already gave the needed approval for this persistent save, including direct first-person stable profile statements, preferred-name or address-preference self-statements, or direct corrections from an identified speaker; leave false only when extra speaker confirmation is still required."
                        ),
                    },
                    "required": ["action"],
                    "additionalProperties": False,
                },
            }
        )
    if "configure_world_intelligence" in available:
        tools.append(
            {
                "type": "function",
                "name": "configure_world_intelligence",
                "description": (
                    "Manage Twinr's ongoing RSS or Atom sources for calm place and world awareness. "
                    "Use this to list, subscribe, discover, deactivate, or force-refresh feed subscriptions. "
                    "Do not use it for ordinary one-off live questions; use it only for installer setup, explicit source changes, "
                    "or occasional recalibration of Twinr's ongoing world-intelligence sources."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": _string_property(
                            "Which world-intelligence action to run.",
                            enum=["list", "subscribe", "discover", "deactivate", "refresh_now"],
                        ),
                        "query": _string_property(
                            "Optional discovery query that asks the live web backend to find source pages exposing RSS or Atom feeds.",
                            min_length=1,
                        ),
                        "label": _string_property(
                            "Optional short label such as Hamburg local politics or Germany energy policy.",
                            min_length=1,
                        ),
                        "location_hint": _string_property(
                            "Optional city, district, or region that should guide feed discovery or labeling.",
                            min_length=1,
                        ),
                        "region": _string_property(
                            "Optional region name that should be stored with the subscription and later world signals.",
                            min_length=1,
                        ),
                        "topics": _array_property(
                            "Optional recurring topics this subscription should cover.",
                            _string_property("Topic label.", min_length=1),
                            unique_items=True,
                        ),
                        "feed_urls": _array_property(
                            "Optional explicit RSS or Atom feed URLs to subscribe or deactivate.",
                            _string_property("Feed URL.", min_length=1),
                            unique_items=True,
                        ),
                        "subscription_refs": _array_property(
                            "Optional subscription ids to deactivate.",
                            _string_property("Subscription id.", min_length=1),
                            unique_items=True,
                        ),
                        "scope": _string_property(
                            "Optional world-awareness scope for new subscriptions.",
                            enum=["local", "regional", "national", "global", "topic"],
                        ),
                        "priority": _number_property(
                            "Optional salience/priority weight for the subscription between 0 and 1.",
                            minimum=0.0,
                            maximum=1.0,
                        ),
                        "refresh_interval_hours": _number_property(
                            "Optional refresh cadence in hours. Keep this calm and infrequent; not less than 24.",
                            minimum=24.0,
                        ),
                        "auto_subscribe": _boolean_property(
                            "For discover, set true if discovered feed URLs should be persisted immediately."
                        ),
                        "refresh_after_change": _boolean_property(
                            "Set true if Twinr should refresh the subscribed feeds immediately after the change."
                        ),
                        "confirmed": _boolean_property(
                            "Set true only after the user clearly confirmed this persistent source change when extra confirmation is needed."
                        ),
                    },
                    "required": ["action"],
                    "additionalProperties": False,
                },
            }
        )
    if "update_simple_setting" in available:
        tools.append(
            {
                "type": "function",
                "name": "update_simple_setting",
                "description": (
                    "Adjust one of Twinr's small bounded runtime settings after an explicit user request. "
                    "Use memory_capacity when the user wants Twinr to remember more or less recent conversation. "
                    "Use spoken_voice when the user wants a different voice and resolve descriptive requests to a supported Twinr voice name before calling the tool. "
                    "Use speech_speed when the user wants Twinr to speak slower or faster."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "setting": _string_property(
                            "Supported setting. memory_capacity changes how much recent conversation Twinr keeps. "
                            f"spoken_voice changes the spoken voice and must be set to one of: {', '.join(spoken_voices)}. "
                            "speech_speed changes the overall speaking speed for both normal TTS and realtime speech. "
                            "speech_pause_ms changes how long Twinr waits for a short pause before stopping recording. "
                            "follow_up_timeout_s changes how long the hands-free follow-up listening window stays open.",
                            enum=setting_names,
                        ),
                        "action": _string_property(
                            "Use increase/decrease for relative requests and set when the user gave a concrete value.",
                            enum=["increase", "decrease", "set"],
                        ),
                        "value": {
                            "anyOf": [
                                {"type": "number"},
                                {"type": "string", "minLength": 1},
                            ],
                            "description": (
                                "Optional concrete value for action=set. "
                                "For memory_capacity use levels 1 to 4. "
                                f"For spoken_voice pass one supported voice name from this catalog: {spoken_voice_catalog}. "
                                "Do not pass a free-form description. "
                                "For speech_speed use a factor between 0.75 and 1.15. "
                                "For speech_pause_ms use milliseconds. "
                                "For follow_up_timeout_s use seconds."
                            ),
                        },
                        "confirmed": _boolean_property(
                            "Set true only after the user clearly confirmed this persistent setting change when extra confirmation is needed."
                        ),
                    },
                    "allOf": _simple_setting_rules(spoken_voices),
                    "required": ["setting", "action"],
                    "additionalProperties": False,
                },
            }
        )
    if "enroll_voice_profile" in available:
        tools.append(
            {
                "type": "function",
                "name": "enroll_voice_profile",
                "description": (
                    "Create or refresh the local Twinr voice profile from the current spoken turn. "
                    "Use only when the user explicitly asks Twinr to learn or update their voice profile."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "confirmed": _boolean_property(
                            "Set true only after the user clearly confirmed a replacement when extra confirmation is needed."
                        )
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            }
        )
    if "get_voice_profile_status" in available:
        tools.append(
            {
                "type": "function",
                "name": "get_voice_profile_status",
                "description": "Read the local voice-profile status and current live speaker signal.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            }
        )
    if "reset_voice_profile" in available:
        tools.append(
            {
                "type": "function",
                "name": "reset_voice_profile",
                "description": "Delete the local Twinr voice profile when the user explicitly asks to remove it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "confirmed": _boolean_property(
                            "Set true only after the user clearly confirmed the reset when extra confirmation is needed."
                        )
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            }
        )
    if "enroll_portrait_identity" in available:
        tools.append(
            {
                "type": "function",
                "name": "enroll_portrait_identity",
                "description": (
                    "Capture the current live camera view and add it to Twinr's local on-device face profile. "
                    "Use only when the user explicitly asks Twinr to remember, learn, refresh, or update their face."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "display_name": _string_property(
                            "Optional friendly display name for the saved local face profile when the user explicitly wants it.",
                            min_length=1,
                        ),
                        "confirmed": _boolean_property(
                            "Set true only after the user clearly confirmed this local face-profile save when extra confirmation is needed."
                        ),
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            }
        )
    if "get_portrait_identity_status" in available:
        tools.append(
            {
                "type": "function",
                "name": "get_portrait_identity_status",
                "description": (
                    "Read Twinr's local on-device face-profile status, including how many portrait references are saved "
                    "and whether the current live camera view matches it."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "confirmed": _boolean_property(
                            "Set true only after the user clearly confirmed this local face-profile status lookup when extra confirmation is needed."
                        )
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            }
        )
    if "reset_portrait_identity" in available:
        tools.append(
            {
                "type": "function",
                "name": "reset_portrait_identity",
                "description": "Delete Twinr's local on-device face profile when the user explicitly asks to remove it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "confirmed": _boolean_property(
                            "Set true only after the user clearly confirmed the local face-profile reset when extra confirmation is needed."
                        )
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            }
        )
    if "manage_household_identity" in available:
        tools.append(
            {
                "type": "function",
                "name": "manage_household_identity",
                "description": (
                    "Manage Twinr's shared local household identity state across face, voice, live matching, "
                    "and explicit confirm or deny feedback. Use this tool for local household identity enrollment, "
                    "identity status, or when the user confirms that Twinr recognized the right or wrong person."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": _string_property(
                            "Choose one supported local household identity action.",
                            enum=("status", "enroll_face", "enroll_voice", "confirm_identity", "deny_identity"),
                        ),
                        "user_id": _string_property(
                            "Optional stable local member identifier when the user explicitly names which enrolled household member should be updated or confirmed.",
                            min_length=1,
                        ),
                        "display_name": _string_property(
                            "Optional friendly household member name when the user explicitly provides it.",
                            min_length=1,
                        ),
                        "confirmed": _boolean_property(
                            "Set true only after the user clearly confirmed a persistent local identity enrollment when extra confirmation is needed."
                        ),
                    },
                    "required": ["action"],
                    "additionalProperties": False,
                },
            }
        )
    if "end_conversation" in available:
        tools.append(
            {
                "type": "function",
                "name": "end_conversation",
                "description": (
                    "End the current follow-up listening loop when the user clearly indicates they are done for now, "
                    "for example by saying thanks, stop, pause, bye, or tschuss."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": _string_property(
                            "Optional short note describing why the conversation should end.",
                            min_length=1,
                        ),
                        "spoken_reply": _string_property(
                            "Short goodbye that Twinr should say immediately while ending the conversation.",
                            min_length=1,
                        ),
                    },
                    "required": [],
                    "additionalProperties": False,
                },
            }
        )
    if "inspect_camera" in available:
        tools.append(
            {
                "type": "function",
                "name": "inspect_camera",
                "description": (
                    "Inspect the current live camera view when the user asks you to look at them, "
                    "an object, a document, or something they are showing."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": _string_property(
                            "The exact user request about what should be inspected in the camera view.",
                            min_length=1,
                        )
                    },
                    "required": ["question"],
                    "additionalProperties": False,
                },
            }
        )
    return tools


def build_compact_agent_tool_schemas(tool_names: Iterable[str] | str | bytes | bytearray | None) -> list[dict[str, Any]]:
    """Build compact tool schemas with shortened descriptions.

    Args:
        tool_names: Tool names to include in the compact schema set.

    Returns:
        The canonical schemas with descriptions and nested nodes compacted for
        smaller prompt payloads.
    """
    return [_compact_tool_schema(schema) for schema in build_agent_tool_schemas(tool_names)]


def _compact_tool_schema(schema: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "type": schema.get("type", "function"),
        "name": schema.get("name"),
    }
    short_description = _compact_description(schema.get("description"))
    if short_description:
        compact["description"] = short_description
    parameters = schema.get("parameters")
    if isinstance(parameters, dict):
        compact["parameters"] = _compact_schema_node(parameters)
    return compact


def _compact_schema_node(node: Any) -> Any:
    if isinstance(node, dict):
        compact: dict[str, Any] = {}
        for key, value in node.items():
            if key == "description":
                continue
            # AUDIT-FIX(#3): Preserve the full JSON-Schema shape during compaction instead of
            # whitelisting a few keywords and silently dropping anyOf/allOf/if/then/const/format.
            compact[key] = _compact_schema_node(value)
        return compact
    if isinstance(node, list):
        return [_compact_schema_node(item) for item in node]
    return node


def _compact_description(value: object) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    first_sentence = text.split(". ", 1)[0].strip()
    if len(first_sentence) <= 72:
        return first_sentence
    return first_sentence[:69].rstrip() + "..."


def _append_realtime_validation_note(description: object) -> str:
    base = str(description or "").strip()
    if not base:
        return _REALTIME_VALIDATION_NOTE
    if _REALTIME_VALIDATION_NOTE in base:
        return base
    return f"{base} {_REALTIME_VALIDATION_NOTE}"


def _build_realtime_parameters(parameters: dict[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
    realtime_parameters: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }
    removed_keys = tuple(
        key for key in sorted(_REALTIME_TOP_LEVEL_UNSUPPORTED_SCHEMA_KEYS) if key in parameters
    )
    properties = parameters.get("properties")
    if isinstance(properties, dict):
        realtime_parameters["properties"] = deepcopy(properties)
    required = parameters.get("required")
    if isinstance(required, list):
        realtime_parameters["required"] = deepcopy(required)
    additional_properties = parameters.get("additionalProperties")
    if isinstance(additional_properties, bool):
        realtime_parameters["additionalProperties"] = additional_properties
    description = parameters.get("description")
    if isinstance(description, str) and description.strip():
        realtime_parameters["description"] = description.strip()
    return realtime_parameters, removed_keys


def _build_realtime_tool_schema(schema: dict[str, Any]) -> dict[str, Any]:
    realtime_schema = deepcopy(schema)
    parameters = schema.get("parameters")
    if not isinstance(parameters, dict):
        return realtime_schema
    realtime_parameters, removed_keys = _build_realtime_parameters(parameters)
    realtime_schema["parameters"] = realtime_parameters
    if removed_keys:
        realtime_schema["description"] = _append_realtime_validation_note(realtime_schema.get("description"))
    return realtime_schema


def build_realtime_tool_schemas(tool_names: Iterable[str] | str | bytes | bytearray | None) -> list[dict[str, Any]]:
    """Build realtime-safe tool schemas for providers with reduced support.

    Args:
        tool_names: Tool names to include in the realtime schema set.

    Returns:
        The canonical schemas with unsupported top-level JSON Schema keywords
        stripped from their parameter blocks and explanatory notes appended
        where Twinr must enforce remaining validation at execution time.
    """
    return [_build_realtime_tool_schema(schema) for schema in build_agent_tool_schemas(tool_names)]
