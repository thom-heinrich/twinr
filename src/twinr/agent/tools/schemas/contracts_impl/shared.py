"""Shared schema helpers for Twinr tool contract builders."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

CANONICAL_ENGLISH_MEMORY_NOTE = (
    "All semantic text fields must be canonical English, even if the user spoke another language. "
    "Keep names, phone numbers, email addresses, IDs, codes, and exact quoted text verbatim."
)
LOCAL_TIME_HHMM_PATTERN = r"^(?:[01]\d|2[0-3]):[0-5]\d$"
REALTIME_TOP_LEVEL_UNSUPPORTED_SCHEMA_KEYS: frozenset[str] = frozenset(
    {"allOf", "anyOf", "oneOf", "not", "enum", "if", "then", "else"}
)
REALTIME_VALIDATION_NOTE = (
    "Realtime compatibility note: some cross-field validation rules are enforced by Twinr at tool execution time "
    "instead of the API schema. Follow the field descriptions exactly and only call this tool with complete, valid arguments."
)


def normalize_tool_names(tool_names: Iterable[str] | str | bytes | bytearray | None) -> tuple[str, ...]:
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


def unique_strings(values: Iterable[Any]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        unique.append(item)
        seen.add(item)
    return sorted(unique, key=str.casefold)


def string_property(
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


def number_property(
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


def boolean_property(description: str) -> dict[str, Any]:
    return {"type": "boolean", "description": description}


def array_property(
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


def iso8601_datetime_property(description: str) -> dict[str, Any]:
    return string_property(description, string_format="date-time")


def time_of_day_property(description: str) -> dict[str, Any]:
    return string_property(description, pattern=LOCAL_TIME_HHMM_PATTERN)


def weekdays_property(description: str) -> dict[str, Any]:
    return array_property(
        description,
        number_property(
            "Weekday number where Monday is 0 and Sunday is 6.",
            minimum=0,
            maximum=6,
            integer=True,
        ),
        min_items=1,
        max_items=7,
        unique_items=True,
    )


def create_time_schedule_rules() -> list[dict[str, Any]]:
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


def update_time_schedule_rules() -> list[dict[str, Any]]:
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


def update_sensor_rules() -> list[dict[str, Any]]:
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


def simple_setting_rules(spoken_voices: Sequence[str]) -> list[dict[str, Any]]:
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
