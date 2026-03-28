"""Schema transformations for compact and realtime-safe variants."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .shared import REALTIME_TOP_LEVEL_UNSUPPORTED_SCHEMA_KEYS, REALTIME_VALIDATION_NOTE


def compact_tool_schema(schema: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "type": schema.get("type", "function"),
        "name": schema.get("name"),
    }
    short_description = compact_description(schema.get("description"))
    if short_description:
        compact["description"] = short_description
    parameters = schema.get("parameters")
    if isinstance(parameters, dict):
        compact["parameters"] = compact_schema_node(parameters)
    return compact


def compact_schema_node(node: Any) -> Any:
    if isinstance(node, dict):
        compact: dict[str, Any] = {}
        for key, value in node.items():
            if key == "description":
                continue
            compact[key] = compact_schema_node(value)
        return compact
    if isinstance(node, list):
        return [compact_schema_node(item) for item in node]
    return node


def compact_description(value: object) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    first_sentence = text.split(". ", 1)[0].strip()
    if len(first_sentence) <= 72:
        return first_sentence
    return first_sentence[:69].rstrip() + "..."


def append_realtime_validation_note(description: object) -> str:
    base = str(description or "").strip()
    if not base:
        return REALTIME_VALIDATION_NOTE
    if REALTIME_VALIDATION_NOTE in base:
        return base
    return f"{base} {REALTIME_VALIDATION_NOTE}"


def build_realtime_parameters(parameters: dict[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
    realtime_parameters: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }
    removed_keys = tuple(
        key for key in sorted(REALTIME_TOP_LEVEL_UNSUPPORTED_SCHEMA_KEYS) if key in parameters
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


def build_realtime_tool_schema(schema: dict[str, Any]) -> dict[str, Any]:
    realtime_schema = deepcopy(schema)
    parameters = schema.get("parameters")
    if not isinstance(parameters, dict):
        return realtime_schema
    realtime_parameters, removed_keys = build_realtime_parameters(parameters)
    realtime_schema["parameters"] = realtime_parameters
    if removed_keys:
        realtime_schema["description"] = append_realtime_validation_note(realtime_schema.get("description"))
    return realtime_schema
