"""Smart-home tool schema families."""

from __future__ import annotations

from typing import Any

from .context import SchemaBuildContext
from .shared import array_property, boolean_property, number_property, string_property


def build_list_smart_home_entities_schema(context: SchemaBuildContext) -> dict[str, Any]:
    return {
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
                "entity_ids": array_property(
                    "Optional exact entity IDs to limit the listing to a known subset.",
                    string_property("Exact smart-home entity ID.", min_length=1),
                    min_items=1,
                    max_items=8,
                    unique_items=True,
                ),
                "entity_class": string_property(
                    "Optional legacy single entity-class filter such as light, scene, motion_sensor, or device_health.",
                    enum=context.smart_home_entity_classes,
                ),
                "entity_classes": array_property(
                    "Optional entity-class filter set. Prefer this over entity_class when more than one class is relevant.",
                    string_property(
                        "Smart-home entity class.",
                        enum=context.smart_home_entity_classes,
                    ),
                    min_items=1,
                    max_items=8,
                    unique_items=True,
                ),
                "areas": array_property(
                    "Optional exact area labels to keep, such as Wohnzimmer or Flur.",
                    string_property("Exact smart-home area label.", min_length=1),
                    min_items=1,
                    max_items=8,
                    unique_items=True,
                ),
                "providers": array_property(
                    "Optional provider IDs to keep, such as hue.",
                    string_property("Exact smart-home provider ID.", min_length=1),
                    min_items=1,
                    max_items=8,
                    unique_items=True,
                ),
                "online": boolean_property(
                    "Optional online-status filter. Set false to focus on offline or unavailable entities."
                ),
                "controllable": boolean_property(
                    "Optional controllability filter."
                ),
                "readable": boolean_property(
                    "Optional readability filter."
                ),
                "include_unavailable": boolean_property(
                    "Set true to include currently unavailable or offline entities."
                ),
                "state_filters": array_property(
                    "Optional exact scalar state filters. Use keys such as on, brightness, motion, or nested dotted paths when the provider exposes nested state.",
                    {
                        "type": "object",
                        "properties": {
                            "key": string_property(
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
                "aggregate_by": array_property(
                    "Optional entity fields to aggregate counts by. Prefer this first for broader house-status scans in larger homes before narrowing to exact entity names.",
                    string_property(
                        "Aggregate field.",
                        enum=context.smart_home_entity_aggregate_fields,
                    ),
                    min_items=1,
                    max_items=6,
                    unique_items=True,
                ),
                "limit": number_property(
                    "Maximum number of matching entities to return after filtering.",
                    minimum=1,
                    maximum=32,
                    integer=True,
                ),
                "cursor": string_property(
                    "Optional opaque offset cursor from the previous entity list response.",
                    min_length=1,
                ),
            },
            "required": [],
            "additionalProperties": False,
        },
    }


def build_read_smart_home_state_schema(_context: SchemaBuildContext) -> dict[str, Any]:
    return {
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
                "entity_ids": array_property(
                    "One or more exact smart-home entity IDs to inspect.",
                    string_property("Exact smart-home entity ID.", min_length=1),
                    min_items=1,
                    max_items=8,
                    unique_items=True,
                ),
                "include_unavailable": boolean_property(
                    "Set true to include entities even if they are currently unavailable."
                ),
            },
            "required": ["entity_ids"],
            "additionalProperties": False,
        },
    }


def build_control_smart_home_entities_schema(context: SchemaBuildContext) -> dict[str, Any]:
    return {
        "type": "function",
        "name": "control_smart_home_entities",
        "description": (
            "Control allowed low-risk smart-home targets such as lights, grouped lights, switches, and scenes. "
            "Use this only for explicit user requests to change device state."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entity_ids": array_property(
                    "One or more exact smart-home entity IDs to control.",
                    string_property("Exact smart-home entity ID.", min_length=1),
                    min_items=1,
                    max_items=8,
                    unique_items=True,
                ),
                "command": string_property(
                    "Generic control command to execute.",
                    enum=context.smart_home_commands,
                ),
                "brightness": number_property(
                    "Brightness percentage from 0 to 100. Required only for set_brightness.",
                    minimum=0,
                    maximum=100,
                ),
                "confirmed": boolean_property(
                    "Set true only after the user clearly confirmed this control action when explicit confirmation is needed."
                ),
            },
            "required": ["entity_ids", "command"],
            "additionalProperties": False,
        },
    }


def build_read_smart_home_sensor_stream_schema(context: SchemaBuildContext) -> dict[str, Any]:
    return {
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
                "entity_ids": array_property(
                    "Optional exact smart-home entity IDs to keep in the event stream.",
                    string_property("Exact smart-home entity ID.", min_length=1),
                    min_items=1,
                    max_items=8,
                    unique_items=True,
                ),
                "event_kinds": array_property(
                    "Optional event-kind filters such as motion_detected, device_offline, or alarm_triggered.",
                    string_property(
                        "Normalized smart-home event kind.",
                        enum=context.smart_home_event_kinds,
                    ),
                    min_items=1,
                    max_items=8,
                    unique_items=True,
                ),
                "areas": array_property(
                    "Optional exact area labels to keep in the event stream.",
                    string_property("Exact smart-home area label.", min_length=1),
                    min_items=1,
                    max_items=8,
                    unique_items=True,
                ),
                "providers": array_property(
                    "Optional provider IDs to keep in the event stream.",
                    string_property("Exact smart-home provider ID.", min_length=1),
                    min_items=1,
                    max_items=8,
                    unique_items=True,
                ),
                "aggregate_by": array_property(
                    "Optional event fields to aggregate counts by.",
                    string_property(
                        "Aggregate field.",
                        enum=context.smart_home_event_aggregate_fields,
                    ),
                    min_items=1,
                    max_items=4,
                    unique_items=True,
                ),
                "cursor": string_property(
                    "Optional provider cursor from the previous stream read when available.",
                    min_length=1,
                ),
                "limit": number_property(
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


TOOL_BUILDERS = (
    ("list_smart_home_entities", build_list_smart_home_entities_schema),
    ("read_smart_home_state", build_read_smart_home_state_schema),
    ("control_smart_home_entities", build_control_smart_home_entities_schema),
    ("read_smart_home_sensor_stream", build_read_smart_home_sensor_stream_schema),
)
