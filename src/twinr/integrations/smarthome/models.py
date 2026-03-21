"""Define generic smart-home entities, commands, and sensor events.

This package keeps vendor-neutral read/control/stream contracts separate from
provider-specific APIs so future Hue, Matter, Zigbee, Z-Wave, or cloud-backed
integrations can share one stable Twinr surface.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
import math


def _ensure_non_empty_text(field_name: str, value: object) -> str:
    """Return one stripped non-empty string."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")
    if any(ord(character) < 32 for character in normalized):
        raise ValueError(f"{field_name} must not contain control characters.")
    return normalized


def _ensure_bool(field_name: str, value: object) -> bool:
    """Require a real boolean value."""

    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool.")
    return value


def _normalize_json_value(value: object, *, field_name: str) -> object:
    """Normalize nested values into deterministic JSON-safe structures."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must not contain non-finite floats.")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for raw_key, raw_value in value.items():
            key = _ensure_non_empty_text(field_name, str(raw_key))
            normalized[key] = _normalize_json_value(raw_value, field_name=f"{field_name}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _normalize_json_value(item, field_name=f"{field_name}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{field_name} contains unsupported value type {type(value).__name__}.")


def _normalize_json_mapping(field_name: str, value: object) -> dict[str, object]:
    """Normalize one mapping into a JSON-safe dictionary."""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping.")
    normalized = _normalize_json_value(value, field_name=field_name)
    if not isinstance(normalized, dict):  # pragma: no cover - defensive
        raise TypeError(f"{field_name} must normalize to a dictionary.")
    return normalized


def _normalize_text_tuple(field_name: str, value: object) -> tuple[str, ...]:
    """Normalize a text collection into a deduplicated tuple."""

    if value is None:
        return ()
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        raise TypeError(f"{field_name} must be a string, list, tuple, or None.")
    return tuple(dict.fromkeys(_ensure_non_empty_text(f"{field_name}[{index}]", item) for index, item in enumerate(values)))


class SmartHomeEntityClass(StrEnum):
    """Describe the coarse class of one smart-home entity."""

    LIGHT = "light"
    LIGHT_GROUP = "light_group"
    SCENE = "scene"
    SWITCH = "switch"
    MOTION_SENSOR = "motion_sensor"
    LIGHT_SENSOR = "light_sensor"
    TEMPERATURE_SENSOR = "temperature_sensor"
    BATTERY_SENSOR = "battery_sensor"
    DEVICE_HEALTH = "device_health"
    BUTTON = "button"
    ALARM = "alarm"
    UNKNOWN = "unknown"


class SmartHomeCommand(StrEnum):
    """Describe one generic smart-home control command."""

    TURN_ON = "turn_on"
    TURN_OFF = "turn_off"
    SET_BRIGHTNESS = "set_brightness"
    ACTIVATE = "activate"


class SmartHomeEntityAggregateField(StrEnum):
    """Describe one entity field that generic queries may aggregate by."""

    ENTITY_CLASS = "entity_class"
    AREA = "area"
    PROVIDER = "provider"
    ONLINE = "online"
    CONTROLLABLE = "controllable"
    READABLE = "readable"


class SmartHomeEventKind(StrEnum):
    """Describe one normalized smart-home stream event kind."""

    MOTION_DETECTED = "motion_detected"
    MOTION_CLEARED = "motion_cleared"
    BUTTON_PRESSED = "button_pressed"
    DEVICE_ONLINE = "device_online"
    DEVICE_OFFLINE = "device_offline"
    ALARM_TRIGGERED = "alarm_triggered"
    ALARM_CLEARED = "alarm_cleared"
    STATE_CHANGED = "state_changed"


class SmartHomeEventAggregateField(StrEnum):
    """Describe one event field that generic queries may aggregate by."""

    EVENT_KIND = "event_kind"
    AREA = "area"
    PROVIDER = "provider"
    ENTITY_ID = "entity_id"


@dataclass(frozen=True, slots=True)
class SmartHomeEntity:
    """Describe one generic smart-home entity plus its current state."""

    entity_id: str
    provider: str
    label: str
    entity_class: SmartHomeEntityClass = SmartHomeEntityClass.UNKNOWN
    area: str = ""
    readable: bool = True
    controllable: bool = False
    online: bool = True
    supported_commands: tuple[SmartHomeCommand, ...] = ()
    state: dict[str, object] = field(default_factory=dict)
    attributes: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize entity metadata and nested JSON fields."""

        object.__setattr__(self, "entity_id", _ensure_non_empty_text("entity_id", self.entity_id))
        object.__setattr__(self, "provider", _ensure_non_empty_text("provider", self.provider))
        object.__setattr__(self, "label", _ensure_non_empty_text("label", self.label))
        object.__setattr__(self, "entity_class", SmartHomeEntityClass(self.entity_class))
        object.__setattr__(self, "area", "" if self.area == "" else _ensure_non_empty_text("area", self.area))
        object.__setattr__(self, "readable", _ensure_bool("readable", self.readable))
        object.__setattr__(self, "controllable", _ensure_bool("controllable", self.controllable))
        object.__setattr__(self, "online", _ensure_bool("online", self.online))
        object.__setattr__(
            self,
            "supported_commands",
            tuple(
                SmartHomeCommand(command)
                for command in (self.supported_commands if isinstance(self.supported_commands, tuple) else tuple(self.supported_commands))
            ),
        )
        object.__setattr__(self, "state", _normalize_json_mapping("state", self.state))
        object.__setattr__(self, "attributes", _normalize_json_mapping("attributes", self.attributes))

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation used by integration results."""

        return {
            "entity_id": self.entity_id,
            "provider": self.provider,
            "label": self.label,
            "entity_class": self.entity_class.value,
            "area": self.area,
            "readable": self.readable,
            "controllable": self.controllable,
            "online": self.online,
            "supported_commands": [command.value for command in self.supported_commands],
            "state": dict(self.state),
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True, slots=True)
class SmartHomeEvent:
    """Describe one normalized smart-home event suitable for background use."""

    event_id: str
    provider: str
    entity_id: str
    event_kind: SmartHomeEventKind
    observed_at: str
    label: str = ""
    area: str = ""
    details: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize identifiers and JSON-safe detail payloads."""

        object.__setattr__(self, "event_id", _ensure_non_empty_text("event_id", self.event_id))
        object.__setattr__(self, "provider", _ensure_non_empty_text("provider", self.provider))
        object.__setattr__(self, "entity_id", _ensure_non_empty_text("entity_id", self.entity_id))
        object.__setattr__(self, "event_kind", SmartHomeEventKind(self.event_kind))
        object.__setattr__(self, "observed_at", _ensure_non_empty_text("observed_at", self.observed_at))
        object.__setattr__(self, "label", "" if self.label == "" else _ensure_non_empty_text("label", self.label))
        object.__setattr__(self, "area", "" if self.area == "" else _ensure_non_empty_text("area", self.area))
        object.__setattr__(self, "details", _normalize_json_mapping("details", self.details))

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe event payload."""

        return {
            "event_id": self.event_id,
            "provider": self.provider,
            "entity_id": self.entity_id,
            "event_kind": self.event_kind.value,
            "observed_at": self.observed_at,
            "label": self.label,
            "area": self.area,
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class SmartHomeEventBatch:
    """Return a bounded slice of smart-home stream events plus a cursor."""

    events: tuple[SmartHomeEvent, ...]
    next_cursor: str | None = None
    stream_live: bool = True

    def __post_init__(self) -> None:
        """Validate event collections and stream metadata."""

        if not isinstance(self.events, tuple):
            raise TypeError("events must be a tuple.")
        if not all(isinstance(event, SmartHomeEvent) for event in self.events):
            raise TypeError("events must contain SmartHomeEvent items only.")
        if self.next_cursor is not None:
            object.__setattr__(self, "next_cursor", _ensure_non_empty_text("next_cursor", self.next_cursor))
        object.__setattr__(self, "stream_live", _ensure_bool("stream_live", self.stream_live))

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe event batch payload."""

        return {
            "events": [event.as_dict() for event in self.events],
            "next_cursor": self.next_cursor,
            "stream_live": self.stream_live,
        }


__all__ = [
    "SmartHomeCommand",
    "SmartHomeEntity",
    "SmartHomeEntityAggregateField",
    "SmartHomeEntityClass",
    "SmartHomeEvent",
    "SmartHomeEventBatch",
    "SmartHomeEventAggregateField",
    "SmartHomeEventKind",
]
