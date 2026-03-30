# CHANGELOG: 2026-03-30
# BUG-1: Reject silent normalized-key collisions in state/attributes/details.
# BUG-2: Deep-freeze nested JSON so frozen models become actually immutable and race-safe.
# BUG-3: Validate and canonicalize timestamps to timezone-aware RFC3339 UTC strings.
# SEC-1: Bound nested JSON depth, fanout, and string sizes to reduce payload-based DoS risk on Raspberry Pi 4.
# IMP-1: Expand generic entity/command/event coverage to modern Matter/Home-Assistant-style domains.
# IMP-2: Add explicit command request/result contracts, floor/device metadata, revisioning, and stream-gap metadata.
# IMP-3: Add from_dict/to_json helpers plus an optional msgspec fast-path when installed.

"""Define generic smart-home entities, commands, command lifecycle objects, and sensor events.

This package keeps vendor-neutral read/control/stream contracts separate from
provider-specific APIs so Matter, Hue, Zigbee, Z-Wave, Home Assistant, or
cloud-backed integrations can share one stable Twinr surface.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
import json
import math
from types import MappingProxyType
from typing import Final, TypeAlias

try:  # Optional frontier fast-path; stdlib fallback keeps the module drop-in.
    import msgspec as _msgspec  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    _msgspec = None


SCHEMA_VERSION: Final[str] = "2026-03-30"
WIRE_PROTOCOL: Final[str] = "twinr.smarthome.v2"

_MAX_GENERIC_TEXT_LENGTH: Final[int] = 256
_MAX_LABEL_LENGTH: Final[int] = 256
_MAX_AREA_LENGTH: Final[int] = 128
_MAX_PROVIDER_LENGTH: Final[int] = 128
_MAX_ID_LENGTH: Final[int] = 256
_MAX_CURSOR_LENGTH: Final[int] = 512
_MAX_JSON_DEPTH: Final[int] = 8
_MAX_JSON_MAPPING_ITEMS: Final[int] = 256
_MAX_JSON_SEQUENCE_ITEMS: Final[int] = 512
_MAX_JSON_STRING_LENGTH: Final[int] = 8192
_MAX_EVENT_BATCH_EVENTS: Final[int] = 512

FrozenJsonScalar: TypeAlias = None | bool | int | float | str
FrozenJsonValue: TypeAlias = FrozenJsonScalar | tuple["FrozenJsonValue", ...] | Mapping[str, "FrozenJsonValue"]
JsonScalar: TypeAlias = FrozenJsonScalar
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def _ensure_text(
    field_name: str,
    value: object,
    *,
    allow_empty: bool = False,
    max_length: int = _MAX_GENERIC_TEXT_LENGTH,
) -> str:
    """Normalize one printable text field."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        if allow_empty:
            return ""
        raise ValueError(f"{field_name} must not be empty.")
    if len(normalized) > max_length:
        raise ValueError(f"{field_name} must not exceed {max_length} characters.")
    if any((not character.isprintable()) or ord(character) == 127 for character in normalized):
        raise ValueError(f"{field_name} must not contain control characters.")
    return normalized


def _ensure_bool(field_name: str, value: object) -> bool:
    """Require a real boolean value."""

    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool.")
    return value


def _ensure_optional_non_negative_int(field_name: str, value: object) -> int | None:
    """Require one optional non-negative integer."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int or None.")
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0.")
    return value


def _ensure_non_negative_int(field_name: str, value: object) -> int:
    """Require one non-negative integer."""

    normalized = _ensure_optional_non_negative_int(field_name, value)
    if normalized is None:  # pragma: no cover - defensive
        raise TypeError(f"{field_name} must be an int.")
    return normalized


def _ensure_rfc3339_timestamp(
    field_name: str,
    value: object,
    *,
    allow_none: bool = False,
) -> str | None:
    """Normalize one timestamp into UTC RFC3339 text."""

    if value is None:
        if allow_none:
            return None
        raise TypeError(f"{field_name} must be a timezone-aware RFC3339 string or datetime.")

    if isinstance(value, datetime):
        timestamp = value
    elif isinstance(value, str):
        text = _ensure_text(field_name, value, max_length=64)
        normalized = f"{text[:-1]}+00:00" if text.endswith("Z") else text
        try:
            timestamp = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError(
                f"{field_name} must be a valid RFC3339/ISO8601 timestamp with timezone."
            ) from exc
    else:
        raise TypeError(f"{field_name} must be a timezone-aware RFC3339 string or datetime.")

    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"{field_name} must include timezone information.")

    canonical = timestamp.astimezone(timezone.utc)
    timespec = "microseconds" if canonical.microsecond else "seconds"
    return canonical.isoformat(timespec=timespec).replace("+00:00", "Z")


def _normalize_json_value(
    value: object,
    *,
    field_name: str,
    depth: int = 0,
) -> FrozenJsonValue:
    """Normalize nested values into bounded, deterministic, immutable JSON-safe structures."""

    if depth > _MAX_JSON_DEPTH:
        raise ValueError(f"{field_name} exceeds the maximum nesting depth of {_MAX_JSON_DEPTH}.")

    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str) and len(value) > _MAX_JSON_STRING_LENGTH:
            raise ValueError(
                f"{field_name} must not contain strings longer than {_MAX_JSON_STRING_LENGTH} characters."
            )
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must not contain non-finite floats.")
        return value

    if isinstance(value, Mapping):
        if len(value) > _MAX_JSON_MAPPING_ITEMS:
            raise ValueError(
                f"{field_name} must not contain more than {_MAX_JSON_MAPPING_ITEMS} mapping items."
            )
        normalized: dict[str, FrozenJsonValue] = {}
        for raw_key, raw_value in value.items():
            key = _ensure_text(
                f"{field_name}.<key>",
                str(raw_key),
                max_length=_MAX_GENERIC_TEXT_LENGTH,
            )
            if key in normalized:
                raise ValueError(
                    f"{field_name} contains duplicate keys after normalization: {key!r}."
                )
            normalized[key] = _normalize_json_value(
                raw_value,
                field_name=f"{field_name}.{key}",
                depth=depth + 1,
            )
        return MappingProxyType(normalized)

    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_JSON_SEQUENCE_ITEMS:
            raise ValueError(
                f"{field_name} must not contain more than {_MAX_JSON_SEQUENCE_ITEMS} items."
            )
        return tuple(
            _normalize_json_value(item, field_name=f"{field_name}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        )

    raise TypeError(f"{field_name} contains unsupported value type {type(value).__name__}.")


def _normalize_json_mapping(field_name: str, value: object) -> Mapping[str, FrozenJsonValue]:
    """Normalize one mapping into an immutable JSON-safe dictionary."""

    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping.")
    normalized = _normalize_json_value(value, field_name=field_name)
    if not isinstance(normalized, Mapping):  # pragma: no cover - defensive
        raise TypeError(f"{field_name} must normalize to a dictionary.")
    return normalized


def _json_to_mutable(value: FrozenJsonValue) -> JsonValue:
    """Return one mutable JSON-safe structure."""

    if isinstance(value, Mapping):
        return {key: _json_to_mutable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_to_mutable(item) for item in value]
    return value


def _json_mapping_to_mutable(value: Mapping[str, FrozenJsonValue]) -> dict[str, JsonValue]:
    """Return one mutable JSON-safe dictionary."""

    return {key: _json_to_mutable(item) for key, item in value.items()}


def _normalize_text_tuple(
    field_name: str,
    value: object,
    *,
    max_length: int = _MAX_GENERIC_TEXT_LENGTH,
) -> tuple[str, ...]:
    """Normalize a text collection into a deduplicated tuple."""

    if value is None:
        return ()
    if isinstance(value, str):
        values: Sequence[object] = (value,)
    elif isinstance(value, Sequence):
        values = value
    else:
        raise TypeError(f"{field_name} must be a string, sequence, or None.")
    normalized: dict[str, None] = {}
    for index, item in enumerate(values):
        text = _ensure_text(f"{field_name}[{index}]", item, max_length=max_length)
        normalized[text] = None
    return tuple(normalized.keys())


def _normalize_enum_tuple(enum_type: type[StrEnum], field_name: str, value: object) -> tuple[StrEnum, ...]:
    """Normalize an enum collection into a deduplicated tuple preserving order."""

    if value is None:
        return ()
    if isinstance(value, enum_type) or isinstance(value, str):
        values = (value,)
    elif isinstance(value, Iterable):
        values = tuple(value)
    else:
        raise TypeError(f"{field_name} must be a string, enum, iterable, or None.")

    normalized: dict[StrEnum, None] = {}
    for index, item in enumerate(values):
        normalized[enum_type(item)] = None
    return tuple(normalized.keys())


def _encode_json_payload(payload: object) -> bytes:
    """Encode one payload using msgspec when available, otherwise stdlib json."""

    if _msgspec is not None:
        return _msgspec.json.encode(payload)
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=False,
    ).encode("utf-8")


class _WireModel:
    """Shared serialization helpers for Twinr wire models."""

    def as_wire_dict(self) -> dict[str, object]:
        """Return a versioned envelope for cross-process transport."""

        return {
            "schema_version": SCHEMA_VERSION,
            "wire_protocol": WIRE_PROTOCOL,
            "payload": self.as_dict(),
        }

    def to_json_bytes(self, *, envelope: bool = False) -> bytes:
        """Return a compact UTF-8 JSON payload."""

        payload = self.as_wire_dict() if envelope else self.as_dict()
        return _encode_json_payload(payload)

    def to_json(self, *, envelope: bool = False) -> str:
        """Return a compact JSON string."""

        return self.to_json_bytes(envelope=envelope).decode("utf-8")


class SmartHomeEntityClass(StrEnum):
    """Describe the coarse class of one smart-home entity."""

    LIGHT = "light"
    LIGHT_GROUP = "light_group"
    SCENE = "scene"
    SWITCH = "switch"
    OUTLET = "outlet"
    COVER = "cover"
    LOCK = "lock"
    CLIMATE = "climate"
    FAN = "fan"
    VALVE = "valve"
    MEDIA_PLAYER = "media_player"
    VACUUM = "vacuum"
    CAMERA = "camera"
    SIREN = "siren"
    SENSOR = "sensor"
    MOTION_SENSOR = "motion_sensor"
    OCCUPANCY_SENSOR = "occupancy_sensor"
    CONTACT_SENSOR = "contact_sensor"
    LIGHT_SENSOR = "light_sensor"
    TEMPERATURE_SENSOR = "temperature_sensor"
    HUMIDITY_SENSOR = "humidity_sensor"
    BATTERY_SENSOR = "battery_sensor"
    AIR_QUALITY_SENSOR = "air_quality_sensor"
    DEVICE_HEALTH = "device_health"
    BUTTON = "button"
    ALARM = "alarm"
    UNKNOWN = "unknown"


class SmartHomeCapability(StrEnum):
    """Describe one normalized capability exposed by an entity."""

    POWER = "power"
    BRIGHTNESS = "brightness"
    COLOR = "color"
    COLOR_TEMPERATURE = "color_temperature"
    EFFECT = "effect"
    POSITION = "position"
    LOCK = "lock"
    TARGET_TEMPERATURE = "target_temperature"
    MODE = "mode"
    VOLUME = "volume"
    MUTE = "mute"
    SCENE = "scene"
    MOTION = "motion"
    OCCUPANCY = "occupancy"
    BATTERY = "battery"
    CONNECTIVITY = "connectivity"
    SENSOR_VALUE = "sensor_value"


class SmartHomeCommand(StrEnum):
    """Describe one generic smart-home control command."""

    TURN_ON = "turn_on"
    TURN_OFF = "turn_off"
    TOGGLE = "toggle"
    SET_BRIGHTNESS = "set_brightness"
    SET_COLOR = "set_color"
    SET_COLOR_TEMPERATURE = "set_color_temperature"
    SET_POSITION = "set_position"
    OPEN = "open"
    CLOSE = "close"
    STOP = "stop"
    LOCK = "lock"
    UNLOCK = "unlock"
    SET_TEMPERATURE = "set_temperature"
    SET_MODE = "set_mode"
    ACTIVATE = "activate"
    DEACTIVATE = "deactivate"
    SET_VOLUME = "set_volume"
    MUTE = "mute"
    UNMUTE = "unmute"
    START = "start"
    PAUSE = "pause"
    RESUME = "resume"
    RETURN_TO_BASE = "return_to_base"
    ACKNOWLEDGE = "acknowledge"


class SmartHomeCommandStatus(StrEnum):
    """Describe the lifecycle state of one command."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    NOT_SUPPORTED = "not_supported"


class SmartHomeEntityAggregateField(StrEnum):
    """Describe one entity field that generic queries may aggregate by."""

    ENTITY_CLASS = "entity_class"
    AREA = "area"
    FLOOR = "floor"
    DEVICE_ID = "device_id"
    PROVIDER = "provider"
    ONLINE = "online"
    CONTROLLABLE = "controllable"
    READABLE = "readable"
    CAPABILITY = "capability"


class SmartHomeEventKind(StrEnum):
    """Describe one normalized smart-home stream event kind."""

    MOTION_DETECTED = "motion_detected"
    MOTION_CLEARED = "motion_cleared"
    OCCUPANCY_DETECTED = "occupancy_detected"
    OCCUPANCY_CLEARED = "occupancy_cleared"
    BUTTON_PRESSED = "button_pressed"
    BUTTON_DOUBLE_PRESSED = "button_double_pressed"
    BUTTON_LONG_PRESSED = "button_long_pressed"
    CONTACT_OPENED = "contact_opened"
    CONTACT_CLOSED = "contact_closed"
    LOCKED = "locked"
    UNLOCKED = "unlocked"
    DEVICE_ONLINE = "device_online"
    DEVICE_OFFLINE = "device_offline"
    ALARM_TRIGGERED = "alarm_triggered"
    ALARM_CLEARED = "alarm_cleared"
    SENSOR_REPORTED = "sensor_reported"
    STATE_CHANGED = "state_changed"


class SmartHomeEventAggregateField(StrEnum):
    """Describe one event field that generic queries may aggregate by."""

    EVENT_KIND = "event_kind"
    AREA = "area"
    FLOOR = "floor"
    DEVICE_ID = "device_id"
    PROVIDER = "provider"
    ENTITY_ID = "entity_id"
    CORRELATION_ID = "correlation_id"


@dataclass(frozen=True, slots=True)
class SmartHomeEntity(_WireModel):
    """Describe one generic smart-home entity plus its current state."""

    entity_id: str
    provider: str
    label: str
    entity_class: SmartHomeEntityClass = SmartHomeEntityClass.UNKNOWN
    area: str = ""
    floor: str = ""
    device_id: str = ""
    readable: bool = True
    controllable: bool = False
    online: bool = True
    observed_at: str | datetime | None = None
    last_changed_at: str | datetime | None = None
    capabilities: tuple[SmartHomeCapability, ...] = ()
    supported_commands: tuple[SmartHomeCommand, ...] = ()
    aliases: tuple[str, ...] = ()
    revision: int | None = None
    # BREAKING: state/attributes are now deep-frozen internally; mutate as_dict() output instead.
    state: Mapping[str, FrozenJsonValue] = field(default_factory=dict)
    attributes: Mapping[str, FrozenJsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize entity metadata and nested JSON fields."""

        object.__setattr__(self, "entity_id", _ensure_text("entity_id", self.entity_id, max_length=_MAX_ID_LENGTH))
        object.__setattr__(self, "provider", _ensure_text("provider", self.provider, max_length=_MAX_PROVIDER_LENGTH))
        object.__setattr__(self, "label", _ensure_text("label", self.label, max_length=_MAX_LABEL_LENGTH))
        object.__setattr__(self, "entity_class", SmartHomeEntityClass(self.entity_class))
        object.__setattr__(self, "area", _ensure_text("area", self.area, allow_empty=True, max_length=_MAX_AREA_LENGTH))
        object.__setattr__(self, "floor", _ensure_text("floor", self.floor, allow_empty=True, max_length=_MAX_AREA_LENGTH))
        object.__setattr__(self, "device_id", _ensure_text("device_id", self.device_id, allow_empty=True, max_length=_MAX_ID_LENGTH))
        readable = _ensure_bool("readable", self.readable)
        controllable = _ensure_bool("controllable", self.controllable)
        online = _ensure_bool("online", self.online)
        observed_at = _ensure_rfc3339_timestamp("observed_at", self.observed_at, allow_none=True)
        last_changed_at = _ensure_rfc3339_timestamp("last_changed_at", self.last_changed_at, allow_none=True)
        capabilities = tuple(
            capability
            for capability in _normalize_enum_tuple(
                SmartHomeCapability, "capabilities", self.capabilities
            )
        )
        supported_commands = tuple(
            command
            for command in _normalize_enum_tuple(
                SmartHomeCommand, "supported_commands", self.supported_commands
            )
        )
        if supported_commands and not controllable:
            controllable = True

        object.__setattr__(self, "readable", readable)
        object.__setattr__(self, "controllable", controllable)
        object.__setattr__(self, "online", online)
        object.__setattr__(self, "observed_at", observed_at)
        object.__setattr__(self, "last_changed_at", last_changed_at)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "supported_commands", supported_commands)
        object.__setattr__(self, "aliases", _normalize_text_tuple("aliases", self.aliases, max_length=_MAX_LABEL_LENGTH))
        object.__setattr__(self, "revision", _ensure_optional_non_negative_int("revision", self.revision))
        object.__setattr__(self, "state", _normalize_json_mapping("state", self.state))
        object.__setattr__(self, "attributes", _normalize_json_mapping("attributes", self.attributes))

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SmartHomeEntity":
        """Build one entity from a plain dictionary."""

        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping.")
        return cls(
            entity_id=payload["entity_id"],
            provider=payload["provider"],
            label=payload["label"],
            entity_class=payload.get("entity_class", SmartHomeEntityClass.UNKNOWN),
            area=payload.get("area", ""),
            floor=payload.get("floor", ""),
            device_id=payload.get("device_id", ""),
            readable=payload.get("readable", True),
            controllable=payload.get("controllable", False),
            online=payload.get("online", True),
            observed_at=payload.get("observed_at"),
            last_changed_at=payload.get("last_changed_at"),
            capabilities=payload.get("capabilities", ()),
            supported_commands=payload.get("supported_commands", ()),
            aliases=payload.get("aliases", ()),
            revision=payload.get("revision"),
            state=payload.get("state", {}),
            attributes=payload.get("attributes", {}),
        )

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation used by integration results."""

        return {
            "entity_id": self.entity_id,
            "provider": self.provider,
            "label": self.label,
            "entity_class": self.entity_class.value,
            "area": self.area,
            "floor": self.floor,
            "device_id": self.device_id,
            "readable": self.readable,
            "controllable": self.controllable,
            "online": self.online,
            "observed_at": self.observed_at,
            "last_changed_at": self.last_changed_at,
            "capabilities": [capability.value for capability in self.capabilities],
            "supported_commands": [command.value for command in self.supported_commands],
            "aliases": list(self.aliases),
            "revision": self.revision,
            "state": _json_mapping_to_mutable(self.state),
            "attributes": _json_mapping_to_mutable(self.attributes),
        }


@dataclass(frozen=True, slots=True)
class SmartHomeCommandRequest(_WireModel):
    """Describe one generic smart-home control request."""

    command_id: str
    provider: str
    entity_id: str
    command: SmartHomeCommand
    requested_at: str | datetime
    correlation_id: str = ""
    requested_by: str = ""
    timeout_ms: int | None = None
    arguments: Mapping[str, FrozenJsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize command request fields."""

        object.__setattr__(self, "command_id", _ensure_text("command_id", self.command_id, max_length=_MAX_ID_LENGTH))
        object.__setattr__(self, "provider", _ensure_text("provider", self.provider, max_length=_MAX_PROVIDER_LENGTH))
        object.__setattr__(self, "entity_id", _ensure_text("entity_id", self.entity_id, max_length=_MAX_ID_LENGTH))
        object.__setattr__(self, "command", SmartHomeCommand(self.command))
        # BREAKING: timestamp fields now require timezone-aware RFC3339/ISO8601 input and are canonicalized to UTC.
        object.__setattr__(self, "requested_at", _ensure_rfc3339_timestamp("requested_at", self.requested_at))
        object.__setattr__(self, "correlation_id", _ensure_text("correlation_id", self.correlation_id, allow_empty=True, max_length=_MAX_ID_LENGTH))
        object.__setattr__(self, "requested_by", _ensure_text("requested_by", self.requested_by, allow_empty=True, max_length=_MAX_LABEL_LENGTH))
        object.__setattr__(self, "timeout_ms", _ensure_optional_non_negative_int("timeout_ms", self.timeout_ms))
        object.__setattr__(self, "arguments", _normalize_json_mapping("arguments", self.arguments))

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SmartHomeCommandRequest":
        """Build one command request from a plain dictionary."""

        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping.")
        return cls(
            command_id=payload["command_id"],
            provider=payload["provider"],
            entity_id=payload["entity_id"],
            command=payload["command"],
            requested_at=payload["requested_at"],
            correlation_id=payload.get("correlation_id", ""),
            requested_by=payload.get("requested_by", ""),
            timeout_ms=payload.get("timeout_ms"),
            arguments=payload.get("arguments", {}),
        )

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe command request payload."""

        return {
            "command_id": self.command_id,
            "provider": self.provider,
            "entity_id": self.entity_id,
            "command": self.command.value,
            "requested_at": self.requested_at,
            "correlation_id": self.correlation_id,
            "requested_by": self.requested_by,
            "timeout_ms": self.timeout_ms,
            "arguments": _json_mapping_to_mutable(self.arguments),
        }


@dataclass(frozen=True, slots=True)
class SmartHomeCommandResult(_WireModel):
    """Describe one normalized control result emitted by a provider or orchestrator."""

    command_id: str
    provider: str
    entity_id: str
    command: SmartHomeCommand
    status: SmartHomeCommandStatus
    observed_at: str | datetime
    correlation_id: str = ""
    message: str = ""
    error_code: str = ""
    state: Mapping[str, FrozenJsonValue] = field(default_factory=dict)
    details: Mapping[str, FrozenJsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize command result fields."""

        object.__setattr__(self, "command_id", _ensure_text("command_id", self.command_id, max_length=_MAX_ID_LENGTH))
        object.__setattr__(self, "provider", _ensure_text("provider", self.provider, max_length=_MAX_PROVIDER_LENGTH))
        object.__setattr__(self, "entity_id", _ensure_text("entity_id", self.entity_id, max_length=_MAX_ID_LENGTH))
        object.__setattr__(self, "command", SmartHomeCommand(self.command))
        object.__setattr__(self, "status", SmartHomeCommandStatus(self.status))
        object.__setattr__(self, "observed_at", _ensure_rfc3339_timestamp("observed_at", self.observed_at))
        object.__setattr__(self, "correlation_id", _ensure_text("correlation_id", self.correlation_id, allow_empty=True, max_length=_MAX_ID_LENGTH))
        object.__setattr__(self, "message", _ensure_text("message", self.message, allow_empty=True, max_length=_MAX_JSON_STRING_LENGTH))
        object.__setattr__(self, "error_code", _ensure_text("error_code", self.error_code, allow_empty=True, max_length=_MAX_GENERIC_TEXT_LENGTH))
        object.__setattr__(self, "state", _normalize_json_mapping("state", self.state))
        object.__setattr__(self, "details", _normalize_json_mapping("details", self.details))

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SmartHomeCommandResult":
        """Build one command result from a plain dictionary."""

        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping.")
        return cls(
            command_id=payload["command_id"],
            provider=payload["provider"],
            entity_id=payload["entity_id"],
            command=payload["command"],
            status=payload["status"],
            observed_at=payload["observed_at"],
            correlation_id=payload.get("correlation_id", ""),
            message=payload.get("message", ""),
            error_code=payload.get("error_code", ""),
            state=payload.get("state", {}),
            details=payload.get("details", {}),
        )

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe command result payload."""

        return {
            "command_id": self.command_id,
            "provider": self.provider,
            "entity_id": self.entity_id,
            "command": self.command.value,
            "status": self.status.value,
            "observed_at": self.observed_at,
            "correlation_id": self.correlation_id,
            "message": self.message,
            "error_code": self.error_code,
            "state": _json_mapping_to_mutable(self.state),
            "details": _json_mapping_to_mutable(self.details),
        }


@dataclass(frozen=True, slots=True)
class SmartHomeEvent(_WireModel):
    """Describe one normalized smart-home event suitable for background use."""

    event_id: str
    provider: str
    entity_id: str
    event_kind: SmartHomeEventKind
    observed_at: str | datetime
    label: str = ""
    area: str = ""
    floor: str = ""
    device_id: str = ""
    correlation_id: str = ""
    sequence: int | None = None
    details: Mapping[str, FrozenJsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize identifiers and JSON-safe detail payloads."""

        object.__setattr__(self, "event_id", _ensure_text("event_id", self.event_id, max_length=_MAX_ID_LENGTH))
        object.__setattr__(self, "provider", _ensure_text("provider", self.provider, max_length=_MAX_PROVIDER_LENGTH))
        object.__setattr__(self, "entity_id", _ensure_text("entity_id", self.entity_id, max_length=_MAX_ID_LENGTH))
        object.__setattr__(self, "event_kind", SmartHomeEventKind(self.event_kind))
        object.__setattr__(self, "observed_at", _ensure_rfc3339_timestamp("observed_at", self.observed_at))
        object.__setattr__(self, "label", _ensure_text("label", self.label, allow_empty=True, max_length=_MAX_LABEL_LENGTH))
        object.__setattr__(self, "area", _ensure_text("area", self.area, allow_empty=True, max_length=_MAX_AREA_LENGTH))
        object.__setattr__(self, "floor", _ensure_text("floor", self.floor, allow_empty=True, max_length=_MAX_AREA_LENGTH))
        object.__setattr__(self, "device_id", _ensure_text("device_id", self.device_id, allow_empty=True, max_length=_MAX_ID_LENGTH))
        object.__setattr__(self, "correlation_id", _ensure_text("correlation_id", self.correlation_id, allow_empty=True, max_length=_MAX_ID_LENGTH))
        object.__setattr__(self, "sequence", _ensure_optional_non_negative_int("sequence", self.sequence))
        object.__setattr__(self, "details", _normalize_json_mapping("details", self.details))

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SmartHomeEvent":
        """Build one event from a plain dictionary."""

        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping.")
        return cls(
            event_id=payload["event_id"],
            provider=payload["provider"],
            entity_id=payload["entity_id"],
            event_kind=payload["event_kind"],
            observed_at=payload["observed_at"],
            label=payload.get("label", ""),
            area=payload.get("area", ""),
            floor=payload.get("floor", ""),
            device_id=payload.get("device_id", ""),
            correlation_id=payload.get("correlation_id", ""),
            sequence=payload.get("sequence"),
            details=payload.get("details", {}),
        )

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
            "floor": self.floor,
            "device_id": self.device_id,
            "correlation_id": self.correlation_id,
            "sequence": self.sequence,
            "details": _json_mapping_to_mutable(self.details),
        }


@dataclass(frozen=True, slots=True)
class SmartHomeEventBatch(_WireModel):
    """Return a bounded slice of smart-home stream events plus stream metadata."""

    events: tuple[SmartHomeEvent, ...] | Sequence[SmartHomeEvent]
    next_cursor: str | None = None
    stream_live: bool = True
    dropped_events: int = 0
    batch_observed_at: str | datetime | None = None

    def __post_init__(self) -> None:
        """Validate event collections and stream metadata."""

        if not isinstance(self.events, tuple):
            if isinstance(self.events, Iterable) and not isinstance(self.events, (str, bytes)):
                object.__setattr__(self, "events", tuple(self.events))
            else:
                raise TypeError("events must be a tuple or iterable of SmartHomeEvent items.")
        if len(self.events) > _MAX_EVENT_BATCH_EVENTS:
            raise ValueError(f"events must not contain more than {_MAX_EVENT_BATCH_EVENTS} items.")
        if not all(isinstance(event, SmartHomeEvent) for event in self.events):
            raise TypeError("events must contain SmartHomeEvent items only.")
        if self.next_cursor is not None:
            object.__setattr__(self, "next_cursor", _ensure_text("next_cursor", self.next_cursor, max_length=_MAX_CURSOR_LENGTH))
        object.__setattr__(self, "stream_live", _ensure_bool("stream_live", self.stream_live))
        object.__setattr__(self, "dropped_events", _ensure_non_negative_int("dropped_events", self.dropped_events))
        object.__setattr__(self, "batch_observed_at", _ensure_rfc3339_timestamp("batch_observed_at", self.batch_observed_at, allow_none=True))

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SmartHomeEventBatch":
        """Build one event batch from a plain dictionary."""

        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping.")
        raw_events = payload.get("events", ())
        if not isinstance(raw_events, Iterable) or isinstance(raw_events, (str, bytes)):
            raise TypeError("events must be an iterable.")
        events = tuple(
            event if isinstance(event, SmartHomeEvent) else SmartHomeEvent.from_dict(event)
            for event in raw_events
        )
        return cls(
            events=events,
            next_cursor=payload.get("next_cursor"),
            stream_live=payload.get("stream_live", True),
            dropped_events=payload.get("dropped_events", 0),
            batch_observed_at=payload.get("batch_observed_at"),
        )

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe event batch payload."""

        return {
            "events": [event.as_dict() for event in self.events],
            "next_cursor": self.next_cursor,
            "stream_live": self.stream_live,
            "dropped_events": self.dropped_events,
            "batch_observed_at": self.batch_observed_at,
        }


__all__ = [
    "SCHEMA_VERSION",
    "WIRE_PROTOCOL",
    "SmartHomeCapability",
    "SmartHomeCommand",
    "SmartHomeCommandRequest",
    "SmartHomeCommandResult",
    "SmartHomeCommandStatus",
    "SmartHomeEntity",
    "SmartHomeEntityAggregateField",
    "SmartHomeEntityClass",
    "SmartHomeEvent",
    "SmartHomeEventBatch",
    "SmartHomeEventAggregateField",
    "SmartHomeEventKind",
]