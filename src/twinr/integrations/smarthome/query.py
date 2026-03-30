# CHANGELOG: 2026-03-30
# BUG-1: Fixed exact numeric comparisons; large integers no longer collide via float coercion in state filters.
# BUG-2: Fixed whole-number parsing and parameter alias fallback; decimal inputs no longer truncate silently and singular aliases are no longer ignored when the plural key is null.
# SEC-1: Added hard caps for selector-list size, text length, cursor length, aggregate-field count, state-filter count, and path depth to mitigate practical resource-exhaustion attacks on Raspberry Pi 4 deployments.
# IMP-1: Added standards-aligned JSON Pointer state paths with list-index traversal while preserving legacy dotted-path filters for drop-in compatibility.
# IMP-2: Added compiled-query caches and precompiled state-filter paths to reduce repeated normalization/splitting overhead in hot loops.
"""Parse, filter, and aggregate generic smart-home queries.

This module keeps provider-neutral smart-home selection logic separate from the
integration adapter. It lets the agent query entities and event batches through
generic selectors, exact scalar state filters, and simple aggregations without
introducing hardcoded house-summary operations.

State-filter keys now support two path syntaxes:

* Legacy dotted paths, e.g. ``battery.level``.
* JSON Pointer (RFC 6901), e.g. ``/battery/level`` or ``/rooms/0/name``.

JSON Pointer is the preferred format when keys themselves may contain dots or
when list indexing is required. Legacy dotted paths remain supported for drop-in
compatibility.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
import math

from twinr.integrations.smarthome.models import (
    SmartHomeEntity,
    SmartHomeEntityAggregateField,
    SmartHomeEntityClass,
    SmartHomeEvent,
    SmartHomeEventAggregateField,
    SmartHomeEventKind,
)

# BREAKING: query selector sizes and path complexity are now bounded so hostile
# requests cannot monopolize CPU or memory on small edge devices.
_MAX_SELECTOR_ITEMS = 256
_MAX_TEXT_LENGTH = 256
_MAX_CURSOR_LENGTH = 1024
_MAX_AGGREGATE_FIELDS = 8
_MAX_STATE_FILTERS = 32
_MAX_STATE_FILTER_KEY_LENGTH = 256
_MAX_STATE_FILTER_PATH_DEPTH = 32
_MAX_PATH_SEGMENT_LENGTH = 128


def _normalize_text(value: object) -> str:
    """Return one stripped text representation."""

    if value is None:
        return ""
    return value.strip() if isinstance(value, str) else str(value).strip()


def _normalized_lookup_key(value: object) -> str:
    """Return one case-insensitive lookup key."""

    return _normalize_text(value).casefold()


def _parameter_value(params: Mapping[str, object], *keys: str) -> object:
    """Return the first non-null parameter value across aliases."""

    for key in keys:
        if key in params:
            value = params.get(key)
            if value is not None:
                return value
    return None


def _parse_optional_bool(value: object, *, field_name: str) -> bool | None:
    """Parse one optional boolean value."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{field_name} must be a boolean value.")


def _parse_bool(value: object, *, field_name: str, default: bool) -> bool:
    """Parse one boolean value with a default."""

    parsed = _parse_optional_bool(value, field_name=field_name)
    return default if parsed is None else parsed


def _parse_whole_number_string(
    value: str,
    *,
    field_name: str,
    requirement: str,
) -> int:
    """Parse one whole number from text."""

    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must be {requirement}.")
    if stripped.startswith("+"):
        digits = stripped[1:]
    else:
        digits = stripped
    if not digits.isdigit():
        raise ValueError(f"{field_name} must be {requirement}.")
    return int(stripped)


def _parse_positive_int(value: object, *, field_name: str, default: int, maximum: int) -> int:
    """Parse one positive integer and clamp it to a maximum."""

    requirement = "a positive whole number"
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be {requirement}.")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        parsed = _parse_whole_number_string(
            value,
            field_name=field_name,
            requirement=requirement,
        )
    else:
        raise ValueError(f"{field_name} must be {requirement}.")
    if parsed < 1:
        raise ValueError(f"{field_name} must be {requirement}.")
    return min(maximum, parsed)


def _parse_offset_cursor(value: object) -> int:
    """Parse one list cursor as a non-negative offset."""

    requirement = "a non-negative whole number"
    if value is None:
        return 0
    if isinstance(value, bool):
        raise ValueError(f"cursor must be {requirement}.")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        parsed = _parse_whole_number_string(
            value,
            field_name="cursor",
            requirement=requirement,
        )
    else:
        raise ValueError(f"cursor must be {requirement}.")
    if parsed < 0:
        raise ValueError(f"cursor must be {requirement}.")
    return parsed


def _parse_optional_text(
    value: object,
    *,
    field_name: str,
    maximum_length: int,
) -> str | None:
    """Parse one optional textual value."""

    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a string.")
    normalized = _normalize_text(value)
    if not normalized:
        return None
    if len(normalized) > maximum_length:
        raise ValueError(f"{field_name} must not exceed {maximum_length} characters.")
    return normalized


def _parse_text_tuple(
    value: object,
    *,
    field_name: str,
    maximum_items: int = _MAX_SELECTOR_ITEMS,
    maximum_item_length: int = _MAX_TEXT_LENGTH,
) -> tuple[str, ...]:
    """Parse one text value or list of text values."""

    if value is None:
        return ()
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return ()
        if len(normalized) > maximum_item_length:
            raise ValueError(f"{field_name} entries must not exceed {maximum_item_length} characters.")
        return (normalized,)
    if isinstance(value, (list, tuple)):
        if len(value) > maximum_items:
            raise ValueError(f"{field_name} must not contain more than {maximum_items} items.")
        collected: list[str] = []
        seen: set[str] = set()
        for index, item in enumerate(value):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"{field_name}[{index}] must be a non-empty string.")
            normalized = item.strip()
            if len(normalized) > maximum_item_length:
                raise ValueError(
                    f"{field_name}[{index}] must not exceed {maximum_item_length} characters."
                )
            marker = normalized.casefold()
            if marker in seen:
                continue
            seen.add(marker)
            collected.append(normalized)
        return tuple(collected)
    raise ValueError(f"{field_name} must be a string, list, tuple, or null.")


def _parse_scalar_filter_value(value: object, *, field_name: str) -> str | bool | int | float:
    """Parse one scalar state-filter value."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be a finite number.")
        return value
    if isinstance(value, str):
        normalized = value.strip()
        if normalized:
            if len(normalized) > _MAX_TEXT_LENGTH:
                raise ValueError(f"{field_name} must not exceed {_MAX_TEXT_LENGTH} characters.")
            return normalized
    raise ValueError(f"{field_name} must be a non-empty string, number, or boolean.")


def _parse_enum_tuple(
    value: object,
    *,
    enum_type,
    field_name: str,
    maximum_items: int = _MAX_SELECTOR_ITEMS,
) -> tuple[object, ...]:
    """Parse one enum value or list of enum values."""

    items = _parse_text_tuple(
        value,
        field_name=field_name,
        maximum_items=maximum_items,
    )
    parsed = tuple(dict.fromkeys(enum_type(item) for item in items))
    return parsed


def _is_sequence_container(value: object) -> bool:
    """Return whether one object should be treated as an indexable sequence."""

    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _decode_json_pointer_segment(segment: str) -> str:
    """Decode one RFC 6901 JSON Pointer segment."""

    if "~" not in segment:
        return segment
    decoded: list[str] = []
    index = 0
    while index < len(segment):
        character = segment[index]
        if character != "~":
            decoded.append(character)
            index += 1
            continue
        if index + 1 >= len(segment):
            raise ValueError("state filter key contains an invalid JSON Pointer escape.")
        escaped = segment[index + 1]
        if escaped == "0":
            decoded.append("~")
        elif escaped == "1":
            decoded.append("/")
        else:
            raise ValueError("state filter key contains an invalid JSON Pointer escape.")
        index += 2
    return "".join(decoded)


@lru_cache(maxsize=1024)
def _compile_state_path(key: str) -> tuple[str, ...]:
    """Compile one state-filter key into path segments."""

    if len(key) > _MAX_STATE_FILTER_KEY_LENGTH:
        raise ValueError(
            f"state filter key must not exceed {_MAX_STATE_FILTER_KEY_LENGTH} characters."
        )
    if key.startswith("/"):
        segments = tuple(_decode_json_pointer_segment(segment) for segment in key.split("/")[1:])
    else:
        segments = tuple(segment.strip() for segment in key.split("."))
        if any(not segment for segment in segments):
            raise ValueError("state filter key contains an empty dotted path segment.")
    if not segments:
        raise ValueError("state filter key must not be empty.")
    if len(segments) > _MAX_STATE_FILTER_PATH_DEPTH:
        raise ValueError(
            f"state filter path depth must not exceed {_MAX_STATE_FILTER_PATH_DEPTH} segments."
        )
    for index, segment in enumerate(segments):
        if len(segment) > _MAX_PATH_SEGMENT_LENGTH:
            raise ValueError(
                f"state filter segment {index} must not exceed {_MAX_PATH_SEGMENT_LENGTH} characters."
            )
    return segments


def _path_value(container: object, path_segments: tuple[str, ...]) -> object:
    """Return one nested value selected by compiled path segments."""

    current = container
    for segment in path_segments:
        if isinstance(current, Mapping):
            current = current.get(segment)
            continue
        if _is_sequence_container(current):
            if not segment.isdigit():
                return None
            index = int(segment)
            if index >= len(current):
                return None
            current = current[index]
            continue
        return None
    return current


def _scalar_values_equal(left: object, right: object) -> bool:
    """Return whether two scalar values match exactly after normalization."""

    if isinstance(right, bool):
        return isinstance(left, bool) and left is right
    if isinstance(right, (int, float)) and not isinstance(right, bool):
        if not isinstance(left, (int, float)) or isinstance(left, bool):
            return False
        return left == right
    if isinstance(right, str):
        return _normalized_lookup_key(left) == _normalized_lookup_key(right)
    return left == right


@dataclass(frozen=True, slots=True)
class SmartHomeStateFilter:
    """Describe one exact-match filter against an entity state payload."""

    key: str
    value: str | bool | int | float

    def __post_init__(self) -> None:
        normalized_key = _normalize_text(self.key)
        if not normalized_key:
            raise ValueError("state filter key must not be empty.")
        _compile_state_path(normalized_key)
        object.__setattr__(self, "key", normalized_key)


@dataclass(frozen=True, slots=True)
class SmartHomeEntityQuery:
    """Describe one generic smart-home entity query."""

    entity_ids: tuple[str, ...] = ()
    entity_classes: tuple[SmartHomeEntityClass, ...] = ()
    providers: tuple[str, ...] = ()
    areas: tuple[str, ...] = ()
    online: bool | None = None
    controllable: bool | None = None
    readable: bool | None = None
    include_unavailable: bool = False
    state_filters: tuple[SmartHomeStateFilter, ...] = ()
    aggregate_by: tuple[SmartHomeEntityAggregateField, ...] = ()
    limit: int = 32
    cursor_offset: int = 0


@dataclass(frozen=True, slots=True)
class SmartHomeEventQuery:
    """Describe one generic smart-home event query."""

    entity_ids: tuple[str, ...] = ()
    event_kinds: tuple[SmartHomeEventKind, ...] = ()
    providers: tuple[str, ...] = ()
    areas: tuple[str, ...] = ()
    aggregate_by: tuple[SmartHomeEventAggregateField, ...] = ()
    limit: int = 20
    cursor: str | None = None


@dataclass(frozen=True, slots=True)
class _CompiledStateFilter:
    """Describe one compiled state filter for hot-loop matching."""

    path_segments: tuple[str, ...]
    value: str | bool | int | float


@dataclass(frozen=True, slots=True)
class _CompiledEntityQuery:
    """Describe one compiled entity query for hot-loop matching."""

    requested_ids: frozenset[str]
    requested_classes: frozenset[SmartHomeEntityClass]
    provider_keys: frozenset[str]
    area_keys: frozenset[str]
    online: bool | None
    controllable: bool | None
    readable: bool | None
    include_unavailable: bool
    state_filters: tuple[_CompiledStateFilter, ...]


@dataclass(frozen=True, slots=True)
class _CompiledEventQuery:
    """Describe one compiled event query for hot-loop matching."""

    requested_ids: frozenset[str]
    requested_kinds: frozenset[SmartHomeEventKind]
    provider_keys: frozenset[str]
    area_keys: frozenset[str]


@lru_cache(maxsize=512)
def _compile_entity_query(query: SmartHomeEntityQuery) -> _CompiledEntityQuery:
    """Compile one parsed entity query into hot-loop structures."""

    return _CompiledEntityQuery(
        requested_ids=frozenset(query.entity_ids),
        requested_classes=frozenset(query.entity_classes),
        provider_keys=frozenset(_normalized_lookup_key(item) for item in query.providers),
        area_keys=frozenset(_normalized_lookup_key(item) for item in query.areas),
        online=query.online,
        controllable=query.controllable,
        readable=query.readable,
        include_unavailable=query.include_unavailable,
        state_filters=tuple(
            _CompiledStateFilter(
                path_segments=_compile_state_path(state_filter.key),
                value=state_filter.value,
            )
            for state_filter in query.state_filters
        ),
    )


@lru_cache(maxsize=512)
def _compile_event_query(query: SmartHomeEventQuery) -> _CompiledEventQuery:
    """Compile one parsed event query into hot-loop structures."""

    return _CompiledEventQuery(
        requested_ids=frozenset(query.entity_ids),
        requested_kinds=frozenset(query.event_kinds),
        provider_keys=frozenset(_normalized_lookup_key(item) for item in query.providers),
        area_keys=frozenset(_normalized_lookup_key(item) for item in query.areas),
    )


def parse_entity_query_parameters(
    params: Mapping[str, object],
    *,
    default_limit: int,
    maximum_limit: int,
) -> SmartHomeEntityQuery:
    """Parse request parameters into one entity query."""

    entity_ids = _parse_text_tuple(
        _parameter_value(params, "entity_ids", "entity_id"),
        field_name="entity_ids",
    )
    entity_class = _parse_text_tuple(
        _parameter_value(params, "entity_class"),
        field_name="entity_class",
    )
    entity_classes = tuple(
        dict.fromkeys(
            (
                *(
                    _parse_enum_tuple(
                        _parameter_value(params, "entity_classes"),
                        enum_type=SmartHomeEntityClass,
                        field_name="entity_classes",
                    )
                ),
                *(SmartHomeEntityClass(item) for item in entity_class),
            )
        )
    )
    state_filters = _parse_state_filters(_parameter_value(params, "state_filters", "state_filter"))
    aggregate_by = _parse_enum_tuple(
        _parameter_value(params, "aggregate_by"),
        enum_type=SmartHomeEntityAggregateField,
        field_name="aggregate_by",
        maximum_items=_MAX_AGGREGATE_FIELDS,
    )
    return SmartHomeEntityQuery(
        entity_ids=entity_ids,
        entity_classes=entity_classes,
        providers=_parse_text_tuple(
            _parameter_value(params, "providers", "provider"),
            field_name="providers",
        ),
        areas=_parse_text_tuple(
            _parameter_value(params, "areas", "area"),
            field_name="areas",
        ),
        online=_parse_optional_bool(_parameter_value(params, "online"), field_name="online"),
        controllable=_parse_optional_bool(
            _parameter_value(params, "controllable"),
            field_name="controllable",
        ),
        readable=_parse_optional_bool(
            _parameter_value(params, "readable"),
            field_name="readable",
        ),
        include_unavailable=_parse_bool(
            _parameter_value(params, "include_unavailable"),
            field_name="include_unavailable",
            default=False,
        ),
        state_filters=state_filters,
        aggregate_by=aggregate_by,
        limit=_parse_positive_int(
            _parameter_value(params, "limit"),
            field_name="limit",
            default=default_limit,
            maximum=maximum_limit,
        ),
        cursor_offset=_parse_offset_cursor(_parameter_value(params, "cursor")),
    )


def parse_event_query_parameters(
    params: Mapping[str, object],
    *,
    default_limit: int,
    maximum_limit: int,
) -> SmartHomeEventQuery:
    """Parse request parameters into one event query."""

    aggregate_by = _parse_enum_tuple(
        _parameter_value(params, "aggregate_by"),
        enum_type=SmartHomeEventAggregateField,
        field_name="aggregate_by",
        maximum_items=_MAX_AGGREGATE_FIELDS,
    )
    return SmartHomeEventQuery(
        entity_ids=_parse_text_tuple(
            _parameter_value(params, "entity_ids", "entity_id"),
            field_name="entity_ids",
        ),
        event_kinds=_parse_enum_tuple(
            _parameter_value(params, "event_kinds", "event_kind"),
            enum_type=SmartHomeEventKind,
            field_name="event_kinds",
        ),
        providers=_parse_text_tuple(
            _parameter_value(params, "providers", "provider"),
            field_name="providers",
        ),
        areas=_parse_text_tuple(
            _parameter_value(params, "areas", "area"),
            field_name="areas",
        ),
        aggregate_by=aggregate_by,
        limit=_parse_positive_int(
            _parameter_value(params, "limit"),
            field_name="limit",
            default=default_limit,
            maximum=maximum_limit,
        ),
        cursor=_parse_optional_text(
            _parameter_value(params, "cursor"),
            field_name="cursor",
            maximum_length=_MAX_CURSOR_LENGTH,
        ),
    )


def filter_entities(entities: list[SmartHomeEntity], query: SmartHomeEntityQuery) -> list[SmartHomeEntity]:
    """Return entities that match one parsed entity query."""

    compiled = _compile_entity_query(query)
    filtered: list[SmartHomeEntity] = []
    for entity in entities:
        if compiled.requested_ids and entity.entity_id not in compiled.requested_ids:
            continue
        if compiled.requested_classes and entity.entity_class not in compiled.requested_classes:
            continue
        if compiled.provider_keys and _normalized_lookup_key(entity.provider) not in compiled.provider_keys:
            continue
        if compiled.area_keys and _normalized_lookup_key(entity.area) not in compiled.area_keys:
            continue
        if compiled.online is None:
            if not compiled.include_unavailable and not entity.online:
                continue
        elif entity.online is not compiled.online:
            continue
        if compiled.controllable is not None and entity.controllable is not compiled.controllable:
            continue
        if compiled.readable is not None and entity.readable is not compiled.readable:
            continue
        if any(not _state_filter_matches(entity, state_filter) for state_filter in compiled.state_filters):
            continue
        filtered.append(entity)
    return filtered


def filter_events(events: tuple[SmartHomeEvent, ...], query: SmartHomeEventQuery) -> tuple[SmartHomeEvent, ...]:
    """Return stream events that match one parsed event query."""

    compiled = _compile_event_query(query)
    filtered: list[SmartHomeEvent] = []
    for event in events:
        if compiled.requested_ids and event.entity_id not in compiled.requested_ids:
            continue
        if compiled.requested_kinds and event.event_kind not in compiled.requested_kinds:
            continue
        if compiled.provider_keys and _normalized_lookup_key(event.provider) not in compiled.provider_keys:
            continue
        if compiled.area_keys and _normalized_lookup_key(event.area) not in compiled.area_keys:
            continue
        filtered.append(event)
    return tuple(filtered)


def paginate_entities(
    entities: list[SmartHomeEntity],
    query: SmartHomeEntityQuery,
) -> tuple[list[SmartHomeEntity], str | None]:
    """Apply offset pagination to filtered entities."""

    if query.cursor_offset >= len(entities):
        return [], None
    start = query.cursor_offset
    end = min(len(entities), start + query.limit)
    next_cursor = str(end) if end < len(entities) else None
    return entities[start:end], next_cursor


def aggregate_entities(
    entities: list[SmartHomeEntity],
    fields: tuple[SmartHomeEntityAggregateField, ...],
) -> list[dict[str, object]]:
    """Return generic aggregate counts over entity fields."""

    return _aggregate_values(
        values=(
            (field.value, _entity_aggregate_value(entity, field))
            for field in fields
            for entity in entities
        )
    )


def aggregate_events(
    events: tuple[SmartHomeEvent, ...],
    fields: tuple[SmartHomeEventAggregateField, ...],
) -> list[dict[str, object]]:
    """Return generic aggregate counts over event fields."""

    return _aggregate_values(
        values=(
            (field.value, _event_aggregate_value(event, field))
            for field in fields
            for event in events
        )
    )


def entity_query_filters_payload(query: SmartHomeEntityQuery) -> dict[str, object]:
    """Render one entity-query filter payload for result metadata."""

    payload: dict[str, object] = {}
    if query.entity_ids:
        payload["entity_ids"] = list(query.entity_ids)
    if query.entity_classes:
        payload["entity_classes"] = [item.value for item in query.entity_classes]
    if query.providers:
        payload["providers"] = list(query.providers)
    if query.areas:
        payload["areas"] = list(query.areas)
    if query.online is not None:
        payload["online"] = query.online
    if query.controllable is not None:
        payload["controllable"] = query.controllable
    if query.readable is not None:
        payload["readable"] = query.readable
    if query.include_unavailable:
        payload["include_unavailable"] = True
    if query.state_filters:
        payload["state_filters"] = [
            {"key": state_filter.key, "value": state_filter.value}
            for state_filter in query.state_filters
        ]
    if query.aggregate_by:
        payload["aggregate_by"] = [item.value for item in query.aggregate_by]
    payload["limit"] = query.limit
    payload["cursor_offset"] = query.cursor_offset
    return payload


def event_query_filters_payload(query: SmartHomeEventQuery) -> dict[str, object]:
    """Render one event-query filter payload for result metadata."""

    payload: dict[str, object] = {}
    if query.entity_ids:
        payload["entity_ids"] = list(query.entity_ids)
    if query.event_kinds:
        payload["event_kinds"] = [item.value for item in query.event_kinds]
    if query.providers:
        payload["providers"] = list(query.providers)
    if query.areas:
        payload["areas"] = list(query.areas)
    if query.aggregate_by:
        payload["aggregate_by"] = [item.value for item in query.aggregate_by]
    payload["limit"] = query.limit
    if query.cursor is not None:
        payload["cursor"] = query.cursor
    return payload


def _parse_state_filters(value: object) -> tuple[SmartHomeStateFilter, ...]:
    """Parse one list of exact-match state filters."""

    if value is None:
        return ()
    if isinstance(value, Mapping):
        items = (value,)
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        raise ValueError("state_filters must be one filter object or a list of filter objects.")
    if len(items) > _MAX_STATE_FILTERS:
        raise ValueError(f"state_filters must not contain more than {_MAX_STATE_FILTERS} items.")
    collected: list[SmartHomeStateFilter] = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError(f"state_filters[{index}] must be an object.")
        key = _normalize_text(item.get("key"))
        if not key:
            raise ValueError(f"state_filters[{index}].key must be a non-empty string.")
        if "value" not in item:
            raise ValueError(f"state_filters[{index}].value must be provided.")
        collected.append(
            SmartHomeStateFilter(
                key=key,
                value=_parse_scalar_filter_value(
                    item.get("value"),
                    field_name=f"state_filters[{index}].value",
                ),
            )
        )
    return tuple(collected)


def _state_filter_matches(entity: SmartHomeEntity, state_filter: _CompiledStateFilter) -> bool:
    """Return whether one entity state matches one exact-match filter."""

    actual = _path_value(entity.state, state_filter.path_segments)
    return _scalar_values_equal(actual, state_filter.value)


def _entity_aggregate_value(entity: SmartHomeEntity, field: SmartHomeEntityAggregateField) -> object:
    """Return one aggregate bucket value for an entity."""

    if field is SmartHomeEntityAggregateField.ENTITY_CLASS:
        return entity.entity_class.value
    if field is SmartHomeEntityAggregateField.AREA:
        return entity.area
    if field is SmartHomeEntityAggregateField.PROVIDER:
        return entity.provider
    if field is SmartHomeEntityAggregateField.ONLINE:
        return entity.online
    if field is SmartHomeEntityAggregateField.CONTROLLABLE:
        return entity.controllable
    if field is SmartHomeEntityAggregateField.READABLE:
        return entity.readable
    return ""


def _event_aggregate_value(event: SmartHomeEvent, field: SmartHomeEventAggregateField) -> object:
    """Return one aggregate bucket value for an event."""

    if field is SmartHomeEventAggregateField.EVENT_KIND:
        return event.event_kind.value
    if field is SmartHomeEventAggregateField.AREA:
        return event.area
    if field is SmartHomeEventAggregateField.PROVIDER:
        return event.provider
    if field is SmartHomeEventAggregateField.ENTITY_ID:
        return event.entity_id
    return ""


def _aggregate_values(values) -> list[dict[str, object]]:
    """Aggregate ``(field, value)`` pairs into deterministic JSON payloads."""

    counter: Counter[tuple[str, object]] = Counter(values)
    return [
        {"field": field, "value": value, "count": count}
        for field, value, count in sorted(
            ((field, value, count) for (field, value), count in counter.items()),
            key=lambda item: (item[0], _normalized_lookup_key(item[1]), item[2]),
        )
    ]


__all__ = [
    "SmartHomeEntityQuery",
    "SmartHomeEventQuery",
    "SmartHomeStateFilter",
    "aggregate_entities",
    "aggregate_events",
    "entity_query_filters_payload",
    "event_query_filters_payload",
    "filter_entities",
    "filter_events",
    "paginate_entities",
    "parse_entity_query_parameters",
    "parse_event_query_parameters",
]