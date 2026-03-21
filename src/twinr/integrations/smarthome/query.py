"""Parse, filter, and aggregate generic smart-home queries.

This module keeps provider-neutral smart-home selection logic separate from the
integration adapter. It lets the agent query entities and event batches through
generic selectors, exact scalar state filters, and simple aggregations without
introducing hardcoded house-summary operations.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import math

from twinr.integrations.smarthome.models import (
    SmartHomeEntity,
    SmartHomeEntityAggregateField,
    SmartHomeEntityClass,
    SmartHomeEvent,
    SmartHomeEventAggregateField,
    SmartHomeEventKind,
)


def _normalize_text(value: object) -> str:
    """Return one stripped text representation."""

    if value is None:
        return ""
    return value.strip() if isinstance(value, str) else str(value).strip()


def _normalized_lookup_key(value: object) -> str:
    """Return one case-insensitive lookup key."""

    return _normalize_text(value).casefold()


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


def _parse_positive_int(value: object, *, field_name: str, default: int, maximum: int) -> int:
    """Parse one positive integer and clamp it to a maximum."""

    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive whole number.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive whole number.") from exc
    if parsed < 1:
        raise ValueError(f"{field_name} must be a positive whole number.")
    return min(maximum, parsed)


def _parse_offset_cursor(value: object) -> int:
    """Parse one list cursor as a non-negative offset."""

    if value is None:
        return 0
    if isinstance(value, bool):
        raise ValueError("cursor must be a non-negative whole number.")
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError("cursor must be a non-negative whole number.") from exc
    if parsed < 0:
        raise ValueError("cursor must be a non-negative whole number.")
    return parsed


def _parse_text_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    """Parse one text value or list of text values."""

    if value is None:
        return ()
    if isinstance(value, str):
        normalized = value.strip()
        return (normalized,) if normalized else ()
    if isinstance(value, (list, tuple)):
        collected: list[str] = []
        seen: set[str] = set()
        for index, item in enumerate(value):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"{field_name}[{index}] must be a non-empty string.")
            normalized = item.strip()
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
            return normalized
    raise ValueError(f"{field_name} must be a non-empty string, number, or boolean.")


def _parse_enum_tuple(value: object, *, enum_type, field_name: str) -> tuple[object, ...]:
    """Parse one enum value or list of enum values."""

    items = _parse_text_tuple(value, field_name=field_name)
    return tuple(dict.fromkeys(enum_type(item) for item in items))


def _path_value(mapping: Mapping[str, object], dotted_key: str) -> object:
    """Return one nested mapping value selected by a dotted path."""

    current: object = mapping
    for segment in dotted_key.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(segment)
    return current


def _scalar_values_equal(left: object, right: object) -> bool:
    """Return whether two scalar values match exactly after normalization."""

    if isinstance(right, bool):
        return isinstance(left, bool) and left is right
    if isinstance(right, (int, float)) and not isinstance(right, bool):
        if not isinstance(left, (int, float)) or isinstance(left, bool):
            return False
        return float(left) == float(right)
    if isinstance(right, str):
        return _normalized_lookup_key(left) == _normalized_lookup_key(right)
    return left == right


@dataclass(frozen=True, slots=True)
class SmartHomeStateFilter:
    """Describe one exact-match filter against an entity state payload."""

    key: str
    value: str | bool | int | float

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _normalize_text(self.key))
        if not self.key:
            raise ValueError("state filter key must not be empty.")


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


def parse_entity_query_parameters(
    params: Mapping[str, object],
    *,
    default_limit: int,
    maximum_limit: int,
) -> SmartHomeEntityQuery:
    """Parse request parameters into one entity query."""

    entity_ids = _parse_text_tuple(params.get("entity_ids", params.get("entity_id")), field_name="entity_ids")
    entity_class = _parse_text_tuple(params.get("entity_class"), field_name="entity_class")
    entity_classes = tuple(
        dict.fromkeys(
            (
                *(_parse_enum_tuple(params.get("entity_classes"), enum_type=SmartHomeEntityClass, field_name="entity_classes")),
                *(SmartHomeEntityClass(item) for item in entity_class),
            )
        )
    )
    state_filters = _parse_state_filters(params.get("state_filters"))
    return SmartHomeEntityQuery(
        entity_ids=entity_ids,
        entity_classes=entity_classes,
        providers=_parse_text_tuple(params.get("providers"), field_name="providers"),
        areas=_parse_text_tuple(params.get("areas"), field_name="areas"),
        online=_parse_optional_bool(params.get("online"), field_name="online"),
        controllable=_parse_optional_bool(params.get("controllable"), field_name="controllable"),
        readable=_parse_optional_bool(params.get("readable"), field_name="readable"),
        include_unavailable=_parse_bool(
            params.get("include_unavailable"),
            field_name="include_unavailable",
            default=False,
        ),
        state_filters=state_filters,
        aggregate_by=_parse_enum_tuple(
            params.get("aggregate_by"),
            enum_type=SmartHomeEntityAggregateField,
            field_name="aggregate_by",
        ),
        limit=_parse_positive_int(
            params.get("limit"),
            field_name="limit",
            default=default_limit,
            maximum=maximum_limit,
        ),
        cursor_offset=_parse_offset_cursor(params.get("cursor")),
    )


def parse_event_query_parameters(
    params: Mapping[str, object],
    *,
    default_limit: int,
    maximum_limit: int,
) -> SmartHomeEventQuery:
    """Parse request parameters into one event query."""

    return SmartHomeEventQuery(
        entity_ids=_parse_text_tuple(params.get("entity_ids", params.get("entity_id")), field_name="entity_ids"),
        event_kinds=_parse_enum_tuple(
            params.get("event_kinds"),
            enum_type=SmartHomeEventKind,
            field_name="event_kinds",
        ),
        providers=_parse_text_tuple(params.get("providers"), field_name="providers"),
        areas=_parse_text_tuple(params.get("areas"), field_name="areas"),
        aggregate_by=_parse_enum_tuple(
            params.get("aggregate_by"),
            enum_type=SmartHomeEventAggregateField,
            field_name="aggregate_by",
        ),
        limit=_parse_positive_int(
            params.get("limit"),
            field_name="limit",
            default=default_limit,
            maximum=maximum_limit,
        ),
        cursor=_normalize_text(params.get("cursor")) or None,
    )


def filter_entities(entities: list[SmartHomeEntity], query: SmartHomeEntityQuery) -> list[SmartHomeEntity]:
    """Return entities that match one parsed entity query."""

    requested_ids = set(query.entity_ids)
    requested_classes = set(query.entity_classes)
    provider_keys = {_normalized_lookup_key(item) for item in query.providers}
    area_keys = {_normalized_lookup_key(item) for item in query.areas}
    filtered: list[SmartHomeEntity] = []
    for entity in entities:
        if requested_ids and entity.entity_id not in requested_ids:
            continue
        if requested_classes and entity.entity_class not in requested_classes:
            continue
        if provider_keys and _normalized_lookup_key(entity.provider) not in provider_keys:
            continue
        if area_keys and _normalized_lookup_key(entity.area) not in area_keys:
            continue
        if query.online is None:
            if not query.include_unavailable and not entity.online:
                continue
        elif entity.online is not query.online:
            continue
        if query.controllable is not None and entity.controllable is not query.controllable:
            continue
        if query.readable is not None and entity.readable is not query.readable:
            continue
        if any(not _state_filter_matches(entity, state_filter) for state_filter in query.state_filters):
            continue
        filtered.append(entity)
    return filtered


def filter_events(events: tuple[SmartHomeEvent, ...], query: SmartHomeEventQuery) -> tuple[SmartHomeEvent, ...]:
    """Return stream events that match one parsed event query."""

    requested_ids = set(query.entity_ids)
    requested_kinds = set(query.event_kinds)
    provider_keys = {_normalized_lookup_key(item) for item in query.providers}
    area_keys = {_normalized_lookup_key(item) for item in query.areas}
    filtered: list[SmartHomeEvent] = []
    for event in events:
        if requested_ids and event.entity_id not in requested_ids:
            continue
        if requested_kinds and event.event_kind not in requested_kinds:
            continue
        if provider_keys and _normalized_lookup_key(event.provider) not in provider_keys:
            continue
        if area_keys and _normalized_lookup_key(event.area) not in area_keys:
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
    if not isinstance(value, (list, tuple)):
        raise ValueError("state_filters must be a list of filter objects.")
    collected: list[SmartHomeStateFilter] = []
    for index, item in enumerate(value):
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


def _state_filter_matches(entity: SmartHomeEntity, state_filter: SmartHomeStateFilter) -> bool:
    """Return whether one entity state matches one exact-match filter."""

    actual = _path_value(entity.state, state_filter.key)
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
