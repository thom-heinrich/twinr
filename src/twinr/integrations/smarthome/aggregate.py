# CHANGELOG: 2026-03-30
# BUG-1: Fixed read_sensor_stream pagination correctness. The old equal-slice fan-out could
#        drop globally newer events and permanently skip unseen child events by advancing child
#        cursors without buffering leftovers.
# BUG-2: Fixed read_sensor_stream(limit=0) consuming child events and returning an empty page.
#        limit is now strictly validated and bounded.
# BUG-3: Fixed control-result target rewriting for tuple/sequence payloads; the previous code
#        only rewrote list payloads and could silently leak unqualified child entity IDs.
# BUG-4: Fixed routed ID construction/parsing to validate empty/control-character/oversized source IDs.
#        The previous builders could emit malformed IDs and the decoders trusted unvalidated input.
# BUG-5: Fixed label disambiguation for entities without an area; the previous implementation could
#        crash on None areas.
# SEC-1: Added strict size bounds for cursors and routed identifiers to prevent practical memory/CPU
#        abuse on Raspberry Pi deployments.
# SEC-2: Added opaque cursor encoding with optional HMAC signing (cursor_secret or
#        TWINR_SMARTHOME_CURSOR_SECRET) so callers no longer need to depend on cursor internals.
# IMP-1: Added bounded parallel fan-out for list/control/initial stream reads using ThreadPoolExecutor,
#        which is better suited to multi-bridge network I/O than serialized calls.
# IMP-2: Added correct heap-based multi-route stream merge with buffered leftovers and opaque v2 cursors,
#        matching frontier event-feed patterns that require deterministic global ordering.
# IMP-3: Added public_route_id + display_name so stable external IDs and user-facing labels can be
#        decoupled from mutable transport details such as bridge hosts/IPs.
# IMP-4: Added parse_routed_event_id() for API symmetry with build_routed_event_id().

"""Aggregate multiple routed smart-home providers behind one bounded surface.

This module keeps multi-bridge or other multi-route wiring out of provider
packages. It rewrites child entity and event identifiers into route-qualified
IDs, dispatches read/control calls back to the correct child provider, and
combines bounded sensor-stream batches into one generic Twinr stream.

2026 upgrade notes:
- External/public routed IDs can now use ``public_route_id`` while dispatch
  continues to use ``route_id`` internally.
- Aggregate sensor cursors are now opaque v2 tokens and may contain buffered
  unseen events so pagination stays correct across routes.
- # BREAKING: ``read_sensor_stream(limit=...)`` is now validated against
  ``max_read_limit`` (default: 256) to keep latency/memory bounded on Pi-class
  hardware.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, fields, is_dataclass, replace
from datetime import date, datetime, time
from enum import Enum
from functools import lru_cache
import base64
import binascii
import hashlib
import hmac
import heapq
import json
import os
import sys
from types import UnionType
import typing
from typing import Any, get_args, get_origin, get_type_hints
from urllib.parse import quote, unquote

from twinr.integrations.smarthome.adapter import SmartHomeController, SmartHomeEntityProvider, SmartHomeSensorStream
from twinr.integrations.smarthome.models import SmartHomeCommand, SmartHomeEntity, SmartHomeEntityClass, SmartHomeEvent, SmartHomeEventBatch

_ROUTED_ENTITY_PREFIX = "route:"
_ROUTED_EVENT_PREFIX = "route-event:"
_CURSOR_VERSION = 2
_LEGACY_CURSOR_VERSION = 1
_CURSOR_TOKEN_PREFIX = "smarthome-route-cursor:"
_CURSOR_SECRET_ENV = "TWINR_SMARTHOME_CURSOR_SECRET"

_DEFAULT_MAX_CONCURRENCY = 4
_DEFAULT_SENSOR_PAGE_SIZE = 32
_DEFAULT_MAX_READ_LIMIT = 256
_DEFAULT_MAX_CURSOR_BYTES = 65536
_MAX_ROUTE_ID_LENGTH = 256
_MAX_SOURCE_ID_LENGTH = 2048
_MAX_DISPLAY_NAME_LENGTH = 128


def _normalize_identifier(
    value: object,
    *,
    label: str,
    max_length: int,
) -> str:
    """Return one stripped identifier after strict validation."""

    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} must not be empty.")
    if len(normalized) > max_length:
        raise ValueError(f"{label} must be at most {max_length} characters.")
    if any(ord(character) < 32 or ord(character) == 127 for character in normalized):
        raise ValueError(f"{label} must not contain control characters.")
    return normalized


def _normalize_route_id(route_id: object) -> str:
    """Return one stripped internal route identifier."""

    return _normalize_identifier(route_id, label="route_id", max_length=_MAX_ROUTE_ID_LENGTH)


def _normalize_public_route_id(route_id: object) -> str:
    """Return one stripped external/public route identifier."""

    return _normalize_identifier(route_id, label="public_route_id", max_length=_MAX_ROUTE_ID_LENGTH)


def _normalize_source_id(value: object, *, label: str) -> str:
    """Return one stripped child entity/event identifier."""

    return _normalize_identifier(value, label=label, max_length=_MAX_SOURCE_ID_LENGTH)


def _normalize_display_name(value: object) -> str:
    """Return one stripped human-facing route display name."""

    return _normalize_identifier(value, label="display_name", max_length=_MAX_DISPLAY_NAME_LENGTH)


def _encode_routed_id(prefix: str, route_id: str, source_id: str) -> str:
    """Return one route-qualified identifier safe for generic transport."""

    return f"{prefix}{quote(route_id, safe='')}:{quote(source_id, safe='')}"


def _decode_routed_id(value: str, *, prefix: str, label: str) -> tuple[str, str]:
    """Parse one route-qualified entity or event identifier."""

    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    if not value.startswith(prefix):
        raise ValueError(f"{label} must be a routed smart-home identifier.")
    route_token, separator, source_token = value[len(prefix) :].partition(":")
    if separator != ":" or not route_token or not source_token:
        raise ValueError(f"{label} must be a routed smart-home identifier.")
    route_id = _normalize_public_route_id(unquote(route_token))
    source_id = _normalize_source_id(unquote(source_token), label=f"{label} source identifier")
    return route_id, source_id


def _casefold_or_empty(value: object) -> str:
    """Return a casefolded string or an empty fallback."""

    return value.casefold() if isinstance(value, str) else ""


def _event_sort_key(event: SmartHomeEvent) -> tuple[object, str]:
    """Return the deterministic ordering key for one event."""

    return event.observed_at, event.event_id


def _urlsafe_b64encode(value: bytes) -> str:
    """Return one URL-safe base64 token without trailing padding."""

    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _urlsafe_b64decode(value: str) -> bytes:
    """Decode one URL-safe base64 token that may omit padding."""

    padding = "=" * ((4 - (len(value) % 4)) % 4)
    try:
        return base64.urlsafe_b64decode(value + padding)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("cursor must be a valid smart-home route cursor.") from exc


@lru_cache(maxsize=None)
def _dataclass_type_hints(model_type: type[object]) -> dict[str, object]:
    """Return resolved type hints for one dataclass model."""

    module = sys.modules.get(model_type.__module__)
    module_globals = vars(module) if module is not None else {}
    return get_type_hints(model_type, globalns=module_globals, localns=module_globals)


def _encode_cursor_value(value: object) -> object:
    """Convert one arbitrary value into a JSON-safe cursor payload."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return _encode_cursor_value(value.value)
    if isinstance(value, datetime):
        return {"__t": "datetime", "v": value.isoformat()}
    if isinstance(value, date):
        return {"__t": "date", "v": value.isoformat()}
    if isinstance(value, time):
        return {"__t": "time", "v": value.isoformat()}
    if isinstance(value, bytes):
        return {"__t": "bytes", "v": _urlsafe_b64encode(value)}
    if is_dataclass(value):
        return {
            field_definition.name: _encode_cursor_value(getattr(value, field_definition.name))
            for field_definition in fields(value)
        }
    if isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value):
            return {str(key): _encode_cursor_value(item) for key, item in value.items()}
        return {
            "__t": "mapping",
            "v": [
                [_encode_cursor_value(key), _encode_cursor_value(item)]
                for key, item in value.items()
            ],
        }
    if isinstance(value, tuple):
        return {"__t": "tuple", "v": [_encode_cursor_value(item) for item in value]}
    if isinstance(value, list):
        return [_encode_cursor_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return {"__t": "set", "v": [_encode_cursor_value(item) for item in value]}
    return str(value)


def _decode_cursor_value_untyped(value: object) -> object:
    """Decode one JSON-safe cursor value without relying on static typing."""

    if isinstance(value, list):
        return [_decode_cursor_value_untyped(item) for item in value]
    if isinstance(value, Mapping):
        tag = value.get("__t")
        if tag == "datetime":
            return datetime.fromisoformat(str(value["v"]))
        if tag == "date":
            return date.fromisoformat(str(value["v"]))
        if tag == "time":
            return time.fromisoformat(str(value["v"]))
        if tag == "bytes":
            return _urlsafe_b64decode(str(value["v"]))
        if tag == "tuple":
            items = value.get("v", ())
            if not isinstance(items, list):
                raise ValueError("cursor must be a valid smart-home route cursor.")
            return tuple(_decode_cursor_value_untyped(item) for item in items)
        if tag == "set":
            items = value.get("v", ())
            if not isinstance(items, list):
                raise ValueError("cursor must be a valid smart-home route cursor.")
            return {_decode_cursor_value_untyped(item) for item in items}
        if tag == "mapping":
            items = value.get("v", ())
            if not isinstance(items, list):
                raise ValueError("cursor must be a valid smart-home route cursor.")
            return {
                _decode_cursor_value_untyped(key): _decode_cursor_value_untyped(item)
                for key, item in items
            }
        return {
            str(key): _decode_cursor_value_untyped(item)
            for key, item in value.items()
        }
    return value


def _decode_cursor_value(value: object, expected_type: object = Any) -> object:
    """Decode one cursor value using dataclass type information when available."""

    if expected_type in (Any, object):
        return _decode_cursor_value_untyped(value)

    origin = get_origin(expected_type)
    if origin in (UnionType, typing.Union):
        arguments = get_args(expected_type)
        if value is None and type(None) in arguments:
            return None
        for argument in arguments:
            if argument is type(None):
                continue
            try:
                return _decode_cursor_value(value, argument)
            except (TypeError, ValueError, KeyError):
                continue
        return _decode_cursor_value_untyped(value)

    if isinstance(expected_type, type) and issubclass(expected_type, Enum):
        raw_value = _decode_cursor_value_untyped(value)
        return expected_type(raw_value)

    if expected_type is datetime:
        raw_value = _decode_cursor_value_untyped(value)
        if isinstance(raw_value, datetime):
            return raw_value
        if isinstance(raw_value, str):
            return datetime.fromisoformat(raw_value)
        raise ValueError("cursor must be a valid smart-home route cursor.")

    if expected_type is date:
        raw_value = _decode_cursor_value_untyped(value)
        if isinstance(raw_value, date) and not isinstance(raw_value, datetime):
            return raw_value
        if isinstance(raw_value, str):
            return date.fromisoformat(raw_value)
        raise ValueError("cursor must be a valid smart-home route cursor.")

    if expected_type is time:
        raw_value = _decode_cursor_value_untyped(value)
        if isinstance(raw_value, time):
            return raw_value
        if isinstance(raw_value, str):
            return time.fromisoformat(raw_value)
        raise ValueError("cursor must be a valid smart-home route cursor.")

    if expected_type is bytes:
        raw_value = _decode_cursor_value_untyped(value)
        if isinstance(raw_value, bytes):
            return raw_value
        if isinstance(raw_value, str):
            return _urlsafe_b64decode(raw_value)
        raise ValueError("cursor must be a valid smart-home route cursor.")

    if origin in (list, Sequence):
        item_type = get_args(expected_type)[0] if get_args(expected_type) else Any
        if not isinstance(value, list):
            raise ValueError("cursor must be a valid smart-home route cursor.")
        return [_decode_cursor_value(item, item_type) for item in value]

    if origin is tuple:
        arguments = get_args(expected_type)
        tuple_items = value.get("v") if isinstance(value, Mapping) and value.get("__t") == "tuple" else value
        if not isinstance(tuple_items, list):
            raise ValueError("cursor must be a valid smart-home route cursor.")
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return tuple(_decode_cursor_value(item, arguments[0]) for item in tuple_items)
        if arguments and len(arguments) != len(tuple_items):
            raise ValueError("cursor must be a valid smart-home route cursor.")
        item_types = arguments or (Any,) * len(tuple_items)
        return tuple(
            _decode_cursor_value(item, item_type)
            for item, item_type in zip(tuple_items, item_types)
        )

    if origin in (set, frozenset):
        item_type = get_args(expected_type)[0] if get_args(expected_type) else Any
        set_items = value.get("v") if isinstance(value, Mapping) and value.get("__t") == "set" else value
        if not isinstance(set_items, list):
            raise ValueError("cursor must be a valid smart-home route cursor.")
        decoded_items = {_decode_cursor_value(item, item_type) for item in set_items}
        return decoded_items if origin is set else frozenset(decoded_items)

    if origin in (dict, Mapping):
        arguments = get_args(expected_type)
        key_type = arguments[0] if len(arguments) >= 1 else Any
        value_type = arguments[1] if len(arguments) >= 2 else Any
        if isinstance(value, Mapping) and value.get("__t") == "mapping":
            items = value.get("v", ())
            if not isinstance(items, list):
                raise ValueError("cursor must be a valid smart-home route cursor.")
            return {
                _decode_cursor_value(key, key_type): _decode_cursor_value(item, value_type)
                for key, item in items
            }
        if not isinstance(value, Mapping):
            raise ValueError("cursor must be a valid smart-home route cursor.")
        return {
            _decode_cursor_value(str(key), key_type): _decode_cursor_value(item, value_type)
            for key, item in value.items()
        }

    if isinstance(expected_type, type) and is_dataclass(expected_type):
        if not isinstance(value, Mapping):
            raise ValueError("cursor must be a valid smart-home route cursor.")
        return _decode_dataclass_payload(expected_type, value)

    return _decode_cursor_value_untyped(value)


def _encode_dataclass_payload(instance: object) -> dict[str, object]:
    """Encode one dataclass instance into a cursor payload."""

    return {
        field_definition.name: _encode_cursor_value(getattr(instance, field_definition.name))
        for field_definition in fields(instance)
    }


def _decode_dataclass_payload(model_type: type[object], payload: Mapping[str, object]) -> object:
    """Decode one dataclass payload using the model's resolved type hints."""

    type_hints = _dataclass_type_hints(model_type)
    decoded_kwargs: dict[str, object] = {}
    for field_definition in fields(model_type):
        if field_definition.name not in payload:
            raise ValueError("cursor must be a valid smart-home route cursor.")
        decoded_kwargs[field_definition.name] = _decode_cursor_value(
            payload[field_definition.name],
            type_hints.get(field_definition.name, Any),
        )
    return model_type(**decoded_kwargs)


def build_routed_entity_id(route_id: str, entity_id: str) -> str:
    """Return one route-qualified smart-home entity identifier."""

    return _encode_routed_id(
        _ROUTED_ENTITY_PREFIX,
        _normalize_public_route_id(route_id),
        _normalize_source_id(entity_id, label="entity_id"),
    )


def parse_routed_entity_id(entity_id: str) -> tuple[str, str]:
    """Return the public route ID and child entity ID for one routed entity."""

    return _decode_routed_id(entity_id, prefix=_ROUTED_ENTITY_PREFIX, label="entity_id")


def build_routed_event_id(route_id: str, event_id: str) -> str:
    """Return one route-qualified smart-home event identifier."""

    return _encode_routed_id(
        _ROUTED_EVENT_PREFIX,
        _normalize_public_route_id(route_id),
        _normalize_source_id(event_id, label="event_id"),
    )


def parse_routed_event_id(event_id: str) -> tuple[str, str]:
    """Return the public route ID and child event ID for one routed event."""

    return _decode_routed_id(event_id, prefix=_ROUTED_EVENT_PREFIX, label="event_id")


@dataclass(frozen=True, slots=True)
class RoutedSmartHomeProvider:
    """Describe one child smart-home provider bound to internal/public route IDs.

    ``route_id`` is the internal dispatch key used by the aggregate layer.
    ``public_route_id`` is the external stable identifier written into entity IDs,
    event IDs, and user-facing responses. Use a stable opaque key here rather than
    mutable transport details such as bridge hosts/IPs. ``display_name`` is only
    used for labels shown to people.
    """

    route_id: str
    entity_provider: SmartHomeEntityProvider
    controller: SmartHomeController | None = None
    sensor_stream: SmartHomeSensorStream | None = None
    public_route_id: str | None = None
    display_name: str | None = None

    def __post_init__(self) -> None:
        """Normalize route identifiers and default optional child surfaces."""

        normalized_route_id = _normalize_route_id(self.route_id)
        controller = self.controller
        if controller is None and isinstance(self.entity_provider, SmartHomeController):
            controller = self.entity_provider
        sensor_stream = self.sensor_stream
        if sensor_stream is None and isinstance(self.entity_provider, SmartHomeSensorStream):
            sensor_stream = self.entity_provider
        public_route_id = (
            normalized_route_id
            if self.public_route_id is None
            else _normalize_public_route_id(self.public_route_id)
        )
        display_name = public_route_id if self.display_name is None else _normalize_display_name(self.display_name)
        object.__setattr__(self, "route_id", normalized_route_id)
        object.__setattr__(self, "controller", controller)
        object.__setattr__(self, "sensor_stream", sensor_stream)
        object.__setattr__(self, "public_route_id", public_route_id)
        object.__setattr__(self, "display_name", display_name)


@dataclass(slots=True)
class _RouteStreamState:
    """Track buffered events and child cursor state for one route."""

    buffer: deque[SmartHomeEvent] = field(default_factory=deque)
    next_cursor: str | None = None
    stream_live: bool = True
    loaded: bool = False


class _DescendingSortKey:
    """Invert tuple ordering so heapq pops the newest event first."""

    __slots__ = ("value",)

    def __init__(self, value: tuple[object, str]) -> None:
        self.value = value

    def __lt__(self, other: "_DescendingSortKey") -> bool:
        return self.value > other.value


@dataclass(frozen=True, slots=True)
class AggregatedSmartHomeProvider(SmartHomeEntityProvider, SmartHomeController, SmartHomeSensorStream):
    """Expose multiple child smart-home routes as one bounded provider."""

    providers: tuple[RoutedSmartHomeProvider, ...]
    max_concurrency: int = _DEFAULT_MAX_CONCURRENCY
    sensor_page_size: int = _DEFAULT_SENSOR_PAGE_SIZE
    # BREAKING: sensor stream page sizes are now server-bounded to keep Pi-class
    # deployments predictable under load.
    max_read_limit: int = _DEFAULT_MAX_READ_LIMIT
    max_cursor_bytes: int = _DEFAULT_MAX_CURSOR_BYTES
    cursor_secret: str | None = None
    allow_legacy_cursor: bool = True
    _providers_by_route_id: dict[str, RoutedSmartHomeProvider] = field(init=False, repr=False)
    _providers_by_public_route_id: dict[str, RoutedSmartHomeProvider] = field(init=False, repr=False)
    _sensor_routes: tuple[RoutedSmartHomeProvider, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate child routes and cache route lookups."""

        if not self.providers:
            raise ValueError("providers must not be empty.")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1.")
        if self.sensor_page_size < 1:
            raise ValueError("sensor_page_size must be at least 1.")
        if self.max_read_limit < 1:
            raise ValueError("max_read_limit must be at least 1.")
        if self.max_cursor_bytes < 1024:
            raise ValueError("max_cursor_bytes must be at least 1024.")
        if self.cursor_secret is not None and not self.cursor_secret.strip():
            raise ValueError("cursor_secret must not be blank.")

        providers_by_route_id: dict[str, RoutedSmartHomeProvider] = {}
        providers_by_public_route_id: dict[str, RoutedSmartHomeProvider] = {}
        for provider in self.providers:
            if provider.route_id in providers_by_route_id:
                raise ValueError(f"Duplicate smart-home route: {provider.route_id}")
            if provider.public_route_id in providers_by_public_route_id:
                raise ValueError(f"Duplicate smart-home public route: {provider.public_route_id}")
            providers_by_route_id[provider.route_id] = provider
            providers_by_public_route_id[provider.public_route_id] = provider
        object.__setattr__(self, "_providers_by_route_id", providers_by_route_id)
        object.__setattr__(self, "_providers_by_public_route_id", providers_by_public_route_id)
        object.__setattr__(
            self,
            "_sensor_routes",
            tuple(provider for provider in self.providers if provider.sensor_stream is not None),
        )

    def list_entities(
        self,
        *,
        entity_ids: tuple[str, ...] = (),
        entity_class: SmartHomeEntityClass | None = None,
        include_unavailable: bool = False,
    ) -> list[SmartHomeEntity]:
        """Return route-qualified entities from every relevant child provider."""

        requested_ids_by_route = self._requested_entity_ids_by_route(entity_ids)
        scheduled: list[tuple[int, Callable[[], list[SmartHomeEntity]]]] = []
        ordered_providers: list[RoutedSmartHomeProvider] = []
        for provider in self.providers:
            child_entity_ids = requested_ids_by_route.get(provider.route_id, ())
            if entity_ids and not child_entity_ids:
                continue
            ordered_providers.append(provider)
            scheduled.append(
                (
                    len(ordered_providers) - 1,
                    lambda provider=provider, child_entity_ids=child_entity_ids: provider.entity_provider.list_entities(
                        entity_ids=child_entity_ids,
                        entity_class=entity_class,
                        include_unavailable=include_unavailable,
                    ),
                )
            )

        results = self._execute_calls(scheduled)
        collected: list[SmartHomeEntity] = []
        for index, provider in enumerate(ordered_providers):
            entities = results[index]
            collected.extend(self._wrap_entity(provider, entity) for entity in entities)
        return self._disambiguate_labels(collected)

    def control(
        self,
        *,
        command: SmartHomeCommand,
        entity_ids: tuple[str, ...],
        parameters: Mapping[str, object],
    ) -> dict[str, object]:
        """Dispatch one generic control command to the correct child routes."""

        requested_ids_by_route = self._requested_entity_ids_by_route(entity_ids)
        scheduled: list[tuple[int, Callable[[], dict[str, object]]]] = []
        ordered_providers: list[RoutedSmartHomeProvider] = []
        for provider in self.providers:
            child_entity_ids = requested_ids_by_route.get(provider.route_id, ())
            if not child_entity_ids:
                continue
            if provider.controller is None:
                raise RuntimeError(f"Smart-home route {provider.public_route_id} does not support control.")
            ordered_providers.append(provider)
            scheduled.append(
                (
                    len(ordered_providers) - 1,
                    lambda provider=provider, child_entity_ids=child_entity_ids: provider.controller.control(
                        command=command,
                        entity_ids=child_entity_ids,
                        parameters=parameters,
                    ),
                )
            )

        results = self._execute_calls(scheduled)
        targets: list[dict[str, object]] = []
        route_results: list[dict[str, object]] = []
        for index, provider in enumerate(ordered_providers):
            normalized_result = self._normalize_control_result(provider, results[index])
            route_results.append(normalized_result)
            normalized_targets = normalized_result.get("targets", ())
            if isinstance(normalized_targets, list):
                targets.extend(normalized_targets)
        return {
            "targets": targets,
            "routes": route_results,
        }

    def read_sensor_stream(
        self,
        *,
        cursor: str | None = None,
        limit: int,
    ) -> SmartHomeEventBatch:
        """Read a bounded, globally ordered merged sensor batch across all routes."""

        if not isinstance(limit, int):
            raise TypeError("limit must be an integer.")
        if limit < 1:
            raise ValueError("limit must be at least 1.")
        if limit > self.max_read_limit:
            raise ValueError(f"limit must be at most {self.max_read_limit}.")
        if not self._sensor_routes:
            raise RuntimeError("No smart-home sensor stream is configured.")

        route_states = self._decode_cursor_map(cursor)
        for provider in self._sensor_routes:
            route_states.setdefault(provider.route_id, _RouteStreamState())
        page_limit = min(limit, self.sensor_page_size)

        initial_fetches: list[tuple[int, Callable[[], None]]] = []
        for provider in self._sensor_routes:
            state = route_states[provider.route_id]
            if self._route_state_needs_fetch(state):
                initial_fetches.append(
                    (
                        len(initial_fetches),
                        lambda provider=provider, state=state: self._fill_route_buffer(
                            provider,
                            state,
                            page_limit=page_limit,
                        ),
                    )
                )
        self._execute_calls(initial_fetches)

        heap: list[tuple[_DescendingSortKey, int, str, SmartHomeEvent]] = []
        sequence = 0
        for provider in self._sensor_routes:
            state = route_states[provider.route_id]
            if state.buffer:
                head_event = state.buffer[0]
                heapq.heappush(
                    heap,
                    (_DescendingSortKey(_event_sort_key(head_event)), sequence, provider.route_id, head_event),
                )
                sequence += 1

        collected_events: list[SmartHomeEvent] = []
        while heap and len(collected_events) < limit:
            _, _, route_id, event = heapq.heappop(heap)
            state = route_states[route_id]
            state.buffer.popleft()
            collected_events.append(event)

            if not state.buffer and self._route_state_needs_fetch(state):
                provider = self._providers_by_route_id[route_id]
                self._fill_route_buffer(provider, state, page_limit=page_limit)

            if state.buffer:
                head_event = state.buffer[0]
                heapq.heappush(
                    heap,
                    (_DescendingSortKey(_event_sort_key(head_event)), sequence, route_id, head_event),
                )
                sequence += 1

        stream_live = all(route_states[provider.route_id].stream_live for provider in self._sensor_routes)
        next_cursor = self._encode_cursor_map(route_states)
        return SmartHomeEventBatch(
            events=tuple(collected_events),
            next_cursor=next_cursor,
            stream_live=stream_live,
        )

    def _requested_entity_ids_by_route(self, entity_ids: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
        """Group routed entity IDs by child route while preserving request order."""

        requested_ids_by_route: dict[str, list[str]] = {}
        seen_ids_by_route: dict[str, set[str]] = {}
        for entity_id in entity_ids:
            public_route_id, source_entity_id = parse_routed_entity_id(entity_id)
            provider = self._providers_by_public_route_id.get(public_route_id)
            if provider is None:
                raise ValueError(f"Unknown smart-home route for entity {entity_id}.")
            route_id = provider.route_id
            seen = seen_ids_by_route.setdefault(route_id, set())
            if source_entity_id in seen:
                continue
            seen.add(source_entity_id)
            requested_ids_by_route.setdefault(route_id, []).append(source_entity_id)
        return {
            route_id: tuple(source_entity_ids)
            for route_id, source_entity_ids in requested_ids_by_route.items()
        }

    def _wrap_entity(self, provider: RoutedSmartHomeProvider, entity: SmartHomeEntity) -> SmartHomeEntity:
        """Attach public route metadata and a routed identifier to one entity."""

        base_attributes = entity.attributes if isinstance(entity.attributes, Mapping) else {}
        attributes = dict(base_attributes)
        attributes["route_id"] = provider.public_route_id
        attributes["route_display_name"] = provider.display_name
        attributes["source_entity_id"] = entity.entity_id
        return replace(
            entity,
            entity_id=build_routed_entity_id(provider.public_route_id, entity.entity_id),
            attributes=attributes,
        )

    def _wrap_event(self, provider: RoutedSmartHomeProvider, event: SmartHomeEvent) -> SmartHomeEvent:
        """Attach public route metadata and routed identifiers to one event."""

        base_details = event.details if isinstance(event.details, Mapping) else {}
        details = dict(base_details)
        details["route_id"] = provider.public_route_id
        details["route_display_name"] = provider.display_name
        details["source_event_id"] = event.event_id
        details["source_entity_id"] = event.entity_id
        return replace(
            event,
            event_id=build_routed_event_id(provider.public_route_id, event.event_id),
            entity_id=build_routed_entity_id(provider.public_route_id, event.entity_id),
            details=details,
        )

    def _disambiguate_labels(self, entities: list[SmartHomeEntity]) -> list[SmartHomeEntity]:
        """Append the route display name only when visible labels would collide."""

        label_counts = Counter(
            (
                _casefold_or_empty(entity.label),
                _casefold_or_empty(getattr(entity, "area", None)),
                getattr(entity.entity_class, "value", str(entity.entity_class)),
            )
            for entity in entities
        )
        return [
            replace(entity, label=f"{entity.label} ({entity.attributes['route_display_name']})")
            if label_counts[
                (
                    _casefold_or_empty(entity.label),
                    _casefold_or_empty(getattr(entity, "area", None)),
                    getattr(entity.entity_class, "value", str(entity.entity_class)),
                )
            ] > 1
            else entity
            for entity in entities
        ]

    def _normalize_control_result(
        self,
        provider: RoutedSmartHomeProvider,
        child_result: Mapping[str, object] | dict[str, object],
    ) -> dict[str, object]:
        """Attach public route-qualified IDs to one child control response."""

        normalized = dict(child_result)
        normalized["route_id"] = provider.public_route_id
        normalized["route_display_name"] = provider.display_name

        source_entity_id = normalized.get("entity_id")
        if isinstance(source_entity_id, str) and source_entity_id.strip():
            normalized["source_entity_id"] = source_entity_id.strip()
            normalized["entity_id"] = build_routed_entity_id(provider.public_route_id, source_entity_id)

        raw_targets = normalized.get("targets", ())
        if isinstance(raw_targets, Mapping):
            iterable_targets: Sequence[object] = (raw_targets,)
        elif isinstance(raw_targets, Sequence) and not isinstance(raw_targets, (str, bytes, bytearray)):
            iterable_targets = raw_targets
        else:
            iterable_targets = ()

        rewritten_targets: list[dict[str, object]] = []
        for target in iterable_targets:
            if not isinstance(target, Mapping):
                continue
            rewritten_target = dict(target)
            rewritten_target["route_id"] = provider.public_route_id
            rewritten_target["route_display_name"] = provider.display_name
            raw_entity_id = target.get("entity_id")
            if isinstance(raw_entity_id, str) and raw_entity_id.strip():
                source_entity_id = raw_entity_id.strip()
                rewritten_target["source_entity_id"] = source_entity_id
                rewritten_target["entity_id"] = build_routed_entity_id(provider.public_route_id, source_entity_id)
            rewritten_targets.append(rewritten_target)
        normalized["targets"] = rewritten_targets
        return normalized

    def _execute_calls(self, scheduled: Sequence[tuple[int, Callable[[], object]]]) -> dict[int, object]:
        """Execute scheduled provider calls sequentially or in a bounded thread pool."""

        if not scheduled:
            return {}
        if len(scheduled) == 1 or self.max_concurrency == 1:
            return {index: call() for index, call in scheduled}

        max_workers = min(self.max_concurrency, len(scheduled))
        results: dict[int, object] = {}
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="twinr-smarthome") as executor:
            future_to_index = {
                executor.submit(call): index
                for index, call in scheduled
            }
            for future in as_completed(future_to_index):
                results[future_to_index[future]] = future.result()
        return results

    @staticmethod
    def _route_state_needs_fetch(state: _RouteStreamState) -> bool:
        """Return whether the route must fetch another child page."""

        return not state.buffer and (not state.loaded or state.next_cursor is not None)

    def _fill_route_buffer(
        self,
        provider: RoutedSmartHomeProvider,
        state: _RouteStreamState,
        *,
        page_limit: int,
    ) -> None:
        """Fetch one child page into the route buffer, preserving leftovers for later pages."""

        if provider.sensor_stream is None or not self._route_state_needs_fetch(state):
            return

        request_cursor = state.next_cursor
        batch = provider.sensor_stream.read_sensor_stream(
            cursor=request_cursor,
            limit=page_limit,
        )
        state.loaded = True
        state.stream_live = batch.stream_live

        next_cursor = batch.next_cursor
        if next_cursor is not None:
            if not isinstance(next_cursor, str) or not next_cursor.strip():
                raise ValueError("child sensor stream returned an invalid cursor.")
            next_cursor = next_cursor.strip()
        state.next_cursor = next_cursor

        wrapped_events = [self._wrap_event(provider, event) for event in batch.events]
        wrapped_events.sort(key=_event_sort_key, reverse=True)
        state.buffer.extend(wrapped_events)

        if not wrapped_events and request_cursor == state.next_cursor:
            state.next_cursor = None

    def _cursor_secret_bytes(self) -> bytes | None:
        """Return the configured HMAC secret as bytes when available."""

        secret = self.cursor_secret
        if secret is None:
            env_secret = os.getenv(_CURSOR_SECRET_ENV)
            if env_secret is not None and env_secret.strip():
                secret = env_secret
        return None if secret is None else secret.encode("utf-8")

    def _decode_cursor_map(self, cursor: str | None) -> dict[str, _RouteStreamState]:
        """Parse an aggregate cursor into buffered per-route child state."""

        if cursor is None or not cursor.strip():
            return {}
        if len(cursor.encode("utf-8")) > self.max_cursor_bytes:
            raise ValueError("cursor exceeds the configured maximum size.")

        stripped_cursor = cursor.strip()
        if stripped_cursor.startswith(_CURSOR_TOKEN_PREFIX):
            token = stripped_cursor[len(_CURSOR_TOKEN_PREFIX) :]
            body_token, separator, signature_token = token.partition(".")
            if not body_token:
                raise ValueError("cursor must be a valid smart-home route cursor.")
            payload_bytes = _urlsafe_b64decode(body_token)
            secret = self._cursor_secret_bytes()
            if separator:
                if secret is None:
                    raise ValueError("cursor signing is enabled but no cursor secret is configured.")
                expected_signature = hmac.new(secret, payload_bytes, hashlib.sha256).digest()
                actual_signature = _urlsafe_b64decode(signature_token)
                if not hmac.compare_digest(actual_signature, expected_signature):
                    raise ValueError("cursor must be a valid smart-home route cursor.")
            try:
                payload = json.loads(payload_bytes)
            except json.JSONDecodeError as exc:
                raise ValueError("cursor must be a valid smart-home route cursor.") from exc
            return self._decode_cursor_payload(payload)

        if not self.allow_legacy_cursor:
            raise ValueError("cursor must be a valid smart-home route cursor.")

        try:
            payload = json.loads(stripped_cursor)
        except json.JSONDecodeError as exc:
            raise ValueError("cursor must be a valid smart-home route cursor.") from exc

        if not isinstance(payload, Mapping):
            raise ValueError("cursor must be a valid smart-home route cursor.")
        if payload.get("version") != _LEGACY_CURSOR_VERSION:
            raise ValueError("cursor must be a valid smart-home route cursor.")
        routes = payload.get("routes", {})
        if not isinstance(routes, Mapping):
            raise ValueError("cursor must be a valid smart-home route cursor.")

        result: dict[str, _RouteStreamState] = {}
        for route_id, child_cursor in routes.items():
            normalized_route_id = _normalize_route_id(route_id)
            provider = self._providers_by_route_id.get(normalized_route_id)
            if provider is None or provider.sensor_stream is None:
                raise ValueError("cursor must be a valid smart-home route cursor.")
            if not isinstance(child_cursor, str) or not child_cursor.strip():
                raise ValueError("cursor must be a valid smart-home route cursor.")
            result[normalized_route_id] = _RouteStreamState(
                next_cursor=child_cursor.strip(),
                stream_live=True,
                loaded=False,
            )
        return result

    def _decode_cursor_payload(self, payload: Mapping[str, object]) -> dict[str, _RouteStreamState]:
        """Decode one opaque v2 cursor payload."""

        if payload.get("version") != _CURSOR_VERSION:
            raise ValueError("cursor must be a valid smart-home route cursor.")
        if not isinstance(payload.get("routes", {}), Mapping):
            raise ValueError("cursor must be a valid smart-home route cursor.")

        routes = payload["routes"]
        result: dict[str, _RouteStreamState] = {}
        for route_id, route_state_payload in routes.items():
            normalized_route_id = _normalize_route_id(route_id)
            provider = self._providers_by_route_id.get(normalized_route_id)
            if provider is None or provider.sensor_stream is None:
                raise ValueError("cursor must be a valid smart-home route cursor.")
            if not isinstance(route_state_payload, Mapping):
                raise ValueError("cursor must be a valid smart-home route cursor.")

            raw_child_cursor = route_state_payload.get("cursor")
            child_cursor: str | None
            if raw_child_cursor is None:
                child_cursor = None
            elif isinstance(raw_child_cursor, str) and raw_child_cursor.strip():
                child_cursor = raw_child_cursor.strip()
            else:
                raise ValueError("cursor must be a valid smart-home route cursor.")

            stream_live = route_state_payload.get("stream_live", True)
            if not isinstance(stream_live, bool):
                raise ValueError("cursor must be a valid smart-home route cursor.")

            raw_buffer = route_state_payload.get("buffer", ())
            if not isinstance(raw_buffer, list):
                raise ValueError("cursor must be a valid smart-home route cursor.")

            state = _RouteStreamState(
                next_cursor=child_cursor,
                stream_live=stream_live,
                loaded=True,
            )
            for event_payload in raw_buffer:
                if not isinstance(event_payload, Mapping):
                    raise ValueError("cursor must be a valid smart-home route cursor.")
                state.buffer.append(self._decode_event_payload(event_payload))
            result[normalized_route_id] = state
        return result

    def _decode_event_payload(self, payload: Mapping[str, object]) -> SmartHomeEvent:
        """Decode one buffered event from the cursor payload."""

        decoded = _decode_dataclass_payload(SmartHomeEvent, payload)
        if not isinstance(decoded, SmartHomeEvent):
            raise ValueError("cursor must be a valid smart-home route cursor.")
        return decoded

    def _encode_cursor_map(self, route_states: Mapping[str, _RouteStreamState]) -> str | None:
        """Encode per-route buffers/cursors when at least one route can continue."""

        # BREAKING: aggregate cursors are emitted as opaque v2 tokens. Legacy v1 JSON
        # cursors are still accepted when allow_legacy_cursor=True.
        routes_payload: dict[str, object] = {}
        for provider in self._sensor_routes:
            state = route_states[provider.route_id]
            if not state.buffer and state.next_cursor is None:
                continue
            routes_payload[provider.route_id] = {
                "cursor": state.next_cursor,
                "buffer": [_encode_dataclass_payload(event) for event in state.buffer],
                "stream_live": state.stream_live,
            }

        if not routes_payload:
            return None

        payload_bytes = json.dumps(
            {
                "version": _CURSOR_VERSION,
                "routes": routes_payload,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

        cursor_token = _urlsafe_b64encode(payload_bytes)
        secret = self._cursor_secret_bytes()
        if secret is not None:
            signature = _urlsafe_b64encode(hmac.new(secret, payload_bytes, hashlib.sha256).digest())
            cursor_token = f"{cursor_token}.{signature}"
        cursor = f"{_CURSOR_TOKEN_PREFIX}{cursor_token}"
        if len(cursor.encode("utf-8")) > self.max_cursor_bytes:
            raise RuntimeError(
                "encoded cursor exceeds max_cursor_bytes; lower sensor_page_size or max_read_limit."
            )
        return cursor


__all__ = [
    "AggregatedSmartHomeProvider",
    "RoutedSmartHomeProvider",
    "build_routed_entity_id",
    "build_routed_event_id",
    "parse_routed_entity_id",
    "parse_routed_event_id",
]