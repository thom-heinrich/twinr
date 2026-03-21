"""Aggregate multiple routed smart-home providers behind one bounded surface.

This module keeps multi-bridge or other multi-route wiring out of provider
packages. It rewrites child entity and event identifiers into route-qualified
IDs, dispatches read/control calls back to the correct child provider, and
combines bounded sensor-stream batches into one generic Twinr stream.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
import json
from urllib.parse import quote, unquote

from twinr.integrations.smarthome.adapter import SmartHomeController, SmartHomeEntityProvider, SmartHomeSensorStream
from twinr.integrations.smarthome.models import SmartHomeCommand, SmartHomeEntity, SmartHomeEntityClass, SmartHomeEvent, SmartHomeEventBatch

_ROUTED_ENTITY_PREFIX = "route:"
_ROUTED_EVENT_PREFIX = "route-event:"
_CURSOR_VERSION = 1


def _normalize_route_id(route_id: object) -> str:
    """Return one stripped route identifier."""

    if not isinstance(route_id, str):
        raise TypeError("route_id must be a string.")
    normalized = route_id.strip()
    if not normalized:
        raise ValueError("route_id must not be empty.")
    if any(ord(character) < 32 for character in normalized):
        raise ValueError("route_id must not contain control characters.")
    return normalized


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
    route_id = unquote(route_token)
    source_id = unquote(source_token)
    if not route_id or not source_id:
        raise ValueError(f"{label} must be a routed smart-home identifier.")
    return route_id, source_id


def build_routed_entity_id(route_id: str, entity_id: str) -> str:
    """Return one route-qualified smart-home entity identifier."""

    return _encode_routed_id(
        _ROUTED_ENTITY_PREFIX,
        _normalize_route_id(route_id),
        entity_id.strip(),
    )


def parse_routed_entity_id(entity_id: str) -> tuple[str, str]:
    """Return the route ID and child entity ID for one routed entity."""

    return _decode_routed_id(entity_id, prefix=_ROUTED_ENTITY_PREFIX, label="entity_id")


def build_routed_event_id(route_id: str, event_id: str) -> str:
    """Return one route-qualified smart-home event identifier."""

    return _encode_routed_id(
        _ROUTED_EVENT_PREFIX,
        _normalize_route_id(route_id),
        event_id.strip(),
    )


@dataclass(frozen=True, slots=True)
class RoutedSmartHomeProvider:
    """Describe one child smart-home provider bound to a route identifier.

    A route is an opaque stable key used by the aggregate layer to dispatch
    reads, controls, and sensor-stream reads back to the correct child surface.
    Hue currently uses the bridge host as ``route_id``.
    """

    route_id: str
    entity_provider: SmartHomeEntityProvider
    controller: SmartHomeController | None = None
    sensor_stream: SmartHomeSensorStream | None = None

    def __post_init__(self) -> None:
        """Normalize the route ID and default optional child surfaces."""

        normalized_route_id = _normalize_route_id(self.route_id)
        controller = self.controller
        if controller is None and isinstance(self.entity_provider, SmartHomeController):
            controller = self.entity_provider
        sensor_stream = self.sensor_stream
        if sensor_stream is None and isinstance(self.entity_provider, SmartHomeSensorStream):
            sensor_stream = self.entity_provider
        object.__setattr__(self, "route_id", normalized_route_id)
        object.__setattr__(self, "controller", controller)
        object.__setattr__(self, "sensor_stream", sensor_stream)


@dataclass(frozen=True, slots=True)
class AggregatedSmartHomeProvider(SmartHomeEntityProvider, SmartHomeController, SmartHomeSensorStream):
    """Expose multiple child smart-home routes as one bounded provider."""

    providers: tuple[RoutedSmartHomeProvider, ...]

    def __post_init__(self) -> None:
        """Validate the child route table."""

        if not self.providers:
            raise ValueError("providers must not be empty.")
        route_ids: set[str] = set()
        for provider in self.providers:
            route_id = provider.route_id
            if route_id in route_ids:
                raise ValueError(f"Duplicate smart-home route: {route_id}")
            route_ids.add(route_id)

    def list_entities(
        self,
        *,
        entity_ids: tuple[str, ...] = (),
        entity_class: SmartHomeEntityClass | None = None,
        include_unavailable: bool = False,
    ) -> list[SmartHomeEntity]:
        """Return route-qualified entities from every relevant child provider."""

        requested_ids_by_route = self._requested_entity_ids_by_route(entity_ids)
        collected: list[SmartHomeEntity] = []
        for provider in self.providers:
            child_entity_ids = requested_ids_by_route.get(provider.route_id, ())
            if entity_ids and not child_entity_ids:
                continue
            entities = provider.entity_provider.list_entities(
                entity_ids=child_entity_ids,
                entity_class=entity_class,
                include_unavailable=include_unavailable,
            )
            collected.extend(self._wrap_entity(provider.route_id, entity) for entity in entities)
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
        targets: list[dict[str, object]] = []
        route_results: list[dict[str, object]] = []
        for provider in self.providers:
            child_entity_ids = requested_ids_by_route.get(provider.route_id, ())
            if not child_entity_ids:
                continue
            if provider.controller is None:
                raise RuntimeError(f"Smart-home route {provider.route_id} does not support control.")
            child_result = provider.controller.control(
                command=command,
                entity_ids=child_entity_ids,
                parameters=parameters,
            )
            normalized_result = self._normalize_control_result(provider.route_id, child_result)
            route_results.append(normalized_result)
            for target in normalized_result.get("targets", ()):
                if isinstance(target, Mapping):
                    targets.append(dict(target))
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
        """Read a bounded merged sensor batch across all routed streams."""

        sensor_routes = tuple(provider for provider in self.providers if provider.sensor_stream is not None)
        if not sensor_routes:
            raise RuntimeError("No smart-home sensor stream is configured.")

        cursors = self._decode_cursor_map(cursor)
        base_limit, extra = divmod(limit, len(sensor_routes))
        collected_events: list[SmartHomeEvent] = []
        next_cursors: dict[str, str] = {}
        stream_live = True
        for index, provider in enumerate(sensor_routes):
            child_limit = max(1, base_limit + (1 if index < extra else 0))
            batch = provider.sensor_stream.read_sensor_stream(
                cursor=cursors.get(provider.route_id),
                limit=child_limit,
            )
            stream_live = stream_live and batch.stream_live
            if batch.next_cursor is not None:
                next_cursors[provider.route_id] = batch.next_cursor
            collected_events.extend(
                self._wrap_event(provider.route_id, event)
                for event in batch.events
            )

        collected_events.sort(key=lambda item: (item.observed_at, item.event_id), reverse=True)
        return SmartHomeEventBatch(
            events=tuple(collected_events[:limit]),
            next_cursor=self._encode_cursor_map(next_cursors),
            stream_live=stream_live,
        )

    def _requested_entity_ids_by_route(self, entity_ids: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
        """Group routed entity IDs by child route while preserving request order."""

        requested_ids_by_route: dict[str, list[str]] = {}
        known_routes = {provider.route_id for provider in self.providers}
        for entity_id in entity_ids:
            route_id, source_entity_id = parse_routed_entity_id(entity_id)
            if route_id not in known_routes:
                raise ValueError(f"Unknown smart-home route for entity {entity_id}.")
            requested_ids_by_route.setdefault(route_id, []).append(source_entity_id)
        return {
            route_id: tuple(source_entity_ids)
            for route_id, source_entity_ids in requested_ids_by_route.items()
        }

    def _wrap_entity(self, route_id: str, entity: SmartHomeEntity) -> SmartHomeEntity:
        """Attach route metadata and a routed identifier to one entity."""

        attributes = dict(entity.attributes)
        attributes["route_id"] = route_id
        attributes["source_entity_id"] = entity.entity_id
        return replace(
            entity,
            entity_id=build_routed_entity_id(route_id, entity.entity_id),
            attributes=attributes,
        )

    def _wrap_event(self, route_id: str, event: SmartHomeEvent) -> SmartHomeEvent:
        """Attach route metadata and routed identifiers to one event."""

        details = dict(event.details)
        details["route_id"] = route_id
        details["source_event_id"] = event.event_id
        details["source_entity_id"] = event.entity_id
        return replace(
            event,
            event_id=build_routed_event_id(route_id, event.event_id),
            entity_id=build_routed_entity_id(route_id, event.entity_id),
            details=details,
        )

    def _disambiguate_labels(self, entities: list[SmartHomeEntity]) -> list[SmartHomeEntity]:
        """Append the route ID only when visible entity labels would collide."""

        label_counts = Counter(
            (entity.label.casefold(), entity.area.casefold(), entity.entity_class.value)
            for entity in entities
        )
        return [
            replace(entity, label=f"{entity.label} ({entity.attributes['route_id']})")
            if label_counts[(entity.label.casefold(), entity.area.casefold(), entity.entity_class.value)] > 1
            else entity
            for entity in entities
        ]

    def _normalize_control_result(
        self,
        route_id: str,
        child_result: Mapping[str, object] | dict[str, object],
    ) -> dict[str, object]:
        """Attach route-qualified IDs to one child control response."""

        normalized = dict(child_result)
        normalized["route_id"] = route_id
        raw_targets = normalized.get("targets", ())
        if isinstance(raw_targets, list):
            rewritten_targets: list[dict[str, object]] = []
            for target in raw_targets:
                if not isinstance(target, Mapping):
                    continue
                source_entity_id = str(target.get("entity_id", "")).strip()
                rewritten_target = dict(target)
                rewritten_target["route_id"] = route_id
                if source_entity_id:
                    rewritten_target["source_entity_id"] = source_entity_id
                    rewritten_target["entity_id"] = build_routed_entity_id(route_id, source_entity_id)
                rewritten_targets.append(rewritten_target)
            normalized["targets"] = rewritten_targets
        return normalized

    @staticmethod
    def _decode_cursor_map(cursor: str | None) -> dict[str, str]:
        """Parse the aggregate cursor into per-route child cursors."""

        if cursor is None or not cursor.strip():
            return {}
        try:
            payload = json.loads(cursor)
        except json.JSONDecodeError as exc:
            raise ValueError("cursor must be a valid smart-home route cursor.") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("cursor must be a valid smart-home route cursor.")
        if payload.get("version") != _CURSOR_VERSION:
            raise ValueError("cursor must be a valid smart-home route cursor.")
        routes = payload.get("routes", {})
        if not isinstance(routes, Mapping):
            raise ValueError("cursor must be a valid smart-home route cursor.")
        result: dict[str, str] = {}
        for route_id, child_cursor in routes.items():
            if not isinstance(route_id, str) or not route_id.strip():
                raise ValueError("cursor must be a valid smart-home route cursor.")
            if not isinstance(child_cursor, str) or not child_cursor.strip():
                raise ValueError("cursor must be a valid smart-home route cursor.")
            result[route_id] = child_cursor
        return result

    @staticmethod
    def _encode_cursor_map(cursors: Mapping[str, str]) -> str | None:
        """Encode per-route cursors when at least one child provided one."""

        if not cursors:
            return None
        return json.dumps(
            {
                "version": _CURSOR_VERSION,
                "routes": dict(sorted(cursors.items())),
            },
            sort_keys=True,
            separators=(",", ":"),
        )


__all__ = [
    "AggregatedSmartHomeProvider",
    "RoutedSmartHomeProvider",
    "build_routed_entity_id",
    "build_routed_event_id",
    "parse_routed_entity_id",
]
