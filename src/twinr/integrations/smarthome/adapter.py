"""Bridge generic smart-home providers onto Twinr's integration contract.

The adapter exposes one bounded surface for:
- user-driven read requests
- explicit control requests
- background sensor/event stream reads

Provider-specific logic belongs in child packages such as ``smarthome.hue``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from twinr.integrations.adapter import IntegrationAdapter
from twinr.integrations.models import IntegrationManifest, IntegrationRequest, IntegrationResult
from twinr.integrations.smarthome.models import (
    SmartHomeCommand,
    SmartHomeEntity,
    SmartHomeEntityAggregateField,
    SmartHomeEntityClass,
    SmartHomeEventBatch,
    SmartHomeEventAggregateField,
)
from twinr.integrations.smarthome.query import (
    aggregate_entities,
    aggregate_events,
    entity_query_filters_payload,
    event_query_filters_payload,
    filter_entities,
    filter_events,
    paginate_entities,
    parse_entity_query_parameters,
    parse_event_query_parameters,
)

logger = logging.getLogger(__name__)


def _parse_positive_int(value: object, *, field_name: str) -> int:
    """Parse a positive integer from settings or request input."""

    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive whole number.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive whole number.") from exc
    if parsed < 1:
        raise ValueError(f"{field_name} must be a positive whole number.")
    return parsed


def _parse_strict_bool(value: object, *, field_name: str) -> bool:
    """Parse one boolean or supported boolean-like literal."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", ""}:
            return False
    raise ValueError(f"{field_name} must be a boolean value.")


def _parse_optional_text(value: object, *, field_name: str) -> str | None:
    """Return one stripped text value or ``None`` when blank."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")
    normalized = value.strip()
    return normalized or None


def _parse_string_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    """Normalize one request parameter into a string tuple."""

    if value is None:
        return ()
    if isinstance(value, str):
        normalized = value.strip()
        return (normalized,) if normalized else ()
    if isinstance(value, (list, tuple)):
        collected: list[str] = []
        for index, item in enumerate(value):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"{field_name}[{index}] must be a non-empty string.")
            collected.append(item.strip())
        return tuple(collected)
    raise ValueError(f"{field_name} must be a string, list, tuple, or null.")


def _parameters(request: IntegrationRequest) -> Mapping[str, object]:
    """Return request parameters as a validated mapping."""

    params = getattr(request, "parameters", None)
    if not isinstance(params, Mapping):
        raise ValueError("parameters must be a mapping.")
    return params


@runtime_checkable
class SmartHomeEntityProvider(Protocol):
    """Describe the smart-home read surface required by the adapter."""

    def list_entities(
        self,
        *,
        entity_ids: tuple[str, ...] = (),
        entity_class: SmartHomeEntityClass | None = None,
        include_unavailable: bool = False,
    ) -> list[SmartHomeEntity]:
        """Return a bounded list of normalized smart-home entities."""

        ...


@runtime_checkable
class SmartHomeController(Protocol):
    """Describe the smart-home control surface required by the adapter."""

    def control(
        self,
        *,
        command: SmartHomeCommand,
        entity_ids: tuple[str, ...],
        parameters: Mapping[str, object],
    ) -> dict[str, object]:
        """Execute one generic control command."""

        ...


@runtime_checkable
class SmartHomeSensorStream(Protocol):
    """Describe the bounded sensor/event stream surface."""

    def read_sensor_stream(
        self,
        *,
        cursor: str | None = None,
        limit: int,
    ) -> SmartHomeEventBatch:
        """Return a bounded batch of normalized smart-home events."""

        ...


@dataclass(frozen=True, slots=True)
class SmartHomeAdapterSettings:
    """Hold safety and size limits for generic smart-home operations."""

    max_entity_results: int = 32
    max_event_results: int = 32
    max_control_targets: int = 8
    allowed_control_classes: tuple[SmartHomeEntityClass, ...] = (
        SmartHomeEntityClass.LIGHT,
        SmartHomeEntityClass.LIGHT_GROUP,
        SmartHomeEntityClass.SCENE,
        SmartHomeEntityClass.SWITCH,
    )

    def __post_init__(self) -> None:
        """Validate size limits and allowed control classes."""

        object.__setattr__(self, "max_entity_results", _parse_positive_int(self.max_entity_results, field_name="max_entity_results"))
        object.__setattr__(self, "max_event_results", _parse_positive_int(self.max_event_results, field_name="max_event_results"))
        object.__setattr__(self, "max_control_targets", _parse_positive_int(self.max_control_targets, field_name="max_control_targets"))
        object.__setattr__(
            self,
            "allowed_control_classes",
            tuple(SmartHomeEntityClass(item) for item in self.allowed_control_classes),
        )


@dataclass(slots=True)
class SmartHomeIntegrationAdapter(IntegrationAdapter):
    """Execute generic smart-home integration requests through one provider."""

    manifest: IntegrationManifest
    entity_provider: SmartHomeEntityProvider
    controller: SmartHomeController | None = None
    sensor_stream: SmartHomeSensorStream | None = None
    settings: SmartHomeAdapterSettings = field(default_factory=SmartHomeAdapterSettings)

    def execute(self, request: IntegrationRequest) -> IntegrationResult:
        """Dispatch one generic smart-home request into a normalized result."""

        operation_id = getattr(request, "operation_id", None)
        try:
            if operation_id in {"list_entities", "read_device_state"}:
                return self._list_entities(request)
            if operation_id == "control_entities":
                return self._control_entities(request)
            if operation_id == "run_safe_scene":
                return self._run_safe_scene(request)
            if operation_id == "read_sensor_stream":
                return self._read_sensor_stream(request)
            return self._failure_result(
                summary="I couldn't understand that smart-home action.",
                error_code="unsupported_operation",
                operation_id=operation_id,
            )
        except ValueError as exc:
            return self._failure_result(
                summary=str(exc),
                error_code="invalid_request",
                operation_id=operation_id,
            )
        except RuntimeError as exc:
            logger.warning("Smart-home operation %r failed cleanly.", operation_id, exc_info=True)
            return self._failure_result(
                summary=str(exc),
                error_code="service_unavailable",
                operation_id=operation_id,
            )
        except Exception:
            logger.exception("Unexpected smart-home adapter failure for operation %r.", operation_id)
            return self._failure_result(
                summary="I couldn't complete the smart-home action right now. Please try again.",
                error_code="unexpected_error",
                operation_id=operation_id,
            )

    def _list_entities(self, request: IntegrationRequest) -> IntegrationResult:
        """Return filtered smart-home entities with optional aggregations."""

        params = _parameters(request)
        query = parse_entity_query_parameters(
            params,
            default_limit=self.settings.max_entity_results,
            maximum_limit=self.settings.max_entity_results,
        )
        provider_entity_class = query.entity_classes[0] if len(query.entity_classes) == 1 else None
        provider_entities = self.entity_provider.list_entities(
            entity_ids=query.entity_ids,
            entity_class=provider_entity_class,
            include_unavailable=query.include_unavailable or query.online is False,
        )
        filtered_entities = filter_entities(provider_entities, query)
        bounded_entities, next_cursor = paginate_entities(filtered_entities, query)
        aggregates = aggregate_entities(filtered_entities, query.aggregate_by)
        return IntegrationResult(
            ok=True,
            summary=f"{len(bounded_entities)} smart-home entities ready.",
            details={
                "count": len(bounded_entities),
                "total_count": len(filtered_entities),
                "returned_count": len(bounded_entities),
                "next_cursor": next_cursor,
                "truncated": next_cursor is not None,
                "applied_filters": entity_query_filters_payload(query),
                "entities": [entity.as_dict() for entity in bounded_entities],
                "aggregates": aggregates,
            },
        )

    def _control_entities(self, request: IntegrationRequest) -> IntegrationResult:
        """Validate one generic control request and dispatch it to the provider."""

        if self.controller is None:
            raise RuntimeError("I couldn't control the smart-home devices because the control service is unavailable.")

        params = _parameters(request)
        command_text = _parse_optional_text(params.get("command"), field_name="command")
        if command_text is None:
            raise ValueError("command must be provided for smart-home control.")
        command = SmartHomeCommand(command_text)
        requested_ids = _parse_string_tuple(
            params.get("entity_ids", params.get("entity_id")),
            field_name="entity_ids",
        )
        if not requested_ids:
            raise ValueError("entity_ids must contain at least one target.")
        if len(requested_ids) > self.settings.max_control_targets:
            raise ValueError(f"entity_ids must contain at most {self.settings.max_control_targets} targets.")

        entities = self.entity_provider.list_entities(entity_ids=requested_ids, include_unavailable=True)
        found_by_id = {entity.entity_id: entity for entity in entities}
        missing_ids = [entity_id for entity_id in requested_ids if entity_id not in found_by_id]
        if missing_ids:
            raise ValueError(f"Unknown smart-home target(s): {', '.join(missing_ids)}")

        for entity_id in requested_ids:
            entity = found_by_id[entity_id]
            if entity.entity_class not in self.settings.allowed_control_classes:
                raise ValueError(f"{entity.label} is not an allowed smart-home control target.")
            if command not in entity.supported_commands:
                raise ValueError(f"{entity.label} does not support {command.value}.")

        passthrough_parameters = {
            str(key): value
            for key, value in params.items()
            if key not in {"entity_ids", "entity_id", "command"}
        }
        details = self.controller.control(
            command=command,
            entity_ids=requested_ids,
            parameters=passthrough_parameters,
        )
        return IntegrationResult(
            ok=True,
            summary=f"Smart-home command {command.value} prepared for {len(requested_ids)} target(s).",
            details={
                "command": command.value,
                "entity_ids": list(requested_ids),
                "result": details,
            },
        )

    def _run_safe_scene(self, request: IntegrationRequest) -> IntegrationResult:
        """Preserve the legacy safe-scene operation on top of generic control."""

        params = _parameters(request)
        scene_id = _parse_optional_text(
            params.get("scene_id", params.get("entity_id")),
            field_name="scene_id",
        )
        if scene_id is None:
            raise ValueError("scene_id must be provided for run_safe_scene.")
        synthetic_request = IntegrationRequest(
            integration_id=request.integration_id,
            operation_id="control_entities",
            parameters={
                "command": SmartHomeCommand.ACTIVATE.value,
                "entity_ids": [scene_id],
            },
            origin=request.origin,
            explicit_user_confirmation=request.explicit_user_confirmation,
            explicit_caregiver_confirmation=request.explicit_caregiver_confirmation,
            dry_run=request.dry_run,
            background_trigger=request.background_trigger,
        )
        return self._control_entities(synthetic_request)

    def _read_sensor_stream(self, request: IntegrationRequest) -> IntegrationResult:
        """Return a filtered normalized smart-home event batch with aggregates."""

        if self.sensor_stream is None:
            raise RuntimeError("I couldn't read the smart-home sensor stream because the stream service is unavailable.")

        params = _parameters(request)
        query = parse_event_query_parameters(
            params,
            default_limit=self.settings.max_event_results,
            maximum_limit=self.settings.max_event_results,
        )
        raw_limit = self.settings.max_event_results if (
            query.aggregate_by
            or query.entity_ids
            or query.event_kinds
            or query.providers
            or query.areas
        ) else query.limit
        batch = self.sensor_stream.read_sensor_stream(cursor=query.cursor, limit=raw_limit)
        filtered_events = filter_events(batch.events, query)
        bounded_events = tuple(filtered_events[: query.limit])
        aggregates = aggregate_events(filtered_events, query.aggregate_by)
        return IntegrationResult(
            ok=True,
            summary=f"{len(bounded_events)} smart-home event(s) ready.",
            details={
                "events": [event.as_dict() for event in bounded_events],
                "next_cursor": batch.next_cursor,
                "stream_live": batch.stream_live,
                "count": len(bounded_events),
                "matched_count": len(filtered_events),
                "applied_filters": event_query_filters_payload(query),
                "aggregates": aggregates,
            },
        )

    @staticmethod
    def _failure_result(
        *,
        summary: str,
        error_code: str,
        operation_id: object,
    ) -> IntegrationResult:
        """Build one structured failure result."""

        return IntegrationResult(
            ok=False,
            summary=summary,
            details={
                "error_code": error_code,
                "operation_id": "" if operation_id is None else str(operation_id),
            },
        )


__all__ = [
    "SmartHomeAdapterSettings",
    "SmartHomeController",
    "SmartHomeEntityProvider",
    "SmartHomeIntegrationAdapter",
    "SmartHomeSensorStream",
]
