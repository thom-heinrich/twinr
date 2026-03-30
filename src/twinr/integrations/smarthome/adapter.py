# CHANGELOG: 2026-03-30
# BUG-1: `dry_run=True` no longer reaches the provider control path and cannot actuate real devices.
# BUG-2: Duplicate `entity_ids` are now deduplicated to prevent repeated/toggle-like double execution.
# BUG-3: Missing `parameters` are treated as `{}` so no-filter reads do not fail.
# BUG-4: Filtered sensor-stream pagination now preserves intra-batch offsets, preventing skipped events and misleading pagination.
# SEC-1: BREAKING: mutating control now requires explicit confirmation by default and blocks background-triggered control unless enabled.
# SEC-2: Added bounded control-parameter sanitization to block unsafe arbitrary passthrough payloads.
# SEC-3: Added in-process control rate limiting plus structured audit logging for smart-home actions.
# IMP-1: `run_safe_scene` now reuses the direct control path instead of constructing a synthetic `IntegrationRequest`.
# IMP-2: Added safe-scene allowlisting hooks and richer sensor-stream metadata (`scan_truncated`, scan counters, dry-run details).

"""Bridge generic smart-home providers onto Twinr's integration contract.

The adapter exposes one bounded surface for:
- user-driven read requests
- explicit control requests
- background sensor/event stream reads

Provider-specific logic belongs in child packages such as ``smarthome.hue``.
"""


from __future__ import annotations

import base64
import json
import logging
import math
import re
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from threading import Lock
from typing import Protocol, runtime_checkable

from twinr.integrations.adapter import IntegrationAdapter
from twinr.integrations.models import IntegrationManifest, IntegrationRequest, IntegrationResult
from twinr.integrations.smarthome.models import (
    SmartHomeCommand,
    SmartHomeEntity,
    SmartHomeEntityClass,
    SmartHomeEventBatch,
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

_EMPTY_PARAMETERS: Mapping[str, object] = {}
_CONTROL_PARAMETER_KEY_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_EVENT_CURSOR_PREFIX = "smarthome-stream:v1:"


def _parse_positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive whole number.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive whole number.") from exc
    if parsed < 1:
        raise ValueError(f"{field_name} must be a positive whole number.")
    return parsed


def _parse_optional_positive_int(value: object, *, field_name: str) -> int | None:
    if value in (None, "", 0, "0", False):
        return None
    return _parse_positive_int(value, field_name=field_name)


def _parse_non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative whole number.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a non-negative whole number.") from exc
    if parsed < 0:
        raise ValueError(f"{field_name} must be a non-negative whole number.")
    return parsed


def _parse_strict_bool(value: object, *, field_name: str) -> bool:
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
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")
    normalized = value.strip()
    return normalized or None


def _parse_string_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
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


def _dedupe_strings(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return tuple(deduped)


def _parameters(request: IntegrationRequest) -> Mapping[str, object]:
    params = getattr(request, "parameters", None)
    if params is None:
        return _EMPTY_PARAMETERS
    if not isinstance(params, Mapping):
        raise ValueError("parameters must be a mapping.")
    return params


def _parse_event_cursor(cursor: object) -> tuple[str | None, int]:
    if cursor is None:
        return None, 0
    if not isinstance(cursor, str):
        raise ValueError("cursor must be a string.")
    normalized = cursor.strip()
    if not normalized:
        return None, 0
    if not normalized.startswith(_EVENT_CURSOR_PREFIX):
        return normalized, 0

    token = normalized[len(_EVENT_CURSOR_PREFIX) :]
    if len(token) > 2048:
        raise ValueError("cursor is invalid.")
    padded = token + ("=" * (-len(token) % 4))
    try:
        raw_payload = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
        payload = json.loads(raw_payload)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("cursor is invalid.") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("cursor is invalid.")

    provider_cursor = payload.get("provider_cursor")
    if provider_cursor is not None and not isinstance(provider_cursor, str):
        raise ValueError("cursor is invalid.")
    filtered_offset = _parse_non_negative_int(payload.get("filtered_offset", 0), field_name="filtered_offset")
    return provider_cursor, filtered_offset


def _build_event_cursor(*, provider_cursor: str | None, filtered_offset: int) -> str:
    if provider_cursor is None and filtered_offset == 0:
        return ""
    payload = {
        "provider_cursor": provider_cursor,
        "filtered_offset": filtered_offset,
    }
    raw_payload = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    token = base64.urlsafe_b64encode(raw_payload).decode("ascii").rstrip("=")
    return f"{_EVENT_CURSOR_PREFIX}{token}"


@runtime_checkable
class SmartHomeEntityProvider(Protocol):
    def list_entities(
        self,
        *,
        entity_ids: tuple[str, ...] = (),
        entity_class: SmartHomeEntityClass | None = None,
        include_unavailable: bool = False,
    ) -> list[SmartHomeEntity]:
        ...


@runtime_checkable
class SmartHomeController(Protocol):
    def control(
        self,
        *,
        command: SmartHomeCommand,
        entity_ids: tuple[str, ...],
        parameters: Mapping[str, object],
    ) -> dict[str, object]:
        ...


@runtime_checkable
class SmartHomeSensorStream(Protocol):
    def read_sensor_stream(
        self,
        *,
        cursor: str | None = None,
        limit: int,
    ) -> SmartHomeEventBatch:
        ...


@dataclass(frozen=True, slots=True)
class SmartHomeAdapterSettings:
    max_entity_results: int = 32
    max_event_results: int = 32
    max_event_scan_batches: int = 4
    max_control_targets: int = 8
    max_control_parameter_keys: int = 16
    max_control_parameter_depth: int = 4
    max_control_sequence_length: int = 32
    max_control_string_length: int = 256
    # BREAKING: Mutating smart-home operations now require explicit confirmation by default.
    require_explicit_control_confirmation: bool = True
    # BREAKING: Background-triggered control is denied by default; opt in only for trusted automations.
    allow_background_control: bool = False
    # BREAKING: Rapid control bursts are rate-limited by default to reduce prompt-loop abuse.
    control_rate_limit_count: int | None = 12
    control_rate_limit_window_seconds: int = 30
    allowed_safe_scene_ids: tuple[str, ...] = ()
    allowed_control_classes: tuple[SmartHomeEntityClass, ...] = (
        SmartHomeEntityClass.LIGHT,
        SmartHomeEntityClass.LIGHT_GROUP,
        SmartHomeEntityClass.SCENE,
        SmartHomeEntityClass.SWITCH,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_entity_results",
            _parse_positive_int(self.max_entity_results, field_name="max_entity_results"),
        )
        object.__setattr__(
            self,
            "max_event_results",
            _parse_positive_int(self.max_event_results, field_name="max_event_results"),
        )
        object.__setattr__(
            self,
            "max_event_scan_batches",
            _parse_positive_int(self.max_event_scan_batches, field_name="max_event_scan_batches"),
        )
        object.__setattr__(
            self,
            "max_control_targets",
            _parse_positive_int(self.max_control_targets, field_name="max_control_targets"),
        )
        object.__setattr__(
            self,
            "max_control_parameter_keys",
            _parse_positive_int(self.max_control_parameter_keys, field_name="max_control_parameter_keys"),
        )
        object.__setattr__(
            self,
            "max_control_parameter_depth",
            _parse_positive_int(self.max_control_parameter_depth, field_name="max_control_parameter_depth"),
        )
        object.__setattr__(
            self,
            "max_control_sequence_length",
            _parse_positive_int(self.max_control_sequence_length, field_name="max_control_sequence_length"),
        )
        object.__setattr__(
            self,
            "max_control_string_length",
            _parse_positive_int(self.max_control_string_length, field_name="max_control_string_length"),
        )
        object.__setattr__(
            self,
            "require_explicit_control_confirmation",
            _parse_strict_bool(
                self.require_explicit_control_confirmation,
                field_name="require_explicit_control_confirmation",
            ),
        )
        object.__setattr__(
            self,
            "allow_background_control",
            _parse_strict_bool(self.allow_background_control, field_name="allow_background_control"),
        )
        object.__setattr__(
            self,
            "control_rate_limit_count",
            _parse_optional_positive_int(self.control_rate_limit_count, field_name="control_rate_limit_count"),
        )
        object.__setattr__(
            self,
            "control_rate_limit_window_seconds",
            _parse_positive_int(
                self.control_rate_limit_window_seconds,
                field_name="control_rate_limit_window_seconds",
            ),
        )
        object.__setattr__(
            self,
            "allowed_safe_scene_ids",
            _parse_string_tuple(self.allowed_safe_scene_ids, field_name="allowed_safe_scene_ids"),
        )
        object.__setattr__(
            self,
            "allowed_control_classes",
            tuple(SmartHomeEntityClass(item) for item in self.allowed_control_classes),
        )


@dataclass(slots=True)
class SmartHomeIntegrationAdapter(IntegrationAdapter):
    manifest: IntegrationManifest
    entity_provider: SmartHomeEntityProvider
    controller: SmartHomeController | None = None
    sensor_stream: SmartHomeSensorStream | None = None
    settings: SmartHomeAdapterSettings = field(default_factory=SmartHomeAdapterSettings)
    _control_rate_limit_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _control_rate_limit_timestamps: deque[float] = field(default_factory=deque, init=False, repr=False)

    def execute(self, request: IntegrationRequest) -> IntegrationResult:
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
        return self._execute_control_request(request)

    def _run_safe_scene(self, request: IntegrationRequest) -> IntegrationResult:
        params = _parameters(request)
        scene_id = _parse_optional_text(
            params.get("scene_id", params.get("entity_id")),
            field_name="scene_id",
        )
        if scene_id is None:
            raise ValueError("scene_id must be provided for run_safe_scene.")
        if self.settings.allowed_safe_scene_ids and scene_id not in self.settings.allowed_safe_scene_ids:
            raise ValueError(f"{scene_id} is not an approved safe scene.")
        return self._execute_control_request(
            request,
            override_command=SmartHomeCommand.ACTIVATE,
            override_entity_ids=(scene_id,),
        )

    def _execute_control_request(
        self,
        request: IntegrationRequest,
        *,
        override_command: SmartHomeCommand | None = None,
        override_entity_ids: tuple[str, ...] | None = None,
    ) -> IntegrationResult:
        if self.controller is None:
            raise RuntimeError("I couldn't control the smart-home devices because the control service is unavailable.")

        params = _parameters(request)
        command = override_command
        if command is None:
            command_text = _parse_optional_text(params.get("command"), field_name="command")
            if command_text is None:
                raise ValueError("command must be provided for smart-home control.")
            try:
                command = SmartHomeCommand(command_text)
            except ValueError as exc:
                raise ValueError(f"Unsupported smart-home command: {command_text}") from exc

        requested_ids = override_entity_ids
        if requested_ids is None:
            requested_ids = _parse_string_tuple(
                params.get("entity_ids", params.get("entity_id")),
                field_name="entity_ids",
            )
        requested_ids = _dedupe_strings(requested_ids)
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

        passthrough_parameters = self._sanitize_control_parameters(params)
        self._authorize_control_request(request)

        if bool(getattr(request, "dry_run", False)):
            logger.info(
                "Validated dry-run smart-home control op=%r command=%s targets=%s origin=%r.",
                getattr(request, "operation_id", None),
                command.value,
                requested_ids,
                getattr(request, "origin", None),
            )
            return IntegrationResult(
                ok=True,
                summary=f"Dry-run validated smart-home command {command.value} for {len(requested_ids)} target(s).",
                details={
                    "command": command.value,
                    "entity_ids": list(requested_ids),
                    "dry_run": True,
                    "validated": True,
                    "parameters": passthrough_parameters,
                },
            )

        self._enforce_control_rate_limit()
        logger.info(
            "Dispatching smart-home control op=%r command=%s targets=%s origin=%r background=%r confirmed=%r.",
            getattr(request, "operation_id", None),
            command.value,
            requested_ids,
            getattr(request, "origin", None),
            bool(getattr(request, "background_trigger", False)),
            bool(
                getattr(request, "explicit_user_confirmation", False)
                or getattr(request, "explicit_caregiver_confirmation", False)
            ),
        )
        details = self.controller.control(
            command=command,
            entity_ids=requested_ids,
            parameters=passthrough_parameters,
        )
        return IntegrationResult(
            ok=True,
            summary=f"Smart-home command {command.value} executed for {len(requested_ids)} target(s).",
            details={
                "command": command.value,
                "entity_ids": list(requested_ids),
                "dry_run": False,
                "result": details,
            },
        )

    def _authorize_control_request(self, request: IntegrationRequest) -> None:
        if bool(getattr(request, "dry_run", False)):
            return
        if bool(getattr(request, "background_trigger", False)) and not self.settings.allow_background_control:
            raise ValueError("Background-triggered smart-home control is disabled.")
        if self.settings.require_explicit_control_confirmation and not (
            bool(getattr(request, "explicit_user_confirmation", False))
            or bool(getattr(request, "explicit_caregiver_confirmation", False))
        ):
            raise ValueError("Smart-home control requires explicit confirmation.")

    def _enforce_control_rate_limit(self) -> None:
        if self.settings.control_rate_limit_count is None:
            return
        now = time.monotonic()
        window_seconds = self.settings.control_rate_limit_window_seconds
        cutoff = now - window_seconds
        with self._control_rate_limit_lock:
            while self._control_rate_limit_timestamps and self._control_rate_limit_timestamps[0] < cutoff:
                self._control_rate_limit_timestamps.popleft()
            if len(self._control_rate_limit_timestamps) >= self.settings.control_rate_limit_count:
                raise RuntimeError("Too many smart-home control requests arrived too quickly. Please wait a moment and try again.")
            self._control_rate_limit_timestamps.append(now)

    def _sanitize_control_parameters(self, params: Mapping[str, object]) -> dict[str, object]:
        sanitized: dict[str, object] = {}
        for raw_key, raw_value in params.items():
            if raw_key in {"entity_ids", "entity_id", "command", "scene_id"}:
                continue
            if not isinstance(raw_key, str):
                raise ValueError("control parameter names must be strings.")
            key = raw_key.strip()
            if not key:
                raise ValueError("control parameter names must not be blank.")
            if not _CONTROL_PARAMETER_KEY_RE.fullmatch(key):
                raise ValueError(f"Unsupported control parameter name: {raw_key}")
            if len(sanitized) >= self.settings.max_control_parameter_keys:
                raise ValueError(
                    f"At most {self.settings.max_control_parameter_keys} control parameters are allowed."
                )
            sanitized[key] = self._sanitize_control_parameter_value(raw_value, depth=0, field_name=key)
        return sanitized

    def _sanitize_control_parameter_value(
        self,
        value: object,
        *,
        depth: int,
        field_name: str,
    ) -> object:
        if depth > self.settings.max_control_parameter_depth:
            raise ValueError(
                f"{field_name} exceeds the maximum nesting depth of {self.settings.max_control_parameter_depth}."
            )
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be a finite number.")
            return value
        if isinstance(value, str):
            if len(value) > self.settings.max_control_string_length:
                raise ValueError(
                    f"{field_name} must be at most {self.settings.max_control_string_length} characters."
                )
            return value
        if isinstance(value, (list, tuple)):
            if len(value) > self.settings.max_control_sequence_length:
                raise ValueError(
                    f"{field_name} must contain at most {self.settings.max_control_sequence_length} items."
                )
            return [
                self._sanitize_control_parameter_value(
                    item,
                    depth=depth + 1,
                    field_name=f"{field_name}[{index}]",
                )
                for index, item in enumerate(value)
            ]
        if isinstance(value, Mapping):
            if len(value) > self.settings.max_control_parameter_keys:
                raise ValueError(
                    f"{field_name} must contain at most {self.settings.max_control_parameter_keys} keys."
                )
            sanitized_mapping: dict[str, object] = {}
            for raw_key, raw_item in value.items():
                if not isinstance(raw_key, str):
                    raise ValueError(f"{field_name} keys must be strings.")
                nested_key = raw_key.strip()
                if not nested_key:
                    raise ValueError(f"{field_name} keys must not be blank.")
                if not _CONTROL_PARAMETER_KEY_RE.fullmatch(nested_key):
                    raise ValueError(f"Unsupported nested control parameter name: {raw_key}")
                sanitized_mapping[nested_key] = self._sanitize_control_parameter_value(
                    raw_item,
                    depth=depth + 1,
                    field_name=f"{field_name}.{nested_key}",
                )
            return sanitized_mapping
        raise ValueError(f"{field_name} contains an unsupported value type.")

    def _read_sensor_stream(self, request: IntegrationRequest) -> IntegrationResult:
        if self.sensor_stream is None:
            raise RuntimeError("I couldn't read the smart-home sensor stream because the stream service is unavailable.")

        params = _parameters(request)
        query = parse_event_query_parameters(
            params,
            default_limit=self.settings.max_event_results,
            maximum_limit=self.settings.max_event_results,
        )

        provider_cursor, filtered_offset = _parse_event_cursor(query.cursor)
        returned_events: list[object] = []
        matched_events_scanned: list[object] = []
        next_cursor: str | None = None
        scan_truncated = False
        stream_live = False
        raw_batches_scanned = 0
        raw_events_scanned = 0

        for batch_index in range(self.settings.max_event_scan_batches):
            batch = self.sensor_stream.read_sensor_stream(
                cursor=provider_cursor,
                limit=self.settings.max_event_results,
            )
            raw_batches_scanned += 1
            raw_events_scanned += len(batch.events)
            stream_live = stream_live or batch.stream_live

            matched_in_batch = list(filter_events(batch.events, query))
            matched_events_scanned.extend(matched_in_batch)

            start_index = filtered_offset if batch_index == 0 else 0
            if start_index > len(matched_in_batch):
                raise ValueError("cursor is invalid or no longer available.")

            if next_cursor is None and len(returned_events) < query.limit:
                remaining_in_batch = matched_in_batch[start_index:]
                take_count = min(query.limit - len(returned_events), len(remaining_in_batch))
                returned_events.extend(remaining_in_batch[:take_count])

                if take_count < len(remaining_in_batch):
                    next_cursor = _build_event_cursor(
                        provider_cursor=provider_cursor,
                        filtered_offset=start_index + take_count,
                    )
                elif len(returned_events) >= query.limit:
                    next_cursor = batch.next_cursor

            filtered_offset = 0
            provider_cursor = batch.next_cursor

            need_more_for_page = len(returned_events) < query.limit and provider_cursor is not None
            need_more_for_aggregates = bool(query.aggregate_by) and provider_cursor is not None

            if not need_more_for_page and not need_more_for_aggregates:
                break
        else:
            if provider_cursor is not None:
                scan_truncated = True

        if next_cursor is None and provider_cursor is not None and len(returned_events) < query.limit:
            next_cursor = provider_cursor

        if provider_cursor is not None and (
            scan_truncated or (bool(query.aggregate_by) and raw_batches_scanned >= self.settings.max_event_scan_batches)
        ):
            scan_truncated = True

        aggregates = aggregate_events(matched_events_scanned, query.aggregate_by)
        return IntegrationResult(
            ok=True,
            summary=f"{len(returned_events)} smart-home event(s) ready.",
            details={
                "events": [event.as_dict() for event in returned_events],
                "next_cursor": next_cursor,
                "stream_live": stream_live,
                "count": len(returned_events),
                "returned_count": len(returned_events),
                "matched_count": len(matched_events_scanned),
                "applied_filters": event_query_filters_payload(query),
                "aggregates": aggregates,
                "scan_truncated": scan_truncated,
                "raw_batches_scanned": raw_batches_scanned,
                "raw_events_scanned": raw_events_scanned,
            },
        )

    @staticmethod
    def _failure_result(
        *,
        summary: str,
        error_code: str,
        operation_id: object,
    ) -> IntegrationResult:
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
