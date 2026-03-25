"""Handle smart-home read, control, and sensor-stream tool calls."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from twinr.integrations import IntegrationRequest, SmartHomeIntegrationAdapter, build_smart_home_hub_adapter

from .handler_telemetry import emit_best_effort, record_event_best_effort
from .support import optional_bool, require_sensitive_voice_confirmation

_MAX_ENTITY_IDS = 8
_MAX_ENTITY_LIMIT = 32
_MAX_STREAM_LIMIT = 20


def handle_list_smart_home_entities(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """List smart-home entities with optional generic filters and aggregations."""

    adapter = _require_smart_home_adapter(owner)
    params = _collect_parameters(arguments, allow_confirmed=False)
    if "limit" in params:
        params["limit"] = _bounded_positive_int(
            params["limit"],
            field_name="limit",
            maximum=_MAX_ENTITY_LIMIT,
        )
    result = adapter.execute(
        IntegrationRequest(
            integration_id="smart_home_hub",
            operation_id="list_entities",
            parameters=params,
        )
    )
    return _result_payload(owner, "list_smart_home_entities", result)


def handle_read_smart_home_state(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """Read exact state for one or more smart-home entities."""

    adapter = _require_smart_home_adapter(owner)
    params = _collect_parameters(arguments, allow_confirmed=False)
    entity_ids = _string_list(params.get("entity_ids"), field_name="entity_ids")
    if not entity_ids:
        raise RuntimeError("read_smart_home_state requires at least one entity_id.")
    params["entity_ids"] = entity_ids
    result = adapter.execute(
        IntegrationRequest(
            integration_id="smart_home_hub",
            operation_id="read_device_state",
            parameters=params,
        )
    )
    return _result_payload(owner, "read_smart_home_state", result)


def handle_control_smart_home_entities(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """Control allowed low-risk smart-home entities such as lights and scenes."""

    require_sensitive_voice_confirmation(owner, arguments, action_label="control smart-home devices")
    adapter = _require_smart_home_adapter(owner)
    params = _collect_parameters(arguments, allow_confirmed=True)
    entity_ids = _string_list(params.get("entity_ids"), field_name="entity_ids")
    if not entity_ids:
        raise RuntimeError("control_smart_home_entities requires at least one entity_id.")
    params["entity_ids"] = entity_ids
    result = adapter.execute(
        IntegrationRequest(
            integration_id="smart_home_hub",
            operation_id="control_entities",
            parameters=params,
            explicit_user_confirmation=bool(optional_bool(arguments, "confirmed", default=False)),
        )
    )
    return _result_payload(owner, "control_smart_home_entities", result)


def handle_read_smart_home_sensor_stream(owner: Any, arguments: dict[str, object]) -> dict[str, object]:
    """Read a bounded smart-home event batch with optional selectors and aggregates."""

    adapter = _require_smart_home_adapter(owner)
    params = _collect_parameters(arguments, allow_confirmed=False)
    if "limit" in params:
        params["limit"] = _bounded_positive_int(
            params["limit"],
            field_name="limit",
            maximum=_MAX_STREAM_LIMIT,
        )
    result = adapter.execute(
        IntegrationRequest(
            integration_id="smart_home_hub",
            operation_id="read_sensor_stream",
            parameters=params,
        )
    )
    return _result_payload(owner, "read_smart_home_sensor_stream", result)


def _require_smart_home_adapter(owner: Any) -> SmartHomeIntegrationAdapter:
    config = getattr(owner, "config", None)
    project_root = Path(str(getattr(config, "project_root", ".") or ".")).resolve()
    adapter = build_smart_home_hub_adapter(project_root)
    if not isinstance(adapter, SmartHomeIntegrationAdapter):
        raise RuntimeError(
            "The smart-home integration is not ready. Check the Hue bridge settings on the Integrations page."
        )
    return adapter


def _collect_parameters(arguments: dict[str, object], *, allow_confirmed: bool) -> dict[str, object]:
    if not isinstance(arguments, Mapping):
        raise RuntimeError("tool arguments must be a JSON object")
    params: dict[str, object] = {}
    for raw_key, raw_value in arguments.items():
        key = str(raw_key).strip()
        if not key:
            continue
        if not allow_confirmed and key == "confirmed":
            continue
        params[key] = raw_value
    if "entity_ids" in params:
        params["entity_ids"] = _string_list(params["entity_ids"], field_name="entity_ids")
    return params


def _string_list(value: object, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        normalized = value.strip()
        return [normalized] if normalized else []
    if not isinstance(value, (list, tuple)):
        raise RuntimeError(f"{field_name} must be a string or a list of strings.")
    normalized_values: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise RuntimeError(f"{field_name}[{index}] must be a non-empty string.")
        normalized_values.append(item.strip())
    if len(normalized_values) > _MAX_ENTITY_IDS:
        raise RuntimeError(f"{field_name} must contain at most {_MAX_ENTITY_IDS} values.")
    return normalized_values


def _bounded_positive_int(value: object, *, field_name: str, maximum: int) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"{field_name} must be a positive whole number.")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        if not value.is_integer():
            raise RuntimeError(f"{field_name} must be a positive whole number.")
        parsed = int(value)
    elif isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError as exc:
            raise RuntimeError(f"{field_name} must be a positive whole number.") from exc
    else:
        raise RuntimeError(f"{field_name} must be a positive whole number.")
    if parsed < 1:
        raise RuntimeError(f"{field_name} must be a positive whole number.")
    return min(maximum, parsed)


def _result_payload(owner: Any, tool_name: str, result) -> dict[str, object]:
    if not bool(getattr(result, "ok", False)):
        raise RuntimeError(str(getattr(result, "summary", "The smart-home request failed.")))
    _emit_safe(owner, "smart_home_tool_call=true")
    _emit_safe(owner, f"smart_home_tool_name={tool_name}")
    _record_event_safe(owner, "smart_home_tool_succeeded", f"Realtime tool ran {tool_name}.")
    payload = {
        "status": "ok",
        "summary": result.summary,
        **dict(getattr(result, "details", {}) or {}),
    }
    warnings = tuple(getattr(result, "warnings", ()) or ())
    if warnings:
        payload["warnings"] = list(warnings)
    return payload


def _emit_safe(owner: Any, message: str) -> None:
    emit_best_effort(owner, message)


def _record_event_safe(owner: Any, event_name: str, message: str) -> None:
    record_event_best_effort(owner, event_name, message)


__all__ = [
    "handle_control_smart_home_entities",
    "handle_list_smart_home_entities",
    "handle_read_smart_home_sensor_stream",
    "handle_read_smart_home_state",
]
