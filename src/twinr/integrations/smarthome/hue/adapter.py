"""Normalize Hue bridge resources into Twinr's generic smart-home surface."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from twinr.integrations.smarthome.adapter import SmartHomeController, SmartHomeEntityProvider, SmartHomeSensorStream
from twinr.integrations.smarthome.hue.client import HueBridgeClient
from twinr.integrations.smarthome.models import (
    SmartHomeCommand,
    SmartHomeEntity,
    SmartHomeEntityClass,
    SmartHomeEvent,
    SmartHomeEventBatch,
    SmartHomeEventKind,
)


class HueClientLike(Protocol):
    """Describe the Hue client surface used by the provider."""

    def list_resources(self) -> list[dict[str, object]]:
        """Return raw bridge resources."""

        ...

    def put_resource(
        self,
        resource_type: str,
        resource_id: str,
        payload: Mapping[str, object],
    ) -> dict[str, object]:
        """Update one raw bridge resource."""

        ...

    def read_event_stream(
        self,
        *,
        timeout_s: float | None = None,
        max_events: int = 20,
    ) -> list[dict[str, object]]:
        """Read a bounded batch of raw event-stream messages."""

        ...


def _coerce_mapping(value: object) -> dict[str, object]:
    """Convert a mapping-like object into a plain dict."""

    return dict(value) if isinstance(value, Mapping) else {}


def _mapping_get(mapping: Mapping[str, object], key: str) -> dict[str, object]:
    """Return one nested mapping or an empty dict."""

    value = mapping.get(key)
    return _coerce_mapping(value)


def _first_non_empty_text(*values: object) -> str | None:
    """Return the first non-empty text value from a candidate list."""

    for value in values:
        if isinstance(value, str):
            normalized = value.strip()
            if normalized:
                return normalized
    return None


_RESOURCE_CLASS_MAP: dict[str, SmartHomeEntityClass] = {
    "light": SmartHomeEntityClass.LIGHT,
    "grouped_light": SmartHomeEntityClass.LIGHT_GROUP,
    "scene": SmartHomeEntityClass.SCENE,
    "motion": SmartHomeEntityClass.MOTION_SENSOR,
    "light_level": SmartHomeEntityClass.LIGHT_SENSOR,
    "temperature": SmartHomeEntityClass.TEMPERATURE_SENSOR,
    "device_power": SmartHomeEntityClass.BATTERY_SENSOR,
    "zigbee_connectivity": SmartHomeEntityClass.DEVICE_HEALTH,
    "button": SmartHomeEntityClass.BUTTON,
}

_CONTROL_COMMANDS_BY_RESOURCE_TYPE: dict[str, tuple[SmartHomeCommand, ...]] = {
    "light": (
        SmartHomeCommand.TURN_ON,
        SmartHomeCommand.TURN_OFF,
        SmartHomeCommand.SET_BRIGHTNESS,
    ),
    "grouped_light": (
        SmartHomeCommand.TURN_ON,
        SmartHomeCommand.TURN_OFF,
        SmartHomeCommand.SET_BRIGHTNESS,
    ),
    "scene": (SmartHomeCommand.ACTIVATE,),
}


@dataclass(slots=True)
class HueSmartHomeProvider(SmartHomeEntityProvider, SmartHomeController, SmartHomeSensorStream):
    """Expose Hue lights, scenes, sensors, and bridge events through generic Twinr contracts."""

    client: HueClientLike
    provider_id: str = "hue"
    event_timeout_s: float = 2.0

    def list_entities(
        self,
        *,
        entity_ids: tuple[str, ...] = (),
        entity_class: SmartHomeEntityClass | None = None,
        include_unavailable: bool = False,
    ) -> list[SmartHomeEntity]:
        """Return normalized Hue resources filtered for Twinr callers."""

        resources = self.client.list_resources()
        device_names = self._device_name_index(resources)
        area_names = self._area_name_index(resources)
        requested_id_set = set(entity_ids)
        collected: list[SmartHomeEntity] = []
        for resource in resources:
            entity = self._entity_from_resource(resource, device_names=device_names, area_names=area_names)
            if entity is None:
                continue
            if requested_id_set and entity.entity_id not in requested_id_set:
                continue
            if entity_class is not None and entity.entity_class is not entity_class:
                continue
            if not include_unavailable and not entity.online:
                continue
            collected.append(entity)
        collected.sort(key=lambda item: (item.label.casefold(), item.entity_id))
        return collected

    def control(
        self,
        *,
        command: SmartHomeCommand,
        entity_ids: tuple[str, ...],
        parameters: Mapping[str, object],
    ) -> dict[str, object]:
        """Translate generic commands into Hue CLIP v2 resource updates."""

        if not entity_ids:
            raise ValueError("entity_ids must not be empty.")
        resources = {str(resource.get("id", "")).strip(): resource for resource in self.client.list_resources()}
        results: list[dict[str, object]] = []
        for entity_id in entity_ids:
            resource = resources.get(entity_id)
            if not isinstance(resource, Mapping):
                raise ValueError(f"Unknown Hue entity: {entity_id}")
            resource_type = str(resource.get("type", "")).strip()
            payload = self._control_payload(resource_type, command=command, parameters=parameters)
            response = self.client.put_resource(resource_type, entity_id, payload)
            results.append(
                {
                    "entity_id": entity_id,
                    "resource_type": resource_type,
                    "response": response,
                }
            )
        return {"targets": results}

    def read_sensor_stream(
        self,
        *,
        cursor: str | None = None,
        limit: int,
    ) -> SmartHomeEventBatch:
        """Return a bounded batch of normalized Hue event-stream updates."""

        del cursor  # Hue live streams do not support replay cursors through this bounded helper.
        raw_events = self.client.read_event_stream(timeout_s=self.event_timeout_s, max_events=limit)
        normalized_events: list[SmartHomeEvent] = []
        last_cursor: str | None = None
        for event in raw_events:
            last_cursor = _first_non_empty_text(event.get("id")) or last_cursor
            normalized_events.extend(self._normalize_stream_event(event))
            if len(normalized_events) >= limit:
                break
        return SmartHomeEventBatch(events=tuple(normalized_events[:limit]), next_cursor=last_cursor, stream_live=True)

    def _device_name_index(self, resources: list[dict[str, object]]) -> dict[str, str]:
        """Map Hue device resource ids onto human-readable labels."""

        names: dict[str, str] = {}
        for resource in resources:
            if str(resource.get("type", "")).strip() != "device":
                continue
            metadata = _mapping_get(resource, "metadata")
            label = _first_non_empty_text(
                metadata.get("name"),
                _mapping_get(resource, "product_data").get("product_name"),
            )
            resource_id = _first_non_empty_text(resource.get("id"))
            if label is None or resource_id is None:
                continue
            names[resource_id] = label
        return names

    def _area_name_index(self, resources: list[dict[str, object]]) -> dict[str, str]:
        """Map child resource ids onto room or zone names."""

        areas: dict[str, str] = {}
        for resource in resources:
            resource_type = str(resource.get("type", "")).strip()
            if resource_type not in {"room", "zone"}:
                continue
            metadata = _mapping_get(resource, "metadata")
            area_name = _first_non_empty_text(metadata.get("name"))
            if area_name is None:
                continue
            children = resource.get("children", ())
            if not isinstance(children, list):
                continue
            for child in children:
                if isinstance(child, Mapping):
                    child_id = _first_non_empty_text(child.get("rid"))
                    if child_id is not None:
                        areas[child_id] = area_name
        return areas

    def _entity_from_resource(
        self,
        resource: Mapping[str, object],
        *,
        device_names: Mapping[str, str],
        area_names: Mapping[str, str],
    ) -> SmartHomeEntity | None:
        """Convert one Hue resource into a generic smart-home entity."""

        resource_type = str(resource.get("type", "")).strip()
        entity_class = _RESOURCE_CLASS_MAP.get(resource_type)
        if entity_class is None:
            return None
        entity_id = _first_non_empty_text(resource.get("id"))
        if entity_id is None:
            return None
        owner = _mapping_get(resource, "owner")
        owner_id = _first_non_empty_text(owner.get("rid"))
        metadata = _mapping_get(resource, "metadata")
        base_label = _first_non_empty_text(metadata.get("name"))
        owner_label = None if owner_id is None else device_names.get(owner_id)
        label = base_label or owner_label or f"{resource_type.replace('_', ' ')} {entity_id[:8]}"
        area = area_names.get(entity_id) or ("" if owner_id is None else area_names.get(owner_id, ""))
        supported_commands = _CONTROL_COMMANDS_BY_RESOURCE_TYPE.get(resource_type, ())
        return SmartHomeEntity(
            entity_id=entity_id,
            provider=self.provider_id,
            label=label,
            entity_class=entity_class,
            area=area,
            readable=True,
            controllable=bool(supported_commands),
            online=self._is_online(resource_type, resource),
            supported_commands=supported_commands,
            state=self._state_from_resource(resource_type, resource),
            attributes={"resource_type": resource_type},
        )

    @staticmethod
    def _is_online(resource_type: str, resource: Mapping[str, object]) -> bool:
        """Infer a conservative online flag from one Hue resource."""

        if resource_type == "zigbee_connectivity":
            return str(resource.get("status", "")).strip().lower() == "connected"
        return True

    @staticmethod
    def _state_from_resource(resource_type: str, resource: Mapping[str, object]) -> dict[str, object]:
        """Extract a compact current-state payload from one Hue resource."""

        if resource_type in {"light", "grouped_light"}:
            return {
                "on": bool(_mapping_get(resource, "on").get("on", False)),
                "brightness": _mapping_get(resource, "dimming").get("brightness"),
            }
        if resource_type == "scene":
            return {
                "status": _mapping_get(resource, "status").get("active"),
            }
        if resource_type == "motion":
            motion = _mapping_get(resource, "motion")
            report = _mapping_get(motion, "motion_report")
            return {
                "motion": motion.get("motion", report.get("motion")),
                "motion_valid": motion.get("motion_valid"),
            }
        if resource_type == "zigbee_connectivity":
            return {"status": resource.get("status")}
        if resource_type == "button":
            return {"last_event": _mapping_get(resource, "button").get("last_event")}
        if resource_type == "device_power":
            return {"power_state": _mapping_get(resource, "power_state")}
        if resource_type == "temperature":
            return {"temperature": _mapping_get(resource, "temperature")}
        if resource_type == "light_level":
            return {"light": _mapping_get(resource, "light"), "light_level": _mapping_get(resource, "light_level")}
        return {}

    def _control_payload(
        self,
        resource_type: str,
        *,
        command: SmartHomeCommand,
        parameters: Mapping[str, object],
    ) -> dict[str, object]:
        """Build one Hue resource update payload from a generic command."""

        if command in {SmartHomeCommand.TURN_ON, SmartHomeCommand.TURN_OFF}:
            if resource_type not in {"light", "grouped_light"}:
                raise ValueError(f"{command.value} is not supported for Hue resource type {resource_type}.")
            return {"on": {"on": command is SmartHomeCommand.TURN_ON}}
        if command is SmartHomeCommand.SET_BRIGHTNESS:
            if resource_type not in {"light", "grouped_light"}:
                raise ValueError("set_brightness is only supported for Hue lights.")
            brightness = parameters.get("brightness")
            if isinstance(brightness, bool):
                raise ValueError("brightness must be a number between 0 and 100.")
            try:
                brightness_value = float(brightness)
            except (TypeError, ValueError) as exc:
                raise ValueError("brightness must be a number between 0 and 100.") from exc
            if brightness_value < 0.0 or brightness_value > 100.0:
                raise ValueError("brightness must be between 0 and 100.")
            return {"on": {"on": True}, "dimming": {"brightness": brightness_value}}
        if command is SmartHomeCommand.ACTIVATE:
            if resource_type != "scene":
                raise ValueError("activate is only supported for Hue scenes.")
            return {"recall": {"action": "active"}}
        raise ValueError(f"Unsupported Hue smart-home command: {command.value}")

    def _normalize_stream_event(self, event: Mapping[str, object]) -> list[SmartHomeEvent]:
        """Convert one raw Hue SSE payload into normalized smart-home events."""

        raw_data = event.get("data", ())
        if not isinstance(raw_data, list):
            return []
        event_id_prefix = _first_non_empty_text(event.get("id")) or "hue"
        collected: list[SmartHomeEvent] = []
        for batch_index, batch_entry in enumerate(raw_data):
            if not isinstance(batch_entry, Mapping):
                continue
            changed_resources = batch_entry.get("data", ())
            if not isinstance(changed_resources, list):
                continue
            for resource_index, raw_resource in enumerate(changed_resources):
                if not isinstance(raw_resource, Mapping):
                    continue
                normalized = self._event_from_changed_resource(
                    raw_resource,
                    event_id=f"{event_id_prefix}:{batch_index}:{resource_index}",
                )
                if normalized is not None:
                    collected.append(normalized)
        return collected

    def _event_from_changed_resource(
        self,
        resource: Mapping[str, object],
        *,
        event_id: str,
    ) -> SmartHomeEvent | None:
        """Convert one changed Hue resource into a generic smart-home event."""

        resource_type = str(resource.get("type", "")).strip()
        entity_id = _first_non_empty_text(resource.get("id"))
        if entity_id is None:
            return None
        observed_at = self._observed_at_for_event(resource_type, resource) or event_id
        details = {"resource_type": resource_type, "raw": dict(resource)}

        if resource_type == "motion":
            motion = _mapping_get(resource, "motion")
            report = _mapping_get(motion, "motion_report")
            motion_active = report.get("motion", motion.get("motion"))
            if motion_active is True:
                event_kind = SmartHomeEventKind.MOTION_DETECTED
            elif motion_active is False:
                event_kind = SmartHomeEventKind.MOTION_CLEARED
            else:
                event_kind = SmartHomeEventKind.STATE_CHANGED
        elif resource_type == "zigbee_connectivity":
            event_kind = (
                SmartHomeEventKind.DEVICE_ONLINE
                if str(resource.get("status", "")).strip().lower() == "connected"
                else SmartHomeEventKind.DEVICE_OFFLINE
            )
        elif resource_type == "button":
            event_kind = SmartHomeEventKind.BUTTON_PRESSED
        else:
            event_kind = SmartHomeEventKind.STATE_CHANGED

        return SmartHomeEvent(
            event_id=event_id,
            provider=self.provider_id,
            entity_id=entity_id,
            event_kind=event_kind,
            observed_at=observed_at,
            details=details,
        )

    @staticmethod
    def _observed_at_for_event(resource_type: str, resource: Mapping[str, object]) -> str | None:
        """Extract the most specific timestamp available for one changed resource."""

        if resource_type == "motion":
            motion = _mapping_get(resource, "motion")
            report = _mapping_get(motion, "motion_report")
            return _first_non_empty_text(report.get("changed"))
        return _first_non_empty_text(resource.get("changed"))


def build_hue_smart_home_provider(client: HueBridgeClient) -> HueSmartHomeProvider:
    """Build the generic Hue provider from one bridge client."""

    return HueSmartHomeProvider(client=client)


__all__ = ["HueSmartHomeProvider", "build_hue_smart_home_provider"]
