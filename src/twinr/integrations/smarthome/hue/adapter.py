# CHANGELOG: 2026-03-30
# BUG-1: Fixed Hue room/zone area resolution for grouped_light and scene resources by indexing both area resources and their services/children.
# BUG-2: Fixed silent brightness=0 misbehavior; Hue dimming cannot write 0 and otherwise falls back to minimum brightness instead of off.
# BUG-3: Fixed non-temporal event timestamps and broadened SSE parsing to handle both direct event objects and array-wrapped Hue eventstream messages.
# SEC-1: Added strict validation for bridge-sourced resource ids/types before issuing PUT requests.
# SEC-2: Bounded copied event payloads in SmartHomeEvent.details["raw"] to reduce practical memory/log-amplification risk on Raspberry Pi deployments.
# IMP-1: Added 2026 Hue-v2 capability support for transition-aware light/group control, scene recall options (dynamic, speed, duration, brightness), modern effects/effects_v2 payloads, and state-while-off semantics via power_on=false.
# IMP-2: Added support for newer Hue resource families where Twinr has compatible generic classes (camera_motion, convenience_area_motion, security_area_motion, grouped_motion, grouped_light_level, bell_button, and optional contact/tamper mappings).
# IMP-3: Added short-lived resource caching and richer normalized state extraction to reduce bridge load and expose current Hue-v2 feature state more accurately.

"""Normalize Hue bridge resources into Twinr's generic smart-home surface."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import monotonic
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
        return []

    def put_resource(
        self,
        resource_type: str,
        resource_id: str,
        payload: Mapping[str, object],
    ) -> dict[str, object]:
        """Update one raw bridge resource."""
        del resource_type, resource_id, payload
        return {}

    def read_event_stream(
        self,
        *,
        timeout_s: float | None = None,
        max_events: int = 20,
    ) -> list[object]:
        """Read a bounded batch of raw event-stream messages."""
        del timeout_s, max_events
        return []


_VALID_RESOURCE_TYPE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_VALID_RESOURCE_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_STREAM_EVENT_TYPES = {"update", "add", "delete", "error"}
_CONNECTIVITY_RESOURCE_TYPES = {"zigbee_connectivity", "matter_fabric", "matter"}
_MOTION_RESOURCE_TYPES = {
    "motion",
    "camera_motion",
    "convenience_area_motion",
    "security_area_motion",
    "grouped_motion",
}
_LIGHT_LEVEL_RESOURCE_TYPES = {"light_level", "grouped_light_level"}
_BUTTON_RESOURCE_TYPES = {"button", "bell_button"}


def _coerce_mapping(value: object) -> dict[str, object]:
    """Convert a mapping-like object into a plain dict."""
    return dict(value) if isinstance(value, Mapping) else {}


def _mapping_get(mapping: Mapping[str, object], key: str) -> dict[str, object]:
    """Return one nested mapping or an empty dict."""
    value = mapping.get(key)
    return _coerce_mapping(value)


def _as_list(value: object) -> list[object]:
    """Return a list-like value or an empty list."""
    return value if isinstance(value, list) else []


def _first_non_empty_text(*values: object) -> str | None:
    """Return the first non-empty text value from a candidate list."""
    for value in values:
        if isinstance(value, str):
            normalized = value.strip()
            if normalized:
                return normalized
    return None


def _coerce_bool(value: object) -> bool | None:
    """Coerce one generic value to bool when possible."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return None


def _coerce_float(value: object) -> float | None:
    """Coerce one generic numeric-looking value to float."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _enum_member(enum_type: type[object], *names: str) -> object | None:
    """Return the first existing enum member from a candidate-name list."""
    for name in names:
        member = getattr(enum_type, name, None)
        if member is not None:
            return member
    return None


def _existing_command_members(*candidate_name_groups: tuple[str, ...]) -> tuple[SmartHomeCommand, ...]:
    """Resolve only those SmartHomeCommand members that exist in the current Twinr build."""
    collected: list[SmartHomeCommand] = []
    for group in candidate_name_groups:
        member = _enum_member(SmartHomeCommand, *group)
        if isinstance(member, SmartHomeCommand):
            collected.append(member)
    return tuple(collected)


def _validate_resource_address(resource_type: str, resource_id: str) -> None:
    """Validate bridge resource path fragments before they are used in a write request."""
    if not _VALID_RESOURCE_TYPE_RE.fullmatch(resource_type):
        raise ValueError(f"Invalid Hue resource type: {resource_type!r}")
    if not _VALID_RESOURCE_ID_RE.fullmatch(resource_id):
        raise ValueError(f"Invalid Hue resource id: {resource_id!r}")


_RESOURCE_CLASS_MAP: dict[str, SmartHomeEntityClass] = {
    "light": SmartHomeEntityClass.LIGHT,
    "grouped_light": SmartHomeEntityClass.LIGHT_GROUP,
    "scene": SmartHomeEntityClass.SCENE,
    "motion": SmartHomeEntityClass.MOTION_SENSOR,
    "camera_motion": SmartHomeEntityClass.MOTION_SENSOR,
    "convenience_area_motion": SmartHomeEntityClass.MOTION_SENSOR,
    "security_area_motion": SmartHomeEntityClass.MOTION_SENSOR,
    "grouped_motion": SmartHomeEntityClass.MOTION_SENSOR,
    "light_level": SmartHomeEntityClass.LIGHT_SENSOR,
    "grouped_light_level": SmartHomeEntityClass.LIGHT_SENSOR,
    "temperature": SmartHomeEntityClass.TEMPERATURE_SENSOR,
    "device_power": SmartHomeEntityClass.BATTERY_SENSOR,
    "zigbee_connectivity": SmartHomeEntityClass.DEVICE_HEALTH,
    "button": SmartHomeEntityClass.BUTTON,
    "bell_button": SmartHomeEntityClass.BUTTON,
}
_optional_contact_class = _enum_member(SmartHomeEntityClass, "CONTACT_SENSOR", "DOOR_SENSOR", "OPEN_CLOSE_SENSOR")
if isinstance(_optional_contact_class, SmartHomeEntityClass):
    _RESOURCE_CLASS_MAP["contact"] = _optional_contact_class
_optional_tamper_class = _enum_member(SmartHomeEntityClass, "TAMPER_SENSOR", "DEVICE_HEALTH")
if isinstance(_optional_tamper_class, SmartHomeEntityClass):
    _RESOURCE_CLASS_MAP["tamper"] = _optional_tamper_class

_LIGHT_CONTROL_COMMANDS = _existing_command_members(
    ("TURN_ON",),
    ("TURN_OFF",),
    ("SET_BRIGHTNESS",),
    ("SET_COLOR_TEMPERATURE", "SET_COLOR_TEMP"),
    ("SET_COLOR", "SET_XY_COLOR"),
    ("SET_EFFECT",),
)
_CONTROL_COMMANDS_BY_RESOURCE_TYPE: dict[str, tuple[SmartHomeCommand, ...]] = {
    "light": _LIGHT_CONTROL_COMMANDS,
    "grouped_light": _LIGHT_CONTROL_COMMANDS,
    "scene": _existing_command_members(("ACTIVATE",)),
}


@dataclass(slots=True)
class _HueIndexes:
    """Derived indexes built from one Hue resource snapshot."""

    device_names: dict[str, str]
    area_names: dict[str, str]
    group_names: dict[str, str]
    scene_groups: dict[str, str]
    device_online: dict[str, bool]
    group_online: dict[str, bool]


@dataclass(slots=True)
class HueSmartHomeProvider(SmartHomeEntityProvider, SmartHomeController, SmartHomeSensorStream):
    """Expose Hue lights, scenes, sensors, and bridge events through generic Twinr contracts."""

    client: HueClientLike
    provider_id: str = "hue"
    event_timeout_s: float = 2.0
    resource_cache_ttl_s: float = 1.0
    max_event_detail_depth: int = 3
    max_event_detail_keys: int = 32
    max_event_detail_items: int = 16
    max_event_text_length: int = 256
    _resource_cache: tuple[float, tuple[dict[str, object], ...]] | None = field(default=None, init=False, repr=False)

    def list_entities(
        self,
        *,
        entity_ids: tuple[str, ...] = (),
        entity_class: SmartHomeEntityClass | None = None,
        include_unavailable: bool = False,
    ) -> list[SmartHomeEntity]:
        """Return normalized Hue resources filtered for Twinr callers."""
        resources = self._get_resources()
        indexes = self._build_indexes(resources)
        requested_id_set = set(entity_ids)
        collected: list[SmartHomeEntity] = []
        for resource in resources:
            entity = self._entity_from_resource(resource, indexes=indexes)
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

        resources = {str(resource.get("id", "")).strip(): resource for resource in self._get_resources()}
        results: list[dict[str, object]] = []
        for entity_id in entity_ids:
            resource = resources.get(entity_id)
            if not isinstance(resource, Mapping):
                self._invalidate_resource_cache()
                resources = {str(item.get("id", "")).strip(): item for item in self._get_resources()}
                resource = resources.get(entity_id)
            if not isinstance(resource, Mapping):
                raise ValueError(f"Unknown Hue entity: {entity_id}")

            resource_type = str(resource.get("type", "")).strip()
            _validate_resource_address(resource_type, entity_id)
            payload = self._control_payload(resource_type, command=command, parameters=parameters)
            response = self.client.put_resource(resource_type, entity_id, payload)
            results.append(
                {
                    "entity_id": entity_id,
                    "resource_type": resource_type,
                    "request": payload,
                    "response": response,
                }
            )

        self._invalidate_resource_cache()
        return {"targets": results}

    def read_sensor_stream(
        self,
        *,
        cursor: str | None = None,
        limit: int,
    ) -> SmartHomeEventBatch:
        """Return a bounded batch of normalized Hue event-stream updates."""
        del cursor  # Hue live streams do not support replay cursors through this bounded helper.
        if limit <= 0:
            return SmartHomeEventBatch(events=(), next_cursor=None, stream_live=True)

        raw_messages = self.client.read_event_stream(
            timeout_s=self.event_timeout_s,
            max_events=max(1, min(limit * 2, 100)),
        )

        normalized_events: list[SmartHomeEvent] = []
        last_cursor: str | None = None
        for raw_message in raw_messages:
            for event in self._iter_stream_events(raw_message):
                last_cursor = _first_non_empty_text(event.get("id")) or last_cursor
                event_creationtime = _first_non_empty_text(event.get("creationtime"))
                normalized_events.extend(
                    self._normalize_stream_event(event, event_creationtime=event_creationtime)
                )
                if len(normalized_events) >= limit:
                    break
            if len(normalized_events) >= limit:
                break

        return SmartHomeEventBatch(
            events=tuple(normalized_events[:limit]),
            next_cursor=last_cursor,
            stream_live=True,
        )

    def _get_resources(self) -> list[dict[str, object]]:
        """Return a short-lived cached bridge snapshot."""
        cached = self._resource_cache
        now = monotonic()
        if cached is not None:
            expires_at, resources = cached
            if now < expires_at:
                return [dict(resource) for resource in resources]

        resources = self.client.list_resources()
        normalized_resources = tuple(dict(resource) for resource in resources if isinstance(resource, Mapping))
        self._resource_cache = (now + max(self.resource_cache_ttl_s, 0.0), normalized_resources)
        return [dict(resource) for resource in normalized_resources]

    def _invalidate_resource_cache(self) -> None:
        """Invalidate the short-lived bridge snapshot."""
        self._resource_cache = None

    def _build_indexes(self, resources: list[dict[str, object]]) -> _HueIndexes:
        """Build helper indexes from one resource snapshot."""
        device_names = self._device_name_index(resources)
        service_to_device = self._service_to_device_index(resources)
        scene_groups = self._scene_group_index(resources)
        group_names, area_names = self._area_name_index(resources)
        device_online = self._device_online_index(resources, service_to_device=service_to_device)
        group_online = self._group_online_index(
            resources,
            area_names=area_names,
            scene_groups=scene_groups,
            service_to_device=service_to_device,
            device_online=device_online,
        )
        return _HueIndexes(
            device_names=device_names,
            area_names=area_names,
            group_names=group_names,
            scene_groups=scene_groups,
            device_online=device_online,
            group_online=group_online,
        )

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

    def _service_to_device_index(self, resources: list[dict[str, object]]) -> dict[str, str]:
        """Map service resource ids back to their parent Hue device ids."""
        service_to_device: dict[str, str] = {}
        for resource in resources:
            if str(resource.get("type", "")).strip() != "device":
                continue
            device_id = _first_non_empty_text(resource.get("id"))
            if device_id is None:
                continue
            for service in _as_list(resource.get("services")):
                if isinstance(service, Mapping):
                    service_id = _first_non_empty_text(service.get("rid"))
                    if service_id is not None:
                        service_to_device[service_id] = device_id
        return service_to_device

    def _scene_group_index(self, resources: list[dict[str, object]]) -> dict[str, str]:
        """Map Hue scene ids onto the room/zone/group they belong to."""
        scene_groups: dict[str, str] = {}
        for resource in resources:
            if str(resource.get("type", "")).strip() != "scene":
                continue
            scene_id = _first_non_empty_text(resource.get("id"))
            group_id = _first_non_empty_text(_mapping_get(resource, "group").get("rid"))
            if scene_id is not None and group_id is not None:
                scene_groups[scene_id] = group_id
        return scene_groups

    def _area_name_index(self, resources: list[dict[str, object]]) -> tuple[dict[str, str], dict[str, str]]:
        """Map room/zone ids, their services, and their children onto area names."""
        group_names: dict[str, str] = {}
        area_names: dict[str, str] = {}
        for resource in resources:
            resource_type = str(resource.get("type", "")).strip()
            if resource_type not in {"room", "zone"}:
                continue
            metadata = _mapping_get(resource, "metadata")
            area_name = _first_non_empty_text(metadata.get("name"))
            group_id = _first_non_empty_text(resource.get("id"))
            if area_name is None or group_id is None:
                continue
            group_names[group_id] = area_name
            area_names[group_id] = area_name

            for service in _as_list(resource.get("services")):
                if isinstance(service, Mapping):
                    service_id = _first_non_empty_text(service.get("rid"))
                    if service_id is not None:
                        area_names[service_id] = area_name

            for child in _as_list(resource.get("children")):
                if isinstance(child, Mapping):
                    child_id = _first_non_empty_text(child.get("rid"))
                    if child_id is not None:
                        area_names[child_id] = area_name
        return group_names, area_names

    def _device_online_index(
        self,
        resources: list[dict[str, object]],
        *,
        service_to_device: Mapping[str, str],
    ) -> dict[str, bool]:
        """Map parent device ids onto a conservative online flag."""
        device_online: dict[str, bool] = {}
        for resource in resources:
            resource_type = str(resource.get("type", "")).strip()
            if resource_type not in _CONNECTIVITY_RESOURCE_TYPES:
                continue
            owner = _mapping_get(resource, "owner")
            owner_id = _first_non_empty_text(owner.get("rid"))
            owner_type = _first_non_empty_text(owner.get("rtype"))
            online = self._connectivity_online(resource)

            if owner_type == "device" and owner_id is not None:
                device_online[owner_id] = device_online.get(owner_id, True) and online
                continue

            if owner_id is not None and owner_id in service_to_device:
                device_id = service_to_device[owner_id]
                device_online[device_id] = device_online.get(device_id, True) and online

        return device_online

    def _group_online_index(
        self,
        resources: list[dict[str, object]],
        *,
        area_names: Mapping[str, str],
        scene_groups: Mapping[str, str],
        service_to_device: Mapping[str, str],
        device_online: Mapping[str, bool],
    ) -> dict[str, bool]:
        """Map room/zone/group resources onto an aggregate availability flag."""
        group_members: dict[str, set[str]] = {}
        for resource in resources:
            owner = _mapping_get(resource, "owner")
            owner_id = _first_non_empty_text(owner.get("rid"))
            owner_type = _first_non_empty_text(owner.get("rtype"))
            entity_id = _first_non_empty_text(resource.get("id"))
            if entity_id is None:
                continue

            if owner_id is not None and owner_type in {"room", "zone"}:
                group_members.setdefault(owner_id, set()).add(entity_id)

            group_id = _first_non_empty_text(_mapping_get(resource, "group").get("rid"))
            if group_id is not None:
                group_members.setdefault(group_id, set()).add(entity_id)

            if entity_id in scene_groups:
                group_members.setdefault(scene_groups[entity_id], set()).add(entity_id)

        group_online: dict[str, bool] = {}
        for group_id in area_names:
            members = group_members.get(group_id, set())
            if not members:
                continue
            member_online_states: list[bool] = []
            for member_id in members:
                device_id = service_to_device.get(member_id)
                if device_id is not None and device_id in device_online:
                    member_online_states.append(device_online[device_id])
            if member_online_states:
                group_online[group_id] = any(member_online_states)
        return group_online

    def _entity_from_resource(
        self,
        resource: Mapping[str, object],
        *,
        indexes: _HueIndexes,
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
        owner_type = _first_non_empty_text(owner.get("rtype"))
        metadata = _mapping_get(resource, "metadata")

        if resource_type == "scene":
            group_id = _first_non_empty_text(_mapping_get(resource, "group").get("rid")) or indexes.scene_groups.get(entity_id)
            owner_label = None if group_id is None else indexes.group_names.get(group_id)
            base_label = _first_non_empty_text(metadata.get("name"))
            label = base_label or owner_label or f"scene {entity_id[:8]}"
            area = "" if group_id is None else indexes.area_names.get(group_id, "")
        else:
            base_label = _first_non_empty_text(metadata.get("name"))
            owner_label = None
            if owner_type == "device" and owner_id is not None:
                owner_label = indexes.device_names.get(owner_id)
            elif owner_id is not None:
                owner_label = indexes.group_names.get(owner_id) or indexes.device_names.get(owner_id)
            label = base_label or owner_label or f"{resource_type.replace('_', ' ')} {entity_id[:8]}"
            area = (
                indexes.area_names.get(entity_id)
                or ("" if owner_id is None else indexes.area_names.get(owner_id, ""))
            )

        supported_commands = _CONTROL_COMMANDS_BY_RESOURCE_TYPE.get(resource_type, ())
        attributes: dict[str, object] = {"resource_type": resource_type}
        if owner_id is not None:
            attributes["owner_id"] = owner_id
        if owner_type is not None:
            attributes["owner_type"] = owner_type
        if resource_type == "scene":
            group_id = _first_non_empty_text(_mapping_get(resource, "group").get("rid")) or indexes.scene_groups.get(entity_id)
            if group_id is not None:
                attributes["group_id"] = group_id

        return SmartHomeEntity(
            entity_id=entity_id,
            provider=self.provider_id,
            label=label,
            entity_class=entity_class,
            area=area,
            readable=True,
            controllable=bool(supported_commands),
            online=self._is_online(resource_type, resource, indexes=indexes),
            supported_commands=supported_commands,
            state=self._state_from_resource(resource_type, resource),
            attributes=attributes,
        )

    def _is_online(
        self,
        resource_type: str,
        resource: Mapping[str, object],
        *,
        indexes: _HueIndexes,
    ) -> bool:
        """Infer a conservative online flag from one Hue resource."""
        entity_id = _first_non_empty_text(resource.get("id"))
        if resource_type in _CONNECTIVITY_RESOURCE_TYPES:
            return self._connectivity_online(resource)

        owner = _mapping_get(resource, "owner")
        owner_id = _first_non_empty_text(owner.get("rid"))
        owner_type = _first_non_empty_text(owner.get("rtype"))
        if owner_type == "device" and owner_id is not None and owner_id in indexes.device_online:
            return indexes.device_online[owner_id]
        if owner_id is not None and owner_id in indexes.group_online:
            return indexes.group_online[owner_id]

        group_id = _first_non_empty_text(_mapping_get(resource, "group").get("rid")) or indexes.scene_groups.get(
            entity_id or ""
        )
        if group_id is not None and group_id in indexes.group_online:
            return indexes.group_online[group_id]

        if entity_id is not None and entity_id in indexes.group_online:
            return indexes.group_online[entity_id]

        return True

    @staticmethod
    def _connectivity_online(resource: Mapping[str, object]) -> bool:
        """Interpret a connectivity resource status."""
        return str(resource.get("status", "")).strip().lower() == "connected"

    @staticmethod
    def _state_from_resource(resource_type: str, resource: Mapping[str, object]) -> dict[str, object]:
        """Extract a compact current-state payload from one Hue resource."""
        if resource_type in {"light", "grouped_light"}:
            color_temperature = _mapping_get(resource, "color_temperature")
            color = _mapping_get(resource, "color")
            dynamics = _mapping_get(resource, "dynamics")
            effects = _mapping_get(resource, "effects")
            effects_v2 = _mapping_get(resource, "effects_v2")
            timed_effects = _mapping_get(resource, "timed_effects")
            return {
                "on": bool(_mapping_get(resource, "on").get("on", False)),
                "brightness": _mapping_get(resource, "dimming").get("brightness"),
                "color_temperature_mirek": color_temperature.get("mirek"),
                "xy_color": _mapping_get(color, "xy") or None,
                "dynamic_status": dynamics.get("status"),
                "effect": effects.get("effect"),
                "effect_v2": effects_v2.get("effect"),
                "effect_speed": effects_v2.get("effect_values"),
                "timed_effect": timed_effects.get("effect"),
            }

        if resource_type == "scene":
            status = _mapping_get(resource, "status")
            return {
                "active": status.get("active"),
                "last_recall": status.get("last_recall"),
                "speed": resource.get("speed"),
                "auto_dynamic": resource.get("auto_dynamic"),
            }

        if resource_type in _MOTION_RESOURCE_TYPES:
            motion_feature = _mapping_get(resource, "motion")
            report = _mapping_get(motion_feature, "motion_report")
            return {
                "motion": report.get("motion", motion_feature.get("motion")),
                "motion_changed": report.get("changed"),
                "enabled": motion_feature.get("enabled"),
            }

        if resource_type in _CONNECTIVITY_RESOURCE_TYPES:
            return {"status": resource.get("status")}

        if resource_type in _BUTTON_RESOURCE_TYPES:
            button_feature = _mapping_get(resource, "button")
            report = _mapping_get(button_feature, "button_report")
            return {
                "event": report.get("event", button_feature.get("last_event")),
                "updated": report.get("updated"),
            }

        if resource_type == "device_power":
            return {
                "battery_level": resource.get("battery_level"),
                "battery_state": resource.get("battery_state"),
            }

        if resource_type == "temperature":
            feature = _mapping_get(resource, "temperature")
            report = _mapping_get(feature, "temperature_report")
            return {
                "temperature": report.get("temperature", feature.get("temperature")),
                "changed": report.get("changed"),
            }

        if resource_type in _LIGHT_LEVEL_RESOURCE_TYPES:
            feature = _mapping_get(resource, "light")
            report = _mapping_get(feature, "light_level_report")
            return {
                "light_level": report.get("light_level", feature.get("light_level")),
                "changed": report.get("changed"),
            }

        if resource_type == "contact":
            report = _mapping_get(resource, "contact_report")
            return {
                "open": report.get("state"),
                "changed": report.get("changed"),
            }

        if resource_type == "tamper":
            reports = [item for item in _as_list(resource.get("tamper_reports")) if isinstance(item, Mapping)]
            latest = reports[-1] if reports else {}
            return {
                "tampered": latest.get("state"),
                "changed": latest.get("changed"),
            }

        return {}

    def _control_payload(
        self,
        resource_type: str,
        *,
        command: SmartHomeCommand,
        parameters: Mapping[str, object],
    ) -> dict[str, object]:
        """Build one Hue resource update payload from a generic command."""
        if self._command_matches(command, "TURN_ON"):
            return self._light_payload(resource_type, parameters=parameters, on=True, default_power_on=True)

        if self._command_matches(command, "TURN_OFF"):
            return self._light_payload(resource_type, parameters=parameters, on=False, default_power_on=False)

        if self._command_matches(command, "SET_BRIGHTNESS"):
            brightness = self._extract_brightness(parameters, allow_zero=True)
            if brightness is None:
                raise ValueError("brightness, brightness_pct, or brightness_255 must be provided.")
            if brightness == 0.0:
                return self._light_payload(resource_type, parameters=parameters, on=False, default_power_on=False)
            return self._light_payload(
                resource_type,
                parameters=parameters,
                on=None,
                default_power_on=True,
                forced_brightness=brightness,
            )

        if self._command_matches(command, "SET_COLOR_TEMPERATURE", "SET_COLOR_TEMP"):
            color_temperature = self._extract_color_temperature_mirek(parameters)
            if color_temperature is None:
                raise ValueError("color_temperature_mirek or color_temperature_kelvin must be provided.")
            return self._light_payload(
                resource_type,
                parameters=parameters,
                on=None,
                default_power_on=True,
                forced_color_temperature_mirek=color_temperature,
            )

        if self._command_matches(command, "SET_COLOR", "SET_XY_COLOR"):
            xy_color = self._extract_xy_color(parameters)
            if xy_color is None:
                raise ValueError("xy_color or color_xy must be provided.")
            return self._light_payload(
                resource_type,
                parameters=parameters,
                on=None,
                default_power_on=True,
                forced_xy_color=xy_color,
            )

        if self._command_matches(command, "SET_EFFECT"):
            effect = self._extract_effect(parameters)
            if effect is None:
                raise ValueError("effect or effect_name must be provided.")
            return self._light_payload(
                resource_type,
                parameters=parameters,
                on=None,
                default_power_on=True,
                forced_effect=effect,
            )

        if self._command_matches(command, "ACTIVATE"):
            return self._scene_payload(resource_type, parameters=parameters)

        raise ValueError(f"Unsupported Hue smart-home command: {getattr(command, 'value', command)!r}")

    def _light_payload(
        self,
        resource_type: str,
        *,
        parameters: Mapping[str, object],
        on: bool | None,
        default_power_on: bool,
        forced_brightness: float | None = None,
        forced_color_temperature_mirek: int | None = None,
        forced_xy_color: tuple[float, float] | None = None,
        forced_effect: str | None = None,
    ) -> dict[str, object]:
        """Build a modern Hue light/grouped_light payload."""
        if resource_type not in {"light", "grouped_light"}:
            raise ValueError(f"This command is not supported for Hue resource type {resource_type}.")

        payload: dict[str, object] = {}
        if on is not None:
            payload["on"] = {"on": on}

        transition_ms = self._extract_transition_ms(parameters)
        brightness = forced_brightness if forced_brightness is not None else self._extract_brightness(parameters, allow_zero=True)
        color_temperature_mirek = (
            forced_color_temperature_mirek
            if forced_color_temperature_mirek is not None
            else self._extract_color_temperature_mirek(parameters)
        )
        xy_color = forced_xy_color if forced_xy_color is not None else self._extract_xy_color(parameters)
        effect = forced_effect if forced_effect is not None else self._extract_effect(parameters)
        power_on = self._extract_power_on(parameters, default=default_power_on)

        if brightness is not None:
            if brightness == 0.0:
                payload["on"] = {"on": False}
            else:
                payload["dimming"] = {"brightness": brightness}
                if power_on and "on" not in payload:
                    payload["on"] = {"on": True}

        if color_temperature_mirek is not None:
            payload["color_temperature"] = {"mirek": color_temperature_mirek}
            if power_on and "on" not in payload:
                payload["on"] = {"on": True}

        if xy_color is not None:
            payload["color"] = {"xy": {"x": xy_color[0], "y": xy_color[1]}}
            if power_on and "on" not in payload:
                payload["on"] = {"on": True}

        if transition_ms is not None:
            payload.setdefault("dynamics", {})
            if isinstance(payload["dynamics"], dict):
                payload["dynamics"]["duration"] = transition_ms

        if effect is not None:
            payload.update(
                self._effect_payload(
                    effect=effect,
                    transition_ms=transition_ms,
                    color_temperature_mirek=color_temperature_mirek,
                    xy_color=xy_color,
                    parameters=parameters,
                )
            )
            if power_on and "on" not in payload and effect not in {"off", "no_effect"}:
                payload["on"] = {"on": True}

        if not power_on and ("color" in payload or "color_temperature" in payload or "dimming" in payload):
            payload.setdefault("on", {"on": False})

        return payload

    def _scene_payload(
        self,
        resource_type: str,
        *,
        parameters: Mapping[str, object],
    ) -> dict[str, object]:
        """Build a modern Hue scene recall payload."""
        if resource_type != "scene":
            raise ValueError("activate is only supported for Hue scenes.")

        action = self._extract_scene_action(parameters)
        payload: dict[str, object] = {"recall": {"action": action}}

        transition_ms = self._extract_transition_ms(parameters)
        if transition_ms is not None:
            payload["recall"]["duration"] = transition_ms  # type: ignore[index]

        brightness = self._extract_brightness(parameters, allow_zero=False)
        if brightness is not None:
            payload["recall"]["dimming"] = {"brightness": brightness}  # type: ignore[index]

        speed = self._extract_speed(parameters)
        if speed is not None:
            payload["speed"] = speed

        auto_dynamic = _coerce_bool(parameters.get("auto_dynamic"))
        if auto_dynamic is not None:
            payload["auto_dynamic"] = auto_dynamic

        return payload

    @staticmethod
    def _command_matches(command: SmartHomeCommand, *candidate_names: str) -> bool:
        """Return whether one command matches any known alias."""
        normalized_candidates = {candidate.strip().lower() for candidate in candidate_names}
        command_name = str(getattr(command, "name", "")).strip().lower()
        command_value = str(getattr(command, "value", "")).strip().lower()
        return command_name in normalized_candidates or command_value in normalized_candidates

    @staticmethod
    def _extract_power_on(parameters: Mapping[str, object], *, default: bool) -> bool:
        """Return the requested power-on behavior for state-setting commands."""
        if "power_on" not in parameters:
            return default
        parsed = _coerce_bool(parameters.get("power_on"))
        if parsed is None:
            raise ValueError("power_on must be a boolean-like value.")
        return parsed

    @staticmethod
    def _extract_brightness(parameters: Mapping[str, object], *, allow_zero: bool) -> float | None:
        """Extract a Hue brightness value in the native 0..100 range."""
        for key in ("brightness", "brightness_pct"):
            if key in parameters:
                value = _coerce_float(parameters.get(key))
                if value is None:
                    raise ValueError("brightness must be a number.")
                if value == 0.0 and allow_zero:
                    return 0.0
                if value <= 0.0 or value > 100.0:
                    raise ValueError("brightness must be between 0 and 100.")
                return value

        if "brightness_255" in parameters:
            value = _coerce_float(parameters.get("brightness_255"))
            if value is None:
                raise ValueError("brightness_255 must be a number.")
            if value == 0.0 and allow_zero:
                return 0.0
            if value <= 0.0 or value > 255.0:
                raise ValueError("brightness_255 must be between 0 and 255.")
            return (value / 255.0) * 100.0

        return None

    @staticmethod
    def _extract_transition_ms(parameters: Mapping[str, object]) -> int | None:
        """Extract a Hue transition duration in milliseconds rounded to 100 ms."""
        for key in ("transition_ms", "duration_ms"):
            if key in parameters:
                value = _coerce_float(parameters.get(key))
                if value is None:
                    raise ValueError(f"{key} must be a number.")
                if value < 0.0 or value > 6_000_000.0:
                    raise ValueError(f"{key} must be between 0 and 6000000 milliseconds.")
                return int(round(value / 100.0) * 100)

        for key in ("transition_s", "duration_s", "transition", "duration"):
            if key in parameters:
                value = _coerce_float(parameters.get(key))
                if value is None:
                    raise ValueError(f"{key} must be a number.")
                if value < 0.0 or value > 6000.0:
                    raise ValueError(f"{key} must be between 0 and 6000 seconds.")
                return int(round(value, 1) * 1000)

        return None

    @staticmethod
    def _extract_color_temperature_mirek(parameters: Mapping[str, object]) -> int | None:
        """Extract a Hue mirek value and clamp it to the common Hue range."""
        for key in ("color_temperature_mirek", "mirek"):
            if key in parameters:
                value = _coerce_float(parameters.get(key))
                if value is None:
                    raise ValueError(f"{key} must be a number.")
                return max(153, min(500, int(round(value))))

        for key in ("color_temperature_kelvin", "color_temp_kelvin", "kelvin"):
            if key in parameters:
                value = _coerce_float(parameters.get(key))
                if value is None:
                    raise ValueError(f"{key} must be a number.")
                if value <= 0.0:
                    raise ValueError(f"{key} must be greater than 0.")
                mirek = int(round(1_000_000.0 / value))
                return max(153, min(500, mirek))

        return None

    @staticmethod
    def _extract_xy_color(parameters: Mapping[str, object]) -> tuple[float, float] | None:
        """Extract an XY color from tuple/list or mapping input."""
        raw_value = parameters.get("xy_color", parameters.get("color_xy"))
        if raw_value is None:
            return None

        if isinstance(raw_value, Mapping):
            x = _coerce_float(raw_value.get("x"))
            y = _coerce_float(raw_value.get("y"))
        elif isinstance(raw_value, (list, tuple)) and len(raw_value) == 2:
            x = _coerce_float(raw_value[0])
            y = _coerce_float(raw_value[1])
        else:
            raise ValueError("xy_color must be a 2-item sequence or {'x': ..., 'y': ...} mapping.")

        if x is None or y is None:
            raise ValueError("xy_color values must be numeric.")
        if not (-1.0 <= x <= 1.0 and -1.0 <= y <= 1.0):
            raise ValueError("xy_color values must be between -1 and 1.")
        return (x, y)

    @staticmethod
    def _extract_effect(parameters: Mapping[str, object]) -> str | None:
        """Extract one requested effect name."""
        effect = _first_non_empty_text(parameters.get("effect"), parameters.get("effect_name"))
        return None if effect is None else effect.strip().lower()

    @staticmethod
    def _extract_speed(parameters: Mapping[str, object]) -> float | None:
        """Extract a Hue speed value in the native 0..1 range."""
        if "speed" not in parameters:
            return None
        value = _coerce_float(parameters.get("speed"))
        if value is None:
            raise ValueError("speed must be a number.")
        if 0.0 <= value <= 1.0:
            return value
        if 0.0 <= value <= 100.0:
            return value / 100.0
        raise ValueError("speed must be between 0..1 or 0..100.")

    @staticmethod
    def _extract_scene_action(parameters: Mapping[str, object]) -> str:
        """Extract a Hue scene recall action."""
        action = _first_non_empty_text(parameters.get("scene_action"), parameters.get("action"))
        if action is None:
            return "dynamic_palette" if _coerce_bool(parameters.get("dynamic")) is True else "active"
        normalized = action.strip().lower()
        if normalized not in {"active", "static", "dynamic_palette"}:
            raise ValueError("scene_action must be one of: active, static, dynamic_palette.")
        return normalized

    def _effect_payload(
        self,
        *,
        effect: str,
        transition_ms: int | None,
        color_temperature_mirek: int | None,
        xy_color: tuple[float, float] | None,
        parameters: Mapping[str, object],
    ) -> dict[str, object]:
        """Build Hue effect payloads using effects_v2 where it adds value."""
        if effect in {"off", "no_effect"}:
            return {"effects": {"effect": "no_effect"}}

        if effect in {"sunrise", "sunset"}:
            duration_ms = transition_ms if transition_ms is not None else 600_000
            return {"timed_effects": {"effect": effect, "duration": duration_ms}}

        effect_speed = None
        if "effect_speed" in parameters:
            value = _coerce_float(parameters.get("effect_speed"))
            if value is None:
                raise ValueError("effect_speed must be a number.")
            if 0.0 <= value <= 1.0:
                effect_speed = value
            elif 0.0 <= value <= 100.0:
                effect_speed = value / 100.0
            else:
                raise ValueError("effect_speed must be between 0..1 or 0..100.")

        if effect_speed is not None or color_temperature_mirek is not None or xy_color is not None:
            action: dict[str, object] = {"effect": effect, "parameters": {}}
            parameters_payload = action["parameters"]
            if isinstance(parameters_payload, dict):
                if xy_color is not None:
                    parameters_payload["color"] = {"xy": {"x": xy_color[0], "y": xy_color[1]}}
                if color_temperature_mirek is not None:
                    parameters_payload["color_temperature"] = {"mirek": color_temperature_mirek}
                if effect_speed is not None:
                    parameters_payload["speed"] = effect_speed
            return {"effects_v2": {"action": action}}

        return {"effects": {"effect": effect}}

    def _iter_stream_events(self, raw_message: object) -> tuple[Mapping[str, object], ...]:
        """Yield one or more Hue event objects from one raw client message."""
        if isinstance(raw_message, Mapping):
            return (raw_message,)
        if isinstance(raw_message, list):
            return tuple(item for item in raw_message if isinstance(item, Mapping))
        return ()

    def _normalize_stream_event(
        self,
        event: Mapping[str, object],
        *,
        event_creationtime: str | None,
    ) -> list[SmartHomeEvent]:
        """Convert one raw Hue SSE payload into normalized smart-home events."""
        raw_data = event.get("data", ())
        if not isinstance(raw_data, list):
            return []

        event_id_prefix = _first_non_empty_text(event.get("id")) or "hue"
        collected: list[SmartHomeEvent] = []
        for resource_index, entry in enumerate(raw_data):
            if not isinstance(entry, Mapping):
                continue

            if self._looks_like_stream_event(entry):
                nested_creationtime = _first_non_empty_text(entry.get("creationtime")) or event_creationtime
                collected.extend(self._normalize_stream_event(entry, event_creationtime=nested_creationtime))
                continue

            normalized = self._event_from_changed_resource(
                entry,
                event_id=f"{event_id_prefix}:{resource_index}",
                event_creationtime=event_creationtime,
            )
            if normalized is not None:
                collected.append(normalized)

        return collected

    @staticmethod
    def _looks_like_stream_event(value: Mapping[str, object]) -> bool:
        """Detect a nested Hue stream event envelope."""
        event_type = _first_non_empty_text(value.get("type"))
        return event_type in _STREAM_EVENT_TYPES and isinstance(value.get("data"), list)

    def _event_from_changed_resource(
        self,
        resource: Mapping[str, object],
        *,
        event_id: str,
        event_creationtime: str | None,
    ) -> SmartHomeEvent | None:
        """Convert one changed Hue resource into a generic smart-home event."""
        resource_type = str(resource.get("type", "")).strip()
        entity_id = _first_non_empty_text(resource.get("id"))
        if entity_id is None:
            return None

        observed_at = self._observed_at_for_event(resource_type, resource) or event_creationtime or _utc_now_iso()

        if resource_type in {"motion", "camera_motion", "convenience_area_motion", "security_area_motion", "grouped_motion"}:
            motion_value = self._state_from_resource(resource_type, resource).get("motion")
            if motion_value is True:
                event_kind = SmartHomeEventKind.MOTION_DETECTED
            elif motion_value is False:
                event_kind = SmartHomeEventKind.MOTION_CLEARED
            else:
                event_kind = SmartHomeEventKind.STATE_CHANGED
        elif resource_type in _CONNECTIVITY_RESOURCE_TYPES:
            event_kind = (
                SmartHomeEventKind.DEVICE_ONLINE
                if self._connectivity_online(resource)
                else SmartHomeEventKind.DEVICE_OFFLINE
            )
        elif resource_type in {"button", "bell_button"}:
            button_state = self._state_from_resource(resource_type, resource).get("event")
            event_kind = SmartHomeEventKind.BUTTON_PRESSED if button_state is not None else SmartHomeEventKind.STATE_CHANGED
        else:
            event_kind = SmartHomeEventKind.STATE_CHANGED

        return SmartHomeEvent(
            event_id=event_id,
            provider=self.provider_id,
            entity_id=entity_id,
            event_kind=event_kind,
            observed_at=observed_at,
            details=self._event_details(resource_type, resource),
        )

    @staticmethod
    def _observed_at_for_event(resource_type: str, resource: Mapping[str, object]) -> str | None:
        """Extract the most specific timestamp available for one changed resource."""
        if resource_type in {"motion", "camera_motion", "convenience_area_motion", "security_area_motion", "grouped_motion"}:
            motion = _mapping_get(resource, "motion")
            report = _mapping_get(motion, "motion_report")
            return _first_non_empty_text(report.get("changed"))

        if resource_type in {"button", "bell_button"}:
            button = _mapping_get(resource, "button")
            report = _mapping_get(button, "button_report")
            return _first_non_empty_text(report.get("updated"))

        if resource_type == "temperature":
            feature = _mapping_get(resource, "temperature")
            report = _mapping_get(feature, "temperature_report")
            return _first_non_empty_text(report.get("changed"))

        if resource_type in {"light_level", "grouped_light_level"}:
            feature = _mapping_get(resource, "light")
            report = _mapping_get(feature, "light_level_report")
            return _first_non_empty_text(report.get("changed"))

        if resource_type == "contact":
            report = _mapping_get(resource, "contact_report")
            return _first_non_empty_text(report.get("changed"))

        if resource_type == "tamper":
            reports = [item for item in _as_list(resource.get("tamper_reports")) if isinstance(item, Mapping)]
            if reports:
                return _first_non_empty_text(reports[-1].get("changed"))

        return _first_non_empty_text(resource.get("changed"))

    def _event_details(self, resource_type: str, resource: Mapping[str, object]) -> dict[str, object]:
        """Build bounded event details for downstream consumers."""
        details: dict[str, object] = {
            "resource_type": resource_type,
            "state": self._state_from_resource(resource_type, resource),
        }
        owner = _mapping_get(resource, "owner")
        owner_id = _first_non_empty_text(owner.get("rid"))
        owner_type = _first_non_empty_text(owner.get("rtype"))
        if owner_id is not None:
            details["owner_id"] = owner_id
        if owner_type is not None:
            details["owner_type"] = owner_type
        group_id = _first_non_empty_text(_mapping_get(resource, "group").get("rid"))
        if group_id is not None:
            details["group_id"] = group_id
        # BREAKING: details["raw"] is now size-bounded and depth-limited for safety; consumers must not rely on full bridge payload passthrough.
        details["raw"] = self._bounded_value(resource, depth=self.max_event_detail_depth)
        return details

    def _bounded_value(self, value: object, *, depth: int) -> object:
        """Return a bounded copy of one raw payload value."""
        if depth <= 0:
            return "…"

        if isinstance(value, Mapping):
            bounded: dict[str, object] = {}
            items = list(value.items())
            for key, inner_value in items[: self.max_event_detail_keys]:
                bounded[str(key)] = self._bounded_value(inner_value, depth=depth - 1)
            if len(items) > self.max_event_detail_keys:
                bounded["__truncated_keys__"] = len(items) - self.max_event_detail_keys
            return bounded

        if isinstance(value, list):
            bounded_list = [
                self._bounded_value(item, depth=depth - 1)
                for item in value[: self.max_event_detail_items]
            ]
            if len(value) > self.max_event_detail_items:
                bounded_list.append(f"…+{len(value) - self.max_event_detail_items} items")
            return bounded_list

        if isinstance(value, tuple):
            return tuple(self._bounded_value(item, depth=depth - 1) for item in value[: self.max_event_detail_items])

        if isinstance(value, str):
            if len(value) <= self.max_event_text_length:
                return value
            return value[: self.max_event_text_length] + "…"

        if isinstance(value, (int, float, bool)) or value is None:
            return value

        return repr(value)[: self.max_event_text_length]


def build_hue_smart_home_provider(client: HueBridgeClient) -> HueSmartHomeProvider:
    """Build the generic Hue provider from one bridge client."""
    return HueSmartHomeProvider(client=client)


__all__ = ["HueSmartHomeProvider", "build_hue_smart_home_provider"]
