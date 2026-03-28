"""Dynamic schema-build context captured once per schema build."""

from __future__ import annotations

from dataclasses import dataclass

from twinr.agent.base_agent.settings.simple_settings import (
    spoken_voice_options_context,
    supported_setting_names,
    supported_spoken_voices,
)
from twinr.automations import supported_sensor_trigger_kinds
from twinr.integrations.smarthome.models import (
    SmartHomeCommand,
    SmartHomeEntityAggregateField,
    SmartHomeEntityClass,
    SmartHomeEventAggregateField,
    SmartHomeEventKind,
)

from .shared import unique_strings


@dataclass(frozen=True)
class SchemaBuildContext:
    sensor_trigger_kinds: tuple[str, ...]
    sensor_trigger_kind_help: str
    setting_names: tuple[str, ...]
    spoken_voices: tuple[str, ...]
    spoken_voice_catalog: str
    smart_home_entity_classes: tuple[str, ...]
    smart_home_commands: tuple[str, ...]
    smart_home_entity_aggregate_fields: tuple[str, ...]
    smart_home_event_aggregate_fields: tuple[str, ...]
    smart_home_event_kinds: tuple[str, ...]


def build_schema_context() -> SchemaBuildContext:
    return SchemaBuildContext(
        sensor_trigger_kinds=tuple(unique_strings(supported_sensor_trigger_kinds())),
        sensor_trigger_kind_help=(
            "Supported sensor trigger type. "
            "Use pir_no_motion when no motion or no presence/activity should be detected for hold_seconds. "
            "Use vad_quiet only when the room microphone should stay quiet for hold_seconds. "
            "Use camera_person_visible when a person should be visible in the camera view."
        ),
        setting_names=tuple(unique_strings(supported_setting_names())),
        spoken_voices=tuple(unique_strings(supported_spoken_voices())),
        spoken_voice_catalog=str(spoken_voice_options_context()).strip(),
        smart_home_entity_classes=tuple(unique_strings(item.value for item in SmartHomeEntityClass)),
        smart_home_commands=tuple(unique_strings(item.value for item in SmartHomeCommand)),
        smart_home_entity_aggregate_fields=tuple(
            unique_strings(item.value for item in SmartHomeEntityAggregateField)
        ),
        smart_home_event_aggregate_fields=tuple(
            unique_strings(item.value for item in SmartHomeEventAggregateField)
        ),
        smart_home_event_kinds=tuple(unique_strings(item.value for item in SmartHomeEventKind)),
    )
