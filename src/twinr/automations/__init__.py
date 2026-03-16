"""Expose Twinr automation models, storage, and sensor-trigger helpers."""

from twinr.automations.sensors import (
    SensorTriggerSpec,
    build_sensor_trigger,
    describe_sensor_trigger,
    describe_sensor_trigger_text,
    supported_sensor_trigger_kinds,
)
from twinr.automations.store import (
    AutomationAction,
    AutomationCondition,
    AutomationDefinition,
    AutomationEngine,
    AutomationStore,
    IfThenAutomationTrigger,
    TimeAutomationTrigger,
)

__all__ = [
    "AutomationAction",
    "AutomationCondition",
    "AutomationDefinition",
    "AutomationEngine",
    "AutomationStore",
    "IfThenAutomationTrigger",
    "SensorTriggerSpec",
    "TimeAutomationTrigger",
    "build_sensor_trigger",
    "describe_sensor_trigger",
    "describe_sensor_trigger_text",
    "supported_sensor_trigger_kinds",
]
