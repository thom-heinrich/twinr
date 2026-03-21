"""Expose generic smart-home integration contracts and helpers."""

from twinr.integrations.smarthome.adapter import (
    SmartHomeAdapterSettings,
    SmartHomeController,
    SmartHomeEntityProvider,
    SmartHomeIntegrationAdapter,
    SmartHomeSensorStream,
)
from twinr.integrations.smarthome.models import (
    SmartHomeCommand,
    SmartHomeEntity,
    SmartHomeEntityClass,
    SmartHomeEvent,
    SmartHomeEventBatch,
    SmartHomeEventKind,
)

__all__ = [
    "SmartHomeAdapterSettings",
    "SmartHomeCommand",
    "SmartHomeController",
    "SmartHomeEntity",
    "SmartHomeEntityClass",
    "SmartHomeEntityProvider",
    "SmartHomeEvent",
    "SmartHomeEventBatch",
    "SmartHomeEventKind",
    "SmartHomeIntegrationAdapter",
    "SmartHomeSensorStream",
]
