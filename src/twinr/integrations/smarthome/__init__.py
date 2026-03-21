"""Expose generic smart-home integration contracts and helpers."""

from twinr.integrations.smarthome.adapter import (
    SmartHomeAdapterSettings,
    SmartHomeController,
    SmartHomeEntityProvider,
    SmartHomeIntegrationAdapter,
    SmartHomeSensorStream,
)
from twinr.integrations.smarthome.aggregate import (
    AggregatedSmartHomeProvider,
    RoutedSmartHomeProvider,
    build_routed_entity_id,
    build_routed_event_id,
    parse_routed_entity_id,
)
from twinr.integrations.smarthome.models import (
    SmartHomeCommand,
    SmartHomeEntity,
    SmartHomeEntityAggregateField,
    SmartHomeEntityClass,
    SmartHomeEvent,
    SmartHomeEventBatch,
    SmartHomeEventAggregateField,
    SmartHomeEventKind,
)
from twinr.integrations.smarthome.runtime import (
    SmartHomeObservation,
    SmartHomeObservationBuilder,
    SmartHomeSensorWorker,
)

__all__ = [
    "AggregatedSmartHomeProvider",
    "RoutedSmartHomeProvider",
    "SmartHomeAdapterSettings",
    "SmartHomeCommand",
    "SmartHomeController",
    "SmartHomeEntity",
    "SmartHomeEntityAggregateField",
    "SmartHomeEntityClass",
    "SmartHomeEntityProvider",
    "SmartHomeEvent",
    "SmartHomeEventBatch",
    "SmartHomeEventAggregateField",
    "SmartHomeEventKind",
    "SmartHomeIntegrationAdapter",
    "SmartHomeObservation",
    "SmartHomeObservationBuilder",
    "SmartHomeSensorWorker",
    "SmartHomeSensorStream",
    "build_routed_entity_id",
    "build_routed_event_id",
    "parse_routed_entity_id",
]
