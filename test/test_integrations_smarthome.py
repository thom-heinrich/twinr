from pathlib import Path
import sys
import json
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.integrations import (
    AggregatedSmartHomeProvider,
    IntegrationRequest,
    RoutedSmartHomeProvider,
    SmartHomeCommand,
    SmartHomeEntity,
    SmartHomeEntityClass,
    SmartHomeEvent,
    SmartHomeEventBatch,
    SmartHomeEventKind,
    SmartHomeIntegrationAdapter,
    build_routed_entity_id,
    manifest_for_id,
)


class _FakeEntityProvider:
    def __init__(self, entities):
        self.entities = list(entities)

    def list_entities(self, *, entity_ids=(), entity_class=None, include_unavailable=False):
        requested = set(entity_ids)
        result = []
        for entity in self.entities:
            if requested and entity.entity_id not in requested:
                continue
            if entity_class is not None and entity.entity_class is not entity_class:
                continue
            if not include_unavailable and not entity.online:
                continue
            result.append(entity)
        return result


class _FakeController:
    def __init__(self):
        self.calls = []

    def control(self, *, command, entity_ids, parameters):
        self.calls.append(
            {
                "command": command,
                "entity_ids": entity_ids,
                "parameters": dict(parameters),
            }
        )
        return {
            "applied": True,
            "targets": [{"entity_id": entity_id} for entity_id in entity_ids],
        }


class _FakeSensorStream:
    def read_sensor_stream(self, *, cursor=None, limit):
        del limit
        return SmartHomeEventBatch(
            events=(
                SmartHomeEvent(
                    event_id="evt-1",
                    provider="fake",
                    entity_id="motion-1",
                    event_kind=SmartHomeEventKind.MOTION_DETECTED,
                    observed_at="2026-03-21T10:00:00Z",
                    details={"source_cursor": cursor},
                ),
            ),
            next_cursor="cursor-2",
        )


class _StaticSensorStream:
    def __init__(self, batch):
        self.batch = batch
        self.calls = []
        self._returned_initial_batch = False

    def read_sensor_stream(self, *, cursor=None, limit):
        self.calls.append({"cursor": cursor, "limit": limit})
        if self._returned_initial_batch:
            return SmartHomeEventBatch(events=(), next_cursor=None, stream_live=self.batch.stream_live)
        self._returned_initial_batch = True
        return self.batch


class SmartHomeIntegrationAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        manifest = manifest_for_id("smart_home_hub")
        assert manifest is not None
        self.manifest = manifest

    def test_list_entities_returns_bounded_entity_payloads(self) -> None:
        provider = _FakeEntityProvider(
            (
                SmartHomeEntity(
                    entity_id="light-1",
                    provider="fake",
                    label="Wohnzimmerlicht",
                    entity_class=SmartHomeEntityClass.LIGHT,
                    controllable=True,
                    supported_commands=(SmartHomeCommand.TURN_ON, SmartHomeCommand.TURN_OFF),
                    state={"on": True},
                ),
            )
        )
        adapter = SmartHomeIntegrationAdapter(manifest=self.manifest, entity_provider=provider)

        result = adapter.execute(
            IntegrationRequest(integration_id="smart_home_hub", operation_id="list_entities")
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.details["count"], 1)
        self.assertEqual(result.details["entities"][0]["label"], "Wohnzimmerlicht")

    def test_list_entities_supports_generic_filters_aggregations_and_pagination(self) -> None:
        provider = _FakeEntityProvider(
            (
                SmartHomeEntity(
                    entity_id="light-1",
                    provider="hue",
                    label="Wohnzimmerlicht",
                    area="Wohnzimmer",
                    entity_class=SmartHomeEntityClass.LIGHT,
                    controllable=True,
                    supported_commands=(SmartHomeCommand.TURN_ON, SmartHomeCommand.TURN_OFF),
                    state={"on": True},
                ),
                SmartHomeEntity(
                    entity_id="light-2",
                    provider="hue",
                    label="Kuechenlicht",
                    area="Kueche",
                    entity_class=SmartHomeEntityClass.LIGHT,
                    controllable=True,
                    online=False,
                    supported_commands=(SmartHomeCommand.TURN_ON, SmartHomeCommand.TURN_OFF),
                    state={"on": True},
                ),
                SmartHomeEntity(
                    entity_id="motion-1",
                    provider="hue",
                    label="Flur Bewegung",
                    area="Flur",
                    entity_class=SmartHomeEntityClass.MOTION_SENSOR,
                    state={"motion": True},
                ),
            )
        )
        adapter = SmartHomeIntegrationAdapter(manifest=self.manifest, entity_provider=provider)

        result = adapter.execute(
            IntegrationRequest(
                integration_id="smart_home_hub",
                operation_id="list_entities",
                parameters={
                    "entity_classes": ["light"],
                    "providers": ["hue"],
                    "state_filters": [{"key": "on", "value": True}],
                    "include_unavailable": True,
                    "aggregate_by": ["area", "online"],
                    "limit": 1,
                },
            )
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.details["count"], 1)
        self.assertEqual(result.details["total_count"], 2)
        self.assertEqual(result.details["returned_count"], 1)
        self.assertEqual(result.details["next_cursor"], "1")
        self.assertTrue(result.details["truncated"])
        self.assertEqual(result.details["entities"][0]["entity_class"], "light")
        self.assertEqual(
            result.details["applied_filters"]["state_filters"],
            [{"key": "on", "value": True}],
        )
        self.assertIn(
            {"field": "area", "value": "Kueche", "count": 1},
            result.details["aggregates"],
        )
        self.assertIn(
            {"field": "online", "value": False, "count": 1},
            result.details["aggregates"],
        )

    def test_control_entities_dispatches_to_controller(self) -> None:
        provider = _FakeEntityProvider(
            (
                SmartHomeEntity(
                    entity_id="light-1",
                    provider="fake",
                    label="Wohnzimmerlicht",
                    entity_class=SmartHomeEntityClass.LIGHT,
                    controllable=True,
                    supported_commands=(
                        SmartHomeCommand.TURN_ON,
                        SmartHomeCommand.TURN_OFF,
                        SmartHomeCommand.SET_BRIGHTNESS,
                    ),
                ),
            )
        )
        controller = _FakeController()
        adapter = SmartHomeIntegrationAdapter(
            manifest=self.manifest,
            entity_provider=provider,
            controller=controller,
        )

        result = adapter.execute(
            IntegrationRequest(
                integration_id="smart_home_hub",
                operation_id="control_entities",
                parameters={
                    "command": "set_brightness",
                    "entity_ids": ["light-1"],
                    "brightness": 55,
                },
                explicit_user_confirmation=True,
            )
        )

        self.assertTrue(result.ok)
        self.assertEqual(len(controller.calls), 1)
        self.assertEqual(controller.calls[0]["command"], SmartHomeCommand.SET_BRIGHTNESS)
        self.assertEqual(controller.calls[0]["entity_ids"], ("light-1",))
        self.assertEqual(controller.calls[0]["parameters"]["brightness"], 55)

    def test_control_entities_rejects_unsafe_targets(self) -> None:
        provider = _FakeEntityProvider(
            (
                SmartHomeEntity(
                    entity_id="motion-1",
                    provider="fake",
                    label="Flur Bewegung",
                    entity_class=SmartHomeEntityClass.MOTION_SENSOR,
                    controllable=False,
                ),
            )
        )
        adapter = SmartHomeIntegrationAdapter(
            manifest=self.manifest,
            entity_provider=provider,
            controller=_FakeController(),
        )

        result = adapter.execute(
            IntegrationRequest(
                integration_id="smart_home_hub",
                operation_id="control_entities",
                parameters={"command": "turn_on", "entity_ids": ["motion-1"]},
            )
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.details["error_code"], "invalid_request")

    def test_run_safe_scene_routes_through_generic_control(self) -> None:
        provider = _FakeEntityProvider(
            (
                SmartHomeEntity(
                    entity_id="scene-1",
                    provider="fake",
                    label="Abend",
                    entity_class=SmartHomeEntityClass.SCENE,
                    controllable=True,
                    supported_commands=(SmartHomeCommand.ACTIVATE,),
                ),
            )
        )
        controller = _FakeController()
        adapter = SmartHomeIntegrationAdapter(
            manifest=self.manifest,
            entity_provider=provider,
            controller=controller,
        )

        result = adapter.execute(
            IntegrationRequest(
                integration_id="smart_home_hub",
                operation_id="run_safe_scene",
                parameters={"scene_id": "scene-1"},
                explicit_user_confirmation=True,
            )
        )

        self.assertTrue(result.ok)
        self.assertEqual(controller.calls[0]["command"], SmartHomeCommand.ACTIVATE)
        self.assertEqual(controller.calls[0]["entity_ids"], ("scene-1",))

    def test_read_sensor_stream_returns_normalized_batch(self) -> None:
        adapter = SmartHomeIntegrationAdapter(
            manifest=self.manifest,
            entity_provider=_FakeEntityProvider(()),
            sensor_stream=_FakeSensorStream(),
        )

        result = adapter.execute(
            IntegrationRequest(
                integration_id="smart_home_hub",
                operation_id="read_sensor_stream",
                parameters={"cursor": "cursor-1", "limit": 5},
                background_trigger=True,
            )
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.details["next_cursor"], "cursor-2")
        self.assertEqual(result.details["events"][0]["event_kind"], "motion_detected")

    def test_read_sensor_stream_supports_generic_filters_and_aggregations(self) -> None:
        stream = _StaticSensorStream(
            SmartHomeEventBatch(
                events=(
                    SmartHomeEvent(
                        event_id="evt-1",
                        provider="hue",
                        entity_id="motion-1",
                        event_kind=SmartHomeEventKind.MOTION_DETECTED,
                        observed_at="2026-03-21T10:00:00Z",
                        area="Flur",
                    ),
                    SmartHomeEvent(
                        event_id="evt-2",
                        provider="hue",
                        entity_id="conn-1",
                        event_kind=SmartHomeEventKind.DEVICE_OFFLINE,
                        observed_at="2026-03-21T10:00:01Z",
                        area="Keller",
                    ),
                    SmartHomeEvent(
                        event_id="evt-3",
                        provider="matter",
                        entity_id="motion-2",
                        event_kind=SmartHomeEventKind.MOTION_CLEARED,
                        observed_at="2026-03-21T10:00:02Z",
                        area="Flur",
                    ),
                ),
                next_cursor="cursor-next",
            )
        )
        adapter = SmartHomeIntegrationAdapter(
            manifest=self.manifest,
            entity_provider=_FakeEntityProvider(()),
            sensor_stream=stream,
        )

        result = adapter.execute(
            IntegrationRequest(
                integration_id="smart_home_hub",
                operation_id="read_sensor_stream",
                parameters={
                    "event_kinds": ["motion_detected", "motion_cleared"],
                    "areas": ["Flur"],
                    "aggregate_by": ["event_kind", "provider"],
                    "limit": 1,
                },
            )
        )

        self.assertTrue(result.ok)
        self.assertEqual(stream.calls[0]["limit"], adapter.settings.max_event_results)
        self.assertEqual(result.details["count"], 1)
        self.assertEqual(result.details["matched_count"], 2)
        self.assertEqual(result.details["events"][0]["entity_id"], "motion-1")
        self.assertEqual(
            result.details["applied_filters"]["event_kinds"],
            ["motion_detected", "motion_cleared"],
        )
        self.assertIn(
            {"field": "provider", "value": "hue", "count": 1},
            result.details["aggregates"],
        )
        self.assertIn(
            {"field": "provider", "value": "matter", "count": 1},
            result.details["aggregates"],
        )

    def test_aggregated_provider_routes_entity_ids_and_controls(self) -> None:
        left_controller = _FakeController()
        right_controller = _FakeController()
        aggregate = AggregatedSmartHomeProvider(
            (
                RoutedSmartHomeProvider(
                    route_id="bridge-a",
                    entity_provider=_FakeEntityProvider(
                        (
                            SmartHomeEntity(
                                entity_id="light-1",
                                provider="hue",
                                label="Deckenlicht",
                                area="Wohnzimmer",
                                entity_class=SmartHomeEntityClass.LIGHT,
                                controllable=True,
                                supported_commands=(SmartHomeCommand.TURN_ON,),
                            ),
                        )
                    ),
                    controller=left_controller,
                ),
                RoutedSmartHomeProvider(
                    route_id="bridge-b",
                    entity_provider=_FakeEntityProvider(
                        (
                            SmartHomeEntity(
                                entity_id="light-2",
                                provider="hue",
                                label="Deckenlicht",
                                area="Wohnzimmer",
                                entity_class=SmartHomeEntityClass.LIGHT,
                                controllable=True,
                                supported_commands=(SmartHomeCommand.TURN_ON,),
                            ),
                        )
                    ),
                    controller=right_controller,
                ),
            )
        )

        entities = aggregate.list_entities()
        self.assertEqual(
            {entity.entity_id for entity in entities},
            {
                build_routed_entity_id("bridge-a", "light-1"),
                build_routed_entity_id("bridge-b", "light-2"),
            },
        )
        self.assertEqual(
            {entity.label for entity in entities},
            {"Deckenlicht (bridge-a)", "Deckenlicht (bridge-b)"},
        )

        result = aggregate.control(
            command=SmartHomeCommand.TURN_ON,
            entity_ids=tuple(entity.entity_id for entity in entities),
            parameters={},
        )

        self.assertEqual(left_controller.calls[0]["entity_ids"], ("light-1",))
        self.assertEqual(right_controller.calls[0]["entity_ids"], ("light-2",))
        self.assertEqual(
            {target["entity_id"] for target in result["targets"]},
            {
                build_routed_entity_id("bridge-a", "light-1"),
                build_routed_entity_id("bridge-b", "light-2"),
            },
        )

    def test_aggregated_provider_merges_sensor_stream_batches(self) -> None:
        left_stream = _StaticSensorStream(
            SmartHomeEventBatch(
                events=(
                    SmartHomeEvent(
                        event_id="evt-left",
                        provider="hue",
                        entity_id="motion-left",
                        event_kind=SmartHomeEventKind.MOTION_DETECTED,
                        observed_at="2026-03-21T10:00:00Z",
                    ),
                ),
                next_cursor="cursor-left-next",
            )
        )
        right_stream = _StaticSensorStream(
            SmartHomeEventBatch(
                events=(
                    SmartHomeEvent(
                        event_id="evt-right",
                        provider="hue",
                        entity_id="motion-right",
                        event_kind=SmartHomeEventKind.MOTION_CLEARED,
                        observed_at="2026-03-21T10:00:01Z",
                    ),
                ),
                next_cursor="cursor-right-next",
            )
        )
        aggregate = AggregatedSmartHomeProvider(
            (
                RoutedSmartHomeProvider(
                    route_id="bridge-a",
                    entity_provider=_FakeEntityProvider(()),
                    sensor_stream=left_stream,
                ),
                RoutedSmartHomeProvider(
                    route_id="bridge-b",
                    entity_provider=_FakeEntityProvider(()),
                    sensor_stream=right_stream,
                ),
            )
        )

        batch = aggregate.read_sensor_stream(
            cursor=json.dumps(
                {
                    "version": 1,
                    "routes": {
                        "bridge-a": "cursor-left",
                        "bridge-b": "cursor-right",
                    },
                }
            ),
            limit=2,
        )

        self.assertEqual(left_stream.calls[0], {"cursor": "cursor-left", "limit": 2})
        self.assertEqual(right_stream.calls[0], {"cursor": "cursor-right", "limit": 2})
        self.assertEqual(
            [event.entity_id for event in batch.events],
            [
                build_routed_entity_id("bridge-b", "motion-right"),
                build_routed_entity_id("bridge-a", "motion-left"),
            ],
        )
        self.assertEqual(
            [event.details["route_id"] for event in batch.events],
            ["bridge-b", "bridge-a"],
        )
        self.assertIsNone(batch.next_cursor)

    def test_event_batch_accepts_opaque_route_cursor_tokens(self) -> None:
        cursor = "smarthome-route-cursor:" + ("a" * 600)

        batch = SmartHomeEventBatch(events=(), next_cursor=cursor)

        self.assertEqual(batch.next_cursor, cursor)


if __name__ == "__main__":
    unittest.main()
