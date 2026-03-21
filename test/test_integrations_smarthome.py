from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.integrations import (
    IntegrationRequest,
    SmartHomeCommand,
    SmartHomeEntity,
    SmartHomeEntityClass,
    SmartHomeEvent,
    SmartHomeEventBatch,
    SmartHomeEventKind,
    SmartHomeIntegrationAdapter,
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
        return {"applied": True}


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


if __name__ == "__main__":
    unittest.main()
