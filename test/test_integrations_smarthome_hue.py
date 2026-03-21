from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.integrations import SmartHomeCommand, SmartHomeEntityClass
from twinr.integrations.smarthome.hue import HueBridgeClient, HueBridgeConfig, HueSmartHomeProvider


class _FakeHueClient:
    def __init__(self, *, resources, stream_events=()):
        self._resources = list(resources)
        self._stream_events = list(stream_events)
        self.put_calls = []

    def list_resources(self):
        return list(self._resources)

    def put_resource(self, resource_type, resource_id, payload):
        call = {
            "resource_type": resource_type,
            "resource_id": resource_id,
            "payload": dict(payload),
        }
        self.put_calls.append(call)
        return {"data": [{"rid": resource_id, "rtype": resource_type}]}

    def read_event_stream(self, *, timeout_s=None, max_events=20):
        del timeout_s
        return list(self._stream_events[:max_events])


class HueSmartHomeProviderTests(unittest.TestCase):
    def test_hue_client_treats_idle_event_timeout_as_empty_batch(self) -> None:
        client = HueBridgeClient(
            HueBridgeConfig(
                bridge_host="192.168.1.20",
                application_key="local-key",
                verify_tls=False,
                timeout_s=2.0,
            ),
            event_reader=lambda path, timeout_s, max_events: (_ for _ in ()).throw(TimeoutError("idle")),
        )

        events = client.read_event_stream(timeout_s=1.0, max_events=4)

        self.assertEqual(events, [])

    def test_list_entities_normalizes_hue_resources(self) -> None:
        client = _FakeHueClient(
            resources=(
                {
                    "id": "device-light",
                    "type": "device",
                    "metadata": {"name": "Wohnzimmerlampe"},
                },
                {
                    "id": "light-1",
                    "type": "light",
                    "owner": {"rid": "device-light", "rtype": "device"},
                    "on": {"on": True},
                    "dimming": {"brightness": 42.0},
                },
                {
                    "id": "scene-1",
                    "type": "scene",
                    "metadata": {"name": "Lesen"},
                    "status": {"active": "inactive"},
                },
                {
                    "id": "room-1",
                    "type": "room",
                    "metadata": {"name": "Wohnzimmer"},
                    "children": [{"rid": "device-light", "rtype": "device"}],
                },
            )
        )
        provider = HueSmartHomeProvider(client=client)

        entities = provider.list_entities()

        self.assertEqual(len(entities), 2)
        light = next(item for item in entities if item.entity_id == "light-1")
        scene = next(item for item in entities if item.entity_id == "scene-1")
        self.assertEqual(light.entity_class, SmartHomeEntityClass.LIGHT)
        self.assertEqual(light.label, "Wohnzimmerlampe")
        self.assertEqual(light.area, "Wohnzimmer")
        self.assertTrue(light.state["on"])
        self.assertEqual(scene.entity_class, SmartHomeEntityClass.SCENE)
        self.assertEqual(scene.label, "Lesen")

    def test_control_maps_generic_commands_to_hue_payloads(self) -> None:
        client = _FakeHueClient(
            resources=(
                {"id": "light-1", "type": "light"},
                {"id": "scene-1", "type": "scene"},
            )
        )
        provider = HueSmartHomeProvider(client=client)

        provider.control(
            command=SmartHomeCommand.TURN_ON,
            entity_ids=("light-1",),
            parameters={},
        )
        provider.control(
            command=SmartHomeCommand.ACTIVATE,
            entity_ids=("scene-1",),
            parameters={},
        )

        self.assertEqual(
            client.put_calls[0],
            {
                "resource_type": "light",
                "resource_id": "light-1",
                "payload": {"on": {"on": True}},
            },
        )
        self.assertEqual(
            client.put_calls[1],
            {
                "resource_type": "scene",
                "resource_id": "scene-1",
                "payload": {"recall": {"action": "active"}},
            },
        )

    def test_read_sensor_stream_normalizes_motion_and_connectivity_events(self) -> None:
        client = _FakeHueClient(
            resources=(),
            stream_events=(
                {
                    "id": "evt-1",
                    "event": "message",
                    "data": [
                        {
                            "type": "update",
                            "creationtime": "2026-03-21T10:00:00Z",
                            "data": [
                                {
                                    "id": "motion-1",
                                    "type": "motion",
                                    "motion": {
                                        "motion_report": {
                                            "motion": True,
                                            "changed": "2026-03-21T10:00:00Z",
                                        }
                                    },
                                },
                                {
                                    "id": "conn-1",
                                    "type": "zigbee_connectivity",
                                    "status": "disconnected",
                                },
                            ],
                        }
                    ],
                },
            ),
        )
        provider = HueSmartHomeProvider(client=client)

        batch = provider.read_sensor_stream(limit=5)

        self.assertEqual(batch.next_cursor, "evt-1")
        self.assertEqual(len(batch.events), 2)
        self.assertEqual(batch.events[0].event_kind.value, "motion_detected")
        self.assertEqual(batch.events[1].event_kind.value, "device_offline")


if __name__ == "__main__":
    unittest.main()
