"""Validate smart-home stream normalization and worker behavior."""

from pathlib import Path
import sys
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.integrations import IntegrationRequest, SmartHomeIntegrationAdapter, SmartHomeEvent, SmartHomeEventBatch, SmartHomeEventKind, manifest_for_id
from twinr.integrations.smarthome.runtime import SmartHomeObservationBuilder, SmartHomeSensorWorker


class _FakeEntityProvider:
    def list_entities(self, *, entity_ids=(), entity_class=None, include_unavailable=False):
        del entity_ids, entity_class, include_unavailable
        return []


class _FakeSensorStream:
    def __init__(self):
        self.calls = 0

    def read_sensor_stream(self, *, cursor=None, limit):
        del cursor, limit
        self.calls += 1
        if self.calls == 1:
            return SmartHomeEventBatch(
                events=(
                    SmartHomeEvent(
                        event_id="evt-1",
                        provider="fake",
                        entity_id="motion-1",
                        event_kind=SmartHomeEventKind.MOTION_DETECTED,
                        observed_at="2026-03-21T10:00:00Z",
                    ),
                ),
                next_cursor="cursor-1",
            )
        return SmartHomeEventBatch(events=(), next_cursor="cursor-1")


class SmartHomeRuntimeTests(unittest.TestCase):
    def test_observation_builder_tracks_motion_and_device_offline_state(self) -> None:
        builder = SmartHomeObservationBuilder()

        first = builder.build(
            SmartHomeEventBatch(
                events=(
                    SmartHomeEvent(
                        event_id="evt-1",
                        provider="fake",
                        entity_id="motion-1",
                        event_kind=SmartHomeEventKind.MOTION_DETECTED,
                        observed_at="2026-03-21T10:00:00Z",
                    ),
                )
            )
        )
        second = builder.build(
            SmartHomeEventBatch(
                events=(
                    SmartHomeEvent(
                        event_id="evt-2",
                        provider="fake",
                        entity_id="motion-1",
                        event_kind=SmartHomeEventKind.MOTION_CLEARED,
                        observed_at="2026-03-21T10:00:05Z",
                    ),
                    SmartHomeEvent(
                        event_id="evt-3",
                        provider="fake",
                        entity_id="conn-1",
                        event_kind=SmartHomeEventKind.DEVICE_OFFLINE,
                        observed_at="2026-03-21T10:00:06Z",
                    ),
                )
            )
        )

        assert first is not None
        assert second is not None
        self.assertEqual(first.event_names, ("smart_home.motion_detected",))
        self.assertTrue(first.facts["smart_home"]["motion_detected"])
        self.assertFalse(second.facts["smart_home"]["motion_detected"])
        self.assertTrue(second.facts["smart_home"]["device_offline"])
        self.assertEqual(second.facts["smart_home"]["offline_entity_ids"], ["conn-1"])

    def test_sensor_worker_publishes_observations(self) -> None:
        manifest = manifest_for_id("smart_home_hub")
        assert manifest is not None
        adapter = SmartHomeIntegrationAdapter(
            manifest=manifest,
            entity_provider=_FakeEntityProvider(),
            sensor_stream=_FakeSensorStream(),
        )
        received = []
        ready = threading.Event()

        worker = SmartHomeSensorWorker(
            adapter_loader=lambda: adapter,
            observation_callback=lambda observation: received.append(observation) or ready.set(),
            idle_sleep_s=0.01,
            retry_delay_s=0.01,
            batch_limit=4,
        )
        worker.start()
        try:
            self.assertTrue(ready.wait(1.0))
        finally:
            worker.stop(timeout_s=1.0)

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].event_names, ("smart_home.motion_detected",))


if __name__ == "__main__":
    unittest.main()
