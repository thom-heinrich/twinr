from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.hardware.buttons import ButtonAction
from twinr.hardware.pir import GpioPirMonitor, build_pir_binding, configured_pir_monitor


class FakeButtonMonitor:
    def __init__(self, *, snapshot_value: int = 0, events: list[object] | None = None) -> None:
        self.snapshot_value = snapshot_value
        self.events = list(events or [])
        self.opened = False

    def open(self):
        self.opened = True
        return self

    def close(self) -> None:
        self.opened = False

    def snapshot_values(self) -> dict[int, int]:
        return {26: self.snapshot_value}

    def poll(self, timeout=None):
        if not self.events:
            return None
        return self.events.pop(0)

    def iter_events(self, *, duration_s=None, poll_timeout=0.5):
        while self.events:
            yield self.events.pop(0)


class PirHelperTests(unittest.TestCase):
    def test_build_pir_binding_uses_configured_line(self) -> None:
        config = TwinrConfig(pir_motion_gpio=26)

        binding = build_pir_binding(config)

        self.assertEqual(binding.name, "pir")
        self.assertEqual(binding.line_offset, 26)

    def test_configured_monitor_requires_gpio(self) -> None:
        with self.assertRaisesRegex(ValueError, "TWINR_PIR_MOTION_GPIO"):
            configured_pir_monitor(TwinrConfig())

    def test_motion_detected_uses_current_high_level(self) -> None:
        monitor = GpioPirMonitor(
            chip_name="gpiochip0",
            binding=build_pir_binding(TwinrConfig(pir_motion_gpio=26)),
            active_high=True,
            monitor=FakeButtonMonitor(snapshot_value=1),
        )

        with monitor:
            self.assertTrue(monitor.motion_detected())

    def test_wait_for_motion_translates_rising_event(self) -> None:
        fake_monitor = FakeButtonMonitor(
            snapshot_value=0,
            events=[
                SimpleNamespace(
                    action=ButtonAction.PRESSED,
                    raw_edge="rising",
                    timestamp_ns=123,
                )
            ],
        )
        monitor = GpioPirMonitor(
            chip_name="gpiochip0",
            binding=build_pir_binding(TwinrConfig(pir_motion_gpio=26)),
            active_high=True,
            monitor=fake_monitor,
        )

        with monitor:
            event = monitor.wait_for_motion(duration_s=1.0)

        self.assertIsNotNone(event)
        assert event is not None
        self.assertTrue(event.motion_detected)
        self.assertEqual(event.raw_edge, "rising")

    def test_active_low_sensor_inverts_current_level(self) -> None:
        monitor = GpioPirMonitor(
            chip_name="gpiochip0",
            binding=build_pir_binding(TwinrConfig(pir_motion_gpio=26, pir_active_high=False)),
            active_high=False,
            monitor=FakeButtonMonitor(snapshot_value=0),
        )

        with monitor:
            self.assertTrue(monitor.motion_detected())


if __name__ == "__main__":
    unittest.main()
