from pathlib import Path
import sys
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.hardware.buttons import (
    ButtonAction,
    ButtonBinding,
    GpioButtonMonitor,
    build_button_bindings,
    build_probe_bindings,
    edge_to_action,
)


class _FakeLine:
    def __init__(self, state: dict[str, int]) -> None:
        self._state = state

    def get_value(self) -> int:
        return self._state["value"]

    def release(self) -> None:
        return None


class ButtonHelperTests(unittest.TestCase):
    def test_build_button_bindings_uses_configured_lines(self) -> None:
        config = TwinrConfig(green_button_gpio=17, yellow_button_gpio=27)

        bindings = build_button_bindings(config)

        self.assertEqual(bindings[0].name, "green")
        self.assertEqual(bindings[0].line_offset, 17)
        self.assertEqual(bindings[1].name, "yellow")
        self.assertEqual(bindings[1].line_offset, 27)

    def test_build_probe_bindings_sorts_and_deduplicates_lines(self) -> None:
        bindings = build_probe_bindings([27, 17, 27, 22])

        self.assertEqual([binding.line_offset for binding in bindings], [17, 22, 27])
        self.assertEqual([binding.name for binding in bindings], ["gpio17", "gpio22", "gpio27"])

    def test_active_low_edge_mapping_matches_button_press(self) -> None:
        self.assertEqual(
            edge_to_action(0, active_low=True),
            ButtonAction.PRESSED,
        )
        self.assertEqual(
            edge_to_action(1, active_low=True),
            ButtonAction.RELEASED,
        )

    def test_legacy_monitor_buffers_presses_in_background(self) -> None:
        state = {"value": 1}
        monitor = GpioButtonMonitor(
            "gpiochip0",
            bindings=(ButtonBinding(name="yellow", line_offset=22),),
            active_low=True,
            debounce_ms=40,
        )
        monitor._line_by_offset = {22: _FakeLine(state)}
        monitor._last_values = {22: 1}
        monitor._start_background_sampler()
        self.addCleanup(monitor.close)

        time.sleep(0.05)
        state["value"] = 0
        time.sleep(0.12)
        state["value"] = 1

        pressed = monitor.poll(timeout=0.5)
        released = monitor.poll(timeout=0.5)

        self.assertIsNotNone(pressed)
        self.assertIsNotNone(released)
        self.assertEqual(pressed.name, "yellow")
        self.assertEqual(pressed.action, ButtonAction.PRESSED)
        self.assertEqual(released.action, ButtonAction.RELEASED)
