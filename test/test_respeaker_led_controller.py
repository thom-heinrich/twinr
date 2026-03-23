from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.respeaker.led_controller import ReSpeakerLedController
from twinr.hardware.respeaker.led_profiles import WAITING_LED_PROFILE
from twinr.hardware.respeaker.models import ReSpeakerTransportAvailability


class _FakeTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def write_parameter(self, spec, values, *, probe=None):
        del probe
        self.calls.append((spec.name, tuple(values)))
        return ReSpeakerTransportAvailability(backend="libusb", available=True)


class ReSpeakerLedControllerTests(unittest.TestCase):
    def test_render_sets_single_color_effect_then_color(self) -> None:
        transport = _FakeTransport()
        controller = ReSpeakerLedController(transport=transport, emit=lambda _line: None)

        changed = controller.render(WAITING_LED_PROFILE, at_monotonic_s=0.0)

        self.assertTrue(changed)
        self.assertEqual(transport.calls[0], ("LED_EFFECT", (3,)))
        self.assertEqual(transport.calls[1][0], "LED_COLOR")

    def test_render_skips_duplicate_color_write(self) -> None:
        transport = _FakeTransport()
        controller = ReSpeakerLedController(
            transport=transport,
            emit=lambda _line: None,
            effect_refresh_interval_s=10.0,
        )

        controller.render(WAITING_LED_PROFILE, at_monotonic_s=0.0)
        changed = controller.render(WAITING_LED_PROFILE, at_monotonic_s=0.0)

        self.assertFalse(changed)
        self.assertEqual(len(transport.calls), 2)

    def test_render_reasserts_effect_after_refresh_interval(self) -> None:
        transport = _FakeTransport()
        controller = ReSpeakerLedController(
            transport=transport,
            emit=lambda _line: None,
            effect_refresh_interval_s=1.0,
        )

        controller.render(WAITING_LED_PROFILE, at_monotonic_s=0.0)
        transport.calls.clear()

        changed = controller.render(WAITING_LED_PROFILE, at_monotonic_s=1.1)

        self.assertTrue(changed)
        self.assertEqual(transport.calls[0], ("LED_EFFECT", (3,)))
        self.assertEqual(transport.calls[1][0], "LED_COLOR")

    def test_off_writes_led_effect_zero(self) -> None:
        transport = _FakeTransport()
        controller = ReSpeakerLedController(transport=transport, emit=lambda _line: None)

        controller.render(WAITING_LED_PROFILE, at_monotonic_s=0.0)
        turned_off = controller.off()

        self.assertTrue(turned_off)
        self.assertEqual(transport.calls[-1], ("LED_EFFECT", (0,)))


if __name__ == "__main__":
    unittest.main()
