from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.respeaker.led_profiles import (
    ANSWERING_LED_PROFILE,
    ERROR_LED_PROFILE,
    _FIELD_TUNED_PULSE_SCALE,
    LISTENING_LED_PROFILE,
    PROCESSING_LED_PROFILE,
    WAITING_LED_PROFILE,
    resolve_respeaker_led_profile,
)


class ReSpeakerLedProfileTests(unittest.TestCase):
    def test_resolves_waiting_profile_by_default(self) -> None:
        profile = resolve_respeaker_led_profile(runtime_status=None)

        self.assertEqual(profile, WAITING_LED_PROFILE)

    def test_resolves_runtime_states_to_expected_profiles(self) -> None:
        self.assertEqual(
            resolve_respeaker_led_profile(runtime_status="listening"),
            LISTENING_LED_PROFILE,
        )
        self.assertEqual(
            resolve_respeaker_led_profile(runtime_status="processing"),
            PROCESSING_LED_PROFILE,
        )
        self.assertEqual(
            resolve_respeaker_led_profile(runtime_status="printing"),
            PROCESSING_LED_PROFILE,
        )
        self.assertEqual(
            resolve_respeaker_led_profile(runtime_status="answering"),
            ANSWERING_LED_PROFILE,
        )

    def test_error_message_forces_error_profile(self) -> None:
        profile = resolve_respeaker_led_profile(
            runtime_status="waiting",
            error_message="transport blocked",
        )

        self.assertEqual(profile, ERROR_LED_PROFILE)

    def test_scaled_rgb_stays_bounded(self) -> None:
        rgb = LISTENING_LED_PROFILE.scaled_rgb(timestamp_s=0.125)

        self.assertEqual(len(rgb), 3)
        self.assertTrue(all(0 <= channel <= 255 for channel in rgb))

    def test_profiles_use_field_tuned_pulse_scale(self) -> None:
        self.assertAlmostEqual(WAITING_LED_PROFILE.pulse_hz, 0.8 * _FIELD_TUNED_PULSE_SCALE)
        self.assertAlmostEqual(LISTENING_LED_PROFILE.pulse_hz, 1.6 * _FIELD_TUNED_PULSE_SCALE)
        self.assertAlmostEqual(PROCESSING_LED_PROFILE.pulse_hz, 1.0 * _FIELD_TUNED_PULSE_SCALE)
        self.assertAlmostEqual(ANSWERING_LED_PROFILE.pulse_hz, 1.6 * _FIELD_TUNED_PULSE_SCALE)
        self.assertAlmostEqual(ERROR_LED_PROFILE.pulse_hz, 0.5 * _FIELD_TUNED_PULSE_SCALE)


if __name__ == "__main__":
    unittest.main()
