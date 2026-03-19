from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.respeaker.indicator_policy import resolve_respeaker_indicator_state


class ReSpeakerIndicatorPolicyTests(unittest.TestCase):
    def test_indicator_prefers_muted_over_listening(self) -> None:
        state = resolve_respeaker_indicator_state(
            runtime_status="listening",
            runtime_alert_code="mic_muted",
            mute_active=True,
        )

        self.assertEqual(state.semantics, "listening_mute_only")
        self.assertEqual(state.mode, "muted")
        self.assertFalse(state.direction_hint_mirrored)

    def test_indicator_uses_listening_and_never_direction(self) -> None:
        state = resolve_respeaker_indicator_state(
            runtime_status="listening",
            runtime_alert_code="ready",
            mute_active=False,
        )

        self.assertEqual(state.mode, "listening")
        self.assertEqual(state.reason, "runtime_listening")
        self.assertFalse(state.direction_hint_mirrored)


if __name__ == "__main__":
    unittest.main()
