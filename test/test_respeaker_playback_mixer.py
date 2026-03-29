from pathlib import Path
from unittest import mock
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.respeaker_playback_mixer import ensure_respeaker_playback_mixer


class RespeakerPlaybackMixerTests(unittest.TestCase):
    def test_skips_non_respeaker_playback_device(self) -> None:
        result = ensure_respeaker_playback_mixer("default")

        self.assertFalse(result.attempted)
        self.assertEqual(result.reason, "non_respeaker_playback")
        self.assertIsNone(result.card_index)

    def test_normalizes_matching_respeaker_card_controls_and_stores_state(self) -> None:
        with (
            mock.patch(
                "twinr.hardware.respeaker_playback_mixer._resolve_respeaker_card_index",
                return_value=3,
            ),
            mock.patch(
                "twinr.hardware.respeaker_playback_mixer._list_simple_mixer_controls",
                return_value=("PCM,0", "PCM,1", "Capture,0", "Twinr Playback,0"),
            ),
            mock.patch(
                "twinr.hardware.respeaker_playback_mixer._control_has_playback_capability",
                side_effect=lambda **kwargs: kwargs["control_ref"] != "Capture,0",
            ),
            mock.patch(
                "twinr.hardware.respeaker_playback_mixer._set_control_percent",
                return_value=True,
            ) as set_control,
            mock.patch(
                "twinr.hardware.respeaker_playback_mixer._store_card_playback_state",
                return_value=True,
            ) as store_state,
            mock.patch("twinr.hardware.respeaker_playback_mixer.shutil.which", return_value="/usr/bin/amixer"),
        ):
            result = ensure_respeaker_playback_mixer("twinr_playback_softvol")

        self.assertTrue(result.attempted)
        self.assertEqual(result.card_index, 3)
        self.assertEqual(result.normalized_controls, ("PCM,0", "PCM,1", "Twinr Playback,0"))
        self.assertTrue(result.stored_state)
        self.assertEqual(set_control.call_count, 3)
        set_control.assert_any_call(
            amixer_path="/usr/bin/amixer",
            card_index=3,
            control_ref="PCM,0",
            percent=100,
        )
        store_state.assert_called_once_with(3)


if __name__ == "__main__":
    unittest.main()
