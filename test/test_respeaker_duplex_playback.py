from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import twinr.hardware.respeaker_duplex_playback as respeaker_duplex_playback


class ReSpeakerDuplexPlaybackTests(unittest.TestCase):
    def setUp(self) -> None:
        self._shared_guard_keys = tuple(respeaker_duplex_playback._SHARED_GUARDS.keys())
        self._sample_rate_keys = tuple(respeaker_duplex_playback._SUCCESSFUL_SAMPLE_RATE_BY_DEVICE.keys())
        respeaker_duplex_playback._SHARED_GUARDS.clear()
        respeaker_duplex_playback._SUCCESSFUL_SAMPLE_RATE_BY_DEVICE.clear()

    def tearDown(self) -> None:
        respeaker_duplex_playback._SHARED_GUARDS.clear()
        respeaker_duplex_playback._SUCCESSFUL_SAMPLE_RATE_BY_DEVICE.clear()
        for key in self._shared_guard_keys:
            # Other tests in this file do not preserve external state objects.
            # Restore only the key layout contract if this file is extended later.
            respeaker_duplex_playback._SHARED_GUARDS.pop(key, None)
        for key in self._sample_rate_keys:
            respeaker_duplex_playback._SUCCESSFUL_SAMPLE_RATE_BY_DEVICE.pop(key, None)

    def test_spawn_aplay_handle_sets_parent_death_preexec(self) -> None:
        fake_process = mock.Mock()

        with (
            mock.patch.object(
                respeaker_duplex_playback,
                "_resolve_aplay_executable",
                return_value="/usr/bin/aplay",
            ),
            mock.patch.object(
                respeaker_duplex_playback,
                "_start_stderr_drain_thread",
            ) as stderr_thread,
            mock.patch.object(
                respeaker_duplex_playback.subprocess,
                "Popen",
                return_value=fake_process,
            ) as popen,
        ):
            handle = respeaker_duplex_playback._spawn_aplay_handle(
                playback_device="twinr_playback_softvol",
                sample_rate_hz=24_000,
                env={},
            )

        self.assertIs(handle.process, fake_process)
        self.assertTrue(callable(popen.call_args.kwargs["preexec_fn"]))
        self.assertTrue(popen.call_args.kwargs["close_fds"])
        stderr_thread.assert_called_once_with(handle)

    def test_acquire_shared_guard_raises_when_playback_device_is_busy(self) -> None:
        fake_handle = mock.Mock()
        fake_handle.diagnostics.return_value = (
            "aplay: main:831: audio open error: Device or resource busy"
        )

        with (
            mock.patch.object(
                respeaker_duplex_playback,
                "_spawn_aplay_handle",
                return_value=fake_handle,
            ),
            mock.patch.object(
                respeaker_duplex_playback,
                "_wait_for_playback_ready",
                return_value=False,
            ),
            mock.patch.object(respeaker_duplex_playback, "_stop_process_handle") as stop_handle,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "could not acquire twinr_playback_softvol",
            ):
                respeaker_duplex_playback._acquire_shared_guard(
                    playback_device="twinr_playback_softvol",
                    preferred_sample_rate_hz=24_000,
                )

        stop_handle.assert_called_once_with(fake_handle)
        self.assertEqual(respeaker_duplex_playback._SHARED_GUARDS, {})


if __name__ == "__main__":
    unittest.main()
