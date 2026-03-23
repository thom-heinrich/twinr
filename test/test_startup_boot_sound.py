from pathlib import Path
from threading import Event
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.agent.workflows.playback_coordinator import PlaybackPriority
from twinr.agent.workflows.startup_boot_sound import (
    build_startup_boot_sound_wav_bytes,
    play_startup_boot_sound,
    start_startup_boot_sound,
)


class _FakePlaybackCoordinator:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def play_wav_bytes(self, *, owner: str, priority: int, wav_bytes: bytes, should_stop=None, atomic: bool = False):
        del should_stop
        self.requests.append(
            {
                "owner": owner,
                "priority": priority,
                "wav_bytes": wav_bytes,
                "atomic": atomic,
            }
        )
        return SimpleNamespace(preempted=False)


class StartupBootSoundTests(unittest.TestCase):
    def test_build_startup_boot_sound_wav_bytes_returns_none_when_media_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)

            self.assertIsNone(build_startup_boot_sound_wav_bytes(config))

    def test_build_startup_boot_sound_wav_bytes_uses_trim_and_fade_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)

            with mock.patch(
                "twinr.agent.workflows.startup_boot_sound.build_rendered_audio_clip_wav_bytes",
                return_value=b"NORMALIZED",
            ) as render_mock:
                rendered = build_startup_boot_sound_wav_bytes(config)

        self.assertEqual(rendered, b"NORMALIZED")
        render_mock.assert_called_once()
        self.assertIs(render_mock.call_args.args[0], config)
        spec = render_mock.call_args.args[1]
        self.assertEqual(spec.relative_path, Path("media") / "boot.mp3")
        self.assertEqual(spec.clip_start_s, 4.0)
        self.assertEqual(spec.clip_duration_s, 7.0)
        self.assertEqual(spec.fade_out_duration_s, 3.0)
        self.assertEqual(spec.output_gain, 0.2)

    def test_play_startup_boot_sound_queues_feedback_priority_clip(self) -> None:
        coordinator = _FakePlaybackCoordinator()
        lines: list[str] = []

        with mock.patch(
            "twinr.agent.workflows.startup_boot_sound.build_startup_boot_sound_wav_bytes",
            return_value=b"RIFFBOOT",
        ):
            played = play_startup_boot_sound(
                config=TwinrConfig(),
                playback_coordinator=coordinator,
                emit=lines.append,
            )

        self.assertTrue(played)
        self.assertEqual(
            coordinator.requests,
            [
                {
                    "owner": "startup_boot_sound",
                    "priority": PlaybackPriority.FEEDBACK,
                    "wav_bytes": b"RIFFBOOT",
                    "atomic": False,
                }
            ],
        )
        self.assertIn("boot_sound=played", lines)

    def test_start_startup_boot_sound_spawns_background_worker(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            boot_path = Path(temp_dir) / "media" / "boot.mp3"
            boot_path.parent.mkdir(parents=True, exist_ok=True)
            boot_path.write_bytes(b"ID3-BOOT")
            called = Event()

            def _mark_called(**_kwargs) -> bool:
                called.set()
                return True

            with mock.patch(
                "twinr.agent.workflows.startup_boot_sound.play_startup_boot_sound",
                side_effect=_mark_called,
            ):
                thread = start_startup_boot_sound(
                    config=TwinrConfig(project_root=temp_dir),
                    playback_coordinator=_FakePlaybackCoordinator(),
                    emit=None,
                )

        self.assertIsNotNone(thread)
        assert thread is not None
        self.assertTrue(thread.daemon)
        thread.join(timeout=1.0)
        self.assertFalse(thread.is_alive())
        self.assertTrue(called.is_set())


if __name__ == "__main__":
    unittest.main()
