from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.workflows.rendered_audio_clip import RenderedAudioClipSpec, build_rendered_audio_clip_wav_bytes


class RenderedAudioClipTests(unittest.TestCase):
    def test_build_rendered_audio_clip_wav_bytes_returns_none_when_media_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            spec = RenderedAudioClipSpec(
                relative_path=Path("media") / "missing.mp3",
                clip_start_s=1.0,
                clip_duration_s=2.0,
                fade_in_duration_s=0.5,
                fade_out_start_s=1.5,
                fade_out_duration_s=0.3,
                output_gain=0.5,
            )

            self.assertIsNone(build_rendered_audio_clip_wav_bytes(config, spec))

    def test_build_rendered_audio_clip_wav_bytes_uses_trim_fade_and_gain_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            clip_path = Path(temp_dir) / "media" / "waiting.mp3"
            clip_path.parent.mkdir(parents=True, exist_ok=True)
            clip_path.write_bytes(b"ID3-WAITING")
            config = TwinrConfig(project_root=temp_dir)
            spec = RenderedAudioClipSpec(
                relative_path=Path("media") / "waiting.mp3",
                clip_start_s=3.0,
                clip_duration_s=3.0,
                fade_in_duration_s=0.9,
                fade_out_start_s=2.75,
                fade_out_duration_s=0.25,
                output_gain=0.5,
                normalize_max_gain=1.0,
            )

            with mock.patch("twinr.agent.workflows.rendered_audio_clip.shutil.which", return_value="/usr/bin/ffmpeg"):
                with mock.patch(
                    "twinr.agent.workflows.rendered_audio_clip.subprocess.run",
                    return_value=SimpleNamespace(returncode=0, stdout=b"RIFFWAVE", stderr=b""),
                ) as run_mock:
                    with mock.patch(
                        "twinr.agent.workflows.rendered_audio_clip.normalize_wav_playback_level",
                        return_value=b"NORMALIZED",
                    ) as normalize_mock:
                        rendered = build_rendered_audio_clip_wav_bytes(config, spec)

        self.assertEqual(rendered, b"NORMALIZED")
        normalize_mock.assert_called_once_with(b"RIFFWAVE", max_gain=1.0)
        command = run_mock.call_args.args[0]
        self.assertEqual(command[0], "/usr/bin/ffmpeg")
        self.assertIn("3.000", command)
        filter_index = command.index("-af") + 1
        self.assertEqual(
            command[filter_index],
            "volume=0.500,afade=t=in:st=0:d=0.900,afade=t=out:st=2.750:d=0.250",
        )
        self.assertIn(str(clip_path), command)

    def test_build_rendered_audio_clip_wav_bytes_applies_requested_playback_speed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            clip_path = Path(temp_dir) / "media" / "dragon.mp3"
            clip_path.parent.mkdir(parents=True, exist_ok=True)
            clip_path.write_bytes(b"ID3-DRAGON")
            config = TwinrConfig(project_root=temp_dir)
            spec = RenderedAudioClipSpec(
                relative_path=Path("media") / "dragon.mp3",
                clip_start_s=0.0,
                clip_duration_s=0.8,
                fade_in_duration_s=0.09,
                fade_out_start_s=1.08,
                fade_out_duration_s=0.15,
                output_gain=0.105,
                playback_speed=0.65,
            )

            with mock.patch("twinr.agent.workflows.rendered_audio_clip.shutil.which", return_value="/usr/bin/ffmpeg"):
                with mock.patch(
                    "twinr.agent.workflows.rendered_audio_clip.subprocess.run",
                    return_value=SimpleNamespace(returncode=0, stdout=b"RIFFWAVE", stderr=b""),
                ) as run_mock:
                    with mock.patch(
                        "twinr.agent.workflows.rendered_audio_clip.normalize_wav_playback_level",
                        return_value=b"NORMALIZED",
                    ):
                        rendered = build_rendered_audio_clip_wav_bytes(config, spec)

        self.assertEqual(rendered, b"NORMALIZED")
        command = run_mock.call_args.args[0]
        filter_index = command.index("-af") + 1
        self.assertEqual(
            command[filter_index],
            "volume=0.105,atempo=0.650,afade=t=in:st=0:d=0.090,afade=t=out:st=1.080:d=0.150",
        )
        self.assertIn(str(clip_path), command)


if __name__ == "__main__":
    unittest.main()
