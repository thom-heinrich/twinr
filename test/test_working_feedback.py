import audioop
import io
from pathlib import Path
from threading import Event
import sys
import tempfile
import time
import unittest
from unittest import mock
import wave
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.workflows import working_feedback as working_feedback_mod
from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.workflows.rendered_audio_clip import (
    RenderedAudioClipConfigurationError,
    RenderedAudioClipSpec,
)
from twinr.agent.workflows.working_feedback import (
    WorkingFeedbackMediaSpec,
    WorkingFeedbackProfile,
    start_working_feedback_loop,
)

_PROCESSING_ASSET_RELATIVE_PATH = (
    Path("media") / "dragon-studio-computer-startup-sound-effect-312870.mp3"
)


class _FakeFeedbackPlayer:
    def __init__(self) -> None:
        self.wav_payloads: list[bytes] = []
        self.tone_sequences: list[tuple[tuple[int, int], ...]] = []
        self.tone_volumes: list[float] = []
        self.wav_started = Event()

    def play_wav_chunks(self, chunks, *, should_stop=None) -> None:
        payload = bytearray()
        for chunk in chunks:
            payload.extend(chunk)
        self.wav_payloads.append(bytes(payload))
        self.wav_started.set()
        deadline = time.monotonic() + 0.2
        while time.monotonic() < deadline:
            if should_stop is not None and should_stop():
                return
            time.sleep(0.005)

    def play_tone_sequence(self, sequence, *, volume: float, sample_rate: int, gap_ms: int = 0) -> None:
        del sample_rate, gap_ms
        self.tone_sequences.append(tuple(sequence))
        self.tone_volumes.append(volume)

    def stop_playback(self) -> None:
        return


class _ToneOnlyFeedbackPlayer:
    def __init__(self) -> None:
        self.tone_sequences: list[tuple[tuple[int, int], ...]] = []
        self.tone_volumes: list[float] = []

    def play_tone_sequence(self, sequence, *, volume: float, sample_rate: int, gap_ms: int = 0) -> None:
        del sample_rate, gap_ms
        self.tone_sequences.append(tuple(sequence))
        self.tone_volumes.append(volume)

    def stop_playback(self) -> None:
        return


class _InterruptingFeedbackPlayer(_FakeFeedbackPlayer):
    def play_wav_chunks(self, chunks, *, should_stop=None) -> None:
        payload = bytearray()
        for chunk in chunks:
            payload.extend(chunk)
        self.wav_payloads.append(bytes(payload))
        self.wav_started.set()
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            if should_stop is not None and should_stop():
                raise RuntimeError("interrupted")
            time.sleep(0.005)


class _PCMFeedbackPlayer(_FakeFeedbackPlayer):
    def __init__(self) -> None:
        super().__init__()
        self.pcm_chunks: list[bytes] = []
        self.pcm_started = Event()
        self.sample_rates: list[int] = []
        self.channels: list[int] = []

    def play_pcm16_chunks(self, chunks, *, sample_rate: int, channels: int = 1, should_stop=None) -> None:
        self.sample_rates.append(sample_rate)
        self.channels.append(channels)
        for chunk in chunks:
            self.pcm_chunks.append(bytes(chunk))
            if len(self.pcm_chunks) >= 3:
                self.pcm_started.set()
            if should_stop is not None and should_stop():
                return
        self.pcm_started.set()


class _ImmediateLoopingFeedbackPlayer(_FakeFeedbackPlayer):
    def __init__(self) -> None:
        super().__init__()
        self.looped_twice = Event()

    def play_wav_chunks(self, chunks, *, should_stop=None) -> None:
        del should_stop
        payload = bytearray()
        for chunk in chunks:
            payload.extend(chunk)
        self.wav_payloads.append(bytes(payload))
        self.wav_started.set()
        if len(self.wav_payloads) >= 2:
            self.looped_twice.set()


def _constant_pcm16_bytes(*, duration_s: float = 3.0, sample_rate: int = 24000, amplitude: int = 8000) -> bytes:
    pcm_frame = int(amplitude).to_bytes(2, byteorder="little", signed=True)
    return pcm_frame * int(round(sample_rate * duration_s))


def _constant_pcm16_wav_bytes(*, duration_s: float = 3.0, sample_rate: int = 24000, amplitude: int = 8000) -> bytes:
    buffer = io.BytesIO()
    # pylint: disable=no-member
    with cast(wave.Wave_write, wave.open(buffer, "wb")) as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(
            _constant_pcm16_bytes(
                duration_s=duration_s,
                sample_rate=sample_rate,
                amplitude=amplitude,
            )
        )
    return buffer.getvalue()


def _write_placeholder_processing_asset(temp_dir: str) -> None:
    asset_path = Path(temp_dir) / _PROCESSING_ASSET_RELATIVE_PATH
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    asset_path.write_bytes(b"ID3-DRAGON")


def _build_processing_media_spec() -> WorkingFeedbackMediaSpec:
    return WorkingFeedbackMediaSpec(
        clip=RenderedAudioClipSpec(
            relative_path=_PROCESSING_ASSET_RELATIVE_PATH,
            clip_start_s=0.0,
            clip_duration_s=0.8,
            fade_in_duration_s=0.09,
            fade_out_start_s=1.08,
            fade_out_duration_s=0.15,
            output_gain=0.105,
            playback_speed=0.65,
            normalize_max_gain=1.0,
        ),
        pause_ms=0,
        attenuation_start_s=4.0,
        attenuation_reach_floor_s=30.0,
        minimum_output_gain_ratio=0.15,
        attenuation_step_ms=160,
    )


class WorkingFeedbackTests(unittest.TestCase):
    def setUp(self) -> None:
        with working_feedback_mod._RENDERED_MEDIA_CACHE_LOCK:
            working_feedback_mod._RENDERED_MEDIA_CACHE.clear()
            working_feedback_mod._RENDERED_MEDIA_CACHE_FINALIZERS.clear()
        with working_feedback_mod._RENDERED_MEDIA_PREWARM_LOCK:
            working_feedback_mod._RENDERED_MEDIA_PREWARM.clear()

    def test_processing_feedback_defaults_to_dragon_media_clip_when_available(self) -> None:
        player = _FakeFeedbackPlayer()

        with tempfile.TemporaryDirectory() as temp_dir:
            _write_placeholder_processing_asset(temp_dir)
            stop = None
            with mock.patch(
                "twinr.agent.workflows.working_feedback.build_rendered_audio_clip_wav_bytes",
                return_value=b"RIFFDRAGON",
            ) as render_mock:
                stop = start_working_feedback_loop(
                    player,
                    kind="processing",
                    sample_rate=24000,
                    config=TwinrConfig(project_root=temp_dir),
                    delay_override_ms=0,
                )
                self.assertTrue(player.wav_started.wait(timeout=1.0))
            assert stop is not None
            stop()

        self.assertTrue(player.wav_payloads)
        self.assertEqual(player.wav_payloads, [b"RIFFDRAGON"])
        self.assertEqual(player.tone_sequences, [])
        rendered_spec = render_mock.call_args.args[1]
        media_spec = _build_processing_media_spec()
        self.assertEqual(rendered_spec.relative_path, media_spec.clip.relative_path)
        self.assertEqual(rendered_spec.clip_start_s, media_spec.clip.clip_start_s)
        self.assertEqual(rendered_spec.clip_duration_s, media_spec.clip.clip_duration_s)
        self.assertEqual(rendered_spec.fade_in_duration_s, media_spec.clip.fade_in_duration_s)
        self.assertEqual(rendered_spec.fade_out_start_s, media_spec.clip.fade_out_start_s)
        self.assertEqual(rendered_spec.fade_out_duration_s, media_spec.clip.fade_out_duration_s)
        self.assertEqual(rendered_spec.output_gain, media_spec.clip.output_gain)
        self.assertEqual(rendered_spec.playback_speed, media_spec.clip.playback_speed)

    def test_processing_feedback_falls_back_to_swelling_tone_when_default_media_is_missing(self) -> None:
        player = _PCMFeedbackPlayer()

        with mock.patch(
            "twinr.agent.workflows.working_feedback.build_rendered_audio_clip_wav_bytes",
            return_value=None,
        ) as render_mock:
            stop = start_working_feedback_loop(
                player,
                kind="processing",
                sample_rate=24000,
                config=TwinrConfig(project_root="."),
                delay_override_ms=0,
            )
            try:
                self.assertTrue(player.pcm_started.wait(timeout=1.0))
            finally:
                stop()

        render_mock.assert_called_once()
        self.assertEqual(player.sample_rates, [24000])
        self.assertEqual(player.channels, [1])
        self.assertGreaterEqual(len(player.pcm_chunks), 3)

    def test_processing_feedback_falls_back_to_tones_when_media_render_fails(self) -> None:
        player = _ToneOnlyFeedbackPlayer()
        lines: list[str] = []
        profile = WorkingFeedbackProfile(
            delay_ms=0,
            pause_ms=1000,
            volume=0.1,
            gap_ms=0,
            patterns=(((440, 40),),),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            _write_placeholder_processing_asset(temp_dir)
            stop = None
            with (
                mock.patch(
                    "twinr.agent.workflows.working_feedback._resolve_media_spec",
                    return_value=_build_processing_media_spec(),
                ),
                mock.patch(
                    "twinr.agent.workflows.working_feedback.build_rendered_audio_clip_wav_bytes",
                    side_effect=RenderedAudioClipConfigurationError("ffmpeg_missing"),
                ),
            ):
                stop = start_working_feedback_loop(
                    player,
                    kind="processing",
                    sample_rate=24000,
                    config=TwinrConfig(project_root=temp_dir),
                    profiles={"processing": profile},
                    delay_override_ms=0,
                    emit=lines.append,
                )
                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline and not player.tone_sequences:
                    time.sleep(0.01)
            assert stop is not None
            stop()

        self.assertEqual(player.tone_sequences, [((440, 40),)])
        self.assertIn(
            "working_feedback_media_fallback=processing:RenderedAudioClipConfigurationError",
            lines,
        )

    def test_processing_feedback_falls_back_to_tones_when_processing_media_is_missing(self) -> None:
        player = _ToneOnlyFeedbackPlayer()
        lines: list[str] = []
        profile = WorkingFeedbackProfile(
            delay_ms=0,
            pause_ms=1000,
            volume=0.1,
            gap_ms=0,
            patterns=(((440, 40),),),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            stop = None
            with (
                mock.patch(
                    "twinr.agent.workflows.working_feedback._resolve_media_spec",
                    return_value=_build_processing_media_spec(),
                ),
                mock.patch(
                    "twinr.agent.workflows.working_feedback.build_rendered_audio_clip_wav_bytes",
                    return_value=None,
                ),
            ):
                stop = start_working_feedback_loop(
                    player,
                    kind="processing",
                    sample_rate=24000,
                    config=TwinrConfig(project_root=temp_dir),
                    profiles={"processing": profile},
                    delay_override_ms=0,
                    emit=lines.append,
                )
                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline and not player.tone_sequences:
                    time.sleep(0.01)
            assert stop is not None
            stop()

        self.assertEqual(player.tone_sequences, [((440, 40),)])
        self.assertIn("working_feedback_media_fallback=processing:missing_asset", lines)

    def test_processing_feedback_default_tone_volume_is_reduced(self) -> None:
        player = _ToneOnlyFeedbackPlayer()

        with tempfile.TemporaryDirectory() as temp_dir:
            stop = None
            with mock.patch(
                "twinr.agent.workflows.working_feedback.build_rendered_audio_clip_wav_bytes",
                return_value=None,
            ):
                stop = start_working_feedback_loop(
                    player,
                    kind="processing",
                    sample_rate=24000,
                    config=TwinrConfig(project_root=temp_dir),
                    delay_override_ms=0,
                )
                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline and not player.tone_volumes:
                    time.sleep(0.01)
            assert stop is not None
            stop()

        self.assertEqual(player.tone_volumes, [0.065])

    def test_processing_feedback_stop_suppresses_expected_interrupt_error(self) -> None:
        player = _InterruptingFeedbackPlayer()
        lines: list[str] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            _write_placeholder_processing_asset(temp_dir)
            with (
                mock.patch(
                    "twinr.agent.workflows.working_feedback._resolve_media_spec",
                    return_value=_build_processing_media_spec(),
                ),
                mock.patch(
                    "twinr.agent.workflows.working_feedback.build_rendered_audio_clip_wav_bytes",
                    return_value=b"RIFFDRAGON",
                ),
            ):
                stop = start_working_feedback_loop(
                    player,
                    kind="processing",
                    sample_rate=24000,
                    config=TwinrConfig(project_root=temp_dir),
                    delay_override_ms=0,
                    emit=lines.append,
                )
                self.assertTrue(player.wav_started.wait(timeout=1.0))
                stop()

        self.assertNotIn("working_feedback_error=processing:RuntimeError", lines)

    def test_processing_feedback_zero_pause_restarts_media_clip_without_forced_gap(self) -> None:
        player = _ImmediateLoopingFeedbackPlayer()

        with tempfile.TemporaryDirectory() as temp_dir:
            _write_placeholder_processing_asset(temp_dir)
            with (
                mock.patch(
                    "twinr.agent.workflows.working_feedback._resolve_media_spec",
                    return_value=_build_processing_media_spec(),
                ),
                mock.patch(
                    "twinr.agent.workflows.working_feedback.build_rendered_audio_clip_wav_bytes",
                    return_value=b"RIFFDRAGON",
                ),
            ):
                stop = start_working_feedback_loop(
                    player,
                    kind="processing",
                    sample_rate=24000,
                    config=TwinrConfig(project_root=temp_dir),
                    delay_override_ms=0,
                )
                try:
                    self.assertTrue(player.looped_twice.wait(timeout=0.04))
                finally:
                    stop()

        self.assertGreaterEqual(len(player.wav_payloads), 2)

    def test_processing_feedback_zero_pause_pcm_path_uses_one_long_lived_playback_request(self) -> None:
        player = _PCMFeedbackPlayer()

        with tempfile.TemporaryDirectory() as temp_dir:
            _write_placeholder_processing_asset(temp_dir)
            with (
                mock.patch(
                    "twinr.agent.workflows.working_feedback._resolve_media_spec",
                    return_value=_build_processing_media_spec(),
                ),
                mock.patch(
                    "twinr.agent.workflows.working_feedback.build_rendered_audio_clip_wav_bytes",
                    return_value=_constant_pcm16_wav_bytes(duration_s=0.05),
                ),
            ):
                stop = start_working_feedback_loop(
                    player,
                    kind="processing",
                    sample_rate=24000,
                    config=TwinrConfig(project_root=temp_dir),
                    delay_override_ms=0,
                )
                try:
                    self.assertTrue(player.pcm_started.wait(timeout=1.0))
                finally:
                    stop()

        self.assertEqual(player.sample_rates, [24000])
        self.assertEqual(player.channels, [1])
        self.assertGreaterEqual(len(player.pcm_chunks), 3)

    def test_processing_feedback_long_think_pcm_path_quiets_each_second(self) -> None:
        player = _PCMFeedbackPlayer()
        profile = WorkingFeedbackProfile(
            delay_ms=0,
            pause_ms=1000,
            volume=0.1,
            gap_ms=0,
            patterns=(((440, 40),),),
        )
        media_spec = WorkingFeedbackMediaSpec(
            clip=RenderedAudioClipSpec(
                relative_path=_PROCESSING_ASSET_RELATIVE_PATH,
                clip_start_s=0.0,
                clip_duration_s=0.8,
                fade_in_duration_s=0.07,
                fade_out_start_s=0.8,
                fade_out_duration_s=0.14,
                output_gain=0.3,
            ),
            pause_ms=1000,
            attenuation_start_s=0.0,
            attenuation_reach_floor_s=2.0,
            minimum_output_gain=0.15,
            attenuation_step_ms=1000,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            _write_placeholder_processing_asset(temp_dir)
            with mock.patch(
                "twinr.agent.workflows.working_feedback._resolve_media_spec",
                return_value=media_spec,
            ):
                with mock.patch(
                    "twinr.agent.workflows.working_feedback.build_rendered_audio_clip_wav_bytes",
                    return_value=_constant_pcm16_wav_bytes(),
                ):
                    stop = start_working_feedback_loop(
                        player,
                        kind="processing",
                        sample_rate=24000,
                        config=TwinrConfig(project_root=temp_dir),
                        profiles={"processing": profile},
                        delay_override_ms=0,
                    )
                    self.assertTrue(player.pcm_started.wait(timeout=1.0))
                    stop()

        self.assertGreaterEqual(len(player.pcm_chunks), 3)
        first_three_rms = [audioop.rms(chunk, 2) for chunk in player.pcm_chunks[:3]]
        self.assertGreater(first_three_rms[0], first_three_rms[1])
        self.assertGreater(first_three_rms[1], first_three_rms[2])
        self.assertEqual(player.sample_rates, [24000])
        self.assertEqual(player.channels, [1])

    def test_processing_feedback_start_stays_non_blocking_while_media_warms_in_background(self) -> None:
        player = _ToneOnlyFeedbackPlayer()
        profile = WorkingFeedbackProfile(
            delay_ms=0,
            pause_ms=1000,
            volume=0.1,
            gap_ms=0,
            patterns=(((440, 40),),),
        )
        render_started = Event()
        release_render = Event()

        def slow_render(*_args, **_kwargs) -> bytes:
            render_started.set()
            release_render.wait(timeout=1.0)
            return b"RIFFDRAGON"

        with tempfile.TemporaryDirectory() as temp_dir:
            _write_placeholder_processing_asset(temp_dir)
            with (
                mock.patch(
                    "twinr.agent.workflows.working_feedback._resolve_media_spec",
                    return_value=_build_processing_media_spec(),
                ),
                mock.patch(
                    "twinr.agent.workflows.working_feedback.build_rendered_audio_clip_wav_bytes",
                    side_effect=slow_render,
                ),
            ):
                started_at = time.monotonic()
                stop = start_working_feedback_loop(
                    player,
                    kind="processing",
                    sample_rate=24000,
                    config=TwinrConfig(project_root=temp_dir),
                    profiles={"processing": profile},
                    delay_override_ms=0,
                )
                try:
                    self.assertLess(time.monotonic() - started_at, 0.1)
                    self.assertTrue(render_started.wait(timeout=0.2))
                    deadline = time.monotonic() + 0.5
                    while time.monotonic() < deadline and not player.tone_sequences:
                        time.sleep(0.01)
                    self.assertEqual(player.tone_sequences, [((440, 40),)])
                finally:
                    release_render.set()
                    stop()
