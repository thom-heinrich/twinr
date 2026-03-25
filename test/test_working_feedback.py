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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.agent.workflows.rendered_audio_clip import RenderedAudioClipSpec
from twinr.agent.workflows.working_feedback import (
    WorkingFeedbackMediaSpec,
    WorkingFeedbackProfile,
    start_working_feedback_loop,
)


class _FakeFeedbackPlayer:
    def __init__(self) -> None:
        self.wav_payloads: list[bytes] = []
        self.tone_sequences: list[tuple[tuple[int, int], ...]] = []
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
        del volume, sample_rate, gap_ms
        self.tone_sequences.append(tuple(sequence))

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


def _constant_wav_bytes(*, duration_s: int = 3, sample_rate: int = 24000, amplitude: int = 8000) -> bytes:
    buffer = io.BytesIO()
    pcm_frame = int(amplitude).to_bytes(2, byteorder="little", signed=True)
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm_frame * sample_rate * duration_s)
    return buffer.getvalue()


class WorkingFeedbackTests(unittest.TestCase):
    def test_processing_feedback_prefers_waiting_media_clip_when_available(self) -> None:
        player = _FakeFeedbackPlayer()
        profile = WorkingFeedbackProfile(
            delay_ms=0,
            pause_ms=1000,
            volume=0.1,
            gap_ms=0,
            patterns=(((440, 40),),),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            stop = None
            with mock.patch(
                "twinr.agent.workflows.working_feedback.build_rendered_audio_clip_wav_bytes",
                return_value=b"RIFFWAITING",
            ):
                stop = start_working_feedback_loop(
                    player,
                    kind="processing",
                    sample_rate=24000,
                    config=TwinrConfig(project_root=temp_dir),
                    profiles={"processing": profile},
                    delay_override_ms=0,
                )
                self.assertTrue(player.wav_started.wait(timeout=1.0))
            assert stop is not None
            stop()

        self.assertEqual(player.wav_payloads, [b"RIFFWAITING"])
        self.assertEqual(player.tone_sequences, [])

    def test_processing_feedback_falls_back_to_tones_when_waiting_media_is_missing(self) -> None:
        player = _FakeFeedbackPlayer()
        profile = WorkingFeedbackProfile(
            delay_ms=0,
            pause_ms=1000,
            volume=0.1,
            gap_ms=0,
            patterns=(((440, 40),),),
        )
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
                    profiles={"processing": profile},
                    delay_override_ms=0,
                )
                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline and not player.tone_sequences:
                    time.sleep(0.01)
            assert stop is not None
            stop()

        self.assertEqual(player.wav_payloads, [])
        self.assertEqual(player.tone_sequences, [((440, 40),)])

    def test_processing_feedback_stop_suppresses_expected_interrupt_error(self) -> None:
        player = _InterruptingFeedbackPlayer()
        lines: list[str] = []
        profile = WorkingFeedbackProfile(
            delay_ms=0,
            pause_ms=1000,
            volume=0.1,
            gap_ms=0,
            patterns=(((440, 40),),),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch(
                "twinr.agent.workflows.working_feedback.build_rendered_audio_clip_wav_bytes",
                return_value=b"RIFFWAITING",
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
                self.assertTrue(player.wav_started.wait(timeout=1.0))
                stop()

        self.assertNotIn("working_feedback_error=processing:RuntimeError", lines)

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
                relative_path=Path("media") / "waiting.mp3",
                clip_start_s=3.0,
                clip_duration_s=3.0,
                fade_in_duration_s=0.9,
                fade_out_start_s=2.75,
                fade_out_duration_s=0.25,
                output_gain=0.3,
            ),
            pause_ms=1000,
            attenuation_start_s=0.0,
            attenuation_reach_floor_s=2.0,
            minimum_output_gain=0.15,
            attenuation_step_ms=1000,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch(
                "twinr.agent.workflows.working_feedback._resolve_media_spec",
                return_value=media_spec,
            ):
                with mock.patch(
                    "twinr.agent.workflows.working_feedback.build_rendered_audio_clip_wav_bytes",
                    return_value=_constant_wav_bytes(),
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
