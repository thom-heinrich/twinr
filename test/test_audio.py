from pathlib import Path
from tempfile import TemporaryFile
from unittest import mock
import io
import os
import sys
import unittest
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.audio_env import build_audio_subprocess_env
from twinr.hardware.audio import (
    AmbientAudioSampler,
    AudioCaptureReadinessError,
    SilenceDetectedRecorder,
    SpeechStartTimeoutError,
    WaveAudioPlayer,
    normalize_wav_playback_level,
    resolve_dynamic_pause_thresholds,
    resolve_pause_resume_confirmation,
)


class DynamicPauseThresholdTests(unittest.TestCase):
    def test_single_resume_spike_does_not_confirm_resume(self) -> None:
        confirmed, consecutive_chunks = resolve_pause_resume_confirmation(
            consecutive_resume_chunks=0,
            required_resume_chunks=2,
        )

        self.assertFalse(confirmed)
        self.assertEqual(consecutive_chunks, 1)

    def test_second_resume_chunk_confirms_resume(self) -> None:
        confirmed, consecutive_chunks = resolve_pause_resume_confirmation(
            consecutive_resume_chunks=1,
            required_resume_chunks=2,
        )

        self.assertTrue(confirmed)
        self.assertEqual(consecutive_chunks, 0)

    def test_short_utterance_gets_extra_tail(self) -> None:
        pause_ms, pause_grace_ms = resolve_dynamic_pause_thresholds(
            base_pause_ms=1200,
            base_pause_grace_ms=450,
            speech_window_ms=800,
            enabled=True,
            short_utterance_max_ms=1000,
            long_utterance_min_ms=5000,
            short_pause_bonus_ms=120,
            short_pause_grace_bonus_ms=0,
            medium_pause_penalty_ms=120,
            medium_pause_grace_penalty_ms=250,
            long_pause_penalty_ms=320,
            long_pause_grace_penalty_ms=220,
        )

        self.assertEqual((pause_ms, pause_grace_ms), (1320, 450))

    def test_medium_utterance_keeps_baseline(self) -> None:
        pause_ms, pause_grace_ms = resolve_dynamic_pause_thresholds(
            base_pause_ms=1200,
            base_pause_grace_ms=450,
            speech_window_ms=2400,
            enabled=True,
            short_utterance_max_ms=1000,
            long_utterance_min_ms=5000,
            short_pause_bonus_ms=120,
            short_pause_grace_bonus_ms=0,
            medium_pause_penalty_ms=120,
            medium_pause_grace_penalty_ms=250,
            long_pause_penalty_ms=320,
            long_pause_grace_penalty_ms=220,
        )

        self.assertEqual((pause_ms, pause_grace_ms), (1080, 200))

    def test_long_utterance_ends_sooner(self) -> None:
        pause_ms, pause_grace_ms = resolve_dynamic_pause_thresholds(
            base_pause_ms=1200,
            base_pause_grace_ms=450,
            speech_window_ms=6400,
            enabled=True,
            short_utterance_max_ms=1000,
            long_utterance_min_ms=5000,
            short_pause_bonus_ms=120,
            short_pause_grace_bonus_ms=0,
            medium_pause_penalty_ms=120,
            medium_pause_grace_penalty_ms=250,
            long_pause_penalty_ms=320,
            long_pause_grace_penalty_ms=220,
        )

        self.assertEqual((pause_ms, pause_grace_ms), (880, 230))


class AudioEnvTests(unittest.TestCase):
    def test_build_audio_subprocess_env_strips_display_vars_and_mismatched_session_audio(self) -> None:
        base_env = {
            "XDG_RUNTIME_DIR": "/run/user/1000",
            "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1000/bus",
            "PULSE_SERVER": "unix:/run/user/1000/pulse/native",
            "WAYLAND_DISPLAY": "wayland-0",
            "QT_QPA_PLATFORM": "wayland",
            "SDL_VIDEODRIVER": "wayland",
            "KEEP_ME": "1",
        }
        with (
            mock.patch("twinr.hardware.audio_env.os.getuid", return_value=0),
            mock.patch("twinr.hardware.audio_env.runtime_dir_owner_uid", return_value=1000),
        ):
            env = build_audio_subprocess_env(base_env)

        self.assertNotIn("XDG_RUNTIME_DIR", env)
        self.assertNotIn("DBUS_SESSION_BUS_ADDRESS", env)
        self.assertNotIn("PULSE_SERVER", env)
        self.assertNotIn("WAYLAND_DISPLAY", env)
        self.assertNotIn("QT_QPA_PLATFORM", env)
        self.assertNotIn("SDL_VIDEODRIVER", env)
        self.assertEqual(env["KEEP_ME"], "1")

    def test_build_audio_subprocess_env_keeps_same_owner_session_audio(self) -> None:
        base_env = {
            "XDG_RUNTIME_DIR": "/run/user/1000",
            "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1000/bus",
            "PULSE_SERVER": "unix:/run/user/1000/pulse/native",
            "WAYLAND_DISPLAY": "wayland-0",
            "KEEP_ME": "1",
        }
        with (
            mock.patch("twinr.hardware.audio_env.os.getuid", return_value=1000),
            mock.patch("twinr.hardware.audio_env.runtime_dir_owner_uid", return_value=1000),
        ):
            env = build_audio_subprocess_env(base_env)

        self.assertEqual(env["XDG_RUNTIME_DIR"], "/run/user/1000")
        self.assertEqual(env["DBUS_SESSION_BUS_ADDRESS"], "unix:path=/run/user/1000/bus")
        self.assertEqual(env["PULSE_SERVER"], "unix:/run/user/1000/pulse/native")
        self.assertNotIn("WAYLAND_DISPLAY", env)
        self.assertEqual(env["KEEP_ME"], "1")

    def test_spawn_audio_process_uses_sanitized_env(self) -> None:
        fake_process = _FakeCaptureProcess()
        with (
            mock.patch("twinr.hardware.audio.build_audio_subprocess_env", return_value={"KEEP_ME": "1"}) as env_builder,
            mock.patch("twinr.hardware.audio.subprocess.Popen", return_value=fake_process) as popen,
        ):
            process = __import__("twinr.hardware.audio", fromlist=["_spawn_audio_process"])._spawn_audio_process(
                ["arecord", "-D", "default"],
                stdout=-1,
                stderr=-1,
                purpose="Audio capture",
            )

        self.assertIs(process, fake_process)
        env_builder.assert_called_once_with()
        self.assertEqual(popen.call_args.kwargs["env"], {"KEEP_ME": "1"})


class _FakePlaybackProcess:
    def __init__(self) -> None:
        self.stdin = TemporaryFile()
        self.stderr = TemporaryFile()
        self.stdout = None
        self.returncode = None
        self.terminate_calls = 0
        self.wait_calls = 0
        self.wait_timeouts: list[float | None] = []

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.wait_calls += 1
        self.wait_timeouts.append(timeout)
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


class _FakeReadableStream:
    def fileno(self) -> int:
        return 0


class _FakeCaptureProcess:
    def __init__(self) -> None:
        self.stdin = None
        self.stdout = _FakeReadableStream()
        self.stderr = TemporaryFile()
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        del timeout
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


class WaveAudioPlayerTests(unittest.TestCase):
    def test_normalize_wav_playback_level_boosts_quiet_pcm16_wav(self) -> None:
        frames = (b"\x10\x00" + b"\xf0\xff") * 800
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(24000)
            writer.writeframes(frames)

        normalized = normalize_wav_playback_level(buffer.getvalue(), target_peak=20000, max_gain=4.0)

        with wave.open(io.BytesIO(normalized), "rb") as reader:
            boosted_frames = reader.readframes(reader.getnframes())
        self.assertGreater(len(boosted_frames), 0)
        self.assertGreater(abs(int.from_bytes(boosted_frames[:2], "little", signed=True)), 0x10)

    def test_normalize_wav_playback_level_keeps_invalid_payload_unchanged(self) -> None:
        self.assertEqual(normalize_wav_playback_level(b"WAVPCM"), b"WAVPCM")

    def test_preempted_stream_terminates_aplay_without_waiting_for_drain(self) -> None:
        player = WaveAudioPlayer(device="default")
        process = _FakePlaybackProcess()
        stop_requested = [False]
        original_os_write = os.write

        def fake_write(fd, view):
            written = original_os_write(fd, view)
            stop_requested[0] = True
            return written

        with (
            mock.patch("twinr.hardware.audio._spawn_audio_process", return_value=process),
            mock.patch("twinr.hardware.audio._wait_for_writable", return_value=True),
            mock.patch("twinr.hardware.audio.os.set_blocking"),
            mock.patch("twinr.hardware.audio.os.write", side_effect=fake_write),
        ):
            player.play_wav_chunks(
                [b"chunk-1", b"chunk-2"],
                should_stop=lambda: stop_requested[0],
            )

        self.assertGreaterEqual(process.terminate_calls, 1)
        self.assertEqual(process.wait_timeouts, [1.0])

    def test_stop_playback_terminates_active_process(self) -> None:
        player = WaveAudioPlayer(device="default")
        process = _FakePlaybackProcess()

        player._set_active_process(process)
        player.stop_playback()

        self.assertGreaterEqual(process.terminate_calls, 1)

    def test_stream_error_tolerates_closed_stderr_after_stop(self) -> None:
        player = WaveAudioPlayer(device="default")
        process = _FakePlaybackProcess()

        player._set_active_process(process)
        player.stop_playback()

        with self.assertRaisesRegex(RuntimeError, "Audio playback failed: exit code -15"):
            player._raise_stream_error(process)


class SilenceDetectedRecorderTests(unittest.TestCase):
    def test_start_timeout_carries_pre_speech_capture_diagnostics(self) -> None:
        recorder = SilenceDetectedRecorder(
            device="default",
            sample_rate=16000,
            channels=1,
            chunk_ms=100,
            speech_threshold=700,
            speech_start_chunks=3,
            start_timeout_s=0.18,
        )
        process = _FakeCaptureProcess()
        quiet_chunk = b"\x00\x00" * 1600

        def fake_wait_for_readable(*_args, **_kwargs):
            fake_wait_for_readable.calls += 1
            return fake_wait_for_readable.calls <= 2

        fake_wait_for_readable.calls = 0

        class _AdvancingMonotonic:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                current = self.value
                self.value += 0.05
                return current

        monotonic = _AdvancingMonotonic()

        with (
            mock.patch("twinr.hardware.audio._spawn_audio_process", return_value=process),
            mock.patch("twinr.hardware.audio._wait_for_readable", side_effect=fake_wait_for_readable),
            mock.patch("twinr.hardware.audio.os.read", side_effect=[quiet_chunk, quiet_chunk]),
            mock.patch("twinr.hardware.audio.time.monotonic", side_effect=monotonic),
        ):
            with self.assertRaises(SpeechStartTimeoutError) as captured:
                recorder.capture_pcm_until_pause_with_options(pause_ms=1200)

        diagnostics = captured.exception.diagnostics
        self.assertEqual(diagnostics.device, "default")
        self.assertGreaterEqual(diagnostics.chunk_count, 1)
        self.assertEqual(diagnostics.active_chunk_count, 0)
        self.assertEqual(diagnostics.average_rms, 0)
        self.assertEqual(diagnostics.peak_rms, 0)
        self.assertGreaterEqual(diagnostics.listened_ms, 150)


class AmbientAudioSamplerTests(unittest.TestCase):
    def test_require_readable_frames_returns_probe_after_first_chunk(self) -> None:
        sampler = AmbientAudioSampler(
            device="default",
            sample_rate=16000,
            channels=1,
            chunk_ms=100,
            default_duration_ms=200,
        )
        process = _FakeCaptureProcess()
        pcm_chunk = b"\x01\x00" * 1600

        with (
            mock.patch("twinr.hardware.audio._spawn_audio_process", return_value=process),
            mock.patch("twinr.hardware.audio._wait_for_readable", return_value=True),
            mock.patch("twinr.hardware.audio.os.read", return_value=pcm_chunk),
        ):
            probe = sampler.require_readable_frames(duration_ms=200)

        self.assertTrue(probe.ready)
        self.assertEqual(probe.captured_chunk_count, 1)
        self.assertEqual(probe.captured_bytes, len(pcm_chunk))
        self.assertIsNone(probe.failure_reason)

    def test_sample_window_raises_readiness_error_when_capture_stalls(self) -> None:
        sampler = AmbientAudioSampler(
            device="default",
            sample_rate=16000,
            channels=1,
            chunk_ms=100,
            default_duration_ms=200,
        )
        process = _FakeCaptureProcess()

        class _AdvancingMonotonic:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                current = self.value
                self.value += 1.1
                return current

        with (
            mock.patch("twinr.hardware.audio._spawn_audio_process", return_value=process),
            mock.patch("twinr.hardware.audio._wait_for_readable", return_value=False),
            mock.patch("twinr.hardware.audio.time.monotonic", side_effect=_AdvancingMonotonic()),
        ):
            with self.assertRaises(AudioCaptureReadinessError) as captured:
                sampler.sample_window(duration_ms=200)

        self.assertEqual(captured.exception.probe.failure_reason, "stalled_waiting")
        self.assertIn("waiting for microphone data", str(captured.exception))


if __name__ == "__main__":
    unittest.main()
