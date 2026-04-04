from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryFile
from unittest import mock
import io
import itertools
import os
import sys
import unittest
import wave
from typing import BinaryIO, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.audio_env import (
    build_audio_subprocess_env,
    build_audio_subprocess_env_for_mode,
)
from twinr.hardware import respeaker_duplex_playback
from twinr.hardware.audio import (
    AmbientAudioSampler,
    AudioCaptureReadinessError,
    SilenceDetectedRecorder,
    SpeechStartTimeoutError,
    WaveAudioPlayer,
    capture_device_identity,
    normalize_wav_playback_level,
    resolve_capture_device,
    resolve_dynamic_pause_thresholds,
    resolve_pause_resume_confirmation,
)
from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.respeaker_capture_recovery import recover_stalled_respeaker_capture


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


class CaptureDeviceResolutionTests(unittest.TestCase):
    def test_resolve_capture_device_prefers_specific_fallback_over_generic_alias(self) -> None:
        resolved = resolve_capture_device(
            "default",
            None,
            "sysdefault:CARD=Array",
        )

        self.assertEqual(resolved, "sysdefault:CARD=Array")

    def test_capture_device_identity_normalizes_same_card_aliases(self) -> None:
        identity_a = capture_device_identity("sysdefault:CARD=Array")
        identity_b = capture_device_identity("plughw:CARD=Array,DEV=0")

        self.assertEqual(identity_a, identity_b)


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

    def test_build_audio_subprocess_env_for_mode_keeps_root_borrowed_session_audio(self) -> None:
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
            env = build_audio_subprocess_env_for_mode(
                base_env,
                allow_root_borrowed_session_audio=True,
            )

        self.assertEqual(env["XDG_RUNTIME_DIR"], "/run/user/1000")
        self.assertEqual(env["DBUS_SESSION_BUS_ADDRESS"], "unix:path=/run/user/1000/bus")
        self.assertEqual(env["PULSE_SERVER"], "unix:/run/user/1000/pulse/native")
        self.assertNotIn("WAYLAND_DISPLAY", env)
        self.assertNotIn("QT_QPA_PLATFORM", env)
        self.assertNotIn("SDL_VIDEODRIVER", env)
        self.assertEqual(env["KEEP_ME"], "1")


class ReSpeakerCaptureRecoveryTests(unittest.TestCase):
    def test_recover_stalled_capture_reboots_after_transient_wait_fails(self) -> None:
        with (
            mock.patch(
                "twinr.hardware.respeaker_capture_recovery.wait_for_transient_respeaker_capture_ready",
                side_effect=[False, True],
            ) as wait_ready,
            mock.patch(
                "twinr.hardware.respeaker_capture_recovery._attempt_respeaker_host_control_reboot",
                return_value=True,
            ) as reboot_capture,
        ):
            recovered = recover_stalled_respeaker_capture(
                device="plughw:CARD=Array,DEV=0",
                sample_rate=16000,
                channels=1,
                chunk_ms=100,
                max_wait_s=2.5,
            )

        self.assertTrue(recovered)
        reboot_capture.assert_called_once_with(device="plughw:CARD=Array,DEV=0")
        self.assertEqual(wait_ready.call_count, 2)
        self.assertGreaterEqual(wait_ready.call_args_list[1].kwargs["max_wait_s"], 8.0)

    def test_recover_stalled_capture_stops_when_reboot_is_unavailable(self) -> None:
        with (
            mock.patch(
                "twinr.hardware.respeaker_capture_recovery.wait_for_transient_respeaker_capture_ready",
                return_value=False,
            ) as wait_ready,
            mock.patch(
                "twinr.hardware.respeaker_capture_recovery._attempt_respeaker_host_control_reboot",
                return_value=False,
            ) as reboot_capture,
        ):
            recovered = recover_stalled_respeaker_capture(
                device="plughw:CARD=Array,DEV=0",
                sample_rate=16000,
                channels=1,
                chunk_ms=100,
            )

        self.assertFalse(recovered)
        reboot_capture.assert_called_once_with(device="plughw:CARD=Array,DEV=0")
        wait_ready.assert_called_once()

    def test_recover_stalled_capture_tries_usb_reset_after_reboot_recovery_fails(self) -> None:
        with (
            mock.patch(
                "twinr.hardware.respeaker_capture_recovery.wait_for_transient_respeaker_capture_ready",
                side_effect=[False, False, True],
            ) as wait_ready,
            mock.patch(
                "twinr.hardware.respeaker_capture_recovery._attempt_respeaker_host_control_reboot",
                return_value=True,
            ) as reboot_capture,
            mock.patch(
                "twinr.hardware.respeaker_capture_recovery._attempt_respeaker_usb_port_reset",
                return_value=True,
            ) as usb_reset,
        ):
            recovered = recover_stalled_respeaker_capture(
                device="plughw:CARD=Array,DEV=0",
                sample_rate=16000,
                channels=1,
                chunk_ms=100,
                max_wait_s=2.5,
            )

        self.assertTrue(recovered)
        reboot_capture.assert_called_once_with(device="plughw:CARD=Array,DEV=0")
        usb_reset.assert_called_once_with(device="plughw:CARD=Array,DEV=0")
        self.assertEqual(wait_ready.call_count, 3)
        self.assertGreaterEqual(wait_ready.call_args_list[1].kwargs["max_wait_s"], 8.0)
        self.assertGreaterEqual(wait_ready.call_args_list[2].kwargs["max_wait_s"], 12.0)

    def test_recover_stalled_capture_fails_when_usb_reset_is_unavailable_after_reboot_path(self) -> None:
        with (
            mock.patch(
                "twinr.hardware.respeaker_capture_recovery.wait_for_transient_respeaker_capture_ready",
                side_effect=[False, False],
            ) as wait_ready,
            mock.patch(
                "twinr.hardware.respeaker_capture_recovery._attempt_respeaker_host_control_reboot",
                return_value=True,
            ) as reboot_capture,
            mock.patch(
                "twinr.hardware.respeaker_capture_recovery._attempt_respeaker_usb_port_reset",
                return_value=False,
            ) as usb_reset,
        ):
            recovered = recover_stalled_respeaker_capture(
                device="plughw:CARD=Array,DEV=0",
                sample_rate=16000,
                channels=1,
                chunk_ms=100,
            )

        self.assertFalse(recovered)
        reboot_capture.assert_called_once_with(device="plughw:CARD=Array,DEV=0")
        usb_reset.assert_called_once_with(device="plughw:CARD=Array,DEV=0")
        self.assertEqual(wait_ready.call_count, 2)

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

    def test_spawn_audio_process_uses_sanitized_env_by_default(self) -> None:
        fake_process = _FakeCaptureProcess()
        with (
            mock.patch(
                "twinr.hardware.audio.build_audio_subprocess_env_for_mode",
                return_value={"KEEP_ME": "1"},
            ) as env_builder,
            mock.patch("twinr.hardware.audio.subprocess.Popen", return_value=fake_process) as popen,
        ):
            process = __import__("twinr.hardware.audio", fromlist=["_spawn_audio_process"])._spawn_audio_process(
                ["aplay", "-D", "default"],
                stdout=-1,
                stderr=-1,
                purpose="Audio playback",
            )

        self.assertIs(process, fake_process)
        env_builder.assert_called_once_with(allow_root_borrowed_session_audio=False)
        self.assertEqual(popen.call_args.kwargs["env"], {"KEEP_ME": "1"})

    def test_spawn_audio_process_can_keep_root_borrowed_session_audio(self) -> None:
        fake_process = _FakeCaptureProcess()
        with (
            mock.patch(
                "twinr.hardware.audio.build_audio_subprocess_env_for_mode",
                return_value={"KEEP_ME": "1"},
            ) as env_builder,
            mock.patch("twinr.hardware.audio.subprocess.Popen", return_value=fake_process),
        ):
            process = __import__("twinr.hardware.audio", fromlist=["_spawn_audio_process"])._spawn_audio_process(
                ["arecord", "-D", "default"],
                stdout=-1,
                stderr=-1,
                purpose="Audio capture",
                allow_root_borrowed_session_audio=True,
            )

        self.assertIs(process, fake_process)
        env_builder.assert_called_once_with(allow_root_borrowed_session_audio=True)

    def test_respeaker_duplex_guard_spawn_stays_in_parent_session(self) -> None:
        fake_process = _FakeCaptureProcess()
        with (
            mock.patch(
                "twinr.hardware.respeaker_duplex_playback._resolve_aplay_executable",
                return_value="/usr/bin/aplay",
            ),
            mock.patch(
                "twinr.hardware.respeaker_duplex_playback._start_stderr_drain_thread",
            ) as start_drain,
            mock.patch(
                "twinr.hardware.respeaker_duplex_playback.subprocess.Popen",
                return_value=fake_process,
            ) as popen,
        ):
            handle = respeaker_duplex_playback._spawn_aplay_handle(
                playback_device="twinr_playback_softvol",
                sample_rate_hz=24000,
                env={"PATH": "/usr/bin"},
            )

        self.assertIs(handle.process, fake_process)
        start_drain.assert_called_once_with(handle)
        self.assertTrue(popen.call_args.kwargs["close_fds"])
        self.assertNotIn("start_new_session", popen.call_args.kwargs)
        self.assertEqual(popen.call_args.kwargs["env"], {"PATH": "/usr/bin"})


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
    def test_from_config_reasserts_respeaker_playback_mixer(self) -> None:
        with mock.patch("twinr.hardware.audio.ensure_respeaker_playback_mixer") as normalize_mixer:
            player = WaveAudioPlayer.from_config(
                TwinrConfig(audio_output_device="twinr_playback_softvol")
            )

        normalize_mixer.assert_called_once_with("twinr_playback_softvol")
        self.assertEqual(player.device, "twinr_playback_softvol")

    def test_normalize_wav_playback_level_boosts_quiet_pcm16_wav(self) -> None:
        frames = (b"\x10\x00" + b"\xf0\xff") * 800
        buffer = io.BytesIO()
        # pylint misclassifies stdlib wave writers from wave.open(..., "wb") as Wave_read.
        # pylint: disable=no-member
        with cast(wave.Wave_write, wave.open(cast(BinaryIO, buffer), "wb")) as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(24000)
            writer.writeframes(frames)
        # pylint: enable=no-member

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

    def test_stream_playback_keeps_root_borrowed_session_audio_for_respeaker_output(self) -> None:
        player = WaveAudioPlayer(device="twinr_playback_softvol")
        process = _FakePlaybackProcess()

        with (
            mock.patch("twinr.hardware.audio._spawn_audio_process", return_value=process) as spawn_process,
            mock.patch("twinr.hardware.audio._wait_for_writable", return_value=True),
            mock.patch("twinr.hardware.audio.os.set_blocking"),
            mock.patch("twinr.hardware.audio.os.write", return_value=5),
        ):
            player.play_wav_chunks([b"chunk"])

        self.assertTrue(spawn_process.call_args.kwargs["allow_root_borrowed_session_audio"])

    def test_stream_playback_keeps_sanitized_env_for_non_respeaker_output(self) -> None:
        player = WaveAudioPlayer(device="default")
        process = _FakePlaybackProcess()

        with (
            mock.patch("twinr.hardware.audio._spawn_audio_process", return_value=process) as spawn_process,
            mock.patch("twinr.hardware.audio._wait_for_writable", return_value=True),
            mock.patch("twinr.hardware.audio.os.set_blocking"),
            mock.patch("twinr.hardware.audio.os.write", return_value=5),
        ):
            player.play_wav_chunks([b"chunk"])

        self.assertFalse(spawn_process.call_args.kwargs["allow_root_borrowed_session_audio"])

    def test_file_playback_keeps_root_borrowed_session_audio_for_respeaker_output(self) -> None:
        player = WaveAudioPlayer(device="twinr_playback_softvol")
        process = _FakePlaybackProcess()

        with NamedTemporaryFile() as handle:
            handle.write(b"wav")
            handle.flush()

            with mock.patch(
                "twinr.hardware.audio._spawn_audio_process",
                return_value=process,
            ) as spawn_process:
                player.play_file(handle.name)

        self.assertTrue(spawn_process.call_args.kwargs["allow_root_borrowed_session_audio"])

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
    def test_capture_uses_respeaker_duplex_guard_for_respeaker_input(self) -> None:
        recorder = SilenceDetectedRecorder(
            device="plughw:CARD=Array,DEV=0",
            sample_rate=16000,
            channels=1,
            chunk_ms=100,
            speech_threshold=700,
            speech_start_chunks=1,
            start_timeout_s=1.0,
            duplex_playback_device="twinr_playback_softvol",
            duplex_playback_sample_rate_hz=24000,
            vad_mode="rms",
            adaptive_noise_enabled=False,
        )
        process = _FakeCaptureProcess()
        speech_chunk = b"\xff\x7f" * 1600
        silence_chunk = b"\x00\x00" * 1600

        class _Guard:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

        class _AdvancingMonotonic:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                current = self.value
                self.value += 0.02
                return current

        with (
            mock.patch("twinr.hardware.audio.maybe_open_respeaker_duplex_playback_guard", return_value=_Guard()) as guard_factory,
            mock.patch("twinr.hardware.audio._spawn_audio_process", return_value=process),
            mock.patch(
                "twinr.hardware.audio._wait_for_readable",
                side_effect=itertools.chain([True, True], itertools.repeat(False)),
            ),
            mock.patch("twinr.hardware.audio.os.read", side_effect=[speech_chunk, silence_chunk]),
            mock.patch("twinr.hardware.audio.time.monotonic", side_effect=_AdvancingMonotonic()),
        ):
            result = recorder.capture_pcm_until_pause_with_options(
                pause_ms=0,
                speech_start_chunks=1,
            )

        guard_factory.assert_called_once_with(
            capture_device="plughw:CARD=Array,DEV=0",
            playback_device="twinr_playback_softvol",
            sample_rate_hz=24000,
        )
        self.assertGreater(len(result.pcm_bytes), 0)

    def test_retries_transient_respeaker_capture_loss_before_speech(self) -> None:
        recorder = SilenceDetectedRecorder(
            device="plughw:CARD=Array,DEV=0",
            sample_rate=16000,
            channels=1,
            chunk_ms=100,
            speech_threshold=700,
            speech_start_chunks=1,
            start_timeout_s=1.0,
            vad_mode="rms",
            adaptive_noise_enabled=False,
        )
        failed_process = _FakeCaptureProcess()
        failed_process.returncode = 1
        recovered_process = _FakeCaptureProcess()
        speech_chunk = b"\xff\x7f" * 1600
        silence_chunk = b"\x00\x00" * 1600

        class _AdvancingMonotonic:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                current = self.value
                self.value += 0.02
                return current

        with (
            mock.patch(
                "twinr.hardware.audio._spawn_audio_process",
                side_effect=[failed_process, recovered_process],
            ) as spawn_process,
            mock.patch(
                "twinr.hardware.respeaker_capture_recovery.wait_for_transient_respeaker_capture_ready",
                return_value=True,
            ) as recover_capture,
            mock.patch(
                "twinr.hardware.audio._wait_for_readable",
                side_effect=itertools.chain([True, True], itertools.repeat(False)),
            ),
            mock.patch("twinr.hardware.audio.os.read", side_effect=[speech_chunk, silence_chunk]),
            mock.patch("twinr.hardware.audio.time.monotonic", side_effect=_AdvancingMonotonic()),
        ):
            result = recorder.capture_pcm_until_pause_with_options(
                pause_ms=0,
                speech_start_chunks=1,
            )

        recover_capture.assert_called_once()
        self.assertEqual(spawn_process.call_count, 2)
        self.assertGreater(len(result.pcm_bytes), 0)

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
    def test_require_readable_frames_uses_respeaker_duplex_guard_for_respeaker_input(self) -> None:
        sampler = AmbientAudioSampler(
            device="plughw:CARD=Array,DEV=0",
            sample_rate=16000,
            channels=1,
            chunk_ms=100,
            default_duration_ms=200,
            duplex_playback_device="twinr_playback_softvol",
            duplex_playback_sample_rate_hz=24000,
        )
        process = _FakeCaptureProcess()
        pcm_chunk = b"\x01\x00" * 1600

        class _Guard:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

        with (
            mock.patch("twinr.hardware.audio.maybe_open_respeaker_duplex_playback_guard", return_value=_Guard()) as guard_factory,
            mock.patch("twinr.hardware.audio._spawn_audio_process", return_value=process),
            mock.patch("twinr.hardware.audio._wait_for_readable", return_value=True),
            mock.patch("twinr.hardware.audio.os.read", return_value=pcm_chunk),
        ):
            probe = sampler.require_readable_frames(duration_ms=200)

        guard_factory.assert_called_once_with(
            capture_device="plughw:CARD=Array,DEV=0",
            playback_device="twinr_playback_softvol",
            sample_rate_hz=24000,
        )
        self.assertTrue(probe.ready)

    def test_sample_window_surfaces_duplex_guard_failure_as_readiness_error(self) -> None:
        sampler = AmbientAudioSampler(
            device="plughw:CARD=Array,DEV=0",
            sample_rate=16000,
            channels=1,
            chunk_ms=100,
            default_duration_ms=200,
            duplex_playback_device="twinr_playback_softvol",
            duplex_playback_sample_rate_hz=24000,
        )

        with mock.patch(
            "twinr.hardware.audio.maybe_open_respeaker_duplex_playback_guard",
            side_effect=RuntimeError("Required ReSpeaker duplex playback guard exited immediately"),
        ):
            with self.assertRaises(AudioCaptureReadinessError) as captured:
                sampler.sample_window(duration_ms=200)

        self.assertEqual(captured.exception.probe.failure_reason, "duplex_playback_failed")
        self.assertIn("duplex playback guard", str(captured.exception))

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
