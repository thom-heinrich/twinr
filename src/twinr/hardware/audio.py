from __future__ import annotations

import io
import math
import os
import sys
from array import array
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import select
import subprocess
import time
import wave

from twinr.agent.base_agent.config import TwinrConfig

_SAMPLE_WIDTH_BYTES = 2


def _pcm16_rms(samples: bytes) -> int:
    if not samples:
        return 0
    pcm_samples = array("h")
    pcm_samples.frombytes(samples)
    if sys.byteorder != "little":
        pcm_samples.byteswap()
    mean_square = sum(sample * sample for sample in pcm_samples) / len(pcm_samples)
    return int(math.sqrt(mean_square))


@dataclass(frozen=True, slots=True)
class AmbientAudioLevelSample:
    duration_ms: int
    chunk_count: int
    active_chunk_count: int
    average_rms: int
    peak_rms: int
    active_ratio: float


@dataclass(frozen=True, slots=True)
class AmbientAudioCaptureWindow:
    sample: AmbientAudioLevelSample
    pcm_bytes: bytes
    sample_rate: int
    channels: int


@dataclass(frozen=True, slots=True)
class SpeechCaptureResult:
    pcm_bytes: bytes
    speech_started_after_ms: int
    resumed_after_pause_count: int


def resolve_pause_resume_confirmation(
    *,
    consecutive_resume_chunks: int,
    required_resume_chunks: int,
) -> tuple[bool, int]:
    required_chunks = max(1, int(required_resume_chunks))
    next_chunks = max(0, int(consecutive_resume_chunks)) + 1
    if next_chunks >= required_chunks:
        return True, 0
    return False, next_chunks


def resolve_dynamic_pause_thresholds(
    *,
    base_pause_ms: int,
    base_pause_grace_ms: int,
    speech_window_ms: int,
    enabled: bool,
    short_utterance_max_ms: int,
    long_utterance_min_ms: int,
    short_pause_bonus_ms: int,
    short_pause_grace_bonus_ms: int,
    medium_pause_penalty_ms: int,
    medium_pause_grace_penalty_ms: int,
    long_pause_penalty_ms: int,
    long_pause_grace_penalty_ms: int,
) -> tuple[int, int]:
    pause_ms = max(0, int(base_pause_ms))
    pause_grace_ms = max(0, int(base_pause_grace_ms))
    utterance_ms = max(0, int(speech_window_ms))
    if not enabled or utterance_ms <= 0:
        return pause_ms, pause_grace_ms
    if utterance_ms <= max(0, int(short_utterance_max_ms)):
        return (
            pause_ms + max(0, int(short_pause_bonus_ms)),
            pause_grace_ms + max(0, int(short_pause_grace_bonus_ms)),
        )
    if utterance_ms < max(0, int(long_utterance_min_ms)):
        return (
            max(0, pause_ms - max(0, int(medium_pause_penalty_ms))),
            max(0, pause_grace_ms - max(0, int(medium_pause_grace_penalty_ms))),
        )
    if utterance_ms >= max(0, int(long_utterance_min_ms)):
        return (
            max(0, pause_ms - max(0, int(long_pause_penalty_ms))),
            max(0, pause_grace_ms - max(0, int(long_pause_grace_penalty_ms))),
        )
    return pause_ms, pause_grace_ms


def pcm16_to_wav_bytes(
    pcm_bytes: bytes,
    *,
    sample_rate: int,
    channels: int,
) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(_SAMPLE_WIDTH_BYTES)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return buffer.getvalue()


class SilenceDetectedRecorder:
    def __init__(
        self,
        *,
        device: str = "default",
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_ms: int = 100,
        preroll_ms: int = 300,
        speech_threshold: int = 700,
        speech_start_chunks: int = 1,
        start_timeout_s: float = 8.0,
        max_record_seconds: float = 20.0,
        dynamic_pause_enabled: bool = True,
        dynamic_pause_short_utterance_max_ms: int = 1000,
        dynamic_pause_long_utterance_min_ms: int = 5000,
        dynamic_pause_short_pause_bonus_ms: int = 120,
        dynamic_pause_short_pause_grace_bonus_ms: int = 0,
        dynamic_pause_medium_pause_penalty_ms: int = 120,
        dynamic_pause_medium_pause_grace_penalty_ms: int = 250,
        dynamic_pause_long_pause_penalty_ms: int = 320,
        dynamic_pause_long_pause_grace_penalty_ms: int = 220,
        pause_resume_chunks: int = 2,
    ) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_ms = chunk_ms
        self.preroll_ms = preroll_ms
        self.speech_threshold = speech_threshold
        self.speech_start_chunks = max(1, speech_start_chunks)
        self.start_timeout_s = start_timeout_s
        self.max_record_seconds = max_record_seconds
        self.dynamic_pause_enabled = dynamic_pause_enabled
        self.dynamic_pause_short_utterance_max_ms = max(0, dynamic_pause_short_utterance_max_ms)
        self.dynamic_pause_long_utterance_min_ms = max(0, dynamic_pause_long_utterance_min_ms)
        self.dynamic_pause_short_pause_bonus_ms = max(0, dynamic_pause_short_pause_bonus_ms)
        self.dynamic_pause_short_pause_grace_bonus_ms = max(
            0,
            dynamic_pause_short_pause_grace_bonus_ms,
        )
        self.dynamic_pause_medium_pause_penalty_ms = max(0, dynamic_pause_medium_pause_penalty_ms)
        self.dynamic_pause_medium_pause_grace_penalty_ms = max(
            0,
            dynamic_pause_medium_pause_grace_penalty_ms,
        )
        self.dynamic_pause_long_pause_penalty_ms = max(0, dynamic_pause_long_pause_penalty_ms)
        self.dynamic_pause_long_pause_grace_penalty_ms = max(
            0,
            dynamic_pause_long_pause_grace_penalty_ms,
        )
        self.pause_resume_chunks = max(1, pause_resume_chunks)

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "SilenceDetectedRecorder":
        return cls(
            device=config.audio_input_device,
            sample_rate=config.audio_sample_rate,
            channels=config.audio_channels,
            chunk_ms=config.audio_chunk_ms,
            preroll_ms=config.audio_preroll_ms,
            speech_threshold=config.audio_speech_threshold,
            speech_start_chunks=config.audio_speech_start_chunks,
            start_timeout_s=config.audio_start_timeout_s,
            max_record_seconds=config.audio_max_record_seconds,
            dynamic_pause_enabled=config.audio_dynamic_pause_enabled,
            dynamic_pause_short_utterance_max_ms=config.audio_dynamic_pause_short_utterance_max_ms,
            dynamic_pause_long_utterance_min_ms=config.audio_dynamic_pause_long_utterance_min_ms,
            dynamic_pause_short_pause_bonus_ms=config.audio_dynamic_pause_short_pause_bonus_ms,
            dynamic_pause_short_pause_grace_bonus_ms=config.audio_dynamic_pause_short_pause_grace_bonus_ms,
            dynamic_pause_medium_pause_penalty_ms=config.audio_dynamic_pause_medium_pause_penalty_ms,
            dynamic_pause_medium_pause_grace_penalty_ms=config.audio_dynamic_pause_medium_pause_grace_penalty_ms,
            dynamic_pause_long_pause_penalty_ms=config.audio_dynamic_pause_long_pause_penalty_ms,
            dynamic_pause_long_pause_grace_penalty_ms=config.audio_dynamic_pause_long_pause_grace_penalty_ms,
            pause_resume_chunks=config.audio_pause_resume_chunks,
        )

    def record_until_pause(self, *, pause_ms: int) -> bytes:
        return self._pcm_to_wav(self.record_pcm_until_pause(pause_ms=pause_ms))

    def record_pcm_until_pause(self, *, pause_ms: int) -> bytes:
        return self.record_pcm_until_pause_with_options(pause_ms=pause_ms)

    def capture_pcm_until_pause_with_options(
        self,
        *,
        pause_ms: int,
        start_timeout_s: float | None = None,
        max_record_seconds: float | None = None,
        speech_start_chunks: int | None = None,
        ignore_initial_ms: int = 0,
        pause_grace_ms: int = 0,
        on_chunk: Callable[[bytes], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> SpeechCaptureResult:
        command = [
            "arecord",
            "-D",
            self.device,
            "-q",
            "-t",
            "raw",
            "-f",
            "S16_LE",
            "-c",
            str(self.channels),
            "-r",
            str(self.sample_rate),
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        captured = bytearray()
        preroll: deque[bytes] = deque(maxlen=max(1, self.preroll_ms // max(1, self.chunk_ms)))
        heard_speech = False
        consecutive_speech_chunks = 0
        consecutive_resume_chunks = 0
        speech_started_at: float | None = None
        last_non_silent_at: float | None = None
        pause_candidate_deadline_at: float | None = None
        started_at = time.monotonic()
        effective_start_timeout_s = self.start_timeout_s if start_timeout_s is None else start_timeout_s
        effective_max_record_seconds = (
            self.max_record_seconds if max_record_seconds is None else max_record_seconds
        )
        effective_speech_start_chunks = (
            self.speech_start_chunks if speech_start_chunks is None else max(1, speech_start_chunks)
        )
        effective_pause_grace_ms = max(0, int(pause_grace_ms))
        speech_started_after_ms = 0
        resumed_after_pause_count = 0
        chunk_bytes = max(
            _SAMPLE_WIDTH_BYTES * self.channels,
            int((self.sample_rate * self.channels * _SAMPLE_WIDTH_BYTES * self.chunk_ms) / 1000),
        )

        try:
            while True:
                now = time.monotonic()
                if not heard_speech and now - started_at >= effective_start_timeout_s:
                    raise RuntimeError("No speech detected before timeout")
                if heard_speech and now - started_at >= effective_max_record_seconds:
                    break
                if heard_speech and should_stop is not None and should_stop():
                    break
                if process.poll() is not None:
                    self._raise_process_error(process)
                if process.stdout is None:
                    raise RuntimeError("arecord did not expose stdout")

                ready, _write_ready, _error_ready = select.select([process.stdout], [], [], 0.25)
                if not ready:
                    if heard_speech and pause_candidate_deadline_at is not None and time.monotonic() >= pause_candidate_deadline_at:
                        break
                    if heard_speech and should_stop is not None and should_stop():
                        break
                    continue

                chunk = os.read(process.stdout.fileno(), chunk_bytes)
                if not chunk:
                    continue
                if len(chunk) % _SAMPLE_WIDTH_BYTES:
                    chunk = chunk[: -(len(chunk) % _SAMPLE_WIDTH_BYTES)]
                if not chunk:
                    continue

                rms = _pcm16_rms(chunk)
                now = time.monotonic()

                if not heard_speech:
                    if ignore_initial_ms > 0 and (now - started_at) * 1000 < ignore_initial_ms:
                        preroll.clear()
                        consecutive_speech_chunks = 0
                        continue
                    if on_chunk is not None:
                        on_chunk(chunk)
                    preroll.append(chunk)
                    if rms >= self.speech_threshold:
                        consecutive_speech_chunks += 1
                        if consecutive_speech_chunks >= effective_speech_start_chunks:
                            heard_speech = True
                            speech_started_at = now
                            last_non_silent_at = now
                            pause_candidate_deadline_at = None
                            speech_started_after_ms = int((now - started_at) * 1000)
                            for buffered_chunk in preroll:
                                captured.extend(buffered_chunk)
                    else:
                        consecutive_speech_chunks = 0
                    continue

                captured.extend(chunk)
                if on_chunk is not None:
                    on_chunk(chunk)
                if rms >= self.speech_threshold:
                    if pause_candidate_deadline_at is not None:
                        confirmed_resume, consecutive_resume_chunks = (
                            resolve_pause_resume_confirmation(
                                consecutive_resume_chunks=consecutive_resume_chunks,
                                required_resume_chunks=self.pause_resume_chunks,
                            )
                        )
                        if not confirmed_resume:
                            continue
                        resumed_after_pause_count += 1
                    consecutive_resume_chunks = 0
                    pause_candidate_deadline_at = None
                    last_non_silent_at = now
                    continue
                consecutive_resume_chunks = 0
                if last_non_silent_at is None:
                    continue
                speech_window_ms = 0
                if speech_started_at is not None:
                    speech_window_ms = int((last_non_silent_at - speech_started_at) * 1000)
                effective_pause_ms, effective_pause_grace_ms = resolve_dynamic_pause_thresholds(
                    base_pause_ms=pause_ms,
                    base_pause_grace_ms=pause_grace_ms,
                    speech_window_ms=speech_window_ms,
                    enabled=self.dynamic_pause_enabled,
                    short_utterance_max_ms=self.dynamic_pause_short_utterance_max_ms,
                    long_utterance_min_ms=self.dynamic_pause_long_utterance_min_ms,
                    short_pause_bonus_ms=self.dynamic_pause_short_pause_bonus_ms,
                    short_pause_grace_bonus_ms=self.dynamic_pause_short_pause_grace_bonus_ms,
                    medium_pause_penalty_ms=self.dynamic_pause_medium_pause_penalty_ms,
                    medium_pause_grace_penalty_ms=self.dynamic_pause_medium_pause_grace_penalty_ms,
                    long_pause_penalty_ms=self.dynamic_pause_long_pause_penalty_ms,
                    long_pause_grace_penalty_ms=self.dynamic_pause_long_pause_grace_penalty_ms,
                )
                silence_ms = int((now - last_non_silent_at) * 1000)
                if silence_ms < effective_pause_ms:
                    continue
                if effective_pause_grace_ms <= 0:
                    break
                if pause_candidate_deadline_at is None:
                    consecutive_resume_chunks = 0
                    pause_candidate_deadline_at = last_non_silent_at + (
                        effective_pause_ms + effective_pause_grace_ms
                    ) / 1000.0
                    continue
                if now >= pause_candidate_deadline_at:
                    break
        finally:
            self._stop_process(process)

        if not heard_speech or not captured:
            raise RuntimeError("Speech capture ended without usable audio")
        return SpeechCaptureResult(
            pcm_bytes=bytes(captured),
            speech_started_after_ms=max(0, speech_started_after_ms),
            resumed_after_pause_count=max(0, resumed_after_pause_count),
        )

    def record_pcm_until_pause_with_options(
        self,
        *,
        pause_ms: int,
        start_timeout_s: float | None = None,
        max_record_seconds: float | None = None,
        speech_start_chunks: int | None = None,
        ignore_initial_ms: int = 0,
        pause_grace_ms: int = 0,
    ) -> bytes:
        result = self.capture_pcm_until_pause_with_options(
            pause_ms=pause_ms,
            start_timeout_s=start_timeout_s,
            max_record_seconds=max_record_seconds,
            speech_start_chunks=speech_start_chunks,
            ignore_initial_ms=ignore_initial_ms,
            pause_grace_ms=pause_grace_ms,
            on_chunk=None,
            should_stop=None,
        )
        return result.pcm_bytes

    def _pcm_to_wav(self, pcm_bytes: bytes) -> bytes:
        return pcm16_to_wav_bytes(
            pcm_bytes,
            sample_rate=self.sample_rate,
            channels=self.channels,
        )

    def _raise_process_error(self, process: subprocess.Popen[bytes]) -> None:
        stderr = b""
        if process.stderr is not None:
            stderr = process.stderr.read().strip()
        message = stderr.decode("utf-8", errors="ignore") if stderr else f"exit code {process.returncode}"
        raise RuntimeError(f"Audio capture failed: {message}")

    def _stop_process(self, process: subprocess.Popen[bytes]) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)


class AmbientAudioSampler:
    def __init__(
        self,
        *,
        device: str = "default",
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_ms: int = 100,
        speech_threshold: int = 700,
        default_duration_ms: int = 1000,
    ) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_ms = max(20, chunk_ms)
        self.speech_threshold = speech_threshold
        self.default_duration_ms = max(self.chunk_ms, default_duration_ms)

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "AmbientAudioSampler":
        device = (config.proactive_audio_input_device or config.audio_input_device).strip()
        return cls(
            device=device,
            sample_rate=config.audio_sample_rate,
            channels=config.audio_channels,
            chunk_ms=config.audio_chunk_ms,
            speech_threshold=config.audio_speech_threshold,
            default_duration_ms=config.proactive_audio_sample_ms,
        )

    def sample_levels(self, *, duration_ms: int | None = None) -> AmbientAudioLevelSample:
        return self.sample_window(duration_ms=duration_ms).sample

    def sample_window(self, *, duration_ms: int | None = None) -> AmbientAudioCaptureWindow:
        effective_duration_ms = max(self.chunk_ms, duration_ms or self.default_duration_ms)
        chunk_bytes = max(
            _SAMPLE_WIDTH_BYTES * self.channels,
            int((self.sample_rate * self.channels * _SAMPLE_WIDTH_BYTES * self.chunk_ms) / 1000),
        )
        target_chunk_count = max(1, math.ceil(effective_duration_ms / self.chunk_ms))
        command = [
            "arecord",
            "-D",
            self.device,
            "-q",
            "-t",
            "raw",
            "-f",
            "S16_LE",
            "-c",
            str(self.channels),
            "-r",
            str(self.sample_rate),
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        rms_values: list[int] = []
        pcm_fragments: list[bytes] = []

        try:
            while len(rms_values) < target_chunk_count:
                if process.poll() is not None:
                    self._raise_process_error(process)
                if process.stdout is None:
                    raise RuntimeError("arecord did not expose stdout")
                ready, _write_ready, _error_ready = select.select([process.stdout], [], [], 0.25)
                if not ready:
                    continue
                chunk = os.read(process.stdout.fileno(), chunk_bytes)
                if not chunk:
                    continue
                if len(chunk) % _SAMPLE_WIDTH_BYTES:
                    chunk = chunk[: -(len(chunk) % _SAMPLE_WIDTH_BYTES)]
                if not chunk:
                    continue
                rms_values.append(_pcm16_rms(chunk))
                pcm_fragments.append(chunk)
        finally:
            self._stop_process(process)

        if not rms_values:
            raise RuntimeError("Ambient audio capture ended without usable samples")

        active_chunk_count = sum(1 for rms in rms_values if rms >= self.speech_threshold)
        average_rms = int(sum(rms_values) / len(rms_values))
        peak_rms = max(rms_values)
        sample = AmbientAudioLevelSample(
            duration_ms=effective_duration_ms,
            chunk_count=len(rms_values),
            active_chunk_count=active_chunk_count,
            average_rms=average_rms,
            peak_rms=peak_rms,
            active_ratio=active_chunk_count / max(1, len(rms_values)),
        )
        return AmbientAudioCaptureWindow(
            sample=sample,
            pcm_bytes=b"".join(pcm_fragments),
            sample_rate=self.sample_rate,
            channels=self.channels,
        )

    def _raise_process_error(self, process: subprocess.Popen[bytes]) -> None:
        stderr = b""
        if process.stderr is not None:
            stderr = process.stderr.read().strip()
        message = stderr.decode("utf-8", errors="ignore") if stderr else f"exit code {process.returncode}"
        raise RuntimeError(f"Ambient audio capture failed: {message}")

    def _stop_process(self, process: subprocess.Popen[bytes]) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)


class WaveAudioPlayer:
    def __init__(self, *, device: str = "default") -> None:
        self.device = device

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "WaveAudioPlayer":
        return cls(device=config.audio_output_device)

    def play_wav_bytes(self, audio_bytes: bytes) -> None:
        temp_path = Path("/tmp/twinr-playback.wav")
        temp_path.write_bytes(audio_bytes)
        try:
            self.play_file(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)

    def play_tone(
        self,
        *,
        frequency_hz: int = 1046,
        duration_ms: int = 180,
        volume: float = 0.8,
        sample_rate: int = 24000,
    ) -> None:
        self.play_pcm16_chunks(
            [
                self._render_tone_pcm(
                    frequency_hz=frequency_hz,
                    duration_ms=duration_ms,
                    volume=volume,
                    sample_rate=sample_rate,
                )
            ],
            sample_rate=sample_rate,
            channels=1,
        )

    def play_tone_sequence(
        self,
        tones: Iterable[tuple[int, int]],
        *,
        volume: float = 0.28,
        sample_rate: int = 24000,
        gap_ms: int = 34,
    ) -> None:
        pcm = bytearray()
        for index, (frequency_hz, duration_ms) in enumerate(tones):
            if index > 0 and gap_ms > 0:
                pcm.extend(self._render_silence_pcm(duration_ms=gap_ms, sample_rate=sample_rate))
            pcm.extend(
                self._render_tone_pcm(
                    frequency_hz=frequency_hz,
                    duration_ms=duration_ms,
                    volume=volume,
                    sample_rate=sample_rate,
                )
            )
        if not pcm:
            return
        self.play_pcm16_chunks([bytes(pcm)], sample_rate=sample_rate, channels=1)

    def play_wav_chunks(self, chunks: Iterable[bytes]) -> None:
        self._play_stream(
            ["aplay", "-q", "-D", self.device, "-"],
            chunks,
        )

    def play_pcm16_chunks(
        self,
        chunks: Iterable[bytes],
        *,
        sample_rate: int,
        channels: int = 1,
    ) -> None:
        self._play_stream(
            [
                "aplay",
                "-q",
                "-D",
                self.device,
                "-t",
                "raw",
                "-f",
                "S16_LE",
                "-c",
                str(channels),
                "-r",
                str(sample_rate),
                "-",
            ],
            chunks,
        )

    def _play_stream(self, command: list[str], chunks: Iterable[bytes]) -> None:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            if process.stdin is None:
                raise RuntimeError("aplay did not expose stdin")
            for chunk in chunks:
                if process.poll() is not None:
                    self._raise_stream_error(process)
                process.stdin.write(chunk)
                process.stdin.flush()
            process.stdin.close()
            process.wait()
            if process.returncode != 0:
                self._raise_stream_error(process)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1.0)

    def play_file(self, path: str | Path) -> None:
        result = subprocess.run(
            ["aplay", "-q", "-D", self.device, str(path)],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(f"Audio playback failed: {stderr or result.returncode}")

    def _raise_stream_error(self, process: subprocess.Popen[bytes]) -> None:
        stderr = b""
        if process.stderr is not None:
            stderr = process.stderr.read().strip()
        message = stderr.decode("utf-8", errors="ignore") if stderr else f"exit code {process.returncode}"
        raise RuntimeError(f"Audio playback failed: {message}")

    def _render_tone_pcm(
        self,
        *,
        frequency_hz: int,
        duration_ms: int,
        volume: float,
        sample_rate: int,
    ) -> bytes:
        amplitude = max(0.0, min(volume, 1.0)) * 32767.0 * 0.6
        frame_count = max(1, int(sample_rate * (duration_ms / 1000.0)))
        pcm = bytearray()
        for frame_index in range(frame_count):
            sample = int(amplitude * math.sin(2.0 * math.pi * frequency_hz * frame_index / sample_rate))
            pcm.extend(sample.to_bytes(2, byteorder="little", signed=True))
        return bytes(pcm)

    def _render_silence_pcm(self, *, duration_ms: int, sample_rate: int) -> bytes:
        frame_count = max(1, int(sample_rate * (duration_ms / 1000.0)))
        return b"\x00\x00" * frame_count
