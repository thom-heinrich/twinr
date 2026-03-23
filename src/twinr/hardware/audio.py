"""Capture and play bounded audio for Twinr hardware workflows.

This module exposes microphone recording, ambient sampling, and WAV/PCM
playback helpers that keep ALSA subprocesses bounded and configuration
validation local to the hardware adapter layer.
"""

from __future__ import annotations

import io
import math
import os
import sys
import audioop
from array import array
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import select
import subprocess
import tempfile
import time
import wave
from threading import Lock

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.audio_env import build_audio_subprocess_env
from twinr.hardware.respeaker_capture_recovery import wait_for_transient_respeaker_capture_ready

_SAMPLE_WIDTH_BYTES = 2
# AUDIT-FIX(#3): Bound blocking device I/O so broken ALSA devices cannot wedge the process forever.
_SELECT_TIMEOUT_S = 0.25
_AMBIENT_CAPTURE_EXTRA_TIMEOUT_S = 2.0
_STREAM_IO_STALL_TIMEOUT_S = 5.0
_PLAYBACK_FINALIZE_TIMEOUT_S = 10.0
_PLAYBACK_FILE_TIMEOUT_S = 120.0
_PROCESS_STOP_TIMEOUT_S = 1.0
_PCM16_MAX_ABS = 32767
_DEFAULT_WAV_TARGET_PEAK = 28000
_DEFAULT_WAV_MAX_GAIN = 4.0
_TONE_HEADROOM_SCALE = 0.92


def _normalize_audio_device(device: str | None) -> str:
    # AUDIT-FIX(#7): Normalize blank/None device names to a safe default instead of passing invalid values to ALSA.
    normalized = str(device or "default").strip()
    return normalized or "default"


def _ensure_int(name: str, value: object, *, minimum: int) -> int:
    # AUDIT-FIX(#7): Fail fast with clear config errors for invalid numeric parameters.
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if normalized < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return normalized


def _ensure_float(name: str, value: object, *, minimum: float) -> float:
    # AUDIT-FIX(#7): Fail fast with clear config errors for invalid timeout parameters.
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a float >= {minimum}") from exc
    if normalized < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return normalized


def normalize_wav_playback_level(
    audio_bytes: bytes,
    *,
    target_peak: int = _DEFAULT_WAV_TARGET_PEAK,
    max_gain: float = _DEFAULT_WAV_MAX_GAIN,
) -> bytes:
    """Boost a quiet PCM16 WAV payload to an audible playback level.

    The Pi ReSpeaker path can already be at full mixer volume while provider TTS
    returns under-driven audio. In that case the only reliable place to recover
    audible output is the WAV payload itself. Invalid or already-loud audio
    stays untouched.
    """

    if not audio_bytes:
        return audio_bytes
    normalized_target_peak = max(1, min(int(target_peak), _PCM16_MAX_ABS))
    normalized_max_gain = max(1.0, float(max_gain))
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wave_reader:
            channels = wave_reader.getnchannels()
            sample_width = wave_reader.getsampwidth()
            sample_rate = wave_reader.getframerate()
            frames = wave_reader.readframes(wave_reader.getnframes())
    except (wave.Error, EOFError):
        return audio_bytes
    if sample_width != _SAMPLE_WIDTH_BYTES or not frames:
        return audio_bytes
    peak = audioop.max(frames, sample_width)
    if peak <= 0 or peak >= normalized_target_peak:
        return audio_bytes
    gain = min(normalized_max_gain, normalized_target_peak / float(peak))
    if gain <= 1.0:
        return audio_bytes
    boosted_frames = audioop.mul(frames, sample_width, gain)
    output_buffer = io.BytesIO()
    with wave.open(output_buffer, "wb") as wave_writer:
        wave_writer.setnchannels(channels)
        wave_writer.setsampwidth(sample_width)
        wave_writer.setframerate(sample_rate)
        wave_writer.writeframes(boosted_frames)
    return output_buffer.getvalue()


def _bytes_per_frame(channels: int) -> int:
    return _SAMPLE_WIDTH_BYTES * _ensure_int("channels", channels, minimum=1)


def _chunk_byte_count(*, sample_rate: int, channels: int, chunk_ms: int) -> int:
    normalized_channels = _ensure_int("channels", channels, minimum=1)
    frame_bytes = _bytes_per_frame(normalized_channels)
    normalized_sample_rate = _ensure_int("sample_rate", sample_rate, minimum=1)
    normalized_chunk_ms = _ensure_int("chunk_ms", chunk_ms, minimum=1)
    return max(
        frame_bytes,
        int(
            (
                normalized_sample_rate
                * normalized_channels
                * _SAMPLE_WIDTH_BYTES
                * normalized_chunk_ms
            )
            / 1000
        ),
    )


def _trim_incomplete_bytes(payload: bytes, *, alignment: int) -> bytes:
    if alignment <= 1 or not payload:
        return payload
    usable_length = len(payload) - (len(payload) % alignment)
    if usable_length <= 0:
        return b""
    return payload[:usable_length]


def _compute_read_stall_timeout(chunk_ms: int) -> float:
    normalized_chunk_ms = _ensure_int("chunk_ms", chunk_ms, minimum=1)
    return max(2.0, (normalized_chunk_ms / 1000.0) * 4.0)


def _spawn_audio_process(
    command: list[str],
    *,
    stdin: int | None = None,
    stdout: int | None = None,
    stderr: int | None = None,
    purpose: str,
) -> subprocess.Popen[bytes]:
    # AUDIT-FIX(#5): Convert raw OS-level spawn failures into actionable audio-specific runtime errors.
    try:
        return subprocess.Popen(
            command,
            env=build_audio_subprocess_env(),
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            close_fds=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"{purpose} failed because '{command[0]}' is not installed") from exc
    except OSError as exc:
        raise RuntimeError(f"{purpose} failed to start: {exc}") from exc


def _close_pipe(pipe: object | None) -> None:
    try:
        if pipe is not None and hasattr(pipe, "close"):
            pipe.close()  # type: ignore[call-arg]
    except OSError:
        return


def _read_process_stderr(process: subprocess.Popen[bytes]) -> bytes:
    stderr = process.stderr
    if stderr is None or getattr(stderr, "closed", False):
        return b""
    try:
        return stderr.read().strip()
    except (OSError, ValueError):
        return b""


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    # AUDIT-FIX(#6): Make shutdown idempotent and non-masking so cleanup never hides the real failure.
    try:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=_PROCESS_STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=_PROCESS_STOP_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    pass
    finally:
        _close_pipe(process.stdin)
        _close_pipe(process.stdout)
        _close_pipe(process.stderr)


def _wait_for_readable(stream: object, *, timeout_s: float, purpose: str) -> bool:
    try:
        readable, _write_ready, _error_ready = select.select([stream], [], [], timeout_s)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"{purpose} failed while waiting for input data") from exc
    return bool(readable)


def _wait_for_writable(fd: int, *, timeout_s: float, purpose: str) -> bool:
    try:
        _read_ready, write_ready, _error_ready = select.select([], [fd], [], timeout_s)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"{purpose} failed while waiting for output device") from exc
    return bool(write_ready)


def _process_failure_message(process: subprocess.Popen[bytes], *, default_action: str) -> str:
    stderr = _read_process_stderr(process)
    if stderr:
        return stderr.decode("utf-8", errors="ignore")
    if process.returncode == 0:
        return f"{default_action} ended unexpectedly"
    return f"exit code {process.returncode}"


def _pcm16_rms(samples: bytes) -> int:
    if not samples:
        return 0
    # AUDIT-FIX(#9): Defensively trim partial samples so malformed buffers never crash RMS calculation.
    aligned_samples = _trim_incomplete_bytes(samples, alignment=_SAMPLE_WIDTH_BYTES)
    if not aligned_samples:
        return 0
    pcm_samples = array("h")
    pcm_samples.frombytes(aligned_samples)
    if sys.byteorder != "little":
        pcm_samples.byteswap()
    mean_square = sum(sample * sample for sample in pcm_samples) / len(pcm_samples)
    return int(math.sqrt(mean_square))


@dataclass(frozen=True, slots=True)
class AmbientAudioLevelSample:
    """Represent aggregate loudness metrics for one ambient capture window."""

    duration_ms: int
    chunk_count: int
    active_chunk_count: int
    average_rms: int
    peak_rms: int
    active_ratio: float


@dataclass(frozen=True, slots=True)
class AmbientAudioCaptureWindow:
    """Represent ambient PCM bytes together with sampled loudness metadata."""

    sample: AmbientAudioLevelSample
    pcm_bytes: bytes
    sample_rate: int
    channels: int


@dataclass(frozen=True, slots=True)
class SpeechCaptureResult:
    """Represent captured PCM speech and pause-resume timing metadata."""

    pcm_bytes: bytes
    speech_started_after_ms: int
    resumed_after_pause_count: int


@dataclass(frozen=True, slots=True)
class ListenTimeoutCaptureDiagnostics:
    """Summarize pre-speech capture evidence for one listen-timeout."""

    device: str
    sample_rate: int
    channels: int
    chunk_ms: int
    speech_threshold: int
    chunk_count: int
    active_chunk_count: int
    average_rms: int
    peak_rms: int
    listened_ms: int

    @property
    def active_ratio(self) -> float:
        """Return the fraction of pre-speech chunks above the speech threshold."""

        return self.active_chunk_count / max(1, self.chunk_count)


@dataclass(frozen=True, slots=True)
class AudioCaptureReadinessProbe:
    """Describe whether a bounded audio-capture probe yielded readable frames."""

    device: str
    sample_rate: int
    channels: int
    chunk_ms: int
    duration_ms: int
    target_chunk_count: int
    captured_chunk_count: int
    captured_bytes: int
    failure_reason: str | None = None
    detail: str | None = None

    @property
    def ready(self) -> bool:
        """Return whether the probe captured the requested readable chunks."""

        return (
            self.failure_reason is None
            and self.captured_chunk_count >= self.target_chunk_count
            and self.captured_bytes > 0
        )


class SpeechStartTimeoutError(RuntimeError):
    """Raise when speech never starts and retain bounded capture diagnostics."""

    def __init__(self, message: str, *, diagnostics: ListenTimeoutCaptureDiagnostics) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


class AudioCaptureReadinessError(RuntimeError):
    """Raise when a bounded microphone probe cannot read usable audio frames."""

    def __init__(self, message: str, *, probe: AudioCaptureReadinessProbe) -> None:
        super().__init__(message)
        self.probe = probe


def resolve_pause_resume_confirmation(
    *,
    consecutive_resume_chunks: int,
    required_resume_chunks: int,
) -> tuple[bool, int]:
    """Confirm whether resumed speech is stable enough to cancel a pause."""

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
    """Adjust pause thresholds based on how long the utterance has lasted."""

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
    """Wrap raw PCM16 audio bytes in an in-memory WAV container."""

    normalized_sample_rate = _ensure_int("sample_rate", sample_rate, minimum=1)
    normalized_channels = _ensure_int("channels", channels, minimum=1)
    # AUDIT-FIX(#9): Drop incomplete trailing frames so emitted WAV data stays frame-aligned.
    aligned_pcm_bytes = _trim_incomplete_bytes(
        pcm_bytes,
        alignment=_SAMPLE_WIDTH_BYTES * normalized_channels,
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(normalized_channels)
        wav_file.setsampwidth(_SAMPLE_WIDTH_BYTES)
        wav_file.setframerate(normalized_sample_rate)
        wav_file.writeframes(aligned_pcm_bytes)
    return buffer.getvalue()


def pcm16_duration_ms(
    pcm_bytes: bytes,
    *,
    sample_rate: int,
    channels: int,
) -> int:
    """Return the duration of a PCM16 payload after frame alignment."""

    normalized_sample_rate = _ensure_int("sample_rate", sample_rate, minimum=1)
    normalized_channels = _ensure_int("channels", channels, minimum=1)
    frame_bytes = _SAMPLE_WIDTH_BYTES * normalized_channels
    aligned_pcm_bytes = _trim_incomplete_bytes(pcm_bytes, alignment=frame_bytes)
    if not aligned_pcm_bytes:
        return 0
    frame_count = len(aligned_pcm_bytes) // frame_bytes
    return int((frame_count / normalized_sample_rate) * 1000)


class SilenceDetectedRecorder:
    """Record microphone audio until speech pauses or capture limits fire."""

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
        # AUDIT-FIX(#7): Validate recorder config at construction time so bad .env values fail early and clearly.
        self.device = _normalize_audio_device(device)
        self.sample_rate = _ensure_int("sample_rate", sample_rate, minimum=1)
        self.channels = _ensure_int("channels", channels, minimum=1)
        self.chunk_ms = _ensure_int("chunk_ms", chunk_ms, minimum=1)
        self.preroll_ms = _ensure_int("preroll_ms", preroll_ms, minimum=0)
        self.speech_threshold = _ensure_int("speech_threshold", speech_threshold, minimum=0)
        self.speech_start_chunks = _ensure_int("speech_start_chunks", speech_start_chunks, minimum=1)
        self.start_timeout_s = _ensure_float("start_timeout_s", start_timeout_s, minimum=0.0)
        self.max_record_seconds = _ensure_float("max_record_seconds", max_record_seconds, minimum=0.1)
        self.dynamic_pause_enabled = bool(dynamic_pause_enabled)
        self.dynamic_pause_short_utterance_max_ms = max(0, int(dynamic_pause_short_utterance_max_ms))
        self.dynamic_pause_long_utterance_min_ms = max(0, int(dynamic_pause_long_utterance_min_ms))
        self.dynamic_pause_short_pause_bonus_ms = max(0, int(dynamic_pause_short_pause_bonus_ms))
        self.dynamic_pause_short_pause_grace_bonus_ms = max(
            0,
            int(dynamic_pause_short_pause_grace_bonus_ms),
        )
        self.dynamic_pause_medium_pause_penalty_ms = max(0, int(dynamic_pause_medium_pause_penalty_ms))
        self.dynamic_pause_medium_pause_grace_penalty_ms = max(
            0,
            int(dynamic_pause_medium_pause_grace_penalty_ms),
        )
        self.dynamic_pause_long_pause_penalty_ms = max(0, int(dynamic_pause_long_pause_penalty_ms))
        self.dynamic_pause_long_pause_grace_penalty_ms = max(
            0,
            int(dynamic_pause_long_pause_grace_penalty_ms),
        )
        self.pause_resume_chunks = _ensure_int("pause_resume_chunks", pause_resume_chunks, minimum=1)

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "SilenceDetectedRecorder":
        """Build a recorder from ``TwinrConfig`` values."""

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
        """Record speech and return the result as WAV bytes."""

        return self._pcm_to_wav(self.record_pcm_until_pause(pause_ms=pause_ms))

    def record_pcm_until_pause(self, *, pause_ms: int) -> bytes:
        """Record speech and return the result as raw PCM16 bytes."""

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
        """Record PCM16 speech until a confirmed pause and return timing data.

        Args:
            pause_ms: Base silence duration required to stop recording.
            start_timeout_s: Optional override for how long to wait for speech.
            max_record_seconds: Optional override for maximum speech length.
            speech_start_chunks: Optional override for speech-start detection.
            ignore_initial_ms: Milliseconds to ignore before speech detection.
            pause_grace_ms: Extra grace time after a pause candidate.
            on_chunk: Optional callback invoked for each captured chunk.
            should_stop: Optional cancellation callback checked during capture.

        Returns:
            Captured PCM16 bytes plus speech-start and resume metadata.

        Raises:
            RuntimeError: If recording times out, stalls, is cancelled, or ends
                without usable speech audio.
        """

        normalized_pause_ms = max(0, int(pause_ms))
        normalized_ignore_initial_ms = max(0, int(ignore_initial_ms))
        normalized_pause_grace_ms = max(0, int(pause_grace_ms))
        effective_start_timeout_s = (
            self.start_timeout_s
            if start_timeout_s is None
            else _ensure_float("start_timeout_s", start_timeout_s, minimum=0.0)
        )
        effective_max_record_seconds = (
            self.max_record_seconds
            if max_record_seconds is None
            else _ensure_float("max_record_seconds", max_record_seconds, minimum=0.1)
        )
        effective_speech_start_chunks = (
            self.speech_start_chunks
            if speech_start_chunks is None
            else _ensure_int("speech_start_chunks", speech_start_chunks, minimum=1)
        )
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
        process = _spawn_audio_process(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            purpose="Audio capture",
        )
        captured = bytearray()
        # AUDIT-FIX(#8): Use ceil for preroll so we never under-buffer and clip the user's first syllable.
        preroll_chunk_count = max(1, math.ceil(self.preroll_ms / self.chunk_ms))
        preroll: deque[bytes] = deque(maxlen=preroll_chunk_count)
        heard_speech = False
        consecutive_speech_chunks = 0
        consecutive_resume_chunks = 0
        speech_started_at: float | None = None
        last_non_silent_at: float | None = None
        pause_candidate_deadline_at: float | None = None
        started_at = time.monotonic()
        last_chunk_at = started_at
        speech_started_after_ms = 0
        resumed_after_pause_count = 0
        pre_speech_chunk_count = 0
        pre_speech_active_chunk_count = 0
        pre_speech_total_rms = 0
        pre_speech_peak_rms = 0
        chunk_bytes = _chunk_byte_count(
            sample_rate=self.sample_rate,
            channels=self.channels,
            chunk_ms=self.chunk_ms,
        )
        read_stall_timeout_s = _compute_read_stall_timeout(self.chunk_ms)
        respeaker_recovery_attempted = False

        def _listen_timeout_diagnostics(*, listened_ms: int) -> ListenTimeoutCaptureDiagnostics:
            average_rms = 0
            if pre_speech_chunk_count > 0:
                average_rms = int(pre_speech_total_rms / pre_speech_chunk_count)
            return ListenTimeoutCaptureDiagnostics(
                device=self.device,
                sample_rate=self.sample_rate,
                channels=self.channels,
                chunk_ms=self.chunk_ms,
                speech_threshold=self.speech_threshold,
                chunk_count=pre_speech_chunk_count,
                active_chunk_count=pre_speech_active_chunk_count,
                average_rms=average_rms,
                peak_rms=pre_speech_peak_rms,
                listened_ms=max(0, int(listened_ms)),
            )

        try:
            while True:
                now = time.monotonic()
                # AUDIT-FIX(#11): Honor cancellation before and after speech start so TALK-button releases can abort promptly.
                stop_requested = should_stop is not None and should_stop()
                if stop_requested:
                    if heard_speech:
                        break
                    raise RuntimeError("Audio capture stopped before speech started")
                if not heard_speech and now - started_at >= effective_start_timeout_s:
                    raise SpeechStartTimeoutError(
                        "No speech detected before timeout",
                        diagnostics=_listen_timeout_diagnostics(
                            listened_ms=(now - started_at) * 1000.0,
                        ),
                    )
                # AUDIT-FIX(#2): Enforce max recording length from speech start, not from when the microphone opened.
                if (
                    heard_speech
                    and speech_started_at is not None
                    and now - speech_started_at >= effective_max_record_seconds
                ):
                    break
                if process.poll() is not None:
                    if (
                        not heard_speech
                        and not captured
                        and not respeaker_recovery_attempted
                    ):
                        respeaker_recovery_attempted = True
                        recovered = wait_for_transient_respeaker_capture_ready(
                            device=self.device,
                            sample_rate=self.sample_rate,
                            channels=self.channels,
                            chunk_ms=self.chunk_ms,
                            should_stop=should_stop,
                        )
                        if recovered:
                            self._stop_process(process)
                            process = _spawn_audio_process(
                                command,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                purpose="Audio capture",
                            )
                            preroll.clear()
                            consecutive_speech_chunks = 0
                            consecutive_resume_chunks = 0
                            last_chunk_at = time.monotonic()
                            continue
                        if should_stop is not None and should_stop():
                            raise RuntimeError("Audio capture stopped before speech started")
                    self._raise_process_error(process)
                if process.stdout is None:
                    raise RuntimeError("arecord did not expose stdout")

                is_ready = _wait_for_readable(
                    process.stdout,
                    timeout_s=_SELECT_TIMEOUT_S,
                    purpose="Audio capture",
                )
                if not is_ready:
                    if time.monotonic() - last_chunk_at >= read_stall_timeout_s:
                        raise RuntimeError("Audio capture stalled while waiting for microphone data")
                    if heard_speech and pause_candidate_deadline_at is not None and time.monotonic() >= pause_candidate_deadline_at:
                        break
                    continue

                chunk = os.read(process.stdout.fileno(), chunk_bytes)
                if not chunk:
                    if time.monotonic() - last_chunk_at >= read_stall_timeout_s:
                        raise RuntimeError("Audio capture stalled while reading microphone data")
                    continue
                last_chunk_at = time.monotonic()
                chunk = _trim_incomplete_bytes(chunk, alignment=_SAMPLE_WIDTH_BYTES)
                if not chunk:
                    continue

                rms = _pcm16_rms(chunk)
                now = time.monotonic()

                if not heard_speech:
                    if normalized_ignore_initial_ms > 0 and (now - started_at) * 1000 < normalized_ignore_initial_ms:
                        preroll.clear()
                        consecutive_speech_chunks = 0
                        continue
                    if on_chunk is not None:
                        on_chunk(chunk)
                    pre_speech_chunk_count += 1
                    pre_speech_total_rms += rms
                    pre_speech_peak_rms = max(pre_speech_peak_rms, rms)
                    if rms >= self.speech_threshold:
                        pre_speech_active_chunk_count += 1
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
                        confirmed_resume, consecutive_resume_chunks = resolve_pause_resume_confirmation(
                            consecutive_resume_chunks=consecutive_resume_chunks,
                            required_resume_chunks=self.pause_resume_chunks,
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
                    base_pause_ms=normalized_pause_ms,
                    base_pause_grace_ms=normalized_pause_grace_ms,
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

        if not heard_speech:
            raise SpeechStartTimeoutError(
                "Speech capture ended without usable audio",
                diagnostics=_listen_timeout_diagnostics(
                    listened_ms=(time.monotonic() - started_at) * 1000.0,
                ),
            )
        if not captured:
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
        """Record PCM16 speech until pause detection stops the capture."""

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
        message = _process_failure_message(process, default_action="Audio capture")
        raise RuntimeError(f"Audio capture failed: {message}")

    def _stop_process(self, process: subprocess.Popen[bytes]) -> None:
        _stop_process(process)


class AmbientAudioSampler:
    """Sample short ambient microphone windows for loudness analysis."""

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
        # AUDIT-FIX(#7): Validate ambient sampler config up front to avoid opaque ALSA and math failures at runtime.
        self.device = _normalize_audio_device(device)
        self.sample_rate = _ensure_int("sample_rate", sample_rate, minimum=1)
        self.channels = _ensure_int("channels", channels, minimum=1)
        self.chunk_ms = max(20, _ensure_int("chunk_ms", chunk_ms, minimum=1))
        self.speech_threshold = _ensure_int("speech_threshold", speech_threshold, minimum=0)
        self.default_duration_ms = max(self.chunk_ms, _ensure_int("default_duration_ms", default_duration_ms, minimum=1))

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "AmbientAudioSampler":
        """Build an ambient sampler from ``TwinrConfig`` values."""

        # AUDIT-FIX(#7): Normalize config-provided device names before accessing ALSA.
        device = _normalize_audio_device(config.proactive_audio_input_device or config.audio_input_device)
        return cls(
            device=device,
            sample_rate=config.audio_sample_rate,
            channels=config.audio_channels,
            chunk_ms=config.audio_chunk_ms,
            speech_threshold=config.audio_speech_threshold,
            default_duration_ms=config.proactive_audio_sample_ms,
        )

    def sample_levels(self, *, duration_ms: int | None = None) -> AmbientAudioLevelSample:
        """Sample ambient audio and return only the aggregate loudness metrics."""

        return self.sample_window(duration_ms=duration_ms).sample

    def require_readable_frames(
        self,
        *,
        duration_ms: int | None = None,
        chunk_count: int = 1,
    ) -> AudioCaptureReadinessProbe:
        """Assert that the configured capture path yields readable PCM frames."""

        effective_duration_ms = self._resolve_duration_ms(duration_ms)
        normalized_chunk_count = _ensure_int("chunk_count", chunk_count, minimum=1)
        chunks = self._capture_chunks(
            target_chunk_count=normalized_chunk_count,
            effective_duration_ms=effective_duration_ms,
        )
        return AudioCaptureReadinessProbe(
            device=self.device,
            sample_rate=self.sample_rate,
            channels=self.channels,
            chunk_ms=self.chunk_ms,
            duration_ms=effective_duration_ms,
            target_chunk_count=normalized_chunk_count,
            captured_chunk_count=len(chunks),
            captured_bytes=sum(len(chunk) for chunk in chunks),
            detail=(
                f"Captured {len(chunks)} readable chunk(s) from {self.device} "
                f"at {self.sample_rate} Hz/{self.channels} channel(s)."
            ),
        )

    def sample_window(self, *, duration_ms: int | None = None) -> AmbientAudioCaptureWindow:
        """Sample ambient audio and return both metrics and captured PCM bytes."""

        effective_duration_ms = self._resolve_duration_ms(duration_ms)
        pcm_fragments = self._capture_chunks(
            target_chunk_count=max(1, math.ceil(effective_duration_ms / self.chunk_ms)),
            effective_duration_ms=effective_duration_ms,
        )
        rms_values = [_pcm16_rms(chunk) for chunk in pcm_fragments]

        active_chunk_count = sum(1 for rms in rms_values if rms >= self.speech_threshold)
        average_rms = int(sum(rms_values) / len(rms_values))
        peak_rms = max(rms_values)
        # AUDIT-FIX(#10): Report the actual sampled duration instead of the requested nominal duration.
        actual_duration_ms = len(rms_values) * self.chunk_ms
        sample = AmbientAudioLevelSample(
            duration_ms=actual_duration_ms,
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
        message = _process_failure_message(process, default_action="Ambient audio capture")
        raise RuntimeError(f"Ambient audio capture failed: {message}")

    def _stop_process(self, process: subprocess.Popen[bytes]) -> None:
        _stop_process(process)

    def _resolve_duration_ms(self, duration_ms: int | None) -> int:
        """Normalize one optional sample duration to a bounded chunk-aligned window."""

        if duration_ms is None:
            normalized_duration_ms = self.default_duration_ms
        else:
            requested_duration_ms = _ensure_int("duration_ms", duration_ms, minimum=0)
            normalized_duration_ms = (
                self.default_duration_ms if requested_duration_ms == 0 else requested_duration_ms
            )
        return max(self.chunk_ms, normalized_duration_ms)

    def _capture_chunks(
        self,
        *,
        target_chunk_count: int,
        effective_duration_ms: int,
    ) -> list[bytes]:
        """Collect a bounded number of readable PCM chunks from the current device."""

        normalized_target_chunk_count = _ensure_int(
            "target_chunk_count",
            target_chunk_count,
            minimum=1,
        )
        chunk_bytes = _chunk_byte_count(
            sample_rate=self.sample_rate,
            channels=self.channels,
            chunk_ms=self.chunk_ms,
        )
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
        process = _spawn_audio_process(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            purpose="Ambient audio capture",
        )
        pcm_fragments: list[bytes] = []
        bytes_captured = 0
        started_at = time.monotonic()
        last_chunk_at = started_at
        capture_deadline_at = started_at + (effective_duration_ms / 1000.0) + _AMBIENT_CAPTURE_EXTRA_TIMEOUT_S
        read_stall_timeout_s = _compute_read_stall_timeout(self.chunk_ms)

        def _raise_capture_readiness_error(
            *,
            failure_reason: str,
            detail: str,
        ) -> None:
            raise AudioCaptureReadinessError(
                detail,
                probe=AudioCaptureReadinessProbe(
                    device=self.device,
                    sample_rate=self.sample_rate,
                    channels=self.channels,
                    chunk_ms=self.chunk_ms,
                    duration_ms=effective_duration_ms,
                    target_chunk_count=normalized_target_chunk_count,
                    captured_chunk_count=len(pcm_fragments),
                    captured_bytes=bytes_captured,
                    failure_reason=failure_reason,
                    detail=detail,
                ),
            )

        try:
            while len(pcm_fragments) < normalized_target_chunk_count:
                now = time.monotonic()
                # AUDIT-FIX(#3): Add absolute and stall timeouts so ambient sampling cannot hang forever on dead audio input.
                if now >= capture_deadline_at:
                    _raise_capture_readiness_error(
                        failure_reason="timed_out",
                        detail="Ambient audio capture timed out before collecting enough samples",
                    )
                if process.poll() is not None:
                    _raise_capture_readiness_error(
                        failure_reason="process_exited",
                        detail=(
                            "Ambient audio capture failed: "
                            f"{_process_failure_message(process, default_action='Ambient audio capture')}"
                        ),
                    )
                if process.stdout is None:
                    _raise_capture_readiness_error(
                        failure_reason="stdout_missing",
                        detail="arecord did not expose stdout",
                    )
                is_ready = _wait_for_readable(
                    process.stdout,
                    timeout_s=_SELECT_TIMEOUT_S,
                    purpose="Ambient audio capture",
                )
                if not is_ready:
                    if time.monotonic() - last_chunk_at >= read_stall_timeout_s:
                        _raise_capture_readiness_error(
                            failure_reason="stalled_waiting",
                            detail="Ambient audio capture stalled while waiting for microphone data",
                        )
                    continue
                chunk = os.read(process.stdout.fileno(), chunk_bytes)
                if not chunk:
                    if time.monotonic() - last_chunk_at >= read_stall_timeout_s:
                        _raise_capture_readiness_error(
                            failure_reason="stalled_reading",
                            detail="Ambient audio capture stalled while reading microphone data",
                        )
                    continue
                last_chunk_at = time.monotonic()
                chunk = _trim_incomplete_bytes(chunk, alignment=_SAMPLE_WIDTH_BYTES)
                if not chunk:
                    continue
                pcm_fragments.append(chunk)
                bytes_captured += len(chunk)
        finally:
            self._stop_process(process)

        if not pcm_fragments:
            _raise_capture_readiness_error(
                failure_reason="no_usable_samples",
                detail="Ambient audio capture ended without usable samples",
            )
        return pcm_fragments


class WaveAudioPlayer:
    """Play Twinr WAV, PCM, and synthesized tone output through ALSA."""

    def __init__(self, *, device: str = "default") -> None:
        # AUDIT-FIX(#7): Normalize playback device names early so blank config values still resolve safely.
        self.device = _normalize_audio_device(device)
        self._active_process_lock = Lock()
        self._active_process: subprocess.Popen[bytes] | None = None

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "WaveAudioPlayer":
        """Build a playback adapter from ``TwinrConfig`` values."""

        return cls(device=config.audio_output_device)

    def play_wav_bytes(self, audio_bytes: bytes) -> None:
        """Play an in-memory WAV file through the configured output device."""

        # AUDIT-FIX(#1): Use a unique secure temp file instead of a fixed /tmp path to prevent symlink attacks and concurrent clobbering.
        with tempfile.NamedTemporaryFile(prefix="twinr-playback-", suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = Path(temp_file.name)
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
        """Play one synthesized confirmation tone."""

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
        """Play a short sequence of synthesized tones."""

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

    def play_wav_chunks(
        self,
        chunks: Iterable[bytes],
        *,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        """Stream WAV chunks to ``aplay`` until playback completes or stops."""

        self._play_stream(
            ["aplay", "-q", "-D", self.device, "-t", "wav", "-"],
            chunks,
            should_stop=should_stop,
        )

    def play_pcm16_chunks(
        self,
        chunks: Iterable[bytes],
        *,
        sample_rate: int,
        channels: int = 1,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        """Stream raw PCM16 chunks to ``aplay`` with explicit format settings."""

        normalized_sample_rate = _ensure_int("sample_rate", sample_rate, minimum=1)
        normalized_channels = _ensure_int("channels", channels, minimum=1)
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
                str(normalized_channels),
                "-r",
                str(normalized_sample_rate),
                "-",
            ],
            chunks,
            should_stop=should_stop,
        )

    def _play_stream(
        self,
        command: list[str],
        chunks: Iterable[bytes],
        *,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        process = _spawn_audio_process(
            command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            purpose="Audio playback",
        )
        self._set_active_process(process)
        stopped_early = False
        try:
            if process.stdin is None:
                raise RuntimeError("aplay did not expose stdin")
            stdin_fd = process.stdin.fileno()
            os.set_blocking(stdin_fd, False)
            # AUDIT-FIX(#4): Write to aplay with backpressure-aware, timeout-bounded non-blocking I/O.
            for chunk in chunks:
                if should_stop is not None and should_stop():
                    stopped_early = True
                    break
                if not chunk:
                    continue
                view = memoryview(chunk)
                while view:
                    if should_stop is not None and should_stop():
                        stopped_early = True
                        view = view[:0]
                        break
                    if process.poll() is not None:
                        self._raise_stream_error(process)
                    if not _wait_for_writable(
                        stdin_fd,
                        timeout_s=_STREAM_IO_STALL_TIMEOUT_S,
                        purpose="Audio playback",
                    ):
                        raise RuntimeError("Audio playback timed out while waiting for output device")
                    try:
                        written = os.write(stdin_fd, view)
                    except BlockingIOError:
                        continue
                    except BrokenPipeError:
                        self._raise_stream_error(process)
                    except OSError as exc:
                        raise RuntimeError(f"Audio playback failed while streaming: {exc}") from exc
                    if written <= 0:
                        self._raise_stream_error(process)
                    view = view[written:]
            if not stopped_early and should_stop is not None and should_stop():
                stopped_early = True
            if stopped_early:
                self._stop_process(process)
                return
            try:
                process.stdin.close()
            except BrokenPipeError:
                self._raise_stream_error(process)
            except OSError as exc:
                raise RuntimeError(f"Audio playback failed while closing stream: {exc}") from exc
            process.wait(timeout=_PLAYBACK_FINALIZE_TIMEOUT_S)
            if process.returncode != 0:
                self._raise_stream_error(process)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Audio playback timed out while waiting for aplay to finish") from exc
        finally:
            self._clear_active_process(process)
            self._stop_process(process)

    def play_file(self, path: str | Path) -> None:
        """Play one existing audio file through ``aplay``."""

        path_obj = Path(path)
        if not path_obj.is_file():
            raise RuntimeError(f"Audio playback file not found: {path_obj}")
        try:
            # AUDIT-FIX(#4): Bound file playback runtime so a broken output device cannot block the process forever.
            result = subprocess.run(
                ["aplay", "-q", "-D", self.device, str(path_obj)],
                capture_output=True,
                check=False,
                timeout=_PLAYBACK_FILE_TIMEOUT_S,
            )
        except FileNotFoundError as exc:
            # AUDIT-FIX(#5): Surface missing playback binaries as explicit runtime errors.
            raise RuntimeError("Audio playback failed because 'aplay' is not installed") from exc
        except OSError as exc:
            # AUDIT-FIX(#5): Surface playback start failures with actionable context.
            raise RuntimeError(f"Audio playback failed to start: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            # AUDIT-FIX(#4): Convert subprocess timeout into a clear playback error.
            raise RuntimeError("Audio playback timed out") from exc
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(f"Audio playback failed: {stderr or result.returncode}")

    def _raise_stream_error(self, process: subprocess.Popen[bytes]) -> None:
        message = _process_failure_message(process, default_action="Audio playback")
        raise RuntimeError(f"Audio playback failed: {message}")

    def stop_playback(self) -> None:
        """Stop the currently active playback process, if any."""

        with self._active_process_lock:
            process = self._active_process
        if process is None:
            return
        self._stop_process(process)

    def _set_active_process(self, process: subprocess.Popen[bytes]) -> None:
        with self._active_process_lock:
            self._active_process = process

    def _clear_active_process(self, process: subprocess.Popen[bytes]) -> None:
        with self._active_process_lock:
            if self._active_process is process:
                self._active_process = None

    def _render_tone_pcm(
        self,
        *,
        frequency_hz: int,
        duration_ms: int,
        volume: float,
        sample_rate: int,
    ) -> bytes:
        # AUDIT-FIX(#7): Validate tone parameters so invalid caller input cannot cause divide-by-zero or nonsense playback.
        normalized_frequency_hz = _ensure_int("frequency_hz", frequency_hz, minimum=0)
        normalized_duration_ms = _ensure_int("duration_ms", duration_ms, minimum=0)
        normalized_sample_rate = _ensure_int("sample_rate", sample_rate, minimum=1)
        # Keep some headroom, but do not throw away most of the available
        # speaker range on the Pi confirmation-beep path.
        amplitude = max(0.0, min(float(volume), 1.0)) * 32767.0 * _TONE_HEADROOM_SCALE
        if normalized_duration_ms == 0 or amplitude == 0.0:
            return b""
        frame_count = int(normalized_sample_rate * (normalized_duration_ms / 1000.0))
        if frame_count <= 0:
            return b""
        pcm = bytearray()
        for frame_index in range(frame_count):
            sample = int(
                amplitude
                * math.sin(
                    2.0 * math.pi * normalized_frequency_hz * frame_index / normalized_sample_rate
                )
            )
            pcm.extend(sample.to_bytes(2, byteorder="little", signed=True))
        return bytes(pcm)

    def _render_silence_pcm(self, *, duration_ms: int, sample_rate: int) -> bytes:
        normalized_duration_ms = _ensure_int("duration_ms", duration_ms, minimum=0)
        normalized_sample_rate = _ensure_int("sample_rate", sample_rate, minimum=1)
        if normalized_duration_ms == 0:
            return b""
        frame_count = int(normalized_sample_rate * (normalized_duration_ms / 1000.0))
        if frame_count <= 0:
            return b""
        return b"\x00\x00" * frame_count

    def _stop_process(self, process: subprocess.Popen[bytes]) -> None:
        _stop_process(process)
