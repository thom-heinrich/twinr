# CHANGELOG: 2026-03-28
# BUG-1: Removed the hard dependency on stdlib audioop so the module imports on Python 3.13+.
# BUG-2: Fixed capture logic that treated short pipe reads as full analysis chunks, which skewed VAD, preroll, and ambient metrics.
# BUG-3: Fixed file playback to use the bounded ALSA subprocess environment and active-process tracking, so stop_playback() works for file-backed playback too.
# BUG-4: Fixed PCM alignment to full frames, preventing truncated multi-channel frames from leaking into capture/playback paths.
# BUG-5: Reassert the ReSpeaker XVF3800 playback mixer during runtime player construction so spoken replies do not stay whisper-quiet after Linux mixer drift.
# SEC-1: Removed temp-file-based play_wav_bytes() disk amplification and added bounded streamed-playback byte limits to prevent practical SD-card/tmp exhaustion and endless remote audio streams on Raspberry Pi deployments.
# IMP-1: Added optional hybrid WebRTC VAD + adaptive noise floor endpointing with graceful RMS fallback for Pi-class edge devices.
# IMP-2: Added chunk-accurate PCM buffering and more accurate duration accounting so downstream speech timing is stable under pipe fragmentation.
# IMP-3: Added extension-friendly VAD configuration hooks via optional config attributes while keeping the existing public API drop-in compatible.

"""Capture and play bounded audio for Twinr hardware workflows."""

from __future__ import annotations

import io
import math
import os
import sys
import hashlib
import select
import subprocess
import time
import wave
from array import array
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, BinaryIO, Callable, cast

try:
    import audioop as _audioop  # type: ignore[import-not-found]
except ModuleNotFoundError:
    _audioop = None

try:
    import webrtcvad as _webrtcvad  # type: ignore[import-not-found]
except ModuleNotFoundError:
    _webrtcvad = None

from twinr.hardware.audio_env import build_audio_subprocess_env_for_mode
from twinr.hardware.respeaker_duplex_playback import (
    maybe_open_respeaker_duplex_playback_guard,
    resolve_respeaker_duplex_playback_sample_rate_hz,
)
from twinr.hardware.respeaker_playback_mixer import ensure_respeaker_playback_mixer

if TYPE_CHECKING:
    from twinr.agent.base_agent.config import TwinrConfig


_SAMPLE_WIDTH_BYTES = 2
_PCM16_MIN = -32768
_PCM16_MAX = 32767
_PCM16_MAX_ABS = 32767

_SELECT_TIMEOUT_S = 0.25
_AMBIENT_CAPTURE_EXTRA_TIMEOUT_S = 2.0
_STREAM_IO_STALL_TIMEOUT_S = 5.0
_PLAYBACK_FINALIZE_TIMEOUT_S = 10.0
_PLAYBACK_FILE_TIMEOUT_S = 120.0
_PROCESS_STOP_TIMEOUT_S = 1.0
_PROCESS_PIPE_READ_SIZE_BYTES = 8192

_DEFAULT_WAV_TARGET_PEAK = 28000
_DEFAULT_WAV_MAX_GAIN = 4.0
_TONE_HEADROOM_SCALE = 0.92

_GENERIC_CAPTURE_DEVICE_ALIASES = frozenset({"default", "pulse", "sysdefault"})
_CANONICAL_CAPTURE_PREFIXES = frozenset({"dsnoop", "front", "hw", "plughw", "sysdefault"})

_DEFAULT_VAD_MODE = "auto"
_DEFAULT_WEBRTC_VAD_AGGRESSIVENESS = 2
_DEFAULT_WEBRTC_VAD_FRAME_MS = 20
_DEFAULT_WEBRTC_VAD_MIN_VOICED_FRAME_RATIO = 0.40
_DEFAULT_ADAPTIVE_NOISE_MULTIPLIER = 1.8
_DEFAULT_ADAPTIVE_NOISE_MARGIN_RMS = 180
_DEFAULT_MAX_PLAYBACK_STREAM_BYTES = 64 * 1024 * 1024

_WEBRTC_VAD_VALID_SAMPLE_RATES = frozenset({8000, 16000, 32000, 48000})
_WEBRTC_VAD_VALID_FRAME_MS = frozenset({10, 20, 30})


def _normalize_audio_device(device: str | None) -> str:
    normalized = str(device or "default").strip()
    return normalized or "default"


def resolve_capture_device(*candidates: str | None) -> str:
    normalized_candidates = [
        str(candidate or "").strip()
        for candidate in candidates
        if str(candidate or "").strip()
    ]
    if not normalized_candidates:
        return "default"
    first = normalized_candidates[0]
    if first.lower() not in _GENERIC_CAPTURE_DEVICE_ALIASES:
        return first
    for candidate in normalized_candidates[1:]:
        if candidate.lower() not in _GENERIC_CAPTURE_DEVICE_ALIASES:
            return candidate
    return first


def capture_device_identity(device: str | None) -> str:
    normalized = _normalize_audio_device(device)
    prefix, separator, suffix = normalized.partition(":")
    normalized_prefix = prefix.strip().lower()
    if not separator or normalized_prefix not in _CANONICAL_CAPTURE_PREFIXES:
        return normalized.lower()
    tokens: dict[str, str] = {}
    for section in suffix.split(":"):
        for field in section.split(","):
            field = field.strip()
            if "=" not in field:
                continue
            key, value = field.split("=", 1)
            key = key.strip().upper()
            value = value.strip().lower()
            if key and value:
                tokens[key] = value
    card = tokens.get("CARD")
    if not card:
        return normalized.lower()
    device_index = tokens.get("DEV") or "0"
    return f"alsa-card={card};dev={device_index}"


def _ensure_int(name: str, value: object, *, minimum: int) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if normalized < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return normalized


def _ensure_float(name: str, value: object, *, minimum: float) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a float >= {minimum}") from exc
    if normalized < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return normalized


def _ensure_ratio(
    name: str,
    value: object,
    *,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a float in [{minimum}, {maximum}]") from exc
    if normalized < minimum or normalized > maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return normalized


def _normalize_vad_mode(mode: str | None) -> str:
    normalized = str(mode or _DEFAULT_VAD_MODE).strip().lower()
    if normalized not in {"auto", "rms", "webrtc", "hybrid"}:
        raise ValueError("vad_mode must be one of: auto, rms, webrtc, hybrid")
    return normalized


def _normalize_webrtc_frame_ms(frame_ms: int) -> int:
    normalized = _ensure_int("vad_frame_ms", frame_ms, minimum=1)
    if normalized in _WEBRTC_VAD_VALID_FRAME_MS:
        return normalized
    return min(_WEBRTC_VAD_VALID_FRAME_MS, key=lambda candidate: abs(candidate - normalized))


def _resolve_effective_vad_frame_ms(*, chunk_ms: int, requested_frame_ms: int) -> int:
    normalized_chunk_ms = _ensure_int("chunk_ms", chunk_ms, minimum=1)
    normalized_frame_ms = _normalize_webrtc_frame_ms(requested_frame_ms)
    eligible_frames = sorted(frame_ms for frame_ms in _WEBRTC_VAD_VALID_FRAME_MS if frame_ms <= normalized_chunk_ms)
    if not eligible_frames:
        return normalized_frame_ms
    if normalized_frame_ms in eligible_frames:
        return normalized_frame_ms
    return min(eligible_frames, key=lambda candidate: abs(candidate - normalized_frame_ms))


def _trim_incomplete_bytes(payload: bytes, *, alignment: int) -> bytes:
    if alignment <= 1 or not payload:
        return payload
    usable_length = len(payload) - (len(payload) % alignment)
    if usable_length <= 0:
        return b""
    return payload[:usable_length]


def _bytes_per_frame(channels: int) -> int:
    return _SAMPLE_WIDTH_BYTES * _ensure_int("channels", channels, minimum=1)


def _pcm16_array_from_bytes(samples: bytes) -> array[int]:
    aligned_samples = _trim_incomplete_bytes(samples, alignment=_SAMPLE_WIDTH_BYTES)
    pcm_samples = array("h")
    if not aligned_samples:
        return pcm_samples
    pcm_samples.frombytes(aligned_samples)
    if sys.byteorder != "little":
        pcm_samples.byteswap()
    return pcm_samples


def _pcm16_bytes_from_array(pcm_samples: array[int]) -> bytes:
    output = array("h", pcm_samples)
    if sys.byteorder != "little":
        output.byteswap()
    return output.tobytes()


def _pcm16_peak_abs(samples: bytes) -> int:
    if not samples:
        return 0
    aligned_samples = _trim_incomplete_bytes(samples, alignment=_SAMPLE_WIDTH_BYTES)
    if not aligned_samples:
        return 0
    if _audioop is not None:
        return int(_audioop.max(aligned_samples, _SAMPLE_WIDTH_BYTES))
    pcm_samples = _pcm16_array_from_bytes(aligned_samples)
    return max((min(abs(sample), _PCM16_MAX_ABS) for sample in pcm_samples), default=0)


def _pcm16_mul_gain(samples: bytes, gain: float) -> bytes:
    if not samples:
        return samples
    aligned_samples = _trim_incomplete_bytes(samples, alignment=_SAMPLE_WIDTH_BYTES)
    if not aligned_samples:
        return b""
    if gain == 1.0:
        return aligned_samples
    if _audioop is not None:
        return _audioop.mul(aligned_samples, _SAMPLE_WIDTH_BYTES, gain)
    pcm_samples = _pcm16_array_from_bytes(aligned_samples)
    scaled = array(
        "h",
        (
            max(_PCM16_MIN, min(_PCM16_MAX, int(round(sample * gain))))
            for sample in pcm_samples
        ),
    )
    return _pcm16_bytes_from_array(scaled)


def _pcm16_downmix_to_mono(samples: bytes, *, channels: int) -> bytes:
    normalized_channels = _ensure_int("channels", channels, minimum=1)
    if normalized_channels == 1:
        return _trim_incomplete_bytes(samples, alignment=_SAMPLE_WIDTH_BYTES)
    frame_bytes = _bytes_per_frame(normalized_channels)
    aligned_samples = _trim_incomplete_bytes(samples, alignment=frame_bytes)
    if not aligned_samples:
        return b""
    pcm_samples = _pcm16_array_from_bytes(aligned_samples)
    mono_samples = array("h")
    for index in range(0, len(pcm_samples), normalized_channels):
        frame = pcm_samples[index:index + normalized_channels]
        mono_samples.append(int(sum(frame) / normalized_channels))
    return _pcm16_bytes_from_array(mono_samples)


def _pcm16_resample_linear(samples: bytes, *, from_rate: int, to_rate: int) -> bytes:
    normalized_from_rate = _ensure_int("from_rate", from_rate, minimum=1)
    normalized_to_rate = _ensure_int("to_rate", to_rate, minimum=1)
    aligned_samples = _trim_incomplete_bytes(samples, alignment=_SAMPLE_WIDTH_BYTES)
    if not aligned_samples or normalized_from_rate == normalized_to_rate:
        return aligned_samples
    pcm_samples = _pcm16_array_from_bytes(aligned_samples)
    if not pcm_samples:
        return b""
    source_count = len(pcm_samples)
    target_count = max(1, int(round(source_count * normalized_to_rate / normalized_from_rate)))
    if target_count == source_count:
        return aligned_samples
    if source_count == 1:
        return _pcm16_bytes_from_array(array("h", [pcm_samples[0]] * target_count))
    scale = (source_count - 1) / max(1, target_count - 1)
    resampled = array("h")
    for index in range(target_count):
        position = index * scale
        left_index = int(position)
        right_index = min(source_count - 1, left_index + 1)
        fraction = position - left_index
        sample_value = int(
            round(
                pcm_samples[left_index] * (1.0 - fraction)
                + pcm_samples[right_index] * fraction
            )
        )
        resampled.append(max(_PCM16_MIN, min(_PCM16_MAX, sample_value)))
    return _pcm16_bytes_from_array(resampled)


def _compute_read_stall_timeout(chunk_ms: int) -> float:
    normalized_chunk_ms = _ensure_int("chunk_ms", chunk_ms, minimum=1)
    return max(2.0, (normalized_chunk_ms / 1000.0) * 4.0)


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


def _spawn_audio_process(
    command: list[str],
    *,
    stdin: int | None = None,
    stdout: int | None = None,
    stderr: int | None = None,
    purpose: str,
    allow_root_borrowed_session_audio: bool = False,
) -> subprocess.Popen[bytes]:
    try:
        return subprocess.Popen(
            command,
            env=build_audio_subprocess_env_for_mode(
                allow_root_borrowed_session_audio=allow_root_borrowed_session_audio,
            ),
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
        pass


def _read_process_stderr(process: subprocess.Popen[bytes]) -> bytes:
    stderr = process.stderr
    if stderr is None or getattr(stderr, "closed", False):
        return b""
    try:
        return stderr.read().strip()
    except (OSError, ValueError):
        return b""


def _process_failure_message(process: subprocess.Popen[bytes], *, default_action: str) -> str:
    stderr = _read_process_stderr(process)
    if stderr:
        return stderr.decode("utf-8", errors="ignore")
    if process.returncode == 0:
        return f"{default_action} ended unexpectedly"
    return f"exit code {process.returncode}"


def _stop_process(process: subprocess.Popen[bytes]) -> None:
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


def _pcm16_rms(samples: bytes) -> int:
    if not samples:
        return 0
    aligned_samples = _trim_incomplete_bytes(samples, alignment=_SAMPLE_WIDTH_BYTES)
    if not aligned_samples:
        return 0
    pcm_samples = _pcm16_array_from_bytes(aligned_samples)
    if not pcm_samples:
        return 0
    mean_square = sum(sample * sample for sample in pcm_samples) / len(pcm_samples)
    return int(math.sqrt(mean_square))


@dataclass(frozen=True, slots=True)
class _SpeechActivityAssessment:
    rms: int
    active: bool
    energy_active: bool
    vad_active: bool
    effective_threshold: int
    noise_floor_rms: int


class _SpeechActivityDetector:
    def __init__(
        self,
        *,
        sample_rate: int,
        channels: int,
        speech_threshold: int,
        vad_mode: str = _DEFAULT_VAD_MODE,
        vad_webrtc_aggressiveness: int = _DEFAULT_WEBRTC_VAD_AGGRESSIVENESS,
        adaptive_noise_enabled: bool = True,
        adaptive_noise_multiplier: float = _DEFAULT_ADAPTIVE_NOISE_MULTIPLIER,
        adaptive_noise_margin_rms: int = _DEFAULT_ADAPTIVE_NOISE_MARGIN_RMS,
        vad_frame_ms: int = _DEFAULT_WEBRTC_VAD_FRAME_MS,
        vad_min_voiced_frame_ratio: float = _DEFAULT_WEBRTC_VAD_MIN_VOICED_FRAME_RATIO,
    ) -> None:
        self.sample_rate = _ensure_int("sample_rate", sample_rate, minimum=1)
        self.channels = _ensure_int("channels", channels, minimum=1)
        self.speech_threshold = _ensure_int("speech_threshold", speech_threshold, minimum=0)

        requested_mode = _normalize_vad_mode(vad_mode)
        self.adaptive_noise_enabled = bool(adaptive_noise_enabled)
        self.adaptive_noise_multiplier = max(1.0, float(adaptive_noise_multiplier))
        self.adaptive_noise_margin_rms = _ensure_int(
            "adaptive_noise_margin_rms",
            adaptive_noise_margin_rms,
            minimum=0,
        )
        self.vad_frame_ms = _normalize_webrtc_frame_ms(vad_frame_ms)
        self.vad_min_voiced_frame_ratio = _ensure_ratio(
            "vad_min_voiced_frame_ratio",
            vad_min_voiced_frame_ratio,
        )

        self._noise_floor_rms: float | None = None
        self._vad = None

        effective_mode = requested_mode
        if effective_mode == "auto":
            effective_mode = "hybrid" if _webrtcvad is not None else "rms"

        if effective_mode in {"webrtc", "hybrid"} and _webrtcvad is not None:
            aggressiveness = min(
                3,
                max(
                    0,
                    _ensure_int(
                        "vad_webrtc_aggressiveness",
                        vad_webrtc_aggressiveness,
                        minimum=0,
                    ),
                ),
            )
            try:
                self._vad = _webrtcvad.Vad(aggressiveness)
            except TypeError:
                self._vad = _webrtcvad.Vad()
                self._vad.set_mode(aggressiveness)
        elif effective_mode == "webrtc":
            effective_mode = "rms"

        self.vad_mode = effective_mode
        self._vad_sample_rate = (
            self.sample_rate if self.sample_rate in _WEBRTC_VAD_VALID_SAMPLE_RATES else 16000
        )
        self._vad_frame_bytes = int(
            self._vad_sample_rate * (self.vad_frame_ms / 1000.0) * _SAMPLE_WIDTH_BYTES
        )

    def assess(self, chunk: bytes, *, update_noise_floor: bool = True) -> _SpeechActivityAssessment:
        aligned_chunk = _trim_incomplete_bytes(chunk, alignment=_bytes_per_frame(self.channels))
        rms = _pcm16_rms(aligned_chunk)

        effective_threshold = self.speech_threshold
        noise_floor_rms = int(self._noise_floor_rms) if self._noise_floor_rms is not None else 0

        if self.adaptive_noise_enabled and self._noise_floor_rms is not None:
            effective_threshold = max(
                effective_threshold,
                int(self._noise_floor_rms * self.adaptive_noise_multiplier),
                int(self._noise_floor_rms + self.adaptive_noise_margin_rms),
            )

        energy_active = rms >= effective_threshold
        vad_active = self._chunk_contains_speech(aligned_chunk)

        if self.vad_mode == "rms":
            active = energy_active
        elif self.vad_mode == "webrtc":
            active = vad_active
        else:
            if self._vad is not None and self._noise_floor_rms is None and not vad_active:
                active = False
            else:
                active = vad_active or energy_active

        if self.adaptive_noise_enabled and update_noise_floor and not vad_active:
            if self._noise_floor_rms is None:
                self._noise_floor_rms = float(rms)
            else:
                self._noise_floor_rms = (self._noise_floor_rms * 0.85) + (rms * 0.15)

            noise_floor_rms = int(self._noise_floor_rms)
            effective_threshold = max(
                self.speech_threshold,
                int(self._noise_floor_rms * self.adaptive_noise_multiplier),
                int(self._noise_floor_rms + self.adaptive_noise_margin_rms),
            )
            if self.vad_mode == "rms":
                energy_active = rms >= effective_threshold
                active = energy_active
            elif self.vad_mode == "hybrid" and self._vad is None:
                energy_active = rms >= effective_threshold
                active = energy_active

        return _SpeechActivityAssessment(
            rms=rms,
            active=active,
            energy_active=energy_active,
            vad_active=vad_active,
            effective_threshold=effective_threshold,
            noise_floor_rms=noise_floor_rms,
        )

    def _chunk_contains_speech(self, chunk: bytes) -> bool:
        if self._vad is None or not chunk:
            return False

        mono_chunk = _pcm16_downmix_to_mono(chunk, channels=self.channels)
        if not mono_chunk:
            return False

        if self.sample_rate != self._vad_sample_rate:
            mono_chunk = _pcm16_resample_linear(
                mono_chunk,
                from_rate=self.sample_rate,
                to_rate=self._vad_sample_rate,
            )

        aligned_chunk = _trim_incomplete_bytes(mono_chunk, alignment=self._vad_frame_bytes)
        if not aligned_chunk or self._vad_frame_bytes <= 0:
            return False

        frame_count = len(aligned_chunk) // self._vad_frame_bytes
        if frame_count <= 0:
            return False

        required_voiced_frames = max(1, math.ceil(frame_count * self.vad_min_voiced_frame_ratio))
        voiced_frames = 0
        for frame_index in range(frame_count):
            offset = frame_index * self._vad_frame_bytes
            frame = aligned_chunk[offset:offset + self._vad_frame_bytes]
            try:
                if self._vad.is_speech(frame, self._vad_sample_rate):
                    voiced_frames += 1
                    if voiced_frames >= required_voiced_frames:
                        return True
            except Exception:
                return False
        return False


class _PcmChunkBuffer:
    def __init__(self, *, target_chunk_bytes: int, frame_bytes: int) -> None:
        self._target_chunk_bytes = _ensure_int("target_chunk_bytes", target_chunk_bytes, minimum=1)
        self._frame_bytes = _ensure_int("frame_bytes", frame_bytes, minimum=1)
        self._buffer = bytearray()

    def append(self, payload: bytes) -> list[bytes]:
        if payload:
            self._buffer.extend(payload)
        emitted: list[bytes] = []
        aligned_length = len(self._buffer) - (len(self._buffer) % self._frame_bytes)
        while aligned_length >= self._target_chunk_bytes:
            emitted.append(bytes(self._buffer[:self._target_chunk_bytes]))
            del self._buffer[:self._target_chunk_bytes]
            aligned_length = len(self._buffer) - (len(self._buffer) % self._frame_bytes)
        return emitted

    def flush_aligned(self) -> bytes:
        aligned_length = len(self._buffer) - (len(self._buffer) % self._frame_bytes)
        if aligned_length <= 0:
            return b""
        payload = bytes(self._buffer[:aligned_length])
        del self._buffer[:aligned_length]
        return payload

    def clear(self) -> None:
        self._buffer.clear()


def normalize_wav_playback_level(
    audio_bytes: bytes,
    *,
    target_peak: int = _DEFAULT_WAV_TARGET_PEAK,
    max_gain: float = _DEFAULT_WAV_MAX_GAIN,
) -> bytes:
    if not audio_bytes:
        return audio_bytes

    normalized_target_peak = max(1, min(int(target_peak), _PCM16_MAX_ABS))
    normalized_max_gain = max(1.0, float(max_gain))

    try:
        with cast(wave.Wave_read, wave.open(io.BytesIO(audio_bytes), "rb")) as wave_reader:
            channels = wave_reader.getnchannels()
            sample_width = wave_reader.getsampwidth()
            sample_rate = wave_reader.getframerate()
            frames = wave_reader.readframes(wave_reader.getnframes())
    except (wave.Error, EOFError):
        return audio_bytes

    if sample_width != _SAMPLE_WIDTH_BYTES or not frames:
        return audio_bytes

    frames = _trim_incomplete_bytes(frames, alignment=_SAMPLE_WIDTH_BYTES * max(1, channels))
    if not frames:
        return audio_bytes

    peak = _pcm16_peak_abs(frames)
    if peak <= 0 or peak >= normalized_target_peak:
        return audio_bytes

    gain = min(normalized_max_gain, normalized_target_peak / float(peak))
    if gain <= 1.0:
        return audio_bytes

    boosted_frames = _pcm16_mul_gain(frames, gain)
    output_buffer = io.BytesIO()
    # Pylint resolves this context manager as Wave_read under Python 3.11 stubs.
    # The runtime mode is "wb", so these writer methods are valid here.
    # pylint: disable=no-member
    wave_writer: Any
    with wave.open(cast(BinaryIO, output_buffer), "wb") as wave_writer:
        wave_writer.setnchannels(channels)
        wave_writer.setsampwidth(sample_width)
        wave_writer.setframerate(sample_rate)
        wave_writer.writeframes(boosted_frames)
    # pylint: enable=no-member
    return output_buffer.getvalue()


@dataclass(frozen=True, slots=True)
class AmbientAudioLevelSample:
    duration_ms: int
    chunk_count: int
    active_chunk_count: int
    average_rms: int
    peak_rms: int
    active_ratio: float


@dataclass(frozen=True, slots=True)
class Pcm16SignalProfile:
    sample_count: int
    rms: int
    mean_abs: int
    peak_abs: int
    dc_offset: int
    nonzero_sample_count: int
    nonzero_sample_ratio: float
    clipped_sample_count: int
    clipped_sample_ratio: float
    zero_crossing_count: int
    zero_crossing_ratio: float
    sha256: str


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


@dataclass(frozen=True, slots=True)
class ListenTimeoutCaptureDiagnostics:
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
        return self.active_chunk_count / max(1, self.chunk_count)


@dataclass(frozen=True, slots=True)
class AudioCaptureReadinessProbe:
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
        return (
            self.failure_reason is None
            and self.captured_chunk_count >= self.target_chunk_count
            and self.captured_bytes > 0
        )


class SpeechStartTimeoutError(RuntimeError):
    def __init__(self, message: str, *, diagnostics: ListenTimeoutCaptureDiagnostics) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


class AudioCaptureReadinessError(RuntimeError):
    def __init__(self, message: str, *, probe: AudioCaptureReadinessProbe) -> None:
        super().__init__(message)
        self.probe = probe


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

    return (
        max(0, pause_ms - max(0, int(long_pause_penalty_ms))),
        max(0, pause_grace_ms - max(0, int(long_pause_grace_penalty_ms))),
    )


def pcm16_to_wav_bytes(
    pcm_bytes: bytes,
    *,
    sample_rate: int,
    channels: int,
) -> bytes:
    normalized_sample_rate = _ensure_int("sample_rate", sample_rate, minimum=1)
    normalized_channels = _ensure_int("channels", channels, minimum=1)
    aligned_pcm_bytes = _trim_incomplete_bytes(
        pcm_bytes,
        alignment=_SAMPLE_WIDTH_BYTES * normalized_channels,
    )
    buffer = io.BytesIO()
    # Pylint resolves this context manager as Wave_read under Python 3.11 stubs.
    # The runtime mode is "wb", so these writer methods are valid here.
    # pylint: disable=no-member
    wav_file: Any
    with wave.open(cast(BinaryIO, buffer), "wb") as wav_file:
        wav_file.setnchannels(normalized_channels)
        wav_file.setsampwidth(_SAMPLE_WIDTH_BYTES)
        wav_file.setframerate(normalized_sample_rate)
        wav_file.writeframes(aligned_pcm_bytes)
    # pylint: enable=no-member
    return buffer.getvalue()


def pcm16_duration_ms(
    pcm_bytes: bytes,
    *,
    sample_rate: int,
    channels: int,
) -> int:
    normalized_sample_rate = _ensure_int("sample_rate", sample_rate, minimum=1)
    normalized_channels = _ensure_int("channels", channels, minimum=1)
    frame_bytes = _SAMPLE_WIDTH_BYTES * normalized_channels
    aligned_pcm_bytes = _trim_incomplete_bytes(pcm_bytes, alignment=frame_bytes)
    if not aligned_pcm_bytes:
        return 0
    frame_count = len(aligned_pcm_bytes) // frame_bytes
    return int((frame_count / normalized_sample_rate) * 1000)


def pcm16_signal_profile(samples: bytes) -> Pcm16SignalProfile:
    aligned_samples = _trim_incomplete_bytes(samples, alignment=_SAMPLE_WIDTH_BYTES)
    digest = hashlib.sha256(aligned_samples).hexdigest()[:12]
    if not aligned_samples:
        return Pcm16SignalProfile(
            sample_count=0,
            rms=0,
            mean_abs=0,
            peak_abs=0,
            dc_offset=0,
            nonzero_sample_count=0,
            nonzero_sample_ratio=0.0,
            clipped_sample_count=0,
            clipped_sample_ratio=0.0,
            zero_crossing_count=0,
            zero_crossing_ratio=0.0,
            sha256=digest,
        )

    pcm_samples = _pcm16_array_from_bytes(aligned_samples)
    sample_count = len(pcm_samples)
    if sample_count <= 0:
        return Pcm16SignalProfile(
            sample_count=0,
            rms=0,
            mean_abs=0,
            peak_abs=0,
            dc_offset=0,
            nonzero_sample_count=0,
            nonzero_sample_ratio=0.0,
            clipped_sample_count=0,
            clipped_sample_ratio=0.0,
            zero_crossing_count=0,
            zero_crossing_ratio=0.0,
            sha256=digest,
        )

    rms = _pcm16_rms(aligned_samples)
    absolute_values = [abs(sample) for sample in pcm_samples]
    mean_abs = int(sum(absolute_values) / sample_count)
    peak_abs = min(max(absolute_values), _PCM16_MAX_ABS)
    dc_offset = int(sum(pcm_samples) / sample_count)
    nonzero_sample_count = sum(1 for sample in pcm_samples if sample != 0)
    clipped_sample_count = sum(1 for value in absolute_values if value >= _PCM16_MAX_ABS)

    zero_crossing_count = 0
    previous_sample = pcm_samples[0]
    for sample in pcm_samples[1:]:
        if (previous_sample < 0 <= sample) or (previous_sample > 0 >= sample):
            zero_crossing_count += 1
        if sample != 0:
            previous_sample = sample

    return Pcm16SignalProfile(
        sample_count=sample_count,
        rms=rms,
        mean_abs=mean_abs,
        peak_abs=peak_abs,
        dc_offset=dc_offset,
        nonzero_sample_count=nonzero_sample_count,
        nonzero_sample_ratio=nonzero_sample_count / sample_count,
        clipped_sample_count=clipped_sample_count,
        clipped_sample_ratio=clipped_sample_count / sample_count,
        zero_crossing_count=zero_crossing_count,
        zero_crossing_ratio=zero_crossing_count / max(1, sample_count - 1),
        sha256=digest,
    )


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
        duplex_playback_device: str | None = None,
        duplex_playback_sample_rate_hz: int = 24000,
        vad_mode: str = _DEFAULT_VAD_MODE,
        vad_webrtc_aggressiveness: int = _DEFAULT_WEBRTC_VAD_AGGRESSIVENESS,
        adaptive_noise_enabled: bool = True,
        adaptive_noise_multiplier: float = _DEFAULT_ADAPTIVE_NOISE_MULTIPLIER,
        adaptive_noise_margin_rms: int = _DEFAULT_ADAPTIVE_NOISE_MARGIN_RMS,
        vad_frame_ms: int = _DEFAULT_WEBRTC_VAD_FRAME_MS,
        vad_min_voiced_frame_ratio: float = _DEFAULT_WEBRTC_VAD_MIN_VOICED_FRAME_RATIO,
    ) -> None:
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
        self.dynamic_pause_short_pause_grace_bonus_ms = max(0, int(dynamic_pause_short_pause_grace_bonus_ms))
        self.dynamic_pause_medium_pause_penalty_ms = max(0, int(dynamic_pause_medium_pause_penalty_ms))
        self.dynamic_pause_medium_pause_grace_penalty_ms = max(0, int(dynamic_pause_medium_pause_grace_penalty_ms))
        self.dynamic_pause_long_pause_penalty_ms = max(0, int(dynamic_pause_long_pause_penalty_ms))
        self.dynamic_pause_long_pause_grace_penalty_ms = max(0, int(dynamic_pause_long_pause_grace_penalty_ms))

        self.pause_resume_chunks = _ensure_int("pause_resume_chunks", pause_resume_chunks, minimum=1)
        self.duplex_playback_device = str(duplex_playback_device or "").strip() or None
        self.duplex_playback_sample_rate_hz = _ensure_int(
            "duplex_playback_sample_rate_hz",
            duplex_playback_sample_rate_hz,
            minimum=8000,
        )

        self.vad_mode = _normalize_vad_mode(vad_mode)
        self.vad_webrtc_aggressiveness = min(
            3,
            max(0, _ensure_int("vad_webrtc_aggressiveness", vad_webrtc_aggressiveness, minimum=0)),
        )
        self.adaptive_noise_enabled = bool(adaptive_noise_enabled)
        self.adaptive_noise_multiplier = max(1.0, float(adaptive_noise_multiplier))
        self.adaptive_noise_margin_rms = _ensure_int(
            "adaptive_noise_margin_rms",
            adaptive_noise_margin_rms,
            minimum=0,
        )
        self.vad_frame_ms = _resolve_effective_vad_frame_ms(
            chunk_ms=self.chunk_ms,
            requested_frame_ms=vad_frame_ms,
        )
        self.vad_min_voiced_frame_ratio = _ensure_ratio(
            "vad_min_voiced_frame_ratio",
            vad_min_voiced_frame_ratio,
        )

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
            duplex_playback_device=config.audio_output_device,
            duplex_playback_sample_rate_hz=resolve_respeaker_duplex_playback_sample_rate_hz(config),
            vad_mode=getattr(config, "audio_vad_mode", _DEFAULT_VAD_MODE),
            vad_webrtc_aggressiveness=getattr(
                config,
                "audio_vad_webrtc_aggressiveness",
                _DEFAULT_WEBRTC_VAD_AGGRESSIVENESS,
            ),
            adaptive_noise_enabled=getattr(config, "audio_adaptive_noise_enabled", True),
            adaptive_noise_multiplier=getattr(
                config,
                "audio_adaptive_noise_multiplier",
                _DEFAULT_ADAPTIVE_NOISE_MULTIPLIER,
            ),
            adaptive_noise_margin_rms=getattr(
                config,
                "audio_adaptive_noise_margin_rms",
                _DEFAULT_ADAPTIVE_NOISE_MARGIN_RMS,
            ),
            vad_frame_ms=getattr(config, "audio_vad_frame_ms", _DEFAULT_WEBRTC_VAD_FRAME_MS),
            vad_min_voiced_frame_ratio=getattr(
                config,
                "audio_vad_min_voiced_frame_ratio",
                _DEFAULT_WEBRTC_VAD_MIN_VOICED_FRAME_RATIO,
            ),
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

        captured = bytearray()
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
        frame_bytes = _bytes_per_frame(self.channels)
        read_size_bytes = max(chunk_bytes, _PROCESS_PIPE_READ_SIZE_BYTES)
        chunk_buffer = _PcmChunkBuffer(target_chunk_bytes=chunk_bytes, frame_bytes=frame_bytes)

        read_stall_timeout_s = _compute_read_stall_timeout(self.chunk_ms)
        respeaker_recovery_attempted = False

        detector = _SpeechActivityDetector(
            sample_rate=self.sample_rate,
            channels=self.channels,
            speech_threshold=self.speech_threshold,
            vad_mode=self.vad_mode,
            vad_webrtc_aggressiveness=self.vad_webrtc_aggressiveness,
            adaptive_noise_enabled=self.adaptive_noise_enabled,
            adaptive_noise_multiplier=self.adaptive_noise_multiplier,
            adaptive_noise_margin_rms=self.adaptive_noise_margin_rms,
            vad_frame_ms=self.vad_frame_ms,
            vad_min_voiced_frame_ratio=self.vad_min_voiced_frame_ratio,
        )

        def _listen_timeout_diagnostics(*, listened_ms: int) -> ListenTimeoutCaptureDiagnostics:
            average_rms = int(pre_speech_total_rms / pre_speech_chunk_count) if pre_speech_chunk_count > 0 else 0
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

        def _process_chunk(chunk: bytes) -> None:
            nonlocal heard_speech
            nonlocal consecutive_speech_chunks
            nonlocal consecutive_resume_chunks
            nonlocal speech_started_at
            nonlocal last_non_silent_at
            nonlocal pause_candidate_deadline_at
            nonlocal speech_started_after_ms
            nonlocal resumed_after_pause_count
            nonlocal pre_speech_chunk_count
            nonlocal pre_speech_active_chunk_count
            nonlocal pre_speech_total_rms
            nonlocal pre_speech_peak_rms

            now = time.monotonic()
            assessment = detector.assess(chunk, update_noise_floor=not heard_speech)
            rms = assessment.rms

            if not heard_speech:
                if normalized_ignore_initial_ms > 0 and (now - started_at) * 1000 < normalized_ignore_initial_ms:
                    preroll.clear()
                    consecutive_speech_chunks = 0
                    return

                if on_chunk is not None:
                    on_chunk(chunk)

                pre_speech_chunk_count += 1
                pre_speech_total_rms += rms
                pre_speech_peak_rms = max(pre_speech_peak_rms, rms)
                if assessment.active:
                    pre_speech_active_chunk_count += 1

                preroll.append(chunk)

                if assessment.active:
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
                return

            captured.extend(chunk)
            if on_chunk is not None:
                on_chunk(chunk)

            if assessment.active:
                if pause_candidate_deadline_at is not None:
                    confirmed_resume, consecutive_resume_chunks = resolve_pause_resume_confirmation(
                        consecutive_resume_chunks=consecutive_resume_chunks,
                        required_resume_chunks=self.pause_resume_chunks,
                    )
                    if not confirmed_resume:
                        return
                    resumed_after_pause_count += 1

                consecutive_resume_chunks = 0
                pause_candidate_deadline_at = None
                last_non_silent_at = now
                return

            consecutive_resume_chunks = 0
            if last_non_silent_at is None:
                return

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
                return

            if effective_pause_grace_ms <= 0:
                pause_candidate_deadline_at = now
                return

            if pause_candidate_deadline_at is None:
                consecutive_resume_chunks = 0
                pause_candidate_deadline_at = last_non_silent_at + (
                    effective_pause_ms + effective_pause_grace_ms
                ) / 1000.0

        with maybe_open_respeaker_duplex_playback_guard(
            capture_device=self.device,
            playback_device=self.duplex_playback_device,
            sample_rate_hz=self.duplex_playback_sample_rate_hz,
        ):
            process = _spawn_audio_process(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                purpose="Audio capture",
                allow_root_borrowed_session_audio=True,
            )
            try:
                while True:
                    now = time.monotonic()

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

                    if (
                        heard_speech
                        and speech_started_at is not None
                        and now - speech_started_at >= effective_max_record_seconds
                    ):
                        break

                    if process.poll() is not None:
                        if not heard_speech and not captured and not respeaker_recovery_attempted:
                            respeaker_recovery_attempted = True
                            from twinr.hardware.respeaker_capture_recovery import (
                                wait_for_transient_respeaker_capture_ready,
                            )

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
                                    allow_root_borrowed_session_audio=True,
                                )
                                chunk_buffer.clear()
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
                        now = time.monotonic()
                        if now - last_chunk_at >= read_stall_timeout_s:
                            raise RuntimeError("Audio capture stalled while waiting for microphone data")
                        if heard_speech and pause_candidate_deadline_at is not None and now >= pause_candidate_deadline_at:
                            break
                        continue

                    raw = os.read(process.stdout.fileno(), read_size_bytes)
                    if not raw:
                        if time.monotonic() - last_chunk_at >= read_stall_timeout_s:
                            raise RuntimeError("Audio capture stalled while reading microphone data")
                        continue

                    emitted_chunks = chunk_buffer.append(raw)
                    if not emitted_chunks:
                        if time.monotonic() - last_chunk_at >= read_stall_timeout_s:
                            raise RuntimeError("Audio capture stalled while buffering microphone data")
                        continue

                    for chunk in emitted_chunks:
                        last_chunk_at = time.monotonic()
                        _process_chunk(chunk)
                        if heard_speech and pause_candidate_deadline_at is not None and time.monotonic() >= pause_candidate_deadline_at:
                            break

                    if heard_speech and pause_candidate_deadline_at is not None and time.monotonic() >= pause_candidate_deadline_at:
                        break
            finally:
                self._stop_process(process)

        trailing_chunk = chunk_buffer.flush_aligned()
        if heard_speech and trailing_chunk:
            captured.extend(trailing_chunk)
            if on_chunk is not None:
                on_chunk(trailing_chunk)

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
    def __init__(
        self,
        *,
        device: str = "default",
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_ms: int = 100,
        speech_threshold: int = 700,
        default_duration_ms: int = 1000,
        duplex_playback_device: str | None = None,
        duplex_playback_sample_rate_hz: int = 24000,
        vad_mode: str = _DEFAULT_VAD_MODE,
        vad_webrtc_aggressiveness: int = _DEFAULT_WEBRTC_VAD_AGGRESSIVENESS,
        adaptive_noise_enabled: bool = True,
        adaptive_noise_multiplier: float = _DEFAULT_ADAPTIVE_NOISE_MULTIPLIER,
        adaptive_noise_margin_rms: int = _DEFAULT_ADAPTIVE_NOISE_MARGIN_RMS,
        vad_frame_ms: int = _DEFAULT_WEBRTC_VAD_FRAME_MS,
        vad_min_voiced_frame_ratio: float = _DEFAULT_WEBRTC_VAD_MIN_VOICED_FRAME_RATIO,
    ) -> None:
        self.device = _normalize_audio_device(device)
        self.sample_rate = _ensure_int("sample_rate", sample_rate, minimum=1)
        self.channels = _ensure_int("channels", channels, minimum=1)
        self.chunk_ms = max(20, _ensure_int("chunk_ms", chunk_ms, minimum=1))
        self.speech_threshold = _ensure_int("speech_threshold", speech_threshold, minimum=0)
        self.default_duration_ms = max(
            self.chunk_ms,
            _ensure_int("default_duration_ms", default_duration_ms, minimum=1),
        )

        self.duplex_playback_device = str(duplex_playback_device or "").strip() or None
        self.duplex_playback_sample_rate_hz = _ensure_int(
            "duplex_playback_sample_rate_hz",
            duplex_playback_sample_rate_hz,
            minimum=8000,
        )

        self.vad_mode = _normalize_vad_mode(vad_mode)
        self.vad_webrtc_aggressiveness = min(
            3,
            max(0, _ensure_int("vad_webrtc_aggressiveness", vad_webrtc_aggressiveness, minimum=0)),
        )
        self.adaptive_noise_enabled = bool(adaptive_noise_enabled)
        self.adaptive_noise_multiplier = max(1.0, float(adaptive_noise_multiplier))
        self.adaptive_noise_margin_rms = _ensure_int(
            "adaptive_noise_margin_rms",
            adaptive_noise_margin_rms,
            minimum=0,
        )
        self.vad_frame_ms = _resolve_effective_vad_frame_ms(
            chunk_ms=self.chunk_ms,
            requested_frame_ms=vad_frame_ms,
        )
        self.vad_min_voiced_frame_ratio = _ensure_ratio(
            "vad_min_voiced_frame_ratio",
            vad_min_voiced_frame_ratio,
        )

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "AmbientAudioSampler":
        device = _normalize_audio_device(config.proactive_audio_input_device or config.audio_input_device)
        return cls(
            device=device,
            sample_rate=config.audio_sample_rate,
            channels=config.audio_channels,
            chunk_ms=config.audio_chunk_ms,
            speech_threshold=config.audio_speech_threshold,
            default_duration_ms=config.proactive_audio_sample_ms,
            duplex_playback_device=config.audio_output_device,
            duplex_playback_sample_rate_hz=resolve_respeaker_duplex_playback_sample_rate_hz(config),
            vad_mode=getattr(config, "audio_vad_mode", _DEFAULT_VAD_MODE),
            vad_webrtc_aggressiveness=getattr(
                config,
                "audio_vad_webrtc_aggressiveness",
                _DEFAULT_WEBRTC_VAD_AGGRESSIVENESS,
            ),
            adaptive_noise_enabled=getattr(config, "audio_adaptive_noise_enabled", True),
            adaptive_noise_multiplier=getattr(
                config,
                "audio_adaptive_noise_multiplier",
                _DEFAULT_ADAPTIVE_NOISE_MULTIPLIER,
            ),
            adaptive_noise_margin_rms=getattr(
                config,
                "audio_adaptive_noise_margin_rms",
                _DEFAULT_ADAPTIVE_NOISE_MARGIN_RMS,
            ),
            vad_frame_ms=getattr(config, "audio_vad_frame_ms", _DEFAULT_WEBRTC_VAD_FRAME_MS),
            vad_min_voiced_frame_ratio=getattr(
                config,
                "audio_vad_min_voiced_frame_ratio",
                _DEFAULT_WEBRTC_VAD_MIN_VOICED_FRAME_RATIO,
            ),
        )

    def sample_levels(self, *, duration_ms: int | None = None) -> AmbientAudioLevelSample:
        return self.sample_window(duration_ms=duration_ms).sample

    def require_readable_frames(
        self,
        *,
        duration_ms: int | None = None,
        chunk_count: int = 1,
    ) -> AudioCaptureReadinessProbe:
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
        effective_duration_ms = self._resolve_duration_ms(duration_ms)
        pcm_fragments = self._capture_chunks(
            target_chunk_count=max(1, math.ceil(effective_duration_ms / self.chunk_ms)),
            effective_duration_ms=effective_duration_ms,
        )

        detector = _SpeechActivityDetector(
            sample_rate=self.sample_rate,
            channels=self.channels,
            speech_threshold=self.speech_threshold,
            vad_mode=self.vad_mode,
            vad_webrtc_aggressiveness=self.vad_webrtc_aggressiveness,
            adaptive_noise_enabled=self.adaptive_noise_enabled,
            adaptive_noise_multiplier=self.adaptive_noise_multiplier,
            adaptive_noise_margin_rms=self.adaptive_noise_margin_rms,
            vad_frame_ms=self.vad_frame_ms,
            vad_min_voiced_frame_ratio=self.vad_min_voiced_frame_ratio,
        )

        assessments = [detector.assess(chunk, update_noise_floor=True) for chunk in pcm_fragments]
        rms_values = [assessment.rms for assessment in assessments]
        active_chunk_count = sum(1 for assessment in assessments if assessment.active)
        average_rms = int(sum(rms_values) / len(rms_values))
        peak_rms = max(rms_values)

        pcm_bytes = b"".join(pcm_fragments)
        actual_duration_ms = pcm16_duration_ms(
            pcm_bytes,
            sample_rate=self.sample_rate,
            channels=self.channels,
        )

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
            pcm_bytes=pcm_bytes,
            sample_rate=self.sample_rate,
            channels=self.channels,
        )

    def _resolve_duration_ms(self, duration_ms: int | None) -> int:
        if duration_ms is None:
            normalized_duration_ms = self.default_duration_ms
        else:
            requested_duration_ms = _ensure_int("duration_ms", duration_ms, minimum=0)
            normalized_duration_ms = self.default_duration_ms if requested_duration_ms == 0 else requested_duration_ms
        return max(self.chunk_ms, normalized_duration_ms)

    def _capture_chunks(
        self,
        *,
        target_chunk_count: int,
        effective_duration_ms: int,
    ) -> list[bytes]:
        normalized_target_chunk_count = _ensure_int("target_chunk_count", target_chunk_count, minimum=1)
        chunk_bytes = _chunk_byte_count(
            sample_rate=self.sample_rate,
            channels=self.channels,
            chunk_ms=self.chunk_ms,
        )
        frame_bytes = _bytes_per_frame(self.channels)
        read_size_bytes = max(chunk_bytes, _PROCESS_PIPE_READ_SIZE_BYTES)

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

        pcm_fragments: list[bytes] = []
        bytes_captured = 0
        chunk_buffer = _PcmChunkBuffer(target_chunk_bytes=chunk_bytes, frame_bytes=frame_bytes)

        def _raise_capture_readiness_error(*, failure_reason: str, detail: str) -> None:
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
            with maybe_open_respeaker_duplex_playback_guard(
                capture_device=self.device,
                playback_device=self.duplex_playback_device,
                sample_rate_hz=self.duplex_playback_sample_rate_hz,
            ):
                process = _spawn_audio_process(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    purpose="Ambient audio capture",
                    allow_root_borrowed_session_audio=True,
                )
                started_at = time.monotonic()
                last_chunk_at = started_at
                capture_deadline_at = started_at + (effective_duration_ms / 1000.0) + _AMBIENT_CAPTURE_EXTRA_TIMEOUT_S
                read_stall_timeout_s = _compute_read_stall_timeout(self.chunk_ms)

                try:
                    while len(pcm_fragments) < normalized_target_chunk_count:
                        now = time.monotonic()
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

                        raw = os.read(process.stdout.fileno(), read_size_bytes)
                        if not raw:
                            if time.monotonic() - last_chunk_at >= read_stall_timeout_s:
                                _raise_capture_readiness_error(
                                    failure_reason="stalled_reading",
                                    detail="Ambient audio capture stalled while reading microphone data",
                                )
                            continue

                        emitted_chunks = chunk_buffer.append(raw)
                        if not emitted_chunks:
                            if time.monotonic() - last_chunk_at >= read_stall_timeout_s:
                                _raise_capture_readiness_error(
                                    failure_reason="stalled_buffering",
                                    detail="Ambient audio capture stalled while buffering microphone data",
                                )
                            continue

                        for chunk in emitted_chunks:
                            last_chunk_at = time.monotonic()
                            pcm_fragments.append(chunk)
                            bytes_captured += len(chunk)
                            if len(pcm_fragments) >= normalized_target_chunk_count:
                                break
                finally:
                    _stop_process(process)
        except RuntimeError as exc:
            if "duplex playback guard" not in str(exc).lower():
                raise
            _raise_capture_readiness_error(
                failure_reason="duplex_playback_failed",
                detail=f"Ambient audio capture could not start the required ReSpeaker duplex playback guard: {exc}",
            )

        if not pcm_fragments:
            _raise_capture_readiness_error(
                failure_reason="no_usable_samples",
                detail="Ambient audio capture ended without usable samples",
            )

        return pcm_fragments


class WaveAudioPlayer:
    def __init__(
        self,
        *,
        device: str = "default",
        max_stream_bytes: int = _DEFAULT_MAX_PLAYBACK_STREAM_BYTES,
    ) -> None:
        self.device = _normalize_audio_device(device)
        # BREAKING: streamed playback now aborts after a bounded payload size to
        # prevent practical Pi SD-card/tmp exhaustion and endless remote TTS
        # streams from monopolizing the audio path.
        self.max_stream_bytes = _ensure_int(
            "max_stream_bytes",
            max_stream_bytes,
            minimum=1024,
        )
        self._active_process_lock = Lock()
        self._active_process: subprocess.Popen[bytes] | None = None
        self._stopped_process_ids: set[int] = set()

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "WaveAudioPlayer":
        ensure_respeaker_playback_mixer(config.audio_output_device)
        return cls(
            device=config.audio_output_device,
            max_stream_bytes=getattr(
                config,
                "audio_max_playback_stream_bytes",
                _DEFAULT_MAX_PLAYBACK_STREAM_BYTES,
            ),
        )

    def play_wav_bytes(self, audio_bytes: bytes) -> None:
        if not audio_bytes:
            return
        if len(audio_bytes) > self.max_stream_bytes:
            raise RuntimeError("Audio playback payload exceeds the configured safe stream limit")
        self.play_wav_chunks([audio_bytes])

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

    def play_wav_chunks(
        self,
        chunks: Iterable[bytes],
        *,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
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
        total_bytes_streamed = 0

        try:
            if process.stdin is None:
                raise RuntimeError("aplay did not expose stdin")

            stdin_fd = process.stdin.fileno()
            os.set_blocking(stdin_fd, False)

            for chunk in chunks:
                if should_stop is not None and should_stop():
                    stopped_early = True
                    break
                if not chunk:
                    continue

                total_bytes_streamed += len(chunk)
                if total_bytes_streamed > self.max_stream_bytes:
                    raise RuntimeError("Audio playback payload exceeds the configured safe stream limit")

                view = memoryview(chunk)
                while view:
                    if should_stop is not None and should_stop():
                        stopped_early = True
                        view = view[:0]
                        break

                    if process.poll() is not None:
                        if self._was_stopped(process):
                            return
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
                        if self._was_stopped(process):
                            return
                        self._raise_stream_error(process)
                    except OSError as exc:
                        if self._was_stopped(process):
                            return
                        raise RuntimeError(f"Audio playback failed while streaming: {exc}") from exc

                    if written <= 0:
                        if self._was_stopped(process):
                            return
                        self._raise_stream_error(process)

                    view = view[written:]

            if not stopped_early and should_stop is not None and should_stop():
                stopped_early = True

            if stopped_early:
                self._mark_process_stopped(process)
                _stop_process(process)
                return

            try:
                process.stdin.close()
            except BrokenPipeError:
                if self._was_stopped(process):
                    return
                self._raise_stream_error(process)
            except OSError as exc:
                if self._was_stopped(process):
                    return
                raise RuntimeError(f"Audio playback failed while closing stream: {exc}") from exc

            process.wait(timeout=_PLAYBACK_FINALIZE_TIMEOUT_S)
            if self._was_stopped(process):
                return
            if process.returncode != 0:
                self._raise_stream_error(process)

        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Audio playback timed out while waiting for aplay to finish") from exc
        finally:
            self._clear_active_process(process)
            self._forget_process_stopped(process)
            _stop_process(process)

    def play_file(self, path: str | Path) -> None:
        path_obj = Path(path)
        if not path_obj.is_file():
            raise RuntimeError(f"Audio playback file not found: {path_obj}")

        process = _spawn_audio_process(
            ["aplay", "-q", "-D", self.device, str(path_obj)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            purpose="Audio playback",
        )
        self._set_active_process(process)
        try:
            process.wait(timeout=_PLAYBACK_FILE_TIMEOUT_S)
            if self._was_stopped(process):
                return
            if process.returncode != 0:
                self._raise_stream_error(process)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Audio playback timed out") from exc
        finally:
            self._clear_active_process(process)
            self._forget_process_stopped(process)
            _stop_process(process)

    def _raise_stream_error(self, process: subprocess.Popen[bytes]) -> None:
        message = _process_failure_message(process, default_action="Audio playback")
        raise RuntimeError(f"Audio playback failed: {message}")

    def stop_playback(self) -> None:
        with self._active_process_lock:
            process = self._active_process
            if process is not None:
                self._stopped_process_ids.add(id(process))
        if process is None:
            return
        _stop_process(process)

    def _set_active_process(self, process: subprocess.Popen[bytes]) -> None:
        with self._active_process_lock:
            self._active_process = process

    def _clear_active_process(self, process: subprocess.Popen[bytes]) -> None:
        with self._active_process_lock:
            if self._active_process is process:
                self._active_process = None

    def _mark_process_stopped(self, process: subprocess.Popen[bytes]) -> None:
        with self._active_process_lock:
            self._stopped_process_ids.add(id(process))

    def _was_stopped(self, process: subprocess.Popen[bytes]) -> bool:
        with self._active_process_lock:
            return id(process) in self._stopped_process_ids

    def _forget_process_stopped(self, process: subprocess.Popen[bytes]) -> None:
        with self._active_process_lock:
            self._stopped_process_ids.discard(id(process))

    def _render_tone_pcm(
        self,
        *,
        frequency_hz: int,
        duration_ms: int,
        volume: float,
        sample_rate: int,
    ) -> bytes:
        normalized_frequency_hz = _ensure_int("frequency_hz", frequency_hz, minimum=0)
        normalized_duration_ms = _ensure_int("duration_ms", duration_ms, minimum=0)
        normalized_sample_rate = _ensure_int("sample_rate", sample_rate, minimum=1)

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
