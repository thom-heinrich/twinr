"""Compact forensic helpers for Twinr's live voice transport.

The physical Pi voice path needs enough telemetry to distinguish capture,
transport, and STT failures without writing raw household audio into traces.
This module keeps that summarization logic focused and reusable across the
edge capture client and the host-side websocket session.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import time
from dataclasses import dataclass, field
from typing import Final

from twinr.hardware.audio import Pcm16SignalProfile, pcm16_signal_profile

# CHANGELOG: 2026-03-29
# BUG-1: audio_ms no longer blindly increments chunk_ms for empty/short/corrupt
#        frames; it is now sample-rate aware and byte-derived when
#        sample_rate_hz is configured.
# BUG-2: sequence duplicates/backtracks no longer silently corrupt transport
#        telemetry; they are counted explicitly, and stream continuity can be
#        reset on demand.
# BUG-3: flush_details() no longer emits a synthetic all-zero summary for an
#        empty window; it returns {} instead.
# SEC-1: raw per-frame SHA256 is no longer exposed by default; a keyed,
#        process-scoped fingerprint is emitted instead, and raw SHA256 is now
#        opt-in via TWINR_VOICE_TELEMETRY_INCLUDE_SHA256.
# SEC-2: malformed/oversized payloads and invalid sequence values are bounded
#        and counted instead of crashing or pinning the Pi hot path.
# IMP-1: add_frame() now accepts upstream VAD decisions/confidence so speech
#        telemetry can follow modern WebRTC/Silero pipelines instead of only an
#        RMS proxy.
# IMP-2: telemetry now tracks profiled vs observed bytes/frames, arrival-time
#        variance, dual flush thresholds, and stream regressions for stronger
#        capture/transport forensics.

_DEFAULT_MAX_FRAME_BYTES: Final[int] = 256 * 1024
_ENABLE_RAW_SHA256_ENV: Final[str] = "TWINR_VOICE_TELEMETRY_INCLUDE_SHA256"
_FINGERPRINT_KEY_ENV: Final[str] = "TWINR_TELEMETRY_FINGERPRINT_KEY"
_FINGERPRINT_MODE: Final[str] = "blake2s8_process_scoped"
_EMPTY_SHA256: Final[str] = hashlib.sha256(b"").hexdigest()
_LABEL_SANITIZER: Final[re.Pattern[str]] = re.compile(r"[^0-9A-Za-z_]+")


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _load_fingerprint_key() -> bytes:
    raw = os.getenv(_FINGERPRINT_KEY_ENV)
    if raw:
        try:
            data = bytes.fromhex(raw)
        except ValueError:
            data = raw.encode("utf-8", "ignore")
        return (data or b"twinr-voice-telemetry")[:32]
    return secrets.token_bytes(16)


_PROCESS_FINGERPRINT_KEY: Final[bytes] = _load_fingerprint_key()


def _sanitize_label(prefix: str) -> str:
    label = _LABEL_SANITIZER.sub("_", str(prefix or "signal").strip()).strip("_")
    if not label:
        return "signal"
    return label[:48]


def _coerce_bytes(payload: bytes | bytearray | memoryview) -> bytes:
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, bytearray):
        return bytes(payload)
    if isinstance(payload, memoryview):
        return payload.tobytes()
    raise TypeError("pcm_bytes must be bytes-like")


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


def _bytes_to_audio_ms(byte_count: int, sample_rate_hz: int, channels: int) -> int:
    if byte_count <= 0 or sample_rate_hz <= 0 or channels <= 0:
        return 0
    bytes_per_second = sample_rate_hz * channels * 2
    return int(round((byte_count / bytes_per_second) * 1000.0))


def _fingerprint_from_sha256(sha256_hex: str | None) -> str | None:
    if not sha256_hex:
        return None
    digest = hashlib.blake2s(
        sha256_hex.encode("ascii", "ignore"),
        key=_PROCESS_FINGERPRINT_KEY,
        digest_size=8,
        person=b"twinrvt1",
    )
    return digest.hexdigest()


def _zero_signal_profile() -> Pcm16SignalProfile:
    try:
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
            sha256=_EMPTY_SHA256,
        )
    except Exception:
        return pcm16_signal_profile(b"")


def prefixed_signal_profile_details(
    profile: Pcm16SignalProfile,
    *,
    prefix: str,
    include_sha256: bool | None = None,
) -> dict[str, object]:
    """Return one bounded mapping for a signal profile."""

    label = _sanitize_label(prefix)
    expose_sha256 = _env_flag(_ENABLE_RAW_SHA256_ENV) if include_sha256 is None else bool(include_sha256)
    sha256_hex = getattr(profile, "sha256", None)
    return {
        f"{label}_sample_count": int(profile.sample_count),
        f"{label}_rms": int(profile.rms),
        f"{label}_mean_abs": int(profile.mean_abs),
        f"{label}_peak_abs": int(profile.peak_abs),
        f"{label}_dc_offset": int(profile.dc_offset),
        f"{label}_nonzero_ratio": round(float(profile.nonzero_sample_ratio), 6),
        f"{label}_clipped_ratio": round(float(profile.clipped_sample_ratio), 6),
        f"{label}_zero_crossing_ratio": round(float(profile.zero_crossing_ratio), 6),
        # BREAKING: raw SHA256 emission is now opt-in; the default trace/log-safe
        # identifier is a keyed, process-scoped fingerprint.
        f"{label}_sha256": sha256_hex if expose_sha256 else None,
        f"{label}_payload_fingerprint": _fingerprint_from_sha256(sha256_hex),
        f"{label}_payload_fingerprint_mode": _FINGERPRINT_MODE,
    }


@dataclass(slots=True)
class VoiceFrameTelemetryBucket:
    """Aggregate a bounded run of live voice frames into one trace event."""

    chunk_ms: int
    speech_threshold: int
    flush_every_frames: int = 10
    sample_rate_hz: int | None = None
    channels: int = 1
    flush_every_ms: int | None = None
    max_frame_bytes: int = _DEFAULT_MAX_FRAME_BYTES
    enable_raw_sha256: bool = field(default_factory=lambda: _env_flag(_ENABLE_RAW_SHA256_ENV))
    window_start_sequence: int | None = None
    window_end_sequence: int | None = None
    _last_sequence_seen: int | None = None
    _window_start_arrival_monotonic_ns: int | None = None
    _window_end_arrival_monotonic_ns: int | None = None
    _last_arrival_monotonic_ns: int | None = None
    frame_count: int = 0
    profiled_frame_count: int = 0
    audio_ms: int = 0
    payload_bytes_total: int = 0
    profiled_payload_bytes_total: int = 0
    sample_count_total: int = 0
    zero_frame_count: int = 0
    empty_frame_count: int = 0
    speech_frame_count: int = 0
    vad_decision_count: int = 0
    vad_override_count: int = 0
    vad_confidence_sum: float = 0.0
    vad_confidence_count: int = 0
    invalid_frame_count: int = 0
    oversize_frame_count: int = 0
    invalid_sequence_count: int = 0
    sequence_duplicate_count: int = 0
    sequence_regression_count: int = 0
    max_sequence_backtrack: int = 0
    sequence_gap_count: int = 0
    sequence_gap_frames: int = 0
    max_sequence_gap: int = 0
    interarrival_count: int = 0
    interarrival_ms_sum: float = 0.0
    max_interarrival_ms: float = 0.0
    rms_sum: int = 0
    peak_rms: int = 0
    mean_abs_sum: int = 0
    peak_abs: int = 0
    nonzero_ratio_sum: float = 0.0
    clipped_ratio_sum: float = 0.0
    zero_crossing_ratio_sum: float = 0.0
    max_abs_dc_offset: int = 0
    last_payload_sha256: str | None = None
    last_payload_fingerprint: str | None = None
    last_frame_error: str | None = None

    def __post_init__(self) -> None:
        self.chunk_ms = max(0, _safe_int(self.chunk_ms, 0))
        self.speech_threshold = max(0, _safe_int(self.speech_threshold, 0))
        self.flush_every_frames = max(1, _safe_int(self.flush_every_frames, 1))
        self.channels = max(1, _safe_int(self.channels, 1))
        self.max_frame_bytes = max(2, _safe_int(self.max_frame_bytes, _DEFAULT_MAX_FRAME_BYTES))
        if self.max_frame_bytes % 2:
            self.max_frame_bytes += 1
        if self.flush_every_ms is not None:
            self.flush_every_ms = max(1, _safe_int(self.flush_every_ms, 1))
        if self.sample_rate_hz is not None:
            sample_rate_hz = _safe_int(self.sample_rate_hz, 0)
            self.sample_rate_hz = sample_rate_hz if sample_rate_hz > 0 else None

    def _current_audio_ms(self) -> int:
        if self.sample_rate_hz is not None:
            return _bytes_to_audio_ms(
                self.profiled_payload_bytes_total,
                self.sample_rate_hz,
                self.channels,
            )
        return self.profiled_frame_count * self.chunk_ms

    def _current_window_wall_ms(self) -> int:
        if self._window_start_arrival_monotonic_ns is None or self._window_end_arrival_monotonic_ns is None:
            return 0
        delta_ns = max(0, self._window_end_arrival_monotonic_ns - self._window_start_arrival_monotonic_ns)
        return int(round(delta_ns / 1_000_000.0))

    def _normalize_sequence(self, sequence: int) -> int:
        try:
            normalized = int(sequence)
        except (TypeError, ValueError):
            self.invalid_sequence_count += 1
            return (self._last_sequence_seen + 1) if self._last_sequence_seen is not None else 0
        if normalized < 0:
            self.invalid_sequence_count += 1
            return (self._last_sequence_seen + 1) if self._last_sequence_seen is not None else 0
        if isinstance(sequence, float) and not sequence.is_integer():
            self.invalid_sequence_count += 1
        return normalized

    def _record_arrival(self, arrival_monotonic_ns: int | None) -> None:
        arrival_ns = time.monotonic_ns() if arrival_monotonic_ns is None else max(0, _safe_int(arrival_monotonic_ns, 0))
        if self._last_arrival_monotonic_ns is not None and arrival_ns < self._last_arrival_monotonic_ns:
            arrival_ns = self._last_arrival_monotonic_ns
        if self._window_start_arrival_monotonic_ns is None:
            self._window_start_arrival_monotonic_ns = arrival_ns
        if self._last_arrival_monotonic_ns is not None:
            delta_ms = (arrival_ns - self._last_arrival_monotonic_ns) / 1_000_000.0
            self.interarrival_count += 1
            self.interarrival_ms_sum += delta_ms
            self.max_interarrival_ms = max(self.max_interarrival_ms, delta_ms)
        self._window_end_arrival_monotonic_ns = arrival_ns
        self._last_arrival_monotonic_ns = arrival_ns

    def _update_sequence_state(self, normalized_sequence: int) -> None:
        if self.window_start_sequence is None:
            self.window_start_sequence = normalized_sequence
        if self._last_sequence_seen is not None:
            if normalized_sequence > self._last_sequence_seen + 1:
                gap = normalized_sequence - self._last_sequence_seen - 1
                self.sequence_gap_count += 1
                self.sequence_gap_frames += gap
                self.max_sequence_gap = max(self.max_sequence_gap, gap)
            elif normalized_sequence == self._last_sequence_seen:
                self.sequence_duplicate_count += 1
            elif normalized_sequence < self._last_sequence_seen:
                backtrack = self._last_sequence_seen - normalized_sequence
                self.sequence_regression_count += 1
                self.max_sequence_backtrack = max(self.max_sequence_backtrack, backtrack)
        self.window_end_sequence = normalized_sequence
        self._last_sequence_seen = normalized_sequence

    def _mark_invalid_frame(self, error: str) -> Pcm16SignalProfile:
        self.invalid_frame_count += 1
        self.last_frame_error = error
        self.last_payload_sha256 = None
        self.last_payload_fingerprint = None
        self.audio_ms = self._current_audio_ms()
        return _zero_signal_profile()

    def _reset_window(self) -> None:
        self.window_start_sequence = None
        self.window_end_sequence = None
        self._window_start_arrival_monotonic_ns = None
        self._window_end_arrival_monotonic_ns = None
        self.frame_count = 0
        self.profiled_frame_count = 0
        self.audio_ms = 0
        self.payload_bytes_total = 0
        self.profiled_payload_bytes_total = 0
        self.sample_count_total = 0
        self.zero_frame_count = 0
        self.empty_frame_count = 0
        self.speech_frame_count = 0
        self.vad_decision_count = 0
        self.vad_override_count = 0
        self.vad_confidence_sum = 0.0
        self.vad_confidence_count = 0
        self.invalid_frame_count = 0
        self.oversize_frame_count = 0
        self.invalid_sequence_count = 0
        self.sequence_duplicate_count = 0
        self.sequence_regression_count = 0
        self.max_sequence_backtrack = 0
        self.sequence_gap_count = 0
        self.sequence_gap_frames = 0
        self.max_sequence_gap = 0
        self.interarrival_count = 0
        self.interarrival_ms_sum = 0.0
        self.max_interarrival_ms = 0.0
        self.rms_sum = 0
        self.peak_rms = 0
        self.mean_abs_sum = 0
        self.peak_abs = 0
        self.nonzero_ratio_sum = 0.0
        self.clipped_ratio_sum = 0.0
        self.zero_crossing_ratio_sum = 0.0
        self.max_abs_dc_offset = 0
        self.last_payload_sha256 = None
        self.last_payload_fingerprint = None
        self.last_frame_error = None

    def reset_stream_state(self) -> None:
        """Reset continuity state after reconnects or explicit stream boundaries."""

        self._last_sequence_seen = None
        self._last_arrival_monotonic_ns = None

    def reset(self) -> None:
        """Reset both the current window and the cross-window continuity state."""

        self._reset_window()
        self.reset_stream_state()

    def add_frame(
        self,
        *,
        sequence: int,
        pcm_bytes: bytes,
        speech: bool | None = None,
        speech_confidence: float | None = None,
        arrival_monotonic_ns: int | None = None,
    ) -> Pcm16SignalProfile:
        """Fold one PCM payload into the current telemetry window."""

        normalized_sequence = self._normalize_sequence(sequence)
        self._record_arrival(arrival_monotonic_ns)
        self._update_sequence_state(normalized_sequence)
        self.frame_count += 1
        self.last_frame_error = None

        try:
            payload = _coerce_bytes(pcm_bytes)
        except TypeError as exc:
            return self._mark_invalid_frame(exc.__class__.__name__)

        payload_size = len(payload)
        self.payload_bytes_total += payload_size

        if payload_size == 0:
            self.empty_frame_count += 1
            return self._mark_invalid_frame("empty_payload")
        if payload_size > self.max_frame_bytes:
            self.oversize_frame_count += 1
            return self._mark_invalid_frame(f"oversize_payload:{payload_size}")
        if payload_size % 2:
            return self._mark_invalid_frame(f"odd_length_payload:{payload_size}")

        try:
            profile = pcm16_signal_profile(payload)
        except Exception as exc:
            return self._mark_invalid_frame(exc.__class__.__name__)

        self.profiled_frame_count += 1
        self.profiled_payload_bytes_total += payload_size
        self.sample_count_total += int(profile.sample_count)
        self.audio_ms = self._current_audio_ms()

        if int(profile.sample_count) <= 0 or float(profile.nonzero_sample_ratio) <= 0.0:
            self.zero_frame_count += 1

        confidence_value: float | None = None
        if speech_confidence is not None:
            try:
                confidence_value = _clamp01(float(speech_confidence))
            except (TypeError, ValueError):
                confidence_value = None

        external_speech = speech
        if external_speech is None and confidence_value is not None:
            external_speech = confidence_value >= 0.5

        rms_speech = int(profile.rms) >= self.speech_threshold
        if external_speech is None:
            speech_detected = rms_speech
        else:
            speech_detected = bool(external_speech)
            self.vad_decision_count += 1
            if speech_detected != rms_speech:
                self.vad_override_count += 1

        if confidence_value is not None:
            self.vad_confidence_sum += confidence_value
            self.vad_confidence_count += 1

        if speech_detected:
            self.speech_frame_count += 1

        self.rms_sum += int(profile.rms)
        self.peak_rms = max(self.peak_rms, int(profile.rms))
        self.mean_abs_sum += int(profile.mean_abs)
        self.peak_abs = max(self.peak_abs, int(profile.peak_abs))
        self.nonzero_ratio_sum += float(profile.nonzero_sample_ratio)
        self.clipped_ratio_sum += float(profile.clipped_sample_ratio)
        self.zero_crossing_ratio_sum += float(profile.zero_crossing_ratio)
        self.max_abs_dc_offset = max(self.max_abs_dc_offset, abs(int(profile.dc_offset)))
        self.last_payload_sha256 = profile.sha256 if self.enable_raw_sha256 else None
        self.last_payload_fingerprint = _fingerprint_from_sha256(profile.sha256)
        return profile

    def has_data(self) -> bool:
        """Return whether the current window contains any observed frames."""

        return self.frame_count > 0

    def should_flush(self) -> bool:
        """Return whether the current window reached a configured flush bound."""

        if not self.has_data():
            return False
        if self.frame_count >= self.flush_every_frames:
            return True
        if self.flush_every_ms is not None and self.audio_ms >= self.flush_every_ms:
            return True
        return False

    def flush_details(self) -> dict[str, object]:
        """Return one summary payload and reset the current window."""

        # BREAKING: empty windows now return {} instead of a synthetic all-zero
        # summary because the old behavior polluted traces with false events.
        if not self.has_data():
            return {}

        profiled_frame_count = max(1, self.profiled_frame_count)
        if self.profiled_frame_count <= 0:
            speech_decision_source = "none"
        elif self.vad_decision_count <= 0:
            speech_decision_source = "rms_threshold"
        elif self.vad_decision_count >= self.profiled_frame_count:
            speech_decision_source = "vad"
        else:
            speech_decision_source = "mixed"

        details = {
            "sequence_start": self.window_start_sequence,
            "sequence_end": self.window_end_sequence,
            "frame_count": self.frame_count,
            "profiled_frame_count": self.profiled_frame_count,
            "audio_ms": self.audio_ms,
            "audio_ms_source": "profiled_pcm_bytes" if self.sample_rate_hz is not None else "chunk_ms_estimate",
            "window_wall_ms": self._current_window_wall_ms(),
            "payload_bytes_total": self.payload_bytes_total,
            "profiled_payload_bytes_total": self.profiled_payload_bytes_total,
            "sample_count_total": self.sample_count_total,
            "invalid_frame_count": self.invalid_frame_count,
            "empty_frame_count": self.empty_frame_count,
            "oversize_frame_count": self.oversize_frame_count,
            "zero_frame_count": self.zero_frame_count,
            "speech_frame_count": self.speech_frame_count,
            "speech_decision_source": speech_decision_source,
            "vad_decision_count": self.vad_decision_count,
            "vad_override_count": self.vad_override_count,
            "average_vad_confidence": (
                round(self.vad_confidence_sum / self.vad_confidence_count, 6)
                if self.vad_confidence_count > 0
                else None
            ),
            "invalid_sequence_count": self.invalid_sequence_count,
            "sequence_duplicate_count": self.sequence_duplicate_count,
            "sequence_regression_count": self.sequence_regression_count,
            "max_sequence_backtrack": self.max_sequence_backtrack,
            "sequence_gap_count": self.sequence_gap_count,
            "sequence_gap_frames": self.sequence_gap_frames,
            "max_sequence_gap": self.max_sequence_gap,
            "average_interarrival_ms": (
                round(self.interarrival_ms_sum / self.interarrival_count, 3)
                if self.interarrival_count > 0
                else None
            ),
            "max_interarrival_ms": round(self.max_interarrival_ms, 3) if self.interarrival_count > 0 else None,
            "average_rms": int(self.rms_sum / profiled_frame_count) if self.profiled_frame_count > 0 else 0,
            "peak_rms": int(self.peak_rms),
            "average_mean_abs": int(self.mean_abs_sum / profiled_frame_count) if self.profiled_frame_count > 0 else 0,
            "peak_abs": int(self.peak_abs),
            "average_nonzero_ratio": (
                round(self.nonzero_ratio_sum / profiled_frame_count, 6)
                if self.profiled_frame_count > 0
                else 0.0
            ),
            "average_clipped_ratio": (
                round(self.clipped_ratio_sum / profiled_frame_count, 6)
                if self.profiled_frame_count > 0
                else 0.0
            ),
            "average_zero_crossing_ratio": (
                round(self.zero_crossing_ratio_sum / profiled_frame_count, 6)
                if self.profiled_frame_count > 0
                else 0.0
            ),
            "max_abs_dc_offset": int(self.max_abs_dc_offset),
            "last_payload_sha256": self.last_payload_sha256,
            "last_payload_fingerprint": self.last_payload_fingerprint,
            "payload_fingerprint_mode": _FINGERPRINT_MODE,
            "last_frame_error": self.last_frame_error,
        }
        self._reset_window()
        return details


__all__ = [
    "VoiceFrameTelemetryBucket",
    "prefixed_signal_profile_details",
]
