"""Compact forensic helpers for Twinr's live voice transport.

The physical Pi voice path needs enough telemetry to distinguish capture,
transport, and STT failures without writing raw household audio into traces.
This module keeps that summarization logic focused and reusable across the
edge capture client and the host-side websocket session.
"""

from __future__ import annotations

from dataclasses import dataclass

from twinr.hardware.audio import Pcm16SignalProfile, pcm16_signal_profile


def prefixed_signal_profile_details(
    profile: Pcm16SignalProfile,
    *,
    prefix: str,
) -> dict[str, object]:
    """Return one bounded mapping for a signal profile."""

    label = str(prefix or "signal").strip() or "signal"
    return {
        f"{label}_sample_count": int(profile.sample_count),
        f"{label}_rms": int(profile.rms),
        f"{label}_mean_abs": int(profile.mean_abs),
        f"{label}_peak_abs": int(profile.peak_abs),
        f"{label}_dc_offset": int(profile.dc_offset),
        f"{label}_nonzero_ratio": round(float(profile.nonzero_sample_ratio), 6),
        f"{label}_clipped_ratio": round(float(profile.clipped_sample_ratio), 6),
        f"{label}_zero_crossing_ratio": round(float(profile.zero_crossing_ratio), 6),
        f"{label}_sha256": profile.sha256,
    }


@dataclass(slots=True)
class VoiceFrameTelemetryBucket:
    """Aggregate a bounded run of live voice frames into one trace event."""

    chunk_ms: int
    speech_threshold: int
    flush_every_frames: int = 10
    window_start_sequence: int | None = None
    window_end_sequence: int | None = None
    _last_sequence_seen: int | None = None
    frame_count: int = 0
    audio_ms: int = 0
    zero_frame_count: int = 0
    speech_frame_count: int = 0
    sequence_gap_count: int = 0
    sequence_gap_frames: int = 0
    max_sequence_gap: int = 0
    rms_sum: int = 0
    peak_rms: int = 0
    mean_abs_sum: int = 0
    peak_abs: int = 0
    nonzero_ratio_sum: float = 0.0
    clipped_ratio_sum: float = 0.0
    zero_crossing_ratio_sum: float = 0.0
    max_abs_dc_offset: int = 0
    last_payload_sha256: str | None = None

    def add_frame(
        self,
        *,
        sequence: int,
        pcm_bytes: bytes,
    ) -> Pcm16SignalProfile:
        """Fold one PCM payload into the current telemetry window."""

        profile = pcm16_signal_profile(pcm_bytes)
        normalized_sequence = max(0, int(sequence))
        if self.window_start_sequence is None:
            self.window_start_sequence = normalized_sequence
        if self._last_sequence_seen is not None and normalized_sequence > self._last_sequence_seen + 1:
            gap = normalized_sequence - self._last_sequence_seen - 1
            self.sequence_gap_count += 1
            self.sequence_gap_frames += gap
            self.max_sequence_gap = max(self.max_sequence_gap, gap)
        self.window_end_sequence = normalized_sequence
        self._last_sequence_seen = normalized_sequence
        self.frame_count += 1
        self.audio_ms += max(1, int(self.chunk_ms))
        if profile.nonzero_sample_ratio <= 0.0:
            self.zero_frame_count += 1
        if profile.rms >= self.speech_threshold:
            self.speech_frame_count += 1
        self.rms_sum += int(profile.rms)
        self.peak_rms = max(self.peak_rms, int(profile.rms))
        self.mean_abs_sum += int(profile.mean_abs)
        self.peak_abs = max(self.peak_abs, int(profile.peak_abs))
        self.nonzero_ratio_sum += float(profile.nonzero_sample_ratio)
        self.clipped_ratio_sum += float(profile.clipped_sample_ratio)
        self.zero_crossing_ratio_sum += float(profile.zero_crossing_ratio)
        self.max_abs_dc_offset = max(self.max_abs_dc_offset, abs(int(profile.dc_offset)))
        self.last_payload_sha256 = profile.sha256
        return profile

    def has_data(self) -> bool:
        """Return whether the current window contains any frames."""

        return self.frame_count > 0

    def should_flush(self) -> bool:
        """Return whether the current window reached its bounded flush size."""

        return self.frame_count >= max(1, int(self.flush_every_frames))

    def flush_details(self) -> dict[str, object]:
        """Return one summary payload and reset the current window."""

        frame_count = max(1, self.frame_count)
        details = {
            "sequence_start": self.window_start_sequence,
            "sequence_end": self.window_end_sequence,
            "frame_count": self.frame_count,
            "audio_ms": self.audio_ms,
            "zero_frame_count": self.zero_frame_count,
            "speech_frame_count": self.speech_frame_count,
            "sequence_gap_count": self.sequence_gap_count,
            "sequence_gap_frames": self.sequence_gap_frames,
            "max_sequence_gap": self.max_sequence_gap,
            "average_rms": int(self.rms_sum / frame_count),
            "peak_rms": int(self.peak_rms),
            "average_mean_abs": int(self.mean_abs_sum / frame_count),
            "peak_abs": int(self.peak_abs),
            "average_nonzero_ratio": round(float(self.nonzero_ratio_sum / frame_count), 6),
            "average_clipped_ratio": round(float(self.clipped_ratio_sum / frame_count), 6),
            "average_zero_crossing_ratio": round(float(self.zero_crossing_ratio_sum / frame_count), 6),
            "max_abs_dc_offset": int(self.max_abs_dc_offset),
            "last_payload_sha256": self.last_payload_sha256,
        }
        self.window_start_sequence = None
        self.window_end_sequence = None
        self.frame_count = 0
        self.audio_ms = 0
        self.zero_frame_count = 0
        self.speech_frame_count = 0
        self.sequence_gap_count = 0
        self.sequence_gap_frames = 0
        self.max_sequence_gap = 0
        self.rms_sum = 0
        self.peak_rms = 0
        self.mean_abs_sum = 0
        self.peak_abs = 0
        self.nonzero_ratio_sum = 0.0
        self.clipped_ratio_sum = 0.0
        self.zero_crossing_ratio_sum = 0.0
        self.max_abs_dc_offset = 0
        self.last_payload_sha256 = None
        return details


__all__ = [
    "VoiceFrameTelemetryBucket",
    "prefixed_signal_profile_details",
]
