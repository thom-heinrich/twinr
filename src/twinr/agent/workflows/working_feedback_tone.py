"""Synthesize calm swelling tones for Twinr's processing feedback loop.

The default thinking cue should feel gentle and positive rather than like a
sharp notification. This helper generates one mono PCM16 tone that swells up
and down softly in a low humming register and can therefore replace the older
disk-backed waiting ambience as the primary processing sound.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import math

from twinr.hardware.audio import pcm16_to_wav_bytes

_PCM16_MAX_ABS = 32767
_MONO_CHANNELS = 1


@dataclass(frozen=True, slots=True)
class SwellingFeedbackToneSpec:
    """Describe one synthesized swelling feedback tone."""

    duration_ms: int
    base_frequency_hz: float
    crest_frequency_hz: float
    harmonic_ratio: float = 1.5
    harmonic_gain: float = 0.16
    envelope_power: float = 1.35


PROCESSING_SWELL_TONE_SPEC = SwellingFeedbackToneSpec(
    duration_ms=5000,
    base_frequency_hz=123.5,
    crest_frequency_hz=155.6,
    harmonic_ratio=2.0,
    harmonic_gain=0.12,
    envelope_power=2.8,
)


def build_swelling_feedback_tone_pcm16(
    spec: SwellingFeedbackToneSpec,
    *,
    sample_rate: int,
    peak_gain: float,
) -> bytes:
    """Return one mono PCM16 swelling tone."""

    normalized_sample_rate = max(1, int(sample_rate))
    normalized_duration_ms = max(1, int(spec.duration_ms))
    normalized_peak_gain = max(0.0, min(1.0, float(peak_gain)))
    if normalized_peak_gain <= 0.0:
        return b""

    frame_count = max(1, int((normalized_sample_rate * normalized_duration_ms) / 1000))
    base_frequency_hz = max(1.0, float(spec.base_frequency_hz))
    crest_frequency_hz = max(base_frequency_hz, float(spec.crest_frequency_hz))
    harmonic_ratio = max(1.0, float(spec.harmonic_ratio))
    harmonic_gain = max(0.0, float(spec.harmonic_gain))
    envelope_power = max(1.0, float(spec.envelope_power))

    fundamental_phase = 0.0
    harmonic_phase = 0.0
    pcm_samples = array("h")
    for sample_index in range(frame_count):
        progress = sample_index / max(1, frame_count - 1)
        swell_progress = math.sin(math.pi * progress)
        envelope = swell_progress**envelope_power
        current_frequency_hz = base_frequency_hz + (
            (crest_frequency_hz - base_frequency_hz) * (swell_progress * swell_progress)
        )
        phase_step = (2.0 * math.pi * current_frequency_hz) / normalized_sample_rate
        sample_value = math.sin(fundamental_phase) + (harmonic_gain * math.sin(harmonic_phase))
        sample_value *= normalized_peak_gain * envelope
        pcm_samples.append(int(max(-1.0, min(1.0, sample_value)) * _PCM16_MAX_ABS))
        fundamental_phase += phase_step
        harmonic_phase += phase_step * harmonic_ratio
    return pcm_samples.tobytes()


def build_swelling_feedback_tone_wav_bytes(
    spec: SwellingFeedbackToneSpec,
    *,
    sample_rate: int,
    peak_gain: float,
) -> bytes:
    """Wrap the synthesized swelling tone in a WAV container."""

    pcm_bytes = build_swelling_feedback_tone_pcm16(
        spec,
        sample_rate=sample_rate,
        peak_gain=peak_gain,
    )
    return pcm16_to_wav_bytes(
        pcm_bytes,
        sample_rate=sample_rate,
        channels=_MONO_CHANNELS,
    )


__all__ = [
    "PROCESSING_SWELL_TONE_SPEC",
    "SwellingFeedbackToneSpec",
    "build_swelling_feedback_tone_pcm16",
    "build_swelling_feedback_tone_wav_bytes",
]
