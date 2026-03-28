# CHANGELOG: 2026-03-28
# BUG-1: Fixed WAV-header/PCM sample-rate mismatch by normalizing render inputs once
#        and reusing the normalized rate in both PCM and WAV paths.
# BUG-2: Fixed hard clipping when additive harmonics push the waveform above full scale
#        (default spec clipped at peak_gain=1.0). Rendering now preserves intent while
#        preventing digital distortion.
# SEC-1: Added explicit Pi-safe bounds for sample_rate, duration, and frame count to
#        block practical CPU/RAM exhaustion from oversized or malicious inputs.
# IMP-1: Added a NumPy-accelerated render path with a stdlib fallback. The synthesis now
#        uses explicit phase integration over the time-varying frequency trajectory.
# IMP-2: Added bounded LRU caching, explicit little-endian PCM16 output, and harmonic
#        anti-alias suppression above Nyquist for cleaner, more production-safe output.

"""Synthesize calm swelling tones for Twinr's processing feedback loop.

The default thinking cue should feel gentle and positive rather than like a
sharp notification. This helper generates one mono PCM16 tone that swells up
and down softly in a low humming register and can therefore replace the older
disk-backed waiting ambience as the primary processing sound.

This module intentionally stays focused on the low-disruption non-speech layer.
Higher-level explanatory or speech cues for older-adult interaction design
belong above this helper in the interaction stack.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from functools import lru_cache
import math
import sys

from twinr.hardware.audio import pcm16_to_wav_bytes

try:
    import numpy as _np
except Exception:  # pragma: no cover - optional acceleration path
    _np = None


_PCM16_MAX_ABS = 32767
_MONO_CHANNELS = 1

_INT16_HEADROOM = 0.999
_NYQUIST_MARGIN = 0.98

# Practical deployment bounds for a Raspberry Pi 4 feedback-tone helper.
_MAX_SAMPLE_RATE_HZ = 48_000
_MAX_DURATION_MS = 20_000
_MAX_FRAME_COUNT = (_MAX_SAMPLE_RATE_HZ * _MAX_DURATION_MS) // 1000
_RENDER_CACHE_SIZE = 32


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


@dataclass(frozen=True, slots=True)
class _NormalizedRenderRequest:
    frame_count: int
    sample_rate: int
    peak_gain: float
    base_frequency_hz: float
    crest_frequency_hz: float
    harmonic_ratio: float
    harmonic_gain: float
    envelope_power: float


def _coerce_int(name: str, value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer-compatible value.") from exc


def _coerce_float(name: str, value: object) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite float-compatible value.") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


def _clamp(value: float, lower: float, upper: float) -> float:
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def _normalize_render_request(
    spec: SwellingFeedbackToneSpec,
    *,
    sample_rate: int,
    peak_gain: float,
) -> _NormalizedRenderRequest:
    normalized_sample_rate = _coerce_int("sample_rate", sample_rate)
    # BREAKING: structural invalid inputs now fail fast instead of being silently
    # coerced to 1 Hz.
    if normalized_sample_rate <= 0:
        raise ValueError("sample_rate must be > 0.")
    # BREAKING: oversized renders are now rejected instead of consuming unbounded
    # CPU/RAM on-device.
    if normalized_sample_rate > _MAX_SAMPLE_RATE_HZ:
        raise ValueError(
            f"sample_rate must be <= {_MAX_SAMPLE_RATE_HZ} Hz for safe on-device rendering."
        )

    normalized_duration_ms = _coerce_int("spec.duration_ms", spec.duration_ms)
    # BREAKING: structural invalid durations now fail fast instead of being silently
    # coerced to 1 ms.
    if normalized_duration_ms <= 0:
        raise ValueError("spec.duration_ms must be > 0.")
    if normalized_duration_ms > _MAX_DURATION_MS:
        raise ValueError(
            f"spec.duration_ms must be <= {_MAX_DURATION_MS} ms for safe on-device rendering."
        )

    frame_count = max(1, round((normalized_sample_rate * normalized_duration_ms) / 1000))
    if frame_count > _MAX_FRAME_COUNT:
        raise ValueError(
            f"Requested render of {frame_count} frames exceeds the safe limit of {_MAX_FRAME_COUNT}."
        )

    normalized_peak_gain = _clamp(_coerce_float("peak_gain", peak_gain), 0.0, 1.0)

    base_frequency_hz = _coerce_float("spec.base_frequency_hz", spec.base_frequency_hz)
    if base_frequency_hz <= 0.0:
        raise ValueError("spec.base_frequency_hz must be > 0.")

    crest_frequency_hz = max(
        base_frequency_hz,
        _coerce_float("spec.crest_frequency_hz", spec.crest_frequency_hz),
    )

    nyquist_margin_hz = normalized_sample_rate * 0.5 * _NYQUIST_MARGIN
    # BREAKING: the renderer now rejects fundamentals that would alias badly for the
    # chosen sample rate instead of emitting invalid audio.
    if crest_frequency_hz >= nyquist_margin_hz:
        raise ValueError(
            "spec.crest_frequency_hz must remain below the practical Nyquist limit "
            "for the chosen sample_rate."
        )

    harmonic_ratio = max(1.0, _coerce_float("spec.harmonic_ratio", spec.harmonic_ratio))
    harmonic_gain = max(0.0, _coerce_float("spec.harmonic_gain", spec.harmonic_gain))
    envelope_power = max(1.0, _coerce_float("spec.envelope_power", spec.envelope_power))

    return _NormalizedRenderRequest(
        frame_count=frame_count,
        sample_rate=normalized_sample_rate,
        peak_gain=normalized_peak_gain,
        base_frequency_hz=base_frequency_hz,
        crest_frequency_hz=crest_frequency_hz,
        harmonic_ratio=harmonic_ratio,
        harmonic_gain=harmonic_gain,
        envelope_power=envelope_power,
    )


def _quantize_pcm16_bytes(
    samples: array | list[float],
    *,
    frame_count: int,
    peak_gain: float,
) -> bytes:
    if frame_count <= 0:
        return b""
    if peak_gain <= 0.0:
        return b""

    peak_abs = max((abs(sample) for sample in samples), default=0.0)
    if peak_abs <= 0.0:
        return b"\x00\x00" * frame_count

    # Preserve original peak_gain behavior when the waveform already fits in range,
    # but automatically back off if additive harmonics would otherwise clip.
    scale = (peak_gain * _INT16_HEADROOM) / max(1.0, peak_gain * peak_abs)

    pcm_samples = array(
        "h",
        (
            int(round(_clamp(sample * scale, -1.0, 1.0) * _PCM16_MAX_ABS))
            for sample in samples
        ),
    )
    if sys.byteorder != "little":
        pcm_samples.byteswap()
    return pcm_samples.tobytes()


def _render_swelling_feedback_tone_pcm16_python(
    render: _NormalizedRenderRequest,
) -> bytes:
    if render.peak_gain <= 0.0:
        return b""

    samples = array("f")
    frame_denominator = max(1, render.frame_count - 1)
    nyquist_margin_hz = render.sample_rate * 0.5 * _NYQUIST_MARGIN

    fundamental_phase = 0.0
    harmonic_phase = 0.0

    for sample_index in range(render.frame_count):
        progress = sample_index / frame_denominator
        swell_progress = math.sin(math.pi * progress)
        envelope = swell_progress**render.envelope_power

        current_frequency_hz = render.base_frequency_hz + (
            (render.crest_frequency_hz - render.base_frequency_hz)
            * (swell_progress * swell_progress)
        )
        phase_step = (math.tau * current_frequency_hz) / render.sample_rate

        sample_value = math.sin(fundamental_phase)
        if (
            render.harmonic_gain > 0.0
            and (current_frequency_hz * render.harmonic_ratio) < nyquist_margin_hz
        ):
            sample_value += render.harmonic_gain * math.sin(harmonic_phase)

        samples.append(sample_value * envelope)
        fundamental_phase += phase_step
        harmonic_phase += phase_step * render.harmonic_ratio

    return _quantize_pcm16_bytes(
        samples,
        frame_count=render.frame_count,
        peak_gain=render.peak_gain,
    )


def _render_swelling_feedback_tone_pcm16_numpy(
    render: _NormalizedRenderRequest,
) -> bytes:
    if render.peak_gain <= 0.0:
        return b""

    progress = _np.linspace(0.0, 1.0, render.frame_count, dtype=_np.float64)
    swell_progress = _np.sin(math.pi * progress)
    envelope = _np.power(swell_progress, render.envelope_power)

    current_frequency_hz = render.base_frequency_hz + (
        (render.crest_frequency_hz - render.base_frequency_hz)
        * (swell_progress * swell_progress)
    )

    phase_step = (math.tau * current_frequency_hz) / render.sample_rate
    fundamental_phase = _np.empty(render.frame_count, dtype=_np.float64)
    fundamental_phase[0] = 0.0
    if render.frame_count > 1:
        _np.cumsum(phase_step[:-1], out=fundamental_phase[1:])

    waveform = _np.sin(fundamental_phase)

    if render.harmonic_gain > 0.0:
        harmonic_frequency_hz = current_frequency_hz * render.harmonic_ratio
        harmonic_phase = fundamental_phase * render.harmonic_ratio
        harmonic_mask = harmonic_frequency_hz < (
            render.sample_rate * 0.5 * _NYQUIST_MARGIN
        )
        waveform += render.harmonic_gain * _np.sin(harmonic_phase) * harmonic_mask

    waveform *= envelope

    peak_abs = float(_np.max(_np.abs(waveform))) if render.frame_count > 0 else 0.0
    if peak_abs <= 0.0:
        return b"\x00\x00" * render.frame_count

    scale = (render.peak_gain * _INT16_HEADROOM) / max(
        1.0,
        render.peak_gain * peak_abs,
    )
    waveform *= scale

    pcm = _np.clip(
        _np.rint(waveform * _PCM16_MAX_ABS),
        -_PCM16_MAX_ABS,
        _PCM16_MAX_ABS,
    ).astype("<i2", copy=False)
    return pcm.tobytes()


@lru_cache(maxsize=_RENDER_CACHE_SIZE)
def _build_swelling_feedback_tone_pcm16_cached(
    render: _NormalizedRenderRequest,
) -> bytes:
    if _np is not None:
        return _render_swelling_feedback_tone_pcm16_numpy(render)
    return _render_swelling_feedback_tone_pcm16_python(render)


def build_swelling_feedback_tone_pcm16(
    spec: SwellingFeedbackToneSpec,
    *,
    sample_rate: int,
    peak_gain: float,
) -> bytes:
    """Return one mono PCM16 swelling tone."""

    render = _normalize_render_request(
        spec,
        sample_rate=sample_rate,
        peak_gain=peak_gain,
    )
    return _build_swelling_feedback_tone_pcm16_cached(render)


def build_swelling_feedback_tone_wav_bytes(
    spec: SwellingFeedbackToneSpec,
    *,
    sample_rate: int,
    peak_gain: float,
) -> bytes:
    """Wrap the synthesized swelling tone in a WAV container."""

    render = _normalize_render_request(
        spec,
        sample_rate=sample_rate,
        peak_gain=peak_gain,
    )
    pcm_bytes = _build_swelling_feedback_tone_pcm16_cached(render)
    return pcm16_to_wav_bytes(
        pcm_bytes,
        sample_rate=render.sample_rate,
        channels=_MONO_CHANNELS,
    )


__all__ = [
    "PROCESSING_SWELL_TONE_SPEC",
    "SwellingFeedbackToneSpec",
    "build_swelling_feedback_tone_pcm16",
    "build_swelling_feedback_tone_wav_bytes",
]