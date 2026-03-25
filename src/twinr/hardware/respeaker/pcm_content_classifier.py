"""Classify short PCM windows as speech-like or non-speech-like.

This module adds one local content discriminator on top of XVF3800 host-control
signals. The XVF3800 can report confident directional speech on some strong
loudspeaker playback or other room sounds. To avoid trusting those flags
blindly, Twinr also inspects the captured PCM window itself with a lightweight
deterministic classifier derived from generic synthetic speech/music/noise
signals.

The classifier is intentionally conservative:
- it only overrides XVF3800 speech when the PCM window looks strongly
  non-speech-like
- ambiguous windows stay undecided so upstream host-control can continue to win
- it is not a voice-activation or ASR model and should only be used for speech vs.
  non-speech/media suppression decisions

The frozen weights come from one mixed corpus:
- generic synthetic speech/music/noise windows for broad category coverage
- device-near playback and capture anchors from the real ReSpeaker Pi path so
  the classifier can reject playback-shaped "speech" that XVF3800 host-control
  alone may over-call
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.signal import resample_poly


_TARGET_SAMPLE_RATE = 16_000
_FRAME_SIZE = 512
_FRAME_HOP = 256
_ROLL_PERCENTILE = 0.85
_SPEECH_AUTOCORR_MIN_HZ = 70
_SPEECH_AUTOCORR_MAX_HZ = 400

# These weights were fit once from one deterministic mixed corpus of synthetic
# speech/music/noise plus bounded ReSpeaker device anchors, then frozen for
# runtime use.
_FEATURE_MEAN = np.asarray(
    [
        0.23916731567505385,
        0.3185235470156526,
        0.10037905373562183,
        0.2573938221774599,
        0.11402453900913947,
        0.15283665381964506,
        0.3097999227104236,
        0.0007237913960996225,
        0.4969423688167092,
    ],
    dtype=np.float32,
)
_FEATURE_STD = np.asarray(
    [
        0.08761466228086797,
        0.2546660455010195,
        0.07721877607365182,
        0.18758581967655743,
        0.13073389908345735,
        0.09158623570191045,
        0.19973594591850777,
        0.00043447964285781676,
        0.1644866210715742,
    ],
    dtype=np.float32,
)
_FEATURE_WEIGHTS = np.asarray(
    [
        2.3848971281314274,
        1.9679986671909917,
        1.254771607951864,
        0.08297516944327602,
        2.553471296798467,
        0.09270477731203835,
        -0.34503742613134366,
        1.289488019021462,
        1.1097800751192295,
    ],
    dtype=np.float32,
)
_FEATURE_INTERCEPT = -2.585492454986859
_STRONG_NON_SPEECH_MAX = 0.30
_SPEECH_LIKELY_MIN = 0.50


@dataclass(frozen=True, slots=True)
class PcmSpeechDiscriminatorEvidence:
    """Store bounded speech-likeness evidence for one PCM window."""

    speech_probability: float | None = None
    strong_non_speech: bool | None = None
    speech_likely: bool | None = None


def classify_pcm_speech_likeness(
    pcm_bytes: bytes | None,
    *,
    sample_rate: int | None,
    channels: int | None,
) -> PcmSpeechDiscriminatorEvidence:
    """Return conservative speech-likeness evidence for one PCM capture window."""

    samples = _load_mono_float32_samples(
        pcm_bytes,
        sample_rate=sample_rate,
        channels=channels,
    )
    if samples is None:
        return PcmSpeechDiscriminatorEvidence()
    feature_vector = _compute_feature_vector(samples)
    if feature_vector is None:
        return PcmSpeechDiscriminatorEvidence()
    normalized = (feature_vector - _FEATURE_MEAN) / _FEATURE_STD
    score = float(np.dot(normalized, _FEATURE_WEIGHTS) + _FEATURE_INTERCEPT)
    probability = _sigmoid(score)
    return PcmSpeechDiscriminatorEvidence(
        speech_probability=probability,
        strong_non_speech=probability <= _STRONG_NON_SPEECH_MAX,
        speech_likely=probability >= _SPEECH_LIKELY_MIN,
    )


def _load_mono_float32_samples(
    pcm_bytes: bytes | None,
    *,
    sample_rate: int | None,
    channels: int | None,
) -> np.ndarray | None:
    """Normalize one PCM16 window into mono float32 samples at 16 kHz."""

    if not pcm_bytes:
        return None
    normalized_sample_rate = _coerce_positive_int(sample_rate)
    normalized_channels = _coerce_positive_int(channels)
    if normalized_sample_rate is None or normalized_channels is None:
        return None
    usable_bytes = len(pcm_bytes) - (len(pcm_bytes) % 2)
    if usable_bytes <= 0:
        return None
    samples = np.frombuffer(pcm_bytes[:usable_bytes], dtype=np.int16).astype(np.float32)
    usable_samples = len(samples) - (len(samples) % normalized_channels)
    if usable_samples <= 0:
        return None
    samples = samples[:usable_samples]
    if normalized_channels > 1:
        samples = samples.reshape(-1, normalized_channels).mean(axis=1)
    if normalized_sample_rate != _TARGET_SAMPLE_RATE:
        samples = resample_poly(samples, _TARGET_SAMPLE_RATE, normalized_sample_rate)
    samples = samples / 32768.0
    if samples.size <= 0:
        return None
    return np.clip(samples.astype(np.float32, copy=False), -1.0, 1.0)


def _compute_feature_vector(samples: np.ndarray) -> np.ndarray | None:
    """Extract one fixed feature vector from normalized mono samples."""

    frame_count = max(1, ((samples.shape[0] - _FRAME_SIZE) // _FRAME_HOP) + 1)
    if frame_count <= 0:
        return None
    rms_values: list[float] = []
    zcr_values: list[float] = []
    flatness_values: list[float] = []
    flatness_std_values: list[float] = []
    centroid_values: list[float] = []
    roll_values: list[float] = []
    flux_values: list[float] = []
    autocorr_values: list[float] = []
    previous_distribution: np.ndarray | None = None
    window = np.hanning(_FRAME_SIZE).astype(np.float32)
    frequencies = np.fft.rfftfreq(_FRAME_SIZE, 1.0 / _TARGET_SAMPLE_RATE).astype(np.float32)
    minimum_lag = max(1, int(_TARGET_SAMPLE_RATE / _SPEECH_AUTOCORR_MAX_HZ))
    maximum_lag = max(minimum_lag + 1, int(_TARGET_SAMPLE_RATE / _SPEECH_AUTOCORR_MIN_HZ))

    for start in range(0, max(1, samples.shape[0] - _FRAME_SIZE + 1), _FRAME_HOP):
        frame = samples[start:start + _FRAME_SIZE]
        if frame.shape[0] < _FRAME_SIZE:
            frame = np.pad(frame, (0, _FRAME_SIZE - frame.shape[0]))
        rms = float(np.sqrt(np.mean(frame * frame) + 1e-12))
        rms_values.append(rms)
        zcr_values.append(float(np.mean(np.abs(np.diff(np.signbit(frame))))))

        spectrum = np.abs(np.fft.rfft(frame * window)) + 1e-9
        spectral_distribution = spectrum / float(np.sum(spectrum))
        flatness = float(np.exp(np.mean(np.log(spectrum))) / np.mean(spectrum))
        flatness_values.append(flatness)
        centroid_values.append(
            float(np.sum(frequencies * spectral_distribution) / (_TARGET_SAMPLE_RATE / 2.0))
        )
        cumulative = np.cumsum(spectral_distribution)
        roll_index = int(np.searchsorted(cumulative, _ROLL_PERCENTILE))
        roll_index = min(max(0, roll_index), len(frequencies) - 1)
        roll_values.append(float(frequencies[roll_index] / (_TARGET_SAMPLE_RATE / 2.0)))
        if previous_distribution is not None:
            flux_values.append(
                float(np.mean(np.maximum(0.0, spectral_distribution - previous_distribution)))
            )
        previous_distribution = spectral_distribution

        autocorrelation = np.correlate(frame, frame, mode="full")[_FRAME_SIZE - 1:]
        autocorrelation[0] = 0.0
        if autocorrelation.shape[0] > maximum_lag:
            peak = float(np.max(autocorrelation[minimum_lag:maximum_lag]))
        else:
            peak = 0.0
        autocorr_values.append(peak / float(np.sum(frame * frame) + 1e-9))

    rms_array = np.asarray(rms_values, dtype=np.float32)
    flatness_array = np.asarray(flatness_values, dtype=np.float32)
    flux_array = np.asarray(flux_values if flux_values else [0.0], dtype=np.float32)
    return np.asarray(
        [
            float(np.mean(rms_array)),
            float(np.std(rms_array) / (np.mean(rms_array) + 1e-9)),
            float(np.mean(np.asarray(zcr_values, dtype=np.float32))),
            float(np.mean(flatness_array)),
            float(np.std(flatness_array)),
            float(np.mean(np.asarray(centroid_values, dtype=np.float32))),
            float(np.mean(np.asarray(roll_values, dtype=np.float32))),
            float(np.mean(flux_array)),
            float(np.mean(np.asarray(autocorr_values, dtype=np.float32))),
        ],
        dtype=np.float32,
    )


def _coerce_positive_int(value: object) -> int | None:
    """Return one positive integer or ``None`` when malformed."""

    if isinstance(value, bool):
        return None
    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if normalized <= 0:
        return None
    return normalized


def _sigmoid(score: float) -> float:
    """Return one bounded logistic probability."""

    if score >= 0.0:
        exp = math.exp(-score)
        return 1.0 / (1.0 + exp)
    exp = math.exp(score)
    return exp / (1.0 + exp)
