# CHANGELOG: 2026-03-28
# BUG-1: Included the trailing partial frame during feature extraction instead of silently
#        dropping the last 1..255 samples of most windows.
# BUG-2: Replaced naive multi-channel averaging with phase-aware energy-preserving fold-down
#        to avoid false non-speech on multi-mic / anti-phase inputs.
# SEC-1: Added hard limits for input size / duration / sample rate / channel count plus bounded
#        LRU stream caches so malformed or malicious inputs cannot burn CPU/RAM on a Raspberry Pi 4.
# IMP-1: Added optional stateful Silero VAD ONNX fusion backend with one shared ORT session and
#        per-stream state, while keeping the old DSP classifier as a deterministic guardrail.
# IMP-2: Added optional playback-reference-aware echo suppression / similarity scoring so
#        loudspeaker playback can be penalized before speech decisions.

"""Classify short PCM windows as speech-like or non-speech-like.

This module adds one local content discriminator on top of XVF3800 host-control
signals. The XVF3800 can report confident directional speech on some strong
loudspeaker playback or other room sounds. To avoid trusting those flags blindly,
Twinr also inspects the captured PCM window itself.

2026 upgrade:
- keep the frozen DSP classifier as one deterministic media/playback guardrail
- optionally fuse it with a streaming Silero VAD ONNX backend
- optionally use a playback reference window to subtract loudspeaker leakage
- preserve the original call shape while exposing optional stream/reference args

The classifier remains conservative:
- it only overrides XVF3800 speech when the PCM window looks strongly
  non-speech-like
- ambiguous windows stay undecided so upstream host-control can continue to win
- it is not a wake word, endpointing, diarization, or ASR module
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import threading
import time
from typing import Iterable

import numpy as np
from scipy.signal import resample_poly


_TARGET_SAMPLE_RATE = 16_000
_FRAME_SIZE = 512
_FRAME_HOP = 256
_ROLL_PERCENTILE = 0.85
_SPEECH_AUTOCORR_MIN_HZ = 70
_SPEECH_AUTOCORR_MAX_HZ = 400

_MAX_WINDOW_SECONDS = 2.0
_MAX_SAMPLE_RATE = 192_000
_MAX_CHANNELS = 16
_MAX_STREAM_CACHE_SIZE = 64
_STREAM_STATE_TTL_SECONDS = 30.0
_MAX_STREAM_ID_BYTES = 256

_DEFAULT_MODEL_ENV = "TWINR_SILERO_VAD_ONNX_PATH"
_DEFAULT_MODEL_CANDIDATES = (
    "/opt/twinr/models/silero_vad_v6_16k.onnx",
    "/opt/twinr/models/silero_vad_16k.onnx",
    "/usr/local/share/twinr/silero_vad_v6_16k.onnx",
    "/usr/local/share/twinr/silero_vad_16k.onnx",
)

_REFERENCE_ALIGN_DOWNSAMPLE = 4
_REFERENCE_ALIGN_MAX_LAG_MS = 120
_REFERENCE_ALIGN_COARSE_STEP = 4
_REFERENCE_ALIGN_REFINE_RADIUS = 32

_STRONG_NON_SPEECH_MAX = 0.30
_SPEECH_LIKELY_MIN = 0.50

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

_WINDOW = np.hanning(_FRAME_SIZE).astype(np.float32)
_FREQUENCIES = np.fft.rfftfreq(_FRAME_SIZE, 1.0 / _TARGET_SAMPLE_RATE).astype(np.float32)
_MINIMUM_LAG = max(1, int(_TARGET_SAMPLE_RATE / _SPEECH_AUTOCORR_MAX_HZ))
_MAXIMUM_LAG = max(_MINIMUM_LAG + 1, int(_TARGET_SAMPLE_RATE / _SPEECH_AUTOCORR_MIN_HZ))
_SAFE_FEATURE_STD = np.where(_FEATURE_STD > 1e-9, _FEATURE_STD, 1.0).astype(np.float32)

_STREAM_FUSION_LOCK = threading.RLock()
_STREAM_FUSION_STATE: OrderedDict[str, "_FusionStreamState"] = OrderedDict()
_BACKEND_LOCK = threading.RLock()
_ONNX_BACKENDS: dict[Path, "_SileroOnnxBackend"] = {}


@dataclass(frozen=True, slots=True)
class PcmSpeechDiscriminatorEvidence:
    speech_probability: float | None = None
    strong_non_speech: bool | None = None
    speech_likely: bool | None = None
    backend: str | None = None
    dsp_speech_probability: float | None = None
    neural_speech_probability: float | None = None
    playback_similarity: float | None = None
    echo_dominance: float | None = None


@dataclass(slots=True)
class _FusionStreamState:
    smoothed_probability: float
    last_seen: float


@dataclass(slots=True)
class _OnnxStreamState:
    state: np.ndarray
    context: np.ndarray
    last_seen: float


class _SileroOnnxBackend:
    """Shared Silero VAD ONNX session with bounded per-stream state."""

    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            os.fspath(model_path),
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
        inputs = self._session.get_inputs()
        self._input_name = inputs[0].name
        self._state_name = inputs[1].name
        self._sr_name = inputs[2].name
        self._lock = threading.RLock()
        self._states: OrderedDict[str, _OnnxStreamState] = OrderedDict()

    def score_window(self, samples: np.ndarray, *, stream_key: str | None, end_of_stream: bool) -> float | None:
        if samples.size <= 0:
            if stream_key is not None and end_of_stream:
                self.clear_stream(stream_key)
            return None

        chunk_size, context_size = 512, 64
        if samples.shape[0] > int(_TARGET_SAMPLE_RATE * _MAX_WINDOW_SECONDS):
            samples = samples[-int(_TARGET_SAMPLE_RATE * _MAX_WINDOW_SECONDS):]

        state: np.ndarray | None = None
        context: np.ndarray | None = None
        with self._lock:
            self._evict_stale_locked()
            if stream_key is not None:
                entry = self._states.get(stream_key)
                if entry is not None:
                    entry.last_seen = time.monotonic()
                    self._states.move_to_end(stream_key)
                    state = entry.state.copy()
                    context = entry.context.copy()

        if state is None:
            state = np.zeros((2, 1, 128), dtype=np.float32)
        if context is None:
            context = np.zeros((1, context_size), dtype=np.float32)

        padded_length = int(math.ceil(samples.shape[0] / chunk_size) * chunk_size)
        if padded_length != samples.shape[0]:
            samples = np.pad(samples, (0, padded_length - samples.shape[0]))

        probabilities: list[float] = []
        for start in range(0, samples.shape[0], chunk_size):
            chunk = samples[start:start + chunk_size]
            x = np.concatenate((context, chunk[np.newaxis, :]), axis=1).astype(np.float32, copy=False)
            ort_inputs = {
                self._input_name: x,
                self._state_name: state,
                self._sr_name: np.asarray(_TARGET_SAMPLE_RATE, dtype=np.int64),
            }
            try:
                out, state = self._session.run(None, ort_inputs)
            except Exception:
                if stream_key is not None and end_of_stream:
                    self.clear_stream(stream_key)
                return None
            probabilities.append(float(np.asarray(out, dtype=np.float32).reshape(-1)[0]))
            context = x[:, -context_size:]

        if stream_key is not None and not end_of_stream:
            with self._lock:
                self._states[stream_key] = _OnnxStreamState(
                    state=np.asarray(state, dtype=np.float32),
                    context=np.asarray(context, dtype=np.float32),
                    last_seen=time.monotonic(),
                )
                self._states.move_to_end(stream_key)
                while len(self._states) > _MAX_STREAM_CACHE_SIZE:
                    self._states.popitem(last=False)
        elif stream_key is not None:
            self.clear_stream(stream_key)

        return _aggregate_chunk_probabilities(probabilities)

    def clear_stream(self, stream_key: str) -> None:
        with self._lock:
            self._states.pop(stream_key, None)

    def _evict_stale_locked(self) -> None:
        now = time.monotonic()
        stale = [k for k, v in self._states.items() if (now - v.last_seen) > _STREAM_STATE_TTL_SECONDS]
        for key in stale:
            self._states.pop(key, None)


def classify_pcm_speech_likeness(
    pcm_bytes: bytes | None,
    *,
    sample_rate: int | None,
    channels: int | None,
    stream_id: object | None = None,
    end_of_stream: bool = False,
    playback_pcm_bytes: bytes | None = None,
    playback_sample_rate: int | None = None,
    playback_channels: int | None = None,
    use_neural_vad: bool = True,
    neural_vad_model_path: str | os.PathLike[str] | None = None,
) -> PcmSpeechDiscriminatorEvidence:
    """Return conservative speech-likeness evidence for one PCM capture window.

    Existing callers can keep using the original three arguments. New optional
    parameters enable stateful streaming inference and playback-aware scoring.
    """

    stream_key = _normalize_stream_key(stream_id)
    try:
        samples = _load_mono_float32_samples(pcm_bytes, sample_rate=sample_rate, channels=channels)
    except Exception:
        samples = None
    if samples is None:
        if stream_key is not None and end_of_stream:
            reset_pcm_speech_discriminator_stream(stream_id)
        return PcmSpeechDiscriminatorEvidence()

    playback_similarity: float | None = None
    echo_dominance: float | None = None
    analysis_samples = samples
    if playback_pcm_bytes:
        try:
            playback_samples = _load_mono_float32_samples(
                playback_pcm_bytes,
                sample_rate=playback_sample_rate,
                channels=playback_channels,
            )
        except Exception:
            playback_samples = None
        if playback_samples is not None:
            ref_result = _estimate_playback_leakage(samples, playback_samples)
            if ref_result is not None:
                analysis_samples, playback_similarity, echo_dominance = ref_result

    feature_vector = _compute_feature_vector(analysis_samples)
    if feature_vector is None:
        if stream_key is not None and end_of_stream:
            reset_pcm_speech_discriminator_stream(stream_id)
        return PcmSpeechDiscriminatorEvidence()

    dsp_probability = _score_dsp_feature_vector(feature_vector)
    neural_probability = None
    backend_name = "dsp-only"

    if use_neural_vad:
        backend = _get_silero_onnx_backend(neural_vad_model_path)
        if backend is not None:
            backend_name = "silero-onnx+dsp"
            neural_probability = backend.score_window(
                analysis_samples,
                stream_key=stream_key,
                end_of_stream=end_of_stream,
            )

    fused_probability = _fuse_probabilities(
        dsp_probability=dsp_probability,
        neural_probability=neural_probability,
        playback_similarity=playback_similarity,
        echo_dominance=echo_dominance,
    )
    fused_probability = _apply_stream_smoothing(
        fused_probability,
        stream_key=stream_key,
        end_of_stream=end_of_stream,
    )

    return PcmSpeechDiscriminatorEvidence(
        speech_probability=fused_probability,
        strong_non_speech=fused_probability <= _STRONG_NON_SPEECH_MAX,
        speech_likely=fused_probability >= _SPEECH_LIKELY_MIN,
        backend=backend_name,
        dsp_speech_probability=dsp_probability,
        neural_speech_probability=neural_probability,
        playback_similarity=playback_similarity,
        echo_dominance=echo_dominance,
    )


def reset_pcm_speech_discriminator_stream(stream_id: object) -> None:
    """Drop any cached per-stream state for one stream identifier."""
    stream_key = _normalize_stream_key(stream_id)
    if stream_key is None:
        return
    with _STREAM_FUSION_LOCK:
        _STREAM_FUSION_STATE.pop(stream_key, None)
    with _BACKEND_LOCK:
        for backend in _ONNX_BACKENDS.values():
            backend.clear_stream(stream_key)


def _load_mono_float32_samples(
    pcm_bytes: bytes | None,
    *,
    sample_rate: int | None,
    channels: int | None,
) -> np.ndarray | None:
    """Normalize one PCM16 window into mono float32 samples at 16 kHz."""
    if not pcm_bytes:
        return None

    sr = _coerce_positive_int(sample_rate)
    ch = _coerce_positive_int(channels)
    if sr is None or ch is None or sr > _MAX_SAMPLE_RATE or ch > _MAX_CHANNELS:
        return None

    max_input_samples_per_channel = max(1, int(sr * _MAX_WINDOW_SECONDS))
    max_total_bytes = max_input_samples_per_channel * ch * 2
    if len(pcm_bytes) > max_total_bytes:
        pcm_bytes = pcm_bytes[-max_total_bytes:]

    usable_bytes = len(pcm_bytes) - (len(pcm_bytes) % 2)
    if usable_bytes <= 0:
        return None

    try:
        samples = np.frombuffer(memoryview(pcm_bytes)[:usable_bytes], dtype="<i2").astype(np.float32, copy=True)
    except (TypeError, ValueError, BufferError, MemoryError):
        return None

    usable_samples = len(samples) - (len(samples) % ch)
    if usable_samples <= 0:
        return None
    samples = samples[:usable_samples]

    if ch > 1:
        samples = _fold_channels_phase_aware(samples.reshape(-1, ch))
    else:
        samples = samples.astype(np.float32, copy=False)

    samples -= float(np.mean(samples))
    if sr != _TARGET_SAMPLE_RATE:
        up, down = _TARGET_SAMPLE_RATE, sr
        gcd = math.gcd(up, down)
        up //= gcd
        down //= gcd
        try:
            samples = resample_poly(samples, up, down).astype(np.float32, copy=False)
        except (ValueError, MemoryError):
            return None

    if samples.size <= 0:
        return None

    max_target_samples = int(_TARGET_SAMPLE_RATE * _MAX_WINDOW_SECONDS)
    if samples.shape[0] > max_target_samples:
        samples = samples[-max_target_samples:]

    samples = samples / 32768.0
    samples -= float(np.mean(samples))
    return np.clip(samples.astype(np.float32, copy=False), -1.0, 1.0)


def _fold_channels_phase_aware(samples: np.ndarray) -> np.ndarray:
    """Collapse multichannel PCM without destructive anti-phase loss."""
    if samples.ndim != 2 or samples.shape[0] <= 0:
        return samples.reshape(-1).astype(np.float32, copy=False)

    rms = np.sqrt(np.mean(samples * samples, axis=0) + 1e-12, dtype=np.float32)
    dominant = int(np.argmax(rms))
    reference = samples[:, dominant].astype(np.float32, copy=False)

    aligned = samples.astype(np.float32, copy=True)
    for i in range(aligned.shape[1]):
        if i == dominant:
            continue
        if float(np.dot(reference, aligned[:, i])) < 0.0:
            aligned[:, i] *= -1.0

    weight_sum = float(np.sum(rms))
    if weight_sum <= 1e-9:
        return np.mean(aligned, axis=1, dtype=np.float32)
    weights = (rms / weight_sum).astype(np.float32, copy=False)
    return np.sum(aligned * weights[np.newaxis, :], axis=1, dtype=np.float32)


def _compute_feature_vector(samples: np.ndarray) -> np.ndarray | None:
    """Extract one fixed feature vector from normalized mono samples."""
    if samples.size <= 0:
        return None

    starts = _frame_starts(samples.shape[0])
    rms_values: list[float] = []
    zcr_values: list[float] = []
    flatness_values: list[float] = []
    centroid_values: list[float] = []
    roll_values: list[float] = []
    flux_values: list[float] = []
    autocorr_values: list[float] = []
    previous_distribution: np.ndarray | None = None

    for start in starts:
        frame = samples[start:start + _FRAME_SIZE]
        if frame.shape[0] < _FRAME_SIZE:
            frame = np.pad(frame, (0, _FRAME_SIZE - frame.shape[0]))
        frame = frame.astype(np.float32, copy=False)

        rms = float(np.sqrt(float(np.mean(frame * frame)) + 1e-12))
        rms_values.append(rms)
        zcr_values.append(float(np.mean(np.abs(np.diff(np.signbit(frame))))))

        spectrum = np.abs(np.fft.rfft(frame * _WINDOW)) + 1e-9
        spectrum = spectrum.astype(np.float32, copy=False)
        spectral_distribution = spectrum / float(np.sum(spectrum))
        flatness_values.append(float(np.exp(np.mean(np.log(spectrum))) / np.mean(spectrum)))
        centroid_values.append(
            float(np.sum(_FREQUENCIES * spectral_distribution) / (_TARGET_SAMPLE_RATE / 2.0))
        )
        cumulative = np.cumsum(spectral_distribution)
        roll_index = int(np.searchsorted(cumulative, _ROLL_PERCENTILE))
        roll_index = min(max(0, roll_index), len(_FREQUENCIES) - 1)
        roll_values.append(float(_FREQUENCIES[roll_index] / (_TARGET_SAMPLE_RATE / 2.0)))

        if previous_distribution is not None:
            flux_values.append(float(np.mean(np.maximum(0.0, spectral_distribution - previous_distribution))))
        previous_distribution = spectral_distribution

        autocorrelation = np.correlate(frame, frame, mode="full")[_FRAME_SIZE - 1:]
        autocorrelation[0] = 0.0
        peak = float(np.max(autocorrelation[_MINIMUM_LAG:_MAXIMUM_LAG])) if autocorrelation.shape[0] > _MAXIMUM_LAG else 0.0
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


def _frame_starts(sample_count: int) -> tuple[int, ...]:
    """Return frame starts and always include the trailing partial frame."""
    if sample_count <= 0 or sample_count <= _FRAME_SIZE:
        return (0,)
    starts = list(range(0, sample_count - _FRAME_SIZE + 1, _FRAME_HOP))
    last_start = sample_count - _FRAME_SIZE
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return tuple(starts)


def _estimate_playback_leakage(
    microphone_samples: np.ndarray,
    playback_samples: np.ndarray,
) -> tuple[np.ndarray, float, float] | None:
    """Estimate playback leakage, subtract it, and return residual audio."""
    if microphone_samples.size < 256 or playback_samples.size < 256:
        return None

    max_len = min(microphone_samples.size, playback_samples.size)
    if max_len <= 0:
        return None

    mic = microphone_samples[-max_len:].astype(np.float32, copy=False)
    ref = playback_samples[-max_len:].astype(np.float32, copy=False)

    lag = _find_best_alignment_lag(mic, ref)
    aligned_ref, mic_view = _align_pair(mic, ref, lag)
    if aligned_ref.size < 256 or mic_view.size < 256:
        return None

    ref_energy = float(np.dot(aligned_ref, aligned_ref))
    if ref_energy <= 1e-9:
        return None

    gain = float(np.dot(mic_view, aligned_ref) / (ref_energy + 1e-9))
    gain = float(np.clip(gain, -2.0, 2.0))
    residual_view = mic_view - (gain * aligned_ref)

    similarity = float(
        abs(np.dot(mic_view, aligned_ref))
        / math.sqrt(float(np.dot(mic_view, mic_view) * ref_energy) + 1e-9)
    )
    original_rms = float(np.sqrt(np.mean(mic_view * mic_view) + 1e-12))
    residual_rms = float(np.sqrt(np.mean(residual_view * residual_view) + 1e-12))
    echo_dominance = float(np.clip(1.0 - (residual_rms / (original_rms + 1e-9)), 0.0, 1.0))

    residual = microphone_samples.copy()
    if lag >= 0:
        residual[lag:lag + residual_view.size] = residual_view
    else:
        residual[:residual_view.size] = residual_view

    if similarity >= 0.35 and echo_dominance >= 0.15:
        return residual.astype(np.float32, copy=False), similarity, echo_dominance
    return microphone_samples.astype(np.float32, copy=False), similarity, echo_dominance


def _find_best_alignment_lag(microphone_samples: np.ndarray, playback_samples: np.ndarray) -> int:
    """Find one bounded lag with a cheap coarse-to-fine cosine similarity search."""
    coarse_mic = microphone_samples[::_REFERENCE_ALIGN_DOWNSAMPLE]
    coarse_ref = playback_samples[::_REFERENCE_ALIGN_DOWNSAMPLE]
    coarse_max_lag = max(
        1,
        int((_REFERENCE_ALIGN_MAX_LAG_MS / 1000.0) * _TARGET_SAMPLE_RATE / _REFERENCE_ALIGN_DOWNSAMPLE),
    )

    best_lag, best_score = 0, -1.0
    for lag in range(-coarse_max_lag, coarse_max_lag + 1, _REFERENCE_ALIGN_COARSE_STEP):
        ref_view, mic_view = _align_pair(coarse_mic, coarse_ref, lag)
        if ref_view.size < 64:
            continue
        score = _cosine_similarity(mic_view, ref_view)
        if score > best_score:
            best_lag, best_score = lag, score

    coarse_lag_in_samples = best_lag * _REFERENCE_ALIGN_DOWNSAMPLE
    fine_start = coarse_lag_in_samples - _REFERENCE_ALIGN_REFINE_RADIUS
    fine_stop = coarse_lag_in_samples + _REFERENCE_ALIGN_REFINE_RADIUS + 1

    best_fine_lag, best_score = 0, -1.0
    for lag in range(fine_start, fine_stop):
        ref_view, mic_view = _align_pair(microphone_samples, playback_samples, lag)
        if ref_view.size < 256:
            continue
        score = _cosine_similarity(mic_view, ref_view)
        if score > best_score:
            best_fine_lag, best_score = lag, score
    return best_fine_lag


def _align_pair(
    microphone_samples: np.ndarray,
    playback_samples: np.ndarray,
    lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned playback and microphone views for one lag."""
    if lag >= 0:
        mic_view = microphone_samples[lag:]
        ref_view = playback_samples[:mic_view.size]
    else:
        ref_view = playback_samples[-lag:]
        mic_view = microphone_samples[:ref_view.size]
    size = min(mic_view.size, ref_view.size)
    if size <= 0:
        zero = np.zeros(0, dtype=np.float32)
        return zero, zero
    return ref_view[:size].astype(np.float32, copy=False), mic_view[:size].astype(np.float32, copy=False)


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = math.sqrt(float(np.dot(left, left) * np.dot(right, right)) + 1e-9)
    return 0.0 if denominator <= 1e-12 else abs(float(np.dot(left, right))) / denominator


def _score_dsp_feature_vector(feature_vector: np.ndarray) -> float:
    normalized = (feature_vector - _FEATURE_MEAN) / _SAFE_FEATURE_STD
    score = float(np.dot(normalized, _FEATURE_WEIGHTS) + _FEATURE_INTERCEPT)
    return _sigmoid(score)


def _fuse_probabilities(
    *,
    dsp_probability: float,
    neural_probability: float | None,
    playback_similarity: float | None,
    echo_dominance: float | None,
) -> float:
    """Fuse deterministic and neural probabilities conservatively."""
    fused = dsp_probability if neural_probability is None else (0.65 * neural_probability) + (0.35 * dsp_probability)
    if playback_similarity is not None and echo_dominance is not None:
        playback_penalty = playback_similarity * echo_dominance
        if playback_penalty >= 0.80:
            fused = min(fused, 0.10 if neural_probability is None else 0.15)
        elif playback_penalty >= 0.60:
            fused = min(fused, 0.18 if neural_probability is None else 0.22)
        elif playback_penalty >= 0.45:
            fused = min(fused, 0.28)
    return float(np.clip(fused, 0.0, 1.0))


def _apply_stream_smoothing(
    probability: float,
    *,
    stream_key: str | None,
    end_of_stream: bool,
) -> float:
    if stream_key is None:
        return probability

    now = time.monotonic()
    with _STREAM_FUSION_LOCK:
        _evict_stale_fusion_states_locked(now)
        entry = _STREAM_FUSION_STATE.get(stream_key)
        if entry is None:
            smoothed = probability
        else:
            previous = entry.smoothed_probability
            alpha = 0.40 if probability >= previous else 0.75
            smoothed = (alpha * previous) + ((1.0 - alpha) * probability)

        if end_of_stream:
            _STREAM_FUSION_STATE.pop(stream_key, None)
        else:
            _STREAM_FUSION_STATE[stream_key] = _FusionStreamState(smoothed_probability=smoothed, last_seen=now)
            _STREAM_FUSION_STATE.move_to_end(stream_key)
            while len(_STREAM_FUSION_STATE) > _MAX_STREAM_CACHE_SIZE:
                _STREAM_FUSION_STATE.popitem(last=False)

    return float(np.clip(smoothed, 0.0, 1.0))


def _evict_stale_fusion_states_locked(now: float | None = None) -> None:
    if now is None:
        now = time.monotonic()
    stale = [k for k, v in _STREAM_FUSION_STATE.items() if (now - v.last_seen) > _STREAM_STATE_TTL_SECONDS]
    for key in stale:
        _STREAM_FUSION_STATE.pop(key, None)


def _aggregate_chunk_probabilities(probabilities: Iterable[float]) -> float | None:
    probs = np.asarray(list(probabilities), dtype=np.float32)
    if probs.size <= 0:
        return None
    if probs.size == 1:
        return float(np.clip(probs[0], 0.0, 1.0))
    top_k = max(1, int(math.ceil(probs.size / 3.0)))
    top_mean = float(np.mean(np.partition(probs, -top_k)[-top_k:]))
    combined = (0.60 * top_mean) + (0.40 * float(np.mean(probs)))
    return float(np.clip(combined, 0.0, 1.0))


def _normalize_stream_key(stream_id: object | None) -> str | None:
    if stream_id is None:
        return None
    raw = repr(stream_id).encode("utf-8", errors="replace")
    if len(raw) > _MAX_STREAM_ID_BYTES:
        raw = raw[:_MAX_STREAM_ID_BYTES]
    return hashlib.sha1(raw).hexdigest()


def _get_silero_onnx_backend(
    neural_vad_model_path: str | os.PathLike[str] | None,
) -> _SileroOnnxBackend | None:
    model_path = _resolve_model_path(neural_vad_model_path)
    if model_path is None:
        return None
    with _BACKEND_LOCK:
        backend = _ONNX_BACKENDS.get(model_path)
        if backend is not None:
            return backend
        try:
            backend = _SileroOnnxBackend(model_path)
        except Exception:
            return None
        _ONNX_BACKENDS[model_path] = backend
        return backend


def _resolve_model_path(model_path: str | os.PathLike[str] | None) -> Path | None:
    if model_path is not None:
        candidate = Path(model_path).expanduser()
        return candidate.resolve() if candidate.is_file() else None

    env_path = os.environ.get(_DEFAULT_MODEL_ENV)
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.is_file():
            return candidate.resolve()

    for candidate_str in _DEFAULT_MODEL_CANDIDATES:
        candidate = Path(candidate_str)
        if candidate.is_file():
            return candidate.resolve()
    return None


def _coerce_positive_int(value: object) -> int | None:
    """Return one positive integer or None when malformed."""
    if isinstance(value, bool):
        return None
    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return normalized if normalized > 0 else None


def _sigmoid(score: float) -> float:
    """Return one bounded logistic probability."""
    if score >= 0.0:
        exp = math.exp(-score)
        return 1.0 / (1.0 + exp)
    exp = math.exp(score)
    return exp / (1.0 + exp)