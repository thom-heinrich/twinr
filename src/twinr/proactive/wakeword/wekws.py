"""Run Twinr wakeword detection through a trained WeKws-style ONNX model.

This module keeps conventional keyword-spotting assets, ONNX session state,
feature extraction, and clip/stream detector logic outside the runtime loop.
It mirrors the existing local-backend contract used by openWakeWord and
sherpa-onnx so Twinr can promote and replay custom stage-1 wakeword models
without inventing a parallel orchestration path.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import logging
import math
from pathlib import Path
from threading import RLock
from typing import Any, Callable

import numpy as np
import yaml

from twinr.hardware.audio import AmbientAudioCaptureWindow

from .matching import WakewordMatch, normalize_detector_label, phrase_from_detector_label

LOGGER = logging.getLogger(__name__)

_DEFAULT_FRAME_LENGTH_MS = 25.0
_DEFAULT_FRAME_SHIFT_MS = 10.0
_DEFAULT_LOW_FREQ_HZ = 20.0
_DEFAULT_HIGH_FREQ_OFFSET_HZ = -400.0
_ALLOWED_WEKWS_CMVN_MODES = frozenset({"auto", "embedded", "external", "none"})


@dataclass(frozen=True, slots=True)
class WakewordWekwsAssetBundle:
    """Describe one resolved WeKws asset bundle for runtime inference."""

    model_path: str
    config_path: str
    words_path: str
    cmvn_path: str | None = None


@dataclass(frozen=True, slots=True)
class WakewordWekwsModelConfig:
    """Describe the runtime feature contract extracted from one WeKws config."""

    sample_rate: int
    feature_dim: int
    frame_length_ms: float
    frame_shift_ms: float
    keyword_labels: tuple[str, ...]
    cmvn_mode: str = "none"
    cmvn_mean: tuple[float, ...] | None = None
    cmvn_istd: tuple[float, ...] | None = None


def _normalize_nonempty_text(name: str, value: object | None) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} must be a non-empty string.")
    return text


def _resolve_existing_file(
    path_value: object | None,
    *,
    name: str,
    project_root: str | Path | None,
) -> str:
    raw_path = _normalize_nonempty_text(name, path_value)
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute() and project_root is not None:
        candidate = Path(project_root).expanduser().resolve(strict=False) / candidate
    resolved = candidate.resolve(strict=False)
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} does not exist: {resolved}")
    return str(resolved)


def _coerce_positive_int(name: str, value: object, *, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}.") from exc
    if normalized < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return normalized


def _coerce_nonnegative_float(name: str, value: object, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a float >= {minimum}.")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a float >= {minimum}.") from exc
    if normalized < minimum:
        raise ValueError(f"{name} must be a float >= {minimum}.")
    return normalized


def _normalize_wekws_cmvn_mode(value: object | None, *, default: str = "auto") -> str:
    raw_value = default if value is None or not str(value).strip() else value
    normalized = str(raw_value).strip().lower()
    if normalized not in _ALLOWED_WEKWS_CMVN_MODES:
        allowed = ", ".join(sorted(_ALLOWED_WEKWS_CMVN_MODES))
        raise ValueError(f"WeKws cmvn mode must be one of: {allowed}.")
    return normalized


def _pcm16_to_int16_mono_samples(pcm_bytes: bytes, *, channels: int) -> np.ndarray:
    validated_channels = _coerce_positive_int("channels", channels)
    usable = len(pcm_bytes) - (len(pcm_bytes) % 2)
    if usable <= 0:
        return np.zeros(0, dtype=np.int16)
    samples = np.frombuffer(pcm_bytes[:usable], dtype=np.int16)
    usable_samples = len(samples) - (len(samples) % validated_channels)
    if usable_samples <= 0:
        return np.zeros(0, dtype=np.int16)
    samples = samples[:usable_samples]
    if validated_channels == 1:
        return samples.copy()
    mixed = (
        samples.reshape(-1, validated_channels)
        .astype(np.int32)
        .mean(axis=1)
        .round()
        .clip(-32768, 32767)
        .astype(np.int16)
    )
    return mixed


def _default_onnx_session_factory(*, model_path: str, provider: str, num_threads: int):
    """Create one WeKws ONNX runtime session lazily."""

    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - depends on runtime env.
        raise RuntimeError(
            "onnxruntime is required for the WeKws wakeword backend."
        ) from exc

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = _coerce_positive_int("num_threads", num_threads)
    session_options.inter_op_num_threads = _coerce_positive_int("num_threads", num_threads)
    normalized_provider = _normalize_nonempty_text("provider", provider).strip().lower()
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if normalized_provider == "cuda"
        else ["CPUExecutionProvider"]
    )
    return ort.InferenceSession(model_path, sess_options=session_options, providers=providers)


def _load_cmvn(path: Path) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Load WeKws JSON CMVN stats and convert them into mean/istd vectors."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    mean_stat = list(payload["mean_stat"])
    var_stat = list(payload["var_stat"])
    frame_num = float(payload["frame_num"])
    if frame_num <= 0:
        raise ValueError(f"{path} must contain a positive frame_num.")
    means: list[float] = []
    inverse_stddevs: list[float] = []
    for mean_value, var_value in zip(mean_stat, var_stat, strict=True):
        normalized_mean = float(mean_value) / frame_num
        variance = float(var_value) / frame_num - (normalized_mean * normalized_mean)
        variance = max(variance, 1.0e-20)
        means.append(normalized_mean)
        inverse_stddevs.append(float(1.0 / np.sqrt(variance)))
    return tuple(means), tuple(inverse_stddevs)


def _load_keyword_labels(path: Path) -> tuple[str, ...]:
    """Load WeKws output keyword labels from ``words.txt``."""

    labels: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        token = raw_line.strip()
        if not token:
            continue
        normalized = normalize_detector_label(token)
        if not normalized or "filler" in normalized.split():
            continue
        labels.append(token)
    if not labels:
        raise ValueError(f"{path} did not contain any keyword labels.")
    return tuple(labels)


def _resolve_cmvn_path(
    *,
    config_path: Path,
    explicit_cmvn_path: str | None,
    config_payload: dict[str, Any],
) -> str | None:
    if explicit_cmvn_path:
        return str(Path(explicit_cmvn_path).expanduser().resolve(strict=False))
    cmvn_config = dict(config_payload.get("model", {}).get("cmvn", {}))
    raw_path = str(cmvn_config.get("cmvn_file") or "").strip()
    if not raw_path:
        return None
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = config_path.parent / candidate
    resolved = candidate.resolve(strict=False)
    if not resolved.is_file():
        raise FileNotFoundError(f"cmvn_file does not exist: {resolved}")
    return str(resolved)


def _resolve_wekws_cmvn_mode(
    *,
    metadata: object,
    configured_mode: str,
    config_path: Path,
    explicit_cmvn_path: str | None,
    config_payload: dict[str, Any],
) -> str:
    normalized_configured_mode = _normalize_wekws_cmvn_mode(configured_mode)
    if normalized_configured_mode != "auto":
        return normalized_configured_mode
    metadata_mode = _optional_metadata_value(metadata, "cmvn_mode")
    if metadata_mode is not None:
        return _normalize_wekws_cmvn_mode(metadata_mode, default="embedded")
    # Twinr's WeKws export path embeds GlobalCMVN into the ONNX graph. Older
    # bundles may still lack explicit metadata, so treat any readable CMVN
    # sidecar as an embedded-model signal unless the caller forces otherwise.
    resolved_cmvn_path = _resolve_cmvn_path(
        config_path=config_path,
        explicit_cmvn_path=explicit_cmvn_path,
        config_payload=config_payload,
    )
    return "embedded" if resolved_cmvn_path is not None else "none"


def _load_wekws_model_config(
    *,
    config_path: Path,
    words_path: Path,
    cmvn_path: str | None,
    cmvn_mode: str = "auto",
    config_payload: dict[str, Any] | None = None,
) -> WakewordWekwsModelConfig:
    """Load one WeKws runtime config from exported training artifacts."""

    payload = dict(config_payload) if config_payload is not None else yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    dataset_conf = dict(payload.get("dataset_conf", {}))
    if dataset_conf.get("feats_type", "fbank") != "fbank":
        raise ValueError("WeKws runtime currently supports only fbank features.")
    fbank_conf = dict(dataset_conf.get("fbank_conf", {}))
    sample_rate = int(dict(dataset_conf.get("resample_conf", {})).get("resample_rate", 16000))
    feature_dim = int(fbank_conf.get("num_mel_bins") or dict(payload.get("model", {})).get("input_dim") or 80)
    frame_length_ms = float(fbank_conf.get("frame_length", _DEFAULT_FRAME_LENGTH_MS))
    frame_shift_ms = float(fbank_conf.get("frame_shift", _DEFAULT_FRAME_SHIFT_MS))
    keyword_labels = _load_keyword_labels(words_path)
    normalized_cmvn_mode = _normalize_wekws_cmvn_mode(cmvn_mode, default="none")
    if normalized_cmvn_mode == "auto":
        raise ValueError("WeKws cmvn mode must be resolved before loading model config.")
    if normalized_cmvn_mode in {"embedded", "none"}:
        return WakewordWekwsModelConfig(
            sample_rate=sample_rate,
            feature_dim=feature_dim,
            frame_length_ms=frame_length_ms,
            frame_shift_ms=frame_shift_ms,
            keyword_labels=keyword_labels,
            cmvn_mode=normalized_cmvn_mode,
        )
    resolved_cmvn_path = _resolve_cmvn_path(
        config_path=config_path,
        explicit_cmvn_path=cmvn_path,
        config_payload=payload,
    )
    if resolved_cmvn_path is None:
        raise ValueError("WeKws external CMVN mode requires a readable CMVN stats file.")
    mean, inverse_stddevs = _load_cmvn(Path(resolved_cmvn_path))
    if len(mean) != feature_dim or len(inverse_stddevs) != feature_dim:
        raise ValueError(
            f"CMVN dimensions {len(mean)} do not match configured feature_dim {feature_dim}."
        )
    return WakewordWekwsModelConfig(
        sample_rate=sample_rate,
        feature_dim=feature_dim,
        frame_length_ms=frame_length_ms,
        frame_shift_ms=frame_shift_ms,
        keyword_labels=keyword_labels,
        cmvn_mode=normalized_cmvn_mode,
        cmvn_mean=mean,
        cmvn_istd=inverse_stddevs,
    )


def _default_feature_extractor(
    samples: np.ndarray,
    *,
    source_sample_rate: int,
    target_sample_rate: int,
    feature_dim: int,
    frame_length_ms: float,
    frame_shift_ms: float,
) -> np.ndarray:
    """Extract Kaldi-style fbank frames for WeKws runtime inference."""

    if samples.size == 0:
        return np.zeros((0, feature_dim), dtype=np.float32)
    try:
        from scipy import signal
    except ImportError as exc:  # pragma: no cover - depends on runtime env.
        raise RuntimeError("scipy is required for the current WeKws feature extractor.") from exc

    waveform = samples.astype(np.float32, copy=False)
    sample_rate = int(source_sample_rate)
    target_rate = int(target_sample_rate)
    if sample_rate != target_rate:
        factor = math.gcd(sample_rate, target_rate)
        waveform = signal.resample_poly(waveform, target_rate // factor, sample_rate // factor).astype(
            np.float32,
            copy=False,
        )
        sample_rate = target_rate
    minimum_samples = max(1, int(round(sample_rate * (float(frame_length_ms) / 1000.0))))
    if int(waveform.shape[0]) < minimum_samples:
        return np.zeros((0, feature_dim), dtype=np.float32)
    waveform = _pre_emphasize(waveform)
    frame_length_samples = max(1, int(round(sample_rate * (float(frame_length_ms) / 1000.0))))
    frame_shift_samples = max(1, int(round(sample_rate * (float(frame_shift_ms) / 1000.0))))
    frames = _frame_waveform(
        waveform,
        frame_length_samples=frame_length_samples,
        frame_shift_samples=frame_shift_samples,
    )
    if frames.size == 0:
        return np.zeros((0, feature_dim), dtype=np.float32)
    window = _povey_window(frame_length_samples)
    windowed = frames * window
    padded_fft_size = 1 << max(1, frame_length_samples - 1).bit_length()
    power = np.abs(np.fft.rfft(windowed, n=padded_fft_size, axis=1)) ** 2
    mel_filters = _mel_filterbank(
        sample_rate=sample_rate,
        fft_size=padded_fft_size,
        feature_dim=int(feature_dim),
    )
    mel_energies = power @ mel_filters.T
    mel_energies = np.maximum(mel_energies, 1.0e-10)
    return np.log(mel_energies).astype(np.float32, copy=False)


def _ensure_default_feature_stack_available() -> None:
    """Fail closed when the default WeKws frontend dependencies are absent."""

    try:
        import scipy  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on runtime env.
        raise RuntimeError(
            "WeKws currently requires scipy for feature extraction."
        ) from exc


def _pre_emphasize(samples: np.ndarray, *, coefficient: float = 0.97) -> np.ndarray:
    """Apply Kaldi-style pre-emphasis before windowing."""

    if samples.size <= 1:
        return samples.astype(np.float32, copy=True)
    emphasized = np.empty_like(samples, dtype=np.float32)
    emphasized[0] = samples[0]
    emphasized[1:] = samples[1:] - (float(coefficient) * samples[:-1])
    return emphasized


def _frame_waveform(
    samples: np.ndarray,
    *,
    frame_length_samples: int,
    frame_shift_samples: int,
) -> np.ndarray:
    """Slice one waveform into overlapping frames with centered padding."""

    if samples.size < frame_length_samples:
        return np.zeros((0, frame_length_samples), dtype=np.float32)
    pad = frame_length_samples // 2
    padded = np.pad(samples, (pad, pad), mode="reflect")
    frame_count = 1 + max(0, (padded.size - frame_length_samples) // frame_shift_samples)
    if frame_count <= 0:
        return np.zeros((0, frame_length_samples), dtype=np.float32)
    shape = (frame_count, frame_length_samples)
    strides = (padded.strides[0] * frame_shift_samples, padded.strides[0])
    frames = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    return np.array(frames, dtype=np.float32, copy=True)


def _povey_window(frame_length_samples: int) -> np.ndarray:
    """Approximate Kaldi's Povey window used by the WeKws training frontend."""

    if frame_length_samples <= 1:
        return np.ones((max(1, frame_length_samples),), dtype=np.float32)
    base = 0.5 - (0.5 * np.cos((2.0 * np.pi * np.arange(frame_length_samples)) / (frame_length_samples - 1)))
    return np.power(base, 0.85, dtype=np.float32).astype(np.float32, copy=False)


def _hz_to_mel(frequency_hz: np.ndarray) -> np.ndarray:
    return 1127.0 * np.log1p(frequency_hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * np.expm1(mel / 1127.0)


@lru_cache(maxsize=16)
def _mel_filterbank(
    *,
    sample_rate: int,
    fft_size: int,
    feature_dim: int,
    low_freq_hz: float = _DEFAULT_LOW_FREQ_HZ,
    high_freq_offset_hz: float = _DEFAULT_HIGH_FREQ_OFFSET_HZ,
) -> np.ndarray:
    """Build one cached triangular mel filterbank matrix."""

    nyquist = float(sample_rate) / 2.0
    high_freq_hz = nyquist + float(high_freq_offset_hz)
    if high_freq_hz <= low_freq_hz:
        high_freq_hz = nyquist
    mel_points = np.linspace(
        _hz_to_mel(np.asarray([low_freq_hz], dtype=np.float64))[0],
        _hz_to_mel(np.asarray([high_freq_hz], dtype=np.float64))[0],
        int(feature_dim) + 2,
    )
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor(((fft_size + 1) * hz_points) / float(sample_rate)).astype(int)
    max_bin = (fft_size // 2) + 1
    filters = np.zeros((int(feature_dim), max_bin), dtype=np.float32)
    for index in range(int(feature_dim)):
        left = max(0, min(int(bins[index]), max_bin - 1))
        center = max(left + 1, min(int(bins[index + 1]), max_bin - 1))
        right = max(center + 1, min(int(bins[index + 2]), max_bin))
        for bin_index in range(left, center):
            filters[index, bin_index] = (bin_index - left) / max(center - left, 1)
        for bin_index in range(center, right):
            filters[index, bin_index] = (right - bin_index) / max(right - center, 1)
    return filters


def _metadata_value(metadata: object, key: str) -> str:
    if hasattr(metadata, "custom_metadata_map"):
        mapping = getattr(metadata, "custom_metadata_map") or {}
        if key in mapping:
            return str(mapping[key])
    lookup = getattr(metadata, "LookupCustomMetadataMap", None)
    if callable(lookup):  # pragma: no cover - compatibility with C++-style fakes.
        return str(lookup(key))
    raise KeyError(key)


def _optional_metadata_value(metadata: object, key: str) -> str | None:
    try:
        value = _metadata_value(metadata, key)
    except KeyError:
        return None
    normalized = str(value).strip()
    return normalized or None


class WakewordWekwsFrameSpotter:
    """Detect wakewords from live PCM bytes through a WeKws ONNX model."""

    def __init__(
        self,
        *,
        model_path: str,
        config_path: str,
        words_path: str,
        phrases: tuple[str, ...] | list[str],
        project_root: str | Path | None = None,
        cmvn_path: str | None = None,
        cmvn_mode: str = "auto",
        threshold: float = 0.5,
        chunk_ms: int = 100,
        num_threads: int = 2,
        provider: str = "cpu",
        session_factory: Callable[..., Any] | None = None,
        feature_extractor: Callable[..., np.ndarray] | None = None,
    ) -> None:
        self.assets = WakewordWekwsAssetBundle(
            model_path=_resolve_existing_file(model_path, name="model_path", project_root=project_root),
            config_path=_resolve_existing_file(config_path, name="config_path", project_root=project_root),
            words_path=_resolve_existing_file(words_path, name="words_path", project_root=project_root),
            cmvn_path=(
                None
                if not str(cmvn_path or "").strip()
                else _resolve_existing_file(cmvn_path, name="cmvn_path", project_root=project_root)
            ),
        )
        self.phrases = tuple(str(item).strip() for item in phrases if str(item).strip())
        if not self.phrases:
            raise ValueError("phrases must contain at least one wakeword phrase.")
        self.threshold = _coerce_nonnegative_float("threshold", threshold)
        self.chunk_ms = _coerce_positive_int("chunk_ms", chunk_ms)
        self.num_threads = _coerce_positive_int("num_threads", num_threads)
        self.provider = _normalize_nonempty_text("provider", provider).lower()
        self._feature_extractor = feature_extractor or _default_feature_extractor
        if feature_extractor is None:
            _ensure_default_feature_stack_available()
        factory = session_factory or _default_onnx_session_factory
        try:
            self._session = factory(
                model_path=self.assets.model_path,
                provider=self.provider,
                num_threads=self.num_threads,
            )
        except Exception as exc:  # pragma: no cover - depends on runtime env.
            raise RuntimeError("Failed to initialize WeKws ONNX runtime session.") from exc
        try:
            metadata = self._session.get_modelmeta()
            self._cache_dim = _coerce_positive_int("cache_dim", _metadata_value(metadata, "cache_dim"))
            self._cache_len = _coerce_positive_int("cache_len", _metadata_value(metadata, "cache_len"))
        except Exception as exc:  # pragma: no cover - depends on external runtime/model.
            raise RuntimeError("WeKws ONNX model metadata is missing cache_dim/cache_len.") from exc
        config_path_obj = Path(self.assets.config_path)
        config_payload = yaml.safe_load(config_path_obj.read_text(encoding="utf-8")) or {}
        effective_cmvn_mode = _resolve_wekws_cmvn_mode(
            metadata=metadata,
            configured_mode=cmvn_mode,
            config_path=config_path_obj,
            explicit_cmvn_path=self.assets.cmvn_path,
            config_payload=config_payload,
        )
        self.model_config = _load_wekws_model_config(
            config_path=config_path_obj,
            words_path=Path(self.assets.words_path),
            cmvn_path=self.assets.cmvn_path,
            cmvn_mode=effective_cmvn_mode,
            config_payload=config_payload,
        )
        self._frame_shift_samples = max(
            1,
            int(round(self.model_config.sample_rate * (self.model_config.frame_shift_ms / 1000.0))),
        )
        self._chunk_samples = max(
            1,
            int(round(self.model_config.sample_rate * (self.chunk_ms / 1000.0))),
        )
        self._lock = RLock()
        self.reset()

    @property
    def frame_bytes(self) -> int:
        return self.frame_bytes_for_channels(1)

    def frame_bytes_for_channels(self, channels: int) -> int:
        validated_channels = _coerce_positive_int("channels", channels)
        return self._chunk_samples * validated_channels * 2

    def reset(self) -> None:
        with self._lock:
            self._remained_samples = np.zeros(0, dtype=np.int16)
            self._cache = np.zeros((1, self._cache_dim, self._cache_len), dtype=np.float32)

    def process_pcm_bytes(self, pcm_bytes: bytes, *, channels: int = 1) -> WakewordMatch | None:
        if not pcm_bytes:
            return None
        validated_channels = _coerce_positive_int("channels", channels)
        samples = _pcm16_to_int16_mono_samples(pcm_bytes, channels=validated_channels)
        if samples.size == 0:
            return None
        with self._lock:
            try:
                features = self._extract_incremental_features(samples)
                if features.size == 0:
                    return None
                detected_match = self._match_from_scores(self._run_model(features))
                if detected_match is not None:
                    self.reset()
                return detected_match
            except Exception:  # pragma: no cover - depends on runtime env/model.
                LOGGER.exception("WeKws frame spotting failed.")
                self.reset()
                return None

    def detect(self, capture: AmbientAudioCaptureWindow) -> WakewordMatch:
        try:
            best_score, detector_label = self.score_capture(capture)
        except Exception:  # pragma: no cover - depends on runtime env/model.
            LOGGER.exception("WeKws clip spotting failed.")
            return WakewordMatch(detected=False, transcript="", backend="wekws")
        if detector_label is None or best_score < self.threshold:
            return WakewordMatch(detected=False, transcript="", backend="wekws")
        matched_phrase = phrase_from_detector_label(detector_label, phrases=self.phrases)
        return WakewordMatch(
            detected=True,
            transcript="",
            matched_phrase=matched_phrase,
            remaining_text="",
            normalized_transcript="",
            backend="wekws",
            detector_label=detector_label,
            score=best_score,
        )

    def score_capture(self, capture: AmbientAudioCaptureWindow) -> tuple[float, str | None]:
        pcm_bytes = capture.pcm_bytes or b""
        if not pcm_bytes:
            return 0.0, None
        try:
            channels = _coerce_positive_int("channels", capture.channels)
            source_sample_rate = _coerce_positive_int("sample_rate", capture.sample_rate)
        except ValueError:
            LOGGER.warning(
                "Ignoring WeKws capture with invalid audio metadata: sample_rate=%r channels=%r",
                capture.sample_rate,
                capture.channels,
            )
            return 0.0, None
        samples = _pcm16_to_int16_mono_samples(pcm_bytes, channels=channels)
        if samples.size == 0:
            return 0.0, None
        with self._lock:
            features = self._extract_feature_matrix(samples, source_sample_rate=source_sample_rate)
            if features.size == 0:
                return 0.0, None
            scores = self._run_model(features, cache=np.zeros_like(self._cache))
        return self._best_score_from_frames(scores)

    def _extract_incremental_features(self, samples: np.ndarray) -> np.ndarray:
        waveform = np.concatenate((self._remained_samples, samples))
        features = self._extract_feature_matrix(
            waveform,
            source_sample_rate=self.model_config.sample_rate,
        )
        if features.shape[0] <= 0:
            self._remained_samples = waveform
            return np.zeros((0, self.model_config.feature_dim), dtype=np.float32)
        consumed = self._frame_shift_samples * int(features.shape[0])
        self._remained_samples = waveform[consumed:].copy()
        return features

    def _extract_feature_matrix(self, samples: np.ndarray, *, source_sample_rate: int) -> np.ndarray:
        features = self._feature_extractor(
            samples,
            source_sample_rate=int(source_sample_rate),
            target_sample_rate=self.model_config.sample_rate,
            feature_dim=self.model_config.feature_dim,
            frame_length_ms=self.model_config.frame_length_ms,
            frame_shift_ms=self.model_config.frame_shift_ms,
        )
        if features.size == 0:
            return np.zeros((0, self.model_config.feature_dim), dtype=np.float32)
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != self.model_config.feature_dim:
            raise ValueError(
                "WeKws feature extractor must return a [frames, feature_dim] float32 array."
            )
        if self.model_config.cmvn_mean is not None:
            mean = np.asarray(self.model_config.cmvn_mean, dtype=np.float32)
            features = features - mean
            if self.model_config.cmvn_istd is not None:
                features = features * np.asarray(self.model_config.cmvn_istd, dtype=np.float32)
        return features

    def _run_model(self, features: np.ndarray, *, cache: np.ndarray | None = None) -> np.ndarray:
        feeds = {
            "input": np.expand_dims(features.astype(np.float32, copy=False), axis=0),
            "cache": (self._cache if cache is None else cache).astype(np.float32, copy=False),
        }
        outputs = self._session.run(None, feeds)
        if len(outputs) != 2:
            raise ValueError("WeKws ONNX session must return [output, r_cache].")
        score_frames = np.asarray(outputs[0], dtype=np.float32)
        if score_frames.ndim != 3 or score_frames.shape[0] != 1:
            raise ValueError("WeKws ONNX output must have shape [1, frames, keywords].")
        next_cache = np.asarray(outputs[1], dtype=np.float32)
        if cache is None:
            self._cache = next_cache
        return score_frames[0]

    def _best_score_from_frames(self, score_frames: np.ndarray) -> tuple[float, str | None]:
        if score_frames.size == 0:
            return 0.0, None
        if score_frames.ndim != 2:
            raise ValueError("WeKws score frames must have shape [frames, keywords].")
        keyword_count = score_frames.shape[1]
        if keyword_count != len(self.model_config.keyword_labels):
            raise ValueError(
                "WeKws score output width does not match keyword_labels."
            )
        best_index = np.unravel_index(int(np.argmax(score_frames)), score_frames.shape)
        best_score = float(score_frames[best_index])
        detector_label = self.model_config.keyword_labels[int(best_index[1])]
        return best_score, detector_label

    def _match_from_scores(self, score_frames: np.ndarray) -> WakewordMatch | None:
        best_score, detector_label = self._best_score_from_frames(score_frames)
        if detector_label is None:
            return None
        if best_score < self.threshold:
            return None
        matched_phrase = phrase_from_detector_label(detector_label, phrases=self.phrases)
        return WakewordMatch(
            detected=True,
            transcript="",
            matched_phrase=matched_phrase,
            remaining_text="",
            normalized_transcript="",
            backend="wekws",
            detector_label=detector_label,
            score=best_score,
        )


class WakewordWekwsSpotter(WakewordWekwsFrameSpotter):
    """Detect wakewords from one buffered capture via a WeKws ONNX model."""


__all__ = [
    "WakewordWekwsAssetBundle",
    "WakewordWekwsFrameSpotter",
    "WakewordWekwsModelConfig",
    "WakewordWekwsSpotter",
]
