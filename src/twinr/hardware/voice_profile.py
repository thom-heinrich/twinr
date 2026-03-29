# CHANGELOG: 2026-03-28
# BUG-1: Fixed sample-rate-dependent embeddings by canonicalizing all audio to mono 16 kHz
# BUG-1: before feature/model extraction.
# BUG-2: Moved the runtime hot path off the old high-overhead handcrafted frame pipeline and
# BUG-2: onto a bounded VAD-first model path that behaves better on Raspberry Pi 4.
# BUG-3: Fixed silent profile poisoning: new enrollment samples must now match the enrolled
# BUG-3: speaker instead of always being averaged into the template.
# SEC-1: Added guarded enrollment, template/backend compatibility checks, and upload-size/
# SEC-1: quality bounds to fail closed on malformed or unsafe inputs.
# SEC-2: Added conservative spoof-suspicion downgrades so obviously suspicious audio is not
# SEC-2: surfaced as "Likely user".
# IMP-1: Added a 2026-style ONNX speaker backend with canonical preprocessing, cosine scoring,
# IMP-1: and bounded ONNX Runtime session settings for edge CPU deployment.
# IMP-2: Preserved a bounded legacy fallback so the module remains deployable before the model
# IMP-2: artifact is rolled out everywhere.
# BREAKING: The preferred backend is now embedding_version=2 (real speaker embeddings). Legacy
# BREAKING: v1 templates remain readable but require re-enrollment when the active backend
# BREAKING: changes.
"""Persist and assess Twinr's local on-device voice profiles.

This module owns lightweight speaker embeddings, the bounded file-backed store,
and the user-facing enrollment and verification helpers used by runtime and web
flows.

2026 edge/frontier path
-----------------------
The preferred backend is now a real speaker-embedding model (e.g. a local
WeSpeaker/ECAPA-class ONNX model) with canonical 16 kHz preprocessing, VAD-
first speech selection, cosine scoring, and bounded ONNX Runtime settings that
behave well on Raspberry Pi 4. A bounded legacy DSP fallback remains available
for deployments that have not rolled out the model file yet.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from array import array
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import functools
import fcntl
import io
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile
import threading
from typing import Callable, Any
import wave
from twinr.agent.base_agent.config import TwinrConfig
try:
    import numpy as np
except Exception:
    np = None
try:
    import onnxruntime as ort
except Exception:
    ort = None
try:
    import torch
    import torchaudio.compliance.kaldi as ta_kaldi
except Exception:
    torch = None
    ta_kaldi = None
try:
    from silero_vad import get_speech_timestamps, load_silero_vad
except Exception:
    get_speech_timestamps = None
    load_silero_vad = None
try:
    from scipy.signal import resample_poly
except Exception:
    resample_poly = None
_FRAME_MS = 30
_HOP_MS = 15
_MIN_PITCH_HZ = 85.0
_MAX_PITCH_HZ = 350.0
_BAND_FREQUENCIES_HZ = (250.0, 500.0, 1000.0, 1800.0, 2800.0)
_EMBEDDING_VERSION = 2
# BREAKING: v2 is the preferred real-speaker embedding format; v1 is legacy-only.
_LEGACY_EMBEDDING_VERSION = 1
_DEFAULT_MAX_SAMPLE_MS = 30000
_MAX_ALLOWED_SAMPLE_MS = 60000
_MAX_STORE_BYTES = 64 * 1024
_MAX_AUDIO_UPLOAD_BYTES = 12 * 1024 * 1024
_DEFAULT_MODEL_SAMPLE_RATE = 16000
_DEFAULT_MAX_RAW_DURATION_MULTIPLIER = 3
_DEFAULT_SPOOF_GUARD_THRESHOLD = 0.9
_DEFAULT_MIN_ENROLL_QUALITY = 0.56
_DEFAULT_MIN_ASSESS_QUALITY = 0.38
_DEFAULT_ENROLL_MATCH_THRESHOLD = 0.7
_DEFAULT_TEMPLATE_BACKEND = 'onnx_speaker'
_LEGACY_TEMPLATE_BACKEND = 'legacy_dsp'
_PROFILE_WEIGHTS = (0.3, 0.45, 0.9, 0.5, 0.75, 0.4, 0.75, 1.0, 0.55, 1.0, 1.0, 1.0, 1.0, 1.0)

def _utcnow() -> datetime:
    return datetime.now(UTC)

def _normalize_datetime_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)

def _utc_iso(value: datetime | None=None) -> str:
    effective = _utcnow() if value is None else value
    return _normalize_datetime_utc(effective).isoformat()

def _is_finite_vector(values: tuple[float, ...]) -> bool:
    return all((math.isfinite(value) for value in values))

def _coerce_int_like(value: object, *, minimum: int, default: int | None) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    result: int | None = None
    if isinstance(value, int):
        result = value
    elif isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            result = int(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped:
            try:
                result = int(stripped, 10)
            except ValueError:
                result = None
    if result is None:
        return default
    return max(minimum, result)

def _coerce_bool_like(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and (not isinstance(value, bool)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {'1', 'true', 'yes', 'on'}:
            return True
        if normalized in {'0', 'false', 'no', 'off', ''}:
            return False
    return default

def _coerce_bounded_float(value: object, *, default: float, minimum: float, maximum: float) -> float:
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        candidate = default
    if not math.isfinite(candidate):
        candidate = default
    return min(maximum, max(minimum, candidate))

def _normalize_thresholds(likely_threshold: object, uncertain_threshold: object) -> tuple[float, float]:
    likely = _coerce_bounded_float(likely_threshold, default=0.82, minimum=0.0, maximum=1.0)
    uncertain = _coerce_bounded_float(uncertain_threshold, default=0.58, minimum=0.0, maximum=1.0)
    if likely < uncertain:
        likely, uncertain = (uncertain, likely)
    return (likely, uncertain)

def _coerce_backend_name(value: object, *, default: str) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower().replace('-', '_')
        if normalized:
            return normalized
    return default

def _env_or_config(config: TwinrConfig | None, attr_name: str, env_name: str, default: object=None) -> object:
    if config is not None and hasattr(config, attr_name):
        return getattr(config, attr_name)
    return os.getenv(env_name, default)

def voice_profile_store_path(config: TwinrConfig) -> Path:
    project_root = Path(config.project_root).expanduser().resolve(strict=False)
    configured_raw = str(config.voice_profile_store_path or '').strip()
    if not configured_raw:
        raise ValueError('voice_profile_store_path must point to a JSON file.')
    configured = Path(configured_raw).expanduser()
    allow_external_store = _coerce_bool_like(getattr(config, 'voice_profile_allow_external_store', False), default=False)
    try:
        candidate = configured.resolve(strict=False) if configured.is_absolute() else (project_root / configured).resolve(strict=False)
    except OSError as exc:
        raise ValueError('voice_profile_store_path could not be resolved safely.') from exc
    if candidate.name in {'', '.', '..'}:
        raise ValueError('voice_profile_store_path must point to a file, not a directory.')
    if not allow_external_store and (not candidate.is_relative_to(project_root)):
        raise ValueError('voice_profile_store_path must stay inside project_root unless voice_profile_allow_external_store=true.')
    return candidate

@dataclass(frozen=True, slots=True)
class VoiceProfileTemplate:
    embedding: tuple[float, ...]
    sample_count: int
    updated_at: str
    embedding_version: int = _EMBEDDING_VERSION
    average_duration_ms: int = 0
    backend: str = _DEFAULT_TEMPLATE_BACKEND
    average_speech_duration_ms: int = 0
    quality_score: float = 0.0

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload['embedding'] = list(self.embedding)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> 'VoiceProfileTemplate | None':
        embedding = payload.get('embedding', ())
        if not isinstance(embedding, list) or not embedding:
            return None
        try:
            vector = tuple((float(value) for value in embedding))
        except (TypeError, ValueError):
            return None
        if not _is_finite_vector(vector):
            return None
        embedding_version = _coerce_int_like(payload.get('embedding_version', _LEGACY_EMBEDDING_VERSION), minimum=1, default=None)
        if embedding_version not in {_LEGACY_EMBEDDING_VERSION, _EMBEDDING_VERSION}:
            return None
        if embedding_version == _LEGACY_EMBEDDING_VERSION and len(vector) != len(_PROFILE_WEIGHTS):
            return None
        if embedding_version == _EMBEDDING_VERSION and (not vector):
            return None
        sample_count = _coerce_int_like(payload.get('sample_count', 1), minimum=1, default=None)
        average_duration_ms = _coerce_int_like(payload.get('average_duration_ms', 0), minimum=0, default=None)
        average_speech_duration_ms = _coerce_int_like(payload.get('average_speech_duration_ms', 0), minimum=0, default=0)
        if sample_count is None or average_duration_ms is None:
            return None
        backend_raw = payload.get('backend')
        backend = backend_raw.strip() if isinstance(backend_raw, str) and backend_raw.strip() else _LEGACY_TEMPLATE_BACKEND if embedding_version == _LEGACY_EMBEDDING_VERSION else _DEFAULT_TEMPLATE_BACKEND
        quality_score = _coerce_bounded_float(payload.get('quality_score', 0.0), default=0.0, minimum=0.0, maximum=1.0)
        updated_at_raw = payload.get('updated_at')
        updated_at = updated_at_raw.strip() if isinstance(updated_at_raw, str) and updated_at_raw.strip() else _utc_iso()
        return cls(embedding=vector, sample_count=sample_count, updated_at=updated_at, embedding_version=embedding_version, average_duration_ms=min(average_duration_ms, _MAX_ALLOWED_SAMPLE_MS), backend=backend, average_speech_duration_ms=min(average_speech_duration_ms, _MAX_ALLOWED_SAMPLE_MS), quality_score=quality_score)

@dataclass(frozen=True, slots=True)
class VoiceProfileSummary:
    enrolled: bool
    sample_count: int = 0
    updated_at: str | None = None
    average_duration_ms: int = 0
    store_path: str | None = None
    backend: str | None = None
    average_speech_duration_ms: int = 0
    quality_score: float = 0.0

@dataclass(frozen=True, slots=True)
class VoiceAssessment:
    status: str
    label: str
    detail: str
    confidence: float | None = None
    checked_at: str | None = None

    @property
    def should_persist(self) -> bool:
        return self.status not in {'disabled', 'not_enrolled', 'invalid_sample', 'spoof_suspected'}

    def confidence_percent(self) -> str:
        if self.confidence is None:
            return '—'
        return f'{self.confidence * 100:.0f}%'

@dataclass(frozen=True, slots=True)
class _VoiceEmbedding:
    vector: tuple[float, ...]
    duration_ms: int
    speech_duration_ms: int
    quality: float
    backend: str
    embedding_version: int
    spoof_risk: float = 0.0

@dataclass(frozen=True, slots=True)
class _PreparedAudio:
    samples: Any
    sample_rate: int
    duration_ms: int
    speech_samples: Any
    speech_duration_ms: int
    speech_ratio: float
    quality: float
    clipping_ratio: float
    spectral_rolloff_hz: float
    spectral_flatness: float
    snr_db: float

class VoiceProfileStore:

    def __init__(self, path: str | Path) -> None:
        self.path = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
        self._process_lock = threading.RLock()

    @classmethod
    def from_config(cls, config: TwinrConfig) -> 'VoiceProfileStore':
        return cls(voice_profile_store_path(config))

    def load(self) -> VoiceProfileTemplate | None:
        parent = self.path.parent
        if not parent.exists():
            return None
        try:
            with self._exclusive_lock(create_parent=False):
                return self._load_unlocked()
        except OSError:
            return None

    def save(self, template: VoiceProfileTemplate) -> VoiceProfileTemplate:
        self._validate_template(template)
        with self._exclusive_lock(create_parent=True):
            self._write_unlocked(template)
        return template

    def clear(self) -> None:
        parent = self.path.parent
        if not parent.exists():
            return
        try:
            with self._exclusive_lock(create_parent=False):
                self._clear_unlocked()
        except OSError:
            return

    def update(self, updater: Callable[[VoiceProfileTemplate | None], VoiceProfileTemplate]) -> VoiceProfileTemplate:
        with self._exclusive_lock(create_parent=True):
            current = self._load_unlocked()
            updated = updater(current)
            self._validate_template(updated)
            self._write_unlocked(updated)
            return updated

    @contextmanager
    def _exclusive_lock(self, *, create_parent: bool) -> Iterator[None]:
        with self._process_lock:
            self._ensure_safe_parent_directory(create=create_parent)
            lock_fd = self._open_lock_file()
            try:
                with os.fdopen(lock_fd, 'a+', encoding='utf-8') as lock_file:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                    try:
                        yield
                    finally:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except Exception:
                try:
                    os.close(lock_fd)
                except OSError:
                    pass
                raise

    def _ensure_safe_parent_directory(self, *, create: bool) -> None:
        if not self.path.is_absolute():
            raise OSError('Voice profile store path must be absolute.')
        current = Path(self.path.anchor)
        for part in self.path.parts[1:-1]:
            current = current / part
            if current.exists():
                metadata = os.lstat(current)
                if stat.S_ISLNK(metadata.st_mode):
                    raise OSError(f'Unsafe symlinked parent directory for voice profile store: {current}')
                if not stat.S_ISDIR(metadata.st_mode):
                    raise OSError(f'Voice profile store parent is not a directory: {current}')
                continue
            if not create:
                raise FileNotFoundError(current)
            try:
                current.mkdir(mode=448)
            except FileExistsError:
                pass
            metadata = os.lstat(current)
            if stat.S_ISLNK(metadata.st_mode) or (not stat.S_ISDIR(metadata.st_mode)):
                raise OSError(f'Voice profile store parent became unsafe during creation: {current}')

    def _open_lock_file(self) -> int:
        flags = os.O_RDWR | os.O_CREAT | getattr(os, 'O_CLOEXEC', 0) | getattr(os, 'O_NOFOLLOW', 0)
        return os.open(self._lock_path(), flags, 384)

    def _lock_path(self) -> str:
        return str(self.path.with_name(f'.{self.path.name}.lock'))

    def _load_unlocked(self) -> VoiceProfileTemplate | None:
        try:
            payload_text = self._read_text_unlocked()
        except FileNotFoundError:
            return None
        except OSError:
            return None
        if not payload_text:
            return None
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        return VoiceProfileTemplate.from_dict(payload)

    def _read_text_unlocked(self) -> str:
        flags = os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0) | getattr(os, 'O_NOFOLLOW', 0)
        fd = os.open(self.path, flags)
        try:
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode):
                raise OSError('Voice profile store must be a regular file.')
            if metadata.st_size > _MAX_STORE_BYTES:
                raise OSError('Voice profile store is unexpectedly large.')
            with os.fdopen(fd, 'r', encoding='utf-8') as store_file:
                return store_file.read(_MAX_STORE_BYTES + 1)
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            raise

    def _write_unlocked(self, template: VoiceProfileTemplate) -> None:
        payload_text = json.dumps(template.to_dict(), indent=2, sort_keys=True, allow_nan=False) + '\n'
        payload_bytes = payload_text.encode('utf-8')
        if len(payload_bytes) > _MAX_STORE_BYTES:
            raise ValueError('Voice profile store payload is unexpectedly large.')
        temp_fd, temp_path = tempfile.mkstemp(prefix=f'.{self.path.name}.', suffix='.tmp', dir=str(self.path.parent))
        try:
            os.fchmod(temp_fd, 384)
            with os.fdopen(temp_fd, 'wb') as temp_file:
                temp_file.write(payload_bytes)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            try:
                existing_metadata = os.lstat(self.path)
            except FileNotFoundError:
                existing_metadata = None
            if existing_metadata is not None and stat.S_ISDIR(existing_metadata.st_mode):
                raise OSError(f'Voice profile store target is a directory: {self.path}')
            os.replace(temp_path, self.path)
            self._fsync_directory(self.path.parent)
        finally:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass

    def _clear_unlocked(self) -> None:
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            return
        self._fsync_directory(self.path.parent)

    def _fsync_directory(self, directory: Path) -> None:
        flags = os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0)
        if hasattr(os, 'O_DIRECTORY'):
            flags |= os.O_DIRECTORY
        directory_fd = os.open(directory, flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def _validate_template(self, template: VoiceProfileTemplate) -> None:
        if not template.embedding:
            raise ValueError('Voice profile embedding must not be empty.')
        if not _is_finite_vector(template.embedding):
            raise ValueError('Voice profile embedding contains non-finite values.')
        if template.embedding_version == _LEGACY_EMBEDDING_VERSION:
            if len(template.embedding) != len(_PROFILE_WEIGHTS):
                raise ValueError('Legacy voice profile embedding has the wrong length.')
        elif template.embedding_version != _EMBEDDING_VERSION:
            raise ValueError('Voice profile embedding version is not supported.')
        if template.sample_count < 1:
            raise ValueError('Voice profile sample_count must be at least 1.')
        if template.average_duration_ms < 0 or template.average_duration_ms > _MAX_ALLOWED_SAMPLE_MS:
            raise ValueError('Voice profile average_duration_ms is outside the supported range.')
        if template.average_speech_duration_ms < 0 or template.average_speech_duration_ms > _MAX_ALLOWED_SAMPLE_MS:
            raise ValueError('Voice profile average_speech_duration_ms is outside the supported range.')
        if not math.isfinite(template.quality_score) or (not 0.0 <= template.quality_score <= 1.0):
            raise ValueError('Voice profile quality_score must be within [0, 1].')

class _VoiceEmbeddingBackend(ABC):
    name: str
    embedding_version: int
    target_sample_rate: int

    @abstractmethod
    def extract_from_wav_bytes(self, wav_bytes: bytes, *, min_sample_ms: int, max_sample_ms: int) -> _VoiceEmbedding:
        raise NotImplementedError

    @abstractmethod
    def extract_from_pcm16(self, pcm_bytes: bytes, *, sample_rate: int, channels: int, min_sample_ms: int, max_sample_ms: int) -> _VoiceEmbedding:
        raise NotImplementedError

    @abstractmethod
    def similarity(self, left: tuple[float, ...], right: tuple[float, ...]) -> float:
        raise NotImplementedError

    def is_template_compatible(self, template: VoiceProfileTemplate) -> bool:
        return template.embedding_version == self.embedding_version and template.backend == self.name and bool(template.embedding)

class LegacyDSPEmbeddingBackend(_VoiceEmbeddingBackend):
    name = _LEGACY_TEMPLATE_BACKEND
    embedding_version = _LEGACY_EMBEDDING_VERSION
    target_sample_rate = _DEFAULT_MODEL_SAMPLE_RATE

    def extract_from_wav_bytes(self, wav_bytes: bytes, *, min_sample_ms: int, max_sample_ms: int) -> _VoiceEmbedding:
        try:
            with wave.open(io.BytesIO(wav_bytes), 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frame_count = wav_file.getnframes()
                if sample_width != 2:
                    raise ValueError('Voice profiling currently expects 16-bit WAV input.')
                if sample_rate <= 0:
                    raise ValueError('Sample rate must be greater than zero.')
                if channels <= 0:
                    raise ValueError('Channel count must be greater than zero.')
                duration_ms = int(frame_count * 1000 / sample_rate)
                if duration_ms > max_sample_ms * _DEFAULT_MAX_RAW_DURATION_MULTIPLIER:
                    raise ValueError(f'Speech sample is too long for voice profiling. Please keep it under {max_sample_ms // 1000} seconds.')
                pcm_bytes = wav_file.readframes(frame_count)
        except (EOFError, wave.Error) as exc:
            raise ValueError('Audio input is not a valid WAV recording.') from exc
        return self.extract_from_pcm16(pcm_bytes, sample_rate=sample_rate, channels=channels, min_sample_ms=min_sample_ms, max_sample_ms=max_sample_ms)

    def extract_from_pcm16(self, pcm_bytes: bytes, *, sample_rate: int, channels: int, min_sample_ms: int, max_sample_ms: int) -> _VoiceEmbedding:
        sample_rate_value = _coerce_int_like(sample_rate, minimum=1, default=None)
        if sample_rate_value is None:
            raise ValueError('Sample rate must be greater than zero.')
        channels_value = _coerce_int_like(channels, minimum=1, default=None)
        if channels_value is None:
            raise ValueError('Channel count must be greater than zero.')
        max_pcm_bytes = int(math.ceil(sample_rate_value * channels_value * 2 * (max_sample_ms * 3 / 1000.0)))
        if len(pcm_bytes) > max_pcm_bytes:
            raise ValueError(f'Speech sample is too long for voice profiling. Please keep it under {max_sample_ms // 1000} seconds.')
        samples = _mono_samples(pcm_bytes, channels=channels_value)
        if sample_rate_value != self.target_sample_rate:
            samples = _resample_tuple(samples, src_rate=sample_rate_value, dst_rate=self.target_sample_rate)
            sample_rate_value = self.target_sample_rate
        duration_ms = int(len(samples) * 1000 / sample_rate_value)
        if duration_ms < min_sample_ms:
            raise ValueError('Need a longer speech sample for voice profiling.')
        if duration_ms > max_sample_ms:
            max_samples = int(sample_rate_value * max_sample_ms / 1000)
            samples = samples[:max_samples]
            duration_ms = int(len(samples) * 1000 / sample_rate_value)
        frame_size = max(32, int(sample_rate_value * (_FRAME_MS / 1000.0)))
        hop_size = max(16, int(sample_rate_value * (_HOP_MS / 1000.0)))
        frames = _frame_windows(samples, frame_size=frame_size, hop_size=hop_size)
        if not frames:
            raise ValueError('Speech sample is too short for voice profiling.')
        rms_values = tuple((_frame_rms(frame) for frame in frames))
        peak_rms = max(rms_values)
        active_threshold = max(0.015, peak_rms * 0.35)
        active_frames = [frame for frame, rms in zip(frames, rms_values, strict=True) if rms >= active_threshold]
        if len(active_frames) < 6:
            raise ValueError('Need more steady speech for voice profiling.')
        zcr_values = [_zero_crossing_rate(frame) for frame in active_frames]
        delta_values = [_average_absolute_delta(frame) for frame in active_frames]
        active_ratio = len(active_frames) / max(1, len(frames))
        mean_rms = sum((_frame_rms(frame) for frame in active_frames)) / max(1, len(active_frames))
        pitch_values = [pitch for pitch in (_estimate_pitch_hz(frame, sample_rate_value) for frame in active_frames) if pitch is not None]
        voiced_ratio = len(pitch_values) / max(1, len(active_frames))
        mean_pitch = sum(pitch_values) / len(pitch_values) if pitch_values else 0.0
        pitch_std = _stddev(pitch_values, mean_pitch) if pitch_values else 0.0
        band_totals = [0.0 for _ in _BAND_FREQUENCIES_HZ]
        for frame in active_frames:
            for index, frequency_hz in enumerate(_BAND_FREQUENCIES_HZ):
                band_totals[index] += _goertzel_power(frame, sample_rate_value, frequency_hz)
        total_band_power = sum(band_totals) or 1.0
        band_ratios = tuple((power / total_band_power for power in band_totals))
        vector = (round(mean_rms, 6), round(active_ratio, 6), round(sum(zcr_values) / len(zcr_values), 6), round(_stddev(zcr_values), 6), round(sum(delta_values) / len(delta_values), 6), round(_stddev(delta_values), 6), round(voiced_ratio, 6), round(min(mean_pitch / _MAX_PITCH_HZ, 1.5), 6), round(min(pitch_std / 200.0, 1.5), 6), *(round(value, 6) for value in band_ratios))
        quality = _legacy_quality_from_features(duration_ms=duration_ms, active_ratio=active_ratio, voiced_ratio=voiced_ratio, mean_rms=mean_rms)
        speech_ms = int(round(duration_ms * active_ratio))
        return _VoiceEmbedding(vector=vector, duration_ms=duration_ms, speech_duration_ms=speech_ms, quality=quality, backend=self.name, embedding_version=self.embedding_version, spoof_risk=0.0)

    def similarity(self, left: tuple[float, ...], right: tuple[float, ...]) -> float:
        distance = _weighted_distance(left, right)
        return max(0.0, min(1.0, math.exp(-4.0 * distance)))

class OnnxSpeakerEmbeddingBackend(_VoiceEmbeddingBackend):

    def __init__(self, *, model_path: str | Path, vad_backend: str='auto', vad_threshold: float=0.5, intra_threads: int=1, inter_threads: int=1, graph_optimization_level: str='all', enable_spoof_heuristics: bool=True) -> None:
        if np is None:
            raise ImportError('numpy is required for the ONNX voice-profile backend.')
        if ort is None:
            raise ImportError('onnxruntime is required for the ONNX voice-profile backend.')
        resolved_model = Path(model_path).expanduser().resolve(strict=False)
        if not resolved_model.exists():
            raise ValueError(f'Voice-profile ONNX model does not exist: {resolved_model}')
        self.model_path = resolved_model
        self.name = _DEFAULT_TEMPLATE_BACKEND
        self.embedding_version = _EMBEDDING_VERSION
        self.target_sample_rate = _DEFAULT_MODEL_SAMPLE_RATE
        self.vad_backend = _coerce_backend_name(vad_backend, default='auto')
        self.vad_threshold = _coerce_bounded_float(vad_threshold, default=0.5, minimum=0.05, maximum=0.95)
        self.intra_threads = max(1, _coerce_int_like(intra_threads, minimum=1, default=1) or 1)
        self.inter_threads = max(1, _coerce_int_like(inter_threads, minimum=1, default=1) or 1)
        self.graph_optimization_level = _coerce_backend_name(graph_optimization_level, default='all')
        self.enable_spoof_heuristics = enable_spoof_heuristics
        self._session_lock = threading.RLock()
        self._session = None
        self._input_name: str | None = None
        self._output_name: str | None = None
        self._silero_model = None

    def extract_from_wav_bytes(self, wav_bytes: bytes, *, min_sample_ms: int, max_sample_ms: int) -> _VoiceEmbedding:
        pcm_bytes, sample_rate, channels, raw_duration_ms = _decode_wav_bytes(wav_bytes)
        if raw_duration_ms > max_sample_ms * _DEFAULT_MAX_RAW_DURATION_MULTIPLIER:
            raise ValueError(f'Speech sample is too long for voice profiling. Please keep it under {max_sample_ms // 1000} seconds.')
        return self.extract_from_pcm16(pcm_bytes, sample_rate=sample_rate, channels=channels, min_sample_ms=min_sample_ms, max_sample_ms=max_sample_ms)

    def extract_from_pcm16(self, pcm_bytes: bytes, *, sample_rate: int, channels: int, min_sample_ms: int, max_sample_ms: int) -> _VoiceEmbedding:
        prepared = _prepare_audio_for_model(pcm_bytes, sample_rate=sample_rate, channels=channels, target_sample_rate=self.target_sample_rate, min_sample_ms=min_sample_ms, max_sample_ms=max_sample_ms, vad_fn=self._vad_segments if self._use_silero_vad() else None)
        features = self._compute_fbank(prepared.speech_samples)
        session = self._ensure_session()
        output = session.run([self._output_name], {self._input_name: features[None, :, :]})[0]
        vector_np = np.asarray(output[0], dtype=np.float32)
        norm = float(np.linalg.norm(vector_np))
        if not math.isfinite(norm) or norm <= 0.0:
            raise ValueError('Voice embedding model returned an invalid embedding.')
        vector_np = vector_np / norm
        vector = tuple((float(value) for value in vector_np.tolist()))
        spoof_risk = 0.0
        if self.enable_spoof_heuristics:
            spoof_risk = _spoof_risk_from_prepared_audio(prepared)
        return _VoiceEmbedding(vector=vector, duration_ms=prepared.duration_ms, speech_duration_ms=prepared.speech_duration_ms, quality=prepared.quality, backend=self.name, embedding_version=self.embedding_version, spoof_risk=spoof_risk)

    def similarity(self, left: tuple[float, ...], right: tuple[float, ...]) -> float:
        if len(left) != len(right) or (not left):
            raise ValueError('Voice profile embeddings must match the expected length.')
        if not _is_finite_vector(left) or not _is_finite_vector(right):
            raise ValueError('Voice profile embeddings must be finite.')
        if np is None:
            raise ImportError('numpy is required for cosine speaker similarity.')
        left_np = np.asarray(left, dtype=np.float32)
        right_np = np.asarray(right, dtype=np.float32)
        left_norm = float(np.linalg.norm(left_np))
        right_norm = float(np.linalg.norm(right_np))
        if left_norm <= 0.0 or right_norm <= 0.0:
            raise ValueError('Voice profile embeddings must have non-zero norm.')
        cosine = float(np.dot(left_np, right_np) / (left_norm * right_norm))
        return max(-1.0, min(1.0, cosine))

    def is_template_compatible(self, template: VoiceProfileTemplate) -> bool:
        return template.embedding_version == self.embedding_version and template.backend == self.name and bool(template.embedding)

    def _ensure_session(self):
        if self._session is not None:
            return self._session
        with self._session_lock:
            if self._session is not None:
                return self._session
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = self.intra_threads
            session_options.inter_op_num_threads = self.inter_threads
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session_options.graph_optimization_level = _ort_graph_optimization_level(self.graph_optimization_level)
            self._session = ort.InferenceSession(str(self.model_path), sess_options=session_options, providers=['CPUExecutionProvider'])
            inputs = self._session.get_inputs()
            outputs = self._session.get_outputs()
            if len(inputs) != 1:
                raise ValueError('Voice-profile ONNX model must expose exactly one input.')
            if not outputs:
                raise ValueError('Voice-profile ONNX model must expose at least one output.')
            self._input_name = str(inputs[0].name)
            self._output_name = str(outputs[0].name)
            return self._session

    def _compute_fbank(self, speech_samples: Any) -> Any:
        if np is None:
            raise ImportError('numpy is required for fbank extraction.')
        waveform = np.asarray(speech_samples, dtype=np.float32)
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError('Need a non-empty mono waveform for voice profiling.')
        if ta_kaldi is not None and torch is not None:
            waveform_t = torch.from_numpy(waveform.copy()).unsqueeze(0)
            features = ta_kaldi.fbank(waveform_t * (1 << 15), num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0, sample_frequency=self.target_sample_rate, window_type='hamming', use_energy=False)
            features = features - torch.mean(features, dim=0, keepdim=True)
            return features.numpy(force=True).astype('float32', copy=False)
        features = _numpy_log_mel_fbank(waveform=waveform, sample_rate=self.target_sample_rate, num_mel_bins=80, frame_length_ms=25.0, frame_shift_ms=10.0)
        features = features - np.mean(features, axis=0, keepdims=True)
        return features.astype('float32', copy=False)

    def _use_silero_vad(self) -> bool:
        if self.vad_backend == 'none':
            return False
        if self.vad_backend == 'energy':
            return False
        return load_silero_vad is not None and get_speech_timestamps is not None and np is not None

    def _vad_segments(self, speech_samples: Any, sample_rate: int) -> list[tuple[int, int]]:
        if not self._use_silero_vad():
            return _energy_vad_segments(np.asarray(speech_samples, dtype=np.float32), sample_rate=sample_rate)
        if self._silero_model is None:
            self._silero_model = load_silero_vad()
        audio = np.asarray(speech_samples, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError('Silero VAD expects a mono waveform.')
        if torch is None:
            return _energy_vad_segments(audio, sample_rate=sample_rate)
        timestamps = get_speech_timestamps(torch.from_numpy(audio.copy()), self._silero_model, threshold=self.vad_threshold, sampling_rate=sample_rate, min_speech_duration_ms=200, min_silence_duration_ms=100, speech_pad_ms=80, return_seconds=False)
        segments: list[tuple[int, int]] = []
        for item in timestamps:
            if not isinstance(item, dict):
                continue
            start = _coerce_int_like(item.get('start'), minimum=0, default=0) or 0
            end = _coerce_int_like(item.get('end'), minimum=0, default=0) or 0
            if end > start:
                segments.append((start, end))
        return segments

@functools.lru_cache(maxsize=1)
def _default_voice_embedding_backend_from_env() -> _VoiceEmbeddingBackend:
    backend_name = _coerce_backend_name(os.getenv('TWINR_VOICE_PROFILE_BACKEND', ''), default='')
    model_path = str(os.getenv('TWINR_VOICE_PROFILE_MODEL_PATH', '')).strip()
    if backend_name in {'onnx', 'onnx_speaker', 'wespeaker', 'ecapa'} or model_path:
        if not model_path:
            raise ValueError('TWINR_VOICE_PROFILE_MODEL_PATH must be set for the ONNX voice backend.')
        return OnnxSpeakerEmbeddingBackend(model_path=model_path, vad_backend=_coerce_backend_name(os.getenv('TWINR_VOICE_PROFILE_VAD_BACKEND', 'auto'), default='auto'), vad_threshold=_coerce_bounded_float(os.getenv('TWINR_VOICE_PROFILE_VAD_THRESHOLD', 0.5), default=0.5, minimum=0.05, maximum=0.95), intra_threads=_coerce_int_like(os.getenv('TWINR_VOICE_PROFILE_ORT_INTRA_THREADS', 1), minimum=1, default=1) or 1, inter_threads=_coerce_int_like(os.getenv('TWINR_VOICE_PROFILE_ORT_INTER_THREADS', 1), minimum=1, default=1) or 1, graph_optimization_level=_coerce_backend_name(os.getenv('TWINR_VOICE_PROFILE_ORT_GRAPH_OPT_LEVEL', 'all'), default='all'), enable_spoof_heuristics=_coerce_bool_like(os.getenv('TWINR_VOICE_PROFILE_ENABLE_SPOOF_HEURISTICS', 'true'), default=True))
    return LegacyDSPEmbeddingBackend()

def build_voice_embedding_backend_from_config(config: TwinrConfig | None) -> _VoiceEmbeddingBackend:
    backend_name = _coerce_backend_name(_env_or_config(config, 'voice_profile_backend', 'TWINR_VOICE_PROFILE_BACKEND', ''), default='')
    model_path_raw = str(_env_or_config(config, 'voice_profile_model_path', 'TWINR_VOICE_PROFILE_MODEL_PATH', '') or '').strip()
    if backend_name in {'onnx', 'onnx_speaker', 'wespeaker', 'ecapa'} or model_path_raw:
        if not model_path_raw:
            raise ValueError('voice_profile_model_path must point to a local ONNX/ORT speaker-embedding model.')
        return OnnxSpeakerEmbeddingBackend(model_path=model_path_raw, vad_backend=_coerce_backend_name(_env_or_config(config, 'voice_profile_vad_backend', 'TWINR_VOICE_PROFILE_VAD_BACKEND', 'auto'), default='auto'), vad_threshold=_coerce_bounded_float(_env_or_config(config, 'voice_profile_vad_threshold', 'TWINR_VOICE_PROFILE_VAD_THRESHOLD', 0.5), default=0.5, minimum=0.05, maximum=0.95), intra_threads=_coerce_int_like(_env_or_config(config, 'voice_profile_ort_intra_threads', 'TWINR_VOICE_PROFILE_ORT_INTRA_THREADS', 1), minimum=1, default=1) or 1, inter_threads=_coerce_int_like(_env_or_config(config, 'voice_profile_ort_inter_threads', 'TWINR_VOICE_PROFILE_ORT_INTER_THREADS', 1), minimum=1, default=1) or 1, graph_optimization_level=_coerce_backend_name(_env_or_config(config, 'voice_profile_ort_graph_opt_level', 'TWINR_VOICE_PROFILE_ORT_GRAPH_OPT_LEVEL', 'all'), default='all'), enable_spoof_heuristics=_coerce_bool_like(_env_or_config(config, 'voice_profile_enable_spoof_heuristics', 'TWINR_VOICE_PROFILE_ENABLE_SPOOF_HEURISTICS', True), default=True))
    return LegacyDSPEmbeddingBackend()

class VoiceProfileMonitor:

    def __init__(self, *, store: VoiceProfileStore, likely_threshold: float, uncertain_threshold: float, max_enrollment_samples: int, min_sample_ms: int, clock: Callable[[], datetime]=_utcnow, max_sample_ms: int=_DEFAULT_MAX_SAMPLE_MS, backend: _VoiceEmbeddingBackend | None=None, enrollment_match_threshold: float=_DEFAULT_ENROLL_MATCH_THRESHOLD, min_enrollment_quality: float=_DEFAULT_MIN_ENROLL_QUALITY, min_assessment_quality: float=_DEFAULT_MIN_ASSESS_QUALITY, spoof_guard_threshold: float=_DEFAULT_SPOOF_GUARD_THRESHOLD) -> None:
        self.store = store
        self.likely_threshold, self.uncertain_threshold = _normalize_thresholds(likely_threshold, uncertain_threshold)
        self.max_enrollment_samples = _coerce_int_like(max_enrollment_samples, minimum=1, default=1) or 1
        self.min_sample_ms = min(_MAX_ALLOWED_SAMPLE_MS, max(600, _coerce_int_like(min_sample_ms, minimum=0, default=600) or 600))
        configured_max_sample_ms = _coerce_int_like(max_sample_ms, minimum=1000, default=_DEFAULT_MAX_SAMPLE_MS) or _DEFAULT_MAX_SAMPLE_MS
        self.max_sample_ms = max(self.min_sample_ms, min(_MAX_ALLOWED_SAMPLE_MS, configured_max_sample_ms))
        self.clock = clock
        self.backend = backend or LegacyDSPEmbeddingBackend()
        self.enrollment_match_threshold = _coerce_bounded_float(enrollment_match_threshold, default=_DEFAULT_ENROLL_MATCH_THRESHOLD, minimum=0.0, maximum=1.0)
        self.min_enrollment_quality = _coerce_bounded_float(min_enrollment_quality, default=_DEFAULT_MIN_ENROLL_QUALITY, minimum=0.0, maximum=1.0)
        self.min_assessment_quality = _coerce_bounded_float(min_assessment_quality, default=_DEFAULT_MIN_ASSESS_QUALITY, minimum=0.0, maximum=1.0)
        self.spoof_guard_threshold = _coerce_bounded_float(spoof_guard_threshold, default=_DEFAULT_SPOOF_GUARD_THRESHOLD, minimum=0.0, maximum=1.0)

    @classmethod
    def from_config(cls, config: TwinrConfig) -> 'VoiceProfileMonitor':
        return cls(store=VoiceProfileStore.from_config(config), likely_threshold=config.voice_profile_likely_threshold, uncertain_threshold=config.voice_profile_uncertain_threshold, max_enrollment_samples=config.voice_profile_max_samples, min_sample_ms=config.voice_profile_min_sample_ms, max_sample_ms=getattr(config, 'voice_profile_max_sample_ms', _DEFAULT_MAX_SAMPLE_MS), backend=build_voice_embedding_backend_from_config(config), enrollment_match_threshold=_env_or_config(config, 'voice_profile_enrollment_match_threshold', 'TWINR_VOICE_PROFILE_ENROLL_MATCH_THRESHOLD', _DEFAULT_ENROLL_MATCH_THRESHOLD), min_enrollment_quality=_env_or_config(config, 'voice_profile_min_enrollment_quality', 'TWINR_VOICE_PROFILE_MIN_ENROLL_QUALITY', _DEFAULT_MIN_ENROLL_QUALITY), min_assessment_quality=_env_or_config(config, 'voice_profile_min_assessment_quality', 'TWINR_VOICE_PROFILE_MIN_ASSESS_QUALITY', _DEFAULT_MIN_ASSESS_QUALITY), spoof_guard_threshold=_env_or_config(config, 'voice_profile_spoof_guard_threshold', 'TWINR_VOICE_PROFILE_SPOOF_GUARD_THRESHOLD', _DEFAULT_SPOOF_GUARD_THRESHOLD))

    def summary(self) -> VoiceProfileSummary:
        template = self.store.load()
        if template is None or (not self.backend.is_template_compatible(template)):
            return VoiceProfileSummary(enrolled=False, store_path=str(self.store.path), backend=self.backend.name)
        return VoiceProfileSummary(enrolled=True, sample_count=template.sample_count, updated_at=template.updated_at, average_duration_ms=template.average_duration_ms, average_speech_duration_ms=template.average_speech_duration_ms, quality_score=template.quality_score, store_path=str(self.store.path), backend=template.backend)

    def reset(self) -> VoiceProfileSummary:
        self.store.clear()
        return self.summary()

    def enroll_wav_bytes(self, wav_bytes: bytes) -> VoiceProfileTemplate:
        embedding = self.backend.extract_from_wav_bytes(wav_bytes, min_sample_ms=self.min_sample_ms, max_sample_ms=self.max_sample_ms)
        return self._merge_embedding(embedding)

    def enroll_pcm16(self, pcm_bytes: bytes, *, sample_rate: int, channels: int) -> VoiceProfileTemplate:
        embedding = self.backend.extract_from_pcm16(pcm_bytes, sample_rate=sample_rate, channels=channels, min_sample_ms=self.min_sample_ms, max_sample_ms=self.max_sample_ms)
        return self._merge_embedding(embedding)

    def assess_wav_bytes(self, wav_bytes: bytes) -> VoiceAssessment:
        template = self.store.load()
        if template is None or (not self.backend.is_template_compatible(template)):
            return VoiceAssessment(status='not_enrolled', label='Not enrolled', detail='No compatible saved voice profile is available on this device yet.')
        try:
            embedding = self.backend.extract_from_wav_bytes(wav_bytes, min_sample_ms=self.min_sample_ms, max_sample_ms=self.max_sample_ms)
        except ValueError as exc:
            return self._invalid_assessment(str(exc))
        return self._assessment_for_embedding(template, embedding)

    def assess_pcm16(self, pcm_bytes: bytes, *, sample_rate: int, channels: int) -> VoiceAssessment:
        template = self.store.load()
        if template is None or (not self.backend.is_template_compatible(template)):
            return VoiceAssessment(status='not_enrolled', label='Not enrolled', detail='No compatible saved voice profile is available on this device yet.')
        try:
            embedding = self.backend.extract_from_pcm16(pcm_bytes, sample_rate=sample_rate, channels=channels, min_sample_ms=self.min_sample_ms, max_sample_ms=self.max_sample_ms)
        except ValueError as exc:
            return self._invalid_assessment(str(exc))
        return self._assessment_for_embedding(template, embedding)

    def _merge_embedding(self, embedding: _VoiceEmbedding) -> VoiceProfileTemplate:
        if embedding.quality < self.min_enrollment_quality:
            raise ValueError('That voice sample is not clear enough to save. Please try again in a quieter room and speak a little longer.')
        now_iso = _utc_iso(self.clock())

        def updater(existing: VoiceProfileTemplate | None) -> VoiceProfileTemplate:
            if existing is None or (not self.backend.is_template_compatible(existing)) or len(existing.embedding) != len(embedding.vector):
                return VoiceProfileTemplate(embedding=embedding.vector, sample_count=1, updated_at=now_iso, embedding_version=embedding.embedding_version, average_duration_ms=embedding.duration_ms, backend=embedding.backend, average_speech_duration_ms=embedding.speech_duration_ms, quality_score=embedding.quality)
            try:
                similarity = self.backend.similarity(existing.embedding, embedding.vector)
            except ValueError as exc:
                raise ValueError('The saved voice profile is not compatible with the active backend. Please re-enroll.') from exc
            if similarity < self.enrollment_match_threshold:
                raise ValueError('That sample does not sound like the currently enrolled speaker. Reset the profile before enrolling a different voice.')
            previous_weight = min(existing.sample_count, self.max_enrollment_samples - 1)
            combined_weight = previous_weight + 1
            merged_vector = _merge_vectors(existing.embedding, embedding.vector, previous_weight, combined_weight)
            average_duration_ms = int(round((existing.average_duration_ms * previous_weight + embedding.duration_ms) / combined_weight))
            average_speech_duration_ms = int(round((existing.average_speech_duration_ms * previous_weight + embedding.speech_duration_ms) / combined_weight))
            quality_score = (existing.quality_score * previous_weight + embedding.quality) / combined_weight
            return VoiceProfileTemplate(embedding=merged_vector, sample_count=min(existing.sample_count + 1, self.max_enrollment_samples), updated_at=now_iso, embedding_version=embedding.embedding_version, average_duration_ms=average_duration_ms, backend=embedding.backend, average_speech_duration_ms=average_speech_duration_ms, quality_score=round(max(0.0, min(1.0, quality_score)), 6))
        return self.store.update(updater)

    def _invalid_assessment(self, reason: str) -> VoiceAssessment:
        detail = f'{_friendly_audio_error(reason)} Please ask the speaker to try again in a quiet room.'
        return VoiceAssessment(status='invalid_sample', label='Could not verify', detail=detail, checked_at=_utc_iso(self.clock()))

    def _assessment_for_embedding(self, template: VoiceProfileTemplate, embedding: _VoiceEmbedding) -> VoiceAssessment:
        try:
            raw_similarity = self.backend.similarity(template.embedding, embedding.vector)
        except ValueError:
            return self._invalid_assessment('Voice match could not be calculated safely.')
        checked_at = _utc_iso(self.clock())
        confidence = _calibrated_confidence(raw_similarity=raw_similarity, backend=self.backend.name, quality=embedding.quality, speech_duration_ms=embedding.speech_duration_ms)
        if embedding.spoof_risk >= self.spoof_guard_threshold:
            return VoiceAssessment(status='spoof_suspected', label='Possible replay', detail='This audio may be replayed or synthetic. Please confirm in person before acting.', confidence=confidence, checked_at=checked_at)
        if embedding.quality < self.min_assessment_quality:
            confidence = min(confidence, self.uncertain_threshold)
            return VoiceAssessment(status='uncertain', label='Uncertain', detail='The recording was too weak or noisy for a reliable voice match. Please ask again before doing anything important.', confidence=confidence, checked_at=checked_at)
        if confidence >= self.likely_threshold:
            return VoiceAssessment(status='likely_user', label='Likely user', detail='This voice sounds like the saved voice profile.', confidence=confidence, checked_at=checked_at)
        if confidence >= self.uncertain_threshold:
            return VoiceAssessment(status='uncertain', label='Uncertain', detail='This might be the saved voice. Ask before doing anything important.', confidence=confidence, checked_at=checked_at)
        return VoiceAssessment(status='unknown_voice', label='Unknown voice', detail='This voice does not sound like the saved voice profile.', confidence=confidence, checked_at=checked_at)

def _friendly_audio_error(reason: str) -> str:
    normalized = reason.strip().lower()
    if 'too long' in normalized:
        return 'That recording is longer than needed.'
    if 'too short' in normalized or 'longer speech sample' in normalized or 'steady speech' in normalized:
        return 'I need a little more clear speech.'
    if 'quiet room' in normalized:
        return reason.strip()
    return 'I could not use that recording.'

def extract_voice_embedding_from_wav_bytes(wav_bytes: bytes, *, min_sample_ms: int, max_sample_ms: int=_DEFAULT_MAX_SAMPLE_MS) -> _VoiceEmbedding:
    backend = _default_voice_embedding_backend_from_env()
    return backend.extract_from_wav_bytes(wav_bytes, min_sample_ms=min_sample_ms, max_sample_ms=max_sample_ms)

def extract_voice_embedding_from_pcm16(pcm_bytes: bytes, *, sample_rate: int, channels: int, min_sample_ms: int, max_sample_ms: int=_DEFAULT_MAX_SAMPLE_MS) -> _VoiceEmbedding:
    backend = _default_voice_embedding_backend_from_env()
    return backend.extract_from_pcm16(pcm_bytes, sample_rate=sample_rate, channels=channels, min_sample_ms=min_sample_ms, max_sample_ms=max_sample_ms)

def _decode_wav_bytes(wav_bytes: bytes) -> tuple[bytes, int, int, int]:
    if len(wav_bytes) > _MAX_AUDIO_UPLOAD_BYTES:
        raise ValueError('Audio input is too large for voice profiling.')
    try:
        with wave.open(io.BytesIO(wav_bytes), 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_count = wav_file.getnframes()
            if sample_width != 2:
                raise ValueError('Voice profiling currently expects 16-bit WAV input.')
            if sample_rate <= 0:
                raise ValueError('Sample rate must be greater than zero.')
            if channels <= 0:
                raise ValueError('Channel count must be greater than zero.')
            duration_ms = int(frame_count * 1000 / sample_rate)
            pcm_bytes = wav_file.readframes(frame_count)
    except (EOFError, wave.Error) as exc:
        raise ValueError('Audio input is not a valid WAV recording.') from exc
    return (pcm_bytes, sample_rate, channels, duration_ms)

def _prepare_audio_for_model(pcm_bytes: bytes, *, sample_rate: int, channels: int, target_sample_rate: int, min_sample_ms: int, max_sample_ms: int, vad_fn: Callable[[Any, int], list[tuple[int, int]]] | None) -> _PreparedAudio:
    if np is None:
        raise ImportError('numpy is required for the model-based voice backend.')
    sample_rate_value = _coerce_int_like(sample_rate, minimum=1, default=None)
    if sample_rate_value is None:
        raise ValueError('Sample rate must be greater than zero.')
    channels_value = _coerce_int_like(channels, minimum=1, default=None)
    if channels_value is None:
        raise ValueError('Channel count must be greater than zero.')
    if len(pcm_bytes) > _MAX_AUDIO_UPLOAD_BYTES:
        raise ValueError('Audio input is too large for voice profiling.')
    waveform = _pcm16_to_mono_numpy(pcm_bytes, channels=channels_value)
    raw_duration_ms = int(waveform.shape[0] * 1000 / sample_rate_value)
    if raw_duration_ms < min_sample_ms:
        raise ValueError('Need a longer speech sample for voice profiling.')
    if raw_duration_ms > max_sample_ms * _DEFAULT_MAX_RAW_DURATION_MULTIPLIER:
        raise ValueError(f'Speech sample is too long for voice profiling. Please keep it under {max_sample_ms // 1000} seconds.')
    if sample_rate_value != target_sample_rate:
        waveform = _resample_numpy(waveform, src_rate=sample_rate_value, dst_rate=target_sample_rate)
        sample_rate_value = target_sample_rate
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.size == 0:
        raise ValueError('Need a longer speech sample for voice profiling.')
    waveform = waveform - np.mean(waveform)
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak > 1.0:
        waveform = np.clip(waveform, -1.0, 1.0)
        peak = 1.0
    if peak > 0.98:
        waveform = waveform / peak * 0.98
    segments = vad_fn(waveform, sample_rate_value) if vad_fn is not None else _energy_vad_segments(waveform, sample_rate=sample_rate_value)
    if not segments:
        segments = _energy_vad_segments(waveform, sample_rate=sample_rate_value)
    speech = _collect_segments(waveform, segments)
    if speech.size == 0:
        raise ValueError('Need more steady speech for voice profiling.')
    max_samples = int(sample_rate_value * max_sample_ms / 1000)
    if speech.shape[0] > max_samples:
        speech = speech[:max_samples]
    speech_duration_ms = int(speech.shape[0] * 1000 / sample_rate_value)
    if speech_duration_ms < min_sample_ms:
        raise ValueError('Need a longer speech sample for voice profiling.')
    duration_ms = int(waveform.shape[0] * 1000 / sample_rate_value)
    speech_ratio = speech.shape[0] / max(1, waveform.shape[0])
    clipping_ratio = float(np.mean(np.abs(speech) >= 0.985)) if speech.size else 0.0
    snr_db = _approx_snr_db(speech, sample_rate=sample_rate_value)
    spectral_rolloff_hz = _spectral_rolloff_hz(speech, sample_rate=sample_rate_value)
    spectral_flatness = _spectral_flatness(speech, sample_rate=sample_rate_value)
    quality = _quality_score_from_audio(speech_duration_ms=speech_duration_ms, speech_ratio=speech_ratio, clipping_ratio=clipping_ratio, snr_db=snr_db)
    return _PreparedAudio(samples=waveform, sample_rate=sample_rate_value, duration_ms=duration_ms, speech_samples=speech, speech_duration_ms=speech_duration_ms, speech_ratio=float(max(0.0, min(1.0, speech_ratio))), quality=quality, clipping_ratio=clipping_ratio, spectral_rolloff_hz=spectral_rolloff_hz, spectral_flatness=spectral_flatness, snr_db=snr_db)

def _quality_score_from_audio(*, speech_duration_ms: int, speech_ratio: float, clipping_ratio: float, snr_db: float) -> float:
    duration_factor = min(1.0, speech_duration_ms / 2800.0)
    speech_ratio_factor = min(1.0, max(0.0, (speech_ratio - 0.1) / 0.45))
    snr_factor = min(1.0, max(0.0, (snr_db - 5.0) / 18.0))
    clip_factor = min(1.0, max(0.0, 1.0 - clipping_ratio / 0.03))
    quality = 0.38 * duration_factor + 0.27 * snr_factor + 0.2 * speech_ratio_factor + 0.15 * clip_factor
    return round(max(0.0, min(1.0, quality)), 6)

def _legacy_quality_from_features(*, duration_ms: int, active_ratio: float, voiced_ratio: float, mean_rms: float) -> float:
    duration_factor = min(1.0, duration_ms / 2800.0)
    activity_factor = min(1.0, max(0.0, (active_ratio - 0.15) / 0.55))
    voiced_factor = min(1.0, max(0.0, (voiced_ratio - 0.1) / 0.7))
    rms_factor = min(1.0, max(0.0, (mean_rms - 0.01) / 0.1))
    quality = 0.35 * duration_factor + 0.25 * activity_factor + 0.2 * voiced_factor + 0.2 * rms_factor
    return round(max(0.0, min(1.0, quality)), 6)

def _spoof_risk_from_prepared_audio(prepared: _PreparedAudio) -> float:
    low_bandwidth_risk = min(1.0, max(0.0, (2600.0 - prepared.spectral_rolloff_hz) / 1600.0))
    ultra_flat_risk = min(1.0, max(0.0, (0.04 - prepared.spectral_flatness) / 0.04))
    clipping_risk = min(1.0, prepared.clipping_ratio / 0.03)
    clean_but_narrow = 1.0 if prepared.snr_db > 18.0 and prepared.spectral_rolloff_hz < 2300.0 else 0.0
    risk = max(0.0, min(1.0, 0.35 * low_bandwidth_risk + 0.2 * ultra_flat_risk + 0.2 * clipping_risk + 0.25 * clean_but_narrow))
    return round(risk, 6)

def _calibrated_confidence(*, raw_similarity: float, backend: str, quality: float, speech_duration_ms: int) -> float:
    if backend == _LEGACY_TEMPLATE_BACKEND:
        base = max(0.0, min(1.0, raw_similarity))
        duration_factor = min(1.0, speech_duration_ms / 3000.0)
        confidence = base * (0.65 + 0.35 * min(1.0, max(quality, duration_factor)))
        return round(max(0.0, min(1.0, confidence)), 6)
    centered = (raw_similarity - 0.42) * 10.0
    base = 1.0 / (1.0 + math.exp(-centered))
    duration_factor = min(1.0, speech_duration_ms / 3500.0)
    quality_factor = 0.45 + 0.55 * min(1.0, max(quality, duration_factor))
    confidence = base * quality_factor
    return round(max(0.0, min(1.0, confidence)), 6)

def _merge_vectors(left: tuple[float, ...], right: tuple[float, ...], previous_weight: int, combined_weight: int) -> tuple[float, ...]:
    if len(left) != len(right):
        raise ValueError('Voice profile embeddings must match the expected length.')
    if previous_weight <= 0:
        merged = right
    else:
        merged = tuple((((old_value * previous_weight + new_value) / combined_weight) for old_value, new_value in zip(left, right, strict=True)))
    if right and abs(_l2_norm(merged) - 1.0) > 0.03 and len(merged) > len(_PROFILE_WEIGHTS):
        norm = _l2_norm(merged)
        if norm > 0:
            merged = tuple((value / norm for value in merged))
    return merged

def _l2_norm(values: tuple[float, ...]) -> float:
    return math.sqrt(sum((value * value for value in values)))

def _pcm16_to_mono_numpy(pcm_bytes: bytes, *, channels: int) -> Any:
    if np is None:
        raise ImportError('numpy is required for the model-based voice backend.')
    if channels <= 0:
        raise ValueError('Channel count must be greater than zero.')
    if len(pcm_bytes) % 2 != 0:
        raise ValueError('PCM input must contain complete 16-bit samples.')
    samples = np.frombuffer(pcm_bytes, dtype='<i2')
    if channels > 1:
        if samples.size % channels != 0:
            raise ValueError('PCM input must contain whole frames for every channel.')
        samples = samples.reshape(-1, channels).astype(np.float32).mean(axis=1)
    else:
        samples = samples.astype(np.float32)
    return np.clip(samples / 32768.0, -1.0, 1.0).astype(np.float32, copy=False)

def _resample_numpy(waveform: Any, *, src_rate: int, dst_rate: int) -> Any:
    if np is None:
        raise ImportError('numpy is required for the model-based voice backend.')
    if src_rate == dst_rate:
        return np.asarray(waveform, dtype=np.float32)
    if waveform.size == 0:
        return np.asarray(waveform, dtype=np.float32)
    if resample_poly is not None:
        gcd = math.gcd(src_rate, dst_rate)
        up = dst_rate // gcd
        down = src_rate // gcd
        resampled = resample_poly(np.asarray(waveform, dtype=np.float32), up, down)
        return np.asarray(resampled, dtype=np.float32)
    ratio = dst_rate / src_rate
    output_length = max(1, int(round(waveform.shape[0] * ratio)))
    x_old = np.linspace(0.0, 1.0, num=waveform.shape[0], endpoint=False, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, num=output_length, endpoint=False, dtype=np.float64)
    return np.interp(x_new, x_old, np.asarray(waveform, dtype=np.float32)).astype(np.float32)

def _resample_tuple(samples: tuple[float, ...], *, src_rate: int, dst_rate: int) -> tuple[float, ...]:
    if src_rate == dst_rate or (not samples):
        return samples
    if np is not None:
        resampled = _resample_numpy(np.asarray(samples, dtype=np.float32), src_rate=src_rate, dst_rate=dst_rate)
        return tuple((float(value) for value in resampled.tolist()))
    ratio = dst_rate / src_rate
    output_length = max(1, int(round(len(samples) * ratio)))
    result: list[float] = []
    for index in range(output_length):
        position = index / ratio
        left = int(math.floor(position))
        right = min(left + 1, len(samples) - 1)
        frac = position - left
        left_value = samples[left]
        right_value = samples[right]
        result.append(left_value * (1.0 - frac) + right_value * frac)
    return tuple(result)

def _energy_vad_segments(waveform: Any, *, sample_rate: int) -> list[tuple[int, int]]:
    if np is None:
        samples = tuple((float(v) for v in waveform))
        frame_size = max(1, int(sample_rate * 0.03))
        hop = max(1, int(sample_rate * 0.015))
        frames = _frame_windows(samples, frame_size=frame_size, hop_size=hop)
        if not frames:
            return []
        rms_values = [_frame_rms(frame) for frame in frames]
        peak = max(rms_values) if rms_values else 0.0
        threshold = max(0.01, peak * 0.28)
        segments: list[tuple[int, int]] = []
        current_start: int | None = None
        for index, rms in enumerate(rms_values):
            if rms >= threshold:
                if current_start is None:
                    current_start = index * hop
            elif current_start is not None:
                end = min(len(samples), index * hop + frame_size)
                if end > current_start:
                    segments.append((current_start, end))
                current_start = None
        if current_start is not None:
            segments.append((current_start, len(samples)))
        return _merge_segments(segments, sample_rate=sample_rate)
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.size == 0:
        return []
    frame_size = max(160, int(round(sample_rate * 0.03)))
    hop = max(80, int(round(sample_rate * 0.015)))
    if audio.shape[0] < frame_size:
        return [(0, int(audio.shape[0]))] if audio.shape[0] else []
    pad = frame_size - 1
    padded = np.pad(audio, (0, pad), mode='constant')
    frame_count = 1 + (audio.shape[0] - frame_size) // hop
    frames = np.lib.stride_tricks.as_strided(padded, shape=(frame_count, frame_size), strides=(hop * padded.strides[0], padded.strides[0]), writeable=False)
    rms = np.sqrt(np.mean(frames.astype(np.float32) ** 2, axis=1))
    peak = float(np.max(rms)) if rms.size else 0.0
    threshold = max(0.01, peak * 0.28)
    active = rms >= threshold
    segments: list[tuple[int, int]] = []
    current_start: int | None = None
    for index, is_active in enumerate(active.tolist()):
        start = index * hop
        end = min(audio.shape[0], start + frame_size)
        if is_active:
            if current_start is None:
                current_start = start
        elif current_start is not None:
            if end > current_start:
                segments.append((current_start, end))
            current_start = None
    if current_start is not None:
        segments.append((current_start, int(audio.shape[0])))
    return _merge_segments(segments, sample_rate=sample_rate)

def _merge_segments(segments: list[tuple[int, int]], *, sample_rate: int) -> list[tuple[int, int]]:
    if not segments:
        return []
    gap = max(1, int(round(sample_rate * 0.1)))
    min_len = max(1, int(round(sample_rate * 0.2)))
    merged: list[tuple[int, int]] = []
    current_start, current_end = segments[0]
    for start, end in segments[1:]:
        if start - current_end <= gap:
            current_end = max(current_end, end)
            continue
        if current_end - current_start >= min_len:
            merged.append((current_start, current_end))
        current_start, current_end = (start, end)
    if current_end - current_start >= min_len:
        merged.append((current_start, current_end))
    return merged

def _collect_segments(waveform: Any, segments: list[tuple[int, int]]) -> Any:
    if np is None:
        samples = tuple((float(v) for v in waveform))
        out: list[float] = []
        for start, end in segments:
            out.extend(samples[start:end])
        return tuple(out)
    parts = [np.asarray(waveform[start:end], dtype=np.float32) for start, end in segments if end > start]
    if not parts:
        return np.asarray([], dtype=np.float32)
    return np.concatenate(parts, axis=0).astype(np.float32, copy=False)

def _approx_snr_db(waveform: Any, *, sample_rate: int) -> float:
    if np is None:
        return 0.0
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.size < max(32, sample_rate // 10):
        return 0.0
    frame_size = max(160, int(round(sample_rate * 0.03)))
    hop = max(80, int(round(sample_rate * 0.015)))
    if audio.shape[0] < frame_size:
        rms = float(np.sqrt(np.mean(audio ** 2))) + 1e-06
        return max(0.0, 20.0 * math.log10(rms / 0.0001))
    frame_count = 1 + (audio.shape[0] - frame_size) // hop
    padded = np.pad(audio, (0, frame_size), mode='constant')
    frames = np.lib.stride_tricks.as_strided(padded, shape=(frame_count, frame_size), strides=(hop * padded.strides[0], padded.strides[0]), writeable=False)
    rms = np.sqrt(np.mean(frames.astype(np.float32) ** 2, axis=1)) + 1e-06
    noise_floor = float(np.percentile(rms, 10))
    speech_level = float(np.percentile(rms, 90))
    return max(0.0, 20.0 * math.log10(max(speech_level, 1e-06) / max(noise_floor, 1e-06)))

def _spectral_rolloff_hz(waveform: Any, *, sample_rate: int) -> float:
    if np is None:
        return 0.0
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.size == 0:
        return 0.0
    n_fft = 1 << max(8, int(math.ceil(math.log2(min(max(audio.shape[0], 256), 4096)))))
    spectrum = np.abs(np.fft.rfft(audio[:min(audio.shape[0], n_fft)], n=n_fft)) ** 2
    if spectrum.size <= 1:
        return 0.0
    cumulative = np.cumsum(spectrum)
    threshold = float(cumulative[-1]) * 0.95
    index = int(np.searchsorted(cumulative, threshold))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    return float(freqs[min(index, freqs.shape[0] - 1)])

def _spectral_flatness(waveform: Any, *, sample_rate: int) -> float:
    del sample_rate
    if np is None:
        return 0.0
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.size == 0:
        return 0.0
    n_fft = 1 << max(8, int(math.ceil(math.log2(min(max(audio.shape[0], 256), 4096)))))
    power = np.abs(np.fft.rfft(audio[:min(audio.shape[0], n_fft)], n=n_fft)) ** 2
    power = np.asarray(power, dtype=np.float64) + 1e-12
    geometric = float(np.exp(np.mean(np.log(power))))
    arithmetic = float(np.mean(power))
    return max(0.0, min(1.0, geometric / max(arithmetic, 1e-12)))

def _ort_graph_optimization_level(name: str):
    if ort is None:
        raise ImportError('onnxruntime is required for graph optimization settings.')
    normalized = _coerce_backend_name(name, default='all')
    mapping = {'disabled': ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 'basic': ort.GraphOptimizationLevel.ORT_ENABLE_BASIC, 'extended': ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED, 'layout': ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 'all': ort.GraphOptimizationLevel.ORT_ENABLE_ALL}
    return mapping.get(normalized, ort.GraphOptimizationLevel.ORT_ENABLE_ALL)

def _numpy_log_mel_fbank(*, waveform: Any, sample_rate: int, num_mel_bins: int, frame_length_ms: float, frame_shift_ms: float) -> Any:
    if np is None:
        raise ImportError('numpy is required for fbank extraction.')
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.size == 0:
        raise ValueError('Need more steady speech for voice profiling.')
    frame_length = max(1, int(round(sample_rate * (frame_length_ms / 1000.0))))
    frame_shift = max(1, int(round(sample_rate * (frame_shift_ms / 1000.0))))
    n_fft = 1
    while n_fft < frame_length:
        n_fft <<= 1
    if audio.shape[0] < frame_length:
        audio = np.pad(audio, (0, frame_length - audio.shape[0]), mode='constant')
    remainder = (audio.shape[0] - frame_length) % frame_shift
    if remainder:
        audio = np.pad(audio, (0, frame_shift - remainder), mode='constant')
    frame_count = 1 + (audio.shape[0] - frame_length) // frame_shift
    frames = np.lib.stride_tricks.as_strided(audio, shape=(frame_count, frame_length), strides=(frame_shift * audio.strides[0], audio.strides[0]), writeable=False).copy()
    hamming = np.hamming(frame_length).astype(np.float32)
    frames *= hamming[None, :]
    spectrum = np.fft.rfft(frames, n=n_fft, axis=1)
    power = (np.abs(spectrum) ** 2).astype(np.float32)
    mel_filters = _mel_filterbank(sample_rate=sample_rate, n_fft=n_fft, num_mel_bins=num_mel_bins, low_freq=20.0, high_freq=sample_rate / 2.0 - 100.0)
    mel_energy = np.maximum(np.matmul(power, mel_filters.T), 1e-10)
    return np.log(mel_energy).astype(np.float32)

@functools.lru_cache(maxsize=16)
def _mel_filterbank(*, sample_rate: int, n_fft: int, num_mel_bins: int, low_freq: float, high_freq: float) -> Any:
    if np is None:
        raise ImportError('numpy is required for mel filterbanks.')

    def hz_to_mel(freq_hz: float) -> float:
        return 2595.0 * math.log10(1.0 + freq_hz / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    low_mel = hz_to_mel(low_freq)
    high_mel = hz_to_mel(max(low_freq + 1.0, high_freq))
    mel_points = np.linspace(low_mel, high_mel, num_mel_bins + 2, dtype=np.float64)
    hz_points = np.array([mel_to_hz(value) for value in mel_points], dtype=np.float64)
    fft_bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(np.int32)
    fft_bins = np.clip(fft_bins, 0, n_fft // 2)
    filters = np.zeros((num_mel_bins, n_fft // 2 + 1), dtype=np.float32)
    for index in range(1, num_mel_bins + 1):
        left = int(fft_bins[index - 1])
        center = int(fft_bins[index])
        right = int(fft_bins[index + 1])
        if center <= left:
            center = min(left + 1, filters.shape[1] - 1)
        if right <= center:
            right = min(center + 1, filters.shape[1])
        for bin_index in range(left, center):
            filters[index - 1, bin_index] = (bin_index - left) / max(1, center - left)
        for bin_index in range(center, right):
            filters[index - 1, bin_index] = (right - bin_index) / max(1, right - center)
    return filters

def _mono_samples(pcm_bytes: bytes, *, channels: int) -> tuple[float, ...]:
    if channels <= 0:
        raise ValueError('Channel count must be greater than zero.')
    if len(pcm_bytes) % 2 != 0:
        raise ValueError('PCM input must contain complete 16-bit samples.')
    samples = array('h')
    try:
        samples.frombytes(pcm_bytes)
    except ValueError as exc:
        raise ValueError('PCM input must contain complete 16-bit samples.') from exc
    if sys.byteorder != 'little':
        samples.byteswap()
    if len(samples) % channels != 0:
        raise ValueError('PCM input must contain whole frames for every channel.')
    if channels == 1:
        return tuple((sample / 32768.0 for sample in samples))
    mono: list[float] = []
    stride = channels
    for index in range(0, len(samples), stride):
        mono.append(sum(samples[index:index + stride]) / (stride * 32768.0))
    return tuple(mono)

def _frame_windows(samples: tuple[float, ...], *, frame_size: int, hop_size: int) -> tuple[tuple[float, ...], ...]:
    if len(samples) < frame_size:
        return ()
    return tuple((tuple(samples[offset:offset + frame_size]) for offset in range(0, len(samples) - frame_size + 1, hop_size)))

def _frame_rms(frame: tuple[float, ...]) -> float:
    return math.sqrt(sum((sample * sample for sample in frame)) / max(1, len(frame)))

def _zero_crossing_rate(frame: tuple[float, ...]) -> float:
    crossings = 0
    for left, right in zip(frame, frame[1:], strict=False):
        if left >= 0 > right or left < 0 <= right:
            crossings += 1
    return crossings / max(1, len(frame) - 1)

def _average_absolute_delta(frame: tuple[float, ...]) -> float:
    deltas = [abs(right - left) for left, right in zip(frame, frame[1:], strict=False)]
    return sum(deltas) / max(1, len(deltas))

def _estimate_pitch_hz(frame: tuple[float, ...], sample_rate: int) -> float | None:
    min_lag = max(2, int(sample_rate / _MAX_PITCH_HZ))
    max_lag = min(len(frame) - 2, int(sample_rate / _MIN_PITCH_HZ))
    if max_lag <= min_lag:
        return None
    best_correlation = 0.0
    best_lag = 0
    for lag in range(min_lag, max_lag + 1):
        numerator = 0.0
        left_energy = 0.0
        right_energy = 0.0
        for index in range(len(frame) - lag):
            left = frame[index]
            right = frame[index + lag]
            numerator += left * right
            left_energy += left * left
            right_energy += right * right
        if left_energy <= 0 or right_energy <= 0:
            continue
        correlation = numerator / math.sqrt(left_energy * right_energy)
        if correlation > best_correlation:
            best_correlation = correlation
            best_lag = lag
    if best_lag <= 0 or best_correlation < 0.35:
        return None
    return sample_rate / best_lag

def _goertzel_power(frame: tuple[float, ...], sample_rate: int, frequency_hz: float) -> float:
    if not frame or frequency_hz <= 0:
        return 0.0
    bin_index = int(0.5 + len(frame) * frequency_hz / sample_rate)
    omega = 2.0 * math.pi * bin_index / len(frame)
    coefficient = 2.0 * math.cos(omega)
    prev = 0.0
    prev2 = 0.0
    for sample in frame:
        current = sample + coefficient * prev - prev2
        prev2 = prev
        prev = current
    return max(0.0, prev2 * prev2 + prev * prev - coefficient * prev * prev2)

def _stddev(values: list[float] | tuple[float, ...], mean: float | None=None) -> float:
    if not values:
        return 0.0
    effective_mean = sum(values) / len(values) if mean is None else mean
    variance = sum(((value - effective_mean) ** 2 for value in values)) / len(values)
    return math.sqrt(max(0.0, variance))

def _weighted_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(_PROFILE_WEIGHTS) or len(right) != len(_PROFILE_WEIGHTS):
        raise ValueError('Voice profile embeddings must match the expected length.')
    if not _is_finite_vector(left) or not _is_finite_vector(right):
        raise ValueError('Voice profile embeddings must be finite.')
    total_weight = sum(_PROFILE_WEIGHTS)
    weighted_error = sum((weight * abs(left_value - right_value) for weight, left_value, right_value in zip(_PROFILE_WEIGHTS, left, right, strict=True)))
    return weighted_error / max(0.0001, total_weight)

def voice_embedding_confidence(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) == len(_PROFILE_WEIGHTS) and len(right) == len(_PROFILE_WEIGHTS):
        distance = _weighted_distance(left, right)
        return max(0.0, min(1.0, math.exp(-4.0 * distance)))
    if len(left) != len(right) or not left:
        raise ValueError('Voice profile embeddings must match the expected length.')
    if not _is_finite_vector(left) or not _is_finite_vector(right):
        raise ValueError('Voice profile embeddings must be finite.')
    if np is None:
        raise ImportError('numpy is required for cosine speaker similarity.')
    left_np = np.asarray(left, dtype=np.float32)
    right_np = np.asarray(right, dtype=np.float32)
    left_norm = float(np.linalg.norm(left_np))
    right_norm = float(np.linalg.norm(right_np))
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise ValueError('Voice profile embeddings must have non-zero norm.')
    cosine = float(np.dot(left_np, right_np) / (left_norm * right_norm))
    return _calibrated_confidence(raw_similarity=cosine, backend=_DEFAULT_TEMPLATE_BACKEND, quality=1.0, speech_duration_ms=3500)