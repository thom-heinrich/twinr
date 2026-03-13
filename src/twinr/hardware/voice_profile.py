from __future__ import annotations

from array import array
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import io
import json
import math
from pathlib import Path
from typing import Callable
import sys
import wave

from twinr.agent.base_agent.config import TwinrConfig

_FRAME_MS = 30
_HOP_MS = 15
_MIN_PITCH_HZ = 85.0
_MAX_PITCH_HZ = 350.0
_BAND_FREQUENCIES_HZ = (250.0, 500.0, 1000.0, 1800.0, 2800.0)
_EMBEDDING_VERSION = 1
_PROFILE_WEIGHTS = (
    0.30,
    0.45,
    0.90,
    0.50,
    0.75,
    0.40,
    0.75,
    1.00,
    0.55,
    1.00,
    1.00,
    1.00,
    1.00,
    1.00,
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def voice_profile_store_path(config: TwinrConfig) -> Path:
    configured = Path(config.voice_profile_store_path)
    if configured.is_absolute():
        return configured
    return (Path(config.project_root) / configured).resolve()


@dataclass(frozen=True, slots=True)
class VoiceProfileTemplate:
    embedding: tuple[float, ...]
    sample_count: int
    updated_at: str
    embedding_version: int = _EMBEDDING_VERSION
    average_duration_ms: int = 0

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["embedding"] = list(self.embedding)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "VoiceProfileTemplate | None":
        embedding = payload.get("embedding", ())
        if not isinstance(embedding, list) or not embedding:
            return None
        try:
            vector = tuple(float(value) for value in embedding)
        except (TypeError, ValueError):
            return None
        if len(vector) != len(_PROFILE_WEIGHTS):
            return None
        return cls(
            embedding=vector,
            sample_count=max(1, int(payload.get("sample_count", 1))),
            updated_at=str(payload.get("updated_at") or _utcnow().isoformat()),
            embedding_version=int(payload.get("embedding_version", _EMBEDDING_VERSION)),
            average_duration_ms=max(0, int(payload.get("average_duration_ms", 0))),
        )


@dataclass(frozen=True, slots=True)
class VoiceProfileSummary:
    enrolled: bool
    sample_count: int = 0
    updated_at: str | None = None
    average_duration_ms: int = 0
    store_path: str | None = None


@dataclass(frozen=True, slots=True)
class VoiceAssessment:
    status: str
    label: str
    detail: str
    confidence: float | None = None
    checked_at: str | None = None

    @property
    def should_persist(self) -> bool:
        return self.status not in {"disabled", "not_enrolled"}

    def confidence_percent(self) -> str:
        if self.confidence is None:
            return "—"
        return f"{self.confidence * 100:.0f}%"


@dataclass(frozen=True, slots=True)
class _VoiceEmbedding:
    vector: tuple[float, ...]
    duration_ms: int


class VoiceProfileStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "VoiceProfileStore":
        return cls(voice_profile_store_path(config))

    def load(self) -> VoiceProfileTemplate | None:
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        return VoiceProfileTemplate.from_dict(payload)

    def save(self, template: VoiceProfileTemplate) -> VoiceProfileTemplate:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(template.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return template

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)


class VoiceProfileMonitor:
    def __init__(
        self,
        *,
        store: VoiceProfileStore,
        likely_threshold: float,
        uncertain_threshold: float,
        max_enrollment_samples: int,
        min_sample_ms: int,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        self.store = store
        self.likely_threshold = likely_threshold
        self.uncertain_threshold = uncertain_threshold
        self.max_enrollment_samples = max(1, max_enrollment_samples)
        self.min_sample_ms = max(600, min_sample_ms)
        self.clock = clock

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "VoiceProfileMonitor":
        return cls(
            store=VoiceProfileStore.from_config(config),
            likely_threshold=config.voice_profile_likely_threshold,
            uncertain_threshold=config.voice_profile_uncertain_threshold,
            max_enrollment_samples=config.voice_profile_max_samples,
            min_sample_ms=config.voice_profile_min_sample_ms,
        )

    def summary(self) -> VoiceProfileSummary:
        template = self.store.load()
        if template is None:
            return VoiceProfileSummary(enrolled=False, store_path=str(self.store.path))
        return VoiceProfileSummary(
            enrolled=True,
            sample_count=template.sample_count,
            updated_at=template.updated_at,
            average_duration_ms=template.average_duration_ms,
            store_path=str(self.store.path),
        )

    def reset(self) -> VoiceProfileSummary:
        self.store.clear()
        return self.summary()

    def enroll_wav_bytes(self, wav_bytes: bytes) -> VoiceProfileTemplate:
        embedding = extract_voice_embedding_from_wav_bytes(wav_bytes, min_sample_ms=self.min_sample_ms)
        return self._merge_embedding(embedding)

    def enroll_pcm16(self, pcm_bytes: bytes, *, sample_rate: int, channels: int) -> VoiceProfileTemplate:
        embedding = extract_voice_embedding_from_pcm16(
            pcm_bytes,
            sample_rate=sample_rate,
            channels=channels,
            min_sample_ms=self.min_sample_ms,
        )
        return self._merge_embedding(embedding)

    def assess_wav_bytes(self, wav_bytes: bytes) -> VoiceAssessment:
        template = self.store.load()
        if template is None:
            return VoiceAssessment(status="not_enrolled", label="Not enrolled", detail="No local voice profile is stored yet.")
        embedding = extract_voice_embedding_from_wav_bytes(wav_bytes, min_sample_ms=self.min_sample_ms)
        return self._assessment_for_embedding(template, embedding)

    def assess_pcm16(self, pcm_bytes: bytes, *, sample_rate: int, channels: int) -> VoiceAssessment:
        template = self.store.load()
        if template is None:
            return VoiceAssessment(status="not_enrolled", label="Not enrolled", detail="No local voice profile is stored yet.")
        embedding = extract_voice_embedding_from_pcm16(
            pcm_bytes,
            sample_rate=sample_rate,
            channels=channels,
            min_sample_ms=self.min_sample_ms,
        )
        return self._assessment_for_embedding(template, embedding)

    def _merge_embedding(self, embedding: _VoiceEmbedding) -> VoiceProfileTemplate:
        existing = self.store.load()
        now_iso = self.clock().isoformat()
        if existing is None:
            template = VoiceProfileTemplate(
                embedding=embedding.vector,
                sample_count=1,
                updated_at=now_iso,
                average_duration_ms=embedding.duration_ms,
            )
            return self.store.save(template)

        previous_weight = min(existing.sample_count, self.max_enrollment_samples - 1)
        combined_weight = previous_weight + 1
        merged_vector = tuple(
            ((old_value * previous_weight) + new_value) / combined_weight
            for old_value, new_value in zip(existing.embedding, embedding.vector, strict=True)
        )
        average_duration_ms = int(round(((existing.average_duration_ms * previous_weight) + embedding.duration_ms) / combined_weight))
        template = VoiceProfileTemplate(
            embedding=merged_vector,
            sample_count=min(existing.sample_count + 1, self.max_enrollment_samples),
            updated_at=now_iso,
            average_duration_ms=average_duration_ms,
        )
        return self.store.save(template)

    def _assessment_for_embedding(self, template: VoiceProfileTemplate, embedding: _VoiceEmbedding) -> VoiceAssessment:
        distance = _weighted_distance(template.embedding, embedding.vector)
        confidence = max(0.0, min(1.0, math.exp(-4.0 * distance)))
        checked_at = self.clock().isoformat()
        if confidence >= self.likely_threshold:
            return VoiceAssessment(
                status="likely_user",
                label="Likely user",
                detail="The current speech sample is close to the enrolled local voice profile.",
                confidence=confidence,
                checked_at=checked_at,
            )
        if confidence >= self.uncertain_threshold:
            return VoiceAssessment(
                status="uncertain",
                label="Uncertain",
                detail="The current speech sample is only a partial match. Keep explicit confirmation for sensitive actions.",
                confidence=confidence,
                checked_at=checked_at,
            )
        return VoiceAssessment(
            status="unknown_voice",
            label="Unknown voice",
            detail="The current speech sample does not match the enrolled local voice profile closely enough.",
            confidence=confidence,
            checked_at=checked_at,
        )


def extract_voice_embedding_from_wav_bytes(wav_bytes: bytes, *, min_sample_ms: int) -> _VoiceEmbedding:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        if sample_width != 2:
            raise ValueError("Voice profiling currently expects 16-bit WAV input.")
        pcm_bytes = wav_file.readframes(wav_file.getnframes())
    return extract_voice_embedding_from_pcm16(
        pcm_bytes,
        sample_rate=sample_rate,
        channels=channels,
        min_sample_ms=min_sample_ms,
    )


def extract_voice_embedding_from_pcm16(
    pcm_bytes: bytes,
    *,
    sample_rate: int,
    channels: int,
    min_sample_ms: int,
) -> _VoiceEmbedding:
    samples = _mono_samples(pcm_bytes, channels=channels)
    if sample_rate <= 0:
        raise ValueError("Sample rate must be greater than zero.")
    duration_ms = int((len(samples) * 1000) / sample_rate)
    if duration_ms < min_sample_ms:
        raise ValueError("Need a longer speech sample for voice profiling.")
    frame_size = max(32, int(sample_rate * (_FRAME_MS / 1000.0)))
    hop_size = max(16, int(sample_rate * (_HOP_MS / 1000.0)))
    frames = _frame_windows(samples, frame_size=frame_size, hop_size=hop_size)
    if not frames:
        raise ValueError("Speech sample is too short for voice profiling.")

    rms_values = tuple(_frame_rms(frame) for frame in frames)
    peak_rms = max(rms_values)
    active_threshold = max(0.015, peak_rms * 0.35)
    active_frames = [frame for frame, rms in zip(frames, rms_values, strict=True) if rms >= active_threshold]
    if len(active_frames) < 6:
        raise ValueError("Need more steady speech for voice profiling.")

    zcr_values = [_zero_crossing_rate(frame) for frame in active_frames]
    delta_values = [_average_absolute_delta(frame) for frame in active_frames]
    active_ratio = len(active_frames) / max(1, len(frames))
    mean_rms = sum(_frame_rms(frame) for frame in active_frames) / max(1, len(active_frames))

    pitch_values = [pitch for pitch in (_estimate_pitch_hz(frame, sample_rate) for frame in active_frames) if pitch is not None]
    voiced_ratio = len(pitch_values) / max(1, len(active_frames))
    mean_pitch = (sum(pitch_values) / len(pitch_values)) if pitch_values else 0.0
    pitch_std = _stddev(pitch_values, mean_pitch) if pitch_values else 0.0

    band_totals = [0.0 for _ in _BAND_FREQUENCIES_HZ]
    for frame in active_frames:
        for index, frequency_hz in enumerate(_BAND_FREQUENCIES_HZ):
            band_totals[index] += _goertzel_power(frame, sample_rate, frequency_hz)
    total_band_power = sum(band_totals) or 1.0
    band_ratios = tuple(power / total_band_power for power in band_totals)

    vector = (
        round(mean_rms, 6),
        round(active_ratio, 6),
        round(sum(zcr_values) / len(zcr_values), 6),
        round(_stddev(zcr_values), 6),
        round(sum(delta_values) / len(delta_values), 6),
        round(_stddev(delta_values), 6),
        round(voiced_ratio, 6),
        round(min(mean_pitch / _MAX_PITCH_HZ, 1.5), 6),
        round(min(pitch_std / 200.0, 1.5), 6),
        *(round(value, 6) for value in band_ratios),
    )
    return _VoiceEmbedding(vector=vector, duration_ms=duration_ms)


def _mono_samples(pcm_bytes: bytes, *, channels: int) -> tuple[float, ...]:
    if channels <= 0:
        raise ValueError("Channel count must be greater than zero.")
    samples = array("h")
    samples.frombytes(pcm_bytes)
    if sys.byteorder != "little":
        samples.byteswap()
    if channels == 1:
        return tuple(sample / 32768.0 for sample in samples)

    mono: list[float] = []
    stride = channels
    sample_count = len(samples) - (len(samples) % stride)
    for index in range(0, sample_count, stride):
        mono.append(sum(samples[index : index + stride]) / (stride * 32768.0))
    return tuple(mono)


def _frame_windows(samples: tuple[float, ...], *, frame_size: int, hop_size: int) -> tuple[tuple[float, ...], ...]:
    if len(samples) < frame_size:
        return ()
    return tuple(
        tuple(samples[offset : offset + frame_size])
        for offset in range(0, len(samples) - frame_size + 1, hop_size)
    )


def _frame_rms(frame: tuple[float, ...]) -> float:
    return math.sqrt(sum(sample * sample for sample in frame) / max(1, len(frame)))


def _zero_crossing_rate(frame: tuple[float, ...]) -> float:
    crossings = 0
    for left, right in zip(frame, frame[1:], strict=False):
        if (left >= 0 > right) or (left < 0 <= right):
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
        left = frame[:-lag]
        right = frame[lag:]
        numerator = sum(a * b for a, b in zip(left, right, strict=True))
        left_energy = sum(a * a for a in left)
        right_energy = sum(b * b for b in right)
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
    bin_index = int(0.5 + ((len(frame) * frequency_hz) / sample_rate))
    omega = (2.0 * math.pi * bin_index) / len(frame)
    coefficient = 2.0 * math.cos(omega)
    prev = 0.0
    prev2 = 0.0
    for sample in frame:
        current = sample + (coefficient * prev) - prev2
        prev2 = prev
        prev = current
    return max(0.0, (prev2 * prev2) + (prev * prev) - (coefficient * prev * prev2))


def _stddev(values: list[float] | tuple[float, ...], mean: float | None = None) -> float:
    if not values:
        return 0.0
    effective_mean = (sum(values) / len(values)) if mean is None else mean
    variance = sum((value - effective_mean) ** 2 for value in values) / len(values)
    return math.sqrt(max(0.0, variance))


def _weighted_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    total_weight = sum(_PROFILE_WEIGHTS)
    weighted_error = sum(
        weight * abs(left_value - right_value)
        for weight, left_value, right_value in zip(_PROFILE_WEIGHTS, left, right, strict=True)
    )
    return weighted_error / max(0.0001, total_weight)
