from __future__ import annotations

from array import array
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import fcntl  # AUDIT-FIX(#2): Use cross-process file locking for the file-backed store on Linux/RPi.
import io
import json
import math
import os  # AUDIT-FIX(#1): Use low-level no-follow, atomic replace, and fsync file operations.
from pathlib import Path
import stat  # AUDIT-FIX(#2): Validate file types and reject unsafe store path components.
import sys
import tempfile  # AUDIT-FIX(#2): Write via temp files before atomic replace to avoid torn JSON.
import threading  # AUDIT-FIX(#2): Prevent same-process lost updates across concurrent enroll/reset calls.
from typing import Callable
import wave

from twinr.agent.base_agent.config import TwinrConfig

_FRAME_MS = 30
_HOP_MS = 15
_MIN_PITCH_HZ = 85.0
_MAX_PITCH_HZ = 350.0
_BAND_FREQUENCIES_HZ = (250.0, 500.0, 1000.0, 1800.0, 2800.0)
_EMBEDDING_VERSION = 1
# AUDIT-FIX(#5): Bound per-sample work on the RPi and expose a safe default for optional config overrides.
_DEFAULT_MAX_SAMPLE_MS = 30_000
# AUDIT-FIX(#5): Clamp any override so a bad config cannot force arbitrarily long CPU-heavy profiling runs.
_MAX_ALLOWED_SAMPLE_MS = 60_000
# AUDIT-FIX(#2): The persisted JSON file is tiny; reject unexpectedly large files to fail closed on tampering/corruption.
_MAX_STORE_BYTES = 64 * 1024
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


# AUDIT-FIX(#7): Normalize all internally generated timestamps to UTC-aware datetimes.
def _normalize_datetime_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


# AUDIT-FIX(#7): Generate stable ISO-8601 UTC timestamps even when a caller injects a naive clock.
def _utc_iso(value: datetime | None = None) -> str:
    effective = _utcnow() if value is None else value
    return _normalize_datetime_utc(effective).isoformat()


# AUDIT-FIX(#3): Reject non-finite vectors because NaN/Infinity can poison scoring and produce false accepts.
def _is_finite_vector(values: tuple[float, ...]) -> bool:
    return all(math.isfinite(value) for value in values)


# AUDIT-FIX(#3): Parse persisted integer fields strictly to avoid silent coercions and crashy malformed state.
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
                numeric = int(stripped, 10)
            except ValueError:
                result = None
            else:
                result = numeric

    if result is None:
        return default
    return max(minimum, result)


# AUDIT-FIX(#1): Parse optional config flags safely so strings like "false" cannot be treated as truthy.
def _coerce_bool_like(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return default


# AUDIT-FIX(#6): Clamp thresholds and numeric config to safe ranges so bad .env values fail closed instead of misclassifying.
def _coerce_bounded_float(value: object, *, default: float, minimum: float, maximum: float) -> float:
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        candidate = default
    if not math.isfinite(candidate):
        candidate = default
    return min(maximum, max(minimum, candidate))


# AUDIT-FIX(#6): Enforce likely >= uncertain to prevent inverted threshold configs from accepting the wrong speaker.
def _normalize_thresholds(likely_threshold: object, uncertain_threshold: object) -> tuple[float, float]:
    likely = _coerce_bounded_float(likely_threshold, default=0.82, minimum=0.0, maximum=1.0)
    uncertain = _coerce_bounded_float(uncertain_threshold, default=0.58, minimum=0.0, maximum=1.0)
    if likely < uncertain:
        likely, uncertain = uncertain, likely
    return likely, uncertain


# AUDIT-FIX(#1): Constrain the store to project_root by default and require an explicit opt-in for external absolute paths.
def voice_profile_store_path(config: TwinrConfig) -> Path:
    project_root = Path(config.project_root).expanduser().resolve(strict=False)
    configured_raw = str(config.voice_profile_store_path or "").strip()
    if not configured_raw:
        raise ValueError("voice_profile_store_path must point to a JSON file.")

    configured = Path(configured_raw).expanduser()
    allow_external_store = _coerce_bool_like(getattr(config, "voice_profile_allow_external_store", False), default=False)
    try:
        candidate = configured.resolve(strict=False) if configured.is_absolute() else (project_root / configured).resolve(strict=False)
    except OSError as exc:
        raise ValueError("voice_profile_store_path could not be resolved safely.") from exc

    if candidate.name in {"", ".", ".."}:
        raise ValueError("voice_profile_store_path must point to a file, not a directory.")

    if not allow_external_store and not candidate.is_relative_to(project_root):
        raise ValueError("voice_profile_store_path must stay inside project_root unless voice_profile_allow_external_store=true.")

    return candidate


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
        # AUDIT-FIX(#3): Reject malformed, non-finite, or wrong-version payloads instead of crashing or scoring garbage.
        if len(vector) != len(_PROFILE_WEIGHTS) or not _is_finite_vector(vector):
            return None

        embedding_version = _coerce_int_like(payload.get("embedding_version", _EMBEDDING_VERSION), minimum=1, default=None)
        if embedding_version != _EMBEDDING_VERSION:
            return None

        sample_count = _coerce_int_like(payload.get("sample_count", 1), minimum=1, default=None)
        average_duration_ms = _coerce_int_like(payload.get("average_duration_ms", 0), minimum=0, default=None)
        if sample_count is None or average_duration_ms is None:
            return None

        updated_at_raw = payload.get("updated_at")
        updated_at = updated_at_raw.strip() if isinstance(updated_at_raw, str) and updated_at_raw.strip() else _utc_iso()

        return cls(
            embedding=vector,
            sample_count=sample_count,
            updated_at=updated_at,
            embedding_version=embedding_version,
            average_duration_ms=min(average_duration_ms, _MAX_ALLOWED_SAMPLE_MS),
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
        # AUDIT-FIX(#4): Invalid audio must not be persisted as if it were a real speaker verification result.
        return self.status not in {"disabled", "not_enrolled", "invalid_sample"}

    def confidence_percent(self) -> str:
        if self.confidence is None:
            return "—"
        return f"{self.confidence * 100:.0f}%"


@dataclass(frozen=True, slots=True)
class _VoiceEmbedding:
    vector: tuple[float, ...]
    duration_ms: int


# AUDIT-FIX(#2): Serialize every file-store mutation under a process lock plus a Linux file lock.
class VoiceProfileStore:
    def __init__(self, path: str | Path) -> None:
        # AUDIT-FIX(#1): Canonicalize to an absolute path without resolving the final filesystem entry through symlinks.
        self.path = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
        self._process_lock = threading.RLock()

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "VoiceProfileStore":
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

    # AUDIT-FIX(#2): Provide an atomic read-modify-write transaction so concurrent enrollments do not lose samples.
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
                with os.fdopen(lock_fd, "a+", encoding="utf-8") as lock_file:
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

    # AUDIT-FIX(#2): Reject symlinked or non-directory parents before creating temp files in them.
    def _ensure_safe_parent_directory(self, *, create: bool) -> None:
        if not self.path.is_absolute():
            raise OSError("Voice profile store path must be absolute.")
        current = Path(self.path.anchor)
        for part in self.path.parts[1:-1]:
            current = current / part
            if current.exists():
                metadata = os.lstat(current)
                if stat.S_ISLNK(metadata.st_mode):
                    raise OSError(f"Unsafe symlinked parent directory for voice profile store: {current}")
                if not stat.S_ISDIR(metadata.st_mode):
                    raise OSError(f"Voice profile store parent is not a directory: {current}")
                continue
            if not create:
                raise FileNotFoundError(current)
            try:
                current.mkdir(mode=0o700)
            except FileExistsError:
                pass
            metadata = os.lstat(current)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise OSError(f"Voice profile store parent became unsafe during creation: {current}")

    def _open_lock_file(self) -> int:
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        return os.open(self._lock_path(), flags, 0o600)

    def _lock_path(self) -> str:
        return str(self.path.with_name(f".{self.path.name}.lock"))

    # AUDIT-FIX(#2): Read via O_NOFOLLOW and a regular-file check so symlinks and device nodes are ignored.
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
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(self.path, flags)
        try:
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode):
                raise OSError("Voice profile store must be a regular file.")
            if metadata.st_size > _MAX_STORE_BYTES:
                raise OSError("Voice profile store is unexpectedly large.")
            with os.fdopen(fd, "r", encoding="utf-8") as store_file:
                return store_file.read(_MAX_STORE_BYTES + 1)
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            raise

    # AUDIT-FIX(#2): Persist with fsynced temp-file writes plus atomic replace to survive power loss and crashes.
    def _write_unlocked(self, template: VoiceProfileTemplate) -> None:
        payload_text = json.dumps(template.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
        payload_bytes = payload_text.encode("utf-8")
        if len(payload_bytes) > _MAX_STORE_BYTES:
            raise ValueError("Voice profile store payload is unexpectedly large.")

        temp_fd, temp_path = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent))
        try:
            os.fchmod(temp_fd, 0o600)
            with os.fdopen(temp_fd, "wb") as temp_file:
                temp_file.write(payload_bytes)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            try:
                existing_metadata = os.lstat(self.path)
            except FileNotFoundError:
                existing_metadata = None
            if existing_metadata is not None and stat.S_ISDIR(existing_metadata.st_mode):
                raise OSError(f"Voice profile store target is a directory: {self.path}")
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
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        directory_fd = os.open(directory, flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    # AUDIT-FIX(#3): Validate outgoing templates too so invalid in-memory state never reaches disk.
    def _validate_template(self, template: VoiceProfileTemplate) -> None:
        if len(template.embedding) != len(_PROFILE_WEIGHTS):
            raise ValueError("Voice profile embedding has the wrong length.")
        if not _is_finite_vector(template.embedding):
            raise ValueError("Voice profile embedding contains non-finite values.")
        if template.embedding_version != _EMBEDDING_VERSION:
            raise ValueError("Voice profile embedding version is not supported.")
        if template.sample_count < 1:
            raise ValueError("Voice profile sample_count must be at least 1.")
        if template.average_duration_ms < 0 or template.average_duration_ms > _MAX_ALLOWED_SAMPLE_MS:
            raise ValueError("Voice profile average_duration_ms is outside the supported range.")


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
        max_sample_ms: int = _DEFAULT_MAX_SAMPLE_MS,
    ) -> None:
        self.store = store
        # AUDIT-FIX(#6): Normalize runtime config so a bad .env cannot silently invert speaker-classification behavior.
        self.likely_threshold, self.uncertain_threshold = _normalize_thresholds(likely_threshold, uncertain_threshold)
        self.max_enrollment_samples = _coerce_int_like(max_enrollment_samples, minimum=1, default=1) or 1
        self.min_sample_ms = min(_MAX_ALLOWED_SAMPLE_MS, max(600, _coerce_int_like(min_sample_ms, minimum=0, default=600) or 600))
        # AUDIT-FIX(#5): Bound the maximum accepted sample duration with a backward-compatible optional config default.
        configured_max_sample_ms = _coerce_int_like(max_sample_ms, minimum=1000, default=_DEFAULT_MAX_SAMPLE_MS) or _DEFAULT_MAX_SAMPLE_MS
        self.max_sample_ms = max(self.min_sample_ms, min(_MAX_ALLOWED_SAMPLE_MS, configured_max_sample_ms))
        self.clock = clock

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "VoiceProfileMonitor":
        return cls(
            store=VoiceProfileStore.from_config(config),
            likely_threshold=config.voice_profile_likely_threshold,
            uncertain_threshold=config.voice_profile_uncertain_threshold,
            max_enrollment_samples=config.voice_profile_max_samples,
            min_sample_ms=config.voice_profile_min_sample_ms,
            max_sample_ms=getattr(config, "voice_profile_max_sample_ms", _DEFAULT_MAX_SAMPLE_MS),
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
        embedding = extract_voice_embedding_from_wav_bytes(
            wav_bytes,
            min_sample_ms=self.min_sample_ms,
            max_sample_ms=self.max_sample_ms,
        )
        return self._merge_embedding(embedding)

    def enroll_pcm16(self, pcm_bytes: bytes, *, sample_rate: int, channels: int) -> VoiceProfileTemplate:
        embedding = extract_voice_embedding_from_pcm16(
            pcm_bytes,
            sample_rate=sample_rate,
            channels=channels,
            min_sample_ms=self.min_sample_ms,
            max_sample_ms=self.max_sample_ms,
        )
        return self._merge_embedding(embedding)

    # AUDIT-FIX(#4): Invalid assessment audio now degrades to a retryable assessment instead of bubbling a 500.
    def assess_wav_bytes(self, wav_bytes: bytes) -> VoiceAssessment:
        template = self.store.load()
        if template is None:
            return VoiceAssessment(
                status="not_enrolled",
                label="Not enrolled",
                detail="No saved voice profile is available on this device yet.",
            )
        try:
            embedding = extract_voice_embedding_from_wav_bytes(
                wav_bytes,
                min_sample_ms=self.min_sample_ms,
                max_sample_ms=self.max_sample_ms,
            )
        except ValueError as exc:
            return self._invalid_assessment(str(exc))
        return self._assessment_for_embedding(template, embedding)

    def assess_pcm16(self, pcm_bytes: bytes, *, sample_rate: int, channels: int) -> VoiceAssessment:
        template = self.store.load()
        if template is None:
            return VoiceAssessment(
                status="not_enrolled",
                label="Not enrolled",
                detail="No saved voice profile is available on this device yet.",
            )
        try:
            embedding = extract_voice_embedding_from_pcm16(
                pcm_bytes,
                sample_rate=sample_rate,
                channels=channels,
                min_sample_ms=self.min_sample_ms,
                max_sample_ms=self.max_sample_ms,
            )
        except ValueError as exc:
            return self._invalid_assessment(str(exc))
        return self._assessment_for_embedding(template, embedding)

    # AUDIT-FIX(#2): Merge enrollment samples inside a single locked store transaction to prevent lost updates.
    def _merge_embedding(self, embedding: _VoiceEmbedding) -> VoiceProfileTemplate:
        now_iso = _utc_iso(self.clock())

        def updater(existing: VoiceProfileTemplate | None) -> VoiceProfileTemplate:
            if existing is None:
                return VoiceProfileTemplate(
                    embedding=embedding.vector,
                    sample_count=1,
                    updated_at=now_iso,
                    average_duration_ms=embedding.duration_ms,
                )

            previous_weight = min(existing.sample_count, self.max_enrollment_samples - 1)
            combined_weight = previous_weight + 1
            merged_vector = tuple(
                ((old_value * previous_weight) + new_value) / combined_weight
                for old_value, new_value in zip(existing.embedding, embedding.vector, strict=True)
            )
            average_duration_ms = int(
                round(((existing.average_duration_ms * previous_weight) + embedding.duration_ms) / combined_weight)
            )
            return VoiceProfileTemplate(
                embedding=merged_vector,
                sample_count=min(existing.sample_count + 1, self.max_enrollment_samples),
                updated_at=now_iso,
                average_duration_ms=average_duration_ms,
            )

        return self.store.update(updater)

    # AUDIT-FIX(#8): Convert low-level audio-validation failures into plain-language retry guidance for seniors.
    def _invalid_assessment(self, reason: str) -> VoiceAssessment:
        detail = f"{_friendly_audio_error(reason)} Please ask the speaker to try again in a quiet room."
        return VoiceAssessment(
            status="invalid_sample",
            label="Could not verify",
            detail=detail,
            checked_at=_utc_iso(self.clock()),
        )

    def _assessment_for_embedding(self, template: VoiceProfileTemplate, embedding: _VoiceEmbedding) -> VoiceAssessment:
        try:
            distance = _weighted_distance(template.embedding, embedding.vector)
        except ValueError:
            return self._invalid_assessment("Voice match could not be calculated safely.")
        confidence = max(0.0, min(1.0, math.exp(-4.0 * distance)))
        checked_at = _utc_iso(self.clock())
        if confidence >= self.likely_threshold:
            return VoiceAssessment(
                status="likely_user",
                label="Likely user",
                detail="This voice sounds like the saved voice profile.",
                confidence=confidence,
                checked_at=checked_at,
            )
        if confidence >= self.uncertain_threshold:
            return VoiceAssessment(
                status="uncertain",
                label="Uncertain",
                detail="This might be the saved voice. Ask before doing anything important.",
                confidence=confidence,
                checked_at=checked_at,
            )
        return VoiceAssessment(
            status="unknown_voice",
            label="Unknown voice",
            detail="This voice does not sound like the saved voice profile.",
            confidence=confidence,
            checked_at=checked_at,
        )


# AUDIT-FIX(#8): Map technical audio parser errors to short, non-jargon user-facing guidance.
def _friendly_audio_error(reason: str) -> str:
    normalized = reason.strip().lower()
    if "too long" in normalized:
        return "That recording is longer than needed."
    if "too short" in normalized or "longer speech sample" in normalized or "steady speech" in normalized:
        return "I need a little more clear speech."
    if "quiet room" in normalized:
        return reason.strip()
    return "I could not use that recording."


def extract_voice_embedding_from_wav_bytes(
    wav_bytes: bytes,
    *,
    min_sample_ms: int,
    max_sample_ms: int = _DEFAULT_MAX_SAMPLE_MS,
) -> _VoiceEmbedding:
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_count = wav_file.getnframes()
            if sample_width != 2:
                raise ValueError("Voice profiling currently expects 16-bit WAV input.")
            if sample_rate <= 0:
                raise ValueError("Sample rate must be greater than zero.")
            if channels <= 0:
                raise ValueError("Channel count must be greater than zero.")
            # AUDIT-FIX(#5): Reject oversized recordings before reading all frames into memory.
            duration_ms = int((frame_count * 1000) / sample_rate)
            if duration_ms > max_sample_ms:
                raise ValueError(f"Speech sample is too long for voice profiling. Please keep it under {max_sample_ms // 1000} seconds.")
            pcm_bytes = wav_file.readframes(frame_count)
    except (EOFError, wave.Error) as exc:
        # AUDIT-FIX(#4): Normalize decoder failures to ValueError so callers can handle bad uploads safely.
        raise ValueError("Audio input is not a valid WAV recording.") from exc

    return extract_voice_embedding_from_pcm16(
        pcm_bytes,
        sample_rate=sample_rate,
        channels=channels,
        min_sample_ms=min_sample_ms,
        max_sample_ms=max_sample_ms,
    )


def extract_voice_embedding_from_pcm16(
    pcm_bytes: bytes,
    *,
    sample_rate: int,
    channels: int,
    min_sample_ms: int,
    max_sample_ms: int = _DEFAULT_MAX_SAMPLE_MS,
) -> _VoiceEmbedding:
    sample_rate_value = _coerce_int_like(sample_rate, minimum=1, default=None)
    if sample_rate_value is None:
        raise ValueError("Sample rate must be greater than zero.")

    channels_value = _coerce_int_like(channels, minimum=1, default=None)
    if channels_value is None:
        raise ValueError("Channel count must be greater than zero.")

    min_sample_ms_value = min(_MAX_ALLOWED_SAMPLE_MS, _coerce_int_like(min_sample_ms, minimum=0, default=0) or 0)
    configured_max_sample_ms = _coerce_int_like(max_sample_ms, minimum=1, default=_DEFAULT_MAX_SAMPLE_MS) or _DEFAULT_MAX_SAMPLE_MS
    max_sample_ms_value = max(
        max(min_sample_ms_value, 1),
        min(_MAX_ALLOWED_SAMPLE_MS, configured_max_sample_ms),
    )

    # AUDIT-FIX(#5): Bound raw PCM size before materializing large tuples to avoid RPi CPU/RAM blowups.
    max_pcm_bytes = int(math.ceil(sample_rate_value * channels_value * 2 * (max_sample_ms_value / 1000.0)))
    if len(pcm_bytes) > max_pcm_bytes:
        raise ValueError(f"Speech sample is too long for voice profiling. Please keep it under {max_sample_ms_value // 1000} seconds.")

    samples = _mono_samples(pcm_bytes, channels=channels_value)
    duration_ms = int((len(samples) * 1000) / sample_rate_value)
    if duration_ms < min_sample_ms_value:
        raise ValueError("Need a longer speech sample for voice profiling.")
    frame_size = max(32, int(sample_rate_value * (_FRAME_MS / 1000.0)))
    hop_size = max(16, int(sample_rate_value * (_HOP_MS / 1000.0)))
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

    pitch_values = [pitch for pitch in (_estimate_pitch_hz(frame, sample_rate_value) for frame in active_frames) if pitch is not None]
    voiced_ratio = len(pitch_values) / max(1, len(active_frames))
    mean_pitch = (sum(pitch_values) / len(pitch_values)) if pitch_values else 0.0
    pitch_std = _stddev(pitch_values, mean_pitch) if pitch_values else 0.0

    band_totals = [0.0 for _ in _BAND_FREQUENCIES_HZ]
    for frame in active_frames:
        for index, frequency_hz in enumerate(_BAND_FREQUENCIES_HZ):
            band_totals[index] += _goertzel_power(frame, sample_rate_value, frequency_hz)
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
    # AUDIT-FIX(#5): Reject truncated PCM instead of silently dropping tail bytes or partial channel frames.
    if len(pcm_bytes) % 2 != 0:
        raise ValueError("PCM input must contain complete 16-bit samples.")
    samples = array("h")
    try:
        samples.frombytes(pcm_bytes)
    except ValueError as exc:
        raise ValueError("PCM input must contain complete 16-bit samples.") from exc
    if sys.byteorder != "little":
        samples.byteswap()
    if len(samples) % channels != 0:
        raise ValueError("PCM input must contain whole frames for every channel.")
    if channels == 1:
        return tuple(sample / 32768.0 for sample in samples)

    mono: list[float] = []
    stride = channels
    for index in range(0, len(samples), stride):
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
    # AUDIT-FIX(#3): Fail closed on malformed or non-finite vectors before confidence scoring.
    if len(left) != len(_PROFILE_WEIGHTS) or len(right) != len(_PROFILE_WEIGHTS):
        raise ValueError("Voice profile embeddings must match the expected length.")
    if not _is_finite_vector(left) or not _is_finite_vector(right):
        raise ValueError("Voice profile embeddings must be finite.")
    total_weight = sum(_PROFILE_WEIGHTS)
    weighted_error = sum(
        weight * abs(left_value - right_value)
        for weight, left_value, right_value in zip(_PROFILE_WEIGHTS, left, right, strict=True)
    )
    return weighted_error / max(0.0001, total_weight)