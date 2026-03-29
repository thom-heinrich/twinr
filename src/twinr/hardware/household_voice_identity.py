# CHANGELOG: 2026-03-28
# BUG-1: Handle embedding schema/dimension changes without crashing enrollment or misclassifying users; incompatible stores now trigger re-enrollment paths.
# BUG-2: Reject non-finite embeddings and invalid JSON writes so one bad sample cannot silently brick the whole store.
# BUG-3: Add backup recovery plus directory fsync after atomic replace to reduce profile loss on Raspberry Pi power cuts.
# SEC-1: Enforce private permissions on store, backup, and lock files and use symlink-safe opens for local biometric data.
# SEC-2: Stop silently treating incompatible/corrupt profiles as ordinary "not enrolled" state during assessment.
# IMP-1: Upgrade from centroid-only enrollment to bounded exemplar history with normalized centroid recomputation.
# IMP-2: Add duration-aware and uncertainty-aware scoring with dynamic ambiguity margins for short/noisy household turns.
# IMP-3: Add duplicate-profile recovery, legacy v1 migration, embedding-schema metadata, and in-process snapshot caching.

"""Persist and assess multi-user local voice identities on device.

This module extends the legacy single-profile ``voice_profile`` path with a
secure, versioned, bounded multi-user store and matcher that can identify
enrolled household members from the current turn audio.

2026 upgrade highlights:
- secure private file permissions, atomic writes, and backup recovery;
- embedding-schema compatibility checks to survive model upgrades cleanly;
- bounded exemplar history per user to better model intra-speaker variability;
- duration-aware and uncertainty-aware scoring that stays CPU-cheap on a Pi 4.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
import fcntl
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import stat
import tempfile
import threading

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.voice_profile import (
    extract_voice_embedding_from_pcm16,
    extract_voice_embedding_from_wav_bytes,
    voice_embedding_confidence,
)


logger = logging.getLogger(__name__)

_STORE_VERSION = 2
_DEFAULT_STORE_PATH = "state/household_voice_identities.json"
_DEFAULT_IDENTITY_MARGIN = 0.06
_DEFAULT_MIN_SAMPLE_MS = 1200
_MAX_STORE_BYTES = 256 * 1024
_PRIVATE_FILE_MODE = 0o600
_PRIVATE_DIR_MODE = 0o700
_MAX_SCORE_PENALTY = 0.08


def _utc_iso(value: datetime | None = None) -> str:
    moment = datetime.now(UTC) if value is None else value
    if moment.tzinfo is None or moment.utcoffset() is None:
        moment = moment.replace(tzinfo=UTC)
    else:
        moment = moment.astimezone(UTC)
    return moment.isoformat()


def _parse_utc_iso_sort_key(value: str | None) -> tuple[int, str]:
    text = _normalize_text(value).strip()
    if not text:
        return (0, "")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return (0, text)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=UTC)
    else:
        parsed = parsed.astimezone(UTC)
    return (int(parsed.timestamp()), text)


def _normalize_text(value: object | None) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _normalize_user_id(value: object | None, *, default: str = "main_user") -> str:
    raw = _normalize_text(value or default).strip().lower()
    normalized: list[str] = []
    previous_separator = False
    for char in raw:
        if char.isalnum():
            normalized.append(char)
            previous_separator = False
            continue
        if char in {"-", "_", " "}:
            if not previous_separator:
                normalized.append("_")
                previous_separator = True
    text = "".join(normalized).strip("_")
    return text or default


def _normalize_display_name(value: object | None) -> str | None:
    text = _normalize_text(value).strip()
    return text or None


def _normalize_schema_id(value: object | None) -> str | None:
    text = _normalize_text(value).strip()
    return text or None


def _resolve_store_path(config: TwinrConfig) -> Path:
    project_root = Path(config.project_root).expanduser().resolve(strict=False)
    raw_value = getattr(config, "household_voice_identity_store_path", None)
    text = str(raw_value or "").strip() or _DEFAULT_STORE_PATH
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    return (project_root / candidate).resolve(strict=False)


def _resolve_embedding_schema(config: TwinrConfig) -> str | None:
    for attribute in (
        "household_voice_embedding_schema",
        "voice_profile_embedding_schema",
        "voice_profile_model_id",
        "voice_embedding_model_id",
    ):
        value = getattr(config, attribute, None)
        normalized = _normalize_schema_id(value)
        if normalized:
            return normalized
    return None


def _resolve_primary_user_id(config: TwinrConfig) -> str:
    value = (
        getattr(config, "household_voice_primary_user_id", None)
        or getattr(config, "portrait_match_primary_user_id", None)
        or "main_user"
    )
    return _normalize_user_id(value, default="main_user")


def _coerce_positive_int(value: object | None, *, default: int) -> int:
    if value is None or isinstance(value, bool):
        return default
    if not isinstance(value, (int, float, str, bytes, bytearray)):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def _coerce_non_negative_int(value: object | None, *, default: int) -> int:
    if value is None or isinstance(value, bool):
        return default
    if not isinstance(value, (int, float, str, bytes, bytearray)):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, parsed)


def _coerce_ratio(value: object | None, *, default: float) -> float:
    if value is None or isinstance(value, bool):
        return default
    if not isinstance(value, (int, float, str, bytes, bytearray)):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return max(0.0, min(1.0, parsed))


def _coerce_positive_float(
    value: object | None,
    *,
    default: float,
    minimum: float = 0.05,
    maximum: float = 32.0,
) -> float:
    if value is None or isinstance(value, bool):
        return default
    if not isinstance(value, (int, float, str, bytes, bytearray)):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return max(minimum, min(maximum, parsed))


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _normalize_embedding(
    value: tuple[float, ...] | list[float] | object,
    *,
    expected_dim: int | None = None,
) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError("voice_embedding_missing")
    try:
        embedding = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError("voice_embedding_invalid") from exc
    if not all(math.isfinite(item) for item in embedding):
        raise ValueError("voice_embedding_non_finite")
    if expected_dim is not None and len(embedding) != expected_dim:
        raise ValueError("voice_embedding_dim_mismatch")
    norm = math.sqrt(sum(item * item for item in embedding))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("voice_embedding_zero_norm")
    return tuple(item / norm for item in embedding)


def _duration_weight(duration_ms: int, *, min_sample_ms: int) -> float:
    baseline = max(400, min_sample_ms)
    ratio = max(0.5, min(2.5, float(max(1, duration_ms)) / float(baseline)))
    return _clamp(math.sqrt(ratio), 0.7, 1.35)


def _duration_penalty(duration_ms: int | None, *, min_sample_ms: int) -> float:
    if duration_ms is None:
        return 0.0
    floor = max(300, min_sample_ms)
    full_confidence_ms = max(floor * 2, 2400)
    bounded = max(0, duration_ms)
    if bounded >= full_confidence_ms:
        return 0.0
    ratio = bounded / float(full_confidence_ms)
    return _clamp((1.0 - ratio) * 0.05, 0.0, 0.05)


def _ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, _PRIVATE_DIR_MODE)
    except OSError:
        logger.debug("Could not chmod private directory %s", path, exc_info=True)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _path_signature(path: Path) -> tuple[int, int, int] | None:
    try:
        metadata = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(metadata.st_mode):
        return None
    return (metadata.st_ino, metadata.st_mtime_ns, metadata.st_size)


@dataclass(frozen=True, slots=True)
class HouseholdVoiceEnrollmentSample:
    """Persist one bounded enrollment sample used to build a user prototype."""

    embedding: tuple[float, ...]
    duration_ms: int
    recorded_at: str
    weight: float = 1.0

    def to_dict(self) -> dict[str, object]:
        return {
            "embedding": list(self.embedding),
            "duration_ms": self.duration_ms,
            "recorded_at": self.recorded_at,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object],
        *,
        expected_dim: int | None = None,
    ) -> "HouseholdVoiceEnrollmentSample | None":
        try:
            embedding = _normalize_embedding(payload.get("embedding", ()), expected_dim=expected_dim)
        except ValueError:
            return None
        return cls(
            embedding=embedding,
            duration_ms=_coerce_non_negative_int(payload.get("duration_ms"), default=0),
            recorded_at=_normalize_text(payload.get("recorded_at")).strip() or _utc_iso(),
            weight=_coerce_positive_float(payload.get("weight"), default=1.0),
        )


@dataclass(frozen=True, slots=True)
class HouseholdVoiceProfile:
    """Persist one enrolled household voice template."""

    user_id: str
    display_name: str | None
    primary_user: bool
    embedding: tuple[float, ...]
    sample_count: int
    average_duration_ms: int
    updated_at: str
    enrollment_samples: tuple[HouseholdVoiceEnrollmentSample, ...] = ()
    embedding_schema: str | None = None

    @property
    def embedding_dim(self) -> int:
        return len(self.embedding)

    def to_dict(self) -> dict[str, object]:
        return {
            "user_id": self.user_id,
            "display_name": self.display_name,
            "primary_user": self.primary_user,
            "embedding": list(self.embedding),
            "sample_count": self.sample_count,
            "average_duration_ms": self.average_duration_ms,
            "updated_at": self.updated_at,
            "embedding_schema": self.embedding_schema,
            "enrollment_samples": [sample.to_dict() for sample in self.enrollment_samples],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "HouseholdVoiceProfile | None":
        try:
            embedding = _normalize_embedding(payload.get("embedding", ()))
        except ValueError:
            return None
        sample_count = _coerce_positive_int(payload.get("sample_count"), default=1)
        average_duration_ms = _coerce_non_negative_int(
            payload.get("average_duration_ms"),
            default=0,
        )
        updated_at = _normalize_text(payload.get("updated_at")).strip() or _utc_iso()
        enrollment_samples_raw = payload.get("enrollment_samples", ())
        enrollment_samples: list[HouseholdVoiceEnrollmentSample] = []
        if isinstance(enrollment_samples_raw, list):
            for item in enrollment_samples_raw:
                if not isinstance(item, dict):
                    return None
                sample = HouseholdVoiceEnrollmentSample.from_dict(
                    item,
                    expected_dim=len(embedding),
                )
                if sample is None:
                    return None
                enrollment_samples.append(sample)
        if not enrollment_samples:
            # Legacy v1 compatibility: synthesize one weighted sample from the centroid.
            enrollment_samples = [
                HouseholdVoiceEnrollmentSample(
                    embedding=embedding,
                    duration_ms=average_duration_ms,
                    recorded_at=updated_at,
                    weight=float(sample_count),
                )
            ]
        return cls(
            user_id=_normalize_user_id(payload.get("user_id")),
            display_name=_normalize_display_name(payload.get("display_name")),
            primary_user=bool(payload.get("primary_user")),
            embedding=embedding,
            sample_count=sample_count,
            average_duration_ms=average_duration_ms,
            updated_at=updated_at,
            enrollment_samples=tuple(enrollment_samples),
            embedding_schema=_normalize_schema_id(payload.get("embedding_schema")),
        )


@dataclass(frozen=True, slots=True)
class HouseholdVoiceSummary:
    """Summarize one enrolled household voice identity."""

    user_id: str
    display_name: str | None
    primary_user: bool
    enrolled: bool
    sample_count: int
    average_duration_ms: int
    updated_at: str | None
    store_path: str


@dataclass(frozen=True, slots=True)
class HouseholdVoiceAssessment:
    """Describe one local voice match against enrolled household members."""

    status: str
    label: str
    detail: str
    confidence: float | None = None
    checked_at: str | None = None
    matched_user_id: str | None = None
    matched_user_display_name: str | None = None
    candidate_user_count: int = 0

    @property
    def should_persist(self) -> bool:
        return self.status not in {
            "not_enrolled",
            "invalid_sample",
            "re_enrollment_required",
        }


@dataclass(frozen=True, slots=True)
class _VoiceCandidate:
    user_id: str
    display_name: str | None
    primary_user: bool
    confidence: float
    centroid_confidence: float
    best_sample_confidence: float
    sample_count: int
    dispersion_penalty: float
    duration_penalty: float


def household_voice_profiles_revision(
    profiles: tuple[HouseholdVoiceProfile, ...],
) -> str:
    """Return one stable revision token for a profile snapshot."""

    digest = hashlib.sha256()
    for profile in sorted(profiles, key=lambda item: item.user_id):
        encoded = json.dumps(
            profile.to_dict(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        digest.update(encoded)
    return digest.hexdigest()[:16]


def _profile_centroid_from_samples(
    samples: tuple[HouseholdVoiceEnrollmentSample, ...] | list[HouseholdVoiceEnrollmentSample],
    *,
    expected_dim: int | None = None,
) -> tuple[float, ...]:
    items = tuple(samples)
    if not items:
        raise ValueError("voice_profile_samples_missing")
    dimension = expected_dim or len(items[0].embedding)
    weighted = [0.0] * dimension
    total_weight = 0.0
    for sample in items:
        embedding = _normalize_embedding(sample.embedding, expected_dim=dimension)
        weight = _coerce_positive_float(sample.weight, default=1.0)
        total_weight += weight
        for index, value in enumerate(embedding):
            weighted[index] += value * weight
    if total_weight <= 0.0:
        raise ValueError("voice_profile_samples_zero_weight")
    return _normalize_embedding(tuple(weighted), expected_dim=dimension)


def _profile_dispersion_penalty(
    profile: HouseholdVoiceProfile,
) -> float:
    if len(profile.enrollment_samples) < 2:
        return 0.0
    similarities: list[float] = []
    for sample in profile.enrollment_samples:
        try:
            similarities.append(voice_embedding_confidence(profile.embedding, sample.embedding))
        except ValueError:
            continue
    if not similarities:
        return 0.0
    mean_similarity = sum(similarities) / len(similarities)
    return _clamp((1.0 - mean_similarity) * 0.12, 0.0, 0.03)


def _score_profile_candidate(
    profile: HouseholdVoiceProfile,
    embedding: tuple[float, ...],
    *,
    sample_duration_ms: int | None,
    min_sample_ms: int,
) -> _VoiceCandidate | None:
    expected_dim = profile.embedding_dim
    if len(embedding) != expected_dim:
        return None
    try:
        centroid_confidence = float(voice_embedding_confidence(profile.embedding, embedding))
    except ValueError:
        return None

    best_sample_confidence = centroid_confidence
    if profile.enrollment_samples:
        exemplar_scores: list[float] = []
        for sample in profile.enrollment_samples:
            try:
                exemplar_scores.append(float(voice_embedding_confidence(sample.embedding, embedding)))
            except ValueError:
                continue
        if exemplar_scores:
            exemplar_scores.sort(reverse=True)
            top_k = exemplar_scores[: min(2, len(exemplar_scores))]
            best_sample_confidence = exemplar_scores[0]
            support_confidence = sum(top_k) / len(top_k)
        else:
            support_confidence = centroid_confidence
    else:
        support_confidence = centroid_confidence

    dispersion_penalty = _profile_dispersion_penalty(profile)
    duration_penalty = _duration_penalty(sample_duration_ms, min_sample_ms=min_sample_ms)
    fused_confidence = max(centroid_confidence, support_confidence)
    confidence = _clamp(
        fused_confidence - dispersion_penalty - duration_penalty,
        0.0,
        1.0,
    )
    return _VoiceCandidate(
        user_id=profile.user_id,
        display_name=profile.display_name,
        primary_user=profile.primary_user,
        confidence=confidence,
        centroid_confidence=centroid_confidence,
        best_sample_confidence=best_sample_confidence,
        sample_count=max(1, profile.sample_count),
        dispersion_penalty=dispersion_penalty,
        duration_penalty=duration_penalty,
    )


def assess_household_voice_embedding(
    embedding: tuple[float, ...],
    *,
    checked_at: str | None,
    profiles: tuple[HouseholdVoiceProfile, ...],
    primary_user_id: str,
    likely_threshold: float,
    uncertain_threshold: float,
    identity_margin: float,
    sample_duration_ms: int | None = None,
    min_sample_ms: int = _DEFAULT_MIN_SAMPLE_MS,
) -> HouseholdVoiceAssessment:
    """Score one embedding against the enrolled household voices."""

    if not profiles:
        return HouseholdVoiceAssessment(
            status="not_enrolled",
            label="Not enrolled",
            detail="No enrolled household voice identity is available on this device yet.",
        )
    try:
        normalized_embedding = _normalize_embedding(embedding)
    except ValueError:
        return HouseholdVoiceAssessment(
            status="invalid_sample",
            label="Could not verify",
            detail="The current voice sample produced an invalid embedding and could not be verified.",
            checked_at=_normalize_text(checked_at).strip() or _utc_iso(),
        )

    normalized_primary_user_id = _normalize_user_id(primary_user_id)
    normalized_likely_threshold = _coerce_ratio(likely_threshold, default=0.72)
    normalized_uncertain_threshold = _coerce_ratio(uncertain_threshold, default=0.55)
    if normalized_likely_threshold < normalized_uncertain_threshold:
        normalized_likely_threshold, normalized_uncertain_threshold = (
            normalized_uncertain_threshold,
            normalized_likely_threshold,
        )
    normalized_identity_margin = _coerce_ratio(
        identity_margin,
        default=_DEFAULT_IDENTITY_MARGIN,
    )
    normalized_min_sample_ms = _coerce_positive_int(min_sample_ms, default=_DEFAULT_MIN_SAMPLE_MS)
    resolved_checked_at = _normalize_text(checked_at).strip() or _utc_iso()

    candidates: list[_VoiceCandidate] = []
    incompatible_candidate_count = 0
    for profile in profiles:
        candidate = _score_profile_candidate(
            profile,
            normalized_embedding,
            sample_duration_ms=sample_duration_ms,
            min_sample_ms=normalized_min_sample_ms,
        )
        if candidate is None:
            incompatible_candidate_count += 1
            continue
        candidates.append(
            _VoiceCandidate(
                user_id=profile.user_id,
                display_name=profile.display_name,
                primary_user=(
                    profile.primary_user or profile.user_id == normalized_primary_user_id
                ),
                confidence=candidate.confidence,
                centroid_confidence=candidate.centroid_confidence,
                best_sample_confidence=candidate.best_sample_confidence,
                sample_count=candidate.sample_count,
                dispersion_penalty=candidate.dispersion_penalty,
                duration_penalty=candidate.duration_penalty,
            )
        )

    if not candidates:
        if incompatible_candidate_count:
            # BREAKING: assessment callers may now receive "re_enrollment_required"
            # when stored profiles are incompatible with the active embedding model.
            return HouseholdVoiceAssessment(
                status="re_enrollment_required",
                label="Re-enrollment required",
                detail=(
                    "The enrolled household voice profiles are incompatible with the current "
                    "voice embedding schema. Please re-enroll household voices on this device."
                ),
                checked_at=resolved_checked_at,
            )
        return HouseholdVoiceAssessment(
            status="invalid_sample",
            label="Could not verify",
            detail="The current voice sample could not be compared to the enrolled household profiles.",
            checked_at=resolved_checked_at,
        )

    candidates.sort(key=lambda item: item.confidence, reverse=True)
    best = candidates[0]
    second = None if len(candidates) < 2 else candidates[1]
    dynamic_margin = normalized_identity_margin + best.duration_penalty + best.dispersion_penalty
    ambiguous = (
        second is not None
        and best.confidence >= normalized_uncertain_threshold
        and second.confidence >= normalized_uncertain_threshold
        and (best.confidence - second.confidence) < dynamic_margin
    )
    if ambiguous:
        return HouseholdVoiceAssessment(
            status="ambiguous_match",
            label="Ambiguous household match",
            detail="The current voice could match more than one enrolled household member.",
            confidence=best.confidence,
            checked_at=resolved_checked_at,
            matched_user_id=best.user_id,
            matched_user_display_name=best.display_name,
            candidate_user_count=len(candidates),
        )
    if best.confidence >= normalized_likely_threshold:
        return HouseholdVoiceAssessment(
            status="likely_user" if best.primary_user else "known_other_user",
            label="Enrolled household user",
            detail=(
                "This voice sounds like the enrolled main user."
                if best.primary_user
                else "This voice sounds like another enrolled household user."
            ),
            confidence=best.confidence,
            checked_at=resolved_checked_at,
            matched_user_id=best.user_id,
            matched_user_display_name=best.display_name,
            candidate_user_count=len(candidates),
        )
    if best.confidence >= normalized_uncertain_threshold:
        return HouseholdVoiceAssessment(
            status="uncertain_match",
            label="Uncertain household match",
            detail="This voice may belong to an enrolled household member, but the match is still uncertain.",
            confidence=best.confidence,
            checked_at=resolved_checked_at,
            matched_user_id=best.user_id,
            matched_user_display_name=best.display_name,
            candidate_user_count=len(candidates),
        )
    return HouseholdVoiceAssessment(
        status="unknown_voice",
        label="Unknown voice",
        detail="This voice does not match an enrolled household voice identity confidently.",
        confidence=best.confidence,
        checked_at=resolved_checked_at,
        candidate_user_count=len(candidates),
    )


def assess_household_voice_pcm16(
    pcm_bytes: bytes,
    *,
    sample_rate: int,
    channels: int,
    checked_at: str | None,
    profiles: tuple[HouseholdVoiceProfile, ...],
    primary_user_id: str,
    likely_threshold: float,
    uncertain_threshold: float,
    identity_margin: float,
    min_sample_ms: int,
) -> HouseholdVoiceAssessment:
    """Extract one embedding from PCM16 audio and score it against the household."""

    if not profiles:
        return HouseholdVoiceAssessment(
            status="not_enrolled",
            label="Not enrolled",
            detail="No enrolled household voice identity is available on this device yet.",
        )
    try:
        embedding = extract_voice_embedding_from_pcm16(
            pcm_bytes,
            sample_rate=sample_rate,
            channels=channels,
            min_sample_ms=min_sample_ms,
        )
    except ValueError as exc:
        return HouseholdVoiceAssessment(
            status="invalid_sample",
            label="Could not verify",
            detail=f"{str(exc).strip()} Please ask the speaker to try again in a quiet room.",
            checked_at=_normalize_text(checked_at).strip() or _utc_iso(),
        )
    return assess_household_voice_embedding(
        embedding.vector,
        checked_at=checked_at,
        profiles=profiles,
        primary_user_id=primary_user_id,
        likely_threshold=likely_threshold,
        uncertain_threshold=uncertain_threshold,
        identity_margin=identity_margin,
        sample_duration_ms=getattr(embedding, "duration_ms", None),
        min_sample_ms=min_sample_ms,
    )


class HouseholdVoiceIdentityStore:
    """Persist multi-user voice templates in one bounded atomic JSON file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
        self.backup_path = self.path.with_suffix(f"{self.path.suffix}.bak")
        self._process_lock = threading.RLock()
        self._cache_lock = threading.RLock()
        self._cached_profiles: tuple[HouseholdVoiceProfile, ...] = ()
        self._cached_signature: tuple[int, int, int] | None = None

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "HouseholdVoiceIdentityStore":
        return cls(_resolve_store_path(config))

    def list_profiles(self) -> tuple[HouseholdVoiceProfile, ...]:
        signature = _path_signature(self.path)
        with self._cache_lock:
            if signature == self._cached_signature:
                return self._cached_profiles

        parent = self.path.parent
        if not parent.exists():
            with self._cache_lock:
                self._cached_profiles = ()
                self._cached_signature = None
            return ()

        try:
            with self._exclusive_lock(create_parent=False):
                profiles, loaded_from_path = self._load_with_recovery_unlocked()
                with self._cache_lock:
                    self._cached_profiles = profiles
                    self._cached_signature = _path_signature(self.path) if loaded_from_path == self.path else None
                return profiles
        except OSError:
            logger.warning(
                "Failed to read household voice store from %s",
                self.path,
                exc_info=True,
            )
            with self._cache_lock:
                if signature == self._cached_signature:
                    return self._cached_profiles
            return ()

    def load_profile(self, user_id: str) -> HouseholdVoiceProfile | None:
        normalized_user_id = _normalize_user_id(user_id)
        for profile in self.list_profiles():
            if profile.user_id == normalized_user_id:
                return profile
        return None

    def summary(self, user_id: str) -> HouseholdVoiceSummary:
        normalized_user_id = _normalize_user_id(user_id)
        profile = self.load_profile(normalized_user_id)
        if profile is None:
            return HouseholdVoiceSummary(
                user_id=normalized_user_id,
                display_name=None,
                primary_user=False,
                enrolled=False,
                sample_count=0,
                average_duration_ms=0,
                updated_at=None,
                store_path=str(self.path),
            )
        return HouseholdVoiceSummary(
            user_id=profile.user_id,
            display_name=profile.display_name,
            primary_user=profile.primary_user,
            enrolled=True,
            sample_count=profile.sample_count,
            average_duration_ms=profile.average_duration_ms,
            updated_at=profile.updated_at,
            store_path=str(self.path),
        )

    def upsert_profile(
        self,
        *,
        user_id: str,
        display_name: str | None,
        primary_user: bool,
        embedding: tuple[float, ...],
        duration_ms: int,
        max_samples: int,
        embedding_schema: str | None = None,
        min_sample_ms: int = _DEFAULT_MIN_SAMPLE_MS,
    ) -> HouseholdVoiceProfile:
        normalized_user_id = _normalize_user_id(user_id)
        normalized_display_name = _normalize_display_name(display_name)
        normalized_embedding = _normalize_embedding(embedding)
        normalized_duration_ms = _coerce_non_negative_int(duration_ms, default=0)
        normalized_max_samples = _coerce_positive_int(max_samples, default=6)
        normalized_schema = _normalize_schema_id(embedding_schema)
        normalized_min_sample_ms = _coerce_positive_int(min_sample_ms, default=_DEFAULT_MIN_SAMPLE_MS)

        new_sample = HouseholdVoiceEnrollmentSample(
            embedding=normalized_embedding,
            duration_ms=normalized_duration_ms,
            recorded_at=_utc_iso(),
            weight=_duration_weight(normalized_duration_ms, min_sample_ms=normalized_min_sample_ms),
        )

        with self._exclusive_lock(create_parent=True):
            profiles = list(self._load_with_recovery_unlocked()[0])
            index = self._profile_index(profiles, normalized_user_id)
            now_iso = _utc_iso()
            if index is None:
                enrollment_samples = (new_sample,)
                profile = HouseholdVoiceProfile(
                    user_id=normalized_user_id,
                    display_name=normalized_display_name,
                    primary_user=bool(primary_user),
                    embedding=_profile_centroid_from_samples(enrollment_samples),
                    sample_count=1,
                    average_duration_ms=normalized_duration_ms,
                    updated_at=now_iso,
                    enrollment_samples=enrollment_samples,
                    embedding_schema=normalized_schema,
                )
                profiles.append(profile)
            else:
                existing = profiles[index]
                incompatible_schema = (
                    existing.embedding_schema
                    and normalized_schema
                    and existing.embedding_schema != normalized_schema
                )
                incompatible_dim = existing.embedding_dim != len(normalized_embedding)
                if incompatible_schema or incompatible_dim:
                    # BREAKING: existing incompatible enrollments are reset on model/schema changes.
                    enrollment_samples = (new_sample,)
                    profile = HouseholdVoiceProfile(
                        user_id=existing.user_id,
                        display_name=normalized_display_name or existing.display_name,
                        primary_user=bool(primary_user or existing.primary_user),
                        embedding=_profile_centroid_from_samples(enrollment_samples),
                        sample_count=1,
                        average_duration_ms=normalized_duration_ms,
                        updated_at=now_iso,
                        enrollment_samples=enrollment_samples,
                        embedding_schema=normalized_schema or existing.embedding_schema,
                    )
                else:
                    enrollment_samples = tuple(
                        sorted(
                            (*existing.enrollment_samples, new_sample),
                            key=lambda item: _parse_utc_iso_sort_key(item.recorded_at),
                        )[-normalized_max_samples:]
                    )
                    centroid = _profile_centroid_from_samples(
                        enrollment_samples,
                        expected_dim=existing.embedding_dim,
                    )
                    total_weight = sum(sample.weight for sample in enrollment_samples) or 1.0
                    average_duration_ms = int(
                        round(
                            sum(sample.duration_ms * sample.weight for sample in enrollment_samples)
                            / total_weight
                        )
                    )
                    profile = HouseholdVoiceProfile(
                        user_id=existing.user_id,
                        display_name=normalized_display_name or existing.display_name,
                        primary_user=bool(primary_user or existing.primary_user),
                        embedding=centroid,
                        sample_count=min(existing.sample_count + 1, normalized_max_samples),
                        average_duration_ms=average_duration_ms,
                        updated_at=now_iso,
                        enrollment_samples=enrollment_samples,
                        embedding_schema=normalized_schema or existing.embedding_schema,
                    )
                profiles[index] = profile
            self._write_unlocked(profiles)
            return profile

    def clear_profile(self, *, user_id: str) -> bool:
        normalized_user_id = _normalize_user_id(user_id)
        parent = self.path.parent
        if not parent.exists():
            return False
        try:
            with self._exclusive_lock(create_parent=False):
                profiles = list(self._load_with_recovery_unlocked()[0])
                index = self._profile_index(profiles, normalized_user_id)
                if index is None:
                    return False
                del profiles[index]
                self._write_unlocked(profiles)
                return True
        except OSError:
            logger.warning(
                "Failed to clear household voice profile %s from %s",
                normalized_user_id,
                self.path,
                exc_info=True,
            )
            return False

    def _profile_index(self, profiles: list[HouseholdVoiceProfile], user_id: str) -> int | None:
        for index, profile in enumerate(profiles):
            if profile.user_id == user_id:
                return index
        return None

    @contextmanager
    def _exclusive_lock(self, *, create_parent: bool) -> Iterator[None]:
        parent = self.path.parent
        if create_parent:
            _ensure_private_directory(parent)
        elif not parent.exists():
            raise FileNotFoundError(parent)
        lock_path = parent / f".{self.path.name}.lock"
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        with self._process_lock:
            fd = os.open(lock_path, flags, _PRIVATE_FILE_MODE)
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise OSError("household_voice_lock_not_regular")
                with os.fdopen(fd, "a+", encoding="utf-8") as handle:
                    try:
                        os.chmod(lock_path, _PRIVATE_FILE_MODE)
                    except OSError:
                        logger.debug("Could not chmod lock file %s", lock_path, exc_info=True)
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                    try:
                        yield
                    finally:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                try:
                    os.close(fd)
                except OSError:
                    pass
                raise

    def _load_with_recovery_unlocked(self) -> tuple[tuple[HouseholdVoiceProfile, ...], Path | None]:
        last_error: Exception | None = None
        for candidate_path in (self.path, self.backup_path):
            if not candidate_path.exists():
                continue
            try:
                payload = self._read_json_unlocked(candidate_path)
                profiles = self._parse_store_payload(payload)
                if candidate_path == self.backup_path:
                    logger.warning(
                        "Recovered household voice profiles from backup store %s",
                        self.backup_path,
                    )
                return (profiles, candidate_path)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Failed to read household voice store candidate %s",
                    candidate_path,
                    exc_info=True,
                )
        if last_error is not None:
            raise OSError("household_voice_store_unreadable") from last_error
        return ((), None)

    def _read_json_unlocked(self, path: Path) -> dict[str, object]:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags)
        try:
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode):
                raise OSError("household_voice_store_not_regular")
            if metadata.st_size > _MAX_STORE_BYTES:
                raise OSError("household_voice_store_too_large")
            with os.fdopen(fd, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            raise
        if not isinstance(payload, dict):
            raise ValueError("household_voice_store_invalid_payload")
        return payload

    def _parse_store_payload(self, payload: dict[str, object]) -> tuple[HouseholdVoiceProfile, ...]:
        version = int(payload.get("store_version", 1) or 1)
        if version not in {1, 2}:
            raise ValueError("household_voice_store_unsupported_version")

        profiles_raw = payload.get("profiles", ())
        if not isinstance(profiles_raw, list):
            raise ValueError("household_voice_store_invalid_profiles")

        by_user_id: dict[str, HouseholdVoiceProfile] = {}
        for item in profiles_raw:
            if not isinstance(item, dict):
                raise ValueError("household_voice_store_invalid_profile")
            profile = HouseholdVoiceProfile.from_dict(item)
            if profile is None:
                raise ValueError("household_voice_store_profile_parse_failed")
            existing = by_user_id.get(profile.user_id)
            if existing is None:
                by_user_id[profile.user_id] = profile
                continue
            by_user_id[profile.user_id] = self._merge_duplicate_profiles(existing, profile)

        profiles = list(by_user_id.values())
        profiles.sort(key=lambda item: (not item.primary_user, item.user_id))
        return tuple(profiles)

    def _merge_duplicate_profiles(
        self,
        left: HouseholdVoiceProfile,
        right: HouseholdVoiceProfile,
    ) -> HouseholdVoiceProfile:
        if left.embedding_dim != right.embedding_dim:
            return max((left, right), key=lambda item: _parse_utc_iso_sort_key(item.updated_at))
        merged_samples = tuple(
            sorted(
                (*left.enrollment_samples, *right.enrollment_samples),
                key=lambda item: _parse_utc_iso_sort_key(item.recorded_at),
            )[-max(left.sample_count, right.sample_count, 1):]
        )
        try:
            merged_embedding = _profile_centroid_from_samples(
                merged_samples,
                expected_dim=left.embedding_dim,
            )
        except ValueError:
            merged_embedding = left.embedding
        total_weight = sum(sample.weight for sample in merged_samples) or 1.0
        average_duration_ms = int(
            round(sum(sample.duration_ms * sample.weight for sample in merged_samples) / total_weight)
        )
        return HouseholdVoiceProfile(
            user_id=left.user_id,
            display_name=right.display_name or left.display_name,
            primary_user=left.primary_user or right.primary_user,
            embedding=merged_embedding,
            sample_count=max(left.sample_count, right.sample_count, len(merged_samples)),
            average_duration_ms=average_duration_ms,
            updated_at=max(left.updated_at, right.updated_at, key=_parse_utc_iso_sort_key),
            enrollment_samples=merged_samples,
            embedding_schema=right.embedding_schema or left.embedding_schema,
        )

    def _write_unlocked(self, profiles: list[HouseholdVoiceProfile]) -> None:
        _ensure_private_directory(self.path.parent)
        payload = {
            "store_version": _STORE_VERSION,
            "profiles": [profile.to_dict() for profile in profiles],
            "written_at": _utc_iso(),
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        if len(encoded.encode("utf-8")) > _MAX_STORE_BYTES:
            raise OSError("household_voice_store_too_large")
        self._write_text_atomic(self.path, encoded)
        try:
            self._write_text_atomic(self.backup_path, encoded)
        except OSError:
            logger.warning(
                "Failed to update household voice backup store %s",
                self.backup_path,
                exc_info=True,
            )
        with self._cache_lock:
            self._cached_profiles = tuple(sorted(profiles, key=lambda item: (not item.primary_user, item.user_id)))
            self._cached_signature = _path_signature(self.path)

    def _write_text_atomic(self, path: Path, data: str) -> None:
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            text=True,
        )
        try:
            try:
                os.fchmod(fd, _PRIVATE_FILE_MODE)
            except OSError:
                logger.debug("Could not fchmod temp store file %s", tmp_path, exc_info=True)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
            try:
                os.chmod(path, _PRIVATE_FILE_MODE)
            except OSError:
                logger.debug("Could not chmod store file %s", path, exc_info=True)
            _fsync_directory(path.parent)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass


class HouseholdVoiceIdentityMonitor:
    """Enroll and assess multi-user local voice identities."""

    def __init__(
        self,
        *,
        store: HouseholdVoiceIdentityStore,
        primary_user_id: str,
        likely_threshold: float,
        uncertain_threshold: float,
        identity_margin: float,
        min_sample_ms: int,
        max_enrollment_samples: int,
        embedding_schema: str | None = None,
    ) -> None:
        self.store = store
        self.primary_user_id = _normalize_user_id(primary_user_id)
        self.likely_threshold = _coerce_ratio(likely_threshold, default=0.72)
        self.uncertain_threshold = _coerce_ratio(uncertain_threshold, default=0.55)
        if self.likely_threshold < self.uncertain_threshold:
            self.likely_threshold, self.uncertain_threshold = self.uncertain_threshold, self.likely_threshold
        self.identity_margin = _coerce_ratio(identity_margin, default=_DEFAULT_IDENTITY_MARGIN)
        self.min_sample_ms = _coerce_positive_int(min_sample_ms, default=_DEFAULT_MIN_SAMPLE_MS)
        self.max_enrollment_samples = _coerce_positive_int(max_enrollment_samples, default=6)
        self.embedding_schema = _normalize_schema_id(embedding_schema)

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "HouseholdVoiceIdentityMonitor":
        return cls(
            store=HouseholdVoiceIdentityStore.from_config(config),
            primary_user_id=_resolve_primary_user_id(config),
            likely_threshold=getattr(config, "voice_profile_likely_threshold", 0.72),
            uncertain_threshold=getattr(config, "voice_profile_uncertain_threshold", 0.55),
            identity_margin=getattr(config, "household_voice_identity_margin", _DEFAULT_IDENTITY_MARGIN),
            min_sample_ms=getattr(config, "voice_profile_min_sample_ms", _DEFAULT_MIN_SAMPLE_MS),
            max_enrollment_samples=getattr(config, "voice_profile_max_samples", 6),
            embedding_schema=_resolve_embedding_schema(config),
        )

    def list_summaries(self) -> tuple[HouseholdVoiceSummary, ...]:
        return tuple(
            HouseholdVoiceSummary(
                user_id=profile.user_id,
                display_name=profile.display_name,
                primary_user=profile.primary_user,
                enrolled=True,
                sample_count=profile.sample_count,
                average_duration_ms=profile.average_duration_ms,
                updated_at=profile.updated_at,
                store_path=str(self.store.path),
            )
            for profile in self.store.list_profiles()
        )

    def voice_profiles(self) -> tuple[HouseholdVoiceProfile, ...]:
        """Return the current enrolled profile snapshot."""

        return self.store.list_profiles()

    def profile_revision(self) -> str:
        """Return one stable revision token for the current profile snapshot."""

        return household_voice_profiles_revision(self.voice_profiles())

    def summary(self, user_id: str) -> HouseholdVoiceSummary:
        return self.store.summary(user_id)

    def reset(self, *, user_id: str) -> HouseholdVoiceSummary:
        self.store.clear_profile(user_id=user_id)
        return self.summary(user_id)

    def enroll_pcm16(
        self,
        pcm_bytes: bytes,
        *,
        sample_rate: int,
        channels: int,
        user_id: str,
        display_name: str | None = None,
    ) -> HouseholdVoiceSummary:
        embedding = extract_voice_embedding_from_pcm16(
            pcm_bytes,
            sample_rate=sample_rate,
            channels=channels,
            min_sample_ms=self.min_sample_ms,
        )
        profile = self.store.upsert_profile(
            user_id=user_id,
            display_name=display_name,
            primary_user=_normalize_user_id(user_id) == self.primary_user_id,
            embedding=embedding.vector,
            duration_ms=embedding.duration_ms,
            max_samples=self.max_enrollment_samples,
            embedding_schema=self.embedding_schema,
            min_sample_ms=self.min_sample_ms,
        )
        return HouseholdVoiceSummary(
            user_id=profile.user_id,
            display_name=profile.display_name,
            primary_user=profile.primary_user,
            enrolled=True,
            sample_count=profile.sample_count,
            average_duration_ms=profile.average_duration_ms,
            updated_at=profile.updated_at,
            store_path=str(self.store.path),
        )

    def enroll_wav_bytes(
        self,
        wav_bytes: bytes,
        *,
        user_id: str,
        display_name: str | None = None,
    ) -> HouseholdVoiceSummary:
        embedding = extract_voice_embedding_from_wav_bytes(
            wav_bytes,
            min_sample_ms=self.min_sample_ms,
        )
        profile = self.store.upsert_profile(
            user_id=user_id,
            display_name=display_name,
            primary_user=_normalize_user_id(user_id) == self.primary_user_id,
            embedding=embedding.vector,
            duration_ms=embedding.duration_ms,
            max_samples=self.max_enrollment_samples,
            embedding_schema=self.embedding_schema,
            min_sample_ms=self.min_sample_ms,
        )
        return HouseholdVoiceSummary(
            user_id=profile.user_id,
            display_name=profile.display_name,
            primary_user=profile.primary_user,
            enrolled=True,
            sample_count=profile.sample_count,
            average_duration_ms=profile.average_duration_ms,
            updated_at=profile.updated_at,
            store_path=str(self.store.path),
        )

    def assess_pcm16(
        self,
        pcm_bytes: bytes,
        *,
        sample_rate: int,
        channels: int,
    ) -> HouseholdVoiceAssessment:
        return assess_household_voice_pcm16(
            pcm_bytes,
            sample_rate=sample_rate,
            channels=channels,
            checked_at=_utc_iso(),
            profiles=self.voice_profiles(),
            primary_user_id=self.primary_user_id,
            likely_threshold=self.likely_threshold,
            uncertain_threshold=self.uncertain_threshold,
            identity_margin=self.identity_margin,
            min_sample_ms=self.min_sample_ms,
        )

    def assess_wav_bytes(self, wav_bytes: bytes) -> HouseholdVoiceAssessment:
        profiles = self.store.list_profiles()
        if not profiles:
            return HouseholdVoiceAssessment(
                status="not_enrolled",
                label="Not enrolled",
                detail="No enrolled household voice identity is available on this device yet.",
            )
        try:
            embedding = extract_voice_embedding_from_wav_bytes(
                wav_bytes,
                min_sample_ms=self.min_sample_ms,
            )
        except ValueError as exc:
            return HouseholdVoiceAssessment(
                status="invalid_sample",
                label="Could not verify",
                detail=f"{str(exc).strip()} Please ask the speaker to try again in a quiet room.",
                checked_at=_utc_iso(),
            )
        return self._assessment_for_embedding(
            embedding.vector,
            checked_at=_utc_iso(),
            profiles=profiles,
            sample_duration_ms=getattr(embedding, "duration_ms", None),
        )

    def _assessment_for_embedding(
        self,
        embedding: tuple[float, ...],
        *,
        checked_at: str,
        profiles: tuple[HouseholdVoiceProfile, ...],
        sample_duration_ms: int | None = None,
    ) -> HouseholdVoiceAssessment:
        return assess_household_voice_embedding(
            embedding,
            checked_at=checked_at,
            profiles=profiles,
            primary_user_id=self.primary_user_id,
            likely_threshold=self.likely_threshold,
            uncertain_threshold=self.uncertain_threshold,
            identity_margin=self.identity_margin,
            sample_duration_ms=sample_duration_ms,
            min_sample_ms=self.min_sample_ms,
        )


__all__ = [
    "assess_household_voice_embedding",
    "assess_household_voice_pcm16",
    "household_voice_profiles_revision",
    "HouseholdVoiceAssessment",
    "HouseholdVoiceEnrollmentSample",
    "HouseholdVoiceIdentityMonitor",
    "HouseholdVoiceIdentityStore",
    "HouseholdVoiceProfile",
    "HouseholdVoiceSummary",
]