"""Persist and assess multi-user local voice identities on device.

This module complements the legacy single-profile ``voice_profile`` path with a
small multi-user store and matcher that can identify enrolled household members
from the current turn audio. It intentionally stays local and bounded: one
embedding per user, capped enrollment history, and conservative confidence
states for ambiguous or weak matches.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
import fcntl
import hashlib
import json
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


_STORE_VERSION = 1
_DEFAULT_STORE_PATH = "state/household_voice_identities.json"
_DEFAULT_IDENTITY_MARGIN = 0.06
_MAX_STORE_BYTES = 256 * 1024


def _utc_iso(value: datetime | None = None) -> str:
    moment = datetime.now(UTC) if value is None else value
    if moment.tzinfo is None or moment.utcoffset() is None:
        moment = moment.replace(tzinfo=UTC)
    else:
        moment = moment.astimezone(UTC)
    return moment.isoformat()


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


def _resolve_store_path(config: TwinrConfig) -> Path:
    project_root = Path(config.project_root).expanduser().resolve(strict=False)
    raw_value = getattr(config, "household_voice_identity_store_path", None)
    text = str(raw_value or "").strip() or _DEFAULT_STORE_PATH
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    return (project_root / candidate).resolve(strict=False)


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

    def to_dict(self) -> dict[str, object]:
        return {
            "user_id": self.user_id,
            "display_name": self.display_name,
            "primary_user": self.primary_user,
            "embedding": list(self.embedding),
            "sample_count": self.sample_count,
            "average_duration_ms": self.average_duration_ms,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "HouseholdVoiceProfile | None":
        embedding_raw = payload.get("embedding", ())
        if not isinstance(embedding_raw, list) or not embedding_raw:
            return None
        try:
            embedding = tuple(float(value) for value in embedding_raw)
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in embedding):
            return None
        sample_count = _coerce_positive_int(payload.get("sample_count"), default=1)
        average_duration_ms = _coerce_non_negative_int(
            payload.get("average_duration_ms"),
            default=0,
        )
        return cls(
            user_id=_normalize_user_id(payload.get("user_id")),
            display_name=_normalize_display_name(payload.get("display_name")),
            primary_user=bool(payload.get("primary_user")),
            embedding=embedding,
            sample_count=sample_count,
            average_duration_ms=average_duration_ms,
            updated_at=_normalize_text(payload.get("updated_at")).strip() or _utc_iso(),
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
        return self.status not in {"not_enrolled", "invalid_sample"}


@dataclass(frozen=True, slots=True)
class _VoiceCandidate:
    user_id: str
    display_name: str | None
    primary_user: bool
    confidence: float


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
        ).encode("utf-8")
        digest.update(encoded)
    return digest.hexdigest()[:16]


def assess_household_voice_embedding(
    embedding: tuple[float, ...],
    *,
    checked_at: str | None,
    profiles: tuple[HouseholdVoiceProfile, ...],
    primary_user_id: str,
    likely_threshold: float,
    uncertain_threshold: float,
    identity_margin: float,
) -> HouseholdVoiceAssessment:
    """Score one embedding against the enrolled household voices."""

    if not profiles:
        return HouseholdVoiceAssessment(
            status="not_enrolled",
            label="Not enrolled",
            detail="No enrolled household voice identity is available on this device yet.",
        )
    normalized_primary_user_id = _normalize_user_id(primary_user_id)
    normalized_likely_threshold = _coerce_ratio(likely_threshold, default=0.72)
    normalized_uncertain_threshold = _coerce_ratio(uncertain_threshold, default=0.55)
    if normalized_likely_threshold < normalized_uncertain_threshold:
        normalized_likely_threshold, normalized_uncertain_threshold = (
            normalized_uncertain_threshold,
            normalized_likely_threshold,
        )
    normalized_identity_margin = _coerce_ratio(identity_margin, default=_DEFAULT_IDENTITY_MARGIN)
    resolved_checked_at = _normalize_text(checked_at).strip() or _utc_iso()

    candidates: list[_VoiceCandidate] = []
    for profile in profiles:
        try:
            confidence = voice_embedding_confidence(profile.embedding, embedding)
        except ValueError:
            continue
        candidates.append(
            _VoiceCandidate(
                user_id=profile.user_id,
                display_name=profile.display_name,
                primary_user=(
                    profile.primary_user or profile.user_id == normalized_primary_user_id
                ),
                confidence=confidence,
            )
        )
    if not candidates:
        return HouseholdVoiceAssessment(
            status="invalid_sample",
            label="Could not verify",
            detail="The current voice sample could not be compared to the enrolled household profiles.",
            checked_at=resolved_checked_at,
        )
    candidates.sort(key=lambda item: item.confidence, reverse=True)
    best = candidates[0]
    second = None if len(candidates) < 2 else candidates[1]
    ambiguous = (
        second is not None
        and best.confidence >= normalized_uncertain_threshold
        and second.confidence >= normalized_uncertain_threshold
        and (best.confidence - second.confidence) < normalized_identity_margin
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
    )


class HouseholdVoiceIdentityStore:
    """Persist multi-user voice templates in one bounded atomic JSON file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
        self._process_lock = threading.RLock()

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "HouseholdVoiceIdentityStore":
        return cls(_resolve_store_path(config))

    def list_profiles(self) -> tuple[HouseholdVoiceProfile, ...]:
        parent = self.path.parent
        if not parent.exists():
            return ()
        try:
            with self._exclusive_lock(create_parent=False):
                return self._load_unlocked()
        except OSError:
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
    ) -> HouseholdVoiceProfile:
        normalized_user_id = _normalize_user_id(user_id)
        normalized_display_name = _normalize_display_name(display_name)
        max_enrollment_samples = max(1, int(max_samples))
        with self._exclusive_lock(create_parent=True):
            profiles = list(self._load_unlocked())
            index = self._profile_index(profiles, normalized_user_id)
            now_iso = _utc_iso()
            if index is None:
                profile = HouseholdVoiceProfile(
                    user_id=normalized_user_id,
                    display_name=normalized_display_name,
                    primary_user=bool(primary_user),
                    embedding=embedding,
                    sample_count=1,
                    average_duration_ms=max(0, int(duration_ms)),
                    updated_at=now_iso,
                )
                profiles.append(profile)
            else:
                existing = profiles[index]
                previous_weight = min(existing.sample_count, max_enrollment_samples - 1)
                combined_weight = previous_weight + 1
                merged_embedding = tuple(
                    ((old_value * previous_weight) + new_value) / combined_weight
                    for old_value, new_value in zip(existing.embedding, embedding, strict=True)
                )
                average_duration_ms = int(
                    round(((existing.average_duration_ms * previous_weight) + max(0, int(duration_ms))) / combined_weight)
                )
                profile = HouseholdVoiceProfile(
                    user_id=existing.user_id,
                    display_name=normalized_display_name or existing.display_name,
                    primary_user=bool(primary_user or existing.primary_user),
                    embedding=merged_embedding,
                    sample_count=min(existing.sample_count + 1, max_enrollment_samples),
                    average_duration_ms=average_duration_ms,
                    updated_at=now_iso,
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
                profiles = list(self._load_unlocked())
                index = self._profile_index(profiles, normalized_user_id)
                if index is None:
                    return False
                del profiles[index]
                self._write_unlocked(profiles)
                return True
        except OSError:
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
            parent.mkdir(parents=True, exist_ok=True)
        elif not parent.exists():
            raise FileNotFoundError(parent)
        lock_path = parent / f".{self.path.name}.lock"
        with self._process_lock:
            with open(lock_path, "a+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _load_unlocked(self) -> tuple[HouseholdVoiceProfile, ...]:
        if not self.path.exists():
            return ()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(self.path, flags)
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
            return ()
        version = int(payload.get("store_version", _STORE_VERSION) or _STORE_VERSION)
        if version != _STORE_VERSION:
            return ()
        profiles_raw = payload.get("profiles", ())
        if not isinstance(profiles_raw, list):
            return ()
        profiles: list[HouseholdVoiceProfile] = []
        for item in profiles_raw:
            if not isinstance(item, dict):
                return ()
            profile = HouseholdVoiceProfile.from_dict(item)
            if profile is None:
                return ()
            profiles.append(profile)
        profiles.sort(key=lambda item: (not item.primary_user, item.user_id))
        return tuple(profiles)

    def _write_unlocked(self, profiles: list[HouseholdVoiceProfile]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "store_version": _STORE_VERSION,
            "profiles": [profile.to_dict() for profile in profiles],
        }
        encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        fd, tmp_path = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.path)
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
    ) -> None:
        self.store = store
        self.primary_user_id = _normalize_user_id(primary_user_id)
        self.likely_threshold = _coerce_ratio(likely_threshold, default=0.72)
        self.uncertain_threshold = _coerce_ratio(uncertain_threshold, default=0.55)
        if self.likely_threshold < self.uncertain_threshold:
            self.likely_threshold, self.uncertain_threshold = self.uncertain_threshold, self.likely_threshold
        self.identity_margin = _coerce_ratio(identity_margin, default=_DEFAULT_IDENTITY_MARGIN)
        self.min_sample_ms = _coerce_positive_int(min_sample_ms, default=1200)
        self.max_enrollment_samples = _coerce_positive_int(max_enrollment_samples, default=6)

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "HouseholdVoiceIdentityMonitor":
        return cls(
            store=HouseholdVoiceIdentityStore.from_config(config),
            primary_user_id=getattr(config, "portrait_match_primary_user_id", "main_user") or "main_user",
            likely_threshold=getattr(config, "voice_profile_likely_threshold", 0.72),
            uncertain_threshold=getattr(config, "voice_profile_uncertain_threshold", 0.55),
            identity_margin=getattr(config, "household_voice_identity_margin", _DEFAULT_IDENTITY_MARGIN),
            min_sample_ms=getattr(config, "voice_profile_min_sample_ms", 1200),
            max_enrollment_samples=getattr(config, "voice_profile_max_samples", 6),
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
        return self._assessment_for_embedding(embedding.vector, checked_at=_utc_iso(), profiles=profiles)

    def _assessment_for_embedding(
        self,
        embedding: tuple[float, ...],
        *,
        checked_at: str,
        profiles: tuple[HouseholdVoiceProfile, ...],
    ) -> HouseholdVoiceAssessment:
        return assess_household_voice_embedding(
            embedding,
            checked_at=checked_at,
            profiles=profiles,
            primary_user_id=self.primary_user_id,
            likely_threshold=self.likely_threshold,
            uncertain_threshold=self.uncertain_threshold,
            identity_margin=self.identity_margin,
        )


__all__ = [
    "assess_household_voice_embedding",
    "assess_household_voice_pcm16",
    "household_voice_profiles_revision",
    "HouseholdVoiceAssessment",
    "HouseholdVoiceIdentityMonitor",
    "HouseholdVoiceIdentityStore",
    "HouseholdVoiceProfile",
    "HouseholdVoiceSummary",
]
