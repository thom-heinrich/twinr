"""Persist local portrait-identity profiles and reference images on device.

This module owns the file-backed identity enrollment state used by local
portrait matching: per-user reference images, extracted embeddings, and small
summary helpers. Live camera capture and matching policy stay in
``portrait_match.py``.
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


_ALLOWED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
_STORE_VERSION = 1
_EMBEDDING_VERSION = 1
_DEFAULT_STORE_PATH = "state/portrait_identities.json"
_DEFAULT_IMAGE_DIR = "state/portrait_identities"
_MAX_STORE_BYTES = 512 * 1024


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


def _normalize_source(value: object | None) -> str:
    text = _normalize_text(value).strip().lower()
    normalized: list[str] = []
    previous_separator = False
    for char in text:
        if char.isalnum():
            normalized.append(char)
            previous_separator = False
            continue
        if char in {"-", "_", " ", "/"}:
            if not previous_separator:
                normalized.append("_")
                previous_separator = True
    collapsed = "".join(normalized).strip("_")
    return collapsed or "manual_import"


def _normalize_embedding(values: tuple[float, ...]) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0.0 or not math.isfinite(norm):
        return values
    return tuple(round(value / norm, 8) for value in values)


def _coerce_optional_float(value: object | None) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _safe_suffix(filename_hint: object | None) -> str:
    suffix = Path(str(filename_hint or "reference.jpg")).suffix.casefold()
    if suffix in _ALLOWED_IMAGE_SUFFIXES:
        return suffix
    return ".jpg"


def _resolve_path(*, project_root: Path, raw_value: object | None, default: str) -> Path:
    text = str(raw_value or "").strip() or default
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    return (project_root / candidate).resolve(strict=False)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


@dataclass(frozen=True, slots=True)
class PortraitReferenceImage:
    """Store one enrolled reference image and embedding for a user."""

    reference_id: str
    relative_path: str
    image_sha256: str
    source: str
    added_at: str
    embedding: tuple[float, ...]
    detector_confidence: float | None = None
    embedding_version: int = _EMBEDDING_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_id": self.reference_id,
            "relative_path": self.relative_path,
            "image_sha256": self.image_sha256,
            "source": self.source,
            "added_at": self.added_at,
            "embedding": list(self.embedding),
            "detector_confidence": self.detector_confidence,
            "embedding_version": self.embedding_version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "PortraitReferenceImage | None":
        reference_id = _normalize_text(payload.get("reference_id")).strip()
        relative_path = _normalize_text(payload.get("relative_path")).strip()
        image_sha256 = _normalize_text(payload.get("image_sha256")).strip().lower()
        source = _normalize_source(payload.get("source"))
        added_at = _normalize_text(payload.get("added_at")).strip() or _utc_iso()
        embedding_raw = payload.get("embedding", ())
        if not isinstance(embedding_raw, list) or not embedding_raw:
            return None
        try:
            embedding = tuple(float(value) for value in embedding_raw)
        except (TypeError, ValueError):
            return None
        if not reference_id or not relative_path or not image_sha256:
            return None
        if not all(math.isfinite(value) for value in embedding):
            return None
        embedding_version = int(payload.get("embedding_version", _EMBEDDING_VERSION) or _EMBEDDING_VERSION)
        if embedding_version != _EMBEDDING_VERSION:
            return None
        return cls(
            reference_id=reference_id,
            relative_path=relative_path,
            image_sha256=image_sha256,
            source=source,
            added_at=added_at,
            embedding=_normalize_embedding(embedding),
            detector_confidence=_coerce_optional_float(payload.get("detector_confidence")),
            embedding_version=embedding_version,
        )


@dataclass(frozen=True, slots=True)
class PortraitIdentityProfile:
    """Describe one enrolled local portrait identity."""

    user_id: str
    display_name: str | None
    primary_user: bool
    created_at: str
    updated_at: str
    reference_images: tuple[PortraitReferenceImage, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "user_id": self.user_id,
            "display_name": self.display_name,
            "primary_user": self.primary_user,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "reference_images": [image.to_dict() for image in self.reference_images],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "PortraitIdentityProfile | None":
        user_id = _normalize_user_id(payload.get("user_id"))
        display_name = _normalize_display_name(payload.get("display_name"))
        created_at = _normalize_text(payload.get("created_at")).strip() or _utc_iso()
        updated_at = _normalize_text(payload.get("updated_at")).strip() or created_at
        images_raw = payload.get("reference_images", ())
        if not isinstance(images_raw, list):
            return None
        images: list[PortraitReferenceImage] = []
        for item in images_raw:
            if not isinstance(item, dict):
                return None
            parsed = PortraitReferenceImage.from_dict(item)
            if parsed is None:
                return None
            images.append(parsed)
        return cls(
            user_id=user_id,
            display_name=display_name,
            primary_user=bool(payload.get("primary_user")),
            created_at=created_at,
            updated_at=updated_at,
            reference_images=tuple(images),
        )


@dataclass(frozen=True, slots=True)
class PortraitIdentitySummary:
    """Summarize one enrolled portrait identity for runtime or operator callers."""

    user_id: str
    display_name: str | None
    primary_user: bool
    enrolled: bool
    reference_image_count: int
    updated_at: str | None
    store_path: str


def portrait_identity_store_path(config: TwinrConfig) -> Path:
    """Resolve the configured portrait identity store path."""

    project_root = Path(config.project_root).expanduser().resolve(strict=False)
    return _resolve_path(
        project_root=project_root,
        raw_value=getattr(config, "portrait_match_store_path", None),
        default=_DEFAULT_STORE_PATH,
    )


def portrait_identity_image_dir(config: TwinrConfig) -> Path:
    """Resolve the configured portrait reference-image directory."""

    project_root = Path(config.project_root).expanduser().resolve(strict=False)
    return _resolve_path(
        project_root=project_root,
        raw_value=getattr(config, "portrait_match_reference_image_dir", None),
        default=_DEFAULT_IMAGE_DIR,
    )


class PortraitIdentityStore:
    """Persist local portrait identity profiles in a small atomic JSON store."""

    def __init__(self, store_path: str | Path, image_dir: str | Path) -> None:
        self.path = Path(os.path.abspath(os.fspath(Path(store_path).expanduser())))
        self.image_dir = Path(os.path.abspath(os.fspath(Path(image_dir).expanduser())))
        self._process_lock = threading.RLock()

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "PortraitIdentityStore":
        return cls(
            store_path=portrait_identity_store_path(config),
            image_dir=portrait_identity_image_dir(config),
        )

    def list_profiles(self) -> tuple[PortraitIdentityProfile, ...]:
        try:
            with self._exclusive_lock(create_parent=False):
                return tuple(self._load_profiles_unlocked())
        except OSError:
            return ()

    def load_profile(self, user_id: str) -> PortraitIdentityProfile | None:
        normalized_user_id = _normalize_user_id(user_id)
        for profile in self.list_profiles():
            if profile.user_id == normalized_user_id:
                return profile
        return None

    def summary(self, user_id: str) -> PortraitIdentitySummary:
        normalized_user_id = _normalize_user_id(user_id)
        profile = self.load_profile(normalized_user_id)
        if profile is None:
            return PortraitIdentitySummary(
                user_id=normalized_user_id,
                display_name=None,
                primary_user=False,
                enrolled=False,
                reference_image_count=0,
                updated_at=None,
                store_path=str(self.path),
            )
        return PortraitIdentitySummary(
            user_id=profile.user_id,
            display_name=profile.display_name,
            primary_user=profile.primary_user,
            enrolled=bool(profile.reference_images),
            reference_image_count=len(profile.reference_images),
            updated_at=profile.updated_at,
            store_path=str(self.path),
        )

    def upsert_reference(
        self,
        *,
        user_id: str,
        display_name: str | None,
        primary_user: bool,
        image_bytes: bytes,
        filename_hint: str,
        embedding: tuple[float, ...],
        detector_confidence: float | None,
        source: str,
        max_reference_images: int,
    ) -> tuple[PortraitIdentityProfile, PortraitReferenceImage, bool]:
        normalized_user_id = _normalize_user_id(user_id)
        normalized_display_name = _normalize_display_name(display_name)
        normalized_source = _normalize_source(source)
        image_sha256 = hashlib.sha256(image_bytes).hexdigest()
        suffix = _safe_suffix(filename_hint)
        max_references = max(1, int(max_reference_images))

        with self._exclusive_lock(create_parent=True):
            profiles = list(self._load_profiles_unlocked())
            profile_index = self._profile_index(profiles, normalized_user_id)
            if profile_index is None:
                created_at = _utc_iso()
                profile = PortraitIdentityProfile(
                    user_id=normalized_user_id,
                    display_name=normalized_display_name,
                    primary_user=bool(primary_user),
                    created_at=created_at,
                    updated_at=created_at,
                    reference_images=(),
                )
                profiles.append(profile)
                profile_index = len(profiles) - 1
            else:
                profile = profiles[profile_index]

            for existing in profile.reference_images:
                if existing.image_sha256 == image_sha256:
                    updated_profile = PortraitIdentityProfile(
                        user_id=profile.user_id,
                        display_name=normalized_display_name or profile.display_name,
                        primary_user=bool(primary_user or profile.primary_user),
                        created_at=profile.created_at,
                        updated_at=_utc_iso(),
                        reference_images=profile.reference_images,
                    )
                    profiles[profile_index] = updated_profile
                    self._write_profiles_unlocked(profiles)
                    return updated_profile, existing, True

            reference_id = f"{int(datetime.now(UTC).timestamp() * 1000)}_{image_sha256[:12]}"
            relative_path = Path(self._user_directory_name(normalized_user_id)) / f"{reference_id}{suffix}"
            absolute_path = self._absolute_reference_path(relative_path)
            absolute_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_reference_image_unlocked(absolute_path, image_bytes)

            reference = PortraitReferenceImage(
                reference_id=reference_id,
                relative_path=relative_path.as_posix(),
                image_sha256=image_sha256,
                source=normalized_source,
                added_at=_utc_iso(),
                embedding=_normalize_embedding(embedding),
                detector_confidence=detector_confidence,
            )
            images = list(profile.reference_images)
            images.append(reference)
            removed_images: list[PortraitReferenceImage] = []
            if len(images) > max_references:
                images.sort(key=lambda item: item.added_at)
                removed_images = images[:-max_references]
                images = images[-max_references:]
            updated_profile = PortraitIdentityProfile(
                user_id=profile.user_id,
                display_name=normalized_display_name or profile.display_name,
                primary_user=bool(primary_user or profile.primary_user),
                created_at=profile.created_at,
                updated_at=_utc_iso(),
                reference_images=tuple(images),
            )
            profiles[profile_index] = updated_profile
            self._write_profiles_unlocked(profiles)
            for removed in removed_images:
                self._unlink_reference_unlocked(removed.relative_path)
            return updated_profile, reference, False

    def remove_reference(
        self,
        *,
        user_id: str,
        reference_id: str,
    ) -> PortraitIdentityProfile | None:
        normalized_user_id = _normalize_user_id(user_id)
        normalized_reference_id = _normalize_text(reference_id).strip()
        if not normalized_reference_id:
            return self.load_profile(normalized_user_id)

        with self._exclusive_lock(create_parent=False):
            profiles = list(self._load_profiles_unlocked())
            profile_index = self._profile_index(profiles, normalized_user_id)
            if profile_index is None:
                return None
            profile = profiles[profile_index]
            kept: list[PortraitReferenceImage] = []
            removed: PortraitReferenceImage | None = None
            for image in profile.reference_images:
                if image.reference_id == normalized_reference_id and removed is None:
                    removed = image
                    continue
                kept.append(image)
            if removed is None:
                return profile
            if kept:
                updated_profile = PortraitIdentityProfile(
                    user_id=profile.user_id,
                    display_name=profile.display_name,
                    primary_user=profile.primary_user,
                    created_at=profile.created_at,
                    updated_at=_utc_iso(),
                    reference_images=tuple(kept),
                )
                profiles[profile_index] = updated_profile
            else:
                updated_profile = None
                del profiles[profile_index]
            self._write_profiles_unlocked(profiles)
            self._unlink_reference_unlocked(removed.relative_path)
            return updated_profile

    def clear_profile(self, *, user_id: str) -> bool:
        normalized_user_id = _normalize_user_id(user_id)
        with self._exclusive_lock(create_parent=False):
            profiles = list(self._load_profiles_unlocked())
            profile_index = self._profile_index(profiles, normalized_user_id)
            if profile_index is None:
                return False
            profile = profiles.pop(profile_index)
            self._write_profiles_unlocked(profiles)
            for image in profile.reference_images:
                self._unlink_reference_unlocked(image.relative_path)
            self._remove_user_directory_if_empty(normalized_user_id)
            return True

    def _profile_index(self, profiles: list[PortraitIdentityProfile], user_id: str) -> int | None:
        for index, profile in enumerate(profiles):
            if profile.user_id == user_id:
                return index
        return None

    def _load_profiles_unlocked(self) -> tuple[PortraitIdentityProfile, ...]:
        if not self.path.exists():
            return ()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(self.path, flags)
        try:
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode):
                raise OSError("portrait_identity_store_not_regular")
            if metadata.st_size > _MAX_STORE_BYTES:
                raise OSError("portrait_identity_store_too_large")
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
        profiles: list[PortraitIdentityProfile] = []
        for item in profiles_raw:
            if not isinstance(item, dict):
                return ()
            profile = PortraitIdentityProfile.from_dict(item)
            if profile is None:
                return ()
            profiles.append(profile)
        profiles.sort(key=lambda item: (not item.primary_user, item.user_id))
        return tuple(profiles)

    def _write_profiles_unlocked(self, profiles: list[PortraitIdentityProfile]) -> None:
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

    def _write_reference_image_unlocked(self, path: Path, image_bytes: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(image_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass

    def _absolute_reference_path(self, relative_path: Path) -> Path:
        candidate = (self.image_dir / relative_path).resolve(strict=False)
        root = self.image_dir.resolve(strict=False)
        if not _is_relative_to(candidate, root):
            raise ValueError("portrait_reference_path_escaped_image_dir")
        return candidate

    def _unlink_reference_unlocked(self, relative_path: str) -> None:
        try:
            absolute_path = self._absolute_reference_path(Path(relative_path))
        except ValueError:
            return
        try:
            absolute_path.unlink()
        except FileNotFoundError:
            return
        except OSError:
            return
        try:
            absolute_path.parent.rmdir()
        except OSError:
            return

    def _remove_user_directory_if_empty(self, user_id: str) -> None:
        directory = (self.image_dir / self._user_directory_name(user_id)).resolve(strict=False)
        root = self.image_dir.resolve(strict=False)
        if not _is_relative_to(directory, root):
            return
        try:
            directory.rmdir()
        except OSError:
            return

    def _user_directory_name(self, user_id: str) -> str:
        return _normalize_user_id(user_id)

    @contextmanager
    def _exclusive_lock(self, *, create_parent: bool) -> Iterator[None]:
        parent = self.path.parent
        if create_parent:
            parent.mkdir(parents=True, exist_ok=True)
            self.image_dir.mkdir(parents=True, exist_ok=True)
        elif not parent.exists():
            raise OSError("portrait_identity_store_parent_missing")
        lock_path = parent / f".{self.path.name}.lock"
        with self._process_lock:
            with open(lock_path, "a+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


__all__ = [
    "PortraitIdentityProfile",
    "PortraitIdentityStore",
    "PortraitIdentitySummary",
    "PortraitReferenceImage",
    "portrait_identity_image_dir",
    "portrait_identity_store_path",
]
