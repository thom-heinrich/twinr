# CHANGELOG: 2026-03-28
# BUG-1: Replaced the ad-hoc JSON store with a crash-resilient SQLite/WAL store; fixes silent full-store disappearance on malformed JSON/embeddings.
# BUG-2: Rejects empty/zero-norm/non-finite embeddings at write time instead of persisting state that becomes unreadable after restart.
# BUG-3: Eliminates metadata/file divergence via transactional DB updates plus orphan-file reconciliation and backup/restore.
# SEC-1: Adds practical Pi-safe upload hardening with a max reference-image size limit to block trivial disk-exhaustion attacks.
# SEC-2: Hardens on-disk state with 0700/0600 permissions, SQLite defensive/trusted-schema settings, integrity checks, secure delete, and bounded WAL maintenance.
# IMP-1: Upgrades storage to SQLite STRICT/WAL with automatic integrity check, point-in-time backup, restore, and legacy JSON migration.
# IMP-2: Stores embeddings as normalized float32 BLOBs and persists optional embedding-model/quality metadata for 2026-grade re-embedding and FIQA workflows.

"""Persist local portrait-identity profiles and reference images on device.

This module owns the file-backed identity enrollment state used by local
portrait matching: per-user reference images, extracted embeddings, and small
summary helpers. Live camera capture and matching policy stay in
``portrait_match.py``.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import sqlite3
import struct
import tempfile
import threading

from twinr.agent.base_agent.config import TwinrConfig


_ALLOWED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
_DB_STORE_VERSION = 2
_EMBEDDING_VERSION = 1
# BREAKING: the canonical store is now SQLite-backed; legacy .json stores are auto-migrated.
_DEFAULT_STORE_PATH = "state/portrait_identities.sqlite3"
_DEFAULT_IMAGE_DIR = "state/portrait_identities"
_DEFAULT_MAX_IMAGE_BYTES = 12 * 1024 * 1024
_MAX_LEGACY_JSON_BYTES = 2 * 1024 * 1024
_JOURNAL_SIZE_LIMIT_BYTES = 1 * 1024 * 1024
_WAL_AUTOCHECKPOINT_PAGES = 128
_SQLITE_BUSY_TIMEOUT_MS = 5000


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


def _normalize_embedding_model_id(value: object | None) -> str:
    text = _normalize_text(value).strip().lower()
    normalized: list[str] = []
    previous_separator = False
    for char in text:
        if char.isalnum():
            normalized.append(char)
            previous_separator = False
            continue
        if char in {"-", "_", " ", "/", "."}:
            if not previous_separator:
                normalized.append("_")
                previous_separator = True
    collapsed = "".join(normalized).strip("_")
    return collapsed or f"portrait_embedding_v{_EMBEDDING_VERSION}"


def _normalize_embedding(values: Sequence[float]) -> tuple[float, ...]:
    if not values:
        raise ValueError("portrait_embedding_empty")
    try:
        numeric = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("portrait_embedding_not_numeric") from exc
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("portrait_embedding_not_finite")
    norm = math.sqrt(sum(value * value for value in numeric))
    if norm <= 1e-12 or not math.isfinite(norm):
        raise ValueError("portrait_embedding_zero_norm")
    return tuple(value / norm for value in numeric)


def _pack_embedding(values: Sequence[float]) -> tuple[bytes, tuple[float, ...], int]:
    normalized = _normalize_embedding(values)
    try:
        payload = struct.pack(f"<{len(normalized)}f", *normalized)
    except struct.error as exc:
        raise ValueError("portrait_embedding_pack_failed") from exc
    return payload, normalized, len(normalized)


def _unpack_embedding(blob: bytes, dimension: int) -> tuple[float, ...]:
    if dimension <= 0:
        raise ValueError("portrait_embedding_dimension_invalid")
    expected = 4 * dimension
    if len(blob) != expected:
        raise ValueError("portrait_embedding_blob_size_mismatch")
    values = struct.unpack(f"<{dimension}f", blob)
    return _normalize_embedding(values)


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


def _normalize_store_path(path: Path) -> Path:
    # BREAKING: a configured legacy .json path is treated as a migration source;
    # the live store is a sibling .sqlite3 database.
    if path.suffix.casefold() == ".json":
        return path.with_suffix(".sqlite3")
    return path


def _resolve_path(*, project_root: Path, raw_value: object | None, default: str) -> Path:
    text = str(raw_value or "").strip() or default
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return _normalize_store_path(candidate.resolve(strict=False))
    return _normalize_store_path((project_root / candidate).resolve(strict=False))


def _resolve_image_dir(*, project_root: Path, raw_value: object | None, default: str) -> Path:
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


def _fsync_directory(path: Path) -> None:
    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _ensure_directory(path: Path, *, mode: int = 0o700) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def _ensure_file_mode(path: Path, *, mode: int = 0o600) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        pass


@dataclass(frozen=True, slots=True)
class PortraitReferenceImage:
    reference_id: str
    relative_path: str
    image_sha256: str
    source: str
    added_at: str
    embedding: tuple[float, ...]
    detector_confidence: float | None = None
    embedding_version: int = _EMBEDDING_VERSION
    image_size_bytes: int = 0
    quality_score: float | None = None
    embedding_model_id: str = f"portrait_embedding_v{_EMBEDDING_VERSION}"

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
            "image_size_bytes": self.image_size_bytes,
            "quality_score": self.quality_score,
            "embedding_model_id": self.embedding_model_id,
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
        if not reference_id or not relative_path or not image_sha256:
            return None
        try:
            embedding = _normalize_embedding(embedding_raw)
        except ValueError:
            return None
        try:
            embedding_version = int(payload.get("embedding_version", _EMBEDDING_VERSION) or _EMBEDDING_VERSION)
        except (TypeError, ValueError):
            return None
        return cls(
            reference_id=reference_id,
            relative_path=relative_path,
            image_sha256=image_sha256,
            source=source,
            added_at=added_at,
            embedding=embedding,
            detector_confidence=_coerce_optional_float(payload.get("detector_confidence")),
            embedding_version=embedding_version,
            image_size_bytes=max(0, int(payload.get("image_size_bytes", 0) or 0)),
            quality_score=_coerce_optional_float(payload.get("quality_score")),
            embedding_model_id=_normalize_embedding_model_id(payload.get("embedding_model_id")),
        )


@dataclass(frozen=True, slots=True)
class PortraitIdentityProfile:
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
                continue
            parsed = PortraitReferenceImage.from_dict(item)
            if parsed is None:
                continue
            images.append(parsed)
        images.sort(key=lambda item: (item.added_at, item.reference_id))
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
    user_id: str
    display_name: str | None
    primary_user: bool
    enrolled: bool
    reference_image_count: int
    updated_at: str | None
    store_path: str


class PortraitIdentityStoreError(RuntimeError):
    """Raised when the persistent portrait store cannot be trusted."""


def portrait_identity_store_path(config: TwinrConfig) -> Path:
    project_root = Path(config.project_root).expanduser().resolve(strict=False)
    return _resolve_path(
        project_root=project_root,
        raw_value=getattr(config, "portrait_match_store_path", None),
        default=_DEFAULT_STORE_PATH,
    )


def portrait_identity_image_dir(config: TwinrConfig) -> Path:
    project_root = Path(config.project_root).expanduser().resolve(strict=False)
    return _resolve_image_dir(
        project_root=project_root,
        raw_value=getattr(config, "portrait_match_reference_image_dir", None),
        default=_DEFAULT_IMAGE_DIR,
    )


class PortraitIdentityStore:
    """Persist local portrait identity profiles in a hardened SQLite store."""

    def __init__(
        self,
        store_path: str | Path,
        image_dir: str | Path,
        *,
        max_image_bytes: int = _DEFAULT_MAX_IMAGE_BYTES,
    ) -> None:
        configured_store_path = Path(os.path.abspath(os.fspath(Path(store_path).expanduser())))
        self.path = _normalize_store_path(configured_store_path.resolve(strict=False))
        self.backup_path = self.path.with_suffix(self.path.suffix + ".bak")
        self.legacy_json_path = configured_store_path.resolve(strict=False).with_suffix(".json")
        self.image_dir = Path(os.path.abspath(os.fspath(Path(image_dir).expanduser()))).resolve(strict=False)
        self.max_image_bytes = max(1024, int(max_image_bytes))
        self._process_lock = threading.RLock()
        self._initialized = False

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "PortraitIdentityStore":
        return cls(
            store_path=portrait_identity_store_path(config),
            image_dir=portrait_identity_image_dir(config),
            max_image_bytes=max(
                1024,
                int(getattr(config, "portrait_match_max_reference_image_bytes", _DEFAULT_MAX_IMAGE_BYTES)),
            ),
        )

    def list_profiles(self) -> tuple[PortraitIdentityProfile, ...]:
        if not self._store_artifacts_exist():
            return ()
        self._ensure_ready(create=False)
        try:
            with self._connect() as conn:
                return self._fetch_all_profiles(conn)
        except PortraitIdentityStoreError:
            with self._exclusive_lock():
                self._restore_from_backup_locked()
            with self._connect() as conn:
                return self._fetch_all_profiles(conn)

    def load_profile(self, user_id: str) -> PortraitIdentityProfile | None:
        normalized_user_id = _normalize_user_id(user_id)
        if not self._store_artifacts_exist():
            return None
        self._ensure_ready(create=False)
        try:
            with self._connect() as conn:
                return self._fetch_profile(conn, normalized_user_id)
        except PortraitIdentityStoreError:
            with self._exclusive_lock():
                self._restore_from_backup_locked()
            with self._connect() as conn:
                return self._fetch_profile(conn, normalized_user_id)

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
        quality_score: float | None = None,
        embedding_model_id: str | None = None,
    ) -> tuple[PortraitIdentityProfile, PortraitReferenceImage, bool]:
        normalized_user_id = _normalize_user_id(user_id)
        normalized_display_name = _normalize_display_name(display_name)
        normalized_source = _normalize_source(source)
        normalized_detector_confidence = _coerce_optional_float(detector_confidence)
        normalized_quality_score = _coerce_optional_float(quality_score)
        normalized_embedding_model_id = _normalize_embedding_model_id(embedding_model_id)

        if not isinstance(image_bytes, (bytes, bytearray, memoryview)):
            raise TypeError("portrait_reference_image_bytes_must_be_bytes_like")
        image_bytes = bytes(image_bytes)
        if not image_bytes:
            raise ValueError("portrait_reference_image_empty")
        if len(image_bytes) > self.max_image_bytes:
            raise ValueError("portrait_reference_image_too_large")

        image_sha256 = hashlib.sha256(image_bytes).hexdigest()
        suffix = _safe_suffix(filename_hint)
        max_references = max(1, int(max_reference_images))
        embedding_blob, _, embedding_dim = _pack_embedding(embedding)

        self._ensure_ready(create=True)
        with self._exclusive_lock():
            self._ensure_ready(create=True)
            user_dir = self._absolute_reference_path(Path(self._user_directory_name(normalized_user_id)))
            _ensure_directory(user_dir)
            staged_path = self._write_staged_image(user_dir, image_bytes)
            final_relative_path: str | None = None
            removed_relative_paths: list[str] = []
            commit_succeeded = False
            try:
                with self._connect() as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    profile_row = conn.execute(
                        "SELECT user_id, display_name, primary_user, created_at, updated_at "
                        "FROM profiles WHERE user_id = ?",
                        (normalized_user_id,),
                    ).fetchone()

                    if profile_row is None:
                        created_at = _utc_iso()
                        conn.execute(
                            "INSERT INTO profiles(user_id, display_name, primary_user, created_at, updated_at) "
                            "VALUES(?, ?, ?, ?, ?)",
                            (normalized_user_id, normalized_display_name, 1 if primary_user else 0, created_at, created_at),
                        )
                        profile_display_name = normalized_display_name
                        profile_primary = bool(primary_user)
                    else:
                        profile_display_name = normalized_display_name or (
                            str(profile_row["display_name"]) if profile_row["display_name"] is not None else None
                        )
                        profile_primary = bool(primary_user or bool(profile_row["primary_user"]))

                    existing_rows = conn.execute(
                        "SELECT reference_id, relative_path, image_sha256, source, added_at, embedding, embedding_dim, "
                        "embedding_version, detector_confidence, image_size_bytes, quality_score, embedding_model_id "
                        "FROM reference_images WHERE user_id = ? ORDER BY added_at ASC, reference_id ASC",
                        (normalized_user_id,),
                    ).fetchall()

                    duplicate_row = None
                    rows_to_replace: list[sqlite3.Row] = []
                    for row in existing_rows:
                        if (
                            str(row["image_sha256"]) == image_sha256
                            and int(row["embedding_version"]) == _EMBEDDING_VERSION
                            and str(row["embedding_model_id"]) == normalized_embedding_model_id
                        ):
                            duplicate_row = row
                            break
                        if str(row["image_sha256"]) == image_sha256:
                            rows_to_replace.append(row)

                    if duplicate_row is not None:
                        for row in rows_to_replace:
                            removed_relative_paths.append(str(row["relative_path"]))
                            conn.execute(
                                "DELETE FROM reference_images WHERE reference_id = ?",
                                (str(row["reference_id"]),),
                            )
                        updated_at = _utc_iso()
                        conn.execute(
                            "UPDATE profiles SET display_name = ?, primary_user = ?, updated_at = ? WHERE user_id = ?",
                            (profile_display_name, 1 if profile_primary else 0, updated_at, normalized_user_id),
                        )
                        conn.commit()
                        commit_succeeded = True
                        os.unlink(staged_path)
                        try:
                            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                        except sqlite3.DatabaseError:
                            pass
                        profile = self._fetch_profile(conn, normalized_user_id)
                        assert profile is not None
                        existing_ref = self._reference_from_row(duplicate_row)
                        with suppress(Exception):
                            self._reconcile_files_locked()
                            self._refresh_backup_locked()
                        return profile, existing_ref, True

                    reference_id = f"{int(datetime.now(UTC).timestamp() * 1000)}_{secrets.token_hex(8)}"
                    relative_path = (Path(self._user_directory_name(normalized_user_id)) / f"{reference_id}{suffix}").as_posix()
                    final_absolute_path = self._absolute_reference_path(Path(relative_path))
                    _ensure_directory(final_absolute_path.parent)
                    os.replace(staged_path, final_absolute_path)
                    _ensure_file_mode(final_absolute_path)
                    _fsync_directory(final_absolute_path.parent)
                    final_relative_path = relative_path

                    for row in rows_to_replace:
                        removed_relative_paths.append(str(row["relative_path"]))
                        conn.execute(
                            "DELETE FROM reference_images WHERE reference_id = ?",
                            (str(row["reference_id"]),),
                        )

                    added_at = _utc_iso()
                    conn.execute(
                        "INSERT INTO reference_images("
                        "reference_id, user_id, relative_path, image_sha256, image_size_bytes, source, added_at, "
                        "embedding, embedding_dim, embedding_dtype, embedding_version, detector_confidence, quality_score, embedding_model_id"
                        ") VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            reference_id,
                            normalized_user_id,
                            relative_path,
                            image_sha256,
                            len(image_bytes),
                            normalized_source,
                            added_at,
                            sqlite3.Binary(embedding_blob),
                            embedding_dim,
                            "f32",
                            _EMBEDDING_VERSION,
                            normalized_detector_confidence,
                            normalized_quality_score,
                            normalized_embedding_model_id,
                        ),
                    )
                    removed_relative_paths.extend(self._trim_profile_references(conn, normalized_user_id, max_references))
                    updated_at = _utc_iso()
                    conn.execute(
                        "UPDATE profiles SET display_name = ?, primary_user = ?, updated_at = ? WHERE user_id = ?",
                        (profile_display_name, 1 if profile_primary else 0, updated_at, normalized_user_id),
                    )
                    conn.commit()
                    commit_succeeded = True
                    try:
                        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    except sqlite3.DatabaseError:
                        pass
                    profile = self._fetch_profile(conn, normalized_user_id)
                    reference_row = conn.execute(
                        "SELECT reference_id, relative_path, image_sha256, source, added_at, embedding, embedding_dim, "
                        "embedding_version, detector_confidence, image_size_bytes, quality_score, embedding_model_id "
                        "FROM reference_images WHERE reference_id = ?",
                        (reference_id,),
                    ).fetchone()
                    if profile is None or reference_row is None:
                        raise PortraitIdentityStoreError("portrait_identity_store_post_commit_reload_failed")

                with suppress(Exception):
                    self._cleanup_removed_files(removed_relative_paths)
                    self._reconcile_files_locked()
                    self._refresh_backup_locked()

                return profile, self._reference_from_row(reference_row), False
            except Exception:
                try:
                    if staged_path.exists():
                        os.unlink(staged_path)
                except OSError:
                    pass
                if final_relative_path and not commit_succeeded:
                    self._unlink_relative_path(final_relative_path)
                raise

    def remove_reference(self, *, user_id: str, reference_id: str) -> PortraitIdentityProfile | None:
        normalized_user_id = _normalize_user_id(user_id)
        normalized_reference_id = _normalize_text(reference_id).strip()
        if not normalized_reference_id:
            return self.load_profile(normalized_user_id)
        if not self._store_artifacts_exist():
            return None

        self._ensure_ready(create=False)
        with self._exclusive_lock():
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT relative_path FROM reference_images WHERE user_id = ? AND reference_id = ?",
                    (normalized_user_id, normalized_reference_id),
                ).fetchone()
                if row is None:
                    profile = self._fetch_profile(conn, normalized_user_id)
                    conn.rollback()
                    return profile

                removed_relative_path = str(row["relative_path"])
                conn.execute("DELETE FROM reference_images WHERE reference_id = ?", (normalized_reference_id,))
                remaining = conn.execute(
                    "SELECT COUNT(*) FROM reference_images WHERE user_id = ?",
                    (normalized_user_id,),
                ).fetchone()[0]
                if remaining:
                    conn.execute(
                        "UPDATE profiles SET updated_at = ? WHERE user_id = ?",
                        (_utc_iso(), normalized_user_id),
                    )
                else:
                    conn.execute("DELETE FROM profiles WHERE user_id = ?", (normalized_user_id,))
                conn.commit()
                try:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except sqlite3.DatabaseError:
                    pass
                profile = self._fetch_profile(conn, normalized_user_id)

            with suppress(Exception):
                self._cleanup_removed_files([removed_relative_path])
                self._remove_user_directory_if_empty(normalized_user_id)
                self._reconcile_files_locked()
                self._refresh_backup_locked()
            return profile

    def clear_profile(self, *, user_id: str) -> bool:
        normalized_user_id = _normalize_user_id(user_id)
        if not self._store_artifacts_exist():
            return False

        self._ensure_ready(create=False)
        with self._exclusive_lock():
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                rows = conn.execute(
                    "SELECT relative_path FROM reference_images WHERE user_id = ?",
                    (normalized_user_id,),
                ).fetchall()
                if not rows and conn.execute(
                    "SELECT 1 FROM profiles WHERE user_id = ?",
                    (normalized_user_id,),
                ).fetchone() is None:
                    conn.rollback()
                    return False

                conn.execute("DELETE FROM reference_images WHERE user_id = ?", (normalized_user_id,))
                conn.execute("DELETE FROM profiles WHERE user_id = ?", (normalized_user_id,))
                conn.commit()
                try:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except sqlite3.DatabaseError:
                    pass

            with suppress(Exception):
                self._cleanup_removed_files([str(row["relative_path"]) for row in rows])
                self._remove_user_directory_if_empty(normalized_user_id)
                self._reconcile_files_locked()
                self._refresh_backup_locked()
            return True

    def _user_directory_name(self, user_id: str) -> str:
        return _normalize_user_id(user_id)

    def _store_artifacts_exist(self) -> bool:
        if self.path.exists() or self.backup_path.exists():
            return True
        return self.legacy_json_path.exists()

    def _ensure_ready(self, *, create: bool) -> None:
        if self._initialized and (create or self.path.exists() or self.backup_path.exists()):
            return

        with self._exclusive_lock():
            if self._initialized and (create or self.path.exists() or self.backup_path.exists()):
                return

            self._ensure_filesystem_layout(create=create)

            if self.path.exists():
                self._ensure_sqlite_database_valid_locked()
                self._initialize_schema_locked()
                self._reconcile_files_locked()
                self._initialized = True
                return

            if self.legacy_json_path.exists():
                self._migrate_legacy_json_locked()
                self._initialize_schema_locked()
                self._reconcile_files_locked()
                self._refresh_backup_locked()
                self._initialized = True
                return

            if self.backup_path.exists():
                self._restore_from_backup_locked()
                self._initialize_schema_locked()
                self._reconcile_files_locked()
                self._initialized = True
                return

            if create:
                self._initialize_schema_locked()
                self._refresh_backup_locked()
                self._initialized = True

    def _ensure_filesystem_layout(self, *, create: bool) -> None:
        if create:
            _ensure_directory(self.path.parent)
            _ensure_directory(self.image_dir)
        elif self.path.parent.exists():
            try:
                os.chmod(self.path.parent, 0o700)
            except OSError:
                pass
            if self.image_dir.exists():
                try:
                    os.chmod(self.image_dir, 0o700)
                except OSError:
                    pass

    def _initialize_schema_locked(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS profiles(
                    user_id TEXT PRIMARY KEY NOT NULL,
                    display_name TEXT,
                    primary_user INTEGER NOT NULL CHECK(primary_user IN (0, 1)),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                ) STRICT, WITHOUT ROWID
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reference_images(
                    reference_id TEXT PRIMARY KEY NOT NULL,
                    user_id TEXT NOT NULL,
                    relative_path TEXT NOT NULL UNIQUE,
                    image_sha256 TEXT NOT NULL,
                    image_size_bytes INTEGER NOT NULL CHECK(image_size_bytes >= 0),
                    source TEXT NOT NULL,
                    added_at TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL CHECK(embedding_dim > 0),
                    embedding_dtype TEXT NOT NULL CHECK(embedding_dtype = 'f32'),
                    embedding_version INTEGER NOT NULL CHECK(embedding_version >= 1),
                    detector_confidence REAL,
                    quality_score REAL,
                    embedding_model_id TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES profiles(user_id) ON DELETE CASCADE,
                    UNIQUE(user_id, image_sha256, embedding_version, embedding_model_id)
                ) STRICT, WITHOUT ROWID
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_references_user_added "
                "ON reference_images(user_id, added_at, reference_id)"
            )
            conn.execute(f"PRAGMA user_version = {_DB_STORE_VERSION}")
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        _ensure_directory(self.path.parent)
        connection = sqlite3.connect(self.path, timeout=_SQLITE_BUSY_TIMEOUT_MS / 1000.0, isolation_level=None)
        connection.row_factory = sqlite3.Row

        try:
            connection.enable_load_extension(False)
        except Exception:
            pass

        try:
            if hasattr(connection, "setconfig"):
                if hasattr(sqlite3, "SQLITE_DBCONFIG_DEFENSIVE"):
                    connection.setconfig(sqlite3.SQLITE_DBCONFIG_DEFENSIVE, True)
                if hasattr(sqlite3, "SQLITE_DBCONFIG_TRUSTED_SCHEMA"):
                    connection.setconfig(sqlite3.SQLITE_DBCONFIG_TRUSTED_SCHEMA, False)
        except Exception:
            pass

        try:
            if hasattr(connection, "setlimit") and hasattr(sqlite3, "SQLITE_LIMIT_ATTACHED"):
                connection.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED, 0)
        except Exception:
            pass

        if self.path.exists() and self.path.stat().st_size > 0:
            try:
                result = connection.execute("PRAGMA quick_check").fetchone()
            except sqlite3.DatabaseError as exc:
                connection.close()
                raise PortraitIdentityStoreError("portrait_identity_store_corrupt") from exc
            if result is None or str(result[0]).lower() != "ok":
                connection.close()
                raise PortraitIdentityStoreError(
                    f"portrait_identity_store_corrupt:{result[0] if result else 'unknown'}"
                )

        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {_SQLITE_BUSY_TIMEOUT_MS}")
        connection.execute("PRAGMA mmap_size = 0")
        connection.execute("PRAGMA cell_size_check = ON")
        connection.execute("PRAGMA secure_delete = ON")
        connection.execute(f"PRAGMA wal_autocheckpoint = {_WAL_AUTOCHECKPOINT_PAGES}")
        connection.execute(f"PRAGMA journal_size_limit = {_JOURNAL_SIZE_LIMIT_BYTES}")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    def _ensure_sqlite_database_valid_locked(self) -> None:
        try:
            with self._connect():
                return
        except PortraitIdentityStoreError as primary_exc:
            if self.backup_path.exists():
                try:
                    self._restore_from_backup_locked()
                    with self._connect():
                        return
                except Exception as backup_exc:
                    raise PortraitIdentityStoreError("portrait_identity_store_restore_failed") from backup_exc
            raise primary_exc

    def _migrate_legacy_json_locked(self) -> None:
        if not self.legacy_json_path.exists():
            return
        _ensure_directory(self.path.parent)
        if self.legacy_json_path.stat().st_size > _MAX_LEGACY_JSON_BYTES:
            raise PortraitIdentityStoreError("portrait_identity_legacy_store_too_large")

        with open(self.legacy_json_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        profiles = self._parse_legacy_profiles(payload)
        self._initialize_schema_locked()

        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM reference_images")
            conn.execute("DELETE FROM profiles")
            for profile in profiles:
                conn.execute(
                    "INSERT OR REPLACE INTO profiles(user_id, display_name, primary_user, created_at, updated_at) "
                    "VALUES(?, ?, ?, ?, ?)",
                    (
                        profile.user_id,
                        profile.display_name,
                        1 if profile.primary_user else 0,
                        profile.created_at,
                        profile.updated_at,
                    ),
                )
                for reference in profile.reference_images:
                    embedding_blob, _, embedding_dim = _pack_embedding(reference.embedding)
                    conn.execute(
                        "INSERT OR REPLACE INTO reference_images("
                        "reference_id, user_id, relative_path, image_sha256, image_size_bytes, source, added_at, "
                        "embedding, embedding_dim, embedding_dtype, embedding_version, detector_confidence, quality_score, embedding_model_id"
                        ") VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            reference.reference_id,
                            profile.user_id,
                            reference.relative_path,
                            reference.image_sha256,
                            max(0, reference.image_size_bytes),
                            reference.source,
                            reference.added_at,
                            sqlite3.Binary(embedding_blob),
                            embedding_dim,
                            "f32",
                            max(1, int(reference.embedding_version)),
                            reference.detector_confidence,
                            reference.quality_score,
                            _normalize_embedding_model_id(reference.embedding_model_id),
                        ),
                    )
            conn.commit()

        backup_legacy_path = self.legacy_json_path.with_suffix(self.legacy_json_path.suffix + ".migrated.bak")
        try:
            shutil.copy2(self.legacy_json_path, backup_legacy_path)
            _ensure_file_mode(backup_legacy_path)
            _fsync_directory(backup_legacy_path.parent)
        except OSError:
            pass

    def _parse_legacy_profiles(self, payload: object) -> tuple[PortraitIdentityProfile, ...]:
        if not isinstance(payload, dict):
            return ()
        profiles_raw = payload.get("profiles", ())
        if not isinstance(profiles_raw, list):
            return ()

        merged: dict[str, PortraitIdentityProfile] = {}
        for item in profiles_raw:
            if not isinstance(item, dict):
                continue
            profile = PortraitIdentityProfile.from_dict(item)
            if profile is None:
                continue

            existing = merged.get(profile.user_id)
            if existing is None:
                merged[profile.user_id] = profile
                continue

            images_by_id = {image.reference_id: image for image in existing.reference_images}
            for image in profile.reference_images:
                images_by_id.setdefault(image.reference_id, image)

            images = tuple(sorted(images_by_id.values(), key=lambda entry: (entry.added_at, entry.reference_id)))
            merged[profile.user_id] = PortraitIdentityProfile(
                user_id=profile.user_id,
                display_name=profile.display_name or existing.display_name,
                primary_user=bool(profile.primary_user or existing.primary_user),
                created_at=min(existing.created_at, profile.created_at),
                updated_at=max(existing.updated_at, profile.updated_at),
                reference_images=images,
            )

        return tuple(sorted(merged.values(), key=lambda entry: (not entry.primary_user, entry.user_id)))

    def _restore_from_backup_locked(self) -> None:
        if not self.backup_path.exists():
            raise PortraitIdentityStoreError("portrait_identity_store_backup_missing")

        tmp_fd, tmp_path_raw = tempfile.mkstemp(
            prefix=f".{self.path.name}.restore.",
            suffix=".tmp",
            dir=self.path.parent,
        )
        os.close(tmp_fd)
        tmp_path = Path(tmp_path_raw)

        try:
            with sqlite3.connect(self.backup_path, isolation_level=None) as src, sqlite3.connect(
                tmp_path, isolation_level=None
            ) as dst:
                src.execute("PRAGMA mmap_size = 0")
                src.execute("PRAGMA cell_size_check = ON")
                result = src.execute("PRAGMA quick_check").fetchone()
                if result is None or str(result[0]).lower() != "ok":
                    raise PortraitIdentityStoreError("portrait_identity_store_backup_corrupt")
                src.backup(dst)
            _ensure_file_mode(tmp_path)
            os.replace(tmp_path, self.path)
            _ensure_file_mode(self.path)
            _fsync_directory(self.path.parent)
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass

        self._remove_sidecar(self.path.with_name(self.path.name + "-wal"))
        self._remove_sidecar(self.path.with_name(self.path.name + "-shm"))

    def _remove_sidecar(self, path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            return
        except OSError:
            return

    def _trim_profile_references(self, conn: sqlite3.Connection, user_id: str, max_references: int) -> list[str]:
        rows = conn.execute(
            "SELECT reference_id, relative_path FROM reference_images WHERE user_id = ? "
            "ORDER BY added_at ASC, reference_id ASC",
            (user_id,),
        ).fetchall()
        if len(rows) <= max_references:
            return []

        removed_paths: list[str] = []
        for row in rows[:-max_references]:
            removed_paths.append(str(row["relative_path"]))
            conn.execute("DELETE FROM reference_images WHERE reference_id = ?", (str(row["reference_id"]),))
        return removed_paths

    def _refresh_backup_locked(self) -> None:
        _ensure_directory(self.backup_path.parent)
        tmp_fd, tmp_path_raw = tempfile.mkstemp(
            prefix=f".{self.backup_path.name}.",
            suffix=".tmp",
            dir=self.backup_path.parent,
        )
        os.close(tmp_fd)
        tmp_path = Path(tmp_path_raw)

        try:
            with self._connect() as src, sqlite3.connect(tmp_path, isolation_level=None) as dst:
                dst.execute("PRAGMA journal_mode = DELETE")
                dst.execute("PRAGMA synchronous = FULL")
                src.backup(dst)
            _ensure_file_mode(tmp_path)
            os.replace(tmp_path, self.backup_path)
            _ensure_file_mode(self.backup_path)
            _fsync_directory(self.backup_path.parent)
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass

    def _fetch_all_profiles(self, conn: sqlite3.Connection) -> tuple[PortraitIdentityProfile, ...]:
        profile_rows = conn.execute(
            "SELECT user_id, display_name, primary_user, created_at, updated_at "
            "FROM profiles ORDER BY primary_user DESC, user_id ASC"
        ).fetchall()
        if not profile_rows:
            return ()

        reference_rows = conn.execute(
            "SELECT reference_id, user_id, relative_path, image_sha256, source, added_at, embedding, embedding_dim, "
            "embedding_version, detector_confidence, image_size_bytes, quality_score, embedding_model_id "
            "FROM reference_images ORDER BY user_id ASC, added_at ASC, reference_id ASC"
        ).fetchall()
        refs_by_user: dict[str, list[PortraitReferenceImage]] = {}
        for row in reference_rows:
            refs_by_user.setdefault(str(row["user_id"]), []).append(self._reference_from_row(row))

        profiles: list[PortraitIdentityProfile] = []
        for row in profile_rows:
            user_id = str(row["user_id"])
            profiles.append(
                PortraitIdentityProfile(
                    user_id=user_id,
                    display_name=str(row["display_name"]) if row["display_name"] is not None else None,
                    primary_user=bool(row["primary_user"]),
                    created_at=str(row["created_at"]),
                    updated_at=str(row["updated_at"]),
                    reference_images=tuple(refs_by_user.get(user_id, ())),
                )
            )
        return tuple(profiles)

    def _fetch_profile(self, conn: sqlite3.Connection, user_id: str) -> PortraitIdentityProfile | None:
        row = conn.execute(
            "SELECT user_id, display_name, primary_user, created_at, updated_at FROM profiles WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if row is None:
            return None

        refs = conn.execute(
            "SELECT reference_id, user_id, relative_path, image_sha256, source, added_at, embedding, embedding_dim, "
            "embedding_version, detector_confidence, image_size_bytes, quality_score, embedding_model_id "
            "FROM reference_images WHERE user_id = ? ORDER BY added_at ASC, reference_id ASC",
            (user_id,),
        ).fetchall()
        return PortraitIdentityProfile(
            user_id=str(row["user_id"]),
            display_name=str(row["display_name"]) if row["display_name"] is not None else None,
            primary_user=bool(row["primary_user"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            reference_images=tuple(self._reference_from_row(reference_row) for reference_row in refs),
        )

    def _reference_from_row(self, row: sqlite3.Row) -> PortraitReferenceImage:
        embedding = _unpack_embedding(bytes(row["embedding"]), int(row["embedding_dim"]))
        return PortraitReferenceImage(
            reference_id=str(row["reference_id"]),
            relative_path=str(row["relative_path"]),
            image_sha256=str(row["image_sha256"]),
            source=_normalize_source(row["source"]),
            added_at=str(row["added_at"]),
            embedding=embedding,
            detector_confidence=_coerce_optional_float(row["detector_confidence"]),
            embedding_version=int(row["embedding_version"]),
            image_size_bytes=max(0, int(row["image_size_bytes"])),
            quality_score=_coerce_optional_float(row["quality_score"]),
            embedding_model_id=_normalize_embedding_model_id(row["embedding_model_id"]),
        )

    def _write_staged_image(self, user_dir: Path, image_bytes: bytes) -> Path:
        fd, tmp_path_raw = tempfile.mkstemp(
            prefix=".portrait_ref.",
            suffix=".tmp",
            dir=user_dir,
        )
        tmp_path = Path(tmp_path_raw)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(image_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            _ensure_file_mode(tmp_path)
            _fsync_directory(user_dir)
            return tmp_path
        except Exception:
            try:
                tmp_path.unlink()
            except OSError:
                pass
            raise

    def _absolute_reference_path(self, relative_path: Path) -> Path:
        candidate = (self.image_dir / relative_path).resolve(strict=False)
        root = self.image_dir.resolve(strict=False)
        if not _is_relative_to(candidate, root):
            raise ValueError("portrait_reference_path_escaped_image_dir")
        return candidate

    def _cleanup_removed_files(self, relative_paths: Sequence[str]) -> None:
        for relative_path in relative_paths:
            self._unlink_relative_path(relative_path)

    def _unlink_relative_path(self, relative_path: str) -> None:
        try:
            absolute_path = self._absolute_reference_path(Path(relative_path))
        except ValueError:
            return

        try:
            absolute_path.unlink()
            _fsync_directory(absolute_path.parent)
        except FileNotFoundError:
            pass
        except OSError:
            return

        try:
            absolute_path.parent.rmdir()
            _fsync_directory(absolute_path.parent.parent)
        except OSError:
            return

    def _remove_user_directory_if_empty(self, user_id: str) -> None:
        directory = (self.image_dir / self._user_directory_name(user_id)).resolve(strict=False)
        root = self.image_dir.resolve(strict=False)
        if not _is_relative_to(directory, root):
            return
        try:
            directory.rmdir()
            _fsync_directory(directory.parent)
        except OSError:
            return

    def _reconcile_files_locked(self) -> None:
        if not self.image_dir.exists():
            return

        tracked: set[str] = set()
        if self.path.exists():
            with self._connect() as conn:
                rows = conn.execute("SELECT relative_path FROM reference_images").fetchall()
                tracked = {str(row["relative_path"]) for row in rows}

        root = self.image_dir.resolve(strict=False)
        for path in sorted(root.rglob("*"), key=lambda entry: (entry.is_file(), str(entry)), reverse=True):
            try:
                resolved = path.resolve(strict=False)
            except OSError:
                continue
            if not _is_relative_to(resolved, root):
                continue

            if path.is_file():
                relative = resolved.relative_to(root).as_posix()
                if path.name.startswith(".portrait_ref.") and path.suffix == ".tmp":
                    try:
                        path.unlink()
                    except OSError:
                        pass
                    continue
                if relative not in tracked:
                    try:
                        path.unlink()
                    except OSError:
                        pass
                    continue

            if path.is_dir() and path != root:
                try:
                    path.rmdir()
                except OSError:
                    pass

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        _ensure_directory(self.path.parent)
        lock_path = self.path.parent / f".{self.path.name}.lock"
        with self._process_lock:
            with open(lock_path, "a+", encoding="utf-8") as handle:
                _ensure_file_mode(lock_path)
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


__all__ = [
    "PortraitIdentityProfile",
    "PortraitIdentityStore",
    "PortraitIdentityStoreError",
    "PortraitIdentitySummary",
    "PortraitReferenceImage",
    "portrait_identity_image_dir",
    "portrait_identity_store_path",
]