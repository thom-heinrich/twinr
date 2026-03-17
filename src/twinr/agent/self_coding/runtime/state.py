"""Persist materialized files and runtime state for compiled self-coding skills."""

from __future__ import annotations

from hashlib import sha256
import json
import logging
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import threading
from typing import Any

from twinr.agent.self_coding.runtime.contracts import SkillPackage


logger = logging.getLogger(__name__)

_MANIFEST_FILENAME = "_package_manifest.json"
_HEX_DIGITS = frozenset("0123456789abcdef")
_STATE_CORRUPT_SUFFIX = ".corrupt"

_DIR_OPEN_FLAGS = os.O_RDONLY
if hasattr(os, "O_DIRECTORY"):
    _DIR_OPEN_FLAGS |= os.O_DIRECTORY
if hasattr(os, "O_NOFOLLOW"):
    _DIR_OPEN_FLAGS |= os.O_NOFOLLOW

_FILE_READ_FLAGS = os.O_RDONLY
if hasattr(os, "O_NOFOLLOW"):
    _FILE_READ_FLAGS |= os.O_NOFOLLOW


def _validate_version(version: int) -> int:
    # AUDIT-FIX(#2): Reject silent int coercions such as bools, negatives, and arbitrary __int__ objects.
    if isinstance(version, bool) or not isinstance(version, int):
        raise TypeError("version must be an integer")
    if version < 0:
        raise ValueError("version must be >= 0")
    return version


def _normalize_relative_path(raw_path: str, *, label: str, allow_nested: bool) -> str:
    # AUDIT-FIX(#2): Normalize filesystem identifiers before any join so absolute paths and dot-segments cannot escape the store.
    if not isinstance(raw_path, str):
        raise TypeError(f"{label} must be a string")
    candidate = raw_path
    if not candidate or candidate.strip() == "":
        raise ValueError(f"{label} must not be empty")
    if "\x00" in candidate:
        raise ValueError(f"{label} must not contain NUL bytes")
    if "\\" in candidate:
        raise ValueError(f"{label} must use POSIX-style separators")
    pure = PurePosixPath(candidate)
    if pure.is_absolute():
        raise ValueError(f"{label} must be a relative path")
    parts = pure.parts
    if not parts:
        raise ValueError(f"{label} must not be empty")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"{label} must not contain '.', '..', or empty path segments")
    if not allow_nested and len(parts) != 1:
        raise ValueError(f"{label} must be a single path segment")
    return str(PurePosixPath(*parts))


def _normalize_skill_id(skill_id: str) -> str:
    # AUDIT-FIX(#9): Use one normalization path for every method so materialize/delete/state paths stay symmetric.
    return _normalize_relative_path(skill_id.strip(), label="skill_id", allow_nested=False)


def _relative_parts(relative_path: str) -> tuple[str, ...]:
    return PurePosixPath(relative_path).parts


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _temporary_name(stem: str, *, suffix: str) -> str:
    safe_stem = stem.replace(os.sep, "_")
    if os.altsep:
        safe_stem = safe_stem.replace(os.altsep, "_")
    safe_stem = safe_stem[:64] or "item"
    # AUDIT-FIX(#1): Use an internal random temp name so writes stay within the anchored directory without path races.
    return f".{safe_stem}.{tempfile.gettempprefix()}.{os.getpid()}.{threading.get_ident()}.{os.urandom(8).hex()}{suffix}"


def _ensure_root_dir(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not root.is_dir():
        raise NotADirectoryError(f"runtime store root is not a directory: {root}")


def _open_directory_fd(path: Path) -> int:
    return os.open(path, _DIR_OPEN_FLAGS)


def _fsync_fd(fd: int) -> None:
    # AUDIT-FIX(#4): Flush directory metadata as well, otherwise abrupt power loss can drop a successful rename.
    os.fsync(fd)


def _open_relative_directory_fd(root: Path, parts: tuple[str, ...], *, create: bool) -> int:
    # AUDIT-FIX(#1): Traverse the store via anchored dir_fds and refuse symlink components to reduce symlink/TOCTOU path redirection risk.
    if create:
        _ensure_root_dir(root)
    current_fd = _open_directory_fd(root)
    try:
        for part in parts:
            if create:
                created = False
                try:
                    os.mkdir(part, 0o700, dir_fd=current_fd)
                    created = True
                except FileExistsError:
                    created = False
                if created:
                    _fsync_fd(current_fd)
            entry_stat = os.stat(part, dir_fd=current_fd, follow_symlinks=False)
            if stat.S_ISLNK(entry_stat.st_mode):
                raise ValueError(f"runtime store path component must not be a symlink: {part}")
            if not stat.S_ISDIR(entry_stat.st_mode):
                raise NotADirectoryError(f"runtime store path component is not a directory: {part}")
            next_fd = os.open(part, _DIR_OPEN_FLAGS, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except Exception:
        os.close(current_fd)
        raise


def _stat_relative_path(root: Path, parts: tuple[str, ...]) -> os.stat_result | None:
    # AUDIT-FIX(#10): Inspect entries with follow_symlinks=False so existence/type checks cannot be tricked by symlink targets.
    if not parts:
        return root.stat()
    try:
        parent_fd = _open_relative_directory_fd(root, parts[:-1], create=False)
    except FileNotFoundError:
        return None
    try:
        return os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    finally:
        os.close(parent_fd)


def _path_type_from_stat(result: os.stat_result | None) -> str | None:
    if result is None:
        return None
    mode = result.st_mode
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISDIR(mode):
        return "dir"
    if stat.S_ISLNK(mode):
        return "symlink"
    return "other"


def _write_bytes_atomic(root: Path, parts: tuple[str, ...], payload: bytes) -> None:
    if not parts:
        raise ValueError("target path must not be empty")
    parent_parts = parts[:-1]
    file_name = parts[-1]
    if not file_name:
        raise ValueError("target file name must not be empty")
    parent_fd = _open_relative_directory_fd(root, parent_parts, create=True)
    temp_name = _temporary_name(file_name, suffix=".tmp")
    try:
        fd = os.open(temp_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=parent_fd)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, file_name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            _fsync_fd(parent_fd)
        finally:
            try:
                os.unlink(temp_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
    finally:
        os.close(parent_fd)


def _read_bytes(root: Path, parts: tuple[str, ...]) -> bytes:
    if not parts:
        raise ValueError("source path must not be empty")
    parent_fd = _open_relative_directory_fd(root, parts[:-1], create=False)
    try:
        entry_stat = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
        if stat.S_ISLNK(entry_stat.st_mode):
            raise ValueError(f"runtime store path must not be a symlink: {'/'.join(parts)}")
        if not stat.S_ISREG(entry_stat.st_mode):
            raise ValueError(f"runtime store path is not a regular file: {'/'.join(parts)}")
        fd = os.open(parts[-1], _FILE_READ_FLAGS, dir_fd=parent_fd)
        try:
            with os.fdopen(fd, "rb") as handle:
                return handle.read()
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            raise
    finally:
        os.close(parent_fd)


def _unlink_relative_file(root: Path, parts: tuple[str, ...]) -> None:
    if not parts:
        raise ValueError("target path must not be empty")
    parent_fd = _open_relative_directory_fd(root, parts[:-1], create=False)
    try:
        os.unlink(parts[-1], dir_fd=parent_fd)
        _fsync_fd(parent_fd)
    finally:
        os.close(parent_fd)


def _rename_relative_entry(root: Path, parent_parts: tuple[str, ...], source_name: str, target_name: str) -> None:
    parent_fd = _open_relative_directory_fd(root, parent_parts, create=True)
    try:
        os.replace(source_name, target_name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        _fsync_fd(parent_fd)
    finally:
        os.close(parent_fd)


def _ensure_relative_directory(root: Path, parts: tuple[str, ...]) -> None:
    fd = _open_relative_directory_fd(root, parts, create=True)
    try:
        _fsync_fd(fd)
    finally:
        os.close(fd)


def _remove_tree_relative(root: Path, parts: tuple[str, ...]) -> bool:
    if not parts:
        raise ValueError("refusing to remove the runtime store root")
    if not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
        raise RuntimeError("shutil.rmtree() on this platform is not symlink-attack resistant")
    parent_fd = _open_relative_directory_fd(root, parts[:-1], create=False)
    try:
        shutil.rmtree(parts[-1], dir_fd=parent_fd)
        _fsync_fd(parent_fd)
        return True
    except FileNotFoundError:
        return False
    finally:
        os.close(parent_fd)


def _prune_empty_relative_parents(root: Path, parts: tuple[str, ...], *, stop_at: tuple[str, ...]) -> None:
    current_parts = parts
    while len(current_parts) > len(stop_at):
        parent_parts = current_parts[:-1]
        name = current_parts[-1]
        try:
            parent_fd = _open_relative_directory_fd(root, parent_parts, create=False)
        except FileNotFoundError:
            return
        try:
            os.rmdir(name, dir_fd=parent_fd)
            _fsync_fd(parent_fd)
        except OSError:
            return
        finally:
            os.close(parent_fd)
        current_parts = parent_parts


def _prepare_package(package: SkillPackage) -> tuple[str, list[tuple[str, bytes, str]], dict[str, Any]]:
    # AUDIT-FIX(#3): Validate package file paths and entry_module as safe relative paths before any filesystem write occurs.
    # AUDIT-FIX(#6): Snapshot package.files once so generators are not exhausted by a second iteration.
    files = list(package.files)
    entry_module = _normalize_relative_path(package.entry_module, label="entry_module", allow_nested=True)

    prepared_files: list[tuple[str, bytes, str]] = []
    seen_paths: set[str] = set()
    manifest_files: dict[str, dict[str, Any]] = {}

    for item in files:
        normalized_path = _normalize_relative_path(item.path, label="package file path", allow_nested=True)
        if normalized_path == _MANIFEST_FILENAME:
            raise ValueError(f"package file path is reserved: {_MANIFEST_FILENAME}")
        if normalized_path in seen_paths:
            raise ValueError(f"duplicate package file path: {normalized_path}")
        seen_paths.add(normalized_path)

        if not isinstance(item.content, str):
            raise TypeError(f"package file content must be text: {normalized_path}")
        try:
            content_bytes = item.content.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError(f"package file content is not valid UTF-8 text: {normalized_path}") from exc

        if not isinstance(item.sha256, str):
            raise TypeError(f"package file sha256 must be a string: {normalized_path}")
        declared_sha256 = item.sha256.strip().lower()
        # AUDIT-FIX(#7): Validate declared package integrity instead of trusting the metadata blindly.
        if len(declared_sha256) != 64 or any(ch not in _HEX_DIGITS for ch in declared_sha256):
            raise ValueError(f"package file sha256 is invalid: {normalized_path}")
        actual_sha256 = sha256(content_bytes).hexdigest()
        if declared_sha256 != actual_sha256:
            raise ValueError(f"package file sha256 mismatch: {normalized_path}")

        prepared_files.append((normalized_path, content_bytes, actual_sha256))
        manifest_files[normalized_path] = {"sha256": actual_sha256, "bytes": len(content_bytes)}

    if entry_module not in seen_paths:
        raise ValueError(f"entry_module is not present in package files: {entry_module}")

    manifest_payload = {"entry_module": entry_module, "files": manifest_files}
    return entry_module, prepared_files, manifest_payload


class SelfCodingSkillRuntimeStore:
    """Own the materialized runtime filesystem for compiled skill packages."""

    def __init__(self, root: str | Path) -> None:
        try:
            resolved_root = Path(root).expanduser().resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"invalid runtime store root: {root!r}") from exc
        self.root = resolved_root
        self.runtime_dir = self.root / "runtime"
        self.materialized_dir = self.runtime_dir / "materialized"
        self.state_dir = self.runtime_dir / "state"
        # AUDIT-FIX(#5): Public methods may run concurrently from sync threadpools or async callers; serialize store mutations in-process.
        self._lock = threading.RLock()

    def materialize_package(self, *, skill_id: str, version: int, package: SkillPackage) -> Path:
        normalized_skill_id = _normalize_skill_id(skill_id)
        normalized_version = _validate_version(version)
        _, prepared_files, manifest_payload = _prepare_package(package)

        skill_parts = ("runtime", "materialized", normalized_skill_id)
        target_dir_name = f"v{normalized_version}"
        target_parts = (*skill_parts, target_dir_name)
        manifest_parts = (*target_parts, _MANIFEST_FILENAME)

        with self._lock:
            existing_target = _path_type_from_stat(_stat_relative_path(self.root, target_parts))
            if existing_target is not None:
                if existing_target != "dir":
                    raise ValueError(f"materialized path exists but is not a directory: {self.materialized_dir / normalized_skill_id / target_dir_name}")
                try:
                    existing_manifest = json.loads(_read_bytes(self.root, manifest_parts).decode("utf-8"))
                except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                    raise FileExistsError(f"materialized package exists with invalid or missing manifest: {self.materialized_dir / normalized_skill_id / target_dir_name}") from exc
                if existing_manifest == manifest_payload:
                    return self.materialized_dir / normalized_skill_id / target_dir_name
                raise FileExistsError(f"materialized package version already exists with different contents: {self.materialized_dir / normalized_skill_id / target_dir_name}")

            temp_dir_name = _temporary_name(target_dir_name, suffix=".dir")
            temp_parts = (*skill_parts, temp_dir_name)
            _ensure_relative_directory(self.root, temp_parts)
            try:
                for relative_path, content_bytes, _sha256 in prepared_files:
                    _write_bytes_atomic(self.root, (*temp_parts, *_relative_parts(relative_path)), content_bytes)
                _write_bytes_atomic(self.root, (*temp_parts, _MANIFEST_FILENAME), _json_bytes(manifest_payload))
                _rename_relative_entry(self.root, skill_parts, temp_dir_name, target_dir_name)
            except Exception:
                try:
                    _remove_tree_relative(self.root, temp_parts)
                except FileNotFoundError:
                    pass
                except OSError:
                    logger.exception("failed to clean up temporary materialized package directory")
                raise

        return self.materialized_dir / normalized_skill_id / target_dir_name

    def load_state(self, *, skill_id: str, version: int) -> dict[str, Any]:
        normalized_skill_id = _normalize_skill_id(skill_id)
        normalized_version = _validate_version(version)
        parts = ("runtime", "state", normalized_skill_id, f"v{normalized_version}.json")
        with self._lock:
            try:
                raw_payload = _read_bytes(self.root, parts)
            except FileNotFoundError:
                return {}
            except ValueError:
                logger.warning("state path is not a regular file; returning empty state", extra={"skill_id": normalized_skill_id, "version": normalized_version})
                return {}

            try:
                payload = json.loads(raw_payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                # AUDIT-FIX(#8): Quarantine corrupt state snapshots instead of crashing the runtime on startup/reload.
                corrupt_name = f"v{normalized_version}.json{_STATE_CORRUPT_SUFFIX}.{os.urandom(6).hex()}"
                try:
                    _rename_relative_entry(self.root, ("runtime", "state", normalized_skill_id), f"v{normalized_version}.json", corrupt_name)
                except FileNotFoundError:
                    pass
                except OSError:
                    logger.exception("failed to quarantine corrupt state file", extra={"skill_id": normalized_skill_id, "version": normalized_version})
                logger.warning("corrupt state file ignored", extra={"skill_id": normalized_skill_id, "version": normalized_version, "error": str(exc)})
                return {}

            if not isinstance(payload, dict):
                corrupt_name = f"v{normalized_version}.json{_STATE_CORRUPT_SUFFIX}.{os.urandom(6).hex()}"
                try:
                    _rename_relative_entry(self.root, ("runtime", "state", normalized_skill_id), f"v{normalized_version}.json", corrupt_name)
                except FileNotFoundError:
                    pass
                except OSError:
                    logger.exception("failed to quarantine non-object state file", extra={"skill_id": normalized_skill_id, "version": normalized_version})
                logger.warning("non-object state file ignored", extra={"skill_id": normalized_skill_id, "version": normalized_version})
                return {}

            return payload

    def save_state(self, *, skill_id: str, version: int, payload: dict[str, Any]) -> None:
        normalized_skill_id = _normalize_skill_id(skill_id)
        normalized_version = _validate_version(version)
        if not isinstance(payload, dict):
            raise TypeError("payload must be a mapping")
        try:
            # AUDIT-FIX(#11): Fail with a clear payload contract error instead of bubbling a generic JSON exception.
            normalized = json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise TypeError("payload must be JSON-serializable and must not contain NaN or Infinity") from exc
        if not isinstance(normalized, dict):
            raise TypeError("payload must serialize to a JSON object")
        with self._lock:
            _write_bytes_atomic(self.root, ("runtime", "state", normalized_skill_id, f"v{normalized_version}.json"), _json_bytes(normalized))

    def delete_materialized_package(self, *, skill_id: str, version: int) -> bool:
        """Remove one materialized skill-package directory from the runtime store."""

        normalized_skill_id = _normalize_skill_id(skill_id)
        normalized_version = _validate_version(version)
        skill_parts = ("runtime", "materialized", normalized_skill_id)
        target_parts = (*skill_parts, f"v{normalized_version}")

        with self._lock:
            target_type = _path_type_from_stat(_stat_relative_path(self.root, target_parts))
            if target_type is None:
                return False
            if target_type != "dir":
                raise ValueError(f"materialized path is not a directory: {self.materialized_dir / normalized_skill_id / f'v{normalized_version}'}")
            deleted = _remove_tree_relative(self.root, target_parts)
            if not deleted:
                return False
            _prune_empty_relative_parents(self.root, skill_parts, stop_at=("runtime", "materialized"))
            return True

    def delete_state(self, *, skill_id: str, version: int) -> bool:
        """Remove one persisted skill-state snapshot from the runtime store."""

        normalized_skill_id = _normalize_skill_id(skill_id)
        normalized_version = _validate_version(version)
        parts = ("runtime", "state", normalized_skill_id, f"v{normalized_version}.json")

        with self._lock:
            target_type = _path_type_from_stat(_stat_relative_path(self.root, parts))
            if target_type is None:
                return False
            if target_type != "file":
                raise ValueError(f"state path is not a regular file: {self.state_dir / normalized_skill_id / f'v{normalized_version}.json'}")
            try:
                _unlink_relative_file(self.root, parts)
            except FileNotFoundError:
                return False
            _prune_empty_relative_parents(self.root, ("runtime", "state", normalized_skill_id), stop_at=("runtime", "state"))
            return True

    def _state_path(self, *, skill_id: str, version: int) -> Path:
        normalized_skill_id = _normalize_skill_id(skill_id)
        normalized_version = _validate_version(version)
        return self.state_dir / normalized_skill_id / f"v{normalized_version}.json"

    @staticmethod
    def _prune_empty_parents(path: Path, *, stop_at: Path) -> None:
        current = Path(path)
        stop = Path(stop_at)
        while current != stop and current.exists():
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent


def materialized_module_hash(materialized_root: Path, *, entry_module: str) -> str:
    """Return a stable content hash for one materialized entry module."""

    normalized_entry_module = _normalize_relative_path(entry_module, label="entry_module", allow_nested=True)
    root = Path(materialized_root).expanduser().resolve(strict=False)
    # AUDIT-FIX(#3): Secure the hash input path so callers cannot read arbitrary files outside the materialized package root.
    return sha256(_read_bytes(root, _relative_parts(normalized_entry_module))).hexdigest()


__all__ = [
    "SelfCodingSkillRuntimeStore",
    "materialized_module_hash",
]