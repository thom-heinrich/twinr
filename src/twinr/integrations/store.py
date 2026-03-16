"""Persist Twinr integration configuration on disk.

This module owns the bounded, atomic, file-backed store used by runtime and web
surfaces to read and update managed integration configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
# AUDIT-FIX(#1): Add filesystem and locking primitives for hardened file access on Linux/RPi.
import errno
import json
import logging
import os
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import ClassVar

try:
    import fcntl
except ImportError:  # pragma: no cover - unavailable on non-POSIX platforms
    fcntl = None  # type: ignore[assignment]


_STORE_VERSION = 1
# AUDIT-FIX(#1): Force owner-only store permissions because integration settings may contain secrets.
_SECURE_FILE_MODE = 0o600
# AUDIT-FIX(#8): Keep the JSON store bounded so a broken file cannot grow without limit on the device.
_MAX_STORE_BYTES = 1024 * 1024


logger = logging.getLogger(__name__)


def integration_store_path(project_root: str | Path) -> Path:
    """Return the canonical JSON store path below one project root."""

    return Path(project_root).resolve() / "artifacts" / "stores" / "integrations" / "integrations.json"


# AUDIT-FIX(#4): Use a dedicated exception type so callers can fail predictably and surface user-safe recovery messaging.
class IntegrationStoreError(RuntimeError):
    """Raised when the integration store cannot be safely read or written."""


def _default_payload() -> dict[str, object]:
    """Return a fresh empty payload for a new integration store."""

    # AUDIT-FIX(#5): Always return a fresh default payload to avoid accidental shared mutable state.
    return {"version": _STORE_VERSION, "integrations": {}}


def _coerce_enabled(value: object) -> bool:
    """Parse persisted enabled flags into a safe boolean."""

    # AUDIT-FIX(#3): Parse booleans strictly so strings like "false" do not accidentally enable an integration.
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return False


def _normalize_updated_at(value: object) -> str | None:
    """Normalize persisted timestamps to UTC ISO-8601 text."""

    # AUDIT-FIX(#6): Keep only timezone-aware ISO-8601 timestamps and normalize them to UTC.
    if value is None:
        return None
    candidate = str(value).strip()
    if not candidate:
        return None
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC).isoformat()


def _normalize_integration_id(value: object) -> str:
    """Validate one persisted integration ID."""

    # AUDIT-FIX(#7): Reject empty and control-character IDs so the store cannot accumulate unaddressable entries.
    candidate = str(value).strip()
    if not candidate:
        raise ValueError("integration_id must not be empty")
    if any(ord(character) < 32 for character in candidate):
        raise ValueError("integration_id must not contain control characters")
    return candidate


@dataclass(frozen=True, slots=True)
class ManagedIntegrationConfig:
    """Store one integration's enabled flag, settings, and update timestamp."""

    integration_id: str
    enabled: bool = False
    settings: dict[str, str] = field(default_factory=dict)
    updated_at: str | None = None

    @classmethod
    def from_dict(cls, integration_id: str, payload: dict[str, object]) -> "ManagedIntegrationConfig":
        """Build a normalized config record from persisted JSON data."""

        normalized_integration_id = _normalize_integration_id(integration_id)
        settings = payload.get("settings", {})
        if not isinstance(settings, dict):
            settings = {}
        return cls(
            integration_id=normalized_integration_id,
            enabled=_coerce_enabled(payload.get("enabled", False)),
            settings={str(key): str(value) for key, value in settings.items()},
            updated_at=_normalize_updated_at(payload.get("updated_at")),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the config record into store-ready JSON data."""

        return {
            "enabled": self.enabled,
            "settings": dict(sorted(self.settings.items())),
            "updated_at": self.updated_at,
        }

    def value(self, key: str, default: str = "") -> str:
        """Return one string setting with a default fallback."""

        return self.settings.get(key, default)


class TwinrIntegrationStore:
    """Read and write the file-backed managed integration store."""

    _thread_locks: ClassVar[dict[Path, threading.RLock]] = {}
    _thread_locks_guard: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, path: str | Path) -> None:
        """Initialize the store for one explicit JSON path."""

        normalized_path = Path(path).expanduser()
        # AUDIT-FIX(#1): Convert relative paths to absolute ones without resolving symlinks away.
        if not normalized_path.is_absolute():
            normalized_path = Path.cwd() / normalized_path
        self.path = Path(os.path.abspath(normalized_path))
        self._backup_path = self.path.with_name(f"{self.path.name}.bak")
        self._lock_path = self.path.with_name(f"{self.path.name}.lock")

    @classmethod
    def from_project_root(cls, project_root: str | Path) -> "TwinrIntegrationStore":
        """Build the store that belongs to one Twinr project root."""

        return cls(integration_store_path(project_root))

    def load_all(self) -> dict[str, ManagedIntegrationConfig]:
        """Load all normalized integration records from disk."""

        # AUDIT-FIX(#2): Hold the store lock across the whole read to prevent concurrent readers from observing partial recovery/write state.
        with self._exclusive_lock(create_parent=False):
            payload = self._read_payload()
        integrations = payload.get("integrations", {})
        if not isinstance(integrations, dict):
            return {}

        loaded: dict[str, ManagedIntegrationConfig] = {}
        for integration_id, record in integrations.items():
            if not isinstance(record, dict):
                continue
            try:
                managed = ManagedIntegrationConfig.from_dict(str(integration_id), record)
            except ValueError:
                logger.warning("Skipping invalid integration_id in store %s: %r", self.path, integration_id)
                continue
            loaded[managed.integration_id] = managed
        return loaded

    def get(self, integration_id: str) -> ManagedIntegrationConfig:
        """Return one config record or an empty default record."""

        # AUDIT-FIX(#7): Normalize lookup IDs so callers cannot accidentally read/write different logical keys.
        normalized_integration_id = _normalize_integration_id(integration_id)
        return self.load_all().get(
            normalized_integration_id,
            ManagedIntegrationConfig(integration_id=normalized_integration_id),
        )

    def save(self, record: ManagedIntegrationConfig) -> ManagedIntegrationConfig:
        """Persist one config record and return the normalized saved copy."""

        raw_settings = record.settings if isinstance(record.settings, dict) else {}
        normalized_record = ManagedIntegrationConfig(
            # AUDIT-FIX(#7): Normalize IDs on write to keep the persisted key-space valid.
            integration_id=_normalize_integration_id(record.integration_id),
            enabled=_coerce_enabled(record.enabled),
            settings={str(key): str(value) for key, value in raw_settings.items()},
            # AUDIT-FIX(#6): Sanitize carried timestamps even though save() will stamp a fresh UTC value.
            updated_at=_normalize_updated_at(record.updated_at),
        )

        # AUDIT-FIX(#2): Serialize the complete read-modify-write cycle so concurrent saves cannot clobber each other.
        with self._exclusive_lock(create_parent=True):
            payload = self._read_payload()
            integrations = payload.get("integrations", {})
            if not isinstance(integrations, dict):
                integrations = {}
                payload["integrations"] = integrations

            saved_record = ManagedIntegrationConfig(
                integration_id=normalized_record.integration_id,
                enabled=normalized_record.enabled,
                settings=dict(sorted(normalized_record.settings.items())),
                updated_at=datetime.now(UTC).isoformat(),
            )
            integrations[saved_record.integration_id] = saved_record.to_dict()
            self._write_payload(payload)
        return saved_record

    def _read_payload(self) -> dict[str, object]:
        """Load the primary payload, recovering from backup when needed."""

        # AUDIT-FIX(#4): Replace brittle exists()/read_text() logic with deterministic file-state handling and explicit errors.
        primary_state, primary_payload = self._load_payload_file(self.path)
        if primary_state == "ok" and primary_payload is not None:
            return primary_payload

        # AUDIT-FIX(#5): Recover from a corrupt primary file by falling back to the last known-good backup instead of erasing config.
        backup_state, backup_payload = self._load_payload_file(self._backup_path)
        if backup_state == "ok" and backup_payload is not None:
            logger.warning("Recovered integration store %s from backup %s", self.path, self._backup_path)
            return backup_payload

        if primary_state == "missing":
            return _default_payload()
        if primary_state == "invalid" and backup_state == "missing":
            raise IntegrationStoreError(f"Integration store is corrupt and has no usable backup: {self.path}")
        if primary_state == "invalid" and backup_state == "invalid":
            raise IntegrationStoreError(
                f"Integration store and backup are corrupt: {self.path} / {self._backup_path}"
            )
        if primary_state == "missing" and backup_state == "invalid":
            raise IntegrationStoreError(f"Integration store backup is corrupt: {self._backup_path}")
        raise IntegrationStoreError(f"Integration store cannot be loaded safely: {self.path}")

    def _write_payload(self, payload: dict[str, object]) -> None:
        """Persist the payload and refresh the last-known-good backup."""

        # AUDIT-FIX(#2): Persist via atomic replace and fsync so power loss cannot leave a truncated JSON file behind.
        serialized_payload = self._serialize_payload(payload)
        self._write_bytes_atomic(self.path, serialized_payload)

        # AUDIT-FIX(#5): Keep a last-known-good backup for crash/corruption recovery.
        try:
            self._write_bytes_atomic(self._backup_path, serialized_payload)
        except IntegrationStoreError as exc:
            # AUDIT-FIX(#5): The primary store is already durable; keep save() semantics stable if only the backup refresh fails.
            logger.warning("Failed to refresh integration store backup %s: %s", self._backup_path, exc)

    def _serialize_payload(self, payload: dict[str, object]) -> bytes:
        """Encode one payload into bounded UTF-8 JSON bytes."""

        try:
            serialized = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
        except (TypeError, ValueError) as exc:
            raise IntegrationStoreError("Integration store payload is not JSON-serializable") from exc
        # AUDIT-FIX(#8): Enforce the same bounded size on writes so the store stays cheap to load on an RPi.
        if len(serialized) > _MAX_STORE_BYTES:
            raise IntegrationStoreError(
                f"Integration store payload exceeds the {_MAX_STORE_BYTES}-byte safety limit"
            )
        return serialized

    def _load_payload_file(self, path: Path) -> tuple[str, dict[str, object] | None]:
        """Read one store file and classify it as ok, missing, or invalid."""

        # AUDIT-FIX(#1): Reject symlinked files so attackers cannot redirect reads/writes to arbitrary targets.
        try:
            stat_result = path.lstat()
        except FileNotFoundError:
            return "missing", None
        except OSError as exc:
            raise IntegrationStoreError(f"Failed to inspect integration store file: {path}") from exc

        if path.is_symlink():
            raise IntegrationStoreError(f"Refusing to use symlinked integration store file: {path}")
        if not path.is_file():
            return "invalid", None

        # AUDIT-FIX(#8): Cap file size because this store should stay small on an RPi and oversized files can stall the process.
        if stat_result.st_size > _MAX_STORE_BYTES:
            raise IntegrationStoreError(
                f"Integration store file exceeds the {_MAX_STORE_BYTES}-byte safety limit: {path}"
            )

        try:
            with path.open("r", encoding="utf-8") as file_handle:
                payload = json.load(file_handle)
        except FileNotFoundError:
            return "missing", None
        except (json.JSONDecodeError, UnicodeDecodeError):
            return "invalid", None
        except OSError as exc:
            raise IntegrationStoreError(f"Failed to read integration store file: {path}") from exc

        if not isinstance(payload, dict):
            return "invalid", None

        payload.setdefault("version", _STORE_VERSION)
        integrations = payload.setdefault("integrations", {})
        if not isinstance(integrations, dict):
            return "invalid", None
        return "ok", payload

    @contextmanager
    def _exclusive_lock(self, create_parent: bool) -> Iterator[None]:
        """Hold thread and process locks around one critical section."""

        thread_lock = self._get_thread_lock()
        with thread_lock:
            if create_parent:
                self._ensure_store_parent()
            else:
                self._ensure_no_symlink_ancestors(self.path.parent)

            if not self.path.parent.exists():
                yield
                return

            # AUDIT-FIX(#2): Use an OS-level lock file so concurrent threads/processes cannot interleave read-modify-write cycles.
            lock_file = self._open_lock_file()
            try:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                yield
            finally:
                if fcntl is not None:
                    try:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                    except OSError:
                        pass
                lock_file.close()

    def _get_thread_lock(self) -> threading.RLock:
        """Return the process-local lock associated with this store path."""

        with self._thread_locks_guard:
            lock = self._thread_locks.get(self.path)
            if lock is None:
                lock = threading.RLock()
                self._thread_locks[self.path] = lock
            return lock

    def _ensure_store_parent(self) -> None:
        """Create and validate the store parent directory."""

        # AUDIT-FIX(#1): Refuse symlinked path components before creating any state files.
        self._ensure_no_symlink_ancestors(self.path.parent)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise IntegrationStoreError(f"Failed to create integration store directory: {self.path.parent}") from exc
        self._ensure_no_symlink_ancestors(self.path.parent)
        if not self.path.parent.is_dir():
            raise IntegrationStoreError(f"Integration store parent is not a directory: {self.path.parent}")

    def _ensure_no_symlink_ancestors(self, path: Path) -> None:
        """Reject symlinked path components below the requested directory."""

        current = Path(path.anchor) if path.anchor else Path()
        parts = path.parts[1:] if path.anchor else path.parts

        for part in parts:
            current = current / part
            try:
                if current.is_symlink():
                    raise IntegrationStoreError(f"Refusing symlinked integration store path component: {current}")
            except OSError as exc:
                raise IntegrationStoreError(f"Failed to inspect integration store path component: {current}") from exc

    def _open_lock_file(self):
        """Open the OS-level lock file used for cross-process coordination."""

        try:
            if self._lock_path.is_symlink():
                raise IntegrationStoreError(f"Refusing symlinked integration store lock file: {self._lock_path}")
        except OSError as exc:
            raise IntegrationStoreError(f"Failed to inspect integration store lock file: {self._lock_path}") from exc

        open_flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            open_flags |= os.O_NOFOLLOW

        try:
            descriptor = os.open(self._lock_path, open_flags, _SECURE_FILE_MODE)
            os.chmod(self._lock_path, _SECURE_FILE_MODE)
        except OSError as exc:
            raise IntegrationStoreError(f"Failed to open integration store lock file: {self._lock_path}") from exc
        return os.fdopen(descriptor, "a+b", buffering=0)

    def _write_bytes_atomic(self, path: Path, payload_bytes: bytes) -> None:
        """Write bytes to a path via tempfile plus atomic replace."""

        temp_path: str | None = None
        replaced = False

        try:
            descriptor, temp_path = tempfile.mkstemp(
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=str(path.parent),
                text=False,
            )
            try:
                os.fchmod(descriptor, _SECURE_FILE_MODE)
            except AttributeError:
                pass

            with os.fdopen(descriptor, "wb") as file_handle:
                file_handle.write(payload_bytes)
                file_handle.flush()
                os.fsync(file_handle.fileno())

            # AUDIT-FIX(#1): Replace the destination entry atomically instead of following an existing symlink with write_text().
            os.replace(temp_path, path)
            replaced = True
            os.chmod(path, _SECURE_FILE_MODE)
            self._fsync_directory(path.parent)
        except OSError as exc:
            if temp_path is not None and not replaced:
                try:
                    os.unlink(temp_path)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass
            raise IntegrationStoreError(f"Failed to write integration store file atomically: {path}") from exc

    def _fsync_directory(self, directory: Path) -> None:
        """Flush directory metadata after atomic file replacement."""

        directory_flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            directory_flags |= os.O_DIRECTORY

        try:
            descriptor = os.open(directory, directory_flags)
        except OSError as exc:
            raise IntegrationStoreError(f"Failed to open integration store directory for fsync: {directory}") from exc

        try:
            os.fsync(descriptor)
        except OSError as exc:
            if exc.errno not in {errno.EBADF, errno.EINVAL, errno.ENOTSUP, errno.EROFS}:
                raise IntegrationStoreError(f"Failed to fsync integration store directory: {directory}") from exc
        finally:
            os.close(descriptor)
